//! Generic L1 kernel engine: per-kernel files implement `KernelSpec` once and
//! `Kernel<S>` provides build/eval + the `Probe` blanket impl.
//!
//! Engine knows nothing about specific kernel kinds: it only sees the sweep
//! grid, the cache kind, the bridge args, and the sweep-coord projection of
//! the runtime input. Comm / compute / distribution-sensitive kernels are all
//! the same shape here — variations live in each per-kernel `KernelSpec`.

use std::collections::BTreeMap;
use std::marker::PhantomData;

use crate::timing::bridge::{ArgsPayload, KernelKind, KernelMetrics, PerfApiBridge};
use crate::timing::cache::interp::LeafMetrics;
use crate::timing::cache::{BackendCache, CacheKind, OutlierWarning, PeakRates};
use crate::timing::result::CacheProbe;
use crate::timing::sweep::{SweepCoords, SweepGrid};
use crate::timing::{BuildError, DType, Probe};

/// Per-kernel `*KernelConfig` contract: identity (`Hash + Eq`) + the required
/// `backends: Vec<&'static str>` field exposed via `backends()`. The proc-macro
/// `#[derive(KernelConfig)]` in `timing-kernel-derive` generates this impl by
/// reading `&self.backends` directly; structs without a `backends` field fail
/// to derive at the generated access site.
pub trait KernelConfig:
    std::hash::Hash + Eq + Clone + std::fmt::Debug + serde::Serialize + 'static
{
    fn backends(&self) -> &[&'static str];
    /// Replace the candidate backend set. The `#[derive(KernelConfig)]` macro
    /// generates `self.backends = backends`. Called at `Kernel::build` when a
    /// per-role user override is active on the bridge, so the override lands in
    /// the config's identity (`Hash`/`describe_config`) before caches are fit.
    fn set_backends(&mut self, backends: Vec<&'static str>);
    /// Used in build-error messages, e.g. `"SingleGemmKernelConfig.backends"`.
    /// Auto-derived as `"{StructName}.backends"`.
    const BACKENDS_FIELD: &'static str;

    /// The GPU whose profiled rows this config caches. Part of the config
    /// identity (`Hash + Eq`), so distinct GPUs are distinct kernels / caches;
    /// passed to the bridge at `build` (`get_times` / `count_missing`) as the DB
    /// `gpu_name` key.
    fn gpu_name(&self) -> &str;

    /// The compute/activation dtype (a GEMM's `dtype`, attention's `q_dtype`), or
    /// `None` for a dtype-agnostic kernel (size-keyed comm, byte-keyed
    /// elementwise). `#[derive(KernelConfig)]` overrides the default from the
    /// field tagged `#[compute_dtype]`. Emitted as a typed field in the enumerate
    /// record so the launcher's capability gate reads it directly — never
    /// re-parses `describe_config`.
    fn compute_dtype(&self) -> Option<DType> {
        None
    }

    /// The KV-cache dtype (attention only; from the field tagged `#[kv_dtype]`),
    /// or `None` elsewhere. Independent of `compute_dtype`: a bf16 query with an
    /// fp8 KV cache is a valid attention config.
    fn kv_dtype(&self) -> Option<DType> {
        None
    }

    /// The single structured kernel-config representation used by manifests,
    /// introspection and presentation. `Dim` values retain formula provenance.
    fn describe_config(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("KernelConfig must serialize to JSON")
    }

    /// The `symbol -> value` legend for this config's `Dim` shape fields — the
    /// union of each field's [`Dim::bindings`]. Lets a UI resolve a leaf's
    /// rendered formula (`n=(num_qo_heads/attn_tp+…)*head_dim`) to its parts.
    /// `#[derive(KernelConfig)]` overrides it, unioning `bindings()` over every
    /// `Dim`-typed field; the default (comm / no-shape configs) is empty.
    fn symbol_bindings(&self) -> BTreeMap<&'static str, u32> {
        BTreeMap::new()
    }
}

/// One impl per kernel kind. Declares the per-kernel types (Config / Input),
/// the `KIND` identifier, and the 3 dispatch fns (sweep_grid / cache_kind /
/// enumerate). Everything else lives in `Kernel<S>`; the `backends` invariant
/// lives on `KernelConfig`.
///
/// `KIND` is the single source of truth for this kernel's name: it doubles as
/// the error tag and as the Python facade stem. The bridge derives Python fn
/// names via `format!("get_{kind}_times")` / `format!("count_missing_{kind}")`,
/// matching `profiling/facade.py:_GENERATED_FACADES`. Adding a new kernel only
/// requires declaring this constant in `kernels/<name>.rs` — no central enum.
///
/// `enumerate` returns `Vec<ArgsPayload>` directly: the wire format is what
/// the bridge consumes, so a typed `Args` struct would be a pure intermediate.
/// Python's `KernelArgs` dataclass (and `coerce_args`) owns the schema check.
pub trait KernelSpec: 'static {
    type Config: KernelConfig;
    type Input: SweepCoords;

    const KIND: KernelKind;

    /// Profiler facade / `profile.db` table the specs are profiled against
    /// (`get_{profile_kind}_times`). Defaults to `KIND`. Override when a cache
    /// *variant* (same physical kernel, different cache axes) reuses an existing
    /// kernel's profiled rows: the variant carries its own `KIND` for the
    /// registry/identity but profiles through the base kind's facade + table.
    fn profile_kind() -> KernelKind {
        Self::KIND
    }

    fn sweep_grid(config: &Self::Config) -> SweepGrid;
    /// Decided by backend only — Config is already fixed inside any one Spec.
    /// Cache-shape differences driven by Config (e.g. fp8 vs fp16) should live
    /// in separate Spec types, not in this function's body.
    fn cache_kind(backend: &'static str) -> CacheKind;

    /// Grid cells (row-major, aligned with `enumerate`) that are physically
    /// infeasible. Their profiled sample is forced non-finite at build so
    /// `Cache2DLinear` drops them and renormalizes over the feasible corners,
    /// instead of caching a fabricated value for a shape that can't occur.
    /// Default: all feasible (empty mask). Used by re-axis variants whose
    /// rectangular cache grid covers a non-rectangular feasible region (e.g. the
    /// `(A,B)` attn variant's `A < B/2` corner).
    fn infeasible_mask(_config: &Self::Config, _grid: &SweepGrid) -> Vec<bool> {
        Vec::new()
    }
    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload>;
}

/// Generic kernel struct. Per-kernel files export
/// `pub type FooKernel = Kernel<FooSpec>;`.
pub struct Kernel<S: KernelSpec> {
    pub config: S::Config,
    pub outlier_warnings: Vec<OutlierWarning>,
    backend_caches: Vec<BackendCache>,
    _spec: PhantomData<fn() -> S>,
}

impl<S: KernelSpec> Kernel<S> {
    pub fn build(
        name: String,
        mut config: S::Config,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        // User-configurable backends: if a per-role override is active on the
        // bridge (set by the deployment layer for the pool currently building),
        // replace this kernel's const-default candidate set by its dotted role
        // `name`. Interned against the bridge, so the config still owns only
        // `&'static str`. Overriding before `ensure_has_backends` means an empty
        // override is rejected here too, and before caches fit means the override
        // is reflected in `describe_config` / the cost manifest.
        if let Some(backends) = bridge.backend_override_for(&name) {
            config.set_backends(backends);
        }
        ensure_has_backends(
            S::KIND,
            <S::Config as KernelConfig>::BACKENDS_FIELD,
            config.backends(),
        )?;
        // The `cost_log` slot_backend index is a position-local `u8` with
        // [`LeafMetrics::NO_BACKEND`] (255) reserved for "no backend", so a
        // position may carry at most 255 candidates (indices 0..=254).
        assert!(
            config.backends().len() <= LeafMetrics::NO_BACKEND as usize,
            "kernel {name} ({}) declares {} candidate backends; the cost_log \
             slot_backend index supports at most {}",
            S::KIND,
            config.backends().len(),
            LeafMetrics::NO_BACKEND,
        );

        // Enumerate mode (`emit-backends`): record this kernel's structural facts
        // and return an empty kernel WITHOUT any profile.db lookup. GPU-free and
        // profiling-free — the cost-tree structure is all that's built. Returns
        // before the sweep grid / dry-run / cache-fit paths below.
        if bridge.is_enumerate() {
            bridge.record_enum(
                name,
                S::KIND,
                config.gpu_name(),
                config.compute_dtype(),
                config.kv_dtype(),
                config.describe_config(),
                config.backends(),
            );
            return Ok(Self {
                config,
                outlier_warnings: Vec::new(),
                backend_caches: Vec::new(),
                _spec: PhantomData,
            });
        }

        let sweep_grid = S::sweep_grid(&config);
        let backends = config.backends();

        // Physically-infeasible grid cells (row-major). We never send these to
        // the profiler — `drop_infeasible` strips them from a spec list before
        // profiling, so the profiler is only ever asked about shapes that exist.
        let infeasible = S::infeasible_mask(&config, &sweep_grid);
        let drop_infeasible = |specs: Vec<ArgsPayload>| -> Vec<ArgsPayload> {
            if infeasible.is_empty() {
                return specs;
            }
            assert_eq!(
                infeasible.len(),
                specs.len(),
                "infeasible_mask must align row-major with the sweep grid"
            );
            specs
                .into_iter()
                .zip(&infeasible)
                .filter_map(|(spec, &drop)| (!drop).then_some(spec))
                .collect()
        };

        // Dry-run mode: don't fit caches — just tally how many specs are missing
        // from profile.db (the JIT work a real build would do) and report one line
        // for this kernel. The returned `Kernel` has empty caches; dry-run exits
        // before the tick loop so it's never looked up.
        if bridge.is_dry_run() {
            let mut missing = 0;
            let mut total = 0;
            for &backend in backends {
                let specs = drop_infeasible(S::enumerate(&config, &sweep_grid, backend));
                total += specs.len();
                missing += bridge
                    .count_missing(specs, S::profile_kind(), backend, config.gpu_name())
                    .map_err(|err| BuildError::from_perf_api(S::KIND, backend, err))?;
            }
            bridge.record_missing(name, S::KIND, missing, total);
            return Ok(Self {
                config,
                outlier_warnings: Vec::new(),
                backend_caches: Vec::new(),
                _spec: PhantomData,
            });
        }

        let mut backend_caches = Vec::with_capacity(backends.len());
        let mut outlier_warnings = Vec::new();
        for &backend in backends {
            let specs = S::enumerate(&config, &sweep_grid, backend);
            let profiled = bridge
                .get_times(drop_infeasible(specs), S::profile_kind(), config.gpu_name())
                .map_err(|err| BuildError::from_perf_api(S::KIND, backend, err))?;
            // Reassemble the row-major sample grid: a non-finite placeholder at
            // each infeasible cell (so `Cache2DLinear` drops + renormalizes),
            // profiled samples filling the feasible cells in order.
            let samples: Vec<KernelMetrics> = if infeasible.is_empty() {
                profiled
            } else {
                let mut profiled = profiled.into_iter();
                infeasible
                    .iter()
                    .map(|&drop| {
                        if drop {
                            KernelMetrics::non_finite()
                        } else {
                            profiled
                                .next()
                                .expect("feasible sample count must match the unmasked cells")
                        }
                    })
                    .collect()
            };
            let (cache, warnings) = BackendCache::fit(
                S::KIND,
                backend,
                S::cache_kind(backend),
                &sweep_grid,
                &samples,
            )?;
            backend_caches.push(cache);
            outlier_warnings.extend(warnings);
            // Per-backend progress: one line per profile.db query (the slow unit).
            tracing::info!(
                "[build]   kernel {name} ({}) backend={backend} done ({} samples)",
                S::KIND,
                samples.len()
            );
        }
        Ok(Self {
            config,
            outlier_warnings,
            backend_caches,
            _spec: PhantomData,
        })
    }

    /// All-four-metrics best-of-N for the CostTree eval path: return the
    /// [`LeafMetrics`] from the backend with the smallest non-negative wallclock,
    /// preserving that backend's coverage bits.
    pub fn eval(&self, input: &S::Input) -> LeafMetrics {
        let coords = input.coords();
        match self.backend_caches.as_slice() {
            [] => panic!("kernel config validation must create at least one backend cache"),
            [backend_cache] => {
                // Single candidate: it is trivially the selected one. Stamp its
                // position-local index (0) so the `cost_log` slot_backend column
                // carries a real choice, not the [`LeafMetrics::NO_BACKEND`]
                // sentinel a bare cache eval returns.
                let mut only = backend_cache.eval(&coords);
                only.backend_index = 0;
                only
            }
            [first_cache, rest @ ..] => {
                let mut best = first_cache.eval(&coords);
                let mut best_time_ms = best.m.time_ms.max(0.0);
                let mut best_index = 0u8;
                for (offset, backend_cache) in rest.iter().enumerate() {
                    let candidate = backend_cache.eval(&coords);
                    let candidate_time_ms = candidate.m.time_ms.max(0.0);
                    if candidate_time_ms < best_time_ms {
                        best = candidate;
                        best_time_ms = candidate_time_ms;
                        best_index = (offset + 1) as u8;
                    }
                }
                // Position-local index into this kernel's ordered candidate list
                // (== manifest `LeafDesc.backends` order, == cache fit order).
                best.backend_index = best_index;
                best
            }
        }
    }

    /// Best achievable compute / BW rates over this kernel's fitted grid, across
    /// all its backend caches (best-of-N applies to peaks too). The per-config
    /// "best batching" ceiling the optimality analyzer reads via `kernel-query
    /// peak`. Empty (all zero) in enumerate / dry-run mode, which builds no caches.
    pub fn peak_rates(&self) -> PeakRates {
        self.backend_caches
            .iter()
            .map(BackendCache::peak_rates)
            .fold(PeakRates::default(), PeakRates::merge)
    }
}

impl<S: KernelSpec> CacheProbe for Kernel<S>
where
    S::Input: serde::de::DeserializeOwned,
{
    fn kind(&self) -> &'static str {
        S::KIND
    }

    fn describe_config(&self) -> serde_json::Value {
        KernelConfig::describe_config(&self.config)
    }

    fn grid_axes(&self) -> Vec<Vec<f64>> {
        S::sweep_grid(&self.config).axes().to_vec()
    }

    fn eval_json(&self, input: &serde_json::Value) -> anyhow::Result<LeafMetrics> {
        // Deserialize straight into the kernel's own Input struct, then call the
        // kernel's existing best-of-N `eval` — the real `coords()` projection runs
        // unchanged. No reimplementation, no slice gymnastics.
        let input: S::Input = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("query point does not match {} Input: {e}", S::KIND))?;
        Ok(self.eval(&input))
    }

    fn peak_rates(&self) -> PeakRates {
        Kernel::peak_rates(self)
    }
}

impl<S: KernelSpec> Probe for Kernel<S> {
    type Input = S::Input;
    fn eval(&self, input: &Self::Input) -> LeafMetrics {
        Self::eval(self, input)
    }
    /// The KIND tag + one-line config summary the CostTree compile captures into
    /// the leaf's manifest entry (the old `Describe` leaf line). One blanket impl
    /// covers every kernel since all are `Kernel<S>`.
    fn kind(&self) -> &'static str {
        S::KIND
    }
    fn describe_config(&self) -> serde_json::Value {
        self.config.describe_config()
    }
}

// ─── kernel-query registry (cache-fidelity introspection) ───────────────────

/// One entry per kernel kind, collected at link time by `inventory` so the
/// `kernel-query` subcommand maps a runtime `kind` string to the right
/// `Kernel<S>` builder WITHOUT a central match. Each kernel registers itself via
/// [`register_kernel!`] (which also emits its `pub type FooKernel = Kernel<FooSpec>`
/// alias), so adding a kernel touches only that kernel's own file.
pub(crate) struct KernelQueryEntry {
    pub kind: &'static str,
    /// `eval` path: deserialize config, build the kernel (profiles missing grid
    /// rows), box it for interpolation. Needs the bridge.
    pub build: fn(serde_json::Value, &PerfApiBridge) -> anyhow::Result<Box<dyn CacheProbe>>,
    /// `grid` path: deserialize config and report
    /// `(describe_config, grid_axes, input_field_names)` from `sweep_grid` +
    /// the Input's `SweepCoords` alone — no bridge, no profiling, no GPU.
    pub describe: fn(
        serde_json::Value,
    )
        -> anyhow::Result<(serde_json::Value, Vec<Vec<f64>>, &'static [&'static str])>,
}

inventory::collect!(KernelQueryEntry);

impl KernelQueryEntry {
    /// Registry entry for kernel `S`: its `KIND` + monomorphized config-driven
    /// builders for both query ops. Bounds match the blanket `CacheProbe` impl
    /// (config/input must deserialize from a request).
    pub(crate) const fn of<S>() -> Self
    where
        S: KernelSpec,
        S::Config: serde::de::DeserializeOwned,
        S::Input: serde::de::DeserializeOwned,
    {
        KernelQueryEntry {
            kind: S::KIND,
            build: build_probe_from_json::<S>,
            describe: describe_from_json::<S>,
        }
    }
}

/// Deserialize `config` into `S::Config`, build that one kernel, box it as a
/// `dyn CacheProbe`. The `eval`-path fn pointer each `KernelQueryEntry` stores.
fn build_probe_from_json<S>(
    config: serde_json::Value,
    bridge: &PerfApiBridge,
) -> anyhow::Result<Box<dyn CacheProbe>>
where
    S: KernelSpec,
    S::Config: serde::de::DeserializeOwned,
    S::Input: serde::de::DeserializeOwned,
{
    let config: S::Config = serde_json::from_value(config)
        .map_err(|e| anyhow::anyhow!("config does not match {} KernelConfig: {e}", S::KIND))?;
    let kernel = Kernel::<S>::build(S::KIND.to_string(), config, bridge)?;
    Ok(Box::new(kernel))
}

/// Deserialize `config` into `S::Config` and report its one-line summary, the
/// fitted grid axes from `sweep_grid`, and the Input's field names (axis labels).
/// The `grid`-path fn pointer — pure metadata, so it needs neither the bridge nor
/// a built cache.
fn describe_from_json<S>(
    config: serde_json::Value,
) -> anyhow::Result<(serde_json::Value, Vec<Vec<f64>>, &'static [&'static str])>
where
    S: KernelSpec,
    S::Config: serde::de::DeserializeOwned,
{
    let config: S::Config = serde_json::from_value(config)
        .map_err(|e| anyhow::anyhow!("config does not match {} KernelConfig: {e}", S::KIND))?;
    let grid_axes = S::sweep_grid(&config).axes().to_vec();
    Ok((
        config.describe_config(),
        grid_axes,
        <S::Input as SweepCoords>::coord_field_names(),
    ))
}

/// Declare a kernel's public `Kernel<S>` type alias AND register it for
/// `kernel-query`, in one line — so adding a kernel never touches the dispatch.
/// Replaces the bare `pub type FooKernel = Kernel<FooSpec>;`.
macro_rules! register_kernel {
    ($alias:ident, $spec:ty) => {
        pub type $alias = $crate::timing::kernels::engine::Kernel<$spec>;
        ::inventory::submit! {
            $crate::timing::kernels::engine::KernelQueryEntry::of::<$spec>()
        }
    };
}
pub(crate) use register_kernel;

// ─── internal helpers ───────────────────────────────────────────────────────

fn ensure_has_backends(
    kernel_kind: KernelKind,
    field_name: &'static str,
    backends: &[&'static str],
) -> Result<(), BuildError> {
    if backends.is_empty() {
        return Err(BuildError::FitFailed {
            kind: kernel_kind,
            reason: format!("{field_name} must not be empty"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ensure_has_backends;
    use crate::timing::BuildError;

    #[test]
    fn ensure_has_backends_rejects_empty_backend_lists() {
        assert!(matches!(
            ensure_has_backends("single_gemm", "backends", &[]),
            Err(BuildError::FitFailed { .. })
        ));
    }
}
