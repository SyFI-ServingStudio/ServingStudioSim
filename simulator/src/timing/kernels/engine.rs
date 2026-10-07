//! Generic L1 kernel engine: per-kernel files implement `KernelSpec` once and
//! `Kernel<S>` provides build/eval + the `Probe` blanket impl.
//!
//! Engine knows nothing about specific kernel kinds: it only sees the sweep
//! grid, the cache kind, the bridge args, and the cache-coordinate projection
//! of the runtime input. Comm / compute / distribution-sensitive kernels are
//! all the same shape here — variations live in each per-kernel `KernelSpec`.

use std::collections::BTreeMap;
use std::marker::PhantomData;

use crate::timing::bridge::{ArgsPayload, ConfigGrid, KernelKind, KernelMetrics, PerfApiBridge};
use crate::timing::cache::interp::LeafMetrics;
use crate::timing::cache::{BackendCache, CacheKind, OutlierWarning, PeakRates};
use crate::timing::result::CacheProbe;
use crate::timing::sweep::{SweepCoords, SweepGrid};
use crate::timing::{BuildError, Coords, DType, Probe};

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

    /// The field tagged `#[compute_dtype]`, if any. Derived.
    const COMPUTE_DTYPE_FIELD: Option<&'static str> = None;
    /// The field tagged `#[kv_dtype]`, if any. Derived.
    const KV_DTYPE_FIELD: Option<&'static str> = None;

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

    /// The config's identity in profile.db's kernel-config registry: every
    /// field except `gpu_name` and `backends`, which profile.db rows key in
    /// their own columns, with each `Dim` reduced to its value. Two configs
    /// that fold to the same shape share one identity however their dims were
    /// derived.
    fn identity(&self) -> serde_json::Value {
        let mut identity = crate::timing::dims::values_only(|| {
            serde_json::to_value(self).expect("KernelConfig must serialize to JSON")
        });
        let fields = identity
            .as_object_mut()
            .expect("KernelConfig must serialize as a struct");
        fields.remove("gpu_name");
        fields.remove("backends");
        identity
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
/// the `KIND` identifier, the three required dispatch fns (sweep_grid /
/// cache_kind / enumerate), and an optional config-aware cache projection.
/// Everything else lives in `Kernel<S>`; the `backends` invariant lives on
/// `KernelConfig`.
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

    /// Project a physical runtime input into this spec's cache coordinates.
    /// Most kernels cache directly in [`SweepCoords::coords`] space; specs that
    /// need config-aware axes can override this hook without changing their
    /// public Input schema or the generic cache implementations.
    fn cache_coords(_config: &Self::Config, input: &Self::Input) -> Coords {
        input.coords()
    }

    /// Config-aware correction applied to the interpolated cache result in
    /// [`Kernel::eval`], before the metrics reach the CostTree. Default:
    /// identity (the cache row is the whole cost). Override ONLY where a
    /// measured/analytic cache row cannot by itself represent a cost component
    /// that the row's axes do not carry. The one current user is the
    /// Infinity-Fabric all-reduce, whose profiled row is a pure bandwidth
    /// roofline (data-movement time) and cannot encode the RCCL/PYNCCL
    /// cross-rank sync-barrier latency — a per-collective, message-size-
    /// independent cost added here (`all_reduce_fusion.rs`). Keeping it in the
    /// cost model rather than in the row avoids double-counting if the analytic
    /// bandwidth row is ever regenerated.
    fn adjust_metrics(
        _config: &Self::Config,
        _input: &Self::Input,
        metrics: LeafMetrics,
    ) -> LeafMetrics {
        metrics
    }

    /// Grid cells (row-major, aligned with `enumerate`) that are physically
    /// infeasible. Their profiled sample is forced non-finite at build so a
    /// multilinear cache drops them and renormalizes over feasible corners,
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

    /// Reject a config whose external inputs cannot be read, before anything
    /// builds on them.
    ///
    /// `enumerate` has no error channel -- it fills a grid, and a kernel with
    /// nothing to say about a cell has no way to say so -- which is right for
    /// the arithmetic every kernel does there, but not for a config that names
    /// a file. An arch builder proves such a file readable when it constructs
    /// the config; a config deserialized straight from JSON never went through
    /// one, so this runs at that boundary instead.
    ///
    /// Default: nothing to check.
    fn validate_config(_config: &Self::Config) -> anyhow::Result<()> {
        Ok(())
    }
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

        // Kernel-config registry: record the config and the grid it asks for,
        // whatever the mode. Recording enumerates the grid itself, so it does not
        // depend on which of the paths below runs.
        if bridge.records_configs() {
            bridge.record_config(
                &name,
                S::KIND,
                S::profile_kind(),
                config.gpu_name(),
                config.identity(),
                || config_grid::<S>(&config),
            )?;
        }

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
                let specs =
                    bridge.unique_dry_run_specs(S::profile_kind(), config.gpu_name(), specs);
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
        let coords = S::cache_coords(&self.config, input);
        let metrics = self.eval_cache_coords(&coords);
        S::adjust_metrics(&self.config, input, metrics)
    }

    fn eval_cache_coords(&self, coords: &Coords) -> LeafMetrics {
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
        // kernel's existing best-of-N `eval` — the real config-aware
        // `cache_coords()` projection runs unchanged. No reimplementation, no
        // slice gymnastics.
        let input: S::Input = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("query point does not match {} Input: {e}", S::KIND))?;
        Ok(self.eval(&input))
    }

    fn eval_coords(&self, coords: &[f64]) -> anyhow::Result<LeafMetrics> {
        let expected_dims = S::sweep_grid(&self.config).axes().len();
        anyhow::ensure!(
            coords.len() == expected_dims,
            "query point has {} cache coordinates for {}D {} grid",
            coords.len(),
            expected_dims,
            S::KIND,
        );
        anyhow::ensure!(
            coords.iter().all(|coordinate| coordinate.is_finite()),
            "query point for {} contains a non-finite cache coordinate",
            S::KIND,
        );
        Ok(self.eval_cache_coords(&Coords::from_slice(coords)))
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
    /// The DB kind whose rows the kernel reads (`KernelSpec::profile_kind`).
    pub profile_kind: fn() -> KernelKind,
    /// `KernelConfig` fields (serde names): what fixes one kernel instance.
    pub config_fields: fn() -> &'static [&'static str],
    /// `Input` fields (serde names): the physical query, which is what the
    /// profile.db rows sweep.
    pub input_fields: fn() -> &'static [&'static str],
    /// The cache coordinates an `Input` projects onto (`SweepCoords`).
    pub coord_fields: fn() -> &'static [&'static str],
    pub compute_dtype_field: Option<&'static str>,
    pub kv_dtype_field: Option<&'static str>,
    /// `rows` path: the profile.db rows one config measures and the rows each
    /// runtime input reads ([`rows_from_json`]). No bridge, DB or GPU.
    pub rows: fn(
        serde_json::Value,
        Option<&str>,
        &[serde_json::Value],
    ) -> anyhow::Result<serde_json::Value>,
    /// `eval` path: deserialize config, build the kernel (profiles missing grid
    /// rows), box it for interpolation. Needs the bridge.
    pub build: fn(serde_json::Value, &PerfApiBridge) -> anyhow::Result<Box<dyn CacheProbe>>,
    /// `grid` path: deserialize config and report
    /// `(describe_config, grid_axes, input_field_names)` from `sweep_grid` plus
    /// the Input's physical field names — no bridge, no profiling, no GPU.
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
            profile_kind: S::profile_kind,
            config_fields: serde_field_names::<S::Config>,
            input_fields: serde_field_names::<S::Input>,
            coord_fields: <S::Input as SweepCoords>::coord_field_names,
            compute_dtype_field: <S::Config as KernelConfig>::COMPUTE_DTYPE_FIELD,
            kv_dtype_field: <S::Config as KernelConfig>::KV_DTYPE_FIELD,
            rows: rows_from_json::<S>,
            build: build_probe_from_json::<S>,
            describe: describe_from_json::<S>,
        }
    }
}

/// The profile.db rows one kernel instance measures, and which of them each
/// runtime input reads. The `rows`-path fn pointer; pure metadata (no bridge,
/// DB or GPU).
///
/// Two transforms, both the kernel's own code:
/// - config → rows: `enumerate` over `sweep_grid`, one row of DB args per grid
///   cell, row-major. A column whose value changes across the cells is swept;
///   one that never changes is fixed by the config.
/// - input → rows: `cache_coords` places a runtime input in the grid. The rows
///   it reads are the grid cells around it: on an axis value that value, between
///   two values both, beyond the grid the nearest edge (`extrapolated`). How the
///   cache weighs those cells depends on the cache kind and is not reported.
///
/// `backend` defaults to the config's first backend.
fn rows_from_json<S>(
    config: serde_json::Value,
    backend: Option<&str>,
    inputs: &[serde_json::Value],
) -> anyhow::Result<serde_json::Value>
where
    S: KernelSpec,
    S::Config: serde::de::DeserializeOwned,
    S::Input: serde::de::DeserializeOwned,
{
    use serde_json::{json, Map, Value};

    let config: S::Config = serde_json::from_value(config)
        .map_err(|e| anyhow::anyhow!("config does not match {} KernelConfig: {e}", S::KIND))?;
    S::validate_config(&config)?;
    let backend = match backend {
        Some(name) => *config
            .backends()
            .iter()
            .find(|b| **b == name)
            .ok_or_else(|| anyhow::anyhow!("{} config has no backend {name}", S::KIND))?,
        None => *config
            .backends()
            .first()
            .ok_or_else(|| anyhow::anyhow!("{} config lists no backend", S::KIND))?,
    };
    let grid = S::sweep_grid(&config);
    let axes = grid.axes();
    let cells = grid.expand(|coords| coords.to_vec());
    let args = S::enumerate(&config, &grid, backend);
    anyhow::ensure!(
        args.len() == cells.len(),
        "{} enumerate returned {} rows for {} grid cells",
        S::KIND,
        args.len(),
        cells.len()
    );
    let infeasible = S::infeasible_mask(&config, &grid);

    let mut swept = Vec::new();
    let mut fixed = Map::new();
    for (column, first) in args[0].fields() {
        if args
            .iter()
            .all(|row| row.fields().get(column) == Some(first))
        {
            fixed.insert(column.clone(), first.clone());
        } else {
            swept.push(column.clone());
        }
    }

    let rows: Vec<Value> = cells
        .iter()
        .zip(&args)
        .enumerate()
        .map(|(i, (coords, row))| {
            json!({
                "coords": coords,
                "args": row.fields(),
                "feasible": !infeasible.get(i).copied().unwrap_or(false),
            })
        })
        .collect();

    let mut reads = Vec::with_capacity(inputs.len());
    for input in inputs {
        let parsed: S::Input = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("input does not match {} Input: {e}", S::KIND))?;
        let coords = S::cache_coords(&config, &parsed);
        let mut extrapolated = false;
        // Per axis, the indices of the cells around the coordinate.
        let around: Vec<Vec<usize>> = axes
            .iter()
            .zip(coords.as_slice())
            .map(|(axis, &x)| {
                let last = axis.len() - 1;
                if x < axis[0] || x > axis[last] {
                    extrapolated = true;
                    vec![if x < axis[0] { 0 } else { last }]
                } else if let Some(i) = axis.iter().position(|&v| v == x) {
                    vec![i]
                } else {
                    let upper = axis
                        .iter()
                        .position(|&v| v > x)
                        .expect("x is inside the axis");
                    vec![upper - 1, upper]
                }
            })
            .collect();
        let mut indices = vec![0usize];
        for (d, picks) in around.iter().enumerate() {
            let stride: usize = axes[d + 1..].iter().map(Vec::len).product();
            indices = indices
                .iter()
                .flat_map(|base| picks.iter().map(move |p| base + p * stride))
                .collect();
        }
        reads.push(json!({
            "input": input,
            "coords": coords.as_slice(),
            "extrapolated": extrapolated,
            "rows": indices,
        }));
    }

    Ok(json!({
        "kind": S::KIND,
        "profile_kind": S::profile_kind(),
        "backend": backend,
        "cache_coords": <S::Input as SweepCoords>::coord_field_names(),
        "grid_axes": axes,
        "swept": swept,
        "fixed": fixed,
        "rows": rows,
        "inputs": reads,
    }))
}

/// The field names serde passes to `deserialize_struct` for `T`: a derived
/// struct's fields, in order, with any `rename` applied. Read by a deserializer
/// that records them and then refuses the value, so no `T` is built. Empty for
/// a type that does not deserialize as a struct (e.g. one with a flattened
/// field, which deserializes as a map).
fn serde_field_names<T: serde::de::DeserializeOwned>() -> &'static [&'static str] {
    use serde::de::{self, Visitor};

    struct Probe<'a>(&'a mut &'static [&'static str]);

    impl<'de> de::Deserializer<'de> for Probe<'_> {
        type Error = de::value::Error;

        fn deserialize_any<V: Visitor<'de>>(self, _: V) -> Result<V::Value, Self::Error> {
            Err(de::Error::custom("field-name probe"))
        }

        fn deserialize_struct<V: Visitor<'de>>(
            self,
            _name: &'static str,
            fields: &'static [&'static str],
            _visitor: V,
        ) -> Result<V::Value, Self::Error> {
            *self.0 = fields;
            Err(de::Error::custom("field-name probe"))
        }

        serde::forward_to_deserialize_any! {
            bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
            bytes byte_buf option unit unit_struct newtype_struct seq tuple
            tuple_struct map enum identifier ignored_any
        }
    }

    let mut fields: &'static [&'static str] = &[];
    let _ = T::deserialize(Probe(&mut fields));
    fields
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
    S::validate_config(&config)?;
    let kernel = Kernel::<S>::build(S::KIND.to_string(), config, bridge)?;
    Ok(Box::new(kernel))
}

/// Deserialize `config` into `S::Config` and report its one-line summary, the
/// fitted cache axes from `sweep_grid`, and the Input's physical query-field
/// names. The `grid`-path fn pointer — pure metadata, so it needs neither the
/// bridge nor a built cache.
fn describe_from_json<S>(
    config: serde_json::Value,
) -> anyhow::Result<(serde_json::Value, Vec<Vec<f64>>, &'static [&'static str])>
where
    S: KernelSpec,
    S::Config: serde::de::DeserializeOwned,
{
    let config: S::Config = serde_json::from_value(config)
        .map_err(|e| anyhow::anyhow!("config does not match {} KernelConfig: {e}", S::KIND))?;
    S::validate_config(&config)?;
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

/// The grid `config` asks profile.db for: cache axes, the args of each cell
/// (without `backend`) and the infeasible cells. Enumerates every backend so a
/// kernel whose args depend on the backend fails here instead of recording
/// rows that only one backend reads.
fn config_grid<S: KernelSpec>(config: &S::Config) -> Result<ConfigGrid, BuildError> {
    let grid = S::sweep_grid(config);
    let cells_for = |backend: &'static str| -> Vec<BTreeMap<String, serde_json::Value>> {
        S::enumerate(config, &grid, backend)
            .into_iter()
            .map(|payload| {
                let mut args = payload.fields().clone();
                args.remove("backend");
                args
            })
            .collect()
    };
    let (&first, rest) = config
        .backends()
        .split_first()
        .expect("ensure_has_backends ran before recording");
    let cells = cells_for(first);
    for &other in rest {
        if cells_for(other) != cells {
            return Err(BuildError::BackendDependentArgs {
                kind: S::KIND,
                first,
                other,
            });
        }
    }
    let infeasible = S::infeasible_mask(config, &grid)
        .iter()
        .enumerate()
        .filter_map(|(i, &drop)| drop.then_some(i))
        .collect();
    Ok(ConfigGrid {
        cache_coords: <S::Input as SweepCoords>::coord_field_names(),
        axes: grid.axes().to_vec(),
        cells,
        infeasible,
    })
}

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
    use super::{ensure_has_backends, KernelConfig, KernelSpec};
    use crate::timing::bridge::{ArgsPayload, KernelKind};
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::{BuildError, Coords, SweepCoords, SweepGrid};

    #[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize)]
    struct DefaultCoordsConfig {
        backends: Vec<&'static str>,
        gpu_name: String,
    }

    impl KernelConfig for DefaultCoordsConfig {
        fn backends(&self) -> &[&'static str] {
            &self.backends
        }

        fn set_backends(&mut self, backends: Vec<&'static str>) {
            self.backends = backends;
        }

        const BACKENDS_FIELD: &'static str = "DefaultCoordsConfig.backends";

        fn gpu_name(&self) -> &str {
            &self.gpu_name
        }
    }

    struct DefaultCoordsInput;

    impl SweepCoords for DefaultCoordsInput {
        fn coords(&self) -> Coords {
            Coords::new([3.0, 5.0])
        }

        fn coord_field_names() -> &'static [&'static str] {
            &["x", "y"]
        }
    }

    struct DefaultCoordsSpec;

    impl KernelSpec for DefaultCoordsSpec {
        type Config = DefaultCoordsConfig;
        type Input = DefaultCoordsInput;

        const KIND: KernelKind = "default_coords_test";

        fn sweep_grid(_config: &Self::Config) -> SweepGrid {
            SweepGrid::new(vec![vec![1.0], vec![1.0]])
        }

        fn cache_kind(_backend: &'static str) -> CacheKind {
            CacheKind::Cache2DLinear(Extrapolation::Clamp)
        }

        fn enumerate(
            _config: &Self::Config,
            _grid: &SweepGrid,
            _backend: &'static str,
        ) -> Vec<ArgsPayload> {
            Vec::new()
        }
    }

    /// A 2x2 grid whose cell (1, 1) is infeasible; `by_backend` makes one args
    /// column follow the backend.
    struct GridSpec;

    impl KernelSpec for GridSpec {
        type Config = DefaultCoordsConfig;
        type Input = DefaultCoordsInput;

        const KIND: KernelKind = "grid_test";

        fn sweep_grid(_config: &Self::Config) -> SweepGrid {
            SweepGrid::new(vec![vec![1.0, 2.0], vec![10.0, 20.0]])
        }

        fn cache_kind(_backend: &'static str) -> CacheKind {
            CacheKind::Cache2DLinear(Extrapolation::Clamp)
        }

        fn infeasible_mask(_config: &Self::Config, _grid: &SweepGrid) -> Vec<bool> {
            vec![false, false, false, true]
        }

        fn enumerate(
            config: &Self::Config,
            grid: &SweepGrid,
            backend: &'static str,
        ) -> Vec<ArgsPayload> {
            grid.expand(|coords| {
                let payload = ArgsPayload::new()
                    .with("backend", backend)
                    .with("x", coords[0] as u32)
                    .with("y", coords[1] as u32);
                if config.gpu_name == "by_backend" {
                    payload.with("tile", backend)
                } else {
                    payload
                }
            })
        }
    }

    #[test]
    fn config_grid_records_args_without_backend_and_infeasible_cells() {
        let config = DefaultCoordsConfig {
            backends: vec!["a", "b"],
            gpu_name: "test-gpu".to_string(),
        };
        let grid = super::config_grid::<GridSpec>(&config).expect("grid");
        assert_eq!(grid.cache_coords, &["x", "y"]);
        assert_eq!(grid.axes, vec![vec![1.0, 2.0], vec![10.0, 20.0]]);
        let cells: Vec<_> = grid
            .cells
            .iter()
            .map(|cell| serde_json::to_value(cell).unwrap())
            .collect();
        assert_eq!(
            cells,
            vec![
                serde_json::json!({"x": 1, "y": 10}),
                serde_json::json!({"x": 1, "y": 20}),
                serde_json::json!({"x": 2, "y": 10}),
                serde_json::json!({"x": 2, "y": 20}),
            ]
        );
        assert_eq!(grid.infeasible, vec![3]);
    }

    #[test]
    fn config_grid_rejects_args_that_follow_the_backend() {
        let config = DefaultCoordsConfig {
            backends: vec!["a", "b"],
            gpu_name: "by_backend".to_string(),
        };
        assert!(matches!(
            super::config_grid::<GridSpec>(&config),
            Err(BuildError::BackendDependentArgs {
                kind: "grid_test",
                first: "a",
                other: "b",
            })
        ));
    }

    #[test]
    fn identity_leaves_out_gpu_and_backends() {
        let config = DefaultCoordsConfig {
            backends: vec!["a"],
            gpu_name: "test-gpu".to_string(),
        };
        assert_eq!(config.identity(), serde_json::json!({}));
    }

    #[test]
    fn ensure_has_backends_rejects_empty_backend_lists() {
        assert!(matches!(
            ensure_has_backends("single_gemm", "backends", &[]),
            Err(BuildError::FitFailed { .. })
        ));
    }

    #[test]
    fn default_cache_coords_preserve_input_sweep_coords() {
        let config = DefaultCoordsConfig {
            backends: vec!["test"],
            gpu_name: "test-gpu".to_string(),
        };

        assert_eq!(
            &*DefaultCoordsSpec::cache_coords(&config, &DefaultCoordsInput),
            &[3.0, 5.0]
        );
    }
}
