//! Generic L1 kernel engine: per-kernel files implement `KernelSpec` once and
//! `Kernel<S>` provides init/lookup/dry_run + Probe/DryRun blanket impls.
//!
//! Engine knows nothing about specific kernel kinds: it only sees the sweep
//! grid, the cache kind, the bridge args, and the sweep-coord projection of
//! the runtime input. Comm / compute / distribution-sensitive kernels are all
//! the same shape here — variations live in each per-kernel `KernelSpec`.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::common::time::Time;
use crate::timing::bridge::{ArgsPayload, KernelKind, PerfApiBridge};
use crate::timing::cache::{BackendCache, CacheKind, OutlierWarning};
use crate::timing::sweep::{SweepCoords, SweepGrid};
use crate::timing::{BuildError, Describe, DryRun, JitPlan, LookupResult, Probe};

/// Per-kernel `*KernelConfig` contract: identity (`Hash + Eq`) + the required
/// `backends: Vec<&'static str>` field exposed via `backends()`. The proc-macro
/// `#[derive(KernelConfig)]` in `timing-kernel-derive` generates this impl by
/// reading `&self.backends` directly; structs without a `backends` field fail
/// to derive at the generated access site.
pub trait KernelConfig: std::hash::Hash + Eq + Clone + std::fmt::Debug + 'static {
    fn backends(&self) -> &[&'static str];
    /// Used in build-error messages, e.g. `"SingleGemmKernelConfig.backends"`.
    /// Auto-derived as `"{StructName}.backends"`.
    const BACKENDS_FIELD: &'static str;

    /// The GPU whose profiled rows this config caches. Part of the config
    /// identity (`Hash + Eq`), so distinct GPUs are distinct kernels / caches;
    /// passed to the bridge at `init` / `dry_run` as the DB `gpu_name` key.
    fn gpu_name(&self) -> &str;

    /// One-line config summary for the `Describe` leaf line — the `<cfg>` after
    /// `<name> (<KIND>)`. The default is the full `{self:?}`; `#[derive(KernelConfig)]`
    /// overrides it with a tidy `field=value` list over every field (including
    /// `backends`), dropping only the struct-name + braces wrapper.
    fn describe_config(&self) -> String {
        format!("{self:?}")
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

    fn sweep_grid(config: &Self::Config) -> SweepGrid;
    /// Decided by backend only — Config is already fixed inside any one Spec.
    /// Cache-shape differences driven by Config (e.g. fp8 vs fp16) should live
    /// in separate Spec types, not in this function's body.
    fn cache_kind(backend: &'static str) -> CacheKind;
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
    /// `Arc<str>` so `lookup` stamps this onto each result with a refcount bump,
    /// not a per-tick `String` clone (hot path — see `lookup`).
    name: Arc<str>,
    backend_caches: Vec<BackendCache>,
    _spec: PhantomData<fn() -> S>,
}

impl<S: KernelSpec> Kernel<S> {
    pub fn init(
        name: String,
        config: S::Config,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        ensure_has_backends(
            S::KIND,
            <S::Config as KernelConfig>::BACKENDS_FIELD,
            config.backends(),
        )?;
        let sweep_grid = S::sweep_grid(&config);
        let backends = config.backends();
        let mut backend_caches = Vec::with_capacity(backends.len());
        let mut outlier_warnings = Vec::new();
        for &backend in backends {
            let specs = S::enumerate(&config, &sweep_grid, backend);
            let samples = bridge
                .get_times(specs, S::KIND, config.gpu_name())
                .map_err(|err| BuildError::from_perf_api(S::KIND, backend, err))?;
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
            name: name.into(),
            backend_caches,
            _spec: PhantomData,
        })
    }

    pub fn lookup(&self, input: &S::Input) -> LookupResult {
        let coords = input.coords();
        let mut result = fastest_lookup(&self.backend_caches, &coords);
        result.name = self.name.clone();
        result
    }

    /// Time-only fast path: the minimum wallclock across backends, with none of
    /// the `LookupResult` machinery (`Arc<str>` name clone, flops/bytes/energy,
    /// warning/breakdown `Vec`s). For per-tick callers that only advance the
    /// sim clock. Best-of-N still holds: it returns `min` over the backends.
    pub fn lookup_time(&self, input: &S::Input) -> Time {
        let coords = input.coords();
        self.backend_caches
            .iter()
            .map(|backend_cache| backend_cache.lookup_time(&coords))
            .min()
            .expect("kernel config validation must create at least one backend cache")
    }

    pub fn dry_run(
        name: &str,
        config: &S::Config,
        bridge: &PerfApiBridge,
    ) -> Result<JitPlan, BuildError> {
        ensure_has_backends(
            S::KIND,
            <S::Config as KernelConfig>::BACKENDS_FIELD,
            config.backends(),
        )?;
        let sweep_grid = S::sweep_grid(config);
        let backends = config.backends();
        let mut parts = Vec::with_capacity(backends.len());
        for &backend in backends {
            let specs = S::enumerate(config, &sweep_grid, backend);
            let total = specs.len();
            let missing = bridge
                .count_missing(specs, S::KIND, backend, config.gpu_name())
                .map_err(|err| BuildError::from_perf_api(S::KIND, backend, err))?;
            parts.push(JitPlan::from_missing_count(
                S::KIND,
                name,
                backend,
                total,
                missing,
            )?);
        }
        Ok(JitPlan::sum(name.to_string(), parts))
    }
}

impl<S: KernelSpec> Probe for Kernel<S> {
    type Input = S::Input;
    fn lookup(&self, input: &Self::Input) -> LookupResult {
        Self::lookup(self, input)
    }
    fn lookup_time(&self, input: &Self::Input) -> Time {
        Self::lookup_time(self, input)
    }
}

impl<S: KernelSpec> Describe for Kernel<S> {
    /// Leaf line: `<name> (<KIND>) <cfg fields>`. One blanket impl covers every
    /// kernel because all of them are `Kernel<S>` — no per-kernel macro. The cfg
    /// fields come from `KernelConfig: Debug`.
    fn describe(&self, depth: usize, out: &mut String) {
        use std::fmt::Write;
        writeln!(
            out,
            "{}{} ({}) {}",
            "│  ".repeat(depth),
            self.name,
            S::KIND,
            self.config.describe_config(),
        )
        .unwrap();
    }
}

impl<S: KernelSpec> DryRun for Kernel<S> {
    type Config = S::Config;
    fn dry_run(
        name: &str,
        config: &Self::Config,
        bridge: &PerfApiBridge,
    ) -> Result<JitPlan, BuildError> {
        Self::dry_run(name, config, bridge)
    }
}

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

fn fastest_lookup<'a>(
    backend_caches: impl IntoIterator<Item = &'a BackendCache>,
    sweep: &[f64],
) -> LookupResult {
    backend_caches
        .into_iter()
        .map(|backend_cache| backend_cache.lookup(sweep))
        .min_by_key(|result| result.time)
        .expect("kernel config validation must create at least one backend cache")
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
