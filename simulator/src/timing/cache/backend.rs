//! Per-backend wrapper around a fitted `Box<dyn Cache>`.
//!
//! `*Kernel` holds a `Vec<BackendCache>` (one per entry in
//! `*KernelConfig.backends`) and runs best-of-N over their `.lookup()` results.
//! Keeping this struct in its own file isolates the "wrap + annotate + rewrite
//! warnings" concern from the abstract `Cache` trait and its variants in
//! `cache/mod.rs`.

use crate::common::time::Time;
use crate::timing::bridge::{BuildError, KernelKind, KernelMetrics};
use crate::timing::cache::interp::LeafMetrics;
use crate::timing::cache::{build_cache, Cache, CacheKind, OutlierWarning};
use crate::timing::sweep::SweepGrid;
use crate::timing::LookupResult;

/// Per-backend cache wrapper: a `Box<dyn Cache>` annotated with the backend
/// label and the `CacheKind` it was built as. `lookup` rewrites warning details
/// with `"{backend}:{cache_kind}: ..."` so the final warning still identifies
/// the source row after best-of-N selection.
pub(crate) struct BackendCache {
    backend: &'static str,
    cache_kind: CacheKind,
    cache: Box<dyn Cache>,
}

impl BackendCache {
    pub(crate) fn fit(
        kernel_kind: KernelKind,
        backend: &'static str,
        cache_kind: CacheKind,
        sweep_grid: &SweepGrid,
        samples: &[KernelMetrics],
    ) -> Result<(Self, Vec<OutlierWarning>), BuildError> {
        let (cache, warnings) = build_cache(kernel_kind, cache_kind, sweep_grid, samples)?;
        Ok((
            Self {
                backend,
                cache_kind,
                cache,
            },
            warnings,
        ))
    }

    pub(crate) fn lookup(&self, sweep: &[f64]) -> LookupResult {
        let mut result = self.cache.lookup(sweep);
        result.selected_backend = Some(self.backend);
        for warning in &mut result.warnings {
            warning.detail = format!("{}:{:?}: {}", self.backend, self.cache_kind, warning.detail);
        }
        result
    }

    /// Time-only fast path — no backend stamping or warning rewrite needed, the
    /// caller is comparing wallclocks for best-of-N selection. See
    /// `Cache::lookup_time`.
    pub(crate) fn lookup_time(&self, sweep: &[f64]) -> Time {
        self.cache.lookup_time(sweep)
    }

    /// Metrics fast path for CostTree eval — no backend stamping, the caller
    /// (`Kernel::lookup_metrics`) selects best-of-N itself. See
    /// `Cache::lookup_metrics`.
    pub(crate) fn lookup_metrics(&self, sweep: &[f64]) -> LeafMetrics {
        self.cache.lookup_metrics(sweep)
    }
}

#[cfg(test)]
mod tests {
    use super::BackendCache;
    use crate::timing::bridge::KernelMetrics;
    use crate::timing::cache::CacheKind;
    use crate::timing::sweep::SweepGrid;

    fn sample(time_ms: f64) -> KernelMetrics {
        KernelMetrics {
            time_ms,
            tflops: Some(1.0),
            memory_bandwidth_gbps: Some(2.0),
            algbw_gbps: None,
            busbw_gbps: None,
            message_size_bytes: None,
            energy_j: 1.0,
        }
    }

    #[test]
    fn backend_cache_lookup_stamps_selected_backend_on_inner_leaf() {
        let grid = SweepGrid::new(vec![vec![1.0, 2.0]]);
        let (cache, _warnings) = BackendCache::fit(
            "single_gemm",
            "torch",
            CacheKind::Cache1DLinear,
            &grid,
            &[sample(1.0), sample(3.0)],
        )
        .expect("fit must succeed for a finite 1D batch");

        let result = cache.lookup(&[1.5]);
        assert_eq!(result.selected_backend, Some("torch"));
    }
}
