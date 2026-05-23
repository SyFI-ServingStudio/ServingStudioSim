//! Per-backend wrapper around a fitted `Box<dyn Cache>`.
//!
//! `*Kernel` holds a `Vec<BackendCache>` (one per entry in
//! `*KernelConfig.backends`) and runs best-of-N over their `.eval()`
//! results. Keeping this struct in its own file isolates the wrapping concern
//! from the abstract `Cache` trait and its variants in `cache/mod.rs`.

use crate::timing::bridge::{BuildError, KernelKind, KernelMetrics};
use crate::timing::cache::interp::LeafMetrics;
use crate::timing::cache::{build_cache, Cache, CacheKind, OutlierWarning};
use crate::timing::sweep::SweepGrid;

/// Per-backend cache wrapper: a fitted `Box<dyn Cache>`. `*Kernel` runs
/// best-of-N over a `Vec<BackendCache>` via `eval`.
pub(crate) struct BackendCache {
    cache: Box<dyn Cache>,
}

impl BackendCache {
    pub(crate) fn fit(
        kernel_kind: KernelKind,
        _backend: &'static str,
        cache_kind: CacheKind,
        sweep_grid: &SweepGrid,
        samples: &[KernelMetrics],
    ) -> Result<(Self, Vec<OutlierWarning>), BuildError> {
        let (cache, warnings) = build_cache(kernel_kind, cache_kind, sweep_grid, samples)?;
        Ok((Self { cache }, warnings))
    }

    /// Metrics fast path for CostTree eval — the caller (`Kernel::eval`)
    /// selects best-of-N itself. See `Cache::eval`.
    pub(crate) fn eval(&self, sweep: &[f64]) -> LeafMetrics {
        self.cache.eval(sweep)
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
    fn backend_cache_eval_interpolates() {
        let grid = SweepGrid::new(vec![vec![1.0, 2.0]]);
        let (cache, _warnings) = BackendCache::fit(
            "single_gemm",
            "torch",
            CacheKind::Cache1DLinear,
            &grid,
            &[sample(1.0), sample(3.0)],
        )
        .expect("fit must succeed for a finite 1D batch");

        // Midpoint between the two profiled times (1.0, 3.0) → 2.0 ms.
        let leaf = cache.eval(&[1.5]);
        assert_eq!(leaf.m.time_ms, 2.0);
    }
}
