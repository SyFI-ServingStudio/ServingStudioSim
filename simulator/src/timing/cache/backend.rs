//! Per-backend wrapper around a fitted `Box<dyn Cache>`.
//!
//! `*Kernel` holds a `Vec<BackendCache>` (one per entry in
//! `*KernelConfig.backends`) and runs best-of-N over their `.eval()`
//! results. Keeping this struct in its own file isolates the wrapping concern
//! from the abstract `Cache` trait and its variants in `cache/mod.rs`.

use crate::timing::bridge::{BuildError, KernelKind, KernelMetrics};
use crate::timing::cache::interp::{CoverageFlags, LeafMetrics};
use crate::timing::cache::{build_cache, Cache, CacheKind, OutlierWarning, PeakRates};
use crate::timing::sweep::SweepGrid;

/// Per-backend cache wrapper: a fitted `Box<dyn Cache>` and the range of each
/// grid axis. `*Kernel` runs best-of-N over a `Vec<BackendCache>` via `eval`.
pub(crate) struct BackendCache {
    cache: Box<dyn Cache>,
    bounds: Vec<(f64, f64)>,
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
        let bounds = sweep_grid
            .axes()
            .iter()
            .map(|axis| {
                axis.iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                        (lo.min(x), hi.max(x))
                    })
            })
            .collect();
        Ok((Self { cache, bounds }, warnings))
    }

    /// Whether `sweep` lies within every axis's profiled range.
    pub(crate) fn contains(&self, sweep: &[f64]) -> bool {
        sweep
            .iter()
            .zip(&self.bounds)
            .all(|(&x, &(lo, hi))| (lo..=hi).contains(&x))
    }

    /// The answer past the grid that holds the nearest grid point's bandwidth:
    /// clamp `sweep` into the grid, read that point, and scale all its metrics
    /// by the input's logical `bytes` over the point's own, so the input runs
    /// at the edge's bandwidth. Flagged `EXTRAPOLATED`. Falls back to the
    /// cache's own extrapolation when the edge reports no bytes.
    pub(crate) fn eval_at_edge_bandwidth(&self, sweep: &[f64], bytes: f64) -> LeafMetrics {
        let clamped: Vec<f64> = sweep
            .iter()
            .zip(&self.bounds)
            .map(|(&x, &(lo, hi))| x.clamp(lo, hi))
            .collect();
        let mut edge = self.cache.eval(&clamped);
        let edge_bytes = f64::from(edge.m.bytes);
        if !(edge_bytes > 0.0 && edge_bytes.is_finite() && bytes.is_finite() && bytes >= 0.0) {
            return self.cache.eval(sweep);
        }
        edge.m.scale((bytes / edge_bytes) as f32);
        edge.coverage |= CoverageFlags::EXTRAPOLATED;
        edge
    }

    /// Metrics fast path for CostTree eval — the caller (`Kernel::eval`)
    /// selects best-of-N itself. See `Cache::eval`.
    pub(crate) fn eval(&self, sweep: &[f64]) -> LeafMetrics {
        self.cache.eval(sweep)
    }

    /// Peak achieved rates over this backend's fitted grid — see
    /// [`Cache::peak_rates`]. `Kernel::peak_rates` merges these across backends.
    pub(crate) fn peak_rates(&self) -> PeakRates {
        self.cache.peak_rates()
    }
}

#[cfg(test)]
mod tests {
    use super::BackendCache;
    use crate::timing::bridge::KernelMetrics;
    use crate::timing::cache::interp::CoverageFlags;
    use crate::timing::cache::CacheKind;
    use crate::timing::sweep::SweepGrid;

    fn sample(time_ms: f64) -> KernelMetrics {
        KernelMetrics {
            time_ms,
            tflops: Some(1.0),
            memory_bandwidth_gbps: Some(2.0),
            algbw_gbps: None,
            busbw_gbps: None,
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

    #[test]
    fn past_the_grid_holds_the_edge_bandwidth() {
        let grid = SweepGrid::new(vec![vec![1.0, 2.0]]);
        let (cache, _warnings) = BackendCache::fit(
            "single_gemm",
            "torch",
            CacheKind::Cache1DLinear,
            &grid,
            &[sample(1.0), sample(3.0)],
        )
        .expect("fit must succeed for a finite 1D batch");
        assert!(cache.contains(&[2.0]));
        assert!(!cache.contains(&[10.0]));

        // The edge (x=2) runs 3 ms at 2 GB/s: 6e6 bytes. The cache's own line
        // would give 19 ms at x=10.
        let at_bandwidth = cache.eval_at_edge_bandwidth(&[10.0], 2e7);
        assert!((at_bandwidth.m.time_ms - 10.0).abs() < 1e-4);
        assert!((at_bandwidth.m.energy_j - 10.0 / 3.0).abs() < 1e-4);
        assert!(at_bandwidth.coverage.contains(CoverageFlags::EXTRAPOLATED));
    }
}
