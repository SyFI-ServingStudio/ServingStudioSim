//! Cache interpolation primitives for L1 kernels.

use crate::timing::bridge::{BuildError, KernelKind, KernelMetrics};
use crate::timing::cache::interp::{LeafMetrics, Metrics4};
use crate::timing::sweep::SweepGrid;

pub mod backend;
pub mod cliff_2d;
pub mod direct_1d;
pub mod interp;
pub mod linear_1d;
pub mod linear_2d;
pub mod log_2d;

pub(crate) use backend::BackendCache;
pub use cliff_2d::Cache2DCliff;
pub use direct_1d::Cache1DDirect;
pub use linear_1d::Cache1DLinear;
pub use linear_2d::Cache2DLinear;
pub use log_2d::Cache2DLog;

pub trait Cache: Send + Sync {
    fn from_samples(grid: &SweepGrid, samples: &[KernelMetrics]) -> (Self, Vec<OutlierWarning>)
    where
        Self: Sized;

    /// All four metrics + coverage flags, allocation-free — the CostTree eval
    /// path's per-leaf value (streamed into `buf[slot]`, then rolled up by
    /// [`CostTree::aggregate`](crate::timing::CostTree)). NaN/empty → zero, fields
    /// clamped non-negative; off-grid lookups set `EXTRAPOLATED`/`NO_COVERAGE`.
    fn eval(&self, sweep: &[f64]) -> LeafMetrics;

    /// Peak achieved compute / bandwidth rates over this cache's fitted grid
    /// cells — the per-config "best batching" ceiling the optimality analyzer
    /// divides work by. Reads the stored cells directly (no interpolation, no
    /// coords remap), so it is correct for re-axis caches too. Default zero for
    /// variants that carry no per-cell metrics (never built today).
    fn peak_rates(&self) -> PeakRates {
        PeakRates::default()
    }
}

/// Peak achieved compute / bandwidth rates over a kernel cache's fitted grid —
/// the per-config batching ceiling. `tflops` is the fastest compute rate
/// (`flops / time`) any profiled shape reached; `gbps` the fastest bandwidth
/// (`bytes / time`); `max_arithmetic_intensity_flops_per_byte` records whether
/// any fitted shape can cross the GPU's hardware ridge point. A comm kernel (no
/// flops) reports zero compute rate/intensity and a real `gbps`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PeakRates {
    pub tflops: f64,
    pub gbps: f64,
    pub max_arithmetic_intensity_flops_per_byte: f64,
}

impl PeakRates {
    /// Element-wise max — the best over a kernel's several backend caches
    /// (best-of-N applies to peaks too).
    pub fn merge(self, other: PeakRates) -> PeakRates {
        PeakRates {
            tflops: self.tflops.max(other.tflops),
            gbps: self.gbps.max(other.gbps),
            max_arithmetic_intensity_flops_per_byte: self
                .max_arithmetic_intensity_flops_per_byte
                .max(other.max_arithmetic_intensity_flops_per_byte),
        }
    }
}

/// Fold a cache's stored cells into their peak achieved rates. A cell counts only
/// when its time is finite and strictly positive (dropped / infeasible cells hold
/// `Metrics4::ZERO` or a non-finite value and are skipped); each rate counts only
/// when its numerator is finite and positive (a comm cell's `flops == 0` adds no
/// compute rate).
pub(crate) fn peak_over_cells(cells: impl IntoIterator<Item = Metrics4>) -> PeakRates {
    let mut peak = PeakRates::default();
    for c in cells {
        let time_ms = c.time_ms as f64;
        if !(time_ms.is_finite() && time_ms > 0.0) {
            continue;
        }
        let secs = time_ms / 1e3;
        let flops = c.flops as f64;
        if flops.is_finite() && flops > 0.0 {
            peak.tflops = peak.tflops.max(flops / secs / 1e12);
        }
        let bytes = c.bytes as f64;
        if bytes.is_finite() && bytes > 0.0 {
            peak.gbps = peak.gbps.max(bytes / secs / 1e9);
            if flops.is_finite() && flops > 0.0 {
                peak.max_arithmetic_intensity_flops_per_byte = peak
                    .max_arithmetic_intensity_flops_per_byte
                    .max(flops / bytes);
            }
        }
    }
    peak
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheKind {
    Cache1DLinear,
    Cache1DDirect,
    Cache2DLinear,
    Cache2DLog,
    Cache2DCliff,
}

/// Structured per-sample / per-cache warning surfaced by `from_samples` and
/// aggregated into `*Kernel::outlier_warnings` for the sim manifest.
///
/// `detail` is a human-readable line for logs; `kind` is the analyzer-facing
/// taxonomy. New variants land here as stage-3 outlier scans go in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutlierKind {
    /// A sample's numeric metrics were not all finite + non-negative.
    NonFinite,
    /// Adjacent grid points violate the expected monotonic-in-time relation.
    MonotonicityBreak,
    /// A scheduler-cliff was detected (e.g. vLLM FA3 block boundary).
    Cliff,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutlierWarning {
    pub kind: OutlierKind,
    pub detail: String,
}

/// Build the cache for one (kernel_kind, cache_kind) pair. `kernel_kind` only
/// flows in so `BuildError::FitFailed` can carry it on the "not yet
/// implemented" / "fit failed" paths; the cache impl itself never sees it.
pub fn build_cache(
    kernel_kind: KernelKind,
    kind: CacheKind,
    grid: &SweepGrid,
    samples: &[KernelMetrics],
) -> Result<(Box<dyn Cache>, Vec<OutlierWarning>), BuildError> {
    match kind {
        CacheKind::Cache1DLinear => {
            let (cache, warnings) = Cache1DLinear::from_samples(grid, samples);
            Ok((Box::new(cache), warnings))
        }
        CacheKind::Cache1DDirect => {
            let (cache, warnings) = Cache1DDirect::from_samples(grid, samples);
            Ok((Box::new(cache), warnings))
        }
        CacheKind::Cache2DLinear => {
            let (cache, warnings) = Cache2DLinear::from_samples(grid, samples);
            Ok((Box::new(cache), warnings))
        }
        CacheKind::Cache2DLog | CacheKind::Cache2DCliff => Err(BuildError::FitFailed {
            kind: kernel_kind,
            reason: format!("cache variant {kind:?} is not yet implemented"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{peak_over_cells, Metrics4};

    #[test]
    fn grid_peaks_include_maximum_arithmetic_intensity() {
        let peak = peak_over_cells([
            Metrics4 {
                time_ms: 2.0,
                flops: 400.0,
                bytes: 20.0,
                energy_j: 0.0,
            },
            Metrics4 {
                time_ms: 1.0,
                flops: 300.0,
                bytes: 10.0,
                energy_j: 0.0,
            },
        ]);

        assert_eq!(peak.max_arithmetic_intensity_flops_per_byte, 30.0);
    }
}
