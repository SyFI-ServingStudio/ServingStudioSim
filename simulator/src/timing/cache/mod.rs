//! Cache interpolation primitives for L1 kernels.

use crate::common::time::Time;
use crate::timing::bridge::{BuildError, KernelKind, KernelMetrics};
use crate::timing::cache::interp::{CoverageFlags, LeafMetrics, Metrics4};
use crate::timing::sweep::SweepGrid;
use crate::timing::LookupResult;

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

    fn lookup(&self, sweep: &[f64]) -> LookupResult;

    /// Hot-path fast path: interpolate only the wallclock time, skipping the
    /// `Arc<str>` name, flops/bytes/energy, and the warning/breakdown `Vec`s
    /// that `lookup` builds. Per-tick sim callers that only advance the clock
    /// use this. Impls must keep it consistent with `lookup().time` by routing
    /// both through the same interval-locate helper.
    fn lookup_time(&self, sweep: &[f64]) -> Time;

    /// All four metrics + coverage flags, allocation-free — the CostTree eval
    /// path's per-leaf value (streamed into `buf[slot]`, then rolled up by
    /// [`CostTree::aggregate`](crate::timing::CostTree)). Same numbers as
    /// `lookup` (NaN/empty → zero, fields clamped non-negative) and the same
    /// coverage *kinds*, but without the `Arc<str>` name or warning `Vec`/`String`
    /// details. The default delegates to `lookup` and narrows; the interpolating
    /// caches override it with their alloc-free `interpolate` path.
    fn lookup_metrics(&self, sweep: &[f64]) -> LeafMetrics {
        let r = self.lookup(sweep);
        let mut coverage = CoverageFlags::EMPTY;
        for w in &r.warnings {
            coverage |= CoverageFlags::from_kind(w.kind);
        }
        LeafMetrics {
            m: Metrics4 {
                time_ms: r.time.as_ms() as f32,
                flops: r.flops as f32,
                bytes: r.bytes as f32,
                energy_j: r.energy_j as f32,
            },
            coverage,
        }
    }
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
