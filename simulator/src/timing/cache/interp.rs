//! Shared interpolation primitives for the linear caches (`Cache1DLinear`,
//! `Cache2DLinear`): the branchless interval `locate` and the f32 `Metrics4`
//! sample type with its lerp / weighted-blend ops.
//!
//! Metrics are stored and interpolated in `f32`. A timing model needs ~3-4
//! significant digits; f32 carries ~7, and halving the footprint roughly
//! doubles cache-line density of the axis/cell arrays — and the lookup hot path
//! is bound on those binary-search loads, not on arithmetic width (scalar f32
//! and f64 cost the same on x86).

use crate::timing::bridge::KernelMetrics;

/// Max fractional time drop tolerated between adjacent (axis-sorted) profile
/// points before a cache flags `OutlierKind::MonotonicityBreak`. 10% absorbs
/// measurement noise; a larger dip signals a real outlier (scheduler cliff,
/// bad sample). Shared by the 1D and 2D fit-time scans.
pub(crate) const MONOTONICITY_TOLERANCE: f32 = 0.10;

/// One profiled point's metrics in f32. The 1D cache stores a `Vec<Metrics4>`;
/// the 2D cache a `Vec<Option<Metrics4>>` grid (`None` = dropped non-finite).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Metrics4 {
    pub time_ms: f32,
    pub flops: f32,
    pub bytes: f32,
    pub energy_j: f32,
}

impl Metrics4 {
    pub const ZERO: Metrics4 = Metrics4 {
        time_ms: 0.0,
        flops: 0.0,
        bytes: 0.0,
        energy_j: 0.0,
    };

    /// Narrow a profiled `KernelMetrics` (f64 / u64) into the f32 cache form.
    pub fn from_sample(sample: &KernelMetrics) -> Self {
        Metrics4 {
            time_ms: sample.time_ms as f32,
            flops: sample.flops() as f32,
            bytes: sample.bytes() as f32,
            energy_j: sample.energy_j as f32,
        }
    }

    /// Field-wise linear blend `self*(1-t) + other*t`. `t` outside `[0, 1]`
    /// extrapolates (the 1D two-point path).
    pub fn lerp(self, other: Metrics4, t: f32) -> Metrics4 {
        Metrics4 {
            time_ms: self.time_ms + (other.time_ms - self.time_ms) * t,
            flops: self.flops + (other.flops - self.flops) * t,
            bytes: self.bytes + (other.bytes - self.bytes) * t,
            energy_j: self.energy_j + (other.energy_j - self.energy_j) * t,
        }
    }

    /// `self += other * weight`, field-wise (the 2D bilinear accumulation).
    pub fn add_scaled(&mut self, other: Metrics4, weight: f32) {
        self.time_ms += other.time_ms * weight;
        self.flops += other.flops * weight;
        self.bytes += other.bytes * weight;
        self.energy_j += other.energy_j * weight;
    }

    pub fn scale(&mut self, factor: f32) {
        self.time_ms *= factor;
        self.flops *= factor;
        self.bytes *= factor;
        self.energy_j *= factor;
    }
}

/// Branchless interval locate on an ascending axis. Returns `(lo, hi, t,
/// outside)`: the bracketing segment `[xs[lo], xs[hi]]` (`hi == lo` only for a
/// single-point axis), the interpolation fraction `t` (outside `[0, 1]` when
/// extrapolating past an edge), and whether `x` fell beyond `[xs[0], xs[last]]`.
///
/// The lower-bound search uses a cmov (`if .. { mid } else { lo }`) instead of a
/// textbook binary search: on ~6 comparisons the branchy form spends most of
/// its time recovering from mispredicts, which the cache benchmark showed
/// dominated lookup latency (a single 63-element `partition_point`-based bracket
/// cost ~24ns; branchless cut it to ~10ns).
pub(crate) fn locate(xs: &[f32], x: f32) -> (usize, usize, f32, bool) {
    let n = xs.len();
    if n == 1 {
        return (0, 0, 0.0, (x - xs[0]).abs() > f32::EPSILON);
    }
    let last = n - 1;
    let mut lo = 0usize;
    let mut size = n;
    while size > 1 {
        let half = size / 2;
        let mid = lo + half;
        lo = if xs[mid] <= x { mid } else { lo };
        size -= half;
    }
    // Clamp to an interior segment so out-of-range `x` extrapolates off the
    // boundary segment rather than indexing past the end.
    let lo = lo.min(last - 1);
    let hi = lo + 1;
    let width = xs[hi] - xs[lo];
    let t = if width.abs() <= f32::EPSILON {
        0.0
    } else {
        (x - xs[lo]) / width
    };
    let outside = x < xs[0] || x > xs[last];
    (lo, hi, t, outside)
}
