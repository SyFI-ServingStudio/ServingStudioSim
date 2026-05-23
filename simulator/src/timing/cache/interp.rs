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
use crate::timing::CoverageKind;

/// Max fractional time drop tolerated between adjacent (axis-sorted) profile
/// points before a cache flags `OutlierKind::MonotonicityBreak`. 10% absorbs
/// measurement noise; a larger dip signals a real outlier (scheduler cliff,
/// bad sample). Shared by the 1D and 2D fit-time scans.
pub(crate) const MONOTONICITY_TOLERANCE: f32 = 0.10;

/// One profiled point's metrics in f32. The 1D cache stores a `Vec<Metrics4>`;
/// the 2D cache a `Vec<Option<Metrics4>>` grid (`None` = dropped non-finite).
/// Also the CostTree eval path's per-leaf / per-subtree unit (the `buf[slot]`
/// values [`CostTree::aggregate`](crate::timing::CostTree) rolls up).
#[derive(Clone, Copy, Debug)]
pub struct Metrics4 {
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

    /// Field-wise `max(0)` copy. Mirrors the per-field `.max(0.0)` each cache
    /// `lookup` applies before emitting a `LookupResult`, so the metrics fast
    /// path (`Cache::lookup_metrics`) returns identical non-negative numbers.
    pub fn clamped(self) -> Metrics4 {
        Metrics4 {
            time_ms: self.time_ms.max(0.0),
            flops: self.flops.max(0.0),
            bytes: self.bytes.max(0.0),
            energy_j: self.energy_j.max(0.0),
        }
    }
}

/// The per-leaf coverage signal, packed into a `u8` — one bit per
/// [`CoverageKind`]. This is the CostTree eval path's allocation-free analogue
/// of the `LookupResult` `Vec<CoverageWarning>`: it keeps the analytically
/// useful *kind* (did this leaf extrapolate off-grid? was it a JIT/no-coverage
/// placeholder?) while dropping the per-call `detail: String` (reconstructable
/// off the hot path from the slot name). `aggregate` ORs these up the tree, so a
/// warning anywhere in a subtree surfaces at its root.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoverageFlags(u8);

impl CoverageFlags {
    pub const EMPTY: Self = Self(0);
    pub const EXTRAPOLATED: Self = Self(1 << 0);
    pub const JIT: Self = Self(1 << 1);
    pub const NO_COVERAGE: Self = Self(1 << 2);

    /// One-to-one with the runtime [`CoverageKind`] variants, for the
    /// `lookup`-delegating defaults that narrow a `LookupResult`'s warnings.
    pub fn from_kind(kind: CoverageKind) -> Self {
        match kind {
            CoverageKind::Extrapolated => Self::EXTRAPOLATED,
            CoverageKind::Jit => Self::JIT,
            CoverageKind::NoCoverage => Self::NO_COVERAGE,
        }
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Raw bits, for logging the per-slot coverage as a `u8` column.
    pub fn bits(self) -> u8 {
        self.0
    }
}

impl std::ops::BitOr for CoverageFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for CoverageFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// One cost *query* result: the numeric [`Metrics4`] plus the leaf's
/// [`CoverageFlags`]. Distinct from the bare `Metrics4` stored/blended inside
/// caches (kept lean for cache-line density) — this is what `lookup_metrics`,
/// the CostTree eval buffer, and `aggregate` carry, so coverage rides alongside
/// the numbers without bloating the profiled arrays.
#[derive(Clone, Copy, Debug)]
pub struct LeafMetrics {
    pub m: Metrics4,
    pub coverage: CoverageFlags,
}

impl LeafMetrics {
    pub const ZERO: LeafMetrics = LeafMetrics {
        m: Metrics4::ZERO,
        coverage: CoverageFlags::EMPTY,
    };

    /// Serial composition (a `Sum` node, or the per-request prefill fan-in):
    /// field-wise sum of metrics, union of coverage flags.
    pub fn add(&mut self, other: LeafMetrics) {
        self.m.add_scaled(other.m, 1.0);
        self.coverage |= other.coverage;
    }

    /// Homogeneous-layer fold (`Scale{n}`): scale the metrics; coverage passes
    /// through unchanged (the repeated layer has the same per-leaf coverage).
    pub fn scale(&mut self, factor: f32) {
        self.m.scale(factor);
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
