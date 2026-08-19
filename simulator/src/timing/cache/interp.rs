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
/// the 2D cache a dense `Vec<Metrics4>` grid. Also the `CostTree` eval path's
/// per-leaf / per-subtree unit (the `buf[slot]` values
/// [`CostTree::aggregate`](crate::timing::CostTree) rolls up).
///
/// `repr(C, align(16))` packs the four f32 into exactly one 16-byte SIMD lane so
/// the field-wise `lerp` / `add_scaled` / `scale` blends below autovectorize to
/// `mulps`/`addps` — the 2D bilinear hot path does four of these per lookup.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
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
    #[must_use]
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
    #[must_use]
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

    /// Field-wise `max(0)` copy. `Cache::eval` uses this before returning
    /// [`LeafMetrics`], so extrapolated metrics cannot go negative on the hot
    /// path.
    #[must_use]
    pub fn clamped(self) -> Metrics4 {
        Metrics4 {
            time_ms: self.time_ms.max(0.0),
            flops: self.flops.max(0.0),
            bytes: self.bytes.max(0.0),
            energy_j: self.energy_j.max(0.0),
        }
    }
}

/// The per-leaf coverage signal, packed into a `u8` — one bit per coverage
/// concern. The `CostTree` eval path's allocation-free coverage carrier: it keeps
/// the analytically useful *kind* (did this leaf extrapolate off-grid? was it a
/// JIT/no-coverage placeholder?) as a bitset. `aggregate` ORs these up the tree,
/// so a warning anywhere in a subtree surfaces at its root.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoverageFlags(u8);

impl CoverageFlags {
    pub const EMPTY: Self = Self(0);
    pub const EXTRAPOLATED: Self = Self(1 << 0);
    pub const JIT: Self = Self(1 << 1);
    pub const NO_COVERAGE: Self = Self(1 << 2);

    #[must_use]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[must_use]
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Raw bits, for logging the per-slot coverage as a `u8` column.
    #[must_use]
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
/// [`CoverageFlags`] and the position-local backend that best-of-N selected.
/// Distinct from the bare `Metrics4` stored/blended inside caches (kept lean for
/// cache-line density) — this is what `eval`, the `CostTree` eval buffer, and
/// `aggregate` carry, so coverage and the chosen backend ride alongside the
/// numbers without bloating the profiled arrays.
///
/// `backend_index` is the index into this leaf's ordered candidate backend list
/// (the manifest `LeafDesc.backends`, same order as the kernel's fitted caches),
/// set by [`crate::timing::Probe::eval`]'s best-of-N. [`Self::NO_BACKEND`] marks
/// a value that is not a selected leaf: a placeholder cache miss, an aggregated
/// (`Sum`/`Max`/`Scale`) node, or a leaf not executed this iteration.
#[derive(Clone, Copy, Debug)]
pub struct LeafMetrics {
    pub m: Metrics4,
    pub coverage: CoverageFlags,
    pub backend_index: u8,
}

impl LeafMetrics {
    /// Sentinel `backend_index`: no selected backend (placeholder / aggregate /
    /// leaf not executed this iteration). Reserves the top `u8`, so a position may
    /// carry up to 255 candidate backends (validated at build).
    pub const NO_BACKEND: u8 = u8::MAX;

    pub const ZERO: LeafMetrics = LeafMetrics {
        m: Metrics4::ZERO,
        coverage: CoverageFlags::EMPTY,
        backend_index: Self::NO_BACKEND,
    };

    /// Cache-miss / empty placeholder: zero metrics flagged
    /// [`CoverageFlags::NO_COVERAGE`] so a 0-time result can't pass silently as a
    /// real measurement, and [`Self::NO_BACKEND`] (no leaf was selected).
    pub const MISS: LeafMetrics = LeafMetrics {
        m: Metrics4::ZERO,
        coverage: CoverageFlags::NO_COVERAGE,
        backend_index: Self::NO_BACKEND,
    };

    /// Serial composition of aggregate NODES (a `Sum` / `Max` fold in
    /// [`crate::timing::CostTree::aggregate`]): field-wise sum of metrics, union of
    /// coverage. An aggregate node is not a single leaf's backend selection, so
    /// `backend_index` stays [`Self::NO_BACKEND`] — use [`Self::add_fanin`] for the
    /// leaf fan-in that must carry the selected backend into its logged slot.
    pub fn add(&mut self, other: LeafMetrics) {
        self.m.add_scaled(other.m, 1.0);
        self.coverage |= other.coverage;
    }

    /// Fan-in of per-request LEAF evals into ONE aggregating leaf slot — the
    /// attention prefill slot sums `prefill.eval` over each `(prefix, append)`
    /// request. Like [`Self::add`], but also adopts the first executed
    /// contribution's `backend_index`, so the logged `slot_backend` records which
    /// backend best-of-N selected rather than the [`Self::NO_BACKEND`] sentinel the
    /// `ZERO` accumulator starts at. The v1 attention model has ≤1 prefill request
    /// per step in the common continuous-batching case, so first-wins is exact;
    /// a rare multi-prefill step records the first request's pick (a single `u8`
    /// slot can hold only one).
    pub fn add_fanin(&mut self, other: LeafMetrics) {
        self.add(other);
        if self.backend_index == Self::NO_BACKEND {
            self.backend_index = other.backend_index;
        }
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
///
/// This cmov binary search is the measured floor — three alternatives were
/// benched against it on the real attention grids and all lost, so don't
/// re-try them: (1) a branchless `count`-of-`<= x` scan (hoping to
/// autovectorize) ran ~4× slower on the 63-point axis and +40% even on the
/// 9–18-point axes; (2) an early-exit forward linear probe ran ~2× slower on
/// 63 points and ~4% slower on the small axes (branch mispredicts); (3) an
/// interleaved two-axis `locate2` (overlapping the two searches' load chains)
/// was neutral-to-worse — the compiler already overlaps the two scalar calls.
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

#[cfg(test)]
mod tests {
    use super::{CoverageFlags, LeafMetrics, Metrics4};

    fn leaf(time_ms: f32, backend: u8) -> LeafMetrics {
        LeafMetrics {
            m: Metrics4 {
                time_ms,
                flops: 0.0,
                bytes: 0.0,
                energy_j: 0.0,
            },
            coverage: CoverageFlags::EMPTY,
            backend_index: backend,
        }
    }

    #[test]
    fn add_fanin_adopts_first_executed_backend() {
        // The prefill fan-in starts at ZERO (NO_BACKEND) and accumulates per-request
        // leaf evals; the aggregated slot must carry the selected backend.
        let mut acc = LeafMetrics::ZERO;
        assert_eq!(acc.backend_index, LeafMetrics::NO_BACKEND);
        acc.add_fanin(leaf(2.0, 1)); // first request picked candidate 1
        assert_eq!(acc.backend_index, 1);
        assert_eq!(acc.m.time_ms, 2.0);
        // A second request picking a different backend does not overwrite (a single
        // u8 slot holds one; first-executed wins), but its time still sums in.
        acc.add_fanin(leaf(3.0, 0));
        assert_eq!(acc.backend_index, 1);
        assert_eq!(acc.m.time_ms, 5.0);
    }

    #[test]
    fn plain_add_stays_backendless() {
        // `add` is the aggregate-node fold (Sum/Max); it must NOT adopt a backend —
        // an internal node is not a single leaf's selection.
        let mut acc = LeafMetrics::ZERO;
        acc.add(leaf(2.0, 1));
        assert_eq!(acc.backend_index, LeafMetrics::NO_BACKEND);
        assert_eq!(acc.m.time_ms, 2.0);
    }
}
