//! Sweep grid + axis helpers for L1 kernel profiling.
//!
//! Per-kernel files compose their sweep with `Axis::*` primitives or named
//! domain presets like `Axis::token_axis()`. The presets are profiler↔simulator
//! contract constants: changing a preset's shape requires the Python profiler
//! to resample the same points, so do it deliberately.

#[derive(Clone, Debug, PartialEq)]
pub struct SweepGrid {
    axes: Vec<Vec<f64>>,
}

impl SweepGrid {
    pub fn new(axes: Vec<Vec<f64>>) -> Self {
        assert!(!axes.is_empty(), "sweep grid must have at least one axis");
        assert!(
            axes.iter().all(|axis| !axis.is_empty()),
            "sweep axes must not be empty"
        );
        Self { axes }
    }

    pub fn axes(&self) -> &[Vec<f64>] {
        &self.axes
    }

    pub fn expand_1d<T>(&self, mut f: impl FnMut(f64) -> T) -> Vec<T> {
        assert_eq!(self.axes.len(), 1, "expand_1d requires a 1D sweep grid");
        self.axes[0].iter().copied().map(&mut f).collect()
    }

    pub fn expand_2d<T>(&self, mut f: impl FnMut(f64, f64) -> T) -> Vec<T> {
        assert_eq!(self.axes.len(), 2, "expand_2d requires a 2D sweep grid");
        let mut out = Vec::with_capacity(self.axes[0].len() * self.axes[1].len());
        for &a in &self.axes[0] {
            for &b in &self.axes[1] {
                out.push(f(a, b));
            }
        }
        out
    }

    pub fn expand_3d<T>(&self, mut f: impl FnMut(f64, f64, f64) -> T) -> Vec<T> {
        assert_eq!(self.axes.len(), 3, "expand_3d requires a 3D sweep grid");
        let mut out =
            Vec::with_capacity(self.axes[0].len() * self.axes[1].len() * self.axes[2].len());
        for &a in &self.axes[0] {
            for &b in &self.axes[1] {
                for &c in &self.axes[2] {
                    out.push(f(a, b, c));
                }
            }
        }
        out
    }

    /// N-D escape hatch. Calls `f` with a slice of length `axes.len()` for every
    /// point in the cartesian product, row-major over the axes.
    pub fn expand<T>(&self, mut f: impl FnMut(&[f64]) -> T) -> Vec<T> {
        let total: usize = self.axes.iter().map(Vec::len).product();
        let mut out = Vec::with_capacity(total);
        let mut coords = vec![0.0; self.axes.len()];
        expand_recursive(&self.axes, 0, &mut coords, &mut out, &mut f);
        out
    }
}

fn expand_recursive<T>(
    axes: &[Vec<f64>],
    depth: usize,
    coords: &mut [f64],
    out: &mut Vec<T>,
    f: &mut impl FnMut(&[f64]) -> T,
) {
    if depth == axes.len() {
        out.push(f(coords));
        return;
    }
    for &v in &axes[depth] {
        coords[depth] = v;
        expand_recursive(axes, depth + 1, coords, out, f);
    }
}

/// Upper bound on a kernel's sweep dimensionality. `Coords` is a fixed inline
/// buffer of this width so `Input::coords()` projects to sweep space without
/// heap allocation. Bump this if a kernel ever needs >4 sweep axes (today the
/// widest is 3D); `Coords::new` asserts the actual arity fits.
pub const MAX_SWEEP_DIMS: usize = 4;

/// Stack-allocated sweep coordinate vector. Holds up to `MAX_SWEEP_DIMS`
/// `f64`s inline and derefs to `&[f64]`, so the hot-path `Input::coords()`
/// projection never touches the heap. Built via `Coords::new([..])` (the
/// `#[derive(SweepCoords)]` output) or `Coords::from_slice`.
#[derive(Clone, Copy, Debug)]
pub struct Coords {
    buf: [f64; MAX_SWEEP_DIMS],
    len: usize,
}

impl Coords {
    /// Build from a fixed-size array of coordinates (the derive emits this).
    /// `N` is the kernel's sweep arity; it must fit `MAX_SWEEP_DIMS`.
    pub fn new<const N: usize>(values: [f64; N]) -> Self {
        assert!(
            N <= MAX_SWEEP_DIMS,
            "sweep arity {N} exceeds MAX_SWEEP_DIMS {MAX_SWEEP_DIMS}"
        );
        let mut buf = [0.0; MAX_SWEEP_DIMS];
        buf[..N].copy_from_slice(&values);
        Self { buf, len: N }
    }

    /// Build from a slice (for manual `SweepCoords` impls over ragged inputs).
    pub fn from_slice(values: &[f64]) -> Self {
        assert!(
            values.len() <= MAX_SWEEP_DIMS,
            "sweep arity {} exceeds MAX_SWEEP_DIMS {MAX_SWEEP_DIMS}",
            values.len()
        );
        let mut buf = [0.0; MAX_SWEEP_DIMS];
        buf[..values.len()].copy_from_slice(values);
        Self {
            buf,
            len: values.len(),
        }
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.buf[..self.len]
    }
}

impl std::ops::Deref for Coords {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        self.as_slice()
    }
}

/// Flatten a kernel `Input` struct into sweep-space coordinates. Usually
/// `#[derive(SweepCoords)]` on the Input struct (numeric fields, in declaration
/// order, each `as f64`). Implement manually for inputs containing non-scalar
/// fields (e.g. ragged-attention seq-length summaries), returning
/// `Coords::from_slice(..)`.
pub trait SweepCoords {
    fn coords(&self) -> Coords;

    /// The Input field names in `coords()` order. Lets a `kernel-query` `grid`
    /// response label each grid axis with the input key it sweeps. Generated by
    /// `#[derive(SweepCoords)]`; a manual impl returning the same order as
    /// `coords()` keeps the two aligned.
    fn coord_field_names() -> &'static [&'static str]
    where
        Self: Sized;
}

/// Axis builders. Use named domain presets (`Axis::token_axis()`) for curves
/// shared across kernels; compose primitives (`pow2 / arithmetic / values /
/// chain`) for custom shapes; pass raw `Vec<f64>` to `SweepGrid::new` for the
/// fully bespoke case.
pub struct Axis;

impl Axis {
    /// `[2^min_log2, 2^(min_log2+1), ..., 2^max_log2]`, both endpoints included.
    pub fn pow2(min_log2: u32, max_log2: u32) -> Vec<f64> {
        assert!(
            min_log2 <= max_log2,
            "Axis::pow2: min_log2 must be <= max_log2"
        );
        (min_log2..=max_log2).map(|i| (1u64 << i) as f64).collect()
    }

    /// Arithmetic progression `start, start+step, ...` up to and including any
    /// term <= `end`.
    pub fn arithmetic(start: u32, end: u32, step: u32) -> Vec<f64> {
        assert!(step > 0, "Axis::arithmetic: step must be > 0");
        assert!(start <= end, "Axis::arithmetic: start must be <= end");
        let mut out = Vec::new();
        let mut v = start;
        while v <= end {
            out.push(v as f64);
            match v.checked_add(step) {
                Some(next) => v = next,
                None => break,
            }
        }
        out
    }

    /// Hand-picked values; preserves declaration order.
    pub fn values(vs: impl IntoIterator<Item = u32>) -> Vec<f64> {
        vs.into_iter().map(|v| v as f64).collect()
    }

    /// Concatenate segments, deduplicating any value already emitted earlier.
    /// First-seen order is preserved, so segment seams (`segment_a` ends at the
    /// same value `segment_b` starts at) collapse cleanly.
    pub fn chain(segments: impl IntoIterator<Item = Vec<f64>>) -> Vec<f64> {
        let mut seen = std::collections::HashSet::<u64>::new();
        let mut out = Vec::new();
        for segment in segments {
            for v in segment {
                if seen.insert(v.to_bits()) {
                    out.push(v);
                }
            }
        }
        out
    }

    /// Shared "token-axis" curve. Used by GEMM-M, RmsNorm-M, AllReduce
    /// message_size, attention seq_len, etc.
    ///
    /// Curve shape (63 strictly-increasing points, dedupe-on-seam):
    ///   - `[32, 64, 128, 256]`             pow2 doublings
    ///   - `256..=4096` step 128            dense mid-range
    ///   - `4096..=8192` step 256
    ///   - `8192..=16384` step 1024
    ///   - `16384..=32768` step 4096
    ///   - `[65536]`                        long-context cap
    ///
    /// Profiler-side contract: every value in this curve must be a row in the
    /// corresponding Python `perf_api` table. Adding or removing points here
    /// requires the Python profiler to resample, otherwise `Kernel::build` will
    /// fail with `BuildError::MissingEntry`. Build-cache-only paths can
    /// `enable_jit_profiling()` to fill missing rows on demand.
    pub fn token_axis() -> Vec<f64> {
        Self::chain([
            Self::pow2(5, 8),
            Self::arithmetic(256, 4096, 128),
            Self::arithmetic(4096, 8192, 256),
            Self::arithmetic(8192, 16384, 1024),
            Self::arithmetic(16384, 32768, 4096),
            Self::values([65536]),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_pow2_inclusive_endpoints() {
        assert_eq!(Axis::pow2(0, 3), vec![1.0, 2.0, 4.0, 8.0]);
        assert_eq!(Axis::pow2(5, 5), vec![32.0]);
    }

    #[test]
    fn axis_arithmetic_includes_endpoint_when_step_divides() {
        assert_eq!(Axis::arithmetic(0, 9, 3), vec![0.0, 3.0, 6.0, 9.0]);
        assert_eq!(Axis::arithmetic(2, 8, 2), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn axis_chain_dedupes_seam_values() {
        let chained = Axis::chain([vec![1.0, 2.0, 4.0], vec![4.0, 8.0, 16.0]]);
        assert_eq!(chained, vec![1.0, 2.0, 4.0, 8.0, 16.0]);
    }

    #[test]
    fn axis_token_axis_progressive_curve_shape() {
        let curve = Axis::token_axis();
        // Endpoints.
        assert_eq!(curve.first().copied(), Some(32.0));
        assert_eq!(curve.last().copied(), Some(65536.0));
        // Strictly increasing — segments are sorted ascending and Axis::chain
        // dedupes seam values, so no equal-adjacent pairs survive.
        for window in curve.windows(2) {
            assert!(
                window[0] < window[1],
                "token_axis must be strictly increasing, found {} >= {}",
                window[0],
                window[1]
            );
        }
        // Segment seams each appear exactly once (post-dedupe).
        for seam in [256.0, 4096.0, 8192.0, 16384.0] {
            assert_eq!(
                curve.iter().filter(|&&v| v == seam).count(),
                1,
                "seam value {seam} must appear exactly once",
            );
        }
        // 4 + 31 + 17 + 9 + 5 + 1 = 67 raw points, minus 4 seam dedupes = 63.
        assert_eq!(curve.len(), 63);
    }

    #[test]
    fn expand_2d_cartesian_product_row_major() {
        let grid = SweepGrid::new(vec![vec![1.0, 2.0], vec![10.0, 20.0, 30.0]]);
        let out: Vec<(f64, f64)> = grid.expand_2d(|a, b| (a, b));
        assert_eq!(
            out,
            vec![
                (1.0, 10.0),
                (1.0, 20.0),
                (1.0, 30.0),
                (2.0, 10.0),
                (2.0, 20.0),
                (2.0, 30.0),
            ]
        );
    }

    #[test]
    fn expand_3d_cartesian_product_row_major() {
        let grid = SweepGrid::new(vec![vec![1.0, 2.0], vec![10.0, 20.0], vec![100.0, 200.0]]);
        let out: Vec<(f64, f64, f64)> = grid.expand_3d(|a, b, c| (a, b, c));
        assert_eq!(out.len(), 8);
        assert_eq!(out[0], (1.0, 10.0, 100.0));
        assert_eq!(out[1], (1.0, 10.0, 200.0));
        assert_eq!(out.last().copied(), Some((2.0, 20.0, 200.0)));
    }

    #[test]
    fn expand_generic_matches_n_d_cartesian() {
        let grid = SweepGrid::new(vec![vec![1.0, 2.0], vec![10.0, 20.0]]);
        let out: Vec<Vec<f64>> = grid.expand(|coords| coords.to_vec());
        assert_eq!(
            out,
            vec![
                vec![1.0, 10.0],
                vec![1.0, 20.0],
                vec![2.0, 10.0],
                vec![2.0, 20.0],
            ]
        );
    }
}
