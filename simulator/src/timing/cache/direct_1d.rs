use crate::timing::bridge::KernelMetrics;
use crate::timing::cache::interp::{CoverageFlags, LeafMetrics, Metrics4, MONOTONICITY_TOLERANCE};
use crate::timing::cache::{Cache, OutlierKind, OutlierWarning};
use crate::timing::sweep::SweepGrid;

/// Direct-indexed 1D cache for bounded-range, uniformly-sampled data — e.g. a
/// batch / kv-length axis capped at ~32k with 64-wide spacing (512 buckets).
///
/// Lookup is O(1): `idx = floor((x - start) / spacing)` indexes straight into
/// the bucket array, no binary search. The benchmark showed the search is what
/// `Cache1DLinear` lookups are bound on, so this is the fast path for hot,
/// bounded sweeps where the trade-offs are acceptable:
/// - **nearest bucket, not interpolated**: the result is the floor bucket's
///   value, so x carries up to `spacing`-wide quantization;
/// - **right-edge extrapolation is linear-through-origin**: beyond the last
///   bucket the result is `(x / x_right) * rightmost_value` (2× the max
///   coordinate ⇒ 2× its metrics), flagged `Extrapolated`. Below `start` it
///   clamps to the first bucket (also flagged). This cache is *for* bounded data.
///
/// A bucket whose fit-time sample was non-finite is `None` and a lookup landing
/// on it returns `NoCoverage` (same loud-zero policy as the linear caches).
#[derive(Clone, Debug)]
pub struct Cache1DDirect {
    start: f32,
    spacing: f32,
    buckets: Vec<Option<Metrics4>>,
}

impl Cache for Cache1DDirect {
    fn from_samples(grid: &SweepGrid, samples: &[KernelMetrics]) -> (Self, Vec<OutlierWarning>) {
        assert_eq!(grid.axes().len(), 1, "Cache1DDirect requires a 1D grid");
        let axis = &grid.axes()[0];
        assert_eq!(
            axis.len(),
            samples.len(),
            "sample count must match 1D sweep grid length"
        );
        assert!(
            axis.len() >= 2,
            "Cache1DDirect needs >=2 grid points to define a spacing"
        );

        let start = axis[0] as f32;
        let spacing = (axis[1] - axis[0]) as f32;
        assert!(
            spacing > 0.0,
            "Cache1DDirect grid must be strictly ascending"
        );
        // Direct indexing assumes `axis[i] == start + i*spacing`. Enforce the
        // uniform-spacing contract so a misconfigured grid fails loudly at build
        // rather than silently returning the wrong bucket at lookup.
        assert!(
            axis.windows(2)
                .all(|w| ((w[1] - w[0]) as f32 - spacing).abs() <= spacing * 1e-3),
            "Cache1DDirect requires a uniformly-spaced grid (got non-uniform axis)"
        );

        let mut warnings = Vec::new();
        let mut buckets: Vec<Option<Metrics4>> = Vec::with_capacity(samples.len());
        for (&x, sample) in axis.iter().zip(samples.iter()) {
            if !sample.is_finite() {
                warnings.push(OutlierWarning {
                    kind: OutlierKind::NonFinite,
                    detail: format!("x={x}: sample has non-finite or negative metric; dropped"),
                });
                buckets.push(None);
                continue;
            }
            buckets.push(Some(Metrics4::from_sample(sample)));
        }

        // Monotonicity scan over present buckets: time should be non-decreasing
        // as x grows. Same 10% tolerance as the linear caches.
        let mut prev: Option<(f32, f32)> = None;
        for (i, bucket) in buckets.iter().enumerate() {
            if let Some(m) = bucket {
                let x = start + i as f32 * spacing;
                if let Some((px, ptime)) = prev {
                    if m.time_ms < ptime * (1.0 - MONOTONICITY_TOLERANCE) {
                        warnings.push(OutlierWarning {
                            kind: OutlierKind::MonotonicityBreak,
                            detail: format!(
                                "time {:.4}ms (x={px}) → {:.4}ms (x={x}) drops >{:.0}%, non-monotonic",
                                ptime,
                                m.time_ms,
                                MONOTONICITY_TOLERANCE * 100.0
                            ),
                        });
                    }
                }
                prev = Some((x, m.time_ms));
            }
        }

        (
            Self {
                start,
                spacing,
                buckets,
            },
            warnings,
        )
    }

    fn eval(&self, sweep: &[f64]) -> LeafMetrics {
        assert_eq!(
            sweep.len(),
            1,
            "Cache1DDirect lookup requires one coordinate"
        );
        if sweep[0].is_nan() {
            return LeafMetrics {
                m: Metrics4::ZERO,
                coverage: CoverageFlags::NO_COVERAGE,
            };
        }
        let (idx, scale, outside) = self.index(sweep[0] as f32);
        match self.buckets[idx] {
            Some(mut m) => {
                m.scale(scale);
                LeafMetrics {
                    m: m.clamped(),
                    coverage: if outside {
                        CoverageFlags::EXTRAPOLATED
                    } else {
                        CoverageFlags::EMPTY
                    },
                }
            }
            // Landed bucket's fit-time sample was dropped as non-finite.
            None => LeafMetrics {
                m: Metrics4::ZERO,
                coverage: CoverageFlags::NO_COVERAGE,
            },
        }
    }
}

impl Cache1DDirect {
    /// Floor-index `x` into the bucket array. Returns `(idx, scale, outside)`:
    /// the bucket to read, a multiplicative `scale` on its metrics, and whether
    /// `x` fell outside `[start, x_right]` (flagged `Extrapolated`).
    ///
    /// - In range → `(floor_idx, 1.0, false)`.
    /// - Beyond the right edge (`x > x_right`) → `(last, x / x_right, true)`: a
    ///   linear-through-origin extrapolation off the rightmost bucket, so 2× the
    ///   max coordinate yields 2× its value. Guarded against a near-zero
    ///   `x_right` (singular grid for this formula) by falling back to
    ///   `scale = 1.0`.
    /// - Below `start` → `(0, 1.0, true)`: clamped to the first bucket. The
    ///   float→int `as` cast saturates negative values to 0, so no explicit
    ///   low-side branch is needed for the index itself. Public callers handle
    ///   NaN before invoking this helper.
    #[inline]
    fn index(&self, x: f32) -> (usize, f32, bool) {
        let last = self.buckets.len() - 1;
        let x_right = self.start + last as f32 * self.spacing;
        if x > x_right {
            let scale = if x_right.abs() > f32::EPSILON {
                x / x_right
            } else {
                1.0
            };
            return (last, scale, true);
        }
        let idx = (((x - self.start) / self.spacing) as usize).min(last);
        (idx, 1.0, x < self.start)
    }
}

#[cfg(test)]
mod tests {
    use crate::timing::bridge::KernelMetrics;
    use crate::timing::cache::interp::CoverageFlags;
    use crate::timing::cache::{Cache, Cache1DDirect, OutlierKind};
    use crate::timing::sweep::{Axis, SweepGrid};

    #[test]
    #[ignore = "internal microbench; run: cargo test --release --lib component_breakdown_1d_direct -- --ignored --nocapture"]
    fn component_breakdown_1d_direct() {
        use std::hint::black_box;
        use std::time::Instant;

        // 512 buckets, spacing 64, range [0, 32704] — the bounded batch/kv LUT.
        let axis = Axis::arithmetic(0, 64 * 511, 64);
        assert_eq!(axis.len(), 512);
        let grid = SweepGrid::new(vec![axis.clone()]);
        let samples: Vec<KernelMetrics> = (0..axis.len())
            .map(|idx| finite_sample(1.0 + idx as f64 * 0.5))
            .collect();
        let (cache, warnings) = Cache1DDirect::from_samples(&grid, &samples);
        assert!(warnings.is_empty());

        // Interior probes plus one right-edge extrapolation.
        let probes = [100.0f32, 3000.0, 12345.0, 20000.0, 32000.0, 99999.0];
        let run = |label: &str, f: &dyn Fn(f32) -> f32| {
            for &x in &probes {
                black_box(f(black_box(x)));
            }
            let reps = 5_000_000usize / probes.len();
            let mut acc = 0.0f32;
            let start = Instant::now();
            for _ in 0..reps {
                for &x in &probes {
                    acc += f(black_box(x));
                }
            }
            black_box(acc);
            let ns = start.elapsed().as_nanos() as f64 / (reps * probes.len()) as f64;
            println!("{label:<40} {ns:>7.2} ns");
        };

        // Each layer adds exactly one component over the previous, isolating where
        // the ~12ns goes: arithmetic vs. the dependent heap load vs. the slice +
        // LeafMetrics plumbing that `eval` adds on top.
        run("baseline (return x)", &|x| x);
        run("index() only (arith + clamp)", &|x| cache.index(x).0 as f32);
        run("index() + bucket load (Option match)", &|x| {
            let (idx, scale, _) = cache.index(x);
            match cache.buckets[idx] {
                Some(m) => m.time_ms * scale,
                None => 0.0,
            }
        });
        run("eval (+ &[f64] slice + LeafMetrics)", &|x| {
            cache.eval(&[x as f64]).m.time_ms
        });
    }

    fn finite_sample(time_ms: f64) -> KernelMetrics {
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

    /// Grid 0,64,128,192,256 with times 1,2,3,4,5.
    fn grid_64() -> (SweepGrid, Vec<KernelMetrics>) {
        let grid = SweepGrid::new(vec![Axis::arithmetic(0, 256, 64)]);
        let samples = (1..=5).map(|t| finite_sample(t as f64)).collect();
        (grid, samples)
    }

    #[test]
    fn floor_indexes_to_the_containing_bucket() {
        let (grid, samples) = grid_64();
        let (cache, warnings) = Cache1DDirect::from_samples(&grid, &samples);
        assert!(warnings.is_empty());

        // Exact grid points return their bucket.
        assert_eq!(cache.eval(&[0.0]).m.time_ms, 1.0);
        assert_eq!(cache.eval(&[128.0]).m.time_ms, 3.0);
        // Between buckets: floor(x/64). 100→bucket1 (x=64)=2; 191→bucket2=3.
        assert_eq!(cache.eval(&[100.0]).m.time_ms, 2.0);
        assert_eq!(cache.eval(&[191.0]).m.time_ms, 3.0);
        // No interpolation: 192→bucket3=4 exactly.
        assert_eq!(cache.eval(&[192.0]).m.time_ms, 4.0);
        assert!(cache.eval(&[100.0]).coverage.is_empty());
    }

    #[test]
    fn out_of_range_extrapolates_right_and_clamps_left() {
        let (grid, samples) = grid_64();
        let (cache, _) = Cache1DDirect::from_samples(&grid, &samples);

        // Right edge: x_right=256, rightmost time=5.0. Linear-through-origin:
        // 512 is 2× the max coordinate → 2× its value = 10.0.
        let double = cache.eval(&[512.0]);
        assert_eq!(double.m.time_ms, 10.0);
        assert!(double.coverage.contains(CoverageFlags::EXTRAPOLATED));

        // Same scaling at an arbitrary far point: 9999/256 * 5.0.
        let high = cache.eval(&[9999.0]);
        assert!((high.m.time_ms - (9999.0 / 256.0 * 5.0)).abs() < 0.05);
        assert!(high.coverage.contains(CoverageFlags::EXTRAPOLATED));

        // Left edge still clamps to the first bucket (no slope through origin
        // below `start`).
        let low = cache.eval(&[-50.0]);
        assert_eq!(low.m.time_ms, 1.0);
        assert!(low.coverage.contains(CoverageFlags::EXTRAPOLATED));
    }

    #[test]
    fn right_edge_extrapolation_uses_ratio_for_negative_right_edge() {
        let grid = SweepGrid::new(vec![vec![-128.0, -64.0]]);
        let (cache, _) =
            Cache1DDirect::from_samples(&grid, &[finite_sample(2.0), finite_sample(4.0)]);

        // x_right=-64 and x=-32 is beyond the right edge. The documented
        // through-origin scale is x / x_right = 0.5, so the rightmost time 4.0
        // scales down to 2.0 instead of silently clamping at 4.0.
        let result = cache.eval(&[-32.0]);
        assert_eq!(result.m.time_ms, 2.0);
        assert!(result.coverage.contains(CoverageFlags::EXTRAPOLATED));
    }

    #[test]
    fn nan_coordinate_returns_no_coverage_zero() {
        let (grid, samples) = grid_64();
        let (cache, _) = Cache1DDirect::from_samples(&grid, &samples);

        let result = cache.eval(&[f64::NAN]);
        assert_eq!(result.m.time_ms, 0.0);
        assert!(result.coverage.contains(CoverageFlags::NO_COVERAGE));
    }

    #[test]
    fn dropped_bucket_returns_no_coverage() {
        let grid = SweepGrid::new(vec![Axis::arithmetic(0, 192, 64)]); // 0,64,128,192
        let mut nan = finite_sample(0.0);
        nan.time_ms = f64::NAN;
        let samples = vec![
            finite_sample(1.0),
            nan,
            finite_sample(3.0),
            finite_sample(4.0),
        ];
        let (cache, warnings) = Cache1DDirect::from_samples(&grid, &samples);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].kind, OutlierKind::NonFinite);

        // Landing on the dropped bucket (x=64) → NoCoverage, not a silent 0.
        let dropped = cache.eval(&[80.0]);
        assert_eq!(dropped.m.time_ms, 0.0);
        assert!(dropped.coverage.contains(CoverageFlags::NO_COVERAGE));
        // Neighboring live buckets still resolve.
        assert_eq!(cache.eval(&[128.0]).m.time_ms, 3.0);
    }

    #[test]
    fn non_monotonic_time_flags_break() {
        let grid = SweepGrid::new(vec![Axis::arithmetic(0, 128, 64)]);
        let (_cache, warnings) = Cache1DDirect::from_samples(
            &grid,
            &[finite_sample(1.0), finite_sample(0.4), finite_sample(2.0)],
        );
        assert!(warnings
            .iter()
            .any(|w| w.kind == OutlierKind::MonotonicityBreak));
    }
}
