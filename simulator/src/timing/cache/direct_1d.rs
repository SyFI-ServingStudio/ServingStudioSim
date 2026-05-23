use std::sync::{Arc, LazyLock};

use crate::common::time::Time;
use crate::timing::bridge::KernelMetrics;
use crate::timing::cache::interp::{CoverageFlags, LeafMetrics, Metrics4, MONOTONICITY_TOLERANCE};
use crate::timing::cache::{Cache, OutlierKind, OutlierWarning};
use crate::timing::sweep::SweepGrid;
use crate::timing::{CoverageKind, CoverageWarning, LookupResult};

/// Interned leaf name; `Kernel::lookup` overwrites it. See `linear_1d.rs`.
static CACHE_NAME: LazyLock<Arc<str>> = LazyLock::new(|| Arc::from("cache_1d_direct"));

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

    fn lookup(&self, sweep: &[f64]) -> LookupResult {
        assert_eq!(
            sweep.len(),
            1,
            "Cache1DDirect lookup requires one coordinate"
        );
        let x = sweep[0];
        if x.is_nan() {
            return LookupResult::leaf(
                Arc::clone(&CACHE_NAME),
                Time::ZERO,
                0,
                0,
                0.0,
                vec![CoverageWarning {
                    kind: CoverageKind::NoCoverage,
                    detail: "x is NaN; returning zero (not a real measurement)".to_string(),
                }],
            );
        }
        let (idx, scale, outside) = self.index(x as f32);
        match self.buckets[idx] {
            Some(m) => {
                let warnings = if outside {
                    let detail = if scale != 1.0 {
                        format!("x={x} beyond right edge, linearly extrapolated from rightmost bucket")
                    } else {
                        format!("x={x} below 1D direct-cache range, clamped to first bucket")
                    };
                    vec![CoverageWarning {
                        kind: CoverageKind::Extrapolated,
                        detail,
                    }]
                } else {
                    Vec::new()
                };
                LookupResult::leaf(
                    Arc::clone(&CACHE_NAME),
                    Time::from_ms((m.time_ms * scale).max(0.0) as f64),
                    (m.flops * scale).max(0.0) as u64,
                    (m.bytes * scale).max(0.0) as u64,
                    (m.energy_j * scale).max(0.0) as f64,
                    warnings,
                )
            }
            // The landed bucket's fit-time sample was dropped as non-finite.
            None => LookupResult::leaf(
                Arc::clone(&CACHE_NAME),
                Time::ZERO,
                0,
                0,
                0.0,
                vec![CoverageWarning {
                    kind: CoverageKind::NoCoverage,
                    detail: format!(
                        "x={x}: direct-cache bucket has no finite sample; returning zero (not a real measurement)"
                    ),
                }],
            ),
        }
    }

    fn lookup_time(&self, sweep: &[f64]) -> Time {
        assert_eq!(
            sweep.len(),
            1,
            "Cache1DDirect lookup requires one coordinate"
        );
        if sweep[0].is_nan() {
            return Time::ZERO;
        }
        let (idx, scale, _) = self.index(sweep[0] as f32);
        match self.buckets[idx] {
            Some(m) => Time::from_ms((m.time_ms * scale).max(0.0) as f64),
            None => Time::ZERO,
        }
    }

    fn lookup_metrics(&self, sweep: &[f64]) -> LeafMetrics {
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
    use crate::timing::cache::{Cache, Cache1DDirect, OutlierKind};
    use crate::timing::sweep::{Axis, SweepGrid};
    use crate::timing::CoverageKind;

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
        // Time-newtype plumbing that `lookup_time` adds on top.
        run("baseline (return x)", &|x| x);
        run("index() only (arith + clamp)", &|x| cache.index(x).0 as f32);
        run("index() + bucket load (Option match)", &|x| {
            let (idx, scale, _) = cache.index(x);
            match cache.buckets[idx] {
                Some(m) => m.time_ms * scale,
                None => 0.0,
            }
        });
        run("lookup_time (+ &[f64] slice + Time)", &|x| {
            cache.lookup_time(&[x as f64]).as_ms() as f32
        });
        run("lookup (+ full LookupResult + Arc)", &|x| {
            cache.lookup(&[x as f64]).time.as_ms() as f32
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
        assert_eq!(cache.lookup(&[0.0]).time.as_ms(), 1.0);
        assert_eq!(cache.lookup(&[128.0]).time.as_ms(), 3.0);
        // Between buckets: floor(x/64). 100→bucket1 (x=64)=2; 191→bucket2=3.
        assert_eq!(cache.lookup(&[100.0]).time.as_ms(), 2.0);
        assert_eq!(cache.lookup(&[191.0]).time.as_ms(), 3.0);
        // No interpolation: 192→bucket3=4 exactly.
        assert_eq!(cache.lookup(&[192.0]).time.as_ms(), 4.0);
        assert!(cache.lookup(&[100.0]).warnings.is_empty());
    }

    #[test]
    fn out_of_range_extrapolates_right_and_clamps_left() {
        let (grid, samples) = grid_64();
        let (cache, _) = Cache1DDirect::from_samples(&grid, &samples);

        // Right edge: x_right=256, rightmost time=5.0. Linear-through-origin:
        // 512 is 2× the max coordinate → 2× its value = 10.0.
        let double = cache.lookup(&[512.0]);
        assert_eq!(double.time.as_ms(), 10.0);
        assert_eq!(double.warnings[0].kind, CoverageKind::Extrapolated);

        // Same scaling at an arbitrary far point: 9999/256 * 5.0.
        let high = cache.lookup(&[9999.0]);
        assert!((high.time.as_ms() - (9999.0 / 256.0 * 5.0)).abs() < 0.05);
        assert_eq!(high.warnings[0].kind, CoverageKind::Extrapolated);

        // Left edge still clamps to the first bucket (no slope through origin
        // below `start`).
        let low = cache.lookup(&[-50.0]);
        assert_eq!(low.time.as_ms(), 1.0);
        assert_eq!(low.warnings[0].kind, CoverageKind::Extrapolated);
    }

    #[test]
    fn right_edge_extrapolation_uses_ratio_for_negative_right_edge() {
        let grid = SweepGrid::new(vec![vec![-128.0, -64.0]]);
        let (cache, _) =
            Cache1DDirect::from_samples(&grid, &[finite_sample(2.0), finite_sample(4.0)]);

        // x_right=-64 and x=-32 is beyond the right edge. The documented
        // through-origin scale is x / x_right = 0.5, so the rightmost time 4.0
        // scales down to 2.0 instead of silently clamping at 4.0.
        let result = cache.lookup(&[-32.0]);
        assert_eq!(result.time.as_ms(), 2.0);
        assert_eq!(cache.lookup_time(&[-32.0]), result.time);
        assert_eq!(result.warnings[0].kind, CoverageKind::Extrapolated);
    }

    #[test]
    fn nan_coordinate_returns_no_coverage_zero() {
        let (grid, samples) = grid_64();
        let (cache, _) = Cache1DDirect::from_samples(&grid, &samples);

        let result = cache.lookup(&[f64::NAN]);
        assert_eq!(result.time.as_ms(), 0.0);
        assert_eq!(result.warnings[0].kind, CoverageKind::NoCoverage);
        assert_eq!(cache.lookup_time(&[f64::NAN]), result.time);
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
        let dropped = cache.lookup(&[80.0]);
        assert_eq!(dropped.time.as_ms(), 0.0);
        assert_eq!(dropped.warnings[0].kind, CoverageKind::NoCoverage);
        // Neighboring live buckets still resolve.
        assert_eq!(cache.lookup(&[128.0]).time.as_ms(), 3.0);
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

    #[test]
    fn lookup_time_matches_lookup_time_field() {
        let (grid, samples) = grid_64();
        let (cache, _) = Cache1DDirect::from_samples(&grid, &samples);
        for x in [0.0, 80.0, 128.0, 250.0, 9999.0, -10.0] {
            assert_eq!(cache.lookup_time(&[x]), cache.lookup(&[x]).time);
        }
    }
}
