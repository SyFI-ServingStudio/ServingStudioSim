use crate::timing::bridge::KernelMetrics;
use crate::timing::cache::interp::{locate, CoverageFlags, LeafMetrics, Metrics4, MONOTONICITY_TOLERANCE};
use crate::timing::cache::{Cache, OutlierKind, OutlierWarning};
use crate::timing::sweep::SweepGrid;

/// Piecewise-linear interpolation over one monotonic axis. Stored struct-of-
/// arrays: `xs` is the dense ascending search key (so the branchless `locate`
/// scans a contiguous f32 slice), `metrics` holds the parallel sample values.
#[derive(Clone, Debug)]
pub struct Cache1DLinear {
    xs: Vec<f32>,
    metrics: Vec<Metrics4>,
}

impl Cache for Cache1DLinear {
    fn from_samples(grid: &SweepGrid, samples: &[KernelMetrics]) -> (Self, Vec<OutlierWarning>) {
        assert_eq!(grid.axes().len(), 1, "Cache1DLinear requires a 1D grid");
        assert_eq!(
            grid.axes()[0].len(),
            samples.len(),
            "sample count must match 1D sweep grid length"
        );
        let mut warnings = Vec::new();
        let mut rows: Vec<(f32, Metrics4)> = Vec::with_capacity(samples.len());
        for (&x, sample) in grid.axes()[0].iter().zip(samples.iter()) {
            if !sample.is_finite() {
                warnings.push(OutlierWarning {
                    kind: OutlierKind::NonFinite,
                    detail: format!("x={x}: sample has non-finite or negative metric; dropped"),
                });
                continue;
            }
            rows.push((x as f32, Metrics4::from_sample(sample)));
        }
        rows.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));

        // Monotonicity scan on the x-sorted finite points: time should be
        // non-decreasing as x grows (bigger problem size costs more). A drop
        // beyond `MONOTONICITY_TOLERANCE` of the previous point flags a
        // `MonotonicityBreak` — a noisy sample or scheduler cliff worth surfacing.
        for pair in rows.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            if hi.1.time_ms < lo.1.time_ms * (1.0 - MONOTONICITY_TOLERANCE) {
                warnings.push(OutlierWarning {
                    kind: OutlierKind::MonotonicityBreak,
                    detail: format!(
                        "time {:.4}ms (x={}) → {:.4}ms (x={}) drops >{:.0}%, non-monotonic",
                        lo.1.time_ms,
                        lo.0,
                        hi.1.time_ms,
                        hi.0,
                        MONOTONICITY_TOLERANCE * 100.0
                    ),
                });
            }
        }

        let xs = rows.iter().map(|row| row.0).collect();
        let metrics = rows.iter().map(|row| row.1).collect();
        (Self { xs, metrics }, warnings)
    }

    fn eval(&self, sweep: &[f64]) -> LeafMetrics {
        assert_eq!(
            sweep.len(),
            1,
            "Cache1DLinear lookup requires one coordinate"
        );
        let x = sweep[0];
        if self.xs.is_empty() || x.is_nan() {
            // Empty cache / NaN coord → zero placeholder, flagged NoCoverage so a
            // 0-time result can't pass silently as a real measurement downstream.
            return LeafMetrics {
                m: Metrics4::ZERO,
                coverage: CoverageFlags::NO_COVERAGE,
            };
        }
        let (m, extrapolated) = self.interpolate(x as f32);
        LeafMetrics {
            m: m.clamped(),
            coverage: if extrapolated {
                CoverageFlags::EXTRAPOLATED
            } else {
                CoverageFlags::EMPTY
            },
        }
    }
}

impl Cache1DLinear {
    /// Locate the bracketing interval for `x` once (shared branchless `locate`),
    /// then lerp all four metrics in a single pass. Returns the interpolated
    /// `Metrics4` plus whether `x` fell outside the grid (edge segments
    /// extrapolate; never clamp). Callers that need only one field still pay one
    /// locate, not four. Precondition: `xs` is non-empty (caller checks).
    fn interpolate(&self, x: f32) -> (Metrics4, bool) {
        let (lo, hi, t, outside) = locate(&self.xs, x);
        (self.metrics[lo].lerp(self.metrics[hi], t), outside)
    }
}

#[cfg(test)]
mod tests {
    use crate::timing::bridge::KernelMetrics;
    use crate::timing::cache::interp::CoverageFlags;
    use crate::timing::cache::{Cache, Cache1DLinear, OutlierKind};
    use crate::timing::sweep::SweepGrid;

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

    #[test]
    fn linear_1d_interpolates_and_extrapolates() {
        let grid = SweepGrid::new(vec![vec![1.0, 2.0]]);
        let samples = vec![
            KernelMetrics {
                time_ms: 1.0,
                tflops: Some(1.0),
                memory_bandwidth_gbps: Some(2.0),
                algbw_gbps: None,
                busbw_gbps: None,
                message_size_bytes: None,
                energy_j: 3.0,
            },
            KernelMetrics {
                time_ms: 3.0,
                tflops: Some(1.0),
                memory_bandwidth_gbps: Some(2.0),
                algbw_gbps: None,
                busbw_gbps: None,
                message_size_bytes: None,
                energy_j: 5.0,
            },
        ];
        let (cache, warnings) = Cache1DLinear::from_samples(&grid, &samples);
        assert!(warnings.is_empty());

        let inside = cache.eval(&[1.5]);
        assert_eq!(inside.m.time_ms, 2.0);
        assert!(inside.coverage.is_empty());

        let outside = cache.eval(&[3.0]);
        assert_eq!(outside.m.time_ms, 5.0);
        assert!(outside.coverage.contains(CoverageFlags::EXTRAPOLATED));
    }

    #[test]
    fn non_finite_samples_become_outlier_warnings_and_are_dropped() {
        let grid = SweepGrid::new(vec![vec![1.0, 2.0, 3.0]]);
        let samples = vec![
            KernelMetrics {
                time_ms: 1.0,
                tflops: Some(1.0),
                memory_bandwidth_gbps: Some(2.0),
                algbw_gbps: None,
                busbw_gbps: None,
                message_size_bytes: None,
                energy_j: 1.0,
            },
            KernelMetrics {
                time_ms: f64::NAN,
                tflops: Some(1.0),
                memory_bandwidth_gbps: Some(2.0),
                algbw_gbps: None,
                busbw_gbps: None,
                message_size_bytes: None,
                energy_j: 2.0,
            },
            KernelMetrics {
                time_ms: 3.0,
                tflops: Some(1.0),
                memory_bandwidth_gbps: Some(2.0),
                algbw_gbps: None,
                busbw_gbps: None,
                message_size_bytes: None,
                energy_j: 3.0,
            },
        ];
        let (cache, warnings) = Cache1DLinear::from_samples(&grid, &samples);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].kind, OutlierKind::NonFinite);

        // The dropped middle point still leaves a 2-point grid that interpolates
        // 1.0→3.0 linearly across x=1..3, so lookup(2.0) ≈ 2.0.
        let mid = cache.eval(&[2.0]);
        assert_eq!(mid.m.time_ms, 2.0);
    }

    #[test]
    fn all_non_finite_samples_make_empty_cache_flagged_no_coverage() {
        let grid = SweepGrid::new(vec![vec![1.0, 2.0]]);
        let samples = vec![
            KernelMetrics {
                time_ms: f64::NAN,
                tflops: Some(1.0),
                memory_bandwidth_gbps: Some(2.0),
                algbw_gbps: None,
                busbw_gbps: None,
                message_size_bytes: None,
                energy_j: 1.0,
            },
            KernelMetrics {
                time_ms: f64::INFINITY,
                tflops: Some(1.0),
                memory_bandwidth_gbps: Some(2.0),
                algbw_gbps: None,
                busbw_gbps: None,
                message_size_bytes: None,
                energy_j: 1.0,
            },
        ];
        let (cache, warnings) = Cache1DLinear::from_samples(&grid, &samples);
        // Both samples dropped at fit time → empty cache.
        assert_eq!(warnings.len(), 2);
        assert!(warnings.iter().all(|w| w.kind == OutlierKind::NonFinite));

        // Lookup on the empty cache returns zero, loudly flagged NoCoverage
        // (not a silent valid-looking 0-time result).
        let result = cache.eval(&[1.5]);
        assert_eq!(result.m.time_ms, 0.0);
        assert_eq!(result.m.flops, 0.0);
        assert!(result.coverage.contains(CoverageFlags::NO_COVERAGE));
    }

    #[test]
    fn non_monotonic_time_flags_monotonicity_break() {
        // x = 1,2,3; times 1.0 → 0.5 → 2.0. The 1.0→0.5 drop (50%) exceeds the 10% tol.
        let grid = SweepGrid::new(vec![vec![1.0, 2.0, 3.0]]);
        let (_cache, warnings) = Cache1DLinear::from_samples(
            &grid,
            &[finite_sample(1.0), finite_sample(0.5), finite_sample(2.0)],
        );
        assert!(warnings
            .iter()
            .any(|w| w.kind == OutlierKind::MonotonicityBreak));
    }

    #[test]
    fn eval_on_empty_cache_is_zero() {
        let grid = SweepGrid::new(vec![vec![1.0, 2.0]]);
        let mut nan = finite_sample(0.0);
        nan.time_ms = f64::NAN;
        let (cache, _) = Cache1DLinear::from_samples(&grid, &[nan.clone(), nan]);
        let leaf = cache.eval(&[1.5]);
        assert_eq!(leaf.m.time_ms, 0.0);
        assert!(leaf.coverage.contains(CoverageFlags::NO_COVERAGE));
    }

    #[test]
    fn nan_coordinate_returns_no_coverage_zero() {
        let grid = SweepGrid::new(vec![vec![1.0, 2.0]]);
        let (cache, _) =
            Cache1DLinear::from_samples(&grid, &[finite_sample(1.0), finite_sample(3.0)]);

        let leaf = cache.eval(&[f64::NAN]);
        assert_eq!(leaf.m.time_ms, 0.0);
        assert!(leaf.coverage.contains(CoverageFlags::NO_COVERAGE));
    }

    #[test]
    fn small_time_dip_within_tolerance_is_not_flagged() {
        // 1.0 → 0.95 is a 5% dip, within the 10% tolerance — no MonotonicityBreak.
        let grid = SweepGrid::new(vec![vec![1.0, 2.0]]);
        let (_cache, warnings) =
            Cache1DLinear::from_samples(&grid, &[finite_sample(1.0), finite_sample(0.95)]);
        assert!(warnings
            .iter()
            .all(|w| w.kind != OutlierKind::MonotonicityBreak));
    }
}
