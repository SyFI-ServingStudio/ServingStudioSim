use crate::timing::bridge::KernelMetrics;
use crate::timing::cache::interp::{
    locate, CoverageFlags, LeafMetrics, Metrics4, MONOTONICITY_TOLERANCE,
};
use crate::timing::cache::{peak_over_cells, Cache, OutlierKind, OutlierWarning, PeakRates};
use crate::timing::sweep::SweepGrid;

/// Trilinear interpolation over an exactly three-dimensional rectangular grid.
/// Samples are dense row-major over `(x0, x1, x2)`, with axis 2 varying fastest.
/// Non-finite samples follow the same safe convex-renormalization policy as
/// [`super::Cache2DLinear`].
#[derive(Clone, Debug)]
pub struct Cache3DLinear {
    axes: [Vec<f32>; 3],
    dims: [usize; 3],
    cells: Vec<Metrics4>,
    valid: Vec<bool>,
    all_finite: bool,
    any_valid: bool,
}

impl Cache for Cache3DLinear {
    fn from_samples(grid: &SweepGrid, samples: &[KernelMetrics]) -> (Self, Vec<OutlierWarning>) {
        assert_eq!(grid.axes().len(), 3, "Cache3DLinear requires a 3D grid");
        let dims: [usize; 3] = std::array::from_fn(|axis| grid.axes()[axis].len());
        let expected = dims.iter().product::<usize>();
        assert_eq!(
            expected,
            samples.len(),
            "sample count must match 3D sweep grid size (A*B*C)"
        );

        let orders: [Vec<usize>; 3] = std::array::from_fn(|axis| {
            let mut order: Vec<usize> = (0..dims[axis]).collect();
            order.sort_by(|&a, &b| grid.axes()[axis][a].total_cmp(&grid.axes()[axis][b]));
            order
        });
        let axes: [Vec<f32>; 3] = std::array::from_fn(|axis| {
            orders[axis]
                .iter()
                .map(|&index| grid.axes()[axis][index] as f32)
                .collect()
        });

        let mut warnings = Vec::new();
        let mut cells = Vec::with_capacity(expected);
        let mut valid = Vec::with_capacity(expected);
        for &i in &orders[0] {
            for &j in &orders[1] {
                for &k in &orders[2] {
                    let original = (i * dims[1] + j) * dims[2] + k;
                    let sample = &samples[original];
                    if sample.is_finite() {
                        cells.push(Metrics4::from_sample(sample));
                        valid.push(true);
                    } else {
                        warnings.push(OutlierWarning {
                            kind: OutlierKind::NonFinite,
                            detail: format!(
                                "x0={}, x1={}, x2={}: sample has non-finite or negative metric; dropped",
                                grid.axes()[0][i], grid.axes()[1][j], grid.axes()[2][k]
                            ),
                        });
                        cells.push(Metrics4::ZERO);
                        valid.push(false);
                    }
                }
            }
        }

        scan_monotonicity(&axes, dims, &cells, &valid, &mut warnings);
        let all_finite = valid.iter().all(|&value| value);
        let any_valid = valid.iter().any(|&value| value);
        (
            Self {
                axes,
                dims,
                cells,
                valid,
                all_finite,
                any_valid,
            },
            warnings,
        )
    }

    fn eval(&self, sweep: &[f64]) -> LeafMetrics {
        assert_eq!(
            sweep.len(),
            3,
            "Cache3DLinear lookup requires three coordinates"
        );
        if sweep.iter().any(|value| value.is_nan()) || !self.any_valid {
            return LeafMetrics::MISS;
        }
        let query = std::array::from_fn(|axis| sweep[axis] as f32);
        let (metrics, extrapolated) = self.interpolate_cell(query);
        LeafMetrics {
            m: metrics.clamped(),
            coverage: if extrapolated {
                CoverageFlags::EXTRAPOLATED
            } else {
                CoverageFlags::EMPTY
            },
            backend_index: LeafMetrics::NO_BACKEND,
        }
    }

    fn peak_rates(&self) -> PeakRates {
        peak_over_cells(
            self.cells
                .iter()
                .zip(&self.valid)
                .filter_map(|(&cell, &valid)| valid.then_some(cell)),
        )
    }
}

impl Cache3DLinear {
    fn interpolate_cell(&self, query: [f32; 3]) -> (Metrics4, bool) {
        let located: [(usize, usize, f32, bool); 3] =
            std::array::from_fn(|axis| locate(&self.axes[axis], query[axis]));
        let outside = located.iter().any(|entry| entry.3);
        let corners: [([usize; 3], usize, f32); 8] = std::array::from_fn(|mask| {
            let indices = std::array::from_fn(|axis| {
                if mask & (1 << axis) == 0 {
                    located[axis].0
                } else {
                    located[axis].1
                }
            });
            let weight = (0..3).fold(1.0, |weight, axis| {
                let t = located[axis].2;
                weight * if mask & (1 << axis) == 0 { 1.0 - t } else { t }
            });
            (indices, flat_index(self.dims, indices), weight)
        });

        if self.all_finite {
            let mut acc = Metrics4::ZERO;
            for &(_, index, weight) in &corners {
                acc.add_scaled(self.cells[index], weight);
            }
            return (acc, outside);
        }

        let meaningful_drop = corners
            .iter()
            .any(|&(_, index, weight)| !self.valid[index] && weight.abs() > f32::EPSILON);
        if !meaningful_drop {
            let mut acc = Metrics4::ZERO;
            for &(_, index, weight) in &corners {
                if self.valid[index] {
                    acc.add_scaled(self.cells[index], weight);
                }
            }
            return (acc, outside);
        }

        let mut acc = Metrics4::ZERO;
        let mut weight_sum = 0.0;
        for &(_, index, weight) in &corners {
            if self.valid[index] {
                let weight = weight.max(0.0);
                acc.add_scaled(self.cells[index], weight);
                weight_sum += weight;
            }
        }
        if weight_sum > f32::EPSILON {
            acc.scale(1.0 / weight_sum);
            return (acc, true);
        }

        let nearest = corners
            .iter()
            .filter(|&&(_indices, index, _)| self.valid[index])
            .map(|&(indices, index, _)| {
                let distance = (0..3)
                    .map(|axis| {
                        let (lo, hi, _, _) = located[axis];
                        let width = (self.axes[axis][hi] - self.axes[axis][lo]).abs();
                        if width <= f32::EPSILON {
                            0.0
                        } else {
                            let delta = (query[axis] - self.axes[axis][indices[axis]]) / width;
                            delta * delta
                        }
                    })
                    .sum::<f32>();
                (self.cells[index], distance)
            })
            .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
            .map_or(Metrics4::ZERO, |(cell, _)| cell);
        (nearest, true)
    }
}

fn flat_index(dims: [usize; 3], indices: [usize; 3]) -> usize {
    (indices[0] * dims[1] + indices[1]) * dims[2] + indices[2]
}

fn decode_index(dims: [usize; 3], mut index: usize) -> [usize; 3] {
    let mut out = [0; 3];
    for axis in (0..3).rev() {
        out[axis] = index % dims[axis];
        index /= dims[axis];
    }
    out
}

fn scan_monotonicity(
    axes: &[Vec<f32>; 3],
    dims: [usize; 3],
    cells: &[Metrics4],
    valid: &[bool],
    warnings: &mut Vec<OutlierWarning>,
) {
    for axis in 0..3 {
        for index in 0..cells.len() {
            let mut current = decode_index(dims, index);
            if current[axis] != 0 {
                continue;
            }
            let mut previous: Option<([usize; 3], f32)> = None;
            for position in 0..dims[axis] {
                current[axis] = position;
                let current_index = flat_index(dims, current);
                if !valid[current_index] {
                    continue;
                }
                let time = cells[current_index].time_ms;
                if let Some((before_indices, before_time)) = previous {
                    if time < before_time * (1.0 - MONOTONICITY_TOLERANCE) {
                        let before: [f32; 3] = std::array::from_fn(|a| axes[a][before_indices[a]]);
                        let after: [f32; 3] = std::array::from_fn(|a| axes[a][current[a]]);
                        warnings.push(OutlierWarning {
                            kind: OutlierKind::MonotonicityBreak,
                            detail: format!(
                                "time {:.4}ms (x0={}, x1={}, x2={}) → {:.4}ms (x0={}, x1={}, x2={}) drops >{:.0}% along axis-{axis}, non-monotonic",
                                before_time,
                                before[0], before[1], before[2],
                                time,
                                after[0], after[1], after[2],
                                MONOTONICITY_TOLERANCE * 100.0
                            ),
                        });
                    }
                }
                previous = Some((current, time));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Cache3DLinear;
    use crate::timing::bridge::KernelMetrics;
    use crate::timing::cache::interp::CoverageFlags;
    use crate::timing::cache::{build_cache, Cache, CacheKind, Extrapolation, OutlierKind};
    use crate::timing::sweep::SweepGrid;

    fn sample(time_ms: f64) -> KernelMetrics {
        KernelMetrics {
            time_ms,
            tflops: Some(1.0),
            memory_bandwidth_gbps: Some(2.0),
            algbw_gbps: None,
            busbw_gbps: None,
            energy_j: time_ms * 2.0,
        }
    }

    fn nan_sample() -> KernelMetrics {
        let mut value = sample(1.0);
        value.time_ms = f64::NAN;
        value
    }

    fn cube_samples(f: impl Fn([f64; 3]) -> f64) -> (SweepGrid, Vec<KernelMetrics>) {
        let grid = SweepGrid::new(vec![vec![0.0, 1.0]; 3]);
        let samples = grid.expand(|coords| sample(f(coords.try_into().unwrap())));
        (grid, samples)
    }

    #[test]
    #[should_panic(expected = "requires a 3D grid")]
    fn rejects_wrong_dimension() {
        Cache3DLinear::from_samples(&SweepGrid::new(vec![vec![1.0]; 2]), &[sample(1.0)]);
    }

    #[test]
    #[should_panic(expected = "A*B*C")]
    fn rejects_wrong_sample_count() {
        Cache3DLinear::from_samples(
            &SweepGrid::new(vec![vec![1.0, 2.0], vec![1.0], vec![1.0]]),
            &[sample(1.0)],
        );
    }

    #[test]
    fn exact_points_center_and_trilinear_surface_interpolate_all_metrics() {
        let (grid, samples) =
            cube_samples(|x| 2.0 + x[0] + 2.0 * x[1] + 3.0 * x[2] + 5.0 * x[0] * x[1] * x[2]);
        let (cache, _) = Cache3DLinear::from_samples(&grid, &samples);
        assert_eq!(cache.eval(&[0.0; 3]).m.time_ms, 2.0);
        assert_eq!(cache.eval(&[1.0; 3]).m.time_ms, 13.0);
        let center = cache.eval(&[0.5; 3]);
        assert!((center.m.time_ms - 5.625).abs() < 1e-5);
        assert!((center.m.flops - 5.625e9).abs() < 4096.0);
        assert!((center.m.bytes - 11.25e6).abs() < 4.0);
        assert!((center.m.energy_j - 11.25).abs() < 1e-5);
        assert_eq!(center.coverage, CoverageFlags::EMPTY);
    }

    #[test]
    fn all_eight_corner_weights_are_used() {
        let grid = SweepGrid::new(vec![vec![0.0, 1.0]; 3]);
        for active in 0..8 {
            let samples = (0..8)
                .map(|index| sample(if index == active { 8.0 } else { 0.0 }))
                .collect::<Vec<_>>();
            let (cache, _) = Cache3DLinear::from_samples(&grid, &samples);
            assert!((cache.eval(&[0.5; 3]).m.time_ms - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn independently_unsorted_axes_permute_cells_correctly() {
        let grid = SweepGrid::new(vec![vec![10.0, 0.0], vec![20.0, 0.0], vec![30.0, 0.0]]);
        let samples = grid.expand(|x| sample(x.iter().sum()));
        let (cache, _) = Cache3DLinear::from_samples(&grid, &samples);
        assert_eq!(cache.eval(&[0.0, 0.0, 0.0]).m.time_ms, 0.0);
        assert_eq!(cache.eval(&[10.0, 20.0, 30.0]).m.time_ms, 60.0);
        assert_eq!(cache.eval(&[5.0, 10.0, 15.0]).m.time_ms, 30.0);
    }

    #[test]
    fn faces_edges_singletons_and_each_axis_extrapolation_are_supported() {
        let (grid, samples) = cube_samples(|x| x.iter().sum::<f64>() + 1.0);
        let (cache, _) = Cache3DLinear::from_samples(&grid, &samples);
        assert_eq!(cache.eval(&[0.0, 0.5, 0.5]).m.time_ms, 2.0);
        assert_eq!(cache.eval(&[0.0, 0.0, 0.5]).m.time_ms, 1.5);
        for axis in 0..3 {
            let mut query = [0.5; 3];
            query[axis] = 1.5;
            assert_eq!(cache.eval(&query).coverage, CoverageFlags::EXTRAPOLATED);
        }
        assert_eq!(
            cache.eval(&[-1.0, 2.0, 0.5]).coverage,
            CoverageFlags::EXTRAPOLATED
        );

        let singleton = SweepGrid::new(vec![vec![2.0], vec![3.0], vec![4.0]]);
        let (cache, _) = Cache3DLinear::from_samples(&singleton, &[sample(7.0)]);
        assert_eq!(cache.eval(&[2.0, 3.0, 4.0]).m.time_ms, 7.0);
        assert_eq!(
            cache.eval(&[3.0, 3.0, 4.0]).coverage,
            CoverageFlags::EXTRAPOLATED
        );
    }

    #[test]
    fn dropped_corner_zero_weight_and_convex_renormalization_match_safe_policy() {
        let grid = SweepGrid::new(vec![vec![0.0, 1.0]; 3]);
        let mut samples = (0..8).map(|index| sample(index as f64)).collect::<Vec<_>>();
        samples[7] = nan_sample();
        let (cache, warnings) = Cache3DLinear::from_samples(&grid, &samples);
        let exact = cache.eval(&[0.0; 3]);
        assert_eq!(exact.coverage, CoverageFlags::EMPTY);
        assert_eq!(exact.m.time_ms, 0.0);
        let center = cache.eval(&[0.5; 3]);
        assert_eq!(center.coverage, CoverageFlags::EXTRAPOLATED);
        assert!((center.m.time_ms - 3.0).abs() < 1e-5);
        assert!(warnings
            .iter()
            .any(|warning| warning.kind == OutlierKind::NonFinite));
    }

    #[test]
    fn multiple_drops_dropped_face_and_nearest_fallback_are_deterministic() {
        let grid = SweepGrid::new(vec![vec![0.0, 1.0]; 3]);
        let mut samples = (0..8)
            .map(|index| sample(index as f64 + 1.0))
            .collect::<Vec<_>>();
        for sample in &mut samples[..4] {
            *sample = nan_sample();
        }
        let (cache, _) = Cache3DLinear::from_samples(&grid, &samples);
        let center = cache.eval(&[0.5; 3]);
        assert_eq!(center.coverage, CoverageFlags::EXTRAPOLATED);
        assert!(center.m.time_ms >= 5.0 && center.m.time_ms <= 8.0);
        let dropped_corner = cache.eval(&[0.0; 3]);
        assert_eq!(dropped_corner.coverage, CoverageFlags::EXTRAPOLATED);
        assert_eq!(dropped_corner.m.time_ms, 5.0);
    }

    #[test]
    fn all_invalid_and_nan_queries_return_no_coverage() {
        let grid = SweepGrid::new(vec![vec![0.0, 1.0]; 3]);
        let (cache, _) = Cache3DLinear::from_samples(&grid, &vec![nan_sample(); 8]);
        let miss = cache.eval(&[0.5; 3]);
        assert_eq!(miss.coverage, CoverageFlags::NO_COVERAGE);
        assert_eq!(miss.m.time_ms, 0.0);

        let (_, samples) = cube_samples(|_| 1.0);
        let (cache, _) = Cache3DLinear::from_samples(&grid, &samples);
        let miss = cache.eval(&[f64::NAN, 0.0, 0.0]);
        assert_eq!(miss.coverage, CoverageFlags::NO_COVERAGE);
        assert_eq!(miss.m.time_ms, 0.0);
    }

    #[test]
    fn output_clamps_nonnegative_and_peak_rates_ignore_drops() {
        let grid = SweepGrid::new(vec![vec![0.0, 1.0]; 3]);
        let mut samples = vec![sample(2.0); 8];
        samples[0] = sample(1.0);
        samples[7] = nan_sample();
        let (cache, _) = Cache3DLinear::from_samples(&grid, &samples);
        let extrapolated = cache.eval(&[-10.0, 0.0, 0.0]);
        assert_eq!(extrapolated.m.time_ms, 0.0);
        assert!(extrapolated.m.flops >= 0.0);
        assert!(cache.peak_rates().tflops > 0.0);
        assert!(cache.peak_rates().gbps > 0.0);
    }

    #[test]
    fn monotonicity_scans_all_directions_and_dispatch_preserves_existing_variants() {
        for axis in 0..3 {
            let grid = SweepGrid::new(vec![vec![0.0, 1.0]; 3]);
            let samples =
                grid.expand(|coords| sample(if coords[axis] == 0.0 { 10.0 } else { 1.0 }));
            let (_, warnings) = Cache3DLinear::from_samples(&grid, &samples);
            assert!(warnings.iter().any(|warning| {
                warning.kind == OutlierKind::MonotonicityBreak
                    && warning.detail.contains(&format!("axis-{axis}"))
            }));
        }

        let (grid, samples) = cube_samples(|x| x.iter().sum::<f64>() + 1.0);
        let (cache, _) = build_cache("test", CacheKind::Cache3DLinear, &grid, &samples).unwrap();
        assert_eq!(cache.eval(&[0.5; 3]).coverage, CoverageFlags::EMPTY);

        for (kind, grid, samples) in [
            (
                CacheKind::Cache1DLinear,
                SweepGrid::new(vec![vec![1.0, 2.0]]),
                vec![sample(1.0), sample(2.0)],
            ),
            (
                CacheKind::Cache1DDirect,
                SweepGrid::new(vec![vec![1.0, 2.0]]),
                vec![sample(1.0), sample(2.0)],
            ),
            (
                CacheKind::Cache2DLinear(Extrapolation::Clamp),
                SweepGrid::new(vec![vec![1.0], vec![1.0]]),
                vec![sample(1.0)],
            ),
        ] {
            assert!(build_cache("test", kind, &grid, &samples).is_ok());
        }
    }
}
