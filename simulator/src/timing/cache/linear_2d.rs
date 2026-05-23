use crate::timing::bridge::KernelMetrics;
use crate::timing::cache::interp::{locate, CoverageFlags, LeafMetrics, Metrics4, MONOTONICITY_TOLERANCE};
use crate::timing::cache::{Cache, OutlierKind, OutlierWarning};
use crate::timing::sweep::SweepGrid;

/// Bilinear interpolation over a rectangular 2D profile grid. Both axes are
/// expected monotonic-in-time (bigger coordinate ⇒ more work ⇒ more time), e.g.
/// AttnPrefill's `seq_len_q × kv_cache_len`. The grid is the cartesian product
/// of the two sweep axes; `samples` arrive row-major (`samples[i*C + j]` is the
/// point at `axis0[i] × axis1[j]`, matching `SweepGrid::expand_2d`).
#[derive(Clone, Debug)]
pub struct Cache2DLinear {
    /// Axis-0 coordinates, sorted ascending (len R).
    xs0: Vec<f32>,
    /// Axis-1 coordinates, sorted ascending (len C).
    xs1: Vec<f32>,
    /// R*C cells, row-major over `(xs0, xs1)`. `None` = the sample at that grid
    /// point was dropped as non-finite (see `from_samples`); lookups that lean
    /// on a dropped corner renormalize over the surviving corners.
    cells: Vec<Option<Metrics4>>,
}

impl Cache for Cache2DLinear {
    fn from_samples(grid: &SweepGrid, samples: &[KernelMetrics]) -> (Self, Vec<OutlierWarning>) {
        assert_eq!(grid.axes().len(), 2, "Cache2DLinear requires a 2D grid");
        let axis0 = &grid.axes()[0];
        let axis1 = &grid.axes()[1];
        let (r, c) = (axis0.len(), axis1.len());
        assert_eq!(
            r * c,
            samples.len(),
            "sample count must match 2D sweep grid size (R*C)"
        );

        let mut warnings = Vec::new();

        // Axis builders emit ascending coordinates, but `Axis::values` preserves
        // caller order — sort each axis independently and permute the cells so
        // lookup can bracket on ascending coordinates.
        let mut order0: Vec<usize> = (0..r).collect();
        order0.sort_by(|&a, &b| axis0[a].total_cmp(&axis0[b]));
        let mut order1: Vec<usize> = (0..c).collect();
        order1.sort_by(|&a, &b| axis1[a].total_cmp(&axis1[b]));

        let xs0: Vec<f32> = order0.iter().map(|&i| axis0[i] as f32).collect();
        let xs1: Vec<f32> = order1.iter().map(|&j| axis1[j] as f32).collect();

        let mut cells: Vec<Option<Metrics4>> = Vec::with_capacity(r * c);
        for &i in &order0 {
            for &j in &order1 {
                let sample = &samples[i * c + j];
                if !sample.is_finite() {
                    warnings.push(OutlierWarning {
                        kind: OutlierKind::NonFinite,
                        detail: format!(
                            "x0={}, x1={}: sample has non-finite or negative metric; dropped",
                            axis0[i], axis1[j]
                        ),
                    });
                    cells.push(None);
                    continue;
                }
                cells.push(Some(Metrics4::from_sample(sample)));
            }
        }

        // Monotonicity scan along each axis: time should be non-decreasing as
        // either coordinate grows. Walk every row (axis-1 sweep at fixed axis-0)
        // and every column (axis-0 sweep at fixed axis-1), comparing consecutive
        // finite cells; a drop beyond `MONOTONICITY_TOLERANCE` flags a break.
        for i in 0..r {
            let mut prev: Option<(usize, f32)> = None;
            for j in 0..c {
                if let Some(cell) = cells[i * c + j] {
                    if let Some((pj, ptime)) = prev {
                        if cell.time_ms < ptime * (1.0 - MONOTONICITY_TOLERANCE) {
                            warnings.push(OutlierWarning {
                                kind: OutlierKind::MonotonicityBreak,
                                detail: format!(
                                    "time {:.4}ms (x0={}, x1={}) → {:.4}ms (x0={}, x1={}) drops >{:.0}% along axis-1, non-monotonic",
                                    ptime, xs0[i], xs1[pj], cell.time_ms, xs0[i], xs1[j],
                                    MONOTONICITY_TOLERANCE * 100.0
                                ),
                            });
                        }
                    }
                    prev = Some((j, cell.time_ms));
                }
            }
        }
        for j in 0..c {
            let mut prev: Option<(usize, f32)> = None;
            for i in 0..r {
                if let Some(cell) = cells[i * c + j] {
                    if let Some((pi, ptime)) = prev {
                        if cell.time_ms < ptime * (1.0 - MONOTONICITY_TOLERANCE) {
                            warnings.push(OutlierWarning {
                                kind: OutlierKind::MonotonicityBreak,
                                detail: format!(
                                    "time {:.4}ms (x0={}, x1={}) → {:.4}ms (x0={}, x1={}) drops >{:.0}% along axis-0, non-monotonic",
                                    ptime, xs0[pi], xs1[j], cell.time_ms, xs0[i], xs1[j],
                                    MONOTONICITY_TOLERANCE * 100.0
                                ),
                            });
                        }
                    }
                    prev = Some((i, cell.time_ms));
                }
            }
        }

        (Self { xs0, xs1, cells }, warnings)
    }

    fn eval(&self, sweep: &[f64]) -> LeafMetrics {
        assert_eq!(
            sweep.len(),
            2,
            "Cache2DLinear lookup requires two coordinates"
        );
        let (x0, x1) = (sweep[0], sweep[1]);
        if x0.is_nan() || x1.is_nan() || self.cells.iter().all(Option::is_none) {
            return LeafMetrics {
                m: Metrics4::ZERO,
                coverage: CoverageFlags::NO_COVERAGE,
            };
        }
        let (cell, extrapolated) = self.interpolate_cell(x0 as f32, x1 as f32);
        LeafMetrics {
            m: cell.clamped(),
            coverage: if extrapolated {
                CoverageFlags::EXTRAPOLATED
            } else {
                CoverageFlags::EMPTY
            },
        }
    }
}

impl Cache2DLinear {
    /// Locate the 2D bracketing cell once (one branchless `locate` per axis),
    /// then bilinear-blend all four metrics of the (up to) four corners in a
    /// single pass — the corner cells and their weights are computed once and
    /// shared across metrics. Returns the interpolated `Metrics4` plus whether
    /// `(x0, x1)` fell outside the grid.
    fn interpolate_cell(&self, x0: f32, x1: f32) -> (Metrics4, bool) {
        let (i0, i1, t0, out0) = locate(&self.xs0, x0);
        let (j0, j1, t1, out1) = locate(&self.xs1, x1);
        let c = self.xs1.len();
        let get = |i: usize, j: usize| self.cells[i * c + j];

        // Bilinear weights always sum to 1 regardless of t (including the
        // out-of-[0,1] t used for extrapolation), so a full set of finite
        // corners yields the exact bilinear value / linear extrapolation.
        let corners = [
            (i0, j0, get(i0, j0), (1.0 - t0) * (1.0 - t1)),
            (i0, j1, get(i0, j1), (1.0 - t0) * t1),
            (i1, j0, get(i1, j0), t0 * (1.0 - t1)),
            (i1, j1, get(i1, j1), t0 * t1),
        ];

        // Fast path: every corner that carries weight is present. A missing
        // corner whose weight is ~0 — e.g. querying exactly on a surviving edge
        // or grid point — contributes nothing and is not coverage loss.
        let meaningful_drop = corners
            .iter()
            .any(|(_, _, cell, weight)| cell.is_none() && weight.abs() > f32::EPSILON);
        if !meaningful_drop {
            let mut acc = Metrics4::ZERO;
            for &(_, _, cell, weight) in &corners {
                if let Some(cell) = cell {
                    acc.add_scaled(cell, weight);
                }
            }
            return (acc, out0 || out1);
        }

        // A weighted corner was dropped as non-finite. Blend the surviving
        // corners using their weights clamped to non-negative, so the result
        // stays a convex combination of real measurements — signed extrapolation
        // weights can't distort or cancel it. When no positive weight remains
        // (the query sits on the dropped corner itself), fall back to the
        // surviving corner nearest to the query in normalized cell coordinates.
        // Coverage is partial either way → flag extrapolated.
        let mut acc = Metrics4::ZERO;
        let mut weight_sum = 0.0;
        for &(_, _, cell, weight) in &corners {
            if let Some(cell) = cell {
                let weight = weight.max(0.0);
                acc.add_scaled(cell, weight);
                weight_sum += weight;
            }
        }
        if weight_sum > f32::EPSILON {
            acc.scale(1.0 / weight_sum);
            return (acc, true);
        }
        let normalized_distance = |axis: &[f32], lo: usize, hi: usize, x: f32, idx: usize| {
            let width = (axis[hi] - axis[lo]).abs();
            if width <= f32::EPSILON {
                0.0
            } else {
                ((x - axis[idx]) / width).abs()
            }
        };
        let nearest = corners
            .iter()
            .filter_map(|&(i, j, cell, _)| {
                cell.map(|cell| {
                    let d0 = normalized_distance(&self.xs0, i0, i1, x0, i);
                    let d1 = normalized_distance(&self.xs1, j0, j1, x1, j);
                    (cell, d0 * d0 + d1 * d1)
                })
            })
            .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
            .map(|(cell, _)| cell)
            .unwrap_or(Metrics4::ZERO);
        (nearest, true)
    }
}

#[cfg(test)]
mod tests {
    use crate::timing::bridge::KernelMetrics;
    use crate::timing::cache::interp::CoverageFlags;
    use crate::timing::cache::{Cache, Cache2DLinear, OutlierKind};
    use crate::timing::sweep::SweepGrid;

    fn sample(time_ms: f64) -> KernelMetrics {
        KernelMetrics {
            time_ms,
            tflops: Some(1.0),
            memory_bandwidth_gbps: Some(2.0),
            algbw_gbps: None,
            busbw_gbps: None,
            message_size_bytes: None,
            energy_j: time_ms,
        }
    }

    fn nan_sample() -> KernelMetrics {
        let mut m = sample(0.0);
        m.time_ms = f64::NAN;
        m
    }

    #[test]
    #[ignore = "internal microbench; run: cargo test --release --lib component_breakdown_2d -- --ignored --nocapture"]
    fn component_breakdown_2d() {
        use super::Cache2DLinear;
        use crate::timing::cache::interp::locate;
        use crate::timing::sweep::Axis;
        use std::hint::black_box;
        use std::time::Instant;

        // 63x63 (token_axis²) — a realistic 2D attention-style fit.
        let axis = Axis::token_axis();
        let (r, c) = (axis.len(), axis.len());
        let grid = SweepGrid::new(vec![axis.clone(), axis.clone()]);
        let samples: Vec<KernelMetrics> = (0..r)
            .flat_map(|i| (0..c).map(move |j| (i, j)))
            .map(|(i, j)| sample(1.0 + i as f64 + j as f64))
            .collect();
        let (cache, warnings) = Cache2DLinear::from_samples(&grid, &samples);
        assert!(warnings.is_empty());

        let probes = [
            (200.0f32, 200.0f32),
            (1000.0, 1500.0),
            (40000.0, 8000.0),
            (64.0, 64.0),
            (3000.0, 1000.0),
        ];
        let run = |label: &str, f: &dyn Fn(f32, f32) -> f32| {
            for &(a, b) in &probes {
                black_box(f(a, b));
            }
            let reps = 5_000_000usize / probes.len();
            let mut acc = 0.0f32;
            let start = Instant::now();
            for _ in 0..reps {
                for &(a, b) in &probes {
                    acc += f(black_box(a), black_box(b));
                }
            }
            black_box(acc);
            let ns = start.elapsed().as_nanos() as f64 / (reps * probes.len()) as f64;
            println!("{label:<34} {ns:>7.2} ns");
        };

        run("baseline (a + b)", &|a, b| a + b);
        run("1x locate (axis0)", &|a, _| locate(&cache.xs0, a).2);
        run("2x locate (both axes)", &|a, b| {
            locate(&cache.xs0, a).2 + locate(&cache.xs1, b).2
        });
        run("interpolate_cell (4 fields)", &|a, b| {
            cache.interpolate_cell(a, b).0.time_ms
        });
        run("eval", &|a, b| {
            cache.eval(&[a as f64, b as f64]).m.time_ms
        });
    }

    /// 2x2 grid, axis0 = [1,2], axis1 = [10,20], times row-major:
    /// (1,10)=1, (1,20)=2, (2,10)=3, (2,20)=4.
    fn grid_2x2() -> (SweepGrid, Vec<KernelMetrics>) {
        let grid = SweepGrid::new(vec![vec![1.0, 2.0], vec![10.0, 20.0]]);
        let samples = vec![sample(1.0), sample(2.0), sample(3.0), sample(4.0)];
        (grid, samples)
    }

    #[test]
    fn bilinear_interpolates_grid_center() {
        let (grid, samples) = grid_2x2();
        let (cache, warnings) = Cache2DLinear::from_samples(&grid, &samples);
        assert!(warnings.is_empty());

        // Center (1.5, 15): bilinear of {1,2,3,4} = mean = 2.5.
        let center = cache.eval(&[1.5, 15.0]);
        assert_eq!(center.m.time_ms, 2.5);
        assert!(center.coverage.is_empty());

        // Corner reproduces the sampled value exactly.
        let corner = cache.eval(&[2.0, 20.0]);
        assert_eq!(corner.m.time_ms, 4.0);

        // Edge midpoint along axis-1 at x0=1: between (1,10)=1 and (1,20)=2 → 1.5.
        let edge = cache.eval(&[1.0, 15.0]);
        assert_eq!(edge.m.time_ms, 1.5);
    }

    #[test]
    fn out_of_grid_extrapolates_and_flags() {
        let (grid, samples) = grid_2x2();
        let (cache, _) = Cache2DLinear::from_samples(&grid, &samples);

        // x0=3 is one full axis-0 step past the edge; along axis-1 at x1=10 the
        // gradient is (3-1)=2 per unit x0, so extrapolating to x0=3 gives 5.0.
        let beyond = cache.eval(&[3.0, 10.0]);
        assert_eq!(beyond.m.time_ms, 5.0);
        assert!(beyond.coverage.contains(CoverageFlags::EXTRAPOLATED));
    }

    #[test]
    fn non_finite_sample_is_dropped_and_flagged() {
        let grid = SweepGrid::new(vec![vec![1.0, 2.0], vec![10.0, 20.0]]);
        // Drop the (2,20) corner; the other three remain.
        let samples = vec![sample(1.0), sample(2.0), sample(3.0), nan_sample()];
        let (cache, warnings) = Cache2DLinear::from_samples(&grid, &samples);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].kind, OutlierKind::NonFinite);

        // The three surviving corners still reproduce exactly at their points.
        // The dropped (2,20) corner carries zero bilinear weight at these
        // queries, so it is not coverage loss → no spurious Extrapolated flag.
        let s10 = cache.eval(&[1.0, 10.0]);
        assert_eq!(s10.m.time_ms, 1.0);
        assert!(s10.coverage.is_empty());
        let s30 = cache.eval(&[2.0, 10.0]);
        assert_eq!(s30.m.time_ms, 3.0);
        assert!(s30.coverage.is_empty());

        // A lookup leaning on the dropped corner blends the survivors and flags
        // coverage loss rather than emitting a silent value.
        let leans = cache.eval(&[2.0, 20.0]);
        assert_eq!(leans.m.time_ms, 2.0);
        assert!(leans.coverage.contains(CoverageFlags::EXTRAPOLATED));
    }

    #[test]
    fn extrapolation_past_edge_is_unaffected_by_off_weight_dropped_corner() {
        // Drop (2,20). Extrapolating along the x1=10 edge to x0=3 only leans on
        // the (1,10)=1 and (2,10)=3 column, whose gradient is 2 per unit x0, so
        // the result is 5.0 — the dropped corner (zero weight here) must not
        // perturb it, matching the all-finite extrapolation case.
        let grid = SweepGrid::new(vec![vec![1.0, 2.0], vec![10.0, 20.0]]);
        let samples = vec![sample(1.0), sample(2.0), sample(3.0), nan_sample()];
        let (cache, _) = Cache2DLinear::from_samples(&grid, &samples);

        let beyond = cache.eval(&[3.0, 10.0]);
        assert_eq!(beyond.m.time_ms, 5.0);
        assert!(beyond.coverage.contains(CoverageFlags::EXTRAPOLATED));
    }

    #[test]
    fn all_non_finite_samples_make_empty_cache_flagged_no_coverage() {
        let grid = SweepGrid::new(vec![vec![1.0, 2.0], vec![10.0, 20.0]]);
        let samples = vec![nan_sample(), nan_sample(), nan_sample(), nan_sample()];
        let (cache, warnings) = Cache2DLinear::from_samples(&grid, &samples);
        assert_eq!(warnings.len(), 4);
        assert!(warnings.iter().all(|w| w.kind == OutlierKind::NonFinite));

        let result = cache.eval(&[1.5, 15.0]);
        assert_eq!(result.m.time_ms, 0.0);
        assert_eq!(result.m.flops, 0.0);
        assert!(result.coverage.contains(CoverageFlags::NO_COVERAGE));
    }

    #[test]
    fn non_monotonic_time_along_axis_flags_break() {
        // Along axis-1 at x0=1: 1.0 → 0.4 is a 60% drop (> 10% tol).
        let grid = SweepGrid::new(vec![vec![1.0, 2.0], vec![10.0, 20.0]]);
        let samples = vec![sample(1.0), sample(0.4), sample(2.0), sample(3.0)];
        let (_cache, warnings) = Cache2DLinear::from_samples(&grid, &samples);
        assert!(warnings
            .iter()
            .any(|w| w.kind == OutlierKind::MonotonicityBreak));
    }

    #[test]
    fn nan_coordinate_returns_no_coverage_zero() {
        let (grid, samples) = grid_2x2();
        let (cache, _) = Cache2DLinear::from_samples(&grid, &samples);

        let result = cache.eval(&[f64::NAN, 15.0]);
        assert_eq!(result.m.time_ms, 0.0);
        assert!(result.coverage.contains(CoverageFlags::NO_COVERAGE));
    }

    #[test]
    fn unsorted_axis_is_sorted_before_bracketing() {
        // axis0 given descending; cache must sort it so lookups bracket correctly.
        let grid = SweepGrid::new(vec![vec![2.0, 1.0], vec![10.0, 20.0]]);
        // row-major over given order: (2,10)=3, (2,20)=4, (1,10)=1, (1,20)=2.
        let samples = vec![sample(3.0), sample(4.0), sample(1.0), sample(2.0)];
        let (cache, warnings) = Cache2DLinear::from_samples(&grid, &samples);
        assert!(warnings
            .iter()
            .all(|w| w.kind != OutlierKind::MonotonicityBreak));

        // Same center as the sorted grid → 2.5.
        assert_eq!(cache.eval(&[1.5, 15.0]).m.time_ms, 2.5);
        assert_eq!(cache.eval(&[1.0, 10.0]).m.time_ms, 1.0);
        assert_eq!(cache.eval(&[2.0, 20.0]).m.time_ms, 4.0);
    }
}
