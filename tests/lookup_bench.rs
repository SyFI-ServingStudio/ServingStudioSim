//! Hot-path microbenchmark for cache lookups — the dominant cost in the `CostTree`
//! eval path (`Cache::eval`: bilinear/linear interpolation).
//!
//! Dependency-free (no criterion) and `#[ignore]`d so it never runs in the
//! normal suite. It needs no Python, so plain cargo works:
//!
//!     cargo test --release --test lookup_bench -- --ignored --nocapture
//!
//! Use `--release` for representative numbers; the debug build is ~10x slower.
//! Each case prints ns/lookup so a before/after comparison is a glance.

use std::hint::black_box;
use std::time::Instant;

use simulator::timing::bridge::KernelMetrics;
use simulator::timing::cache::{Cache, Cache1DDirect, Cache1DLinear, Cache2DLinear};
use simulator::timing::{Axis, SweepGrid};

const ITERS: u32 = 5_000_000;

fn compute_sample(time_ms: f64) -> KernelMetrics {
    KernelMetrics {
        time_ms,
        tflops: Some(1.0),
        memory_bandwidth_gbps: Some(2.0),
        algbw_gbps: None,
        busbw_gbps: None,
        energy_j: time_ms * 0.1,
    }
}

/// Sweep a set of probe coordinates `iters` times, reporting ns per lookup.
/// `black_box` on both the coordinate and the result defeats hoisting/DCE so
/// the loop measures real interpolation work.
fn bench<F: Fn(f64, f64) -> f64>(label: &str, probes: &[(f64, f64)], lookup: F) {
    // Warm up off the clock.
    for &(a, b) in probes {
        black_box(lookup(black_box(a), black_box(b)));
    }
    // Loop over the probe slice directly (no per-iter `% len` division, which
    // would add ~6-12ns of harness noise) and accumulate the result so the
    // optimizer can't elide the call.
    let reps = (ITERS as usize).div_ceil(probes.len());
    let total = reps * probes.len();
    let mut acc = 0.0f64;
    let start = Instant::now();
    for _ in 0..reps {
        for &(a, b) in probes {
            acc += lookup(black_box(a), black_box(b));
        }
    }
    black_box(acc);
    let elapsed = start.elapsed();
    #[allow(
        clippy::cast_precision_loss,
        reason = "elapsed ns and total iters converted to f64 for a human-readable ns/op ratio; \
                  precision loss is irrelevant at benchmark magnitudes"
    )]
    let ns_per = elapsed.as_nanos() as f64 / total as f64;
    println!("{label:<30} {ns_per:>7.2} ns/op  ({total} iters, {elapsed:?})");
}

#[test]
#[ignore = "microbenchmark; run with: cargo test --release --test lookup_bench -- --ignored --nocapture"]
fn cache_lookup_throughput() {
    // 1D: the shared 63-point token axis with a monotonic time curve.
    let axis = Axis::token_axis();
    let grid_1d = SweepGrid::new(vec![axis.clone()]);
    let samples_1d: Vec<KernelMetrics> = axis
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            #[allow(
                clippy::cast_precision_loss,
                reason = "idx is a small axis index (<=63), far below f64's exact-integer range"
            )]
            compute_sample(1.0 + idx as f64 * 0.5)
        })
        .collect();
    let (cache_1d, warnings) = Cache1DLinear::from_samples(&grid_1d, &samples_1d);
    assert!(warnings.is_empty(), "1D fixture must fit cleanly");

    // Probes span interior interpolation and both extrapolation edges.
    let probes_1d: Vec<(f64, f64)> = [16.0, 48.0, 300.0, 5000.0, 40000.0, 70000.0]
        .into_iter()
        .map(|x| (x, 0.0))
        .collect();
    // Harness floor: a trivial closure measures loop + black_box + accumulate
    // overhead, so the cache numbers below can be read net of it.
    bench("baseline (a + b)", &probes_1d, |a, b| a + b);
    bench("Cache1DLinear::eval", &probes_1d, |x, _| {
        f64::from(cache_1d.eval(&[x]).m.time_ms)
    });

    // 1D direct-indexed: 512 buckets, spacing 64, range [0, 32704] — the bounded
    // batch/kv-length LUT the cache is built for. Lookup is O(1) (floor-index, no
    // search), so it should beat Cache1DLinear's binary search on the same probes.
    let axis_direct = Axis::arithmetic(0, 64 * 511, 64); // 512 points, spacing 64
    assert_eq!(axis_direct.len(), 512);
    let grid_direct = SweepGrid::new(vec![axis_direct.clone()]);
    let samples_direct: Vec<KernelMetrics> = (0..axis_direct.len())
        .map(|idx| {
            #[allow(
                clippy::cast_precision_loss,
                reason = "idx is a small direct-cache bucket index (<=512), far below f64's \
                          exact-integer range"
            )]
            compute_sample(1.0 + idx as f64 * 0.5)
        })
        .collect();
    let (cache_direct, warnings) = Cache1DDirect::from_samples(&grid_direct, &samples_direct);
    assert!(warnings.is_empty(), "direct fixture must fit cleanly");
    // Probes: interior buckets plus a clamped out-of-range high.
    let probes_direct: Vec<(f64, f64)> = [100.0, 3000.0, 12345.0, 20000.0, 32000.0, 99999.0]
        .into_iter()
        .map(|x| (x, 0.0))
        .collect();
    bench("Cache1DDirect::eval", &probes_direct, |x, _| {
        f64::from(cache_direct.eval(&[x]).m.time_ms)
    });

    // 2D: the three real attention grids, smallest-to-largest. The hot question
    // is L1 residency — the all-finite hot path touches only `cells` at 16
    // bytes/cell, so prefill (90 cells = 1.4KB) and decode (162 = 2.5KB) sit
    // entirely in L1; rect (63² = 3969 = 62KB) spills to L2, so its 4-corner
    // loads (two rows c apart) cost a line fill each.
    let bench_2d = |label: &str, axis0: Vec<f64>, axis1: Vec<f64>, probes: &[(f64, f64)]| {
        let (r, c) = (axis0.len(), axis1.len());
        let grid = SweepGrid::new(vec![axis0, axis1]);
        let samples: Vec<KernelMetrics> = (0..r)
            .flat_map(|i| (0..c).map(move |j| (i, j)))
            .map(|(i, j)| {
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "i,j are small 2D grid indices from test fixtures, far below f64's \
                              exact-integer range"
                )]
                compute_sample(1.0 + i as f64 + j as f64)
            })
            .collect();
        let (cache, warnings) = Cache2DLinear::from_samples(&grid, &samples);
        assert!(warnings.is_empty(), "2D fixture {label} must fit cleanly");
        bench(label, probes, move |x0, x1| {
            f64::from(cache.eval(&[x0, x1]).m.time_ms)
        });
    };

    // prefill: prefix_len (chain [0] + pow2 7..15) × append_len (pow2 7..15).
    bench_2d(
        "Cache2DLinear prefill 10x9",
        Axis::chain([Axis::values([0]), Axis::pow2(7, 15)]),
        Axis::pow2(7, 15),
        &[
            (0.0, 256.0),
            (4096.0, 8192.0),
            (32768.0, 32768.0),
            (1500.0, 700.0),
            (40000.0, 1000.0),
        ],
    );
    // decode: batch_size (pow2 0..8) × total_tokens (pow2 5..22).
    bench_2d(
        "Cache2DLinear decode 9x18",
        Axis::pow2(0, 8),
        Axis::pow2(5, 22),
        &[
            (4.0, 1024.0),
            (32.0, 65536.0),
            (256.0, 4194304.0),
            (3.0, 700.0),
            (500.0, 100.0),
        ],
    );
    // rect: token_axis² (largest, L2-resident).
    bench_2d(
        "Cache2DLinear rect 63x63",
        Axis::token_axis(),
        Axis::token_axis(),
        &[
            (200.0, 200.0),
            (4096.0, 8192.0),
            (65536.0, 65536.0),
            (64.0, 64.0),
            (40000.0, 1000.0),
        ],
    );
}
