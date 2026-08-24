//! Statistical microbenchmark for the cache-lookup hot path (`Cache::eval`),
//! the dominant per-leaf cost in CostTree evaluation. Complements the
//! dependency-free `tests/lookup_bench.rs` (quick sanity glance) with
//! criterion's warmup + outlier rejection + confidence intervals, which the
//! bare `Instant` loop can't give for the few-ns deltas we're chasing.
//!
//!     cargo bench --bench cache_lookup
//!
//! 2D cases use the *real* attention sweep grids, not a synthetic square:
//!   - prefill: chain([0], pow2 7..15) × pow2 7..15   = 10 × 9  = 90 cells
//!   - decode:  pow2 0..8 × pow2 5..22                 = 9 × 18 = 162 cells
//!   - rect:    token_axis² (63 × 63)                  = 3969 cells
//!
//! Measurement showed lookup time is nearly flat across these sizes (~45ns),
//! so the cost is compute latency (2× binary search + 4-corner blend), not
//! cache residency — which is what these benches let us optimize against.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};

use simulator::timing::bridge::KernelMetrics;
use simulator::timing::cache::{Cache, Cache1DDirect, Cache1DLinear, Cache2DLinear};
use simulator::timing::{Axis, SweepGrid};

fn sample(time_ms: f64) -> KernelMetrics {
    KernelMetrics {
        time_ms,
        tflops: Some(1.0),
        memory_bandwidth_gbps: Some(2.0),
        algbw_gbps: None,
        busbw_gbps: None,
        energy_j: time_ms * 0.1,
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "benchmark axis indices are at most a few thousand, far below f64's exact-integer limit"
)]
fn index_as_f64(index: usize) -> f64 {
    index as f64
}

/// Drive a 1D cache over a fixed probe set, summing results so the optimizer
/// can't elide the calls.
fn bench_1d<C: Cache>(c: &mut Criterion, label: &str, cache: &C, probes: &[f64]) {
    c.bench_function(label, |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for &x in probes {
                acc += cache.eval(&[black_box(x)]).m.time_ms as f64;
            }
            black_box(acc)
        })
    });
}

fn bench_2d(c: &mut Criterion, label: &str, cache: &Cache2DLinear, probes: &[(f64, f64)]) {
    c.bench_function(label, |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for &(x0, x1) in probes {
                acc += cache.eval(&[black_box(x0), black_box(x1)]).m.time_ms as f64;
            }
            black_box(acc)
        })
    });
}

fn make_2d(axis0: Vec<f64>, axis1: Vec<f64>) -> Cache2DLinear {
    let (r, c) = (axis0.len(), axis1.len());
    let grid = SweepGrid::new(vec![axis0, axis1]);
    let samples: Vec<KernelMetrics> = (0..r)
        .flat_map(|i| (0..c).map(move |j| (i, j)))
        .map(|(i, j)| sample(1.0 + index_as_f64(i) + index_as_f64(j)))
        .collect();
    let (cache, warnings) = Cache2DLinear::from_samples(&grid, &samples);
    assert!(warnings.is_empty(), "2D fixture must fit cleanly");
    cache
}

fn cache_lookup(c: &mut Criterion) {
    // 1D linear: shared 63-point token axis, monotonic time curve.
    let axis = Axis::token_axis();
    let grid_1d = SweepGrid::new(vec![axis.clone()]);
    let samples_1d: Vec<KernelMetrics> = (0..axis.len())
        .map(|idx| sample(1.0 + index_as_f64(idx) * 0.5))
        .collect();
    let (cache_1d, w) = Cache1DLinear::from_samples(&grid_1d, &samples_1d);
    assert!(w.is_empty());
    let probes_1d = [16.0, 48.0, 300.0, 5000.0, 40000.0, 70000.0];
    bench_1d(c, "1d_linear", &cache_1d, &probes_1d);

    // 1D direct: 512 buckets, spacing 64.
    let axis_d = Axis::arithmetic(0, 64 * 511, 64);
    let grid_d = SweepGrid::new(vec![axis_d.clone()]);
    let samples_d: Vec<KernelMetrics> = (0..axis_d.len())
        .map(|idx| sample(1.0 + index_as_f64(idx) * 0.5))
        .collect();
    let (cache_d, w) = Cache1DDirect::from_samples(&grid_d, &samples_d);
    assert!(w.is_empty());
    let probes_d = [100.0, 3000.0, 12345.0, 20000.0, 32000.0, 99999.0];
    bench_1d(c, "1d_direct", &cache_d, &probes_d);

    // 2D: the three real attention grids.
    let prefill = make_2d(
        Axis::chain([Axis::values([0]), Axis::pow2(7, 15)]),
        Axis::pow2(7, 15),
    );
    bench_2d(
        c,
        "2d_prefill_10x9",
        &prefill,
        &[
            (0.0, 256.0),
            (4096.0, 8192.0),
            (32768.0, 32768.0),
            (1500.0, 700.0),
            (40000.0, 1000.0),
        ],
    );

    let decode = make_2d(Axis::pow2(0, 8), Axis::pow2(5, 22));
    bench_2d(
        c,
        "2d_decode_9x18",
        &decode,
        &[
            (4.0, 1024.0),
            (32.0, 65536.0),
            (256.0, 4194304.0),
            (3.0, 700.0),
            (500.0, 100.0),
        ],
    );

    let rect = make_2d(Axis::token_axis(), Axis::token_axis());
    bench_2d(
        c,
        "2d_rect_63x63",
        &rect,
        &[
            (200.0, 200.0),
            (4096.0, 8192.0),
            (65536.0, 65536.0),
            (64.0, 64.0),
            (40000.0, 1000.0),
        ],
    );
}

criterion_group!(benches, cache_lookup);
criterion_main!(benches);
