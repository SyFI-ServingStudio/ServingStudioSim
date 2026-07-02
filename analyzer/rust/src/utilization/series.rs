//! Per-pool GPU utilization over time: the mean fraction of a pool's workers busy
//! computing, binned into equal-width time segments (ref's interval-spread).
//!
//! `cost_log` records each worker iteration's busy interval
//! `[wall_start_ms, wall_start_ms + total_time_ms]`. Grouped by pool (worker→pool
//! from `run_meta.json`), the busy time landing in a bin / `(bin_width ×
//! workers_in_pool)` is the pool's mean busy fraction (0–1) over that bin. A unified
//! worker runs its whole replica in lockstep across its GPUs, so worker-busy ≡
//! GPU-busy and the fraction reads as a per-GPU utilization.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::Result;
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::io::{read_run_meta, read_worker_pools, SCHEMA_VERSION};
use crate::session::{col, collect, column_f64, register_cost_log, require_columns, COST_LOG_TABLE};

/// cost_log columns the utilization subject depends on (drift guard).
const COST_COLS: &[&str] = &["worker_id", "wall_start_ms", "total_time_ms"];

/// Equal-width time bins for the (single) fine view.
const FINE_BINS: usize = 200;

/// One worker iteration's busy interval: `(worker_id, start_ms, dur_ms)`.
type Iter = (u64, f64, f64);

pub async fn run_utilization(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    if !register_cost_log(ctx, log_dir).await? {
        let reason = "cost_log/ dir not found";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }
    require_columns(ctx, COST_LOG_TABLE, COST_COLS).await?;

    let (_num_gpus, gpu_name) = read_run_meta(log_dir).unwrap_or((1, String::new()));
    // worker→pool from run_meta; absent ⇒ every worker is pool 0 (single DP pool).
    let worker_pool: BTreeMap<u64, u64> = read_worker_pools(log_dir)
        .unwrap_or_default()
        .into_iter()
        .collect();

    let iters = collect_iters(ctx).await?;
    if iters.is_empty() {
        let reason = "cost_log has no iterations to bin";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }

    // Group busy intervals by pool, and tally each pool's worker roster (seeded from
    // run_meta so a pool's idle worker still counts toward its capacity, then unioned
    // with whatever cost_log actually shows).
    let mut pool_iters: BTreeMap<u64, Vec<(f64, f64)>> = BTreeMap::new();
    let mut pool_workers: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
    for (&w, &p) in &worker_pool {
        pool_workers.entry(p).or_default().insert(w);
    }
    for (w, start, dur) in &iters {
        let pool = worker_pool.get(w).copied().unwrap_or(0);
        pool_iters.entry(pool).or_default().push((*start, *dur));
        pool_workers.entry(pool).or_default().insert(*w);
    }

    let t_min = iters.iter().map(|i| i.1).fold(f64::INFINITY, f64::min);
    let t_max = iters.iter().map(|i| i.1 + i.2).fold(f64::NEG_INFINITY, f64::max);
    if !(t_max > t_min) {
        let reason = "cost_log busy span is zero (no positive iteration durations)";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }
    let span_ms = t_max - t_min;
    let n_bins = FINE_BINS;
    let bin_width = span_ms / n_bins as f64;
    let t_start: Vec<f64> = (0..n_bins).map(|b| t_min + b as f64 * bin_width).collect();
    let t_end: Vec<f64> = (0..n_bins).map(|b| t_min + (b + 1) as f64 * bin_width).collect();

    // Per-pool fine series + run-average. Sorted by pool id for a stable plot order.
    let mut series = Vec::new();
    let mut totals_per_pool = Vec::new();
    let mut avg = serde_json::Map::new();
    let mut overall_busy = 0.0;
    let mut overall_worker_ms = 0.0;
    for (pool, ivals) in &pool_iters {
        let n_workers = pool_workers.get(pool).map_or(1, BTreeSet::len).max(1) as f64;
        let util = bin_pool(ivals, t_min, bin_width, n_bins, n_workers);
        let busy: f64 = ivals.iter().map(|&(_, d)| d.max(0.0)).sum();
        let avg_util = busy / (span_ms * n_workers);
        let key = format!("pool_{pool}");
        series.push(json!({"key": key, "label": format!("Pool {pool}"), "util": util}));
        totals_per_pool.push(json!({
            "pool": pool,
            "avg_util": avg_util,
            "n_workers": n_workers as u64,
        }));
        avg.insert(key, json!(avg_util));
        overall_busy += busy;
        overall_worker_ms += span_ms * n_workers;
    }
    let overall_avg = if overall_worker_ms > 0.0 { overall_busy / overall_worker_ms } else { 0.0 };
    let num_workers: usize = pool_workers.values().map(BTreeSet::len).sum();

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "gpu_name": gpu_name,
            "num_pools": pool_iters.len(),
            "num_workers": num_workers,
            "bin_width_ms": bin_width,
            "span_ms": span_ms,
            "num_bins": n_bins,
        },
        "available": true,
        "totals": {
            "per_pool": totals_per_pool,
            "overall_avg": overall_avg,
        },
        "definitions": definitions(),
    });

    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "gpu_name": gpu_name,
            "unit": "fraction of pool workers busy (0-1)",
            // Per-pool run-average — the fine plot's dashed reference line.
            "avg": Value::Object(avg),
        },
        "t_start_ms": t_start,
        "t_end_ms": t_end,
        "series": series,
        "definitions": definitions(),
    });

    Ok((report, payload))
}

/// Ref's interval-spread binning (`analyze-rust/src/utilization.rs`): distribute each
/// busy interval's duration across the equal-width bins it overlaps — partial edge
/// bins directly, fully-covered middle bins via a `diff`/prefix-sum — then normalize
/// each bin by `capacity = bin_width × n_workers` to get the mean busy fraction (0–1).
fn bin_pool(
    intervals: &[(f64, f64)],
    t_min: f64,
    bin_width: f64,
    n_bins: usize,
    n_workers: f64,
) -> Vec<f64> {
    let mut util = vec![0.0f64; n_bins];
    let mut diff = vec![0.0f64; n_bins + 1];
    let last = n_bins as isize - 1;
    for &(start, dur) in intervals {
        if dur <= 0.0 {
            continue;
        }
        let end = start + dur;
        let sb = (((start - t_min) / bin_width).floor() as isize).clamp(0, last) as usize;
        let eb = (((end - t_min) / bin_width).floor() as isize).clamp(0, last) as usize;
        if sb == eb {
            util[sb] += dur;
            continue;
        }
        let first_edge = t_min + (sb as f64 + 1.0) * bin_width;
        let last_edge = t_min + eb as f64 * bin_width;
        util[sb] += first_edge - start;
        util[eb] += end - last_edge;
        if eb > sb + 1 {
            diff[sb + 1] += bin_width;
            diff[eb] -= bin_width;
        }
    }
    let mut carry = 0.0;
    for b in 0..n_bins {
        carry += diff[b];
        util[b] += carry;
    }
    let capacity = bin_width * n_workers;
    util.into_iter()
        .map(|v| if capacity > 0.0 { (v / capacity).clamp(0.0, 1.0) } else { 0.0 })
        .collect()
}

/// All `(worker_id, wall_start_ms, total_time_ms)` rows from `cost_log`.
async fn collect_iters(ctx: &SessionContext) -> Result<Vec<Iter>> {
    let batches = collect(
        ctx,
        "SELECT worker_id, wall_start_ms, total_time_ms FROM cost_log",
    )
    .await?;
    let mut out = Vec::new();
    for batch in &batches {
        // One typed pass per column beats a per-row 11-branch dispatch across the
        // whole cost_log (tens of millions of rows on an AFD run).
        let w = column_f64(col(batch, "worker_id")?)?;
        let s = column_f64(col(batch, "wall_start_ms")?)?;
        let d = column_f64(col(batch, "total_time_ms")?)?;
        for row in 0..batch.num_rows() {
            out.push((w[row] as u64, s[row], d[row]));
        }
    }
    Ok(out)
}

fn definitions() -> Value {
    json!({
        "scope": "all cost_log iterations, grouped by pool",
        "metric": "mean fraction of a pool's workers busy computing in each time bin",
        "bin": "one equal-width time segment (fine view, 200 bins over the run span)",
        "capacity": "bin_width × workers_in_pool — the denominator making util a 0-1 fraction",
        "avg_util": "pool busy time / (span × workers_in_pool) — the run-average reference",
        "note": "a unified worker runs its replica in lockstep across its GPUs, so \
                 worker-busy ≡ GPU-busy and the fraction reads as per-GPU utilization",
    })
}

fn unavailable(log_dir: &Path, reason: &str) -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {"log_dir": log_dir.display().to_string()},
        "available": false,
        "reason": reason,
        "definitions": definitions(),
    })
}

fn unavailable_payload(log_dir: &Path, reason: &str) -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {"log_dir": log_dir.display().to_string(), "available": false, "reason": reason},
        "t_start_ms": [],
        "t_end_ms": [],
        "series": [],
    })
}
