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
use arrow_array::Array;
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::io::{read_run_meta, read_worker_pools, SCHEMA_VERSION};
use crate::session::{
    col, collect, column_f64, register_cost_log, require_columns, value_f64, COST_LOG_TABLE,
};

/// cost_log columns the utilization subject depends on (drift guard).
const COST_COLS: &[&str] = &["worker_id", "wall_start_ms", "total_time_ms"];

/// Equal-width time bins for the (single) fine view.
const FINE_BINS: usize = 200;

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

    // Run span from a single SQL row so we can size the bins without pulling every
    // iteration into Rust.
    let (t_min, t_max) = match query_span(ctx).await? {
        Some(span) => span,
        None => {
            let reason = "cost_log has no iterations to bin";
            return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
        }
    };
    if !(t_max > t_min) {
        let reason = "cost_log busy span is zero (no positive iteration durations)";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }
    let span_ms = t_max - t_min;
    let n_bins = FINE_BINS;
    let bin_width = span_ms / n_bins as f64;
    let t_start: Vec<f64> = (0..n_bins).map(|b| t_min + b as f64 * bin_width).collect();
    let t_end: Vec<f64> = (0..n_bins).map(|b| t_min + (b + 1) as f64 * bin_width).collect();

    // Per-(worker, bin) busy-ms via a SQL GROUP BY — collapses tens of millions of
    // rows to num_workers × n_bins. Each iteration's whole duration lands in the bin
    // of its start time (no cross-bin interval spread; negligible at ~run/200 bins vs
    // ms-scale iterations).
    let worker_bins = collect_worker_bins(ctx, t_min, bin_width, n_bins).await?;

    // Accumulate per pool (worker→pool from run_meta), seeding rosters from run_meta so
    // a pool's idle worker still counts toward its capacity, then unioning observed.
    let mut pool_bins: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
    let mut pool_workers: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
    for (&w, &p) in &worker_pool {
        pool_workers.entry(p).or_default().insert(w);
    }
    for &(w, bin, busy) in &worker_bins {
        let pool = worker_pool.get(&w).copied().unwrap_or(0);
        pool_bins.entry(pool).or_insert_with(|| vec![0.0; n_bins])[bin] += busy;
        pool_workers.entry(pool).or_default().insert(w);
    }

    // Per-pool fine series + run-average. Sorted by pool id for a stable plot order.
    let mut series = Vec::new();
    let mut totals_per_pool = Vec::new();
    let mut avg = serde_json::Map::new();
    let mut overall_busy = 0.0;
    let mut overall_worker_ms = 0.0;
    for (pool, bins) in &pool_bins {
        let n_workers = pool_workers.get(pool).map_or(1, BTreeSet::len).max(1) as f64;
        let capacity = bin_width * n_workers;
        let util: Vec<f64> = bins
            .iter()
            .map(|&v| if capacity > 0.0 { (v / capacity).clamp(0.0, 1.0) } else { 0.0 })
            .collect();
        let busy: f64 = bins.iter().sum();
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
            "num_pools": pool_bins.len(),
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

/// Run busy span `(t_min, t_max)` from one SQL row — `MIN(wall_start_ms)` and
/// `MAX(wall_start_ms + total_time_ms)`. `None` when cost_log is empty (aggregates
/// return a null row).
async fn query_span(ctx: &SessionContext) -> Result<Option<(f64, f64)>> {
    let batches = collect(
        ctx,
        "SELECT MIN(wall_start_ms) AS t0, MAX(wall_start_ms + total_time_ms) AS t1 FROM cost_log",
    )
    .await?;
    let b = match batches.first() {
        Some(b) if b.num_rows() > 0 => b,
        _ => return Ok(None),
    };
    let (t0, t1) = (col(b, "t0")?, col(b, "t1")?);
    if t0.is_null(0) || t1.is_null(0) {
        return Ok(None);
    }
    Ok(Some((value_f64(t0, 0)?, value_f64(t1, 0)?)))
}

/// Per-(worker, bin) busy-ms via a SQL GROUP BY: `bin` = truncated bin index of the
/// iteration's start time (`CAST(... AS BIGINT)` = floor for a non-negative offset),
/// `busy` = `SUM(total_time_ms)`. Returns num_workers × n_bins rows at most, clamped
/// into `[0, n_bins)`.
async fn collect_worker_bins(
    ctx: &SessionContext,
    t_min: f64,
    bin_width: f64,
    n_bins: usize,
) -> Result<Vec<(u64, usize, f64)>> {
    let sql = format!(
        "SELECT worker_id, \
                CAST((wall_start_ms - {t_min}) / {bin_width} AS BIGINT) AS bin, \
                SUM(total_time_ms) AS busy \
         FROM cost_log GROUP BY worker_id, bin"
    );
    let batches = collect(ctx, &sql).await?;
    let last = n_bins as isize - 1;
    let mut out = Vec::new();
    for batch in &batches {
        let w = column_f64(col(batch, "worker_id")?)?;
        let bin = column_f64(col(batch, "bin")?)?;
        let busy = column_f64(col(batch, "busy")?)?;
        for row in 0..batch.num_rows() {
            let bi = (bin[row] as isize).clamp(0, last) as usize;
            out.push((w[row] as u64, bi, busy[row]));
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
