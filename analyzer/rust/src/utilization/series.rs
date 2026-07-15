//! Per-pool GPU utilization over time: the mean fraction of a pool's workers busy
//! computing, binned into equal-width time segments (ref's interval-spread).
//!
//! `cost_log` records each worker iteration's busy interval
//! `[wall_start_ms, wall_start_ms + total_time_ms]`. Grouped by pool (the
//! `(pool_tag, worker_id)` identity joins to `run_meta.json`), the busy time
//! landing in a bin / `(bin_width ×
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
    col, collect, column_f64, register_cost_log, require_columns, value_f64, value_string,
    COST_LOG_TABLE,
};

/// cost_log columns the utilization subject depends on (drift guard).
const COST_COLS: &[&str] = &["pool_tag", "worker_id", "wall_start_ms", "total_time_ms"];

/// Equal-width time bins for the (single) fine view.
const FINE_BINS: usize = 200;

type WorkerKey = (String, u64);

#[derive(Debug, PartialEq)]
struct WorkerBin {
    pool_tag: String,
    worker_id: u64,
    bin: usize,
    busy_ms: f64,
}

#[derive(Debug, PartialEq)]
struct PoolUsage {
    pool_id: u64,
    pool_tag: String,
    worker_ids: BTreeSet<u64>,
    bins: Vec<f64>,
}

pub async fn run_utilization(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    if !register_cost_log(ctx, log_dir).await? {
        let reason = "cost_log/ dir not found";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }
    require_columns(ctx, COST_LOG_TABLE, COST_COLS).await?;

    let (_num_gpus, gpu_name) = read_run_meta(log_dir).unwrap_or((1, String::new()));
    // Worker ids are per-pool. Keep the pool tag in the roster key so workers
    // such as AFD attn/0 and ffn/0 cannot overwrite each other.
    let pool_id_by_worker: BTreeMap<WorkerKey, u64> = read_worker_pools(log_dir)
        .unwrap_or_default()
        .into_iter()
        .map(|(pool_tag, worker_id, pool)| ((pool_tag, worker_id), pool))
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

    // Per-((pool_tag, worker), bin) busy-ms via a SQL GROUP BY — collapses tens of
    // millions of rows to num_workers × n_bins. Each iteration's whole duration
    // lands in the bin of its start time (no cross-bin interval spread; negligible
    // at ~run/200 bins vs ms-scale iterations).
    let worker_bins = collect_worker_bins(ctx, t_min, bin_width, n_bins).await?;

    // Seed from run_meta so a pool's idle worker still counts toward capacity,
    // then union the composite worker identities observed in cost_log. Pool tags
    // remain the aggregation key even when an old run lacks run_meta; the numeric
    // id is only a stable, backwards-compatible output label.
    let pool_usage = collect_pool_usage(&pool_id_by_worker, &worker_bins, n_bins);

    // Per-pool fine series + run-average. Sorted by pool id for a stable plot order.
    let mut series = Vec::new();
    let mut totals_per_pool = Vec::new();
    let mut avg = serde_json::Map::new();
    let mut overall_busy = 0.0;
    let mut overall_worker_ms = 0.0;
    for pool in &pool_usage {
        let n_workers = pool.worker_ids.len().max(1) as f64;
        // Do not clamp here: values above 1 expose overlapping/corrupt busy
        // intervals instead of turning an analyzer defect into a plausible plot.
        let util = utilization_values(&pool.bins, bin_width, n_workers);
        let busy: f64 = pool.bins.iter().sum();
        let avg_util = busy / (span_ms * n_workers);
        let key = format!("pool_{}", pool.pool_id);
        series.push(json!({
            "key": key,
            "label": format!("Pool {}", pool.pool_id),
            "pool_tag": pool.pool_tag,
            "util": util,
        }));
        totals_per_pool.push(json!({
            "pool": pool.pool_id,
            "pool_tag": pool.pool_tag,
            "avg_util": avg_util,
            "n_workers": n_workers as u64,
        }));
        avg.insert(key, json!(avg_util));
        overall_busy += busy;
        overall_worker_ms += span_ms * n_workers;
    }
    let overall_avg = if overall_worker_ms > 0.0 { overall_busy / overall_worker_ms } else { 0.0 };

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "gpu_name": gpu_name,
            "num_pools": pool_usage.len(),
            "num_workers": pool_usage.iter().map(|pool| pool.worker_ids.len()).sum::<usize>(),
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

/// Build the per-pool roster and busy bins without ever collapsing the
/// `(pool_tag, worker_id)` identity. Numeric pool ids come from run_meta when the
/// tag↔id relation is unambiguous; old runs without a tagged roster receive
/// deterministic, unique ids in tag order so the existing `pool_<id>` series-key
/// contract remains intact.
fn collect_pool_usage(
    pool_id_by_worker: &BTreeMap<WorkerKey, u64>,
    worker_bins: &[WorkerBin],
    n_bins: usize,
) -> Vec<PoolUsage> {
    let mut candidate_ids: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut pool_tags = BTreeSet::new();
    for ((pool_tag, _worker_id), pool_id) in pool_id_by_worker {
        pool_tags.insert(pool_tag.clone());
        candidate_ids
            .entry(pool_tag.clone())
            .or_default()
            .insert(*pool_id);
    }
    for worker_bin in worker_bins {
        pool_tags.insert(worker_bin.pool_tag.clone());
    }

    let mut pool_ids = BTreeMap::new();
    let mut used_ids = BTreeSet::new();
    for pool_tag in &pool_tags {
        let candidate = candidate_ids
            .get(pool_tag)
            .filter(|ids| ids.len() == 1)
            .and_then(|ids| ids.first().copied());
        if let Some(pool_id) = candidate.filter(|id| !used_ids.contains(id)) {
            pool_ids.insert(pool_tag.clone(), pool_id);
            used_ids.insert(pool_id);
        }
    }
    for pool_tag in &pool_tags {
        if pool_ids.contains_key(pool_tag) {
            continue;
        }
        let pool_id = (0..).find(|id| !used_ids.contains(id)).unwrap();
        pool_ids.insert(pool_tag.clone(), pool_id);
        used_ids.insert(pool_id);
    }

    let mut usage_by_tag: BTreeMap<String, PoolUsage> = pool_tags
        .into_iter()
        .map(|pool_tag| {
            let pool_id = pool_ids[&pool_tag];
            (
                pool_tag.clone(),
                PoolUsage {
                    pool_id,
                    pool_tag,
                    worker_ids: BTreeSet::new(),
                    bins: vec![0.0; n_bins],
                },
            )
        })
        .collect();
    for (pool_tag, worker_id) in pool_id_by_worker.keys() {
        usage_by_tag
            .get_mut(pool_tag)
            .expect("run_meta pool tag was seeded")
            .worker_ids
            .insert(*worker_id);
    }
    for worker_bin in worker_bins {
        let pool = usage_by_tag
            .get_mut(&worker_bin.pool_tag)
            .expect("cost_log pool tag was seeded");
        pool.worker_ids.insert(worker_bin.worker_id);
        pool.bins[worker_bin.bin] += worker_bin.busy_ms;
    }

    let mut usage: Vec<PoolUsage> = usage_by_tag.into_values().collect();
    usage.sort_by(|left, right| {
        left.pool_id
            .cmp(&right.pool_id)
            .then_with(|| left.pool_tag.cmp(&right.pool_tag))
    });
    usage
}

fn utilization_values(bins: &[f64], bin_width: f64, n_workers: f64) -> Vec<f64> {
    let capacity = bin_width * n_workers;
    bins.iter()
        .map(|busy| if capacity > 0.0 { busy / capacity } else { 0.0 })
        .collect()
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

/// Per-((pool_tag, worker), bin) busy-ms via a SQL GROUP BY: `bin` = truncated
/// bin index of the iteration's start time (`CAST(... AS BIGINT)` = floor for a
/// non-negative offset), `busy` = `SUM(total_time_ms)`. Returns num_workers ×
/// n_bins rows at most, with bin indices clamped into `[0, n_bins)`.
async fn collect_worker_bins(
    ctx: &SessionContext,
    t_min: f64,
    bin_width: f64,
    n_bins: usize,
) -> Result<Vec<WorkerBin>> {
    let sql = format!(
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, worker_id, \
                CAST((wall_start_ms - {t_min}) / {bin_width} AS BIGINT) AS bin, \
                SUM(total_time_ms) AS busy \
         FROM cost_log GROUP BY CAST(pool_tag AS VARCHAR), worker_id, bin"
    );
    let batches = collect(ctx, &sql).await?;
    let last = n_bins as isize - 1;
    let mut out = Vec::new();
    for batch in &batches {
        let pool_tag = col(batch, "pool_tag")?;
        let w = column_f64(col(batch, "worker_id")?)?;
        let bin = column_f64(col(batch, "bin")?)?;
        let busy = column_f64(col(batch, "busy")?)?;
        for row in 0..batch.num_rows() {
            let bi = (bin[row] as isize).clamp(0, last) as usize;
            out.push(WorkerBin {
                pool_tag: value_string(pool_tag, row)?,
                worker_id: w[row] as u64,
                bin: bi,
                busy_ms: busy[row],
            });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_worker_ids_across_pools_stay_distinct_and_visible() {
        let pool_id_by_worker =
            BTreeMap::from([(("attn".to_owned(), 0), 0), (("ffn".to_owned(), 0), 1)]);
        let worker_bins = vec![
            WorkerBin {
                pool_tag: "attn".to_owned(),
                worker_id: 0,
                bin: 0,
                busy_ms: 6.0,
            },
            WorkerBin {
                pool_tag: "ffn".to_owned(),
                worker_id: 0,
                bin: 0,
                busy_ms: 12.0,
            },
        ];

        let usage = collect_pool_usage(&pool_id_by_worker, &worker_bins, 1);

        assert_eq!(usage.len(), 2);
        assert_eq!(
            usage
                .iter()
                .map(|pool| pool.worker_ids.len())
                .sum::<usize>(),
            2
        );
        assert_eq!(usage[0].pool_id, 0);
        assert_eq!(usage[0].pool_tag, "attn");
        assert_eq!(usage[0].bins, vec![6.0]);
        assert_eq!(usage[1].pool_id, 1);
        assert_eq!(usage[1].pool_tag, "ffn");
        assert_eq!(usage[1].bins, vec![12.0]);

        // The analyzer must expose an impossible overlap instead of hiding it
        // behind an output clamp; this makes future accounting bugs diagnosable.
        let ffn_util = utilization_values(&usage[1].bins, 10.0, 1.0);
        assert!((ffn_util[0] - 1.2).abs() < 1e-12);
    }
}
