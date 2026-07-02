//! Per-invocation batch composition over time, **grouped by pool**.
//!
//! Each `cost_log` row is one kernel invocation (`worker × iter × batch × layer ×
//! section`); its `batch_tokens` / `prefill_tokens` / `decode_request_count` are
//! summed across the row's `groups` — the groups are the DP shards ONE invocation
//! is split across, so their sum is that invocation's logical batch size (a batch
//! GPUs then split, not several batches to keep apart).
//!
//! Grain is per-invocation (layer/section), NOT deduped: on AFD a worker logs one
//! row per layer/section, and those rows are genuinely different work (the FFN is a
//! layer-pipeline carrying different micro-batches on different layers), so keeping
//! them all makes `call` counts line up across pools.
//!
//! Reported **per pool** because the pools' batch scales differ and don't belong in
//! one distribution: attn workers are DP shards (each `batch_tokens` is that shard's
//! local slice), the ffn pool is the aggregator (each row is the summed batch). An
//! iter-wise deployment (unified/PD) is the degenerate case: one pool, one row per
//! iteration.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use arrow_array::{Array, ListArray, StringArray, StructArray};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::cdf::{clean_nonnegative_sorted, stats, MetricStats};
use crate::io::SCHEMA_VERSION;
use crate::session::{col, collect, column_f64, register_cost_log, require_columns, COST_LOG_TABLE};

/// cost_log columns the batch subject depends on (drift guard).
const COST_COLS: &[&str] = &["pool_tag", "wall_start_ms", "groups"];

/// Cap on scatter points in the payload, per pool. The run can have millions of
/// invocations; the scatter is an even time-downsample to this many (stats stay
/// over ALL rows).
const MAX_SCATTER_POINTS: usize = 4000;

/// One invocation: `(wall_start_ms, batch_tokens, prefill_tokens, decode_request_count)`.
type Row = (f64, f64, f64, f64);

pub async fn run_batch(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    if !register_cost_log(ctx, log_dir).await? {
        let reason = "cost_log/ dir not found";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }
    require_columns(ctx, COST_LOG_TABLE, COST_COLS).await?;

    let by_pool = collect_batches(ctx).await?;
    if by_pool.is_empty() {
        let reason = "cost_log has no invocations";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }
    let total_calls: usize = by_pool.values().map(Vec::len).sum();

    let mut report_pools = Vec::new();
    let mut payload_pools = Vec::new();
    for (pool, mut rows) in by_pool {
        // Time-order so the even-stride scatter downsample is uniform over sim time
        // (cost_log is iter-ordered per worker; sorting handles the multi-worker case).
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Distribution stats over EVERY invocation in this pool (per field).
        let bt: Vec<f64> = rows.iter().map(|r| r.1).collect();
        let pt: Vec<f64> = rows.iter().map(|r| r.2).collect();
        let dc: Vec<f64> = rows.iter().map(|r| r.3).collect();
        let bt_stats = stats(&clean_nonnegative_sorted(&bt));
        let pt_stats = stats(&clean_nonnegative_sorted(&pt));
        let dc_stats = stats(&clean_nonnegative_sorted(&dc));

        let (t_s, bt_s, pt_s, dc_s) = downsample_scatter(&rows);

        report_pools.push(json!({
            "pool": pool,
            "num_calls": rows.len(),
            "metrics": {
                "batch_tokens": bt_stats,
                "prefill_tokens": pt_stats,
                "decode_request_count": dc_stats,
            },
        }));
        payload_pools.push(json!({
            "pool": pool,
            "num_calls": rows.len(),
            "plotted_points": t_s.len(),
            // Per-field pool average — the scatter's corner-box reference.
            "avg": {
                "batch_tokens": mean(&bt_stats),
                "prefill_tokens": mean(&pt_stats),
                "decode_request_count": mean(&dc_stats),
            },
            "time_ms": t_s,
            "series": [
                {"key": "batch_tokens", "label": "Batch tokens", "values": bt_s},
                {"key": "prefill_tokens", "label": "Prefill tokens", "values": pt_s},
                {"key": "decode_request_count", "label": "Decode requests", "values": dc_s},
            ],
        }));
    }

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "num_calls": total_calls,
        },
        "available": true,
        "pools": report_pools,
        "definitions": definitions(),
    });

    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "num_calls": total_calls,
        },
        "available": true,
        "pools": payload_pools,
        "definitions": definitions(),
    });

    Ok((report, payload))
}

fn mean(s: &MetricStats) -> Value {
    json!(s.mean)
}

/// Pull `(wall_start_ms, batch_tokens, prefill_tokens, decode_request_count)` per
/// cost_log row, summing each count across the row's `groups` via the list offsets
/// in one typed pass per column (see [`column_f64`]), and bucket by `pool_tag`. Each
/// row is one invocation; the per-row group sum is that invocation's logical batch.
async fn collect_batches(ctx: &SessionContext) -> Result<BTreeMap<String, Vec<Row>>> {
    let batches = collect(
        ctx,
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, wall_start_ms, groups FROM cost_log",
    )
    .await?;
    let mut by_pool: BTreeMap<String, Vec<Row>> = BTreeMap::new();
    for batch in &batches {
        let pool_arr = col(batch, "pool_tag")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("`pool_tag` is not a Utf8 array"))?;
        let ws = column_f64(col(batch, "wall_start_ms")?)?;
        let list = col(batch, "groups")?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow!("`groups` is not a List array"))?;
        let gs = list
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| anyhow!("`groups` elements are not Structs"))?;
        let offsets = list.value_offsets();
        let field = |name: &str| -> Result<Vec<f64>> {
            column_f64(
                gs.column_by_name(name)
                    .ok_or_else(|| anyhow!("`groups` struct missing field `{name}`"))?,
            )
        };
        let (bt, pt, dc) = (
            field("batch_tokens")?,
            field("prefill_tokens")?,
            field("decode_request_count")?,
        );
        // Sum each field over a row's groups (`offsets[i]..offsets[i+1]`) — the DP
        // shards the one invocation is split across.
        let row_sum = |v: &[f64], i: usize| -> f64 {
            v[offsets[i] as usize..offsets[i + 1] as usize].iter().sum()
        };
        for i in 0..batch.num_rows() {
            by_pool
                .entry(pool_arr.value(i).to_string())
                .or_default()
                .push((ws[i], row_sum(&bt, i), row_sum(&pt, i), row_sum(&dc, i)));
        }
    }
    Ok(by_pool)
}

/// Even-stride downsample of the time-ordered rows to ≤`MAX_SCATTER_POINTS` per
/// series (keeps the time spread; stats are computed separately over all rows).
fn downsample_scatter(rows: &[Row]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = rows.len();
    let stride = n.div_ceil(MAX_SCATTER_POINTS).max(1);
    let (mut t, mut bt, mut pt, mut dc) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut i = 0;
    while i < n {
        let r = rows[i];
        t.push(r.0);
        bt.push(r.1);
        pt.push(r.2);
        dc.push(r.3);
        i += stride;
    }
    (t, bt, pt, dc)
}

fn definitions() -> Value {
    json!({
        "scope": "every cost_log row = one kernel invocation (worker × iter × batch × layer × \
                  section), grouped by pool",
        "grain": "per-invocation (layer/section), NOT deduped — AFD logs per layer/section and \
                  those are genuinely different work, so `call` counts align across pools",
        "batch_tokens": "tokens in the invocation's batch, summed over its `groups` (the DP shards \
                         one invocation is split across) = the logical batch size",
        "prefill_tokens": "tokens being prefilled in the invocation",
        "decode_request_count": "number of decode requests in the invocation",
        "pools": "reported separately because scales differ: attn workers are DP shards (local \
                  slice), the ffn pool is the aggregator (summed batch)",
        "stats_vs_scatter": "report metrics are over ALL invocations in the pool; the payload \
                             scatter is an even time-downsample to <=4000 points per pool",
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
        "pools": [],
    })
}
