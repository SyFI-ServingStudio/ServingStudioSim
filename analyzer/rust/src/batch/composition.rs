//! Per-batch composition over time: each `cost_log` iteration's batch / prefill /
//! decode token counts, emitted as a scatter over sim time (the payload) plus the
//! run-wide distribution stats (the report).
//!
//! `cost_log` logs these counts inside the per-HP-group `groups` struct list. We
//! sum across that list per iteration. Today there is exactly ONE group (unified
//! dense asserts a single HP group), so the sum == `groups[0]`. When EP/HP
//! multi-group lands the correct reduction depends on group semantics — **sum** for
//! partition-style groups (different token subsets), **pick-one** for replicate-style
//! HP groups (same tokens, split heads) — so revisit the reduction in `read_field`
//! then. The worker/pool axis extends cleanly without a redefine: a future
//! multi-worker run groups/colors by pool via `run_meta.json` (as `utilization` does).

use std::path::Path;

use anyhow::{anyhow, Result};
use arrow_array::{Array, ListArray, StructArray};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::cdf::{clean_nonnegative_sorted, stats, MetricStats};
use crate::io::SCHEMA_VERSION;
use crate::session::{col, collect, register_cost_log, require_columns, value_f64, COST_LOG_TABLE};

/// cost_log columns the batch subject depends on (drift guard).
const COST_COLS: &[&str] = &["wall_start_ms", "groups"];

/// Cap on scatter points in the payload. The run can have ~millions of iterations;
/// the scatter is an even time-downsample to this many (stats stay over ALL rows).
const MAX_SCATTER_POINTS: usize = 4000;

/// One iteration: `(wall_start_ms, batch_tokens, prefill_tokens, decode_request_count)`.
type Row = (f64, f64, f64, f64);

pub async fn run_batch(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    if !register_cost_log(ctx, log_dir).await? {
        let reason = "cost_log/ dir not found";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }
    require_columns(ctx, COST_LOG_TABLE, COST_COLS).await?;

    let mut rows = collect_batches(ctx).await?;
    if rows.is_empty() {
        let reason = "cost_log has no iterations";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }
    // Time-order so the even-stride scatter downsample is uniform over sim time
    // (cost_log is iter-ordered per worker; sorting handles the multi-worker case).
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Distribution stats over EVERY iteration (per field) — reuse the cdf kernels.
    let bt: Vec<f64> = rows.iter().map(|r| r.1).collect();
    let pt: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let dc: Vec<f64> = rows.iter().map(|r| r.3).collect();
    let bt_stats = stats(&clean_nonnegative_sorted(&bt));
    let pt_stats = stats(&clean_nonnegative_sorted(&pt));
    let dc_stats = stats(&clean_nonnegative_sorted(&dc));

    let (t_s, bt_s, pt_s, dc_s) = downsample_scatter(&rows);

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "num_iterations": rows.len(),
        },
        "available": true,
        "metrics": {
            "batch_tokens": bt_stats,
            "prefill_tokens": pt_stats,
            "decode_request_count": dc_stats,
        },
        "definitions": definitions(),
    });

    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "num_iterations": rows.len(),
            "plotted_points": t_s.len(),
            // Run-average per field — the scatter's corner-box reference.
            "avg": {
                "batch_tokens": mean(&bt_stats),
                "prefill_tokens": mean(&pt_stats),
                "decode_request_count": mean(&dc_stats),
            },
        },
        "time_ms": t_s,
        "series": [
            {"key": "batch_tokens", "label": "Batch tokens", "values": bt_s},
            {"key": "prefill_tokens", "label": "Prefill tokens", "values": pt_s},
            {"key": "decode_request_count", "label": "Decode requests", "values": dc_s},
        ],
        "definitions": definitions(),
    });

    Ok((report, payload))
}

fn mean(s: &MetricStats) -> Value {
    json!(s.mean)
}

/// Pull `(wall_start_ms, batch_tokens, prefill_tokens, decode_request_count)` per
/// iteration, summing each count across the iteration's `groups` struct list.
async fn collect_batches(ctx: &SessionContext) -> Result<Vec<Row>> {
    let batches = collect(ctx, "SELECT wall_start_ms, groups FROM cost_log").await?;
    let mut out = Vec::new();
    for batch in &batches {
        let wall = col(batch, "wall_start_ms")?;
        let groups = col(batch, "groups")?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow!("`groups` is not a List array"))?;
        for row in 0..batch.num_rows() {
            let t = value_f64(wall, row)?;
            if groups.is_null(row) {
                out.push((t, 0.0, 0.0, 0.0));
                continue;
            }
            let g = groups.value(row);
            let gs = g
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| anyhow!("`groups` elements are not Structs"))?;
            out.push((
                t,
                sum_field(gs, "batch_tokens")?,
                sum_field(gs, "prefill_tokens")?,
                sum_field(gs, "decode_request_count")?,
            ));
        }
    }
    Ok(out)
}

/// Sum one struct field across all entries of an iteration's `groups` list. See
/// the module note on why this is a sum today and what changes at multi-group.
fn sum_field(structs: &StructArray, field: &str) -> Result<f64> {
    let arr = structs
        .column_by_name(field)
        .ok_or_else(|| anyhow!("`groups` struct missing field `{field}`"))?;
    let mut total = 0.0;
    for i in 0..arr.len() {
        total += value_f64(arr, i)?;
    }
    Ok(total)
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
        "scope": "every cost_log iteration (one batch per worker tick)",
        "batch_tokens": "total tokens processed in the batch (prefill + decode)",
        "prefill_tokens": "tokens being prefilled in the batch",
        "decode_request_count": "number of decode requests in the batch (= decode tokens that iteration)",
        "stats_vs_scatter": "report metrics are over ALL iterations; the payload scatter is an \
                             even time-downsample to <=4000 points",
        "groups_note": "counts are summed across the per-HP-group `groups` list; one group today \
                        (unified dense) — revisit the reduction at EP/HP multi-group",
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
        "time_ms": [],
        "series": [],
    })
}
