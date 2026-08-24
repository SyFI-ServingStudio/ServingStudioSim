//! Per-pool KV-cache occupancy over time: how full each KV pool runs, in tokens
//! and as a fraction of its static capacity, binned into equal-width time segments.
//!
//! The sim's `kv_snapshot` stream records, per KV-bearing worker each iteration
//! (throttled — a running-max over a stride window), four token levels:
//! `active_kv` (resident/committed KV, the *current* occupancy),
//! `retained_prefix_kv` (the prefix-cache component captured at the same active
//! peak), `projected_peak` (the admission gate's *future* estimate — each admitted
//! request's full decode horizon), and `promised_kv` (reserved-but-not-resident
//! footprint).
//!
//! A pool's DP shards are symmetric independent KV pools of equal capacity, so the
//! subject keeps BOTH views: each shard's own binned series (the per-server "shadow"
//! lines) AND the across-shard `mean`/`min`/`max` per bin (the standout aggregate
//! that also exposes shard imbalance — a wide min↔max band means skew). Everything
//! is per `(pool_tag, group_id)`; `pool_tag` matches the `run_meta.json` (v3) key so
//! the capacity divisor joins exactly even when `worker_id` collides across pools.
//!
//! `projected_peak` is deliberately NOT clamped to ≤100%: it exceeding capacity is
//! the meaningful signal that the admitted horizon outruns the pool (imminent
//! preemption / admission stall), which a clamp would hide.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use arrow_array::{Array, StringArray};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::io::{read_kv_capacities, SCHEMA_VERSION};
use crate::session::{
    col, collect, column_f64, register_kv_snapshot, require_columns, KV_SNAPSHOT_TABLE,
};

/// kv_snapshot columns the occupancy subject depends on (drift guard).
const KV_COLS: &[&str] = &[
    "pool_tag",
    "worker_id",
    "group_id",
    "time_ms",
    "active_kv",
    "projected_peak",
    "promised_kv",
];

/// Equal-width time bins for the occupancy view (matches the utilization subject).
const FINE_BINS: usize = 200;

/// One normalized snapshot row. Named fields keep the four related token measures
/// from becoming error-prone tuple offsets as the raw stream evolves.
struct Row {
    pool_tag: String,
    group_id: u64,
    worker_id: u64,
    time_ms: f64,
    active_kv: f64,
    retained_prefix_kv: f64,
    projected_peak: f64,
    promised_kv: f64,
}

/// Per-bin across-shard aggregate of one metric: `mean`, `min`, `max`.
struct Agg {
    mean: Vec<f64>,
    min: Vec<f64>,
    max: Vec<f64>,
}

pub async fn run_kv_occupancy(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    if !register_kv_snapshot(ctx, log_dir).await? {
        let reason = "kv_snapshot/ dir not found (KV logging off or no KV pool)";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }
    require_columns(ctx, KV_SNAPSHOT_TABLE, KV_COLS).await?;
    // `retained_prefix_kv` was added after the original stream. Keep old runs
    // analyzable, but surface that their zero-filled breakdown is unavailable so
    // consumers never mistake compatibility data for a measured zero.
    let has_retained_prefix_breakdown =
        has_column(ctx, KV_SNAPSHOT_TABLE, "retained_prefix_kv").await?;

    // (pool_tag, group_id) → capacity_tokens. Shards of a pool are symmetric, so
    // one capacity per pool×group; take the max across shards defensively (they
    // should be identical). Absent ⇒ raw-token-only mode (no pct reference).
    let mut capacity_by_pool: BTreeMap<(String, u64), u64> = BTreeMap::new();
    if let Some(capacities) = read_kv_capacities(log_dir) {
        for (pool_tag, _worker, group_id, capacity) in capacities {
            let entry = capacity_by_pool.entry((pool_tag, group_id)).or_insert(0);
            *entry = (*entry).max(capacity);
        }
    }

    let rows = collect_rows(ctx, has_retained_prefix_breakdown).await?;
    if rows.is_empty() {
        let reason = "kv_snapshot has no rows to bin";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }

    let t_min = rows
        .iter()
        .map(|row| row.time_ms)
        .fold(f64::INFINITY, f64::min);
    let t_max = rows
        .iter()
        .map(|row| row.time_ms)
        .fold(f64::NEG_INFINITY, f64::max);
    // A single-sample run (t_max == t_min) still deserves a one-bin series; widen a
    // degenerate span to 1ms so bin math stays finite.
    let span_ms = (t_max - t_min).max(1.0);
    let n_bins = FINE_BINS;
    #[allow(
        clippy::cast_precision_loss,
        reason = "n_bins is the FINE_BINS constant (200), far below 2^53"
    )]
    let bin_width = span_ms / n_bins as f64;
    #[allow(
        clippy::cast_precision_loss,
        reason = "bin_index is bounded by n_bins == FINE_BINS (200), far below 2^53"
    )]
    let t_start: Vec<f64> = (0..n_bins)
        .map(|bin_index| t_min + bin_index as f64 * bin_width)
        .collect();
    #[allow(
        clippy::cast_precision_loss,
        reason = "bin_index is bounded by n_bins == FINE_BINS (200), far below 2^53"
    )]
    let t_end: Vec<f64> = (0..n_bins)
        .map(|bin_index| t_min + (bin_index + 1) as f64 * bin_width)
        .collect();

    // Group rows by (pool_tag, group_id) → one plotted series; count groups per
    // pool_tag so a single-group pool gets a clean label (no `· g0` noise).
    let mut by_series: BTreeMap<(String, u64), Vec<&Row>> = BTreeMap::new();
    for row in &rows {
        by_series
            .entry((row.pool_tag.clone(), row.group_id))
            .or_default()
            .push(row);
    }
    let mut groups_per_tag: BTreeMap<String, usize> = BTreeMap::new();
    for (tag, _group_id) in by_series.keys() {
        *groups_per_tag.entry(tag.clone()).or_default() += 1;
    }

    let mut series = Vec::new();
    let mut totals = Vec::new();
    for ((pool_tag, group_id), series_rows) in &by_series {
        let capacity = capacity_by_pool
            .get(&(pool_tag.clone(), *group_id))
            .copied();

        // Per-shard binned step series (each shard samples ~once per stride window,
        // so `bin_mean` = that shard's held level in the bin), then the across-shard
        // mean/min/max per bin.
        let mut by_worker: BTreeMap<u64, Vec<&Row>> = BTreeMap::new();
        for row in series_rows {
            by_worker.entry(row.worker_id).or_default().push(row);
        }
        let mut workers_json = Vec::new();
        let mut active_by_worker: Vec<Vec<f64>> = Vec::new();
        let mut retained_prefix_by_worker: Vec<Vec<f64>> = Vec::new();
        let mut projected_by_worker: Vec<Vec<f64>> = Vec::new();
        let mut promised_by_worker: Vec<Vec<f64>> = Vec::new();
        for (worker_id, worker_rows) in &by_worker {
            let active = bin_mean(worker_rows, |row| row.active_kv, t_min, bin_width, n_bins);
            let retained_prefix = bin_mean(
                worker_rows,
                |row| row.retained_prefix_kv,
                t_min,
                bin_width,
                n_bins,
            );
            let projected = bin_mean(
                worker_rows,
                |row| row.projected_peak,
                t_min,
                bin_width,
                n_bins,
            );
            let promised = bin_mean(worker_rows, |row| row.promised_kv, t_min, bin_width, n_bins);
            workers_json.push(json!({
                "worker_id": worker_id,
                "active_tokens": active,
                "retained_prefix_tokens": retained_prefix,
                "projected_tokens": projected,
                "promised_tokens": promised,
            }));
            active_by_worker.push(active);
            retained_prefix_by_worker.push(retained_prefix);
            projected_by_worker.push(projected);
            promised_by_worker.push(promised);
        }
        let active = agg_across(&active_by_worker, n_bins);
        let retained_prefix = agg_across(&retained_prefix_by_worker, n_bins);
        let projected = agg_across(&projected_by_worker, n_bins);
        let promised = agg_across(&promised_by_worker, n_bins);

        let peak = |values: &[f64]| values.iter().copied().fold(0.0, f64::max);
        let peak_active_mean = peak(&active.mean);
        let peak_active_max = peak(&active.max);
        let peak_retained_prefix_mean = peak(&retained_prefix.mean);
        let peak_retained_prefix_max = peak(&retained_prefix.max);
        let peak_projected_mean = peak(&projected.mean);
        let peak_projected_max = peak(&projected.max);
        let peak_promised_max = peak(&promised.max);
        let active_bin_count: usize = active.mean.iter().filter(|value| **value > 0.0).count();
        #[allow(
            clippy::cast_precision_loss,
            reason = "active_bin_count is bounded by n_bins == FINE_BINS (200), far below 2^53"
        )]
        let mean_active = if active_bin_count > 0 {
            active.mean.iter().sum::<f64>() / active_bin_count as f64
        } else {
            0.0
        };

        let key = format!("{pool_tag}/g{group_id}");
        let label = if groups_per_tag.get(pool_tag).copied().unwrap_or(1) > 1 {
            format!("{pool_tag} · g{group_id}")
        } else {
            pool_tag.clone()
        };
        #[allow(
            clippy::cast_precision_loss,
            reason = "capacity is a per-shard KV token capacity derived from realistic GPU memory \
                      sizes, far below 2^53 tokens"
        )]
        let percentage = |tokens: f64| {
            capacity.map(|capacity| {
                if capacity > 0 {
                    tokens / capacity as f64
                } else {
                    0.0
                }
            })
        };

        series.push(json!({
            "key": key,
            "label": label,
            "pool_tag": pool_tag,
            "group_id": group_id,
            "capacity_tokens": capacity,
            "n_workers": by_worker.len(),
            // Per-shard "shadow" lines.
            "workers": workers_json,
            // Across-shard standout aggregates (mean/min/max per bin).
            "active": {"mean": active.mean, "min": active.min, "max": active.max},
            "retained_prefix": {
                "mean": retained_prefix.mean,
                "min": retained_prefix.min,
                "max": retained_prefix.max,
            },
            "projected": {"mean": projected.mean, "min": projected.min, "max": projected.max},
            "promised": {"mean": promised.mean, "min": promised.min, "max": promised.max},
        }));
        totals.push(json!({
            "pool_tag": pool_tag,
            "group_id": group_id,
            "capacity_tokens": capacity,
            "n_workers": by_worker.len(),
            "peak_active_mean_tokens": peak_active_mean,
            "peak_active_mean_pct": percentage(peak_active_mean),
            // Worst individual shard — the imbalance-aware headroom number.
            "peak_active_max_tokens": peak_active_max,
            "peak_active_max_pct": percentage(peak_active_max),
            "mean_active_pct": percentage(mean_active),
            "peak_retained_prefix_mean_tokens": peak_retained_prefix_mean,
            "peak_retained_prefix_mean_pct": percentage(peak_retained_prefix_mean),
            "peak_retained_prefix_max_tokens": peak_retained_prefix_max,
            "peak_retained_prefix_max_pct": percentage(peak_retained_prefix_max),
            "peak_projected_mean_tokens": peak_projected_mean,
            "peak_projected_mean_pct": percentage(peak_projected_mean),
            "peak_projected_max_pct": percentage(peak_projected_max),
            "peak_promised_max_tokens": peak_promised_max,
            "peak_promised_max_pct": percentage(peak_promised_max),
        }));
    }

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "num_series": series.len(),
            "num_bins": n_bins,
            "bin_width_ms": bin_width,
            "span_ms": span_ms,
            "has_capacity": !capacity_by_pool.is_empty(),
            "has_retained_prefix_breakdown": has_retained_prefix_breakdown,
        },
        "available": true,
        "totals": {
            "per_series": totals,
        },
        "definitions": definitions(),
    });

    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "unit": "KV tokens (per shard); fraction = tokens / capacity_tokens",
            "has_capacity": !capacity_by_pool.is_empty(),
            "has_retained_prefix_breakdown": has_retained_prefix_breakdown,
        },
        "t_start_ms": t_start,
        "t_end_ms": t_end,
        "series": series,
        "definitions": definitions(),
    });

    Ok((report, payload))
}

/// Across-shard aggregate per bin: `mean` (average of the shards' held levels),
/// `min`/`max` (the band edges that expose imbalance). With one shard all three
/// coincide.
fn agg_across(per_worker: &[Vec<f64>], n_bins: usize) -> Agg {
    let mut mean = vec![0.0f64; n_bins];
    let mut min = vec![0.0f64; n_bins];
    let mut max = vec![0.0f64; n_bins];
    #[allow(
        clippy::cast_precision_loss,
        reason = "per_worker.len() is the shard count for one KV pool, bounded by a realistic \
                  cluster topology and far below 2^53"
    )]
    let worker_count = per_worker.len().max(1) as f64;
    for bin_index in 0..n_bins {
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        let mut sum = 0.0;
        for worker_values in per_worker {
            let value = worker_values[bin_index];
            sum += value;
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
        mean[bin_index] = sum / worker_count;
        min[bin_index] = if minimum.is_finite() { minimum } else { 0.0 };
        max[bin_index] = if maximum.is_finite() { maximum } else { 0.0 };
    }
    Agg { mean, min, max }
}

/// Mean of `value_of(row)` over the rows landing in each equal-width bin, holding
/// the last known value across empty bins (a step series); leading empty bins are
/// 0. Applied per shard, so mean-in-bin is just that shard's held level for the bin.
fn bin_mean(
    rows: &[&Row],
    value_of: impl Fn(&Row) -> f64,
    t_min: f64,
    bin_width: f64,
    n_bins: usize,
) -> Vec<f64> {
    let mut sum = vec![0.0f64; n_bins];
    let mut count = vec![0u32; n_bins];
    #[allow(
        clippy::cast_possible_wrap,
        reason = "n_bins is FINE_BINS (200) at every call site, far below isize::MAX"
    )]
    let last = n_bins as isize - 1;
    for row in rows {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "Rust's float-to-int cast saturates rather than wrapping, and .clamp(0, last) \
                      forces the value into [0, n_bins) before the final usize cast"
        )]
        let bin = (((row.time_ms - t_min) / bin_width).floor() as isize).clamp(0, last) as usize;
        sum[bin] += value_of(row);
        count[bin] += 1;
    }
    let mut out = vec![0.0f64; n_bins];
    let mut carry = 0.0;
    for bin_index in 0..n_bins {
        if count[bin_index] > 0 {
            carry = sum[bin_index] / count[bin_index] as f64;
        }
        out[bin_index] = carry;
    }
    out
}

/// All normalized KV snapshot rows. Numeric columns are read in one typed pass each
/// (`column_f64`); `pool_tag` is the only string, read from its `StringArray` row
/// by row. `arrow_cast` pins `pool_tag` to `Utf8`: DataFusion 45 reads Parquet
/// string columns as `Utf8View` by default, which the downcast below would reject.
async fn collect_rows(
    ctx: &SessionContext,
    has_retained_prefix_breakdown: bool,
) -> Result<Vec<Row>> {
    let retained_prefix_projection = if has_retained_prefix_breakdown {
        "retained_prefix_kv"
    } else {
        "CAST(0 AS BIGINT) AS retained_prefix_kv"
    };
    let sql = format!(
        "SELECT arrow_cast(pool_tag, 'Utf8') AS pool_tag, group_id, worker_id, time_ms, \
                active_kv, {retained_prefix_projection}, projected_peak, promised_kv \
         FROM kv_snapshot"
    );
    let batches = collect(ctx, &sql).await?;
    let mut out = Vec::new();
    for batch in &batches {
        let tags = col(batch, "pool_tag")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("pool_tag is not a Utf8 column"))?;
        let group_ids = column_f64(col(batch, "group_id")?)?;
        let worker_ids = column_f64(col(batch, "worker_id")?)?;
        let times_ms = column_f64(col(batch, "time_ms")?)?;
        let active_kv = column_f64(col(batch, "active_kv")?)?;
        let retained_prefix = column_f64(col(batch, "retained_prefix_kv")?)?;
        let projected_peak = column_f64(col(batch, "projected_peak")?)?;
        let promised = column_f64(col(batch, "promised_kv")?)?;
        for row in 0..batch.num_rows() {
            let tag = if tags.is_null(row) {
                ""
            } else {
                tags.value(row)
            };
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "group_id and worker_id come from this run's own kv_snapshot log \
                          (simulator-generated, not external input); both are small nonnegative \
                          indices by construction and Rust's float-to-int cast saturates rather \
                          than wrapping"
            )]
            out.push(Row {
                pool_tag: tag.to_string(),
                group_id: group_ids[row] as u64,
                worker_id: worker_ids[row] as u64,
                time_ms: times_ms[row],
                active_kv: active_kv[row],
                retained_prefix_kv: retained_prefix[row],
                projected_peak: projected_peak[row],
                promised_kv: promised[row],
            });
        }
    }
    Ok(out)
}

/// Soft probe for an append-only optional column. Unlike `require_columns`, this
/// deliberately accepts old log schemas and lets the payload expose that the
/// retained-prefix breakdown was unavailable.
async fn has_column(ctx: &SessionContext, table: &str, column: &str) -> Result<bool> {
    let data_frame = ctx.table(table).await?;
    Ok(data_frame.schema().field_with_name(None, column).is_ok())
}

fn definitions() -> Value {
    json!({
        "scope": "all kv_snapshot rows, one series per (pool_tag, group_id)",
        "per_shard": "each worker's own binned step series (the plot's faint 'shadow' lines)",
        "aggregate": "across-shard mean / min / max per time bin (the standout lines); a wide min↔max band = shard imbalance",
        "active_tokens": "resident/committed KV — the CURRENT occupancy, including retained prefix-cache entries",
        "retained_prefix_tokens": "prefix-cache component captured at the same raw submit that set the active_kv throttle-window peak; active - retained_prefix is non-prefix active KV",
        "has_retained_prefix_breakdown": "false for old kv_snapshot logs without the retained_prefix_kv column; their compatibility series is zero-filled and must not be interpreted as a measured zero",
        "projected_tokens": "admission gate's FUTURE estimate (each admitted request's full decode horizon); may exceed capacity",
        "promised_tokens": "reserved-but-not-resident KV footprint (admitted, not yet realized)",
        "capacity_tokens": "static per-shard KV token budget from run_meta.json kv_pools; null when KV logging carries no capacity",
        "fraction": "tokens / capacity_tokens (percent on the plot); projected is intentionally not clamped to 100%",
        "peak_active_max_pct": "worst individual shard's peak — the imbalance-aware headroom",
        "bin": "one equal-width time segment (200 bins over the run span)",
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
