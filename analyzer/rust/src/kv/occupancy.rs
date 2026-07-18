//! Per-pool KV-cache occupancy over time: how full each KV pool runs, in tokens
//! and as a fraction of its static capacity, binned into equal-width time segments.
//!
//! The sim's `kv_snapshot` stream records, per KV-bearing worker each iteration
//! (throttled — a running-max over a stride window), three token levels:
//! `active_kv` (resident/committed KV, the *current* occupancy), `projected_peak`
//! (the admission gate's *future* estimate — each admitted request's full decode
//! horizon), and `promised_kv` (reserved-but-not-resident footprint).
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

/// One snapshot row: `(pool_tag, group_id, worker_id, time_ms, active, projected,
/// promised)`.
type Row = (String, u64, u64, f64, f64, f64, f64);

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

    // (pool_tag, group_id) → capacity_tokens. Shards of a pool are symmetric, so
    // one capacity per pool×group; take the max across shards defensively (they
    // should be identical). Absent ⇒ raw-token-only mode (no pct reference).
    let mut cap_by_pool: BTreeMap<(String, u64), u64> = BTreeMap::new();
    if let Some(caps) = read_kv_capacities(log_dir) {
        for (pool_tag, _worker, group_id, capacity) in caps {
            let e = cap_by_pool.entry((pool_tag, group_id)).or_insert(0);
            *e = (*e).max(capacity);
        }
    }

    let rows = collect_rows(ctx).await?;
    if rows.is_empty() {
        let reason = "kv_snapshot has no rows to bin";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }

    let t_min = rows.iter().map(|r| r.3).fold(f64::INFINITY, f64::min);
    let t_max = rows.iter().map(|r| r.3).fold(f64::NEG_INFINITY, f64::max);
    // A single-sample run (t_max == t_min) still deserves a one-bin series; widen a
    // degenerate span to 1ms so bin math stays finite.
    let span_ms = (t_max - t_min).max(1.0);
    let n_bins = FINE_BINS;
    let bin_width = span_ms / n_bins as f64;
    let t_start: Vec<f64> = (0..n_bins).map(|b| t_min + b as f64 * bin_width).collect();
    let t_end: Vec<f64> = (0..n_bins)
        .map(|b| t_min + (b + 1) as f64 * bin_width)
        .collect();

    // Group rows by (pool_tag, group_id) → one plotted series; count groups per
    // pool_tag so a single-group pool gets a clean label (no `· g0` noise).
    let mut by_series: BTreeMap<(String, u64), Vec<&Row>> = BTreeMap::new();
    for r in &rows {
        by_series.entry((r.0.clone(), r.1)).or_default().push(r);
    }
    let mut groups_per_tag: BTreeMap<String, usize> = BTreeMap::new();
    for (tag, _g) in by_series.keys() {
        *groups_per_tag.entry(tag.clone()).or_default() += 1;
    }

    let mut series = Vec::new();
    let mut totals = Vec::new();
    for ((pool_tag, group_id), srows) in &by_series {
        let capacity = cap_by_pool.get(&(pool_tag.clone(), *group_id)).copied();

        // Per-shard binned step series (each shard samples ~once per stride window,
        // so `bin_mean` = that shard's held level in the bin), then the across-shard
        // mean/min/max per bin.
        let mut by_worker: BTreeMap<u64, Vec<&Row>> = BTreeMap::new();
        for r in srows {
            by_worker.entry(r.2).or_default().push(r);
        }
        let mut workers_json = Vec::new();
        let mut active_w: Vec<Vec<f64>> = Vec::new();
        let mut projected_w: Vec<Vec<f64>> = Vec::new();
        let mut promised_w: Vec<Vec<f64>> = Vec::new();
        for (wid, wrows) in &by_worker {
            let a = bin_mean(wrows, |r| r.4, t_min, bin_width, n_bins);
            let p = bin_mean(wrows, |r| r.5, t_min, bin_width, n_bins);
            let pr = bin_mean(wrows, |r| r.6, t_min, bin_width, n_bins);
            workers_json.push(json!({
                "worker_id": wid,
                "active_tokens": a,
                "projected_tokens": p,
                "promised_tokens": pr,
            }));
            active_w.push(a);
            projected_w.push(p);
            promised_w.push(pr);
        }
        let active = agg_across(&active_w, n_bins);
        let projected = agg_across(&projected_w, n_bins);
        let promised = agg_across(&promised_w, n_bins);

        let peak = |v: &[f64]| v.iter().copied().fold(0.0, f64::max);
        let peak_active_mean = peak(&active.mean);
        let peak_active_max = peak(&active.max);
        let peak_projected_mean = peak(&projected.mean);
        let peak_projected_max = peak(&projected.max);
        let peak_promised_max = peak(&promised.max);
        let n_active: usize = active.mean.iter().filter(|v| **v > 0.0).count();
        let mean_active = if n_active > 0 {
            active.mean.iter().sum::<f64>() / n_active as f64
        } else {
            0.0
        };

        let key = format!("{pool_tag}/g{group_id}");
        let label = if groups_per_tag.get(pool_tag).copied().unwrap_or(1) > 1 {
            format!("{pool_tag} · g{group_id}")
        } else {
            pool_tag.clone()
        };
        let pct = |tok: f64| capacity.map(|c| if c > 0 { tok / c as f64 } else { 0.0 });

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
            "projected": {"mean": projected.mean, "min": projected.min, "max": projected.max},
            "promised": {"mean": promised.mean, "min": promised.min, "max": promised.max},
        }));
        totals.push(json!({
            "pool_tag": pool_tag,
            "group_id": group_id,
            "capacity_tokens": capacity,
            "n_workers": by_worker.len(),
            "peak_active_mean_tokens": peak_active_mean,
            "peak_active_mean_pct": pct(peak_active_mean),
            // Worst individual shard — the imbalance-aware headroom number.
            "peak_active_max_tokens": peak_active_max,
            "peak_active_max_pct": pct(peak_active_max),
            "mean_active_pct": pct(mean_active),
            "peak_projected_mean_tokens": peak_projected_mean,
            "peak_projected_mean_pct": pct(peak_projected_mean),
            "peak_projected_max_pct": pct(peak_projected_max),
            "peak_promised_max_tokens": peak_promised_max,
            "peak_promised_max_pct": pct(peak_promised_max),
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
            "has_capacity": !cap_by_pool.is_empty(),
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
            "has_capacity": !cap_by_pool.is_empty(),
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
    let n = per_worker.len().max(1) as f64;
    for b in 0..n_bins {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut sum = 0.0;
        for w in per_worker {
            let v = w[b];
            sum += v;
            lo = lo.min(v);
            hi = hi.max(v);
        }
        mean[b] = sum / n;
        min[b] = if lo.is_finite() { lo } else { 0.0 };
        max[b] = if hi.is_finite() { hi } else { 0.0 };
    }
    Agg { mean, min, max }
}

/// Mean of `f(row)` over the rows landing in each equal-width bin, holding the last
/// known value across empty bins (a step series); leading empty bins are 0. Applied
/// per shard, so mean-in-bin is just that shard's held level for the bin.
fn bin_mean(
    rows: &[&Row],
    f: impl Fn(&Row) -> f64,
    t_min: f64,
    bin_width: f64,
    n_bins: usize,
) -> Vec<f64> {
    let mut sum = vec![0.0f64; n_bins];
    let mut cnt = vec![0u32; n_bins];
    let last = n_bins as isize - 1;
    for r in rows {
        let b = (((r.3 - t_min) / bin_width).floor() as isize).clamp(0, last) as usize;
        sum[b] += f(r);
        cnt[b] += 1;
    }
    let mut out = vec![0.0f64; n_bins];
    let mut carry = 0.0;
    for b in 0..n_bins {
        if cnt[b] > 0 {
            carry = sum[b] / cnt[b] as f64;
        }
        out[b] = carry;
    }
    out
}

/// All `(pool_tag, group_id, worker_id, time_ms, active_kv, projected_peak,
/// promised_kv)` rows. Numeric columns are read in one typed pass each
/// (`column_f64`); `pool_tag` is the only string, read from its `StringArray` row
/// by row. `arrow_cast` pins `pool_tag` to `Utf8`: DataFusion 45 reads Parquet
/// string columns as `Utf8View` by default, which the downcast below would reject.
async fn collect_rows(ctx: &SessionContext) -> Result<Vec<Row>> {
    let batches = collect(
        ctx,
        "SELECT arrow_cast(pool_tag, 'Utf8') AS pool_tag, group_id, worker_id, time_ms, \
                active_kv, projected_peak, promised_kv \
         FROM kv_snapshot",
    )
    .await?;
    let mut out = Vec::new();
    for batch in &batches {
        let tags = col(batch, "pool_tag")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("pool_tag is not a Utf8 column"))?;
        let g = column_f64(col(batch, "group_id")?)?;
        let w = column_f64(col(batch, "worker_id")?)?;
        let t = column_f64(col(batch, "time_ms")?)?;
        let a = column_f64(col(batch, "active_kv")?)?;
        let p = column_f64(col(batch, "projected_peak")?)?;
        let r = column_f64(col(batch, "promised_kv")?)?;
        for row in 0..batch.num_rows() {
            let tag = if tags.is_null(row) {
                ""
            } else {
                tags.value(row)
            };
            out.push((
                tag.to_string(),
                g[row] as u64,
                w[row] as u64,
                t[row],
                a[row],
                p[row],
                r[row],
            ));
        }
    }
    Ok(out)
}

fn definitions() -> Value {
    json!({
        "scope": "all kv_snapshot rows, one series per (pool_tag, group_id)",
        "per_shard": "each worker's own binned step series (the plot's faint 'shadow' lines)",
        "aggregate": "across-shard mean / min / max per time bin (the standout lines); a wide min↔max band = shard imbalance",
        "active_tokens": "resident/committed KV — the CURRENT occupancy",
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
