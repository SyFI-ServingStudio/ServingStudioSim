//! Per-segment throughput: prefill / decode / total tokens-per-second over each
//! `request_state` snapshot interval, normalized per-GPU.
//!
//! `request_state` logs every admitted request's cumulative `completed_input_len`
//! (prefill progress) and `completed_output_len` (decode progress) at each tick.
//! Summed across requests, those columns are monotonic non-decreasing (completed
//! requests re-emit frozen values; never-admitted ones are skipped), so the tokens
//! processed in a segment are just the diff of consecutive ticks' column sums.
//! GPU count + name come from `run_meta.json` (sim-written); per-GPU = ÷ num_gpus.

use std::path::Path;

use anyhow::Result;
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::io::{read_run_meta, resolve_artifact_path, SCHEMA_VERSION};
use crate::session::{col, collect, register_if_exists, require_columns, value_f64};

/// request_state columns the throughput subject depends on (drift guard).
const STATE_COLS: &[&str] = &["logging_time", "completed_input_len", "completed_output_len"];

/// Coarse view cap: the binned plot uses at most this many equal-width segments
/// (ref's `compute_throughput_timeseries` shape), so a long run reads as a handful
/// of trend bars instead of hundreds of noisy snapshot intervals.
const MAX_BINS: usize = 10;

/// One snapshot tick's cumulative column sums: `(time_ms, prefill_cum, decode_cum)`.
type Tick = (f64, f64, f64);

/// Segments built from a list of boundary points — both the fine (per-tick) and
/// coarse (≤10 equal-width bins) views are "the deltas between consecutive
/// boundaries", so they share this.
struct Segments {
    json: Vec<Value>,
    t_start: Vec<f64>,
    t_end: Vec<f64>,
    prefill_pg: Vec<f64>,
    decode_pg: Vec<f64>,
    total_pg: Vec<f64>,
    tot_prefill_tok: f64,
    tot_decode_tok: f64,
}

pub async fn run_throughput(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    let state_path = resolve_artifact_path(log_dir, "request_state.parquet");
    if !register_if_exists(ctx, "state", state_path).await? {
        let reason = "request_state.parquet not found";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }
    require_columns(ctx, "state", STATE_COLS).await?;

    // GPU facts for per-GPU normalization. Absent run_meta.json ⇒ treat as 1 GPU
    // (the honest default for a run predating the sidecar).
    let (num_gpus, gpu_name) = read_run_meta(log_dir).unwrap_or((1, String::new()));
    let num_gpus = num_gpus.max(1);
    let g = num_gpus as f64;

    let ticks = collect_ticks(ctx).await?;
    if ticks.len() < 2 {
        let reason = "fewer than 2 request_state snapshot ticks (no segment to diff)";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }

    // Fine view: one segment per snapshot interval (boundaries = the ticks).
    let fine = build_segments(&ticks, g);

    // Coarse view: ≤10 equal-width time bins. Bin-edge cumulative comes from
    // linear interpolation of the (monotonic) tick cumulative, so the bins are
    // exact when edges land on ticks and well-defined when they don't.
    let t0 = ticks[0].0;
    let tn = ticks.last().unwrap().0;
    let n_bins = MAX_BINS.min(ticks.len() - 1);
    let bin_edges: Vec<Tick> = (0..=n_bins)
        .map(|k| {
            let tq = t0 + (tn - t0) * (k as f64) / (n_bins as f64);
            let (p, d) = interp_cum(&ticks, tq);
            (tq, p, d)
        })
        .collect();
    let coarse = build_segments(&bin_edges, g);

    let span_s = (tn - t0) / 1000.0;
    let safe_span = span_s.max(1e-9);
    let total_tok = fine.tot_prefill_tok + fine.tot_decode_tok;
    // Run-average serving rate (tokens / simulated second), per-GPU — the avg line
    // the binned plot draws.
    let avg_prefill_pg = fine.tot_prefill_tok / safe_span / g;
    let avg_decode_pg = fine.tot_decode_tok / safe_span / g;
    let avg_total_pg = total_tok / safe_span / g;

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "num_gpus": num_gpus,
            "gpu_name": gpu_name,
            "num_segments": fine.json.len(),
            "num_bins": coarse.json.len(),
            "span_ms": tn - t0,
        },
        "available": true,
        "totals": {
            "prefill_tokens": fine.tot_prefill_tok,
            "decode_tokens": fine.tot_decode_tok,
            "total_tokens": total_tok,
            "prefill_tps": fine.tot_prefill_tok / safe_span,
            "decode_tps": fine.tot_decode_tok / safe_span,
            "total_tps": total_tok / safe_span,
            "total_tps_per_gpu": avg_total_pg,
        },
        "segments": fine.json,
        "binned_segments": coarse.json,
        "definitions": definitions(),
    });

    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "num_gpus": num_gpus,
            "gpu_name": gpu_name,
            "unit": "tokens/s per GPU",
            // Per-GPU run-average per series — the binned plot's avg reference lines.
            "avg_per_gpu": {
                "total": avg_total_pg,
                "prefill": avg_prefill_pg,
                "decode": avg_decode_pg,
            },
        },
        // Fine (per-snapshot-interval) view — the existing detailed plot.
        "t_start_ms": fine.t_start,
        "t_end_ms": fine.t_end,
        "series": [
            {"key": "total", "label": "Total", "per_gpu": fine.total_pg},
            {"key": "prefill", "label": "Prefill", "per_gpu": fine.prefill_pg},
            {"key": "decode", "label": "Decode", "per_gpu": fine.decode_pg},
        ],
        // Coarse (≤10 equal-width bins) view — the new trend plot with avg lines.
        "coarse": {
            "t_start_ms": coarse.t_start,
            "t_end_ms": coarse.t_end,
            "series": [
                {"key": "total", "label": "Total", "per_gpu": coarse.total_pg},
                {"key": "prefill", "label": "Prefill", "per_gpu": coarse.prefill_pg},
                {"key": "decode", "label": "Decode", "per_gpu": coarse.decode_pg},
            ],
        },
        "definitions": definitions(),
    });

    Ok((report, payload))
}

/// Build per-segment throughput from `boundaries` (sorted `(t, p_cum, d_cum)`):
/// one segment between each consecutive pair, rate = Δtokens / Δt. Shared by the
/// fine (boundaries = ticks) and coarse (boundaries = bin edges) views.
fn build_segments(boundaries: &[Tick], g: f64) -> Segments {
    let mut s = Segments {
        json: Vec::new(),
        t_start: Vec::new(),
        t_end: Vec::new(),
        prefill_pg: Vec::new(),
        decode_pg: Vec::new(),
        total_pg: Vec::new(),
        tot_prefill_tok: 0.0,
        tot_decode_tok: 0.0,
    };
    for w in boundaries.windows(2) {
        let (t0, p0, d0) = w[0];
        let (t1, p1, d1) = w[1];
        let dt_s = (t1 - t0) / 1000.0;
        if !dt_s.is_finite() || dt_s <= 0.0 {
            continue;
        }
        // Column sums are monotonic; `.max(0.0)` is defensive against any reorder.
        let prefill_tok = (p1 - p0).max(0.0);
        let decode_tok = (d1 - d0).max(0.0);
        s.tot_prefill_tok += prefill_tok;
        s.tot_decode_tok += decode_tok;

        let prefill_tps = prefill_tok / dt_s;
        let decode_tps = decode_tok / dt_s;
        let total_tps = prefill_tps + decode_tps;

        s.json.push(json!({
            "t_start_ms": t0,
            "t_end_ms": t1,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "total_tps": total_tps,
            "prefill_tps_per_gpu": prefill_tps / g,
            "decode_tps_per_gpu": decode_tps / g,
            "total_tps_per_gpu": total_tps / g,
        }));
        s.t_start.push(t0);
        s.t_end.push(t1);
        s.prefill_pg.push(prefill_tps / g);
        s.decode_pg.push(decode_tps / g);
        s.total_pg.push(total_tps / g);
    }
    s
}

/// Linearly interpolate the cumulative `(prefill, decode)` at time `tq` from the
/// monotonic tick series. Clamps to the endpoints outside the range.
fn interp_cum(ticks: &[Tick], tq: f64) -> (f64, f64) {
    let first = ticks[0];
    let last = *ticks.last().unwrap();
    if tq <= first.0 {
        return (first.1, first.2);
    }
    if tq >= last.0 {
        return (last.1, last.2);
    }
    // First tick at/after tq; interpolate against its predecessor.
    let hi = ticks.partition_point(|t| t.0 < tq);
    let (t1, p1, d1) = ticks[hi];
    let (t0, p0, d0) = ticks[hi - 1];
    let frac = if t1 > t0 { (tq - t0) / (t1 - t0) } else { 0.0 };
    (p0 + frac * (p1 - p0), d0 + frac * (d1 - d0))
}

/// Per-tick cumulative prefill/decode token sums, ordered by snapshot time.
///
/// We collapse to one row per `(request_id, logging_time)` (MAX) before summing.
/// The current sim never emits a duplicate `(request_id, logging_time)` — its
/// end-of-run `finalize` skips the census when the run ends on a snapshot tick
/// (`sim/run.rs`) — so this is purely defensive: it keeps the analyzer correct on
/// logs from older sim builds (which re-logged the final tick, double-counting the
/// last segment) and absorbs any future re-introduction of same-clock duplicates.
async fn collect_ticks(ctx: &SessionContext) -> Result<Vec<Tick>> {
    let batches = collect(
        ctx,
        "SELECT logging_time, SUM(p) AS p_cum, SUM(d) AS d_cum FROM ( \
             SELECT request_id, logging_time, \
                    MAX(completed_input_len) AS p, \
                    MAX(completed_output_len) AS d \
             FROM state GROUP BY request_id, logging_time \
         ) GROUP BY logging_time ORDER BY logging_time",
    )
    .await?;
    let mut ticks = Vec::new();
    for batch in &batches {
        let t = col(batch, "logging_time")?;
        let p = col(batch, "p_cum")?;
        let d = col(batch, "d_cum")?;
        for row in 0..batch.num_rows() {
            ticks.push((value_f64(t, row)?, value_f64(p, row)?, value_f64(d, row)?));
        }
    }
    // SQL orders by time, but guard against cross-batch reordering.
    ticks.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(ticks)
}

fn definitions() -> Value {
    json!({
        "scope": "all admitted requests, per request_state snapshot interval",
        "segment": "one interval between consecutive request_state logging_time ticks",
        "prefill_tps": "Δ(Σ completed_input_len) / Δt — prefill tokens processed per second",
        "decode_tps": "Δ(Σ completed_output_len) / Δt — output tokens generated per second",
        "total_tps": "prefill_tps + decode_tps",
        "per_gpu": "the corresponding rate divided by num_gpus (from run_meta.json)",
        "binned_segments": "the same metric re-aggregated into <=10 equal-width time bins \
                            (a coarse trend view); avg_per_gpu is the run-average reference",
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
