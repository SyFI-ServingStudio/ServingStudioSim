//! SLO analysis: request-level TTFT / TPOT / E2E / ITL + session-level E2E,
//! over **completed requests only** (user decision). Reads `request_slo.parquet`
//! (per-request terminal row: pre-agg'd ttft/tpot + the full `output_token_times`
//! list) and `request_state.parquet` (for the session_id grouping that session
//! E2E needs — `request_slo` has no session_id).
//!
//! Emits the report (numbers) + payload (CDF arrays) pair via the caller.

use std::path::Path;

use anyhow::Result;
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::cdf::{cdf_series, clean_nonnegative_sorted, stats, CdfSeries};
use crate::io::{resolve_artifact_path, SCHEMA_VERSION};
use crate::session::{
    col, collect, register_if_exists, require_columns, value_f32_list, value_f64,
};

/// request_slo columns the analyzer depends on (drift guard).
const SLO_COLS: &[&str] = &[
    "completed",
    "arrival_time_ms",
    "ttft_ms",
    "tpot_mean_ms",
    "finish_decode_time_ms",
    "output_token_times",
];
/// request_state columns used for session-level rollup.
const STATE_COLS: &[&str] = &[
    "session_id",
    "session_arrival_time_ms",
    "completion_time_ms",
    "completed",
];

#[derive(Default)]
struct Samples {
    ttft: Vec<f64>,
    tpot: Vec<f64>,
    e2e: Vec<f64>,
    itl: Vec<f64>,
    session_e2e: Vec<f64>,
}

/// Run the SLO analysis. Returns `(report, payload)` JSON values. `request_slo`
/// is required; `request_state` is optional (session E2E omitted if absent).
pub async fn run_slo(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    let slo_path = resolve_artifact_path(log_dir, "request_slo.parquet");
    if !register_if_exists(ctx, "slo", slo_path).await? {
        let reason = "request_slo.parquet not found";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }
    require_columns(ctx, "slo", SLO_COLS).await?;

    let mut s = Samples::default();
    let total = scalar_count(ctx, "SELECT COUNT(*) AS c FROM slo").await?;
    collect_request_samples(ctx, &mut s).await?;
    let completed = s.e2e_request_count();

    // Session E2E (optional): group completed request_state rows by session_id.
    let state_path = resolve_artifact_path(log_dir, "request_state.parquet");
    let has_state = register_if_exists(ctx, "state", state_path).await?;
    if has_state {
        require_columns(ctx, "state", STATE_COLS).await?;
        collect_session_e2e(ctx, &mut s).await?;
    }

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "total_requests": total,
            "completed_requests": completed,
            "excluded_requests": total.saturating_sub(completed as u64),
            "session_level_available": has_state,
        },
        "available": true,
        "metrics": {
            "ttft": stats(&clean_nonnegative_sorted(&s.ttft)),
            "tpot": stats(&clean_nonnegative_sorted(&s.tpot)),
            "e2e": stats(&clean_nonnegative_sorted(&s.e2e)),
            "itl": stats(&clean_nonnegative_sorted(&s.itl)),
            "session_e2e": stats(&clean_nonnegative_sorted(&s.session_e2e)),
        },
        "definitions": definitions(),
    });

    let series: Vec<CdfSeries> = vec![
        cdf_series("ttft", "TTFT", "ms", &s.ttft),
        cdf_series("tpot", "TPOT", "ms/token", &s.tpot),
        cdf_series("itl", "ITL", "ms", &s.itl),
        cdf_series("e2e", "End-to-end", "ms", &s.e2e),
        cdf_series("session_e2e", "Session E2E", "ms", &s.session_e2e),
    ];
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "max_cdf_points": crate::cdf::MAX_CDF_POINTS,
        },
        "series": series,
        "definitions": definitions(),
    });

    Ok((report, payload))
}

impl Samples {
    fn e2e_request_count(&self) -> usize {
        self.e2e.len()
    }
}

/// Per-request metrics from completed `request_slo` rows. TTFT/TPOT/E2E come
/// from pre-agg'd scalar columns (E2E = `finish_decode_time_ms − arrival`, which
/// the sim always logs). ITL needs the per-token series, so it is only populated
/// when `output_token_times` was logged (`io.log_token_times` on); with the
/// array off, `itl` is simply empty.
async fn collect_request_samples(ctx: &SessionContext, s: &mut Samples) -> Result<()> {
    let batches = collect(
        ctx,
        "SELECT arrival_time_ms, ttft_ms, tpot_mean_ms, finish_decode_time_ms, output_token_times \
         FROM slo WHERE completed",
    )
    .await?;
    for batch in &batches {
        let arrival = col(batch, "arrival_time_ms")?;
        let ttft = col(batch, "ttft_ms")?;
        let tpot = col(batch, "tpot_mean_ms")?;
        let finish = col(batch, "finish_decode_time_ms")?;
        let times = col(batch, "output_token_times")?;
        for row in 0..batch.num_rows() {
            let t = value_f64(ttft, row)?;
            if t.is_finite() {
                s.ttft.push(t);
            }
            let p = value_f64(tpot, row)?;
            if p.is_finite() {
                s.tpot.push(p);
            }
            let f = value_f64(finish, row)?;
            if f.is_finite() {
                s.e2e.push(f - value_f64(arrival, row)?);
            }
            // ITL only when the per-token array was logged (empty otherwise).
            let token_times = value_f32_list(times, row)?;
            for w in token_times.windows(2) {
                s.itl.push(w[1] - w[0]);
            }
        }
    }
    Ok(())
}

/// Session E2E = max(completion_time_ms) − session_arrival_time_ms per session,
/// over completed `request_state` rows. Single-round runs set
/// `session_id = request_id`, so each session is a size-1 group (degenerate, no
/// special case). Multi-round: this rolls up to the latest completed round.
async fn collect_session_e2e(ctx: &SessionContext, s: &mut Samples) -> Result<()> {
    let batches = collect(
        ctx,
        "SELECT MAX(completion_time_ms) AS last_completion, \
                MIN(session_arrival_time_ms) AS sess_arrival \
         FROM state WHERE completed GROUP BY session_id",
    )
    .await?;
    for batch in &batches {
        let last = col(batch, "last_completion")?;
        let arrival = col(batch, "sess_arrival")?;
        for row in 0..batch.num_rows() {
            let e2e = value_f64(last, row)? - value_f64(arrival, row)?;
            if e2e.is_finite() {
                s.session_e2e.push(e2e);
            }
        }
    }
    Ok(())
}

async fn scalar_count(ctx: &SessionContext, sql: &str) -> Result<u64> {
    let batches = collect(ctx, sql).await?;
    match batches.first() {
        Some(b) if b.num_rows() > 0 => {
            // COUNT(*) is Int64 in DataFusion; read as f64 then cast.
            Ok(value_f64(col(b, "c")?, 0)? as u64)
        }
        _ => Ok(0),
    }
}

fn definitions() -> Value {
    json!({
        "scope": "completed requests only",
        "ttft": "ttft_ms = first_token_time - arrival (pre-aggregated by the sim)",
        "tpot": "tpot_mean_ms = mean inter-token gap during decode (pre-aggregated)",
        "e2e": "finish_decode_time_ms - arrival_time_ms (scalar; survives log_token_times off)",
        "itl": "consecutive diffs of output_token_times, pooled across requests \
                (empty unless io.log_token_times was on)",
        "session_e2e": "max(completion_time_ms) - session_arrival_time_ms per session_id, \
                        over completed request_state rows",
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
        "series": [],
    })
}
