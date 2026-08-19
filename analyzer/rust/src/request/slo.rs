//! SLO analysis, split into two subjects (see `registry.rs`):
//!   - `slo-general` — TTFT / TPOT / E2E, from the pre-aggregated scalar columns
//!     only. Always available — survives `io.log_output_token_times` being off.
//!   - `slo-detailed` — per-token ITL, derived from the `output_token_times`
//!     array. Available only when that array was logged (the flag on); otherwise
//!     the array is empty and the subject reports unavailable.
//!
//! Both read `request_slo.parquet` (per-request terminal row). Scope: completed
//! requests only (user decision). Each runner emits the `(report, payload)` pair
//! via the caller.
//!
//! Session E2E (multi-round rollup) is deferred: with the new aggregate
//! `request_state` schema there is no per-request `session_id`/`session_arrival_time_ms`,
//! and the previous single-round defaults made the metric a degenerate duplicate of
//! per-request E2E (one request per "session"). When multi-turn lifecycle work
//! lands (see `common/request.rs` "L7 lifecycle" comment), add session columns to
//! `request_slo` and reintroduce the `session_e2e` metric here.

use std::path::Path;

use anyhow::Result;
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::cdf::{cdf_series, clean_nonnegative_sorted, stats, CdfSeries};
use crate::io::{resolve_artifact_path, SCHEMA_VERSION};
use crate::session::{
    col, collect, register_if_exists, require_columns, value_f32_list, value_f64,
};

/// request_slo scalar columns `slo-general` depends on (drift guard).
const GENERAL_COLS: &[&str] = &[
    "completed",
    "arrival_time_ms",
    "ttft_ms",
    "tpot_mean_ms",
    "finish_decode_time_ms",
];
/// request_slo column `slo-detailed` depends on.
const DETAILED_COLS: &[&str] = &["completed", "output_token_times"];

// ════════════════════════════════════════════════════════════════════════════
// slo-general — scalar-derived latency SLOs (always available)
// ════════════════════════════════════════════════════════════════════════════

#[derive(Default)]
struct GeneralSamples {
    ttft: Vec<f64>,
    tpot: Vec<f64>,
    e2e: Vec<f64>,
}

/// TTFT / TPOT / E2E from the pre-aggregated scalar columns of `request_slo`.
pub async fn run_slo_general(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    let slo_path = resolve_artifact_path(log_dir, "request_slo.parquet");
    if !register_if_exists(ctx, "slo", slo_path).await? {
        let reason = "request_slo.parquet not found";
        return Ok((
            unavailable(log_dir, reason, general_definitions()),
            unavailable_payload(log_dir, reason),
        ));
    }
    require_columns(ctx, "slo", GENERAL_COLS).await?;

    let mut s = GeneralSamples::default();
    let total = scalar_count(ctx, "SELECT COUNT(*) AS c FROM slo").await?;
    collect_general_samples(ctx, &mut s).await?;
    let completed = s.e2e.len();

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "total_requests": total,
            "completed_requests": completed,
            "excluded_requests": total.saturating_sub(completed as u64),
        },
        "available": true,
        "metrics": {
            "ttft": stats(&clean_nonnegative_sorted(&s.ttft)),
            "tpot": stats(&clean_nonnegative_sorted(&s.tpot)),
            "e2e": stats(&clean_nonnegative_sorted(&s.e2e)),
        },
        "definitions": general_definitions(),
    });

    let series: Vec<CdfSeries> = vec![
        cdf_series("ttft", "TTFT", "ms", &s.ttft),
        cdf_series("tpot", "TPOT", "ms/token", &s.tpot),
        cdf_series("e2e", "End-to-end", "ms", &s.e2e),
    ];
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "max_cdf_points": crate::cdf::MAX_CDF_POINTS,
        },
        "series": series,
        "definitions": general_definitions(),
    });

    Ok((report, payload))
}

/// TTFT/TPOT/E2E from pre-agg'd scalar columns over completed rows (E2E =
/// `finish_decode_time_ms − arrival`, which the sim always logs). No per-token
/// array is read, so this is unaffected by `io.log_output_token_times`.
async fn collect_general_samples(ctx: &SessionContext, s: &mut GeneralSamples) -> Result<()> {
    let batches = collect(
        ctx,
        "SELECT arrival_time_ms, ttft_ms, tpot_mean_ms, finish_decode_time_ms \
         FROM slo WHERE completed",
    )
    .await?;
    for batch in &batches {
        let arrival = col(batch, "arrival_time_ms")?;
        let ttft = col(batch, "ttft_ms")?;
        let tpot = col(batch, "tpot_mean_ms")?;
        let finish = col(batch, "finish_decode_time_ms")?;
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
        }
    }
    Ok(())
}

// Session E2E (multi-round rollup) was here; it has been removed pending real
// multi-turn lifecycle (see file header). Reintroduce when `request_slo` carries
// session_id / session_arrival_time_ms populated from real session metadata.

// ════════════════════════════════════════════════════════════════════════════
// slo-detailed — per-token ITL (needs the logged `output_token_times` array)
// ════════════════════════════════════════════════════════════════════════════

/// Per-token ITL CDF. Requires the `output_token_times` array to have been
/// logged; with `io.log_output_token_times` off the column is empty per row, so
/// no gaps are produced and the subject reports unavailable.
pub async fn run_slo_detailed(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    let slo_path = resolve_artifact_path(log_dir, "request_slo.parquet");
    if !register_if_exists(ctx, "slo", slo_path).await? {
        let reason = "request_slo.parquet not found";
        return Ok((
            unavailable(log_dir, reason, detailed_definitions()),
            unavailable_payload(log_dir, reason),
        ));
    }
    require_columns(ctx, "slo", DETAILED_COLS).await?;

    let mut itl: Vec<f64> = Vec::new();
    collect_itl(ctx, &mut itl).await?;

    if itl.is_empty() {
        let reason = "no per-token gaps: output_token_times not logged \
                      (io.log_output_token_times was off) or no completed requests";
        return Ok((
            unavailable(log_dir, reason, detailed_definitions()),
            unavailable_payload(log_dir, reason),
        ));
    }

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "itl_gaps": itl.len(),
        },
        "available": true,
        "metrics": {
            "itl": stats(&clean_nonnegative_sorted(&itl)),
        },
        "definitions": detailed_definitions(),
    });

    let series: Vec<CdfSeries> = vec![cdf_series("itl", "ITL", "ms", &itl)];
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "max_cdf_points": crate::cdf::MAX_CDF_POINTS,
        },
        "series": series,
        "definitions": detailed_definitions(),
    });

    Ok((report, payload))
}

/// Consecutive diffs of `output_token_times`, pooled across completed requests.
/// Empty when the array was not logged.
async fn collect_itl(ctx: &SessionContext, itl: &mut Vec<f64>) -> Result<()> {
    let batches = collect(ctx, "SELECT output_token_times FROM slo WHERE completed").await?;
    for batch in &batches {
        let times = col(batch, "output_token_times")?;
        for row in 0..batch.num_rows() {
            let token_times = value_f32_list(times, row)?;
            for w in token_times.windows(2) {
                itl.push(w[1] - w[0]);
            }
        }
    }
    Ok(())
}

// ════════════════════════════════════════════════════════════════════════════
// shared helpers
// ════════════════════════════════════════════════════════════════════════════

async fn scalar_count(ctx: &SessionContext, sql: &str) -> Result<u64> {
    let batches = collect(ctx, sql).await?;
    match batches.first() {
        Some(b) if b.num_rows() > 0 => {
            // COUNT(*) is Int64 in DataFusion; read as f64 then cast.
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a SQL COUNT(*) is always non-negative and, for any real run's row count, \
                          far under u64::MAX / f64's exact integer range"
            )]
            let count = value_f64(col(b, "c")?, 0)? as u64;
            Ok(count)
        }
        _ => Ok(0),
    }
}

fn general_definitions() -> Value {
    json!({
        "scope": "completed requests only",
        "ttft": "ttft_ms = first_token_time - arrival (pre-aggregated by the sim)",
        "tpot": "tpot_mean_ms = mean inter-token gap during decode (pre-aggregated)",
        "e2e": "finish_decode_time_ms - arrival_time_ms (scalar; survives log_output_token_times off)",
    })
}

fn detailed_definitions() -> Value {
    json!({
        "scope": "completed requests only",
        "itl": "consecutive diffs of output_token_times, pooled across requests \
                (available only when io.log_output_token_times was on)",
    })
}

fn unavailable(log_dir: &Path, reason: &str, definitions: Value) -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {"log_dir": log_dir.display().to_string()},
        "available": false,
        "reason": reason,
        "definitions": definitions,
    })
}

fn unavailable_payload(log_dir: &Path, reason: &str) -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {"log_dir": log_dir.display().to_string(), "available": false, "reason": reason},
        "series": [],
    })
}
