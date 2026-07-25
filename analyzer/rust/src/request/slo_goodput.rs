//! Trusted-cutoff SLO-goodput prediction over every request row present at the
//! configured hard window. This intentionally differs from completed-only
//! `slo-general`: partial requests and their emitted tokens remain in scope.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use datafusion::prelude::SessionContext;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::cdf::{cdf_series, clean_nonnegative_sorted, stats, CdfSeries};
use crate::io::{resolve_artifact_path, SCHEMA_VERSION};
use crate::session::{col, collect, register_if_exists, require_columns, value_f64};

const SLO_COLS: &[&str] = &["completed", "num_output_tokens", "tpot_mean_ms"];
const TPOT_SLO_MS: f64 = 30.0;

#[derive(Debug, Deserialize)]
struct ParamsDoc {
    workload: WorkloadDoc,
}

#[derive(Debug, Deserialize)]
struct WorkloadDoc {
    duration_ms: f64,
    request_rate: f64,
}

#[derive(Debug)]
struct RequestAggregate {
    total_requests: u64,
    completed_requests: u64,
    partial_with_output_requests: u64,
    zero_output_requests: u64,
    total_emitted_output_tokens: u64,
}

pub async fn run_slo_goodput(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    let workload = match read_workload(log_dir) {
        Ok(workload) => workload,
        Err(reason) => {
            return Ok((
                unavailable(log_dir, &reason),
                unavailable_payload(log_dir, &reason),
            ));
        }
    };

    let slo_path = resolve_artifact_path(log_dir, "request_slo.parquet");
    if !register_if_exists(ctx, "slo", slo_path).await? {
        let reason = "request_slo.parquet not found";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }
    require_columns(ctx, "slo", SLO_COLS).await?;

    let aggregate = collect_request_aggregate(ctx).await?;
    if aggregate.total_requests == 0 {
        let reason = "request_slo.parquet contains no arrived request rows";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }

    // Only this bounded per-request scalar is collected. Counts and token totals
    // stay in DataFusion; the shared cleaner defines the finite, nonnegative TPOT
    // population used by both trusted arithmetic mean and the CDF.
    let tpot_samples = collect_tpot_samples(ctx).await?;
    let sorted_tpot = clean_nonnegative_sorted(&tpot_samples);
    if sorted_tpot.is_empty() {
        let reason = "no request row has a defined finite nonnegative TPOT";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }

    let tpot_stats = stats(&sorted_tpot);
    let tpot_mean_ms = tpot_stats
        .mean
        .expect("non-empty cleaned TPOT population has a mean");
    let duration_s = workload.duration_ms / 1000.0;
    let output_throughput_tok_s = aggregate.total_emitted_output_tokens as f64 / duration_s;
    let transport_errors = 0_u64;
    let passed = aggregate.total_requests > 0
        && !sorted_tpot.is_empty()
        && transport_errors == 0
        && tpot_mean_ms < TPOT_SLO_MS;
    let slo_goodput_tok_s = if passed { output_throughput_tok_s } else { 0.0 };
    let incomplete_requests = aggregate
        .total_requests
        .saturating_sub(aggregate.completed_requests);

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
        },
        "available": true,
        "metrics": {
            "total_requests": aggregate.total_requests,
            "completed_requests": aggregate.completed_requests,
            "incomplete_requests": incomplete_requests,
            "partial_with_output_requests": aggregate.partial_with_output_requests,
            "zero_output_requests": aggregate.zero_output_requests,
            "tpot_observed_requests": sorted_tpot.len(),
            "configured_duration_ms": workload.duration_ms,
            "configured_request_rate": workload.request_rate,
            "total_emitted_output_tokens": aggregate.total_emitted_output_tokens,
            "output_throughput_tok_s": output_throughput_tok_s,
            "tpot": tpot_stats,
            "tpot_slo_ms": TPOT_SLO_MS,
            "transport_errors": transport_errors,
            "transport_errors_modeled": false,
            "passed": passed,
            "slo_goodput_tok_s": slo_goodput_tok_s,
        },
        "definitions": definitions(),
    });

    let series: Vec<CdfSeries> = vec![cdf_series(
        "slo_goodput_tpot",
        "Cutoff TPOT",
        "ms/token",
        &sorted_tpot,
    )];
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "available": true,
            "max_cdf_points": crate::cdf::MAX_CDF_POINTS,
            "total_requests": aggregate.total_requests,
            "tpot_observed_requests": sorted_tpot.len(),
        },
        "series": series,
        "definitions": definitions(),
    });

    Ok((report, payload))
}

fn read_workload(log_dir: &Path) -> std::result::Result<WorkloadDoc, String> {
    let path = resolve_artifact_path(log_dir, "params.json");
    let text = fs::read_to_string(&path)
        .map_err(|_| format!("params.json not found or unreadable: {}", path.display()))?;
    let params: ParamsDoc =
        serde_json::from_str(&text).map_err(|error| format!("invalid params.json: {error}"))?;
    if !params.workload.duration_ms.is_finite() || params.workload.duration_ms <= 0.0 {
        return Err("workload.duration_ms must be finite and greater than zero".to_string());
    }
    if !params.workload.request_rate.is_finite() || params.workload.request_rate <= 0.0 {
        return Err("workload.request_rate must be finite and greater than zero".to_string());
    }
    Ok(params.workload)
}

async fn collect_request_aggregate(ctx: &SessionContext) -> Result<RequestAggregate> {
    let batches = collect(
        ctx,
        "SELECT \
           COUNT(*) AS total_requests, \
           SUM(CASE WHEN completed THEN 1 ELSE 0 END) AS completed_requests, \
           SUM(CASE WHEN NOT COALESCE(completed, FALSE) \
                    AND COALESCE(num_output_tokens, 0) > 0 THEN 1 ELSE 0 END) \
             AS partial_with_output_requests, \
           SUM(CASE WHEN COALESCE(num_output_tokens, 0) = 0 THEN 1 ELSE 0 END) \
             AS zero_output_requests, \
           SUM(COALESCE(CAST(num_output_tokens AS BIGINT), 0)) \
             AS total_emitted_output_tokens \
         FROM slo",
    )
    .await?;
    let batch = batches
        .first()
        .filter(|batch| batch.num_rows() > 0)
        .context("request aggregate query returned no row")?;
    Ok(RequestAggregate {
        total_requests: value_f64(col(batch, "total_requests")?, 0)? as u64,
        completed_requests: value_f64(col(batch, "completed_requests")?, 0)? as u64,
        partial_with_output_requests: value_f64(col(batch, "partial_with_output_requests")?, 0)?
            as u64,
        zero_output_requests: value_f64(col(batch, "zero_output_requests")?, 0)? as u64,
        total_emitted_output_tokens: value_f64(col(batch, "total_emitted_output_tokens")?, 0)?
            as u64,
    })
}

async fn collect_tpot_samples(ctx: &SessionContext) -> Result<Vec<f64>> {
    let batches = collect(
        ctx,
        "SELECT tpot_mean_ms FROM slo WHERE tpot_mean_ms IS NOT NULL",
    )
    .await?;
    let mut samples = Vec::new();
    for batch in &batches {
        let tpot = col(batch, "tpot_mean_ms")?;
        for row in 0..batch.num_rows() {
            samples.push(value_f64(tpot, row)?);
        }
    }
    Ok(samples)
}

fn definitions() -> Value {
    json!({
        "scope": "all arrived requests represented in request_slo.parquet at the configured hard cutoff; completed and incomplete rows are both in scope",
        "partial_requests": "incomplete rows remain in request counts, and every output token they emitted remains in the throughput numerator",
        "tpot": "per-request TPOT is (last token time - first token time) / (observed output tokens - 1); only rows with defined finite nonnegative TPOT enter the arithmetic mean and CDF, matching the trusted client",
        "throughput": "total emitted output tokens from completed and partial requests divided by configured workload.duration_ms; never drain time or last logging time",
        "threshold": "the predicted point passes only when requests and TPOT are observed and arithmetic mean TPOT is strictly < 30.0 ms",
        "transport_errors": "transport errors are not modeled by this simulator subject and are represented as zero for the predicted pass decision",
        "slo_goodput_tok_s": "output_throughput_tok_s when the strict cutoff decision passes, otherwise 0",
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
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "available": false,
            "reason": reason,
        },
        "series": [],
        "definitions": definitions(),
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, BooleanArray, Float32Array, RecordBatch, UInt32Array};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use tempfile::TempDir;

    use super::*;
    use crate::session::build_session;

    fn request_batch(
        completed: Vec<bool>,
        output_tokens: Vec<u32>,
        tpot_ms: Vec<Option<f32>>,
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("completed", DataType::Boolean, false),
            Field::new("num_output_tokens", DataType::UInt32, false),
            Field::new("tpot_mean_ms", DataType::Float32, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(BooleanArray::from(completed)) as ArrayRef,
                Arc::new(UInt32Array::from(output_tokens)) as ArrayRef,
                Arc::new(Float32Array::from(tpot_ms)) as ArrayRef,
            ],
        )
        .expect("request batch")
    }

    fn write_params(run: &TempDir, duration_ms: f64, request_rate: f64) {
        let raw = run.path().join("raw");
        fs::create_dir_all(&raw).expect("raw directory");
        fs::write(
            raw.join("params.json"),
            serde_json::to_vec(&json!({
                "workload": {
                    "duration_ms": duration_ms,
                    "request_rate": request_rate,
                }
            }))
            .expect("params json"),
        )
        .expect("write params");
    }

    async fn run_with_batch(run: &TempDir, batch: RecordBatch, duration_ms: f64) -> (Value, Value) {
        write_params(run, duration_ms, 4.0);
        let ctx = build_session();
        ctx.register_batch("slo", batch).expect("register batch");
        run_slo_goodput(&ctx, run.path())
            .await
            .expect("run slo-goodput")
    }

    #[tokio::test]
    async fn counts_all_seen_requests_and_uses_fixed_window_throughput() {
        let run = TempDir::new().expect("temporary run");
        let (report, payload) = run_with_batch(
            &run,
            request_batch(
                vec![true, false, false, false],
                vec![512, 256, 0, 1],
                vec![Some(20.0), Some(25.0), None, None],
            ),
            90_000.0,
        )
        .await;
        let metrics = &report["metrics"];

        assert_eq!(metrics["total_requests"], 4);
        assert_eq!(metrics["completed_requests"], 1);
        assert_eq!(metrics["incomplete_requests"], 3);
        assert_eq!(metrics["partial_with_output_requests"], 2);
        assert_eq!(metrics["zero_output_requests"], 1);
        assert_eq!(metrics["tpot_observed_requests"], 2);
        assert_eq!(metrics["total_emitted_output_tokens"], 769);
        assert!(
            (metrics["output_throughput_tok_s"].as_f64().unwrap() - 769.0 / 90.0).abs() < 1e-12
        );
        assert!(metrics["passed"].as_bool().unwrap());
        assert_eq!(payload["series"][0]["n"], 2);
        assert!((payload["series"][0]["markers"]["p99"].as_f64().unwrap() - 24.95).abs() < 1e-12);
    }

    #[tokio::test]
    async fn strict_thirty_ms_mean_fails() {
        let run = TempDir::new().expect("temporary run");
        let (report, _) = run_with_batch(
            &run,
            request_batch(
                vec![true, false],
                vec![10, 20],
                vec![Some(30.0), Some(30.0)],
            ),
            1_000.0,
        )
        .await;
        let metrics = &report["metrics"];

        assert_eq!(metrics["tpot"]["mean"], 30.0);
        assert!(!metrics["passed"].as_bool().unwrap());
        assert_eq!(metrics["slo_goodput_tok_s"], 0.0);
    }

    #[tokio::test]
    async fn null_tpot_is_excluded_without_dropping_tokens_or_request() {
        let run = TempDir::new().expect("temporary run");
        let (report, _) = run_with_batch(
            &run,
            request_batch(vec![true, false], vec![100, 200], vec![Some(10.0), None]),
            1_000.0,
        )
        .await;
        let metrics = &report["metrics"];

        assert_eq!(metrics["total_requests"], 2);
        assert_eq!(metrics["total_emitted_output_tokens"], 300);
        assert_eq!(metrics["tpot_observed_requests"], 1);
        assert_eq!(metrics["tpot"]["mean"], 10.0);
        assert_eq!(metrics["output_throughput_tok_s"], 300.0);
    }

    #[tokio::test]
    async fn unavailable_for_missing_or_invalid_inputs() {
        let missing_parquet = TempDir::new().expect("temporary run");
        write_params(&missing_parquet, 1_000.0, 1.0);
        let ctx = build_session();
        let (report, _) = run_slo_goodput(&ctx, missing_parquet.path())
            .await
            .expect("missing parquet is unavailable");
        assert!(!report["available"].as_bool().unwrap());

        let missing_params = TempDir::new().expect("temporary run");
        let ctx = build_session();
        ctx.register_batch("slo", request_batch(vec![true], vec![2], vec![Some(1.0)]))
            .expect("register batch");
        let (report, _) = run_slo_goodput(&ctx, missing_params.path())
            .await
            .expect("missing params is unavailable");
        assert!(!report["available"].as_bool().unwrap());

        let invalid_duration = TempDir::new().expect("temporary run");
        let (report, _) = run_with_batch(
            &invalid_duration,
            request_batch(vec![true], vec![2], vec![Some(1.0)]),
            0.0,
        )
        .await;
        assert!(!report["available"].as_bool().unwrap());

        let no_rows = TempDir::new().expect("temporary run");
        let (report, _) =
            run_with_batch(&no_rows, request_batch(vec![], vec![], vec![]), 1_000.0).await;
        assert!(!report["available"].as_bool().unwrap());

        let no_tpot = TempDir::new().expect("temporary run");
        let (report, payload) = run_with_batch(
            &no_tpot,
            request_batch(vec![false], vec![1], vec![None]),
            1_000.0,
        )
        .await;
        assert!(!report["available"].as_bool().unwrap());
        assert_eq!(payload["series"], json!([]));
    }
}
