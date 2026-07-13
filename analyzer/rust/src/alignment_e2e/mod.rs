//! End-to-end measured ↔ simulated request alignment.
//!
//! Client TTFT/TPOT/E2E and server EngineCore TTFT/TPOT are compared as
//! independent raw distributions. Request ids only audit whether either run lost
//! requests: execution order can differ even when both runs consume the same
//! trace, so per-id latency subtraction would compare scheduler positions that
//! are not equivalent. Completion throughput is intentionally named: TraceLab
//! does not log every token timestamp, so both sides assign a request's output
//! tokens to its completion bin rather than pretending to have an instantaneous
//! token-production trace.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::alignment_input;
use crate::cdf::{cdf_series, clean_nonnegative_sorted, stats};
use crate::io::{resolve_artifact_path, SCHEMA_VERSION};
use crate::session::{col, collect, register_if_exists, require_columns, value_f64};

const SIM_SLO_TABLE: &str = "alignment_sim_slo";
const SIM_SLO_COLUMNS: &[&str] = &[
    "request_id",
    "completed",
    "arrival_time_ms",
    "num_output_tokens",
    "ttft_ms",
    "tpot_mean_ms",
    "finish_decode_time_ms",
];

#[derive(Clone)]
struct RequestMetrics {
    output_tokens: u64,
    completion_ms: f64,
    ttft_ms: Option<f64>,
    tpot_ms: Option<f64>,
    e2e_ms: Option<f64>,
}

struct ServerRequestTimings {
    request_count: usize,
    ttft_ms: Vec<f64>,
    // Schema v1 files do not carry TPOT. Schema v2 files carry Some, which may
    // still be empty when every completed request generated exactly one token.
    tpot_ms: Option<Vec<f64>>,
}

pub async fn run(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    let input = alignment_input::read(log_dir)?;
    if !input.e2e.enabled {
        return Ok(unavailable(
            log_dir,
            "E2E alignment disabled in profiling config",
        ));
    }
    ensure!(input.e2e.throughput_bins > 0, "throughput_bins must be > 0");

    let measured = read_measured_requests(&input.replay_result)?;
    let server_timings = input
        .request_timings_result
        .as_deref()
        .map(read_server_request_timings)
        .transpose()?;
    let simulated = read_sim_requests(ctx, &input.simulation_log_dir).await?;
    ensure!(
        !measured.is_empty(),
        "measured replay contains no successful VibeSim requests"
    );
    ensure!(
        !simulated.is_empty(),
        "simulation request_slo contains no completed requests"
    );
    if let Some(server_timings) = &server_timings {
        ensure!(
            server_timings.request_count == measured.len(),
            "server timing request count {} does not match client replay request count {}",
            server_timings.request_count,
            measured.len()
        );
    }

    let measured_ttft = latency_samples(&measured, |request| request.ttft_ms);
    let simulated_ttft = latency_samples(&simulated, |request| request.ttft_ms);
    let measured_tpot = latency_samples(&measured, |request| request.tpot_ms);
    let simulated_tpot = latency_samples(&simulated, |request| request.tpot_ms);
    let measured_e2e = latency_samples(&measured, |request| request.e2e_ms);
    let simulated_e2e = latency_samples(&simulated, |request| request.e2e_ms);

    let shared_request_ids = measured
        .keys()
        .filter(|request_id| simulated.contains_key(*request_id))
        .count();

    let throughput = throughput_series(&measured, &simulated, input.e2e.throughput_bins);
    let mut latency_cdf_comparisons = vec![latency_cdf_comparison(
        "client_ttft",
        "Client-observed TTFT",
        "Client measured",
        &measured_ttft,
        &simulated_ttft,
    )];
    if let Some(server_timings) = &server_timings {
        latency_cdf_comparisons.push(latency_cdf_comparison(
            "server_ttft",
            "Server engine-core TTFT",
            "Server measured",
            &server_timings.ttft_ms,
            &simulated_ttft,
        ));
    }
    latency_cdf_comparisons.push(latency_cdf_comparison(
        "tpot",
        "Client-accounted TPOT",
        "Client measured",
        &measured_tpot,
        &simulated_tpot,
    ));
    if let Some(server_tpot) = server_timings
        .as_ref()
        .and_then(|timings| timings.tpot_ms.as_ref())
        .filter(|samples| !samples.is_empty())
    {
        latency_cdf_comparisons.push(latency_cdf_comparison(
            "server_tpot",
            "Server engine-core TPOT",
            "Server measured",
            server_tpot,
            &simulated_tpot,
        ));
    }
    latency_cdf_comparisons.push(latency_cdf_comparison(
        "e2e",
        "E2E",
        "Client measured",
        &measured_e2e,
        &simulated_e2e,
    ));
    let server_ttft_report = server_timings.as_ref().map_or_else(
        || {
            json!({
                "available": false,
                "reason": "profile does not contain engine-core request timing instrumentation",
            })
        },
        |timings| latency_distribution_report(&timings.ttft_ms, &simulated_ttft),
    );
    let server_tpot_report = server_timings.as_ref().map_or_else(
        || {
            json!({
                "available": false,
                "reason": "profile does not contain engine-core request timing instrumentation",
            })
        },
        |timings| match timings.tpot_ms.as_ref() {
            None => json!({
                "available": false,
                "reason": "profile request timing schema v1 does not contain engine-core TPOT",
            }),
            Some(samples) if samples.is_empty() => json!({
                "available": false,
                "reason": "profile contains no multi-token request with a defined engine-core TPOT",
            }),
            Some(samples) => latency_distribution_report(samples, &simulated_tpot),
        },
    );
    let definitions = definitions();
    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "analysis_log_dir": log_dir.display().to_string(),
            "profile_log_dir": input.profile_log_dir.display().to_string(),
            "simulation_log_dir": input.simulation_log_dir.display().to_string(),
            "measured_successful_requests": measured.len(),
            "server_measured_requests": server_timings.as_ref().map(|timings| timings.request_count),
            "server_tpot_requests": server_timings
                .as_ref()
                .and_then(|timings| timings.tpot_ms.as_ref())
                .map(Vec::len),
            "simulated_completed_requests": simulated.len(),
            "request_id_audit": {
                "shared_ids": shared_request_ids,
                "measured_only_ids": measured.len().saturating_sub(shared_request_ids),
                "simulated_only_ids": simulated.len().saturating_sub(shared_request_ids),
            },
        },
        "available": true,
        "latency": {
            "client_ttft": latency_distribution_report(&measured_ttft, &simulated_ttft),
            "server_ttft": server_ttft_report,
            "tpot": latency_distribution_report(&measured_tpot, &simulated_tpot),
            "server_tpot": server_tpot_report,
            "e2e": latency_distribution_report(&measured_e2e, &simulated_e2e),
        },
        "throughput": throughput["summary"],
        "definitions": definitions.clone(),
    });
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "analysis_log_dir": log_dir.display().to_string(),
            "profile_log_dir": input.profile_log_dir.display().to_string(),
            "throughput_bins": input.e2e.throughput_bins,
        },
        "throughput": throughput["series"],
        "latency_cdf_comparisons": latency_cdf_comparisons,
        "definitions": definitions,
    });
    Ok((report, payload))
}

fn read_server_request_timings(path: &Path) -> Result<ServerRequestTimings> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut request_ids = std::collections::BTreeSet::new();
    let mut schema_version = None;
    let mut ttft_ms = Vec::new();
    let mut tpot_ms = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("parse {} line {}", path.display(), line_index + 1))?;
        let row_schema_version = value
            .get("schema_version")
            .and_then(Value::as_u64)
            .context("server request timing missing schema_version")?;
        ensure!(
            matches!(row_schema_version, 1 | 2),
            "unsupported server request timing schema {} at {} line {}",
            row_schema_version,
            path.display(),
            line_index + 1
        );
        if let Some(expected_schema_version) = schema_version {
            ensure!(
                expected_schema_version == row_schema_version,
                "mixed server request timing schemas {} and {} in {}",
                expected_schema_version,
                row_schema_version,
                path.display()
            );
        } else {
            schema_version = Some(row_schema_version);
        }
        let request_id = value
            .get("request_id")
            .and_then(Value::as_str)
            .context("server request timing missing request_id")?;
        ensure!(
            request_ids.insert(request_id.to_string()),
            "duplicate server request timing id {request_id:?}"
        );
        let ttft_sample_ms = value
            .get("engine_core_ttft_ms")
            .and_then(Value::as_f64)
            .context("server request timing missing engine_core_ttft_ms")?;
        ensure!(
            ttft_sample_ms.is_finite() && ttft_sample_ms >= 0.0,
            "server engine_core_ttft_ms must be finite and nonnegative"
        );
        ttft_ms.push(ttft_sample_ms);

        if row_schema_version == 2 {
            let num_output_tokens = value
                .get("num_output_tokens")
                .and_then(Value::as_u64)
                .context("server request timing missing num_output_tokens")?;
            ensure!(
                num_output_tokens > 0,
                "server request timing num_output_tokens must be positive"
            );
            let decode_ms = value
                .get("engine_core_decode_ms")
                .and_then(Value::as_f64)
                .context("server request timing missing engine_core_decode_ms")?;
            ensure!(
                decode_ms.is_finite() && decode_ms >= 0.0,
                "server engine_core_decode_ms must be finite and nonnegative"
            );
            if num_output_tokens == 1 {
                ensure!(
                    value.get("engine_core_tpot_ms") == Some(&Value::Null),
                    "server engine_core_tpot_ms must be null for a one-token request"
                );
            } else {
                let sample = value
                    .get("engine_core_tpot_ms")
                    .and_then(Value::as_f64)
                    .context("server request timing missing engine_core_tpot_ms")?;
                ensure!(
                    sample.is_finite() && sample >= 0.0,
                    "server engine_core_tpot_ms must be finite and nonnegative"
                );
                let expected = decode_ms / (num_output_tokens - 1) as f64;
                ensure!(
                    (sample - expected).abs() <= 1e-6,
                    "server engine_core_tpot_ms does not equal engine_core_decode_ms/(num_output_tokens-1)"
                );
                tpot_ms.push(sample);
            }
        }
    }
    ensure!(
        !ttft_ms.is_empty(),
        "server request timing result contains no requests"
    );
    Ok(ServerRequestTimings {
        request_count: ttft_ms.len(),
        ttft_ms,
        tpot_ms: (schema_version == Some(2)).then_some(tpot_ms),
    })
}

fn read_measured_requests(path: &Path) -> Result<BTreeMap<String, RequestMetrics>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut raw = Vec::new();
    let mut origin_s = f64::INFINITY;
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("parse {} line {}", path.display(), line_index + 1))?;
        if value.pointer("/source/type").and_then(Value::as_str) != Some("vibe_sim_request")
            || value.pointer("/outcome/status").and_then(Value::as_str) != Some("SUCCESS")
        {
            continue;
        }
        let id = value
            .pointer("/source/data/id")
            .and_then(Value::as_str)
            .context("successful replay row missing source.data.id")?
            .to_string();
        let outcome = value
            .pointer("/outcome")
            .context("replay row missing outcome")?;
        let post = outcome
            .get("post_timestamp")
            .and_then(Value::as_f64)
            .or_else(|| outcome.get("submit_timestamp").and_then(Value::as_f64))
            .context("successful replay row missing submit/post timestamp")?;
        origin_s = origin_s.min(post);
        raw.push((id, outcome.clone(), post));
    }

    let mut requests = BTreeMap::new();
    for (id, outcome, _post) in raw {
        let complete = outcome
            .get("complete_timestamp")
            .and_then(Value::as_f64)
            .context("successful replay row missing complete_timestamp")?;
        let output_tokens = outcome
            .get("output_len_actual")
            .and_then(Value::as_u64)
            .context("successful replay row missing output_len_actual")?;
        let ttft = outcome.get("first_token_ms").and_then(Value::as_f64);
        let e2e = outcome.get("total_duration_ms").and_then(Value::as_f64);
        let tpot = match (ttft, e2e) {
            (Some(first), Some(total)) if output_tokens > 1 && total >= first => {
                Some((total - first) / (output_tokens - 1) as f64)
            }
            _ => None,
        };
        ensure!(
            requests
                .insert(
                    id.clone(),
                    RequestMetrics {
                        output_tokens,
                        completion_ms: (complete - origin_s) * 1000.0,
                        ttft_ms: ttft,
                        tpot_ms: tpot,
                        e2e_ms: e2e,
                    },
                )
                .is_none(),
            "duplicate measured request id {id:?}"
        );
    }
    Ok(requests)
}

async fn read_sim_requests(
    ctx: &SessionContext,
    simulation_log_dir: &Path,
) -> Result<BTreeMap<String, RequestMetrics>> {
    let path = resolve_artifact_path(simulation_log_dir, "request_slo.parquet");
    ensure!(
        register_if_exists(ctx, SIM_SLO_TABLE, path).await?,
        "request_slo.parquet not found under {}",
        simulation_log_dir.display()
    );
    require_columns(ctx, SIM_SLO_TABLE, SIM_SLO_COLUMNS).await?;
    let batches = collect(
        ctx,
        "SELECT request_id, arrival_time_ms, num_output_tokens, ttft_ms, \
         tpot_mean_ms, finish_decode_time_ms FROM alignment_sim_slo WHERE completed",
    )
    .await?;
    let mut rows = Vec::new();
    let mut origin_ms = f64::INFINITY;
    for batch in &batches {
        let ids = col(batch, "request_id")?;
        let arrivals = col(batch, "arrival_time_ms")?;
        let tokens = col(batch, "num_output_tokens")?;
        let ttft = col(batch, "ttft_ms")?;
        let tpot = col(batch, "tpot_mean_ms")?;
        let finish = col(batch, "finish_decode_time_ms")?;
        for row in 0..batch.num_rows() {
            let arrival = value_f64(arrivals, row)?;
            origin_ms = origin_ms.min(arrival);
            rows.push((
                value_f64(ids, row)? as u64,
                arrival,
                value_f64(tokens, row)? as u64,
                finite(value_f64(ttft, row)?),
                finite(value_f64(tpot, row)?),
                finite(value_f64(finish, row)?),
            ));
        }
    }
    let mut requests = BTreeMap::new();
    for (id, arrival, output_tokens, ttft, tpot, finish) in rows {
        let e2e = finish.map(|value| value - arrival);
        let completion_ms = finish.map(|value| value - origin_ms).unwrap_or(f64::NAN);
        requests.insert(
            id.to_string(),
            RequestMetrics {
                output_tokens,
                completion_ms,
                ttft_ms: ttft,
                tpot_ms: tpot,
                e2e_ms: e2e,
            },
        );
    }
    Ok(requests)
}

fn latency_samples(
    requests: &BTreeMap<String, RequestMetrics>,
    select: impl Fn(&RequestMetrics) -> Option<f64>,
) -> Vec<f64> {
    requests
        .values()
        .filter_map(select)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .collect()
}

fn latency_distribution_report(measured: &[f64], simulated: &[f64]) -> Value {
    json!({
        "measured_ms": stats(&clean_nonnegative_sorted(measured)),
        "simulated_ms": stats(&clean_nonnegative_sorted(simulated)),
    })
}

fn latency_cdf_comparison(
    key: &str,
    label: &str,
    measured_label: &str,
    measured: &[f64],
    simulated: &[f64],
) -> Value {
    json!({
        "key": key,
        "label": label,
        "unit": "ms",
        "measured": cdf_series(&format!("{key}_measured"), measured_label, "ms", measured),
        "simulated": cdf_series(&format!("{key}_simulated"), "Simulated", "ms", simulated),
    })
}

fn throughput_series(
    measured: &BTreeMap<String, RequestMetrics>,
    simulated: &BTreeMap<String, RequestMetrics>,
    bins: usize,
) -> Value {
    let measured_end = max_completion(measured);
    let simulated_end = max_completion(simulated);
    let end_ms = measured_end.max(simulated_end).max(1e-9);
    let width_ms = end_ms / bins as f64;
    let mut measured_tokens = vec![0u64; bins];
    let mut simulated_tokens = vec![0u64; bins];
    bin_completions(measured, width_ms, &mut measured_tokens);
    bin_completions(simulated, width_ms, &mut simulated_tokens);
    let t_start_ms: Vec<_> = (0..bins).map(|index| index as f64 * width_ms).collect();
    let t_end_ms: Vec<_> = (1..=bins).map(|index| index as f64 * width_ms).collect();
    let seconds = width_ms / 1000.0;
    let measured_tps: Vec<_> = measured_tokens
        .iter()
        .map(|v| *v as f64 / seconds)
        .collect();
    let simulated_tps: Vec<_> = simulated_tokens
        .iter()
        .map(|v| *v as f64 / seconds)
        .collect();
    let measured_total: u64 = measured.values().map(|r| r.output_tokens).sum();
    let simulated_total: u64 = simulated.values().map(|r| r.output_tokens).sum();
    json!({
        "summary": {
            "bins": bins,
            "common_span_ms": end_ms,
            "measured_output_tokens": measured_total,
            "simulated_output_tokens": simulated_total,
            "measured_completion_tps": measured_total as f64 / (measured_end / 1000.0).max(1e-9),
            "simulated_completion_tps": simulated_total as f64 / (simulated_end / 1000.0).max(1e-9),
        },
        "series": {
            "t_start_ms": t_start_ms,
            "t_end_ms": t_end_ms,
            "measured_output_tps": measured_tps,
            "simulated_output_tps": simulated_tps,
        }
    })
}

fn bin_completions(requests: &BTreeMap<String, RequestMetrics>, width_ms: f64, bins: &mut [u64]) {
    for request in requests.values() {
        if !request.completion_ms.is_finite() || request.completion_ms < 0.0 {
            continue;
        }
        let index = ((request.completion_ms / width_ms).floor() as usize).min(bins.len() - 1);
        bins[index] += request.output_tokens;
    }
}

fn max_completion(requests: &BTreeMap<String, RequestMetrics>) -> f64 {
    requests
        .values()
        .map(|request| request.completion_ms)
        .filter(|value| value.is_finite())
        .fold(0.0, f64::max)
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn definitions() -> Value {
    json!({
        "request_id_audit": "TraceLab source.data.id and simulator request_slo.request_id are intersected only to detect missing requests; ids do not pair latency samples",
        "latency_comparison": "measured and simulated raw latency distributions are summarized independently and overlaid as two CDF curves; no per-request subtraction or division",
        "client_ttft": "TraceLab client-observed first_token_ms distribution vs simulator ttft_ms distribution; includes frontend/network/tokenization and response-path overhead outside the engine",
        "server_ttft": "vLLM EngineCore queued timestamp to first-token EngineCore output timestamp distribution vs the same simulator ttft_ms distribution; excludes client/frontend transport",
        "tpot": "client-accounted (total_duration_ms - first_token_ms)/(output_tokens-1) distribution vs simulator tpot_mean_ms distribution; total_duration_ms includes response completion and client post-processing",
        "server_tpot": "vLLM EngineCore (last-token output timestamp - first-token output timestamp)/(num_output_tokens-1) distribution vs simulator tpot_mean_ms distribution; excludes HTTP/SSE/client completion overhead",
        "e2e": "measured total_duration_ms distribution vs simulator finish_decode_time_ms-arrival_time_ms distribution",
        "completion_throughput": "output tokens assigned to the request completion bin on both sides; not instantaneous token-production throughput",
    })
}

fn unavailable(log_dir: &Path, reason: &str) -> (Value, Value) {
    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {"analysis_log_dir": log_dir.display().to_string()},
        "available": false,
        "reason": reason,
        "definitions": definitions(),
    });
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {"analysis_log_dir": log_dir.display().to_string(), "available": false, "reason": reason},
        "throughput": {},
        "latency_cdf_comparisons": [],
        "definitions": definitions(),
    });
    (report, payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_bins_use_one_common_axis() {
        let measured = BTreeMap::from([(
            "1".into(),
            RequestMetrics {
                output_tokens: 10,
                completion_ms: 10.0,
                ttft_ms: None,
                tpot_ms: None,
                e2e_ms: None,
            },
        )]);
        let simulated = BTreeMap::from([(
            "1".into(),
            RequestMetrics {
                output_tokens: 10,
                completion_ms: 20.0,
                ttft_ms: None,
                tpot_ms: None,
                e2e_ms: None,
            },
        )]);
        let value = throughput_series(&measured, &simulated, 2);
        assert_eq!(value["series"]["measured_output_tps"][1], 1000.0);
        assert_eq!(value["series"]["simulated_output_tps"][1], 1000.0);
    }

    #[test]
    fn latency_distributions_do_not_pair_by_request_id() {
        let measured = BTreeMap::from([
            ("1".into(), request_with_ttft(10.0)),
            ("2".into(), request_with_ttft(20.0)),
        ]);
        let simulated = BTreeMap::from([
            ("1".into(), request_with_ttft(200.0)),
            ("2".into(), request_with_ttft(100.0)),
        ]);

        let measured_samples = latency_samples(&measured, |request| request.ttft_ms);
        let simulated_samples = latency_samples(&simulated, |request| request.ttft_ms);
        let comparison = latency_cdf_comparison(
            "client_ttft",
            "Client-observed TTFT",
            "Client measured",
            &measured_samples,
            &simulated_samples,
        );

        assert_eq!(comparison["measured"]["x"], json!([10.0, 20.0]));
        assert_eq!(comparison["simulated"]["x"], json!([100.0, 200.0]));
    }

    #[test]
    fn schema_v1_server_timing_keeps_ttft_and_marks_tpot_unavailable() {
        let path = std::env::temp_dir().join(format!(
            "vibesim_alignment_server_timing_v1_{}.jsonl",
            std::process::id()
        ));
        fs::write(
            &path,
            r#"{"schema_version":1,"request_id":"vibesim_1","engine_core_ttft_ms":12.5}"#,
        )
        .unwrap();

        let timings = read_server_request_timings(&path).unwrap();
        let _ = fs::remove_file(path);

        assert_eq!(timings.request_count, 1);
        assert_eq!(timings.ttft_ms, vec![12.5]);
        assert!(timings.tpot_ms.is_none());
    }

    fn request_with_ttft(ttft_ms: f64) -> RequestMetrics {
        RequestMetrics {
            output_tokens: 1,
            completion_ms: 0.0,
            ttft_ms: Some(ttft_ms),
            tpot_ms: None,
            e2e_ms: None,
        }
    }
}
