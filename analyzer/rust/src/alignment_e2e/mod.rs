//! End-to-end measured ↔ simulated request alignment.
//!
//! The shared VibeSim trace gives both sides the same request ids. This subject
//! pairs completed requests for TTFT/TPOT/E2E error statistics and compares
//! output-token completion throughput on one common time/bin axis. Completion
//! throughput is intentionally named: TraceLab does not log every token timestamp,
//! so both sides assign a request's output tokens to its completion bin rather
//! than pretending to have an instantaneous token-production trace.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::alignment_input;
use crate::cdf::{cdf_series, clean_nonnegative_sorted, percentile_sorted, stats, CdfSeries};
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

#[derive(Default)]
struct PairedSamples {
    measured: Vec<f64>,
    simulated: Vec<f64>,
    delta: Vec<f64>,
    relative_pct: Vec<f64>,
    abs_relative_pct: Vec<f64>,
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
    let simulated = read_sim_requests(ctx, &input.simulation_log_dir).await?;
    ensure!(
        !measured.is_empty(),
        "measured replay contains no successful VibeSim requests"
    );
    ensure!(
        !simulated.is_empty(),
        "simulation request_slo contains no completed requests"
    );

    let mut ttft = PairedSamples::default();
    let mut tpot = PairedSamples::default();
    let mut e2e = PairedSamples::default();
    let mut paired_requests = 0usize;
    for (request_id, real) in &measured {
        let Some(sim) = simulated.get(request_id) else {
            continue;
        };
        paired_requests += 1;
        add_pair(&mut ttft, real.ttft_ms, sim.ttft_ms);
        add_pair(&mut tpot, real.tpot_ms, sim.tpot_ms);
        add_pair(&mut e2e, real.e2e_ms, sim.e2e_ms);
    }

    let throughput = throughput_series(&measured, &simulated, input.e2e.throughput_bins);
    let latency_cdfs: Vec<CdfSeries> = vec![
        cdf_series(
            "ttft_abs_relative_error",
            "TTFT absolute relative error",
            "%",
            &ttft.abs_relative_pct,
        ),
        cdf_series(
            "tpot_abs_relative_error",
            "TPOT absolute relative error",
            "%",
            &tpot.abs_relative_pct,
        ),
        cdf_series(
            "e2e_abs_relative_error",
            "E2E absolute relative error",
            "%",
            &e2e.abs_relative_pct,
        ),
    ];
    let definitions = definitions();
    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "analysis_log_dir": log_dir.display().to_string(),
            "profile_log_dir": input.profile_log_dir.display().to_string(),
            "simulation_log_dir": input.simulation_log_dir.display().to_string(),
            "measured_successful_requests": measured.len(),
            "simulated_completed_requests": simulated.len(),
            "paired_requests": paired_requests,
            "unpaired_measured_requests": measured.len().saturating_sub(paired_requests),
            "unpaired_simulated_requests": simulated.len().saturating_sub(paired_requests),
        },
        "available": true,
        "latency": {
            "ttft": paired_report(&ttft),
            "tpot": paired_report(&tpot),
            "e2e": paired_report(&e2e),
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
        "latency_abs_relative_cdf": latency_cdfs,
        "definitions": definitions,
    });
    Ok((report, payload))
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

fn add_pair(samples: &mut PairedSamples, measured: Option<f64>, simulated: Option<f64>) {
    let (Some(measured), Some(simulated)) = (measured, simulated) else {
        return;
    };
    if !measured.is_finite() || !simulated.is_finite() || measured < 0.0 || simulated < 0.0 {
        return;
    }
    let delta = simulated - measured;
    samples.measured.push(measured);
    samples.simulated.push(simulated);
    samples.delta.push(delta);
    if measured > 1e-12 {
        let relative = delta / measured * 100.0;
        samples.relative_pct.push(relative);
        samples.abs_relative_pct.push(relative.abs());
    }
}

fn paired_report(samples: &PairedSamples) -> Value {
    json!({
        "n": samples.delta.len(),
        "measured_ms": signed_stats(&samples.measured),
        "simulated_ms": signed_stats(&samples.simulated),
        "delta_ms": signed_stats(&samples.delta),
        "relative_diff_pct": signed_stats(&samples.relative_pct),
        "abs_relative_error_pct": stats(&clean_nonnegative_sorted(&samples.abs_relative_pct)),
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

fn signed_stats(samples: &[f64]) -> Value {
    let mut sorted: Vec<_> = samples
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    sorted.sort_by(f64::total_cmp);
    if sorted.is_empty() {
        return json!({"n": 0, "mean": null, "p50": null, "p90": null, "p99": null, "min": null, "max": null});
    }
    json!({
        "n": sorted.len(),
        "mean": sorted.iter().sum::<f64>() / sorted.len() as f64,
        "p50": percentile_sorted(&sorted, 50.0),
        "p90": percentile_sorted(&sorted, 90.0),
        "p99": percentile_sorted(&sorted, 99.0),
        "min": sorted.first(),
        "max": sorted.last(),
    })
}

fn definitions() -> Value {
    json!({
        "pairing": "TraceLab source.data.id joined to simulator request_slo.request_id; completed/successful requests only",
        "delta": "simulated - measured; positive means the simulator is slower",
        "ttft": "measured first_token_ms vs simulator ttft_ms",
        "tpot": "measured (total_duration_ms - first_token_ms)/(output_tokens-1) vs simulator tpot_mean_ms",
        "e2e": "measured total_duration_ms vs simulator finish_decode_time_ms-arrival_time_ms",
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
        "latency_abs_relative_cdf": [],
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
}
