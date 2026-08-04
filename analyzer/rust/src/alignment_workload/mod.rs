//! Scheduler-workload alignment by each run's recorded iteration id.
//!
//! This subject deliberately compares *scheduled attention work*, not resident
//! KV-cache occupancy.  For one iteration that work is the decode requests'
//! current KV lengths plus every scheduled prefill request's `(prefix + append)`
//! length.  Resident cache state belongs to the normal `kv-occupancy` subject.

use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use datafusion::prelude::SessionContext;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::alignment_input;
use crate::cdf::{clean_nonnegative_sorted, stats};
use crate::io::SCHEMA_VERSION;
use crate::session::{
    col, collect, register_cost_log, require_columns, value_f64, value_groups, value_string,
    COST_LOG_TABLE,
};

const COST_COLUMNS: &[&str] = &[
    "pool_tag",
    "worker_id",
    "iter_id",
    "batch_id",
    "wall_start_ms",
    "groups",
    "section",
    "layer",
];

#[derive(Debug, Deserialize)]
struct ParsedNsys {
    iteration_details: Vec<MeasuredIteration>,
}

#[derive(Debug, Clone, Deserialize)]
struct FullMeasuredMetrics {
    schema_version: u32,
    input_adapter: String,
    iteration_index: u64,
    prefill_tokens: u64,
    #[serde(default)]
    decode_kv_lens: Vec<u64>,
    #[serde(default)]
    prefill_chunk_pairs: Vec<(u64, u64)>,
    #[serde(default)]
    observed_start_monotonic_ns: Option<u64>,
    #[serde(default)]
    observed_end_monotonic_ns: Option<u64>,
    #[serde(default)]
    observed_elapsed_ms: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct MeasuredIteration {
    iteration: u64,
    metrics: Option<MeasuredMetrics>,
    ranges: Vec<MeasuredRange>,
}

#[derive(Debug, Deserialize)]
struct MeasuredMetrics {
    prefill_tokens: u64,
    #[serde(default)]
    decode_kv_lens: Vec<u64>,
    #[serde(default)]
    prefill_chunk_pairs: Vec<(u64, u64)>,
}

#[derive(Debug, Deserialize)]
struct MeasuredRange {
    kernels: Vec<MeasuredKernel>,
}

#[derive(Debug, Deserialize)]
struct MeasuredKernel {
    start_ns: u64,
}

#[derive(Debug, Clone)]
struct WorkloadPoint {
    iteration_id: u64,
    time_ms: f64,
    /// Actual cycle boundary to the next iteration. The final point has no next
    /// boundary and therefore remains `None` rather than inventing a duration.
    iteration_cycle_ms: Option<f64>,
    prefill_tokens: u64,
    decode_batch_size: u64,
    scheduled_kv_tokens: u64,
    /// EngineCore's observed result-wait/sampling interval. This is retained
    /// for audit and is distinct from the adjacent-start cadence above.
    observed_elapsed_ms: Option<f64>,
}

pub async fn run(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    let input = alignment_input::read_e2e_align(log_dir)?;
    let simulation_log_dir = input.simulation_log_dir.as_path();
    let measured = read_measured_points(&input.metrics_jsonl, &input.parsed_nsys)?;
    ensure!(
        !measured.is_empty(),
        "full-run vLLM metrics contain no workload iterations"
    );

    if !register_cost_log(ctx, simulation_log_dir).await? {
        return Ok(unavailable_pair(
            log_dir,
            "simulation cost_log/ dir not found",
        ));
    }
    require_columns(ctx, COST_LOG_TABLE, COST_COLUMNS).await?;
    let simulated = read_simulated_points(ctx).await?;
    ensure!(
        !simulated.is_empty(),
        "simulation cost_log contains no top-level iteration rows"
    );

    let definitions = definitions();
    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "analysis_log_dir": log_dir.display().to_string(),
            "profile_log_dir": input.profile_log_dir.display().to_string(),
            "workload_profile_log_dir": input.workload_profile_log_dir.display().to_string(),
            "simulation_log_dir": simulation_log_dir.display().to_string(),
            "measured_timeline_source": "full_run_engine_observation",
            "measured_iterations": measured.len(),
            "simulated_iterations": simulated.len(),
            "measured_span_ms": measured.last().map(|point| point.time_ms),
            "simulated_span_ms": simulated.last().map(|point| point.time_ms),
            "measured_iteration_cycles": measured.iter().filter(|point| point.iteration_cycle_ms.is_some()).count(),
            "simulated_iteration_cycles": simulated.iter().filter(|point| point.iteration_cycle_ms.is_some()).count(),
        },
        "available": true,
        "metrics": {
            "prefill_tokens": paired_stats(&measured, &simulated, |point| point.prefill_tokens),
            "decode_batch_size": paired_stats(
                &measured,
                &simulated,
                |point| point.decode_batch_size,
            ),
            "scheduled_kv_tokens": paired_stats(
                &measured,
                &simulated,
                |point| point.scheduled_kv_tokens,
            ),
            "iteration_cycle_ms": paired_cycle_stats(&measured, &simulated),
            "observed_elapsed_ms": measured_optional_stats(
                &measured,
                |point| point.observed_elapsed_ms,
            ),
        },
        "definitions": definitions.clone(),
    });
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "analysis_log_dir": log_dir.display().to_string(),
            "profile_log_dir": input.profile_log_dir.display().to_string(),
            "workload_profile_log_dir": input.workload_profile_log_dir.display().to_string(),
            "simulation_log_dir": simulation_log_dir.display().to_string(),
            "measured_timeline_source": "full_run_engine_observation",
        },
        "available": true,
        "measured": point_series(&measured),
        "simulated": point_series(&simulated),
        "definitions": definitions,
    });
    Ok((report, payload))
}

fn read_measured_points(
    metrics_path: &Path,
    parsed_nsys_path: &Path,
) -> Result<Vec<WorkloadPoint>> {
    let parsed_text = fs::read_to_string(parsed_nsys_path)
        .with_context(|| format!("read {}", parsed_nsys_path.display()))?;
    let parsed: ParsedNsys = serde_json::from_str(&parsed_text)
        .with_context(|| format!("parse {}", parsed_nsys_path.display()))?;
    let captured_iteration_ids: HashSet<u64> = parsed
        .iteration_details
        .iter()
        .map(|detail| detail.iteration)
        .collect();

    let metrics_text = fs::read_to_string(metrics_path)
        .with_context(|| format!("read {}", metrics_path.display()))?;
    let mut records = Vec::new();
    for (line_index, line) in metrics_text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: FullMeasuredMetrics = serde_json::from_str(line)
            .with_context(|| format!("parse {} line {}", metrics_path.display(), line_index + 1))?;
        ensure!(
            record.input_adapter == "vllm_text",
            "unsupported full-run metrics input_adapter {:?} at iteration {}",
            record.input_adapter,
            record.iteration_index
        );
        ensure!(
            matches!(record.schema_version, 1 | 2),
            "unsupported full-run metrics schema_version {} at iteration {}",
            record.schema_version,
            record.iteration_index
        );
        records.push(record);
    }
    ensure!(
        !records.is_empty(),
        "{} contains no iteration records",
        metrics_path.display()
    );

    // Prefix-cache preflight requests run before the measured replay and use
    // the same logger. Their iteration ids form short, disjoint runs (for
    // example 0 and 3). Select the unique contiguous run containing the NSYS
    // window, which is guaranteed to sit inside the actual replay.
    let mut contiguous_runs: Vec<Vec<FullMeasuredMetrics>> = Vec::new();
    for record in records {
        let continues_last = contiguous_runs
            .last()
            .and_then(|run| run.last())
            .is_some_and(|previous| previous.iteration_index + 1 == record.iteration_index);
        if !continues_last {
            contiguous_runs.push(Vec::new());
        }
        contiguous_runs.last_mut().unwrap().push(record);
    }
    let mut matching_runs = contiguous_runs.into_iter().filter(|run| {
        let run_ids: HashSet<u64> = run.iter().map(|record| record.iteration_index).collect();
        !captured_iteration_ids.is_empty() && captured_iteration_ids.is_subset(&run_ids)
    });
    let selected = matching_runs.next().with_context(|| {
        format!(
            "no contiguous full-run metrics segment contains all {} NSYS iterations",
            captured_iteration_ids.len()
        )
    })?;
    ensure!(
        matching_runs.next().is_none(),
        "multiple full-run metrics segments contain the NSYS iteration window"
    );

    if selected.iter().any(|record| record.schema_version == 1) {
        ensure!(
            selected.iter().all(|record| record.schema_version == 1),
            "full-run metrics segment mixes schema-v1 and schema-v2 records"
        );
        return read_measured_points_from_nsys(parsed_nsys_path);
    }

    let origin_ns = selected[0]
        .observed_start_monotonic_ns
        .context("schema-v2 full-run metrics omit observed_start_monotonic_ns")?;
    let mut points = Vec::with_capacity(selected.len());
    for record in selected {
        let start_ns = record
            .observed_start_monotonic_ns
            .context("schema-v2 full-run metrics omit observed_start_monotonic_ns")?;
        let end_ns = record
            .observed_end_monotonic_ns
            .context("schema-v2 full-run metrics omit observed_end_monotonic_ns")?;
        let observed_elapsed_ms = record
            .observed_elapsed_ms
            .context("schema-v2 full-run metrics omit observed_elapsed_ms")?;
        ensure!(
            end_ns >= start_ns,
            "full-run metrics iteration {} ends before it starts",
            record.iteration_index
        );
        ensure!(
            observed_elapsed_ms.is_finite() && observed_elapsed_ms >= 0.0,
            "full-run metrics iteration {} has invalid observed_elapsed_ms {}",
            record.iteration_index,
            observed_elapsed_ms
        );
        let scheduled_kv_tokens =
            scheduled_kv_tokens(&record.decode_kv_lens, &record.prefill_chunk_pairs);
        points.push(WorkloadPoint {
            iteration_id: record.iteration_index,
            time_ms: start_ns.saturating_sub(origin_ns) as f64 / 1e6,
            iteration_cycle_ms: None,
            prefill_tokens: record.prefill_tokens,
            decode_batch_size: record.decode_kv_lens.len() as u64,
            scheduled_kv_tokens,
            observed_elapsed_ms: Some(observed_elapsed_ms),
        });
    }
    assign_iteration_cycles(&mut points);
    Ok(points)
}

fn read_measured_points_from_nsys(path: &Path) -> Result<Vec<WorkloadPoint>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let parsed: ParsedNsys =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;

    // The first kernel launch is the iteration's GPU boundary.  Empty trailing
    // marker ranges are ignored because they have no observable GPU iteration.
    let mut raw = Vec::new();
    for detail in parsed.iteration_details {
        let first_kernel_ns = detail
            .ranges
            .iter()
            .flat_map(|range| range.kernels.iter())
            .map(|kernel| kernel.start_ns)
            .min();
        let Some(first_kernel_ns) = first_kernel_ns else {
            continue;
        };
        let metrics = detail.metrics.with_context(|| {
            format!(
                "NSYS iteration {} has kernels but no scheduler metrics",
                detail.iteration
            )
        })?;
        let scheduled_kv_tokens =
            scheduled_kv_tokens(&metrics.decode_kv_lens, &metrics.prefill_chunk_pairs);
        raw.push((
            first_kernel_ns,
            detail.iteration,
            metrics,
            scheduled_kv_tokens,
        ));
    }
    raw.sort_by_key(|row| row.0);
    let Some(origin_ns) = raw.first().map(|row| row.0) else {
        return Ok(Vec::new());
    };
    let mut points: Vec<WorkloadPoint> = raw
        .into_iter()
        .map(
            |(start_ns, iteration_id, metrics, scheduled_kv_tokens)| WorkloadPoint {
                iteration_id,
                time_ms: start_ns.saturating_sub(origin_ns) as f64 / 1e6,
                iteration_cycle_ms: None,
                prefill_tokens: metrics.prefill_tokens,
                decode_batch_size: metrics.decode_kv_lens.len() as u64,
                scheduled_kv_tokens,
                observed_elapsed_ms: None,
            },
        )
        .collect();
    assign_iteration_cycles(&mut points);
    Ok(points)
}

async fn read_simulated_points(ctx: &SessionContext) -> Result<Vec<WorkloadPoint>> {
    let batches = collect(
        ctx,
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, worker_id, iter_id, batch_id, \
         wall_start_ms, groups FROM cost_log WHERE section = 'iter' AND layer = -1 \
         ORDER BY wall_start_ms, iter_id, batch_id",
    )
    .await?;

    let mut streams = BTreeSet::new();
    let mut seen_rows = HashSet::new();
    let mut raw = Vec::new();
    for batch in &batches {
        let pool_tags = col(batch, "pool_tag")?;
        let worker_ids = col(batch, "worker_id")?;
        let iteration_ids = col(batch, "iter_id")?;
        let batch_ids = col(batch, "batch_id")?;
        let wall_start_ms = col(batch, "wall_start_ms")?;
        let groups = col(batch, "groups")?;
        for row in 0..batch.num_rows() {
            let pool_tag = value_string(pool_tags, row)?;
            let worker_id = value_f64(worker_ids, row)? as u64;
            let iteration_id = value_f64(iteration_ids, row)? as u64;
            let batch_id = value_f64(batch_ids, row)? as u64;
            streams.insert((pool_tag.clone(), worker_id));
            ensure!(
                seen_rows.insert((pool_tag, worker_id, iteration_id, batch_id)),
                "duplicate simulation top-level iteration row for worker={worker_id}, \
                 iter={iteration_id}, batch={batch_id}"
            );

            let groups = value_groups(groups, row)?;
            let prefill_tokens = groups
                .iter()
                .map(|group| u64::from(group.prefill_tokens))
                .sum();
            let decode_batch_size = groups
                .iter()
                .map(|group| u64::from(group.decode_request_count))
                .sum();
            let scheduled_kv_tokens = groups
                .iter()
                .map(|group| {
                    u64::from(group.decode_kv_total)
                        + group
                            .prefill_chunk_pairs
                            .iter()
                            .map(|(prefix, append)| u64::from(*prefix) + u64::from(*append))
                            .sum::<u64>()
                })
                .sum();
            raw.push(WorkloadPoint {
                iteration_id,
                time_ms: value_f64(wall_start_ms, row)?,
                iteration_cycle_ms: None,
                prefill_tokens,
                decode_batch_size,
                scheduled_kv_tokens,
                observed_elapsed_ms: None,
            });
        }
    }
    // Alignment v1 compares one unified model stream.  Silently merging PD/AFD
    // workers would manufacture a scheduler timeline that neither worker ran.
    ensure!(
        streams.len() <= 1,
        "alignment-workload currently requires one simulation worker stream; found {:?}",
        streams
    );
    raw.sort_by(|left, right| left.time_ms.total_cmp(&right.time_ms));
    assign_iteration_cycles(&mut raw);
    if let Some(origin_ms) = raw.first().map(|point| point.time_ms) {
        for point in &mut raw {
            point.time_ms -= origin_ms;
        }
    }
    Ok(raw)
}

fn scheduled_kv_tokens(decode_kv_lens: &[u64], prefill_chunk_pairs: &[(u64, u64)]) -> u64 {
    decode_kv_lens.iter().copied().sum::<u64>()
        + prefill_chunk_pairs
            .iter()
            .map(|(prefix, append)| prefix + append)
            .sum::<u64>()
}

/// Derive actual iteration cadence from adjacent execution boundaries. On the
/// measured schema-v2 path `time_ms` comes from full-run EngineCore observation
/// starts; schema-v1 archives fall back to NSYS first-kernel starts. Simulation
/// uses worker `wall_start_ms`, after multiplier and tick scheduling.
fn assign_iteration_cycles(points: &mut [WorkloadPoint]) {
    for index in 0..points.len().saturating_sub(1) {
        points[index].iteration_cycle_ms = Some(points[index + 1].time_ms - points[index].time_ms);
    }
}

fn measured_optional_stats<F>(measured: &[WorkloadPoint], select: F) -> Value
where
    F: Fn(&WorkloadPoint) -> Option<f64>,
{
    let measured_values: Vec<f64> = measured.iter().filter_map(select).collect();
    json!({
        "measured": stats(&clean_nonnegative_sorted(&measured_values)),
        "simulated": Value::Null,
    })
}

fn paired_stats<F>(measured: &[WorkloadPoint], simulated: &[WorkloadPoint], select: F) -> Value
where
    F: Fn(&WorkloadPoint) -> u64,
{
    let measured_values: Vec<f64> = measured.iter().map(|point| select(point) as f64).collect();
    let simulated_values: Vec<f64> = simulated.iter().map(|point| select(point) as f64).collect();
    json!({
        "measured": stats(&clean_nonnegative_sorted(&measured_values)),
        "simulated": stats(&clean_nonnegative_sorted(&simulated_values)),
    })
}

fn paired_cycle_stats(measured: &[WorkloadPoint], simulated: &[WorkloadPoint]) -> Value {
    let measured_values: Vec<f64> = measured
        .iter()
        .filter_map(|point| point.iteration_cycle_ms)
        .collect();
    let simulated_values: Vec<f64> = simulated
        .iter()
        .filter_map(|point| point.iteration_cycle_ms)
        .collect();
    json!({
        "measured": stats(&clean_nonnegative_sorted(&measured_values)),
        "simulated": stats(&clean_nonnegative_sorted(&simulated_values)),
    })
}

fn point_series(points: &[WorkloadPoint]) -> Value {
    json!({
        "time_ms": points.iter().map(|point| point.time_ms).collect::<Vec<_>>(),
        "iteration_id": points.iter().map(|point| point.iteration_id).collect::<Vec<_>>(),
        "iteration_cycle_ms": points
            .iter()
            .map(|point| point.iteration_cycle_ms)
            .collect::<Vec<_>>(),
        "observed_elapsed_ms": points
            .iter()
            .map(|point| point.observed_elapsed_ms)
            .collect::<Vec<_>>(),
        "prefill_tokens": points.iter().map(|point| point.prefill_tokens).collect::<Vec<_>>(),
        "decode_batch_size": points
            .iter()
            .map(|point| point.decode_batch_size)
            .collect::<Vec<_>>(),
        "scheduled_kv_tokens": points
            .iter()
            .map(|point| point.scheduled_kv_tokens)
            .collect::<Vec<_>>(),
    })
}

fn definitions() -> Value {
    json!({
        "grain": "one scheduler iteration; measured and simulated series keep their own iteration ids",
        "plot_axis": "recorded iteration_id on each side; ids are not renumbered or paired",
        "iteration_cycle_ms": {
            "measured": "this EngineCore observation start to the next observation start over the full replay; includes intervening update, scheduling, submission, and queue gaps",
            "simulated": "this actual top-level iter cost_log wall_start_ms to the next; includes gpu_time_multiplier, tick quantization, and any scheduler gap",
            "last_iteration": "null because no next boundary exists",
        },
        "observed_elapsed_ms": "vLLM EngineCore result-wait plus sampling observation recorded for every iteration; retained separately from adjacent-start cadence",
        "time_axis": {
            "measured": "EngineCore observation start minus the full replay's first observation start; independent of the bounded NSYS window",
            "simulated": "top-level iter cost_log wall_start_ms minus the first simulated iteration's wall_start_ms; this includes simulator gpu_time_multiplier effects",
        },
        "prefill_tokens": "number of prompt/chunk tokens scheduled in the iteration",
        "decode_batch_size": "number of decode requests scheduled in the iteration",
        "scheduled_kv_tokens": "sum(decode request KV lengths) + sum(prefill prefix_len + append_len) for the iteration",
        "scheduled_not_resident": "scheduled_kv_tokens is the attention/KV workload touched by this iteration, not logical or physical resident KV-cache occupancy",
    })
}

fn unavailable_pair(log_dir: &Path, reason: &str) -> (Value, Value) {
    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {"analysis_log_dir": log_dir.display().to_string()},
        "available": false,
        "reason": reason,
        "definitions": definitions(),
    });
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "analysis_log_dir": log_dir.display().to_string(),
            "reason": reason,
        },
        "available": false,
        "measured": {},
        "simulated": {},
        "definitions": definitions(),
    });
    (report, payload)
}
