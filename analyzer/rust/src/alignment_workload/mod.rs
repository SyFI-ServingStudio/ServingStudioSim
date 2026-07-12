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
    prefill_tokens: u64,
    decode_batch_size: u64,
    scheduled_kv_tokens: u64,
}

pub async fn run(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    let input = alignment_input::read(log_dir)?;
    if !input.workload.enabled {
        return Ok(unavailable_pair(
            log_dir,
            "workload alignment disabled in analyze config",
        ));
    }

    let measured = read_measured_points(&input.parsed_nsys)?;
    ensure!(
        !measured.is_empty(),
        "parsed NSYS contains no iterations with both scheduler metrics and kernels"
    );

    if !register_cost_log(ctx, &input.simulation_log_dir).await? {
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
            "simulation_log_dir": input.simulation_log_dir.display().to_string(),
            "measured_iterations": measured.len(),
            "simulated_iterations": simulated.len(),
            "measured_span_ms": measured.last().map(|point| point.time_ms),
            "simulated_span_ms": simulated.last().map(|point| point.time_ms),
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
        },
        "definitions": definitions.clone(),
    });
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "analysis_log_dir": log_dir.display().to_string(),
            "profile_log_dir": input.profile_log_dir.display().to_string(),
            "simulation_log_dir": input.simulation_log_dir.display().to_string(),
        },
        "available": true,
        "measured": point_series(&measured),
        "simulated": point_series(&simulated),
        "definitions": definitions,
    });
    Ok((report, payload))
}

fn read_measured_points(path: &Path) -> Result<Vec<WorkloadPoint>> {
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
        let scheduled_kv_tokens = metrics.decode_kv_lens.iter().copied().sum::<u64>()
            + metrics
                .prefill_chunk_pairs
                .iter()
                .map(|(prefix, append)| prefix + append)
                .sum::<u64>();
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
    Ok(raw
        .into_iter()
        .map(
            |(start_ns, iteration_id, metrics, scheduled_kv_tokens)| WorkloadPoint {
                iteration_id,
                time_ms: start_ns.saturating_sub(origin_ns) as f64 / 1e6,
                prefill_tokens: metrics.prefill_tokens,
                decode_batch_size: metrics.decode_kv_lens.len() as u64,
                scheduled_kv_tokens,
            },
        )
        .collect())
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
                prefill_tokens,
                decode_batch_size,
                scheduled_kv_tokens,
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
    if let Some(origin_ms) = raw.first().map(|point| point.time_ms) {
        for point in &mut raw {
            point.time_ms -= origin_ms;
        }
    }
    Ok(raw)
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

fn point_series(points: &[WorkloadPoint]) -> Value {
    json!({
        "time_ms": points.iter().map(|point| point.time_ms).collect::<Vec<_>>(),
        "iteration_id": points.iter().map(|point| point.iteration_id).collect::<Vec<_>>(),
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
        "time_axis": {
            "measured": "first kernel launch of an NSYS iteration minus the first captured iteration's first kernel launch",
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
