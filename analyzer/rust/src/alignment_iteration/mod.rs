//! Per-iteration measured ↔ predicted alignment.
//!
//! The launcher has already normalized the vLLM input into timing-predict cases.
//! This subject owns the actual comparison: join case indices to measured
//! iteration ids, expand the user-labeled folded inventory across every measured
//! phase, fold sim leaf multiplicities from the CostTree, and emit totals,
//! operation errors, kernel inventory, and plot arrays.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use datafusion::prelude::SessionContext;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::alignment_input;
use crate::cdf::{clean_nonnegative_sorted, percentile_sorted, stats};
use crate::io::{read_cost_manifests, resolve_artifact_path, SCHEMA_VERSION};
use crate::session::{
    col, collect, register_if_exists, require_columns, value_f32_list, value_f64,
};
use crate::trace::manifest::{node_time, FlatCostNode, Manifest};

const PREDICT_TABLE: &str = "alignment_predict_cost";
const PREDICT_COLUMNS: &[&str] = &["iter_id", "total_time_ms", "slot_time_ms", "section"];

#[derive(Deserialize)]
struct ParsedTrace {
    kernel_names: BTreeMap<String, String>,
    iteration_details: Vec<MeasuredIteration>,
}

#[derive(Deserialize)]
struct FoldedSequenceDoc {
    schema_version: u32,
    encoding: String,
    phases: BTreeMap<String, FoldedPhase>,
}

#[derive(Deserialize)]
struct FoldedPhase {
    unique_sequences: Vec<FoldedSequence>,
}

#[derive(Deserialize)]
struct FoldedSequence {
    sequence_id: String,
    iterations: Vec<u64>,
    expanded_kernel_count: usize,
    program: Vec<ProgramNode>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ProgramNode {
    Kernels { kernels: Vec<LabeledKernel> },
    Repeat { repeat: RepeatNode },
}

#[derive(Deserialize)]
struct RepeatNode {
    count: usize,
    body: RepeatBody,
}

#[derive(Deserialize)]
struct RepeatBody {
    kernels: Vec<LabeledKernel>,
}

#[derive(Clone, Deserialize)]
struct LabeledKernel {
    name: String,
    suggested_category: String,
    label: EmbeddedLabel,
}

#[derive(Clone, Deserialize)]
struct EmbeddedLabel {
    status: String,
    operation: Option<String>,
    simulated_slots: Option<Vec<String>>,
    #[serde(rename = "type")]
    kernel_type: Option<String>,
    role: Option<String>,
}

#[derive(Deserialize)]
struct MeasuredIteration {
    iteration: u64,
    iteration_type: String,
    ranges: Vec<MeasuredRange>,
}

#[derive(Deserialize)]
struct MeasuredRange {
    device_id: Option<i64>,
    phase: String,
    kernels: Vec<MeasuredKernel>,
}

#[derive(Deserialize)]
struct MeasuredKernel {
    name_id: u64,
    category: String,
    start_ns: u64,
    end_ns: u64,
}

#[derive(Deserialize)]
struct CaseMapDoc {
    schema_version: u32,
    cases: Vec<CaseMap>,
}

#[derive(Deserialize)]
struct CaseMap {
    case_index: u64,
    measured_iteration: u64,
    stage: String,
}

#[derive(Clone)]
struct OperationRule {
    operation: String,
    kernel_type: String,
    role: String,
    simulated_slots: Vec<String>,
    measured_rows: usize,
}

struct CompiledInventory {
    phases: BTreeMap<String, PhaseInventory>,
    operations: BTreeMap<String, OperationRule>,
    simulated_slots: BTreeMap<String, String>,
}

struct PhaseInventory {
    sequences: Vec<ExpandedSequence>,
    sequence_by_iteration: BTreeMap<u64, usize>,
}

struct ExpandedSequence {
    sequence_id: String,
    rows: Vec<ExpandedRow>,
}

#[derive(Clone)]
struct ExpandedRow {
    row_id: String,
    name: String,
    suggested_category: String,
    operation: Option<String>,
}

#[derive(Default)]
struct KernelAggregate {
    phase: String,
    row_id: String,
    name: String,
    category: String,
    operation: Option<String>,
    calls: usize,
    total_ns: u64,
    iterations: BTreeSet<u64>,
}

#[derive(Default)]
struct IterationKernelAggregate {
    phase: String,
    row_id: String,
    name: String,
    category: String,
    operation: Option<String>,
    calls: usize,
    total_ns: u64,
    first_start_ns: Option<u64>,
}

#[derive(Default)]
struct OperationAggregate {
    measured_ms: Vec<f64>,
    simulated_ms: Vec<f64>,
    delta_ms: Vec<f64>,
    relative_pct: Vec<f64>,
    abs_relative_pct: Vec<f64>,
    missing_measured: usize,
    missing_simulated: usize,
}

struct SimCase {
    total_ms: f64,
    slot_ms: Vec<f64>,
}

pub async fn run(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    let input = alignment_input::read(log_dir)?;
    if !input.iteration.enabled {
        return Ok(unavailable(
            log_dir,
            "iteration alignment disabled in profiling config",
        ));
    }
    let measured: ParsedTrace = read_json(&input.parsed_nsys)?;
    let case_map: CaseMapDoc = read_json(&input.timing_predict_case_map)?;
    ensure!(
        case_map.schema_version == 1,
        "case-map schema_version must be 1"
    );
    let labeled_path = input
        .labeled_kernel_sequences
        .as_deref()
        .context("iteration alignment requires labeled_kernel_sequences")?;
    let inventory = load_inventory(labeled_path)?;

    let manifests = read_cost_manifests(&input.predict_log_dir)?;
    let iter_manifests: Vec<_> = manifests
        .iter()
        .filter_map(|(key, doc)| doc.section("iter").map(|manifest| (key, manifest)))
        .collect();
    ensure!(
        iter_manifests.len() == 1,
        "alignment v1 expects exactly one predict worker with an iter manifest; found {}",
        iter_manifests.len()
    );
    let ((pool_tag, worker_id), manifest) = iter_manifests[0];
    let scales = leaf_scales(manifest)?;
    let sim_cases =
        load_sim_cases(ctx, &input.predict_log_dir, pool_tag, *worker_id, manifest).await?;

    let measured_by_id: BTreeMap<u64, &MeasuredIteration> = measured
        .iteration_details
        .iter()
        .map(|item| (item.iteration, item))
        .collect();
    let kernel_names: BTreeMap<u64, &str> = measured
        .kernel_names
        .iter()
        .map(|(id, name)| {
            id.parse::<u64>()
                .map(|id| (id, name.as_str()))
                .with_context(|| format!("parse kernel name id {id:?}"))
        })
        .collect::<Result<_>>()?;
    inventory.validate_slots(manifest)?;

    let mut iteration_rows = Vec::new();
    let mut breakdowns = Vec::new();
    let mut total_delta = Vec::new();
    let mut total_relative = Vec::new();
    let mut total_abs_relative = Vec::new();
    let mut cumulative_measured = 0.0;
    let mut cumulative_simulated = 0.0;
    let mut kernel_inventory: BTreeMap<String, KernelAggregate> = BTreeMap::new();
    let mut operation_stats: BTreeMap<String, OperationAggregate> = BTreeMap::new();
    let mut unmapped_measured: BTreeMap<(String, String), (String, usize, f64)> = BTreeMap::new();
    let mut unmapped_simulated: BTreeMap<String, f64> = BTreeMap::new();
    let mut measured_workload_ms = 0.0;
    let mut measured_mapped_ms = 0.0;
    let mut simulated_workload_ms = 0.0;
    let mut simulated_mapped_ms = 0.0;

    for joined in &case_map.cases {
        let measured_iter = measured_by_id
            .get(&joined.measured_iteration)
            .with_context(|| {
                format!(
                    "case {} points to missing measured iteration {}",
                    joined.case_index, joined.measured_iteration
                )
            })?;
        let sim = sim_cases.get(&joined.case_index).with_context(|| {
            format!(
                "missing timing-predict cost row for case {}",
                joined.case_index
            )
        })?;
        ensure!(
            sim.slot_ms.len() == manifest.slots.len(),
            "case {} slot_time_ms length {} != manifest slot count {}",
            joined.case_index,
            sim.slot_ms.len(),
            manifest.slots.len()
        );

        let ranges: Vec<_> = measured_iter
            .ranges
            .iter()
            .filter(|range| !range.kernels.is_empty())
            .collect();
        ensure!(
            !ranges.is_empty(),
            "measured iteration {} has no GPU kernel ranges",
            measured_iter.iteration
        );
        let devices: BTreeSet<_> = ranges.iter().filter_map(|range| range.device_id).collect();
        ensure!(
            devices.len() <= 1,
            "alignment v1 expects one measured device per iteration; iteration {} has {:?}",
            measured_iter.iteration,
            devices
        );

        let measured_intervals: Vec<_> = ranges
            .iter()
            .flat_map(|range| {
                range
                    .kernels
                    .iter()
                    .map(|kernel| (kernel.start_ns, kernel.end_ns))
            })
            .collect();
        let measured_total_ms = interval_union_ns(&measured_intervals) as f64 / 1e6;
        let delta_ms = sim.total_ms - measured_total_ms;
        let relative_pct = ratio_pct(delta_ms, measured_total_ms);
        cumulative_measured += measured_total_ms;
        cumulative_simulated += sim.total_ms;
        let cumulative_delta_ms = cumulative_simulated - cumulative_measured;
        let cumulative_relative_pct = ratio_pct(cumulative_delta_ms, cumulative_measured);
        total_delta.push(delta_ms);
        if let Some(value) = relative_pct {
            total_relative.push(value);
            total_abs_relative.push(value.abs());
        }

        let mut measured_ops: BTreeMap<String, f64> = BTreeMap::new();
        let mut sim_ops: BTreeMap<String, f64> = BTreeMap::new();
        let mut measured_kernels: BTreeMap<String, IterationKernelAggregate> = BTreeMap::new();
        let mut simulated_kernels = Vec::new();
        let mut phase_summaries = Vec::new();
        let mut iteration_unmapped_measured_ms = 0.0;
        let mut iteration_unmapped_simulated_ms = 0.0;

        let mut ranges_by_phase: BTreeMap<&str, Vec<&MeasuredRange>> = BTreeMap::new();
        for range in &ranges {
            ranges_by_phase.entry(&range.phase).or_default().push(range);
        }
        let mut phase_order: Vec<_> = ranges_by_phase
            .iter()
            .map(|(phase, phase_ranges)| {
                let first_start = phase_ranges
                    .iter()
                    .flat_map(|range| range.kernels.iter().map(|kernel| kernel.start_ns))
                    .min()
                    .unwrap_or(u64::MAX);
                (*phase, first_start)
            })
            .collect();
        phase_order.sort_by_key(|(_, first_start)| *first_start);

        for (phase, _) in phase_order {
            let phase_ranges = &ranges_by_phase[phase];
            let phase_inventory = inventory
                .phases
                .get(phase)
                .with_context(|| format!("labeled inventory has no phase {phase:?}"))?;
            let sequence_index = phase_inventory
                .sequence_by_iteration
                .get(&measured_iter.iteration)
                .with_context(|| {
                    format!(
                        "phase {phase:?} has no sequence for measured iteration {}",
                        measured_iter.iteration
                    )
                })?;
            let sequence = &phase_inventory.sequences[*sequence_index];
            let phase_kernels: Vec<_> = phase_ranges
                .iter()
                .flat_map(|range| range.kernels.iter())
                .collect();
            ensure!(
                phase_kernels.len() == sequence.rows.len(),
                "iteration {} phase {phase:?} has {} kernels but sequence {:?} has {} rows",
                measured_iter.iteration,
                phase_kernels.len(),
                sequence.sequence_id,
                sequence.rows.len()
            );

            let phase_intervals: Vec<_> = phase_kernels
                .iter()
                .map(|kernel| (kernel.start_ns, kernel.end_ns))
                .collect();
            let phase_kernel_sum_ms = phase_kernels
                .iter()
                .map(|kernel| (kernel.end_ns - kernel.start_ns) as f64 / 1e6)
                .sum::<f64>();
            phase_summaries.push(json!({
                "phase": phase,
                "busy_union_ms": interval_union_ns(&phase_intervals) as f64 / 1e6,
                "kernel_sum_ms": phase_kernel_sum_ms,
                "kernel_count": phase_kernels.len(),
            }));

            for (kernel, sequence_row) in phase_kernels.into_iter().zip(&sequence.rows) {
                ensure!(
                    kernel.end_ns >= kernel.start_ns,
                    "kernel has negative duration"
                );
                let name = kernel_names.get(&kernel.name_id).with_context(|| {
                    format!(
                        "kernel name_id {} missing from kernel_names",
                        kernel.name_id
                    )
                })?;
                ensure!(
                    sequence_row.name == *name,
                    "iteration {} phase {phase:?} row {} name mismatch",
                    measured_iter.iteration,
                    sequence_row.row_id,
                );
                ensure!(
                    sequence_row.suggested_category == kernel.category,
                    "iteration {} phase {phase:?} row {} category {:?} != {:?}",
                    measured_iter.iteration,
                    sequence_row.row_id,
                    sequence_row.suggested_category,
                    kernel.category,
                );
                let duration_ms = (kernel.end_ns - kernel.start_ns) as f64 / 1e6;
                measured_workload_ms += duration_ms;
                let operation = sequence_row
                    .operation
                    .as_ref()
                    .map(|operation| {
                        inventory
                            .operations
                            .get(operation)
                            .with_context(|| format!("unknown operation {operation:?}"))
                    })
                    .transpose()?;
                if let Some(operation) = operation {
                    *measured_ops.entry(operation.operation.clone()).or_default() += duration_ms;
                    measured_mapped_ms += duration_ms;
                } else {
                    iteration_unmapped_measured_ms += duration_ms;
                    let entry = unmapped_measured
                        .entry((phase.to_string(), sequence_row.row_id.clone()))
                        .or_insert_with(|| (sequence_row.name.clone(), 0, 0.0));
                    entry.1 += 1;
                    entry.2 += duration_ms;
                }
                let aggregate_key = format!("{phase}/{}", sequence_row.row_id);
                let aggregate = kernel_inventory.entry(aggregate_key.clone()).or_default();
                aggregate.phase = phase.to_string();
                aggregate.row_id = sequence_row.row_id.clone();
                aggregate.name = sequence_row.name.clone();
                aggregate.category = kernel.category.clone();
                aggregate.calls += 1;
                aggregate.total_ns += kernel.end_ns - kernel.start_ns;
                aggregate.iterations.insert(measured_iter.iteration);
                aggregate.operation = operation.map(|value| value.operation.clone());

                // The primary iteration breakdown is kernel-granular. Keep one
                // row per actual NSYS kernel identity, aggregating only repeated
                // launches of that exact name within this iteration.
                let iteration_kernel = measured_kernels.entry(aggregate_key).or_default();
                iteration_kernel.phase = phase.to_string();
                iteration_kernel.row_id = sequence_row.row_id.clone();
                iteration_kernel.name = sequence_row.name.clone();
                iteration_kernel.category = kernel.category.clone();
                iteration_kernel.calls += 1;
                iteration_kernel.total_ns += kernel.end_ns - kernel.start_ns;
                iteration_kernel.first_start_ns = Some(
                    iteration_kernel
                        .first_start_ns
                        .map_or(kernel.start_ns, |old| old.min(kernel.start_ns)),
                );
                iteration_kernel.operation = operation.map(|value| value.operation.clone());
            }
        }

        for (index, (slot, time_ms)) in manifest.slots.iter().zip(&sim.slot_ms).enumerate() {
            let folded_ms = *time_ms * scales[index] as f64;
            simulated_workload_ms += folded_ms;
            let operation = inventory
                .simulated_slots
                .get(&slot.name)
                .and_then(|operation| inventory.operations.get(operation));
            if let Some(operation) = operation {
                *sim_ops.entry(operation.operation.clone()).or_default() += folded_ms;
                simulated_mapped_ms += folded_ms;
            } else {
                iteration_unmapped_simulated_ms += folded_ms;
                *unmapped_simulated
                    .entry(format!("{} ({})", slot.name, slot.kind))
                    .or_default() += folded_ms;
            }
            simulated_kernels.push(json!({
                "slot_index": index,
                "name": slot.name,
                "kind": slot.kind,
                "operation": operation.map(|value| value.operation.as_str()),
                "multiplicity": scales[index],
                "unit_ms": time_ms,
                "folded_ms": folded_ms,
            }));
        }

        let operations: BTreeSet<_> = measured_ops.keys().chain(sim_ops.keys()).cloned().collect();
        let mut operation_rows = Vec::new();
        for operation in operations {
            let measured_ms = measured_ops.get(&operation).copied();
            let simulated_ms = sim_ops.get(&operation).copied();
            let (op_delta, op_relative) = match (measured_ms, simulated_ms) {
                (Some(m), Some(s)) => (Some(s - m), ratio_pct(s - m, m)),
                _ => (None, None),
            };
            let aggregate = operation_stats.entry(operation.clone()).or_default();
            match (measured_ms, simulated_ms) {
                (Some(m), Some(s)) => {
                    aggregate.measured_ms.push(m);
                    aggregate.simulated_ms.push(s);
                    aggregate.delta_ms.push(s - m);
                    if let Some(rel) = ratio_pct(s - m, m) {
                        aggregate.relative_pct.push(rel);
                        aggregate.abs_relative_pct.push(rel.abs());
                    }
                }
                (None, Some(_)) => aggregate.missing_measured += 1,
                (Some(_), None) => aggregate.missing_simulated += 1,
                (None, None) => unreachable!(),
            }
            operation_rows.push(json!({
                "operation": operation,
                "measured_ms": measured_ms,
                "simulated_ms": simulated_ms,
                "delta_ms": op_delta,
                "relative_diff_pct": op_relative,
            }));
        }

        let mut measured_kernel_entries: Vec<_> = measured_kernels.into_iter().collect();
        measured_kernel_entries.sort_by_key(|(_, item)| item.first_start_ns.unwrap_or(u64::MAX));
        let measured_kernel_rows: Vec<_> = measured_kernel_entries
            .into_iter()
            .map(|(_, item)| {
                json!({
                    "phase": item.phase,
                    "row_id": item.row_id,
                    "name": item.name,
                    "category": item.category,
                    "operation": item.operation,
                    "calls": item.calls,
                    "duration_ms": item.total_ns as f64 / 1e6,
                    "first_start_ns": item.first_start_ns,
                })
            })
            .collect();
        let measured_kernel_sum_ms = measured_kernel_rows
            .iter()
            .filter_map(|row| row["duration_ms"].as_f64())
            .sum::<f64>();
        let simulated_leaf_workload_ms = simulated_kernels
            .iter()
            .filter_map(|row| row["folded_ms"].as_f64())
            .sum::<f64>();

        iteration_rows.push(json!({
            "case_index": joined.case_index,
            "iteration_id": measured_iter.iteration,
            "stage": joined.stage,
            "iteration_type": measured_iter.iteration_type,
            "measured_ms": measured_total_ms,
            "simulated_ms": sim.total_ms,
            "delta_ms": delta_ms,
            "relative_diff_pct": relative_pct,
            "cumulative_delta_ms": cumulative_delta_ms,
            "cumulative_relative_diff_pct": cumulative_relative_pct,
        }));
        breakdowns.push(json!({
            "case_index": joined.case_index,
            "iteration_id": measured_iter.iteration,
            "stage": joined.stage,
            "measured_kernels": measured_kernel_rows,
            "simulated_kernels": simulated_kernels,
            "phase_summary": phase_summaries,
            "operation_summary": operation_rows,
            "measured_kernel_sum_ms": measured_kernel_sum_ms,
            "simulated_leaf_workload_ms": simulated_leaf_workload_ms,
            "unmapped_measured_ms": iteration_unmapped_measured_ms,
            "unmapped_simulated_ms": iteration_unmapped_simulated_ms,
        }));
    }

    let operation_report: Vec<_> = operation_stats
        .into_iter()
        .map(|(operation, samples)| {
            json!({
                "operation": operation,
                "n_paired": samples.delta_ms.len(),
                "missing_measured": samples.missing_measured,
                "missing_simulated": samples.missing_simulated,
                "measured_ms": signed_stats(&samples.measured_ms),
                "simulated_ms": signed_stats(&samples.simulated_ms),
                "delta_ms": signed_stats(&samples.delta_ms),
                "relative_diff_pct": signed_stats(&samples.relative_pct),
                "abs_relative_error_pct": stats(&clean_nonnegative_sorted(&samples.abs_relative_pct)),
            })
        })
        .collect();
    let kernel_report: Vec<_> = kernel_inventory
        .into_iter()
        .map(|(_, item)| {
            json!({
                "phase": item.phase,
                "row_id": item.row_id,
                "name": item.name,
                "category": item.category,
                "operation": item.operation,
                "calls": item.calls,
                "iterations": item.iterations.len(),
                "calls_per_iteration": item.calls as f64 / item.iterations.len().max(1) as f64,
                "total_ms": item.total_ns as f64 / 1e6,
                "mean_call_us": item.total_ns as f64 / item.calls.max(1) as f64 / 1e3,
            })
        })
        .collect();
    let mapping_operations: Vec<_> = inventory
        .operations
        .values()
        .map(|operation| {
            json!({
                "operation": operation.operation,
                "type": operation.kernel_type,
                "role": operation.role,
                "simulated_slots": operation.simulated_slots,
                "measured_rows": operation.measured_rows,
            })
        })
        .collect();

    let definitions = definitions();
    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "analysis_log_dir": log_dir.display().to_string(),
            "profile_log_dir": input.profile_log_dir.display().to_string(),
            "simulation_log_dir": input.simulation_log_dir.display().to_string(),
            "predict_log_dir": input.predict_log_dir.display().to_string(),
            "measured_phases": inventory.phases.keys().collect::<Vec<_>>(),
            "iterations": iteration_rows.len(),
        },
        "available": true,
        "total_iteration": {
            "delta_ms": signed_stats(&total_delta),
            "relative_diff_pct": signed_stats(&total_relative),
            "abs_relative_error_pct": stats(&clean_nonnegative_sorted(&total_abs_relative)),
        },
        "mapping": {
            "configured": true,
            "operations": mapping_operations,
            "coverage": {
                "measured_duration_fraction": fraction(measured_mapped_ms, measured_workload_ms),
                "simulated_workload_fraction": fraction(simulated_mapped_ms, simulated_workload_ms),
                "measured_mapped_ms": measured_mapped_ms,
                "measured_total_kernel_ms": measured_workload_ms,
                "simulated_mapped_ms": simulated_mapped_ms,
                "simulated_total_leaf_workload_ms": simulated_workload_ms,
            },
            "unmapped_measured_kernels": unmapped_measured.into_iter().map(|((phase, row_id), (name, calls, total_ms))| json!({
                "phase": phase, "row_id": row_id, "name": name, "calls": calls, "total_ms": total_ms
            })).collect::<Vec<_>>(),
            "unmapped_simulated_slots": unmapped_simulated.into_iter().map(|(slot, total_ms)| json!({
                "slot": slot, "total_ms": total_ms
            })).collect::<Vec<_>>(),
        },
        "operations": operation_report,
        "kernels": kernel_report,
        "iterations": iteration_rows,
        "definitions": definitions,
    });
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "analysis_log_dir": log_dir.display().to_string(),
            "profile_log_dir": input.profile_log_dir.display().to_string(),
            "measured_phases": inventory.phases.keys().collect::<Vec<_>>(),
        },
        "iterations": report["iterations"],
        "breakdowns": breakdowns,
        "definitions": definitions,
    });
    Ok((report, payload))
}

async fn load_sim_cases(
    ctx: &SessionContext,
    predict_log_dir: &Path,
    pool_tag: &str,
    worker_id: u16,
    manifest: &Manifest,
) -> Result<BTreeMap<u64, SimCase>> {
    let path = resolve_artifact_path(predict_log_dir, "cost_log");
    ensure!(
        register_if_exists(ctx, PREDICT_TABLE, path).await?,
        "timing-predict cost_log not found under {}",
        predict_log_dir.display()
    );
    require_columns(ctx, PREDICT_TABLE, PREDICT_COLUMNS).await?;
    let safe_pool = pool_tag.replace('\'', "''");
    let sql = format!(
        "SELECT iter_id, total_time_ms, slot_time_ms FROM {PREDICT_TABLE} \
         WHERE CAST(section AS VARCHAR) = 'iter' \
           AND CAST(pool_tag AS VARCHAR) = '{safe_pool}' AND worker_id = {worker_id} \
         ORDER BY iter_id"
    );
    let batches = collect(ctx, &sql).await?;
    let mut out = BTreeMap::new();
    for batch in &batches {
        let ids = col(batch, "iter_id")?;
        let totals = col(batch, "total_time_ms")?;
        let slots = col(batch, "slot_time_ms")?;
        for row in 0..batch.num_rows() {
            let case_index = value_f64(ids, row)? as u64;
            let total_ms = value_f64(totals, row)?;
            let slot_ms = value_f32_list(slots, row)?;
            let slot_ns: Vec<i64> = slot_ms.iter().map(|v| (v * 1e6).round() as i64).collect();
            let reconstructed_ms = node_time(manifest, 0, &slot_ns) as f64 / 1e6;
            ensure!(
                (reconstructed_ms - total_ms).abs() <= (total_ms.abs() * 1e-3).max(1e-6),
                "case {case_index}: cost-tree total {reconstructed_ms:.6} ms != row total {total_ms:.6} ms"
            );
            ensure!(
                out.insert(case_index, SimCase { total_ms, slot_ms })
                    .is_none(),
                "duplicate timing-predict case {case_index}"
            );
        }
    }
    Ok(out)
}

fn leaf_scales(manifest: &Manifest) -> Result<Vec<u64>> {
    let mut scales = vec![None; manifest.slots.len()];
    fn visit(manifest: &Manifest, index: usize, scale: u64, out: &mut [Option<u64>]) -> Result<()> {
        match manifest
            .nodes
            .get(index)
            .context("cost-tree node index out of bounds")?
        {
            FlatCostNode::Leaf(slot) => {
                let target = out
                    .get_mut(*slot)
                    .context("cost-tree leaf slot out of bounds")?;
                ensure!(
                    target.replace(scale).is_none(),
                    "cost-tree slot {slot} appears twice"
                );
            }
            FlatCostNode::Scale { n, children } => {
                ensure!(
                    children.len() == 1,
                    "Scale node must have exactly one child"
                );
                visit(manifest, children.start, scale * u64::from(*n), out)?;
            }
            FlatCostNode::Sum { children } | FlatCostNode::Max { children, .. } => {
                for child in children.clone() {
                    visit(manifest, child, scale, out)?;
                }
            }
        }
        Ok(())
    }
    visit(manifest, 0, 1, &mut scales)?;
    scales
        .into_iter()
        .enumerate()
        .map(|(index, value)| value.with_context(|| format!("manifest slot {index} unreachable")))
        .collect()
}

impl CompiledInventory {
    fn validate_slots(&self, manifest: &Manifest) -> Result<()> {
        let known_slots: BTreeSet<_> = manifest
            .slots
            .iter()
            .map(|slot| slot.name.as_str())
            .collect();
        for operation in self.operations.values() {
            for simulated_slot in &operation.simulated_slots {
                ensure!(
                    known_slots.contains(simulated_slot.as_str()),
                    "operation {:?} references unknown simulated slot {:?}",
                    operation.operation,
                    simulated_slot
                );
            }
        }
        Ok(())
    }
}

fn load_inventory(path: &Path) -> Result<CompiledInventory> {
    let doc: FoldedSequenceDoc = read_json(path)?;
    ensure!(
        doc.schema_version == 2,
        "labeled kernel sequences schema_version must be 2"
    );
    ensure!(
        doc.encoding == "folded-v1",
        "unsupported kernel sequence encoding"
    );
    ensure!(
        !doc.phases.is_empty(),
        "labeled kernel sequences has no phases"
    );

    let mut phases = BTreeMap::new();
    let mut operations: BTreeMap<String, OperationRule> = BTreeMap::new();
    let mut simulated_slots = BTreeMap::new();
    for (phase_name, phase) in doc.phases {
        ensure!(
            !phase.unique_sequences.is_empty(),
            "phase {phase_name:?} has no sequences"
        );
        let mut sequences = Vec::new();
        let mut sequence_by_iteration = BTreeMap::new();
        for sequence in phase.unique_sequences {
            let kernels = expand_nodes(&sequence.program)?;
            ensure!(
                kernels.len() == sequence.expanded_kernel_count,
                "sequence {:?} expands to {} kernels, expected {}",
                sequence.sequence_id,
                kernels.len(),
                sequence.expanded_kernel_count
            );
            let mut rows = Vec::with_capacity(kernels.len());
            for (ordinal, kernel) in kernels.into_iter().enumerate() {
                let operation =
                    compile_label(&kernel.label, &mut operations, &mut simulated_slots)?;
                rows.push(ExpandedRow {
                    row_id: format!("{}:{}", sequence.sequence_id, ordinal + 1),
                    name: kernel.name,
                    suggested_category: kernel.suggested_category,
                    operation,
                });
            }
            let sequence_index = sequences.len();
            for iteration in sequence.iterations {
                ensure!(
                    sequence_by_iteration
                        .insert(iteration, sequence_index)
                        .is_none(),
                    "phase {phase_name:?} assigns iteration {iteration} twice"
                );
            }
            sequences.push(ExpandedSequence {
                sequence_id: sequence.sequence_id,
                rows,
            });
        }
        ensure!(
            phases
                .insert(
                    phase_name,
                    PhaseInventory {
                        sequences,
                        sequence_by_iteration,
                    },
                )
                .is_none(),
            "duplicate phase"
        );
    }
    Ok(CompiledInventory {
        phases,
        operations,
        simulated_slots,
    })
}

fn expand_nodes(nodes: &[ProgramNode]) -> Result<Vec<LabeledKernel>> {
    ensure!(!nodes.is_empty(), "folded sequence program is empty");
    let mut kernels = Vec::new();
    for node in nodes {
        match node {
            ProgramNode::Kernels { kernels: literal } => {
                ensure!(!literal.is_empty(), "literal kernel node is empty");
                kernels.extend(literal.iter().cloned());
            }
            ProgramNode::Repeat { repeat } => {
                ensure!(repeat.count >= 2, "repeat count must be >= 2");
                ensure!(!repeat.body.kernels.is_empty(), "repeat body is empty");
                for _ in 0..repeat.count {
                    kernels.extend(repeat.body.kernels.iter().cloned());
                }
            }
        }
    }
    Ok(kernels)
}

fn compile_label(
    label: &EmbeddedLabel,
    operations: &mut BTreeMap<String, OperationRule>,
    simulated_slots: &mut BTreeMap<String, String>,
) -> Result<Option<String>> {
    if label.status == "unmapped" {
        ensure!(
            label.operation.is_none()
                && label.simulated_slots.is_none()
                && label.kernel_type.is_none()
                && label.role.is_none(),
            "unmapped label cannot contain mapping fields"
        );
        return Ok(None);
    }
    ensure!(
        label.status == "mapped",
        "label status must be mapped or unmapped"
    );
    let operation = label
        .operation
        .as_ref()
        .context("mapped label missing operation")?;
    let operation_slots = label
        .simulated_slots
        .as_ref()
        .context("mapped label missing simulated_slots")?;
    let kernel_type = label
        .kernel_type
        .as_ref()
        .context("mapped label missing type")?;
    let role = label.role.as_ref().context("mapped label missing role")?;
    ensure!(
        !operation.is_empty()
            && !operation_slots.is_empty()
            && operation_slots.iter().all(|slot| !slot.is_empty())
            && !kernel_type.is_empty()
            && !role.is_empty(),
        "mapped label fields must be non-empty"
    );
    let unique_slots: BTreeSet<_> = operation_slots.iter().collect();
    ensure!(
        unique_slots.len() == operation_slots.len(),
        "operation {operation:?} has duplicate simulated slots"
    );
    if let Some(existing) = operations.get_mut(operation) {
        ensure!(
            existing.simulated_slots == *operation_slots
                && existing.kernel_type == *kernel_type
                && existing.role == *role,
            "operation {operation:?} has inconsistent label metadata"
        );
        existing.measured_rows += 1;
    } else {
        operations.insert(
            operation.clone(),
            OperationRule {
                operation: operation.clone(),
                kernel_type: kernel_type.clone(),
                role: role.clone(),
                simulated_slots: operation_slots.clone(),
                measured_rows: 1,
            },
        );
    }
    for simulated_slot in operation_slots {
        if let Some(existing) = simulated_slots.insert(simulated_slot.clone(), operation.clone()) {
            ensure!(
                existing == *operation,
                "simulated slot {simulated_slot:?} belongs to multiple operations"
            );
        }
    }
    Ok(Some(operation.clone()))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

fn interval_union_ns(intervals: &[(u64, u64)]) -> u64 {
    let mut sorted: Vec<_> = intervals
        .iter()
        .copied()
        .filter(|(start, end)| end >= start)
        .collect();
    sorted.sort_unstable_by_key(|(start, end)| (*start, *end));
    let Some((mut start, mut end)) = sorted.first().copied() else {
        return 0;
    };
    let mut total = 0;
    for (next_start, next_end) in sorted.into_iter().skip(1) {
        if next_start <= end {
            end = end.max(next_end);
        } else {
            total += end - start;
            start = next_start;
            end = next_end;
        }
    }
    total + end - start
}

fn ratio_pct(delta: f64, reference: f64) -> Option<f64> {
    (reference.is_finite() && reference.abs() > 1e-12).then_some(delta / reference * 100.0)
}

fn fraction(part: f64, total: f64) -> Option<f64> {
    (total.is_finite() && total > 0.0).then_some(part / total)
}

fn signed_stats(samples: &[f64]) -> Value {
    let mut sorted: Vec<_> = samples.iter().copied().filter(|v| v.is_finite()).collect();
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
        "total_measured_ms": "union of CUDA kernel intervals across every captured NSYS phase",
        "total_simulated_ms": "timing-predict cost-tree total_time_ms (Sum/Max/Scale semantics preserved)",
        "relative_diff_pct": "(simulated - measured) / measured * 100; positive means overprediction",
        "operation_measured_ms": "sum of durations of measured CUDA kernels mapped to the operation",
        "operation_simulated_ms": "sum of mapped sim leaf times after CostTree Scale multiplicity; workload time, not additive wall time when Max/overlap exists",
        "measured_kernel_duration_ms": "sum of launch durations for one exact NSYS kernel identity within an iteration; repeated calls remain visible through calls",
        "simulated_kernel_folded_ms": "one L1 leaf slot time multiplied by its exact CostTree Scale multiplicity",
        "mapping_coverage": "duration/workload fraction assigned by embedded labels; unmatched entries stay explicit and are never filled with zero",
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
        "iterations": [],
        "breakdowns": [],
        "definitions": definitions(),
    });
    (report, payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_union_merges_overlap_across_phases() {
        assert_eq!(interval_union_ns(&[(10, 20), (15, 30), (40, 45)]), 25);
    }

    #[test]
    fn repeat_expansion_preserves_embedded_label() {
        let kernel = LabeledKernel {
            name: "kernel".into(),
            suggested_category: "other".into(),
            label: EmbeddedLabel {
                status: "unmapped".into(),
                operation: None,
                simulated_slots: None,
                kernel_type: None,
                role: None,
            },
        };
        let expanded = expand_nodes(&[ProgramNode::Repeat {
            repeat: RepeatNode {
                count: 3,
                body: RepeatBody {
                    kernels: vec![kernel],
                },
            },
        }])
        .unwrap();
        assert_eq!(expanded.len(), 3);
        assert!(expanded
            .iter()
            .all(|kernel| kernel.label.status == "unmapped"));
    }

    #[test]
    fn mapped_label_assigns_multiple_simulated_slots_once() {
        let label = EmbeddedLabel {
            status: "mapped".into(),
            operation: Some("attention".into()),
            simulated_slots: Some(vec!["attention.main".into(), "attention.combine".into()]),
            kernel_type: Some("attention".into()),
            role: Some("attention main and combine".into()),
        };
        let mut operations = BTreeMap::new();
        let mut slots = BTreeMap::new();

        let operation = compile_label(&label, &mut operations, &mut slots).unwrap();

        assert_eq!(operation.as_deref(), Some("attention"));
        assert_eq!(
            operations["attention"].simulated_slots,
            ["attention.main", "attention.combine"]
        );
        assert_eq!(slots["attention.main"], "attention");
        assert_eq!(slots["attention.combine"], "attention");
    }
}
