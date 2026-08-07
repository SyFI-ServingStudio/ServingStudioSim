//! Per-iteration measured ↔ predicted alignment.
//!
//! The launcher has already normalized the vLLM input into timing-predict cases.
//! This subject owns the actual comparison: join case indices to measured
//! iteration ids, expand the user-labeled folded inventory across every measured
//! phase, attribute sim leaves through the exact CostTree critical path, and
//! emit totals, operation errors, kernel inventory, and plot arrays. Raw folded
//! leaf workload remains separate for mapping-coverage audit.
//!
//! The measured half — validating a capture against the labeled inventory and
//! reducing it to per-kernel-position rows — is shared with [`timeline`], which
//! lays the same rows on a time axis instead of summing them.

pub(crate) mod host;
pub(crate) mod timeline;

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
use crate::trace::manifest::{node_time, FlatCostNode, Manifest, ManifestDoc};

/// Sibling of the payload holding one iteration's kernel breakdown per line.
const BREAKDOWN_DETAIL_FILE: &str = "alignment_iteration_breakdowns.jsonl";

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
    #[serde(default)]
    device_ids: Option<Vec<i64>>,
    #[serde(default)]
    representative_device_id: Option<i64>,
    phases: BTreeMap<String, FoldedPhase>,
}

#[derive(Deserialize)]
struct FoldedPhase {
    unique_sequences: Vec<FoldedSequence>,
}

#[derive(Deserialize)]
struct FoldedSequence {
    sequence_id: String,
    /// Schema 2/3: the sequence applies to every measured device. Schema 4
    /// replaces this with `occurrences`, which carries the device axis.
    #[serde(default)]
    iterations: Option<Vec<u64>>,
    #[serde(default)]
    occurrences: Option<Vec<FoldedOccurrence>>,
    expanded_kernel_count: usize,
    program: Vec<ProgramNode>,
}

#[derive(Deserialize)]
struct FoldedOccurrence {
    device_id: i64,
    iterations: Vec<u64>,
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
    /// Per-kernel cross-rank reduction class from the mapping table:
    /// "synchronizing" (a collective barrier) or "independent" (default when
    /// absent, e.g. schema-v2 single-rank captures). The analyzer never derives
    /// this from a category or name; it only consumes what the labeler emits.
    #[serde(default)]
    cross_rank: Option<String>,
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
    /// The NVTX phase marker's own bounds — a HOST fact, and the only one the
    /// device side reads. Under CUDA graphs the range routinely closes before
    /// its own kernels finish, so it bounds nothing on the device; it is here to
    /// widen an iteration's host window, never to measure GPU time.
    start_ns: u64,
    end_ns: u64,
    kernels: Vec<MeasuredKernel>,
}

#[derive(Deserialize)]
struct MeasuredKernel {
    name_id: u64,
    category: String,
    start_ns: u64,
    end_ns: u64,
    #[serde(default)]
    correlation_id: Option<u64>,
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
    /// One simulated slot may be declared by several operations — a fused
    /// boundary owns it together with its norm, while an unfused boundary's
    /// split collective op owns it alone. The claiming operation is resolved
    /// per iteration from the operations actually present that iteration.
    simulated_slots: BTreeMap<String, BTreeSet<String>>,
    device_ids: Option<BTreeSet<i64>>,
    representative_device_id: Option<i64>,
    /// Schema 4: sequences carry a device axis and ranks may diverge.
    is_union_catalog: bool,
}

struct PhaseInventory {
    sequences: Vec<ExpandedSequence>,
    /// `(device, iteration) -> sequence`. A `None` device is the schema-2/3
    /// shape: one labeling decision that every measured device shares. Schema 4
    /// (data-parallel captures) names the device, because ranks schedule their
    /// own batches and so can run different kernel sequences in the same step.
    sequence_by_position: BTreeMap<(Option<i64>, u64), usize>,
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
    synchronizing: bool,
}

#[derive(Default)]
struct KernelAggregate {
    phase: String,
    row_id: String,
    name: String,
    category: String,
    operation: Option<String>,
    calls: usize,
    intervals: Vec<(u64, u64)>,
    iterations: BTreeSet<u64>,
    device_ids: BTreeSet<i64>,
}

/// One rank's launch of one kernel position, kept unreduced.
///
/// The cross-rank reduction is a judgment (`occurrence_ns`), so the raw per-rank
/// interval has to survive up to the point that judgment is made — and the
/// timeline subject needs every rank's real interval, not the reduction.
#[derive(Clone, Copy)]
struct KernelLaunch {
    device_id: i64,
    start_ns: u64,
    end_ns: u64,
    correlation_id: Option<u64>,
}

#[derive(Default)]
struct IterationKernelAggregate {
    phase: String,
    sequence_id: String,
    row_id: String,
    name: String,
    name_id: u64,
    category: String,
    operation: Option<String>,
    synchronizing: bool,
    launches: Vec<KernelLaunch>,
    first_start_ns: Option<u64>,
    device_ids: BTreeSet<i64>,
}

/// One `(device, phase)` group's GPU occupancy, per measured iteration.
struct PhaseSummary {
    device_id: i64,
    phase: String,
    busy_union_ms: f64,
    kernel_sum_ms: f64,
    kernel_count: usize,
    /// First kernel start and last kernel end on this rank in this phase — the
    /// correlated kernel span. Idle is measured against THIS, never against the
    /// NVTX range: under CUDA graphs the range can close before its own kernels
    /// finish, and the resulting negative idle clamps to zero and hides the stall.
    span_ns: (u64, u64),
}

/// Everything one measured iteration says about the GPU, before any comparison
/// against a simulation. Shared by the `alignment-iteration` report and the
/// `alignment-timeline` payload so the two can never disagree about what was
/// measured — only about what they do with it.
struct IterationMeasurement {
    /// One entry per `{phase}/{row_id}`, ordered by first launch start.
    kernels: Vec<(String, IterationKernelAggregate)>,
    phase_summaries: Vec<PhaseSummary>,
    device_ids: BTreeSet<i64>,
    /// Audit-only union across every rank and phase.
    busy_union_ms: f64,
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
    let input = alignment_input::read_kernel_align(log_dir)?;
    let measured: ParsedTrace = read_json(&input.parsed_nsys)?;
    let case_map: CaseMapDoc = read_json(&input.timing_predict_case_map)?;
    ensure!(
        case_map.schema_version == 1,
        "case-map schema_version must be 1"
    );
    let inventory = load_inventory(&input.labeled_kernel_sequences)?;

    let manifests = read_cost_manifests(&input.predict_log_dir)?;
    let (pool_tag, worker_id, manifest) = single_iter_manifest(&manifests)?;
    let scales = leaf_scales(manifest)?;
    let sim_cases =
        load_sim_cases(ctx, &input.predict_log_dir, pool_tag, worker_id, manifest).await?;

    let measured_by_id: BTreeMap<u64, &MeasuredIteration> = measured
        .iteration_details
        .iter()
        .map(|item| (item.iteration, item))
        .collect();
    let measured_gpu_cycles_ms = measured_gpu_cycles_ms(&measured.iteration_details)?;
    let kernel_names = kernel_name_index(&measured)?;
    inventory.validate_slots(manifest)?;

    let mut iteration_rows = Vec::new();
    // Per-iteration kernel breakdowns are the bulk of this subject — 550 MB of
    // the old single-document payload, for a view that opens one iteration at a
    // time. They are serialized straight into a shard buffer as they are built,
    // so neither this process nor a reader ever holds all of them.
    let mut breakdown_bytes: Vec<u8> = Vec::new();
    let mut breakdown_ranges = serde_json::Map::new();
    let mut total_delta = Vec::new();
    let mut total_relative = Vec::new();
    let mut total_abs_relative = Vec::new();
    let mut cumulative_measured = 0.0;
    let mut cumulative_simulated = 0.0;
    // The GPU-cycle columns depend on the pooled duty-cycle multiplier
    // (Σ measured GPU cycle / Σ measured_ms), which is only known after the loop
    // has seen every iteration. Collect the per-iteration inputs here and fill
    // those columns in a second pass below.
    let mut gpu_cycle_inputs: Vec<(Option<f64>, f64)> = Vec::new();
    let mut sum_measured_gpu_cycle = 0.0;
    let mut sum_measured_ms_with_cycle = 0.0;
    let mut kernel_inventory: BTreeMap<String, KernelAggregate> = BTreeMap::new();
    let mut operation_stats: BTreeMap<String, OperationAggregate> = BTreeMap::new();
    let mut unmapped_measured: BTreeMap<
        (String, String),
        (String, usize, Vec<(u64, u64)>, BTreeSet<i64>),
    > = BTreeMap::new();
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

        let measurement = measure_iteration(measured_iter, &inventory, &kernel_names)?;
        let measured_busy_union_ms = measurement.busy_union_ms;
        let measured_gpu_cycle_ms = measured_gpu_cycles_ms
            .get(&measured_iter.iteration)
            .copied();

        let mut sim_ops: BTreeMap<String, f64> = BTreeMap::new();
        let mut simulated_kernels = Vec::new();
        let mut iteration_unmapped_simulated_ms = 0.0;
        // Keep folded workload for mapping coverage, but attribute visible
        // simulated time through the CostTree's Sum/Max/Scale semantics. An
        // EP Max branch must contribute only its winning leaf to the UI path.
        let critical_path_ms_by_slot = critical_path_leaf_ms(manifest, &sim.slot_ms)?;
        let attributed_simulated_ms: f64 = critical_path_ms_by_slot.iter().sum();
        ensure!(
            (attributed_simulated_ms - sim.total_ms).abs() <= (sim.total_ms.abs() * 1e-3).max(1e-6),
            "case {}: attributed critical path {:.6} ms != row total {:.6} ms",
            joined.case_index,
            attributed_simulated_ms,
            sim.total_ms,
        );

        let phase_summaries: Vec<_> = measurement
            .phase_summaries
            .iter()
            .map(|summary| {
                json!({
                    "device_id": summary.device_id,
                    "phase": summary.phase,
                    "busy_union_ms": summary.busy_union_ms,
                    "kernel_sum_ms": summary.kernel_sum_ms,
                    "kernel_count": summary.kernel_count,
                })
            })
            .collect();

        // Fold this iteration's rows into the run-wide inventory and the
        // unmapped-kernel audit. Both are order-insensitive (their durations go
        // through `interval_union_ns`, which sorts), so folding per row here is
        // equivalent to folding per launch during the scan.
        for (aggregate_key, item) in &measurement.kernels {
            if item.operation.is_none() {
                let entry = unmapped_measured
                    .entry((item.phase.clone(), item.row_id.clone()))
                    .or_insert_with(|| (item.name.clone(), 0, Vec::new(), BTreeSet::new()));
                entry.1 += item.launches.len();
                entry
                    .2
                    .extend(item.launches.iter().map(|l| (l.start_ns, l.end_ns)));
                entry.3.extend(item.device_ids.iter().copied());
            }
            let aggregate = kernel_inventory.entry(aggregate_key.clone()).or_default();
            aggregate.phase = item.phase.clone();
            aggregate.row_id = item.row_id.clone();
            aggregate.name = item.name.clone();
            aggregate.category = item.category.clone();
            aggregate.calls += item.launches.len();
            aggregate
                .intervals
                .extend(item.launches.iter().map(|l| (l.start_ns, l.end_ns)));
            aggregate.iterations.insert(measured_iter.iteration);
            aggregate.device_ids.extend(item.device_ids.iter().copied());
            aggregate.operation = item.operation.clone();
        }

        // Reduce every measured occurrence across its ranks into one replica
        // critical-path contribution, then roll those up. Occurrences merge only
        // the ranks that ran the same kernel sequence, so under data parallelism
        // a step's rows are split across disjoint rank sets. Those sets run
        // concurrently and must not be summed. Charge every reduced occurrence
        // back to each rank that raised it, sum along each rank's own timeline,
        // then take the slowest rank. With every rank on one sequence each rank
        // accumulates the identical series in the identical order, so this is
        // arithmetically identical to the flat sum it replaces.
        let mut device_ops: BTreeMap<i64, BTreeMap<String, f64>> = BTreeMap::new();
        let mut device_mapped_ms: BTreeMap<i64, f64> = BTreeMap::new();
        let mut device_unmapped_ms: BTreeMap<i64, f64> = BTreeMap::new();
        let mut measured_kernel_rows = Vec::with_capacity(measurement.kernels.len());
        for (_, item) in &measurement.kernels {
            let device_count = item.device_ids.len().max(1);
            let duration_ms = occurrence_ns(&item.launches, item.synchronizing) as f64 / 1e6;
            for device_id in &item.device_ids {
                match &item.operation {
                    Some(operation) => {
                        *device_ops
                            .entry(*device_id)
                            .or_default()
                            .entry(operation.clone())
                            .or_default() += duration_ms;
                        *device_mapped_ms.entry(*device_id).or_default() += duration_ms;
                    }
                    None => *device_unmapped_ms.entry(*device_id).or_default() += duration_ms,
                }
            }
            measured_kernel_rows.push(json!({
                "phase": item.phase,
                "sequence_id": item.sequence_id,
                "row_id": item.row_id,
                "name": item.name,
                "category": item.category,
                "operation": item.operation,
                "synchronizing": item.synchronizing,
                "calls": item.launches.len(),
                "rank_launches": item.launches.len(),
                "replica_calls": item.launches.len() as f64 / device_count as f64,
                "duration_ms": duration_ms,
                "first_start_ns": item.first_start_ns,
                "device_ids": item.device_ids,
            }));
        }
        let mut measured_ops: BTreeMap<String, f64> = BTreeMap::new();
        for operations in device_ops.values() {
            for (operation, device_ms) in operations {
                let slowest = measured_ops.entry(operation.clone()).or_default();
                *slowest = slowest.max(*device_ms);
            }
        }
        let iteration_mapped_ms = device_mapped_ms.values().copied().fold(0.0f64, f64::max);
        let iteration_unmapped_measured_ms =
            device_unmapped_ms.values().copied().fold(0.0f64, f64::max);
        let measured_kernel_sum_ms = iteration_mapped_ms + iteration_unmapped_measured_ms;
        let measured_critical_path_ms = measured_kernel_sum_ms;
        // Feed the pooled duty-cycle multiplier. Only iterations that have a
        // measured GPU cycle contribute, so the ratio's numerator and denominator
        // span exactly the same iterations (the final iteration has no cycle).
        if let Some(cycle) = measured_gpu_cycle_ms {
            sum_measured_gpu_cycle += cycle;
            sum_measured_ms_with_cycle += measured_critical_path_ms;
        }
        gpu_cycle_inputs.push((measured_gpu_cycle_ms, sim.total_ms));

        // Operations that actually appear in this iteration's measured kernels.
        // A sim slot declared by several operations (fused aggregate vs unfused
        // split) resolves to whichever of them is present this iteration.
        let present_operations: BTreeSet<&str> = measured_ops.keys().map(String::as_str).collect();
        for (index, (slot, time_ms)) in manifest.slots.iter().zip(&sim.slot_ms).enumerate() {
            let folded_ms = *time_ms * scales[index] as f64;
            let critical_path_ms = critical_path_ms_by_slot[index];
            simulated_workload_ms += folded_ms;
            let operation = inventory.resolve_slot_operation(
                &slot.name,
                &present_operations,
                measured_iter.iteration,
            )?;
            if let Some(operation) = operation {
                *sim_ops.entry(operation.operation.clone()).or_default() += critical_path_ms;
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
                "critical_path_ms": critical_path_ms,
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

        let simulated_leaf_workload_ms = simulated_kernels
            .iter()
            .filter_map(|row| row["folded_ms"].as_f64())
            .sum::<f64>();
        let simulated_critical_path_ms = simulated_kernels
            .iter()
            .filter_map(|row| row["critical_path_ms"].as_f64())
            .sum::<f64>();

        // Headline totals use the critical-path sum, so per-operation rows and
        // per-kernel segments add up to it. The busy union is kept for audit.
        measured_workload_ms += measured_critical_path_ms;
        measured_mapped_ms += iteration_mapped_ms;
        let delta_ms = sim.total_ms - measured_critical_path_ms;
        let relative_pct = ratio_pct(delta_ms, measured_critical_path_ms);
        cumulative_measured += measured_critical_path_ms;
        cumulative_simulated += sim.total_ms;
        let cumulative_delta_ms = cumulative_simulated - cumulative_measured;
        let cumulative_relative_pct = ratio_pct(cumulative_delta_ms, cumulative_measured);
        total_delta.push(delta_ms);
        if let Some(value) = relative_pct {
            total_relative.push(value);
            total_abs_relative.push(value.abs());
        }

        iteration_rows.push(json!({
            "case_index": joined.case_index,
            "iteration_id": measured_iter.iteration,
            "stage": joined.stage,
            "iteration_type": measured_iter.iteration_type,
            "measured_ms": measured_critical_path_ms,
            "measured_busy_union_ms": measured_busy_union_ms,
            "simulated_ms": sim.total_ms,
            "delta_ms": delta_ms,
            "relative_diff_pct": relative_pct,
            "cumulative_delta_ms": cumulative_delta_ms,
            "cumulative_relative_diff_pct": cumulative_relative_pct,
        }));
        let breakdown = json!({
            "case_index": joined.case_index,
            "iteration_id": measured_iter.iteration,
            "stage": joined.stage,
            "measured_kernels": measured_kernel_rows,
            "simulated_kernels": simulated_kernels,
            "phase_summary": phase_summaries,
            "operation_summary": operation_rows,
            "measured_kernel_sum_ms": measured_kernel_sum_ms,
            "simulated_leaf_workload_ms": simulated_leaf_workload_ms,
            "simulated_critical_path_ms": simulated_critical_path_ms,
            "unmapped_measured_ms": iteration_unmapped_measured_ms,
            "unmapped_simulated_ms": iteration_unmapped_simulated_ms,
        });
        let line = serde_json::to_vec(&breakdown)?;
        breakdown_ranges.insert(
            measured_iter.iteration.to_string(),
            json!([breakdown_bytes.len(), line.len()]),
        );
        breakdown_bytes.extend_from_slice(&line);
        breakdown_bytes.push(b'\n');
    }

    // Pooled duty-cycle correction, derived from the measured side only:
    // Σ measured GPU cycle / Σ measured_ms over iterations that have a cycle.
    // Applying it as simulated_gpu_cycle = simulated_ms × recommended makes the
    // pooled GPU-cycle gap equal the pooled kernel gap by construction, so the
    // kernel-align and duty-cycle levels can never disagree. This value is what
    // the aligned simulation worker should bake into its clock.
    let recommended_gpu_time_multiplier =
        pooled_gpu_time_multiplier(sum_measured_gpu_cycle, sum_measured_ms_with_cycle);
    ensure!(
        recommended_gpu_time_multiplier.is_finite() && recommended_gpu_time_multiplier >= 1.0,
        "recommended_gpu_time_multiplier must be finite and >= 1.0; found {recommended_gpu_time_multiplier}"
    );
    let mut cumulative_measured_gpu_cycle = 0.0;
    let mut cumulative_simulated_gpu_cycle = 0.0;
    for (row, &(measured_gpu_cycle_ms, sim_total_ms)) in
        iteration_rows.iter_mut().zip(&gpu_cycle_inputs)
    {
        let simulated_gpu_cycle_ms =
            measured_gpu_cycle_ms.map(|_| sim_total_ms * recommended_gpu_time_multiplier);
        let (
            gpu_cycle_delta_ms,
            gpu_cycle_relative_diff_pct,
            gpu_cycle_cumulative_delta_ms,
            gpu_cycle_cumulative_relative_diff_pct,
        ) = match (measured_gpu_cycle_ms, simulated_gpu_cycle_ms) {
            (Some(measured_cycle), Some(simulated_cycle)) => {
                cumulative_measured_gpu_cycle += measured_cycle;
                cumulative_simulated_gpu_cycle += simulated_cycle;
                let delta = simulated_cycle - measured_cycle;
                let cumulative_delta =
                    cumulative_simulated_gpu_cycle - cumulative_measured_gpu_cycle;
                (
                    Some(delta),
                    ratio_pct(delta, measured_cycle),
                    Some(cumulative_delta),
                    ratio_pct(cumulative_delta, cumulative_measured_gpu_cycle),
                )
            }
            _ => (None, None, None, None),
        };
        if let Value::Object(map) = row {
            map.insert("measured_gpu_cycle_ms".into(), json!(measured_gpu_cycle_ms));
            map.insert(
                "simulated_gpu_cycle_ms".into(),
                json!(simulated_gpu_cycle_ms),
            );
            map.insert("gpu_cycle_delta_ms".into(), json!(gpu_cycle_delta_ms));
            map.insert(
                "gpu_cycle_relative_diff_pct".into(),
                json!(gpu_cycle_relative_diff_pct),
            );
            map.insert(
                "gpu_cycle_cumulative_delta_ms".into(),
                json!(gpu_cycle_cumulative_delta_ms),
            );
            map.insert(
                "gpu_cycle_cumulative_relative_diff_pct".into(),
                json!(gpu_cycle_cumulative_relative_diff_pct),
            );
        }
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
            let device_count = item.device_ids.len().max(1);
            let replica_calls = item.calls as f64 / device_count as f64;
            let total_union_ns = interval_union_ns(&item.intervals) as f64;
            json!({
                "phase": item.phase,
                "row_id": item.row_id,
                "name": item.name,
                "category": item.category,
                "operation": item.operation,
                "calls": item.calls,
                "rank_launches": item.calls,
                "replica_calls": replica_calls,
                "iterations": item.iterations.len(),
                "calls_per_iteration": item.calls as f64 / item.iterations.len().max(1) as f64,
                "replica_calls_per_iteration": replica_calls
                    / item.iterations.len().max(1) as f64,
                "total_ms": total_union_ns / 1e6,
                "mean_call_us": total_union_ns / replica_calls.max(1.0) / 1e3,
                "device_ids": item.device_ids,
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
            "predict_log_dir": input.predict_log_dir.display().to_string(),
            "measured_phases": inventory.phases.keys().collect::<Vec<_>>(),
            "measured_device_ids": inventory.device_ids,
            "representative_device_id": inventory.representative_device_id,
            "iterations": iteration_rows.len(),
            "recommended_gpu_time_multiplier": recommended_gpu_time_multiplier,
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
            "unmapped_measured_kernels": unmapped_measured.into_iter().map(|((phase, row_id), (name, calls, intervals, device_ids))| json!({
                "phase": phase,
                "row_id": row_id,
                "name": name,
                "calls": calls,
                "total_ms": interval_union_ns(&intervals) as f64 / 1e6,
                "device_ids": device_ids,
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
    // Written before the payload is returned, so an index can never name a shard
    // the caller failed to write.
    let breakdown_path = crate::io::payload_path(log_dir, BREAKDOWN_DETAIL_FILE);
    if let Some(parent) = breakdown_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&breakdown_path, &breakdown_bytes)
        .with_context(|| format!("write {}", breakdown_path.display()))?;
    println!("wrote {}", breakdown_path.display());

    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "analysis_log_dir": log_dir.display().to_string(),
            "profile_log_dir": input.profile_log_dir.display().to_string(),
            "measured_phases": inventory.phases.keys().collect::<Vec<_>>(),
            "recommended_gpu_time_multiplier": recommended_gpu_time_multiplier,
        },
        "iterations": report["iterations"],
        // The labelled kernel programs, verbatim. Carried here so a client
        // reads one artifact family instead of also opening the labeler's input
        // file, which is not a payload and has its own lifecycle.
        "sequences": sequence_document(&input.labeled_kernel_sequences)?,
        "breakdown_detail": {
            "file": BREAKDOWN_DETAIL_FILE,
            "encoding": "one JSON object per line, in the order of `iterations`",
            "byte_ranges": breakdown_ranges,
        },
        "definitions": definitions,
    });
    Ok((report, payload))
}

/// The labeled sequence inventory as written, minus its provenance envelope.
///
/// Read again rather than reconstructed from [`CompiledInventory`]: that struct
/// expands the folded program, and the folding is exactly what a reader wants to
/// draw — one `repeat{32}` band, not 32 identical rows.
fn sequence_document(path: &Path) -> Result<Value> {
    let document: Value = read_json(path)?;
    Ok(json!({
        "encoding": document.get("encoding"),
        "folding_policy": document.get("folding_policy"),
        "device_ids": document.get("device_ids"),
        "representative_device_id": document.get("representative_device_id"),
        "phases": document.get("phases"),
    }))
}

/// The one predict worker whose `iter` cost tree the alignment compares against.
/// More than one would make "the simulated iteration" ambiguous, so it is an
/// error rather than a pick.
fn single_iter_manifest(
    manifests: &BTreeMap<(String, u16), ManifestDoc>,
) -> Result<(&str, u16, &Manifest)> {
    let found: Vec<_> = manifests
        .iter()
        .filter_map(|(key, doc)| doc.section("iter").map(|manifest| (key, manifest)))
        .collect();
    ensure!(
        found.len() == 1,
        "alignment v1 expects exactly one predict worker with an iter manifest; found {}",
        found.len()
    );
    let ((pool_tag, worker_id), manifest) = found[0];
    Ok((pool_tag.as_str(), *worker_id, manifest))
}

/// `parsed.json` keys its kernel-name dictionary by stringified id; every consumer
/// wants it back as the numeric id the kernel rows carry.
fn kernel_name_index(measured: &ParsedTrace) -> Result<BTreeMap<u64, &str>> {
    measured
        .kernel_names
        .iter()
        .map(|(id, name)| {
            id.parse::<u64>()
                .map(|id| (id, name.as_str()))
                .with_context(|| format!("parse kernel name id {id:?}"))
        })
        .collect()
}

/// Reduce one measured iteration's NSYS ranges to per-kernel-position rows.
///
/// This is the whole measured side of alignment: validate the capture against the
/// labeled inventory, group launches by `{phase}/{row_id}`, and keep every rank's
/// raw interval. It deliberately stops before any comparison — `run` turns these
/// rows into operation totals, `timeline` turns the same rows into slices, and
/// neither can drift from the other about what the GPU did.
fn measure_iteration(
    measured_iter: &MeasuredIteration,
    inventory: &CompiledInventory,
    kernel_names: &BTreeMap<u64, &str>,
) -> Result<IterationMeasurement> {
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
    let device_ids: BTreeSet<_> = ranges.iter().filter_map(|range| range.device_id).collect();
    if let Some(expected_devices) = &inventory.device_ids {
        // A data-parallel rank can sit out a step entirely (it has no work to
        // schedule), so under the union catalog the measured devices are a
        // non-empty SUBSET of the labeled population, not necessarily all of it.
        // Symmetric captures still demand every rank every step.
        if inventory.is_union_catalog {
            ensure!(
                device_ids.is_subset(expected_devices),
                "measured iteration {} devices {:?} are not all in the labeled inventory {:?}",
                measured_iter.iteration,
                device_ids,
                expected_devices
            );
        } else {
            ensure!(
                &device_ids == expected_devices,
                "measured iteration {} devices {:?} != labeled inventory devices {:?}",
                measured_iter.iteration,
                device_ids,
                expected_devices
            );
        }
    } else {
        ensure!(
            device_ids.len() <= 1,
            "schema-v2 labeled inventory supports one measured device; iteration {} has {:?}",
            measured_iter.iteration,
            device_ids
        );
    }

    // Audit-only replica GPU-busy union (every rank, every kernel, once). The
    // headline `measured_ms` is the critical-path sum the caller builds from
    // per-occurrence cross-rank reductions; the two differ by cross-op rank skew
    // and are both reported.
    let measured_intervals: Vec<_> = ranges
        .iter()
        .flat_map(|range| {
            range
                .kernels
                .iter()
                .map(|kernel| (kernel.start_ns, kernel.end_ns))
        })
        .collect();
    let busy_union_ms = interval_union_ns(&measured_intervals) as f64 / 1e6;

    let mut ranges_by_device_phase: BTreeMap<(i64, &str), Vec<&MeasuredRange>> = BTreeMap::new();
    for range in &ranges {
        let device_id = range.device_id.context("measured range has no device_id")?;
        ranges_by_device_phase
            .entry((device_id, &range.phase))
            .or_default()
            .push(range);
    }
    let mut phase_order: Vec<_> = ranges_by_device_phase
        .iter()
        .map(|((device_id, phase), phase_ranges)| {
            let first_start = phase_ranges
                .iter()
                .flat_map(|range| range.kernels.iter().map(|kernel| kernel.start_ns))
                .min()
                .unwrap_or(u64::MAX);
            (*device_id, *phase, first_start)
        })
        .collect();
    phase_order.sort_by_key(|(device_id, _, first_start)| (*first_start, *device_id));

    let mut kernels: BTreeMap<String, IterationKernelAggregate> = BTreeMap::new();
    let mut phase_summaries = Vec::new();
    for (device_id, phase, _) in phase_order {
        let phase_ranges = &ranges_by_device_phase[&(device_id, phase)];
        let phase_inventory = inventory
            .phases
            .get(phase)
            .with_context(|| format!("labeled inventory has no phase {phase:?}"))?;
        // Prefer this device's own labeling decision (schema 4); fall back to
        // the device-agnostic one that schema 2/3 records for all ranks.
        let sequence_index = phase_inventory
            .sequence_by_position
            .get(&(Some(device_id), measured_iter.iteration))
            .or_else(|| {
                phase_inventory
                    .sequence_by_position
                    .get(&(None, measured_iter.iteration))
            })
            .with_context(|| {
                format!(
                    "phase {phase:?} has no sequence for measured iteration {} on device \
                     {device_id}",
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
        phase_summaries.push(PhaseSummary {
            device_id,
            phase: phase.to_string(),
            busy_union_ms: interval_union_ns(&phase_intervals) as f64 / 1e6,
            kernel_sum_ms: phase_kernel_sum_ms,
            kernel_count: phase_kernels.len(),
            span_ns: (
                phase_intervals
                    .iter()
                    .map(|(start, _)| *start)
                    .min()
                    .unwrap_or(0),
                phase_intervals
                    .iter()
                    .map(|(_, end)| *end)
                    .max()
                    .unwrap_or(0),
            ),
        });

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

            // One row per actual NSYS kernel identity, aggregating only repeated
            // launches of that exact position within this iteration.
            let item = kernels
                .entry(format!("{phase}/{}", sequence_row.row_id))
                .or_default();
            item.phase = phase.to_string();
            item.row_id = sequence_row.row_id.clone();
            item.name = sequence_row.name.clone();
            item.name_id = kernel.name_id;
            item.category = kernel.category.clone();
            item.synchronizing = sequence_row.synchronizing;
            item.launches.push(KernelLaunch {
                device_id,
                start_ns: kernel.start_ns,
                end_ns: kernel.end_ns,
                correlation_id: kernel.correlation_id,
            });
            item.device_ids.insert(device_id);
            item.first_start_ns = Some(
                item.first_start_ns
                    .map_or(kernel.start_ns, |old| old.min(kernel.start_ns)),
            );
            item.operation = operation.map(|value| value.operation.clone());
        }
    }

    let mut kernels: Vec<_> = kernels.into_iter().collect();
    kernels.sort_by_key(|(_, item)| item.first_start_ns.unwrap_or(u64::MAX));
    Ok(IterationMeasurement {
        kernels,
        phase_summaries,
        device_ids,
        busy_union_ms,
    })
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

/// Attribute one prediction row back to leaf slots while exactly preserving
/// the CostTree root. A `Max` selects its slowest child; exact ties retain the
/// first child, which is the stable rank-0 branch for EP fan-out trees.
fn critical_path_leaf_ms(manifest: &Manifest, slot_ms: &[f64]) -> Result<Vec<f64>> {
    const TIME_EPSILON_MS: f64 = 1e-12;

    ensure!(!manifest.nodes.is_empty(), "cost manifest has no root node");
    ensure!(
        slot_ms.len() == manifest.slots.len(),
        "slot_time_ms length {} != manifest slot count {}",
        slot_ms.len(),
        manifest.slots.len(),
    );

    let mut node_times = vec![0.0; manifest.nodes.len()];
    for index in (0..manifest.nodes.len()).rev() {
        node_times[index] = match &manifest.nodes[index] {
            FlatCostNode::Leaf(slot) => {
                let value = *slot_ms.get(*slot).with_context(|| {
                    format!("cost-tree leaf node {index} references missing slot {slot}")
                })?;
                ensure!(
                    value.is_finite() && value >= 0.0,
                    "slot {slot} has invalid time {value}"
                );
                value
            }
            FlatCostNode::Sum { children } => {
                validate_child_range(manifest, index, children, "Sum")?;
                children.clone().map(|child| node_times[child]).sum()
            }
            FlatCostNode::Max { overlap, children } => {
                validate_child_range(manifest, index, children, "Max")?;
                let critical_child = first_max_child(children.clone(), &node_times);
                node_times[critical_child] / f64::from(*overlap).max(TIME_EPSILON_MS)
            }
            FlatCostNode::Scale { n, children } => {
                validate_child_range(manifest, index, children, "Scale")?;
                ensure!(
                    children.len() == 1,
                    "Scale node {index} must own exactly one child"
                );
                f64::from(*n) * node_times[children.start]
            }
        };
    }

    let mut node_weights = vec![0.0; manifest.nodes.len()];
    let mut leaf_ms = vec![0.0; slot_ms.len()];
    node_weights[0] = 1.0;
    for index in 0..manifest.nodes.len() {
        let weight = node_weights[index];
        if weight == 0.0 {
            continue;
        }
        match &manifest.nodes[index] {
            FlatCostNode::Leaf(slot) => leaf_ms[*slot] += slot_ms[*slot] * weight,
            FlatCostNode::Sum { children } => {
                for child in children.clone() {
                    node_weights[child] += weight;
                }
            }
            FlatCostNode::Scale { n, children } => {
                node_weights[children.start] += weight * f64::from(*n);
            }
            FlatCostNode::Max { overlap, children } => {
                let critical_child = first_max_child(children.clone(), &node_times);
                node_weights[critical_child] += weight / f64::from(*overlap).max(TIME_EPSILON_MS);
            }
        }
    }

    let attributed_ms: f64 = leaf_ms.iter().sum();
    ensure!(
        (attributed_ms - node_times[0]).abs() <= (node_times[0].abs() * 1e-9).max(1e-9),
        "critical-path leaf attribution {attributed_ms:.9} ms != root {:.9} ms",
        node_times[0],
    );
    Ok(leaf_ms)
}

fn validate_child_range(
    manifest: &Manifest,
    index: usize,
    children: &std::ops::Range<usize>,
    kind: &str,
) -> Result<()> {
    ensure!(
        !children.is_empty() && children.start > index && children.end <= manifest.nodes.len(),
        "{kind} node {index} has invalid children {children:?}"
    );
    Ok(())
}

fn first_max_child(children: std::ops::Range<usize>, node_times: &[f64]) -> usize {
    let mut critical_child = children.start;
    for child in children.start + 1..children.end {
        if node_times[child] > node_times[critical_child] {
            critical_child = child;
        }
    }
    critical_child
}

impl CompiledInventory {
    /// Which operation owns a simulated slot **this iteration**.
    ///
    /// A slot may be declared by several operations — a fused boundary owns it
    /// together with its norm, an unfused boundary's split collective owns it
    /// alone — so ownership is only decided once the iteration's measured
    /// operations are known. Two present claimants would make the attribution
    /// ambiguous, and that is an error, not a tie to break.
    fn resolve_slot_operation(
        &self,
        slot_name: &str,
        present_operations: &BTreeSet<&str>,
        iteration: u64,
    ) -> Result<Option<&OperationRule>> {
        let Some(declaring) = self.simulated_slots.get(slot_name) else {
            return Ok(None);
        };
        let present: Vec<&String> = declaring
            .iter()
            .filter(|op| present_operations.contains(op.as_str()))
            .collect();
        ensure!(
            present.len() <= 1,
            "iteration {iteration} simulated slot {slot_name:?} is claimed by multiple present \
             operations {present:?}"
        );
        Ok(present.first().and_then(|op| self.operations.get(*op)))
    }

    /// The folded sequence a phase ran for one iteration — the identity that makes
    /// two iterations comparable (same kernel program, different measurements).
    ///
    /// Schema 2/3 records one device-agnostic decision per iteration. Schema 4
    /// records one per device, and data-parallel ranks may diverge within a
    /// step; this returns the LOWEST device's sequence so the identity stays
    /// deterministic. Two iterations that agree on rank 0 but differ on a higher
    /// rank therefore share an identity — acceptable for the timeline's
    /// comparability grouping, but not a claim that every rank matched.
    fn sequence_id(&self, phase: &str, iteration: u64) -> Option<&str> {
        let phase_inventory = self.phases.get(phase)?;
        let index = phase_inventory
            .sequence_by_position
            .get(&(None, iteration))
            .or_else(|| {
                phase_inventory
                    .sequence_by_position
                    .iter()
                    .find(|((_, sequence_iteration), _)| *sequence_iteration == iteration)
                    .map(|(_, index)| index)
            })?;
        Some(phase_inventory.sequences[*index].sequence_id.as_str())
    }

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
    compile_inventory(read_json(path)?)
}

fn compile_inventory(doc: FoldedSequenceDoc) -> Result<CompiledInventory> {
    ensure!(
        matches!(doc.schema_version, 2 | 3 | 4),
        "labeled kernel sequences schema_version must be 2, 3 or 4"
    );
    ensure!(
        doc.encoding == "folded-v1",
        "unsupported kernel sequence encoding"
    );
    ensure!(
        !doc.phases.is_empty(),
        "labeled kernel sequences has no phases"
    );

    let device_ids = doc
        .device_ids
        .as_ref()
        .map(|ids| ids.iter().copied().collect::<BTreeSet<_>>());
    if doc.schema_version == 3 {
        let ids = device_ids
            .as_ref()
            .context("schema-v3 labeled inventory requires device_ids")?;
        ensure!(!ids.is_empty(), "schema-v3 device_ids cannot be empty");
        ensure!(
            doc.representative_device_id == ids.first().copied(),
            "schema-v3 representative_device_id must be the smallest device id"
        );
    } else if doc.schema_version == 4 {
        let ids = device_ids
            .as_ref()
            .context("schema-v4 labeled inventory requires device_ids")?;
        ensure!(!ids.is_empty(), "schema-v4 device_ids cannot be empty");
        // Schema 4 is a union catalog: no device is representative, because
        // ranks may run different sequences in the same iteration.
        ensure!(
            doc.representative_device_id.is_none(),
            "schema-v4 labeled inventory cannot declare a representative device"
        );
    } else {
        ensure!(
            device_ids.is_none() && doc.representative_device_id.is_none(),
            "schema-v2 labeled inventory cannot declare device metadata"
        );
    }

    let mut phases = BTreeMap::new();
    let mut operations: BTreeMap<String, OperationRule> = BTreeMap::new();
    let mut simulated_slots = BTreeMap::new();
    for (phase_name, phase) in doc.phases {
        ensure!(
            !phase.unique_sequences.is_empty(),
            "phase {phase_name:?} has no sequences"
        );
        let mut sequences = Vec::new();
        let mut sequence_by_position = BTreeMap::new();
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
                let synchronizing = label_is_synchronizing(&kernel.label)?;
                let operation =
                    compile_label(&kernel.label, &mut operations, &mut simulated_slots)?;
                rows.push(ExpandedRow {
                    row_id: format!("{}:{}", sequence.sequence_id, ordinal + 1),
                    name: kernel.name,
                    suggested_category: kernel.suggested_category,
                    operation,
                    synchronizing,
                });
            }
            let sequence_index = sequences.len();
            let positions: Vec<(Option<i64>, u64)> = if doc.schema_version == 4 {
                ensure!(
                    sequence.iterations.is_none(),
                    "schema-v4 sequence {:?} must carry occurrences, not iterations",
                    sequence.sequence_id
                );
                let occurrences = sequence.occurrences.as_ref().with_context(|| {
                    format!(
                        "schema-v4 sequence {:?} has no occurrences",
                        sequence.sequence_id
                    )
                })?;
                ensure!(
                    !occurrences.is_empty(),
                    "schema-v4 sequence {:?} has an empty occurrence list",
                    sequence.sequence_id
                );
                occurrences
                    .iter()
                    .flat_map(|occurrence| {
                        occurrence
                            .iterations
                            .iter()
                            .map(|iteration| (Some(occurrence.device_id), *iteration))
                    })
                    .collect()
            } else {
                ensure!(
                    sequence.occurrences.is_none(),
                    "schema-v{} sequence {:?} cannot carry occurrences",
                    doc.schema_version,
                    sequence.sequence_id
                );
                let iterations = sequence.iterations.as_ref().with_context(|| {
                    format!("sequence {:?} has no iterations", sequence.sequence_id)
                })?;
                iterations
                    .iter()
                    .map(|iteration| (None, *iteration))
                    .collect()
            };
            for position in positions {
                ensure!(
                    sequence_by_position
                        .insert(position, sequence_index)
                        .is_none(),
                    "phase {phase_name:?} assigns iteration {} twice{}",
                    position.1,
                    position
                        .0
                        .map(|device| format!(" on device {device}"))
                        .unwrap_or_default()
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
                        sequence_by_position,
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
        device_ids,
        representative_device_id: doc.representative_device_id,
        is_union_catalog: doc.schema_version == 4,
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

/// Read the mapping table's per-kernel cross-rank class. Absent defaults to
/// independent so schema-v2 single-rank captures keep working unchanged.
fn label_is_synchronizing(label: &EmbeddedLabel) -> Result<bool> {
    match label.cross_rank.as_deref() {
        None | Some("independent") => Ok(false),
        Some("synchronizing") => Ok(true),
        Some(other) => {
            anyhow::bail!("label cross_rank must be synchronizing or independent, got {other:?}")
        }
    }
}

fn compile_label(
    label: &EmbeddedLabel,
    operations: &mut BTreeMap<String, OperationRule>,
    simulated_slots: &mut BTreeMap<String, BTreeSet<String>>,
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
        // A slot may be declared by several operations (fused aggregate vs
        // unfused split); per-iteration resolution picks the one present.
        simulated_slots
            .entry(simulated_slot.clone())
            .or_default()
            .insert(operation.clone());
    }
    Ok(Some(operation.clone()))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

/// Pooled duty-cycle correction: Σ measured GPU cycle / Σ measured_ms over the
/// iterations that have a measured cycle. This is what scales kernel-only
/// timing-predict totals up to wall-clock; by construction, applying it makes the
/// pooled GPU-cycle gap equal the pooled kernel gap, so the kernel-align and
/// duty-cycle alignment levels agree. Derived from the measured side alone, it is
/// independent of any simulation run and supersedes the old
/// `gpu_cycle / global_kernel_busy` factor (whose denominator kept collective
/// arrival-wait that `measured_ms` deliberately drops).
///
/// With no measured cycle at all (e.g. a single-iteration capture) it degrades to
/// the identity 1.0.
fn pooled_gpu_time_multiplier(sum_measured_gpu_cycle_ms: f64, sum_measured_ms: f64) -> f64 {
    if sum_measured_ms > 0.0 {
        sum_measured_gpu_cycle_ms / sum_measured_ms
    } else {
        1.0
    }
}

/// Build first-kernel(i) -> first-kernel(i+1) GPU cycles in execution order.
/// The final valid iteration intentionally has no entry because its next GPU
/// boundary is unknown; the overview renderer excludes that row.
fn measured_gpu_cycles_ms(iterations: &[MeasuredIteration]) -> Result<BTreeMap<u64, f64>> {
    let mut starts = Vec::new();
    let mut seen_iterations = BTreeSet::new();
    for iteration in iterations {
        ensure!(
            seen_iterations.insert(iteration.iteration),
            "duplicate measured iteration {}",
            iteration.iteration
        );
        let first_start_ns = iteration
            .ranges
            .iter()
            .flat_map(|range| range.kernels.iter().map(|kernel| kernel.start_ns))
            .min();
        if let Some(first_start_ns) = first_start_ns {
            starts.push((first_start_ns, iteration.iteration));
        }
    }
    starts.sort_unstable();

    let mut cycles = BTreeMap::new();
    for pair in starts.windows(2) {
        let (start_ns, iteration_id) = pair[0];
        let (next_start_ns, next_iteration_id) = pair[1];
        ensure!(
            next_start_ns > start_ns,
            "measured iterations {iteration_id} and {next_iteration_id} have non-increasing first-kernel timestamps"
        );
        cycles.insert(iteration_id, (next_start_ns - start_ns) as f64 / 1e6);
    }
    Ok(cycles)
}

/// One measured occurrence's critical-path contribution, reducing its per-rank
/// intervals for a single replica-worker timeline.
///
/// `synchronizing` collectives (all-reduce / all-gather / all-to-all, or a fused
/// all-reduce+norm) are barriers: every rank's kernel ends together, so each
/// rank's measured duration is inflated by how long it waited for the slowest
/// rank to arrive. `max(end) - max(start)` measures from "last rank arrived"
/// (barrier satisfied) to "collective truly done", dropping the arrival wait
/// while keeping a self-imbalanced collective's bottleneck rank. For a symmetric
/// all-reduce this equals `min(duration)`; for an imbalanced all-to-all it does
/// not, which is the point.
///
/// Independent ops have no cross-rank sync, so the occurrence costs the slowest
/// rank's own duration — real compute imbalance stays real work on the path.
fn occurrence_ns(launches: &[KernelLaunch], synchronizing: bool) -> u64 {
    if synchronizing {
        let max_start = launches.iter().map(|l| l.start_ns).max().unwrap_or(0);
        let max_end = launches.iter().map(|l| l.end_ns).max().unwrap_or(0);
        max_end.saturating_sub(max_start)
    } else {
        launches
            .iter()
            .map(|l| l.end_ns.saturating_sub(l.start_ns))
            .max()
            .unwrap_or(0)
    }
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
        "measured_ms": "replica critical-path sum: every measured occurrence reduced across ranks (independent op = slowest rank duration; synchronizing collective = max(end) - max(start), i.e. last-arrival to done) then summed. This is the baseline compared to the sim cost tree",
        "measured_busy_union_ms": "audit only: union of CUDA kernel intervals across every NSYS phase and TP rank; it removes cross-op rank-skew overlap that the critical-path sum keeps, so it reads below measured_ms",
        "total_simulated_ms": "timing-predict cost-tree total_time_ms (Sum/Max/Scale semantics preserved)",
        "relative_diff_pct": "(simulated - measured) / measured * 100; positive means overprediction",
        "measured_gpu_cycle_ms": "CUPTI first-kernel start of the next valid measured iteration minus first-kernel start of this iteration; the final valid iteration has no cycle. Collective arrival-wait and launch gaps live here, not in measured_ms; the recommended_gpu_time_multiplier is the correction that spans them",
        "recommended_gpu_time_multiplier": "pooled duty-cycle correction = Σ measured_gpu_cycle_ms / Σ measured_ms over iterations that have a measured cycle. Applying it as simulated_gpu_cycle = simulated_ms × this makes the pooled GPU-cycle gap equal the pooled kernel gap by construction. Derived from the measured side only (no simulation input); the aligned simulation worker should bake this value into its clock",
        "simulated_gpu_cycle_ms": "timing-predict total_time_ms multiplied by recommended_gpu_time_multiplier (the measured pooled duty-cycle correction), i.e. the kernel-only prediction scaled up to wall-clock",
        "gpu_cycle_relative_diff_pct": "(scaled timing-predict GPU cycle - measured GPU cycle) / measured GPU cycle * 100; positive means overprediction",
        "operation_measured_ms": "each occurrence is first reduced across the ranks that raised it (independent = slowest rank duration; synchronizing = max(end) - max(start)); arrival wait is dropped, not attributed to the collective. Those reduced durations are then summed along each rank's own timeline and the slowest rank is taken, so ranks that ran different kernel sequences (data parallelism) combine concurrently rather than serially. When every rank runs one sequence this is exactly the flat sum over occurrences",
        "operation_simulated_ms": "mapped sim leaf contribution after exact CostTree Sum/Scale/Max/overlap attribution; operation contributions plus unmapped critical-path leaves add to total_simulated_ms",
        "measured_kernel_duration_ms": "one occurrence's cross-rank critical-path contribution per the cross_rank class: independent = max over ranks of (end-start); synchronizing collective = max(end) - max(start). rank_launches counts raw launches; replica_calls divides symmetric launches by captured device count",
        "cross_rank": "the mapping table's per-kernel reduction class: synchronizing (a collective barrier) or independent; the analyzer applies min/max from this, never from a category or name",
        "simulated_kernel_folded_ms": "one L1 leaf slot time multiplied by its exact CostTree Scale multiplicity",
        "simulated_kernel_critical_path_ms": "the leaf's contribution to timing-predict total_time_ms after exact CostTree Sum/Scale/Max/overlap attribution; an exact Max tie selects the first child",
        "mapping_coverage": "duration/workload fraction assigned by embedded labels; unmatched entries stay explicit and are never filled with zero",
        "sequences": "the labelled kernel programs as the labeler wrote them, still folded: a `repeat{n}` band is one layer repeated, not n rows. Joined to a breakdown by row_id, which is `sequence_id:expanded_ordinal`",
        "breakdown_detail.byte_ranges": "iteration_id -> [byte offset, byte length] into the sibling .jsonl holding that iteration's measured and simulated kernel rows. Read that range and parse it as one JSON object; the whole file is never needed at once",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::manifest::LeafDesc;

    fn test_leaf(name: &str) -> LeafDesc {
        LeafDesc {
            name: name.into(),
            kind: "test".into(),
            kernel_config: json!({"backends": []}),
        }
    }

    #[test]
    fn critical_path_leaf_ms_preserves_sum_scale_and_max() {
        let manifest = Manifest {
            slots: vec![test_leaf("a"), test_leaf("b"), test_leaf("c")],
            // Sum(Leaf a, Scale{2}(Max(Leaf b, Leaf c))).
            nodes: vec![
                FlatCostNode::Sum { children: 1..3 },
                FlatCostNode::Leaf(0),
                FlatCostNode::Scale {
                    n: 2,
                    children: 3..4,
                },
                FlatCostNode::Max {
                    overlap: 2.0,
                    children: 4..6,
                },
                FlatCostNode::Leaf(1),
                FlatCostNode::Leaf(2),
            ],
            node_labels: vec![None; 6],
        };

        // Root = 4 + 2 * (max(6, 10) / 2) = 14. Only c is critical.
        assert_eq!(
            critical_path_leaf_ms(&manifest, &[4.0, 6.0, 10.0]).unwrap(),
            vec![4.0, 0.0, 10.0]
        );
    }

    #[test]
    fn critical_path_leaf_ms_breaks_exact_max_tie_to_first_child() {
        let manifest = Manifest {
            slots: vec![test_leaf("rank0"), test_leaf("rank1")],
            nodes: vec![
                FlatCostNode::Max {
                    overlap: 1.0,
                    children: 1..3,
                },
                FlatCostNode::Leaf(0),
                FlatCostNode::Leaf(1),
            ],
            node_labels: vec![None; 3],
        };

        assert_eq!(
            critical_path_leaf_ms(&manifest, &[5.0, 5.0]).unwrap(),
            vec![5.0, 0.0]
        );
    }

    #[test]
    fn interval_union_merges_overlap_across_phases() {
        assert_eq!(interval_union_ns(&[(10, 20), (15, 30), (40, 45)]), 25);
    }

    /// Two ranks of one kernel position, given as `(device, start, end)`.
    fn launches(rows: &[(i64, u64, u64)]) -> Vec<KernelLaunch> {
        rows.iter()
            .map(|(device_id, start_ns, end_ns)| KernelLaunch {
                device_id: *device_id,
                start_ns: *start_ns,
                end_ns: *end_ns,
                correlation_id: None,
            })
            .collect()
    }

    #[test]
    fn occurrence_independent_takes_slowest_rank_duration() {
        // Two ranks, unbalanced compute: the critical path is the slower one.
        assert_eq!(occurrence_ns(&launches(&[(0, 0, 3), (1, 0, 5)]), false), 5);
    }

    #[test]
    fn occurrence_synchronizing_drops_arrival_wait() {
        // rank0 launches at 0 and waits inside the barrier to 7; rank1 arrives
        // at 3 and the collective completes at 7. The barrier-to-done cost is
        // 7 - 3 = 4, not rank0's inflated 7 and not a summed union.
        assert_eq!(occurrence_ns(&launches(&[(0, 0, 7), (1, 3, 7)]), true), 4);
    }

    #[test]
    fn occurrence_synchronizing_keeps_imbalanced_bottleneck() {
        // Self-imbalanced all-to-all: rank1 arrives last (start 3) but rank0
        // moves the most data and ends last (9). max(end)-max(start)=9-3=6
        // keeps the bottleneck while still removing the arrival skew; a plain
        // min(duration)=min(9,6)=6 would only coincide here, and undercount if
        // the bottleneck rank were also the last to arrive.
        assert_eq!(occurrence_ns(&launches(&[(0, 0, 9), (1, 3, 9)]), true), 6);
    }

    #[test]
    fn occurrence_single_rank_is_identity() {
        assert_eq!(occurrence_ns(&launches(&[(0, 10, 25)]), false), 15);
        assert_eq!(occurrence_ns(&launches(&[(0, 10, 25)]), true), 15);
    }

    #[test]
    fn pooled_multiplier_is_cycle_over_measured() {
        // Wall-clock cycle exceeds critical-path work by the duty-cycle factor.
        assert_eq!(pooled_gpu_time_multiplier(1200.0, 1000.0), 1.2);
        // No measured cycle (single-iteration capture) → identity, not a NaN.
        assert_eq!(pooled_gpu_time_multiplier(0.0, 0.0), 1.0);
    }

    #[test]
    fn gpu_cycles_use_chronological_first_kernel_boundaries() {
        let measured_iteration = |iteration, start_ns| MeasuredIteration {
            iteration,
            iteration_type: "decode".into(),
            ranges: vec![MeasuredRange {
                device_id: Some(0),
                phase: "forward".into(),
                start_ns,
                end_ns: start_ns + 100,
                kernels: vec![MeasuredKernel {
                    name_id: 1,
                    category: "gemm".into(),
                    start_ns,
                    end_ns: start_ns + 100,
                    correlation_id: None,
                }],
            }],
        };
        let iterations = vec![
            measured_iteration(9, 20_000_000),
            measured_iteration(7, 5_000_000),
            measured_iteration(8, 12_000_000),
        ];

        let cycles = measured_gpu_cycles_ms(&iterations).unwrap();

        assert_eq!(cycles[&7], 7.0);
        assert_eq!(cycles[&8], 8.0);
        assert!(!cycles.contains_key(&9));
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
                cross_rank: None,
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
            cross_rank: None,
        };
        let mut operations = BTreeMap::new();
        let mut slots = BTreeMap::new();

        let operation = compile_label(&label, &mut operations, &mut slots).unwrap();

        assert_eq!(operation.as_deref(), Some("attention"));
        assert_eq!(
            operations["attention"].simulated_slots,
            ["attention.main", "attention.combine"]
        );
        assert_eq!(
            slots["attention.main"],
            BTreeSet::from(["attention".to_string()])
        );
        assert_eq!(
            slots["attention.combine"],
            BTreeSet::from(["attention".to_string()])
        );
    }
    /// One-kernel `forward` sequence carrying an unmapped label, which is
    /// enough to exercise the inventory's device/iteration bookkeeping.
    fn folded_sequence(sequence_id: &str, assignment: serde_json::Value) -> serde_json::Value {
        let mut sequence = json!({
            "sequence_id": sequence_id,
            "expanded_kernel_count": 1,
            "program": [{"kernels": [{
                "name": format!("{sequence_id}_kernel"),
                "suggested_category": "compute",
                "label": {"status": "unmapped"},
            }]}],
        });
        let object = sequence.as_object_mut().unwrap();
        for (key, value) in assignment.as_object().unwrap() {
            object.insert(key.clone(), value.clone());
        }
        sequence
    }

    fn inventory_doc(
        schema_version: u32,
        device_metadata: serde_json::Value,
        sequences: Vec<serde_json::Value>,
    ) -> FoldedSequenceDoc {
        let mut doc = json!({
            "schema_version": schema_version,
            "encoding": "folded-v1",
            "phases": {"forward": {"unique_sequences": sequences}},
        });
        let object = doc.as_object_mut().unwrap();
        for (key, value) in device_metadata.as_object().unwrap() {
            object.insert(key.clone(), value.clone());
        }
        serde_json::from_value(doc).unwrap()
    }

    #[test]
    fn schema_four_keeps_each_device_on_its_own_sequence() {
        // Data parallelism: device 0 and device 1 run different kernel
        // sequences in the same iteration. Both labeling decisions survive.
        let inventory = compile_inventory(inventory_doc(
            4,
            json!({"device_ids": [0, 1]}),
            vec![
                folded_sequence(
                    "sequence_aaa",
                    json!({"occurrences": [{"device_id": 0, "iterations": [7]}]}),
                ),
                folded_sequence(
                    "sequence_bbb",
                    json!({"occurrences": [{"device_id": 1, "iterations": [7]}]}),
                ),
            ],
        ))
        .unwrap();

        let phase = &inventory.phases["forward"];
        assert_eq!(
            phase.sequences[phase.sequence_by_position[&(Some(0), 7)]].sequence_id,
            "sequence_aaa"
        );
        assert_eq!(
            phase.sequences[phase.sequence_by_position[&(Some(1), 7)]].sequence_id,
            "sequence_bbb"
        );
        assert_eq!(inventory.representative_device_id, None);
    }

    #[test]
    fn schema_three_labels_stay_device_agnostic() {
        // The pre-DP shape: one decision per iteration, shared by every rank.
        let inventory = compile_inventory(inventory_doc(
            3,
            json!({"device_ids": [0, 1], "representative_device_id": 0}),
            vec![folded_sequence("sequence_aaa", json!({"iterations": [7]}))],
        ))
        .unwrap();

        let phase = &inventory.phases["forward"];
        assert_eq!(phase.sequence_by_position.get(&(Some(0), 7)), None);
        assert_eq!(phase.sequence_by_position[&(None, 7)], 0);
    }

    #[test]
    fn schema_four_rejects_a_representative_device() {
        // A union catalog has no representative: ranks may disagree.
        let error = compile_inventory(inventory_doc(
            4,
            json!({"device_ids": [0, 1], "representative_device_id": 0}),
            vec![folded_sequence(
                "sequence_aaa",
                json!({"occurrences": [{"device_id": 0, "iterations": [7]}]}),
            )],
        ))
        .map(|_| ())
        .unwrap_err();
        assert!(
            error.to_string().contains("representative device"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn schema_four_rejects_device_agnostic_iterations() {
        let error = compile_inventory(inventory_doc(
            4,
            json!({"device_ids": [0]}),
            vec![folded_sequence("sequence_aaa", json!({"iterations": [7]}))],
        ))
        .map(|_| ())
        .unwrap_err();
        assert!(
            error.to_string().contains("must carry occurrences"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn schema_four_rejects_two_sequences_for_one_device_iteration() {
        let error = compile_inventory(inventory_doc(
            4,
            json!({"device_ids": [0]}),
            vec![
                folded_sequence(
                    "sequence_aaa",
                    json!({"occurrences": [{"device_id": 0, "iterations": [7]}]}),
                ),
                folded_sequence(
                    "sequence_bbb",
                    json!({"occurrences": [{"device_id": 0, "iterations": [7]}]}),
                ),
            ],
        ))
        .map(|_| ())
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("assigns iteration 7 twice on device 0"),
            "unexpected error: {error}"
        );
    }
}
