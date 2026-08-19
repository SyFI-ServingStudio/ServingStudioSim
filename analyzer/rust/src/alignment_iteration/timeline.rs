//! One measured iteration and its simulated cost tree, on one time axis.
//!
//! `alignment-iteration` answers *how far off is the cost model* as numbers. It
//! keeps exactly one timestamp per kernel (`first_start_ns`) and only as a sort
//! key, so nothing downstream can reconstruct where the GPU actually was. This
//! subject keeps the timestamps.
//!
//! What it emits is deliberately raw:
//!
//! - **measured** — every rank's real `(start, end)` for every kernel position,
//!   unreduced. Bubbles are the *client's* complement of that interval union;
//!   deriving them here would bake in one window choice, and deriving them from
//!   drawn geometry would turn every sub-pixel kernel into a gap that is not
//!   there.
//! - **simulated** — the per-slot UNIT times plus the shared cost manifest. The
//!   renderer unfolds them the same way `trace::place` does, so the sim lane and
//!   a `.pftrace` of the same iteration agree by construction. Unit times, not
//!   critical-path-attributed ones: a losing `Max` branch is 0 there and would
//!   render as a zero-width leaf.
//!
//! **Nothing here pairs a measured kernel with a simulated leaf.** In this very
//! capture the ratio is 1:1, 2:1 and 3:1 depending on the operation, and some
//! rows fuse 1:2. The only sound join is the operation string, which is what
//! `operation_totals` compares — and what the renderer is expected to colour by.
//!
//! **The payload is an index, not a document.** Per-kernel intervals for all
//! 2040 iterations of a real capture run to hundreds of megabytes — past what a
//! browser can hold as one string, for a view that draws one iteration at a
//! time. So the payload carries what every iteration shares (cost manifest,
//! operations, host roster, string pool) plus one summary row each, and the
//! detail goes line by line into a sibling `.jsonl` with a byte-range index. A
//! reader seeks to the one iteration it is drawing.
//!
//! [`MAX_ITERATIONS`] therefore bounds only the *default* selection — which
//! iterations are worth naming when nobody asked for specific ones. A manifest
//! that names them is not capped.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{ensure, Context, Result};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};
use std::path::Path;

use super::host::{self, HostWindow};
use super::{
    interval_union_ns, kernel_name_index, leaf_scales, load_inventory, load_sim_cases,
    measure_iteration, measured_gpu_cycles_ms, occurrence_ns, read_json, single_iter_manifest,
    CaseMapDoc, CompiledInventory, IterationMeasurement, MeasuredIteration, ParsedTrace, SimCase,
};
use crate::alignment_input;
use crate::io::{read_cost_manifests, resolve_artifact_path, SCHEMA_VERSION};

/// How many representatives the default selection names. Not a payload-size
/// bound — the detail is sharded — but a curation one: a picker listing two
/// thousand indistinguishable decode iterations is not a picker.
const MAX_ITERATIONS: usize = 32;

/// Sibling of the payload holding one iteration's full detail per line.
const ITERATION_DETAIL_FILE: &str = "alignment_timeline_iterations.jsonl";

/// The phase whose folded kernel program identifies an iteration's shape. The
/// other phases (preprocess / sample / bookkeep …) are the same few kernels
/// regardless of what the model iteration did.
const IDENTITY_PHASE: &str = "forward";

/// How many of the reference rank's largest GPU gaps the report names. Enough to
/// see whether idle time is one stall or a thousand launch gaps.
const TOP_GAPS: usize = 8;

const SELECTION_RULE: &str = "every iteration in the capture, in iteration order. At most 32 of \
     them carry a distinguishing `selected_as`: one representative per \
     (iteration_type, forward sequence_id) at that group's median \
     |relative_diff_pct| with groups taken largest first, plus the capture's \
     global argmax and argmin of |relative_diff_pct|. The rest are `full_capture`. \
     Naming `timeline_iterations` in the alignment manifest restricts the file to \
     those iterations instead.";

const ANCHOR_RULE: &str = "min kernel start over the reference rank. Deliberately the same \
     expression `measured_gpu_cycle_ms` uses to bound a GPU iteration, so the sim \
     lane's left edge and the duty-cycle denominator agree by construction. It is \
     an assumption about where a modelled iteration begins, not a measurement.";

/// One iteration reduced to the scalars the selection rule needs. Cheap enough to
/// hold for every case in the capture.
struct IterationSummary {
    case_index: u64,
    iteration_id: u64,
    stage: String,
    iteration_type: String,
    identity_sequence: String,
    measured_ms: f64,
    simulated_ms: f64,
    relative_diff_pct: f64,
}

impl IterationSummary {
    fn abs_relative(&self) -> f64 {
        self.relative_diff_pct.abs()
    }
}

/// Why an iteration made the cut — worth carrying, because "the worst iteration"
/// and "a typical iteration" are read very differently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Selection {
    GroupMedian,
    GlobalMaxError,
    GlobalMinError,
    ManifestOverride,
    /// Emitted because everything is, with nothing to distinguish it. The label
    /// exists so "typical of its kernel program" stays a claim the subject makes
    /// about specific iterations rather than one a reader infers from presence.
    FullCapture,
}

impl Selection {
    fn label(self) -> &'static str {
        match self {
            Selection::GroupMedian => "group_median",
            Selection::GlobalMaxError => "global_max_error",
            Selection::GlobalMinError => "global_min_error",
            Selection::ManifestOverride => "manifest_override",
            Selection::FullCapture => "full_capture",
        }
    }
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
    inventory.validate_slots(manifest)?;

    let measured_by_id: BTreeMap<u64, &super::MeasuredIteration> = measured
        .iteration_details
        .iter()
        .map(|item| (item.iteration, item))
        .collect();
    let kernel_names = kernel_name_index(&measured)?;
    let gpu_cycles_ms = measured_gpu_cycles_ms(&measured.iteration_details)?;

    // Every timestamp in the payload is an offset from here. Not cosmetic:
    // JSON numbers land in a JS `number`, and a capture that carries absolute
    // epoch nanoseconds (1.7e18) would silently lose the low bits. An offset
    // from the capture's own first kernel is bounded by the capture's duration.
    let time_origin_ns = measured
        .iteration_details
        .iter()
        .flat_map(|iteration| {
            iteration
                .ranges
                .iter()
                .flat_map(|range| range.kernels.iter().map(|kernel| kernel.start_ns))
        })
        .min()
        .context("capture has no GPU kernels")?;

    // Pass one: the scalars the selection needs, for every case. This repeats
    // `alignment-iteration`'s measured reduction rather than reading its report,
    // because a subject may not depend on another subject's output.
    let mut summaries = Vec::with_capacity(case_map.cases.len());
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
        let measurement = measure_iteration(measured_iter, &inventory, &kernel_names)?;
        let measured_ms = critical_path_ms(&measurement);
        summaries.push(IterationSummary {
            case_index: joined.case_index,
            iteration_id: measured_iter.iteration,
            stage: joined.stage.clone(),
            iteration_type: measured_iter.iteration_type.clone(),
            identity_sequence: inventory
                .sequence_id(IDENTITY_PHASE, measured_iter.iteration)
                .unwrap_or("")
                .to_string(),
            measured_ms,
            simulated_ms: sim.total_ms,
            relative_diff_pct: relative_pct(sim.total_ms - measured_ms, measured_ms),
        });
    }
    ensure!(!summaries.is_empty(), "alignment case map is empty");

    // The pooled duty-cycle correction, over EVERY iteration — not just the ones
    // drawn. Recomputed here with `alignment-iteration`'s accumulation order
    // rather than read from its report, because a subject may not depend on
    // another subject's output; the sim lane's `x duty` bar is meaningless if the
    // two disagree.
    let mut sum_measured_gpu_cycle_ms = 0.0;
    let mut sum_measured_ms_with_cycle = 0.0;
    for summary in &summaries {
        if let Some(cycle_ms) = gpu_cycles_ms.get(&summary.iteration_id) {
            sum_measured_gpu_cycle_ms += cycle_ms;
            sum_measured_ms_with_cycle += summary.measured_ms;
        }
    }
    let gpu_time_multiplier = if sum_measured_ms_with_cycle > 0.0 {
        sum_measured_gpu_cycle_ms / sum_measured_ms_with_cycle
    } else {
        1.0
    };

    let selected = select_iterations(&summaries, input.timeline_iterations.as_deref())?;
    eprintln!(
        "[analyze] alignment-timeline: emitting {} of {} iterations ({})",
        selected.len(),
        summaries.len(),
        if input.timeline_iterations.is_some() {
            "manifest override"
        } else {
            "representatives"
        }
    );

    let reference_device_id = inventory.representative_device_id;
    // Under a union catalog there is no representative device — eight DP ranks
    // each run their own sequence — so the reference lane falls back to the
    // lowest measured device, which is the rule `reference_span` and
    // `build_iteration` already apply per iteration. The meta reports the
    // device the lanes are actually drawn from; reporting null instead leaves a
    // consumer unable to label the lane it is being shown.
    let reported_reference_device_id = reference_device_id.or_else(|| {
        inventory
            .device_ids
            .as_ref()
            .and_then(|device_ids| device_ids.iter().copied().min())
    });

    // The host lane attributes a million API rows to every window they overlap,
    // which is one pass over the events against all the windows at once — so
    // every window has to be known before the first iteration is built. They are
    // resolved straight from the parsed ranges; `build_iteration` asserts the
    // anchor it derives independently from the reduced measurement agrees, so
    // the two lanes can never be drawn from different origins.
    let mut host_windows: Vec<HostWindow> = Vec::with_capacity(selected.len());
    for (index, _) in &selected {
        let iteration_id = summaries[*index].iteration_id;
        let measured_iter = measured_by_id[&iteration_id];
        let Some(span) = reference_span(measured_iter, reference_device_id) else {
            continue;
        };
        host_windows.push(host::window(
            iteration_id,
            span.0,
            span,
            // Every phase marker, including the ones that launched no kernel at
            // all. `bookkeep` and `eplb` are pure host time, which is exactly
            // the time this lane exists to name.
            measured_iter
                .ranges
                .iter()
                .map(|range| (range.start_ns, range.end_ns)),
        ));
    }
    let host_timeline = host::load_if_configured(input.host_timeline.as_deref(), &host_windows)?;
    let host_anchors: BTreeMap<u64, u64> = host_windows
        .iter()
        .map(|window| (window.iteration_id, window.anchor_ns))
        .collect();

    // Pass two: the full detail. Each iteration's detail is serialized straight
    // into the shard buffer rather than collected, so peak memory is one
    // iteration and not the whole capture.
    let mut index_iterations = Vec::with_capacity(selected.len());
    let mut report_iterations = Vec::with_capacity(selected.len());
    let mut detail_bytes: Vec<u8> = Vec::new();
    let mut byte_ranges = serde_json::Map::new();
    let mut used_name_ids = BTreeSet::new();
    for (index, reason) in &selected {
        let summary = &summaries[*index];
        let measured_iter = measured_by_id[&summary.iteration_id];
        let sim = &sim_cases[&summary.case_index];
        let measurement = measure_iteration(measured_iter, &inventory, &kernel_names)?;
        let reference = reference_device_id
            .or_else(|| measurement.device_ids.first().copied())
            .context("measured iteration has no device ids")?;

        let built = build_iteration(
            summary,
            *reason,
            &measurement,
            sim,
            &inventory,
            manifest,
            &scales,
            reference,
            time_origin_ns,
            gpu_cycles_ms.get(&summary.iteration_id).copied(),
            gpu_time_multiplier,
            host_anchors
                .get(&summary.iteration_id)
                .map(|anchor_ns| HostContext {
                    anchor_ns: *anchor_ns,
                    rows: host_timeline
                        .as_ref()
                        .and_then(|timeline| timeline.iteration(summary.iteration_id)),
                }),
            &mut used_name_ids,
        )?;
        let line = serde_json::to_vec(&built.detail)?;
        byte_ranges.insert(
            summary.iteration_id.to_string(),
            json!([detail_bytes.len(), line.len()]),
        );
        detail_bytes.extend_from_slice(&line);
        detail_bytes.push(b'\n');
        index_iterations.push(built.index);
        // The report is the read-by-a-person half, and a person cannot read two
        // thousand iterations. It keeps the ones the selection distinguished;
        // the payload keeps everything, seekable.
        if *reason != Selection::FullCapture {
            report_iterations.push(built.report);
        }
    }

    let operations: Vec<_> = inventory
        .operations
        .values()
        .map(|operation| {
            json!({
                "operation": operation.operation,
                "type": operation.kernel_type,
                "role": operation.role,
                "simulated_slots": operation.simulated_slots,
            })
        })
        .collect();
    let kernel_name_dictionary: BTreeMap<String, &str> = used_name_ids
        .iter()
        .filter_map(|id| kernel_names.get(id).map(|name| (id.to_string(), *name)))
        .collect();

    // The cost manifest verbatim, so the renderer rebuilds the nested tree with
    // the same reader the gantt uses. One document for the whole file: the shape
    // is per-worker and static, only the per-iteration slot times vary.
    let manifest_path = resolve_artifact_path(&input.predict_log_dir, "cost_manifest")
        .join(format!("worker_{pool_tag}_{worker_id}.json"));
    let manifest_document: Value = read_json(&manifest_path)?;
    let sim_manifest = manifest_document
        .get("sections")
        .and_then(Value::as_array)
        .and_then(|sections| {
            sections
                .iter()
                .find(|section| section.get("section") == Some(&json!("iter")))
        })
        .cloned()
        .with_context(|| format!("no iter section in {}", manifest_path.display()))?;

    let meta = json!({
        "analysis_log_dir": log_dir.display().to_string(),
        "profile_log_dir": input.profile_log_dir.display().to_string(),
        "predict_log_dir": input.predict_log_dir.display().to_string(),
        "measured_device_ids": inventory.device_ids,
        "reference_device_id": reported_reference_device_id,
        // Kernel-only sim time scaled by this is wall-clock GPU time, which is
        // what the sim lane's `x duty` bar draws against the measured GPU cycle.
        "recommended_gpu_time_multiplier": gpu_time_multiplier,
        "iterations_available": summaries.len(),
        // Two different counts on purpose: the payload indexes every iteration,
        // the report writes up the distinguished ones.
        "iterations_emitted": index_iterations.len(),
        "iterations_reported": report_iterations.len(),
        "selection_rule": SELECTION_RULE,
        "anchor_rule": ANCHOR_RULE,
        "time_origin_ns": time_origin_ns,
        "time_base": "capture-relative nanoseconds: every *_ns in this file is \
                      (nsys timestamp - meta.time_origin_ns)",
        // The roster, rules and string pool of the host lane, or null when the
        // capture was parsed before the sidecar existed. What is still missing
        // even when this is present is time spent OUTSIDE any CUDA call: the
        // capture carries no OSRT / CPU sampling / context switches, so a thread
        // between two runtime calls is unaccounted for rather than idle.
        "host_timeline": host_timeline
            .as_ref()
            .map_or(Value::Null, |timeline| timeline.meta()),
    });
    let definitions = definitions();

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "available": true,
        "iterations": report_iterations,
        "definitions": definitions,
    });
    // The detail shard is written here rather than returned, because the
    // subject contract is one report and one payload. Doing it before the
    // payload is returned means an index can never name a shard that the caller
    // failed to write.
    let detail_path = crate::io::payload_path(log_dir, ITERATION_DETAIL_FILE);
    if let Some(parent) = detail_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&detail_path, &detail_bytes)
        .with_context(|| format!("write {}", detail_path.display()))?;
    println!("wrote {}", detail_path.display());

    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "operations": operations,
        "kernel_names": kernel_name_dictionary,
        "sim_manifest": sim_manifest,
        "slot_multiplicity": scales,
        "iterations": index_iterations,
        "iteration_detail": {
            "file": ITERATION_DETAIL_FILE,
            "encoding": "one JSON object per line, in the order of `iterations`",
            "byte_ranges": byte_ranges,
        },
        "definitions": definitions,
    });
    Ok((report, payload))
}

/// The headline measured duration: every occurrence reduced across its ranks,
/// then summed. Identical to `alignment-iteration`'s `measured_ms` — including
/// the float summation ORDER, which is why the mapped and unmapped halves are
/// accumulated separately and added at the end. Two subjects that disagree in the
/// last ulp about the same quantity are a bug report waiting to happen.
fn critical_path_ms(measurement: &IterationMeasurement) -> f64 {
    let mut mapped_ms = 0.0;
    let mut unmapped_ms = 0.0;
    for (_, item) in &measurement.kernels {
        #[allow(
            clippy::cast_precision_loss,
            reason = "occurrence_ns is one kernel's ns duration within one iteration, far below \
                      2^53 ns"
        )]
        let duration_ms = occurrence_ns(&item.launches, item.synchronizing) as f64 / 1e6;
        if item.operation.is_some() {
            mapped_ms += duration_ms;
        } else {
            unmapped_ms += duration_ms;
        }
    }
    mapped_ms + unmapped_ms
}

/// Label every iteration, distinguishing the ones worth opening first.
///
/// The detail is sharded, so emitting the whole capture costs a reader nothing
/// it does not ask for — and a picker that cannot reach the iteration someone
/// is asking about is useless. But a list of two thousand indistinguishable
/// decode iterations is not navigable either, so a bounded few are named: group
/// by what makes two iterations comparable — type plus forward sequence — take
/// each group's median error as its representative, and force in the capture's
/// best and worst so the extremes are never averaged away. Everything else is
/// emitted as `full_capture`.
fn select_iterations(
    summaries: &[IterationSummary],
    override_ids: Option<&[u64]>,
) -> Result<Vec<(usize, Selection)>> {
    if let Some(wanted) = override_ids {
        let by_iteration: BTreeMap<u64, usize> = summaries
            .iter()
            .enumerate()
            .map(|(index, summary)| (summary.iteration_id, index))
            .collect();
        return wanted
            .iter()
            .map(|iteration_id| {
                by_iteration
                    .get(iteration_id)
                    .map(|index| (*index, Selection::ManifestOverride))
                    .with_context(|| {
                        format!(
                            "timeline_iterations names iteration {iteration_id}, which the \
                                 alignment case map does not contain"
                        )
                    })
            })
            .collect();
    }

    // The extremes are the point of the view, so they are reserved before any
    // group gets a slot rather than displacing one at the cap.
    let extreme = |pick: fn(&f64, &f64) -> std::cmp::Ordering| {
        summaries
            .iter()
            .enumerate()
            .max_by(|left, right| {
                pick(&left.1.abs_relative(), &right.1.abs_relative())
                    .then_with(|| right.1.iteration_id.cmp(&left.1.iteration_id))
            })
            .map(|(index, _)| index)
    };
    let mut forced: Vec<(usize, Selection)> = Vec::new();
    for (index, reason) in [
        (extreme(f64::total_cmp), Selection::GlobalMaxError),
        (
            extreme(|left, right| right.total_cmp(left)),
            Selection::GlobalMinError,
        ),
    ] {
        let Some(index) = index else { continue };
        if !forced.iter().any(|(taken, _)| *taken == index) {
            forced.push((index, reason));
        }
    }

    let mut groups: BTreeMap<(&str, &str), Vec<usize>> = BTreeMap::new();
    for (index, summary) in summaries.iter().enumerate() {
        groups
            .entry((
                summary.iteration_type.as_str(),
                summary.identity_sequence.as_str(),
            ))
            .or_default()
            .push(index);
    }
    let mut ordered: Vec<_> = groups
        .into_iter()
        .map(|(key, mut members)| {
            members.sort_by(|left, right| {
                summaries[*left]
                    .abs_relative()
                    .total_cmp(&summaries[*right].abs_relative())
                    .then_with(|| {
                        summaries[*left]
                            .iteration_id
                            .cmp(&summaries[*right].iteration_id)
                    })
            });
            let representative = members[members.len() / 2];
            (key, members.len(), representative)
        })
        .collect();
    // Largest group first, ties broken by the group key so the choice is
    // reproducible across runs.
    ordered.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));

    // One group per iteration_type gets in ahead of the rest. Prefill groups are
    // rare in a steady-state capture — a purely size-ordered list drops every one
    // of them, and a payload with no prefill iteration cannot show the regime the
    // cost model most often gets wrong.
    let mut seen_types: BTreeSet<&str> = BTreeSet::new();
    let mut priority = Vec::new();
    let mut rest = Vec::new();
    for entry in ordered {
        if seen_types.insert(entry.0 .0) {
            priority.push(entry);
        } else {
            rest.push(entry);
        }
    }

    let budget = MAX_ITERATIONS.saturating_sub(forced.len());
    let mut labels: BTreeMap<usize, Selection> = forced.into_iter().collect();
    for (_, _, representative) in priority.into_iter().chain(rest) {
        if labels.len() >= MAX_ITERATIONS || budget == 0 {
            break;
        }
        labels
            .entry(representative)
            .or_insert(Selection::GroupMedian);
    }

    let mut chosen: Vec<(usize, Selection)> = (0..summaries.len())
        .map(|index| {
            (
                index,
                labels
                    .get(&index)
                    .copied()
                    .unwrap_or(Selection::FullCapture),
            )
        })
        .collect();
    chosen.sort_by_key(|(index, _)| summaries[*index].iteration_id);
    Ok(chosen)
}

/// What the host lane contributes to one iteration: its rows, and the anchor
/// they were drawn from so the device lane can assert the two agree.
struct HostContext<'a> {
    anchor_ns: u64,
    rows: Option<&'a host::IterationHost>,
}

/// One iteration, in the three shapes it is read in.
struct BuiltIteration {
    /// Everything, one line of the detail shard.
    detail: Value,
    /// The analyst-facing numbers, including the ranked gaps.
    report: Value,
    /// The scalars a picker needs before it fetches anything.
    index: Value,
}

/// The reference rank's kernel span, straight from the parsed ranges.
///
/// `ReferenceRank::of` derives the same span from the reduced measurement.
/// Having both is the point: the host window is resolved before any measurement
/// exists, and `build_iteration` refuses to draw the two lanes if they disagree.
fn reference_span(
    measured_iter: &MeasuredIteration,
    reference_device_id: Option<i64>,
) -> Option<(u64, u64)> {
    let device_id = reference_device_id.or_else(|| {
        measured_iter
            .ranges
            .iter()
            .filter(|range| !range.kernels.is_empty())
            .filter_map(|range| range.device_id)
            .min()
    })?;
    measured_iter
        .ranges
        .iter()
        .filter(|range| range.device_id == Some(device_id))
        .flat_map(|range| range.kernels.iter())
        .fold(None, |span: Option<(u64, u64)>, kernel| {
            Some(match span {
                None => (kernel.start_ns, kernel.end_ns),
                Some((start, end)) => (start.min(kernel.start_ns), end.max(kernel.end_ns)),
            })
        })
}

#[allow(clippy::too_many_arguments)]
fn build_iteration(
    summary: &IterationSummary,
    reason: Selection,
    measurement: &IterationMeasurement,
    sim: &SimCase,
    inventory: &CompiledInventory,
    manifest: &crate::trace::manifest::Manifest,
    scales: &[u64],
    reference_device_id: i64,
    time_origin_ns: u64,
    measured_gpu_cycle_ms: Option<f64>,
    gpu_time_multiplier: f64,
    host_context: Option<HostContext<'_>>,
    used_name_ids: &mut BTreeSet<u64>,
) -> Result<BuiltIteration> {
    let simulated_gpu_cycle_ms = sim.total_ms * gpu_time_multiplier;
    #[allow(
        clippy::cast_possible_wrap,
        reason = "ns is a trace timestamp; realistic trace/run spans are far below i64::MAX ns \
                  (~292 years), so this never wraps"
    )]
    let offset = |ns: u64| (ns as i64) - (time_origin_ns as i64);

    // ---- measured -----------------------------------------------------------
    let mut measured_operation_ms: BTreeMap<&str, f64> = BTreeMap::new();
    let mut measured_operation_rows: BTreeMap<&str, usize> = BTreeMap::new();
    let mut kernel_rows = Vec::with_capacity(measurement.kernels.len());
    for (_, item) in &measurement.kernels {
        let occurrence = occurrence_ns(&item.launches, item.synchronizing);
        if let Some(operation) = &item.operation {
            #[allow(
                clippy::cast_precision_loss,
                reason = "occurrence is one kernel's ns duration within one iteration, far below \
                          2^53 ns"
            )]
            let occurrence_ms = occurrence as f64 / 1e6;
            *measured_operation_ms.entry(operation.as_str()).or_default() += occurrence_ms;
            *measured_operation_rows
                .entry(operation.as_str())
                .or_default() += 1;
        }
        used_name_ids.insert(item.name_id);
        let mut launches: Vec<_> = item.launches.clone();
        launches.sort_by_key(|launch| (launch.device_id, launch.start_ns));
        kernel_rows.push(json!({
            "ph": item.phase,
            "row": item.row_id,
            "name_id": item.name_id,
            "cat": item.category,
            "op": item.operation,
            "sync": item.synchronizing,
            "occ_ns": occurrence,
            "iv": launches
                .iter()
                .map(|launch| {
                    json!([
                        launch.device_id,
                        offset(launch.start_ns),
                        offset(launch.end_ns),
                        launch.correlation_id,
                        launch.track_index,
                    ])
                })
                .collect::<Vec<_>>(),
        }));
    }

    // ---- simulated ----------------------------------------------------------
    // Slot ownership is resolved against the operations this iteration actually
    // measured, so a slot declared by both a fused and an unfused boundary lands
    // on whichever of them ran.
    let present_operations: BTreeSet<&str> = measured_operation_ms.keys().copied().collect();
    ensure!(
        sim.slot_ms.len() == manifest.slots.len(),
        "case {} slot_time_ms length {} != manifest slot count {}",
        summary.case_index,
        sim.slot_ms.len(),
        manifest.slots.len()
    );
    let mut slot_operations = Vec::with_capacity(manifest.slots.len());
    let mut simulated_operation_ms: BTreeMap<&str, f64> = BTreeMap::new();
    let mut simulated_operation_occurrences: BTreeMap<&str, u64> = BTreeMap::new();
    for (index, slot) in manifest.slots.iter().enumerate() {
        let operation = inventory.resolve_slot_operation(
            &slot.name,
            &present_operations,
            summary.iteration_id,
        )?;
        match operation {
            Some(rule) => {
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "scales[index] is a slot's replay multiplicity (small replication \
                              factor), far below 2^53"
                )]
                let folded_ms = sim.slot_ms[index] * scales[index] as f64;
                *simulated_operation_ms
                    .entry(rule.operation.as_str())
                    .or_default() += folded_ms;
                *simulated_operation_occurrences
                    .entry(rule.operation.as_str())
                    .or_default() += scales[index];
                slot_operations.push(json!(rule.operation));
            }
            None => slot_operations.push(Value::Null),
        }
    }

    let mut operation_totals = Vec::new();
    let every_operation: BTreeSet<&str> = measured_operation_ms
        .keys()
        .chain(simulated_operation_ms.keys())
        .copied()
        .collect();
    for operation in every_operation {
        let measured_ms = measured_operation_ms.get(operation).copied();
        let simulated_ms = simulated_operation_ms.get(operation).copied();
        let measured_occurrences = measured_operation_rows.get(operation).copied().unwrap_or(0);
        let simulated_occurrences = simulated_operation_occurrences
            .get(operation)
            .copied()
            .unwrap_or(0);
        #[allow(
            clippy::cast_precision_loss,
            reason = "measured/simulated occurrence counts are per-operation kernel counts within \
                      one iteration, far below 2^53"
        )]
        operation_totals.push(json!({
            "op": operation,
            "measured_ms": measured_ms,
            "simulated_ms": simulated_ms,
            "measured_occurrences": measured_occurrences,
            "simulated_occurrences": simulated_occurrences,
            // Occurrence ratio, NOT a per-kernel correspondence: 3 measured
            // kernels to 1 modelled leaf means the model prices the trio as one
            // leaf, not that any one of them maps to it.
            "occurrence_ratio": (simulated_occurrences > 0)
                .then(|| measured_occurrences as f64 / simulated_occurrences as f64),
        }));
    }

    // ---- reference-rank occupancy (the report's half) -----------------------
    let reference = ReferenceRank::of(measurement, reference_device_id);
    let anchor_ns = reference.span.map(|(start, _)| start);

    if let (Some(context), Some(anchor)) = (&host_context, anchor_ns) {
        ensure!(
            context.anchor_ns == anchor,
            "iteration {}: host lane anchored at {} but the device lane at {}",
            summary.iteration_id,
            context.anchor_ns,
            anchor
        );
    }

    let occupancy = reference.occupancy();
    let host_rows = host_context.and_then(|context| context.rows);
    let detail = json!({
        "iteration_id": summary.iteration_id,
        "case_index": summary.case_index,
        "stage": summary.stage,
        "iteration_type": summary.iteration_type,
        "identity_sequence": summary.identity_sequence,
        "selected_as": reason.label(),
        "anchor_ns": anchor_ns.map(offset),
        "gpu_span_ns": reference
            .span
            .map(|(start, end)| json!([offset(start), offset(end)])),
        "measured_gpu_cycle_ms": measured_gpu_cycle_ms,
        "simulated_gpu_cycle_ms": simulated_gpu_cycle_ms,
        "measured": {
            "critical_path_ms": summary.measured_ms,
            "busy_union_ms": measurement.busy_union_ms,
            "kernels": kernel_rows,
        },
        "simulated": {
            "total_ms": sim.total_ms,
            "slot_ms": sim.slot_ms,
            "slot_op": slot_operations,
        },
        "operation_totals": operation_totals,
        // Absent when the capture carries no host sidecar, which is a different
        // fact from "the CPU did nothing" and must not render as an empty lane.
        "host": host_rows.map(host::IterationHost::value),
    });

    let report = json!({
        "iteration_id": summary.iteration_id,
        "case_index": summary.case_index,
        "stage": summary.stage,
        "iteration_type": summary.iteration_type,
        "identity_sequence": summary.identity_sequence,
        "selected_as": reason.label(),
        "measured_ms": summary.measured_ms,
        "simulated_ms": summary.simulated_ms,
        "delta_ms": summary.simulated_ms - summary.measured_ms,
        "relative_diff_pct": summary.relative_diff_pct,
        "measured_gpu_cycle_ms": measured_gpu_cycle_ms,
        "simulated_gpu_cycle_ms": measured_gpu_cycle_ms.map(|_| simulated_gpu_cycle_ms),
        "gpu_cycle_delta_ms": measured_gpu_cycle_ms.map(|cycle| simulated_gpu_cycle_ms - cycle),
        "measured_busy_union_ms": measurement.busy_union_ms,
        "measured_rows": measurement.kernels.len(),
        "reference_rank": reference.report(
            &occupancy,
            reference_device_id,
            measurement,
            time_origin_ns,
        ),
    });

    // The picker's row. Everything here is a scalar the reader sorts or filters
    // on before choosing what to fetch; anything per-kernel stays in the shard.
    let index = json!({
        "iteration_id": summary.iteration_id,
        "case_index": summary.case_index,
        "stage": summary.stage,
        "iteration_type": summary.iteration_type,
        "identity_sequence": summary.identity_sequence,
        "selected_as": reason.label(),
        "anchor_ns": anchor_ns.map(offset),
        "measured_ms": summary.measured_ms,
        "simulated_ms": summary.simulated_ms,
        "relative_diff_pct": summary.relative_diff_pct,
        "measured_gpu_cycle_ms": measured_gpu_cycle_ms,
        "simulated_gpu_cycle_ms": simulated_gpu_cycle_ms,
        "span_ms": occupancy.span_ms,
        "busy_ms": occupancy.busy_ms,
        "idle_ms": occupancy.idle_ms,
        "idle_fraction": occupancy.idle_fraction(),
        "gap_count": reference.gaps.len(),
        "has_host_lane": host_rows.is_some(),
    });
    Ok(BuiltIteration {
        detail,
        report,
        index,
    })
}

/// The reference rank's own occupancy, which is what the GPU lane draws.
///
/// Span is `max(end) - min(start)` over that rank's kernels — the correlated
/// kernel span, never the NVTX range. Under CUDA graphs the NVTX `forward` range
/// can close before its own kernels finish, so an idle figure computed against it
/// clamps to zero and hides real stalls.
struct ReferenceRank {
    span: Option<(u64, u64)>,
    intervals: Vec<(u64, u64)>,
    gaps: Vec<Gap>,
}

/// The reference rank's span split into kernel time and bubble.
struct Occupancy {
    span_ms: f64,
    busy_ms: f64,
    idle_ms: f64,
}

impl Occupancy {
    fn idle_fraction(&self) -> Option<f64> {
        (self.span_ms > 0.0).then(|| self.idle_ms / self.span_ms)
    }
}

/// One stretch of the reference rank's timeline with no kernel on it, named by
/// what sits on either side. An unmapped kernel has no operation, so the kernel's
/// own identity is carried too — most launch gaps live between bookkeeping
/// kernels the mapping deliberately ignores, and a gap labelled `null → null`
/// says nothing.
struct Gap {
    start_ns: u64,
    end_ns: u64,
    after: GapEdge,
    before: GapEdge,
}

#[derive(Clone)]
struct GapEdge {
    phase: String,
    operation: Option<String>,
    kernel: String,
}

impl GapEdge {
    fn value(&self) -> Value {
        json!({
            "phase": self.phase,
            "operation": self.operation,
            "kernel": self.kernel,
        })
    }
}

impl ReferenceRank {
    fn of(measurement: &IterationMeasurement, device_id: i64) -> Self {
        // One entry per launch on this rank, in time order, carrying its identity
        // so a gap can name what it sits between.
        let mut launches: Vec<(u64, u64, GapEdge)> = Vec::new();
        for (_, item) in &measurement.kernels {
            for launch in &item.launches {
                if launch.device_id == device_id {
                    launches.push((
                        launch.start_ns,
                        launch.end_ns,
                        GapEdge {
                            phase: item.phase.clone(),
                            operation: item.operation.clone(),
                            kernel: item.name.clone(),
                        },
                    ));
                }
            }
        }
        launches.sort_by_key(|(start, end, _)| (*start, *end));
        let intervals: Vec<(u64, u64)> = launches
            .iter()
            .map(|(start, end, _)| (*start, *end))
            .collect();
        let span = match (
            intervals.first(),
            intervals.iter().map(|(_, end)| *end).max(),
        ) {
            (Some((start, _)), Some(end)) => Some((*start, end)),
            _ => None,
        };

        // Gaps walk the raw launches, not drawn geometry and not a merged union,
        // so each keeps the kernels on either side of it.
        let mut gaps = Vec::new();
        let mut frontier: Option<(u64, GapEdge)> = None;
        for (start, end, edge) in launches {
            match frontier {
                Some((cursor, ref after)) => {
                    if start > cursor {
                        gaps.push(Gap {
                            start_ns: cursor,
                            end_ns: start,
                            after: after.clone(),
                            before: edge.clone(),
                        });
                    }
                    if end > cursor {
                        frontier = Some((end, edge));
                    }
                }
                None => frontier = Some((end, edge)),
            }
        }
        Self {
            span,
            intervals,
            gaps,
        }
    }

    /// How much of the reference rank's span had a kernel on it.
    ///
    /// Computed once and handed to both readers: a picker ordered by idle share
    /// and the report opened next to it must not disagree in the last ulp about
    /// the same iteration.
    #[allow(
        clippy::cast_precision_loss,
        reason = "both are ns spans/unions within one iteration, far below 2^53 ns"
    )]
    fn occupancy(&self) -> Occupancy {
        let span_ms = self
            .span
            .map(|(start, end)| (end - start) as f64 / 1e6)
            .unwrap_or(0.0);
        let busy_ms = interval_union_ns(&self.intervals) as f64 / 1e6;
        Occupancy {
            span_ms,
            busy_ms,
            idle_ms: (span_ms - busy_ms).max(0.0),
        }
    }

    fn report(
        &self,
        occupancy: &Occupancy,
        device_id: i64,
        measurement: &IterationMeasurement,
        time_origin_ns: u64,
    ) -> Value {
        let Occupancy {
            span_ms,
            busy_ms,
            idle_ms,
        } = *occupancy;
        // Ranked within each phase, because the phases have wildly different
        // scales: the host time between `preprocess` and `forward` is hundreds of
        // microseconds and would crowd out every stall inside the forward pass,
        // which is the thing worth looking at.
        let largest_gaps = |phase: &str| {
            let mut ranked: Vec<_> = self
                .gaps
                .iter()
                .filter(|gap| gap.after.phase == phase && gap.before.phase == phase)
                .collect();
            ranked.sort_by(|left, right| {
                (right.end_ns - right.start_ns).cmp(&(left.end_ns - left.start_ns))
            });
            ranked
                .into_iter()
                .take(TOP_GAPS)
                .map(|gap| {
                    #[allow(
                        clippy::cast_possible_wrap,
                        reason = "gap.start_ns is a trace timestamp; realistic trace/run spans \
                                  are far below i64::MAX ns (~292 years), so this never wraps"
                    )]
                    let start_ns = (gap.start_ns as i64) - (time_origin_ns as i64);
                    #[allow(
                        clippy::cast_precision_loss,
                        reason = "gap duration is a ns span within one iteration, far below 2^53 ns"
                    )]
                    let duration_us = (gap.end_ns - gap.start_ns) as f64 / 1e3;
                    json!({
                        "start_ns": start_ns,
                        "duration_us": duration_us,
                        "after": gap.after.value(),
                        "before": gap.before.value(),
                    })
                })
                .collect::<Vec<_>>()
        };
        let mut crossing: Vec<_> = self
            .gaps
            .iter()
            .filter(|gap| gap.after.phase != gap.before.phase)
            .collect();
        crossing.sort_by(|left, right| {
            (right.end_ns - right.start_ns).cmp(&(left.end_ns - left.start_ns))
        });
        json!({
            "device_id": device_id,
            "span_ms": span_ms,
            "busy_ms": busy_ms,
            "idle_ms": idle_ms,
            "idle_fraction": occupancy.idle_fraction(),
            "gap_count": self.gaps.len(),
            // Per phase, because a whole-iteration idle figure is dominated by
            // the host time BETWEEN phases, which is a different fact from a
            // stall inside the forward pass.
            "phases": measurement
                .phase_summaries
                .iter()
                .filter(|summary| summary.device_id == device_id)
                .map(|summary| {
                    #[allow(
                        clippy::cast_precision_loss,
                        reason = "phase span is a ns span within one iteration, far below 2^53 ns"
                    )]
                    let phase_span_ms = (summary.span_ns.1 - summary.span_ns.0) as f64 / 1e6;
                    json!({
                        "phase": summary.phase,
                        "span_ms": phase_span_ms,
                        "busy_ms": summary.busy_union_ms,
                        "idle_ms": (phase_span_ms - summary.busy_union_ms).max(0.0),
                        "idle_fraction": (phase_span_ms > 0.0).then(|| {
                            (phase_span_ms - summary.busy_union_ms).max(0.0) / phase_span_ms
                        }),
                        "kernel_sum_ms": summary.kernel_sum_ms,
                        "kernel_count": summary.kernel_count,
                        "largest_gaps": largest_gaps(&summary.phase),
                    })
                })
                .collect::<Vec<_>>(),
            // The host time between phases. Not GPU stall in the same sense as a
            // gap inside `forward`, and much larger, so it is kept apart.
            "largest_inter_phase_gaps": crossing
                .into_iter()
                .take(TOP_GAPS)
                .map(|gap| {
                    #[allow(
                        clippy::cast_possible_wrap,
                        reason = "gap.start_ns is a trace timestamp; realistic trace/run spans \
                                  are far below i64::MAX ns (~292 years), so this never wraps"
                    )]
                    let start_ns = (gap.start_ns as i64) - (time_origin_ns as i64);
                    #[allow(
                        clippy::cast_precision_loss,
                        reason = "gap duration is a ns span within one iteration, far below 2^53 ns"
                    )]
                    let duration_us = (gap.end_ns - gap.start_ns) as f64 / 1e3;
                    json!({
                        "start_ns": start_ns,
                        "duration_us": duration_us,
                        "after": gap.after.value(),
                        "before": gap.before.value(),
                    })
                })
                .collect::<Vec<_>>(),
        })
    }
}

fn relative_pct(delta: f64, reference: f64) -> f64 {
    if reference.is_finite() && reference.abs() > 1e-12 {
        delta / reference * 100.0
    } else {
        0.0
    }
}

fn definitions() -> Value {
    json!({
        "time_origin_ns": "absolute nsys timestamp every *_ns in this file is measured from; kept as an offset so a JSON number never loses nanosecond precision",
        "anchor_ns": "where the modelled iteration is drawn from: the reference rank's first kernel start. An assumption, not a measurement — see meta.anchor_rule",
        "gpu_span_ns": "reference rank's [first kernel start, last kernel end]. The correlated kernel span, NOT the NVTX range: under CUDA graphs the range can close before its own kernels finish",
        "measured.kernels[].iv": "one [device_id, start_ns, end_ns, correlation_id, track_index] per rank launch of this kernel position, unreduced. Bubbles are the complement of the union of these over the span, and must be computed from these raw intervals rather than from drawn geometry; correlation_id is the NSYS launch identity used to connect the CUDA API lane",
        "measured.kernels[].iv[4]": "the concurrent CUDA stream (track) this launch ran on, 0 being the one that opened the range. Launches on different tracks of one device overlap in wall time, so a single lane per device would draw them as if they had been serial: give each (device, track) its own lane. A single-stream capture has track 0 only and draws exactly as before",
        "measured.kernels[].occ_ns": "this occurrence's cross-rank critical-path contribution (independent = slowest rank's duration; synchronizing collective = max(end) - max(start)). Their sum is measured.critical_path_ms",
        "simulated.slot_ms": "per-slot UNIT time. Feed these to the cost-tree unfold and let Scale{n} repeat them; critical-path-attributed times would render a losing Max branch as a zero-width leaf",
        "slot_multiplicity": "each slot's exact CostTree Scale multiplicity; folded workload = slot_ms x slot_multiplicity. Shared by every iteration because the tree shape is static",
        "operation_totals": "the ONLY sound join between the two lanes. occurrence_ratio is a counting ratio (3 measured kernels priced as 1 modelled leaf), never a per-kernel correspondence",
        "reference_rank.idle_ms": "span_ms - busy_ms over the reference rank's correlated kernels; this is the bubble budget the GPU lane draws",
        "selected_as": "why this iteration is in the file: group_median (a typical member of its kernel program), global_max_error / global_min_error (the capture's extremes), or manifest_override",
        "host.nvtx": "per thread, one [start_ns, duration_ns, string_id, depth] per NVTX range overlapping this iteration's host window. start_ns is anchor-relative like host.window_ns, NOT capture-relative like gpu_span_ns and the measured kernel intervals — the host block keeps small numbers because there are millions of these rows. Depth is measured containment on that thread, not an assumed outer/inner split",
        "host.api": "per thread, one [start_ns, duration_ns, string_id, class_index, correlation_id] per CUDA runtime call, anchor-relative like host.nvtx. class_index indexes meta.host_timeline.api_classes; correlation_id is the NSYS launch identity used to connect this call to one measured kernel; a thread between two calls is unaccounted for, not idle — the capture carries no CPU sampling",
        "host.window_ns": "this iteration's host window, anchor-relative. Wider than the GPU span on both sides, and overlapping its neighbours' — that overlap is the pipelining, see meta.host_timeline.window_rule",
        "iterations": "one summary row per emitted iteration — the scalars a picker sorts on. Every per-kernel and per-host-event field lives in the detail shard instead",
        "iteration_detail.byte_ranges": "iteration_id -> [byte offset, byte length] into the sibling .jsonl. Read that range and parse it as one JSON object; the whole file is never needed at once",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(iteration_id: u64, kind: &str, sequence: &str, relative: f64) -> IterationSummary {
        IterationSummary {
            case_index: iteration_id,
            iteration_id,
            stage: kind.to_string(),
            iteration_type: kind.to_string(),
            identity_sequence: sequence.to_string(),
            measured_ms: 1.0,
            simulated_ms: 1.0 + relative / 100.0,
            relative_diff_pct: relative,
        }
    }

    #[test]
    fn every_iteration_is_emitted_and_the_notable_ones_are_named() {
        // One big decode group and one small prefill group. The medians are the
        // typical members; -40 and 0 are the capture's extremes. Iterations 1
        // and 3 are unremarkable — emitted, but with nothing claimed about them.
        let summaries = vec![
            summary(1, "decode", "seq_a", 1.0),
            summary(2, "decode", "seq_a", 5.0),
            summary(3, "decode", "seq_a", 9.0),
            summary(4, "prefill", "seq_b", -40.0),
            summary(5, "prefill", "seq_b", 0.0),
        ];

        let chosen = select_iterations(&summaries, None).unwrap();
        let picked: Vec<_> = chosen
            .iter()
            .map(|(index, reason)| (summaries[*index].iteration_id, reason.label()))
            .collect();

        assert_eq!(
            picked,
            vec![
                (1, "full_capture"),
                (2, "group_median"),
                (3, "full_capture"),
                (4, "global_max_error"),
                (5, "global_min_error"),
            ]
        );
    }

    #[test]
    fn at_most_max_iterations_are_distinguished_however_many_groups_there_are() {
        // One group per iteration, so every one of them is a candidate median.
        let summaries: Vec<_> = (0..MAX_ITERATIONS as u64 + 10)
            .map(|index| {
                summary(
                    index,
                    "decode",
                    Box::leak(format!("seq_{index}").into_boxed_str()),
                    index as f64,
                )
            })
            .collect();

        let chosen = select_iterations(&summaries, None).unwrap();

        assert_eq!(chosen.len(), summaries.len());
        let named = chosen
            .iter()
            .filter(|(_, reason)| *reason != Selection::FullCapture)
            .count();
        assert_eq!(named, MAX_ITERATIONS);
    }

    #[test]
    fn selection_is_emitted_in_iteration_order() {
        let summaries = vec![
            summary(30, "decode", "seq_a", 2.0),
            summary(10, "prefill", "seq_b", 3.0),
            summary(20, "mixed", "seq_c", 4.0),
        ];

        let chosen = select_iterations(&summaries, None).unwrap();
        let ids: Vec<_> = chosen
            .iter()
            .map(|(index, _)| summaries[*index].iteration_id)
            .collect();

        assert_eq!(ids, vec![10, 20, 30]);
    }

    #[test]
    fn manifest_override_is_taken_verbatim() {
        let summaries = vec![
            summary(7, "decode", "seq_a", 1.0),
            summary(8, "decode", "seq_a", 2.0),
        ];

        let chosen = select_iterations(&summaries, Some(&[8])).unwrap();

        assert_eq!(chosen.len(), 1);
        assert_eq!(summaries[chosen[0].0].iteration_id, 8);
        assert_eq!(chosen[0].1.label(), "manifest_override");
    }

    #[test]
    fn manifest_override_rejects_an_iteration_the_capture_lacks() {
        let summaries = vec![summary(7, "decode", "seq_a", 1.0)];

        let error = select_iterations(&summaries, Some(&[7, 99])).unwrap_err();

        assert!(error.to_string().contains("99"), "{error}");
    }
}
