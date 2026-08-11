//! Scheduler-workload alignment by each run's recorded iteration id.
//!
//! This subject deliberately compares *scheduled attention work*, not resident
//! KV-cache occupancy.  For one iteration that work is the decode requests'
//! current KV lengths plus every scheduled prefill request's `(prefix + append)`
//! length.  Resident cache state belongs to the normal `kv-occupancy` subject.

use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
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

/// The `input_adapter` tags this build knows how to read, one per instrumented
/// serving engine. The record shape behind them is identical.
const SUPPORTED_INPUT_ADAPTERS: &[&str] = &["vllm_text", "sglang_text"];

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
    dp_rank: Option<u64>,
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
        // Both instrumented forks emit this record in the same shape; the tag
        // names which one produced it. Unknown tags are still rejected -- the
        // check is that the dialect is one this build knows, not that it is vLLM.
        ensure!(
            SUPPORTED_INPUT_ADAPTERS.contains(&record.input_adapter.as_str()),
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

    // Data-parallel ranks each log the same step, so one step appears as several
    // records. Fold them into the replica's batch shape (the union of the ranks'
    // local batches) before anything reads a step's workload, and before the
    // contiguity scan below — several ranks' records are the same step, not a
    // break in the sequence.
    let records = fold_data_parallel_ranks(records)?;

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

/// Fold every data-parallel rank's record for one step into the replica's batch
/// shape. DP ranks execute one step in lockstep behind the expert-parallel
/// collectives, so the replica's workload for that step is the union of the
/// ranks' local batches.
///
/// What identifies "one step" is the question this function used to get wrong.
/// `iteration_index` counts a single EngineCore's own scheduled steps, and the
/// ranks do not start counting together: in the GLM-5.2 DP8 capture the very
/// same wall-clock step is iteration 11 on rank 0, 7 on ranks 1-4 and 6 on
/// ranks 5-7. Folding by index therefore merged eight *different* steps, and
/// the mixed prefill steps it produced were pure fiction. When the records
/// carry engine-observed timestamps (schema 2) the fold pairs ranks by
/// wall-clock overlap against a reference rank, exactly as `alignment/nsys/
/// parse.py` does for the kernel side.
///
/// Schema-1 records have no timestamps, so there is nothing to pair on and the
/// index fold is kept — its result only feeds the preflight-vs-replay segment
/// scan, after which schema-1 captures fall back to the NSYS timeline entirely.
/// A single-rank capture passes through unchanged either way.
fn fold_data_parallel_ranks(records: Vec<FullMeasuredMetrics>) -> Result<Vec<FullMeasuredMetrics>> {
    let observed = records
        .iter()
        .all(|record| record.observed_start_monotonic_ns.is_some())
        && records.iter().all(|record| record.dp_rank.is_some());
    if observed {
        fold_by_wall_clock(records)
    } else {
        fold_by_iteration_index(records)
    }
}

/// Merge one rank's record into the step being built.
fn merge_rank_record(merged: &mut FullMeasuredMetrics, record: FullMeasuredMetrics) -> Result<()> {
    ensure!(
        merged.schema_version == record.schema_version,
        "step at iteration {} mixes full-run metrics schema versions {} and {}",
        merged.iteration_index,
        merged.schema_version,
        record.schema_version
    );
    ensure!(
        merged.input_adapter == record.input_adapter,
        "step at iteration {} mixes input adapters {:?} and {:?}",
        merged.iteration_index,
        merged.input_adapter,
        record.input_adapter
    );
    merged.prefill_tokens += record.prefill_tokens;
    merged.decode_kv_lens.extend(record.decode_kv_lens);
    merged
        .prefill_chunk_pairs
        .extend(record.prefill_chunk_pairs);
    // The step spans from the earliest rank's start to the latest rank's end;
    // its duration is the slowest rank's, since the collectives make every rank
    // wait for it.
    merged.observed_start_monotonic_ns = match (
        merged.observed_start_monotonic_ns,
        record.observed_start_monotonic_ns,
    ) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (left, right) => left.or(right),
    };
    merged.observed_end_monotonic_ns = match (
        merged.observed_end_monotonic_ns,
        record.observed_end_monotonic_ns,
    ) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (left, right) => left.or(right),
    };
    merged.observed_elapsed_ms = match (merged.observed_elapsed_ms, record.observed_elapsed_ms) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (left, right) => left.or(right),
    };
    Ok(())
}

fn fold_by_iteration_index(records: Vec<FullMeasuredMetrics>) -> Result<Vec<FullMeasuredMetrics>> {
    let mut order: Vec<u64> = Vec::new();
    let mut folded: HashMap<u64, FullMeasuredMetrics> = HashMap::new();
    for record in records {
        match folded.entry(record.iteration_index) {
            Entry::Vacant(slot) => {
                order.push(record.iteration_index);
                slot.insert(record);
            }
            Entry::Occupied(mut slot) => merge_rank_record(slot.get_mut(), record)?,
        }
    }
    Ok(order
        .into_iter()
        .map(|iteration| folded.remove(&iteration).expect("iteration was inserted"))
        .collect())
}

/// The observed window of a record, which the caller has already checked exists.
fn observed_window(record: &FullMeasuredMetrics) -> (u64, u64) {
    let start = record
        .observed_start_monotonic_ns
        .expect("caller checked observed_start_monotonic_ns");
    let end = record.observed_end_monotonic_ns.unwrap_or(start);
    (start, end.max(start))
}

/// Pair ranks by wall-clock overlap against the lowest-numbered rank.
///
/// The reference rank defines the steps; every other rank's record joins the
/// reference step it overlaps most. A peer record that overlaps no reference
/// step *inside* the reference's span is refused rather than dropped or given a
/// step of its own — it would mean the ranks are not stepping together, which
/// invalidates the union the fold is built on, and that deserves to be read by a
/// person rather than averaged away.
///
/// The two RAGGED ENDS are a different thing and are dropped with a count. Data
/// parallel ranks do not start or retire on the same step: in the GLM-5.2 DP8
/// run the ranks began 0.0-1.6 s apart and ran 6154..6159 iterations each, so
/// rank 3's last step opened 1.09 ms after rank 0 had finished for good. Exactly
/// one peer record of 43,091 fell outside, at the very end, with zero misses
/// inside the span — and refusing on it withheld the whole subject. That is the
/// wrong trade: this is the one view that compares SCHEDULED WORK per step, and
/// it is what would have shown the simulator serialising its prefills across DP
/// groups (one group prefilling per iteration where vLLM ran all eight), a bug
/// that instead went unseen until the e2e throughput was traced by hand.
///
/// A record in a gap BETWEEN reference steps still bails: that is a real desync,
/// not an edge.
fn fold_by_wall_clock(records: Vec<FullMeasuredMetrics>) -> Result<Vec<FullMeasuredMetrics>> {
    let reference_rank = records
        .iter()
        .filter_map(|record| record.dp_rank)
        .min()
        .expect("caller checked dp_rank");
    let (reference_records, peer_records): (Vec<_>, Vec<_>) = records
        .into_iter()
        .partition(|record| record.dp_rank == Some(reference_rank));
    let mut steps: Vec<FullMeasuredMetrics> = reference_records;
    steps.sort_by_key(|record| observed_window(record).0);
    let mut windows: Vec<(u64, u64)> = steps.iter().map(observed_window).collect();

    let mut peers = peer_records;
    peers.sort_by_key(|record| observed_window(record).0);
    // The reference's own span. Anything wholly outside it is a ragged end.
    let (span_start, span_end) = match (windows.first(), windows.last()) {
        (Some(first), Some(last)) => (first.0, last.1),
        _ => bail!("reference rank {reference_rank} contributed no steps"),
    };
    let mut dropped_before = 0usize;
    let mut dropped_after = 0usize;
    let mut cursor = 0usize;
    for peer in peers {
        let (peer_start, peer_end) = observed_window(&peer);
        if peer_end <= span_start {
            dropped_before += 1;
            continue;
        }
        if peer_start >= span_end {
            dropped_after += 1;
            continue;
        }
        // The reference windows are sorted and disjoint, so a peer that starts
        // after this one ends can never match an earlier step: the cursor only
        // moves forward across the whole pass.
        while cursor + 1 < windows.len() && windows[cursor].1 <= peer_start {
            cursor += 1;
        }
        let best = (cursor..windows.len())
            .take_while(|&index| windows[index].0 < peer_end)
            .max_by_key(|&index| {
                windows[index]
                    .1
                    .min(peer_end)
                    .saturating_sub(windows[index].0.max(peer_start))
            })
            .filter(|&index| windows[index].1.min(peer_end) > windows[index].0.max(peer_start));
        let Some(index) = best else {
            // Inside the reference's span, so not a ragged end: the peer landed
            // in a gap BETWEEN two reference steps, which is a real desync.
            bail!(
                "full-run metrics rank {:?} iteration {} falls between steps of reference rank \
                 {} (inside its {:.3} s span); the data-parallel ranks are not stepping together",
                peer.dp_rank,
                peer.iteration_index,
                reference_rank,
                (span_end - span_start) as f64 / 1e9,
            );
        };
        merge_rank_record(&mut steps[index], peer)?;
        windows[index] = observed_window(&steps[index]);
    }
    if dropped_before + dropped_after > 0 {
        eprintln!(
            "[alignment-workload] dropped {} peer record(s) outside reference rank {}'s span \
             ({} before its first step, {} after its last): data-parallel ranks do not start or \
             retire on the same step",
            dropped_before + dropped_after,
            reference_rank,
            dropped_before,
            dropped_after,
        );
    }
    Ok(steps)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn rank_record(
        iteration_index: u64,
        prefill_tokens: u64,
        decode_kv_lens: Vec<u64>,
        observed: Option<(u64, u64, f64)>,
    ) -> FullMeasuredMetrics {
        FullMeasuredMetrics {
            schema_version: if observed.is_some() { 2 } else { 1 },
            input_adapter: "vllm_text".to_string(),
            iteration_index,
            prefill_tokens,
            decode_kv_lens,
            prefill_chunk_pairs: Vec::new(),
            dp_rank: None,
            observed_start_monotonic_ns: observed.map(|(start, _, _)| start),
            observed_end_monotonic_ns: observed.map(|(_, end, _)| end),
            observed_elapsed_ms: observed.map(|(_, _, elapsed)| elapsed),
        }
    }

    /// The same record, attributed to a data-parallel rank. Only records that
    /// carry both a rank and a timestamp can be paired by wall clock.
    fn ranked_record(
        dp_rank: u64,
        iteration_index: u64,
        prefill_tokens: u64,
        decode_kv_lens: Vec<u64>,
        observed: (u64, u64, f64),
    ) -> FullMeasuredMetrics {
        FullMeasuredMetrics {
            dp_rank: Some(dp_rank),
            ..rank_record(
                iteration_index,
                prefill_tokens,
                decode_kv_lens,
                Some(observed),
            )
        }
    }

    #[test]
    fn a_single_rank_capture_passes_through_unchanged() {
        let folded = fold_data_parallel_ranks(vec![
            rank_record(7, 4, vec![10], None),
            rank_record(8, 0, vec![11, 12], None),
        ])
        .unwrap();

        assert_eq!(
            folded
                .iter()
                .map(|record| (record.iteration_index, record.prefill_tokens))
                .collect::<Vec<_>>(),
            vec![(7, 4), (8, 0)]
        );
    }

    #[test]
    fn data_parallel_ranks_fold_into_the_replica_batch() {
        // Two ranks log the same step. The replica's batch is their union, and
        // the step spans the earliest start to the latest end.
        let folded = fold_data_parallel_ranks(vec![
            rank_record(7, 4, vec![10], Some((100, 300, 2.0))),
            rank_record(7, 6, vec![11, 12], Some((110, 350, 2.4))),
        ])
        .unwrap();

        assert_eq!(folded.len(), 1);
        let step = &folded[0];
        assert_eq!(step.iteration_index, 7);
        assert_eq!(step.prefill_tokens, 10);
        assert_eq!(step.decode_kv_lens, vec![10, 11, 12]);
        assert_eq!(step.observed_start_monotonic_ns, Some(100));
        assert_eq!(step.observed_end_monotonic_ns, Some(350));
        assert_eq!(step.observed_elapsed_ms, Some(2.4));
    }

    #[test]
    fn folding_keeps_a_repeated_index_from_breaking_the_contiguity_scan() {
        // The DP shape that used to split every step into its own segment:
        // 8 records per iteration means the raw `+1` scan never continues.
        let records: Vec<_> = (7..=9)
            .flat_map(|iteration| (0..8).map(move |_| rank_record(iteration, 1, vec![5], None)))
            .collect();

        let folded = fold_data_parallel_ranks(records).unwrap();

        assert_eq!(
            folded
                .iter()
                .map(|record| record.iteration_index)
                .collect::<Vec<_>>(),
            vec![7, 8, 9]
        );
        assert!(folded
            .windows(2)
            .all(|pair| pair[0].iteration_index + 1 == pair[1].iteration_index));
    }

    #[test]
    fn mixed_schema_versions_within_one_step_are_rejected() {
        let error = fold_data_parallel_ranks(vec![
            rank_record(7, 4, vec![10], None),
            rank_record(7, 6, vec![11], Some((100, 200, 1.0))),
        ])
        .unwrap_err();

        assert!(
            error.to_string().contains("mixes full-run metrics schema"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn ranks_that_count_their_own_steps_still_fold_by_wall_clock() {
        // The GLM-5.2 DP8 shape, minimised: one wall-clock step is iteration 11
        // on the reference rank and 7 on its peer, and the next step is 12/8.
        // Folding by index would pair 11 with the peer's *later* step.
        let folded = fold_data_parallel_ranks(vec![
            ranked_record(0, 11, 4, vec![10], (100, 300, 2.0)),
            ranked_record(0, 12, 5, vec![20], (400, 600, 2.0)),
            ranked_record(3, 7, 6, vec![11], (110, 310, 2.4)),
            ranked_record(3, 8, 7, vec![21], (405, 615, 2.5)),
        ])
        .unwrap();

        assert_eq!(folded.len(), 2);
        assert_eq!(folded[0].iteration_index, 11);
        assert_eq!(folded[0].prefill_tokens, 10);
        assert_eq!(folded[0].decode_kv_lens, vec![10, 11]);
        assert_eq!(folded[0].observed_start_monotonic_ns, Some(100));
        assert_eq!(folded[0].observed_end_monotonic_ns, Some(310));
        assert_eq!(folded[1].iteration_index, 12);
        assert_eq!(folded[1].prefill_tokens, 12);
        assert_eq!(folded[1].decode_kv_lens, vec![20, 21]);
    }

    #[test]
    fn a_peer_step_in_a_gap_between_reference_steps_is_refused_rather_than_averaged_away() {
        // Two reference steps with 300..400 empty between them, and a peer that
        // lives entirely in that hole. Inside the span and matching nothing is a
        // real desync: the union the fold is built on does not hold, and that
        // must reach a person instead of being averaged in.
        let error = fold_data_parallel_ranks(vec![
            ranked_record(0, 11, 4, vec![10], (100, 300, 2.0)),
            ranked_record(0, 12, 5, vec![20], (400, 600, 2.0)),
            ranked_record(3, 7, 6, vec![11], (310, 390, 0.8)),
        ])
        .unwrap_err();

        assert!(
            error.to_string().contains("not stepping together"),
            "unexpected error: {error}"
        );
        assert!(
            error.to_string().contains("falls between steps"),
            "the message must say WHERE it fell, or the ragged-end case reads the \
             same as a desync: {error}"
        );
    }

    #[test]
    fn a_peer_step_past_the_reference_span_is_a_ragged_end_and_is_dropped() {
        // Data-parallel ranks do not retire on the same step. In the GLM-5.2 DP8
        // run the ranks ran 6154..6159 iterations each and rank 3's last step
        // opened 1.09 ms after rank 0 had finished for good -- one peer record of
        // 43,091, with zero misses inside the span. Refusing on that withheld the
        // whole scheduler-shape subject, which is the one view that compares
        // scheduled work per step.
        let folded = fold_data_parallel_ranks(vec![
            ranked_record(0, 11, 4, vec![10], (100, 300, 2.0)),
            ranked_record(3, 7, 6, vec![11], (110, 310, 2.4)),
            ranked_record(3, 8, 7, vec![21], (900, 1000, 1.0)),
        ])
        .expect("a ragged end must not withhold the subject");

        // The overlapping peer still folded; only the trailing one went.
        assert_eq!(folded.len(), 1);
        assert_eq!(folded[0].iteration_index, 11);
        assert_eq!(folded[0].decode_kv_lens, vec![10, 11]);
        assert_eq!(folded[0].prefill_tokens, 10);
    }

    #[test]
    fn a_peer_step_before_the_reference_span_is_dropped_the_same_way() {
        // The other end: ranks do not start together either (0.0..1.6 s apart in
        // the measured run), so the reference rank may open after a peer.
        let folded = fold_data_parallel_ranks(vec![
            ranked_record(0, 11, 4, vec![10], (100, 300, 2.0)),
            ranked_record(3, 6, 9, vec![9], (1, 50, 0.5)),
            ranked_record(3, 7, 6, vec![11], (110, 310, 2.4)),
        ])
        .expect("a ragged start must not withhold the subject either");

        assert_eq!(folded.len(), 1);
        assert_eq!(folded[0].decode_kv_lens, vec![10, 11]);
    }

    #[test]
    fn records_without_timestamps_keep_the_index_fold() {
        // Schema 1 has nothing to pair on. The index fold is what feeds the
        // preflight-vs-replay segment scan, so it must survive untouched.
        let folded = fold_data_parallel_ranks(vec![
            FullMeasuredMetrics {
                dp_rank: Some(0),
                ..rank_record(7, 4, vec![10], None)
            },
            FullMeasuredMetrics {
                dp_rank: Some(3),
                ..rank_record(7, 6, vec![11], None)
            },
        ])
        .unwrap();

        assert_eq!(folded.len(), 1);
        assert_eq!(folded[0].prefill_tokens, 10);
    }
}
