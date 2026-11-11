//! Scheduler-workload alignment by each run's recorded iteration id.
//!
//! This subject deliberately compares *scheduled attention work*, not resident
//! KV-cache occupancy.  For one iteration that work is the decode requests'
//! current KV lengths plus every scheduled prefill request's `(prefix + append)`
//! length.  Resident cache state belongs to the normal `kv-occupancy` subject.

use std::collections::{BTreeMap, BTreeSet, HashSet};
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

#[derive(Debug, Clone, Deserialize)]
struct FullMeasuredMetrics {
    schema_version: u32,
    input_adapter: String,
    /// The rank's scheduled-step index; null on a DP dummy forward (schema 5).
    #[serde(default)]
    iteration_index: Option<u64>,
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
    /// Per-rank count of model forwards, DP dummies included (schema 5).
    #[serde(default)]
    forward_ordinal: Option<u64>,
    /// A DP dummy forward: the rank's participation, no scheduled work.
    #[serde(default)]
    is_dummy: bool,
    /// How many forwards the engine can have launched but not finished: its
    /// batch-queue depth (schema 5).
    #[serde(default)]
    max_forwards_in_flight: Option<u64>,
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
    let (measured, fold) = read_measured_points(
        &input.metrics_jsonl,
        input.replay_start_monotonic_ns,
        input.replay_end_monotonic_ns,
    )?;
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
            "workload_profile_log_dir": input.workload_profile_log_dir.display().to_string(),
            "simulation_log_dir": simulation_log_dir.display().to_string(),
            "measured_timeline_source": "full_run_engine_observation",
            "measured_iterations": measured.len(),
            "simulated_iterations": simulated.len(),
            "measured_span_ms": measured.last().map(|point| point.time_ms),
            "simulated_span_ms": simulated.last().map(|point| point.time_ms),
            "measured_iteration_cycles": measured.iter().filter(|point| point.iteration_cycle_ms.is_some()).count(),
            "simulated_iteration_cycles": simulated.iter().filter(|point| point.iteration_cycle_ms.is_some()).count(),
            "measured_dp_fold": fold.as_json(),
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
    replay_start_monotonic_ns: Option<u64>,
    replay_end_monotonic_ns: Option<u64>,
) -> Result<(Vec<WorkloadPoint>, FoldStats)> {
    ensure!(
        replay_start_monotonic_ns.is_some() == replay_end_monotonic_ns.is_some(),
        "workload replay window must provide both start and end monotonic timestamps"
    );
    if let (Some(start), Some(end)) = (replay_start_monotonic_ns, replay_end_monotonic_ns) {
        ensure!(end >= start, "workload replay window ends before it starts");
    }

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
            "unsupported full-run metrics input_adapter {:?} at iteration {:?}",
            record.input_adapter,
            record.iteration_index
        );
        ensure!(
            (2..=5).contains(&record.schema_version),
            "alignment-workload requires schema-v2..v5 full-run metrics with host-monotonic timestamps; got schema_version {} at iteration {:?}",
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

    // Each data-parallel rank logs its own part of a global step. Pair the
    // ranks by forward ordinal and fold them into the replica's batch shape.
    let (records, fold) = fold_data_parallel_ranks(records)?;

    // New captures record req-frontend's replay window in the same host
    // CLOCK_MONOTONIC domain as EngineCore. Filter by that window to exclude
    // startup/preflight work directly. Older schema-v2 captures lack this pair;
    // retain their complete extracted stream instead of consulting NSYS.
    let selected: Vec<_> = records
        .into_iter()
        .filter(
            |record| match (replay_start_monotonic_ns, replay_end_monotonic_ns) {
                (Some(window_start), Some(window_end)) => {
                    let start = record.observed_start_monotonic_ns.unwrap_or(0);
                    let end = record.observed_end_monotonic_ns.unwrap_or(start);
                    end >= window_start && start <= window_end
                }
                (None, None) => true,
                _ => unreachable!("window pair was validated above"),
            },
        )
        .collect();
    ensure!(
        !selected.is_empty(),
        "full-run metrics contain no iterations inside the recorded replay window"
    );

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
            "full-run metrics iteration {:?} ends before it starts",
            record.iteration_index
        );
        ensure!(
            observed_elapsed_ms.is_finite() && observed_elapsed_ms >= 0.0,
            "full-run metrics iteration {:?} has invalid observed_elapsed_ms {}",
            record.iteration_index,
            observed_elapsed_ms
        );
        let scheduled_kv_tokens =
            scheduled_kv_tokens(&record.decode_kv_lens, &record.prefill_chunk_pairs);
        points.push(WorkloadPoint {
            iteration_id: record
                .iteration_index
                .context("a folded workload step carries no iteration id")?,
            time_ms: start_ns.saturating_sub(origin_ns) as f64 / 1e6,
            iteration_cycle_ms: None,
            prefill_tokens: record.prefill_tokens,
            decode_batch_size: record.decode_kv_lens.len() as u64,
            scheduled_kv_tokens,
            observed_elapsed_ms: Some(observed_elapsed_ms),
        });
    }
    assign_iteration_cycles(&mut points);
    Ok((points, fold))
}

/// What folding the data-parallel ranks did, reported in the report's meta so a
/// reader can see which records became which steps and how well the ranks'
/// observation windows agree.
#[derive(Debug, Clone, Default, PartialEq)]
struct FoldStats {
    /// `forward_ordinal` or `single_rank`.
    method: &'static str,
    ranks: usize,
    /// Records outside the ordinal range every rank covers (ranks start and
    /// stop recording at different forwards).
    dropped_before: usize,
    dropped_after: usize,
    /// Records of DP dummy forwards: a rank's participation in a step, with no
    /// scheduled work.
    dummy_records: usize,
    /// Global forwards in which every rank ran a dummy. They schedule no work
    /// and are not workload steps.
    all_dummy_steps: usize,
    /// Steps whose ranks' observation windows do not all share an instant.
    /// Allowed: a dummy's window brackets the launch-time DP sync, a real
    /// step's the later result wait.
    non_overlapping_steps: usize,
    /// Steps whose start (earliest real-record start) precedes the previous
    /// step's; their cycle is negative and is dropped from the cycle stats.
    non_monotone_starts: usize,
    /// Spread of the ranks' window starts within one step, in ms.
    start_skew_ms: Option<Value>,
}

impl FoldStats {
    fn as_json(&self) -> Value {
        json!({
            "method": self.method,
            "ranks": self.ranks,
            "ragged_dropped_before": self.dropped_before,
            "ragged_dropped_after": self.dropped_after,
            "dummy_records": self.dummy_records,
            "all_dummy_steps": self.all_dummy_steps,
            "non_overlapping_steps": self.non_overlapping_steps,
            "non_monotone_starts": self.non_monotone_starts,
            "start_skew_ms": self.start_skew_ms,
        })
    }
}

/// Fold every data-parallel rank's record of one global step into the
/// replica's batch shape: the union of the ranks' local batches.
///
/// One global step is one `forward_ordinal`. Each DP rank runs exactly one
/// model forward per EngineCore busy-loop pass while a wave runs (a real step,
/// or a dummy forward when it has no ready work), and every forward syncs its
/// batch across the DP ranks, so the k-th forward of every rank is the same
/// global step. Schema-5 captures record every forward, dummies included, with
/// that ordinal (vLLM reference patch 0019).
///
/// Wall clock only checks the pairing. Pairing by wall clock is what this fold
/// used to do, and the records do not support it: a record's window is the
/// EngineCore's result wait plus sampling, not the GPU forward, so under async
/// scheduling the windows of one global step are offset across ranks, and
/// idle ranks' dummy forwards were not recorded at all.
///
/// A multi-rank capture without ordinals is refused. A single rank passes
/// through, sorted by observation start when it has timestamps.
fn fold_data_parallel_ranks(
    records: Vec<FullMeasuredMetrics>,
) -> Result<(Vec<FullMeasuredMetrics>, FoldStats)> {
    let ranks: BTreeSet<u64> = records
        .iter()
        .map(|record| record.dp_rank.unwrap_or(0))
        .collect();
    if ranks.len() <= 1 {
        return fold_single_rank(records);
    }
    ensure!(
        records
            .iter()
            .all(|record| record.forward_ordinal.is_some()),
        "this capture has {} data-parallel ranks but its records carry no forward_ordinal: it \
         predates forward ordinals (full-run metrics schema 5, vLLM reference patch 0019), so its \
         ranks' records cannot be joined into global steps; recapture the workload pass",
        ranks.len()
    );
    fold_by_forward_ordinal(records, ranks.len())
}

fn fold_single_rank(
    records: Vec<FullMeasuredMetrics>,
) -> Result<(Vec<FullMeasuredMetrics>, FoldStats)> {
    let dummy_records = records.iter().filter(|record| record.is_dummy).count();
    let mut steps: Vec<_> = records
        .into_iter()
        .filter(|record| !record.is_dummy)
        .collect();
    let mut seen = HashSet::new();
    for step in &steps {
        let index = step
            .iteration_index
            .context("a real full-run metrics record carries no iteration_index")?;
        // Without dp_rank an unlabeled multi-rank capture looks like one rank
        // whose steps repeat; its ranks cannot be joined into global steps.
        ensure!(
            seen.insert(index),
            "full-run metrics repeat iteration_index {index} without a dp_rank: the records come \
             from several data-parallel ranks but carry no rank or forward_ordinal: the \
             capture predates forward ordinals (schema 5, vLLM reference patch 0019); recapture the \
             workload pass"
        );
    }
    if steps
        .iter()
        .all(|step| step.observed_start_monotonic_ns.is_some())
    {
        steps.sort_by_key(|step| {
            (
                step.observed_start_monotonic_ns,
                step.forward_ordinal,
                step.iteration_index,
            )
        });
    }
    Ok((
        steps,
        FoldStats {
            method: "single_rank",
            ranks: 1,
            dummy_records,
            ..FoldStats::default()
        },
    ))
}

/// Merge one rank's record into the step being built.
fn merge_rank_record(merged: &mut FullMeasuredMetrics, record: FullMeasuredMetrics) -> Result<()> {
    ensure!(
        merged.schema_version == record.schema_version,
        "step at forward {:?} mixes full-run metrics schema versions {} and {}",
        merged.forward_ordinal,
        merged.schema_version,
        record.schema_version
    );
    ensure!(
        merged.input_adapter == record.input_adapter,
        "step at forward {:?} mixes input adapters {:?} and {:?}",
        merged.forward_ordinal,
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

/// The observed window of a record, which the caller has already checked exists.
fn observed_window(record: &FullMeasuredMetrics) -> Option<(u64, u64)> {
    let start = record.observed_start_monotonic_ns?;
    let end = record.observed_end_monotonic_ns.unwrap_or(start);
    Some((start, end.max(start)))
}

fn fold_by_forward_ordinal(
    records: Vec<FullMeasuredMetrics>,
    rank_count: usize,
) -> Result<(Vec<FullMeasuredMetrics>, FoldStats)> {
    let dummy_records = records.iter().filter(|record| record.is_dummy).count();
    // A rank launches forward k only after it began waiting on forward
    // k - in_flight (its batch queue is full until then).
    let mut in_flight = 1;
    for record in &records {
        let depth = record.max_forwards_in_flight.with_context(|| {
            format!(
                "dp_rank {:?} forward {:?} carries no max_forwards_in_flight",
                record.dp_rank, record.forward_ordinal
            )
        })?;
        ensure!(depth >= 1, "max_forwards_in_flight must be at least 1");
        in_flight = in_flight.max(depth);
    }
    let mut by_rank: BTreeMap<u64, BTreeMap<u64, FullMeasuredMetrics>> = BTreeMap::new();
    for record in records {
        let rank = record.dp_rank.unwrap_or(0);
        let ordinal = record
            .forward_ordinal
            .expect("caller checked forward_ordinal");
        ensure!(
            record.is_dummy || record.iteration_index.is_some(),
            "a real record of dp_rank {rank} forward {ordinal} carries no iteration_index"
        );
        let previous = by_rank.entry(rank).or_default().insert(ordinal, record);
        ensure!(
            previous.is_none(),
            "dp_rank {rank} records forward {ordinal} twice"
        );
    }

    // The ordinals every rank covers. Ranks start and stop recording at
    // different forwards (the measurement log starts at an offset, the last
    // records can be cut), so the ends are ragged and dropped with a count.
    let first = by_rank
        .values()
        .map(|forwards| *forwards.keys().next().expect("non-empty"))
        .max()
        .expect("ranks");
    let last = by_rank
        .values()
        .map(|forwards| *forwards.keys().next_back().expect("non-empty"))
        .min()
        .expect("ranks");
    ensure!(
        first <= last,
        "the data-parallel ranks' recorded forwards do not overlap: the common ordinal range \
         would be {first}..={last}"
    );
    let mut dropped_before = 0usize;
    let mut dropped_after = 0usize;
    for forwards in by_rank.values() {
        dropped_before += forwards.range(..first).count();
        dropped_after += forwards.range(last + 1..).count();
    }
    for (rank, forwards) in &by_rank {
        let present = forwards.range(first..=last).count() as u64;
        let expected = last - first + 1;
        if present != expected {
            let missing = (first..=last)
                .find(|ordinal| !forwards.contains_key(ordinal))
                .expect("a gap exists");
            bail!(
                "dp_rank {rank} records {present} of the {expected} forwards {first}..={last} \
                 every rank covers (first missing: {missing}); every rank runs one forward per \
                 global step, so a missing record means a lost log line, not an idle rank"
            );
        }
    }

    // Wall clock checks the pairing. A rank's forward k cannot finish before
    // every rank launched it, and a rank launches k only after it began waiting
    // on k - in_flight, so one rank's window for k ends after any rank's window
    // for k - in_flight starts. A violation means
    // the ordinals do not name the same global step.
    let windows: Vec<BTreeMap<u64, Option<(u64, u64)>>> = by_rank
        .values()
        .map(|forwards| {
            forwards
                .range(first..=last)
                .map(|(ordinal, record)| (*ordinal, observed_window(record)))
                .collect()
        })
        .collect();
    for ordinal in (first + in_flight)..=last {
        let earliest_end = windows
            .iter()
            .filter_map(|rank| rank[&ordinal].map(|window| window.1))
            .min();
        let latest_earlier_start = windows
            .iter()
            .filter_map(|rank| rank[&(ordinal - in_flight)].map(|window| window.0))
            .max();
        if let (Some(end), Some(start)) = (earliest_end, latest_earlier_start) {
            ensure!(
                end > start,
                "forward {ordinal} ended on one rank ({end} ns) before another rank began waiting \
                 on forward {} ({start} ns): the ranks' forward ordinals do not name the same \
                 global step",
                ordinal - in_flight
            );
        }
    }

    let mut steps = Vec::new();
    let mut all_dummy_steps = 0usize;
    let mut non_overlapping_steps = 0usize;
    let mut skews_ms = Vec::new();
    let mut ranks_iter: Vec<_> = by_rank.into_values().collect();
    for ordinal in first..=last {
        let records: Vec<FullMeasuredMetrics> = ranks_iter
            .iter_mut()
            .map(|forwards| forwards.remove(&ordinal).expect("checked contiguous"))
            .collect();
        let observed: Vec<(u64, u64)> = records.iter().filter_map(observed_window).collect();
        if observed.len() == records.len() {
            let latest_start = observed.iter().map(|window| window.0).max().expect("ranks");
            let earliest_start = observed.iter().map(|window| window.0).min().expect("ranks");
            let earliest_end = observed.iter().map(|window| window.1).min().expect("ranks");
            if earliest_end < latest_start {
                non_overlapping_steps += 1;
            }
            skews_ms.push((latest_start - earliest_start) as f64 / 1e6);
        }
        // Dummy forwards are the idle ranks' participation: no scheduled work.
        let mut real = records.into_iter().filter(|record| !record.is_dummy);
        let Some(mut step) = real.next() else {
            all_dummy_steps += 1;
            continue;
        };
        step.dp_rank = None;
        for record in real {
            merge_rank_record(&mut step, record)?;
        }
        // Under DP a step's identity is its global forward ordinal.
        step.iteration_index = Some(ordinal);
        steps.push(step);
    }
    let non_monotone_starts = steps
        .windows(2)
        .filter(|pair| {
            matches!(
                (pair[0].observed_start_monotonic_ns, pair[1].observed_start_monotonic_ns),
                (Some(left), Some(right)) if right < left
            )
        })
        .count();
    let start_skew_ms = (!skews_ms.is_empty())
        .then(|| serde_json::to_value(stats(&clean_nonnegative_sorted(&skews_ms))))
        .transpose()?;
    let fold = FoldStats {
        method: "forward_ordinal",
        ranks: rank_count,
        dropped_before,
        dropped_after,
        dummy_records,
        all_dummy_steps,
        non_overlapping_steps,
        non_monotone_starts,
        start_skew_ms,
    };
    if dropped_before + dropped_after > 0 {
        eprintln!(
            "[alignment-workload] dropped {} record(s) outside the forwards every rank covers \
             ({dropped_before} before, {dropped_after} after): ranks start and stop recording at \
             different forwards",
            dropped_before + dropped_after
        );
    }
    Ok((steps, fold))
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
            iteration_index: Some(iteration_index),
            prefill_tokens,
            decode_kv_lens,
            prefill_chunk_pairs: Vec::new(),
            dp_rank: None,
            observed_start_monotonic_ns: observed.map(|(start, _, _)| start),
            observed_end_monotonic_ns: observed.map(|(_, end, _)| end),
            observed_elapsed_ms: observed.map(|(_, _, elapsed)| elapsed),
            forward_ordinal: None,
            is_dummy: false,
            max_forwards_in_flight: None,
        }
    }

    /// A schema-5 real forward of one data-parallel rank. Its window is
    /// `[10 * ordinal + offset, 10 * ordinal + offset + 8]` ns.
    fn forward(
        dp_rank: u64,
        ordinal: u64,
        iteration_index: u64,
        prefill_tokens: u64,
        decode_kv_lens: Vec<u64>,
        offset: u64,
    ) -> FullMeasuredMetrics {
        let start = 10 * ordinal + offset;
        FullMeasuredMetrics {
            schema_version: 5,
            dp_rank: Some(dp_rank),
            forward_ordinal: Some(ordinal),
            max_forwards_in_flight: Some(2),
            ..rank_record(
                iteration_index,
                prefill_tokens,
                decode_kv_lens,
                Some((start, start + 8, 8e-6)),
            )
        }
    }

    /// A schema-5 dummy forward: the idle rank's participation in a step.
    fn dummy(dp_rank: u64, ordinal: u64, offset: u64) -> FullMeasuredMetrics {
        FullMeasuredMetrics {
            iteration_index: None,
            is_dummy: true,
            ..forward(dp_rank, ordinal, 0, 0, Vec::new(), offset)
        }
    }

    #[test]
    fn workload_metrics_run_may_have_a_shifted_iteration_origin() {
        let directory = tempfile::tempdir().unwrap();
        let metrics_path = directory.path().join("metrics.jsonl");
        fs::write(
            &metrics_path,
            concat!(
                "{\"schema_version\":2,\"input_adapter\":\"vllm_text\",\"iteration_index\":1200,\"prefill_tokens\":8,\"decode_kv_lens\":[],\"prefill_chunk_pairs\":[[0,8]],\"observed_start_monotonic_ns\":100,\"observed_end_monotonic_ns\":200,\"observed_elapsed_ms\":0.0001}\n",
                "{\"schema_version\":2,\"input_adapter\":\"vllm_text\",\"iteration_index\":1201,\"prefill_tokens\":0,\"decode_kv_lens\":[8],\"prefill_chunk_pairs\":[],\"observed_start_monotonic_ns\":300,\"observed_end_monotonic_ns\":400,\"observed_elapsed_ms\":0.0001}\n",
            ),
        )
        .unwrap();

        let (points, _) = read_measured_points(&metrics_path, None, None).unwrap();

        assert_eq!(
            points
                .iter()
                .map(|point| point.iteration_id)
                .collect::<Vec<_>>(),
            vec![1200, 1201]
        );
    }

    #[test]
    fn speculative_metric_versions_preserve_request_count_and_resident_kv() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        for version in [3, 4] {
            let row = json!({
                "schema_version": version, "input_adapter": "vllm_text",
                "iteration_index": 7, "prefill_tokens": 0,
                "decode_requests": 2, "decode_tokens_scheduled": 12,
                "decode_kv_lens": [100, 200], "decode_query_lens": [6, 6],
                "prefill_chunk_pairs": [],
                "observed_start_monotonic_ns": 100,
                "observed_end_monotonic_ns": 200,
                "observed_elapsed_ms": 0.0001
            });
            fs::write(&path, format!("{row}\n")).unwrap();
            let (points, _) = read_measured_points(&path, None, None).unwrap();
            assert_eq!(points[0].decode_batch_size, 2);
            assert_eq!(points[0].scheduled_kv_tokens, 300);
        }
    }

    #[test]
    fn replay_window_selects_full_run_without_nsys() {
        let directory = tempfile::tempdir().unwrap();
        let metrics_path = directory.path().join("metrics.jsonl");
        let record = |iteration_index: u64, start_ns: u64| {
            format!(
                "{{\"schema_version\":2,\"input_adapter\":\"vllm_text\",\"iteration_index\":{iteration_index},\"prefill_tokens\":8,\"decode_kv_lens\":[],\"prefill_chunk_pairs\":[[0,8]],\"observed_start_monotonic_ns\":{start_ns},\"observed_end_monotonic_ns\":{},\"observed_elapsed_ms\":0.0001}}\n",
                start_ns + 100
            )
        };
        fs::write(
            &metrics_path,
            [
                record(1200, 100),
                record(1201, 300),
                record(1300, 500),
                record(1301, 700),
            ]
            .concat(),
        )
        .unwrap();

        let (points, _) = read_measured_points(&metrics_path, Some(450), Some(850)).unwrap();

        assert_eq!(
            points
                .iter()
                .map(|point| point.iteration_id)
                .collect::<Vec<_>>(),
            vec![1300, 1301]
        );
    }

    #[test]
    fn a_single_rank_capture_passes_through_unchanged() {
        let (folded, fold) = fold_data_parallel_ranks(vec![
            rank_record(7, 4, vec![10], None),
            rank_record(8, 0, vec![11, 12], None),
        ])
        .unwrap();

        assert_eq!(
            folded
                .iter()
                .map(|record| (record.iteration_index, record.prefill_tokens))
                .collect::<Vec<_>>(),
            vec![(Some(7), 4), (Some(8), 0)]
        );
        assert_eq!(fold.method, "single_rank");
    }

    #[test]
    fn a_single_rank_sorts_by_start_and_drops_its_dummies() {
        let (folded, fold) = fold_data_parallel_ranks(vec![
            forward(0, 1, 8, 2, vec![], 0),
            dummy(0, 2, 0),
            forward(0, 0, 7, 1, vec![], 0),
        ])
        .unwrap();

        assert_eq!(
            folded
                .iter()
                .map(|record| record.iteration_index)
                .collect::<Vec<_>>(),
            vec![Some(7), Some(8)]
        );
        assert_eq!(fold.method, "single_rank");
        assert_eq!(fold.dummy_records, 1);
    }

    #[test]
    fn data_parallel_ranks_pair_by_forward_ordinal_with_dummies() {
        // Rank 0 counts its own scheduled steps from 11, rank 1 from 3; the
        // ordinal, not the index, names the global step. Rank 1 is idle at
        // forward 1 and runs a dummy, which joins the step with no work.
        let records = vec![
            forward(0, 0, 11, 4, vec![10], 0),
            forward(0, 1, 12, 0, vec![10, 20], 0),
            forward(0, 2, 13, 0, vec![30], 0),
            forward(1, 0, 3, 6, vec![11], 2),
            dummy(1, 1, 2),
            forward(1, 2, 4, 0, vec![40], 2),
        ];

        let (folded, fold) = fold_data_parallel_ranks(records).unwrap();

        assert_eq!(fold.method, "forward_ordinal");
        assert_eq!(fold.ranks, 2);
        assert_eq!(fold.dummy_records, 1);
        assert_eq!(fold.all_dummy_steps, 0);
        assert_eq!(fold.non_overlapping_steps, 0);
        assert_eq!(
            folded
                .iter()
                .map(|step| (
                    step.iteration_index,
                    step.prefill_tokens,
                    step.decode_kv_lens.clone()
                ))
                .collect::<Vec<_>>(),
            vec![
                (Some(0), 10, vec![10, 11]),
                (Some(1), 0, vec![10, 20]),
                (Some(2), 0, vec![30, 40]),
            ]
        );
        // The step's window is the union of its real records' windows.
        assert_eq!(folded[0].observed_start_monotonic_ns, Some(0));
        assert_eq!(folded[0].observed_end_monotonic_ns, Some(10));
        assert_eq!(folded[1].observed_end_monotonic_ns, Some(18));
        let skew = fold.start_skew_ms.unwrap();
        assert_eq!(skew["max"], json!(2e-6));
    }

    #[test]
    fn a_step_every_rank_ran_as_a_dummy_is_not_a_workload_step() {
        let (folded, fold) = fold_data_parallel_ranks(vec![
            forward(0, 0, 1, 1, vec![], 0),
            forward(1, 0, 1, 1, vec![], 0),
            dummy(0, 1, 0),
            dummy(1, 1, 0),
            forward(0, 2, 2, 1, vec![], 0),
            forward(1, 2, 2, 1, vec![], 0),
        ])
        .unwrap();

        assert_eq!(
            folded
                .iter()
                .map(|step| step.iteration_index)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(2)]
        );
        assert_eq!(fold.all_dummy_steps, 1);
        assert_eq!(fold.dummy_records, 2);
    }

    #[test]
    fn ragged_ends_are_dropped_with_a_count() {
        // Rank 1 starts recording at forward 1 and rank 0 stops at forward 2.
        let (folded, fold) = fold_data_parallel_ranks(vec![
            forward(0, 0, 0, 1, vec![], 0),
            forward(0, 1, 1, 1, vec![], 0),
            forward(0, 2, 2, 1, vec![], 0),
            forward(1, 1, 0, 1, vec![], 0),
            forward(1, 2, 1, 1, vec![], 0),
            forward(1, 3, 2, 1, vec![], 0),
            forward(1, 4, 3, 1, vec![], 0),
        ])
        .unwrap();

        assert_eq!(folded.len(), 2);
        assert_eq!(fold.dropped_before, 1);
        assert_eq!(fold.dropped_after, 2);
    }

    #[test]
    fn a_missing_ordinal_inside_the_common_range_is_refused() {
        let error = fold_data_parallel_ranks(vec![
            forward(0, 0, 0, 1, vec![], 0),
            forward(0, 1, 1, 1, vec![], 0),
            forward(0, 2, 2, 1, vec![], 0),
            forward(1, 0, 0, 1, vec![], 0),
            forward(1, 2, 1, 1, vec![], 0),
        ])
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("dp_rank 1 records 2 of the 3 forwards 0..=2")
                && error.contains("first missing: 1"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_rank_that_records_one_forward_twice_is_refused() {
        let error = fold_data_parallel_ranks(vec![
            forward(0, 0, 0, 1, vec![], 0),
            forward(1, 0, 0, 1, vec![], 0),
            dummy(1, 0, 0),
        ])
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("records forward 0 twice"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_multi_rank_capture_without_forward_ordinals_is_refused() {
        let legacy = |rank, index| FullMeasuredMetrics {
            dp_rank: Some(rank),
            ..rank_record(index, 1, vec![], Some((10 * index, 10 * index + 8, 8e-6)))
        };
        let error = fold_data_parallel_ranks(vec![legacy(0, 0), legacy(1, 0)])
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("predates forward ordinals") && error.contains("recapture"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn records_that_repeat_an_index_without_a_rank_are_refused() {
        // An unlabeled multi-rank capture used to be folded by iteration_index,
        // which merged different global steps.
        let error = fold_data_parallel_ranks(vec![
            rank_record(7, 4, vec![10], None),
            rank_record(7, 6, vec![11], None),
        ])
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("predates forward ordinals"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn ordinals_that_contradict_the_wall_clock_are_refused() {
        // Rank 1's forward 2 ends before rank 0 began forward 0's successor
        // window by more than the two forwards a rank can have in flight: the
        // ordinals cannot name the same global step.
        let error = fold_data_parallel_ranks(vec![
            forward(0, 0, 0, 1, vec![], 1000),
            forward(0, 1, 1, 1, vec![], 1000),
            forward(0, 2, 2, 1, vec![], 1000),
            forward(1, 0, 0, 1, vec![], 0),
            forward(1, 1, 1, 1, vec![], 0),
            forward(1, 2, 2, 1, vec![], 0),
        ])
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("do not name the same global step"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn the_wall_clock_check_reads_the_capture_in_flight_depth() {
        // Rank 1 runs two forwards ahead: legal with three forwards in
        // flight, impossible with two.
        let records = |depth| {
            (0..4)
                .flat_map(|ordinal| {
                    [(0, 30), (1, 0)].map(|(rank, offset)| FullMeasuredMetrics {
                        max_forwards_in_flight: Some(depth),
                        ..forward(rank, ordinal, ordinal, 1, vec![], offset)
                    })
                })
                .collect::<Vec<_>>()
        };

        assert!(fold_data_parallel_ranks(records(3)).is_ok());
        let error = fold_data_parallel_ranks(records(2))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("do not name the same global step"),
            "unexpected error: {error}"
        );
        let missing: Vec<_> = records(2)
            .into_iter()
            .map(|record| FullMeasuredMetrics {
                max_forwards_in_flight: None,
                ..record
            })
            .collect();
        let error = fold_data_parallel_ranks(missing).unwrap_err().to_string();
        assert!(
            error.contains("carries no max_forwards_in_flight"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_skewed_but_possible_step_is_counted_not_refused() {
        // Rank 1 runs a full forward behind rank 0 (async scheduling offsets
        // the observation windows): no overlap, still a legal pairing.
        let (folded, fold) = fold_data_parallel_ranks(vec![
            forward(0, 0, 0, 1, vec![], 0),
            forward(0, 1, 1, 1, vec![], 0),
            forward(0, 2, 2, 1, vec![], 0),
            forward(1, 0, 0, 1, vec![], 9),
            forward(1, 1, 1, 1, vec![], 9),
            forward(1, 2, 2, 1, vec![], 9),
        ])
        .unwrap();

        assert_eq!(folded.len(), 3);
        assert_eq!(fold.non_overlapping_steps, 3);
    }

    #[test]
    fn mixed_schema_versions_within_one_step_are_rejected() {
        let error = fold_data_parallel_ranks(vec![
            forward(0, 0, 0, 1, vec![], 0),
            FullMeasuredMetrics {
                schema_version: 6,
                ..forward(1, 0, 0, 1, vec![], 0)
            },
        ])
        .unwrap_err();

        assert!(
            error.to_string().contains("mixes full-run metrics schema"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn schema_five_metrics_read_through_the_ordinal_fold() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        let row = |rank: u64, ordinal: u64, dummy: bool| {
            let start = 100 * (ordinal + 1);
            json!({
                "schema_version": 5, "input_adapter": "vllm_text",
                "iteration_index": if dummy { Value::Null } else { json!(ordinal + 40 * rank) },
                "forward_ordinal": ordinal, "is_dummy": dummy, "dp_rank": rank,
                "max_forwards_in_flight": 2,
                "prefill_tokens": if dummy { 0 } else { 4 },
                "decode_kv_lens": [], "prefill_chunk_pairs": [],
                "decode_request_progress": [],
                "observed_start_monotonic_ns": start,
                "observed_end_monotonic_ns": start + 50,
                "observed_elapsed_ms": 5e-5
            })
            .to_string()
        };
        let lines = [
            row(0, 0, false),
            row(1, 0, false),
            row(0, 1, false),
            row(1, 1, true),
        ];
        fs::write(&path, lines.join("\n") + "\n").unwrap();

        let (points, fold) = read_measured_points(&path, None, None).unwrap();

        assert_eq!(
            points
                .iter()
                .map(|point| (point.iteration_id, point.prefill_tokens))
                .collect::<Vec<_>>(),
            vec![(0, 8), (1, 4)]
        );
        assert_eq!(fold.method, "forward_ordinal");
        assert_eq!(fold.dummy_records, 1);
    }
}
