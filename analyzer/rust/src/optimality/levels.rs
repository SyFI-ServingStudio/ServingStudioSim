//! Worker / pool / cluster tiers — turn the fold's per-worker accumulators into
//! the R0..R5 rung arrays, then roll them up. Pool and cluster are trivial sums of
//! the worker tier, so they live here alongside it rather than in a third file.
//! Also owns the level → payload JSON (`level_entry_json`) and the report-side rung
//! object (`rung_report_json`).

use std::collections::BTreeMap;

use serde_json::{json, Value};

use super::fold::WorkerFoldAccumulator;
use super::{ms_to_s, ratio, BUCKET_KEYS, R0, R1, R2, R3, R4, R5, RUNG_KEYS};

/// The rung arrays at every non-kernel tier, plus the per-worker anchor factors the
/// kernel tier needs. `worker_anchor_factors` and `worker_rungs_in_fold_order` use
/// the original worker order (matching the fold's `(location, worker)` cells);
/// `sorted_worker_rungs` contains the same values re-keyed and sorted for stable output.
pub(super) struct TierAggregates {
    pub(super) cluster_rungs: [f64; 6],
    pub(super) pool_rungs_by_tag: BTreeMap<String, [f64; 6]>,
    pub(super) sorted_worker_rungs: Vec<(String, u16, [f64; 6])>,
    pub(super) worker_rungs_in_fold_order: Vec<[f64; 6]>,
    pub(super) worker_anchor_factors: Vec<f64>,
}

// --- worker ---

/// Assemble each worker's rung array (GPU·ms) from its exact + sampled accumulators,
/// then sum into pool and cluster. The anchor factor per worker (GPU·ms per sampled
/// fold-ms) folds in G and the exact/sampled busy upscale, so per-worker rungs and
/// per-kernel sums agree.
pub(super) fn assemble_tiers(workers: &[WorkerFoldAccumulator]) -> TierAggregates {
    let mut cluster_rungs = [0.0f64; 6];
    let mut pool_rungs_by_tag: BTreeMap<String, [f64; 6]> = BTreeMap::new();
    let mut sorted_worker_rungs: Vec<(String, u16, [f64; 6])> = Vec::new();
    let mut worker_rungs_in_fold_order: Vec<[f64; 6]> = Vec::with_capacity(workers.len());
    let mut worker_anchor_factors: Vec<f64> = vec![0.0; workers.len()];
    for (worker_index, worker) in workers.iter().enumerate() {
        let mut rungs = [0.0f64; 6];
        rungs[R0] = worker.span_ms * worker.gpu_count;
        rungs[R1] = worker.busy_ms * worker.gpu_count;
        if worker.sampled_busy_ms > 0.0 {
            let anchor_factor = worker.busy_ms * worker.gpu_count / worker.sampled_busy_ms;
            worker_anchor_factors[worker_index] = anchor_factor;
            rungs[R2] = anchor_factor * worker.sampled_rungs_ms[0];
            rungs[R3] = anchor_factor * worker.sampled_rungs_ms[1];
            rungs[R4] = anchor_factor * worker.sampled_rungs_ms[2];
            rungs[R5] = anchor_factor * worker.sampled_rungs_ms[3];
        } else {
            // No sampled rows for this worker: leave R2..R5 = R1 (all lower buckets
            // 0) rather than 0 (which would over-attribute to imbalance).
            rungs[R2] = rungs[R1];
            rungs[R3] = rungs[R1];
            rungs[R4] = rungs[R1];
            rungs[R5] = rungs[R1];
        }
        // Enforce monotonicity defensively (sampling noise / clamp interplay).
        for rung_index in 1..6 {
            rungs[rung_index] = rungs[rung_index].min(rungs[rung_index - 1]).max(0.0);
        }
        // --- pool + cluster ---
        for rung_index in 0..6 {
            cluster_rungs[rung_index] += rungs[rung_index];
            *pool_rungs_by_tag
                .entry(worker.pool_tag.clone())
                .or_insert([0.0; 6])
                .get_mut(rung_index)
                .unwrap() += rungs[rung_index];
        }
        worker_rungs_in_fold_order.push(rungs);
        sorted_worker_rungs.push((worker.pool_tag.clone(), worker.worker_id, rungs));
    }
    sorted_worker_rungs
        .sort_by(|left, right| (left.0.as_str(), left.1).cmp(&(right.0.as_str(), right.1)));
    TierAggregates {
        cluster_rungs,
        pool_rungs_by_tag,
        sorted_worker_rungs,
        worker_rungs_in_fold_order,
        worker_anchor_factors,
    }
}

/// Build the payload `levels` array: cluster, each pool (sorted), each worker
/// (sorted), and the Busy-anchored iteration level (idle 0 by construction).
pub(super) fn levels_json(aggregates: &TierAggregates) -> Vec<Value> {
    let mut levels: Vec<Value> = Vec::new();
    levels.push(level_entry_json(
        "cluster",
        "cluster",
        "Cluster",
        &aggregates.cluster_rungs,
        false,
    ));
    for (pool_tag, pool_rungs) in &aggregates.pool_rungs_by_tag {
        levels.push(level_entry_json(
            "pool",
            pool_tag,
            &format!("Pool {pool_tag}"),
            pool_rungs,
            false,
        ));
    }
    for (pool_tag, worker_id, worker_rungs) in &aggregates.sorted_worker_rungs {
        let worker_key = format!("{pool_tag}/{worker_id}");
        levels.push(level_entry_json(
            "worker",
            &worker_key,
            &worker_key,
            worker_rungs,
            false,
        ));
    }
    // Iteration level: the cluster waterfall with idle forced to 0 (idle is a
    // between-iteration gap, none within an iteration), so the bar tops at Busy.
    levels.push(level_entry_json(
        "iteration",
        "iteration",
        "Iteration",
        &aggregates.cluster_rungs,
        true,
    ));
    levels
}

/// Convert a rung array (GPU·ms) into one level's payload object: the six waterfall
/// buckets (GPU·s) plus the rung values + optimality ratio. `idle_zero` tops the
/// bar at Busy (the iteration level) instead of Real.
fn level_entry_json(
    level: &str,
    key: &str,
    label: &str,
    rungs: &[f64; 6],
    idle_zero: bool,
) -> Value {
    let total = if idle_zero { rungs[R1] } else { rungs[R0] };
    let idle = if idle_zero {
        0.0
    } else {
        rungs[R0] - rungs[R1]
    };
    let buckets = [
        idle,
        rungs[R1] - rungs[R2],
        rungs[R2] - rungs[R3],
        rungs[R3] - rungs[R4],
        rungs[R4] - rungs[R5],
        rungs[R5],
    ];
    let bucket_obj: serde_json::Map<String, Value> = BUCKET_KEYS
        .iter()
        .zip(buckets.iter())
        .map(|(bucket_key, bucket_gpu_ms)| {
            (
                bucket_key.to_string(),
                json!(ms_to_s(bucket_gpu_ms.max(0.0))),
            )
        })
        .collect();
    let rung_obj: serde_json::Map<String, Value> = RUNG_KEYS
        .iter()
        .zip(rungs.iter())
        .map(|(rung_key, rung_gpu_ms)| (rung_key.to_string(), json!(ms_to_s(*rung_gpu_ms))))
        .collect();
    json!({
        "level": level,
        "key": key,
        "label": label,
        "total": ms_to_s(total),
        "buckets": bucket_obj,
        "rungs": rung_obj,
        "optimality_ratio": ratio(rungs[R5], total),
    })
}

/// Report-side rung object for one tier: the six rung values (GPU·s), the telescoping
/// buckets as `{gpu_s, frac}`, and the tier's optimality ratio.
pub(super) fn rung_report_json(rungs: &[f64; 6]) -> Value {
    let real_gpu_ms = rungs[R0].max(1e-9);
    let bucket = |bucket_gpu_ms: f64| {
        json!({
            "gpu_s": ms_to_s(bucket_gpu_ms.max(0.0)),
            "frac": bucket_gpu_ms.max(0.0) / real_gpu_ms,
        })
    };
    json!({
        "real": ms_to_s(rungs[R0]),
        "busy": ms_to_s(rungs[R1]),
        "balanced": ms_to_s(rungs[R2]),
        "per_config_best": ms_to_s(rungs[R3]),
        "ignore_network": ms_to_s(rungs[R4]),
        "hardware_limit": ms_to_s(rungs[R5]),
        "optimality_ratio": ratio(rungs[R5], rungs[R0]),
        "buckets": {
            "idle": bucket(rungs[R0] - rungs[R1]),
            "imbalance": bucket(rungs[R1] - rungs[R2]),
            "batching": bucket(rungs[R2] - rungs[R3]),
            "communication": bucket(rungs[R3] - rungs[R4]),
            "hardware_gap": bucket(rungs[R4] - rungs[R5]),
            "hardware_optimal": bucket(rungs[R5]),
        },
    })
}
