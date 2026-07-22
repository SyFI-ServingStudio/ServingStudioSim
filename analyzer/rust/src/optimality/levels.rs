//! Worker / pool / cluster tiers — turn the fold's per-worker accumulators into
//! the named R0..R5 rung values, then roll them up. Pool and cluster are sums of
//! the worker tier, so they live here alongside it rather than in a third file.
//! Also owns the level → payload JSON (`level_entry_json`) and the report-side rung
//! object (`rung_report_json`).

use std::collections::BTreeMap;

use serde_json::{json, Value};

use super::floors::{Floors, FloorsByLevel};
use super::fold::WorkerFoldAccumulator;
use super::{ms_to_s, ratio};

/// Named R0..R5 GPU·ms values. This is the additive hierarchy unit; keeping
/// fields named prevents the rung-order assumptions that a `[f64; 6]` leaks to
/// every consumer.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BaseRungs {
    pub(super) real: f64,
    pub(super) busy: f64,
    pub(super) balanced: f64,
    pub(super) per_config_best: f64,
    pub(super) ignore_network: f64,
    pub(super) hardware_limit: f64,
}

impl BaseRungs {
    fn add_assign(&mut self, other: Self) {
        self.real += other.real;
        self.busy += other.busy;
        self.balanced += other.balanced;
        self.per_config_best += other.per_config_best;
        self.ignore_network += other.ignore_network;
        self.hardware_limit += other.hardware_limit;
    }

    fn enforce_monotonicity(&mut self) {
        self.busy = self.busy.min(self.real).max(0.0);
        self.balanced = self.balanced.min(self.busy).max(0.0);
        self.per_config_best = self.per_config_best.min(self.balanced).max(0.0);
        self.ignore_network = self.ignore_network.min(self.per_config_best).max(0.0);
        self.hardware_limit = self.hardware_limit.min(self.ignore_network).max(0.0);
    }
}

/// The named rung values at every non-kernel tier, plus the per-worker anchor factors the
/// kernel tier needs. `worker_anchor_factors` and `worker_rungs_in_fold_order` use
/// the original worker order (matching the fold's `(location, worker)` cells);
/// `sorted_worker_rungs` contains the same values re-keyed and sorted for stable output.
pub(super) struct TierAggregates {
    pub(super) cluster_rungs: BaseRungs,
    pub(super) pool_rungs_by_tag: BTreeMap<String, BaseRungs>,
    pub(super) sorted_worker_rungs: Vec<(String, u16, BaseRungs)>,
    pub(super) worker_rungs_in_fold_order: Vec<BaseRungs>,
    pub(super) worker_anchor_factors: Vec<f64>,
}

// --- worker ---

/// Assemble each worker's named rungs (GPU·ms) from its exact + sampled accumulators,
/// then sum into pool and cluster. The anchor factor per worker (GPU·ms per sampled
/// fold-ms) folds in G and the exact/sampled busy upscale, so per-worker rungs and
/// per-kernel sums agree.
pub(super) fn assemble_tiers(workers: &[WorkerFoldAccumulator]) -> TierAggregates {
    let mut cluster_rungs = BaseRungs::default();
    let mut pool_rungs_by_tag: BTreeMap<String, BaseRungs> = BTreeMap::new();
    let mut sorted_worker_rungs: Vec<(String, u16, BaseRungs)> = Vec::new();
    let mut worker_rungs_in_fold_order: Vec<BaseRungs> = Vec::with_capacity(workers.len());
    let mut worker_anchor_factors: Vec<f64> = vec![0.0; workers.len()];
    for (worker_index, worker) in workers.iter().enumerate() {
        let mut rungs = BaseRungs {
            real: worker.span_ms * worker.gpu_count,
            busy: worker.busy_ms * worker.gpu_count,
            ..BaseRungs::default()
        };
        if worker.sampled_busy_ms > 0.0 {
            let anchor_factor = worker.busy_ms * worker.gpu_count / worker.sampled_busy_ms;
            worker_anchor_factors[worker_index] = anchor_factor;
            rungs.balanced = anchor_factor * worker.sampled_rungs_ms[0];
            rungs.per_config_best = anchor_factor * worker.sampled_rungs_ms[1];
            rungs.ignore_network = anchor_factor * worker.sampled_rungs_ms[2];
            rungs.hardware_limit = anchor_factor * worker.sampled_rungs_ms[3];
        } else {
            // No sampled rows for this worker: leave R2..R5 = R1 (all lower buckets
            // 0) rather than 0 (which would over-attribute to imbalance).
            rungs.balanced = rungs.busy;
            rungs.per_config_best = rungs.busy;
            rungs.ignore_network = rungs.busy;
            rungs.hardware_limit = rungs.busy;
        }
        // Enforce monotonicity defensively (sampling noise / clamp interplay).
        rungs.enforce_monotonicity();
        // --- pool + cluster ---
        cluster_rungs.add_assign(rungs);
        pool_rungs_by_tag
            .entry(worker.pool_tag.clone())
            .or_default()
            .add_assign(rungs);
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
/// (sorted), and the Busy-anchored iteration level (idle 0 by construction). A
/// level's green `hardware_optimal` splits only when that exact scope has
/// labeler floors (the synthetic iteration level reuses the cluster floor).
pub(super) fn levels_json(
    aggregates: &TierAggregates,
    floors: Option<&FloorsByLevel>,
) -> Vec<Value> {
    let floor_of = |key: &str| floors.and_then(|by_level| by_level.get(key)).copied();

    let mut levels: Vec<Value> = Vec::new();
    let cluster_floor = floor_of("cluster");
    levels.push(level_entry_json(
        "cluster",
        "cluster",
        "Cluster",
        &aggregates.cluster_rungs,
        false,
        cluster_floor,
    ));
    for (pool_tag, pool_rungs) in &aggregates.pool_rungs_by_tag {
        let pool_floor = floor_of(pool_tag);
        levels.push(level_entry_json(
            "pool",
            pool_tag,
            &format!("Pool {pool_tag}"),
            pool_rungs,
            false,
            pool_floor,
        ));
    }
    for (pool_tag, worker_id, worker_rungs) in &aggregates.sorted_worker_rungs {
        let worker_key = format!("{pool_tag}/{worker_id}");
        let worker_floor = floor_of(&worker_key);
        levels.push(level_entry_json(
            "worker",
            &worker_key,
            &worker_key,
            worker_rungs,
            false,
            worker_floor,
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
        cluster_floor,
    ));
    levels
}

/// Resolve clamped `(fused, segmented)` floors against R5. Enforces
/// `0 ≤ fused ≤ segmented ≤ green`, so all sub-buckets stay non-negative.
fn clamp_floor(floor: Floors, green_s: f64) -> (f64, f64) {
    let fused = floor.fused.clamp(0.0, green_s);
    let segmented = floor.segmented.clamp(fused, green_s);
    (fused, segmented)
}

/// Convert named rungs (GPU·ms) into one level's payload object: the waterfall buckets
/// (GPU·s) plus the rung values + optimality ratio. `idle_zero` tops the bar at Busy
/// (the iteration level) instead of Real. A present floor replaces the green
/// `hardware_optimal` bucket with three telescoping necessary-work buckets.
fn level_entry_json(
    level: &str,
    key: &str,
    label: &str,
    rungs: &BaseRungs,
    idle_zero: bool,
    floor: Option<Floors>,
) -> Value {
    let total = if idle_zero { rungs.busy } else { rungs.real };
    let idle = if idle_zero {
        0.0
    } else {
        rungs.real - rungs.busy
    };
    let green_s = ms_to_s(rungs.hardware_limit);

    // The five buckets above the green floor are the same telescoping rung diffs.
    let mut bucket_obj = serde_json::Map::new();
    for (bucket_key, bucket_gpu_ms) in [
        ("idle", idle),
        ("imbalance", rungs.busy - rungs.balanced),
        ("batching", rungs.balanced - rungs.per_config_best),
        (
            "communication",
            rungs.per_config_best - rungs.ignore_network,
        ),
        ("hardware_gap", rungs.ignore_network - rungs.hardware_limit),
    ] {
        bucket_obj.insert(
            bucket_key.to_string(),
            json!(ms_to_s(bucket_gpu_ms.max(0.0))),
        );
    }
    let mut rung_obj = json!({
        "real": ms_to_s(rungs.real),
        "busy": ms_to_s(rungs.busy),
        "balanced": ms_to_s(rungs.balanced),
        "per_config_best": ms_to_s(rungs.per_config_best),
        "ignore_network": ms_to_s(rungs.ignore_network),
        "hardware_limit": ms_to_s(rungs.hardware_limit),
    })
    .as_object()
    .expect("base rung serialization is an object")
    .clone();

    let necessary_ratio = if let Some(floor) = floor {
        let (fused, segmented) = clamp_floor(floor, green_s);
        bucket_obj.insert(
            "excess_over_necessary".into(),
            json!((green_s - segmented).max(0.0)),
        );
        bucket_obj.insert("fusion".into(), json!((segmented - fused).max(0.0)));
        bucket_obj.insert("hardware_necessary".into(), json!(fused));
        rung_obj.insert("segmented_necessary".into(), json!(segmented));
        rung_obj.insert("hardware_necessary".into(), json!(fused));
        Some(ratio(fused, ms_to_s(total)))
    } else {
        bucket_obj.insert("hardware_optimal".into(), json!(green_s));
        None
    };

    json!({
        "level": level,
        "key": key,
        "label": label,
        "total": ms_to_s(total),
        "buckets": bucket_obj,
        "rungs": rung_obj,
        "optimality_ratio": ratio(rungs.hardware_limit, total),
        "necessary_ratio": necessary_ratio,
    })
}

/// Exact iteration waterfall for one selected worker. Unlike the run-level
/// synthetic `iteration` row, this receives only that worker iteration's rungs.
/// Keeping this constructor here guarantees the on-demand endpoint uses the
/// same telescoping bucket contract as cluster, pool, and worker waterfalls.
pub(super) fn exact_iteration_level_json(
    key: &str,
    label: &str,
    rungs: &BaseRungs,
    floor: Option<Floors>,
) -> Value {
    level_entry_json("iteration", key, label, rungs, true, floor)
}

/// Report-side rung object for one tier: the rung values (GPU·s), the telescoping
/// buckets as `{gpu_s, frac}`, and the tier's optimality ratio. When `floor` is present
/// the two floor rungs are added and the `hardware_optimal` bucket splits into the
/// three floor sub-buckets (fractions are of Real, same as the others).
pub(super) fn rung_report_json(rungs: &BaseRungs, floor: Option<Floors>) -> Value {
    let real_gpu_ms = rungs.real.max(1e-9);
    let bucket_ms = |bucket_gpu_ms: f64| {
        json!({
            "gpu_s": ms_to_s(bucket_gpu_ms.max(0.0)),
            "frac": bucket_gpu_ms.max(0.0) / real_gpu_ms,
        })
    };
    // Floor sub-buckets arrive already in GPU·s (from the labeler); mirror the shape.
    let real_gpu_s = ms_to_s(real_gpu_ms);
    let bucket_s = |bucket_gpu_s: f64| json!({ "gpu_s": bucket_gpu_s.max(0.0), "frac": bucket_gpu_s.max(0.0) / real_gpu_s });

    let mut report = json!({
        "real": ms_to_s(rungs.real),
        "busy": ms_to_s(rungs.busy),
        "balanced": ms_to_s(rungs.balanced),
        "per_config_best": ms_to_s(rungs.per_config_best),
        "ignore_network": ms_to_s(rungs.ignore_network),
        "hardware_limit": ms_to_s(rungs.hardware_limit),
        "optimality_ratio": ratio(rungs.hardware_limit, rungs.real),
    });
    let mut buckets = json!({
        "idle": bucket_ms(rungs.real - rungs.busy),
        "imbalance": bucket_ms(rungs.busy - rungs.balanced),
        "batching": bucket_ms(rungs.balanced - rungs.per_config_best),
        "communication": bucket_ms(rungs.per_config_best - rungs.ignore_network),
        "hardware_gap": bucket_ms(rungs.ignore_network - rungs.hardware_limit),
    });
    let green_s = ms_to_s(rungs.hardware_limit);
    if let Some(floor) = floor {
        let (fused, segmented) = clamp_floor(floor, green_s);
        report["segmented_necessary"] = json!(segmented);
        report["hardware_necessary"] = json!(fused);
        report["necessary_ratio"] = json!(ratio(fused, real_gpu_s));
        buckets["excess_over_necessary"] = bucket_s(green_s - segmented);
        buckets["fusion"] = bucket_s(segmented - fused);
        buckets["hardware_necessary"] = bucket_s(fused);
    } else {
        buckets["hardware_optimal"] = bucket_ms(rungs.hardware_limit);
    }
    report["buckets"] = buckets;
    report
}

#[cfg(test)]
mod tests {
    use super::{rung_report_json, BaseRungs, Floors};

    #[test]
    fn necessary_work_split_preserves_green_and_uses_seconds_for_ratio() {
        let rungs = BaseRungs {
            real: 10_000.0,
            busy: 9_000.0,
            balanced: 8_000.0,
            per_config_best: 7_000.0,
            ignore_network: 6_000.0,
            hardware_limit: 5_000.0,
        };
        let report = rung_report_json(
            &rungs,
            Some(Floors {
                fused: 2.0,
                segmented: 3.0,
            }),
        );

        assert_eq!(report["necessary_ratio"], 0.2);
        assert_eq!(report["buckets"]["hardware_necessary"]["gpu_s"], 2.0);
        assert_eq!(report["buckets"]["fusion"]["gpu_s"], 1.0);
        assert_eq!(report["buckets"]["excess_over_necessary"]["gpu_s"], 2.0);
    }
}
