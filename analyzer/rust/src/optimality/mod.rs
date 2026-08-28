//! `optimality` subject — how far a run is from the optimal use of its GPUs.
//!
//! Optimal = the minimal GPU·seconds to complete the same work with **no idle**,
//! **perfect load balancing**, and **kernels at their best batching / rate**. Not
//! one optimal but a **ladder of increasingly-idealized lower bounds**, so each
//! successive gap is an attributable source of sub-optimality. All values are in
//! unit **GPU·seconds** (= wall time × the worker's physical GPU count), computed
//! by re-folding each worker's per-row CostTree manifest with a different leaf-rate
//! / structure substitution per rung:
//!
//! | Rung | leaf value / structure          | gap vs previous = cause         |
//! |------|---------------------------------|---------------------------------|
//! | R0 Real            | `span × G`         | — (GPU·s actually held)         |
//! | R1 Busy            | `Σ total_time × G` | **idle** (scheduler gaps)       |
//! | R2 Balanced        | real times, Max→mean | **imbalance** (DP/EP straggler)|
//! | R3 per-config best | work / grid-peak rate | **batching** (small-batch loss)|
//! | R4 ignore network  | R3, comm leaves→0  | **communication**               |
//! | R5 hardware limit  | active work unit / matching spec peak | **profiled↔hardware** |
//! | R6 segmented necessary | Σ per-location semantic rooflines | **redundant work**    |
//! | R7 scope-fused necessary | one roofline over this scope's semantic work | **fusion** |
//!
//! Buckets telescope and sum exactly back to Real, so the output is an additive
//! stacked **waterfall** rendered at five levels (cluster / pool / worker /
//! iteration — idle 0 by construction / per-kernel). In both modes, the
//! `model.work` labeler adds segmented and scope-fused bounds below R5 and splits
//! `hw-optimal` into `[excess-over-necessary | fusion | hardware-necessary]`.
//! Unlocked run aggregation may fuse additive work into a saturated batch. Locked
//! aggregation evaluates each sampled iteration's rooflines first, then adds them
//! through worker / pool / cluster without rebatching. An exact iteration waterfall
//! uses the same distinction: locked labels the exact batch, while unlocked labels
//! 10,000 independent copies and normalizes back to one iteration. Exact and run kernel ladders append
//! mapped segmented R6 plus aggregate-only scope-fused R7 when a strict semantic-location
//! map covers the manifest. Run pool/cluster ladders are summed and reconciled by the
//! analyzer rather than reconstructed by the UI.
//!
//! Cost model. `G_worker` is read from `run_meta` (`workers[].gpu_ids.len()`),
//! never inferred from tp×dp×ep. R0/R1 are exact SQL sums over every row; R2..R5
//! fold a **stride-sampled** set of rows (a location's rates are near-constant
//! across iterations) and are anchored to the exact R1 by their sampled ratio, so
//! the ladder stays monotone and the buckets stay exact. The mean-mode fold is
//! linear, so each rung factors into a precomputed per-leaf weight `α` times that
//! leaf's value — one dot product per row, and the same `α` gives the additive
//! per-kernel attribution for the kernel-level bars.
//!
//! The locked R6/R7 floors are **stratified**, not exact. They used to label every
//! iteration, which only looked affordable because equal shapes deduplicate — and
//! they barely do: `decode_kv` is a running sum, so the 8h GLM-5.2 trace had 911,149
//! distinct shapes across 978,623 iterations, one label per iteration. Every
//! prefill-carrying iteration is still labeled (0.76% of iterations, 70.7% of the
//! matmul-token mass, and all of the variance); the decode-only remainder is sampled
//! to `FLOORS_TARGET_SAMPLED_ITERS` per worker and reweighted in integers so each
//! worker's weights still sum to its exact iteration count. See
//! `conservation::workload::collect_workload_shapes_by_worker`.
//!
//! ## Files
//!
//! - `run.rs` — orchestration: `run_optimality` wires the stages and assembles the
//!   report/payload JSON (+ the `unavailable` degrade paths).
//! - `prepare.rs` — preparation: interns manifest leaves into locations and
//!   precomputes each section's fold weights + rate ceilings (`build_section_fold_plans`).
//! - `fold.rs` — the algorithm: exact R0/R1 SQL sums + the stride-sampled R2..R5
//!   mean-fold (`read_exact_worker_totals`, `accumulate_fold`,
//!   `leaf_selected_throughput_ms`).
//! - `levels.rs` — the worker / pool / cluster tiers: rung assembly + rollup +
//!   level/report JSON.
//! - `kernel.rs` — the kernel tier: per-location bars, per-worker kernel ladders,
//!   and analyzer-owned pool/cluster ladder rollups.
//! - `grid_peaks.rs` — the R3 grid-peak ceiling sidecar.
//! - `spec.rs` — the R5 GPU-spec compute/bandwidth ceilings.

mod floors;
mod fold;
mod grid_peaks;
mod iteration;
mod kernel;
mod ladder;
mod levels;
mod location;
mod prepare;
mod run;
mod scoped;
mod spec;

pub(crate) use iteration::{
    iteration_kernel_ladder, iteration_waterfall, prediction_kernel_ladder, prediction_waterfall,
};
pub use run::run_optimality;
pub(crate) use scoped::run_scoped;

/// Waterfall segment order (top of the Real bar → the irreducible floor).
pub(crate) const BUCKET_KEYS: [&str; 6] = [
    "idle",
    "imbalance",
    "batching",
    "communication",
    "hardware_gap",
    "hardware_optimal",
];
pub(crate) const RUNG_KEYS: [&str; 6] = [
    "real",
    "busy",
    "balanced",
    "per_config_best",
    "ignore_network",
    "hardware_limit",
];

/// In unlocked mode, when the labeler floors are available, the single green
/// `hardware_optimal` bucket
/// (= R5) is split into these three sub-buckets, listed top-of-bar → floor (so they
/// slot in right after `hardware_gap`, replacing `hardware_optimal`). They telescope:
/// `excess_over_necessary = R5 − segmented`, `fusion = segmented − necessary`,
/// `hardware_necessary = necessary`; their sum is exactly R5, so the bar is unchanged.
pub(crate) const FLOOR_BUCKET_KEYS: [&str; 3] =
    ["excess_over_necessary", "fusion", "hardware_necessary"];
/// The two extra rungs the floors add below R5 (`hardware_limit`), for the report.
pub(crate) const FLOOR_RUNG_KEYS: [&str; 2] = ["segmented_necessary", "hardware_necessary"];

/// Waterfall bucket keys in top-down order, with the green split into the three floor
/// sub-buckets when the labeler floors are present (else the plain 6).
pub(crate) fn bucket_keys(with_floors: bool) -> Vec<&'static str> {
    if with_floors {
        BUCKET_KEYS[..5]
            .iter()
            .copied()
            .chain(FLOOR_BUCKET_KEYS)
            .collect()
    } else {
        BUCKET_KEYS.to_vec()
    }
}

/// Rung keys with the two floor rungs appended below `hardware_limit` when present.
pub(crate) fn rung_keys(with_floors: bool) -> Vec<&'static str> {
    if with_floors {
        RUNG_KEYS.iter().copied().chain(FLOOR_RUNG_KEYS).collect()
    } else {
        RUNG_KEYS.to_vec()
    }
}
/// The four attributable rungs a kernel carries (R2..R5); no idle/imbalance.
pub(crate) const KERNEL_RUNG_KEYS: [&str; 4] = [
    "balanced",
    "per_config_best",
    "ignore_network",
    "hardware_limit",
];

/// Iteration sampling bounds for the R2..R5 fold (mirrors `kernel-throughput`).
pub(crate) const MAX_STRIDE: u64 = 50;
pub(crate) const TARGET_SAMPLED_ITERS: u64 = 80;

/// How many **decode-only** iterations the locked R6/R7 floors draw per worker.
/// Deliberately its own target, three orders of magnitude above
/// `TARGET_SAMPLED_ITERS`: that one sizes a sample of per-location *rates*, which are
/// near-constant across a run, whereas a workload *shape* drifts as context grows.
/// Prefill-carrying iterations are not sampled at all (see
/// `collect_workload_shapes_by_worker`); the labeler costs roughly 0.06 ms per
/// distinct shape, so 50,000 keeps this stage at a few seconds against the ~49 s the
/// rest of the subject takes. No upper clamp on the resulting stride: a longer run
/// samples harder rather than labeling more.
pub(crate) const FLOORS_TARGET_SAMPLED_ITERS: u64 = 50_000;

/// Top-N kernels drawn as individual bars at the kernel level; the rest fold into
/// an `other` bar so the figure stays legible on a many-location deployment.
pub(crate) const TOP_KERNELS: usize = 16;

/// Large enough that one-time model weights no longer determine the replicated
/// workload's bound, without materializing any requests (the labeler receives only
/// scaled additive totals).
pub(crate) const UNLOCKED_ITERATION_REPLICATION_FACTOR: u32 = 10_000;

/// R5 and R6 read the same stride sample but reduce it differently (α-weighted leaf
/// fold vs per-shape roofline). Preserve their raw difference, but do not classify
/// sampling-scale noise as missing simulator work.
pub(crate) const UNDER_ACCOUNTED_RELATIVE_TOLERANCE: f64 = 0.005;

pub(crate) fn under_accounted_difference(
    hardware_limit_gpu_s: f64,
    necessary_limit_gpu_s: f64,
) -> (f64, f64, f64) {
    let raw_gpu_s = (necessary_limit_gpu_s - hardware_limit_gpu_s).max(0.0);
    let tolerance_gpu_s = necessary_limit_gpu_s.max(0.0) * UNDER_ACCOUNTED_RELATIVE_TOLERANCE;
    let material_gpu_s = if raw_gpu_s > tolerance_gpu_s {
        raw_gpu_s
    } else {
        0.0
    };
    (material_gpu_s, raw_gpu_s, tolerance_gpu_s)
}

pub(crate) fn ms_to_s(ms: f64) -> f64 {
    ms / 1000.0
}

pub(crate) fn ratio(num: f64, den: f64) -> f64 {
    if den > 0.0 {
        (num / den).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod speculative_tests;

#[cfg(test)]
mod tests {
    use super::under_accounted_difference;

    #[test]
    fn under_accounted_requires_more_than_half_a_percent() {
        let (material, raw, tolerance) = under_accounted_difference(0.999, 1.0);
        assert_eq!(material, 0.0);
        assert!((raw - 0.001).abs() < 1e-12);
        assert_eq!(tolerance, 0.005);

        let (material, raw, _) = under_accounted_difference(0.994, 1.0);
        assert_eq!(material, raw);
        assert!((material - 0.006).abs() < 1e-12);
    }
}
