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
//!
//! Buckets telescope and sum exactly back to Real, so the output is an additive
//! stacked **waterfall** rendered at five levels (cluster / pool / worker /
//! iteration — idle 0 by construction / per-kernel). In unlocked mode only, the
//! `model.work` labeler adds two global necessary-work bounds below R5 and splits
//! `hw-optimal` into `[excess-over-necessary | fusion | hardware-necessary]`.
//! Batch-locked analysis keeps the plain six-bucket R0..R5 ladder because its
//! current operating points must not be globally rebatchable.
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
//! - `kernel.rs` — the kernel tier: per-location bars + the per-worker kernel ladders.
//! - `grid_peaks.rs` — the R3 grid-peak ceiling sidecar.
//! - `spec.rs` — the R5 GPU-spec compute/bandwidth ceilings.

mod floors;
mod fold;
mod grid_peaks;
mod iteration;
mod kernel;
mod levels;
mod prepare;
mod run;
mod spec;

pub(crate) use iteration::iteration_kernel_ladder;
pub use run::run_optimality;

/// Rung index into the per-unit `[f64; 6]` GPU·ms accumulators (R0..R5).
pub(crate) const R0: usize = 0;
pub(crate) const R1: usize = 1;
pub(crate) const R2: usize = 2;
pub(crate) const R3: usize = 3;
pub(crate) const R4: usize = 4;
pub(crate) const R5: usize = 5;

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

/// Top-N kernels drawn as individual bars at the kernel level; the rest fold into
/// an `other` bar so the figure stays legible on a many-location deployment.
pub(crate) const TOP_KERNELS: usize = 16;

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
