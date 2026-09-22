//! Pool + group topology config (L6).
//!
//! A pool is always `{ placement, groups: [...] }` — never inlined. A homogeneous
//! pool has a `groups` list of length 1; heterogeneous = several groups on
//! different GPUs/arch (parse-only this round — `build()` accepts a single
//! group). The `Arch` / `Worker` type parameters ARE the contract constraint: a
//! layer-wise arch cannot be paired with an iter-wise worker. They are generic
//! here, so this layer holds the topology shape without depending on the concrete
//! L4/L5 selector types (the per-deployment configs in L7 instantiate them).
//!
//! Pool-level policy is L6 routing among replicas — `placement` (where a fresh
//! arrival goes) and `migration` (when resident work moves between replicas);
//! everything provider-specific lives on the arch/worker tags. The launcher
//! schema is derived: `#[derive(ParamStruct)]` on [`GroupSpec`] emits the flat
//! group fields (gpu / replicas) and on [`PoolSpec`] the flat pool fields
//! (placement + the migration trio); `groups` / `arch` / `worker` are
//! `#[param(skip)]` (they are nested sub-trees, not scalar params). The params
//! do not touch `Arch` / `Worker`, so `dump` reads them off a `<(), ()>`
//! instantiation.

use serde::Deserialize;

use schema_derive::ParamStruct;

/// L6 per-pool worker selection policy. Closed set → serde enum (kebab-case so
/// `least-queued` / `round-robin` match the wire spelling).
///
/// `trace-directed` is not a load heuristic: it obeys the `target_worker`
/// column of the trace's `placement` tag, which is how a run reproduces an
/// exact placement sequence instead of whatever a load metric happened to pick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PlacementPolicy {
    LeastQueued,
    RoundRobin,
    TraceDirected,
}

const PLACEMENT_CHOICES: [&str; 3] = ["least-queued", "round-robin", "trace-directed"];

/// When a pool moves resident work between its own workers.
///
/// `off` is the default and costs nothing: the flow holds no policy at all, so
/// an untouched preset's tick does not gain a load snapshot or a virtual call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MigrationPolicySel {
    #[default]
    Off,
    /// Drain a worker whose active batch has fallen below
    /// `migration_threshold` onto the busiest remaining worker.
    ActiveBatchBelow,
    /// Hand a whole block of `migration_workers_per_train_group` workers back at
    /// once, when the samples still in flight across it fall below
    /// `migration_threshold`, scattering its prompt groups over the workers of
    /// the other blocks.
    TrainGroupSamplesBelow,
}

const MIGRATION_CHOICES: [&str; 3] = ["off", "active-batch-below", "train-group-samples-below"];

/// One pool: a placement policy plus one-or-more homogeneous groups.
#[derive(Debug, Clone, Deserialize, ParamStruct)]
#[serde(deny_unknown_fields)]
pub struct PoolSpec<Arch, Worker> {
    /// Worker placement policy within the pool.
    #[param(string, default = "least-queued", choices = PLACEMENT_CHOICES)]
    pub placement: PlacementPolicy,
    /// When this pool migrates resident work between its own workers.
    #[serde(default)]
    #[param(string, default = "off", choices = MIGRATION_CHOICES)]
    pub migration: MigrationPolicySel,
    /// The count under which a migration fires. `active-batch-below` reads it
    /// as one worker's active batch; `train-group-samples-below` reads it as the
    /// samples still in flight across a whole block of workers.
    #[serde(default = "default_migration_threshold")]
    #[param(default = 32)]
    pub migration_threshold: u32,
    /// Requests per prompt group, numbered in consecutive id blocks. One means
    /// "no grouping". Read only by `train-group-samples-below`, which counts a
    /// group's full size until its slowest member lands.
    #[serde(default = "default_migration_group_size")]
    #[param(default = 1)]
    pub migration_group_size: u32,
    /// Workers per train group, blocked by worker id. Read only by
    /// `train-group-samples-below`, which releases a whole block at a time
    /// because a block is useful to training only once all of it is free.
    #[serde(default = "default_migration_workers_per_train_group")]
    #[param(default = 1)]
    pub migration_workers_per_train_group: u16,
    /// Minimum sim time between two migrations of this pool. Zero lets the
    /// policy fire on consecutive ticks, which is only sane for a trigger that
    /// cannot re-arm — the built-in one retires its source, so it cannot.
    #[serde(default)]
    #[param(default = 0.0)]
    pub migration_cooldown_ms: f64,
    /// What handing back one prompt group costs in wall clock. Read only by
    /// `train-group-samples-below`. Zero, the default, releases a block in one
    /// step; anything positive makes it hand its groups over one at a time,
    /// `migration_group_latency_ms` apart, decoding what it still holds the
    /// whole way out and retiring only once the last one has left. A real
    /// router aborts and re-dispatches one group at a time, so a block holding
    /// fifteen of them takes ten seconds to go.
    #[serde(default)]
    #[param(default = 0.0)]
    pub migration_group_latency_ms: f64,
    #[param(skip)]
    pub groups: Vec<GroupSpec<Arch, Worker>>,
}

/// Mirrors the launcher schema default so a hand-written preset that omits the
/// field gets the same number the launcher would have filled in.
const fn default_migration_threshold() -> u32 {
    32
}

const fn default_migration_group_size() -> u32 {
    1
}

const fn default_migration_workers_per_train_group() -> u16 {
    1
}

/// One homogeneous group: a GPU type, a replica count (= DP fan-out), and the
/// arch (L4) + worker (L5) providers — both tagged enums constrained to the
/// pool's contract class.
#[derive(Debug, Clone, Deserialize, ParamStruct)]
#[serde(deny_unknown_fields)]
pub struct GroupSpec<Arch, Worker> {
    /// GPU type this group runs on (profile.db key).
    pub gpu: String,
    /// Data-parallel replica count for this group.
    #[param(default = 1)]
    pub replicas: u16,
    #[param(skip)]
    pub arch: Arch,
    #[param(skip)]
    pub worker: Worker,
}
