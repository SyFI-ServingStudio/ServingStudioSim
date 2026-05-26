//! Pool + group topology config (L6) — new-interface-design §6.1 / §8.
//!
//! A pool is always `{ placement, groups: [...] }` — never inlined. A homogeneous
//! pool has a `groups` list of length 1; heterogeneous = several groups on
//! different GPUs/arch (parse-only this round — `build()` accepts a single
//! group). The `Arch` / `Worker` type parameters ARE the contract constraint: a
//! layer-wise arch cannot be paired with an iter-wise worker. They are generic
//! here, so this layer holds the topology shape without depending on the concrete
//! L4/L5 selector types (the per-deployment configs in L7 instantiate them).
//!
//! `placement` is the only pool-level policy (L6 routing among replicas);
//! everything provider-specific lives on the arch/worker tags. The launcher
//! schema is derived: `#[derive(ParamStruct)]` on [`GroupSpec`] emits the flat
//! group fields (gpu / replicas) and on [`PoolSpec`] the flat pool fields
//! (placement); `groups` / `arch` / `worker` are `#[param(skip)]` (they are
//! nested sub-trees, not scalar params). The params do not touch `Arch` /
//! `Worker`, so `dump` reads them off a `<(), ()>` instantiation.

use serde::Deserialize;

use schema_derive::ParamStruct;

/// L6 per-pool worker selection policy. Closed set → serde enum (kebab-case so
/// `least-queued` / `round-robin` match the wire spelling).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PlacementPolicy {
    LeastQueued,
    RoundRobin,
}

const PLACEMENT_CHOICES: [&str; 2] = ["least-queued", "round-robin"];

/// One pool: a placement policy plus one-or-more homogeneous groups.
#[derive(Debug, Clone, Deserialize, ParamStruct)]
#[serde(deny_unknown_fields)]
pub struct PoolSpec<Arch, Worker> {
    /// Worker placement policy within the pool.
    #[param(string, default = "least-queued", choices = PLACEMENT_CHOICES)]
    pub placement: PlacementPolicy,
    #[param(skip)]
    pub groups: Vec<GroupSpec<Arch, Worker>>,
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
