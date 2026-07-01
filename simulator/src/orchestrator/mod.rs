//! `orchestrator` (L6) — pool-local orchestration (L6a) + inter-pool deployment
//! flow (L6b). See docs/detailed_design/L6/design.md.
//!
//! Module root holds only the `Flow` contract (the single surface L7 needs) and
//! the module wiring. Shared vocabulary + worker-stamping infra live in
//! `common`; concrete deployments live under `impls`.

pub mod common;
pub mod impls;
pub mod config;

pub use self::common::{GpuCluster, GpuInfo, OrchAction, UnifiedWorkerFactory};
pub use impls::{
    AfdAttnPoolController, AfdFfnPoolController, AfdFlow, DpPlacementPolicy, PdFlow, SimpleDpConfig,
    SimpleDpFlow, SimpleDpPoolConfig, SimpleDpPoolController, AFD_ATTN_POOL, AFD_FFN_POOL,
    PD_DECODE_POOL, PD_PREFILL_POOL,
};
pub use config::{GroupSpec, PlacementPolicy, PoolSpec};

use crate::common::{Request, Time};
use crate::worker::SharedGpuCluster;

/// L6b deployment flow — the only object L7 calls. `on_arrival` takes the full
/// `Request` (the flow inserts its facts into the shared store, then admits the
/// id); `tick` drives the pools and surfaces deployment actions.
pub trait Flow {
    fn on_arrival(&mut self, req: Request);
    fn tick(&mut self, now: Time) -> Vec<OrchAction>;
    /// The shared GPU cluster — both the run's GPU registry (the `gpus` field is
    /// what L7 serializes into `raw/run_meta.json`) and the inter-worker
    /// transfer timing oracle. L7 borrows it to read either side.
    fn cluster(&self) -> &SharedGpuCluster;
}
