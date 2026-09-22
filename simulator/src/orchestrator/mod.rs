//! `orchestrator` (L6) — pool-local orchestration (L6a) + inter-pool deployment
//! flow (L6b). See doc/detailed_design/L6.md.
//!
//! Module root holds only the `Flow` contract (the single surface L7 needs) and
//! the module wiring. Shared vocabulary + worker-stamping infra live in
//! `common`; concrete deployments live under `impls`.

pub mod common;
pub mod config;
pub mod impls;
pub mod migration;
pub mod training;

pub use self::common::{GpuCluster, GpuInfo, OrchAction, UnifiedWorkerFactory, WorkerFactory};
pub use config::{GroupSpec, MigrationPolicySel, PlacementPolicy, PoolSpec, TrainingSel};
pub use impls::{
    AfdAttnPoolController, AfdFfnPoolController, AfdFlow, DpPlacementPolicy, PdFlow,
    SimpleDpConfig, SimpleDpFlow, SimpleDpPoolConfig, SimpleDpPoolController, AFD_ATTN_POOL,
    AFD_FFN_POOL, PD_DECODE_POOL, PD_PREFILL_POOL,
};
pub use migration::{MigrationOrder, MigrationPolicy, MigrationTrigger, WorkerLoad};
pub use training::{TrainingConfig, TrainingPool};

use crate::common::{Request, RequestDefinition, TextGenerationDefinition, Time};
use crate::worker::SharedGpuCluster;

/// L6b deployment flow — the only object L7 calls. `on_arrival` takes the full
/// `Request` (the flow inserts its facts into the shared store, then admits the
/// id); `tick` drives the pools and surfaces deployment actions.
pub trait Flow<Definition: RequestDefinition = TextGenerationDefinition> {
    fn on_arrival(&mut self, req: Request<Definition>);
    fn tick(&mut self, now: Time) -> Vec<OrchAction>;
    /// The shared GPU cluster — both the run's GPU registry (the `gpus` field is
    /// what L7 serializes into `raw/run_meta.json`) and the inter-worker
    /// transfer timing oracle. L7 borrows it to read either side.
    fn cluster(&self) -> &SharedGpuCluster;

    /// Work this flow owns that the request stream cannot see. Default: none,
    /// which is what every deployment whose only state is requests reports.
    ///
    /// The run loop reads it for both of its exits — it may not stop while
    /// something is outstanding, and it may not call a run stuck while that
    /// something is still finishing units.
    fn background(&self) -> BackgroundWork {
        BackgroundWork::default()
    }
}

/// A flow's outstanding non-request work, as the run loop needs to see it.
///
/// An RL deployment keeps training after its last sample lands: nothing is in
/// flight by the request ledger's reckoning, yet the run is plainly not over.
/// Two numbers cover both questions the loop asks — is there more to do, and is
/// it getting done.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackgroundWork {
    /// Still owed. While true the loop keeps ticking even with every request
    /// complete.
    pub outstanding: bool,
    /// Monotone count of finished units — the stuck watchdog's progress signal,
    /// which would otherwise see a completely idle ledger and call a healthy
    /// training tail a deadlock.
    pub completed: u64,
}
