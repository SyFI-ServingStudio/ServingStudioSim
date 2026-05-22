//! `orchestrator` (L6) — pool-local orchestration (L6a) + inter-pool deployment
//! flow (L6b). See docs/detailed_design/L6/design.md.
//!
//! Module root holds only the `Flow` contract (the single surface L7 needs) and
//! the module wiring. Shared vocabulary + worker-stamping infra live in
//! `common`; concrete deployments live under `impls`.

pub mod common;
pub mod impls;

pub use self::common::{OrchAction, PoolEvent, UnifiedWorkerFactory};
pub use impls::{
    DpPlacementPolicy, SimpleDpConfig, SimpleDpFlow, SimpleDpPoolConfig, SimpleDpPoolController,
};

use crate::common::{Request, Time};

/// L6b deployment flow — the only object L7 calls. `on_arrival` takes the full
/// `Request` (the flow inserts its facts into the shared store, then admits the
/// id); `tick` drives the pools and surfaces deployment actions.
pub trait Flow {
    fn on_arrival(&mut self, req: Request);
    fn tick(&mut self, now: Time) -> Vec<OrchAction>;
}
