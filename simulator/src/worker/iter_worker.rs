//! `IterWorker` — the L6-facing surface shared by every iter-wise unified worker
//! (barebone single-group, HP/DP multi-group, …). Factored into a trait so the
//! orchestrator's worker-stamping infra (`UnifiedWorkerFactory`) and DP flow
//! (`SimpleDpFlow`) are generic over the concrete worker type, not pinned to one.
//!
//! Construction stays OFF the trait: each worker keeps its own `new(...)` (all
//! sharing the same signature), and the factory is handed that `new` as a plain
//! function pointer — so the trait carries only the per-tick driving methods.

use crate::common::{Time, WorkerId};
use crate::worker::unified::{BareboneWorker, WorkerEvent, WorkerMsg, WorkerStatus};
use crate::arch::contract::IterwiseUnifiedModel;

/// The surface L6 drives a unified worker through each tick: read its id (for GPU
/// inventory + event attribution), hand it work (`enqueue`), advance its FSM
/// (`tick`), collect completions (`drain_events`), and read its load for placement
/// (`status`).
pub trait IterWorker {
    fn id(&self) -> WorkerId;
    fn enqueue(&mut self, msg: WorkerMsg);
    fn tick(&mut self, now: Time);
    fn drain_events(&mut self) -> Vec<WorkerEvent>;
    fn status(&self) -> WorkerStatus;
}

/// Barebone worker (§3.4) plugs in by delegating to its existing inherent methods
/// — its bodies are unchanged; this only exposes them through the trait.
impl<M: IterwiseUnifiedModel> IterWorker for BareboneWorker<M> {
    fn id(&self) -> WorkerId {
        self.id
    }
    fn enqueue(&mut self, msg: WorkerMsg) {
        BareboneWorker::enqueue(self, msg)
    }
    fn tick(&mut self, now: Time) {
        BareboneWorker::tick(self, now)
    }
    fn drain_events(&mut self) -> Vec<WorkerEvent> {
        BareboneWorker::drain_events(self)
    }
    fn status(&self) -> WorkerStatus {
        BareboneWorker::status(self)
    }
}
