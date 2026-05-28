//! `IterWorker` — the L6-facing surface shared by every iter-wise
//! `IterwiseUnifiedModel`-backed worker (barebone, HP/DP, PD prefill/decode).
//! Factored into a trait so the orchestrator's worker-stamping infra
//! (`UnifiedWorkerFactory`) and DP flow (`SimpleDpFlow`) are generic over the
//! concrete worker type, not pinned to one.
//!
//! Construction stays OFF the trait: each worker keeps its own `new(...)` (all
//! sharing the same signature), and the factory is handed that `new` as a plain
//! function pointer — so the trait carries only the per-tick driving methods.

use crate::common::{Time, WorkerId};
use crate::worker::types::{WorkerEvent, WorkerMsg, WorkerStatus};
use crate::worker::unified::BareboneWorker;
use crate::arch::contract::IterwiseUnifiedModel;

/// The surface L6 drives an iter-wise worker through each tick: read its id (for
/// GPU inventory + placement), hand it work (`enqueue`), advance its FSM while
/// pushing `WorkerEvent`s into the caller's `events` sink (`tick`), and read its
/// load for placement (`status`). `tick` returns the worker's next wakeup so the
/// pool can skip non-due workers on the fixed global clock without entering
/// their FSM. Events are pushed (self-tagged with the worker id) rather than
/// buffered + pulled, so the pool needs no separate per-worker drain sweep.
pub trait IterWorker {
    fn id(&self) -> WorkerId;
    fn enqueue(&mut self, msg: WorkerMsg);
    fn tick(&mut self, now: Time, events: &mut Vec<WorkerEvent>) -> Option<Time>;
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
    fn tick(&mut self, now: Time, events: &mut Vec<WorkerEvent>) -> Option<Time> {
        BareboneWorker::tick(self, now, events)
    }
    fn status(&self) -> WorkerStatus {
        BareboneWorker::status(self)
    }
}
