//! `IterWorker` — the L6-facing surface shared by every iter-wise worker.
//! A worker may be backed by `IterwiseUnifiedModel`, a paired model-specific
//! input contract, or multiple resident models; L6 deliberately sees none of
//! those construction details. `WorkerFactory<W>` stamps concrete workers and
//! `SimpleDpFlow<W>` drives only this trait.
//!
//! Construction stays OFF the trait: family-local `build_*_worker` recipes choose
//! the concrete composition, while factories/controllers keep only that builder
//! call. The trait therefore carries only per-tick driving methods.

use crate::common::{Time, WorkerId};
use crate::worker::types::WorkerStatus;
use crate::worker::types::{AttnWorkerEvent, AttnWorkerMsg, FfnWorkerEvent, FfnWorkerMsg};

/// The surface L6 drives an iter-wise worker through each tick: read its id (for
/// GPU inventory + placement), hand it work (`enqueue`), advance its FSM while
/// pushing role-specific events into the caller's `events` sink (`tick`), and
/// read its load for placement (`status`). `tick` returns the worker's next
/// wakeup so the pool can skip non-due workers on the fixed global clock without
/// entering their FSM. Events are pushed (self-tagged with the worker id)
/// rather than buffered + pulled, so the pool needs no separate per-worker
/// drain sweep.
///
/// Each worker carries its own `Msg` / `Event` associated types — so adding a
/// new worker with role-specific traffic never forces a change in existing
/// workers. Barebone / HP use `WorkerMsgCommon` / `WorkerEventCommon`; PD and
/// AFD use their own flat, role-specific enums.
///
/// The shared `GpuCluster` is not part of this trait — every worker takes a
/// `SharedGpuCluster` at construction (used by PD workers for runtime
/// transfers; ignored by barebone/hp after their one-shot `allocate`).
pub trait IterWorker {
    /// Role-specific message set this worker accepts. Always wraps
    /// `WorkerMsgCommon` (which carries the universal `Request`).
    type Msg;
    /// Role-specific event set this worker emits. Always wraps
    /// `WorkerEventCommon` (which carries the universal `RequestComplete`).
    type Event;

    fn id(&self) -> WorkerId;
    fn enqueue(&mut self, msg: Self::Msg);
    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time>;
    fn status(&self) -> WorkerStatus;
}

/// Extra L6-facing capability required by the AFD attention pool.
///
/// The pool constructs the common slot/barrier messages through `From` and
/// balances new requests by the worker-owned projected KV peak. A PD-for-AFD
/// variant may use a wider message enum (for example, adding a prefill handoff)
/// while still accepting the common AFD control protocol.
pub trait AfdAttnWorker: IterWorker<Event = AttnWorkerEvent>
where
    Self::Msg: From<AttnWorkerMsg>,
{
    fn estimated_peak_kv(&self) -> u64;
}

/// L6-facing capability for an AFD FFN executor.  Unlike attention there is no
/// placement metric: a FFN worker consumes independent aggregated tasks, so its
/// complete contract is the typed `IterWorker` message/event surface.
pub trait AfdFfnWorker: IterWorker<Msg = FfnWorkerMsg, Event = FfnWorkerEvent> {}

impl<T> AfdFfnWorker for T where T: IterWorker<Msg = FfnWorkerMsg, Event = FfnWorkerEvent> {}
