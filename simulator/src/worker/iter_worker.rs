//! `IterWorker` — the L6-facing surface shared by every iter-wise worker.
//! A worker may be backed by `IterwiseUnifiedModel`, a paired model-specific
//! input contract, or multiple resident models; L6 deliberately sees none of
//! those construction details. `WorkerFactory<W>` stamps concrete workers and
//! `SimpleDpFlow<W>` drives only this trait.
//!
//! Construction stays OFF the trait: family-local `build_*_worker` recipes choose
//! the concrete composition, while factories/controllers keep only that builder
//! call. The trait therefore carries only per-tick driving methods.

use crate::common::{RequestId, Time, WorkerId};
use crate::worker::types::WorkerStatus;
use crate::worker::types::{
    AttnWorkerEvent, AttnWorkerMsg, FfnWorkerEvent, FfnWorkerMsg, WorkerMsgCommon,
};

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

/// L6-facing capability required by a DP pool that migrates work.
///
/// Deliberately not on [`IterWorker`]: handing back everything a worker holds
/// is meaningless or wrong for the other families. An AFD attention worker's
/// placement is sticky for the KV lifetime (`doc/detailed_design/L6.md`), and a
/// PD prefill worker may be holding KV a decode worker has not pulled yet, so
/// dropping it would strand the handoff protocol. Widening L6's reach over a
/// subset of workers through a capability trait is the same shape
/// [`AfdAttnWorker`] uses.
pub trait MigratableWorker: IterWorker
where
    Self::Msg: From<WorkerMsgCommon>,
{
    /// Hand back every request this worker holds — queued, prefilling, and
    /// decoding — releasing their KV and dropping their queue entries, and
    /// append their ids to `out`.
    ///
    /// What a drained request *becomes* is not this worker's call: it returns
    /// opaque ids and leaves the shared record's progress alone, so the worker
    /// that takes them over re-derives the remaining work itself.
    ///
    /// Does not cancel an in-flight iteration. The GPU is genuinely still busy
    /// with a forward pass that was already launched, and pretending otherwise
    /// would make migration free.
    fn drain_resident(&mut self, now: Time, out: &mut Vec<RequestId>);

    /// The same, restricted to the named requests. Ids this worker does not
    /// hold are skipped, so a caller working from its own ledger does not have
    /// to know which of them have already finished.
    ///
    /// This is what makes a *partial* release expressible: a pool handing a
    /// worker back one prompt group at a time needs the worker to keep running
    /// the rest, which `drain_resident` cannot say.
    fn drain_requests(&mut self, now: Time, requests: &[RequestId], out: &mut Vec<RequestId>);

    /// Discard every retained prefix this worker holds.
    ///
    /// Not part of a drain: a drain moves live requests, this drops KV kept for
    /// sessions between rounds. The pool calls it when it retires the worker,
    /// because those bytes sit on a GPU that is about to be someone else's —
    /// and the sessions themselves will land elsewhere and miss.
    fn drop_retained_prefixes(&mut self, now: Time);
}

/// L6-facing capability for an AFD FFN executor.  Unlike attention there is no
/// placement metric: a FFN worker consumes independent aggregated tasks, so its
/// complete contract is the typed `IterWorker` message/event surface.
pub trait AfdFfnWorker: IterWorker<Msg = FfnWorkerMsg, Event = FfnWorkerEvent> {}

impl<T> AfdFfnWorker for T where T: IterWorker<Msg = FfnWorkerMsg, Event = FfnWorkerEvent> {}
