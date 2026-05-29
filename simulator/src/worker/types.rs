//! Shared L5 worker vocabulary — the iter-wise FSM scaffolding plus the L6-facing
//! message / event / status / config types reused by every iter-wise worker
//! (`BareboneWorker`, `HpUnifiedWorker`, `PdPrefillWorker`, `PdDecodeWorker`) and
//! the [`IterWorker`](crate::worker::iter_worker::IterWorker) trait.
//!
//! These used to live in `unified.rs` (the barebone worker file) for historical
//! reasons — barebone was the first worker, so the shared types were defined
//! alongside it and later workers imported from there. They are pulled out here so
//! the vocabulary has a neutral home and no single worker "owns" it. `unified.rs`
//! now holds only `BareboneWorker`.

use crate::common::{RequestId, Time, WorkerId};
use crate::worker::admission_helpers::{KvAdmission, LoadBalance};

// ── FSM types ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerFsmState {
    Idle,
    Active,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IterCursor {
    NotStarted,
    Computing,
    Done,
}

#[derive(Clone, Copy, Debug)]
pub struct BatchFsmState {
    pub cursor: IterCursor,
    pub compute_end: Time,
}

// ── Messages / events (L6 interface) ──────────────────────────────────────────
//
// Each worker carries its own `type Msg` / `type Event` (associated types on
// `IterWorker`), so adding a new worker with role-specific messages or events
// never forces a change to existing workers' files. Workers that have NO role-
// specific traffic (barebone, HP) use `WorkerMsgCommon` / `WorkerEventCommon`
// directly; PD prefill / PD decode each have their own flat enum that inlines
// the universal `Request` / `RequestComplete` variant alongside the role-
// specific ones. Flat (not nested) — a new universal message would touch each
// flat enum once, but every call site is single-level (no `::Common(...)`).

/// Universal request admission. Worker types that have no role-specific
/// messages (`BareboneWorker`, `HpUnifiedWorker`) use this as their `Msg`
/// directly; PD workers inline the `Request` variant in their own enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerMsgCommon {
    Request(RequestId),
}

/// Universal completion event. Worker types that have no role-specific events
/// (`BareboneWorker`, `HpUnifiedWorker`) use this as their `Event` directly;
/// PD workers inline the `RequestComplete` variant in their own enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerEventCommon {
    RequestComplete { worker: WorkerId, req: RequestId },
}

/// PD prefill worker's full message set: the universal `Request` (inlined,
/// single-level) plus the decode-side ack `ReleaseKv` that lets it drop a
/// held KV reservation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PdPrefillMsg {
    /// Admit a new request — same shape as the universal `Request` (kept
    /// flat so call sites are single-level).
    Request(RequestId),
    /// Decode → prefill ack: a request's KV has fully landed at the decode
    /// side, so this prefill worker can drop its held reservation. Routed by
    /// L6 from a `PdDecodeEvent::PullComplete` to the originating prefill
    /// worker by `WorkerId` (*not* placement-chosen) — only that worker
    /// holds the request's KV slot.
    ReleaseKv { req: RequestId },
}

/// PD prefill worker's full event set: the universal `RequestComplete`
/// (inlined; single-token requests finish on prefill) plus the handoff signal
/// that L6 forwards to the decode pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PdPrefillEvent {
    /// Universal completion (inlined). Single-token requests finish at the
    /// prefill worker; multi-token requests emit `PrefillDone` instead and
    /// hand off to decode.
    RequestComplete { worker: WorkerId, req: RequestId },
    /// A PD prefill worker finished a request's prefill; L6 hands it off to a
    /// decode pool. `send_gid` is the sender's comm-group id (registered at
    /// prefill worker construction); `kv_tokens` is the request's KV token
    /// count (`prompt_len + prefix_kv`). Together they let L6 build the
    /// matching `PdDecodeMsg::Handoff` without re-deriving the model.
    PrefillDone {
        worker: WorkerId,
        req: RequestId,
        send_gid: u16,
        kv_tokens: u64,
    },
}

/// PD decode worker's full message set: the universal `Request` (inlined) plus
/// the prefill-side handoff that admits an already-prefilled request and
/// triggers the KV pull from the sender's comm group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PdDecodeMsg {
    /// Admit a new request — same shape as the universal `Request` (kept
    /// flat so call sites are single-level).
    Request(RequestId),
    /// PD handoff: this decode pool admits an already-prefilled request and must
    /// pull its KV from the sender's comm group. Only the sender side is in the
    /// wire message — the destination is the receiving decode worker's own
    /// pre-registered comm group, which it already knows; the worker fills its
    /// `recv_gid` in when assembling the internal `TransferPlan`.
    Handoff {
        req: RequestId,
        /// Sender's comm-group id (registered at prefill worker construction).
        send_gid: u16,
        /// Total KV **tokens** to transfer (the request's `prompt_len + prefix_kv`).
        /// The decode side multiplies by its arch's `total_kv_bytes_per_token`
        /// when calling `cluster.submit_transfer` to recover wire bytes.
        tokens: u64,
        /// The prefill worker that holds this request's KV until the pull
        /// completes. Carried through the decode worker's pending-pull → in-
        /// flight lifecycle so the worker can later ack the prefill side
        /// (`PdPrefillMsg::ReleaseKv`) and let it free its held capacity.
        prefill_worker: WorkerId,
    },
}

/// PD decode worker's full event set: the universal `RequestComplete` (inlined)
/// plus the pull-landed ack that triggers `ReleaseKv` back to the source
/// prefill worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PdDecodeEvent {
    /// Universal completion (inlined). The decode worker finishes a request
    /// after emitting all its output tokens.
    RequestComplete { worker: WorkerId, req: RequestId },
    /// A PD decode worker's pull just landed — the corresponding prefill worker
    /// can drop its held KV. `worker` is the decode worker (the emitter);
    /// `prefill_worker` is the target prefill worker, copied from the pull's
    /// `TransferPlan` so L6 can route the ack without consulting the request store.
    PullComplete {
        worker: WorkerId,
        req: RequestId,
        prefill_worker: WorkerId,
    },
}

// ── Universal-request ergonomics ──────────────────────────────────────────────
//
// `From<RequestId>` on every Msg enum lets the pool controller's universal
// `admit(rid)` convenience build the right concrete `W::Msg` via `rid.into()`,
// without each call site naming the variant.

impl From<RequestId> for WorkerMsgCommon {
    fn from(req: RequestId) -> Self { Self::Request(req) }
}
impl From<RequestId> for PdPrefillMsg {
    fn from(req: RequestId) -> Self { Self::Request(req) }
}
impl From<RequestId> for PdDecodeMsg {
    fn from(req: RequestId) -> Self { Self::Request(req) }
}

// ── PD transfer vocabulary ────────────────────────────────────────────────────

/// Fully-resolved transfer plan used inside a decode worker's pending-pull
/// queue. Sender side comes off the incoming `Handoff` message; destination
/// side is the receiving decode worker's own comm-group id (pre-resolved at
/// construction). Both are cluster-internal `u16` indices, so the whole plan
/// is `Copy` and no allocation is needed per handoff. `prefill_worker` is the
/// owner of the held KV at the source side, threaded through so the decode
/// worker can ack the prefill side once the pull lands. `tokens` is the wire
/// KV count; `cluster.submit_transfer` is called with `tokens × total_kv_bytes_per_token`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransferPlan {
    pub send_gid: u16,
    pub recv_gid: u16,
    pub tokens: u64,
    pub prefill_worker: WorkerId,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WorkerStatus {
    pub queued_requests: u32,
    pub active_requests: u32,
}

// ── Config ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct WorkerConfig {
    pub admission: KvAdmission,
    pub balance: LoadBalance,
    /// This worker's KV-cache memory allowance in bytes (its GPU's attention
    /// budget). Same per-worker tier as `gpu_name`; the worker divides it by the
    /// model's `kv_bytes_per_token` to size its `KvPool`.
    pub attn_kv_bytes: u64,
    /// Mirror of `io.log_output_token_times`: when off, decodes do not build the
    /// per-token timestamp array (the hot-path cost on saturated runs). Threaded
    /// to `RequestRecord::record_token` / `record_first_token`.
    pub log_output_token_times: bool,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            admission: KvAdmission::Strict,
            balance: LoadBalance::Single,
            attn_kv_bytes: 80_000_000_000, // 80 GB
            log_output_token_times: false,
        }
    }
}
