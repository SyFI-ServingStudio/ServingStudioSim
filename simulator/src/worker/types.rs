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

// ── Messages / events / status (L6 interface) ─────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum WorkerMsg {
    Request(RequestId),
    /// PD handoff: the decode pool admits an already-prefilled request and must
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
        /// (`WorkerMsg::ReleaseKv`) and let it free its held capacity.
        prefill_worker: WorkerId,
    },
    /// Decode → prefill ack: a request's KV has fully landed at the decode
    /// side, so the prefill worker can drop its held reservation. Routed by
    /// L6 from a `WorkerEvent::PullComplete` to the originating prefill
    /// worker (by `WorkerId`, *not* placement-chosen) — only that worker
    /// holds the request's KV slot.
    ReleaseKv { req: RequestId },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerEvent {
    /// A request finished all its output tokens on this worker. `worker` is the
    /// emitting worker's id — the worker self-tags at push time (it knows its own
    /// id), so the pool no longer needs a separate drain sweep to attribute it.
    RequestComplete { worker: WorkerId, req: RequestId },
    /// A PD prefill worker finished a request's prefill; L6 hands it off to a
    /// decode pool. Never emitted by unified / decode workers. `send_spec` is the
    /// sender's KV layout (total bytes + this worker's attn-shard count), enough
    /// for L6 to build the matching `TransferPlan` without re-deriving the model.
    PrefillDone {
        worker: WorkerId,
        req: RequestId,
        send_spec: SendSpec,
    },
    /// A PD decode worker's pull just landed at this side — the corresponding
    /// prefill worker can drop its held KV. `worker` is the decode worker (the
    /// emitter); `prefill_worker` is the target prefill worker, copied from the
    /// pull's `TransferPlan` so L6 can route the ack without consulting the
    /// request store.
    PullComplete {
        worker: WorkerId,
        req: RequestId,
        prefill_worker: WorkerId,
    },
}

// ── PD transfer vocabulary ────────────────────────────────────────────────────

/// Sender's KV layout for a PD handoff — what a prefill worker declares when
/// its iter finishes. `send_gid` is the prefill worker's comm-group id (the
/// attn-shard endpoint set registered once at construction with the shared
/// cluster); `kv_tokens` is the request's KV token count (`prompt_len +
/// prefix_kv`). Token-based at the worker boundary so it lines up with
/// `KvPool`'s accounting; the decode worker converts to wire bytes at
/// `cluster.submit_transfer` time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SendSpec {
    pub kv_tokens: u64,
    pub send_gid: u16,
}

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
