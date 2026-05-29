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
        /// Total KV bytes to transfer.
        bytes: u64,
    },
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
}

// ── PD transfer vocabulary ────────────────────────────────────────────────────

/// Sender's KV layout for a PD handoff — what a prefill worker declares when
/// its iter finishes. `send_gid` is the prefill worker's comm-group id (the
/// attn-shard endpoint set registered once at construction with the shared
/// cluster); `kv_bytes` is the request's full KV size across all layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SendSpec {
    pub kv_bytes: u64,
    pub send_gid: u16,
}

/// Fully-resolved transfer plan used inside a decode worker's pending-pull
/// queue. Sender side comes off the incoming `Handoff` message; destination
/// side is the receiving decode worker's own comm-group id (pre-resolved at
/// construction). Both are cluster-internal `u16` indices, so the whole plan
/// is `Copy` and no allocation is needed per handoff.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransferPlan {
    pub send_gid: u16,
    pub recv_gid: u16,
    pub bytes: u64,
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
