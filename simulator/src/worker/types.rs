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

use crate::common::{RequestId, Time};
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

#[derive(Clone, Debug)]
pub enum WorkerMsg {
    Request(RequestId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerEvent {
    /// A request finished all its output tokens on this worker.
    RequestComplete { req: RequestId },
    /// A PD prefill worker finished a request's prefill; L6 hands it off to a
    /// decode pool. Never emitted by unified / decode workers.
    PrefillDone { req: RequestId },
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
