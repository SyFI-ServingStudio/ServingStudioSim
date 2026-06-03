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

// ── layer-wise AFD vocabulary (disagg attn / ffn workers) ──────────────────────
//
// AFD splits a decoder layer at the attn/ffn boundary into two worker pools.
// Requests never enter the attn worker directly: they enter the ffn pool, which
// drives the per-layer handshake (ffn pre-attn → attn → ffn post-attn) and signals
// the attn worker via `ReadyNotification`. Flat enums, matching the PD style above.
// A few fields (per-shard byte split, batch composition) are provisional and may
// be refined when the AFD flow (L6) lands.

/// Attn worker messages. A request is first `Admit`ted (sticky slot + KV
/// reservation), then its micro-batch enters the pull/compute pipeline ONLY via
/// `ReadyNotification`. L6 passes ids only — the worker reads the shared store for
/// per-request facts (prompt/decode/prefix, token counts) and tracks its own KV.
/// (`Vec<RequestId>` fields → no `Copy`, like the ffn types.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttnWorkerMsg {
    /// KV-admit: the ffn pool admitted `req`; this attn worker reserves its KV and
    /// picks a sticky slot. Per-request (admitted once). The worker reads
    /// `prompt_len` / `decode_len` / `prefix_kv` from the store — hence id only.
    Admit { req: RequestId },
    /// ffn→attn per-layer handshake and the worker's ONLY compute entry: layer
    /// `layer`'s QKV for this **micro-batch** is ready to pull from `send_gid`.
    /// `bytes` is the ffn's producer-computed handoff size — the attn side reads it,
    /// never recomputes (it lacks `ffn_to_attn_bytes_per_token`). `reqs` is the
    /// batch (this DP shard's slice of the ffn micro-batch); the worker computes
    /// their layer-`layer` attention as one group (D4) and reads each req's
    /// token / KV state from the store.
    ReadyNotification {
        reqs: Vec<RequestId>,
        layer: u16,
        send_gid: u16,
        bytes: u64,
    },
    /// Drop `req`'s KV (the ffn Terminal completed it; the flow routes the release
    /// here). Per-request; the worker tracks its own KV count.
    Release { req: RequestId },
}

/// Attn worker events. The attn worker is a pure attention+KV engine — it has NO
/// completion semantics (it never sees `decode_len` or token counts), so it emits
/// no `RequestComplete`. Iteration completion is decided by the ffn Terminal
/// (`FfnWorkerEvent::IterComplete { completed }`); the flow then sends this worker
/// an `AttnWorkerMsg::Release` to drop the KV (no ack event needed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttnWorkerEvent {
    /// This **micro-batch** finished `layer`'s attention; the ffn side pulls `bytes`
    /// (= `attn_to_ffn_bytes_per_token × tokens`, producer-computed here). Batch-
    /// granular, mirroring the inbound `ReadyNotification`.
    AttnLayerOutputsReady {
        worker: WorkerId,
        reqs: Vec<RequestId>,
        layer: u16,
        bytes: u64,
    },
}

/// Which ffn section(s) a task computes. The `downstream` attn layer is derived:
/// `Bootstrap → 0`, `Bridge{upstream} → upstream + 1`, `Terminal → none`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnTaskKind {
    /// embed + pre_attn(0); feeds attn layer 0. Input is local (no pull).
    Bootstrap,
    /// post_attn(upstream) + fused pre_attn(upstream+1); feeds attn layer upstream+1.
    Bridge { upstream: u16 },
    /// post_attn(last) + epilogue; emits the iteration's output token.
    Terminal,
}

/// Whether a request has more decode iterations after a Terminal task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IterEndState {
    MoreWork,
    Complete,
}

/// One unit of ffn work: a micro-batch's section for one layer step. `reqs` is the
/// **total** workload — the flat request set, NOT pre-grouped. The worker (L5) owns
/// the DP-shard partition: it round-robins `reqs` across `num_dp_groups`, reads the
/// store for each request's token count, and builds the L4 `FfnArchInput` itself.
/// L6 only hands over the workload — it never groups, never counts tokens, never
/// constructs arch vocabulary (the only L5↔L4 bridge is the worker). `send_gid` /
/// `pull_bytes` source the attn→ffn input pull (`pull_bytes == 0` for Bootstrap,
/// whose input is local). Built by the AFD ffn pool (L6).
#[derive(Clone, Debug)]
pub struct FfnTask {
    pub kind: FfnTaskKind,
    pub reqs: Vec<RequestId>,
    pub send_gid: u16,
    pub pull_bytes: u64,
}

/// Ffn worker message set: a single `Task` (the pool composes the batch + token
/// groups, so there is no degenerate `From<RequestId>` admit).
#[derive(Clone, Debug)]
pub enum FfnWorkerMsg {
    Task(FfnTask),
}

/// Ffn worker event set. The two variants mirror the two distinct downstream
/// actions (PD-style: distinct action → distinct variant), so each carries only
/// its own payload — no dead fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FfnWorkerEvent {
    /// A Bootstrap/Bridge section finished: its QKV output is ready for the
    /// downstream attn layer to pull (`out_bytes` from `out_send_gid`). The worker
    /// stays layer-graph-agnostic — L6 maps `kind` to the downstream attn layer
    /// (`Bootstrap → 0`, `Bridge{upstream} → upstream + 1`). `kind` is never
    /// `Terminal` here (the worker emits `IterComplete` for that).
    SectionReady {
        worker: WorkerId,
        kind: FfnTaskKind,
        reqs: Vec<RequestId>,
        out_send_gid: u16,
        out_bytes: u64,
    },
    /// A Terminal section finished: the iteration's output token was emitted for
    /// every request in `reqs`. `completed` lists those that hit `decode_len` this
    /// iteration (the rest loop back for another decode — L6's call).
    IterComplete {
        worker: WorkerId,
        reqs: Vec<RequestId>,
        completed: Vec<RequestId>,
    },
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
