//! `DisaggAttnWorker` — the attn half of an AFD (attention-FFN disaggregation)
//! deployment. It owns ONLY the per-layer attention kernel + the KV cache for its
//! DP shard; everything else (qkv / o_proj / router / MoE) lives on the ffn side.
//! Completion is NOT decided here (D16): the ffn Terminal owns it and the flow
//! routes a `Release`; the worker just drops the KV.
//!
//! **One worker = one DP shard, one shared KV pool** (D4/D11). It runs **three
//! pipelined slots**, each a persistent micro-batch walking the layers through
//! `Wait → WaitComplete → Pull → PullComplete → Compute`. The slots overlap one
//! attn↔ffn pull against one attention compute: explicit resource gates admit at
//! most one slot to `Pull` and one to `Compute` at a time (ref `attn_worker`
//! serialises the same way off a rotating `head` pointer; this port drops the head
//! for stateless gates, which compose cleanly with dormant empty slots — below).
//!
//! **attn-initiated iteration** (diverges from ref's ffn-driven `submit_arrival`):
//! when a slot reaches layer-0 it emits `IterStart` (once per iteration, guarded by
//! `iter_announced`) even when the local request set is empty. The controller treats
//! this as the layer-(-1) barrier and triggers the ffn prolog only after every worker
//! has reported that slot. The prolog's layer-0 QKV returns as a
//! `ReadyNotification { slot, 0 }` — the SAME slot-addressed message as every later
//! layer (no special bootstrap path). Notifications are **slot-addressed**: the worker
//! owns slot assignment, stamps the tag on each `IterStart` /
//! `AttnLayerOutputsReady`, and L6 echoes it back, so a handshake routes to a slot in
//! O(1) (no request set on the wire).
//!
//! **All workers march every active slot in lockstep** (L6 §source-side fan-in:
//! `AttnLayerDone` fires only once EVERY worker reported the same `(slot, layer)`).
//! Once any worker's `IterStart` activates a slot, the controller scatters that
//! slot's per-layer `ReadyNotification` to ALL workers and, when the barrier
//! completes, a `SlotFlushed` to ALL workers. So a worker whose slot is EMPTY for an
//! active batch still runs each layer as a 0-token no-op and reports it — keeping it a
//! barrier member. To stop an empty slot (vacuously `input_ready`) from bursting
//! through all layers in one tick ahead of the pool, it **parks in `AwaitFlush`** after
//! each layer and advances exactly one layer per `SlotFlushed` (ref's
//! `deferred_empty_advance`, but worker-self-contained — the gate is a slot state, not
//! a scheduler reaching in). A truly idle slot (never notified) stays at layer-0
//! `Wait` and the worker quiesces — `SlotFlushed`/`ReadyNotification` only flow while
//! some worker drives real work, so the empty ring never spins on its own. A decode
//! iteration loops by the slot wrapping to layer-0 and re-emitting `IterStart`; a slot
//! whose members all complete (Released) keeps reporting as empty no-op work until a
//! later admit joins at a layer-0 wrap.
//!
//! Admission is the shared two-level scheme (ref / hp_unified): `Admit` only
//! enqueues to `worker_pending` (Level-1, ref `enqueue_to_pending`); each tick
//! `drain_pending_admits` (Level-2, ref `drain_worker_pending_to_batches`) runs the
//! projected-peak KV gate head-of-line — a rejected request blocks the queue and
//! retries when a `Release` frees KV, never "admit + warn". An admitted request
//! reserves its FULL footprint (`prompt_len + prefix_kv + decode_len`, since AFD
//! prefill is NOT done at admit) in `promised`, picks its least-KV slot (`wlb`), and
//! lands in that slot's `pending_insert`. It joins `reqs` at the slot's next layer-0
//! iteration open.
//!
//! KV accounting is two-phase. `promised` holds reserved-but-not-resident KV;
//! `Batch` (the shared shard KV pool + decode set) holds resident KV. A request
//! leaves `promised` and enters `Batch.decodes` (`begin_decode`, ref
//! `finalize_to_decode`) at its prefill→decode boundary: a prefilled handoff
//! (`!is_prefill()` at admit) at activation; a fresh prefill at the last layer of its
//! prefill pass. The prefill/decode **discriminator** is the request's authoritative
//! store status (`is_prefill()`, which the ffn Terminal advances at the boundary),
//! NOT `promised` membership — `promised` is the reserved-KV admission accounting
//! only. (This attn worker never writes the store; the ffn owns the status flip.) A
//! decode token adds one KV slot per request **per iteration**, applied via
//! `advance_subset` exactly when a slot finishes its **last** layer — per layer
//! would over-count by `num_layers`.
//!
//! Reading order: types → construction → message handling (the `IterWorker` entry
//! points) → the tick / pipeline loop → its helpers, each following the function
//! that calls it → tests.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, AttnArchInput, AttnLayerwiseModel};
use crate::common::{IdMap, PoolId, RequestId, SharedRequests, Time, WorkerId};
use crate::log::{KvSampler, KvSubmit};
use crate::worker::admission_helpers::Batch;
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::iter_worker::IterWorker;
use crate::worker::types::{AttnWorkerEvent, AttnWorkerMsg, WorkerConfig, WorkerStatus};

// ── types ─────────────────────────────────────────────────────────────────────

/// Pipeline depth — slots overlapping pull against compute (ref fixes 3).
const NUM_SLOTS: usize = 3;

/// Per-slot pipeline stage (ref `BatchState`). The single-pull / single-compute
/// invariant is enforced by the worker's resource gates in `try_transitions`, not
/// by a head pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    /// Waiting for this layer's `ReadyNotification` (dormant if the slot is empty).
    Wait,
    /// Notification in hand; ready to pull.
    WaitComplete,
    /// QKV input for `current_layer` is transferring.
    Pull,
    /// Pull landed; ready to compute.
    PullComplete,
    /// Attention for `current_layer` is running.
    Compute,
    /// Compute finished + `AttnLayerOutputsReady` emitted; **parked** until the pool's
    /// per-layer flush barrier releases this `(slot, layer)` via `SlotFlushed`. This is
    /// the gate that keeps a slot in lockstep with the whole pool — critically an EMPTY
    /// slot, which is vacuously `input_ready` (no notification to wait on) and would
    /// otherwise burst Wait→Compute→advance through every layer in one tick, flooding
    /// the ffn with out-of-order empty notifications. One layer advanced per flush.
    AwaitFlush,
}

struct Slot {
    /// Micro-batch membership. Empty ⇒ a dormant slot (no work until `pending_insert`
    /// brings it members at a layer-0 open). Grows at the layer-0 iteration open,
    /// shrinks as completed requests are Released.
    reqs: Vec<RequestId>,
    /// Admitted (Level-2) and assigned here, awaiting this slot's next layer-0 open
    /// to join `reqs`.
    pending_insert: Vec<RequestId>,
    /// The layer this slot is processing (advances `(cur + 1) % num_layers` when a
    /// compute finishes). Matches incoming notifications.
    current_layer: u16,
    /// Membership lock for the current iteration: no late `pending_insert` join and no
    /// re-announce while set. Raised when this worker emits its own `IterStart` for the
    /// layer-(-1) barrier, even if the report is empty. Cleared when the slot wraps to
    /// layer 0, reopening the next iteration (a decode loop-back / a held
    /// `pending_insert` request then announces).
    iter_announced: bool,
    state: SlotState,
    /// Slot-level notification flag: this layer's QKV input is ready to pull.
    notified: bool,
    /// Pull descriptor carried from the notification until `start_pull` consumes it.
    pull_send_gid: u16,
    pull_bytes: u64,
    /// Live event timestamps for the current `Pull` / `Compute`.
    pull_end: Time,
    compute_end: Time,
    /// Per-slot forward-pass counter (the `cost_log` row `iter_id`), bumped when the
    /// slot wraps back to layer 0. Groups all of one forward's per-layer `attn` rows.
    iter: u64,
    /// Whether this slot's cached attention input (`slot_inputs[idx]`) is up to date.
    /// The input is constant across an iteration's layers, so it is rebuilt lazily on
    /// the first compute after the flag is cleared (iteration wrap / `pending_insert`
    /// drain / `Release`) and reused for the rest of the iteration's layers.
    input_valid: bool,
}

impl Slot {
    fn new() -> Self {
        Self {
            reqs: Vec::new(),
            pending_insert: Vec::new(),
            current_layer: 0,
            iter_announced: false,
            state: SlotState::Wait,
            notified: false,
            pull_send_gid: 0,
            pull_bytes: 0,
            pull_end: Time::ZERO,
            compute_end: Time::ZERO,
            iter: 0,
            input_valid: false,
        }
    }

    /// This slot may leave `Wait`: this layer's QKV notification has arrived (ref
    /// `all_notified`). NO empty-slot short-circuit — an empty slot the controller
    /// notified (because it scatters to ALL workers each active layer) advances too, as
    /// a 0-token no-op, to keep the whole pool in per-layer lockstep. A never-notified
    /// slot stays put (`notified == false` is the dormancy / quiescence guard).
    fn input_ready(&self) -> bool {
        self.notified
    }

    /// This slot is at the layer-(-1) start boundary and has not reported it for the
    /// current iteration yet. The report may carry an empty request set.
    fn start_boundary_ready(&self) -> bool {
        self.current_layer == 0 && self.state == SlotState::Wait && !self.iter_announced
    }
}

/// Level-1 admission backlog: a FIFO of KV-gated pending admits that also carries a
/// running full-footprint KV sum.
///
/// Why this exists: the attn pool's least-KV placement calls
/// [`DisaggAttnWorker::estimated_peak_kv`] on every worker for every arrival, which
/// folded the raw deque — O(|pending|) per arrival. Under a saturated KV pool the
/// backlog grows to tens of thousands of entries, so placement was O(N²) over a run.
/// Each entry carries its full-footprint weight (`prompt_kv + remaining`), which is
/// **constant while the request is queued** — a queued request has not started
/// decoding (`tokens_emitted == 0`) — so the incrementally-maintained `reserved_kv`
/// equals a fresh fold of the deque exactly (u64, order-independent), read in O(1).
/// Order is preserved for the head-of-line drain.
///
/// No membership index: in the AFD flow a request is `Admit`ed exactly once
/// (`AfdFlow::on_arrival`, dense store) and `Release`d only after it has drained
/// pending→slot, so the old `on_msg_admit` idempotency `contains` and `on_msg_release`
/// `retain` never actually removed/found anything. Those invariants are now
/// `debug_assert!`ed (via [`PendingQueue::contains`], a debug-only O(n) scan) and
/// compile out of `--release` — the path the simulator runs — entirely.
#[derive(Default)]
struct PendingQueue {
    /// FIFO of (request, its full-footprint KV weight at enqueue time).
    order: VecDeque<(RequestId, u64)>,
    /// Running sum of the weights in `order` (the `estimated_peak_kv` L1 term).
    reserved_kv: u64,
}

impl PendingQueue {
    fn len(&self) -> usize {
        self.order.len()
    }

    fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Membership by linear scan — **debug-only** (only `debug_assert!` invariant
    /// checks call it; it compiles out of release). Not a hot-path operation.
    fn contains(&self, req: RequestId) -> bool {
        self.order.iter().any(|&(r, _)| r == req)
    }

    /// The L1 queued footprint — the running sum, O(1) (replaces a deque fold).
    fn reserved_kv(&self) -> u64 {
        self.reserved_kv
    }

    /// Head request without dequeuing (the drain peeks before the KV gate).
    fn front(&self) -> Option<RequestId> {
        self.order.front().map(|&(r, _)| r)
    }

    /// Enqueue a fresh admit at its full-footprint `weight`.
    fn push_back(&mut self, req: RequestId, weight: u64) {
        self.order.push_back((req, weight));
        self.reserved_kv += weight;
    }

    /// Pop the head (the drain admitted it into a slot). Keeps the sum in step.
    fn pop_front(&mut self) -> Option<RequestId> {
        let (req, weight) = self.order.pop_front()?;
        self.reserved_kv -= weight;
        Some(req)
    }
}

pub struct DisaggAttnWorker<M: AttnLayerwiseModel> {
    pub id: WorkerId,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cluster: SharedGpuCluster,
    /// This shard's recv endpoint — the QKV pull from the ffn lands here, and it
    /// doubles as the send endpoint the ffn pulls attention outputs from.
    recv_gid: u16,
    /// One shard-level `Batch` shared across all slots (the slots are pipelined
    /// micro-batches, NOT DP shards — D11). It bundles the KV pool with the
    /// per-request decode state (`current_kv` + `remaining_decode`); resident KV
    /// only. Reserved-not-resident KV lives in `promised`.
    batch: Batch,
    slots: [Slot; NUM_SLOTS],
    /// Level-1 admission backlog (FIFO) with O(1) membership + running reserved-KV
    /// sum. Drained under the KV gate each tick. See [`PendingQueue`].
    worker_pending: PendingQueue,
    /// Reserved-but-not-resident KV per request (full footprint). Admission
    /// accounting only — feeds the `try_admit` group-promised total; a request
    /// leaves it at `begin_decode`. (NOT the prefill/decode discriminator: that is
    /// the request's store `is_prefill()` status.)
    promised: IdMap<RequestId, u64>,
    /// Sticky request → slot routing (set at admission; survives decode re-entry).
    request_to_slot: IdMap<RequestId, usize>,
    /// Eval scratch + the per-section `cost_log` writer. The attn side has one cost
    /// group, so every row is `section = "attn"` (the layer is the row's `layer`,
    /// the pipeline slot its `batch_id`). Honest per-layer rows; ZSTD compresses the
    /// homogeneous layers.
    cost: CostBuffers,
    /// Per-worker KV-occupancy sampler (running-max + stride throttle). `None` when
    /// the run has no log dir. Fed one `submit` per iteration at the KV settle point
    /// (last-layer `advance_subset`); group 0 (one shard-level pool).
    kv: Option<KvSampler>,
    /// Per-slot attention-input cache (one `AttnArchInput` per pipeline slot). The
    /// built input depends only on the slot's `reqs`, each request's `is_prefill()`,
    /// and each resident decode's `current_kv` — all of which change **only at the
    /// iteration boundary** (reqs at the layer-0 open / a mid-iteration `Release`;
    /// `current_kv` at the last-layer `advance_subset`; `is_prefill` at the ffn
    /// Terminal's boundary flip, D16). So the same input would otherwise be rebuilt
    /// ~num_layers times per iteration. Each slot keeps its input here and rebuilds it
    /// only when [`Slot::input_valid`] is cleared — at the iteration wrap
    /// (`advance_slot_layer`), a `pending_insert` drain (`open_iterations`), and a
    /// `Release` (`on_msg_release`). Per-slot (not one shared buffer) because a slot's
    /// cache must survive while another slot computes.
    slot_inputs: [AttnArchInput; NUM_SLOTS],
}

// ── construction ────────────────────────────────────────────────────────────────

impl<M: AttnLayerwiseModel> DisaggAttnWorker<M> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: WorkerId,
        model: Arc<M>,
        requests: SharedRequests,
        config: WorkerConfig,
        pool: PoolId,
        gpu_name: &str,
        cluster: SharedGpuCluster,
        cost_log_dir: Option<PathBuf>,
        pool_tag: &'static str,
    ) -> Self {
        let recv_gid = {
            let mut c = cluster.borrow_mut();
            let base = c.allocate(pool.0, id.0, model.gpus_per_replica(), gpu_name);
            // The attn-shard prefix is the comm group used as the QKV recv endpoint.
            c.register_comm_group(base, model.num_attn_shards().max(1), pool_tag, id.0)
        };
        // KV capacity in tokens: per-GPU budget × the shard's GPUs / total wire KV
        // bytes per token (same sizing as the PD decode worker).
        let shard_bytes = config
            .attn_kv_bytes
            .saturating_mul(model.num_attn_shards().max(1) as u64);
        let kv_capacity = (shard_bytes / model.total_kv_bytes_per_token().max(1)).max(1);
        // Register the static capacity for `run_meta` and open the occupancy sampler
        // BEFORE `cost_log_dir` is moved into `CostBuffers::new` below. One shard-level
        // pool → group 0, `num_groups = 1`.
        cluster
            .borrow_mut()
            .register_kv_capacity(pool_tag, pool.0, id.0, 0, kv_capacity);
        let kv = KvSampler::open_opt(cost_log_dir.as_deref(), pool_tag, id, 1, config.kv_log_stride);
        let cost = CostBuffers::new(cost_log_dir, pool_tag, id, &model.cost_log_manifest());
        Self {
            id,
            model,
            requests,
            config,
            cluster,
            recv_gid,
            batch: Batch::new(0, kv_capacity),
            slots: std::array::from_fn(|_| Slot::new()),
            worker_pending: PendingQueue::default(),
            promised: IdMap::default(),
            request_to_slot: IdMap::default(),
            cost,
            kv,
            slot_inputs: std::array::from_fn(|_| AttnArchInput { groups: Vec::new() }),
        }
    }
}

// ── message handling (entry points) ─────────────────────────────────────────────
//
// The worker's whole external surface: `enqueue` receives the three messages
// (`Admit` / `ReadyNotification` / `Release`) and dispatches each to a handler;
// `tick` drives the slot pipeline. Everything below this block is the body those
// two methods call into.

impl<M: AttnLayerwiseModel> IterWorker for DisaggAttnWorker<M> {
    type Msg = AttnWorkerMsg;
    type Event = AttnWorkerEvent;

    fn id(&self) -> WorkerId {
        self.id
    }

    fn enqueue(&mut self, msg: Self::Msg) {
        match msg {
            AttnWorkerMsg::Admit { req } => self.on_msg_admit(req),
            AttnWorkerMsg::Release { req } => self.on_msg_release(req),
            AttnWorkerMsg::ReadyNotification {
                slot,
                layer,
                send_gid,
                bytes,
            } => self.on_msg_ready_notification(slot as usize, layer, send_gid, bytes),
            AttnWorkerMsg::SlotFlushed { slot, layer } => {
                self.on_msg_slot_flushed(slot as usize, layer)
            }
        }
    }

    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time> {
        self.tick_inner(now, events)
    }

    fn status(&self) -> WorkerStatus {
        let active = self.slots.iter().filter(|s| !s.reqs.is_empty()).count() as u32;
        WorkerStatus {
            queued_requests: self.worker_pending.len() as u32,
            active_requests: active,
        }
    }
}

impl<M: AttnLayerwiseModel> DisaggAttnWorker<M> {
    // ── admission entry: Admit / Release ─────────────────────────────────────────

    /// Level-1 enqueue (ref `enqueue_to_pending`): record the request as pending. No
    /// KV is reserved yet — the projected-peak gate runs at the Level-2 drain so a
    /// rejected request blocks the queue instead of being force-admitted.
    ///
    /// A request is `Admit`ed exactly once (`AfdFlow::on_arrival`, dense store) and is
    /// never queued while promised / placed, so this is an unconditional enqueue. The
    /// old idempotency drop is kept as a `debug_assert!` (compiled out of release):
    /// if it ever fires, the flow's once-per-request invariant broke upstream.
    fn on_msg_admit(&mut self, req: RequestId) {
        debug_assert!(
            !self.worker_pending.contains(req)
                && !self.promised.contains_key(&req)
                && !self.request_to_slot.contains_key(&req),
            "AFD Admit is once-per-request; re-Admit of {req:?} means the flow invariant broke"
        );
        let (prompt_kv, remaining) = self.footprint(req);
        self.worker_pending.push_back(req, prompt_kv + remaining as u64);
    }

    /// Drop a completed request's KV (the ffn Terminal completed it; the flow routed
    /// the release here). Removes it from every level it can sit at; a slot that
    /// empties drops back to dormant.
    fn on_msg_release(&mut self, req: RequestId) {
        if let Some(current_kv) = self.batch.decode_current_kv(req) {
            self.batch.release(req, current_kv);
        }
        self.promised.remove(&req);
        // A Release arrives only after the request drained pending→slot (the ffn
        // Terminal completed it), so it is never still queued here — the old
        // `worker_pending.retain` always removed nothing. Assert that in debug.
        debug_assert!(
            !self.worker_pending.contains(req),
            "Release of {req:?} while still queued; a pending request cannot complete"
        );
        if let Some(slot_idx) = self.request_to_slot.remove(&req) {
            let s = &mut self.slots[slot_idx];
            s.reqs.retain(|&r| r != req);
            s.pending_insert.retain(|&r| r != req);
            // `reqs` changed mid-iteration ⇒ the cached attention input is stale.
            s.input_valid = false;
            // A slot emptied by release drops back to dormant: clear `iter_announced`
            // so the next members admitted into it reopen the iteration (otherwise the
            // stale announce flag would block `open_iterations`).
            if s.reqs.is_empty() {
                s.iter_announced = false;
            }
        }
    }

    /// Slot-addressed notification: the micro-batch in slot `slot` has its layer-input
    /// QKV ready to pull from `send_gid`. Routed in O(1) by the tag L6 echoed back —
    /// no request resolution, a single per-slot `notified` flag. The ffn drives layers
    /// in lockstep (layer L's QKV is produced only after the attn emits layer L-1), so
    /// a notification ALWAYS finds its slot already waiting at exactly that layer —
    /// there is no out-of-order arrival to buffer. A mismatch is a tag / protocol
    /// violation, not a retry.
    fn on_msg_ready_notification(&mut self, slot: usize, layer: u16, send_gid: u16, bytes: u64) {
        let Some(s) = self.slots.get_mut(slot) else {
            debug_assert!(false, "notification slot tag {slot} out of range");
            return;
        };
        debug_assert!(
            layer == s.current_layer && s.state == SlotState::Wait,
            "slot {slot} got a layer-{layer} notification but is at layer {} in {:?} — \
             layers run in lockstep, so this should never reach",
            s.current_layer,
            s.state
        );
        if layer != s.current_layer || s.state != SlotState::Wait {
            return;
        }
        s.notified = true;
        s.pull_send_gid = send_gid;
        s.pull_bytes = bytes;
    }

    /// The pool's per-layer barrier completed for `(slot, layer)` — release the slot
    /// parked in `AwaitFlush` and advance it one layer. This is the lockstep gate: a
    /// slot does NOT self-advance after `complete_layer`; it waits here so it (empty or
    /// not) stays in step with every other worker on this batch slot. `advance_slot_layer`
    /// does the actual layer bump (and the layer-0 wrap that reopens the next iteration).
    fn on_msg_slot_flushed(&mut self, slot: usize, layer: u16) {
        let Some(s) = self.slots.get_mut(slot) else {
            debug_assert!(false, "SlotFlushed slot tag {slot} out of range");
            return;
        };
        debug_assert!(
            s.state == SlotState::AwaitFlush && layer == s.current_layer,
            "slot {slot} got SlotFlushed for layer {layer} but is at layer {} in {:?} — \
             flush releases exactly the layer the slot parked on",
            s.current_layer,
            s.state
        );
        if s.state != SlotState::AwaitFlush || layer != s.current_layer {
            return;
        }
        self.advance_slot_layer(slot);
    }

    // ── tick & pipeline loop ─────────────────────────────────────────────────────

    fn tick_inner(&mut self, now: Time, events: &mut Vec<AttnWorkerEvent>) -> Option<Time> {
        // Idle guard: no admits pending, no slot mid-pipeline, and no slot waiting to
        // publish its layer-(-1) start report ⇒ nothing to do.
        if !self.has_work() {
            return None;
        }
        self.drain_pending_admits();
        // Fixpoint: open each layer-0 iteration (activate admits + announce
        // `IterStart`), settle event-time completions, then start the next pull /
        // compute under the single-pull / single-compute gates.
        loop {
            let mut progressed = false;
            progressed |= self.open_iterations(events);
            progressed |= self.advance_completions(now, events);
            progressed |= self.try_transitions(now);
            if !progressed {
                break;
            }
        }
        self.next_wakeup(now)
    }

    /// A request is queued, a slot must report the layer-(-1) boundary, or a slot is
    /// mid-pipeline for an active batch — including an EMPTY slot the controller
    /// notified (it must run its 0-token layer) or one parked in `AwaitFlush` awaiting
    /// `SlotFlushed`.
    fn has_work(&self) -> bool {
        !self.worker_pending.is_empty()
            || self.slots.iter().any(|s| {
                !s.reqs.is_empty()
                    || !s.pending_insert.is_empty()
                    || s.start_boundary_ready()
                    || s.notified
                    || s.state != SlotState::Wait
            })
    }

    /// Level-2 drain (ref `drain_worker_pending_to_batches`): admit from the head of
    /// `worker_pending` under the projected-peak KV gate, head-of-line (stop at the
    /// first reject, do not skip). Each admit reserves the FULL footprint in
    /// `promised` and assigns the least-KV slot.
    fn drain_pending_admits(&mut self) {
        while let Some(req) = self.worker_pending.front() {
            let (prompt_kv, remaining) = self.footprint(req);
            if remaining == 0 {
                self.worker_pending.pop_front(); // nothing left to decode
                continue;
            }
            // AFD prefill is NOT done at admit, so the footprint is the whole ramp:
            // prompt_kv (= prompt_len + prefix_kv) + the decode horizon.
            if !self.config.admission.try_admit(
                &self.batch,
                self.promised_kv(),
                prompt_kv as u32,
                remaining,
            ) {
                break; // head-of-line: retry when a Release frees KV
            }
            self.worker_pending.pop_front();
            // Admission point (mirrors unified/pd `mark_admitted`): the request
            // leaves the pending queue and starts prefill. Advance the store's
            // admitted-prefix watermark so the sim's stuck-watchdog sees progress
            // (and dense `request_state` snapshots log it). This is the ONLY store
            // write this worker makes — the ffn Terminal still owns the
            // prefill→decode status flip (D16).
            self.requests.borrow_mut().mark_admitted(req);
            self.promised.insert(req, prompt_kv + remaining as u64);
            let slot = self.wlb_choose_least_kv();
            self.slots[slot].pending_insert.push(req);
            self.request_to_slot.insert(req, slot);
        }
    }

    /// A request's `(prompt_kv, remaining_decode)` from the shared store. Reading it
    /// predicts KV pressure only — completion stays the ffn Terminal's call (D16).
    fn footprint(&self, req: RequestId) -> (u64, u32) {
        let store = self.requests.borrow();
        let r = &store[req];
        let prompt_kv = (r.prompt_len + r.prefix_kv) as u64;
        let remaining = r.decode_len.saturating_sub(r.tokens_emitted);
        (prompt_kv, remaining)
    }

    fn promised_kv(&self) -> u64 {
        self.promised.values().sum()
    }

    /// This worker's estimated peak KV — the metric the attn pool balances on when it
    /// pins a fresh request (KV is the binding resource on the attn side, so a
    /// request-count proxy ignores context-length skew). Sum of this worker's three
    /// disjoint KV-accounting levels — a request flows L1→L2→L3 and each transition
    /// moves it out of the prior level, so there is no double count:
    ///   * L3 resident decodes — the rigorous projected peak ([`Batch::projected_peak_kv`]),
    ///     modeling each live decode's per-step growth AND the KV freed as the
    ///     earliest-finishing decodes exit (tighter than a naive sum);
    ///   * L2 reserved prefills (`promised`) — admitted + prefilling, not yet resident,
    ///     at full footprint (`prompt_kv + remaining`);
    ///   * L1 queued admits (`worker_pending`) — enqueued by the pool, not yet drained,
    ///     at full footprint. `enqueue` reaches `worker_pending` synchronously, so a
    ///     fresh admit shows up here at once — the pool can spread several same-tick
    ///     arrivals across shards instead of piling them onto one.
    pub fn estimated_peak_kv(&self) -> u64 {
        // L1 queued footprint is the running sum maintained by `PendingQueue` (each
        // entry's `prompt_kv + remaining`, constant while queued) — O(1) here instead
        // of folding the whole backlog on every placement probe.
        let queued = self.worker_pending.reserved_kv();
        self.batch.projected_peak_kv() + self.promised_kv() + queued
    }

    /// Least-KV slot (D-D): the load balancer balances reserved+resident KV across
    /// slots (the user's metric; ref balances request count). Ties pick the lowest
    /// index — irrelevant, slots are addressed by identity only.
    fn wlb_choose_least_kv(&self) -> usize {
        (0..NUM_SLOTS)
            .min_by_key(|&i| self.slot_kv_load(i))
            .unwrap_or(0)
    }

    fn slot_kv_load(&self, idx: usize) -> u64 {
        let s = &self.slots[idx];
        s.reqs
            .iter()
            .chain(s.pending_insert.iter())
            .map(|&r| self.reserved_kv(r))
            .sum()
    }

    /// A request's KV weight for load balancing: its promised footprint while
    /// prefilling, else its resident decode length.
    fn reserved_kv(&self, rid: RequestId) -> u64 {
        self.promised
            .get(&rid)
            .copied()
            .unwrap_or_else(|| self.current_kv(rid))
    }

    /// Open each layer-0 slot's iteration (ref :435 activate + the attn-initiated
    /// announce, D4): drain admitted `pending_insert` into `reqs` (a prefilled handoff
    /// becomes resident decode now; a fresh prefill stays in `promised` until its last
    /// prefill layer), then emit `IterStart` once so the controller's layer-(-1)
    /// barrier can trigger the ffn prolog. `reqs` may be empty; `iter_announced` dedups
    /// the report and locks membership for this worker's slot until the next wrap.
    fn open_iterations(&mut self, events: &mut Vec<AttnWorkerEvent>) -> bool {
        let mut progressed = false;
        for idx in 0..NUM_SLOTS {
            let s = &self.slots[idx];
            if !s.start_boundary_ready() {
                continue;
            }
            if !self.slots[idx].pending_insert.is_empty() {
                let newly: Vec<RequestId> = self.slots[idx].pending_insert.drain(..).collect();
                for rid in newly {
                    self.slots[idx].reqs.push(rid);
                    if !self.req_is_prefill(rid) {
                        self.begin_decode(rid); // prefilled handoff: resident decode now
                    }
                }
                // `reqs` grew ⇒ rebuild the cached attention input on the next compute.
                self.slots[idx].input_valid = false;
            }
            let reqs = self.slots[idx].reqs.clone();
            events.push(AttnWorkerEvent::IterStart {
                worker: self.id,
                slot: idx as u8,
                reqs,
            });
            self.slots[idx].iter_announced = true;
            progressed = true;
        }
        progressed
    }

    /// Move a request from reserved (`promised`) to resident decode (ref
    /// `on_decode_start` / `finalize_to_decode`). The initial resident KV is its
    /// prompt KV; the budget is its remaining decode horizon. Fires once.
    fn begin_decode(&mut self, rid: RequestId) {
        let (initial_kv, decode_budget) = {
            let store = self.requests.borrow();
            let r = &store[rid];
            (
                (r.prompt_len + r.prefix_kv) as u64,
                r.decode_len.saturating_sub(r.tokens_emitted),
            )
        };
        self.batch
            .finalize_to_decode(rid, initial_kv, decode_budget);
        self.promised.remove(&rid);
    }

    fn req_is_prefill(&self, rid: RequestId) -> bool {
        self.requests.borrow()[rid].is_prefill()
    }

    /// Event-time completions: a notified `Wait` slot becomes pullable, a landed pull
    /// becomes computable, a finished compute emits its handoff and advances a layer.
    fn advance_completions(&mut self, now: Time, events: &mut Vec<AttnWorkerEvent>) -> bool {
        let mut progressed = false;
        for idx in 0..NUM_SLOTS {
            match self.slots[idx].state {
                SlotState::Wait if self.slots[idx].input_ready() => {
                    self.slots[idx].state = SlotState::WaitComplete;
                    progressed = true;
                }
                SlotState::Pull if now >= self.slots[idx].pull_end => {
                    self.slots[idx].state = SlotState::PullComplete;
                    progressed = true;
                }
                SlotState::Compute if now >= self.slots[idx].compute_end => {
                    self.complete_layer(idx, now, events);
                    progressed = true;
                }
                _ => {}
            }
        }
        progressed
    }

    /// Finish a slot's layer: emit the attn→ffn handoff (producer-computed bytes, so
    /// the ffn reads them off the message); at the iteration boundary (last layer)
    /// move finished prefills into decode and grow decoders' KV by one; then **park in
    /// `AwaitFlush`** — the slot does NOT self-advance, it waits for the pool barrier's
    /// `SlotFlushed` so all workers on this batch slot step together. `reqs` MAY be
    /// empty: an active batch slot the controller notified runs here as a 0-token no-op
    /// (`tokens = 0`, `out_bytes = 0`, the boundary loops skip the empty set) so it
    /// stays a barrier member.
    fn complete_layer(&mut self, idx: usize, now: Time, events: &mut Vec<AttnWorkerEvent>) {
        let layer = self.slots[idx].current_layer;
        // Token count reads the slot's `reqs` in place (no clone). The event needs an
        // owned request set, so clone exactly once and MOVE it in — this runs per
        // layer (≈num_layers × per iteration), so the old second clone was pure churn.
        let tokens = self.batch_query_tokens(&self.slots[idx].reqs);
        let out_bytes = self.model.attn_to_ffn_bytes_per_token() * tokens;
        events.push(AttnWorkerEvent::AttnLayerOutputsReady {
            worker: self.id,
            slot: idx as u8,
            reqs: self.slots[idx].reqs.clone(),
            tokens,
            layer,
            send_gid: self.recv_gid,
            bytes: out_bytes,
        });

        // Iteration boundary (last layer): a fresh prefill that just finished its pass
        // enters decode (`promised` → resident); every already-decoding request grows
        // KV by exactly one token (per iteration, not per layer).
        if layer == self.num_layers().saturating_sub(1) {
            let mut decoding = Vec::new();
            // Index by position so each `rid` copy releases the `slots` borrow before
            // `begin_decode` / `advance_subset` take `&mut self` — same order as the
            // old cloned-`reqs` iteration, so bit-identical, but no clone.
            for k in 0..self.slots[idx].reqs.len() {
                let rid = self.slots[idx].reqs[k];
                if self.promised.contains_key(&rid) {
                    self.begin_decode(rid);
                } else {
                    decoding.push(rid);
                }
            }
            self.batch.advance_subset(&decoding);
            // KV settle point for this iteration: sample resident (current), projected
            // peak (future estimate), and reserved-not-resident (promised) occupancy.
            if self.kv.is_some() {
                let submit = KvSubmit {
                    active_kv: self.batch.kv.active_kv,
                    projected_peak: self.batch.projected_peak_kv(),
                    promised_kv: self.promised_kv(),
                };
                self.kv.as_mut().unwrap().submit(0, submit, now);
            }
        }
        // Park, do NOT advance: the pool's `SlotFlushed` (after the all-workers barrier)
        // calls `advance_slot_layer` via `on_msg_slot_flushed`.
        self.slots[idx].state = SlotState::AwaitFlush;
    }

    /// Query-token count of a micro-batch this iteration (prefill = its prompt length,
    /// decode = 1 per request), keyed off the request's `is_prefill()`.
    fn batch_query_tokens(&self, reqs: &[RequestId]) -> u64 {
        let store = self.requests.borrow();
        reqs.iter()
            .map(|&rid| {
                if store[rid].is_prefill() {
                    store[rid].prompt_len as u64
                } else {
                    1
                }
            })
            .sum()
    }

    /// Advance a finished slot to its next layer: bump `current_layer`, reset to
    /// `Wait`, and clear the per-layer pull / notify state (ref `advance_batch_layer`,
    /// minus the head rotation). Wrapping to layer 0 reopens the next iteration
    /// (clears `iter_announced` so a continuing decode re-announces via
    /// `open_iterations`).
    fn advance_slot_layer(&mut self, idx: usize) {
        let num_layers = self.model.num_layers() as u16;
        let s = &mut self.slots[idx];
        s.current_layer = (s.current_layer + 1) % num_layers;
        s.state = SlotState::Wait;
        s.notified = false;
        s.pull_send_gid = 0;
        s.pull_bytes = 0;
        if s.current_layer == 0 {
            s.iter_announced = false;
            // Wrapped past the last layer → a new forward pass for this slot opens.
            s.iter += 1;
            // Last-layer `advance_subset` grew every decode's `current_kv` and the ffn
            // Terminal may have flipped a prefill→decode at this boundary ⇒ the cached
            // attention input must be rebuilt for the new iteration's first compute.
            s.input_valid = false;
        }
    }

    /// Start the next pull and compute under the single-pull / single-compute gates:
    /// one slot may be `Pull`ing and one `Compute`ing at a time (the attn↔ffn pull
    /// overlapping the attention compute — the pipeline's whole point). ref serialises
    /// the same way off its rotating `head`; here it is a stateless scan.
    fn try_transitions(&mut self, now: Time) -> bool {
        let mut progressed = false;
        if !self.any_slot_in(SlotState::Pull) {
            if let Some(idx) = self.first_slot_in(SlotState::WaitComplete) {
                self.start_pull(idx, now);
                progressed = true;
            }
        }
        if !self.any_slot_in(SlotState::Compute) {
            if let Some(idx) = self.first_slot_in(SlotState::PullComplete) {
                self.start_compute(idx, now);
                progressed = true;
            }
        }
        progressed
    }

    fn any_slot_in(&self, state: SlotState) -> bool {
        self.slots.iter().any(|s| s.state == state)
    }

    fn first_slot_in(&self, state: SlotState) -> Option<usize> {
        (0..NUM_SLOTS).find(|&i| self.slots[i].state == state)
    }

    fn start_pull(&mut self, idx: usize, now: Time) {
        self.slots[idx].state = SlotState::Pull;
        let bytes = self.slots[idx].pull_bytes;
        if bytes == 0 {
            self.slots[idx].pull_end = now; // zero-byte handoff: nothing on the wire
            return;
        }
        let send_gid = self.slots[idx].pull_send_gid;
        let end = self
            .cluster
            .borrow_mut()
            .submit_transfer(now, send_gid, self.recv_gid, bytes, "afd_attn_pull", "");
        self.slots[idx].pull_end = end;
    }

    fn start_compute(&mut self, idx: usize, now: Time) {
        self.slots[idx].state = SlotState::Compute;
        if self.slots[idx].reqs.is_empty() {
            // Empty shards are control-plane participants only. They must emit the
            // normal layer-ready event, but zero-token shapes should not reach the
            // model/profile lookup path.
            self.slots[idx].compute_end = now;
            return;
        }
        self.build_attn_input(idx);
        let layer = self.slots[idx].current_layer as usize;
        let iter_id = self.slots[idx].iter;
        // Clone the Arc so the immutable model borrow does not alias the mutable
        // `cost` buffer borrow (mirrors the ffn worker). `run_section` evals via the
        // closure (same aggregate the timing uses) and writes one `attn` row when
        // logging — `batch_id` = the pipeline slot, `layer` = this layer.
        let model = Arc::clone(&self.model);
        let input = &self.slot_inputs[idx];
        let m = self.cost.run_section(
            "attn",
            layer as i16,
            iter_id,
            idx as u64,
            &input.groups,
            now,
            |s, sc, inp| match inp {
                Some(i) => model.attn_cost_with_inputs(layer, input, s, sc, i),
                None => model.attn_cost(layer, input, s, sc),
            },
        );
        self.slots[idx].compute_end = now + Time::from_ms(m.m.time_ms as f64);
    }

    /// Rebuild slot `idx`'s cached attention input ([`slot_inputs`](Self::slot_inputs))
    /// in place (D4: exactly one DP-shard group), unless it is still valid for this
    /// iteration. The group's `Vec` capacities persist across rebuilds. Filling reads
    /// disjoint fields (`slot_inputs` vs `slots`/`batch`/`requests`) so no `mem::replace`
    /// dance is needed. See [`Slot::input_valid`] for the invalidation points.
    fn build_attn_input(&mut self, idx: usize) {
        if self.slots[idx].input_valid {
            return; // cache hit: `slot_inputs[idx]` is up to date for this iteration
        }
        let buf = &mut self.slot_inputs[idx];
        if buf.groups.is_empty() {
            buf.groups.push(ArchGroupInput::default());
        }
        buf.groups.truncate(1);
        let g = &mut buf.groups[0];
        g.clear();
        let store = self.requests.borrow();
        for &rid in &self.slots[idx].reqs {
            if store[rid].is_prefill() {
                // Prefilling (v1 non-chunk: whole prompt in one pass): `prompt_len`
                // query tokens over `prefix_kv` KV.
                let r = &store[rid];
                g.prefill_chunk_pairs.push((r.prefix_kv, r.prompt_len));
                g.prefill_tokens += r.prompt_len;
                g.batch_tokens += r.prompt_len;
            } else {
                // Decode: one query token; KV length is this shard's tracked length.
                let cur_kv = self.batch.decode_current_kv(rid).unwrap_or(0) as u32;
                g.decode_kv_lens.push(cur_kv);
                g.decode_tokens += 1;
                g.batch_tokens += 1;
                g.total_kv_len += cur_kv;
            }
        }
        drop(store);
        self.slots[idx].input_valid = true;
    }

    /// This shard's authoritative resident KV length for a request (fed to the
    /// attention cost). `0` if not yet finalized into the decode set.
    fn current_kv(&self, rid: RequestId) -> u64 {
        // O(1) via the Batch id index — this is on the hot per-token attn-input path
        // (`build_attn_input`), where a linear scan was O(decodes) per token.
        self.batch.decode_current_kv(rid).unwrap_or(0)
    }

    fn next_wakeup(&self, now: Time) -> Option<Time> {
        let mut wake: Option<Time> = None;
        for s in &self.slots {
            let t = match s.state {
                SlotState::Compute => s.compute_end,
                SlotState::Pull => s.pull_end,
                _ => continue,
            };
            if t > now {
                wake = Some(wake.map_or(t, |c| c.min(t)));
            }
        }
        wake
    }

    fn num_layers(&self) -> u16 {
        self.model.num_layers() as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{prefilled_store, shared_with, test_cluster};
    use crate::timing::cache::interp::{CoverageFlags, Metrics4};
    use crate::timing::LeafMetrics;
    use crate::worker::gpu_cluster::SharedGpuCluster;
    use std::rc::Rc;

    struct FakeAttn {
        ms: f64,
        layers: u32,
    }
    fn lm(ms: f64) -> LeafMetrics {
        LeafMetrics {
            m: Metrics4 {
                time_ms: ms as f32,
                flops: 0.0,
                bytes: 0.0,
                energy_j: 0.0,
            },
            coverage: CoverageFlags::EMPTY,
        }
    }
    impl AttnLayerwiseModel for FakeAttn {
        fn num_layers(&self) -> u32 {
            self.layers
        }
        fn gpus_per_replica(&self) -> u16 {
            1
        }
        fn total_kv_bytes_per_token(&self) -> u64 {
            1
        }
        fn attn_to_ffn_bytes_per_token(&self) -> u64 {
            2
        }
        fn attn_cost(
            &self,
            _layer: usize,
            batch: &AttnArchInput,
            slots: &mut Vec<LeafMetrics>,
            _scratch: &mut Vec<LeafMetrics>,
        ) -> LeafMetrics {
            assert_eq!(batch.groups.len(), 1, "attn worker feeds exactly one group");
            slots.clear();
            lm(self.ms)
        }
    }

    fn worker_cfg(
        store: SharedRequests,
        cluster: SharedGpuCluster,
        config: WorkerConfig,
    ) -> DisaggAttnWorker<FakeAttn> {
        DisaggAttnWorker::new(
            WorkerId(0),
            Arc::new(FakeAttn { ms: 1.0, layers: 2 }),
            store,
            config,
            PoolId(0),
            "test-gpu",
            cluster,
            None,
            "attn",
        )
    }

    fn worker(store: SharedRequests, cluster: SharedGpuCluster) -> DisaggAttnWorker<FakeAttn> {
        worker_cfg(store, cluster, WorkerConfig::default())
    }

    fn register_test_sender(cluster: &SharedGpuCluster) -> u16 {
        let mut c = cluster.borrow_mut();
        c.allocate(99, 99, 1, "ffn-gpu");
        c.register_comm_group(0, 1, "ffn", 99)
    }

    /// All `AttnLayerOutputsReady` events emitted so far (the per-layer handoffs),
    /// filtering out the `IterStart` announcements.
    fn outputs(events: &[AttnWorkerEvent]) -> Vec<&AttnWorkerEvent> {
        events
            .iter()
            .filter(|e| matches!(e, AttnWorkerEvent::AttnLayerOutputsReady { .. }))
            .collect()
    }

    fn iter_starts(events: &[AttnWorkerEvent]) -> Vec<&AttnWorkerEvent> {
        events
            .iter()
            .filter(|e| matches!(e, AttnWorkerEvent::IterStart { .. }))
            .collect()
    }

    /// Drive the micro-batch in `slot` through every layer of `w`, one slot-addressed
    /// notification per layer, accumulating emitted events. After each layer's compute
    /// the slot parks in `AwaitFlush`; the pool's per-layer barrier would release it
    /// with a `SlotFlushed`, so this helper feeds one per layer (the last layer's flush
    /// wraps the slot to layer 0 and reopens the next iteration). One layer per flush.
    fn run_iteration(
        w: &mut DisaggAttnWorker<FakeAttn>,
        slot: u8,
        sender: u16,
        bytes: u64,
        base_ms: u64,
        events: &mut Vec<AttnWorkerEvent>,
    ) {
        for layer in 0u16..2 {
            w.enqueue(AttnWorkerMsg::ReadyNotification {
                slot,
                layer,
                send_gid: sender,
                bytes,
            });
            for step in 0..60u64 {
                w.tick(
                    Time::from_ms((base_ms + (layer as u64) * 60 + step) as f64),
                    events,
                );
            }
            // Slot is now parked in AwaitFlush — release it one layer (and, on the last
            // layer, tick once more so `open_iterations` re-announces after the wrap).
            w.enqueue(AttnWorkerMsg::SlotFlushed { slot, layer });
            w.tick(
                Time::from_ms((base_ms + (layer as u64) * 60 + 60) as f64),
                events,
            );
        }
    }

    /// Admit only enqueues (Level-1). The Level-2 drain + iteration open runs in tick,
    /// a prefilled handoff finalizes to resident decode at activation, and the slot
    /// announces its iteration with an `IterStart`.
    #[test]
    fn admit_drains_reserves_kv_and_announces() {
        let store = prefilled_store(&[(0, 8, 4)]);
        let cluster = test_cluster();
        let mut w = worker(Rc::clone(&store), Rc::clone(&cluster));

        w.enqueue(AttnWorkerMsg::Admit { req: RequestId(0) });
        assert_eq!(
            w.batch.kv.active_kv, 0,
            "Admit alone reserves no resident KV"
        );
        assert!(w.promised.is_empty());

        let mut events = Vec::new();
        w.tick(Time::ZERO, &mut events);

        // Drain admitted it (prefilled handoff ⇒ finalized at activation): resident
        // KV = prompt 8, in the decode set, sticky to slot 0.
        assert_eq!(w.batch.kv.active_kv, 8);
        assert!(w.batch.decodes.iter().any(|(r, _)| *r == RequestId(0)));
        assert_eq!(w.request_to_slot.get(&RequestId(0)), Some(&0));
        assert!(
            !w.promised.contains_key(&RequestId(0)),
            "finalized ⇒ not promised"
        );
        assert!(w.slots[0].reqs.contains(&RequestId(0)));
        // Every slot reports the layer-(-1) boundary; slot 0 carries the admitted
        // request and the other slots are empty sync participants.
        assert_eq!(
            iter_starts(&events),
            vec![
                &AttnWorkerEvent::IterStart {
                    worker: WorkerId(0),
                    slot: 0,
                    reqs: vec![RequestId(0)],
                },
                &AttnWorkerEvent::IterStart {
                    worker: WorkerId(0),
                    slot: 1,
                    reqs: Vec::new(),
                },
                &AttnWorkerEvent::IterStart {
                    worker: WorkerId(0),
                    slot: 2,
                    reqs: Vec::new(),
                },
            ]
        );
        assert!(w.slots[0].iter_announced);
    }

    /// One micro-batch walks all layers of an iteration: one handoff per layer (each
    /// carrying the shard's send_gid), and KV grows by exactly 1 after the last layer
    /// (per iteration, not per layer).
    #[test]
    fn one_microbatch_walks_all_layers_and_grows_kv_once() {
        let store = prefilled_store(&[(0, 8, 4)]);
        let cluster = test_cluster();
        let sender = register_test_sender(&cluster);
        let mut w = worker(Rc::clone(&store), Rc::clone(&cluster));

        w.enqueue(AttnWorkerMsg::Admit { req: RequestId(0) });
        let mut events = Vec::new();
        w.tick(Time::ZERO, &mut events); // drain + open iteration
        let kv_after_activate = w.batch.kv.active_kv;
        assert_eq!(kv_after_activate, 8);
        let slot = w.request_to_slot[&RequestId(0)] as u8;

        run_iteration(&mut w, slot, sender, 64, 1, &mut events);

        // One handoff per layer, batch-granular, bytes = attn_to_ffn(2) × 1 token,
        // send_gid = this shard's recv/send endpoint.
        let outs = outputs(&events);
        assert_eq!(outs.len(), 2);
        for (layer, e) in outs.iter().enumerate() {
            assert_eq!(
                *e,
                &AttnWorkerEvent::AttnLayerOutputsReady {
                    worker: WorkerId(0),
                    slot,
                    reqs: vec![RequestId(0)],
                    tokens: 1,
                    layer: layer as u16,
                    send_gid: w.recv_gid,
                    bytes: 2,
                }
            );
        }
        // KV grew by exactly 1 (one iteration), not by 2 (num_layers).
        assert_eq!(w.batch.kv.active_kv, kv_after_activate + 1);
        assert_eq!(w.current_kv(RequestId(0)), 9);
    }

    /// Decode loop-back: after the last layer the slot wraps to layer 0 and
    /// re-announces a fresh `IterStart` for the continuing decode (clearing
    /// `iter_announced` reopens the iteration).
    #[test]
    fn decode_loopback_reannounces_iter_start() {
        let store = prefilled_store(&[(0, 8, 4)]);
        let cluster = test_cluster();
        let sender = register_test_sender(&cluster);
        let mut w = worker(Rc::clone(&store), Rc::clone(&cluster));
        w.enqueue(AttnWorkerMsg::Admit { req: RequestId(0) });
        let mut events = Vec::new();
        w.tick(Time::ZERO, &mut events); // open iteration 1
        let slot = w.request_to_slot[&RequestId(0)] as u8;

        run_iteration(&mut w, slot, sender, 64, 1, &mut events);

        // Slot 0 announces iteration 1 (at admit) and iteration 2 (decode loop-back
        // after the last-layer wrap). Empty slots also reported their initial boundary.
        let slot0_starts = iter_starts(&events)
            .into_iter()
            .filter(|e| matches!(e, AttnWorkerEvent::IterStart { slot: 0, .. }))
            .count();
        assert_eq!(slot0_starts, 2, "slot 0 announces once per iteration");
        assert_eq!(w.slots[slot as usize].current_layer, 0);
        assert!(w.slots[slot as usize].reqs.contains(&RequestId(0)));
        assert!(w.slots[slot as usize].iter_announced);
    }

    /// An idle worker reports each slot's layer-(-1) boundary once, then quiesces
    /// until the pool sends the no-op Bootstrap / layer notifications.
    #[test]
    fn empty_slots_report_start_boundary_once() {
        let store = prefilled_store(&[(0, 8, 4)]);
        let cluster = test_cluster();
        let mut w = worker(Rc::clone(&store), Rc::clone(&cluster));
        let mut events = Vec::new();
        let wake = w.tick(Time::ZERO, &mut events);
        assert_eq!(wake, None, "start boundary reports need no timed wakeup");
        assert!(
            iter_starts(&events).len() == NUM_SLOTS,
            "each empty slot reports the layer-(-1) boundary once"
        );
        assert!(w
            .slots
            .iter()
            .all(|s| s.state == SlotState::Wait && s.iter_announced));

        events.clear();
        let wake = w.tick(Time::from_ms(1.0), &mut events);
        assert_eq!(wake, None);
        assert!(
            events.is_empty(),
            "already-reported empty slots do not spin"
        );
    }

    /// An EMPTY active slot — one the controller notified so it stays a barrier member
    /// — runs each layer as a 0-token no-op but PARKS in `AwaitFlush`; it does NOT burst
    /// through every layer in one tick (the failure mode an empty, vacuously
    /// `input_ready` slot would hit without the gate). Exactly one `SlotFlushed`
    /// advances exactly one layer.
    #[test]
    fn empty_slot_advances_one_layer_per_flush() {
        let store = prefilled_store(&[(0, 8, 4)]); // a request exists but is never admitted
        let cluster = test_cluster();
        let mut w = worker(Rc::clone(&store), Rc::clone(&cluster));
        let mut events = Vec::new();

        // Notify slot 0 at layer 0 with no resident requests → a 0-token no-op layer.
        w.enqueue(AttnWorkerMsg::ReadyNotification {
            slot: 0,
            layer: 0,
            send_gid: 0,
            bytes: 0,
        });
        for step in 0..10u64 {
            w.tick(Time::from_ms(step as f64), &mut events);
        }

        // It ran layer 0 and PARKED — it did NOT burst ahead to layer 1.
        assert_eq!(w.slots[0].state, SlotState::AwaitFlush);
        assert_eq!(
            w.slots[0].current_layer, 0,
            "parked at its layer, did not burst"
        );
        assert_eq!(
            outputs(&events).len(),
            1,
            "the 0-token layer still reports to the barrier"
        );
        match outputs(&events)[0] {
            AttnWorkerEvent::AttnLayerOutputsReady {
                reqs,
                tokens,
                bytes,
                layer,
                ..
            } => {
                assert!(reqs.is_empty(), "empty report carries no requests");
                assert_eq!(*tokens, 0, "empty slot ⇒ 0 query tokens");
                assert_eq!(*bytes, 0, "0-token layer ⇒ 0 handoff bytes");
                assert_eq!(*layer, 0);
            }
            _ => unreachable!(),
        }

        // One SlotFlushed advances exactly one layer (no burst, no skip).
        w.enqueue(AttnWorkerMsg::SlotFlushed { slot: 0, layer: 0 });
        assert_eq!(w.slots[0].current_layer, 1);
        assert_eq!(w.slots[0].state, SlotState::Wait);
    }

    /// Release frees the request's KV and drops its slot back to dormant.
    #[test]
    fn release_frees_kv_and_empties_slot() {
        // decode_len 2 with prefilled `tokens_emitted == 1` ⇒ one decode token left,
        // so the request actually enters the decode set (decode_len 1 would have zero
        // remaining and be dropped at admit — never resident to release).
        let store = prefilled_store(&[(0, 8, 2)]);
        let cluster = test_cluster();
        let mut w = worker(Rc::clone(&store), Rc::clone(&cluster));
        w.enqueue(AttnWorkerMsg::Admit { req: RequestId(0) });
        let mut events = Vec::new();
        w.tick(Time::ZERO, &mut events);
        assert!(w.batch.decodes.iter().any(|(r, _)| *r == RequestId(0)));

        w.enqueue(AttnWorkerMsg::Release { req: RequestId(0) });
        assert!(!w.batch.decodes.iter().any(|(r, _)| *r == RequestId(0)));
        assert_eq!(w.batch.kv.active_kv, 0);
        assert!(w.slots.iter().all(|s| s.reqs.is_empty()));
        assert!(w.request_to_slot.is_empty());
    }

    /// KV-full: the projected-peak gate blocks the second request head-of-line; it
    /// stays in `worker_pending` until a Release frees capacity, then drains.
    #[test]
    fn kv_full_head_of_line_then_admits_after_release() {
        // cap = 15 tokens. req0 = prompt 8 + decode 4 (peak 12); req1 = prompt 8 +
        // decode 4 would push the projected peak over 15 → blocked until req0 frees.
        let store = prefilled_store(&[(0, 8, 4), (1, 8, 4)]);
        let cluster = test_cluster();
        let mut cfg = WorkerConfig::default();
        cfg.attn_kv_bytes = 15; // total_kv_bytes_per_token = 1 ⇒ capacity 15 tokens
        let mut w = worker_cfg(Rc::clone(&store), Rc::clone(&cluster), cfg);

        w.enqueue(AttnWorkerMsg::Admit { req: RequestId(0) });
        w.enqueue(AttnWorkerMsg::Admit { req: RequestId(1) });
        let mut events = Vec::new();
        w.tick(Time::ZERO, &mut events);

        assert!(
            w.batch.decodes.iter().any(|(r, _)| *r == RequestId(0)),
            "req0 admitted"
        );
        assert!(
            !w.batch.decodes.iter().any(|(r, _)| *r == RequestId(1)),
            "req1 blocked"
        );
        assert_eq!(
            w.worker_pending.front(),
            Some(RequestId(1)),
            "req1 head-of-line"
        );

        // Free req0 → req1 now fits and drains in on the next tick.
        w.enqueue(AttnWorkerMsg::Release { req: RequestId(0) });
        w.tick(Time::from_ms(1.0), &mut events);
        assert!(
            w.batch.decodes.iter().any(|(r, _)| *r == RequestId(1)),
            "req1 admitted after release"
        );
        assert!(w.worker_pending.is_empty());
    }

    /// The least-KV load balancer spreads two same-size admits across two slots.
    #[test]
    fn wlb_picks_least_kv_slot() {
        let store = prefilled_store(&[(0, 8, 4), (1, 8, 4)]);
        let cluster = test_cluster();
        let mut w = worker(Rc::clone(&store), Rc::clone(&cluster));
        w.enqueue(AttnWorkerMsg::Admit { req: RequestId(0) });
        w.enqueue(AttnWorkerMsg::Admit { req: RequestId(1) });
        let mut events = Vec::new();
        w.tick(Time::ZERO, &mut events);

        let s0 = w.request_to_slot[&RequestId(0)];
        let s1 = w.request_to_slot[&RequestId(1)];
        assert_ne!(
            s0, s1,
            "least-KV spreads the two requests onto different slots"
        );
    }

    /// A fresh (un-prefilled) request prefills across all layers before entering
    /// decode: no resident KV at activation, finalized at the last prefill layer.
    #[test]
    fn fresh_request_prefills_then_finalizes() {
        let store = shared_with(&[(0, 8, 4)]); // prefill_processed = 0 ⇒ is_prefill
        let cluster = test_cluster();
        let sender = register_test_sender(&cluster);
        let mut w = worker(Rc::clone(&store), Rc::clone(&cluster));
        w.enqueue(AttnWorkerMsg::Admit { req: RequestId(0) });
        let mut events = Vec::new();
        w.tick(Time::ZERO, &mut events);

        // Activated but NOT finalized: still reserved in `promised`, no resident KV.
        assert!(w.slots[0].reqs.contains(&RequestId(0)));
        assert!(
            w.promised.contains_key(&RequestId(0)),
            "fresh ⇒ still promised"
        );
        assert_eq!(w.batch.kv.active_kv, 0, "prefill KV is not resident yet");
        let slot = w.request_to_slot[&RequestId(0)] as u8;

        run_iteration(&mut w, slot, sender, 64, 1, &mut events);

        // After the prefill pass (last layer) it finalized into decode: resident KV
        // = prompt 8, no longer promised.
        assert!(
            !w.promised.contains_key(&RequestId(0)),
            "finalized at last prefill layer"
        );
        assert_eq!(w.batch.kv.active_kv, 8);
        assert!(w.batch.decodes.iter().any(|(r, _)| *r == RequestId(0)));
    }
}
