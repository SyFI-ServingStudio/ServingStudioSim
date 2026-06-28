//! `DisaggAttnWorker` — the attn half of an AFD (attention-FFN disaggregation)
//! deployment. It owns ONLY the per-layer attention kernel + the KV cache for its
//! DP shard; everything else (qkv / o_proj / router / MoE) lives on the ffn side.
//!
//! **One worker = one DP shard, one shared KV pool** (D4/D11). It runs a **circular
//! 3-slot pipeline** with a single `head` pointer (ported faithfully from
//! `ref/moesim-rs` attn_worker), so attn compute overlaps the attn↔ffn pull:
//!
//!   `head` = Compute slot · `head+1` = Pull slot · `head+2` = Wait slot
//!
//! Notifications are **slot-addressed**: the worker owns slot assignment, stamps
//! the slot tag on each `AttnLayerOutputsReady`, and L6 echoes it in the next
//! `ReadyNotification`, so the per-layer handshake routes to a slot in O(1) (no
//! request set on the wire) and readiness is a single per-slot flag. (The layer-0
//! bootstrap tag round-trip is a Phase-4 flow concern; the worker assumes a live
//! tag.)
//!
//! Each slot is a persistent micro-batch walking the layers through the state
//! machine `Wait → WaitComplete → Pull → PullComplete → Compute → LayerDone`. The
//! head only advances (`(head+1)%3`) when its slot reaches `LayerDone`, rotating
//! the just-computed slot to the back. All three slots are **always in the ring**:
//! an empty slot is `input_ready` by definition, so it flows through as a no-op
//! (instant 0-byte pull, instant empty compute) to let the head rotate past it.
//! A non-empty slot waiting for its layer notification is *not* `input_ready`, so
//! it stalls in `Wait` and halts the rotation when the head reaches it — that
//! notification gate is what paces the ring and bounds empty-slot churn. A fully
//! idle worker (no live reqs, nothing pending) does no work at all.
//!
//! Admission is the shared two-level scheme (ref / hp_unified): `Admit` only
//! enqueues to `worker_pending` (Level-1); each tick `drain_pending_admits`
//! (Level-2) runs the projected-peak KV gate (`KvAdmission::try_admit`) head-of-
//! line — a rejected request blocks the queue and retries when a `Release` frees
//! KV, never "admit + warn". An admitted request reserves its FULL footprint
//! (`prompt_len + prefix_kv + decode_len`, since AFD prefill is NOT done at admit)
//! in `promised`, picks its least-KV slot (`wlb`), and lands in that slot's
//! `pending_insert`. It joins `reqs` at the slot's next layer-0 activation.
//!
//! KV accounting is two-phase. `promised` holds reserved-but-not-resident KV;
//! `Batch` (the shared shard KV pool + decode set) holds resident KV. A request
//! leaves `promised` and enters `Batch.decodes` (`finalize_to_decode`) at its
//! prefill→decode boundary: a prefilled handoff (`!is_prefill()` at admit) at
//! activation; a fresh prefill at the last layer of its prefill pass. The
//! prefill/decode **discriminator**, though, is the request's authoritative store
//! status (`is_prefill()`, which the ffn Terminal advances at the boundary), NOT
//! `promised` membership — `promised` is the reserved-KV admission accounting only.
//! (This attn worker still never writes the store; the ffn owns the status flip.)
//! A decode token then adds one KV slot per
//! request **per iteration**, applied via `advance_subset` exactly when a slot
//! finishes its **last** layer (the iteration boundary) — per layer would
//! over-count by `num_layers`.
//!
//! Completion is NOT decided here (D16): the ffn Terminal owns it and the flow
//! routes a `Release`; the worker just drops the KV.
//!
//! Reading order: types → construction → message handling (the `IterWorker` entry
//! points) → the tick / pipeline loop → its helpers, each following the function
//! that calls it → tests.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, AttnArchInput, AttnLayerwiseModel};
use crate::common::{PoolId, RequestId, SharedRequests, Time, WorkerId};
use crate::timing::LeafMetrics;
use crate::worker::admission_helpers::Batch;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::iter_worker::IterWorker;
use crate::worker::types::{AttnWorkerEvent, AttnWorkerMsg, WorkerConfig, WorkerStatus};

// ── types ─────────────────────────────────────────────────────────────────────

/// Pipeline depth — circular slots overlapping pull against compute (ref fixes 3).
const NUM_SLOTS: usize = 3;

/// Per-slot pipeline stage (ref `BatchState`). A slot owns exactly one compute /
/// pull resource only while at `Compute` / `Pull`, and the head-relative
/// transitions guarantee ≤1 of each across the worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    /// Waiting for this layer's notification (or empty ⇒ trivially ready).
    Wait,
    /// Notification in hand (or empty); ready to pull.
    WaitComplete,
    /// QKV input for `current_layer` is transferring.
    Pull,
    /// Pull landed; ready to compute.
    PullComplete,
    /// Attention for `current_layer` is running.
    Compute,
    /// Layer finished (output emitted); waiting for the head to rotate it on.
    LayerDone,
}

struct Slot {
    /// Micro-batch membership. Empty ⇒ an idle slot that still rides the ring as a
    /// no-op. Grows at layer-0 activation, shrinks as requests are released.
    reqs: Vec<RequestId>,
    /// Admitted (Level-2) and assigned here, awaiting this slot's next layer-0
    /// activation to join `reqs`.
    pending_insert: Vec<RequestId>,
    /// The layer this slot is processing (advances `(cur + 1) % num_layers` at
    /// `LayerDone`). Used only to match notifications, never to order the ring.
    current_layer: u16,
    /// Set true when layer-0 pull starts (membership locked for this iteration);
    /// reopened when the slot wraps back to layer 0.
    closed: bool,
    state: SlotState,
    /// Slot-level notification flag (D-H): the whole micro-batch's layer input is
    /// ready. Empty slots are treated as notified (`input_ready`).
    notified: bool,
    /// Pull descriptor carried from the notification until `start_pull` consumes it.
    pull_send_gid: u16,
    pull_bytes: u64,
    /// Live event timestamps for the current `Pull` / `Compute`.
    pull_end: Time,
    compute_end: Time,
}

impl Slot {
    fn new() -> Self {
        Self {
            reqs: Vec::new(),
            pending_insert: Vec::new(),
            current_layer: 0,
            closed: false,
            state: SlotState::Wait,
            notified: false,
            pull_send_gid: 0,
            pull_bytes: 0,
            pull_end: Time::ZERO,
            compute_end: Time::ZERO,
        }
    }

    /// This slot's layer input is ready, so it may leave `Wait`: its notification
    /// has arrived — or it is empty (empty batches never block; ref `all_notified`).
    fn input_ready(&self) -> bool {
        self.reqs.is_empty() || self.notified
    }
}

pub struct DisaggAttnWorker<M: AttnLayerwiseModel> {
    pub id: WorkerId,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cluster: SharedGpuCluster,
    /// This shard's recv endpoint — the QKV pull from the ffn lands here.
    recv_gid: u16,
    /// One shard-level `Batch` shared across all slots (the slots are pipelined
    /// micro-batches, NOT DP shards — D11). It bundles the KV pool with the
    /// per-request decode state (`current_kv` + `remaining_decode`); resident KV
    /// only. Reserved-not-resident KV lives in `promised`.
    batch: Batch,
    slots: [Slot; NUM_SLOTS],
    /// Pipeline head: `head` = Compute slot, `head+1` = Pull, `head+2` = Wait.
    head: usize,
    /// Level-1 admission backlog (FIFO). Drained under the KV gate each tick.
    worker_pending: VecDeque<RequestId>,
    /// Reserved-but-not-resident KV per request (full footprint). Admission
    /// accounting only — feeds the `try_admit` group-promised total; a request
    /// leaves it at `finalize_to_decode`. (NOT the prefill/decode discriminator:
    /// that is the request's store `is_prefill()` status.)
    promised: HashMap<RequestId, u64>,
    /// Sticky request → slot routing (set at admission; survives decode re-entry).
    request_to_slot: HashMap<RequestId, usize>,
    /// Own eval buffers (cost_log deferred — D10, no `CostBuffers`).
    slots_buf: Vec<LeafMetrics>,
    scratch: Vec<LeafMetrics>,
}

// ── construction ────────────────────────────────────────────────────────────────

impl<M: AttnLayerwiseModel> DisaggAttnWorker<M> {
    pub fn new(
        id: WorkerId,
        model: Arc<M>,
        requests: SharedRequests,
        config: WorkerConfig,
        pool: PoolId,
        gpu_name: &str,
        cluster: SharedGpuCluster,
    ) -> Self {
        let recv_gid = {
            let mut c = cluster.borrow_mut();
            let base = c.allocate(pool.0, id.0, model.gpus_per_replica(), gpu_name);
            // The attn-shard prefix is the comm group used as the QKV recv endpoint.
            c.register_comm_group(base, model.num_attn_shards().max(1))
        };
        // KV capacity in tokens: per-GPU budget × the shard's GPUs / total wire KV
        // bytes per token (same sizing as the PD decode worker).
        let shard_bytes = config
            .attn_kv_bytes
            .saturating_mul(model.num_attn_shards().max(1) as u64);
        let kv_capacity = (shard_bytes / model.total_kv_bytes_per_token().max(1)).max(1);
        Self {
            id,
            model,
            requests,
            config,
            cluster,
            recv_gid,
            batch: Batch::new(0, kv_capacity),
            slots: std::array::from_fn(|_| Slot::new()),
            head: 0,
            worker_pending: VecDeque::new(),
            promised: HashMap::new(),
            request_to_slot: HashMap::new(),
            slots_buf: Vec::new(),
            scratch: Vec::new(),
        }
    }
}

// ── message handling (entry points) ─────────────────────────────────────────────
//
// The worker's whole external surface: `enqueue` receives the three messages
// (`Admit` / `ReadyNotification` / `Release`) and dispatches each to a helper;
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

    /// Level-1 enqueue (ref `enqueue_to_pending`): record the request as pending.
    /// No KV is reserved yet — the projected-peak gate runs at the Level-2 drain so
    /// a rejected request blocks the queue instead of being force-admitted.
    /// Idempotent: a re-`Admit` of a request already pending / promised / placed is
    /// dropped.
    fn on_msg_admit(&mut self, req: RequestId) {
        let known = self.worker_pending.contains(&req)
            || self.promised.contains_key(&req)
            || self.request_to_slot.contains_key(&req);
        if !known {
            self.worker_pending.push_back(req);
        }
    }

    /// Drop a completed request's KV (the ffn Terminal completed it; the flow
    /// routed the release here). Removes it from every level it can sit at; a slot
    /// that empties stays in the ring (it just rides as a no-op).
    fn on_msg_release(&mut self, req: RequestId) {
        if let Some(current_kv) = self
            .batch
            .decodes
            .iter()
            .find(|(r, _)| *r == req)
            .map(|(_, s)| s.current_kv)
        {
            self.batch.release(req, current_kv);
        }
        self.promised.remove(&req);
        self.worker_pending.retain(|&r| r != req);
        if let Some(slot_idx) = self.request_to_slot.remove(&req) {
            let s = &mut self.slots[slot_idx];
            s.reqs.retain(|&r| r != req);
            s.pending_insert.retain(|&r| r != req);
        }
    }

    /// Slot-addressed notification (D-H): the micro-batch in slot `slot` has its
    /// layer input ready. Routed in O(1) by the tag L6 echoed back — no request
    /// resolution, a single per-slot `notified` flag (the whole notification IS the
    /// slot). The ffn drives layers in lockstep (layer L's QKV is produced only
    /// after the attn emits layer L-1), so a notification ALWAYS finds its slot
    /// already waiting at exactly that layer — there is no out-of-order arrival to
    /// buffer. A mismatch is a tag / protocol violation, not a retry.
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

    // ── tick & pipeline loop ─────────────────────────────────────────────────────

    fn tick_inner(&mut self, now: Time, events: &mut Vec<AttnWorkerEvent>) -> Option<Time> {
        // Idle guard: with no live reqs and nothing queued there is no work
        // to pace the ring, so empty slots must not churn — do nothing.
        if !self.has_work() {
            return None;
        }
        self.drain_pending_admits();
        // The ring is paced by non-empty slots. If the drain landed nothing in a slot
        // — the head-of-line admit was KV-blocked, or the request had no decode work
        // left (`remaining == 0`) and was dropped — every slot is empty, and running
        // the loop would spin the empty ring forever (each empty slot rotates as an
        // instant no-op with no notification gate to stop it). Bail; a later Release /
        // Notification re-ticks us once there is real slot work. (The entry guard does
        // not cover this: it can pass on `worker_pending` alone, which the drain may
        // then empty without placing anything.)
        if !self.has_slot_work() {
            return None;
        }
        // Pipeline to a fixpoint. Each pass mirrors ref's step() + advance_batch_layer:
        // activate → completions → head transitions → rotate head.
        loop {
            let mut progressed = false;
            progressed |= self.activate_slots();
            progressed |= self.advance_completions(now, events);
            progressed |= self.try_transitions(now);
            progressed |= self.rotate_head();
            if !progressed {
                break;
            }
        }
        self.next_wakeup(now)
    }

    /// Any work that should drive the ring? Empty slots alone (idle worker) must
    /// not — they would churn forever with no notification gate to pace them.
    fn has_work(&self) -> bool {
        !self.worker_pending.is_empty() || self.has_slot_work()
    }

    /// A slot holds (or is about to hold) a request — the only thing that paces the
    /// ring. `worker_pending` alone does NOT: a drained-to-nothing or KV-blocked
    /// backlog leaves the slots empty, and an empty ring must not churn.
    fn has_slot_work(&self) -> bool {
        self.slots
            .iter()
            .any(|s| !s.reqs.is_empty() || !s.pending_insert.is_empty())
    }

    /// Level-2 drain (ref `drain_worker_pending_to_batches`): admit from the head of
    /// `worker_pending` under the projected-peak KV gate, head-of-line (stop at the
    /// first reject, do not skip). Each admit reserves the FULL footprint in
    /// `promised` and assigns the least-KV slot.
    fn drain_pending_admits(&mut self) {
        while let Some(&req) = self.worker_pending.front() {
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

    /// Least-KV slot (D-D): the load balancer balances reserved+resident KV across
    /// slots (the user's metric; ref balances request count). Ties pick the lowest
    /// index — irrelevant to ordering, slots are addressed by identity only.
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

    /// Activate layer-0 slots (ref :435): once a slot is back at layer 0 and open,
    /// its `pending_insert` joins `reqs`. A prefilled handoff (`!is_prefill()`)
    /// becomes resident decode now; a fresh prefill stays in `promised` until the
    /// last layer of its prefill pass.
    fn activate_slots(&mut self) -> bool {
        let mut progressed = false;
        for idx in 0..NUM_SLOTS {
            let s = &self.slots[idx];
            if !(s.current_layer == 0
                && !s.closed
                && s.state == SlotState::Wait
                && !s.pending_insert.is_empty())
            {
                continue;
            }
            let newly: Vec<RequestId> = self.slots[idx].pending_insert.drain(..).collect();
            for rid in newly {
                self.slots[idx].reqs.push(rid);
                if !self.req_is_prefill(rid) {
                    self.begin_decode(rid); // prefilled handoff: resident decode immediately
                }
                progressed = true;
            }
        }
        progressed
    }

    /// Event-time completions (ref :484): a notified `Wait` slot becomes ready, a
    /// landed pull becomes ready, a finished compute emits + closes its layer.
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
                    self.complete_layer(idx, events); // sets LayerDone
                    progressed = true;
                }
                _ => {}
            }
        }
        progressed
    }

    /// Head-relative transitions (ref :511): the head computes (or, fresh off a
    /// rotation, pulls), and the next slot pulls while the head computes — the
    /// pull/compute overlap. Single pull / single compute fall out of "pull only on
    /// head/next, compute only on head".
    fn try_transitions(&mut self, now: Time) -> bool {
        let head = self.head;
        let next = (head + 1) % NUM_SLOTS;
        let mut progressed = false;
        if self.slots[head].state == SlotState::PullComplete {
            self.start_compute(head, now);
            progressed = true;
        }
        if self.slots[head].state == SlotState::WaitComplete {
            self.start_pull(head, now);
            progressed = true;
        }
        if matches!(
            self.slots[head].state,
            SlotState::Compute | SlotState::LayerDone
        ) && self.slots[next].state == SlotState::WaitComplete
        {
            self.start_pull(next, now);
            progressed = true;
        }
        progressed
    }

    /// Advance the head when its slot has finished its layer (ref `advance_batch_layer`):
    /// bump the slot's layer (reopening membership at the layer-0 wrap) and rotate
    /// the head to the next slot.
    fn rotate_head(&mut self) -> bool {
        let idx = self.head;
        if self.slots[idx].state != SlotState::LayerDone {
            return false;
        }
        let num_layers = self.model.num_layers() as u16;
        let s = &mut self.slots[idx];
        s.current_layer = (s.current_layer + 1) % num_layers;
        s.state = SlotState::Wait;
        s.notified = false;
        s.pull_bytes = 0;
        s.pull_send_gid = 0;
        if s.current_layer == 0 {
            s.closed = false;
        }
        self.head = (self.head + 1) % NUM_SLOTS;
        true
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

    // ── per-slot work: pull → compute → complete ────────────────────────────────

    fn start_pull(&mut self, idx: usize, now: Time) {
        if self.slots[idx].current_layer == 0 {
            self.slots[idx].closed = true;
        }
        self.slots[idx].state = SlotState::Pull;
        // Empty slots (and zero-byte handoffs) pull instantly — the ring just needs
        // them to flow so the head can rotate.
        if self.slots[idx].reqs.is_empty() || self.slots[idx].pull_bytes == 0 {
            self.slots[idx].pull_end = now;
            return;
        }
        let (send_gid, bytes) = (self.slots[idx].pull_send_gid, self.slots[idx].pull_bytes);
        let end = self
            .cluster
            .borrow_mut()
            .submit_transfer(now, send_gid, self.recv_gid, bytes);
        self.slots[idx].pull_end = end;
    }

    fn start_compute(&mut self, idx: usize, now: Time) {
        self.slots[idx].state = SlotState::Compute;
        if self.slots[idx].reqs.is_empty() {
            self.slots[idx].compute_end = now; // empty slot: instant no-op, no GPU submit
            return;
        }
        let input = self.build_attn_input(idx);
        let layer = self.slots[idx].current_layer as usize;
        // Clone the Arc so the immutable model borrow does not alias the mutable
        // slots_buf / scratch borrows (mirrors the ffn worker).
        let model = Arc::clone(&self.model);
        let m = model.attn_cost(layer, &input, &mut self.slots_buf, &mut self.scratch);
        self.slots[idx].compute_end = now + Time::from_ms(m.m.time_ms as f64);
    }

    fn build_attn_input(&self, idx: usize) -> AttnArchInput {
        let store = self.requests.borrow();
        let mut g = ArchGroupInput::default();
        for &rid in &self.slots[idx].reqs {
            if self.req_is_prefill(rid) {
                // Prefilling (v1 non-chunk: whole prompt in one pass): `prompt_len`
                // query tokens over `prefix_kv` KV.
                let r = &store[rid];
                g.prefill_chunk_pairs.push((r.prefix_kv, r.prompt_len));
                g.prefill_tokens += r.prompt_len;
                g.batch_tokens += r.prompt_len;
            } else {
                // Decode: one query token; KV length is this shard's tracked length.
                let cur_kv = self.current_kv(rid) as u32;
                g.decode_kv_lens.push(cur_kv);
                g.decode_tokens += 1;
                g.batch_tokens += 1;
                g.total_kv_len += cur_kv;
            }
        }
        // D4: the attn worker computes exactly one DP-shard group.
        AttnArchInput { groups: vec![g] }
    }

    /// This shard's authoritative resident KV length for a request (fed to the
    /// attention cost). `0` if not yet finalized into the decode set.
    fn current_kv(&self, rid: RequestId) -> u64 {
        self.batch
            .decodes
            .iter()
            .find(|(r, _)| *r == rid)
            .map_or(0, |(_, s)| s.current_kv)
    }

    /// Finish a slot's layer: emit the handoff, and at the iteration boundary move
    /// finished prefills into decode and grow the decoders' KV by one.
    fn complete_layer(&mut self, idx: usize, events: &mut Vec<AttnWorkerEvent>) {
        let layer = self.slots[idx].current_layer;
        let reqs = self.slots[idx].reqs.clone();
        if !reqs.is_empty() {
            // Outgoing handoff: producer-computed here so the ffn receiver reads it
            // off the message (never recomputes).
            let tokens = self.batch_query_tokens(&reqs);
            let out_bytes = self.model.attn_to_ffn_bytes_per_token() * tokens;
            events.push(AttnWorkerEvent::AttnLayerOutputsReady {
                worker: self.id,
                slot: idx as u8,
                reqs: reqs.clone(),
                layer,
                bytes: out_bytes,
            });

            // Iteration boundary (last layer): a fresh prefill that just finished its
            // pass enters decode (`promised` → resident); every already-decoding
            // request grows KV by exactly one token (per iteration, not per layer).
            if layer == self.num_layers().saturating_sub(1) {
                let mut decoding = Vec::new();
                for &rid in &reqs {
                    if self.promised.contains_key(&rid) {
                        self.begin_decode(rid);
                    } else {
                        decoding.push(rid);
                    }
                }
                self.batch.advance_subset(&decoding);
            }
        }
        self.slots[idx].state = SlotState::LayerDone;
    }

    /// Move a request from reserved (`promised`) to resident decode (ref's
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
        self.batch.finalize_to_decode(rid, initial_kv, decode_budget);
        self.promised.remove(&rid);
    }

    /// Query-token count of a micro-batch this iteration (prefill = its prompt
    /// length, decode = 1 per request), keyed off the request's `is_prefill()`.
    fn batch_query_tokens(&self, reqs: &[RequestId]) -> u64 {
        let store = self.requests.borrow();
        reqs.iter()
            .map(|&rid| {
                if self.req_is_prefill(rid) {
                    store[rid].prompt_len as u64
                } else {
                    1
                }
            })
            .sum()
    }

    fn req_is_prefill(&self, rid: RequestId) -> bool {
        self.requests.borrow()[rid].is_prefill()
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
        )
    }

    fn worker(store: SharedRequests, cluster: SharedGpuCluster) -> DisaggAttnWorker<FakeAttn> {
        worker_cfg(store, cluster, WorkerConfig::default())
    }

    fn register_test_sender(cluster: &SharedGpuCluster) -> u16 {
        let mut c = cluster.borrow_mut();
        c.allocate(99, 99, 1, "ffn-gpu");
        c.register_comm_group(0, 1)
    }

    /// Drive the micro-batch in `slot` through every layer of `w`, one slot-addressed
    /// notification per layer, accumulating emitted events.
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
                w.tick(Time::from_ms((base_ms + (layer as u64) * 60 + step) as f64), events);
            }
        }
    }

    /// Admit only enqueues (Level-1). The Level-2 drain + activation runs in tick,
    /// and a prefilled handoff finalizes to resident decode at activation.
    #[test]
    fn admit_drains_reserves_kv_and_activates() {
        let store = prefilled_store(&[(0, 8, 4)]);
        let cluster = test_cluster();
        let mut w = worker(Rc::clone(&store), Rc::clone(&cluster));

        w.enqueue(AttnWorkerMsg::Admit { req: RequestId(0) });
        assert_eq!(w.batch.kv.active_kv, 0, "Admit alone reserves no resident KV");
        assert!(w.promised.is_empty());

        let mut events = Vec::new();
        w.tick(Time::ZERO, &mut events);

        // Drain admitted it (prefilled handoff ⇒ finalized at activation): resident
        // KV = prompt 8, in the decode set, sticky to slot 0.
        assert_eq!(w.batch.kv.active_kv, 8);
        assert!(w.batch.decodes.iter().any(|(r, _)| *r == RequestId(0)));
        assert_eq!(w.request_to_slot.get(&RequestId(0)), Some(&0));
        assert!(!w.promised.contains_key(&RequestId(0)), "finalized ⇒ not promised");
        assert!(w.slots[0].reqs.contains(&RequestId(0)));
    }

    /// One micro-batch walks all layers of an iteration: one handoff per layer, and
    /// KV grows by exactly 1 after the last layer (per iteration, not per layer).
    #[test]
    fn one_microbatch_walks_all_layers_and_grows_kv_once() {
        let store = prefilled_store(&[(0, 8, 4)]);
        let cluster = test_cluster();
        let sender = register_test_sender(&cluster);
        let mut w = worker(Rc::clone(&store), Rc::clone(&cluster));

        w.enqueue(AttnWorkerMsg::Admit { req: RequestId(0) });
        let mut events = Vec::new();
        w.tick(Time::ZERO, &mut events); // drain + activate
        let kv_after_activate = w.batch.kv.active_kv;
        assert_eq!(kv_after_activate, 8);
        let slot = w.request_to_slot[&RequestId(0)] as u8;

        run_iteration(&mut w, slot, sender, 64, 1, &mut events);

        // One handoff per layer, batch-granular, bytes = attn_to_ffn(2) × 1 token.
        assert_eq!(events.len(), 2);
        for (layer, e) in events.iter().enumerate() {
            assert_eq!(
                e,
                &AttnWorkerEvent::AttnLayerOutputsReady {
                    worker: WorkerId(0),
                    slot,
                    reqs: vec![RequestId(0)],
                    layer: layer as u16,
                    bytes: 2,
                }
            );
        }
        // KV grew by exactly 1 (one iteration), not by 2 (num_layers).
        assert_eq!(w.batch.kv.active_kv, kv_after_activate + 1);
        assert_eq!(w.current_kv(RequestId(0)), 9);
    }

    /// Release frees the request's KV and empties its slot (the slot stays in the
    /// ring as a no-op).
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

        assert!(w.batch.decodes.iter().any(|(r, _)| *r == RequestId(0)), "req0 admitted");
        assert!(!w.batch.decodes.iter().any(|(r, _)| *r == RequestId(1)), "req1 blocked");
        assert_eq!(w.worker_pending.front(), Some(&RequestId(1)), "req1 head-of-line");

        // Free req0 → req1 now fits and drains in on the next tick.
        w.enqueue(AttnWorkerMsg::Release { req: RequestId(0) });
        w.tick(Time::from_ms(1.0), &mut events);
        assert!(w.batch.decodes.iter().any(|(r, _)| *r == RequestId(1)), "req1 admitted after release");
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
        assert_ne!(s0, s1, "least-KV spreads the two requests onto different slots");
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
        assert!(w.promised.contains_key(&RequestId(0)), "fresh ⇒ still promised");
        assert_eq!(w.batch.kv.active_kv, 0, "prefill KV is not resident yet");
        let slot = w.request_to_slot[&RequestId(0)] as u8;

        run_iteration(&mut w, slot, sender, 64, 1, &mut events);

        // After the prefill pass (last layer) it finalized into decode: resident KV
        // = prompt 8, no longer promised.
        assert!(!w.promised.contains_key(&RequestId(0)), "finalized at last prefill layer");
        assert_eq!(w.batch.kv.active_kv, 8);
        assert!(w.batch.decodes.iter().any(|(r, _)| *r == RequestId(0)));
    }
}
