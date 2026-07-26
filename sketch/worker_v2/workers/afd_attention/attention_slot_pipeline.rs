//! `AttentionSlotPipeline<K, E>` — the shared AFD-attn slot pipeline (interfaces doc §5). This is
//! the concrete slot FSM + comm seam factored OUT of the shell so that BOTH the
//! colocated AFD-attn worker (`SlotAttentionWorker`, FreshRequestSlotAdmission ingress) and the PD-for-AFD
//! decode-attn worker (`PullSlotAttentionWorker`, KV-pull ingress) drive the *same* pipeline —
//! the interfaces doc's "share the private slot core, do NOT embed one concrete worker
//! inside another." The only thing that differs between the two shells is the INGRESS
//! (how a request becomes ready to slot); everything downstream (per-layer lockstep,
//! single-pull / single-compute gates, `IterStart` / `AttnLayerOutputsReady`, the
//! last-layer prefill→decode classification) is identical and lives here.
//!
//! `context` is NOT a field: it lives on the owning shell and is threaded in as a `&WorkerContext`
//! param, so the shell can hand the ingress `&mut kv_store` (via `kv_mut`) and `&context` in one
//! expression without a self-borrow conflict.
//!
//! Reproduces the real `DisaggAttnWorker` cadence (disagg_attn.rs): NUM_SLOTS pipelined
//! micro-batches over ONE shard `Batch`, per-layer `AwaitFlush`/`SlotFlushed` lockstep.
//! Rough spots (flagged): slot placement is least-request-count (not reserved-KV load);
//! the last-layer prefill→decode split reads the store's `is_prefill()` directly.

use std::collections::HashMap;

use crate::common::{RequestId, Time};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::AttnWorkerEvent;

use super::super::super::execution::AttentionLayerExecution;
use super::super::super::kv::{KvStore, SlotPipelineKv};
use super::super::super::shared::advance_scope::AdvanceScope;
use super::super::super::shared::context::WorkerContext;

pub(super) const NUM_SLOTS: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AttentionSlotPhase {
    /// Waiting for this layer's QKV `ReadyNotification`.
    Wait,
    /// Notified; may start the pull once the single-pull gate is free.
    WaitComplete,
    /// Pulling QKV from the ffn (`submit_gather` in flight).
    Pull,
    /// Pull landed; may start compute once the single-compute gate is free.
    PullComplete,
    /// Attention for `current_layer` is running.
    Compute,
    /// Compute done + handoff emitted; parked until the pool's `SlotFlushed`.
    AwaitFlush,
}

/// Per-slot pipeline state (data owned by the shell). Mirrors the real `Slot`.
struct AttentionSlot {
    request_ids: Vec<RequestId>,
    pending_request_ids: Vec<RequestId>,
    current_layer: u16,
    iteration_started_event_sent: bool,
    state: AttentionSlotPhase,
    input_ready: bool,
    pull_send_group_id: u16,
    pull_bytes: u64,
    pull_end: Time,
    compute_end: Time,
    iteration_sequence: u64,
    slot_input_is_current: bool,
    /// Query-token count of the cached input (from `AttentionLayerExecution::build_slot_input`), reused
    /// for the handoff byte size without re-reading the opaque `Input`.
    tokens: u64,
}

impl AttentionSlot {
    fn new() -> Self {
        Self {
            request_ids: Vec::new(),
            pending_request_ids: Vec::new(),
            current_layer: 0,
            iteration_started_event_sent: false,
            state: AttentionSlotPhase::Wait,
            input_ready: false,
            pull_send_group_id: 0,
            pull_bytes: 0,
            pull_end: Time::ZERO,
            compute_end: Time::ZERO,
            iteration_sequence: 0,
            slot_input_is_current: false,
            tokens: 0,
        }
    }

    #[inline]
    fn start_boundary_ready(&self) -> bool {
        self.current_layer == 0
            && self.state == AttentionSlotPhase::Wait
            && !self.iteration_started_event_sent
    }
}

pub(super) struct AttentionSlotPipeline<K, E>
where
    K: KvStore + SlotPipelineKv,
    E: AttentionLayerExecution,
{
    kv_store: K,
    execution: E,
    slots: [AttentionSlot; NUM_SLOTS],
    slot_inputs: [E::Input; NUM_SLOTS],
    /// Sticky request → slot routing (set at placement; survives decode re-entry).
    request_to_slot: HashMap<RequestId, usize>,
    cluster: SharedGpuCluster,
    /// This shard's recv/send comm endpoint (QKV lands here; ffn pulls outputs).
    receive_group_id: u16,
    num_layers: u16,
}

impl<K, E> AttentionSlotPipeline<K, E>
where
    K: KvStore + SlotPipelineKv,
    E: AttentionLayerExecution,
{
    pub(super) fn from_components(
        kv_store: K,
        execution: E,
        cluster: SharedGpuCluster,
        receive_group_id: u16,
        num_layers: u16,
    ) -> Self {
        Self {
            kv_store,
            execution,
            slots: std::array::from_fn(|_| AttentionSlot::new()),
            slot_inputs: std::array::from_fn(|_| E::Input::default()),
            request_to_slot: HashMap::new(),
            cluster,
            receive_group_id,
            num_layers,
        }
    }

    /// The ingress reserves/commits KV through this handle before `place_request`.
    #[inline]
    pub(super) fn kv_mut(&mut self) -> &mut K {
        &mut self.kv_store
    }

    // ── ingress → slot placement ─────────────────────────────────────────────────

    /// Route an admitted/landed request into the least-loaded slot's insert list
    /// (its KV was already reserved by FreshRequestSlotAdmission or pulled by the pull front-end;
    /// the layer-0 boundary turns `pending_request_ids` into live request IDs).
    pub(super) fn place_request(&mut self, req: RequestId) {
        let slot = self.least_loaded_slot();
        self.slots[slot].pending_request_ids.push(req);
        self.request_to_slot.insert(req, slot);
    }

    /// Rough: least-request-count slot. The real worker balances reserved+resident
    /// KV per slot; that weight now lives in KV/admission, so a faithful port would
    /// add a `kv_store.reserved_tokens_for_request(req)` read — deferred (a load heuristic, not a seam).
    fn least_loaded_slot(&self) -> usize {
        (0..NUM_SLOTS)
            .min_by_key(|&i| {
                self.slots[i].request_ids.len() + self.slots[i].pending_request_ids.len()
            })
            .unwrap_or(0)
    }

    // ── control-message handlers (shell forwards these) ──────────────────────────

    pub(super) fn on_msg_ready_notification(
        &mut self,
        slot: usize,
        layer: u16,
        send_group_id: u16,
        bytes: u64,
    ) {
        let Some(s) = self.slots.get_mut(slot) else {
            return;
        };
        // Layers run in lockstep, so a notification always finds its slot waiting.
        if layer != s.current_layer || s.state != AttentionSlotPhase::Wait {
            return;
        }
        s.input_ready = true;
        s.pull_send_group_id = send_group_id;
        s.pull_bytes = bytes;
    }

    pub(super) fn on_msg_slot_flushed(&mut self, slot: usize, layer: u16) {
        let Some(s) = self.slots.get_mut(slot) else {
            return;
        };
        if s.state != AttentionSlotPhase::AwaitFlush || layer != s.current_layer {
            return;
        }
        self.move_slot_to_next_layer(slot);
    }

    /// Cancellation for a request that has reached a slot: drop its live KV and forget
    /// its slot membership. Returns whether the request was slotted (the shell falls
    /// back to its ingress-queue cancel if not). Called by both shells' `Release`.
    pub(super) fn release_slotted_request(&mut self, req: RequestId) -> bool {
        let Some(slot_idx) = self.request_to_slot.remove(&req) else {
            return false;
        };
        self.kv_store.release(req, 0);
        let slot = &mut self.slots[slot_idx];
        slot.request_ids.retain(|&r| r != req);
        slot.pending_request_ids.retain(|&r| r != req);
        slot.slot_input_is_current = false; // membership changed mid-iteration
        if slot.request_ids.is_empty() {
            slot.iteration_started_event_sent = false; // emptied slot reopens on next placement
        }
        true
    }

    // ── pipeline loop ────────────────────────────────────────────────────────────

    /// Drive every slot to a fixpoint: open layer-0 iterations, advance pull/compute
    /// completions, then fire the single-pull / single-compute gates.
    pub(super) fn drive_slots(
        &mut self,
        context: &WorkerContext,
        now: Time,
        events: &mut Vec<AttnWorkerEvent>,
    ) {
        loop {
            let mut progressed = false;
            progressed |= self.open_ready_iterations(context, events);
            progressed |= self.finish_elapsed_stages(context, now, events);
            progressed |= self.start_ready_stages(context, now);
            if !progressed {
                break;
            }
        }
    }

    pub(super) fn has_pending_slot_work(&self) -> bool {
        self.slots.iter().any(|slot| {
            !slot.request_ids.is_empty()
                || !slot.pending_request_ids.is_empty()
                || slot.start_boundary_ready()
                || slot.input_ready
                || slot.state != AttentionSlotPhase::Wait
        })
    }

    /// Open each layer-0 slot's iteration: drain `pending_request_ids` into live request IDs
    /// (a prefilled handoff becomes resident decode now; a fresh prefill stays
    /// reserved until its last prefill layer), then announce `IterStart` once.
    fn open_ready_iterations(
        &mut self,
        context: &WorkerContext,
        events: &mut Vec<AttnWorkerEvent>,
    ) -> bool {
        let mut progressed = false;
        for slot_index in 0..NUM_SLOTS {
            if !self.slots[slot_index].start_boundary_ready() {
                continue;
            }
            if !self.slots[slot_index].pending_request_ids.is_empty() {
                let newly: Vec<RequestId> = self.slots[slot_index]
                    .pending_request_ids
                    .drain(..)
                    .collect();
                for rid in newly {
                    self.slots[slot_index].request_ids.push(rid);
                    if !self.req_is_prefill(context, rid) {
                        self.begin_decode(context, rid); // prefilled handoff: resident decode now
                    }
                }
                self.slots[slot_index].slot_input_is_current = false;
            }
            let request_ids = self.slots[slot_index].request_ids.clone();
            events.push(AttnWorkerEvent::IterStart {
                worker: context.id,
                slot: slot_index as u8,
                reqs: request_ids,
            });
            self.slots[slot_index].iteration_started_event_sent = true;
            progressed = true;
        }
        progressed
    }

    fn finish_elapsed_stages(
        &mut self,
        context: &WorkerContext,
        now: Time,
        events: &mut Vec<AttnWorkerEvent>,
    ) -> bool {
        let mut progressed = false;
        for slot_index in 0..NUM_SLOTS {
            match self.slots[slot_index].state {
                AttentionSlotPhase::Wait if self.slots[slot_index].input_ready => {
                    self.slots[slot_index].state = AttentionSlotPhase::WaitComplete;
                    progressed = true;
                }
                AttentionSlotPhase::Pull if now >= self.slots[slot_index].pull_end => {
                    self.slots[slot_index].state = AttentionSlotPhase::PullComplete;
                    progressed = true;
                }
                AttentionSlotPhase::Compute if now >= self.slots[slot_index].compute_end => {
                    self.finish_attention_layer(context, slot_index, now, events);
                    progressed = true;
                }
                _ => {}
            }
        }
        progressed
    }

    /// Finish a slot's layer: emit the attn→ffn handoff; at the iteration boundary
    /// (last layer) move finished prefills into decode and grow decoders' KV by
    /// one; then park in `AwaitFlush` (the pool barrier releases via `SlotFlushed`).
    fn finish_attention_layer(
        &mut self,
        context: &WorkerContext,
        slot_index: usize,
        now: Time,
        events: &mut Vec<AttnWorkerEvent>,
    ) {
        let layer = self.slots[slot_index].current_layer;
        self.build_slot_input_if_stale(context, slot_index);
        let tokens = self.slots[slot_index].tokens;
        let out_bytes = self.execution.attn_to_ffn_bytes_per_token() * tokens;
        events.push(AttnWorkerEvent::AttnLayerOutputsReady {
            worker: context.id,
            slot: slot_index as u8,
            reqs: self.slots[slot_index].request_ids.clone(),
            tokens,
            layer,
            send_gid: self.receive_group_id,
            bytes: out_bytes,
        });

        if layer == self.num_layers.saturating_sub(1) {
            // Classify the slot: fresh prefills that just finished → begin decode;
            // already-decoding requests → grow KV by one. (Rough: reads store
            // `is_prefill()` directly; the ffn Terminal's flip is external here.)
            let mut decoding: Vec<RequestId> = Vec::new();
            let mut to_commit: Vec<(RequestId, u64, u32)> = Vec::new();
            {
                let store = context.requests.borrow();
                for &rid in &self.slots[slot_index].request_ids {
                    let record = &store[rid];
                    if record.is_prefill() {
                        let initial_kv = u64::from(record.prompt_len + record.prefix_kv);
                        let remaining = record.decode_len.saturating_sub(record.tokens_emitted);
                        to_commit.push((rid, initial_kv, remaining));
                    } else {
                        decoding.push(rid);
                    }
                }
            }
            for (rid, initial_kv, remaining) in to_commit {
                self.kv_store.commit_resident(rid, 0, initial_kv, remaining);
            }
            self.kv_store.advance(
                AdvanceScope::RequestSubset {
                    partition: 0,
                    request_ids: &decoding,
                },
                1,
            );
            self.kv_store.sample_submit(0, now);
        }
        self.slots[slot_index].state = AttentionSlotPhase::AwaitFlush;
    }

    /// promised → resident decode. Reads the initial KV + horizon from the store.
    fn begin_decode(&mut self, context: &WorkerContext, rid: RequestId) {
        let (initial_kv, remaining) = {
            let store = context.requests.borrow();
            let record = &store[rid];
            (
                u64::from(record.prompt_len + record.prefix_kv),
                record.decode_len.saturating_sub(record.tokens_emitted),
            )
        };
        self.kv_store.commit_resident(rid, 0, initial_kv, remaining);
    }

    #[inline]
    fn req_is_prefill(&self, context: &WorkerContext, rid: RequestId) -> bool {
        context.requests.borrow()[rid].is_prefill()
    }

    fn move_slot_to_next_layer(&mut self, slot_index: usize) {
        let slot = &mut self.slots[slot_index];
        slot.current_layer = (slot.current_layer + 1) % self.num_layers.max(1);
        slot.state = AttentionSlotPhase::Wait;
        slot.input_ready = false;
        slot.pull_send_group_id = 0;
        slot.pull_bytes = 0;
        if slot.current_layer == 0 {
            slot.iteration_started_event_sent = false;
            slot.iteration_sequence += 1;
            slot.slot_input_is_current = false; // last-layer advance grew KV; rebuild next iter
        }
    }

    /// Single-pull / single-compute gates: at most one slot pulling and one
    /// computing at a time (the attn↔ffn pull overlapping the attention compute).
    fn start_ready_stages(&mut self, context: &WorkerContext, now: Time) -> bool {
        let mut progressed = false;
        if !self.has_slot_in_phase(AttentionSlotPhase::Pull) {
            if let Some(slot_index) = self.first_slot_in_phase(AttentionSlotPhase::WaitComplete) {
                self.start_slot_pull(slot_index, now);
                progressed = true;
            }
        }
        if !self.has_slot_in_phase(AttentionSlotPhase::Compute) {
            if let Some(slot_index) = self.first_slot_in_phase(AttentionSlotPhase::PullComplete) {
                self.start_slot_compute(context, slot_index, now);
                progressed = true;
            }
        }
        progressed
    }

    fn has_slot_in_phase(&self, state: AttentionSlotPhase) -> bool {
        self.slots.iter().any(|slot| slot.state == state)
    }

    fn first_slot_in_phase(&self, state: AttentionSlotPhase) -> Option<usize> {
        (0..NUM_SLOTS).find(|&slot_index| self.slots[slot_index].state == state)
    }

    fn start_slot_pull(&mut self, slot_index: usize, now: Time) {
        self.slots[slot_index].state = AttentionSlotPhase::Pull;
        let bytes = self.slots[slot_index].pull_bytes;
        if bytes == 0 {
            self.slots[slot_index].pull_end = now; // zero-byte handoff: nothing on the wire
            return;
        }
        let send_group_id = self.slots[slot_index].pull_send_group_id;
        // Single-source gather: the sender frees after its transmission slice, so a
        // ffn worker pulled by several attn workers overlaps its sends by α.
        let end = self.cluster.borrow_mut().submit_gather(
            now,
            &[(send_group_id, bytes)],
            self.receive_group_id,
            "afd_attn_pull",
            "",
        );
        self.slots[slot_index].pull_end = end;
    }

    fn start_slot_compute(&mut self, context: &WorkerContext, slot_index: usize, now: Time) {
        self.slots[slot_index].state = AttentionSlotPhase::Compute;
        if self.slots[slot_index].request_ids.is_empty() {
            // Empty shards are control-plane participants: emit the layer event but
            // keep zero-token shapes off the model/profile lookup path.
            self.slots[slot_index].compute_end = now;
            return;
        }
        self.build_slot_input_if_stale(context, slot_index);
        let layer = self.slots[slot_index].current_layer;
        let iteration_id = self.slots[slot_index].iteration_sequence;
        // `evaluate_attention_layer` returns the segment's wall-time DELTA (same convention as the
        // iter family's `eval` + `run_iter`); the shell arms the absolute end.
        let seg = self.execution.evaluate_attention_layer(
            layer,
            slot_index as u8,
            &self.slot_inputs[slot_index],
            iteration_id,
            now,
        );
        self.slots[slot_index].compute_end = now + seg;
    }

    /// Rebuild slot `slot_index`'s cached attention input via `AttentionLayerExecution::build_slot_input`, unless
    /// still valid for this iteration. Caches the token count for the handoff bytes.
    fn build_slot_input_if_stale(&mut self, context: &WorkerContext, slot_index: usize) {
        if self.slots[slot_index].slot_input_is_current {
            return;
        }
        let request_ids = self.slots[slot_index].request_ids.clone();
        let tokens = self.execution.build_slot_input(
            AdvanceScope::RequestSubset {
                partition: 0,
                request_ids: &request_ids,
            },
            &self.kv_store,
            &context.requests,
            &mut self.slot_inputs[slot_index],
        );
        self.slots[slot_index].tokens = tokens;
        self.slots[slot_index].slot_input_is_current = true;
    }

    pub(super) fn next_wakeup(&self, now: Time) -> Option<Time> {
        let mut next_wakeup: Option<Time> = None;
        for slot in &self.slots {
            let candidate_wakeup = match slot.state {
                AttentionSlotPhase::Compute => slot.compute_end,
                AttentionSlotPhase::Pull => slot.pull_end,
                _ => continue,
            };
            if candidate_wakeup > now {
                next_wakeup = Some(next_wakeup.map_or(candidate_wakeup, |current_wakeup| {
                    current_wakeup.min(candidate_wakeup)
                }));
            }
        }
        next_wakeup
    }

    /// Slots with live members (for `WorkerStatus::active_requests`).
    pub(super) fn active_slot_count(&self) -> u32 {
        self.slots
            .iter()
            .filter(|slot| !slot.request_ids.is_empty())
            .count() as u32
    }

    /// L3 resident projected-peak KV for partition 0 (the shell adds its ingress's
    /// reserved/pending tokens for the L6 `estimated_peak_kv`).
    pub(super) fn estimated_peak_kv(&self) -> u64 {
        self.kv_store.estimated_peak(0)
    }
}
