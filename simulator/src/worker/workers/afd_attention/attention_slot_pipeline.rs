//! Shared AFD attention slot pipeline.
//!
//! This family-private core owns slot membership, the per-layer lockstep FSM,
//! and the attention↔FFN communication seam. It does not own request admission
//! or L6 message dispatch. KV membership lives in `K`; layer input construction
//! and cost evaluation live in `E`.

use crate::common::{IdMap, RequestId, Time};
use crate::worker::execution::AttentionLayerExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::{PrefixKv, SlotPipelineKv};
use crate::worker::shared::advance_scope::AdvanceScope;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::AttnWorkerEvent;

pub(super) const NUM_SLOTS: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttentionSlotState {
    Wait,
    WaitComplete,
    Pull,
    PullComplete,
    Compute,
    AwaitFlush,
}

struct AttentionSlot {
    request_ids: Vec<RequestId>,
    pending_request_ids: Vec<RequestId>,
    current_layer: u16,
    iteration_announced: bool,
    state: AttentionSlotState,
    notified: bool,
    pull_send_group_id: u16,
    pull_bytes: u64,
    pull_end: Time,
    compute_end: Time,
    iteration: u64,
    input_valid: bool,
    input_tokens: u64,
}

impl AttentionSlot {
    fn new() -> Self {
        Self {
            request_ids: Vec::new(),
            pending_request_ids: Vec::new(),
            current_layer: 0,
            iteration_announced: false,
            state: AttentionSlotState::Wait,
            notified: false,
            pull_send_group_id: 0,
            pull_bytes: 0,
            pull_end: Time::ZERO,
            compute_end: Time::ZERO,
            iteration: 0,
            input_valid: false,
            input_tokens: 0,
        }
    }

    fn start_boundary_ready(&self) -> bool {
        self.current_layer == 0
            && self.state == AttentionSlotState::Wait
            && !self.iteration_announced
    }
}

pub(super) struct AttentionSlotPipeline<K, E>
where
    K: SlotPipelineKv + PrefixKv,
    E: AttentionLayerExecution,
{
    kv_store: K,
    execution: E,
    slots: [AttentionSlot; NUM_SLOTS],
    slot_inputs: [E::Input; NUM_SLOTS],
    request_to_slot: IdMap<RequestId, usize>,
    cluster: SharedGpuCluster,
    receive_group_id: u16,
    num_layers: u16,
}

impl<K, E> AttentionSlotPipeline<K, E>
where
    K: SlotPipelineKv + PrefixKv,
    E: AttentionLayerExecution,
{
    pub(super) fn from_components(
        kv_store: K,
        execution: E,
        cluster: SharedGpuCluster,
        receive_group_id: u16,
    ) -> Self {
        let num_layers = execution.num_layers();
        Self {
            kv_store,
            execution,
            slots: std::array::from_fn(|_| AttentionSlot::new()),
            slot_inputs: std::array::from_fn(|_| E::Input::default()),
            request_to_slot: IdMap::default(),
            cluster,
            receive_group_id,
            num_layers,
        }
    }

    pub(super) fn kv_mut(&mut self) -> &mut K {
        &mut self.kv_store
    }

    pub(super) fn kv(&self) -> &K {
        &self.kv_store
    }

    pub(super) fn place_request(&mut self, request: RequestId) {
        let slot = self.choose_least_kv_slot();
        self.slots[slot].pending_request_ids.push(request);
        self.request_to_slot.insert(request, slot);
    }

    fn choose_least_kv_slot(&self) -> usize {
        (0..NUM_SLOTS)
            .min_by_key(|&slot| self.slot_kv_load(slot))
            .unwrap_or(0)
    }

    fn slot_kv_load(&self, slot: usize) -> u64 {
        self.slots[slot]
            .request_ids
            .iter()
            .chain(self.slots[slot].pending_request_ids.iter())
            .map(|&request| self.kv_store.request_kv_weight(request))
            .sum()
    }

    pub(super) fn on_msg_ready_notification(
        &mut self,
        slot: usize,
        layer: u16,
        send_group_id: u16,
        bytes: u64,
    ) {
        let Some(slot_state) = self.slots.get_mut(slot) else {
            debug_assert!(false, "notification slot tag {slot} out of range");
            return;
        };
        debug_assert!(
            layer == slot_state.current_layer && slot_state.state == AttentionSlotState::Wait,
            "slot {slot} got layer {layer}, but is at layer {} in {:?}",
            slot_state.current_layer,
            slot_state.state
        );
        if layer != slot_state.current_layer || slot_state.state != AttentionSlotState::Wait {
            return;
        }
        slot_state.notified = true;
        slot_state.pull_send_group_id = send_group_id;
        slot_state.pull_bytes = bytes;
    }

    pub(super) fn on_msg_slot_flushed(&mut self, slot: usize, layer: u16) {
        let Some(slot_state) = self.slots.get_mut(slot) else {
            debug_assert!(false, "SlotFlushed slot tag {slot} out of range");
            return;
        };
        debug_assert!(
            slot_state.state == AttentionSlotState::AwaitFlush && layer == slot_state.current_layer,
            "slot {slot} got flush for layer {layer}, but is at layer {} in {:?}",
            slot_state.current_layer,
            slot_state.state
        );
        if slot_state.state != AttentionSlotState::AwaitFlush || layer != slot_state.current_layer {
            return;
        }
        self.advance_slot_layer(slot);
    }

    pub(super) fn release_slotted_request(&mut self, request: RequestId, now: Time) -> bool {
        let Some(slot) = self.request_to_slot.remove(&request) else {
            return false;
        };
        self.kv_store.release_retaining_prefix(request, 0, now);
        // FFN Terminal release is asynchronous with respect to attention-layer
        // sampling and may be this worker's final activity. Capture the post-
        // release active/retained composition at the same lifecycle boundary.
        self.kv_store.sample_submit(0, now);
        let slot_state = &mut self.slots[slot];
        slot_state.request_ids.retain(|&member| member != request);
        slot_state
            .pending_request_ids
            .retain(|&member| member != request);
        slot_state.input_valid = false;
        if slot_state.request_ids.is_empty() {
            slot_state.iteration_announced = false;
        }
        true
    }

    pub(super) fn has_pending_slot_work(&self) -> bool {
        self.slots.iter().any(|slot| {
            !slot.request_ids.is_empty()
                || !slot.pending_request_ids.is_empty()
                || slot.start_boundary_ready()
                || slot.notified
                || slot.state != AttentionSlotState::Wait
        })
    }

    pub(super) fn drive_slots(
        &mut self,
        context: &WorkerContext,
        now: Time,
        events: &mut Vec<AttnWorkerEvent>,
    ) {
        loop {
            let mut progressed = false;
            progressed |= self.open_iterations(context, events);
            progressed |= self.advance_completions(context, now, events);
            progressed |= self.start_ready_transitions(context, now);
            if !progressed {
                break;
            }
        }
    }

    fn open_iterations(
        &mut self,
        context: &WorkerContext,
        events: &mut Vec<AttnWorkerEvent>,
    ) -> bool {
        let mut progressed = false;
        for slot in 0..NUM_SLOTS {
            if !self.slots[slot].start_boundary_ready() {
                continue;
            }
            if !self.slots[slot].pending_request_ids.is_empty() {
                let newly_admitted: Vec<RequestId> =
                    self.slots[slot].pending_request_ids.drain(..).collect();
                for request in newly_admitted {
                    self.slots[slot].request_ids.push(request);
                    if !self.request_is_prefill(context, request) {
                        self.begin_decode(context, request);
                    }
                }
                self.slots[slot].input_valid = false;
            }
            let tokens = self.slots[slot]
                .request_ids
                .iter()
                .map(|&request| {
                    if self.request_is_prefill(context, request) {
                        u64::from(self.kv_store.prefill_tokens_to_compute(request))
                    } else {
                        1
                    }
                })
                .sum();
            events.push(AttnWorkerEvent::IterStart {
                worker: context.id,
                slot: slot as u8,
                reqs: self.slots[slot].request_ids.clone(),
                tokens,
            });
            self.slots[slot].iteration_announced = true;
            progressed = true;
        }
        progressed
    }

    fn request_is_prefill(&self, context: &WorkerContext, request: RequestId) -> bool {
        context.requests.borrow()[request]
            .progress
            .output_tokens_emitted
            == 0
    }

    fn begin_decode(&mut self, context: &WorkerContext, request: RequestId) {
        let post_prefill_context_tokens = self.kv_store.post_prefill_context_tokens(request);
        let remaining_output_tokens = {
            let mut store = context.requests.borrow_mut();
            let record = &mut store[request];
            if record.progress.output_tokens_emitted == 0 {
                record.progress.prefill_tokens_processed =
                    self.kv_store.prefill_tokens_to_compute(request);
            }
            record
                .request
                .definition
                .target_output_tokens
                .saturating_sub(record.progress.output_tokens_emitted)
        };
        self.kv_store.commit_resident(
            request,
            0,
            post_prefill_context_tokens,
            remaining_output_tokens,
        );
    }

    fn advance_completions(
        &mut self,
        context: &WorkerContext,
        now: Time,
        events: &mut Vec<AttnWorkerEvent>,
    ) -> bool {
        let mut progressed = false;
        for slot in 0..NUM_SLOTS {
            match self.slots[slot].state {
                AttentionSlotState::Wait if self.slots[slot].notified => {
                    self.slots[slot].state = AttentionSlotState::WaitComplete;
                    progressed = true;
                }
                AttentionSlotState::Pull if now >= self.slots[slot].pull_end => {
                    self.slots[slot].state = AttentionSlotState::PullComplete;
                    progressed = true;
                }
                AttentionSlotState::Compute if now >= self.slots[slot].compute_end => {
                    self.complete_layer(context, slot, now, events);
                    progressed = true;
                }
                _ => {}
            }
        }
        progressed
    }

    fn complete_layer(
        &mut self,
        context: &WorkerContext,
        slot: usize,
        now: Time,
        events: &mut Vec<AttnWorkerEvent>,
    ) {
        let layer = self.slots[slot].current_layer;
        self.build_slot_input(context, slot);
        let tokens = self.slots[slot].input_tokens;
        let bytes = self.execution.attn_to_ffn_bytes_per_token() * tokens;
        events.push(AttnWorkerEvent::AttnLayerOutputsReady {
            worker: context.id,
            slot: slot as u8,
            reqs: self.slots[slot].request_ids.clone(),
            tokens,
            layer,
            send_gid: self.receive_group_id,
            bytes,
        });

        if layer == self.num_layers.saturating_sub(1) {
            let mut decoding = Vec::new();
            for member_index in 0..self.slots[slot].request_ids.len() {
                let request = self.slots[slot].request_ids[member_index];
                if self.kv_store.has_reservation(request) {
                    self.begin_decode(context, request);
                } else {
                    decoding.push(request);
                }
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
        self.slots[slot].state = AttentionSlotState::AwaitFlush;
    }

    fn advance_slot_layer(&mut self, slot: usize) {
        let slot_state = &mut self.slots[slot];
        slot_state.current_layer = (slot_state.current_layer + 1) % self.num_layers.max(1);
        slot_state.state = AttentionSlotState::Wait;
        slot_state.notified = false;
        slot_state.pull_send_group_id = 0;
        slot_state.pull_bytes = 0;
        if slot_state.current_layer == 0 {
            slot_state.iteration_announced = false;
            slot_state.iteration += 1;
            slot_state.input_valid = false;
        }
    }

    fn start_ready_transitions(&mut self, context: &WorkerContext, now: Time) -> bool {
        let mut progressed = false;
        if !self.any_slot_in(AttentionSlotState::Pull) {
            if let Some(slot) = self.first_slot_in(AttentionSlotState::WaitComplete) {
                self.start_pull(slot, now);
                progressed = true;
            }
        }
        if !self.any_slot_in(AttentionSlotState::Compute) {
            if let Some(slot) = self.first_slot_in(AttentionSlotState::PullComplete) {
                self.start_compute(context, slot, now);
                progressed = true;
            }
        }
        progressed
    }

    fn any_slot_in(&self, state: AttentionSlotState) -> bool {
        self.slots.iter().any(|slot| slot.state == state)
    }

    fn first_slot_in(&self, state: AttentionSlotState) -> Option<usize> {
        (0..NUM_SLOTS).find(|&slot| self.slots[slot].state == state)
    }

    fn start_pull(&mut self, slot: usize, now: Time) {
        self.slots[slot].state = AttentionSlotState::Pull;
        let bytes = self.slots[slot].pull_bytes;
        if bytes == 0 {
            self.slots[slot].pull_end = now;
            return;
        }
        let send_group_id = self.slots[slot].pull_send_group_id;
        self.slots[slot].pull_end = self.cluster.borrow_mut().submit_gather(
            now,
            &[(send_group_id, bytes)],
            self.receive_group_id,
            "afd_attn_pull",
            "",
        );
    }

    fn start_compute(&mut self, context: &WorkerContext, slot: usize, now: Time) {
        self.slots[slot].state = AttentionSlotState::Compute;
        if self.slots[slot].request_ids.is_empty() {
            self.slots[slot].compute_end = now;
            return;
        }
        self.build_slot_input(context, slot);
        let duration = self.execution.evaluate_attention_layer(
            self.slots[slot].current_layer,
            slot as u8,
            &self.slot_inputs[slot],
            self.slots[slot].iteration,
            now,
        );
        self.slots[slot].compute_end = now + duration;
    }

    fn build_slot_input(&mut self, context: &WorkerContext, slot: usize) {
        if self.slots[slot].input_valid {
            return;
        }
        let request_ids = &self.slots[slot].request_ids;
        self.slots[slot].input_tokens = self.execution.build_slot_input(
            AdvanceScope::RequestSubset {
                partition: 0,
                request_ids,
            },
            &self.kv_store,
            &context.requests,
            &mut self.slot_inputs[slot],
        );
        self.slots[slot].input_valid = true;
    }

    pub(super) fn next_wakeup(&self, now: Time) -> Option<Time> {
        let mut next_wakeup = None;
        for slot in &self.slots {
            let candidate = match slot.state {
                AttentionSlotState::Compute => slot.compute_end,
                AttentionSlotState::Pull => slot.pull_end,
                _ => continue,
            };
            if candidate > now {
                next_wakeup =
                    Some(next_wakeup.map_or(candidate, |current: Time| current.min(candidate)));
            }
        }
        next_wakeup
    }

    pub(super) fn active_slot_count(&self) -> u32 {
        self.slots
            .iter()
            .filter(|slot| !slot.request_ids.is_empty())
            .count() as u32
    }

    pub(super) fn estimated_peak_kv(&self) -> u64 {
        self.kv_store.estimated_peak(0)
    }

    #[cfg(test)]
    pub(super) fn request_slot(&self, request: RequestId) -> Option<usize> {
        self.request_to_slot.get(&request).copied()
    }

    #[cfg(test)]
    pub(super) fn current_kv(&self, request: RequestId) -> Option<u64> {
        self.kv_store.current_kv(0, request)
    }
}
