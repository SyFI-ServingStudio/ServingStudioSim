//! `SlotAttentionWorker<K, A, E>` — the colocated AFD-attn shell (interfaces doc §5,
//! family = AFD-attn). Now a THIN wrapper: the slot pipeline + comm seam live in the
//! shared `AttentionSlotPipeline` (so the PD-for-AFD decode-attn worker drives the identical core);
//! this shell only owns the FreshRequestSlotAdmission *ingress* — the L1 enqueue + L2 KV-gated
//! reserve that hands admitted ids to the core. Generic over the three per-family axes:
//! `K: SlotPipelineKv`, `A: SlotPipelineAdmission<K>`, `E: AttentionLayerExecution`.
//!
//! Reproduces the real `DisaggAttnWorker`: a fresh `Admit` enqueues (no KV); each tick
//! `reserve_fitting_requests` reserves the KV-fitting head-of-line requests and `place_request`s them into
//! pipeline slots; the core drives the per-layer lockstep. `Release` drops a slotted
//! request's KV via the core, else cancels it out of the pending queue.

use crate::common::{Time, WorkerId};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::iter_worker::{AfdAttnWorker, IterWorker};
use crate::worker::types::{AttnWorkerEvent, AttnWorkerMsg, WorkerStatus};

use super::super::super::admission::SlotPipelineAdmission;
use super::super::super::execution::AttentionLayerExecution;
use super::super::super::kv::{KvStore, SlotPipelineKv};
use super::super::super::shared::context::WorkerContext;
use super::attention_slot_pipeline::AttentionSlotPipeline;

pub struct SlotAttentionWorker<K, A, E>
where
    K: KvStore + SlotPipelineKv,
    A: SlotPipelineAdmission<K>,
    E: AttentionLayerExecution,
{
    context: WorkerContext,
    admission: A,
    core: AttentionSlotPipeline<K, E>,
}

impl<K, A, E> SlotAttentionWorker<K, A, E>
where
    K: KvStore + SlotPipelineKv,
    A: SlotPipelineAdmission<K>,
    E: AttentionLayerExecution,
{
    pub(super) fn from_components(
        context: WorkerContext,
        kv_store: K,
        admission: A,
        execution: E,
        cluster: SharedGpuCluster,
        receive_group_id: u16,
        num_layers: u16,
    ) -> Self {
        Self {
            context,
            admission,
            core: AttentionSlotPipeline::from_components(
                kv_store,
                execution,
                cluster,
                receive_group_id,
                num_layers,
            ),
        }
    }

    fn drive_slot_pipeline(
        &mut self,
        now: Time,
        events: &mut Vec<AttnWorkerEvent>,
    ) -> Option<Time> {
        if self.admission.queued_requests() == 0 && !self.core.has_pending_slot_work() {
            return None;
        }
        // Level-2 drain: reserve the KV-fitting head-of-line requests, place each.
        let admitted =
            self.admission
                .reserve_fitting_requests(self.core.kv_mut(), &self.context, now);
        for req in admitted {
            self.core.place_request(req);
        }
        self.core.drive_slots(&self.context, now, events);
        self.core.next_wakeup(now)
    }

    fn on_msg_admit(&mut self, req: crate::common::RequestId) {
        self.admission.enqueue_fresh_request(req, &self.context);
    }

    fn on_msg_release(&mut self, req: crate::common::RequestId) {
        if !self.core.release_slotted_request(req) {
            self.admission
                .cancel_or_release_request(self.core.kv_mut(), req);
        }
    }

    fn on_msg_ready_notification(
        &mut self,
        slot: usize,
        layer: u16,
        send_group_id: u16,
        bytes: u64,
    ) {
        self.core
            .on_msg_ready_notification(slot, layer, send_group_id, bytes);
    }

    fn on_msg_slot_flushed(&mut self, slot: usize, layer: u16) {
        self.core.on_msg_slot_flushed(slot, layer);
    }
}

impl<K, A, E> IterWorker for SlotAttentionWorker<K, A, E>
where
    K: KvStore + SlotPipelineKv,
    A: SlotPipelineAdmission<K>,
    E: AttentionLayerExecution,
{
    type Msg = AttnWorkerMsg;
    type Event = AttnWorkerEvent;

    fn id(&self) -> WorkerId {
        self.context.id
    }

    fn enqueue(&mut self, msg: AttnWorkerMsg) {
        match msg {
            // Level-1 enqueue (no KV yet); the KV gate runs at the Level-2 drain.
            AttnWorkerMsg::Admit { req } => self.on_msg_admit(req),
            AttnWorkerMsg::Release { req } => self.on_msg_release(req),
            AttnWorkerMsg::ReadyNotification {
                slot,
                layer,
                send_gid: send_group_id,
                bytes,
            } => self.on_msg_ready_notification(slot as usize, layer, send_group_id, bytes),
            AttnWorkerMsg::SlotFlushed { slot, layer } => {
                self.on_msg_slot_flushed(slot as usize, layer)
            }
        }
    }

    fn tick(&mut self, now: Time, events: &mut Vec<AttnWorkerEvent>) -> Option<Time> {
        self.drive_slot_pipeline(now, events)
    }

    fn status(&self) -> WorkerStatus {
        WorkerStatus {
            queued_requests: self.admission.queued_requests(),
            active_requests: self.core.active_slot_count(),
        }
    }
}

impl<K, A, E> AfdAttnWorker for SlotAttentionWorker<K, A, E>
where
    K: KvStore + SlotPipelineKv,
    A: SlotPipelineAdmission<K>,
    E: AttentionLayerExecution,
{
    /// Three disjoint KV levels: L3 resident projected peak + L2 reserved (both in
    /// `core.estimated_peak_kv`) + L1 queued (`admission.queued_kv_tokens`).
    fn estimated_peak_kv(&self) -> u64 {
        self.core.estimated_peak_kv() + self.admission.queued_kv_tokens()
    }
}
