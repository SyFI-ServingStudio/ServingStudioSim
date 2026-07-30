//! `SlotAttentionWorker<K, A, E>` — production AFD-attention shell.
//!
//! The shell owns L6 message dispatch and combines a fresh-request admission
//! ingress with the family-private `AttentionSlotPipeline`. The injected policy
//! owns pending membership/order; admission owns the KV gate; the pipeline owns
//! slot cadence and communication; `K` owns all reservation/resident KV facts;
//! `E` owns attention inputs and cost evaluation.
//! Completion remains FFN Terminal-owned: `Release` only drops this worker's KV.
//!
//! Reading order: type and construction → `IterWorker` message dispatch → message
//! handlers → tick integration → placement/status helpers.

use crate::common::{RequestId, Time, WorkerId};
use crate::worker::admission::{FifoOrder, FreshRequestSlotAdmission, SlotPipelineAdmission};
use crate::worker::execution::{AttentionLayerExecution, AttentionLayerExecutionAdapter};
use crate::worker::iter_worker::{AfdAttnWorker, IterWorker};
use crate::worker::kv::{FullAttnKv, SlotPipelineKv};
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::{AttnWorkerEvent, AttnWorkerMsg, WorkerStatus};

use super::attention_slot_pipeline::AttentionSlotPipeline;

pub struct SlotAttentionWorker<K, A, E>
where
    K: SlotPipelineKv,
    A: SlotPipelineAdmission<K>,
    E: AttentionLayerExecution,
{
    context: WorkerContext,
    admission: A,
    pipeline: AttentionSlotPipeline<K, E>,
}

/// Compatibility name retained at the L6 controller surface.
pub type DisaggAttnWorker<M> = SlotAttentionWorker<
    FullAttnKv,
    FreshRequestSlotAdmission<FifoOrder>,
    AttentionLayerExecutionAdapter<M>,
>;

impl<K, A, E> SlotAttentionWorker<K, A, E>
where
    K: SlotPipelineKv,
    A: SlotPipelineAdmission<K>,
    E: AttentionLayerExecution,
{
    pub(super) fn from_components(
        context: WorkerContext,
        kv_store: K,
        admission: A,
        execution: E,
        cluster: crate::worker::gpu_cluster::SharedGpuCluster,
        receive_group_id: u16,
    ) -> Self {
        Self {
            context,
            admission,
            pipeline: AttentionSlotPipeline::from_components(
                kv_store,
                execution,
                cluster,
                receive_group_id,
            ),
        }
    }
}

impl<K, A, E> IterWorker for SlotAttentionWorker<K, A, E>
where
    K: SlotPipelineKv,
    A: SlotPipelineAdmission<K>,
    E: AttentionLayerExecution,
{
    type Msg = AttnWorkerMsg;
    type Event = AttnWorkerEvent;

    fn id(&self) -> WorkerId {
        self.context.id
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
        WorkerStatus {
            queued_requests: self.admission.queued_requests(),
            active_requests: self.pipeline.active_slot_count(),
        }
    }
}

impl<K, A, E> AfdAttnWorker for SlotAttentionWorker<K, A, E>
where
    K: SlotPipelineKv,
    A: SlotPipelineAdmission<K>,
    E: AttentionLayerExecution,
{
    fn estimated_peak_kv(&self) -> u64 {
        self.pipeline.estimated_peak_kv() + self.admission.queued_kv_tokens()
    }
}

impl<K, A, E> SlotAttentionWorker<K, A, E>
where
    K: SlotPipelineKv,
    A: SlotPipelineAdmission<K>,
    E: AttentionLayerExecution,
{
    fn on_msg_admit(&mut self, request: RequestId) {
        self.admission.enqueue_fresh_request(request, &self.context);
    }

    fn on_msg_release(&mut self, request: RequestId) {
        if !self.pipeline.release_slotted_request(request) {
            debug_assert!(
                !self.admission.cancel_pending(request),
                "a pending AFD request cannot complete before slot placement"
            );
        }
    }

    fn on_msg_ready_notification(
        &mut self,
        slot: usize,
        layer: u16,
        send_group_id: u16,
        bytes: u64,
    ) {
        self.pipeline
            .on_msg_ready_notification(slot, layer, send_group_id, bytes);
    }

    fn on_msg_slot_flushed(&mut self, slot: usize, layer: u16) {
        self.pipeline.on_msg_slot_flushed(slot, layer);
    }

    fn tick_inner(&mut self, now: Time, events: &mut Vec<AttnWorkerEvent>) -> Option<Time> {
        if self.admission.queued_requests() == 0 && !self.pipeline.has_pending_slot_work() {
            return None;
        }

        let admitted =
            self.admission
                .reserve_fitting_requests(self.pipeline.kv_mut(), &self.context, now);
        for &request in admitted {
            self.pipeline.place_request(request);
        }
        self.pipeline.drive_slots(&self.context, now, events);
        self.pipeline.next_wakeup(now)
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::Arc;

    use super::*;
    use crate::common::{PoolId, SharedRequests};
    use crate::test_helpers::{prefilled_store, shared_with, test_cluster, FakeAttn};
    use crate::worker::admission::ShortestJobFirst;
    use crate::worker::build_afd_attention_worker;
    use crate::worker::gpu_cluster::SharedGpuCluster;
    use crate::worker::types::WorkerConfig;

    fn assert_afd_attn_worker<W>()
    where
        W: AfdAttnWorker,
        W::Msg: From<AttnWorkerMsg>,
    {
    }

    #[test]
    fn pending_policy_is_a_composition_axis_for_afd_admission() {
        assert_afd_attn_worker::<
            SlotAttentionWorker<
                FullAttnKv,
                FreshRequestSlotAdmission<FifoOrder>,
                AttentionLayerExecutionAdapter<FakeAttn>,
            >,
        >();
        assert_afd_attn_worker::<
            SlotAttentionWorker<
                FullAttnKv,
                FreshRequestSlotAdmission<ShortestJobFirst>,
                AttentionLayerExecutionAdapter<FakeAttn>,
            >,
        >();
    }

    fn worker_with_config(
        store: SharedRequests,
        cluster: SharedGpuCluster,
        config: WorkerConfig,
    ) -> DisaggAttnWorker<FakeAttn> {
        build_afd_attention_worker(
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
        worker_with_config(store, cluster, WorkerConfig::default())
    }

    fn register_sender(cluster: &SharedGpuCluster) -> u16 {
        let mut cluster = cluster.borrow_mut();
        cluster.allocate(99, 99, 1, "ffn-gpu", "ffn");
        cluster.register_comm_group(0, 1, "ffn", 99)
    }

    fn run_iteration(
        worker: &mut DisaggAttnWorker<FakeAttn>,
        slot: u8,
        sender_group_id: u16,
        events: &mut Vec<AttnWorkerEvent>,
    ) {
        for layer in 0u16..2 {
            worker.enqueue(AttnWorkerMsg::ReadyNotification {
                slot,
                layer,
                send_gid: sender_group_id,
                bytes: 64,
            });
            for step in 0..20 {
                worker.tick(Time::from_ms((layer as u64 * 20 + step) as f64), events);
            }
            worker.enqueue(AttnWorkerMsg::SlotFlushed { slot, layer });
            worker.tick(Time::from_ms((layer as u64 * 20 + 20) as f64), events);
        }
    }

    #[test]
    fn prefilled_request_enters_decode_on_iteration_open() {
        let store = prefilled_store(&[(0, 8, 4)]);
        let mut worker = worker(Rc::clone(&store), test_cluster());
        worker.enqueue(AttnWorkerMsg::Admit { req: RequestId(0) });
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        assert_eq!(worker.pipeline.current_kv(RequestId(0)), Some(8));
        assert_eq!(worker.pipeline.request_slot(RequestId(0)), Some(0));
    }

    #[test]
    fn fresh_request_commits_after_last_attention_layer() {
        let store = shared_with(&[(0, 8, 4)]);
        let cluster = test_cluster();
        let sender_group_id = register_sender(&cluster);
        let mut worker = worker(Rc::clone(&store), Rc::clone(&cluster));
        worker.enqueue(AttnWorkerMsg::Admit { req: RequestId(0) });
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        assert_eq!(worker.pipeline.current_kv(RequestId(0)), None);
        let slot = worker.pipeline.request_slot(RequestId(0)).unwrap() as u8;

        run_iteration(&mut worker, slot, sender_group_id, &mut events);
        assert_eq!(worker.pipeline.current_kv(RequestId(0)), Some(8));
    }

    #[test]
    fn kv_full_head_stays_queued_until_release() {
        let store = prefilled_store(&[(0, 8, 4), (1, 8, 4)]);
        let config = WorkerConfig {
            attn_kv_bytes: 15,
            ..WorkerConfig::default()
        };
        let mut worker = worker_with_config(Rc::clone(&store), test_cluster(), config);
        worker.enqueue(AttnWorkerMsg::Admit { req: RequestId(0) });
        worker.enqueue(AttnWorkerMsg::Admit { req: RequestId(1) });
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        assert_eq!(worker.status().queued_requests, 1);
        assert_eq!(worker.pipeline.current_kv(RequestId(1)), None);

        worker.enqueue(AttnWorkerMsg::Release { req: RequestId(0) });
        worker.tick(Time::from_ms(1.0), &mut events);
        assert_eq!(worker.status().queued_requests, 0);
        assert_eq!(worker.pipeline.current_kv(RequestId(1)), Some(8));
    }

    #[test]
    fn least_kv_placement_spreads_equal_requests() {
        let store = prefilled_store(&[(0, 8, 4), (1, 8, 4)]);
        let mut worker = worker(Rc::clone(&store), test_cluster());
        for request in 0..2 {
            worker.enqueue(AttnWorkerMsg::Admit {
                req: RequestId(request),
            });
        }
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        assert_ne!(
            worker.pipeline.request_slot(RequestId(0)),
            worker.pipeline.request_slot(RequestId(1))
        );
    }

    #[test]
    fn release_drops_resident_kv_and_slot_membership() {
        let store = prefilled_store(&[(0, 8, 2)]);
        let mut worker = worker(Rc::clone(&store), test_cluster());
        worker.enqueue(AttnWorkerMsg::Admit { req: RequestId(0) });
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        worker.enqueue(AttnWorkerMsg::Release { req: RequestId(0) });
        assert_eq!(worker.pipeline.current_kv(RequestId(0)), None);
        assert_eq!(worker.pipeline.request_slot(RequestId(0)), None);
        assert_eq!(worker.status().active_requests, 0);
    }
}
