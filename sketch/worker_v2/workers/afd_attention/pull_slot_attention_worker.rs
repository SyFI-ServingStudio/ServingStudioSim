//! `PullSlotAttentionWorker<K, E>` — the PD-for-AFD *decode-attn* shell (matrix; the third
//! pool of the three-pool disaggregation). It drives the SAME `AttentionSlotPipeline` as the
//! colocated `SlotAttentionWorker` (interfaces doc §5: "share the private slot core, do NOT
//! embed one concrete worker inside another") — the entire layer-wise slot pipeline is
//! reused verbatim. The ONLY thing that changes is the INGRESS: instead of a
//! `FreshRequestSlotAdmission` reserve-on-admit, a request arrives already prefilled (a `Handoff`
//! from a prefill worker) and its prompt KV must be PULLED across the fabric before it
//! can decode. The pull front-end (`KvPullIngress`) is the same shape as `PullDecodeWorker`'s
//! (single in-flight `submit_transfer`, `TransferPlan`), except it acks the prefill
//! side with `AttnWorkerEvent::KvPullComplete` (the "PD-for-AFD only" event variant)
//! and hands the landed request to `AttentionSlotPipeline::place_request` — where the core's existing
//! prefilled-handoff path (`begin_decode` at the layer-0 boundary) commits it resident.
//!
//! Two interface results this exercises: (a) the AFD-attn family is axis-OPTIONAL like
//! AFD-ffn — this worker composes as `<K, E>` with NO admission axis (the pull front is
//! shell-only, thin like `PullDecodeWorker`'s decode lifecycle); (b) the L6 `AfdAttnWorker`
//! bound `Self::Msg: From<AttnWorkerMsg>` lets a PD variant carry a WIDER message enum
//! (`PullAttentionWorkerMsg` = the common control protocol + a `Handoff`) while the pool still
//! drives it through the common `From`. Prefill side reuses `build_pd_prefill_worker` (#4a);
//! decode-ffn side reuses `build_afd_ffn_worker` (#8) — this is the only genuinely new piece.

use std::collections::VecDeque;

use crate::common::{PdStage, RequestId, Time, WorkerId};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::iter_worker::{AfdAttnWorker, IterWorker};
use crate::worker::types::{AttnWorkerEvent, AttnWorkerMsg, TransferPlan, WorkerStatus};

use super::super::super::execution::AttentionLayerExecution;
use super::super::super::kv::{KvStore, SlotPipelineKv};
use super::super::super::shared::context::WorkerContext;
use super::attention_slot_pipeline::AttentionSlotPipeline;

/// Wider decode-attn message enum: the common AFD control protocol (via `From`) plus
/// the PD handoff. `Admit` is unused here — a fresh request goes to the PREFILL pool,
/// and this pool receives it as a `Handoff` once its prefill KV exists.
pub enum PullAttentionWorkerMsg {
    Common(AttnWorkerMsg),
    /// `req` was prefilled on `prefill_worker`; pull its `tokens` of prompt KV from
    /// `send_group_id` (the prefill worker's held-KV endpoint) before decoding here.
    Handoff {
        req: RequestId,
        send_group_id: u16,
        tokens: u64,
        prefill_worker: WorkerId,
    },
}

impl From<AttnWorkerMsg> for PullAttentionWorkerMsg {
    fn from(msg: AttnWorkerMsg) -> Self {
        PullAttentionWorkerMsg::Common(msg)
    }
}

#[derive(Clone, Copy)]
struct ActiveKvPull {
    req: RequestId,
    pull_end: Time,
    prefill_worker: WorkerId,
    tokens: u64,
}

/// The KV-pull ingress (comm seam on the shell). Single in-flight transfer (NCCL is
/// serial); on land it acks the prefill side and yields the request to the slot core.
struct KvPullIngress {
    cluster: SharedGpuCluster,
    receive_group_id: u16,
    kv_bytes_per_token: u64,
    /// Handoffs awaiting submission (not yet on the wire).
    pending_pulls: VecDeque<(RequestId, TransferPlan)>,
    active_pull: Option<ActiveKvPull>,
}

impl KvPullIngress {
    fn new(cluster: SharedGpuCluster, receive_group_id: u16, kv_bytes_per_token: u64) -> Self {
        Self {
            cluster,
            receive_group_id,
            kv_bytes_per_token,
            pending_pulls: VecDeque::new(),
            active_pull: None,
        }
    }

    /// Queue a handoff for pulling; fill the destination (this worker's recv group).
    fn enqueue_handoff(
        &mut self,
        req: RequestId,
        send_group_id: u16,
        tokens: u64,
        prefill_worker: WorkerId,
        context: &WorkerContext,
    ) {
        self.pending_pulls.push_back((
            req,
            TransferPlan {
                send_gid: send_group_id,
                recv_gid: self.receive_group_id,
                tokens,
                prefill_worker,
            },
        ));
        let arrival = { context.requests.borrow()[req].arrival_time };
        let mut store = context.requests.borrow_mut();
        context.stamp_stage(&mut store[req], arrival, PdStage::PendingDecode as u16);
    }

    /// Promote a landed pull (ack the prefill side via `KvPullComplete`, return the
    /// request for the shell to `place_request`), then submit the next queued_pull_count handoff if the
    /// wire is free. Rough: the only throttle is single-in-flight; the real 5% token
    /// backlog gate is deferred (a load heuristic, like `PullDecodeWorker`'s budget).
    fn drive_pulls(
        &mut self,
        context: &WorkerContext,
        now: Time,
        events: &mut Vec<AttnWorkerEvent>,
    ) -> Vec<RequestId> {
        let mut landed: Vec<RequestId> = Vec::new();
        loop {
            if let Some(pull) = self.active_pull {
                if now >= pull.pull_end {
                    self.active_pull = None;
                    {
                        let mut store = context.requests.borrow_mut();
                        context.stamp_stage(&mut store[pull.req], now, PdStage::Decode as u16);
                    }
                    // PD-for-AFD only: ack the originating prefill worker so it can
                    // drop its held reservation (L6 routes this side-band event).
                    events.push(AttnWorkerEvent::KvPullComplete {
                        worker: context.id,
                        req: pull.req,
                        prefill_worker: pull.prefill_worker,
                    });
                    landed.push(pull.req);
                } else {
                    break; // wire busy with a non-landed pull
                }
            }
            let Some((rid, transfer)) = self.pending_pulls.pop_front() else {
                break;
            };
            let bytes = transfer.tokens.saturating_mul(self.kv_bytes_per_token);
            let pull_end = self.cluster.borrow_mut().submit_transfer(
                now,
                transfer.send_gid,
                transfer.recv_gid,
                bytes,
                "pd_afd_kv_pull",
                "",
            );
            self.active_pull = Some(ActiveKvPull {
                req: rid,
                pull_end,
                prefill_worker: transfer.prefill_worker,
                tokens: transfer.tokens,
            });
            {
                let mut store = context.requests.borrow_mut();
                context.stamp_stage(&mut store[rid], now, PdStage::Transfer as u16);
            }
            // Loop: an instant transfer (pull_end <= now) lands this tick.
        }
        landed
    }

    /// Cancel a handoff still in the pull front (before it reaches a slot).
    fn cancel_pull(&mut self, req: RequestId) -> bool {
        if let Some(pos) = self.pending_pulls.iter().position(|(r, _)| *r == req) {
            self.pending_pulls.remove(pos);
            return true;
        }
        if self.active_pull.map(|pull| pull.req) == Some(req) {
            self.active_pull = None;
            return true;
        }
        false
    }

    fn next_wakeup(&self) -> Option<Time> {
        self.active_pull.map(|pull| pull.pull_end)
    }

    fn queued_pull_count(&self) -> u32 {
        (self.pending_pulls.len() + usize::from(self.active_pull.is_some())) as u32
    }

    /// Tokens in the pull front (queued_pull_count + in-flight) — the not-yet-resident KV the L6
    /// placement adds to the slot core's resident projected peak.
    fn pending_pull_kv_tokens(&self) -> u64 {
        let queued_tokens: u64 = self.pending_pulls.iter().map(|(_, t)| t.tokens).sum();
        queued_tokens + self.active_pull.map_or(0, |pull| pull.tokens)
    }
}

fn earliest_wakeup(first: Option<Time>, second: Option<Time>) -> Option<Time> {
    match (first, second) {
        (Some(first), Some(second)) => Some(first.min(second)),
        (Some(wakeup), None) | (None, Some(wakeup)) => Some(wakeup),
        (None, None) => None,
    }
}

pub struct PullSlotAttentionWorker<K, E>
where
    K: KvStore + SlotPipelineKv,
    E: AttentionLayerExecution,
{
    context: WorkerContext,
    pull: KvPullIngress,
    core: AttentionSlotPipeline<K, E>,
}

impl<K, E> PullSlotAttentionWorker<K, E>
where
    K: KvStore + SlotPipelineKv,
    E: AttentionLayerExecution,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_components(
        context: WorkerContext,
        kv_store: K,
        execution: E,
        cluster: SharedGpuCluster,
        receive_group_id: u16,
        num_layers: u16,
        kv_bytes_per_token: u64,
    ) -> Self {
        Self {
            context,
            pull: KvPullIngress::new(cluster.clone(), receive_group_id, kv_bytes_per_token),
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
        // Ingress: advance pulls; each landed handoff is placed into a slot (the core's
        // begin_decode commits its pulled KV resident at the next layer-0 boundary).
        let landed = self.pull.drive_pulls(&self.context, now, events);
        for req in landed {
            self.core.place_request(req);
        }
        self.core.drive_slots(&self.context, now, events);
        earliest_wakeup(self.pull.next_wakeup(), self.core.next_wakeup(now))
    }

    fn on_msg_handoff(
        &mut self,
        req: RequestId,
        send_group_id: u16,
        tokens: u64,
        prefill_worker: WorkerId,
    ) {
        self.pull
            .enqueue_handoff(req, send_group_id, tokens, prefill_worker, &self.context);
    }

    fn on_msg_common(&mut self, msg: AttnWorkerMsg) {
        match msg {
            AttnWorkerMsg::ReadyNotification {
                slot,
                layer,
                send_gid: send_group_id,
                bytes,
            } => self
                .core
                .on_msg_ready_notification(slot as usize, layer, send_group_id, bytes),
            AttnWorkerMsg::SlotFlushed { slot, layer } => {
                self.core.on_msg_slot_flushed(slot as usize, layer)
            }
            AttnWorkerMsg::Release { req } => {
                if !self.core.release_slotted_request(req) {
                    self.pull.cancel_pull(req);
                }
            }
            // Fresh requests go to the prefill pool and return as a handoff.
            AttnWorkerMsg::Admit { .. } => {}
        }
    }
}

impl<K, E> IterWorker for PullSlotAttentionWorker<K, E>
where
    K: KvStore + SlotPipelineKv,
    E: AttentionLayerExecution,
{
    type Msg = PullAttentionWorkerMsg;
    type Event = AttnWorkerEvent;

    fn id(&self) -> WorkerId {
        self.context.id
    }

    fn enqueue(&mut self, msg: PullAttentionWorkerMsg) {
        match msg {
            PullAttentionWorkerMsg::Handoff {
                req,
                send_group_id,
                tokens,
                prefill_worker,
            } => self.on_msg_handoff(req, send_group_id, tokens, prefill_worker),
            PullAttentionWorkerMsg::Common(msg) => self.on_msg_common(msg),
        }
    }

    fn tick(&mut self, now: Time, events: &mut Vec<AttnWorkerEvent>) -> Option<Time> {
        self.drive_slot_pipeline(now, events)
    }

    fn status(&self) -> WorkerStatus {
        WorkerStatus {
            queued_requests: self.pull.queued_pull_count(),
            active_requests: self.core.active_slot_count(),
        }
    }
}

impl<K, E> AfdAttnWorker for PullSlotAttentionWorker<K, E>
where
    K: KvStore + SlotPipelineKv,
    E: AttentionLayerExecution,
{
    /// L3 resident projected peak (slot core) + not-yet-resident pull-front tokens.
    fn estimated_peak_kv(&self) -> u64 {
        self.core.estimated_peak_kv() + self.pull.pending_pull_kv_tokens()
    }
}
