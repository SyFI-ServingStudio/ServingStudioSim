//! `MultiModelAdmission<P>` — the multi-model co-serve lifecycle (multi-arch).
//! Escalates its bound to `K: IterWorkerKv + ModelSwitchKv`: a `SwitchModel`
//! control message routes
//! through `kv_store.set_active`, and fresh admits land in whichever model is active (its KV
//! partition). Iteration completion loops ALL partitions — every co-resident model
//! decodes concurrently, so all are drained each iter (the execution `Max`es their cost).
//!
//! This is the multi-model analogue of PD's wider message enum: the lifecycle carries a
//! `MultiModelAdmissionMsg { Request | SwitchModel }` while `IterBatchWorker`
//! transports it unchanged via `A::Msg`. The lifecycle constrains only on the
//! `ModelSwitchKv` read-view — it never names `ModelPartitionedKv`. External calls
//! verified: `RequestStore::mark_admitted`,
//! `RequestRecord::{record_token,record_first_token,is_complete}`.

use crate::common::{RequestId, Time, UnifiedStage};
use crate::worker::types::WorkerEventCommon;

use super::super::kv::{IterWorkerKv, KvStore, ModelId, ModelSwitchKv};
use super::super::shared::advance_scope::AdvanceScope;
use super::super::shared::context::WorkerContext;
use super::policy::{AdmissionCandidate, PendingOrderPolicy};
use super::IterAdmission;

/// Wider iter-family message: an ordinary request, or a control message switching which
/// co-resident model subsequent admits target.
pub enum MultiModelAdmissionMsg {
    Request(RequestId),
    SwitchModel(ModelId),
}

pub struct MultiModelAdmission<P: PendingOrderPolicy> {
    /// The pending queue lives inside the policy; `form_batch` touches only its head.
    policy: P,
    policy_ctx: P::Context,
}

impl<P: PendingOrderPolicy> MultiModelAdmission<P> {
    pub fn new(policy: P, policy_ctx: P::Context) -> Self {
        Self { policy, policy_ctx }
    }
}

impl<P: PendingOrderPolicy, K: KvStore + IterWorkerKv + ModelSwitchKv> IterAdmission<K>
    for MultiModelAdmission<P>
{
    type Msg = MultiModelAdmissionMsg;
    type Event = WorkerEventCommon;

    fn accept_message(
        &mut self,
        kv_store: &mut K,
        msg: MultiModelAdmissionMsg,
        context: &WorkerContext,
    ) {
        match msg {
            MultiModelAdmissionMsg::Request(rid) => {
                let candidate = {
                    let mut store = context.requests.borrow_mut();
                    let record = &mut store[rid];
                    let arrival = record.arrival_time;
                    context.stamp_stage(record, arrival, UnifiedStage::Pending as u16);
                    AdmissionCandidate {
                        request: rid,
                        arrival_seq: arrival.0,
                        prompt: record.prompt_len,
                        decode: record.decode_len,
                        deadline: None,
                        matched_tokens: 0,
                    }
                };
                self.policy.push(candidate, &mut self.policy_ctx);
            }
            // Control message: route subsequent admits (+ the execution) to `model`'s pool.
            MultiModelAdmissionMsg::SwitchModel(model) => kv_store.set_active(model),
        }
    }

    fn form_batch(&mut self, kv_store: &mut K, context: &WorkerContext, now: Time) -> bool {
        let num_partitions = kv_store.num_partitions();
        let had_live_decode =
            (0..num_partitions as u16).any(|partition| kv_store.has_live_decode(partition));

        // One fresh prefill per iter, into the ACTIVE model's partition. Head only.
        if let Some(candidate) = self.policy.peek() {
            let target = kv_store.active().0;
            let footprint =
                kv_store.footprint(candidate.request, candidate.prompt, candidate.decode);
            if kv_store.fits(target, &footprint) {
                kv_store.reserve(candidate.request, target, footprint);
                self.policy.pop(&mut self.policy_ctx);
                let mut store = context.requests.borrow_mut();
                store.mark_admitted(candidate.request);
                let record = &mut store[candidate.request];
                record.active_chunk_len = candidate.prompt;
                record.prefix_kv = 0;
                context.stamp_stage(record, now, UnifiedStage::Prefill as u16);
            }
            // Declined: the head stays queued (peek did not dequeue it).
        }

        kv_store.drain_ready();
        had_live_decode
            || (0..num_partitions as u16).any(|partition| kv_store.has_prefill_admit(partition))
    }

    fn complete_iteration(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        events: &mut Vec<WorkerEventCommon>,
        now: Time,
    ) {
        let num_partitions = kv_store.num_partitions();
        for partition in 0..num_partitions as u16 {
            let mut completed: Vec<RequestId> = Vec::new();
            {
                let log_tokens = context.log_tokens();
                let members = kv_store.decode_members(partition);
                let mut store = context.requests.borrow_mut();
                for (rid, _current_kv) in members {
                    let record = &mut store[rid];
                    if record.tokens_emitted < record.decode_len {
                        record.record_token(now, log_tokens);
                    }
                    if record.is_complete() {
                        context.stamp_stage(record, now, UnifiedStage::Done as u16);
                        completed.push(rid);
                    }
                }
            }
            kv_store.advance(AdvanceScope::WholePartition(partition), 1);

            let mut to_finalize: Vec<(RequestId, u64, u32)> = Vec::new();
            {
                let log_tokens = context.log_tokens();
                let prefills = kv_store.prefill_admits(partition);
                let mut store = context.requests.borrow_mut();
                for rid in prefills {
                    let record = &mut store[rid];
                    record.prefill_processed = record.prompt_len;
                    record.record_first_token(now, log_tokens);
                    if record.is_complete() {
                        context.stamp_stage(record, now, UnifiedStage::Done as u16);
                        completed.push(rid);
                    } else {
                        let kv_len = u64::from(record.prompt_len + record.prefix_kv);
                        let remaining = record.decode_len.saturating_sub(record.tokens_emitted);
                        context.stamp_stage(record, now, UnifiedStage::Decode as u16);
                        to_finalize.push((rid, kv_len, remaining));
                    }
                }
            }
            for (rid, kv_len, remaining) in to_finalize {
                kv_store.commit_resident(rid, partition, kv_len, remaining);
            }
            kv_store.clear_prefill_admits(partition);

            for rid in completed {
                kv_store.release(rid, partition);
                events.push(WorkerEventCommon::RequestComplete {
                    worker: context.id,
                    req: rid,
                });
            }
            kv_store.sample_submit(partition, now);
        }
    }

    #[inline]
    fn queued_requests(&self) -> u32 {
        self.policy.len() as u32
    }

    #[inline]
    fn cancel_pending(&mut self, rid: RequestId) -> bool {
        self.policy.remove(rid)
    }
}
