//! `PrefixPrefillDecodeAdmission<P>` — the prefix-cache-aware lifecycle (F1).
//! Identical to `LocalPrefillDecodeAdmission` EXCEPT it escalates its KV bound
//! to `K: IterWorkerKv + PrefixCacheKv` (the same M2 capability escalation as
//! `ChunkedPrefillAdmission`'s `ChunkedPrefillKv`) and, at admit, probes every
//! KV partition, chooses the best feasible prefix placement, and stamps
//! the record's `prefix_kv` — the resident prior context that
//! `UnifiedIterExecution::build_iteration_input` renders. `KvStore::reserve`
//! records the chosen partition,
//! making the request sticky through prefill/decode/release. External calls verified:
//! `RequestStore::mark_admitted`,
//! `RequestRecord::{record_token,record_first_token,is_complete}`.

use crate::common::{RequestId, Time, UnifiedStage};
use crate::worker::types::{WorkerEventCommon, WorkerMsgCommon};

use super::super::kv::{
    IterWorkerKv, KvCapacityPressure, KvStore, PrefixCacheKv, PrefixPlacementProbe,
};
use super::super::shared::advance_scope::{AdvanceScope, PartitionId};
use super::super::shared::context::WorkerContext;
use super::policy::{AdmissionCandidate, PendingOrderPolicy};
use super::IterAdmission;

pub struct PrefixPrefillDecodeAdmission<P: PendingOrderPolicy> {
    /// The pending queue lives inside the policy; `form_batch` touches only its head.
    policy: P,
    policy_ctx: P::Context,
}

struct ChosenPrefixPlacement<Footprint> {
    partition: PartitionId,
    probe: PrefixPlacementProbe<Footprint>,
    pressure: KvCapacityPressure,
}

impl<P: PendingOrderPolicy> PrefixPrefillDecodeAdmission<P> {
    pub fn new(policy: P, policy_ctx: P::Context) -> Self {
        Self { policy, policy_ctx }
    }

    /// Prefer more reusable prefix, then lower normalized KV pressure, then the
    /// lowest partition id for deterministic ties.
    fn placement_precedes<Footprint>(
        candidate: &ChosenPrefixPlacement<Footprint>,
        current: &ChosenPrefixPlacement<Footprint>,
    ) -> bool {
        if candidate.probe.matched_tokens != current.probe.matched_tokens {
            return candidate.probe.matched_tokens > current.probe.matched_tokens;
        }
        let candidate_used =
            candidate.pressure.resident_tokens_equiv + candidate.pressure.reserved_tokens_equiv;
        let current_used =
            current.pressure.resident_tokens_equiv + current.pressure.reserved_tokens_equiv;
        let candidate_scaled =
            u128::from(candidate_used) * u128::from(current.pressure.capacity_tokens_equiv.max(1));
        let current_scaled =
            u128::from(current_used) * u128::from(candidate.pressure.capacity_tokens_equiv.max(1));
        candidate_scaled < current_scaled
            || (candidate_scaled == current_scaled && candidate.partition < current.partition)
    }

    fn choose_placement<K: KvStore + PrefixCacheKv>(
        kv_store: &K,
        candidate: &AdmissionCandidate,
    ) -> Option<ChosenPrefixPlacement<K::Footprint>> {
        let mut chosen: Option<ChosenPrefixPlacement<K::Footprint>> = None;
        for partition in 0..kv_store.num_partitions() as PartitionId {
            let probe = kv_store.probe_prefix(
                partition,
                candidate.request,
                candidate.prompt,
                candidate.decode,
            );
            if !kv_store.fits(partition, &probe.footprint) {
                continue;
            }
            let placement = ChosenPrefixPlacement {
                partition,
                probe,
                pressure: kv_store.pressure(partition),
            };
            if chosen
                .as_ref()
                .is_none_or(|current| Self::placement_precedes(&placement, current))
            {
                chosen = Some(placement);
            }
        }
        chosen
    }
}

impl<P: PendingOrderPolicy, K: KvStore + IterWorkerKv + PrefixCacheKv> IterAdmission<K>
    for PrefixPrefillDecodeAdmission<P>
{
    type Msg = WorkerMsgCommon;
    type Event = WorkerEventCommon;

    fn accept_message(&mut self, kv_store: &mut K, msg: WorkerMsgCommon, context: &WorkerContext) {
        let WorkerMsgCommon::Request(rid) = msg;
        let candidate = {
            let mut store = context.requests.borrow_mut();
            let record = &mut store[rid];
            let arrival = record.arrival_time;
            let prompt = record.prompt_len;
            context.stamp_stage(record, arrival, UnifiedStage::Pending as u16);
            AdmissionCandidate {
                request: rid,
                arrival_seq: arrival.0,
                prompt,
                decode: record.decode_len,
                deadline: None,
                // Arrival-time best-partition match SNAPSHOT for policy ranking. Admit
                // re-probes all partitions, so cache drift cannot corrupt accounting.
                matched_tokens: (0..kv_store.num_partitions() as PartitionId)
                    .map(|partition| {
                        kv_store
                            .probe_prefix(partition, rid, prompt, record.decode_len)
                            .matched_tokens
                    })
                    .max()
                    .unwrap_or(0),
            }
        };
        self.policy.push(candidate, &mut self.policy_ctx);
    }

    fn form_batch(&mut self, kv_store: &mut K, context: &WorkerContext, now: Time) -> bool {
        let num_partitions = kv_store.num_partitions();
        let had_decode =
            (0..num_partitions as PartitionId).any(|partition| kv_store.has_live_decode(partition));

        // Head only: one fresh prefill per iter, O(1) whatever the backlog depth.
        if let Some(candidate) = self.policy.peek() {
            if let Some(placement) = Self::choose_placement(kv_store, &candidate) {
                kv_store.reserve(
                    candidate.request,
                    placement.partition,
                    placement.probe.footprint,
                );
                self.policy.pop(&mut self.policy_ctx);
                let mut store = context.requests.borrow_mut();
                store.mark_admitted(candidate.request);
                let record = &mut store[candidate.request];
                record.prefix_kv = placement.probe.matched_tokens;
                record.active_chunk_len = candidate.prompt;
                context.stamp_stage(record, now, UnifiedStage::Prefill as u16);
            }
            // No feasible partition: the head stays queued (peek did not dequeue it).
        }

        kv_store.drain_ready();
        had_decode
            || (0..num_partitions as PartitionId)
                .any(|partition| kv_store.has_prefill_admit(partition))
    }

    fn complete_iteration(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        events: &mut Vec<WorkerEventCommon>,
        now: Time,
    ) {
        let num_partitions = kv_store.num_partitions();
        let mut completed: Vec<(PartitionId, RequestId)> = Vec::new();

        for partition in 0..num_partitions as PartitionId {
            // (a) live decodes each produced one token on their sticky partition.
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
                        completed.push((partition, rid));
                    }
                }
            }
            kv_store.advance(AdvanceScope::WholePartition(partition), 1);

            // (b) resolved prefills enter decode on the SAME chosen partition.
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
                        completed.push((partition, rid));
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
        }

        for (partition, rid) in completed {
            kv_store.release(rid, partition);
            events.push(WorkerEventCommon::RequestComplete {
                worker: context.id,
                req: rid,
            });
        }
        for partition in 0..num_partitions as PartitionId {
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

#[cfg(test)]
mod tests {
    use crate::common::{PoolId, RequestId, Time, WorkerId};
    use crate::test_helpers::shared_with;
    use crate::worker::admission_helpers::KvAdmission;

    use super::super::super::kv::ModeledPrefixCacheKv;
    use super::super::super::shared::context::WorkerContext;
    use super::super::policy::FifoOrder;
    use super::*;

    fn test_ctx(requests: &[(u32, u32, u32)]) -> WorkerContext {
        WorkerContext {
            id: WorkerId(0),
            pool: PoolId(0),
            requests: shared_with(requests),
            log_output_token_times: false,
            log_stage_transitions: false,
        }
    }

    #[test]
    fn chooses_best_prefix_partition_and_keeps_decode_sticky() {
        let context = test_ctx(&[(0, 8, 3)]);
        let mut kv_store =
            ModeledPrefixCacheKv::new(2, 100, KvAdmission::Strict, None, vec![25, 75]);
        let mut admission = PrefixPrefillDecodeAdmission::new(FifoOrder::new(), ());
        admission.accept_message(
            &mut kv_store,
            WorkerMsgCommon::Request(RequestId(0)),
            &context,
        );

        assert!(admission.form_batch(&mut kv_store, &context, Time::ZERO));
        assert!(kv_store.prefill_admits(0).is_empty());
        assert_eq!(kv_store.prefill_admits(1), vec![RequestId(0)]);
        assert_eq!(context.requests.borrow()[RequestId(0)].prefix_kv, 6);

        admission.complete_iteration(&mut kv_store, &context, &mut Vec::new(), Time::from_ms(1.0));
        assert!(kv_store.decode_members(0).is_empty());
        assert_eq!(kv_store.decode_members(1), vec![(RequestId(0), 14)]);
    }

    #[test]
    fn falls_back_when_best_prefix_partition_cannot_fit() {
        let context = test_ctx(&[(0, 8, 2)]);
        // Partition 1 matches all 8 tokens, making its conservative footprint 18;
        // partition 0 has no hit and needs 10, so only partition 0 fits capacity 12.
        let mut kv_store =
            ModeledPrefixCacheKv::new(2, 12, KvAdmission::Strict, None, vec![0, 100]);
        let mut admission = PrefixPrefillDecodeAdmission::new(FifoOrder::new(), ());
        admission.accept_message(
            &mut kv_store,
            WorkerMsgCommon::Request(RequestId(0)),
            &context,
        );

        assert!(admission.form_batch(&mut kv_store, &context, Time::ZERO));
        assert_eq!(kv_store.prefill_admits(0), vec![RequestId(0)]);
        assert!(kv_store.prefill_admits(1).is_empty());
        assert_eq!(context.requests.borrow()[RequestId(0)].prefix_kv, 0);
    }
}
