//! `DraftVerifyAdmission<P>` — A1 fresh/prefill admission paired with S6
//! runtime-variable speculative completion.
//!
//! Queueing and batch formation delegate to `LocalPrefillDecodeAdmission`; only
//! completion differs. The execution result names the exact requests and token
//! counts to commit, so this lifecycle never invents an accepted length.

use crate::common::{RequestId, Time, UnifiedStage};
use crate::worker::types::{WorkerEventCommon, WorkerMsgCommon};

use super::super::execution::DraftVerifyResult;
use super::super::kv::SpeculativeKv;
use super::super::shared::context::WorkerContext;
use super::policy::PendingOrderPolicy;
use super::{DraftVerifyAdmissionLifecycle, IterAdmission, LocalPrefillDecodeAdmission};

pub struct DraftVerifyAdmission<P: PendingOrderPolicy> {
    local_admission: LocalPrefillDecodeAdmission<P>,
}

impl<P: PendingOrderPolicy> DraftVerifyAdmission<P> {
    pub fn new(local_admission: LocalPrefillDecodeAdmission<P>) -> Self {
        Self { local_admission }
    }
}

impl<P, K> DraftVerifyAdmissionLifecycle<K> for DraftVerifyAdmission<P>
where
    P: PendingOrderPolicy,
    K: SpeculativeKv,
{
    type Msg = WorkerMsgCommon;
    type Event = WorkerEventCommon;

    fn accept_message(&mut self, kv_store: &mut K, msg: Self::Msg, context: &WorkerContext) {
        self.local_admission.accept_message(kv_store, msg, context);
    }

    fn form_batch(&mut self, kv_store: &mut K, context: &WorkerContext, now: Time) -> bool {
        self.local_admission.form_batch(kv_store, context, now)
    }

    fn complete_draft_verify_iteration(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        result: DraftVerifyResult,
        events: &mut Vec<Self::Event>,
        now: Time,
    ) {
        let num_partitions = kv_store.num_partitions();
        let mut completed_by_partition = vec![Vec::<RequestId>::new(); num_partitions];

        // Resolve every tentative proposal using the execution result. Store and
        // KV advance by the exact same clamped request-local count.
        {
            let log_tokens = context.log_tokens();
            let mut store = context.requests.borrow_mut();
            for outcome in result.requests {
                let record = &mut store[outcome.request];
                let remaining = record.decode_len.saturating_sub(record.tokens_emitted);
                let committed_tokens = outcome.committed_tokens.min(remaining);
                assert!(
                    committed_tokens > 0,
                    "a live draft/verify request must commit at least one token"
                );
                for _ in 0..committed_tokens {
                    record.record_token(now, log_tokens);
                }
                kv_store.commit_accepted(outcome.request, outcome.partition, committed_tokens);
                kv_store.discard_rejected(outcome.request, outcome.partition);
                if record.is_complete() {
                    context.stamp_stage(record, now, UnifiedStage::Done as u16);
                    completed_by_partition[outcome.partition as usize].push(outcome.request);
                }
            }
        }

        // Prefills in the same mixed batch still follow ordinary A1 semantics:
        // emit their first token, then either finish or enter decode. They do not
        // participate in draft/verify until the next iteration.
        for partition in 0..num_partitions as u16 {
            let mut to_finalize: Vec<(RequestId, u64, u32)> = Vec::new();
            {
                let log_tokens = context.log_tokens();
                let prefills = kv_store.prefill_admits(partition);
                let mut store = context.requests.borrow_mut();
                for request in prefills {
                    let record = &mut store[request];
                    record.prefill_processed = record.prompt_len;
                    record.record_first_token(now, log_tokens);
                    if record.is_complete() {
                        context.stamp_stage(record, now, UnifiedStage::Done as u16);
                        completed_by_partition[partition as usize].push(request);
                    } else {
                        let kv_len = u64::from(record.prompt_len + record.prefix_kv);
                        let remaining = record.decode_len.saturating_sub(record.tokens_emitted);
                        context.stamp_stage(record, now, UnifiedStage::Decode as u16);
                        to_finalize.push((request, kv_len, remaining));
                    }
                }
            }
            for (request, kv_len, remaining) in to_finalize {
                kv_store.commit_resident(request, partition, kv_len, remaining);
            }
            kv_store.clear_prefill_admits(partition);

            for request in completed_by_partition[partition as usize].drain(..) {
                kv_store.release(request, partition);
                events.push(WorkerEventCommon::RequestComplete {
                    worker: context.id,
                    req: request,
                });
            }
            kv_store.sample_submit(partition, now);
        }
    }

    fn queued_requests(&self) -> u32 {
        <LocalPrefillDecodeAdmission<P> as IterAdmission<K>>::queued_requests(&self.local_admission)
    }

    fn cancel_pending(&mut self, request: RequestId) -> bool {
        <LocalPrefillDecodeAdmission<P> as IterAdmission<K>>::cancel_pending(
            &mut self.local_admission,
            request,
        )
    }
}
