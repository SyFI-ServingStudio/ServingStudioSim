//! Fresh-request admission for the layer-wise AFD attention family.
//!
//! `Admit` is level 1: freeze candidate facts, hand membership to the selection
//! policy, and stamp `Pending`. At the next worker tick, level 2 repeatedly checks
//! the selected head against KV capacity, reserves its full footprint, stamps
//! `Prefill`, and hands admitted ids to the slot shell. A blocked head stays queued.

use crate::common::{AfdStage, RequestId, Time};
use crate::worker::admission::{
    AdmissionCandidate, EnqueueSequence, PendingOrderPolicy, SlotPipelineAdmission,
};
use crate::worker::kv::PrefixKv;
use crate::worker::shared::context::WorkerContext;

pub struct FreshRequestSlotAdmission<P: PendingOrderPolicy> {
    policy: P,
    policy_context: P::Context,
    enqueue_sequence: EnqueueSequence,
    /// Reused handoff buffer: the shell needs admitted ids after the KV borrow ends.
    admitted_requests: Vec<RequestId>,
}

impl<P: PendingOrderPolicy> FreshRequestSlotAdmission<P> {
    pub(crate) fn new(policy: P, policy_context: P::Context) -> Self {
        Self {
            policy,
            policy_context,
            enqueue_sequence: EnqueueSequence::default(),
            admitted_requests: Vec::new(),
        }
    }

    fn pop_selected(&mut self, candidate: AdmissionCandidate) {
        let popped = self.policy.pop(&mut self.policy_context);
        debug_assert_eq!(
            popped,
            Some(candidate),
            "policy pop must remove peeked head"
        );
    }
}

impl<P, K> SlotPipelineAdmission<K> for FreshRequestSlotAdmission<P>
where
    P: PendingOrderPolicy,
    K: PrefixKv,
{
    fn enqueue_fresh_request(&mut self, kv_store: &K, request: RequestId, context: &WorkerContext) {
        debug_assert!(
            !self.policy.contains(request),
            "AFD Admit is once-per-request; duplicate pending request {request:?}"
        );
        let (fresh_prompt_tokens, remaining_output_tokens, session_input, conversation_start_time) = {
            let mut store = context.requests.borrow_mut();
            let record = &mut store[request];
            let fresh_prompt_tokens = record.request.definition.prompt_tokens;
            let remaining_output_tokens = record
                .request
                .definition
                .target_output_tokens
                .saturating_sub(record.progress.output_tokens_emitted);
            let session_input = record.request.definition.session;
            let arrival_time = record.request.core.arrival_time;
            let conversation_start_time = session_input.session_start_or(arrival_time);
            context.stamp_stage(record, arrival_time, AfdStage::Pending as u16);
            (
                fresh_prompt_tokens,
                remaining_output_tokens,
                session_input,
                conversation_start_time,
            )
        };
        let candidate = self.enqueue_sequence.freeze(
            request,
            fresh_prompt_tokens,
            remaining_output_tokens,
            session_input,
            conversation_start_time,
            kv_store.resident_prefix_tokens(fresh_prompt_tokens, session_input),
        );
        self.policy.push(candidate, &mut self.policy_context);
    }

    fn reserve_fitting_requests<'a>(
        &'a mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        now: Time,
    ) -> &'a [RequestId] {
        self.admitted_requests.clear();
        while let Some(candidate) = self.policy.peek() {
            if candidate.remaining_output_tokens == 0 {
                self.pop_selected(candidate);
                continue;
            }
            let resolved_prefill = kv_store.preview_prefill_context(
                0,
                candidate.fresh_prompt_tokens,
                candidate.session_input,
            );
            let footprint = kv_store.footprint(
                candidate.request_id,
                resolved_prefill.post_prefill_context_tokens(),
                candidate.remaining_output_tokens,
            );
            if !kv_store.fits(0, &footprint) {
                break;
            }
            self.pop_selected(candidate);
            kv_store.reserve_prefill_context(
                candidate.request_id,
                0,
                resolved_prefill,
                footprint,
                now,
            );

            {
                let mut store = context.requests.borrow_mut();
                store.mark_admitted(candidate.request_id);
                let record = &mut store[candidate.request_id];
                record.record_prefix_cache_hit_tokens(resolved_prefill.resident_prefix_tokens());
                context.stamp_stage(record, now, AfdStage::Prefill as u16);
            }
            self.admitted_requests.push(candidate.request_id);
        }
        &self.admitted_requests
    }

    fn cancel_pending(&mut self, request: RequestId) -> bool {
        self.policy.remove(request).is_some()
    }

    fn queued_kv_tokens(&self) -> u64 {
        self.policy.queued_kv_tokens()
    }

    fn queued_requests(&self) -> u32 {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "queued request count is bounded by this worker's admission queue capacity, far below u32::MAX"
        )]
        let count = self.policy.len() as u32;
        count
    }
}
