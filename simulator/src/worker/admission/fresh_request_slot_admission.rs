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
use crate::worker::kv::KvStore;
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
    K: KvStore,
{
    fn enqueue_fresh_request(&mut self, request: RequestId, context: &WorkerContext) {
        debug_assert!(
            !self.policy.contains(request),
            "AFD Admit is once-per-request; duplicate pending request {request:?}"
        );
        let (prompt_tokens, remaining) = {
            let mut store = context.requests.borrow_mut();
            let record = &mut store[request];
            let prompt_tokens = record.request.definition.prompt_tokens;
            let remaining = record
                .request
                .definition
                .target_output_tokens
                .saturating_sub(record.progress.output_tokens_emitted);
            let arrival = record.request.core.arrival_time;
            context.stamp_stage(record, arrival, AfdStage::Pending as u16);
            (prompt_tokens, remaining)
        };
        let candidate = self
            .enqueue_sequence
            .freeze(request, prompt_tokens, remaining);
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
            if candidate.decode == 0 {
                self.pop_selected(candidate);
                continue;
            }
            let footprint =
                kv_store.footprint(candidate.request, candidate.prompt, candidate.decode);
            if !kv_store.fits(0, &footprint) {
                break;
            }
            self.pop_selected(candidate);
            kv_store.reserve(candidate.request, 0, footprint);

            {
                let mut store = context.requests.borrow_mut();
                store.mark_admitted(candidate.request);
                context.stamp_stage(&mut store[candidate.request], now, AfdStage::Prefill as u16);
            }
            self.admitted_requests.push(candidate.request);
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
        self.policy.len() as u32
    }
}
