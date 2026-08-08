//! PD-prefill lifecycle: fresh request → local prefill → held KV handoff.

use crate::common::{PdStage, RequestId, Time};
use crate::worker::admission::{
    AdmissionCandidate, EnqueueSequence, IterAdmission, PendingOrderPolicy,
};
use crate::worker::kv::{HandoffKv, PrefixKv, PrefixResolution};
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::{PdPrefillEvent, PdPrefillMsg};

pub struct PrefillHandoffAdmission<P: PendingOrderPolicy> {
    policy: P,
    policy_context: P::Context,
    enqueue_sequence: EnqueueSequence,
    send_group_id: u16,
    /// Reused deferred transitions avoid mutating KV while its membership view is
    /// borrowed, without allocating a fresh membership snapshot each iteration.
    transitions: Vec<(RequestId, u64, bool)>,
}

impl<P: PendingOrderPolicy> PrefillHandoffAdmission<P> {
    pub(crate) fn new(policy: P, policy_context: P::Context, send_group_id: u16) -> Self {
        Self {
            policy,
            policy_context,
            enqueue_sequence: EnqueueSequence::default(),
            send_group_id,
            transitions: Vec::new(),
        }
    }

    fn admit<K: PrefixKv>(
        &self,
        kv_store: &mut K,
        context: &WorkerContext,
        candidate: AdmissionCandidate,
        resolution: PrefixResolution,
        footprint: K::Footprint,
        now: Time,
    ) {
        let partition = 0;
        kv_store.reserve_prefix(candidate.request, partition, resolution, footprint);

        let mut store = context.requests.borrow_mut();
        store.mark_admitted(candidate.request);
        let record = &mut store[candidate.request];
        context.stamp_stage(record, now, PdStage::Prefill as u16);
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

impl<P, K> IterAdmission<K> for PrefillHandoffAdmission<P>
where
    P: PendingOrderPolicy,
    K: HandoffKv + PrefixKv,
{
    type Msg = PdPrefillMsg;
    type Event = PdPrefillEvent;

    fn accept_message(&mut self, kv_store: &mut K, msg: Self::Msg, context: &WorkerContext) {
        match msg {
            PdPrefillMsg::Request(request) => {
                debug_assert!(
                    !self.policy.contains(request),
                    "PD prefill admission is once-per-request; duplicate pending request {request:?}"
                );
                let (prompt, prefix) = {
                    let mut store = context.requests.borrow_mut();
                    let record = &mut store[request];
                    let prompt = record.request.definition.prompt_tokens;
                    let prefix = record.request.definition.prefix;
                    let arrival = record.request.core.arrival_time;
                    context.stamp_stage(record, arrival, PdStage::PendingPrefill as u16);
                    (prompt, prefix)
                };
                let candidate = self.enqueue_sequence.freeze(request, prompt, 0, prefix);
                self.policy.push(candidate, &mut self.policy_context);
            }
            PdPrefillMsg::ReleaseKv { req } => kv_store.complete_handoff(req),
        }
    }

    fn form_batch(&mut self, kv_store: &mut K, context: &WorkerContext, now: Time) -> bool {
        let partition = 0;
        if let Some(candidate) = self.policy.peek() {
            let resolution = kv_store.plan_prefix(partition, candidate.prompt, candidate.prefix);
            let footprint =
                kv_store.footprint(candidate.request, resolution.initial_context_tokens(), 0);
            if kv_store.fits(partition, &footprint) {
                self.pop_selected(candidate);
                self.admit(kv_store, context, candidate, resolution, footprint, now);
            }
        }

        kv_store.drain_ready();
        kv_store.has_prefill_admit(partition)
    }

    fn complete_iteration(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        events: &mut Vec<Self::Event>,
        now: Time,
    ) {
        let partition = 0;
        self.transitions.clear();
        {
            let mut store = context.requests.borrow_mut();
            kv_store.visit_prefill_admits(partition, |request| {
                let record = &mut store[request];
                record.progress.prefill_tokens_processed = kv_store.prefill_compute_tokens(request);
                record.record_first_token(now, context.log_tokens());
                let kv_tokens = kv_store.initial_context_tokens(request);
                let complete = record.is_complete();
                context.stamp_stage(
                    record,
                    now,
                    if complete {
                        PdStage::Done
                    } else {
                        PdStage::PrefillDoneAwaitPull
                    } as u16,
                );
                self.transitions.push((request, kv_tokens, complete));
            });
        }

        for &(request, kv_tokens, complete) in &self.transitions {
            if complete {
                kv_store.release_retaining_prefix(request, partition);
                events.push(PdPrefillEvent::RequestComplete {
                    worker: context.id,
                    req: request,
                });
            } else {
                kv_store.hold(partition, request, kv_tokens);
                events.push(PdPrefillEvent::PrefillDone {
                    worker: context.id,
                    req: request,
                    send_gid: self.send_group_id,
                    kv_tokens,
                });
            }
        }
        kv_store.clear_prefill_admits(partition);
        kv_store.sample_submit(partition, now);
    }

    fn queued_requests(&self) -> u32 {
        self.policy.len() as u32
    }

    fn cancel_pending(&mut self, request: RequestId) -> bool {
        self.policy.remove(request).is_some()
    }
}
