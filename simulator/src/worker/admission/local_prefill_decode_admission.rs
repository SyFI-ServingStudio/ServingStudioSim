//! Local prefill→decode lifecycle for the whole-iteration worker family.
//!
//! The policy owns pending membership and selection order; this lifecycle owns
//! token/KV gates and prefill/decode/done transitions. Candidate facts are frozen
//! once at enqueue, so batch formation never re-reads the request store.

use crate::common::{RequestId, SessionInput, Time, UnifiedStage};
use crate::worker::admission::{
    prefill_fits_budget, AdmissionCandidate, EnqueueSequence, IterAdmission, LoadBalance,
    PendingOrderPolicy,
};
use crate::worker::kv::{IterWorkerKv, PrefixKv, ResolvedPrefillContext};
use crate::worker::shared::advance_scope::AdvanceScope;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::{IterBatchPlan, WorkerEventCommon, WorkerMsgCommon};

/// Resolved prefill context and its matching full-context KV footprint must be
/// reserved together; keeping them paired prevents admission from mixing facts
/// produced by different cache snapshots.
struct PlannedReservation<Footprint> {
    resolved_prefill: ResolvedPrefillContext,
    footprint: Footprint,
}

/// Per-partition token usage accumulated while one batch is being formed.
///
/// The admission object reuses this scratch storage across iterations so the
/// unified single-/multi-partition path does not allocate on every batch.
#[derive(Clone, Copy, Debug, Default)]
struct PartitionTokenUsage {
    live_decode_tokens: u32,
    admitted_prefill_tokens: u32,
}

pub struct LocalPrefillDecodeAdmission<P: PendingOrderPolicy> {
    policy: P,
    policy_context: P::Context,
    enqueue_sequence: EnqueueSequence,
    per_partition_token_budget: u32,
    balance: LoadBalance,
    partition_token_usage: Vec<PartitionTokenUsage>,
}

impl<P: PendingOrderPolicy> LocalPrefillDecodeAdmission<P> {
    pub(crate) fn new(
        policy: P,
        policy_context: P::Context,
        max_batch_tokens: Option<u32>,
        balance: LoadBalance,
    ) -> Self {
        Self {
            policy,
            policy_context,
            enqueue_sequence: EnqueueSequence::default(),
            per_partition_token_budget: max_batch_tokens.unwrap_or(u32::MAX),
            balance,
            partition_token_usage: Vec::new(),
        }
    }

    pub(crate) fn accept_message(
        &mut self,
        kv_store: &impl PrefixKv,
        msg: WorkerMsgCommon,
        context: &WorkerContext,
    ) {
        let request = match msg {
            WorkerMsgCommon::Request(request) => request,
            // This lifecycle has no recompute path: it accumulates prefill
            // straight into `progress.prefill_tokens_processed` and calls
            // `record_first_token` unconditionally, so re-admitting a request
            // that already emitted tokens would silently reset its progress to
            // one token and re-run its whole decode. Chunked prefill is the
            // lifecycle that models retraction, and migration rides on that.
            WorkerMsgCommon::Resume { req, .. } => unimplemented!(
                "worker `barebone`/`hp_unified` cannot take over migrated {req:?}: \
                 its admission has no reprocessed-prefill path. Use worker \
                 `chunked_prefill` for a pool that migrates."
            ),
        };
        debug_assert!(
            !self.policy.contains(request),
            "local admission is once-per-request; duplicate pending request {request:?}"
        );
        let (fresh_prompt_tokens, remaining_output_tokens, session_input, conversation_start_time) = {
            let mut store = context.requests.borrow_mut();
            let record = &mut store[request];
            let fresh_prompt_tokens = record.request.definition.prompt_tokens;
            let remaining_output_tokens = record.request.definition.target_output_tokens;
            let session_input = record.request.definition.session;
            let arrival_time = record.request.core.arrival_time;
            let conversation_start_time = session_input.session_start_or(arrival_time);
            context.stamp_stage(record, arrival_time, UnifiedStage::Pending as u16);
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

    pub(crate) fn form_batch(
        &mut self,
        kv_store: &mut (impl IterWorkerKv + PrefixKv),
        context: &WorkerContext,
        now: Time,
    ) -> bool {
        let num_partitions = kv_store.num_partitions();
        let had_decode =
            (0..num_partitions as u16).any(|partition| kv_store.has_live_decode(partition));

        self.partition_token_usage.clear();
        self.partition_token_usage
            .extend(
                (0..num_partitions as u16).map(|partition| PartitionTokenUsage {
                    live_decode_tokens: kv_store.live_decode_count(partition),
                    admitted_prefill_tokens: 0,
                }),
            );

        loop {
            // Bring the head back in line with what the prefix cache holds right
            // now. No-op for every enqueue-time-frozen policy; O(1) amortized for
            // a cache-aware one, which reconciles only the head (see
            // `LongestPrefixMatch`). Re-run each pass because admitting can
            // evict, which invalidates the next head.
            self.policy.refresh_head(&mut |candidate| {
                kv_store
                    .resident_prefix_tokens(candidate.fresh_prompt_tokens, candidate.session_input)
            });
            let Some(candidate) = self.policy.peek() else {
                break;
            };
            let partition =
                self.choose_prefill_partition(kv_store, candidate.session_input, num_partitions);
            let partition_index = usize::from(partition);
            let resolved_prefill = kv_store.preview_prefill_context(
                partition,
                candidate.fresh_prompt_tokens,
                candidate.session_input,
            );
            let prefill_tokens_to_compute = resolved_prefill.prefill_tokens_to_compute();
            let partition_usage = self.partition_token_usage[partition_index];
            if !prefill_fits_budget(
                self.per_partition_token_budget,
                partition_usage.live_decode_tokens,
                partition_usage.admitted_prefill_tokens,
                prefill_tokens_to_compute,
            ) {
                break;
            }
            let footprint = kv_store.footprint(
                candidate.request_id,
                resolved_prefill.post_prefill_context_tokens(),
                candidate.remaining_output_tokens,
            );
            if !kv_store.fits(partition, &footprint) {
                break;
            }
            self.pop_selected(candidate);
            self.admit(
                kv_store,
                context,
                candidate,
                partition,
                PlannedReservation {
                    resolved_prefill,
                    footprint,
                },
                now,
            );
            self.partition_token_usage[partition_index].admitted_prefill_tokens = partition_usage
                .admitted_prefill_tokens
                .checked_add(prefill_tokens_to_compute)
                .expect("admitted prefill token count overflow");
        }

        kv_store.drain_ready();
        had_decode
            || (0..num_partitions as u16).any(|partition| kv_store.has_prefill_admit(partition))
    }

    /// Retained session KV is a hard worker-local affinity. Round-robin is
    /// consulted only for a cold or evicted session, so a blocked cache owner
    /// waits instead of creating another copy on a different partition.
    fn choose_prefill_partition<K: PrefixKv>(
        &mut self,
        kv_store: &K,
        session_input: SessionInput,
        num_partitions: usize,
    ) -> u16 {
        kv_store
            .retained_prefix_partition(session_input)
            .unwrap_or_else(|| self.balance.choose(num_partitions) as u16)
    }

    fn admit<K: PrefixKv>(
        &self,
        kv_store: &mut K,
        context: &WorkerContext,
        candidate: AdmissionCandidate,
        partition: u16,
        reservation: PlannedReservation<K::Footprint>,
        now: Time,
    ) {
        kv_store.reserve_prefill_context(
            candidate.request_id,
            partition,
            reservation.resolved_prefill,
            reservation.footprint,
            now,
        );

        let mut store = context.requests.borrow_mut();
        store.mark_admitted(candidate.request_id);
        let record = &mut store[candidate.request_id];
        record
            .record_prefix_cache_hit_tokens(reservation.resolved_prefill.resident_prefix_tokens());
        context.stamp_stage(record, now, UnifiedStage::Prefill as u16);
    }

    fn pop_selected(&mut self, candidate: AdmissionCandidate) {
        let popped = self.policy.pop(&mut self.policy_context);
        debug_assert_eq!(
            popped,
            Some(candidate),
            "policy pop must remove peeked head"
        );
    }

    pub(crate) fn complete_iteration(
        &mut self,
        kv_store: &mut (impl IterWorkerKv + PrefixKv),
        context: &WorkerContext,
        events: &mut Vec<WorkerEventCommon>,
        now: Time,
    ) {
        for partition in 0..kv_store.num_partitions() as u16 {
            let mut completed = Vec::new();

            {
                let mut store = context.requests.borrow_mut();
                kv_store.visit_decode_members(partition, |request, _| {
                    let record = &mut store[request];
                    record.record_token(now, context.log_tokens());
                    if record.is_complete() {
                        context.stamp_stage(record, now, UnifiedStage::Done as u16);
                        completed.push(request);
                    }
                });
            }

            kv_store.advance(AdvanceScope::WholePartition(partition), 1);

            let mut to_finalize = Vec::new();
            {
                let mut store = context.requests.borrow_mut();
                kv_store.visit_prefill_admits(partition, |request| {
                    let record = &mut store[request];
                    record.progress.prefill_tokens_processed =
                        kv_store.prefill_tokens_to_compute(request);
                    record.record_first_token(now, context.log_tokens());
                    if record.is_complete() {
                        context.stamp_stage(record, now, UnifiedStage::Done as u16);
                        completed.push(request);
                    } else {
                        let post_prefill_context_tokens =
                            kv_store.post_prefill_context_tokens(request);
                        let remaining_output_tokens = record
                            .request
                            .definition
                            .target_output_tokens
                            .saturating_sub(record.progress.output_tokens_emitted);
                        context.stamp_stage(record, now, UnifiedStage::Decode as u16);
                        to_finalize.push((
                            request,
                            post_prefill_context_tokens,
                            remaining_output_tokens,
                        ));
                    }
                });
            }
            for (request, post_prefill_context_tokens, remaining_output_tokens) in to_finalize {
                kv_store.commit_resident(
                    request,
                    partition,
                    post_prefill_context_tokens,
                    remaining_output_tokens,
                );
            }
            kv_store.clear_prefill_admits(partition);

            for request in completed {
                kv_store.release_retaining_prefix(request, partition, now);
                events.push(WorkerEventCommon::RequestComplete {
                    worker: context.id,
                    req: request,
                });
            }

            kv_store.sample_submit(partition, now);
        }
    }

    #[inline]
    pub(crate) fn queued_requests(&self) -> u32 {
        self.policy.len() as u32
    }

    pub(crate) fn cancel_pending(&mut self, request: RequestId) -> bool {
        self.policy.remove(request).is_some()
    }
}

impl<P, K> IterAdmission<K> for LocalPrefillDecodeAdmission<P>
where
    P: PendingOrderPolicy,
    K: IterWorkerKv + PrefixKv,
{
    type Msg = WorkerMsgCommon;
    type Event = WorkerEventCommon;

    fn accept_message(&mut self, kv_store: &mut K, msg: Self::Msg, context: &WorkerContext) {
        LocalPrefillDecodeAdmission::accept_message(self, &*kv_store, msg, context);
    }

    fn form_batch(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        batch_plan: &mut IterBatchPlan,
        now: Time,
    ) -> bool {
        batch_plan.reset_decode_participation(kv_store.num_partitions(), true);
        LocalPrefillDecodeAdmission::form_batch(self, kv_store, context, now)
    }

    fn complete_iteration(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        _batch_plan: &IterBatchPlan,
        events: &mut Vec<Self::Event>,
        now: Time,
    ) {
        LocalPrefillDecodeAdmission::complete_iteration(self, kv_store, context, events, now);
    }

    fn queued_requests(&self) -> u32 {
        LocalPrefillDecodeAdmission::queued_requests(self)
    }

    fn cancel_pending(&mut self, request: RequestId) -> bool {
        LocalPrefillDecodeAdmission::cancel_pending(self, request)
    }

    fn drain_pending(&mut self, out: &mut Vec<RequestId>) {
        while let Some(candidate) = self.policy.pop(&mut self.policy_context) {
            out.push(candidate.request_id);
        }
    }
}
