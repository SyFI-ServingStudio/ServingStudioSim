//! Local prefill→decode lifecycle for the whole-iteration worker family.
//!
//! The policy owns pending membership and selection order; this lifecycle owns
//! token/KV gates and prefill/decode/done transitions. Candidate facts are frozen
//! once at enqueue, so batch formation never re-reads the request store.

use crate::common::{PrefixInput, RequestId, Time, UnifiedStage};
use crate::worker::admission::{
    prefill_fits_budget, AdmissionCandidate, EnqueueSequence, IterAdmission, LoadBalance,
    PendingOrderPolicy,
};
use crate::worker::kv::{IterWorkerKv, PrefixKv, PrefixResolution};
use crate::worker::shared::advance_scope::AdvanceScope;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::{WorkerEventCommon, WorkerMsgCommon};

/// Prefix resolution and its matching full-context KV footprint must be
/// reserved together; keeping them paired prevents admission from mixing facts
/// produced by different cache snapshots.
struct PlannedReservation<Footprint> {
    resolution: PrefixResolution,
    footprint: Footprint,
}

pub struct LocalPrefillDecodeAdmission<P: PendingOrderPolicy> {
    policy: P,
    policy_context: P::Context,
    enqueue_sequence: EnqueueSequence,
    max_batch_tokens: Option<u32>,
    balance: LoadBalance,
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
            max_batch_tokens,
            balance,
        }
    }

    pub(crate) fn accept_message(&mut self, msg: WorkerMsgCommon, context: &WorkerContext) {
        let WorkerMsgCommon::Request(request) = msg;
        debug_assert!(
            !self.policy.contains(request),
            "local admission is once-per-request; duplicate pending request {request:?}"
        );
        let (prompt, decode, prefix) = {
            let mut store = context.requests.borrow_mut();
            let record = &mut store[request];
            let prompt = record.request.definition.prompt_tokens;
            let decode = record.request.definition.target_output_tokens;
            let prefix = record.request.definition.prefix;
            let arrival_time = record.request.core.arrival_time;
            context.stamp_stage(record, arrival_time, UnifiedStage::Pending as u16);
            (prompt, decode, prefix)
        };
        let candidate = self
            .enqueue_sequence
            .freeze(request, prompt, decode, prefix);
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

        match self.max_batch_tokens {
            None => {
                if let Some(candidate) = self.policy.peek() {
                    let partition =
                        self.choose_prefill_partition(kv_store, candidate.prefix, num_partitions);
                    let resolution =
                        kv_store.plan_prefix(partition, candidate.prompt, candidate.prefix);
                    let footprint = kv_store.footprint(
                        candidate.request,
                        resolution.initial_context_tokens(),
                        candidate.decode,
                    );
                    if kv_store.fits(partition, &footprint) {
                        self.pop_selected(candidate);
                        self.admit(
                            kv_store,
                            context,
                            candidate,
                            partition,
                            PlannedReservation {
                                resolution,
                                footprint,
                            },
                            now,
                        );
                    }
                }
            }
            Some(budget) => {
                if num_partitions == 1 {
                    self.fill_single_partition_budget(kv_store, context, now, budget);
                } else {
                    self.fill_partitioned_budget(kv_store, context, now, budget, num_partitions);
                }
            }
        }

        kv_store.drain_ready();
        had_decode
            || (0..num_partitions as u16).any(|partition| kv_store.has_prefill_admit(partition))
    }

    /// Preserve the old barebone worker's scalar, allocation-free budget path.
    fn fill_single_partition_budget<K: IterWorkerKv + PrefixKv>(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        now: Time,
        budget: u32,
    ) {
        let partition = 0;
        let decode_tokens = kv_store.live_decode_count(partition);
        let mut admitted_tokens = 0;
        while let Some(candidate) = self.policy.peek() {
            let resolution = kv_store.plan_prefix(partition, candidate.prompt, candidate.prefix);
            let prefill_compute_tokens = resolution.prefill_compute_tokens();
            if !prefill_fits_budget(
                budget,
                decode_tokens,
                admitted_tokens,
                prefill_compute_tokens,
            ) {
                break;
            }
            let footprint = kv_store.footprint(
                candidate.request,
                resolution.initial_context_tokens(),
                candidate.decode,
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
                    resolution,
                    footprint,
                },
                now,
            );
            admitted_tokens += prefill_compute_tokens;
        }
    }

    /// Preserve HP's per-partition token budgets and advancing round-robin
    /// cursor, including cursor movement when the selected head is blocked.
    fn fill_partitioned_budget<K: IterWorkerKv + PrefixKv>(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        now: Time,
        budget: u32,
        num_partitions: usize,
    ) {
        let decode_tokens: Vec<u32> = (0..num_partitions as u16)
            .map(|partition| kv_store.live_decode_count(partition))
            .collect();
        let mut admitted_tokens = vec![0; num_partitions];

        while let Some(candidate) = self.policy.peek() {
            let partition =
                self.choose_prefill_partition(kv_store, candidate.prefix, num_partitions);
            let partition_index = partition as usize;
            let resolution = kv_store.plan_prefix(partition, candidate.prompt, candidate.prefix);
            let prefill_compute_tokens = resolution.prefill_compute_tokens();
            if !prefill_fits_budget(
                budget,
                decode_tokens[partition_index],
                admitted_tokens[partition_index],
                prefill_compute_tokens,
            ) {
                break;
            }
            let footprint = kv_store.footprint(
                candidate.request,
                resolution.initial_context_tokens(),
                candidate.decode,
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
                    resolution,
                    footprint,
                },
                now,
            );
            admitted_tokens[partition_index] += prefill_compute_tokens;
        }
    }

    /// Retained session KV is a hard worker-local affinity. Round-robin is
    /// consulted only for a cold or evicted session, so a blocked cache owner
    /// waits instead of creating another copy on a different partition.
    fn choose_prefill_partition<K: PrefixKv>(
        &mut self,
        kv_store: &K,
        prefix: PrefixInput,
        num_partitions: usize,
    ) -> u16 {
        kv_store
            .retained_prefix_partition(prefix)
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
        kv_store.reserve_prefix(
            candidate.request,
            partition,
            reservation.resolution,
            reservation.footprint,
        );

        let mut store = context.requests.borrow_mut();
        store.mark_admitted(candidate.request);
        let record = &mut store[candidate.request];
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
                        kv_store.prefill_compute_tokens(request);
                    record.record_first_token(now, context.log_tokens());
                    if record.is_complete() {
                        context.stamp_stage(record, now, UnifiedStage::Done as u16);
                        completed.push(request);
                    } else {
                        let initial_kv = kv_store.initial_context_tokens(request);
                        let remaining = record
                            .request
                            .definition
                            .target_output_tokens
                            .saturating_sub(record.progress.output_tokens_emitted);
                        context.stamp_stage(record, now, UnifiedStage::Decode as u16);
                        to_finalize.push((request, initial_kv, remaining));
                    }
                });
            }
            for (request, initial_kv, remaining) in to_finalize {
                kv_store.commit_resident(request, partition, initial_kv, remaining);
            }
            kv_store.clear_prefill_admits(partition);

            for request in completed {
                kv_store.release_retaining_prefix(request, partition);
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

    fn accept_message(&mut self, _kv_store: &mut K, msg: Self::Msg, context: &WorkerContext) {
        LocalPrefillDecodeAdmission::accept_message(self, msg, context);
    }

    fn form_batch(&mut self, kv_store: &mut K, context: &WorkerContext, now: Time) -> bool {
        LocalPrefillDecodeAdmission::form_batch(self, kv_store, context, now)
    }

    fn complete_iteration(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
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
}
