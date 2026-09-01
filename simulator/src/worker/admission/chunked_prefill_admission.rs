//! Hard-capped chunked-prefill lifecycle for the whole-iteration shell.
//!
//! Pending policies own fresh requests. Once a request starts, its complete KV
//! footprint stays reserved and at most one continuation is retained per
//! attention partition; partial requests never rotate through the policy again.
//!
//! Batch composition is independent of KV membership. `Mix` lets resident
//! decode share the remaining chunk budget. `SeparatePrefillPriority` emits a
//! prefill-only iteration whenever a prompt chunk can run, leaving decode
//! resident and unadvanced until a later decode-only iteration. This is the
//! mechanism selected by SGLang when `enable_mixed_chunk` is false; admission
//! capacity estimation and decode retraction are separate policies.

use crate::common::{RequestId, SessionInput, Time, UnifiedStage};
use crate::worker::config::BatchPolicy;
use crate::worker::kv::ChunkedPrefillKv;
use crate::worker::shared::advance_scope::AdvanceScope;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::{IterBatchPlan, WorkerEventCommon, WorkerMsgCommon};

use super::{AdmissionCandidate, EnqueueSequence, IterAdmission, LoadBalance, PendingOrderPolicy};

pub struct ChunkedPrefillAdmission<P: PendingOrderPolicy> {
    partition_policies: Vec<(P, P::Context)>,
    enqueue_sequence: EnqueueSequence,
    max_batch_tokens: u32,
    batch_policy: BatchPolicy,
    balance: LoadBalance,
    active_chunks: Vec<Option<AdmissionCandidate>>,
}

impl<P: PendingOrderPolicy> ChunkedPrefillAdmission<P> {
    pub(crate) fn new(
        partition_policies: Vec<(P, P::Context)>,
        max_batch_tokens: u32,
        batch_policy: BatchPolicy,
        balance: LoadBalance,
    ) -> Self {
        assert!(max_batch_tokens > 0, "chunked prefill cap must be positive");
        assert!(
            !partition_policies.is_empty(),
            "chunked prefill requires at least one partition"
        );
        Self {
            partition_policies,
            enqueue_sequence: EnqueueSequence::default(),
            max_batch_tokens,
            batch_policy,
            balance,
            active_chunks: Vec::new(),
        }
    }

    fn choose_partition<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &K,
        session_input: SessionInput,
    ) -> usize {
        kv_store
            .retained_prefix_partition(session_input)
            .map(usize::from)
            .unwrap_or_else(|| self.balance.choose(self.partition_policies.len()))
    }
}

impl<P, K> IterAdmission<K> for ChunkedPrefillAdmission<P>
where
    P: PendingOrderPolicy,
    K: ChunkedPrefillKv,
{
    type Msg = WorkerMsgCommon;
    type Event = WorkerEventCommon;

    fn accept_message(&mut self, kv_store: &mut K, msg: Self::Msg, context: &WorkerContext) {
        let WorkerMsgCommon::Request(request) = msg;
        debug_assert_eq!(self.partition_policies.len(), kv_store.num_partitions());
        let (fresh_prompt_tokens, remaining_output_tokens, session_input, conversation_start_time) = {
            let mut store = context.requests.borrow_mut();
            let record = &mut store[request];
            let arrival_time = record.request.core.arrival_time;
            context.stamp_stage(record, arrival_time, UnifiedStage::Pending as u16);
            (
                record.request.definition.prompt_tokens,
                record.request.definition.target_output_tokens,
                record.request.definition.session,
                record
                    .request
                    .definition
                    .session
                    .session_start_or(arrival_time),
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
        // Placement is frozen when the request enters the worker. Retained KV
        // has hard affinity; cold requests use the same generic balance policy
        // as ordinary local admission.
        let partition = self.choose_partition(kv_store, session_input);
        let (policy, policy_context) = &mut self.partition_policies[partition];
        debug_assert!(!policy.contains(request));
        policy.push(candidate, policy_context);
    }

    fn form_batch(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        batch_plan: &mut IterBatchPlan,
        now: Time,
    ) -> bool {
        let num_partitions = kv_store.num_partitions();
        debug_assert_eq!(self.partition_policies.len(), num_partitions);
        self.active_chunks.resize(num_partitions, None);
        batch_plan.reset_decode_participation(num_partitions, true);
        let had_decode =
            (0..num_partitions as u16).any(|partition| kv_store.has_live_decode(partition));
        let mixes_prefill_with_decode = self.batch_policy == BatchPolicy::Mix;
        let mut remaining_budgets: Vec<u32> = (0..num_partitions as u16)
            .map(|partition| {
                if mixes_prefill_with_decode {
                    self.max_batch_tokens
                        .saturating_sub(kv_store.live_decode_count(partition))
                } else {
                    self.max_batch_tokens
                }
            })
            .collect();

        // A partial request keeps partition-local priority until its prompt is
        // complete, matching one EngineCore scheduler queue per DP partition.
        for partition_index in 0..num_partitions {
            let Some(candidate) = self.active_chunks[partition_index] else {
                continue;
            };
            let remaining = kv_store
                .resolved_prefill_context(candidate.request_id)
                .remaining_prefill_tokens();
            let chunk_tokens = remaining.min(remaining_budgets[partition_index]);
            if chunk_tokens == 0 {
                continue;
            }
            kv_store.schedule_prefill_chunk(
                candidate.request_id,
                partition_index as u16,
                chunk_tokens,
            );
            remaining_budgets[partition_index] -= chunk_tokens;
            if chunk_tokens == remaining {
                self.active_chunks[partition_index] = None;
            }
        }

        for partition_index in 0..num_partitions {
            let partition = partition_index as u16;
            while self.active_chunks[partition_index].is_none()
                && remaining_budgets[partition_index] > 0
            {
                let (policy, policy_context) = &mut self.partition_policies[partition_index];
                policy.refresh_head(&mut |candidate| {
                    kv_store.resident_prefix_tokens(
                        candidate.fresh_prompt_tokens,
                        candidate.session_input,
                    )
                });
                let Some(candidate) = policy.peek() else {
                    break;
                };
                let resolved_prefill = kv_store.preview_prefill_context(
                    partition,
                    candidate.fresh_prompt_tokens,
                    candidate.session_input,
                );
                let footprint = kv_store.footprint(
                    candidate.request_id,
                    resolved_prefill.post_prefill_context_tokens(),
                    candidate.remaining_output_tokens,
                );
                if !kv_store.fits(partition, &footprint) {
                    break;
                }
                let popped = policy.pop(policy_context);
                debug_assert_eq!(popped, Some(candidate));
                kv_store.reserve_chunked_prefill_context(
                    candidate.request_id,
                    partition,
                    resolved_prefill,
                    footprint,
                    now,
                );
                {
                    let mut store = context.requests.borrow_mut();
                    store.mark_admitted(candidate.request_id);
                    let record = &mut store[candidate.request_id];
                    record
                        .record_prefix_cache_hit_tokens(resolved_prefill.resident_prefix_tokens());
                    context.stamp_stage(record, now, UnifiedStage::Prefill as u16);
                }
                let remaining = resolved_prefill.remaining_prefill_tokens();
                let chunk_tokens = remaining.min(remaining_budgets[partition_index]);
                kv_store.schedule_prefill_chunk(candidate.request_id, partition, chunk_tokens);
                remaining_budgets[partition_index] -= chunk_tokens;
                if chunk_tokens < remaining {
                    self.active_chunks[partition_index] = Some(candidate);
                }
            }
        }

        let mut had_prefill = false;
        for partition in 0..num_partitions as u16 {
            let partition_has_prefill = kv_store.has_prefill_admit(partition);
            had_prefill |= partition_has_prefill;
            if !mixes_prefill_with_decode && partition_has_prefill {
                batch_plan.set_partition_runs_decode(partition, false);
            }
        }

        had_decode || had_prefill
    }

    fn complete_iteration(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        batch_plan: &IterBatchPlan,
        events: &mut Vec<Self::Event>,
        now: Time,
    ) {
        for partition in 0..kv_store.num_partitions() as u16 {
            let mut completed = Vec::new();
            if batch_plan.partition_runs_decode(partition) {
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
            }

            let mut prefills = Vec::new();
            kv_store.visit_prefill_admits(partition, |request| prefills.push(request));
            for request in prefills {
                let chunk_tokens = kv_store.resolved_prefill_context(request).active_chunk().1;
                kv_store.complete_prefill_chunk(request);
                let finished = kv_store
                    .resolved_prefill_context(request)
                    .remaining_prefill_tokens()
                    == 0;
                {
                    let mut store = context.requests.borrow_mut();
                    let record = &mut store[request];
                    record.progress.prefill_tokens_processed = record
                        .progress
                        .prefill_tokens_processed
                        .checked_add(chunk_tokens)
                        .expect("prefill progress overflows u32");
                    if finished {
                        record.record_first_token(now, context.log_tokens());
                        context.stamp_stage(
                            record,
                            now,
                            if record.is_complete() {
                                UnifiedStage::Done
                            } else {
                                UnifiedStage::Decode
                            } as u16,
                        );
                    }
                }
                if finished {
                    let remaining_output_tokens = {
                        let store = context.requests.borrow();
                        let record = &store[request];
                        record
                            .request
                            .definition
                            .target_output_tokens
                            .saturating_sub(record.progress.output_tokens_emitted)
                    };
                    kv_store.finish_chunked_prefill(request, partition, remaining_output_tokens);
                    if context.requests.borrow()[request].is_complete() {
                        completed.push(request);
                    }
                }
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

    fn queued_requests(&self) -> u32 {
        self.partition_policies
            .iter()
            .map(|(policy, _)| policy.len() as u32)
            .sum()
    }

    fn cancel_pending(&mut self, request: RequestId) -> bool {
        for (policy, _) in &mut self.partition_policies {
            if policy.remove(request).is_some() {
                return true;
            }
        }
        for active_chunk in &mut self.active_chunks {
            if active_chunk.is_some_and(|candidate| candidate.request_id == request) {
                *active_chunk = None;
            }
        }
        false
    }
}
