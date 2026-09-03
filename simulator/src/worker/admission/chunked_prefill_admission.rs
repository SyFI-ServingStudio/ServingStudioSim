//! Hard-capped chunked-prefill lifecycle for the whole-iteration shell.
//!
//! Pending policies own fresh and retracted requests. Once a prompt starts, at
//! most one continuation is retained per attention partition; partial requests
//! never rotate through the policy again. KV ownership is policy-defined:
//! historical deployments reserve the complete request footprint, whereas a
//! bounded-future deployment pairs a waiting-request estimate with decode-time
//! physical allocation checks and retraction.
//!
//! Batch composition is independent of KV membership. `Mix` lets resident
//! decode share the remaining chunk budget. `SeparatePrefillPriority` emits a
//! prefill-only iteration whenever a prompt chunk can run, leaving decode
//! resident and unadvanced until a later decode-only iteration. This is the
//! mechanism selected by SGLang when `enable_mixed_chunk` is false; admission
//! capacity estimation and decode retraction are separate policies.

use std::cmp::Reverse;
use std::collections::HashMap;

use crate::common::{RequestId, SessionInput, Time, UnifiedStage};
use crate::worker::config::{
    BatchPolicy, BoundedFutureKvAdmissionConfig, DecodeRetractionPolicy, KvAdmissionConfig,
};
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
    kv_admission: KvAdmissionConfig,
    current_new_token_ratio: f64,
    balance: LoadBalance,
    active_chunks: Vec<Option<AdmissionCandidate>>,
    prefill_episodes: HashMap<RequestId, bool>,
}

impl<P: PendingOrderPolicy> ChunkedPrefillAdmission<P> {
    pub(crate) fn new(
        partition_policies: Vec<(P, P::Context)>,
        max_batch_tokens: u32,
        batch_policy: BatchPolicy,
        kv_admission: KvAdmissionConfig,
        balance: LoadBalance,
    ) -> Self {
        assert!(max_batch_tokens > 0, "chunked prefill cap must be positive");
        assert!(
            !partition_policies.is_empty(),
            "chunked prefill requires at least one partition"
        );
        assert!(
            !matches!(kv_admission, KvAdmissionConfig::BoundedFuture(_))
                || partition_policies.len() == 1,
            "bounded-future KV admission is validated only for one attention partition"
        );
        assert!(
            !matches!(
                kv_admission,
                KvAdmissionConfig::BoundedFuture(config) if config.page_size != 1
            ),
            "bounded-future currently requires page_size=1; larger pages need allocated-length state"
        );
        let current_new_token_ratio = match kv_admission {
            KvAdmissionConfig::FullFootprint => 1.0,
            KvAdmissionConfig::BoundedFuture(config) => config.initial_new_token_ratio,
        };
        Self {
            partition_policies,
            enqueue_sequence: EnqueueSequence::default(),
            max_batch_tokens,
            batch_policy,
            kv_admission,
            current_new_token_ratio,
            balance,
            active_chunks: Vec::new(),
            prefill_episodes: HashMap::new(),
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

    fn bounded_config(&self) -> Option<BoundedFutureKvAdmissionConfig> {
        match self.kv_admission {
            KvAdmissionConfig::FullFootprint => None,
            KvAdmissionConfig::BoundedFuture(config) => Some(config),
        }
    }

    fn decay_new_token_ratio(&mut self, config: BoundedFutureKvAdmissionConfig) {
        let decay = (config.initial_new_token_ratio - config.minimum_new_token_ratio)
            / f64::from(config.new_token_ratio_decay_steps);
        self.current_new_token_ratio =
            (self.current_new_token_ratio - decay).max(config.minimum_new_token_ratio);
    }

    fn update_ratio_after_retraction<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &K,
        context: &WorkerContext,
        config: BoundedFutureKvAdmissionConfig,
    ) {
        let store = context.requests.borrow();
        let mut decoded_tokens = 0u64;
        let mut max_new_tokens = 0u64;
        let mut requests = 0u64;
        for partition in 0..kv_store.num_partitions() as u16 {
            kv_store.visit_decode_states(partition, |request, _, _| {
                let record = &store[request];
                decoded_tokens += u64::from(record.progress.output_tokens_emitted);
                max_new_tokens += u64::from(record.request.definition.target_output_tokens);
                requests += 1;
            });
        }
        let numerator = decoded_tokens
            .checked_add(u64::from(config.retract_decode_steps) * requests)
            .expect("post-retraction ratio numerator overflow");
        self.current_new_token_ratio = (numerator as f64 / (max_new_tokens + 1) as f64).min(1.0);
    }

    fn select_length_retraction<K: ChunkedPrefillKv>(
        &self,
        kv_store: &K,
        context: &WorkerContext,
        partition: u16,
    ) -> Option<RequestId> {
        let mut candidates = Vec::new();
        kv_store.visit_decode_states(partition, |request, _, _| candidates.push(request));
        let store = context.requests.borrow();
        candidates
            .into_iter()
            .enumerate()
            .min_by_key(|(index, request)| {
                let record = &store[*request];
                let input_tokens = record
                    .request
                    .definition
                    .prompt_tokens
                    .checked_add(record.request.definition.session.declared_prefix_tokens())
                    .expect("retraction input length overflow");
                (
                    record.progress.output_tokens_emitted,
                    Reverse(input_tokens),
                    Reverse(*index),
                )
            })
            .map(|(_, request)| request)
    }

    fn requeue_retracted<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &K,
        context: &WorkerContext,
        partition: u16,
        request: RequestId,
        now: Time,
    ) {
        let (
            reprocessed_input_tokens,
            remaining_output_tokens,
            session_input,
            conversation_start_time,
        ) = {
            let mut store = context.requests.borrow_mut();
            let record = &mut store[request];
            record.record_retraction();
            context.stamp_stage(record, now, UnifiedStage::Pending as u16);
            (
                record
                    .request
                    .definition
                    .prompt_tokens
                    .checked_add(record.progress.output_tokens_emitted)
                    .expect("reprocessed input length overflow"),
                record
                    .request
                    .definition
                    .target_output_tokens
                    .saturating_sub(record.progress.output_tokens_emitted),
                record.request.definition.session,
                record
                    .request
                    .definition
                    .session
                    .session_start_or(record.request.core.arrival_time),
            )
        };
        let resident_prefix_tokens =
            kv_store.resident_prefix_tokens(reprocessed_input_tokens, session_input);
        let candidate = self.enqueue_sequence.freeze_retracted(
            request,
            reprocessed_input_tokens,
            remaining_output_tokens,
            session_input,
            conversation_start_time,
            resident_prefix_tokens,
        );
        let (policy, policy_context) = &mut self.partition_policies[partition as usize];
        policy.push(candidate, policy_context);
    }

    fn prepare_decode<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        now: Time,
        config: BoundedFutureKvAdmissionConfig,
    ) -> bool {
        let mut retracted_any = false;
        for partition in 0..kv_store.num_partitions() as u16 {
            while kv_store.has_live_decode(partition)
                && kv_store.prepare_next_decode(partition, config.page_size, now) > 0
            {
                let live_count = kv_store.live_decode_count(partition);
                assert!(
                    live_count > 1,
                    "bounded-future decode cannot fit its last request; abort lifecycle is required"
                );
                let request = match config.retraction_policy {
                    DecodeRetractionPolicy::Length => self
                        .select_length_retraction(kv_store, context, partition)
                        .expect("decode shortfall requires a retraction candidate"),
                };
                kv_store.release(request, partition);
                self.requeue_retracted(kv_store, context, partition, request, now);
                retracted_any = true;
            }
        }
        if retracted_any {
            self.update_ratio_after_retraction(kv_store, context, config);
        } else {
            self.decay_new_token_ratio(config);
        }
        (0..kv_store.num_partitions() as u16).any(|partition| kv_store.has_live_decode(partition))
    }

    #[cfg(test)]
    pub(crate) fn current_new_token_ratio(&self) -> f64 {
        self.current_new_token_ratio
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
        let bounded_config = self.bounded_config();
        let current_new_token_ratio = self.current_new_token_ratio;
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
                let footprint = match bounded_config {
                    None => kv_store.footprint(
                        candidate.request_id,
                        resolved_prefill.post_prefill_context_tokens(),
                        candidate.remaining_output_tokens,
                    ),
                    Some(config) => kv_store.bounded_future_footprint(
                        candidate.request_id,
                        resolved_prefill.post_prefill_context_tokens(),
                        candidate.remaining_output_tokens,
                        config.max_future_tokens,
                        config.page_size,
                    ),
                };
                let fits = match bounded_config {
                    None => kv_store.fits(partition, &footprint),
                    Some(config) => kv_store.fits_bounded_future(
                        partition,
                        &footprint,
                        config.max_future_tokens,
                        current_new_token_ratio,
                    ),
                };
                if !fits {
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
                    if candidate.retracted {
                        record.begin_reprocessed_prefill(resolved_prefill.resident_prefix_tokens());
                    } else {
                        record.record_prefix_cache_hit_tokens(
                            resolved_prefill.resident_prefix_tokens(),
                        );
                    }
                    context.stamp_stage(record, now, UnifiedStage::Prefill as u16);
                }
                assert!(
                    self.prefill_episodes
                        .insert(candidate.request_id, candidate.retracted)
                        .is_none(),
                    "request entered two simultaneous prefill episodes"
                );
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

        let had_decode = if !had_prefill && had_decode {
            match bounded_config {
                None => true,
                Some(config) => self.prepare_decode(kv_store, context, now, config),
            }
        } else {
            had_decode
        };

        let has_batch = had_decode || had_prefill;
        if !has_batch
            && self
                .partition_policies
                .iter()
                .all(|(policy, _)| policy.len() == 0)
            && self.active_chunks.iter().all(Option::is_none)
        {
            // SGLang resets its tracker only after running, chunked, and
            // waiting queues are all empty (`Scheduler::on_idle`). A blocked
            // waiting request is therefore deliberately not a reset boundary.
            if let Some(config) = bounded_config {
                self.current_new_token_ratio = config.initial_new_token_ratio;
            }
        }
        has_batch
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
                let reprocessed = *self
                    .prefill_episodes
                    .get(&request)
                    .expect("scheduled prefill requires episode metadata");
                let chunk_tokens = kv_store.resolved_prefill_context(request).active_chunk().1;
                kv_store.complete_prefill_chunk(request);
                let finished = kv_store
                    .resolved_prefill_context(request)
                    .remaining_prefill_tokens()
                    == 0;
                {
                    let mut store = context.requests.borrow_mut();
                    let record = &mut store[request];
                    if reprocessed {
                        record.record_reprocessed_prefill_tokens(chunk_tokens);
                    } else {
                        record.progress.prefill_tokens_processed = record
                            .progress
                            .prefill_tokens_processed
                            .checked_add(chunk_tokens)
                            .expect("prefill progress overflows u32");
                    }
                    if finished {
                        if reprocessed {
                            record.complete_reprocessed_prefill();
                            record.record_token(now, context.log_tokens());
                        } else {
                            record.record_first_token(now, context.log_tokens());
                        }
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
                    self.prefill_episodes.remove(&request);
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
                self.prefill_episodes.remove(&request);
            }
        }
        false
    }
}
