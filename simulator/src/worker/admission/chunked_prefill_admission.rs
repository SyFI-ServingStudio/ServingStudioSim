//! Hard-capped chunked-prefill lifecycle for the whole-iteration shell.
//!
//! Pending policies own fresh and retracted requests. Once a prompt starts, its
//! continuation stays in a partition-local started list, in start order, and
//! never rotates through the policy again. Without a chunk-end quantum a partial
//! chunk always spends the rest of the budget, so that list holds at most one
//! request. A chunk-end quantum (vLLM's Mamba `align` cache mode) clips chunks
//! to recurrent-state checkpoint boundaries; the leftover budget then admits
//! further prompts and several may be partial at once. KV ownership is policy-defined:
//! historical deployments reserve the complete request footprint, whereas a
//! bounded-future deployment pairs a waiting-request estimate with decode-time
//! physical allocation checks and retraction. Under `Mix` the decode check runs
//! before admission and a retracting step admits nothing (vLLM); under
//! `SeparatePrefillPriority` it runs on decode-only iterations (SGLang).
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
use crate::worker::kv::{ChunkedPrefillKv, ResolvedPrefillContext};
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::{IterBatchPlan, WorkerEventCommon, WorkerMsgCommon};

use super::{
    AdmissionCandidate, DecodeCompletion, EnqueueSequence, IterAdmission, LoadBalance,
    PendingOrderPolicy, SingleTokenDecodeCompletion,
};

pub struct ChunkedPrefillAdmission<P: PendingOrderPolicy, D = SingleTokenDecodeCompletion> {
    partition_policies: Vec<(P, P::Context)>,
    enqueue_sequence: EnqueueSequence,
    max_batch_tokens: u32,
    batch_policy: BatchPolicy,
    kv_admission: KvAdmissionConfig,
    current_new_token_ratio: f64,
    balance: LoadBalance,
    /// Per partition: prompts that have started and still have prefill tokens
    /// left to schedule, in start order.
    started_prefills: Vec<Vec<AdmissionCandidate>>,
    /// Context-token spacing that every non-final chunk must end on, when the
    /// engine can only checkpoint recurrent state at those boundaries.
    chunk_end_quantum: Option<u32>,
    prefill_episodes: HashMap<RequestId, bool>,
    /// Order in which each running request was last admitted from the waiting
    /// queue; FCFS retraction evicts the most recent one.
    admission_order: HashMap<RequestId, u64>,
    next_admission: u64,
    /// How far one resident decode moves per iteration. Ordinary decode retires
    /// one token; speculative decode retires up to the verify width.
    decode_completion: D,
    /// Budget tokens charged to every scheduled request on top of its own rows
    /// (see `SpeculativeUnifiedModel::drafting_slots_per_request`). Zero for
    /// ordinary decode and sequential drafters.
    drafting_slots: u32,
}

impl<P: PendingOrderPolicy> ChunkedPrefillAdmission<P, SingleTokenDecodeCompletion> {
    pub(crate) fn new(
        partition_policies: Vec<(P, P::Context)>,
        max_batch_tokens: u32,
        batch_policy: BatchPolicy,
        kv_admission: KvAdmissionConfig,
        balance: LoadBalance,
    ) -> Self {
        Self::with_decode_completion(
            partition_policies,
            max_batch_tokens,
            batch_policy,
            kv_admission,
            balance,
            SingleTokenDecodeCompletion,
            0,
        )
    }
}

impl<P: PendingOrderPolicy, D> ChunkedPrefillAdmission<P, D> {
    pub(crate) fn with_decode_completion(
        partition_policies: Vec<(P, P::Context)>,
        max_batch_tokens: u32,
        batch_policy: BatchPolicy,
        kv_admission: KvAdmissionConfig,
        balance: LoadBalance,
        decode_completion: D,
        drafting_slots: u32,
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
            started_prefills: Vec::new(),
            chunk_end_quantum: None,
            prefill_episodes: HashMap::new(),
            admission_order: HashMap::new(),
            next_admission: 0,
            decode_completion,
            drafting_slots,
        }
    }

    /// End every non-final chunk on a multiple of `quantum` context tokens.
    pub(crate) fn with_chunk_end_quantum(mut self, quantum: u32) -> Self {
        assert!(quantum > 0, "chunk-end quantum must be positive");
        self.chunk_end_quantum = Some(quantum);
        self
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

    /// vLLM's FCFS preemption: the request that most recently entered the
    /// running set goes first.
    fn select_fcfs_retraction<K: ChunkedPrefillKv>(
        &self,
        kv_store: &K,
        partition: u16,
    ) -> Option<RequestId> {
        let mut victim = None;
        kv_store.visit_decode_states(partition, |request, _, _| {
            let order = self.admission_order[&request];
            if victim.is_none_or(|(best, _)| order > best) {
                victim = Some((order, request));
            }
        });
        victim.map(|(_, request)| request)
    }

    fn requeue_retracted<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &K,
        context: &WorkerContext,
        partition: u16,
        request: RequestId,
        retraction_policy: DecodeRetractionPolicy,
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
        self.admission_order.remove(&request);
        let (policy, policy_context) = &mut self.partition_policies[partition as usize];
        match retraction_policy {
            DecodeRetractionPolicy::Length => policy.push(candidate, policy_context),
            DecodeRetractionPolicy::Fcfs => policy.push_front(candidate, policy_context),
        }
    }

    /// Retract decodes until every partition fits its next decode step, each
    /// request advancing up to `step_tokens`. Returns whether it retracted.
    fn prepare_decode<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        now: Time,
        config: BoundedFutureKvAdmissionConfig,
        step_tokens: u32,
    ) -> bool {
        let mut retracted_any = false;
        for partition in 0..kv_store.num_partitions() as u16 {
            while kv_store.has_live_decode(partition)
                && kv_store.prepare_next_decode(partition, config.page_size, step_tokens, now) > 0
            {
                let live_count = kv_store.live_decode_count(partition);
                assert!(
                    live_count > 1,
                    "bounded-future decode cannot fit its last request; abort lifecycle is required"
                );
                let request = match config.retraction_policy {
                    DecodeRetractionPolicy::Length => {
                        self.select_length_retraction(kv_store, context, partition)
                    }
                    DecodeRetractionPolicy::Fcfs => {
                        self.select_fcfs_retraction(kv_store, partition)
                    }
                }
                .expect("decode shortfall requires a retraction candidate");
                kv_store.release(request, partition);
                self.requeue_retracted(
                    kv_store,
                    context,
                    partition,
                    request,
                    config.retraction_policy,
                    now,
                );
                retracted_any = true;
            }
        }
        if retracted_any {
            self.update_ratio_after_retraction(kv_store, context, config);
        } else {
            self.decay_new_token_ratio(config);
        }
        retracted_any
    }

    #[cfg(test)]
    pub(crate) fn current_new_token_ratio(&self) -> f64 {
        self.current_new_token_ratio
    }
}

impl<P, D, K> IterAdmission<K> for ChunkedPrefillAdmission<P, D>
where
    P: PendingOrderPolicy,
    D: DecodeCompletion<K>,
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
        self.started_prefills.resize_with(num_partitions, Vec::new);
        batch_plan.reset_decode_participation(num_partitions, true);
        let has_live_decode = |kv_store: &K| {
            (0..num_partitions as u16).any(|partition| kv_store.has_live_decode(partition))
        };
        let mixes_prefill_with_decode = self.batch_policy == BatchPolicy::Mix;
        let bounded_config = self.bounded_config();
        let query_width = self.decode_completion.query_tokens_per_request();
        // A mixed engine (vLLM) schedules its running decodes first, each
        // claiming KV for its next step, and admits no waiting request in a step
        // that had to preempt. A separate-prefill engine (SGLang) checks decode
        // headroom only on the decode-only steps it runs, below.
        let retracted_before_admission = match bounded_config {
            Some(config) if mixes_prefill_with_decode && has_live_decode(kv_store) => {
                self.prepare_decode(kv_store, context, now, config, query_width)
            }
            _ => false,
        };
        let had_decode = has_live_decode(kv_store);
        let current_new_token_ratio = self.current_new_token_ratio;
        // Every scheduled request, decode or prefill chunk, also spends the
        // drafter's slots.
        let drafting_slots = self.drafting_slots;
        let chunk_end_quantum = self.chunk_end_quantum;
        let max_batch_tokens = self.max_batch_tokens;
        let decode_budget = query_width + drafting_slots;
        // One short prompt can create a whole verify window next iteration.
        // Reserve those future query slots before admitting its prefill; an
        // unfinished started prompt keeps its reservation across iterations.
        let mut future_decode_slots = (query_width > 1).then(|| {
            (0..num_partitions)
                .map(|partition| {
                    let started = self.started_prefills[partition]
                        .iter()
                        .filter(|candidate| candidate.remaining_output_tokens > 1)
                        .count() as u32;
                    (self.max_batch_tokens / decode_budget)
                        .saturating_sub(kv_store.live_decode_count(partition as u16))
                        .saturating_sub(started)
                })
                .collect::<Vec<_>>()
        });
        let mut remaining_budgets: Vec<u32> = (0..num_partitions as u16)
            .map(|partition| {
                if mixes_prefill_with_decode {
                    // Decode spends the shared cap in query rows, not requests:
                    // a speculating engine submits a whole verify window per
                    // resident request, so a mixed iteration must leave room
                    // for all of it.
                    self.max_batch_tokens.saturating_sub(
                        kv_store
                            .live_decode_count(partition)
                            .saturating_mul(decode_budget),
                    )
                } else {
                    self.max_batch_tokens
                }
            })
            .collect();

        // Started prompts keep partition-local priority, in start order, until
        // their prompt is complete, matching one EngineCore scheduler queue per
        // DP partition. One that cannot run this iteration keeps its place.
        for partition_index in 0..num_partitions {
            let mut started_prefills = std::mem::take(&mut self.started_prefills[partition_index]);
            started_prefills.retain(|candidate| {
                let resolved_prefill = kv_store.resolved_prefill_context(candidate.request_id);
                let chunk_tokens = next_chunk_tokens(
                    resolved_prefill,
                    remaining_budgets[partition_index].saturating_sub(drafting_slots),
                    chunk_end_quantum,
                    max_batch_tokens,
                );
                if chunk_tokens == 0 {
                    return true;
                }
                kv_store.schedule_prefill_chunk(
                    candidate.request_id,
                    partition_index as u16,
                    chunk_tokens,
                );
                remaining_budgets[partition_index] -= chunk_tokens + drafting_slots;
                chunk_tokens < resolved_prefill.remaining_prefill_tokens()
            });
            self.started_prefills[partition_index] = started_prefills;
        }

        for partition_index in 0..num_partitions {
            let partition = partition_index as u16;
            while !retracted_before_admission && remaining_budgets[partition_index] > drafting_slots
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
                if candidate.remaining_output_tokens > 1
                    && future_decode_slots
                        .as_ref()
                        .is_some_and(|slots| slots[partition_index] == 0)
                {
                    break;
                }
                let resolved_prefill = kv_store.preview_prefill_context(
                    partition,
                    candidate.fresh_prompt_tokens,
                    candidate.session_input,
                );
                // A fresh prompt whose first chunk cannot end on a checkpoint
                // boundary within the budget stops admission (vLLM breaks its
                // waiting loop here rather than skipping ahead).
                let chunk_tokens = next_chunk_tokens(
                    resolved_prefill,
                    remaining_budgets[partition_index] - drafting_slots,
                    chunk_end_quantum,
                    max_batch_tokens,
                );
                if chunk_tokens == 0 {
                    break;
                }
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
                        if mixes_prefill_with_decode {
                            query_width
                        } else {
                            0
                        },
                    ),
                };
                if !fits {
                    break;
                }
                let popped = policy.pop(policy_context);
                debug_assert_eq!(popped, Some(candidate));
                self.admission_order
                    .insert(candidate.request_id, self.next_admission);
                self.next_admission += 1;
                if candidate.remaining_output_tokens > 1 {
                    if let Some(slots) = future_decode_slots.as_mut() {
                        slots[partition_index] -= 1;
                    }
                }
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
                kv_store.schedule_prefill_chunk(candidate.request_id, partition, chunk_tokens);
                remaining_budgets[partition_index] -= chunk_tokens + drafting_slots;
                if chunk_tokens < resolved_prefill.remaining_prefill_tokens() {
                    self.started_prefills[partition_index].push(candidate);
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

        let had_decode = match bounded_config {
            Some(config) if !mixes_prefill_with_decode && !had_prefill && had_decode => {
                self.prepare_decode(kv_store, context, now, config, query_width);
                has_live_decode(kv_store)
            }
            _ => had_decode,
        };

        let has_batch = had_decode || had_prefill;
        if !has_batch
            && self
                .partition_policies
                .iter()
                .all(|(policy, _)| policy.len() == 0)
            && self.started_prefills.iter().all(Vec::is_empty)
        {
            // SGLang resets its tracker only after running, chunked, and
            // waiting queues are all empty (`Scheduler::on_idle`). A blocked
            // waiting request is therefore deliberately not a reset boundary.
            if let Some(config) = bounded_config {
                self.current_new_token_ratio = config.initial_new_token_ratio;
            }
        }
        if has_batch && query_width > 1 {
            let mut store = context.requests.borrow_mut();
            for partition in 0..kv_store.num_partitions() as u16 {
                kv_store.visit_prefill_admits(partition, |request| {
                    let observation = store[request]
                        .telemetry
                        .speculative
                        .get_or_insert_with(Default::default);
                    observation.query_width = query_width;
                    observation.pending_prefill =
                        Some(kv_store.resolved_prefill_context(request).active_chunk());
                });
                if batch_plan.partition_runs_decode(partition) {
                    kv_store.visit_decode_members(partition, |request, resident| {
                        let observation = store[request]
                            .telemetry
                            .speculative
                            .get_or_insert_with(Default::default);
                        observation.query_width = query_width;
                        observation.pending_decode = Some(resident as u64);
                    });
                }
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
                self.decode_completion.complete_decodes(
                    kv_store,
                    partition,
                    context,
                    &mut completed,
                    now,
                );
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
                    let query_width = self.decode_completion.query_tokens_per_request();
                    if query_width > 1 {
                        let observation = record
                            .telemetry
                            .speculative
                            .get_or_insert_with(Default::default);
                        observation.query_width = query_width;
                        observation.prefill_chunks += 1;
                        observation.pending_prefill = None;
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
                self.admission_order.remove(&request);
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
        for started_prefills in &mut self.started_prefills {
            let started_count = started_prefills.len();
            started_prefills.retain(|candidate| candidate.request_id != request);
            if started_prefills.len() != started_count {
                self.prefill_episodes.remove(&request);
            }
        }
        false
    }
}

/// The next chunk of `resolved_prefill` that `budget` tokens can carry, ending
/// on a `chunk_end_quantum` boundary when one is set. `0` means the request
/// cannot run this iteration.
fn next_chunk_tokens(
    resolved_prefill: ResolvedPrefillContext,
    budget: u32,
    chunk_end_quantum: Option<u32>,
    max_chunk_tokens: u32,
) -> u32 {
    let remaining = resolved_prefill.remaining_prefill_tokens();
    let chunk_tokens = remaining.min(budget);
    match chunk_end_quantum {
        None => chunk_tokens,
        Some(quantum) => {
            let start = resolved_prefill.active_chunk().0;
            checkpoint_aligned_chunk_tokens(
                start,
                start + remaining,
                chunk_tokens,
                quantum,
                max_chunk_tokens,
            )
        }
    }
}

/// vLLM's `Scheduler._mamba_block_aligned_split` for a prompt whose next
/// unprocessed context position is `start` and whose prefill ends at
/// `prefill_end`: clip a `chunk_tokens`-long chunk so the recurrent state it
/// leaves behind sits on a `quantum` boundary and can be checkpointed.
///
/// - A non-final chunk ends on the last boundary it reaches. When `quantum`
///   exceeds the chunk cap no boundary may fit, so the chunk runs sub-block
///   instead and re-aligns at the next boundary.
/// - A chunk starting mid-block stops at the next boundary.
/// - No chunk runs past the prompt's last boundary, so its state is cached
///   before the unaligned tail.
///
/// The fork's partial-tail hash stop, shared-prefix junction stop, and in-chunk
/// prefill checkpoints are not modeled.
fn checkpoint_aligned_chunk_tokens(
    start: u32,
    prefill_end: u32,
    chunk_tokens: u32,
    quantum: u32,
    max_chunk_tokens: u32,
) -> u32 {
    if chunk_tokens == 0 || start >= prefill_end {
        return chunk_tokens;
    }
    let mut end = start + chunk_tokens;
    if end < prefill_end {
        let aligned_end = end / quantum * quantum;
        if aligned_end > start || quantum <= max_chunk_tokens {
            end = aligned_end;
        }
    }
    let next_boundary = (start / quantum + 1) * quantum;
    let last_boundary = prefill_end / quantum * quantum;
    let stops = [
        (start % quantum != 0).then_some(next_boundary),
        Some(last_boundary),
    ];
    let end = stops
        .into_iter()
        .flatten()
        .filter(|&stop| start < stop && stop < end)
        .min()
        .unwrap_or(end);
    end.saturating_sub(start)
}

#[cfg(test)]
mod tests {
    use super::checkpoint_aligned_chunk_tokens;

    const BLOCK: u32 = 2_176;

    #[test]
    fn a_non_final_chunk_ends_on_the_last_checkpoint_boundary_it_reaches() {
        // GLM-5.3-Flash at an 8192-token cap: 3 x 2176, as the trace shows.
        assert_eq!(
            checkpoint_aligned_chunk_tokens(0, 40_000, 8_192, BLOCK, 8_192),
            6_528
        );
        assert_eq!(
            checkpoint_aligned_chunk_tokens(6_528, 40_000, 8_192, BLOCK, 8_192),
            6_528
        );
        // Budget shared with decode still floors to a boundary.
        assert_eq!(
            checkpoint_aligned_chunk_tokens(0, 40_000, 3_837, BLOCK, 8_192),
            2_176
        );
    }

    #[test]
    fn a_budget_below_one_block_cannot_start_an_aligned_chunk() {
        assert_eq!(
            checkpoint_aligned_chunk_tokens(0, 40_000, 1_664, BLOCK, 8_192),
            0
        );
    }

    #[test]
    fn no_chunk_runs_past_the_prompt_last_boundary() {
        // The last 4532 tokens cross 43520 = 20 x 2176; the tail runs separately.
        let prefill_end = 43_700;
        assert_eq!(
            checkpoint_aligned_chunk_tokens(39_168, prefill_end, 4_532, BLOCK, 8_192),
            4_352
        );
        assert_eq!(
            checkpoint_aligned_chunk_tokens(43_520, prefill_end, 180, BLOCK, 8_192),
            180
        );
    }

    #[test]
    fn a_final_chunk_that_crosses_no_boundary_is_not_clipped() {
        assert_eq!(
            checkpoint_aligned_chunk_tokens(0, 1_801, 1_801, BLOCK, 8_192),
            1_801
        );
    }

    #[test]
    fn a_block_wider_than_the_cap_runs_sub_block_then_realigns() {
        // GLM-5.3-Flash at a 2048-token cap: 0 -> 2034 -> 2176, as the trace shows.
        assert_eq!(
            checkpoint_aligned_chunk_tokens(0, 5_000, 2_034, BLOCK, 2_048),
            2_034
        );
        assert_eq!(
            checkpoint_aligned_chunk_tokens(2_034, 5_000, 2_048, BLOCK, 2_048),
            142
        );
        assert_eq!(
            checkpoint_aligned_chunk_tokens(2_176, 5_000, 2_048, BLOCK, 2_048),
            2_048
        );
    }
}
