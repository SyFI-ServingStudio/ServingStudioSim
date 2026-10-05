//! Chunked prefill and decode lifecycle for a pipeline-parallel head stage.
//!
//! This follows vLLM V1 under pipeline parallelism. The scheduler advances a
//! request's computed-token count when it schedules a chunk, not when the
//! chunk's outputs return, so the next microbatch can carry the same prompt's
//! following chunk while the previous one is still on a later stage. A token is
//! emitted only when the microbatch holding it leaves the last stage.
//!
//! Decode needs that token as its next input, so a running request has at most
//! one step in flight: vLLM finds no new tokens to schedule for it until the
//! output returns (`num_new_tokens == 0`), or with async scheduling holds it
//! `pp_size` steps (`next_decode_eligible_step`). A microbatch therefore takes
//! the ready decodes, one query row each, then started prompts, then fresh
//! prompts, under one token budget. By default scheduling is greedy, as in
//! vLLM: nothing balances decodes across the in-flight microbatches, so
//! requests that become ready together stay in one microbatch and the other
//! stages idle. `with_balanced_decodes(depth)` caps each microbatch at
//! `ceil(resident decodes / depth)` decodes instead, so the running requests
//! split across the `depth` microbatches the pipeline holds and each returns
//! to the next round on its own.
//!
//! The pending policy owns fresh requests. A prompt that has started and still
//! has unscheduled prefill tokens stays in `started_prefills`, in start order,
//! and keeps priority over fresh prompts. A request whose last chunk is in flight
//! is in neither list; its ticket is in the shell's in-flight queue. It joins KV's
//! decode membership when that chunk exits, and a decode step's request is in
//! `in_flight_decodes` until its microbatch exits.
//!
//! KV reserves the full request footprint once at admission and releases it at
//! completion. One attention partition only.

use std::collections::HashSet;

use crate::common::{RequestId, Time, UnifiedStage};
use crate::worker::kv::ChunkedPrefillKv;
use crate::worker::shared::advance_scope::AdvanceScope;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::IterBatchPlan;

use super::chunked_prefill_admission::next_chunk_tokens;
use super::{AdmissionCandidate, EnqueueSequence, MicrobatchAdmission, PendingOrderPolicy};

/// The only attention partition of a pipeline head.
const PARTITION: u16 = 0;

pub struct PipelinedChunkedPrefillAdmission<P: PendingOrderPolicy> {
    policy: P,
    policy_context: P::Context,
    enqueue_sequence: EnqueueSequence,
    max_batch_tokens: u32,
    /// Prompts that have started and still have prefill tokens to schedule, in
    /// start order.
    started_prefills: Vec<AdmissionCandidate>,
    /// Running requests whose decode step is in a microbatch still in flight.
    in_flight_decodes: HashSet<RequestId>,
    /// The decodes the microbatch being formed carries, until it is committed.
    scheduled_decodes: Vec<RequestId>,
    /// Pipeline depth when decodes are balanced across microbatches; `None`
    /// schedules every ready decode (vLLM).
    balanced_depth: Option<u16>,
}

/// One request's chunk in one microbatch, snapshotted when it was committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillChunk {
    request: RequestId,
    chunk_tokens: u32,
    /// This chunk carried the prompt's last prefill tokens.
    finishes_prefill: bool,
}

/// Every prefill chunk and decode step one microbatch carries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MicrobatchTicket {
    chunks: Vec<PrefillChunk>,
    decodes: Vec<RequestId>,
}

impl MicrobatchTicket {
    /// Prefill tokens the microbatch carries.
    pub fn tokens(&self) -> u64 {
        self.chunks
            .iter()
            .map(|chunk| u64::from(chunk.chunk_tokens))
            .sum()
    }
}

impl<P: PendingOrderPolicy> PipelinedChunkedPrefillAdmission<P> {
    pub(crate) fn new(policy: P, policy_context: P::Context, max_batch_tokens: u32) -> Self {
        assert!(
            max_batch_tokens > 0,
            "pipelined prefill cap must be positive"
        );
        Self {
            policy,
            policy_context,
            enqueue_sequence: EnqueueSequence::default(),
            max_batch_tokens,
            started_prefills: Vec::new(),
            in_flight_decodes: HashSet::new(),
            scheduled_decodes: Vec::new(),
            balanced_depth: None,
        }
    }

    /// Cap each microbatch's decodes at `ceil(resident decodes / depth)`.
    pub(crate) fn with_balanced_decodes(mut self, depth: u16) -> Self {
        assert!(depth > 0, "pipeline depth must be positive");
        self.balanced_depth = Some(depth);
        self
    }

    /// Admit fresh prompts into the budget left after started prompts.
    fn admit_fresh_prompts<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        mut remaining_budget: u32,
        now: Time,
    ) {
        while remaining_budget > 0 {
            self.policy.refresh_head(&mut |candidate| {
                kv_store
                    .resident_prefix_tokens(candidate.fresh_prompt_tokens, candidate.session_input)
            });
            let Some(candidate) = self.policy.peek() else {
                break;
            };
            let resolved_prefill = kv_store.preview_prefill_context(
                PARTITION,
                candidate.fresh_prompt_tokens,
                candidate.session_input,
            );
            let chunk_tokens = next_chunk_tokens(
                resolved_prefill,
                remaining_budget,
                None,
                self.max_batch_tokens,
            );
            if chunk_tokens == 0 {
                break;
            }
            let footprint = kv_store.footprint(
                candidate.request_id,
                resolved_prefill.post_prefill_context_tokens(),
                candidate.remaining_output_tokens,
            );
            if !kv_store.fits(PARTITION, &footprint) {
                break;
            }
            let popped = self.policy.pop(&mut self.policy_context);
            debug_assert_eq!(popped, Some(candidate));
            kv_store.reserve_chunked_prefill_context(
                candidate.request_id,
                PARTITION,
                resolved_prefill,
                footprint,
                now,
            );
            {
                let mut store = context.requests.borrow_mut();
                store.mark_admitted(candidate.request_id);
                let record = &mut store[candidate.request_id];
                record.record_prefix_cache_hit_tokens(resolved_prefill.resident_prefix_tokens());
                context.stamp_stage(record, now, UnifiedStage::Prefill as u16);
            }
            kv_store.schedule_prefill_chunk(candidate.request_id, PARTITION, chunk_tokens);
            remaining_budget -= chunk_tokens;
            if chunk_tokens < resolved_prefill.remaining_prefill_tokens() {
                self.started_prefills.push(candidate);
            }
        }
    }
}

impl<P, K> MicrobatchAdmission<K> for PipelinedChunkedPrefillAdmission<P>
where
    P: PendingOrderPolicy,
    K: ChunkedPrefillKv,
{
    type Ticket = MicrobatchTicket;

    fn accept_request(&mut self, kv_store: &mut K, request: RequestId, context: &WorkerContext) {
        debug_assert_eq!(kv_store.num_partitions(), 1);
        let (fresh_prompt_tokens, target_output_tokens, session_input, conversation_start_time) = {
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
            target_output_tokens,
            session_input,
            conversation_start_time,
            kv_store.resident_prefix_tokens(fresh_prompt_tokens, session_input),
        );
        debug_assert!(!self.policy.contains(request));
        self.policy.push(candidate, &mut self.policy_context);
    }

    fn form_microbatch(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        batch_plan: &mut IterBatchPlan,
        now: Time,
    ) -> bool {
        let max_batch_tokens = self.max_batch_tokens;
        let mut remaining_budget = max_batch_tokens;
        // Running decodes first, as vLLM schedules its running queue before
        // waiting prompts. One whose previous step has not returned sits out.
        let decode_cap = match self.balanced_depth {
            None => usize::MAX,
            Some(depth) => {
                let mut resident_decodes = 0_usize;
                kv_store.visit_decode_members(PARTITION, |_, _| resident_decodes += 1);
                resident_decodes.div_ceil(usize::from(depth))
            }
        };
        let (in_flight_decodes, scheduled_decodes) =
            (&self.in_flight_decodes, &mut self.scheduled_decodes);
        scheduled_decodes.clear();
        kv_store.visit_decode_members(PARTITION, |request, _| {
            if remaining_budget > 0
                && scheduled_decodes.len() < decode_cap
                && !in_flight_decodes.contains(&request)
            {
                scheduled_decodes.push(request);
                remaining_budget -= 1;
            }
        });
        batch_plan.reset_decode_participation(1, false);
        batch_plan.select_partition_decodes(PARTITION, self.scheduled_decodes.clone());
        // Started prompts keep start order. One that cannot run keeps its place.
        self.started_prefills.retain(|candidate| {
            let resolved_prefill = kv_store.resolved_prefill_context(candidate.request_id);
            let chunk_tokens =
                next_chunk_tokens(resolved_prefill, remaining_budget, None, max_batch_tokens);
            if chunk_tokens == 0 {
                return true;
            }
            kv_store.schedule_prefill_chunk(candidate.request_id, PARTITION, chunk_tokens);
            remaining_budget -= chunk_tokens;
            chunk_tokens < resolved_prefill.remaining_prefill_tokens()
        });
        self.admit_fresh_prompts(kv_store, context, remaining_budget, now);
        kv_store.has_prefill_admit(PARTITION) || !self.scheduled_decodes.is_empty()
    }

    fn commit_microbatch(&mut self, kv_store: &mut K, now: Time) -> MicrobatchTicket {
        let mut ticket = MicrobatchTicket {
            decodes: std::mem::take(&mut self.scheduled_decodes),
            ..MicrobatchTicket::default()
        };
        self.in_flight_decodes
            .extend(ticket.decodes.iter().copied());
        kv_store.visit_prefill_admits(PARTITION, |request| {
            ticket.chunks.push(PrefillChunk {
                request,
                chunk_tokens: 0,
                finishes_prefill: false,
            });
        });
        for chunk in &mut ticket.chunks {
            chunk.chunk_tokens = kv_store
                .resolved_prefill_context(chunk.request)
                .active_chunk()
                .1;
            kv_store.complete_prefill_chunk(chunk.request);
            chunk.finishes_prefill = kv_store
                .resolved_prefill_context(chunk.request)
                .remaining_prefill_tokens()
                == 0;
        }
        kv_store.clear_prefill_admits(PARTITION);
        kv_store.sample_submit(PARTITION, now);
        ticket
    }

    fn complete_microbatch(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        ticket: MicrobatchTicket,
        completed: &mut Vec<RequestId>,
        at: Time,
    ) {
        let MicrobatchTicket { chunks, decodes } = ticket;
        let mut finished_decodes = Vec::new();
        {
            let mut store = context.requests.borrow_mut();
            for &request in &decodes {
                let record = &mut store[request];
                record.record_token(at, context.log_tokens());
                if record.is_complete() {
                    context.stamp_stage(record, at, UnifiedStage::Done as u16);
                    finished_decodes.push(request);
                }
            }
        }
        if !decodes.is_empty() {
            kv_store.advance(
                AdvanceScope::RequestSubset {
                    partition: PARTITION,
                    request_ids: &decodes,
                },
                1,
            );
        }
        for request in &decodes {
            self.in_flight_decodes.remove(request);
        }
        for request in finished_decodes {
            kv_store.release_retaining_prefix(request, PARTITION, at);
            completed.push(request);
        }

        for chunk in chunks {
            let remaining_output_tokens = {
                let mut store = context.requests.borrow_mut();
                let record = &mut store[chunk.request];
                record.progress.prefill_tokens_processed = record
                    .progress
                    .prefill_tokens_processed
                    .checked_add(chunk.chunk_tokens)
                    .expect("prefill progress overflows u32");
                if !chunk.finishes_prefill {
                    continue;
                }
                record.record_first_token(at, context.log_tokens());
                let stage = if record.is_complete() {
                    UnifiedStage::Done
                } else {
                    UnifiedStage::Decode
                };
                context.stamp_stage(record, at, stage as u16);
                record
                    .request
                    .definition
                    .target_output_tokens
                    .saturating_sub(record.progress.output_tokens_emitted)
            };
            kv_store.finish_chunked_prefill(chunk.request, PARTITION, remaining_output_tokens);
            if remaining_output_tokens == 0 {
                kv_store.release_retaining_prefix(chunk.request, PARTITION, at);
                completed.push(chunk.request);
            }
        }
        kv_store.sample_submit(PARTITION, at);
    }

    fn queued_requests(&self) -> u32 {
        self.policy.len() as u32
    }
}
