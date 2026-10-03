//! Prefill-only chunked lifecycle for a pipeline-parallel head stage.
//!
//! This follows vLLM V1 under pipeline parallelism. The scheduler advances a
//! request's computed-token count when it schedules a chunk, not when the
//! chunk's outputs return, so the next microbatch can carry the same prompt's
//! following chunk while the previous one is still on a later stage. A token is
//! emitted, and the request finishes, only when the microbatch holding its last
//! chunk leaves the last stage.
//!
//! The pending policy owns fresh requests. A prompt that has started and still
//! has unscheduled prefill tokens stays in `started_prefills`, in start order,
//! and keeps priority over fresh prompts. A request whose last chunk is in flight
//! is in neither list; its ticket is in the shell's in-flight queue.
//!
//! KV reserves the full request footprint once at admission and releases it at
//! completion. A hybrid recurrent model with prefix caching ends every non-final
//! chunk on a state-checkpoint boundary (vLLM's `_mamba_block_aligned_split`,
//! which does not change under PP), via the same rule as unified chunked
//! prefill. One attention partition only. Decode is out of scope: a request
//! that wants more than one output token fails at enqueue.

use crate::common::{RequestId, Time, UnifiedStage};
use crate::worker::kv::ChunkedPrefillKv;
use crate::worker::shared::context::WorkerContext;

use super::chunked_prefill_admission::next_chunk_tokens;
use super::{AdmissionCandidate, EnqueueSequence, MicrobatchAdmission, PendingOrderPolicy};

/// The only attention partition of a pipeline head.
const PARTITION: u16 = 0;

pub struct PipelinedPrefillAdmission<P: PendingOrderPolicy> {
    policy: P,
    policy_context: P::Context,
    enqueue_sequence: EnqueueSequence,
    max_batch_tokens: u32,
    /// Checkpoint interval a non-final chunk must end on, if any.
    chunk_end_quantum: Option<u32>,
    /// Prompts that have started and still have prefill tokens to schedule, in
    /// start order.
    started_prefills: Vec<AdmissionCandidate>,
}

/// One request's chunk in one microbatch, snapshotted when it was committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillChunk {
    request: RequestId,
    chunk_tokens: u32,
    /// This chunk carried the prompt's last prefill tokens.
    finishes_prefill: bool,
}

/// Every chunk one microbatch carries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefillChunkTicket {
    chunks: Vec<PrefillChunk>,
}

impl PrefillChunkTicket {
    pub fn tokens(&self) -> u64 {
        self.chunks
            .iter()
            .map(|chunk| u64::from(chunk.chunk_tokens))
            .sum()
    }
}

impl<P: PendingOrderPolicy> PipelinedPrefillAdmission<P> {
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
            chunk_end_quantum: None,
            started_prefills: Vec::new(),
        }
    }

    /// End every non-final chunk on a multiple of `quantum` context tokens.
    pub(crate) fn with_chunk_end_quantum(mut self, quantum: u32) -> Self {
        assert!(quantum > 0, "chunk-end quantum must be positive");
        self.chunk_end_quantum = Some(quantum);
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
                self.chunk_end_quantum,
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

impl<P, K> MicrobatchAdmission<K> for PipelinedPrefillAdmission<P>
where
    P: PendingOrderPolicy,
    K: ChunkedPrefillKv,
{
    type Ticket = PrefillChunkTicket;

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
        assert!(
            target_output_tokens <= 1,
            "request {} wants {target_output_tokens} output tokens; the pipeline head \
             models prefill only (output_len <= 1)",
            request.0,
        );
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

    fn form_microbatch(&mut self, kv_store: &mut K, context: &WorkerContext, now: Time) -> bool {
        let max_batch_tokens = self.max_batch_tokens;
        let chunk_end_quantum = self.chunk_end_quantum;
        let mut remaining_budget = max_batch_tokens;
        // Started prompts keep start order. One that cannot run keeps its place.
        self.started_prefills.retain(|candidate| {
            let resolved_prefill = kv_store.resolved_prefill_context(candidate.request_id);
            let chunk_tokens = next_chunk_tokens(
                resolved_prefill,
                remaining_budget,
                chunk_end_quantum,
                max_batch_tokens,
            );
            if chunk_tokens == 0 {
                return true;
            }
            kv_store.schedule_prefill_chunk(candidate.request_id, PARTITION, chunk_tokens);
            remaining_budget -= chunk_tokens;
            chunk_tokens < resolved_prefill.remaining_prefill_tokens()
        });
        self.admit_fresh_prompts(kv_store, context, remaining_budget, now);
        kv_store.has_prefill_admit(PARTITION)
    }

    fn commit_microbatch(&mut self, kv_store: &mut K, now: Time) -> PrefillChunkTicket {
        let mut ticket = PrefillChunkTicket::default();
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
        ticket: PrefillChunkTicket,
        completed: &mut Vec<RequestId>,
        at: Time,
    ) {
        for chunk in ticket.chunks {
            {
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
                debug_assert!(record.is_complete());
                context.stamp_stage(record, at, UnifiedStage::Done as u16);
            }
            kv_store.finish_chunked_prefill(chunk.request, PARTITION, 0);
            kv_store.release_retaining_prefix(chunk.request, PARTITION, at);
            completed.push(chunk.request);
        }
        kv_store.sample_submit(PARTITION, at);
    }

    fn queued_requests(&self) -> u32 {
        self.policy.len() as u32
    }
}
