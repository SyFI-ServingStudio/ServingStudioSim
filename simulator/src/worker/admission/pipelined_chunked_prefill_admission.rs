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
//! Prefill takes as much pending work as the budget allows by default (vLLM).
//! `with_microbatch_budget` can size it instead (`microbatch_budget.rs`): an
//! even split over the pipeline depth, a budget that follows the backlog, and a
//! per-request cap. Their round is the prefill the pipeline holds: started
//! prompts' unscheduled tokens, queued prompts' tokens (before any prefix hit,
//! whether or not KV admits them yet), and the prefill tokens of microbatches
//! formed and not yet exited. Counting the in-flight tokens keeps an even size
//! steady through a round: one `P`-token prompt into an empty pipeline runs as
//! `depth` slices of `P / depth`, not a shrinking `P/depth, 3P/depth^2, ...`
//! series.
//!
//! The pending policy owns fresh requests, behind a force-schedule `Deadline`
//! that is a pass-through unless a bound is set. A prompt that has started and
//! still has unscheduled prefill tokens stays in `started_prefills`, in start
//! order, and keeps priority over fresh prompts unless the started order is
//! shortest-remaining (`started_order.rs`). A request whose last chunk is in flight
//! is in neither list; its ticket is in the shell's in-flight queue. It joins KV's
//! decode membership when that chunk exits, and a decode step's request is in
//! `in_flight_decodes` until its microbatch exits.
//!
//! With host tiers behind HBM (`TieredPrefixKv`), a fresh request goes pending
//! -> reading -> landed -> started, as under vLLM's KV connector: when
//! admission reaches a request whose context a tier holds beyond HBM, its read
//! starts if the whole request fits in HBM, holding that footprint, and the
//! request leaves its queue (`WAITING_FOR_REMOTE_KVS`) while other requests go
//! ahead; one that does not fit stops admission. Once the read lands the
//! request is offered before every queue, in landing order, and its context
//! goes back into HBM just before admission resolves its prefix; one evicted
//! again before admission is read again. A `ReadWait` bound passes over or
//! stops at a request whose tier would begin its read too late
//! (`read_wait.rs`).
//!
//! KV reserves the full request footprint once at admission and releases it at
//! completion. A hybrid recurrent model with prefix caching ends every non-final
//! chunk on a state-checkpoint boundary (vLLM's `_mamba_block_aligned_split`,
//! which does not change under PP), via the same rule as unified chunked
//! prefill. One attention partition only.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::common::{RequestId, Time, UnifiedStage};
use crate::worker::kv::{ChunkedPrefillKv, TieredPrefixKv};
use crate::worker::shared::advance_scope::AdvanceScope;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::IterBatchPlan;

use super::chunked_prefill_admission::ChunkEndAlignment;
use super::microbatch_budget::MicrobatchBudget;
use super::read_wait::{ReadWait, ReadWaitPolicy};
use super::started_order::{FreshGate, PrefillFill, StartedOrder};
use super::{
    AdmissionCandidate, Deadline, EnqueueSequence, MicrobatchAdmission, PendingOrderPolicy,
};

/// The only attention partition of a pipeline head.
const PARTITION: u16 = 0;

pub struct PipelinedChunkedPrefillAdmission<P: PendingOrderPolicy> {
    policy: Deadline<P>,
    policy_context: P::Context,
    enqueue_sequence: EnqueueSequence,
    max_batch_tokens: u32,
    /// How much prefill each microbatch takes.
    budget: MicrobatchBudget,
    /// Microbatches the pipeline holds, which an even split divides the round by.
    depth: u16,
    /// Order of started prompts against queued ones.
    started_order: StartedOrder,
    /// Where chunks may end, when the engine can only checkpoint recurrent
    /// state on block boundaries.
    chunk_end_alignment: Option<ChunkEndAlignment>,
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
    /// Prefill tokens of the committed microbatches that have not exited.
    in_flight_prefill_tokens: u64,
    /// Fresh requests out of their queue while their host-tier read runs.
    reading: HashMap<RequestId, AdmissionCandidate>,
    /// Requests whose read has landed, in landing order, offered before every
    /// queue.
    landed: VecDeque<Landed>,
    /// When a read may start; `None`: whenever it fits (vLLM).
    read_wait: Option<ReadWait>,
}

/// A request whose read has landed.
#[derive(Clone, Copy, Debug)]
struct Landed {
    candidate: AdmissionCandidate,
    /// Its context is back in HBM.
    restored: bool,
}

/// Where a fresh-admission pass found its candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FreshSource {
    Landed,
    Overdue,
    Pending,
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
            policy: Deadline::new(policy),
            policy_context,
            enqueue_sequence: EnqueueSequence::default(),
            max_batch_tokens,
            budget: MicrobatchBudget::default(),
            depth: 1,
            started_order: StartedOrder::default(),
            chunk_end_alignment: None,
            started_prefills: Vec::new(),
            in_flight_decodes: HashSet::new(),
            scheduled_decodes: Vec::new(),
            balanced_depth: None,
            in_flight_prefill_tokens: 0,
            reading: HashMap::new(),
            landed: VecDeque::new(),
            read_wait: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn reading_requests(&self) -> usize {
        self.reading.len()
    }

    /// Start a host-tier read only within `read_wait` of its tier's channel.
    /// Not vLLM.
    pub(crate) fn with_read_wait(mut self, read_wait: ReadWait) -> Self {
        assert!(
            read_wait.max > Time::ZERO,
            "a read-wait bound must be positive"
        );
        self.read_wait = Some(read_wait);
        self
    }

    /// The next fresh candidate: a landed read, an overdue request, then the
    /// pending order's head re-ranked by what HBM or a tier now holds. Under
    /// `OverdueOnly` only overdue requests.
    fn peek_fresh<K: TieredPrefixKv>(
        &mut self,
        kv_store: &K,
        gate: FreshGate,
    ) -> Option<(FreshSource, AdmissionCandidate)> {
        if gate != FreshGate::OverdueOnly {
            if let Some(landed) = self.landed.front() {
                return Some((FreshSource::Landed, landed.candidate));
            }
        }
        if self.policy.head_is_overdue() {
            return self
                .policy
                .peek()
                .map(|candidate| (FreshSource::Overdue, candidate));
        }
        if gate == FreshGate::OverdueOnly {
            return None;
        }
        self.policy.refresh_head(&mut |candidate| {
            kv_store.reusable_prefix_tokens(candidate.fresh_prompt_tokens, candidate.session_input)
        });
        self.policy
            .peek()
            .map(|candidate| (FreshSource::Pending, candidate))
    }

    /// Take the candidate [`Self::peek_fresh`] returned out of its source.
    fn take_fresh(&mut self, source: FreshSource, candidate: AdmissionCandidate) {
        match source {
            FreshSource::Landed => {
                let landed = self.landed.pop_front();
                debug_assert_eq!(landed.map(|landed| landed.candidate), Some(candidate));
            }
            FreshSource::Overdue | FreshSource::Pending => {
                let popped = self.policy.pop(&mut self.policy_context);
                debug_assert_eq!(popped, Some(candidate));
            }
        }
    }

    /// Put the candidates passed over this pass back at the front of the
    /// queues they came from, in their order. A landed one, whose context HBM
    /// evicted again, goes back to the pending order.
    fn return_passed_over(&mut self, passed_over: Vec<(FreshSource, AdmissionCandidate)>) {
        for (source, candidate) in passed_over.into_iter().rev() {
            match source {
                FreshSource::Overdue => self.policy.push_front_overdue(candidate),
                FreshSource::Landed | FreshSource::Pending => {
                    self.policy.push_front(candidate, &mut self.policy_context);
                }
            }
        }
    }

    /// Move the reads landed by `now` from reading to landed; returns the
    /// latest landing.
    fn land<K: TieredPrefixKv>(&mut self, kv_store: &mut K, now: Time) -> Option<Time> {
        let mut latest = None;
        for (request, ready) in kv_store.take_landed_reads(now) {
            let candidate = self
                .reading
                .remove(&request)
                .expect("a landed read belongs to a reading request");
            self.landed.push_back(Landed {
                candidate,
                restored: false,
            });
            latest = Some(ready);
        }
        latest
    }

    /// Size each microbatch's prefill by `budget` over a `depth`-deep pipeline.
    pub(crate) fn with_microbatch_budget(mut self, budget: MicrobatchBudget, depth: u16) -> Self {
        assert!(depth > 0, "pipeline depth must be positive");
        self.budget = budget;
        self.depth = depth;
        self
    }

    /// Order started prompts against queued ones by `order`.
    pub(crate) fn with_started_order(mut self, order: StartedOrder) -> Self {
        self.started_order = order;
        self
    }

    /// Serve a request that has waited `after` since arrival before every
    /// other one, queued or started. Not vLLM.
    pub(crate) fn with_force_schedule_after(mut self, after: Time) -> Self {
        self.policy = self.policy.with_bound(after);
        self
    }

    /// End every non-final chunk on a checkpoint boundary of `alignment`.
    pub(crate) fn with_chunk_end_alignment(mut self, alignment: ChunkEndAlignment) -> Self {
        assert!(alignment.quantum > 0, "chunk-end quantum must be positive");
        self.chunk_end_alignment = Some(alignment);
        self
    }

    /// Cap each microbatch's decodes at `ceil(resident decodes / depth)`.
    pub(crate) fn with_balanced_decodes(mut self, depth: u16) -> Self {
        assert!(depth > 0, "pipeline depth must be positive");
        self.balanced_depth = Some(depth);
        self
    }

    /// The prefill the pipeline holds: started prompts' unscheduled tokens,
    /// queued prompts' tokens and the prefill tokens in flight.
    fn prefill_round_tokens<K: ChunkedPrefillKv>(&self, kv_store: &K) -> u64 {
        let started: u64 = self
            .started_prefills
            .iter()
            .map(|candidate| {
                u64::from(
                    kv_store
                        .resolved_prefill_context(candidate.request_id)
                        .remaining_prefill_tokens(),
                )
            })
            .sum();
        started + self.policy.queued_prompt_tokens() + self.in_flight_prefill_tokens
    }
}

impl<P, K> PrefillFill<K> for PipelinedChunkedPrefillAdmission<P>
where
    P: PendingOrderPolicy,
    K: TieredPrefixKv,
{
    type Step = ();

    fn is_overdue(&self, _partition: u16, arrival: Time, now: Time) -> bool {
        self.policy.is_overdue(arrival, now)
    }

    fn started_prefills_mut(&mut self, _partition: u16) -> &mut Vec<AdmissionCandidate> {
        &mut self.started_prefills
    }

    fn admit_fresh_prompts(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        _partition: u16,
        _step: &mut (),
        budget: u32,
        now: Time,
        gate: FreshGate,
    ) -> u32 {
        let mut remaining_budget = budget;
        let mut passed_over = Vec::new();
        while remaining_budget > 0 {
            let Some((source, candidate)) = self.peek_fresh(kv_store, gate) else {
                break;
            };
            if source == FreshSource::Landed && !self.landed[0].restored {
                // Its blocks were held since the read started: the context
                // fills them and the admission that follows takes them over.
                kv_store.restore_landed_read(PARTITION, candidate.request_id, now);
                self.landed[0].restored = true;
            } else if let Some(read) = kv_store.plan_tier_read(
                PARTITION,
                candidate.request_id,
                candidate.fresh_prompt_tokens,
                candidate.remaining_output_tokens,
                candidate.session_input,
                now,
            ) {
                if let Some(read_wait) = self
                    .read_wait
                    .filter(|read_wait| read_wait.exceeded(read.queue_wait))
                {
                    match read_wait.policy {
                        ReadWaitPolicy::PassOver => {
                            self.take_fresh(source, candidate);
                            passed_over.push((source, candidate));
                            continue;
                        }
                        ReadWaitPolicy::Block => break,
                    }
                }
                // The whole request must fit before its read starts (vLLM's
                // waiting loop stops at an async load that cannot allocate).
                if !kv_store.fits(PARTITION, &read.hold) {
                    break;
                }
                kv_store.start_tier_read(
                    PARTITION,
                    candidate.request_id,
                    candidate.session_input,
                    read,
                    now,
                );
                self.take_fresh(source, candidate);
                self.reading.insert(candidate.request_id, candidate);
                continue;
            }
            let resolved_prefill = kv_store.preview_prefill_context(
                PARTITION,
                candidate.fresh_prompt_tokens,
                candidate.session_input,
            );
            if !gate.admits(
                source != FreshSource::Pending,
                resolved_prefill.remaining_prefill_tokens(),
            ) {
                break;
            }
            let chunk_tokens = self.budget.chunk_tokens(
                resolved_prefill,
                remaining_budget,
                self.chunk_end_alignment,
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
            self.take_fresh(source, candidate);
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
        self.return_passed_over(passed_over);
        budget - remaining_budget
    }

    fn schedule_started_chunk(
        &mut self,
        kv_store: &mut K,
        _partition: u16,
        candidate: AdmissionCandidate,
        budget: u32,
    ) -> (u32, u32) {
        let chunk_tokens = self.budget.chunk_tokens(
            kv_store.resolved_prefill_context(candidate.request_id),
            budget,
            self.chunk_end_alignment,
            self.max_batch_tokens,
        );
        if chunk_tokens > 0 {
            kv_store.schedule_prefill_chunk(candidate.request_id, PARTITION, chunk_tokens);
        }
        (chunk_tokens, chunk_tokens)
    }
}

impl<P, K> MicrobatchAdmission<K> for PipelinedChunkedPrefillAdmission<P>
where
    P: PendingOrderPolicy,
    K: TieredPrefixKv,
{
    type Ticket = MicrobatchTicket;

    fn accept_request(&mut self, kv_store: &mut K, request: RequestId, context: &WorkerContext) {
        debug_assert_eq!(kv_store.num_partitions(), 1);
        let (
            arrival_time,
            fresh_prompt_tokens,
            target_output_tokens,
            session_input,
            conversation_start_time,
        ) = {
            let mut store = context.requests.borrow_mut();
            let record = &mut store[request];
            let arrival_time = record.request.core.arrival_time;
            context.stamp_stage(record, arrival_time, UnifiedStage::Pending as u16);
            (
                arrival_time,
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
            kv_store.reusable_prefix_tokens(fresh_prompt_tokens, session_input),
        );
        debug_assert!(!self.policy.contains(request));
        self.policy
            .push_arrived(candidate, arrival_time, &mut self.policy_context);
    }

    fn form_microbatch(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        batch_plan: &mut IterBatchPlan,
        now: Time,
    ) -> bool {
        self.land(kv_store, now);
        self.policy.promote(now);
        let mut remaining_budget = self.max_batch_tokens;
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
        if self.budget.follows_backlog() {
            remaining_budget = self.budget.prefill_budget(
                remaining_budget,
                self.prefill_round_tokens(kv_store),
                self.depth,
                self.max_batch_tokens,
            );
        }
        self.fill_prefill(
            self.started_order,
            kv_store,
            context,
            PARTITION,
            &mut (),
            remaining_budget,
            now,
        );
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
        self.in_flight_prefill_tokens += ticket.tokens();
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
        self.in_flight_prefill_tokens -= ticket.tokens();
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
        (self.policy.len() + self.reading.len() + self.landed.len()) as u32
    }

    fn land_reads(&mut self, kv_store: &mut K, now: Time) -> Option<Time> {
        self.land(kv_store, now)
    }

    fn next_read_landing(&self, kv_store: &K) -> Option<Time> {
        kv_store.next_tier_landing()
    }
}
