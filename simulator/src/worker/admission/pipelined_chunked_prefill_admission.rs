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
//! that is a pass-through unless a bound is set. Under hits-first-when-at-risk
//! the lifecycle turns the deadline's cost-class split on at each formation
//! while serving the overdue misses oldest first would evict a queued
//! request's held tier context ([`TieredPrefixKv::tier_sessions_at_risk`]). A prompt that has started and
//! still has unscheduled prefill tokens stays in `started_prefills`, in start
//! order, and keeps priority over fresh prompts unless the started order is
//! shortest-remaining (`started_order.rs`). A request whose last chunk is in flight
//! is in neither list; its ticket is in the shell's in-flight queue. It joins KV's
//! decode membership when that chunk exits, and a decode step's request is in
//! `in_flight_decodes` until its microbatch exits.
//!
//! With host tiers behind HBM (`TieredPrefixKv`), a fresh request whose
//! context a tier holds beyond HBM is read back before admission, as under
//! vLLM's KV connector, or through loading areas under staged loading; the
//! reading -> landed lifecycle is shared with the whole-iteration chunked
//! prefill (`fresh_reads.rs`).
//!
//! KV reserves the full request footprint once at admission and releases it at
//! completion. A hybrid recurrent model with prefix caching ends every non-final
//! chunk on a state-checkpoint boundary (vLLM's `_mamba_block_aligned_split`,
//! which does not change under PP), via the same rule as unified chunked
//! prefill. One attention partition only.
//!
//! Under balanced retention (`with_balanced_retention`, not vLLM) a retained
//! context lies once over the stages, and a hit first pulls each stage's
//! copies back with the copy engine (`worker::kv::BalancedRetention`). The
//! request is admitted as any other, its KV reserved and its prefix taken from
//! the cache, but its first chunk waits in `copying` until the pull lands.
//! Pulls run one after another on the pipeline's copy queue and overlap the
//! stages' compute; a landed request joins the started prompts.

use std::collections::{HashSet, VecDeque};

use crate::worker::shared::back_stash::{BackEntry, BackStash};

use crate::common::{RequestId, Time, UnifiedStage};
use crate::worker::kv::{BalancedRetention, ChunkedPrefillKv, TieredPrefixKv};
use crate::worker::shared::advance_scope::AdvanceScope;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::IterBatchPlan;

use super::chunked_prefill_admission::ChunkEndAlignment;
use super::external_decode::ExternalDecode;
use super::fresh_reads::{AtHead, FreshReads, FreshSource};
use super::microbatch_budget::{BudgetTally, MicrobatchBudget};
use super::policy::{ForceSchedule, Queued};
use super::read_wait::ReadWait;
use super::prefill_rank::{PrefillRank, Rank};
use super::started_order::{Admitted, FillPrefillBudget, FreshGate, StartedOrder};
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
    /// Per-minute share of full-cap budgets, kept when the budget varies.
    tally: Option<BudgetTally>,
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
    /// Fresh requests out of their queue for a host-tier read.
    reads: FreshReads,
    /// The decode runs elsewhere: a request completes at its first token.
    external_decode: Option<ExternalDecode>,
    /// Back-pass batching (see [`crate::worker::shared::back_stash`]): a
    /// microbatch's tails enter the stash when it exits, and a later
    /// microbatch's back pass carries them; the request's first token is
    /// emitted when that microbatch exits.
    back: Option<PipelineBack>,
    /// Balanced retention: hits whose copy-engine pull has not landed.
    copies: Option<PrefixCopies>,
}

/// Hits waiting on their pull, and the pipeline's copy queue.
struct PrefixCopies {
    balanced: BalancedRetention,
    /// When the copy queue is free again.
    free_at: Time,
    /// Admitted hits in landing order (the queue is FIFO), with when they land.
    copying: VecDeque<(Time, AdmissionCandidate)>,
    tally: CopyTally,
}

/// Per-minute log of the pulls ("prefix copy tally"): count, bytes, and how
/// long each request waited from admission to the landing.
#[derive(Default)]
struct CopyTally {
    worker: u16,
    minute: u64,
    copies: u64,
    bytes: u64,
    wait_us_sum: u64,
    wait_us_max: u64,
    /// Waits by upper edge [`COPY_WAIT_EDGES_US`], the last bucket open.
    buckets: [u64; COPY_WAIT_EDGES_US.len() + 1],
}

const COPY_WAIT_EDGES_US: [u64; 12] = [
    50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 1_000_000,
];

impl CopyTally {
    fn record(&mut self, now: Time, bytes: u64, wait: Time) {
        let minute = (now.as_ms() / 60_000.0) as u64;
        if minute != self.minute {
            self.flush();
            self.minute = minute;
        }
        let wait_us = wait.as_ns() / 1_000;
        self.copies += 1;
        self.bytes += bytes;
        self.wait_us_sum += wait_us;
        self.wait_us_max = self.wait_us_max.max(wait_us);
        let bucket = COPY_WAIT_EDGES_US
            .iter()
            .position(|&edge| wait_us <= edge)
            .unwrap_or(COPY_WAIT_EDGES_US.len());
        self.buckets[bucket] += 1;
    }

    fn flush(&mut self) {
        if self.copies > 0 {
            tracing::info!(
                worker = self.worker,
                minute = self.minute,
                copies = self.copies,
                bytes = self.bytes,
                wait_us_sum = self.wait_us_sum,
                wait_us_max = self.wait_us_max,
                buckets = ?self.buckets,
                "prefix copy tally"
            );
        }
        // Field by field: assigning a fresh tally would drop (and flush) this one.
        self.copies = 0;
        self.bytes = 0;
        self.wait_us_sum = 0;
        self.wait_us_max = 0;
        self.buckets = [0; COPY_WAIT_EDGES_US.len() + 1];
    }
}

impl Drop for CopyTally {
    fn drop(&mut self) {
        self.flush();
    }
}

struct PipelineBack {
    stash: BackStash,
    /// Microbatches formed so far.
    step: u64,
    /// The requests the microbatch being formed flushes.
    flushing: Vec<RequestId>,
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
    /// Back batching: the entries this microbatch's chunks leave, stashed
    /// when it exits.
    back_entries: Vec<BackEntry>,
    /// Back batching: the requests whose tails this microbatch's back pass
    /// carries, completed when it exits.
    back_requests: Vec<RequestId>,
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
            tally: None,
            started_order: StartedOrder::default(),
            chunk_end_alignment: None,
            started_prefills: Vec::new(),
            in_flight_decodes: HashSet::new(),
            scheduled_decodes: Vec::new(),
            balanced_depth: None,
            in_flight_prefill_tokens: 0,
            reads: FreshReads::new(1),
            external_decode: None,
            back: None,
            copies: None,
        }
    }

    /// Store retained contexts once over the stages and pull each hit's
    /// copies back before its first chunk (`balanced`). Not vLLM.
    pub(crate) fn with_balanced_retention(
        mut self,
        balanced: BalancedRetention,
        worker: u16,
    ) -> Self {
        self.copies = Some(PrefixCopies {
            balanced,
            free_at: Time::ZERO,
            copying: VecDeque::new(),
            tally: CopyTally {
                worker,
                ..CopyTally::default()
            },
        });
        self
    }

    /// Move the hits whose pull has landed by `now` to the started prompts.
    fn land_copies(&mut self, now: Time) {
        let Some(copies) = self.copies.as_mut() else {
            return;
        };
        while copies
            .copying
            .front()
            .is_some_and(|&(ready, _)| ready <= now)
        {
            let (_, candidate) = copies.copying.pop_front().unwrap();
            self.started_prefills.push(candidate);
        }
    }

    #[cfg(test)]
    pub(crate) fn reading_requests(&self) -> usize {
        self.reads.reading()
    }

    /// Run each request to its first token and take its context back from a
    /// decode elsewhere. Not vLLM.
    pub(crate) fn with_external_decode(mut self) -> Self {
        self.external_decode = Some(ExternalDecode::default());
        self
    }

    /// Start a host-tier read only within `read_wait` of its tier's channel.
    /// Not vLLM.
    pub(crate) fn with_read_wait(mut self, read_wait: ReadWait) -> Self {
        self.reads.set_read_wait(read_wait);
        self
    }

    /// Size each microbatch's prefill by `budget` over a `depth`-deep pipeline.
    pub(crate) fn with_microbatch_budget(mut self, budget: MicrobatchBudget, depth: u16) -> Self {
        assert!(depth > 0, "pipeline depth must be positive");
        self.budget = budget;
        self.depth = depth;
        self
    }

    /// Log the per-minute share of full-cap budgets as worker `worker`'s,
    /// when the budget varies.
    pub(crate) fn with_budget_tally(mut self, worker: u16) -> Self {
        self.tally = self.budget.varies().then(|| BudgetTally::new(worker));
        self
    }

    /// Order started prompts against queued ones by `order`.
    pub(crate) fn with_started_order(mut self, order: StartedOrder) -> Self {
        self.started_order = order;
        self
    }

    /// Serve a request that has waited `after` since arrival before every
    /// other one, queued or started, in the cost classes `schedule` names.
    /// Not vLLM.
    pub(crate) fn with_force_schedule(mut self, schedule: ForceSchedule) -> Self {
        self.policy = self.policy.with_force_schedule(schedule);
        self
    }

    /// Batch the back pass of a bounded-replay arch through `stash`.
    pub(crate) fn with_back_stash(mut self, stash: BackStash) -> Self {
        self.back = Some(PipelineBack {
            stash,
            step: 0,
            flushing: Vec::new(),
        });
        self
    }

    /// Order started and queued prompts by `rank`. Not vLLM.
    pub(crate) fn with_prefill_rank(mut self, rank: PrefillRank) -> Self {
        self.policy = self.policy.with_rank(rank);
        self.started_order = StartedOrder::Ranked;
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
        let copying: u64 = self.copies.as_ref().map_or(0, |copies| {
            copies
                .copying
                .iter()
                .map(|(_, candidate)| {
                    u64::from(
                        kv_store
                            .resolved_prefill_context(candidate.request_id)
                            .remaining_prefill_tokens(),
                    )
                })
                .sum()
        });
        started + copying + self.policy.queued_prompt_tokens() + self.in_flight_prefill_tokens
    }
}

impl<P, K> FillPrefillBudget<K> for PipelinedChunkedPrefillAdmission<P>
where
    P: PendingOrderPolicy,
    K: TieredPrefixKv,
{
    type Step = ();

    fn is_overdue(&self, _partition: u16, arrival: Time, now: Time) -> bool {
        self.policy.is_overdue(arrival, now)
    }

    fn splits_by_class(&self, _partition: u16) -> bool {
        self.policy.splits_by_class()
    }

    fn rank_of(
        &self,
        _partition: u16,
        request: RequestId,
        remaining: u32,
        now: Time,
    ) -> Option<Rank> {
        self.policy.rank_of(request, remaining, now)
    }

    fn pads(&self, _partition: u16) -> bool {
        self.policy.pads()
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
    ) -> Admitted {
        let mut finished = 0;
        let mut remaining_budget = budget;
        let mut passed_over = Vec::new();
        let (policy, policy_context) = (&mut self.policy, &mut self.policy_context);
        if kv_store.stages_tier_reads() && gate.queues.takes_pending() {
            self.reads
                .start_staged_reads(PARTITION, policy, policy_context, kv_store, now);
        }
        while remaining_budget > 0 {
            let Some((source, candidate)) =
                self.reads
                    .peek(PARTITION, policy, policy_context, kv_store, gate.queues)
            else {
                break;
            };
            match self.reads.at_head(
                PARTITION,
                policy,
                policy_context,
                kv_store,
                source,
                candidate,
                &mut passed_over,
                now,
            ) {
                AtHead::Admit => {}
                AtHead::Moved => continue,
                AtHead::Stop => break,
            }
            let resolved_prefill = kv_store.preview_prefill_context(
                PARTITION,
                candidate.fresh_prompt_tokens,
                candidate.session_input,
            );
            // A landed read goes first, whatever the rank.
            let head_rank = match source {
                FreshSource::Landed => None,
                FreshSource::Queued(_) => policy.head_rank(),
            };
            if !gate.admits(
                source == FreshSource::Queued(Queued::Pending),
                resolved_prefill.remaining_prefill_tokens(),
                head_rank,
            ) {
                break;
            }
            let chunk_tokens = self.budget.chunk_tokens(
                resolved_prefill,
                remaining_budget,
                self.chunk_end_alignment,
                self.max_batch_tokens,
            );
            if chunk_tokens == 0
                || gate.pads_out(
                    finished,
                    chunk_tokens,
                    resolved_prefill.remaining_prefill_tokens(),
                    head_rank,
                )
            {
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
            self.reads
                .take(PARTITION, policy, policy_context, source, candidate);
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
            // A retained hit's copies come back first; it takes no budget.
            let pull = self.copies.as_ref().and_then(|copies| {
                resolved_prefill.session_id().and_then(|_| {
                    copies
                        .balanced
                        .copy(resolved_prefill.resident_prefix_tokens())
                })
            });
            if let (Some(pull), Some(copies)) = (pull, self.copies.as_mut()) {
                let ready = copies.free_at.max(now) + pull.duration;
                copies.free_at = ready;
                copies.copying.push_back((ready, candidate));
                copies.tally.record(now, pull.bytes, ready - now);
                continue;
            }
            kv_store.schedule_prefill_chunk(candidate.request_id, PARTITION, chunk_tokens);
            remaining_budget -= chunk_tokens;
            if chunk_tokens < resolved_prefill.remaining_prefill_tokens() {
                self.started_prefills.push(candidate);
            } else {
                finished += 1;
            }
        }
        FreshReads::return_passed_over(policy, policy_context, passed_over);
        Admitted {
            taken: budget - remaining_budget,
            finished,
        }
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
            if let Some(external_decode) = &mut self.external_decode {
                external_decode.accept(request, &mut record.request.definition);
            }
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
            kv_store.reusable_prefix_tokens(PARTITION, fresh_prompt_tokens, session_input),
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
        self.reads.land(kv_store, now);
        // A hit computes only its new tokens: its context is in HBM or a tier.
        self.policy.promote(now, |candidate| {
            kv_store.reusable_prefix_tokens(
                PARTITION,
                candidate.fresh_prompt_tokens,
                candidate.session_input,
            )
        });
        if let Some(horizon) = self.policy.at_risk_horizon() {
            let at_risk = horizon > 0
                && FreshReads::held_context_at_risk(PARTITION, &self.policy, kv_store, horizon);
            self.policy.set_hits_first(at_risk);
        }
        self.land_copies(now);
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
        if self.budget.varies() {
            let round = if self.budget.reads_round() {
                self.prefill_round_tokens(kv_store)
            } else {
                0
            };
            let rule = self.budget.rule_budget(
                round,
                self.depth,
                self.policy.has_late(),
                self.max_batch_tokens,
            );
            if let Some(tally) = self.tally.as_mut() {
                tally.record(now, rule >= self.max_batch_tokens);
            }
            remaining_budget = remaining_budget.min(rule);
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
        let has_front = kv_store.has_prefill_admit(PARTITION) || !self.scheduled_decodes.is_empty();
        // A front that cannot run this formation (every waiting prompt blocked,
        // e.g. on KV the stash's own requests hold) counts as not pending: the
        // stash flushes rather than wait for a formation that never comes.
        let front_pending = has_front
            && (!self.started_prefills.is_empty()
                || self.reads.len() > self.reads.reading()
                || self.policy.len() > 0);
        let Some(back) = self.back.as_mut() else {
            return has_front;
        };
        // Only exited microbatches' tails are in the stash.
        let flushes = back.stash.should_flush(&[], front_pending, back.step + 1);
        back.flushing.clear();
        if flushes {
            let batch = back.stash.take(&[]);
            back.flushing = batch.requests;
            batch_plan.back_pass = Some(batch.groups);
        } else {
            batch_plan.back_pass = Some(back.stash.empty_pass());
        }
        has_front || flushes
    }

    fn commit_microbatch(&mut self, kv_store: &mut K, now: Time) -> MicrobatchTicket {
        let mut ticket = MicrobatchTicket {
            decodes: std::mem::take(&mut self.scheduled_decodes),
            ..MicrobatchTicket::default()
        };
        if let Some(back) = self.back.as_mut() {
            back.step += 1;
            ticket.back_requests = std::mem::take(&mut back.flushing);
            let (stash, step) = (&back.stash, back.step);
            kv_store.visit_prefill_admits(PARTITION, |request| {
                let resolved = kv_store.resolved_prefill_context(request);
                let (prefix, chunk) = resolved.active_chunk();
                let finishes = chunk == resolved.remaining_prefill_tokens();
                if let Some((prefix, rows)) = stash.chunk_entry(
                    prefix,
                    chunk,
                    resolved.resident_prefix_tokens(),
                    prefix + resolved.remaining_prefill_tokens(),
                ) {
                    ticket.back_entries.push(BackEntry {
                        group: 0,
                        prefix,
                        rows,
                        request: finishes.then_some(request),
                        step,
                    });
                }
            });
        }
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

    fn split_back(&mut self, ticket: &mut MicrobatchTicket) -> Option<MicrobatchTicket> {
        if self.back.is_none() || !ticket.decodes.is_empty() {
            return None;
        }
        Some(MicrobatchTicket {
            back_requests: std::mem::take(&mut ticket.back_requests),
            ..MicrobatchTicket::default()
        })
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
        let MicrobatchTicket {
            chunks,
            decodes,
            back_entries,
            back_requests,
        } = ticket;
        let deferred = self.back.is_some();
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
                if !chunk.finishes_prefill || deferred {
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
                match &mut self.external_decode {
                    Some(external_decode) => {
                        external_decode.release(kv_store, chunk.request, PARTITION, at);
                    }
                    None => kv_store.release_retaining_prefix(chunk.request, PARTITION, at),
                }
                completed.push(chunk.request);
            }
        }
        if let Some(back) = self.back.as_mut() {
            for entry in back_entries {
                back.stash.push(entry);
            }
        }
        // The requests whose back pass this microbatch carried.
        for request in back_requests {
            let remaining_output_tokens = {
                let mut store = context.requests.borrow_mut();
                let record = &mut store[request];
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
            kv_store.finish_chunked_prefill(request, PARTITION, remaining_output_tokens);
            if remaining_output_tokens == 0 {
                kv_store.release_retaining_prefix(request, PARTITION, at);
                completed.push(request);
            }
        }
        kv_store.sample_submit(PARTITION, at);
        if let Some(external_decode) = &mut self.external_decode {
            external_decode.hand_back(kv_store, PARTITION, at);
        }
    }

    fn queued_requests(&self) -> u32 {
        (self.policy.len() + self.reads.len()) as u32
    }

    fn land_reads(&mut self, kv_store: &mut K, now: Time) -> Option<Time> {
        self.reads.land(kv_store, now)
    }

    fn next_read_landing(&self, kv_store: &K) -> Option<Time> {
        kv_store.next_tier_landing()
    }

    fn next_landing(&self) -> Option<Time> {
        self.copies
            .as_ref()
            .and_then(|copies| copies.copying.front().map(|&(ready, _)| ready))
    }

    fn next_landing_after(&self, now: Time) -> Option<Time> {
        let copying = &self.copies.as_ref()?.copying;
        let first = copying.partition_point(|&(ready, _)| ready <= now);
        copying.get(first).map(|&(ready, _)| ready)
    }
}
