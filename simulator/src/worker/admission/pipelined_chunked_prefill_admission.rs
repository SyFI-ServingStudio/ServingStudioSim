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
//! Prefill takes as much pending work as the budget allows by default (vLLM),
//! so at moderate load a small microbatch forms from whatever arrived while
//! stage 0 was busy, and the backlog that builds meanwhile fills the next one to
//! `max_batch_tokens`. `with_even_split(depth, min)` instead sizes each
//! microbatch's prefill to one `depth`-th of the round the pipeline holds:
//! `clamp(ceil((pending + in-flight prefill tokens) / depth), min,
//! max_batch_tokens)`. Pending counts started prompts' unscheduled tokens and
//! queued fresh prompts' tokens (before any prefix hit, and whether or not KV
//! admits them yet); in-flight counts the prefill tokens of microbatches formed
//! and not yet exited. Counting the in-flight tokens keeps the size steady
//! through a round: one `P`-token prompt into an empty pipeline runs as `depth`
//! slices of `P / depth`, not a shrinking `P/depth, 3P/depth^2, ...` series.
//! The split never waits: a microbatch takes `min(pending, target)` and starts.
//!
//! The pending policy owns fresh requests. A prompt that has started and still
//! has unscheduled prefill tokens stays in `started_prefills`, in start order,
//! and keeps priority over fresh prompts unless `srpt` merges the two orders. A
//! request whose last chunk is in flight is in neither list; its ticket is in the
//! shell's in-flight queue. It joins KV's decode membership when that chunk
//! exits, and a decode step's request is in `in_flight_decodes` until its
//! microbatch exits.
//!
//! KV reserves the full request footprint once at admission and releases it at
//! completion. A hybrid recurrent model with prefix caching ends every non-final
//! chunk on a state-checkpoint boundary (vLLM's `_mamba_block_aligned_split`,
//! which does not change under PP), via the same rule as unified chunked
//! prefill. `with_long_prefill_threshold(t)` caps any one request's chunk at `t`
//! tokens per microbatch (vLLM's `long_prefill_token_threshold`), on top of the
//! greedy or even budget. One attention partition only.
//!
//! One ordering and one budget rule are not vLLM. `with_srpt()` orders prefill
//! by remaining tokens across started and queued prompts instead of started
//! first, so a short prompt does not wait behind a long one's chunks.
//! `with_load_budget(low, lo, hi)` runs microbatches at `low` tokens while the
//! prefill backlog (queued, started and in-flight tokens) is at most `lo`, rising
//! linearly to `max_batch_tokens` at `hi`, so a light load runs small
//! microbatches that drain the pipeline sooner.

use std::collections::HashSet;

use crate::common::{RequestId, Time, UnifiedStage};
use crate::worker::kv::ChunkedPrefillKv;
use crate::worker::shared::advance_scope::AdvanceScope;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::IterBatchPlan;

use super::chunked_prefill_admission::next_chunk_tokens;
use super::prefix_fetch::{AtHead, PrefixFetch};
use super::{AdmissionCandidate, EnqueueSequence, MicrobatchAdmission, PendingOrderPolicy};

/// The only attention partition of a pipeline head.
const PARTITION: u16 = 0;

pub struct PipelinedChunkedPrefillAdmission<P: PendingOrderPolicy> {
    policy: P,
    policy_context: P::Context,
    enqueue_sequence: EnqueueSequence,
    max_batch_tokens: u32,
    /// Checkpoint interval a non-final chunk must end on, if any.
    chunk_end_quantum: Option<u32>,
    /// vLLM's `long_prefill_token_threshold`: the most prefill tokens one
    /// request may take per microbatch.
    long_prefill_threshold: Option<u32>,
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
    /// Prefill target rule when microbatches are sized evenly over the round;
    /// `None` fills each microbatch greedily (vLLM).
    even_split: Option<EvenSplit>,
    /// Prefill tokens of the committed microbatches that have not exited.
    in_flight_prefill_tokens: u64,
    /// Budget that follows the prefill backlog; `None` keeps `max_batch_tokens`
    /// at every load.
    load_budget: Option<LoadBudget>,
    /// Shortest remaining prefill first across started and queued prompts;
    /// `false` keeps vLLM's started-first order.
    srpt: bool,
    /// DRAM/SSD tiers behind HBM, read when a request reaches the head.
    prefix_fetch: Option<PrefixFetch>,
}

/// [`PipelinedChunkedPrefillAdmission::with_load_budget`]: the microbatch
/// budget runs at `low_tokens` while the prefill backlog is at most
/// `backlog_lo_tokens` and rises linearly to `max_batch_tokens` at
/// `backlog_hi_tokens`. The backlog counts queued, started and in-flight
/// prefill tokens.
#[derive(Clone, Copy, Debug)]
struct LoadBudget {
    low_tokens: u32,
    backlog_lo_tokens: u64,
    backlog_hi_tokens: u64,
}

impl LoadBudget {
    fn target(&self, backlog_tokens: u64, max_tokens: u32) -> u32 {
        let rise = ((backlog_tokens as f64 - self.backlog_lo_tokens as f64)
            / (self.backlog_hi_tokens - self.backlog_lo_tokens) as f64)
            .clamp(0.0, 1.0);
        self.low_tokens + (f64::from(max_tokens - self.low_tokens) * rise).round() as u32
    }
}

/// Prefill sizing of [`PipelinedChunkedPrefillAdmission::with_even_split`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EvenSplit {
    depth: u16,
    /// Floor on the target, so a thin backlog does not run as tiny slices.
    min_tokens: u32,
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
            chunk_end_quantum: None,
            long_prefill_threshold: None,
            started_prefills: Vec::new(),
            in_flight_decodes: HashSet::new(),
            scheduled_decodes: Vec::new(),
            balanced_depth: None,
            even_split: None,
            in_flight_prefill_tokens: 0,
            load_budget: None,
            srpt: false,
            prefix_fetch: None,
        }
    }

    /// Read session contexts back from DRAM/SSD tiers when a request reaches
    /// the head of the queue (`prefix_fetch.rs`).
    pub(crate) fn with_prefix_fetch(mut self, fetch: PrefixFetch) -> Self {
        self.prefix_fetch = Some(fetch);
        self
    }

    /// End every non-final chunk on a multiple of `quantum` context tokens.
    pub(crate) fn with_chunk_end_quantum(mut self, quantum: u32) -> Self {
        assert!(quantum > 0, "chunk-end quantum must be positive");
        self.chunk_end_quantum = Some(quantum);
        self
    }

    /// Give no request more than `threshold` prefill tokens per microbatch, so a
    /// long prompt leaves the rest of the budget to other requests.
    pub(crate) fn with_long_prefill_threshold(mut self, threshold: u32) -> Self {
        assert!(threshold > 0, "long prefill threshold must be positive");
        self.long_prefill_threshold = Some(threshold);
        self
    }

    /// Run microbatches at `low_tokens` while the prefill backlog (queued,
    /// started and in-flight tokens) is at most `backlog_lo_tokens`, rising to
    /// `max_batch_tokens` at `backlog_hi_tokens`. Not vLLM, whose budget is
    /// fixed.
    pub(crate) fn with_load_budget(
        mut self,
        low_tokens: u32,
        backlog_lo_tokens: u64,
        backlog_hi_tokens: u64,
    ) -> Self {
        assert!(
            low_tokens > 0 && low_tokens <= self.max_batch_tokens,
            "load budget floor {low_tokens} must lie in 1..=max_batch_tokens {}",
            self.max_batch_tokens
        );
        assert!(
            backlog_lo_tokens < backlog_hi_tokens,
            "load budget needs backlog lo < hi tokens, got {backlog_lo_tokens}..{backlog_hi_tokens}"
        );
        self.load_budget = Some(LoadBudget {
            low_tokens,
            backlog_lo_tokens,
            backlog_hi_tokens,
        });
        self
    }

    /// Order prefill by remaining tokens across started and queued prompts: a
    /// queued prompt goes ahead of a started one with more left, and the reverse.
    /// The queue must offer its shortest prompt first. Not vLLM.
    pub(crate) fn with_srpt(mut self) -> Self {
        self.srpt = true;
        self
    }

    /// Cap each microbatch's decodes at `ceil(resident decodes / depth)`.
    pub(crate) fn with_balanced_decodes(mut self, depth: u16) -> Self {
        assert!(depth > 0, "pipeline depth must be positive");
        self.balanced_depth = Some(depth);
        self
    }

    /// Size each microbatch's prefill to `clamp(ceil(round tokens / depth),
    /// min_tokens, max_batch_tokens)`, where the round is the pending prefill
    /// plus the prefill already in flight.
    pub(crate) fn with_even_split(mut self, depth: u16, min_tokens: u32) -> Self {
        assert!(depth > 0, "pipeline depth must be positive");
        assert!(
            min_tokens <= self.max_batch_tokens,
            "min microbatch tokens {min_tokens} exceed max_batch_tokens {}",
            self.max_batch_tokens
        );
        self.even_split = Some(EvenSplit { depth, min_tokens });
        self
    }

    /// The prefill-token budget of the microbatch being formed, given what is
    /// left of `max_batch_tokens` after decodes.
    fn prefill_budget<K: ChunkedPrefillKv>(&self, kv_store: &K, remaining_budget: u32) -> u32 {
        if self.load_budget.is_none() && self.even_split.is_none() {
            return remaining_budget;
        }
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
        let round_tokens =
            started + self.policy.queued_prompt_tokens() + self.in_flight_prefill_tokens;
        let remaining_budget = match &self.load_budget {
            None => remaining_budget,
            Some(load) => remaining_budget.min(load.target(round_tokens, self.max_batch_tokens)),
        };
        let Some(EvenSplit { depth, min_tokens }) = self.even_split else {
            return remaining_budget;
        };
        let target = round_tokens
            .div_ceil(u64::from(depth))
            .clamp(u64::from(min_tokens), u64::from(self.max_batch_tokens));
        remaining_budget.min(target as u32)
    }

    /// Admit fresh prompts into `budget`, stopping at the first one with more
    /// than `no_longer_than` prefill tokens; returns the tokens they took.
    fn admit_fresh_prompts<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        budget: u32,
        now: Time,
        no_longer_than: Option<u32>,
    ) -> u32 {
        let mut remaining_budget = budget;
        while remaining_budget > 0 {
            let landed = self
                .prefix_fetch
                .as_ref()
                .and_then(|fetch| fetch.peek_landed(PARTITION));
            let candidate = match landed {
                Some(candidate) => candidate,
                None => {
                    let fetch = &self.prefix_fetch;
                    self.policy.refresh_head(&mut |candidate| {
                        let hbm = kv_store.resident_prefix_tokens(
                            candidate.fresh_prompt_tokens,
                            candidate.session_input,
                        );
                        fetch.as_ref().map_or(hbm, |fetch| {
                            fetch.rank_tokens(PARTITION, candidate.session_input, hbm)
                        })
                    });
                    let Some(candidate) = self.policy.peek() else {
                        break;
                    };
                    candidate
                }
            };
            if let Some(fetch) = &mut self.prefix_fetch {
                match fetch.at_head(kv_store, PARTITION, candidate, now) {
                    AtHead::Admit => {}
                    AtHead::Read => {
                        self.take_candidate(landed.is_some(), candidate);
                        continue;
                    }
                    AtHead::Blocked => break,
                }
            }
            let resolved_prefill = kv_store.preview_prefill_context(
                PARTITION,
                candidate.fresh_prompt_tokens,
                candidate.session_input,
            );
            if no_longer_than
                .is_some_and(|bound| resolved_prefill.remaining_prefill_tokens() > bound)
            {
                break;
            }
            let chunk_tokens = next_chunk_tokens(
                resolved_prefill,
                remaining_budget,
                self.chunk_end_quantum,
                self.max_batch_tokens,
                self.long_prefill_threshold,
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
            self.take_candidate(landed.is_some(), candidate);
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
            if let Some(fetch) = &mut self.prefix_fetch {
                fetch.admitted(
                    PARTITION,
                    candidate,
                    resolved_prefill.resident_prefix_tokens(),
                    now,
                );
            }
            kv_store.schedule_prefill_chunk(candidate.request_id, PARTITION, chunk_tokens);
            remaining_budget -= chunk_tokens;
            if chunk_tokens < resolved_prefill.remaining_prefill_tokens() {
                self.started_prefills.push(candidate);
            }
        }
        budget - remaining_budget
    }

    /// Take `candidate` out of the landed queue or the pending order, wherever
    /// [`Self::admit_fresh_prompts`] found it.
    fn take_candidate(&mut self, landed: bool, candidate: AdmissionCandidate) {
        if landed {
            self.prefix_fetch
                .as_mut()
                .expect("a landed request implies prefix fetch")
                .pop_landed(PARTITION);
        } else {
            let popped = self.policy.pop(&mut self.policy_context);
            debug_assert_eq!(popped, Some(candidate));
        }
    }

    /// Fill `budget` shortest remaining prefill first: before each started
    /// prompt, queued prompts with no more tokens left than it go first.
    fn form_prefill_srpt<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        mut budget: u32,
        now: Time,
    ) {
        let remaining_of = |kv_store: &K, candidate: &AdmissionCandidate| {
            kv_store
                .resolved_prefill_context(candidate.request_id)
                .remaining_prefill_tokens()
        };
        self.started_prefills
            .sort_by_key(|candidate| remaining_of(kv_store, candidate));
        let started = std::mem::take(&mut self.started_prefills);
        let mut still_started = Vec::with_capacity(started.len());
        for candidate in started {
            let remaining = remaining_of(kv_store, &candidate);
            if budget > 0 {
                budget -= self.admit_fresh_prompts(kv_store, context, budget, now, Some(remaining));
            }
            let resolved_prefill = kv_store.resolved_prefill_context(candidate.request_id);
            let chunk_tokens = next_chunk_tokens(
                resolved_prefill,
                budget,
                self.chunk_end_quantum,
                self.max_batch_tokens,
                self.long_prefill_threshold,
            );
            if chunk_tokens > 0 {
                kv_store.schedule_prefill_chunk(candidate.request_id, PARTITION, chunk_tokens);
                budget -= chunk_tokens;
            }
            if chunk_tokens < remaining {
                still_started.push(candidate);
            }
        }
        if budget > 0 {
            self.admit_fresh_prompts(kv_store, context, budget, now, None);
        }
        // Prompts admitted now and not finished join after the ones already started.
        still_started.append(&mut self.started_prefills);
        self.started_prefills = still_started;
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
        let hbm_tokens = kv_store.resident_prefix_tokens(fresh_prompt_tokens, session_input);
        let candidate = self.enqueue_sequence.freeze(
            request,
            fresh_prompt_tokens,
            target_output_tokens,
            session_input,
            conversation_start_time,
            self.prefix_fetch.as_ref().map_or(hbm_tokens, |fetch| {
                fetch.rank_tokens(PARTITION, session_input, hbm_tokens)
            }),
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
        self.land_prefix_reads(kv_store, now);
        let max_batch_tokens = self.max_batch_tokens;
        let chunk_end_quantum = self.chunk_end_quantum;
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
        let mut remaining_budget = self.prefill_budget(kv_store, remaining_budget);
        if self.srpt {
            self.form_prefill_srpt(kv_store, context, remaining_budget, now);
        } else {
            let long_prefill_threshold = self.long_prefill_threshold;
            // Started prompts keep start order. One that cannot run keeps its place.
            self.started_prefills.retain(|candidate| {
                let resolved_prefill = kv_store.resolved_prefill_context(candidate.request_id);
                let chunk_tokens = next_chunk_tokens(
                    resolved_prefill,
                    remaining_budget,
                    chunk_end_quantum,
                    max_batch_tokens,
                    long_prefill_threshold,
                );
                if chunk_tokens == 0 {
                    return true;
                }
                kv_store.schedule_prefill_chunk(candidate.request_id, PARTITION, chunk_tokens);
                remaining_budget -= chunk_tokens;
                chunk_tokens < resolved_prefill.remaining_prefill_tokens()
            });
            self.admit_fresh_prompts(kv_store, context, remaining_budget, now, None);
        }
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
        self.policy.len() as u32
    }

    fn land_prefix_reads(&mut self, kv_store: &mut K, now: Time) -> Option<Time> {
        self.prefix_fetch.as_mut()?.land(kv_store, now)
    }

    fn next_prefix_read(&self) -> Option<Time> {
        self.prefix_fetch.as_ref()?.next_landing()
    }

    fn reading_requests(&self) -> u32 {
        self.prefix_fetch.as_ref().map_or(0, PrefixFetch::reading)
    }

    fn store_session_context(&mut self, _partition: u16, session_id: u32, tokens: u64) {
        if let Some(fetch) = &mut self.prefix_fetch {
            fetch.store(PARTITION, session_id, tokens);
        }
    }
}
