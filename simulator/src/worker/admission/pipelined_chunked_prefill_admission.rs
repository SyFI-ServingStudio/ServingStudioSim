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
//! `with_slot_plan(depth)` instead keeps a plan of the next `depth`
//! microbatches (`MicrobatchSlotPlan`) and bin-packs whole requests into the
//! least-loaded one, splitting a request only at `max_batch_tokens`. A request
//! is admitted, and its full KV footprint reserved, when it is first placed:
//! a planned chunk can then always launch, and at most `depth` microbatches'
//! worth of prompts hold KV before their first chunk runs. Requests that find no
//! room stay in the pending policy. At launch the head slot's chunks are
//! scheduled as planned; ready decodes still come first, and a chunk they push
//! past `max_batch_tokens` is cut and its request re-planned.
//!
//! The pending policy owns fresh requests. A prompt that has started and still
//! has unscheduled prefill tokens stays in `started_prefills`, in start order,
//! and keeps priority over fresh prompts. A request whose last chunk is in flight
//! is in neither list; its ticket is in the shell's in-flight queue. It joins KV's
//! decode membership when that chunk exits, and a decode step's request is in
//! `in_flight_decodes` until its microbatch exits.
//!
//! KV reserves the full request footprint once at admission and releases it at
//! completion. A hybrid recurrent model with prefix caching ends every non-final
//! chunk on a state-checkpoint boundary (vLLM's `_mamba_block_aligned_split`,
//! which does not change under PP), via the same rule as unified chunked
//! prefill. `with_long_prefill_threshold(t)` caps any one request's chunk at `t`
//! tokens per microbatch (vLLM's `long_prefill_token_threshold`), on top of the
//! greedy or even budget. One attention partition only.

use std::collections::{HashSet, VecDeque};

use crate::common::{RequestId, Time, UnifiedStage};
use crate::worker::config::LongPrefillCapMode;
use crate::worker::kv::{ChunkedPrefillKv, ResolvedPrefillContext};
use crate::worker::shared::advance_scope::AdvanceScope;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::IterBatchPlan;

use super::chunked_prefill_admission::next_chunk_tokens;
use super::microbatch_slot_plan::{MicrobatchSlotPlan, UnplannedPrefill};
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
    /// When `long_prefill_threshold` applies.
    cap_mode: LongPrefillCapMode,
    /// The threshold applies only to prompts with at least this many fresh
    /// tokens (`0`: every prompt).
    cap_min_prompt_tokens: u32,
    /// Fresh-first prefill order: fresh prompts fill the microbatch before
    /// started ones, except for up to this many tokens each started prompt
    /// keeps. `None` is vLLM's started-first order.
    fresh_first_reserve: Option<u32>,
    /// Queued fresh prompt tokens when the microbatch being formed began.
    queued_at_formation: u64,
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
    /// The next microbatches' planned prefill; `None` schedules at formation.
    slot_plan: Option<MicrobatchSlotPlan>,
    /// Budget that follows stage 0's recent busy fraction; `None` keeps
    /// `max_batch_tokens` at every load.
    load_budget: Option<LoadBudget>,
    /// Chunk sizes that shrink toward each prompt's end; `None` lets a prompt
    /// take full chunks up to its last.
    tail_ladder: Option<TailLadder>,
    /// Shortest remaining prefill first across started and queued prompts;
    /// `false` keeps the started-first or fresh-first order.
    srpt: bool,
}

/// [`PipelinedChunkedPrefillAdmission::with_load_budget`]: the microbatch
/// budget runs at `low_tokens` while the head often runs out of work and rises
/// linearly to `max_batch_tokens` as the fraction of `window` it had work goes
/// from `busy_lo` to `busy_hi`. Having work means a formation could schedule
/// something; time blocked only by the in-flight window counts as busy, so a
/// pipeline with bubbles still reads 1 when saturated.
#[derive(Clone, Debug)]
struct LoadBudget {
    low_tokens: u32,
    window: Time,
    busy_lo: f64,
    busy_hi: f64,
    /// Rise with the prefill backlog instead, from `.0` to `.1` tokens queued,
    /// started and in flight; `None` uses the busy fraction.
    backlog: Option<(u64, u64)>,
    /// Since when formations have found nothing to schedule, if they do now.
    starved_since: Option<Time>,
    /// Past starved intervals that end inside the window, oldest first.
    starved: VecDeque<(Time, Time)>,
}

impl LoadBudget {
    fn note_formation(&mut self, now: Time, formed: bool) {
        match (self.starved_since, formed) {
            (None, false) => self.starved_since = Some(now),
            (Some(since), true) => {
                self.starved_since = None;
                self.starved.push_back((since, now));
            }
            _ => {}
        }
        while self
            .starved
            .front()
            .is_some_and(|&(_, end)| end.0 + self.window.0 < now.0)
        {
            self.starved.pop_front();
        }
    }

    fn busy_fraction(&self, now: Time) -> f64 {
        let from = now.0.saturating_sub(self.window.0);
        let starved: u64 = self
            .starved
            .iter()
            .copied()
            .chain(self.starved_since.map(|since| (since, now)))
            .map(|(start, end)| end.0.min(now.0).saturating_sub(start.0.max(from)))
            .sum();
        1.0 - starved as f64 / self.window.0 as f64
    }

    fn target(&self, now: Time, backlog_tokens: u64, max_tokens: u32) -> u32 {
        let rise = match self.backlog {
            None => (self.busy_fraction(now) - self.busy_lo) / (self.busy_hi - self.busy_lo),
            Some((lo, hi)) => (backlog_tokens as f64 - lo as f64) / (hi - lo) as f64,
        }
        .clamp(0.0, 1.0);
        self.low_tokens + (f64::from(max_tokens - self.low_tokens) * rise).round() as u32
    }
}

/// [`PipelinedChunkedPrefillAdmission::with_tail_ladder`]: the largest chunk a
/// prompt may take given how many prefill tokens it has left. A prompt's last
/// chunk still crosses every later stage after stage 0 is done with it, so its
/// first token lands `depth - 1` stage times after its last stage-0 slot. With
/// a stage time of `overhead + tokens`, a chunk at most `(depth - 2) / (depth -
/// 1)` the stage time of the one before it never waits behind that one on a
/// later stage, so shrinking the tail that way shortens the drain without
/// stalling it.
#[derive(Clone, Debug)]
struct TailLadder {
    /// `(chunk tokens, tokens from the end through this step)`, last chunk first.
    steps: Vec<(u32, u64)>,
}

impl TailLadder {
    fn new(depth: u16, min_tokens: u32, overhead_tokens: u32, max_tokens: u32) -> Self {
        assert!(min_tokens > 0, "tail ladder floor must be positive");
        let growth = if depth > 2 {
            f64::from(depth - 1) / f64::from(depth - 2)
        } else {
            f64::INFINITY
        };
        let overhead = f64::from(overhead_tokens);
        let mut steps = Vec::new();
        let mut chunk = min_tokens.min(max_tokens);
        let mut through = 0_u64;
        loop {
            through += u64::from(chunk);
            steps.push((chunk, through));
            let next = (growth * (overhead + f64::from(chunk)) - overhead).ceil();
            if next >= f64::from(max_tokens) {
                break;
            }
            chunk = next as u32;
        }
        Self { steps }
    }

    /// The chunk cap for a prompt with `remaining` prefill tokens left. Above
    /// the ladder a prompt may take what lands it on the top step, but no less
    /// than that step, so its chunks never grow.
    fn cap(&self, remaining: u32) -> u32 {
        let remaining = u64::from(remaining);
        match self
            .steps
            .iter()
            .find(|&&(_, through)| remaining <= through)
        {
            Some(&(chunk, _)) => chunk,
            None => {
                let &(top, through) = self.steps.last().expect("a ladder has a step");
                (remaining - through)
                    .max(u64::from(top))
                    .min(u64::from(u32::MAX)) as u32
            }
        }
    }
}

/// The tighter of two optional caps.
fn tighter(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (cap, None) | (None, cap) => cap,
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
            cap_mode: LongPrefillCapMode::Always,
            cap_min_prompt_tokens: 0,
            fresh_first_reserve: None,
            queued_at_formation: 0,
            started_prefills: Vec::new(),
            in_flight_decodes: HashSet::new(),
            scheduled_decodes: Vec::new(),
            balanced_depth: None,
            even_split: None,
            in_flight_prefill_tokens: 0,
            slot_plan: None,
            load_budget: None,
            tail_ladder: None,
            srpt: false,
        }
    }

    /// End every non-final chunk on a multiple of `quantum` context tokens.
    pub(crate) fn with_chunk_end_quantum(mut self, quantum: u32) -> Self {
        assert!(quantum > 0, "chunk-end quantum must be positive");
        assert!(
            self.slot_plan.is_none(),
            "the microbatch slot plan does not align chunk ends; use plain chunk alignment"
        );
        self.chunk_end_quantum = Some(quantum);
        self
    }

    /// Give no request more than `threshold` prefill tokens per microbatch, so a
    /// long prompt leaves the rest of the budget to other requests.
    pub(crate) fn with_long_prefill_threshold(mut self, threshold: u32) -> Self {
        assert!(threshold > 0, "long prefill threshold must be positive");
        assert!(
            self.slot_plan.is_none(),
            "the microbatch slot plan sizes chunks itself; drop the long prefill threshold"
        );
        self.long_prefill_threshold = Some(threshold);
        self
    }

    /// Choose when the long prefill threshold applies. Anything but `Always`
    /// is not vLLM, which caps unconditionally.
    pub(crate) fn with_long_prefill_cap_mode(mut self, mode: LongPrefillCapMode) -> Self {
        assert!(
            self.long_prefill_threshold.is_some() || mode == LongPrefillCapMode::Always,
            "a long prefill cap mode needs a long prefill threshold"
        );
        self.cap_mode = mode;
        self
    }

    /// Offer the budget to fresh prompts before started ones, keeping up to
    /// `reserve` tokens for each started prompt. Not vLLM, which runs started
    /// prompts first.
    pub(crate) fn with_fresh_first(mut self, reserve: u32) -> Self {
        assert!(
            self.slot_plan.is_none(),
            "the microbatch slot plan orders prompts itself; drop fresh-first"
        );
        self.fresh_first_reserve = Some(reserve);
        self
    }

    /// Apply the long prefill threshold only to prompts of at least
    /// `min_prompt_tokens` fresh tokens. Not vLLM.
    pub(crate) fn with_cap_min_prompt_tokens(mut self, min_prompt_tokens: u32) -> Self {
        self.cap_min_prompt_tokens = min_prompt_tokens;
        self
    }

    /// The per-request prefill cap in force now. `fresh_tokens` is the fresh
    /// prompts' share of the microbatch being formed (`None`: they come after
    /// the started prompts, so it is not known yet).
    /// `own_queued` is the request's own tokens in the queue, which a fresh
    /// prompt must not count toward its own backlog.
    fn request_prefill_cap(&self, fresh_tokens: Option<u32>, own_queued: u64) -> Option<u32> {
        let threshold = self.long_prefill_threshold?;
        match self.cap_mode {
            LongPrefillCapMode::Always => Some(threshold),
            LongPrefillCapMode::Contended => {
                (self.policy.len() + self.started_prefills.len() > 1).then_some(threshold)
            }
            LongPrefillCapMode::Alone => {
                let fresh_joined = fresh_tokens.map_or(self.policy.len() > 0, |tokens| tokens > 0);
                (!fresh_joined).then_some(threshold)
            }
            LongPrefillCapMode::Matched => Some(threshold.max(fresh_tokens.unwrap_or(0))),
            LongPrefillCapMode::Backlog => {
                let depth = self
                    .even_split
                    .expect("a backlog cap needs the even split")
                    .depth;
                // The queue as the microbatch began forming, before fresh-first
                // admitted any of it: a started prompt's own in-flight chunks
                // must not raise its cap, or it never settles back to the floor.
                let share = self
                    .queued_at_formation
                    .saturating_sub(own_queued)
                    .div_ceil(u64::from(depth))
                    .min(u64::from(self.max_batch_tokens)) as u32;
                Some(threshold.max(share))
            }
        }
    }

    /// Run microbatches at `low_tokens` while the head often runs out of work,
    /// rising to `max_batch_tokens` as the fraction of `window` it had work
    /// climbs from `busy_lo` to `busy_hi`. Not vLLM, whose budget is fixed.
    pub(crate) fn with_load_budget(
        mut self,
        low_tokens: u32,
        window: Time,
        busy_lo: f64,
        busy_hi: f64,
    ) -> Self {
        assert!(
            low_tokens > 0 && low_tokens <= self.max_batch_tokens,
            "load budget floor {low_tokens} must lie in 1..=max_batch_tokens {}",
            self.max_batch_tokens
        );
        assert!(window.0 > 0, "load budget window must be positive");
        assert!(
            (0.0..busy_hi).contains(&busy_lo) && busy_hi <= 1.0,
            "load budget needs 0 <= busy_lo < busy_hi <= 1, got {busy_lo}..{busy_hi}"
        );
        assert!(
            self.slot_plan.is_none(),
            "the microbatch slot plan sizes microbatches itself; drop the load budget"
        );
        self.load_budget = Some(LoadBudget {
            low_tokens,
            window,
            busy_lo,
            busy_hi,
            backlog: None,
            starved_since: None,
            starved: VecDeque::new(),
        });
        self
    }

    /// Raise the load budget with the prefill backlog (queued, started and in
    /// flight tokens) from `lo_tokens` to `hi_tokens` instead of the busy
    /// fraction, which a single long prompt keeps at 1 while it runs.
    pub(crate) fn with_backlog_signal(mut self, lo_tokens: u64, hi_tokens: u64) -> Self {
        assert!(lo_tokens < hi_tokens, "backlog signal needs lo < hi tokens");
        self.load_budget
            .as_mut()
            .expect("the backlog signal drives a load budget")
            .backlog = Some((lo_tokens, hi_tokens));
        self
    }

    /// Shrink each prompt's last chunks from `max_batch_tokens` down to
    /// `min_tokens`, stepping by the stage-time ratio `(depth - 2) / (depth -
    /// 1)` with a stage time of `overhead_tokens + tokens`. Not vLLM.
    pub(crate) fn with_tail_ladder(
        mut self,
        depth: u16,
        min_tokens: u32,
        overhead_tokens: u32,
    ) -> Self {
        assert!(
            self.slot_plan.is_none(),
            "the microbatch slot plan sizes chunks itself; drop the tail ladder"
        );
        self.tail_ladder = Some(TailLadder::new(
            depth,
            min_tokens,
            overhead_tokens,
            self.max_batch_tokens,
        ));
        self
    }

    /// Order prefill by remaining tokens across started and queued prompts: a
    /// queued prompt goes ahead of a started one with more left, and the reverse.
    /// The queue must offer its shortest prompt first. Not vLLM.
    pub(crate) fn with_srpt(mut self) -> Self {
        assert!(
            self.slot_plan.is_none(),
            "the microbatch slot plan orders prompts itself; drop srpt"
        );
        self.srpt = true;
        self
    }

    /// The tail ladder's cap for a prompt with `remaining` prefill tokens.
    fn tail_cap(&self, remaining: u32) -> Option<u32> {
        Some(self.tail_ladder.as_ref()?.cap(remaining))
    }

    /// Cap each microbatch's decodes at `ceil(resident decodes / depth)`.
    pub(crate) fn with_balanced_decodes(mut self, depth: u16) -> Self {
        assert!(depth > 0, "pipeline depth must be positive");
        self.balanced_depth = Some(depth);
        self
    }

    /// Bin-pack whole requests into a plan of the next `depth` microbatches.
    pub(crate) fn with_slot_plan(mut self, depth: u16) -> Self {
        assert!(
            self.chunk_end_quantum.is_none(),
            "the microbatch slot plan does not align chunk ends; use plain chunk alignment"
        );
        assert!(self.even_split.is_none(), "choose one microbatch split");
        assert!(
            self.long_prefill_threshold.is_none(),
            "the microbatch slot plan sizes chunks itself; drop the long prefill threshold"
        );
        self.slot_plan = Some(MicrobatchSlotPlan::new(depth, self.max_batch_tokens));
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
    fn prefill_budget<K: ChunkedPrefillKv>(
        &self,
        kv_store: &K,
        remaining_budget: u32,
        now: Time,
    ) -> u32 {
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
            Some(load) => {
                remaining_budget.min(load.target(now, round_tokens, self.max_batch_tokens))
            }
        };
        let Some(EvenSplit { depth, min_tokens }) = self.even_split else {
            return remaining_budget;
        };
        let target = round_tokens
            .div_ceil(u64::from(depth))
            .clamp(u64::from(min_tokens), u64::from(self.max_batch_tokens));
        remaining_budget.min(target as u32)
    }

    /// Pop the policy head, reserve its KV, and stamp it into prefill.
    fn admit_head<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        candidate: AdmissionCandidate,
        resolved_prefill: ResolvedPrefillContext,
        footprint: K::Footprint,
        now: Time,
    ) {
        let popped = self.policy.pop(&mut self.policy_context);
        debug_assert_eq!(popped, Some(candidate));
        kv_store.reserve_chunked_prefill_context(
            candidate.request_id,
            PARTITION,
            resolved_prefill,
            footprint,
            now,
        );
        let mut store = context.requests.borrow_mut();
        store.mark_admitted(candidate.request_id);
        let record = &mut store[candidate.request_id];
        record.record_prefix_cache_hit_tokens(resolved_prefill.resident_prefix_tokens());
        context.stamp_stage(record, now, UnifiedStage::Prefill as u16);
    }

    /// Place started remainders, then fresh prompts while a future slot has
    /// room and KV admits them.
    fn plan_pending_prefill<K: ChunkedPrefillKv>(
        &mut self,
        plan: &mut MicrobatchSlotPlan,
        kv_store: &mut K,
        context: &WorkerContext,
        now: Time,
    ) {
        let mut index = 0;
        while index < plan.unplanned.len() {
            let UnplannedPrefill { request, tokens } = plan.unplanned[index];
            let left = plan.place(request, tokens);
            if left == 0 {
                plan.unplanned.remove(index);
            } else {
                plan.unplanned[index].tokens = left;
                index += 1;
            }
        }
        while plan.has_room() {
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
            let tokens = resolved_prefill.remaining_prefill_tokens();
            if tokens == 0 {
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
            self.admit_head(
                kv_store,
                context,
                candidate,
                resolved_prefill,
                footprint,
                now,
            );
            let left = plan.place(candidate.request_id, tokens);
            if left > 0 {
                plan.unplanned.push_back(UnplannedPrefill {
                    request: candidate.request_id,
                    tokens: left,
                });
            }
        }
    }

    /// Schedule the head slot's chunks into KV within `budget`. A chunk the
    /// budget cuts returns, with its request's later chunks, to the front of
    /// the unplanned list.
    fn launch_planned_slot<K: ChunkedPrefillKv>(
        plan: &mut MicrobatchSlotPlan,
        kv_store: &mut K,
        mut budget: u32,
    ) {
        let head = plan.launch_head_slot();
        for chunk in head.chunks {
            let take = chunk.tokens.min(budget);
            if take > 0 {
                kv_store.schedule_prefill_chunk(chunk.request, PARTITION, take);
                budget -= take;
            }
            if take < chunk.tokens {
                let tokens = chunk.tokens - take + plan.withdraw(chunk.request);
                plan.unplanned.push_front(UnplannedPrefill {
                    request: chunk.request,
                    tokens,
                });
            }
        }
    }

    /// Admit fresh prompts into `budget`; returns the tokens they took.
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
                // A fresh prompt counts itself as joining; under `Matched` it may
                // take as much as the fresh prompts before it.
                tighter(
                    self.request_prefill_cap(
                        Some((budget - remaining_budget).max(1)),
                        u64::from(candidate.fresh_prompt_tokens),
                    )
                    .filter(|_| candidate.fresh_prompt_tokens >= self.cap_min_prompt_tokens),
                    self.tail_cap(resolved_prefill.remaining_prefill_tokens()),
                ),
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
            self.admit_head(
                kv_store,
                context,
                candidate,
                resolved_prefill,
                footprint,
                now,
            );
            kv_store.schedule_prefill_chunk(candidate.request_id, PARTITION, chunk_tokens);
            remaining_budget -= chunk_tokens;
            if chunk_tokens < resolved_prefill.remaining_prefill_tokens() {
                self.started_prefills.push(candidate);
            }
        }
        budget - remaining_budget
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
        let cap = self.request_prefill_cap(None, 0);
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
                tighter(
                    cap.filter(|_| candidate.fresh_prompt_tokens >= self.cap_min_prompt_tokens),
                    self.tail_cap(remaining),
                ),
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
        if let Some(mut plan) = self.slot_plan.take() {
            self.plan_pending_prefill(&mut plan, kv_store, context, now);
            Self::launch_planned_slot(&mut plan, kv_store, remaining_budget);
            self.slot_plan = Some(plan);
            return kv_store.has_prefill_admit(PARTITION) || !self.scheduled_decodes.is_empty();
        }
        let mut remaining_budget = self.prefill_budget(kv_store, remaining_budget, now);
        self.queued_at_formation = self.policy.queued_prompt_tokens();
        if self.srpt {
            self.form_prefill_srpt(kv_store, context, remaining_budget, now);
            let formed =
                kv_store.has_prefill_admit(PARTITION) || !self.scheduled_decodes.is_empty();
            if let Some(load) = &mut self.load_budget {
                load.note_formation(now, formed);
            }
            return formed;
        }
        // Fresh-first: each started prompt holds at most `reserve` tokens while
        // fresh prompts fill the budget first; started prompts then take what
        // is left, shortest remaining first. Prompts admitted now join after.
        let mut newly_started = Vec::new();
        let mut fresh_tokens = None;
        self.queued_at_formation = self.policy.queued_prompt_tokens();
        if let Some(reserve) = self.fresh_first_reserve {
            let held = self
                .started_prefills
                .iter()
                .map(|candidate| {
                    kv_store
                        .resolved_prefill_context(candidate.request_id)
                        .remaining_prefill_tokens()
                        .min(reserve)
                })
                .fold(0_u32, u32::saturating_add)
                .min(remaining_budget);
            // Shortest remaining first among started prompts, so a short
            // prompt's tail does not queue behind a long prompt.
            self.started_prefills.sort_by_key(|candidate| {
                kv_store
                    .resolved_prefill_context(candidate.request_id)
                    .remaining_prefill_tokens()
            });
            let started = std::mem::take(&mut self.started_prefills);
            let taken =
                self.admit_fresh_prompts(kv_store, context, remaining_budget - held, now, None);
            remaining_budget -= taken;
            fresh_tokens = Some(taken);
            newly_started = std::mem::replace(&mut self.started_prefills, started);
        }
        let long_prefill_threshold = self.request_prefill_cap(fresh_tokens, 0);
        let cap_min_prompt_tokens = self.cap_min_prompt_tokens;
        let tail_ladder = self.tail_ladder.as_ref();
        // Started prompts keep start order. One that cannot run keeps its place.
        self.started_prefills.retain(|candidate| {
            let resolved_prefill = kv_store.resolved_prefill_context(candidate.request_id);
            let chunk_tokens = next_chunk_tokens(
                resolved_prefill,
                remaining_budget,
                chunk_end_quantum,
                max_batch_tokens,
                tighter(
                    long_prefill_threshold
                        .filter(|_| candidate.fresh_prompt_tokens >= cap_min_prompt_tokens),
                    tail_ladder
                        .map(|ladder| ladder.cap(resolved_prefill.remaining_prefill_tokens())),
                ),
            );
            if chunk_tokens == 0 {
                return true;
            }
            kv_store.schedule_prefill_chunk(candidate.request_id, PARTITION, chunk_tokens);
            remaining_budget -= chunk_tokens;
            chunk_tokens < resolved_prefill.remaining_prefill_tokens()
        });
        if self.fresh_first_reserve.is_some() {
            self.started_prefills.extend(newly_started);
        } else {
            self.admit_fresh_prompts(kv_store, context, remaining_budget, now, None);
        }
        let formed = kv_store.has_prefill_admit(PARTITION) || !self.scheduled_decodes.is_empty();
        if let Some(load) = &mut self.load_budget {
            load.note_formation(now, formed);
        }
        formed
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
}
