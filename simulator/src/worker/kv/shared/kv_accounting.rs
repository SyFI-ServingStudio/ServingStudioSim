//! KV charges of one request, resolved from the arch's [`KvCacheLayout`].
//!
//! The store never sees layout parts. It asks this module three questions, all
//! answered in **units** of `bytes_per_unit` (the gcd of every part's entry
//! bytes, so a layout with a single per-token part counts plain tokens):
//!
//! - [`KvAccounting::held_while_running`]: what a request with `len` context
//!   tokens holds while a step processes `step` of them;
//! - [`KvAccounting::needed_to_admit`]: what a waiting request needs free before
//!   it starts (vLLM gates on the whole prompt);
//! - [`KvAccounting::entry_charge`]: what a finished request's retained prefix
//!   keeps, given the positions a later request can resume from.
//!
//! [`KvAccounting::concurrency_at`] answers a fourth, for the report only:
//! vLLM's logged "Maximum concurrency", whose estimate charges a recurrent
//! state one more copy than a running request holds (see
//! [`state_copies`]).
//!
//! `kv_overhead` picks the byte counts. `vllm` charges what vLLM allocates:
//! padded entries, extra state copies, whole engine blocks, and a sliding window
//! that holds `W - 1` tokens plus the tokens in flight. `none` charges only what
//! the model itself must keep.
//!
//! [`KvAccounting::report`] restates one resolved pool as a
//! [`KvCacheReport`](super::kv_cache_report::KvCacheReport), built once per
//! pool at construction.

use std::sync::Arc;

use anyhow::ensure;
use serde::{Deserialize, Serialize};

use crate::arch::kv_layout::{KvCacheLayout, KvCachePart, KvGrowth};

/// Whether KV charges include what the serving engine allocates on top of the
/// model's own state.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KvOverhead {
    /// What vLLM allocates: padding, state copies, engine blocks, in-flight
    /// sliding-window tokens. Its prefix cache resumes at the prompt end.
    #[default]
    Vllm,
    /// Only the state the model must keep. An upper bound on capacity and
    /// prefix reuse; the gap to `vllm` is the engine's overhead.
    None,
}

impl KvOverhead {
    pub const CHOICES: &'static [&'static str] = &["vllm", "none"];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
            Self::None => "none",
        }
    }

    /// The other mode, which a report compares against.
    pub(crate) fn other(self) -> Self {
        match self {
            Self::Vllm => Self::None,
            Self::None => Self::Vllm,
        }
    }
}

/// Positions a finished request keeps sliding-window tails and recurrent state
/// at, so that a later request of the same session can resume there. A layout
/// with only per-token parts resumes anywhere and ignores this.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PrefixResumePoints {
    /// The end of the request's own prompt (vLLM's
    /// `prefix_cache_retention_interval = 0`): a resend or a sibling resumes,
    /// a next turn replays the previous answer.
    PromptEnd,
    /// The end of the request's response: a next turn that keeps the answer in
    /// its history resumes after it.
    ResponseEnd,
    PromptAndResponseEnd,
    /// Every engine block (vLLM's `prefix_cache_retention_interval = None`).
    EveryBlock,
}

impl PrefixResumePoints {
    pub const CHOICES: &'static [&'static str] = &[
        "prompt_end",
        "response_end",
        "prompt_and_response_end",
        "every_block",
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::PromptEnd => "prompt_end",
            Self::ResponseEnd => "response_end",
            Self::PromptAndResponseEnd => "prompt_and_response_end",
            Self::EveryBlock => "every_block",
        }
    }
}

/// Worker options that shape the charges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvAccountingOptions {
    pub overhead: KvOverhead,
    /// `None` follows the overhead mode: `prompt_end` under `vllm`,
    /// `response_end` under `none`.
    pub resume_points: Option<PrefixResumePoints>,
    /// Also resume at every multiple of this many tokens.
    pub resume_interval: Option<u32>,
    /// Tokens one step may have in flight across the batch (vLLM:
    /// `max_concurrent_batches × max_num_batched_tokens`). It caps what
    /// admission charges a sliding window; `None` charges the whole prompt.
    pub max_in_flight_tokens: Option<u32>,
    /// Whether finished requests are retained for prefix reuse. vLLM then runs
    /// a recurrent state in align mode, whose concurrency estimate counts the
    /// previous step's state block beside the running one: one more copy per
    /// request in [`KvAccounting::concurrency_at`], none in what a running
    /// request holds.
    pub prefix_caching: bool,
}

/// One part's share of [`KvAccounting::estimated_request_units`] (vLLM's concurrency estimate).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PartInEstimate {
    /// Context tokens the part holds: the block-rounded prompt for a per-token
    /// part, the kept window for a sliding window; `None` for per-request state.
    pub tokens_held: Option<u64>,
    /// Entries charged: tokens over `tokens_per_entry`, window tokens, or state
    /// copies.
    pub entries: u64,
    pub units: u64,
}

#[derive(Clone, Copy, Debug)]
struct PerTokenCharge {
    units_per_entry: u64,
    tokens_per_entry: u64,
}

#[derive(Clone, Copy, Debug)]
struct WindowCharge {
    units_per_token: u64,
    window_tokens: u64,
    prefix_reusable: bool,
}

/// Resolved charges of one KV partition. Built once per worker.
#[derive(Clone, Debug)]
pub(crate) struct KvAccounting {
    /// What it was resolved from, kept only for [`Self::report`].
    layout: KvCacheLayout,
    options: KvAccountingOptions,
    budget_bytes: u64,
    /// GPUs one partition's budget spans, kept only for [`Self::report`].
    attention_shards: u16,
    /// Units [`Self::reserving`] set aside out of the resolved capacity.
    reserved_units: u64,
    overhead: KvOverhead,
    bytes_per_unit: u64,
    capacity_units: u64,
    per_token: Vec<PerTokenCharge>,
    windows: Vec<WindowCharge>,
    /// All per-request state, with the copies a running request holds.
    state_units: u64,
    /// The align copies vLLM's concurrency estimate adds on top.
    align_estimate_units: u64,
    /// One copy of the per-request state a later request can resume from.
    reusable_state_units: u64,
    /// Allocation and resume granularity; 1 when blocks are not modeled.
    block_tokens: u64,
    /// The layout's engine block, whatever the mode: the boundary a recurrent
    /// state can be checkpointed at when prefill chunks align to it.
    checkpoint_block_tokens: Option<u32>,
    max_in_flight_tokens: Option<u64>,
    /// `Some(units per token)` when every charge is linear in context.
    linear_units_per_token: Option<u64>,
    resume_points: PrefixResumePoints,
    resume_interval: Option<u64>,
}

pub(crate) type SharedKvAccounting = Arc<KvAccounting>;

impl KvAccounting {
    /// Plain token accounting: one unit per token, `capacity_tokens` of them.
    #[cfg(test)]
    pub(crate) fn tokens_for_test(capacity_tokens: u64) -> SharedKvAccounting {
        let layout =
            KvCacheLayout::new().with_part(crate::arch::kv_layout::KvCachePart::per_token("kv", 1));
        Arc::new(Self::resolve(&layout, KvAccountingOptions::default(), capacity_tokens).unwrap())
    }

    /// Resolve `layout` against a partition budget of `budget_bytes` (the
    /// per-GPU KV budget times the partition's ranks).
    pub(crate) fn resolve(
        layout: &KvCacheLayout,
        options: KvAccountingOptions,
        budget_bytes: u64,
    ) -> anyhow::Result<Self> {
        layout.validate()?;
        let with_overhead = options.overhead == KvOverhead::Vllm;
        let bytes_per_unit = layout
            .parts
            .iter()
            .map(|part| charged_bytes_per_entry(part, options.overhead))
            .fold(0, gcd);

        let mut per_token = Vec::new();
        let mut windows = Vec::new();
        let mut state_units = 0;
        let mut align_estimate_units = 0;
        let mut reusable_state_units = 0;
        for part in &layout.parts {
            let units = charged_bytes_per_entry(part, options.overhead) / bytes_per_unit;
            match part.growth {
                KvGrowth::PerToken { tokens_per_entry } => per_token.push(PerTokenCharge {
                    units_per_entry: units,
                    tokens_per_entry: u64::from(tokens_per_entry),
                }),
                KvGrowth::SlidingWindow { window_tokens } => windows.push(WindowCharge {
                    units_per_token: units,
                    window_tokens: u64::from(window_tokens),
                    prefix_reusable: part.prefix_reusable,
                }),
                KvGrowth::PerRequest => {
                    let (copies, aligns) = state_copies(part, layout, options);
                    state_units += units * u64::from(copies);
                    align_estimate_units += units * u64::from(aligns);
                    if part.prefix_reusable {
                        reusable_state_units += units;
                    }
                }
            }
        }

        let block_tokens = if with_overhead {
            u64::from(layout.engine.block_tokens.unwrap_or(1))
        } else {
            1
        };
        let fixed_bytes = if with_overhead {
            layout.engine.fixed_device_bytes
        } else {
            0
        };
        let capacity_units = (budget_bytes.saturating_sub(fixed_bytes) / bytes_per_unit).max(1);
        let linear_units_per_token = (windows.is_empty()
            && block_tokens == 1
            && per_token.iter().all(|part| part.tokens_per_entry == 1))
        .then(|| per_token.iter().map(|part| part.units_per_entry).sum());

        let resume_points = options.resume_points.unwrap_or(match options.overhead {
            KvOverhead::Vllm => PrefixResumePoints::PromptEnd,
            KvOverhead::None => PrefixResumePoints::ResponseEnd,
        });
        let resume_interval = options.resume_interval.map(u64::from);
        if let Some(interval) = resume_interval {
            ensure!(
                interval > 0 && interval % block_tokens == 0,
                "prefix_resume_interval {interval} must be a positive multiple of the engine block ({block_tokens} tokens)"
            );
        }

        Ok(Self {
            layout: layout.clone(),
            options,
            budget_bytes,
            attention_shards: 1,
            reserved_units: 0,
            overhead: options.overhead,
            bytes_per_unit,
            capacity_units,
            per_token,
            windows,
            state_units,
            align_estimate_units,
            reusable_state_units,
            block_tokens,
            checkpoint_block_tokens: layout.engine.block_tokens,
            max_in_flight_tokens: options.max_in_flight_tokens.map(u64::from),
            linear_units_per_token,
            resume_points,
            resume_interval,
        })
    }

    pub(crate) fn bytes_per_unit(&self) -> u64 {
        self.bytes_per_unit
    }

    pub(crate) fn capacity_units(&self) -> u64 {
        self.capacity_units
    }

    /// Requests of `max_model_len` tokens the pool admits at once, as vLLM's
    /// "Maximum concurrency for N tokens per request" log line counts them:
    /// the admission charge plus the align copies its estimate adds.
    pub(crate) fn concurrency_at(&self, max_model_len: u64) -> f64 {
        self.capacity_units as f64 / self.estimated_request_units(max_model_len) as f64
    }

    /// Units vLLM's concurrency estimate charges a `prompt_tokens` request.
    pub(super) fn estimated_request_units(&self, prompt_tokens: u64) -> u64 {
        self.needed_to_admit(prompt_tokens) + self.align_estimate_units
    }

    /// The same accounting with `units` of capacity set aside, as PD decode
    /// does for its pull backlog.
    pub(crate) fn reserving(&self, units: u64) -> Self {
        Self {
            capacity_units: self.capacity_units.saturating_sub(units).max(1),
            reserved_units: self.reserved_units + units,
            ..self.clone()
        }
    }

    /// The same accounting, recorded as spanning `attention_shards` GPUs: the
    /// report divides the partition's bytes by them. Charges do not change.
    pub(crate) fn over_attention_shards(self, attention_shards: u16) -> Self {
        Self {
            attention_shards: attention_shards.max(1),
            ..self
        }
    }

    pub(super) fn attention_shards(&self) -> u16 {
        self.attention_shards
    }

    pub(super) fn layout(&self) -> &KvCacheLayout {
        &self.layout
    }

    pub(super) fn options(&self) -> KvAccountingOptions {
        self.options
    }

    pub(super) fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    pub(super) fn reserved_units(&self) -> u64 {
        self.reserved_units
    }

    /// Allocation and resume granularity; 1 when blocks are not modeled.
    pub(super) fn block_tokens(&self) -> u64 {
        self.block_tokens
    }

    pub(super) fn resume_points(&self) -> PrefixResumePoints {
        self.resume_points
    }

    /// The longest context whose running charge fits `units`.
    pub(crate) fn tokens_fitting(&self, units: u64) -> u64 {
        let (mut low, mut high) = (0u64, units.saturating_mul(self.block_tokens).max(1));
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            if self.held_while_running(mid, 1) <= units {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        low
    }

    pub(crate) fn overhead(&self) -> KvOverhead {
        self.overhead
    }

    /// Units of `bytes` of retained-prefix budget (a preset's byte ceiling).
    pub(crate) fn units_of_bytes(&self, bytes: u64) -> u64 {
        bytes / self.bytes_per_unit
    }

    /// The boundary prefill chunks end on so a recurrent state can be
    /// checkpointed there, when the layout has a resumable per-request state
    /// and an engine block.
    pub(crate) fn checkpoint_block_tokens(&self) -> Option<u32> {
        (self.reusable_state_units > 0)
            .then_some(self.checkpoint_block_tokens)
            .flatten()
    }

    /// Whether the overhead mode aligns prefill chunks to checkpoints when the
    /// preset leaves it open: vLLM does, the model alone does not need to.
    pub(crate) fn aligns_chunks_by_default(&self) -> bool {
        self.overhead == KvOverhead::Vllm
    }

    fn per_token_units(&self, len: u64) -> u64 {
        if len == 0 {
            return 0;
        }
        let allocated = len.div_ceil(self.block_tokens) * self.block_tokens;
        self.per_token
            .iter()
            .map(|part| part.units_per_entry * allocated.div_ceil(part.tokens_per_entry))
            .sum()
    }

    fn window_units(&self, len: u64, kept: impl Fn(u64) -> u64) -> u64 {
        self.windows
            .iter()
            .map(|part| part.units_per_token * len.min(kept(part.window_tokens)))
            .sum()
    }

    /// Units a request with `len` context tokens holds while a step processes
    /// `step_tokens` of them. vLLM keeps a window's `W - 1` previous tokens plus
    /// the step's own; the model alone needs only `W`.
    pub(crate) fn held_while_running(&self, len: u64, step_tokens: u64) -> u64 {
        if let Some(units_per_token) = self.linear_units_per_token {
            return len * units_per_token + self.state_units;
        }
        let window = match self.overhead {
            KvOverhead::Vllm => self.window_units(len, |w| w - 1 + step_tokens.max(1)),
            KvOverhead::None => self.window_units(len, |w| w),
        };
        self.per_token_units(len) + window + self.state_units
    }

    /// Units a waiting request with a `prompt_tokens` prompt needs free before
    /// it starts: vLLM's whole-prompt gate, with each window capped at what the
    /// step's in-flight tokens can make it hold.
    pub(crate) fn needed_to_admit(&self, prompt_tokens: u64) -> u64 {
        if let Some(units_per_token) = self.linear_units_per_token {
            return prompt_tokens * units_per_token + self.state_units;
        }
        let window = match self.overhead {
            KvOverhead::Vllm => {
                let in_flight = self.max_in_flight_tokens.unwrap_or(prompt_tokens);
                self.window_units(prompt_tokens, |w| w - 1 + in_flight)
            }
            KvOverhead::None => self.window_units(prompt_tokens, |w| w),
        };
        self.per_token_units(prompt_tokens) + window + self.state_units
    }

    /// [`Self::estimated_request_units`] split over the layout's parts, in
    /// their order: what each part holds for the request and its units.
    /// Built for the report only; the parts sum to `estimated_request_units`.
    pub(super) fn estimated_request_by_part(&self, prompt_tokens: u64) -> Vec<PartInEstimate> {
        let in_flight = self.max_in_flight_tokens.unwrap_or(prompt_tokens);
        let allocated = prompt_tokens.div_ceil(self.block_tokens) * self.block_tokens;
        self.layout
            .parts
            .iter()
            .map(|part| {
                let units_per_entry =
                    charged_bytes_per_entry(part, self.overhead) / self.bytes_per_unit;
                let (tokens_held, entries) = match part.growth {
                    KvGrowth::PerToken { tokens_per_entry } => {
                        let tokens = if prompt_tokens == 0 { 0 } else { allocated };
                        (Some(tokens), tokens.div_ceil(u64::from(tokens_per_entry)))
                    }
                    KvGrowth::SlidingWindow { window_tokens } => {
                        let window = u64::from(window_tokens);
                        let kept = match self.overhead {
                            KvOverhead::Vllm => window - 1 + in_flight,
                            KvOverhead::None => window,
                        };
                        let tokens = prompt_tokens.min(kept);
                        (Some(tokens), tokens)
                    }
                    KvGrowth::PerRequest => {
                        let (copies, aligns) = state_copies(part, &self.layout, self.options);
                        (None, u64::from(copies + u32::from(aligns)))
                    }
                };
                PartInEstimate {
                    tokens_held,
                    entries,
                    units: units_per_entry * entries,
                }
            })
            .collect()
    }

    /// What admission reserves for a request reaching `context_tokens` after
    /// prefill and `future_tokens` more by the end of the reservation.
    pub(crate) fn footprint(&self, context_tokens: u64, future_tokens: u64) -> u64 {
        self.held_while_running(context_tokens + future_tokens, 1)
            .max(self.needed_to_admit(context_tokens))
    }

    /// Units `tokens` more decode tokens add to a request at `len`.
    pub(crate) fn growth_over(&self, len: u64, tokens: u64) -> u64 {
        if let Some(units_per_token) = self.linear_units_per_token {
            return tokens * units_per_token;
        }
        self.held_while_running(len + tokens, 1) - self.held_while_running(len, 1)
    }

    /// Whether every charge grows linearly with context.
    pub(crate) fn linear_units_per_token(&self) -> Option<u64> {
        self.linear_units_per_token
    }

    /// Bytes of a request at `len` context tokens, as a KV transfer moves them.
    pub(crate) fn transfer_bytes(&self, len: u64) -> u64 {
        self.held_while_running(len, 1) * self.bytes_per_unit
    }

    fn sparse(&self) -> bool {
        self.reusable_state_units > 0 || self.windows.iter().any(|part| part.prefix_reusable)
    }

    /// Whether a running request keeps soft checkpoints
    /// ([`Self::checkpoint_units`]) and retained entries resume only at chosen
    /// positions: some reusable part does not grow per token.
    pub(crate) fn keeps_checkpoints(&self) -> bool {
        self.sparse()
    }

    /// Positions a request with a `prompt_tokens` prompt that finished at
    /// `context_tokens` can be resumed from.
    pub(crate) fn resume_positions(
        &self,
        prompt_tokens: u64,
        context_tokens: u64,
    ) -> ResumePositions {
        let block = self.block_tokens;
        let floor = |tokens: u64| tokens / block * block;
        if !self.sparse() {
            return ResumePositions::every(block, floor(context_tokens));
        }
        // vLLM recomputes a resend's last prompt token, so its replay boundary
        // is the prompt minus one, floored to a block.
        let prompt_end = floor(match self.overhead {
            KvOverhead::Vllm => prompt_tokens.saturating_sub(1),
            KvOverhead::None => prompt_tokens,
        });
        let response_end = floor(context_tokens);
        let (explicit, every) = match self.resume_points {
            PrefixResumePoints::PromptEnd => ([Some(prompt_end), None], self.resume_interval),
            PrefixResumePoints::ResponseEnd => ([Some(response_end), None], self.resume_interval),
            PrefixResumePoints::PromptAndResponseEnd => {
                ([Some(prompt_end), Some(response_end)], self.resume_interval)
            }
            PrefixResumePoints::EveryBlock => ([None, None], Some(block)),
        };
        ResumePositions::new(explicit, every, context_tokens)
    }

    /// Units a retained entry occupies: the per-token parts up to its last
    /// position, each reusable window's tails before every position (counted
    /// once where they overlap), and one reusable state per position.
    pub(crate) fn entry_charge(&self, positions: &ResumePositions) -> u64 {
        let mut units = self.per_token_units(positions.end);
        if !self.sparse() {
            return units;
        }
        for part in self.windows.iter().filter(|part| part.prefix_reusable) {
            units += part.units_per_token * positions.window_tail_union(part.window_tokens - 1);
        }
        units + self.reusable_state_units * positions.count()
    }

    /// Soft units a running request at `len` already keeps for its future
    /// entry: tails and states at positions it has passed. Like a retained
    /// prefix they are evictable, so they never gate admission.
    pub(crate) fn checkpoint_units(&self, prompt_tokens: u64, len: u64) -> u64 {
        if !self.sparse() || len == 0 {
            return 0;
        }
        let passed = self.resume_positions(prompt_tokens, len).truncated(len - 1);
        self.entry_charge(&passed) - self.per_token_units(passed.end)
    }

    /// The longest prefix of `positions` whose entry fits `capacity_units`.
    pub(crate) fn largest_fitting(
        &self,
        positions: &ResumePositions,
        capacity_units: u64,
    ) -> ResumePositions {
        if self.entry_charge(positions) <= capacity_units {
            return *positions;
        }
        // Charge is monotone in the truncation point: binary-search the largest
        // context length whose truncated entry still fits.
        let (mut low, mut high) = (0u64, positions.end);
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            let candidate = positions.truncated(positions.floor(mid));
            if self.entry_charge(&candidate) <= capacity_units {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        positions.truncated(positions.floor(low))
    }
}

/// Where a retained entry can be resumed: up to two explicit positions plus,
/// optionally, every multiple of a spacing, all at or below `end`. `end` is
/// the last position and the context the per-token parts keep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResumePositions {
    end: u64,
    explicit: [Option<u64>; 2],
    every: Option<u64>,
}

impl ResumePositions {
    fn every(spacing: u64, end: u64) -> Self {
        Self {
            end,
            explicit: [None, None],
            every: Some(spacing),
        }
    }

    fn new(explicit: [Option<u64>; 2], every: Option<u64>, context_tokens: u64) -> Self {
        let mut points: Vec<u64> = explicit
            .into_iter()
            .flatten()
            .filter(|&point| point > 0 && point <= context_tokens)
            .collect();
        points.sort_unstable();
        points.dedup();
        let series_end = every.map_or(0, |spacing| context_tokens / spacing * spacing);
        let end = points.last().copied().unwrap_or(0).max(series_end);
        Self {
            end,
            explicit: [points.first().copied(), points.get(1).copied()],
            every,
        }
    }

    /// Context tokens the entry keeps; also its last position.
    pub(crate) fn end(&self) -> u64 {
        self.end
    }

    /// The largest position at or below `tokens`, or 0.
    pub(crate) fn floor(&self, tokens: u64) -> u64 {
        let limit = tokens.min(self.end);
        let series = self.every.map_or(0, |spacing| limit / spacing * spacing);
        self.explicit
            .into_iter()
            .flatten()
            .filter(|&point| point <= limit)
            .fold(series, u64::max)
    }

    /// The same positions, cut at `last` (itself a position or 0).
    pub(crate) fn truncated(&self, last: u64) -> Self {
        let keep = |point: Option<u64>| point.filter(|&p| p <= last);
        Self {
            end: last.min(self.end),
            explicit: [keep(self.explicit[0]), keep(self.explicit[1])],
            every: self.every,
        }
    }

    fn series_count(&self) -> u64 {
        self.every.map_or(0, |spacing| self.end / spacing)
    }

    fn on_series(&self, point: u64) -> bool {
        self.every.is_some_and(|spacing| point % spacing == 0)
    }

    fn off_series(&self) -> impl Iterator<Item = u64> + '_ {
        self.explicit
            .into_iter()
            .flatten()
            .filter(|&point| !self.on_series(point))
    }

    /// Number of distinct positions.
    pub(crate) fn count(&self) -> u64 {
        self.series_count() + self.off_series().count() as u64
    }

    /// Tokens covered by the `tail`-token windows ending at each position,
    /// counted once where windows overlap.
    pub(crate) fn window_tail_union(&self, tail: u64) -> u64 {
        let gap_cover = |from: u64, to: u64| (to - from).min(tail);
        let Some(spacing) = self.every else {
            let mut previous = 0;
            let mut covered = 0;
            for point in self.off_series() {
                covered += gap_cover(previous, point);
                previous = point;
            }
            return covered;
        };
        // The series alone covers `min(tail, spacing)` before each of its
        // points; an explicit point inside a series gap splits that gap.
        let mut covered = self.series_count() * spacing.min(tail);
        let off: Vec<u64> = self.off_series().collect();
        let mut index = 0;
        while index < off.len() {
            let gap_start = off[index] / spacing * spacing;
            let gap_end = gap_start + spacing;
            let closes = gap_end <= self.end;
            let mut previous = gap_start;
            let mut split = 0;
            while index < off.len() && off[index] < gap_end {
                split += gap_cover(previous, off[index]);
                previous = off[index];
                index += 1;
            }
            if closes {
                split += gap_cover(previous, gap_end);
                covered -= spacing.min(tail);
            }
            covered += split;
        }
        covered
    }
}

/// Bytes one entry of `part` is charged: the engine's padded entry under
/// `vllm`, the model's own under `none`.
pub(super) fn charged_bytes_per_entry(part: &KvCachePart, overhead: KvOverhead) -> u64 {
    match overhead {
        KvOverhead::Vllm => part
            .overhead
            .padded_bytes_per_entry
            .unwrap_or(part.bytes_per_entry),
        KvOverhead::None => part.bytes_per_entry,
    }
}

/// Copies of a per-request `part` a running request holds, and whether vLLM's
/// concurrency estimate adds align mode's previous-step copy on top: under
/// `vllm`, the declared copies, plus that one when prefix caching is on, the
/// state is reusable and the engine has a block; under `none`, one copy.
///
/// vLLM's `MambaSpec.max_memory_usage_bytes` charges align mode
/// `2 + num_speculative_blocks` pages, but `MambaManager` keeps `1 +
/// num_speculative_blocks` per running request: blocks before the running
/// state are null, and the previous step's state block is freed the step
/// after a block boundary is crossed (`remove_skipped_blocks`). That one-step
/// copy is left out of what a request holds.
pub(super) fn state_copies(
    part: &KvCachePart,
    layout: &KvCacheLayout,
    options: KvAccountingOptions,
) -> (u32, bool) {
    if options.overhead == KvOverhead::None {
        return (1, false);
    }
    let aligns =
        options.prefix_caching && part.prefix_reusable && layout.engine.block_tokens.is_some();
    (part.overhead.copies.unwrap_or(1), aligns)
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::kv_layout::{EngineOverhead, KvCachePart};

    fn options(overhead: KvOverhead) -> KvAccountingOptions {
        KvAccountingOptions {
            overhead,
            ..KvAccountingOptions::default()
        }
    }

    fn full_only(bytes_per_token: u64) -> KvCacheLayout {
        KvCacheLayout::new().with_part(KvCachePart::per_token("kv", bytes_per_token))
    }

    #[test]
    fn a_single_per_token_part_counts_plain_tokens() {
        let accounting =
            KvAccounting::resolve(&full_only(1_000), options(KvOverhead::Vllm), 80_000_500)
                .unwrap();
        assert_eq!(accounting.bytes_per_unit(), 1_000);
        assert_eq!(accounting.capacity_units(), 80_000);
        assert_eq!(accounting.held_while_running(1_234, 64), 1_234);
        assert_eq!(accounting.needed_to_admit(1_234), 1_234);
        assert_eq!(accounting.footprint(100, 50), 150);
        assert_eq!(accounting.growth_over(100, 7), 7);
        assert_eq!(accounting.transfer_bytes(10), 10_000);
        let positions = accounting.resume_positions(80, 120);
        assert_eq!(positions.floor(100), 100, "any length resumes");
        assert_eq!(accounting.entry_charge(&positions), 120);
        assert_eq!(accounting.checkpoint_units(80, 120), 0);
    }

    /// Inkling-like: one global part, a 512-token SWA part and a 4-token conv
    /// window, all with the same per-token bytes for easy arithmetic.
    fn swa_layout() -> KvCacheLayout {
        KvCacheLayout::new()
            .with_part(KvCachePart::per_token("global", 10))
            .with_part(KvCachePart::sliding_window("swa", 512, 10))
            .with_part(KvCachePart::sliding_window("sconv", 4, 20).padded_to(30))
    }

    #[test]
    fn sliding_windows_hold_their_window_plus_the_step_under_vllm() {
        let vllm = KvAccounting::resolve(
            &swa_layout(),
            KvAccountingOptions {
                max_in_flight_tokens: Some(4_096),
                ..options(KvOverhead::Vllm)
            },
            1 << 30,
        )
        .unwrap();
        // Padded conv: unit gcd(10, 10, 30) = 10 bytes; conv is 3 units/token.
        assert_eq!(vllm.bytes_per_unit(), 10);
        // Decode at 10_000 tokens: global 10_000, swa 512, conv 4 × 3.
        assert_eq!(vllm.held_while_running(10_000, 1), 10_000 + 512 + 12);
        // A 2048-token chunk holds W - 1 + 2048 of each window.
        assert_eq!(
            vllm.held_while_running(10_000, 2_048),
            10_000 + 2_559 + 3 * 2_051
        );
        // Admission caps each window at W - 1 + in-flight tokens.
        assert_eq!(vllm.needed_to_admit(100_000), 100_000 + 4_607 + 3 * 4_099);
        assert_eq!(vllm.growth_over(10_000, 5), 5);
        assert_eq!(vllm.growth_over(100, 5), 5 + 5);

        let none =
            KvAccounting::resolve(&swa_layout(), options(KvOverhead::None), 1 << 30).unwrap();
        // Unpadded conv: 2 units/token, window W only.
        assert_eq!(none.held_while_running(10_000, 2_048), 10_000 + 512 + 8);
        assert_eq!(none.needed_to_admit(100_000), 100_000 + 512 + 8);
    }

    #[test]
    fn admission_without_a_chunk_cap_charges_the_whole_prompt_window() {
        let accounting =
            KvAccounting::resolve(&swa_layout(), options(KvOverhead::Vllm), 1 << 30).unwrap();
        assert_eq!(accounting.needed_to_admit(1_000), 1_000 + 1_000 + 3 * 1_000);
    }

    /// A GDN-like hybrid: per-token attention plus one recurrent state, with a
    /// mamba-aligned 1056-token engine block.
    fn gdn_layout() -> KvCacheLayout {
        KvCacheLayout::new()
            .with_part(KvCachePart::per_token("gqa", 100))
            .with_part(KvCachePart::per_request("gdn", 300_000).padded_to(316_800))
            .with_engine(EngineOverhead {
                block_tokens: Some(1_056),
                fixed_device_bytes: 0,
            })
    }

    #[test]
    fn per_request_state_pays_its_copies_and_blocks_round_up_under_vllm() {
        let cached = |layout: &KvCacheLayout| {
            KvAccounting::resolve(
                layout,
                KvAccountingOptions {
                    prefix_caching: true,
                    ..options(KvOverhead::Vllm)
                },
                1 << 40,
            )
            .unwrap()
        };
        let vllm = cached(&gdn_layout());
        assert_eq!(vllm.bytes_per_unit(), 100);
        // 2000 tokens occupy two 1056-token blocks; a running request keeps
        // the 3168-unit state once. Align mode's previous-step copy enters
        // only vLLM's concurrency estimate.
        assert_eq!(vllm.held_while_running(2_000, 1), 2_112 + 3_168);
        assert_eq!(vllm.needed_to_admit(2_000), 2_112 + 3_168);
        assert_eq!(vllm.estimated_request_units(2_000), 2_112 + 2 * 3_168);
        let no_prefix_cache =
            KvAccounting::resolve(&gdn_layout(), options(KvOverhead::Vllm), 1 << 40).unwrap();
        assert_eq!(no_prefix_cache.held_while_running(2_000, 1), 2_112 + 3_168);
        assert_eq!(
            no_prefix_cache.estimated_request_units(2_000),
            2_112 + 3_168
        );
        // A speculative block per draft is held; the estimate adds the align copy.
        let mut speculative = gdn_layout();
        speculative.parts[1] = speculative.parts[1].with_copies(2);
        assert_eq!(
            cached(&speculative).held_while_running(2_000, 1),
            2_112 + 2 * 3_168
        );
        assert_eq!(
            cached(&speculative).estimated_request_units(2_000),
            2_112 + 3 * 3_168
        );
        assert_eq!(
            vllm.growth_over(2_000, 100),
            0,
            "still inside the second block"
        );
        assert_eq!(vllm.growth_over(2_000, 200), 1_056);
        assert_eq!(vllm.checkpoint_block_tokens(), Some(1_056));

        let none =
            KvAccounting::resolve(&gdn_layout(), options(KvOverhead::None), 1 << 40).unwrap();
        assert_eq!(none.held_while_running(2_000, 1), 2_000 + 3_000);
        assert_eq!(none.linear_units_per_token(), Some(1));
    }

    /// The proposal's example: turn 1 prompt 8000 + 500 output, turn 2
    /// declares 8500 shared tokens.
    #[test]
    fn resume_points_follow_the_policy() {
        let resolve = |layout: &KvCacheLayout, overhead, points| {
            KvAccounting::resolve(
                layout,
                KvAccountingOptions {
                    overhead,
                    resume_points: points,
                    ..KvAccountingOptions::default()
                },
                1 << 40,
            )
            .unwrap()
        };
        let hit =
            |accounting: &KvAccounting| accounting.resume_positions(8_000, 8_500).floor(8_500);

        assert_eq!(hit(&resolve(&full_only(8), KvOverhead::Vllm, None)), 8_500);
        assert_eq!(hit(&resolve(&swa_layout(), KvOverhead::Vllm, None)), 7_999);
        assert_eq!(hit(&resolve(&gdn_layout(), KvOverhead::Vllm, None)), 7_392);
        assert_eq!(hit(&resolve(&swa_layout(), KvOverhead::None, None)), 8_500);
        assert_eq!(hit(&resolve(&gdn_layout(), KvOverhead::None, None)), 8_500);
        assert_eq!(
            hit(&resolve(
                &swa_layout(),
                KvOverhead::Vllm,
                Some(PrefixResumePoints::ResponseEnd)
            )),
            8_500
        );
        assert_eq!(
            hit(&resolve(
                &gdn_layout(),
                KvOverhead::Vllm,
                Some(PrefixResumePoints::EveryBlock)
            )),
            8_448
        );
        // A history edited below the only position resumes nowhere.
        let none = resolve(&gdn_layout(), KvOverhead::None, None);
        assert_eq!(none.resume_positions(8_000, 8_500).floor(8_400), 0);
    }

    #[test]
    fn entry_charge_counts_tails_once_and_one_state_per_position() {
        let accounting = KvAccounting::resolve(
            &swa_layout(),
            KvAccountingOptions {
                resume_points: Some(PrefixResumePoints::PromptAndResponseEnd),
                ..options(KvOverhead::Vllm)
            },
            1 << 40,
        )
        .unwrap();
        // Positions 7999 and 8500: the swa tails (511 each) overlap by 10 tokens.
        let positions = accounting.resume_positions(8_000, 8_500);
        assert_eq!(positions.count(), 2);
        assert_eq!(positions.window_tail_union(511), 511 + 501);
        assert_eq!(
            accounting.entry_charge(&positions),
            8_500 + (511 + 501) + 3 * (3 + 3)
        );
        // Truncating to fit drops the later position first.
        let fitted = accounting.largest_fitting(&positions, 8_600);
        assert_eq!(fitted.end(), 7_999);
        assert_eq!(accounting.entry_charge(&fitted), 7_999 + 511 + 9);
    }

    #[test]
    fn a_periodic_series_with_explicit_points_counts_each_gap_once() {
        let positions = ResumePositions::new([Some(250), Some(410)], Some(100), 420);
        // Series 100, 200, 300, 400; explicit 250 splits (200, 300], 410 trails.
        assert_eq!(positions.count(), 6);
        assert_eq!(positions.end(), 410);
        assert_eq!(positions.floor(399), 300);
        assert_eq!(positions.floor(1_000), 410);
        // Tail 30: four series gaps of 30, plus 250's 30 and 410's 10; the
        // (250, 300] gap still covers 30.
        assert_eq!(positions.window_tail_union(30), 4 * 30 + 30 + 10);
        // Tail 80: gaps 80, 80, 50+50 split, 80, 10.
        assert_eq!(positions.window_tail_union(80), 80 + 80 + 50 + 50 + 80 + 10);
    }

    #[test]
    fn live_checkpoints_charge_passed_positions_only() {
        let accounting = KvAccounting::resolve(
            &gdn_layout(),
            KvAccountingOptions {
                resume_points: Some(PrefixResumePoints::EveryBlock),
                ..options(KvOverhead::Vllm)
            },
            1 << 40,
        )
        .unwrap();
        // At 4100 tokens three block boundaries have passed: three states.
        assert_eq!(accounting.checkpoint_units(2_000, 4_100), 3 * 3_168);
        let prompt_end =
            KvAccounting::resolve(&gdn_layout(), options(KvOverhead::Vllm), 1 << 40).unwrap();
        assert_eq!(prompt_end.checkpoint_units(2_000, 4_100), 3_168);
        assert_eq!(prompt_end.checkpoint_units(2_000, 1_000), 0);
    }

    #[test]
    fn fixed_device_bytes_leave_the_budget_only_with_overhead() {
        let layout = full_only(10).with_engine(EngineOverhead {
            block_tokens: None,
            fixed_device_bytes: 1_000,
        });
        let vllm = KvAccounting::resolve(&layout, options(KvOverhead::Vllm), 10_000).unwrap();
        let none = KvAccounting::resolve(&layout, options(KvOverhead::None), 10_000).unwrap();
        assert_eq!((vllm.capacity_units(), none.capacity_units()), (900, 1_000));
    }

    #[test]
    fn a_resume_interval_must_land_on_blocks() {
        let result = KvAccounting::resolve(
            &gdn_layout(),
            KvAccountingOptions {
                resume_interval: Some(1_000),
                ..options(KvOverhead::Vllm)
            },
            1 << 40,
        );
        assert!(result.is_err());
    }
}
