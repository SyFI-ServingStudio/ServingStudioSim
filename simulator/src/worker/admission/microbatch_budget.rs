//! How many prefill tokens one batch takes: a pipeline microbatch, or one
//! attention DP rank's share of a whole iteration.
//!
//! vLLM fills the batch greedily up to `max_batch_tokens`. Three rules can lower
//! that, and they compose by taking the smallest budget:
//!
//! - `PrefillSplit::Even` (pipeline only, not vLLM) sizes each microbatch to one
//!   `depth`-th of the round the pipeline holds,
//!   `clamp(ceil(round / depth), min_tokens, max_batch_tokens)`, so one prompt
//!   into an empty pipeline runs as `depth` equal slices rather than one full
//!   microbatch and a small tail. The floor sizes; it never holds a microbatch.
//! - A [`LoadBudget`] (not vLLM) runs batches at `low_tokens` while the backlog
//!   is light, rising linearly to `max_batch_tokens`, so a light load drains
//!   sooner.
//! - `long_prefill_token_threshold` (vLLM) caps any one request's chunk, so a
//!   long prompt leaves budget to the requests behind it.
//!
//! The round (or backlog) is the lifecycle's to count, since only it knows what
//! is queued, started and in flight; this type only turns it into a budget.

use crate::worker::kv::ResolvedPrefillContext;

use super::chunked_prefill_admission::{next_chunk_tokens, ChunkEndAlignment};

/// How a batch's prefill budget follows the work waiting for it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrefillSplit {
    /// Take as much as `max_batch_tokens` allows (vLLM).
    #[default]
    Greedy,
    /// One pipeline-depth-th of the round, floored at `min_tokens`.
    Even { min_tokens: u32 },
}

/// A budget that follows the prefill backlog: `low_tokens` up to
/// `backlog_lo_tokens`, rising linearly to the cap at `backlog_hi_tokens`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadBudget {
    pub low_tokens: u32,
    pub backlog_lo_tokens: u64,
    pub backlog_hi_tokens: u64,
}

impl LoadBudget {
    /// The budget at `backlog_tokens` for a `max_tokens` cap.
    pub fn target(&self, backlog_tokens: u64, max_tokens: u32) -> u32 {
        let rise = ((backlog_tokens as f64 - self.backlog_lo_tokens as f64)
            / (self.backlog_hi_tokens - self.backlog_lo_tokens) as f64)
            .clamp(0.0, 1.0);
        self.low_tokens + (f64::from(max_tokens - self.low_tokens) * rise).round() as u32
    }
}

/// The validated budget rules of one worker; the default is vLLM's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MicrobatchBudget {
    pub split: PrefillSplit,
    pub long_prefill_token_threshold: Option<u32>,
    pub load: Option<LoadBudget>,
}

impl MicrobatchBudget {
    /// Whether [`Self::prefill_budget`] reads the round; a lifecycle may skip
    /// counting it otherwise.
    pub(crate) fn follows_backlog(&self) -> bool {
        self.load.is_some() || matches!(self.split, PrefillSplit::Even { .. })
    }

    /// The prefill budget of a batch with `remaining` tokens left after its
    /// decodes, given `round_tokens` of prefill work and a pipeline `depth`
    /// (`1` without a pipeline).
    pub(crate) fn prefill_budget(
        &self,
        remaining: u32,
        round_tokens: u64,
        depth: u16,
        max_batch_tokens: u32,
    ) -> u32 {
        let mut budget = remaining;
        if let Some(load) = self.load {
            budget = budget.min(load.target(round_tokens, max_batch_tokens));
        }
        if let PrefillSplit::Even { min_tokens } = self.split {
            let target = round_tokens
                .div_ceil(u64::from(depth))
                .clamp(u64::from(min_tokens), u64::from(max_batch_tokens));
            budget = budget.min(target as u32);
        }
        budget
    }

    /// The next chunk of `resolved_prefill` within `budget`, capped per request
    /// by the threshold. The threshold also lowers the chunk cap the checkpoint
    /// alignment compares its quantum against, as vLLM's
    /// `long_prefill_token_threshold` caps `num_new_tokens` before the split.
    pub(crate) fn chunk_tokens(
        &self,
        resolved_prefill: ResolvedPrefillContext,
        budget: u32,
        alignment: Option<ChunkEndAlignment>,
        max_batch_tokens: u32,
    ) -> u32 {
        let threshold = self.long_prefill_token_threshold.unwrap_or(u32::MAX);
        next_chunk_tokens(
            resolved_prefill,
            budget.min(threshold),
            alignment,
            max_batch_tokens.min(threshold),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_load_budget_rises_linearly_between_its_backlog_bounds() {
        let load = LoadBudget {
            low_tokens: 4096,
            backlog_lo_tokens: 1_000_000,
            backlog_hi_tokens: 3_000_000,
        };
        assert_eq!(load.target(0, 8192), 4096);
        assert_eq!(load.target(2_000_000, 8192), 6144);
        assert_eq!(load.target(9_000_000, 8192), 8192);
    }
}
