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
//! A [`LateBudget`] (not vLLM) sets the cap the first two rules work under:
//! `base_tokens` while no queued request is late (SquareCost's late tier is
//! empty), `max_batch_tokens` while one is. A late request already misses its
//! bound, so the step's throughput matters more than its latency then.
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

/// A cap that follows the late tier: `base_tokens` while no queued request
/// is late, `max_batch_tokens` while one is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LateBudget {
    pub base_tokens: u32,
    /// `chunked_prefill`: one late request on any attention DP rank lifts
    /// every rank's cap, instead of each rank following its own late tier.
    pub any_rank: bool,
}

/// The validated budget rules of one worker; the default is vLLM's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MicrobatchBudget {
    pub split: PrefillSplit,
    pub long_prefill_token_threshold: Option<u32>,
    pub load: Option<LoadBudget>,
    pub late: Option<LateBudget>,
}

impl MicrobatchBudget {
    /// Whether [`Self::rule_budget`] can be below `max_batch_tokens`; a
    /// lifecycle may skip it otherwise.
    pub(crate) fn varies(&self) -> bool {
        self.reads_round() || self.late.is_some()
    }

    /// Whether [`Self::rule_budget`] reads the round; a lifecycle may skip
    /// counting it otherwise.
    pub(crate) fn reads_round(&self) -> bool {
        self.load.is_some() || matches!(self.split, PrefillSplit::Even { .. })
    }

    /// The prefill budget the rules allow (a batch takes the smaller of it and
    /// what its decodes leave), given `round_tokens` of prefill work (read only
    /// when [`Self::reads_round`]), a pipeline `depth` (`1` without a
    /// pipeline) and whether a queued request is `late`.
    pub(crate) fn rule_budget(
        &self,
        round_tokens: u64,
        depth: u16,
        late: bool,
        max_batch_tokens: u32,
    ) -> u32 {
        let cap = match self.late {
            Some(rule) if !late => rule.base_tokens,
            _ => max_batch_tokens,
        };
        let mut budget = cap;
        if let Some(load) = self.load {
            budget = budget.min(load.target(round_tokens, cap));
        }
        if let PrefillSplit::Even { min_tokens } = self.split {
            let target = round_tokens
                .div_ceil(u64::from(depth))
                .clamp(u64::from(min_tokens), u64::from(cap));
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

/// Per-minute count of budget decisions and of those at `max_batch_tokens`,
/// written to the run log as each minute closes (and at drop) so a run's
/// share of full-cap batches can be read per window. Only a worker whose
/// budget varies keeps one.
#[derive(Debug, Default)]
pub(crate) struct BudgetTally {
    worker: u16,
    minute: u64,
    decisions: u64,
    at_cap: u64,
}

impl BudgetTally {
    pub(crate) fn new(worker: u16) -> Self {
        Self {
            worker,
            ..Self::default()
        }
    }

    /// Count one decision at `now` (sim time), `at_cap` if it allowed
    /// `max_batch_tokens`.
    pub(crate) fn record(&mut self, now: crate::common::Time, at_cap: bool) {
        let minute = (now.as_ms() / 60_000.0) as u64;
        if minute != self.minute {
            self.flush();
            self.minute = minute;
        }
        self.decisions += 1;
        self.at_cap += u64::from(at_cap);
    }

    fn flush(&mut self) {
        if self.decisions > 0 {
            tracing::info!(
                worker = self.worker,
                minute = self.minute,
                decisions = self.decisions,
                at_cap = self.at_cap,
                "prefill budget tally"
            );
        }
        self.decisions = 0;
        self.at_cap = 0;
    }
}

impl Drop for BudgetTally {
    fn drop(&mut self) {
        self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_late_budget_lifts_the_cap_only_while_a_request_is_late() {
        let budget = MicrobatchBudget {
            late: Some(LateBudget {
                base_tokens: 16_384,
                any_rank: false,
            }),
            ..MicrobatchBudget::default()
        };
        assert!(budget.varies() && !budget.reads_round());
        assert_eq!(budget.rule_budget(0, 1, false, 32_768), 16_384);
        assert_eq!(budget.rule_budget(0, 1, true, 32_768), 32_768);
    }

    #[test]
    fn a_load_budget_under_a_late_budget_rises_to_the_base_until_a_request_is_late() {
        let budget = MicrobatchBudget {
            load: Some(LoadBudget {
                low_tokens: 2_000,
                backlog_lo_tokens: 2_000,
                backlog_hi_tokens: 42_000,
            }),
            late: Some(LateBudget {
                base_tokens: 16_384,
                any_rank: false,
            }),
            ..MicrobatchBudget::default()
        };
        assert_eq!(budget.rule_budget(1_000, 1, false, 32_768), 2_000);
        assert_eq!(budget.rule_budget(22_000, 1, false, 32_768), 9_192);
        assert_eq!(budget.rule_budget(100_000, 1, false, 32_768), 16_384);
        assert_eq!(budget.rule_budget(22_000, 1, true, 32_768), 17_384);
        assert_eq!(budget.rule_budget(100_000, 1, true, 32_768), 32_768);
    }

    #[test]
    fn without_a_late_budget_the_late_flag_changes_nothing() {
        let budget = MicrobatchBudget {
            load: Some(LoadBudget {
                low_tokens: 2_000,
                backlog_lo_tokens: 2_000,
                backlog_hi_tokens: 42_000,
            }),
            ..MicrobatchBudget::default()
        };
        for round in [0, 22_000, 100_000] {
            assert_eq!(
                budget.rule_budget(round, 1, true, 32_768),
                budget.rule_budget(round, 1, false, 32_768)
            );
        }
    }

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
