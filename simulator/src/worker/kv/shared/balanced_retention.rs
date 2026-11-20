//! Balanced retention of a pipeline replica's retained prefixes.
//!
//! A pipeline keeps one block pool, sized by its most constrained stage: a
//! running request holds, on every stage, each KV source that stage's layers
//! read, its own and the copies of sources another stage computes. A retained
//! context needs none of that until it is hit again, so under balanced
//! retention the stages store it once between them, `1 / n` of the model's
//! per-token cache each ([`PipelineKvShares::unique`]): a retained entry costs
//! the pool `unique / n` per token instead of the binding stage's bytes
//! ([`RetainedScale`]).
//!
//! A hit puts the copies back before the request's first chunk: each stage
//! pulls the bytes it reads and does not store from its peers with the copy
//! engine (`cuMemcpyBatchAsync` of peer pages, no SMs), so the pull overlaps
//! compute but the request waits for it. A stage stores only bytes it reads
//! where it can (its share is drawn from what it reads), so stage `s` pulls
//! `max(0, need_s - unique / n)` per token; the stages pull at once and the
//! slowest sets the time ([`BalancedRetention::copy`]). Copies run one session
//! after another (FIFO), as one copy queue per pipeline.
//!
//! The pull's time is measured, not derived: a batch of page copies costs a
//! fixed submit-and-sync time, a time per piece (the copy engine's descriptor
//! rate dominates pages of tens of KB), and its bytes at the contiguous line
//! rate ([`BalancedRetentionSpec`]).
//!
//! Not modeled: the copy's HBM traffic slowing the stage's kernels; the
//! rebalancing when a request's context returns to the cache (a stage whose
//! share is not all bytes it holds would receive some; for a stage that reads
//! at least `unique / n` per token it only frees its copies); per-request
//! states and windows, which the scale shrinks with the per-token parts and the
//! pull leaves out.

use crate::arch::kv_layout::{KvCacheLayout, KvGrowth, PipelineKvShares};
use crate::common::Time;

use super::kv_accounting::{charged_bytes_per_entry, KvAccountingOptions};

/// Tokens the per-token byte counts are taken over, so that compressed
/// parts (one entry per few tokens) stay exact.
const SPAN_TOKENS: u64 = 1024;
/// Allocation block when the layout models none (vLLM's default block size).
const DEFAULT_BLOCK_TOKENS: u64 = 16;

/// The measured cost of a stage's peer pull, from the preset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BalancedRetentionSpec {
    /// Submit and sync of one batch.
    pub copy_latency: Time,
    /// Copy-engine time per contiguous piece.
    pub copy_us_per_piece: f64,
    /// Contiguous line rate, bytes per microsecond.
    pub copy_bytes_per_us: f64,
    /// Contiguous pieces a stage pulls per engine block.
    pub pieces_per_block: u32,
}

/// A retained entry's charge as a fraction `num / den` of what the pool's
/// layout charges it. [`Self::ONE`] leaves every charge as it was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RetainedScale {
    pub(super) num: u64,
    pub(super) den: u64,
}

impl RetainedScale {
    pub(crate) const ONE: Self = Self { num: 1, den: 1 };

    /// The scaled charge of `units`, rounded up.
    pub(crate) fn charge(self, units: u64) -> u64 {
        if self == Self::ONE {
            return units;
        }
        (u128::from(units) * u128::from(self.num)).div_ceil(u128::from(self.den)) as u64
    }

    /// The largest unscaled charge whose scaled charge fits `capacity`.
    pub(crate) fn unscaled_within(self, capacity: u64) -> u64 {
        if self == Self::ONE {
            return capacity;
        }
        (u128::from(capacity) * u128::from(self.den) / u128::from(self.num))
            .min(u128::from(u64::MAX)) as u64
    }
}

/// One hit's pull: the slowest stage's bytes and the time all stages take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixCopy {
    pub bytes: u64,
    pub duration: Time,
}

/// Balanced retention resolved for one pipeline head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BalancedRetention {
    scale: RetainedScale,
    /// Bytes per [`SPAN_TOKENS`] the most-pulling stage pulls on a hit.
    pull_bytes_per_span: u64,
    block_tokens: u64,
    spec: BalancedRetentionSpec,
}

/// Charged bytes of `layout`'s per-token parts over [`SPAN_TOKENS`] tokens.
fn charged_bytes_per_span(layout: &KvCacheLayout, options: &KvAccountingOptions) -> u64 {
    layout
        .parts
        .iter()
        .filter_map(|part| match part.growth {
            KvGrowth::PerToken { tokens_per_entry } => {
                let entries = SPAN_TOKENS / u64::from(tokens_per_entry.max(1));
                debug_assert_eq!(SPAN_TOKENS % u64::from(tokens_per_entry.max(1)), 0);
                Some(charged_bytes_per_entry(part, options) * entries)
            }
            _ => None,
        })
        .sum()
}

impl BalancedRetention {
    /// `pool` is the layout the head's pool charges (the most constrained
    /// stage's); `shares` the stages' and the model's caches.
    pub fn resolve(
        pool: &KvCacheLayout,
        shares: &PipelineKvShares,
        options: &KvAccountingOptions,
        spec: BalancedRetentionSpec,
    ) -> anyhow::Result<Self> {
        let n = shares.stages.len() as u64;
        anyhow::ensure!(n > 0, "balanced retention needs a pipeline with stages");
        anyhow::ensure!(
            spec.copy_bytes_per_us > 0.0
                && spec.copy_us_per_piece >= 0.0
                && spec.pieces_per_block > 0,
            "balanced retention needs a positive copy rate and pieces per block"
        );
        let pool_span = charged_bytes_per_span(pool, options);
        let unique_span = charged_bytes_per_span(&shares.unique, options);
        anyhow::ensure!(
            pool_span > 0 && unique_span > 0,
            "balanced retention needs per-token KV"
        );
        let share = unique_span.div_ceil(n);
        let pull_bytes_per_span = shares
            .stages
            .iter()
            .map(|stage| charged_bytes_per_span(stage, options).saturating_sub(share))
            .max()
            .unwrap_or(0);
        Ok(Self {
            scale: RetainedScale {
                num: unique_span,
                den: n * pool_span,
            },
            pull_bytes_per_span,
            block_tokens: pool
                .engine
                .block_tokens
                .map_or(DEFAULT_BLOCK_TOKENS, u64::from),
            spec,
        })
    }

    pub(crate) fn scale(&self) -> RetainedScale {
        self.scale
    }

    /// The pull a hit of `hit_tokens` makes before its first chunk; `None`
    /// when no stage pulls anything.
    pub fn copy(&self, hit_tokens: u32) -> Option<PrefixCopy> {
        let tokens = u64::from(hit_tokens);
        if tokens == 0 || self.pull_bytes_per_span == 0 {
            return None;
        }
        let bytes = (tokens * self.pull_bytes_per_span).div_ceil(SPAN_TOKENS);
        let pieces = tokens.div_ceil(self.block_tokens) * u64::from(self.spec.pieces_per_block);
        let us = pieces as f64 * self.spec.copy_us_per_piece
            + bytes as f64 / self.spec.copy_bytes_per_us;
        Some(PrefixCopy {
            bytes,
            duration: self.spec.copy_latency + Time::from_ns((us * 1e3).round() as u64),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::kv_layout::KvCachePart;

    fn per_token(bytes: u64) -> KvCacheLayout {
        KvCacheLayout::new().with_part(KvCachePart::per_token("kv", bytes))
    }

    fn spec() -> BalancedRetentionSpec {
        BalancedRetentionSpec {
            copy_latency: Time::from_us(29),
            copy_us_per_piece: 0.8,
            copy_bytes_per_us: 577_800.0,
            pieces_per_block: 1,
        }
    }

    /// Two stages that each read 720 B/token of an 891 B/token model (the
    /// second copies sources the first computes): a retained token costs the
    /// pool 445.5 B instead of 720, and each stage pulls 274.5 B back on a hit.
    #[test]
    fn copies_are_dropped_while_retained_and_pulled_back_on_a_hit() {
        let pool = per_token(720);
        let shares = PipelineKvShares {
            stages: vec![per_token(720), per_token(720)],
            unique: per_token(891),
        };
        let balanced =
            BalancedRetention::resolve(&pool, &shares, &KvAccountingOptions::default(), spec())
                .unwrap();
        assert_eq!(balanced.scale().charge(720_000), 445_500);
        assert_eq!(balanced.scale().unscaled_within(445_500), 720_000);
        let copy = balanced.copy(1024).unwrap();
        assert_eq!(copy.bytes, 1024 * 720 - (1024 * 891_u64).div_ceil(2));
        // 1024 tokens in 16-token blocks: 64 pieces.
        let us = 64.0 * 0.8 + copy.bytes as f64 / 577_800.0;
        assert_eq!(
            copy.duration,
            Time::from_us(29) + Time::from_ns((us * 1e3).round() as u64)
        );
    }

    /// Stages without copies: the shares sum to the model, so a balanced
    /// pipeline keeps the mean stage per token and only the larger stage pulls.
    #[test]
    fn stages_without_copies_balance_to_their_mean() {
        let pool = per_token(600);
        let shares = PipelineKvShares::without_copies(vec![per_token(600), per_token(200)]);
        let balanced =
            BalancedRetention::resolve(&pool, &shares, &KvAccountingOptions::default(), spec())
                .unwrap();
        assert_eq!(balanced.scale().charge(600), 400);
        assert_eq!(balanced.copy(1024).unwrap().bytes, 1024 * 200);
    }

    #[test]
    fn a_split_that_stores_what_each_stage_reads_pulls_nothing() {
        let pool = per_token(300);
        let shares = PipelineKvShares::without_copies(vec![per_token(300), per_token(300)]);
        let balanced =
            BalancedRetention::resolve(&pool, &shares, &KvAccountingOptions::default(), spec())
                .unwrap();
        assert_eq!(balanced.scale().charge(300), 300);
        assert_eq!(balanced.copy(1024), None);
    }

    #[test]
    fn the_unit_scale_leaves_charges_unchanged() {
        assert_eq!(RetainedScale::ONE.charge(12_345), 12_345);
        assert_eq!(RetainedScale::ONE.unscaled_within(u64::MAX), u64::MAX);
    }
}
