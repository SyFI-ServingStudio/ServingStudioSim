//! Turning prefix-cache mutations into logged events.
//!
//! [`super::prefix_cache::PrefixCache`] returns mutation receipts rather than
//! logging them itself, so it stays a pure ledger. This module owns the other
//! half: the optional logger and the mutation-kind → event-kind mapping. Both
//! stores retain prefixes the same way, so the translation lives here once.

use crate::common::{RequestId, Time};
use crate::log::{
    PrefixCacheActivation, PrefixCacheEvent, PrefixCacheEventKind, PrefixCacheEvictionReason,
    PrefixCacheLogger, PrefixCacheRetentionReason, HBM_TIER,
};
use crate::worker::shared::advance_scope::PartitionId;

use super::prefix_cache::{PrefixCacheMutation, PrefixCacheMutationKind};
use super::tier_chain::{TierEvent, TierEventKind};

pub(crate) struct PrefixCacheJournal {
    logger: Option<PrefixCacheLogger>,
}

impl PrefixCacheJournal {
    pub(crate) fn new(logger: Option<PrefixCacheLogger>) -> Self {
        Self { logger }
    }

    pub(crate) fn is_logging(&self) -> bool {
        self.logger.is_some()
    }

    /// One host tier's entry change, on behalf of `request`.
    pub(crate) fn record_tier_event(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        now: Time,
        tier: &'static str,
        event: TierEvent,
    ) {
        let kind = match event.kind {
            TierEventKind::WriteThrough => {
                PrefixCacheEventKind::Retain(PrefixCacheRetentionReason::TierWriteThrough)
            }
            TierEventKind::Victim => {
                PrefixCacheEventKind::Retain(PrefixCacheRetentionReason::TierVictim)
            }
            TierEventKind::ReadCopy => {
                PrefixCacheEventKind::Retain(PrefixCacheRetentionReason::TierReadCopy)
            }
            TierEventKind::WarmStart => {
                PrefixCacheEventKind::Retain(PrefixCacheRetentionReason::TierWarmStart)
            }
            TierEventKind::Evict => {
                PrefixCacheEventKind::Evict(PrefixCacheEvictionReason::ReplacementPolicy)
            }
            TierEventKind::TooLarge => {
                PrefixCacheEventKind::Retain(PrefixCacheRetentionReason::NoCacheCapacity)
            }
            TierEventKind::Cut => PrefixCacheEventKind::Evict(PrefixCacheEvictionReason::TierCut),
            TierEventKind::Read => PrefixCacheEventKind::Activate(PrefixCacheActivation::TierRead),
            TierEventKind::Hop => PrefixCacheEventKind::Activate(PrefixCacheActivation::TierHop),
        };
        let requested_tokens = match kind {
            PrefixCacheEventKind::Activate(_) => event.hit_tokens,
            PrefixCacheEventKind::Retain(_) => event.entry_tokens_after,
            PrefixCacheEventKind::Evict(_) => 0,
        };
        self.record(PrefixCacheEvent {
            tier,
            partition_id: partition,
            time: now,
            request_id: request,
            session_id: event.session_id,
            kind,
            entry_tokens_before: event.entry_tokens_before,
            entry_tokens_after: event.entry_tokens_after,
            cache_used_before: event.used_tokens_before,
            cache_used_after: event.used_tokens_after,
            requested_tokens,
            hit_tokens: event.hit_tokens,
            full_hit_tokens: event.hit_tokens,
            cache_used_units: Some((event.used_units_before, event.used_units_after)),
            ready: event.ready,
        });
    }

    pub(crate) fn record(&mut self, event: PrefixCacheEvent) {
        if let Some(logger) = &mut self.logger {
            logger.record(event);
        }
    }

    /// One mutation receipt, classified. `retain_requested_tokens` is what the
    /// caller *asked* to retain, which only a `Retain` receipt reports — an
    /// eviction did not request anything.
    pub(crate) fn record_mutation(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        now: Time,
        mutation: PrefixCacheMutation,
        event_kind: PrefixCacheEventKind,
        retain_requested_tokens: u64,
    ) {
        self.record(PrefixCacheEvent {
            tier: HBM_TIER,
            partition_id: partition,
            time: now,
            request_id: request,
            session_id: mutation.session_id,
            kind: event_kind,
            entry_tokens_before: mutation.entry_tokens_before,
            entry_tokens_after: mutation.entry_tokens_after,
            cache_used_before: mutation.cache_used_before,
            cache_used_after: mutation.cache_used_after,
            requested_tokens: if mutation.kind == PrefixCacheMutationKind::Retain {
                retain_requested_tokens
            } else {
                0
            },
            hit_tokens: 0,
            full_hit_tokens: 0,
            cache_used_units: None,
            ready: None,
        });
    }

    /// The eviction/retention taxonomy a retain-path mutation maps to. Only the
    /// `Retain` arm needs the caller's reason; every eviction reason is implied
    /// by the mutation kind itself.
    pub(crate) fn retain_event_kind(
        mutation_kind: PrefixCacheMutationKind,
        retain_reason: PrefixCacheRetentionReason,
    ) -> PrefixCacheEventKind {
        match mutation_kind {
            PrefixCacheMutationKind::SameSessionReplacement => {
                PrefixCacheEventKind::Evict(PrefixCacheEvictionReason::SameSessionReplacement)
            }
            PrefixCacheMutationKind::CapacityEviction => {
                PrefixCacheEventKind::Evict(PrefixCacheEvictionReason::RetentionCapacity)
            }
            PrefixCacheMutationKind::ReplacementPolicyEviction => {
                PrefixCacheEventKind::Evict(PrefixCacheEvictionReason::ReplacementPolicy)
            }
            PrefixCacheMutationKind::Retain => PrefixCacheEventKind::Retain(retain_reason),
        }
    }
}
