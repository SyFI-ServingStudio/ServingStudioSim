//! Turning prefix-cache mutations into logged events.
//!
//! [`super::prefix_cache::PrefixCache`] returns mutation receipts rather than
//! logging them itself, so it stays a pure ledger. This module owns the other
//! half: the optional logger and the mutation-kind → event-kind mapping. Both
//! stores retain prefixes the same way, so the translation lives here once.
//!
//! It also hands the contexts the cache lets go of, evicted or never kept, to
//! a victim tier behind HBM when there is one ([`HbmEvictionLog`]).

use std::cell::RefCell;
use std::rc::Rc;

use crate::common::{RequestId, Time};
use crate::log::{
    PrefixCacheEvent, PrefixCacheEventKind, PrefixCacheEvictionReason, PrefixCacheLogger,
    PrefixCacheRetentionReason,
};
use crate::worker::shared::advance_scope::PartitionId;

use super::prefix_cache::{PrefixCacheMutation, PrefixCacheMutationKind};

/// A session context that left the HBM prefix cache: evicted, or not kept
/// when it was retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HbmEviction {
    pub(crate) partition: PartitionId,
    pub(crate) session_id: u32,
    pub(crate) tokens: u64,
    /// Only the state at `tokens` left (an earlier round-end state); the
    /// context's KV stays in HBM.
    pub(crate) state_only: bool,
}

/// The contexts the KV store's prefix cache let go of, in order, until the
/// victim tier behind it (`admission/prefix_fetch.rs`) takes them.
pub(crate) type HbmEvictionLog = Rc<RefCell<Vec<HbmEviction>>>;

pub(crate) struct PrefixCacheJournal {
    logger: Option<PrefixCacheLogger>,
    evictions: Option<HbmEvictionLog>,
}

impl PrefixCacheJournal {
    pub(crate) fn new(logger: Option<PrefixCacheLogger>) -> Self {
        Self {
            logger,
            evictions: None,
        }
    }

    /// Append every context the cache lets go of to `log`.
    pub(crate) fn set_eviction_log(&mut self, log: HbmEvictionLog) {
        self.evictions = Some(log);
    }

    /// A retention kept less of the session's `tokens` than the cache keeps
    /// with room: the context leaves HBM at once.
    pub(crate) fn record_unkept(&mut self, partition: PartitionId, session_id: u32, tokens: u64) {
        if let Some(log) = &self.evictions {
            log.borrow_mut().push(HbmEviction {
                partition,
                session_id,
                tokens,
                state_only: false,
            });
        }
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
        if mutation.kind == PrefixCacheMutationKind::EarlierStateEviction {
            // The context stays in HBM; a victim tier gets the state alone, to
            // keep beside the session's KV if it holds it. Not a token-level
            // cache event.
            if let Some(log) = &self.evictions {
                log.borrow_mut().push(HbmEviction {
                    partition,
                    session_id: mutation.session_id,
                    tokens: mutation.state_at,
                    state_only: true,
                });
            }
            return;
        }
        if let Some(log) = &self.evictions {
            if matches!(
                mutation.kind,
                PrefixCacheMutationKind::CapacityEviction
                    | PrefixCacheMutationKind::ReplacementPolicyEviction
            ) {
                log.borrow_mut().push(HbmEviction {
                    partition,
                    session_id: mutation.session_id,
                    tokens: mutation.entry_tokens_before,
                    state_only: false,
                });
            }
        }
        self.record(PrefixCacheEvent {
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
            // Never logged (`record_mutation` returns first); its context stays.
            PrefixCacheMutationKind::EarlierStateEviction => {
                PrefixCacheEventKind::Evict(PrefixCacheEvictionReason::RetentionCapacity)
            }
        }
    }
}
