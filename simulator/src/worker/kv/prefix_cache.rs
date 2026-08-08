//! Session-scoped retained KV for autoregressive prefix reuse.
//!
//! Entries are count-only because traces carry token counts, not token content.
//! A lookup is destructive: retained KV ownership moves to the admitted request,
//! so this cache never models inter-request sharing or ref-counted prefix pages.

use std::collections::HashMap;

use serde::Deserialize;

/// Victim selection when retained session KV must be evicted.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PrefixCachePolicy {
    /// Hits refresh recency; the least recently used session is evicted first.
    #[default]
    Lru,
    /// Hits do not refresh order; the oldest inserted session is evicted first.
    Fifo,
    /// The least frequently consumed session is evicted, with age as the tie-break.
    Lfu,
    /// The largest retained session is evicted, with age as the tie-break.
    LargestFirst,
}

#[derive(Clone, Copy, Debug)]
struct PrefixEntry {
    tokens: u64,
    insertion_sequence: u64,
    last_access_sequence: u64,
    frequency: u64,
}

/// Replacement metadata that follows a destructively consumed session until
/// its active request returns KV to the cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PrefixCacheLease {
    insertion_sequence: u64,
    frequency: u64,
}

/// One attention partition's evictable retained-session tier.
pub(super) struct PrefixCache {
    capacity_tokens: u64,
    used_tokens: u64,
    policy: PrefixCachePolicy,
    entries: HashMap<u32, PrefixEntry>,
    next_sequence: u64,
}

impl PrefixCache {
    pub(super) fn new(capacity_tokens: u64, policy: PrefixCachePolicy) -> Self {
        Self {
            capacity_tokens,
            used_tokens: 0,
            policy,
            entries: HashMap::new(),
            next_sequence: 0,
        }
    }

    pub(super) fn used_tokens(&self) -> u64 {
        self.used_tokens
    }

    /// Pure preview used by admission's token-budget and capacity gates.
    pub(super) fn peek(&self, session_id: u32, requested_tokens: u32) -> u32 {
        self.entries
            .get(&session_id)
            .map(|entry| entry.tokens.min(u64::from(requested_tokens)) as u32)
            .unwrap_or(0)
    }

    /// Move one retained session into an active request.
    ///
    /// The whole entry is removed even when the request declares a shorter
    /// prefix: count-only traces cannot prove that the unused remainder is a
    /// separately reusable page range.
    pub(super) fn take(
        &mut self,
        session_id: u32,
        requested_tokens: u32,
    ) -> (u32, Option<PrefixCacheLease>) {
        let hit = self.peek(session_id, requested_tokens);
        let lease = self.remove(session_id).map(|entry| PrefixCacheLease {
            insertion_sequence: entry.insertion_sequence,
            frequency: entry.frequency.saturating_add(1),
        });
        (hit, lease)
    }

    /// Retain a completed request's physically resident KV.
    ///
    /// `physical_limit` is the current slack left by active/promised/held KV in
    /// the partition. The cache's configured ceiling and physical slack are both
    /// hard limits; the inserted entry is clamped to the smaller one.
    pub(super) fn insert(
        &mut self,
        session_id: u32,
        tokens: u64,
        physical_limit: u64,
        lease: Option<PrefixCacheLease>,
    ) {
        self.remove(session_id);
        let effective_capacity = self.capacity_tokens.min(physical_limit);
        self.shrink_to(effective_capacity);
        let retained_tokens = tokens.min(effective_capacity);
        if retained_tokens == 0 {
            return;
        }
        while self.used_tokens.saturating_add(retained_tokens) > effective_capacity {
            let Some(victim) = self.victim() else {
                break;
            };
            self.remove(victim);
        }
        let sequence = self.take_sequence();
        let insertion_sequence = lease
            .map(|value| value.insertion_sequence)
            .unwrap_or(sequence);
        let frequency = lease.map(|value| value.frequency).unwrap_or(0);
        self.entries.insert(
            session_id,
            PrefixEntry {
                tokens: retained_tokens,
                insertion_sequence,
                last_access_sequence: sequence,
                frequency,
            },
        );
        self.used_tokens += retained_tokens;
    }

    pub(super) fn shrink_to(&mut self, physical_limit: u64) {
        let effective_capacity = self.capacity_tokens.min(physical_limit);
        while self.used_tokens > effective_capacity {
            let Some(victim) = self.victim() else {
                break;
            };
            self.remove(victim);
        }
    }

    fn victim(&self) -> Option<u32> {
        match self.policy {
            PrefixCachePolicy::Lru => self
                .entries
                .iter()
                .min_by_key(|(session_id, entry)| (entry.last_access_sequence, **session_id)),
            PrefixCachePolicy::Fifo => self
                .entries
                .iter()
                .min_by_key(|(session_id, entry)| (entry.insertion_sequence, **session_id)),
            PrefixCachePolicy::Lfu => self.entries.iter().min_by_key(|(session_id, entry)| {
                (entry.frequency, entry.insertion_sequence, **session_id)
            }),
            PrefixCachePolicy::LargestFirst => {
                self.entries.iter().min_by_key(|(session_id, entry)| {
                    (
                        std::cmp::Reverse(entry.tokens),
                        entry.insertion_sequence,
                        **session_id,
                    )
                })
            }
        }
        .map(|(session_id, _)| *session_id)
    }

    fn remove(&mut self, session_id: u32) -> Option<PrefixEntry> {
        let entry = self.entries.remove(&session_id)?;
        self.used_tokens = self.used_tokens.saturating_sub(entry.tokens);
        Some(entry)
    }

    fn take_sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        sequence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_transfers_ownership_instead_of_sharing() {
        let mut cache = PrefixCache::new(100, PrefixCachePolicy::Lru);
        cache.insert(7, 80, 100, None);
        assert_eq!(cache.take(7, 50).0, 50);
        assert_eq!(cache.peek(7, 50), 0);
        assert_eq!(cache.used_tokens(), 0);
    }

    #[test]
    fn lru_evicts_the_oldest_retained_session() {
        let mut cache = PrefixCache::new(100, PrefixCachePolicy::Lru);
        cache.insert(1, 60, 100, None);
        cache.insert(2, 60, 100, None);
        assert_eq!(cache.peek(1, 60), 0);
        assert_eq!(cache.peek(2, 60), 60);
    }

    #[test]
    fn physical_slack_can_shrink_below_the_configured_ceiling() {
        let mut cache = PrefixCache::new(100, PrefixCachePolicy::Fifo);
        cache.insert(1, 40, 100, None);
        cache.insert(2, 40, 100, None);
        cache.shrink_to(40);
        assert_eq!(cache.used_tokens(), 40);
        assert_eq!(cache.peek(1, 40), 0);
        assert_eq!(cache.peek(2, 40), 40);
    }

    #[test]
    fn lru_refreshes_a_returned_lease_but_fifo_preserves_first_residency() {
        for (policy, expected_victim) in [(PrefixCachePolicy::Lru, 2), (PrefixCachePolicy::Fifo, 1)]
        {
            let mut cache = PrefixCache::new(100, policy);
            cache.insert(1, 40, 100, None);
            cache.insert(2, 40, 100, None);
            let (_, lease) = cache.take(1, 40);
            cache.insert(1, 40, 100, lease);
            cache.insert(3, 40, 100, None);
            assert_eq!(cache.peek(expected_victim, 40), 0);
        }
    }

    #[test]
    fn largest_first_evicts_the_biggest_retained_session() {
        let mut cache = PrefixCache::new(100, PrefixCachePolicy::LargestFirst);
        cache.insert(1, 70, 100, None);
        cache.insert(2, 20, 100, None);
        cache.insert(3, 30, 100, None);
        assert_eq!(cache.peek(1, 70), 0);
        assert_eq!(cache.peek(2, 20), 20);
        assert_eq!(cache.peek(3, 30), 30);
    }
}
