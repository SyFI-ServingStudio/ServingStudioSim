//! Session-scoped retained KV for autoregressive prefix reuse.
//!
//! Entries are count-only because traces carry token counts, not token content.
//! A lookup is destructive: retained KV ownership moves to the admitted request,
//! so this cache never models inter-request sharing or ref-counted prefix pages.
//!
//! ## Resume positions
//!
//! Per-token KV can be resumed from any length. A sliding window or a
//! recurrent state can only be resumed where the engine kept it, so an entry
//! records the [`ResumePositions`] its request left behind and a hit is the
//! largest of them a later request's declared prefix reaches. The entry pays
//! for its per-token parts up to its last position plus the window tails and
//! states at each position ([`KvAccounting::entry_charge`]); with only
//! per-token parts that is its token count, as before.

use std::collections::{BTreeSet, HashMap};

use serde::Deserialize;

use crate::worker::kv::shared::kv_accounting::{KvAccounting, ResumePositions, SharedKvAccounting};

/// Whether completed-session KV may be retained after a request finishes.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PrefixCacheMode {
    /// Drop completed-session KV. This is an explicit no-reuse baseline.
    Disabled,
    /// Retain into whatever physical attention slack is currently available.
    #[default]
    Opportunistic,
}

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

/// Validated runtime configuration for the retained-session tier.
///
/// The default has no carved-out capacity: completed prefixes may occupy all
/// physical attention slack, while active/promised/held KV remains strictly
/// higher priority. `max_retained_bytes` is only an optional experiment ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixCacheConfig {
    Disabled,
    Opportunistic {
        policy: PrefixCachePolicy,
        max_retained_bytes: Option<u64>,
    },
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self::Opportunistic {
            policy: PrefixCachePolicy::Lru,
            max_retained_bytes: None,
        }
    }
}

/// Construction-time form consumed by `KvCacheStore`, in the partition's KV
/// units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefixCacheBudget {
    pub(crate) max_retained_units: u64,
    pub(crate) policy: PrefixCachePolicy,
}

impl PrefixCacheBudget {
    pub(crate) const DISABLED: Self = Self {
        max_retained_units: 0,
        policy: PrefixCachePolicy::Lru,
    };
}

impl PrefixCacheConfig {
    /// Convert the optional per-GPU byte ceiling into units of `accounting`,
    /// over the partition's `num_attn_shards` GPUs. An uncapped opportunistic
    /// cache resolves to the full capacity; runtime physical slack still
    /// shrinks it below that value.
    pub(crate) fn resolve_units(
        self,
        accounting: &KvAccounting,
        num_attn_shards: u16,
    ) -> PrefixCacheBudget {
        match self {
            Self::Disabled => PrefixCacheBudget::DISABLED,
            Self::Opportunistic {
                policy,
                max_retained_bytes,
            } => {
                let capacity_units = accounting.capacity_units();
                let max_retained_units = max_retained_bytes
                    .map(|bytes| {
                        accounting
                            .units_of_bytes(bytes.saturating_mul(u64::from(num_attn_shards.max(1))))
                    })
                    .unwrap_or(capacity_units)
                    .min(capacity_units);
                PrefixCacheBudget {
                    max_retained_units,
                    policy,
                }
            }
        }
    }

    pub(crate) fn enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

#[derive(Clone, Copy, Debug)]
struct PrefixEntry {
    positions: ResumePositions,
    /// Units the entry occupies.
    charge: u64,
    insertion_sequence: u64,
    last_access_sequence: u64,
    frequency: u64,
}

/// Replacement metadata that follows a destructively consumed session until
/// its active request returns KV to the cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PrefixCacheReturnMetadata {
    insertion_sequence: u64,
    frequency: u64,
}

/// One authoritative cache-entry transition returned to the KV owner.
///
/// Logging consumes these receipts instead of reconstructing mutations from
/// sampled occupancy. That keeps `PrefixCache` as the only cache ledger while
/// still exposing enough information for an exact operation replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefixCacheMutation {
    pub(crate) kind: PrefixCacheMutationKind,
    pub(crate) session_id: u32,
    pub(crate) entry_tokens_before: u64,
    pub(crate) entry_tokens_after: u64,
    pub(crate) cache_used_before: u64,
    pub(crate) cache_used_after: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefixCacheMutationKind {
    SameSessionReplacement,
    CapacityEviction,
    ReplacementPolicyEviction,
    Retain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefixCacheTakeResult {
    pub(crate) hit_tokens: u32,
    /// What the hit would be if every position were resumable: the declared
    /// prefix, capped at the entry's context. The gap to `hit_tokens` is what
    /// the entry's resume positions cost.
    pub(crate) full_hit_tokens: u32,
    pub(crate) entry_tokens: u64,
    pub(crate) cache_used_before: u64,
    pub(crate) cache_used_after: u64,
    pub(crate) return_metadata: Option<PrefixCacheReturnMetadata>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PrefixCacheInsertResult {
    pub(crate) mutations: Vec<PrefixCacheMutation>,
    pub(crate) retained_tokens: u64,
}

/// One attention partition's evictable retained-session tier.
pub(crate) struct PrefixCache {
    accounting: SharedKvAccounting,
    max_retained_units: u64,
    /// Units the entries occupy, the sum of their charges.
    used_units: u64,
    /// Context tokens the entries keep; reported in receipts.
    used_tokens: u64,
    policy: PrefixCachePolicy,
    entries: HashMap<u32, PrefixEntry>,
    /// `entries` ordered by the policy's eviction key, session id last, so the
    /// victim is the first element. A retained entry never changes (a hit
    /// removes it), so only insert and remove touch this.
    eviction_order: BTreeSet<(u64, u64, u32)>,
    next_sequence: u64,
}

impl PrefixCache {
    pub(crate) fn new(accounting: SharedKvAccounting, budget: PrefixCacheBudget) -> Self {
        Self {
            max_retained_units: budget.max_retained_units.min(accounting.capacity_units()),
            accounting,
            used_units: 0,
            used_tokens: 0,
            policy: budget.policy,
            entries: HashMap::new(),
            eviction_order: BTreeSet::new(),
            next_sequence: 0,
        }
    }

    /// Context tokens held, as receipts report them. Every capacity decision
    /// is about occupancy, which is [`Self::used_units`].
    pub(crate) fn used_tokens(&self) -> u64 {
        self.used_tokens
    }

    /// Units this tier occupies.
    pub(crate) fn used_units(&self) -> u64 {
        self.used_units
    }

    /// Pure preview used by admission's token-budget and capacity gates: the
    /// largest position of the entry within `requested_tokens`.
    pub(crate) fn peek(&self, session_id: u32, requested_tokens: u32) -> u32 {
        self.entries
            .get(&session_id)
            .map(|entry| entry.positions.floor(u64::from(requested_tokens)) as u32)
            .unwrap_or(0)
    }

    /// Move one retained session into an active request.
    ///
    /// The whole entry is removed even when the request declares a shorter
    /// prefix: count-only traces cannot prove that the unused remainder is a
    /// separately reusable page range.
    pub(crate) fn take(&mut self, session_id: u32, requested_tokens: u32) -> PrefixCacheTakeResult {
        let cache_used_before = self.used_tokens;
        let hit_tokens = self.peek(session_id, requested_tokens);
        let removed_entry = self.remove(session_id);
        let entry_tokens = removed_entry
            .map(|entry| entry.positions.end())
            .unwrap_or(0);
        let return_metadata = removed_entry.map(|entry| PrefixCacheReturnMetadata {
            insertion_sequence: entry.insertion_sequence,
            frequency: entry.frequency.saturating_add(1),
        });
        PrefixCacheTakeResult {
            hit_tokens,
            full_hit_tokens: entry_tokens.min(u64::from(requested_tokens)) as u32,
            entry_tokens,
            cache_used_before,
            cache_used_after: self.used_tokens,
            return_metadata,
        }
    }

    /// Retain a completed request's physically resident KV: `context_tokens`
    /// of context after a `prompt_tokens` prompt.
    ///
    /// `physical_limit` is the current slack left by active/promised/held KV in
    /// the partition. An optional experiment ceiling was already resolved into
    /// `max_retained_units`; the entry is cut at the last position that fits the
    /// smaller limit.
    pub(crate) fn insert(
        &mut self,
        session_id: u32,
        prompt_tokens: u64,
        context_tokens: u64,
        physical_limit: u64,
        return_metadata: Option<PrefixCacheReturnMetadata>,
    ) -> PrefixCacheInsertResult {
        let mut mutations = Vec::new();
        if let Some(mutation) =
            self.remove_with_receipt(session_id, PrefixCacheMutationKind::SameSessionReplacement)
        {
            mutations.push(mutation);
        }
        let effective_capacity = self.max_retained_units.min(physical_limit);
        mutations.extend(self.shrink_to(effective_capacity));
        // Only kept positions are resumable, so context past the last one that
        // fits is dropped rather than retained as unusable KV.
        let positions = self
            .accounting
            .resume_positions(prompt_tokens, context_tokens);
        let positions = self
            .accounting
            .largest_fitting(&positions, effective_capacity);
        let retained_tokens = positions.end();
        if retained_tokens == 0 {
            return PrefixCacheInsertResult {
                mutations,
                retained_tokens,
            };
        }
        let charge = self.accounting.entry_charge(&positions);
        while self.used_units.saturating_add(charge) > effective_capacity {
            let Some(victim) = self.victim() else {
                break;
            };
            if let Some(mutation) =
                self.remove_with_receipt(victim, PrefixCacheMutationKind::ReplacementPolicyEviction)
            {
                mutations.push(mutation);
            }
        }
        let sequence = self.take_sequence();
        let insertion_sequence = return_metadata
            .map(|value| value.insertion_sequence)
            .unwrap_or(sequence);
        let frequency = return_metadata.map(|value| value.frequency).unwrap_or(0);
        let cache_used_before = self.used_tokens;
        let entry = PrefixEntry {
            positions,
            charge,
            insertion_sequence,
            last_access_sequence: sequence,
            frequency,
        };
        self.eviction_order
            .insert(self.eviction_key(session_id, &entry));
        self.entries.insert(session_id, entry);
        self.used_units += charge;
        self.used_tokens += retained_tokens;
        mutations.push(PrefixCacheMutation {
            kind: PrefixCacheMutationKind::Retain,
            session_id,
            entry_tokens_before: 0,
            entry_tokens_after: retained_tokens,
            cache_used_before,
            cache_used_after: self.used_tokens,
        });
        PrefixCacheInsertResult {
            mutations,
            retained_tokens,
        }
    }

    pub(crate) fn shrink_to(&mut self, physical_limit: u64) -> Vec<PrefixCacheMutation> {
        let effective_capacity = self.max_retained_units.min(physical_limit);
        let mut mutations = Vec::new();
        while self.used_units > effective_capacity {
            let Some(victim) = self.victim() else {
                break;
            };
            if let Some(mutation) =
                self.remove_with_receipt(victim, PrefixCacheMutationKind::CapacityEviction)
            {
                mutations.push(mutation);
            }
        }
        mutations
    }

    fn victim(&self) -> Option<u32> {
        self.eviction_order
            .first()
            .map(|&(_, _, session_id)| session_id)
    }

    /// Where `entry` sorts in [`Self::eviction_order`]: smallest is evicted
    /// first, with the session id breaking every tie.
    fn eviction_key(&self, session_id: u32, entry: &PrefixEntry) -> (u64, u64, u32) {
        match self.policy {
            PrefixCachePolicy::Lru => (entry.last_access_sequence, 0, session_id),
            PrefixCachePolicy::Fifo => (entry.insertion_sequence, 0, session_id),
            PrefixCachePolicy::Lfu => (entry.frequency, entry.insertion_sequence, session_id),
            // Largest first: a larger charge sorts earlier.
            PrefixCachePolicy::LargestFirst => (
                u64::MAX - entry.charge,
                entry.insertion_sequence,
                session_id,
            ),
        }
    }

    fn remove(&mut self, session_id: u32) -> Option<PrefixEntry> {
        let entry = self.entries.remove(&session_id)?;
        self.eviction_order
            .remove(&self.eviction_key(session_id, &entry));
        self.used_units -= entry.charge;
        self.used_tokens -= entry.positions.end();
        Some(entry)
    }

    fn remove_with_receipt(
        &mut self,
        session_id: u32,
        kind: PrefixCacheMutationKind,
    ) -> Option<PrefixCacheMutation> {
        let cache_used_before = self.used_tokens;
        let entry = self.remove(session_id)?;
        Some(PrefixCacheMutation {
            kind,
            session_id,
            entry_tokens_before: entry.positions.end(),
            entry_tokens_after: 0,
            cache_used_before,
            cache_used_after: self.used_tokens,
        })
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

    /// Plain token accounting: every prefix length is resumable and a retained
    /// entry costs exactly its tokens. Positions and charges of other layouts
    /// are covered by `kv_accounting` and the store's tests.
    fn cache(max_retained_units: u64, policy: PrefixCachePolicy) -> PrefixCache {
        PrefixCache::new(
            KvAccounting::tokens_for_test(max_retained_units),
            PrefixCacheBudget {
                max_retained_units,
                policy,
            },
        )
    }

    #[test]
    fn runtime_config_resolves_disabled_uncapped_and_bounded_unit_limits() {
        // 2000-byte units, 1000 of them over two shards.
        let layout = crate::arch::kv_layout::KvCacheLayout::new()
            .with_part(crate::arch::kv_layout::KvCachePart::per_token("kv", 2_000));
        let accounting = KvAccounting::resolve(
            &layout,
            crate::worker::kv::shared::kv_accounting::KvAccountingOptions::default(),
            2_000_000,
        )
        .unwrap();
        assert_eq!(
            PrefixCacheConfig::Disabled.resolve_units(&accounting, 2),
            PrefixCacheBudget::DISABLED
        );
        assert_eq!(
            PrefixCacheConfig::default().resolve_units(&accounting, 2),
            PrefixCacheBudget {
                max_retained_units: 1_000,
                policy: PrefixCachePolicy::Lru,
            }
        );
        assert_eq!(
            PrefixCacheConfig::Opportunistic {
                policy: PrefixCachePolicy::Fifo,
                max_retained_bytes: Some(400_000),
            }
            .resolve_units(&accounting, 2),
            PrefixCacheBudget {
                max_retained_units: 400,
                policy: PrefixCachePolicy::Fifo,
            }
        );
    }

    #[test]
    fn take_transfers_ownership_instead_of_sharing() {
        let mut cache = cache(100, PrefixCachePolicy::Lru);
        cache.insert(7, 80, 80, 100, None);
        assert_eq!(cache.take(7, 50).hit_tokens, 50);
        assert_eq!(cache.peek(7, 50), 0);
        assert_eq!(cache.used_tokens(), 0);
    }

    #[test]
    fn lru_evicts_the_oldest_retained_session() {
        let mut cache = cache(100, PrefixCachePolicy::Lru);
        cache.insert(1, 60, 60, 100, None);
        cache.insert(2, 60, 60, 100, None);
        assert_eq!(cache.peek(1, 60), 0);
        assert_eq!(cache.peek(2, 60), 60);
    }

    #[test]
    fn physical_slack_can_shrink_below_the_configured_ceiling() {
        let mut cache = cache(100, PrefixCachePolicy::Fifo);
        cache.insert(1, 40, 40, 100, None);
        cache.insert(2, 40, 40, 100, None);
        cache.shrink_to(40);
        assert_eq!(cache.used_tokens(), 40);
        assert_eq!(cache.peek(1, 40), 0);
        assert_eq!(cache.peek(2, 40), 40);
    }

    #[test]
    fn lru_refreshes_returned_metadata_but_fifo_preserves_first_residency() {
        for (policy, expected_victim) in [(PrefixCachePolicy::Lru, 2), (PrefixCachePolicy::Fifo, 1)]
        {
            let mut cache = cache(100, policy);
            cache.insert(1, 40, 40, 100, None);
            cache.insert(2, 40, 40, 100, None);
            let take_result = cache.take(1, 40);
            cache.insert(1, 40, 40, 100, take_result.return_metadata);
            cache.insert(3, 40, 40, 100, None);
            assert_eq!(cache.peek(expected_victim, 40), 0);
        }
    }

    #[test]
    fn lfu_evicts_the_least_consumed_session_then_the_oldest() {
        let mut cache = cache(100, PrefixCachePolicy::Lfu);
        cache.insert(1, 30, 30, 100, None);
        cache.insert(2, 30, 30, 100, None);
        cache.insert(3, 30, 30, 100, None);
        // Session 1 was consumed once, so 2 (older than 3) goes first, then 3.
        let take_result = cache.take(1, 30);
        cache.insert(1, 30, 30, 100, take_result.return_metadata);
        cache.insert(4, 30, 30, 100, None);
        assert_eq!(cache.peek(2, 30), 0);
        cache.insert(5, 30, 30, 100, None);
        assert_eq!(cache.peek(3, 30), 0);
        assert_eq!(cache.peek(1, 30), 30);
    }

    #[test]
    fn largest_first_evicts_the_biggest_retained_session() {
        let mut cache = cache(100, PrefixCachePolicy::LargestFirst);
        cache.insert(1, 70, 70, 100, None);
        cache.insert(2, 20, 20, 100, None);
        cache.insert(3, 30, 30, 100, None);
        assert_eq!(cache.peek(1, 70), 0);
        assert_eq!(cache.peek(2, 20), 20);
        assert_eq!(cache.peek(3, 30), 30);
    }

    #[test]
    fn mutation_receipts_form_an_exact_occupancy_replay() {
        let mut cache = cache(100, PrefixCachePolicy::Lru);
        let first_insert = cache.insert(1, 60, 60, 100, None);
        let second_insert = cache.insert(2, 60, 60, 100, None);
        let take = cache.take(2, 40);

        let mutations: Vec<PrefixCacheMutation> = first_insert
            .mutations
            .into_iter()
            .chain(second_insert.mutations)
            .collect();
        assert_eq!(mutations.len(), 3);
        for mutation in &mutations {
            assert_eq!(
                mutation.cache_used_after,
                mutation.cache_used_before - mutation.entry_tokens_before
                    + mutation.entry_tokens_after
            );
        }
        assert_eq!(mutations[0].cache_used_before, 0);
        assert_eq!(mutations[0].cache_used_after, 60);
        assert_eq!(
            mutations[1].kind,
            PrefixCacheMutationKind::ReplacementPolicyEviction
        );
        assert_eq!(mutations[1].session_id, 1);
        assert_eq!(mutations[2].cache_used_before, 0);
        assert_eq!(mutations[2].cache_used_after, 60);
        assert_eq!(take.hit_tokens, 40);
        assert_eq!(take.entry_tokens, 60);
        assert_eq!(take.cache_used_before, 60);
        assert_eq!(take.cache_used_after, 0);
    }
}
