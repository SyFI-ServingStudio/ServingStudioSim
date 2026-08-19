//! Session-scoped retained KV for autoregressive prefix reuse.
//!
//! Entries are count-only because traces carry token counts, not token content.
//! A lookup is destructive: retained KV ownership moves to the admitted request,
//! so this cache never models inter-request sharing or ref-counted prefix pages.
//!
//! ## Reuse quantum
//!
//! Full-attention KV is per-token, so any prefix length is reusable — that is a
//! quantum of one. A recurrent state is a single rolling snapshot instead, only
//! written at multiples of the model's checkpoint interval, so its reuse
//! quantizes to that interval and each retained entry additionally carries one
//! `charge_per_checkpoint` for every snapshot it keeps. Both knobs are
//! constructor parameters; `(1, 0)` is exactly the pre-hybrid behavior.

use std::collections::HashMap;

use serde::Deserialize;

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

/// Construction-time token-space form consumed by `FullAttnKv`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefixCacheTokenConfig {
    pub(crate) max_retained_tokens: u64,
    pub(crate) policy: PrefixCachePolicy,
}

impl PrefixCacheConfig {
    /// Convert the optional byte ceiling with the same model KV layout used for
    /// total attention capacity. An uncapped opportunistic cache resolves to the
    /// full capacity; runtime physical slack still shrinks it below that value.
    pub(crate) fn resolve_tokens(
        self,
        total_capacity_tokens: u64,
        total_kv_bytes_per_token: u64,
        num_attn_shards: u16,
    ) -> PrefixCacheTokenConfig {
        match self {
            Self::Disabled => PrefixCacheTokenConfig {
                max_retained_tokens: 0,
                policy: PrefixCachePolicy::Lru,
            },
            Self::Opportunistic {
                policy,
                max_retained_bytes,
            } => {
                let max_retained_tokens = max_retained_bytes
                    .map(|bytes| {
                        bytes.saturating_mul(u64::from(num_attn_shards.max(1)))
                            / total_kv_bytes_per_token.max(1)
                    })
                    .unwrap_or(total_capacity_tokens)
                    .min(total_capacity_tokens);
                PrefixCacheTokenConfig {
                    max_retained_tokens,
                    policy,
                }
            }
        }
    }
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
    max_retained_tokens: u64,
    used_tokens: u64,
    /// Snapshots behind `used_tokens`, tracked incrementally so `used_charge`
    /// stays O(1) inside the eviction loops.
    used_checkpoints: u64,
    policy: PrefixCachePolicy,
    /// Reuse granularity in context tokens. `1` = any prefix length is reusable.
    quantum_tokens: u64,
    /// Extra occupancy every retained snapshot costs on top of its context KV.
    charge_per_checkpoint: u64,
    entries: HashMap<u32, PrefixEntry>,
    next_sequence: u64,
}

impl PrefixCache {
    pub(crate) fn new(
        max_retained_tokens: u64,
        policy: PrefixCachePolicy,
        quantum_tokens: u32,
        charge_per_checkpoint: u64,
    ) -> Self {
        Self {
            max_retained_tokens,
            used_tokens: 0,
            used_checkpoints: 0,
            policy,
            quantum_tokens: u64::from(quantum_tokens.max(1)),
            charge_per_checkpoint,
            entries: HashMap::new(),
            next_sequence: 0,
        }
    }

    /// Context tokens held. Deliberately test-only: every production decision
    /// is about occupancy, which is [`Self::used_charge`]. The two differ once
    /// retained snapshots carry a charge of their own.
    #[cfg(test)]
    pub(crate) fn used_tokens(&self) -> u64 {
        self.used_tokens
    }

    /// What this tier actually occupies: retained context KV plus one charge per
    /// retained snapshot. Equals [`Self::used_tokens`] when there is no per-
    /// checkpoint charge.
    pub(crate) fn used_charge(&self) -> u64 {
        self.used_tokens + self.used_checkpoints * self.charge_per_checkpoint
    }

    /// Round a length down to the coarsest boundary this cache can resume from.
    fn floor_to_quantum(&self, tokens: u64) -> u64 {
        tokens / self.quantum_tokens * self.quantum_tokens
    }

    /// Occupancy a quantized `tokens` would add: the context KV plus one charge
    /// for each snapshot inside it.
    fn charge_of(&self, tokens: u64) -> u64 {
        tokens + tokens / self.quantum_tokens * self.charge_per_checkpoint
    }

    /// The largest quantized length not exceeding `tokens` whose charge fits
    /// `capacity`. Each retained snapshot costs `quantum + charge_per_checkpoint`,
    /// so the count is a plain division; at quantum one with no charge this is
    /// `min(tokens, capacity)`, the pre-hybrid rule.
    fn largest_fitting_length(&self, tokens: u64, capacity: u64) -> u64 {
        let per_checkpoint = self.quantum_tokens + self.charge_per_checkpoint;
        let checkpoints = (tokens / self.quantum_tokens).min(capacity / per_checkpoint);
        checkpoints * self.quantum_tokens
    }

    /// Pure preview used by admission's token-budget and capacity gates.
    ///
    /// A request declaring `requested_tokens` can only resume at a boundary it
    /// shares with the retained entry, so the hit is the smaller of the two
    /// quantized lengths. At quantum one this is the plain `min`.
    pub(crate) fn peek(&self, session_id: u32, requested_tokens: u32) -> u32 {
        self.entries
            .get(&session_id)
            .map(|entry| {
                entry
                    .tokens
                    .min(self.floor_to_quantum(u64::from(requested_tokens))) as u32
            })
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
        let entry_tokens = removed_entry.map(|entry| entry.tokens).unwrap_or(0);
        let return_metadata = removed_entry.map(|entry| PrefixCacheReturnMetadata {
            insertion_sequence: entry.insertion_sequence,
            frequency: entry.frequency.saturating_add(1),
        });
        PrefixCacheTakeResult {
            hit_tokens,
            entry_tokens,
            cache_used_before,
            cache_used_after: self.used_tokens,
            return_metadata,
        }
    }

    /// Retain a completed request's physically resident KV.
    ///
    /// `physical_limit` is the current slack left by active/promised/held KV in
    /// the partition. An optional experiment ceiling was already resolved into
    /// `max_retained_tokens`; the inserted entry is clamped to the smaller limit.
    pub(crate) fn insert(
        &mut self,
        session_id: u32,
        tokens: u64,
        physical_limit: u64,
        return_metadata: Option<PrefixCacheReturnMetadata>,
    ) -> PrefixCacheInsertResult {
        let mut mutations = Vec::new();
        if let Some(mutation) =
            self.remove_with_receipt(session_id, PrefixCacheMutationKind::SameSessionReplacement)
        {
            mutations.push(mutation);
        }
        let effective_capacity = self.max_retained_tokens.min(physical_limit);
        mutations.extend(self.shrink_to(effective_capacity));
        // Only whole snapshots are resumable, so a tail shorter than one quantum
        // is dropped rather than retained as unusable KV.
        let retained_tokens = self.largest_fitting_length(tokens, effective_capacity);
        if retained_tokens == 0 {
            return PrefixCacheInsertResult {
                mutations,
                retained_tokens,
            };
        }
        let retained_charge = self.charge_of(retained_tokens);
        while self.used_charge().saturating_add(retained_charge) > effective_capacity {
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
        self.used_checkpoints += retained_tokens / self.quantum_tokens;
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
        let effective_capacity = self.max_retained_tokens.min(physical_limit);
        let mut mutations = Vec::new();
        while self.used_charge() > effective_capacity {
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
        self.used_checkpoints = self
            .used_checkpoints
            .saturating_sub(entry.tokens / self.quantum_tokens);
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
            entry_tokens_before: entry.tokens,
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

    /// Full-attention settings: every prefix length is resumable and a retained
    /// entry costs exactly its tokens. The quantized/charged behaviour is
    /// covered by the hybrid store's own tests.
    const TOKEN_QUANTUM: u32 = 1;
    const NO_EXTRA_CHARGE: u64 = 0;

    #[test]
    fn runtime_config_resolves_disabled_uncapped_and_bounded_token_limits() {
        assert_eq!(
            PrefixCacheConfig::Disabled.resolve_tokens(1_000, 2_000, 2),
            PrefixCacheTokenConfig {
                max_retained_tokens: 0,
                policy: PrefixCachePolicy::Lru,
            }
        );
        assert_eq!(
            PrefixCacheConfig::default().resolve_tokens(1_000, 2_000, 2),
            PrefixCacheTokenConfig {
                max_retained_tokens: 1_000,
                policy: PrefixCachePolicy::Lru,
            }
        );
        assert_eq!(
            PrefixCacheConfig::Opportunistic {
                policy: PrefixCachePolicy::Fifo,
                max_retained_bytes: Some(400_000),
            }
            .resolve_tokens(1_000, 2_000, 2),
            PrefixCacheTokenConfig {
                max_retained_tokens: 400,
                policy: PrefixCachePolicy::Fifo,
            }
        );
    }

    #[test]
    fn take_transfers_ownership_instead_of_sharing() {
        let mut cache =
            PrefixCache::new(100, PrefixCachePolicy::Lru, TOKEN_QUANTUM, NO_EXTRA_CHARGE);
        cache.insert(7, 80, 100, None);
        assert_eq!(cache.take(7, 50).hit_tokens, 50);
        assert_eq!(cache.peek(7, 50), 0);
        assert_eq!(cache.used_tokens(), 0);
    }

    #[test]
    fn lru_evicts_the_oldest_retained_session() {
        let mut cache =
            PrefixCache::new(100, PrefixCachePolicy::Lru, TOKEN_QUANTUM, NO_EXTRA_CHARGE);
        cache.insert(1, 60, 100, None);
        cache.insert(2, 60, 100, None);
        assert_eq!(cache.peek(1, 60), 0);
        assert_eq!(cache.peek(2, 60), 60);
    }

    #[test]
    fn physical_slack_can_shrink_below_the_configured_ceiling() {
        let mut cache =
            PrefixCache::new(100, PrefixCachePolicy::Fifo, TOKEN_QUANTUM, NO_EXTRA_CHARGE);
        cache.insert(1, 40, 100, None);
        cache.insert(2, 40, 100, None);
        cache.shrink_to(40);
        assert_eq!(cache.used_tokens(), 40);
        assert_eq!(cache.peek(1, 40), 0);
        assert_eq!(cache.peek(2, 40), 40);
    }

    #[test]
    fn lru_refreshes_returned_metadata_but_fifo_preserves_first_residency() {
        for (policy, expected_victim) in [(PrefixCachePolicy::Lru, 2), (PrefixCachePolicy::Fifo, 1)]
        {
            let mut cache = PrefixCache::new(100, policy, TOKEN_QUANTUM, NO_EXTRA_CHARGE);
            cache.insert(1, 40, 100, None);
            cache.insert(2, 40, 100, None);
            let take_result = cache.take(1, 40);
            cache.insert(1, 40, 100, take_result.return_metadata);
            cache.insert(3, 40, 100, None);
            assert_eq!(cache.peek(expected_victim, 40), 0);
        }
    }

    #[test]
    fn largest_first_evicts_the_biggest_retained_session() {
        let mut cache = PrefixCache::new(
            100,
            PrefixCachePolicy::LargestFirst,
            TOKEN_QUANTUM,
            NO_EXTRA_CHARGE,
        );
        cache.insert(1, 70, 100, None);
        cache.insert(2, 20, 100, None);
        cache.insert(3, 30, 100, None);
        assert_eq!(cache.peek(1, 70), 0);
        assert_eq!(cache.peek(2, 20), 20);
        assert_eq!(cache.peek(3, 30), 30);
    }

    #[test]
    fn mutation_receipts_form_an_exact_occupancy_replay() {
        let mut cache =
            PrefixCache::new(100, PrefixCachePolicy::Lru, TOKEN_QUANTUM, NO_EXTRA_CHARGE);
        let first_insert = cache.insert(1, 60, 100, None);
        let second_insert = cache.insert(2, 60, 100, None);
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
