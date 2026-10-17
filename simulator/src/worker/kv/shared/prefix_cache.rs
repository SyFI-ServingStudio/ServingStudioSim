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
//!
//! An engine that keeps the state only at a context's end instead
//! ([`PrefixCache::with_end_state`]) can resume only there: a later prompt
//! reuses the whole retained context when it extends it, and nothing when it
//! shares less, since no state exists inside the context. Each entry pays one
//! `charge_per_entry` for that final state. [`ResumePoint`] states the three
//! rules.
//!
//! ## Round-end states
//!
//! Such an engine may also keep earlier states of the same context
//! ([`PrefixCache::with_end_state`]'s `states` > 1): the states it computed at
//! the end of each earlier pass over the session's context (a prefill's end, a
//! handed-back decode's end), the latest `states` of them, each charged one
//! `charge_per_entry`. A prompt that ends inside the context then resumes from
//! the latest kept state at or before its end. Invariant: every kept state
//! sits at a prefix of the entry's context, since an entry only keeps the
//! earlier states the new context shares (`shared`), and an earlier state
//! never displaces a context: under pressure the cache drops earlier states
//! (the victim-ordered entry's oldest first) before it evicts any entry. A
//! request that takes an entry takes its KV and the state it resumes from (a
//! copy); the earlier states at or before that point stay behind as a
//! states-only entry, evicted first, until the request's context returns.

use std::collections::{BTreeSet, HashMap};

use serde::Deserialize;

/// Where a retained context can be resumed from, which decides how much of it a
/// later prompt that shares `declared` tokens with it reuses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResumePoint {
    /// Per-token KV only (full attention): any prefix of the context.
    AnyToken,
    /// Recurrent state saved every `tokens` tokens (vLLM's Mamba `align`
    /// mode): the longest shared multiple of `tokens`.
    EveryInterval { tokens: u32 },
    /// Recurrent state saved only at the context's end: the whole context when
    /// the prompt extends it, and nothing when the prompt ends inside it.
    ContextEnd,
}

impl ResumePoint {
    /// Tokens of a `kept`-token context a prompt sharing `declared` tokens with
    /// it resumes from.
    pub(crate) fn reusable_tokens(self, kept: u64, declared: u64) -> u64 {
        match self {
            Self::AnyToken => kept.min(declared),
            Self::EveryInterval { tokens } => {
                let interval = u64::from(tokens.max(1));
                kept.min(declared) / interval * interval
            }
            Self::ContextEnd if kept <= declared => kept,
            Self::ContextEnd => 0,
        }
    }
}

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

#[derive(Clone, Debug)]
struct PrefixEntry {
    tokens: u64,
    insertion_sequence: u64,
    last_access_sequence: u64,
    frequency: u64,
    /// Earlier round-end states kept beside the one at `tokens`, ascending,
    /// each below `tokens` (all of them when `tokens` is 0: a states-only
    /// entry whose KV a running request holds). Empty unless the cache keeps
    /// more than one state per entry.
    earlier_states: Vec<u64>,
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
    /// [`PrefixCacheMutationKind::EarlierStateEviction`]: the dropped state's
    /// context length; 0 otherwise.
    pub(crate) state_at: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefixCacheMutationKind {
    SameSessionReplacement,
    CapacityEviction,
    ReplacementPolicyEviction,
    Retain,
    /// One earlier round-end state dropped from an entry whose context stays;
    /// `state_at` is where it was.
    EarlierStateEviction,
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
    /// Retention granularity in context tokens: an entry keeps whole quanta.
    quantum_tokens: u64,
    /// Where a retained entry can be resumed from.
    resume: ResumePoint,
    /// Extra occupancy every retained snapshot costs on top of its context KV.
    charge_per_checkpoint: u64,
    /// Extra occupancy every kept state costs: the one at an entry's end and
    /// each earlier one ([`Self::with_end_state`]).
    charge_per_entry: u64,
    /// Round-end states an entry keeps, its end's included (1: the end only).
    max_states: usize,
    /// Entries holding context tokens (each with its end state), and the
    /// earlier states of every entry.
    end_states: u64,
    earlier_states: u64,
    entries: HashMap<u32, PrefixEntry>,
    /// `entries` ordered by the policy's eviction key, session id last, so the
    /// victim is the first element. A retained entry's key never changes (a
    /// hit removes it), so only insert and remove touch this.
    eviction_order: BTreeSet<(u64, u64, u32)>,
    /// The keys of the entries with earlier states, in the same order: their
    /// states go before any entry does.
    with_earlier_states: BTreeSet<(u64, u64, u32)>,
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
            resume: if quantum_tokens > 1 {
                ResumePoint::EveryInterval {
                    tokens: quantum_tokens,
                }
            } else {
                ResumePoint::AnyToken
            },
            charge_per_checkpoint,
            charge_per_entry: 0,
            max_states: 1,
            end_states: 0,
            earlier_states: 0,
            entries: HashMap::new(),
            eviction_order: BTreeSet::new(),
            with_earlier_states: BTreeSet::new(),
            next_sequence: 0,
        }
    }

    /// State only where a pass over a context ended: every entry keeps the
    /// `charge` state at its end and nothing inside, so it is reused whole by
    /// a prompt that extends it, or not at all; with `states` > 1 it also
    /// keeps up to `states - 1` earlier round-end states (module docs).
    /// Replaces the snapshot quantum and charge.
    pub(crate) fn with_end_state(mut self, charge: u64, states: u32) -> Self {
        assert!(self.entries.is_empty(), "set the end state before use");
        assert!(states >= 1, "an entry keeps at least its end state");
        self.quantum_tokens = 1;
        self.resume = ResumePoint::ContextEnd;
        self.charge_per_checkpoint = 0;
        self.charge_per_entry = charge;
        self.max_states = states as usize;
        self
    }

    /// Where a prompt sharing `declared` tokens with `entry` resumes.
    fn resume_tokens(&self, entry: &PrefixEntry, declared: u64) -> u64 {
        let end = if entry.tokens > 0 {
            self.resume.reusable_tokens(entry.tokens, declared)
        } else {
            0
        };
        if end > 0 {
            return end;
        }
        // Earlier states need the context's KV up to them.
        entry
            .earlier_states
            .iter()
            .rev()
            .copied()
            .find(|&at| at <= declared && at <= entry.tokens)
            .unwrap_or(0)
    }

    /// Context tokens held. Deliberately test-only: every production decision
    /// is about occupancy, which is [`Self::used_charge`]. The two differ once
    /// retained snapshots carry a charge of their own.
    #[cfg(test)]
    pub(crate) fn used_tokens(&self) -> u64 {
        self.used_tokens
    }

    /// What this tier actually occupies: retained context KV plus one charge per
    /// retained snapshot or entry. Equals [`Self::used_tokens`] when there is
    /// neither charge.
    pub(crate) fn used_charge(&self) -> u64 {
        self.used_tokens
            + self.used_checkpoints * self.charge_per_checkpoint
            + (self.end_states + self.earlier_states) * self.charge_per_entry
    }

    /// Occupancy a quantized `tokens` would add: the context KV plus one charge
    /// for each snapshot inside it.
    fn charge_of(&self, tokens: u64) -> u64 {
        let entry = if tokens > 0 { self.charge_per_entry } else { 0 };
        tokens + tokens / self.quantum_tokens * self.charge_per_checkpoint + entry
    }

    /// The largest quantized length not exceeding `tokens` whose charge fits
    /// `capacity`. Each retained snapshot costs `quantum + charge_per_checkpoint`,
    /// so the count is a plain division; at quantum one with no charge this is
    /// `min(tokens, capacity)`, the pre-hybrid rule. A context resumable only
    /// at its end is kept whole or not at all: a cut-off part has no state.
    fn largest_fitting_length(&self, tokens: u64, capacity: u64) -> u64 {
        if self.resume == ResumePoint::ContextEnd {
            return if self.charge_of(tokens) <= capacity {
                tokens
            } else {
                0
            };
        }
        let per_checkpoint = self.quantum_tokens + self.charge_per_checkpoint;
        let checkpoints = (tokens / self.quantum_tokens).min(capacity / per_checkpoint);
        checkpoints * self.quantum_tokens
    }

    /// What a retention of `tokens` keeps given room: whole resumable quanta.
    pub(crate) fn retainable_tokens(&self, tokens: u64) -> u64 {
        self.largest_fitting_length(tokens, u64::MAX)
    }

    /// Pure preview used by admission's token-budget and capacity gates.
    ///
    /// A request declaring `requested_tokens` can only resume at a point it
    /// shares with the retained entry ([`ResumePoint::reusable_tokens`]).
    pub(crate) fn peek(&self, session_id: u32, requested_tokens: u32) -> u32 {
        self.entries
            .get(&session_id)
            .map(|entry| self.resume_tokens(entry, u64::from(requested_tokens)) as u32)
            .unwrap_or(0)
    }

    /// Move one retained session into an active request.
    ///
    /// The whole entry is removed even when the request declares a shorter
    /// prefix: count-only traces cannot prove that the unused remainder is a
    /// separately reusable page range. With round-end states, those at or
    /// before the resume point stay as a states-only entry (module docs).
    pub(crate) fn take(&mut self, session_id: u32, requested_tokens: u32) -> PrefixCacheTakeResult {
        let cache_used_before = self.used_tokens;
        let hit_tokens = self.peek(session_id, requested_tokens);
        let removed_entry = self.remove(session_id);
        let entry_tokens = removed_entry.as_ref().map(|entry| entry.tokens).unwrap_or(0);
        let return_metadata = removed_entry.as_ref().map(|entry| PrefixCacheReturnMetadata {
            insertion_sequence: entry.insertion_sequence,
            frequency: entry.frequency.saturating_add(1),
        });
        if let Some(entry) = removed_entry.filter(|_| self.max_states > 1 && hit_tokens > 0) {
            // The request leaves a state at its own end, so these keep room for it.
            let states = self.earlier_states_up_to(&entry, u64::from(hit_tokens), self.max_states - 1);
            if !states.is_empty() {
                self.add_entry(
                    session_id,
                    PrefixEntry {
                        tokens: 0,
                        earlier_states: states,
                        ..entry
                    },
                );
            }
        }
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
        self.insert_sharing(session_id, tokens, 0, physical_limit, return_metadata)
    }

    /// [`Self::insert`] of a context that shares its first `shared` tokens
    /// with the session's entry: the entry's round-end states within them
    /// stay, below the new end state, up to the per-entry count.
    pub(crate) fn insert_sharing(
        &mut self,
        session_id: u32,
        tokens: u64,
        shared: u64,
        physical_limit: u64,
        return_metadata: Option<PrefixCacheReturnMetadata>,
    ) -> PrefixCacheInsertResult {
        let mut mutations = Vec::new();
        let mut earlier = match self.entries.get(&session_id) {
            Some(entry) if self.max_states > 1 => self.earlier_states_up_to(
                entry,
                shared.min(tokens.saturating_sub(1)),
                self.max_states - 1,
            ),
            _ => Vec::new(),
        };
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
        let per_state = self.charge_per_entry;
        // Earlier states never displace a context: other entries' go first,
        // then this one's, and only then whole entries.
        while self
            .used_charge()
            .saturating_add(retained_charge + earlier.len() as u64 * per_state)
            > effective_capacity
        {
            if let Some(mutation) = self.drop_earlier_state() {
                mutations.push(mutation);
                continue;
            }
            if !earlier.is_empty() {
                earlier.remove(0);
                continue;
            }
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
        self.add_entry(
            session_id,
            PrefixEntry {
                tokens: retained_tokens,
                insertion_sequence,
                last_access_sequence: sequence,
                frequency,
                earlier_states: earlier,
            },
        );
        mutations.push(PrefixCacheMutation {
            kind: PrefixCacheMutationKind::Retain,
            session_id,
            entry_tokens_before: 0,
            entry_tokens_after: retained_tokens,
            cache_used_before,
            cache_used_after: self.used_tokens,
            state_at: 0,
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
            if let Some(mutation) = self.drop_earlier_state() {
                mutations.push(mutation);
                continue;
            }
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
            // Largest first: more tokens sort earlier.
            PrefixCachePolicy::LargestFirst => (
                u64::MAX - entry.tokens,
                entry.insertion_sequence,
                session_id,
            ),
        }
    }

    fn add_entry(&mut self, session_id: u32, entry: PrefixEntry) {
        let key = self.eviction_key(session_id, &entry);
        self.eviction_order.insert(key);
        if !entry.earlier_states.is_empty() {
            self.with_earlier_states.insert(key);
        }
        self.used_tokens += entry.tokens;
        self.used_checkpoints += entry.tokens / self.quantum_tokens;
        self.end_states += u64::from(entry.tokens > 0);
        self.earlier_states += entry.earlier_states.len() as u64;
        self.entries.insert(session_id, entry);
    }

    fn remove(&mut self, session_id: u32) -> Option<PrefixEntry> {
        let entry = self.entries.remove(&session_id)?;
        let key = self.eviction_key(session_id, &entry);
        self.eviction_order.remove(&key);
        self.with_earlier_states.remove(&key);
        self.used_tokens = self.used_tokens.saturating_sub(entry.tokens);
        self.used_checkpoints = self
            .used_checkpoints
            .saturating_sub(entry.tokens / self.quantum_tokens);
        self.end_states -= u64::from(entry.tokens > 0);
        self.earlier_states -= entry.earlier_states.len() as u64;
        Some(entry)
    }

    /// The entry's round-end states (its end's included) at or before `limit`,
    /// the latest `count` of them, ascending.
    fn earlier_states_up_to(&self, entry: &PrefixEntry, limit: u64, count: usize) -> Vec<u64> {
        let mut states: Vec<u64> = entry
            .earlier_states
            .iter()
            .copied()
            .chain((entry.tokens > 0).then_some(entry.tokens))
            .filter(|&at| at > 0 && at <= limit)
            .collect();
        states.drain(..states.len().saturating_sub(count));
        states
    }

    /// Drop the oldest earlier state of the first victim-ordered entry that
    /// has one; `None` when no entry has any.
    fn drop_earlier_state(&mut self) -> Option<PrefixCacheMutation> {
        let key = *self.with_earlier_states.first()?;
        let session_id = key.2;
        let entry = self.entries.get_mut(&session_id).expect("indexed entry");
        let state_at = entry.earlier_states.remove(0);
        let tokens = entry.tokens;
        let empty = entry.earlier_states.is_empty();
        self.earlier_states -= 1;
        if empty {
            self.with_earlier_states.remove(&key);
            if tokens == 0 {
                // A states-only entry with nothing left.
                self.remove(session_id);
            }
        }
        Some(PrefixCacheMutation {
            kind: PrefixCacheMutationKind::EarlierStateEviction,
            session_id,
            entry_tokens_before: tokens,
            entry_tokens_after: tokens,
            cache_used_before: self.used_tokens,
            cache_used_after: self.used_tokens,
            state_at,
        })
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
            state_at: 0,
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
    fn lfu_evicts_the_least_consumed_session_then_the_oldest() {
        let mut cache =
            PrefixCache::new(100, PrefixCachePolicy::Lfu, TOKEN_QUANTUM, NO_EXTRA_CHARGE);
        cache.insert(1, 30, 100, None);
        cache.insert(2, 30, 100, None);
        cache.insert(3, 30, 100, None);
        // Session 1 was consumed once, so 2 (older than 3) goes first, then 3.
        let take_result = cache.take(1, 30);
        cache.insert(1, 30, 100, take_result.return_metadata);
        cache.insert(4, 30, 100, None);
        assert_eq!(cache.peek(2, 30), 0);
        cache.insert(5, 30, 100, None);
        assert_eq!(cache.peek(3, 30), 0);
        assert_eq!(cache.peek(1, 30), 30);
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

    #[test]
    fn end_state_reuses_the_whole_context_or_nothing() {
        let mut cache =
            PrefixCache::new(10_000, PrefixCachePolicy::Lru, 1_000, 500).with_end_state(500, 1);
        cache.insert(1, 2_345, 10_000, None);
        assert_eq!(cache.used_charge(), 2_845);
        // A prompt that extends the kept context resumes from its end state.
        assert_eq!(cache.peek(1, 2_345), 2_345);
        assert_eq!(cache.peek(1, 2_346), 2_345);
        // One that ends inside it has no state to resume from.
        assert_eq!(cache.peek(1, 2_344), 0);
        assert_eq!(cache.take(1, 1_000).hit_tokens, 0);
        assert_eq!(cache.used_charge(), 0, "the unusable entry is consumed");
    }

    #[test]
    fn end_state_charges_one_state_per_entry_and_keeps_entries_whole() {
        let mut cache =
            PrefixCache::new(10_000, PrefixCachePolicy::Lru, 1_000, 500).with_end_state(500, 1);
        cache.insert(1, 2_345, 10_000, None);
        // A second entry needs its own state: 6_000 + 500 does not fit beside
        // 2_845, so session 1 is evicted.
        cache.insert(2, 6_000, 9_000, None);
        assert_eq!(cache.peek(1, 2_345), 0);
        assert_eq!(cache.used_charge(), 6_500);
        // A context larger than the room left is not cut to fit: a cut-off part
        // has no state at its end.
        let result = cache.insert(3, 8_000, 8_000, None);
        assert_eq!(result.retained_tokens, 0);
        assert_eq!(cache.peek(3, 8_000), 0);
    }

    /// State-at-end entries keeping up to `states` round-end states, a state
    /// costing 500 tokens.
    fn round_end_cache(capacity: u64, states: u32) -> PrefixCache {
        PrefixCache::new(capacity, PrefixCachePolicy::Lru, 1, 0).with_end_state(500, states)
    }

    #[test]
    fn earlier_round_end_states_serve_a_prompt_that_ends_inside_the_context() {
        let mut cache = round_end_cache(100_000, 3);
        cache.insert(1, 1_000, 100_000, None);
        // The next round extends it (shares 1_000); a decode hands back more.
        cache.insert_sharing(1, 1_500, 1_000, 100_000, None);
        cache.insert_sharing(1, 2_000, 1_500, 100_000, None);
        assert_eq!(cache.used_charge(), 2_000 + 3 * 500);
        assert_eq!(cache.peek(1, 2_001), 2_000);
        assert_eq!(cache.peek(1, 1_999), 1_500);
        assert_eq!(cache.peek(1, 1_200), 1_000);
        assert_eq!(cache.peek(1, 999), 0);
        // A fourth state pushes the oldest out.
        cache.insert_sharing(1, 2_500, 2_000, 100_000, None);
        assert_eq!(cache.peek(1, 1_200), 0);
        assert_eq!(cache.used_charge(), 2_500 + 3 * 500);
    }

    #[test]
    fn only_states_within_the_shared_prefix_survive_a_new_context() {
        let mut cache = round_end_cache(100_000, 4);
        cache.insert(1, 1_000, 100_000, None);
        cache.insert_sharing(1, 2_000, 1_000, 100_000, None);
        // A context that shares only 1_500 tokens: the state at 2_000 is on
        // another branch.
        cache.insert_sharing(1, 1_800, 1_500, 100_000, None);
        assert_eq!(cache.peek(1, 1_799), 1_000);
        assert_eq!(cache.used_charge(), 1_800 + 2 * 500);
    }

    #[test]
    fn a_taken_entry_leaves_its_states_up_to_the_resume_point_behind() {
        let mut cache = round_end_cache(100_000, 3);
        cache.insert(1, 1_000, 100_000, None);
        cache.insert_sharing(1, 2_000, 1_000, 100_000, None);
        // The round resumes at 1_000 (its prompt ends at 1_200): the request
        // takes the KV; the state at 1_000 stays, without KV.
        let take = cache.take(1, 1_200);
        assert_eq!(take.hit_tokens, 1_000);
        assert_eq!(cache.used_tokens(), 0);
        assert_eq!(cache.used_charge(), 500);
        assert_eq!(cache.peek(1, 1_200), 0, "no KV, no hit");
        // The request's context returns and keeps it below its own end.
        cache.insert_sharing(1, 1_300, 1_200, 100_000, take.return_metadata);
        assert_eq!(cache.used_charge(), 1_300 + 2 * 500);
        assert_eq!(cache.peek(1, 1_100), 1_000);
    }

    #[test]
    fn earlier_states_go_before_any_context() {
        let mut cache = round_end_cache(4_000, 2);
        cache.insert(1, 1_000, 4_000, None);
        cache.insert_sharing(1, 1_200, 1_000, 4_000, None); // 1_200 + 2 x 500
        cache.insert(2, 1_000, 4_000, None); // 1_000 + 500
        assert_eq!(cache.used_charge(), 3_700);
        // Session 3 needs 800: session 1's earlier state goes, no context does.
        let result = cache.insert(3, 300, 4_000, None);
        let kinds: Vec<_> = result.mutations.iter().map(|m| (m.kind, m.state_at)).collect();
        assert_eq!(
            kinds,
            [
                (PrefixCacheMutationKind::EarlierStateEviction, 1_000),
                (PrefixCacheMutationKind::Retain, 0)
            ]
        );
        assert_eq!(cache.peek(1, 1_200), 1_200);
        assert_eq!(cache.peek(1, 1_100), 0);
        assert_eq!(cache.peek(2, 1_000), 1_000);
        // With no earlier state left, contexts are evicted as before.
        cache.insert(4, 700, 4_000, None);
        assert_eq!(cache.peek(1, 1_200), 0);
    }

    #[test]
    fn one_state_per_entry_keeps_no_earlier_state() {
        let mut cache = round_end_cache(100_000, 1);
        cache.insert(1, 1_000, 100_000, None);
        cache.insert_sharing(1, 2_000, 1_000, 100_000, None);
        assert_eq!(cache.peek(1, 1_500), 0);
        assert_eq!(cache.used_charge(), 2_500);
        cache.take(1, 2_001);
        assert_eq!(cache.used_charge(), 0);
    }

    #[test]
    fn resume_points_bound_what_a_shorter_prompt_reuses() {
        assert_eq!(ResumePoint::AnyToken.reusable_tokens(5_000, 3_210), 3_210);
        assert_eq!(
            ResumePoint::EveryInterval { tokens: 1_000 }.reusable_tokens(5_000, 3_210),
            3_000
        );
        assert_eq!(ResumePoint::ContextEnd.reusable_tokens(5_000, 3_210), 0);
        for rule in [
            ResumePoint::AnyToken,
            ResumePoint::EveryInterval { tokens: 1_000 },
            ResumePoint::ContextEnd,
        ] {
            assert_eq!(
                rule.reusable_tokens(5_000, 5_001),
                5_000,
                "{rule:?}: a prompt extending the context reuses all of it"
            );
        }
    }
}
