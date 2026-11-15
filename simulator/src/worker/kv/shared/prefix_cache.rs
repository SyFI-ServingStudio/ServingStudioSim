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
//!
//! ## Round-end states
//!
//! Under a [`StateRetention`] that keeps earlier states, an entry resumable
//! only at its context end also keeps the session's latest round-end states
//! below that end, one reusable state each, and a prompt that ends inside the
//! context resumes from the latest of them at or before it. The entry is one
//! unit with them. A request that takes it carries the states that survive
//! its round, charged here but not evictable, until its context returns
//! ([`PrefixCache::take_carried`], [`PrefixCache::insert_sharing`]). A context
//! that a decode run elsewhere hands back extends the prompt's entry, its
//! prompt-end state kept as the retention picks it ([`PrefixCache::hand_back`]).

use std::collections::{BTreeSet, HashMap};

use serde::Deserialize;

use crate::worker::kv::shared::kv_accounting::{KvAccounting, ResumePositions, SharedKvAccounting};
use crate::worker::kv::shared::state_retention::{
    normalize, Carried, RoundState, StateRetention, StateTag,
};
use crate::worker::kv::shared::tier_chain::LeavingContext;

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

#[derive(Clone, Debug)]
struct PrefixEntry {
    positions: ResumePositions,
    /// The state at `positions.end()`.
    end: StateTag,
    /// Round-end states kept below `positions.end()`, ascending; empty unless
    /// the retention keeps earlier states.
    earlier: Vec<RoundState>,
    /// Units the entry occupies.
    charge: u64,
    insertion_sequence: u64,
    last_access_sequence: u64,
    frequency: u64,
}

impl PrefixEntry {
    /// Its states, the end's last.
    fn states(&self) -> impl Iterator<Item = RoundState> + '_ {
        self.earlier
            .iter()
            .copied()
            .chain(std::iter::once(RoundState {
                at: self.positions.end(),
                tag: self.end,
            }))
    }

    /// The state a prompt sharing `declared` tokens resumes from: the
    /// largest position at or below it, else the latest earlier state there.
    fn resume_state(&self, declared: u64) -> Option<RoundState> {
        let at = self.positions.floor(declared);
        if at > 0 {
            return Some(RoundState { at, tag: self.end });
        }
        self.earlier
            .iter()
            .rev()
            .copied()
            .find(|state| state.at <= declared)
    }
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PrefixCacheMutation {
    pub(crate) kind: PrefixCacheMutationKind,
    pub(crate) session_id: u32,
    pub(crate) entry_tokens_before: u64,
    pub(crate) entry_tokens_after: u64,
    pub(crate) cache_used_before: u64,
    pub(crate) cache_used_after: u64,
    /// The entry a removal took out, for a host tier behind the cache.
    pub(crate) removed: Option<LeavingContext>,
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
    /// The whole context, when the cache kept less of it than it would with
    /// room: it leaves HBM at once.
    pub(crate) unkept: Option<LeavingContext>,
}

/// One attention partition's evictable retained-session tier.
pub(crate) struct PrefixCache {
    accounting: SharedKvAccounting,
    retention: StateRetention,
    max_retained_units: u64,
    /// Units the entries occupy, the sum of their charges, plus the states
    /// running requests carry.
    used_units: u64,
    /// Context tokens the entries keep; reported in receipts.
    used_tokens: u64,
    policy: PrefixCachePolicy,
    entries: HashMap<u32, PrefixEntry>,
    /// The states a running request of the session carries from the entry it
    /// took, until its context returns.
    carried: HashMap<u32, Carried>,
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
            retention: accounting.state_retention(),
            accounting,
            used_units: 0,
            used_tokens: 0,
            policy: budget.policy,
            entries: HashMap::new(),
            carried: HashMap::new(),
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

    /// Context tokens of the session's entry; 0 without one.
    pub(crate) fn entry_tokens(&self, session_id: u32) -> u64 {
        self.entries
            .get(&session_id)
            .map_or(0, |entry| entry.positions.end())
    }

    /// Pure preview used by admission's token-budget and capacity gates: the
    /// largest position or kept state of the entry within `requested_tokens`.
    pub(crate) fn peek(&self, session_id: u32, requested_tokens: u32) -> u32 {
        self.entries
            .get(&session_id)
            .and_then(|entry| entry.resume_state(u64::from(requested_tokens)))
            .map_or(0, |state| state.at as u32)
    }

    /// Move one retained session into an active request.
    ///
    /// The whole entry is removed even when the request declares a shorter
    /// prefix: count-only traces cannot prove that the unused remainder is a
    /// separately reusable page range. A hit on an entry with earlier states
    /// carries those that survive the round, still charged here, until the
    /// request's context returns.
    pub(crate) fn take(&mut self, session_id: u32, requested_tokens: u32) -> PrefixCacheTakeResult {
        let cache_used_before = self.used_tokens;
        let hit_tokens = self.peek(session_id, requested_tokens);
        let removed_entry = self.remove(session_id);
        let entry_tokens = removed_entry
            .as_ref()
            .map(|entry| entry.positions.end())
            .unwrap_or(0);
        let return_metadata = removed_entry
            .as_ref()
            .map(|entry| PrefixCacheReturnMetadata {
                insertion_sequence: entry.insertion_sequence,
                frequency: entry.frequency.saturating_add(1),
            });
        if let Some(entry) =
            removed_entry.filter(|_| self.retention.keeps_earlier() && hit_tokens > 0)
        {
            let resume = entry
                .resume_state(u64::from(requested_tokens))
                .expect("a hit resumes from a state");
            let below: Vec<RoundState> = entry
                .states()
                .filter(|state| state.at <= u64::from(hit_tokens))
                .collect();
            // HBM leaves room for one state the round adds, also when its
            // decode runs elsewhere; a tier read leaves room for both.
            let carried = Carried {
                states: self.retention.survivors(&below, resume.tag, 1),
                resume: Some(resume.tag),
            };
            self.used_units += self.state_units(carried.states.len());
            if let Some(previous) = self.carried.insert(session_id, carried) {
                self.used_units -= self.state_units(previous.states.len());
            }
        }
        PrefixCacheTakeResult {
            hit_tokens,
            full_hit_tokens: entry_tokens.min(u64::from(requested_tokens)) as u32,
            entry_tokens,
            cache_used_before,
            cache_used_after: self.used_tokens,
            return_metadata,
        }
    }

    /// The states a running request of the session carries ([`Self::take`]),
    /// out of this cache's charge: they return with its context
    /// ([`Self::insert_sharing`]). Empty when it carries none.
    pub(crate) fn take_carried(&mut self, session_id: u32) -> Carried {
        let carried = self.carried.remove(&session_id).unwrap_or_default();
        self.used_units -= self.state_units(carried.states.len());
        carried
    }

    /// [`Self::insert_sharing`] of a context that shares nothing with the
    /// session's previous one and carries no states.
    #[cfg(test)]
    pub(crate) fn insert(
        &mut self,
        session_id: u32,
        prompt_tokens: u64,
        context_tokens: u64,
        physical_limit: u64,
        return_metadata: Option<PrefixCacheReturnMetadata>,
    ) -> PrefixCacheInsertResult {
        self.insert_sharing(
            session_id,
            prompt_tokens,
            context_tokens,
            0,
            Carried::default(),
            physical_limit,
            return_metadata,
        )
    }

    /// Retain a completed request's physically resident KV: `context_tokens`
    /// of context after a `prompt_tokens` prompt, of which the first
    /// `shared_tokens` it shares with the session's previous context, arriving
    /// with the `incoming` states its request carried.
    ///
    /// `physical_limit` is the current slack left by active/promised/held KV in
    /// the partition. An optional experiment ceiling was already resolved into
    /// `max_retained_units`; the entry is cut at the last position that fits the
    /// smaller limit. Under a retention that keeps earlier states, the session
    /// entry's and the incoming states within the shared tokens stay below the
    /// new end as the retention picks them, and the entry is kept with all of
    /// them or not at all.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn insert_sharing(
        &mut self,
        session_id: u32,
        prompt_tokens: u64,
        context_tokens: u64,
        shared_tokens: u64,
        incoming: Carried,
        physical_limit: u64,
        return_metadata: Option<PrefixCacheReturnMetadata>,
    ) -> PrefixCacheInsertResult {
        let own = self
            .accounting
            .resume_positions(prompt_tokens, context_tokens);
        self.insert_context(
            session_id,
            own,
            context_tokens,
            shared_tokens,
            StateTag::after(incoming.resume),
            incoming.states,
            physical_limit,
            return_metadata,
        )
    }

    /// Put a context read back from a host tier into HBM: `positions`, with
    /// the state `end` at its end and the `earlier` states the read moved.
    /// It is a prefix of the session's latest context, as an entry the cache
    /// holds for the session is, so the states of both within it stay as the
    /// retention picks them.
    pub(crate) fn restore(
        &mut self,
        session_id: u32,
        positions: ResumePositions,
        end: StateTag,
        earlier: Vec<RoundState>,
        physical_limit: u64,
    ) -> PrefixCacheInsertResult {
        self.insert_context(
            session_id,
            positions,
            positions.end(),
            positions.end(),
            end,
            earlier,
            physical_limit,
            None,
        )
    }

    /// Put back a session's context that a decode run elsewhere extended past
    /// the prompt's end, `shared` (the end of the entry the prompt left):
    /// `positions`, with the handed-back decode's end at its end, a state of
    /// the round of the entry's latest state at or before `shared`.
    pub(crate) fn hand_back(
        &mut self,
        session_id: u32,
        positions: ResumePositions,
        shared: u64,
        physical_limit: u64,
    ) -> PrefixCacheInsertResult {
        let limit = shared.min(positions.end().saturating_sub(1));
        let latest = self
            .entries
            .get(&session_id)
            .into_iter()
            .flat_map(PrefixEntry::states)
            .filter(|state| state.at > 0 && state.at <= limit)
            .max();
        self.insert_context(
            session_id,
            positions,
            positions.end(),
            shared,
            StateTag::handed_back(latest),
            Vec::new(),
            physical_limit,
            None,
        )
    }

    /// Insert a `context_tokens` context resumable at `own`, sharing
    /// `shared_tokens` with the session's previous one, with the state `end`
    /// at its end and the `incoming` states it brings.
    #[allow(clippy::too_many_arguments)]
    fn insert_context(
        &mut self,
        session_id: u32,
        own: ResumePositions,
        context_tokens: u64,
        shared_tokens: u64,
        end: StateTag,
        incoming: Vec<RoundState>,
        physical_limit: u64,
        return_metadata: Option<PrefixCacheReturnMetadata>,
    ) -> PrefixCacheInsertResult {
        let mut mutations = Vec::new();
        let earlier = if self.retention.keeps_earlier() {
            let limit = shared_tokens.min(context_tokens.saturating_sub(1));
            let mut states: Vec<RoundState> = self
                .entries
                .get(&session_id)
                .into_iter()
                .flat_map(PrefixEntry::states)
                .chain(incoming)
                .filter(|state| state.at > 0 && state.at <= limit)
                .collect();
            normalize(&mut states);
            states.push(RoundState {
                at: own.end(),
                tag: end,
            });
            let mut kept = self.retention.keep(states);
            kept.pop();
            kept
        } else {
            Vec::new()
        };
        if let Some(mutation) =
            self.remove_with_receipt(session_id, PrefixCacheMutationKind::SameSessionReplacement)
        {
            mutations.push(mutation);
        }
        let effective_capacity = self.max_retained_units.min(physical_limit);
        mutations.extend(self.shrink_to(effective_capacity));
        // Only kept positions are resumable, so context past the last one that
        // fits is dropped rather than retained as unusable KV. Earlier states
        // exist only beside a single context-end position, which fits whole
        // or not at all.
        let earlier_units = self.state_units(earlier.len());
        let positions = self
            .accounting
            .largest_fitting(&own, effective_capacity.saturating_sub(earlier_units));
        let retained_tokens = positions.end();
        let unkept = (retained_tokens < own.end()).then(|| LeavingContext {
            session_id,
            positions: own,
            end,
            earlier: earlier.clone(),
        });
        if retained_tokens == 0 {
            return PrefixCacheInsertResult {
                mutations,
                retained_tokens,
                unkept,
            };
        }
        let charge = self.accounting.entry_charge(&positions) + earlier_units;
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
            end,
            earlier,
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
            removed: None,
        });
        PrefixCacheInsertResult {
            mutations,
            retained_tokens,
            unkept,
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
            removed: Some(LeavingContext {
                session_id,
                positions: entry.positions,
                end: entry.end,
                earlier: entry.earlier,
            }),
        })
    }

    /// Units `count` reusable states occupy.
    fn state_units(&self, count: usize) -> u64 {
        count as u64 * self.accounting.reusable_state_units()
    }

    fn take_sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        sequence
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::arch::kv_layout::{KvCacheLayout, KvCachePart};
    use crate::worker::kv::shared::kv_accounting::{KvAccountingOptions, PrefixResumePoints};

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

    /// One unit per token plus a 500-unit recurrent state, resumable only at
    /// a context's end, `capacity` units in all.
    fn end_state_cache(capacity: u64) -> PrefixCache {
        round_end_cache(capacity, 1)
    }

    /// [`end_state_cache`] keeping up to `states` round-end states per entry.
    fn round_end_cache(capacity: u64, states: u32) -> PrefixCache {
        retention_cache(capacity, StateRetention::Latest { k: states })
    }

    /// [`end_state_cache`] keeping the round-end states `retention` picks.
    fn retention_cache(capacity: u64, retention: StateRetention) -> PrefixCache {
        let layout = KvCacheLayout::new()
            .with_part(KvCachePart::per_token("kv", 1))
            .with_part(KvCachePart::per_request("state", 500));
        let accounting = KvAccounting::resolve(
            &layout,
            KvAccountingOptions {
                resume_points: Some(PrefixResumePoints::ContextEnd),
                state_retention: retention,
                ..KvAccountingOptions::default()
            },
            capacity,
        )
        .unwrap();
        PrefixCache::new(
            Arc::new(accounting),
            PrefixCacheBudget {
                max_retained_units: capacity,
                policy: PrefixCachePolicy::Lru,
            },
        )
    }

    #[test]
    fn runtime_config_resolves_disabled_uncapped_and_bounded_unit_limits() {
        // 2000-byte units, 1000 of them over two shards.
        let layout = KvCacheLayout::new().with_part(KvCachePart::per_token("kv", 2_000));
        let accounting =
            KvAccounting::resolve(&layout, KvAccountingOptions::default(), 2_000_000).unwrap();
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

    #[test]
    fn end_state_reuses_the_whole_context_or_nothing() {
        let mut cache = end_state_cache(10_000);
        cache.insert(1, 2_000, 2_345, 10_000, None);
        assert_eq!(cache.used_units(), 2_345 + 500);
        // A prompt that extends the kept context resumes from its end state.
        assert_eq!(cache.peek(1, 2_345), 2_345);
        assert_eq!(cache.peek(1, 2_346), 2_345);
        // One that ends inside it has no state to resume from.
        assert_eq!(cache.peek(1, 2_344), 0);
        assert_eq!(cache.take(1, 1_000).hit_tokens, 0);
        assert_eq!(cache.used_units(), 0, "the unusable entry is consumed");
    }

    #[test]
    fn end_state_charges_one_state_per_entry_and_keeps_entries_whole() {
        let mut cache = end_state_cache(10_000);
        cache.insert(1, 2_000, 2_345, 10_000, None);
        // A second entry needs its own state: 6_000 + 500 does not fit beside
        // 2_845, so session 1 is evicted.
        cache.insert(2, 6_000, 6_000, 9_000, None);
        assert_eq!(cache.peek(1, 2_345), 0);
        assert_eq!(cache.used_units(), 6_500);
        // A context larger than the room left is not cut to fit: a cut-off part
        // has no state at its end.
        let result = cache.insert(3, 8_000, 8_000, 8_000, None);
        assert_eq!(result.retained_tokens, 0);
        assert_eq!(cache.peek(3, 8_000), 0);
    }

    impl PrefixCache {
        /// [`Self::insert_sharing`] of a `tokens`-token context sharing
        /// `shared` tokens with the session's previous one, whose request
        /// carried `incoming`.
        fn share(
            &mut self,
            session_id: u32,
            tokens: u64,
            shared: u64,
            incoming: Carried,
            physical_limit: u64,
        ) -> PrefixCacheInsertResult {
            self.insert_sharing(
                session_id,
                tokens,
                tokens,
                shared,
                incoming,
                physical_limit,
                None,
            )
        }
    }

    fn positions(carried: &Carried) -> Vec<u64> {
        carried.states.iter().map(|state| state.at).collect()
    }

    #[test]
    fn earlier_round_end_states_serve_a_prompt_that_ends_inside_the_context() {
        let mut cache = round_end_cache(100_000, 3);
        cache.insert(1, 1_000, 1_000, 100_000, None);
        // Two more rounds, each extending the context the previous one left.
        cache.share(1, 1_500, 1_000, Carried::default(), 100_000);
        cache.share(1, 2_000, 1_500, Carried::default(), 100_000);
        assert_eq!(cache.used_units(), 2_000 + 3 * 500);
        assert_eq!(cache.peek(1, 2_001), 2_000);
        assert_eq!(cache.peek(1, 1_999), 1_500);
        assert_eq!(cache.peek(1, 1_200), 1_000);
        assert_eq!(cache.peek(1, 999), 0);
        // A fourth state pushes the oldest out.
        cache.share(1, 2_500, 2_000, Carried::default(), 100_000);
        assert_eq!(cache.peek(1, 1_200), 0);
        assert_eq!(cache.used_units(), 2_500 + 3 * 500);
    }

    #[test]
    fn a_truncating_round_keeps_only_the_states_before_the_cut() {
        let mut cache = round_end_cache(100_000, 4);
        cache.insert(1, 1_000, 1_000, 100_000, None);
        cache.share(1, 2_000, 1_000, Carried::default(), 100_000);
        // A context that shares only 1_500 tokens: the state at 2_000 is on
        // another branch.
        cache.share(1, 1_800, 1_500, Carried::default(), 100_000);
        assert_eq!(cache.peek(1, 1_799), 1_000);
        assert_eq!(cache.used_units(), 1_800 + 2 * 500);
    }

    #[test]
    fn a_taken_entry_carries_its_states_back_with_its_context() {
        let mut cache = round_end_cache(100_000, 3);
        cache.insert(1, 1_000, 1_000, 100_000, None);
        cache.share(1, 2_000, 1_000, Carried::default(), 100_000);
        // The round resumes at 1_000 (its prompt shares 1_200): the request
        // takes the KV and carries the state at 1_000, still charged here
        // but not an entry.
        let take = cache.take(1, 1_200);
        assert_eq!(take.hit_tokens, 1_000);
        assert_eq!(cache.used_tokens(), 0);
        assert_eq!(cache.used_units(), 500);
        assert_eq!(cache.peek(1, 1_200), 0, "no entry while the request runs");
        assert!(
            cache.shrink_to(0).is_empty(),
            "a carried state is not evictable"
        );
        // Its context returns with the carried state below its own end.
        let carried = cache.take_carried(1);
        assert_eq!(positions(&carried), [1_000]);
        assert_eq!(cache.used_units(), 0);
        cache.share(1, 1_300, 1_200, carried, 100_000);
        assert_eq!(cache.used_units(), 1_300 + 2 * 500);
        assert_eq!(cache.peek(1, 1_100), 1_000);
        // Resuming at the end carries the latest k - 1 states, the end's
        // included; the returning context appends its own and keeps 3.
        cache.take(1, 1_301);
        let carried = cache.take_carried(1);
        assert_eq!(positions(&carried), [1_000, 1_300]);
        cache.share(1, 1_600, 1_301, carried, 100_000);
        assert_eq!(cache.peek(1, 1_350), 1_300);
        assert_eq!(cache.peek(1, 1_100), 1_000);
        assert_eq!(cache.used_units(), 1_600 + 3 * 500);
    }

    fn anchor_cache(count: u32, stride: u32) -> PrefixCache {
        retention_cache(1_000_000, StateRetention::Anchors { count, stride })
    }

    /// One pipeline round whose decode runs elsewhere: the request takes the
    /// entry (sharing `declared`), retains its prompt's context to
    /// `prompt_end`, and the decode hands back 10 more tokens.
    fn hand_back_round(cache: &mut PrefixCache, declared: u64, prompt_end: u64) {
        cache.take(1, declared as u32);
        let carried = cache.take_carried(1);
        cache.share(1, prompt_end, declared, carried, 1_000_000);
        let positions = cache
            .accounting
            .resume_positions(prompt_end, prompt_end + 10);
        cache.hand_back(1, positions, prompt_end, 1_000_000);
    }

    /// Rounds `0..rounds`: round r's prompt ends at 1_000 (r + 1), and the
    /// next round extends its handed-back context.
    fn hand_back_rounds(cache: &mut PrefixCache, rounds: u64) -> Vec<Vec<(u64, u32, bool)>> {
        (0..rounds)
            .map(|round| {
                let declared = if round == 0 { 0 } else { 1_000 * round + 11 };
                hand_back_round(cache, declared, 1_000 * (round + 1));
                entry_states(cache)
            })
            .collect()
    }

    /// The session's entry's states: (position, round, hand-back).
    fn entry_states(cache: &PrefixCache) -> Vec<(u64, u32, bool)> {
        cache.entries[&1]
            .states()
            .map(|state| (state.at, state.tag.round, state.tag.handback))
            .collect()
    }

    #[test]
    fn stride_two_anchors_keep_a_prefill_end_one_to_two_rounds_back() {
        let mut cache = anchor_cache(1, 2);
        assert_eq!(
            hand_back_rounds(&mut cache, 6),
            [
                vec![(1_000, 0, false), (1_010, 0, true)],
                // The current prefill end and round 0's.
                vec![(1_000, 0, false), (2_000, 1, false), (2_010, 1, true)],
                vec![(1_000, 0, false), (3_000, 2, false), (3_010, 2, true)],
                // Round 2 is an anchor once round 3 runs: one round back.
                vec![(3_000, 2, false), (4_000, 3, false), (4_010, 3, true)],
                vec![(3_000, 2, false), (5_000, 4, false), (5_010, 4, true)],
                vec![(5_000, 4, false), (6_000, 5, false), (6_010, 5, true)],
            ]
        );
        // Two anchors at stride 2 keep rounds 2 and 4; stride 1 the latest
        // earlier prefill ends.
        for (count, stride, anchors) in [(2, 2, [3_000, 5_000]), (2, 1, [4_000, 5_000])] {
            let mut cache = anchor_cache(count, stride);
            hand_back_rounds(&mut cache, 6);
            let at: Vec<u64> = entry_states(&cache).iter().map(|state| state.0).collect();
            assert_eq!(
                at,
                [anchors[0], anchors[1], 6_000, 6_010],
                "m{count}s{stride}"
            );
        }
    }

    #[test]
    fn an_inside_round_resumes_from_the_deepest_anchor_at_or_before_its_prefix() {
        let mut cache = anchor_cache(1, 2);
        hand_back_rounds(&mut cache, 6);
        // States at 5_000 (round 4), 6_000 and 6_010.
        assert_eq!(cache.peek(1, 6_011), 6_010);
        assert_eq!(cache.peek(1, 6_005), 6_000);
        assert_eq!(cache.peek(1, 5_500), 5_000);
        assert_eq!(cache.peek(1, 4_999), 0, "older states were not kept");
        // The inside round resumes at round 4's state, so it is round 5 of
        // its lineage: the states past its prefix go, round 4 stays its anchor.
        hand_back_round(&mut cache, 5_500, 5_800);
        assert_eq!(
            entry_states(&cache),
            [(5_000, 4, false), (5_800, 5, false), (5_810, 5, true)]
        );
    }

    #[test]
    fn every_kept_anchor_state_is_charged_and_carried_states_stay_charged() {
        let mut cache = anchor_cache(1, 2);
        hand_back_rounds(&mut cache, 3);
        // 3_010 tokens and states at 1_000, 3_000 and 3_010.
        assert_eq!(cache.used_units(), 3_010 + 3 * 500);
        // Round 3 takes the entry: it carries the states that survive its
        // pass, still charged, while its own state is live. HBM leaves room
        // for one added state only, so the hand-back end at 3_010 survives
        // beside the round-2 anchor at 3_000.
        cache.take(1, 3_011);
        assert_eq!(cache.used_units(), 2 * 500);
        let carried = cache.take_carried(1);
        assert_eq!(positions(&carried), [3_000, 3_010]);
        assert_eq!(carried.resume.map(|tag| tag.round), Some(2));
        cache.share(1, 4_000, 3_011, carried, 1_000_000);
        assert_eq!(cache.used_units(), 4_000 + 3 * 500);
    }

    #[test]
    fn eviction_takes_whole_entries_with_all_their_states() {
        let mut cache = round_end_cache(4_000, 2);
        cache.insert(1, 1_000, 1_000, 4_000, None);
        cache.share(1, 1_200, 1_000, Carried::default(), 4_000); // 1_200 + 2 x 500
        cache.insert(2, 1_000, 1_000, 4_000, None); // 1_000 + 500
        assert_eq!(cache.used_units(), 3_700);
        // Session 3 needs 800: the LRU entry, session 1, goes whole, its
        // earlier state with it.
        let result = cache.insert(3, 300, 300, 4_000, None);
        let evicted: Vec<_> = result
            .mutations
            .iter()
            .map(|m| (m.kind, m.session_id, m.entry_tokens_before))
            .collect();
        assert_eq!(
            evicted,
            [
                (PrefixCacheMutationKind::ReplacementPolicyEviction, 1, 1_200),
                (PrefixCacheMutationKind::Retain, 3, 0)
            ]
        );
        assert_eq!(cache.peek(1, 1_200), 0);
        assert_eq!(cache.peek(2, 1_000), 1_000);
        assert_eq!(cache.used_units(), 1_500 + 800);
    }

    #[test]
    fn an_entry_is_kept_with_all_its_states_or_not_at_all() {
        let mut cache = round_end_cache(2_500, 3);
        cache.insert(1, 1_000, 1_000, 2_500, None);
        // 1_200 + 2 x 500 = 2_200 fits; 1_400 + 3 x 500 does not, and no
        // state is dropped to make it fit.
        cache.share(1, 1_200, 1_000, Carried::default(), 2_500);
        let result = cache.share(1, 1_400, 1_200, Carried::default(), 2_500);
        assert_eq!(result.retained_tokens, 0);
        assert_eq!(cache.used_units(), 0);
    }

    #[test]
    fn one_state_per_entry_keeps_no_earlier_state() {
        let mut cache = round_end_cache(100_000, 1);
        cache.insert(1, 1_000, 1_000, 100_000, None);
        let incoming = Carried {
            states: vec![RoundState {
                at: 1_000,
                tag: StateTag::default(),
            }],
            resume: None,
        };
        cache.share(1, 2_000, 1_000, incoming, 100_000);
        assert_eq!(cache.peek(1, 1_500), 0);
        assert_eq!(cache.used_units(), 2_500);
        cache.take(1, 2_001);
        assert_eq!(cache.used_units(), 0);
        assert!(cache.take_carried(1).states.is_empty());
    }
}
