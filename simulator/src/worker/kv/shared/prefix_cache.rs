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
//! Such an engine may keep `k` states per context
//! ([`PrefixCache::with_end_state`]'s `states`): the states it computed at the
//! end of the latest `k` passes over the session's context (a prefill's end, a
//! handed-back decode's end), fewer only when the session has had fewer passes
//! or a round that did not extend the context cut the later ones off. A prompt
//! that ends inside the context resumes from the latest kept state at or before
//! its end. Every kept state sits at a prefix of the entry's context, since an
//! entry only keeps the states the new context shares (`shared`).
//!
//! An entry is one unit: its KV and all its states, charged together
//! (`tokens + states x charge_per_entry`), kept or evicted whole (no state
//! leaves alone), and handed on whole to a victim tier behind HBM. A request
//! that takes an entry takes its KV, and carries its states at or before the
//! point it resumes from (the latest `k - 1`, which still sit in HBM and stay
//! charged here, not evictable) until its context returns: the returning entry
//! keeps them below its new end state, the oldest beyond `k` dropped.

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

/// What a round-end state is, beside where it sits: the round that computed
/// it, counted along the context's lineage (a round is one past the round of
/// the state it resumed from, 0 when it resumed from none), and whether it is
/// a handed-back decode's end rather than the end of a pass this engine ran
/// (a prefill's, or a local decode's).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateTag {
    pub round: u32,
    pub handback: bool,
}

/// One round-end state: where it sits in the context, and what it is.
/// Ordered by position first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RoundState {
    pub at: u64,
    pub tag: StateTag,
}

impl RoundState {
    pub(crate) fn new(at: u64, tag: StateTag) -> Self {
        Self { at, tag }
    }
}

/// Positions of `states`, ascending.
#[cfg(test)]
pub(crate) fn positions(states: &[RoundState]) -> Vec<u64> {
    states.iter().map(|state| state.at).collect()
}

/// Sort `states` by position and keep one per position, the latest round's.
pub(crate) fn normalize_states(states: &mut Vec<RoundState>) {
    states.sort_unstable();
    states.reverse();
    states.dedup_by_key(|state| state.at);
    states.reverse();
}

/// Which round-end states of a context resumable only at a state stay with
/// it, in HBM and every tier. Rule-based: its only parameters are its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateRetention {
    /// The latest `n` round-end states (`recurrent_state_checkpoints`).
    Latest(u32),
    /// The latest handed-back decode's end, the latest pass's end (a
    /// prefill's), and `count` anchors: the latest earlier passes' ends from
    /// rounds that are a multiple of `stride`. `stride` 2 records an anchor
    /// every other round, so the kept anchor is one to two rounds back at a
    /// constant state count; `stride` 1 is the latest earlier pass ends.
    Anchors { count: u32, stride: u32 },
}

impl Default for StateRetention {
    fn default() -> Self {
        Self::Latest(1)
    }
}

impl StateRetention {
    /// The most states an entry keeps, its end's included.
    pub(crate) fn max_states(self) -> usize {
        match self {
            Self::Latest(n) => n as usize,
            Self::Anchors { count, .. } => 2 + count as usize,
        }
    }

    /// Whether an entry keeps only the state at its end.
    pub(crate) fn end_only(self) -> bool {
        self == Self::Latest(1)
    }

    /// The states of `states` (ascending, one per position, the entry's end
    /// last) that stay.
    pub(crate) fn keep(self, mut states: Vec<RoundState>) -> Vec<RoundState> {
        match self {
            Self::Latest(n) => {
                states.drain(..states.len().saturating_sub(n as usize));
                states
            }
            Self::Anchors { count, stride } => {
                let handback = states.iter().rposition(|s| s.tag.handback);
                let pass = states.iter().rposition(|s| !s.tag.handback);
                let mut keep = vec![false; states.len()];
                for index in handback.into_iter().chain(pass) {
                    keep[index] = true;
                }
                if let Some(pass) = pass {
                    let current = states[pass].tag.round;
                    let mut anchors = 0;
                    for index in (0..pass).rev() {
                        let tag = states[index].tag;
                        if anchors == count {
                            break;
                        }
                        if !tag.handback && tag.round < current && tag.round % stride.max(1) == 0 {
                            keep[index] = true;
                            anchors += 1;
                        }
                    }
                }
                let mut index = 0;
                states.retain(|_| {
                    index += 1;
                    keep[index - 1]
                });
                states
            }
        }
    }

    /// Of `states` (ascending, at or before a resume point whose state is
    /// `resume`), those still kept after the round resuming there adds its
    /// `added` states: its pass's end, and with 2 a handed-back decode's end.
    pub(crate) fn survivors(
        self,
        states: &[RoundState],
        resume: StateTag,
        added: u32,
    ) -> Vec<RoundState> {
        let round = StateTag {
            round: resume.round + 1,
            handback: false,
        };
        let mut all = states.to_vec();
        all.push(RoundState::new(u64::MAX - 1, round));
        if added >= 2 {
            all.push(RoundState::new(
                u64::MAX,
                StateTag {
                    handback: true,
                    ..round
                },
            ));
        }
        let mut kept = self.keep(all);
        kept.retain(|state| state.at < u64::MAX - 1);
        kept
    }
}

/// Which state a retained context's end is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndState {
    /// The end of a pass this engine ran, one round past the state its
    /// request resumed from.
    Pass,
    /// A decode run elsewhere handed the context back: the end of the round
    /// whose pass ended at the shared tokens.
    HandBack,
    /// A slower tier's state, read back with its tag.
    Read(StateTag),
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
    /// The state at `tokens`.
    end: StateTag,
    /// Round-end states kept beside the one at `tokens`, ascending, each
    /// below `tokens`. Empty unless the cache keeps more than one state per
    /// entry.
    earlier_states: Vec<RoundState>,
}

impl PrefixEntry {
    /// Its states, the end's last.
    fn states(&self) -> impl Iterator<Item = RoundState> + '_ {
        self.earlier_states
            .iter()
            .copied()
            .chain(std::iter::once(RoundState::new(self.tokens, self.end)))
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
    /// An evicted entry's states below `entry_tokens_before`, which leave
    /// with it; empty otherwise.
    pub(crate) earlier_states: Vec<RoundState>,
    /// An evicted entry's state at `entry_tokens_before`.
    pub(crate) end: StateTag,
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
    /// The states below the context's end the entry keeps, or would have
    /// kept had it fit.
    pub(crate) earlier_states: Vec<RoundState>,
    /// The state at the context's end.
    pub(crate) end: StateTag,
}

/// The states a running request carries from the session's entry it took
/// ([`PrefixCache::take`]), and the tag of the state it resumed from (`None`:
/// it resumed from none).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Carried {
    pub(crate) states: Vec<RoundState>,
    pub(crate) resume: Option<StateTag>,
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
    /// Which round-end states an entry keeps (`Latest(1)`: the end only).
    retention: StateRetention,
    /// States kept: one at each entry's end, every entry's earlier ones, and
    /// those requests carry ([`Self::take`]).
    states: u64,
    entries: HashMap<u32, PrefixEntry>,
    /// The states a running request of the session carries from the entry it
    /// took, until its context returns ([`Self::take_carried`]).
    carried: HashMap<u32, Carried>,
    /// `entries` ordered by the policy's eviction key, session id last, so the
    /// victim is the first element. A retained entry's key never changes (a
    /// hit removes it), so only insert and remove touch this.
    eviction_order: BTreeSet<(u64, u64, u32)>,
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
            retention: StateRetention::Latest(1),
            states: 0,
            entries: HashMap::new(),
            carried: HashMap::new(),
            eviction_order: BTreeSet::new(),
            next_sequence: 0,
        }
    }

    /// State only where a pass over a context ended: every entry keeps the
    /// `charge` state at its end and nothing inside, so it is reused whole by
    /// a prompt that extends it, or not at all; with `states` > 1 it also
    /// keeps up to `states - 1` earlier round-end states, all of them one
    /// unit with the context (module docs).
    /// Replaces the snapshot quantum and charge.
    pub(crate) fn with_end_state(mut self, charge: u64, retention: StateRetention) -> Self {
        assert!(self.entries.is_empty(), "set the end state before use");
        assert!(retention.max_states() >= 1, "an entry keeps at least its end state");
        self.quantum_tokens = 1;
        self.resume = ResumePoint::ContextEnd;
        self.charge_per_checkpoint = 0;
        self.charge_per_entry = charge;
        self.retention = retention;
        self
    }

    /// Where a prompt sharing `declared` tokens with `entry` resumes.
    fn resume_tokens(&self, entry: &PrefixEntry, declared: u64) -> u64 {
        self.resume_state(entry, declared).map_or(0, |state| state.at)
    }

    /// The state a prompt sharing `declared` tokens with `entry` resumes
    /// from (the end's tag for a resume that is not at a state).
    fn resume_state(&self, entry: &PrefixEntry, declared: u64) -> Option<RoundState> {
        let end = self.resume.reusable_tokens(entry.tokens, declared);
        if end > 0 {
            return Some(RoundState::new(end, entry.end));
        }
        entry
            .earlier_states
            .iter()
            .rev()
            .copied()
            .find(|state| state.at <= declared)
    }

    /// Context tokens held. Deliberately test-only: every production decision
    /// is about occupancy, which is [`Self::used_charge`]. The two differ once
    /// retained snapshots carry a charge of their own.
    #[cfg(test)]
    pub(crate) fn used_tokens(&self) -> u64 {
        self.used_tokens
    }

    /// What this tier actually occupies: retained context KV plus one charge per
    /// retained snapshot or state. Equals [`Self::used_tokens`] when there is
    /// neither charge.
    pub(crate) fn used_charge(&self) -> u64 {
        self.used_tokens
            + self.used_checkpoints * self.charge_per_checkpoint
            + self.states * self.charge_per_entry
    }

    /// Occupancy a quantized `tokens` would add with `earlier` states below
    /// its end: the context KV plus one charge for each snapshot inside it
    /// and for each state.
    fn charge_of(&self, tokens: u64, earlier: usize) -> u64 {
        let states = if tokens > 0 { 1 + earlier as u64 } else { 0 };
        tokens
            + tokens / self.quantum_tokens * self.charge_per_checkpoint
            + states * self.charge_per_entry
    }

    /// The largest quantized length not exceeding `tokens` whose charge fits
    /// `capacity`. Each retained snapshot costs `quantum + charge_per_checkpoint`,
    /// so the count is a plain division; at quantum one with no charge this is
    /// `min(tokens, capacity)`, the pre-hybrid rule. A context resumable only
    /// at its end is kept whole, with its `earlier` states, or not at all: a
    /// cut-off part has no state.
    fn largest_fitting_length(&self, tokens: u64, earlier: usize, capacity: u64) -> u64 {
        if self.resume == ResumePoint::ContextEnd {
            return if self.charge_of(tokens, earlier) <= capacity {
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
        self.largest_fitting_length(tokens, 0, u64::MAX)
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
    /// separately reusable page range. With round-end states, the request
    /// carries the latest `k - 1` at or before its resume point, still
    /// charged here, until its context returns (module docs).
    pub(crate) fn take(&mut self, session_id: u32, requested_tokens: u32) -> PrefixCacheTakeResult {
        let cache_used_before = self.used_tokens;
        let hit_tokens = self.peek(session_id, requested_tokens);
        let removed_entry = self.remove(session_id);
        let entry_tokens = removed_entry.as_ref().map(|entry| entry.tokens).unwrap_or(0);
        let return_metadata = removed_entry.as_ref().map(|entry| PrefixCacheReturnMetadata {
            insertion_sequence: entry.insertion_sequence,
            frequency: entry.frequency.saturating_add(1),
        });
        if let Some(entry) = removed_entry.filter(|_| !self.retention.end_only() && hit_tokens > 0)
        {
            // The request leaves a state at its own end, so these keep room for it.
            let resume = self
                .resume_state(&entry, u64::from(requested_tokens))
                .expect("a hit resumes from a state");
            let below: Vec<RoundState> = entry
                .states()
                .filter(|state| state.at > 0 && state.at <= u64::from(hit_tokens))
                .collect();
            let states = self.retention.survivors(&below, resume.tag, 1);
            self.states += states.len() as u64;
            let carried = Carried {
                states,
                resume: Some(resume.tag),
            };
            if let Some(old) = self.carried.insert(session_id, carried) {
                self.states -= old.states.len() as u64;
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

    /// The states a running request of the session carries ([`Self::take`]),
    /// out of this cache's charge: they return with its context
    /// ([`Self::insert_sharing`]'s `incoming`). Empty when it carries none.
    pub(crate) fn take_carried(&mut self, session_id: u32) -> Carried {
        let carried = self.carried.remove(&session_id).unwrap_or_default();
        self.states -= carried.states.len() as u64;
        carried
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
        self.insert_sharing(
            session_id,
            tokens,
            0,
            &Carried::default(),
            EndState::Pass,
            physical_limit,
            return_metadata,
        )
    }

    /// [`Self::insert`] of a context that shares its first `shared` tokens
    /// with the session's entry and arrives with `incoming` states of its
    /// own (carried by its request, or read from a slower tier), its end
    /// being the `end` state: the entry's and the incoming states within the
    /// shared tokens that the retention keeps stay, below the new end state.
    /// The entry is one unit: kept with all of them, or not at all.
    pub(crate) fn insert_sharing(
        &mut self,
        session_id: u32,
        tokens: u64,
        shared: u64,
        incoming: &Carried,
        end: EndState,
        physical_limit: u64,
        return_metadata: Option<PrefixCacheReturnMetadata>,
    ) -> PrefixCacheInsertResult {
        let mut mutations = Vec::new();
        let limit = shared.min(tokens.saturating_sub(1));
        let (earlier, end) = if self.retention.end_only() {
            (Vec::new(), StateTag::default())
        } else {
            let mut states: Vec<RoundState> = self
                .entries
                .get(&session_id)
                .map(|entry| entry.states().filter(|s| s.at > 0 && s.at <= limit).collect())
                .unwrap_or_default();
            states.extend(incoming.states.iter().copied().filter(|s| s.at > 0 && s.at <= limit));
            normalize_states(&mut states);
            let end = match end {
                EndState::Pass => StateTag {
                    round: incoming.resume.map_or(0, |tag| tag.round + 1),
                    handback: false,
                },
                EndState::HandBack => StateTag {
                    round: states.last().map_or(0, |s| s.tag.round),
                    handback: true,
                },
                EndState::Read(tag) => tag,
            };
            states.push(RoundState::new(tokens, end));
            let mut kept = self.retention.keep(states);
            kept.pop();
            (kept, end)
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
        let retained_tokens =
            self.largest_fitting_length(tokens, earlier.len(), effective_capacity);
        if retained_tokens == 0 {
            return PrefixCacheInsertResult {
                mutations,
                retained_tokens,
                earlier_states: earlier,
                end,
            };
        }
        let retained_charge = self.charge_of(retained_tokens, earlier.len());
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
        self.add_entry(
            session_id,
            PrefixEntry {
                tokens: retained_tokens,
                insertion_sequence,
                last_access_sequence: sequence,
                frequency,
                end,
                earlier_states: earlier.clone(),
            },
        );
        mutations.push(PrefixCacheMutation {
            kind: PrefixCacheMutationKind::Retain,
            session_id,
            entry_tokens_before: 0,
            entry_tokens_after: retained_tokens,
            cache_used_before,
            cache_used_after: self.used_tokens,
            earlier_states: Vec::new(),
            end,
        });
        PrefixCacheInsertResult {
            mutations,
            retained_tokens,
            earlier_states: earlier,
            end,
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
        self.used_tokens += entry.tokens;
        self.used_checkpoints += entry.tokens / self.quantum_tokens;
        self.states += 1 + entry.earlier_states.len() as u64;
        self.entries.insert(session_id, entry);
    }

    fn remove(&mut self, session_id: u32) -> Option<PrefixEntry> {
        let entry = self.entries.remove(&session_id)?;
        let key = self.eviction_key(session_id, &entry);
        self.eviction_order.remove(&key);
        self.used_tokens = self.used_tokens.saturating_sub(entry.tokens);
        self.used_checkpoints = self
            .used_checkpoints
            .saturating_sub(entry.tokens / self.quantum_tokens);
        self.states -= 1 + entry.earlier_states.len() as u64;
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
            earlier_states: entry.earlier_states,
            end: entry.end,
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

    impl PrefixCache {
        /// [`Self::insert_sharing`] of a pass's context arriving with
        /// untagged `incoming` states.
        fn share(
            &mut self,
            session_id: u32,
            tokens: u64,
            shared: u64,
            incoming: &[u64],
            physical_limit: u64,
            return_metadata: Option<PrefixCacheReturnMetadata>,
        ) -> PrefixCacheInsertResult {
            let incoming = Carried {
                states: incoming
                    .iter()
                    .map(|&at| RoundState::new(at, StateTag::default()))
                    .collect(),
                resume: None,
            };
            self.insert_sharing(
                session_id,
                tokens,
                shared,
                &incoming,
                EndState::Pass,
                physical_limit,
                return_metadata,
            )
        }
    }

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
            PrefixCache::new(10_000, PrefixCachePolicy::Lru, 1_000, 500).with_end_state(500, StateRetention::Latest(1));
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
            PrefixCache::new(10_000, PrefixCachePolicy::Lru, 1_000, 500).with_end_state(500, StateRetention::Latest(1));
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
        PrefixCache::new(capacity, PrefixCachePolicy::Lru, 1, 0).with_end_state(500, StateRetention::Latest(states))
    }

    #[test]
    fn earlier_round_end_states_serve_a_prompt_that_ends_inside_the_context() {
        let mut cache = round_end_cache(100_000, 3);
        cache.insert(1, 1_000, 100_000, None);
        // The next round extends it (shares 1_000); a decode hands back more.
        cache.share(1, 1_500, 1_000, &[], 100_000, None);
        cache.share(1, 2_000, 1_500, &[], 100_000, None);
        assert_eq!(cache.used_charge(), 2_000 + 3 * 500);
        assert_eq!(cache.peek(1, 2_001), 2_000);
        assert_eq!(cache.peek(1, 1_999), 1_500);
        assert_eq!(cache.peek(1, 1_200), 1_000);
        assert_eq!(cache.peek(1, 999), 0);
        // A fourth state pushes the oldest out.
        cache.share(1, 2_500, 2_000, &[], 100_000, None);
        assert_eq!(cache.peek(1, 1_200), 0);
        assert_eq!(cache.used_charge(), 2_500 + 3 * 500);
    }

    #[test]
    fn a_truncating_round_keeps_only_the_states_before_the_cut() {
        let mut cache = round_end_cache(100_000, 4);
        cache.insert(1, 1_000, 100_000, None);
        cache.share(1, 2_000, 1_000, &[], 100_000, None);
        // A context that shares only 1_500 tokens: the state at 2_000 is on
        // another branch.
        cache.share(1, 1_800, 1_500, &[], 100_000, None);
        assert_eq!(cache.peek(1, 1_799), 1_000);
        assert_eq!(cache.used_charge(), 1_800 + 2 * 500);
    }

    #[test]
    fn a_taken_entry_carries_its_states_back_with_its_context() {
        let mut cache = round_end_cache(100_000, 3);
        cache.insert(1, 1_000, 100_000, None);
        cache.share(1, 2_000, 1_000, &[], 100_000, None);
        // The round resumes at 1_000 (its prompt ends at 1_200): the request
        // takes the KV and carries the state at 1_000, still charged here
        // but not an entry.
        let take = cache.take(1, 1_200);
        assert_eq!(take.hit_tokens, 1_000);
        assert_eq!(cache.used_tokens(), 0);
        assert_eq!(cache.used_charge(), 500);
        assert_eq!(cache.peek(1, 1_200), 0, "no entry while the request runs");
        assert!(cache.shrink_to(0).is_empty(), "a carried state is not evictable");
        // Its context returns with the carried state below its own end.
        let carried = positions(&cache.take_carried(1).states);
        assert_eq!(carried, [1_000]);
        assert_eq!(cache.used_charge(), 0);
        cache.share(1, 1_300, 1_200, &carried, 100_000, take.return_metadata);
        assert_eq!(cache.used_charge(), 1_300 + 2 * 500);
        assert_eq!(cache.peek(1, 1_100), 1_000);
        // Resuming at the end carries the latest k - 1 states, the end's
        // included; the returning context appends its own and keeps 3.
        cache.take(1, 1_301);
        let carried = positions(&cache.take_carried(1).states);
        assert_eq!(carried, [1_000, 1_300]);
        cache.share(1, 1_600, 1_301, &carried, 100_000, None);
        assert_eq!(cache.peek(1, 1_350), 1_300);
        assert_eq!(cache.peek(1, 1_100), 1_000);
        assert_eq!(cache.used_charge(), 1_600 + 3 * 500);
    }

    #[test]
    fn eviction_takes_whole_entries_with_all_their_states() {
        let mut cache = round_end_cache(4_000, 2);
        cache.insert(1, 1_000, 4_000, None);
        cache.share(1, 1_200, 1_000, &[], 4_000, None); // 1_200 + 2 x 500
        cache.insert(2, 1_000, 4_000, None); // 1_000 + 500
        assert_eq!(cache.used_charge(), 3_700);
        // Session 3 needs 800: the LRU entry, session 1, goes whole, its
        // earlier state with it (for a victim tier behind HBM).
        let result = cache.insert(3, 300, 4_000, None);
        let evicted: Vec<_> = result
            .mutations
            .iter()
            .map(|m| (m.kind, m.session_id, m.entry_tokens_before, positions(&m.earlier_states)))
            .collect();
        assert_eq!(
            evicted,
            [
                (PrefixCacheMutationKind::ReplacementPolicyEviction, 1, 1_200, vec![1_000]),
                (PrefixCacheMutationKind::Retain, 3, 0, vec![])
            ]
        );
        assert_eq!(cache.peek(1, 1_200), 0);
        assert_eq!(cache.peek(2, 1_000), 1_000);
        assert_eq!(cache.used_charge(), 1_500 + 800);
    }

    #[test]
    fn an_entry_is_kept_with_all_its_states_or_not_at_all() {
        let mut cache = round_end_cache(2_500, 3);
        cache.insert(1, 1_000, 2_500, None);
        // 1_200 + 2 x 500 = 2_200 fits; 1_400 + 3 x 500 does not, and no
        // state is dropped to make it fit.
        cache.share(1, 1_200, 1_000, &[], 2_500, None);
        let result = cache.share(1, 1_400, 1_200, &[], 2_500, None);
        assert_eq!(result.retained_tokens, 0);
        assert_eq!(positions(&result.earlier_states), [1_000, 1_200]);
        assert_eq!(cache.used_charge(), 0);
    }

    #[test]
    fn one_state_per_entry_keeps_no_earlier_state() {
        let mut cache = round_end_cache(100_000, 1);
        cache.insert(1, 1_000, 100_000, None);
        cache.share(1, 2_000, 1_000, &[1_000], 100_000, None);
        assert_eq!(cache.peek(1, 1_500), 0);
        assert_eq!(cache.used_charge(), 2_500);
        cache.take(1, 2_001);
        assert_eq!(cache.used_charge(), 0);
        assert!(cache.take_carried(1).states.is_empty());
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

    /// State-at-end entries under anchor retention, a state costing 500.
    fn anchor_cache(count: u32, stride: u32) -> PrefixCache {
        PrefixCache::new(1_000_000, PrefixCachePolicy::Lru, 1, 0)
            .with_end_state(500, StateRetention::Anchors { count, stride })
    }

    /// One pipeline round with a hand-back: the request takes the entry
    /// (sharing `declared`), retains its prompt's context to `prompt_end`,
    /// and the decode run elsewhere hands back 10 more tokens.
    fn hand_back_round(cache: &mut PrefixCache, declared: u64, prompt_end: u64) {
        cache.take(1, declared as u32);
        let carried = cache.take_carried(1);
        cache.insert_sharing(1, prompt_end, declared, &carried, EndState::Pass, 1_000_000, None);
        let none = Carried::default();
        cache.insert_sharing(1, prompt_end + 10, prompt_end, &none, EndState::HandBack, 1_000_000, None);
    }

    /// The session's entry's states: (position, round, hand-back).
    fn entry_states(cache: &PrefixCache) -> Vec<(u64, u32, bool)> {
        cache.entries[&1]
            .states()
            .map(|s| (s.at, s.tag.round, s.tag.handback))
            .collect()
    }

    #[test]
    fn stride_two_anchors_keep_a_prefill_end_one_to_two_rounds_back() {
        // Round r's prompt ends at 1_000 (r + 1), its decode at 10 later; the
        // next round extends it.
        let mut cache = anchor_cache(1, 2);
        let mut after = Vec::new();
        for round in 0..6u64 {
            let declared = if round == 0 { 0 } else { 1_000 * round + 11 };
            hand_back_round(&mut cache, declared, 1_000 * (round + 1));
            after.push(entry_states(&cache));
        }
        assert_eq!(
            after,
            [
                vec![(1_000, 0, false), (1_010, 0, true)],
                // The current prefill end and round 0's ("-2" is round 0 here).
                vec![(1_000, 0, false), (2_000, 1, false), (2_010, 1, true)],
                vec![(1_000, 0, false), (3_000, 2, false), (3_010, 2, true)],
                // Round 2 is recorded once round 3 runs: one round back.
                vec![(3_000, 2, false), (4_000, 3, false), (4_010, 3, true)],
                vec![(3_000, 2, false), (5_000, 4, false), (5_010, 4, true)],
                vec![(5_000, 4, false), (6_000, 5, false), (6_010, 5, true)],
            ]
        );
        // Two anchors at stride 2 keep rounds 2 and 4; stride 1 is the latest
        // earlier prefill ends.
        for (count, stride, anchors) in [(2, 2, [3_000, 5_000]), (2, 1, [4_000, 5_000])] {
            let mut cache = anchor_cache(count, stride);
            for round in 0..6u64 {
                let declared = if round == 0 { 0 } else { 1_000 * round + 11 };
                hand_back_round(&mut cache, declared, 1_000 * (round + 1));
            }
            let at: Vec<u64> = entry_states(&cache).iter().map(|s| s.0).collect();
            assert_eq!(at, [anchors[0], anchors[1], 6_000, 6_010], "m{count}s{stride}");
        }
    }

    #[test]
    fn an_inside_round_resumes_from_the_deepest_anchor_at_or_before_its_prefix() {
        let mut cache = anchor_cache(1, 2);
        for round in 0..6u64 {
            let declared = if round == 0 { 0 } else { 1_000 * round + 11 };
            hand_back_round(&mut cache, declared, 1_000 * (round + 1));
        }
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
        for round in 0..3u64 {
            let declared = if round == 0 { 0 } else { 1_000 * round + 11 };
            hand_back_round(&mut cache, declared, 1_000 * (round + 1));
        }
        // 3_010 tokens and states at 1_000, 3_000 and 3_010.
        assert_eq!(cache.used_charge(), 3_010 + 3 * 500);
        // Round 3 takes the entry: it carries the states that survive its
        // prefill (the decode end at 3_010 and the round-2 anchor at
        // 3_000), still charged, while its own state is live.
        cache.take(1, 3_011);
        assert_eq!(cache.used_charge(), 2 * 500);
        let carried = cache.take_carried(1);
        assert_eq!(positions(&carried.states), [3_000, 3_010]);
        assert_eq!(carried.resume.map(|tag| tag.round), Some(2));
        cache.insert_sharing(1, 4_000, 3_011, &carried, EndState::Pass, 1_000_000, None);
        assert_eq!(cache.used_charge(), 4_000 + 3 * 500);
    }
}
