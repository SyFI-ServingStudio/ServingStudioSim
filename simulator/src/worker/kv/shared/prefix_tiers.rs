//! Host DRAM and SSD tiers behind the HBM retained-prefix cache.
//!
//! A tier is a per-replica LRU of session contexts, sized and loaded per GPU:
//! every GPU of the replica keeps its own slice of each token (a pipeline stage
//! its layers' KV), so capacity and load time are one GPU's bytes over one
//! GPU's capacity and link, and all GPUs load their slices in parallel.
//!
//! By default writes are write-through, as LMCache and vLLM's offloading
//! connector store a request's KV when it finishes: every tier receives the
//! session's latest context, and each evicts on its own. A tier never holds an
//! older context than a faster one, so a lookup takes the fastest tier with the
//! longest hit.
//!
//! A tier may instead be the victim tier of the one above it
//! ([`PrefixTierWrites::FasterTierEvictions`]; above the fastest tier is the
//! HBM prefix cache): it receives a session's context only when the tier
//! above evicts it (or cannot hold it), as what its entry for the session
//! lacks; an entry that already holds it all is only refreshed. A chain of
//! victim tiers (HBM -> DRAM -> SSD) is exclusive at the moment of writing and
//! inclusive after reads: a read back into HBM leaves the read tier's entry in
//! place, a clean copy, refreshed, and copies into no faster victim tier (HBM
//! evicting it again lands it there). A read passes through a faster
//! write-through tier, which keeps a copy. An entry the session's next round
//! does not extend is cut to what that round shares with it, in every victim
//! tier, so it stays a prefix of the session's latest context.
//!
//! Loads queue FIFO on one channel per tier (its read bandwidth), from the
//! moment admission reaches the request (`admission/prefix_fetch.rs`). Writes
//! are free: their bandwidth is not modeled, only counted, also per 10-s
//! window of the last time admission passed the tiers ([`PrefixTiers::set_clock`]).
//!
//! Units are KV tokens of the most loaded GPU (the pipeline's
//! `kv_bytes_per_token`). A hybrid model's entry also charges one recurrent
//! state: a tier keeps the state at the context's end, the only one a session
//! resumes from, not the per-block snapshots the HBM tier holds. How much of an
//! entry a lookup reuses follows the HBM tier's [`ResumePoint`].
//!
//! With [`PrefixTiers::with_round_end_states`] `k` > 1, an entry of a context
//! resumable only at a state keeps the latest `k` round-end states of the
//! context (where its prefills and handed-back decodes ended), each charged
//! one state, and a lookup resumes from the latest kept state at or before the
//! prompt's end. The rule is HBM's (`prefix_cache.rs`): an entry is one unit,
//! its KV and all its states. A tier evicts only whole least recently used
//! entries and drops no state alone; whatever moves an entry moves all its
//! states: an eviction into a victim tier, a read into HBM (it moves the
//! states at or before the hit), an exclusive tier's release (the states are
//! in HBM with the context), and every write counts each state it adds. A
//! victim tier merges what it receives into its own entry for the session
//! (both are prefixes of the session's latest context): the longer context
//! with the latest `k` states among both. A cut keeps the states at or before
//! it.

use std::collections::{BTreeMap, HashMap};

use crate::common::Time;

use super::prefix_cache::ResumePoint;

/// One tier's size and read bandwidth, per GPU, and what it is written.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrefixTierSpec {
    pub name: &'static str,
    pub capacity_gb_per_gpu: f64,
    pub read_gb_per_s_per_gpu: f64,
    pub writes: PrefixTierWrites,
}

/// What a tier is written.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrefixTierWrites {
    /// Every finished context, as it finishes (write-through).
    #[default]
    EveryContext,
    /// Only the contexts the tier above it evicts, or cannot hold, as they
    /// leave it (a victim tier); above the fastest tier is the HBM prefix
    /// cache. Configured as `dram_tier_writes: hbm-evictions` and
    /// `ssd_tier_writes: dram-evictions`.
    FasterTierEvictions,
    /// A victim tier that moves a read back into HBM instead of keeping a
    /// clean copy: the read releases the entry, so the tier holds only what
    /// HBM lacks (strictly exclusive), and HBM evicting the context again
    /// writes it whole. The read moved the entry's states into HBM with its
    /// context. Configured as `dram_tier_writes: hbm-evictions-exclusive`.
    FasterTierEvictionsMoved,
}

impl PrefixTierWrites {
    /// Whether the tier is the victim tier of the one above it.
    pub fn is_victim(self) -> bool {
        self != Self::EveryContext
    }
}

/// A finished round's context, as a write-through tier stores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundContext {
    /// Context tokens with KV; the state at their end.
    pub tokens: u64,
    /// Where the round's prompt ended: the state its prefill computed, below
    /// `tokens` when a decode run elsewhere handed back more.
    pub prompt_end: u64,
    /// Tokens it shares with the session's previous context: its declared
    /// prefix.
    pub shared: u64,
}

/// Where a lookup found a session's context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefixTierHit {
    pub(crate) tier: usize,
    pub(crate) tokens: u32,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefixTierCounters {
    pub(crate) stored_entries: u64,
    /// Tokens newly written: each store writes only what the session's
    /// previous entry in this tier did not already hold.
    pub(crate) written_tokens: u64,
    pub(crate) evicted_entries: u64,
    pub(crate) evicted_tokens: u64,
    pub(crate) loads: u64,
    pub(crate) loaded_tokens: u64,
    /// States newly written (an entry's end or an earlier round-end state it
    /// did not hold).
    pub(crate) written_states: u64,
    /// States reads moved into HBM.
    pub(crate) loaded_states: u64,
    /// Entries a read moved into HBM ([`PrefixTierWrites::FasterTierEvictionsMoved`]).
    pub(crate) released_entries: u64,
}

/// What happened to a tier's entry for a session, for the optional event log
/// ([`PrefixTiers::record_events`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefixTierEventKind {
    /// The entry now holds `tokens`; it held `prev_tokens` before.
    Write,
    /// LRU eviction of the entry's `tokens`.
    Evict,
    /// A context larger than the tier: the entry's `prev_tokens` are dropped
    /// and nothing is written.
    TooLarge,
    /// The session's next round cut the entry from `prev_tokens` to `tokens`
    /// (`0`: dropped); it shares `declared` tokens with the context.
    Cut,
    /// A read of the entry's first `tokens`; `prev_tokens` of them HBM held.
    Read,
    /// A read moved the entry's `tokens` into HBM and released it.
    Release,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefixTierEvent {
    pub(crate) at: Time,
    pub(crate) kind: PrefixTierEventKind,
    pub(crate) session_id: u32,
    pub(crate) tokens: u64,
    pub(crate) prev_tokens: u64,
    pub(crate) declared: u64,
}

/// The window of [`PrefixTier::window_writes`].
pub(crate) const WRITE_WINDOW: Time = Time::from_ms_u64(10_000);

/// What leaves a tier for the one behind it: a session's context of
/// `tokens` with its state at the end and its `earlier` round-end states.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Leaving {
    session_id: u32,
    tokens: u64,
    earlier: Vec<u64>,
}

struct TierEntry {
    tokens: u64,
    /// LRU stamp.
    stamp: u64,
    /// Earlier round-end states, ascending, each below `tokens`.
    earlier: Vec<u64>,
}

impl TierEntry {
    /// Its states, the end's last.
    fn states(&self) -> impl Iterator<Item = u64> + '_ {
        self.earlier.iter().copied().chain(std::iter::once(self.tokens))
    }
}

struct PrefixTier {
    spec: PrefixTierSpec,
    capacity: u64,
    used: u64,
    entries: HashMap<u32, TierEntry>,
    lru: BTreeMap<u64, u32>,
    earlier_states: u64,
    next_stamp: u64,
    /// When the read channel is next free.
    channel_free: Time,
    /// One charged token's bytes, and the read bandwidth in bytes per second.
    bytes_per_token: f64,
    read_bytes_per_s: f64,
    counters: PrefixTierCounters,
    /// Per [`WRITE_WINDOW`] of the clock: (entries stored, tokens written).
    window_writes: Vec<(u64, u64)>,
    /// Per [`WRITE_WINDOW`] of the clock: states written.
    window_states: Vec<u64>,
    clock: Time,
    events: Option<Vec<PrefixTierEvent>>,
}

impl PrefixTier {
    fn log(
        &mut self,
        kind: PrefixTierEventKind,
        session_id: u32,
        tokens: u64,
        prev_tokens: u64,
        declared: u64,
    ) {
        if let Some(events) = &mut self.events {
            events.push(PrefixTierEvent {
                at: self.clock,
                kind,
                session_id,
                tokens,
                prev_tokens,
                declared,
            });
        }
    }

    fn count_window_states(&mut self, states: u64) {
        let window = (self.clock.0 / WRITE_WINDOW.0) as usize;
        if self.window_states.len() <= window {
            self.window_states.resize(window + 1, 0);
        }
        self.window_states[window] += states;
    }

    /// `tokens` of KV and `states` states.
    fn charge(&self, tokens: u64, states: usize, state_tokens: u64) -> u64 {
        tokens + states as u64 * state_tokens
    }

    fn touch(&mut self, session_id: u32) {
        if let Some(entry) = self.entries.get_mut(&session_id) {
            self.lru.remove(&entry.stamp);
            entry.stamp = self.next_stamp;
            self.lru.insert(self.next_stamp, session_id);
            self.next_stamp += 1;
        }
    }

    /// Cut the session's entry to its first `tokens` (one of its states when
    /// it keeps earlier ones), or drop it if that leaves none.
    fn truncate(&mut self, session_id: u32, tokens: u64, state_tokens: u64) {
        let Some(entry) = self.entries.get_mut(&session_id) else {
            return;
        };
        if tokens >= entry.tokens {
            return;
        }
        if tokens == 0 {
            self.remove(session_id, state_tokens);
            return;
        }
        // States past the cut go; one at the cut becomes the end.
        let before = entry.earlier.len();
        entry.earlier.retain(|&at| at < tokens);
        // The cut is at one of the dropped states, which becomes the end in
        // place of the old one: `dropped` states leave.
        let dropped = before - entry.earlier.len();
        self.used -= (entry.tokens - tokens) + dropped as u64 * state_tokens;
        self.earlier_states -= dropped as u64;
        entry.tokens = tokens;
    }

    fn remove(&mut self, session_id: u32, state_tokens: u64) -> Option<TierEntry> {
        let entry = self.entries.remove(&session_id)?;
        self.lru.remove(&entry.stamp);
        self.earlier_states -= entry.earlier.len() as u64;
        self.used -= self.charge(entry.tokens, 1 + entry.earlier.len(), state_tokens);
        Some(entry)
    }

    /// Evict the least recently used entry whole, its earlier states with it,
    /// and return it as a context that leaves the tier.
    fn evict_lru(&mut self, state_tokens: u64) -> Option<Leaving> {
        let victim = *self.lru.values().next()?;
        let entry = self.remove(victim, state_tokens).expect("indexed entry");
        self.counters.evicted_entries += 1;
        self.counters.evicted_tokens += entry.tokens;
        self.log(PrefixTierEventKind::Evict, victim, entry.tokens, entry.tokens, 0);
        Some(Leaving {
            session_id: victim,
            tokens: entry.tokens,
            earlier: entry.earlier,
        })
    }

    /// Store the session's context with its `earlier` round-end states, one
    /// unit, and return what leaves the tier for it: the whole LRU entries it
    /// evicts, or the context itself when it is larger than the whole tier.
    fn store(
        &mut self,
        session_id: u32,
        tokens: u64,
        earlier: Vec<u64>,
        state_tokens: u64,
    ) -> Vec<Leaving> {
        let previous = self.remove(session_id, state_tokens);
        let previous_tokens = previous.as_ref().map_or(0, |e| e.tokens);
        let charge = self.charge(tokens, 1 + earlier.len(), state_tokens);
        if charge > self.capacity {
            self.log(PrefixTierEventKind::TooLarge, session_id, 0, previous_tokens, 0);
            return vec![Leaving {
                session_id,
                tokens,
                earlier,
            }];
        }
        let mut left = Vec::new();
        while self.used + charge > self.capacity {
            left.push(
                self.evict_lru(state_tokens)
                    .expect("a full tier has an entry to evict"),
            );
        }
        let written_states = std::iter::once(tokens)
            .chain(earlier.iter().copied())
            .filter(|at| !previous.as_ref().is_some_and(|e| e.states().any(|x| x == *at)))
            .count() as u64;
        self.used += charge;
        self.earlier_states += earlier.len() as u64;
        self.entries.insert(
            session_id,
            TierEntry {
                tokens,
                stamp: self.next_stamp,
                earlier,
            },
        );
        self.lru.insert(self.next_stamp, session_id);
        self.next_stamp += 1;
        self.counters.stored_entries += 1;
        self.counters.written_tokens += tokens.saturating_sub(previous_tokens);
        self.counters.written_states += written_states;
        let window = (self.clock.0 / WRITE_WINDOW.0) as usize;
        if self.window_writes.len() <= window {
            self.window_writes.resize(window + 1, (0, 0));
        }
        self.window_writes[window].0 += 1;
        self.window_writes[window].1 += tokens.saturating_sub(previous_tokens);
        self.count_window_states(written_states);
        self.log(PrefixTierEventKind::Write, session_id, tokens, previous_tokens, 0);
        left
    }
}

/// The DRAM/SSD tiers of one replica, fastest first.
pub(crate) struct PrefixTiers {
    tiers: Vec<PrefixTier>,
    /// One recurrent state in KV tokens (0 for a full-attention model).
    state_tokens: u64,
    /// Where an entry can be resumed from.
    resume: ResumePoint,
    /// Round-end states an entry keeps, its end's included.
    max_states: usize,
}

impl PrefixTiers {
    /// `kv_bytes_per_token`: one GPU's bytes per token; `state_tokens`: one
    /// recurrent state in those tokens; `resume`: where an entry can be
    /// resumed from.
    pub(crate) fn new(
        specs: &[PrefixTierSpec],
        kv_bytes_per_token: u64,
        state_tokens: u64,
        resume: ResumePoint,
    ) -> Self {
        let bytes = kv_bytes_per_token.max(1) as f64;
        let tiers = specs
            .iter()
            .map(|&spec| {
                assert!(
                    spec.capacity_gb_per_gpu > 0.0 && spec.read_gb_per_s_per_gpu > 0.0,
                    "prefix tier {} needs a positive size and bandwidth",
                    spec.name
                );
                PrefixTier {
                    spec,
                    capacity: (spec.capacity_gb_per_gpu * 1e9 / bytes) as u64,
                    used: 0,
                    entries: HashMap::new(),
                    lru: BTreeMap::new(),
                    earlier_states: 0,
                    next_stamp: 0,
                    channel_free: Time::ZERO,
                    bytes_per_token: bytes,
                    read_bytes_per_s: spec.read_gb_per_s_per_gpu * 1e9,
                    counters: PrefixTierCounters::default(),
                    window_writes: Vec::new(),
                    window_states: Vec::new(),
                    clock: Time::ZERO,
                    events: None,
                }
            })
            .collect();
        Self {
            tiers,
            state_tokens,
            resume,
            max_states: 1,
        }
    }

    /// Keep up to `states` round-end states per entry, its end's included
    /// (module docs); only for a context resumable only at a state.
    pub(crate) fn with_round_end_states(mut self, states: u32) -> Self {
        assert!(states >= 1, "an entry keeps at least its end state");
        assert!(
            states == 1 || self.resume == ResumePoint::ContextEnd,
            "earlier round-end states need a state-at-end context"
        );
        self.max_states = states as usize;
        self
    }

    /// Where a prompt sharing `declared` tokens with `entry` resumes.
    fn resume_tokens(&self, entry: &TierEntry, declared: u64) -> u64 {
        let end = self.resume.reusable_tokens(entry.tokens, declared);
        if end > 0 {
            return end;
        }
        entry
            .earlier
            .iter()
            .rev()
            .copied()
            .find(|&at| at <= declared)
            .unwrap_or(0)
    }

    /// The earlier states a new entry for the session in `tier` keeps below
    /// its end: the old entry's states (its end's included) and `extra`'s, those
    /// at or before `limit`, the latest `max_states - 1` of them, ascending.
    fn earlier_for(
        &self,
        tier: usize,
        session_id: u32,
        limit: u64,
        extra: impl IntoIterator<Item = u64>,
    ) -> Vec<u64> {
        if self.max_states <= 1 {
            return Vec::new();
        }
        let mut states: Vec<u64> = self.tiers[tier]
            .entries
            .get(&session_id)
            .into_iter()
            .flat_map(TierEntry::states)
            .chain(extra)
            .filter(|&at| at > 0 && at <= limit)
            .collect();
        states.sort_unstable();
        states.dedup();
        states.drain(..states.len().saturating_sub(self.max_states - 1));
        states
    }

    pub(crate) fn tier_name(&self, tier: usize) -> &'static str {
        self.tiers[tier].spec.name
    }

    pub(crate) fn len(&self) -> usize {
        self.tiers.len()
    }

    /// The fastest tier whose entry for the session a prompt sharing
    /// `requested` tokens with it reuses the most of.
    pub(crate) fn lookup(&self, session_id: u32, requested: u32) -> Option<PrefixTierHit> {
        let mut best: Option<PrefixTierHit> = None;
        for (tier, state) in self.tiers.iter().enumerate() {
            if let Some(entry) = state.entries.get(&session_id) {
                let tokens = self.resume_tokens(entry, u64::from(requested)) as u32;
                if tokens > 0 && best.is_none_or(|hit| tokens > hit.tokens) {
                    best = Some(PrefixTierHit { tier, tokens });
                }
            }
        }
        best
    }

    /// The states a read of `hit` moves below its end: the entry's states
    /// before the hit, ascending (the state at the hit itself moves too).
    pub(crate) fn read_states(&self, session_id: u32, hit: PrefixTierHit) -> Vec<u64> {
        self.tiers[hit.tier]
            .entries
            .get(&session_id)
            .map(|entry| {
                entry
                    .states()
                    .filter(|&at| at < u64::from(hit.tokens))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// What a read of `hit` moves, in tier tokens: the KV past
    /// `resident_tokens` (what HBM still holds), the state at the hit and
    /// every state of the entry before it.
    pub(crate) fn read_charge(
        &self,
        session_id: u32,
        hit: PrefixTierHit,
        resident_tokens: u32,
    ) -> u64 {
        let states = 1 + self.read_states(session_id, hit).len();
        u64::from(hit.tokens.saturating_sub(resident_tokens)) + states as u64 * self.state_tokens
    }

    /// Queue a read of `hit` behind the tier's earlier reads; returns when it
    /// lands and the states it moves below the hit ([`Self::read_charge`]).
    /// The entry stays (the tiers are inclusive), or an exclusive tier
    /// releases it whole; every faster write-through tier receives a copy,
    /// since the read passes through it. A faster victim tier receives it
    /// only when the tier above it evicts it.
    pub(crate) fn load(
        &mut self,
        session_id: u32,
        hit: PrefixTierHit,
        resident_tokens: u32,
        now: Time,
    ) -> (Time, Vec<u64>) {
        let states = self.read_states(session_id, hit);
        let charge = self.read_charge(session_id, hit, resident_tokens);
        let state_tokens = self.state_tokens;
        let tier = &mut self.tiers[hit.tier];
        let start = tier.channel_free.max(now);
        let ready =
            start + Time::from_s(charge as f64 * tier.bytes_per_token / tier.read_bytes_per_s);
        tier.channel_free = ready;
        tier.counters.loads += 1;
        tier.counters.loaded_tokens += u64::from(hit.tokens.saturating_sub(resident_tokens));
        tier.counters.loaded_states += 1 + states.len() as u64;
        tier.touch(session_id);
        tier.log(
            PrefixTierEventKind::Read,
            session_id,
            u64::from(hit.tokens),
            u64::from(resident_tokens),
            0,
        );
        if self.tiers[hit.tier].spec.writes == PrefixTierWrites::FasterTierEvictionsMoved {
            let tier = &mut self.tiers[hit.tier];
            // The read moved the context and its states into HBM.
            let entry = tier.remove(session_id, state_tokens).expect("a hit has an entry");
            tier.counters.released_entries += 1;
            tier.log(PrefixTierEventKind::Release, session_id, entry.tokens, entry.tokens, 0);
        }
        for faster in 0..hit.tier {
            if self.tiers[faster].spec.writes == PrefixTierWrites::EveryContext {
                // The copy is a prefix of the session's latest context, as is
                // the faster tier's own entry: the latest states of both stay.
                let tokens = u64::from(hit.tokens);
                let earlier = self.earlier_for(
                    faster,
                    session_id,
                    tokens.saturating_sub(1),
                    states.iter().copied(),
                );
                self.store_in(faster, session_id, tokens, earlier);
            }
        }
        (ready, states)
    }

    /// How long a read queued on `tier` now would wait for its channel.
    pub(crate) fn queue_wait(&self, tier: usize, now: Time) -> Time {
        Time(self.tiers[tier].channel_free.0.saturating_sub(now.0))
    }

    /// Put a context that predates the run into the slowest tier only, where
    /// a long-lived session's context would have settled.
    pub(crate) fn seed(&mut self, session_id: u32, tokens: u64) {
        let state_tokens = self.state_tokens;
        if let Some(tier) = self.tiers.last_mut() {
            // The slowest tier has no tier behind it.
            let _ = tier.store(session_id, tokens, Vec::new(), state_tokens);
        }
    }

    /// A finished context of `tokens` with no earlier states (tests).
    #[cfg(test)]
    pub(crate) fn store(&mut self, session_id: u32, tokens: u64) {
        self.store_round(
            session_id,
            RoundContext {
                tokens,
                prompt_end: tokens,
                shared: 0,
            },
        );
    }

    /// Write-through of a finished request's context to every tier but a
    /// victim tier; what that pushes out of a tier goes down to a victim tier
    /// behind it. An entry keeping earlier round-end states keeps the old
    /// entry's within what the round shares with it, and the round's prompt
    /// end when a decode handed back more.
    pub(crate) fn store_round(&mut self, session_id: u32, round: RoundContext) {
        let prompt_end = (round.prompt_end < round.tokens).then_some(round.prompt_end);
        for tier in 0..self.tiers.len() {
            if self.tiers[tier].spec.writes == PrefixTierWrites::EveryContext {
                let limit = round.shared.min(round.tokens.saturating_sub(1));
                let mut earlier = self.earlier_for(tier, session_id, limit, None);
                if let Some(at) = prompt_end.filter(|_| self.max_states > 1) {
                    earlier.push(at);
                    earlier.sort_unstable();
                    earlier.dedup();
                    earlier.drain(..earlier.len().saturating_sub(self.max_states - 1));
                }
                self.store_in(tier, session_id, round.tokens, earlier);
            }
        }
    }

    /// Store into `tier`, and pass what leaves it to the tier behind it if
    /// that one is its victim tier.
    fn store_in(&mut self, tier: usize, session_id: u32, tokens: u64, earlier: Vec<u64>) {
        let state_tokens = self.state_tokens;
        let left = self.tiers[tier].store(session_id, tokens, earlier, state_tokens);
        self.pass_down(tier, left);
    }

    /// Hand what left `tier` to the tier behind it, if that is its victim tier.
    fn pass_down(&mut self, tier: usize, left: Vec<Leaving>) {
        for leaving in left {
            self.take_victim(tier + 1, leaving.session_id, leaving.tokens, &leaving.earlier);
        }
    }

    /// The tier above `tier` let go of the session's `tokens`-token context
    /// (with its state at the end and its `earlier` states): if `tier` is its
    /// victim tier, write what its entry lacks, or only refresh an entry that
    /// already holds it all. Both are prefixes of the session's latest
    /// context (a victim entry is cut at each round's admission), so they
    /// merge into the longer context with the latest `k` states of both.
    fn take_victim(&mut self, tier: usize, session_id: u32, tokens: u64, earlier: &[u64]) {
        let Some(state) = self
            .tiers
            .get(tier)
            .filter(|state| state.spec.writes.is_victim())
        else {
            return;
        };
        let held = state.entries.get(&session_id);
        let end = held.map_or(tokens, |e| e.tokens.max(tokens));
        let incoming = earlier.iter().copied().chain(std::iter::once(tokens));
        let earlier = self.earlier_for(tier, session_id, end.saturating_sub(1), incoming);
        if held.is_some_and(|e| e.tokens == end && e.earlier == earlier) {
            self.tiers[tier].touch(session_id);
        } else {
            self.store_in(tier, session_id, end, earlier);
        }
    }

    /// Whether the fastest tier is HBM's victim tier.
    pub(crate) fn takes_hbm_evictions(&self) -> bool {
        self.tiers
            .first()
            .is_some_and(|tier| tier.spec.writes.is_victim())
    }

    /// HBM evicted the session's `tokens`-token context with its `earlier`
    /// states, or could not keep it: a victim fastest tier writes what its
    /// entry lacks, or only refreshes an entry that already holds it all.
    pub(crate) fn store_evicted(&mut self, session_id: u32, tokens: u64, earlier: &[u64]) {
        self.take_victim(0, session_id, tokens, earlier);
    }

    /// The time the next writes are counted at, per [`WRITE_WINDOW`].
    pub(crate) fn set_clock(&mut self, now: Time) {
        for tier in &mut self.tiers {
            tier.clock = tier.clock.max(now);
        }
    }

    /// Per [`WRITE_WINDOW`]: (entries stored, tokens written).
    pub(crate) fn window_writes(&self, tier: usize) -> &[(u64, u64)] {
        &self.tiers[tier].window_writes
    }

    /// Per [`WRITE_WINDOW`]: states written.
    pub(crate) fn window_states(&self, tier: usize) -> &[u64] {
        &self.tiers[tier].window_states
    }

    /// Keep every write, eviction, cut and read of each tier from now on.
    pub(crate) fn record_events(&mut self) {
        for tier in &mut self.tiers {
            tier.events.get_or_insert_with(Vec::new);
        }
    }

    /// The events `tier` kept since the last call.
    pub(crate) fn take_events(&mut self, tier: usize) -> Vec<PrefixTierEvent> {
        self.tiers[tier]
            .events
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// The session's next round shares `declared` tokens with its previous
    /// context: cut a victim tier's older entry to what that round can resume
    /// from, so it stays a prefix of the session's latest context.
    pub(crate) fn drop_stale(&mut self, session_id: u32, declared: u64) {
        let state_tokens = self.state_tokens;
        for tier in 0..self.tiers.len() {
            if !self.tiers[tier].spec.writes.is_victim() {
                continue;
            }
            if let Some(entry) = self.tiers[tier].entries.get(&session_id) {
                let tokens = entry.tokens;
                let kept = self.resume_tokens(entry, declared);
                let tier = &mut self.tiers[tier];
                if kept < tokens {
                    tier.log(PrefixTierEventKind::Cut, session_id, kept, tokens, declared);
                }
                tier.truncate(session_id, kept, state_tokens);
            }
        }
    }

    pub(crate) fn counters(&self, tier: usize) -> PrefixTierCounters {
        self.tiers[tier].counters
    }

    pub(crate) fn used_tokens(&self, tier: usize) -> u64 {
        self.tiers[tier].used
    }

    /// Earlier round-end states `tier` holds now.
    pub(crate) fn earlier_states(&self, tier: usize) -> u64 {
        self.tiers[tier].earlier_states
    }

    pub(crate) fn capacity_tokens(&self, tier: usize) -> u64 {
        self.tiers[tier].capacity
    }

    /// One recurrent state, in tier tokens.
    pub(crate) fn state_tokens(&self) -> u64 {
        self.state_tokens
    }

    /// Take `fraction` of `tier`'s capacity out of what its entries may use,
    /// as a loading area reads stage through; returns the area in tokens.
    pub(crate) fn carve_loading_area(&mut self, tier: usize, fraction: f64) -> u64 {
        assert!((0.0..1.0).contains(&fraction), "a loading area is a fraction below 1");
        let tier = &mut self.tiers[tier];
        let area = (tier.capacity as f64 * fraction) as u64;
        tier.capacity -= area;
        area
    }

    /// Queue a move of `charge` tokens (KV and states) on `tier`'s channel,
    /// behind its earlier reads, and return when it lands: the second hop of
    /// a read staged through `tier`'s loading area. Counts no tier access.
    pub(crate) fn transfer(&mut self, tier: usize, charge: u64, now: Time) -> Time {
        let tier = &mut self.tiers[tier];
        let start = tier.channel_free.max(now);
        let ready = start + Time::from_s(charge as f64 * tier.bytes_per_token / tier.read_bytes_per_s);
        tier.channel_free = ready;
        ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiers_writing(dram_gb: f64, ssd_gb: f64, dram_writes: PrefixTierWrites) -> PrefixTiers {
        tiers_chain(dram_gb, ssd_gb, dram_writes, PrefixTierWrites::EveryContext)
    }

    fn tiers_chain(
        dram_gb: f64,
        ssd_gb: f64,
        dram_writes: PrefixTierWrites,
        ssd_writes: PrefixTierWrites,
    ) -> PrefixTiers {
        // 1000 B/token: 1 GB = 1M tokens; 1 GB/s reads 1M tokens per second.
        PrefixTiers::new(
            &[
                PrefixTierSpec {
                    name: "dram",
                    capacity_gb_per_gpu: dram_gb,
                    read_gb_per_s_per_gpu: 10.0,
                    writes: dram_writes,
                },
                PrefixTierSpec {
                    name: "ssd",
                    capacity_gb_per_gpu: ssd_gb,
                    read_gb_per_s_per_gpu: 1.0,
                    writes: ssd_writes,
                },
            ],
            1000,
            0,
            ResumePoint::AnyToken,
        )
    }

    fn tiers(dram_gb: f64, ssd_gb: f64) -> PrefixTiers {
        tiers_writing(dram_gb, ssd_gb, PrefixTierWrites::EveryContext)
    }

    fn victim_tiers(dram_gb: f64, ssd_gb: f64) -> PrefixTiers {
        tiers_writing(dram_gb, ssd_gb, PrefixTierWrites::FasterTierEvictions)
    }

    #[test]
    fn a_store_reaches_every_tier_and_the_fastest_serves_it() {
        let mut t = tiers(1.0, 10.0);
        t.store(7, 400_000);
        assert_eq!(
            t.lookup(7, 500_000),
            Some(PrefixTierHit {
                tier: 0,
                tokens: 400_000
            })
        );
        assert_eq!(
            t.lookup(7, 100_000),
            Some(PrefixTierHit {
                tier: 0,
                tokens: 100_000
            })
        );
        assert_eq!(t.lookup(8, 100_000), None);
    }

    #[test]
    fn dram_evicts_least_recent_while_ssd_keeps_it() {
        let mut t = tiers(1.0, 10.0);
        t.store(1, 400_000);
        t.store(2, 400_000);
        // Reading session 1 refreshes it, so session 2 is the DRAM victim.
        t.load(1, t.lookup(1, 400_000).unwrap(), 0, Time::ZERO);
        t.store(3, 400_000);
        assert_eq!(
            t.lookup(2, 400_000),
            Some(PrefixTierHit {
                tier: 1,
                tokens: 400_000
            })
        );
        assert_eq!(t.lookup(1, 400_000).unwrap().tier, 0);
        assert_eq!(t.counters(0).evicted_entries, 1);
    }

    #[test]
    fn reads_queue_on_the_tier_channel_and_copy_into_faster_tiers() {
        let mut t = tiers(1.0, 10.0);
        t.store(1, 400_000);
        t.store(2, 400_000);
        t.store(3, 400_000); // DRAM holds 2 and 3; SSD holds all three.
        let hit = t.lookup(1, 400_000).unwrap();
        assert_eq!(hit.tier, 1);
        let first = t.load(1, hit, 0, Time::ZERO).0;
        assert_eq!(first, Time::from_ms(400.0));
        // A second SSD read waits for the first.
        t.store(4, 400_000);
        t.store(5, 400_000);
        let second = t.load(2, t.lookup(2, 400_000).unwrap(), 0, Time::from_ms(100.0)).0;
        assert_eq!(second, Time::from_ms(800.0));
        // The read put session 1 back in DRAM.
        assert_eq!(t.counters(1).loads, 2);
    }

    #[test]
    fn a_rewrite_counts_only_the_new_tokens() {
        let mut t = tiers(1.0, 10.0);
        t.store(1, 100_000);
        t.store(1, 150_000);
        assert_eq!(t.counters(0).written_tokens, 150_000);
        assert_eq!(t.used_tokens(0), 150_000);
    }

    #[test]
    fn a_hybrid_entry_charges_one_state() {
        let mut t = PrefixTiers::new(
            &[PrefixTierSpec {
                name: "dram",
                capacity_gb_per_gpu: 1.0,
                read_gb_per_s_per_gpu: 1.0,
                writes: PrefixTierWrites::EveryContext,
            }],
            1000,
            50_000,
            ResumePoint::ContextEnd,
        );
        t.store(1, 100_000);
        assert_eq!(t.used_tokens(0), 150_000);
        let ready = t.load(1, t.lookup(1, 100_000).unwrap(), 0, Time::ZERO).0;
        assert_eq!(ready, Time::from_ms(150.0));
    }

    #[test]
    fn a_read_moves_only_what_hbm_lacks() {
        let mut t = tiers(1.0, 10.0);
        t.store(1, 400_000);
        // HBM still holds 300k: 100k cross the 10 GB/s DRAM link.
        let ready = t.load(1, t.lookup(1, 400_000).unwrap(), 300_000, Time::ZERO).0;
        assert_eq!(ready, Time::from_ms(10.0));
        assert_eq!(t.counters(0).loaded_tokens, 100_000);
    }

    #[test]
    fn a_hybrid_entry_serves_only_a_prompt_that_extends_it() {
        let mut t = PrefixTiers::new(
            &[PrefixTierSpec {
                name: "dram",
                capacity_gb_per_gpu: 1.0,
                read_gb_per_s_per_gpu: 1.0,
                writes: PrefixTierWrites::EveryContext,
            }],
            1000,
            50_000,
            ResumePoint::ContextEnd,
        );
        t.store(1, 100_000);
        assert_eq!(t.lookup(1, 100_001).map(|hit| hit.tokens), Some(100_000));
        // The state is at token 100_000 only: a prompt sharing fewer tokens
        // computes from the start.
        assert_eq!(t.lookup(1, 99_000), None);
    }

    #[test]
    fn a_victim_tier_takes_only_what_hbm_evicts() {
        let mut t = victim_tiers(1.0, 10.0);
        t.store(1, 400_000);
        // A finished context reaches the write-through SSD, not DRAM.
        assert_eq!(t.counters(0).stored_entries, 0);
        assert_eq!(t.lookup(1, 400_000).unwrap().tier, 1);
        t.store_evicted(1, 400_000, &[]);
        assert_eq!(
            t.lookup(1, 400_000),
            Some(PrefixTierHit {
                tier: 0,
                tokens: 400_000
            })
        );
        assert_eq!(t.counters(0).written_tokens, 400_000);
    }

    #[test]
    fn a_victim_tier_writes_only_what_its_entry_lacks() {
        let mut t = victim_tiers(1.0, 10.0);
        t.store_evicted(1, 400_000, &[]);
        // The next round extends the entry; HBM then evicts the longer context.
        t.drop_stale(1, 400_001);
        t.store_evicted(1, 500_000, &[]);
        assert_eq!(t.counters(0).written_tokens, 500_000);
        assert_eq!(t.used_tokens(0), 500_000);
        // An eviction of what the entry already holds writes nothing.
        let entries = t.counters(0).stored_entries;
        t.store_evicted(1, 500_000, &[]);
        assert_eq!(t.counters(0).stored_entries, entries);
        assert_eq!(t.counters(0).written_tokens, 500_000);
    }

    #[test]
    fn a_read_back_keeps_the_victim_entry_and_refreshes_it() {
        let mut t = victim_tiers(1.0, 10.0);
        t.store_evicted(1, 400_000, &[]);
        t.store_evicted(2, 400_000, &[]);
        // Reading 1 back into HBM keeps its DRAM copy and makes 2 the LRU.
        t.load(1, t.lookup(1, 400_000).unwrap(), 0, Time::ZERO);
        assert_eq!(t.lookup(1, 400_000).unwrap().tier, 0);
        // HBM evicts 1 again unchanged: no write; then 3 pushes out 2.
        t.store_evicted(1, 400_000, &[]);
        t.store_evicted(3, 400_000, &[]);
        assert_eq!(t.lookup(1, 400_000).unwrap().tier, 0);
        assert_eq!(t.lookup(2, 400_000), None);
        assert_eq!(t.counters(0).written_tokens, 1_200_000);
    }

    #[test]
    fn an_ssd_read_does_not_copy_into_a_victim_tier() {
        let mut t = victim_tiers(1.0, 10.0);
        t.store(1, 400_000);
        t.load(1, t.lookup(1, 400_000).unwrap(), 0, Time::ZERO);
        assert_eq!(t.counters(0).stored_entries, 0);
        assert_eq!(t.lookup(1, 400_000).unwrap().tier, 1);
    }

    #[test]
    fn a_loading_area_shrinks_the_tier_and_hops_queue_on_its_channel() {
        let mut t = victim_tiers(1.0, 10.0);
        let capacity = t.capacity_tokens(0);
        let area = t.carve_loading_area(0, 0.05);
        assert_eq!(area, capacity / 20);
        assert_eq!(t.capacity_tokens(0), capacity - area);
        // 1000 B/token at 10 GB/s: 1M tokens take 100 ms, back to back.
        let first = t.transfer(0, 1_000_000, Time::ZERO);
        let second = t.transfer(0, 1_000_000, Time::ZERO);
        assert!((first.as_ms() - 100.0).abs() < 1e-6);
        assert!((second.as_ms() - 200.0).abs() < 1e-6);
    }

    #[test]
    fn an_exclusive_victim_tier_releases_what_a_read_moves_into_hbm() {
        let mut t = PrefixTiers::new(
            &[PrefixTierSpec {
                name: "dram",
                capacity_gb_per_gpu: 1.0,
                read_gb_per_s_per_gpu: 1.0,
                writes: PrefixTierWrites::FasterTierEvictionsMoved,
            }],
            1000,
            0,
            ResumePoint::AnyToken,
        );
        t.store_evicted(1, 400_000, &[]);
        t.load(1, t.lookup(1, 400_000).unwrap(), 0, Time::ZERO);
        assert_eq!(t.used_tokens(0), 0);
        assert_eq!(t.lookup(1, 400_000), None);
        assert_eq!(t.counters(0).released_entries, 1);
        // HBM evicting it again writes it whole.
        t.store_evicted(1, 500_000, &[]);
        assert_eq!(t.counters(0).written_tokens, 900_000);
    }

    #[test]
    fn a_round_that_does_not_extend_the_entry_cuts_it() {
        let mut t = victim_tiers(1.0, 10.0);
        t.store_evicted(1, 600_000, &[]);
        // The next round shares only 300k tokens with the entry.
        t.drop_stale(1, 300_000);
        assert_eq!(t.used_tokens(0), 300_000);
        assert_eq!(t.lookup(1, 600_000).unwrap().tokens, 300_000);
        // A context resumable only at its end is dropped instead.
        let mut t = PrefixTiers::new(
            &[PrefixTierSpec {
                name: "dram",
                capacity_gb_per_gpu: 1.0,
                read_gb_per_s_per_gpu: 1.0,
                writes: PrefixTierWrites::FasterTierEvictions,
            }],
            1000,
            50_000,
            ResumePoint::ContextEnd,
        );
        t.store_evicted(1, 100_000, &[]);
        t.drop_stale(1, 100_001);
        assert_eq!(t.used_tokens(0), 150_000);
        t.drop_stale(1, 99_000);
        assert_eq!(t.used_tokens(0), 0);
        assert_eq!(t.lookup(1, 200_000), None);
    }

    /// One tier of `gb` GB (1000 B/token: 1 GB = 1M tokens, a state 50k
    /// tokens), resumable only at states, keeping up to `states` per entry.
    fn round_end_tier(gb: f64, writes: PrefixTierWrites, states: u32) -> PrefixTiers {
        PrefixTiers::new(
            &[PrefixTierSpec {
                name: "dram",
                capacity_gb_per_gpu: gb,
                read_gb_per_s_per_gpu: 1.0,
                writes,
            }],
            1000,
            50_000,
            ResumePoint::ContextEnd,
        )
        .with_round_end_states(states)
    }

    fn round(tokens: u64, prompt_end: u64, shared: u64) -> RoundContext {
        RoundContext {
            tokens,
            prompt_end,
            shared,
        }
    }

    #[test]
    fn a_handed_back_round_keeps_its_prompt_end_state() {
        let mut t = round_end_tier(10.0, PrefixTierWrites::EveryContext, 2);
        // Prompt ends at 100k; the decode hands back 1k more.
        t.store_round(1, round(101_000, 100_000, 0));
        assert_eq!(t.used_tokens(0), 101_000 + 2 * 50_000);
        assert_eq!(t.counters(0).written_states, 2);
        // The next prompt keeps only part of the output.
        assert_eq!(t.lookup(1, 100_500).unwrap().tokens, 100_000);
        assert_eq!(t.lookup(1, 101_001).unwrap().tokens, 101_000);
        assert_eq!(t.lookup(1, 99_999), None);
    }

    #[test]
    fn a_read_moves_every_state_at_or_before_the_hit() {
        let mut t = round_end_tier(10.0, PrefixTierWrites::EveryContext, 3);
        t.store_round(1, round(100_000, 100_000, 0));
        t.store_round(1, round(120_000, 120_000, 100_000));
        t.store_round(1, round(150_000, 150_000, 120_000));
        // A hit at the end moves its KV and all three states: 300k tokens
        // at 1M tokens/s.
        let hit = t.lookup(1, 150_001).unwrap();
        assert_eq!(t.read_charge(1, hit, 0), 150_000 + 3 * 50_000);
        let (ready, states) = t.load(1, hit, 0, Time::ZERO);
        assert_eq!(ready, Time::from_ms(300.0));
        assert_eq!(states, [100_000, 120_000]);
        // A hit at an earlier state moves the states up to it.
        let hit = t.lookup(1, 130_000).unwrap();
        assert_eq!(hit.tokens, 120_000);
        let (_, states) = t.load(1, hit, 0, Time::ZERO);
        assert_eq!(states, [100_000]);
        assert_eq!(t.counters(0).loaded_states, 3 + 2);
    }

    #[test]
    fn a_full_tier_evicts_whole_entries_with_all_their_states() {
        let mut t = round_end_tier(0.5, PrefixTierWrites::EveryContext, 2);
        t.store_round(1, round(101_000, 100_000, 0)); // 201k
        t.store_round(2, round(101_000, 100_000, 0)); // 201k
        // 3 needs 151k: session 1 goes whole; no state is dropped alone.
        t.store_round(3, round(101_000, 101_000, 0));
        assert_eq!(t.counters(0).evicted_entries, 1);
        assert_eq!(t.lookup(1, 101_001), None);
        assert_eq!(t.lookup(2, 100_500).unwrap().tokens, 100_000);
        assert_eq!(t.used_tokens(0), 201_000 + 151_000);
        assert_eq!(t.earlier_states(0), 1);
        // A context that does not fit with all its states is not stored.
        t.store_round(4, round(420_000, 300_000, 0));
        assert_eq!(t.lookup(4, 420_001), None);
        assert_eq!(t.lookup(2, 100_500).unwrap().tokens, 100_000);
    }

    #[test]
    fn a_write_through_entry_keeps_the_latest_round_ends_it_shares() {
        let mut t = round_end_tier(10.0, PrefixTierWrites::EveryContext, 3);
        t.store_round(1, round(100_000, 100_000, 0));
        t.store_round(1, round(120_000, 120_000, 100_000));
        t.store_round(1, round(150_000, 150_000, 120_000));
        assert_eq!(t.lookup(1, 110_000).unwrap().tokens, 100_000);
        assert_eq!(t.earlier_states(0), 2);
        // A round sharing only 130k: 150k is dropped, 120k and 100k stay.
        t.store_round(1, round(140_000, 140_000, 130_000));
        assert_eq!(t.lookup(1, 139_000).unwrap().tokens, 120_000);
        assert_eq!(t.lookup(1, 119_000).unwrap().tokens, 100_000);
        assert_eq!(t.used_tokens(0), 140_000 + 3 * 50_000);
        // One state per entry keeps none of them.
        let mut t = round_end_tier(10.0, PrefixTierWrites::EveryContext, 1);
        t.store_round(1, round(101_000, 100_000, 0));
        assert_eq!(t.lookup(1, 100_500), None);
        assert_eq!(t.used_tokens(0), 151_000);
    }

    #[test]
    fn a_victim_tier_merges_states_and_cuts_at_one() {
        let mut t = round_end_tier(10.0, PrefixTierWrites::FasterTierEvictions, 3);
        t.store_evicted(1, 100_000, &[]);
        t.drop_stale(1, 100_001);
        t.store_evicted(1, 120_000, &[]);
        assert_eq!(t.used_tokens(0), 120_000 + 2 * 50_000);
        assert_eq!(t.counters(0).written_states, 2);
        // An eviction of a state it holds writes nothing.
        t.store_evicted(1, 100_000, &[]);
        assert_eq!(t.counters(0).written_states, 2);
        // A round ending inside the context cuts the entry at 100k.
        t.drop_stale(1, 110_000);
        assert_eq!(t.used_tokens(0), 100_000 + 50_000);
        assert_eq!(t.lookup(1, 110_000).unwrap().tokens, 100_000);
    }

    /// HBM -> DRAM -> SSD victim chain (1000 B/token, a state 50k tokens),
    /// resumable only at states, up to `states` per entry.
    fn round_end_chain(dram_gb: f64, ssd_gb: f64, states: u32) -> PrefixTiers {
        let spec = |name, gb| PrefixTierSpec {
            name,
            capacity_gb_per_gpu: gb,
            read_gb_per_s_per_gpu: 1.0,
            writes: PrefixTierWrites::FasterTierEvictions,
        };
        PrefixTiers::new(
            &[spec("dram", dram_gb), spec("ssd", ssd_gb)],
            1000,
            50_000,
            ResumePoint::ContextEnd,
        )
        .with_round_end_states(states)
    }

    #[test]
    fn a_cut_keeps_the_states_at_or_before_it() {
        let mut t = round_end_tier(10.0, PrefixTierWrites::FasterTierEvictions, 3);
        t.store_evicted(1, 150_000, &[100_000, 120_000]);
        assert_eq!(t.used_tokens(0), 150_000 + 3 * 50_000);
        // The next round shares 130k: the entry is cut at 120k, keeping 100k.
        t.drop_stale(1, 130_000);
        assert_eq!(t.used_tokens(0), 120_000 + 2 * 50_000);
        assert_eq!(t.lookup(1, 125_000).unwrap().tokens, 120_000);
        assert_eq!(t.lookup(1, 110_000).unwrap().tokens, 100_000);
    }

    #[test]
    fn a_victim_merge_keeps_the_longer_context_and_the_latest_k_states() {
        let mut t = round_end_tier(10.0, PrefixTierWrites::FasterTierEvictions, 3);
        t.store_evicted(1, 120_000, &[100_000]);
        // HBM evicts a longer context of the session with its own states.
        t.store_evicted(1, 160_000, &[130_000, 150_000]);
        assert_eq!(t.used_tokens(0), 160_000 + 3 * 50_000);
        assert_eq!(t.lookup(1, 140_000).unwrap().tokens, 130_000);
        assert_eq!(t.lookup(1, 125_000), None, "120k is older than the latest 3");
        assert_eq!(t.counters(0).written_states, 2 + 3);
        // A shorter context with an older state merges into the longer one.
        let mut t = round_end_tier(10.0, PrefixTierWrites::FasterTierEvictions, 3);
        t.store_evicted(1, 200_000, &[150_000]);
        t.store_evicted(1, 120_000, &[100_000]);
        assert_eq!(t.lookup(1, 200_001).unwrap().tokens, 200_000);
        assert_eq!(t.lookup(1, 140_000).unwrap().tokens, 120_000);
        assert_eq!(t.lookup(1, 110_000), None);
        assert_eq!(t.used_tokens(0), 200_000 + 3 * 50_000);
    }

    #[test]
    fn an_exclusive_tier_moves_an_entry_and_its_states_into_hbm() {
        let spec = |name, writes| PrefixTierSpec {
            name,
            capacity_gb_per_gpu: 10.0,
            read_gb_per_s_per_gpu: 1.0,
            writes,
        };
        let mut t = PrefixTiers::new(
            &[
                spec("dram", PrefixTierWrites::FasterTierEvictionsMoved),
                spec("ssd", PrefixTierWrites::FasterTierEvictions),
            ],
            1000,
            50_000,
            ResumePoint::ContextEnd,
        )
        .with_round_end_states(3);
        t.store_evicted(1, 150_000, &[100_000, 120_000]);
        let (_, states) = t.load(1, t.lookup(1, 150_001).unwrap(), 0, Time::ZERO);
        // The states went to HBM with the context: none is left behind or
        // handed down to SSD.
        assert_eq!(states, [100_000, 120_000]);
        assert_eq!(t.used_tokens(0), 0);
        assert_eq!(t.used_tokens(1), 0);
        assert_eq!(t.counters(1).stored_entries, 0);
        // HBM evicting it again writes it whole, all its states included.
        t.store_evicted(1, 170_000, &[120_000, 150_000]);
        assert_eq!(t.used_tokens(0), 170_000 + 3 * 50_000);
        assert_eq!(t.counters(0).written_states, 3 + 3);
    }

    #[test]
    fn a_victim_chain_demotes_states_with_their_kv() {
        let mut t = round_end_chain(0.5, 10.0, 4);
        t.store_evicted(1, 200_000, &[100_000, 150_000]);
        assert_eq!(t.used_tokens(0), 200_000 + 3 * 50_000);
        assert_eq!(t.lookup(1, 160_000).unwrap(), PrefixTierHit { tier: 0, tokens: 150_000 });
        // DRAM needs room for session 2: it demotes 1 whole, its states with
        // its KV.
        t.store_evicted(2, 150_000, &[]);
        assert_eq!(t.counters(0).evicted_entries, 1);
        assert_eq!(t.lookup(1, 160_000).unwrap(), PrefixTierHit { tier: 1, tokens: 150_000 });
        assert_eq!(t.lookup(1, 120_000).unwrap().tokens, 100_000);
        assert_eq!(t.used_tokens(1), 200_000 + 3 * 50_000);
        assert_eq!(t.counters(1).written_tokens, 200_000);
        assert_eq!(t.counters(1).written_states, 3);
        // Session 1 resumes at 150k from SSD (the read moves 100k and 150k)
        // and extends to 250k, which HBM later evicts into DRAM, pushing 2
        // down to SSD.
        let (_, states) = t.load(1, t.lookup(1, 160_000).unwrap(), 0, Time::ZERO);
        assert_eq!(states, [100_000]);
        t.drop_stale(1, 160_000);
        assert_eq!(t.used_tokens(1), 150_000 + 2 * 50_000);
        t.store_evicted(1, 250_000, &[100_000, 150_000]);
        assert_eq!(t.lookup(1, 250_000).unwrap(), PrefixTierHit { tier: 0, tokens: 250_000 });
        // DRAM demotes 1: SSD merges it, writing only the 100k tokens and the
        // one state it lacks.
        t.store_evicted(3, 300_000, &[]);
        assert_eq!(t.counters(1).written_tokens, 200_000 + 150_000 + 100_000);
        assert_eq!(t.counters(1).written_states, 3 + 1 + 1);
        assert_eq!(t.lookup(1, 250_000).unwrap(), PrefixTierHit { tier: 1, tokens: 250_000 });
        assert_eq!(t.lookup(1, 200_000).unwrap().tokens, 150_000);
        assert_eq!(t.lookup(1, 140_000).unwrap().tokens, 100_000);
        // The last tier also evicts whole entries.
        let mut t = round_end_chain(0.3, 0.5, 4);
        t.store_evicted(1, 150_000, &[100_000]); // 250k
        t.store_evicted(2, 150_000, &[]); // DRAM demotes 1 whole to SSD (250k)
        t.store_evicted(3, 150_000, &[]); // and 2 (200k): SSD holds 450k
        t.store_evicted(4, 150_000, &[]); // 3 to SSD: 1 goes whole
        assert_eq!(t.counters(1).evicted_entries, 1);
        assert_eq!(t.lookup(1, 150_001), None);
        assert_eq!(t.earlier_states(1), 0);
    }

    use PrefixTierWrites::{EveryContext, FasterTierEvictions};

    /// HBM -> DRAM -> SSD, each the victim tier of the one above.
    fn chain(dram_gb: f64, ssd_gb: f64) -> PrefixTiers {
        tiers_chain(dram_gb, ssd_gb, FasterTierEvictions, FasterTierEvictions)
    }

    #[test]
    fn a_victim_chain_writes_each_tier_only_with_what_the_one_above_evicts() {
        let mut t = chain(1.0, 10.0);
        // A finished context stays in HBM: neither tier is written.
        t.store(1, 400_000);
        assert_eq!(t.lookup(1, 400_000), None);
        t.store_evicted(1, 400_000, &[]);
        t.store_evicted(2, 400_000, &[]);
        assert_eq!(t.counters(1).stored_entries, 0);
        // A third eviction from HBM pushes session 1 out of DRAM into SSD.
        t.store_evicted(3, 400_000, &[]);
        assert_eq!(
            t.lookup(1, 400_000),
            Some(PrefixTierHit {
                tier: 1,
                tokens: 400_000
            })
        );
        assert_eq!(t.counters(0).evicted_entries, 1);
        assert_eq!(t.counters(1).written_tokens, 400_000);
        assert_eq!(t.lookup(3, 400_000).unwrap().tier, 0);
    }

    #[test]
    fn an_ssd_read_keeps_the_ssd_copy_and_skips_a_victim_dram() {
        let mut t = chain(1.0, 10.0);
        t.store_evicted(1, 400_000, &[]);
        t.store_evicted(2, 400_000, &[]);
        t.store_evicted(3, 400_000, &[]); // 1 goes to SSD
        t.store_evicted(4, 400_000, &[]); // 2 goes to SSD; SSD's LRU: 1, 2
        t.load(1, t.lookup(1, 400_000).unwrap(), 0, Time::ZERO);
        // The read neither copies 1 into DRAM nor drops SSD's copy, and it
        // refreshes 1 in SSD's order.
        assert_eq!(t.counters(0).stored_entries, 4);
        assert_eq!(t.lookup(1, 400_000).unwrap().tier, 1);
        assert_eq!(t.tiers[1].lru.values().copied().collect::<Vec<_>>(), [2, 1]);
        // HBM evicts 1 unchanged: it lands in DRAM and pushes 3 down to SSD.
        t.store_evicted(1, 400_000, &[]);
        assert_eq!(t.lookup(1, 400_000).unwrap().tier, 0);
        let ssd = t.counters(1);
        // DRAM later evicts 1, which SSD holds whole: SSD is only refreshed.
        t.store_evicted(5, 400_000, &[]); // pushes out 4
        t.store_evicted(6, 400_000, &[]); // pushes out 1
        assert_eq!(t.counters(1).stored_entries, ssd.stored_entries + 1);
        assert_eq!(t.counters(1).written_tokens, ssd.written_tokens + 400_000);
        assert_eq!(t.lookup(1, 400_000).unwrap().tier, 1);
        assert_eq!(t.tiers[1].lru.values().last(), Some(&1));
    }

    #[test]
    fn dram_evicting_a_longer_context_writes_ssd_only_what_it_lacks() {
        let mut t = chain(1.0, 10.0);
        t.store_evicted(1, 400_000, &[]);
        t.store_evicted(2, 400_000, &[]);
        t.store_evicted(3, 400_000, &[]); // 1 goes to SSD at 400k
                                     // Session 1's next round extends it; HBM later evicts 500k.
        t.drop_stale(1, 400_001);
        t.store_evicted(1, 500_000, &[]); // pushes 2 down
        t.store_evicted(4, 400_000, &[]); // pushes 3 down
        t.store_evicted(5, 400_000, &[]); // pushes 1 down: SSD adds its 100k new tokens
        assert_eq!(t.counters(1).written_tokens, 400_000 * 3 + 100_000);
        assert_eq!(
            t.lookup(1, 500_000).unwrap(),
            PrefixTierHit {
                tier: 1,
                tokens: 500_000
            }
        );
    }

    #[test]
    fn a_round_that_shares_less_cuts_every_victim_tier() {
        let mut t = chain(1.0, 10.0);
        t.store_evicted(1, 600_000, &[]);
        t.store_evicted(2, 600_000, &[]); // 1 goes to SSD at 600k
        t.store_evicted(1, 700_000, &[]); // HBM evicts a longer context into DRAM
        t.drop_stale(1, 300_000);
        assert_eq!(t.lookup(1, 700_000).unwrap().tokens, 300_000);
        assert_eq!(t.tiers[1].entries[&1].tokens, 300_000);
        // A round that starts over drops both.
        t.drop_stale(1, 0);
        assert_eq!(t.lookup(1, 700_000), None);
    }

    #[test]
    fn a_write_through_dram_hands_its_evictions_to_a_victim_ssd() {
        let mut t = tiers_chain(1.0, 10.0, EveryContext, FasterTierEvictions);
        t.store(1, 400_000);
        t.store(1, 450_000);
        t.store(2, 400_000);
        assert_eq!(t.counters(1).stored_entries, 0);
        t.store(3, 400_000); // DRAM evicts 1: SSD receives its latest context
        assert_eq!(
            t.lookup(1, 450_000),
            Some(PrefixTierHit {
                tier: 1,
                tokens: 450_000
            })
        );
        // An SSD read passes through the write-through DRAM, which keeps a copy
        // and pushes 2 down.
        t.load(1, t.lookup(1, 450_000).unwrap(), 0, Time::ZERO);
        assert_eq!(t.lookup(1, 450_000).unwrap().tier, 0);
        assert_eq!(t.lookup(2, 400_000).unwrap().tier, 1);
        // A context larger than DRAM goes straight to SSD.
        t.store(4, 2_000_000);
        assert_eq!(t.lookup(4, 2_000_000).unwrap().tier, 1);
    }

    #[test]
    fn writes_are_counted_per_window_of_the_clock() {
        let mut t = chain(1.0, 10.0);
        t.store_evicted(1, 400_000, &[]);
        t.set_clock(Time::from_ms(25_000.0));
        t.store_evicted(2, 400_000, &[]);
        t.store_evicted(3, 300_000, &[]); // pushes 1 down to SSD
        assert_eq!(t.window_writes(0), [(1, 400_000), (0, 0), (2, 700_000)]);
        assert_eq!(t.window_writes(1), [(0, 0), (0, 0), (1, 400_000)]);
    }

    #[test]
    fn the_event_log_keeps_writes_evictions_cuts_and_reads() {
        use PrefixTierEventKind::*;
        let mut t = chain(1.0, 10.0);
        t.record_events();
        t.store_evicted(1, 400_000, &[]);
        t.store_evicted(2, 400_000, &[]);
        t.store_evicted(3, 400_000, &[]); // 1 goes to SSD
        t.load(1, t.lookup(1, 400_000).unwrap(), 100, Time::ZERO);
        t.drop_stale(1, 300_000);
        let kinds = |events: Vec<PrefixTierEvent>| {
            events
                .iter()
                .map(|e| (e.kind, e.session_id, e.tokens, e.prev_tokens))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            kinds(t.take_events(0)),
            [
                (Write, 1, 400_000, 0),
                (Write, 2, 400_000, 0),
                (Evict, 1, 400_000, 400_000),
                (Write, 3, 400_000, 0)
            ]
        );
        assert_eq!(
            kinds(t.take_events(1)),
            [
                (Write, 1, 400_000, 0),
                (Read, 1, 400_000, 100),
                (Cut, 1, 300_000, 400_000)
            ]
        );
        assert!(t.take_events(1).is_empty());
    }
}
