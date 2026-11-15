//! Host DRAM and SSD tiers behind one partition's HBM prefix cache.
//!
//! A tier is an LRU of session contexts, sized and read per GPU: every GPU of
//! the partition keeps its own slice of each token (a pipeline stage its
//! layers' KV), so capacity and read time are one GPU's bytes over one GPU's
//! capacity and link, and all GPUs read their slices in parallel. Tiers are
//! ordered fastest first.
//!
//! A write-through tier ([`TierWrites::EveryContext`]) receives every finished
//! context, as LMCache and vLLM's offloading connector store a request's KV
//! when it finishes, and evicts on its own. A victim tier
//! ([`TierWrites::Evictions`]) receives a session's context only when the tier
//! above it (above the fastest tier: HBM's prefix cache) evicts it or cannot
//! hold it, merged into its own entry for the session: both are prefixes of
//! the session's latest context, so the merge is the longer context with the
//! states the retention keeps among both. An entry that already holds all of
//! it is only refreshed. A read back into HBM leaves the read tier's entry in
//! place, refreshed, unless the tier is exclusive
//! ([`TierWrites::EvictionsExclusive`]), whose read releases it. A read passes
//! through every faster write-through tier, which keeps a copy; a faster victim
//! tier receives the context only when the tier above it evicts it. A session
//! request's admission cuts each victim entry to what that request shares with
//! it ([`TierChain::cut`]), so it stays a prefix of the session's latest
//! context.
//!
//! Units are tier tokens: one GPU's context bytes per token
//! ([`TierBytes::per_token`]). An entry also charges one state for its end and
//! one for each earlier round-end state it keeps, each
//! [`TierBytes::state_tokens`]. An entry is one unit: a tier evicts whole least
//! recently used entries, and whatever moves an entry moves all its states.
//! Where an entry can be resumed follows its [`ResumePositions`], as in HBM,
//! else its latest kept state at or before the prompt's end.
//!
//! Reads queue FIFO on one channel per tier (its read bandwidth). A read moves
//! the KV HBM lacks and, of the entry's states at or before the hit, only those
//! that survive the round it serves ([`StateRetention::survivors`]): the state
//! it resumes from and the latest earlier ones still kept once the round adds
//! its own. Writes are free: only counted.

use std::collections::{BTreeMap, HashMap};

use crate::arch::kv_layout::{KvCacheLayout, KvGrowth};
use crate::common::Time;

use super::kv_accounting::ResumePositions;
use super::state_retention::{normalize, RoundState, StateRetention, StateTag};

/// What a tier is written.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TierWrites {
    /// Every finished context, as it finishes (write-through).
    #[default]
    EveryContext,
    /// Only what the tier above evicts or cannot hold (a victim tier).
    Evictions,
    /// A victim tier whose read moves the entry into HBM and releases it, so
    /// it holds only what HBM lacks.
    EvictionsExclusive,
}

impl TierWrites {
    /// Whether the tier is the victim tier of the one above it.
    pub fn is_victim(self) -> bool {
        self != Self::EveryContext
    }
}

/// One tier's size and read bandwidth per GPU, and what it is written.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TierSpec {
    pub name: &'static str,
    pub capacity_gb: f64,
    pub read_gb_per_s: f64,
    pub writes: TierWrites,
}

/// One GPU's bytes a tier stores per context token and per recurrent state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TierBytes {
    pub per_token: u64,
    pub state: u64,
}

impl TierBytes {
    /// One state in tier tokens, rounded up.
    pub fn state_tokens(self) -> u64 {
        self.state.div_ceil(self.per_token.max(1))
    }

    /// A one-GPU layout's bytes: its per-token parts, and the model's own
    /// bytes (no padding) of its prefix-reusable per-request parts.
    pub fn of_layout(layout: &KvCacheLayout) -> Self {
        Self {
            per_token: layout.per_token_bytes(),
            state: layout
                .parts
                .iter()
                .filter(|part| part.prefix_reusable && part.growth == KvGrowth::PerRequest)
                .map(|part| part.bytes_per_entry)
                .sum(),
        }
    }

    /// What one GPU of a pipeline reads, given each stage's bytes: the first
    /// stage with the most bytes per token, or with `balanced` the stages'
    /// means, each rounded up.
    pub fn of_stages(stages: &[TierBytes], balanced: bool) -> Self {
        if balanced {
            let depth = stages.len().max(1) as u64;
            return Self {
                per_token: stages
                    .iter()
                    .map(|s| s.per_token)
                    .sum::<u64>()
                    .div_ceil(depth),
                state: stages.iter().map(|s| s.state).sum::<u64>().div_ceil(depth),
            };
        }
        stages
            .iter()
            .fold(None::<TierBytes>, |best, &stage| match best {
                Some(best) if best.per_token >= stage.per_token => Some(best),
                _ => Some(stage),
            })
            .unwrap_or_default()
    }
}

/// A finished request's context, as a write-through tier stores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RoundContext {
    /// Where the context can be resumed; its end is the context.
    pub(crate) positions: ResumePositions,
    /// Tokens it shares with the session's previous context.
    pub(crate) shared: u64,
}

/// A context leaving HBM or a tier for the tier behind it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LeavingContext {
    pub(crate) session_id: u32,
    pub(crate) positions: ResumePositions,
    /// The state at `positions.end()`.
    pub(crate) end: StateTag,
    /// Round-end states below the end, ascending.
    pub(crate) earlier: Vec<RoundState>,
}

/// Where a lookup found a session's context, and how much of it resumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TierHit {
    pub(crate) tier: usize,
    pub(crate) tokens: u64,
}

/// A read started on a tier's channel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TierRead {
    pub(crate) started: Time,
    pub(crate) ready: Time,
    /// Where the read context can be resumed; its end is the hit.
    pub(crate) positions: ResumePositions,
    /// The state at the hit.
    pub(crate) end: StateTag,
    /// The earlier states it moves below the hit, ascending.
    pub(crate) earlier: Vec<RoundState>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TierCounters {
    pub(crate) stored_entries: u64,
    /// Tokens newly written: each store writes only what the session's
    /// previous entry in the tier did not hold.
    pub(crate) written_tokens: u64,
    /// States newly written (an end or earlier state the entry did not hold).
    pub(crate) written_states: u64,
    pub(crate) evicted_entries: u64,
    pub(crate) evicted_tokens: u64,
    pub(crate) loads: u64,
    /// Context tokens reads moved (what HBM lacked).
    pub(crate) loaded_tokens: u64,
    pub(crate) loaded_states: u64,
    /// Entries an exclusive tier's reads released.
    pub(crate) released_entries: u64,
}

/// What happened to a tier's entry for a session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TierEventKind {
    /// Write-through of a finished context.
    WriteThrough,
    /// What the tier above evicted or could not hold.
    Victim,
    /// A copy of a read from a slower tier passing through.
    ReadCopy,
    /// A pre-run context put into the slowest tier.
    WarmStart,
    /// Least recently used eviction.
    Evict,
    /// A context larger than the tier: the previous entry is dropped and
    /// nothing is written.
    TooLarge,
    /// The session's next request shares less than the entry holds.
    Cut,
    /// A read of the entry's first `hit_tokens`.
    Read,
}

/// One entry change, for the prefix-cache event log.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TierEvent {
    pub(crate) tier: usize,
    pub(crate) kind: TierEventKind,
    pub(crate) session_id: u32,
    pub(crate) entry_tokens_before: u64,
    pub(crate) entry_tokens_after: u64,
    /// Context tokens and tier tokens the tier holds, before and after.
    pub(crate) used_tokens_before: u64,
    pub(crate) used_tokens_after: u64,
    pub(crate) used_units_before: u64,
    pub(crate) used_units_after: u64,
    /// Reads: the hit and when it lands.
    pub(crate) hit_tokens: u64,
    pub(crate) ready: Option<Time>,
}

#[derive(Clone, Debug)]
struct TierEntry {
    positions: ResumePositions,
    /// The state at `positions.end()`.
    end: StateTag,
    /// Earlier round-end states, ascending, each below the end.
    earlier: Vec<RoundState>,
    /// LRU stamp.
    stamp: u64,
}

impl TierEntry {
    fn tokens(&self) -> u64 {
        self.positions.end()
    }

    /// Its states, the end's last.
    fn states(&self) -> impl Iterator<Item = RoundState> + '_ {
        self.earlier
            .iter()
            .copied()
            .chain(std::iter::once(RoundState {
                at: self.tokens(),
                tag: self.end,
            }))
    }

    /// The state a prompt sharing `declared` tokens resumes from: the largest
    /// position at or below it, else the latest earlier state there.
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

    /// Its positions cut at `last`, an earlier state or one of its positions.
    fn positions_cut_at(&self, last: u64) -> ResumePositions {
        if self.positions.floor(last) == last {
            self.positions.truncated(last)
        } else {
            ResumePositions::at_state(last)
        }
    }
}

struct Tier {
    spec: TierSpec,
    /// Tier tokens.
    capacity: u64,
    used: u64,
    used_tokens: u64,
    entries: HashMap<u32, TierEntry>,
    lru: BTreeMap<u64, u32>,
    earlier_states: u64,
    next_stamp: u64,
    /// When the read channel is next free.
    channel_free: Time,
    counters: TierCounters,
}

impl Tier {
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

    fn insert(&mut self, session_id: u32, entry: TierEntry, state_tokens: u64) {
        self.used += self.charge(entry.tokens(), 1 + entry.earlier.len(), state_tokens);
        self.used_tokens += entry.tokens();
        self.earlier_states += entry.earlier.len() as u64;
        self.lru.insert(entry.stamp, session_id);
        self.entries.insert(session_id, entry);
    }

    fn remove(&mut self, session_id: u32, state_tokens: u64) -> Option<TierEntry> {
        let entry = self.entries.remove(&session_id)?;
        self.lru.remove(&entry.stamp);
        self.earlier_states -= entry.earlier.len() as u64;
        self.used -= self.charge(entry.tokens(), 1 + entry.earlier.len(), state_tokens);
        self.used_tokens -= entry.tokens();
        Some(entry)
    }

    fn take_stamp(&mut self) -> u64 {
        let stamp = self.next_stamp;
        self.next_stamp += 1;
        stamp
    }
}

/// One partition's DRAM/SSD tiers, fastest first.
pub(crate) struct TierChain {
    tiers: Vec<Tier>,
    bytes: TierBytes,
    /// One state in tier tokens.
    state_tokens: u64,
    retention: StateRetention,
    /// Entry changes since the last [`Self::take_events`], when recorded.
    events: Option<Vec<TierEvent>>,
}

impl TierChain {
    pub(crate) fn new(specs: &[TierSpec], bytes: TierBytes, retention: StateRetention) -> Self {
        let per_token = bytes.per_token.max(1) as f64;
        let tiers = specs
            .iter()
            .map(|&spec| {
                assert!(
                    spec.capacity_gb > 0.0 && spec.read_gb_per_s > 0.0,
                    "prefix tier {} needs a positive size and bandwidth",
                    spec.name
                );
                Tier {
                    spec,
                    capacity: (spec.capacity_gb * 1e9 / per_token) as u64,
                    used: 0,
                    used_tokens: 0,
                    entries: HashMap::new(),
                    lru: BTreeMap::new(),
                    earlier_states: 0,
                    next_stamp: 0,
                    channel_free: Time::ZERO,
                    counters: TierCounters::default(),
                }
            })
            .collect();
        Self {
            tiers,
            bytes,
            state_tokens: bytes.state_tokens(),
            retention,
            events: None,
        }
    }

    /// Keep every entry change from now on, for [`Self::take_events`].
    pub(crate) fn record_events(&mut self) {
        self.events.get_or_insert_with(Vec::new);
    }

    /// The entry changes since the last call.
    pub(crate) fn take_events(&mut self) -> Vec<TierEvent> {
        self.events.as_mut().map(std::mem::take).unwrap_or_default()
    }

    pub(crate) fn len(&self) -> usize {
        self.tiers.len()
    }

    pub(crate) fn tier_name(&self, tier: usize) -> &'static str {
        self.tiers[tier].spec.name
    }

    pub(crate) fn counters(&self, tier: usize) -> TierCounters {
        self.tiers[tier].counters
    }

    pub(crate) fn capacity_units(&self, tier: usize) -> u64 {
        self.tiers[tier].capacity
    }

    pub(crate) fn used_units(&self, tier: usize) -> u64 {
        self.tiers[tier].used
    }

    /// Earlier round-end states `tier` holds now.
    pub(crate) fn earlier_states(&self, tier: usize) -> u64 {
        self.tiers[tier].earlier_states
    }

    #[allow(clippy::too_many_arguments)]
    fn log(
        &mut self,
        tier: usize,
        kind: TierEventKind,
        session_id: u32,
        entry_tokens: (u64, u64),
        before: (u64, u64),
        hit_tokens: u64,
        ready: Option<Time>,
    ) {
        let Some(events) = &mut self.events else {
            return;
        };
        let state = &self.tiers[tier];
        events.push(TierEvent {
            tier,
            kind,
            session_id,
            entry_tokens_before: entry_tokens.0,
            entry_tokens_after: entry_tokens.1,
            used_tokens_before: before.0,
            used_tokens_after: state.used_tokens,
            used_units_before: before.1,
            used_units_after: state.used,
            hit_tokens,
            ready,
        });
    }

    fn used(&self, tier: usize) -> (u64, u64) {
        (self.tiers[tier].used_tokens, self.tiers[tier].used)
    }

    /// Of `states` (each below `end`), those the retention keeps beside the
    /// `end` state, ascending.
    fn kept_below(&self, mut states: Vec<RoundState>, end: RoundState) -> Vec<RoundState> {
        normalize(&mut states);
        states.push(end);
        let mut kept = self.retention.keep(states);
        kept.pop();
        kept
    }

    /// The earlier states a new entry for the session in `tier` ending at
    /// `end` keeps: of the old entry's states and `extra`'s at or before
    /// `limit`, those the retention keeps beside `end`.
    fn earlier_for(
        &self,
        tier: usize,
        session_id: u32,
        limit: u64,
        extra: impl IntoIterator<Item = RoundState>,
        end: RoundState,
    ) -> Vec<RoundState> {
        let states: Vec<RoundState> = self.tiers[tier]
            .entries
            .get(&session_id)
            .into_iter()
            .flat_map(TierEntry::states)
            .chain(extra)
            .filter(|state| state.at > 0 && state.at <= limit)
            .collect();
        self.kept_below(states, end)
    }

    /// Store the session's context in `tier` as one unit, and hand what leaves
    /// the tier for it (whole LRU entries, or the context itself when it is
    /// larger than the tier) to the tier behind it, if that is a victim tier.
    fn store_in(
        &mut self,
        tier: usize,
        kind: TierEventKind,
        session_id: u32,
        positions: ResumePositions,
        end: StateTag,
        earlier: Vec<RoundState>,
    ) {
        let state_tokens = self.state_tokens;
        let before = self.used(tier);
        let previous = self.tiers[tier].remove(session_id, state_tokens);
        let previous_tokens = previous.as_ref().map_or(0, TierEntry::tokens);
        let tokens = positions.end();
        let charge = self.tiers[tier].charge(tokens, 1 + earlier.len(), state_tokens);
        if charge > self.tiers[tier].capacity {
            self.log(
                tier,
                TierEventKind::TooLarge,
                session_id,
                (previous_tokens, 0),
                before,
                0,
                None,
            );
            self.pass_down(
                tier,
                LeavingContext {
                    session_id,
                    positions,
                    end,
                    earlier,
                },
            );
            return;
        }
        let mut left = Vec::new();
        while self.tiers[tier].used + charge > self.tiers[tier].capacity {
            let victim = *self.tiers[tier]
                .lru
                .values()
                .next()
                .expect("a full tier has an entry to evict");
            let used_before = self.used(tier);
            let entry = self.tiers[tier]
                .remove(victim, state_tokens)
                .expect("indexed entry");
            self.tiers[tier].counters.evicted_entries += 1;
            self.tiers[tier].counters.evicted_tokens += entry.tokens();
            self.log(
                tier,
                TierEventKind::Evict,
                victim,
                (entry.tokens(), 0),
                used_before,
                0,
                None,
            );
            left.push(LeavingContext {
                session_id: victim,
                positions: entry.positions,
                end: entry.end,
                earlier: entry.earlier,
            });
        }
        let written_states = std::iter::once(tokens)
            .chain(earlier.iter().map(|state| state.at))
            .filter(|&at| {
                !previous
                    .as_ref()
                    .is_some_and(|entry| entry.states().any(|state| state.at == at))
            })
            .count() as u64;
        // The write replaces the session's previous entry: its row counts
        // that entry as still held before.
        let previous_units = previous.as_ref().map_or(0, |entry| {
            self.tiers[tier].charge(entry.tokens(), 1 + entry.earlier.len(), state_tokens)
        });
        let (used_tokens, used_units) = self.used(tier);
        let before = (used_tokens + previous_tokens, used_units + previous_units);
        let stamp = self.tiers[tier].take_stamp();
        self.tiers[tier].insert(
            session_id,
            TierEntry {
                positions,
                end,
                earlier,
                stamp,
            },
            state_tokens,
        );
        let counters = &mut self.tiers[tier].counters;
        counters.stored_entries += 1;
        counters.written_tokens += tokens.saturating_sub(previous_tokens);
        counters.written_states += written_states;
        self.log(
            tier,
            kind,
            session_id,
            (previous_tokens, tokens),
            before,
            0,
            None,
        );
        for leaving in left {
            self.pass_down(tier, leaving);
        }
    }

    /// Hand a context that left `tier` to the tier behind it.
    fn pass_down(&mut self, tier: usize, leaving: LeavingContext) {
        self.take_victim(tier + 1, leaving);
    }

    /// The tier above `tier` let go of a context: if `tier` is its victim
    /// tier, merge it into the session's entry (the longer context with the
    /// states the retention keeps among both), or only refresh an entry that
    /// already holds it all.
    fn take_victim(&mut self, tier: usize, leaving: LeavingContext) {
        if !self
            .tiers
            .get(tier)
            .is_some_and(|state| state.spec.writes.is_victim())
        {
            return;
        }
        let session_id = leaving.session_id;
        let held = self.tiers[tier].entries.get(&session_id);
        let tokens = leaving.positions.end();
        let (positions, end) = match held {
            Some(entry) if entry.tokens() > tokens => (entry.positions, entry.end),
            _ => (leaving.positions, leaving.end),
        };
        let incoming = leaving
            .earlier
            .iter()
            .copied()
            .chain(std::iter::once(RoundState {
                at: tokens,
                tag: leaving.end,
            }));
        let earlier = self.earlier_for(
            tier,
            session_id,
            positions.end().saturating_sub(1),
            incoming,
            RoundState {
                at: positions.end(),
                tag: end,
            },
        );
        let held = self.tiers[tier].entries.get(&session_id);
        if held.is_some_and(|entry| {
            entry.positions == positions && entry.end == end && entry.earlier == earlier
        }) {
            self.tiers[tier].touch(session_id);
        } else {
            self.store_in(
                tier,
                TierEventKind::Victim,
                session_id,
                positions,
                end,
                earlier,
            );
        }
    }

    /// HBM evicted a context or could not keep it: a victim fastest tier
    /// takes it.
    pub(crate) fn take_hbm_eviction(&mut self, leaving: LeavingContext) {
        self.take_victim(0, leaving);
    }

    /// Write a finished context through to every write-through tier; what
    /// that pushes out of a tier goes to a victim tier behind it. An entry
    /// keeping earlier round-end states keeps the old entry's within what the
    /// round shares with it.
    pub(crate) fn store_round(&mut self, session_id: u32, round: RoundContext) {
        let tokens = round.positions.end();
        for tier in 0..self.tiers.len() {
            if self.tiers[tier].spec.writes != TierWrites::EveryContext {
                continue;
            }
            let limit = round.shared.min(tokens.saturating_sub(1));
            let states: Vec<RoundState> = self.tiers[tier]
                .entries
                .get(&session_id)
                .into_iter()
                .flat_map(TierEntry::states)
                .filter(|state| state.at > 0 && state.at <= limit)
                .collect();
            // The round is one past the latest state it shares.
            let end = StateTag {
                round: states.last().map_or(0, |state| state.tag.round + 1),
            };
            let earlier = self.kept_below(
                states,
                RoundState {
                    at: tokens,
                    tag: end,
                },
            );
            self.store_in(
                tier,
                TierEventKind::WriteThrough,
                session_id,
                round.positions,
                end,
                earlier,
            );
        }
    }

    /// Put a context that predates the run into the slowest tier only, where
    /// a long-lived session's context would have settled.
    pub(crate) fn seed(&mut self, session_id: u32, positions: ResumePositions) {
        let Some(slowest) = self.tiers.len().checked_sub(1) else {
            return;
        };
        self.store_in(
            slowest,
            TierEventKind::WarmStart,
            session_id,
            positions,
            StateTag::default(),
            Vec::new(),
        );
    }

    /// The fastest tier whose entry a prompt sharing `declared` tokens with
    /// it resumes the most of.
    pub(crate) fn lookup(&self, session_id: u32, declared: u64) -> Option<TierHit> {
        let mut best: Option<TierHit> = None;
        for (tier, state) in self.tiers.iter().enumerate() {
            let Some(entry) = state.entries.get(&session_id) else {
                continue;
            };
            let tokens = entry.resume_state(declared).map_or(0, |state| state.at);
            if tokens > 0 && best.is_none_or(|hit| tokens > hit.tokens) {
                best = Some(TierHit { tier, tokens });
            }
        }
        best
    }

    /// The states a read of `hit` moves below it: of the entry's states at or
    /// before the hit, those that survive the round it serves.
    pub(crate) fn read_states(&self, session_id: u32, hit: TierHit) -> Vec<RoundState> {
        let Some(entry) = self.tiers[hit.tier].entries.get(&session_id) else {
            return Vec::new();
        };
        let resume = entry
            .resume_state(hit.tokens)
            .expect("a hit resumes from a state");
        let below: Vec<RoundState> = entry
            .states()
            .filter(|state| state.at <= hit.tokens)
            .collect();
        let mut moved = self.retention.survivors(&below, resume.tag);
        moved.retain(|state| state.at < hit.tokens);
        moved
    }

    /// Tier tokens a read of `hit` moves: the KV past HBM's `resident`, the
    /// state at the hit and the earlier states it moves.
    pub(crate) fn read_units(&self, session_id: u32, hit: TierHit, resident: u64) -> u64 {
        let states = 1 + self.read_states(session_id, hit).len() as u64;
        hit.tokens.saturating_sub(resident) + states * self.state_tokens
    }

    /// How long a read queued on `tier` now waits for its channel.
    pub(crate) fn queue_wait(&self, tier: usize, now: Time) -> Time {
        Time(self.tiers[tier].channel_free.0.saturating_sub(now.0))
    }

    /// Queue a read of `hit`, of what HBM's `resident` tokens lack, behind the
    /// tier's earlier reads. The entry stays, refreshed, or an exclusive tier
    /// releases it; every faster write-through tier receives a copy.
    pub(crate) fn read(
        &mut self,
        session_id: u32,
        hit: TierHit,
        resident: u64,
        now: Time,
    ) -> TierRead {
        let state_tokens = self.state_tokens;
        let earlier = self.read_states(session_id, hit);
        let units = self.read_units(session_id, hit, resident);
        let (positions, end) = {
            let entry = &self.tiers[hit.tier].entries[&session_id];
            let resume = entry
                .resume_state(hit.tokens)
                .expect("a hit resumes from a state");
            (entry.positions_cut_at(hit.tokens), resume.tag)
        };
        let per_token = self.bytes.per_token.max(1) as f64;
        let before = self.used(hit.tier);
        let tier = &mut self.tiers[hit.tier];
        let started = tier.channel_free.max(now);
        let ready =
            started + Time::from_s(units as f64 * per_token / (tier.spec.read_gb_per_s * 1e9));
        tier.channel_free = ready;
        tier.counters.loads += 1;
        tier.counters.loaded_tokens += hit.tokens.saturating_sub(resident);
        tier.counters.loaded_states += 1 + earlier.len() as u64;
        tier.touch(session_id);
        let entry_tokens = tier.entries[&session_id].tokens();
        let released = tier.spec.writes == TierWrites::EvictionsExclusive;
        if released {
            tier.remove(session_id, state_tokens);
            tier.counters.released_entries += 1;
        }
        self.log(
            hit.tier,
            TierEventKind::Read,
            session_id,
            (entry_tokens, if released { 0 } else { entry_tokens }),
            before,
            hit.tokens,
            Some(ready),
        );
        for faster in 0..hit.tier {
            if self.tiers[faster].spec.writes == TierWrites::EveryContext {
                // The copy is a prefix of the session's latest context, as is
                // the faster tier's own entry: the latest states of both stay.
                let kept = self.earlier_for(
                    faster,
                    session_id,
                    hit.tokens.saturating_sub(1),
                    earlier.iter().copied(),
                    RoundState {
                        at: hit.tokens,
                        tag: end,
                    },
                );
                self.store_in(
                    faster,
                    TierEventKind::ReadCopy,
                    session_id,
                    positions,
                    end,
                    kept,
                );
            }
        }
        TierRead {
            started: now,
            ready,
            positions,
            end,
            earlier,
        }
    }

    /// The session's next request shares `declared` tokens with its previous
    /// context: cut each victim entry to what that request can resume from.
    pub(crate) fn cut(&mut self, session_id: u32, declared: u64) {
        let state_tokens = self.state_tokens;
        for tier in 0..self.tiers.len() {
            if !self.tiers[tier].spec.writes.is_victim() {
                continue;
            }
            let Some(entry) = self.tiers[tier].entries.get(&session_id) else {
                continue;
            };
            let tokens = entry.tokens();
            let kept = entry.resume_state(declared).map_or(0, |state| state.at);
            if kept >= tokens {
                continue;
            }
            let before = self.used(tier);
            let mut entry = self.tiers[tier]
                .remove(session_id, state_tokens)
                .expect("checked above");
            if kept > 0 {
                // States past the cut go; one at the cut becomes the end.
                if let Some(cut) = entry.earlier.iter().find(|state| state.at == kept) {
                    entry.end = cut.tag;
                }
                entry.earlier.retain(|state| state.at < kept);
                entry.positions = entry.positions_cut_at(kept);
                self.tiers[tier].insert(session_id, entry, state_tokens);
            }
            self.log(
                tier,
                TierEventKind::Cut,
                session_id,
                (tokens, kept),
                before,
                0,
                None,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DRAM: &str = "dram";
    const SSD: &str = "ssd";

    fn spec(name: &'static str, gb: f64, gb_per_s: f64, writes: TierWrites) -> TierSpec {
        TierSpec {
            name,
            capacity_gb: gb,
            read_gb_per_s: gb_per_s,
            writes,
        }
    }

    /// 1000 B per token: `gb` holds `gb * 1e6` tokens, read at `gb_per_s *
    /// 1e3` tokens per ms.
    fn chain(specs: &[TierSpec], state_bytes: u64, states: u32) -> TierChain {
        TierChain::new(
            specs,
            TierBytes {
                per_token: 1000,
                state: state_bytes,
            },
            StateRetention::Latest { k: states },
        )
    }

    fn dram_ssd(dram: TierWrites, ssd: TierWrites) -> TierChain {
        chain(
            &[spec(DRAM, 1e-3, 1.0, dram), spec(SSD, 1e-2, 0.1, ssd)],
            0,
            1,
        )
    }

    /// A context resumable at its end only, as `context_end` keeps it.
    fn at(tokens: u64) -> ResumePositions {
        ResumePositions::at_state(tokens)
    }

    fn round(tokens: u64, shared: u64) -> RoundContext {
        RoundContext {
            positions: at(tokens),
            shared,
        }
    }

    fn leaving(session_id: u32, tokens: u64, earlier: &[u64]) -> LeavingContext {
        LeavingContext {
            session_id,
            positions: at(tokens),
            end: StateTag::default(),
            earlier: earlier
                .iter()
                .map(|&at| RoundState {
                    at,
                    tag: StateTag::default(),
                })
                .collect(),
        }
    }

    fn hit(chain: &TierChain, session_id: u32, declared: u64) -> Option<(usize, u64)> {
        chain
            .lookup(session_id, declared)
            .map(|hit| (hit.tier, hit.tokens))
    }

    fn positions_of(states: &[RoundState]) -> Vec<u64> {
        states.iter().map(|state| state.at).collect()
    }

    fn entry_states(chain: &TierChain, tier: usize, session_id: u32) -> Vec<u64> {
        chain.tiers[tier]
            .entries
            .get(&session_id)
            .map(|entry| entry.states().map(|state| state.at).collect())
            .unwrap_or_default()
    }

    #[test]
    fn a_store_reaches_every_write_through_tier_and_the_fastest_serves_it() {
        let mut chain = dram_ssd(TierWrites::EveryContext, TierWrites::EveryContext);
        chain.store_round(1, round(400, 0));
        assert_eq!(hit(&chain, 1, 400), Some((0, 400)));
        assert_eq!(hit(&chain, 1, 1_000), Some((0, 400)));
        assert_eq!(
            hit(&chain, 1, 300),
            None,
            "a context-end entry resumes at its end only"
        );
        for tier in 0..2 {
            assert_eq!(chain.counters(tier).stored_entries, 1);
            assert_eq!(chain.counters(tier).written_tokens, 400);
        }
    }

    #[test]
    fn dram_evicts_least_recent_while_ssd_keeps_it() {
        let mut chain = dram_ssd(TierWrites::EveryContext, TierWrites::EveryContext);
        chain.store_round(1, round(600, 0));
        chain.store_round(2, round(600, 0));
        assert_eq!(
            hit(&chain, 1, 600),
            Some((1, 600)),
            "DRAM's 1000 tokens hold one"
        );
        assert_eq!(hit(&chain, 2, 600), Some((0, 600)));
        assert_eq!(chain.counters(0).evicted_entries, 1);
        assert_eq!(chain.counters(0).evicted_tokens, 600);
        assert_eq!(chain.counters(1).evicted_entries, 0);
    }

    #[test]
    fn a_rewrite_counts_only_the_new_tokens() {
        let mut chain = dram_ssd(TierWrites::EveryContext, TierWrites::EveryContext);
        chain.store_round(1, round(300, 0));
        chain.store_round(1, round(500, 300));
        assert_eq!(chain.counters(0).stored_entries, 2);
        assert_eq!(chain.counters(0).written_tokens, 500);
        assert_eq!(chain.used_units(0), 500);
    }

    #[test]
    fn a_context_larger_than_a_tier_drops_its_entry_there() {
        let mut chain = dram_ssd(TierWrites::EveryContext, TierWrites::EveryContext);
        chain.store_round(1, round(300, 0));
        chain.store_round(1, round(2_000, 300));
        assert_eq!(hit(&chain, 1, 2_000), Some((1, 2_000)));
        assert_eq!(chain.used_units(0), 0);
        assert_eq!(chain.counters(0).stored_entries, 1);
    }

    #[test]
    fn reads_queue_on_the_tier_channel_and_copy_into_faster_write_through_tiers() {
        let mut chain = dram_ssd(TierWrites::EveryContext, TierWrites::EveryContext);
        chain.store_round(1, round(600, 0));
        chain.store_round(2, round(600, 0));
        // SSD reads 100 tokens/ms: session 1's 600 take 6 ms, session 2's
        // DRAM read (1000 tokens/ms) its own channel.
        let first = chain.read(1, chain.lookup(1, 600).unwrap(), 0, Time::ZERO);
        assert_eq!(first.ready, Time::from_ms(6.0));
        assert_eq!(chain.queue_wait(1, Time::from_ms(2.0)), Time::from_ms(4.0));
        assert_eq!(
            hit(&chain, 1, 600),
            Some((0, 600)),
            "the read passed through DRAM"
        );
        assert_eq!(
            hit(&chain, 2, 600),
            Some((1, 600)),
            "and evicted session 2 there"
        );
        chain.store_round(3, round(100, 0));
        let second = chain.read(2, chain.lookup(2, 600).unwrap(), 200, Time::from_ms(1.0));
        assert_eq!(second.started, Time::from_ms(1.0));
        assert_eq!(
            second.ready,
            Time::from_ms(10.0),
            "queued behind the first, 400 tokens"
        );
        assert_eq!(chain.counters(1).loads, 2);
        assert_eq!(chain.counters(1).loaded_tokens, 1_000);
    }

    #[test]
    fn a_hybrid_entry_charges_one_state_and_a_read_moves_it() {
        // A 50,000-byte state is 50 tier tokens.
        let mut chain = chain(
            &[spec(DRAM, 1e-3, 1.0, TierWrites::EveryContext)],
            50_000,
            1,
        );
        chain.store_round(1, round(400, 0));
        assert_eq!(chain.used_units(0), 450);
        let hit = chain.lookup(1, 400).unwrap();
        assert_eq!(chain.read_units(1, hit, 100), 300 + 50);
        assert_eq!(
            chain.read(1, hit, 100, Time::ZERO).ready,
            Time::from_ms(0.35)
        );
        assert_eq!(chain.counters(0).loaded_states, 1);
    }

    #[test]
    fn a_victim_tier_takes_only_what_hbm_evicts() {
        let mut chain = dram_ssd(TierWrites::Evictions, TierWrites::EveryContext);
        chain.store_round(1, round(400, 0));
        assert_eq!(
            hit(&chain, 1, 400),
            Some((1, 400)),
            "only SSD is written through"
        );
        chain.take_hbm_eviction(leaving(1, 400, &[]));
        assert_eq!(hit(&chain, 1, 400), Some((0, 400)));
        assert_eq!(chain.counters(0).written_tokens, 400);
    }

    #[test]
    fn a_victim_tier_writes_only_what_its_entry_lacks() {
        let mut chain = dram_ssd(TierWrites::Evictions, TierWrites::Evictions);
        chain.take_hbm_eviction(leaving(1, 400, &[]));
        chain.take_hbm_eviction(leaving(1, 400, &[]));
        assert_eq!(
            chain.counters(0).stored_entries,
            1,
            "an identical context only refreshes"
        );
        chain.take_hbm_eviction(leaving(1, 300, &[]));
        assert_eq!(
            hit(&chain, 1, 400),
            Some((0, 400)),
            "the longer context stays"
        );
        chain.take_hbm_eviction(leaving(1, 700, &[]));
        assert_eq!(chain.counters(0).written_tokens, 700);
    }

    #[test]
    fn a_read_back_keeps_the_victim_entry_and_refreshes_it() {
        let mut chain = dram_ssd(TierWrites::Evictions, TierWrites::Evictions);
        chain.take_hbm_eviction(leaving(1, 500, &[]));
        chain.take_hbm_eviction(leaving(2, 400, &[]));
        chain.read(1, chain.lookup(1, 500).unwrap(), 0, Time::ZERO);
        // DRAM is full: session 2, now the least recent, leaves for SSD.
        chain.take_hbm_eviction(leaving(3, 300, &[]));
        assert_eq!(hit(&chain, 1, 500), Some((0, 500)));
        assert_eq!(hit(&chain, 2, 400), Some((1, 400)));
    }

    #[test]
    fn an_ssd_read_does_not_copy_into_a_victim_dram() {
        let mut chain = dram_ssd(TierWrites::Evictions, TierWrites::EveryContext);
        chain.store_round(1, round(400, 0));
        chain.read(1, chain.lookup(1, 400).unwrap(), 0, Time::ZERO);
        assert_eq!(hit(&chain, 1, 400), Some((1, 400)));
        assert_eq!(chain.counters(0).stored_entries, 0);
    }

    #[test]
    fn an_exclusive_victim_tier_releases_what_a_read_moves_into_hbm() {
        let mut chain = dram_ssd(TierWrites::EvictionsExclusive, TierWrites::Evictions);
        chain.take_hbm_eviction(leaving(1, 400, &[]));
        chain.read(1, chain.lookup(1, 400).unwrap(), 0, Time::ZERO);
        assert_eq!(hit(&chain, 1, 400), None);
        assert_eq!(chain.counters(0).released_entries, 1);
        assert_eq!(chain.used_units(0), 0);
        // HBM evicting it again writes it whole.
        chain.take_hbm_eviction(leaving(1, 400, &[]));
        assert_eq!(chain.counters(0).written_tokens, 800);
    }

    #[test]
    fn a_victim_chain_writes_each_tier_only_with_what_the_one_above_evicts() {
        let mut chain = dram_ssd(TierWrites::Evictions, TierWrites::Evictions);
        chain.take_hbm_eviction(leaving(1, 600, &[]));
        assert_eq!(chain.counters(1).stored_entries, 0);
        chain.take_hbm_eviction(leaving(2, 600, &[]));
        assert_eq!(hit(&chain, 1, 600), Some((1, 600)));
        assert_eq!(hit(&chain, 2, 600), Some((0, 600)));
        assert_eq!(chain.counters(1).written_tokens, 600);
    }

    #[test]
    fn a_write_through_dram_hands_its_evictions_to_a_victim_ssd() {
        let mut chain = dram_ssd(TierWrites::EveryContext, TierWrites::Evictions);
        chain.store_round(1, round(600, 0));
        assert_eq!(chain.counters(1).stored_entries, 0);
        chain.store_round(2, round(600, 0));
        assert_eq!(hit(&chain, 1, 600), Some((1, 600)));
        // A context larger than DRAM goes straight down.
        chain.store_round(3, round(5_000, 0));
        assert_eq!(hit(&chain, 3, 5_000), Some((1, 5_000)));
    }

    #[test]
    fn dram_evicting_a_longer_context_writes_ssd_only_what_it_lacks() {
        let mut chain = dram_ssd(TierWrites::Evictions, TierWrites::Evictions);
        chain.take_hbm_eviction(leaving(1, 400, &[]));
        chain.take_hbm_eviction(leaving(2, 700, &[]));
        assert_eq!(chain.counters(1).written_tokens, 400);
        chain.take_hbm_eviction(leaving(1, 900, &[]));
        // Session 1's 900 evict session 2 (700) into SSD, then session 2's
        // 800 evict session 1's 900, of which SSD lacks 500.
        chain.take_hbm_eviction(leaving(2, 800, &[]));
        assert_eq!(hit(&chain, 2, 800), Some((0, 800)));
        assert_eq!(chain.counters(1).written_tokens, 400 + 700 + 500);
    }

    #[test]
    fn a_round_that_shares_less_cuts_every_victim_tier() {
        let mut chain = dram_ssd(TierWrites::Evictions, TierWrites::Evictions);
        chain.take_hbm_eviction(leaving(1, 600, &[]));
        chain.take_hbm_eviction(leaving(2, 600, &[]));
        chain.take_hbm_eviction(leaving(1, 600, &[]));
        assert_eq!(entry_states(&chain, 1, 1), [600]);
        // A context-end entry shared below its end resumes nowhere: dropped.
        chain.cut(1, 500);
        assert_eq!(hit(&chain, 1, 600), None);
        assert_eq!(chain.used_units(1), 600, "session 2's entry only");
        chain.cut(2, 600);
        assert_eq!(hit(&chain, 2, 600), Some((1, 600)), "shared whole: kept");
    }

    #[test]
    fn a_write_through_tier_is_not_cut() {
        let mut chain = dram_ssd(TierWrites::EveryContext, TierWrites::EveryContext);
        chain.store_round(1, round(600, 0));
        chain.cut(1, 100);
        assert_eq!(hit(&chain, 1, 600), Some((0, 600)));
    }

    fn tagged(at: u64, round: u32) -> RoundState {
        RoundState {
            at,
            tag: StateTag { round },
        }
    }

    #[test]
    fn a_write_through_entry_keeps_the_latest_round_ends_it_shares() {
        let mut chain = chain(&[spec(DRAM, 1e-2, 1e-3, TierWrites::EveryContext)], 0, 3);
        chain.store_round(1, round(100, 0));
        chain.store_round(1, round(200, 100));
        chain.store_round(1, round(300, 200));
        chain.store_round(1, round(400, 300));
        assert_eq!(entry_states(&chain, 0, 1), [200, 300, 400]);
        // A round sharing only 250 tokens keeps the states at or below them.
        chain.store_round(1, round(500, 250));
        assert_eq!(entry_states(&chain, 0, 1), [200, 500]);
        assert_eq!(chain.tiers[0].entries[&1].end, StateTag { round: 2 });
        assert_eq!(chain.earlier_states(0), 1);
    }

    #[test]
    fn earlier_states_serve_a_prompt_that_ends_inside_the_context() {
        let mut chain = chain(&[spec(DRAM, 1e-2, 1e-3, TierWrites::EveryContext)], 0, 2);
        chain.store_round(1, round(100, 0));
        chain.store_round(1, round(300, 100));
        assert_eq!(hit(&chain, 1, 250), Some((0, 100)));
        assert_eq!(hit(&chain, 1, 300), Some((0, 300)));
    }

    #[test]
    fn a_full_tier_evicts_whole_entries_with_all_their_states() {
        // A state is 100 tier tokens; DRAM holds 1000.
        let mut chain = chain(
            &[spec(DRAM, 1e-3, 1e-3, TierWrites::EveryContext)],
            100_000,
            2,
        );
        chain.store_round(1, round(100, 0));
        chain.store_round(1, round(300, 100));
        assert_eq!(chain.used_units(0), 300 + 2 * 100);
        chain.store_round(2, round(600, 0));
        assert_eq!(hit(&chain, 1, 300), None);
        assert_eq!(chain.used_units(0), 700);
        assert_eq!(chain.earlier_states(0), 0);
    }

    #[test]
    fn a_read_moves_only_the_states_that_survive_its_round() {
        let mut chain = chain(
            &[spec(DRAM, 1e-2, 1e-3, TierWrites::EveryContext)],
            10_000,
            3,
        );
        for (tokens, shared) in [(100, 0), (200, 100), (300, 200)] {
            chain.store_round(1, round(tokens, shared));
        }
        // k = 3, one state added per round: the read moves two.
        let hit = chain.lookup(1, 300).unwrap();
        let read = chain.read(1, hit, 0, Time::ZERO);
        assert_eq!(positions_of(&read.earlier), [200]);
        assert_eq!(read.end, StateTag { round: 2 });
        assert_eq!(chain.counters(0).loaded_states, 2);
        // A hit on an earlier state moves the states up to it.
        let hit = chain.lookup(1, 250).unwrap();
        assert_eq!(hit.tokens, 200);
        assert_eq!(positions_of(&chain.read_states(1, hit)), [100]);
        assert_eq!(chain.read_units(1, hit, 0), 200 + 2 * 10);
    }

    #[test]
    fn a_victim_merge_keeps_the_longer_context_and_the_latest_k_states() {
        let mut chain = chain(&[spec(DRAM, 1e-2, 1e-3, TierWrites::Evictions)], 0, 2);
        chain.take_hbm_eviction(leaving(1, 300, &[100]));
        chain.take_hbm_eviction(leaving(1, 200, &[50]));
        assert_eq!(entry_states(&chain, 0, 1), [200, 300]);
        let mut tagged_leaving = leaving(1, 500, &[]);
        tagged_leaving.earlier = vec![tagged(400, 3)];
        chain.take_hbm_eviction(tagged_leaving);
        assert_eq!(entry_states(&chain, 0, 1), [400, 500]);
    }

    #[test]
    fn a_cut_keeps_the_states_at_or_before_it() {
        let mut chain = chain(&[spec(DRAM, 1e-2, 1e-3, TierWrites::Evictions)], 10_000, 3);
        let mut entry = leaving(1, 300, &[]);
        entry.earlier = vec![tagged(100, 0), tagged(200, 1)];
        entry.end = StateTag { round: 2 };
        chain.take_hbm_eviction(entry);
        chain.cut(1, 250);
        assert_eq!(entry_states(&chain, 0, 1), [100, 200]);
        assert_eq!(chain.tiers[0].entries[&1].end, StateTag { round: 1 });
        assert_eq!(chain.used_units(0), 200 + 2 * 10);
        assert_eq!(hit(&chain, 1, 250), Some((0, 200)));
    }

    #[test]
    fn an_exclusive_tier_moves_an_entry_and_its_states_into_hbm() {
        let mut chain = chain(
            &[spec(DRAM, 1e-2, 1e-3, TierWrites::EvictionsExclusive)],
            10_000,
            2,
        );
        chain.take_hbm_eviction(leaving(1, 300, &[100, 200]));
        let read = chain.read(1, chain.lookup(1, 300).unwrap(), 0, Time::ZERO);
        assert!(
            read.earlier.is_empty(),
            "k = 2 moves the resume state alone"
        );
        assert_eq!(chain.used_units(0), 0);
        assert_eq!(chain.earlier_states(0), 0);
    }

    #[test]
    fn a_victim_chain_demotes_states_with_their_kv() {
        let mut chain = chain(
            &[
                spec(DRAM, 1e-3, 1e-3, TierWrites::Evictions),
                spec(SSD, 1e-2, 1e-4, TierWrites::Evictions),
            ],
            50_000,
            2,
        );
        chain.take_hbm_eviction(leaving(1, 400, &[200]));
        chain.take_hbm_eviction(leaving(2, 600, &[]));
        assert_eq!(entry_states(&chain, 1, 1), [200, 400]);
        assert_eq!(chain.counters(1).written_states, 2);
    }

    #[test]
    fn warm_start_seeds_the_slowest_tier() {
        let mut chain = dram_ssd(TierWrites::EveryContext, TierWrites::EveryContext);
        chain.seed(7, at(500));
        assert_eq!(hit(&chain, 7, 500), Some((1, 500)));
        assert_eq!(chain.counters(1).stored_entries, 1);
    }

    #[test]
    fn the_event_log_keeps_writes_evictions_cuts_and_reads() {
        let mut chain = dram_ssd(TierWrites::Evictions, TierWrites::Evictions);
        chain.record_events();
        chain.take_hbm_eviction(leaving(1, 600, &[]));
        chain.take_hbm_eviction(leaving(2, 600, &[]));
        chain.read(2, chain.lookup(2, 600).unwrap(), 100, Time::ZERO);
        chain.cut(1, 10);
        let kinds: Vec<(usize, TierEventKind, u32)> = chain
            .take_events()
            .iter()
            .map(|event| (event.tier, event.kind, event.session_id))
            .collect();
        assert_eq!(
            kinds,
            [
                (0, TierEventKind::Victim, 1),
                (0, TierEventKind::Evict, 1),
                (0, TierEventKind::Victim, 2),
                (1, TierEventKind::Victim, 1),
                (0, TierEventKind::Read, 2),
                (1, TierEventKind::Cut, 1),
            ]
        );
        assert!(chain.take_events().is_empty());
    }

    #[test]
    fn tier_bytes_follow_the_most_loaded_stage_unless_balanced() {
        // PP8 of GLM-5.3-Flash: 11 DSA layers of 545 B, at most 2 per stage.
        let stages: Vec<TierBytes> = [1, 2, 2, 1, 1, 2, 1, 1]
            .into_iter()
            .map(|layers| TierBytes {
                per_token: layers * 545,
                state: 0,
            })
            .collect();
        assert_eq!(TierBytes::of_stages(&stages, false).per_token, 1_090);
        assert_eq!(
            TierBytes::of_stages(&stages, true).per_token,
            750,
            "5995 B over 8 stages, rounded up"
        );
    }

    #[test]
    fn a_layout_stores_its_per_token_parts_and_its_reusable_states_unpadded() {
        use crate::arch::kv_layout::KvCachePart;
        let layout = KvCacheLayout::new()
            .with_part(KvCachePart::per_token("mla", 512))
            .with_part(KvCachePart::compressed("indexer", 4, 132))
            .with_part(KvCachePart::per_request("kda", 1_000).padded_to(4_096))
            .with_part(KvCachePart::per_request("tail", 64).not_prefix_reusable());
        assert_eq!(
            TierBytes::of_layout(&layout),
            TierBytes {
                per_token: 545,
                state: 1_000
            }
        );
    }

    #[test]
    fn tiers_read_one_state_per_context_on_the_same_gpu_as_the_kv() {
        let stages = [(545, 30), (1_090, 20), (1_090, 50), (0, 7)]
            .map(|(per_token, state)| TierBytes { per_token, state });
        assert_eq!(
            TierBytes::of_stages(&stages, false).state,
            20,
            "the first stage with the most KV bytes per token"
        );
        assert_eq!(
            TierBytes::of_stages(&stages, true).state,
            27,
            "107 B over 4 stages, rounded up"
        );
        assert_eq!(TierBytes::of_stages(&[], false), TierBytes::default());
        assert_eq!(
            TierBytes {
                per_token: 1_635,
                state: 39_081_984
            }
            .state_tokens(),
            23_904
        );
    }
}
