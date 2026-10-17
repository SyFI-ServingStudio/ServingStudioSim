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
}

/// The window of [`PrefixTier::window_writes`].
pub(crate) const WRITE_WINDOW: Time = Time::from_ms_u64(10_000);

struct PrefixTier {
    spec: PrefixTierSpec,
    capacity: u64,
    used: u64,
    /// Session -> (context tokens, LRU stamp).
    entries: HashMap<u32, (u64, u64)>,
    lru: BTreeMap<u64, u32>,
    next_stamp: u64,
    /// When the read channel is next free.
    channel_free: Time,
    /// One charged token's bytes, and the read bandwidth in bytes per second.
    bytes_per_token: f64,
    read_bytes_per_s: f64,
    counters: PrefixTierCounters,
    /// Per [`WRITE_WINDOW`] of the clock: (entries stored, tokens written).
    window_writes: Vec<(u64, u64)>,
    clock: Time,
}

impl PrefixTier {
    fn charge(&self, tokens: u64, state_tokens: u64) -> u64 {
        tokens + state_tokens
    }

    fn touch(&mut self, session_id: u32) {
        if let Some(entry) = self.entries.get_mut(&session_id) {
            self.lru.remove(&entry.1);
            entry.1 = self.next_stamp;
            self.lru.insert(self.next_stamp, session_id);
            self.next_stamp += 1;
        }
    }

    /// Cut the session's entry to its first `tokens`, or drop it if that
    /// leaves none.
    fn truncate(&mut self, session_id: u32, tokens: u64, state_tokens: u64) {
        let Some(entry) = self.entries.get_mut(&session_id) else {
            return;
        };
        if tokens >= entry.0 {
            return;
        }
        if tokens == 0 {
            self.remove(session_id, state_tokens);
            return;
        }
        self.used -= entry.0 - tokens;
        entry.0 = tokens;
    }

    fn remove(&mut self, session_id: u32, state_tokens: u64) -> Option<u64> {
        let (tokens, stamp) = self.entries.remove(&session_id)?;
        self.lru.remove(&stamp);
        self.used -= self.charge(tokens, state_tokens);
        Some(tokens)
    }

    /// Store the session's context and return the contexts that leave the
    /// tier for it, as (session, tokens): the LRU victims, or the context
    /// itself when it is larger than the whole tier.
    fn store(&mut self, session_id: u32, tokens: u64, state_tokens: u64) -> Vec<(u32, u64)> {
        let previous = self.remove(session_id, state_tokens).unwrap_or(0);
        let charge = self.charge(tokens, state_tokens);
        if charge > self.capacity {
            return vec![(session_id, tokens)];
        }
        let mut left = Vec::new();
        while self.used + charge > self.capacity {
            let (_, &victim) = self
                .lru
                .iter()
                .next()
                .expect("a full tier has an entry to evict");
            let evicted = self.remove(victim, state_tokens).unwrap();
            self.counters.evicted_entries += 1;
            self.counters.evicted_tokens += evicted;
            left.push((victim, evicted));
        }
        self.entries.insert(session_id, (tokens, self.next_stamp));
        self.lru.insert(self.next_stamp, session_id);
        self.next_stamp += 1;
        self.used += charge;
        self.counters.stored_entries += 1;
        self.counters.written_tokens += tokens.saturating_sub(previous);
        let window = (self.clock.0 / WRITE_WINDOW.0) as usize;
        if self.window_writes.len() <= window {
            self.window_writes.resize(window + 1, (0, 0));
        }
        self.window_writes[window].0 += 1;
        self.window_writes[window].1 += tokens.saturating_sub(previous);
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
                    next_stamp: 0,
                    channel_free: Time::ZERO,
                    bytes_per_token: bytes,
                    read_bytes_per_s: spec.read_gb_per_s_per_gpu * 1e9,
                    counters: PrefixTierCounters::default(),
                    window_writes: Vec::new(),
                    clock: Time::ZERO,
                }
            })
            .collect();
        Self {
            tiers,
            state_tokens,
            resume,
        }
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
            if let Some(&(tokens, _)) = state.entries.get(&session_id) {
                let tokens = self.resume.reusable_tokens(tokens, u64::from(requested)) as u32;
                if tokens > 0 && best.is_none_or(|hit| tokens > hit.tokens) {
                    best = Some(PrefixTierHit { tier, tokens });
                }
            }
        }
        best
    }

    /// Queue a read of `hit` behind the tier's earlier reads and return when
    /// it lands. Only the tokens past `resident_tokens` (what HBM still holds)
    /// cross the link, plus the state at the hit's end. The entry stays (the
    /// tiers are inclusive); every faster write-through tier receives a copy,
    /// since the read passes through it. A faster victim tier receives it
    /// only when the tier above it evicts it.
    pub(crate) fn load(
        &mut self,
        session_id: u32,
        hit: PrefixTierHit,
        resident_tokens: u32,
        now: Time,
    ) -> Time {
        let state_tokens = self.state_tokens;
        let tier = &mut self.tiers[hit.tier];
        let charge = tier.charge(
            u64::from(hit.tokens.saturating_sub(resident_tokens)),
            state_tokens,
        );
        let start = tier.channel_free.max(now);
        let ready =
            start + Time::from_s(charge as f64 * tier.bytes_per_token / tier.read_bytes_per_s);
        tier.channel_free = ready;
        tier.counters.loads += 1;
        tier.counters.loaded_tokens += u64::from(hit.tokens.saturating_sub(resident_tokens));
        tier.touch(session_id);
        for faster in 0..hit.tier {
            if self.tiers[faster].spec.writes == PrefixTierWrites::EveryContext {
                self.store_in(faster, session_id, u64::from(hit.tokens));
            }
        }
        ready
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
            // The slowest tier has no victim tier behind it.
            let _ = tier.store(session_id, tokens, state_tokens);
        }
    }

    /// Write-through of a finished request's context to every tier but a
    /// victim tier; what that pushes out of a tier goes down to a victim tier
    /// behind it.
    pub(crate) fn store(&mut self, session_id: u32, tokens: u64) {
        for tier in 0..self.tiers.len() {
            if self.tiers[tier].spec.writes == PrefixTierWrites::EveryContext {
                self.store_in(tier, session_id, tokens);
            }
        }
    }

    /// Store into `tier`, and pass what leaves it to the tier behind it if
    /// that one is its victim tier.
    fn store_in(&mut self, tier: usize, session_id: u32, tokens: u64) {
        let state_tokens = self.state_tokens;
        let left = self.tiers[tier].store(session_id, tokens, state_tokens);
        for (victim, victim_tokens) in left {
            self.take_victim(tier + 1, victim, victim_tokens);
        }
    }

    /// The tier above `tier` let go of the session's `tokens`-token context:
    /// if `tier` is its victim tier, write what its entry lacks, or only
    /// refresh an entry that already holds it all.
    fn take_victim(&mut self, tier: usize, session_id: u32, tokens: u64) {
        let Some(state) = self
            .tiers
            .get_mut(tier)
            .filter(|state| state.spec.writes == PrefixTierWrites::FasterTierEvictions)
        else {
            return;
        };
        if state
            .entries
            .get(&session_id)
            .is_some_and(|e| e.0 >= tokens)
        {
            state.touch(session_id);
        } else {
            self.store_in(tier, session_id, tokens);
        }
    }

    /// Whether the fastest tier is HBM's victim tier.
    pub(crate) fn takes_hbm_evictions(&self) -> bool {
        self.tiers
            .first()
            .is_some_and(|tier| tier.spec.writes == PrefixTierWrites::FasterTierEvictions)
    }

    /// HBM evicted the session's `tokens`-token context, or could not keep it:
    /// a victim fastest tier writes what its entry lacks, or only refreshes an
    /// entry that already holds it all.
    pub(crate) fn store_evicted(&mut self, session_id: u32, tokens: u64) {
        self.take_victim(0, session_id, tokens);
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

    /// The session's next round shares `declared` tokens with its previous
    /// context: cut a victim tier's older entry to what that round can resume
    /// from, so it stays a prefix of the session's latest context.
    pub(crate) fn drop_stale(&mut self, session_id: u32, declared: u64) {
        let state_tokens = self.state_tokens;
        let resume = self.resume;
        for tier in &mut self.tiers {
            if tier.spec.writes != PrefixTierWrites::FasterTierEvictions {
                continue;
            }
            if let Some(&(tokens, _)) = tier.entries.get(&session_id) {
                tier.truncate(
                    session_id,
                    resume.reusable_tokens(tokens, declared),
                    state_tokens,
                );
            }
        }
    }

    pub(crate) fn counters(&self, tier: usize) -> PrefixTierCounters {
        self.tiers[tier].counters
    }

    pub(crate) fn used_tokens(&self, tier: usize) -> u64 {
        self.tiers[tier].used
    }

    pub(crate) fn capacity_tokens(&self, tier: usize) -> u64 {
        self.tiers[tier].capacity
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
        let first = t.load(1, hit, 0, Time::ZERO);
        assert_eq!(first, Time::from_ms(400.0));
        // A second SSD read waits for the first.
        t.store(4, 400_000);
        t.store(5, 400_000);
        let second = t.load(2, t.lookup(2, 400_000).unwrap(), 0, Time::from_ms(100.0));
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
        let ready = t.load(1, t.lookup(1, 100_000).unwrap(), 0, Time::ZERO);
        assert_eq!(ready, Time::from_ms(150.0));
    }

    #[test]
    fn a_read_moves_only_what_hbm_lacks() {
        let mut t = tiers(1.0, 10.0);
        t.store(1, 400_000);
        // HBM still holds 300k: 100k cross the 10 GB/s DRAM link.
        let ready = t.load(1, t.lookup(1, 400_000).unwrap(), 300_000, Time::ZERO);
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
        t.store_evicted(1, 400_000);
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
        t.store_evicted(1, 400_000);
        // The next round extends the entry; HBM then evicts the longer context.
        t.drop_stale(1, 400_001);
        t.store_evicted(1, 500_000);
        assert_eq!(t.counters(0).written_tokens, 500_000);
        assert_eq!(t.used_tokens(0), 500_000);
        // An eviction of what the entry already holds writes nothing.
        let entries = t.counters(0).stored_entries;
        t.store_evicted(1, 500_000);
        assert_eq!(t.counters(0).stored_entries, entries);
        assert_eq!(t.counters(0).written_tokens, 500_000);
    }

    #[test]
    fn a_read_back_keeps_the_victim_entry_and_refreshes_it() {
        let mut t = victim_tiers(1.0, 10.0);
        t.store_evicted(1, 400_000);
        t.store_evicted(2, 400_000);
        // Reading 1 back into HBM keeps its DRAM copy and makes 2 the LRU.
        t.load(1, t.lookup(1, 400_000).unwrap(), 0, Time::ZERO);
        assert_eq!(t.lookup(1, 400_000).unwrap().tier, 0);
        // HBM evicts 1 again unchanged: no write; then 3 pushes out 2.
        t.store_evicted(1, 400_000);
        t.store_evicted(3, 400_000);
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
    fn a_round_that_does_not_extend_the_entry_cuts_it() {
        let mut t = victim_tiers(1.0, 10.0);
        t.store_evicted(1, 600_000);
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
        t.store_evicted(1, 100_000);
        t.drop_stale(1, 100_001);
        assert_eq!(t.used_tokens(0), 150_000);
        t.drop_stale(1, 99_000);
        assert_eq!(t.used_tokens(0), 0);
        assert_eq!(t.lookup(1, 200_000), None);
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
        t.store_evicted(1, 400_000);
        t.store_evicted(2, 400_000);
        assert_eq!(t.counters(1).stored_entries, 0);
        // A third eviction from HBM pushes session 1 out of DRAM into SSD.
        t.store_evicted(3, 400_000);
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
        t.store_evicted(1, 400_000);
        t.store_evicted(2, 400_000);
        t.store_evicted(3, 400_000); // 1 goes to SSD
        t.store_evicted(4, 400_000); // 2 goes to SSD; SSD's LRU: 1, 2
        t.load(1, t.lookup(1, 400_000).unwrap(), 0, Time::ZERO);
        // The read neither copies 1 into DRAM nor drops SSD's copy, and it
        // refreshes 1 in SSD's order.
        assert_eq!(t.counters(0).stored_entries, 4);
        assert_eq!(t.lookup(1, 400_000).unwrap().tier, 1);
        assert_eq!(t.tiers[1].lru.values().copied().collect::<Vec<_>>(), [2, 1]);
        // HBM evicts 1 unchanged: it lands in DRAM and pushes 3 down to SSD.
        t.store_evicted(1, 400_000);
        assert_eq!(t.lookup(1, 400_000).unwrap().tier, 0);
        let ssd = t.counters(1);
        // DRAM later evicts 1, which SSD holds whole: SSD is only refreshed.
        t.store_evicted(5, 400_000); // pushes out 4
        t.store_evicted(6, 400_000); // pushes out 1
        assert_eq!(t.counters(1).stored_entries, ssd.stored_entries + 1);
        assert_eq!(t.counters(1).written_tokens, ssd.written_tokens + 400_000);
        assert_eq!(t.lookup(1, 400_000).unwrap().tier, 1);
        assert_eq!(t.tiers[1].lru.values().last(), Some(&1));
    }

    #[test]
    fn dram_evicting_a_longer_context_writes_ssd_only_what_it_lacks() {
        let mut t = chain(1.0, 10.0);
        t.store_evicted(1, 400_000);
        t.store_evicted(2, 400_000);
        t.store_evicted(3, 400_000); // 1 goes to SSD at 400k
                                     // Session 1's next round extends it; HBM later evicts 500k.
        t.drop_stale(1, 400_001);
        t.store_evicted(1, 500_000); // pushes 2 down
        t.store_evicted(4, 400_000); // pushes 3 down
        t.store_evicted(5, 400_000); // pushes 1 down: SSD adds its 100k new tokens
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
        t.store_evicted(1, 600_000);
        t.store_evicted(2, 600_000); // 1 goes to SSD at 600k
        t.store_evicted(1, 700_000); // HBM evicts a longer context into DRAM
        t.drop_stale(1, 300_000);
        assert_eq!(t.lookup(1, 700_000).unwrap().tokens, 300_000);
        assert_eq!(t.tiers[1].entries[&1].0, 300_000);
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
        t.store_evicted(1, 400_000);
        t.set_clock(Time::from_ms(25_000.0));
        t.store_evicted(2, 400_000);
        t.store_evicted(3, 300_000); // pushes 1 down to SSD
        assert_eq!(t.window_writes(0), [(1, 400_000), (0, 0), (2, 700_000)]);
        assert_eq!(t.window_writes(1), [(0, 0), (0, 0), (1, 400_000)]);
    }
}
