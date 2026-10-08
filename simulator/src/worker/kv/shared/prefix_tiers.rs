//! Host DRAM and SSD tiers behind the HBM retained-prefix cache.
//!
//! A tier is a per-replica LRU of session contexts, sized and loaded per GPU:
//! every GPU of the replica keeps its own slice of each token (a pipeline stage
//! its layers' KV), so capacity and load time are one GPU's bytes over one
//! GPU's capacity and link, and all GPUs load their slices in parallel.
//!
//! Writes are write-through, as LMCache and vLLM's offloading connector store
//! a request's KV when it finishes: every tier receives the session's latest
//! context, and each evicts on its own. A tier never holds an older context
//! than a faster one, so a lookup takes the fastest tier with the longest hit.
//!
//! Loads queue FIFO on one channel per tier (its read bandwidth), from the
//! moment the request arrives, before admission. Writes are free: their
//! bandwidth is not modeled, only counted.
//!
//! Units are KV tokens of the most loaded GPU (the pipeline's
//! `kv_bytes_per_token`). A hybrid model's entry also charges one recurrent
//! state: a tier keeps the state at the context's end, the only one a session
//! resumes from, not the per-block snapshots the HBM tier holds.

use std::collections::{BTreeMap, HashMap};

use crate::common::Time;

/// One tier's size and read bandwidth, per GPU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrefixTierSpec {
    pub name: &'static str,
    pub capacity_gb_per_gpu: f64,
    pub read_gb_per_s_per_gpu: f64,
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

    fn remove(&mut self, session_id: u32, state_tokens: u64) -> Option<u64> {
        let (tokens, stamp) = self.entries.remove(&session_id)?;
        self.lru.remove(&stamp);
        self.used -= self.charge(tokens, state_tokens);
        Some(tokens)
    }

    fn store(&mut self, session_id: u32, tokens: u64, state_tokens: u64) {
        let previous = self.remove(session_id, state_tokens).unwrap_or(0);
        let charge = self.charge(tokens, state_tokens);
        if charge > self.capacity {
            return;
        }
        while self.used + charge > self.capacity {
            let (_, &victim) = self
                .lru
                .iter()
                .next()
                .expect("a full tier has an entry to evict");
            let evicted = self.remove(victim, state_tokens).unwrap();
            self.counters.evicted_entries += 1;
            self.counters.evicted_tokens += evicted;
        }
        self.entries.insert(session_id, (tokens, self.next_stamp));
        self.lru.insert(self.next_stamp, session_id);
        self.next_stamp += 1;
        self.used += charge;
        self.counters.stored_entries += 1;
        self.counters.written_tokens += tokens.saturating_sub(previous);
    }
}

/// The DRAM/SSD tiers of one replica, fastest first.
pub(crate) struct PrefixTiers {
    tiers: Vec<PrefixTier>,
    /// One recurrent state in KV tokens (0 for a full-attention model).
    state_tokens: u64,
}

impl PrefixTiers {
    /// `kv_bytes_per_token`: one GPU's bytes per token; `state_tokens`: one
    /// recurrent state in those tokens.
    pub(crate) fn new(
        specs: &[PrefixTierSpec],
        kv_bytes_per_token: u64,
        state_tokens: u64,
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
                }
            })
            .collect();
        Self {
            tiers,
            state_tokens,
        }
    }

    pub(crate) fn tier_name(&self, tier: usize) -> &'static str {
        self.tiers[tier].spec.name
    }

    pub(crate) fn len(&self) -> usize {
        self.tiers.len()
    }

    /// The fastest tier holding the most of `requested` tokens of the session.
    pub(crate) fn lookup(&self, session_id: u32, requested: u32) -> Option<PrefixTierHit> {
        let mut best: Option<PrefixTierHit> = None;
        for (tier, state) in self.tiers.iter().enumerate() {
            if let Some(&(tokens, _)) = state.entries.get(&session_id) {
                let tokens = tokens.min(u64::from(requested)) as u32;
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
    /// tiers are inclusive); every faster tier receives a copy, since the read
    /// passes through it.
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
        for faster in &mut self.tiers[..hit.tier] {
            faster.store(session_id, u64::from(hit.tokens), state_tokens);
        }
        ready
    }

    /// Write-through of a finished request's context to every tier.
    pub(crate) fn store(&mut self, session_id: u32, tokens: u64) {
        let state_tokens = self.state_tokens;
        for tier in &mut self.tiers {
            tier.store(session_id, tokens, state_tokens);
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

    fn tiers(dram_gb: f64, ssd_gb: f64) -> PrefixTiers {
        // 1000 B/token: 1 GB = 1M tokens; 1 GB/s reads 1M tokens per second.
        PrefixTiers::new(
            &[
                PrefixTierSpec {
                    name: "dram",
                    capacity_gb_per_gpu: dram_gb,
                    read_gb_per_s_per_gpu: 10.0,
                },
                PrefixTierSpec {
                    name: "ssd",
                    capacity_gb_per_gpu: ssd_gb,
                    read_gb_per_s_per_gpu: 1.0,
                },
            ],
            1000,
            0,
        )
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
            }],
            1000,
            50_000,
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
}
