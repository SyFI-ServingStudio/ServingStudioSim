//! Session-scoped prefix-cache model (L5).
//!
//! A worker (or DP group) retains the KV of completed requests, keyed by the
//! request's `session` id, under a token-capacity budget with a configurable
//! eviction policy. A later request declaring `prefix_kv` (the trace-measured
//! cached prefix) HITS up to the session's retained tokens; the shortfall is a
//! MISS the worker must recompute (`prefill_target` grows by the missed tokens).
//!
//! This models the two mechanisms an always-hit replay ignores: capacity
//! pressure (eviction under load) and placement (the prefix only exists where
//! the session previously ran — pair with `session-sticky` placement).
//! Content-level dedup across sessions (radix-tree sharing) is out of scope:
//! the trace carries token counts, not tokens.

use std::collections::HashMap;

use serde::Deserialize;

/// Which resident session to evict when an insert exceeds capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EvictPolicy {
    /// Least-recently-used: hits refresh recency (the default).
    #[default]
    Lru,
    /// Insertion order only; hits do not refresh.
    Fifo,
    /// Fewest cache hits since first residency (ties → oldest).
    Lfu,
    /// Largest resident session first (maximizes the number of retained
    /// sessions at the cost of evicting the most expensive-to-recompute one).
    LargestFirst,
}

impl EvictPolicy {
    /// Parse the config wire spelling (kebab-case, as in the preset YAML).
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s {
            "lru" => Ok(Self::Lru),
            "fifo" => Ok(Self::Fifo),
            "lfu" => Ok(Self::Lfu),
            "largest-first" => Ok(Self::LargestFirst),
            other => anyhow::bail!(
                "unknown prefix_cache_policy `{other}` (choices: lru, fifo, lfu, largest-first)"
            ),
        }
    }
}

/// Map session → retained KV tokens, bounded by `capacity_tokens`, with
/// policy-selected eviction. `order` holds sessions oldest-first (recency
/// order under LRU, insertion order otherwise).
#[derive(Debug)]
pub struct PrefixCache {
    capacity_tokens: u64,
    used_tokens: u64,
    policy: EvictPolicy,
    /// Retained tokens per session; `order` holds sessions least-recent-first.
    resident: HashMap<u32, u64>,
    order: Vec<u32>,
    /// Hits per session since it first became resident (LFU victim key; kept
    /// across re-inserts so a session's popularity survives its next round).
    freq: HashMap<u32, u64>,
}

impl PrefixCache {
    pub fn new(capacity_tokens: u64, policy: EvictPolicy) -> Self {
        Self {
            capacity_tokens,
            used_tokens: 0,
            policy,
            resident: HashMap::new(),
            order: Vec::new(),
            freq: HashMap::new(),
        }
    }

    /// Tokens of `want_prefix` resident for `session` (0 on a cold session).
    /// A hit bumps the session's frequency and (under LRU) its recency.
    pub fn lookup_touch(&mut self, session: u32, want_prefix: u32) -> u32 {
        let Some(&tokens) = self.resident.get(&session) else {
            return 0;
        };
        *self.freq.entry(session).or_insert(0) += 1;
        if self.policy == EvictPolicy::Lru {
            self.touch(session);
        }
        tokens.min(u64::from(want_prefix)) as u32
    }

    /// Retain `tokens` for `session` (replacing any prior entry), evicting
    /// policy-selected victims until it fits. An entry larger than the whole
    /// capacity is clamped — the session keeps its most useful suffix.
    pub fn insert(&mut self, session: u32, tokens: u64) {
        self.remove(session);
        let tokens = tokens.min(self.capacity_tokens);
        if tokens == 0 {
            return;
        }
        while self.used_tokens + tokens > self.capacity_tokens {
            let Some(victim) = self.victim() else {
                break;
            };
            self.remove(victim);
            self.freq.remove(&victim);
        }
        self.resident.insert(session, tokens);
        self.order.push(session);
        self.used_tokens += tokens;
    }

    /// Policy-selected eviction victim among resident sessions.
    fn victim(&self) -> Option<u32> {
        match self.policy {
            // `order` is recency order under LRU, insertion order under FIFO —
            // the front is the right victim for both.
            EvictPolicy::Lru | EvictPolicy::Fifo => self.order.first().copied(),
            // First minimum → oldest among the least-hit.
            EvictPolicy::Lfu => self
                .order
                .iter()
                .copied()
                .min_by_key(|s| self.freq.get(s).copied().unwrap_or(0)),
            // Reverse-min → oldest among the largest.
            EvictPolicy::LargestFirst => self
                .order
                .iter()
                .copied()
                .min_by_key(|s| std::cmp::Reverse(self.resident.get(s).copied().unwrap_or(0))),
        }
    }

    fn touch(&mut self, session: u32) {
        if let Some(pos) = self.order.iter().position(|&s| s == session) {
            self.order.remove(pos);
            self.order.push(session);
        }
    }

    fn remove(&mut self, session: u32) {
        if let Some(tokens) = self.resident.remove(&session) {
            self.used_tokens -= tokens;
            if let Some(pos) = self.order.iter().position(|&s| s == session) {
                self.order.remove(pos);
            }
        }
    }

    #[cfg(test)]
    pub fn used_tokens(&self) -> u64 {
        self.used_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_session_misses_then_hits_after_insert() {
        let mut c = PrefixCache::new(1000, EvictPolicy::Lru);
        assert_eq!(c.lookup_touch(7, 500), 0);
        c.insert(7, 400);
        assert_eq!(c.lookup_touch(7, 500), 400); // partial: retained < wanted
        assert_eq!(c.lookup_touch(7, 300), 300); // full: retained >= wanted
    }

    #[test]
    fn lru_eviction_under_capacity_pressure() {
        let mut c = PrefixCache::new(1000, EvictPolicy::Lru);
        c.insert(1, 400);
        c.insert(2, 400);
        c.lookup_touch(1, 1); // refresh 1 → 2 becomes LRU
        c.insert(3, 400); // evicts 2
        assert_eq!(c.lookup_touch(2, 400), 0);
        assert_eq!(c.lookup_touch(1, 400), 400);
        assert_eq!(c.lookup_touch(3, 400), 400);
        assert!(c.used_tokens() <= 1000);
    }

    #[test]
    fn fifo_ignores_recency() {
        let mut c = PrefixCache::new(1000, EvictPolicy::Fifo);
        c.insert(1, 400);
        c.insert(2, 400);
        c.lookup_touch(1, 1); // would refresh under LRU; FIFO ignores it
        c.insert(3, 400); // evicts 1 (oldest insert)
        assert_eq!(c.lookup_touch(1, 400), 0);
        assert_eq!(c.lookup_touch(2, 400), 400);
    }

    #[test]
    fn lfu_evicts_least_hit_session() {
        let mut c = PrefixCache::new(1000, EvictPolicy::Lfu);
        c.insert(1, 400);
        c.insert(2, 400);
        c.lookup_touch(1, 1);
        c.lookup_touch(1, 1); // freq: 1→2, 2→0
        c.insert(3, 400); // evicts 2 (fewest hits)
        assert_eq!(c.lookup_touch(2, 400), 0);
        assert_eq!(c.lookup_touch(1, 400), 400);
    }

    #[test]
    fn largest_first_evicts_biggest_session() {
        let mut c = PrefixCache::new(1000, EvictPolicy::LargestFirst);
        c.insert(1, 700);
        c.insert(2, 200);
        c.insert(3, 300); // over budget → evicts 1 (largest), keeps 2+3
        assert_eq!(c.lookup_touch(1, 700), 0);
        assert_eq!(c.lookup_touch(2, 200), 200);
        assert_eq!(c.lookup_touch(3, 300), 300);
        assert_eq!(c.used_tokens(), 500);
    }

    #[test]
    fn oversized_session_clamps_to_capacity() {
        let mut c = PrefixCache::new(1000, EvictPolicy::Lru);
        c.insert(1, 5000);
        assert_eq!(c.lookup_touch(1, 4000), 1000);
        assert_eq!(c.used_tokens(), 1000);
    }
}
