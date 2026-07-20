//! Session-scoped prefix-cache model (L5).
//!
//! A worker (or DP group) retains the KV of completed requests, keyed by the
//! request's `session` id, under a token-capacity budget with LRU eviction.
//! A later request declaring `prefix_kv` (the trace-measured cached prefix)
//! HITS up to the session's retained tokens; the shortfall is a MISS the worker
//! must recompute (`prefill_target` grows by the missed tokens).
//!
//! This models the two mechanisms an always-hit replay ignores: capacity
//! pressure (eviction under load) and placement (the prefix only exists where
//! the session previously ran — pair with `session-sticky` placement).
//! Content-level dedup across sessions (radix-tree sharing) is out of scope:
//! the trace carries token counts, not tokens.

use std::collections::HashMap;

/// LRU map session → retained KV tokens, bounded by `capacity_tokens`.
#[derive(Debug)]
pub struct PrefixCache {
    capacity_tokens: u64,
    used_tokens: u64,
    /// Retained tokens per session; `order` holds sessions least-recent-first.
    resident: HashMap<u32, u64>,
    order: Vec<u32>,
}

impl PrefixCache {
    pub fn new(capacity_tokens: u64) -> Self {
        Self {
            capacity_tokens,
            used_tokens: 0,
            resident: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Tokens of `want_prefix` resident for `session` (0 on a cold session).
    /// A hit refreshes the session's LRU position.
    pub fn lookup_touch(&mut self, session: u32, want_prefix: u32) -> u32 {
        let Some(&tokens) = self.resident.get(&session) else {
            return 0;
        };
        self.touch(session);
        tokens.min(u64::from(want_prefix)) as u32
    }

    /// Retain `tokens` for `session` (replacing any prior entry), evicting
    /// least-recently-used sessions until it fits. An entry larger than the
    /// whole capacity is clamped — the session keeps its most useful suffix.
    pub fn insert(&mut self, session: u32, tokens: u64) {
        self.remove(session);
        let tokens = tokens.min(self.capacity_tokens);
        if tokens == 0 {
            return;
        }
        while self.used_tokens + tokens > self.capacity_tokens {
            let Some(&victim) = self.order.first() else {
                break;
            };
            self.remove(victim);
        }
        self.resident.insert(session, tokens);
        self.order.push(session);
        self.used_tokens += tokens;
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
        let mut c = PrefixCache::new(1000);
        assert_eq!(c.lookup_touch(7, 500), 0);
        c.insert(7, 400);
        assert_eq!(c.lookup_touch(7, 500), 400); // partial: retained < wanted
        assert_eq!(c.lookup_touch(7, 300), 300); // full: retained >= wanted
    }

    #[test]
    fn lru_eviction_under_capacity_pressure() {
        let mut c = PrefixCache::new(1000);
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
    fn oversized_session_clamps_to_capacity() {
        let mut c = PrefixCache::new(1000);
        c.insert(1, 5000);
        assert_eq!(c.lookup_touch(1, 4000), 1000);
        assert_eq!(c.used_tokens(), 1000);
    }
}
