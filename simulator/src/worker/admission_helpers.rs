//! Shared L5 primitives: KV resource + per-group container + the two pure
//! policies. See L5 design.md §2. These helpers are shared by barebone, HP/DP,
//! and PD workers: per-group `Batch`, `Strict` / `Tentative` admission, and
//! `Single` / `RoundRobin` load balance. Reentry / chunked-prefill fields from
//! §2.2 are omitted until those features land (§3.6 / §3.7).
//!
//! std-only on purpose — `decodes` is a `Vec<(RequestId, DecodeReqState)>` to
//! keep insertion order deterministic without pulling in `indexmap`/`rustc_hash`.

use crate::common::RequestId;

// ════════════════════════════════════════════════════════════════════════════
// KvPool — passive KV resource (§2.1). Holds no per-request state; the decode
// view is supplied by the owning `Batch` for projection.
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct KvPool {
    pub kv_capacity: u64,
    pub active_kv: u64,
}

impl KvPool {
    pub fn new(kv_capacity: u64) -> Self {
        Self {
            kv_capacity,
            active_kv: 0,
        }
    }

    pub fn remaining_now(&self) -> u64 {
        self.kv_capacity.saturating_sub(self.active_kv)
    }

    pub fn add_kv(&mut self, n: u64) {
        self.active_kv += n;
    }

    pub fn sub_kv(&mut self, n: u64) {
        self.active_kv = self.active_kv.saturating_sub(n);
    }

    pub fn reset(&mut self) {
        self.active_kv = 0;
    }

    /// Peak cumulative KV from now until every live decode drains, given the
    /// decode view. KV(t) rises linearly between decode-exit steps (every
    /// still-live decode emits one token/step) and drops when a decode finishes
    /// and frees its KV, so the peak always lands *at* an exit step. Sort decodes
    /// by `remaining_decode` ascending, then evaluate KV(t) at a sampled subset
    /// of those exit steps (the full set when `n < 8`):
    ///
    ///   KV(t) = active_kv + (n − k)·t − kv_prefix[k]
    ///
    /// where the `n − k` survivors at `t` each grew by `t` tokens and the `k`
    /// decodes that finished before `t` released their KV (`kv_prefix[k]`).
    /// Sampling makes this a heuristic guard, not an exact bound. O(n log n).
    /// (§2.1)
    pub fn projected_peak<'a>(
        &self,
        decodes: impl Iterator<Item = &'a DecodeReqState>,
    ) -> u64 {
        let mut entries: Vec<(u32, u64)> =
            decodes.map(|s| (s.remaining_decode, s.current_kv)).collect();
        if entries.is_empty() {
            return self.active_kv;
        }
        entries.sort_by_key(|(rem, _)| *rem);

        let n = entries.len();
        // kv_prefix[k] = Σ current_kv of the k soonest-finishing decodes.
        let mut kv_prefix = Vec::with_capacity(n + 1);
        kv_prefix.push(0u64);
        for (_, kv) in &entries {
            kv_prefix.push(kv_prefix.last().unwrap() + kv);
        }

        let mut peak = self.active_kv;
        let mut last_t: Option<u32> = None;
        for idx in Self::sample_indices(n) {
            let t = entries[idx].0;
            if last_t == Some(t) {
                continue;
            }
            last_t = Some(t);
            // Decodes that finish strictly before `t` are gone by then.
            let k = entries.partition_point(|(rem, _)| *rem < t);
            let kv_at_t = self.active_kv + (n - k) as u64 * t as u64 - kv_prefix[k];
            peak = peak.max(kv_at_t);
        }
        peak
    }

    /// Indices into the `remaining`-sorted decode list at which to evaluate
    /// `KV(t)`. Exact (all of them) for `n < 8`; otherwise the front 8 plus 8
    /// evenly-spaced probes — the front is sampled densely because the peak is
    /// usually early, while most decodes are still live.
    fn sample_indices(n: usize) -> Vec<usize> {
        if n == 0 {
            return Vec::new();
        }
        if n < 8 {
            return (0..n).collect();
        }
        let mut indices: Vec<usize> = (0..8).collect();
        for k in 1..8 {
            indices.push(k * n / 8);
        }
        indices.push(n - 1);
        indices.sort_unstable();
        indices.dedup();
        indices
    }
}

#[derive(Clone, Debug)]
pub struct DecodeReqState {
    pub current_kv: u64,
    pub remaining_decode: u32,
}

// ════════════════════════════════════════════════════════════════════════════
// Batch — per-group sticky container (§2.2). Current workers do not carry the
// reentry heap or chunked-prefill queue yet.
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct Batch {
    pub group_id: u16,
    pub kv: KvPool,
    /// Insertion-ordered live decode set.
    pub decodes: Vec<(RequestId, DecodeReqState)>,
    /// Transient: ids admitted as prefill this iter; cleared in `complete_iter`.
    pub prefill_admits: Vec<RequestId>,
    /// Cached `projected_peak`. The projection is non-increasing as decodes
    /// advance (each live decode's per-step growth is already counted in the
    /// peak), so only a decode *entering* (`finalize_to_decode`) can raise it and
    /// only one *leaving* (`release`) lowers it. We recompute the sort on those
    /// two events and serve `projected_peak_kv` from the cache, skipping the
    /// O(n log n) sort on every admission probe.
    cached_peak: u64,
}

impl Batch {
    pub fn new(group_id: u16, kv_capacity: u64) -> Self {
        Self {
            group_id,
            kv: KvPool::new(kv_capacity),
            decodes: Vec::new(),
            prefill_admits: Vec::new(),
            cached_peak: 0,
        }
    }

    /// Live decodes (those with tokens still to emit), in insertion order.
    pub fn iter_decoding(&self) -> impl Iterator<Item = (RequestId, &DecodeReqState)> {
        self.decodes
            .iter()
            .filter(|(_, s)| s.remaining_decode > 0)
            .map(|(r, s)| (*r, s))
    }

    /// Iter-end: every live decode produced one token this iter.
    pub fn advance_decodes(&mut self) {
        for (_, s) in &mut self.decodes {
            if s.remaining_decode > 0 {
                s.current_kv += 1;
                s.remaining_decode -= 1;
                self.kv.add_kv(1);
            }
        }
    }

    /// Iter-end transition: a realized prefill enters the decode set.
    pub fn finalize_to_decode(&mut self, req_id: RequestId, initial_kv: u64, decode_budget: u32) {
        self.kv.add_kv(initial_kv);
        self.decodes.push((
            req_id,
            DecodeReqState {
                current_kv: initial_kv,
                remaining_decode: decode_budget,
            },
        ));
        // A new decode is the only event that can raise the projected peak.
        self.recompute_peak();
    }

    /// External release: drop from the decode set (if present) and decrement KV.
    pub fn release(&mut self, req_id: RequestId, current_kv: u64) {
        if let Some(pos) = self.decodes.iter().position(|(r, _)| *r == req_id) {
            self.decodes.remove(pos);
            self.kv.sub_kv(current_kv);
            // One decode left → peak can only fall; refresh the cache.
            self.recompute_peak();
        }
    }

    /// Run the O(n log n) `projected_peak` and store it. Called only on the two
    /// events that change the projection — a decode entering or leaving.
    fn recompute_peak(&mut self) {
        self.cached_peak = self.kv.projected_peak(self.decodes.iter().map(|(_, s)| s));
    }

    pub fn projected_peak_kv(&self) -> u64 {
        self.cached_peak
    }
}

// ════════════════════════════════════════════════════════════════════════════
// KvAdmission — pure decision (§2.3). Barebone only uses `Strict`.
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
pub enum KvAdmission {
    Strict,
    Tentative { headroom_tokens: u64 },
}

impl KvAdmission {
    /// Will `(p + d)` tokens fit on top of the group's projected peak and the
    /// already-promised (admitted-not-yet-realized) tokens?
    pub fn try_admit(&self, group: &Batch, group_promised: u64, p: u32, d: u32) -> bool {
        let cap = match self {
            Self::Strict => group.kv.kv_capacity,
            Self::Tentative { headroom_tokens } => {
                group.kv.kv_capacity.saturating_sub(*headroom_tokens)
            }
        };
        let demand = group_promised + (p + d) as u64;
        // `projected_peak` is always ≥ `active_kv` (it starts there and only
        // takes max), so `active_kv + demand ≤ cap` is a *necessary* condition
        // for admission. Test that cheap bound before the O(n log n)
        // `projected_peak_kv` sort: on a saturated, KV-full pool (admission
        // rejected almost every iteration) this skips the sort entirely, while
        // the exact peak is still computed whenever a freed slot makes the cheap
        // bound pass. Decision is identical to computing the peak unconditionally.
        if group.kv.active_kv + demand > cap {
            return false;
        }
        group.projected_peak_kv() + demand <= cap
    }
}

// ════════════════════════════════════════════════════════════════════════════
// LoadBalance — pure group ranking (§2.4). Barebone only uses `Single`.
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
pub enum LoadBalance {
    Single,
    RoundRobin { next: u16 },
}

impl LoadBalance {
    /// Returns the chosen group index (ignoring admittability).
    pub fn choose(&mut self, num_groups: usize) -> usize {
        match self {
            Self::Single => 0,
            Self::RoundRobin { next } => {
                let g = (*next as usize) % num_groups;
                *next = ((*next as usize + 1) % num_groups) as u16;
                g
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rid(n: u32) -> RequestId {
        RequestId(n)
    }

    #[test]
    fn kv_pool_grow_and_release() {
        let mut p = KvPool::new(100);
        p.add_kv(30);
        assert_eq!(p.active_kv, 30);
        assert_eq!(p.remaining_now(), 70);
        p.sub_kv(40); // saturating
        assert_eq!(p.active_kv, 0);
    }

    #[test]
    fn batch_finalize_then_advance_then_release() {
        let mut b = Batch::new(0, 1000);
        // A prefill of 10 tokens finalizes into decode with budget 3.
        b.finalize_to_decode(rid(1), 10, 3);
        assert_eq!(b.kv.active_kv, 10);
        assert_eq!(b.iter_decoding().count(), 1);

        // One decode step: current_kv 10→11, remaining 3→2, pool +1.
        b.advance_decodes();
        let (_, s) = &b.decodes[0];
        assert_eq!(s.current_kv, 11);
        assert_eq!(s.remaining_decode, 2);
        assert_eq!(b.kv.active_kv, 11);

        // Release frees its KV and drops it from the decode set.
        b.release(rid(1), 11);
        assert_eq!(b.kv.active_kv, 0);
        assert_eq!(b.decodes.len(), 0);
    }

    #[test]
    fn projected_peak_no_decodes_is_active_kv() {
        let mut p = KvPool::new(1000);
        p.add_kv(42);
        assert_eq!(p.projected_peak(std::iter::empty::<&DecodeReqState>()), 42);
    }

    #[test]
    fn projected_peak_staggered_drain_excludes_departed_growth() {
        // A drains in 1 step, B in 10; both start at kv 0, active_kv 0. Once A
        // leaves, only B keeps growing → peak is B alone reaching 10, NOT 19
        // (the over-estimate from counting A's phantom growth after it exits).
        let p = KvPool {
            kv_capacity: 1000,
            active_kv: 0,
        };
        let decodes = [
            DecodeReqState {
                current_kv: 0,
                remaining_decode: 1,
            },
            DecodeReqState {
                current_kv: 0,
                remaining_decode: 10,
            },
        ];
        assert_eq!(p.projected_peak(decodes.iter()), 10);
    }

    #[test]
    fn projected_peak_simultaneous_growth_peaks_before_first_exit() {
        // A kv10/rem2, B kv20/rem5, active_kv 30. At t=2 both still live:
        // 12 + 22 = 34; after A exits B alone tops out at 25. Peak = 34.
        let p = KvPool {
            kv_capacity: 1000,
            active_kv: 30,
        };
        let decodes = [
            DecodeReqState {
                current_kv: 10,
                remaining_decode: 2,
            },
            DecodeReqState {
                current_kv: 20,
                remaining_decode: 5,
            },
        ];
        assert_eq!(p.projected_peak(decodes.iter()), 34);
    }

    #[test]
    fn strict_admission_respects_capacity() {
        let b = Batch::new(0, 100);
        let adm = KvAdmission::Strict;
        // empty pool: 60 prefill + 30 decode = 90 <= 100 → ok
        assert!(adm.try_admit(&b, 0, 60, 30));
        // with 20 already promised: 90 + 20 = 110 > 100 → reject
        assert!(!adm.try_admit(&b, 20, 60, 30));
    }
}
