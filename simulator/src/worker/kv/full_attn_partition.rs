//! Per-partition resident state for full-attention KV.
//!
//! `FullAttnKv` owns partition placement and cross-partition ledgers; this module
//! owns the state of one partition: resident token accounting, insertion-ordered
//! decode membership, this iteration's prefill admits, and the projected-peak
//! cache. The partition id is intentionally not duplicated here: the owning
//! `FullAttnKv::partitions` index is the single placement identity.

use crate::common::{IdMap, RequestId};

#[derive(Clone, Debug)]
struct KvCapacityState {
    token_capacity: u64,
    resident_tokens: u64,
}

impl KvCapacityState {
    fn new(token_capacity: u64) -> Self {
        Self {
            token_capacity,
            resident_tokens: 0,
        }
    }

    #[cfg(test)]
    fn remaining_tokens(&self) -> u64 {
        self.token_capacity.saturating_sub(self.resident_tokens)
    }

    fn add_resident_tokens(&mut self, tokens: u64) {
        self.resident_tokens += tokens;
    }

    fn remove_resident_tokens(&mut self, tokens: u64) {
        self.resident_tokens = self.resident_tokens.saturating_sub(tokens);
    }

    /// Peak cumulative KV from now until every live decode drains.
    ///
    /// KV(t) rises linearly between decode-exit steps and drops when a decode
    /// finishes. The peak therefore lands at an exit step. Decodes are sorted by
    /// `remaining_decode`, then sampled at all exit steps for fewer than eight
    /// members or at the dense-front heuristic probes used by the original
    /// implementation. This remains an O(n log n) heuristic guard.
    fn projected_peak<'a>(&self, decodes: impl Iterator<Item = &'a ResidentDecodeState>) -> u64 {
        let mut entries: Vec<(u32, u64)> = decodes
            .map(|state| (state.remaining_decode, state.current_kv))
            .collect();
        if entries.is_empty() {
            return self.resident_tokens;
        }
        entries.sort_by_key(|(remaining_decode, _)| *remaining_decode);

        let num_decodes = entries.len();
        // `kv_prefix[count]` is the resident KV of the `count`
        // soonest-finishing decodes.
        let mut kv_prefix = Vec::with_capacity(num_decodes + 1);
        kv_prefix.push(0u64);
        for (_, current_kv) in &entries {
            kv_prefix.push(kv_prefix.last().unwrap() + current_kv);
        }

        let mut peak = self.resident_tokens;
        let mut last_exit_step: Option<u32> = None;
        for index in Self::sample_indices(num_decodes) {
            let exit_step = entries[index].0;
            if last_exit_step == Some(exit_step) {
                continue;
            }
            last_exit_step = Some(exit_step);
            let departed =
                entries.partition_point(|(remaining_decode, _)| *remaining_decode < exit_step);
            let kv_at_exit = self.resident_tokens
                + (num_decodes - departed) as u64 * exit_step as u64
                - kv_prefix[departed];
            peak = peak.max(kv_at_exit);
        }
        peak
    }

    /// Exit-step indices used by [`Self::projected_peak`].
    fn sample_indices(num_decodes: usize) -> Vec<usize> {
        if num_decodes == 0 {
            return Vec::new();
        }
        if num_decodes < 8 {
            return (0..num_decodes).collect();
        }
        let mut indices: Vec<usize> = (0..8).collect();
        for sample in 1..8 {
            indices.push(sample * num_decodes / 8);
        }
        indices.push(num_decodes - 1);
        indices.sort_unstable();
        indices.dedup();
        indices
    }
}

#[derive(Clone, Debug)]
struct ResidentDecodeState {
    current_kv: u64,
    remaining_decode: u32,
}

/// KV-owned state for one full-attention partition.
#[derive(Clone, Debug)]
pub(super) struct FullAttnPartitionState {
    capacity: KvCapacityState,
    /// Insertion order is observable through model input and completion events.
    decodes: Vec<(RequestId, ResidentDecodeState)>,
    /// O(1) by-id lookup mirroring `decodes`; `decodes` remains the source of
    /// truth for ordering.
    decode_index: IdMap<RequestId, usize>,
    /// Requests realized as prefill for the current iteration.
    prefill_admits: Vec<RequestId>,
    /// Cached projected peak. Decode entry may raise it; decode exit may lower it.
    cached_peak: u64,
}

impl FullAttnPartitionState {
    pub(super) fn new(kv_capacity: u64) -> Self {
        Self {
            capacity: KvCapacityState::new(kv_capacity),
            decodes: Vec::new(),
            decode_index: IdMap::default(),
            prefill_admits: Vec::new(),
            cached_peak: 0,
        }
    }

    pub(super) fn capacity_tokens(&self) -> u64 {
        self.capacity.token_capacity
    }

    pub(super) fn resident_tokens(&self) -> u64 {
        self.capacity.resident_tokens
    }

    /// A live decode's resident KV length by id, O(1).
    pub(super) fn decode_current_kv(&self, request: RequestId) -> Option<u64> {
        self.decode_index
            .get(&request)
            .map(|&index| self.decodes[index].1.current_kv)
    }

    /// Live decodes in insertion order.
    fn iter_decoding(&self) -> impl Iterator<Item = (RequestId, &ResidentDecodeState)> {
        self.decodes
            .iter()
            .filter(|(_, state)| state.remaining_decode > 0)
            .map(|(request, state)| (*request, state))
    }

    pub(super) fn decode_members(&self) -> impl Iterator<Item = (RequestId, u64)> + '_ {
        self.iter_decoding()
            .map(|(request, state)| (request, state.current_kv))
    }

    pub(super) fn has_live_decode(&self) -> bool {
        self.iter_decoding().next().is_some()
    }

    pub(super) fn live_decode_count(&self) -> u32 {
        self.iter_decoding().count() as u32
    }

    pub(super) fn add_prefill_admit(&mut self, request: RequestId) {
        self.prefill_admits.push(request);
    }

    pub(super) fn clear_prefill_admits(&mut self) {
        self.prefill_admits.clear();
    }

    pub(super) fn remove_prefill_admit(&mut self, request: RequestId) {
        if let Some(position) = self
            .prefill_admits
            .iter()
            .position(|&admit| admit == request)
        {
            // Preserve the original cancellation behavior: pending iteration
            // order is irrelevant after the removed request is cancelled.
            self.prefill_admits.swap_remove(position);
        }
    }

    pub(super) fn has_prefill_admit(&self) -> bool {
        !self.prefill_admits.is_empty()
    }

    pub(super) fn prefill_admit_count(&self) -> u32 {
        self.prefill_admits.len() as u32
    }

    pub(super) fn iter_prefill_admits(&self) -> impl Iterator<Item = RequestId> + '_ {
        self.prefill_admits.iter().copied()
    }

    /// Advance every resident decode by one token.
    pub(super) fn advance_decodes(&mut self) {
        for (_, state) in &mut self.decodes {
            if state.remaining_decode > 0 {
                state.current_kv += 1;
                state.remaining_decode -= 1;
                self.capacity.add_resident_tokens(1);
            }
        }
    }

    /// Advance only a named slot micro-batch.
    ///
    /// The by-id index preserves the original insertion order while avoiding
    /// the old O(decodes × requests) scan.
    pub(super) fn advance_subset(&mut self, requests: &[RequestId]) {
        for &request in requests {
            if let Some(&position) = self.decode_index.get(&request) {
                let state = &mut self.decodes[position].1;
                if state.remaining_decode > 0 {
                    state.current_kv += 1;
                    state.remaining_decode -= 1;
                    self.capacity.add_resident_tokens(1);
                }
            }
        }
    }

    /// A realized prefill enters resident decode state.
    ///
    /// This is the partition-local implementation of
    /// `KvStore::commit_resident`.
    pub(super) fn begin_decode(&mut self, request: RequestId, initial_kv: u64, decode_budget: u32) {
        self.capacity.add_resident_tokens(initial_kv);
        debug_assert!(
            !self.decode_index.contains_key(&request),
            "begin_decode: {request:?} already resident — decode ids are unique"
        );
        self.decode_index.insert(request, self.decodes.len());
        self.decodes.push((
            request,
            ResidentDecodeState {
                current_kv: initial_kv,
                remaining_decode: decode_budget,
            },
        ));
        self.recompute_peak();
    }

    /// Remove one resident decode while preserving the order of all survivors.
    pub(super) fn release_decode(&mut self, request: RequestId, current_kv: u64) {
        if let Some(position) = self.decode_index.remove(&request) {
            self.decodes.remove(position);
            for (index, (decode_request, _)) in self.decodes.iter().enumerate().skip(position) {
                self.decode_index.insert(*decode_request, index);
            }
            self.capacity.remove_resident_tokens(current_kv);
            self.recompute_peak();
        }
    }

    fn recompute_peak(&mut self) {
        self.cached_peak = self
            .capacity
            .projected_peak(self.decodes.iter().map(|(_, state)| state));
    }

    pub(super) fn projected_peak_kv(&self) -> u64 {
        self.cached_peak
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_id(value: u32) -> RequestId {
        RequestId(value)
    }

    #[test]
    fn capacity_grows_and_releases_saturating() {
        let mut capacity = KvCapacityState::new(100);
        capacity.add_resident_tokens(30);
        assert_eq!(capacity.resident_tokens, 30);
        assert_eq!(capacity.remaining_tokens(), 70);
        capacity.remove_resident_tokens(40);
        assert_eq!(capacity.resident_tokens, 0);
    }

    #[test]
    fn partition_begins_decode_advances_and_releases() {
        let mut partition = FullAttnPartitionState::new(1000);
        partition.begin_decode(request_id(1), 10, 3);
        assert_eq!(partition.resident_tokens(), 10);
        assert_eq!(partition.iter_decoding().count(), 1);

        partition.advance_decodes();
        let (_, state) = &partition.decodes[0];
        assert_eq!(state.current_kv, 11);
        assert_eq!(state.remaining_decode, 2);
        assert_eq!(partition.resident_tokens(), 11);

        partition.release_decode(request_id(1), 11);
        assert_eq!(partition.resident_tokens(), 0);
        assert!(partition.decodes.is_empty());
    }

    #[test]
    fn advance_subset_advances_only_named_decodes() {
        let mut partition = FullAttnPartitionState::new(1000);
        partition.begin_decode(request_id(1), 10, 3);
        partition.begin_decode(request_id(2), 20, 3);

        partition.advance_subset(&[request_id(1)]);
        let first = &partition
            .decodes
            .iter()
            .find(|(request, _)| *request == request_id(1))
            .unwrap()
            .1;
        let second = &partition
            .decodes
            .iter()
            .find(|(request, _)| *request == request_id(2))
            .unwrap()
            .1;
        assert_eq!((first.current_kv, first.remaining_decode), (11, 2));
        assert_eq!((second.current_kv, second.remaining_decode), (20, 3));
        assert_eq!(partition.resident_tokens(), 31);
    }

    #[test]
    fn projected_peak_without_decodes_is_resident_kv() {
        let mut capacity = KvCapacityState::new(1000);
        capacity.add_resident_tokens(42);
        assert_eq!(
            capacity.projected_peak(std::iter::empty::<&ResidentDecodeState>()),
            42
        );
    }

    #[test]
    fn projected_peak_excludes_departed_decode_growth() {
        let capacity = KvCapacityState {
            token_capacity: 1000,
            resident_tokens: 0,
        };
        let decodes = [
            ResidentDecodeState {
                current_kv: 0,
                remaining_decode: 1,
            },
            ResidentDecodeState {
                current_kv: 0,
                remaining_decode: 10,
            },
        ];
        assert_eq!(capacity.projected_peak(decodes.iter()), 10);
    }

    #[test]
    fn projected_peak_counts_simultaneous_growth_before_first_exit() {
        let capacity = KvCapacityState {
            token_capacity: 1000,
            resident_tokens: 30,
        };
        let decodes = [
            ResidentDecodeState {
                current_kv: 10,
                remaining_decode: 2,
            },
            ResidentDecodeState {
                current_kv: 20,
                remaining_decode: 5,
            },
        ];
        assert_eq!(capacity.projected_peak(decodes.iter()), 34);
    }
}
