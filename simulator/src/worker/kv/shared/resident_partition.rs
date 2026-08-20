//! One KV partition's resident set.
//!
//! The store owns partition placement and the cross-partition ledgers; this
//! module owns the state of one partition: resident charge accounting,
//! insertion-ordered decode membership, this iteration's prefill admits, and the
//! projected-peak cache. The partition id is intentionally not duplicated here:
//! the owning store's `partitions` index is the single placement identity.
//!
//! ## Charge vs. context tokens
//!
//! A resident request occupies `current_kv + fixed_charge_per_request`, where the
//! second term is a hybrid model's per-request recurrent state — fixed at
//! admission and never grown by [`Self::advance_decodes`]. A full-attention store
//! passes `0` and the two quantities coincide.
//!
//! The distinction is load-bearing: [`Self::decode_members`] must keep reporting
//! **context tokens**, because that value is rendered into the arch input's
//! `decode_kv_lens` and drives the attention cost model. Charge is derived where
//! capacity is being decided and never written back into `current_kv`.

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

    /// Peak cumulative charge from now until every live decode drains.
    ///
    /// Charge(t) rises linearly between decode-exit steps and drops when a decode
    /// finishes. The peak therefore lands at an exit step. Decodes are sorted by
    /// `remaining_decode`, then sampled at all exit steps for fewer than eight
    /// members or at the dense-front heuristic probes used by the original
    /// implementation. This remains an O(n log n) heuristic guard.
    ///
    /// A departing decode releases its context KV **and** its fixed charge, so
    /// the prefix sums below run over `charge`, not `current_kv`. Growth stays
    /// one token per step either way: the fixed part does not grow.
    fn projected_peak<'a>(
        &self,
        decodes: impl Iterator<Item = &'a ResidentDecodeState>,
        fixed_charge_per_request: u64,
    ) -> u64 {
        let mut entries: Vec<(u32, u64)> = decodes
            .map(|state| {
                (
                    state.remaining_decode,
                    state.current_kv + fixed_charge_per_request,
                )
            })
            .collect();
        if entries.is_empty() {
            return self.resident_tokens;
        }
        entries.sort_by_key(|(remaining_decode, _)| *remaining_decode);

        let num_decodes = entries.len();
        // `kv_prefix[count]` is the resident charge of the `count`
        // soonest-finishing decodes.
        let mut kv_prefix = Vec::with_capacity(num_decodes + 1);
        kv_prefix.push(0u64);
        for (_, charge) in &entries {
            kv_prefix.push(kv_prefix.last().unwrap() + charge);
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

/// KV-owned state for one partition.
#[derive(Clone, Debug)]
pub(crate) struct ResidentPartitionState {
    capacity: KvCapacityState,
    /// Per-resident-request charge on top of its context KV — a hybrid model's
    /// recurrent state. `0` for a pure full-attention store.
    fixed_charge_per_request: u64,
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

impl ResidentPartitionState {
    pub(crate) fn new(kv_capacity: u64, fixed_charge_per_request: u64) -> Self {
        Self {
            capacity: KvCapacityState::new(kv_capacity),
            fixed_charge_per_request,
            decodes: Vec::new(),
            decode_index: IdMap::default(),
            prefill_admits: Vec::new(),
            cached_peak: 0,
        }
    }

    pub(crate) fn capacity_tokens(&self) -> u64 {
        self.capacity.token_capacity
    }

    pub(crate) fn resident_tokens(&self) -> u64 {
        self.capacity.resident_tokens
    }

    /// A live decode's resident KV length by id, O(1).
    pub(crate) fn decode_current_kv(&self, request: RequestId) -> Option<u64> {
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

    pub(crate) fn decode_members(&self) -> impl Iterator<Item = (RequestId, u64)> + '_ {
        self.iter_decoding()
            .map(|(request, state)| (request, state.current_kv))
    }

    pub(crate) fn has_live_decode(&self) -> bool {
        self.iter_decoding().next().is_some()
    }

    pub(crate) fn live_decode_count(&self) -> u32 {
        self.iter_decoding().count() as u32
    }

    pub(crate) fn add_prefill_admit(&mut self, request: RequestId) {
        self.prefill_admits.push(request);
    }

    pub(crate) fn clear_prefill_admits(&mut self) {
        self.prefill_admits.clear();
    }

    pub(crate) fn remove_prefill_admit(&mut self, request: RequestId) {
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

    pub(crate) fn has_prefill_admit(&self) -> bool {
        !self.prefill_admits.is_empty()
    }

    pub(crate) fn prefill_admit_count(&self) -> u32 {
        self.prefill_admits.len() as u32
    }

    pub(crate) fn iter_prefill_admits(&self) -> impl Iterator<Item = RequestId> + '_ {
        self.prefill_admits.iter().copied()
    }

    /// Advance every resident decode by one token.
    pub(crate) fn advance_decodes(&mut self) {
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
    pub(crate) fn advance_subset(&mut self, requests: &[RequestId]) {
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
    /// `KvStore::commit_resident`. The fixed charge lands here, once, and is
    /// released by [`Self::release_decode`] — it is never advanced.
    pub(crate) fn begin_decode(
        &mut self,
        request: RequestId,
        post_prefill_context_tokens: u64,
        decode_budget: u32,
    ) {
        self.capacity
            .add_resident_tokens(post_prefill_context_tokens + self.fixed_charge_per_request);
        debug_assert!(
            !self.decode_index.contains_key(&request),
            "begin_decode: {request:?} already resident — decode ids are unique"
        );
        self.decode_index.insert(request, self.decodes.len());
        self.decodes.push((
            request,
            ResidentDecodeState {
                current_kv: post_prefill_context_tokens,
                remaining_decode: decode_budget,
            },
        ));
        self.recompute_peak();
    }

    /// Remove one resident decode while preserving the order of all survivors.
    pub(crate) fn release_decode(&mut self, request: RequestId, current_kv: u64) {
        if let Some(position) = self.decode_index.remove(&request) {
            self.decodes.remove(position);
            for (index, (decode_request, _)) in self.decodes.iter().enumerate().skip(position) {
                self.decode_index.insert(*decode_request, index);
            }
            self.capacity
                .remove_resident_tokens(current_kv + self.fixed_charge_per_request);
            self.recompute_peak();
        }
    }

    fn recompute_peak(&mut self) {
        self.cached_peak = self.capacity.projected_peak(
            self.decodes.iter().map(|(_, state)| state),
            self.fixed_charge_per_request,
        );
    }

    pub(crate) fn projected_peak_kv(&self) -> u64 {
        self.cached_peak
    }

    /// How many checkpoint boundaries every live decode has crossed, in units of
    /// `interval` context tokens. This is the soft, evictable half of a hybrid
    /// store's occupancy — see `hybrid_gdn`. Full-attention stores never call it.
    pub(crate) fn live_checkpoint_count(&self, interval: u32) -> u64 {
        if interval == 0 {
            return 0;
        }
        self.iter_decoding()
            .map(|(_, state)| state.current_kv / u64::from(interval))
            .sum()
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
        let mut partition = ResidentPartitionState::new(1000, 0);
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
        let mut partition = ResidentPartitionState::new(1000, 0);
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
    fn fixed_charge_is_paid_once_and_never_advanced() {
        let mut partition = ResidentPartitionState::new(1000, 100);
        partition.begin_decode(request_id(1), 10, 3);
        assert_eq!(
            partition.resident_tokens(),
            110,
            "context + one fixed charge"
        );

        partition.advance_decodes();
        assert_eq!(
            partition.resident_tokens(),
            111,
            "a decode step grows context only",
        );
        assert_eq!(
            partition.decode_members().collect::<Vec<_>>(),
            [(request_id(1), 11)],
            "membership reports context tokens, never charge",
        );

        partition.release_decode(request_id(1), 11);
        assert_eq!(
            partition.resident_tokens(),
            0,
            "release returns both halves"
        );
    }

    #[test]
    fn projected_peak_releases_the_fixed_charge_with_the_departing_decode() {
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
        // Two residents at 100 each; the first leaves after one step, so the
        // peak is 200 + 1 step of growth, not 200 + 10.
        let capacity = KvCapacityState {
            token_capacity: 1000,
            resident_tokens: 200,
        };
        assert_eq!(capacity.projected_peak(decodes.iter(), 100), 202);
    }

    #[test]
    fn live_checkpoint_count_sums_crossed_boundaries_and_is_off_at_zero() {
        let mut partition = ResidentPartitionState::new(100_000, 0);
        partition.begin_decode(request_id(1), 4_100, 10);
        partition.begin_decode(request_id(2), 2_047, 10);
        assert_eq!(partition.live_checkpoint_count(2_048), 2 + 0);
        partition.advance_decodes();
        assert_eq!(partition.live_checkpoint_count(2_048), 2 + 1);
        assert_eq!(partition.live_checkpoint_count(0), 0);
    }

    #[test]
    fn projected_peak_without_decodes_is_resident_kv() {
        let mut capacity = KvCapacityState::new(1000);
        capacity.add_resident_tokens(42);
        assert_eq!(
            capacity.projected_peak(std::iter::empty::<&ResidentDecodeState>(), 0),
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
        assert_eq!(capacity.projected_peak(decodes.iter(), 0), 10);
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
        assert_eq!(capacity.projected_peak(decodes.iter(), 0), 34);
    }
}
