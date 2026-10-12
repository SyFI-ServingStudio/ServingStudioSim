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
//! A resident request occupies [`KvAccounting::held_while_running`] units of its
//! context: per-token parts, sliding-window tails and per-request state. Each
//! decode caches its own charge; a step adds the difference the extra token
//! makes.
//!
//! The distinction is load-bearing: [`Self::decode_members`] must keep reporting
//! **context tokens**, because that value is rendered into the arch input's
//! `decode_kv_lens` and drives the attention cost model. Charge is derived where
//! capacity is being decided and never written back into `current_kv`.

use crate::common::{IdMap, RequestId};
use crate::worker::kv::shared::kv_accounting::{KvAccounting, SharedKvAccounting};

#[derive(Clone, Debug)]
struct ResidentDecodeState {
    current_kv: u64,
    remaining_decode: u32,
    /// Prompt length, which places the request's resume positions.
    prompt_tokens: u64,
    /// Units held at `current_kv`.
    charge: u64,
    /// Soft units of checkpoints already passed at `current_kv`.
    checkpoint: u64,
}

impl ResidentDecodeState {
    /// Grow by one token and return the added hard and soft units.
    fn advance(&mut self, accounting: &KvAccounting) -> (u64, u64) {
        self.current_kv += 1;
        self.remaining_decode -= 1;
        let charge = accounting.held_while_running(self.current_kv, 1);
        let added = charge - self.charge;
        self.charge = charge;
        if !accounting.keeps_checkpoints() {
            return (added, 0);
        }
        let checkpoint = accounting.checkpoint_units(self.prompt_tokens, self.current_kv);
        let added_checkpoint = checkpoint - self.checkpoint;
        self.checkpoint = checkpoint;
        (added, added_checkpoint)
    }
}

/// KV-owned state for one partition.
#[derive(Clone, Debug)]
pub(crate) struct ResidentPartitionState {
    accounting: SharedKvAccounting,
    resident_units: u64,
    /// Checkpoints live decodes have passed: soft, evictable units that are
    /// never part of `resident_units`.
    checkpoint_units: u64,
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
    pub(crate) fn new(accounting: SharedKvAccounting) -> Self {
        Self {
            accounting,
            resident_units: 0,
            checkpoint_units: 0,
            decodes: Vec::new(),
            decode_index: IdMap::default(),
            prefill_admits: Vec::new(),
            cached_peak: 0,
        }
    }

    pub(crate) fn capacity_units(&self) -> u64 {
        self.accounting.capacity_units()
    }

    pub(crate) fn resident_units(&self) -> u64 {
        self.resident_units
    }

    /// Soft units of the checkpoints live decodes have passed, before any clamp.
    pub(crate) fn live_checkpoint_units(&self) -> u64 {
        self.checkpoint_units
    }

    /// A live decode's resident KV length by id, O(1).
    pub(crate) fn decode_current_kv(&self, request: RequestId) -> Option<u64> {
        self.decode_index
            .get(&request)
            .map(|&index| self.decodes[index].1.current_kv)
    }

    /// A resident decode's held units by id, O(1).
    pub(crate) fn decode_charge(&self, request: RequestId) -> Option<u64> {
        self.decode_index
            .get(&request)
            .map(|&index| self.decodes[index].1.charge)
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

    pub(crate) fn decode_states(&self) -> impl Iterator<Item = (RequestId, u64, u32)> + '_ {
        self.iter_decoding()
            .map(|(request, state)| (request, state.current_kv, state.remaining_decode))
    }

    /// Units the live decodes' next `max_future_tokens` (or fewer) tokens add,
    /// scaled by `ratio`.
    pub(crate) fn bounded_future_decode_units(&self, max_future_tokens: u32, ratio: f64) -> f64 {
        let future =
            |state: &ResidentDecodeState| u64::from(state.remaining_decode.min(max_future_tokens));
        match self.accounting.linear_units_per_token() {
            Some(units_per_token) => self
                .iter_decoding()
                .map(|(_, state)| (future(state) * units_per_token) as f64 * ratio)
                .sum(),
            None => self
                .iter_decoding()
                .map(|(_, state)| {
                    self.accounting.growth_over(state.current_kv, future(state)) as f64 * ratio
                })
                .sum(),
        }
    }

    /// Units the next decode step may allocate. Each decode advances
    /// `step_tokens` capped by its remaining output; a speculating engine sets
    /// `step_tokens` to its verify width, as vLLM allocates slots for the new
    /// token plus lookahead.
    ///
    /// A `page_size` above one charges whole pages, predicting a crossing from
    /// `current_kv % page_size == 0`, which only holds at one token per step. It
    /// applies to linear layouts only: a layout with windows, state or engine
    /// blocks already rounds through its own accounting.
    pub(crate) fn next_decode_allocation_units(&self, page_size: u32, step_tokens: u32) -> u64 {
        assert!(step_tokens > 0, "a decode step advances at least one token");
        let step = |state: &ResidentDecodeState| u64::from(state.remaining_decode.min(step_tokens));
        let Some(units_per_token) = self.accounting.linear_units_per_token() else {
            return self
                .iter_decoding()
                .map(|(_, state)| self.accounting.growth_over(state.current_kv, step(state)))
                .sum();
        };
        if page_size == 1 {
            return self
                .iter_decoding()
                .map(|(_, state)| step(state) * units_per_token)
                .sum();
        }
        assert_eq!(
            step_tokens, 1,
            "multi-token decode steps need one-token pages: page crossings are predicted from single-token advance"
        );
        let page_size = u64::from(page_size);
        self.iter_decoding()
            .filter(|(_, state)| state.current_kv % page_size == 0)
            .count() as u64
            * page_size
            * units_per_token
    }

    pub(crate) fn first_decode_request(&self) -> Option<RequestId> {
        self.iter_decoding().next().map(|(request, _)| request)
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
                let (added, added_checkpoint) = state.advance(&self.accounting);
                self.resident_units += added;
                self.checkpoint_units += added_checkpoint;
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
                    let (added, added_checkpoint) = state.advance(&self.accounting);
                    self.resident_units += added;
                    self.checkpoint_units += added_checkpoint;
                }
            }
        }
    }

    /// A realized prefill enters resident decode state.
    ///
    /// This is the partition-local implementation of
    /// `KvStore::commit_resident`. `prompt_tokens` places the request's resume
    /// positions, and so its soft checkpoints.
    pub(crate) fn begin_decode(
        &mut self,
        request: RequestId,
        post_prefill_context_tokens: u64,
        decode_budget: u32,
        prompt_tokens: u64,
    ) {
        let charge = self
            .accounting
            .held_while_running(post_prefill_context_tokens, 1);
        let checkpoint = if self.accounting.keeps_checkpoints() {
            self.accounting
                .checkpoint_units(prompt_tokens, post_prefill_context_tokens)
        } else {
            0
        };
        self.resident_units += charge;
        self.checkpoint_units += checkpoint;
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
                prompt_tokens,
                charge,
                checkpoint,
            },
        ));
        self.recompute_peak();
    }

    /// Remove one resident decode while preserving the order of all survivors.
    pub(crate) fn release_decode(&mut self, request: RequestId) {
        if let Some(position) = self.decode_index.remove(&request) {
            let (_, state) = self.decodes.remove(position);
            for (index, (decode_request, _)) in self.decodes.iter().enumerate().skip(position) {
                self.decode_index.insert(*decode_request, index);
            }
            self.resident_units -= state.charge;
            self.checkpoint_units -= state.checkpoint;
            self.recompute_peak();
        }
    }

    fn recompute_peak(&mut self) {
        self.cached_peak = projected_peak(
            &self.accounting,
            self.resident_units,
            self.decodes.iter().map(|(_, state)| state),
        );
    }

    pub(crate) fn projected_peak_units(&self) -> u64 {
        self.cached_peak
    }
}

/// Peak resident units from now until every live decode drains.
///
/// Charge rises between decode-exit steps and drops when a decode finishes,
/// so the peak lands at an exit step. Decodes are sorted by
/// `remaining_decode`, then sampled at all exit steps for fewer than eight
/// members or at the dense-front heuristic probes otherwise. This remains an
/// O(n log n) heuristic guard when every charge is linear; otherwise each probe
/// sums the surviving decodes' growth, O(n) per probe.
///
/// A departing decode releases its whole charge, so the prefix sums below run
/// over charge, not `current_kv`.
fn projected_peak<'a>(
    accounting: &KvAccounting,
    resident_units: u64,
    decodes: impl Iterator<Item = &'a ResidentDecodeState>,
) -> u64 {
    // The linear case sorts (exit step, charge) pairs only: this runs on every
    // admission check, and the sort dominates it.
    if let Some(units_per_token) = accounting.linear_units_per_token() {
        let entries = decodes
            .map(|state| (state.remaining_decode, state.charge))
            .collect();
        return peak_over_exits(
            resident_units,
            entries,
            |&(exit, charge)| (exit, charge),
            |remaining, exit_step| remaining.len() as u64 * u64::from(exit_step) * units_per_token,
        );
    }
    let entries = decodes
        .map(|state| (state.remaining_decode, state.charge, state.current_kv))
        .collect();
    peak_over_exits(
        resident_units,
        entries,
        |&(exit, charge, _)| (exit, charge),
        |remaining, exit_step| {
            remaining
                .iter()
                .map(|&(_, _, current_kv)| accounting.growth_over(current_kv, u64::from(exit_step)))
                .sum()
        },
    )
}

/// The highest resident charge at the sampled decode exits: every decode
/// still running at an exit adds `growth(still_running, exit_step)`, and every
/// decode gone by then releases its charge. `key` gives an entry's exit step
/// and charge.
fn peak_over_exits<E>(
    resident_units: u64,
    mut entries: Vec<E>,
    key: impl Fn(&E) -> (u32, u64),
    growth: impl Fn(&[E], u32) -> u64,
) -> u64 {
    if entries.is_empty() {
        return resident_units;
    }
    entries.sort_by_key(|entry| key(entry).0);

    let num_decodes = entries.len();
    // `charge_prefix[count]` is the resident charge of the `count`
    // soonest-finishing decodes.
    let mut charge_prefix = Vec::with_capacity(num_decodes + 1);
    charge_prefix.push(0u64);
    for entry in &entries {
        charge_prefix.push(charge_prefix.last().unwrap() + key(entry).1);
    }

    let mut peak = resident_units;
    let mut last_exit_step: Option<u32> = None;
    for index in sample_indices(num_decodes) {
        let exit_step = key(&entries[index]).0;
        if last_exit_step == Some(exit_step) {
            continue;
        }
        last_exit_step = Some(exit_step);
        let departed = entries.partition_point(|entry| key(entry).0 < exit_step);
        let added = growth(&entries[departed..], exit_step);
        peak = peak.max(resident_units + added - charge_prefix[departed]);
    }
    peak
}

/// Exit-step indices used by [`projected_peak`].
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::arch::kv_layout::{EngineOverhead, KvCacheLayout, KvCachePart};
    use crate::worker::kv::shared::kv_accounting::{KvAccountingOptions, KvOverhead};

    fn request_id(value: u32) -> RequestId {
        RequestId(value)
    }

    fn tokens(capacity: u64) -> ResidentPartitionState {
        ResidentPartitionState::new(KvAccounting::tokens_for_test(capacity))
    }

    /// One token of KV per unit plus a 100-unit state per request.
    fn with_state(capacity: u64) -> ResidentPartitionState {
        let layout = KvCacheLayout::new()
            .with_part(KvCachePart::per_token("kv", 1))
            .with_part(KvCachePart::per_request("state", 100));
        let accounting =
            KvAccounting::resolve(&layout, KvAccountingOptions::default(), capacity).unwrap();
        ResidentPartitionState::new(Arc::new(accounting))
    }

    fn decode(current_kv: u64, remaining_decode: u32, charge: u64) -> ResidentDecodeState {
        ResidentDecodeState {
            current_kv,
            remaining_decode,
            prompt_tokens: current_kv,
            charge,
            checkpoint: 0,
        }
    }

    #[test]
    fn partition_begins_decode_advances_and_releases() {
        let mut partition = tokens(1000);
        partition.begin_decode(request_id(1), 10, 3, 10);
        assert_eq!(partition.resident_units(), 10);
        assert_eq!(partition.iter_decoding().count(), 1);

        partition.advance_decodes();
        let (_, state) = &partition.decodes[0];
        assert_eq!(state.current_kv, 11);
        assert_eq!(state.remaining_decode, 2);
        assert_eq!(partition.resident_units(), 11);

        partition.release_decode(request_id(1));
        assert_eq!(partition.resident_units(), 0);
        assert!(partition.decodes.is_empty());
    }

    #[test]
    fn advance_subset_advances_only_named_decodes() {
        let mut partition = tokens(1000);
        partition.begin_decode(request_id(1), 10, 3, 10);
        partition.begin_decode(request_id(2), 20, 3, 20);

        partition.advance_subset(&[request_id(1)]);
        let state = |id| {
            &partition
                .decodes
                .iter()
                .find(|(request, _)| *request == request_id(id))
                .unwrap()
                .1
        };
        assert_eq!((state(1).current_kv, state(1).remaining_decode), (11, 2));
        assert_eq!((state(2).current_kv, state(2).remaining_decode), (20, 3));
        assert_eq!(partition.resident_units(), 31);
    }

    #[test]
    fn per_request_state_is_paid_once_and_never_advanced() {
        let mut partition = with_state(1000);
        partition.begin_decode(request_id(1), 10, 3, 10);
        assert_eq!(partition.resident_units(), 110, "context + one state");

        partition.advance_decodes();
        assert_eq!(
            partition.resident_units(),
            111,
            "a decode step grows context only",
        );
        assert_eq!(
            partition.decode_members().collect::<Vec<_>>(),
            [(request_id(1), 11)],
            "membership reports context tokens, never charge",
        );

        partition.release_decode(request_id(1));
        assert_eq!(partition.resident_units(), 0, "release returns both halves");
    }

    #[test]
    fn projected_peak_releases_the_state_with_the_departing_decode() {
        let accounting = KvAccounting::tokens_for_test(1000);
        // Two residents at 100 each; the first leaves after one step, so the
        // peak is 200 + 1 step of growth, not 200 + 10.
        let decodes = [decode(0, 1, 100), decode(0, 10, 100)];
        assert_eq!(projected_peak(&accounting, 200, decodes.iter()), 202);
    }

    #[test]
    fn projected_peak_without_decodes_is_resident_kv() {
        let accounting = KvAccounting::tokens_for_test(1000);
        assert_eq!(projected_peak(&accounting, 42, std::iter::empty()), 42);
    }

    #[test]
    fn projected_peak_excludes_departed_decode_growth() {
        let accounting = KvAccounting::tokens_for_test(1000);
        let decodes = [decode(0, 1, 0), decode(0, 10, 0)];
        assert_eq!(projected_peak(&accounting, 0, decodes.iter()), 10);
    }

    #[test]
    fn projected_peak_counts_simultaneous_growth_before_first_exit() {
        let accounting = KvAccounting::tokens_for_test(1000);
        let decodes = [decode(10, 2, 10), decode(20, 5, 20)];
        assert_eq!(projected_peak(&accounting, 30, decodes.iter()), 34);
    }

    /// A 4-token window: a decode past its window adds nothing to it, so the
    /// projected peak grows by the per-token part alone.
    #[test]
    fn projected_peak_follows_non_linear_growth() {
        let layout = KvCacheLayout::new()
            .with_part(KvCachePart::per_token("global", 1))
            .with_part(KvCachePart::sliding_window("swa", 4, 1));
        let accounting = KvAccounting::resolve(
            &layout,
            KvAccountingOptions {
                overhead: KvOverhead::None,
                ..KvAccountingOptions::default()
            },
            1 << 20,
        )
        .unwrap();
        assert_eq!(accounting.held_while_running(1, 1), 2);
        // Request 0 sits at 1 token and grows its window for 3 more tokens;
        // request 1 at 100 tokens grows only its global part.
        let decodes = [decode(1, 10, 2), decode(100, 10, 104)];
        assert_eq!(
            projected_peak(&accounting, 106, decodes.iter()),
            106 + (10 + 3) + 10
        );
    }

    #[test]
    fn live_checkpoints_track_the_positions_a_decode_has_passed() {
        // A GDN-like state of 50 units on a 64-token engine block, resumable
        // at the response end only, so every block boundary a decode crosses
        // leaves one state behind.
        let layout = KvCacheLayout::new()
            .with_part(KvCachePart::per_token("gqa", 1))
            .with_part(KvCachePart::per_request("gdn", 50))
            .with_engine(EngineOverhead {
                block_tokens: Some(64),
                fixed_device_bytes: 0,
            });
        let accounting = KvAccounting::resolve(
            &layout,
            KvAccountingOptions {
                resume_points: Some(
                    crate::worker::kv::shared::kv_accounting::PrefixResumePoints::EveryBlock,
                ),
                ..KvAccountingOptions::default()
            },
            1 << 20,
        )
        .unwrap();
        let mut partition = ResidentPartitionState::new(Arc::new(accounting));
        partition.begin_decode(request_id(1), 128, 10, 100);
        partition.begin_decode(request_id(2), 63, 10, 63);
        // Request 1 has passed 64 (128 is its current length, not passed).
        assert_eq!(partition.live_checkpoint_units(), 50);
        partition.advance_decodes();
        // 129 has passed 128. Request 2 now sits at 64: that is its running
        // state, not a checkpoint behind it.
        assert_eq!(partition.live_checkpoint_units(), 50 * 2);
        partition.advance_decodes();
        assert_eq!(partition.live_checkpoint_units(), 50 * 3);
        partition.release_decode(request_id(1));
        assert_eq!(partition.live_checkpoint_units(), 50);
    }
}
