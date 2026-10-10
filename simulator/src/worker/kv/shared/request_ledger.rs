//! The KV owner's request-keyed side tables.
//!
//! Everything a store knows about a request that is *not* resident partition
//! state lives here, in one place, because these four maps share a lifecycle:
//! a request is placed, promises a footprint, may hold prefilled KV awaiting a
//! pull, carries a resolved prefill context, and drops all of it on release.
//! Splitting them across the store is how the "second admission ledger" the KV
//! owner is supposed to be the sole authority for gets introduced by accident.
//!
//! Three invariants the store relies on and this module preserves:
//!
//! - A request is in `promised` **or** resident, never both — `drain_ready` and
//!   `commit_resident` are the two exits, and both remove the promise.
//! - `held` is a separate ledger from `promised` with its own per-partition
//!   running total, because held KV is physically resident while its owner has
//!   left the local decode set.
//! - `reserved_by_partition` is the per-partition charge and count over
//!   `promised` and `chunked_prefill` together, updated by every insert and
//!   remove of either map, so the capacity checks that read it on every
//!   admission are O(1) rather than a scan of both maps.
//! - [`Self::drain_promised`] reports promises in the order they were made: it
//!   feeds prefill-admit order, which feeds model input and event order. A
//!   `HashMap`'s iteration order is not that order, and std seeds it afresh in
//!   every native process, so it must not leak into the drain.

use std::collections::HashMap;

use crate::common::RequestId;
use crate::worker::kv::ResolvedPrefillContext;
use crate::worker::shared::advance_scope::PartitionId;

pub(crate) struct RequestLedger {
    /// Reserved-but-not-yet-resident footprints, as `(partition, charge)`,
    /// with the sequence number each was promised at.
    promised: HashMap<RequestId, (PartitionId, u64, u64)>,
    /// The sequence number the next promise gets.
    next_promise: u64,
    /// Full-footprint reservations held across partial-prefill iterations.
    chunked_prefill: HashMap<RequestId, (PartitionId, u64)>,
    /// Charge and count of `promised` plus `chunked_prefill`, per partition.
    reserved_by_partition: Vec<(u64, u32)>,
    /// Prefilled KV awaiting a decode-side pull acknowledgement.
    held: HashMap<RequestId, (PartitionId, u64)>,
    held_by_partition: Vec<u64>,
    /// Sticky placement. Outlives `promised`; cleared on release.
    placement: HashMap<RequestId, PartitionId>,
    /// Runtime prefill facts stay beside placement; the shared request record
    /// carries only the immutable declaration.
    resolved_prefill_contexts: HashMap<RequestId, ResolvedPrefillContext>,
}

impl RequestLedger {
    pub(crate) fn new(num_partitions: usize) -> Self {
        Self {
            promised: HashMap::new(),
            next_promise: 0,
            chunked_prefill: HashMap::new(),
            reserved_by_partition: vec![(0, 0); num_partitions],
            held: HashMap::new(),
            held_by_partition: vec![0; num_partitions],
            placement: HashMap::new(),
            resolved_prefill_contexts: HashMap::new(),
        }
    }

    // ── placement ────────────────────────────────────────────────────────────

    pub(crate) fn placement(&self, request: RequestId) -> Option<PartitionId> {
        self.placement.get(&request).copied()
    }

    pub(crate) fn forget_placement(&mut self, request: RequestId) -> Option<PartitionId> {
        self.placement.remove(&request)
    }

    pub(crate) fn all_placed_on(&self, partition: PartitionId, requests: &[RequestId]) -> bool {
        requests
            .iter()
            .all(|request| self.placement.get(request).copied() == Some(partition))
    }

    // ── promised ─────────────────────────────────────────────────────────────

    pub(crate) fn promise(&mut self, request: RequestId, partition: PartitionId, charge: u64) {
        if let Some((previous_partition, previous_charge, _)) = self
            .promised
            .insert(request, (partition, charge, self.next_promise))
        {
            self.remove_reserved(previous_partition, previous_charge);
        }
        self.add_reserved(partition, charge);
        self.next_promise += 1;
        self.placement.insert(request, partition);
    }

    pub(crate) fn forget_promise(&mut self, request: RequestId) {
        if let Some((partition, charge, _)) = self.promised.remove(&request) {
            self.remove_reserved(partition, charge);
        }
    }

    pub(crate) fn has_promise(&self, request: RequestId) -> bool {
        self.promised.contains_key(&request) || self.chunked_prefill.contains_key(&request)
    }

    pub(crate) fn promised_charge(&self, request: RequestId) -> Option<u64> {
        self.promised
            .get(&request)
            .map(|&(_, charge, _)| charge)
            .or_else(|| {
                self.chunked_prefill
                    .get(&request)
                    .map(|&(_, charge)| charge)
            })
    }

    #[inline]
    pub(crate) fn partition_promised(&self, partition: PartitionId) -> u64 {
        self.reserved_by_partition[partition as usize].0
    }

    #[inline]
    pub(crate) fn partition_promised_count(&self, partition: PartitionId) -> u32 {
        self.reserved_by_partition[partition as usize].1
    }

    fn add_reserved(&mut self, partition: PartitionId, charge: u64) {
        let (total, count) = &mut self.reserved_by_partition[partition as usize];
        *total += charge;
        *count = count
            .checked_add(1)
            .expect("partition reservation count exceeds u32");
    }

    fn remove_reserved(&mut self, partition: PartitionId, charge: u64) {
        let (total, count) = &mut self.reserved_by_partition[partition as usize];
        *total -= charge;
        *count -= 1;
    }

    pub(crate) fn promote_promise_to_chunked_prefill(&mut self, request: RequestId) {
        let (partition, charge, _) = self
            .promised
            .remove(&request)
            .expect("chunked prefill must promote an existing promise");
        // The reservation only changes maps, so the partition total keeps it.
        if let Some((previous_partition, previous_charge)) =
            self.chunked_prefill.insert(request, (partition, charge))
        {
            self.remove_reserved(previous_partition, previous_charge);
        }
    }

    pub(crate) fn forget_chunked_prefill(&mut self, request: RequestId) {
        if let Some((partition, charge)) = self.chunked_prefill.remove(&request) {
            self.remove_reserved(partition, charge);
        }
    }

    pub(crate) fn has_chunked_prefill(&self, request: RequestId) -> bool {
        self.chunked_prefill.contains_key(&request)
    }

    /// Empty `promised` and report `(partition, request)` in the order the
    /// promises were made.
    pub(crate) fn drain_promised(&mut self) -> Vec<(PartitionId, RequestId)> {
        let mut drained: Vec<(u64, PartitionId, RequestId, u64)> = self
            .promised
            .drain()
            .map(|(request, (partition, charge, seq))| (seq, partition, request, charge))
            .collect();
        for &(_, partition, _, charge) in &drained {
            self.remove_reserved(partition, charge);
        }
        drained.sort_unstable_by_key(|&(seq, _, _, _)| seq);
        drained
            .into_iter()
            .map(|(_, partition, request, _)| (partition, request))
            .collect()
    }

    // ── held ─────────────────────────────────────────────────────────────────

    /// Move a prefilled request's KV into the held ledger. Held KV is no longer
    /// placed: its owner has handed the request off.
    pub(crate) fn hold(&mut self, request: RequestId, partition: PartitionId, kv_tokens: u64) {
        self.placement.remove(&request);
        if let Some((previous_partition, previous_tokens)) =
            self.held.insert(request, (partition, kv_tokens))
        {
            let total = &mut self.held_by_partition[previous_partition as usize];
            *total = total.saturating_sub(previous_tokens);
        }
        self.held_by_partition[partition as usize] += kv_tokens;
    }

    /// Drop a held reservation and report what it was, if anything.
    pub(crate) fn take_held(&mut self, request: RequestId) -> Option<(PartitionId, u64)> {
        let (partition, kv_tokens) = self.held.remove(&request)?;
        let total = &mut self.held_by_partition[partition as usize];
        *total = total.saturating_sub(kv_tokens);
        Some((partition, kv_tokens))
    }

    #[inline]
    pub(crate) fn partition_held(&self, partition: PartitionId) -> u64 {
        self.held_by_partition[partition as usize]
    }

    // ── resolved prefill context ─────────────────────────────────────────────

    pub(crate) fn set_prefill_context(
        &mut self,
        request: RequestId,
        resolved_prefill: ResolvedPrefillContext,
    ) {
        self.resolved_prefill_contexts
            .insert(request, resolved_prefill);
    }

    pub(crate) fn schedule_prefill_chunk(&mut self, request: RequestId, chunk_tokens: u32) {
        self.resolved_prefill_contexts
            .get_mut(&request)
            .expect("chunked prefill context must exist")
            .schedule_chunk(chunk_tokens);
    }

    pub(crate) fn complete_prefill_chunk(&mut self, request: RequestId) {
        self.resolved_prefill_contexts
            .get_mut(&request)
            .expect("chunked prefill context must exist")
            .complete_chunk();
    }

    pub(crate) fn prefill_context(&self, request: RequestId) -> Option<ResolvedPrefillContext> {
        self.resolved_prefill_contexts.get(&request).copied()
    }

    pub(crate) fn take_prefill_context(
        &mut self,
        request: RequestId,
    ) -> Option<ResolvedPrefillContext> {
        self.resolved_prefill_contexts.remove(&request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promises_drain_in_the_order_they_were_made() {
        // Not id order, and enough of them that a hash map's order would differ
        // from one process to the next.
        let order: Vec<u32> = (0..64).map(|i| (i * 37) % 64).collect();
        let mut ledger = RequestLedger::new(1);
        for &id in &order {
            ledger.promise(RequestId(id), 0, 1);
        }
        ledger.forget_promise(RequestId(order[5]));
        let drained: Vec<u32> = ledger
            .drain_promised()
            .into_iter()
            .map(|(_, r)| r.0)
            .collect();
        let expected: Vec<u32> = order.iter().copied().filter(|&id| id != order[5]).collect();
        assert_eq!(drained, expected);
    }

    #[test]
    fn promise_places_and_drain_clears_both_the_promise_and_nothing_else() {
        let mut ledger = RequestLedger::new(2);
        ledger.promise(RequestId(0), 1, 30);
        ledger.promise(RequestId(1), 1, 40);
        ledger.promise(RequestId(2), 0, 50);
        assert_eq!(ledger.partition_promised(1), 70);
        assert_eq!(ledger.partition_promised_count(1), 2);
        assert_eq!(ledger.partition_promised(0), 50);

        assert_eq!(
            ledger.drain_promised(),
            [(1, RequestId(0)), (1, RequestId(1)), (0, RequestId(2))]
        );
        assert_eq!(ledger.partition_promised(1), 0);
        assert_eq!(
            ledger.placement(RequestId(0)),
            Some(1),
            "draining a promise must not forget where the request went",
        );
    }

    #[test]
    fn reserved_totals_follow_a_promise_through_chunked_prefill_and_re_promises() {
        let mut ledger = RequestLedger::new(2);
        ledger.promise(RequestId(0), 0, 30);
        ledger.promise(RequestId(1), 0, 40);
        ledger.promote_promise_to_chunked_prefill(RequestId(0));
        assert_eq!(ledger.partition_promised(0), 70);
        assert_eq!(ledger.partition_promised_count(0), 2);

        // A chunked request may promise again; the re-promise replaces only
        // its earlier promise, and both maps count until each is forgotten.
        ledger.promise(RequestId(0), 1, 5);
        ledger.promise(RequestId(0), 1, 8);
        assert_eq!(
            (ledger.partition_promised(0), ledger.partition_promised(1)),
            (70, 8)
        );
        ledger.forget_chunked_prefill(RequestId(0));
        ledger.forget_promise(RequestId(1));
        assert_eq!(
            (ledger.partition_promised(0), ledger.partition_promised(1)),
            (0, 8)
        );
        assert_eq!(ledger.partition_promised_count(1), 1);
        ledger.drain_promised();
        assert_eq!(ledger.partition_promised_count(1), 0);
    }

    #[test]
    fn holding_moves_a_request_out_of_placement_and_totals_per_partition() {
        let mut ledger = RequestLedger::new(2);
        ledger.promise(RequestId(0), 0, 10);
        ledger.hold(RequestId(0), 0, 64);
        assert_eq!(ledger.placement(RequestId(0)), None);
        assert_eq!(ledger.partition_held(0), 64);

        // Re-holding on another partition moves the whole total, never doubles it.
        ledger.hold(RequestId(0), 1, 64);
        assert_eq!(
            (ledger.partition_held(0), ledger.partition_held(1)),
            (0, 64)
        );

        assert_eq!(ledger.take_held(RequestId(0)), Some((1, 64)));
        assert_eq!(ledger.partition_held(1), 0);
        assert_eq!(ledger.take_held(RequestId(0)), None);
    }
}
