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
//! - `HashMap` iteration order is observable: [`Self::drain_promised`] feeds
//!   prefill-admit order, which feeds model input and event order. Do not swap
//!   the container for one with a different order without re-recording goldens.

use std::collections::HashMap;

use crate::common::RequestId;
use crate::worker::kv::ResolvedPrefillContext;
use crate::worker::shared::advance_scope::PartitionId;

pub(crate) struct RequestLedger {
    /// Reserved-but-not-yet-resident footprints, as `(partition, charge)`.
    promised: HashMap<RequestId, (PartitionId, u64)>,
    /// Full-footprint reservations held across partial-prefill iterations.
    chunked_prefill: HashMap<RequestId, (PartitionId, u64)>,
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
            chunked_prefill: HashMap::new(),
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
        self.promised.insert(request, (partition, charge));
        self.placement.insert(request, partition);
    }

    pub(crate) fn forget_promise(&mut self, request: RequestId) {
        self.promised.remove(&request);
    }

    pub(crate) fn has_promise(&self, request: RequestId) -> bool {
        self.promised.contains_key(&request) || self.chunked_prefill.contains_key(&request)
    }

    pub(crate) fn promised_charge(&self, request: RequestId) -> Option<u64> {
        self.promised
            .get(&request)
            .or_else(|| self.chunked_prefill.get(&request))
            .map(|(_, charge)| *charge)
    }

    #[inline]
    pub(crate) fn partition_promised(&self, partition: PartitionId) -> u64 {
        let newly_promised: u64 = self
            .promised
            .values()
            .filter(|(promised_partition, _)| *promised_partition == partition)
            .map(|(_, charge)| *charge)
            .sum();
        let chunked_prefill: u64 = self
            .chunked_prefill
            .values()
            .filter(|(reserved_partition, _)| *reserved_partition == partition)
            .map(|(_, charge)| *charge)
            .sum();
        newly_promised + chunked_prefill
    }

    #[inline]
    pub(crate) fn partition_promised_count(&self, partition: PartitionId) -> u32 {
        let newly_promised = self
            .promised
            .values()
            .filter(|(promised_partition, _)| *promised_partition == partition)
            .count();
        let chunked_prefill = self
            .chunked_prefill
            .values()
            .filter(|(reserved_partition, _)| *reserved_partition == partition)
            .count();
        u32::try_from(newly_promised + chunked_prefill)
            .expect("partition reservation count exceeds u32")
    }

    pub(crate) fn promote_promise_to_chunked_prefill(&mut self, request: RequestId) {
        let reservation = self
            .promised
            .remove(&request)
            .expect("chunked prefill must promote an existing promise");
        self.chunked_prefill.insert(request, reservation);
    }

    pub(crate) fn forget_chunked_prefill(&mut self, request: RequestId) {
        self.chunked_prefill.remove(&request);
    }

    pub(crate) fn has_chunked_prefill(&self, request: RequestId) -> bool {
        self.chunked_prefill.contains_key(&request)
    }

    /// Empty `promised` and report `(partition, request)` in iteration order.
    ///
    /// Collect-then-clear rather than `drain()` so the order the store observes
    /// is the same `HashMap` order the pre-split code produced.
    pub(crate) fn drain_promised(&mut self) -> Vec<(PartitionId, RequestId)> {
        let drained: Vec<(PartitionId, RequestId)> = self
            .promised
            .iter()
            .map(|(&request, &(partition, _))| (partition, request))
            .collect();
        for (_, request) in &drained {
            self.promised.remove(request);
        }
        drained
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
    fn promise_places_and_drain_clears_both_the_promise_and_nothing_else() {
        let mut ledger = RequestLedger::new(2);
        ledger.promise(RequestId(0), 1, 30);
        ledger.promise(RequestId(1), 1, 40);
        ledger.promise(RequestId(2), 0, 50);
        assert_eq!(ledger.partition_promised(1), 70);
        assert_eq!(ledger.partition_promised_count(1), 2);
        assert_eq!(ledger.partition_promised(0), 50);

        let mut drained = ledger.drain_promised();
        drained.sort_by_key(|(_, request)| request.0);
        assert_eq!(
            drained,
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
