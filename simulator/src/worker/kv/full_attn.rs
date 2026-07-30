//! Full-attention KV accounting for composed L5 workers.
//!
//! `FullAttnKv` owns partition placement, promised/held ledgers, and occupancy
//! sampling. Each partition's resident decode membership and capacity accounting
//! live in `FullAttnPartitionState`. Iteration input remains owned by Execution;
//! borrowed iterators avoid per-iteration membership copies.

use std::collections::HashMap;

use crate::common::{RequestId, Time};
use crate::log::{KvSampler, KvSubmit};
use crate::worker::kv::{HandoffKv, IterWorkerKv, KvStore, SlotPipelineKv};
use crate::worker::shared::advance_scope::{AdvanceScope, PartitionId};

use super::full_attn_partition::FullAttnPartitionState;

pub struct FullAttentionKvFootprint {
    prompt: u32,
    decode: u32,
}

impl FullAttentionKvFootprint {
    #[inline]
    fn reserved_tokens(&self) -> u64 {
        (self.prompt + self.decode) as u64
    }
}

pub struct FullAttnKv {
    partitions: Vec<FullAttnPartitionState>,
    /// Preserve `HashMap` and its drain order: it feeds prefill/input/event order.
    promised: HashMap<RequestId, (PartitionId, u64)>,
    /// Prefilled KV awaiting a decode-side pull acknowledgement.
    held: HashMap<RequestId, (PartitionId, u64)>,
    held_tokens_by_partition: Vec<u64>,
    request_to_partition: HashMap<RequestId, PartitionId>,
    sampler: Option<KvSampler>,
}

impl KvStore for FullAttnKv {
    type Footprint = FullAttentionKvFootprint;

    fn num_partitions(&self) -> usize {
        FullAttnKv::num_partitions(self)
    }

    fn footprint(&self, request: RequestId, prompt: u32, decode: u32) -> Self::Footprint {
        FullAttnKv::footprint(self, request, prompt, decode)
    }

    fn fits(&self, partition: PartitionId, footprint: &Self::Footprint) -> bool {
        FullAttnKv::fits(self, partition, footprint)
    }

    fn reserve(&mut self, request: RequestId, partition: PartitionId, footprint: Self::Footprint) {
        FullAttnKv::reserve(self, request, partition, footprint);
    }

    fn commit_resident(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        initial_kv: u64,
        remaining: u32,
    ) {
        FullAttnKv::commit_resident(self, request, partition, initial_kv, remaining);
    }

    fn advance(&mut self, scope: AdvanceScope<'_>, steps: u32) {
        FullAttnKv::advance(self, scope, steps);
    }

    fn release(&mut self, request: RequestId, partition: PartitionId) {
        FullAttnKv::release(self, request, partition);
    }

    fn sample_submit(&mut self, partition: PartitionId, now: Time) {
        FullAttnKv::sample_submit(self, partition, now);
    }
}

impl IterWorkerKv for FullAttnKv {
    fn drain_ready(&mut self) {
        FullAttnKv::drain_ready(self);
    }

    fn clear_prefill_admits(&mut self, partition: PartitionId) {
        FullAttnKv::clear_prefill_admits(self, partition);
    }

    fn has_live_decode(&self, partition: PartitionId) -> bool {
        FullAttnKv::has_live_decode(self, partition)
    }

    fn live_decode_count(&self, partition: PartitionId) -> u32 {
        FullAttnKv::live_decode_count(self, partition)
    }

    fn has_prefill_admit(&self, partition: PartitionId) -> bool {
        FullAttnKv::has_prefill_admit(self, partition)
    }

    fn status_active(&self, partition: PartitionId) -> u32 {
        FullAttnKv::status_active(self, partition)
    }

    fn release_external(&mut self, request: RequestId, current_kv: u64) -> Option<PartitionId> {
        FullAttnKv::release_external(self, request, current_kv)
    }

    fn visit_prefill_admits(&self, partition: PartitionId, mut visitor: impl FnMut(RequestId)) {
        for request in self.prefill_admits(partition) {
            visitor(request);
        }
    }

    fn visit_decode_members(
        &self,
        partition: PartitionId,
        mut visitor: impl FnMut(RequestId, u64),
    ) {
        for (request, current_kv) in self.decode_members(partition) {
            visitor(request, current_kv);
        }
    }
}

impl HandoffKv for FullAttnKv {
    fn hold(&mut self, partition: PartitionId, request: RequestId, kv_tokens: u64) {
        FullAttnKv::hold(self, partition, request, kv_tokens);
    }

    fn drop_held(&mut self, request: RequestId) {
        FullAttnKv::drop_held(self, request);
    }
}

impl SlotPipelineKv for FullAttnKv {
    fn current_kv(&self, partition: PartitionId, request: RequestId) -> Option<u64> {
        self.partitions[partition as usize].decode_current_kv(request)
    }

    fn estimated_peak(&self, partition: PartitionId) -> u64 {
        self.partitions[partition as usize].projected_peak_kv()
            + self.partition_promised(partition)
            + self.held_tokens_by_partition[partition as usize]
    }

    fn has_reservation(&self, request: RequestId) -> bool {
        self.promised.contains_key(&request)
    }

    fn request_kv_weight(&self, request: RequestId) -> u64 {
        self.promised
            .get(&request)
            .map(|(_, tokens)| *tokens)
            .or_else(|| {
                self.request_to_partition
                    .get(&request)
                    .and_then(|partition| {
                        self.partitions[*partition as usize].decode_current_kv(request)
                    })
            })
            .unwrap_or(0)
    }
}

impl FullAttnKv {
    pub(crate) fn new(num_partitions: usize, kv_capacity: u64, sampler: Option<KvSampler>) -> Self {
        Self {
            partitions: (0..num_partitions)
                .map(|_| FullAttnPartitionState::new(kv_capacity))
                .collect(),
            promised: HashMap::new(),
            held: HashMap::new(),
            held_tokens_by_partition: vec![0; num_partitions],
            request_to_partition: HashMap::new(),
            sampler,
        }
    }

    #[inline]
    pub(crate) fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    #[inline]
    pub(crate) fn footprint(
        &self,
        _request: RequestId,
        prompt: u32,
        decode: u32,
    ) -> FullAttentionKvFootprint {
        FullAttentionKvFootprint { prompt, decode }
    }

    #[inline]
    pub(crate) fn fits(
        &self,
        partition: PartitionId,
        footprint: &FullAttentionKvFootprint,
    ) -> bool {
        let partition_state = &self.partitions[partition as usize];
        let reserved_tokens = self.partition_promised(partition)
            + self.held_tokens_by_partition[partition as usize]
            + footprint.reserved_tokens();
        partition_state.resident_tokens() + reserved_tokens <= partition_state.capacity_tokens()
            && partition_state.projected_peak_kv() + reserved_tokens
                <= partition_state.capacity_tokens()
    }

    pub(crate) fn reserve(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        footprint: FullAttentionKvFootprint,
    ) {
        self.promised
            .insert(request, (partition, footprint.reserved_tokens()));
        self.request_to_partition.insert(request, partition);
    }

    /// Local full-attention KV is immediately ready. Preserve the existing
    /// collect-then-remove choreography and `HashMap` iteration order.
    pub(crate) fn drain_ready(&mut self) {
        let drained: Vec<(PartitionId, RequestId)> = self
            .promised
            .iter()
            .map(|(&request, &(partition, _))| (partition, request))
            .collect();
        for (partition, request) in drained {
            self.promised.remove(&request);
            self.partitions[partition as usize].add_prefill_admit(request);
        }
    }

    pub(crate) fn commit_resident(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        initial_kv: u64,
        remaining: u32,
    ) {
        // Iter-prefill already drained this reservation; pull-decode commits
        // directly after reserve, so clearing here keeps both lifecycles on the
        // same KV-owned request→partition ledger.
        self.promised.remove(&request);
        self.partitions[partition as usize].begin_decode(request, initial_kv, remaining);
    }

    pub(crate) fn advance(&mut self, scope: AdvanceScope<'_>, steps: u32) {
        for _ in 0..steps {
            match scope {
                AdvanceScope::WholePartition(partition) => {
                    self.partitions[partition as usize].advance_decodes();
                }
                AdvanceScope::RequestSubset {
                    partition,
                    request_ids,
                } => {
                    self.debug_assert_partition(partition, request_ids);
                    self.partitions[partition as usize].advance_subset(request_ids);
                }
            }
        }
    }

    pub(crate) fn clear_prefill_admits(&mut self, partition: PartitionId) {
        self.partitions[partition as usize].clear_prefill_admits();
    }

    pub(crate) fn release(&mut self, request: RequestId, partition: PartitionId) {
        self.drop_held(request);
        self.promised.remove(&request);
        let current_kv = self.partitions[partition as usize]
            .decode_current_kv(request)
            .unwrap_or(0);
        self.partitions[partition as usize].release_decode(request, current_kv);
        self.request_to_partition.remove(&request);
    }

    /// Cancellation preserves the old caller-provided `current_kv` and
    /// `swap_remove` behavior for a request still in `prefill_admits`.
    pub(crate) fn release_external(
        &mut self,
        request: RequestId,
        current_kv: u64,
    ) -> Option<PartitionId> {
        let Some(partition) = self.request_to_partition.remove(&request) else {
            self.drop_held(request);
            return None;
        };
        let partition_state = &mut self.partitions[partition as usize];
        partition_state.release_decode(request, current_kv);
        partition_state.remove_prefill_admit(request);
        self.promised.remove(&request);
        Some(partition)
    }

    pub(crate) fn hold(&mut self, partition: PartitionId, request: RequestId, kv_tokens: u64) {
        self.request_to_partition.remove(&request);
        if let Some((previous_partition, previous_tokens)) =
            self.held.insert(request, (partition, kv_tokens))
        {
            let total = &mut self.held_tokens_by_partition[previous_partition as usize];
            *total = total.saturating_sub(previous_tokens);
        }
        self.held_tokens_by_partition[partition as usize] += kv_tokens;
    }

    pub(crate) fn drop_held(&mut self, request: RequestId) {
        if let Some((partition, kv_tokens)) = self.held.remove(&request) {
            let total = &mut self.held_tokens_by_partition[partition as usize];
            *total = total.saturating_sub(kv_tokens);
        }
    }

    pub(crate) fn sample_submit(&mut self, partition: PartitionId, now: Time) {
        if self.sampler.is_none() {
            return;
        }
        let submit = KvSubmit {
            active_kv: self.partitions[partition as usize].resident_tokens()
                + self.held_tokens_by_partition[partition as usize],
            projected_peak: self.partitions[partition as usize].projected_peak_kv()
                + self.held_tokens_by_partition[partition as usize],
            promised_kv: self.partition_promised(partition),
        };
        self.sampler
            .as_mut()
            .unwrap()
            .submit(partition, submit, now);
    }

    #[inline]
    pub(crate) fn has_live_decode(&self, partition: PartitionId) -> bool {
        self.partitions[partition as usize].has_live_decode()
    }

    #[inline]
    pub(crate) fn live_decode_count(&self, partition: PartitionId) -> u32 {
        self.partitions[partition as usize].live_decode_count()
    }

    #[inline]
    pub(crate) fn has_prefill_admit(&self, partition: PartitionId) -> bool {
        self.partitions[partition as usize].has_prefill_admit()
    }

    #[inline]
    pub(crate) fn status_active(&self, partition: PartitionId) -> u32 {
        self.live_decode_count(partition)
            + self.partition_promised_count(partition)
            + self.partitions[partition as usize].prefill_admit_count()
    }

    /// Borrowed views keep input construction and completion bookkeeping
    /// allocation-equivalent to the pre-refactor worker.
    pub(crate) fn prefill_admits(
        &self,
        partition: PartitionId,
    ) -> impl Iterator<Item = RequestId> + '_ {
        self.partitions[partition as usize].iter_prefill_admits()
    }

    pub(crate) fn decode_members(
        &self,
        partition: PartitionId,
    ) -> impl Iterator<Item = (RequestId, u64)> + '_ {
        self.partitions[partition as usize].decode_members()
    }

    #[inline]
    fn partition_promised(&self, partition: PartitionId) -> u64 {
        self.promised
            .values()
            .filter(|(reserved_partition, _)| *reserved_partition == partition)
            .map(|(_, tokens)| *tokens)
            .sum()
    }

    #[inline]
    fn partition_promised_count(&self, partition: PartitionId) -> u32 {
        self.promised
            .values()
            .filter(|(reserved_partition, _)| *reserved_partition == partition)
            .count() as u32
    }

    #[inline]
    fn debug_assert_partition(&self, partition: PartitionId, requests: &[RequestId]) {
        debug_assert!(
            requests.iter().all(|request| {
                self.request_to_partition.get(request).copied() == Some(partition)
            }),
            "advance scope contains a request outside KV partition {partition}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_capacity_gate_counts_promised_tokens() {
        let mut kv_store = FullAttnKv::new(1, 100, None);
        let candidate = kv_store.footprint(RequestId(0), 60, 30);
        assert!(kv_store.fits(0, &candidate));

        let reservation = kv_store.footprint(RequestId(1), 10, 10);
        kv_store.reserve(RequestId(1), 0, reservation);
        assert!(!kv_store.fits(0, &candidate));
    }
}
