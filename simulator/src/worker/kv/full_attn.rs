//! Full-attention KV accounting for composed L5 workers.
//!
//! `FullAttnKv` owns partition placement, promised/held ledgers, and occupancy
//! sampling. Each partition's resident decode membership and capacity accounting
//! live in `FullAttnPartitionState`. Iteration input remains owned by Execution;
//! borrowed iterators avoid per-iteration membership copies.

use std::collections::HashMap;

use crate::common::{PrefixInput, RequestId, Time};
use crate::log::{KvSampler, KvSubmit};
use crate::worker::kv::{
    HandoffKv, IterWorkerKv, KvStore, PrefixCachePolicy, PrefixKv, PrefixResolution, SlotPipelineKv,
};
use crate::worker::shared::advance_scope::{AdvanceScope, PartitionId};

use super::full_attn_partition::FullAttnPartitionState;
use super::prefix_cache::{PrefixCache, PrefixCacheLease};

pub struct FullAttentionKvFootprint {
    initial_context: u32,
    decode: u32,
}

impl FullAttentionKvFootprint {
    #[inline]
    fn reserved_tokens(&self) -> u64 {
        u64::from(self.initial_context) + u64::from(self.decode)
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
    /// Runtime prefix resolution stays beside request placement and residency;
    /// the shared request record carries only the immutable declaration.
    prefix_resolutions: HashMap<RequestId, PrefixResolution>,
    /// Evictable completed-session KV, local to each attention partition.
    prefix_caches: Vec<PrefixCache>,
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

    fn complete_handoff(&mut self, request: RequestId) {
        FullAttnKv::complete_handoff(self, request);
    }
}

impl PrefixKv for FullAttnKv {
    fn retained_prefix_partition(&self, prefix: PrefixInput) -> Option<PartitionId> {
        let PrefixInput::Session {
            session_id,
            declared_prefix_tokens,
        } = prefix
        else {
            return None;
        };

        let mut best_partition = None;
        let mut best_resident_tokens = 0;
        for (partition_index, prefix_cache) in self.prefix_caches.iter().enumerate() {
            let resident_tokens = prefix_cache.peek(session_id, declared_prefix_tokens);
            if resident_tokens > best_resident_tokens {
                best_resident_tokens = resident_tokens;
                best_partition = Some(
                    PartitionId::try_from(partition_index)
                        .expect("FullAttnKv partition count must fit PartitionId"),
                );
            }
        }
        best_partition
    }

    fn plan_prefix(
        &self,
        partition: PartitionId,
        fresh_prompt_tokens: u32,
        prefix: PrefixInput,
    ) -> PrefixResolution {
        let resident_prefix_tokens = match prefix {
            PrefixInput::None => 0,
            PrefixInput::Session {
                session_id,
                declared_prefix_tokens,
            } => self.prefix_caches[partition as usize].peek(session_id, declared_prefix_tokens),
        };
        PrefixResolution::new(prefix, resident_prefix_tokens, fresh_prompt_tokens)
    }

    fn reserve_prefix(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        mut resolution: PrefixResolution,
        footprint: Self::Footprint,
    ) {
        if let Some(session_id) = resolution
            .session_id
            .filter(|_| resolution.resident_prefix_tokens > 0)
        {
            let (consumed, cache_lease) = self.prefix_caches[partition as usize]
                .take(session_id, resolution.resident_prefix_tokens);
            debug_assert_eq!(
                consumed, resolution.resident_prefix_tokens,
                "prefix plan changed between preview and reservation"
            );
            resolution = resolution.with_cache_lease(cache_lease);
        }
        self.prefix_resolutions.insert(request, resolution);
        self.reserve(request, partition, footprint);
    }

    fn prefix_resolution(&self, request: RequestId) -> PrefixResolution {
        *self
            .prefix_resolutions
            .get(&request)
            .expect("prefix resolution must exist for an admitted fresh request")
    }

    fn release_retaining_prefix(&mut self, request: RequestId, partition: PartitionId) {
        FullAttnKv::release_retaining_prefix(self, request, partition);
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
            + self.prefix_caches[partition as usize].used_tokens()
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
        Self::with_prefix_cache(
            num_partitions,
            kv_capacity,
            0,
            PrefixCachePolicy::Lru,
            sampler,
        )
    }

    pub(crate) fn with_prefix_cache(
        num_partitions: usize,
        kv_capacity: u64,
        prefix_cache_capacity: u64,
        prefix_cache_policy: PrefixCachePolicy,
        sampler: Option<KvSampler>,
    ) -> Self {
        Self {
            partitions: (0..num_partitions)
                .map(|_| FullAttnPartitionState::new(kv_capacity))
                .collect(),
            promised: HashMap::new(),
            held: HashMap::new(),
            held_tokens_by_partition: vec![0; num_partitions],
            request_to_partition: HashMap::new(),
            prefix_resolutions: HashMap::new(),
            prefix_caches: (0..num_partitions)
                .map(|_| {
                    PrefixCache::new(prefix_cache_capacity.min(kv_capacity), prefix_cache_policy)
                })
                .collect(),
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
        initial_context: u32,
        decode: u32,
    ) -> FullAttentionKvFootprint {
        FullAttentionKvFootprint {
            initial_context,
            decode,
        }
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
        self.trim_prefix_cache_to_physical_slack(partition);
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
        self.discard_held(request);
        self.promised.remove(&request);
        let current_kv = self.partitions[partition as usize]
            .decode_current_kv(request)
            .unwrap_or(0);
        self.partitions[partition as usize].release_decode(request, current_kv);
        self.request_to_partition.remove(&request);
        self.prefix_resolutions.remove(&request);
    }

    pub(crate) fn release_retaining_prefix(&mut self, request: RequestId, partition: PartitionId) {
        let resolution = self.prefix_resolutions.get(&request).copied();
        let retained_tokens = self.partitions[partition as usize]
            .decode_current_kv(request)
            .or_else(|| resolution.map(|value| u64::from(value.initial_context_tokens())))
            .unwrap_or(0);
        let session_id = resolution.and_then(|value| value.session_id);
        let cache_lease = resolution.and_then(PrefixResolution::cache_lease);
        self.release(request, partition);
        if let Some(session_id) = session_id {
            self.retain_prefix(partition, session_id, retained_tokens, cache_lease);
        }
    }

    /// Cancellation preserves the old caller-provided `current_kv` and
    /// `swap_remove` behavior for a request still in `prefill_admits`.
    pub(crate) fn release_external(
        &mut self,
        request: RequestId,
        current_kv: u64,
    ) -> Option<PartitionId> {
        let Some(partition) = self.request_to_partition.remove(&request) else {
            self.discard_held(request);
            self.prefix_resolutions.remove(&request);
            return None;
        };
        let partition_state = &mut self.partitions[partition as usize];
        partition_state.release_decode(request, current_kv);
        partition_state.remove_prefill_admit(request);
        self.promised.remove(&request);
        self.prefix_resolutions.remove(&request);
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

    fn discard_held(&mut self, request: RequestId) {
        if let Some((partition, kv_tokens)) = self.held.remove(&request) {
            let total = &mut self.held_tokens_by_partition[partition as usize];
            *total = total.saturating_sub(kv_tokens);
        }
    }

    pub(crate) fn complete_handoff(&mut self, request: RequestId) {
        let Some((partition, kv_tokens)) = self.held.remove(&request) else {
            return;
        };
        let total = &mut self.held_tokens_by_partition[partition as usize];
        *total = total.saturating_sub(kv_tokens);
        let resolution = self.prefix_resolutions.remove(&request);
        let session_id = resolution.and_then(|value| value.session_id);
        let cache_lease = resolution.and_then(PrefixResolution::cache_lease);
        if let Some(session_id) = session_id {
            self.retain_prefix(partition, session_id, kv_tokens, cache_lease);
        }
    }

    pub(crate) fn sample_submit(&mut self, partition: PartitionId, now: Time) {
        if self.sampler.is_none() {
            return;
        }
        let submit = KvSubmit {
            active_kv: self.partitions[partition as usize].resident_tokens()
                + self.held_tokens_by_partition[partition as usize]
                + self.prefix_caches[partition as usize].used_tokens(),
            projected_peak: self.partitions[partition as usize].projected_peak_kv()
                + self.held_tokens_by_partition[partition as usize]
                + self.prefix_caches[partition as usize].used_tokens(),
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

    fn non_cache_peak(&self, partition: PartitionId) -> u64 {
        let reserved =
            self.partition_promised(partition) + self.held_tokens_by_partition[partition as usize];
        let partition_state = &self.partitions[partition as usize];
        (partition_state.resident_tokens() + reserved)
            .max(partition_state.projected_peak_kv() + reserved)
    }

    fn prefix_cache_physical_limit(&self, partition: PartitionId) -> u64 {
        self.partitions[partition as usize]
            .capacity_tokens()
            .saturating_sub(self.non_cache_peak(partition))
    }

    fn trim_prefix_cache_to_physical_slack(&mut self, partition: PartitionId) {
        let physical_limit = self.prefix_cache_physical_limit(partition);
        self.prefix_caches[partition as usize].shrink_to(physical_limit);
    }

    fn retain_prefix(
        &mut self,
        partition: PartitionId,
        session_id: u32,
        tokens: u64,
        cache_lease: Option<PrefixCacheLease>,
    ) {
        let physical_limit = self.prefix_cache_physical_limit(partition);
        self.prefix_caches[partition as usize].insert(
            session_id,
            tokens,
            physical_limit,
            cache_lease,
        );
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
    use crate::common::PrefixInput;

    #[test]
    fn strict_capacity_gate_counts_promised_tokens() {
        let mut kv_store = FullAttnKv::new(1, 100, None);
        let candidate = kv_store.footprint(RequestId(0), 60, 30);
        assert!(kv_store.fits(0, &candidate));

        let reservation = kv_store.footprint(RequestId(1), 10, 10);
        kv_store.reserve(RequestId(1), 0, reservation);
        assert!(!kv_store.fits(0, &candidate));
    }

    #[test]
    fn disabled_prefix_cache_recomputes_the_declared_prefix() {
        let kv_store = FullAttnKv::new(1, 1_000, None);
        let resolution = kv_store.plan_prefix(
            0,
            20,
            PrefixInput::Session {
                session_id: 7,
                declared_prefix_tokens: 100,
            },
        );
        assert_eq!(resolution.resident_prefix_tokens(), 0);
        assert_eq!(resolution.prefill_compute_tokens(), 120);
        assert_eq!(resolution.initial_context_tokens(), 120);
    }

    #[test]
    fn retained_prefix_partition_prefers_the_largest_hit_then_lower_partition() {
        let mut kv_store = FullAttnKv::with_prefix_cache(3, 100, 100, PrefixCachePolicy::Lru, None);
        kv_store.prefix_caches[1].insert(7, 50, 100, None);
        kv_store.prefix_caches[2].insert(7, 80, 100, None);

        assert_eq!(
            kv_store.retained_prefix_partition(PrefixInput::Session {
                session_id: 7,
                declared_prefix_tokens: 60,
            }),
            Some(2)
        );
        assert_eq!(
            kv_store.retained_prefix_partition(PrefixInput::Session {
                session_id: 7,
                declared_prefix_tokens: 40,
            }),
            Some(1),
            "equal reusable lengths keep the lowest partition deterministic"
        );
        assert_eq!(kv_store.retained_prefix_partition(PrefixInput::None), None);
    }

    #[test]
    fn active_reservation_evicts_prefix_cache_within_one_total_capacity() {
        let mut kv_store = FullAttnKv::with_prefix_cache(1, 100, 100, PrefixCachePolicy::Lru, None);
        kv_store.prefix_caches[0].insert(1, 80, 100, None);
        assert_eq!(kv_store.prefix_caches[0].used_tokens(), 80);

        let footprint = kv_store.footprint(RequestId(0), 60, 30);
        assert!(kv_store.fits(0, &footprint));
        kv_store.reserve(RequestId(0), 0, footprint);

        assert_eq!(kv_store.partition_promised(0), 90);
        assert_eq!(kv_store.prefix_caches[0].used_tokens(), 0);
        assert!(kv_store.non_cache_peak(0) + kv_store.prefix_caches[0].used_tokens() <= 100);
    }

    #[test]
    fn prefix_hit_moves_cache_ownership_to_the_request_then_returns_on_release() {
        let mut kv_store = FullAttnKv::with_prefix_cache(1, 100, 100, PrefixCachePolicy::Lru, None);
        kv_store.prefix_caches[0].insert(7, 60, 100, None);
        let resolution = kv_store.plan_prefix(
            0,
            10,
            PrefixInput::Session {
                session_id: 7,
                declared_prefix_tokens: 50,
            },
        );
        assert_eq!(resolution.resident_prefix_tokens(), 50);
        assert_eq!(resolution.prefill_compute_tokens(), 10);

        let footprint = kv_store.footprint(RequestId(0), 60, 10);
        assert!(kv_store.fits(0, &footprint));
        kv_store.reserve_prefix(RequestId(0), 0, resolution, footprint);
        assert_eq!(kv_store.prefix_caches[0].used_tokens(), 0);

        kv_store.drain_ready();
        kv_store.release_retaining_prefix(RequestId(0), 0);
        assert_eq!(kv_store.prefix_caches[0].peek(7, 60), 60);
        assert_eq!(kv_store.prefix_caches[0].used_tokens(), 60);
    }
}
