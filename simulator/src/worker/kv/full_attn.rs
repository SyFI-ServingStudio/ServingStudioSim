//! Full-attention KV accounting for composed L5 workers.
//!
//! `FullAttnKv` owns partition placement, promised/held ledgers, and occupancy
//! sampling. Each partition's resident decode membership and capacity accounting
//! live in a [`ResidentPartitionState`]. Iteration input remains owned by
//! Execution; borrowed iterators avoid per-iteration membership copies.
//!
//! Every token of this model's cache is per-token attention KV, so the pieces
//! from [`super::shared`] are configured with their identity values: no fixed
//! per-request charge, and a prefix reuse quantum of one token. A model that
//! also carries recurrent state gets a different store (`hybrid_gdn`) rather
//! than a flag here.

use crate::common::{RequestId, SessionInput, Time};
use crate::log::{
    KvSampler, KvSubmit, PrefixCacheActivation, PrefixCacheEvent, PrefixCacheEventKind,
    PrefixCacheEvictionReason, PrefixCacheLogger, PrefixCacheRetentionReason,
};
use crate::worker::kv::shared::prefix_cache::{
    PrefixCache, PrefixCacheMutation, PrefixCacheReturnMetadata,
};
use crate::worker::kv::shared::prefix_cache_journal::PrefixCacheJournal;
use crate::worker::kv::shared::request_ledger::RequestLedger;
use crate::worker::kv::shared::resident_partition::ResidentPartitionState;
use crate::worker::kv::{
    ChunkedPrefillKv, HandoffKv, IterWorkerKv, KvStore, PrefixCachePolicy, PrefixCacheTokenConfig,
    PrefixKv, ResolvedPrefillContext, SlotPipelineKv,
};
use crate::worker::shared::advance_scope::{AdvanceScope, PartitionId};

/// No fixed per-request state: a full-attention request occupies exactly its
/// context tokens.
const NO_FIXED_CHARGE: u64 = 0;
/// Any prefix length is resumable, so retained KV is reusable token by token.
const TOKEN_QUANTUM: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReservationBasis {
    FullFootprint,
    BoundedFuture,
}

pub struct FullAttentionKvFootprint {
    reserved_tokens: u64,
    basis: ReservationBasis,
}

impl FullAttentionKvFootprint {
    #[inline]
    fn reserved_tokens(&self) -> u64 {
        self.reserved_tokens
    }
}

pub struct FullAttnKv {
    partitions: Vec<ResidentPartitionState>,
    ledger: RequestLedger,
    /// Evictable completed-session KV, local to each attention partition.
    prefix_caches: Vec<PrefixCache>,
    journal: PrefixCacheJournal,
    sampler: Option<KvSampler>,
}

impl KvStore for FullAttnKv {
    type Footprint = FullAttentionKvFootprint;

    fn num_partitions(&self) -> usize {
        FullAttnKv::num_partitions(self)
    }

    fn footprint(
        &self,
        request: RequestId,
        post_prefill_context_tokens: u32,
        remaining_output_tokens: u32,
    ) -> Self::Footprint {
        FullAttnKv::footprint(
            self,
            request,
            post_prefill_context_tokens,
            remaining_output_tokens,
        )
    }

    fn fits(&self, partition: PartitionId, footprint: &Self::Footprint) -> bool {
        FullAttnKv::fits(self, partition, footprint)
    }

    fn reserve(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        footprint: Self::Footprint,
        now: Time,
    ) {
        FullAttnKv::reserve(self, request, partition, footprint, now);
    }

    fn commit_resident(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        post_prefill_context_tokens: u64,
        remaining_output_tokens: u32,
    ) {
        FullAttnKv::commit_resident(
            self,
            request,
            partition,
            post_prefill_context_tokens,
            remaining_output_tokens,
        );
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

    fn complete_handoff(&mut self, request: RequestId, now: Time) {
        FullAttnKv::complete_handoff(self, request, now);
    }
}

impl ChunkedPrefillKv for FullAttnKv {
    fn bounded_future_footprint(
        &self,
        _request: RequestId,
        post_prefill_context_tokens: u32,
        remaining_output_tokens: u32,
        max_future_tokens: u32,
        page_size: u32,
    ) -> Self::Footprint {
        let reserved_tokens = u64::from(post_prefill_context_tokens)
            .checked_add(u64::from(remaining_output_tokens.min(max_future_tokens)))
            .and_then(|tokens| tokens.checked_add(u64::from(page_size)))
            .expect("bounded-future KV footprint overflow");
        FullAttentionKvFootprint {
            reserved_tokens,
            basis: ReservationBasis::BoundedFuture,
        }
    }

    fn fits_bounded_future(
        &self,
        partition: PartitionId,
        footprint: &Self::Footprint,
        max_future_tokens: u32,
        new_token_ratio: f64,
        decode_step_tokens: u32,
    ) -> bool {
        debug_assert_eq!(footprint.basis, ReservationBasis::BoundedFuture);
        let partition_state = &self.partitions[partition as usize];
        let protected_tokens = partition_state.resident_tokens()
            + self.ledger.partition_promised(partition)
            + self.ledger.partition_held(partition)
            + footprint.reserved_tokens();
        let mut running_future =
            partition_state.bounded_future_decode_tokens(max_future_tokens, new_token_ratio);
        if decode_step_tokens > 0 {
            // Bounded-future admission runs on one-token pages only.
            let next_step = partition_state.next_decode_allocation_tokens(1, decode_step_tokens);
            running_future = running_future.max(next_step as f64);
        }
        protected_tokens as f64 + running_future < partition_state.capacity_tokens() as f64
    }

    fn prepare_next_decode(
        &mut self,
        partition: PartitionId,
        page_size: u32,
        step_tokens: u32,
        now: Time,
    ) -> u64 {
        assert!(page_size > 0, "decode page size must be positive");
        let partition_state = &self.partitions[partition as usize];
        let required = partition_state.next_decode_allocation_tokens(page_size, step_tokens);
        if required == 0 {
            return 0;
        }
        let trigger = partition_state
            .first_decode_request()
            .expect("positive decode allocation requires a live request");
        let protected = partition_state.resident_tokens()
            + self.ledger.partition_promised(partition)
            + self.ledger.partition_held(partition);
        let cache_limit = partition_state
            .capacity_tokens()
            .saturating_sub(protected.saturating_add(required));
        let evictions = self.prefix_caches[partition as usize].shrink_to(cache_limit);
        for eviction in evictions {
            self.journal.record_mutation(
                trigger,
                partition,
                now,
                eviction,
                PrefixCacheEventKind::Evict(PrefixCacheEvictionReason::ActiveKvPressure),
                0,
            );
        }
        let cache_tokens = self.prefix_caches[partition as usize].used_charge();
        let available = partition_state
            .capacity_tokens()
            .saturating_sub(protected.saturating_add(cache_tokens));
        required.saturating_sub(available)
    }

    fn visit_decode_states(
        &self,
        partition: PartitionId,
        mut visitor: impl FnMut(RequestId, u64, u32),
    ) {
        for (request, current_kv, remaining_decode) in
            self.partitions[partition as usize].decode_states()
        {
            visitor(request, current_kv, remaining_decode);
        }
    }

    fn reserve_chunked_prefill_context(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        resolved_prefill: ResolvedPrefillContext,
        footprint: Self::Footprint,
        now: Time,
    ) {
        self.reserve_prefill_context(request, partition, resolved_prefill, footprint, now);
        self.ledger.promote_promise_to_chunked_prefill(request);
    }

    fn schedule_prefill_chunk(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        chunk_tokens: u32,
    ) {
        debug_assert!(self.ledger.has_chunked_prefill(request));
        self.ledger.schedule_prefill_chunk(request, chunk_tokens);
        self.partitions[partition as usize].add_prefill_admit(request);
    }

    fn complete_prefill_chunk(&mut self, request: RequestId) {
        self.ledger.complete_prefill_chunk(request);
    }

    fn finish_chunked_prefill(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        remaining_output_tokens: u32,
    ) {
        let context_tokens = self
            .ledger
            .prefill_context(request)
            .expect("chunked prefill context must exist")
            .post_prefill_context_tokens();
        self.ledger.forget_chunked_prefill(request);
        self.commit_resident(
            request,
            partition,
            u64::from(context_tokens),
            remaining_output_tokens,
        );
    }
}

impl PrefixKv for FullAttnKv {
    fn retained_prefix_partition(&self, session_input: SessionInput) -> Option<PartitionId> {
        let SessionInput::Session {
            session_id,
            declared_prefix_tokens,
            ..
        } = session_input
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

    fn preview_prefill_context(
        &self,
        partition: PartitionId,
        fresh_prompt_tokens: u32,
        session_input: SessionInput,
    ) -> ResolvedPrefillContext {
        let resident_prefix_tokens = match session_input {
            SessionInput::Standalone => 0,
            SessionInput::Session {
                session_id,
                declared_prefix_tokens,
                ..
            } => self.prefix_caches[partition as usize].peek(session_id, declared_prefix_tokens),
        };
        ResolvedPrefillContext::new(session_input, resident_prefix_tokens, fresh_prompt_tokens)
    }

    fn reserve_prefill_context(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        mut resolved_prefill: ResolvedPrefillContext,
        footprint: Self::Footprint,
        now: Time,
    ) {
        if let Some(session_id) = resolved_prefill.session_id() {
            let take_result = self.prefix_caches[partition as usize]
                .take(session_id, resolved_prefill.declared_prefix_tokens());
            debug_assert_eq!(
                take_result.hit_tokens,
                resolved_prefill.resident_prefix_tokens(),
                "prefill context changed between preview and reservation"
            );
            self.journal.record(PrefixCacheEvent {
                partition_id: partition,
                time: now,
                request_id: request,
                session_id,
                kind: PrefixCacheEventKind::Activate(if take_result.hit_tokens > 0 {
                    PrefixCacheActivation::Hit
                } else {
                    PrefixCacheActivation::Miss
                }),
                entry_tokens_before: take_result.entry_tokens,
                entry_tokens_after: 0,
                cache_used_before: take_result.cache_used_before,
                cache_used_after: take_result.cache_used_after,
                requested_tokens: u64::from(resolved_prefill.declared_prefix_tokens()),
                hit_tokens: u64::from(take_result.hit_tokens),
            });
            resolved_prefill =
                resolved_prefill.with_cache_return_metadata(take_result.return_metadata);
        }
        self.ledger.set_prefill_context(request, resolved_prefill);
        self.reserve(request, partition, footprint, now);
    }

    fn resolved_prefill_context(&self, request: RequestId) -> ResolvedPrefillContext {
        self.ledger
            .prefill_context(request)
            .expect("resolved prefill context must exist for an admitted fresh request")
    }

    fn release_retaining_prefix(&mut self, request: RequestId, partition: PartitionId, now: Time) {
        FullAttnKv::release_retaining_prefix(self, request, partition, now);
    }
}

impl SlotPipelineKv for FullAttnKv {
    fn current_kv(&self, partition: PartitionId, request: RequestId) -> Option<u64> {
        self.partitions[partition as usize].decode_current_kv(request)
    }

    fn estimated_peak(&self, partition: PartitionId) -> u64 {
        self.partitions[partition as usize].projected_peak_kv()
            + self.ledger.partition_promised(partition)
            + self.ledger.partition_held(partition)
            + self.prefix_caches[partition as usize].used_charge()
    }

    fn has_reservation(&self, request: RequestId) -> bool {
        self.ledger.has_promise(request)
    }

    fn request_kv_weight(&self, request: RequestId) -> u64 {
        self.ledger
            .promised_charge(request)
            .or_else(|| {
                self.ledger.placement(request).and_then(|partition| {
                    self.partitions[partition as usize].decode_current_kv(request)
                })
            })
            .unwrap_or(0)
    }
}

impl FullAttnKv {
    pub(crate) fn without_prefix_cache(
        num_partitions: usize,
        kv_capacity: u64,
        sampler: Option<KvSampler>,
    ) -> Self {
        Self::with_prefix_cache(
            num_partitions,
            kv_capacity,
            PrefixCacheTokenConfig {
                max_retained_tokens: 0,
                policy: PrefixCachePolicy::Lru,
            },
            sampler,
            None,
        )
    }

    pub(crate) fn with_prefix_cache(
        num_partitions: usize,
        kv_capacity: u64,
        prefix_cache: PrefixCacheTokenConfig,
        sampler: Option<KvSampler>,
        prefix_cache_logger: Option<PrefixCacheLogger>,
    ) -> Self {
        Self {
            partitions: (0..num_partitions)
                .map(|_| ResidentPartitionState::new(kv_capacity, NO_FIXED_CHARGE))
                .collect(),
            ledger: RequestLedger::new(num_partitions),
            prefix_caches: (0..num_partitions)
                .map(|_| {
                    PrefixCache::new(
                        prefix_cache.max_retained_tokens.min(kv_capacity),
                        prefix_cache.policy,
                        TOKEN_QUANTUM,
                        NO_FIXED_CHARGE,
                    )
                })
                .collect(),
            journal: PrefixCacheJournal::new(prefix_cache_logger),
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
        post_prefill_context_tokens: u32,
        remaining_output_tokens: u32,
    ) -> FullAttentionKvFootprint {
        FullAttentionKvFootprint {
            reserved_tokens: u64::from(post_prefill_context_tokens)
                + u64::from(remaining_output_tokens),
            basis: ReservationBasis::FullFootprint,
        }
    }

    #[inline]
    pub(crate) fn fits(
        &self,
        partition: PartitionId,
        footprint: &FullAttentionKvFootprint,
    ) -> bool {
        let partition_state = &self.partitions[partition as usize];
        let reserved_tokens = self.ledger.partition_promised(partition)
            + self.ledger.partition_held(partition)
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
        now: Time,
    ) {
        self.ledger
            .promise(request, partition, footprint.reserved_tokens());
        let evictions = match footprint.basis {
            ReservationBasis::FullFootprint => self.trim_prefix_cache_to_physical_slack(partition),
            ReservationBasis::BoundedFuture => self.trim_prefix_cache_to_bounded_slack(partition),
        };
        for eviction in evictions {
            self.journal.record_mutation(
                request,
                partition,
                now,
                eviction,
                PrefixCacheEventKind::Evict(PrefixCacheEvictionReason::ActiveKvPressure),
                0,
            );
        }
    }

    /// Local full-attention KV is immediately ready.
    pub(crate) fn drain_ready(&mut self) {
        for (partition, request) in self.ledger.drain_promised() {
            self.partitions[partition as usize].add_prefill_admit(request);
        }
    }

    pub(crate) fn commit_resident(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        post_prefill_context_tokens: u64,
        remaining_output_tokens: u32,
    ) {
        // Iter-prefill already drained this reservation; pull-decode commits
        // directly after reserve, so clearing here keeps both lifecycles on the
        // same KV-owned request→partition ledger.
        self.ledger.forget_promise(request);
        self.partitions[partition as usize].begin_decode(
            request,
            post_prefill_context_tokens,
            remaining_output_tokens,
        );
    }

    pub(crate) fn advance(&mut self, scope: AdvanceScope<'_>, steps: u32) {
        let partition = match scope {
            AdvanceScope::WholePartition(partition)
            | AdvanceScope::RequestSubset { partition, .. } => partition,
        };
        for _ in 0..steps {
            match scope {
                AdvanceScope::WholePartition(partition) => {
                    self.partitions[partition as usize].advance_decodes();
                }
                AdvanceScope::RequestSubset {
                    partition,
                    request_ids,
                } => {
                    debug_assert!(
                        self.ledger.all_placed_on(partition, request_ids),
                        "advance scope contains a request outside KV partition {partition}"
                    );
                    self.partitions[partition as usize].advance_subset(request_ids);
                }
            }
        }
        // Admission must have claimed every step's tokens before the step ran;
        // advancing never checks capacity itself.
        let state = &self.partitions[partition as usize];
        debug_assert!(
            state.resident_tokens() <= state.capacity_tokens(),
            "decode advance overfilled KV partition {partition}: {} of {} tokens",
            state.resident_tokens(),
            state.capacity_tokens()
        );
    }

    pub(crate) fn clear_prefill_admits(&mut self, partition: PartitionId) {
        self.partitions[partition as usize].clear_prefill_admits();
    }

    pub(crate) fn release(&mut self, request: RequestId, partition: PartitionId) {
        self.ledger.take_held(request);
        self.ledger.forget_promise(request);
        self.ledger.forget_chunked_prefill(request);
        let current_kv = self.partitions[partition as usize]
            .decode_current_kv(request)
            .unwrap_or(0);
        self.partitions[partition as usize].release_decode(request, current_kv);
        self.ledger.forget_placement(request);
        self.ledger.take_prefill_context(request);
    }

    pub(crate) fn release_retaining_prefix(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        now: Time,
    ) {
        let resolved_prefill = self.ledger.prefill_context(request);
        let retained_tokens = self.partitions[partition as usize]
            .decode_current_kv(request)
            .or_else(|| {
                resolved_prefill.map(|value| u64::from(value.post_prefill_context_tokens()))
            })
            .unwrap_or(0);
        let session_id = resolved_prefill.and_then(ResolvedPrefillContext::session_id);
        let cache_return_metadata =
            resolved_prefill.and_then(ResolvedPrefillContext::cache_return_metadata);
        self.release(request, partition);
        if let Some(session_id) = session_id {
            self.retain_prefix(
                request,
                partition,
                session_id,
                retained_tokens,
                cache_return_metadata,
                now,
                PrefixCacheRetentionReason::RequestComplete,
            );
        }
    }

    /// Cancellation preserves the old caller-provided `current_kv` and
    /// `swap_remove` behavior for a request still in `prefill_admits`.
    pub(crate) fn release_external(
        &mut self,
        request: RequestId,
        current_kv: u64,
    ) -> Option<PartitionId> {
        let Some(partition) = self.ledger.forget_placement(request) else {
            self.ledger.take_held(request);
            self.ledger.forget_chunked_prefill(request);
            self.ledger.take_prefill_context(request);
            return None;
        };
        let partition_state = &mut self.partitions[partition as usize];
        partition_state.release_decode(request, current_kv);
        partition_state.remove_prefill_admit(request);
        self.ledger.forget_promise(request);
        self.ledger.forget_chunked_prefill(request);
        self.ledger.take_prefill_context(request);
        Some(partition)
    }

    pub(crate) fn hold(&mut self, partition: PartitionId, request: RequestId, kv_tokens: u64) {
        self.ledger.hold(request, partition, kv_tokens);
    }

    pub(crate) fn complete_handoff(&mut self, request: RequestId, now: Time) {
        let Some((partition, kv_tokens)) = self.ledger.take_held(request) else {
            return;
        };
        let resolved_prefill = self.ledger.take_prefill_context(request);
        let session_id = resolved_prefill.and_then(ResolvedPrefillContext::session_id);
        let cache_return_metadata =
            resolved_prefill.and_then(ResolvedPrefillContext::cache_return_metadata);
        if let Some(session_id) = session_id {
            self.retain_prefix(
                request,
                partition,
                session_id,
                kv_tokens,
                cache_return_metadata,
                now,
                PrefixCacheRetentionReason::HandoffComplete,
            );
        }
        // A pull acknowledgement can be the prefill worker's final activity.
        // Submit here so `kv_snapshot` observes both the held-KV release and any
        // retained-prefix insertion instead of ending at the pre-ack state.
        self.sample_submit(partition, now);
    }

    pub(crate) fn sample_submit(&mut self, partition: PartitionId, now: Time) {
        if self.sampler.is_none() {
            return;
        }
        let retained_prefix_kv = self.prefix_caches[partition as usize].used_charge();
        let held = self.ledger.partition_held(partition);
        let submit = KvSubmit {
            active_kv: self.partitions[partition as usize].resident_tokens()
                + held
                + retained_prefix_kv,
            retained_prefix_kv,
            projected_peak: self.partitions[partition as usize].projected_peak_kv()
                + held
                + retained_prefix_kv,
            promised_kv: self.ledger.partition_promised(partition),
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
        let scheduled_chunked_prefills = self.partitions[partition as usize]
            .iter_prefill_admits()
            .filter(|request| self.ledger.has_chunked_prefill(*request))
            .count() as u32;
        self.live_decode_count(partition)
            + self.ledger.partition_promised_count(partition)
            + self.partitions[partition as usize].prefill_admit_count()
            - scheduled_chunked_prefills
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

    fn non_cache_peak(&self, partition: PartitionId) -> u64 {
        let reserved =
            self.ledger.partition_promised(partition) + self.ledger.partition_held(partition);
        let partition_state = &self.partitions[partition as usize];
        (partition_state.resident_tokens() + reserved)
            .max(partition_state.projected_peak_kv() + reserved)
    }

    fn prefix_cache_physical_limit(&self, partition: PartitionId) -> u64 {
        self.partitions[partition as usize]
            .capacity_tokens()
            .saturating_sub(self.non_cache_peak(partition))
    }

    fn trim_prefix_cache_to_physical_slack(
        &mut self,
        partition: PartitionId,
    ) -> Vec<PrefixCacheMutation> {
        let physical_limit = self.prefix_cache_physical_limit(partition);
        self.prefix_caches[partition as usize].shrink_to(physical_limit)
    }

    fn trim_prefix_cache_to_bounded_slack(
        &mut self,
        partition: PartitionId,
    ) -> Vec<PrefixCacheMutation> {
        let partition_state = &self.partitions[partition as usize];
        let protected = partition_state.resident_tokens()
            + self.ledger.partition_promised(partition)
            + self.ledger.partition_held(partition);
        let physical_limit = partition_state.capacity_tokens().saturating_sub(protected);
        self.prefix_caches[partition as usize].shrink_to(physical_limit)
    }

    fn retain_prefix(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        session_id: u32,
        tokens: u64,
        cache_return_metadata: Option<PrefixCacheReturnMetadata>,
        now: Time,
        retain_reason: PrefixCacheRetentionReason,
    ) {
        let physical_limit = self.prefix_cache_physical_limit(partition);
        let insert_result = self.prefix_caches[partition as usize].insert(
            session_id,
            tokens,
            physical_limit,
            cache_return_metadata,
        );
        for mutation in insert_result.mutations {
            let event_kind = PrefixCacheJournal::retain_event_kind(mutation.kind, retain_reason);
            self.journal
                .record_mutation(request, partition, now, mutation, event_kind, tokens);
        }
        if insert_result.retained_tokens == 0 {
            let cache_used = self.prefix_caches[partition as usize].used_charge();
            self.journal.record(PrefixCacheEvent {
                partition_id: partition,
                time: now,
                request_id: request,
                session_id,
                kind: PrefixCacheEventKind::Retain(PrefixCacheRetentionReason::NoCacheCapacity),
                entry_tokens_before: 0,
                entry_tokens_after: 0,
                cache_used_before: cache_used,
                cache_used_after: cache_used,
                requested_tokens: tokens,
                hit_tokens: 0,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;

    use arrow_array::{StringArray, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::tempdir;

    use crate::common::SessionInput;

    fn capped_prefix_cache(
        max_retained_tokens: u64,
        policy: PrefixCachePolicy,
    ) -> PrefixCacheTokenConfig {
        PrefixCacheTokenConfig {
            max_retained_tokens,
            policy,
        }
    }

    #[test]
    fn strict_capacity_gate_counts_promised_tokens() {
        let mut kv_store = FullAttnKv::without_prefix_cache(1, 100, None);
        let candidate = kv_store.footprint(RequestId(0), 60, 30);
        assert!(kv_store.fits(0, &candidate));

        let reservation = kv_store.footprint(RequestId(1), 10, 10);
        kv_store.reserve(RequestId(1), 0, reservation, Time::ZERO);
        assert!(!kv_store.fits(0, &candidate));
    }

    #[test]
    fn disabled_prefix_cache_recomputes_the_declared_prefix() {
        let kv_store = FullAttnKv::without_prefix_cache(1, 1_000, None);
        let resolved_prefill = kv_store.preview_prefill_context(
            0,
            20,
            SessionInput::Session {
                session_id: 7,
                session_start_time: Time::ZERO,
                declared_prefix_tokens: 100,
            },
        );
        assert_eq!(resolved_prefill.resident_prefix_tokens(), 0);
        assert_eq!(resolved_prefill.prefill_tokens_to_compute(), 120);
        assert_eq!(resolved_prefill.post_prefill_context_tokens(), 120);
    }

    #[test]
    fn retained_prefix_partition_prefers_the_largest_hit_then_lower_partition() {
        let mut kv_store = FullAttnKv::with_prefix_cache(
            3,
            100,
            capped_prefix_cache(100, PrefixCachePolicy::Lru),
            None,
            None,
        );
        kv_store.prefix_caches[1].insert(7, 50, 100, None);
        kv_store.prefix_caches[2].insert(7, 80, 100, None);

        assert_eq!(
            kv_store.retained_prefix_partition(SessionInput::Session {
                session_id: 7,
                session_start_time: Time::ZERO,
                declared_prefix_tokens: 60,
            }),
            Some(2)
        );
        assert_eq!(
            kv_store.retained_prefix_partition(SessionInput::Session {
                session_id: 7,
                session_start_time: Time::ZERO,
                declared_prefix_tokens: 40,
            }),
            Some(1),
            "equal reusable lengths keep the lowest partition deterministic"
        );
        assert_eq!(
            kv_store.retained_prefix_partition(SessionInput::Standalone),
            None
        );
    }

    #[test]
    fn active_reservation_evicts_prefix_cache_within_one_total_capacity() {
        let mut kv_store = FullAttnKv::with_prefix_cache(
            1,
            100,
            capped_prefix_cache(100, PrefixCachePolicy::Lru),
            None,
            None,
        );
        kv_store.prefix_caches[0].insert(1, 80, 100, None);
        assert_eq!(kv_store.prefix_caches[0].used_tokens(), 80);

        let footprint = kv_store.footprint(RequestId(0), 60, 30);
        assert!(kv_store.fits(0, &footprint));
        kv_store.reserve(RequestId(0), 0, footprint, Time::ZERO);

        assert_eq!(kv_store.ledger.partition_promised(0), 90);
        assert_eq!(kv_store.prefix_caches[0].used_tokens(), 0);
        assert!(kv_store.non_cache_peak(0) + kv_store.prefix_caches[0].used_tokens() <= 100);
    }

    #[test]
    fn direct_reservation_logs_active_kv_pressure_eviction() {
        let log_directory = tempdir().unwrap();
        let prefix_cache_logger =
            PrefixCacheLogger::open(log_directory.path(), "main", crate::common::WorkerId(0))
                .unwrap();
        let mut kv_store = FullAttnKv::with_prefix_cache(
            1,
            100,
            capped_prefix_cache(100, PrefixCachePolicy::Lru),
            None,
            Some(prefix_cache_logger),
        );
        let session_input = SessionInput::Session {
            session_id: 7,
            session_start_time: Time::ZERO,
            declared_prefix_tokens: 0,
        };
        let resolved_prefill = kv_store.preview_prefill_context(0, 80, session_input);
        let session_footprint = kv_store.footprint(RequestId(0), 80, 0);
        kv_store.reserve_prefill_context(
            RequestId(0),
            0,
            resolved_prefill,
            session_footprint,
            Time::ZERO,
        );
        kv_store.drain_ready();
        kv_store.clear_prefill_admits(0);
        kv_store.release_retaining_prefix(RequestId(0), 0, Time::from_ms(1.0));

        let active_footprint = kv_store.footprint(RequestId(1), 60, 30);
        assert!(kv_store.fits(0, &active_footprint));
        kv_store.reserve(RequestId(1), 0, active_footprint, Time::from_ms(2.0));
        drop(kv_store);

        let path = log_directory
            .path()
            .join("raw/prefix_cache_event/worker_main_0.parquet");
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 3);

        let operations = batch
            .column_by_name("operation")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let reasons = batch
            .column_by_name("reason")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            (0..3).map(|row| operations.value(row)).collect::<Vec<_>>(),
            ["activate", "retain", "evict"]
        );
        assert_eq!(
            (0..3).map(|row| reasons.value(row)).collect::<Vec<_>>(),
            ["miss", "request-complete", "active-kv-pressure"]
        );

        let cache_used_before = batch
            .column_by_name("cache_used_before")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let cache_used_after = batch
            .column_by_name("cache_used_after")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(cache_used_before.values(), &[0, 0, 80]);
        assert_eq!(cache_used_after.values(), &[0, 80, 0]);
    }

    #[test]
    fn prefix_hit_moves_cache_ownership_to_the_request_then_returns_on_release() {
        let mut kv_store = FullAttnKv::with_prefix_cache(
            1,
            100,
            capped_prefix_cache(100, PrefixCachePolicy::Lru),
            None,
            None,
        );
        kv_store.prefix_caches[0].insert(7, 60, 100, None);
        let resolved_prefill = kv_store.preview_prefill_context(
            0,
            10,
            SessionInput::Session {
                session_id: 7,
                session_start_time: Time::ZERO,
                declared_prefix_tokens: 50,
            },
        );
        assert_eq!(resolved_prefill.resident_prefix_tokens(), 50);
        assert_eq!(resolved_prefill.prefill_tokens_to_compute(), 10);
        assert_eq!(resolved_prefill.post_prefill_context_tokens(), 60);

        let footprint = kv_store.footprint(RequestId(0), 60, 10);
        assert!(kv_store.fits(0, &footprint));
        kv_store.reserve_prefill_context(RequestId(0), 0, resolved_prefill, footprint, Time::ZERO);
        assert_eq!(kv_store.prefix_caches[0].used_tokens(), 0);

        kv_store.drain_ready();
        kv_store.release_retaining_prefix(RequestId(0), 0, Time::from_ms(1.0));
        assert_eq!(kv_store.prefix_caches[0].peek(7, 60), 60);
        assert_eq!(kv_store.prefix_caches[0].used_tokens(), 60);
    }
}
