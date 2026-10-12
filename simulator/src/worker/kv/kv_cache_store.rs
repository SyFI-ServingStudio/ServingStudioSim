//! The KV store of every composed L5 worker.
//!
//! `KvCacheStore` owns partition placement, promised/held ledgers, and occupancy
//! sampling. Each partition's resident decode membership and capacity accounting
//! live in a [`ResidentPartitionState`]. Iteration input remains owned by
//! Execution; borrowed iterators avoid per-iteration membership copies.
//!
//! Every charge is in units of the partition's [`KvAccounting`], resolved from
//! the arch's `KvCacheLayout`; the store never sees layout parts. A layout of
//! one per-token part counts plain tokens.
//!
//! ## Two tiers, one capacity
//!
//! | tier | contents | evictable | enters [`KvStore::fits`] |
//! |---|---|---|---|
//! | hard | resident, promised and held KV, including per-request state and sliding windows | no | yes |
//! | soft | retained prefix entries; checkpoints live decodes have passed | yes | no |
//!
//! The soft tier is clamped to whatever capacity the hard tier leaves. Live
//! checkpoints (window tails and states a running request keeps at the resume
//! positions it has passed) take precedence over retained prefixes, since they
//! are about to become one. Neither ever counts toward an admission decision.
//!
//! ## Not modeled
//!
//! Live checkpoints yield to admission but are never *proactively* shed: there
//! is no boundary-protection policy picking which one to drop first. vLLM's own
//! work in that direction is <https://github.com/vllm-project/vllm/pull/43447>
//! and Marconi (<https://arxiv.org/abs/2411.19379>).

use crate::common::{RequestId, SessionInput, Time};
use crate::log::{
    KvSampler, KvSubmit, PrefixCacheActivation, PrefixCacheEvent, PrefixCacheEventKind,
    PrefixCacheEvictionReason, PrefixCacheLogger, PrefixCacheRetentionReason,
};
use crate::worker::kv::shared::kv_accounting::SharedKvAccounting;
use crate::worker::kv::shared::prefix_cache::{
    PrefixCache, PrefixCacheBudget, PrefixCacheMutation, PrefixCacheReturnMetadata,
};
use crate::worker::kv::shared::prefix_cache_journal::PrefixCacheJournal;
use crate::worker::kv::shared::request_ledger::RequestLedger;
use crate::worker::kv::shared::resident_partition::ResidentPartitionState;
use crate::worker::kv::{
    ChunkedPrefillKv, HandoffKv, IterWorkerKv, KvStore, PrefixKv, ResolvedPrefillContext,
    SlotPipelineKv,
};
use crate::worker::shared::advance_scope::{AdvanceScope, PartitionId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReservationBasis {
    FullFootprint,
    BoundedFuture,
}

/// Hard-tier units one admission reserves.
pub struct KvFootprint {
    reserved_units: u64,
    basis: ReservationBasis,
}

impl KvFootprint {
    #[inline]
    fn reserved_units(&self) -> u64 {
        self.reserved_units
    }
}

pub struct KvCacheStore {
    accounting: SharedKvAccounting,
    partitions: Vec<ResidentPartitionState>,
    ledger: RequestLedger,
    /// Evictable completed-session KV, local to each attention partition.
    prefix_caches: Vec<PrefixCache>,
    journal: PrefixCacheJournal,
    sampler: Option<KvSampler>,
}

impl KvStore for KvCacheStore {
    type Footprint = KvFootprint;

    fn num_partitions(&self) -> usize {
        KvCacheStore::num_partitions(self)
    }

    fn footprint(
        &self,
        request: RequestId,
        post_prefill_context_tokens: u32,
        remaining_output_tokens: u32,
    ) -> Self::Footprint {
        KvCacheStore::footprint(
            self,
            request,
            post_prefill_context_tokens,
            remaining_output_tokens,
        )
    }

    fn fits(&self, partition: PartitionId, footprint: &Self::Footprint) -> bool {
        KvCacheStore::fits(self, partition, footprint)
    }

    fn reserve(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        footprint: Self::Footprint,
        now: Time,
    ) {
        KvCacheStore::reserve(self, request, partition, footprint, now);
    }

    fn commit_resident(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        post_prefill_context_tokens: u64,
        remaining_output_tokens: u32,
    ) {
        KvCacheStore::commit_resident(
            self,
            request,
            partition,
            post_prefill_context_tokens,
            remaining_output_tokens,
        );
    }

    fn advance(&mut self, scope: AdvanceScope<'_>, steps: u32) {
        KvCacheStore::advance(self, scope, steps);
    }

    fn release(&mut self, request: RequestId, partition: PartitionId) {
        KvCacheStore::release(self, request, partition);
    }

    fn sample_submit(&mut self, partition: PartitionId, now: Time) {
        KvCacheStore::sample_submit(self, partition, now);
    }
}

impl IterWorkerKv for KvCacheStore {
    fn drain_ready(&mut self) {
        KvCacheStore::drain_ready(self);
    }

    fn clear_prefill_admits(&mut self, partition: PartitionId) {
        KvCacheStore::clear_prefill_admits(self, partition);
    }

    fn has_live_decode(&self, partition: PartitionId) -> bool {
        KvCacheStore::has_live_decode(self, partition)
    }

    fn live_decode_count(&self, partition: PartitionId) -> u32 {
        KvCacheStore::live_decode_count(self, partition)
    }

    fn has_prefill_admit(&self, partition: PartitionId) -> bool {
        KvCacheStore::has_prefill_admit(self, partition)
    }

    fn status_active(&self, partition: PartitionId) -> u32 {
        KvCacheStore::status_active(self, partition)
    }

    fn kv_usage(&self, partition: PartitionId) -> f64 {
        let partition_state = &self.partitions[partition as usize];
        let used = partition_state.resident_units()
            + self.ledger.partition_held(partition)
            + self.ledger.partition_promised(partition);
        used as f64 / partition_state.capacity_units().max(1) as f64
    }

    fn release_external(&mut self, request: RequestId, current_kv: u64) -> Option<PartitionId> {
        KvCacheStore::release_external(self, request, current_kv)
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

impl HandoffKv for KvCacheStore {
    fn hold(&mut self, partition: PartitionId, request: RequestId, kv_tokens: u64) {
        KvCacheStore::hold(self, partition, request, kv_tokens);
    }

    fn complete_handoff(&mut self, request: RequestId, now: Time) {
        KvCacheStore::complete_handoff(self, request, now);
    }
}

impl ChunkedPrefillKv for KvCacheStore {
    fn bounded_future_footprint(
        &self,
        _request: RequestId,
        post_prefill_context_tokens: u32,
        remaining_output_tokens: u32,
        max_future_tokens: u32,
        page_size: u32,
    ) -> Self::Footprint {
        let future_tokens = u64::from(remaining_output_tokens.min(max_future_tokens))
            .checked_add(u64::from(page_size))
            .expect("bounded-future KV footprint overflow");
        KvFootprint {
            reserved_units: self
                .accounting
                .footprint(u64::from(post_prefill_context_tokens), future_tokens),
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
        let protected_units = partition_state.resident_units()
            + self.ledger.partition_promised(partition)
            + self.ledger.partition_held(partition)
            + footprint.reserved_units();
        let mut running_future =
            partition_state.bounded_future_decode_units(max_future_tokens, new_token_ratio);
        if decode_step_tokens > 0 {
            // Bounded-future admission runs on one-token pages only.
            let next_step = partition_state.next_decode_allocation_units(1, decode_step_tokens);
            running_future = running_future.max(next_step as f64);
        }
        protected_units as f64 + running_future < partition_state.capacity_units() as f64
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
        let required = partition_state.next_decode_allocation_units(page_size, step_tokens);
        if required == 0 {
            return 0;
        }
        let trigger = partition_state
            .first_decode_request()
            .expect("positive decode allocation requires a live request");
        let protected = partition_state.resident_units()
            + self.ledger.partition_promised(partition)
            + self.ledger.partition_held(partition);
        let capacity = partition_state.capacity_units();
        let slack_after_step = capacity.saturating_sub(protected.saturating_add(required));
        // Live checkpoints keep their precedence over retained prefixes but,
        // being soft, never count toward the shortfall.
        let live_checkpoints = partition_state
            .live_checkpoint_units()
            .min(slack_after_step);
        let cache_limit = slack_after_step - live_checkpoints;
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
        let cache_units = self.prefix_caches[partition as usize].used_units();
        let available = capacity.saturating_sub(protected.saturating_add(cache_units));
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

impl PrefixKv for KvCacheStore {
    fn retained_prefix_partition(&self, session_input: SessionInput) -> Option<PartitionId> {
        let (session_id, declared_prefix_tokens) = match session_input {
            // A pinned prefix is resident on every partition, so it creates no
            // placement affinity.
            SessionInput::Standalone | SessionInput::PinnedPrefix { .. } => return None,
            SessionInput::Session {
                session_id,
                declared_prefix_tokens,
                ..
            } => (session_id, declared_prefix_tokens),
        };

        let mut best_partition = None;
        let mut best_resident_tokens = 0;
        for (partition_index, prefix_cache) in self.prefix_caches.iter().enumerate() {
            let resident_tokens = prefix_cache.peek(session_id, declared_prefix_tokens);
            if resident_tokens > best_resident_tokens {
                best_resident_tokens = resident_tokens;
                best_partition = Some(
                    PartitionId::try_from(partition_index)
                        .expect("KvCacheStore partition count must fit PartitionId"),
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
            // Declared resident on arrival: a hit regardless of this
            // partition's cache. Admission reserves it with the request.
            SessionInput::PinnedPrefix { prefix_tokens } => prefix_tokens,
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
            // `ResolvedPrefillContext::new` borrows one token back from a fully
            // resident prefix so that a zero-length prefill never reaches the
            // model. The entry's position still sits at the boundary, so the
            // resume point is off by that one borrowed token.
            debug_assert!(
                take_result
                    .hit_tokens
                    .abs_diff(resolved_prefill.resident_prefix_tokens())
                    <= 1,
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
                full_hit_tokens: u64::from(take_result.full_hit_tokens),
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
        KvCacheStore::release_retaining_prefix(self, request, partition, now);
    }
}

impl SlotPipelineKv for KvCacheStore {
    fn current_kv(&self, partition: PartitionId, request: RequestId) -> Option<u64> {
        self.partitions[partition as usize].decode_current_kv(request)
    }

    fn estimated_peak(&self, partition: PartitionId) -> u64 {
        self.partitions[partition as usize].projected_peak_units()
            + self.ledger.partition_promised(partition)
            + self.ledger.partition_held(partition)
            + self.prefix_caches[partition as usize].used_units()
    }

    fn has_reservation(&self, request: RequestId) -> bool {
        self.ledger.has_promise(request)
    }

    fn request_kv_weight(&self, request: RequestId) -> u64 {
        self.ledger
            .promised_charge(request)
            .or_else(|| {
                self.ledger.placement(request).and_then(|partition| {
                    self.partitions[partition as usize].decode_charge(request)
                })
            })
            .unwrap_or(0)
    }
}

impl KvCacheStore {
    pub(crate) fn without_prefix_cache(
        num_partitions: usize,
        accounting: SharedKvAccounting,
        sampler: Option<KvSampler>,
    ) -> Self {
        Self::new(
            num_partitions,
            accounting,
            PrefixCacheBudget::DISABLED,
            sampler,
            None,
        )
    }

    pub(crate) fn new(
        num_partitions: usize,
        accounting: SharedKvAccounting,
        prefix_cache: PrefixCacheBudget,
        sampler: Option<KvSampler>,
        prefix_cache_logger: Option<PrefixCacheLogger>,
    ) -> Self {
        Self {
            partitions: (0..num_partitions)
                .map(|_| ResidentPartitionState::new(accounting.clone()))
                .collect(),
            ledger: RequestLedger::new(num_partitions),
            prefix_caches: (0..num_partitions)
                .map(|_| PrefixCache::new(accounting.clone(), prefix_cache))
                .collect(),
            journal: PrefixCacheJournal::new(prefix_cache_logger),
            sampler,
            accounting,
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
    ) -> KvFootprint {
        KvFootprint {
            reserved_units: self.accounting.footprint(
                u64::from(post_prefill_context_tokens),
                u64::from(remaining_output_tokens),
            ),
            basis: ReservationBasis::FullFootprint,
        }
    }

    #[inline]
    pub(crate) fn fits(&self, partition: PartitionId, footprint: &KvFootprint) -> bool {
        let partition_state = &self.partitions[partition as usize];
        let reserved_units = self.ledger.partition_promised(partition)
            + self.ledger.partition_held(partition)
            + footprint.reserved_units();
        partition_state.resident_units() + reserved_units <= partition_state.capacity_units()
            && partition_state.projected_peak_units() + reserved_units
                <= partition_state.capacity_units()
    }

    pub(crate) fn reserve(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        footprint: KvFootprint,
        now: Time,
    ) {
        self.ledger
            .promise(request, partition, footprint.reserved_units());
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
        // The prompt places the request's resume positions. A pulled decode
        // has no prefill context here: its whole context was its prompt.
        let prompt_tokens = self
            .ledger
            .prefill_context(request)
            .map_or(post_prefill_context_tokens, |context| {
                u64::from(context.post_prefill_context_tokens())
            });
        self.partitions[partition as usize].begin_decode(
            request,
            post_prefill_context_tokens,
            remaining_output_tokens,
            prompt_tokens,
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
            state.resident_units() <= state.capacity_units(),
            "decode advance overfilled KV partition {partition}: {} of {} units",
            state.resident_units(),
            state.capacity_units()
        );
    }

    pub(crate) fn clear_prefill_admits(&mut self, partition: PartitionId) {
        self.partitions[partition as usize].clear_prefill_admits();
    }

    pub(crate) fn release(&mut self, request: RequestId, partition: PartitionId) {
        self.ledger.take_held(request);
        self.ledger.forget_promise(request);
        self.ledger.forget_chunked_prefill(request);
        self.partitions[partition as usize].release_decode(request);
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
        let prompt_tokens =
            resolved_prefill.map_or(0, |value| u64::from(value.post_prefill_context_tokens()));
        let retained_tokens = self.partitions[partition as usize]
            .decode_current_kv(request)
            .unwrap_or(prompt_tokens);
        let session_id = resolved_prefill.and_then(ResolvedPrefillContext::session_id);
        let cache_return_metadata =
            resolved_prefill.and_then(ResolvedPrefillContext::cache_return_metadata);
        self.release(request, partition);
        if let Some(session_id) = session_id {
            self.retain_prefix(
                request,
                partition,
                session_id,
                prompt_tokens,
                retained_tokens,
                cache_return_metadata,
                now,
                PrefixCacheRetentionReason::RequestComplete,
            );
        }
    }

    /// Cancellation keeps the `swap_remove` behavior for a request still in
    /// `prefill_admits`. A resident decode releases the charge it holds, so
    /// the caller's `current_kv` is not needed.
    pub(crate) fn release_external(
        &mut self,
        request: RequestId,
        _current_kv: u64,
    ) -> Option<PartitionId> {
        let Some(partition) = self.ledger.forget_placement(request) else {
            self.ledger.take_held(request);
            self.ledger.forget_chunked_prefill(request);
            self.ledger.take_prefill_context(request);
            return None;
        };
        let partition_state = &mut self.partitions[partition as usize];
        partition_state.release_decode(request);
        partition_state.remove_prefill_admit(request);
        self.ledger.forget_promise(request);
        self.ledger.forget_chunked_prefill(request);
        self.ledger.take_prefill_context(request);
        Some(partition)
    }

    pub(crate) fn hold(&mut self, partition: PartitionId, request: RequestId, kv_tokens: u64) {
        let units = self.accounting.held_while_running(kv_tokens, 1);
        self.ledger.hold(request, partition, units, kv_tokens);
    }

    pub(crate) fn complete_handoff(&mut self, request: RequestId, now: Time) {
        let Some((partition, kv_tokens)) = self.ledger.take_held(request) else {
            return;
        };
        let resolved_prefill = self.ledger.take_prefill_context(request);
        let session_id = resolved_prefill.and_then(ResolvedPrefillContext::session_id);
        let cache_return_metadata =
            resolved_prefill.and_then(ResolvedPrefillContext::cache_return_metadata);
        let prompt_tokens = resolved_prefill.map_or(kv_tokens, |value| {
            u64::from(value.post_prefill_context_tokens())
        });
        if let Some(session_id) = session_id {
            self.retain_prefix(
                request,
                partition,
                session_id,
                prompt_tokens,
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
        let retained_prefix_kv = self.prefix_caches[partition as usize].used_units();
        let live_checkpoints = self.clamped_live_checkpoint_units(partition);
        let held = self.ledger.partition_held(partition);
        let submit = KvSubmit {
            active_kv: self.partitions[partition as usize].resident_units()
                + held
                + retained_prefix_kv
                + live_checkpoints,
            retained_prefix_kv,
            projected_peak: self.partitions[partition as usize].projected_peak_units()
                + held
                + retained_prefix_kv
                + live_checkpoints,
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

    /// Hard-tier occupancy: whichever of the immediate and the projected total
    /// is larger, including everything reserved but not yet resident.
    fn non_cache_peak(&self, partition: PartitionId) -> u64 {
        let reserved =
            self.ledger.partition_promised(partition) + self.ledger.partition_held(partition);
        let partition_state = &self.partitions[partition as usize];
        (partition_state.resident_units() + reserved)
            .max(partition_state.projected_peak_units() + reserved)
    }

    /// Capacity the hard tier is not using: the whole soft budget.
    fn soft_slack(&self, partition: PartitionId) -> u64 {
        self.partitions[partition as usize]
            .capacity_units()
            .saturating_sub(self.non_cache_peak(partition))
    }

    /// Checkpoints live decodes have passed, clamped to the soft slack so they
    /// can never starve a resident request, only the retained-prefix tier.
    fn clamped_live_checkpoint_units(&self, partition: PartitionId) -> u64 {
        self.partitions[partition as usize]
            .live_checkpoint_units()
            .min(self.soft_slack(partition))
    }

    /// What the retained-prefix tier may occupy: the soft budget minus the
    /// live checkpoints, which are nearer-term useful and so win.
    fn prefix_cache_physical_limit(&self, partition: PartitionId) -> u64 {
        self.soft_slack(partition)
            .saturating_sub(self.clamped_live_checkpoint_units(partition))
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
        let protected = partition_state.resident_units()
            + self.ledger.partition_promised(partition)
            + self.ledger.partition_held(partition);
        let slack = partition_state.capacity_units().saturating_sub(protected);
        let physical_limit = slack - partition_state.live_checkpoint_units().min(slack);
        self.prefix_caches[partition as usize].shrink_to(physical_limit)
    }

    fn retain_prefix(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        session_id: u32,
        prompt_tokens: u64,
        tokens: u64,
        cache_return_metadata: Option<PrefixCacheReturnMetadata>,
        now: Time,
        retain_reason: PrefixCacheRetentionReason,
    ) {
        let physical_limit = self.prefix_cache_physical_limit(partition);
        let insert_result = self.prefix_caches[partition as usize].insert(
            session_id,
            prompt_tokens,
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
            let cache_used = self.prefix_caches[partition as usize].used_tokens();
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
                full_hit_tokens: 0,
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

    use crate::worker::kv::shared::kv_accounting::KvAccounting;
    use crate::worker::kv::PrefixCachePolicy;

    fn capped_prefix_cache(
        max_retained_units: u64,
        policy: PrefixCachePolicy,
    ) -> PrefixCacheBudget {
        PrefixCacheBudget {
            max_retained_units,
            policy,
        }
    }

    #[test]
    fn strict_capacity_gate_counts_promised_tokens() {
        let mut kv_store =
            KvCacheStore::without_prefix_cache(1, KvAccounting::tokens_for_test(100), None);
        let candidate = kv_store.footprint(RequestId(0), 60, 30);
        assert!(kv_store.fits(0, &candidate));

        let reservation = kv_store.footprint(RequestId(1), 10, 10);
        kv_store.reserve(RequestId(1), 0, reservation, Time::ZERO);
        assert!(!kv_store.fits(0, &candidate));
    }

    #[test]
    fn disabled_prefix_cache_recomputes_the_declared_prefix() {
        let kv_store =
            KvCacheStore::without_prefix_cache(1, KvAccounting::tokens_for_test(1_000), None);
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
        let mut kv_store = KvCacheStore::new(
            3,
            KvAccounting::tokens_for_test(100),
            capped_prefix_cache(100, PrefixCachePolicy::Lru),
            None,
            None,
        );
        kv_store.prefix_caches[1].insert(7, 50, 50, 100, None);
        kv_store.prefix_caches[2].insert(7, 80, 80, 100, None);

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
        let mut kv_store = KvCacheStore::new(
            1,
            KvAccounting::tokens_for_test(100),
            capped_prefix_cache(100, PrefixCachePolicy::Lru),
            None,
            None,
        );
        kv_store.prefix_caches[0].insert(1, 80, 80, 100, None);
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
        let mut kv_store = KvCacheStore::new(
            1,
            KvAccounting::tokens_for_test(100),
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
    fn pinned_prefix_is_resident_whatever_the_cache_holds() {
        let pinned = SessionInput::PinnedPrefix { prefix_tokens: 300 };
        let disabled =
            KvCacheStore::without_prefix_cache(2, KvAccounting::tokens_for_test(1_000), None);
        let mut warm = KvCacheStore::new(
            2,
            KvAccounting::tokens_for_test(1_000),
            capped_prefix_cache(1_000, PrefixCachePolicy::Lru),
            None,
            None,
        );
        warm.prefix_caches[1].insert(7, 500, 500, 1_000, None);

        for kv_store in [&disabled, &warm] {
            assert_eq!(
                kv_store.retained_prefix_partition(pinned),
                None,
                "a pinned prefix is no placement affinity"
            );
            assert_eq!(kv_store.resident_prefix_tokens(40, pinned), 300);
            for partition in 0..2 {
                let resolved_prefill = kv_store.preview_prefill_context(partition, 40, pinned);
                assert_eq!(resolved_prefill.declared_prefix_tokens(), 300);
                assert_eq!(resolved_prefill.resident_prefix_tokens(), 300);
                assert_eq!(resolved_prefill.prefill_tokens_to_compute(), 40);
                assert_eq!(resolved_prefill.post_prefill_context_tokens(), 340);
                assert_eq!(resolved_prefill.active_chunk(), (300, 40));
            }
        }
    }

    #[test]
    fn pinned_prefix_is_reserved_with_the_request_and_released_without_retention() {
        let log_directory = tempdir().unwrap();
        let prefix_cache_logger =
            PrefixCacheLogger::open(log_directory.path(), "main", crate::common::WorkerId(0))
                .unwrap();
        let mut kv_store = KvCacheStore::new(
            1,
            KvAccounting::tokens_for_test(400),
            capped_prefix_cache(400, PrefixCachePolicy::Lru),
            None,
            Some(prefix_cache_logger),
        );
        kv_store.prefix_caches[0].insert(7, 100, 100, 400, None);
        let pinned = SessionInput::PinnedPrefix { prefix_tokens: 300 };
        let resolved_prefill = kv_store.preview_prefill_context(0, 40, pinned);

        // The prefix counts toward capacity: 300 + 40 + 70 > 400.
        let too_large = kv_store.footprint(RequestId(0), 340, 70);
        assert!(!kv_store.fits(0, &too_large));
        let footprint = kv_store.footprint(
            RequestId(0),
            resolved_prefill.post_prefill_context_tokens(),
            10,
        );
        assert!(kv_store.fits(0, &footprint));
        kv_store.reserve_prefill_context(RequestId(0), 0, resolved_prefill, footprint, Time::ZERO);
        assert_eq!(kv_store.ledger.partition_promised(0), 350);
        assert_eq!(
            kv_store.prefix_caches[0].used_tokens(),
            0,
            "the pinned reservation evicts retained entries like any active KV"
        );

        kv_store.drain_ready();
        kv_store.clear_prefill_admits(0);
        kv_store.commit_resident(RequestId(0), 0, 340, 10);
        assert_eq!(kv_store.partitions[0].resident_units(), 340);
        kv_store.release_retaining_prefix(RequestId(0), 0, Time::from_ms(1.0));
        assert_eq!(kv_store.partitions[0].resident_units(), 0);
        assert_eq!(
            kv_store.prefix_caches[0].used_tokens(),
            0,
            "a pinned prefix is never returned to the shared cache"
        );
        drop(kv_store);

        // Only the eviction made room for the request; a pinned prefix neither
        // activates nor retains a cache entry.
        let path = log_directory
            .path()
            .join("raw/prefix_cache_event/worker_main_0.parquet");
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        let operations = batch
            .column_by_name("operation")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            (0..batch.num_rows())
                .map(|row| operations.value(row))
                .collect::<Vec<_>>(),
            ["evict"]
        );
    }

    #[test]
    fn prefix_hit_moves_cache_ownership_to_the_request_then_returns_on_release() {
        let mut kv_store = KvCacheStore::new(
            1,
            KvAccounting::tokens_for_test(100),
            capped_prefix_cache(100, PrefixCachePolicy::Lru),
            None,
            None,
        );
        kv_store.prefix_caches[0].insert(7, 60, 60, 100, None);
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

    // ── per-request state ───────────────────────────────────────────────────

    use std::sync::Arc;

    use crate::arch::kv_layout::{EngineOverhead, KvCacheLayout, KvCachePart};
    use crate::worker::kv::shared::kv_accounting::{KvAccountingOptions, PrefixResumePoints};
    use crate::worker::shared::advance_scope::AdvanceScope;

    /// Qwen3.6-35B-A3B TP1: 30 GDN layers of state over a 10-GQA-layer token
    /// page, and the vLLM block alignment that follows from it.
    const STATE_TOKENS: u64 = 3_168;
    const CHECKPOINT_INTERVAL: u32 = 2_048;

    /// One unit per token plus a `state`-unit recurrent state, on an engine
    /// block of `block` tokens when given.
    fn state_store(
        capacity: u64,
        state: u64,
        block: Option<u32>,
        resume_points: Option<PrefixResumePoints>,
    ) -> KvCacheStore {
        let mut layout = KvCacheLayout::new().with_part(KvCachePart::per_token("gqa", 1));
        if state > 0 {
            layout = layout.with_part(KvCachePart::per_request("gdn", state));
        }
        let layout = layout.with_engine(EngineOverhead {
            block_tokens: block,
            fixed_device_bytes: 0,
        });
        let accounting = KvAccounting::resolve(
            &layout,
            KvAccountingOptions {
                resume_points,
                ..KvAccountingOptions::default()
            },
            capacity,
        )
        .unwrap();
        KvCacheStore::new(
            1,
            Arc::new(accounting),
            capped_prefix_cache(capacity, PrefixCachePolicy::Lru),
            None,
            None,
        )
    }

    fn unblocked(capacity: u64, state: u64) -> KvCacheStore {
        state_store(capacity, state, None, None)
    }

    #[test]
    fn admission_reserves_context_future_decode_and_state_in_one_check() {
        let kv_store = unblocked(1_000, 100);
        // 800 context + 100 decode + 100 state == capacity, to the unit.
        assert!(kv_store.fits(0, &kv_store.footprint(RequestId(0), 800, 100)));
        assert!(
            !kv_store.fits(0, &kv_store.footprint(RequestId(0), 801, 100)),
            "one token over capacity must be refused",
        );
        assert!(
            !kv_store.fits(0, &kv_store.footprint(RequestId(0), 800, 101)),
            "future decode tokens are part of the same atomic reservation",
        );
        let stateless = unblocked(1_000, 0);
        assert!(
            stateless.fits(0, &stateless.footprint(RequestId(0), 900, 100)),
            "the same request fits only once the recurrent state is priced out",
        );
    }

    #[test]
    fn a_promised_state_blocks_the_next_request_before_it_is_resident() {
        let mut kv_store = unblocked(1_000, 100);
        let first = kv_store.footprint(RequestId(0), 400, 0);
        kv_store.reserve(RequestId(0), 0, first, Time::ZERO);
        // 500 promised (400 + 100 state) leaves 500, so 400 context + 100 state fits...
        assert!(kv_store.fits(0, &kv_store.footprint(RequestId(1), 400, 0)));
        // ...but 401 does not, proving the state rode along in the promise.
        assert!(!kv_store.fits(0, &kv_store.footprint(RequestId(1), 401, 0)));
    }

    #[test]
    fn state_is_charged_once_at_commit_never_advanced_and_fully_released() {
        let mut kv_store = unblocked(100_000, STATE_TOKENS);
        let footprint = kv_store.footprint(RequestId(0), 1_000, 20);
        kv_store.reserve(RequestId(0), 0, footprint, Time::ZERO);
        kv_store.drain_ready();
        kv_store.commit_resident(RequestId(0), 0, 1_000, 20);
        assert_eq!(
            kv_store.partitions[0].resident_units(),
            1_000 + STATE_TOKENS
        );

        kv_store.advance(AdvanceScope::WholePartition(0), 10);
        assert_eq!(
            kv_store.partitions[0].resident_units(),
            1_010 + STATE_TOKENS,
            "decode grows context only",
        );
        assert_eq!(
            kv_store.partitions[0].decode_members().collect::<Vec<_>>(),
            [(RequestId(0), 1_010)],
            "decode_kv_lens must stay pure context",
        );

        kv_store.release(RequestId(0), 0);
        assert_eq!(kv_store.partitions[0].resident_units(), 0);
        assert_eq!(kv_store.ledger.partition_promised(0), 0);
    }

    #[test]
    fn a_chunked_prefill_keeps_its_state_promised_until_the_last_chunk_commits_it() {
        let mut kv_store = unblocked(100_000, STATE_TOKENS);
        let resolved = kv_store.preview_prefill_context(0, 5_000, SessionInput::Standalone);
        let footprint = kv_store.footprint(RequestId(0), 5_000, 20);
        kv_store.reserve_chunked_prefill_context(RequestId(0), 0, resolved, footprint, Time::ZERO);
        kv_store.drain_ready();
        let promised = 5_000 + 20 + STATE_TOKENS;

        for chunk in [2_048, 2_048] {
            kv_store.schedule_prefill_chunk(RequestId(0), 0, chunk);
            assert_eq!(kv_store.ledger.partition_promised(0), promised);
            assert_eq!(kv_store.partitions[0].resident_units(), 0);
            kv_store.complete_prefill_chunk(RequestId(0));
            kv_store.clear_prefill_admits(0);
        }
        kv_store.schedule_prefill_chunk(RequestId(0), 0, 904);
        kv_store.complete_prefill_chunk(RequestId(0));
        kv_store.finish_chunked_prefill(RequestId(0), 0, 20);
        assert_eq!(kv_store.ledger.partition_promised(0), 0);
        assert_eq!(
            kv_store.partitions[0].resident_units(),
            5_000 + STATE_TOKENS,
            "state becomes resident exactly once"
        );

        kv_store.release(RequestId(0), 0);
        assert_eq!(kv_store.partitions[0].resident_units(), 0);
    }

    #[test]
    fn a_request_cancelled_mid_prefill_returns_its_whole_reservation() {
        let mut kv_store = unblocked(100_000, STATE_TOKENS);
        let resolved = kv_store.preview_prefill_context(0, 5_000, SessionInput::Standalone);
        let footprint = kv_store.footprint(RequestId(0), 5_000, 20);
        kv_store.reserve_chunked_prefill_context(RequestId(0), 0, resolved, footprint, Time::ZERO);
        kv_store.schedule_prefill_chunk(RequestId(0), 0, 2_048);
        kv_store.complete_prefill_chunk(RequestId(0));

        kv_store.release(RequestId(0), 0);
        assert_eq!(kv_store.ledger.partition_promised(0), 0);
        kv_store.clear_prefill_admits(0);
        assert_eq!(kv_store.status_active(0), 0);
    }

    #[test]
    fn a_bounded_future_footprint_still_carries_the_whole_state() {
        let kv_store = unblocked(100_000, STATE_TOKENS);
        let footprint = kv_store.bounded_future_footprint(RequestId(0), 1_000, 500, 64, 1);
        assert_eq!(footprint.reserved_units(), 1_000 + 64 + 1 + STATE_TOKENS);
        assert!(kv_store.fits_bounded_future(0, &footprint, 64, 0.7, 1));
        let tight = unblocked(1_000 + 65 + STATE_TOKENS, STATE_TOKENS);
        assert!(!tight.fits_bounded_future(0, &footprint, 64, 0.7, 1));
    }

    #[test]
    fn a_pinned_prefix_is_resident_unaligned_and_never_retained() {
        let mut kv_store = state_store(100_000, STATE_TOKENS, Some(CHECKPOINT_INTERVAL), None);
        // Not a block multiple: the trace declares this prefix resident, so it
        // is not rounded down to a resume position.
        let pinned = SessionInput::PinnedPrefix {
            prefix_tokens: CHECKPOINT_INTERVAL + 7,
        };
        assert_eq!(kv_store.retained_prefix_partition(pinned), None);
        let resolved = kv_store.preview_prefill_context(0, 5_000, pinned);
        assert_eq!(resolved.resident_prefix_tokens(), CHECKPOINT_INTERVAL + 7);
        let context = resolved.post_prefill_context_tokens();
        let footprint = kv_store.footprint(RequestId(0), context, 1);
        kv_store.reserve_chunked_prefill_context(RequestId(0), 0, resolved, footprint, Time::ZERO);
        kv_store.drain_ready();
        kv_store.schedule_prefill_chunk(RequestId(0), 0, 5_000);
        kv_store.complete_prefill_chunk(RequestId(0));
        kv_store.finish_chunked_prefill(RequestId(0), 0, 0);
        kv_store.release_retaining_prefix(RequestId(0), 0, Time::from_ms(1.0));
        assert_eq!(kv_store.partitions[0].resident_units(), 0);
        assert_eq!(kv_store.prefix_caches[0].used_units(), 0);
    }

    // ── resume positions ────────────────────────────────────────────────────

    /// Reproduces the hit rates reported for a hybrid Qwen3 model in
    /// <https://github.com/vllm-project/vllm/issues/40696>, whose aligned block
    /// size is 528 tokens. vLLM resumes a resent prompt at its last whole
    /// block before the final token: a prompt shorter than one block gets
    /// nothing, longer prompts reuse only whole blocks.
    #[test]
    fn prompt_end_reuse_reproduces_the_measured_vllm_hit_rates() {
        const BLOCK: u32 = 528;
        let expected = [
            (479u32, 0u32, 0.0f64),
            (552, 528, 95.4),
            (597, 528, 88.2),
            (979, 528, 53.7),
        ];
        for (prompt_tokens, expected_hit, measured_percent) in expected {
            let mut kv_store = state_store(10_000_000, STATE_TOKENS, Some(BLOCK), None);
            let prompt = u64::from(prompt_tokens);
            kv_store.prefix_caches[0].insert(7, prompt, prompt, 10_000_000, None);
            let hit = kv_store.prefix_caches[0].peek(7, prompt_tokens);
            assert_eq!(
                hit, expected_hit,
                "prompt {prompt_tokens}: block-aligned reuse"
            );
            let modeled_percent = 100.0 * f64::from(hit) / f64::from(prompt_tokens);
            assert!(
                (modeled_percent - measured_percent).abs() < 0.5,
                "prompt {prompt_tokens}: modeled {modeled_percent:.1}% vs measured {measured_percent}%",
            );
        }
    }

    #[test]
    fn a_retained_prefix_pays_for_the_states_at_its_positions() {
        let session = SessionInput::Session {
            session_id: 7,
            session_start_time: Time::ZERO,
            declared_prefix_tokens: 0,
        };
        let finish = |resume_points| {
            let mut kv_store = state_store(
                10_000_000,
                STATE_TOKENS,
                Some(CHECKPOINT_INTERVAL),
                resume_points,
            );
            let resolved_prefill = kv_store.preview_prefill_context(0, 5_000, session);
            let footprint = kv_store.footprint(RequestId(0), 5_000, 0);
            kv_store.reserve_prefill_context(
                RequestId(0),
                0,
                resolved_prefill,
                footprint,
                Time::ZERO,
            );
            kv_store.drain_ready();
            kv_store.commit_resident(RequestId(0), 0, 5_000, 0);
            kv_store.release_retaining_prefix(RequestId(0), 0, Time::from_ms(1.0));
            kv_store
        };

        // vLLM's default keeps one state, at the prompt's last whole block.
        let prompt_end = finish(None);
        assert_eq!(prompt_end.prefix_caches[0].used_tokens(), 2 * 2_048);
        assert_eq!(
            prompt_end.prefix_caches[0].used_units(),
            2 * 2_048 + STATE_TOKENS
        );
        assert_eq!(prompt_end.prefix_caches[0].peek(7, 5_000), 2 * 2_048);
        assert_eq!(
            prompt_end.prefix_caches[0].peek(7, 4_095),
            0,
            "a later state cannot serve an earlier resume point",
        );

        // Every block keeps a state at 2048 and 4096; the 904-token tail has
        // none and is not retained.
        let every_block = finish(Some(PrefixResumePoints::EveryBlock));
        assert_eq!(
            every_block.prefix_caches[0].used_units(),
            2 * (2_048 + STATE_TOKENS)
        );
        assert_eq!(every_block.prefix_caches[0].peek(7, 4_095), 2_048);
    }

    /// The reusable length is bounded by the attention KV **and** by the last
    /// kept state. One entry carries both at its positions: they are sized,
    /// charged and evicted together, so they cannot drift apart.
    #[test]
    fn attention_kv_and_states_are_retained_at_one_position() {
        let interval = u64::from(CHECKPOINT_INTERVAL);
        // Room for exactly two positions' worth of both halves, plus change.
        let capacity = 2 * (interval + STATE_TOKENS) + 500;
        let mut kv_store = state_store(
            capacity,
            STATE_TOKENS,
            Some(CHECKPOINT_INTERVAL),
            Some(PrefixResumePoints::EveryBlock),
        );

        // Ask to retain five blocks; only two fit, and the entry is cut at a
        // position rather than at whatever the budget allowed.
        let insert_result =
            kv_store.prefix_caches[0].insert(7, 5 * interval, 5 * interval, capacity, None);
        assert_eq!(insert_result.retained_tokens, 2 * interval);
        assert_eq!(
            kv_store.prefix_caches[0].used_units(),
            2 * (interval + STATE_TOKENS),
            "both halves of every kept position are paid for",
        );

        // Eviction takes both halves at once: no orphaned KV, no orphaned state.
        kv_store.prefix_caches[0].shrink_to(0);
        assert_eq!(kv_store.prefix_caches[0].used_tokens(), 0);
        assert_eq!(kv_store.prefix_caches[0].used_units(), 0);
        assert_eq!(kv_store.prefix_caches[0].peek(7, 5 * interval as u32), 0);
    }

    #[test]
    fn a_prefix_shorter_than_one_block_is_not_retained() {
        let mut kv_store = state_store(10_000_000, STATE_TOKENS, Some(CHECKPOINT_INTERVAL), None);
        let insert_result = kv_store.prefix_caches[0].insert(7, 2_047, 2_047, 10_000_000, None);
        assert_eq!(insert_result.retained_tokens, 0);
        assert_eq!(kv_store.prefix_caches[0].used_units(), 0);
    }

    #[test]
    fn the_soft_tier_yields_entirely_when_the_hard_tier_fills_the_partition() {
        let capacity = 40_000;
        let mut kv_store = state_store(capacity, STATE_TOKENS, Some(CHECKPOINT_INTERVAL), None);
        kv_store.prefix_caches[0].insert(7, 4_097, 4_097, capacity, None);
        assert!(kv_store.prefix_caches[0].used_units() > 0);

        // Fill the hard tier: 8 residents × (one 2048 block + 3168 state) ≈ 41k,
        // so admission has to reclaim the whole retained tier to make room.
        for index in 0..8u32 {
            let footprint = kv_store.footprint(RequestId(index), 2_000, 0);
            kv_store.reserve(RequestId(index), 0, footprint, Time::ZERO);
        }
        assert_eq!(kv_store.prefix_cache_physical_limit(0), 0);
        assert_eq!(kv_store.prefix_caches[0].used_units(), 0);
    }

    // ── soft tier: live checkpoints ──────────────────────────────────────────

    #[test]
    fn live_checkpoints_shrink_the_retained_tier_but_never_admission() {
        let mut kv_store = state_store(
            100_000,
            STATE_TOKENS,
            Some(CHECKPOINT_INTERVAL),
            Some(PrefixResumePoints::EveryBlock),
        );
        let footprint = kv_store.footprint(RequestId(0), 2_047, 4);
        kv_store.reserve(RequestId(0), 0, footprint, Time::ZERO);
        kv_store.drain_ready();
        kv_store.commit_resident(RequestId(0), 0, 2_047, 4);

        assert_eq!(kv_store.clamped_live_checkpoint_units(0), 0);
        let limit_before = kv_store.prefix_cache_physical_limit(0);
        let fits_before = kv_store.fits(0, &kv_store.footprint(RequestId(1), 80_000, 0));

        // At 2048 the state is the running one; one token later it is a
        // checkpoint left behind.
        kv_store.advance(AdvanceScope::WholePartition(0), 1);
        assert_eq!(kv_store.clamped_live_checkpoint_units(0), 0);
        kv_store.advance(AdvanceScope::WholePartition(0), 1);
        assert_eq!(kv_store.clamped_live_checkpoint_units(0), STATE_TOKENS);
        assert_eq!(
            kv_store.prefix_cache_physical_limit(0),
            limit_before - STATE_TOKENS,
            "the new checkpoint comes straight out of the retained-prefix budget",
        );
        assert_eq!(
            kv_store.fits(0, &kv_store.footprint(RequestId(1), 80_000, 0)),
            fits_before,
            "an evictable checkpoint must not change an admission decision",
        );
        // 2049 tokens take a second block; the checkpoint takes nothing.
        let running = kv_store.accounting.held_while_running(2_049, 1);
        assert_eq!(
            kv_store.kv_usage(0),
            running as f64 / kv_store.accounting.capacity_units() as f64,
            "like vLLM's usage, cached checkpoints are free blocks"
        );
    }

    #[test]
    fn live_checkpoints_are_clamped_to_the_slack_they_may_use() {
        // Capacity leaves almost nothing after the hard tier, but the resident
        // request has passed many positions. The soft charge is clamped rather
        // than overcommitting physical memory.
        let mut kv_store = state_store(
            30_000,
            STATE_TOKENS,
            Some(512),
            Some(PrefixResumePoints::EveryBlock),
        );
        let footprint = kv_store.footprint(RequestId(0), 25_000, 5);
        kv_store.reserve(RequestId(0), 0, footprint, Time::ZERO);
        kv_store.drain_ready();
        kv_store.commit_resident(RequestId(0), 0, 25_000, 5);

        let slack = kv_store.soft_slack(0);
        assert!(slack > 0);
        assert!(
            kv_store.partitions[0].live_checkpoint_units() > slack,
            "the unclamped checkpoints must exceed the slack for this to prove anything",
        );
        assert_eq!(kv_store.clamped_live_checkpoint_units(0), slack);
        assert_eq!(kv_store.prefix_cache_physical_limit(0), 0);
    }

    // ── sliding windows ─────────────────────────────────────────────────────

    /// Inkling-like: a sliding-window part beside global attention. A window
    /// holds at most its own tokens, so long contexts cost per-token KV only,
    /// and a retained prefix resumes at the prompt end (vLLM's default).
    #[test]
    fn a_sliding_window_holds_its_window_and_resumes_at_the_prompt_end() {
        let layout = KvCacheLayout::new()
            .with_part(KvCachePart::per_token("global", 1))
            .with_part(KvCachePart::sliding_window("swa", 512, 4));
        let accounting = Arc::new(
            KvAccounting::resolve(
                &layout,
                KvAccountingOptions {
                    max_in_flight_tokens: Some(1_024),
                    ..KvAccountingOptions::default()
                },
                1_000_000,
            )
            .unwrap(),
        );
        let mut kv_store = KvCacheStore::new(
            1,
            accounting,
            capped_prefix_cache(1_000_000, PrefixCachePolicy::Lru),
            None,
            None,
        );
        let session = SessionInput::Session {
            session_id: 7,
            session_start_time: Time::ZERO,
            declared_prefix_tokens: 0,
        };
        let resolved = kv_store.preview_prefill_context(0, 8_000, session);
        let footprint = kv_store.footprint(RequestId(0), 8_000, 500);
        // Admission: 8000 global + the window plus one step's tokens in flight.
        assert_eq!(footprint.reserved_units(), 8_000 + 4 * (511 + 1_024));
        kv_store.reserve_prefill_context(RequestId(0), 0, resolved, footprint, Time::ZERO);
        kv_store.drain_ready();
        kv_store.commit_resident(RequestId(0), 0, 8_000, 500);
        assert_eq!(kv_store.partitions[0].resident_units(), 8_000 + 4 * 512);
        kv_store.advance(AdvanceScope::WholePartition(0), 500);
        assert_eq!(kv_store.partitions[0].resident_units(), 8_500 + 4 * 512);
        kv_store.release_retaining_prefix(RequestId(0), 0, Time::from_ms(1.0));

        // The next turn declares the whole 8500-token history; only the prompt
        // end (7999, the resend recomputes its last token) is resumable.
        let next_turn = SessionInput::Session {
            session_id: 7,
            session_start_time: Time::ZERO,
            declared_prefix_tokens: 8_500,
        };
        let resolved = kv_store.preview_prefill_context(0, 100, next_turn);
        assert_eq!(resolved.resident_prefix_tokens(), 7_999);
        assert_eq!(kv_store.prefix_caches[0].used_units(), 7_999 + 4 * 511);
        let footprint = kv_store.footprint(RequestId(1), 8_600, 10);
        kv_store.reserve_prefill_context(RequestId(1), 0, resolved, footprint, Time::ZERO);
        assert_eq!(kv_store.prefix_caches[0].used_units(), 0);
    }
}
