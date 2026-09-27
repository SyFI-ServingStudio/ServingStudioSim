//! KV accounting for a hybrid model: per-token full attention **plus** a fixed
//! per-request recurrent (SSM + causal-conv) state, both living in the same
//! physical attention memory.
//!
//! ## Two ledgers, one capacity
//!
//! | tier | contents | evictable | enters [`KvStore::fits`] |
//! |---|---|---|---|
//! | hard | full-attention KV (context + future decode) + **one** live recurrent state per resident request | no | yes |
//! | soft | retained prefix KV and its snapshots; snapshots a live decode drops while running | yes | no |
//!
//! The hard tier is what admission reserves atomically: one request's footprint
//! is `post_prefill_context + remaining_output + state_tokens`, all three
//! checked and reserved in a single [`KvStore::reserve`]. The recurrent half is
//! paid once at admission and never grows with decode — a recurrent state is a
//! rolling snapshot, not a per-token page.
//!
//! The soft tier is clamped to whatever capacity the hard tier leaves, exactly
//! like the pre-existing `prefix_cache_physical_limit` rule, so the total
//! constraint covers active + reserved + retained + held and their recurrent
//! shares without a second admission ledger.
//!
//! Chunked prefill ([`ChunkedPrefillKv`]) keeps that single reservation: the
//! state is promised with the request's first chunk and committed resident with
//! its last. Where the engine can only checkpoint state on block boundaries, the
//! lifecycle, not this store, clips chunk ends to them.
//!
//! ## Why prefix reuse quantizes
//!
//! A full-attention cache can resume from any token: page `i` is independent of
//! page `j`. A recurrent state cannot — the state at token `T` is only
//! recoverable from a snapshot taken at `T`, and a *later* snapshot is useless
//! to an *earlier* resume point. vLLM's hybrid allocator therefore snapshots
//! only at multiples of an aligned block size and quantizes hits to it (see
//! `arch::qwen36_local` for the alignment formula and its verification against
//! measured vLLM numbers). This store passes that interval to
//! [`PrefixCache`] as its reuse quantum, and the per-request state size as its
//! per-snapshot charge, so a retained prefix pays for the snapshots that make it
//! resumable.
//!
//! ## Not modeled
//!
//! Live snapshots yield to admission but are never *proactively* shed: there is
//! no retention-interval or boundary-protection policy picking which snapshot to
//! drop first. vLLM's own work in that direction is
//! <https://github.com/vllm-project/vllm/pull/43447> and Marconi
//! (<https://arxiv.org/abs/2411.19379>).

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
    ChunkedPrefillKv, IterWorkerKv, KvStore, PrefixCacheTokenConfig, PrefixKv,
    ResolvedPrefillContext,
};
use crate::worker::shared::advance_scope::{AdvanceScope, PartitionId};

/// Hard-tier charge of one request, in full-attention token equivalents.
pub struct HybridKvFootprint {
    post_prefill_context_tokens: u32,
    /// Decode tokens the reservation covers: the whole remaining output, or a
    /// bounded-future estimate plus one page.
    future_decode_tokens: u32,
    /// The whole model's recurrent state for this one request. Constant.
    state_tokens: u64,
}

impl HybridKvFootprint {
    #[inline]
    fn reserved_charge(&self) -> u64 {
        u64::from(self.post_prefill_context_tokens)
            + u64::from(self.future_decode_tokens)
            + self.state_tokens
    }
}

pub struct HybridGdnKv {
    partitions: Vec<ResidentPartitionState>,
    ledger: RequestLedger,
    prefix_caches: Vec<PrefixCache>,
    journal: PrefixCacheJournal,
    sampler: Option<KvSampler>,
    /// One request's recurrent state, expressed in full-attention token
    /// equivalents so a single capacity number governs both kinds of memory.
    state_tokens_per_request: u64,
    /// Context-token spacing of resumable snapshots. `0` = no recurrent state.
    checkpoint_interval_tokens: u32,
}

impl KvStore for HybridGdnKv {
    type Footprint = HybridKvFootprint;

    fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    fn footprint(
        &self,
        _request: RequestId,
        post_prefill_context_tokens: u32,
        remaining_output_tokens: u32,
    ) -> Self::Footprint {
        HybridKvFootprint {
            post_prefill_context_tokens,
            future_decode_tokens: remaining_output_tokens,
            state_tokens: self.state_tokens_per_request,
        }
    }

    /// Both halves of the footprint are tested against one capacity, against
    /// both the immediate resident total and the projected decode peak. The
    /// soft tier is deliberately absent: it yields to this decision.
    fn fits(&self, partition: PartitionId, footprint: &Self::Footprint) -> bool {
        let partition_state = &self.partitions[partition as usize];
        let reserved_charge = self.ledger.partition_promised(partition)
            + self.ledger.partition_held(partition)
            + footprint.reserved_charge();
        partition_state.resident_tokens() + reserved_charge <= partition_state.capacity_tokens()
            && partition_state.projected_peak_kv() + reserved_charge
                <= partition_state.capacity_tokens()
    }

    fn reserve(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        footprint: Self::Footprint,
        now: Time,
    ) {
        self.ledger
            .promise(request, partition, footprint.reserved_charge());
        let evictions = self.trim_prefix_cache_to_physical_slack(partition);
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

    fn commit_resident(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        post_prefill_context_tokens: u64,
        remaining_output_tokens: u32,
    ) {
        self.ledger.forget_promise(request);
        // `ResidentPartitionState` charges `state_tokens_per_request` here and
        // releases it in `release_decode`; it is never advanced.
        self.partitions[partition as usize].begin_decode(
            request,
            post_prefill_context_tokens,
            remaining_output_tokens,
        );
    }

    fn advance(&mut self, scope: AdvanceScope<'_>, steps: u32) {
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
    }

    fn release(&mut self, request: RequestId, partition: PartitionId) {
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

    fn sample_submit(&mut self, partition: PartitionId, now: Time) {
        if self.sampler.is_none() {
            return;
        }
        let retained_prefix_kv = self.prefix_caches[partition as usize].used_charge();
        let live_checkpoints = self.clamped_live_checkpoint_charge(partition);
        let held = self.ledger.partition_held(partition);
        let partition_state = &self.partitions[partition as usize];
        let submit = KvSubmit {
            active_kv: partition_state.resident_tokens()
                + held
                + retained_prefix_kv
                + live_checkpoints,
            retained_prefix_kv,
            projected_peak: partition_state.projected_peak_kv()
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
}

impl IterWorkerKv for HybridGdnKv {
    fn drain_ready(&mut self) {
        for (partition, request) in self.ledger.drain_promised() {
            self.partitions[partition as usize].add_prefill_admit(request);
        }
    }

    fn clear_prefill_admits(&mut self, partition: PartitionId) {
        self.partitions[partition as usize].clear_prefill_admits();
    }

    fn has_live_decode(&self, partition: PartitionId) -> bool {
        self.partitions[partition as usize].has_live_decode()
    }

    fn live_decode_count(&self, partition: PartitionId) -> u32 {
        self.partitions[partition as usize].live_decode_count()
    }

    fn has_prefill_admit(&self, partition: PartitionId) -> bool {
        self.partitions[partition as usize].has_prefill_admit()
    }

    fn status_active(&self, partition: PartitionId) -> u32 {
        self.partitions[partition as usize].live_decode_count()
            + self.ledger.partition_promised_count(partition)
            + self.partitions[partition as usize].prefill_admit_count()
    }

    fn release_external(&mut self, request: RequestId, current_kv: u64) -> Option<PartitionId> {
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

    fn visit_prefill_admits(&self, partition: PartitionId, mut visitor: impl FnMut(RequestId)) {
        for request in self.partitions[partition as usize].iter_prefill_admits() {
            visitor(request);
        }
    }

    fn visit_decode_members(
        &self,
        partition: PartitionId,
        mut visitor: impl FnMut(RequestId, u64),
    ) {
        // Context tokens, never charge: this feeds `decode_kv_lens` in the arch
        // input and therefore the attention cost model.
        for (request, current_kv) in self.partitions[partition as usize].decode_members() {
            visitor(request, current_kv);
        }
    }
}

impl PrefixKv for HybridGdnKv {
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
                        .expect("HybridGdnKv partition count must fit PartitionId"),
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
            // `ResolvedPrefillContext::new` borrows one token back from a fully
            // resident prefix so that a zero-length prefill never reaches the
            // model. The snapshot still physically sits at the boundary, so the
            // resume point is off by that one borrowed token. Cost is unchanged
            // (a GDN prefill only sees the append length and whether a state
            // exists), so the accounting keeps the boundary and tolerates the
            // one-token gap rather than inventing an unaligned snapshot.
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
}

/// Chunked prefill over the same two tiers. The recurrent state is part of the
/// one footprint reserved at admission, stays promised while chunks run, and
/// becomes resident exactly once, when the last chunk commits the request.
impl ChunkedPrefillKv for HybridGdnKv {
    fn bounded_future_footprint(
        &self,
        _request: RequestId,
        post_prefill_context_tokens: u32,
        remaining_output_tokens: u32,
        max_future_tokens: u32,
        page_size: u32,
    ) -> Self::Footprint {
        let future_decode_tokens = remaining_output_tokens
            .min(max_future_tokens)
            .checked_add(page_size)
            .expect("bounded-future KV footprint overflow");
        HybridKvFootprint {
            post_prefill_context_tokens,
            future_decode_tokens,
            state_tokens: self.state_tokens_per_request,
        }
    }

    /// Like [`KvStore::fits`], only the hard tier is protected: retained
    /// prefixes and live snapshots yield.
    fn fits_bounded_future(
        &self,
        partition: PartitionId,
        footprint: &Self::Footprint,
        max_future_tokens: u32,
        new_token_ratio: f64,
        decode_step_tokens: u32,
    ) -> bool {
        let partition_state = &self.partitions[partition as usize];
        let protected_tokens = partition_state.resident_tokens()
            + self.ledger.partition_promised(partition)
            + self.ledger.partition_held(partition)
            + footprint.reserved_charge();
        let mut running_future =
            partition_state.bounded_future_decode_tokens(max_future_tokens, new_token_ratio);
        if decode_step_tokens > 0 {
            // Bounded-future admission runs on one-token pages only.
            let next_step = partition_state.next_decode_allocation_tokens(1, decode_step_tokens);
            running_future = running_future.max(next_step as f64);
        }
        protected_tokens as f64 + running_future < partition_state.capacity_tokens() as f64
    }

    /// The live-snapshot staircase keeps its precedence over retained prefixes
    /// but, being soft, never counts toward the shortfall.
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
        let capacity = partition_state.capacity_tokens();
        let slack_after_step = capacity.saturating_sub(protected.saturating_add(required));
        let live_checkpoints = partition_state
            .live_checkpoint_count(self.checkpoint_interval_tokens)
            .saturating_mul(self.state_tokens_per_request)
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
        let cache_tokens = self.prefix_caches[partition as usize].used_charge();
        let available = capacity.saturating_sub(protected.saturating_add(cache_tokens));
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

impl HybridGdnKv {
    /// `state_tokens_per_request` and `checkpoint_interval_tokens` come from the
    /// arch (`recurrent_state_bytes_per_request` divided by
    /// `total_kv_bytes_per_token`, and `recurrent_checkpoint_interval_tokens`).
    pub(crate) fn new(
        num_partitions: usize,
        kv_capacity: u64,
        state_tokens_per_request: u64,
        checkpoint_interval_tokens: u32,
        prefix_cache: PrefixCacheTokenConfig,
        sampler: Option<KvSampler>,
        prefix_cache_logger: Option<PrefixCacheLogger>,
    ) -> Self {
        // A model with no recurrent state degrades to full-attention accounting
        // rather than treating every single token as a snapshot boundary. The
        // recipe should not select this store for such a model.
        let (quantum_tokens, charge_per_checkpoint) = if checkpoint_interval_tokens == 0 {
            (1, 0)
        } else {
            (checkpoint_interval_tokens, state_tokens_per_request)
        };
        Self {
            partitions: (0..num_partitions)
                .map(|_| ResidentPartitionState::new(kv_capacity, state_tokens_per_request))
                .collect(),
            ledger: RequestLedger::new(num_partitions),
            prefix_caches: (0..num_partitions)
                .map(|_| {
                    PrefixCache::new(
                        prefix_cache.max_retained_tokens.min(kv_capacity),
                        prefix_cache.policy,
                        quantum_tokens,
                        charge_per_checkpoint,
                    )
                })
                .collect(),
            journal: PrefixCacheJournal::new(prefix_cache_logger),
            sampler,
            state_tokens_per_request,
            checkpoint_interval_tokens,
        }
    }

    /// Hard-tier occupancy: whichever of the immediate and the projected total
    /// is larger, including everything reserved but not yet resident.
    fn non_cache_peak(&self, partition: PartitionId) -> u64 {
        let reserved =
            self.ledger.partition_promised(partition) + self.ledger.partition_held(partition);
        let partition_state = &self.partitions[partition as usize];
        (partition_state.resident_tokens() + reserved)
            .max(partition_state.projected_peak_kv() + reserved)
    }

    /// Capacity the hard tier is not using — the whole soft budget.
    fn soft_slack(&self, partition: PartitionId) -> u64 {
        self.partitions[partition as usize]
            .capacity_tokens()
            .saturating_sub(self.non_cache_peak(partition))
    }

    /// Snapshots live decodes have dropped behind them, priced as full state
    /// copies. Every `checkpoint_interval_tokens` of context a decode produces
    /// leaves one more resumable snapshot on the way.
    ///
    /// This is a soft, evictable staircase: it is clamped to the slack the hard
    /// tier leaves, and never participates in [`KvStore::fits`]. It therefore
    /// cannot starve a resident request, only the retained-prefix tier.
    fn clamped_live_checkpoint_charge(&self, partition: PartitionId) -> u64 {
        let checkpoints = self.partitions[partition as usize]
            .live_checkpoint_count(self.checkpoint_interval_tokens);
        checkpoints
            .saturating_mul(self.state_tokens_per_request)
            .min(self.soft_slack(partition))
    }

    /// What the retained-prefix tier may occupy: the soft budget minus the
    /// live-decode staircase, which is nearer-term useful and so wins.
    fn prefix_cache_physical_limit(&self, partition: PartitionId) -> u64 {
        self.soft_slack(partition)
            .saturating_sub(self.clamped_live_checkpoint_charge(partition))
    }

    fn trim_prefix_cache_to_physical_slack(
        &mut self,
        partition: PartitionId,
    ) -> Vec<PrefixCacheMutation> {
        let physical_limit = self.prefix_cache_physical_limit(partition);
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

    use crate::worker::kv::PrefixCachePolicy;

    /// Qwen3.6-35B-A3B TP1: 30 GDN layers of state over a 10-GQA-layer token
    /// page, and the vLLM block alignment that follows from it.
    const STATE_TOKENS: u64 = 3_168;
    const CHECKPOINT_INTERVAL: u32 = 2_048;

    fn uncapped(max_retained_tokens: u64) -> PrefixCacheTokenConfig {
        PrefixCacheTokenConfig {
            max_retained_tokens,
            policy: PrefixCachePolicy::Lru,
        }
    }

    fn store(kv_capacity: u64, state_tokens: u64, interval: u32) -> HybridGdnKv {
        HybridGdnKv::new(
            1,
            kv_capacity,
            state_tokens,
            interval,
            uncapped(kv_capacity),
            None,
            None,
        )
    }

    // ── hard tier ────────────────────────────────────────────────────────────

    #[test]
    fn admission_reserves_context_future_decode_and_state_in_one_check() {
        let kv_store = store(1_000, 100, 512);
        // 800 context + 100 decode + 100 state == capacity, to the token.
        assert!(kv_store.fits(0, &kv_store.footprint(RequestId(0), 800, 100)));
        assert!(
            !kv_store.fits(0, &kv_store.footprint(RequestId(0), 801, 100)),
            "one token over capacity must be refused",
        );
        assert!(
            !kv_store.fits(0, &kv_store.footprint(RequestId(0), 800, 101)),
            "future decode tokens are part of the same atomic reservation",
        );
        let stateless = store(1_000, 0, 0);
        assert!(
            stateless.fits(0, &stateless.footprint(RequestId(0), 900, 100)),
            "the same request fits only once the recurrent state is priced out",
        );
    }

    #[test]
    fn a_promised_state_blocks_the_next_request_before_it_is_resident() {
        let mut kv_store = store(1_000, 100, 512);
        let first = kv_store.footprint(RequestId(0), 400, 0);
        kv_store.reserve(RequestId(0), 0, first, Time::ZERO);
        // 500 promised (400 + 100 state) leaves 500, so 400 context + 100 state fits...
        assert!(kv_store.fits(0, &kv_store.footprint(RequestId(1), 400, 0)));
        // ...but 401 does not, proving the state rode along in the promise.
        assert!(!kv_store.fits(0, &kv_store.footprint(RequestId(1), 401, 0)));
    }

    #[test]
    fn state_is_charged_once_at_commit_never_advanced_and_fully_released() {
        let mut kv_store = store(100_000, STATE_TOKENS, CHECKPOINT_INTERVAL);
        let footprint = kv_store.footprint(RequestId(0), 1_000, 20);
        kv_store.reserve(RequestId(0), 0, footprint, Time::ZERO);
        kv_store.drain_ready();
        kv_store.commit_resident(RequestId(0), 0, 1_000, 20);
        assert_eq!(
            kv_store.partitions[0].resident_tokens(),
            1_000 + STATE_TOKENS
        );

        kv_store.advance(AdvanceScope::WholePartition(0), 10);
        assert_eq!(
            kv_store.partitions[0].resident_tokens(),
            1_010 + STATE_TOKENS,
            "decode grows context only",
        );
        assert_eq!(
            kv_store.partitions[0].decode_members().collect::<Vec<_>>(),
            [(RequestId(0), 1_010)],
            "decode_kv_lens must stay pure context",
        );

        kv_store.release(RequestId(0), 0);
        assert_eq!(kv_store.partitions[0].resident_tokens(), 0);
        assert_eq!(kv_store.ledger.partition_promised(0), 0);
    }

    // ── chunked prefill ──────────────────────────────────────────────────────

    #[test]
    fn a_chunked_prefill_keeps_its_state_promised_until_the_last_chunk_commits_it() {
        let mut kv_store = store(100_000, STATE_TOKENS, CHECKPOINT_INTERVAL);
        let resolved = kv_store.preview_prefill_context(0, 5_000, SessionInput::Standalone);
        let footprint = kv_store.footprint(RequestId(0), 5_000, 20);
        kv_store.reserve_chunked_prefill_context(RequestId(0), 0, resolved, footprint, Time::ZERO);
        kv_store.drain_ready();
        let promised = 5_000 + 20 + STATE_TOKENS;

        for chunk in [2_048, 2_048] {
            kv_store.schedule_prefill_chunk(RequestId(0), 0, chunk);
            assert_eq!(kv_store.ledger.partition_promised(0), promised);
            assert_eq!(kv_store.partitions[0].resident_tokens(), 0);
            kv_store.complete_prefill_chunk(RequestId(0));
            kv_store.clear_prefill_admits(0);
        }
        kv_store.schedule_prefill_chunk(RequestId(0), 0, 904);
        kv_store.complete_prefill_chunk(RequestId(0));
        kv_store.finish_chunked_prefill(RequestId(0), 0, 20);
        assert_eq!(kv_store.ledger.partition_promised(0), 0);
        assert_eq!(
            kv_store.partitions[0].resident_tokens(),
            5_000 + STATE_TOKENS,
            "state becomes resident exactly once"
        );

        kv_store.release(RequestId(0), 0);
        assert_eq!(kv_store.partitions[0].resident_tokens(), 0);
    }

    #[test]
    fn a_request_cancelled_mid_prefill_returns_its_whole_reservation() {
        let mut kv_store = store(100_000, STATE_TOKENS, CHECKPOINT_INTERVAL);
        let resolved = kv_store.preview_prefill_context(0, 5_000, SessionInput::Standalone);
        let footprint = kv_store.footprint(RequestId(0), 5_000, 20);
        kv_store.reserve_chunked_prefill_context(RequestId(0), 0, resolved, footprint, Time::ZERO);
        kv_store.schedule_prefill_chunk(RequestId(0), 0, 2_048);
        kv_store.complete_prefill_chunk(RequestId(0));

        kv_store.release(RequestId(0), 0);
        assert_eq!(kv_store.ledger.partition_promised(0), 0);
        assert_eq!(
            kv_store.status_active(0),
            1,
            "the prefill admit clears with the iteration"
        );
        kv_store.clear_prefill_admits(0);
        assert_eq!(kv_store.status_active(0), 0);
    }

    #[test]
    fn a_bounded_future_footprint_still_carries_the_whole_state() {
        let kv_store = store(100_000, STATE_TOKENS, CHECKPOINT_INTERVAL);
        let footprint = kv_store.bounded_future_footprint(RequestId(0), 1_000, 500, 64, 1);
        assert_eq!(footprint.reserved_charge(), 1_000 + 64 + 1 + STATE_TOKENS);
        assert!(kv_store.fits_bounded_future(0, &footprint, 64, 0.7, 1));
        let tight = store(1_000 + 65 + STATE_TOKENS, STATE_TOKENS, CHECKPOINT_INTERVAL);
        assert!(!tight.fits_bounded_future(0, &footprint, 64, 0.7, 1));
    }

    // ── soft tier: quantized reuse ───────────────────────────────────────────

    /// Reproduces the hit rates reported for a hybrid Qwen3 model in
    /// <https://github.com/vllm-project/vllm/issues/40696>, whose aligned block
    /// size is 528 tokens. A prompt shorter than one block gets nothing; longer
    /// prompts reuse only whole blocks, so the hit rate saws downward as the
    /// prompt grows past a boundary.
    #[test]
    fn quantized_reuse_reproduces_the_measured_vllm_hit_rates() {
        const BLOCK: u32 = 528;
        let expected = [
            (479u32, 0u32, 0.0f64),
            (552, 528, 95.4),
            (597, 528, 88.2),
            (979, 528, 53.7),
        ];
        for (prompt_tokens, expected_hit, measured_percent) in expected {
            let mut kv_store = store(10_000_000, STATE_TOKENS, BLOCK);
            // First turn retains what it can; the second turn replays it.
            kv_store.prefix_caches[0].insert(7, u64::from(prompt_tokens), 10_000_000, None);
            let hit = kv_store.prefix_caches[0].peek(7, prompt_tokens);
            assert_eq!(
                hit, expected_hit,
                "prompt {prompt_tokens}: block-aligned reuse",
            );
            let modeled_percent = 100.0 * f64::from(hit) / f64::from(prompt_tokens);
            assert!(
                (modeled_percent - measured_percent).abs() < 0.5,
                "prompt {prompt_tokens}: modeled {modeled_percent:.1}% vs measured {measured_percent}%",
            );
        }
    }

    #[test]
    fn a_retained_prefix_pays_for_the_snapshots_that_make_it_resumable() {
        let mut kv_store = store(10_000_000, STATE_TOKENS, CHECKPOINT_INTERVAL);
        let session = SessionInput::Session {
            session_id: 7,
            session_start_time: Time::ZERO,
            declared_prefix_tokens: 0,
        };
        let resolved_prefill = kv_store.preview_prefill_context(0, 5_000, session);
        let footprint = kv_store.footprint(RequestId(0), 5_000, 0);
        kv_store.reserve_prefill_context(RequestId(0), 0, resolved_prefill, footprint, Time::ZERO);
        kv_store.drain_ready();
        kv_store.commit_resident(RequestId(0), 0, 5_000, 0);
        kv_store.release_retaining_prefix(RequestId(0), 0, Time::from_ms(1.0));

        // 5_000 tokens hold two whole 2_048-token snapshots; the 904-token tail
        // is unusable because no snapshot sits at its end.
        assert_eq!(kv_store.prefix_caches[0].used_tokens(), 2 * 2_048);
        assert_eq!(
            kv_store.prefix_caches[0].used_charge(),
            2 * 2_048 + 2 * STATE_TOKENS,
        );
        assert_eq!(kv_store.prefix_caches[0].peek(7, 5_000), 2 * 2_048);
        assert_eq!(
            kv_store.prefix_caches[0].peek(7, 2_047),
            0,
            "a later snapshot cannot serve an earlier resume point",
        );
    }

    /// The reusable length is bounded by the attention KV **and** by the last
    /// recurrent snapshot. Rather than taking a `min` of two independently
    /// tracked lengths, one entry carries both halves at one aligned length:
    /// they are sized together, charged together, and evicted together, so they
    /// cannot drift apart.
    #[test]
    fn attention_kv_and_recurrent_snapshots_are_retained_at_one_aligned_length() {
        let interval = 2_048u64;
        // Room for exactly two snapshots' worth of both halves, plus change.
        let capacity = 2 * (interval + STATE_TOKENS) + 500;
        let mut kv_store = store(capacity, STATE_TOKENS, interval as u32);

        // Ask to retain five intervals; only two fit, and the entry is cut at a
        // snapshot boundary rather than at whatever the byte budget allowed.
        let insert_result = kv_store.prefix_caches[0].insert(7, 5 * interval, capacity, None);
        assert_eq!(insert_result.retained_tokens, 2 * interval);
        assert_eq!(
            kv_store.prefix_caches[0].used_charge(),
            2 * (interval + STATE_TOKENS),
            "both halves of every retained snapshot are paid for",
        );
        assert_eq!(kv_store.prefix_caches[0].used_tokens() % interval, 0);

        // Eviction takes both halves at once: no orphaned KV, no orphaned state.
        kv_store.prefix_caches[0].shrink_to(0);
        assert_eq!(kv_store.prefix_caches[0].used_tokens(), 0);
        assert_eq!(kv_store.prefix_caches[0].used_charge(), 0);
        assert_eq!(kv_store.prefix_caches[0].peek(7, 5 * interval as u32), 0);
    }

    #[test]
    fn a_prefix_shorter_than_one_snapshot_interval_is_not_retained() {
        let mut kv_store = store(10_000_000, STATE_TOKENS, CHECKPOINT_INTERVAL);
        let insert_result = kv_store.prefix_caches[0].insert(7, 2_047, 10_000_000, None);
        assert_eq!(insert_result.retained_tokens, 0);
        assert_eq!(kv_store.prefix_caches[0].used_charge(), 0);
    }

    #[test]
    fn the_soft_tier_yields_entirely_when_the_hard_tier_fills_the_partition() {
        let capacity = 40_000;
        let mut kv_store = store(capacity, STATE_TOKENS, CHECKPOINT_INTERVAL);
        kv_store.prefix_caches[0].insert(7, 4_096, capacity, None);
        assert!(kv_store.prefix_caches[0].used_charge() > 0);

        // Fill the hard tier: 8 residents × (2_000 context + 3_168 state) ≈ 41k,
        // so admission has to reclaim the whole retained tier to make room.
        for index in 0..8u32 {
            let footprint = kv_store.footprint(RequestId(index), 2_000, 0);
            kv_store.reserve(RequestId(index), 0, footprint, Time::ZERO);
        }
        assert_eq!(kv_store.prefix_cache_physical_limit(0), 0);
        assert_eq!(kv_store.prefix_caches[0].used_charge(), 0);
    }

    // ── soft tier: the live-decode staircase ─────────────────────────────────

    #[test]
    fn live_decode_snapshots_shrink_the_retained_tier_but_never_admission() {
        let capacity = 100_000;
        let interval = 2_048u32;
        let mut kv_store = store(capacity, STATE_TOKENS, interval);
        let footprint = kv_store.footprint(RequestId(0), 2_047, 4);
        kv_store.reserve(RequestId(0), 0, footprint, Time::ZERO);
        kv_store.drain_ready();
        kv_store.commit_resident(RequestId(0), 0, 2_047, 4);

        assert_eq!(kv_store.clamped_live_checkpoint_charge(0), 0);
        let limit_before = kv_store.prefix_cache_physical_limit(0);
        let fits_before = kv_store.fits(0, &kv_store.footprint(RequestId(1), 90_000, 0));

        // One more token crosses the 2_048 boundary and leaves a snapshot behind.
        kv_store.advance(AdvanceScope::WholePartition(0), 1);
        assert_eq!(kv_store.clamped_live_checkpoint_charge(0), STATE_TOKENS);
        assert_eq!(
            kv_store.prefix_cache_physical_limit(0),
            limit_before - STATE_TOKENS,
            "the new snapshot comes straight out of the retained-prefix budget",
        );
        assert_eq!(
            kv_store.fits(0, &kv_store.footprint(RequestId(1), 90_000, 0)),
            fits_before,
            "an evictable snapshot must not change an admission decision",
        );
    }

    #[test]
    fn the_staircase_is_clamped_to_the_slack_it_is_allowed_to_use() {
        // Capacity leaves almost nothing after the hard tier, but the resident
        // request has crossed many snapshot boundaries. The soft charge is
        // clamped rather than overcommitting physical memory.
        let mut kv_store = store(30_000, STATE_TOKENS, 512);
        let footprint = kv_store.footprint(RequestId(0), 25_000, 5);
        kv_store.reserve(RequestId(0), 0, footprint, Time::ZERO);
        kv_store.drain_ready();
        kv_store.commit_resident(RequestId(0), 0, 25_000, 5);

        let slack = kv_store.soft_slack(0);
        assert_eq!(slack, 30_000 - 25_000 - STATE_TOKENS - 5);
        assert!(
            kv_store.partitions[0].live_checkpoint_count(512) * STATE_TOKENS > slack,
            "the unclamped staircase must exceed the slack for this to prove anything",
        );
        assert_eq!(kv_store.clamped_live_checkpoint_charge(0), slack);
        assert_eq!(kv_store.prefix_cache_physical_limit(0), 0);
    }

    #[test]
    fn a_zero_interval_model_degrades_to_plain_full_attention_accounting() {
        let mut kv_store = store(10_000, 0, 0);
        kv_store.prefix_caches[0].insert(7, 137, 10_000, None);
        assert_eq!(kv_store.prefix_caches[0].peek(7, 137), 137);
        assert_eq!(kv_store.prefix_caches[0].used_charge(), 137);
        assert_eq!(kv_store.clamped_live_checkpoint_charge(0), 0);
    }
}
