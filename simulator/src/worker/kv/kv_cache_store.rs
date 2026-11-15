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
    PrefixCacheEvictionReason, PrefixCacheLogger, PrefixCacheRetentionReason, HBM_TIER,
};
use crate::worker::kv::shared::kv_accounting::SharedKvAccounting;
use crate::worker::kv::shared::prefix_cache::{
    PrefixCache, PrefixCacheBudget, PrefixCacheInsertResult, PrefixCacheMutation,
    PrefixCacheMutationKind, PrefixCacheReturnMetadata,
};
use crate::worker::kv::shared::prefix_cache_journal::PrefixCacheJournal;
use crate::worker::kv::shared::request_ledger::RequestLedger;
use crate::worker::kv::shared::resident_partition::ResidentPartitionState;
use crate::worker::kv::shared::state_retention::Carried;
use crate::worker::kv::shared::tier_chain::{
    LeavingContext, LoadingQueue, RoundContext, TierBytes, TierChain, TierHit, TierSpec,
    TierStateOptions,
};
use crate::worker::kv::shared::tier_reads::{
    ReadPhase, ReadRecord, SharedTiersReports, TierReads, TiersReport,
};
use crate::worker::kv::{
    ChunkedPrefillKv, HandBack, HandoffKv, IterWorkerKv, KvStore, PrefixKv, ResolvedPrefillContext,
    SlotPipelineKv, TierReadPlan, TieredPrefixKv,
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
    /// Host DRAM/SSD tiers behind each partition's prefix cache; empty
    /// without tiers.
    tier_chains: Vec<TierChain>,
    /// Reads from those tiers; `None` without tiers.
    tier_reads: Option<TierReads>,
    /// Whether a declared prefix longer than anything the run stored for the
    /// session is a pre-run context, read from the slowest tier.
    tier_warm_start: bool,
    /// Where the tiers' end-of-run report goes when the store is dropped.
    tiers_reports: Option<SharedTiersReports>,
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
        self.record_pressure_evictions(trigger, partition, now, evictions);
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

impl TieredPrefixKv for KvCacheStore {
    fn reusable_prefix_tokens(
        &self,
        partition: PartitionId,
        fresh_prompt_tokens: u32,
        session_input: SessionInput,
    ) -> u32 {
        let resident = self.resident_prefix_tokens(fresh_prompt_tokens, session_input);
        let SessionInput::Session {
            session_id,
            declared_prefix_tokens,
            ..
        } = session_input
        else {
            return resident;
        };
        self.tier_lookup(partition, session_id, u64::from(declared_prefix_tokens))
            .map_or(resident, |(hit, _)| resident.max(hit.tokens as u32))
    }

    fn plan_tier_read(
        &self,
        partition: PartitionId,
        request: RequestId,
        fresh_prompt_tokens: u32,
        remaining_output_tokens: u32,
        session_input: SessionInput,
        now: Time,
    ) -> Option<TierReadPlan<KvFootprint>> {
        let SessionInput::Session {
            session_id,
            declared_prefix_tokens,
            ..
        } = session_input
        else {
            return None;
        };
        let resolved = self.preview_prefill_context(partition, fresh_prompt_tokens, session_input);
        let resident = u64::from(resolved.resident_prefix_tokens());
        let reads = self.tier_reads.as_ref()?;
        if let Some(record) = reads
            .record(request)
            .filter(|record| record.phase == ReadPhase::Restored)
        {
            if resident >= record.hit.tokens {
                return None;
            }
        }
        let (hit, seeded) = self
            .tier_lookup(partition, session_id, u64::from(declared_prefix_tokens))
            .filter(|(hit, _)| hit.tokens > resident)?;
        let chain = &self.tier_chains[partition as usize];
        // With anchors in place a round resuming from an anchor a tier holds
        // moves only that state: the KV of HBM's entry up to the hit counts
        // as resident.
        let resident = if chain.anchors_in_place() && !seeded {
            let retained = self.prefix_caches[partition as usize].entry_tokens(session_id);
            resident.max(retained.min(hit.tokens))
        } else {
            resident
        };
        let earlier = if seeded {
            0
        } else {
            chain.read_states(session_id, hit).len() as u64
        };
        let staged = chain.staged_charge(hit.tier, hit.tokens - resident, 1 + earlier);
        // Staged, the read holds only what it lands; its request's footprint
        // waits for admission.
        let (context, outputs) = match staged {
            Some(_) => (hit.tokens - resident, 0),
            None => (
                u64::from(resolved.post_prefill_context_tokens()),
                u64::from(remaining_output_tokens),
            ),
        };
        Some(TierReadPlan {
            tier: hit.tier,
            queue_wait: chain.queue_wait(hit.tier, now),
            hit_tokens: hit.tokens,
            resident_tokens: resident,
            hold: KvFootprint {
                reserved_units: self.accounting.footprint(context, outputs)
                    + earlier * self.accounting.reusable_state_units(),
                basis: ReservationBasis::FullFootprint,
            },
            staged,
        })
    }

    fn tier_sessions_at_risk(&self, partition: PartitionId, horizon: u64) -> Vec<u32> {
        self.tier_chains
            .get(partition as usize)
            .map_or_else(Vec::new, |chain| chain.sessions_at_risk(horizon))
    }

    fn tiers_take_hbm_evictions(&self, partition: PartitionId) -> bool {
        self.tier_chains
            .get(partition as usize)
            .is_some_and(TierChain::takes_hbm_evictions)
    }

    fn stages_tier_reads(&self) -> bool {
        self.tier_chains.first().is_some_and(TierChain::is_staged)
    }

    fn has_loading_room(&self, partition: PartitionId, plan: &TierReadPlan<KvFootprint>) -> bool {
        let staged = plan.staged.expect("a staged read");
        self.tier_chains[partition as usize].area_fits(staged.queue, staged)
    }

    fn start_tier_read(
        &mut self,
        partition: PartitionId,
        request: RequestId,
        session_input: SessionInput,
        plan: TierReadPlan<KvFootprint>,
        now: Time,
    ) {
        let SessionInput::Session {
            session_id,
            declared_prefix_tokens,
            ..
        } = session_input
        else {
            unreachable!("only a session request reads from a tier");
        };
        let resident = plan.resident_tokens;
        let (hit, seeded) = self
            .tier_lookup(partition, session_id, u64::from(declared_prefix_tokens))
            .expect("a planned read has a hit");
        debug_assert_eq!((hit.tier, hit.tokens), (plan.tier, plan.hit_tokens));
        let hold_units = plan.hold.reserved_units();
        let into_dram = plan
            .staged
            .is_some_and(|staged| staged.queue == LoadingQueue::Dram);
        if let Some(staged) = plan.staged {
            self.tier_chains[partition as usize].take_area(staged.queue, staged);
        }
        // One landing in DRAM holds HBM from its hop on.
        let evicted = if into_dram {
            Vec::new()
        } else {
            self.hold_for_read(request, partition, hold_units, hit.tokens, now)
        };
        let reads = self.tier_reads.as_mut().expect("a read implies tiers");
        let chain = &mut self.tier_chains[partition as usize];
        if seeded {
            chain.seed(
                session_id,
                self.accounting.resume_positions(hit.tokens, hit.tokens),
            );
            reads.note_stored(partition, session_id, hit.tokens);
            reads.note_warm_start(partition, hit.tokens);
        }
        let read = chain.read(session_id, hit, resident, now);
        reads.start(
            request,
            ReadRecord {
                partition,
                session_id,
                hit,
                read,
                phase: if into_dram {
                    ReadPhase::IntoDram
                } else {
                    ReadPhase::Reading
                },
                hold_units,
                staged: plan.staged,
            },
        );
        for context in evicted {
            chain.take_hbm_eviction(context);
        }
        self.log_tier_events(partition, request, now);
    }

    fn next_tier_landing(&self) -> Option<Time> {
        self.tier_reads.as_ref()?.next_landing()
    }

    fn take_landed_reads(&mut self, now: Time) -> Vec<(RequestId, Time)> {
        let Some(reads) = self.tier_reads.as_mut() else {
            return Vec::new();
        };
        let landed = reads.land(now);
        for (request, _) in &landed {
            let record = reads.record(*request).expect("a landed read");
            if let Some(staged) = record
                .staged
                .filter(|staged| staged.queue == LoadingQueue::Dram)
            {
                // Its hop landed: DRAM's area is free of it.
                self.tier_chains[record.partition as usize]
                    .release_area(LoadingQueue::Dram, staged);
            }
        }
        self.start_hops(now);
        landed
    }

    fn restore_landed_read(&mut self, partition: PartitionId, request: RequestId, now: Time) {
        self.ledger.take_held(request);
        let record = self
            .tier_reads
            .as_mut()
            .expect("a landed read implies tiers")
            .restore(request)
            .clone();
        if let Some(staged) = record.staged {
            self.tier_chains[partition as usize].release_area(LoadingQueue::Hbm, staged);
        }
        let physical_limit = self.prefix_cache_physical_limit(partition);
        let insert_result = self.prefix_caches[partition as usize].restore(
            record.session_id,
            record.read.positions,
            record.read.end,
            record.read.earlier,
            physical_limit,
        );
        self.record_retention(
            request,
            partition,
            record.session_id,
            record.hit.tokens,
            insert_result,
            now,
            PrefixCacheRetentionReason::TierRead,
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
                tier: HBM_TIER,
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
                cache_used_units: None,
                ready: None,
            });
            resolved_prefill =
                resolved_prefill.with_cache_return_metadata(take_result.return_metadata);
        }
        self.ledger.set_prefill_context(request, resolved_prefill);
        self.reserve(request, partition, footprint, now);
        if let (Some(session_id), Some(reads)) =
            (resolved_prefill.session_id(), self.tier_reads.as_mut())
        {
            let declared = u64::from(resolved_prefill.declared_prefix_tokens());
            reads.admit(
                partition,
                request,
                declared,
                u64::from(resolved_prefill.resident_prefix_tokens()),
            );
            self.tier_chains[partition as usize].cut(session_id, declared);
            self.log_tier_events(partition, request, now);
        }
    }

    fn resolved_prefill_context(&self, request: RequestId) -> ResolvedPrefillContext {
        self.ledger
            .prefill_context(request)
            .expect("resolved prefill context must exist for an admitted fresh request")
    }

    fn release_retaining_prefix(&mut self, request: RequestId, partition: PartitionId, now: Time) {
        KvCacheStore::release_retaining_prefix(self, request, partition, now);
    }

    fn release_handing_back(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        now: Time,
    ) -> Option<HandBack> {
        let round = self.release_retaining_round(request, partition, now)?;
        Some(HandBack {
            request,
            session_id: round.session_id,
            prompt_end: round.prompt_end,
            shared: round.shared,
        })
    }

    fn hand_back_context(
        &mut self,
        partition: PartitionId,
        hand_back: HandBack,
        output_tokens: u32,
        now: Time,
    ) {
        let HandBack {
            request,
            session_id,
            prompt_end,
            shared,
        } = hand_back;
        let tokens = prompt_end + u64::from(output_tokens.saturating_sub(1));
        if tokens > prompt_end {
            let physical_limit = self.prefix_cache_physical_limit(partition);
            let insert_result = self.prefix_caches[partition as usize].hand_back(
                session_id,
                self.accounting.resume_positions(prompt_end, tokens),
                prompt_end,
                physical_limit,
            );
            self.record_retention(
                request,
                partition,
                session_id,
                tokens,
                insert_result,
                now,
                PrefixCacheRetentionReason::HandBack,
            );
        }
        self.store_round(
            partition,
            request,
            FinishedRound {
                session_id,
                prompt_end,
                tokens,
                shared,
            },
            now,
        );
    }
}

/// A session request's round as it left HBM: where its prompt ended, the
/// context it left, and what it shared with the session's previous context.
#[derive(Clone, Copy, Debug)]
struct FinishedRound {
    session_id: u32,
    prompt_end: u64,
    tokens: u64,
    shared: u64,
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
            tier_chains: Vec::new(),
            tier_reads: None,
            tier_warm_start: false,
            tiers_reports: None,
            journal: PrefixCacheJournal::new(prefix_cache_logger),
            sampler,
            accounting,
        }
    }

    /// Put `specs`'s host tiers, fastest first, behind each partition's
    /// prefix cache, storing `bytes` per context token and state on each GPU.
    pub(crate) fn with_prefix_tiers(
        mut self,
        specs: &[TierSpec],
        bytes: TierBytes,
        options: TierStateOptions,
    ) -> Self {
        if !specs.is_empty() {
            let retention = self.accounting.state_retention();
            let logging = self.journal.is_logging();
            self.tier_chains = (0..self.partitions.len())
                .map(|_| {
                    let mut chain = TierChain::new(specs, bytes, retention, options);
                    if logging {
                        chain.record_events();
                    }
                    chain
                })
                .collect();
            self.tier_reads = Some(TierReads::new(self.partitions.len(), specs.len()));
        }
        self
    }

    /// Stage tier reads through loading areas of `fraction` of HBM and, with
    /// an SSD behind DRAM, of DRAM (`TierChain::set_loading_area`). HBM's
    /// area is also kept free of admission and retained prefixes: a
    /// partition holds at least that much.
    pub(crate) fn with_loading_area(mut self, fraction: f64) -> Self {
        assert!(!self.tier_chains.is_empty(), "a loading area needs a tier");
        let bytes_per_unit = self.accounting.bytes_per_unit();
        for (partition, chain) in self.tier_chains.iter_mut().enumerate() {
            let area = (self.partitions[partition].capacity_units() as f64 * fraction) as u64;
            self.ledger.set_loading_area(partition as PartitionId, area);
            chain.set_loading_area(fraction, area * bytes_per_unit);
        }
        self
    }

    /// Treat a declared prefix longer than anything the run stored for the
    /// session, by more than one token, as a conversation that began before
    /// the run: its context is in the slowest tier, as a long-lived session's
    /// would be in steady state. (A stored context lacks its last output
    /// token, whose KV the next request computes, so that request may declare
    /// one token more.)
    pub(crate) fn with_tier_warm_start(mut self) -> Self {
        self.tier_warm_start = true;
        self
    }

    /// Leave each partition's [`TiersReport`] in `reports` when the store is
    /// dropped.
    pub(crate) fn with_tiers_reports(mut self, reports: SharedTiersReports) -> Self {
        self.tiers_reports = Some(reports);
        self
    }

    /// Each partition's tiers now; empty without tiers.
    pub(crate) fn tiers_reports(&self) -> Vec<TiersReport> {
        let Some(reads) = &self.tier_reads else {
            return Vec::new();
        };
        self.tier_chains
            .iter()
            .enumerate()
            .map(|(partition, chain)| {
                let stats = reads.stats(partition as PartitionId);
                TiersReport {
                    bytes: chain.bytes(),
                    session_requests: stats.session_requests,
                    declared_tokens: stats.declared_tokens,
                    hbm_hit_tokens: stats.hbm_hit_tokens,
                    resident_tokens_at_admission: stats.resident_tokens,
                    warm_start_reads: stats.warm_start_reads,
                    warm_start_tokens: stats.warm_start_tokens,
                    rereads: stats.rereads,
                    loading: chain.loading_report(),
                    tiers: (0..chain.len())
                        .map(|tier| {
                            chain.report(
                                tier,
                                stats.tier_hit_requests[tier],
                                stats.tier_hit_tokens[tier],
                            )
                        })
                        .collect(),
                }
            })
            .collect()
    }

    /// Hold `units` of `partition` for `request`'s read of `kv_tokens`; what
    /// the hold pushes out of HBM is returned for a victim tier, which takes
    /// it after the read touched its entry.
    fn hold_for_read(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        units: u64,
        kv_tokens: u64,
        now: Time,
    ) -> Vec<LeavingContext> {
        self.ledger.hold(request, partition, units, kv_tokens);
        let evictions = self.trim_prefix_cache_to_physical_slack(partition);
        let mut evicted = Vec::new();
        for eviction in evictions {
            evicted.extend(self.journal_mutation(
                request,
                partition,
                now,
                eviction,
                PrefixCacheEventKind::Evict(PrefixCacheEvictionReason::ActiveKvPressure),
                0,
            ));
        }
        evicted
    }

    /// Hop the reads waiting in each partition's DRAM loading area on to
    /// HBM's, in landing order, while HBM's area has room.
    fn start_hops(&mut self, now: Time) {
        for partition in 0..self.tier_chains.len() {
            let partition = partition as PartitionId;
            loop {
                let reads = self.tier_reads.as_ref().expect("a hop implies tiers");
                let Some((request, record)) = reads.next_hop(partition) else {
                    break;
                };
                let staged = record.staged.expect("a hop is staged");
                let chain = &self.tier_chains[partition as usize];
                if !chain.area_fits(LoadingQueue::Hbm, staged) {
                    break;
                }
                let (session_id, hit_tokens, hold_units) =
                    (record.session_id, record.hit.tokens, record.hold_units);
                let evicted = self.hold_for_read(request, partition, hold_units, hit_tokens, now);
                let chain = &mut self.tier_chains[partition as usize];
                chain.take_area(LoadingQueue::Hbm, staged);
                let ready = chain.hop(session_id, hit_tokens, staged, now);
                for context in evicted {
                    chain.take_hbm_eviction(context);
                }
                self.tier_reads
                    .as_mut()
                    .expect("a hop implies tiers")
                    .start_hop(partition, ready);
                self.log_tier_events(partition, request, now);
            }
        }
    }

    /// Where a read of the session's `declared` prefix on `partition` would
    /// come from, and whether it is a pre-run context.
    fn tier_lookup(
        &self,
        partition: PartitionId,
        session_id: u32,
        declared: u64,
    ) -> Option<(TierHit, bool)> {
        let chain = self.tier_chains.get(partition as usize)?;
        let reads = self.tier_reads.as_ref()?;
        if self.tier_warm_start && declared > reads.stored(partition, session_id) + 1 {
            let tokens = self.accounting.resume_positions(declared, declared).end();
            return (tokens > 0).then_some((
                TierHit {
                    tier: chain.len() - 1,
                    tokens,
                },
                true,
            ));
        }
        chain.lookup(session_id, declared).map(|hit| (hit, false))
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
        self.record_pressure_evictions(request, partition, now, evictions);
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
        self.drop_carried_states(request, partition);
    }

    /// Forget `request`'s prefill context, and with it the states it carried
    /// from its session's entry: a context released without retention leaves
    /// them nowhere to return to.
    fn drop_carried_states(&mut self, request: RequestId, partition: PartitionId) {
        if let Some(session_id) = self
            .ledger
            .take_prefill_context(request)
            .and_then(|context| context.session_id())
        {
            self.prefix_caches[partition as usize].take_carried(session_id);
        }
    }

    pub(crate) fn release_retaining_prefix(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        now: Time,
    ) {
        if let Some(round) = self.release_retaining_round(request, partition, now) {
            self.store_round(partition, request, round, now);
        }
    }

    /// Release `request`, retaining a session request's context in HBM;
    /// returns its round for the host tiers.
    fn release_retaining_round(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        now: Time,
    ) -> Option<FinishedRound> {
        let resolved_prefill = self.ledger.prefill_context(request);
        let prompt_tokens =
            resolved_prefill.map_or(0, |value| u64::from(value.post_prefill_context_tokens()));
        let retained_tokens = self.partitions[partition as usize]
            .decode_current_kv(request)
            .unwrap_or(prompt_tokens);
        let session_id = resolved_prefill.and_then(ResolvedPrefillContext::session_id);
        let shared_tokens =
            resolved_prefill.map_or(0, |value| u64::from(value.declared_prefix_tokens()));
        let cache_return_metadata =
            resolved_prefill.and_then(ResolvedPrefillContext::cache_return_metadata);
        let carried = session_id
            .map(|session_id| self.prefix_caches[partition as usize].take_carried(session_id))
            .unwrap_or_default();
        self.release(request, partition);
        let session_id = session_id?;
        self.retain_prefix(
            request,
            partition,
            session_id,
            prompt_tokens,
            retained_tokens,
            shared_tokens,
            carried,
            cache_return_metadata,
            now,
            PrefixCacheRetentionReason::RequestComplete,
        );
        Some(FinishedRound {
            session_id,
            prompt_end: prompt_tokens,
            tokens: retained_tokens,
            shared: shared_tokens,
        })
    }

    /// Write `request`'s finished round through to the host tiers.
    fn store_round(
        &mut self,
        partition: PartitionId,
        request: RequestId,
        round: FinishedRound,
        now: Time,
    ) {
        if let Some(reads) = self.tier_reads.as_mut() {
            reads.note_stored(partition, round.session_id, round.tokens);
        }
        if let Some(chain) = self.tier_chains.get_mut(partition as usize) {
            chain.store_round(
                round.session_id,
                RoundContext {
                    positions: self
                        .accounting
                        .resume_positions(round.prompt_end, round.tokens),
                    prompt_end: round.prompt_end,
                    shared: round.shared,
                },
            );
            self.log_tier_events(partition, request, now);
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
        self.drop_carried_states(request, partition);
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
        let shared_tokens =
            resolved_prefill.map_or(0, |value| u64::from(value.declared_prefix_tokens()));
        if let Some(session_id) = session_id {
            let carried = self.prefix_caches[partition as usize].take_carried(session_id);
            self.retain_prefix(
                request,
                partition,
                session_id,
                prompt_tokens,
                kv_tokens,
                shared_tokens,
                carried,
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

    /// Journal evictions active KV forced on the prefix cache.
    fn record_pressure_evictions(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        now: Time,
        evictions: Vec<PrefixCacheMutation>,
    ) {
        for eviction in evictions {
            self.record_mutation(
                request,
                partition,
                now,
                eviction,
                PrefixCacheEventKind::Evict(PrefixCacheEvictionReason::ActiveKvPressure),
                0,
            );
        }
    }

    /// Journal one mutation receipt, and hand an entry the cache evicted to
    /// a victim tier behind it.
    fn record_mutation(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        now: Time,
        mutation: PrefixCacheMutation,
        event_kind: PrefixCacheEventKind,
        retain_requested_tokens: u64,
    ) {
        let evicted = self.journal_mutation(
            request,
            partition,
            now,
            mutation,
            event_kind,
            retain_requested_tokens,
        );
        if let (Some(evicted), Some(chain)) =
            (evicted, self.tier_chains.get_mut(partition as usize))
        {
            chain.take_hbm_eviction(evicted);
            self.log_tier_events(partition, request, now);
        }
    }

    /// Journal the entry changes `partition`'s tiers made on behalf of
    /// `request`.
    fn log_tier_events(&mut self, partition: PartitionId, request: RequestId, now: Time) {
        let Some(chain) = self.tier_chains.get_mut(partition as usize) else {
            return;
        };
        for event in chain.take_events() {
            let tier = chain.tier_name(event.tier);
            self.journal
                .record_tier_event(request, partition, now, tier, event);
        }
    }

    /// Journal one mutation receipt; returns the entry it evicted, if any.
    fn journal_mutation(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        now: Time,
        mut mutation: PrefixCacheMutation,
        event_kind: PrefixCacheEventKind,
        retain_requested_tokens: u64,
    ) -> Option<LeavingContext> {
        let evicted = matches!(
            mutation.kind,
            PrefixCacheMutationKind::CapacityEviction
                | PrefixCacheMutationKind::ReplacementPolicyEviction
        );
        let removed = mutation.removed.take();
        self.journal.record_mutation(
            request,
            partition,
            now,
            mutation,
            event_kind,
            retain_requested_tokens,
        );
        removed.filter(|_| evicted)
    }

    /// Journal what a retention did, hand what HBM let go of to a victim
    /// tier, and record a retention that kept nothing.
    #[allow(clippy::too_many_arguments)]
    fn record_retention(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        session_id: u32,
        tokens: u64,
        insert_result: PrefixCacheInsertResult,
        now: Time,
        retain_reason: PrefixCacheRetentionReason,
    ) {
        for mutation in insert_result.mutations {
            let event_kind = PrefixCacheJournal::retain_event_kind(mutation.kind, retain_reason);
            self.record_mutation(request, partition, now, mutation, event_kind, tokens);
        }
        if let Some(unkept) = insert_result.unkept {
            if let Some(chain) = self.tier_chains.get_mut(partition as usize) {
                chain.take_hbm_eviction(unkept);
                self.log_tier_events(partition, request, now);
            }
        }
        if insert_result.retained_tokens == 0 {
            let cache_used = self.prefix_caches[partition as usize].used_tokens();
            self.journal.record(PrefixCacheEvent {
                tier: HBM_TIER,
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
                cache_used_units: None,
                ready: None,
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn retain_prefix(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        session_id: u32,
        prompt_tokens: u64,
        tokens: u64,
        shared_tokens: u64,
        carried: Carried,
        cache_return_metadata: Option<PrefixCacheReturnMetadata>,
        now: Time,
        retain_reason: PrefixCacheRetentionReason,
    ) {
        let physical_limit = self.prefix_cache_physical_limit(partition);
        let insert_result = self.prefix_caches[partition as usize].insert_sharing(
            session_id,
            prompt_tokens,
            tokens,
            shared_tokens,
            carried,
            physical_limit,
            cache_return_metadata,
        );
        self.record_retention(
            request,
            partition,
            session_id,
            tokens,
            insert_result,
            now,
            retain_reason,
        );
    }
}

impl Drop for KvCacheStore {
    fn drop(&mut self) {
        if let Some(reports) = &self.tiers_reports {
            *reports.borrow_mut() = self.tiers_reports();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::rc::Rc;

    use arrow_array::{StringArray, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::tempdir;

    use crate::common::SessionInput;

    use crate::worker::kv::shared::kv_accounting::KvAccounting;
    use crate::worker::kv::shared::state_retention::StateTag;
    use crate::worker::kv::shared::tier_chain::{LeavingContext, TierWrites};
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

    /// A plain-token store of `capacity` with host tiers of 1000 B per token
    /// behind it: `gb` holds `gb * 1e6` tokens, read at `gb_per_s * 1e3`
    /// tokens per ms.
    fn tiered_store(capacity: u64, tiers: &[(&'static str, f64, f64, TierWrites)]) -> KvCacheStore {
        let specs: Vec<TierSpec> = tiers
            .iter()
            .map(|&(name, capacity_gb, read_gb_per_s, writes)| TierSpec {
                name,
                capacity_gb,
                read_gb_per_s,
                writes,
            })
            .collect();
        KvCacheStore::new(
            1,
            KvAccounting::tokens_for_test(capacity),
            capped_prefix_cache(capacity, PrefixCachePolicy::Lru),
            None,
            None,
        )
        .with_prefix_tiers(
            &specs,
            TierBytes {
                per_token: 1000,
                state: 0,
            },
            TierStateOptions::default(),
        )
    }

    fn session(session_id: u32, declared_prefix_tokens: u32) -> SessionInput {
        SessionInput::Session {
            session_id,
            session_start_time: Time::ZERO,
            declared_prefix_tokens,
        }
    }

    /// Admit request `id` of `session_input` with `fresh` new tokens and one
    /// output, as admission would.
    fn admit(kv_store: &mut KvCacheStore, id: u32, session_input: SessionInput, fresh: u32) {
        let resolved = kv_store.preview_prefill_context(0, fresh, session_input);
        let footprint =
            kv_store.footprint(RequestId(id), resolved.post_prefill_context_tokens(), 1);
        kv_store.reserve_prefill_context(RequestId(id), 0, resolved, footprint, Time::ZERO);
    }

    fn tier_hit(kv_store: &KvCacheStore, session_id: u32, declared: u64) -> Option<(usize, u64)> {
        kv_store.tier_chains[0]
            .lookup(session_id, declared)
            .map(|hit| (hit.tier, hit.tokens))
    }

    #[test]
    fn a_victim_tier_receives_what_hbm_evicts_or_cannot_keep() {
        let mut kv_store = tiered_store(100, &[("dram", 1.0, 1.0, TierWrites::Evictions)]);
        let retain = |kv_store: &mut KvCacheStore, session_id: u32, tokens: u64| {
            kv_store.retain_prefix(
                RequestId(session_id),
                0,
                session_id,
                tokens,
                tokens,
                0,
                Carried::default(),
                None,
                Time::ZERO,
                PrefixCacheRetentionReason::RequestComplete,
            );
        };
        retain(&mut kv_store, 1, 80);
        assert_eq!(tier_hit(&kv_store, 1, 80), None);
        // Session 2's context pushes session 1 out. Session 3's is larger
        // than HBM: it pushes 2 out, HBM keeps only its first 100 tokens, and
        // the whole context leaves HBM.
        retain(&mut kv_store, 2, 60);
        retain(&mut kv_store, 3, 120);
        assert_eq!(tier_hit(&kv_store, 1, 80), Some((0, 80)));
        assert_eq!(tier_hit(&kv_store, 2, 60), Some((0, 60)));
        assert_eq!(tier_hit(&kv_store, 3, 120), Some((0, 120)));
        assert_eq!(kv_store.prefix_caches[0].peek(3, 120), 100);
    }

    #[test]
    fn a_finished_context_is_written_through_and_a_victim_entry_cut_at_admission() {
        let mut kv_store = tiered_store(
            1_000,
            &[
                ("dram", 1.0, 1.0, TierWrites::Evictions),
                ("ssd", 1.0, 1.0, TierWrites::EveryContext),
            ],
        );
        admit(&mut kv_store, 1, session(7, 0), 300);
        kv_store.release_retaining_prefix(RequestId(1), 0, Time::ZERO);
        assert_eq!(
            tier_hit(&kv_store, 7, 300),
            Some((1, 300)),
            "SSD is written through"
        );
        kv_store.tier_chains[0].take_hbm_eviction(LeavingContext {
            session_id: 7,
            positions: kv_store.accounting.resume_positions(300, 300),
            end: StateTag::default(),
            earlier: Vec::new(),
        });
        assert_eq!(tier_hit(&kv_store, 7, 300), Some((0, 300)));
        // The next request shares only 200 tokens: DRAM's entry is cut there.
        admit(&mut kv_store, 2, session(7, 200), 50);
        assert_eq!(tier_hit(&kv_store, 7, 300), Some((1, 300)));
        assert_eq!(tier_hit(&kv_store, 7, 200), Some((0, 200)));
    }

    fn plan(
        kv_store: &KvCacheStore,
        id: u32,
        session_input: SessionInput,
        fresh: u32,
        now: Time,
    ) -> Option<TierReadPlan<KvFootprint>> {
        kv_store.plan_tier_read(0, RequestId(id), fresh, 1, session_input, now)
    }

    /// Complete request `id` of `session_input` with `fresh` new tokens.
    fn complete(kv_store: &mut KvCacheStore, id: u32, session_input: SessionInput, fresh: u32) {
        admit(kv_store, id, session_input, fresh);
        kv_store.release_retaining_prefix(RequestId(id), 0, Time::ZERO);
    }

    #[test]
    fn a_read_holds_its_request_until_its_context_is_put_back() {
        let mut kv_store = tiered_store(10_000, &[("dram", 1.0, 1.0, TierWrites::EveryContext)]);
        complete(&mut kv_store, 1, session(1, 0), 6_000);
        kv_store.prefix_caches[0].shrink_to(0);
        let read = plan(&kv_store, 2, session(1, 6_000), 10, Time::ZERO).expect("DRAM holds more");
        assert_eq!(
            (read.tier, read.hit_tokens, read.resident_tokens),
            (0, 6_000, 0)
        );
        assert_eq!(read.hold.reserved_units(), 6_010 + 1);
        kv_store.start_tier_read(0, RequestId(2), session(1, 6_000), read, Time::ZERO);
        assert_eq!(kv_store.ledger.partition_held(0), 6_011);
        // 6000 tokens at 1000 tokens per ms.
        assert_eq!(kv_store.next_tier_landing(), Some(Time::from_ms(6.0)));
        assert!(kv_store.take_landed_reads(Time::from_ms(5.0)).is_empty());
        assert_eq!(
            kv_store.take_landed_reads(Time::from_ms(7.0)),
            [(RequestId(2), Time::from_ms(6.0))]
        );
        kv_store.restore_landed_read(0, RequestId(2), Time::from_ms(7.0));
        assert_eq!(kv_store.ledger.partition_held(0), 0);
        assert_eq!(kv_store.prefix_caches[0].peek(1, 6_000), 6_000);
        assert!(plan(&kv_store, 2, session(1, 6_000), 10, Time::from_ms(7.0)).is_none());
        admit(&mut kv_store, 2, session(1, 6_000), 10);
        let stats = kv_store.tier_reads.as_ref().unwrap().stats(0);
        assert_eq!(
            (stats.tier_hit_requests[0], stats.tier_hit_tokens[0]),
            (1, 6_000)
        );
        assert_eq!(stats.rereads, 0);
    }

    /// A plain-token store of 100,000 tokens with a 10% loading area (10,000
    /// tokens) and tiers of 1 B per token: DRAM holds 1M tokens read at 1000
    /// per ms, written through, or with `ssd` as HBM's victim tier in front of
    /// a written-through SSD of 10M read at 100 per ms. Sessions 1.. each
    /// complete a context of `contexts[i]` tokens, which only the tiers keep.
    fn staged_store(contexts: &[u32], ssd: bool) -> KvCacheStore {
        let dram = TierSpec {
            name: "dram",
            capacity_gb: 1e-3,
            read_gb_per_s: 1e-3,
            writes: if ssd {
                TierWrites::Evictions
            } else {
                TierWrites::EveryContext
            },
        };
        let ssd_tier = TierSpec {
            name: "ssd",
            capacity_gb: 1e-2,
            read_gb_per_s: 1e-4,
            writes: TierWrites::EveryContext,
        };
        let specs = if ssd {
            vec![dram, ssd_tier]
        } else {
            vec![dram]
        };
        let mut kv_store = KvCacheStore::new(
            1,
            KvAccounting::tokens_for_test(100_000),
            capped_prefix_cache(100_000, PrefixCachePolicy::Lru),
            None,
            None,
        )
        .with_prefix_tiers(
            &specs,
            TierBytes {
                per_token: 1,
                state: 0,
            },
            TierStateOptions::default(),
        )
        .with_loading_area(0.1);
        for (i, &tokens) in contexts.iter().enumerate() {
            let id = i as u32 + 1;
            complete(&mut kv_store, 100 + id, session(id, 0), tokens);
        }
        kv_store.prefix_caches[0].shrink_to(0);
        kv_store
    }

    /// Start request `id`'s read of session `id`'s `declared` context if its
    /// loading area has room.
    fn stage(kv_store: &mut KvCacheStore, id: u32, declared: u32, now: Time) -> bool {
        let read = plan(kv_store, id, session(id, declared), 10, now).expect("a tier holds it");
        if !kv_store.has_loading_room(0, &read) {
            return false;
        }
        kv_store.start_tier_read(0, RequestId(id), session(id, declared), read, now);
        true
    }

    #[test]
    fn a_loading_area_is_kept_free_and_a_staged_read_holds_only_what_it_lands() {
        let mut kv_store = staged_store(&[6_000], false);
        assert!(kv_store.stages_tier_reads());
        assert_eq!(kv_store.ledger.partition_held(0), 10_000);
        let read = plan(&kv_store, 1, session(1, 6_000), 10, Time::ZERO).unwrap();
        assert_eq!(
            read.hold.reserved_units(),
            6_000,
            "not its 6,011-token footprint"
        );
        kv_store.start_tier_read(0, RequestId(1), session(1, 6_000), read, Time::ZERO);
        assert_eq!(
            kv_store.ledger.partition_held(0),
            10_000,
            "inside the area it adds nothing"
        );
        let reports = kv_store.tiers_reports();
        let json = serde_json::to_value(&reports[0]).unwrap();
        assert_eq!(
            (&json["fraction"], &json["hbm_area"], &json["dram_area"]),
            (
                &serde_json::json!(0.1),
                &serde_json::json!(10_000),
                &serde_json::json!(0)
            ),
            "kv_cache.json's tiers carry the fraction and both areas in tier tokens"
        );
    }

    #[test]
    fn a_staged_read_holds_its_area_until_its_context_is_put_back() {
        let mut kv_store = staged_store(&[6_000, 4_000, 4_000], false);
        assert!(stage(&mut kv_store, 1, 6_000, Time::ZERO));
        assert!(stage(&mut kv_store, 2, 4_000, Time::ZERO));
        assert!(!stage(&mut kv_store, 3, 4_000, Time::ZERO));
        // 6000 tokens at 1000 per ms land at 6 ms; landed, they still hold
        // the area until admission puts them back.
        let at = Time::from_ms(6.0);
        assert_eq!(kv_store.take_landed_reads(at), [(RequestId(1), at)]);
        assert!(!stage(&mut kv_store, 3, 4_000, at));
        kv_store.restore_landed_read(0, RequestId(1), at);
        assert!(stage(&mut kv_store, 3, 4_000, at));
    }

    #[test]
    fn a_read_larger_than_the_area_starts_when_it_is_empty() {
        let mut kv_store = staged_store(&[25_000, 100], false);
        assert!(stage(&mut kv_store, 1, 25_000, Time::ZERO));
        assert!(!stage(&mut kv_store, 2, 100, Time::ZERO));
        assert_eq!(
            kv_store.ledger.partition_held(0),
            25_000,
            "past the area it holds what it lands"
        );
        let at = Time::from_ms(25.0);
        kv_store.take_landed_reads(at);
        assert!(!stage(&mut kv_store, 2, 100, at));
        kv_store.restore_landed_read(0, RequestId(1), at);
        assert!(stage(&mut kv_store, 2, 100, at));
    }

    #[test]
    fn an_ssd_read_lands_in_dram_and_hops_on_to_hbm_when_its_area_has_room() {
        // DRAM's area is 10% of its 1M tokens. Session 1's 6000 tokens are
        // in DRAM and land in HBM's area; sessions 2 and 3 (5000 each) are
        // only in SSD.
        let mut kv_store = staged_store(&[6_000, 5_000, 5_000], true);
        kv_store.tier_chains[0].take_hbm_eviction(LeavingContext {
            session_id: 1,
            positions: kv_store.accounting.resume_positions(6_000, 6_000),
            end: StateTag::default(),
            earlier: Vec::new(),
        });
        assert!(stage(&mut kv_store, 1, 6_000, Time::ZERO));
        assert!(stage(&mut kv_store, 2, 5_000, Time::ZERO));
        assert!(stage(&mut kv_store, 3, 5_000, Time::ZERO));
        assert_eq!(
            kv_store.tiers_reports()[0].loading.unwrap().dram_area,
            100_000
        );
        assert_eq!(
            kv_store.ledger.partition_held(0),
            10_000,
            "SSD reads hold no HBM yet"
        );
        // SSD moves 100 tokens per ms: session 2 is in DRAM at 50 ms, 3 at
        // 100 ms. Session 1 (landed at 6 ms) fills HBM's area, so neither hops.
        assert_eq!(
            kv_store.take_landed_reads(Time::from_ms(100.0)),
            [(RequestId(1), Time::from_ms(6.0))]
        );
        assert_eq!(kv_store.next_tier_landing(), None);
        kv_store.restore_landed_read(0, RequestId(1), Time::from_ms(100.0));
        // The next landing pass hops session 2 on (5 ms on DRAM's channel),
        // then session 3 behind it, in landing order.
        assert!(kv_store.take_landed_reads(Time::from_ms(101.0)).is_empty());
        assert_eq!(kv_store.next_tier_landing(), Some(Time::from_ms(106.0)));
        assert_eq!(
            kv_store.take_landed_reads(Time::from_ms(111.0)),
            [
                (RequestId(2), Time::from_ms(106.0)),
                (RequestId(3), Time::from_ms(111.0))
            ]
        );
        assert_eq!(
            kv_store.tier_chains[0].counters(0).loads,
            1,
            "hops are no DRAM reads"
        );
    }

    #[test]
    fn tier_changes_are_logged_beside_hbm_and_reported_when_the_store_drops() {
        let log_directory = tempdir().unwrap();
        let logger =
            PrefixCacheLogger::open(log_directory.path(), "main", crate::common::WorkerId(0))
                .unwrap();
        let reports = SharedTiersReports::default();
        let mut kv_store = KvCacheStore::new(
            1,
            KvAccounting::tokens_for_test(10_000),
            capped_prefix_cache(10_000, PrefixCachePolicy::Lru),
            None,
            Some(logger),
        )
        .with_prefix_tiers(
            &[TierSpec {
                name: "dram",
                capacity_gb: 1.0,
                read_gb_per_s: 1.0,
                writes: TierWrites::EveryContext,
            }],
            TierBytes {
                per_token: 1000,
                state: 0,
            },
            TierStateOptions::default(),
        )
        .with_tiers_reports(Rc::clone(&reports));
        complete(&mut kv_store, 1, session(1, 0), 6_000);
        kv_store.prefix_caches[0].shrink_to(0);
        let read = plan(&kv_store, 2, session(1, 6_000), 10, Time::ZERO).unwrap();
        kv_store.start_tier_read(0, RequestId(2), session(1, 6_000), read, Time::ZERO);
        kv_store.take_landed_reads(Time::from_ms(6.0));
        kv_store.restore_landed_read(0, RequestId(2), Time::from_ms(6.0));
        admit(&mut kv_store, 2, session(1, 6_000), 10);
        drop(kv_store);

        let path = log_directory
            .path()
            .join("raw/prefix_cache_event/worker_main_0.parquet");
        let batch = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
            .unwrap()
            .build()
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let strings = |name: &str| -> Vec<String> {
            let column = batch
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            (0..batch.num_rows())
                .map(|row| column.value(row).to_string())
                .collect()
        };
        let rows: Vec<(String, String)> = strings("tier")
            .into_iter()
            .zip(strings("reason"))
            .filter(|(tier, _)| tier == "dram")
            .collect();
        assert_eq!(
            rows,
            [("dram", "write-through"), ("dram", "read")]
                .map(|(tier, reason)| (tier.to_string(), reason.to_string()))
        );
        let ready = batch
            .column_by_name("ready_ms")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .unwrap();
        let read_row = strings("reason")
            .iter()
            .position(|reason| reason == "read")
            .unwrap();
        assert_eq!(ready.value(read_row), 6.0);

        let reports = reports.borrow();
        assert_eq!(reports.len(), 1);
        let dram = &reports[0].tiers[0];
        assert_eq!(
            (dram.used_units, dram.hit_requests, dram.hit_tokens),
            (6_000, 1, 6_000)
        );
        assert_eq!((dram.counters.stored_entries, dram.counters.loads), (1, 1));
        assert_eq!(reports[0].session_requests, 2);
    }

    #[test]
    fn a_landed_context_evicted_before_admission_is_read_again() {
        let mut kv_store = tiered_store(10_000, &[("dram", 1.0, 1.0, TierWrites::EveryContext)]);
        complete(&mut kv_store, 1, session(1, 0), 6_000);
        kv_store.prefix_caches[0].shrink_to(0);
        let read = plan(&kv_store, 2, session(1, 6_000), 10, Time::ZERO).unwrap();
        kv_store.start_tier_read(0, RequestId(2), session(1, 6_000), read, Time::ZERO);
        let at = Time::from_ms(6.0);
        kv_store.take_landed_reads(at);
        // Admission reaches it and puts the context back in HBM, but it is
        // evicted again before the request is admitted.
        kv_store.restore_landed_read(0, RequestId(2), at);
        kv_store.prefix_caches[0].shrink_to(0);
        let again = plan(&kv_store, 2, session(1, 6_000), 10, at).expect("read again");
        kv_store.start_tier_read(0, RequestId(2), session(1, 6_000), again, at);
        assert_eq!(kv_store.tier_reads.as_ref().unwrap().stats(0).rereads, 1);
        assert_eq!(kv_store.next_tier_landing(), Some(Time::from_ms(12.0)));
    }

    #[test]
    fn warm_start_seeds_only_a_context_the_run_never_stored() {
        let mut kv_store = tiered_store(
            10_000,
            &[
                ("dram", 1.0, 1.0, TierWrites::EveryContext),
                ("ssd", 1.0, 0.1, TierWrites::EveryContext),
            ],
        )
        .with_tier_warm_start();
        // A pre-run context: nothing stored, so it reads from the slowest tier.
        let read = plan(&kv_store, 1, session(7, 500), 10, Time::ZERO).unwrap();
        assert_eq!((read.tier, read.hit_tokens), (1, 500));
        assert_eq!(kv_store.reusable_prefix_tokens(0, 10, session(7, 500)), 500);
        kv_store.start_tier_read(0, RequestId(1), session(7, 500), read, Time::ZERO);
        let stats = kv_store.tier_reads.as_ref().unwrap().stats(0);
        assert_eq!((stats.warm_start_reads, stats.warm_start_tokens), (1, 500));
        assert_eq!(kv_store.tier_chains[0].counters(1).stored_entries, 1);
        // The next request declares one token more than the stored context
        // (the last output's KV): an ordinary hit from the fastest tier.
        complete(&mut kv_store, 2, session(8, 0), 554);
        kv_store.prefix_caches[0].shrink_to(0);
        let read = plan(&kv_store, 3, session(8, 555), 10, Time::ZERO).unwrap();
        assert_eq!((read.tier, read.hit_tokens), (0, 554));
        // A one-token context does not cover a joining request's context.
        complete(&mut kv_store, 4, session(9, 0), 1);
        let read = plan(&kv_store, 5, session(9, 300), 10, Time::ZERO).unwrap();
        assert_eq!((read.tier, read.hit_tokens), (1, 300));
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
    use crate::worker::kv::StateRetention;
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

    /// A state saved where the context ended resumes the whole context or
    /// nothing. The position is not floored to the engine block; the KV it
    /// keeps is still charged in whole blocks.
    #[test]
    fn end_state_reuse_hits_a_whole_context_or_nothing() {
        let mut kv_store = state_store(
            10_000_000,
            STATE_TOKENS,
            Some(CHECKPOINT_INTERVAL),
            Some(PrefixResumePoints::ContextEnd),
        );
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
        assert_eq!(
            kv_store.partitions[0].live_checkpoint_units(),
            0,
            "a running request keeps no state behind it"
        );
        kv_store.release_retaining_prefix(RequestId(0), 0, Time::from_ms(1.0));
        assert_eq!(kv_store.prefix_caches[0].used_tokens(), 5_000);
        assert_eq!(
            kv_store.prefix_caches[0].used_units(),
            3 * 2_048 + STATE_TOKENS
        );
        assert_eq!(kv_store.prefix_caches[0].peek(7, 5_001), 5_000);
        assert_eq!(
            kv_store.prefix_caches[0].peek(7, 4_999),
            0,
            "no state inside the context to resume from"
        );
    }

    /// A hit on an entry with an earlier state carries it while the request
    /// runs: back into the entry when the context is retained, dropped when
    /// it is released without retention.
    #[test]
    fn carried_states_return_with_the_context_or_leave_with_the_request() {
        let mut kv_store = KvCacheStore::new(
            1,
            Arc::new(
                KvAccounting::resolve(
                    &KvCacheLayout::new()
                        .with_part(KvCachePart::per_token("gqa", 1))
                        .with_part(KvCachePart::per_request("gdn", STATE_TOKENS)),
                    KvAccountingOptions {
                        resume_points: Some(PrefixResumePoints::ContextEnd),
                        state_retention: StateRetention::Latest { k: 2 },
                        ..KvAccountingOptions::default()
                    },
                    10_000_000,
                )
                .unwrap(),
            ),
            capped_prefix_cache(10_000_000, PrefixCachePolicy::Lru),
            None,
            None,
        );
        let round = |kv_store: &mut KvCacheStore, request: u32, declared: u32, retain: bool| {
            let session = SessionInput::Session {
                session_id: 7,
                session_start_time: Time::ZERO,
                declared_prefix_tokens: declared,
            };
            let resolved = kv_store.preview_prefill_context(0, 1_000, session);
            let context = resolved.post_prefill_context_tokens();
            let footprint = kv_store.footprint(RequestId(request), context, 0);
            kv_store.reserve_prefill_context(
                RequestId(request),
                0,
                resolved,
                footprint,
                Time::ZERO,
            );
            kv_store.drain_ready();
            kv_store.commit_resident(RequestId(request), 0, u64::from(context), 0);
            if retain {
                kv_store.release_retaining_prefix(RequestId(request), 0, Time::ZERO);
            } else {
                kv_store.release(RequestId(request), 0);
            }
        };
        round(&mut kv_store, 0, 0, true);
        round(&mut kv_store, 1, 1_000, true);
        // States at 1_000 and 2_000 beside 2_000 tokens.
        assert_eq!(
            kv_store.prefix_caches[0].used_units(),
            2_000 + 2 * STATE_TOKENS
        );
        assert_eq!(kv_store.prefix_caches[0].peek(7, 1_500), 1_000);
        // A third round resumes at 2_000 and is released without retention:
        // the state it carried goes with it.
        round(&mut kv_store, 2, 2_000, false);
        assert_eq!(kv_store.prefix_caches[0].used_units(), 0);
    }

    /// A decode run elsewhere hands back every output but the last on top of
    /// the prompt's context: with two states kept, the prompt's end stays a
    /// resume point inside the extended context.
    #[test]
    fn a_handed_back_context_keeps_its_prompt_end_state_when_asked() {
        for (k, inside_hit) in [(1, 0), (2, 5_000)] {
            let mut kv_store = KvCacheStore::new(
                1,
                Arc::new(
                    KvAccounting::resolve(
                        &KvCacheLayout::new()
                            .with_part(KvCachePart::per_token("gqa", 1))
                            .with_part(KvCachePart::per_request("gdn", STATE_TOKENS)),
                        KvAccountingOptions {
                            resume_points: Some(PrefixResumePoints::ContextEnd),
                            state_retention: StateRetention::Latest { k },
                            ..KvAccountingOptions::default()
                        },
                        10_000_000,
                    )
                    .unwrap(),
                ),
                capped_prefix_cache(10_000_000, PrefixCachePolicy::Lru),
                None,
                None,
            );
            let session = SessionInput::Session {
                session_id: 7,
                session_start_time: Time::ZERO,
                declared_prefix_tokens: 0,
            };
            let resolved = kv_store.preview_prefill_context(0, 5_000, session);
            let footprint = kv_store.footprint(RequestId(0), 5_000, 0);
            kv_store.reserve_prefill_context(RequestId(0), 0, resolved, footprint, Time::ZERO);
            kv_store.drain_ready();
            kv_store.commit_resident(RequestId(0), 0, 5_000, 0);
            let hand_back = kv_store
                .release_handing_back(RequestId(0), 0, Time::ZERO)
                .expect("a session request hands its context back");
            assert_eq!(
                kv_store.prefix_caches[0].used_units(),
                5_000 + STATE_TOKENS,
                "the prompt's context waits for its decode"
            );
            kv_store.hand_back_context(0, hand_back, 401, Time::from_ms(1.0));
            assert_eq!(
                kv_store.prefix_caches[0].used_units(),
                5_400 + u64::from(k) * STATE_TOKENS
            );
            assert_eq!(kv_store.prefix_caches[0].peek(7, 5_401), 5_400);
            // A next prompt keeping only part of the output.
            assert_eq!(kv_store.prefix_caches[0].peek(7, 5_200), inside_hit);
        }
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
