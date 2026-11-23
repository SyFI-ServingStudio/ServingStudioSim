//! KV resource axis.
//!
//! `KvStore` is the common resource lifecycle. `IterWorkerKv` is the narrow
//! capability required by the whole-iteration worker family. Membership is
//! exposed through generic visitors, not owned snapshots, so abstraction does
//! not add hot-path allocation or leak partition state.

use crate::common::{RequestId, SessionInput, Time};
use crate::worker::shared::advance_scope::{AdvanceScope, PartitionId};

use self::shared::prefix_cache::PrefixCacheReturnMetadata;

mod kv_cache_store;
mod shared;

pub use kv_cache_store::KvCacheStore;
pub(crate) use shared::balanced_retention::RetainedScale;
pub use shared::balanced_retention::{BalancedRetention, BalancedRetentionSpec};
pub(crate) use shared::kv_accounting::{KvAccounting, KvAccountingOptions, SharedKvAccounting};
pub use shared::kv_accounting::{KvOverhead, PartCharge, PartPadding, PrefixResumePoints};
pub use shared::kv_cache_report::KvCacheReport;
pub use shared::prefix_cache::{PrefixCacheConfig, PrefixCacheMode, PrefixCachePolicy};
pub use shared::state_retention::StateRetention;
pub use shared::tier_chain::{
    LoadingQueue, StagedCharge, TierBytes, TierSpec, TierStateOptions, TierWrites, DRAM_TIER,
};
pub use shared::tier_reads::{SharedTiersReports, TiersReport};

/// KV-owned runtime facts for one request's prefill on one partition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResolvedPrefillContext {
    session_id: Option<u32>,
    declared_prefix_tokens: u32,
    resident_prefix_tokens: u32,
    prefill_tokens_to_compute: u32,
    post_prefill_context_tokens: u32,
    processed_prefill_tokens: u32,
    active_chunk_tokens: u32,
    cache_return_metadata: Option<PrefixCacheReturnMetadata>,
}

impl ResolvedPrefillContext {
    pub(crate) fn new(
        session_input: SessionInput,
        resident_prefix_tokens: u32,
        fresh_prompt_tokens: u32,
    ) -> Self {
        let declared_prefix_tokens = session_input.declared_prefix_tokens();
        debug_assert!(resident_prefix_tokens <= declared_prefix_tokens);
        let mut missing_prefix_tokens = declared_prefix_tokens - resident_prefix_tokens;
        let mut resident_prefix_tokens = resident_prefix_tokens;
        // A prompt whose prefix is entirely resident still has to run one token
        // through the model: the last position is what produces the first output
        // logits, and no engine can emit a token it never computed. Real serving
        // stacks recompute that last token rather than serving a zero-length
        // prefill, and the coding-agent trace does contain such rounds (a
        // resumed turn that adds nothing new). Borrow the token from the
        // resident prefix instead of adding one on top, so
        // `hit + computed == fresh + declared` still holds exactly.
        if fresh_prompt_tokens == 0 && missing_prefix_tokens == 0 && resident_prefix_tokens > 0 {
            resident_prefix_tokens -= 1;
            missing_prefix_tokens = 1;
        }
        let prefill_tokens_to_compute = fresh_prompt_tokens
            .checked_add(missing_prefix_tokens)
            .expect("resolved prefill token count overflow");
        // Cache hit/miss changes compute, never the logical context after
        // prefill: the model sees the full declared prefix plus fresh prompt.
        let post_prefill_context_tokens = fresh_prompt_tokens
            .checked_add(declared_prefix_tokens)
            .expect("post-prefill context token count overflow");
        Self {
            session_id: session_input.session_id(),
            declared_prefix_tokens,
            resident_prefix_tokens,
            prefill_tokens_to_compute,
            post_prefill_context_tokens,
            processed_prefill_tokens: 0,
            active_chunk_tokens: prefill_tokens_to_compute,
            cache_return_metadata: None,
        }
    }

    /// `None` for a standalone request — nothing to look up or retain.
    pub(crate) fn session_id(self) -> Option<u32> {
        self.session_id
    }

    pub fn resident_prefix_tokens(self) -> u32 {
        self.resident_prefix_tokens
    }

    pub fn declared_prefix_tokens(self) -> u32 {
        self.declared_prefix_tokens
    }

    pub fn prefill_tokens_to_compute(self) -> u32 {
        self.prefill_tokens_to_compute
    }

    pub fn post_prefill_context_tokens(self) -> u32 {
        self.post_prefill_context_tokens
    }

    /// Context and append length exposed to the model in this iteration.
    pub fn active_chunk(self) -> (u32, u32) {
        (
            self.resident_prefix_tokens + self.processed_prefill_tokens,
            self.active_chunk_tokens,
        )
    }

    pub fn remaining_prefill_tokens(self) -> u32 {
        self.prefill_tokens_to_compute
            .saturating_sub(self.processed_prefill_tokens)
    }

    fn schedule_chunk(&mut self, chunk_tokens: u32) {
        debug_assert!(chunk_tokens <= self.remaining_prefill_tokens());
        self.active_chunk_tokens = chunk_tokens;
    }

    fn complete_chunk(&mut self) {
        self.processed_prefill_tokens = self
            .processed_prefill_tokens
            .checked_add(self.active_chunk_tokens)
            .expect("processed prefill token count overflow");
        self.active_chunk_tokens = 0;
    }

    fn with_cache_return_metadata(
        mut self,
        cache_return_metadata: Option<PrefixCacheReturnMetadata>,
    ) -> Self {
        self.cache_return_metadata = cache_return_metadata;
        self
    }

    fn cache_return_metadata(self) -> Option<PrefixCacheReturnMetadata> {
        self.cache_return_metadata
    }
}

pub trait KvStore {
    type Footprint;

    fn num_partitions(&self) -> usize;
    fn footprint(
        &self,
        request: RequestId,
        post_prefill_context_tokens: u32,
        remaining_output_tokens: u32,
    ) -> Self::Footprint;
    fn fits(&self, partition: PartitionId, footprint: &Self::Footprint) -> bool;
    fn reserve(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        footprint: Self::Footprint,
        now: Time,
    );
    fn commit_resident(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        post_prefill_context_tokens: u64,
        remaining_output_tokens: u32,
    );
    fn advance(&mut self, scope: AdvanceScope<'_>, steps: u32);
    fn release(&mut self, request: RequestId, partition: PartitionId);
    fn sample_submit(&mut self, partition: PartitionId, now: Time);
}

/// A session request whose decode runs elsewhere, from its release until its
/// context comes back ([`PrefixKv::hand_back_context`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandBack {
    request: RequestId,
    session_id: u32,
    /// Where its prompt ended.
    prompt_end: u64,
    /// Tokens it shared with the session's previous context.
    shared: u64,
}

/// Prefix-aware refinement of the same KV owner.
///
/// Admission first obtains a pure plan, applies its token/KV gates, then reserves
/// that exact plan. The simulator is single-threaded, so no cache mutation can
/// occur between those two calls. Runtime hit/miss facts remain in this store.
pub trait PrefixKv: KvStore {
    /// Locate reusable retained prefix KV before admission applies its fallback
    /// placement policy. Active requests are deliberately excluded because a
    /// destructive cache ownership transfer does not permit inter-request sharing.
    fn retained_prefix_partition(&self, session_input: SessionInput) -> Option<PartitionId>;
    fn preview_prefill_context(
        &self,
        partition: PartitionId,
        fresh_prompt_tokens: u32,
        session_input: SessionInput,
    ) -> ResolvedPrefillContext;
    fn reserve_prefill_context(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        resolved_prefill: ResolvedPrefillContext,
        footprint: Self::Footprint,
        now: Time,
    );
    fn resolved_prefill_context(&self, request: RequestId) -> ResolvedPrefillContext;
    fn release_retaining_prefix(&mut self, request: RequestId, partition: PartitionId, now: Time);
    /// [`Self::release_retaining_prefix`] of a request whose decode runs
    /// elsewhere: HBM retains its prompt's context, and what comes back is
    /// for [`Self::hand_back_context`]; `None` for a standalone request.
    fn release_handing_back(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        now: Time,
    ) -> Option<HandBack>;
    /// The decode run elsewhere hands the context back after `output_tokens`
    /// outputs: every output but the last, whose KV the next round computes,
    /// extends the prompt's context in HBM, and host tiers are written the
    /// whole round.
    fn hand_back_context(
        &mut self,
        partition: PartitionId,
        hand_back: HandBack,
        output_tokens: u32,
        now: Time,
    );

    /// How much of `session_input`'s declared prefix is resident *right now*,
    /// wherever it lives. This is what a cache-aware pending order ranks by;
    /// `declared_prefix_tokens` is only its upper bound. A session with no
    /// retained partition has nothing to reuse and scores zero; a pinned prefix
    /// is resident on every partition.
    ///
    /// Two hash lookups, no mutation — cheap enough for the admission hot path.
    fn resident_prefix_tokens(&self, fresh_prompt_tokens: u32, session_input: SessionInput) -> u32 {
        let partition = match session_input {
            SessionInput::Standalone => return 0,
            SessionInput::PinnedPrefix { .. } => 0,
            SessionInput::Session { .. } => {
                let Some(partition) = self.retained_prefix_partition(session_input) else {
                    return 0;
                };
                partition
            }
        };
        self.preview_prefill_context(partition, fresh_prompt_tokens, session_input)
            .resident_prefix_tokens()
    }

    fn prefill_tokens_to_compute(&self, request: RequestId) -> u32 {
        self.resolved_prefill_context(request)
            .prefill_tokens_to_compute()
    }

    fn post_prefill_context_tokens(&self, request: RequestId) -> u64 {
        u64::from(
            self.resolved_prefill_context(request)
                .post_prefill_context_tokens(),
        )
    }
}

/// Partial-prefill refinement for the whole-iteration worker family.
///
/// Admission reserves one policy-defined waiting-request footprint, while it
/// exposes one hard-capped chunk at a time to execution. Historical workers use
/// the full request footprint. A bounded-future scheduler instead reserves the
/// source-defined waiting-request estimate and checks physical allocation again
/// immediately before every decode step.
pub trait ChunkedPrefillKv: PrefixKv + IterWorkerKv {
    fn bounded_future_footprint(
        &self,
        request: RequestId,
        post_prefill_context_tokens: u32,
        remaining_output_tokens: u32,
        max_future_tokens: u32,
        page_size: u32,
    ) -> Self::Footprint;
    /// `decode_step_tokens` is how far each running decode advances in the
    /// same step as this prefill, 0 when prefill steps run no decode; the
    /// admission then also leaves room for that step.
    fn fits_bounded_future(
        &self,
        partition: PartitionId,
        footprint: &Self::Footprint,
        max_future_tokens: u32,
        new_token_ratio: f64,
        decode_step_tokens: u32,
    ) -> bool;
    /// Shortfall, in KV units, for the next decode step, after shrinking the
    /// prefix cache. `step_tokens` is how far one decode can advance per step:
    /// 1 for autoregressive decode, the verify width for a speculating engine.
    fn prepare_next_decode(
        &mut self,
        partition: PartitionId,
        page_size: u32,
        step_tokens: u32,
        now: Time,
    ) -> u64;
    fn visit_decode_states(&self, partition: PartitionId, visitor: impl FnMut(RequestId, u64, u32));
    fn reserve_chunked_prefill_context(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        resolved_prefill: ResolvedPrefillContext,
        footprint: Self::Footprint,
        now: Time,
    );
    fn schedule_prefill_chunk(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        chunk_tokens: u32,
    );
    fn complete_prefill_chunk(&mut self, request: RequestId);
    fn finish_chunked_prefill(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        remaining_output_tokens: u32,
    );
}

pub trait IterWorkerKv: KvStore {
    fn drain_ready(&mut self);
    fn clear_prefill_admits(&mut self, partition: PartitionId);
    fn has_live_decode(&self, partition: PartitionId) -> bool;
    fn live_decode_count(&self, partition: PartitionId) -> u32;
    fn has_prefill_admit(&self, partition: PartitionId) -> bool;
    fn status_active(&self, partition: PartitionId) -> u32;
    /// Fraction of the partition's KV capacity that running requests hold or
    /// have reserved (retained prefix KV is reclaimable, so it does not count),
    /// the analogue of vLLM's per-engine `kv_cache_usage`.
    fn kv_usage(&self, partition: PartitionId) -> f64;
    fn release_external(&mut self, request: RequestId, current_kv: u64) -> Option<PartitionId>;
    /// Retained tokens of `session_input`'s prefix on this store's best
    /// partition (0 when none): what pool-level session-affine placement reads.
    fn retained_prefix_tokens(&self, session_input: SessionInput) -> u32;

    /// Static-dispatch iteration over this iteration's fresh prefills.
    fn visit_prefill_admits(&self, partition: PartitionId, visitor: impl FnMut(RequestId));

    /// Static-dispatch iteration over live `(request, current_kv)` pairs.
    fn visit_decode_members(&self, partition: PartitionId, visitor: impl FnMut(RequestId, u64));
}

/// KV observations required by the layer-wise AFD attention pipeline.
pub trait SlotPipelineKv: KvStore {
    fn current_kv(&self, partition: PartitionId, request: RequestId) -> Option<u64>;
    fn estimated_peak(&self, partition: PartitionId) -> u64;
    fn has_reservation(&self, request: RequestId) -> bool;
    fn request_kv_weight(&self, request: RequestId) -> u64;
}

/// A read a host tier would serve before a queued request's admission.
pub struct TierReadPlan<F> {
    /// Fastest first.
    pub tier: usize,
    /// How long the tier's channel would make the read wait now.
    pub queue_wait: Time,
    /// Tokens the read resumes from, and of them what HBM already holds.
    pub hit_tokens: u64,
    pub resident_tokens: u64,
    /// What the read holds in HBM until its context is put back: unstaged,
    /// from its start, the request's whole footprint; staged, from when it
    /// lands in HBM's loading area, the KV it lands. Both with every earlier
    /// state it moves.
    pub hold: F,
    /// Staged loading: what it holds in the loading areas; `None` unstaged.
    pub staged: Option<StagedCharge>,
}

/// Prefix refinement with host DRAM/SSD tiers behind HBM, read when admission
/// reaches a request, as vLLM's KV connector does (`WAITING_FOR_REMOTE_KVS`).
/// The store decides what a read moves and how long it takes; admission only
/// decides when to start one and when to admit its request. Without tiers no
/// request needs a read. With a loading area (staged loading, not vLLM) reads
/// go through loading areas the store keeps free, each landing queue in
/// order: admission starts a read only while its area has room.
pub trait TieredPrefixKv: ChunkedPrefillKv {
    /// The longer of HBM's resident prefix and what `partition`'s tiers would
    /// read back: what a cache-aware order of that partition ranks a request
    /// by.
    fn reusable_prefix_tokens(
        &self,
        partition: PartitionId,
        fresh_prompt_tokens: u32,
        session_input: SessionInput,
    ) -> u32;
    /// The read the request needs before admission: a tier holds more of its
    /// declared prefix than `partition`'s HBM. `None`: admit it from HBM.
    #[allow(clippy::too_many_arguments)]
    fn plan_tier_read(
        &self,
        partition: PartitionId,
        request: RequestId,
        fresh_prompt_tokens: u32,
        remaining_output_tokens: u32,
        session_input: SessionInput,
        now: Time,
    ) -> Option<TierReadPlan<Self::Footprint>>;
    /// Whether reads stage through loading areas.
    fn stages_tier_reads(&self) -> bool;
    /// Staged loading: whether the planned read may start, in an area it
    /// fits in or an empty one.
    fn has_loading_room(
        &self,
        partition: PartitionId,
        plan: &TierReadPlan<Self::Footprint>,
    ) -> bool;
    /// Start the planned read, holding its footprint until the context is
    /// put back. A staged read takes its share of the loading area it lands
    /// in first; one landing in DRAM holds nothing in HBM until it hops on,
    /// in landing order, once HBM's area has room.
    fn start_tier_read(
        &mut self,
        partition: PartitionId,
        request: RequestId,
        session_input: SessionInput,
        plan: TierReadPlan<Self::Footprint>,
        now: Time,
    );
    /// When the next read lands.
    fn next_tier_landing(&self) -> Option<Time>;
    /// The requests whose reads landed in HBM by `now`, in landing order,
    /// each with its landing time; then the hops that now fit start.
    fn take_landed_reads(&mut self, now: Time) -> Vec<(RequestId, Time)>;
    /// Put a landed request's context back into HBM, ending its hold and
    /// giving its share of HBM's loading area back, just before admission
    /// resolves its prefix.
    fn restore_landed_read(&mut self, partition: PartitionId, request: RequestId, now: Time);
    /// Sessions whose context `horizon` more tokens written into
    /// `partition`'s tiers would evict from every tier holding it, the next
    /// victims first (an LRU tier's stack distance to eviction). Empty
    /// without tiers.
    fn tier_sessions_at_risk(&self, partition: PartitionId, horizon: u64) -> Vec<u32>;
    /// Whether HBM's evictions go to `partition`'s fastest tier (a victim
    /// tier), so a context HBM holds survives HBM's own evictions.
    fn tiers_take_hbm_evictions(&self, partition: PartitionId) -> bool;
}

/// Additional KV lifecycle required by a PD prefill worker.
pub trait HandoffKv: IterWorkerKv {
    fn hold(&mut self, partition: PartitionId, request: RequestId, kv_tokens: u64);
    fn complete_handoff(&mut self, request: RequestId, now: Time);
}
