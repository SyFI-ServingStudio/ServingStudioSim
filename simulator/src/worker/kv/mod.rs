//! KV resource axis.
//!
//! `KvStore` is the common resource lifecycle. `IterWorkerKv` is the narrow
//! capability required by the whole-iteration worker family. Membership is
//! exposed through generic visitors, not owned snapshots, so abstraction does
//! not add hot-path allocation or leak `FullAttnPartitionState`.

use crate::common::{PrefixInput, RequestId, Time};
use crate::worker::shared::advance_scope::{AdvanceScope, PartitionId};

use self::prefix_cache::PrefixCacheLease;

mod full_attn;
mod full_attn_partition;
mod prefix_cache;

pub use full_attn::FullAttnKv;
pub use prefix_cache::PrefixCachePolicy;

/// KV-owned runtime resolution of one immutable prefix declaration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixResolution {
    session_id: Option<u32>,
    resident_prefix_tokens: u32,
    prefill_compute_tokens: u32,
    initial_context_tokens: u32,
    cache_lease: Option<PrefixCacheLease>,
}

impl PrefixResolution {
    pub(crate) fn new(
        prefix: PrefixInput,
        resident_prefix_tokens: u32,
        fresh_prompt_tokens: u32,
    ) -> Self {
        let declared_prefix_tokens = prefix.declared_prefix_tokens();
        debug_assert!(resident_prefix_tokens <= declared_prefix_tokens);
        let recomputed_prefix_tokens = declared_prefix_tokens - resident_prefix_tokens;
        let prefill_compute_tokens = fresh_prompt_tokens
            .checked_add(recomputed_prefix_tokens)
            .expect("resolved prefill token count overflow");
        let initial_context_tokens = resident_prefix_tokens
            .checked_add(prefill_compute_tokens)
            .expect("resolved initial context token count overflow");
        Self {
            session_id: prefix.session_id(),
            resident_prefix_tokens,
            prefill_compute_tokens,
            initial_context_tokens,
            cache_lease: None,
        }
    }

    pub fn resident_prefix_tokens(self) -> u32 {
        self.resident_prefix_tokens
    }

    pub fn prefill_compute_tokens(self) -> u32 {
        self.prefill_compute_tokens
    }

    pub fn initial_context_tokens(self) -> u32 {
        self.initial_context_tokens
    }

    fn with_cache_lease(mut self, cache_lease: Option<PrefixCacheLease>) -> Self {
        self.cache_lease = cache_lease;
        self
    }

    fn cache_lease(self) -> Option<PrefixCacheLease> {
        self.cache_lease
    }
}

pub trait KvStore {
    type Footprint;

    fn num_partitions(&self) -> usize;
    fn footprint(&self, request: RequestId, prompt: u32, decode: u32) -> Self::Footprint;
    fn fits(&self, partition: PartitionId, footprint: &Self::Footprint) -> bool;
    fn reserve(&mut self, request: RequestId, partition: PartitionId, footprint: Self::Footprint);
    fn commit_resident(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        initial_kv: u64,
        remaining: u32,
    );
    fn advance(&mut self, scope: AdvanceScope<'_>, steps: u32);
    fn release(&mut self, request: RequestId, partition: PartitionId);
    fn sample_submit(&mut self, partition: PartitionId, now: Time);
}

/// Prefix-aware refinement of the same KV owner.
///
/// Admission first obtains a pure plan, applies its token/KV gates, then reserves
/// that exact plan. The simulator is single-threaded, so no cache mutation can
/// occur between those two calls. Runtime hit/miss facts remain in this store.
pub trait PrefixKv: KvStore {
    /// Locate reusable retained prefix KV before admission applies its fallback
    /// placement policy. Active requests are deliberately excluded because a
    /// destructive cache lease does not permit inter-request sharing.
    fn retained_prefix_partition(&self, prefix: PrefixInput) -> Option<PartitionId>;
    fn plan_prefix(
        &self,
        partition: PartitionId,
        fresh_prompt_tokens: u32,
        prefix: PrefixInput,
    ) -> PrefixResolution;
    fn reserve_prefix(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        resolution: PrefixResolution,
        footprint: Self::Footprint,
    );
    fn prefix_resolution(&self, request: RequestId) -> PrefixResolution;
    fn release_retaining_prefix(&mut self, request: RequestId, partition: PartitionId);

    fn prefill_pair(&self, request: RequestId) -> (u32, u32) {
        let resolution = self.prefix_resolution(request);
        (
            resolution.resident_prefix_tokens(),
            resolution.prefill_compute_tokens(),
        )
    }

    fn prefill_compute_tokens(&self, request: RequestId) -> u32 {
        self.prefix_resolution(request).prefill_compute_tokens()
    }

    fn initial_context_tokens(&self, request: RequestId) -> u64 {
        u64::from(self.prefix_resolution(request).initial_context_tokens())
    }
}

pub trait IterWorkerKv: KvStore {
    fn drain_ready(&mut self);
    fn clear_prefill_admits(&mut self, partition: PartitionId);
    fn has_live_decode(&self, partition: PartitionId) -> bool;
    fn live_decode_count(&self, partition: PartitionId) -> u32;
    fn has_prefill_admit(&self, partition: PartitionId) -> bool;
    fn status_active(&self, partition: PartitionId) -> u32;
    fn release_external(&mut self, request: RequestId, current_kv: u64) -> Option<PartitionId>;

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

/// Additional KV lifecycle required by a PD prefill worker.
pub trait HandoffKv: IterWorkerKv {
    fn hold(&mut self, partition: PartitionId, request: RequestId, kv_tokens: u64);
    fn complete_handoff(&mut self, request: RequestId);
}
