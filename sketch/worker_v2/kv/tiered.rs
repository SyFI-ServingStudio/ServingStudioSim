//! `TieredMemoryKv` — two-tier (fast GPU / slow offload) KV — unknown-variant blind
//! test #1.
//!
//! A thin wrapper over `FullAttnKv`: every `KvStore` + `IterWorkerKv` method DELEGATES
//! verbatim (same `Batch`/`KvAdmission`/sampler machinery), and it adds ONE new thing —
//! the `TieredKvView` read-view that reports the fast/slow resident split. The inner
//! `FullAttnKv` is built with the TOTAL capacity (fast + slow), so `fits`/`reserve`/
//! `advance` already account the enlarged budget without change; the tier threshold is a
//! pure read-side projection (`resident_by_tier`) a tier-aware cost model can charge on.
//!
//! This is the blind test's pass criterion in action: a genuinely new KV LAYOUT enters
//! as `new leaf + one narrow read-view + one census line`, touching zero existing bodies
//! and forcing zero existing impl to grow a dummy method. It composes UNCHANGED with the
//! existing `LocalPrefillDecodeAdmission<FifoOrder>` lifecycle and `UnifiedIterExecution` (census asserts the tuple).
//!
//! Superset: the crate has no offload pool, so "compressible/tiered" is modeled as a
//! capacity gain (slow tier) + a resident threshold, not a real page table. What the test
//! exercises is the SEAM — a distinct KV impl carrying a new capability — not fidelity.

use crate::common::{RequestId, Time};
use crate::log::KvSampler;
use crate::worker::admission_helpers::KvAdmission;

use super::super::shared::advance_scope::{AdvanceScope, PartitionId};
use super::full_attn::{FullAttentionKvFootprint, FullAttnKv};
use super::{IterWorkerKv, KvCapacityPressure, KvStore, TieredKvView};

pub struct TieredMemoryKv {
    inner: FullAttnKv,
    /// Fast-tier (GPU HBM) capacity in tokens-equiv; resident beyond this is "spilled"
    /// to the slow tier. Per partition (all partitions share the same tier geometry).
    fast_capacity_tokens: u64,
}

impl TieredMemoryKv {
    pub fn new(
        num_partitions: usize,
        fast_capacity_tokens: u64,
        slow_capacity_tokens: u64,
        admission: KvAdmission,
        sampler: Option<KvSampler>,
    ) -> Self {
        Self {
            // Total budget the gate sees = fast + slow. `fits` needs no change.
            inner: FullAttnKv::new(
                num_partitions,
                fast_capacity_tokens + slow_capacity_tokens,
                admission,
                sampler,
            ),
            fast_capacity_tokens,
        }
    }
}

impl KvStore for TieredMemoryKv {
    type Footprint = FullAttentionKvFootprint;

    #[inline]
    fn num_partitions(&self) -> usize {
        self.inner.num_partitions()
    }
    #[inline]
    fn footprint(&self, req: RequestId, prompt: u32, decode: u32) -> FullAttentionKvFootprint {
        self.inner.footprint(req, prompt, decode)
    }
    #[inline]
    fn fits(&self, partition: PartitionId, footprint: &FullAttentionKvFootprint) -> bool {
        self.inner.fits(partition, footprint)
    }
    #[inline]
    fn pressure(&self, partition: PartitionId) -> KvCapacityPressure {
        self.inner.pressure(partition)
    }
    #[inline]
    fn reserve(
        &mut self,
        req: RequestId,
        partition: PartitionId,
        footprint: FullAttentionKvFootprint,
    ) {
        self.inner.reserve(req, partition, footprint)
    }
    #[inline]
    fn commit_resident(
        &mut self,
        req: RequestId,
        partition: PartitionId,
        initial_kv: u64,
        remaining: u32,
    ) {
        self.inner
            .commit_resident(req, partition, initial_kv, remaining)
    }
    #[inline]
    fn release(&mut self, req: RequestId, partition: PartitionId) {
        self.inner.release(req, partition)
    }
    #[inline]
    fn advance(&mut self, scope: AdvanceScope, steps: u32) {
        self.inner.advance(scope, steps)
    }
    #[inline]
    fn sample_submit(&mut self, partition: PartitionId, now: Time) {
        self.inner.sample_submit(partition, now)
    }
}

impl IterWorkerKv for TieredMemoryKv {
    #[inline]
    fn drain_ready(&mut self) {
        self.inner.drain_ready()
    }
    #[inline]
    fn clear_prefill_admits(&mut self, partition: PartitionId) {
        self.inner.clear_prefill_admits(partition)
    }
    #[inline]
    fn has_live_decode(&self, partition: PartitionId) -> bool {
        self.inner.has_live_decode(partition)
    }
    #[inline]
    fn live_decode_count(&self, partition: PartitionId) -> u32 {
        self.inner.live_decode_count(partition)
    }
    #[inline]
    fn has_prefill_admit(&self, partition: PartitionId) -> bool {
        self.inner.has_prefill_admit(partition)
    }
    #[inline]
    fn status_active(&self, partition: PartitionId) -> u32 {
        self.inner.status_active(partition)
    }
    #[inline]
    fn release_external(&mut self, req: RequestId, current_kv: u64) -> Option<PartitionId> {
        self.inner.release_external(req, current_kv)
    }
    #[inline]
    fn prefill_admits(&self, partition: PartitionId) -> Vec<RequestId> {
        self.inner.prefill_admits(partition)
    }
    #[inline]
    fn decode_members(&self, partition: PartitionId) -> Vec<(RequestId, u64)> {
        self.inner.decode_members(partition)
    }
}

impl TieredKvView for TieredMemoryKv {
    /// The ONE new method: project the resident total onto (fast, slow). The first
    /// `fast_capacity_tokens` live in GPU HBM; the remainder is spilled. A tier-aware
    /// cost model (not wired here — see the trait's boundary note) would charge the slow
    /// portion an offload-transfer surcharge.
    #[inline]
    fn resident_by_tier(&self, partition: PartitionId) -> (u64, u64) {
        let resident = self.inner.pressure(partition).resident_tokens_equiv;
        let fast = resident.min(self.fast_capacity_tokens);
        (fast, resident - fast)
    }
}
