//! `ModeledPrefixCacheKv` — full-attention KV with a prefix-cache front (interfaces doc
//! §2.2, F1).
//!
//! A thin wrapper over `FullAttnKv`: it reuses the exact same `Batch`/`KvAdmission`
//! machinery and adds a partition-local modeled prefix probe. The probe returns the
//! match and the footprint from one snapshot; `PrefixPrefillDecodeAdmission` chooses a partition and
//! `FullAttnKv::reserve` makes that placement sticky.
//!
//! Superset: the crate has no radix/prefix pool, so the "match" is a modeled hit
//! fraction of the prompt, NOT a real tree lookup. Cached tokens share the same
//! capacity gate as request KV but are charged per request; real shared-block
//! accounting needs the unbuilt radix index/refcount/replacement ledger.

use crate::common::{RequestId, Time};
use crate::log::KvSampler;
use crate::worker::admission_helpers::KvAdmission;

use super::super::shared::advance_scope::{AdvanceScope, PartitionId};
use super::full_attn::{FullAttentionKvFootprint, FullAttnKv};
use super::{IterWorkerKv, KvCapacityPressure, KvStore, PrefixCacheKv, PrefixPlacementProbe};

pub struct ModeledPrefixCacheKv {
    inner: FullAttnKv,
    /// Modeled prefix hit per partition (0 = never, 100 = the full prompt).
    prefix_hit_pct_by_partition: Vec<u32>,
}

impl ModeledPrefixCacheKv {
    pub fn new(
        num_partitions: usize,
        kv_capacity: u64,
        admission: KvAdmission,
        sampler: Option<KvSampler>,
        prefix_hit_pct_by_partition: Vec<u32>,
    ) -> Self {
        assert_eq!(
            prefix_hit_pct_by_partition.len(),
            num_partitions,
            "ModeledPrefixCacheKv needs one modeled hit percentage per partition"
        );
        Self {
            inner: FullAttnKv::new(num_partitions, kv_capacity, admission, sampler),
            prefix_hit_pct_by_partition: prefix_hit_pct_by_partition
                .into_iter()
                .map(|percentage| percentage.min(100))
                .collect(),
        }
    }
}

impl KvStore for ModeledPrefixCacheKv {
    type Footprint = FullAttentionKvFootprint;

    #[inline]
    fn num_partitions(&self) -> usize {
        self.inner.num_partitions()
    }

    /// Generic admissions do not claim a prefix-cache hit. `PrefixPrefillDecodeAdmission` uses
    /// `probe_prefix` below so match and footprint come from the chosen partition.
    fn footprint(&self, _req: RequestId, prompt: u32, decode: u32) -> FullAttentionKvFootprint {
        FullAttentionKvFootprint {
            prompt,
            decode,
            cached_prefix: 0,
            fixed_state: 0,
        }
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

impl IterWorkerKv for ModeledPrefixCacheKv {
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

impl PrefixCacheKv for ModeledPrefixCacheKv {
    fn probe_prefix(
        &self,
        partition: PartitionId,
        _req: RequestId,
        prompt: u32,
        decode: u32,
    ) -> PrefixPlacementProbe<FullAttentionKvFootprint> {
        let matched_tokens =
            prompt.saturating_mul(self.prefix_hit_pct_by_partition[partition as usize]) / 100;
        PrefixPlacementProbe {
            matched_tokens,
            footprint: FullAttentionKvFootprint {
                prompt,
                decode,
                cached_prefix: matched_tokens,
                fixed_state: 0,
            },
        }
    }
}
