//! `ModelPartitionedKv` — N co-resident models sharing one GPU's KV budget
//! (interfaces doc §2.2, multi-arch). Each model gets its own resident pool; a
//! `SwitchModel` control message makes one model active, so subsequent admits land in
//! its pool.
//!
//! Implemented as a `FullAttnKv` whose PARTITIONS are the MODELS (partition ≡ `ModelId`):
//! the per-partition `Batch` machinery is exactly one-pool-per-model, so every `KvStore`
//! / `IterWorkerKv` method DELEGATES verbatim. The only additions are an `active` cursor
//! and the `ModelSwitchKv` capability that moves it. Budget sharing is a static split here
//! (each pool gets `total / num_models`); a dynamic shared budget would re-gate across
//! pools in `fits` — deferred (an accounting refinement, not an interface question).
//!
//! The admission/execution never match this concrete type: the admission constrains on
//! `ModelSwitchKv` (to route by active model) and the execution on nothing KV-specific (it reads
//! per-partition groups). That is the axis split's whole claim — a new co-serve KV drops
//! in without touching the lifecycle or the execution surface.

use crate::common::{RequestId, Time};
use crate::log::KvSampler;
use crate::worker::admission_helpers::KvAdmission;

use super::super::shared::advance_scope::{AdvanceScope, PartitionId};
use super::full_attn::{FullAttentionKvFootprint, FullAttnKv};
use super::{IterWorkerKv, KvCapacityPressure, KvStore, ModelId, ModelSwitchKv};

pub struct ModelPartitionedKv {
    /// One `Batch` partition per co-resident model.
    inner: FullAttnKv,
    /// The model new admits route to (moved by `SwitchModel`).
    active: ModelId,
    num_models: u16,
}

impl ModelPartitionedKv {
    pub fn new(
        num_models: usize,
        per_model_capacity: u64,
        admission: KvAdmission,
        sampler: Option<KvSampler>,
    ) -> Self {
        Self {
            inner: FullAttnKv::new(num_models, per_model_capacity, admission, sampler),
            active: ModelId(0),
            num_models: num_models.max(1) as u16,
        }
    }

    #[inline]
    pub fn num_models(&self) -> u16 {
        self.num_models
    }
}

impl KvStore for ModelPartitionedKv {
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

impl IterWorkerKv for ModelPartitionedKv {
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

impl ModelSwitchKv for ModelPartitionedKv {
    #[inline]
    fn set_active(&mut self, model: ModelId) {
        if model.0 < self.num_models {
            self.active = model;
        }
    }
    #[inline]
    fn active(&self) -> ModelId {
        self.active
    }
}
