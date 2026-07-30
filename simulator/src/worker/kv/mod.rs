//! KV resource axis.
//!
//! `KvStore` is the common resource lifecycle. `IterWorkerKv` is the narrow
//! capability required by the whole-iteration worker family. Membership is
//! exposed through generic visitors, not owned snapshots, so abstraction does
//! not add hot-path allocation or leak `FullAttnPartitionState`.

use crate::common::{RequestId, Time};
use crate::worker::shared::advance_scope::{AdvanceScope, PartitionId};

mod full_attn;
mod full_attn_partition;

pub use full_attn::FullAttnKv;

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
    fn drop_held(&mut self, request: RequestId);
}
