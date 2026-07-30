//! Orthogonal keys for KV resources and request advancement.
//!
//! `PartitionId` selects one independently-accounted KV resource. `AdvanceScope`
//! selects which resident requests advance within that resource. Keeping these
//! distinct is required by slot-based workers, where several request groups share
//! one KV partition.

use crate::common::RequestId;

/// KV resource partition. This is not the deployment-level `PoolId`.
pub type PartitionId = u16;

/// Requests affected by one KV-growth transition.
#[derive(Clone, Copy)]
pub enum AdvanceScope<'a> {
    WholePartition(PartitionId),
    /// Used by slot/speculative families once their production workers migrate;
    /// declared now so M2 does not have to reshape the KV seam.
    #[allow(dead_code)]
    RequestSubset {
        partition: PartitionId,
        request_ids: &'a [RequestId],
    },
}
