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
    /// A partition where only some resident requests move, or move by different
    /// distances: the AFD slot pipeline advances one slot's group, and
    /// speculative decode advances each acceptance-length bucket separately.
    RequestSubset {
        partition: PartitionId,
        request_ids: &'a [RequestId],
    },
}
