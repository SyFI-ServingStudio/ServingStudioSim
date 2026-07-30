//! Shared vocabulary that belongs to no single component (interfaces doc §1).
//!
//! `PartitionId` keys the KV **resource** axis; `AdvanceScope` keys the **request**
//! axis. They are orthogonal: AFD is the proof (one shard partition, three slot
//! groupings). Barebone degenerates to `WholePartition(0)` everywhere.

use crate::common::RequestId;

/// KV resource partition: single pool = 0 / DP-attn group / per-model.
/// NOT the deployment `PoolId` (that is the worker pool). Deliberately not "pool".
pub type PartitionId = u16;

/// Which requests one operation touches. Resource questions (fits/reserve/commit/
/// release/pressure) use a bare `PartitionId`; request advance / input rendering
/// use this. `RequestSubset` carries its `partition` explicitly so an empty slot still
/// names its resource partition and KV can validate membership.
pub enum AdvanceScope<'a> {
    WholePartition(PartitionId),
    RequestSubset {
        partition: PartitionId,
        request_ids: &'a [RequestId],
    },
}
