//! KV resource contracts and implementations.
//!
//! Shared resource vocabulary lives here; concrete cache/accounting models live
//! in sibling files. A Worker selects one implementation without making
//! Admission depend on that implementation's private representation.

use crate::common::RequestId;

mod full_attn;

pub(super) use full_attn::FullAttnKv;

/// One independently-accounted KV resource pool. This is not the deployment
/// `PoolId`: barebone uses `0`, DP can use a shard, and multi-model workers can
/// use a model-local partition.
pub(super) type PartitionId = u16;

/// The requests affected by a KV lifecycle operation, orthogonal to the resource
/// partition. The caller owns explicit membership; KV only verifies resource
/// ownership and applies the requested transition.
pub(super) enum Grouping<'a> {
    /// Every resident request in one partition.
    Partition(PartitionId),
    /// A caller-owned request subset. Keeping `partition` explicit makes an empty
    /// lockstep group unambiguous and avoids inferring ownership from its first item.
    Reqs {
        partition: PartitionId,
        request_ids: &'a [RequestId],
    },
}
