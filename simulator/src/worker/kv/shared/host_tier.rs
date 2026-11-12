//! The host half of a KV cache split between GPU and host memory (SGLang
//! HiSparse).
//!
//! The GPU side needs nothing here: the arch's `KvCacheLayout` declares what
//! the GPU keeps of a token as its per-token parts and the fixed device buffer
//! as a per-request part, and [`KvAccounting`](super::kv_accounting::KvAccounting)
//! charges both like any other layout. The host keeps each request's context,
//! and this tier bounds their sum. The store counts context tokens beside the
//! units it already ledgers -- resident decodes in
//! [`ResidentPartitionState`], reservations in the request ledger -- so the
//! host occupancy follows the same lifecycle and no second ledger can drift.

use crate::worker::kv::shared::resident_partition::ResidentPartitionState;

/// The host tier of one KV partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostKvTier {
    /// Context tokens one partition's host memory holds.
    pub host_capacity_tokens: u64,
}

impl HostKvTier {
    /// A tier of `host_bytes_per_partition` bytes holding
    /// `host_bytes_per_token` per context token.
    pub fn from_bytes(host_bytes_per_partition: u64, host_bytes_per_token: u64) -> Self {
        Self {
            host_capacity_tokens: host_bytes_per_partition / host_bytes_per_token.max(1),
        }
    }

    /// The host half of a store's `fits`: the immediate and projected-peak
    /// tests over context tokens alone, with `context_tokens` more requested.
    pub(crate) fn fits(
        &self,
        partition: &ResidentPartitionState,
        promised_context_tokens: u64,
        context_tokens: u64,
    ) -> bool {
        let reserved = promised_context_tokens + context_tokens;
        partition.resident_context_tokens() + reserved <= self.host_capacity_tokens
            && partition.projected_peak_context_tokens() + reserved <= self.host_capacity_tokens
    }

    /// Host occupancy, resident and promised, as a fraction of its capacity.
    pub(crate) fn usage(
        &self,
        partition: &ResidentPartitionState,
        promised_context_tokens: u64,
    ) -> f64 {
        (partition.resident_context_tokens() + promised_context_tokens) as f64
            / self.host_capacity_tokens.max(1) as f64
    }
}
