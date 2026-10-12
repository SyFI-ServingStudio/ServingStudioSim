//! The host half of a KV cache split between GPU and host memory (SGLang
//! HiSparse), shared by every store that can carry one.
//!
//! The GPU keeps a fixed device buffer per request beside the per-token part
//! the store's capacity is counted in; the host keeps each request's context.
//! A store charges the buffer through [`ResidentPartitionState`]'s fixed
//! per-request charge (beside any recurrent state it already charges there),
//! so the host occupancy is derived from the existing ledgers -- resident and
//! promised charge minus the fixed part -- and no second ledger can drift.

use crate::worker::kv::shared::request_ledger::RequestLedger;
use crate::worker::kv::shared::resident_partition::ResidentPartitionState;
use crate::worker::shared::advance_scope::PartitionId;

/// The host tier, in the store's token units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostKvTier {
    /// Fixed GPU charge per resident or promised request, in units of the
    /// per-token GPU bytes the store's capacity is counted in.
    pub device_tokens_per_request: u64,
    /// Context tokens one partition's host memory holds.
    pub host_capacity_tokens: u64,
}

impl HostKvTier {
    /// Context tokens promised on `partition`: each promise's charge less the
    /// store's whole fixed charge per request (this tier's buffer plus any
    /// recurrent state).
    pub(crate) fn promised_context_tokens(
        ledger: &RequestLedger,
        partition: PartitionId,
        fixed_tokens_per_request: u64,
    ) -> u64 {
        ledger.partition_promised(partition)
            - fixed_tokens_per_request * u64::from(ledger.partition_promised_count(partition))
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
