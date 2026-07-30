//! `FullAttnKv` — the full-attention KV impl (interfaces doc §2, mapping row
//! `FullKv = KvStore<FullLayout, PrivateOnly, SingleKvPool>`).
//!
//! Wraps the real shared leaf `Batch` (`admission_helpers.rs`), owns the `promised`
//! reserve ledger + `request_to_partition` map + `KvAdmission` gate + occupancy
//! `KvSampler`. External calls verified against real source: `Batch::new`,
//! `advance_decodes`/`advance_subset`/`finalize_to_decode`/`release`/`iter_decoding`/
//! `decode_current_kv`/`projected_peak_kv`; `KvPool.{active_kv,kv_capacity}`;
//! `KvAdmission::try_admit(group, group_promised, p, d)`; `KvSampler::submit`.

use std::collections::HashMap;

use crate::common::{RequestId, Time};
use crate::log::{KvSampler, KvSubmit};
use crate::worker::admission_helpers::{Batch, KvAdmission};

use super::super::shared::advance_scope::{AdvanceScope, PartitionId};
use super::{
    ChunkedPrefillKv, HandoffKv, IterWorkerKv, KvCapacityPressure, KvStore, SlotPipelineKv,
    SpeculativeKv,
};

/// Opaque-to-Admission footprint. Carrying the components (not a pre-summed u64)
/// lets `fits` call the real `KvAdmission::try_admit(.., p, d)` verbatim.
///
/// `cached_prefix` and `fixed_state` are BOTH "resident the moment the request is
/// admitted, never advanced", so `fits` folds them into the prompt side. They are two
/// fields rather than one shared slot because independent wrappers own them
/// (`ModeledPrefixCacheKv` the former, `HybridStateKv` the latter) and a stack of both
/// must set both — one slot
/// would silently make the wrappers mutually exclusive.
pub struct FullAttentionKvFootprint {
    // pub(super) so sibling wrappers in the kv module can inject their own component —
    // the whole `fits` path already threads the sum, it is just all-zero here.
    pub(super) prompt: u32,
    pub(super) decode: u32,
    /// Prompt tokens already resident via a prefix-cache hit (`PrefixCacheKv`): shared
    /// blocks, accounted distinctly from the request's fresh `prompt`.
    pub(super) cached_prefix: u32,
    /// Fixed-size resident state that never grows under `advance` — a hybrid model's
    /// recurrent/conv state (`HybridKvView` kind 1). Charged once at admit.
    pub(super) fixed_state: u32,
}

impl FullAttentionKvFootprint {
    /// The resident-but-never-advanced components. Independent wrappers each contribute
    /// one, so they SUM (they are not alternatives).
    #[inline]
    fn resident_fixed(&self) -> u32 {
        self.cached_prefix + self.fixed_state
    }

    #[inline]
    fn reserved_tokens(&self) -> u64 {
        u64::from(self.prompt + self.resident_fixed() + self.decode)
    }
}

pub struct FullAttnKv {
    /// One `Batch` per KV partition. Barebone builds exactly one (partition 0).
    batches: Vec<Batch>,
    /// Reserve ledger: admitted-but-not-yet-resident `(PartitionId, reserved tokens)`.
    promised: HashMap<RequestId, (PartitionId, u64)>,
    /// Held ledger (PD prefill): prefilled KV awaiting the decode-side pull. Like
    /// `promised`, it is capacity that gates admission but is not in a decode set.
    held: HashMap<RequestId, (PartitionId, u64)>,
    /// Which KV partition each admitted request lives in.
    request_to_partition: HashMap<RequestId, PartitionId>,
    /// Draft tokens computed but not yet committed by target verification.
    tentative_proposals: HashMap<RequestId, (PartitionId, u32)>,
    /// The strict capacity gate (`KvAdmission::Strict`), distributed out of `WorkerConfig`.
    admission: KvAdmission,
    /// Per-worker occupancy sampler; `None` when no log dir.
    sampler: Option<KvSampler>,
}

impl FullAttnKv {
    pub fn new(
        num_partitions: usize,
        kv_capacity: u64,
        admission: KvAdmission,
        sampler: Option<KvSampler>,
    ) -> Self {
        Self {
            batches: (0..num_partitions as u16)
                .map(|partition| Batch::new(partition, kv_capacity))
                .collect(),
            promised: HashMap::new(),
            held: HashMap::new(),
            request_to_partition: HashMap::new(),
            tentative_proposals: HashMap::new(),
            admission,
            sampler,
        }
    }

    #[inline]
    fn partition_promised(&self, partition: PartitionId) -> u64 {
        self.promised
            .values()
            .filter(|(reserved_partition, _)| *reserved_partition == partition)
            .map(|(_, tokens)| *tokens)
            .sum()
    }

    /// Held KV (PD prefill) for a partition — capacity that gates admission just like
    /// `promised`. Empty (→ 0) for every non-PD-prefill worker.
    #[inline]
    fn partition_held(&self, partition: PartitionId) -> u64 {
        self.held
            .values()
            .filter(|(held_partition, _)| *held_partition == partition)
            .map(|(_, tokens)| *tokens)
            .sum()
    }

    /// `fits` plus `extra_occupied` tokens that occupy THIS pool but are invisible to
    /// the inner `Batch`. A sibling wrapper needs this when it keeps its own resident
    /// ledger over the same memory: `HybridStateKv`'s recurrent state leaves `promised` at
    /// `commit_resident` and never enters `active_kv`, so without this it would stop
    /// gating the moment the request goes resident. Same role as `held`, one level up.
    pub(super) fn fits_with_extra_occupied(
        &self,
        partition: PartitionId,
        footprint: &FullAttentionKvFootprint,
        extra_occupied: u64,
    ) -> bool {
        let batch = &self.batches[partition as usize];
        let group_promised =
            self.partition_promised(partition) + self.partition_held(partition) + extra_occupied;
        self.admission.try_admit(
            batch,
            group_promised,
            footprint.prompt + footprint.resident_fixed(),
            footprint.decode,
        )
    }

    /// `pressure` with the same sibling-ledger correction as `fits_with_extra_occupied`.
    pub(super) fn pressure_with_extra_resident(
        &self,
        partition: PartitionId,
        extra_resident: u64,
    ) -> KvCapacityPressure {
        let batch = &self.batches[partition as usize];
        KvCapacityPressure {
            resident_tokens_equiv: batch.kv.active_kv + extra_resident,
            reserved_tokens_equiv: self.partition_promised(partition),
            capacity_tokens_equiv: batch.kv.kv_capacity,
        }
    }
}

impl KvStore for FullAttnKv {
    type Footprint = FullAttentionKvFootprint;

    #[inline]
    fn num_partitions(&self) -> usize {
        self.batches.len()
    }

    #[inline]
    fn footprint(&self, _req: RequestId, prompt: u32, decode: u32) -> FullAttentionKvFootprint {
        // Plain full attention: no prefix cache and no recurrent state, so both
        // resident-fixed components are 0. `ModeledPrefixCacheKv` / `HybridStateKv`
        // turn them on.
        FullAttentionKvFootprint {
            prompt,
            decode,
            cached_prefix: 0,
            fixed_state: 0,
        }
    }

    /// Delegates to the REAL strict gate — no reproduction, no drift. Held KV (PD
    /// prefill) counts against capacity alongside `promised`; it is 0 for other workers.
    #[inline]
    fn fits(&self, partition: PartitionId, footprint: &FullAttentionKvFootprint) -> bool {
        // No sibling ledger over this pool: plain full attention owns all of it.
        self.fits_with_extra_occupied(partition, footprint, 0)
    }

    #[inline]
    fn pressure(&self, partition: PartitionId) -> KvCapacityPressure {
        self.pressure_with_extra_resident(partition, 0)
    }

    fn reserve(
        &mut self,
        req: RequestId,
        partition: PartitionId,
        footprint: FullAttentionKvFootprint,
    ) {
        self.promised
            .insert(req, (partition, footprint.reserved_tokens()));
        self.request_to_partition.insert(req, partition);
    }

    fn commit_resident(
        &mut self,
        req: RequestId,
        partition: PartitionId,
        initial_kv: u64,
        remaining: u32,
    ) {
        // Clear any still-live reservation. Iter family already drained it into
        // prefill_admits (no-op here); AFD family keeps the reservation in
        // `promised` until a layer boundary, so this is where it clears.
        self.promised.remove(&req);
        self.batches[partition as usize].finalize_to_decode(req, initial_kv, remaining);
    }

    fn release(&mut self, req: RequestId, partition: PartitionId) {
        self.promised.remove(&req);
        self.tentative_proposals.remove(&req);
        let current_kv = self.batches[partition as usize]
            .decode_current_kv(req)
            .unwrap_or(0);
        self.batches[partition as usize].release(req, current_kv);
        self.request_to_partition.remove(&req);
    }

    fn advance(&mut self, scope: AdvanceScope, steps: u32) {
        for _ in 0..steps {
            match scope {
                AdvanceScope::WholePartition(partition) => {
                    self.batches[partition as usize].advance_decodes();
                }
                AdvanceScope::RequestSubset {
                    partition,
                    request_ids,
                } => {
                    self.batches[partition as usize].advance_subset(request_ids);
                }
            }
        }
    }

    fn sample_submit(&mut self, partition: PartitionId, now: Time) {
        if self.sampler.is_none() {
            return;
        }
        let submit = KvSubmit {
            active_kv: self.batches[partition as usize].kv.active_kv,
            projected_peak: self.batches[partition as usize].projected_peak_kv(),
            promised_kv: self.partition_promised(partition),
        };
        self.sampler
            .as_mut()
            .unwrap()
            .submit(partition, submit, now);
    }
}

impl IterWorkerKv for FullAttnKv {
    /// Barebone readiness is always true (local KV, no remote pull), so every
    /// promise drains into its partition's `prefill_admits` this tick.
    fn drain_ready(&mut self) {
        let drained: Vec<(PartitionId, RequestId)> = self
            .promised
            .iter()
            .map(|(&req, &(partition, _))| (partition, req))
            .collect();
        for (partition, req) in drained {
            self.promised.remove(&req);
            self.batches[partition as usize].prefill_admits.push(req);
        }
    }

    fn clear_prefill_admits(&mut self, partition: PartitionId) {
        self.batches[partition as usize].prefill_admits.clear();
    }

    #[inline]
    fn has_live_decode(&self, partition: PartitionId) -> bool {
        self.batches[partition as usize]
            .iter_decoding()
            .next()
            .is_some()
    }

    #[inline]
    fn live_decode_count(&self, partition: PartitionId) -> u32 {
        self.batches[partition as usize].iter_decoding().count() as u32
    }

    #[inline]
    fn has_prefill_admit(&self, partition: PartitionId) -> bool {
        !self.batches[partition as usize].prefill_admits.is_empty()
    }

    fn status_active(&self, partition: PartitionId) -> u32 {
        self.live_decode_count(partition)
            + self.promised.len() as u32
            + self.batches[partition as usize].prefill_admits.len() as u32
    }

    fn release_external(&mut self, req: RequestId, current_kv: u64) -> Option<PartitionId> {
        let partition = self.request_to_partition.remove(&req)?;
        self.tentative_proposals.remove(&req);
        let batch = &mut self.batches[partition as usize];
        batch.release(req, current_kv);
        if let Some(position) = batch.prefill_admits.iter().position(|&admit| admit == req) {
            batch.prefill_admits.swap_remove(position);
        }
        self.promised.remove(&req);
        Some(partition)
    }

    fn prefill_admits(&self, partition: PartitionId) -> Vec<RequestId> {
        self.batches[partition as usize].prefill_admits.clone()
    }

    fn decode_members(&self, partition: PartitionId) -> Vec<(RequestId, u64)> {
        self.batches[partition as usize]
            .iter_decoding()
            .map(|(req, state)| (req, state.current_kv))
            .collect()
    }
}

impl SpeculativeKv for FullAttnKv {
    fn begin_proposal(&mut self, request: RequestId, partition: PartitionId, proposal_tokens: u32) {
        let owner = self.request_to_partition.get(&request).copied();
        assert_eq!(
            owner,
            Some(partition),
            "speculative proposal must use the request's sticky KV partition"
        );
        let previous = self
            .tentative_proposals
            .insert(request, (partition, proposal_tokens));
        assert!(
            previous.is_none(),
            "request cannot begin a second proposal before resolving the first"
        );
    }

    fn commit_accepted(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        committed_tokens: u32,
    ) {
        let proposal = self.tentative_proposals.get(&request).copied();
        let (proposal_partition, proposal_tokens) =
            proposal.expect("accepted tokens require a live speculative proposal");
        assert_eq!(
            proposal_partition, partition,
            "accepted tokens must commit on the proposal's sticky partition"
        );
        assert!(
            committed_tokens > 0 && committed_tokens <= proposal_tokens.saturating_add(1),
            "draft/verify may commit at most the proposal plus one target bonus token"
        );
        let request_ids = [request];
        self.advance(
            AdvanceScope::RequestSubset {
                partition,
                request_ids: &request_ids,
            },
            committed_tokens,
        );
    }

    fn discard_rejected(&mut self, request: RequestId, partition: PartitionId) {
        let proposal = self.tentative_proposals.remove(&request);
        assert_eq!(
            proposal.map(|(proposal_partition, _)| proposal_partition),
            Some(partition),
            "discard must resolve the request's live proposal on its sticky partition"
        );
    }
}

impl SlotPipelineKv for FullAttnKv {
    #[inline]
    fn current_kv(&self, partition: PartitionId, req: RequestId) -> Option<u64> {
        self.batches[partition as usize].decode_current_kv(req)
    }

    #[inline]
    fn estimated_peak(&self, partition: PartitionId) -> u64 {
        self.batches[partition as usize].projected_peak_kv() + self.partition_promised(partition)
    }
}

impl ChunkedPrefillKv for FullAttnKv {
    fn append_prefill_chunk(&mut self, partition: PartitionId, req: RequestId, chunk_tokens: u32) {
        // Reserved → resident: grow the pool and render this iter's chunk. No re-gate
        // (capacity was reserved in full at admit); draw the reservation down so the
        // (active_kv + promised) demand the gate sees is unchanged.
        {
            let batch = &mut self.batches[partition as usize];
            batch.kv.add_kv(u64::from(chunk_tokens));
            batch.prefill_admits.push(req);
        }
        if let Some((_, reserved)) = self.promised.get_mut(&req) {
            *reserved = reserved.saturating_sub(u64::from(chunk_tokens));
        }
    }

    fn finish_chunked_prefill(
        &mut self,
        partition: PartitionId,
        req: RequestId,
        initial_kv: u64,
        remaining: u32,
    ) {
        // Drop the (now decode-only) reservation, then transition to decode through
        // the REAL `finalize_to_decode`. That method re-adds `initial_kv`, so first
        // undo the chunks' incremental adds — net resident unchanged, but the decode
        // set / index / peak are maintained by the real path. (Assumes no prefix
        // cache: `initial_kv == Σ chunk_tokens`; a `PrefixCacheKv` hit would offset this.)
        self.promised.remove(&req);
        let batch = &mut self.batches[partition as usize];
        batch.kv.sub_kv(initial_kv);
        batch.finalize_to_decode(req, initial_kv, remaining);
    }
}

impl HandoffKv for FullAttnKv {
    fn hold(&mut self, partition: PartitionId, req: RequestId, kv_tokens: u64) {
        // The reservation was drained into prefill_admits already, so there is
        // nothing to move out of `promised`; record the held KV as a fresh ledger
        // entry (it keeps gating admission via `partition_held`). The local KvPool
        // stays empty — a prefill worker never finalizes a local decode.
        self.promised.remove(&req);
        self.held.insert(req, (partition, kv_tokens));
    }

    #[inline]
    fn drop_held(&mut self, req: RequestId) {
        self.held.remove(&req);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speculative_commit_advances_only_the_named_request() {
        let mut kv_store = FullAttnKv::new(1, 100, KvAdmission::Strict, None);
        let request = RequestId(7);
        let footprint = kv_store.footprint(request, 10, 20);
        kv_store.reserve(request, 0, footprint);
        kv_store.commit_resident(request, 0, 10, 20);

        kv_store.begin_proposal(request, 0, 4);
        kv_store.commit_accepted(request, 0, 3);
        kv_store.discard_rejected(request, 0);

        assert_eq!(kv_store.decode_members(0), vec![(request, 13)]);
        assert!(kv_store.tentative_proposals.is_empty());
    }
}
