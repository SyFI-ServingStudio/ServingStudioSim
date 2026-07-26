//! `HybridStateKv` — mixed full-attention + recurrent (SSM/SWA) KV (interfaces doc
//! §2.2, F4).
//!
//! A wrapper over `FullAttnKv` for the FULL-ATTENTION kind (its KV grows +1/token — the
//! real `Batch` models exactly that) plus a side ledger for the RECURRENT kind, whose
//! conv/SSM state is FIXED-size (+0/token). The two kinds are the whole point of the
//! `HybridKvView` capability, and the per-kind `advance` difference falls out for free:
//! `advance` delegates to the inner `Batch` (full grows) and never touches the recurrent
//! ledger (stays constant). The recurrent state shares the SAME physical budget (one HBM
//! pool — a byte of state is a byte attention KV cannot have), so it enters the footprint
//! as the `fixed_state` component — `FullAttnKv::fits` already threads that through the
//! real `KvAdmission::try_admit`. A genuinely SEPARATE budget (a static carve-out, or a
//! hard state-slot count in a unit that does not convert to tokens) would instead be a
//! second pool checked here in `fits`; that stays inside this file. The mixed-layer COST is the hybrid model's
//! internal business (opaque to the worker, per the arch contract); the worker only
//! accounts capacity per kind and exposes the breakdown.
//!
//! Superset: no SSM/SWA state exists in-crate, so `recurrent_state_tokens` is a modeled
//! per-request constant (∝ recurrent-layer count × state size), not a measured shape.

use std::collections::HashMap;

use crate::common::{RequestId, Time};
use crate::log::KvSampler;
use crate::worker::admission_helpers::KvAdmission;

use super::super::shared::advance_scope::{AdvanceScope, PartitionId};
use super::full_attn::{FullAttentionKvFootprint, FullAttnKv};
use super::{HybridKvView, IterWorkerKv, KvCapacityPressure, KvStore, SlotPipelineKv};

pub struct HybridStateKv {
    /// Full-attention KV (real `Batch` machinery, grows with `advance`).
    inner: FullAttnKv,
    /// Recurrent/conv state per resident request as `(PartitionId, tokens)` — FIXED
    /// size, never advanced. Partition-keyed like the inner `promised`/`held` ledgers
    /// because it is charged per pool in `fits`.
    recurrent: HashMap<RequestId, (PartitionId, u64)>,
    /// Modeled fixed recurrent state (tokens-equiv) each request occupies.
    recurrent_state_tokens: u64,
}

impl HybridStateKv {
    pub fn new(
        num_partitions: usize,
        kv_capacity: u64,
        admission: KvAdmission,
        sampler: Option<KvSampler>,
        recurrent_state_tokens: u64,
    ) -> Self {
        Self {
            inner: FullAttnKv::new(num_partitions, kv_capacity, admission, sampler),
            recurrent: HashMap::new(),
            recurrent_state_tokens,
        }
    }

    /// Resident recurrent state for a partition. The inner `Batch` cannot see it (it
    /// left `promised` at `commit_resident` and never entered `active_kv`), yet it
    /// occupies the same pool — so it must be charged back into the gate.
    #[inline]
    fn partition_recurrent(&self, partition: PartitionId) -> u64 {
        self.recurrent
            .values()
            .filter(|(state_partition, _)| *state_partition == partition)
            .map(|(_, tokens)| *tokens)
            .sum()
    }
}

impl KvStore for HybridStateKv {
    type Footprint = FullAttentionKvFootprint;

    #[inline]
    fn num_partitions(&self) -> usize {
        self.inner.num_partitions()
    }

    /// Full-attention `(prompt, decode)` + the recurrent state as `fixed_state` — extra
    /// resident tokens the gate must count, charged once and never advanced. It is its
    /// OWN footprint slot, not the prefix-cache one: a hybrid model behind a prefix
    /// cache sets both and they sum.
    fn footprint(&self, _req: RequestId, prompt: u32, decode: u32) -> FullAttentionKvFootprint {
        FullAttentionKvFootprint {
            prompt,
            decode,
            cached_prefix: 0,
            fixed_state: self.recurrent_state_tokens as u32,
        }
    }

    /// NOT a plain delegate. The candidate's OWN recurrent state rides in the
    /// footprint (`fixed_state`), but every ALREADY-RESIDENT request's state sits in
    /// this wrapper's ledger, invisible to the inner `Batch` — charge it as extra
    /// occupancy or the pool is over-admitted by one state per resident request.
    /// No double count: a request is in `promised` until `commit_resident` moves it
    /// into `recurrent`, never in both.
    #[inline]
    fn fits(&self, partition: PartitionId, footprint: &FullAttentionKvFootprint) -> bool {
        self.inner.fits_with_extra_occupied(
            partition,
            footprint,
            self.partition_recurrent(partition),
        )
    }

    /// Same correction on the ranking side: resident recurrent state is real occupancy.
    #[inline]
    fn pressure(&self, partition: PartitionId) -> KvCapacityPressure {
        self.inner
            .pressure_with_extra_resident(partition, self.partition_recurrent(partition))
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

    fn commit_resident(
        &mut self,
        req: RequestId,
        partition: PartitionId,
        initial_kv: u64,
        remaining: u32,
    ) {
        // Full-attention part enters the real decode set; the recurrent state is a
        // separate fixed ledger entry (never grows under `advance`).
        self.inner
            .commit_resident(req, partition, initial_kv, remaining);
        self.recurrent
            .insert(req, (partition, self.recurrent_state_tokens));
    }

    fn release(&mut self, req: RequestId, partition: PartitionId) {
        self.inner.release(req, partition);
        self.recurrent.remove(&req);
    }

    /// The per-kind advance: full-attention KV grows (inner `Batch`), the recurrent
    /// ledger is untouched (+0) — that asymmetry is the hybrid model.
    #[inline]
    fn advance(&mut self, scope: AdvanceScope, steps: u32) {
        self.inner.advance(scope, steps);
    }

    #[inline]
    fn sample_submit(&mut self, partition: PartitionId, now: Time) {
        self.inner.sample_submit(partition, now)
    }
}

impl IterWorkerKv for HybridStateKv {
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
    fn release_external(&mut self, req: RequestId, current_kv: u64) -> Option<PartitionId> {
        let partition = self.inner.release_external(req, current_kv);
        if partition.is_some() {
            self.recurrent.remove(&req);
        }
        partition
    }
    #[inline]
    fn prefill_admits(&self, partition: PartitionId) -> Vec<RequestId> {
        self.inner.prefill_admits(partition)
    }
    #[inline]
    fn decode_members(&self, partition: PartitionId) -> Vec<(RequestId, u64)> {
        // Attention context is the full-attention KV (the recurrent state does not enter
        // the attention length); the hybrid model folds the recurrent cost internally.
        self.inner.decode_members(partition)
    }
}

impl HybridKvView for HybridStateKv {
    #[inline]
    fn num_kinds(&self) -> u16 {
        2 // {full-attention, recurrent}
    }

    fn state_len(&self, partition: PartitionId, req: RequestId, kind: u16) -> Option<u64> {
        match kind {
            0 => self.inner.current_kv(partition, req), // full-attention (grows)
            1 => self.recurrent.get(&req).map(|(_, tokens)| *tokens), // recurrent (fixed)
            _ => None,
        }
    }
}
