//! KV resource axis — the traits from interfaces doc §2.
//!
//! `KvStore` is the core every impl answers; `IterWorkerKv` is the iter-wise narrow
//! facts the IterBatchWorker family (barebone/hp/pd/chunked/spec) reads. Capability
//! sub-traits (`HandoffKv`/`ChunkedPrefillKv`/`PrefixCacheKv`/views) are added by the workers that
//! need them. `Footprint` is opaque to Admission — for full-attn it is a struct
//! carrying `(prompt, decode, cached_prefix, fixed_state)` so `fits` can call the REAL
//! `KvAdmission::try_admit(group, group_promised, p, d)` without reproducing it.

use crate::common::{RequestId, Time};

use super::shared::advance_scope::{AdvanceScope, PartitionId};

mod full_attn;
mod hybrid;
mod multi_model;
mod prefix_attn;
mod tiered;
pub use full_attn::FullAttnKv;
pub use hybrid::HybridStateKv;
pub use multi_model::ModelPartitionedKv;
pub use prefix_attn::ModeledPrefixCacheKv;
pub use tiered::TieredMemoryKv;

/// Model identity for multi-model co-serve. The crate has no `ModelId` (one worker =
/// one arch model today), so this is a superset vocab: `ModelPartitionedKv` /
/// `MultiModelIterExecution` key their per-model pools by it, and `ModelSwitchKv::set_active`
/// selects the live one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ModelId(pub u16);

/// KV-computed neutral report (token-equivalent), NOT `Footprint`. Admission reads
/// this to rank under pressure; it never sees the opaque footprint currency.
pub struct KvCapacityPressure {
    pub resident_tokens_equiv: u64,
    pub reserved_tokens_equiv: u64,
    pub capacity_tokens_equiv: u64,
}

pub trait KvStore {
    /// Opaque to Admission. FullAttn = a `(prompt,decode,prefix)` struct.
    type Footprint;

    fn num_partitions(&self) -> usize;

    /// (prompt, decode) tokens → opaque footprint. Role difference lives in what
    /// Admission passes (prefill: decode=0), NOT in KV.
    fn footprint(&self, req: RequestId, prompt: u32, decode: u32) -> Self::Footprint;

    /// The ONE capacity question. No scalar remaining/capacity/peak leaks.
    fn fits(&self, partition: PartitionId, footprint: &Self::Footprint) -> bool;

    /// Read-only pressure for admission ranking (interfaces doc §2.3).
    fn pressure(&self, partition: PartitionId) -> KvCapacityPressure;

    fn reserve(&mut self, req: RequestId, partition: PartitionId, footprint: Self::Footprint);
    /// promised → resident. `initial_kv` (= prompt+prefix) is the RESIDENT amount,
    /// distinct from the reserved footprint (which also counted decode).
    fn commit_resident(
        &mut self,
        req: RequestId,
        partition: PartitionId,
        initial_kv: u64,
        remaining: u32,
    );
    fn release(&mut self, req: RequestId, partition: PartitionId);

    /// Grow one iteration by `steps` (full +N / recurrent 0 / spec accept N),
    /// keyed by request `AdvanceScope`.
    fn advance(&mut self, scope: AdvanceScope, steps: u32);

    /// Iter-end occupancy sample (realizes doc §2.1 `sample()`; side-effecting log).
    fn sample_submit(&mut self, partition: PartitionId, now: Time);
}

/// Iter-wise narrow facts (doc §2.1 "各具体 Kv 提供本 family template 所需的窄查询").
/// Returned as owned snapshots so the trait does not leak `&Batch`.
pub trait IterWorkerKv: KvStore {
    fn drain_ready(&mut self);
    fn clear_prefill_admits(&mut self, partition: PartitionId);
    fn has_live_decode(&self, partition: PartitionId) -> bool;
    fn live_decode_count(&self, partition: PartitionId) -> u32;
    fn has_prefill_admit(&self, partition: PartitionId) -> bool;
    fn status_active(&self, partition: PartitionId) -> u32;
    /// Cancellation past the pending queue (was `release_request`'s admitted half).
    fn release_external(&mut self, req: RequestId, current_kv: u64) -> Option<PartitionId>;
    /// This iter's fresh prefills (for input rendering + first-token recording).
    fn prefill_admits(&self, partition: PartitionId) -> Vec<RequestId>;
    /// Live decodes as `(request, current_kv)` (for input rendering + token recording).
    fn decode_members(&self, partition: PartitionId) -> Vec<(RequestId, u64)>;
}

/// Tentative-state lifecycle required by S6 draft/verify cadence.
///
/// `begin_proposal` makes unverified draft state visible to the KV implementation.
/// Once target verification completes, `commit_accepted` advances only the
/// request's committed prefix and `discard_rejected` drops the remaining
/// tentative suffix. Full-attention reserves the request's complete output
/// footprint at admission, so proposal creation does not re-run the capacity
/// gate.
pub trait SpeculativeKv: IterWorkerKv {
    fn begin_proposal(&mut self, request: RequestId, partition: PartitionId, proposal_tokens: u32);

    fn commit_accepted(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        committed_tokens: u32,
    );

    fn discard_rejected(&mut self, request: RequestId, partition: PartitionId);
}

/// AFD-attn family read view (doc §2 capability). One shard partition is shared by
/// N pipeline slots; the SHELL owns request→slot, so KV answers only per-request
/// facts by id (not partition-level sets like `IterWorkerKv`). `current_kv` feeds
/// `AttentionLayerExecution::build_slot_input` per slot; `estimated_peak` feeds L6 `AfdAttnWorker`.
pub trait SlotPipelineKv: KvStore {
    fn current_kv(&self, partition: PartitionId, req: RequestId) -> Option<u64>;
    fn estimated_peak(&self, partition: PartitionId) -> u64;
}

/// Held-KV capability (interfaces doc §2.2). A PD prefill worker prefills a request
/// then HOLDS its KV — physically resident, but tracked in a separate `held` ledger
/// (its local decode set stays empty) — until the decode side pulls it and acks via
/// `ReleaseKv`. Held KV still gates admission (it occupies the pool), so `fits`
/// counts it alongside `promised`. The `PrefillHandoffAdmission` admission requires this in
/// its bound (`where K: HandoffKv`).
///
/// NOTE / deviation from the doc's zero-arg `hold(req)`: the doc assumed the
/// reservation persists in the `promised` ledger until `hold`, but the iter family's
/// `drain_ready` clears `promised` at `form_batch`. So `hold` carries the token count
/// (computed from the store at prefill-done, mirroring the real PD worker).
pub trait HandoffKv: KvStore {
    /// Mark a just-prefilled request's `kv_tokens` as held (awaiting the decode-side
    /// pull). Excluded from decode advance; survives a normal `release`.
    fn hold(&mut self, partition: PartitionId, req: RequestId, kv_tokens: u64);
    /// Release the held reservation after the decode side acks the pull (`ReleaseKv`).
    fn drop_held(&mut self, req: RequestId);
}

/// Chunked-prefill capability (interfaces doc §2.2, F3). A request reserved at its
/// FULL footprint (via `KvStore::reserve`) lands its prompt into resident KV a
/// chunk at a time across iterations, drawing its reservation down as it goes; the
/// last chunk transitions it to decode. Only a layout that can partially realize a
/// prefill implements this — the `ChunkedPrefillAdmission` admission requires it in its impl
/// bound (`where K: ChunkedPrefillKv`), which is why it is a type-level capability, not a
/// method flag. Renders each landed chunk as a prefill admit (`prefill_admits`), so
/// `IterWorkerKv` + `IterModelExecution::build_iteration_input` stay unchanged (the builder already reads
/// `active_chunk_len`, not `prompt_len`).
pub trait ChunkedPrefillKv: KvStore {
    /// Land `chunk_tokens` of a fully-reserved request's prompt into resident KV
    /// this iter (reserved → resident; net pool demand unchanged). Renders the
    /// request as a prefill admit for this iteration.
    fn append_prefill_chunk(&mut self, partition: PartitionId, req: RequestId, chunk_tokens: u32);

    /// Last chunk: the prompt is fully resident → transition to decode (the chunked
    /// analogue of `commit_resident`, but at the final chunk). `initial_kv` is the
    /// now-resident prompt KV; `remaining` the decode horizon.
    fn finish_chunked_prefill(
        &mut self,
        partition: PartitionId,
        req: RequestId,
        initial_kv: u64,
        remaining: u32,
    );
}

/// One partition-local prefix lookup plus the opaque capacity footprint computed from
/// the same cache snapshot. Keeping them together prevents admission from probing one
/// partition and accidentally gating/reserving with another partition's match.
pub struct PrefixPlacementProbe<Footprint> {
    pub matched_tokens: u32,
    pub footprint: Footprint,
}

/// Prefix-cache capability (interfaces doc §2.2, F1). Prefix availability is
/// partition-local: admission probes candidate partitions, chooses one, then
/// `KvStore::reserve` records sticky request→partition ownership. The arch renders
/// `matched_tokens` as already-present context through `RequestRecord::prefix_kv`.
///
/// Superset scope: the crate has no radix index/refcounts/replacement, so `ModeledPrefixCacheKv`
/// models a configured hit fraction per partition. Its cached region shares the same
/// capacity gate as request KV but is conservatively charged per request; true block
/// sharing is a KV-implementation extension, not an Admission responsibility.
pub trait PrefixCacheKv: KvStore {
    fn probe_prefix(
        &self,
        partition: PartitionId,
        req: RequestId,
        prompt: u32,
        decode: u32,
    ) -> PrefixPlacementProbe<Self::Footprint>;
}

/// Hybrid-attention capability (interfaces doc §2.2, F4). A model with mixed layer
/// kinds (full-attention + recurrent/SSM/SWA) keeps a DIFFERENT KV state per kind:
/// full-attention grows +1/token, a recurrent/conv state is FIXED-size (+0/token). The
/// view exposes the per-kind breakdown so a report/estimator sees both; the per-kind
/// advance difference is internal to `advance` (it grows only the full-attention kind,
/// leaving the recurrent state constant). Superset: no SSM/SWA state exists in-crate.
pub trait HybridKvView: KvStore {
    /// Number of distinct KV kinds (e.g. 2 = {full-attention, recurrent}).
    fn num_kinds(&self) -> u16;
    /// This request's state length (tokens-equiv) for `kind` (0 = full-attention, grows
    /// with `advance`; 1 = recurrent, fixed). `None` if the request is not resident.
    fn state_len(&self, partition: PartitionId, req: RequestId, kind: u16) -> Option<u64>;
}

/// Multi-model co-serve capability (interfaces doc §2.2, multi-arch). N models share
/// one GPU's KV budget with per-model resident pools; a `SwitchModel` control message
/// makes one model active, so subsequent admits land in its pool. The admission/execution
/// constrain ONLY on this read-view capability, never on the concrete KV impl — the
/// point of the axis split. Superset (no `ModelId` / model-switch message in-crate).
pub trait ModelSwitchKv: KvStore {
    /// Route subsequent admits + the cost model to `model`'s co-resident pool.
    fn set_active(&mut self, model: ModelId);
    fn active(&self) -> ModelId;
}

/// Two-tier (fast GPU / slow offload) KV capability — unknown-variant blind test #1
/// (compatibility matrix §15.4). A tiered layout keeps the hottest `fast_capacity`
/// tokens resident in GPU HBM and spills the rest to a slower CPU/NVMe tier; `fits`
/// gains the slow tier's headroom (total capacity = fast + slow), and a tier-aware cost
/// model reads the split to charge the offload transfer. Only a layout that physically
/// tiers implements it — added as a narrow read-view, NOT a method on the base trait, so
/// non-tiered impls carry no dummy. Superset: no offload pool in-crate, so the fast/slow
/// split is a modeled threshold over resident tokens, not a real page table.
///
/// NOTE (boundary this test surfaces): `KvStore`'s `pressure`/`fits` currency is
/// token-EQUIV, which absorbs compression fine (the KV reports whatever equiv it wants).
/// But the tier SPLIT only reaches a cost model through THIS read-view, and the iter
/// family's `IterModelExecution::build_iteration_input<K: IterWorkerKv>` is method-generic on `IterWorkerKv`
/// (unlike `Admission<K>`, which takes K as a trait param and CAN escalate to a
/// capability). So a tier-aware COST needs either an AFD-style execution or a `IterModelExecution`
/// that takes `K` as a trait parameter. The LAYOUT is leaf-addable; tier-aware COST is
/// the same asymmetry blind test #4 hits.
pub trait TieredKvView: KvStore {
    /// Resident tokens for `partition` split as (fast_tier, slow_tier).
    fn resident_by_tier(&self, partition: PartitionId) -> (u64, u64);
}
