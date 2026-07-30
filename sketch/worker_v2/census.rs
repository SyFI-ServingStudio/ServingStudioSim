//! Compile-time worker census — the coverage proof for the interface experiment.
//!
//! Each `assert_*::<Composition>()` line forces the composition to satisfy its L6 worker
//! trait (`IterWorker` / `AfdAttnWorker`) at type-check time. The generic `census` fn is
//! never CALLED — it exists only so the compiler discharges every bound, proving all the
//! sampled `⟨Kv, IterAdmission, IterModelExecution⟩` tuples actually compose against the real traits.
//! `cargo check` passing IS the census result.
//!
//! Coverage (every axis impl appears ≥ once):
//!   Kv:        FullAttnKv · ModeledPrefixCacheKv · HybridStateKv · ModelPartitionedKv · TieredMemoryKv
//!   IterAdmission: LocalPrefillDecodeAdmission · ChunkedPrefillAdmission · PrefillHandoffAdmission · PrefixPrefillDecodeAdmission ·
//!              MultiModelAdmission · FreshRequestSlotAdmission (AFD)
//!   Policy:    FifoOrder · ShortestJobFirst
//!   IterModelExecution: UnifiedIterExecution · AttentionLayerExecutionAdapter · FfnSectionExecutionAdapter · MultiModelIterExecution · TierAwareIterExecution · DraftVerifyExecution
//!   Shell:     IterBatchWorker · DraftVerifyWorker · PullDecodeWorker · SlotAttentionWorker · PullSlotAttentionWorker · BufferedFfnWorker
//!
//! A "server" is a (composition, config) pair — several servers share one type here
//! (barebone vs HP = N; hybrid vs hybrid-DP = N), so
//! the ≥ 20 servers of the README map onto the distinct TYPES asserted below plus their
//! config variants. See README's server table.

#![allow(dead_code)]

use crate::arch::contract::{AttnLayerwiseModel, FfnLayerwiseModel, IterwiseUnifiedModel};
use crate::worker::iter_worker::{AfdAttnWorker, IterWorker};
use crate::worker::types::AttnWorkerMsg;

use super::admission::policy::{FifoOrder, ShortestJobFirst};
use super::admission::{
    ChunkedPrefillAdmission, DraftVerifyAdmission, FreshRequestSlotAdmission,
    LocalPrefillDecodeAdmission, MultiModelAdmission, PrefillHandoffAdmission,
    PrefixPrefillDecodeAdmission,
};
use super::execution::{
    AcceptanceOracle, AttentionLayerExecutionAdapter, DraftVerifyExecution, DraftVerifyModel,
    FfnSectionExecutionAdapter, MultiModelIterExecution, TierAwareIterExecution,
    UnifiedIterExecution,
};
use super::kv::{
    FullAttnKv, HybridStateKv, ModelPartitionedKv, ModeledPrefixCacheKv, TieredMemoryKv,
};
use super::workers::{
    BufferedFfnWorker, DraftVerifyWorker, IterBatchWorker, PullDecodeWorker,
    PullSlotAttentionWorker, SlotAttentionWorker,
};

fn assert_iter<W: IterWorker>() {}

fn assert_afd_attn<W>()
where
    W: AfdAttnWorker,
    W::Msg: From<AttnWorkerMsg>,
{
}

/// The full sampled census. Every line is one distinct worker TYPE (config variants —
/// N partitions, prefix hit rate, co-resident model count — reuse
/// the same type). Type-checking this function is the coverage proof.
fn census<
    MI: IterwiseUnifiedModel,
    MA: AttnLayerwiseModel,
    MF: FfnLayerwiseModel,
    MS: DraftVerifyModel,
    O: AcceptanceOracle,
>() {
    // ── iter family (IterBatchWorker) · FullAttnKv ─────────────────────────────────────
    // #1 barebone / #3 HP-DP (N=1 vs N) share this type.
    assert_iter::<
        IterBatchWorker<
            FullAttnKv,
            LocalPrefillDecodeAdmission<FifoOrder>,
            UnifiedIterExecution<MI>,
        >,
    >();

    // #10 speculative decode uses its own S6 cadence, runtime acceptance
    // outcome, and tentative KV lifecycle.
    assert_iter::<
        DraftVerifyWorker<FullAttnKv, DraftVerifyAdmission<FifoOrder>, DraftVerifyExecution<MS, O>>,
    >();
    // #11 latency-scheduled (policy swap FifoOrder → ShortestJobFirst).
    assert_iter::<
        IterBatchWorker<
            FullAttnKv,
            LocalPrefillDecodeAdmission<ShortestJobFirst>,
            UnifiedIterExecution<MI>,
        >,
    >();
    // #2 chunked prefill + its policy variant.
    assert_iter::<
        IterBatchWorker<FullAttnKv, ChunkedPrefillAdmission<FifoOrder>, UnifiedIterExecution<MI>>,
    >();
    assert_iter::<
        IterBatchWorker<
            FullAttnKv,
            ChunkedPrefillAdmission<ShortestJobFirst>,
            UnifiedIterExecution<MI>,
        >,
    >();
    // #4a PD prefill (wider PdPrefillMsg via A::Msg).
    assert_iter::<IterBatchWorker<FullAttnKv, PrefillHandoffAdmission, UnifiedIterExecution<MI>>>();

    // ── iter family · ModeledPrefixCacheKv (PrefixCacheKv capability escalation) ──────────────────
    // #12 prefix-cache dense + policy variant.
    assert_iter::<
        IterBatchWorker<
            ModeledPrefixCacheKv,
            PrefixPrefillDecodeAdmission<FifoOrder>,
            UnifiedIterExecution<MI>,
        >,
    >();
    assert_iter::<
        IterBatchWorker<
            ModeledPrefixCacheKv,
            PrefixPrefillDecodeAdmission<ShortestJobFirst>,
            UnifiedIterExecution<MI>,
        >,
    >();

    // ── iter family · HybridStateKv (HybridKvView, per-kind advance) ──────────────────
    // #13 hybrid dense / hybrid-DP (N) + policy variant.
    assert_iter::<
        IterBatchWorker<
            HybridStateKv,
            LocalPrefillDecodeAdmission<FifoOrder>,
            UnifiedIterExecution<MI>,
        >,
    >();
    assert_iter::<
        IterBatchWorker<
            HybridStateKv,
            LocalPrefillDecodeAdmission<ShortestJobFirst>,
            UnifiedIterExecution<MI>,
        >,
    >();

    // ── iter family · ModelPartitionedKv (ModelSwitchKv, per-model execution) ──────────────────
    // #17 multi-model co-serve + policy variant.
    assert_iter::<
        IterBatchWorker<
            ModelPartitionedKv,
            MultiModelAdmission<FifoOrder>,
            MultiModelIterExecution<MI>,
        >,
    >();
    assert_iter::<
        IterBatchWorker<
            ModelPartitionedKv,
            MultiModelAdmission<ShortestJobFirst>,
            MultiModelIterExecution<MI>,
        >,
    >();

    // ── iter family · TieredMemoryKv (unknown-variant blind test #1 + Root ② fix) ──────
    // The tiered LAYOUT composes with a plain execution (capacity gain, tier ignored in cost).
    assert_iter::<
        IterBatchWorker<
            TieredMemoryKv,
            LocalPrefillDecodeAdmission<FifoOrder>,
            UnifiedIterExecution<MI>,
        >,
    >();
    // The tier-aware EXEC escalates to `K: TieredKvView` and charges offload — proof that a
    // KV capability now reaches the cost model (`IterModelExecution<K>`, not method-generic). A
    // `TierAwareIterExecution` over `FullAttnKv` would NOT type-check (no `TieredKvView`) — capability at
    // the recipe, not a dummy on the KV. This is the asymmetry-with-`IterAdmission<K>` fix.
    assert_iter::<
        IterBatchWorker<
            TieredMemoryKv,
            LocalPrefillDecodeAdmission<FifoOrder>,
            TierAwareIterExecution<MI>,
        >,
    >();

    // ── PD decode shell (own cadence, no admission axis) ─────────────────────────
    // #4b PD decode (pull → decode).
    assert_iter::<PullDecodeWorker<FullAttnKv, UnifiedIterExecution<MI>>>();

    // ── AFD-attn slot pipeline (shared AttentionSlotPipeline) ─────────────────────────────────
    // #7 colocated AFD attn.
    assert_afd_attn::<
        SlotAttentionWorker<
            FullAttnKv,
            FreshRequestSlotAdmission,
            AttentionLayerExecutionAdapter<MA>,
        >,
    >();
    // #9 PD-for-AFD decode attn (pull ingress + KvPullComplete).
    assert_afd_attn::<PullSlotAttentionWorker<FullAttnKv, AttentionLayerExecutionAdapter<MA>>>();

    // ── AFD-ffn double-buffer (no Kv, no admission) ──────────────────────────────
    // #8 AFD ffn.
    assert_iter::<BufferedFfnWorker<FfnSectionExecutionAdapter<MF>>>();
}
