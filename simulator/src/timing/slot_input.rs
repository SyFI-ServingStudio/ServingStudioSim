//! `SlotInput` — one leaf's typed kernel input, captured for the `cost_log`
//! `slot_input` column (the per-kernel Perfetto trace). A **closed** enum over
//! every leaf input type, stored inline in the per-iter capture buffer (no
//! per-slot box) and serialized to JSON on the writer thread.
//!
//! This is a deliberate exception to the codebase's "no central enum, kernels
//! self-register via `inventory`" rule (`timing/kernels/engine.rs`): a closed
//! enum is the price of inline capture — it removes the per-slot heap box the
//! `erased_serde` variant pays (measured ~28% sim-thread wall on a saturating
//! run; see memory `slot-input-capture-centralized-enum`). Adding a kernel whose
//! input reaches a leaf means one new `log_inputs!` line below; the
//! `K::Input: Into<SlotInput>` bound on `Op::eval` turns a missing entry into a
//! compile error, so the registry can't silently drift.

use serde::Serialize;

use crate::timing::kernels::{
    AllReduceKernelInput, AllReduceResidualRmsNormKernelInput, BatchedGemmKernelInput,
    DsaIndexCacheAppendKernelInput, DsaMqaLogitsPrefillKernelInput,
    DsaPagedMqaLogitsDecodeKernelInput, DsaPersistentTopkDecodeKernelInput,
    DsaSparseMlaAttentionKernelInput, DsaTopkPrefillKernelInput, ElementwiseKernelInput,
    FlashinferAttnDecodeKernelInput, FlashinferAttnRectKernelInput, Fp8BlockQuantKernelInput,
    Fp8BlockscaleGroupedGemmKernelInput, Fp8PerTokenGroupQuantKernelInput,
    GdnCausalConvDecodeKernelInput, GdnCausalConvPrefillKernelInput, GdnChunkDeltaRuleKernelInput,
    GdnChunkLocalCumsumKernelInput, GdnChunkOutputKernelInput, GdnChunkRecomputeWUKernelInput,
    GdnChunkScaledDotKktKernelInput, GdnChunkSolveTrilKernelInput, GdnChunkStateUpdateKernelInput,
    GdnGatedRmsNormKernelInput, GdnPrefillPostConvKernelInput, GdnRecurrentDecodeKernelInput,
    GroupedGemmKernelInput, KvCacheAppendKernelInput, MlaCacheAppendKernelInput,
    MoeAlignBlockSizeKernelInput, MoeAlltoallKernelInput, MoeAlltoallPrepareKernelInput,
    MoeFinalizeRoutingKernelInput, MoeFusedTopkKernelInput, P2pInterKernelInput,
    P2pIntraKernelInput, ResidualRmsNormKernelInput, RmsNormKernelInput, SingleGemmKernelInput,
    VllmFusedMoeKernelInput, VllmMlaRopeKernelInput,
};

/// The prefill aggregating leaf's input: the full `(prefix_len, append_len)`
/// fan-out the attention op summed into one slot (many prefill requests fold to a
/// single leaf).
#[derive(Clone, Serialize)]
pub struct AttnPrefillLog {
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
}

/// Gated `DeltaNet` causal-convolution prefill fan-in: every request-local
/// sequence length whose metrics were accumulated into the one fixed leaf.
#[derive(Clone, Serialize)]
pub struct GdnCausalConvPrefillLog {
    pub sequence_lengths: Vec<u32>,
}

/// Sparse-MLA prefill aggregating leaf input: every request-local `(Q, S)` cell
/// whose metrics were summed into the one fixed prefill slot.
#[derive(Clone, Serialize)]
pub struct DsaSparseMlaPrefillLog {
    pub prefill_query_cache_pairs: Vec<(u32, u32)>,
}

/// DSA indexer prefill fan-in: every request-local `(Q, N)` cell aggregated by
/// cache gather, logits, or top-k into its corresponding fixed leaf slot.
#[derive(Clone, Serialize)]
pub struct DsaIndexerPrefillLog {
    pub prefill_query_key_pairs: Vec<(u32, u32)>,
}

/// Declare the `SlotInput` enum + a `From<Input>` per variant from one central
/// list. `#[serde(untagged)]` so each variant serializes as just its inner input
/// object (e.g. `{"m":512}`) — the slot's kernel kind is recovered from the
/// matching per-worker `cost_manifest/` sidecar, so no tag is needed in the row.
macro_rules! log_inputs {
    ($($variant:ident => $ty:ty),+ $(,)?) => {
        #[derive(Clone, Serialize)]
        #[serde(untagged)]
        pub enum SlotInput {
            $( $variant($ty), )+
        }
        $(
            impl From<$ty> for SlotInput {
                fn from(v: $ty) -> Self { SlotInput::$variant(v) }
            }
        )+
    };
}

log_inputs! {
    Gemm        => SingleGemmKernelInput,
    BatchedGemm => BatchedGemmKernelInput,
    GroupedGemm => GroupedGemmKernelInput,
    RmsNorm     => RmsNormKernelInput,
    ResidualRmsNorm => ResidualRmsNormKernelInput,
    Elementwise => ElementwiseKernelInput,
    Fp8BlockQuant => Fp8BlockQuantKernelInput,
    Fp8BlockscaleGroupedGemm => Fp8BlockscaleGroupedGemmKernelInput,
    Fp8PerTokenGroupQuant => Fp8PerTokenGroupQuantKernelInput,
    GdnCausalConvDecode => GdnCausalConvDecodeKernelInput,
    GdnCausalConvPrefill => GdnCausalConvPrefillKernelInput,
    GdnCausalConvPrefillFanIn => GdnCausalConvPrefillLog,
    GdnChunkDeltaRule => GdnChunkDeltaRuleKernelInput,
    GdnChunkLocalCumsum => GdnChunkLocalCumsumKernelInput,
    GdnChunkOutput => GdnChunkOutputKernelInput,
    GdnChunkRecomputeWU => GdnChunkRecomputeWUKernelInput,
    GdnChunkScaledDotKkt => GdnChunkScaledDotKktKernelInput,
    GdnChunkSolveTril => GdnChunkSolveTrilKernelInput,
    GdnChunkStateUpdate => GdnChunkStateUpdateKernelInput,
    GdnGatedRmsNorm => GdnGatedRmsNormKernelInput,
    GdnPrefillPostConv => GdnPrefillPostConvKernelInput,
    GdnRecurrentDecode => GdnRecurrentDecodeKernelInput,
    MoeFinalizeRouting => MoeFinalizeRoutingKernelInput,
    MoeFusedTopk => MoeFusedTopkKernelInput,
    MoeAlignBlockSize => MoeAlignBlockSizeKernelInput,
    MoeAlltoall => MoeAlltoallKernelInput,
    MoeAlltoallPrepare => MoeAlltoallPrepareKernelInput,
    AttnPrefill => AttnPrefillLog,
    DsaIndexerPrefill => DsaIndexerPrefillLog,
    DsaSparseMlaPrefill => DsaSparseMlaPrefillLog,
    AttnDecode  => FlashinferAttnDecodeKernelInput,
    AttnRect    => FlashinferAttnRectKernelInput,
    KvCacheAppend => KvCacheAppendKernelInput,
    MlaCacheAppend => MlaCacheAppendKernelInput,
    DsaIndexCacheAppend => DsaIndexCacheAppendKernelInput,
    DsaMqaLogitsPrefill => DsaMqaLogitsPrefillKernelInput,
    DsaPagedMqaLogitsDecode => DsaPagedMqaLogitsDecodeKernelInput,
    DsaPersistentTopkDecode => DsaPersistentTopkDecodeKernelInput,
    DsaSparseMlaAttention => DsaSparseMlaAttentionKernelInput,
    DsaTopkPrefill => DsaTopkPrefillKernelInput,
    VllmFusedMoe => VllmFusedMoeKernelInput,
    VllmMlaRope => VllmMlaRopeKernelInput,
    AllReduce   => AllReduceKernelInput,
    AllReduceResidualRmsNorm => AllReduceResidualRmsNormKernelInput,
    P2pIntra    => P2pIntraKernelInput,
    P2pInter    => P2pInterKernelInput,
}
