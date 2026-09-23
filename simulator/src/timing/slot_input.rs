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
    AllReduceFusionKernelInput, AllReduceKernelInput, AllReduceResidualRmsNormKernelInput,
    BatchedGemmKernelInput, Bf16FusedMoeKernelInput, ClampedSwigluKernelInput,
    DeepseekV4FusedInvRopeFp8QuantKernelInput, DeepseekV4FusedQKvRmsnormKernelInput,
    DeepseekV4IndexerPrefillKernelInput, DeepseekV4IndexerQRopeQuantKernelInput,
    DeepseekV4PackedCacheGatherKernelInput, DeepseekV4QnormRopeKvInsertKernelInput,
    DeepseekV4SparseAttnCompressStoreKernelInput, DeepseekV4SparseMlaDecodeKernelInput,
    DeepseekV4SparseMlaPrefillKernelInput, DsaIndexCacheAppendKernelInput,
    DsaIndexerQRopeQuantKernelInput, DsaMqaLogitsPrefillKernelInput,
    DsaPagedMqaLogitsDecodeKernelInput, DsaPersistentTopkDecodeKernelInput,
    DsaSparseIndexRemapKernelInput, DsaSparseMlaAttentionKernelInput,
    DsaSparseMlaPrefillKernelInput, DsaTopkPrefillKernelInput, ElementwiseKernelInput,
    FlashinferAttnDecodeKernelInput, FlashinferAttnRectKernelInput, Fp8BlockQuantKernelInput,
    Fp8BlockscaleGroupedGemmKernelInput, Fp8PerTokenGroupQuantKernelInput,
    GdnCausalConvDecodeKernelInput, GdnCausalConvPrefillKernelInput, GdnChunkDeltaRuleKernelInput,
    GdnChunkLocalCumsumKernelInput, GdnChunkOutputKernelInput, GdnChunkRecomputeWUKernelInput,
    GdnChunkScaledDotKktKernelInput, GdnChunkSolveTrilKernelInput, GdnChunkStateUpdateKernelInput,
    GdnGatedRmsNormKernelInput, GdnPrefillPostConvKernelInput, GdnRecurrentDecodeKernelInput,
    GemmFp32OutputKernelInput, GroupedGemmKernelInput, KvCacheAppendKernelInput,
    MhcRmsNormKernelInput, MlaCacheAppendKernelInput, MlaRopeQuantizeFp8KernelInput,
    MoeAlignBlockSizeKernelInput, MoeAlltoallKernelInput, MoeAlltoallPrepareKernelInput,
    MoeEpCollectiveKernelInput, MoeFinalizeFuseSharedKernelInput, MoeFinalizeRoutingKernelInput,
    MoeFusedTopkKernelInput, MoeSumKernelInput, MoeTopkSoftplusSqrtKernelInput,
    Mxfp4MarlinMoeGemmKernelInput, Nvfp4FusedMoeKernelInput, Nvfp4QuantKernelInput,
    P2pInterKernelInput, P2pIntraKernelInput, ResidualRmsNormKernelInput, RmsNormKernelInput,
    SingleGemmKernelInput, VllmFusedMoeKernelInput, VllmMlaRopeKernelInput,
};

/// The prefill aggregating leaf's input: the full `(prefix_len, append_len)`
/// fan-out the attention op summed into one slot (many prefill requests fold to a
/// single leaf).
#[derive(Clone, Serialize)]
pub struct AttnPrefillLog {
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
}

/// The DFlash2 draft-attention aggregating leaf's input: every request's
/// `(q_len, kv_len)` rectangle summed into one slot. Unlike prefill there is no
/// causal split -- the draft attends its whole query block over the whole
/// context -- so the faithful record is the rectangle list itself.
#[derive(Clone, Serialize)]
pub struct Dflash2DraftAttnLog {
    pub rectangles: Vec<(u32, u32)>,
}

/// Gated DeltaNet causal-convolution prefill fan-in: every request-local
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

/// Sparse-MLA decode cache input. Preserve the exact request-local lengths in
/// the trace even though the low-dimensional cache evaluates their rounded
/// uniform-equivalent context.
#[derive(Clone, Serialize)]
pub struct DsaSparseMlaDecodeLog {
    pub context_lens: Vec<u32>,
    pub decode_next_n: u32,
    pub projected_context: u32,
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
    Nvfp4FusedMoe => Nvfp4FusedMoeKernelInput,
    Bf16FusedMoe => Bf16FusedMoeKernelInput,
    Nvfp4Quant => Nvfp4QuantKernelInput,
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
    MoeEpCollective => MoeEpCollectiveKernelInput,
    Mxfp4MarlinMoeGemm => Mxfp4MarlinMoeGemmKernelInput,
    MoeSum => MoeSumKernelInput,
    ClampedSwiglu => ClampedSwigluKernelInput,
    DeepseekV4FusedInvRopeFp8Quant => DeepseekV4FusedInvRopeFp8QuantKernelInput,
    DeepseekV4FusedQKvRmsnorm => DeepseekV4FusedQKvRmsnormKernelInput,
    DeepseekV4IndexerQRopeQuant => DeepseekV4IndexerQRopeQuantKernelInput,
    DsaIndexerQRopeQuant => DsaIndexerQRopeQuantKernelInput,
    DeepseekV4IndexerPrefill => DeepseekV4IndexerPrefillKernelInput,
    DeepseekV4PackedCacheGather => DeepseekV4PackedCacheGatherKernelInput,
    DeepseekV4QnormRopeKvInsert => DeepseekV4QnormRopeKvInsertKernelInput,
    DeepseekV4SparseAttnCompressStore => DeepseekV4SparseAttnCompressStoreKernelInput,
    DeepseekV4SparseMlaDecode => DeepseekV4SparseMlaDecodeKernelInput,
    DeepseekV4SparseMlaPrefill => DeepseekV4SparseMlaPrefillKernelInput,
    MoeTopkSoftplusSqrt => MoeTopkSoftplusSqrtKernelInput,
    GemmFp32Output => GemmFp32OutputKernelInput,
    MlaRopeQuantizeFp8 => MlaRopeQuantizeFp8KernelInput,
    MoeFinalizeFuseShared => MoeFinalizeFuseSharedKernelInput,
    MhcRmsNorm => MhcRmsNormKernelInput,
    MoeAlltoall => MoeAlltoallKernelInput,
    MoeAlltoallPrepare => MoeAlltoallPrepareKernelInput,
    AttnPrefill => AttnPrefillLog,
    DsaIndexerPrefill => DsaIndexerPrefillLog,
    DsaSparseMlaPrefill => DsaSparseMlaPrefillLog,
    DsaSparseMlaDecode => DsaSparseMlaDecodeLog,
    AttnDecode  => FlashinferAttnDecodeKernelInput,
    AttnRect    => FlashinferAttnRectKernelInput,
    Dflash2DraftAttn => Dflash2DraftAttnLog,
    KvCacheAppend => KvCacheAppendKernelInput,
    MlaCacheAppend => MlaCacheAppendKernelInput,
    DsaIndexCacheAppend => DsaIndexCacheAppendKernelInput,
    DsaMqaLogitsPrefill => DsaMqaLogitsPrefillKernelInput,
    DsaPagedMqaLogitsDecode => DsaPagedMqaLogitsDecodeKernelInput,
    DsaPersistentTopkDecode => DsaPersistentTopkDecodeKernelInput,
    DsaSparseIndexRemap => DsaSparseIndexRemapKernelInput,
    DsaSparseMlaPrefillKernel => DsaSparseMlaPrefillKernelInput,
    DsaSparseMlaAttention => DsaSparseMlaAttentionKernelInput,
    DsaTopkPrefill => DsaTopkPrefillKernelInput,
    VllmFusedMoe => VllmFusedMoeKernelInput,
    VllmMlaRope => VllmMlaRopeKernelInput,
    AllReduce   => AllReduceKernelInput,
    AllReduceFusion => AllReduceFusionKernelInput,
    AllReduceResidualRmsNorm => AllReduceResidualRmsNormKernelInput,
    P2pIntra    => P2pIntraKernelInput,
    P2pInter    => P2pInterKernelInput,
}
