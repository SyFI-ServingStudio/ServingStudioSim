//! Per-kind L1 kernel structs.

pub mod all_reduce;
pub mod all_reduce_residual_rms_norm;
pub mod batched_gemm;
pub mod dsa_index_cache_append;
pub mod dsa_mqa_logits_prefill;
pub mod dsa_paged_mqa_logits_decode;
pub mod dsa_persistent_topk_decode;
pub mod dsa_sparse_mla_attention;
pub mod dsa_topk_prefill;
pub mod elementwise;
pub mod engine;
pub mod flashinfer_attn_decode;
pub mod flashinfer_attn_prefill;
pub mod flashinfer_attn_rect;
pub mod fp8_block_quant;
pub mod fp8_blockscale_grouped_gemm;
pub mod fp8_per_token_group_quant;
pub mod gdn_causal_conv_decode;
pub mod gdn_causal_conv_prefill;
pub mod gdn_chunk_delta_rule;
pub mod gdn_chunk_local_cumsum;
pub mod gdn_chunk_output;
pub mod gdn_chunk_recompute_w_u;
pub mod gdn_chunk_scaled_dot_kkt;
pub mod gdn_chunk_solve_tril;
pub mod gdn_chunk_state_update;
pub mod gdn_gated_rms_norm;
pub mod gdn_prefill_post_conv;
pub mod gdn_recurrent_decode;
pub mod grouped_gemm;
pub mod kv_cache_append;
pub mod mla_cache_append;
pub mod moe_align_block_size;
pub mod moe_alltoall;
pub mod moe_alltoall_prepare;
pub mod moe_finalize_routing;
pub mod moe_fused_topk;
pub mod p2p_inter;
pub mod p2p_intra;
pub mod residual_rms_norm;
pub mod rms_norm;
pub mod single_gemm;
pub mod vllm_fused_moe;
pub mod vllm_mla_rope;

pub use all_reduce::{AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput, AllReduceSpec};
pub use all_reduce_residual_rms_norm::{
    AllReduceResidualRmsNormKernel, AllReduceResidualRmsNormKernelConfig,
    AllReduceResidualRmsNormKernelInput, AllReduceResidualRmsNormSpec,
};
pub use batched_gemm::{
    BatchedGemmKernel, BatchedGemmKernelConfig, BatchedGemmKernelInput, BatchedGemmSpec,
};
pub use dsa_index_cache_append::{
    DsaIndexCacheAppendKernel, DsaIndexCacheAppendKernelConfig, DsaIndexCacheAppendKernelInput,
    DsaIndexCacheAppendSpec,
};
pub use dsa_mqa_logits_prefill::{
    DsaMqaLogitsPrefillKernel, DsaMqaLogitsPrefillKernelConfig, DsaMqaLogitsPrefillKernelInput,
    DsaMqaLogitsPrefillSpec,
};
pub use dsa_paged_mqa_logits_decode::{
    DsaPagedMqaLogitsDecodeKernel, DsaPagedMqaLogitsDecodeKernelConfig,
    DsaPagedMqaLogitsDecodeKernelInput, DsaPagedMqaLogitsDecodeSpec,
};
pub use dsa_persistent_topk_decode::{
    DsaPersistentTopkDecodeKernel, DsaPersistentTopkDecodeKernelConfig,
    DsaPersistentTopkDecodeKernelInput, DsaPersistentTopkDecodeSpec,
};
pub use dsa_sparse_mla_attention::{
    DsaSparseMlaAttentionKernel, DsaSparseMlaAttentionKernelConfig,
    DsaSparseMlaAttentionKernelInput, DsaSparseMlaAttentionSpec,
};
pub use dsa_topk_prefill::{
    DsaTopkPrefillKernel, DsaTopkPrefillKernelConfig, DsaTopkPrefillKernelInput, DsaTopkPrefillSpec,
};
pub use elementwise::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, ElementwiseSpec,
};
pub use engine::{Kernel, KernelConfig, KernelSpec};
pub use flashinfer_attn_decode::{
    FlashinferAttnDecodeKernel, FlashinferAttnDecodeKernelConfig, FlashinferAttnDecodeKernelInput,
    FlashinferAttnDecodeSpec,
};
pub use flashinfer_attn_prefill::{
    FlashinferAttnPrefillKernel, FlashinferAttnPrefillKernelConfig,
    FlashinferAttnPrefillKernelInput, FlashinferAttnPrefillSpec,
};
pub use flashinfer_attn_rect::{
    FlashinferAttnRectKernel, FlashinferAttnRectKernelConfig, FlashinferAttnRectKernelInput,
    FlashinferAttnRectSpec,
};
pub use fp8_block_quant::{
    Fp8BlockQuantKernel, Fp8BlockQuantKernelConfig, Fp8BlockQuantKernelInput, Fp8BlockQuantSpec,
};
pub use fp8_blockscale_grouped_gemm::{
    Fp8BlockscaleGroupedGemmKernel, Fp8BlockscaleGroupedGemmKernelConfig,
    Fp8BlockscaleGroupedGemmKernelInput, Fp8BlockscaleGroupedGemmSpec,
};
pub use fp8_per_token_group_quant::{
    Fp8PerTokenGroupQuantKernel, Fp8PerTokenGroupQuantKernelConfig,
    Fp8PerTokenGroupQuantKernelInput, Fp8PerTokenGroupQuantSpec,
};
pub use gdn_causal_conv_decode::{
    GdnCausalConvDecodeKernel, GdnCausalConvDecodeKernelConfig, GdnCausalConvDecodeKernelInput,
    GdnCausalConvDecodeSpec,
};
pub use gdn_causal_conv_prefill::{
    GdnCausalConvPrefillKernel, GdnCausalConvPrefillKernelConfig, GdnCausalConvPrefillKernelInput,
    GdnCausalConvPrefillSpec,
};
pub use gdn_chunk_delta_rule::{
    GdnChunkDeltaRuleKernel, GdnChunkDeltaRuleKernelConfig, GdnChunkDeltaRuleKernelInput,
    GdnChunkDeltaRuleSpec,
};
pub use gdn_chunk_local_cumsum::{
    GdnChunkLocalCumsumKernel, GdnChunkLocalCumsumKernelConfig, GdnChunkLocalCumsumKernelInput,
    GdnChunkLocalCumsumSpec,
};
pub use gdn_chunk_output::{
    GdnChunkOutputKernel, GdnChunkOutputKernelConfig, GdnChunkOutputKernelInput, GdnChunkOutputSpec,
};
pub use gdn_chunk_recompute_w_u::{
    GdnChunkRecomputeWUKernel, GdnChunkRecomputeWUKernelConfig, GdnChunkRecomputeWUKernelInput,
    GdnChunkRecomputeWUSpec,
};
pub use gdn_chunk_scaled_dot_kkt::{
    GdnChunkScaledDotKktKernel, GdnChunkScaledDotKktKernelConfig, GdnChunkScaledDotKktKernelInput,
    GdnChunkScaledDotKktSpec,
};
pub use gdn_chunk_solve_tril::{
    GdnChunkSolveTrilKernel, GdnChunkSolveTrilKernelConfig, GdnChunkSolveTrilKernelInput,
    GdnChunkSolveTrilSpec,
};
pub use gdn_chunk_state_update::{
    GdnChunkStateUpdateKernel, GdnChunkStateUpdateKernelConfig, GdnChunkStateUpdateKernelInput,
    GdnChunkStateUpdateSpec,
};
pub use gdn_gated_rms_norm::{
    GdnGatedRmsNormKernel, GdnGatedRmsNormKernelConfig, GdnGatedRmsNormKernelInput,
    GdnGatedRmsNormSpec,
};
pub use gdn_prefill_post_conv::{
    GdnPrefillPostConvKernel, GdnPrefillPostConvKernelConfig, GdnPrefillPostConvKernelInput,
    GdnPrefillPostConvSpec,
};
pub use gdn_recurrent_decode::{
    GdnRecurrentDecodeKernel, GdnRecurrentDecodeKernelConfig, GdnRecurrentDecodeKernelInput,
    GdnRecurrentDecodeSpec,
};
pub use grouped_gemm::{
    GroupedGemmKernel, GroupedGemmKernelConfig, GroupedGemmKernelInput, GroupedGemmSpec,
};
pub use kv_cache_append::{
    KvCacheAppendKernel, KvCacheAppendKernelConfig, KvCacheAppendKernelInput, KvCacheAppendSpec,
};
pub use mla_cache_append::{
    MlaCacheAppendKernel, MlaCacheAppendKernelConfig, MlaCacheAppendKernelInput, MlaCacheAppendSpec,
};
pub use moe_align_block_size::{
    MoeAlignBlockSizeKernel, MoeAlignBlockSizeKernelConfig, MoeAlignBlockSizeKernelInput,
    MoeAlignBlockSizeSpec,
};
pub use moe_alltoall::{
    MoeAlltoallDirection, MoeAlltoallKernel, MoeAlltoallKernelConfig, MoeAlltoallKernelInput,
    MoeAlltoallSpec,
};
pub use moe_alltoall_prepare::{
    MoeAlltoallPrepareKernel, MoeAlltoallPrepareKernelConfig, MoeAlltoallPrepareKernelInput,
    MoeAlltoallPrepareSpec,
};
pub use moe_finalize_routing::{
    MoeFinalizeRoutingKernel, MoeFinalizeRoutingKernelConfig, MoeFinalizeRoutingKernelInput,
    MoeFinalizeRoutingSpec,
};
pub use moe_fused_topk::{
    MoeFusedTopkKernel, MoeFusedTopkKernelConfig, MoeFusedTopkKernelInput, MoeFusedTopkSpec,
};
pub use p2p_inter::{P2pInterKernel, P2pInterKernelConfig, P2pInterKernelInput, P2pInterSpec};
pub use p2p_intra::{P2pIntraKernel, P2pIntraKernelConfig, P2pIntraKernelInput, P2pIntraSpec};
pub use residual_rms_norm::{
    ResidualRmsNormKernel, ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput,
    ResidualRmsNormSpec,
};
pub use rms_norm::{RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput, RmsNormSpec};
pub use single_gemm::{
    SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput, SingleGemmSpec,
};
pub use vllm_fused_moe::{
    VllmFusedMoeKernel, VllmFusedMoeKernelConfig, VllmFusedMoeKernelInput, VllmFusedMoeSpec,
    LAUNCH_ROLE_DOWN, LAUNCH_ROLE_GATE_UP,
};
pub use vllm_mla_rope::{
    VllmMlaRopeKernel, VllmMlaRopeKernelConfig, VllmMlaRopeKernelInput, VllmMlaRopeSpec,
};
