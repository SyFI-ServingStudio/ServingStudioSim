//! Per-kind L1 kernel structs.

pub mod all_reduce;
pub mod all_reduce_fusion;
pub mod all_reduce_residual_rms_norm;
pub mod batched_gemm;
pub mod bf16_fused_moe;
pub mod clamped_swiglu;
pub mod compressed_sparse_mla_decode;
pub mod compressed_sparse_mla_prefill;
pub mod dsa_compressed_mqa_logits_prefill;
pub mod dsa_compressed_topk_prefill;
pub mod dsa_index_cache_append;
pub mod dsa_indexer_q_rope_quant;
pub mod dsa_indexer_q_rope_quant_weight_fold;
pub mod dsa_mqa_logits_prefill;
pub mod dsa_paged_mqa_logits_decode;
pub mod dsa_persistent_topk_decode;
pub mod dsa_sparse_index_remap;
pub mod dsa_sparse_mla_attention;
pub mod dsa_sparse_mla_prefill;
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
pub mod gdn_gated_rms_norm;
pub mod gdn_prefill_post_conv;
pub mod gdn_recurrent_decode;
pub mod gemm_fp32_output;
pub mod grouped_gemm;
pub mod inv_rope_fp8_quant;
pub mod kda_chunk_prefill;
pub mod kda_recurrent_decode;
pub mod kv_cache_append;
pub mod kv_compress_store;
pub mod logits_topk;
pub mod mhc_fused_post_pre_rms_norm;
pub mod mhc_pre_rms_norm;
pub mod mhc_terminal_head;
pub mod mla_cache_append;
pub mod mla_rope_quantize_fp8;
pub mod moe_align_block_size;
pub mod moe_alltoall;
pub mod moe_alltoall_prepare;
pub mod moe_ep_all_gather;
pub mod moe_ep_reduce_scatter;
pub mod moe_finalize_fuse_shared;
pub mod moe_finalize_routing;
pub mod moe_fused_topk;
pub mod moe_sum;
pub mod moe_topk_softplus_sqrt;
pub mod mxfp4_marlin_moe_gemm;
pub mod nvfp4_fused_moe;
pub mod nvfp4_quant;
pub mod p2p_inter;
pub mod p2p_intra;
pub mod packed_kv_cache_gather;
pub mod q_kv_rms_norm;
pub mod qnorm_rope_kv_insert;
pub mod residual_rms_norm;
pub mod rms_norm;
pub mod single_gemm;
pub mod vllm_fused_moe;
pub mod vllm_mla_rope;

pub use all_reduce::{AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput, AllReduceSpec};
pub use all_reduce_fusion::{
    AllReduceFusionKernel, AllReduceFusionKernelConfig, AllReduceFusionKernelInput,
    AllReduceFusionSpec,
};
pub use all_reduce_residual_rms_norm::{
    AllReduceResidualRmsNormKernel, AllReduceResidualRmsNormKernelConfig,
    AllReduceResidualRmsNormKernelInput, AllReduceResidualRmsNormSpec,
};
pub use batched_gemm::{
    BatchedGemmKernel, BatchedGemmKernelConfig, BatchedGemmKernelInput, BatchedGemmSpec,
};
pub use bf16_fused_moe::{
    Bf16FusedMoeKernel, Bf16FusedMoeKernelConfig, Bf16FusedMoeKernelInput, Bf16FusedMoeSpec,
};
pub use clamped_swiglu::{
    ClampedSwigluKernel, ClampedSwigluKernelConfig, ClampedSwigluKernelInput, ClampedSwigluSpec,
};
pub use compressed_sparse_mla_decode::{
    CompressedSparseMlaDecodeKernel, CompressedSparseMlaDecodeKernelConfig,
    CompressedSparseMlaDecodeKernelInput, CompressedSparseMlaDecodeSpec,
};
pub use compressed_sparse_mla_prefill::{
    CompressedSparseMlaPrefillKernel, CompressedSparseMlaPrefillKernelConfig,
    CompressedSparseMlaPrefillKernelInput, CompressedSparseMlaPrefillSpec,
};
pub use dsa_compressed_mqa_logits_prefill::{
    DsaCompressedMqaLogitsPrefillKernel, DsaCompressedMqaLogitsPrefillKernelConfig,
    DsaCompressedMqaLogitsPrefillSpec, DsaCompressedPrefillKernelInput,
};
pub use dsa_compressed_topk_prefill::{
    DsaCompressedTopkPrefillKernel, DsaCompressedTopkPrefillKernelConfig,
    DsaCompressedTopkPrefillSpec,
};
pub use dsa_index_cache_append::{
    DsaIndexCacheAppendKernel, DsaIndexCacheAppendKernelConfig, DsaIndexCacheAppendKernelInput,
    DsaIndexCacheAppendSpec,
};
pub use dsa_indexer_q_rope_quant::{
    DsaIndexerQRopeQuantKernel, DsaIndexerQRopeQuantKernelConfig, DsaIndexerQRopeQuantKernelInput,
    DsaIndexerQRopeQuantSpec,
};
pub use dsa_indexer_q_rope_quant_weight_fold::{
    DsaIndexerQRopeQuantWeightFoldKernel, DsaIndexerQRopeQuantWeightFoldKernelConfig,
    DsaIndexerQRopeQuantWeightFoldKernelInput, DsaIndexerQRopeQuantWeightFoldSpec,
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
pub use dsa_sparse_index_remap::{
    DsaSparseIndexRemapKernel, DsaSparseIndexRemapKernelConfig, DsaSparseIndexRemapKernelInput,
    DsaSparseIndexRemapSpec, DsaSparseIndexRemapWorkspacePartition,
};
pub use dsa_sparse_mla_attention::{
    DsaSparseMlaAttentionKernel, DsaSparseMlaAttentionKernelConfig,
    DsaSparseMlaAttentionKernelInput, DsaSparseMlaAttentionSpec, ValidCountsPattern,
};
pub use dsa_sparse_mla_prefill::{
    DsaSparseMlaPrefillKernel, DsaSparseMlaPrefillKernelConfig, DsaSparseMlaPrefillKernelInput,
    DsaSparseMlaPrefillSpec,
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
pub use gemm_fp32_output::{
    GemmFp32OutputKernel, GemmFp32OutputKernelConfig, GemmFp32OutputKernelInput, GemmFp32OutputSpec,
};
pub use grouped_gemm::{
    GroupedGemmKernel, GroupedGemmKernelConfig, GroupedGemmKernelInput, GroupedGemmSpec,
};
pub use inv_rope_fp8_quant::{
    InvRopeFp8QuantKernel, InvRopeFp8QuantKernelConfig, InvRopeFp8QuantKernelInput,
    InvRopeFp8QuantSpec,
};
pub use kda_chunk_prefill::{
    KdaChunkPrefillKernel, KdaChunkPrefillKernelConfig, KdaChunkPrefillKernelInput,
    KdaChunkPrefillSpec,
};
pub use kda_recurrent_decode::{
    KdaRecurrentDecodeKernel, KdaRecurrentDecodeKernelConfig, KdaRecurrentDecodeKernelInput,
    KdaRecurrentDecodeSpec,
};
pub use kv_cache_append::{
    KvCacheAppendKernel, KvCacheAppendKernelConfig, KvCacheAppendKernelInput, KvCacheAppendSpec,
};
pub use kv_compress_store::{
    KvCompressStoreKernel, KvCompressStoreKernelConfig, KvCompressStoreKernelInput,
    KvCompressStoreSpec,
};
pub use logits_topk::{
    LogitsTopkKernel, LogitsTopkKernelConfig, LogitsTopkKernelInput, LogitsTopkSpec,
};
pub use mhc_fused_post_pre_rms_norm::{MhcFusedPostPreRmsNormKernel, MhcFusedPostPreRmsNormSpec};
pub use mhc_pre_rms_norm::{
    MhcPreRmsNormKernel, MhcPreRmsNormSpec, MhcRmsNormKernelConfig, MhcRmsNormKernelInput,
};
pub use mhc_terminal_head::{MhcTerminalHeadKernel, MhcTerminalHeadSpec};
pub use mla_cache_append::{
    MlaCacheAppendKernel, MlaCacheAppendKernelConfig, MlaCacheAppendKernelInput, MlaCacheAppendSpec,
};
pub use mla_rope_quantize_fp8::{
    MlaRopeQuantizeFp8Kernel, MlaRopeQuantizeFp8KernelConfig, MlaRopeQuantizeFp8KernelInput,
    MlaRopeQuantizeFp8Spec,
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
pub use moe_ep_all_gather::{
    MoeEpAllGatherKernel, MoeEpAllGatherKernelConfig, MoeEpAllGatherSpec,
    MoeEpCollectiveKernelInput,
};
pub use moe_ep_reduce_scatter::{
    MoeEpReduceScatterKernel, MoeEpReduceScatterKernelConfig, MoeEpReduceScatterSpec,
};
pub use moe_finalize_fuse_shared::{
    MoeFinalizeFuseSharedKernel, MoeFinalizeFuseSharedKernelConfig,
    MoeFinalizeFuseSharedKernelInput, MoeFinalizeFuseSharedSpec,
};
pub use moe_finalize_routing::{
    MoeFinalizeRoutingKernel, MoeFinalizeRoutingKernelConfig, MoeFinalizeRoutingKernelInput,
    MoeFinalizeRoutingSpec,
};
pub use moe_fused_topk::{
    MoeFusedTopkKernel, MoeFusedTopkKernelConfig, MoeFusedTopkKernelInput, MoeFusedTopkSpec,
};
pub use moe_sum::{MoeSumKernel, MoeSumKernelConfig, MoeSumKernelInput, MoeSumSpec};
pub use moe_topk_softplus_sqrt::{
    MoeTopkSoftplusSqrtKernel, MoeTopkSoftplusSqrtKernelConfig, MoeTopkSoftplusSqrtKernelInput,
    MoeTopkSoftplusSqrtSpec,
};
pub use mxfp4_marlin_moe_gemm::{
    Mxfp4MarlinMoeFcRole, Mxfp4MarlinMoeGemmKernel, Mxfp4MarlinMoeGemmKernelConfig,
    Mxfp4MarlinMoeGemmKernelInput, Mxfp4MarlinMoeGemmSpec,
};
pub use nvfp4_fused_moe::{
    Nvfp4FusedMoeKernel, Nvfp4FusedMoeKernelConfig, Nvfp4FusedMoeKernelInput, Nvfp4FusedMoeSpec,
};
pub use nvfp4_quant::{
    Nvfp4QuantKernel, Nvfp4QuantKernelConfig, Nvfp4QuantKernelInput, Nvfp4QuantSpec,
};
pub use p2p_inter::{P2pInterKernel, P2pInterKernelConfig, P2pInterKernelInput, P2pInterSpec};
pub use p2p_intra::{P2pIntraKernel, P2pIntraKernelConfig, P2pIntraKernelInput, P2pIntraSpec};
pub use packed_kv_cache_gather::{
    PackedKvCacheGatherKernel, PackedKvCacheGatherKernelConfig, PackedKvCacheGatherKernelInput,
    PackedKvCacheGatherMode, PackedKvCacheGatherSpec,
};
pub use q_kv_rms_norm::{
    QKvRmsNormKernel, QKvRmsNormKernelConfig, QKvRmsNormKernelInput, QKvRmsNormSpec,
};
pub use qnorm_rope_kv_insert::{
    QnormRopeKvInsertKernel, QnormRopeKvInsertKernelConfig, QnormRopeKvInsertKernelInput,
    QnormRopeKvInsertSpec,
};
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
