//! GLM-5.2 NVFP4 as SGLang executes it under pure tensor parallelism on B200.
//!
//! Every rank owns all 256 experts (EP1) and shards attention heads, dense and
//! shared FFN widths, routed-expert intermediate width, and vocabulary by TP.
//! The CostTree represents one symmetric rank; physical state totals account
//! for every TP rank explicitly through the L4 contract.

use std::sync::Arc;

use anyhow::Result;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::glm52_model_cfg::{Glm52ModelCfg, Glm52MtpMode};
use crate::common::Fabric;
use crate::op::attention::DsaSparseMlaExactVarlenConfig;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    AllReduceFusionKernel, AllReduceFusionKernelConfig, AllReduceFusionKernelInput,
    AllReduceFusionSpec, AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput,
    AllReduceResidualRmsNormKernel, AllReduceResidualRmsNormKernelConfig,
    AllReduceResidualRmsNormKernelInput, AllReduceResidualRmsNormSpec, ElementwiseKernel,
    ElementwiseKernelConfig, ElementwiseKernelInput, ResidualRmsNormKernel,
    ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::routing::RoutingDistribution;
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Dim, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, Probe, SlotInput,
};
use crate::worklet::{
    Glm52DenseFfnLocalWorklet, Glm52DenseFfnLocalWorkletConfig, Glm52DenseFfnLocalWorkletInput,
    Glm52DenseFfnLocalWorkletResolved, Glm52MtpHeadLocalWorklet, Glm52MtpHeadLocalWorkletConfig,
    Glm52MtpHeadLocalWorkletInput, Glm52MtpHeadLocalWorkletResolved, Glm52MtpPreludeLocalWorklet,
    Glm52MtpPreludeLocalWorkletConfig, Glm52MtpPreludeLocalWorkletInput,
    Glm52MtpPreludeLocalWorkletResolved, Glm52SharedExpertLocalWorklet,
    Glm52SharedExpertLocalWorkletConfig, Glm52SharedExpertLocalWorkletInput,
    Glm52SharedExpertLocalWorkletResolved, Nvfp4MoeLocalWorklet, Nvfp4MoeLocalWorkletConfig,
    Nvfp4MoeLocalWorkletInput, Nvfp4MoeLocalWorkletResolved, SglangGlm52DsaAttnLocalDecodeInput,
    SglangGlm52DsaAttnLocalWorklet, SglangGlm52DsaAttnLocalWorkletConfig,
    SglangGlm52DsaAttnLocalWorkletInput, SglangGlm52DsaAttnLocalWorkletResolved,
    SglangGlm52MoeRouterLocalWorklet, SglangGlm52MoeRouterLocalWorkletConfig,
    SglangGlm52MoeRouterLocalWorkletInput, SglangGlm52MoeRouterLocalWorkletResolved,
    SglangMoeFinalizeLocalWorklet, SglangMoeFinalizeLocalWorkletConfig,
    SglangMoeFinalizeLocalWorkletInput, SglangMoeFinalizeLocalWorkletResolved,
};

const ARCH_KIND: &str = "glm52_sglang_nvfp4_tp_dsa_moe";
const NUM_LAYERS: u32 = 78;
const NUM_DENSE_LAYERS: u32 = 3;
const NUM_INITIAL_SHARED_LAYERS: u32 = 3;
const NUM_SPARSE_CYCLES: u32 = 18;
const NUM_SHARED_PER_CYCLE: u32 = 3;
const HIDDEN_DIM: u32 = 6_144;
const DENSE_INTERMEDIATE_DIM: u32 = 12_288;
const NUM_ATTN_HEADS: u32 = 64;
const RAW_NUM_KV_HEADS: u32 = 64;
const SPARSE_NUM_KV_HEADS: u32 = 1;
const Q_LORA_RANK: u32 = 2_048;
const KV_LORA_RANK: u32 = 512;
const QK_NOPE_HEAD_DIM: u32 = 192;
const ROPE_DIM: u32 = 64;
const V_HEAD_DIM: u32 = 256;
const MODEL_INDEX_HEADS: u32 = 32;
const PROFILE_INDEX_HEADS: u32 = 32;
const INDEX_HEAD_DIM: u32 = 128;
const INDEX_TOP_K: u32 = 2_048;
const CHECKPOINT_MAX_CONTEXT: u32 = 1_048_576;
const PROFILED_H32_CONTEXTS: [u32; 5] = [8_192, 65_536, 131_072, 262_144, 524_288];
const NUM_EXPERTS: u32 = 256;
const ROUTER_TOP_K: u32 = 8;
const MOE_INTERMEDIATE_DIM: u32 = 2_048;
const NUM_SHARED_EXPERTS: u32 = 1;
const VOCAB_SIZE: u32 = 154_880;
const NUM_MTP_LAYERS: u32 = 1;
const CACHE_BLOCK_SIZE: u32 = 64;
const QUANT_BLOCK_SIZE: u32 = 128;
const SOFTMAX_SCALE_DENOMINATOR: u32 = 16;
const SGLANG_MAX_FUSED_TOKENS: u32 = 2_048;
const FULL_INDEX_LAYERS: [u32; 21] = [
    0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74,
];

const RESIDUAL_NORM_BACKENDS: &[&str] = &["vllm_cuda"];
const RMS_NORM_BACKENDS: &[&str] = &["flashinfer"];
const LINEAR_BACKENDS: &[&str] = &["sglang_bf16_auto"];
const PROJECTION_BACKENDS: &[&str] = &["sglang_fused_a_auto"];
const LM_HEAD_BACKENDS: &[&str] = &["torch_linear"];
const ROUTER_BACKENDS: &[&str] = &["sglang_router_auto"];
const ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
const MAIN_ROPE_BACKENDS: &[&str] = &["flashinfer"];
const Q_ABSORB_BACKENDS: &[&str] = &["torch_mla_q_absorb_glm52"];
const V_UP_BACKENDS: &[&str] = &["torch_mla_v_up_glm52"];
const INDEXER_Q_ROPE_BACKENDS: &[&str] = &["sglang_cuda"];
const INDEX_CACHE_BACKENDS: &[&str] = &["sglang_fused_norm_rope_store"];
const INDEX_LOGITS_BACKENDS: &[&str] = &["deepgemm_fp8"];
const INDEX_PREFILL_TOPK_BACKENDS: &[&str] = &["sglang_cuda"];
const INDEX_DECODE_TOPK_BACKENDS: &[&str] = &["vllm_cuda"];
const SPARSE_ATTN_BACKENDS: &[&str] = &["flashinfer_trtllm_fp8"];
const MLA_APPEND_BACKENDS: &[&str] = &["sglang_cuda"];
const NVFP4_QUANT_BACKENDS: &[&str] = &["flashinfer_cutedsl"];
const NVFP4_MOE_BACKENDS: &[&str] = &["flashinfer_trtllm_sm100_deferred_finalize"];
const MOE_FINALIZE_BACKENDS: &[&str] = &["sglang_cuda"];
const ALLREDUCE_BACKENDS: &[&str] = &["nccl", "nvshmem"];
const FUSED_ALLREDUCE_BACKENDS: &[&str] = &["flashinfer_trtllm"];

#[derive(Clone, Debug)]
pub struct Glm52SglangNvfp4TpDsaMoeParallel {
    pub tp_size: u16,
    pub max_model_len: u32,
    pub gpu_name: String,
}

#[derive(Clone, Debug)]
pub struct Glm52SglangNvfp4TpDsaMoeConfigs {
    pub model: Glm52ModelCfg,
    pub parallel: Glm52SglangNvfp4TpDsaMoeParallel,
    pub mtp_mode: Glm52MtpMode,
    pub dense_full_index_attention: SglangGlm52DsaAttnLocalWorkletConfig,
    pub dense_ffn: Glm52DenseFfnLocalWorkletConfig,
    pub initial_shared_attention: SglangGlm52DsaAttnLocalWorkletConfig,
    pub cycle_full_attention: SglangGlm52DsaAttnLocalWorkletConfig,
    pub cycle_shared_attention: SglangGlm52DsaAttnLocalWorkletConfig,
    pub sparse_router: SglangGlm52MoeRouterLocalWorkletConfig,
    pub shared_expert: Glm52SharedExpertLocalWorkletConfig,
    pub nvfp4_moe: Nvfp4MoeLocalWorkletConfig,
    pub moe_finalize: SglangMoeFinalizeLocalWorkletConfig,
    pub tp_allreduce: AllReduceKernelConfig,
    pub tp_allreduce_fusion: AllReduceFusionKernelConfig,
    pub tp_allreduce_fused_norm: AllReduceResidualRmsNormKernelConfig,
    pub embedding: ElementwiseKernelConfig,
    pub final_norm: ResidualRmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub mtp_prelude: Option<Glm52MtpPreludeLocalWorkletConfig>,
    pub mtp_attention: Option<SglangGlm52DsaAttnLocalWorkletConfig>,
    pub mtp_head: Option<Glm52MtpHeadLocalWorkletConfig>,
}

#[derive(Clone, Debug)]
pub struct Glm52SglangNvfp4TpDsaMoeResolved {
    pub raw_cfg: Glm52SglangNvfp4TpDsaMoeConfigs,
    pub dense_full_index_attention: SglangGlm52DsaAttnLocalWorkletResolved,
    pub dense_ffn: Glm52DenseFfnLocalWorkletResolved,
    pub initial_shared_attention: SglangGlm52DsaAttnLocalWorkletResolved,
    pub cycle_full_attention: SglangGlm52DsaAttnLocalWorkletResolved,
    pub cycle_shared_attention: SglangGlm52DsaAttnLocalWorkletResolved,
    pub sparse_router: SglangGlm52MoeRouterLocalWorkletResolved,
    pub shared_expert: Glm52SharedExpertLocalWorkletResolved,
    pub nvfp4_moe: Nvfp4MoeLocalWorkletResolved,
    pub moe_finalize: SglangMoeFinalizeLocalWorkletResolved,
    pub tp_allreduce: AllReduceKernelConfig,
    pub tp_allreduce_fusion: AllReduceFusionKernelConfig,
    pub tp_allreduce_fused_norm: AllReduceResidualRmsNormKernelConfig,
    pub embedding: ElementwiseKernelConfig,
    pub final_norm: ResidualRmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub mtp_prelude: Option<Glm52MtpPreludeLocalWorkletResolved>,
    pub mtp_attention: Option<SglangGlm52DsaAttnLocalWorkletResolved>,
    pub mtp_head: Option<Glm52MtpHeadLocalWorkletResolved>,
}

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

fn sglang_fusion_cap(spec_cap: u32) -> u32 {
    spec_cap.min(SGLANG_MAX_FUSED_TOKENS)
}

fn attention_config(
    model: &Glm52ModelCfg,
    parallel: &Glm52SglangNvfp4TpDsaMoeParallel,
    include_indexer: bool,
) -> SglangGlm52DsaAttnLocalWorkletConfig {
    SglangGlm52DsaAttnLocalWorkletConfig {
        include_indexer,
        tp_size: parallel.tp_size,
        residual_rms_norm_backends: RESIDUAL_NORM_BACKENDS.to_vec(),
        rms_norm_backends: RMS_NORM_BACKENDS.to_vec(),
        projection_gemm_backends: PROJECTION_BACKENDS.to_vec(),
        output_gemm_backends: LINEAR_BACKENDS.to_vec(),
        main_rope_backends: MAIN_ROPE_BACKENDS.to_vec(),
        q_absorb_backends: Q_ABSORB_BACKENDS.to_vec(),
        v_up_backends: V_UP_BACKENDS.to_vec(),
        indexer_gemm_backends: LINEAR_BACKENDS.to_vec(),
        indexer_elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        indexer_q_rope_backends: INDEXER_Q_ROPE_BACKENDS.to_vec(),
        index_cache_append_backends: INDEX_CACHE_BACKENDS.to_vec(),
        index_prefill_logits_backends: INDEX_LOGITS_BACKENDS.to_vec(),
        index_prefill_topk_backends: INDEX_PREFILL_TOPK_BACKENDS.to_vec(),
        index_decode_logits_backends: INDEX_LOGITS_BACKENDS.to_vec(),
        index_decode_topk_backends: INDEX_DECODE_TOPK_BACKENDS.to_vec(),
        sparse_attention_backends: SPARSE_ATTN_BACKENDS.to_vec(),
        sparse_mla_cache_append_backends: MLA_APPEND_BACKENDS.to_vec(),
        gpu_name: parallel.gpu_name.clone(),
        hidden_dim: model.hidden_dim.clone(),
        num_attention_heads: model.num_attention_heads.clone(),
        num_kv_heads: Dim::param("sparse_num_kv_heads", SPARSE_NUM_KV_HEADS),
        q_lora_rank: model.q_lora_rank.clone(),
        kv_lora_rank: model.kv_lora_rank.clone(),
        qk_nope_head_dim: model.qk_nope_head_dim.clone(),
        rope_dim: model.rope_dim.clone(),
        v_head_dim: model.v_head_dim.clone(),
        model_num_index_heads: model.model_num_index_heads.clone(),
        profile_num_index_heads: Dim::param("profile_num_index_heads", PROFILE_INDEX_HEADS),
        index_head_dim: model.index_head_dim.clone(),
        selected_k: model.index_top_k,
        max_model_len: Dim::param("max_model_len", parallel.max_model_len),
        rope_max_position: model.max_context.clone(),
        logits_row_stride: Dim::param("logits_row_stride", parallel.max_model_len),
        cache_block_size: CACHE_BLOCK_SIZE,
        quant_block_size: QUANT_BLOCK_SIZE,
        softmax_scale_denominator: SOFTMAX_SCALE_DENOMINATOR,
        base_dtype: DType::Bf16,
        index_cache_dtype: DType::Fp8E4m3,
        index_q_dtype: DType::Fp8E4m3,
        scale_dtype: DType::Fp32,
        weight_dtype: DType::Fp32,
        logits_dtype: DType::Fp32,
        index_dtype: "int32".to_string(),
        index_scale_format: "fp32".to_string(),
        index_cache_format: "page_planar_fp8_fp32_scale".to_string(),
        index_prefill_span_mode: "single_causal_tail".to_string(),
        index_decode_context_mode: "uniform".to_string(),
        index_decode_page_mapping: "unique_scattered".to_string(),
        rope_is_neox_style: false,
        index_clean_logits: false,
        // The later F83 audit rejected the uniform-scatter diagnostic as a
        // production identity: real DSA has recency bias, and the current
        // runtime input carries per-request context rather than the global KV
        // pool size that makes scattered access expensive. Keep the accepted
        // contiguous cache curve until both axes are modeled together.
        sparse_index_distribution: "recent_contiguous".to_string(),
        sparse_cache_layout: "hnd_paged_mqa_fp8_latent_rope".to_string(),
        sparse_mla_cache_format: "plain".to_string(),
        sparse_attention_q_dtype: DType::Fp8E4m3,
        sparse_attention_cache_dtype: DType::Fp8E4m3,
        sparse_attention_output_dtype: DType::Bf16,
        sparse_exact_varlen: DsaSparseMlaExactVarlenConfig {
            prefill_backends: SPARSE_ATTN_BACKENDS.to_vec(),
            max_model_len: parallel.max_model_len,
            prefill_index_distribution: "recent_contiguous".to_string(),
            page_table_mapping: None,
        },
        decode_next_n: 1,
    }
}

pub fn build_configs(
    model: &Glm52ModelCfg,
    parallel: &Glm52SglangNvfp4TpDsaMoeParallel,
    routing: &RoutingDistribution,
    fp8: bool,
    mtp_mode: Glm52MtpMode,
) -> Result<Glm52SglangNvfp4TpDsaMoeConfigs, BuildError> {
    validate_model_cfg(model).map_err(fit_failed)?;
    if parallel.tp_size != 4 {
        return Err(fit_failed(format!(
            "only the profiled TP4 deployment is runtime-ready; got tp_size {}",
            parallel.tp_size
        )));
    }
    if MOE_INTERMEDIATE_DIM % u32::from(parallel.tp_size) != 0 {
        return Err(fit_failed(format!(
            "moe intermediate {MOE_INTERMEDIATE_DIM} must divide tp_size {}",
            parallel.tp_size
        )));
    }
    if !PROFILED_H32_CONTEXTS.contains(&parallel.max_model_len) {
        return Err(fit_failed(format!(
            "max_model_len {} is not a profiled H32 config identity; expected one of {PROFILED_H32_CONTEXTS:?}",
            parallel.max_model_len,
        )));
    }
    if routing.num_experts() != NUM_EXPERTS {
        return Err(fit_failed(format!(
            "routing has {} experts, expected {NUM_EXPERTS}",
            routing.num_experts()
        )));
    }
    if fp8 {
        return Err(fit_failed(
            "this NVFP4 arch requires model fp8=false; routed quantization is encoded by the arch",
        ));
    }

    let gpu = parallel.gpu_name.clone();
    let hidden_bytes = HIDDEN_DIM
        .checked_mul(DType::Bf16.size_bytes())
        .ok_or_else(|| fit_failed("hidden byte width overflows u32"))?;
    let embedding_input = hidden_bytes
        .checked_add(8)
        .ok_or_else(|| fit_failed("embedding input byte width overflows u32"))?;
    let mtp_attention = match mtp_mode {
        Glm52MtpMode::Off => None,
        Glm52MtpMode::FullIndex => Some(attention_config(model, parallel, true)),
        Glm52MtpMode::IndexShare => Some(attention_config(model, parallel, false)),
    };
    let mtp_prelude = (mtp_mode != Glm52MtpMode::Off).then(|| Glm52MtpPreludeLocalWorkletConfig {
        elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        rms_norm_backends: RMS_NORM_BACKENDS.to_vec(),
        gemm_backends: LINEAR_BACKENDS.to_vec(),
        gpu_name: gpu.clone(),
        hidden_dim: model.hidden_dim.clone(),
        vocab_size: model.vocab_size.clone(),
        dtype: DType::Bf16,
        gemm_dtype: DType::Bf16,
        token_id_bytes: 8,
        position_bytes: 8,
    });
    let mtp_head = (mtp_mode != Glm52MtpMode::Off).then(|| Glm52MtpHeadLocalWorkletConfig {
        elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        rms_norm_backends: RMS_NORM_BACKENDS.to_vec(),
        gemm_backends: LINEAR_BACKENDS.to_vec(),
        gpu_name: gpu.clone(),
        hidden_dim: model.hidden_dim.clone(),
        vocab_size: model.vocab_size.clone(),
        // Same divisor the main lm_head below uses.
        tp_size: parallel.tp_size,
        dtype: DType::Bf16,
        gemm_dtype: DType::Bf16,
    });

    let nvfp4_moe = Nvfp4MoeLocalWorkletConfig::replicated_for_tp(
        Nvfp4MoeLocalWorkletConfig {
            hidden: model.hidden_dim.clone(),
            moe_intermediate: model.moe_intermediate_dim.clone(),
            num_experts: model.num_experts.clone(),
            ep_size: 1,
            tp_size: parallel.tp_size,
            top_k: model.router_top_k,
            activation_dtype: DType::Bf16,
            gpu_name: gpu.clone(),
            quant_backends: NVFP4_QUANT_BACKENDS.to_vec(),
            moe_backends: NVFP4_MOE_BACKENDS.to_vec(),
            weight_format: "nvfp4_e2m1".to_string(),
            group_size: 16,
            routing_method: "minimax2".to_string(),
            n_group: 1,
            topk_group: 1,
            routed_scaling_numerator: 5,
            routed_scaling_denominator: 2,
            layerwise_global_ppm: Vec::new(),
            token_corpus: None,
            folded_rank_position: 0,
        },
        routing,
        NUM_LAYERS - NUM_DENSE_LAYERS,
    );

    Ok(Glm52SglangNvfp4TpDsaMoeConfigs {
        model: model.clone(),
        parallel: parallel.clone(),
        mtp_mode,
        dense_full_index_attention: attention_config(model, parallel, true),
        dense_ffn: Glm52DenseFfnLocalWorkletConfig {
            residual_norm_backends: RESIDUAL_NORM_BACKENDS.to_vec(),
            gemm_backends: LINEAR_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
            tp_size: parallel.tp_size,
            gpu_name: gpu.clone(),
            hidden_dim: model.hidden_dim.clone(),
            intermediate_dim: model.dense_intermediate_dim.clone(),
            dtype: DType::Bf16,
            gemm_dtype: DType::Bf16,
        },
        initial_shared_attention: attention_config(model, parallel, false),
        cycle_full_attention: attention_config(model, parallel, true),
        cycle_shared_attention: attention_config(model, parallel, false),
        sparse_router: SglangGlm52MoeRouterLocalWorkletConfig {
            residual_norm_backends: RESIDUAL_NORM_BACKENDS.to_vec(),
            gate_gemm_backends: ROUTER_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden_dim: model.hidden_dim.clone(),
            num_experts: model.num_experts.clone(),
            top_k: model.router_top_k,
            base_dtype: DType::Bf16,
            router_output_dtype: DType::Fp32,
            index_dtype: "int32".to_string(),
            scoring_func: "sigmoid".to_string(),
            topk_method: "noaux_tc".to_string(),
            norm_topk_prob: true,
            routed_scaling_numerator: 5,
            routed_scaling_denominator: 2,
        },
        shared_expert: Glm52SharedExpertLocalWorkletConfig {
            gemm_backends: LINEAR_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
            tp_size: parallel.tp_size,
            gpu_name: gpu.clone(),
            hidden_dim: model.hidden_dim.clone(),
            moe_intermediate_dim: model.moe_intermediate_dim.clone(),
            n_shared_experts: model.num_shared_experts,
            dtype: DType::Bf16,
            gemm_dtype: DType::Bf16,
        },
        nvfp4_moe,
        moe_finalize: SglangMoeFinalizeLocalWorkletConfig {
            backends: MOE_FINALIZE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            top_k: model.router_top_k,
            hidden_dim: model.hidden_dim.clone(),
            dtype: DType::Bf16,
            fuse_shared_output: true,
        },
        tp_allreduce: AllReduceKernelConfig {
            backends: ALLREDUCE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            num_gpus: u32::from(parallel.tp_size),
            fabric: Fabric::Nvlink,
        },
        tp_allreduce_fusion: AllReduceFusionKernelConfig {
            backends: FUSED_ALLREDUCE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            num_gpus: u32::from(parallel.tp_size),
            hidden_dim: HIDDEN_DIM,
            dtype: DType::Bf16,
            fabric: Fabric::Nvlink,
        },
        tp_allreduce_fused_norm: AllReduceResidualRmsNormKernelConfig {
            backends: FUSED_ALLREDUCE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            num_gpus: u32::from(parallel.tp_size),
            hidden_dim: HIDDEN_DIM,
            dtype: DType::Bf16,
            fabric: Fabric::Nvlink,
            strategy: "auto".to_string(),
            launch_with_pdl: true,
            fp32_acc: true,
        },
        embedding: ElementwiseKernelConfig {
            backends: ELEMENTWISE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            input_bytes_per_token: embedding_input.into(),
            output_bytes_per_token: hidden_bytes.into(),
        },
        final_norm: ResidualRmsNormKernelConfig {
            backends: RESIDUAL_NORM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden: model.hidden_dim.clone(),
            dtype: DType::Bf16,
        },
        lm_head: SingleGemmKernelConfig {
            backends: LM_HEAD_BACKENDS.to_vec(),
            gpu_name: gpu,
            n: model.vocab_size.clone() / u32::from(parallel.tp_size),
            k: model.hidden_dim.clone(),
            dtype: DType::Bf16,
        },
        mtp_prelude,
        mtp_attention,
        mtp_head,
    })
}

fn validate_model_cfg(model: &Glm52ModelCfg) -> std::result::Result<(), String> {
    for (name, actual, required) in [
        ("hidden_dim", model.hidden_dim.get(), HIDDEN_DIM),
        (
            "dense_intermediate_dim",
            model.dense_intermediate_dim.get(),
            DENSE_INTERMEDIATE_DIM,
        ),
        (
            "num_attention_heads",
            model.num_attention_heads.get(),
            NUM_ATTN_HEADS,
        ),
        (
            "raw_num_kv_heads",
            model.raw_num_kv_heads.get(),
            RAW_NUM_KV_HEADS,
        ),
        ("q_lora_rank", model.q_lora_rank.get(), Q_LORA_RANK),
        ("kv_lora_rank", model.kv_lora_rank.get(), KV_LORA_RANK),
        (
            "qk_nope_head_dim",
            model.qk_nope_head_dim.get(),
            QK_NOPE_HEAD_DIM,
        ),
        ("rope_dim", model.rope_dim.get(), ROPE_DIM),
        ("v_head_dim", model.v_head_dim.get(), V_HEAD_DIM),
        (
            "model_num_index_heads",
            model.model_num_index_heads.get(),
            MODEL_INDEX_HEADS,
        ),
        ("index_head_dim", model.index_head_dim.get(), INDEX_HEAD_DIM),
        ("index_top_k", model.index_top_k, INDEX_TOP_K),
        (
            "max_context",
            model.max_context.get(),
            CHECKPOINT_MAX_CONTEXT,
        ),
        ("num_experts", model.num_experts.get(), NUM_EXPERTS),
        ("router_top_k", model.router_top_k, ROUTER_TOP_K),
        (
            "moe_intermediate_dim",
            model.moe_intermediate_dim.get(),
            MOE_INTERMEDIATE_DIM,
        ),
        (
            "num_shared_experts",
            model.num_shared_experts,
            NUM_SHARED_EXPERTS,
        ),
        ("vocab_size", model.vocab_size.get(), VOCAB_SIZE),
        ("num_layers", model.num_layers, NUM_LAYERS),
        ("num_mtp_layers", model.num_mtp_layers, NUM_MTP_LAYERS),
    ] {
        if actual != required {
            return Err(format!("{name} must be {required}, got {actual}"));
        }
    }
    if model.dtype != DType::Bf16 || model.router_semantic_dtype != DType::Fp32 {
        return Err("model dtypes must be BF16 base and FP32 router".to_string());
    }
    if model.full_index_layers != FULL_INDEX_LAYERS {
        return Err("full-index schedule must match GLM-5.2".to_string());
    }
    if model.indexer_types.len() != NUM_LAYERS as usize
        || model.indexer_types.iter().enumerate().any(|(layer, kind)| {
            let expected = if FULL_INDEX_LAYERS.contains(&(layer as u32)) {
                "full"
            } else {
                "shared"
            };
            kind != expected
        })
    {
        return Err("indexer_types must match GLM-5.2".to_string());
    }
    if model.mlp_layer_types.len() != NUM_LAYERS as usize
        || model
            .mlp_layer_types
            .iter()
            .enumerate()
            .any(|(layer, kind)| {
                kind != if layer < NUM_DENSE_LAYERS as usize {
                    "dense"
                } else {
                    "sparse"
                }
            })
    {
        return Err("mlp_layer_types must match GLM-5.2".to_string());
    }
    if !model.index_share_for_mtp_iteration {
        return Err("index_share_for_mtp_iteration must be enabled".to_string());
    }
    Ok(())
}

pub fn resolve_configs(cfgs: &Glm52SglangNvfp4TpDsaMoeConfigs) -> Glm52SglangNvfp4TpDsaMoeResolved {
    Glm52SglangNvfp4TpDsaMoeResolved {
        dense_full_index_attention: SglangGlm52DsaAttnLocalWorklet::resolve_config(
            &cfgs.dense_full_index_attention,
        ),
        dense_ffn: Glm52DenseFfnLocalWorklet::resolve_config(&cfgs.dense_ffn),
        initial_shared_attention: SglangGlm52DsaAttnLocalWorklet::resolve_config(
            &cfgs.initial_shared_attention,
        ),
        cycle_full_attention: SglangGlm52DsaAttnLocalWorklet::resolve_config(
            &cfgs.cycle_full_attention,
        ),
        cycle_shared_attention: SglangGlm52DsaAttnLocalWorklet::resolve_config(
            &cfgs.cycle_shared_attention,
        ),
        sparse_router: SglangGlm52MoeRouterLocalWorklet::resolve_config(&cfgs.sparse_router),
        shared_expert: Glm52SharedExpertLocalWorklet::resolve_config(&cfgs.shared_expert),
        nvfp4_moe: Nvfp4MoeLocalWorklet::resolve_config(&cfgs.nvfp4_moe),
        moe_finalize: SglangMoeFinalizeLocalWorklet::resolve_config(&cfgs.moe_finalize),
        tp_allreduce: cfgs.tp_allreduce.clone(),
        tp_allreduce_fusion: cfgs.tp_allreduce_fusion.clone(),
        tp_allreduce_fused_norm: cfgs.tp_allreduce_fused_norm.clone(),
        embedding: cfgs.embedding.clone(),
        final_norm: cfgs.final_norm.clone(),
        lm_head: cfgs.lm_head.clone(),
        mtp_prelude: cfgs
            .mtp_prelude
            .as_ref()
            .map(Glm52MtpPreludeLocalWorklet::resolve_config),
        mtp_attention: cfgs
            .mtp_attention
            .as_ref()
            .map(SglangGlm52DsaAttnLocalWorklet::resolve_config),
        mtp_head: cfgs
            .mtp_head
            .as_ref()
            .map(Glm52MtpHeadLocalWorklet::resolve_config),
        raw_cfg: cfgs.clone(),
    }
}

struct SparseBody {
    name: String,
    attention: SglangGlm52DsaAttnLocalWorklet,
    router: SglangGlm52MoeRouterLocalWorklet,
    shared_expert: Glm52SharedExpertLocalWorklet,
    routed_expert: Nvfp4MoeLocalWorklet,
    finalize: SglangMoeFinalizeLocalWorklet,
    attention_allreduce: Op<AllReduceKernel>,
    attention_allreduce_fused_norm: Op<AllReduceResidualRmsNormKernel>,
    attention_fusion_cap: u32,
    ffn_allreduce: Op<AllReduceKernel>,
    ffn_allreduce_fused_norm: Op<AllReduceResidualRmsNormKernel>,
    ffn_fusion_cap: u32,
    tp_size: u16,
    top_k: u32,
}

impl SparseBody {
    fn build(
        name: String,
        attention: SglangGlm52DsaAttnLocalWorkletResolved,
        common: &Glm52SglangNvfp4TpDsaMoeResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let attention =
            SglangGlm52DsaAttnLocalWorklet::build(format!("{name}.attention"), attention, bridge)?;
        let router = SglangGlm52MoeRouterLocalWorklet::build(
            format!("{name}.moe.router"),
            common.sparse_router.clone(),
            bridge,
        )?;
        let shared_expert = Glm52SharedExpertLocalWorklet::build(
            format!("{name}.moe.shared_expert"),
            common.shared_expert.clone(),
            bridge,
        )?;
        let routed_expert = Nvfp4MoeLocalWorklet::build(
            format!("{name}.moe.routed_experts"),
            common.nvfp4_moe.clone(),
            bridge,
        )?;
        let finalize = SglangMoeFinalizeLocalWorklet::build(
            format!("{name}.moe.routed_experts"),
            common.moe_finalize.clone(),
            bridge,
        )?;
        let attention_allreduce = build_atomic(
            format!("{name}.attention.tp_allreduce"),
            common.tp_allreduce.clone(),
            AllReduceKernel::build,
            bridge,
        )?;
        let attention_allreduce_fused_norm = build_atomic(
            format!("{name}.attention.tp_allreduce_residual_norm"),
            common.tp_allreduce_fused_norm.clone(),
            AllReduceResidualRmsNormKernel::build,
            bridge,
        )?;
        let ffn_allreduce = build_atomic(
            format!("{name}.moe.tp_allreduce_fallback"),
            common.tp_allreduce.clone(),
            AllReduceKernel::build,
            bridge,
        )?;
        let ffn_allreduce_fused_norm = build_atomic(
            format!("{name}.moe.tp_allreduce"),
            common.tp_allreduce_fused_norm.clone(),
            AllReduceResidualRmsNormKernel::build,
            bridge,
        )?;
        let fusion_cap = sglang_fusion_cap(AllReduceResidualRmsNormSpec::max_fused_tokens(
            &common.tp_allreduce_fused_norm,
        ));
        Ok(Self {
            name,
            attention,
            router,
            shared_expert,
            routed_expert,
            finalize,
            attention_allreduce,
            attention_allreduce_fused_norm,
            attention_fusion_cap: fusion_cap,
            ffn_allreduce,
            ffn_allreduce_fused_norm,
            ffn_fusion_cap: fusion_cap,
            tp_size: common.raw_cfg.parallel.tp_size,
            top_k: common.raw_cfg.model.router_top_k,
        })
    }

    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        // Slot indices are allocated at compile-call time, so mint in exactly
        // the order the root below and `eval` traverse these sections.
        let attention = self.attention.compile(builder);
        let attention_allreduce = self.attention_allreduce.compile(builder);
        let attention_allreduce_fused_norm = self.attention_allreduce_fused_norm.compile(builder);
        let router = self.router.compile(builder);
        let shared_expert = self.shared_expert.compile(builder);
        let routed_expert = self.routed_expert.compile(builder);
        let finalize = self.finalize.compile(builder);
        let ffn_allreduce = self.ffn_allreduce.compile(builder);
        let ffn_allreduce_fused_norm = self.ffn_allreduce_fused_norm.compile(builder);
        let local_experts = CostNode::Labeled {
            label: format!("{}.moe.local_experts [SGLang dual stream]", self.name),
            child: Box::new(CostNode::Max {
                overlap: 1.0,
                children: vec![shared_expert, CostNode::Sum(vec![routed_expert, finalize])],
            }),
        };
        CostNode::Labeled {
            label: format!(
                "{} [sparse layer; TP{} EP1; top-{} NVFP4]",
                self.name, self.tp_size, self.top_k
            ),
            child: Box::new(CostNode::Sum(vec![
                attention,
                attention_allreduce,
                attention_allreduce_fused_norm,
                CostNode::Labeled {
                    label: format!("{}.moe", self.name),
                    child: Box::new(CostNode::Sum(vec![
                        router,
                        local_experts,
                        ffn_allreduce,
                        ffn_allreduce_fused_norm,
                    ])),
                },
            ])),
        }
    }

    fn eval(&self, batch: &NormalizedBatch, ev: &mut Evaluator) {
        let group = &batch.group;
        let attention_fused = uses_fusion(batch.total_tokens, self.attention_fusion_cap);
        let preceding_ffn_fused = uses_fusion(batch.total_tokens, self.ffn_fusion_cap);
        if preceding_ffn_fused {
            self.attention
                .eval_with_input_norm(&group.attention_input, false, ev);
        } else {
            self.attention.eval(&group.attention_input, ev);
        }
        eval_allreduce_pair(
            &self.attention_allreduce,
            &self.attention_allreduce_fused_norm,
            batch.total_tokens,
            attention_fused,
            ev,
        );
        let router_input = SglangGlm52MoeRouterLocalWorkletInput {
            batch_tokens: group.batch_tokens,
        };
        if attention_fused {
            self.router
                .eval_with_post_attn_norm(&router_input, false, ev);
        } else {
            self.router.eval(&router_input, ev);
        }
        self.shared_expert.eval(
            &Glm52SharedExpertLocalWorkletInput {
                batch_tokens: group.batch_tokens,
            },
            ev,
        );
        self.routed_expert.eval(
            &Nvfp4MoeLocalWorkletInput {
                num_tokens: group.batch_tokens,
            },
            ev,
        );
        self.finalize.eval(
            &SglangMoeFinalizeLocalWorkletInput {
                num_tokens: group.batch_tokens,
            },
            ev,
        );
        eval_allreduce_pair(
            &self.ffn_allreduce,
            &self.ffn_allreduce_fused_norm,
            batch.total_tokens,
            preceding_ffn_fused,
            ev,
        );
    }
}

struct MtpSection {
    prelude: Glm52MtpPreludeLocalWorklet,
    decoder: SparseBody,
    head: Glm52MtpHeadLocalWorklet,
}

pub struct Glm52SglangNvfp4TpDsaMoeModel {
    pub name: String,
    pub mtp_mode: Glm52MtpMode,
    pub tp_size: u16,
    pub max_model_len: u32,
    pub total_state_bytes_per_token: u64,
    embedding: Op<ElementwiseKernel>,
    embedding_allreduce: Op<AllReduceKernel>,
    embedding_allreduce_fusion: Op<AllReduceFusionKernel>,
    embedding_fusion_cap: u32,
    dense_attention: SglangGlm52DsaAttnLocalWorklet,
    dense_ffn: Glm52DenseFfnLocalWorklet,
    dense_attention_allreduce: Op<AllReduceKernel>,
    dense_attention_allreduce_fused_norm: Op<AllReduceResidualRmsNormKernel>,
    dense_attention_fusion_cap: u32,
    dense_ffn_allreduce: Op<AllReduceKernel>,
    dense_ffn_allreduce_fused_norm: Op<AllReduceResidualRmsNormKernel>,
    dense_ffn_fusion_cap: u32,
    initial_shared_sparse: SparseBody,
    cycle_full_sparse: SparseBody,
    cycle_shared_sparse: SparseBody,
    final_norm: Op<ResidualRmsNormKernel>,
    lm_head: Op<SingleGemmKernel>,
    mtp: Option<MtpSection>,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: Glm52SglangNvfp4TpDsaMoeResolved,
    bridge: &PerfApiBridge,
) -> Result<Glm52SglangNvfp4TpDsaMoeModel, BuildError> {
    let tp_size = resolved.raw_cfg.parallel.tp_size;
    let max_model_len = resolved.raw_cfg.parallel.max_model_len;
    let mtp_mode = resolved.raw_cfg.mtp_mode;
    let embedding = build_atomic(
        format!("{name}.main.embedding"),
        resolved.embedding.clone(),
        ElementwiseKernel::build,
        bridge,
    )?;
    let embedding_allreduce = build_atomic(
        format!("{name}.main.embedding.tp_allreduce_fallback"),
        resolved.tp_allreduce.clone(),
        AllReduceKernel::build,
        bridge,
    )?;
    let embedding_allreduce_fusion = build_atomic(
        format!("{name}.main.embedding.tp_allreduce"),
        resolved.tp_allreduce_fusion.clone(),
        AllReduceFusionKernel::build,
        bridge,
    )?;
    let dense_attention = SglangGlm52DsaAttnLocalWorklet::build(
        format!("{name}.body.dense_full_index.attention"),
        resolved.dense_full_index_attention.clone(),
        bridge,
    )?;
    let dense_ffn = Glm52DenseFfnLocalWorklet::build(
        format!("{name}.body.dense_full_index.ffn"),
        resolved.dense_ffn.clone(),
        bridge,
    )?;
    let dense_attention_allreduce = build_atomic(
        format!("{name}.body.dense_full_index.attention.tp_allreduce"),
        resolved.tp_allreduce.clone(),
        AllReduceKernel::build,
        bridge,
    )?;
    let dense_attention_allreduce_fused_norm = build_atomic(
        format!("{name}.body.dense_full_index.attention.tp_allreduce_residual_norm"),
        resolved.tp_allreduce_fused_norm.clone(),
        AllReduceResidualRmsNormKernel::build,
        bridge,
    )?;
    let dense_ffn_allreduce = build_atomic(
        format!("{name}.body.dense_full_index.ffn.tp_allreduce_fallback"),
        resolved.tp_allreduce.clone(),
        AllReduceKernel::build,
        bridge,
    )?;
    let dense_ffn_allreduce_fused_norm = build_atomic(
        format!("{name}.body.dense_full_index.ffn.tp_allreduce"),
        resolved.tp_allreduce_fused_norm.clone(),
        AllReduceResidualRmsNormKernel::build,
        bridge,
    )?;
    let initial_shared_sparse = SparseBody::build(
        format!("{name}.body.sparse_initial_index_share"),
        resolved.initial_shared_attention.clone(),
        &resolved,
        bridge,
    )?;
    let cycle_full_sparse = SparseBody::build(
        format!("{name}.body.sparse_cycle_full_index"),
        resolved.cycle_full_attention.clone(),
        &resolved,
        bridge,
    )?;
    let cycle_shared_sparse = SparseBody::build(
        format!("{name}.body.sparse_cycle_index_share"),
        resolved.cycle_shared_attention.clone(),
        &resolved,
        bridge,
    )?;
    let final_norm = build_atomic(
        format!("{name}.main.final_residual_rms_norm"),
        resolved.final_norm.clone(),
        ResidualRmsNormKernel::build,
        bridge,
    )?;
    let lm_head = build_atomic(
        format!("{name}.main.lm_head"),
        resolved.lm_head.clone(),
        SingleGemmKernel::build,
        bridge,
    )?;
    let mtp = match (
        resolved.mtp_prelude.clone(),
        resolved.mtp_attention.clone(),
        resolved.mtp_head.clone(),
    ) {
        (None, None, None) => None,
        (Some(prelude), Some(attention), Some(head)) => Some(MtpSection {
            prelude: Glm52MtpPreludeLocalWorklet::build(
                format!("{name}.mtp.prelude"),
                prelude,
                bridge,
            )?,
            decoder: SparseBody::build(
                format!("{name}.mtp.decoder_sparse"),
                attention,
                &resolved,
                bridge,
            )?,
            head: Glm52MtpHeadLocalWorklet::build(format!("{name}.mtp.head"), head, bridge)?,
        }),
        _ => return Err(fit_failed("MTP sections must be all present or all absent")),
    };
    let residual_fusion_cap = sglang_fusion_cap(AllReduceResidualRmsNormSpec::max_fused_tokens(
        &resolved.tp_allreduce_fused_norm,
    ));
    let mut model = Glm52SglangNvfp4TpDsaMoeModel {
        name,
        mtp_mode,
        tp_size,
        max_model_len,
        total_state_bytes_per_token: state_bytes_per_token(tp_size, mtp_mode)?,
        embedding,
        embedding_allreduce,
        embedding_allreduce_fusion,
        embedding_fusion_cap: sglang_fusion_cap(AllReduceFusionSpec::max_fused_tokens(
            &resolved.tp_allreduce_fusion,
        )),
        dense_attention,
        dense_ffn,
        dense_attention_allreduce,
        dense_attention_allreduce_fused_norm,
        dense_attention_fusion_cap: residual_fusion_cap,
        dense_ffn_allreduce,
        dense_ffn_allreduce_fused_norm,
        dense_ffn_fusion_cap: residual_fusion_cap,
        initial_shared_sparse,
        cycle_full_sparse,
        cycle_shared_sparse,
        final_norm,
        lm_head,
        mtp,
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

impl Glm52SglangNvfp4TpDsaMoeModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        // Compilation mints slot indices immediately. Keep these calls in the
        // same order as the root below and `eval_into`; merely placing a
        // precompiled node earlier in `children` does not reorder its slots.
        let embedding = self.embedding.compile(&mut builder);
        let embedding_allreduce = self.embedding_allreduce.compile(&mut builder);
        let embedding_allreduce_fusion = self.embedding_allreduce_fusion.compile(&mut builder);
        let dense = CostNode::Labeled {
            label: "layers 0..2: dense + full index".to_string(),
            child: Box::new(CostNode::Scale {
                n: NUM_DENSE_LAYERS,
                child: Box::new(CostNode::Sum(vec![
                    self.dense_attention.compile(&mut builder),
                    self.dense_attention_allreduce.compile(&mut builder),
                    self.dense_attention_allreduce_fused_norm
                        .compile(&mut builder),
                    self.dense_ffn.compile(&mut builder),
                    self.dense_ffn_allreduce.compile(&mut builder),
                    self.dense_ffn_allreduce_fused_norm.compile(&mut builder),
                ])),
            }),
        };
        let initial_shared = CostNode::Labeled {
            label: "layers 3..5: sparse + IndexShare".to_string(),
            child: Box::new(CostNode::Scale {
                n: NUM_INITIAL_SHARED_LAYERS,
                child: Box::new(self.initial_shared_sparse.compile(&mut builder)),
            }),
        };
        let cycle = CostNode::Labeled {
            label: "layers 6..77: 18 × (full index + 3 IndexShare)".to_string(),
            child: Box::new(CostNode::Scale {
                n: NUM_SPARSE_CYCLES,
                child: Box::new(CostNode::Sum(vec![
                    self.cycle_full_sparse.compile(&mut builder),
                    CostNode::Scale {
                        n: NUM_SHARED_PER_CYCLE,
                        child: Box::new(self.cycle_shared_sparse.compile(&mut builder)),
                    },
                ])),
            }),
        };
        let final_norm = self.final_norm.compile(&mut builder);
        let lm_head = self.lm_head.compile(&mut builder);
        let mut children = vec![
            embedding,
            embedding_allreduce,
            embedding_allreduce_fusion,
            dense,
            initial_shared,
            cycle,
            final_norm,
            lm_head,
        ];
        if let Some(mtp) = &self.mtp {
            children.push(CostNode::Labeled {
                label: format!("MTP layer 78 [{:?}; decode-only]", self.mtp_mode),
                child: Box::new(CostNode::Sum(vec![
                    mtp.prelude.compile(&mut builder),
                    mtp.decoder.compile(&mut builder),
                    mtp.head.compile(&mut builder),
                ])),
            });
        }
        builder.finish(CostNode::Labeled {
            label: format!(
                "{} (Glm52SglangNvfp4TpDsaMoeModel) [TP{} EP1; per-rank; MTP={:?}; context<={}]",
                self.name, self.tp_size, self.mtp_mode, self.max_model_len
            ),
            child: Box::new(CostNode::Sum(children)),
        })
    }

    fn eval_into(&self, input: &UnifiedArchInput, ev: &mut Evaluator) {
        let batch = normalize_input(input, self.max_model_len)
            .unwrap_or_else(|reason| panic!("invalid SGLang GLM-5.2 input: {reason}"));
        let group = &batch.group;
        let embedding_fused = uses_fusion(batch.total_tokens, self.embedding_fusion_cap);
        eval_atomic_or_zero(
            &self.embedding,
            ElementwiseKernelInput {
                num_tokens: group.batch_tokens,
            },
            group.batch_tokens == 0,
            ev,
        );
        eval_atomic_or_zero(
            &self.embedding_allreduce,
            AllReduceKernelInput {
                message_size_bytes: message_bytes(batch.total_tokens),
            },
            batch.total_tokens == 0 || embedding_fused,
            ev,
        );
        eval_atomic_or_zero(
            &self.embedding_allreduce_fusion,
            AllReduceFusionKernelInput {
                num_tokens: batch.total_tokens,
            },
            !embedding_fused,
            ev,
        );

        let dense_attention_fused =
            uses_fusion(batch.total_tokens, self.dense_attention_fusion_cap);
        let preceding_dense_ffn_fused = uses_fusion(batch.total_tokens, self.dense_ffn_fusion_cap);
        if preceding_dense_ffn_fused {
            self.dense_attention
                .eval_with_input_norm(&group.attention_input, false, ev);
        } else {
            self.dense_attention.eval(&group.attention_input, ev);
        }
        eval_allreduce_pair(
            &self.dense_attention_allreduce,
            &self.dense_attention_allreduce_fused_norm,
            batch.total_tokens,
            dense_attention_fused,
            ev,
        );
        self.dense_ffn.eval_with_post_attn_norm(
            &Glm52DenseFfnLocalWorkletInput {
                batch_tokens: group.batch_tokens,
            },
            !dense_attention_fused,
            ev,
        );
        eval_allreduce_pair(
            &self.dense_ffn_allreduce,
            &self.dense_ffn_allreduce_fused_norm,
            batch.total_tokens,
            preceding_dense_ffn_fused,
            ev,
        );

        self.initial_shared_sparse.eval(&batch, ev);
        self.cycle_full_sparse.eval(&batch, ev);
        self.cycle_shared_sparse.eval(&batch, ev);
        eval_atomic_or_zero(
            &self.final_norm,
            ResidualRmsNormKernelInput {
                m: group.batch_tokens,
            },
            group.batch_tokens == 0,
            ev,
        );
        eval_atomic_or_zero(
            &self.lm_head,
            SingleGemmKernelInput {
                m: group.request_count,
            },
            group.request_count == 0,
            ev,
        );

        if let Some(mtp) = &self.mtp {
            mtp.prelude.eval(
                &Glm52MtpPreludeLocalWorkletInput {
                    batch_tokens: group.decode_tokens,
                },
                ev,
            );
            mtp.decoder.eval(&batch.decode_only(), ev);
            mtp.head.eval(
                &Glm52MtpHeadLocalWorkletInput {
                    batch_tokens: group.decode_tokens,
                },
                ev,
            );
        }
    }
}

impl IterwiseUnifiedModel for Glm52SglangNvfp4TpDsaMoeModel {
    fn total_kv_bytes_per_token(&self) -> u64 {
        self.total_state_bytes_per_token
    }

    fn gpus_per_replica(&self) -> u16 {
        self.tp_size
    }

    fn num_attn_dp_groups(&self) -> u16 {
        1
    }

    fn num_attn_shards(&self) -> u16 {
        self.tp_size
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut evaluator = Evaluator::new(slots);
        self.eval_into(batch, &mut evaluator);
        assert_eq!(evaluator.filled(), self.n_slots);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut evaluator = Evaluator::with_inputs(slots, inputs);
        self.eval_into(batch, &mut evaluator);
        assert_eq!(evaluator.filled(), self.n_slots);
        let total = CostTree::aggregate(&self.cost_flat, slots, scratch);
        assert_eq!(inputs.len(), self.n_slots);
        total
    }
}

#[derive(Clone, Debug)]
struct NormalizedGroup {
    batch_tokens: u32,
    decode_tokens: u32,
    request_count: u32,
    decode_context: Option<u32>,
    attention_input: SglangGlm52DsaAttnLocalWorkletInput,
}

#[derive(Clone, Debug)]
struct NormalizedBatch {
    group: NormalizedGroup,
    total_tokens: u32,
}

impl NormalizedBatch {
    fn decode_only(&self) -> Self {
        let context_lens = self
            .group
            .attention_input
            .decode
            .as_ref()
            .and_then(|decode| decode.context_lens.clone());
        let group = NormalizedGroup {
            batch_tokens: self.group.decode_tokens,
            decode_tokens: self.group.decode_tokens,
            request_count: self.group.decode_tokens,
            decode_context: self.group.decode_context,
            attention_input: SglangGlm52DsaAttnLocalWorkletInput {
                num_new_tokens: self.group.decode_tokens,
                prefill_query_cache_pairs: Vec::new(),
                decode: self.group.decode_context.map(|context_len| {
                    SglangGlm52DsaAttnLocalDecodeInput {
                        batch_size: self.group.decode_tokens,
                        context_len,
                        context_lens,
                        requires_padding: false,
                    }
                }),
            },
        };
        Self {
            total_tokens: group.batch_tokens,
            group,
        }
    }
}

fn normalize_input(
    input: &UnifiedArchInput,
    max_model_len: u32,
) -> std::result::Result<NormalizedBatch, String> {
    if input.groups.len() != 1 {
        return Err(format!(
            "expected exactly one attention group, got {}",
            input.groups.len()
        ));
    }
    let source = &input.groups[0];
    let mut prefill_tokens = 0_u32;
    let mut prefill_query_cache_pairs = Vec::with_capacity(source.prefill_chunk_pairs.len());
    for (request, &(prefix, append)) in source.prefill_chunk_pairs.iter().enumerate() {
        if append == 0 {
            return Err(format!("prefill request {request} append must be nonzero"));
        }
        let context = prefix
            .checked_add(append)
            .ok_or_else(|| format!("prefill request {request} context overflows u32"))?;
        if context > max_model_len {
            return Err(format!(
                "prefill request {request} context {context} exceeds {max_model_len}"
            ));
        }
        prefill_tokens = prefill_tokens
            .checked_add(append)
            .ok_or_else(|| "prefill token sum overflows u32".to_string())?;
        prefill_query_cache_pairs.push((append, context));
    }
    if source.prefill_tokens != prefill_tokens {
        return Err(format!(
            "prefill_tokens {} must equal append sum {prefill_tokens}",
            source.prefill_tokens
        ));
    }
    let decode_tokens = u32::try_from(source.decode_kv_lens.len())
        .map_err(|_| "decode request count exceeds u32".to_string())?;
    if source.decode_tokens != decode_tokens {
        return Err(format!(
            "decode_tokens {} must equal context count {decode_tokens}",
            source.decode_tokens
        ));
    }
    let mut decode_context = None;
    for (request, &context) in source.decode_kv_lens.iter().enumerate() {
        if !(1..=max_model_len).contains(&context) {
            return Err(format!(
                "decode request {request} context {context} must be in 1..={max_model_len}"
            ));
        }
        decode_context = Some(decode_context.map_or(context, |current: u32| current.max(context)));
    }
    let batch_tokens = prefill_tokens
        .checked_add(decode_tokens)
        .ok_or_else(|| "batch token sum overflows u32".to_string())?;
    if source.batch_tokens != batch_tokens {
        return Err(format!(
            "batch_tokens {} must equal prefill+decode {batch_tokens}",
            source.batch_tokens
        ));
    }
    let request_count =
        u32::try_from(source.prefill_chunk_pairs.len() + source.decode_kv_lens.len())
            .map_err(|_| "request count exceeds u32".to_string())?;
    Ok(NormalizedBatch {
        total_tokens: batch_tokens,
        group: NormalizedGroup {
            batch_tokens,
            decode_tokens,
            request_count,
            decode_context,
            attention_input: SglangGlm52DsaAttnLocalWorkletInput {
                num_new_tokens: batch_tokens,
                prefill_query_cache_pairs,
                decode: decode_context.map(|context_len| SglangGlm52DsaAttnLocalDecodeInput {
                    batch_size: decode_tokens,
                    context_len,
                    context_lens: Some(source.decode_kv_lens.clone()),
                    requires_padding: false,
                }),
            },
        },
    })
}

fn state_bytes_per_token(
    tp_size: u16,
    mtp_mode: Glm52MtpMode,
) -> std::result::Result<u64, BuildError> {
    let mla_per_rank = u64::from(NUM_LAYERS)
        .checked_mul(u64::from(KV_LORA_RANK + ROPE_DIM))
        .ok_or_else(|| fit_failed("MLA state overflow"))?;
    let index_per_rank = u64::from(NUM_LAYERS)
        .checked_mul(u64::from(INDEX_HEAD_DIM + DType::Fp32.size_bytes()))
        .ok_or_else(|| fit_failed("index state overflow"))?;
    let main = mla_per_rank
        .checked_add(index_per_rank)
        .ok_or_else(|| fit_failed("main state overflow"))?;
    let mtp_mla = u64::from(KV_LORA_RANK + ROPE_DIM);
    let per_rank = match mtp_mode {
        Glm52MtpMode::Off => main,
        Glm52MtpMode::FullIndex => main
            .checked_add(mtp_mla)
            .and_then(|value| value.checked_add(u64::from(INDEX_HEAD_DIM + 4)))
            .ok_or_else(|| fit_failed("MTP full-index state overflow"))?,
        Glm52MtpMode::IndexShare => main
            .checked_add(mtp_mla)
            .ok_or_else(|| fit_failed("MTP shared state overflow"))?,
    };
    per_rank
        .checked_mul(u64::from(tp_size))
        .ok_or_else(|| fit_failed("physical TP state overflow"))
}

fn uses_fusion(num_tokens: u32, cap: u32) -> bool {
    num_tokens > 0 && num_tokens <= cap
}

fn message_bytes(num_tokens: u32) -> u64 {
    u64::from(num_tokens) * u64::from(HIDDEN_DIM) * u64::from(DType::Bf16.size_bytes())
}

fn eval_allreduce_pair(
    fallback: &Op<AllReduceKernel>,
    fused: &Op<AllReduceResidualRmsNormKernel>,
    num_tokens: u32,
    use_fused: bool,
    ev: &mut Evaluator,
) {
    eval_atomic_or_zero(
        fallback,
        AllReduceKernelInput {
            message_size_bytes: message_bytes(num_tokens),
        },
        num_tokens == 0 || use_fused,
        ev,
    );
    eval_atomic_or_zero(
        fused,
        AllReduceResidualRmsNormKernelInput { num_tokens },
        !use_fused,
        ev,
    );
}

fn build_atomic<K, C, F>(
    name: String,
    config: C,
    build_kernel: F,
    bridge: &PerfApiBridge,
) -> Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> std::result::Result<K, BuildError>,
{
    Ok(Op::new(
        name.clone(),
        Arc::new(build_kernel(name, config, bridge)?),
    ))
}

fn eval_atomic_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        op.kernel.eval(&input)
    };
    ev.push(metrics, || input.clone().into());
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::Path;

    use super::*;
    use crate::arch::contract::ArchGroupInput;

    fn exact_json_value() -> serde_json::Value {
        let indexer_types: Vec<&str> = (0..NUM_LAYERS)
            .map(|layer| {
                if FULL_INDEX_LAYERS.contains(&layer) {
                    "full"
                } else {
                    "shared"
                }
            })
            .collect();
        let mlp_layer_types: Vec<&str> = (0..NUM_LAYERS)
            .map(|layer| {
                if layer < NUM_DENSE_LAYERS {
                    "dense"
                } else {
                    "sparse"
                }
            })
            .collect();
        let mut value: serde_json::Value = serde_json::from_str(
            r#"{
                "architectures":["GlmMoeDsaForCausalLM"],
                "model_type":"glm_moe_dsa","dtype":"bfloat16",
                "hidden_size":6144,"intermediate_size":12288,
                "num_hidden_layers":78,"first_k_dense_replace":3,
                "num_attention_heads":64,"num_key_value_heads":64,"head_dim":192,
                "q_lora_rank":2048,"kv_lora_rank":512,"qk_head_dim":256,
                "qk_nope_head_dim":192,"qk_rope_head_dim":64,"v_head_dim":256,
                "index_n_heads":32,"index_head_dim":128,"index_topk":2048,
                "index_skip_topk_offset":3,"index_topk_freq":4,"index_topk_pattern":null,
                "index_share_for_mtp_iteration":true,"indexer_rope_interleave":true,
                "rope_interleave":true,"max_position_embeddings":1048576,
                "n_routed_experts":256,"num_experts_per_tok":8,
                "moe_intermediate_size":2048,"n_shared_experts":1,"moe_layer_freq":1,
                "moe_router_dtype":"float32","scoring_func":"sigmoid",
                "topk_method":"noaux_tc","n_group":1,"topk_group":1,
                "norm_topk_prob":true,"routed_scaling_factor":2.5,
                "vocab_size":154880,"num_nextn_predict_layers":1
            }"#,
        )
        .unwrap();
        value["mlp_layer_types"] = serde_json::to_value(mlp_layer_types).unwrap();
        value["indexer_types"] = serde_json::to_value(indexer_types).unwrap();
        value
    }

    fn model() -> Glm52ModelCfg {
        crate::arch::glm52_model_cfg::parse_model_json(&exact_json_value().to_string()).unwrap()
    }

    fn parallel(tp_size: u16) -> Glm52SglangNvfp4TpDsaMoeParallel {
        Glm52SglangNvfp4TpDsaMoeParallel {
            tp_size,
            max_model_len: 8_192,
            gpu_name: "NVIDIA B200".to_string(),
        }
    }

    fn configs(mtp_mode: Glm52MtpMode) -> Glm52SglangNvfp4TpDsaMoeConfigs {
        build_configs(
            &model(),
            &parallel(4),
            &RoutingDistribution::uniform(NUM_EXPERTS),
            false,
            mtp_mode,
        )
        .unwrap()
    }

    fn built(mtp_mode: Glm52MtpMode) -> Glm52SglangNvfp4TpDsaMoeModel {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        build(
            "unified".to_string(),
            resolve_configs(&configs(mtp_mode)),
            &bridge,
        )
        .unwrap()
    }

    #[test]
    fn configs_encode_sglang_tp4_launch_graph_and_pure_tp_partition() {
        let cfg = configs(Glm52MtpMode::Off);
        for attention in [
            &cfg.dense_full_index_attention,
            &cfg.initial_shared_attention,
            &cfg.cycle_full_attention,
            &cfg.cycle_shared_attention,
        ] {
            assert_eq!(attention.tp_size, 4);
            assert_eq!(
                attention.profile_num_index_heads,
                Dim::param("profile_num_index_heads", 32)
            );
            assert_eq!(
                attention.projection_gemm_backends,
                vec!["sglang_fused_a_auto"]
            );
            assert_eq!(attention.output_gemm_backends, vec!["sglang_bf16_auto"]);
            assert_eq!(attention.indexer_q_rope_backends, vec!["sglang_cuda"]);
            assert_eq!(
                attention.index_cache_append_backends,
                vec!["sglang_fused_norm_rope_store"]
            );
            assert_eq!(
                attention.sparse_attention_backends,
                vec!["flashinfer_trtllm_fp8"]
            );
            assert_eq!(attention.sparse_exact_varlen.max_model_len, 8_192);
            assert_eq!(
                attention.sparse_exact_varlen.prefill_index_distribution,
                "recent_contiguous"
            );
            assert_eq!(attention.sparse_exact_varlen.page_table_mapping, None);
        }
        assert!(cfg.dense_full_index_attention.include_indexer);
        assert!(!cfg.initial_shared_attention.include_indexer);
        assert_eq!(cfg.dense_ffn.tp_size, 4);
        assert_eq!(cfg.shared_expert.tp_size, 4);
        assert_eq!(cfg.nvfp4_moe.ep_size, 1);
        assert_eq!(cfg.nvfp4_moe.tp_size, 4);
        assert_eq!(cfg.nvfp4_moe.quant_backends, vec!["flashinfer_cutedsl"]);
        assert_eq!(
            cfg.nvfp4_moe.moe_backends,
            vec!["flashinfer_trtllm_sm100_deferred_finalize"]
        );
        assert_eq!(cfg.moe_finalize.backends, vec!["sglang_cuda"]);
    }

    #[test]
    fn resolved_shapes_shard_only_tp_owned_axes() {
        let resolved = resolve_configs(&configs(Glm52MtpMode::Off));
        assert_eq!(resolved.dense_full_index_attention.main_rope.num_heads, 16);
        assert_eq!(resolved.dense_full_index_attention.q_absorb.num_batches, 16);
        let indexer = resolved
            .dense_full_index_attention
            .indexer
            .as_ref()
            .unwrap();
        assert_eq!(indexer.model_num_index_heads, MODEL_INDEX_HEADS);
        assert_eq!(resolved.dense_ffn.gate_up_proj.n, 6_144);
        assert_eq!(resolved.shared_expert.gate_up_proj.n, 1_024);
        assert_eq!(resolved.nvfp4_moe.experts_per_device, NUM_EXPERTS);
        assert_eq!(resolved.nvfp4_moe.intermediate_per_rank.get(), 512);
        assert_eq!(resolved.nvfp4_moe.fused_moe.num_local_experts, NUM_EXPERTS);
    }

    #[test]
    fn both_lm_heads_shard_the_vocabulary_by_the_same_degree() {
        // The MTP output head is the same `ParallelLMHead` as the main head,
        // one layer later. Billing one sharded and the other whole charges this
        // rank for every other rank's logits.
        let resolved = resolve_configs(&configs(Glm52MtpMode::FullIndex));
        let mtp_head = resolved.mtp_head.as_ref().unwrap();

        assert_eq!(resolved.lm_head.n, mtp_head.lm_head.n);
        assert_eq!(mtp_head.lm_head.n, VOCAB_SIZE / 4);
        assert_eq!(mtp_head.vocab_size_per_rank, VOCAB_SIZE / 4);
        // Hidden is never sharded, on either head.
        assert_eq!(resolved.lm_head.k, mtp_head.lm_head.k);
    }

    fn count_max_nodes(node: &CostNode) -> usize {
        match node {
            CostNode::Leaf(_) => 0,
            CostNode::Sum(children) => children.iter().map(count_max_nodes).sum(),
            CostNode::Max { children, .. } => {
                1 + children.iter().map(count_max_nodes).sum::<usize>()
            }
            CostNode::Scale { child, .. } | CostNode::Labeled { child, .. } => {
                count_max_nodes(child)
            }
        }
    }

    #[test]
    fn tree_is_one_symmetric_rank_with_dual_stream_sparse_experts() {
        let model = built(Glm52MtpMode::Off);
        let tree = model.cost_tree();
        // Three sparse section families are compiled once each. Their routed
        // path includes finalize and overlaps the shared-expert path via Max.
        assert_eq!(count_max_nodes(&tree.root), 3);
        let description = tree.describe();
        assert!(description.contains("TP4 EP1; per-rank"));
        assert!(!description.contains("Max over 4 ranks"));

        let slot_names: Vec<&str> = tree.slots.iter().map(|slot| slot.name.as_str()).collect();
        assert_eq!(slot_names[0], "unified.main.embedding");
        assert_eq!(
            slot_names[1],
            "unified.main.embedding.tp_allreduce_fallback"
        );
        assert_eq!(slot_names[2], "unified.main.embedding.tp_allreduce");
        for body in [
            "unified.body.sparse_initial_index_share",
            "unified.body.sparse_cycle_full_index",
            "unified.body.sparse_cycle_index_share",
        ] {
            assert!(
                slot_names.contains(&format!("{body}.moe.router.router_gemm_bf16_proxy").as_str())
            );
            assert!(slot_names.contains(&format!("{body}.moe.routed_experts.finalize").as_str()));
            assert!(slot_names
                .contains(&format!("{body}.attention.tp_allreduce_residual_norm").as_str()));
            assert!(slot_names.contains(&format!("{body}.moe.tp_allreduce").as_str()));

            let first_attention = slot_names
                .iter()
                .position(|name| name.starts_with(&format!("{body}.attention.")))
                .unwrap();
            let router_name = format!("{body}.moe.router.router_gemm_bf16_proxy");
            let router = slot_names
                .iter()
                .position(|name| *name == router_name)
                .unwrap();
            let first_shared = slot_names
                .iter()
                .position(|name| name.starts_with(&format!("{body}.moe.shared_expert.")))
                .unwrap();
            let routed_name = format!("{body}.moe.routed_experts.input_quant");
            let routed = slot_names
                .iter()
                .position(|name| *name == routed_name)
                .unwrap();
            let finalize_name = format!("{body}.moe.routed_experts.finalize");
            let finalize = slot_names
                .iter()
                .position(|name| *name == finalize_name)
                .unwrap();
            let collective_name = format!("{body}.moe.tp_allreduce_fallback");
            let collective = slot_names
                .iter()
                .position(|name| *name == collective_name)
                .unwrap();
            assert!(first_attention < router);
            assert!(router < first_shared);
            assert!(first_shared < routed);
            assert!(routed < finalize);
            assert!(finalize < collective);
        }

        assert_eq!(model.embedding_fusion_cap, SGLANG_MAX_FUSED_TOKENS);
        assert_eq!(model.dense_attention_fusion_cap, SGLANG_MAX_FUSED_TOKENS);
        assert_eq!(model.dense_ffn_fusion_cap, SGLANG_MAX_FUSED_TOKENS);
        assert_eq!(
            model.initial_shared_sparse.attention_fusion_cap,
            SGLANG_MAX_FUSED_TOKENS
        );
        assert_eq!(
            model.initial_shared_sparse.ffn_fusion_cap,
            SGLANG_MAX_FUSED_TOKENS
        );
    }

    #[test]
    fn location_map_matches_every_unique_noncommunication_manifest_location() {
        let model = built(Glm52MtpMode::Off);
        let is_communication = |kind: &str| {
            matches!(
                kind,
                "all_reduce" | "all_reduce_fusion" | "all_reduce_residual_rms_norm"
            )
        };
        let manifest_locations: BTreeSet<String> = model
            .cost_log_manifest()
            .slots
            .into_iter()
            .filter(|slot| !is_communication(&slot.kind))
            .map(|slot| slot.name)
            .collect();

        let map_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("model/work/location_maps/glm52_sglang_nvfp4_tp_dsa_moe_unified.json");
        let map: serde_json::Value =
            serde_json::from_slice(&std::fs::read(map_path).unwrap()).unwrap();
        let mapped_locations: BTreeSet<String> = map["locations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["location"].as_str().unwrap().to_string())
            .collect();

        assert_eq!(manifest_locations.len(), 103);
        assert_eq!(mapped_locations, manifest_locations);
    }

    #[test]
    fn topology_state_and_mtp_modes_are_exact() {
        assert_eq!(
            state_bytes_per_token(4, Glm52MtpMode::Off).unwrap(),
            220_896
        );
        assert_eq!(
            state_bytes_per_token(4, Glm52MtpMode::FullIndex).unwrap(),
            223_728
        );
        assert_eq!(
            state_bytes_per_token(4, Glm52MtpMode::IndexShare).unwrap(),
            223_200
        );
        for mode in [
            Glm52MtpMode::Off,
            Glm52MtpMode::FullIndex,
            Glm52MtpMode::IndexShare,
        ] {
            let model = built(mode);
            assert_eq!(model.gpus_per_replica(), 4);
            assert_eq!(model.num_attn_dp_groups(), 1);
            assert_eq!(model.num_attn_shards(), 4);
            assert_eq!(model.mtp.is_some(), mode != Glm52MtpMode::Off);
        }
    }

    fn group(prefill_tokens: u32, decode_lens: Vec<u32>, pairs: Vec<(u32, u32)>) -> ArchGroupInput {
        let decode_tokens = decode_lens.len() as u32;
        ArchGroupInput {
            batch_tokens: prefill_tokens + decode_tokens,
            prefill_tokens,
            decode_tokens,
            prefill_chunk_pairs: pairs,
            decode_kv_lens: decode_lens,
            total_kv_len: 0,
        }
    }

    #[test]
    fn input_lowering_preserves_exact_request_shapes_without_padding() {
        let input = UnifiedArchInput {
            groups: vec![group(5, vec![64, 91], vec![(3, 2), (0, 3)])],
            tokens_per_source_rank: Vec::new(),
        };
        let normalized = normalize_input(&input, 8_192).unwrap();
        assert_eq!(normalized.total_tokens, 7);
        assert_eq!(
            normalized.group.attention_input.prefill_query_cache_pairs,
            vec![(2, 5), (3, 3)]
        );
        let decode = normalized.group.attention_input.decode.as_ref().unwrap();
        assert_eq!(decode.batch_size, 2);
        assert_eq!(decode.context_len, 91);
        assert_eq!(decode.context_lens.as_deref(), Some(&[64, 91][..]));
        assert!(!decode.requires_padding);
        assert_eq!(normalized.group.request_count, 4);

        let mtp = normalized.decode_only();
        assert_eq!(mtp.total_tokens, 2);
        assert!(mtp
            .group
            .attention_input
            .prefill_query_cache_pairs
            .is_empty());
        assert_eq!(
            mtp.group
                .attention_input
                .decode
                .as_ref()
                .and_then(|value| value.context_lens.as_deref()),
            Some(&[64, 91][..])
        );
    }

    #[test]
    fn invalid_parallel_and_batch_contracts_fail_closed() {
        let routing = RoutingDistribution::uniform(NUM_EXPERTS);
        for tp_size in [0, 1, 2, 3, 6, 8, 16] {
            assert!(build_configs(
                &model(),
                &parallel(tp_size),
                &routing,
                false,
                Glm52MtpMode::Off,
            )
            .is_err());
        }
        for max_model_len in PROFILED_H32_CONTEXTS {
            let mut profiled = parallel(4);
            profiled.max_model_len = max_model_len;
            assert!(
                build_configs(&model(), &profiled, &routing, false, Glm52MtpMode::Off,).is_ok()
            );
        }
        for max_model_len in [1, 32_768, 524_289, 1_048_576] {
            let mut unsupported = parallel(4);
            unsupported.max_model_len = max_model_len;
            assert!(
                build_configs(&model(), &unsupported, &routing, false, Glm52MtpMode::Off,).is_err()
            );
        }
        let wrong_group_count = UnifiedArchInput {
            groups: vec![ArchGroupInput::default(); 2],
            tokens_per_source_rank: Vec::new(),
        };
        assert!(normalize_input(&wrong_group_count, 8_192).is_err());
        let over_context = UnifiedArchInput {
            groups: vec![group(0, vec![8_193], vec![])],
            tokens_per_source_rank: Vec::new(),
        };
        assert!(normalize_input(&over_context, 8_192).is_err());
    }
}
