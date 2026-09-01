//! GLM-5.2 NVIDIA NVFP4 architecture aligned to vLLM on B200.
//!
//! Tensor parallelism and expert parallelism use the same configurable rank
//! group. EP4 is the profiled preset, while expert ownership and attention-head
//! sharding are derived from `ep_size` so EP8 can use the same graph. The DSA
//! indexer remains replicated at the checkpoint's 32 heads, matching the trace.
//!
//! The checkpoint quantizes routed expert linears only. Attention, dense FFN,
//! and shared experts remain BF16; routed experts use explicit TRTLLM NVFP4
//! activation quantization and fused MoE leaves. Sparse MLA uses the traced B200
//! FlashInfer TRTLLM-gen path with FP8 E4M3 query/cache and BF16 output.
//!
//! Communication is rank-local compute followed by group all-reduce. vLLM
//! fuses the attention all-reduce with the following residual RMSNorm below a
//! GPU-specific size limit; this graph selects that measured fused leaf and
//! retains standalone all-reduce + norm as the large-shape fallback.
//!
//! The checkpoint advertises a 1,048,576-token context, while the deployment's
//! configured `max_model_len` determines the padded DSA logits allocation.

use std::sync::Arc;

use anyhow::Result;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::glm52_dsa_moe::{Glm52ModelCfg, Glm52MtpMode};
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
    Glm52DenseFfnLocalWorkletResolved, Glm52MoeRouterLocalWorklet,
    Glm52MoeRouterLocalWorkletConfig, Glm52MoeRouterLocalWorkletInput,
    Glm52MoeRouterLocalWorkletResolved, Glm52MtpHeadLocalWorklet, Glm52MtpHeadLocalWorkletConfig,
    Glm52MtpHeadLocalWorkletInput, Glm52MtpHeadLocalWorkletResolved, Glm52MtpPreludeLocalWorklet,
    Glm52MtpPreludeLocalWorkletConfig, Glm52MtpPreludeLocalWorkletInput,
    Glm52MtpPreludeLocalWorkletResolved, Glm52SharedExpertLocalWorklet,
    Glm52SharedExpertLocalWorkletConfig, Glm52SharedExpertLocalWorkletInput,
    Glm52SharedExpertLocalWorkletResolved, Nvfp4MoeLocalWorklet, Nvfp4MoeLocalWorkletConfig,
    Nvfp4MoeLocalWorkletInput, Nvfp4MoeLocalWorkletResolved, VllmGlm52DsaAttnLocalDecodeInput,
    VllmGlm52DsaAttnLocalWorklet, VllmGlm52DsaAttnLocalWorkletConfig,
    VllmGlm52DsaAttnLocalWorkletInput, VllmGlm52DsaAttnLocalWorkletResolved,
};

const ARCH_KIND: &str = "glm52_vllm_nvfp4_dsa_moe";
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
const PROFILE_INDEX_HEADS: u32 = 64;
const INDEX_HEAD_DIM: u32 = 128;
const INDEX_TOP_K: u32 = 2_048;
const CHECKPOINT_MAX_CONTEXT: u32 = 1_048_576;
const NUM_EXPERTS: u32 = 256;
const ROUTER_TOP_K: u32 = 8;
const MOE_INTERMEDIATE_DIM: u32 = 2_048;
const NUM_SHARED_EXPERTS: u32 = 1;
const VOCAB_SIZE: u32 = 154_880;
const NUM_MTP_LAYERS: u32 = 1;
const CACHE_BLOCK_SIZE: u32 = 64;
const QUANT_BLOCK_SIZE: u32 = 128;
const SOFTMAX_SCALE_DENOMINATOR: u32 = 16;
const FULL_INDEX_LAYERS: [u32; 21] = [
    0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74,
];

const RESIDUAL_NORM_BACKENDS: &[&str] = &["vllm_cuda"];
const RMS_NORM_BACKENDS: &[&str] = &["flashinfer"];
const SINGLE_GEMM_BACKENDS: &[&str] = &["torch_linear"];
const VLLM_BF16_LINEAR_BACKENDS: &[&str] = &["torch_linear_vllm"];
const ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
// The MLA query RoPE is not a streaming pointwise leaf: vLLM feeds the roped
// q straight back into the attention op, so inductor fuses it into a kernel
// that rewrites all of q. Its own L1 kind measures that fusion.
const MAIN_ROPE_BACKENDS: &[&str] = &["vllm_inductor"];
const FP8_SINGLE_GEMM_BACKENDS: &[&str] = &["deepgemm"];
// vLLM quantises dense and routed activations with two different kernels.
// Dense linears call its own `per_token_group_quant_8bit_kernel` (backend
// `vllm_cuda`); only the routed grouped GEMM reaches TensorRT-LLM's
// `scale_1x128_kernel` (backend `flashinfer_trtllm`). Both appear in the same
// nsys iteration, so this is not a shape or a backend preference -- feeding the
// dense leaves the routed curve over-predicted them ~2.05x at prefill.
const DENSE_FP8_QUANT_BACKENDS: &[&str] = &["vllm_cuda"];
const Q_ABSORB_BACKENDS: &[&str] = &["torch_mla_q_absorb_glm52"];
const V_UP_BACKENDS: &[&str] = &["torch_mla_v_up_glm52"];
const INDEX_CACHE_AND_TOPK_BACKENDS: &[&str] = &["vllm_cuda"];
const INDEX_LOGITS_BACKENDS: &[&str] = &["deepgemm_fp8"];
const SPARSE_ATTN_BACKENDS: &[&str] = &["flashinfer_trtllm_fp8"];
const MLA_APPEND_BACKENDS: &[&str] = &["vllm_cuda"];
const NVFP4_QUANT_BACKENDS: &[&str] = &["vllm_cuda"];
const NVFP4_FUSED_MOE_BACKENDS: &[&str] = &["flashinfer_trtllm_sm100"];
// Generic collective fallbacks remain available above FlashInfer's workspace
// limit. Both FlashInfer attention-boundary fusion and standalone FFN
// all-reduce use shape-aware leaves because vLLM's selection and PDL behavior
// depend on token count, not only payload bytes.
const ALLREDUCE_BACKENDS: &[&str] = &["nccl", "nvshmem"];
const FUSED_ALLREDUCE_BACKENDS: &[&str] = &["flashinfer_trtllm"];

// Leaf counts are properties of the accepted L2/L3 sections. `build` verifies
// the compiled tree against these formulas, so drift cannot be hidden by a
// stale handwritten expectation.
#[cfg(test)]
const ATTN_FULL_SLOTS: usize = 29;
#[cfg(test)]
const ATTN_SHARED_SLOTS: usize = 14;
#[cfg(test)]
const DENSE_FFN_SLOTS: usize = 4;
#[cfg(test)]
const ROUTER_SLOTS: usize = 2;
#[cfg(test)]
/// Input quantization plus one overlap-aware whole fused-MoE leaf.
const EXPERT_SLOTS: usize = 2;
#[cfg(test)]
const SHARED_EXPERT_SLOTS: usize = 3;
#[cfg(test)]
const MTP_PRELUDE_SLOTS: usize = 6;
#[cfg(test)]
const MTP_HEAD_SLOTS: usize = 3;

#[derive(Clone, Debug)]
pub struct Glm52VllmNvfp4DsaMoeParallel {
    pub ep_size: u16,
    pub nvl_num_gpu: u16,
    pub max_model_len: u32,
    pub gpu_name: String,
}

#[derive(Clone, Debug)]
pub struct Glm52VllmNvfp4DsaMoeConfigs {
    pub model: Glm52ModelCfg,
    pub parallel: Glm52VllmNvfp4DsaMoeParallel,
    pub mtp_mode: Glm52MtpMode,
    pub dense_full_index_attention: VllmGlm52DsaAttnLocalWorkletConfig,
    pub dense_ffn: Glm52DenseFfnLocalWorkletConfig,
    pub initial_shared_attention: VllmGlm52DsaAttnLocalWorkletConfig,
    pub cycle_full_attention: VllmGlm52DsaAttnLocalWorkletConfig,
    pub cycle_shared_attention: VllmGlm52DsaAttnLocalWorkletConfig,
    pub sparse_router: Glm52MoeRouterLocalWorkletConfig,
    pub shared_expert: Glm52SharedExpertLocalWorkletConfig,
    /// One identity-free active-count-ranked workload per TP/EP rank. Each
    /// worklet owns the entire physical TRTLLM NVFP4 fused-MoE callable.
    pub nvfp4_moe: Vec<Nvfp4MoeLocalWorkletConfig>,
    pub tp_allreduce: AllReduceKernelConfig,
    pub tp_allreduce_fusion: AllReduceFusionKernelConfig,
    pub tp_allreduce_fused: AllReduceResidualRmsNormKernelConfig,
    pub embedding: ElementwiseKernelConfig,
    pub final_norm: ResidualRmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub mtp_prelude: Option<Glm52MtpPreludeLocalWorkletConfig>,
    pub mtp_attention: Option<VllmGlm52DsaAttnLocalWorkletConfig>,
    pub mtp_head: Option<Glm52MtpHeadLocalWorkletConfig>,
}

#[derive(Clone, Debug)]
pub struct Glm52VllmNvfp4DsaMoeResolved {
    pub raw_cfg: Glm52VllmNvfp4DsaMoeConfigs,
    pub dense_full_index_attention: VllmGlm52DsaAttnLocalWorkletResolved,
    pub dense_ffn: Glm52DenseFfnLocalWorkletResolved,
    pub initial_shared_attention: VllmGlm52DsaAttnLocalWorkletResolved,
    pub cycle_full_attention: VllmGlm52DsaAttnLocalWorkletResolved,
    pub cycle_shared_attention: VllmGlm52DsaAttnLocalWorkletResolved,
    pub sparse_router: Glm52MoeRouterLocalWorkletResolved,
    pub shared_expert: Glm52SharedExpertLocalWorkletResolved,
    pub nvfp4_moe: Vec<Nvfp4MoeLocalWorkletResolved>,
    pub tp_allreduce: AllReduceKernelConfig,
    pub tp_allreduce_fusion: AllReduceFusionKernelConfig,
    pub tp_allreduce_fused: AllReduceResidualRmsNormKernelConfig,
    pub embedding: ElementwiseKernelConfig,
    pub final_norm: ResidualRmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub mtp_prelude: Option<Glm52MtpPreludeLocalWorkletResolved>,
    pub mtp_attention: Option<VllmGlm52DsaAttnLocalWorkletResolved>,
    pub mtp_head: Option<Glm52MtpHeadLocalWorkletResolved>,
}

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

fn attention_config(
    model: &Glm52ModelCfg,
    parallel: &Glm52VllmNvfp4DsaMoeParallel,
    include_indexer: bool,
    fp8: bool,
) -> VllmGlm52DsaAttnLocalWorkletConfig {
    let (gemm_dtype, gemm_backends) = if fp8 {
        (DType::Fp8E4m3, FP8_SINGLE_GEMM_BACKENDS)
    } else {
        (DType::Bf16, SINGLE_GEMM_BACKENDS)
    };
    VllmGlm52DsaAttnLocalWorkletConfig {
        fp8_quant_backends: DENSE_FP8_QUANT_BACKENDS.to_vec(),
        include_indexer,
        tp_size: parallel.ep_size,
        residual_rms_norm_backends: RESIDUAL_NORM_BACKENDS.to_vec(),
        rms_norm_backends: RMS_NORM_BACKENDS.to_vec(),
        single_gemm_backends: gemm_backends.to_vec(),
        main_rope_backends: MAIN_ROPE_BACKENDS.to_vec(),
        q_absorb_backends: Q_ABSORB_BACKENDS.to_vec(),
        v_up_backends: V_UP_BACKENDS.to_vec(),
        indexer_gemm_backends: gemm_backends.to_vec(),
        indexer_elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        index_cache_append_backends: INDEX_CACHE_AND_TOPK_BACKENDS.to_vec(),
        index_prefill_logits_backends: INDEX_LOGITS_BACKENDS.to_vec(),
        index_prefill_topk_backends: INDEX_CACHE_AND_TOPK_BACKENDS.to_vec(),
        index_decode_logits_backends: INDEX_LOGITS_BACKENDS.to_vec(),
        index_decode_topk_backends: INDEX_CACHE_AND_TOPK_BACKENDS.to_vec(),
        sparse_attention_backends: SPARSE_ATTN_BACKENDS.to_vec(),
        sparse_mla_cache_append_backends: MLA_APPEND_BACKENDS.to_vec(),
        sparse_elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        sparse_index_remap_backends: vec!["vllm_triton"],
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
        gemm_dtype,
        index_cache_dtype: DType::Fp8E4m3,
        index_q_dtype: DType::Fp8E4m3,
        scale_dtype: DType::Fp32,
        weight_dtype: DType::Fp32,
        logits_dtype: DType::Fp32,
        index_dtype: "int32".to_string(),
        index_scale_format: "ue8m0".to_string(),
        index_cache_format: "page_planar_fp8_fp32_scale".to_string(),
        index_prefill_span_mode: "single_causal_tail".to_string(),
        index_decode_context_mode: "uniform".to_string(),
        index_decode_page_mapping: "unique_scattered".to_string(),
        rope_is_neox_style: false,
        index_clean_logits: false,
        // Production selected indices are top-k ordered and page-local; this is
        // the smallest supported deterministic profiling identity.
        sparse_index_distribution: "recent_contiguous".to_string(),
        sparse_cache_layout: "hnd_paged_mqa_fp8_latent_rope".to_string(),
        sparse_mla_cache_format: "plain".to_string(),
        sparse_attention_q_dtype: DType::Fp8E4m3,
        sparse_attention_cache_dtype: DType::Fp8E4m3,
        sparse_attention_output_dtype: DType::Bf16,
        sparse_exact_varlen: Some(DsaSparseMlaExactVarlenConfig {
            prefill_backends: vec!["flashinfer_trtllm_fp8"],
            max_model_len: parallel.max_model_len,
            prefill_index_distribution: "recent_contiguous".to_string(),
            page_table_mapping: Some("request_contiguous".to_string()),
        }),
        decode_next_n: 1,
    }
}

pub fn build_configs(
    model: &Glm52ModelCfg,
    parallel: &Glm52VllmNvfp4DsaMoeParallel,
    routing: &RoutingDistribution,
    fp8: bool,
    mtp_mode: Glm52MtpMode,
) -> Result<Glm52VllmNvfp4DsaMoeConfigs, BuildError> {
    validate_model_cfg(model).map_err(fit_failed)?;
    if parallel.ep_size == 0 {
        return Err(fit_failed("ep_size must be positive"));
    }
    if NUM_EXPERTS % u32::from(parallel.ep_size) != 0 {
        return Err(fit_failed(format!(
            "num_experts {NUM_EXPERTS} must be divisible by ep_size {}",
            parallel.ep_size
        )));
    }
    if parallel.nvl_num_gpu == 0
        || parallel.nvl_num_gpu > parallel.ep_size
        || parallel.ep_size % parallel.nvl_num_gpu != 0
    {
        return Err(fit_failed(format!(
            "nvl_num_gpu {} must be a positive divisor of ep_size {}",
            parallel.nvl_num_gpu, parallel.ep_size
        )));
    }
    if !(1..=CHECKPOINT_MAX_CONTEXT).contains(&parallel.max_model_len) {
        return Err(fit_failed(format!(
            "max_model_len {} must be in 1..={CHECKPOINT_MAX_CONTEXT}",
            parallel.max_model_len
        )));
    }
    if routing.num_experts() != NUM_EXPERTS {
        return Err(fit_failed(format!(
            "routing distribution has {} experts, expected {NUM_EXPERTS}",
            routing.num_experts()
        )));
    }
    if fp8 {
        return Err(fit_failed(
            "glm52_vllm_nvfp4_dsa_moe requires fp8=false; NVFP4 routed weights are encoded by the architecture",
        ));
    }
    let (gemm_dtype, gemm_backends) = if fp8 {
        (DType::Fp8E4m3, FP8_SINGLE_GEMM_BACKENDS)
    } else {
        (DType::Bf16, SINGLE_GEMM_BACKENDS)
    };
    let gpu = parallel.gpu_name.clone();
    let dense_full_index_attention = attention_config(model, parallel, true, fp8);
    let initial_shared_attention = attention_config(model, parallel, false, fp8);
    let cycle_full_attention = attention_config(model, parallel, true, fp8);
    let cycle_shared_attention = attention_config(model, parallel, false, fp8);
    let hidden_bytes = HIDDEN_DIM
        .checked_mul(DType::Bf16.size_bytes())
        .ok_or_else(|| fit_failed("hidden byte width overflows u32"))?;
    let embedding_input = hidden_bytes
        .checked_add(8)
        .ok_or_else(|| fit_failed("embedding input byte rate overflows u32"))?;
    let mtp_attention = match mtp_mode {
        Glm52MtpMode::Off => None,
        Glm52MtpMode::FullIndex => Some(attention_config(model, parallel, true, fp8)),
        Glm52MtpMode::IndexShare => Some(attention_config(model, parallel, false, fp8)),
    };
    let mtp_prelude = (mtp_mode != Glm52MtpMode::Off).then(|| Glm52MtpPreludeLocalWorkletConfig {
        elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        rms_norm_backends: RMS_NORM_BACKENDS.to_vec(),
        gemm_backends: gemm_backends.to_vec(),
        gpu_name: gpu.clone(),
        hidden_dim: model.hidden_dim.clone(),
        vocab_size: model.vocab_size.clone(),
        dtype: DType::Bf16,
        gemm_dtype,
        token_id_bytes: 8,
        position_bytes: 8,
    });
    let mtp_head = (mtp_mode != Glm52MtpMode::Off).then(|| Glm52MtpHeadLocalWorkletConfig {
        elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        rms_norm_backends: RMS_NORM_BACKENDS.to_vec(),
        gemm_backends: gemm_backends.to_vec(),
        gpu_name: gpu.clone(),
        hidden_dim: model.hidden_dim.clone(),
        vocab_size: model.vocab_size.clone(),
        dtype: DType::Bf16,
        gemm_dtype,
    });

    let nvfp4_moe = Nvfp4MoeLocalWorkletConfig::split_for_ep(
        Nvfp4MoeLocalWorkletConfig {
            hidden: model.hidden_dim.clone(),
            moe_intermediate: model.moe_intermediate_dim.clone(),
            num_experts: model.num_experts.clone(),
            ep_size: parallel.ep_size,
            // This arch shards experts and leaves each expert's intermediate
            // width whole. A pure-TP consumer sets its real TP degree instead.
            tp_size: 1,
            top_k: model.router_top_k,
            activation_dtype: DType::Bf16,
            gpu_name: gpu.clone(),
            quant_backends: NVFP4_QUANT_BACKENDS.to_vec(),
            moe_backends: NVFP4_FUSED_MOE_BACKENDS.to_vec(),
            weight_format: "nvfp4_e2m1".to_string(),
            group_size: 16,
            routing_method: "minimax2".to_string(),
            n_group: 1,
            topk_group: 1,
            routed_scaling_numerator: 5,
            routed_scaling_denominator: 2,
            layerwise_global_ppm: Vec::new(),
            folded_rank_position: 0,
        },
        routing,
        NUM_LAYERS - NUM_DENSE_LAYERS,
    );

    Ok(Glm52VllmNvfp4DsaMoeConfigs {
        model: model.clone(),
        parallel: parallel.clone(),
        mtp_mode,
        dense_full_index_attention,
        dense_ffn: Glm52DenseFfnLocalWorkletConfig {
            residual_norm_backends: RESIDUAL_NORM_BACKENDS.to_vec(),
            gemm_backends: gemm_backends.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
            tp_size: parallel.ep_size,
            gpu_name: gpu.clone(),
            hidden_dim: model.hidden_dim.clone(),
            intermediate_dim: model.dense_intermediate_dim.clone(),
            dtype: DType::Bf16,
            gemm_dtype,
        },
        initial_shared_attention,
        cycle_full_attention,
        cycle_shared_attention,
        sparse_router: Glm52MoeRouterLocalWorkletConfig {
            residual_norm_backends: RESIDUAL_NORM_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
            proxy_gemm_backends: VLLM_BF16_LINEAR_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden_dim: model.hidden_dim.clone(),
            num_experts: model.num_experts.clone(),
            top_k: model.router_top_k,
            n_group: 1,
            topk_group: 1,
            base_dtype: DType::Bf16,
            router_semantic_dtype: DType::Fp32,
            // vLLM leaves the replicated router unquantized and calls BF16
            // F.linear in its pinned PyTorch/CUDA environment.
            proxy_gemm_dtype: DType::Bf16,
            index_dtype: "int32".to_string(),
            scoring_func: "sigmoid".to_string(),
            topk_method: "noaux_tc".to_string(),
            norm_topk_prob: true,
            routed_scaling_numerator: 5,
            routed_scaling_denominator: 2,
            include_router_input_cast: false,
            include_router_select: false,
        },
        shared_expert: Glm52SharedExpertLocalWorkletConfig {
            // These unquantized projections are F.linear calls inside vLLM's
            // pinned Torch/CUDA environment, like the router above.
            gemm_backends: VLLM_BF16_LINEAR_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
            tp_size: parallel.ep_size,
            gpu_name: gpu.clone(),
            hidden_dim: model.hidden_dim.clone(),
            moe_intermediate_dim: model.moe_intermediate_dim.clone(),
            n_shared_experts: model.num_shared_experts,
            dtype: DType::Bf16,
            gemm_dtype,
        },
        nvfp4_moe,
        tp_allreduce: AllReduceKernelConfig {
            backends: ALLREDUCE_BACKENDS.to_vec(),
            gpu_name: parallel.gpu_name.clone(),
            num_gpus: u32::from(parallel.ep_size),
            fabric: Fabric::Nvlink,
        },
        tp_allreduce_fusion: AllReduceFusionKernelConfig {
            backends: FUSED_ALLREDUCE_BACKENDS.to_vec(),
            gpu_name: parallel.gpu_name.clone(),
            num_gpus: u32::from(parallel.ep_size),
            hidden_dim: model.hidden_dim.get(),
            dtype: DType::Bf16,
            fabric: Fabric::Nvlink,
        },
        tp_allreduce_fused: AllReduceResidualRmsNormKernelConfig {
            backends: FUSED_ALLREDUCE_BACKENDS.to_vec(),
            gpu_name: parallel.gpu_name.clone(),
            num_gpus: u32::from(parallel.ep_size),
            hidden_dim: model.hidden_dim.get(),
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
        // lm_head is NOT quantised. The GLM-5.2-FP8 checkpoint lists `lm_head`
        // and `model.embed_tokens` in `quantization_config.modules_to_not_convert`,
        // and the safetensors header confirms `lm_head.weight` is BF16
        // [154880, 6144]. It therefore has no input_quant leaf and runs the
        // plain BF16 GEMM (`nvjet_tst_*`, cuBLASLt) in the trace.
        //
        // Following `fp8` here priced 0.886 GiB of weights instead of 1.77 GiB:
        // 232.25 us against 463.46 us measured, a ratio of 0.5011. The BF16 DRAM
        // floor alone is ~396 us, so the FP8 value was not even reachable.
        lm_head: SingleGemmKernelConfig {
            backends: SINGLE_GEMM_BACKENDS.to_vec(),
            gpu_name: gpu,
            // ParallelLMHead shards the vocabulary dimension across TP ranks.
            // This arch uses the EP group as its TP group, so each rank owns
            // vocab_size / ep_size output rows rather than the global matrix.
            n: model.vocab_size.clone() / u32::from(parallel.ep_size),
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
        return Err("model dtypes must be BF16 base and FP32 router semantics".to_string());
    }
    if model.full_index_layers != FULL_INDEX_LAYERS {
        return Err("full-index layer schedule must match GLM-5.2".to_string());
    }
    let expected_indexers: Vec<String> = (0..NUM_LAYERS)
        .map(|layer| {
            if FULL_INDEX_LAYERS.contains(&layer) {
                "full"
            } else {
                "shared"
            }
            .to_string()
        })
        .collect();
    if model.indexer_types != expected_indexers {
        return Err("indexer_types must match GLM-5.2".to_string());
    }
    let expected_mlp: Vec<String> = (0..NUM_LAYERS)
        .map(|layer| {
            if layer < NUM_DENSE_LAYERS {
                "dense"
            } else {
                "sparse"
            }
            .to_string()
        })
        .collect();
    if model.mlp_layer_types != expected_mlp {
        return Err("mlp_layer_types must match GLM-5.2".to_string());
    }
    if !model.index_share_for_mtp_iteration {
        return Err("index_share_for_mtp_iteration must be enabled".to_string());
    }
    Ok(())
}

pub fn resolve_configs(cfgs: &Glm52VllmNvfp4DsaMoeConfigs) -> Glm52VllmNvfp4DsaMoeResolved {
    Glm52VllmNvfp4DsaMoeResolved {
        dense_full_index_attention: VllmGlm52DsaAttnLocalWorklet::resolve_config(
            &cfgs.dense_full_index_attention,
        ),
        dense_ffn: Glm52DenseFfnLocalWorklet::resolve_config(&cfgs.dense_ffn),
        initial_shared_attention: VllmGlm52DsaAttnLocalWorklet::resolve_config(
            &cfgs.initial_shared_attention,
        ),
        cycle_full_attention: VllmGlm52DsaAttnLocalWorklet::resolve_config(
            &cfgs.cycle_full_attention,
        ),
        cycle_shared_attention: VllmGlm52DsaAttnLocalWorklet::resolve_config(
            &cfgs.cycle_shared_attention,
        ),
        sparse_router: Glm52MoeRouterLocalWorklet::resolve_config(&cfgs.sparse_router),
        shared_expert: Glm52SharedExpertLocalWorklet::resolve_config(&cfgs.shared_expert),
        nvfp4_moe: cfgs
            .nvfp4_moe
            .iter()
            .map(Nvfp4MoeLocalWorklet::resolve_config)
            .collect(),
        tp_allreduce: cfgs.tp_allreduce.clone(),
        tp_allreduce_fusion: cfgs.tp_allreduce_fusion.clone(),
        tp_allreduce_fused: cfgs.tp_allreduce_fused.clone(),
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
            .map(VllmGlm52DsaAttnLocalWorklet::resolve_config),
        mtp_head: cfgs
            .mtp_head
            .as_ref()
            .map(Glm52MtpHeadLocalWorklet::resolve_config),
        raw_cfg: cfgs.clone(),
    }
}

struct Glm52SparseBody {
    name: String,
    attention: VllmGlm52DsaAttnLocalWorklet,
    router: Glm52MoeRouterLocalWorklet,
    /// One per EP child, ordered by identity-free active-expert workload.
    expert_compute: Vec<Nvfp4MoeLocalWorklet>,
    shared_expert: Glm52SharedExpertLocalWorklet,
    attention_allreduce: Op<AllReduceKernel>,
    attention_allreduce_fused: Op<AllReduceResidualRmsNormKernel>,
    attention_allreduce_max_fused_tokens: u32,
    ffn_allreduce_fallback: Op<AllReduceKernel>,
    ffn_allreduce_fusion: Op<AllReduceFusionKernel>,
    ffn_allreduce_max_fused_tokens: u32,
    ep_size: u16,
    top_k: u32,
}

impl Glm52SparseBody {
    fn build(
        name: String,
        attention: VllmGlm52DsaAttnLocalWorkletResolved,
        common: &Glm52VllmNvfp4DsaMoeResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let attention =
            VllmGlm52DsaAttnLocalWorklet::build(format!("{name}.attention"), attention, bridge)?;
        let router = Glm52MoeRouterLocalWorklet::build(
            format!("{name}.moe.router"),
            common.sparse_router.clone(),
            bridge,
        )?;
        // Every rank keeps the same slot name: the rank axis already shows up
        // as the `Max` node's children, and a rank suffix would rename the
        // slots a labeled kernel inventory refers to.
        let expert_compute = common
            .nvfp4_moe
            .iter()
            .map(|rank_resolved| {
                Nvfp4MoeLocalWorklet::build(
                    format!("{name}.moe.routed_experts"),
                    rank_resolved.clone(),
                    bridge,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shared_expert = Glm52SharedExpertLocalWorklet::build(
            format!("{name}.moe.shared_expert"),
            common.shared_expert.clone(),
            bridge,
        )?;
        let attention_allreduce = build_atomic(
            format!("{name}.attention.tp_allreduce"),
            common.tp_allreduce.clone(),
            AllReduceKernel::build,
            bridge,
        )?;
        let attention_allreduce_fused = build_atomic(
            format!("{name}.attention.tp_allreduce_residual_norm"),
            common.tp_allreduce_fused.clone(),
            AllReduceResidualRmsNormKernel::build,
            bridge,
        )?;
        let attention_allreduce_max_fused_tokens =
            AllReduceResidualRmsNormSpec::max_fused_tokens(&common.tp_allreduce_fused);
        let ffn_allreduce_fallback = build_atomic(
            format!("{name}.moe.tp_allreduce_fallback"),
            common.tp_allreduce.clone(),
            AllReduceKernel::build,
            bridge,
        )?;
        let ffn_allreduce_fusion = build_atomic(
            format!("{name}.moe.tp_allreduce"),
            common.tp_allreduce_fusion.clone(),
            AllReduceFusionKernel::build,
            bridge,
        )?;
        let ffn_allreduce_max_fused_tokens =
            AllReduceFusionSpec::max_fused_tokens(&common.tp_allreduce_fusion);
        Ok(Self {
            name,
            attention,
            router,
            expert_compute,
            shared_expert,
            attention_allreduce,
            attention_allreduce_fused,
            attention_allreduce_max_fused_tokens,
            ffn_allreduce_fallback,
            ffn_allreduce_fusion,
            ffn_allreduce_max_fused_tokens,
            ep_size: common.raw_cfg.parallel.ep_size,
            top_k: common.raw_cfg.model.router_top_k,
        })
    }

    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let attention = labeled_max(
            format!("{}.attention [Max over TP ranks]", self.name),
            (0..self.ep_size)
                .map(|_| self.attention.compile(builder))
                .collect(),
        );
        let attention_allreduce = self.attention_allreduce.compile(builder);
        let attention_allreduce_fused = self.attention_allreduce_fused.compile(builder);
        // Leaves are numbered in the order they are declared here, and `eval`
        // must push in the same order — so these statements follow the measured
        // launch order, which is also the order the `Sum` below lists them in.
        let router = labeled_max(
            format!("{}.moe.router [Max over replicated TP/EP ranks]", self.name),
            (0..self.ep_size)
                .map(|_| self.router.compile(builder))
                .collect(),
        );
        let experts_and_shared = labeled_max(
            format!("{}.moe.local_experts [Max over TP/EP ranks]", self.name),
            self.expert_compute
                .iter()
                .map(|rank_expert| {
                    CostNode::Sum(vec![
                        self.shared_expert.compile(builder),
                        rank_expert.compile(builder),
                    ])
                })
                .collect(),
        );
        let ffn_allreduce_fallback = self.ffn_allreduce_fallback.compile(builder);
        let ffn_allreduce_fusion = self.ffn_allreduce_fusion.compile(builder);
        CostNode::Labeled {
            label: format!(
                "{} [sparse layer; TP=EP={}; top-{} NVFP4 routed experts]",
                self.name, self.ep_size, self.top_k
            ),
            child: Box::new(CostNode::Sum(vec![
                attention,
                attention_allreduce,
                attention_allreduce_fused,
                CostNode::Labeled {
                    label: format!(
                        "{}.moe [router -> local shared+NVFP4 experts -> TP allreduce]",
                        self.name
                    ),
                    child: Box::new(CostNode::Sum(vec![
                        router,
                        experts_and_shared,
                        ffn_allreduce_fallback,
                        ffn_allreduce_fusion,
                    ])),
                },
            ])),
        }
    }

    fn eval(&self, batch: &NormalizedBatch, ev: &mut Evaluator) {
        let group = &batch.groups[0];
        let use_fused = batch.total_tokens > 0
            && batch.total_tokens <= self.attention_allreduce_max_fused_tokens;
        for _ in 0..self.ep_size {
            self.attention.eval(&group.attention_input, ev);
        }
        eval_atomic_or_zero(
            &self.attention_allreduce,
            AllReduceKernelInput {
                message_size_bytes: u64::from(batch.total_tokens) * u64::from(HIDDEN_DIM) * 2,
            },
            batch.total_tokens == 0 || use_fused,
            ev,
        );
        eval_atomic_or_zero(
            &self.attention_allreduce_fused,
            AllReduceResidualRmsNormKernelInput {
                num_tokens: batch.total_tokens,
            },
            !use_fused,
            ev,
        );
        for _ in 0..self.ep_size {
            self.router.eval_with_post_attn_norm(
                &Glm52MoeRouterLocalWorkletInput {
                    batch_tokens: group.batch_tokens,
                },
                !use_fused,
                ev,
            );
        }
        for rank_expert in &self.expert_compute {
            self.shared_expert.eval(
                &Glm52SharedExpertLocalWorkletInput {
                    batch_tokens: group.batch_tokens,
                },
                ev,
            );
            rank_expert.eval(
                &Nvfp4MoeLocalWorkletInput {
                    num_tokens: group.batch_tokens,
                },
                ev,
            );
        }
        let use_ffn_fusion =
            batch.total_tokens > 0 && batch.total_tokens <= self.ffn_allreduce_max_fused_tokens;
        eval_atomic_or_zero(
            &self.ffn_allreduce_fallback,
            AllReduceKernelInput {
                message_size_bytes: u64::from(batch.total_tokens) * u64::from(HIDDEN_DIM) * 2,
            },
            batch.total_tokens == 0 || use_ffn_fusion,
            ev,
        );
        eval_atomic_or_zero(
            &self.ffn_allreduce_fusion,
            AllReduceFusionKernelInput {
                num_tokens: batch.total_tokens,
            },
            !use_ffn_fusion,
            ev,
        );
    }
}

struct Glm52MtpSection {
    prelude: Glm52MtpPreludeLocalWorklet,
    decoder: Glm52SparseBody,
    head: Glm52MtpHeadLocalWorklet,
}

pub struct Glm52VllmNvfp4DsaMoeModel {
    pub name: String,
    pub mtp_mode: Glm52MtpMode,
    pub ep_size: u16,
    pub nvl_num_gpu: u16,
    pub max_model_len: u32,
    pub num_attn_dp_groups: u16,
    pub num_attn_shards: u16,
    pub total_state_bytes_per_token: u64,
    pub embedding: Op<ElementwiseKernel>,
    pub embedding_allreduce_fallback: Op<AllReduceKernel>,
    pub embedding_allreduce_fusion: Op<AllReduceFusionKernel>,
    pub embedding_allreduce_max_fused_tokens: u32,
    pub dense_full_index_attention: VllmGlm52DsaAttnLocalWorklet,
    pub dense_ffn: Glm52DenseFfnLocalWorklet,
    pub dense_attention_allreduce: Op<AllReduceKernel>,
    pub dense_attention_allreduce_fused: Op<AllReduceResidualRmsNormKernel>,
    pub dense_attention_allreduce_max_fused_tokens: u32,
    pub dense_ffn_allreduce_fallback: Op<AllReduceKernel>,
    pub dense_ffn_allreduce_fusion: Op<AllReduceFusionKernel>,
    pub dense_ffn_allreduce_max_fused_tokens: u32,
    initial_shared_sparse: Glm52SparseBody,
    cycle_full_sparse: Glm52SparseBody,
    cycle_shared_sparse: Glm52SparseBody,
    pub final_norm: Op<ResidualRmsNormKernel>,
    pub lm_head: Op<SingleGemmKernel>,
    mtp: Option<Glm52MtpSection>,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: Glm52VllmNvfp4DsaMoeResolved,
    bridge: &PerfApiBridge,
) -> Result<Glm52VllmNvfp4DsaMoeModel, BuildError> {
    let ep_size = resolved.raw_cfg.parallel.ep_size;
    let nvl_num_gpu = resolved.raw_cfg.parallel.nvl_num_gpu;
    let max_model_len = resolved.raw_cfg.parallel.max_model_len;
    let mtp_mode = resolved.raw_cfg.mtp_mode;
    let embedding = build_atomic(
        format!("{name}.main.embedding"),
        resolved.embedding.clone(),
        ElementwiseKernel::build,
        bridge,
    )?;
    let embedding_allreduce_fallback = build_atomic(
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
    let embedding_allreduce_max_fused_tokens =
        AllReduceFusionSpec::max_fused_tokens(&resolved.tp_allreduce_fusion);
    let dense_full_index_attention = VllmGlm52DsaAttnLocalWorklet::build(
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
    let dense_attention_allreduce_fused = build_atomic(
        format!("{name}.body.dense_full_index.attention.tp_allreduce_residual_norm"),
        resolved.tp_allreduce_fused.clone(),
        AllReduceResidualRmsNormKernel::build,
        bridge,
    )?;
    let dense_attention_allreduce_max_fused_tokens =
        AllReduceResidualRmsNormSpec::max_fused_tokens(&resolved.tp_allreduce_fused);
    let dense_ffn_allreduce_fallback = build_atomic(
        format!("{name}.body.dense_full_index.ffn.tp_allreduce_fallback"),
        resolved.tp_allreduce.clone(),
        AllReduceKernel::build,
        bridge,
    )?;
    let dense_ffn_allreduce_fusion = build_atomic(
        format!("{name}.body.dense_full_index.ffn.tp_allreduce"),
        resolved.tp_allreduce_fusion.clone(),
        AllReduceFusionKernel::build,
        bridge,
    )?;
    let dense_ffn_allreduce_max_fused_tokens =
        AllReduceFusionSpec::max_fused_tokens(&resolved.tp_allreduce_fusion);
    let initial_shared_sparse = Glm52SparseBody::build(
        format!("{name}.body.sparse_initial_index_share"),
        resolved.initial_shared_attention.clone(),
        &resolved,
        bridge,
    )?;
    let cycle_full_sparse = Glm52SparseBody::build(
        format!("{name}.body.sparse_cycle_full_index"),
        resolved.cycle_full_attention.clone(),
        &resolved,
        bridge,
    )?;
    let cycle_shared_sparse = Glm52SparseBody::build(
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
        (Some(prelude), Some(attention), Some(head)) => Some(Glm52MtpSection {
            prelude: Glm52MtpPreludeLocalWorklet::build(
                format!("{name}.mtp.prelude"),
                prelude,
                bridge,
            )?,
            decoder: Glm52SparseBody::build(
                format!("{name}.mtp.decoder_sparse"),
                attention,
                &resolved,
                bridge,
            )?,
            head: Glm52MtpHeadLocalWorklet::build(format!("{name}.mtp.head"), head, bridge)?,
        }),
        _ => {
            return Err(fit_failed(
                "MTP prelude/attention/head must be all present or all absent",
            ))
        }
    };

    let total_state_bytes_per_token = state_bytes_per_token(ep_size, mtp_mode)?;
    let mut model = Glm52VllmNvfp4DsaMoeModel {
        name,
        mtp_mode,
        ep_size,
        nvl_num_gpu,
        max_model_len,
        num_attn_dp_groups: 1,
        num_attn_shards: ep_size,
        total_state_bytes_per_token,
        embedding,
        embedding_allreduce_fallback,
        embedding_allreduce_fusion,
        embedding_allreduce_max_fused_tokens,
        dense_full_index_attention,
        dense_ffn,
        dense_attention_allreduce,
        dense_attention_allreduce_fused,
        dense_attention_allreduce_max_fused_tokens,
        dense_ffn_allreduce_fallback,
        dense_ffn_allreduce_fusion,
        dense_ffn_allreduce_max_fused_tokens,
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

impl Glm52VllmNvfp4DsaMoeModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let embedding = labeled_max(
            format!("{}.main.embedding [Max over TP ranks]", self.name),
            (0..self.ep_size)
                .map(|_| self.embedding.compile(&mut builder))
                .collect(),
        );
        let embedding_allreduce_fallback = self.embedding_allreduce_fallback.compile(&mut builder);
        let embedding_allreduce_fusion = self.embedding_allreduce_fusion.compile(&mut builder);
        let dense_attention = labeled_max(
            format!(
                "{}.body.dense_full_index.attention [Max over TP ranks]",
                self.name
            ),
            (0..self.ep_size)
                .map(|_| self.dense_full_index_attention.compile(&mut builder))
                .collect(),
        );
        let dense_attention_allreduce = self.dense_attention_allreduce.compile(&mut builder);
        let dense_attention_allreduce_fused =
            self.dense_attention_allreduce_fused.compile(&mut builder);
        let dense_ffn = labeled_max(
            format!(
                "{}.body.dense_full_index.ffn [Max over TP ranks]",
                self.name
            ),
            (0..self.ep_size)
                .map(|_| self.dense_ffn.compile(&mut builder))
                .collect(),
        );
        let dense_ffn_allreduce_fallback = self.dense_ffn_allreduce_fallback.compile(&mut builder);
        let dense_ffn_allreduce_fusion = self.dense_ffn_allreduce_fusion.compile(&mut builder);
        let dense = CostNode::Labeled {
            label: "layers 0..2: dense + full index (3 layers)".to_string(),
            child: Box::new(CostNode::Scale {
                n: NUM_DENSE_LAYERS,
                child: Box::new(CostNode::Sum(vec![
                    dense_attention,
                    dense_attention_allreduce,
                    dense_attention_allreduce_fused,
                    dense_ffn,
                    dense_ffn_allreduce_fallback,
                    dense_ffn_allreduce_fusion,
                ])),
            }),
        };
        let initial_shared = CostNode::Labeled {
            label: "layers 3..5: sparse + IndexShare (3 layers)".to_string(),
            child: Box::new(CostNode::Scale {
                n: NUM_INITIAL_SHARED_LAYERS,
                child: Box::new(self.initial_shared_sparse.compile(&mut builder)),
            }),
        };
        let cycle = CostNode::Labeled {
            label: "layers 6..77: 18 cycles of full-index + 3 IndexShare".to_string(),
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
        let final_norm = labeled_max(
            format!(
                "{}.main.final_residual_rms_norm [Max over TP ranks]",
                self.name
            ),
            (0..self.ep_size)
                .map(|_| self.final_norm.compile(&mut builder))
                .collect(),
        );
        let lm_head = labeled_max(
            format!("{}.main.lm_head [Max over TP ranks]", self.name),
            (0..self.ep_size)
                .map(|_| self.lm_head.compile(&mut builder))
                .collect(),
        );
        let mut children = vec![
            embedding,
            embedding_allreduce_fallback,
            embedding_allreduce_fusion,
            dense,
            initial_shared,
            cycle,
            final_norm,
            lm_head,
        ];
        if let Some(mtp) = &self.mtp {
            let prelude = labeled_max(
                format!("{}.mtp.prelude [Max over TP ranks]", self.name),
                (0..self.ep_size)
                    .map(|_| mtp.prelude.compile(&mut builder))
                    .collect(),
            );
            let decoder = mtp.decoder.compile(&mut builder);
            let head = labeled_max(
                format!("{}.mtp.head [Max over TP ranks]", self.name),
                (0..self.ep_size)
                    .map(|_| mtp.head.compile(&mut builder))
                    .collect(),
            );
            children.push(CostNode::Labeled {
                label: format!("MTP layer 78 [{:?}; decode-only]", self.mtp_mode),
                child: Box::new(CostNode::Sum(vec![prelude, decoder, head])),
            });
        }
        let root = CostNode::Labeled {
            label: format!(
                "{} (Glm52VllmNvfp4DsaMoeModel) [TP=EP{}; attention DP groups={}; MTP={:?}; \
                 timing_context<={}]",
                self.name, self.ep_size, self.num_attn_dp_groups, self.mtp_mode, self.max_model_len
            ),
            child: Box::new(CostNode::Sum(children)),
        };
        builder.finish(root)
    }

    fn eval_into(&self, input: &UnifiedArchInput, ev: &mut Evaluator) {
        let batch = normalize_input(input, self.ep_size, self.max_model_len)
            .unwrap_or_else(|reason| panic!("invalid Glm52VllmNvfp4DsaMoeModel input: {reason}"));

        let group = &batch.groups[0];
        for _ in 0..self.ep_size {
            eval_atomic_or_zero(
                &self.embedding,
                ElementwiseKernelInput {
                    num_tokens: group.batch_tokens,
                },
                group.batch_tokens == 0,
                ev,
            );
        }
        let use_embedding_fusion = batch.total_tokens > 0
            && batch.total_tokens <= self.embedding_allreduce_max_fused_tokens;
        eval_atomic_or_zero(
            &self.embedding_allreduce_fallback,
            AllReduceKernelInput {
                message_size_bytes: u64::from(batch.total_tokens) * u64::from(HIDDEN_DIM) * 2,
            },
            batch.total_tokens == 0 || use_embedding_fusion,
            ev,
        );
        eval_atomic_or_zero(
            &self.embedding_allreduce_fusion,
            AllReduceFusionKernelInput {
                num_tokens: batch.total_tokens,
            },
            !use_embedding_fusion,
            ev,
        );
        let group = &batch.groups[0];
        for _ in 0..self.ep_size {
            self.dense_full_index_attention
                .eval(&group.attention_input, ev);
        }
        let use_dense_fused = batch.total_tokens > 0
            && batch.total_tokens <= self.dense_attention_allreduce_max_fused_tokens;
        eval_atomic_or_zero(
            &self.dense_attention_allreduce,
            AllReduceKernelInput {
                message_size_bytes: u64::from(batch.total_tokens) * u64::from(HIDDEN_DIM) * 2,
            },
            batch.total_tokens == 0 || use_dense_fused,
            ev,
        );
        eval_atomic_or_zero(
            &self.dense_attention_allreduce_fused,
            AllReduceResidualRmsNormKernelInput {
                num_tokens: batch.total_tokens,
            },
            !use_dense_fused,
            ev,
        );
        for _ in 0..self.ep_size {
            self.dense_ffn.eval_with_post_attn_norm(
                &Glm52DenseFfnLocalWorkletInput {
                    batch_tokens: group.batch_tokens,
                },
                !use_dense_fused,
                ev,
            );
        }
        let use_dense_ffn_fusion = batch.total_tokens > 0
            && batch.total_tokens <= self.dense_ffn_allreduce_max_fused_tokens;
        eval_atomic_or_zero(
            &self.dense_ffn_allreduce_fallback,
            AllReduceKernelInput {
                message_size_bytes: u64::from(batch.total_tokens) * u64::from(HIDDEN_DIM) * 2,
            },
            batch.total_tokens == 0 || use_dense_ffn_fusion,
            ev,
        );
        eval_atomic_or_zero(
            &self.dense_ffn_allreduce_fusion,
            AllReduceFusionKernelInput {
                num_tokens: batch.total_tokens,
            },
            !use_dense_ffn_fusion,
            ev,
        );
        self.initial_shared_sparse.eval(&batch, ev);
        self.cycle_full_sparse.eval(&batch, ev);
        self.cycle_shared_sparse.eval(&batch, ev);
        for _ in 0..self.ep_size {
            eval_atomic_or_zero(
                &self.final_norm,
                ResidualRmsNormKernelInput {
                    m: group.batch_tokens,
                },
                group.batch_tokens == 0,
                ev,
            );
        }
        for _ in 0..self.ep_size {
            eval_atomic_or_zero(
                &self.lm_head,
                SingleGemmKernelInput {
                    m: group.request_count,
                },
                group.request_count == 0,
                ev,
            );
        }

        if let Some(mtp) = &self.mtp {
            for _ in 0..self.ep_size {
                mtp.prelude.eval(
                    &Glm52MtpPreludeLocalWorkletInput {
                        batch_tokens: group.decode_tokens,
                    },
                    ev,
                );
            }
            let mtp_batch = batch.decode_only();
            mtp.decoder.eval(&mtp_batch, ev);
            for _ in 0..self.ep_size {
                mtp.head.eval(
                    &Glm52MtpHeadLocalWorkletInput {
                        batch_tokens: group.decode_tokens,
                    },
                    ev,
                );
            }
        }
    }
}

impl IterwiseUnifiedModel for Glm52VllmNvfp4DsaMoeModel {
    fn total_kv_bytes_per_token(&self) -> u64 {
        self.total_state_bytes_per_token
    }

    fn gpus_per_replica(&self) -> u16 {
        self.ep_size
    }

    fn num_attn_dp_groups(&self) -> u16 {
        self.num_attn_dp_groups
    }

    fn num_attn_shards(&self) -> u16 {
        self.num_attn_shards
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
        assert_eq!(
            evaluator.filled(),
            self.n_slots,
            "eval must fill every compiled slot"
        );
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
        assert_eq!(
            evaluator.filled(),
            self.n_slots,
            "eval must fill every compiled slot"
        );
        let total = CostTree::aggregate(&self.cost_flat, slots, scratch);
        assert_eq!(
            inputs.len(),
            self.n_slots,
            "slot inputs must align with compiled slots"
        );
        total
    }
}

#[derive(Clone, Debug)]
struct NormalizedGroup {
    batch_tokens: u32,
    decode_tokens: u32,
    request_count: u32,
    decode_context: Option<u32>,
    attention_input: VllmGlm52DsaAttnLocalWorkletInput,
}

#[derive(Clone, Debug)]
struct NormalizedBatch {
    groups: Vec<NormalizedGroup>,
    total_tokens: u32,
}

impl NormalizedBatch {
    fn decode_only(&self) -> Self {
        let groups: Vec<NormalizedGroup> = self
            .groups
            .iter()
            .map(|group| NormalizedGroup {
                batch_tokens: group.decode_tokens,
                decode_tokens: group.decode_tokens,
                request_count: group.decode_tokens,
                decode_context: group.decode_context,
                attention_input: VllmGlm52DsaAttnLocalWorkletInput {
                    num_new_tokens: group.decode_tokens,
                    prefill_query_cache_pairs: Vec::new(),
                    decode: group.decode_context.map(|context_len| {
                        VllmGlm52DsaAttnLocalDecodeInput {
                            batch_size: group.decode_tokens,
                            context_len,
                            context_lens: group
                                .attention_input
                                .decode
                                .as_ref()
                                .and_then(|decode| decode.context_lens.clone()),
                            requires_padding: false,
                        }
                    }),
                },
            })
            .collect();
        let total_tokens: u32 = groups.iter().map(|group| group.batch_tokens).sum();
        Self {
            groups,
            total_tokens,
        }
    }
}

fn normalize_input(
    input: &UnifiedArchInput,
    _ep_size: u16,
    max_model_len: u32,
) -> std::result::Result<NormalizedBatch, String> {
    if input.groups.len() != 1 {
        return Err(format!(
            "expected exactly one tensor-parallel attention group, got {}",
            input.groups.len()
        ));
    }
    let mut groups = Vec::with_capacity(input.groups.len());
    let mut total_tokens = 0_u32;
    for (group_index, group) in input.groups.iter().enumerate() {
        let mut prefill_tokens = 0_u32;
        let mut prefill_query_cache_pairs = Vec::with_capacity(group.prefill_chunk_pairs.len());
        for (request_index, &(prefix, append)) in group.prefill_chunk_pairs.iter().enumerate() {
            if append == 0 {
                return Err(format!(
                    "group {group_index} prefill request {request_index} append must be nonzero"
                ));
            }
            let cache_tokens = prefix.checked_add(append).ok_or_else(|| {
                format!("group {group_index} prefill request {request_index} prefix+append overflows u32")
            })?;
            if cache_tokens > max_model_len {
                return Err(format!(
                    "group {group_index} prefill request {request_index} context {cache_tokens} exceeds timing cap {max_model_len}"
                ));
            }
            prefill_tokens = prefill_tokens
                .checked_add(append)
                .ok_or_else(|| format!("group {group_index} prefill token sum overflows u32"))?;
            prefill_query_cache_pairs.push((append, cache_tokens));
        }
        if group.prefill_tokens != prefill_tokens {
            return Err(format!(
                "group {group_index} prefill_tokens {} must equal append sum {prefill_tokens}",
                group.prefill_tokens
            ));
        }
        let decode_tokens = u32::try_from(group.decode_kv_lens.len())
            .map_err(|_| format!("group {group_index} decode request count exceeds u32"))?;
        if group.decode_tokens != decode_tokens {
            return Err(format!(
                "group {group_index} decode_tokens {} must equal decode_kv_lens length {decode_tokens}",
                group.decode_tokens
            ));
        }
        let mut decode_context = None;
        for (request_index, &context) in group.decode_kv_lens.iter().enumerate() {
            if !(1..=max_model_len).contains(&context) {
                return Err(format!(
                    "group {group_index} decode request {request_index} context {context} must be in 1..={max_model_len}"
                ));
            }
            decode_context =
                Some(decode_context.map_or(context, |current: u32| current.max(context)));
        }
        let batch_tokens = prefill_tokens
            .checked_add(decode_tokens)
            .ok_or_else(|| format!("group {group_index} batch token sum overflows u32"))?;
        if group.batch_tokens != batch_tokens {
            return Err(format!(
                "group {group_index} batch_tokens {} must equal prefill+decode {batch_tokens}",
                group.batch_tokens
            ));
        }
        let request_count =
            u32::try_from(group.prefill_chunk_pairs.len() + group.decode_kv_lens.len())
                .map_err(|_| format!("group {group_index} request count exceeds u32"))?;
        total_tokens = total_tokens
            .checked_add(batch_tokens)
            .ok_or_else(|| "pooled token count overflows u32".to_string())?;
        groups.push(NormalizedGroup {
            batch_tokens,
            decode_tokens,
            request_count,
            decode_context,
            attention_input: VllmGlm52DsaAttnLocalWorkletInput {
                num_new_tokens: batch_tokens,
                prefill_query_cache_pairs,
                decode: decode_context.map(|context_len| VllmGlm52DsaAttnLocalDecodeInput {
                    batch_size: decode_tokens,
                    context_len,
                    context_lens: Some(group.decode_kv_lens.clone()),
                    requires_padding: false,
                }),
            },
        });
    }
    Ok(NormalizedBatch {
        groups,
        total_tokens,
    })
}

fn labeled_max(label: String, children: Vec<CostNode>) -> CostNode {
    CostNode::Labeled {
        label,
        child: Box::new(CostNode::Max {
            overlap: 1.0,
            children,
        }),
    }
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
fn expected_slot_count(ep_size: u16, mtp_mode: Glm52MtpMode) -> usize {
    let ep = usize::from(ep_size);
    // Scale nodes do not mint additional slots. Each static section is built
    // once, while rank-local work is represented by Max over TP/EP children.
    let dense = ep * (ATTN_FULL_SLOTS + DENSE_FFN_SLOTS) + 4;
    let sparse_shared =
        ep * (ATTN_SHARED_SLOTS + ROUTER_SLOTS + SHARED_EXPERT_SLOTS + EXPERT_SLOTS) + 4;
    let sparse_full = sparse_shared + ep * (ATTN_FULL_SLOTS - ATTN_SHARED_SLOTS);
    let main = (ep + 2) + dense + sparse_shared + sparse_full + sparse_shared + ep + ep;
    match mtp_mode {
        Glm52MtpMode::Off => main,
        Glm52MtpMode::FullIndex => {
            main + ep * MTP_PRELUDE_SLOTS + sparse_full + ep * MTP_HEAD_SLOTS
        }
        Glm52MtpMode::IndexShare => {
            main + ep * MTP_PRELUDE_SLOTS + sparse_shared + ep * MTP_HEAD_SLOTS
        }
    }
}

fn state_bytes_per_token(
    ep_size: u16,
    mtp_mode: Glm52MtpMode,
) -> std::result::Result<u64, BuildError> {
    // vLLM keeps both caches replicated on every TP rank. MLA stores one FP8
    // (kv_lora_rank + rope_dim) vector. Every decoder layer also allocates its
    // FP8 index key plus one FP32 scale, even when that layer reuses a previous
    // layer's top-k result and skips the indexer compute.
    let main_mla_per_rank = u64::from(NUM_LAYERS)
        .checked_mul(u64::from(KV_LORA_RANK + ROPE_DIM))
        .and_then(|value| value.checked_mul(u64::from(DType::Fp8E4m3.size_bytes())))
        .ok_or_else(|| fit_failed("main MLA state bytes overflow u64"))?;
    let index_per_rank = u64::from(NUM_LAYERS)
        .checked_mul(u64::from(INDEX_HEAD_DIM + DType::Fp32.size_bytes()))
        .ok_or_else(|| fit_failed("index state bytes overflow u64"))?;
    let main_per_rank = main_mla_per_rank
        .checked_add(index_per_rank)
        .ok_or_else(|| fit_failed("main state bytes overflow u64"))?;
    let mtp_mla_per_rank = u64::from(KV_LORA_RANK + ROPE_DIM)
        .checked_mul(u64::from(DType::Fp8E4m3.size_bytes()))
        .ok_or_else(|| fit_failed("MTP MLA state bytes overflow u64"))?;
    let per_rank = match mtp_mode {
        Glm52MtpMode::Off => Some(main_per_rank),
        Glm52MtpMode::FullIndex => main_per_rank
            .checked_add(mtp_mla_per_rank)
            .and_then(|value| {
                value.checked_add(u64::from(INDEX_HEAD_DIM + DType::Fp32.size_bytes()))
            }),
        Glm52MtpMode::IndexShare => main_per_rank.checked_add(mtp_mla_per_rank),
    }
    .ok_or_else(|| fit_failed("per-rank state bytes overflow u64"))?;

    // IterwiseUnifiedModel's KV contract is the physical total across all
    // attention ranks. Multiplying the replicated per-rank allocation by TP
    // keeps both worker capacity and PD wire-size accounting consistent.
    per_rank
        .checked_mul(u64::from(ep_size))
        .ok_or_else(|| fit_failed("replicated state bytes overflow u64"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::ArchGroupInput;
    use std::collections::BTreeSet;
    use std::path::Path;

    fn exact_json_value() -> serde_json::Value {
        let full: Vec<&str> = (0..NUM_LAYERS)
            .map(|layer| {
                if FULL_INDEX_LAYERS.contains(&layer) {
                    "full"
                } else {
                    "shared"
                }
            })
            .collect();
        let mlp: Vec<&str> = (0..NUM_LAYERS)
            .map(|layer| if layer < 3 { "dense" } else { "sparse" })
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
        value["mlp_layer_types"] = serde_json::to_value(mlp).unwrap();
        value["indexer_types"] = serde_json::to_value(full).unwrap();
        value
    }

    fn model() -> Glm52ModelCfg {
        crate::arch::glm52_dsa_moe::parse_model_json(&exact_json_value().to_string()).unwrap()
    }

    fn parallel(ep_size: u16) -> Glm52VllmNvfp4DsaMoeParallel {
        Glm52VllmNvfp4DsaMoeParallel {
            ep_size,
            nvl_num_gpu: ep_size.min(8),
            max_model_len: CHECKPOINT_MAX_CONTEXT,
            gpu_name: "NVIDIA B200".to_string(),
        }
    }

    #[test]
    fn build_configs_bakes_tp4_nvfp4_and_exact_varlen_without_a_bridge() {
        let mut p = parallel(4);
        p.max_model_len = 8_192;
        let cfg = build_configs(
            &model(),
            &p,
            &RoutingDistribution::uniform(NUM_EXPERTS),
            false,
            Glm52MtpMode::Off,
        )
        .unwrap();

        for attention in [
            &cfg.dense_full_index_attention,
            &cfg.initial_shared_attention,
            &cfg.cycle_full_attention,
            &cfg.cycle_shared_attention,
        ] {
            assert_eq!(attention.tp_size, 4);
            assert_eq!(attention.gpu_name, "NVIDIA B200");
            assert_eq!(
                attention.sparse_attention_backends,
                vec!["flashinfer_trtllm_fp8"]
            );
            assert_eq!(attention.sparse_attention_q_dtype, DType::Fp8E4m3);
            assert_eq!(attention.sparse_attention_cache_dtype, DType::Fp8E4m3);
            assert_eq!(attention.sparse_attention_output_dtype, DType::Bf16);
            let exact = attention.sparse_exact_varlen.as_ref().unwrap();
            assert_eq!(attention.sparse_index_remap_backends, vec!["vllm_triton"]);
            assert_eq!(exact.prefill_backends, vec!["flashinfer_trtllm_fp8"]);
            assert_eq!(exact.max_model_len, 8_192);
            assert_eq!(
                exact.page_table_mapping.as_deref(),
                Some("request_contiguous")
            );
        }
        assert!(cfg.dense_full_index_attention.include_indexer);
        assert!(!cfg.initial_shared_attention.include_indexer);
        assert_eq!(cfg.dense_ffn.tp_size, 4);
        assert_eq!(cfg.shared_expert.tp_size, 4);
        assert!(!cfg.sparse_router.include_router_input_cast);
        assert!(!cfg.sparse_router.include_router_select);
        assert_eq!(cfg.nvfp4_moe.len(), 4);
        assert_eq!(
            cfg.nvfp4_moe
                .iter()
                .map(|rank| rank.folded_rank_position)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert!(cfg
            .nvfp4_moe
            .windows(2)
            .all(|pair| pair[0].layerwise_global_ppm == pair[1].layerwise_global_ppm));
        for rank in &cfg.nvfp4_moe {
            assert_eq!(rank.quant_backends, vec!["vllm_cuda"]);
            assert_eq!(rank.moe_backends, vec!["flashinfer_trtllm_sm100"]);
            assert_eq!(rank.weight_format, "nvfp4_e2m1");
            assert_eq!(rank.group_size, 16);
            assert_eq!(rank.routing_method, "minimax2");
        }
        assert_eq!(cfg.tp_allreduce.num_gpus, 4);
        assert_eq!(cfg.tp_allreduce_fusion.num_gpus, 4);
        assert_eq!(cfg.tp_allreduce_fused.num_gpus, 4);
    }

    #[test]
    fn resolve_configs_partitions_only_the_tp_owned_dimensions() {
        let cfg = build_configs(
            &model(),
            &parallel(4),
            &RoutingDistribution::uniform(NUM_EXPERTS),
            false,
            Glm52MtpMode::Off,
        )
        .unwrap();
        let resolved = resolve_configs(&cfg);

        assert_eq!(resolved.dense_full_index_attention.main_rope.num_heads, 16);
        assert_eq!(resolved.dense_full_index_attention.q_absorb.num_batches, 16);
        let indexer = resolved
            .dense_full_index_attention
            .indexer
            .as_ref()
            .unwrap();
        assert_eq!(indexer.q_lora_rank, 2_048);
        assert_eq!(indexer.model_num_index_heads, 32);
        assert_eq!(resolved.dense_ffn.gate_up_proj.n, 6_144);
        assert_eq!(resolved.dense_ffn.gate_up_proj.k, 6_144);
        assert_eq!(resolved.shared_expert.gate_up_proj.n, 1_024);
        assert_eq!(resolved.shared_expert.gate_up_proj.k, 6_144);
        assert_eq!(resolved.nvfp4_moe.len(), 4);
        assert!(resolved
            .nvfp4_moe
            .iter()
            .all(|rank| rank.experts_per_device == 64));
        assert!(resolved.sparse_router.router_fp32_cast.is_none());
        assert!(resolved.sparse_router.router_select.is_none());
    }

    #[test]
    fn parallel_model_and_checkpoint_contracts_fail_closed() {
        let routing = RoutingDistribution::uniform(NUM_EXPERTS);
        let m = model();

        let mut invalid = parallel(4);
        invalid.ep_size = 0;
        assert!(build_configs(&m, &invalid, &routing, false, Glm52MtpMode::Off).is_err());

        let mut invalid = parallel(4);
        invalid.nvl_num_gpu = 3;
        assert!(build_configs(&m, &invalid, &routing, false, Glm52MtpMode::Off).is_err());

        let mut invalid = parallel(4);
        invalid.max_model_len = 0;
        assert!(build_configs(&m, &invalid, &routing, false, Glm52MtpMode::Off).is_err());

        assert!(
            build_configs(&m, &parallel(4), &routing, true, Glm52MtpMode::Off)
                .unwrap_err()
                .to_string()
                .contains("fp8=false")
        );
        assert!(build_configs(
            &m,
            &parallel(4),
            &RoutingDistribution::uniform(128),
            false,
            Glm52MtpMode::Off,
        )
        .is_err());
    }

    #[test]
    fn compiled_tree_matches_the_tp4_schedule_and_collective_boundaries() {
        for (mode, expected) in [
            (Glm52MtpMode::Off, expected_slot_count(4, Glm52MtpMode::Off)),
            (
                Glm52MtpMode::FullIndex,
                expected_slot_count(4, Glm52MtpMode::FullIndex),
            ),
            (
                Glm52MtpMode::IndexShare,
                expected_slot_count(4, Glm52MtpMode::IndexShare),
            ),
        ] {
            let cfg = build_configs(
                &model(),
                &parallel(4),
                &RoutingDistribution::uniform(NUM_EXPERTS),
                false,
                mode,
            )
            .unwrap();
            let bridge = PerfApiBridge::new_uninit_for_test();
            bridge.enable_enumerate();
            let built = build("unified".to_string(), resolve_configs(&cfg), &bridge).unwrap();
            assert_eq!(built.n_slots, expected, "slot count for {mode:?}");
            assert_eq!(built.num_attn_dp_groups(), 1);
            assert_eq!(built.num_attn_shards(), 4);
            let description = built.cost_tree().describe();
            assert!(description.contains("TP=EP4; attention DP groups=1"));
            assert!(!description.contains("attention TP1"));
        }

        let cfg = build_configs(
            &model(),
            &parallel(4),
            &RoutingDistribution::uniform(NUM_EXPERTS),
            false,
            Glm52MtpMode::Off,
        )
        .unwrap();
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let built = build("unified".to_string(), resolve_configs(&cfg), &bridge).unwrap();
        let slots = built.cost_tree().slots;
        for body in [
            "unified.body.sparse_initial_index_share",
            "unified.body.sparse_cycle_full_index",
            "unified.body.sparse_cycle_index_share",
        ] {
            let fallback = slots
                .iter()
                .position(|slot| slot.name == format!("{body}.attention.tp_allreduce"))
                .unwrap();
            let fused = slots
                .iter()
                .position(|slot| {
                    slot.name == format!("{body}.attention.tp_allreduce_residual_norm")
                })
                .unwrap();
            let router = slots
                .iter()
                .position(|slot| slot.name.starts_with(&format!("{body}.moe.router.")))
                .unwrap();
            assert_eq!(fallback + 1, fused);
            assert_eq!(fused + 1, router);

            let ffn_fallback = slots
                .iter()
                .position(|slot| slot.name == format!("{body}.moe.tp_allreduce_fallback"))
                .unwrap();
            let ffn_fused = slots
                .iter()
                .position(|slot| slot.name == format!("{body}.moe.tp_allreduce"))
                .unwrap();
            assert_eq!(ffn_fallback + 1, ffn_fused);
        }
    }

    #[test]
    fn location_map_matches_every_unique_noncommunication_manifest_location() {
        let cfg = build_configs(
            &model(),
            &parallel(4),
            &RoutingDistribution::uniform(NUM_EXPERTS),
            false,
            Glm52MtpMode::Off,
        )
        .unwrap();
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let built = build("unified".to_string(), resolve_configs(&cfg), &bridge).unwrap();
        let is_communication = |kind: &str| {
            matches!(
                kind,
                "all_reduce" | "all_reduce_fusion" | "all_reduce_residual_rms_norm"
            )
        };
        let manifest_locations: BTreeSet<String> = built
            .cost_log_manifest()
            .slots
            .into_iter()
            .filter(|slot| !is_communication(&slot.kind))
            .map(|slot| slot.name)
            .collect();

        let map_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("model/work/location_maps/glm52_vllm_nvfp4_dsa_moe_unified.json");
        let map: serde_json::Value =
            serde_json::from_slice(&std::fs::read(map_path).unwrap()).unwrap();
        let mapped_locations: BTreeSet<String> = map["locations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["location"].as_str().unwrap().to_string())
            .collect();

        assert_eq!(manifest_locations.len(), 114);
        assert_eq!(mapped_locations, manifest_locations);
    }

    #[test]
    fn state_and_topology_contracts_are_exact() {
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
    }

    #[test]
    fn replicated_fp8_cache_matches_the_profiled_vllm_capacity() {
        // B200 vLLM startup evidence at gpu_memory_utilization=0.75 reported
        // 17.55 GiB available per rank and exactly 341,120 cache tokens. The
        // byte value below is independently recovered from that token count
        // and vLLM's per-rank cache allocation, avoiding rounded GiB math.
        const PROFILED_BYTES_PER_RANK: u64 = 18_838_010_880;
        const PROFILED_TOKENS: u64 = 341_120;
        const TP_SIZE: u16 = 4;

        let total_bytes = PROFILED_BYTES_PER_RANK * u64::from(TP_SIZE);
        let bytes_per_token = state_bytes_per_token(TP_SIZE, Glm52MtpMode::Off).unwrap();
        assert_eq!(total_bytes / bytes_per_token, PROFILED_TOKENS);
        assert_eq!(total_bytes % bytes_per_token, 0);
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
    fn input_lowering_preserves_each_prefill_request_boundary() {
        let input = UnifiedArchInput {
            groups: vec![group(5, vec![64, 91], vec![(3, 2), (0, 3)])],
            tokens_per_source_rank: Vec::new(),
        };
        let normalized = normalize_input(&input, 4, 8_192).unwrap();
        assert_eq!(normalized.total_tokens, 7);
        assert_eq!(
            normalized.groups[0]
                .attention_input
                .prefill_query_cache_pairs,
            vec![(2, 5), (3, 3)]
        );
        let decode = normalized.groups[0]
            .attention_input
            .decode
            .as_ref()
            .unwrap();
        assert_eq!(decode.batch_size, 2);
        assert_eq!(decode.context_len, 91);
        assert_eq!(decode.context_lens.as_deref(), Some(&[64, 91][..]));
        assert!(!decode.requires_padding);
        assert_eq!(normalized.groups[0].request_count, 4);

        let mtp = normalized.decode_only();
        assert_eq!(mtp.total_tokens, 2);
        assert!(mtp.groups[0]
            .attention_input
            .prefill_query_cache_pairs
            .is_empty());
        assert_eq!(
            mtp.groups[0]
                .attention_input
                .decode
                .as_ref()
                .and_then(|decode| decode.context_lens.as_deref()),
            Some(&[64, 91][..])
        );
    }

    #[test]
    fn input_validation_rejects_wrong_group_count_and_context() {
        let too_many_groups = UnifiedArchInput {
            groups: vec![ArchGroupInput::default(); 2],
            tokens_per_source_rank: Vec::new(),
        };
        assert!(normalize_input(&too_many_groups, 4, 8_192).is_err());

        let bad_prefill = UnifiedArchInput {
            groups: vec![group(1, vec![], vec![(0, 2)])],
            tokens_per_source_rank: Vec::new(),
        };
        assert!(normalize_input(&bad_prefill, 4, 8_192).is_err());

        let too_long = UnifiedArchInput {
            groups: vec![group(0, vec![8_193], vec![])],
            tokens_per_source_rank: Vec::new(),
        };
        assert!(normalize_input(&too_long, 4, 8_192).is_err());
    }
}
