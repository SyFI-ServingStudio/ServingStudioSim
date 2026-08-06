//! `glm52_vllm_dsa_moe` — GLM-5.2 iter-wise architecture in **vLLM kernel
//! granularity**, for VibeSim-vs-vLLM alignment.
//!
//! Structurally identical to [`super::glm52_dsa_moe`]: attention is TP1 and
//! independently replicated over the EP ranks; decoder layers 0--2 execute the
//! dense FFN and a full DSA indexer; layers 3--5 reuse the layer-2 index; layers
//! 6--77 repeat a four-layer cadence containing one full-index layer and three
//! IndexShare layers; sparse-MoE communication is pure EP
//! (`Placement::RoundRobin`); shared-expert compute is conservatively serialized
//! with routed-expert work.
//!
//! This is a **separate static graph**, not a flag on the native arch -- the same
//! split Qwen uses (`qwen3_moe_dp_attn_ep_ffn` / `_fp8_` /
//! `qwen3_vllm_moe_dp_attn_ep_ffn`). The two files diverge only in leaf
//! granularity, and keeping them apart is what stops either graph from growing
//! `if measuring_vllm` branches.
//!
//! Divergences from the native graph, each traced to measured evidence recorded
//! in `doc/alignment/glm52_dp8_ep8_report.md`:
//!   - **fp8 activation quantisation is an explicit leaf.** vLLM launches
//!     `scale_1x128_kernel<bf16, fp8_e4m3, float>` before every dense fp8 GEMM
//!     (411 launches/iteration, 0.758 ms/iteration = 1.9% of measured forward
//!     kernel time). The native graph has no such leaf, because its L1 GEMM
//!     runner quantises outside the timed closure
//!     (`profiling/runners/gemm/deepgemm.py`) -- so the cost is simply absent
//!     there, not folded in.
//!
//! The checkpoint identity (`Glm52ModelCfg`) and MTP identity (`Glm52MtpMode`)
//! are shared with the native graph rather than re-parsed here: they describe the
//! checkpoint, which does not change with the measurement viewpoint.
//!
//! The checkpoint advertises a 1,048,576-token context. The accepted L1 timing
//! domain is deliberately capped at 131,072 tokens, as in the native graph.

use std::sync::Arc;

use anyhow::Result;

use crate::arch::glm52_dsa_moe::{Glm52ModelCfg, Glm52MtpMode};
use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::Fabric;
use crate::op::moe::{MoeCombineOp, MoeDispatchOp, MoeNetConfig, MoeNetInput, Placement};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, GroupedGemmKernelInput,
    ResidualRmsNormKernel, ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput,
    SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::routing::RoutingDistribution;
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Dim, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, Probe, SlotInput,
};
use crate::worklet::{
    VllmGlm52DenseFfnLocalWorklet, VllmGlm52DenseFfnLocalWorkletConfig,
    VllmGlm52DenseFfnLocalWorkletInput, VllmGlm52DenseFfnLocalWorkletResolved,
    VllmGlm52DsaAttnLocalDecodeInput, VllmGlm52DsaAttnLocalWorklet, VllmGlm52DsaAttnLocalWorkletConfig,
    VllmGlm52DsaAttnLocalWorkletInput, VllmGlm52DsaAttnLocalWorkletResolved, Glm52MoeRouterLocalWorklet,
    Glm52MoeRouterLocalWorkletConfig, Glm52MoeRouterLocalWorkletInput,
    Glm52MoeRouterLocalWorkletResolved, Glm52MtpHeadLocalWorklet, Glm52MtpHeadLocalWorkletConfig,
    Glm52MtpHeadLocalWorkletInput, Glm52MtpHeadLocalWorkletResolved, Glm52MtpPreludeLocalWorklet,
    Glm52MtpPreludeLocalWorkletConfig, Glm52MtpPreludeLocalWorkletInput,
    Glm52MtpPreludeLocalWorkletResolved, VllmGlm52SharedExpertLocalWorklet,
    VllmGlm52SharedExpertLocalWorkletConfig, VllmGlm52SharedExpertLocalWorkletInput,
    VllmGlm52SharedExpertLocalWorkletResolved, MoeExpertComputeLocalWorklet,
    MoeExpertComputeLocalWorkletConfig, MoeExpertComputeLocalWorkletInput,
    MoeExpertComputeLocalWorkletResolved,
};

const ARCH_KIND: &str = "glm52_vllm_dsa_moe";
const TIMING_MAX_MODEL_LEN: u32 = 131_072;
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
const LOGITS_ROW_STRIDE: u32 = 131_072;

const FULL_INDEX_LAYERS: [u32; 21] = [
    0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74,
];

const RESIDUAL_NORM_BACKENDS: &[&str] = &["vllm_cuda"];
const RMS_NORM_BACKENDS: &[&str] = &["flashinfer"];
const SINGLE_GEMM_BACKENDS: &[&str] = &["torch_linear"];
const ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
const GROUPED_GEMM_BACKENDS: &[&str] = &["torch"];
const FP8_SINGLE_GEMM_BACKENDS: &[&str] = &["deepgemm"];
const FP8_QUANT_BACKENDS: &[&str] = &["flashinfer_trtllm"];
const Q_ABSORB_BACKENDS: &[&str] = &["torch_mla_q_absorb_glm52"];
const V_UP_BACKENDS: &[&str] = &["torch_mla_v_up_glm52"];
const INDEX_CACHE_AND_TOPK_BACKENDS: &[&str] = &["vllm_cuda"];
const INDEX_LOGITS_BACKENDS: &[&str] = &["vllm_deepgemm_fp8"];
const SPARSE_ATTN_BACKENDS: &[&str] = &["vllm_flashmla_bf16"];
const MLA_APPEND_BACKENDS: &[&str] = &["vllm_cuda"];
const P2P_BACKENDS: &[&str] = &["nccl"];
const FP8_GROUPED_GEMM_BACKENDS: &[&str] = &["deepgemm"];
const FP8_PRODUCTION_GROUPED_GEMM_BACKENDS: &[&str] = &["flashinfer_trtllm"];
const FP8_P2P_BACKENDS: &[&str] = &["nvshmem"];

// Leaf counts are properties of the accepted L2/L3 sections. `build` verifies
// the compiled tree against these formulas, so drift cannot be hidden by a
// stale handwritten expectation.
const ATTN_FULL_SLOTS: usize = 29;
const ATTN_SHARED_SLOTS: usize = 14;
const DENSE_FFN_SLOTS: usize = 4;
const ROUTER_SLOTS: usize = 4;
const DISPATCH_SLOTS: usize = 2;
const EXPERT_SLOTS: usize = 3;
const COMBINE_SLOTS: usize = 4;
const SHARED_EXPERT_SLOTS: usize = 3;
const FINALIZE_SLOTS: usize = 1;
const MTP_PRELUDE_SLOTS: usize = 6;
const MTP_HEAD_SLOTS: usize = 3;

#[derive(Clone, Debug)]
pub struct Glm52VllmDsaMoeParallel {
    pub ep_size: u16,
    pub nvl_num_gpu: u16,
    pub gpu_name: String,
}

#[derive(Clone, Debug)]
pub struct Glm52VllmDsaMoeConfigs {
    pub model: Glm52ModelCfg,
    pub parallel: Glm52VllmDsaMoeParallel,
    pub mtp_mode: Glm52MtpMode,
    pub dense_full_index_attention: VllmGlm52DsaAttnLocalWorkletConfig,
    pub dense_ffn: VllmGlm52DenseFfnLocalWorkletConfig,
    pub initial_shared_attention: VllmGlm52DsaAttnLocalWorkletConfig,
    pub cycle_full_attention: VllmGlm52DsaAttnLocalWorkletConfig,
    pub cycle_shared_attention: VllmGlm52DsaAttnLocalWorkletConfig,
    pub sparse_router: Glm52MoeRouterLocalWorkletConfig,
    pub moe_dispatch: MoeNetConfig,
    /// One config per EP rank. The rank axis is what makes the routed-expert
    /// grouped GEMMs distribution-sensitive: each rank owns its own
    /// `local_ppm` shard, so a measured or synthetic skew yields a distinct
    /// grouped-GEMM cache identity per rank and the `Max` over ranks lands on
    /// the genuinely heaviest one. A uniform distribution still produces
    /// `ep_size` identical entries, so the slot layout never changes.
    pub moe_expert_compute: Vec<MoeExpertComputeLocalWorkletConfig>,
    pub moe_combine: MoeNetConfig,
    pub shared_expert: VllmGlm52SharedExpertLocalWorkletConfig,
    pub sparse_finalization: ElementwiseKernelConfig,
    pub embedding: ElementwiseKernelConfig,
    pub final_norm: ResidualRmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub mtp_prelude: Option<Glm52MtpPreludeLocalWorkletConfig>,
    pub mtp_attention: Option<VllmGlm52DsaAttnLocalWorkletConfig>,
    pub mtp_head: Option<Glm52MtpHeadLocalWorkletConfig>,
}

#[derive(Clone, Debug)]
pub struct Glm52VllmDsaMoeResolved {
    pub raw_cfg: Glm52VllmDsaMoeConfigs,
    pub dense_full_index_attention: VllmGlm52DsaAttnLocalWorkletResolved,
    pub dense_ffn: VllmGlm52DenseFfnLocalWorkletResolved,
    pub initial_shared_attention: VllmGlm52DsaAttnLocalWorkletResolved,
    pub cycle_full_attention: VllmGlm52DsaAttnLocalWorkletResolved,
    pub cycle_shared_attention: VllmGlm52DsaAttnLocalWorkletResolved,
    pub sparse_router: Glm52MoeRouterLocalWorkletResolved,
    pub moe_dispatch: MoeNetConfig,
    pub moe_expert_compute: Vec<MoeExpertComputeLocalWorkletResolved>,
    pub moe_combine: MoeNetConfig,
    pub shared_expert: VllmGlm52SharedExpertLocalWorkletResolved,
    pub sparse_finalization: ElementwiseKernelConfig,
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
    parallel: &Glm52VllmDsaMoeParallel,
    include_indexer: bool,
    fp8: bool,
) -> VllmGlm52DsaAttnLocalWorkletConfig {
    let (gemm_dtype, gemm_backends) = if fp8 {
        (DType::Fp8E4m3, FP8_SINGLE_GEMM_BACKENDS)
    } else {
        (DType::Bf16, SINGLE_GEMM_BACKENDS)
    };
    VllmGlm52DsaAttnLocalWorkletConfig {
        fp8_quant_backends: FP8_QUANT_BACKENDS.to_vec(),
        include_indexer,
        residual_rms_norm_backends: RESIDUAL_NORM_BACKENDS.to_vec(),
        rms_norm_backends: RMS_NORM_BACKENDS.to_vec(),
        single_gemm_backends: gemm_backends.to_vec(),
        elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
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
        max_model_len: Dim::param("timing_max_model_len", TIMING_MAX_MODEL_LEN),
        logits_row_stride: Dim::param("logits_row_stride", LOGITS_ROW_STRIDE),
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
        index_clean_logits: false,
        // Production selected indices are top-k ordered and page-local; this is
        // the smallest supported deterministic profiling identity.
        sparse_index_distribution: "recent_contiguous".to_string(),
        sparse_cache_layout: "token_major_mqa_bf16_latent_rope".to_string(),
        sparse_mla_cache_format: "plain".to_string(),
        decode_next_n: 1,
    }
}

pub fn build_configs(
    model: &Glm52ModelCfg,
    parallel: &Glm52VllmDsaMoeParallel,
    routing: &RoutingDistribution,
    fp8: bool,
    mtp_mode: Glm52MtpMode,
) -> Result<Glm52VllmDsaMoeConfigs, BuildError> {
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
    if routing.num_experts() != NUM_EXPERTS {
        return Err(fit_failed(format!(
            "routing distribution has {} experts, expected {NUM_EXPERTS}",
            routing.num_experts()
        )));
    }
    let (expert_dtype, expert_backends, p2p_backends) = if fp8 {
        (
            DType::Fp8E4m3,
            FP8_GROUPED_GEMM_BACKENDS,
            FP8_P2P_BACKENDS,
        )
    } else {
        (DType::Bf16, GROUPED_GEMM_BACKENDS, P2P_BACKENDS)
    };
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
    let moe_net = MoeNetConfig {
        backends: p2p_backends.to_vec(),
        gpu_name: gpu.clone(),
        dtype: if fp8 { DType::Fp8E4m3 } else { DType::Bf16 },
        intra_fabric: Fabric::Nvlink,
        inter_fabric: Fabric::Infiniband,
        ep_size: u32::from(parallel.ep_size),
        nvl_num_gpu: u32::from(parallel.nvl_num_gpu),
        h: HIDDEN_DIM,
        top_k: ROUTER_TOP_K,
        routing: routing.clone(),
        placement: Placement::RoundRobin,
    };
    let hidden_bytes = HIDDEN_DIM
        .checked_mul(DType::Bf16.size_bytes())
        .ok_or_else(|| fit_failed("hidden byte width overflows u32"))?;
    let embedding_input = hidden_bytes
        .checked_add(8)
        .ok_or_else(|| fit_failed("embedding input byte rate overflows u32"))?;
    let finalization_input = 9_u32
        .checked_mul(hidden_bytes)
        .ok_or_else(|| fit_failed("MoE finalization input byte rate overflows u32"))?;

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

    Ok(Glm52VllmDsaMoeConfigs {
        model: model.clone(),
        parallel: parallel.clone(),
        mtp_mode,
        dense_full_index_attention,
        dense_ffn: VllmGlm52DenseFfnLocalWorkletConfig {
            fp8_quant_backends: FP8_QUANT_BACKENDS.to_vec(),
            residual_norm_backends: RESIDUAL_NORM_BACKENDS.to_vec(),
            gemm_backends: gemm_backends.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
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
            proxy_gemm_backends: gemm_backends.to_vec(),
            gpu_name: gpu.clone(),
            hidden_dim: model.hidden_dim.clone(),
            num_experts: model.num_experts.clone(),
            top_k: model.router_top_k,
            n_group: 1,
            topk_group: 1,
            base_dtype: DType::Bf16,
            router_semantic_dtype: DType::Fp32,
            proxy_gemm_dtype: gemm_dtype,
            index_dtype: "int32".to_string(),
            scoring_func: "sigmoid".to_string(),
            topk_method: "noaux_tc".to_string(),
            norm_topk_prob: true,
            routed_scaling_numerator: 5,
            routed_scaling_denominator: 2,
        },
        moe_dispatch: moe_net.clone(),
        // `split_for_ep` hands rank *r* the contiguous expert range
        // `[r * experts_per_rank, (r + 1) * experts_per_rank)`. That is the
        // shard the routed grouped GEMMs are keyed by; the `RoundRobin` in
        // `MoeNetConfig` above only governs which peer a token is sent to.
        moe_expert_compute: MoeExpertComputeLocalWorkletConfig::split_for_ep(
            MoeExpertComputeLocalWorkletConfig {
                hidden: model.hidden_dim.clone(),
                moe_intermediate: model.moe_intermediate_dim.clone(),
                num_experts: model.num_experts.clone(),
                ep_size: parallel.ep_size,
                top_k: model.router_top_k,
                dtype: expert_dtype,
                activation_dtype: DType::Bf16,
                gpu_name: gpu.clone(),
                act_backends: ELEMENTWISE_BACKENDS.to_vec(),
                fp8_quant_backends: FP8_QUANT_BACKENDS.to_vec(),
                grouped_gemm_backends: expert_backends.to_vec(),
                fp8_grouped_gemm_backends: FP8_PRODUCTION_GROUPED_GEMM_BACKENDS.to_vec(),
                use_fp8_blockscale_grouped_gemm: true,
                local_ppm: Vec::new(),
            },
            routing.ppm(),
        ),
        moe_combine: moe_net,
        shared_expert: VllmGlm52SharedExpertLocalWorkletConfig {
            gemm_backends: gemm_backends.to_vec(),
            fp8_quant_backends: FP8_QUANT_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden_dim: model.hidden_dim.clone(),
            moe_intermediate_dim: model.moe_intermediate_dim.clone(),
            n_shared_experts: model.num_shared_experts,
            dtype: DType::Bf16,
            gemm_dtype,
        },
        sparse_finalization: ElementwiseKernelConfig {
            backends: ELEMENTWISE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            input_bytes_per_token: finalization_input.into(),
            output_bytes_per_token: hidden_bytes.into(),
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
            backends: gemm_backends.to_vec(),
            gpu_name: gpu,
            n: model.vocab_size.clone(),
            k: model.hidden_dim.clone(),
            dtype: gemm_dtype,
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

pub fn resolve_configs(cfgs: &Glm52VllmDsaMoeConfigs) -> Glm52VllmDsaMoeResolved {
    Glm52VllmDsaMoeResolved {
        dense_full_index_attention: VllmGlm52DsaAttnLocalWorklet::resolve_config(
            &cfgs.dense_full_index_attention,
        ),
        dense_ffn: VllmGlm52DenseFfnLocalWorklet::resolve_config(&cfgs.dense_ffn),
        initial_shared_attention: VllmGlm52DsaAttnLocalWorklet::resolve_config(
            &cfgs.initial_shared_attention,
        ),
        cycle_full_attention: VllmGlm52DsaAttnLocalWorklet::resolve_config(&cfgs.cycle_full_attention),
        cycle_shared_attention: VllmGlm52DsaAttnLocalWorklet::resolve_config(
            &cfgs.cycle_shared_attention,
        ),
        sparse_router: Glm52MoeRouterLocalWorklet::resolve_config(&cfgs.sparse_router),
        moe_dispatch: cfgs.moe_dispatch.clone(),
        moe_expert_compute: cfgs
            .moe_expert_compute
            .iter()
            .map(MoeExpertComputeLocalWorklet::resolve_config)
            .collect(),
        moe_combine: cfgs.moe_combine.clone(),
        shared_expert: VllmGlm52SharedExpertLocalWorklet::resolve_config(&cfgs.shared_expert),
        sparse_finalization: cfgs.sparse_finalization.clone(),
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
    dispatch: MoeDispatchOp,
    /// One per EP rank, in rank order.
    expert_compute: Vec<MoeExpertComputeLocalWorklet>,
    combine: MoeCombineOp,
    shared_expert: VllmGlm52SharedExpertLocalWorklet,
    finalization: Op<ElementwiseKernel>,
    ep_size: u16,
    top_k: u32,
}

impl Glm52SparseBody {
    fn build(
        name: String,
        attention: VllmGlm52DsaAttnLocalWorkletResolved,
        common: &Glm52VllmDsaMoeResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let attention =
            VllmGlm52DsaAttnLocalWorklet::build(format!("{name}.attention"), attention, bridge)?;
        let router = Glm52MoeRouterLocalWorklet::build(
            format!("{name}.moe.router"),
            common.sparse_router.clone(),
            bridge,
        )?;
        let dispatch = MoeDispatchOp::build(
            format!("{name}.moe.dispatch"),
            common.moe_dispatch.clone(),
            bridge,
        )?;
        // Every rank keeps the same slot name: the rank axis already shows up
        // as the `Max` node's children, and a rank suffix would rename the
        // slots a labeled kernel inventory refers to.
        let expert_compute = common
            .moe_expert_compute
            .iter()
            .map(|rank_resolved| {
                MoeExpertComputeLocalWorklet::build(
                    format!("{name}.moe.routed_experts"),
                    rank_resolved.clone(),
                    bridge,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let combine = MoeCombineOp::build(
            format!("{name}.moe.combine"),
            common.moe_combine.clone(),
            bridge,
        )?;
        let shared_expert = VllmGlm52SharedExpertLocalWorklet::build(
            format!("{name}.moe.shared_expert"),
            common.shared_expert.clone(),
            bridge,
        )?;
        let finalization = build_atomic(
            format!("{name}.moe.finalization"),
            common.sparse_finalization.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        Ok(Self {
            name,
            attention,
            router,
            dispatch,
            expert_compute,
            combine,
            shared_expert,
            finalization,
            ep_size: common.raw_cfg.parallel.ep_size,
            top_k: common.raw_cfg.model.router_top_k,
        })
    }

    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let attention = labeled_max(
            format!("{}.attention [Max over attention-DP groups]", self.name),
            (0..self.ep_size)
                .map(|_| self.attention.compile(builder))
                .collect(),
        );
        let router = labeled_max(
            format!("{}.moe.router [Max over home groups]", self.name),
            (0..self.ep_size)
                .map(|_| self.router.compile(builder))
                .collect(),
        );
        let dispatch = self.dispatch.compile(builder);
        let experts = labeled_max(
            format!("{}.moe.routed_experts [Max over EP ranks]", self.name),
            self.expert_compute
                .iter()
                .map(|rank_expert| rank_expert.compile(builder))
                .collect(),
        );
        let combine = self.combine.compile(builder);
        let shared = labeled_max(
            format!("{}.moe.shared_expert [Max over home groups]", self.name),
            (0..self.ep_size)
                .map(|_| self.shared_expert.compile(builder))
                .collect(),
        );
        let finalization = labeled_max(
            format!("{}.moe.finalization [Max over home groups]", self.name),
            (0..self.ep_size)
                .map(|_| self.finalization.compile(builder))
                .collect(),
        );
        CostNode::Labeled {
            label: format!(
                "{} [sparse layer; TP1 attention; EP{} top-{} routed/shared experts serial (conservative no auxiliary-stream overlap)]",
                self.name, self.ep_size, self.top_k
            ),
            child: Box::new(CostNode::Sum(vec![
                attention,
                CostNode::Labeled {
                    label: format!("{}.moe [router -> dispatch -> routed experts -> combine -> shared expert -> finalization]", self.name),
                    child: Box::new(CostNode::Sum(vec![
                        router,
                        dispatch,
                        experts,
                        combine,
                        shared,
                        finalization,
                    ])),
                },
            ])),
        }
    }

    fn eval(&self, batch: &NormalizedBatch, ev: &mut Evaluator) {
        for group in &batch.groups {
            self.attention.eval(&group.attention_input, ev);
        }
        for group in &batch.groups {
            self.router.eval(
                &Glm52MoeRouterLocalWorkletInput {
                    batch_tokens: group.batch_tokens,
                },
                ev,
            );
        }
        self.dispatch.eval(
            &MoeNetInput {
                tokens: u64::from(batch.total_tokens),
            },
            ev,
        );
        for rank_expert in &self.expert_compute {
            eval_expert_or_zero(rank_expert, batch.routed_selections, ev);
        }
        self.combine.eval(
            &MoeNetInput {
                tokens: u64::from(batch.total_tokens),
            },
            ev,
        );
        for group in &batch.groups {
            self.shared_expert.eval(
                &VllmGlm52SharedExpertLocalWorkletInput {
                    batch_tokens: group.batch_tokens,
                },
                ev,
            );
        }
        for group in &batch.groups {
            eval_atomic_or_zero(
                &self.finalization,
                ElementwiseKernelInput {
                    num_tokens: group.batch_tokens,
                },
                group.batch_tokens == 0,
                ev,
            );
        }
    }
}

struct Glm52MtpSection {
    prelude: Glm52MtpPreludeLocalWorklet,
    decoder: Glm52SparseBody,
    head: Glm52MtpHeadLocalWorklet,
}

pub struct Glm52VllmDsaMoeModel {
    pub name: String,
    pub mtp_mode: Glm52MtpMode,
    pub ep_size: u16,
    pub nvl_num_gpu: u16,
    pub num_attn_dp_groups: u16,
    pub num_attn_shards: u16,
    pub total_state_bytes_per_token: u64,
    pub embedding: Op<ElementwiseKernel>,
    pub dense_full_index_attention: VllmGlm52DsaAttnLocalWorklet,
    pub dense_ffn: VllmGlm52DenseFfnLocalWorklet,
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
    resolved: Glm52VllmDsaMoeResolved,
    bridge: &PerfApiBridge,
) -> Result<Glm52VllmDsaMoeModel, BuildError> {
    let ep_size = resolved.raw_cfg.parallel.ep_size;
    let nvl_num_gpu = resolved.raw_cfg.parallel.nvl_num_gpu;
    let mtp_mode = resolved.raw_cfg.mtp_mode;
    let embedding = build_atomic(
        format!("{name}.main.embedding"),
        resolved.embedding.clone(),
        ElementwiseKernel::build,
        bridge,
    )?;
    let dense_full_index_attention = VllmGlm52DsaAttnLocalWorklet::build(
        format!("{name}.body.dense_full_index.attention"),
        resolved.dense_full_index_attention.clone(),
        bridge,
    )?;
    let dense_ffn = VllmGlm52DenseFfnLocalWorklet::build(
        format!("{name}.body.dense_full_index.ffn"),
        resolved.dense_ffn.clone(),
        bridge,
    )?;
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

    let total_state_bytes_per_token = state_bytes_per_token(mtp_mode)?;
    let fp8 = resolved.raw_cfg.moe_expert_compute[0].dtype == DType::Fp8E4m3;
    let mut model = Glm52VllmDsaMoeModel {
        name,
        mtp_mode,
        ep_size,
        nvl_num_gpu,
        num_attn_dp_groups: ep_size,
        num_attn_shards: 1,
        total_state_bytes_per_token,
        embedding,
        dense_full_index_attention,
        dense_ffn,
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
    let expected = expected_slot_count(ep_size, fp8, mtp_mode);
    if tree.n_slots() != expected {
        return Err(fit_failed(format!(
            "compiled slot count {} differs from frozen formula {expected}",
            tree.n_slots()
        )));
    }
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

impl Glm52VllmDsaMoeModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let embedding = labeled_max(
            format!(
                "{}.main.embedding [Max over attention-DP groups]",
                self.name
            ),
            (0..self.ep_size)
                .map(|_| self.embedding.compile(&mut builder))
                .collect(),
        );
        let dense_attention = labeled_max(
            format!(
                "{}.body.dense_full_index.attention [Max over attention-DP groups]",
                self.name
            ),
            (0..self.ep_size)
                .map(|_| self.dense_full_index_attention.compile(&mut builder))
                .collect(),
        );
        let dense_ffn = labeled_max(
            format!(
                "{}.body.dense_full_index.ffn [Max over attention-DP/home groups]",
                self.name
            ),
            (0..self.ep_size)
                .map(|_| self.dense_ffn.compile(&mut builder))
                .collect(),
        );
        let dense = CostNode::Labeled {
            label: "layers 0..2: dense + full index (3 layers)".to_string(),
            child: Box::new(CostNode::Scale {
                n: NUM_DENSE_LAYERS,
                child: Box::new(CostNode::Sum(vec![dense_attention, dense_ffn])),
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
                "{}.main.final_residual_rms_norm [Max over attention-DP/home groups]",
                self.name
            ),
            (0..self.ep_size)
                .map(|_| self.final_norm.compile(&mut builder))
                .collect(),
        );
        let lm_head = labeled_max(
            format!(
                "{}.main.lm_head [Max over attention-DP/home groups]",
                self.name
            ),
            (0..self.ep_size)
                .map(|_| self.lm_head.compile(&mut builder))
                .collect(),
        );
        let mut children = vec![embedding, dense, initial_shared, cycle, final_norm, lm_head];
        if let Some(mtp) = &self.mtp {
            let prelude = labeled_max(
                format!("{}.mtp.prelude [Max over attention-DP groups]", self.name),
                (0..self.ep_size)
                    .map(|_| mtp.prelude.compile(&mut builder))
                    .collect(),
            );
            let decoder = mtp.decoder.compile(&mut builder);
            let head = labeled_max(
                format!("{}.mtp.head [Max over attention-DP/home groups]", self.name),
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
                "{} (Glm52VllmDsaMoeModel) [EP{}; attention TP1/DP{}; MTP={:?}; timing_context<=131072]",
                self.name, self.ep_size, self.num_attn_dp_groups, self.mtp_mode
            ),
            child: Box::new(CostNode::Sum(children)),
        };
        builder.finish(root)
    }

    fn eval_into(&self, input: &UnifiedArchInput, ev: &mut Evaluator) {
        let batch = normalize_input(input, self.ep_size)
            .unwrap_or_else(|reason| panic!("invalid Glm52VllmDsaMoeModel input: {reason}"));

        for group in &batch.groups {
            eval_atomic_or_zero(
                &self.embedding,
                ElementwiseKernelInput {
                    num_tokens: group.batch_tokens,
                },
                group.batch_tokens == 0,
                ev,
            );
        }
        for group in &batch.groups {
            self.dense_full_index_attention
                .eval(&group.attention_input, ev);
        }
        for group in &batch.groups {
            self.dense_ffn.eval(
                &VllmGlm52DenseFfnLocalWorkletInput {
                    batch_tokens: group.batch_tokens,
                },
                ev,
            );
        }
        self.initial_shared_sparse.eval(&batch, ev);
        self.cycle_full_sparse.eval(&batch, ev);
        self.cycle_shared_sparse.eval(&batch, ev);
        for group in &batch.groups {
            eval_atomic_or_zero(
                &self.final_norm,
                ResidualRmsNormKernelInput {
                    m: group.batch_tokens,
                },
                group.batch_tokens == 0,
                ev,
            );
        }
        for group in &batch.groups {
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
            for group in &batch.groups {
                mtp.prelude.eval(
                    &Glm52MtpPreludeLocalWorkletInput {
                        batch_tokens: group.decode_tokens,
                    },
                    ev,
                );
            }
            let mtp_batch = batch.decode_only();
            mtp.decoder.eval(&mtp_batch, ev);
            for group in &batch.groups {
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

impl IterwiseUnifiedModel for Glm52VllmDsaMoeModel {
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
    routed_selections: u32,
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
                    decode: group
                        .decode_context
                        .map(|context_len| VllmGlm52DsaAttnLocalDecodeInput {
                            batch_size: group.decode_tokens,
                            context_len,
                            requires_padding: false,
                        }),
                },
            })
            .collect();
        let total_tokens: u32 = groups.iter().map(|group| group.batch_tokens).sum();
        let routed_selections = total_tokens
            .checked_mul(ROUTER_TOP_K)
            .expect("validated decode-only routed selections must fit u32");
        Self {
            groups,
            total_tokens,
            routed_selections,
        }
    }
}

fn normalize_input(
    input: &UnifiedArchInput,
    ep_size: u16,
) -> std::result::Result<NormalizedBatch, String> {
    if input.groups.len() != usize::from(ep_size) {
        return Err(format!(
            "expected exactly {ep_size} attention-DP groups, got {}",
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
            if cache_tokens > TIMING_MAX_MODEL_LEN {
                return Err(format!(
                    "group {group_index} prefill request {request_index} context {cache_tokens} exceeds timing cap {TIMING_MAX_MODEL_LEN}"
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
            if !(1..=TIMING_MAX_MODEL_LEN).contains(&context) {
                return Err(format!(
                    "group {group_index} decode request {request_index} context {context} must be in 1..={TIMING_MAX_MODEL_LEN}"
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
                    requires_padding: false,
                }),
            },
        });
    }
    let routed_selections = total_tokens
        .checked_mul(ROUTER_TOP_K)
        .ok_or_else(|| "pooled routed-selection count overflows u32".to_string())?;
    Ok(NormalizedBatch {
        groups,
        total_tokens,
        routed_selections,
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

fn eval_expert_or_zero(
    expert: &MoeExpertComputeLocalWorklet,
    global_expert_selections: u32,
    ev: &mut Evaluator,
) {
    if global_expert_selections != 0 {
        expert.eval(
            &MoeExpertComputeLocalWorkletInput {
                global_expert_selections,
            },
            ev,
        );
        return;
    }
    let grouped = GroupedGemmKernelInput {
        global_expert_selections: 0,
    };
    let activation = ElementwiseKernelInput { num_tokens: 0 };
    ev.push(LeafMetrics::ZERO, || grouped.clone().into());
    ev.push(LeafMetrics::ZERO, || activation.into());
    ev.push(LeafMetrics::ZERO, || grouped.into());
}

fn expected_slot_count(ep_size: u16, fp8: bool, mtp_mode: Glm52MtpMode) -> usize {
    let ep = usize::from(ep_size);
    // Each FP8 routed-expert worklet adds one BF16→FP8 block-quant leaf before
    // gate/up and one before down. The three main sparse archetypes and the
    // optional MTP sparse decoder all reuse this worklet, so keep the selector
    // at the L4 formula seam rather than hard-coding FP8 counts.
    let expert_slots = EXPERT_SLOTS + usize::from(fp8) * 2;
    // vLLM granularity: one activation-quantisation leaf per dense FP8 GEMM.
    // Attention contributes three everywhere (fused_qkv_a_proj, q_b_proj,
    // o_proj) plus a fourth on a full-index layer, for the indexer's q_proj.
    // The dense FFN and the shared expert contribute two each.
    let attn_full_slots = ATTN_FULL_SLOTS + usize::from(fp8) * 4;
    let attn_shared_slots = ATTN_SHARED_SLOTS + usize::from(fp8) * 3;
    let dense_ffn_slots = DENSE_FFN_SLOTS + usize::from(fp8) * 2;
    let shared_expert_slots = SHARED_EXPERT_SLOTS + usize::from(fp8) * 2;
    let dense = ep * (attn_full_slots + dense_ffn_slots);
    let sparse_shared = ep
        * (attn_shared_slots
            + ROUTER_SLOTS
            + expert_slots
            + shared_expert_slots
            + FINALIZE_SLOTS)
        + DISPATCH_SLOTS
        + COMBINE_SLOTS;
    let sparse_full = sparse_shared + ep * (attn_full_slots - attn_shared_slots);
    let main = ep + dense + sparse_shared + sparse_full + sparse_shared + ep + ep;
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

fn state_bytes_per_token(mtp_mode: Glm52MtpMode) -> std::result::Result<u64, BuildError> {
    let main_mla = u64::from(NUM_LAYERS)
        .checked_mul(u64::from(KV_LORA_RANK + ROPE_DIM))
        .and_then(|value| value.checked_mul(u64::from(DType::Bf16.size_bytes())))
        .ok_or_else(|| fit_failed("main MLA state bytes overflow u64"))?;
    let index = u64::try_from(FULL_INDEX_LAYERS.len())
        .expect("full-index count fits u64")
        .checked_mul(u64::from(INDEX_HEAD_DIM + DType::Fp32.size_bytes()))
        .ok_or_else(|| fit_failed("index state bytes overflow u64"))?;
    let main = main_mla
        .checked_add(index)
        .ok_or_else(|| fit_failed("main state bytes overflow u64"))?;
    let mtp_mla = u64::from(KV_LORA_RANK + ROPE_DIM)
        .checked_mul(u64::from(DType::Bf16.size_bytes()))
        .ok_or_else(|| fit_failed("MTP MLA state bytes overflow u64"))?;
    Ok(match mtp_mode {
        Glm52MtpMode::Off => main,
        Glm52MtpMode::FullIndex => main + mtp_mla + u64::from(INDEX_HEAD_DIM + 4),
        Glm52MtpMode::IndexShare => main + mtp_mla,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::ArchGroupInput;
    use std::collections::HashSet;

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

    fn parallel(ep_size: u16) -> Glm52VllmDsaMoeParallel {
        Glm52VllmDsaMoeParallel {
            ep_size,
            nvl_num_gpu: ep_size.min(8),
            gpu_name: "NVIDIA H200".to_string(),
        }
    }

    #[test]
    fn routed_expert_shards_carry_the_routing_skew_per_ep_rank() {
        let ep_size = 8_u16;
        let experts_per_rank = (NUM_EXPERTS / u32::from(ep_size)) as usize;

        // Uniform: 1e6 ppm does not divide 256 evenly, so `uniform` hands the
        // 64-ppm remainder to the first 64 experts — ranks 0-1 carry 3907 and
        // ranks 2-7 carry 3906. Each rank is internally flat, and the ±1 ppm
        // is far below the resolution of `to_per_expert_counts`, so this only
        // means two grouped-GEMM cache keys instead of one.
        let uniform = build_configs(
            &model(),
            &parallel(ep_size),
            &RoutingDistribution::uniform(NUM_EXPERTS),
            true,
            Glm52MtpMode::Off,
        )
        .unwrap();
        assert_eq!(uniform.moe_expert_compute.len(), usize::from(ep_size));
        for shard in &uniform.moe_expert_compute {
            assert_eq!(shard.local_ppm.len(), experts_per_rank);
            assert!(shard.local_ppm.iter().all(|ppm| *ppm == shard.local_ppm[0]));
            assert!((3906..=3907).contains(&shard.local_ppm[0]));
        }
        assert_eq!(uniform.moe_expert_compute[0].local_ppm[0], 3907);
        assert_eq!(uniform.moe_expert_compute[7].local_ppm[0], 3906);

        // A profile that loads the first rank's experts twice as heavily as the
        // last rank's must reach the grouped-GEMM cache key, not just the
        // dispatch/combine byte counts.
        let mut ratios = vec![0.5_f32; NUM_EXPERTS as usize];
        ratios[..experts_per_rank].fill(1.5);
        let skewed = build_configs(
            &model(),
            &parallel(ep_size),
            &RoutingDistribution::from_profile(&ratios),
            true,
            Glm52MtpMode::Off,
        )
        .unwrap();
        let shard_totals: Vec<u32> = skewed
            .moe_expert_compute
            .iter()
            .map(|shard| shard.local_ppm.iter().sum())
            .collect();
        assert!(
            shard_totals[0] > 2 * shard_totals[usize::from(ep_size) - 1],
            "heaviest EP rank must dominate the lightest, got {shard_totals:?}"
        );
        assert_ne!(
            skewed.moe_expert_compute[0].local_ppm,
            uniform.moe_expert_compute[0].local_ppm
        );
        // The skew must NOT leak into the dispatch/combine byte model's own
        // copy of the distribution being the only place it lands.
        assert_eq!(skewed.moe_dispatch.routing.ppm().len(), NUM_EXPERTS as usize);
    }

    #[test]
    fn build_configs_is_bridge_free_and_bakes_every_backend_and_shape() {
        let routing = RoutingDistribution::uniform(NUM_EXPERTS);
        let cfg = build_configs(
            &model(),
            &parallel(8),
            &routing,
            false,
            Glm52MtpMode::IndexShare,
        )
        .unwrap();
        assert_eq!(cfg.parallel.ep_size, 8);
        assert!(cfg.dense_full_index_attention.include_indexer);
        assert!(!cfg.initial_shared_attention.include_indexer);
        assert!(cfg.cycle_full_attention.include_indexer);
        assert!(!cfg.cycle_shared_attention.include_indexer);
        assert_eq!(cfg.dense_full_index_attention.decode_next_n, 1);
        assert_eq!(
            cfg.dense_full_index_attention.single_gemm_backends,
            vec!["torch_linear"]
        );
        assert_eq!(
            cfg.dense_full_index_attention.gemm_dtype,
            DType::Bf16
        );
        assert_eq!(
            cfg.dense_full_index_attention.q_absorb_backends,
            vec!["torch_mla_q_absorb_glm52"]
        );
        assert_eq!(
            cfg.dense_full_index_attention.v_up_backends,
            vec!["torch_mla_v_up_glm52"]
        );
        assert_eq!(
            cfg.dense_full_index_attention.sparse_attention_backends,
            vec!["vllm_flashmla_bf16"]
        );
        assert_eq!(
            cfg.dense_full_index_attention.index_prefill_logits_backends,
            vec!["vllm_deepgemm_fp8"]
        );
        assert_eq!(cfg.moe_expert_compute[0].dtype, DType::Bf16);
        assert_eq!(cfg.moe_expert_compute[0].grouped_gemm_backends, vec!["torch"]);
        assert_eq!(cfg.moe_dispatch.backends, vec!["nccl"]);
        assert_eq!(cfg.moe_combine.backends, vec!["nccl"]);
        assert_eq!(cfg.moe_dispatch.dtype, DType::Bf16);
        assert_eq!(cfg.moe_combine.dtype, DType::Bf16);
        assert_eq!(cfg.moe_dispatch.net_params().hidden_bytes, HIDDEN_DIM * 2);
        assert_eq!(cfg.moe_combine.net_params().hidden_bytes, HIDDEN_DIM * 2);
        assert_eq!(cfg.sparse_router.base_dtype, DType::Bf16);
        assert_eq!(cfg.sparse_router.router_semantic_dtype, DType::Fp32);
        assert_eq!(cfg.shared_expert.dtype, DType::Bf16);
        assert_eq!(cfg.shared_expert.gemm_dtype, DType::Bf16);
        assert_eq!(cfg.final_norm.dtype, DType::Bf16);
        assert_eq!(cfg.lm_head.dtype, DType::Bf16);
        assert_eq!(cfg.lm_head.backends, vec!["torch_linear"]);
        assert_eq!(cfg.moe_dispatch.placement, Placement::RoundRobin);
        assert_eq!(cfg.moe_dispatch.intra_fabric, Fabric::Nvlink);
        assert_eq!(cfg.moe_dispatch.inter_fabric, Fabric::Infiniband);
        assert_eq!(cfg.embedding.input_bytes_per_token, 12_296);
        assert_eq!(cfg.embedding.output_bytes_per_token, 12_288);
        assert_eq!(cfg.sparse_finalization.input_bytes_per_token, 110_592);
        assert_eq!(cfg.sparse_finalization.output_bytes_per_token, 12_288);
        assert_eq!(cfg.lm_head.n, VOCAB_SIZE);
        assert_eq!(cfg.lm_head.k, HIDDEN_DIM);
        assert!(cfg.mtp_prelude.is_some());
        assert!(!cfg.mtp_attention.as_ref().unwrap().include_indexer);
    }

    #[test]
    fn fp8_selects_generic_glm_gemms_and_network_without_spilling_into_holdouts() {
        let cfg = build_configs(
            &model(),
            &parallel(8),
            &RoutingDistribution::uniform(NUM_EXPERTS),
            true,
            Glm52MtpMode::IndexShare,
        )
        .unwrap();

        assert_eq!(cfg.moe_expert_compute[0].dtype, DType::Fp8E4m3);
        assert_eq!(cfg.moe_expert_compute[0].activation_dtype, DType::Bf16);
        assert_eq!(cfg.moe_expert_compute[0].grouped_gemm_backends, vec!["deepgemm"]);
        assert_eq!(
            cfg.moe_expert_compute[0].fp8_grouped_gemm_backends,
            vec!["flashinfer_trtllm"]
        );
        assert_eq!(cfg.moe_dispatch.backends, vec!["nvshmem"]);
        assert_eq!(cfg.moe_combine.backends, vec!["nvshmem"]);
        assert_eq!(cfg.moe_dispatch.dtype, DType::Fp8E4m3);
        assert_eq!(cfg.moe_combine.dtype, DType::Fp8E4m3);
        assert_eq!(cfg.moe_dispatch.net_params().hidden_bytes, HIDDEN_DIM);
        assert_eq!(cfg.moe_combine.net_params().hidden_bytes, HIDDEN_DIM);

        for attention in [
            &cfg.dense_full_index_attention,
            &cfg.initial_shared_attention,
            &cfg.cycle_full_attention,
            &cfg.cycle_shared_attention,
            cfg.mtp_attention.as_ref().unwrap(),
        ] {
            assert_eq!(attention.single_gemm_backends, vec!["deepgemm"]);
            assert_eq!(attention.indexer_gemm_backends, vec!["deepgemm"]);
            assert_eq!(attention.gemm_dtype, DType::Fp8E4m3);
            assert_eq!(attention.base_dtype, DType::Bf16);
            assert_eq!(attention.q_absorb_backends, vec!["torch_mla_q_absorb_glm52"]);
            assert_eq!(attention.v_up_backends, vec!["torch_mla_v_up_glm52"]);
        }

        let resolved = resolve_configs(&cfg);
        assert_eq!(
            resolved.moe_expert_compute[0].gate_up_fp8.as_ref().unwrap().gemm.dtype,
            DType::Fp8E4m3
        );
        assert_eq!(
            resolved.moe_expert_compute[0].down_fp8.as_ref().unwrap().gemm.dtype,
            DType::Fp8E4m3
        );
        assert_eq!(resolved.moe_expert_compute[0].act.input_bytes_per_token, 2 * MOE_INTERMEDIATE_DIM * 2);
        assert_eq!(resolved.dense_ffn.gate_up_proj.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.dense_ffn.down_proj.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.dense_ffn.post_attn_add_rms_norm.dtype, DType::Bf16);
        assert_eq!(resolved.dense_ffn.silu_and_mul.input_bytes_per_token, 2 * DENSE_INTERMEDIATE_DIM * 2);
        assert_eq!(resolved.sparse_router.router_gemm_bf16_proxy.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.sparse_router.post_attn_add_rms_norm.dtype, DType::Bf16);
        assert_eq!(resolved.sparse_router.raw_cfg.router_semantic_dtype, DType::Fp32);
        assert_eq!(resolved.shared_expert.gate_up_proj.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.shared_expert.down_proj.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.shared_expert.silu_and_mul.input_bytes_per_token, 2 * MOE_INTERMEDIATE_DIM * 2);
        assert_eq!(resolved.lm_head.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.lm_head.backends, vec!["deepgemm"]);
        assert_eq!(resolved.dense_full_index_attention.fused_qkv_a_proj.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.dense_full_index_attention.q_b_proj.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.dense_full_index_attention.o_proj.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.dense_full_index_attention.q_absorb.dtype, DType::Bf16);
        assert_eq!(resolved.dense_full_index_attention.v_up.dtype, DType::Bf16);
        let indexer = resolved.dense_full_index_attention.indexer.as_ref().unwrap();
        assert_eq!(indexer.input_dtype, DType::Bf16);
        assert_eq!(indexer.gemm_dtype, DType::Fp8E4m3);
        assert_eq!(indexer.cache_dtype, DType::Fp8E4m3);
        assert_eq!(indexer.q_dtype, DType::Fp8E4m3);

        // Elementwise, norm, cache, and semantic holdouts stay on their
        // independent BF16/FP32 contracts even when their neighboring GEMM is
        // FP8.
        assert_eq!(cfg.dense_full_index_attention.base_dtype, DType::Bf16);
        assert_eq!(cfg.dense_ffn.dtype, DType::Bf16);
        assert_eq!(cfg.sparse_router.base_dtype, DType::Bf16);
        assert_eq!(cfg.sparse_router.router_semantic_dtype, DType::Fp32);
        assert_eq!(cfg.shared_expert.dtype, DType::Bf16);
        assert_eq!(cfg.final_norm.dtype, DType::Bf16);
        assert_eq!(cfg.mtp_prelude.as_ref().unwrap().dtype, DType::Bf16);
        assert_eq!(cfg.mtp_prelude.as_ref().unwrap().gemm_dtype, DType::Fp8E4m3);
        assert_eq!(
            cfg.mtp_attention.as_ref().unwrap().base_dtype,
            DType::Bf16
        );
        assert_eq!(cfg.mtp_head.as_ref().unwrap().dtype, DType::Bf16);
        assert_eq!(cfg.mtp_head.as_ref().unwrap().gemm_dtype, DType::Fp8E4m3);
        assert_eq!(resolved.mtp_prelude.as_ref().unwrap().eh_proj.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.mtp_prelude.as_ref().unwrap().embedding_rms_norm.dtype, DType::Bf16);
        assert_eq!(resolved.mtp_head.as_ref().unwrap().lm_head.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.mtp_head.as_ref().unwrap().shared_head_rms_norm.dtype, DType::Bf16);
    }

    #[test]
    fn parallel_and_routing_validation_is_explicit() {
        let routing = RoutingDistribution::uniform(NUM_EXPERTS);
        for p in [
            Glm52VllmDsaMoeParallel {
                ep_size: 0,
                nvl_num_gpu: 1,
                gpu_name: "NVIDIA H200".into(),
            },
            Glm52VllmDsaMoeParallel {
                ep_size: 7,
                nvl_num_gpu: 1,
                gpu_name: "NVIDIA H200".into(),
            },
            Glm52VllmDsaMoeParallel {
                ep_size: 8,
                nvl_num_gpu: 3,
                gpu_name: "NVIDIA H200".into(),
            },
        ] {
            assert!(build_configs(&model(), &p, &routing, false, Glm52MtpMode::Off).is_err());
        }
        assert!(build_configs(
            &model(),
            &parallel(8),
            &RoutingDistribution::uniform(128),
            false,
            Glm52MtpMode::Off
        )
        .is_err());
    }

    #[test]
    fn resolve_configs_preserves_full_vs_shared_and_h32_h64_surrogate() {
        let cfg = build_configs(
            &model(),
            &parallel(8),
            &RoutingDistribution::uniform(NUM_EXPERTS),
            false,
            Glm52MtpMode::FullIndex,
        )
        .unwrap();
        let resolved = resolve_configs(&cfg);
        assert!(resolved.dense_full_index_attention.indexer.is_some());
        assert!(resolved.initial_shared_attention.indexer.is_none());
        let indexer = resolved
            .dense_full_index_attention
            .indexer
            .as_ref()
            .unwrap();
        assert_eq!(indexer.model_num_index_heads, 32);
        assert_eq!(indexer.profile_num_index_heads, 64);
        assert_eq!(resolved.dense_ffn.gate_up_proj.n, 24_576);
        assert_eq!(
            resolved.sparse_router.raw_cfg.router_semantic_dtype,
            DType::Fp32
        );
        assert_eq!(
            resolved.sparse_router.router_gemm_bf16_proxy.dtype,
            DType::Bf16
        );
        assert!(resolved.mtp_attention.as_ref().unwrap().indexer.is_some());
    }

    /// The falsifiable acceptance condition from
    /// `doc/alignment/glm52_dp8_ep8_report.md` section 4.3: every measured
    /// `scale_1x128_kernel<bf16, fp8_e4m3, float>` launch has exactly one
    /// simulated quant leaf, and none is invented. 411 launches/iteration were
    /// decoded there by their successor GEMM's (N, K).
    #[test]
    fn fp8_quant_leaves_reproduce_the_measured_launch_census() {
        // Per layer variant: attention 3 (fused_qkv_a_proj, q_b_proj, o_proj)
        // + FFN 2 (gate_up, down) + 1 more where an indexer runs (its q_proj).
        let dense = 3 + 1 + 2; // dense layer: indexer + dense FFN
        let sparse_full = 3 + 1 + 2; // full-index sparse: indexer + shared expert
        let sparse_share = 3 + 2; // IndexShare sparse: shared expert only
        let measured = NUM_DENSE_LAYERS as usize * dense
            + FULL_INDEX_LAYERS
                .iter()
                .filter(|layer| **layer >= NUM_DENSE_LAYERS)
                .count()
                * sparse_full
            + (NUM_LAYERS as usize
                - NUM_DENSE_LAYERS as usize
                - FULL_INDEX_LAYERS
                    .iter()
                    .filter(|layer| **layer >= NUM_DENSE_LAYERS)
                    .count())
                * sparse_share;
        assert_eq!(measured, 411, "measured quant-launch census per iteration");

        // Read the same census off the slot formula. The FP8-minus-BF16 delta
        // is every quant leaf in the graph, which splits exactly as the two
        // measured kernel templates do:
        //   22 per EP rank -- this graph's new dense-GEMM quants (the 411
        //      `scale_1x128<bf16, fp8_e4m3, float>` launches);
        //    6 per EP rank -- the routed grouped-GEMM quants the native graph
        //      already models (the 150 `scale_1x128<(bool)0, ...>` launches),
        //      contributed by `expert_slots` in the three sparse variants.
        let dense_gemm_quants_per_rank = dense + sparse_full + sparse_share + sparse_share;
        let routed_quants_per_rank = 3 * 2;
        assert_eq!(dense_gemm_quants_per_rank, 22);
        for ep in [1_u16, 2, 4, 8] {
            let delta = expected_slot_count(ep, true, Glm52MtpMode::Off)
                - expected_slot_count(ep, false, Glm52MtpMode::Off);
            assert_eq!(
                delta,
                (dense_gemm_quants_per_rank + routed_quants_per_rank) * usize::from(ep)
            );
        }
    }

    #[test]
    fn schedule_and_slot_formulas_match_the_composed_leaf_contracts() {
        assert_eq!(3 + 3 + 18 * (1 + 3), 78);
        assert_eq!(FULL_INDEX_LAYERS.len(), 21);
        assert_eq!(78 - FULL_INDEX_LAYERS.len(), 57);
        // BF16 counts match the native graph exactly: with no FP8 GEMM there
        // is nothing to quantise, so this graph adds no leaves.
        assert_eq!(expected_slot_count(8, false, Glm52MtpMode::Off), 1_026);
        assert_eq!(expected_slot_count(8, false, Glm52MtpMode::FullIndex), 1_424);
        assert_eq!(expected_slot_count(8, false, Glm52MtpMode::IndexShare), 1_304);
        // FP8 adds the vLLM activation-quantisation leaves: five per layer
        // variant (attention 3 + FFN 2), plus a sixth on the two variants that
        // run an indexer (its q_proj). That is 22 x ep = 176 more than the
        // native graph's 1_074. See `doc/alignment/glm52_dp8_ep8_report.md`
        // section 4.3, which decodes each measured launch by its GEMM shape.
        assert_eq!(expected_slot_count(8, true, Glm52MtpMode::Off), 1_074 + 176);
        assert_eq!(expected_slot_count(8, true, Glm52MtpMode::FullIndex), 1_488 + 224);
        assert_eq!(expected_slot_count(8, true, Glm52MtpMode::IndexShare), 1_368 + 216);
        for ep in [1_u16, 2, 4, 8, 16] {
            let ep = usize::from(ep);
            assert_eq!(expected_slot_count(ep as u16, false, Glm52MtpMode::Off), 126 * ep + 18);
            assert_eq!(expected_slot_count(ep as u16, true, Glm52MtpMode::Off), 154 * ep + 18);
            assert_eq!(
                expected_slot_count(ep as u16, false, Glm52MtpMode::FullIndex),
                175 * ep + 24
            );
            assert_eq!(
                expected_slot_count(ep as u16, true, Glm52MtpMode::FullIndex),
                211 * ep + 24
            );
            assert_eq!(
                expected_slot_count(ep as u16, false, Glm52MtpMode::IndexShare),
                160 * ep + 24
            );
            assert_eq!(
                expected_slot_count(ep as u16, true, Glm52MtpMode::IndexShare),
                195 * ep + 24
            );
        }
    }

    fn max_label_inventory(mtp_mode: Glm52MtpMode) -> Vec<String> {
        let mut labels = vec![
            "unified.main.embedding [Max over attention-DP groups]".to_string(),
            "unified.body.dense_full_index.attention [Max over attention-DP groups]".to_string(),
            "unified.body.dense_full_index.ffn [Max over attention-DP/home groups]".to_string(),
        ];
        for body in [
            "unified.body.sparse_initial_index_share",
            "unified.body.sparse_cycle_full_index",
            "unified.body.sparse_cycle_index_share",
        ] {
            labels.extend([
                format!("{body}.attention [Max over attention-DP groups]"),
                format!("{body}.moe.router [Max over home groups]"),
                format!("{body}.moe.routed_experts [Max over EP ranks]"),
                format!("{body}.moe.shared_expert [Max over home groups]"),
                format!("{body}.moe.finalization [Max over home groups]"),
            ]);
        }
        labels.extend([
            "unified.main.final_residual_rms_norm [Max over attention-DP/home groups]".to_string(),
            "unified.main.lm_head [Max over attention-DP/home groups]".to_string(),
        ]);
        if mtp_mode != Glm52MtpMode::Off {
            labels.push("unified.mtp.prelude [Max over attention-DP groups]".to_string());
            let body = "unified.mtp.decoder_sparse";
            labels.extend([
                format!("{body}.attention [Max over attention-DP groups]"),
                format!("{body}.moe.router [Max over home groups]"),
                format!("{body}.moe.routed_experts [Max over EP ranks]"),
                format!("{body}.moe.shared_expert [Max over home groups]"),
                format!("{body}.moe.finalization [Max over home groups]"),
            ]);
            labels.push("unified.mtp.head [Max over attention-DP/home groups]".to_string());
        }
        labels
    }

    fn compile_max_label_inventory(mtp_mode: Glm52MtpMode) -> CostManifest {
        let mut builder = CostTreeBuilder::new();
        let nodes = max_label_inventory(mtp_mode)
            .into_iter()
            .map(|label| {
                let leaf = builder.leaf(label.clone(), "test", serde_json::json!({}));
                labeled_max(label, vec![leaf])
            })
            .collect();
        builder.finish(CostNode::Sum(nodes)).manifest()
    }

    #[test]
    fn every_l4_max_has_a_unique_semantic_manifest_label_in_every_mtp_mode() {
        for (mode, expected_slots, expected_maxes) in [
            (Glm52MtpMode::Off, 1_026, 20),
            (Glm52MtpMode::FullIndex, 1_424, 27),
            (Glm52MtpMode::IndexShare, 1_304, 27),
        ] {
            let manifest = compile_max_label_inventory(mode);
            let max_labels: Vec<&str> = manifest
                .nodes
                .iter()
                .zip(&manifest.node_labels)
                .filter_map(|(node, label)| {
                    matches!(node, FlatCostNode::Max { .. }).then(|| {
                        label
                            .as_deref()
                            .expect("every GLM L4 Max must have a manifest label")
                    })
                })
                .collect();
            assert_eq!(expected_slot_count(8, false, mode), expected_slots);
            assert_eq!(max_labels.len(), expected_maxes);
            assert!(max_labels.iter().all(|label| label.contains("[Max over ")));
            let unique: HashSet<&str> = max_labels.iter().copied().collect();
            assert_eq!(unique.len(), max_labels.len(), "Max labels must be unique");

            for expected in [
                "unified.main.embedding [Max over attention-DP groups]",
                "unified.body.dense_full_index.attention [Max over attention-DP groups]",
                "unified.body.dense_full_index.ffn [Max over attention-DP/home groups]",
                "unified.body.sparse_initial_index_share.attention [Max over attention-DP groups]",
                "unified.body.sparse_cycle_full_index.moe.router [Max over home groups]",
                "unified.body.sparse_cycle_full_index.moe.routed_experts [Max over EP ranks]",
                "unified.body.sparse_cycle_index_share.moe.shared_expert [Max over home groups]",
                "unified.body.sparse_cycle_index_share.moe.finalization [Max over home groups]",
                "unified.main.final_residual_rms_norm [Max over attention-DP/home groups]",
                "unified.main.lm_head [Max over attention-DP/home groups]",
            ] {
                assert!(
                    max_labels.contains(&expected),
                    "missing Max label {expected}"
                );
            }
            if mode == Glm52MtpMode::Off {
                assert!(max_labels.iter().all(|label| !label.contains(".mtp.")));
            } else {
                for expected in [
                    "unified.mtp.prelude [Max over attention-DP groups]",
                    "unified.mtp.decoder_sparse.attention [Max over attention-DP groups]",
                    "unified.mtp.decoder_sparse.moe.router [Max over home groups]",
                    "unified.mtp.decoder_sparse.moe.routed_experts [Max over EP ranks]",
                    "unified.mtp.decoder_sparse.moe.shared_expert [Max over home groups]",
                    "unified.mtp.decoder_sparse.moe.finalization [Max over home groups]",
                    "unified.mtp.head [Max over attention-DP/home groups]",
                ] {
                    assert!(
                        max_labels.contains(&expected),
                        "missing Max label {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn production_l4_uses_only_the_labeled_max_constructor() {
        let production = include_str!("glm52_dsa_moe.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production source precedes tests");
        assert_eq!(
            production.matches("CostNode::Max {").count(),
            1,
            "the only production Max constructor must be inside labeled_max"
        );
        assert!(!production.contains("max_node("));
    }

    #[test]
    fn labeled_max_is_flattening_and_cost_transparent() {
        fn tree(labeled: bool) -> CostTree {
            let mut builder = CostTreeBuilder::new();
            let max_children = vec![
                builder.leaf("a", "test", serde_json::json!({})),
                builder.leaf("b", "test", serde_json::json!({})),
            ];
            let max = if labeled {
                labeled_max("semantic [Max over groups]".to_string(), max_children)
            } else {
                CostNode::Max {
                    overlap: 1.0,
                    children: max_children,
                }
            };
            let serial = CostNode::Sum(vec![builder.leaf("c", "test", serde_json::json!({}))]);
            builder.finish(CostNode::Sum(vec![max, serial]))
        }

        let plain = tree(false);
        let labeled = tree(true);
        assert_eq!(labeled.n_slots(), plain.n_slots());
        let plain_flat = plain.flatten();
        let labeled_flat = labeled.flatten();
        assert_eq!(labeled_flat.len(), plain_flat.len());
        for predicate in [
            |node: &FlatCostNode| matches!(node, FlatCostNode::Leaf(_)),
            |node: &FlatCostNode| matches!(node, FlatCostNode::Sum { .. }),
            |node: &FlatCostNode| matches!(node, FlatCostNode::Max { .. }),
            |node: &FlatCostNode| matches!(node, FlatCostNode::Scale { .. }),
        ] {
            assert_eq!(
                labeled_flat.iter().filter(|node| predicate(node)).count(),
                plain_flat.iter().filter(|node| predicate(node)).count()
            );
        }
        let mut slots = vec![LeafMetrics::ZERO; 3];
        slots[0].m.time_ms = 2.0;
        slots[1].m.time_ms = 3.0;
        slots[2].m.time_ms = 5.0;
        let mut plain_scratch = Vec::new();
        let mut labeled_scratch = Vec::new();
        let labeled_total = CostTree::aggregate(&labeled_flat, &slots, &mut labeled_scratch);
        let plain_total = CostTree::aggregate(&plain_flat, &slots, &mut plain_scratch);
        assert_eq!(labeled_total.m.time_ms, plain_total.m.time_ms);
        assert_eq!(labeled_total.m.flops, plain_total.m.flops);
        assert_eq!(labeled_total.m.bytes, plain_total.m.bytes);
        assert_eq!(labeled_total.m.energy_j, plain_total.m.energy_j);
        assert_eq!(labeled_total.coverage, plain_total.coverage);
        assert_eq!(labeled_total.backend_index, plain_total.backend_index);
        assert_eq!(
            labeled
                .manifest()
                .node_labels
                .iter()
                .flatten()
                .next()
                .map(String::as_str),
            Some("semantic [Max over groups]")
        );
    }

    #[test]
    fn state_and_topology_contracts_are_exact() {
        assert_eq!(state_bytes_per_token(Glm52MtpMode::Off).unwrap(), 92_628);
        assert_eq!(
            state_bytes_per_token(Glm52MtpMode::FullIndex).unwrap(),
            93_912
        );
        assert_eq!(
            state_bytes_per_token(Glm52MtpMode::IndexShare).unwrap(),
            93_780
        );
        assert_eq!(u32::from(parallel(8).ep_size), 8);
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
    fn input_lowering_handles_prefill_decode_mixed_and_empty_groups() {
        let input = UnifiedArchInput {
            groups: vec![
                group(5, vec![64, 91], vec![(3, 2), (0, 3)]),
                ArchGroupInput::default(),
            ],
            tokens_per_source_rank: Vec::new(),
        };
        let normalized = normalize_input(&input, 2).unwrap();
        assert_eq!(normalized.total_tokens, 7);
        assert_eq!(normalized.routed_selections, 56);
        assert_eq!(
            normalized.groups[0]
                .attention_input
                .prefill_query_cache_pairs,
            vec![(2, 5), (3, 3)]
        );
        assert_eq!(
            normalized.groups[0]
                .attention_input
                .decode
                .as_ref()
                .unwrap()
                .batch_size,
            2
        );
        assert_eq!(
            normalized.groups[0]
                .attention_input
                .decode
                .as_ref()
                .unwrap()
                .context_len,
            91
        );
        assert!(
            !normalized.groups[0]
                .attention_input
                .decode
                .as_ref()
                .unwrap()
                .requires_padding
        );
        assert_eq!(normalized.groups[0].request_count, 4);
        assert_eq!(normalized.groups[1].batch_tokens, 0);
        let mtp = normalized.decode_only();
        assert_eq!(mtp.total_tokens, 2);
        assert!(mtp.groups[0]
            .attention_input
            .prefill_query_cache_pairs
            .is_empty());
    }

    #[test]
    fn input_validation_rejects_group_count_counts_context_and_overflow() {
        let empty = UnifiedArchInput {
            groups: vec![ArchGroupInput::default(); 2],
            tokens_per_source_rank: Vec::new(),
        };
        assert!(normalize_input(&empty, 2).is_ok());
        assert!(normalize_input(&empty, 1).is_err());

        let bad_prefill = UnifiedArchInput {
            groups: vec![group(1, vec![], vec![(0, 2)])],
            tokens_per_source_rank: Vec::new(),
        };
        assert!(normalize_input(&bad_prefill, 1).is_err());
        let bad_decode = UnifiedArchInput {
            groups: vec![group(0, vec![0], vec![])],
            tokens_per_source_rank: Vec::new(),
        };
        assert!(normalize_input(&bad_decode, 1).is_err());
        let too_long = UnifiedArchInput {
            groups: vec![group(0, vec![TIMING_MAX_MODEL_LEN + 1], vec![])],
            tokens_per_source_rank: Vec::new(),
        };
        assert!(normalize_input(&too_long, 1).is_err());
        let overflow = UnifiedArchInput {
            groups: vec![ArchGroupInput {
                batch_tokens: u32::MAX,
                prefill_tokens: u32::MAX,
                decode_tokens: 1,
                prefill_chunk_pairs: vec![(0, u32::MAX)],
                decode_kv_lens: vec![1],
                total_kv_len: 0,
            }],
            tokens_per_source_rank: Vec::new(),
        };
        assert!(normalize_input(&overflow, 1).is_err());
    }

    #[test]
    fn architecture_has_no_attention_tp_sampler_or_hidden_indexshare_copy() {
        let cfg = build_configs(
            &model(),
            &parallel(8),
            &RoutingDistribution::uniform(NUM_EXPERTS),
            false,
            Glm52MtpMode::Off,
        )
        .unwrap();
        assert_eq!(cfg.parallel.ep_size, 8);
        assert_eq!(cfg.dense_full_index_attention.num_attention_heads, 64);
        assert_eq!(cfg.dense_full_index_attention.num_kv_heads, 1);
        assert_eq!(cfg.moe_dispatch.placement, Placement::RoundRobin);
        for forbidden in [
            "all_reduce",
            "sampler",
            "logits_processing",
            "indexshare_copy",
        ] {
            assert!(!format!("{cfg:?}").contains(forbidden));
        }
    }
}
