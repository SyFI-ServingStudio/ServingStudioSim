//! `glm52_vllm_dsa_moe` — GLM-5.2 iter-wise architecture in **vLLM kernel
//! granularity**, for VibeSim-vs-vLLM alignment.
//!
//! Structurally identical to [`super::glm52_dsa_moe`]: attention is TP1 and
//! independently replicated over the EP ranks; decoder layers 0--2 execute the
//! dense FFN and a full DSA indexer; layers 3--5 reuse the layer-2 index; layers
//! 6--77 repeat a four-layer cadence containing one full-index layer and three
//! `IndexShare` layers; shared-expert compute is conservatively serialized with
//! routed-expert work.
//!
//! Sparse-MoE communication is pure EP and is priced by the **profiled**
//! flashinfer MNNVL all-to-all (`moe_alltoall`, plus `moe_alltoall_prepare` for
//! the metadata pass), not by the simulated `p2p_intra`/`p2p_inter` byte model
//! the native graph uses. That buys the real kernel's fan-out and contention.
//!
//! The transfer leaf is keyed by two row counts, `(max_send_rows,
//! max_recv_rows)`, so both imbalances reach it: how the tokens are spread over
//! DP groups sets what leaves the busiest sender, and how expert popularity is
//! spread over EP ranks sets what arrives at the busiest receiver. `local_ppm`
//! therefore moves the transfer as well as the routed grouped GEMMs and
//! `finalizeMoeRoutingKernel`. What is still priced flat is `moe_alltoall_prepare`,
//! whose key is the token count alone; a uniformly-drawn benchmark under-reads a
//! real skewed layer there by ~9%, which is ~3% of the `MoE` communication budget.
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

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::glm52_dsa_moe::{Glm52ModelCfg, Glm52MtpMode};
use crate::common::Fabric;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, GroupedGemmKernelInput,
    MoeAlltoallDirection, MoeAlltoallKernel, MoeAlltoallKernelConfig, MoeAlltoallKernelInput,
    MoeAlltoallPrepareKernel, MoeAlltoallPrepareKernelConfig, MoeAlltoallPrepareKernelInput,
    MoeFinalizeRoutingKernel, MoeFinalizeRoutingKernelConfig, MoeFinalizeRoutingKernelInput,
    ResidualRmsNormKernel, ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput,
    SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::routing::RoutingDistribution;
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Dim, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, Probe, SlotInput,
};
use crate::worklet::{
    Glm52MoeRouterLocalWorklet, Glm52MoeRouterLocalWorkletConfig, Glm52MoeRouterLocalWorkletInput,
    Glm52MoeRouterLocalWorkletResolved, Glm52MtpHeadLocalWorklet, Glm52MtpHeadLocalWorkletConfig,
    Glm52MtpHeadLocalWorkletInput, Glm52MtpHeadLocalWorkletResolved, Glm52MtpPreludeLocalWorklet,
    Glm52MtpPreludeLocalWorkletConfig, Glm52MtpPreludeLocalWorkletInput,
    Glm52MtpPreludeLocalWorkletResolved, MoeExpertComputeLocalWorklet,
    MoeExpertComputeLocalWorkletConfig, MoeExpertComputeLocalWorkletInput,
    MoeExpertComputeLocalWorkletResolved, VllmGlm52DenseFfnLocalWorklet,
    VllmGlm52DenseFfnLocalWorkletConfig, VllmGlm52DenseFfnLocalWorkletInput,
    VllmGlm52DenseFfnLocalWorkletResolved, VllmGlm52DsaAttnLocalDecodeInput,
    VllmGlm52DsaAttnLocalWorklet, VllmGlm52DsaAttnLocalWorkletConfig,
    VllmGlm52DsaAttnLocalWorkletInput, VllmGlm52DsaAttnLocalWorkletResolved,
    VllmGlm52SharedExpertLocalWorklet, VllmGlm52SharedExpertLocalWorkletConfig,
    VllmGlm52SharedExpertLocalWorkletInput, VllmGlm52SharedExpertLocalWorkletResolved,
};

const ARCH_KIND: &str = "glm52_vllm_dsa_moe";
/// Accepted L1 timing domain, in tokens. The checkpoint advertises 1,048,576 and
/// the profiled DSA grids now reach it, so a full-context coding-agent session
/// replays without an extrapolated attention slot. It is a hard bound, not a
/// hint: a request past it bails rather than silently reading off the end of the
/// measured surface.
const TIMING_MAX_MODEL_LEN: u32 = 1_048_576;
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
/// Padded logits row width, mirroring production's `max_model_len`-sized
/// allocation. It tracks [`TIMING_MAX_MODEL_LEN`] and is part of the topk
/// kernels' cache identity, so changing either invalidates their profiled rows.
const LOGITS_ROW_STRIDE: u32 = 1_048_576;

const FULL_INDEX_LAYERS: [u32; 21] = [
    0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74,
];

const RESIDUAL_NORM_BACKENDS: &[&str] = &["vllm_cuda"];
const RMS_NORM_BACKENDS: &[&str] = &["flashinfer"];
const SINGLE_GEMM_BACKENDS: &[&str] = &["torch_linear"];
const ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
// The MLA query RoPE is not a streaming pointwise leaf: vLLM feeds the roped
// q straight back into the attention op, so inductor fuses it into a kernel
// that rewrites all of q. Its own L1 kind measures that fusion.
const MAIN_ROPE_BACKENDS: &[&str] = &["vllm_inductor"];
const GROUPED_GEMM_BACKENDS: &[&str] = &["torch"];
const FP8_SINGLE_GEMM_BACKENDS: &[&str] = &["deepgemm"];
// vLLM quantises dense and routed activations with two different kernels.
// Dense linears call its own `per_token_group_quant_8bit_kernel` (backend
// `vllm_cuda`); only the routed grouped GEMM reaches TensorRT-LLM's
// `scale_1x128_kernel` (backend `flashinfer_trtllm`). Both appear in the same
// nsys iteration, so this is not a shape or a backend preference -- feeding the
// dense leaves the routed curve over-predicted them ~2.05x at prefill.
const DENSE_FP8_QUANT_BACKENDS: &[&str] = &["vllm_cuda"];
const ROUTED_FP8_QUANT_BACKENDS: &[&str] = &["flashinfer_trtllm"];
const Q_ABSORB_BACKENDS: &[&str] = &["torch_mla_q_absorb_glm52"];
const V_UP_BACKENDS: &[&str] = &["torch_mla_v_up_glm52"];
const INDEX_CACHE_AND_TOPK_BACKENDS: &[&str] = &["vllm_cuda"];
const INDEX_LOGITS_BACKENDS: &[&str] = &["vllm_deepgemm_fp8"];
const SPARSE_ATTN_BACKENDS: &[&str] = &["vllm_flashmla_bf16"];
const MLA_APPEND_BACKENDS: &[&str] = &["vllm_cuda"];
// The MoE transfer does not vary with the expert dtype: vLLM defers activation
// quantisation past the all-to-all, so both the bf16 and the fp8 deployment send
// bf16 rows through the same flashinfer MNNVL kernel.
const MOE_ALLTOALL_BACKENDS: &[&str] = &["flashinfer_mnnvl"];
const FP8_GROUPED_GEMM_BACKENDS: &[&str] = &["deepgemm"];
const FP8_PRODUCTION_GROUPED_GEMM_BACKENDS: &[&str] = &["flashinfer_trtllm"];

// Leaf counts are properties of the accepted L2/L3 sections. `build` verifies
// the compiled tree against these formulas, so drift cannot be hidden by a
// stale handwritten expectation.
const ATTN_FULL_SLOTS: usize = 29;
const ATTN_SHARED_SLOTS: usize = 14;
const DENSE_FFN_SLOTS: usize = 4;
const ROUTER_SLOTS: usize = 4;
/// `moe.dispatch.prepare` + `moe.dispatch`. Both are collectives over the whole
/// EP group, so they are one leaf each, not one per rank.
const DISPATCH_SLOTS: usize = 2;
/// `expandInputRows`, one per EP rank: the gather is rank-local work whose size
/// follows that rank's popularity shard.
const EXPAND_INPUT_ROWS_SLOTS: usize = 1;
const EXPERT_SLOTS: usize = 3;
/// The zero fill, the profiled transfer, and the top-k reduction. Three leaves
/// because the wrapper is three launches over two different quantities — see
/// the `moe_combine_output_fill` field's doc comment.
const COMBINE_SLOTS: usize = 3;
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
    /// `mnnvl_moe_alltoallv_prepare_without_allgather`: five index kernels that
    /// decide, from the router's expert ids alone, which rows go to which rank.
    /// A profiled kind of its own because it is atomics- and exchange-bound, not
    /// bandwidth-bound — 0.33 ms/layer to move a quarter of a megabyte, which
    /// any byte-rate model would price at roughly zero.
    pub moe_dispatch_prepare: MoeAlltoallPrepareKernelConfig,
    pub moe_dispatch: MoeAlltoallKernelConfig,
    /// One config per EP rank. The rank axis is what makes the routed-expert
    /// grouped GEMMs distribution-sensitive: each rank owns its own
    /// `local_ppm` shard, so a measured or synthetic skew yields a distinct
    /// grouped-GEMM cache identity per rank and the `Max` over ranks lands on
    /// the genuinely heaviest one. A uniform distribution still produces
    /// `ep_size` identical entries, so the slot layout never changes.
    pub moe_expert_compute: Vec<MoeExpertComputeLocalWorkletConfig>,
    /// `expandInputRows`: the receiving rank gathers the rows it was sent into
    /// one contiguous block per local expert, so the grouped GEMM sees a dense
    /// batch. A pure bandwidth mover (read a row, write a row) — the byte-rate
    /// elementwise curve is the right primitive, and at 8k prefill tokens this
    /// is 42 ms/iteration that had no slot at all.
    pub moe_expand_input_rows: ElementwiseKernelConfig,
    /// `mnnvl_moe_alltoallv_combine` zeroes its `token_count x top_k` output
    /// before the transfer writes into it — the trace's `FillFunctor` launch,
    /// 805 MB per layer at 8k tokens, comparable to the transfer beside it.
    /// It is its own leaf because it scales with the token count while the
    /// transfer scales with rows; folding it into the profiled call would have
    /// forced a third axis onto that kernel's cache.
    pub moe_combine_output_fill: ElementwiseKernelConfig,
    /// The return leg proper: `moe_comm`, and nothing else in the wrapper.
    pub moe_combine: MoeAlltoallKernelConfig,
    /// The `torch.sum` that folds the `token_count x top_k` staging buffer back
    /// down to one row per token. Same reasoning as the fill.
    pub moe_combine_reduce: ElementwiseKernelConfig,
    pub shared_expert: VllmGlm52SharedExpertLocalWorkletConfig,
    /// One config per EP rank, sharing the routed experts' popularity shards:
    /// `finalizeMoeRoutingKernel` runs where the expert GEMMs ran and reduces
    /// exactly the rows they produced, so a skewed rank finalizes more.
    pub sparse_finalization: Vec<MoeFinalizeRoutingKernelConfig>,
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
    pub moe_dispatch_prepare: MoeAlltoallPrepareKernelConfig,
    pub moe_dispatch: MoeAlltoallKernelConfig,
    pub moe_expert_compute: Vec<MoeExpertComputeLocalWorkletResolved>,
    pub moe_expand_input_rows: ElementwiseKernelConfig,
    pub moe_combine_output_fill: ElementwiseKernelConfig,
    pub moe_combine: MoeAlltoallKernelConfig,
    pub moe_combine_reduce: ElementwiseKernelConfig,
    pub shared_expert: VllmGlm52SharedExpertLocalWorkletResolved,
    pub sparse_finalization: Vec<MoeFinalizeRoutingKernelConfig>,
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
        fp8_quant_backends: DENSE_FP8_QUANT_BACKENDS.to_vec(),
        include_indexer,
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
        rope_max_position: model.max_context.clone(),
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
        rope_is_neox_style: false,
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
    if !NUM_EXPERTS.is_multiple_of(u32::from(parallel.ep_size)) {
        return Err(fit_failed(format!(
            "num_experts {NUM_EXPERTS} must be divisible by ep_size {}",
            parallel.ep_size
        )));
    }
    if parallel.nvl_num_gpu == 0
        || parallel.nvl_num_gpu > parallel.ep_size
        || !parallel.ep_size.is_multiple_of(parallel.nvl_num_gpu)
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
    let (expert_dtype, expert_backends) = if fp8 {
        (DType::Fp8E4m3, FP8_GROUPED_GEMM_BACKENDS)
    } else {
        (DType::Bf16, GROUPED_GEMM_BACKENDS)
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
    let hidden_bytes = HIDDEN_DIM
        .checked_mul(DType::Bf16.size_bytes())
        .ok_or_else(|| fit_failed("hidden byte width overflows u32"))?;
    // The all-to-all moves activations, not packed expert weights, and vLLM
    // sends them UNQUANTISED: the flashinfer two-sided path takes the
    // `defer_input_quant` branch for a block-scale fp8 MoE, so the wire payload
    // is bf16 and `scale_1x128_kernel` runs on the receiving rank after
    // `expandInputRows`. The measured launch order says the same thing
    // (alltoall -> expand -> quant -> fp8 gemm). Pinning this to the expert
    // dtype halved every dispatch/combine byte count. Same rule, same reason,
    // as `qwen3_vllm_moe_dp_attn_ep_ffn`.
    let moe_alltoall = |direction| MoeAlltoallKernelConfig {
        backends: MOE_ALLTOALL_BACKENDS.to_vec(),
        gpu_name: gpu.clone(),
        ep_size: u32::from(parallel.ep_size),
        top_k: ROUTER_TOP_K,
        // No EPLB redundancy in this deployment: one slot per expert.
        slot_count: NUM_EXPERTS,
        hidden_bytes: hidden_bytes.into(),
        direction,
        fabric: Fabric::Nvlink,
    };
    // Combine stages one row per (token, selected expert) before reducing.
    let top_k_hidden_bytes = hidden_bytes
        .checked_mul(ROUTER_TOP_K)
        .ok_or_else(|| fit_failed("combine staging byte width overflows u32"))?;
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

    // `split_for_ep` hands rank *r* the contiguous expert range
    // `[r * experts_per_rank, (r + 1) * experts_per_rank)`. That is the shard
    // the routed grouped GEMMs are keyed by; the `RoundRobin` in `MoeNetConfig`
    // above only governs which peer a token is sent to. It is hoisted out of
    // the struct literal because finalize-routing reduces exactly the rows
    // these GEMMs produce and must be keyed by the same shards.
    let moe_expert_compute = MoeExpertComputeLocalWorkletConfig::split_for_ep(
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
            fp8_quant_backends: ROUTED_FP8_QUANT_BACKENDS.to_vec(),
            grouped_gemm_backends: expert_backends.to_vec(),
            fp8_grouped_gemm_backends: FP8_PRODUCTION_GROUPED_GEMM_BACKENDS.to_vec(),
            use_fp8_blockscale_grouped_gemm: true,
            local_ppm: Vec::new(),
        },
        routing.ppm(),
    );
    let experts_per_rank = NUM_EXPERTS / u32::from(parallel.ep_size);
    let sparse_finalization: Vec<MoeFinalizeRoutingKernelConfig> = moe_expert_compute
        .iter()
        .map(|expert_config| MoeFinalizeRoutingKernelConfig {
            backends: FP8_PRODUCTION_GROUPED_GEMM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden_size: model.hidden_dim.clone(),
            top_k: model.router_top_k,
            num_experts_per_rank: experts_per_rank,
            local_ppm: expert_config.local_ppm.clone(),
            // The routed GEMMs emit bf16 no matter how the weights are stored,
            // and this kernel reduces those outputs.
            dtype: DType::Bf16,
        })
        .collect();

    Ok(Glm52VllmDsaMoeConfigs {
        model: model.clone(),
        parallel: parallel.clone(),
        mtp_mode,
        dense_full_index_attention,
        dense_ffn: VllmGlm52DenseFfnLocalWorkletConfig {
            fp8_quant_backends: DENSE_FP8_QUANT_BACKENDS.to_vec(),
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
        moe_dispatch_prepare: MoeAlltoallPrepareKernelConfig {
            backends: MOE_ALLTOALL_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            ep_size: u32::from(parallel.ep_size),
            top_k: ROUTER_TOP_K,
            slot_count: NUM_EXPERTS,
            fabric: Fabric::Nvlink,
        },
        moe_dispatch: moe_alltoall(MoeAlltoallDirection::Dispatch),
        moe_expert_compute: moe_expert_compute.clone(),
        // One received row in, one permuted row out.
        moe_expand_input_rows: ElementwiseKernelConfig {
            backends: ELEMENTWISE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            input_bytes_per_token: hidden_bytes.into(),
            output_bytes_per_token: hidden_bytes.into(),
        },
        // `torch.zeros(token_count * top_k, hidden)`: no reads, `top_k` rows
        // written per token.
        moe_combine_output_fill: ElementwiseKernelConfig {
            backends: ELEMENTWISE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            input_bytes_per_token: 0.into(),
            output_bytes_per_token: top_k_hidden_bytes.into(),
        },
        moe_combine: moe_alltoall(MoeAlltoallDirection::Combine),
        // `torch.sum` over the top-k axis: reads what the fill wrote, writes one
        // row per token.
        moe_combine_reduce: ElementwiseKernelConfig {
            backends: ELEMENTWISE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            input_bytes_per_token: top_k_hidden_bytes.into(),
            output_bytes_per_token: hidden_bytes.into(),
        },
        shared_expert: VllmGlm52SharedExpertLocalWorkletConfig {
            gemm_backends: gemm_backends.to_vec(),
            fp8_quant_backends: DENSE_FP8_QUANT_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden_dim: model.hidden_dim.clone(),
            moe_intermediate_dim: model.moe_intermediate_dim.clone(),
            n_shared_experts: model.num_shared_experts,
            dtype: DType::Bf16,
            gemm_dtype,
        },
        sparse_finalization,
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
            n: model.vocab_size.clone(),
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

pub fn resolve_configs(cfgs: &Glm52VllmDsaMoeConfigs) -> Glm52VllmDsaMoeResolved {
    Glm52VllmDsaMoeResolved {
        dense_full_index_attention: VllmGlm52DsaAttnLocalWorklet::resolve_config(
            &cfgs.dense_full_index_attention,
        ),
        dense_ffn: VllmGlm52DenseFfnLocalWorklet::resolve_config(&cfgs.dense_ffn),
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
        moe_dispatch_prepare: cfgs.moe_dispatch_prepare.clone(),
        moe_dispatch: cfgs.moe_dispatch.clone(),
        moe_expert_compute: cfgs
            .moe_expert_compute
            .iter()
            .map(MoeExpertComputeLocalWorklet::resolve_config)
            .collect(),
        moe_expand_input_rows: cfgs.moe_expand_input_rows.clone(),
        moe_combine_output_fill: cfgs.moe_combine_output_fill.clone(),
        moe_combine: cfgs.moe_combine.clone(),
        moe_combine_reduce: cfgs.moe_combine_reduce.clone(),
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
    /// The three collective leaves. Each is one kernel over the whole EP group,
    /// so none of them is wrapped in a `Max` over ranks the way the rank-local
    /// work is — every rank is inside the same transfer.
    dispatch_prepare: Op<MoeAlltoallPrepareKernel>,
    dispatch: Op<MoeAlltoallKernel>,
    expand_input_rows: Op<ElementwiseKernel>,
    /// One per EP rank, in rank order.
    expert_compute: Vec<MoeExpertComputeLocalWorklet>,
    combine_output_fill: Op<ElementwiseKernel>,
    combine: Op<MoeAlltoallKernel>,
    combine_reduce: Op<ElementwiseKernel>,
    shared_expert: VllmGlm52SharedExpertLocalWorklet,
    /// One per EP rank, in rank order — same shards as `expert_compute`.
    finalization: Vec<Op<MoeFinalizeRoutingKernel>>,
    /// Each EP rank's shard of the routing distribution, in rank order. The
    /// expert worklets own the same shards; they are carried again here because
    /// `expandInputRows` moves exactly the rows the grouped GEMM then reads, so
    /// its token count is that rank's apportioned selection count.
    local_ppms: Vec<Vec<u32>>,
    ep_size: u16,
    top_k: u32,
    /// `destinations_per_token` for this deployment's routing recipe, resolved
    /// once at build: it turns a token count into the wire rows the transfer
    /// actually moves. Static, because `ep_size`/`top_k`/`slot_count` are.
    rows_per_token: f64,
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
        let dispatch_prepare = build_atomic(
            format!("{name}.moe.dispatch.prepare"),
            common.moe_dispatch_prepare.clone(),
            MoeAlltoallPrepareKernel::build,
            bridge,
        )?;
        let dispatch = build_atomic(
            format!("{name}.moe.dispatch"),
            common.moe_dispatch.clone(),
            MoeAlltoallKernel::build,
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
        let expand_input_rows = build_atomic(
            format!("{name}.moe.expand_input_rows"),
            common.moe_expand_input_rows.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        let combine_output_fill = build_atomic(
            format!("{name}.moe.combine.output_fill"),
            common.moe_combine_output_fill.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        let combine = build_atomic(
            format!("{name}.moe.combine"),
            common.moe_combine.clone(),
            MoeAlltoallKernel::build,
            bridge,
        )?;
        let combine_reduce = build_atomic(
            format!("{name}.moe.combine.reduce"),
            common.moe_combine_reduce.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        let shared_expert = VllmGlm52SharedExpertLocalWorklet::build(
            format!("{name}.moe.shared_expert"),
            common.shared_expert.clone(),
            bridge,
        )?;
        // Every rank keeps the same slot name, for the reason spelled out at
        // `expert_compute` above.
        let finalization = common
            .sparse_finalization
            .iter()
            .map(|rank_config| {
                build_atomic(
                    format!("{name}.moe.finalization"),
                    rank_config.clone(),
                    MoeFinalizeRoutingKernel::build,
                    bridge,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            name,
            attention,
            router,
            dispatch_prepare,
            dispatch,
            expand_input_rows,
            expert_compute,
            combine_output_fill,
            combine,
            combine_reduce,
            shared_expert,
            finalization,
            local_ppms: common
                .moe_expert_compute
                .iter()
                .map(|rank_resolved| rank_resolved.raw_cfg.local_ppm.clone())
                .collect(),
            ep_size: common.raw_cfg.parallel.ep_size,
            top_k: common.raw_cfg.model.router_top_k,
            rows_per_token: destinations_per_token(
                u32::from(common.raw_cfg.parallel.ep_size),
                common.raw_cfg.model.router_top_k,
                common.moe_dispatch.slot_count,
            ),
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
        let dispatch_prepare = self.dispatch_prepare.compile(builder);
        let dispatch = self.dispatch.compile(builder);
        let expand = labeled_max(
            format!("{}.moe.expand_input_rows [Max over EP ranks]", self.name),
            (0..self.ep_size)
                .map(|_| self.expand_input_rows.compile(builder))
                .collect(),
        );
        let experts = labeled_max(
            format!("{}.moe.routed_experts [Max over EP ranks]", self.name),
            self.expert_compute
                .iter()
                .map(|rank_expert| rank_expert.compile(builder))
                .collect(),
        );
        // Leaves are numbered in the order they are declared here, and `eval`
        // must push in the same order — so these statements follow the measured
        // launch order, which is also the order the `Sum` below lists them in.
        let finalization = labeled_max(
            format!("{}.moe.finalization [Max over EP ranks]", self.name),
            self.finalization
                .iter()
                .map(|rank_finalize| rank_finalize.compile(builder))
                .collect(),
        );
        let combine_output_fill = self.combine_output_fill.compile(builder);
        let combine = self.combine.compile(builder);
        let combine_reduce = self.combine_reduce.compile(builder);
        let shared = labeled_max(
            format!("{}.moe.shared_expert [Max over home groups]", self.name),
            (0..self.ep_size)
                .map(|_| self.shared_expert.compile(builder))
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
                    label: format!("{}.moe [router -> dispatch prepare -> dispatch -> expand -> routed experts -> finalization -> combine (fill -> transfer -> reduce) -> shared expert]", self.name),
                    child: Box::new(CostNode::Sum(vec![
                        router,
                        dispatch_prepare,
                        dispatch,
                        expand,
                        experts,
                        finalization,
                        combine_output_fill,
                        combine,
                        combine_reduce,
                        shared,
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
        // vLLM sizes every rank's transfer buffers by `max(global_num_tokens)`
        // — one all-gathered scalar, so the busiest DP rank sets the shape of
        // the collective for everyone. That maximum, not the replica total, is
        // this leaf's axis.
        let max_tokens_per_rank = batch
            .groups
            .iter()
            .map(|group| group.batch_tokens)
            .max()
            .unwrap_or(0);
        eval_atomic_or_zero(
            &self.dispatch_prepare,
            MoeAlltoallPrepareKernelInput {
                tokens_per_rank: max_tokens_per_rank,
            },
            max_tokens_per_rank == 0,
            ev,
        );
        // The transfer's two axes are row counts, and the two sides of it are
        // loaded by different things. What leaves a rank is set by how the
        // tokens are spread over DP groups: one row per (token, distinct
        // destination rank), so a group's token count times the fan-out. What
        // arrives at a rank is set by how the experts are spread over EP ranks
        // — that rank's apportioned share of the global selections, deduplicated
        // by the same fan-out (a token picking two experts on one rank still
        // crosses the wire once).
        let max_send_rows = batch
            .groups
            .iter()
            .map(|group| rows_from_tokens(group.batch_tokens, self.rows_per_token))
            .max()
            .unwrap_or(0);
        let max_recv_rows = self
            .local_ppms
            .iter()
            .map(|local_ppm| {
                let selections = rows_for_rank(batch.routed_selections, local_ppm);
                rows_from_tokens(selections, self.rows_per_token / f64::from(self.top_k))
            })
            .max()
            .unwrap_or(0);
        eval_atomic_or_zero(
            &self.dispatch,
            MoeAlltoallKernelInput {
                max_send_rows,
                max_recv_rows,
            },
            max_tokens_per_rank == 0,
            ev,
        );
        // `expandInputRows` moves the rows this rank was sent into the dense
        // per-expert batch the grouped GEMM reads, so its row count is that
        // rank's apportioned share of the global selections — the same quantity
        // the expert worklet derives internally from the same shard.
        for local_ppm in &self.local_ppms {
            let rank_rows = rows_for_rank(batch.routed_selections, local_ppm);
            eval_atomic_or_zero(
                &self.expand_input_rows,
                ElementwiseKernelInput {
                    num_tokens: rank_rows,
                },
                rank_rows == 0,
                ev,
            );
        }
        for rank_expert in &self.expert_compute {
            eval_expert_or_zero(rank_expert, batch.routed_selections, ev);
        }
        // The kernel's own axis is the GLOBAL token count; it re-derives this
        // rank's row share from the same popularity shard, so passing the
        // replica total here is what makes a skewed rank finalize more.
        for rank_finalize in &self.finalization {
            eval_atomic_or_zero(
                rank_finalize,
                MoeFinalizeRoutingKernelInput {
                    token_count: batch.total_tokens,
                },
                batch.total_tokens == 0,
                ev,
            );
        }
        // The return leg is the dispatch transposed — what an EP rank received
        // it now sends back — so the same two numbers swap places.
        eval_atomic_or_zero(
            &self.combine_output_fill,
            ElementwiseKernelInput {
                num_tokens: max_tokens_per_rank,
            },
            max_tokens_per_rank == 0,
            ev,
        );
        eval_atomic_or_zero(
            &self.combine,
            MoeAlltoallKernelInput {
                max_send_rows: max_recv_rows,
                max_recv_rows: max_send_rows,
            },
            max_tokens_per_rank == 0,
            ev,
        );
        eval_atomic_or_zero(
            &self.combine_reduce,
            ElementwiseKernelInput {
                num_tokens: max_tokens_per_rank,
            },
            max_tokens_per_rank == 0,
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
    #[must_use]
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
                "{} (Glm52VllmDsaMoeModel) [EP{}; attention TP1/DP{}; MTP={:?}; \
                 timing_context<={}]",
                self.name,
                self.ep_size,
                self.num_attn_dp_groups,
                self.mtp_mode,
                TIMING_MAX_MODEL_LEN
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
                    decode: group.decode_context.map(|context_len| {
                        VllmGlm52DsaAttnLocalDecodeInput {
                            batch_size: group.decode_tokens,
                            context_len,
                            requires_padding: false,
                        }
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

/// This EP rank's share of the global `(token, expert)` selections, apportioned
/// by its own popularity shard. Same expression the expert worklet uses, so the
/// row counts of `expandInputRows`, the grouped GEMMs and `finalizeMoeRouting`
/// cannot drift apart.
/// Expected distinct destination ranks one token reaches.
///
/// A token picks `top_k` distinct slots out of `slot_count`; a rank owns
/// `slot_count / ep_size` of them and is missed only when all `top_k` picks fall
/// outside its shard. For GLM-5.2 (ep 8, top-8, 256 slots) that is 5.29 of 8 —
/// so a rank sends ~5.3 rows per token, not 8, and not 1.
///
/// This must stay the same formula the profiling runner uses to size its
/// dispatch send buffer (`_destinations_per_token` in
/// `profiling/runners/moe/flashinfer_mnnvl_alltoall.py`); the two are the model
/// and the measurement of one quantity.
fn destinations_per_token(ep_size: u32, top_k: u32, slot_count: u32) -> f64 {
    let experts_per_rank = slot_count / ep_size;
    let mut miss = 1.0_f64;
    for i in 0..top_k {
        let remaining_outside = i64::from(slot_count) - i64::from(experts_per_rank) - i64::from(i);
        if remaining_outside <= 0 {
            return f64::from(ep_size);
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "remaining_outside is a slot-count difference (model expert counts, at most \
                      thousands), far under f64's 52-bit exact integer range"
        )]
        let remaining_outside_f64 = remaining_outside as f64;
        miss *= remaining_outside_f64 / f64::from(slot_count - i);
    }
    f64::from(ep_size) * (1.0 - miss)
}

/// Wire rows for `tokens` tokens at `rows_per_token`, never rounding a live
/// transfer down to nothing.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "rows_per_token is a non-negative per-token wire-row multiplier (destinations_per_token, \
              bounded by ep_size, a small single-digit count); tokens is a per-iteration batch size, \
              so the product stays far under u32::MAX for any realistic batch"
)]
fn rows_from_tokens(tokens: u32, rows_per_token: f64) -> u32 {
    if tokens == 0 {
        return 0;
    }
    ((f64::from(tokens) * rows_per_token).round() as u32).max(1)
}

fn rows_for_rank(global_expert_selections: u32, local_ppm: &[u32]) -> u32 {
    RoutingDistribution::to_per_expert_counts(global_expert_selections, local_ppm)
        .into_iter()
        .sum()
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
            + EXPAND_INPUT_ROWS_SLOTS
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

    #[test]
    fn a_token_reaches_fewer_ranks_than_it_picks_experts() {
        // GLM-5.2: 8 picks out of 256 slots over 8 ranks collide often enough
        // that a token crosses the wire ~5.3 times, not 8. Using top_k here
        // would over-count the transfer by 1.5x.
        let glm = destinations_per_token(8, ROUTER_TOP_K, NUM_EXPERTS);
        assert!(
            (glm - 5.2947).abs() < 1e-3,
            "expected ~5.2947 destinations per token, got {glm}"
        );
        assert!(glm < f64::from(ROUTER_TOP_K));

        // The two degenerate ends are exact, not approximated.
        assert_eq!(destinations_per_token(8, 1, 256), 1.0);
        assert_eq!(destinations_per_token(8, 256, 256), 8.0);
        // Wider groups fan out further at the same top_k.
        assert!(destinations_per_token(16, 8, 256) > destinations_per_token(4, 8, 256));
    }

    #[test]
    fn a_live_transfer_never_rounds_down_to_no_rows() {
        assert_eq!(rows_from_tokens(0, 5.29), 0);
        // One token still crosses the wire even if the rate rounds below 1.
        assert_eq!(rows_from_tokens(1, 0.4), 1);
        assert_eq!(rows_from_tokens(1_024, 5.2947), 5_422);
    }

    #[test]
    fn the_two_transfer_axes_agree_on_a_balanced_batch_and_split_on_a_skewed_one() {
        // Balanced: `ep_size` DP groups of equal size against a uniform
        // popularity shard must land on the diagonal, because every row one rank
        // sends is a row another rank receives and nothing breaks the symmetry.
        let ep_size = 8_u32;
        let tokens_per_group = 1_024_u32;
        let rows_per_token = destinations_per_token(ep_size, ROUTER_TOP_K, NUM_EXPERTS);
        let send = rows_from_tokens(tokens_per_group, rows_per_token);

        let selections = tokens_per_group * ep_size * ROUTER_TOP_K;
        let uniform_ppm = vec![1_000_000 / NUM_EXPERTS; (NUM_EXPERTS / ep_size) as usize];
        let recv = rows_from_tokens(
            rows_for_rank(selections, &uniform_ppm),
            rows_per_token / f64::from(ROUTER_TOP_K),
        );
        let drift = (f64::from(send) - f64::from(recv)).abs() / f64::from(send);
        assert!(
            drift < 0.01,
            "balanced batch should be on the diagonal: {send} vs {recv}"
        );

        // A hot shard pulls the receive side up without touching the send side.
        let mut skewed_ppm = uniform_ppm.clone();
        skewed_ppm[0] *= 4;
        let skewed_recv = rows_from_tokens(
            rows_for_rank(selections, &skewed_ppm),
            rows_per_token / f64::from(ROUTER_TOP_K),
        );
        assert!(
            skewed_recv > recv,
            "a hotter shard must receive more rows: {skewed_recv} vs {recv}"
        );
        // ...and the pair stays inside what conservation allows, which is what
        // the kernel's `infeasible_mask` strips from the grid.
        assert!(f64::from(skewed_recv) <= f64::from(send) * f64::from(ep_size));
    }

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
        // The transfer leaf is profiled under uniform routing and keyed by shape
        // alone, so the skew must be invisible there — it lands on the expert
        // and finalize shards, which is where the measured asymmetry actually
        // shows up.
        assert_eq!(skewed.moe_dispatch, uniform.moe_dispatch);
        assert_eq!(skewed.moe_combine, uniform.moe_combine);
        assert_eq!(skewed.moe_dispatch_prepare, uniform.moe_dispatch_prepare);
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
        assert_eq!(cfg.dense_full_index_attention.gemm_dtype, DType::Bf16);
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
        assert_eq!(
            cfg.moe_expert_compute[0].grouped_gemm_backends,
            vec!["torch"]
        );
        // The transfer is the profiled flashinfer MNNVL all-to-all, keyed by the
        // bf16 wire width vLLM actually sends (quantisation is deferred past it).
        assert_eq!(cfg.moe_dispatch.backends, vec!["flashinfer_mnnvl"]);
        assert_eq!(cfg.moe_combine.backends, vec!["flashinfer_mnnvl"]);
        assert_eq!(cfg.moe_dispatch.direction, MoeAlltoallDirection::Dispatch);
        assert_eq!(cfg.moe_combine.direction, MoeAlltoallDirection::Combine);
        assert_eq!(cfg.moe_dispatch.hidden_bytes.get(), HIDDEN_DIM * 2);
        assert_eq!(cfg.moe_combine.hidden_bytes.get(), HIDDEN_DIM * 2);
        assert_eq!(cfg.moe_dispatch.ep_size, 8);
        assert_eq!(cfg.moe_dispatch.top_k, ROUTER_TOP_K);
        // No EPLB redundancy: one slot per expert.
        assert_eq!(cfg.moe_dispatch.slot_count, NUM_EXPERTS);
        assert_eq!(cfg.moe_dispatch_prepare.backends, vec!["flashinfer_mnnvl"]);
        assert_eq!(cfg.moe_dispatch_prepare.ep_size, 8);
        assert_eq!(cfg.moe_dispatch_prepare.slot_count, NUM_EXPERTS);
        assert_eq!(cfg.sparse_router.base_dtype, DType::Bf16);
        assert_eq!(cfg.sparse_router.router_semantic_dtype, DType::Fp32);
        assert_eq!(cfg.shared_expert.dtype, DType::Bf16);
        assert_eq!(cfg.shared_expert.gemm_dtype, DType::Bf16);
        assert_eq!(cfg.final_norm.dtype, DType::Bf16);
        assert_eq!(cfg.lm_head.dtype, DType::Bf16);
        assert_eq!(cfg.lm_head.backends, vec!["torch_linear"]);
        assert_eq!(cfg.moe_dispatch.fabric, Fabric::Nvlink);
        assert_eq!(cfg.moe_combine.fabric, Fabric::Nvlink);
        assert_eq!(cfg.moe_dispatch_prepare.fabric, Fabric::Nvlink);
        assert_eq!(cfg.embedding.input_bytes_per_token, 12_296);
        assert_eq!(cfg.embedding.output_bytes_per_token, 12_288);
        // Finalize-routing is the profiled TensorRT-LLM kernel, one config per
        // EP rank carrying that rank's popularity shard.
        assert_eq!(cfg.sparse_finalization.len(), 8);
        assert_eq!(cfg.sparse_finalization[0].num_experts_per_rank, 32);
        assert_eq!(cfg.sparse_finalization[0].local_ppm.len(), 32);
        assert_eq!(cfg.sparse_finalization[0].dtype, DType::Bf16);
        assert_eq!(
            cfg.sparse_finalization[0].local_ppm,
            cfg.moe_expert_compute[0].local_ppm
        );
        // One row in, one row out — the gather that feeds the grouped GEMM.
        assert_eq!(cfg.moe_expand_input_rows.input_bytes_per_token, 12_288);
        assert_eq!(cfg.moe_expand_input_rows.output_bytes_per_token, 12_288);
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
        assert_eq!(
            cfg.moe_expert_compute[0].grouped_gemm_backends,
            vec!["deepgemm"]
        );
        assert_eq!(
            cfg.moe_expert_compute[0].fp8_grouped_gemm_backends,
            vec!["flashinfer_trtllm"]
        );
        // fp8 expert weights do NOT make the all-to-all payload fp8: vLLM's
        // flashinfer two-sided path defers the activation quant until after the
        // transfer, so the wire stays bf16 no matter how the experts are stored
        // — and the transfer therefore keeps the same backend and the same key
        // as the bf16 deployment.
        assert_eq!(cfg.moe_dispatch.backends, vec!["flashinfer_mnnvl"]);
        assert_eq!(cfg.moe_combine.backends, vec!["flashinfer_mnnvl"]);
        assert_eq!(cfg.moe_dispatch.hidden_bytes.get(), HIDDEN_DIM * 2);
        assert_eq!(cfg.moe_combine.hidden_bytes.get(), HIDDEN_DIM * 2);

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
            assert_eq!(
                attention.q_absorb_backends,
                vec!["torch_mla_q_absorb_glm52"]
            );
            assert_eq!(attention.v_up_backends, vec!["torch_mla_v_up_glm52"]);
        }

        let resolved = resolve_configs(&cfg);
        assert_eq!(
            resolved.moe_expert_compute[0]
                .gate_up_fp8
                .as_ref()
                .unwrap()
                .gemm
                .dtype(),
            DType::Fp8E4m3
        );
        assert_eq!(
            resolved.moe_expert_compute[0]
                .down_fp8
                .as_ref()
                .unwrap()
                .gemm
                .dtype(),
            DType::Fp8E4m3
        );
        assert_eq!(
            resolved.moe_expert_compute[0].act.input_bytes_per_token,
            2 * MOE_INTERMEDIATE_DIM * 2
        );
        assert_eq!(resolved.dense_ffn.gate_up_proj.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.dense_ffn.down_proj.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.dense_ffn.post_attn_add_rms_norm.dtype, DType::Bf16);
        assert_eq!(
            resolved.dense_ffn.silu_and_mul.input_bytes_per_token,
            2 * DENSE_INTERMEDIATE_DIM * 2
        );
        assert_eq!(
            resolved.sparse_router.router_gemm_bf16_proxy.dtype,
            DType::Fp8E4m3
        );
        assert_eq!(
            resolved.sparse_router.post_attn_add_rms_norm.dtype,
            DType::Bf16
        );
        assert_eq!(
            resolved.sparse_router.raw_cfg.router_semantic_dtype,
            DType::Fp32
        );
        assert_eq!(resolved.shared_expert.gate_up_proj.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.shared_expert.down_proj.dtype, DType::Fp8E4m3);
        assert_eq!(
            resolved.shared_expert.silu_and_mul.input_bytes_per_token,
            2 * MOE_INTERMEDIATE_DIM * 2
        );
        // lm_head stays BF16 even under `fp8`: the checkpoint's
        // `modules_to_not_convert` excludes it, so it never reaches the FP8 GEMM.
        assert_eq!(resolved.lm_head.dtype, DType::Bf16);
        assert_eq!(resolved.lm_head.backends, vec!["torch_linear"]);
        assert_eq!(
            resolved.dense_full_index_attention.fused_qkv_a_proj.dtype,
            DType::Fp8E4m3
        );
        assert_eq!(
            resolved.dense_full_index_attention.q_b_proj.dtype,
            DType::Fp8E4m3
        );
        assert_eq!(
            resolved.dense_full_index_attention.o_proj.dtype,
            DType::Fp8E4m3
        );
        assert_eq!(
            resolved.dense_full_index_attention.q_absorb.dtype,
            DType::Bf16
        );
        assert_eq!(resolved.dense_full_index_attention.v_up.dtype, DType::Bf16);
        let indexer = resolved
            .dense_full_index_attention
            .indexer
            .as_ref()
            .unwrap();
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
        assert_eq!(cfg.mtp_attention.as_ref().unwrap().base_dtype, DType::Bf16);
        assert_eq!(cfg.mtp_head.as_ref().unwrap().dtype, DType::Bf16);
        assert_eq!(cfg.mtp_head.as_ref().unwrap().gemm_dtype, DType::Fp8E4m3);
        assert_eq!(
            resolved.mtp_prelude.as_ref().unwrap().eh_proj.dtype,
            DType::Fp8E4m3
        );
        assert_eq!(
            resolved
                .mtp_prelude
                .as_ref()
                .unwrap()
                .embedding_rms_norm
                .dtype,
            DType::Bf16
        );
        assert_eq!(
            resolved.mtp_head.as_ref().unwrap().lm_head.dtype,
            DType::Fp8E4m3
        );
        assert_eq!(
            resolved
                .mtp_head
                .as_ref()
                .unwrap()
                .shared_head_rms_norm
                .dtype,
            DType::Bf16
        );
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
        // Each sparse-layer variant carries two leaves the native graph has no
        // equivalent of: combine's zero fill and its top-k reduction, split out
        // of the profiled transfer so that leaf's cache stays two-dimensional.
        // Three sparse variants without MTP, four with it.
        assert_eq!(
            expected_slot_count(8, false, Glm52MtpMode::Off),
            1_041 + 3 * 2
        );
        assert_eq!(
            expected_slot_count(8, false, Glm52MtpMode::FullIndex),
            1_444 + 4 * 2
        );
        assert_eq!(
            expected_slot_count(8, false, Glm52MtpMode::IndexShare),
            1_324 + 4 * 2
        );
        // FP8 adds the vLLM activation-quantisation leaves: five per layer
        // variant (attention 3 + FFN 2), plus a sixth on the two variants that
        // run an indexer (its q_proj). That is 22 x ep = 176 more than the
        // native graph's 1_074. See `doc/alignment/glm52_dp8_ep8_report.md`
        // section 4.3, which decodes each measured launch by its GEMM shape.
        assert_eq!(
            expected_slot_count(8, true, Glm52MtpMode::Off),
            1_074 + 176 + 15 + 3 * 2
        );
        assert_eq!(
            expected_slot_count(8, true, Glm52MtpMode::FullIndex),
            1_488 + 224 + 20 + 4 * 2
        );
        assert_eq!(
            expected_slot_count(8, true, Glm52MtpMode::IndexShare),
            1_368 + 216 + 20 + 4 * 2
        );
        #[allow(
            clippy::cast_possible_truncation,
            reason = "ep is looped from the fixed array [1,2,4,8,16] converted to usize then back \
                      to u16; always far under u16::MAX"
        )]
        for ep in [1_u16, 2, 4, 8, 16] {
            let ep = usize::from(ep);
            // The `+ 15` / `+ 20` are the rank-independent collectives: each
            // sparse variant contributes `DISPATCH_SLOTS + COMBINE_SLOTS` = 5,
            // over three variants without MTP and four with it.
            assert_eq!(
                expected_slot_count(ep as u16, false, Glm52MtpMode::Off),
                129 * ep + 15
            );
            assert_eq!(
                expected_slot_count(ep as u16, true, Glm52MtpMode::Off),
                157 * ep + 15
            );
            assert_eq!(
                expected_slot_count(ep as u16, false, Glm52MtpMode::FullIndex),
                179 * ep + 20
            );
            assert_eq!(
                expected_slot_count(ep as u16, true, Glm52MtpMode::FullIndex),
                215 * ep + 20
            );
            assert_eq!(
                expected_slot_count(ep as u16, false, Glm52MtpMode::IndexShare),
                164 * ep + 20
            );
            assert_eq!(
                expected_slot_count(ep as u16, true, Glm52MtpMode::IndexShare),
                199 * ep + 20
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
            (Glm52MtpMode::Off, 1_041 + 3 * 2, 20),
            (Glm52MtpMode::FullIndex, 1_444 + 4 * 2, 27),
            (Glm52MtpMode::IndexShare, 1_324 + 4 * 2, 27),
        ] {
            let manifest = compile_max_label_inventory(mode);
            let max_labels: Vec<&str> = manifest
                .nodes
                .iter()
                .zip(&manifest.node_labels)
                .filter(|&(node, _label)| matches!(node, FlatCostNode::Max { .. }))
                .map(|(_node, label)| {
                    label
                        .as_deref()
                        .expect("every GLM L4 Max must have a manifest label")
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
        #[allow(
            clippy::cast_possible_truncation,
            reason = "decode_lens is a small hardcoded test-fixture vector, far under u32::MAX entries"
        )]
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
        assert_eq!(cfg.moe_dispatch.ep_size, 8);
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
