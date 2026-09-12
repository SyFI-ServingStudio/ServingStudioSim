//! GLM-5.2 local DSA attention worklet in **vLLM kernel granularity**.
//!
//! Models TP1 MLA attention with the optional DSA indexer, from the entry
//! residual RMSNorm through o_proj.
//!
//! The one divergence: vLLM launches a BF16->FP8 block quantisation
//! (`fp8_blockscale_gemm::scale_1x128_kernel`) before **each** dense FP8 GEMM,
//! so this worklet cuts those as explicit leaves. Measured, per iteration
//! (`doc/alignment/glm52_dp8_ep8_report.md` section 4.3 decodes all 411
//! unmodelled quant launches by GEMM shape):
//!   - `fused_qkv_a_proj` (N=2624, K=6144)  -- 78 launches
//!   - `q_b_proj`         (N=16384, K=2048) -- 78 launches
//!   - `o_proj`           (N=6144, K=16384) -- 78 launches
//!   - `indexer.q_proj`   (N=4096, K=2048)  -- 21 launches (full-index layers)
//!
//! The indexer's quant leaf lives here rather than inside the L2 indexer op:
//! this worklet is where the indexer is composed, and forking all fifteen of
//! that op's leaves to add one would duplicate far more than it clarifies.
//!
//! The quant leaves exist only when `gemm_dtype` is FP8.

use std::sync::Arc;

use super::glm52_dsa_attn_common::{
    normalize_glm52_dsa_attn_input as normalize_input, resolve_glm52_dsa_attn_tp_partition,
    Glm52DsaAttnLocalDecodeInput, Glm52DsaAttnLocalInput,
};
use crate::op::attention::{
    DsaIndexerConfig, DsaIndexerInput, DsaIndexerLaunchGraph, DsaIndexerOp,
    DsaSparseMlaAttentionConfig, DsaSparseMlaAttentionInput, DsaSparseMlaAttentionOp,
    DsaSparseMlaExactVarlenConfig, DsaSparseMlaLaunchGraph,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    BatchedGemmKernel, BatchedGemmKernelConfig, BatchedGemmKernelInput,
    Fp8PerTokenGroupQuantKernel, Fp8PerTokenGroupQuantKernelConfig,
    Fp8PerTokenGroupQuantKernelInput, ResidualRmsNormKernel, ResidualRmsNormKernelConfig,
    ResidualRmsNormKernelInput, RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput,
    SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput, VllmMlaRopeKernel,
    VllmMlaRopeKernelConfig, VllmMlaRopeKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

const HIDDEN_DIM: u32 = 6144;
const NUM_ATTENTION_HEADS: u32 = 64;
const NUM_KV_HEADS: u32 = 1;
const Q_LORA_RANK: u32 = 2048;
const KV_LORA_RANK: u32 = 512;
const QK_NOPE_HEAD_DIM: u32 = 192;
const ROPE_DIM: u32 = 64;
const V_HEAD_DIM: u32 = 256;
const MODEL_INDEX_HEADS: u32 = 32;
const PROFILE_INDEX_HEADS: u32 = 64;
const INDEX_HEAD_DIM: u32 = 128;
const SELECTED_K: u32 = 2048;
/// Full-context fixtures for the inherited H200 tests. Production configs may
/// choose a smaller positive context and matching-or-larger logits stride.
#[cfg(test)]
const MAX_MODEL_LEN: u32 = 1_048_576;
#[cfg(test)]
const LOGITS_ROW_STRIDE: u32 = 1_048_576;
const CACHE_BLOCK_SIZE: u32 = 64;
const QUANT_BLOCK_SIZE: u32 = 128;
const SOFTMAX_SCALE_DENOMINATOR: u32 = 16;

#[cfg(test)]
const SOURCE_ORDER_WITH_INDEXER: [&str; 14] = [
    "input_add_rms_norm",
    "fused_qkv_a_proj_input_quant",
    "fused_qkv_a_proj",
    "q_a_rms_norm",
    "q_b_proj_input_quant",
    "q_b_proj",
    "kv_a_rms_norm",
    "main_rope",
    "indexer",
    "q_absorb",
    "sparse_mla",
    "v_up",
    "o_proj_input_quant",
    "o_proj",
];

#[cfg(test)]
const SOURCE_ORDER_INDEX_SHARE: [&str; 13] = [
    "input_add_rms_norm",
    "fused_qkv_a_proj_input_quant",
    "fused_qkv_a_proj",
    "q_a_rms_norm",
    "q_b_proj_input_quant",
    "q_b_proj",
    "kv_a_rms_norm",
    "main_rope",
    "q_absorb",
    "sparse_mla",
    "v_up",
    "o_proj_input_quant",
    "o_proj",
];

/// Raw GLM-5.2 local-attention identity. Backend roles remain independent and
/// are baked into concrete L1/L2 configs by [`Self::resolve_config`].
#[derive(Clone, Debug)]
pub struct VllmGlm52DsaAttnLocalWorkletConfig {
    /// Backends for the pre-GEMM activation quantisation. Unused when
    /// `gemm_dtype` is BF16.
    pub fp8_quant_backends: Vec<&'static str>,
    pub include_indexer: bool,
    /// Number of ranks over which MLA query heads are partitioned. This
    /// worklet owns one rank-local compute segment; the L4 graph owns the
    /// following collective because vLLM may fuse it with the next norm.
    pub tp_size: u16,
    pub residual_rms_norm_backends: Vec<&'static str>,
    pub rms_norm_backends: Vec<&'static str>,
    pub single_gemm_backends: Vec<&'static str>,
    /// Backend for the inductor-fused query RoPE -- the worklet's only
    /// pointwise leaf, and not an `elementwise` one: it is a compiled vLLM
    /// fusion that rewrites all of q, not a streaming kernel over the rope
    /// slice. The indexer and sparse-MLA ops keep their own roles below.
    pub main_rope_backends: Vec<&'static str>,
    pub q_absorb_backends: Vec<&'static str>,
    pub v_up_backends: Vec<&'static str>,
    pub indexer_gemm_backends: Vec<&'static str>,
    pub indexer_elementwise_backends: Vec<&'static str>,
    pub index_cache_append_backends: Vec<&'static str>,
    pub index_prefill_logits_backends: Vec<&'static str>,
    pub index_prefill_topk_backends: Vec<&'static str>,
    pub index_decode_logits_backends: Vec<&'static str>,
    pub index_decode_topk_backends: Vec<&'static str>,
    pub sparse_attention_backends: Vec<&'static str>,
    pub sparse_mla_cache_append_backends: Vec<&'static str>,
    pub sparse_elementwise_backends: Vec<&'static str>,
    pub sparse_index_remap_backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    pub num_attention_heads: Dim,
    pub num_kv_heads: Dim,
    pub q_lora_rank: Dim,
    pub kv_lora_rank: Dim,
    pub qk_nope_head_dim: Dim,
    pub rope_dim: Dim,
    pub v_head_dim: Dim,
    pub model_num_index_heads: Dim,
    pub profile_num_index_heads: Dim,
    pub index_head_dim: Dim,
    pub selected_k: u32,
    pub max_model_len: Dim,
    /// Row count of the RoPE cos/sin table, i.e. the checkpoint's
    /// `max_position_embeddings` -- NOT the runtime `max_model_len`. vLLM
    /// builds the table from the model config at `deepseek_v2.py:503`, so a
    /// shorter serving context does not shrink it.
    pub rope_max_position: Dim,
    pub logits_row_stride: Dim,
    pub cache_block_size: u32,
    pub quant_block_size: u32,
    pub softmax_scale_denominator: u32,
    pub base_dtype: DType,
    /// Dtype used only by generic projection SingleGemm leaves. Specialized
    /// MLA/indexer kernels retain their explicit dtype fields below.
    pub gemm_dtype: DType,
    pub index_cache_dtype: DType,
    pub index_q_dtype: DType,
    pub scale_dtype: DType,
    pub weight_dtype: DType,
    pub logits_dtype: DType,
    pub index_dtype: String,
    pub index_scale_format: String,
    pub index_cache_format: String,
    pub index_prefill_span_mode: String,
    pub index_decode_context_mode: String,
    pub index_decode_page_mapping: String,
    /// GLM-5.2 rotates the MLA query half-and-half (GPT-J style), so this is
    /// `false`; vLLM passes the same literal at `deepseek_v2.py:505`.
    pub rope_is_neox_style: bool,
    pub index_clean_logits: bool,
    pub sparse_index_distribution: String,
    pub sparse_cache_layout: String,
    pub sparse_mla_cache_format: String,
    pub sparse_attention_q_dtype: DType,
    pub sparse_attention_cache_dtype: DType,
    pub sparse_attention_output_dtype: DType,
    /// Dedicated request-remap and one-launch varlen prefill recipe. `None`
    /// retains the established H200 fallback; the B200 FP8 path requires it.
    pub sparse_exact_varlen: Option<DsaSparseMlaExactVarlenConfig>,
    pub decode_next_n: u32,
}

/// Pure resolved data: every atomic config is baked, and the optional indexer
/// config is absent for IndexShare instances.
#[derive(Clone, Debug)]
pub struct VllmGlm52DsaAttnLocalWorkletResolved {
    pub raw_cfg: VllmGlm52DsaAttnLocalWorkletConfig,
    pub input_add_rms_norm: ResidualRmsNormKernelConfig,
    /// `Some` exactly when `gemm_dtype` is FP8.
    pub fused_qkv_a_proj_input_quant: Option<Fp8PerTokenGroupQuantKernelConfig>,
    pub fused_qkv_a_proj: SingleGemmKernelConfig,
    pub q_a_rms_norm: RmsNormKernelConfig,
    /// `Some` exactly when `gemm_dtype` is FP8.
    pub q_b_proj_input_quant: Option<Fp8PerTokenGroupQuantKernelConfig>,
    pub q_b_proj: SingleGemmKernelConfig,
    pub kv_a_rms_norm: RmsNormKernelConfig,
    pub main_rope: VllmMlaRopeKernelConfig,
    /// `Some` only when this layer runs an indexer AND `gemm_dtype` is FP8.
    pub indexer_q_proj_input_quant: Option<Fp8PerTokenGroupQuantKernelConfig>,
    pub indexer: Option<DsaIndexerConfig>,
    pub q_absorb: BatchedGemmKernelConfig,
    pub sparse_mla: DsaSparseMlaAttentionConfig,
    pub v_up: BatchedGemmKernelConfig,
    /// `Some` exactly when `gemm_dtype` is FP8.
    pub o_proj_input_quant: Option<Fp8PerTokenGroupQuantKernelConfig>,
    pub o_proj: SingleGemmKernelConfig,
    pub attention_heads_per_rank: Dim,
}

pub type VllmGlm52DsaAttnLocalDecodeInput = Glm52DsaAttnLocalDecodeInput;
pub type VllmGlm52DsaAttnLocalWorkletInput = Glm52DsaAttnLocalInput;

pub struct VllmGlm52DsaAttnLocalWorklet {
    pub name: String,
    pub input_add_rms_norm: Op<ResidualRmsNormKernel>,
    pub fused_qkv_a_proj_input_quant: Option<Op<Fp8PerTokenGroupQuantKernel>>,
    pub fused_qkv_a_proj: Op<SingleGemmKernel>,
    pub q_a_rms_norm: Op<RmsNormKernel>,
    pub q_b_proj_input_quant: Option<Op<Fp8PerTokenGroupQuantKernel>>,
    pub q_b_proj: Op<SingleGemmKernel>,
    pub kv_a_rms_norm: Op<RmsNormKernel>,
    pub main_rope: Op<VllmMlaRopeKernel>,
    pub indexer_q_proj_input_quant: Option<Op<Fp8PerTokenGroupQuantKernel>>,
    pub indexer: Option<DsaIndexerOp>,
    pub q_absorb: Op<BatchedGemmKernel>,
    pub sparse_mla: DsaSparseMlaAttentionOp,
    pub v_up: Op<BatchedGemmKernel>,
    pub o_proj_input_quant: Option<Op<Fp8PerTokenGroupQuantKernel>>,
    pub o_proj: Op<SingleGemmKernel>,
    resolved: VllmGlm52DsaAttnLocalWorkletResolved,
}

impl VllmGlm52DsaAttnLocalWorklet {
    /// Resolve the one supported GLM-5.2 local identity without touching a
    /// bridge, GPU, cache, or `Arc`.
    pub fn resolve_config(
        cfg: &VllmGlm52DsaAttnLocalWorkletConfig,
    ) -> VllmGlm52DsaAttnLocalWorkletResolved {
        validate_config(cfg).unwrap_or_else(|reason| {
            panic!("invalid VllmGlm52DsaAttnLocalWorkletConfig: {reason}")
        });

        let partition = resolve_glm52_dsa_attn_tp_partition(
            &cfg.num_attention_heads,
            &cfg.q_lora_rank,
            &cfg.kv_lora_rank,
            &cfg.qk_nope_head_dim,
            &cfg.rope_dim,
            &cfg.v_head_dim,
            cfg.tp_size,
        )
        .expect("validated GLM-5.2 TP partition");
        let attention_heads_per_rank = partition.attention_heads_per_rank.clone();

        let indexer = cfg.include_indexer.then(|| DsaIndexerConfig {
            gemm_backends: cfg.indexer_gemm_backends.clone(),
            elementwise_backends: cfg.indexer_elementwise_backends.clone(),
            q_rope_backends: Vec::new(),
            index_cache_append_backends: cfg.index_cache_append_backends.clone(),
            prefill_logits_backends: cfg.index_prefill_logits_backends.clone(),
            prefill_topk_backends: cfg.index_prefill_topk_backends.clone(),
            decode_logits_backends: cfg.index_decode_logits_backends.clone(),
            decode_topk_backends: cfg.index_decode_topk_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden_dim: cfg.hidden_dim.clone(),
            q_lora_rank: cfg.q_lora_rank.clone(),
            model_num_index_heads: cfg.model_num_index_heads.clone(),
            profile_num_index_heads: cfg.profile_num_index_heads.clone(),
            index_head_dim: cfg.index_head_dim.clone(),
            rope_dim: cfg.rope_dim.clone(),
            next_n: cfg.decode_next_n,
            max_model_len: cfg.max_model_len.clone(),
            top_k: cfg.selected_k,
            logits_row_stride: cfg.logits_row_stride.clone(),
            cache_block_size: cfg.cache_block_size,
            quant_block_size: cfg.quant_block_size,
            input_dtype: cfg.base_dtype,
            gemm_dtype: cfg.gemm_dtype,
            cache_dtype: cfg.index_cache_dtype,
            q_dtype: cfg.index_q_dtype,
            scale_dtype: cfg.scale_dtype,
            weight_dtype: cfg.weight_dtype,
            logits_dtype: cfg.logits_dtype,
            index_dtype: cfg.index_dtype.clone(),
            scale_format: cfg.index_scale_format.clone(),
            cache_format: cfg.index_cache_format.clone(),
            prefill_span_mode: cfg.index_prefill_span_mode.clone(),
            decode_context_mode: cfg.index_decode_context_mode.clone(),
            decode_page_mapping: cfg.index_decode_page_mapping.clone(),
            clean_logits: cfg.index_clean_logits,
            launch_graph: DsaIndexerLaunchGraph::Separate,
        });

        // One activation quantisation per dense FP8 GEMM, keyed by that GEMM's
        // K axis (the row width being quantised).
        //
        // vLLM's dense linears call `per_token_group_quant_8bit_kernel`, its own
        // CUDA kernel; only the routed grouped GEMM reaches TensorRT-LLM's
        // `scale_1x128_kernel`. The nsys trace shows both in one iteration, so
        // this is a kernel-identity split, not a shape difference: pricing the
        // dense leaves off the routed curve made them ~2.05x too slow at
        // prefill (o_proj alone +27.5 ms) and ~40% too slow at decode.
        let quant_config = |hidden_size: Dim| {
            (cfg.gemm_dtype == DType::Fp8E4m3).then(|| Fp8PerTokenGroupQuantKernelConfig {
                backends: cfg.fp8_quant_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size,
                group_size: 128,
                input_dtype: cfg.base_dtype,
                scale_format: "ue8m0_column_major".to_string(),
            })
        };

        VllmGlm52DsaAttnLocalWorkletResolved {
            input_add_rms_norm: ResidualRmsNormKernelConfig {
                backends: cfg.residual_rms_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden_dim.clone(),
                dtype: cfg.base_dtype,
            },
            fused_qkv_a_proj_input_quant: quant_config(cfg.hidden_dim.clone()),
            fused_qkv_a_proj: SingleGemmKernelConfig {
                backends: cfg.single_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: partition.fused_qkv_a_n,
                k: cfg.hidden_dim.clone(),
                dtype: cfg.gemm_dtype,
            },
            q_a_rms_norm: RmsNormKernelConfig {
                backends: cfg.rms_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.q_lora_rank.clone(),
                dtype: cfg.base_dtype,
            },
            q_b_proj_input_quant: quant_config(cfg.q_lora_rank.clone()),
            q_b_proj: SingleGemmKernelConfig {
                backends: cfg.single_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: partition.q_b_n,
                k: cfg.q_lora_rank.clone(),
                dtype: cfg.gemm_dtype,
            },
            kv_a_rms_norm: RmsNormKernelConfig {
                backends: cfg.rms_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.kv_lora_rank.clone(),
                dtype: cfg.base_dtype,
            },
            // Not an elementwise leaf: vLLM feeds the rope-mutated q back
            // into `self.attn`, so functionalized inductor materialises a whole
            // new q and the fused kernel reads and writes all
            // `qk_nope + rope` columns to change the rope ones. Pricing it as
            // rope-sized streaming bytes made it 74% too fast at prefill.
            main_rope: VllmMlaRopeKernelConfig {
                backends: cfg.main_rope_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: attention_heads_per_rank.clone(),
                qk_nope_head_dim: cfg.qk_nope_head_dim.clone(),
                rope_dim: cfg.rope_dim.clone(),
                max_position: cfg.rope_max_position.clone(),
                is_neox_style: cfg.rope_is_neox_style,
                input_dtype: cfg.base_dtype,
            },
            indexer_q_proj_input_quant: cfg
                .include_indexer
                .then(|| quant_config(cfg.q_lora_rank.clone()))
                .flatten(),
            indexer,
            q_absorb: BatchedGemmKernelConfig {
                backends: cfg.q_absorb_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_batches: attention_heads_per_rank.clone(),
                n: cfg.kv_lora_rank.clone(),
                k: cfg.qk_nope_head_dim.clone(),
                dtype: cfg.base_dtype,
            },
            sparse_mla: DsaSparseMlaAttentionConfig {
                sparse_attention_backends: cfg.sparse_attention_backends.clone(),
                mla_cache_append_backends: cfg.sparse_mla_cache_append_backends.clone(),
                elementwise_backends: cfg.sparse_elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: attention_heads_per_rank.clone(),
                num_kv_heads: cfg.num_kv_heads.clone(),
                selected_k: cfg.selected_k,
                latent_dim: cfg.kv_lora_rank.clone(),
                rope_dim: cfg.rope_dim.clone(),
                value_dim: cfg.kv_lora_rank.clone(),
                softmax_scale_denominator: cfg.softmax_scale_denominator,
                dtype: cfg.base_dtype,
                attention_q_dtype: cfg.sparse_attention_q_dtype,
                attention_cache_dtype: cfg.sparse_attention_cache_dtype,
                attention_output_dtype: cfg.sparse_attention_output_dtype,
                index_dtype: cfg.index_dtype.clone(),
                index_distribution: cfg.sparse_index_distribution.clone(),
                sparse_cache_layout: cfg.sparse_cache_layout.clone(),
                mla_cache_block_size: cfg.cache_block_size,
                mla_cache_format: cfg.sparse_mla_cache_format.clone(),
                decode_next_n: cfg.decode_next_n,
                launch_graph: DsaSparseMlaLaunchGraph::Separate {
                    index_remap_backends: cfg.sparse_index_remap_backends.clone(),
                },
                exact_varlen: cfg.sparse_exact_varlen.clone(),
            },
            v_up: BatchedGemmKernelConfig {
                backends: cfg.v_up_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_batches: attention_heads_per_rank.clone(),
                n: cfg.v_head_dim.clone(),
                k: cfg.kv_lora_rank.clone(),
                dtype: cfg.base_dtype,
            },
            o_proj_input_quant: quant_config(
                attention_heads_per_rank.clone() * cfg.v_head_dim.clone(),
            ),
            o_proj: SingleGemmKernelConfig {
                backends: cfg.single_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden_dim.clone(),
                k: partition.o_proj_k,
                dtype: cfg.gemm_dtype,
            },
            attention_heads_per_rank,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: VllmGlm52DsaAttnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let input_add_rms_norm = build_atomic(
            &name,
            "input_add_rms_norm",
            resolved.input_add_rms_norm.clone(),
            ResidualRmsNormKernel::build,
            bridge,
        )?;
        let fused_qkv_a_proj_input_quant = resolved
            .fused_qkv_a_proj_input_quant
            .clone()
            .map(|config| {
                build_atomic(
                    &name,
                    "fused_qkv_a_proj_input_quant",
                    config,
                    Fp8PerTokenGroupQuantKernel::build,
                    bridge,
                )
            })
            .transpose()?;
        let fused_qkv_a_proj = build_atomic(
            &name,
            "fused_qkv_a_proj",
            resolved.fused_qkv_a_proj.clone(),
            SingleGemmKernel::build,
            bridge,
        )?;
        let q_a_rms_norm = build_atomic(
            &name,
            "q_a_rms_norm",
            resolved.q_a_rms_norm.clone(),
            RmsNormKernel::build,
            bridge,
        )?;
        let q_b_proj_input_quant = resolved
            .q_b_proj_input_quant
            .clone()
            .map(|config| {
                build_atomic(
                    &name,
                    "q_b_proj_input_quant",
                    config,
                    Fp8PerTokenGroupQuantKernel::build,
                    bridge,
                )
            })
            .transpose()?;
        let q_b_proj = build_atomic(
            &name,
            "q_b_proj",
            resolved.q_b_proj.clone(),
            SingleGemmKernel::build,
            bridge,
        )?;
        let kv_a_rms_norm = build_atomic(
            &name,
            "kv_a_rms_norm",
            resolved.kv_a_rms_norm.clone(),
            RmsNormKernel::build,
            bridge,
        )?;
        let main_rope = build_atomic(
            &name,
            "main_rope",
            resolved.main_rope.clone(),
            VllmMlaRopeKernel::build,
            bridge,
        )?;
        let indexer_q_proj_input_quant = resolved
            .indexer_q_proj_input_quant
            .clone()
            .map(|config| {
                build_atomic(
                    &name,
                    "indexer_q_proj_input_quant",
                    config,
                    Fp8PerTokenGroupQuantKernel::build,
                    bridge,
                )
            })
            .transpose()?;
        let indexer = resolved
            .indexer
            .clone()
            .map(|config| DsaIndexerOp::build(format!("{name}.indexer"), config, bridge))
            .transpose()?;
        let q_absorb = build_atomic(
            &name,
            "q_absorb",
            resolved.q_absorb.clone(),
            BatchedGemmKernel::build,
            bridge,
        )?;
        let sparse_mla = DsaSparseMlaAttentionOp::build(
            format!("{name}.sparse_mla"),
            resolved.sparse_mla.clone(),
            bridge,
        )?;
        let v_up = build_atomic(
            &name,
            "v_up",
            resolved.v_up.clone(),
            BatchedGemmKernel::build,
            bridge,
        )?;
        let o_proj_input_quant = resolved
            .o_proj_input_quant
            .clone()
            .map(|config| {
                build_atomic(
                    &name,
                    "o_proj_input_quant",
                    config,
                    Fp8PerTokenGroupQuantKernel::build,
                    bridge,
                )
            })
            .transpose()?;
        let o_proj = build_atomic(
            &name,
            "o_proj",
            resolved.o_proj.clone(),
            SingleGemmKernel::build,
            bridge,
        )?;

        Ok(Self {
            name,
            input_add_rms_norm,
            fused_qkv_a_proj_input_quant,
            fused_qkv_a_proj,
            q_a_rms_norm,
            q_b_proj_input_quant,
            q_b_proj,
            kv_a_rms_norm,
            main_rope,
            indexer_q_proj_input_quant,
            indexer,
            q_absorb,
            sparse_mla,
            v_up,
            o_proj_input_quant,
            o_proj,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        // Each dense FP8 GEMM is preceded by its own activation quantisation;
        // the quant nodes are absent entirely on a BF16 GEMM.
        let mut children: Vec<CostNode> = [
            Some(self.input_add_rms_norm.compile(builder)),
            self.fused_qkv_a_proj_input_quant
                .as_ref()
                .map(|quant| quant.compile(builder)),
            Some(self.fused_qkv_a_proj.compile(builder)),
            Some(self.q_a_rms_norm.compile(builder)),
            self.q_b_proj_input_quant
                .as_ref()
                .map(|quant| quant.compile(builder)),
            Some(self.q_b_proj.compile(builder)),
            Some(self.kv_a_rms_norm.compile(builder)),
            Some(self.main_rope.compile(builder)),
        ]
        .into_iter()
        .flatten()
        .collect();
        if let Some(quant) = &self.indexer_q_proj_input_quant {
            children.push(quant.compile(builder));
        }
        if let Some(indexer) = &self.indexer {
            children.push(indexer.compile(builder));
        }
        children.extend(
            [
                Some(self.q_absorb.compile(builder)),
                Some(self.sparse_mla.compile(builder)),
                Some(self.v_up.compile(builder)),
                self.o_proj_input_quant
                    .as_ref()
                    .map(|quant| quant.compile(builder)),
                Some(self.o_proj.compile(builder)),
            ]
            .into_iter()
            .flatten(),
        );

        CostNode::Labeled {
            label: worklet_label(&self.name, &self.resolved.raw_cfg),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &VllmGlm52DsaAttnLocalWorkletInput, ev: &mut Evaluator) {
        let normalized = normalize_input(
            input,
            self.resolved.raw_cfg.decode_next_n,
            self.resolved.raw_cfg.max_model_len.get(),
        )
        .unwrap_or_else(|reason| panic!("invalid VllmGlm52DsaAttnLocalWorkletInput: {reason}"));
        let rows = normalized.active_rows;

        eval_atomic_or_zero(
            &self.input_add_rms_norm,
            ResidualRmsNormKernelInput { m: rows },
            rows == 0,
            ev,
        );
        if let Some(quant) = &self.fused_qkv_a_proj_input_quant {
            eval_atomic_or_zero(
                quant,
                Fp8PerTokenGroupQuantKernelInput { num_tokens: rows },
                rows == 0,
                ev,
            );
        }
        eval_atomic_or_zero(
            &self.fused_qkv_a_proj,
            SingleGemmKernelInput { m: rows },
            rows == 0,
            ev,
        );
        eval_atomic_or_zero(
            &self.q_a_rms_norm,
            RmsNormKernelInput { m: rows },
            rows == 0,
            ev,
        );
        if let Some(quant) = &self.q_b_proj_input_quant {
            eval_atomic_or_zero(
                quant,
                Fp8PerTokenGroupQuantKernelInput { num_tokens: rows },
                rows == 0,
                ev,
            );
        }
        eval_atomic_or_zero(
            &self.q_b_proj,
            SingleGemmKernelInput { m: rows },
            rows == 0,
            ev,
        );
        eval_atomic_or_zero(
            &self.kv_a_rms_norm,
            RmsNormKernelInput { m: rows },
            rows == 0,
            ev,
        );
        eval_atomic_or_zero(
            &self.main_rope,
            VllmMlaRopeKernelInput { num_tokens: rows },
            rows == 0,
            ev,
        );

        // The indexer's q_proj eats the same rows as the rest of the block.
        if let Some(quant) = &self.indexer_q_proj_input_quant {
            eval_atomic_or_zero(
                quant,
                Fp8PerTokenGroupQuantKernelInput { num_tokens: rows },
                rows == 0,
                ev,
            );
        }
        if let Some(indexer) = &self.indexer {
            indexer.eval(
                &DsaIndexerInput {
                    num_new_tokens: input.num_new_tokens,
                    prefill_query_key_pairs: input.prefill_query_cache_pairs.clone(),
                    decode: normalized.indexer_decode.clone(),
                },
                ev,
            );
        }

        eval_atomic_or_zero(
            &self.q_absorb,
            BatchedGemmKernelInput { m: rows },
            rows == 0,
            ev,
        );
        self.sparse_mla.eval(
            &DsaSparseMlaAttentionInput {
                num_new_tokens: input.num_new_tokens,
                prefill_query_cache_pairs: input.prefill_query_cache_pairs.clone(),
                decode_query_cache: normalized.sparse_decode,
                decode_context_lens: normalized.sparse_decode_context_lens.clone(),
            },
            ev,
        );
        eval_atomic_or_zero(
            &self.v_up,
            BatchedGemmKernelInput { m: rows },
            rows == 0,
            ev,
        );
        if let Some(quant) = &self.o_proj_input_quant {
            eval_atomic_or_zero(
                quant,
                Fp8PerTokenGroupQuantKernelInput { num_tokens: rows },
                rows == 0,
                ev,
            );
        }
        eval_atomic_or_zero(
            &self.o_proj,
            SingleGemmKernelInput { m: rows },
            rows == 0,
            ev,
        );
    }
}

fn validate_config(cfg: &VllmGlm52DsaAttnLocalWorkletConfig) -> Result<(), String> {
    if cfg.tp_size == 0 {
        return Err("tp_size must be positive".to_string());
    }
    let tp = u32::from(cfg.tp_size);
    if cfg.num_attention_heads.get() % tp != 0 {
        return Err(format!(
            "num_attention_heads {} must be divisible by tp_size {tp}",
            cfg.num_attention_heads
        ));
    }
    for (name, actual, required) in [
        ("hidden_dim", cfg.hidden_dim.get(), HIDDEN_DIM),
        (
            "num_attention_heads",
            cfg.num_attention_heads.get(),
            NUM_ATTENTION_HEADS,
        ),
        ("num_kv_heads", cfg.num_kv_heads.get(), NUM_KV_HEADS),
        ("q_lora_rank", cfg.q_lora_rank.get(), Q_LORA_RANK),
        ("kv_lora_rank", cfg.kv_lora_rank.get(), KV_LORA_RANK),
        (
            "qk_nope_head_dim",
            cfg.qk_nope_head_dim.get(),
            QK_NOPE_HEAD_DIM,
        ),
        ("rope_dim", cfg.rope_dim.get(), ROPE_DIM),
        ("v_head_dim", cfg.v_head_dim.get(), V_HEAD_DIM),
        (
            "model_num_index_heads",
            cfg.model_num_index_heads.get(),
            MODEL_INDEX_HEADS,
        ),
        (
            "profile_num_index_heads",
            cfg.profile_num_index_heads.get(),
            PROFILE_INDEX_HEADS,
        ),
        ("index_head_dim", cfg.index_head_dim.get(), INDEX_HEAD_DIM),
        ("selected_k", cfg.selected_k, SELECTED_K),
        ("cache_block_size", cfg.cache_block_size, CACHE_BLOCK_SIZE),
        ("quant_block_size", cfg.quant_block_size, QUANT_BLOCK_SIZE),
        (
            "softmax_scale_denominator",
            cfg.softmax_scale_denominator,
            SOFTMAX_SCALE_DENOMINATOR,
        ),
    ] {
        if actual != required {
            return Err(format!("{name} must be {required}, got {actual}"));
        }
    }
    let max_model_len = cfg.max_model_len.get();
    let logits_row_stride = cfg.logits_row_stride.get();
    if max_model_len == 0 {
        return Err("max_model_len must be positive".to_string());
    }
    if logits_row_stride < max_model_len {
        return Err(format!(
            "logits_row_stride {logits_row_stride} must be at least max_model_len {max_model_len}"
        ));
    }
    if cfg.decode_next_n == 0 {
        return Err("decode_next_n must be positive".to_string());
    }
    for (name, actual, required) in [
        ("base_dtype", cfg.base_dtype, DType::Bf16),
        ("index_cache_dtype", cfg.index_cache_dtype, DType::Fp8E4m3),
        ("index_q_dtype", cfg.index_q_dtype, DType::Fp8E4m3),
        ("scale_dtype", cfg.scale_dtype, DType::Fp32),
        ("weight_dtype", cfg.weight_dtype, DType::Fp32),
        ("logits_dtype", cfg.logits_dtype, DType::Fp32),
    ] {
        if actual != required {
            return Err(format!(
                "{name} must be {}, got {}",
                required.as_str(),
                actual.as_str()
            ));
        }
    }
    if !matches!(cfg.gemm_dtype, DType::Bf16 | DType::Fp8E4m3) {
        return Err(format!(
            "gemm_dtype must be {} or {}, got {}",
            DType::Bf16.as_str(),
            DType::Fp8E4m3.as_str(),
            cfg.gemm_dtype.as_str()
        ));
    }
    if cfg.index_dtype != "int32" {
        return Err(format!(
            "index_dtype must be int32, got {:?}",
            cfg.index_dtype
        ));
    }
    for (name, actual, required) in [
        (
            "index_scale_format",
            cfg.index_scale_format.as_str(),
            "ue8m0",
        ),
        (
            "index_cache_format",
            cfg.index_cache_format.as_str(),
            "page_planar_fp8_fp32_scale",
        ),
        (
            "index_prefill_span_mode",
            cfg.index_prefill_span_mode.as_str(),
            "single_causal_tail",
        ),
        (
            "index_decode_context_mode",
            cfg.index_decode_context_mode.as_str(),
            "uniform",
        ),
        (
            "index_decode_page_mapping",
            cfg.index_decode_page_mapping.as_str(),
            "unique_scattered",
        ),
        (
            "sparse_mla_cache_format",
            cfg.sparse_mla_cache_format.as_str(),
            "plain",
        ),
    ] {
        if actual != required {
            return Err(format!("{name} must be {required}, got {actual:?}"));
        }
    }
    match (
        cfg.sparse_attention_q_dtype,
        cfg.sparse_attention_cache_dtype,
        cfg.sparse_attention_output_dtype,
        cfg.sparse_cache_layout.as_str(),
        cfg.sparse_exact_varlen.as_ref(),
    ) {
        (DType::Bf16, DType::Bf16, DType::Bf16, "token_major_mqa_bf16_latent_rope", None) => {}
        (
            DType::Fp8E4m3,
            DType::Fp8E4m3,
            DType::Bf16,
            "hnd_paged_mqa_fp8_latent_rope",
            Some(exact),
        ) if exact.max_model_len == max_model_len => {}
        (_, _, _, _, Some(exact)) if exact.max_model_len != max_model_len => {
            return Err(format!(
                "sparse exact-varlen max_model_len {} must equal worklet max_model_len {max_model_len}",
                exact.max_model_len
            ));
        }
        _ => {
            return Err(format!(
                "unsupported sparse attention dtype/layout/varlen tuple ({}, {}, {}, {:?}, exact={})",
                cfg.sparse_attention_q_dtype.as_str(),
                cfg.sparse_attention_cache_dtype.as_str(),
                cfg.sparse_attention_output_dtype.as_str(),
                cfg.sparse_cache_layout,
                cfg.sparse_exact_varlen.is_some(),
            ));
        }
    }
    if !matches!(
        cfg.sparse_index_distribution.as_str(),
        "recent_contiguous" | "unique_scattered_pages" | "clustered_pages" | "uniform_stride"
    ) {
        return Err(format!(
            "unsupported sparse_index_distribution {:?}",
            cfg.sparse_index_distribution
        ));
    }
    Ok(())
}

fn worklet_label(name: &str, cfg: &VllmGlm52DsaAttnLocalWorkletConfig) -> String {
    format!(
        "{name} (VllmGlm52DsaAttnLocalWorklet) [local (1 GPU); indexer={}; heads={}; next_n={}]",
        if cfg.include_indexer {
            "full"
        } else {
            "shared"
        },
        cfg.num_attention_heads,
        cfg.decode_next_n
    )
}

fn build_atomic<K, C, F>(
    parent: &str,
    suffix: &str,
    config: C,
    build: F,
    bridge: &PerfApiBridge,
) -> Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError>,
{
    let name = format!("{parent}.{suffix}");
    Ok(Op::new(
        name.clone(),
        Arc::new(build(name, config, bridge)?),
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
    use super::*;

    fn cfg(include_indexer: bool, decode_next_n: u32) -> VllmGlm52DsaAttnLocalWorkletConfig {
        VllmGlm52DsaAttnLocalWorkletConfig {
            fp8_quant_backends: vec!["flashinfer_trtllm"],
            include_indexer,
            tp_size: 1,
            residual_rms_norm_backends: vec!["vllm_cuda"],
            rms_norm_backends: vec!["flashinfer"],
            single_gemm_backends: vec!["torch"],
            main_rope_backends: vec!["vllm_inductor"],
            q_absorb_backends: vec!["torch_mla_q_absorb_glm52"],
            v_up_backends: vec!["torch_mla_v_up_glm52"],
            indexer_gemm_backends: vec!["torch_indexer"],
            indexer_elementwise_backends: vec!["triton_indexer"],
            index_cache_append_backends: vec!["vllm_cuda"],
            index_prefill_logits_backends: vec!["deepgemm_fp8"],
            index_prefill_topk_backends: vec!["vllm_cuda"],
            index_decode_logits_backends: vec!["deepgemm_fp8"],
            index_decode_topk_backends: vec!["vllm_cuda"],
            sparse_attention_backends: vec!["vllm_flashmla_bf16"],
            sparse_mla_cache_append_backends: vec!["vllm_cuda"],
            sparse_elementwise_backends: vec!["triton_sparse"],
            sparse_index_remap_backends: Vec::new(),
            gpu_name: "NVIDIA H200".to_string(),
            hidden_dim: Dim::param("hidden_dim", 6144),
            num_attention_heads: Dim::param("num_attention_heads", 64),
            num_kv_heads: Dim::param("num_kv_heads", 1),
            q_lora_rank: Dim::param("q_lora_rank", 2048),
            kv_lora_rank: Dim::param("kv_lora_rank", 512),
            qk_nope_head_dim: Dim::param("qk_nope_head_dim", 192),
            rope_dim: Dim::param("qk_rope_head_dim", 64),
            v_head_dim: Dim::param("v_head_dim", 256),
            model_num_index_heads: Dim::param("model_num_index_heads", 32),
            profile_num_index_heads: Dim::param("profile_num_index_heads", 64),
            index_head_dim: Dim::param("index_head_dim", 128),
            selected_k: 2048,
            max_model_len: Dim::param("max_model_len", MAX_MODEL_LEN),
            rope_max_position: Dim::param("max_position_embeddings", 1_048_576),
            logits_row_stride: Dim::param("logits_row_stride", LOGITS_ROW_STRIDE),
            cache_block_size: 64,
            quant_block_size: 128,
            softmax_scale_denominator: 16,
            base_dtype: DType::Bf16,
            gemm_dtype: DType::Bf16,
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
            sparse_index_distribution: "recent_contiguous".to_string(),
            sparse_cache_layout: "token_major_mqa_bf16_latent_rope".to_string(),
            sparse_mla_cache_format: "plain".to_string(),
            sparse_attention_q_dtype: DType::Bf16,
            sparse_attention_cache_dtype: DType::Bf16,
            sparse_attention_output_dtype: DType::Bf16,
            sparse_exact_varlen: None,
            decode_next_n,
        }
    }

    fn b200_tp4_cfg(include_indexer: bool) -> VllmGlm52DsaAttnLocalWorkletConfig {
        let mut config = cfg(include_indexer, 1);
        config.tp_size = 4;
        config.gpu_name = "NVIDIA B200".to_string();
        config.max_model_len = Dim::param("max_model_len", 8192);
        config.logits_row_stride = Dim::param("logits_row_stride", 8192);
        config.sparse_attention_backends = vec!["flashinfer_trtllm_fp8"];
        config.sparse_index_distribution = "recent_contiguous".to_string();
        config.sparse_cache_layout = "hnd_paged_mqa_fp8_latent_rope".to_string();
        config.sparse_attention_q_dtype = DType::Fp8E4m3;
        config.sparse_attention_cache_dtype = DType::Fp8E4m3;
        config.sparse_attention_output_dtype = DType::Bf16;
        config.sparse_index_remap_backends = vec!["vllm_triton"];
        config.sparse_exact_varlen = Some(DsaSparseMlaExactVarlenConfig {
            prefill_backends: vec!["flashinfer_trtllm_fp8"],
            max_model_len: 8192,
            prefill_index_distribution: "recent_contiguous".to_string(),
            page_table_mapping: Some("request_contiguous".to_string()),
        });
        config
    }

    #[test]
    fn source_order_is_exact_and_excludes_other_model_sections() {
        assert_eq!(SOURCE_ORDER_WITH_INDEXER.len(), 14);
        for expected in [
            "fused_qkv_a_proj_input_quant",
            "q_b_proj_input_quant",
            "o_proj_input_quant",
        ] {
            assert!(SOURCE_ORDER_WITH_INDEXER.contains(&expected));
        }
        assert_eq!(SOURCE_ORDER_INDEX_SHARE.len(), 13);
        assert!(!SOURCE_ORDER_INDEX_SHARE.contains(&"indexer"));
        assert!(SOURCE_ORDER_INDEX_SHARE.contains(&"sparse_mla"));
        for forbidden in ["ffn", "router", "all_reduce", "all_to_all", "mtp"] {
            assert!(!SOURCE_ORDER_WITH_INDEXER
                .iter()
                .any(|child| child.contains(forbidden)));
        }
    }

    #[test]
    fn resolve_bakes_every_exact_projection_bmm_and_sparse_shape() {
        let r = VllmGlm52DsaAttnLocalWorklet::resolve_config(&cfg(true, 1));
        assert_eq!(r.input_add_rms_norm.hidden, 6144);
        assert_eq!(r.input_add_rms_norm.dtype, DType::Bf16);
        assert_eq!(r.fused_qkv_a_proj.n, 2624);
        assert_eq!(r.fused_qkv_a_proj.k, 6144);
        assert_eq!(r.q_a_rms_norm.hidden, 2048);
        assert_eq!(r.q_b_proj.n, 16384);
        assert_eq!(r.q_b_proj.k, 2048);
        assert_eq!(r.kv_a_rms_norm.hidden, 512);
        // The fused rope leaf carries the full q row width, not the rope
        // slice: the inductor fusion rewrites every one of q's columns.
        assert_eq!(r.main_rope.num_heads, 64);
        assert_eq!(r.main_rope.qk_nope_head_dim, 192);
        assert_eq!(r.main_rope.rope_dim, 64);
        assert!(!r.main_rope.is_neox_style);
        assert_eq!(r.q_absorb.num_batches, 64);
        assert_eq!(r.q_absorb.n, 512);
        assert_eq!(r.q_absorb.k, 192);
        assert_eq!(r.v_up.num_batches, 64);
        assert_eq!(r.v_up.n, 256);
        assert_eq!(r.v_up.k, 512);
        assert_eq!(r.o_proj.n, 6144);
        assert_eq!(r.o_proj.k, 16384);

        assert_eq!(r.sparse_mla.num_heads, 64);
        assert_eq!(r.sparse_mla.num_kv_heads, 1);
        assert_eq!(r.sparse_mla.latent_dim, 512);
        assert_eq!(r.sparse_mla.value_dim, 512);
        assert_ne!(r.sparse_mla.value_dim, r.raw_cfg.v_head_dim);
        assert_eq!(r.sparse_mla.mla_cache_block_size, 64);
        assert_eq!(r.sparse_mla.mla_cache_format, "plain");
        assert_eq!(r.sparse_mla.dtype, DType::Bf16);
    }

    #[test]
    fn b200_tp4_shards_only_attention_heads_and_uses_exact_sparse_recipe() {
        let config = b200_tp4_cfg(true);
        let r = VllmGlm52DsaAttnLocalWorklet::resolve_config(&config);

        assert_eq!(r.attention_heads_per_rank, 16);
        assert_eq!(r.q_b_proj.n, 4096);
        assert_eq!(r.main_rope.num_heads, 16);
        assert_eq!(r.q_absorb.num_batches, 16);
        assert_eq!(r.sparse_mla.num_heads, 16);
        assert_eq!(r.v_up.num_batches, 16);
        assert_eq!(r.o_proj.k, 4096);

        let indexer = r.indexer.expect("full-index layer");
        assert_eq!(indexer.model_num_index_heads, 32);
        assert_eq!(indexer.profile_num_index_heads, 64);

        assert_eq!(r.sparse_mla.attention_q_dtype, DType::Fp8E4m3);
        assert_eq!(r.sparse_mla.attention_cache_dtype, DType::Fp8E4m3);
        assert_eq!(r.sparse_mla.attention_output_dtype, DType::Bf16);
        assert_eq!(r.sparse_mla.mla_cache_format, "plain");
        let exact = r
            .sparse_mla
            .exact_varlen
            .expect("B200 FP8 sparse attention requires exact varlen");
        assert_eq!(config.sparse_index_remap_backends, vec!["vllm_triton"]);
        assert_eq!(exact.prefill_backends, vec!["flashinfer_trtllm_fp8"]);
        assert_eq!(exact.max_model_len, 8192);
    }

    #[test]
    fn tp_and_exact_varlen_contracts_fail_during_pure_resolution() {
        let mut zero_tp = cfg(true, 1);
        zero_tp.tp_size = 0;
        assert!(validate_config(&zero_tp)
            .unwrap_err()
            .contains("tp_size must be positive"));

        let mut indivisible = cfg(true, 1);
        indivisible.tp_size = 3;
        assert!(validate_config(&indivisible)
            .unwrap_err()
            .contains("num_attention_heads"));

        let mut mismatched_context = b200_tp4_cfg(true);
        mismatched_context
            .sparse_exact_varlen
            .as_mut()
            .unwrap()
            .max_model_len = 4096;
        assert!(validate_config(&mismatched_context)
            .unwrap_err()
            .contains("must equal worklet max_model_len"));

        let mut missing_exact = b200_tp4_cfg(true);
        missing_exact.sparse_exact_varlen = None;
        assert!(validate_config(&missing_exact)
            .unwrap_err()
            .contains("unsupported sparse attention"));
    }

    #[test]
    fn configured_context_size_is_accepted_and_stride_is_bounded() {
        let config = b200_tp4_cfg(true);
        assert!(validate_config(&config).is_ok());

        let mut short_stride = config;
        short_stride.logits_row_stride = Dim::param("logits_row_stride", 4096);
        assert!(validate_config(&short_stride)
            .unwrap_err()
            .contains("at least max_model_len"));
    }

    #[test]
    fn backend_roles_propagate_without_silent_sharing() {
        let r = VllmGlm52DsaAttnLocalWorklet::resolve_config(&cfg(true, 1));
        assert_eq!(r.input_add_rms_norm.backends, vec!["vllm_cuda"]);
        assert_eq!(r.q_a_rms_norm.backends, vec!["flashinfer"]);
        assert_eq!(r.fused_qkv_a_proj.backends, vec!["torch"]);
        assert_eq!(r.main_rope.backends, vec!["vllm_inductor"]);
        assert_eq!(r.q_absorb.backends, vec!["torch_mla_q_absorb_glm52"]);
        assert_eq!(r.v_up.backends, vec!["torch_mla_v_up_glm52"]);
        assert_eq!(r.sparse_mla.elementwise_backends, vec!["triton_sparse"]);
        assert_eq!(
            r.sparse_mla.sparse_attention_backends,
            vec!["vllm_flashmla_bf16"]
        );
        let indexer = r.indexer.unwrap();
        assert_eq!(indexer.gemm_backends, vec!["torch_indexer"]);
        assert_eq!(indexer.elementwise_backends, vec!["triton_indexer"]);
    }

    #[test]
    fn indexer_selection_is_structural_and_preserves_h32_h64_distinction() {
        let full = VllmGlm52DsaAttnLocalWorklet::resolve_config(&cfg(true, 1));
        let indexer = full.indexer.as_ref().expect("full-index layer");
        assert_eq!(indexer.model_num_index_heads, 32);
        assert_eq!(indexer.profile_num_index_heads, 64);
        assert_eq!(indexer.next_n, 1);
        assert!(worklet_label("layer.attn", &full.raw_cfg).contains("indexer=full"));

        let shared = VllmGlm52DsaAttnLocalWorklet::resolve_config(&cfg(false, 1));
        assert!(shared.indexer.is_none());
        assert_eq!(shared.sparse_mla.num_heads, 64);
        assert!(worklet_label("layer.attn", &shared.raw_cfg).contains("indexer=shared"));
    }

    #[test]
    fn any_positive_next_n_reaches_both_compound_ops() {
        // A k=5 draft verifies six rows per request. The width is forwarded to
        // both compound ops unchanged; neither this worklet nor they treat six
        // differently from two.
        for next_n in [1, 2, 3, 6] {
            let r = VllmGlm52DsaAttnLocalWorklet::resolve_config(&cfg(true, next_n));
            assert_eq!(r.indexer.as_ref().unwrap().next_n, next_n);
            assert_eq!(r.sparse_mla.decode_next_n, next_n);
            assert!(validate_config(&cfg(true, next_n)).is_ok());
        }
    }

    #[test]
    fn normalize_maps_prefill_and_decode_shapes_with_padding() {
        let input = VllmGlm52DsaAttnLocalWorkletInput {
            num_new_tokens: 48,
            prefill_query_cache_pairs: vec![(8, 128), (16, 256)],
            decode: Some(VllmGlm52DsaAttnLocalDecodeInput {
                batch_size: 12,
                context_len: 8192,
                context_lens: None,
                requires_padding: true,
            }),
        };
        let n = normalize_input(&input, 2, MAX_MODEL_LEN).unwrap();
        assert_eq!(n.active_rows, 48);
        assert_eq!(n.sparse_decode, Some((24, 8192)));
        let decode = n.indexer_decode.unwrap();
        assert_eq!(decode.batch_size, 12);
        assert_eq!(decode.context_len, 8192);
        assert!(decode.requires_padding);
    }

    #[test]
    fn normalize_preserves_exact_decode_lengths_for_sparse_attention() {
        let input = VllmGlm52DsaAttnLocalWorkletInput {
            num_new_tokens: 4,
            prefill_query_cache_pairs: Vec::new(),
            decode: Some(VllmGlm52DsaAttnLocalDecodeInput {
                batch_size: 4,
                context_len: 190,
                context_lens: Some(vec![12, 190, 12, 190]),
                requires_padding: false,
            }),
        };
        let normalized = normalize_input(&input, 1, MAX_MODEL_LEN).unwrap();
        assert_eq!(normalized.sparse_decode, Some((4, 190)));
        assert_eq!(
            normalized.sparse_decode_context_lens,
            Some(vec![12, 190, 12, 190])
        );
    }

    #[test]
    fn zero_case_and_num_new_token_equality_are_enforced() {
        let zero = normalize_input(
            &VllmGlm52DsaAttnLocalWorkletInput::default(),
            1,
            MAX_MODEL_LEN,
        )
        .unwrap();
        assert_eq!(zero.active_rows, 0);
        assert!(zero.indexer_decode.is_none());
        assert!(zero.sparse_decode.is_none());

        let mismatched = VllmGlm52DsaAttnLocalWorkletInput {
            num_new_tokens: 7,
            prefill_query_cache_pairs: vec![(8, 8)],
            decode: None,
        };
        assert!(normalize_input(&mismatched, 1, MAX_MODEL_LEN)
            .unwrap_err()
            .contains("must equal active query rows"));
    }

    #[test]
    fn malformed_prefill_decode_and_row_overflow_fail_clearly() {
        for pair in [(0, 1), (1, 0), (2, 1)] {
            let input = VllmGlm52DsaAttnLocalWorkletInput {
                prefill_query_cache_pairs: vec![pair],
                ..Default::default()
            };
            assert!(normalize_input(&input, 1, MAX_MODEL_LEN).is_err());
        }
        for (batch_size, context_len) in [(0, 1), (1, 0), (1, MAX_MODEL_LEN + 1)] {
            let input = VllmGlm52DsaAttnLocalWorkletInput {
                decode: Some(VllmGlm52DsaAttnLocalDecodeInput {
                    batch_size,
                    context_len,
                    context_lens: None,
                    requires_padding: false,
                }),
                ..Default::default()
            };
            assert!(normalize_input(&input, 2, MAX_MODEL_LEN).is_err());
        }
        let multiply_overflow = VllmGlm52DsaAttnLocalWorkletInput {
            decode: Some(VllmGlm52DsaAttnLocalDecodeInput {
                batch_size: u32::MAX,
                context_len: 1,
                context_lens: None,
                requires_padding: false,
            }),
            ..Default::default()
        };
        assert!(normalize_input(&multiply_overflow, 2, MAX_MODEL_LEN)
            .unwrap_err()
            .contains("overflows u32"));

        let sum_overflow = VllmGlm52DsaAttnLocalWorkletInput {
            num_new_tokens: 0,
            prefill_query_cache_pairs: vec![(u32::MAX, u32::MAX), (1, 1)],
            decode: None,
        };
        assert!(normalize_input(&sum_overflow, 1, MAX_MODEL_LEN)
            .unwrap_err()
            .contains("overflows u32"));
    }

    #[test]
    fn unsupported_glm_identity_fails_during_pure_resolution() {
        for mutate in [
            |cfg: &mut VllmGlm52DsaAttnLocalWorkletConfig| cfg.num_attention_heads = 32.into(),
            |cfg: &mut VllmGlm52DsaAttnLocalWorkletConfig| cfg.model_num_index_heads = 64.into(),
            |cfg: &mut VllmGlm52DsaAttnLocalWorkletConfig| cfg.profile_num_index_heads = 32.into(),
            |cfg: &mut VllmGlm52DsaAttnLocalWorkletConfig| cfg.base_dtype = DType::Fp32,
            |cfg: &mut VllmGlm52DsaAttnLocalWorkletConfig| cfg.index_cache_dtype = DType::Bf16,
            |cfg: &mut VllmGlm52DsaAttnLocalWorkletConfig| cfg.selected_k = 1024,
            |cfg: &mut VllmGlm52DsaAttnLocalWorkletConfig| {
                cfg.sparse_mla_cache_format = "fp8".to_string()
            },
            |cfg: &mut VllmGlm52DsaAttnLocalWorkletConfig| {
                cfg.sparse_index_distribution = "unknown".to_string()
            },
        ] {
            let mut config = cfg(true, 1);
            mutate(&mut config);
            assert!(validate_config(&config).is_err());
        }
        assert!(validate_config(&cfg(true, 0)).is_err());
    }
}
