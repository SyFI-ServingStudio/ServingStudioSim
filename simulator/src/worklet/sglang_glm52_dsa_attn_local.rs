//! GLM-5.2 TP-local DSA attention in SGLang launch granularity.
//!
//! This sibling is intentionally separate from the vLLM worklet: SGLang uses
//! a fused RoPE/FP8/query-concat launch, fused indexer prologue launches, and
//! sparse MLA without standalone concat/remap launches. Only request-shape
//! normalization and the neutral L2 compound ops are shared.

use std::sync::Arc;

use crate::op::attention::{
    DsaIndexerConfig, DsaIndexerInput, DsaIndexerLaunchGraph, DsaIndexerOp,
    DsaSparseMlaAttentionConfig, DsaSparseMlaAttentionInput, DsaSparseMlaAttentionOp,
    DsaSparseMlaExactVarlenConfig, DsaSparseMlaLaunchGraph,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    BatchedGemmKernel, BatchedGemmKernelConfig, BatchedGemmKernelInput, MlaRopeQuantizeFp8Kernel,
    MlaRopeQuantizeFp8KernelConfig, MlaRopeQuantizeFp8KernelInput, ResidualRmsNormKernel,
    ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput, RmsNormKernel, RmsNormKernelConfig,
    RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

use super::glm52_dsa_attn_common::{
    normalize_glm52_dsa_attn_input as normalize_input, resolve_glm52_dsa_attn_tp_partition,
    Glm52DsaAttnLocalDecodeInput, Glm52DsaAttnLocalInput,
};

const HIDDEN_DIM: u32 = 6_144;
const NUM_ATTENTION_HEADS: u32 = 64;
const NUM_KV_HEADS: u32 = 1;
const Q_LORA_RANK: u32 = 2_048;
const KV_LORA_RANK: u32 = 512;
const QK_NOPE_HEAD_DIM: u32 = 192;
const ROPE_DIM: u32 = 64;
const V_HEAD_DIM: u32 = 256;
const MODEL_INDEX_HEADS: u32 = 32;
const PROFILE_INDEX_HEADS: u32 = 32;
const INDEX_HEAD_DIM: u32 = 128;
const SELECTED_K: u32 = 2_048;
const CACHE_BLOCK_SIZE: u32 = 64;
const QUANT_BLOCK_SIZE: u32 = 128;
const SOFTMAX_SCALE_DENOMINATOR: u32 = 16;

pub type SglangGlm52DsaAttnLocalDecodeInput = Glm52DsaAttnLocalDecodeInput;
pub type SglangGlm52DsaAttnLocalWorkletInput = Glm52DsaAttnLocalInput;

#[derive(Clone, Debug)]
pub struct SglangGlm52DsaAttnLocalWorkletConfig {
    pub include_indexer: bool,
    pub tp_size: u16,
    pub residual_rms_norm_backends: Vec<&'static str>,
    pub rms_norm_backends: Vec<&'static str>,
    pub projection_gemm_backends: Vec<&'static str>,
    pub output_gemm_backends: Vec<&'static str>,
    pub main_rope_backends: Vec<&'static str>,
    pub q_absorb_backends: Vec<&'static str>,
    pub v_up_backends: Vec<&'static str>,
    pub indexer_gemm_backends: Vec<&'static str>,
    pub indexer_elementwise_backends: Vec<&'static str>,
    pub indexer_q_rope_backends: Vec<&'static str>,
    pub index_cache_append_backends: Vec<&'static str>,
    pub index_prefill_logits_backends: Vec<&'static str>,
    pub index_prefill_topk_backends: Vec<&'static str>,
    pub index_decode_logits_backends: Vec<&'static str>,
    pub index_decode_topk_backends: Vec<&'static str>,
    pub sparse_attention_backends: Vec<&'static str>,
    pub sparse_mla_cache_append_backends: Vec<&'static str>,
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
    pub rope_max_position: Dim,
    pub logits_row_stride: Dim,
    pub cache_block_size: u32,
    pub quant_block_size: u32,
    pub softmax_scale_denominator: u32,
    pub base_dtype: DType,
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
    pub rope_is_neox_style: bool,
    pub index_clean_logits: bool,
    pub sparse_index_distribution: String,
    pub sparse_cache_layout: String,
    pub sparse_mla_cache_format: String,
    pub sparse_attention_q_dtype: DType,
    pub sparse_attention_cache_dtype: DType,
    pub sparse_attention_output_dtype: DType,
    pub sparse_exact_varlen: DsaSparseMlaExactVarlenConfig,
    pub decode_next_n: u32,
}

#[derive(Clone, Debug)]
pub struct SglangGlm52DsaAttnLocalWorkletResolved {
    pub raw_cfg: SglangGlm52DsaAttnLocalWorkletConfig,
    pub input_add_rms_norm: ResidualRmsNormKernelConfig,
    pub fused_qkv_a_proj: SingleGemmKernelConfig,
    pub q_a_rms_norm: RmsNormKernelConfig,
    pub q_b_proj: SingleGemmKernelConfig,
    pub kv_a_rms_norm: RmsNormKernelConfig,
    pub main_rope: MlaRopeQuantizeFp8KernelConfig,
    pub indexer: Option<DsaIndexerConfig>,
    pub q_absorb: BatchedGemmKernelConfig,
    pub sparse_mla: DsaSparseMlaAttentionConfig,
    pub v_up: BatchedGemmKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
}

pub struct SglangGlm52DsaAttnLocalWorklet {
    pub name: String,
    pub input_add_rms_norm: Op<ResidualRmsNormKernel>,
    pub fused_qkv_a_proj: Op<SingleGemmKernel>,
    pub q_a_rms_norm: Op<RmsNormKernel>,
    pub q_b_proj: Op<SingleGemmKernel>,
    pub kv_a_rms_norm: Op<RmsNormKernel>,
    pub main_rope: Op<MlaRopeQuantizeFp8Kernel>,
    pub indexer: Option<DsaIndexerOp>,
    pub q_absorb: Op<BatchedGemmKernel>,
    pub sparse_mla: DsaSparseMlaAttentionOp,
    pub v_up: Op<BatchedGemmKernel>,
    pub o_proj: Op<SingleGemmKernel>,
    resolved: SglangGlm52DsaAttnLocalWorkletResolved,
}

impl SglangGlm52DsaAttnLocalWorklet {
    pub fn resolve_config(
        cfg: &SglangGlm52DsaAttnLocalWorkletConfig,
    ) -> SglangGlm52DsaAttnLocalWorkletResolved {
        validate_config(cfg).unwrap_or_else(|reason| {
            panic!("invalid SglangGlm52DsaAttnLocalWorkletConfig: {reason}")
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
        let heads = partition.attention_heads_per_rank.clone();

        let indexer = cfg.include_indexer.then(|| DsaIndexerConfig {
            gemm_backends: cfg.indexer_gemm_backends.clone(),
            elementwise_backends: cfg.indexer_elementwise_backends.clone(),
            q_rope_backends: cfg.indexer_q_rope_backends.clone(),
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
            gemm_dtype: cfg.base_dtype,
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
            launch_graph: DsaIndexerLaunchGraph::FusedQueryRopeAndKeyStore,
        });

        SglangGlm52DsaAttnLocalWorkletResolved {
            input_add_rms_norm: ResidualRmsNormKernelConfig {
                backends: cfg.residual_rms_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden_dim.clone(),
                dtype: cfg.base_dtype,
            },
            fused_qkv_a_proj: SingleGemmKernelConfig {
                backends: cfg.projection_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: partition.fused_qkv_a_n,
                k: cfg.hidden_dim.clone(),
                dtype: cfg.base_dtype,
            },
            q_a_rms_norm: RmsNormKernelConfig {
                backends: cfg.rms_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.q_lora_rank.clone(),
                dtype: cfg.base_dtype,
            },
            q_b_proj: SingleGemmKernelConfig {
                backends: cfg.projection_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: partition.q_b_n,
                k: cfg.q_lora_rank.clone(),
                dtype: cfg.base_dtype,
            },
            kv_a_rms_norm: RmsNormKernelConfig {
                backends: cfg.rms_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.kv_lora_rank.clone(),
                dtype: cfg.base_dtype,
            },
            main_rope: MlaRopeQuantizeFp8KernelConfig {
                backends: cfg.main_rope_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: heads.clone(),
                kv_lora_rank: cfg.kv_lora_rank.clone(),
                rope_dim: cfg.rope_dim.clone(),
                max_position: cfg.rope_max_position.clone(),
                is_neox_style: cfg.rope_is_neox_style,
                input_dtype: cfg.base_dtype,
                quant_dtype: cfg.sparse_attention_cache_dtype,
            },
            indexer,
            q_absorb: BatchedGemmKernelConfig {
                backends: cfg.q_absorb_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_batches: heads.clone(),
                n: cfg.kv_lora_rank.clone(),
                k: cfg.qk_nope_head_dim.clone(),
                dtype: cfg.base_dtype,
            },
            sparse_mla: DsaSparseMlaAttentionConfig {
                sparse_attention_backends: cfg.sparse_attention_backends.clone(),
                mla_cache_append_backends: cfg.sparse_mla_cache_append_backends.clone(),
                // This provider fuses query concat and index remap into adjacent
                // launches, so the shared L2 config must carry no ignored
                // elementwise backend identity.
                elementwise_backends: Vec::new(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: heads.clone(),
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
                launch_graph: DsaSparseMlaLaunchGraph::FusedQueryConcatAndIndexRemap,
                exact_varlen: Some(cfg.sparse_exact_varlen.clone()),
            },
            v_up: BatchedGemmKernelConfig {
                backends: cfg.v_up_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_batches: heads.clone(),
                n: cfg.v_head_dim.clone(),
                k: cfg.kv_lora_rank.clone(),
                dtype: cfg.base_dtype,
            },
            o_proj: SingleGemmKernelConfig {
                backends: cfg.output_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden_dim.clone(),
                k: partition.o_proj_k,
                dtype: cfg.base_dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: SglangGlm52DsaAttnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let input_add_rms_norm = build_atomic(
            &name,
            "input_add_rms_norm",
            resolved.input_add_rms_norm.clone(),
            ResidualRmsNormKernel::build,
            bridge,
        )?;
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
            MlaRopeQuantizeFp8Kernel::build,
            bridge,
        )?;
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
            fused_qkv_a_proj,
            q_a_rms_norm,
            q_b_proj,
            kv_a_rms_norm,
            main_rope,
            indexer,
            q_absorb,
            sparse_mla,
            v_up,
            o_proj,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let mut children = vec![
            self.input_add_rms_norm.compile(builder),
            self.fused_qkv_a_proj.compile(builder),
            self.q_a_rms_norm.compile(builder),
            self.q_b_proj.compile(builder),
            self.kv_a_rms_norm.compile(builder),
            self.main_rope.compile(builder),
        ];
        if let Some(indexer) = &self.indexer {
            children.push(indexer.compile(builder));
        }
        children.extend([
            self.q_absorb.compile(builder),
            self.sparse_mla.compile(builder),
            self.v_up.compile(builder),
            self.o_proj.compile(builder),
        ]);
        CostNode::Labeled {
            label: worklet_label(&self.name, &self.resolved.raw_cfg),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &SglangGlm52DsaAttnLocalWorkletInput, ev: &mut Evaluator) {
        self.eval_with_input_norm(input, true, ev);
    }

    pub fn eval_with_input_norm(
        &self,
        input: &SglangGlm52DsaAttnLocalWorkletInput,
        include_input_norm: bool,
        ev: &mut Evaluator,
    ) {
        let normalized = normalize_input(
            input,
            self.resolved.raw_cfg.decode_next_n,
            self.resolved.raw_cfg.max_model_len.get(),
        )
        .unwrap_or_else(|reason| panic!("invalid SglangGlm52DsaAttnLocalWorkletInput: {reason}"));
        let rows = normalized.active_rows;
        eval_atomic_or_zero(
            &self.input_add_rms_norm,
            ResidualRmsNormKernelInput { m: rows },
            rows == 0 || !include_input_norm,
            ev,
        );
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
            MlaRopeQuantizeFp8KernelInput { num_tokens: rows },
            rows == 0,
            ev,
        );
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
                decode_context_lens: normalized.sparse_decode_context_lens,
            },
            ev,
        );
        eval_atomic_or_zero(
            &self.v_up,
            BatchedGemmKernelInput { m: rows },
            rows == 0,
            ev,
        );
        eval_atomic_or_zero(
            &self.o_proj,
            SingleGemmKernelInput { m: rows },
            rows == 0,
            ev,
        );
    }
}

fn validate_config(cfg: &SglangGlm52DsaAttnLocalWorkletConfig) -> Result<(), String> {
    if cfg.tp_size == 0 || cfg.num_attention_heads.get() % u32::from(cfg.tp_size) != 0 {
        return Err(format!(
            "num_attention_heads {} must be divisible by positive tp_size {}",
            cfg.num_attention_heads, cfg.tp_size
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
    if cfg.max_model_len.get() == 0 || cfg.logits_row_stride.get() < cfg.max_model_len.get() {
        return Err(
            "max_model_len must be positive and no larger than logits_row_stride".to_string(),
        );
    }
    if cfg.sparse_exact_varlen.max_model_len != cfg.max_model_len.get() {
        return Err(
            "sparse exact-varlen max_model_len must equal worklet max_model_len".to_string(),
        );
    }
    if !matches!(cfg.decode_next_n, 1 | 2) {
        return Err("decode_next_n must be 1 or 2".to_string());
    }
    if cfg.base_dtype != DType::Bf16
        || cfg.index_cache_dtype != DType::Fp8E4m3
        || cfg.index_q_dtype != DType::Fp8E4m3
        || cfg.scale_dtype != DType::Fp32
        || cfg.weight_dtype != DType::Fp32
        || cfg.logits_dtype != DType::Fp32
        || cfg.sparse_attention_q_dtype != DType::Fp8E4m3
        || cfg.sparse_attention_cache_dtype != DType::Fp8E4m3
        || cfg.sparse_attention_output_dtype != DType::Bf16
    {
        return Err("unsupported SGLang GLM-5.2 dtype identity".to_string());
    }
    if cfg.index_dtype != "int32"
        || cfg.index_scale_format != "fp32"
        || cfg.index_cache_format != "page_planar_fp8_fp32_scale"
        || cfg.index_prefill_span_mode != "single_causal_tail"
        || cfg.index_decode_context_mode != "uniform"
        || cfg.index_decode_page_mapping != "unique_scattered"
        || cfg.sparse_cache_layout != "hnd_paged_mqa_fp8_latent_rope"
        || cfg.sparse_mla_cache_format != "plain"
    {
        return Err("unsupported SGLang GLM-5.2 layout identity".to_string());
    }
    Ok(())
}

fn worklet_label(name: &str, cfg: &SglangGlm52DsaAttnLocalWorkletConfig) -> String {
    format!("{name} (SglangGlm52DsaAttnLocalWorklet) [TP{} rank-local; indexer={}; heads={}; next_n={}]", cfg.tp_size, if cfg.include_indexer { "full" } else { "shared" }, cfg.num_attention_heads, cfg.decode_next_n)
}

fn build_atomic<K, C, F>(
    name: &str,
    suffix: &str,
    config: C,
    build: F,
    bridge: &PerfApiBridge,
) -> Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError>,
{
    let slot = format!("{name}.{suffix}");
    Ok(Op::new(
        slot.clone(),
        Arc::new(build(slot, config, bridge)?),
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

    fn cfg(include_indexer: bool) -> SglangGlm52DsaAttnLocalWorkletConfig {
        SglangGlm52DsaAttnLocalWorkletConfig {
            include_indexer,
            tp_size: 4,
            residual_rms_norm_backends: vec!["vllm_cuda"],
            rms_norm_backends: vec!["flashinfer"],
            projection_gemm_backends: vec!["sglang_fused_a_auto"],
            output_gemm_backends: vec!["sglang_bf16_auto"],
            main_rope_backends: vec!["flashinfer"],
            q_absorb_backends: vec!["torch_mla_q_absorb_glm52"],
            v_up_backends: vec!["torch_mla_v_up_glm52"],
            indexer_gemm_backends: vec!["sglang_bf16_auto"],
            indexer_elementwise_backends: vec!["triton"],
            indexer_q_rope_backends: vec!["sglang_cuda"],
            index_cache_append_backends: vec!["sglang_fused_norm_rope_store"],
            index_prefill_logits_backends: vec!["deepgemm_fp8"],
            index_prefill_topk_backends: vec!["sglang_cuda"],
            index_decode_logits_backends: vec!["deepgemm_fp8"],
            index_decode_topk_backends: vec!["vllm_cuda"],
            sparse_attention_backends: vec!["flashinfer_trtllm_fp8"],
            sparse_mla_cache_append_backends: vec!["sglang_cuda"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden_dim: 6144.into(),
            num_attention_heads: 64.into(),
            num_kv_heads: 1.into(),
            q_lora_rank: 2048.into(),
            kv_lora_rank: 512.into(),
            qk_nope_head_dim: 192.into(),
            rope_dim: 64.into(),
            v_head_dim: 256.into(),
            model_num_index_heads: 32.into(),
            profile_num_index_heads: 32.into(),
            index_head_dim: 128.into(),
            selected_k: 2048,
            max_model_len: 8192.into(),
            rope_max_position: 1_048_576.into(),
            logits_row_stride: 8192.into(),
            cache_block_size: 64,
            quant_block_size: 128,
            softmax_scale_denominator: 16,
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
            sparse_index_distribution: "recent_contiguous".to_string(),
            sparse_cache_layout: "hnd_paged_mqa_fp8_latent_rope".to_string(),
            sparse_mla_cache_format: "plain".to_string(),
            sparse_attention_q_dtype: DType::Fp8E4m3,
            sparse_attention_cache_dtype: DType::Fp8E4m3,
            sparse_attention_output_dtype: DType::Bf16,
            sparse_exact_varlen: DsaSparseMlaExactVarlenConfig {
                prefill_backends: vec!["flashinfer_trtllm_fp8"],
                max_model_len: 8192,
                prefill_index_distribution: "recent_contiguous".to_string(),
                page_table_mapping: None,
            },
            decode_next_n: 1,
        }
    }

    #[test]
    fn resolve_uses_tp4_h32_and_only_sglang_launch_graphs() {
        let r = SglangGlm52DsaAttnLocalWorklet::resolve_config(&cfg(true));
        assert_eq!(r.main_rope.num_heads, 16);
        assert_eq!(r.q_b_proj.n, 4096);
        assert_eq!(r.o_proj.k, 4096);
        assert_eq!(r.main_rope.num_heads, 16);
        let indexer = r.indexer.as_ref().unwrap();
        assert_eq!(indexer.profile_num_index_heads, 32);
        assert_eq!(
            indexer.launch_graph,
            DsaIndexerLaunchGraph::FusedQueryRopeAndKeyStore
        );
        assert_eq!(
            r.sparse_mla.launch_graph,
            DsaSparseMlaLaunchGraph::FusedQueryConcatAndIndexRemap
        );
        assert_eq!(
            r.sparse_mla.sparse_attention_backends,
            vec!["flashinfer_trtllm_fp8"]
        );
    }

    #[test]
    fn index_share_omits_only_the_indexer() {
        let r = SglangGlm52DsaAttnLocalWorklet::resolve_config(&cfg(false));
        assert!(r.indexer.is_none());
        assert_eq!(r.main_rope.backends, vec!["flashinfer"]);
        assert_eq!(r.sparse_mla.mla_cache_append_backends, vec!["sglang_cuda"]);
    }

    #[test]
    fn provider_identity_rejects_vllm_h64_and_ue8m0() {
        let mut config = cfg(true);
        config.profile_num_index_heads = 64.into();
        assert!(validate_config(&config).is_err());
        config.profile_num_index_heads = 32.into();
        config.index_scale_format = "ue8m0".to_string();
        assert!(validate_config(&config).is_err());
    }
}
