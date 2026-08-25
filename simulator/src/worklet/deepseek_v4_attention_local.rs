//! DeepSeek V4 DP-local attention, preserving vLLM's fused kernel boundaries.
//!
//! Prefill and decode share one projection spine. Their sparse/indexer kernels
//! remain distinct leaves because vLLM launches different physical callables.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    DeepseekV4FusedInvRopeFp8QuantKernel, DeepseekV4FusedInvRopeFp8QuantKernelConfig,
    DeepseekV4FusedInvRopeFp8QuantKernelInput, DeepseekV4FusedQKvRmsnormKernel,
    DeepseekV4FusedQKvRmsnormKernelConfig, DeepseekV4FusedQKvRmsnormKernelInput,
    DeepseekV4IndexerMqaLogitsDecodeKernel, DeepseekV4IndexerMqaLogitsDecodeKernelConfig,
    DeepseekV4IndexerMqaLogitsDecodeKernelInput, DeepseekV4IndexerMqaLogitsPrefillKernel,
    DeepseekV4IndexerMqaLogitsPrefillKernelConfig, DeepseekV4IndexerPrefillKernelInput,
    DeepseekV4IndexerQRopeQuantKernel, DeepseekV4IndexerQRopeQuantKernelConfig,
    DeepseekV4IndexerQRopeQuantKernelInput, DeepseekV4IndexerTopkDecodeKernel,
    DeepseekV4IndexerTopkDecodeKernelConfig, DeepseekV4IndexerTopkDecodeKernelInput,
    DeepseekV4IndexerTopkPrefillKernel, DeepseekV4IndexerTopkPrefillKernelConfig,
    DeepseekV4QnormRopeKvInsertKernel, DeepseekV4QnormRopeKvInsertKernelConfig,
    DeepseekV4QnormRopeKvInsertKernelInput, DeepseekV4SparseAttnCompressStoreKernel,
    DeepseekV4SparseAttnCompressStoreKernelConfig, DeepseekV4SparseAttnCompressStoreKernelInput,
    DeepseekV4SparseMlaDecodeKernel, DeepseekV4SparseMlaDecodeKernelConfig,
    DeepseekV4SparseMlaDecodeKernelInput, DeepseekV4SparseMlaPrefillKernel,
    DeepseekV4SparseMlaPrefillKernelConfig, DeepseekV4SparseMlaPrefillKernelInput,
    Fp8PerTokenGroupQuantKernel, Fp8PerTokenGroupQuantKernelConfig,
    Fp8PerTokenGroupQuantKernelInput, GemmFp32OutputKernel, GemmFp32OutputKernelConfig,
    GemmFp32OutputKernelInput, MhcFusedPostPreRmsNormKernel, MhcPreRmsNormKernel,
    MhcRmsNormKernelConfig, MhcRmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

const HIDDEN_SIZE: u32 = 4096;
const NUM_ATTENTION_HEADS: u32 = 64;
const NUM_KV_HEADS: u32 = 1;
const HEAD_DIM: u32 = 512;
const ROPE_DIM: u32 = 64;
const Q_LORA_RANK: u32 = 1024;
const O_LORA_RANK: u32 = 1024;
const O_GROUPS: u32 = 8;
const INDEX_HEADS: u32 = 64;
const INDEX_HEAD_DIM: u32 = 128;
const SELECTED_K: u32 = 512;
const SWA_WINDOW: u32 = 128;
const FP8_GROUP_SIZE: u32 = 128;
const KV_BLOCK_SIZE: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeepseekV4AttentionEntry {
    Layer0Pre,
    LaterLayerFusedPostPre,
}

#[derive(Clone, Debug)]
pub struct DeepseekV4AttentionLocalWorkletConfig {
    pub entry: DeepseekV4AttentionEntry,
    pub compress_ratio: u32,
    pub planner_mode: String,
    pub serialize_streams: bool,
    pub max_model_len: u32,
    pub max_num_batched_tokens: u32,
    pub hidden_size: Dim,
    pub hc_mult: u32,
    pub num_attention_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub rope_dim: Dim,
    pub q_lora_rank: Dim,
    pub o_lora_rank: Dim,
    pub o_groups: Dim,
    pub index_num_heads: Dim,
    pub index_head_dim: Dim,
    pub selected_k: u32,
    pub activation_dtype: DType,
    pub projection_dtype: DType,
    pub cache_dtype: DType,
    pub gpu_name: String,
    pub mhc_backends: Vec<&'static str>,
    pub quant_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub fp32_gemm_backends: Vec<&'static str>,
    pub fused_q_kv_rmsnorm_backends: Vec<&'static str>,
    pub qnorm_rope_kv_insert_backends: Vec<&'static str>,
    pub compressor_store_backends: Vec<&'static str>,
    pub indexer_compressor_store_backends: Vec<&'static str>,
    pub sparse_prefill_backends: Vec<&'static str>,
    pub sparse_decode_backends: Vec<&'static str>,
    pub inverse_rope_quant_backends: Vec<&'static str>,
    pub indexer_q_rope_quant_backends: Vec<&'static str>,
    pub indexer_prefill_logits_backends: Vec<&'static str>,
    pub indexer_prefill_topk_backends: Vec<&'static str>,
    pub indexer_decode_logits_backends: Vec<&'static str>,
    pub indexer_decode_topk_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct DeepseekV4AttentionLocalWorkletResolved {
    pub raw_cfg: DeepseekV4AttentionLocalWorkletConfig,
    pub entry: MhcRmsNormKernelConfig,
    pub fused_qkv_input_quant: Fp8PerTokenGroupQuantKernelConfig,
    pub fused_qkv: SingleGemmKernelConfig,
    pub fused_q_kv_rmsnorm: DeepseekV4FusedQKvRmsnormKernelConfig,
    pub outer_compressor_proj: Option<GemmFp32OutputKernelConfig>,
    pub indexer_weights_proj: Option<SingleGemmKernelConfig>,
    pub indexer_compressor_proj: Option<GemmFp32OutputKernelConfig>,
    pub q_b_input_quant: Fp8PerTokenGroupQuantKernelConfig,
    pub q_b: SingleGemmKernelConfig,
    pub qnorm_rope_kv_insert: DeepseekV4QnormRopeKvInsertKernelConfig,
    pub compressor_store: Option<DeepseekV4SparseAttnCompressStoreKernelConfig>,
    pub indexer_compressor_store: Option<DeepseekV4SparseAttnCompressStoreKernelConfig>,
    pub indexer_q_input_quant: Option<Fp8PerTokenGroupQuantKernelConfig>,
    pub indexer_q: Option<SingleGemmKernelConfig>,
    pub indexer_q_rope_quant: Option<DeepseekV4IndexerQRopeQuantKernelConfig>,
    pub indexer_prefill_logits: Option<DeepseekV4IndexerMqaLogitsPrefillKernelConfig>,
    pub indexer_prefill_topk: Option<DeepseekV4IndexerTopkPrefillKernelConfig>,
    pub indexer_decode_logits: Option<DeepseekV4IndexerMqaLogitsDecodeKernelConfig>,
    pub indexer_decode_topk: Option<DeepseekV4IndexerTopkDecodeKernelConfig>,
    pub sparse_prefill: DeepseekV4SparseMlaPrefillKernelConfig,
    pub sparse_decode: DeepseekV4SparseMlaDecodeKernelConfig,
    pub inverse_rope_quant: DeepseekV4FusedInvRopeFp8QuantKernelConfig,
    pub wo_a: SingleGemmKernelConfig,
    pub wo_b_input_quant: Fp8PerTokenGroupQuantKernelConfig,
    pub wo_b: SingleGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct DeepseekV4AttentionLocalWorkletInput {
    /// Rows consumed by the common projection spine, including DP padding.
    pub num_tokens: u32,
    /// Actual rows whose KV state is inserted; excludes DP padding.
    pub num_insert_tokens: u32,
    pub prefill_query_context_pairs: Vec<(u32, u32)>,
    /// Resident KV length before each one-token decode row.
    pub decode_kv_lens: Vec<u32>,
}

pub struct DeepseekV4AttentionLocalWorklet {
    pub name: String,
    pub entry_pre: Option<Op<MhcPreRmsNormKernel>>,
    pub entry_fused: Option<Op<MhcFusedPostPreRmsNormKernel>>,
    pub fused_qkv_input_quant: Op<Fp8PerTokenGroupQuantKernel>,
    pub fused_qkv: Op<SingleGemmKernel>,
    pub fused_q_kv_rmsnorm: Op<DeepseekV4FusedQKvRmsnormKernel>,
    pub outer_compressor_proj: Option<Op<GemmFp32OutputKernel>>,
    pub indexer_weights_proj: Option<Op<SingleGemmKernel>>,
    pub indexer_compressor_proj: Option<Op<GemmFp32OutputKernel>>,
    pub q_b_input_quant: Op<Fp8PerTokenGroupQuantKernel>,
    pub q_b: Op<SingleGemmKernel>,
    pub qnorm_rope_kv_insert: Op<DeepseekV4QnormRopeKvInsertKernel>,
    pub compressor_store: Option<Op<DeepseekV4SparseAttnCompressStoreKernel>>,
    pub indexer_compressor_store: Option<Op<DeepseekV4SparseAttnCompressStoreKernel>>,
    pub indexer_q_input_quant: Option<Op<Fp8PerTokenGroupQuantKernel>>,
    pub indexer_q: Option<Op<SingleGemmKernel>>,
    pub indexer_q_rope_quant: Option<Op<DeepseekV4IndexerQRopeQuantKernel>>,
    pub indexer_prefill_logits: Option<Op<DeepseekV4IndexerMqaLogitsPrefillKernel>>,
    pub indexer_prefill_topk: Option<Op<DeepseekV4IndexerTopkPrefillKernel>>,
    pub indexer_decode_logits: Option<Op<DeepseekV4IndexerMqaLogitsDecodeKernel>>,
    pub indexer_decode_topk: Option<Op<DeepseekV4IndexerTopkDecodeKernel>>,
    pub sparse_prefill: Op<DeepseekV4SparseMlaPrefillKernel>,
    pub sparse_decode: Op<DeepseekV4SparseMlaDecodeKernel>,
    pub inverse_rope_quant: Op<DeepseekV4FusedInvRopeFp8QuantKernel>,
    pub wo_a: Op<SingleGemmKernel>,
    pub wo_b_input_quant: Op<Fp8PerTokenGroupQuantKernel>,
    pub wo_b: Op<SingleGemmKernel>,
    resolved: DeepseekV4AttentionLocalWorkletResolved,
}

impl DeepseekV4AttentionLocalWorklet {
    pub fn resolve_config(
        config: &DeepseekV4AttentionLocalWorkletConfig,
    ) -> DeepseekV4AttentionLocalWorkletResolved {
        validate_config(config).unwrap_or_else(|reason| {
            panic!("invalid DeepseekV4AttentionLocalWorkletConfig: {reason}")
        });
        let c4 = config.compress_ratio == 4;
        let compressed = config.compress_ratio != 1;
        let quant = |hidden_size: Dim| Fp8PerTokenGroupQuantKernelConfig {
            backends: config.quant_backends.clone(),
            gpu_name: config.gpu_name.clone(),
            hidden_size,
            group_size: FP8_GROUP_SIZE,
            input_dtype: config.activation_dtype,
            scale_format: "ue8m0_column_major".to_string(),
        };
        let gemm = |n: Dim, k: Dim| SingleGemmKernelConfig {
            backends: config.gemm_backends.clone(),
            gpu_name: config.gpu_name.clone(),
            n,
            k,
            dtype: config.projection_dtype,
        };
        let fp32_gemm = |n: Dim| GemmFp32OutputKernelConfig {
            backends: config.fp32_gemm_backends.clone(),
            gpu_name: config.gpu_name.clone(),
            n,
            k: config.hidden_size.clone(),
            input_dtype: config.activation_dtype,
        };
        let head_width = config.num_attention_heads.clone() * config.head_dim.clone();
        let wo_a_k = head_width.clone() / config.o_groups.clone();
        let wo_b_width = config.o_groups.clone() * config.o_lora_rank.clone();
        let extra_index_capacity = match config.compress_ratio {
            1 => 0,
            4 => SELECTED_K,
            128 => config.max_model_len.div_ceil(128).div_ceil(128) * 128,
            _ => unreachable!(),
        };

        DeepseekV4AttentionLocalWorkletResolved {
            raw_cfg: config.clone(),
            entry: MhcRmsNormKernelConfig {
                backends: config.mhc_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                hidden_size: config.hidden_size.clone(),
                hc_mult: config.hc_mult,
                hidden_dtype: config.activation_dtype,
            },
            fused_qkv_input_quant: quant(config.hidden_size.clone()),
            fused_qkv: gemm(
                config.q_lora_rank.clone() + config.head_dim.clone(),
                config.hidden_size.clone(),
            ),
            fused_q_kv_rmsnorm: DeepseekV4FusedQKvRmsnormKernelConfig {
                backends: config.fused_q_kv_rmsnorm_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                q_dim: (config.q_lora_rank.clone() + config.head_dim.clone()),
                kv_dim: config.head_dim.clone(),
                rms_eps_bits: 1.0e-6_f64.to_bits(),
                dtype: config.activation_dtype,
            },
            outer_compressor_proj: compressed
                .then(|| fp32_gemm(config.head_dim.clone() * if c4 { 4 } else { 2 })),
            indexer_weights_proj: c4
                .then(|| gemm(config.index_num_heads.clone(), config.hidden_size.clone())),
            indexer_compressor_proj: c4.then(|| fp32_gemm(config.index_head_dim.clone() * 4)),
            q_b_input_quant: quant(config.q_lora_rank.clone()),
            q_b: gemm(head_width.clone(), config.q_lora_rank.clone()),
            qnorm_rope_kv_insert: DeepseekV4QnormRopeKvInsertKernelConfig {
                backends: config.qnorm_rope_kv_insert_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                num_heads: config.num_attention_heads.get(),
                padded_heads: config.num_attention_heads.get(),
                head_dim: config.head_dim.clone(),
                rope_dim: config.rope_dim.clone(),
                block_size: KV_BLOCK_SIZE,
                rms_eps_bits: 1.0e-6_f64.to_bits(),
                input_dtype: config.activation_dtype,
                kv_dtype: config.cache_dtype,
                cache_dtype: "fp8_ds_mla".to_string(),
                cache_layout: "block_segregated_data_then_scales".to_string(),
                scale_format: "ue8m0".to_string(),
            },
            compressor_store: compressed.then(|| DeepseekV4SparseAttnCompressStoreKernelConfig {
                backends: config.compressor_store_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                compress_ratio: config.compress_ratio,
                num_kv_heads: config.num_kv_heads.get(),
                head_dim: config.head_dim.clone(),
                rope_head_dim: config.rope_dim.clone(),
                logical_block_size: KV_BLOCK_SIZE,
                state_dtype: DType::Fp32,
                norm_dtype: config.activation_dtype,
                kv_dtype: config.cache_dtype,
                cache_dtype: "fp8_ds_mla".to_string(),
                cache_layout: "block_segregated_data_then_scales".to_string(),
                scale_format: "ue8m0".to_string(),
            }),
            indexer_compressor_store: c4.then(|| DeepseekV4SparseAttnCompressStoreKernelConfig {
                backends: config.indexer_compressor_store_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                compress_ratio: 4,
                num_kv_heads: 1,
                head_dim: config.index_head_dim.clone(),
                rope_head_dim: config.rope_dim.clone(),
                logical_block_size: KV_BLOCK_SIZE,
                state_dtype: DType::Fp32,
                norm_dtype: config.activation_dtype,
                kv_dtype: config.cache_dtype,
                cache_dtype: "fp8_indexer".to_string(),
                cache_layout: "block_segregated_data_then_scales".to_string(),
                scale_format: "fp32_per_token".to_string(),
            }),
            indexer_q_input_quant: c4.then(|| quant(config.q_lora_rank.clone())),
            indexer_q: c4.then(|| {
                gemm(
                    config.index_num_heads.clone() * config.index_head_dim.clone(),
                    config.q_lora_rank.clone(),
                )
            }),
            indexer_q_rope_quant: c4.then(|| DeepseekV4IndexerQRopeQuantKernelConfig {
                backends: config.indexer_q_rope_quant_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                num_heads: config.index_num_heads.clone(),
                head_dim: config.index_head_dim.clone(),
                rope_dim: config.rope_dim.clone(),
                max_model_len: config.max_model_len,
                max_num_batched_tokens: config.max_num_batched_tokens,
                positions_dtype: "int64".to_string(),
                q_dtype: config.activation_dtype,
                rope_dtype: DType::Fp32,
                weight_dtype: config.activation_dtype,
                q_output_dtype: config.cache_dtype,
                weight_output_dtype: DType::Fp32,
                rope_style: "gptj_interleaved_trailing".to_string(),
                quant_mode: "per_token_head_fp8_pow2_ceil_folded_weight".to_string(),
            }),
            indexer_prefill_logits: c4.then(|| DeepseekV4IndexerMqaLogitsPrefillKernelConfig {
                backends: config.indexer_prefill_logits_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                max_model_len: config.max_model_len,
                max_num_batched_tokens: config.max_num_batched_tokens,
                max_logits_bytes: 512 * 1024 * 1024,
                compress_ratio: 4,
                num_heads: config.index_num_heads.clone(),
                head_dim: config.index_head_dim.clone(),
                q_dtype: config.cache_dtype,
                k_dtype: config.cache_dtype,
                k_scale_dtype: DType::Fp32,
                weight_dtype: DType::Fp32,
                output_dtype: DType::Fp32,
                clean_logits: false,
            }),
            indexer_prefill_topk: c4.then(|| DeepseekV4IndexerTopkPrefillKernelConfig {
                backends: config.indexer_prefill_topk_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                max_model_len: config.max_model_len,
                max_num_batched_tokens: config.max_num_batched_tokens,
                max_logits_bytes: 512 * 1024 * 1024,
                compress_ratio: 4,
                top_k: config.selected_k,
                logits_dtype: DType::Fp32,
                index_dtype: "int32".to_string(),
            }),
            indexer_decode_logits: c4.then(|| DeepseekV4IndexerMqaLogitsDecodeKernelConfig {
                backends: config.indexer_decode_logits_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                next_n: 1,
                max_model_len: config.max_model_len.into(),
                num_heads: config.index_num_heads.clone(),
                head_dim: config.index_head_dim.clone(),
                block_size: 64,
                q_dtype: config.cache_dtype,
                cache_dtype: config.cache_dtype,
                scale_dtype: DType::Fp32,
                weight_dtype: DType::Fp32,
                output_dtype: DType::Fp32,
                context_mode: "max_ragged".to_string(),
                page_mapping: "request_contiguous".to_string(),
                cache_format: "fp8_e4m3_ue8m0".to_string(),
                clean_logits: false,
            }),
            indexer_decode_topk: c4.then(|| DeepseekV4IndexerTopkDecodeKernelConfig {
                backends: config.indexer_decode_topk_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                next_n: 1,
                max_model_len: config.max_model_len.into(),
                top_k: config.selected_k,
                logits_row_stride: config.max_model_len.into(),
                logits_dtype: DType::Fp32,
                index_dtype: "int32".to_string(),
                context_mode: "max_ragged".to_string(),
            }),
            sparse_prefill: DeepseekV4SparseMlaPrefillKernelConfig {
                backends: config.sparse_prefill_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                max_model_len: config.max_model_len,
                max_num_batched_tokens: config.max_num_batched_tokens,
                prefill_chunk_size: 4,
                compress_ratio: config.compress_ratio,
                window_size: SWA_WINDOW,
                selected_k: if config.compress_ratio == 1 {
                    0
                } else {
                    SELECTED_K
                },
                selected_index_pattern: "request_local_topk_plus_swa".to_string(),
                num_heads: config.num_attention_heads.clone(),
                num_kv_heads: config.num_kv_heads.clone(),
                head_dim: config.head_dim.clone(),
                value_dim: config.head_dim.clone(),
                q_dtype: config.activation_dtype,
                cache_dtype: DType::Bf16,
                index_dtype: "int32".to_string(),
                output_dtype: config.activation_dtype,
                cache_layout: "request_slot_major_flat_mqa_bf16_d512".to_string(),
            },
            sparse_decode: DeepseekV4SparseMlaDecodeKernelConfig {
                backends: config.sparse_decode_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                num_heads: config.num_attention_heads.clone(),
                num_kv_heads: config.num_kv_heads.clone(),
                head_dim: config.head_dim.clone(),
                value_dim: config.head_dim.clone(),
                swa_window: SWA_WINDOW,
                extra_index_capacity,
                compress_ratio: config.compress_ratio,
                q_dtype: config.activation_dtype,
                cache_dtype: config.cache_dtype,
                output_dtype: config.activation_dtype,
                planner_mode: config.planner_mode.clone(),
            },
            inverse_rope_quant: DeepseekV4FusedInvRopeFp8QuantKernelConfig {
                backends: config.inverse_rope_quant_backends.clone(),
                gpu_name: config.gpu_name.clone(),
            },
            wo_a: gemm(config.o_lora_rank.clone(), wo_a_k),
            wo_b_input_quant: quant(wo_b_width.clone()),
            wo_b: gemm(config.hidden_size.clone(), wo_b_width),
        }
    }

    pub fn build(
        name: String,
        resolved: DeepseekV4AttentionLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        macro_rules! build_op {
            ($field:ident, $kernel:ty, $suffix:literal) => {{
                let op_name = format!("{name}.{}", $suffix);
                Op::new(
                    op_name.clone(),
                    Arc::new(<$kernel>::build(op_name, resolved.$field.clone(), bridge)?),
                )
            }};
        }
        macro_rules! build_optional {
            ($field:ident, $kernel:ty, $suffix:literal) => {
                match resolved.$field.clone() {
                    Some(kernel_config) => {
                        let op_name = format!("{name}.{}", $suffix);
                        Some(Op::new(
                            op_name.clone(),
                            Arc::new(<$kernel>::build(op_name, kernel_config, bridge)?),
                        ))
                    }
                    None => None,
                }
            };
        }
        let entry_pre = (resolved.raw_cfg.entry == DeepseekV4AttentionEntry::Layer0Pre)
            .then(|| {
                let op_name = format!("{name}.entry_mhc_pre");
                MhcPreRmsNormKernel::build(op_name.clone(), resolved.entry.clone(), bridge)
                    .map(|kernel| Op::new(op_name, Arc::new(kernel)))
            })
            .transpose()?;
        let entry_fused = (resolved.raw_cfg.entry
            == DeepseekV4AttentionEntry::LaterLayerFusedPostPre)
            .then(|| {
                let op_name = format!("{name}.entry_mhc_fused_post_pre");
                MhcFusedPostPreRmsNormKernel::build(op_name.clone(), resolved.entry.clone(), bridge)
                    .map(|kernel| Op::new(op_name, Arc::new(kernel)))
            })
            .transpose()?;
        Ok(Self {
            name: name.clone(),
            entry_pre,
            entry_fused,
            fused_qkv_input_quant: build_op!(
                fused_qkv_input_quant,
                Fp8PerTokenGroupQuantKernel,
                "input.fused_qkv_input_quant"
            ),
            fused_qkv: build_op!(fused_qkv, SingleGemmKernel, "input.fused_qkv_proj"),
            fused_q_kv_rmsnorm: build_op!(
                fused_q_kv_rmsnorm,
                DeepseekV4FusedQKvRmsnormKernel,
                "main.fused_q_kv_rmsnorm"
            ),
            outer_compressor_proj: build_optional!(
                outer_compressor_proj,
                GemmFp32OutputKernel,
                "compressor.outer_projection"
            ),
            indexer_weights_proj: build_optional!(
                indexer_weights_proj,
                SingleGemmKernel,
                "indexer.weights_projection"
            ),
            indexer_compressor_proj: build_optional!(
                indexer_compressor_proj,
                GemmFp32OutputKernel,
                "indexer.compressor_projection"
            ),
            q_b_input_quant: build_op!(
                q_b_input_quant,
                Fp8PerTokenGroupQuantKernel,
                "main.q_b_input_quant"
            ),
            q_b: build_op!(q_b, SingleGemmKernel, "main.q_b_projection"),
            qnorm_rope_kv_insert: build_op!(
                qnorm_rope_kv_insert,
                DeepseekV4QnormRopeKvInsertKernel,
                "main.qnorm_rope_kv_insert"
            ),
            compressor_store: build_optional!(
                compressor_store,
                DeepseekV4SparseAttnCompressStoreKernel,
                "compressor.sparse_attn_compress_store"
            ),
            indexer_compressor_store: build_optional!(
                indexer_compressor_store,
                DeepseekV4SparseAttnCompressStoreKernel,
                "indexer.compressor.sparse_attn_compress_store"
            ),
            indexer_q_input_quant: build_optional!(
                indexer_q_input_quant,
                Fp8PerTokenGroupQuantKernel,
                "indexer.q_input_quant"
            ),
            indexer_q: build_optional!(indexer_q, SingleGemmKernel, "indexer.q_projection"),
            indexer_q_rope_quant: build_optional!(
                indexer_q_rope_quant,
                DeepseekV4IndexerQRopeQuantKernel,
                "indexer.q_rope_quant"
            ),
            indexer_prefill_logits: build_optional!(
                indexer_prefill_logits,
                DeepseekV4IndexerMqaLogitsPrefillKernel,
                "indexer.prefill_mqa_logits"
            ),
            indexer_prefill_topk: build_optional!(
                indexer_prefill_topk,
                DeepseekV4IndexerTopkPrefillKernel,
                "indexer.prefill_topk"
            ),
            indexer_decode_logits: build_optional!(
                indexer_decode_logits,
                DeepseekV4IndexerMqaLogitsDecodeKernel,
                "indexer.decode_mqa_logits"
            ),
            indexer_decode_topk: build_optional!(
                indexer_decode_topk,
                DeepseekV4IndexerTopkDecodeKernel,
                "indexer.decode_topk"
            ),
            sparse_prefill: build_op!(
                sparse_prefill,
                DeepseekV4SparseMlaPrefillKernel,
                "attention.prefill_sparse_mla"
            ),
            sparse_decode: build_op!(
                sparse_decode,
                DeepseekV4SparseMlaDecodeKernel,
                "attention.decode_sparse_mla"
            ),
            inverse_rope_quant: build_op!(
                inverse_rope_quant,
                DeepseekV4FusedInvRopeFp8QuantKernel,
                "output.inverse_rope_quant"
            ),
            wo_a: build_op!(wo_a, SingleGemmKernel, "output.wo_a_projection"),
            wo_b_input_quant: build_op!(
                wo_b_input_quant,
                Fp8PerTokenGroupQuantKernel,
                "output.wo_b_input_quant"
            ),
            wo_b: build_op!(wo_b, SingleGemmKernel, "output.wo_b_projection"),
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let entry = match (&self.entry_pre, &self.entry_fused) {
            (Some(pre), None) => pre.compile(builder),
            (None, Some(fused)) => fused.compile(builder),
            _ => unreachable!("resolution creates exactly one entry leaf"),
        };
        // vLLM has a hard join between its projection fanout and fused
        // normalization. Keep the two fanouts separate so non-serial mode
        // cannot overlap work across that barrier.
        let mut projection_branches = vec![CostNode::Sum(vec![
            self.fused_qkv_input_quant.compile(builder),
            self.fused_qkv.compile(builder),
        ])];
        if let Some(projection) = &self.outer_compressor_proj {
            projection_branches.push(projection.compile(builder));
        }
        if let (Some(weights), Some(compressor)) =
            (&self.indexer_weights_proj, &self.indexer_compressor_proj)
        {
            projection_branches.push(weights.compile(builder));
            projection_branches.push(compressor.compile(builder));
        }
        let stage_a =
            compose_parallel(projection_branches, self.resolved.raw_cfg.serialize_streams);

        let main_q_path = CostNode::Sum(vec![
            self.q_b_input_quant.compile(builder),
            self.q_b.compile(builder),
            self.qnorm_rope_kv_insert.compile(builder),
        ]);
        let mut post_norm_branches = vec![main_q_path];
        if let Some(store) = &self.compressor_store {
            post_norm_branches.push(store.compile(builder));
        }
        if let (Some(input_quant), Some(q), Some(q_rope), Some(indexer_store)) = (
            &self.indexer_q_input_quant,
            &self.indexer_q,
            &self.indexer_q_rope_quant,
            &self.indexer_compressor_store,
        ) {
            let indexer_fanout = compose_parallel(
                vec![
                    CostNode::Sum(vec![
                        input_quant.compile(builder),
                        q.compile(builder),
                        q_rope.compile(builder),
                    ]),
                    indexer_store.compile(builder),
                ],
                self.resolved.raw_cfg.serialize_streams,
            );
            post_norm_branches.push(CostNode::Sum(vec![
                indexer_fanout,
                self.indexer_prefill_logits
                    .as_ref()
                    .unwrap()
                    .compile(builder),
                self.indexer_prefill_topk.as_ref().unwrap().compile(builder),
                self.indexer_decode_logits
                    .as_ref()
                    .unwrap()
                    .compile(builder),
                self.indexer_decode_topk.as_ref().unwrap().compile(builder),
            ]));
        }
        let stage_b = compose_parallel(post_norm_branches, self.resolved.raw_cfg.serialize_streams);
        CostNode::Labeled {
            label: format!(
                "{} (DeepseekV4AttentionLocalWorklet) [DP-local; C{}; streams={}]",
                self.name,
                self.resolved.raw_cfg.compress_ratio,
                if self.resolved.raw_cfg.serialize_streams {
                    "serial"
                } else {
                    "overlapped"
                }
            ),
            child: Box::new(CostNode::Sum(vec![
                entry,
                stage_a,
                self.fused_q_kv_rmsnorm.compile(builder),
                stage_b,
                self.sparse_prefill.compile(builder),
                self.sparse_decode.compile(builder),
                self.inverse_rope_quant.compile(builder),
                self.wo_a.compile(builder),
                self.wo_b_input_quant.compile(builder),
                self.wo_b.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &DeepseekV4AttentionLocalWorkletInput, evaluator: &mut Evaluator) {
        let work = normalize_input(input, &self.resolved.raw_cfg).unwrap_or_else(|reason| {
            panic!("invalid DeepseekV4AttentionLocalWorkletInput: {reason}")
        });
        let no_rows = input.num_tokens == 0;
        let no_insert_rows = input.num_insert_tokens == 0;
        match (&self.entry_pre, &self.entry_fused) {
            (Some(pre), None) => eval_or_zero(pre, work.mhc.clone(), no_rows, evaluator),
            (None, Some(fused)) => eval_or_zero(fused, work.mhc.clone(), no_rows, evaluator),
            _ => unreachable!("resolution creates exactly one entry leaf"),
        }
        eval_or_zero(
            &self.fused_qkv_input_quant,
            work.quant.clone(),
            no_rows,
            evaluator,
        );
        eval_or_zero(&self.fused_qkv, work.gemm.clone(), no_rows, evaluator);
        if let Some(projection) = &self.outer_compressor_proj {
            eval_or_zero(projection, work.fp32_gemm.clone(), no_rows, evaluator);
        }
        if let Some(weights) = &self.indexer_weights_proj {
            eval_or_zero(weights, work.gemm.clone(), no_rows, evaluator);
            eval_or_zero(
                self.indexer_compressor_proj.as_ref().unwrap(),
                work.fp32_gemm.clone(),
                no_rows,
                evaluator,
            );
        }
        eval_or_zero(
            &self.fused_q_kv_rmsnorm,
            work.fused_rmsnorm.clone(),
            no_rows,
            evaluator,
        );
        eval_or_zero(
            &self.q_b_input_quant,
            work.quant.clone(),
            no_rows,
            evaluator,
        );
        eval_or_zero(&self.q_b, work.gemm.clone(), no_rows, evaluator);
        eval_or_zero(
            &self.qnorm_rope_kv_insert,
            work.qnorm_insert.clone(),
            no_rows,
            evaluator,
        );
        if self.outer_compressor_proj.is_some() {
            eval_or_zero(
                self.compressor_store.as_ref().unwrap(),
                work.compressor.clone(),
                no_insert_rows,
                evaluator,
            );
        }
        if self.indexer_weights_proj.is_some() {
            eval_or_zero(
                self.indexer_q_input_quant.as_ref().unwrap(),
                work.quant.clone(),
                no_rows,
                evaluator,
            );
            eval_or_zero(
                self.indexer_q.as_ref().unwrap(),
                work.gemm.clone(),
                no_rows,
                evaluator,
            );
            eval_or_zero(
                self.indexer_q_rope_quant.as_ref().unwrap(),
                work.indexer_q.clone(),
                no_rows,
                evaluator,
            );
            eval_or_zero(
                self.indexer_compressor_store.as_ref().unwrap(),
                work.compressor.clone(),
                no_insert_rows,
                evaluator,
            );
            let no_prefill = input.prefill_query_context_pairs.is_empty();
            eval_or_zero(
                self.indexer_prefill_logits.as_ref().unwrap(),
                work.indexer_prefill.clone(),
                no_prefill,
                evaluator,
            );
            eval_or_zero(
                self.indexer_prefill_topk.as_ref().unwrap(),
                work.indexer_prefill.clone(),
                no_prefill,
                evaluator,
            );
            let no_decode = input.decode_kv_lens.is_empty();
            eval_or_zero(
                self.indexer_decode_logits.as_ref().unwrap(),
                work.indexer_decode_logits.clone(),
                no_decode,
                evaluator,
            );
            eval_or_zero(
                self.indexer_decode_topk.as_ref().unwrap(),
                work.indexer_decode_topk.clone(),
                no_decode,
                evaluator,
            );
        }
        eval_or_zero(
            &self.sparse_prefill,
            work.sparse_prefill,
            input.prefill_query_context_pairs.is_empty(),
            evaluator,
        );
        eval_or_zero(
            &self.sparse_decode,
            work.sparse_decode,
            input.decode_kv_lens.is_empty(),
            evaluator,
        );
        eval_or_zero(
            &self.inverse_rope_quant,
            work.inverse_rope,
            no_rows,
            evaluator,
        );
        eval_or_zero(&self.wo_a, work.wo_a, no_rows, evaluator);
        eval_or_zero(&self.wo_b_input_quant, work.quant, no_rows, evaluator);
        eval_or_zero(&self.wo_b, work.gemm, no_rows, evaluator);
    }
}

#[derive(Clone)]
struct NormalizedInput {
    mhc: MhcRmsNormKernelInput,
    quant: Fp8PerTokenGroupQuantKernelInput,
    gemm: SingleGemmKernelInput,
    fp32_gemm: GemmFp32OutputKernelInput,
    fused_rmsnorm: DeepseekV4FusedQKvRmsnormKernelInput,
    qnorm_insert: DeepseekV4QnormRopeKvInsertKernelInput,
    compressor: DeepseekV4SparseAttnCompressStoreKernelInput,
    indexer_q: DeepseekV4IndexerQRopeQuantKernelInput,
    indexer_prefill: DeepseekV4IndexerPrefillKernelInput,
    indexer_decode_logits: DeepseekV4IndexerMqaLogitsDecodeKernelInput,
    indexer_decode_topk: DeepseekV4IndexerTopkDecodeKernelInput,
    sparse_prefill: DeepseekV4SparseMlaPrefillKernelInput,
    sparse_decode: DeepseekV4SparseMlaDecodeKernelInput,
    inverse_rope: DeepseekV4FusedInvRopeFp8QuantKernelInput,
    wo_a: SingleGemmKernelInput,
}

fn normalize_input(
    input: &DeepseekV4AttentionLocalWorkletInput,
    config: &DeepseekV4AttentionLocalWorkletConfig,
) -> Result<NormalizedInput, String> {
    if input.num_tokens > config.max_num_batched_tokens {
        return Err(format!(
            "num_tokens {} exceeds {}",
            input.num_tokens, config.max_num_batched_tokens
        ));
    }
    let decode_rows = u32::try_from(input.decode_kv_lens.len())
        .map_err(|_| "decode row count exceeds u32".to_string())?;
    let mut active_rows = decode_rows;
    let mut row_positions = Vec::new();
    let mut row_request_ids = Vec::new();
    let mut swa_valid_counts = Vec::with_capacity(input.decode_kv_lens.len());
    let mut extra_valid_counts = Vec::with_capacity(input.decode_kv_lens.len());
    let mut index_context_len = 0;
    // vLLM lays mixed batches out decode-first. Preserve that exact row order
    // because the compressor cache key includes the physical row topology.
    for (decode_request, &resident_kv) in input.decode_kv_lens.iter().enumerate() {
        let context = resident_kv
            .checked_add(1)
            .ok_or_else(|| "decode context overflows u32".to_string())?;
        if context > config.max_model_len {
            return Err(format!("decode context {context} exceeds max_model_len"));
        }
        swa_valid_counts.push(context.min(SWA_WINDOW));
        let extra_capacity = match config.compress_ratio {
            1 => 0,
            4 => config.selected_k,
            128 => config.max_model_len.div_ceil(128).div_ceil(128) * 128,
            ratio => return Err(format!("unsupported compress_ratio {ratio}")),
        };
        extra_valid_counts.push((context / config.compress_ratio).min(extra_capacity));
        index_context_len = index_context_len.max(context / 4);
        row_positions.push(resident_kv);
        row_request_ids.push(decode_request as u32);
    }
    let prefill_request_offset = decode_rows;
    for (request_id, &(queries, context)) in input.prefill_query_context_pairs.iter().enumerate() {
        if queries == 0 || queries > context || context > config.max_model_len {
            return Err(format!("invalid prefill pair ({queries}, {context})"));
        }
        active_rows = active_rows
            .checked_add(queries)
            .ok_or_else(|| "active row count overflows u32".to_string())?;
        let start = context - queries;
        for position in start..context {
            row_positions.push(position);
            row_request_ids.push(prefill_request_offset + request_id as u32);
        }
    }
    if active_rows != input.num_insert_tokens || input.num_insert_tokens > input.num_tokens {
        return Err(format!(
            "phase rows {active_rows}, num_insert_tokens {}, num_tokens {} disagree",
            input.num_insert_tokens, input.num_tokens
        ));
    }
    if row_request_ids.iter().copied().max().unwrap_or(0) >= 64 {
        return Err("compressor topology supports at most 64 requests".to_string());
    }
    let logical_block_size = if config.compress_ratio == 4 { 4 } else { 8 };
    let state_block_table_width = row_positions
        .iter()
        .copied()
        .max()
        .map_or(1, |position| position / logical_block_size + 1);
    let rows = input.num_tokens;
    Ok(NormalizedInput {
        mhc: MhcRmsNormKernelInput { num_tokens: rows },
        quant: Fp8PerTokenGroupQuantKernelInput { num_tokens: rows },
        gemm: SingleGemmKernelInput { m: rows },
        fp32_gemm: GemmFp32OutputKernelInput { m: rows },
        fused_rmsnorm: DeepseekV4FusedQKvRmsnormKernelInput { num_tokens: rows },
        qnorm_insert: DeepseekV4QnormRopeKvInsertKernelInput {
            num_tokens: rows,
            num_insert_tokens: input.num_insert_tokens,
        },
        compressor: DeepseekV4SparseAttnCompressStoreKernelInput {
            row_positions,
            row_request_ids,
            state_block_table_width,
        },
        indexer_q: DeepseekV4IndexerQRopeQuantKernelInput { num_tokens: rows },
        indexer_prefill: DeepseekV4IndexerPrefillKernelInput {
            query_context_pairs: input.prefill_query_context_pairs.clone(),
        },
        indexer_decode_logits: DeepseekV4IndexerMqaLogitsDecodeKernelInput {
            batch_size: input.decode_kv_lens.len() as u32,
            context_len: index_context_len,
        },
        indexer_decode_topk: DeepseekV4IndexerTopkDecodeKernelInput {
            batch_size: input.decode_kv_lens.len() as u32,
            context_len: index_context_len,
        },
        sparse_prefill: DeepseekV4SparseMlaPrefillKernelInput {
            query_context_pairs: input.prefill_query_context_pairs.clone(),
        },
        sparse_decode: DeepseekV4SparseMlaDecodeKernelInput {
            swa_valid_counts,
            extra_valid_counts,
        },
        inverse_rope: DeepseekV4FusedInvRopeFp8QuantKernelInput { num_tokens: rows },
        wo_a: SingleGemmKernelInput {
            m: rows
                .checked_mul(config.o_groups.get())
                .ok_or_else(|| "wo_a rows overflow u32".to_string())?,
        },
    })
}

fn validate_config(config: &DeepseekV4AttentionLocalWorkletConfig) -> Result<(), String> {
    if !(1..=1_048_576).contains(&config.max_model_len) {
        return Err(format!(
            "max_model_len must be in 1..=1048576, got {}",
            config.max_model_len
        ));
    }
    let identity = [
        ("hidden_size", config.hidden_size.get(), HIDDEN_SIZE),
        ("hc_mult", config.hc_mult, 4),
        (
            "num_attention_heads",
            config.num_attention_heads.get(),
            NUM_ATTENTION_HEADS,
        ),
        ("num_kv_heads", config.num_kv_heads.get(), NUM_KV_HEADS),
        ("head_dim", config.head_dim.get(), HEAD_DIM),
        ("rope_dim", config.rope_dim.get(), ROPE_DIM),
        ("q_lora_rank", config.q_lora_rank.get(), Q_LORA_RANK),
        ("o_lora_rank", config.o_lora_rank.get(), O_LORA_RANK),
        ("o_groups", config.o_groups.get(), O_GROUPS),
        ("index_num_heads", config.index_num_heads.get(), INDEX_HEADS),
        (
            "index_head_dim",
            config.index_head_dim.get(),
            INDEX_HEAD_DIM,
        ),
        ("selected_k", config.selected_k, SELECTED_K),
        (
            "max_num_batched_tokens",
            config.max_num_batched_tokens,
            8192,
        ),
    ];
    for (name, actual, expected) in identity {
        if actual != expected {
            return Err(format!("{name} must be {expected}, got {actual}"));
        }
    }
    if !matches!(config.compress_ratio, 1 | 4 | 128) {
        return Err(format!(
            "unsupported compress_ratio {}",
            config.compress_ratio
        ));
    }
    if !matches!(config.planner_mode.as_str(), "planned" | "reused") {
        return Err(format!("unsupported planner_mode {}", config.planner_mode));
    }
    if config.activation_dtype != DType::Bf16
        || config.projection_dtype != DType::Fp8E4m3
        || config.cache_dtype != DType::Fp8E4m3
    {
        return Err(
            "production identity requires BF16 activations and FP8 projections/cache".to_string(),
        );
    }
    Ok(())
}

fn eval_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, evaluator: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    if zero {
        evaluator.push(LeafMetrics::ZERO, || input.into());
    } else {
        op.eval(&input, evaluator);
    }
}

fn compose_parallel(children: Vec<CostNode>, serialize_streams: bool) -> CostNode {
    if serialize_streams {
        CostNode::Sum(children)
    } else {
        CostNode::Max {
            overlap: 1.0,
            children,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DeepseekV4AttentionLocalWorkletConfig {
        DeepseekV4AttentionLocalWorkletConfig {
            entry: DeepseekV4AttentionEntry::Layer0Pre,
            compress_ratio: 4,
            planner_mode: "planned".to_string(),
            serialize_streams: true,
            max_model_len: 65_536,
            max_num_batched_tokens: 8192,
            hidden_size: HIDDEN_SIZE.into(),
            hc_mult: 4,
            num_attention_heads: NUM_ATTENTION_HEADS.into(),
            num_kv_heads: NUM_KV_HEADS.into(),
            head_dim: HEAD_DIM.into(),
            rope_dim: ROPE_DIM.into(),
            q_lora_rank: Q_LORA_RANK.into(),
            o_lora_rank: O_LORA_RANK.into(),
            o_groups: O_GROUPS.into(),
            index_num_heads: INDEX_HEADS.into(),
            index_head_dim: INDEX_HEAD_DIM.into(),
            selected_k: SELECTED_K,
            activation_dtype: DType::Bf16,
            projection_dtype: DType::Fp8E4m3,
            cache_dtype: DType::Fp8E4m3,
            gpu_name: "NVIDIA H200".to_string(),
            mhc_backends: vec!["vllm_tilelang"],
            quant_backends: vec!["vllm_cuda"],
            gemm_backends: vec!["torch"],
            fp32_gemm_backends: vec!["torch_cublas"],
            fused_q_kv_rmsnorm_backends: vec!["vllm_triton"],
            qnorm_rope_kv_insert_backends: vec!["vllm_cuda"],
            compressor_store_backends: vec!["vllm_deepseek_v4_cutedsl"],
            indexer_compressor_store_backends: vec!["vllm_deepseek_v4_triton"],
            sparse_prefill_backends: vec!["vllm_flashmla_bf16"],
            sparse_decode_backends: vec!["vllm_flashmla_fp8_cudagraph"],
            inverse_rope_quant_backends: vec!["vllm_triton"],
            indexer_q_rope_quant_backends: vec!["vllm_cutedsl_fp8"],
            indexer_prefill_logits_backends: vec!["vllm_deepgemm_mqa_fp8"],
            indexer_prefill_topk_backends: vec!["vllm_cuda"],
            indexer_decode_logits_backends: vec!["vllm_cuda"],
            indexer_decode_topk_backends: vec!["vllm_cuda"],
        }
    }

    #[test]
    fn mixed_iteration_merges_common_rows_and_preserves_phase_work() {
        let input = DeepseekV4AttentionLocalWorkletInput {
            num_tokens: 8,
            num_insert_tokens: 7,
            prefill_query_context_pairs: vec![(4, 16), (1, 9)],
            decode_kv_lens: vec![7, 15],
        };
        let work = normalize_input(&input, &config()).unwrap();
        assert_eq!(work.gemm.m, 8);
        assert_eq!(work.qnorm_insert.num_insert_tokens, 7);
        assert_eq!(
            work.sparse_prefill.query_context_pairs,
            vec![(4, 16), (1, 9)]
        );
        assert_eq!(work.sparse_decode.swa_valid_counts, vec![8, 16]);
        assert_eq!(work.sparse_decode.extra_valid_counts, vec![2, 4]);
        assert_eq!(
            work.compressor.row_positions,
            vec![7, 15, 12, 13, 14, 15, 8]
        );
        assert_eq!(work.compressor.row_request_ids, vec![0, 1, 2, 2, 2, 2, 3]);
    }

    #[test]
    fn phase_rows_must_equal_actual_insert_rows() {
        let result = normalize_input(
            &DeepseekV4AttentionLocalWorkletInput {
                num_tokens: 8,
                num_insert_tokens: 6,
                prefill_query_context_pairs: vec![(4, 16)],
                decode_kv_lens: vec![7],
            },
            &config(),
        );
        let Err(error) = result else {
            panic!("mismatched insert rows must be rejected")
        };
        assert!(error.contains("phase rows 5"));
    }

    #[test]
    fn c128_decode_uses_runtime_compressed_capacity_not_c4_topk() {
        let mut config = config();
        config.compress_ratio = 128;
        config.max_model_len = 1_048_576;
        let work = normalize_input(
            &DeepseekV4AttentionLocalWorkletInput {
                num_tokens: 1,
                num_insert_tokens: 1,
                prefill_query_context_pairs: Vec::new(),
                decode_kv_lens: vec![1_048_575],
            },
            &config,
        )
        .unwrap();
        assert_eq!(work.sparse_decode.extra_valid_counts, vec![8192]);
    }

    #[test]
    fn padding_only_rank_keeps_projection_rows_without_compressor_rows() {
        let work = normalize_input(
            &DeepseekV4AttentionLocalWorkletInput {
                num_tokens: 8,
                num_insert_tokens: 0,
                prefill_query_context_pairs: Vec::new(),
                decode_kv_lens: Vec::new(),
            },
            &config(),
        )
        .unwrap();
        assert_eq!(work.gemm.m, 8);
        assert_eq!(work.qnorm_insert.num_insert_tokens, 0);
        assert!(work.compressor.row_positions.is_empty());
    }
}
