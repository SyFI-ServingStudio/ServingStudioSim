//! GLM-5.2 full DSA indexer compound op.
//!
//! This op is one complete full-index invocation. Its fifteen fixed serial
//! leaves share the indexer projection/quantization/cache state, while phase
//! normalization folds request-local prefill calls into fixed fan-in slots and
//! one optional decode cell. `IndexShare` scheduling belongs above L2: a full-
//! index layer evaluates this whole op and a shared-index layer omits it.
//! Request-local/global index remap remains in `DsaSparseMlaAttentionOp`, and
//! communication remains outside both attention compound ops.
//!
//! GLM-5.2 has 32 semantic index heads. Projection and elementwise byte shapes
//! therefore always use H32. The accepted prefill/decode logits caches are H64-
//! only, so those two leaves deliberately use a separate H64 timing surrogate.
//! That conservative surrogate changes timing identity, not model semantics.

use std::sync::Arc;

use crate::timing::bridge::DType;
use crate::timing::kernels::{
    DsaIndexCacheAppendKernel, DsaIndexCacheAppendKernelConfig, DsaIndexCacheAppendKernelInput,
    DsaMqaLogitsPrefillKernel, DsaMqaLogitsPrefillKernelConfig, DsaMqaLogitsPrefillKernelInput,
    DsaPagedMqaLogitsDecodeKernel, DsaPagedMqaLogitsDecodeKernelConfig,
    DsaPagedMqaLogitsDecodeKernelInput, DsaPersistentTopkDecodeKernel,
    DsaPersistentTopkDecodeKernelConfig, DsaPersistentTopkDecodeKernelInput, DsaTopkPrefillKernel,
    DsaTopkPrefillKernelConfig, DsaTopkPrefillKernelInput, ElementwiseKernel,
    ElementwiseKernelConfig, ElementwiseKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::slot_input::DsaIndexerPrefillLog;
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
};

const OP_KIND: &str = "dsa_indexer";
const MODEL_INDEX_HEADS: u32 = 32;
const PROFILE_INDEX_HEADS: u32 = 64;
const INDEX_HEAD_DIM: u32 = 128;
const ROPE_DIM: u32 = 64;
const HIDDEN_DIM: u32 = 6144;
const Q_LORA_RANK: u32 = 2048;
const TOP_K: u32 = 2048;
/// The v1 DSA indexer is frozen to one measured shape family, so the accepted
/// context is an equality check rather than a bound. It must move together with
/// `TIMING_MAX_MODEL_LEN` in the GLM-5.2 arch files and with the same constants
/// in the L3 DSA attention worklets — three deliberate copies, one domain.
const MAX_MODEL_LEN: u32 = 1_048_576;
const LOGITS_ROW_STRIDE: u32 = 1_048_576;
const CACHE_BLOCK_SIZE: u32 = 64;
const QUANT_BLOCK_SIZE: u32 = 128;

const SLOT_SUFFIXES: [&str; 15] = [
    "q_proj",
    "wk_weights_proj",
    "k_layernorm",
    "rope",
    "q_quant",
    "weight_scale",
    "index_cache_append",
    "topk_buffer_fill",
    "prefill_cache_gather",
    "prefill_logits",
    "prefill_topk",
    "decode_pack",
    "decode_logits",
    "decode_topk",
    "decode_unpack",
];

/// GLM-specific static identity expanded into the fifteen L1 leaves.
#[derive(Clone, Debug)]
pub struct DsaIndexerConfig {
    pub gemm_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub index_cache_append_backends: Vec<&'static str>,
    pub prefill_logits_backends: Vec<&'static str>,
    pub prefill_topk_backends: Vec<&'static str>,
    pub decode_logits_backends: Vec<&'static str>,
    pub decode_topk_backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    pub q_lora_rank: Dim,
    pub model_num_index_heads: Dim,
    pub profile_num_index_heads: Dim,
    pub index_head_dim: Dim,
    pub rope_dim: Dim,
    pub next_n: u32,
    pub max_model_len: Dim,
    pub top_k: u32,
    pub logits_row_stride: Dim,
    pub cache_block_size: u32,
    pub quant_block_size: u32,
    pub input_dtype: DType,
    /// Dtype used by the two generic indexer projection GEMMs. The input and
    /// all cache/quantization semantics remain governed by their own fields.
    pub gemm_dtype: DType,
    pub cache_dtype: DType,
    pub q_dtype: DType,
    pub scale_dtype: DType,
    pub weight_dtype: DType,
    pub logits_dtype: DType,
    pub index_dtype: String,
    pub scale_format: String,
    pub cache_format: String,
    pub prefill_span_mode: String,
    pub decode_context_mode: String,
    pub decode_page_mapping: String,
    pub clean_logits: bool,
}

#[derive(Clone, Debug)]
pub struct DsaIndexerDecodeInput {
    pub batch_size: u32,
    pub context_len: u32,
    pub requires_padding: bool,
}

#[derive(Clone, Debug, Default)]
pub struct DsaIndexerInput {
    pub num_new_tokens: u32,
    pub prefill_query_key_pairs: Vec<(u32, u32)>,
    pub decode: Option<DsaIndexerDecodeInput>,
}

pub struct DsaIndexerOp {
    pub name: String,
    pub q_proj: Arc<SingleGemmKernel>,
    pub wk_weights_proj: Arc<SingleGemmKernel>,
    pub k_layernorm: Arc<ElementwiseKernel>,
    pub rope: Arc<ElementwiseKernel>,
    pub q_quant: Arc<ElementwiseKernel>,
    pub weight_scale: Arc<ElementwiseKernel>,
    pub index_cache_append: Arc<DsaIndexCacheAppendKernel>,
    pub topk_buffer_fill: Arc<ElementwiseKernel>,
    pub prefill_cache_gather: Arc<ElementwiseKernel>,
    pub prefill_logits: Arc<DsaMqaLogitsPrefillKernel>,
    pub prefill_topk: Arc<DsaTopkPrefillKernel>,
    pub decode_pack: Arc<ElementwiseKernel>,
    pub decode_logits: Arc<DsaPagedMqaLogitsDecodeKernel>,
    pub decode_topk: Arc<DsaPersistentTopkDecodeKernel>,
    pub decode_unpack: Arc<ElementwiseKernel>,
    next_n: u32,
    max_model_len: u32,
}

impl DsaIndexerOp {
    pub fn build(
        name: String,
        cfg: DsaIndexerConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let subcfg = subkernel_configs(&cfg)?;
        let q_proj = Arc::new(SingleGemmKernel::build(
            slot_name(&name, 0),
            subcfg.q_proj,
            bridge,
        )?);
        let wk_weights_proj = Arc::new(SingleGemmKernel::build(
            slot_name(&name, 1),
            subcfg.wk_weights_proj,
            bridge,
        )?);
        let k_layernorm = Arc::new(ElementwiseKernel::build(
            slot_name(&name, 2),
            subcfg.k_layernorm,
            bridge,
        )?);
        let rope = Arc::new(ElementwiseKernel::build(
            slot_name(&name, 3),
            subcfg.rope,
            bridge,
        )?);
        let q_quant = Arc::new(ElementwiseKernel::build(
            slot_name(&name, 4),
            subcfg.q_quant,
            bridge,
        )?);
        let weight_scale = Arc::new(ElementwiseKernel::build(
            slot_name(&name, 5),
            subcfg.weight_scale,
            bridge,
        )?);
        let index_cache_append = Arc::new(DsaIndexCacheAppendKernel::build(
            slot_name(&name, 6),
            subcfg.index_cache_append,
            bridge,
        )?);
        let topk_buffer_fill = Arc::new(ElementwiseKernel::build(
            slot_name(&name, 7),
            subcfg.topk_buffer_fill,
            bridge,
        )?);
        let prefill_cache_gather = Arc::new(ElementwiseKernel::build(
            slot_name(&name, 8),
            subcfg.prefill_cache_gather,
            bridge,
        )?);
        let prefill_logits = Arc::new(DsaMqaLogitsPrefillKernel::build(
            slot_name(&name, 9),
            subcfg.prefill_logits,
            bridge,
        )?);
        let prefill_topk = Arc::new(DsaTopkPrefillKernel::build(
            slot_name(&name, 10),
            subcfg.prefill_topk,
            bridge,
        )?);
        let decode_pack = Arc::new(ElementwiseKernel::build(
            slot_name(&name, 11),
            subcfg.decode_pack,
            bridge,
        )?);
        let decode_logits = Arc::new(DsaPagedMqaLogitsDecodeKernel::build(
            slot_name(&name, 12),
            subcfg.decode_logits,
            bridge,
        )?);
        let decode_topk = Arc::new(DsaPersistentTopkDecodeKernel::build(
            slot_name(&name, 13),
            subcfg.decode_topk,
            bridge,
        )?);
        let decode_unpack = Arc::new(ElementwiseKernel::build(
            slot_name(&name, 14),
            subcfg.decode_unpack,
            bridge,
        )?);

        Ok(Self {
            name,
            q_proj,
            wk_weights_proj,
            k_layernorm,
            rope,
            q_quant,
            weight_scale,
            index_cache_append,
            topk_buffer_fill,
            prefill_cache_gather,
            prefill_logits,
            prefill_topk,
            decode_pack,
            decode_logits,
            decode_topk,
            decode_unpack,
            next_n: cfg.next_n,
            max_model_len: cfg.max_model_len.get(),
        })
    }

    /// Fifteen fixed serial leaves, independent of request count and phase.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            leaf(builder, &self.name, 0, &*self.q_proj),
            leaf(builder, &self.name, 1, &*self.wk_weights_proj),
            leaf(builder, &self.name, 2, &*self.k_layernorm),
            leaf(builder, &self.name, 3, &*self.rope),
            leaf(builder, &self.name, 4, &*self.q_quant),
            leaf(builder, &self.name, 5, &*self.weight_scale),
            leaf(builder, &self.name, 6, &*self.index_cache_append),
            leaf(builder, &self.name, 7, &*self.topk_buffer_fill),
            leaf(builder, &self.name, 8, &*self.prefill_cache_gather),
            leaf(builder, &self.name, 9, &*self.prefill_logits),
            leaf(builder, &self.name, 10, &*self.prefill_topk),
            leaf(builder, &self.name, 11, &*self.decode_pack),
            leaf(builder, &self.name, 12, &*self.decode_logits),
            leaf(builder, &self.name, 13, &*self.decode_topk),
            leaf(builder, &self.name, 14, &*self.decode_unpack),
        ])
    }

    /// Push exactly the fifteen compile slots in the same order (INV-2).
    pub fn eval(&self, input: &DsaIndexerInput, ev: &mut Evaluator) {
        let normalized = normalize_input(input, self.next_n, self.max_model_len)
            .unwrap_or_else(|reason| panic!("invalid DsaIndexerInput: {reason}"));

        let active_gemm = SingleGemmKernelInput {
            m: normalized.active_query_rows,
        };
        ev.push(eval_gemm_or_zero(&self.q_proj, &active_gemm), || {
            active_gemm.clone().into()
        });
        ev.push(
            eval_gemm_or_zero(&self.wk_weights_proj, &active_gemm),
            || active_gemm.clone().into(),
        );

        push_elementwise(&self.k_layernorm, normalized.active_query_rows, ev);
        push_elementwise(&self.rope, normalized.active_query_rows, ev);
        push_elementwise(&self.q_quant, normalized.active_query_rows, ev);
        push_elementwise(&self.weight_scale, normalized.active_query_rows, ev);

        let cache_append = DsaIndexCacheAppendKernelInput {
            num_tokens: input.num_new_tokens,
        };
        let cache_append_metrics = if input.num_new_tokens == 0 {
            LeafMetrics::ZERO
        } else {
            self.index_cache_append.eval(&cache_append)
        };
        ev.push(cache_append_metrics, || cache_append.clone().into());

        push_elementwise(&self.topk_buffer_fill, normalized.active_query_rows, ev);

        let mut gather_metrics = LeafMetrics::ZERO;
        for &(_, num_keys) in &input.prefill_query_key_pairs {
            gather_metrics.add_fanin(self.prefill_cache_gather.eval(&ElementwiseKernelInput {
                num_tokens: num_keys,
            }));
        }
        push_prefill_log(ev, gather_metrics, &input.prefill_query_key_pairs);

        let mut prefill_logits_metrics = LeafMetrics::ZERO;
        for &(num_queries, num_keys) in &input.prefill_query_key_pairs {
            prefill_logits_metrics.add_fanin(self.prefill_logits.eval(
                &DsaMqaLogitsPrefillKernelInput {
                    num_queries,
                    num_keys,
                },
            ));
        }
        push_prefill_log(ev, prefill_logits_metrics, &input.prefill_query_key_pairs);

        let mut prefill_topk_metrics = LeafMetrics::ZERO;
        for &(num_queries, num_keys) in &input.prefill_query_key_pairs {
            prefill_topk_metrics.add_fanin(self.prefill_topk.eval(&DsaTopkPrefillKernelInput {
                num_queries,
                num_keys,
            }));
        }
        push_prefill_log(ev, prefill_topk_metrics, &input.prefill_query_key_pairs);

        push_elementwise(&self.decode_pack, normalized.padded_decode_rows, ev);

        let decode_logits_metrics = normalized
            .decode_logits
            .as_ref()
            .map_or(LeafMetrics::ZERO, |shape| self.decode_logits.eval(shape));
        ev.push(decode_logits_metrics, || {
            normalized
                .decode_logits
                .clone()
                .unwrap_or(DsaPagedMqaLogitsDecodeKernelInput {
                    batch_size: 0,
                    context_len: 0,
                })
                .into()
        });

        let decode_topk_metrics = normalized
            .decode_topk
            .as_ref()
            .map_or(LeafMetrics::ZERO, |shape| self.decode_topk.eval(shape));
        ev.push(decode_topk_metrics, || {
            normalized
                .decode_topk
                .clone()
                .unwrap_or(DsaPersistentTopkDecodeKernelInput {
                    batch_size: 0,
                    context_len: 0,
                })
                .into()
        });

        push_elementwise(&self.decode_unpack, normalized.padded_decode_rows, ev);
    }
}

struct SubkernelConfigs {
    q_proj: SingleGemmKernelConfig,
    wk_weights_proj: SingleGemmKernelConfig,
    k_layernorm: ElementwiseKernelConfig,
    rope: ElementwiseKernelConfig,
    q_quant: ElementwiseKernelConfig,
    weight_scale: ElementwiseKernelConfig,
    index_cache_append: DsaIndexCacheAppendKernelConfig,
    topk_buffer_fill: ElementwiseKernelConfig,
    prefill_cache_gather: ElementwiseKernelConfig,
    prefill_logits: DsaMqaLogitsPrefillKernelConfig,
    prefill_topk: DsaTopkPrefillKernelConfig,
    decode_pack: ElementwiseKernelConfig,
    decode_logits: DsaPagedMqaLogitsDecodeKernelConfig,
    decode_topk: DsaPersistentTopkDecodeKernelConfig,
    decode_unpack: ElementwiseKernelConfig,
}

struct NormalizedInput {
    active_query_rows: u32,
    padded_decode_rows: u32,
    decode_logits: Option<DsaPagedMqaLogitsDecodeKernelInput>,
    decode_topk: Option<DsaPersistentTopkDecodeKernelInput>,
}

fn slot_name(name: &str, index: usize) -> String {
    format!("{name}.{}", SLOT_SUFFIXES[index])
}

fn leaf<K: Probe>(builder: &mut CostTreeBuilder, name: &str, index: usize, kernel: &K) -> CostNode {
    builder.leaf(
        slot_name(name, index),
        kernel.kind(),
        kernel.describe_config(),
    )
}

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: OP_KIND,
        reason: reason.into(),
    }
}

fn subkernel_configs(cfg: &DsaIndexerConfig) -> Result<SubkernelConfigs, BuildError> {
    validate_config(cfg)?;
    let elementwise = elementwise_configs(cfg)?;

    Ok(SubkernelConfigs {
        q_proj: SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: cfg.model_num_index_heads.clone() * cfg.index_head_dim.clone(),
            k: cfg.q_lora_rank.clone(),
            dtype: cfg.gemm_dtype,
        },
        wk_weights_proj: SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: cfg.index_head_dim.clone() + cfg.model_num_index_heads.clone(),
            k: cfg.hidden_dim.clone(),
            dtype: cfg.gemm_dtype,
        },
        k_layernorm: elementwise.k_layernorm,
        rope: elementwise.rope,
        q_quant: elementwise.q_quant,
        weight_scale: elementwise.weight_scale,
        index_cache_append: DsaIndexCacheAppendKernelConfig {
            backends: cfg.index_cache_append_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            index_dim: cfg.index_head_dim.clone(),
            block_size: cfg.cache_block_size,
            quant_block_size: cfg.quant_block_size,
            input_dtype: cfg.input_dtype,
            cache_dtype: cfg.cache_dtype,
            scale_format: cfg.scale_format.clone(),
            cache_format: cfg.cache_format.clone(),
        },
        topk_buffer_fill: elementwise.topk_buffer_fill,
        prefill_cache_gather: elementwise.prefill_cache_gather,
        prefill_logits: DsaMqaLogitsPrefillKernelConfig {
            backends: cfg.prefill_logits_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            num_sequences: 1,
            num_heads: cfg.profile_num_index_heads.clone(),
            head_dim: cfg.index_head_dim.clone(),
            q_dtype: cfg.q_dtype,
            k_dtype: cfg.cache_dtype,
            k_scale_dtype: cfg.scale_dtype,
            weight_dtype: cfg.weight_dtype,
            output_dtype: cfg.logits_dtype,
            span_mode: cfg.prefill_span_mode.clone(),
            clean_logits: cfg.clean_logits,
        },
        prefill_topk: DsaTopkPrefillKernelConfig {
            backends: cfg.prefill_topk_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            num_sequences: 1,
            top_k: cfg.top_k,
            logits_dtype: cfg.logits_dtype,
            index_dtype: cfg.index_dtype.clone(),
            span_mode: cfg.prefill_span_mode.clone(),
        },
        decode_pack: elementwise.decode_pack,
        decode_logits: DsaPagedMqaLogitsDecodeKernelConfig {
            backends: cfg.decode_logits_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            next_n: cfg.next_n,
            max_model_len: cfg.max_model_len.clone(),
            num_heads: cfg.profile_num_index_heads.clone(),
            head_dim: cfg.index_head_dim.clone(),
            block_size: cfg.cache_block_size,
            q_dtype: cfg.q_dtype,
            cache_dtype: cfg.cache_dtype,
            scale_dtype: cfg.scale_dtype,
            weight_dtype: cfg.weight_dtype,
            output_dtype: cfg.logits_dtype,
            context_mode: cfg.decode_context_mode.clone(),
            page_mapping: cfg.decode_page_mapping.clone(),
            cache_format: cfg.cache_format.clone(),
            clean_logits: cfg.clean_logits,
        },
        decode_topk: DsaPersistentTopkDecodeKernelConfig {
            backends: cfg.decode_topk_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            next_n: cfg.next_n,
            max_model_len: cfg.max_model_len.clone(),
            top_k: cfg.top_k,
            logits_row_stride: cfg.logits_row_stride.clone(),
            logits_dtype: cfg.logits_dtype,
            index_dtype: cfg.index_dtype.clone(),
            context_mode: cfg.decode_context_mode.clone(),
        },
        decode_unpack: elementwise.decode_unpack,
    })
}

fn validate_config(cfg: &DsaIndexerConfig) -> Result<(), BuildError> {
    for (name, actual, required) in [
        ("hidden_dim", cfg.hidden_dim.get(), HIDDEN_DIM),
        ("q_lora_rank", cfg.q_lora_rank.get(), Q_LORA_RANK),
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
        ("rope_dim", cfg.rope_dim.get(), ROPE_DIM),
        ("max_model_len", cfg.max_model_len.get(), MAX_MODEL_LEN),
        ("top_k", cfg.top_k, TOP_K),
        (
            "logits_row_stride",
            cfg.logits_row_stride.get(),
            LOGITS_ROW_STRIDE,
        ),
        ("cache_block_size", cfg.cache_block_size, CACHE_BLOCK_SIZE),
        ("quant_block_size", cfg.quant_block_size, QUANT_BLOCK_SIZE),
    ] {
        if actual != required {
            return Err(fit_failed(format!(
                "{name} must be {required}, got {actual}"
            )));
        }
    }
    if !matches!(cfg.next_n, 1 | 2) {
        return Err(fit_failed(format!(
            "next_n must be 1 or 2, got {}",
            cfg.next_n
        )));
    }
    for (name, actual, required) in [
        ("input_dtype", cfg.input_dtype, DType::Bf16),
        ("cache_dtype", cfg.cache_dtype, DType::Fp8E4m3),
        ("q_dtype", cfg.q_dtype, DType::Fp8E4m3),
        ("scale_dtype", cfg.scale_dtype, DType::Fp32),
        ("weight_dtype", cfg.weight_dtype, DType::Fp32),
        ("logits_dtype", cfg.logits_dtype, DType::Fp32),
    ] {
        if actual != required {
            return Err(fit_failed(format!(
                "{name} must be {}, got {}",
                required.as_str(),
                actual.as_str()
            )));
        }
    }
    if !matches!(cfg.gemm_dtype, DType::Bf16 | DType::Fp8E4m3) {
        return Err(fit_failed(format!(
            "gemm_dtype must be {} or {}, got {}",
            DType::Bf16.as_str(),
            DType::Fp8E4m3.as_str(),
            cfg.gemm_dtype.as_str()
        )));
    }
    if cfg.index_dtype != "int32" {
        return Err(fit_failed(format!(
            "index_dtype must be int32, got {:?}",
            cfg.index_dtype
        )));
    }
    Ok(())
}

struct ElementwiseConfigs {
    k_layernorm: ElementwiseKernelConfig,
    rope: ElementwiseKernelConfig,
    q_quant: ElementwiseKernelConfig,
    weight_scale: ElementwiseKernelConfig,
    topk_buffer_fill: ElementwiseKernelConfig,
    prefill_cache_gather: ElementwiseKernelConfig,
    decode_pack: ElementwiseKernelConfig,
    decode_unpack: ElementwiseKernelConfig,
}

fn elementwise_configs(cfg: &DsaIndexerConfig) -> Result<ElementwiseConfigs, BuildError> {
    let model_heads = cfg.model_num_index_heads.get();
    let head_dim = cfg.index_head_dim.get();
    let rope_dim = cfg.rope_dim.get();
    let bf16 = cfg.input_dtype.size_bytes();
    let fp8 = cfg.q_dtype.size_bytes();
    let fp32_scale = cfg.scale_dtype.size_bytes();
    let fp32_weight = cfg.weight_dtype.size_bytes();

    let k_layernorm = checked_product("k_layernorm bytes", &[head_dim, bf16])?;
    // The indexer RoPE leaf is not rope-slice sized: torch.compile fuses the
    // `torch.cat` that rebuilds q and k right after it (vllm deepseek_v2.py
    // :713-726), which is why the traced kernel is named
    // `triton_poi_fused_add_cat_index_select_mul_slice_split_split_with_sizes_stack_...`.
    // Both halves of q are read and the full q is written back, so the kernel
    // moves head_dim per head, not rope_dim. Charging rope_dim alone put this
    // leaf at -41% on decode.
    let rope_heads = checked_add("rope head count", model_heads, 1)?;
    let rope_qk = checked_product("rope q/k bytes", &[rope_heads, head_dim, bf16])?;
    // cos/sin gathered by index_select, one pair per rope element.
    let rope_table = checked_product("rope cos/sin bytes", &[rope_dim, 2, bf16])?;
    let rope_input = checked_add("rope input bytes", rope_qk, rope_table)?;
    let rope_output = rope_qk;
    let q_quant_input = checked_product("q_quant input bytes", &[model_heads, head_dim, bf16])?;
    let q_quant_per_head = checked_add(
        "q_quant output bytes per head",
        checked_product("q_quant FP8 values", &[head_dim, fp8])?,
        fp32_scale,
    )?;
    let q_quant_output = checked_product("q_quant output bytes", &[model_heads, q_quant_per_head])?;
    let weight_scale_input = checked_product(
        "weight_scale input bytes",
        &[
            model_heads,
            checked_add("weight_scale input bytes per head", bf16, fp32_scale)?,
        ],
    )?;
    let weight_scale_output =
        checked_product("weight_scale output bytes", &[model_heads, fp32_weight])?;
    let topk_bytes = checked_product("top-k buffer bytes", &[cfg.top_k, 4])?;
    let packed_per_head = checked_add(
        "decode packed bytes per head",
        checked_product("decode packed FP8 values", &[head_dim, fp8])?,
        fp32_weight,
    )?;
    let decode_pack = checked_product("decode pack bytes", &[model_heads, packed_per_head])?;
    let gather_values = checked_product(
        "prefill gather values",
        &[head_dim, cfg.cache_dtype.size_bytes()],
    )?;
    let gather_input = checked_add(
        "prefill gather input bytes",
        checked_add(
            "prefill gather value+scale bytes",
            gather_values,
            fp32_scale,
        )?,
        4,
    )?;
    let gather_output = checked_add("prefill gather output bytes", gather_values, fp32_scale)?;

    let make = |input: u32, output: u32| ElementwiseKernelConfig {
        backends: cfg.elementwise_backends.clone(),
        gpu_name: cfg.gpu_name.clone(),
        input_bytes_per_token: input.into(),
        output_bytes_per_token: output.into(),
    };
    Ok(ElementwiseConfigs {
        k_layernorm: make(k_layernorm, k_layernorm),
        rope: make(rope_input, rope_output),
        q_quant: make(q_quant_input, q_quant_output),
        weight_scale: make(weight_scale_input, weight_scale_output),
        topk_buffer_fill: make(0, topk_bytes),
        prefill_cache_gather: make(gather_input, gather_output),
        decode_pack: make(decode_pack, decode_pack),
        decode_unpack: make(topk_bytes, topk_bytes),
    })
}

fn checked_add(name: &str, lhs: u32, rhs: u32) -> Result<u32, BuildError> {
    lhs.checked_add(rhs)
        .ok_or_else(|| fit_failed(format!("{name} overflows u32")))
}

fn checked_product(name: &str, factors: &[u32]) -> Result<u32, BuildError> {
    factors.iter().try_fold(1_u32, |value, factor| {
        value
            .checked_mul(*factor)
            .ok_or_else(|| fit_failed(format!("{name} overflows u32")))
    })
}

fn normalize_input(
    input: &DsaIndexerInput,
    next_n: u32,
    max_model_len: u32,
) -> Result<NormalizedInput, String> {
    if !matches!(next_n, 1 | 2) {
        return Err(format!("next_n must be 1 or 2, got {next_n}"));
    }

    let mut active_query_rows = 0_u32;
    for (index, &(num_queries, num_keys)) in input.prefill_query_key_pairs.iter().enumerate() {
        if num_queries == 0 || num_keys == 0 {
            return Err(format!(
                "prefill pair {index} must have nonzero Q and N, got ({num_queries}, {num_keys})"
            ));
        }
        if num_queries > num_keys {
            return Err(format!(
                "prefill pair {index} requires Q<=N, got ({num_queries}, {num_keys})"
            ));
        }
        active_query_rows = active_query_rows
            .checked_add(num_queries)
            .ok_or_else(|| "active query-row sum overflows u32".to_string())?;
    }

    let (padded_decode_rows, decode_logits, decode_topk) = match &input.decode {
        Some(decode) => {
            if decode.batch_size == 0 || decode.context_len == 0 {
                return Err(format!(
                    "decode requires positive batch_size and context_len, got ({}, {})",
                    decode.batch_size, decode.context_len
                ));
            }
            if decode.context_len > max_model_len {
                return Err(format!(
                    "decode context_len {} exceeds max_model_len {max_model_len}",
                    decode.context_len
                ));
            }
            let decode_rows = decode
                .batch_size
                .checked_mul(next_n)
                .ok_or_else(|| "decode row count overflows u32".to_string())?;
            active_query_rows = active_query_rows
                .checked_add(decode_rows)
                .ok_or_else(|| "active query-row sum overflows u32".to_string())?;
            (
                if decode.requires_padding {
                    decode_rows
                } else {
                    0
                },
                Some(DsaPagedMqaLogitsDecodeKernelInput {
                    batch_size: decode.batch_size,
                    context_len: decode.context_len,
                }),
                Some(DsaPersistentTopkDecodeKernelInput {
                    batch_size: decode.batch_size,
                    context_len: decode.context_len,
                }),
            )
        }
        None => (0, None, None),
    };

    Ok(NormalizedInput {
        active_query_rows,
        padded_decode_rows,
        decode_logits,
        decode_topk,
    })
}

fn eval_gemm_or_zero(kernel: &SingleGemmKernel, input: &SingleGemmKernelInput) -> LeafMetrics {
    if input.m == 0 {
        LeafMetrics::ZERO
    } else {
        kernel.eval(input)
    }
}

fn push_elementwise(kernel: &ElementwiseKernel, num_tokens: u32, ev: &mut Evaluator) {
    let input = ElementwiseKernelInput { num_tokens };
    let metrics = if num_tokens == 0 {
        LeafMetrics::ZERO
    } else {
        kernel.eval(&input)
    };
    ev.push(metrics, || input.clone().into());
}

fn push_prefill_log(ev: &mut Evaluator, metrics: LeafMetrics, pairs: &[(u32, u32)]) {
    ev.push(metrics, || {
        DsaIndexerPrefillLog {
            prefill_query_key_pairs: pairs.to_vec(),
        }
        .into()
    });
}

#[cfg(test)]
mod tests {
    use super::{
        elementwise_configs, normalize_input, subkernel_configs, validate_config, DsaIndexerConfig,
        DsaIndexerDecodeInput, DsaIndexerInput, LOGITS_ROW_STRIDE, MAX_MODEL_LEN, SLOT_SUFFIXES,
    };
    use crate::timing::bridge::DType;
    use crate::timing::slot_input::DsaIndexerPrefillLog;
    use crate::timing::{BuildError, Dim, SlotInput};

    fn cfg(next_n: u32) -> DsaIndexerConfig {
        DsaIndexerConfig {
            gemm_backends: vec!["torch"],
            elementwise_backends: vec!["triton"],
            index_cache_append_backends: vec!["vllm_cuda"],
            prefill_logits_backends: vec!["vllm_deepgemm_fp8"],
            prefill_topk_backends: vec!["vllm_cuda"],
            decode_logits_backends: vec!["vllm_deepgemm_fp8"],
            decode_topk_backends: vec!["vllm_cuda"],
            gpu_name: "NVIDIA H200".to_string(),
            hidden_dim: Dim::param("hidden_dim", 6144),
            q_lora_rank: Dim::param("q_lora_rank", 2048),
            model_num_index_heads: Dim::param("model_num_index_heads", 32),
            profile_num_index_heads: Dim::param("profile_num_index_heads", 64),
            index_head_dim: Dim::param("index_head_dim", 128),
            rope_dim: Dim::param("rope_dim", 64),
            next_n,
            max_model_len: Dim::param("max_model_len", MAX_MODEL_LEN),
            top_k: 2048,
            logits_row_stride: Dim::param("logits_row_stride", LOGITS_ROW_STRIDE),
            cache_block_size: 64,
            quant_block_size: 128,
            input_dtype: DType::Bf16,
            gemm_dtype: DType::Bf16,
            cache_dtype: DType::Fp8E4m3,
            q_dtype: DType::Fp8E4m3,
            scale_dtype: DType::Fp32,
            weight_dtype: DType::Fp32,
            logits_dtype: DType::Fp32,
            index_dtype: "int32".to_string(),
            scale_format: "ue8m0".to_string(),
            cache_format: "page_planar_fp8_fp32_scale".to_string(),
            prefill_span_mode: "single_causal_tail".to_string(),
            decode_context_mode: "uniform".to_string(),
            decode_page_mapping: "unique_scattered".to_string(),
            clean_logits: false,
        }
    }

    #[test]
    fn fifteen_slot_order_is_exact_and_excludes_other_boundaries() {
        assert_eq!(
            SLOT_SUFFIXES,
            [
                "q_proj",
                "wk_weights_proj",
                "k_layernorm",
                "rope",
                "q_quant",
                "weight_scale",
                "index_cache_append",
                "topk_buffer_fill",
                "prefill_cache_gather",
                "prefill_logits",
                "prefill_topk",
                "decode_pack",
                "decode_logits",
                "decode_topk",
                "decode_unpack"
            ]
        );
        assert_eq!(SLOT_SUFFIXES.len(), 15);
        assert!(!SLOT_SUFFIXES.iter().any(|slot| slot.contains("remap")));
        assert!(!SLOT_SUFFIXES
            .iter()
            .any(|slot| slot.contains("sparse_attention")));
        assert!(!SLOT_SUFFIXES.iter().any(|slot| slot.contains("all_reduce")));
        assert!(!SLOT_SUFFIXES.iter().any(|slot| slot.contains("all_to_all")));
    }

    #[test]
    fn config_expansion_preserves_h32_semantics_and_h64_logits_surrogate() {
        let configs = subkernel_configs(&cfg(1)).unwrap();

        assert_eq!(configs.q_proj.n, 4096);
        assert_eq!(configs.q_proj.k, 2048);
        assert_eq!(configs.q_proj.dtype, DType::Bf16);
        assert_eq!(configs.wk_weights_proj.n, 160);
        assert_eq!(configs.wk_weights_proj.k, 6144);
        assert_eq!(configs.wk_weights_proj.dtype, DType::Bf16);

        assert_eq!(configs.index_cache_append.index_dim, 128);
        assert_eq!(configs.index_cache_append.block_size, 64);
        assert_eq!(configs.index_cache_append.quant_block_size, 128);
        assert_eq!(configs.index_cache_append.input_dtype, DType::Bf16);
        assert_eq!(configs.index_cache_append.cache_dtype, DType::Fp8E4m3);
        assert_eq!(configs.index_cache_append.scale_format, "ue8m0");
        assert_eq!(
            configs.index_cache_append.cache_format,
            "page_planar_fp8_fp32_scale"
        );

        assert_eq!(configs.prefill_logits.num_heads, 64);
        assert_eq!(configs.decode_logits.num_heads, 64);
        assert_eq!(configs.prefill_logits.head_dim, 128);
        assert_eq!(configs.decode_logits.head_dim, 128);
        assert_eq!(configs.prefill_logits.num_sequences, 1);
        assert_eq!(configs.prefill_topk.num_sequences, 1);
        assert_eq!(configs.prefill_topk.top_k, 2048);
        assert_eq!(configs.prefill_logits.span_mode, "single_causal_tail");
        assert_eq!(configs.prefill_topk.span_mode, "single_causal_tail");
        assert!(!configs.prefill_logits.clean_logits);
        assert_eq!(configs.decode_topk.top_k, 2048);
        assert_eq!(configs.decode_topk.logits_row_stride, LOGITS_ROW_STRIDE);
        assert_eq!(configs.decode_logits.max_model_len, MAX_MODEL_LEN);
        assert_eq!(configs.decode_logits.block_size, 64);
        assert_eq!(configs.decode_logits.q_dtype, DType::Fp8E4m3);
        assert_eq!(configs.decode_logits.cache_dtype, DType::Fp8E4m3);
        assert_eq!(configs.decode_logits.scale_dtype, DType::Fp32);
        assert_eq!(configs.decode_logits.weight_dtype, DType::Fp32);
        assert_eq!(configs.decode_logits.output_dtype, DType::Fp32);
        assert_eq!(
            configs.decode_logits.cache_format,
            "page_planar_fp8_fp32_scale"
        );
        assert!(!configs.decode_logits.clean_logits);
        assert_eq!(configs.decode_topk.logits_dtype, DType::Fp32);
        assert_eq!(configs.decode_topk.index_dtype, "int32");
    }

    #[test]
    fn all_elementwise_byte_formulas_use_model_h32() {
        let e = elementwise_configs(&cfg(1)).unwrap();
        let bytes = |config: &crate::timing::kernels::ElementwiseKernelConfig| {
            (
                config.input_bytes_per_token.get(),
                config.output_bytes_per_token.get(),
            )
        };
        assert_eq!(bytes(&e.k_layernorm), (256, 256));
        // Fused with the `torch.cat` that rebuilds q/k, so it moves head_dim
        // per head (33 * 128 * 2 = 8448) plus the gathered cos/sin table.
        assert_eq!(bytes(&e.rope), (8_704, 8_448));
        assert_eq!(bytes(&e.q_quant), (8_192, 4_224));
        assert_eq!(bytes(&e.weight_scale), (192, 128));
        assert_eq!(bytes(&e.topk_buffer_fill), (0, 8_192));
        assert_eq!(bytes(&e.prefill_cache_gather), (136, 132));
        assert_eq!(bytes(&e.decode_pack), (4_224, 4_224));
        assert_eq!(bytes(&e.decode_unpack), (8_192, 8_192));
    }

    #[test]
    fn next_n_one_and_two_reach_both_decode_kernel_configs() {
        for next_n in [1, 2] {
            let configs = subkernel_configs(&cfg(next_n)).unwrap();
            assert_eq!(configs.decode_logits.next_n, next_n);
            assert_eq!(configs.decode_topk.next_n, next_n);
            assert_eq!(configs.decode_logits.context_mode, "uniform");
            assert_eq!(configs.decode_topk.context_mode, "uniform");
            assert_eq!(configs.decode_logits.page_mapping, "unique_scattered");
        }
    }

    #[test]
    fn invalid_glm_specific_config_fails_before_kernel_builds() {
        for mutate in [
            |cfg: &mut DsaIndexerConfig| cfg.model_num_index_heads = 64.into(),
            |cfg: &mut DsaIndexerConfig| cfg.profile_num_index_heads = 32.into(),
            |cfg: &mut DsaIndexerConfig| cfg.hidden_dim = 4096.into(),
            |cfg: &mut DsaIndexerConfig| cfg.index_head_dim = 64.into(),
            |cfg: &mut DsaIndexerConfig| cfg.top_k = 1024,
            |cfg: &mut DsaIndexerConfig| cfg.cache_dtype = DType::Bf16,
        ] {
            let mut config = cfg(1);
            mutate(&mut config);
            assert!(matches!(
                validate_config(&config),
                Err(BuildError::FitFailed { .. })
            ));
        }
        assert!(matches!(
            validate_config(&cfg(3)),
            Err(BuildError::FitFailed { reason, .. }) if reason.contains("next_n")
        ));

        for mutate in [
            |cfg: &mut DsaIndexerConfig| cfg.input_dtype = DType::Fp32,
            |cfg: &mut DsaIndexerConfig| cfg.q_dtype = DType::Bf16,
            |cfg: &mut DsaIndexerConfig| cfg.scale_dtype = DType::Bf16,
            |cfg: &mut DsaIndexerConfig| cfg.weight_dtype = DType::Bf16,
            |cfg: &mut DsaIndexerConfig| cfg.logits_dtype = DType::Bf16,
            |cfg: &mut DsaIndexerConfig| cfg.index_dtype = "int64".to_string(),
        ] {
            let mut config = cfg(1);
            mutate(&mut config);
            assert!(matches!(
                validate_config(&config),
                Err(BuildError::FitFailed { .. })
            ));
        }
    }

    #[test]
    fn input_normalization_derives_active_and_padded_decode_rows() {
        let input = DsaIndexerInput {
            num_new_tokens: 37,
            prefill_query_key_pairs: vec![(8, 128), (16, 256)],
            decode: Some(DsaIndexerDecodeInput {
                batch_size: 12,
                context_len: 8192,
                requires_padding: true,
            }),
        };
        let normalized = normalize_input(&input, 2, MAX_MODEL_LEN).unwrap();
        assert_eq!(normalized.active_query_rows, 48);
        assert_eq!(normalized.padded_decode_rows, 24);
        assert_eq!(normalized.decode_logits.as_ref().unwrap().batch_size, 12);
        assert_eq!(normalized.decode_topk.as_ref().unwrap().context_len, 8192);
        assert_eq!(input.num_new_tokens, 37);

        let mut unpadded = input;
        unpadded.decode.as_mut().unwrap().requires_padding = false;
        let normalized = normalize_input(&unpadded, 2, MAX_MODEL_LEN).unwrap();
        assert_eq!(normalized.active_query_rows, 48);
        assert_eq!(normalized.padded_decode_rows, 0);
    }

    #[test]
    fn absent_decode_and_empty_prefill_are_zero_work() {
        let normalized = normalize_input(&DsaIndexerInput::default(), 1, MAX_MODEL_LEN).unwrap();
        assert_eq!(normalized.active_query_rows, 0);
        assert_eq!(normalized.padded_decode_rows, 0);
        assert!(normalized.decode_logits.is_none());
        assert!(normalized.decode_topk.is_none());
    }

    #[test]
    fn malformed_prefill_decode_and_checked_overflow_fail_clearly() {
        for pair in [(0, 1), (1, 0), (2, 1)] {
            let input = DsaIndexerInput {
                prefill_query_key_pairs: vec![pair],
                ..Default::default()
            };
            assert!(normalize_input(&input, 1, MAX_MODEL_LEN).is_err());
        }
        for (batch_size, context_len) in [(0, 1), (1, 0), (1, MAX_MODEL_LEN + 1)] {
            let input = DsaIndexerInput {
                decode: Some(DsaIndexerDecodeInput {
                    batch_size,
                    context_len,
                    requires_padding: true,
                }),
                ..Default::default()
            };
            assert!(normalize_input(&input, 2, MAX_MODEL_LEN).is_err());
        }
        let overflow = DsaIndexerInput {
            decode: Some(DsaIndexerDecodeInput {
                batch_size: u32::MAX,
                context_len: 1,
                requires_padding: true,
            }),
            ..Default::default()
        };
        assert!(normalize_input(&overflow, 2, MAX_MODEL_LEN)
            .err()
            .expect("decode multiplication must overflow")
            .contains("overflows u32"));

        let active_overflow = DsaIndexerInput {
            prefill_query_key_pairs: vec![(u32::MAX, u32::MAX), (1, 1)],
            ..Default::default()
        };
        assert!(normalize_input(&active_overflow, 1, MAX_MODEL_LEN)
            .err()
            .expect("active-row sum must overflow")
            .contains("overflows u32"));
    }

    #[test]
    fn aggregate_prefill_log_serializes_the_complete_pair_vector() {
        let slot: SlotInput = DsaIndexerPrefillLog {
            prefill_query_key_pairs: vec![(1, 1), (128, 2049)],
        }
        .into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"prefill_query_key_pairs":[[1,1],[128,2049]]})
        );
    }
}
