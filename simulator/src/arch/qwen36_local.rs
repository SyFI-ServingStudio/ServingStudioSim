//! Qwen3.6-35B-A3B-FP8 text decoder on one local H200 (TP1/EP1).
//!
//! The checkpoint is heterogeneous: three Gated `DeltaNet` layers followed by one
//! gated-GQA layer, repeated ten times. Both layer variants end at the same
//! normalized-hidden boundary and are followed by the same local `MoE` sequence.
//! Routed and shared experts may use concurrent CUDA streams in production, but
//! they share one H200's compute and memory resources. Their isolated leaf costs
//! therefore compose conservatively as a local sum until a measured compound
//! overlap model exists. There are no dispatch, combine, collective, TP, EP,
//! network, vision, or MTP children.

use std::path::Path;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

use crate::arch::config::ModelSpec;
use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::op::Op;
use crate::timing::kernels::{ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput};
use crate::timing::routing::RoutingDistribution;
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, DType, Dim, Evaluator,
    FlatCostNode, LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::{
    Qwen36GatedGqaLocalWorklet, Qwen36GatedGqaLocalWorkletConfig, Qwen36GatedGqaLocalWorkletInput,
    Qwen36GatedGqaLocalWorkletResolved, Qwen36GdnLocalWorklet, Qwen36GdnLocalWorkletConfig,
    Qwen36GdnLocalWorkletInput, Qwen36GdnLocalWorkletResolved, Qwen36HeadLocalWorklet,
    Qwen36HeadLocalWorkletConfig, Qwen36HeadLocalWorkletInput, Qwen36HeadLocalWorkletResolved,
    Qwen36MoeFinalizeLocalWorklet, Qwen36MoeFinalizeLocalWorkletConfig,
    Qwen36MoeFinalizeLocalWorkletInput, Qwen36MoeFinalizeLocalWorkletResolved,
    Qwen36MoeRouterLocalWorklet, Qwen36MoeRouterLocalWorkletConfig,
    Qwen36MoeRouterLocalWorkletInput, Qwen36MoeRouterLocalWorkletResolved,
    Qwen36SharedExpertLocalWorklet, Qwen36SharedExpertLocalWorkletConfig,
    Qwen36SharedExpertLocalWorkletInput, Qwen36SharedExpertLocalWorkletResolved,
    VllmFp8MoeExpertComputeLocalWorklet, VllmFp8MoeExpertComputeLocalWorkletConfig,
    VllmFp8MoeExpertComputeLocalWorkletInput, VllmFp8MoeExpertComputeLocalWorkletResolved,
};

const CHECKPOINT_LAYERS: u32 = 40;
const HIDDEN: u32 = 2_048;
const VOCAB: u32 = 248_320;
const GQA_HEADS: u32 = 16;
const GQA_KV_HEADS: u32 = 2;
const GQA_HEAD_DIM: u32 = 256;
const ROPE_DIM: u32 = 64;
const GDN_KEY_HEADS: u32 = 16;
const GDN_VALUE_HEADS: u32 = 32;
const GDN_KEY_DIM: u32 = 128;
const GDN_VALUE_DIM: u32 = 128;
const GDN_CONV_KERNEL: u32 = 4;
const NUM_EXPERTS: u32 = 256;
const TOP_K: u32 = 8;
const ALIGN_BLOCK: u32 = 16;
const EXPERT_WIDTH: u32 = 512;
const NUM_SHARED_EXPERTS: u32 = 1;
const FP8_GROUP: u32 = 128;
const EMBED_INPUT_BYTES: u32 = 8 + HIDDEN * 2;
const EMBED_OUTPUT_BYTES: u32 = HIDDEN * 2;

const RESIDUAL_NORM_BACKENDS: &[&str] = &["vllm_cuda"];
const QK_NORM_BACKENDS: &[&str] = &["flashinfer"];
const DENSE_FP8_QUANT_BACKENDS: &[&str] = &["vllm_cuda"];
const DENSE_FP8_GEMM_BACKENDS: &[&str] = &["deepgemm"];
const BF16_GEMM_BACKENDS: &[&str] = &["torch_linear"];
/// Slots vLLM hands to a Triton kernel, torch-compile output included.
const ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
/// Slots whose vLLM source is eager tensor arithmetic and therefore land on
/// torch's `TensorIterator`. Not interchangeable with the Triton curve: at these
/// byte rates both kernels are launch-bound and torch's is the heavier one, so
/// costing the shared-expert gate path on `triton` under-predicted it by 52-72%.
const TORCH_ELEMENTWISE_BACKENDS: &[&str] = &["torch"];
const GDN_BACKENDS: &[&str] = &["vllm_triton"];
/// The chunked delta rule is the one GDN launch vLLM does NOT route to Triton.
/// `ChunkGatedDeltaRule._resolve_gdn_prefill_backend` answers the default `auto`
/// request with `flashinfer` on every SM90 part, so a measured H200 run always
/// executes the fused CUTLASS kernel and never the six-launch FLA path.
const GDN_PREFILL_DELTA_RULE_BACKENDS: &[&str] = &["flashinfer"];
/// Both FA2 and FA3, resolved per lookup by best-of-N wallclock — not a
/// preference order. FA3 wins the compute-bound prefill; FA2's decode kernel is
/// the faster of the two by a wide margin at every shape measured so far (FA3
/// decode never exceeds ~1.3 TB/s anywhere in profile.db, against FA2's ~4.5),
/// and decode attention dominates this model's long-context iteration cost.
const GQA_ATTN_BACKENDS: &[&str] = &["fa2", "fa3"];
const KV_APPEND_BACKENDS: &[&str] = &["vllm_cuda"];
const ROUTING_BACKENDS: &[&str] = &["vllm_cuda"];
// The routed activation quant is vLLM's own `per_token_group_quant_8bit_kernel`
// -- the same kernel every dense projection here uses -- not FlashInfer's
// grouped `scale_1x128_kernel`. nsys confirms it on both routed quants.
const ROUTED_QUANT_BACKENDS: &[&str] = &["vllm_cuda"];
// The routed expert GEMMs are vLLM's Triton `fused_moe_kernel`, one launch per
// half. At EP1 vLLM never runs the TRT-LLM grouped GEMM: there is no EP
// prepare/finalize to permute tokens into per-expert order first.
const ROUTED_GEMM_BACKENDS: &[&str] = &["vllm_triton"];
const FINALIZE_BACKENDS: &[&str] = &["flashinfer_trtllm"];

/// Parsed and pinned text-only identity. `logical_num_layers` is the requested
/// checkpoint prefix; `num_layers` is the effective simulated prefix after the
/// standard `sim_num_layers` override.
#[derive(Clone, Debug)]
pub struct Qwen36ModelCfg {
    pub hidden: Dim,
    pub vocab_size: Dim,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub rope_dim: Dim,
    pub linear_num_key_heads: Dim,
    pub linear_num_value_heads: Dim,
    pub linear_key_head_dim: Dim,
    pub linear_value_head_dim: Dim,
    pub linear_conv_kernel_dim: Dim,
    pub num_experts: Dim,
    pub top_k: u32,
    pub alignment_block_size: u32,
    pub moe_intermediate: Dim,
    pub shared_intermediate: Dim,
    pub num_shared_experts: u32,
    pub activation_dtype: DType,
    pub conv_state_dtype: DType,
    pub ssm_state_dtype: DType,
    pub logical_num_layers: u32,
    pub num_layers: u32,
    pub num_gdn_layers: u32,
    pub num_gqa_layers: u32,
}

impl Qwen36ModelCfg {
    pub fn from_json(path: &Path, spec: &ModelSpec) -> Result<Self> {
        ensure!(spec.fp8, "qwen36_local requires fp8=true");
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading Qwen3.6 config {}", path.display()))?;
        parse_model_json(&text, spec.num_layers, spec.sim_num_layers)
            .with_context(|| format!("validating Qwen3.6 config {}", path.display()))
    }
}

#[derive(Deserialize)]
struct JsonQwen36Config {
    architectures: Vec<String>,
    model_type: String,
    tie_word_embeddings: bool,
    text_config: JsonQwen36TextConfig,
    quantization_config: JsonQwen36QuantConfig,
    // Presence is deliberate: this checkpoint is multimodal, while this arch
    // selects only `text_config` and never constructs the vision branch.
    vision_config: serde_json::Value,
}

#[derive(Deserialize)]
struct JsonQwen36TextConfig {
    model_type: String,
    dtype: String,
    vocab_size: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    partial_rotary_factor: f64,
    attn_output_gate: bool,
    full_attention_interval: u32,
    layer_types: Vec<String>,
    linear_conv_kernel_dim: u32,
    linear_key_head_dim: u32,
    linear_value_head_dim: u32,
    linear_num_key_heads: u32,
    linear_num_value_heads: u32,
    mamba_ssm_dtype: String,
    moe_intermediate_size: u32,
    shared_expert_intermediate_size: u32,
    num_experts_per_tok: u32,
    num_experts: u32,
    tie_word_embeddings: bool,
    mtp_num_hidden_layers: u32,
}

#[derive(Deserialize)]
struct JsonQwen36QuantConfig {
    activation_scheme: String,
    fmt: String,
    quant_method: String,
    modules_to_not_convert: Vec<String>,
    weight_block_size: Vec<u32>,
}

fn parse_model_json(
    text: &str,
    requested_layers: Option<u32>,
    simulated_layers: Option<u32>,
) -> Result<Qwen36ModelCfg> {
    let raw: JsonQwen36Config = serde_json::from_str(text).context("parsing JSON")?;
    ensure!(
        raw.architectures == ["Qwen3_5MoeForConditionalGeneration"],
        "architectures must identify Qwen3_5MoeForConditionalGeneration"
    );
    ensure!(
        raw.model_type == "qwen3_5_moe",
        "outer model_type must be qwen3_5_moe"
    );
    ensure!(!raw.tie_word_embeddings, "outer embeddings must be untied");
    ensure!(
        raw.vision_config.is_object(),
        "pinned checkpoint must contain vision_config"
    );

    let t = raw.text_config;
    ensure!(
        t.model_type == "qwen3_5_moe_text",
        "text model_type must be qwen3_5_moe_text"
    );
    ensure!(t.dtype == "bfloat16", "text dtype must be bfloat16");
    ensure!(!t.tie_word_embeddings, "text embeddings must be untied");
    for (name, actual, expected) in [
        ("hidden_size", t.hidden_size, HIDDEN),
        ("vocab_size", t.vocab_size, VOCAB),
        ("num_hidden_layers", t.num_hidden_layers, CHECKPOINT_LAYERS),
        ("num_attention_heads", t.num_attention_heads, GQA_HEADS),
        ("num_key_value_heads", t.num_key_value_heads, GQA_KV_HEADS),
        ("head_dim", t.head_dim, GQA_HEAD_DIM),
        ("full_attention_interval", t.full_attention_interval, 4),
        (
            "linear_num_key_heads",
            t.linear_num_key_heads,
            GDN_KEY_HEADS,
        ),
        (
            "linear_num_value_heads",
            t.linear_num_value_heads,
            GDN_VALUE_HEADS,
        ),
        ("linear_key_head_dim", t.linear_key_head_dim, GDN_KEY_DIM),
        (
            "linear_value_head_dim",
            t.linear_value_head_dim,
            GDN_VALUE_DIM,
        ),
        (
            "linear_conv_kernel_dim",
            t.linear_conv_kernel_dim,
            GDN_CONV_KERNEL,
        ),
        ("num_experts", t.num_experts, NUM_EXPERTS),
        ("num_experts_per_tok", t.num_experts_per_tok, TOP_K),
        (
            "moe_intermediate_size",
            t.moe_intermediate_size,
            EXPERT_WIDTH,
        ),
        (
            "shared_expert_intermediate_size",
            t.shared_expert_intermediate_size,
            EXPERT_WIDTH,
        ),
        ("mtp_num_hidden_layers", t.mtp_num_hidden_layers, 1),
    ] {
        ensure!(
            actual == expected,
            "{name} must be {expected}, got {actual}"
        );
    }
    ensure!(
        t.partial_rotary_factor == 0.25,
        "partial_rotary_factor must be 0.25"
    );
    ensure!(t.attn_output_gate, "attention output gate must be enabled");
    ensure!(
        t.mamba_ssm_dtype == "float32",
        "GDN recurrent state must be float32"
    );
    let expected_layers: Vec<&str> = (0..CHECKPOINT_LAYERS)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    ensure!(
        t.layer_types.iter().map(String::as_str).eq(expected_layers),
        "layer_types must match the exact repeating 3 GDN / 1 gated-GQA schedule"
    );

    let q = raw.quantization_config;
    ensure!(
        q.activation_scheme == "dynamic",
        "FP8 activation scheme must be dynamic"
    );
    ensure!(q.fmt == "e4m3", "FP8 format must be e4m3");
    ensure!(q.quant_method == "fp8", "quant_method must be fp8");
    ensure!(
        q.weight_block_size == [FP8_GROUP, FP8_GROUP],
        "weight block must be 128x128"
    );
    for required in ["lm_head", "model.embed_tokens"] {
        ensure!(
            q.modules_to_not_convert
                .iter()
                .any(|entry| entry == required),
            "missing FP8 exclusion {required}"
        );
    }
    for layer in 0..CHECKPOINT_LAYERS {
        let shared_gate = format!("model.language_model.layers.{layer}.mlp.shared_expert_gate");
        ensure!(
            q.modules_to_not_convert
                .iter()
                .any(|entry| entry == &shared_gate),
            "missing shared gate FP8 exclusion for layer {layer}"
        );
        if layer % 4 != 3 {
            let ba = format!("model.language_model.layers.{layer}.linear_attn.in_proj_ba");
            ensure!(
                q.modules_to_not_convert.iter().any(|entry| entry == &ba),
                "missing BA FP8 exclusion for layer {layer}"
            );
        }
    }

    let logical_num_layers = requested_layers.unwrap_or(CHECKPOINT_LAYERS);
    let num_layers = simulated_layers.unwrap_or(logical_num_layers);
    ensure!(
        (1..=CHECKPOINT_LAYERS).contains(&logical_num_layers),
        "num_layers must be in 1..=40"
    );
    ensure!(
        (1..=logical_num_layers).contains(&num_layers),
        "sim_num_layers must be in 1..=num_layers"
    );
    let num_gqa_layers = num_layers / 4;
    let num_gdn_layers = num_layers - num_gqa_layers;

    Ok(Qwen36ModelCfg {
        hidden: HIDDEN.into(),
        vocab_size: VOCAB.into(),
        num_qo_heads: GQA_HEADS.into(),
        num_kv_heads: GQA_KV_HEADS.into(),
        head_dim: GQA_HEAD_DIM.into(),
        rope_dim: ROPE_DIM.into(),
        linear_num_key_heads: GDN_KEY_HEADS.into(),
        linear_num_value_heads: GDN_VALUE_HEADS.into(),
        linear_key_head_dim: GDN_KEY_DIM.into(),
        linear_value_head_dim: GDN_VALUE_DIM.into(),
        linear_conv_kernel_dim: GDN_CONV_KERNEL.into(),
        num_experts: NUM_EXPERTS.into(),
        top_k: TOP_K,
        alignment_block_size: ALIGN_BLOCK,
        moe_intermediate: EXPERT_WIDTH.into(),
        shared_intermediate: EXPERT_WIDTH.into(),
        num_shared_experts: NUM_SHARED_EXPERTS,
        activation_dtype: DType::Bf16,
        conv_state_dtype: DType::Bf16,
        ssm_state_dtype: DType::Fp32,
        logical_num_layers,
        num_layers,
        num_gdn_layers,
        num_gqa_layers,
    })
}

#[derive(Clone, Debug)]
pub struct Qwen36LocalParallel {
    pub gpu_name: String,
}

pub struct Qwen36LocalConfigs {
    pub embedding: ElementwiseKernelConfig,
    pub gdn: Qwen36GdnLocalWorkletConfig,
    pub gated_gqa: Qwen36GatedGqaLocalWorkletConfig,
    pub router: Qwen36MoeRouterLocalWorkletConfig,
    pub routed_expert: VllmFp8MoeExpertComputeLocalWorkletConfig,
    pub shared_expert: Qwen36SharedExpertLocalWorkletConfig,
    pub finalize: Qwen36MoeFinalizeLocalWorkletConfig,
    pub head: Qwen36HeadLocalWorkletConfig,
    pub logical_num_layers: u32,
    pub num_layers: u32,
    pub num_gdn_layers: u32,
    pub num_gqa_layers: u32,
}

pub struct Qwen36LocalResolved {
    pub embedding: ElementwiseKernelConfig,
    pub gdn: Qwen36GdnLocalWorkletResolved,
    pub gated_gqa: Qwen36GatedGqaLocalWorkletResolved,
    pub router: Qwen36MoeRouterLocalWorkletResolved,
    pub routed_expert: VllmFp8MoeExpertComputeLocalWorkletResolved,
    pub shared_expert: Qwen36SharedExpertLocalWorkletResolved,
    pub finalize: Qwen36MoeFinalizeLocalWorkletResolved,
    pub head: Qwen36HeadLocalWorkletResolved,
    pub logical_num_layers: u32,
    pub num_layers: u32,
    pub num_gdn_layers: u32,
    pub num_gqa_layers: u32,
}

pub struct Qwen36LocalModel {
    pub name: String,
    pub embedding: Op<ElementwiseKernel>,
    pub gdn: Qwen36GdnLocalWorklet,
    pub gated_gqa: Qwen36GatedGqaLocalWorklet,
    pub router: Qwen36MoeRouterLocalWorklet,
    pub routed_expert: VllmFp8MoeExpertComputeLocalWorklet,
    pub shared_expert: Qwen36SharedExpertLocalWorklet,
    pub finalize: Qwen36MoeFinalizeLocalWorklet,
    pub head: Qwen36HeadLocalWorklet,
    pub logical_num_layers: u32,
    pub num_layers: u32,
    pub num_gdn_layers: u32,
    pub num_gqa_layers: u32,
    pub total_kv_bytes_per_token: Dim,
    pub recurrent_state_bytes_per_request: Dim,
    pub recurrent_checkpoint_interval_tokens: Dim,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

/// At EP1 the one rank owns every expert, so the grouped GEMM's local shard IS
/// the global distribution — no `split_for_ep` slicing, just the whole vector.
///
/// Routing matters here even without expert parallelism. Skew does not change
/// the total token-expert selections, so it leaves total FLOPs alone; what it
/// changes is how those selections are grouped, and a grouped GEMM pays per
/// group. Feeding a measured `expert_popularity` profile in is therefore the
/// difference between costing the routing vLLM actually produced and costing an
/// idealized balanced one.
#[must_use]
pub fn build_configs(
    model: &Qwen36ModelCfg,
    parallel: &Qwen36LocalParallel,
    routing: &RoutingDistribution,
) -> Qwen36LocalConfigs {
    let gpu = parallel.gpu_name.clone();
    assert_eq!(
        routing.num_experts(),
        model.num_experts.get(),
        "routing distribution has {} experts, model has {}",
        routing.num_experts(),
        model.num_experts.get(),
    );
    let ppm = routing.ppm().to_vec();
    Qwen36LocalConfigs {
        embedding: ElementwiseKernelConfig {
            backends: ELEMENTWISE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            input_bytes_per_token: EMBED_INPUT_BYTES.into(),
            output_bytes_per_token: EMBED_OUTPUT_BYTES.into(),
        },
        gdn: Qwen36GdnLocalWorkletConfig {
            hidden: model.hidden.clone(),
            num_key_heads: model.linear_num_key_heads.clone(),
            num_value_heads: model.linear_num_value_heads.clone(),
            key_head_dim: model.linear_key_head_dim.clone(),
            value_head_dim: model.linear_value_head_dim.clone(),
            conv_kernel_size: model.linear_conv_kernel_dim.clone(),
            activation_dtype: model.activation_dtype,
            conv_state_dtype: model.conv_state_dtype,
            ssm_state_dtype: model.ssm_state_dtype,
            gpu_name: gpu.clone(),
            residual_rms_norm_backends: RESIDUAL_NORM_BACKENDS.to_vec(),
            fp8_quant_backends: DENSE_FP8_QUANT_BACKENDS.to_vec(),
            fp8_gemm_backends: DENSE_FP8_GEMM_BACKENDS.to_vec(),
            bf16_gemm_backends: BF16_GEMM_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
            causal_conv_prefill_backends: GDN_BACKENDS.to_vec(),
            prefill_post_conv_backends: GDN_BACKENDS.to_vec(),
            chunk_delta_rule_backends: GDN_PREFILL_DELTA_RULE_BACKENDS.to_vec(),
            causal_conv_decode_backends: GDN_BACKENDS.to_vec(),
            recurrent_decode_backends: GDN_BACKENDS.to_vec(),
            gated_rms_norm_backends: GDN_BACKENDS.to_vec(),
        },
        gated_gqa: Qwen36GatedGqaLocalWorkletConfig {
            hidden: model.hidden.clone(),
            num_qo_heads: model.num_qo_heads.clone(),
            num_kv_heads: model.num_kv_heads.clone(),
            head_dim: model.head_dim.clone(),
            rope_dim: model.rope_dim.clone(),
            activation_dtype: model.activation_dtype,
            gpu_name: gpu.clone(),
            kv_cache_block_size: 16,
            kv_cache_layout: "NHD".into(),
            kv_scale_granularity: "tensor".into(),
            residual_rms_norm_backends: RESIDUAL_NORM_BACKENDS.to_vec(),
            fp8_quant_backends: DENSE_FP8_QUANT_BACKENDS.to_vec(),
            fp8_gemm_backends: DENSE_FP8_GEMM_BACKENDS.to_vec(),
            qk_rms_norm_backends: QK_NORM_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
            attention_backends: GQA_ATTN_BACKENDS.to_vec(),
            kv_cache_append_backends: KV_APPEND_BACKENDS.to_vec(),
        },
        router: Qwen36MoeRouterLocalWorkletConfig {
            hidden: model.hidden.clone(),
            num_experts: model.num_experts.clone(),
            top_k: model.top_k,
            block_size: model.alignment_block_size,
            activation_dtype: model.activation_dtype,
            gpu_name: gpu.clone(),
            bf16_gemm_backends: BF16_GEMM_BACKENDS.to_vec(),
            fused_topk_backends: ROUTING_BACKENDS.to_vec(),
            align_backends: ROUTING_BACKENDS.to_vec(),
        },
        routed_expert: VllmFp8MoeExpertComputeLocalWorkletConfig {
            hidden: model.hidden.clone(),
            moe_intermediate: model.moe_intermediate.clone(),
            num_experts: model.num_experts.clone(),
            ep_size: 1,
            top_k: model.top_k,
            activation_dtype: model.activation_dtype,
            gpu_name: gpu.clone(),
            act_backends: ELEMENTWISE_BACKENDS.to_vec(),
            fp8_quant_backends: ROUTED_QUANT_BACKENDS.to_vec(),
            fp8_grouped_gemm_backends: ROUTED_GEMM_BACKENDS.to_vec(),
            local_ppm: ppm.clone(),
        },
        shared_expert: Qwen36SharedExpertLocalWorkletConfig {
            hidden: model.hidden.clone(),
            intermediate: model.shared_intermediate.clone(),
            num_shared_experts: model.num_shared_experts,
            activation_dtype: model.activation_dtype,
            gpu_name: gpu.clone(),
            fp8_quant_backends: DENSE_FP8_QUANT_BACKENDS.to_vec(),
            fp8_gemm_backends: DENSE_FP8_GEMM_BACKENDS.to_vec(),
            bf16_gemm_backends: BF16_GEMM_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
            gate_elementwise_backends: TORCH_ELEMENTWISE_BACKENDS.to_vec(),
        },
        finalize: Qwen36MoeFinalizeLocalWorkletConfig {
            hidden: model.hidden.clone(),
            num_experts: model.num_experts.clone(),
            top_k: model.top_k,
            activation_dtype: model.activation_dtype,
            local_ppm: ppm,
            gpu_name: gpu.clone(),
            finalize_backends: FINALIZE_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        },
        head: Qwen36HeadLocalWorkletConfig {
            hidden: model.hidden.clone(),
            vocab_size: model.vocab_size.clone(),
            activation_dtype: model.activation_dtype,
            gpu_name: gpu,
            residual_rms_norm_backends: RESIDUAL_NORM_BACKENDS.to_vec(),
            bf16_gemm_backends: BF16_GEMM_BACKENDS.to_vec(),
        },
        logical_num_layers: model.logical_num_layers,
        num_layers: model.num_layers,
        num_gdn_layers: model.num_gdn_layers,
        num_gqa_layers: model.num_gqa_layers,
    }
}

#[must_use]
pub fn resolve_configs(cfg: &Qwen36LocalConfigs) -> Qwen36LocalResolved {
    Qwen36LocalResolved {
        embedding: cfg.embedding.clone(),
        gdn: Qwen36GdnLocalWorklet::resolve_config(&cfg.gdn),
        gated_gqa: Qwen36GatedGqaLocalWorklet::resolve_config(&cfg.gated_gqa),
        router: Qwen36MoeRouterLocalWorklet::resolve_config(&cfg.router),
        routed_expert: VllmFp8MoeExpertComputeLocalWorklet::resolve_config(&cfg.routed_expert),
        shared_expert: Qwen36SharedExpertLocalWorklet::resolve_config(&cfg.shared_expert),
        finalize: Qwen36MoeFinalizeLocalWorklet::resolve_config(&cfg.finalize),
        head: Qwen36HeadLocalWorklet::resolve_config(&cfg.head),
        logical_num_layers: cfg.logical_num_layers,
        num_layers: cfg.num_layers,
        num_gdn_layers: cfg.num_gdn_layers,
        num_gqa_layers: cfg.num_gqa_layers,
    }
}

/// One full-attention layer's KV bytes for one token — K and V together. This is
/// the unit vLLM's hybrid page alignment measures the SSM page against, so it is
/// factored out rather than inlined into [`total_kv_bytes_per_token`].
fn attention_page_bytes_per_token_per_layer(resolved: &Qwen36LocalResolved) -> Dim {
    let attn = &resolved.gated_gqa.attention;
    2 * attn.num_kv_heads.clone()
        * attn.head_dim.clone()
        * Dim::param("kv_bytes", attn.kv_dtype().size_bytes())
}

fn total_kv_bytes_per_token(resolved: &Qwen36LocalResolved) -> Dim {
    attention_page_bytes_per_token_per_layer(resolved)
        * Dim::param("num_gqa_layers", resolved.num_gqa_layers)
}

/// One GDN layer's SSM state: the `(num_value_heads, key_head_dim,
/// value_head_dim)` recurrent tensor.
fn ssm_state_bytes_per_layer(resolved: &Qwen36LocalResolved) -> Dim {
    let gdn = &resolved.gdn.raw_cfg;
    gdn.num_value_heads.clone()
        * gdn.key_head_dim.clone()
        * gdn.value_head_dim.clone()
        * Dim::param("ssm_state_bytes", gdn.ssm_state_dtype.size_bytes())
}

/// One GDN layer's causal-conv window over the mixed q/k/v channel stack. vLLM
/// keeps `conv_kernel_size - 1` taps, because the current token supplies the
/// last one.
fn conv_state_bytes_per_layer(resolved: &Qwen36LocalResolved) -> Dim {
    let gdn = &resolved.gdn.raw_cfg;
    let channels = 2 * gdn.num_key_heads.clone() * gdn.key_head_dim.clone()
        + gdn.num_value_heads.clone() * gdn.value_head_dim.clone();
    channels
        * (gdn.conv_kernel_size.clone() - 1)
        * Dim::param("conv_state_bytes", gdn.conv_state_dtype.size_bytes())
}

/// One GDN layer's mamba page: SSM state plus conv window. Both halves are
/// required to resume a sequence, so a checkpoint that stores one without the
/// other is unusable — they are always retained and charged together.
fn mamba_page_bytes_per_layer(resolved: &Qwen36LocalResolved) -> Dim {
    ssm_state_bytes_per_layer(resolved) + conv_state_bytes_per_layer(resolved)
}

/// Every GDN layer's mamba page **as allocated**.
///
/// vLLM's hybrid allocator gives the mamba and attention groups one common page
/// size: it picks the attention block from the mamba page
/// ([`recurrent_checkpoint_interval_tokens`]) and then pads the mamba page up to
/// that attention page, so the charged footprint is the padded one — for this
/// model 2,162,688 B/layer against a 2,146,304 B true state, the "Padding mamba
/// page size by 0.76%" the engine logs at startup.
fn recurrent_state_bytes_per_request(resolved: &Qwen36LocalResolved) -> Dim {
    recurrent_checkpoint_interval_tokens(resolved)
        * attention_page_bytes_per_token_per_layer(resolved)
        * Dim::param("num_gdn_layers", resolved.num_gdn_layers)
}

/// vLLM's hybrid `block_size`: the smallest attention-kernel-aligned token count
/// whose per-layer attention page covers one per-layer mamba page.
///
/// `attn_block_size = kernel_block_size * cdiv(mamba_page, kernel_block_size *
/// attn_page_per_token)`, where `kernel_block_size` is the attention KV-cache
/// block granularity the worklet already configures. Both operands are per-layer
/// per-rank, so the ratio is TP-invariant; this arch is TP1 anyway.
///
/// The numerator is the whole mamba page (SSM **and** conv) and the denominator
/// counts K **and** V. An earlier reading of vllm-ascend#7393 had it as SSM-only
/// over K-only, which lands on 2048 instead; the engine's own startup log for
/// this checkpoint ("Setting attention block size to 1056 tokens", "Padding
/// mamba page size by 0.76%") only reconstructs with both terms whole, and each
/// correction alone misses (2112 and 1024 respectively).
///
/// Returns 0 for a layer prefix with no GDN layer, matching the trait's "no
/// recurrent state" encoding.
fn recurrent_checkpoint_interval_tokens(resolved: &Qwen36LocalResolved) -> Dim {
    if resolved.num_gdn_layers == 0 {
        return Dim::param("no_gdn_layers", 0);
    }
    let kernel_block_size = Dim::param(
        "kv_cache_block_size",
        resolved.gated_gqa.raw_cfg.kv_cache_block_size,
    );
    let quantum = kernel_block_size.clone() * attention_page_bytes_per_token_per_layer(resolved);
    let mamba_page = mamba_page_bytes_per_layer(resolved);
    // cdiv without a helper: Dim has no ceiling divide, and the "+ quantum - 1"
    // form keeps the whole derivation visible in the rendered expression.
    kernel_block_size * ((mamba_page + quantum.clone() - 1) / quantum)
}

pub fn build(
    name: String,
    resolved: Qwen36LocalResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<Qwen36LocalModel, BuildError> {
    let total_kv_bytes_per_token = total_kv_bytes_per_token(&resolved);
    let recurrent_state_bytes_per_request = recurrent_state_bytes_per_request(&resolved);
    let recurrent_checkpoint_interval_tokens = recurrent_checkpoint_interval_tokens(&resolved);
    let embedding_name = format!("{name}.embedding");
    let embedding = Op::new(
        embedding_name.clone(),
        Arc::new(ElementwiseKernel::build(
            embedding_name,
            resolved.embedding,
            bridge,
        )?),
    );
    let mut model = Qwen36LocalModel {
        embedding,
        gdn: Qwen36GdnLocalWorklet::build(format!("{name}.gdn"), resolved.gdn, bridge)?,
        gated_gqa: Qwen36GatedGqaLocalWorklet::build(
            format!("{name}.gated_gqa"),
            resolved.gated_gqa,
            bridge,
        )?,
        router: Qwen36MoeRouterLocalWorklet::build(
            format!("{name}.router"),
            resolved.router,
            bridge,
        )?,
        routed_expert: VllmFp8MoeExpertComputeLocalWorklet::build(
            format!("{name}.routed_expert"),
            resolved.routed_expert,
            bridge,
        )?,
        shared_expert: Qwen36SharedExpertLocalWorklet::build(
            format!("{name}.shared_expert"),
            resolved.shared_expert,
            bridge,
        )?,
        finalize: Qwen36MoeFinalizeLocalWorklet::build(
            format!("{name}.finalize"),
            resolved.finalize,
            bridge,
        )?,
        head: Qwen36HeadLocalWorklet::build(format!("{name}.head"), resolved.head, bridge)?,
        name,
        logical_num_layers: resolved.logical_num_layers,
        num_layers: resolved.num_layers,
        num_gdn_layers: resolved.num_gdn_layers,
        num_gqa_layers: resolved.num_gqa_layers,
        total_kv_bytes_per_token,
        recurrent_state_bytes_per_request,
        recurrent_checkpoint_interval_tokens,
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    tracing::info!(
        "[build] Qwen3.6 local cost tree ({} leaf slots):\n{}",
        tree.n_slots(),
        tree.describe()
    );
    tracing::info!(
        "[build] Qwen3.6 local recurrent state: {} B/request over {} GDN layer(s), \
         reusable every {} tokens ({} B/token of full-attention KV)",
        model.recurrent_state_bytes_per_request.get(),
        model.num_gdn_layers,
        model.recurrent_checkpoint_interval_tokens.get(),
        model.total_kv_bytes_per_token.get(),
    );
    Ok(model)
}

#[derive(Debug)]
struct NormalizedInput {
    active_tokens: u32,
    request_count: u32,
    global_expert_selections: u32,
    gdn: Qwen36GdnLocalWorkletInput,
    gated_gqa: Qwen36GatedGqaLocalWorkletInput,
}

fn normalize_input(input: &UnifiedArchInput) -> std::result::Result<NormalizedInput, String> {
    if input.groups.len() != 1 {
        return Err(format!(
            "qwen36_local requires exactly one local group, got {}",
            input.groups.len()
        ));
    }
    if !input.tokens_per_source_rank.is_empty() {
        return Err("qwen36_local requires empty tokens_per_source_rank".into());
    }
    let group = &input.groups[0];
    let mut prefill_tokens = 0_u32;
    let mut lengths = Vec::with_capacity(group.prefill_chunk_pairs.len());
    let mut has_state = Vec::with_capacity(group.prefill_chunk_pairs.len());
    for (index, &(prefix, append)) in group.prefill_chunk_pairs.iter().enumerate() {
        if append == 0 {
            return Err(format!("prefill request {index} append must be positive"));
        }
        // The exact prefill KV state is already carried by this pair. Validate
        // it here, but do not fold it into `total_kv_len`: the live worker and
        // timing-predict wire define that field as the decode-member aggregate.
        let _prefill_total = prefix
            .checked_add(append)
            .ok_or_else(|| format!("prefill request {index} prefix+append overflow"))?;
        prefill_tokens = prefill_tokens
            .checked_add(append)
            .ok_or_else(|| "prefill token sum overflow".to_string())?;
        lengths.push(append);
        has_state.push(prefix > 0);
    }
    if group.prefill_tokens != prefill_tokens {
        return Err(format!(
            "prefill_tokens {} must equal append sum {prefill_tokens}",
            group.prefill_tokens
        ));
    }
    let decode_tokens = u32::try_from(group.decode_kv_lens.len())
        .map_err(|_| "decode request count exceeds u32".to_string())?;
    if group.decode_tokens != decode_tokens {
        return Err(format!(
            "decode_tokens {} must equal decode request count {decode_tokens}",
            group.decode_tokens
        ));
    }
    let mut expected_decode_kv_len = 0_u32;
    for (index, &length) in group.decode_kv_lens.iter().enumerate() {
        if length == 0 {
            return Err(format!("decode KV length {index} must be positive"));
        }
        expected_decode_kv_len = expected_decode_kv_len
            .checked_add(length)
            .ok_or_else(|| "decode KV length sum overflow".to_string())?;
    }
    let active_tokens = prefill_tokens
        .checked_add(decode_tokens)
        .ok_or_else(|| "active token sum overflow".to_string())?;
    if active_tokens == 0 {
        return Err("qwen36_local requires nonempty active work".into());
    }
    if group.batch_tokens != active_tokens {
        return Err(format!(
            "batch_tokens {} must equal prefill+decode {active_tokens}",
            group.batch_tokens
        ));
    }
    if group.total_kv_len != expected_decode_kv_len {
        return Err(format!(
            "total_kv_len {} must equal decode KV sum {expected_decode_kv_len}",
            group.total_kv_len
        ));
    }
    let request_count = u32::try_from(group.prefill_chunk_pairs.len() + group.decode_kv_lens.len())
        .map_err(|_| "request count exceeds u32".to_string())?;
    let global_expert_selections = active_tokens
        .checked_mul(TOP_K)
        .ok_or_else(|| "routed selection count overflow".to_string())?;
    Ok(NormalizedInput {
        active_tokens,
        request_count,
        global_expert_selections,
        gdn: Qwen36GdnLocalWorkletInput {
            prefill_sequence_lengths: lengths,
            prefill_has_initial_state: has_state,
            decode_batch_size: decode_tokens,
        },
        gated_gqa: Qwen36GatedGqaLocalWorkletInput {
            batch_tokens: active_tokens,
            prefill_chunk_pairs: group.prefill_chunk_pairs.clone(),
            decode_kv_lens: group.decode_kv_lens.clone(),
        },
    })
}

impl Qwen36LocalModel {
    fn layer_node(
        &self,
        attention: CostNode,
        builder: &mut CostTreeBuilder,
        label: &str,
    ) -> CostNode {
        CostNode::Labeled {
            label: label.into(),
            child: Box::new(CostNode::Sum(vec![
                attention,
                self.router.compile(builder),
                CostNode::Labeled {
                    label: "local routed/shared expert resource-conserving section".into(),
                    child: Box::new(CostNode::Sum(vec![
                        self.routed_expert.compile(builder),
                        self.shared_expert.compile(builder),
                    ])),
                },
                self.finalize.compile(builder),
            ])),
        }
    }

    #[must_use]
    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let embedding = CostNode::Labeled {
            label: "embedding".into(),
            child: Box::new(self.embedding.compile(&mut builder)),
        };
        let gdn_layer = self.layer_node(self.gdn.compile(&mut builder), &mut builder, "GDN layer");
        let q = self.num_layers / 4;
        let r = self.num_layers % 4;
        let decoder = if q == 0 {
            CostNode::Labeled {
                label: format!("decoder prefix: {r} GDN layer(s)"),
                child: Box::new(CostNode::Scale {
                    n: r,
                    child: Box::new(gdn_layer),
                }),
            }
        } else {
            let gqa_layer = self.layer_node(
                self.gated_gqa.compile(&mut builder),
                &mut builder,
                "gated-GQA layer",
            );
            let cycle = CostNode::Sum(vec![
                CostNode::Scale {
                    n: 3,
                    child: Box::new(gdn_layer.clone()),
                },
                gqa_layer,
            ]);
            let mut prefix = vec![CostNode::Labeled {
                label: format!("decoder cycles: {q} x [3 GDN + 1 gated-GQA]"),
                child: Box::new(CostNode::Scale {
                    n: q,
                    child: Box::new(cycle),
                }),
            }];
            if r > 0 {
                // Reuse the existing leaf IDs: the remainder is another use of
                // the same physical GDN layer shape, not a second compiled slot set.
                prefix.push(CostNode::Labeled {
                    label: format!("decoder remainder: {r} GDN layer(s)"),
                    child: Box::new(CostNode::Scale {
                        n: r,
                        child: Box::new(gdn_layer),
                    }),
                });
            }
            CostNode::Sum(prefix)
        };
        let head = CostNode::Labeled {
            label: "head".into(),
            child: Box::new(self.head.compile(&mut builder)),
        };
        builder.finish(CostNode::Labeled {
            label: format!(
                "{} (Qwen36LocalModel) [TP1/EP1; {}-layer prefix]",
                self.name, self.num_layers
            ),
            child: Box::new(CostNode::Sum(vec![embedding, decoder, head])),
        })
    }

    fn eval_layer(&self, is_gdn: bool, input: &NormalizedInput, ev: &mut Evaluator) {
        if is_gdn {
            self.gdn.eval(&input.gdn, ev);
        } else {
            self.gated_gqa.eval(&input.gated_gqa, ev);
        }
        self.router.eval(
            &Qwen36MoeRouterLocalWorkletInput {
                batch_tokens: input.active_tokens,
            },
            ev,
        );
        self.routed_expert.eval(
            &VllmFp8MoeExpertComputeLocalWorkletInput {
                global_expert_selections: input.global_expert_selections,
            },
            ev,
        );
        self.shared_expert.eval(
            &Qwen36SharedExpertLocalWorkletInput {
                batch_tokens: input.active_tokens,
            },
            ev,
        );
        self.finalize.eval(
            &Qwen36MoeFinalizeLocalWorkletInput {
                batch_tokens: input.active_tokens,
            },
            ev,
        );
    }

    fn eval_into(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        let input = normalize_input(batch)
            .unwrap_or_else(|reason| panic!("invalid Qwen36LocalModel input: {reason}"));
        self.embedding.eval(
            &ElementwiseKernelInput {
                num_tokens: input.active_tokens,
            },
            ev,
        );
        self.eval_layer(true, &input, ev);
        if self.num_layers >= 4 {
            self.eval_layer(false, &input, ev);
        }
        self.head.eval(
            &Qwen36HeadLocalWorkletInput {
                final_norm_tokens: input.active_tokens,
                logits_tokens: input.request_count,
            },
            ev,
        );
    }
}

impl IterwiseUnifiedModel for Qwen36LocalModel {
    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut ev = Evaluator::new(slots);
        self.eval_into(batch, &mut ev);
        debug_assert_eq!(
            ev.filled(),
            self.n_slots,
            "eval cursor must fill every slot"
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
        let mut ev = Evaluator::with_inputs(slots, inputs);
        self.eval_into(batch, &mut ev);
        debug_assert_eq!(
            ev.filled(),
            self.n_slots,
            "eval cursor must fill every slot"
        );
        let total = CostTree::aggregate(&self.cost_flat, slots, scratch);
        debug_assert_eq!(
            inputs.len(),
            self.n_slots,
            "slot inputs must align with slots"
        );
        total
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }
    fn total_kv_bytes_per_token(&self) -> u64 {
        u64::from(self.total_kv_bytes_per_token.get())
    }
    fn recurrent_state_bytes_per_request(&self) -> u64 {
        u64::from(self.recurrent_state_bytes_per_request.get())
    }
    fn recurrent_checkpoint_interval_tokens(&self) -> u32 {
        self.recurrent_checkpoint_interval_tokens.get()
    }
    fn gpus_per_replica(&self) -> u16 {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::ArchGroupInput;
    use crate::timing::FlatCostNode;

    fn fixture() -> String {
        let layers: Vec<&str> = (0..40)
            .map(|i| {
                if i % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        let mut excluded = vec!["lm_head".to_string(), "model.embed_tokens".to_string()];
        for i in 0..40 {
            excluded.push(format!(
                "model.language_model.layers.{i}.mlp.shared_expert_gate"
            ));
            if i % 4 != 3 {
                excluded.push(format!(
                    "model.language_model.layers.{i}.linear_attn.in_proj_ba"
                ));
            }
        }
        serde_json::json!({
            "architectures":["Qwen3_5MoeForConditionalGeneration"], "model_type":"qwen3_5_moe", "tie_word_embeddings":false,
            "vision_config":{},
            "text_config": {"model_type":"qwen3_5_moe_text","dtype":"bfloat16","vocab_size":248320,"hidden_size":2048,
                "num_hidden_layers":40,"num_attention_heads":16,"num_key_value_heads":2,"head_dim":256,"partial_rotary_factor":0.25,
                "attn_output_gate":true,"full_attention_interval":4,"layer_types":layers,"linear_conv_kernel_dim":4,
                "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_num_key_heads":16,"linear_num_value_heads":32,
                "mamba_ssm_dtype":"float32","moe_intermediate_size":512,"shared_expert_intermediate_size":512,
                "num_experts_per_tok":8,"num_experts":256,"tie_word_embeddings":false,"mtp_num_hidden_layers":1},
            "quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8","modules_to_not_convert":excluded,"weight_block_size":[128,128]}
        }).to_string()
    }

    fn model(layers: u32) -> Qwen36ModelCfg {
        parse_model_json(&fixture(), Some(layers), None).unwrap()
    }
    fn cfgs(layers: u32) -> Qwen36LocalConfigs {
        cfgs_routed(layers, &RoutingDistribution::uniform(NUM_EXPERTS))
    }
    fn cfgs_routed(layers: u32, routing: &RoutingDistribution) -> Qwen36LocalConfigs {
        build_configs(
            &model(layers),
            &Qwen36LocalParallel {
                gpu_name: "NVIDIA H200".into(),
            },
            routing,
        )
    }
    fn enumerate_model(layers: u32) -> Qwen36LocalModel {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        build("model".into(), resolve_configs(&cfgs(layers)), &bridge).unwrap()
    }

    #[test]
    fn parses_pinned_nested_identity_and_layer_controls() {
        let m = parse_model_json(&fixture(), None, None).unwrap();
        assert_eq!(
            (
                m.hidden.get(),
                m.vocab_size.get(),
                m.num_layers,
                m.num_gdn_layers,
                m.num_gqa_layers
            ),
            (2048, 248320, 40, 30, 10)
        );
        for (logical, sim, expected) in [
            (Some(1), None, (1, 1, 0)),
            (Some(3), None, (3, 3, 0)),
            (Some(4), None, (4, 3, 1)),
            (Some(5), None, (5, 4, 1)),
            (Some(40), Some(39), (39, 30, 9)),
        ] {
            let m = parse_model_json(&fixture(), logical, sim).unwrap();
            assert_eq!((m.num_layers, m.num_gdn_layers, m.num_gqa_layers), expected);
        }
        assert!(parse_model_json(&fixture(), Some(0), None).is_err());
        assert!(parse_model_json(&fixture(), Some(39), Some(40)).is_err());
    }

    #[test]
    fn rejects_identity_schedule_and_quantization_drift() {
        for mutate in [
            |v: &mut serde_json::Value| v["text_config"]["hidden_size"] = 4096.into(),
            |v: &mut serde_json::Value| {
                v["text_config"]["layer_types"][3] = "linear_attention".into();
            },
            |v: &mut serde_json::Value| {
                v["quantization_config"]["weight_block_size"] = serde_json::json!([64, 128]);
            },
            |v: &mut serde_json::Value| v["tie_word_embeddings"] = true.into(),
        ] {
            let mut value: serde_json::Value = serde_json::from_str(&fixture()).unwrap();
            mutate(&mut value);
            assert!(parse_model_json(&value.to_string(), None, None).is_err());
        }
    }

    /// EP1 removes dispatch/combine traffic, not the grouped GEMM's dependence
    /// on routing: a measured skew must reach `local_ppm` whole, or the cost is
    /// of an idealized balanced model rather than the one vLLM ran.
    #[test]
    fn a_routing_skew_reaches_the_grouped_gemm_whole_at_ep1() {
        let balanced = cfgs(40);
        let skewed = cfgs_routed(40, &RoutingDistribution::power_law(NUM_EXPERTS, 1.0));

        // No `split_for_ep` shard: the one rank owns every expert.
        assert_eq!(skewed.routed_expert.local_ppm.len(), NUM_EXPERTS as usize);
        assert_ne!(
            skewed.routed_expert.local_ppm, balanced.routed_expert.local_ppm,
            "skew must survive into the grouped GEMM config"
        );
        // Routed compute and the combine that follows it must see one routing,
        // not two.
        assert_eq!(skewed.routed_expert.local_ppm, skewed.finalize.local_ppm);
        // Skew redistributes selections; it does not create or destroy them.
        assert_eq!(
            skewed
                .routed_expert
                .local_ppm
                .iter()
                .map(|&v| u64::from(v))
                .sum::<u64>(),
            1_000_000,
        );
        assert!(
            skewed.routed_expert.local_ppm[0] > skewed.routed_expert.local_ppm[255],
            "power-law routing must leave the head heavier than the tail"
        );
    }

    #[test]
    fn configs_freeze_backends_shapes_ppm_and_embedding() {
        let c = cfgs(40);
        assert_eq!(
            (
                c.embedding.input_bytes_per_token.get(),
                c.embedding.output_bytes_per_token.get()
            ),
            (4104, 4096)
        );
        assert_eq!(c.gdn.gpu_name, "NVIDIA H200");
        assert_eq!(c.gdn.chunk_delta_rule_backends, ["flashinfer"]);
        assert_eq!(c.gated_gqa.attention_backends, ["fa2", "fa3"]);
        assert_eq!(c.gated_gqa.qk_rms_norm_backends, ["flashinfer"]);
        assert_eq!(c.router.bf16_gemm_backends, ["torch_linear"]);
        assert_eq!(c.router.fused_topk_backends, ["vllm_cuda"]);
        // The routed experts run vLLM's Triton fused_moe launch, not the
        // TRT-LLM grouped GEMM: at EP1 there is no dispatch/combine to permute
        // tokens into per-expert order first.
        assert_eq!(c.routed_expert.fp8_grouped_gemm_backends, ["vllm_triton"]);
        assert_eq!(c.routed_expert.fp8_quant_backends, ["vllm_cuda"]);
        assert_eq!(c.routed_expert.local_ppm, c.finalize.local_ppm);
        assert_eq!(c.finalize.local_ppm.len(), 256);
        assert!(c.finalize.local_ppm[..64].iter().all(|&v| v == 3907));
        assert!(c.finalize.local_ppm[64..].iter().all(|&v| v == 3906));
        assert_eq!(
            c.finalize
                .local_ppm
                .iter()
                .map(|&v| u64::from(v))
                .sum::<u64>(),
            1_000_000
        );
        let r = resolve_configs(&c);
        assert_eq!(
            (r.gdn.qkvz.gemm.k.get(), r.gdn.qkvz.gemm.n.get()),
            (2048, 12288)
        );
        assert_eq!(
            (
                r.gated_gqa.qkv_gate.gemm.k.get(),
                r.gated_gqa.qkv_gate.gemm.n.get()
            ),
            (2048, 9216)
        );
        assert_eq!(
            (
                r.router.router.k.get(),
                r.router.router.n.get(),
                r.router.router.dtype
            ),
            (2048, 256, DType::Bf16)
        );
        assert_eq!(r.router.router.backends, ["torch_linear"]);
        assert_eq!(r.router.router.gpu_name, "NVIDIA H200");
        assert_eq!(
            (
                r.head.lm_head.k.get(),
                r.head.lm_head.n.get(),
                r.head.lm_head.dtype
            ),
            (2048, 248320, DType::Bf16)
        );
        assert_eq!(total_kv_bytes_per_token(&r).get(), 20_480);
    }

    #[test]
    fn cost_tree_has_exact_slot_counts_scales_reuse_and_no_communication() {
        for (layers, slots) in [(1, 41), (3, 41), (4, 72), (5, 72), (39, 72), (40, 72)] {
            let m = enumerate_model(layers);
            let tree = m.cost_tree();
            assert_eq!(tree.n_slots(), slots, "L={layers}");
            let kinds: Vec<&str> = tree.slots.iter().map(|s| s.kind.as_str()).collect();
            assert!(!kinds.iter().any(|k| matches!(
                *k,
                "all_reduce" | "all_to_all" | "p2p_intra" | "p2p_inter" | "send_recv"
            )));
            assert!(!tree
                .flatten()
                .iter()
                .any(|n| matches!(n, FlatCostNode::Max { .. })));
            if layers == 5 {
                let gdn_slots = tree
                    .slots
                    .iter()
                    .filter(|s| s.name.starts_with("model.gdn."))
                    .count();
                assert_eq!(gdn_slots, 20);
                assert!(tree
                    .flatten()
                    .iter()
                    .any(|n| matches!(n,FlatCostNode::Scale{n,..} if *n==1)));
                let flat = tree.flatten();
                let leaf_ids: Vec<usize> = flat
                    .iter()
                    .filter_map(|node| match node {
                        FlatCostNode::Leaf(slot) => Some(*slot),
                        _ => None,
                    })
                    .collect();
                assert!(
                    leaf_ids.len() > tree.n_slots(),
                    "remainder must reference existing leaves again"
                );
                let mut unique = leaf_ids.clone();
                unique.sort_unstable();
                unique.dedup();
                assert_eq!(
                    unique.len(),
                    tree.n_slots(),
                    "remainder must not mint a second GDN slot set"
                );
            }
        }
        let tree = enumerate_model(40).cost_tree();
        const ROUTER_LEAVES: usize = 3;
        const GDN_LAYER_LEAVES: usize = 20 + ROUTER_LEAVES + 5 + 7 + 2;
        const GQA_LAYER_LEAVES: usize = 13 + ROUTER_LEAVES + 5 + 7 + 2;
        const CYCLE_LEAVES: usize = GDN_LAYER_LEAVES + GQA_LAYER_LEAVES;
        const MODEL_LEAVES: usize = 1 + CYCLE_LEAVES + 2;
        assert_eq!(
            (
                GDN_LAYER_LEAVES,
                GQA_LAYER_LEAVES,
                CYCLE_LEAVES,
                MODEL_LEAVES
            ),
            (37, 30, 67, 70)
        );
        assert_eq!(
            tree.slots
                .iter()
                .filter(|s| s.name.starts_with("model.gdn."))
                .count(),
            20
        );
        assert_eq!(
            tree.slots
                .iter()
                .filter(|s| s.name.starts_with("model.gated_gqa."))
                .count(),
            13
        );
        assert_eq!(
            tree.slots
                .iter()
                .filter(|s| s.name.starts_with("model.router."))
                .count(),
            6
        );
        assert_eq!(
            tree.slots
                .iter()
                .filter(|s| s.name.starts_with("model.routed_expert."))
                .count(),
            10
        );
        assert_eq!(
            tree.slots
                .iter()
                .filter(|s| s.name.starts_with("model.shared_expert."))
                .count(),
            16
        );
        assert_eq!(
            tree.slots
                .iter()
                .filter(|s| s.name.starts_with("model.finalize."))
                .count(),
            4
        );
        assert!(!tree
            .slots
            .iter()
            .any(|s| s.name == "model.router.router.input_quant"));
        assert_eq!(
            tree.slots
                .iter()
                .filter(|s| s.name == "model.router.router.gemm")
                .count(),
            2
        );
        let first_routed = tree
            .slots
            .iter()
            .position(|s| s.name.starts_with("model.routed_expert."))
            .unwrap();
        let first_shared = tree
            .slots
            .iter()
            .position(|s| s.name.starts_with("model.shared_expert."))
            .unwrap();
        assert!(
            first_routed < first_shared,
            "routed expert must compile before shared expert"
        );
        let manifest = tree.manifest();
        let section_nodes = manifest
            .node_labels
            .iter()
            .enumerate()
            .filter_map(|(index, label)| {
                (label.as_deref() == Some("local routed/shared expert resource-conserving section"))
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            section_nodes.len(),
            2,
            "one section per compiled attention archetype"
        );
        assert!(section_nodes
            .iter()
            .all(|&index| matches!(manifest.nodes[index], FlatCostNode::Sum { .. })));
        let unique_names = tree
            .slots
            .iter()
            .map(|slot| slot.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(unique_names.len(), 54);
    }

    fn unified(prefill: Vec<(u32, u32)>, decode: Vec<u32>) -> UnifiedArchInput {
        let prefill_tokens = prefill.iter().map(|x| x.1).sum();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "decode is a test-fixture Vec built from a handful of literal kv lengths, its length is nowhere near u32::MAX"
        )]
        let decode_tokens = decode.len() as u32;
        // Mirrors UnifiedIterExecution and PredictGroup lowering: total_kv_len
        // is the decode-member aggregate; prefill state stays in the pairs.
        let total_kv_len = decode.iter().sum::<u32>();
        UnifiedArchInput {
            groups: vec![ArchGroupInput {
                batch_tokens: prefill_tokens + decode_tokens,
                prefill_tokens,
                decode_tokens,
                prefill_chunk_pairs: prefill,
                decode_kv_lens: decode,
                total_kv_len,
            }],
            tokens_per_source_rank: vec![],
        }
    }

    #[test]
    fn normalization_maps_prefill_decode_mixed_and_chunked_inputs() {
        let fresh_input = unified(vec![(0, 128)], vec![]);
        assert_eq!(fresh_input.groups[0].total_kv_len, 0);
        let fresh = normalize_input(&fresh_input).unwrap();
        assert_eq!(fresh.gdn.prefill_has_initial_state, [false]);
        let chunked_input = unified(vec![(64, 64)], vec![]);
        assert_eq!(chunked_input.groups[0].total_kv_len, 0);
        let chunked = normalize_input(&chunked_input).unwrap();
        assert_eq!(chunked.gdn.prefill_has_initial_state, [true]);
        let p = normalize_input(&unified(vec![(0, 3), (0, 65), (0, 2)], vec![])).unwrap();
        assert_eq!(
            (p.active_tokens, p.request_count, p.global_expert_selections),
            (70, 3, 560)
        );
        assert_eq!(p.gdn.prefill_sequence_lengths, [3, 65, 2]);
        assert_eq!(p.gdn.prefill_has_initial_state, [false, false, false]);
        let decode_input = unified(vec![], vec![9, 10, 11, 12]);
        assert_eq!(decode_input.groups[0].total_kv_len, 42);
        let d = normalize_input(&decode_input).unwrap();
        assert_eq!(
            (d.active_tokens, d.request_count, d.gdn.decode_batch_size),
            (4, 4, 4)
        );
        let mixed_input = unified(vec![(7, 3), (0, 65)], vec![20, 21]);
        assert_eq!(mixed_input.groups[0].total_kv_len, 41);
        let m = normalize_input(&mixed_input).unwrap();
        assert_eq!(
            (m.active_tokens, m.request_count, m.gdn.decode_batch_size),
            (70, 4, 2)
        );
        assert_eq!(m.gdn.prefill_has_initial_state, [true, false]);
        assert_eq!(m.gated_gqa.prefill_chunk_pairs, [(7, 3), (0, 65)]);
        assert_eq!(m.gated_gqa.decode_kv_lens, [20, 21]);
    }

    #[test]
    fn normalization_rejects_bad_group_accounting_and_overflow() {
        assert!(normalize_input(&UnifiedArchInput::default()).is_err());
        let mut x = unified(vec![(0, 1)], vec![]);
        x.tokens_per_source_rank = vec![1];
        assert!(normalize_input(&x).is_err());
        let mut x = unified(vec![(0, 1)], vec![]);
        x.groups[0].batch_tokens = 2;
        assert!(normalize_input(&x).is_err());
        let mut x = unified(vec![], vec![7, 9]);
        x.groups[0].total_kv_len = 15;
        assert!(normalize_input(&x)
            .unwrap_err()
            .contains("decode KV sum 16"));
        assert!(normalize_input(&unified(vec![(0, 0)], vec![])).is_err());
        let x = UnifiedArchInput {
            groups: vec![ArchGroupInput {
                batch_tokens: 1,
                prefill_tokens: 1,
                decode_tokens: 0,
                prefill_chunk_pairs: vec![(u32::MAX, 1)],
                decode_kv_lens: vec![],
                total_kv_len: 0,
            }],
            tokens_per_source_rank: vec![],
        };
        assert!(normalize_input(&x)
            .unwrap_err()
            .contains("prefix+append overflow"));
        let x = UnifiedArchInput {
            groups: vec![ArchGroupInput {
                batch_tokens: 2,
                prefill_tokens: 0,
                decode_tokens: 2,
                prefill_chunk_pairs: vec![],
                decode_kv_lens: vec![u32::MAX, 1],
                total_kv_len: 0,
            }],
            tokens_per_source_rank: vec![],
        };
        assert!(normalize_input(&x)
            .unwrap_err()
            .contains("decode KV length sum overflow"));
        let x = UnifiedArchInput {
            groups: vec![ArchGroupInput {
                batch_tokens: u32::MAX,
                prefill_tokens: u32::MAX,
                decode_tokens: 1,
                prefill_chunk_pairs: vec![(0, u32::MAX)],
                decode_kv_lens: vec![1],
                total_kv_len: 0,
            }],
            tokens_per_source_rank: vec![],
        };
        assert!(normalize_input(&x).is_err());
    }

    #[test]
    fn kv_bytes_count_only_full_attention_layers() {
        for (layers, expected) in [
            (1, 0),
            (3, 0),
            (4, 2048),
            (5, 2048),
            (39, 18_432),
            (40, 20_480),
        ] {
            assert_eq!(
                total_kv_bytes_per_token(&resolve_configs(&cfgs(layers))).get(),
                expected
            );
        }
    }

    #[test]
    fn recurrent_state_bytes_charge_the_padded_page_on_every_gdn_layer() {
        const PADDED_PER_LAYER: u32 = 1056 * 2048;
        assert_eq!(PADDED_PER_LAYER, 2_162_688);
        for (layers, gdn_layers) in [(1, 1), (3, 3), (4, 3), (5, 4), (39, 30), (40, 30)] {
            let resolved = resolve_configs(&cfgs(layers));
            assert_eq!(resolved.num_gdn_layers, gdn_layers, "L={layers}");
            assert_eq!(
                recurrent_state_bytes_per_request(&resolved).get(),
                PADDED_PER_LAYER * gdn_layers,
                "L={layers}",
            );
        }
        assert_eq!(
            recurrent_state_bytes_per_request(&resolve_configs(&cfgs(40))).get(),
            64_880_640
        );
    }

    /// Reconstructs both numbers the engine logs at startup for this checkpoint
    /// at TP1: "Setting attention block size to 1056 tokens" and "Padding mamba
    /// page size by 0.76%". Only the whole-mamba-page over K-and-V reading of the
    /// alignment rule hits both; SSM-only/K-only gives 2048, and either
    /// correction alone gives 2112 or 1024.
    #[test]
    fn checkpoint_interval_reproduces_the_measured_vllm_block_size() {
        let resolved = resolve_configs(&cfgs(40));
        assert_eq!(
            (
                attention_page_bytes_per_token_per_layer(&resolved).get(),
                ssm_state_bytes_per_layer(&resolved).get(),
                conv_state_bytes_per_layer(&resolved).get(),
                mamba_page_bytes_per_layer(&resolved).get()
            ),
            (2_048, 2_097_152, 49_152, 2_146_304),
            "per-token attention page and per-layer mamba page feeding the rule",
        );
        assert_eq!(recurrent_checkpoint_interval_tokens(&resolved).get(), 1_056);

        let padded = recurrent_checkpoint_interval_tokens(&resolved).get()
            * attention_page_bytes_per_token_per_layer(&resolved).get();
        let padding_percent = 100.0
            * (f64::from(padded) / f64::from(mamba_page_bytes_per_layer(&resolved).get()) - 1.0);
        assert!(
            (padding_percent - 0.76).abs() < 0.005,
            "padding {padding_percent}% != logged 0.76%"
        );

        // The interval is a property of the layer shapes, not of how many
        // layers the prefix keeps — only "no GDN layer at all" turns it off.
        for (layers, expected) in [(1, 1056), (3, 1056), (4, 1056), (39, 1056), (40, 1056)] {
            assert_eq!(
                recurrent_checkpoint_interval_tokens(&resolve_configs(&cfgs(layers))).get(),
                expected,
                "L={layers}"
            );
        }
    }

    #[test]
    fn built_model_exposes_the_recurrent_state_contract() {
        let model = enumerate_model(40);
        assert_eq!(model.recurrent_state_bytes_per_request(), 64_880_640);
        assert_eq!(model.recurrent_checkpoint_interval_tokens(), 1_056);
        // One checkpoint costs 3168 tokens of full-attention KV: the alignment
        // rule equalizes PER-LAYER pages, and this model has 30 GDN : 10 GQA
        // layers, so a checkpoint is 3x the KV of the span it covers.
        let state_tokens = model
            .recurrent_state_bytes_per_request()
            .div_ceil(model.total_kv_bytes_per_token());
        assert_eq!(state_tokens, 3_168);
        assert_eq!(
            state_tokens,
            3 * u64::from(model.recurrent_checkpoint_interval_tokens())
        );
    }

    #[test]
    fn full_manifest_has_one_entry_per_unique_slot() {
        let model = enumerate_model(40);
        let manifest = model.cost_log_manifest();
        assert_eq!(manifest.slots.len(), 72);
        assert_eq!(model.cost_tree().n_slots(), 72);
    }
}
