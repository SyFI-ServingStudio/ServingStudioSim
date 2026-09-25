//! Kimi-K3's B200 SGLang decode-first architecture recipe.
//!
//! The model config owns the heterogeneous layer schedule. This file only
//! consumes the explicit KDA/MLA lists; it does not infer a cadence from layer
//! numbers. The production graph is attention-TP8, EP8, and PP2. A rank1
//! alignment graph uses the same rank-local shapes with explicit
//! `heads_per_rank=12` and `local_experts=112` overrides.

use std::path::Path;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::json;

use crate::arch::config::ModelSpec;
use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::op::Op;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, RmsNormKernel,
    RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, DType, Dim, Evaluator,
    FlatCostNode, LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::{
    KimiK3DenseLocalWorklet, KimiK3DenseLocalWorkletConfig, KimiK3DenseLocalWorkletInput,
    KimiK3DenseLocalWorkletResolved, KimiK3KdaLocalWorklet, KimiK3KdaLocalWorkletConfig,
    KimiK3KdaLocalWorkletInput, KimiK3KdaLocalWorkletResolved, KimiK3MlaLocalWorklet,
    KimiK3MlaLocalWorkletConfig, KimiK3MlaLocalWorkletInput, KimiK3MlaLocalWorkletResolved,
    KimiK3MoeLocalWorklet, KimiK3MoeLocalWorkletConfig, KimiK3MoeLocalWorkletInput,
    KimiK3MoeLocalWorkletResolved,
};

const ARCH_KIND: &str = "kimi_k3_sglang";
const HIDDEN: u32 = 7_168;
const VOCAB_SIZE: u32 = 163_840;
const NUM_LAYERS: u32 = 93;
const NUM_HEADS: u32 = 96;
const HEAD_DIM: u32 = 128;
const Q_LORA_RANK: u32 = 1_536;
const KV_LORA_RANK: u32 = 512;
const QK_NOPE: u32 = 128;
const QK_ROPE: u32 = 64;
const V_HEAD_DIM: u32 = 128;
const NUM_EXPERTS: u32 = 896;
const TOP_K: u32 = 16;
const MOE_INTERMEDIATE: u32 = 3_072;
const LATENT_HIDDEN: u32 = 3_584;
const SHARED_INTERMEDIATE: u32 = 6_144;
const DENSE_INTERMEDIATE: u32 = 33_792;
const CONV_KERNEL: u32 = 4;
const ATTN_RES_BLOCK_SIZE: u32 = 12;
const MAX_MODEL_LEN: u32 = 1_048_576;

const RESIDUAL_NORM_BACKENDS: &[&str] = &["vllm_cuda"];
const RMS_NORM_BACKENDS: &[&str] = &["flashinfer"];
const GEMM_BACKENDS: &[&str] = &["sglang_bf16_auto"];
const K3_PREFILL_GEMM_BACKENDS: &[&str] = &["sglang_k3_raw_bf16"];
const K3_PREFILL_FP32_GEMM_BACKENDS: &[&str] = &["sglang_k3_fp32_auto"];
const K3_PREFILL_BF16_GEMM_BACKENDS: &[&str] = &["sglang_k3_raw_bf16"];
const K3_PREFILL_ACTIVATION_BACKENDS: &[&str] = &["sglang_k3"];
const K3_PREFILL_ADD3_BACKENDS: &[&str] = &["sglang_k3"];
const K3_PREFILL_ATTN_RES_BACKENDS: &[&str] = &["sglang_k3"];
const FUSED_QKV_A_BACKENDS: &[&str] = &["sglang_fused_a_auto"];
const ABSORB_BACKENDS: &[&str] = &["sglang_k3_absorb"];
const CACHE_APPEND_BACKENDS: &[&str] = &["sglang_cuda"];
const MLA_ATTENTION_BACKENDS: &[&str] =
    // cute-dsl is what the cookbook B200 recipes resolve for decode (trtllm-gen is
    // the non-DCP default); the Triton split-KV path is a fallback sglang never
    // picks on Blackwell and its profiler is not yet stable -> not costed.
    &["sglang_cutedsl_mla", "sglang_trtllm_mla"];
const MLA_PREFILL_ATTENTION_BACKENDS: &[&str] = &["sglang_trtllm_mla"];
const KDA_FUSED_BACKENDS: &[&str] = &["sglang_fused"];
const KDA_TRITON_BACKENDS: &[&str] = &["sglang_triton"];
const MOE_BACKENDS: &[&str] = &["sglang_trtllm_mxfp4"];
const MOE_PREFILL_BACKENDS: &[&str] = &["sglang_trtllm_mxfp4_prefill"];
const ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
const LM_HEAD_BACKENDS: &[&str] = &["torch_linear"];

#[derive(Clone, Debug)]
pub struct KimiK3ModelCfg {
    pub hidden: Dim,
    pub intermediate: Dim,
    pub vocab_size: Dim,
    pub num_layers: u32,
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub q_lora_rank: Dim,
    pub kv_lora_rank: Dim,
    pub qk_nope: Dim,
    pub qk_rope: Dim,
    pub v_head_dim: Dim,
    pub num_experts: Dim,
    pub top_k: u32,
    pub moe_intermediate: Dim,
    pub latent_hidden: Dim,
    pub shared_intermediate: Dim,
    pub dense_intermediate: Dim,
    pub conv_kernel: Dim,
    pub gate_lower_bound: i32,
    pub full_attn_layers: Vec<u32>,
    pub kda_layers: Vec<u32>,
    pub max_model_len: u32,
    pub attn_res_block_size: u32,
}

#[derive(Deserialize)]
struct RawKimiK3Config {
    architectures: Vec<String>,
    model_type: String,
    dtype: String,
    text_config: RawKimiK3TextConfig,
}

#[derive(Deserialize)]
struct RawKimiK3TextConfig {
    architectures: Vec<String>,
    model_type: String,
    dtype: String,
    hidden_size: u32,
    intermediate_size: u32,
    vocab_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    max_position_embeddings: u32,
    num_experts: u32,
    num_experts_per_token: u32,
    num_shared_experts: u32,
    moe_intermediate_size: u32,
    q_lora_rank: u32,
    kv_lora_rank: u32,
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
    v_head_dim: u32,
    routed_expert_hidden_size: u32,
    first_k_dense_replace: u32,
    attn_res_block_size: u32,
    linear_attn_config: RawLinearAttentionConfig,
}

#[derive(Deserialize)]
struct RawLinearAttentionConfig {
    full_attn_layers: Vec<u32>,
    kda_layers: Vec<u32>,
    gate_lower_bound: f64,
    head_dim: u32,
    num_heads: u32,
    short_conv_kernel_size: u32,
    use_full_rank_gate: bool,
}

impl KimiK3ModelCfg {
    pub fn from_json(path: &Path, spec: &ModelSpec) -> Result<Self> {
        ensure!(
            !spec.fp8,
            "{ARCH_KIND} uses BF16 activations; fp8 must be false"
        );
        ensure!(
            spec.num_layers.is_none(),
            "{ARCH_KIND} does not accept num_layers"
        );
        ensure!(
            spec.sim_num_layers.is_none(),
            "{ARCH_KIND} uses sim_kda_layers/sim_mla_layers instead of sim_num_layers"
        );
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading Kimi-K3 config {}", path.display()))?;
        let raw: RawKimiK3Config = serde_json::from_str(&text).context("parsing JSON")?;
        let t = raw.text_config;
        ensure!(
            raw.architectures == ["KimiK3ForConditionalGeneration"],
            "outer architectures must identify KimiK3ForConditionalGeneration"
        );
        ensure!(
            raw.model_type == "kimi_k3",
            "outer model_type must be kimi_k3"
        );
        ensure!(raw.dtype == "bfloat16", "outer dtype must be bfloat16");
        ensure!(
            t.architectures == ["KimiLinearForCausalLM"],
            "text architectures must identify KimiLinearForCausalLM"
        );
        ensure!(
            t.model_type == "kimi_linear",
            "text model_type must be kimi_linear"
        );
        ensure!(t.dtype == "bfloat16", "text dtype must be bfloat16");

        for (name, actual, expected) in [
            ("hidden_size", t.hidden_size, HIDDEN),
            ("intermediate_size", t.intermediate_size, DENSE_INTERMEDIATE),
            ("vocab_size", t.vocab_size, VOCAB_SIZE),
            ("num_hidden_layers", t.num_hidden_layers, NUM_LAYERS),
            ("num_attention_heads", t.num_attention_heads, NUM_HEADS),
            ("num_key_value_heads", t.num_key_value_heads, NUM_HEADS),
            (
                "max_position_embeddings",
                t.max_position_embeddings,
                MAX_MODEL_LEN,
            ),
            ("num_experts", t.num_experts, NUM_EXPERTS),
            ("num_experts_per_token", t.num_experts_per_token, TOP_K),
            ("num_shared_experts", t.num_shared_experts, 2),
            (
                "moe_intermediate_size",
                t.moe_intermediate_size,
                MOE_INTERMEDIATE,
            ),
            ("q_lora_rank", t.q_lora_rank, Q_LORA_RANK),
            ("kv_lora_rank", t.kv_lora_rank, KV_LORA_RANK),
            ("qk_nope_head_dim", t.qk_nope_head_dim, QK_NOPE),
            ("qk_rope_head_dim", t.qk_rope_head_dim, QK_ROPE),
            ("v_head_dim", t.v_head_dim, V_HEAD_DIM),
            (
                "routed_expert_hidden_size",
                t.routed_expert_hidden_size,
                LATENT_HIDDEN,
            ),
            ("first_k_dense_replace", t.first_k_dense_replace, 1),
            (
                "attn_res_block_size",
                t.attn_res_block_size,
                ATTN_RES_BLOCK_SIZE,
            ),
            (
                "linear_attn_config.head_dim",
                t.linear_attn_config.head_dim,
                HEAD_DIM,
            ),
            (
                "linear_attn_config.num_heads",
                t.linear_attn_config.num_heads,
                NUM_HEADS,
            ),
            (
                "linear_attn_config.short_conv_kernel_size",
                t.linear_attn_config.short_conv_kernel_size,
                CONV_KERNEL,
            ),
        ] {
            ensure!(
                actual == expected,
                "{name} must be {expected}, got {actual}"
            );
        }
        ensure!(
            t.linear_attn_config.use_full_rank_gate,
            "linear_attn_config.use_full_rank_gate must be true"
        );
        ensure!(
            (t.linear_attn_config.gate_lower_bound + 5.0).abs() < 1e-6,
            "linear_attn_config.gate_lower_bound must be -5"
        );
        validate_schedule(
            NUM_LAYERS,
            &t.linear_attn_config.full_attn_layers,
            &t.linear_attn_config.kda_layers,
        )?;

        Ok(Self {
            hidden: HIDDEN.into(),
            intermediate: DENSE_INTERMEDIATE.into(),
            vocab_size: VOCAB_SIZE.into(),
            num_layers: NUM_LAYERS,
            num_heads: NUM_HEADS.into(),
            head_dim: HEAD_DIM.into(),
            q_lora_rank: Q_LORA_RANK.into(),
            kv_lora_rank: KV_LORA_RANK.into(),
            qk_nope: QK_NOPE.into(),
            qk_rope: QK_ROPE.into(),
            v_head_dim: V_HEAD_DIM.into(),
            num_experts: NUM_EXPERTS.into(),
            top_k: TOP_K,
            moe_intermediate: MOE_INTERMEDIATE.into(),
            latent_hidden: LATENT_HIDDEN.into(),
            shared_intermediate: SHARED_INTERMEDIATE.into(),
            dense_intermediate: DENSE_INTERMEDIATE.into(),
            conv_kernel: CONV_KERNEL.into(),
            gate_lower_bound: -5,
            full_attn_layers: t.linear_attn_config.full_attn_layers,
            kda_layers: t.linear_attn_config.kda_layers,
            max_model_len: MAX_MODEL_LEN,
            attn_res_block_size: ATTN_RES_BLOCK_SIZE,
        })
    }
}

fn validate_schedule(num_layers: u32, mla: &[u32], kda: &[u32]) -> Result<()> {
    ensure!(mla.len() == 24, "full_attn_layers must contain 24 entries");
    ensure!(kda.len() == 69, "kda_layers must contain 69 entries");
    let mut seen = vec![0_u8; num_layers as usize + 1];
    for (kind, layers) in [("full_attn_layers", mla), ("kda_layers", kda)] {
        for &layer in layers {
            ensure!(
                (1..=num_layers).contains(&layer),
                "{kind} contains out-of-range layer {layer}"
            );
            let entry = &mut seen[layer as usize];
            ensure!(
                *entry == 0,
                "layer {layer} occurs more than once in the schedule"
            );
            *entry = 1;
        }
    }
    ensure!(
        seen[1..].iter().all(|entry| *entry == 1),
        "full_attn_layers and kda_layers must partition layers 1..={num_layers}"
    );
    ensure!(kda.contains(&1), "layer 1 must be the dense KDA layer");
    ensure!(!mla.contains(&1), "layer 1 must not be an MLA layer");
    Ok(())
}

#[derive(Clone, Debug)]
pub struct KimiK3SglangParallel {
    pub attn_tp_size: u16,
    pub ep_size: u16,
    pub pp_size: u16,
    pub dcp_size: u16,
    pub kda_state_dtype: DType,
    pub heads_per_rank: Option<u16>,
    pub local_experts: Option<u32>,
    pub local_top_k: Option<u32>,
    pub routing_histogram: Option<Vec<f32>>,
    pub sim_kda_layers: Option<u32>,
    pub sim_mla_layers: Option<u32>,
    pub gpu_name: String,
}

impl KimiK3SglangParallel {
    pub fn validate(&self, model: &KimiK3ModelCfg) -> Result<()> {
        ensure!(self.attn_tp_size > 0, "attn_tp_size must be positive");
        ensure!(self.ep_size > 0, "ep_size must be positive");
        ensure!(self.pp_size > 0, "pp_size must be positive");
        ensure!(self.dcp_size == 1, "Kimi-K3 v1 requires dcp_size=1");
        ensure!(
            matches!(self.kda_state_dtype, DType::Bf16 | DType::Fp32),
            "kda_state_dtype must be bf16 or fp32"
        );
        ensure!(
            self.ep_size % self.attn_tp_size == 0,
            "ep_size must be divisible by attn_tp_size"
        );
        ensure!(
            u32::from(self.attn_tp_size) <= NUM_HEADS,
            "attn_tp_size cannot exceed the number of attention heads"
        );
        if let Some(heads) = self.heads_per_rank {
            ensure!(heads > 0, "heads_per_rank must be positive");
        } else {
            ensure!(
                NUM_HEADS % u32::from(self.attn_tp_size) == 0,
                "NUM_HEADS must be divisible by attn_tp_size when heads_per_rank is absent"
            );
        }
        let local_experts = self
            .local_experts
            .unwrap_or(model.num_experts.get() / u32::from(self.ep_size));
        ensure!(local_experts > 0, "local_experts must be positive");
        ensure!(
            model.num_experts.get() % local_experts == 0,
            "local_experts must divide the global expert count"
        );
        if let Some(local_top_k) = self.local_top_k {
            ensure!(local_top_k > 0, "local_top_k must be positive");
            ensure!(
                local_top_k <= model.top_k,
                "local_top_k must not exceed global top_k"
            );
            if local_top_k != model.top_k {
                ensure!(
                    self.ep_size == 1 && local_experts < model.num_experts.get(),
                    "a reduced local_top_k is only valid for a rank-local probe"
                );
            }
        }
        if self.routing_histogram.is_some() {
            ensure!(
                self.ep_size == 1,
                "routing_histogram is only supported for a rank-local alignment probe"
            );
            ensure!(
                self.local_top_k.is_some(),
                "routing_histogram requires explicit local_top_k"
            );
        }
        ensure!(
            u32::from(self.ep_size) * u32::from(self.pp_size) <= u32::from(u16::MAX),
            "ep_size*pp_size does not fit the worker GPU count"
        );
        if let (Some(kda), Some(mla)) = (self.sim_kda_layers, self.sim_mla_layers) {
            ensure!(
                !(kda > 0 && mla > 0),
                "sim_kda_layers and sim_mla_layers cannot both be positive"
            );
        }
        ensure!(
            self.sim_kda_layers.unwrap_or(0) + self.sim_mla_layers.unwrap_or(0) > 0
                || (self.sim_kda_layers.is_none() && self.sim_mla_layers.is_none()),
            "layer simulation overrides must select at least one layer"
        );
        Ok(())
    }

    pub fn heads_per_rank_value(&self) -> u32 {
        u32::from(
            self.heads_per_rank
                .unwrap_or((NUM_HEADS / u32::from(self.attn_tp_size)) as u16),
        )
    }

    pub fn local_experts_value(&self, model: &KimiK3ModelCfg) -> u32 {
        self.local_experts
            .unwrap_or(model.num_experts.get() / u32::from(self.ep_size))
    }

    pub fn local_experts_value_from_constants(&self) -> u32 {
        self.local_experts
            .unwrap_or(NUM_EXPERTS / u32::from(self.ep_size))
    }

    pub fn routing_top_k_value(&self, model: &KimiK3ModelCfg) -> u32 {
        self.local_top_k.unwrap_or(model.top_k)
    }

    pub fn num_attn_dp_groups(&self) -> u16 {
        self.ep_size / self.attn_tp_size
    }

    pub fn num_attn_shards(&self) -> u16 {
        self.attn_tp_size
    }

    pub fn gpus_per_replica(&self) -> u16 {
        self.ep_size * self.pp_size
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KimiK3LayerCounts {
    pub dense: u32,
    pub kda: u32,
    pub mla: u32,
}

fn layer_counts(
    model: &KimiK3ModelCfg,
    parallel: &KimiK3SglangParallel,
) -> Result<KimiK3LayerCounts> {
    if parallel.sim_kda_layers.is_none() && parallel.sim_mla_layers.is_none() {
        return Ok(KimiK3LayerCounts {
            dense: 1,
            kda: model.kda_layers.len() as u32 - 1,
            mla: model.full_attn_layers.len() as u32,
        });
    }
    let kda = parallel.sim_kda_layers.unwrap_or(0);
    let mla = parallel.sim_mla_layers.unwrap_or(0);
    ensure!(kda + mla > 0, "simulation override selected no layers");
    Ok(KimiK3LayerCounts { dense: 0, kda, mla })
}

fn pipeline_stage_counts(
    model: &KimiK3ModelCfg,
    parallel: &KimiK3SglangParallel,
    counts: KimiK3LayerCounts,
) -> Vec<KimiK3LayerCounts> {
    if parallel.pp_size == 1
        || parallel.sim_kda_layers.is_some()
        || parallel.sim_mla_layers.is_some()
    {
        return vec![counts];
    }
    let pp = u32::from(parallel.pp_size);
    (0..pp)
        .map(|stage| {
            let first = stage * model.num_layers / pp + 1;
            let last = (stage + 1) * model.num_layers / pp;
            let in_stage = |layer: &u32| (*layer >= first) && (*layer <= last);
            KimiK3LayerCounts {
                dense: model
                    .kda_layers
                    .iter()
                    .any(|layer| *layer == 1 && in_stage(layer)) as u32,
                kda: model
                    .kda_layers
                    .iter()
                    .filter(|layer| **layer != 1 && in_stage(layer))
                    .count() as u32,
                mla: model
                    .full_attn_layers
                    .iter()
                    .filter(|layer| in_stage(layer))
                    .count() as u32,
            }
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct KimiK3SglangConfigs {
    pub parallel: KimiK3SglangParallel,
    pub counts: KimiK3LayerCounts,
    pub pp_stage_counts: Vec<KimiK3LayerCounts>,
    pub include_model_io: bool,
    pub embedding: Option<ElementwiseKernelConfig>,
    pub dense_attention: Option<KimiK3KdaLocalWorkletConfig>,
    pub dense_ffn: Option<KimiK3DenseLocalWorkletConfig>,
    pub kda_attention: Option<KimiK3KdaLocalWorkletConfig>,
    pub kda_moe: Option<KimiK3MoeLocalWorkletConfig>,
    pub mla_attention: Option<KimiK3MlaLocalWorkletConfig>,
    pub mla_moe: Option<KimiK3MoeLocalWorkletConfig>,
    pub final_norm: Option<RmsNormKernelConfig>,
    pub lm_head: Option<SingleGemmKernelConfig>,
}

#[derive(Clone, Debug)]
pub struct KimiK3SglangResolved {
    pub parallel: KimiK3SglangParallel,
    pub counts: KimiK3LayerCounts,
    pub pp_stage_counts: Vec<KimiK3LayerCounts>,
    pub include_model_io: bool,
    pub embedding: Option<ElementwiseKernelConfig>,
    pub dense_attention: Option<KimiK3KdaLocalWorkletResolved>,
    pub dense_ffn: Option<KimiK3DenseLocalWorkletResolved>,
    pub kda_attention: Option<KimiK3KdaLocalWorkletResolved>,
    pub kda_moe: Option<KimiK3MoeLocalWorkletResolved>,
    pub mla_attention: Option<KimiK3MlaLocalWorkletResolved>,
    pub mla_moe: Option<KimiK3MoeLocalWorkletResolved>,
    pub final_norm: Option<RmsNormKernelConfig>,
    pub lm_head: Option<SingleGemmKernelConfig>,
}

pub fn build_configs(
    model: &KimiK3ModelCfg,
    parallel: &KimiK3SglangParallel,
) -> Result<KimiK3SglangConfigs> {
    parallel.validate(model)?;
    let counts = layer_counts(model, parallel)?;
    let pp_stage_counts = pipeline_stage_counts(model, parallel, counts);
    let gpu = parallel.gpu_name.clone();
    let heads = parallel.heads_per_rank_value();
    let local_experts = parallel.local_experts_value(model);
    // The standalone rank-1 probe builds SGLang's MoE config with the
    // per-rank expert width (112) and keeps EP only as metadata. Production
    // EP8 keeps the 896-wide router and hands 112 experts to the local rank.
    let routing_experts = if parallel.ep_size == 1 {
        local_experts
    } else {
        model.num_experts.get()
    };
    let routing_top_k = parallel.routing_top_k_value(model);
    let include_model_io = parallel.sim_kda_layers.is_none() && parallel.sim_mla_layers.is_none();

    let kda_cfg = || KimiK3KdaLocalWorkletConfig {
        gpu_name: gpu.clone(),
        hidden: model.hidden.clone(),
        heads: heads.into(),
        head_dim: model.head_dim.clone(),
        conv_kernel: model.conv_kernel.clone(),
        lower_bound: model.gate_lower_bound,
        dtype: DType::Bf16,
        kda_state_dtype: parallel.kda_state_dtype,
        gemm_backends: GEMM_BACKENDS.to_vec(),
        prefill_gemm_backends: K3_PREFILL_GEMM_BACKENDS.to_vec(),
        fused_decode_backends: KDA_FUSED_BACKENDS.to_vec(),
        prefill_backends: KDA_TRITON_BACKENDS.to_vec(),
        causal_conv_decode_backends: KDA_TRITON_BACKENDS.to_vec(),
        recurrent_decode_backends: KDA_TRITON_BACKENDS.to_vec(),
        gated_norm_backends: KDA_TRITON_BACKENDS.to_vec(),
        residual_norm_backends: RESIDUAL_NORM_BACKENDS.to_vec(),
        prefill_attn_res_backends: K3_PREFILL_ATTN_RES_BACKENDS.to_vec(),
        tp_size: parallel.attn_tp_size,
    };
    let moe_cfg = || KimiK3MoeLocalWorkletConfig {
        gpu_name: gpu.clone(),
        hidden: model.hidden.clone(),
        latent_hidden: model.latent_hidden.clone(),
        num_experts: model.num_experts.clone(),
        routing_experts: routing_experts.into(),
        local_experts: local_experts.into(),
        moe_intermediate: model.moe_intermediate.clone(),
        shared_intermediate: model.shared_intermediate.clone(),
        top_k: routing_top_k,
        routing_histogram: parallel.routing_histogram.clone(),
        ep_size: parallel.ep_size,
        dtype: DType::Bf16,
        gemm_backends: GEMM_BACKENDS.to_vec(),
        prefill_gemm_backends: K3_PREFILL_FP32_GEMM_BACKENDS.to_vec(),
        prefill_bf16_gemm_backends: K3_PREFILL_BF16_GEMM_BACKENDS.to_vec(),
        prefill_activation_backends: K3_PREFILL_ACTIVATION_BACKENDS.to_vec(),
        prefill_add3_backends: K3_PREFILL_ADD3_BACKENDS.to_vec(),
        elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        rms_norm_backends: RMS_NORM_BACKENDS.to_vec(),
        moe_backends: MOE_BACKENDS.to_vec(),
        prefill_moe_backends: MOE_PREFILL_BACKENDS.to_vec(),
    };
    let mla_cfg = || KimiK3MlaLocalWorkletConfig {
        gpu_name: gpu.clone(),
        hidden: model.hidden.clone(),
        heads: heads.into(),
        q_lora_rank: model.q_lora_rank.clone(),
        kv_lora_rank: model.kv_lora_rank.clone(),
        qk_nope: model.qk_nope.clone(),
        rope_dim: model.qk_rope.clone(),
        v_head_dim: model.v_head_dim.clone(),
        page_size: 64.into(),
        dtype: DType::Bf16,
        cache_dtype: DType::Fp8E4m3,
        residual_norm_backends: RESIDUAL_NORM_BACKENDS.to_vec(),
        rms_norm_backends: RMS_NORM_BACKENDS.to_vec(),
        fused_qkv_a_backends: FUSED_QKV_A_BACKENDS.to_vec(),
        projection_backends: GEMM_BACKENDS.to_vec(),
        prefill_projection_backends: K3_PREFILL_GEMM_BACKENDS.to_vec(),
        absorb_backends: ABSORB_BACKENDS.to_vec(),
        cache_append_backends: CACHE_APPEND_BACKENDS.to_vec(),
        attention_backends: MLA_ATTENTION_BACKENDS.to_vec(),
        prefill_attention_backends: MLA_PREFILL_ATTENTION_BACKENDS.to_vec(),
        prefill_aux_backends: KDA_TRITON_BACKENDS.to_vec(),
        prefill_attn_res_backends: K3_PREFILL_ATTN_RES_BACKENDS.to_vec(),
        elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        tp_size: parallel.attn_tp_size,
    };

    Ok(KimiK3SglangConfigs {
        parallel: parallel.clone(),
        counts,
        pp_stage_counts,
        include_model_io,
        embedding: include_model_io.then(|| ElementwiseKernelConfig {
            backends: ELEMENTWISE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            input_bytes_per_token: 4.into(),
            output_bytes_per_token: (HIDDEN * 2).into(),
        }),
        dense_attention: (counts.dense > 0).then(kda_cfg),
        dense_ffn: (counts.dense > 0).then(|| KimiK3DenseLocalWorkletConfig {
            gpu_name: gpu.clone(),
            hidden: model.hidden.clone(),
            intermediate: model.dense_intermediate.clone(),
            dtype: DType::Bf16,
            gemm_backends: GEMM_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        }),
        kda_attention: (counts.kda > 0).then(kda_cfg),
        kda_moe: (counts.kda > 0).then(moe_cfg),
        mla_attention: (counts.mla > 0).then(mla_cfg),
        mla_moe: (counts.mla > 0).then(moe_cfg),
        final_norm: include_model_io.then(|| RmsNormKernelConfig {
            backends: RMS_NORM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden: model.hidden.clone(),
            dtype: DType::Bf16,
        }),
        lm_head: include_model_io.then(|| SingleGemmKernelConfig {
            backends: LM_HEAD_BACKENDS.to_vec(),
            gpu_name: gpu,
            n: model.vocab_size.clone(),
            k: model.hidden.clone(),
            dtype: DType::Bf16,
        }),
    })
}

pub fn resolve_configs(cfg: &KimiK3SglangConfigs) -> KimiK3SglangResolved {
    KimiK3SglangResolved {
        parallel: cfg.parallel.clone(),
        counts: cfg.counts,
        pp_stage_counts: cfg.pp_stage_counts.clone(),
        include_model_io: cfg.include_model_io,
        embedding: cfg.embedding.clone(),
        dense_attention: cfg
            .dense_attention
            .as_ref()
            .map(KimiK3KdaLocalWorklet::resolve_config),
        dense_ffn: cfg
            .dense_ffn
            .as_ref()
            .map(KimiK3DenseLocalWorklet::resolve_config),
        kda_attention: cfg
            .kda_attention
            .as_ref()
            .map(KimiK3KdaLocalWorklet::resolve_config),
        kda_moe: cfg
            .kda_moe
            .as_ref()
            .map(KimiK3MoeLocalWorklet::resolve_config),
        mla_attention: cfg
            .mla_attention
            .as_ref()
            .map(KimiK3MlaLocalWorklet::resolve_config),
        mla_moe: cfg
            .mla_moe
            .as_ref()
            .map(KimiK3MoeLocalWorklet::resolve_config),
        final_norm: cfg.final_norm.clone(),
        lm_head: cfg.lm_head.clone(),
    }
}

pub struct KimiK3SglangModel {
    pub name: String,
    pub parallel: KimiK3SglangParallel,
    pub counts: KimiK3LayerCounts,
    pub pp_stage_counts: Vec<KimiK3LayerCounts>,
    pub embedding: Option<Op<ElementwiseKernel>>,
    pub dense_attention: Option<KimiK3KdaLocalWorklet>,
    pub dense_ffn: Option<KimiK3DenseLocalWorklet>,
    pub kda_attention: Option<KimiK3KdaLocalWorklet>,
    pub kda_moe: Option<KimiK3MoeLocalWorklet>,
    pub mla_attention: Option<KimiK3MlaLocalWorklet>,
    pub mla_moe: Option<KimiK3MoeLocalWorklet>,
    pub final_norm: Option<Op<RmsNormKernel>>,
    pub lm_head: Option<Op<SingleGemmKernel>>,
    pub total_kv_bytes_per_token: Dim,
    pub recurrent_state_bytes_per_request: Dim,
    pub recurrent_checkpoint_interval_tokens: Dim,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

fn build_optional<K, C, F>(
    name: &str,
    config: Option<C>,
    build: F,
    bridge: &PerfApiBridge,
) -> Result<Option<Op<K>>, BuildError>
where
    K: crate::timing::Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError>,
{
    config
        .map(|cfg| {
            let full_name = name.to_string();
            Ok(Op::new(
                full_name.clone(),
                Arc::new(build(full_name, cfg, bridge)?),
            ))
        })
        .transpose()
}

fn local_kv_bytes_per_token(mla_layers: u32) -> Dim {
    Dim::param("mla_cache_width_bytes", 576) * Dim::param("mla_layers", mla_layers)
}

fn local_recurrent_state_bytes_per_request(kda_layers: u32, heads: u32, state_dtype: DType) -> Dim {
    let state = Dim::param("kda_heads", heads)
        * Dim::param("kda_state_key_dim", HEAD_DIM)
        * Dim::param("kda_state_value_dim", HEAD_DIM)
        * Dim::param("kda_state_dtype_bytes", state_dtype.size_bytes());
    let conv = Dim::param("kda_conv_taps", CONV_KERNEL - 1)
        * Dim::param("kda_conv_channels", 3 * heads * HEAD_DIM)
        * Dim::param("kda_conv_dtype_bytes", 2);
    (state + conv) * Dim::param("kda_layers", kda_layers)
}

pub fn build(
    name: String,
    resolved: KimiK3SglangResolved,
    bridge: &PerfApiBridge,
) -> Result<KimiK3SglangModel, BuildError> {
    let heads = resolved.parallel.heads_per_rank_value();
    let total_kv_bytes_per_token = local_kv_bytes_per_token(resolved.counts.mla);
    let recurrent_state_bytes_per_request = local_recurrent_state_bytes_per_request(
        resolved.counts.dense + resolved.counts.kda,
        heads,
        resolved.parallel.kda_state_dtype,
    );
    let recurrent_checkpoint_interval_tokens = if resolved.counts.dense + resolved.counts.kda > 0 {
        Dim::param("attn_res_block_size", ATTN_RES_BLOCK_SIZE)
    } else {
        Dim::param("no_kda_layers", 0)
    };
    let model = KimiK3SglangModel {
        embedding: build_optional(
            &format!("{name}.embedding"),
            resolved.embedding,
            ElementwiseKernel::build,
            bridge,
        )?,
        dense_attention: resolved
            .dense_attention
            .map(|cfg| KimiK3KdaLocalWorklet::build(format!("{name}.dense.attention"), cfg, bridge))
            .transpose()?,
        dense_ffn: resolved
            .dense_ffn
            .map(|cfg| KimiK3DenseLocalWorklet::build(format!("{name}.dense.ffn"), cfg, bridge))
            .transpose()?,
        kda_attention: resolved
            .kda_attention
            .map(|cfg| KimiK3KdaLocalWorklet::build(format!("{name}.kda.attention"), cfg, bridge))
            .transpose()?,
        kda_moe: resolved
            .kda_moe
            .map(|cfg| KimiK3MoeLocalWorklet::build(format!("{name}.kda.moe"), cfg, bridge))
            .transpose()?,
        mla_attention: resolved
            .mla_attention
            .map(|cfg| KimiK3MlaLocalWorklet::build(format!("{name}.mla.attention"), cfg, bridge))
            .transpose()?,
        mla_moe: resolved
            .mla_moe
            .map(|cfg| KimiK3MoeLocalWorklet::build(format!("{name}.mla.moe"), cfg, bridge))
            .transpose()?,
        final_norm: build_optional(
            &format!("{name}.final_norm"),
            resolved.final_norm,
            RmsNormKernel::build,
            bridge,
        )?,
        lm_head: build_optional(
            &format!("{name}.lm_head"),
            resolved.lm_head,
            SingleGemmKernel::build,
            bridge,
        )?,
        name,
        parallel: resolved.parallel,
        counts: resolved.counts,
        pp_stage_counts: resolved.pp_stage_counts,
        total_kv_bytes_per_token,
        recurrent_state_bytes_per_request,
        recurrent_checkpoint_interval_tokens,
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    let tree = model.cost_tree();
    let n_slots = tree.n_slots();
    let cost_flat = tree.flatten();
    tracing::info!(
        "[build] Kimi-K3 SGLang cost tree ({} leaf slots):\n{}",
        n_slots,
        tree.describe()
    );
    Ok(KimiK3SglangModel {
        cost_flat,
        n_slots,
        ..model
    })
}

impl KimiK3SglangModel {
    /// Exact model.work scope for this concrete rank-local model.  The timing
    /// tree is built once for the logical model, while the prediction's GPU
    /// extent includes every PP stage.  `work_scale` therefore converts the
    /// logical traversal to the per-stage work that the analyzer multiplies by
    /// the replica GPU count.
    pub fn model_work_scope(&self) -> serde_json::Value {
        let counts = |value: KimiK3LayerCounts| {
            json!({
                "dense": value.dense,
                "kda": value.kda,
                "mla": value.mla,
            })
        };
        json!({
            "schema_version": 1,
            "arch_type": ARCH_KIND,
            "kda_state_dtype": self.parallel.kda_state_dtype.as_str(),
            "heads_per_rank": self.parallel.heads_per_rank_value(),
            "global_heads": NUM_HEADS,
            "local_experts": self.parallel.local_experts_value_from_constants(),
            "global_experts": NUM_EXPERTS,
            "routing_experts": if self.parallel.ep_size == 1 {
                self.parallel.local_experts_value_from_constants()
            } else {
                NUM_EXPERTS
            },
            "routing_top_k": self.parallel.local_top_k.unwrap_or(TOP_K),
            "routing_source": if self.parallel.local_top_k.is_some() {
                if self.parallel.routing_histogram.is_some() {
                    "measured_alignment_payload"
                } else {
                    "analytic_uniform_poisson_like"
                }
            } else {
                "production_global_top16"
            },
            "layer_counts": counts(self.counts),
            "pp_stage_layer_counts": self.pp_stage_counts.iter().copied().map(counts).collect::<Vec<_>>(),
            "pp_size": self.parallel.pp_size,
            "work_scale": 1.0 / f64::from(self.parallel.pp_size),
            "include_model_io": self.embedding.is_some(),
        })
    }

    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let mut children = Vec::new();
        if let Some(embedding) = &self.embedding {
            children.push(CostNode::Labeled {
                label: "embedding gather".into(),
                child: Box::new(embedding.compile(&mut builder)),
            });
        }
        let dense_attention = self
            .dense_attention
            .as_ref()
            .map(|worklet| worklet.compile(&mut builder));
        let dense_ffn = self
            .dense_ffn
            .as_ref()
            .map(|worklet| worklet.compile(&mut builder));
        Self::append_layer(
            &mut children,
            "dense layer 1",
            self.counts.dense,
            dense_attention,
            dense_ffn,
        );
        let kda_attention = self
            .kda_attention
            .as_ref()
            .map(|worklet| worklet.compile(&mut builder));
        let kda_moe = self
            .kda_moe
            .as_ref()
            .map(|worklet| worklet.compile(&mut builder));
        Self::append_layer(
            &mut children,
            "KDA MoE layers",
            self.counts.kda,
            kda_attention,
            kda_moe,
        );
        let mla_attention = self
            .mla_attention
            .as_ref()
            .map(|worklet| worklet.compile(&mut builder));
        let mla_moe = self
            .mla_moe
            .as_ref()
            .map(|worklet| worklet.compile(&mut builder));
        Self::append_layer(
            &mut children,
            "MLA MoE layers",
            self.counts.mla,
            mla_attention,
            mla_moe,
        );
        if let Some(final_norm) = &self.final_norm {
            children.push(CostNode::Labeled {
                label: "final rms_norm".into(),
                child: Box::new(final_norm.compile(&mut builder)),
            });
        }
        if let Some(lm_head) = &self.lm_head {
            children.push(CostNode::Labeled {
                label: "lm_head".into(),
                child: Box::new(lm_head.compile(&mut builder)),
            });
        }
        builder.finish(CostNode::Labeled {
            label: format!(
                "{} (KimiK3SglangModel) [TP{}/EP{} / PP{}; heads/rank={}]",
                self.name,
                self.parallel.attn_tp_size,
                self.parallel.ep_size,
                self.parallel.pp_size,
                self.parallel.heads_per_rank_value()
            ),
            child: Box::new(CostNode::Sum(children)),
        })
    }

    fn append_layer(
        children: &mut Vec<CostNode>,
        label: &str,
        count: u32,
        attention: Option<CostNode>,
        ffn: Option<CostNode>,
    ) {
        if count == 0 {
            return;
        }
        let mut layer = Vec::new();
        if let Some(attention) = attention {
            layer.push(attention);
        }
        if let Some(ffn) = ffn {
            layer.push(ffn);
        }
        let body = CostNode::Labeled {
            label: label.into(),
            child: Box::new(CostNode::Sum(layer)),
        };
        children.push(CostNode::Scale {
            n: count,
            child: Box::new(body),
        });
    }

    fn eval_into(&self, input: &UnifiedArchInput, evaluator: &mut Evaluator) {
        let normalized = normalize_input(input, &self.parallel)
            .unwrap_or_else(|reason| panic!("invalid KimiK3SglangModel input: {reason}"));
        if let Some(embedding) = &self.embedding {
            embedding.eval(
                &ElementwiseKernelInput {
                    num_tokens: normalized.ffn_tokens,
                },
                evaluator,
            );
        }
        if let (Some(attention), Some(ffn)) = (&self.dense_attention, &self.dense_ffn) {
            attention.eval(&normalized.kda_attention, evaluator);
            ffn.eval(
                &KimiK3DenseLocalWorkletInput {
                    batch_tokens: normalized.ffn_tokens,
                    prefill_chunk_pairs: normalized.prefill_chunk_pairs.clone(),
                },
                evaluator,
            );
        }
        if let (Some(attention), Some(moe)) = (&self.kda_attention, &self.kda_moe) {
            attention.eval(&normalized.kda_attention, evaluator);
            moe.eval(
                &KimiK3MoeLocalWorkletInput {
                    num_tokens: normalized.ffn_tokens,
                    prefill_chunk_pairs: normalized.prefill_chunk_pairs.clone(),
                },
                evaluator,
            );
        }
        if let (Some(attention), Some(moe)) = (&self.mla_attention, &self.mla_moe) {
            attention.eval(&normalized.mla_attention, evaluator);
            moe.eval(
                &KimiK3MoeLocalWorkletInput {
                    num_tokens: normalized.ffn_tokens,
                    prefill_chunk_pairs: normalized.prefill_chunk_pairs.clone(),
                },
                evaluator,
            );
        }
        if let Some(final_norm) = &self.final_norm {
            final_norm.eval(
                &RmsNormKernelInput {
                    m: normalized.ffn_tokens,
                },
                evaluator,
            );
        }
        if let Some(lm_head) = &self.lm_head {
            lm_head.eval(
                &SingleGemmKernelInput {
                    m: normalized.request_count,
                },
                evaluator,
            );
        }
    }
}

#[derive(Debug)]
struct NormalizedInput {
    ffn_tokens: u32,
    request_count: u32,
    prefill_chunk_pairs: Vec<(u32, u32)>,
    kda_attention: KimiK3KdaLocalWorkletInput,
    mla_attention: KimiK3MlaLocalWorkletInput,
}

fn normalize_input(
    input: &UnifiedArchInput,
    parallel: &KimiK3SglangParallel,
) -> std::result::Result<NormalizedInput, String> {
    let expected_groups = usize::from(parallel.num_attn_dp_groups());
    if input.groups.len() != expected_groups {
        return Err(format!(
            "Kimi-K3 requires {expected_groups} attention DP group(s), got {}",
            input.groups.len()
        ));
    }
    if expected_groups == 1 {
        if !input.tokens_per_source_rank.is_empty() {
            return Err("single attention group requires empty tokens_per_source_rank".into());
        }
    } else {
        if input.tokens_per_source_rank.len() != expected_groups {
            return Err(format!(
                "tokens_per_source_rank must contain {expected_groups} entries"
            ));
        }
        for (index, (expected, group)) in input
            .tokens_per_source_rank
            .iter()
            .zip(&input.groups)
            .enumerate()
        {
            if *expected != group.batch_tokens {
                return Err(format!(
                    "tokens_per_source_rank[{index}]={} must equal group batch_tokens {}",
                    expected, group.batch_tokens
                ));
            }
        }
    }

    let mut ffn_tokens = 0_u32;
    let mut request_count = 0_u32;
    let mut prefill_chunk_pairs = Vec::new();
    let mut critical_index = 0_usize;
    for (index, group) in input.groups.iter().enumerate() {
        let prefill_tokens =
            group
                .prefill_chunk_pairs
                .iter()
                .try_fold(0_u32, |sum, &(prefix, append)| {
                    if append == 0 {
                        return Err(format!(
                            "group {index} contains a zero-length prefill chunk"
                        ));
                    }
                    let context = prefix
                        .checked_add(append)
                        .ok_or_else(|| format!("group {index} prefill context overflow"))?;
                    if context > MAX_MODEL_LEN {
                        return Err(format!(
                            "group {index} prefill context {context} exceeds {MAX_MODEL_LEN}"
                        ));
                    }
                    sum.checked_add(append)
                        .ok_or_else(|| format!("group {index} prefill token overflow"))
                })?;
        prefill_chunk_pairs.extend_from_slice(&group.prefill_chunk_pairs);
        if group.prefill_tokens != prefill_tokens {
            return Err(format!(
                "group {index} prefill_tokens {} must equal append sum {prefill_tokens}",
                group.prefill_tokens
            ));
        }
        let decode_tokens = u32::try_from(group.decode_kv_lens.len())
            .map_err(|_| format!("group {index} decode request count exceeds u32"))?;
        if group.decode_tokens != decode_tokens {
            return Err(format!(
                "group {index} decode_tokens {} must equal request count {decode_tokens}",
                group.decode_tokens
            ));
        }
        let decode_kv_sum = group.decode_kv_lens.iter().try_fold(0_u32, |sum, &kv| {
            if !(1..=MAX_MODEL_LEN).contains(&kv) {
                return Err(format!(
                    "group {index} decode context {kv} must be in 1..={MAX_MODEL_LEN}"
                ));
            }
            sum.checked_add(kv)
                .ok_or_else(|| format!("group {index} decode KV length overflow"))
        })?;
        if group.total_kv_len != decode_kv_sum {
            return Err(format!(
                "group {index} total_kv_len {} must equal decode KV sum {decode_kv_sum}",
                group.total_kv_len
            ));
        }
        let active = prefill_tokens
            .checked_add(decode_tokens)
            .ok_or_else(|| format!("group {index} active token overflow"))?;
        if group.batch_tokens != active {
            return Err(format!(
                "group {index} batch_tokens {} must equal prefill+decode {active}",
                group.batch_tokens
            ));
        }
        ffn_tokens = ffn_tokens
            .checked_add(active)
            .ok_or_else(|| "FFN token count overflow".to_string())?;
        request_count = request_count
            .checked_add(group.request_count())
            .ok_or_else(|| "request count overflow".to_string())?;
        if input.groups[index].batch_tokens > input.groups[critical_index].batch_tokens {
            critical_index = index;
        }
    }
    if ffn_tokens == 0 {
        return Err("Kimi-K3 requires nonempty active work".into());
    }
    let critical = &input.groups[critical_index];
    Ok(NormalizedInput {
        ffn_tokens,
        request_count,
        prefill_chunk_pairs: prefill_chunk_pairs.clone(),
        kda_attention: KimiK3KdaLocalWorkletInput {
            batch_tokens: critical.batch_tokens,
            decode_tokens: critical.decode_kv_lens.len() as u32,
            prefill_chunk_pairs: critical.prefill_chunk_pairs.clone(),
        },
        mla_attention: KimiK3MlaLocalWorkletInput {
            batch_tokens: critical.batch_tokens,
            decode_kv_lens: critical.decode_kv_lens.clone(),
            prefill_chunk_pairs: critical.prefill_chunk_pairs.clone(),
        },
    })
}

impl IterwiseUnifiedModel for KimiK3SglangModel {
    fn model_work_scope(&self) -> Option<serde_json::Value> {
        Some(self.model_work_scope())
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
        debug_assert_eq!(evaluator.filled(), self.n_slots);
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
        debug_assert_eq!(evaluator.filled(), self.n_slots);
        let total = CostTree::aggregate(&self.cost_flat, slots, scratch);
        debug_assert_eq!(inputs.len(), self.n_slots);
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
        self.parallel.gpus_per_replica()
    }

    fn num_attn_dp_groups(&self) -> u16 {
        self.parallel.num_attn_dp_groups()
    }

    fn num_attn_shards(&self) -> u16 {
        self.parallel.num_attn_shards()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::ArchGroupInput;

    fn model() -> KimiK3ModelCfg {
        KimiK3ModelCfg::from_json(
            Path::new("model/config/kimi_k3.json"),
            &ModelSpec {
                model_config: "model/config/kimi_k3.json".into(),
                num_layers: None,
                sim_num_layers: None,
                fp8: false,
            },
        )
        .unwrap()
    }

    fn rank1(sim_kda_layers: Option<u32>, sim_mla_layers: Option<u32>) -> KimiK3SglangParallel {
        KimiK3SglangParallel {
            attn_tp_size: 1,
            ep_size: 1,
            pp_size: 1,
            dcp_size: 1,
            kda_state_dtype: DType::Bf16,
            heads_per_rank: Some(12),
            local_experts: Some(112),
            local_top_k: Some(2),
            routing_histogram: None,
            sim_kda_layers,
            sim_mla_layers,
            gpu_name: "NVIDIA B200".into(),
        }
    }

    #[test]
    fn config_consumes_the_exact_explicit_schedule() {
        let cfg = model();
        assert_eq!(cfg.full_attn_layers.len(), 24);
        assert_eq!(cfg.kda_layers.len(), 69);
        assert_eq!(cfg.kda_layers[0], 1);
        let production = KimiK3SglangParallel {
            attn_tp_size: 8,
            ep_size: 8,
            pp_size: 2,
            dcp_size: 1,
            kda_state_dtype: DType::Bf16,
            heads_per_rank: None,
            local_experts: None,
            local_top_k: None,
            routing_histogram: None,
            sim_kda_layers: None,
            sim_mla_layers: None,
            gpu_name: "NVIDIA B200".into(),
        };
        let configs = build_configs(&cfg, &production).unwrap();
        assert_eq!(
            configs.counts,
            KimiK3LayerCounts {
                dense: 1,
                kda: 68,
                mla: 24
            }
        );
        assert_eq!(
            configs.pp_stage_counts,
            vec![
                KimiK3LayerCounts {
                    dense: 1,
                    kda: 34,
                    mla: 11,
                },
                KimiK3LayerCounts {
                    dense: 0,
                    kda: 34,
                    mla: 13,
                },
            ]
        );
        assert_eq!(production.num_attn_dp_groups(), 1);
        assert_eq!(production.gpus_per_replica(), 16);

        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let resolved = resolve_configs(&configs);
        assert_eq!(
            resolved
                .mla_moe
                .as_ref()
                .unwrap()
                .mxfp4_fused_moe
                .num_experts,
            896
        );
        let enumerated = build("unified".into(), resolved, &bridge).unwrap();
        let scope = enumerated.model_work_scope();
        assert_eq!(scope["kda_state_dtype"], json!("bf16"));
        assert_eq!(scope["heads_per_rank"], json!(12));
        assert_eq!(scope["local_experts"], json!(112));
        assert_eq!(scope["pp_size"], json!(2));
        assert_eq!(scope["work_scale"], json!(0.5));
        assert_eq!(scope["include_model_io"], json!(true));
        assert_eq!(scope["pp_stage_layer_counts"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn rank1_overrides_select_one_layer_and_keep_local_shapes() {
        let cfg = model();
        let kda = build_configs(&cfg, &rank1(Some(1), Some(0))).unwrap();
        assert_eq!(
            kda.counts,
            KimiK3LayerCounts {
                dense: 0,
                kda: 1,
                mla: 0
            }
        );
        let resolved = resolve_configs(&kda);
        assert_eq!(resolved.kda_attention.as_ref().unwrap().raw_cfg.heads, 12);
        assert_eq!(
            resolved.kda_moe.as_ref().unwrap().raw_cfg.local_experts,
            112
        );
        assert_eq!(
            resolved
                .kda_moe
                .as_ref()
                .unwrap_or_else(|| panic!("rank-1 KDA MoE missing"))
                .mxfp4_fused_moe
                .num_experts,
            112
        );
        let mla = build_configs(&cfg, &rank1(Some(0), Some(1))).unwrap();
        assert_eq!(
            mla.counts,
            KimiK3LayerCounts {
                dense: 0,
                kda: 0,
                mla: 1
            }
        );
        assert!(!mla.include_model_io);
    }

    #[test]
    fn local_cache_and_recurrent_bytes_are_recipe_values() {
        let cfg = model();
        let production = KimiK3SglangParallel {
            attn_tp_size: 8,
            ep_size: 8,
            pp_size: 2,
            dcp_size: 1,
            kda_state_dtype: DType::Bf16,
            heads_per_rank: None,
            local_experts: None,
            local_top_k: None,
            routing_histogram: None,
            sim_kda_layers: None,
            sim_mla_layers: None,
            gpu_name: "NVIDIA B200".into(),
        };
        let configs = build_configs(&cfg, &production).unwrap();
        assert_eq!(local_kv_bytes_per_token(configs.counts.mla).get(), 13_824);
        assert_eq!(
            local_recurrent_state_bytes_per_request(
                configs.counts.dense + configs.counts.kda,
                12,
                DType::Bf16,
            )
            .get(),
            (420_864 * 69)
        );
        assert_eq!(
            local_recurrent_state_bytes_per_request(
                configs.counts.dense + configs.counts.kda,
                12,
                DType::Fp32,
            )
            .get(),
            814_080 * 69
        );
    }

    #[test]
    fn production_cost_tree_freezes_slot_order_and_communication_leaves() {
        let production = KimiK3SglangParallel {
            attn_tp_size: 8,
            ep_size: 8,
            pp_size: 2,
            dcp_size: 1,
            kda_state_dtype: DType::Bf16,
            heads_per_rank: None,
            local_experts: None,
            local_top_k: None,
            routing_histogram: None,
            sim_kda_layers: None,
            sim_mla_layers: None,
            gpu_name: "NVIDIA B200".into(),
        };
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let configs = build_configs(&model(), &production).unwrap();
        let enumerated = build("unified".into(), resolve_configs(&configs), &bridge).unwrap();
        let tree = enumerated.cost_tree();
        let names: Vec<&str> = tree.slots.iter().map(|slot| slot.name.as_str()).collect();
        assert_eq!(tree.n_slots(), 91);
        assert_eq!(
            names,
            [
                "unified.embedding",
                "unified.dense.attention.input_layernorm",
                "unified.dense.attention.attn_res_prefill",
                "unified.dense.attention.qkvbfg_a_proj",
                "unified.dense.attention.qkvbfg_a_proj_bfa",
                "unified.dense.attention.qkvbfg_a_proj_prefill",
                "unified.dense.attention.qkvbfg_a_proj_bfa_prefill",
                "unified.dense.attention.qkvbfg_f_b_prefill",
                "unified.dense.attention.kda_conv_decode",
                "unified.dense.attention.kda_recurrent_decode",
                "unified.dense.attention.kda_gated_norm",
                "unified.dense.attention.kda_conv_prefill",
                "unified.dense.attention.kda_chunk_prefill",
                "unified.dense.attention.kda_gated_norm_prefill",
                "unified.dense.attention.o_proj",
                "unified.dense.attention.o_proj_prefill",
                "unified.dense.attention.tp_allreduce_zero",
                "unified.dense.attention.post_attention_layernorm",
                "unified.dense.ffn.gate_up",
                "unified.dense.ffn.situ",
                "unified.dense.ffn.down",
                "unified.kda.attention.input_layernorm",
                "unified.kda.attention.attn_res_prefill",
                "unified.kda.attention.qkvbfg_a_proj",
                "unified.kda.attention.qkvbfg_a_proj_bfa",
                "unified.kda.attention.qkvbfg_a_proj_prefill",
                "unified.kda.attention.qkvbfg_a_proj_bfa_prefill",
                "unified.kda.attention.qkvbfg_f_b_prefill",
                "unified.kda.attention.kda_conv_decode",
                "unified.kda.attention.kda_recurrent_decode",
                "unified.kda.attention.kda_gated_norm",
                "unified.kda.attention.kda_conv_prefill",
                "unified.kda.attention.kda_chunk_prefill",
                "unified.kda.attention.kda_gated_norm_prefill",
                "unified.kda.attention.o_proj",
                "unified.kda.attention.o_proj_prefill",
                "unified.kda.attention.tp_allreduce_zero",
                "unified.kda.attention.post_attention_layernorm",
                "unified.kda.moe.merged_front",
                "unified.kda.moe.merged_front_prefill",
                "unified.kda.moe.shared_gate_up_activation",
                "unified.kda.moe.shared_gate_up_activation_prefill",
                "unified.kda.moe.shared_down",
                "unified.kda.moe.shared_down_prefill",
                "unified.kda.moe.mxfp4_fused_moe",
                "unified.kda.moe.mxfp4_fused_moe_prefill",
                "unified.kda.moe.routed_norm",
                "unified.kda.moe.latent_up",
                "unified.kda.moe.latent_up_prefill",
                "unified.kda.moe.add3",
                "unified.kda.moe.add3_prefill",
                "unified.kda.moe.ep_alltoall_zero",
                "unified.mla.attention.input_layernorm",
                "unified.mla.attention.attn_res_prefill",
                "unified.mla.attention.fused_qkv_a_proj",
                "unified.mla.attention.q_a_layernorm",
                "unified.mla.attention.q_b_proj",
                "unified.mla.attention.q_b_proj_prefill",
                "unified.mla.attention.kv_a_layernorm",
                "unified.mla.attention.q_absorb",
                "unified.mla.attention.mla_cache_append",
                "unified.mla.attention.mla_decode_attention",
                "unified.mla.attention.v_up",
                "unified.mla.attention.mla_prefix_gather",
                "unified.mla.attention.mla_kv_b_proj_prefill",
                "unified.mla.attention.mla_prefill_attention_prefix",
                "unified.mla.attention.mla_prefill_attention_causal",
                "unified.mla.attention.mla_merge_state",
                "unified.mla.attention.output_gate",
                "unified.mla.attention.output_gate_prefill",
                "unified.mla.attention.sigmoid_mul",
                "unified.mla.attention.o_proj",
                "unified.mla.attention.o_proj_prefill",
                "unified.mla.attention.tp_allreduce_zero",
                "unified.mla.attention.post_attention_layernorm",
                "unified.mla.moe.merged_front",
                "unified.mla.moe.merged_front_prefill",
                "unified.mla.moe.shared_gate_up_activation",
                "unified.mla.moe.shared_gate_up_activation_prefill",
                "unified.mla.moe.shared_down",
                "unified.mla.moe.shared_down_prefill",
                "unified.mla.moe.mxfp4_fused_moe",
                "unified.mla.moe.mxfp4_fused_moe_prefill",
                "unified.mla.moe.routed_norm",
                "unified.mla.moe.latent_up",
                "unified.mla.moe.latent_up_prefill",
                "unified.mla.moe.add3",
                "unified.mla.moe.add3_prefill",
                "unified.mla.moe.ep_alltoall_zero",
                "unified.final_norm",
                "unified.lm_head",
            ]
        );
        assert_eq!(
            tree.slots
                .iter()
                .filter(|slot| slot.kind == "all_reduce" || slot.kind == "moe_alltoall")
                .count(),
            5
        );
    }

    #[test]
    fn normalize_input_uses_the_busiest_attention_group_and_sums_ffn_tokens() {
        let parallel = KimiK3SglangParallel {
            attn_tp_size: 2,
            ep_size: 4,
            pp_size: 1,
            dcp_size: 1,
            kda_state_dtype: DType::Bf16,
            heads_per_rank: None,
            local_experts: None,
            local_top_k: None,
            routing_histogram: None,
            sim_kda_layers: Some(1),
            sim_mla_layers: Some(0),
            gpu_name: "NVIDIA B200".into(),
        };
        let input = UnifiedArchInput {
            groups: vec![
                ArchGroupInput {
                    batch_tokens: 2,
                    decode_tokens: 2,
                    decode_kv_lens: vec![8, 8],
                    total_kv_len: 16,
                    ..ArchGroupInput::default()
                },
                ArchGroupInput {
                    batch_tokens: 3,
                    decode_tokens: 3,
                    decode_kv_lens: vec![32, 32, 32],
                    total_kv_len: 96,
                    ..ArchGroupInput::default()
                },
            ],
            tokens_per_source_rank: vec![2, 3],
        };
        let normalized = normalize_input(&input, &parallel).unwrap();
        assert_eq!(normalized.ffn_tokens, 5);
        assert_eq!(normalized.kda_attention.batch_tokens, 3);
        assert_eq!(normalized.kda_attention.decode_tokens, 3);
    }
}
