//! GLM-5.3-Flash (`Glm5NextForConditionalGeneration`) FP8 block checkpoint on
//! B200, aligned to the vLLM fork's TP4 / EP4 deployment (MTP off).
//!
//! One rank group: attention and the dense FFN are tensor-parallel (16 local
//! KDA heads, 16 local MLA heads, the indexer's 32 heads replicated), the
//! routed experts expert-parallel (72 of 288 per rank). Every sublayer ends in
//! a plain TP all-reduce (`flashinfer_comm.allreduce_fusion`, kAllReduce,
//! MNNVL), 91 per iteration: embedding + 45 attention + 3 dense + 42 MoE.
//!
//! The 45 layers are hybrid: `linear_attn_config.kda_layers` are Kimi Delta
//! Attention, the rest (every 4th, 3..43) DeepSeek sparse attention with the
//! pooled (kpool) indexer; layers 0-2 carry a dense FFN and 3-44 an MoE. The
//! residual stream is 4-wide mHC: layer 0's attention opens with
//! `mhc_pre_rms_norm`, every later sublayer boundary is one fused
//! `mhc_fused_post_pre_rms_norm`, and the last post runs alone before the
//! `hc_contract` mean and the final RMSNorm.
//!
//! Structure (Scale folds over identical layer groups; costs are order-free):
//!
//! ```text
//! prologue       embedding, all-reduce, hc_expand
//! layer 0        mhc_pre + KDA + AR,   mhc_fused + dense FFN + AR   (x1)
//! layers 1-2     mhc_fused + KDA + AR, mhc_fused + dense FFN + AR   (x2)
//! KDA+MoE        mhc_fused + KDA + AR, mhc_fused + MoE + AR         (x31)
//! DSA+MoE        mhc_fused + DSA + AR, mhc_fused + MoE + AR         (x11)
//! epilogue       terminal mHC post, hc_contract mean, final norm, lm_head
//! ```
//!
//! Attention ranks are symmetric, so one rank is compiled. The MoE is a `Max`
//! over the EP ranks' ranked routed workloads; inside each rank the shared
//! expert is a parallel branch, `Sum[Max[shared, routed], shared_serial]`,
//! filled on the aux-stream copy when the batch has at most 256 tokens and on
//! the serial copy otherwise (vLLM `shared_experts.py`).
//!
//! lm_head and sampling are outside the capture's forward phase; the BF16
//! lm_head GEMM over one row per request is kept, as in the GLM-5.3 DFlash2
//! arch, so a unified worker is charged for it.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde_json::Value;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::Fabric;
use crate::op::mhc::{MhcTerminalPostConfig, MhcTerminalPostInput, MhcTerminalPostOp};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{
    AllReduceFusionKernel, AllReduceFusionKernelConfig, AllReduceFusionKernelInput,
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    MhcFusedPostPreRmsNormKernel, MhcPreRmsNormKernel, MhcRmsNormKernelConfig,
    MhcRmsNormKernelInput, RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput,
    SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Dim, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, Probe, SlotInput,
};
use crate::worklet::{
    Glm53DsaAttnLocalWorklet, Glm53DsaAttnLocalWorkletConfig, Glm53DsaAttnLocalWorkletInput,
    Glm53DsaAttnLocalWorkletResolved, Glm53Fp8MlpLocalWorklet, Glm53Fp8MlpLocalWorkletConfig,
    Glm53Fp8MlpLocalWorkletInput, Glm53Fp8MlpLocalWorkletResolved, Glm53KdaAttnLocalWorklet,
    Glm53KdaAttnLocalWorkletConfig, Glm53KdaAttnLocalWorkletInput,
    Glm53KdaAttnLocalWorkletResolved, Glm53MoeRouterLocalWorklet, Glm53MoeRouterLocalWorkletConfig,
    Glm53MoeRouterLocalWorkletInput, Glm53RoutedMoeLocalWorklet, Glm53RoutedMoeLocalWorkletConfig,
    Glm53RoutedMoeLocalWorkletInput, Glm53RoutedMoeLocalWorkletResolved,
};

const ARCH_KIND: &str = "glm53_flash_vllm_fp8_kda_dsa_moe";
const ACTIVATION_DTYPE: DType = DType::Bf16;
/// Sparse page-table width: `round_up(index_topk + index_kpool - 1, 128)`.
const SELECTED_K: u32 = 2176;
const CACHE_BLOCK_SIZE: u32 = 64;
/// vLLM's hybrid block size for this deployment ("attention block size 2176"
/// in the capture's server log): the token interval at which the KDA state is
/// checkpointed and a prefix-cache hit can resume.
const HYBRID_BLOCK_SIZE: u32 = 2176;
/// vLLM runs the shared expert on an aux stream at or below this batch size.
const SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: u32 = 256;
/// `Max{overlap}` of the aux-stream shared expert against its routed slice.
/// The two streams contend for SMs and HBM: decode's routed and shared kernels
/// each run 15-75% above their isolated profile.db rows, while the serial
/// (T > 256) copies match within 2%. Measured union / isolated max over
/// decode: 1.091 (20260924_0, bs 32) and 1.129 (20260925_0 diverse_100,
/// bs 1-24, graph-padded) -> 1/0.9.
const SHARED_EXPERTS_STREAM_OVERLAP: f32 = 0.9;

const BF16_GEMM_BACKENDS: &[&str] = &["torch_linear_vllm"];
const FP8_GEMM_BACKENDS: &[&str] = &["deepgemm_vllm_fork"];
const FP8_QUANT_BACKENDS: &[&str] = &["vllm_fork_cuda"];
const ROUTER_GEMM_BACKENDS: &[&str] = &["torch_cublas"];
const FP32_GEMM_BACKENDS: &[&str] = &["torch_cublas_vllm_fork"];
const ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
const MHC_BACKENDS: &[&str] = &["vllm_tilelang"];
const RMS_NORM_BACKENDS: &[&str] = &["vllm_cuda"];
const KDA_BACKENDS: &[&str] = &["vllm_triton"];
const CONV_BACKENDS: &[&str] = &["vllm_triton"];
const QKV_NORM_BACKENDS: &[&str] = &["vllm_fork_triton"];
const Q_ABSORB_BACKENDS: &[&str] = &["torch_mla_q_absorb_glm53"];
const V_UP_BACKENDS: &[&str] = &["torch_mla_v_up_glm53"];
const MQA_LOGITS_BACKENDS: &[&str] = &["deepgemm_fp8_vllm_fork"];
const TOPK_BACKENDS: &[&str] = &["vllm_fork_cuda"];
/// The fork's `fp8_fp4_mqa_logits` is DeepGEMM's `sm100_mqa_logits`; the
/// packaged `deepgemm_fp8` rows time the same kernel (0.573 ms at 2048 x 65536
/// against 0.55 ms measured at a 261K-token context in capture 20260925_4).
const MQA_LOGITS_PREFILL_BACKENDS: &[&str] = &["deepgemm_fp8"];
const TOPK_PREFILL_BACKENDS: &[&str] = &["vllm_fork_cuda"];
const SPARSE_ATTN_BACKENDS: &[&str] = &["flashinfer_trtllm_fp8_vllm_fork"];
const MLA_APPEND_BACKENDS: &[&str] = &["vllm_cuda"];
const INDEX_REMAP_BACKENDS: &[&str] = &["vllm_fork_triton"];
const FUSED_MOE_BACKENDS: &[&str] = &["flashinfer_trtllm_fp8_block_sm100"];
const ALL_REDUCE_BACKENDS: &[&str] = &["flashinfer_mnnvl"];

/// The checkpoint's dimensions, read strictly from its `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct Glm53FlashModelCfg {
    pub hidden: u32,
    pub num_layers: u32,
    pub kda_layers: Vec<u32>,
    pub first_k_dense_replace: u32,
    pub kda_num_heads: u32,
    pub kda_head_dim: u32,
    pub short_conv_kernel_size: u32,
    pub num_attention_heads: u32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub v_head_dim: u32,
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub index_kpool: u32,
    pub intermediate_size: u32,
    pub moe_intermediate_size: u32,
    pub n_routed_experts: u32,
    pub num_experts_per_tok: u32,
    pub n_shared_experts: u32,
    pub n_group: u32,
    pub topk_group: u32,
    /// `routed_scaling_factor` as an exact fraction.
    pub routed_scaling: (u32, u32),
    pub hc_mult: u32,
    pub rms_norm_eps: f64,
    pub vocab_size: u32,
}

impl Glm53FlashModelCfg {
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading GLM-5.3-Flash config {}", path.display()))?;
        let root: Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing GLM-5.3-Flash config {}", path.display()))?;
        Self::from_value(&root)
    }

    pub fn from_value(root: &Value) -> Result<Self> {
        let cfg = root.get("text_config").unwrap_or(root);
        let u = |key: &str| -> Result<u32> {
            let value = cfg.get(key).and_then(Value::as_u64).with_context(|| {
                format!("GLM-5.3-Flash config field `{key}` must be an integer")
            })?;
            u32::try_from(value).with_context(|| format!("`{key}` exceeds u32"))
        };
        let f = |key: &str| -> Result<f64> {
            cfg.get(key)
                .and_then(Value::as_f64)
                .with_context(|| format!("GLM-5.3-Flash config field `{key}` must be a number"))
        };
        let model_type = cfg.get("model_type").and_then(Value::as_str).unwrap_or("");
        ensure!(
            model_type == "glm5_next_text" || model_type == "glm5_next",
            "{ARCH_KIND} requires a GLM-5.3-Flash (glm5_next) config, got model_type {model_type:?}"
        );
        let linear = cfg
            .get("linear_attn_config")
            .context("GLM-5.3-Flash config needs linear_attn_config")?;
        let lu = |key: &str| -> Result<u32> {
            let value = linear
                .get(key)
                .and_then(Value::as_u64)
                .with_context(|| format!("linear_attn_config.{key} must be an integer"))?;
            u32::try_from(value).with_context(|| format!("linear_attn_config.{key} exceeds u32"))
        };
        let kda_layers: Vec<u32> = linear
            .get("kda_layers")
            .and_then(Value::as_array)
            .context("linear_attn_config.kda_layers must be an array")?
            .iter()
            .map(|layer| {
                layer
                    .as_u64()
                    .and_then(|layer| u32::try_from(layer).ok())
                    .context("kda_layers entries must be u32")
            })
            .collect::<Result<_>>()?;
        let routed_scaling_factor = f("routed_scaling_factor")?;
        let doubled = routed_scaling_factor * 2.0;
        ensure!(
            doubled.fract() == 0.0 && doubled > 0.0,
            "routed_scaling_factor {routed_scaling_factor} must be a positive multiple of 1/2"
        );
        let model = Self {
            hidden: u("hidden_size")?,
            num_layers: u("num_hidden_layers")?,
            kda_layers,
            first_k_dense_replace: u("first_k_dense_replace")?,
            kda_num_heads: lu("num_heads")?,
            kda_head_dim: lu("head_dim")?,
            short_conv_kernel_size: lu("short_conv_kernel_size")?,
            num_attention_heads: u("num_attention_heads")?,
            q_lora_rank: u("q_lora_rank")?,
            kv_lora_rank: u("kv_lora_rank")?,
            qk_nope_head_dim: u("qk_nope_head_dim")?,
            v_head_dim: u("v_head_dim")?,
            index_n_heads: u("index_n_heads")?,
            index_head_dim: u("index_head_dim")?,
            index_topk: u("index_topk")?,
            index_kpool: u("index_kpool")?,
            intermediate_size: u("intermediate_size")?,
            moe_intermediate_size: u("moe_intermediate_size")?,
            n_routed_experts: u("n_routed_experts")?,
            num_experts_per_tok: u("num_experts_per_tok")?,
            n_shared_experts: u("n_shared_experts")?,
            n_group: u("n_group")?,
            topk_group: u("topk_group")?,
            routed_scaling: (doubled as u32, 2),
            hc_mult: u("hc_mult")?,
            rms_norm_eps: f("rms_norm_eps")?,
            vocab_size: u("vocab_size")?,
        };
        ensure!(
            u("qk_rope_head_dim")? == 0,
            "{ARCH_KIND} models the no-rope MLA latent; qk_rope_head_dim must be 0"
        );
        ensure!(
            model.n_shared_experts == 1,
            "exactly one shared expert is modeled"
        );
        ensure!(
            model
                .kda_layers
                .iter()
                .all(|&layer| layer < model.num_layers),
            "kda_layers must index model layers"
        );
        ensure!(
            model.first_k_dense_replace <= model.num_layers,
            "first_k_dense_replace exceeds num_hidden_layers"
        );
        Ok(model)
    }

    fn is_kda(&self, layer: u32) -> bool {
        self.kda_layers.contains(&layer)
    }

    fn is_dense(&self, layer: u32) -> bool {
        layer < self.first_k_dense_replace
    }

    pub fn num_dsa_layers(&self) -> u32 {
        (0..self.num_layers).filter(|&l| !self.is_kda(l)).count() as u32
    }

    pub fn num_moe_layers(&self) -> u32 {
        self.num_layers - self.first_k_dense_replace
    }
}

/// Deployment layout: one TP = EP rank group.
#[derive(Clone, Debug)]
pub struct Glm53FlashVllmParallel {
    pub tp_size: u16,
    pub max_model_len: u32,
    pub gpu_name: String,
    /// vLLM `--cudagraph-capture-sizes`; empty runs eager (no padding).
    pub cudagraph_capture_sizes: Vec<u32>,
}

/// Which attention and FFN a layer group carries, and how it opens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttnKind {
    Kda,
    Dsa,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FfnKind {
    Dense,
    Moe,
}

/// A run of layers with one identical sublayer shape, folded by `Scale{n}`.
#[derive(Clone, Debug)]
pub struct Glm53FlashLayerGroup {
    pub label: String,
    pub layers: Vec<u32>,
    attn: AttnKind,
    ffn: FfnKind,
    /// Only layer 0 opens with a standalone mHC pre.
    opens_stream: bool,
}

/// The ordered layer groups of a model config. Layers with the same attention
/// kind, FFN kind, and opening boundary fold into one group.
fn layer_groups(model: &Glm53FlashModelCfg) -> Vec<Glm53FlashLayerGroup> {
    let mut groups: Vec<Glm53FlashLayerGroup> = Vec::new();
    for layer in 0..model.num_layers {
        let attn = if model.is_kda(layer) {
            AttnKind::Kda
        } else {
            AttnKind::Dsa
        };
        let ffn = if model.is_dense(layer) {
            FfnKind::Dense
        } else {
            FfnKind::Moe
        };
        let opens_stream = layer == 0;
        if let Some(group) = groups
            .iter_mut()
            .find(|g| g.attn == attn && g.ffn == ffn && g.opens_stream == opens_stream)
        {
            group.layers.push(layer);
            continue;
        }
        let label = format!(
            "{}{}_{}",
            if opens_stream { "first_" } else { "" },
            match attn {
                AttnKind::Kda => "kda",
                AttnKind::Dsa => "dsa",
            },
            match ffn {
                FfnKind::Dense => "dense",
                FfnKind::Moe => "moe",
            }
        );
        groups.push(Glm53FlashLayerGroup {
            label,
            layers: vec![layer],
            attn,
            ffn,
            opens_stream,
        });
    }
    groups
}

/// Every configuration the model is built from.
#[derive(Clone, Debug)]
pub struct Glm53FlashVllmConfigs {
    pub model: Glm53FlashModelCfg,
    pub parallel: Glm53FlashVllmParallel,
    pub groups: Vec<Glm53FlashLayerGroup>,
    pub kda: Glm53KdaAttnLocalWorkletConfig,
    pub dsa: Glm53DsaAttnLocalWorkletConfig,
    pub dense_ffn: Glm53Fp8MlpLocalWorkletConfig,
    pub shared_expert: Glm53Fp8MlpLocalWorkletConfig,
    pub router: Glm53MoeRouterLocalWorkletConfig,
    /// One per EP rank, ranked by routed workload.
    pub routed: Vec<Glm53RoutedMoeLocalWorkletConfig>,
    pub embedding: ElementwiseKernelConfig,
    pub hc_expand: ElementwiseKernelConfig,
    pub mhc: MhcRmsNormKernelConfig,
    pub all_reduce: AllReduceFusionKernelConfig,
    /// Routed-input copy ahead of routing, and the shared + routed combine.
    pub moe_input_glue: ElementwiseKernelConfig,
    pub moe_combine_glue: ElementwiseKernelConfig,
    pub hc_contract_mean: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
}

pub fn build_configs(
    model: &Glm53FlashModelCfg,
    parallel: &Glm53FlashVllmParallel,
    demand: &ExpertDemand,
) -> std::result::Result<Glm53FlashVllmConfigs, BuildError> {
    let tp = u32::from(parallel.tp_size);
    let divide = |what: &str, value: u32| -> std::result::Result<u32, BuildError> {
        if tp == 0 || value % tp != 0 {
            return Err(fit_failed(format!("{what} {value} must divide by TP {tp}")));
        }
        Ok(value / tp)
    };
    let gpu = parallel.gpu_name.clone();
    let hidden_bytes = model.hidden * ACTIVATION_DTYPE.size_bytes();
    let stream_bytes = model.hc_mult * hidden_bytes;
    let ew = |input: u32, output: u32| ElementwiseKernelConfig {
        backends: ELEMENTWISE_BACKENDS.to_vec(),
        gpu_name: gpu.clone(),
        input_bytes_per_token: input.into(),
        output_bytes_per_token: output.into(),
    };
    let mlp = |intermediate: u32| Glm53Fp8MlpLocalWorkletConfig {
        hidden: model.hidden.into(),
        intermediate: intermediate.into(),
        activation_dtype: ACTIVATION_DTYPE,
        gpu_name: gpu.clone(),
        quant_backends: FP8_QUANT_BACKENDS.to_vec(),
        fp8_gemm_backends: FP8_GEMM_BACKENDS.to_vec(),
        elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
    };
    if model.index_topk + model.index_kpool - 1 > SELECTED_K {
        return Err(fit_failed(
            "pooled indexer window exceeds the 2176-wide page table",
        ));
    }
    let routed_template = Glm53RoutedMoeLocalWorkletConfig {
        hidden: model.hidden.into(),
        moe_intermediate: model.moe_intermediate_size.into(),
        num_experts: model.n_routed_experts.into(),
        ep_size: parallel.tp_size,
        top_k: model.num_experts_per_tok,
        n_group: model.n_group,
        topk_group: model.topk_group,
        routed_scaling_numerator: model.routed_scaling.0,
        routed_scaling_denominator: model.routed_scaling.1,
        activation_dtype: ACTIVATION_DTYPE,
        gpu_name: gpu.clone(),
        quant_backends: FP8_QUANT_BACKENDS.to_vec(),
        fused_moe_backends: FUSED_MOE_BACKENDS.to_vec(),
        expert_demand: demand.clone(),
        folded_rank_position: 0,
    };
    divide("n_routed_experts", model.n_routed_experts)?;
    Ok(Glm53FlashVllmConfigs {
        groups: layer_groups(model),
        kda: Glm53KdaAttnLocalWorkletConfig {
            hidden: model.hidden.into(),
            num_heads: divide("KDA heads", model.kda_num_heads)?.into(),
            head_dim: model.kda_head_dim.into(),
            gate_rank: model.kda_head_dim,
            conv_kernel_size: model.short_conv_kernel_size.into(),
            activation_dtype: ACTIVATION_DTYPE,
            gpu_name: gpu.clone(),
            bf16_gemm_backends: BF16_GEMM_BACKENDS.to_vec(),
            conv_backends: CONV_BACKENDS.to_vec(),
            core_backends: KDA_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        },
        dsa: Glm53DsaAttnLocalWorkletConfig {
            hidden: model.hidden.into(),
            num_heads: divide("attention heads", model.num_attention_heads)?.into(),
            q_lora_rank: model.q_lora_rank.into(),
            kv_lora_rank: model.kv_lora_rank.into(),
            qk_nope_head_dim: model.qk_nope_head_dim.into(),
            v_head_dim: model.v_head_dim.into(),
            index_num_heads: model.index_n_heads.into(),
            index_head_dim: model.index_head_dim.into(),
            index_topk: model.index_topk,
            index_kpool: model.index_kpool,
            selected_k: SELECTED_K,
            max_model_len: parallel.max_model_len,
            cache_block_size: CACHE_BLOCK_SIZE,
            rms_eps: model.rms_norm_eps,
            gpu_name: gpu.clone(),
            bf16_gemm_backends: BF16_GEMM_BACKENDS.to_vec(),
            fp32_gemm_backends: FP32_GEMM_BACKENDS.to_vec(),
            qkv_norm_backends: QKV_NORM_BACKENDS.to_vec(),
            mla_bmm_q_absorb_backends: Q_ABSORB_BACKENDS.to_vec(),
            mla_bmm_v_up_backends: V_UP_BACKENDS.to_vec(),
            mqa_logits_backends: MQA_LOGITS_BACKENDS.to_vec(),
            topk_backends: TOPK_BACKENDS.to_vec(),
            mqa_logits_prefill_backends: MQA_LOGITS_PREFILL_BACKENDS.to_vec(),
            topk_prefill_backends: TOPK_PREFILL_BACKENDS.to_vec(),
            sparse_attention_backends: SPARSE_ATTN_BACKENDS.to_vec(),
            mla_cache_append_backends: MLA_APPEND_BACKENDS.to_vec(),
            index_remap_backends: INDEX_REMAP_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        },
        dense_ffn: mlp(divide("intermediate_size", model.intermediate_size)?),
        shared_expert: mlp(divide(
            "shared expert intermediate",
            model.moe_intermediate_size * model.n_shared_experts,
        )?),
        router: Glm53MoeRouterLocalWorkletConfig {
            hidden: model.hidden.into(),
            num_experts: model.n_routed_experts.into(),
            activation_dtype: ACTIVATION_DTYPE,
            gpu_name: gpu.clone(),
            gemm_backends: ROUTER_GEMM_BACKENDS.to_vec(),
        },
        routed: Glm53RoutedMoeLocalWorkletConfig::split_for_ep(routed_template, demand.clone()),
        // Row gather: reads and writes one hidden row per token.
        embedding: ew(hidden_bytes, hidden_bytes),
        hc_expand: ew(hidden_bytes, stream_bytes),
        mhc: MhcRmsNormKernelConfig {
            backends: MHC_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden_size: model.hidden.into(),
            hc_mult: model.hc_mult,
            hidden_dtype: ACTIVATION_DTYPE,
        },
        all_reduce: AllReduceFusionKernelConfig {
            backends: ALL_REDUCE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            num_gpus: tp,
            hidden_dim: model.hidden,
            dtype: ACTIVATION_DTYPE,
            fabric: Fabric::Nvlink,
        },
        moe_input_glue: ew(hidden_bytes, hidden_bytes),
        moe_combine_glue: ew(2 * hidden_bytes, hidden_bytes),
        hc_contract_mean: ew(stream_bytes, hidden_bytes),
        final_norm: RmsNormKernelConfig {
            backends: RMS_NORM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden: model.hidden.into(),
            dtype: ACTIVATION_DTYPE,
        },
        lm_head: SingleGemmKernelConfig {
            backends: BF16_GEMM_BACKENDS.to_vec(),
            gpu_name: gpu,
            n: Dim::param("vocab_per_rank", divide("vocab_size", model.vocab_size)?),
            k: model.hidden.into(),
            dtype: ACTIVATION_DTYPE,
        },
        model: model.clone(),
        parallel: parallel.clone(),
    })
}

#[derive(Clone, Debug)]
pub struct Glm53FlashVllmResolved {
    pub raw_cfg: Glm53FlashVllmConfigs,
    pub kda: Glm53KdaAttnLocalWorkletResolved,
    pub dsa: Glm53DsaAttnLocalWorkletResolved,
    pub dense_ffn: Glm53Fp8MlpLocalWorkletResolved,
    pub shared_expert: Glm53Fp8MlpLocalWorkletResolved,
    pub routed: Vec<Glm53RoutedMoeLocalWorkletResolved>,
}

pub fn resolve_configs(cfgs: &Glm53FlashVllmConfigs) -> Glm53FlashVllmResolved {
    Glm53FlashVllmResolved {
        kda: Glm53KdaAttnLocalWorklet::resolve_config(&cfgs.kda),
        dsa: Glm53DsaAttnLocalWorklet::resolve_config(&cfgs.dsa),
        dense_ffn: Glm53Fp8MlpLocalWorklet::resolve_config(&cfgs.dense_ffn),
        shared_expert: Glm53Fp8MlpLocalWorklet::resolve_config(&cfgs.shared_expert),
        routed: cfgs
            .routed
            .iter()
            .map(Glm53RoutedMoeLocalWorklet::resolve_config)
            .collect(),
        raw_cfg: cfgs.clone(),
    }
}

enum Boundary {
    Pre(Op<MhcPreRmsNormKernel>),
    Fused(Op<MhcFusedPostPreRmsNormKernel>),
}

impl Boundary {
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        match self {
            Self::Pre(op) => op.compile(builder),
            Self::Fused(op) => op.compile(builder),
        }
    }

    fn eval(&self, num_tokens: u32, ev: &mut Evaluator) {
        let input = MhcRmsNormKernelInput { num_tokens };
        match self {
            Self::Pre(op) => push(op, input, ev),
            Self::Fused(op) => push(op, input, ev),
        }
    }
}

enum Attention {
    Kda(Glm53KdaAttnLocalWorklet),
    Dsa(Glm53DsaAttnLocalWorklet),
}

struct MoeBlock {
    name: String,
    router: Glm53MoeRouterLocalWorklet,
    input_glue: Op<ElementwiseKernel>,
    shared_expert: Glm53Fp8MlpLocalWorklet,
    routed: Vec<Glm53RoutedMoeLocalWorklet>,
    combine_glue: Op<ElementwiseKernel>,
}

impl MoeBlock {
    /// Router, then per EP rank: that rank's shared expert concurrent with its
    /// routed slice (a contended `Max`), plus serial copies of both for
    /// batches too large for the aux stream. One of the two pairs is filled
    /// per iteration; INV-1 fixes the shape.
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let router = self.router.compile(builder);
        let input_glue = self.input_glue.compile(builder);
        let ranks = self
            .routed
            .iter()
            .map(|routed| {
                let concurrent_shared = self.shared_expert.compile(builder);
                let concurrent_routed = routed.compile(builder);
                CostNode::Sum(vec![
                    CostNode::Max {
                        overlap: SHARED_EXPERTS_STREAM_OVERLAP,
                        children: vec![concurrent_shared, concurrent_routed],
                    },
                    routed.compile(builder),
                    self.shared_expert.compile(builder),
                ])
            })
            .collect();
        let combine_glue = self.combine_glue.compile(builder);
        CostNode::Labeled {
            label: format!(
                "{} (MoE) [EP{} ranked ranks; shared expert overlaps routed at T<={}]",
                self.name,
                self.routed.len(),
                SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            ),
            child: Box::new(CostNode::Sum(vec![
                router,
                input_glue,
                CostNode::Labeled {
                    label: format!("{}.ep_ranks (max over EP ranks)", self.name),
                    child: Box::new(CostNode::Max {
                        overlap: 1.0,
                        children: ranks,
                    }),
                },
                combine_glue,
            ])),
        }
    }

    fn eval(&self, num_tokens: u32, ev: &mut Evaluator) {
        let overlapped = num_tokens <= SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD;
        self.router
            .eval(&Glm53MoeRouterLocalWorkletInput { num_tokens }, ev);
        push(&self.input_glue, ElementwiseKernelInput { num_tokens }, ev);
        let shared = Glm53Fp8MlpLocalWorkletInput { num_tokens };
        // Zero tokens pushes zeros: exactly one of the two copies runs.
        let (concurrent_tokens, serial_tokens) = if overlapped {
            (num_tokens, 0)
        } else {
            (0, num_tokens)
        };
        for routed in &self.routed {
            self.shared_expert.eval_or_zero(&shared, !overlapped, ev);
            routed.eval(
                &Glm53RoutedMoeLocalWorkletInput {
                    num_tokens: concurrent_tokens,
                },
                ev,
            );
            routed.eval(
                &Glm53RoutedMoeLocalWorkletInput {
                    num_tokens: serial_tokens,
                },
                ev,
            );
            self.shared_expert.eval_or_zero(&shared, overlapped, ev);
        }
        push(
            &self.combine_glue,
            ElementwiseKernelInput { num_tokens },
            ev,
        );
    }
}

enum Ffn {
    Dense(Glm53Fp8MlpLocalWorklet),
    Moe(MoeBlock),
}

struct LayerGroup {
    label: String,
    layers: Vec<u32>,
    attn_boundary: Boundary,
    attn: Attention,
    attn_all_reduce: Op<AllReduceFusionKernel>,
    ffn_boundary: Op<MhcFusedPostPreRmsNormKernel>,
    ffn: Ffn,
    ffn_all_reduce: Op<AllReduceFusionKernel>,
}

impl LayerGroup {
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        // Compile in eval push order (INV-2): the boundary's leaf precedes
        // the attention leaves.
        let body = CostNode::Sum(vec![
            self.attn_boundary.compile(builder),
            match &self.attn {
                Attention::Kda(worklet) => worklet.compile(builder),
                Attention::Dsa(worklet) => worklet.compile(builder),
            },
            self.attn_all_reduce.compile(builder),
            self.ffn_boundary.compile(builder),
            match &self.ffn {
                Ffn::Dense(worklet) => worklet.compile(builder),
                Ffn::Moe(block) => block.compile(builder),
            },
            self.ffn_all_reduce.compile(builder),
        ]);
        CostNode::Labeled {
            label: format!("{} layers {:?}", self.label, self.layers),
            child: Box::new(CostNode::Scale {
                n: self.layers.len() as u32,
                child: Box::new(body),
            }),
        }
    }

    /// `tokens` is the graph-padded row count every kernel outside the
    /// attention graph break runs on; attention reads the real rows in `batch`.
    fn eval(&self, batch: &NormalizedBatch, tokens: u32, ev: &mut Evaluator) {
        self.attn_boundary.eval(tokens, ev);
        match &self.attn {
            Attention::Kda(worklet) => worklet.eval(&batch.kda, ev),
            Attention::Dsa(worklet) => worklet.eval(&batch.dsa, ev),
        }
        push(
            &self.attn_all_reduce,
            AllReduceFusionKernelInput { num_tokens: tokens },
            ev,
        );
        push(
            &self.ffn_boundary,
            MhcRmsNormKernelInput { num_tokens: tokens },
            ev,
        );
        match &self.ffn {
            Ffn::Dense(worklet) => {
                worklet.eval(&Glm53Fp8MlpLocalWorkletInput { num_tokens: tokens }, ev)
            }
            Ffn::Moe(block) => block.eval(tokens, ev),
        }
        push(
            &self.ffn_all_reduce,
            AllReduceFusionKernelInput { num_tokens: tokens },
            ev,
        );
    }
}

pub struct Glm53FlashVllmModel {
    pub name: String,
    pub tp_size: u16,
    pub max_model_len: u32,
    /// Sorted, deduplicated capture sizes (see `Glm53FlashVllmParallel`).
    cudagraph_capture_sizes: Vec<u32>,
    embedding: Op<ElementwiseKernel>,
    embedding_all_reduce: Op<AllReduceFusionKernel>,
    hc_expand: Op<ElementwiseKernel>,
    groups: Vec<LayerGroup>,
    terminal_post: MhcTerminalPostOp,
    hc_contract_mean: Op<ElementwiseKernel>,
    final_norm: Op<RmsNormKernel>,
    lm_head: Op<SingleGemmKernel>,
    total_kv_bytes_per_token: u64,
    recurrent_state_bytes_per_request: u64,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: Glm53FlashVllmResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<Glm53FlashVllmModel, BuildError> {
    let cfg = &resolved.raw_cfg;
    let n = name.as_str();
    let mut groups = Vec::with_capacity(cfg.groups.len());
    for group in &cfg.groups {
        let prefix = format!("{n}.{}", group.label);
        let p = prefix.as_str();
        let attn_boundary = if group.opens_stream {
            Boundary::Pre(atomic(
                p,
                "attn_mhc_pre",
                cfg.mhc.clone(),
                MhcPreRmsNormKernel::build,
                bridge,
            )?)
        } else {
            Boundary::Fused(atomic(
                p,
                "attn_mhc_post_pre",
                cfg.mhc.clone(),
                MhcFusedPostPreRmsNormKernel::build,
                bridge,
            )?)
        };
        let attn = match group.attn {
            AttnKind::Kda => Attention::Kda(Glm53KdaAttnLocalWorklet::build(
                format!("{p}.kda"),
                resolved.kda.clone(),
                bridge,
            )?),
            AttnKind::Dsa => Attention::Dsa(Glm53DsaAttnLocalWorklet::build(
                format!("{p}.dsa"),
                resolved.dsa.clone(),
                bridge,
            )?),
        };
        let ffn = match group.ffn {
            FfnKind::Dense => Ffn::Dense(Glm53Fp8MlpLocalWorklet::build(
                format!("{p}.dense_ffn"),
                resolved.dense_ffn.clone(),
                bridge,
            )?),
            FfnKind::Moe => {
                let moe = format!("{p}.moe");
                let m = moe.as_str();
                Ffn::Moe(MoeBlock {
                    router: Glm53MoeRouterLocalWorklet::build(
                        format!("{m}.router"),
                        Glm53MoeRouterLocalWorklet::resolve_config(&cfg.router),
                        bridge,
                    )?,
                    input_glue: atomic(
                        m,
                        "input_glue",
                        cfg.moe_input_glue.clone(),
                        ElementwiseKernel::build,
                        bridge,
                    )?,
                    shared_expert: Glm53Fp8MlpLocalWorklet::build(
                        format!("{m}.shared_expert"),
                        resolved.shared_expert.clone(),
                        bridge,
                    )?,
                    routed: resolved
                        .routed
                        .iter()
                        .enumerate()
                        .map(|(rank, routed)| {
                            Glm53RoutedMoeLocalWorklet::build(
                                format!("{m}.routed_rank{rank}"),
                                routed.clone(),
                                bridge,
                            )
                        })
                        .collect::<std::result::Result<_, _>>()?,
                    combine_glue: atomic(
                        m,
                        "combine_glue",
                        cfg.moe_combine_glue.clone(),
                        ElementwiseKernel::build,
                        bridge,
                    )?,
                    name: moe,
                })
            }
        };
        groups.push(LayerGroup {
            label: group.label.clone(),
            layers: group.layers.clone(),
            attn_boundary,
            attn,
            attn_all_reduce: atomic(
                p,
                "attn_all_reduce",
                cfg.all_reduce.clone(),
                AllReduceFusionKernel::build,
                bridge,
            )?,
            ffn_boundary: atomic(
                p,
                "ffn_mhc_post_pre",
                cfg.mhc.clone(),
                MhcFusedPostPreRmsNormKernel::build,
                bridge,
            )?,
            ffn,
            ffn_all_reduce: atomic(
                p,
                "ffn_all_reduce",
                cfg.all_reduce.clone(),
                AllReduceFusionKernel::build,
                bridge,
            )?,
        });
    }
    let model_cfg = &cfg.model;
    let tp = u64::from(cfg.parallel.tp_size);
    // Per rank and DSA layer: the fp8 MLA latent (no rope part) plus the kpool
    // index cache, one 128-byte fp8 key and a 4-byte scale per 4-token pool.
    let dsa_bytes_per_token = u64::from(model_cfg.kv_lora_rank)
        + u64::from(model_cfg.index_head_dim + 4) / u64::from(model_cfg.index_kpool);
    let total_kv_bytes_per_token = tp * u64::from(model_cfg.num_dsa_layers()) * dsa_bytes_per_token;
    let kda_layers = model_cfg.kda_layers.len() as u64;
    let recurrent_state_bytes_per_request = tp
        * kda_layers
        * u64::from(
            resolved.kda.ssm_state_bytes_per_request + resolved.kda.conv_state_bytes_per_request,
        );
    let mut model = Glm53FlashVllmModel {
        embedding: atomic(
            n,
            "embedding",
            cfg.embedding.clone(),
            ElementwiseKernel::build,
            bridge,
        )?,
        embedding_all_reduce: atomic(
            n,
            "embedding_all_reduce",
            cfg.all_reduce.clone(),
            AllReduceFusionKernel::build,
            bridge,
        )?,
        hc_expand: atomic(
            n,
            "hc_expand",
            cfg.hc_expand.clone(),
            ElementwiseKernel::build,
            bridge,
        )?,
        groups,
        terminal_post: MhcTerminalPostOp::build(
            format!("{n}.final_mhc_post"),
            MhcTerminalPostConfig {
                mhc: cfg.mhc.clone(),
                pre_backends: MHC_BACKENDS.to_vec(),
                fused_backends: MHC_BACKENDS.to_vec(),
            },
            bridge,
        )?,
        hc_contract_mean: atomic(
            n,
            "hc_contract_mean",
            cfg.hc_contract_mean.clone(),
            ElementwiseKernel::build,
            bridge,
        )?,
        final_norm: atomic(
            n,
            "final_norm",
            cfg.final_norm.clone(),
            RmsNormKernel::build,
            bridge,
        )?,
        lm_head: atomic(
            n,
            "lm_head",
            cfg.lm_head.clone(),
            SingleGemmKernel::build,
            bridge,
        )?,
        tp_size: cfg.parallel.tp_size,
        max_model_len: cfg.parallel.max_model_len,
        cudagraph_capture_sizes: {
            let mut sizes = cfg.parallel.cudagraph_capture_sizes.clone();
            sizes.sort_unstable();
            sizes.dedup();
            sizes
        },
        total_kv_bytes_per_token,
        recurrent_state_bytes_per_request,
        cost_flat: Vec::new(),
        n_slots: 0,
        name,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

impl Glm53FlashVllmModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let mut children = vec![
            self.embedding.compile(&mut builder),
            self.embedding_all_reduce.compile(&mut builder),
            self.hc_expand.compile(&mut builder),
        ];
        for group in &self.groups {
            children.push(group.compile(&mut builder));
        }
        children.extend([
            self.terminal_post.compile(&mut builder),
            self.hc_contract_mean.compile(&mut builder),
            self.final_norm.compile(&mut builder),
            self.lm_head.compile(&mut builder),
        ]);
        let root = CostNode::Labeled {
            label: format!(
                "{} (Glm53FlashVllmModel) [TP=EP{}; timing_context<={}]",
                self.name, self.tp_size, self.max_model_len
            ),
            child: Box::new(CostNode::Sum(children)),
        };
        builder.finish(root)
    }

    fn eval_into(&self, input: &UnifiedArchInput, ev: &mut Evaluator) {
        let batch = normalize_input(input, self.max_model_len)
            .unwrap_or_else(|reason| panic!("invalid Glm53FlashVllmModel input: {reason}"));
        let tokens = graph_padded_tokens(&self.cudagraph_capture_sizes, batch.total_tokens);
        push(
            &self.embedding,
            ElementwiseKernelInput { num_tokens: tokens },
            ev,
        );
        push(
            &self.embedding_all_reduce,
            AllReduceFusionKernelInput { num_tokens: tokens },
            ev,
        );
        push(
            &self.hc_expand,
            ElementwiseKernelInput { num_tokens: tokens },
            ev,
        );
        for group in &self.groups {
            group.eval(&batch, tokens, ev);
        }
        self.terminal_post
            .eval(&MhcTerminalPostInput { num_tokens: tokens }, ev);
        push(
            &self.hc_contract_mean,
            ElementwiseKernelInput { num_tokens: tokens },
            ev,
        );
        push(&self.final_norm, RmsNormKernelInput { m: tokens }, ev);
        push(
            &self.lm_head,
            SingleGemmKernelInput {
                m: batch.request_count,
            },
            ev,
        );
    }

    fn eval_checked(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: Option<&mut Vec<SlotInput>>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut evaluator = match inputs {
            Some(inputs) => Evaluator::with_inputs(slots, inputs),
            None => Evaluator::new(slots),
        };
        self.eval_into(batch, &mut evaluator);
        assert_eq!(
            evaluator.filled(),
            self.n_slots,
            "eval must fill every compiled slot"
        );
        drop(evaluator);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }
}

impl IterwiseUnifiedModel for Glm53FlashVllmModel {
    fn total_kv_bytes_per_token(&self) -> u64 {
        self.total_kv_bytes_per_token
    }

    fn recurrent_state_bytes_per_request(&self) -> u64 {
        self.recurrent_state_bytes_per_request
    }

    fn recurrent_checkpoint_interval_tokens(&self) -> u32 {
        HYBRID_BLOCK_SIZE
    }

    /// The kpool DSA's top-k cap makes necessary work per-request in context.
    fn logs_decode_kv_lens(&self) -> bool {
        true
    }

    fn gpus_per_replica(&self) -> u16 {
        self.tp_size
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
        self.eval_checked(batch, slots, scratch, None)
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        let total = self.eval_checked(batch, slots, scratch, Some(inputs));
        assert_eq!(
            inputs.len(),
            self.n_slots,
            "slot inputs must align with compiled slots"
        );
        total
    }
}

/// The row count vLLM runs a CUDA-graph replay on: the smallest captured size
/// that holds `tokens`, or `tokens` itself above the largest (eager).
fn graph_padded_tokens(sorted_sizes: &[u32], tokens: u32) -> u32 {
    match sorted_sizes.binary_search(&tokens) {
        Ok(_) => tokens,
        Err(index) => sorted_sizes.get(index).copied().unwrap_or(tokens),
    }
}

/// One iteration, as every layer sees it (one TP group, no attention DP).
struct NormalizedBatch {
    total_tokens: u32,
    request_count: u32,
    kda: Glm53KdaAttnLocalWorkletInput,
    dsa: Glm53DsaAttnLocalWorkletInput,
}

fn normalize_input(
    input: &UnifiedArchInput,
    max_model_len: u32,
) -> std::result::Result<NormalizedBatch, String> {
    let [group] = input.groups.as_slice() else {
        return Err(format!(
            "expected exactly one tensor-parallel attention group, got {}",
            input.groups.len()
        ));
    };
    let mut prefill_tokens = 0_u32;
    for (request, &(prefix, append)) in group.prefill_chunk_pairs.iter().enumerate() {
        if append == 0 {
            return Err(format!("prefill request {request} append must be nonzero"));
        }
        let context = prefix
            .checked_add(append)
            .ok_or_else(|| format!("prefill request {request} context overflows u32"))?;
        if context > max_model_len {
            return Err(format!(
                "prefill request {request} context {context} exceeds timing cap {max_model_len}"
            ));
        }
        prefill_tokens = prefill_tokens
            .checked_add(append)
            .ok_or("prefill token sum overflows u32")?;
    }
    if group.prefill_tokens != prefill_tokens {
        return Err(format!(
            "prefill_tokens {} must equal append sum {prefill_tokens}",
            group.prefill_tokens
        ));
    }
    for (request, &context) in group.decode_kv_lens.iter().enumerate() {
        if !(1..=max_model_len).contains(&context) {
            return Err(format!(
                "decode request {request} context {context} must be in 1..={max_model_len}"
            ));
        }
    }
    let decode_tokens = group.decode_kv_lens.len() as u32;
    if group.decode_tokens != decode_tokens {
        return Err(format!(
            "decode_tokens {} must equal decode_kv_lens length {decode_tokens}",
            group.decode_tokens
        ));
    }
    let total_tokens = prefill_tokens + decode_tokens;
    if group.batch_tokens != total_tokens {
        return Err(format!(
            "batch_tokens {} must equal prefill+decode {total_tokens}",
            group.batch_tokens
        ));
    }
    if total_tokens == 0 {
        return Err("an iteration must carry at least one token".into());
    }
    Ok(NormalizedBatch {
        total_tokens,
        request_count: group.request_count(),
        kda: Glm53KdaAttnLocalWorkletInput {
            prefill_sequence_lengths: group
                .prefill_chunk_pairs
                .iter()
                .map(|&(_, append)| append)
                .collect(),
            decode_batch_size: decode_tokens,
        },
        dsa: Glm53DsaAttnLocalWorkletInput {
            prefill_chunk_pairs: group.prefill_chunk_pairs.clone(),
            decode_kv_lens: group.decode_kv_lens.clone(),
        },
    })
}

fn atomic<K, C, F>(
    prefix: &str,
    suffix: &str,
    config: C,
    build_kernel: F,
    bridge: &PerfApiBridge,
) -> std::result::Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> std::result::Result<K, BuildError>,
{
    let name = format!("{prefix}.{suffix}");
    Ok(Op::new(
        name.clone(),
        std::sync::Arc::new(build_kernel(name, config, bridge)?),
    ))
}

fn push<K>(op: &Op<K>, input: K::Input, ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    let metrics = op.kernel.eval(&input);
    ev.push(metrics, || input.clone().into());
}

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::ArchGroupInput;
    use crate::timing::routing::RoutingDistribution;

    fn model_cfg() -> Glm53FlashModelCfg {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/glm53_flash.json");
        Glm53FlashModelCfg::from_json(&path).unwrap()
    }

    fn parallel() -> Glm53FlashVllmParallel {
        Glm53FlashVllmParallel {
            tp_size: 4,
            max_model_len: 8192,
            gpu_name: "NVIDIA B200".into(),
            cudagraph_capture_sizes: Vec::new(),
        }
    }

    #[test]
    fn graph_padding_rounds_up_to_the_next_captured_size() {
        let sizes = [1, 2, 4, 8, 16, 24];
        let padded: Vec<u32> = [1, 3, 5, 16, 17, 24, 25]
            .iter()
            .map(|&t| graph_padded_tokens(&sizes, t))
            .collect();
        assert_eq!(padded, [1, 4, 8, 16, 24, 24, 25]);
        assert_eq!(graph_padded_tokens(&[], 7), 7);
    }

    fn built() -> Glm53FlashVllmModel {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let demand = ExpertDemand::popularity(&RoutingDistribution::uniform(288), 42);
        let configs = build_configs(&model_cfg(), &parallel(), &demand).unwrap();
        build("unified".into(), resolve_configs(&configs), &bridge).unwrap()
    }

    #[test]
    fn checkpoint_config_parses_to_the_hybrid_schedule() {
        let cfg = model_cfg();
        assert_eq!((cfg.hidden, cfg.num_layers), (4096, 45));
        assert_eq!(cfg.kda_layers.len(), 34);
        assert_eq!(cfg.num_dsa_layers(), 11);
        assert_eq!(cfg.num_moe_layers(), 42);
        assert_eq!(cfg.routed_scaling, (5, 2));
        assert_eq!((cfg.index_topk, cfg.index_kpool), (2048, 4));
    }

    #[test]
    fn layer_groups_fold_to_four_scaled_runs() {
        let groups = layer_groups(&model_cfg());
        let summary: Vec<_> = groups
            .iter()
            .map(|g| (g.label.as_str(), g.layers.len()))
            .collect();
        assert_eq!(
            summary,
            [
                ("first_kda_dense", 1),
                ("kda_dense", 2),
                ("dsa_moe", 11),
                ("kda_moe", 31),
            ]
        );
        assert_eq!(groups[2].layers[..3], [3, 7, 11]);
        // 1 + 2 + 11 + 31 layers, two all-reduces each, plus the embedding.
        let all_reduces: usize = groups.iter().map(|g| 2 * g.layers.len()).sum::<usize>() + 1;
        assert_eq!(all_reduces, 91);
    }

    #[test]
    fn kv_and_recurrent_state_bytes_are_whole_model_totals() {
        let model = built();
        assert_eq!(model.total_kv_bytes_per_token(), 4 * 11 * (512 + 33));
        assert_eq!(
            model.recurrent_state_bytes_per_request(),
            4 * 34 * ((1 << 20) + 6144 * 3 * 2)
        );
        assert_eq!(model.recurrent_checkpoint_interval_tokens(), 2176);
        assert_eq!((model.gpus_per_replica(), model.num_attn_shards()), (4, 4));
    }

    #[test]
    fn compiled_tree_has_a_fixed_slot_count() {
        let model = built();
        // Prologue 3; per group: 2 boundaries + 2 all-reduces + attention
        // (KDA 13, DSA 31) + FFN (dense 5; MoE 2 router + 2 glue + 4 ranks x
        // (concurrent 5 + 2, serial 2 + 5)); epilogue 4.
        let kda_dense = 4 + 13 + 5;
        let dsa_moe = 4 + 31 + 60;
        let kda_moe = 4 + 13 + 60;
        assert_eq!(model.n_slots, 3 + 2 * kda_dense + dsa_moe + kda_moe + 4);
        assert_eq!(model.cost_log_manifest().slots.len(), model.n_slots);
    }

    #[test]
    fn necessary_work_map_covers_the_compiled_locations() {
        use std::collections::BTreeSet;
        let model = built();
        // Kpool DSA work is per-request in context, so the labeler needs each KV length.
        assert!(model.logs_decode_kv_lens());
        let manifest = model.cost_log_manifest();
        let actual: BTreeSet<_> = manifest
            .slots
            .iter()
            .filter(|slot| slot.kind != "all_reduce_fusion")
            .map(|slot| slot.name.as_str())
            .collect();
        let map: serde_json::Value = serde_json::from_str(include_str!(
            "../../../model/work/location_maps/glm53_flash_vllm_fp8_kda_dsa_moe_unified.json"
        ))
        .unwrap();
        assert_eq!(map["arch_types"], serde_json::json!([ARCH_KIND]));
        let mapped: BTreeSet<_> = map["locations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["location"].as_str().unwrap())
            .collect();
        assert_eq!(actual, mapped);
        assert_eq!(mapped.len(), 128);
    }

    fn leaf_order(node: &CostNode, out: &mut Vec<usize>) {
        match node {
            CostNode::Leaf(slot) => out.push(*slot),
            CostNode::Sum(children) | CostNode::Max { children, .. } => {
                children.iter().for_each(|child| leaf_order(child, out))
            }
            CostNode::Scale { child, .. } | CostNode::Labeled { child, .. } => {
                leaf_order(child, out)
            }
        }
    }

    #[test]
    fn leaves_appear_in_slot_order_so_labels_match_eval_pushes() {
        let mut order = Vec::new();
        leaf_order(&built().cost_tree().root, &mut order);
        assert_eq!(order, (0..order.len()).collect::<Vec<_>>());
    }

    #[test]
    fn normalize_splits_the_mixed_capture_shape() {
        let input = UnifiedArchInput {
            groups: vec![ArchGroupInput {
                batch_tokens: 2048,
                prefill_tokens: 2019,
                decode_tokens: 29,
                prefill_chunk_pairs: vec![(0, 2019)],
                decode_kv_lens: vec![3000; 29],
                total_kv_len: 0,
            }],
            tokens_per_source_rank: Vec::new(),
        };
        let batch = normalize_input(&input, 8192).unwrap();
        assert_eq!((batch.total_tokens, batch.request_count), (2048, 30));
        assert_eq!(batch.kda.prefill_sequence_lengths, [2019]);
        assert_eq!(batch.kda.decode_batch_size, 29);
        assert_eq!(batch.dsa.decode_kv_lens.len(), 29);
        let mut bad = input.clone();
        bad.groups[0].decode_kv_lens = vec![9000; 29];
        assert!(normalize_input(&bad, 8192).is_err());
    }
}
