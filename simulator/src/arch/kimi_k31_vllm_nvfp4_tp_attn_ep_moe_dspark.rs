//! Kimi-K3.1-Pro (`KimiK31ForConditionalGeneration`, NVIDIA NVFP4 checkpoint)
//! under the vLLM Kimi-K3 fork with tensor-parallel attention, EP routed
//! experts and the DSpark proposer (`--speculative-config
//! {"method":"dspark","num_speculative_tokens":7}`).
//!
//! Two deployments share this graph, selected by `dp_size`:
//!
//! - **TP16 x DP1** (GB200 NVL72, `kimi_k31_pro_nvfp4_gb200_tp16_ep16`): every
//!   sublayer is tensor-parallel and closes with a TP all-reduce (FlashInfer
//!   MNNVL `twoshotAllreduce`, NCCL above its workspace). The MoE all-reduces
//!   twice: the routed output in the latent width before the up-projection,
//!   and the shared experts' output in the hidden width.
//! - **TP8 x DP2** (two B200 nodes, `kimi_k31_pro_nvfp4_b200x2_tp8dp2_ep16`,
//!   `--all2all-backend allgather_reducescatter`): sequence parallel. The
//!   residual stream holds `T / tp` rows per rank; attention runs between a
//!   TP row all-gather and a TP reduce-scatter (MNNVL Lamport); the MoE
//!   gathers the quantized latent activations of all 16 EP ranks (NCCL
//!   all-gather over InfiniBand), runs the routed experts on every gathered
//!   row, and reduce-scatters the latent output back. Shared experts are
//!   replicated on each rank's own rows.
//!
//! The 93 layers (`linear_attn_config`, `dsa_config`, 1-based in the
//! checkpoint): 69 KDA layers and 24 MLA+DSA layers (0-based 3, 7, ..., 91,
//! 92); layer 0 carries the dense BF16 FFN and layers 1..92 the latent MoE
//! (896 experts, top-16, routed in the 3584-wide latent space between a BF16
//! down-projection and an RMSNorm + BF16 up-projection). Each sublayer opens
//! with the block attention-residual combine (`attn_res_fwd_online_v2`).
//!
//! ```text
//! prologue      embedding, TP boundary
//! layer 0       attn_res + KDA + TP,  attn_res + dense FFN + TP        (x1)
//! KDA + MoE     attn_res + KDA + TP,  attn_res + MoE                   (x68)
//! MLA + MoE     attn_res + MLA + TP,  attn_res + MoE                   (x24)
//! epilogue      attn_res, final norm, lm_head over the logits rows
//! draft         DSpark: fc + context K/V, 5 SWA layers, head, 7 Markov steps
//! ```
//!
//! The MoE at most [`SHARED_STREAM_TOKENS`] rows per rank runs the shared
//! experts on the main stream concurrent with the router, latent
//! down-projection and routed experts on the aux stream (a `Max`); above it
//! the two run serially. Both copies are compiled; the inactive one is zero.
//!
//! v1 proxies (named at their leaves): the MXFP4 indexer prices on the FP8
//! DeepGEMM rows, the packed conv prefill on `causal_conv1d_fn`, SiTU on
//! `silu_and_mul`, and attn_res / glue launches are byte-sized elementwise
//! placeholders. The fused GEMM + all-reduce o_proj of the prefill path is
//! priced as GEMM then all-reduce; the RecoverSSM commit and the MLA
//! prefill's DCP shard/unshard are not modeled.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde_json::Value;

use crate::arch::contract::{SpeculativeArchInput, SpeculativeUnifiedModel};
use crate::arch::kv_layout::{EngineOverhead, KvCacheLayout, KvCachePart};
use crate::common::Fabric;
use crate::op::spec::{DsparkMarkovStepConfig, DsparkMarkovStepInput, DsparkMarkovStepOp};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{
    AllReduceFusionKernel, AllReduceFusionKernelConfig, AllReduceFusionKernelInput,
    AllReduceFusionSpec, AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput,
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, FlashinferAttnRectKernel,
    FlashinferAttnRectKernelConfig, FlashinferAttnRectKernelInput, MoeEpCollectiveKernelInput,
    MoeEpQuantizedAllGatherKernel, MoeEpQuantizedAllGatherKernelConfig, MoeEpReduceScatterKernel,
    MoeEpReduceScatterKernelConfig, Nvfp4FusedMoeKernel, Nvfp4FusedMoeKernelConfig,
    Nvfp4FusedMoeKernelInput, Nvfp4QuantKernel, Nvfp4QuantKernelConfig, Nvfp4QuantKernelInput,
    RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput, RowAllGatherKernel,
    RowAllGatherKernelConfig, SiluAndMulKernel, SiluAndMulKernelConfig, SiluAndMulKernelInput,
    SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Dim, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, Probe, SlotInput,
};
use crate::worklet::{
    Glm53MoeRouterLocalWorklet, Glm53MoeRouterLocalWorkletConfig, Glm53MoeRouterLocalWorkletInput,
    Glm53RoutedMoeLocalWorklet, Glm53RoutedMoeLocalWorkletConfig, Glm53WeightPrecision,
    KimiAttnInput, KimiK31Backends, KimiKdaAttnLocal, KimiKdaAttnLocalConfig, KimiMlaAttnLocal,
    KimiMlaAttnLocalConfig, KimiMlpLocal, KimiMlpLocalConfig,
};

const ARCH_KIND: &str = "kimi_k31_vllm_nvfp4_tp_attn_ep_moe_dspark";

/// Rows per rank up to which the shared experts overlap the routed path on the
/// aux stream (vLLM `shared_experts.py`, as in the GLM-5.3 graphs).
pub const SHARED_STREAM_TOKENS: u32 = 256;
/// The `Max` overlap of the two MoE streams: the shorter stream hides fully.
const STREAM_OVERLAP: f32 = 1.0;
/// Token alignment of vLLM's packed hybrid block row: the 64-token kernel page
/// of the sparse MLA / indexer caches and the KDA prefill checkpoint grid.
const HYBRID_BLOCK_ALIGNMENT_TOKENS: u64 = 64;
/// The TP the packed block is rounded at (`dsa_cache._PD_REFERENCE_TP`), so
/// block sizes of different TPs stay divisible.
const HYBRID_BLOCK_REFERENCE_TP: u64 = 16;
/// MLA latent cache page (fp8 `kv_lora_rank + rope`) and the MXFP4 indexer
/// key with its scales and pad, bytes per token per MLA layer. 646 is
/// calibrated against the B200 capture's `Extended KV cache to 1482 blocks
/// (32.87 GiB)` over 24-layer blocks.
pub const MLA_BYTES_PER_TOKEN_PER_LAYER: u64 = 646;
const NVFP4_GROUP: u32 = 16;
const NVFP4_SCALE_FORMAT: &str = "linear_e4m3";
const ROUTING_METHOD: &str = "deepseek_v3";

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

// ---------------------------------------------------------------------------
// Checkpoint configs

#[derive(Clone, Debug)]
pub struct KimiK31ModelCfg {
    pub hidden: u32,
    pub num_layers: u32,
    pub vocab_size: u32,
    pub intermediate_size: u32,
    pub first_k_dense_replace: u32,
    pub kda_num_heads: u32,
    pub kda_head_dim: u32,
    pub conv_kernel_size: u32,
    /// 0-based MLA + DSA layers.
    pub mla_layers: Vec<u32>,
    pub num_attention_heads: u32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub index_num_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub num_experts: u32,
    pub experts_per_token: u32,
    pub moe_intermediate_size: u32,
    pub latent_size: u32,
    pub num_shared_experts: u32,
    /// The routed experts' GEMM1 activation as the profiler's
    /// `gated_activation` arg spells it: `hidden_act = "situ"` gives
    /// `situ:beta=<activation_situ_beta>,linear_beta=<activation_situ_linear_beta>`,
    /// what vLLM's TRT-LLM NVFP4 experts pass; `"silu"` leaves it unset (SwiGLU).
    pub routed_gated_activation: Option<String>,
}

/// `gated_activation` for the routed experts: vLLM's `TrtLlmNvFp4ExpertsBase`
/// maps `hidden_act = "situ"` to FlashInfer `ActivationType.Situ` with
/// `gemm1_alpha = activation_situ_beta`, `gemm1_beta = activation_situ_linear_beta`
/// (both asserted > 0), and SiLU to the default SwiGLU.
fn routed_gated_activation(cfg: &Value) -> Result<Option<String>> {
    match cfg.get("hidden_act").and_then(Value::as_str).unwrap_or("silu") {
        "silu" => Ok(None),
        "situ" => {
            let f = |key: &str| -> Result<f64> {
                let v = cfg
                    .get(key)
                    .and_then(Value::as_f64)
                    .with_context(|| format!("SiTU needs a numeric `{key}`"))?;
                ensure!(v > 0.0, "`{key}` must be > 0 for the TRT-LLM SiTuGlu kernel");
                Ok(v)
            };
            Ok(Some(format!(
                "situ:beta={},linear_beta={}",
                f("activation_situ_beta")?,
                f("activation_situ_linear_beta")?
            )))
        }
        other => anyhow::bail!("Kimi-K3.1 hidden_act {other:?}: only silu and situ are supported"),
    }
}

impl KimiK31ModelCfg {
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading Kimi-K3.1 config {}", path.display()))?;
        Self::from_value(&serde_json::from_str(&text)?)
    }

    pub fn from_value(root: &Value) -> Result<Self> {
        let cfg = root.get("text_config").unwrap_or(root);
        let u = |obj: &Value, key: &str| -> Result<u32> {
            let v = obj
                .get(key)
                .and_then(Value::as_u64)
                .with_context(|| format!("Kimi-K3.1 config field `{key}` must be an integer"))?;
            Ok(u32::try_from(v)?)
        };
        let model_type = cfg.get("model_type").and_then(Value::as_str).unwrap_or("");
        ensure!(
            model_type == "kimi_linear",
            "Kimi-K3.1 arches need a kimi_linear text config, got {model_type:?}"
        );
        let linear = cfg.get("linear_attn_config").context("needs linear_attn_config")?;
        let dsa = cfg.get("dsa_config").context("needs dsa_config")?;
        let num_layers = u(cfg, "num_hidden_layers")?;
        let one_based = |obj: &Value, key: &str| -> Result<Vec<u32>> {
            obj.get(key)
                .and_then(Value::as_array)
                .with_context(|| format!("`{key}` must be a list"))?
                .iter()
                .map(|v| {
                    let layer = v.as_u64().context("layer ids are integers")? as u32;
                    ensure!((1..=num_layers).contains(&layer), "layer {layer} out of range");
                    Ok(layer - 1)
                })
                .collect()
        };
        let mla_layers = one_based(linear, "full_attn_layers")?;
        let kda_layers = one_based(linear, "kda_layers")?;
        ensure!(
            mla_layers == one_based(dsa, "dsa_layers")?,
            "every full-attention layer must be a DSA layer"
        );
        ensure!(
            mla_layers.len() + kda_layers.len() == num_layers as usize,
            "KDA and MLA layers must cover the model"
        );
        ensure!(
            cfg.get("mla_use_nope").and_then(Value::as_bool) == Some(true)
                && u(dsa, "index_rope_dim")? == 0,
            "the modeled MLA is the NoPE variant with a rope-free indexer"
        );
        let model = Self {
            hidden: u(cfg, "hidden_size")?,
            num_layers,
            vocab_size: u(cfg, "vocab_size")?,
            intermediate_size: u(cfg, "intermediate_size")?,
            first_k_dense_replace: u(cfg, "first_k_dense_replace")?,
            kda_num_heads: u(linear, "num_heads")?,
            kda_head_dim: u(linear, "head_dim")?,
            conv_kernel_size: u(linear, "short_conv_kernel_size")?,
            mla_layers,
            num_attention_heads: u(cfg, "num_attention_heads")?,
            q_lora_rank: u(cfg, "q_lora_rank")?,
            kv_lora_rank: u(cfg, "kv_lora_rank")?,
            qk_nope_head_dim: u(cfg, "qk_nope_head_dim")?,
            qk_rope_head_dim: u(cfg, "qk_rope_head_dim")?,
            v_head_dim: u(cfg, "v_head_dim")?,
            index_num_heads: u(dsa, "index_num_heads")?,
            index_head_dim: u(dsa, "index_head_dim")?,
            index_topk: u(dsa, "index_topk")?,
            num_experts: u(cfg, "num_experts")?,
            experts_per_token: u(cfg, "num_experts_per_token")?,
            moe_intermediate_size: u(cfg, "moe_intermediate_size")?,
            latent_size: u(cfg, "routed_expert_hidden_size")?,
            num_shared_experts: u(cfg, "num_shared_experts")?,
            routed_gated_activation: routed_gated_activation(cfg)?,
        };
        ensure!(
            model.first_k_dense_replace == 1 && !model.mla_layers.contains(&0),
            "the modeled schedule has one dense KDA layer 0"
        );
        Ok(model)
    }

    pub fn num_moe_layers(&self) -> u32 {
        self.num_layers - self.first_k_dense_replace
    }

    fn kda_moe_layers(&self) -> Vec<u32> {
        (self.first_k_dense_replace..self.num_layers)
            .filter(|l| !self.mla_layers.contains(l))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct KimiDsparkCfg {
    pub num_layers: u32,
    pub hidden: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub sliding_window: u32,
    pub block_size: u32,
    pub num_aux_layers: u32,
    pub markov_rank: u32,
    pub vocab_size: u32,
}

impl KimiDsparkCfg {
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading DSpark config {}", path.display()))?;
        let root: Value = serde_json::from_str(&text)?;
        let layer = root
            .get("transformer_layer_config")
            .context("DSpark config needs transformer_layer_config")?;
        let u = |obj: &Value, key: &str| -> Result<u32> {
            Ok(obj
                .get(key)
                .and_then(Value::as_u64)
                .with_context(|| format!("DSpark config field `{key}` must be an integer"))?
                as u32)
        };
        Ok(Self {
            num_layers: u(layer, "num_hidden_layers")?,
            hidden: u(layer, "hidden_size")?,
            num_heads: u(layer, "num_attention_heads")?,
            num_kv_heads: u(layer, "num_key_value_heads")?,
            head_dim: u(layer, "head_dim")?,
            intermediate_size: u(layer, "intermediate_size")?,
            sliding_window: u(layer, "sliding_window")?,
            block_size: u(&root, "block_size")?,
            num_aux_layers: root
                .get("aux_hidden_state_layer_ids")
                .and_then(Value::as_array)
                .context("aux_hidden_state_layer_ids")?
                .len() as u32,
            markov_rank: u(&root, "markov_rank")?,
            vocab_size: u(&root, "draft_vocab_size")?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct KimiK31Parallel {
    pub tp_size: u16,
    pub dp_size: u16,
    pub gpu_name: String,
    /// Fabric of the EP exchange when `dp_size > 1` (the two B200 nodes:
    /// InfiniBand).
    pub ep_fabric: Fabric,
    pub max_model_len: u32,
    /// Token sizes of the captured graphs (piecewise prefill and FULL decode
    /// share one token grid here).
    pub cudagraph_capture_sizes: Vec<u32>,
    pub draft_tokens: u32,
    pub backends: KimiK31Backends,
}

impl KimiK31Parallel {
    fn ep_size(&self) -> u16 {
        self.tp_size * self.dp_size
    }

    /// The most rows one step runs: the largest captured graph, at least 4096.
    fn max_step_rows(&self) -> u32 {
        self.cudagraph_capture_sizes.iter().copied().max().unwrap_or(4096).max(4096)
    }

    fn sequence_parallel(&self) -> bool {
        self.dp_size > 1
    }
}

// ---------------------------------------------------------------------------
// Leaves shared by both deployments

fn ew(b: &KimiK31Backends, gpu: &str, input: u32, output: u32) -> ElementwiseKernelConfig {
    ElementwiseKernelConfig {
        backends: b.elementwise.clone(),
        gpu_name: gpu.to_string(),
        input_bytes_per_token: input.into(),
        output_bytes_per_token: output.into(),
    }
}

fn gemm(backends: &[&'static str], gpu: &str, n: u32, k: u32, dtype: DType) -> SingleGemmKernelConfig {
    SingleGemmKernelConfig {
        backends: backends.to_vec(),
        gpu_name: gpu.to_string(),
        n: n.into(),
        k: k.into(),
        dtype,
    }
}

fn norm(b: &KimiK31Backends, gpu: &str, hidden: u32) -> RmsNormKernelConfig {
    RmsNormKernelConfig {
        backends: b.rms_norm.clone(),
        gpu_name: gpu.to_string(),
        hidden: hidden.into(),
        dtype: DType::Bf16,
    }
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
    Ok(Op::new(name.clone(), std::sync::Arc::new(build_kernel(name, config, bridge)?)))
}

fn push<K>(op: &Op<K>, input: K::Input, zero: bool, ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    let metrics = if zero { LeafMetrics::ZERO } else { op.kernel.eval(&input) };
    ev.push(metrics, || input.clone().into());
}

fn tokens(n: u32) -> ElementwiseKernelInput {
    ElementwiseKernelInput { num_tokens: n }
}

fn rows(m: u32) -> SingleGemmKernelInput {
    SingleGemmKernelInput { m }
}

/// One TP all-reduce: FlashInfer MNNVL while the rows fit its workspace, NCCL
/// above it (vLLM `CudaCommunicator.all_reduce`). The NCCL slot exists only
/// when a step can outgrow the workspace (not at TP16's 64 MB budget).
struct TpAllReduce {
    fused: Op<AllReduceFusionKernel>,
    large: Option<Op<AllReduceKernel>>,
    max_fused_tokens: u32,
    bytes_per_token: u64,
}

impl TpAllReduce {
    fn build(
        prefix: &str,
        suffix: &str,
        parallel: &KimiK31Parallel,
        hidden: u32,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        let fused = AllReduceFusionKernelConfig {
            backends: vec!["flashinfer_mnnvl"],
            gpu_name: parallel.gpu_name.clone(),
            num_gpus: u32::from(parallel.tp_size),
            hidden_dim: hidden,
            dtype: DType::Bf16,
            fabric: Fabric::Nvlink,
            fused_token_limit: None,
        };
        let max_fused_tokens = AllReduceFusionSpec::max_fused_tokens(&fused);
        let max_rows = parallel.max_step_rows();
        Ok(Self {
            fused: atomic(prefix, suffix, fused, AllReduceFusionKernel::build, bridge)?,
            large: if max_rows <= max_fused_tokens { None } else { Some(atomic(
                prefix,
                &format!("{suffix}_large"),
                AllReduceKernelConfig {
                    backends: vec!["nccl"],
                    gpu_name: parallel.gpu_name.clone(),
                    num_gpus: u32::from(parallel.tp_size),
                    fabric: Fabric::Nvlink,
                },
                AllReduceKernel::build,
                bridge,
            )?) },
            max_fused_tokens,
            bytes_per_token: u64::from(hidden) * 2,
        })
    }

    fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        let mut children = vec![self.fused.compile(b)];
        children.extend(self.large.as_ref().map(|op| op.compile(b)));
        CostNode::Sum(children)
    }

    fn eval(&self, t: u32, ev: &mut Evaluator) {
        let fused = t <= self.max_fused_tokens;
        push(&self.fused, AllReduceFusionKernelInput { num_tokens: t }, t == 0 || !fused, ev);
        if let Some(large) = &self.large {
            push(
                large,
                AllReduceKernelInput { message_size_bytes: u64::from(t) * self.bytes_per_token },
                t == 0 || fused,
                ev,
            );
        } else {
            debug_assert!(fused, "{t} rows outgrow the all-reduce workspace");
        }
    }
}

/// The sequence-parallel attention boundary: TP row all-gather in, TP
/// reduce-scatter out (MNNVL Lamport, `max_total_tokens` from the graph cap).
struct SpBoundary {
    gather: Op<RowAllGatherKernel>,
    scatter: Op<MoeEpReduceScatterKernel>,
    tp: u32,
}

impl SpBoundary {
    fn build(
        prefix: &str,
        parallel: &KimiK31Parallel,
        hidden: u32,
        max_total_tokens: u32,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        let tp = u32::from(parallel.tp_size);
        Ok(Self {
            gather: atomic(
                prefix,
                "sp_all_gather",
                // The capture gathers with NCCL (`ncclDevKernel_AllGather_RING`).
                RowAllGatherKernelConfig {
                    backends: vec!["vllm_pynccl"],
                    gpu_name: parallel.gpu_name.clone(),
                    num_gpus: tp,
                    hidden_size: hidden.into(),
                    dtype: DType::Bf16,
                    fabric: Fabric::Nvlink.as_str().to_string(),
                    max_total_tokens,
                },
                RowAllGatherKernel::build,
                bridge,
            )?,
            scatter: atomic(
                prefix,
                "sp_reduce_scatter",
                MoeEpReduceScatterKernelConfig {
                    backends: vec!["mnnvl_lamport"],
                    gpu_name: parallel.gpu_name.clone(),
                    num_gpus: tp,
                    hidden_size: hidden.into(),
                    dtype: DType::Bf16,
                    fabric: Fabric::Nvlink.as_str().to_string(),
                    max_total_tokens,
                    extra_total_tokens: Vec::new(),
                },
                MoeEpReduceScatterKernel::build,
                bridge,
            )?,
            tp,
        })
    }

    fn input(&self, total: u32) -> MoeEpCollectiveKernelInput {
        MoeEpCollectiveKernelInput { per_rank_tokens: vec![total / self.tp; self.tp as usize] }
    }
}

/// The TP boundary closing a sublayer: an all-reduce (TP mode) or nothing
/// here (sequence parallel closes attention in [`SpBoundary`] and the MoE in
/// its reduce-scatter).
enum Boundary {
    AllReduce(TpAllReduce),
    Sp(SpBoundary),
}

// ---------------------------------------------------------------------------
// MoE

struct MoeBlock {
    name: String,
    router: Glm53MoeRouterLocalWorklet,
    /// `single_group_topk` (sequence parallel; inside the fused MoE in TP).
    topk: Option<Op<ElementwiseKernel>>,
    shared: KimiMlpLocal,
    latent_down: Op<SingleGemmKernel>,
    quant: Op<Nvfp4QuantKernel>,
    gather: Option<Op<MoeEpQuantizedAllGatherKernel>>,
    routed: Vec<Op<Nvfp4FusedMoeKernel>>,
    scatter: Option<Op<MoeEpReduceScatterKernel>>,
    latent_all_reduce: Option<TpAllReduce>,
    latent_norm: Op<RmsNormKernel>,
    up: Op<SingleGemmKernel>,
    shared_all_reduce: Option<TpAllReduce>,
    add: Op<ElementwiseKernel>,
    ep: u32,
}

impl MoeBlock {
    fn compile_aux(&self, b: &mut CostTreeBuilder) -> CostNode {
        let mut aux = vec![self.router.compile(b)];
        aux.extend(self.topk.as_ref().map(|op| op.compile(b)));
        aux.push(self.latent_down.compile(b));
        aux.push(self.quant.compile(b));
        aux.extend(self.gather.as_ref().map(|op| op.compile(b)));
        let ranks = self.routed.iter().map(|op| op.compile(b)).collect();
        aux.push(CostNode::Labeled {
            label: format!("{}.routed (max over EP{} ranks)", self.name, self.ep),
            child: Box::new(CostNode::Max { overlap: 1.0, children: ranks }),
        });
        aux.extend(self.scatter.as_ref().map(|op| op.compile(b)));
        CostNode::Sum(aux)
    }

    fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        let shared_c = self.shared.compile(b);
        let aux_c = self.compile_aux(b);
        let concurrent = CostNode::Labeled {
            label: format!("{}.streams (shared || routed, T<={SHARED_STREAM_TOKENS})", self.name),
            child: Box::new(CostNode::Max { overlap: STREAM_OVERLAP, children: vec![shared_c, aux_c] }),
        };
        let shared_s = self.shared.compile(b);
        let aux_s = self.compile_aux(b);
        let mut children = vec![concurrent, CostNode::Sum(vec![shared_s, aux_s])];
        children.extend(self.latent_all_reduce.as_ref().map(|op| op.compile(b)));
        children.push(self.latent_norm.compile(b));
        children.push(self.up.compile(b));
        children.extend(self.shared_all_reduce.as_ref().map(|op| op.compile(b)));
        children.push(self.add.compile(b));
        CostNode::Labeled {
            label: format!("{} (latent MoE, EP{})", self.name, self.ep),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    fn eval_aux(&self, local: u32, routed_rows: u32, zero: bool, ev: &mut Evaluator) {
        let l = if zero { 0 } else { local };
        self.router.eval(&Glm53MoeRouterLocalWorkletInput { num_tokens: l }, ev);
        if let Some(op) = &self.topk {
            push(op, tokens(local), zero, ev);
        }
        push(&self.latent_down, rows(local), zero, ev);
        push(&self.quant, Nvfp4QuantKernelInput { num_tokens: local }, zero, ev);
        let collective = MoeEpCollectiveKernelInput { per_rank_tokens: vec![local; self.ep as usize] };
        if let Some(op) = &self.gather {
            push(op, collective.clone(), zero, ev);
        }
        for op in &self.routed {
            push(op, Nvfp4FusedMoeKernelInput { num_tokens: routed_rows }, zero, ev);
        }
        if let Some(op) = &self.scatter {
            push(op, collective, zero, ev);
        }
    }

    /// `local` rows on this rank; the routed experts see `routed_rows`.
    fn eval(&self, local: u32, routed_rows: u32, ev: &mut Evaluator) {
        let overlapped = local <= SHARED_STREAM_TOKENS;
        self.shared.eval(if overlapped { local } else { 0 }, ev);
        self.eval_aux(local, routed_rows, !overlapped, ev);
        self.shared.eval(if overlapped { 0 } else { local }, ev);
        self.eval_aux(local, routed_rows, overlapped, ev);
        if let Some(ar) = &self.latent_all_reduce {
            ar.eval(local, ev);
        }
        push(&self.latent_norm, RmsNormKernelInput { m: local }, false, ev);
        push(&self.up, rows(local), false, ev);
        if let Some(ar) = &self.shared_all_reduce {
            ar.eval(local, ev);
        }
        push(&self.add, tokens(local), false, ev);
    }
}

// ---------------------------------------------------------------------------
// Layers

enum Attention {
    Kda(KimiKdaAttnLocal),
    Mla(KimiMlaAttnLocal),
}

impl Attention {
    fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        match self {
            Self::Kda(w) => w.compile(b),
            Self::Mla(w) => w.compile(b),
        }
    }

    fn eval(&self, input: &KimiAttnInput, ev: &mut Evaluator) {
        match self {
            Self::Kda(w) => w.eval(input, ev),
            Self::Mla(w) => w.eval(input, ev),
        }
    }
}

enum Ffn {
    Dense(KimiMlpLocal, Option<TpAllReduce>),
    Moe(MoeBlock),
}

struct LayerGroup {
    label: String,
    layers: Vec<u32>,
    attn_res: Op<ElementwiseKernel>,
    attn: Attention,
    boundary: Boundary,
    ffn_res: Op<ElementwiseKernel>,
    ffn: Ffn,
    dp: usize,
}

impl LayerGroup {
    fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        let mut children = vec![self.attn_res.compile(b)];
        match &self.boundary {
            Boundary::AllReduce(ar) => {
                children.push(self.attn.compile(b));
                children.push(ar.compile(b));
            }
            Boundary::Sp(sp) => {
                children.push(sp.gather.compile(b));
                let groups = (0..self.dp).map(|_| self.attn.compile(b)).collect();
                children.push(CostNode::Labeled {
                    label: format!("{}.attention (max over {} DP groups)", self.label, self.dp),
                    child: Box::new(CostNode::Max { overlap: 1.0, children: groups }),
                });
                children.push(sp.scatter.compile(b));
            }
        }
        children.push(self.ffn_res.compile(b));
        match &self.ffn {
            Ffn::Dense(mlp, ar) => {
                children.push(mlp.compile(b));
                children.extend(ar.as_ref().map(|op| op.compile(b)));
            }
            Ffn::Moe(moe) => children.push(moe.compile(b)),
        }
        CostNode::Labeled {
            label: format!("{} layers {:?}", self.label, self.layers),
            child: Box::new(CostNode::Scale {
                n: self.layers.len() as u32,
                child: Box::new(CostNode::Sum(children)),
            }),
        }
    }

    fn eval(&self, step: &Step, ev: &mut Evaluator) {
        let local = step.local_rows;
        push(&self.attn_res, tokens(local), false, ev);
        match &self.boundary {
            Boundary::AllReduce(ar) => {
                self.attn.eval(&step.groups[0], ev);
                ar.eval(local, ev);
            }
            Boundary::Sp(sp) => {
                push(&sp.gather, sp.input(step.group_rows), false, ev);
                for group in &step.groups {
                    self.attn.eval(group, ev);
                }
                push(&sp.scatter, sp.input(step.group_rows), false, ev);
            }
        }
        push(&self.ffn_res, tokens(local), false, ev);
        match &self.ffn {
            Ffn::Dense(mlp, ar) => {
                mlp.eval(local, ev);
                if let Some(ar) = ar {
                    ar.eval(local, ev);
                }
            }
            Ffn::Moe(moe) => moe.eval(local, step.routed_rows, ev),
        }
    }
}

// ---------------------------------------------------------------------------
// DSpark draft (one per DP group, TP over the group)

struct DraftLayer {
    in_norm: Op<RmsNormKernel>,
    ctx_kv: Op<SingleGemmKernel>,
    ctx_kv_store: Op<ElementwiseKernel>,
    qkv: Op<SingleGemmKernel>,
    qk_norm_rope: Op<ElementwiseKernel>,
    attention: Op<FlashinferAttnRectKernel>,
    o_proj: Op<SingleGemmKernel>,
    attn_ar: TpAllReduce,
    post_norm: Op<RmsNormKernel>,
    gate_up: Op<SingleGemmKernel>,
    act: Op<SiluAndMulKernel>,
    down: Op<SingleGemmKernel>,
    mlp_ar: TpAllReduce,
}

struct DraftStage {
    num_layers: u32,
    block: u32,
    window: u32,
    draft_tokens: u32,
    fc: Op<SingleGemmKernel>,
    embed: Op<ElementwiseKernel>,
    embed_ar: TpAllReduce,
    layer: DraftLayer,
    final_norm: Op<RmsNormKernel>,
    lm_head: Op<SingleGemmKernel>,
    markov: DsparkMarkovStepOp,
}

/// One DP group's draft input.
#[derive(Clone, Debug, Default)]
struct DraftStep {
    requests: u32,
    context_rows: u32,
    mean_context: u32,
}

impl DraftStage {
    fn build(
        name: &str,
        d: &KimiDsparkCfg,
        parallel: &KimiK31Parallel,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        let tp = u32::from(parallel.tp_size);
        let g = parallel.gpu_name.as_str();
        let b = &parallel.backends;
        let bf16 = DType::Bf16;
        let p = format!("{name}.draft");
        let p = p.as_str();
        let lp = format!("{p}.layer");
        let l = lp.as_str();
        let div = |what: &str, v: u32| {
            if v % tp != 0 {
                return Err(fit_failed(format!("draft {what} {v} must divide by TP {tp}")));
            }
            Ok(v / tp)
        };
        let heads = div("heads", d.num_heads)?;
        let kv_heads = div("kv heads", d.num_kv_heads)?;
        let inter = div("intermediate", d.intermediate_size)?;
        let hd = d.head_dim;
        let layer = DraftLayer {
            in_norm: atomic(l, "input_norm", norm(b, g, d.hidden), RmsNormKernel::build, bridge)?,
            ctx_kv: atomic(l, "context_kv", gemm(&b.bf16_gemm, g, 2 * kv_heads * hd, d.hidden, bf16), SingleGemmKernel::build, bridge)?,
            ctx_kv_store: atomic(l, "context_kv_store", ew(b, g, 4 * kv_heads * hd, 2 * kv_heads * hd), ElementwiseKernel::build, bridge)?,
            qkv: atomic(l, "qkv_proj", gemm(&b.bf16_gemm, g, (heads + 2 * kv_heads) * hd, d.hidden, bf16), SingleGemmKernel::build, bridge)?,
            qk_norm_rope: atomic(l, "qk_norm_rope", ew(b, g, 2 * (heads + kv_heads) * hd, 2 * (heads + kv_heads) * hd), ElementwiseKernel::build, bridge)?,
            attention: atomic(
                l,
                "attention",
                FlashinferAttnRectKernelConfig {
                    backends: vec!["vllm_fa4_split32"],
                    gpu_name: g.to_string(),
                    num_qo_heads: heads.into(),
                    num_kv_heads: kv_heads.into(),
                    head_dim: hd.into(),
                    q_dtype: DType::Fp8E4m3,
                    kv_dtype: DType::Fp8E4m3,
                    o_dtype: bf16,
                },
                FlashinferAttnRectKernel::build,
                bridge,
            )?,
            o_proj: atomic(l, "o_proj", gemm(&b.bf16_gemm, g, d.hidden, heads * hd, bf16), SingleGemmKernel::build, bridge)?,
            attn_ar: TpAllReduce::build(l, "attn_all_reduce", parallel, d.hidden, bridge)?,
            post_norm: atomic(l, "post_norm", norm(b, g, d.hidden), RmsNormKernel::build, bridge)?,
            gate_up: atomic(l, "gate_up", gemm(&b.bf16_gemm, g, 2 * inter, d.hidden, bf16), SingleGemmKernel::build, bridge)?,
            act: atomic(
                l,
                "silu_and_mul",
                SiluAndMulKernelConfig {
                    backends: b.act.clone(),
                    gpu_name: g.to_string(),
                    hidden_dim: inter.into(),
                    dtype: bf16,
                },
                SiluAndMulKernel::build,
                bridge,
            )?,
            down: atomic(l, "down", gemm(&b.bf16_gemm, g, d.hidden, inter, bf16), SingleGemmKernel::build, bridge)?,
            mlp_ar: TpAllReduce::build(l, "mlp_all_reduce", parallel, d.hidden, bridge)?,
        };
        let vocab = div("vocab", d.vocab_size)?;
        Ok(Self {
            num_layers: d.num_layers,
            block: d.block_size,
            window: d.sliding_window,
            draft_tokens: parallel.draft_tokens,
            // The aux hidden states of `num_aux_layers` target layers fold
            // to one hidden row (replicated weight).
            fc: atomic(p, "fc", gemm(&b.bf16_gemm, g, d.hidden, d.num_aux_layers * d.hidden, bf16), SingleGemmKernel::build, bridge)?,
            embed: atomic(p, "mask_embedding", ew(b, g, 2 * d.hidden, 2 * d.hidden), ElementwiseKernel::build, bridge)?,
            embed_ar: TpAllReduce::build(p, "embedding_all_reduce", parallel, d.hidden, bridge)?,
            layer,
            final_norm: atomic(p, "final_norm", norm(b, g, d.hidden), RmsNormKernel::build, bridge)?,
            lm_head: atomic(p, "lm_head", gemm(&b.bf16_gemm, g, vocab, d.hidden, bf16), SingleGemmKernel::build, bridge)?,
            markov: DsparkMarkovStepOp::build(
                format!("{p}.markov"),
                DsparkMarkovStepConfig {
                    gpu_name: g.to_string(),
                    vocab_size: d.vocab_size,
                    markov_rank: d.markov_rank,
                    gemm_backends: b.bf16_gemm.clone(),
                    elementwise_backends: b.elementwise.clone(),
                },
                bridge,
            )?,
        })
    }

    fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        // Slots are numbered in compile order and `eval` fills them in push
        // order, so the prologue compiles before the layer body.
        let fc = self.fc.compile(b);
        let embed = self.embed.compile(b);
        let embed_ar = self.embed_ar.compile(b);
        let l = &self.layer;
        let layer = CostNode::Sum(vec![
            l.in_norm.compile(b),
            l.ctx_kv.compile(b),
            l.ctx_kv_store.compile(b),
            l.qkv.compile(b),
            l.qk_norm_rope.compile(b),
            l.attention.compile(b),
            l.o_proj.compile(b),
            l.attn_ar.compile(b),
            l.post_norm.compile(b),
            l.gate_up.compile(b),
            l.act.compile(b),
            l.down.compile(b),
            l.mlp_ar.compile(b),
        ]);
        CostNode::Labeled {
            label: format!(
                "DSpark draft [{} SWA layers, block {}, {} Markov steps]",
                self.num_layers, self.block, self.draft_tokens
            ),
            child: Box::new(CostNode::Sum(vec![
                fc,
                embed,
                embed_ar,
                CostNode::Scale { n: self.num_layers, child: Box::new(layer) },
                self.final_norm.compile(b),
                self.lm_head.compile(b),
                CostNode::Scale { n: self.draft_tokens, child: Box::new(self.markov.compile(b)) },
            ])),
        }
    }

    fn eval(&self, step: &DraftStep, ev: &mut Evaluator) {
        let r = step.requests;
        let q = r * self.block;
        let z = r == 0;
        let ctx = step.context_rows;
        push(&self.fc, rows(ctx), ctx == 0, ev);
        push(&self.embed, tokens(q), z, ev);
        self.embed_ar.eval(if z { 0 } else { q }, ev);
        let l = &self.layer;
        push(&l.in_norm, RmsNormKernelInput { m: q }, z, ev);
        push(&l.ctx_kv, rows(ctx), ctx == 0, ev);
        push(&l.ctx_kv_store, tokens(ctx + q), z, ev);
        push(&l.qkv, rows(q), z, ev);
        push(&l.qk_norm_rope, tokens(q), z, ev);
        push(
            &l.attention,
            FlashinferAttnRectKernelInput {
                num_requests: r,
                q_len: self.block,
                kv_len: step.mean_context.min(self.window) + self.block,
            },
            z,
            ev,
        );
        push(&l.o_proj, rows(q), z, ev);
        l.attn_ar.eval(if z { 0 } else { q }, ev);
        push(&l.post_norm, RmsNormKernelInput { m: q }, z, ev);
        push(&l.gate_up, rows(q), z, ev);
        push(&l.act, SiluAndMulKernelInput { num_tokens: q }, z, ev);
        push(&l.down, rows(q), z, ev);
        l.mlp_ar.eval(if z { 0 } else { q }, ev);
        push(&self.final_norm, RmsNormKernelInput { m: q }, z, ev);
        push(&self.lm_head, rows(q), z, ev);
        self.markov.eval(&DsparkMarkovStepInput { num_requests: r }, ev);
    }
}

// ---------------------------------------------------------------------------
// The model

pub struct KimiK31Model {
    name: String,
    parallel: KimiK31Parallel,
    capture_sizes: Vec<u32>,
    embedding: Op<ElementwiseKernel>,
    embedding_boundary: Boundary,
    groups: Vec<LayerGroup>,
    final_res: Op<ElementwiseKernel>,
    final_norm: Op<RmsNormKernel>,
    lm_head: Op<SingleGemmKernel>,
    draft: DraftStage,
    kv_cache_layout: KvCacheLayout,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

/// One iteration as the graph sees it.
struct Step {
    /// Per DP group: that group's attention requests, `graph_tokens` = the
    /// group's rows entering attention.
    groups: Vec<KimiAttnInput>,
    /// Rows a rank's residual stream holds outside attention.
    local_rows: u32,
    /// Rows each DP group's attention gathers (sequence parallel).
    group_rows: u32,
    /// Rows the routed experts of each EP rank see.
    routed_rows: u32,
    /// Per DP group: lm_head rows (verify rows + one per prefill chunk).
    logits_rows: Vec<u32>,
    draft: Vec<DraftStep>,
}

fn graph_padded(sorted: &[u32], t: u32) -> u32 {
    match sorted.binary_search(&t) {
        Ok(_) => t,
        Err(i) => sorted.get(i).copied().unwrap_or(t),
    }
}

/// Whether a step replays one full CUDA graph. vLLM's FULL_AND_PIECEWISE mode
/// replays a full graph only for a decode batch within the largest captured
/// size, and DP ranks agree on the mode, so one prefill in any group sends
/// every group to the piecewise graphs with attention eager.
fn replays_full_graph(sorted_capture_sizes: &[u32], groups: &[KimiAttnInput], rows: &[u32]) -> bool {
    groups.iter().all(|g| g.prefill_chunks.is_empty())
        && sorted_capture_sizes
            .last()
            .is_some_and(|&largest| rows.iter().all(|&t| t <= largest))
}

pub fn build(
    name: String,
    model: &KimiK31ModelCfg,
    draft: &KimiDsparkCfg,
    parallel: KimiK31Parallel,
    demand: &ExpertDemand,
    bridge: &PerfApiBridge,
) -> std::result::Result<KimiK31Model, BuildError> {
    let tp = u32::from(parallel.tp_size);
    let ep = u32::from(parallel.ep_size());
    let sp = parallel.sequence_parallel();
    let g = parallel.gpu_name.clone();
    let gs = g.as_str();
    let b = parallel.backends.clone();
    let n = name.as_str();
    let div = |what: &str, v: u32, by: u32| {
        if by == 0 || v % by != 0 {
            return Err(fit_failed(format!("{what} {v} must divide by {by}")));
        }
        Ok(v / by)
    };
    if tp < 2 {
        return Err(fit_failed("the TP-attention graph needs tp_size >= 2"));
    }
    let verify_width = parallel.draft_tokens + 1;
    let hidden = model.hidden;
    let hb = 2 * hidden;
    let mut capture_sizes = parallel.cudagraph_capture_sizes.clone();
    capture_sizes.sort_unstable();
    capture_sizes.dedup();
    let max_rows = capture_sizes.last().copied().unwrap_or(4096).max(4096);

    // KDA: q, k, v, g (full-rank gate) heads split; f_a rank replicated; one
    // beta per local head; padded to 16.
    let kda_heads = div("KDA heads", model.kda_num_heads, tp)?;
    let kda_value = kda_heads * model.kda_head_dim;
    let kda = KimiKdaAttnLocalConfig {
        gpu_name: g.clone(),
        hidden,
        num_heads: kda_heads,
        head_dim: model.kda_head_dim,
        in_proj_width: (4 * kda_value + model.kda_head_dim + kda_heads).div_ceil(16) * 16,
        gate_rank: model.kda_head_dim,
        conv_kernel_size: model.conv_kernel_size,
        verify_width,
        state_dtype: DType::Fp32,
        backends: b.clone(),
    };
    let mla = KimiMlaAttnLocalConfig {
        gpu_name: g.clone(),
        hidden,
        num_heads: div("MLA heads", model.num_attention_heads, tp)?,
        q_lora_rank: model.q_lora_rank,
        kv_lora_rank: model.kv_lora_rank,
        qk_nope_head_dim: model.qk_nope_head_dim,
        qk_rope_head_dim: model.qk_rope_head_dim,
        v_head_dim: model.v_head_dim,
        index_num_heads: model.index_num_heads,
        index_head_dim: model.index_head_dim,
        index_topk: model.index_topk,
        verify_width,
        max_model_len: parallel.max_model_len,
        cache_block_size: 64,
        tp_size: tp,
        // vLLM `attention_config.sparse_mla_shard_q_min_tokens` default.
        shard_q_min_tokens: 1024,
        backends: b.clone(),
    };
    let dense = KimiMlpLocalConfig {
        gpu_name: g.clone(),
        hidden,
        intermediate: div("intermediate_size", model.intermediate_size, tp)?,
        weight_dtype: DType::Bf16,
        backends: b.clone(),
    };
    let shared_total = model.moe_intermediate_size * model.num_shared_experts;
    let shared = KimiMlpLocalConfig {
        gpu_name: g.clone(),
        hidden,
        // Replicated under sequence parallelism, TP-sliced otherwise.
        intermediate: if sp { shared_total } else { div("shared intermediate", shared_total, tp)? },
        weight_dtype: DType::Mxfp8E4m3,
        backends: b.clone(),
    };
    div("experts", model.num_experts, ep)?;
    let routed_template = Glm53RoutedMoeLocalWorkletConfig {
        hidden: Dim::param("latent_size", model.latent_size),
        moe_intermediate: model.moe_intermediate_size.into(),
        num_experts: model.num_experts.into(),
        ep_size: parallel.ep_size(),
        top_k: model.experts_per_token,
        n_group: 1,
        topk_group: 1,
        routed_scaling_numerator: 1,
        routed_scaling_denominator: 1,
        precision: Glm53WeightPrecision::Nvfp4,
        routing_method: ROUTING_METHOD.to_string(),
        activation_dtype: DType::Bf16,
        gpu_name: g.clone(),
        quant_backends: vec!["vllm_cuda"],
        fused_moe_backends: vec![if sp { "flashinfer_trtllm_routed_sm100" } else { "flashinfer_trtllm_sm100" }],
        expert_demand: demand.clone(),
        folded_rank_position: 0,
    };
    // Sequence parallel splits a DP group's padded rows over its TP ranks, so
    // a graph replay of capture size c puts c / tp rows on each of the
    // tp * dp EP ranks: the EP exchange's equal-split total is c * dp. Measure
    // those totals instead of interpolating between the shared grid's points.
    let ep_extra_totals: Vec<u32> =
        capture_sizes.iter().map(|&c| c * u32::from(parallel.dp_size)).collect();
    let routed: Vec<Nvfp4FusedMoeKernelConfig> =
        Glm53RoutedMoeLocalWorkletConfig::split_for_ep(routed_template, demand.clone())
            .iter()
            .map(|cfg| {
                let mut fused = Glm53RoutedMoeLocalWorklet::resolve_config(cfg).fused_moe;
                // Graph replays pad every step's routed rows to a capture size,
                // so those are the row counts the MoE sees between the shared
                // grid's points: measure them instead of interpolating.
                fused.extra_token_points = capture_sizes.clone();
                fused.gated_activation = model.routed_gated_activation.clone();
                fused
            })
            .collect();

    let boundary = |prefix: &str| -> std::result::Result<Boundary, BuildError> {
        Ok(if sp {
            Boundary::Sp(SpBoundary::build(prefix, &parallel, hidden, max_rows, bridge)?)
        } else {
            Boundary::AllReduce(TpAllReduce::build(prefix, "attn_all_reduce", &parallel, hidden, bridge)?)
        })
    };
    let res = |prefix: &str, suffix: &str| {
        // `attn_res_fwd_online_v2`: the stream row and the block accumulator
        // in, the combined row and the accumulator out.
        atomic(prefix, suffix, ew(&b, gs, 2 * hb, 2 * hb), ElementwiseKernel::build, bridge)
    };
    let moe_block = |prefix: &str| -> std::result::Result<MoeBlock, BuildError> {
        let m = format!("{prefix}.moe");
        let mm = m.as_str();
        let latent = model.latent_size;
        let collective_fabric = parallel.ep_fabric.as_str().to_string();
        Ok(MoeBlock {
            router: Glm53MoeRouterLocalWorklet::build(
                format!("{mm}.router"),
                Glm53MoeRouterLocalWorklet::resolve_config(&Glm53MoeRouterLocalWorkletConfig {
                    hidden: hidden.into(),
                    num_experts: model.num_experts.into(),
                    activation_dtype: DType::Bf16,
                    gpu_name: g.clone(),
                    // `GateLinear` (kimi_k3/nvidia/model.py:440): ll_bf16 up to
                    // 16 tokens, cuBLAS above.
                    gemm_backends: vec!["vllm_gate_linear"],
                }),
                // The gate runs once: `FusedMoEFactory` gets no `gate`, so the
                // runner does not recompute the logits.
                true,
                bridge,
            )?,
            topk: if sp {
                Some(atomic(mm, "single_group_topk", ew(&b, gs, 4 * model.num_experts, 8 * model.experts_per_token), ElementwiseKernel::build, bridge)?)
            } else {
                None
            },
            shared: KimiMlpLocal::build(format!("{mm}.shared_experts"), &shared, bridge)?,
            latent_down: atomic(mm, "latent_down", gemm(&b.bf16_gemm, gs, latent, hidden, DType::Bf16), SingleGemmKernel::build, bridge)?,
            quant: atomic(
                mm,
                "latent_quant",
                Nvfp4QuantKernelConfig {
                    backends: vec!["vllm_cuda"],
                    gpu_name: g.clone(),
                    hidden_size: Dim::param("latent_size", latent),
                    group_size: NVFP4_GROUP,
                    input_dtype: DType::Bf16,
                    scale_format: NVFP4_SCALE_FORMAT.to_string(),
                },
                Nvfp4QuantKernel::build,
                bridge,
            )?,
            gather: if sp {
                Some(atomic(
                    mm,
                    "ep_all_gather",
                    MoeEpQuantizedAllGatherKernelConfig {
                        backends: vec!["vllm_pynccl"],
                        gpu_name: g.clone(),
                        num_gpus: ep,
                        hidden_size: Dim::param("latent_size", latent),
                        top_k: model.experts_per_token.into(),
                        activation_dtype: DType::Nvfp4E2m1,
                        fabric: collective_fabric.clone(),
                        max_total_tokens: max_rows * u32::from(parallel.dp_size),
                        extra_total_tokens: ep_extra_totals.clone(),
                    },
                    MoeEpQuantizedAllGatherKernel::build,
                    bridge,
                )?)
            } else {
                None
            },
            routed: routed
                .iter()
                .enumerate()
                .map(|(rank, cfg)| {
                    atomic(mm, &format!("routed_rank{rank}"), cfg.clone(), Nvfp4FusedMoeKernel::build, bridge)
                })
                .collect::<std::result::Result<_, _>>()?,
            scatter: if sp {
                Some(atomic(
                    mm,
                    "ep_reduce_scatter",
                    MoeEpReduceScatterKernelConfig {
                        backends: vec!["vllm_pynccl"],
                        gpu_name: g.clone(),
                        num_gpus: ep,
                        hidden_size: Dim::param("latent_size", latent),
                        dtype: DType::Bf16,
                        fabric: collective_fabric,
                        max_total_tokens: max_rows * u32::from(parallel.dp_size),
                        extra_total_tokens: ep_extra_totals.clone(),
                    },
                    MoeEpReduceScatterKernel::build,
                    bridge,
                )?)
            } else {
                None
            },
            latent_all_reduce: if sp {
                None
            } else {
                Some(TpAllReduce::build(mm, "latent_all_reduce", &parallel, latent, bridge)?)
            },
            latent_norm: atomic(mm, "latent_norm", norm(&b, gs, latent), RmsNormKernel::build, bridge)?,
            up: atomic(mm, "latent_up", gemm(&b.bf16_gemm, gs, hidden, latent, DType::Bf16), SingleGemmKernel::build, bridge)?,
            shared_all_reduce: if sp {
                None
            } else {
                Some(TpAllReduce::build(mm, "shared_all_reduce", &parallel, hidden, bridge)?)
            },
            add: atomic(mm, "combine_add", ew(&b, gs, 2 * hb, hb), ElementwiseKernel::build, bridge)?,
            ep,
            name: m,
        })
    };

    let dp = usize::from(parallel.dp_size);
    let mla_moe: Vec<u32> = model.mla_layers.clone();
    let kda_moe = model.kda_moe_layers();
    let mut groups = Vec::new();
    {
        let p = format!("{n}.kda_dense");
        groups.push(LayerGroup {
            label: "kda_dense".into(),
            layers: vec![0],
            attn_res: res(&p, "attn_res")?,
            attn: Attention::Kda(KimiKdaAttnLocal::build(format!("{p}.kda"), &kda, bridge)?),
            boundary: boundary(&p)?,
            ffn_res: res(&p, "ffn_res")?,
            ffn: Ffn::Dense(
                KimiMlpLocal::build(format!("{p}.dense_ffn"), &dense, bridge)?,
                if sp { None } else { Some(TpAllReduce::build(&p, "ffn_all_reduce", &parallel, hidden, bridge)?) },
            ),
            dp,
        });
    }
    for (label, layers, is_mla) in [("kda_moe", kda_moe, false), ("mla_moe", mla_moe, true)] {
        let p = format!("{n}.{label}");
        groups.push(LayerGroup {
            label: label.into(),
            layers,
            attn_res: res(&p, "attn_res")?,
            attn: if is_mla {
                Attention::Mla(KimiMlaAttnLocal::build(format!("{p}.mla"), &mla, bridge)?)
            } else {
                Attention::Kda(KimiKdaAttnLocal::build(format!("{p}.kda"), &kda, bridge)?)
            },
            boundary: boundary(&p)?,
            ffn_res: res(&p, "ffn_res")?,
            ffn: Ffn::Moe(moe_block(&p)?),
            dp,
        });
    }

    let kv_cache_layout = kv_cache_layout(model, draft, &parallel);
    let mut out = KimiK31Model {
        embedding: atomic(n, "embedding", ew(&b, gs, hb, hb), ElementwiseKernel::build, bridge)?,
        embedding_boundary: if sp {
            Boundary::Sp(SpBoundary::build(&format!("{n}.embedding"), &parallel, hidden, max_rows, bridge)?)
        } else {
            Boundary::AllReduce(TpAllReduce::build(n, "embedding_all_reduce", &parallel, hidden, bridge)?)
        },
        groups,
        final_res: res(n, "final_attn_res")?,
        final_norm: atomic(n, "final_norm", norm(&b, gs, hidden), RmsNormKernel::build, bridge)?,
        lm_head: atomic(
            n,
            "lm_head",
            gemm(&b.bf16_gemm, gs, div("vocab", model.vocab_size, tp)?, hidden, DType::Bf16),
            SingleGemmKernel::build,
            bridge,
        )?,
        draft: DraftStage::build(n, draft, &parallel, bridge)?,
        kv_cache_layout,
        capture_sizes,
        parallel,
        cost_flat: Vec::new(),
        n_slots: 0,
        name,
    };
    let tree = out.cost_tree();
    out.cost_flat = tree.flatten();
    out.n_slots = tree.n_slots();
    Ok(out)
}

/// vLLM's hybrid attention block for Kimi K3 (`_align_hybrid_block_size` ->
/// `kimi_k3/common/dsa_cache.packed_dsa_block_size`). A block row holds every
/// MLA layer's fp8 latent + MXFP4 indexer page and fits as many per-rank KDA
/// states (fp32 conv window of `conv - 1 + num_spec` steps plus the fp32 SSM
/// state) as there are MLA layers. The KDA state shards with TP and the row
/// does not, so the block is rounded at the reference TP and scaled by
/// `reference / tp`: 768 tokens at TP16 and 1536 at TP8, as the captures log
/// (`Setting attention block size to N tokens`).
pub fn hybrid_block_tokens(model: &KimiK31ModelCfg, draft: &KimiDsparkCfg, tp: u32) -> u32 {
    let tp = u64::from(tp.max(1));
    let mla = model.mla_layers.len() as u64;
    let kda = u64::from(model.num_layers) - mla;
    let states_per_row = kda.div_ceil(kda.div_ceil(mla.max(1)).max(1));
    let heads = u64::from(model.kda_num_heads);
    let hd = u64::from(model.kda_head_dim);
    let num_spec = u64::from(draft.block_size);
    let conv = 3 * heads * hd / tp * (u64::from(model.conv_kernel_size) - 1 + num_spec) * 4;
    let ssm = heads / tp * hd * hd * 4;
    let row_bytes_per_token = mla * MLA_BYTES_PER_TOKEN_PER_LAYER;
    let reference_tp = lcm_u64(HYBRID_BLOCK_REFERENCE_TP, tp);
    let units = (states_per_row * (conv + ssm) * tp)
        .div_ceil(HYBRID_BLOCK_ALIGNMENT_TOKENS * reference_tp * row_bytes_per_token);
    u32::try_from(HYBRID_BLOCK_ALIGNMENT_TOKENS * units * (reference_tp / tp)).expect("hybrid block size exceeds u32")
}

fn lcm_u64(a: u64, b: u64) -> u64 {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    a / x * b
}

/// One request's cache on the replica, as vLLM's hybrid manager pages it with
/// a [`hybrid_block_tokens`] block: the 24 MLA layers' fp8 latent + MXFP4
/// index pages replicated on every attention rank; the KDA state (fp32 SSM +
/// conv window) split over the TP heads, held as one block per KDA KV-cache
/// group (ceil(69 / 24) = 3) per request; the draft's five sliding-window
/// layers as their own group, one window per request.
pub fn kv_cache_layout(model: &KimiK31ModelCfg, draft: &KimiDsparkCfg, parallel: &KimiK31Parallel) -> KvCacheLayout {
    let tp = u32::from(parallel.tp_size);
    let mla = model.mla_layers.len() as u64;
    let kda = u64::from(model.num_layers) - mla;
    let block_tokens = hybrid_block_tokens(model, draft, tp);
    let block_bytes = mla * u64::from(block_tokens) * MLA_BYTES_PER_TOKEN_PER_LAYER;
    let kda_groups = kda.div_ceil(mla.max(1));
    let heads = u64::from(model.kda_num_heads);
    let hd = u64::from(model.kda_head_dim);
    let ssm = heads * hd * hd * 4;
    let conv = 3 * heads * hd * u64::from(model.conv_kernel_size - 1) * 4;
    let draft_kv_per_token = u64::from(draft.num_layers) * 2 * u64::from(draft.num_kv_heads) * u64::from(draft.head_dim);
    let mut layout = KvCacheLayout::new()
        .with_part(KvCachePart::per_token("mla", mla * MLA_BYTES_PER_TOKEN_PER_LAYER).replicated(tp))
        .with_part(
            KvCachePart::per_request("kda", kda * (ssm + conv))
                .padded_to(u64::from(tp) * kda_groups * block_bytes)
                .sharded(tp),
        );
    layout = layout.with_part(
        KvCachePart::sliding_window("dspark_draft", draft.sliding_window, draft_kv_per_token).sharded(tp),
    );
    layout.with_engine(EngineOverhead {
        block_tokens: Some(block_tokens),
        fixed_device_bytes: u64::from(tp) * block_bytes,
    })
}

impl KimiK31Model {
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let mut children = vec![self.embedding.compile(&mut b)];
        match &self.embedding_boundary {
            Boundary::AllReduce(ar) => children.push(ar.compile(&mut b)),
            Boundary::Sp(sp) => children.push(sp.scatter.compile(&mut b)),
        }
        for g in &self.groups {
            children.push(g.compile(&mut b));
        }
        children.push(self.final_res.compile(&mut b));
        children.push(self.final_norm.compile(&mut b));
        let dp = usize::from(self.parallel.dp_size);
        let heads = (0..dp).map(|_| self.lm_head.compile(&mut b)).collect();
        children.push(CostNode::Max { overlap: 1.0, children: heads });
        let drafts = (0..dp).map(|_| self.draft.compile(&mut b)).collect();
        children.push(CostNode::Labeled {
            label: format!("draft (max over {dp} DP groups)"),
            child: Box::new(CostNode::Max { overlap: 1.0, children: drafts }),
        });
        let layout = format!(
            "TP{} x DP{}, EP{}{}",
            self.parallel.tp_size,
            self.parallel.dp_size,
            self.parallel.ep_size(),
            if self.parallel.sequence_parallel() { ", sequence parallel" } else { "" }
        );
        b.finish(CostNode::Labeled {
            label: format!("{} (KimiK31Model) [{layout}; verify width {}]", self.name, self.parallel.draft_tokens + 1),
            child: Box::new(CostNode::Sum(children)),
        })
    }

    fn normalize(&self, input: &SpeculativeArchInput) -> std::result::Result<Step, String> {
        let dp = usize::from(self.parallel.dp_size);
        let tp = u32::from(self.parallel.tp_size);
        if input.draft_tokens != self.parallel.draft_tokens {
            return Err(format!(
                "built for {} draft tokens, got {}",
                self.parallel.draft_tokens, input.draft_tokens
            ));
        }
        if input.groups.len() != dp {
            return Err(format!("expected {dp} attention DP groups, got {}", input.groups.len()));
        }
        let mut groups = Vec::with_capacity(dp);
        let mut real = Vec::with_capacity(dp);
        let mut logits_rows = Vec::with_capacity(dp);
        let mut draft = Vec::with_capacity(dp);
        for (gi, group) in input.groups.iter().enumerate() {
            let mut attn = KimiAttnInput::default();
            let mut ctx_sum = 0u64;
            for &(prefix, append) in &group.prefill_chunk_pairs {
                if append == 0 {
                    return Err(format!("group {gi}: a prefill chunk appends no tokens"));
                }
                let context = prefix + append;
                if context > self.parallel.max_model_len {
                    return Err(format!("group {gi}: context {context} exceeds max_model_len"));
                }
                attn.prefill_chunks.push((append, context));
                ctx_sum += u64::from(context);
            }
            let mut decode_rows = 0u32;
            for d in &group.decode_requests {
                if d.kv_len == 0 || d.kv_len > self.parallel.max_model_len {
                    return Err(format!("group {gi}: decode context {} out of range", d.kv_len));
                }
                attn.decode_kv_lens.push(d.kv_len);
                decode_rows += d.query_len;
                ctx_sum += u64::from(d.kv_len);
            }
            let prefill: u32 = attn.prefill_chunks.iter().map(|&(q, _)| q).sum();
            let tokens = prefill + decode_rows;
            if tokens != group.batch_tokens {
                return Err(format!("group {gi}: batch_tokens {} != rows {tokens}", group.batch_tokens));
            }
            let requests = group.request_count();
            logits_rows.push(decode_rows + attn.prefill_chunks.len() as u32);
            draft.push(DraftStep {
                requests,
                context_rows: tokens,
                mean_context: if requests == 0 { 0 } else { (ctx_sum / u64::from(requests)) as u32 },
            });
            real.push(tokens);
            groups.push(attn);
        }
        if real.iter().all(|&t| t == 0) {
            return Err("an iteration must carry at least one token".into());
        }
        // Every DP group pads to the busiest group's graph (vLLM
        // `num_tokens_across_dp`); sequence parallel splits rows over TP.
        let padded = real.iter().map(|&t| graph_padded(&self.capture_sizes, t)).max().unwrap_or(0);
        let (local, group_rows, routed) = if self.parallel.sequence_parallel() {
            let rows = padded.div_ceil(tp) * tp;
            (rows / tp, rows, rows / tp * u32::from(self.parallel.ep_size()))
        } else {
            (padded, padded, padded)
        };
        let full_graph = replays_full_graph(&self.capture_sizes, &groups, &real);
        for g in &mut groups {
            g.graph_tokens = group_rows;
            g.full_graph = full_graph;
        }
        Ok(Step { groups, local_rows: local, group_rows, routed_rows: routed, logits_rows, draft })
    }

    fn eval_into(&self, input: &SpeculativeArchInput, ev: &mut Evaluator) {
        let step = self
            .normalize(input)
            .unwrap_or_else(|reason| panic!("invalid KimiK31Model input: {reason}"));
        push(&self.embedding, tokens(step.local_rows), false, ev);
        match &self.embedding_boundary {
            Boundary::AllReduce(ar) => ar.eval(step.local_rows, ev),
            Boundary::Sp(sp) => push(&sp.scatter, sp.input(step.group_rows), false, ev),
        }
        for g in &self.groups {
            g.eval(&step, ev);
        }
        push(&self.final_res, tokens(step.local_rows), false, ev);
        push(&self.final_norm, RmsNormKernelInput { m: step.local_rows }, false, ev);
        for &m in &step.logits_rows {
            push(&self.lm_head, rows(m), m == 0, ev);
        }
        for d in &step.draft {
            self.draft.eval(d, ev);
        }
    }

    fn eval_checked(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: Option<&mut Vec<SlotInput>>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut ev = match inputs {
            Some(inputs) => Evaluator::with_inputs(slots, inputs),
            None => Evaluator::new(slots),
        };
        self.eval_into(batch, &mut ev);
        assert_eq!(ev.filled(), self.n_slots, "eval must fill every compiled slot");
        drop(ev);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }
}

impl SpeculativeUnifiedModel for KimiK31Model {
    fn check_input(&self, batch: &SpeculativeArchInput) -> Result<(), String> {
        self.normalize(batch).map(|_| ())
    }

    fn eval_speculative_iter(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        self.eval_checked(batch, slots, scratch, None)
    }

    fn eval_speculative_iter_with_inputs(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        inputs.clear();
        let total = self.eval_checked(batch, slots, scratch, Some(inputs));
        assert_eq!(inputs.len(), self.n_slots, "slot inputs must align with compiled slots");
        total
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout.clone()
    }

    fn max_model_len(&self) -> u32 {
        self.parallel.max_model_len
    }

    fn gpus_per_replica(&self) -> u16 {
        self.parallel.ep_size()
    }

    fn num_attn_dp_groups(&self) -> u16 {
        self.parallel.dp_size
    }

    fn num_attn_shards(&self) -> u16 {
        self.parallel.tp_size
    }

    /// None: DSpark drafts in its own query pass (vLLM `DSparkSpeculatorConfig`
    /// keeps `SpeculatorConfig.max_num_new_slots_for_drafting = 0`), so its
    /// mask queries take no rows from the target batch's token budget.
    fn drafting_slots_per_request(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> KimiK31ModelCfg {
        let root = env!("CARGO_MANIFEST_DIR");
        KimiK31ModelCfg::from_json(&Path::new(root).join("model/config/kimi_k31_pro_nvfp4.json")).unwrap()
    }

    #[test]
    fn checkpoint_parses_to_the_hybrid_schedule() {
        let m = model();
        assert_eq!(m.num_layers, 93);
        assert_eq!(m.mla_layers.len(), 24);
        assert_eq!(m.mla_layers[0], 3);
        assert_eq!(*m.mla_layers.last().unwrap(), 92);
        assert_eq!(m.kda_moe_layers().len(), 68);
        assert_eq!(m.latent_size, 3584);
        assert_eq!(m.num_moe_layers(), 92);
    }

    #[test]
    fn routed_experts_take_the_checkpoint_situ() {
        // config.json: hidden_act = "situ", activation_situ_beta = 4.0,
        // activation_situ_linear_beta = 25.0; vLLM passes them to the TRT-LLM
        // NVFP4 experts as gemm1_alpha / gemm1_beta (ActivationType.Situ).
        assert_eq!(
            model().routed_gated_activation.as_deref(),
            Some("situ:beta=4,linear_beta=25")
        );
        let mut v: Value = serde_json::from_str(
            &std::fs::read_to_string(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/kimi_k31_pro_nvfp4.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let cfg = v.get_mut("text_config").unwrap();
        cfg["hidden_act"] = Value::from("silu");
        assert_eq!(routed_gated_activation(cfg).unwrap(), None);
        cfg["hidden_act"] = Value::from("gelu");
        assert!(routed_gated_activation(cfg).is_err());
    }

    #[test]
    fn draft_config_parses() {
        let root = env!("CARGO_MANIFEST_DIR");
        let d = KimiDsparkCfg::from_json(&Path::new(root).join("model/config/kimi_k31_pro_dspark.json")).unwrap();
        assert_eq!((d.num_layers, d.hidden, d.num_heads, d.num_kv_heads, d.head_dim), (5, 7168, 96, 16, 64));
        assert_eq!((d.block_size, d.markov_rank, d.num_aux_layers), (7, 512, 5));
    }

    #[test]
    fn hybrid_block_matches_the_logged_attention_block_size() {
        // vLLM: "Setting attention block size to 768 tokens" on GB200 TP16 and
        // "... 1536 tokens" on B200 TP8 (DP2), both with DSpark k=7.
        let root = env!("CARGO_MANIFEST_DIR");
        let d = KimiDsparkCfg::from_json(&Path::new(root).join("model/config/kimi_k31_pro_dspark.json")).unwrap();
        let m = model();
        assert_eq!(hybrid_block_tokens(&m, &d, 16), 768);
        assert_eq!(hybrid_block_tokens(&m, &d, 8), 1536);
    }

    #[test]
    fn full_graph_needs_every_group_decode_only_and_captured() {
        let decode = KimiAttnInput { decode_kv_lens: vec![4096; 12], ..Default::default() };
        let mixed = KimiAttnInput {
            prefill_chunks: vec![(512, 8192)],
            decode_kv_lens: vec![4096; 4],
            ..Default::default()
        };
        let sizes = [8, 16, 96, 512];
        assert!(replays_full_graph(&sizes, &[decode.clone(), decode.clone()], &[96, 0]));
        assert!(!replays_full_graph(&sizes, &[decode.clone(), mixed], &[96, 544]));
        assert!(!replays_full_graph(&sizes, &[decode.clone()], &[600]));
        assert!(!replays_full_graph(&[], &[decode], &[96]));
    }
}
