//! MiMo-V2 (`MiMoV2ForCausalLM`) FP8 checkpoints on B200 under vLLM's TP
//! group with expert parallelism over the same ranks (`--enable-expert-parallel`,
//! TP = EP, MTP, vision and audio off). Measured against the vLLM capture of
//! MiMo-V2.6-Flash-RL at TP4 + EP4 on the diverse_100 trace (job 6264).
//!
//! One rank group of `tp_size` ranks. Attention is tensor-parallel by head;
//! the router is replicated; each rank owns `n_routed_experts / tp_size` whole
//! experts and computes them over every token, so the routed MoE's partial sum
//! closes in the post-FFN all-reduce like any row-parallel output. There is no
//! stand-alone RMSNorm: vLLM's `allreduce_rms` pass fuses every TP all-reduce
//! with the residual add and the next RMSNorm into one FlashInfer call
//! (`all_reduce_residual_rms_norm`), 2 per layer plus the embedding's (whose
//! residual is zero and whose norm is layer 0's input norm); the last layer's
//! applies the final norm. A step above the fusion's token cap, or a TP it does
//! not fuse at, runs the unfused all-reduce and residual RMSNorm instead
//! ([`TpAllReduceRmsNormOp`]).
//!
//! ```text
//! prologue       embedding gather, all-reduce + layer 0's input norm
//! per layer      attention (FP8 quant, QKV, RoPE + V scale, KV insert, FA4, O)
//!                + AR/norm, dense MLP or MoE (router, routed experts)
//!                + AR/norm                                (Scale per group)
//! epilogue       last-row select, lm_head, logits all-gather
//! ```
//!
//! Layers are global or sliding-window (`hybrid_layer_pattern`), each with its
//! own head count, KV page and FA4 config, and carry a dense MLP or an MoE
//! (`moe_layer_freq`). The routed experts are MXFP4 weights run against BF16
//! activations (vLLM's `FLASHINFER_TRTLLM_MXFP4_BF16`: no activation quant,
//! routing inside the call over the router's FP32 logits).
//!
//! A step of at most the largest capture size replays a CUDA graph
//! (FULL_AND_PIECEWISE): every launch but attention runs on the token count
//! padded to the next captured size, and a uniform decode launches attention
//! over the padded request count. Larger steps run eagerly.
//!
//! Sampling is not modeled.

use anyhow::Result;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::glm53_flash_vllm_fp8_kda_dsa_moe::graph_padded_tokens;
use crate::arch::kv_layout::{gqa_kv_bytes_per_token, KvCacheLayout, KvCachePart};
use crate::arch::mimo_v2_model_cfg::{MimoV2AttnShape, MimoV2ModelCfg};
use crate::common::Fabric;
use crate::op::comm::{TpAllReduceRmsNormConfig, TpAllReduceRmsNormOp};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::all_reduce_fusion::fused_all_reduce_refusal;
use crate::timing::kernels::{
    AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput,
    AllReduceResidualRmsNormKernelConfig, ElementwiseKernel, ElementwiseKernelConfig,
    ElementwiseKernelInput, GemmFp32OutputKernel, GemmFp32OutputKernelConfig,
    GemmFp32OutputKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Dim, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::deepseek_v41_common::all_gather_as_all_reduce_bytes;
use crate::worklet::glm53_common::{atomic, push_or_zero};
use crate::worklet::{
    Glm53MlpLocalWorklet, Glm53MlpLocalWorkletConfig, Glm53MlpLocalWorkletInput,
    Glm53MlpLocalWorkletResolved, Glm53WeightPrecision, MimoV2AttnLocalWorklet,
    MimoV2AttnLocalWorkletConfig, MimoV2AttnLocalWorkletInput, MimoV2AttnLocalWorkletResolved,
    Nvfp4MoeLocalWorklet, Nvfp4MoeLocalWorkletConfig, Nvfp4MoeLocalWorkletInput,
    Nvfp4MoeLocalWorkletResolved,
};

const ARCH_KIND: &str = "mimo_v2_vllm_fp8_swa_moe";
const ACTIVATION_DTYPE: DType = DType::Bf16;
/// vLLM `--block-size` (the capture's, and vLLM's default). The layer type
/// with the smaller page scales its block up to the largest page
/// (`unify_kv_cache_spec_page_size`), see [`kv_page_sizes`].
const KV_BLOCK_SIZE: u32 = 16;

const FP8_QUANT_BACKENDS: &[&str] = &["vllm_cuda"];
/// DeepGEMM `fp8_gemm_nt` (`sm100_fp8_fp4_gemm_1d1d` in the capture).
const FP8_GEMM_BACKENDS: &[&str] = &["deepgemm"];
/// `F.linear` in vLLM's torch: the unquantized o_proj and lm_head.
const BF16_GEMM_BACKENDS: &[&str] = &["torch_linear_vllm"];
/// `GateLinear` with `out_dtype=float32`: cuBLAS `torch.mm` at every token
/// count in the capture.
const ROUTER_BACKENDS: &[&str] = &["torch_cublas"];
/// Compiled Triton launches (embedding, RoPE + V scale, SiLU-and-mul, the
/// logits row select).
const TRITON_ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
const KV_APPEND_BACKENDS: &[&str] = &["vllm_triton_diffkv"];
const ATTENTION_BACKENDS: &[&str] = &["vllm_fa4"];
/// `trtllm_fp4_block_scale_moe` with MXFP4 weights and BF16 activations.
const ROUTED_MOE_BACKENDS: &[&str] = &["flashinfer_trtllm_sm100_mxfp4_bf16"];
/// In-call routing: DeepSeek-V3 sigmoid + correction bias top-k.
const ROUTING_METHOD: &str = "deepseek_v3";
/// FlashInfer `auto` picks the MNNVL workspace on single-node B200 (capture
/// server log: `backend=mnnvl`).
const AR_NORM_BACKENDS: &[&str] = &["flashinfer_mnnvl"];
const AR_NORM_STRATEGY: &str = "auto";
/// The unfused all-reduce above the fused cap: pynccl at TP 4 on SM100, and
/// the NCCL curve standing in for torch symmetric memory at TP 8
/// ([`TpAllReduceRmsNormOp`]).
const UNFUSED_ALL_REDUCE_BACKENDS: &[&str] = &["nccl"];
const LOGITS_ALL_GATHER_BACKENDS: &[&str] = &["nccl"];
/// `vocab_parallel_embedding` reads one int32 token id per row.
const INT32_TOKEN_ID_BYTES: u32 = 4;
/// `hidden_states[logits_indices]`: an int64 index per selected row.
const INT64_INDEX_BYTES: u32 = 8;

/// Deployment layout: one tensor-parallel rank group, experts EP over it.
#[derive(Clone, Debug)]
pub struct MimoV2VllmParallel {
    pub tp_size: u16,
    pub max_model_len: u32,
    pub gpu_name: String,
    /// vLLM `--cudagraph-capture-sizes`; empty runs eager (no padding).
    pub cudagraph_capture_sizes: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttnKind {
    Global,
    SlidingWindow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FfnKind {
    Dense,
    Moe,
}

/// A run of layers with one identical sublayer shape, folded by `Scale{n}`.
#[derive(Clone, Debug)]
pub struct MimoV2LayerGroup {
    pub label: String,
    pub layers: Vec<u32>,
    attn: AttnKind,
    ffn: FfnKind,
}

fn layer_groups(model: &MimoV2ModelCfg) -> Vec<MimoV2LayerGroup> {
    let mut groups: Vec<MimoV2LayerGroup> = Vec::new();
    for layer in 0..model.num_layers {
        let attn = if model.is_swa(layer) {
            AttnKind::SlidingWindow
        } else {
            AttnKind::Global
        };
        let ffn = if model.is_moe(layer) {
            FfnKind::Moe
        } else {
            FfnKind::Dense
        };
        if let Some(group) = groups.iter_mut().find(|g| g.attn == attn && g.ffn == ffn) {
            group.layers.push(layer);
            continue;
        }
        let label = format!(
            "{}_{}",
            match attn {
                AttnKind::Global => "global",
                AttnKind::SlidingWindow => "swa",
            },
            match ffn {
                FfnKind::Dense => "dense",
                FfnKind::Moe => "moe",
            }
        );
        groups.push(MimoV2LayerGroup {
            label,
            layers: vec![layer],
            attn,
            ffn,
        });
    }
    groups
}

/// KV heads one rank stores: its share, or one whole head when TP exceeds
/// the head count (`max(1, num_kv_heads // tp)` in `MiMoV2Attention`).
fn kv_heads_per_rank(shape: &MimoV2AttnShape, tp: u32) -> std::result::Result<u32, BuildError> {
    let heads = shape.num_key_value_heads;
    if heads.is_multiple_of(tp) || tp.is_multiple_of(heads) {
        Ok((heads / tp).max(1))
    } else {
        Err(fit_failed(format!(
            "{heads} KV heads neither divide by nor divide TP {tp}"
        )))
    }
}

/// One rank's KV page bytes of one layer at `block` tokens.
fn page_bytes(shape: &MimoV2AttnShape, kv_heads: u32, block: u32) -> u32 {
    block * kv_heads * (shape.head_dim + shape.v_head_dim) * ACTIVATION_DTYPE.size_bytes()
}

/// Tokens per KV page of the global and sliding-window layers. vLLM gives
/// every layer `KV_BLOCK_SIZE` tokens, then scales the block of the type with
/// the smaller page until both pages are the same size
/// (`unify_kv_cache_spec_page_size`): at TP4 a global layer's one KV head
/// takes 32-token pages to match a window layer's two heads at 16.
fn kv_page_sizes(model: &MimoV2ModelCfg, tp: u32) -> std::result::Result<(u32, u32), BuildError> {
    let global = page_bytes(
        &model.global_attn,
        kv_heads_per_rank(&model.global_attn, tp)?,
        KV_BLOCK_SIZE,
    );
    let swa = page_bytes(
        &model.swa_attn,
        kv_heads_per_rank(&model.swa_attn, tp)?,
        KV_BLOCK_SIZE,
    );
    let largest = global.max(swa);
    let scaled = |page: u32| {
        if largest.is_multiple_of(page) {
            Ok(KV_BLOCK_SIZE * (largest / page))
        } else {
            Err(fit_failed(format!(
                "a {page}-byte KV page does not divide the {largest}-byte one (vLLM then pads \
                 the page, which is not modeled)"
            )))
        }
    };
    Ok((scaled(global)?, scaled(swa)?))
}

/// Every configuration the model is built from.
#[derive(Clone, Debug)]
pub struct MimoV2VllmConfigs {
    pub model: MimoV2ModelCfg,
    pub parallel: MimoV2VllmParallel,
    pub groups: Vec<MimoV2LayerGroup>,
    pub attn_global: MimoV2AttnLocalWorkletConfig,
    pub attn_swa: MimoV2AttnLocalWorkletConfig,
    pub dense_ffn: Glm53MlpLocalWorkletConfig,
    pub router: GemmFp32OutputKernelConfig,
    /// One per ranked EP workload.
    pub routed: Vec<Nvfp4MoeLocalWorkletConfig>,
    /// The post-attention and post-FFN boundaries.
    pub ar_norm: TpAllReduceRmsNormConfig,
    /// The embedding's boundary: no residual yet, so its unfused norm reads
    /// and writes one row.
    pub embed_ar_norm: TpAllReduceRmsNormConfig,
    pub embed: ElementwiseKernelConfig,
    pub logits_select: ElementwiseKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    /// `None` on one GPU, which gathers nothing.
    pub logits_all_gather: Option<AllReduceKernelConfig>,
}

pub fn build_configs(
    model: &MimoV2ModelCfg,
    parallel: &MimoV2VllmParallel,
    demand: &ExpertDemand,
) -> std::result::Result<MimoV2VllmConfigs, BuildError> {
    let tp = u32::from(parallel.tp_size);
    let gpu = parallel.gpu_name.clone();
    let divide = |what: &str, value: u32| -> std::result::Result<u32, BuildError> {
        if !value.is_multiple_of(tp) {
            return Err(fit_failed(format!("{what} {value} must divide by TP {tp}")));
        }
        Ok(value / tp)
    };
    let (global_page, swa_page) = kv_page_sizes(model, tp)?;
    let hidden_bytes = model.hidden * ACTIVATION_DTYPE.size_bytes();
    let ew = |input: u32, output: u32| ElementwiseKernelConfig {
        backends: TRITON_ELEMENTWISE_BACKENDS.to_vec(),
        gpu_name: gpu.clone(),
        input_bytes_per_token: input.into(),
        output_bytes_per_token: output.into(),
    };
    let cuda_graph = !parallel.cudagraph_capture_sizes.is_empty();
    // vLLM fuses only at a TP FlashInfer has a budget for (none at TP 1, where
    // nothing is reduced either).
    let fused = fused_all_reduce_refusal(AR_NORM_BACKENDS, &gpu, tp).is_none();
    let tp_boundary = |norm: ElementwiseKernelConfig| TpAllReduceRmsNormConfig {
        fused: fused.then(|| AllReduceResidualRmsNormKernelConfig {
            backends: AR_NORM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            num_gpus: tp,
            hidden_dim: model.hidden,
            dtype: ACTIVATION_DTYPE,
            fabric: Fabric::Nvlink,
            strategy: AR_NORM_STRATEGY.to_string(),
            launch_with_pdl: true,
            fp32_acc: true,
            fused_token_limit: None,
        }),
        all_reduce: (tp > 1).then(|| AllReduceKernelConfig {
            backends: UNFUSED_ALL_REDUCE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            num_gpus: tp,
            fabric: Fabric::Nvlink,
        }),
        norm,
        all_reduce_bytes_per_token: u64::from(hidden_bytes),
    };
    let attention = |shape: &MimoV2AttnShape,
                     window: u32,
                     page_size: u32|
     -> std::result::Result<_, BuildError> {
        Ok(MimoV2AttnLocalWorkletConfig {
            hidden: model.hidden.into(),
            num_q_heads: divide("attention heads", shape.num_attention_heads)?,
            num_kv_heads: kv_heads_per_rank(shape, tp)?,
            head_dim: shape.head_dim,
            v_head_dim: shape.v_head_dim,
            rotary_dim: model.rotary_dim,
            window,
            sinks: shape.sinks,
            page_size,
            quant_group_size: model.fp8_block_size,
            max_model_len: parallel.max_model_len,
            cuda_graph,
            dtype: ACTIVATION_DTYPE,
            gpu_name: gpu.clone(),
            quant_backends: FP8_QUANT_BACKENDS.to_vec(),
            fp8_gemm_backends: FP8_GEMM_BACKENDS.to_vec(),
            rope_backends: TRITON_ELEMENTWISE_BACKENDS.to_vec(),
            kv_append_backends: KV_APPEND_BACKENDS.to_vec(),
            attention_backends: ATTENTION_BACKENDS.to_vec(),
            o_proj_backends: BF16_GEMM_BACKENDS.to_vec(),
        })
    };
    divide("n_routed_experts", model.n_routed_experts)?;
    let routed = Nvfp4MoeLocalWorkletConfig::split_for_ep(
        Nvfp4MoeLocalWorkletConfig {
            hidden: model.hidden.into(),
            moe_intermediate: model.moe_intermediate_size.into(),
            num_experts: model.n_routed_experts.into(),
            ep_size: parallel.tp_size,
            tp_size: 1,
            top_k: model.num_experts_per_tok,
            activation_dtype: ACTIVATION_DTYPE,
            gpu_name: gpu.clone(),
            quant_backends: Vec::new(),
            moe_backends: ROUTED_MOE_BACKENDS.to_vec(),
            weight_format: DType::Mxfp4E2m1,
            activation_format: Some(ACTIVATION_DTYPE),
            group_size: model.mxfp4_block_size,
            routing_method: ROUTING_METHOD.to_string(),
            n_group: model.n_group,
            topk_group: model.topk_group,
            routed_scaling_numerator: model.routed_scaling.0,
            routed_scaling_denominator: model.routed_scaling.1,
            expert_demand: demand.clone(),
            folded_rank_position: 0,
        },
        demand.clone(),
    );
    Ok(MimoV2VllmConfigs {
        groups: layer_groups(model),
        attn_global: attention(&model.global_attn, 0, global_page)?,
        attn_swa: attention(&model.swa_attn, model.sliding_window_size, swa_page)?,
        dense_ffn: Glm53MlpLocalWorkletConfig {
            hidden: model.hidden.into(),
            intermediate: divide("intermediate_size", model.dense_intermediate_size)?.into(),
            precision: Glm53WeightPrecision::Fp8Block,
            activation_dtype: ACTIVATION_DTYPE,
            gpu_name: gpu.clone(),
            quant_backends: FP8_QUANT_BACKENDS.to_vec(),
            gemm_backends: FP8_GEMM_BACKENDS.to_vec(),
            elementwise_backends: TRITON_ELEMENTWISE_BACKENDS.to_vec(),
        },
        router: GemmFp32OutputKernelConfig {
            backends: ROUTER_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            n: model.n_routed_experts.into(),
            k: model.hidden.into(),
            input_dtype: ACTIVATION_DTYPE,
        },
        routed,
        // The add + RMSNorm reads the all-reduced row and the residual and
        // writes both back.
        ar_norm: tp_boundary(ew(2 * hidden_bytes, 2 * hidden_bytes)),
        embed_ar_norm: tp_boundary(ew(hidden_bytes, hidden_bytes)),
        // `vocab_parallel_embedding`: an int32 id and the gathered row in, the
        // masked row out.
        embed: ew(hidden_bytes + INT32_TOKEN_ID_BYTES, hidden_bytes),
        // One hidden row and its int64 index in, the row out.
        logits_select: ew(hidden_bytes + INT64_INDEX_BYTES, hidden_bytes),
        lm_head: SingleGemmKernelConfig {
            backends: BF16_GEMM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            n: Dim::param("vocab_per_rank", divide("vocab_size", model.vocab_size)?),
            k: model.hidden.into(),
            dtype: ACTIVATION_DTYPE,
        },
        // `tensor_model_parallel_all_gather` of the bf16 logits; the NCCL
        // all-reduce curve stands in (as Inkling's and DeepSeek V4.1's heads).
        logits_all_gather: (tp > 1).then(|| AllReduceKernelConfig {
            backends: LOGITS_ALL_GATHER_BACKENDS.to_vec(),
            gpu_name: gpu,
            num_gpus: tp,
            fabric: Fabric::Nvlink,
        }),
        model: model.clone(),
        parallel: parallel.clone(),
    })
}

#[derive(Clone, Debug)]
pub struct MimoV2VllmResolved {
    pub raw_cfg: MimoV2VllmConfigs,
    pub attn_global: MimoV2AttnLocalWorkletResolved,
    pub attn_swa: MimoV2AttnLocalWorkletResolved,
    pub dense_ffn: Glm53MlpLocalWorkletResolved,
    pub routed: Vec<Nvfp4MoeLocalWorkletResolved>,
}

pub fn resolve_configs(cfgs: &MimoV2VllmConfigs) -> MimoV2VllmResolved {
    MimoV2VllmResolved {
        attn_global: MimoV2AttnLocalWorklet::resolve_config(&cfgs.attn_global),
        attn_swa: MimoV2AttnLocalWorklet::resolve_config(&cfgs.attn_swa),
        dense_ffn: Glm53MlpLocalWorklet::resolve_config(&cfgs.dense_ffn),
        routed: cfgs
            .routed
            .iter()
            .map(Nvfp4MoeLocalWorklet::resolve_config)
            .collect(),
        raw_cfg: cfgs.clone(),
    }
}

/// The replicated router, then each EP rank's routed call over every token;
/// the slowest rank closes the section.
struct MoeBlock {
    name: String,
    router: Op<GemmFp32OutputKernel>,
    routed: Vec<Nvfp4MoeLocalWorklet>,
}

impl MoeBlock {
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let router = self.router.compile(builder);
        let ranks = self
            .routed
            .iter()
            .map(|routed| routed.compile(builder))
            .collect();
        CostNode::Labeled {
            label: format!("{} (MoE) [EP{} ranked ranks]", self.name, self.routed.len()),
            child: Box::new(CostNode::Sum(vec![
                router,
                CostNode::Labeled {
                    label: format!("{}.ep_ranks (max over EP ranks)", self.name),
                    child: Box::new(CostNode::Max {
                        overlap: 1.0,
                        children: ranks,
                    }),
                },
            ])),
        }
    }

    fn eval(&self, num_tokens: u32, ev: &mut Evaluator) {
        push_or_zero(
            &self.router,
            GemmFp32OutputKernelInput { m: num_tokens },
            false,
            ev,
        );
        for routed in &self.routed {
            routed.eval(&Nvfp4MoeLocalWorkletInput { num_tokens }, ev);
        }
    }
}

enum Ffn {
    Dense(Glm53MlpLocalWorklet),
    Moe(MoeBlock),
}

/// One built layer group: the leaves of a single layer of its shape.
struct LayerGroup {
    label: String,
    layers: Vec<u32>,
    attn: MimoV2AttnLocalWorklet,
    attn_ar_norm: TpAllReduceRmsNormOp,
    ffn: Ffn,
    ffn_ar_norm: TpAllReduceRmsNormOp,
}

impl LayerGroup {
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let layer = CostNode::Sum(vec![
            self.attn.compile(builder),
            self.attn_ar_norm.compile(builder),
            match &self.ffn {
                Ffn::Dense(mlp) => mlp.compile(builder),
                Ffn::Moe(block) => block.compile(builder),
            },
            self.ffn_ar_norm.compile(builder),
        ]);
        CostNode::Labeled {
            label: format!("{} layers {:?}", self.label, self.layers),
            child: Box::new(CostNode::Scale {
                n: self.layers.len() as u32,
                child: Box::new(layer),
            }),
        }
    }

    fn eval(&self, batch: &NormalizedBatch, tokens: u32, ev: &mut Evaluator) {
        self.attn.eval(
            &MimoV2AttnLocalWorkletInput {
                num_tokens: tokens,
                query_context_pairs: batch.query_context_pairs.clone(),
                padded_requests: attention_padded_requests(batch, tokens),
            },
            ev,
        );
        self.attn_ar_norm.eval(tokens, ev);
        match &self.ffn {
            Ffn::Dense(mlp) => mlp.eval(&Glm53MlpLocalWorkletInput { num_tokens: tokens }, ev),
            Ffn::Moe(block) => block.eval(tokens, ev),
        }
        self.ffn_ar_norm.eval(tokens, ev);
    }
}

fn build_ar_norm(
    prefix: &str,
    config: &TpAllReduceRmsNormConfig,
    bridge: &PerfApiBridge,
) -> std::result::Result<TpAllReduceRmsNormOp, BuildError> {
    TpAllReduceRmsNormOp::build(&format!("{prefix}.ar_norm"), config.clone(), bridge)
}

fn build_layer_group(
    name: &str,
    group: &MimoV2LayerGroup,
    resolved: &MimoV2VllmResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<LayerGroup, BuildError> {
    let cfg = &resolved.raw_cfg;
    let prefix = format!("{name}.{}", group.label);
    let p = prefix.as_str();
    let attn = MimoV2AttnLocalWorklet::build(
        format!("{p}.attn"),
        match group.attn {
            AttnKind::Global => resolved.attn_global.clone(),
            AttnKind::SlidingWindow => resolved.attn_swa.clone(),
        },
        bridge,
    )?;
    let attn_ar_norm = build_ar_norm(&format!("{p}.attn"), &cfg.ar_norm, bridge)?;
    let (ffn, ffn_name) = match group.ffn {
        FfnKind::Dense => {
            let ffn = format!("{p}.ffn");
            (
                Ffn::Dense(Glm53MlpLocalWorklet::build(
                    ffn.clone(),
                    resolved.dense_ffn.clone(),
                    bridge,
                )?),
                ffn,
            )
        }
        FfnKind::Moe => {
            let moe = format!("{p}.moe");
            let routed = resolved
                .routed
                .iter()
                .enumerate()
                .map(|(rank, r)| {
                    Nvfp4MoeLocalWorklet::build(
                        format!("{moe}.routed_rank{rank}"),
                        r.clone(),
                        bridge,
                    )
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            (
                Ffn::Moe(MoeBlock {
                    router: atomic(
                        &moe,
                        "router",
                        cfg.router.clone(),
                        GemmFp32OutputKernel::build,
                        bridge,
                    )?,
                    routed,
                    name: moe.clone(),
                }),
                moe,
            )
        }
    };
    Ok(LayerGroup {
        label: group.label.clone(),
        layers: group.layers.clone(),
        attn,
        attn_ar_norm,
        ffn,
        ffn_ar_norm: build_ar_norm(&ffn_name, &cfg.ar_norm, bridge)?,
    })
}

pub struct MimoV2VllmModel {
    pub name: String,
    pub tp_size: u16,
    pub max_model_len: u32,
    /// Sorted, deduplicated capture sizes.
    cudagraph_capture_sizes: Vec<u32>,
    embed: Op<ElementwiseKernel>,
    embed_ar_norm: TpAllReduceRmsNormOp,
    groups: Vec<LayerGroup>,
    logits_select: Op<ElementwiseKernel>,
    lm_head: Op<SingleGemmKernel>,
    logits_all_gather: Option<Op<AllReduceKernel>>,
    vocab_size: u32,
    kv_cache_layout: KvCacheLayout,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: MimoV2VllmResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<MimoV2VllmModel, BuildError> {
    let cfg = &resolved.raw_cfg;
    let n = name.as_str();
    let groups = cfg
        .groups
        .iter()
        .map(|group| build_layer_group(n, group, &resolved, bridge))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut model = MimoV2VllmModel {
        embed: atomic(
            n,
            "embed",
            cfg.embed.clone(),
            ElementwiseKernel::build,
            bridge,
        )?,
        embed_ar_norm: build_ar_norm(&format!("{n}.embed"), &cfg.embed_ar_norm, bridge)?,
        groups,
        logits_select: atomic(
            n,
            "logits_select",
            cfg.logits_select.clone(),
            ElementwiseKernel::build,
            bridge,
        )?,
        lm_head: atomic(
            n,
            "lm_head",
            cfg.lm_head.clone(),
            SingleGemmKernel::build,
            bridge,
        )?,
        logits_all_gather: cfg
            .logits_all_gather
            .clone()
            .map(|config| {
                atomic(
                    n,
                    "logits_all_gather",
                    config,
                    AllReduceKernel::build,
                    bridge,
                )
            })
            .transpose()?,
        vocab_size: cfg.model.vocab_size,
        kv_cache_layout: kv_cache_layout(&cfg.model, cfg.parallel.tp_size)?,
        tp_size: cfg.parallel.tp_size,
        max_model_len: cfg.parallel.max_model_len,
        cudagraph_capture_sizes: {
            let mut sizes = cfg.parallel.cudagraph_capture_sizes.clone();
            sizes.sort_unstable();
            sizes.dedup();
            sizes
        },
        cost_flat: Vec::new(),
        n_slots: 0,
        name,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

/// vLLM's KV cache group size for layer types of these layer counts
/// (`_get_kv_cache_groups_uniform_page_size`): the smallest count, or the
/// largest when it is under 1.5x the smallest. Each type then fills
/// `ceil(layers / group_size)` groups, its last one padded with empty layers
/// that still take pages.
fn kv_group_size(layer_counts: &[u64]) -> u64 {
    let smallest = layer_counts.iter().copied().min().unwrap_or(1).max(1);
    let largest = layer_counts.iter().copied().max().unwrap_or(1);
    if (largest as f64) < smallest as f64 * 1.5 {
        largest
    } else {
        smallest
    }
}

/// One request's cache over the TP group: K and V of the global layers for
/// every token, and K and V of the sliding-window layers for the last
/// `sliding_window_size` tokens.
///
/// vLLM's hybrid allocator groups the layers of each type into equal-size KV
/// cache groups that share one block pool ([`kv_group_size`]); the padding
/// layers of a type's last group are allocated like real ones, so they are the
/// part's padded bytes. At TP4 the 9 global layers form one group and the 39
/// window layers five (45 layer slots).
fn kv_cache_layout(
    model: &MimoV2ModelCfg,
    tp_size: u16,
) -> std::result::Result<KvCacheLayout, BuildError> {
    let tp = u64::from(tp_size);
    let elem = u64::from(ACTIVATION_DTYPE.size_bytes());
    let layer_bytes = |shape: &MimoV2AttnShape| {
        gqa_kv_bytes_per_token(
            1,
            u64::from(shape.num_key_value_heads),
            u64::from(shape.head_dim),
            u64::from(shape.v_head_dim),
            elem,
            tp,
        )
    };
    let swa_layers = u64::from(model.swa_layer_ids.len() as u32);
    let global_layers = u64::from(model.num_layers) - swa_layers;
    let counts: Vec<u64> = [global_layers, swa_layers]
        .into_iter()
        .filter(|&count| count > 0)
        .collect();
    let group = kv_group_size(&counts);
    let padded = |layers: u64| layers.div_ceil(group) * group;
    let mut layout = KvCacheLayout::new();
    if global_layers > 0 {
        let bytes = layer_bytes(&model.global_attn);
        let mut part = KvCachePart::per_token("global_attn", global_layers * bytes);
        if padded(global_layers) > global_layers {
            part = part.padded_to(padded(global_layers) * bytes);
        }
        layout = layout.with_part(part);
    }
    if swa_layers > 0 {
        let bytes = layer_bytes(&model.swa_attn);
        let mut part =
            KvCachePart::sliding_window("swa_attn", model.sliding_window_size, swa_layers * bytes);
        if padded(swa_layers) > swa_layers {
            part = part.padded_to(padded(swa_layers) * bytes);
        }
        layout = layout.with_part(part);
    }
    if layout.parts.is_empty() {
        return Err(fit_failed("the model has no attention layers"));
    }
    Ok(layout)
}

impl MimoV2VllmModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let mut children = vec![
            self.embed.compile(&mut builder),
            self.embed_ar_norm.compile(&mut builder),
        ];
        for group in &self.groups {
            children.push(group.compile(&mut builder));
        }
        children.push(self.logits_select.compile(&mut builder));
        children.push(self.lm_head.compile(&mut builder));
        if let Some(all_gather) = &self.logits_all_gather {
            children.push(all_gather.compile(&mut builder));
        }
        let root = CostNode::Labeled {
            label: format!(
                "{} (MimoV2VllmModel) [TP=EP{}; timing_context<={}]",
                self.name, self.tp_size, self.max_model_len
            ),
            child: Box::new(CostNode::Sum(children)),
        };
        builder.finish(root)
    }

    fn eval_into(&self, input: &UnifiedArchInput, ev: &mut Evaluator) {
        let batch = self
            .normalize(input)
            .unwrap_or_else(|reason| panic!("invalid MimoV2VllmModel input: {reason}"));
        let tokens = graph_padded_tokens(&self.cudagraph_capture_sizes, batch.total_tokens);
        push_or_zero(
            &self.embed,
            ElementwiseKernelInput { num_tokens: tokens },
            false,
            ev,
        );
        self.embed_ar_norm.eval(tokens, ev);
        for group in &self.groups {
            group.eval(&batch, tokens, ev);
        }
        let logits = batch.request_count;
        push_or_zero(
            &self.logits_select,
            ElementwiseKernelInput { num_tokens: logits },
            false,
            ev,
        );
        push_or_zero(
            &self.lm_head,
            SingleGemmKernelInput { m: logits },
            false,
            ev,
        );
        if let Some(all_gather) = &self.logits_all_gather {
            let gathered = u64::from(logits) * u64::from(self.vocab_size) * 2;
            push_or_zero(
                all_gather,
                AllReduceKernelInput {
                    message_size_bytes: all_gather_as_all_reduce_bytes(gathered),
                },
                false,
                ev,
            );
        }
    }

    fn normalize(&self, input: &UnifiedArchInput) -> std::result::Result<NormalizedBatch, String> {
        normalize_input(input, self.max_model_len)
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

impl IterwiseUnifiedModel for MimoV2VllmModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        self.normalize(batch).map(|_| ())
    }

    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout.clone()
    }

    fn gpus_per_replica(&self) -> u16 {
        self.tp_size
    }

    fn num_attn_shards(&self) -> u16 {
        self.tp_size
    }

    /// Sliding-window attention makes necessary work per-request in context
    /// (`min(context, window)`), which a decode KV total cannot recover.
    fn logs_decode_kv_lens(&self) -> bool {
        true
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

/// The request count attention is launched over. A uniform-decode step that
/// fits a captured size replays a FULL graph, whose attention metadata the
/// model runner pads to the capture size with zero-length requests
/// (`pad_attn`, `num_reqs_padded` in gpu_model_runner.py); FA4's split count
/// follows that size. Steps with a prefill chunk replay a piecewise graph over
/// the real requests (0), and a step past the largest size runs eagerly
/// (`padded_tokens` is then the real count); both mean unpadded.
fn attention_padded_requests(batch: &NormalizedBatch, padded_tokens: u32) -> u32 {
    if batch.is_uniform_decode() {
        padded_tokens
    } else {
        0
    }
}

/// One iteration, as every layer sees it (one TP group).
struct NormalizedBatch {
    total_tokens: u32,
    request_count: u32,
    /// Real requests as `(query_len, context_len)`, prefill chunks first.
    query_context_pairs: Vec<(u32, u32)>,
}

impl NormalizedBatch {
    fn is_uniform_decode(&self) -> bool {
        self.query_context_pairs
            .iter()
            .all(|&(query, _)| query == 1)
    }
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
    let mut pairs =
        Vec::with_capacity(group.prefill_chunk_pairs.len() + group.decode_kv_lens.len());
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
                "prefill request {request} context {context} exceeds max_model_len {max_model_len}"
            ));
        }
        prefill_tokens = prefill_tokens
            .checked_add(append)
            .ok_or("prefill token sum overflows u32")?;
        pairs.push((append, context));
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
        pairs.push((1, context));
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
        query_context_pairs: pairs,
    })
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
    use crate::timing::kernels::AllReduceResidualRmsNormSpec;
    use crate::timing::routing::RoutingDistribution;
    use std::path::Path;

    fn model_cfg() -> MimoV2ModelCfg {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/mimo_v2_6_flash_rl.json");
        MimoV2ModelCfg::from_json(&path).unwrap()
    }

    /// The capture's `cudagraph_capture_sizes` (max_num_seqs 100).
    fn capture_sizes() -> Vec<u32> {
        let mut sizes = vec![1, 2, 4];
        sizes.extend((8..=200).step_by(8));
        sizes.push(100);
        sizes
    }

    fn parallel() -> MimoV2VllmParallel {
        MimoV2VllmParallel {
            tp_size: 4,
            max_model_len: 131072,
            gpu_name: "NVIDIA B200".into(),
            cudagraph_capture_sizes: capture_sizes(),
        }
    }

    fn demand() -> ExpertDemand {
        ExpertDemand::popularity(&RoutingDistribution::uniform(256), 7)
    }

    fn built() -> MimoV2VllmModel {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let configs = build_configs(&model_cfg(), &parallel(), &demand()).unwrap();
        build("unified".into(), resolve_configs(&configs), &bridge).unwrap()
    }

    #[test]
    fn kv_cache_layout_reproduces_vllms_logged_concurrency() {
        use crate::worker::kv::{KvAccounting, KvAccountingOptions};
        // "Maximum concurrency for 131,072 tokens per request" at TP4 with two
        // async batches of 2048 tokens in flight: the draft capture (110.52
        // GiB per rank) and the fused-all-reduce capture 6264 (110.07 GiB).
        let layout = kv_cache_layout(&model_cfg(), 4).unwrap();
        for (gib_per_rank, logged) in [(110.52, 118.76), (110.07, 118.28)] {
            let budget = (gib_per_rank * f64::from(1u32 << 30)) as u64 * 4;
            let options = KvAccountingOptions {
                max_in_flight_tokens: Some(2 * 2048),
                ..Default::default()
            };
            let accounting = KvAccounting::resolve(&layout, options, budget).unwrap();
            let concurrency = accounting.concurrency_at(131072);
            assert!(
                (concurrency / logged - 1.0).abs() < 0.005,
                "{gib_per_rank} GiB: {concurrency} vs {logged}"
            );
        }
    }

    #[test]
    fn kv_layout_pads_the_window_layers_to_whole_groups() {
        // 9 global layers x 4 KV heads and 39 window layers x 8 KV heads, each
        // head 192 + 128 bf16 channels; the window layers fill five groups of
        // nine.
        let layout = kv_cache_layout(&model_cfg(), 4).unwrap();
        let [global, swa] = layout.parts.as_slice() else {
            panic!("two parts expected");
        };
        assert_eq!(global.bytes_per_entry, 9 * 4 * 320 * 2);
        assert_eq!(global.overhead.padded_bytes_per_entry, None);
        assert_eq!(swa.bytes_per_entry, 39 * 8 * 320 * 2);
        assert_eq!(swa.overhead.padded_bytes_per_entry, Some(45 * 8 * 320 * 2));
        assert_eq!(kv_page_sizes(&model_cfg(), 4).unwrap(), (32, 16));
        assert_eq!(kv_group_size(&[12, 13]), 13);
    }

    #[test]
    fn layer_groups_follow_the_attention_and_ffn_schedule() {
        let summary: Vec<(String, usize)> = layer_groups(&model_cfg())
            .into_iter()
            .map(|g| (g.label, g.layers.len()))
            .collect();
        assert_eq!(
            summary,
            [
                ("global_dense".to_string(), 1),
                ("swa_moe".to_string(), 39),
                ("global_moe".to_string(), 8)
            ]
        );
    }

    #[test]
    fn configs_carry_the_capture_shapes() {
        let configs = build_configs(&model_cfg(), &parallel(), &demand()).unwrap();
        assert_eq!(configs.routed.len(), 4);
        let rank = Nvfp4MoeLocalWorklet::resolve_config(&configs.routed[0]);
        assert!(rank.quant.is_none());
        assert_eq!(rank.fused_moe.activation_format, Some(DType::Bf16));
        assert_eq!(rank.fused_moe.num_local_experts.get(), 64);
        assert_eq!(rank.fused_moe.group_size, 32);
        // Layer 0's dense MLP at TP4: 16384 / 4 = 4096 per rank, FP8 block.
        let dense = Glm53MlpLocalWorklet::resolve_config(&configs.dense_ffn);
        assert_eq!((dense.gate_up.n.get(), dense.gate_up.k.get()), (8192, 4096));
        assert_eq!((dense.down.n.get(), dense.down.k.get()), (4096, 4096));
        assert!(dense.gate_up_quant.is_some() && dense.down_quant.is_some());
        assert_eq!(configs.lm_head.n.get(), 38144);
        assert_eq!(
            (configs.router.n.get(), configs.router.k.get()),
            (256, 4096)
        );
        assert_eq!(
            (
                configs.attn_global.num_kv_heads,
                configs.attn_global.page_size
            ),
            (1, 32)
        );
        assert_eq!(
            (configs.attn_swa.num_kv_heads, configs.attn_swa.page_size),
            (2, 16)
        );
        assert!(configs.attn_swa.cuda_graph);
        assert_eq!(
            (
                configs.embed.input_bytes_per_token.get(),
                configs.logits_select.input_bytes_per_token.get()
            ),
            (8196, 8200)
        );
        assert_eq!(
            AllReduceResidualRmsNormSpec::max_fused_tokens(configs.ar_norm.fused.as_ref().unwrap()),
            4096
        );
        assert_eq!(
            (
                configs.ar_norm.norm.input_bytes_per_token.get(),
                configs.embed_ar_norm.norm.input_bytes_per_token.get()
            ),
            (16384, 8192)
        );
    }

    #[test]
    fn manifest_charges_two_fused_all_reduces_per_layer_plus_the_embedding() {
        let model = built();
        let manifest = model.cost_log_manifest();
        let ar_norms: Vec<&str> = manifest
            .slots
            .iter()
            .filter(|slot| slot.kind == "all_reduce_residual_rms_norm")
            .map(|slot| slot.name.as_str())
            .collect();
        // One per sublayer of each folded group, plus the embedding's: with
        // the Scale folds that is 1 + 2 * 48 = 97 calls per iteration.
        assert_eq!(ar_norms.len(), 1 + 2 * 3);
        assert!(ar_norms.contains(&"unified.embed.ar_norm"));
        assert!(manifest
            .slots
            .iter()
            .any(|slot| slot.name == "unified.swa_moe.moe.routed_rank3.fused_moe"));
        assert!(!manifest
            .slots
            .iter()
            .any(|slot| slot.name.ends_with("input_quant") && slot.name.contains(".moe.")));
    }

    fn step(prefill_chunk_pairs: Vec<(u32, u32)>, decode_kv_lens: Vec<u32>) -> UnifiedArchInput {
        let prefill_tokens = prefill_chunk_pairs
            .iter()
            .map(|&(_, append)| append)
            .sum::<u32>();
        let decode_tokens = decode_kv_lens.len() as u32;
        UnifiedArchInput {
            groups: vec![ArchGroupInput {
                batch_tokens: prefill_tokens + decode_tokens,
                prefill_tokens,
                decode_tokens,
                prefill_chunk_pairs,
                decode_kv_lens,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn full_graph_decode_launches_attention_over_the_capture_size() {
        let sizes = {
            let mut sizes = capture_sizes();
            sizes.sort_unstable();
            sizes.dedup();
            sizes
        };
        let padded = |input: UnifiedArchInput| {
            let batch = normalize_input(&input, 131072).unwrap();
            let tokens = graph_padded_tokens(&sizes, batch.total_tokens);
            attention_padded_requests(&batch, tokens)
        };
        assert_eq!(padded(step(vec![], vec![400; 97])), 100);
        assert_eq!(padded(step(vec![], vec![400; 3])), 4);
        assert_eq!(padded(step(vec![(0, 50)], vec![400; 7])), 0);
        // Past the largest captured size a step runs eagerly, unpadded.
        assert_eq!(padded(step(vec![], vec![400; 201])), 201);
    }

    /// Up to the fused cap every boundary is the one FlashInfer call; above it
    /// (vLLM's unfused compile range) each is an all-reduce and a residual
    /// RMSNorm.
    #[test]
    fn a_step_past_the_fused_all_reduce_cap_runs_the_unfused_boundary() {
        let model = built();
        assert!(model.check_input(&step(vec![(0, 8192)], vec![])).is_ok());
        assert!(model.check_input(&step(vec![], vec![131073])).is_err());
        assert!(model.embed_ar_norm.takes_fused(4096));
        assert!(!model.embed_ar_norm.takes_fused(4097));
        let names: Vec<String> = model
            .cost_log_manifest()
            .slots
            .into_iter()
            .map(|slot| slot.name)
            .collect();
        for suffix in ["ar_norm", "ar_norm_all_reduce", "ar_norm_residual_norm"] {
            assert!(
                names.contains(&format!("unified.swa_moe.moe.{suffix}")),
                "{suffix}"
            );
        }
    }

    /// vLLM does not fuse at TP 1 and reduces nothing there: each boundary is
    /// only the residual RMSNorm, and the head gathers nothing.
    #[test]
    fn one_gpu_builds_without_collectives() {
        let mut layout = parallel();
        layout.tp_size = 1;
        let configs = build_configs(&model_cfg(), &layout, &demand()).unwrap();
        assert!(configs.ar_norm.fused.is_none() && configs.ar_norm.all_reduce.is_none());
        assert!(configs.logits_all_gather.is_none());
    }

    /// Every compiled location that is not a collective (the fused all-reduce
    /// + RMSNorm sites, the unfused all-reduces and the logits all-gather) has a row in the
    /// necessary-work map, and the map names nothing the arch does not compile.
    #[test]
    fn necessary_work_map_covers_the_compiled_locations() {
        let model = built();
        assert!(model.logs_decode_kv_lens());
        let actual: std::collections::BTreeSet<String> = model
            .cost_log_manifest()
            .slots
            .into_iter()
            .filter(|slot| {
                !matches!(
                    slot.kind.as_str(),
                    "all_reduce" | "all_reduce_residual_rms_norm"
                )
            })
            .map(|slot| slot.name)
            .collect();
        assert_eq!(actual.len(), 43);
        let map: serde_json::Value = serde_json::from_str(include_str!(
            "../../../model/work/location_maps/mimo_v2_vllm_fp8_swa_moe_tp4_ep4_unified.json"
        ))
        .unwrap();
        assert_eq!(
            map["mapping_id"],
            "mimo-v2-vllm-fp8-swa-moe-tp4-ep4-unified-v1"
        );
        assert_eq!(map["arch_types"][0], ARCH_KIND);
        let mapped: std::collections::BTreeSet<String> = map["locations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["location"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(actual, mapped);
    }
}
