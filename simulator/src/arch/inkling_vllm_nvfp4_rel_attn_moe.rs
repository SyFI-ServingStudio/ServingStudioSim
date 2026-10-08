//! Inkling (`InklingForConditionalGeneration`) ModelOpt NVFP4 checkpoints on
//! B200, aligned to the vLLM fork's TP / EP deployment (`ink/`, MTP, vision
//! and audio off). Measured against Inkling-Small-NVFP4 at TP4 + EP4
//! (`logs/20261007_0_inkling_small_nvfp4_tp4ep4`).
//!
//! One rank group of `tp_size` ranks. Attention, the dense FFN and the two
//! sink experts are tensor-parallel; the router is replicated; the routed
//! experts follow `--enable-expert-parallel` (on: `n_routed_experts / tp`
//! whole experts per rank; off: every expert sliced on the intermediate axis).
//! There is no stand-alone all-reduce or RMSNorm: every sublayer ends in one
//! Lamport call (`LamportRSConv.rs_sconv_ag_add_norm`: reduce-scatter, short
//! conv on the shard, all-gather, residual add and the next sublayer's norm;
//! the last layer's applies the final norm), 2 per layer. The MoE sublayer's
//! call also sums the sink partial (`has_shared`), since the routed MoE
//! skips its own reduction.
//!
//! Layers carry sliding-window (`local_layer_ids`) or global relative-position
//! attention, a dense FFN (`0..dense_mlp_idx`) or an MoE whose routed experts
//! are BF16 where the checkpoint excludes them (layer 2) and NVFP4 elsewhere.
//!
//! ```text
//! prologue       embed + embed_norm (+ layer 0's attn norm), log-scaling tau
//! per layer      attention (QKVR, prep, rel FA4, O) + Lamport,
//!                dense FFN or MoE + Lamport                    (Scale per group)
//! epilogue       lm_head over one row per request, logits all-gather
//! ```
//!
//! The MoE: router, gate select and the top-k copies on every rank, then a
//! `Max` over the EP ranks' ranked routed workloads; inside each rank the sink
//! experts run on vLLM's aux stream concurrently with the routed call when the
//! batch has at most 256 tokens, and serially after it otherwise
//! (`maybe_execute_in_parallel`, `ink/moe.py`). Both copies are compiled and
//! the inactive one is pushed as zeros. The routed call includes its NVFP4
//! activation quant. Layer 2's BF16 experts run FlashInfer CUTLASS in
//! production; the `bf16_fused_moe` trtllm-gen backend stands in for it.
//!
//! Sampling is not modeled.

use anyhow::Result;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::inkling_model_cfg::{InklingExpertPrecision, InklingModelCfg};
use crate::common::Fabric;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{
    AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput, Bf16FusedMoeKernelInput,
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, Nvfp4FusedMoeKernelInput,
    Nvfp4QuantKernelInput, RsSconvAgResidualRmsNormKernel, RsSconvAgResidualRmsNormKernelConfig,
    RsSconvAgResidualRmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Dim, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, Probe, SlotInput,
};
use crate::worklet::deepseek_v41_common::all_gather_as_all_reduce_bytes;
use crate::worklet::{
    Bf16MoeLocalWorklet, Bf16MoeLocalWorkletConfig, Bf16MoeLocalWorkletInput,
    Bf16MoeLocalWorkletResolved, InklingAttnLocalWorklet, InklingAttnLocalWorkletConfig,
    InklingAttnLocalWorkletInput, InklingAttnLocalWorkletResolved, InklingMlpLocalWorklet,
    InklingMlpLocalWorkletConfig, InklingMlpLocalWorkletInput, InklingMlpLocalWorkletResolved,
    InklingRouterLocalWorklet, InklingRouterLocalWorkletConfig, InklingRouterLocalWorkletInput,
    Nvfp4MoeLocalWorklet, Nvfp4MoeLocalWorkletConfig, Nvfp4MoeLocalWorkletInput,
    Nvfp4MoeLocalWorkletResolved,
};

const ARCH_KIND: &str = "inkling_vllm_nvfp4_rel_attn_moe";
const ACTIVATION_DTYPE: DType = DType::Bf16;
/// Attention KV page: vLLM's `unify_kv_cache_spec_page_size` doubles the
/// 16-token default so a page matches the 32 KiB conv-state page.
const PAGE_SIZE: u32 = 32;
/// `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`: the sinks run on the aux
/// stream at or below this batch size.
pub const SINK_STREAM_TOKEN_THRESHOLD: u32 = 256;
/// `Parallel{overlap}` of the aux-stream sinks against the routed call. The
/// capture's decode routed MoE (~75 us at T=32, graph-padded) is within 4%
/// of the worst isolated rank's row (72.4 us), so the contention is not
/// calibrated and the branch is a plain max.
const SINK_STREAM_OVERLAP: f32 = 1.0;
/// `compute_log_scaling_tau`: add, cast, mul, clamp, log, mul, add over [T].
const LOG_SCALING_TAU_LAUNCHES: u32 = 7;

const BF16_GEMM_BACKENDS: &[&str] = &["torch_linear_vllm"];
const QKVR_PREP_BACKENDS: &[&str] = &["vllm_triton"];
const REL_ATTENTION_BACKENDS: &[&str] = &["tml_fa4"];
const LAMPORT_BACKENDS: &[&str] = &["vllm_lamport_triton"];
/// The fork's `_linear_with_fp32_out`: cute-DSL `ll_bf16` at T <= 64, cuBLAS
/// `torch.mm(out_dtype=fp32)` above.
const ROUTER_BACKENDS: &[&str] = &["vllm_inkling_router_auto"];
const GATE_SELECT_BACKENDS: &[&str] = &["vllm_triton"];
/// Triton launches (embed + norm, `_silu_and_mul_triton`, `_sink_epilogue`).
const TRITON_ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
/// Eager torch launches (top-k copies, tau, the dense `global_scale` mul).
const TORCH_ELEMENTWISE_BACKENDS: &[&str] = &["torch"];
const NVFP4_QUANT_BACKENDS: &[&str] = &["vllm_cuda"];
/// `trtllm_fp4_block_scale_routed_moe` on the fork env's FlashInfer 0.7.0,
/// the capture's version; the pinned vLLM image's 0.6.18 picks a slower FC1
/// tile at T >= ~1792 (+70% at 2048).
const NVFP4_ROUTED_MOE_BACKENDS: &[&str] = &["flashinfer_trtllm_routed_sm100_fork"];
/// Proxy for FlashInfer CUTLASS (`cutlass_fused_moe`), which the BF16 layer
/// runs because TRT-LLM BF16 rejects custom routing; same math and shapes.
const BF16_ROUTED_MOE_BACKENDS: &[&str] = &["flashinfer_trtllm_sm100"];
/// Key-only routing tag of the precomputed-routing rows (the fork's custom
/// routing never reaches the kernel as a method).
const ROUTING_METHOD: &str = "minimax2";
const NVFP4_GROUP_SIZE: u32 = 16;
const LOGITS_ALL_GATHER_BACKENDS: &[&str] = &["nccl"];

/// Deployment layout: one tensor-parallel rank group.
#[derive(Clone, Debug)]
pub struct InklingVllmParallel {
    pub tp_size: u16,
    /// vLLM `--enable-expert-parallel`.
    pub enable_expert_parallel: bool,
    pub max_model_len: u32,
    pub gpu_name: String,
    /// vLLM `--cudagraph-capture-sizes`; empty runs eager (no padding).
    pub cudagraph_capture_sizes: Vec<u32>,
}

/// The routed demand for each expert precision's layers, each folded over
/// exactly those layers.
#[derive(Clone, Debug, Default)]
pub struct InklingExpertDemand {
    pub nvfp4: Option<ExpertDemand>,
    pub bf16: Option<ExpertDemand>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttnKind {
    Local,
    Global,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FfnKind {
    Dense,
    Moe(InklingExpertPrecision),
}

/// A run of layers with one identical sublayer shape, folded by `Scale{n}`.
#[derive(Clone, Debug)]
pub struct InklingLayerGroup {
    pub label: String,
    pub layers: Vec<u32>,
    attn: AttnKind,
    ffn: FfnKind,
}

fn layer_groups(model: &InklingModelCfg) -> Vec<InklingLayerGroup> {
    let mut groups: Vec<InklingLayerGroup> = Vec::new();
    for layer in 0..model.num_layers {
        let attn = if model.is_local(layer) {
            AttnKind::Local
        } else {
            AttnKind::Global
        };
        let ffn = match model.expert_precision(layer) {
            None => FfnKind::Dense,
            Some(precision) => FfnKind::Moe(precision),
        };
        if let Some(group) = groups.iter_mut().find(|g| g.attn == attn && g.ffn == ffn) {
            group.layers.push(layer);
            continue;
        }
        let label = format!(
            "{}_{}",
            match attn {
                AttnKind::Local => "local",
                AttnKind::Global => "global",
            },
            match ffn {
                FfnKind::Dense => "dense",
                FfnKind::Moe(InklingExpertPrecision::Bf16) => "bf16_moe",
                FfnKind::Moe(InklingExpertPrecision::Nvfp4) => "nvfp4_moe",
            }
        );
        groups.push(InklingLayerGroup {
            label,
            layers: vec![layer],
            attn,
            ffn,
        });
    }
    groups
}

/// Every configuration the model is built from.
#[derive(Clone, Debug)]
pub struct InklingVllmConfigs {
    pub model: InklingModelCfg,
    pub parallel: InklingVllmParallel,
    pub groups: Vec<InklingLayerGroup>,
    pub attn_local: InklingAttnLocalWorkletConfig,
    pub attn_global: InklingAttnLocalWorkletConfig,
    pub dense_ffn: InklingMlpLocalWorkletConfig,
    pub sink: InklingMlpLocalWorkletConfig,
    pub router: InklingRouterLocalWorkletConfig,
    /// One per ranked EP workload (one under MoE TP); empty without NVFP4
    /// layers.
    pub routed_nvfp4: Vec<Nvfp4MoeLocalWorkletConfig>,
    pub routed_bf16: Vec<Bf16MoeLocalWorkletConfig>,
    /// Lamport after attention and the dense FFN (no sink partial).
    pub reduce: RsSconvAgResidualRmsNormKernelConfig,
    /// Lamport after the MoE, summing the sink partial.
    pub moe_reduce: RsSconvAgResidualRmsNormKernelConfig,
    pub embed: ElementwiseKernelConfig,
    pub log_scaling_tau: ElementwiseKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub logits_all_gather: AllReduceKernelConfig,
}

pub fn build_configs(
    model: &InklingModelCfg,
    parallel: &InklingVllmParallel,
    demand: &InklingExpertDemand,
) -> std::result::Result<InklingVllmConfigs, BuildError> {
    let tp = u32::from(parallel.tp_size);
    if tp < 2 {
        // Every sublayer boundary is the Lamport collective; TP1 runs a
        // different (unfused) sconv + norm path that is not modeled.
        return Err(fit_failed(format!(
            "tp_size {tp}: the Lamport RS + sconv + AG boundary needs at least 2 ranks"
        )));
    }
    let divide = |what: &str, value: u32| -> std::result::Result<u32, BuildError> {
        if value % tp != 0 {
            return Err(fit_failed(format!("{what} {value} must divide by TP {tp}")));
        }
        Ok(value / tp)
    };
    let gpu = parallel.gpu_name.clone();
    let hidden_bytes = model.hidden * ACTIVATION_DTYPE.size_bytes();
    let ew = |backends: &[&'static str], input: u32, output: u32| ElementwiseKernelConfig {
        backends: backends.to_vec(),
        gpu_name: gpu.clone(),
        input_bytes_per_token: input.into(),
        output_bytes_per_token: output.into(),
    };
    let attention = |is_local: bool| -> std::result::Result<_, BuildError> {
        let (q_heads, kv_heads, head_dim, rel_extent) = if is_local {
            (
                model.swa_num_attention_heads,
                model.swa_num_key_value_heads,
                model.swa_head_dim,
                model.sliding_window_size,
            )
        } else {
            (
                model.num_attention_heads,
                model.num_key_value_heads,
                model.head_dim,
                model.rel_extent,
            )
        };
        Ok(InklingAttnLocalWorkletConfig {
            hidden: model.hidden.into(),
            num_q_heads: divide("attention heads", q_heads)?,
            num_kv_heads: divide("KV heads", kv_heads)?,
            head_dim,
            d_rel: model.d_rel,
            is_local,
            rel_extent,
            sconv_kernel_size: model.sconv_kernel_size,
            page_size: PAGE_SIZE,
            max_model_len: parallel.max_model_len,
            dtype: ACTIVATION_DTYPE,
            gpu_name: gpu.clone(),
            gemm_backends: BF16_GEMM_BACKENDS.to_vec(),
            prep_backends: QKVR_PREP_BACKENDS.to_vec(),
            attention_backends: REL_ATTENTION_BACKENDS.to_vec(),
        })
    };
    let mlp = |intermediate: u32, global_scale_mul: bool| InklingMlpLocalWorkletConfig {
        hidden: model.hidden.into(),
        intermediate: intermediate.into(),
        global_scale_mul,
        dtype: ACTIVATION_DTYPE,
        gpu_name: gpu.clone(),
        gemm_backends: BF16_GEMM_BACKENDS.to_vec(),
        activation_backends: TRITON_ELEMENTWISE_BACKENDS.to_vec(),
        scale_backends: TORCH_ELEMENTWISE_BACKENDS.to_vec(),
    };
    let lamport = |has_shared: bool| RsSconvAgResidualRmsNormKernelConfig {
        backends: LAMPORT_BACKENDS.to_vec(),
        gpu_name: gpu.clone(),
        num_gpus: tp,
        hidden_dim: model.hidden,
        dtype: ACTIVATION_DTYPE,
        fabric: Fabric::Nvlink,
        conv_kernel_size: model.sconv_kernel_size,
        has_shared,
    };
    let (ep_size, moe_tp) = if parallel.enable_expert_parallel {
        divide("n_routed_experts", model.n_routed_experts)?;
        (parallel.tp_size, 1)
    } else {
        divide("intermediate_size", model.intermediate_size)?;
        (1, parallel.tp_size)
    };
    let demand_for = |precision: InklingExpertPrecision,
                      supplied: &Option<ExpertDemand>|
     -> std::result::Result<Option<ExpertDemand>, BuildError> {
        let present = model.expert_precision.contains(&precision);
        match (present, supplied) {
            (false, _) => Ok(None),
            (true, Some(demand)) => Ok(Some(demand.clone())),
            (true, None) => Err(fit_failed(format!(
                "no routed demand was supplied for the {precision:?} MoE layers"
            ))),
        }
    };
    let routed_nvfp4 = match demand_for(InklingExpertPrecision::Nvfp4, &demand.nvfp4)? {
        None => Vec::new(),
        Some(demand) => {
            let template = Nvfp4MoeLocalWorkletConfig {
                hidden: model.hidden.into(),
                moe_intermediate: model.intermediate_size.into(),
                num_experts: model.n_routed_experts.into(),
                ep_size,
                tp_size: moe_tp,
                top_k: model.num_experts_per_tok,
                activation_dtype: ACTIVATION_DTYPE,
                gpu_name: gpu.clone(),
                quant_backends: NVFP4_QUANT_BACKENDS.to_vec(),
                moe_backends: NVFP4_ROUTED_MOE_BACKENDS.to_vec(),
                weight_format: DType::Nvfp4E2m1,
                group_size: NVFP4_GROUP_SIZE,
                routing_method: ROUTING_METHOD.to_string(),
                n_group: 1,
                topk_group: 1,
                routed_scaling_numerator: model.route_scale.0,
                routed_scaling_denominator: model.route_scale.1,
                expert_demand: demand.clone(),
                folded_rank_position: 0,
            };
            if parallel.enable_expert_parallel {
                Nvfp4MoeLocalWorkletConfig::split_for_ep(template, demand)
            } else {
                vec![Nvfp4MoeLocalWorkletConfig::replicated_for_tp(
                    template, demand,
                )]
            }
        }
    };
    let routed_bf16 = match demand_for(InklingExpertPrecision::Bf16, &demand.bf16)? {
        None => Vec::new(),
        Some(demand) => {
            let template = Bf16MoeLocalWorkletConfig {
                hidden: model.hidden.into(),
                moe_intermediate: model.intermediate_size.into(),
                num_experts: model.n_routed_experts.into(),
                ep_size,
                tp_size: moe_tp,
                top_k: model.num_experts_per_tok,
                dtype: ACTIVATION_DTYPE,
                gpu_name: gpu.clone(),
                moe_backends: BF16_ROUTED_MOE_BACKENDS.to_vec(),
                routing_method: ROUTING_METHOD.to_string(),
                n_group: 1,
                topk_group: 1,
                routed_scaling_numerator: model.route_scale.0,
                routed_scaling_denominator: model.route_scale.1,
                expert_demand: demand.clone(),
                folded_rank_position: 0,
            };
            if parallel.enable_expert_parallel {
                Bf16MoeLocalWorkletConfig::split_for_ep(template, demand)
            } else {
                vec![Bf16MoeLocalWorkletConfig::replicated_for_tp(
                    template, demand,
                )]
            }
        }
    };
    Ok(InklingVllmConfigs {
        groups: layer_groups(model),
        attn_local: attention(true)?,
        attn_global: attention(false)?,
        dense_ffn: mlp(
            divide("dense_intermediate_size", model.dense_intermediate_size)?,
            model.use_global_scale,
        ),
        sink: mlp(
            divide(
                "sink intermediate",
                model.intermediate_size * model.n_shared_experts,
            )?,
            false,
        ),
        router: InklingRouterLocalWorkletConfig {
            hidden: model.hidden.into(),
            n_routed_experts: model.n_routed_experts,
            n_shared_experts: model.n_shared_experts,
            top_k: model.num_experts_per_tok,
            logits_width: model.router_logits_width(),
            has_bias: model.use_gate_bias,
            has_global_scale: model.use_global_scale,
            activation_dtype: ACTIVATION_DTYPE,
            gpu_name: gpu.clone(),
            router_backends: ROUTER_BACKENDS.to_vec(),
            gate_select_backends: GATE_SELECT_BACKENDS.to_vec(),
            copy_backends: TORCH_ELEMENTWISE_BACKENDS.to_vec(),
        },
        routed_nvfp4,
        routed_bf16,
        reduce: lamport(false),
        moe_reduce: lamport(true),
        // `_embed_rmsnorm_kernel`: an int32 id plus the gathered row in, the
        // residual and the normed row out.
        embed: ew(
            TRITON_ELEMENTWISE_BACKENDS,
            hidden_bytes + 2 * 4,
            2 * hidden_bytes,
        ),
        // One fp32 per token in and out per launch.
        log_scaling_tau: ew(TORCH_ELEMENTWISE_BACKENDS, 4, 4),
        lm_head: SingleGemmKernelConfig {
            backends: BF16_GEMM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            n: Dim::param("vocab_per_rank", divide("vocab_size", model.vocab_size)?),
            k: model.hidden.into(),
            dtype: ACTIVATION_DTYPE,
        },
        // `tensor_model_parallel_all_gather` of the bf16 logits; the NCCL
        // all-reduce curve stands in (DeepSeek V4.1's head).
        logits_all_gather: AllReduceKernelConfig {
            backends: LOGITS_ALL_GATHER_BACKENDS.to_vec(),
            gpu_name: gpu,
            num_gpus: tp,
            fabric: Fabric::Nvlink,
        },
        model: model.clone(),
        parallel: parallel.clone(),
    })
}

#[derive(Clone, Debug)]
pub struct InklingVllmResolved {
    pub raw_cfg: InklingVllmConfigs,
    pub attn_local: InklingAttnLocalWorkletResolved,
    pub attn_global: InklingAttnLocalWorkletResolved,
    pub dense_ffn: InklingMlpLocalWorkletResolved,
    pub sink: InklingMlpLocalWorkletResolved,
    pub routed_nvfp4: Vec<Nvfp4MoeLocalWorkletResolved>,
    pub routed_bf16: Vec<Bf16MoeLocalWorkletResolved>,
}

pub fn resolve_configs(cfgs: &InklingVllmConfigs) -> InklingVllmResolved {
    InklingVllmResolved {
        attn_local: InklingAttnLocalWorklet::resolve_config(&cfgs.attn_local),
        attn_global: InklingAttnLocalWorklet::resolve_config(&cfgs.attn_global),
        dense_ffn: InklingMlpLocalWorklet::resolve_config(&cfgs.dense_ffn),
        sink: InklingMlpLocalWorklet::resolve_config(&cfgs.sink),
        routed_nvfp4: cfgs
            .routed_nvfp4
            .iter()
            .map(Nvfp4MoeLocalWorklet::resolve_config)
            .collect(),
        routed_bf16: cfgs
            .routed_bf16
            .iter()
            .map(Bf16MoeLocalWorklet::resolve_config)
            .collect(),
        raw_cfg: cfgs.clone(),
    }
}

/// One EP rank's routed call, at its expert precision.
enum RoutedRank {
    Nvfp4(Nvfp4MoeLocalWorklet),
    Bf16(Bf16MoeLocalWorklet),
}

impl RoutedRank {
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        match self {
            Self::Nvfp4(worklet) => worklet.compile(builder),
            Self::Bf16(worklet) => worklet.compile(builder),
        }
    }

    fn eval_or_zero(&self, num_tokens: u32, zero: bool, ev: &mut Evaluator) {
        match (self, zero) {
            (Self::Nvfp4(worklet), false) => {
                worklet.eval(&Nvfp4MoeLocalWorkletInput { num_tokens }, ev)
            }
            (Self::Nvfp4(worklet), true) => {
                push_or_zero(&worklet.quant, Nvfp4QuantKernelInput { num_tokens }, true, ev);
                push_or_zero(
                    &worklet.fused_moe,
                    Nvfp4FusedMoeKernelInput { num_tokens },
                    true,
                    ev,
                );
            }
            (Self::Bf16(worklet), false) => {
                worklet.eval(&Bf16MoeLocalWorkletInput { num_tokens }, ev)
            }
            (Self::Bf16(worklet), true) => push_or_zero(
                &worklet.fused_moe,
                Bf16FusedMoeKernelInput { num_tokens },
                true,
                ev,
            ),
        }
    }
}

struct MoeBlock {
    name: String,
    router: InklingRouterLocalWorklet,
    sink: InklingMlpLocalWorklet,
    routed: Vec<RoutedRank>,
}

impl MoeBlock {
    /// Router, then per EP rank: the sinks concurrent with that rank's routed
    /// call, plus serial copies of both for batches too large for the aux
    /// stream. One pair is filled per iteration; INV-1 fixes the shape.
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let ep = self.routed.len();
        let router = self.router.compile(builder);
        let ranks = self
            .routed
            .iter()
            .map(|routed| {
                let concurrent_sink = self.sink.compile(builder);
                let concurrent_routed = routed.compile(builder);
                CostNode::Sum(vec![
                    CostNode::Parallel {
                        overlap: SINK_STREAM_OVERLAP,
                        children: vec![concurrent_sink, concurrent_routed],
                    },
                    routed.compile(builder),
                    self.sink.compile(builder),
                ])
            })
            .collect();
        CostNode::Labeled {
            label: format!(
                "{} (MoE) [EP{ep} ranked ranks; sinks overlap routed at T<={}]",
                self.name, SINK_STREAM_TOKEN_THRESHOLD
            ),
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
        let overlapped = num_tokens <= SINK_STREAM_TOKEN_THRESHOLD;
        self.router
            .eval(&InklingRouterLocalWorkletInput { num_tokens }, ev);
        let sink = InklingMlpLocalWorkletInput { num_tokens };
        for routed in &self.routed {
            self.sink.eval_or_zero(&sink, !overlapped, ev);
            routed.eval_or_zero(num_tokens, !overlapped, ev);
            routed.eval_or_zero(num_tokens, overlapped, ev);
            self.sink.eval_or_zero(&sink, overlapped, ev);
        }
    }
}

enum Ffn {
    Dense(InklingMlpLocalWorklet),
    Moe(MoeBlock),
}

/// One built layer group: the leaves of a single layer of its shape.
struct LayerGroup {
    label: String,
    layers: Vec<u32>,
    attn: InklingAttnLocalWorklet,
    attn_reduce: Op<RsSconvAgResidualRmsNormKernel>,
    ffn: Ffn,
    ffn_reduce: Op<RsSconvAgResidualRmsNormKernel>,
}

impl LayerGroup {
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let layer = CostNode::Sum(vec![
            self.attn.compile(builder),
            self.attn_reduce.compile(builder),
            match &self.ffn {
                Ffn::Dense(worklet) => worklet.compile(builder),
                Ffn::Moe(block) => block.compile(builder),
            },
            self.ffn_reduce.compile(builder),
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
            &InklingAttnLocalWorkletInput {
                num_tokens: tokens,
                query_context_pairs: batch.query_context_pairs.clone(),
            },
            ev,
        );
        let reduce = RsSconvAgResidualRmsNormKernelInput { num_tokens: tokens };
        push_or_zero(&self.attn_reduce, reduce.clone(), false, ev);
        match &self.ffn {
            Ffn::Dense(worklet) => {
                worklet.eval(&InklingMlpLocalWorkletInput { num_tokens: tokens }, ev)
            }
            Ffn::Moe(block) => block.eval(tokens, ev),
        }
        push_or_zero(&self.ffn_reduce, reduce, false, ev);
    }
}

fn build_layer_group(
    name: &str,
    group: &InklingLayerGroup,
    resolved: &InklingVllmResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<LayerGroup, BuildError> {
    let cfg = &resolved.raw_cfg;
    let prefix = format!("{name}.{}", group.label);
    let p = prefix.as_str();
    let attn = InklingAttnLocalWorklet::build(
        format!("{p}.attn"),
        match group.attn {
            AttnKind::Local => resolved.attn_local.clone(),
            AttnKind::Global => resolved.attn_global.clone(),
        },
        bridge,
    )?;
    let attn_reduce = atomic(
        &format!("{p}.attn"),
        "rs_sconv_ag_norm",
        cfg.reduce.clone(),
        RsSconvAgResidualRmsNormKernel::build,
        bridge,
    )?;
    let (ffn, ffn_name, ffn_reduce_cfg) = match group.ffn {
        FfnKind::Dense => (
            Ffn::Dense(InklingMlpLocalWorklet::build(
                format!("{p}.ffn"),
                resolved.dense_ffn.clone(),
                bridge,
            )?),
            format!("{p}.ffn"),
            cfg.reduce.clone(),
        ),
        FfnKind::Moe(precision) => {
            let moe = format!("{p}.moe");
            let routed = match precision {
                InklingExpertPrecision::Nvfp4 => resolved
                    .routed_nvfp4
                    .iter()
                    .enumerate()
                    .map(|(rank, r)| {
                        Nvfp4MoeLocalWorklet::build(
                            format!("{moe}.routed_rank{rank}"),
                            r.clone(),
                            bridge,
                        )
                        .map(RoutedRank::Nvfp4)
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?,
                InklingExpertPrecision::Bf16 => resolved
                    .routed_bf16
                    .iter()
                    .enumerate()
                    .map(|(rank, r)| {
                        Bf16MoeLocalWorklet::build(
                            format!("{moe}.routed_rank{rank}"),
                            r.clone(),
                            bridge,
                        )
                        .map(RoutedRank::Bf16)
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            };
            if routed.is_empty() {
                return Err(fit_failed(format!(
                    "{} has {precision:?} MoE layers but no routed configs",
                    group.label
                )));
            }
            (
                Ffn::Moe(MoeBlock {
                    router: InklingRouterLocalWorklet::build(
                        moe.clone(),
                        InklingRouterLocalWorklet::resolve_config(&cfg.router),
                        bridge,
                    )?,
                    sink: InklingMlpLocalWorklet::build(
                        format!("{moe}.sink"),
                        resolved.sink.clone(),
                        bridge,
                    )?,
                    routed,
                    name: moe.clone(),
                }),
                moe,
                cfg.moe_reduce.clone(),
            )
        }
    };
    Ok(LayerGroup {
        label: group.label.clone(),
        layers: group.layers.clone(),
        attn,
        attn_reduce,
        ffn,
        ffn_reduce: atomic(
            &ffn_name,
            "rs_sconv_ag_norm",
            ffn_reduce_cfg,
            RsSconvAgResidualRmsNormKernel::build,
            bridge,
        )?,
    })
}

pub struct InklingVllmModel {
    pub name: String,
    pub tp_size: u16,
    pub enable_expert_parallel: bool,
    pub max_model_len: u32,
    /// Sorted, deduplicated capture sizes.
    cudagraph_capture_sizes: Vec<u32>,
    embed: Op<ElementwiseKernel>,
    /// `None` without global layers, which alone read tau.
    log_scaling_tau: Option<Op<ElementwiseKernel>>,
    groups: Vec<LayerGroup>,
    lm_head: Op<SingleGemmKernel>,
    logits_all_gather: Op<AllReduceKernel>,
    vocab_size: u32,
    total_kv_bytes_per_token: u64,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: InklingVllmResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<InklingVllmModel, BuildError> {
    let cfg = &resolved.raw_cfg;
    let n = name.as_str();
    let groups = cfg
        .groups
        .iter()
        .map(|group| build_layer_group(n, group, &resolved, bridge))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let model_cfg = &cfg.model;
    let has_global = (0..model_cfg.num_layers).any(|layer| !model_cfg.is_local(layer));
    let mut model = InklingVllmModel {
        embed: atomic(n, "embed", cfg.embed.clone(), ElementwiseKernel::build, bridge)?,
        log_scaling_tau: has_global
            .then(|| {
                atomic(
                    n,
                    "log_scaling_tau",
                    cfg.log_scaling_tau.clone(),
                    ElementwiseKernel::build,
                    bridge,
                )
            })
            .transpose()?,
        groups,
        lm_head: atomic(n, "lm_head", cfg.lm_head.clone(), SingleGemmKernel::build, bridge)?,
        logits_all_gather: atomic(
            n,
            "logits_all_gather",
            cfg.logits_all_gather.clone(),
            AllReduceKernel::build,
            bridge,
        )?,
        vocab_size: model_cfg.vocab_size,
        total_kv_bytes_per_token: total_kv_bytes_per_token(model_cfg),
        tp_size: cfg.parallel.tp_size,
        enable_expert_parallel: cfg.parallel.enable_expert_parallel,
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

/// K and V of every layer's KV heads, summed over the ranks. Sliding-window
/// layers are counted at full length: vLLM frees their pages behind the
/// window, which this per-token total cannot express, so capacity is a
/// lower bound.
fn total_kv_bytes_per_token(model: &InklingModelCfg) -> u64 {
    (0..model.num_layers)
        .map(|layer| {
            let (kv_heads, head_dim) = if model.is_local(layer) {
                (model.swa_num_key_value_heads, model.swa_head_dim)
            } else {
                (model.num_key_value_heads, model.head_dim)
            };
            2 * u64::from(kv_heads * head_dim) * u64::from(ACTIVATION_DTYPE.size_bytes())
        })
        .sum()
}

impl InklingVllmModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let mut children = vec![self.embed.compile(&mut builder)];
        children.extend(self.log_scaling_tau.as_ref().map(|op| CostNode::Scale {
            n: LOG_SCALING_TAU_LAUNCHES,
            child: Box::new(op.compile(&mut builder)),
        }));
        for group in &self.groups {
            children.push(group.compile(&mut builder));
        }
        children.push(self.lm_head.compile(&mut builder));
        children.push(self.logits_all_gather.compile(&mut builder));
        let layout = if self.enable_expert_parallel {
            format!("TP=EP{}", self.tp_size)
        } else {
            format!("TP{} with MoE TP{}", self.tp_size, self.tp_size)
        };
        let root = CostNode::Labeled {
            label: format!(
                "{} (InklingVllmModel) [{layout}; timing_context<={}]",
                self.name, self.max_model_len
            ),
            child: Box::new(CostNode::Sum(children)),
        };
        builder.finish(root)
    }

    fn eval_into(&self, input: &UnifiedArchInput, ev: &mut Evaluator) {
        let batch = normalize_input(input, self.max_model_len)
            .unwrap_or_else(|reason| panic!("invalid InklingVllmModel input: {reason}"));
        let tokens = graph_padded_tokens(&self.cudagraph_capture_sizes, batch.total_tokens);
        let rows = ElementwiseKernelInput { num_tokens: tokens };
        push_or_zero(&self.embed, rows.clone(), false, ev);
        if let Some(op) = &self.log_scaling_tau {
            push_or_zero(op, rows, false, ev);
        }
        for group in &self.groups {
            group.eval(&batch, tokens, ev);
        }
        let logits = batch.request_count;
        push_or_zero(&self.lm_head, SingleGemmKernelInput { m: logits }, false, ev);
        let gathered = u64::from(logits) * u64::from(self.vocab_size) * 2;
        push_or_zero(
            &self.logits_all_gather,
            AllReduceKernelInput {
                message_size_bytes: all_gather_as_all_reduce_bytes(gathered),
            },
            false,
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

impl IterwiseUnifiedModel for InklingVllmModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        normalize_input(batch, self.max_model_len).map(|_| ())
    }

    fn total_kv_bytes_per_token(&self) -> u64 {
        self.total_kv_bytes_per_token
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

/// One iteration, as every layer sees it (one TP group).
struct NormalizedBatch {
    total_tokens: u32,
    request_count: u32,
    /// Real requests as `(query_len, context_len)`, prefill chunks first.
    query_context_pairs: Vec<(u32, u32)>,
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
    let mut pairs = Vec::with_capacity(group.prefill_chunk_pairs.len() + group.decode_kv_lens.len());
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

fn push_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, ev: &mut Evaluator)
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
    use std::path::Path;

    fn model_cfg() -> InklingModelCfg {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/inkling_small_nvfp4.json");
        InklingModelCfg::from_json(&path).unwrap()
    }

    fn parallel() -> InklingVllmParallel {
        InklingVllmParallel {
            tp_size: 4,
            enable_expert_parallel: true,
            max_model_len: 8192,
            gpu_name: "NVIDIA B200".into(),
            cudagraph_capture_sizes: vec![1, 2, 4, 8, 16, 24, 32],
        }
    }

    fn demand() -> InklingExpertDemand {
        let uniform = ExpertDemand::popularity(&RoutingDistribution::uniform(256), 7);
        InklingExpertDemand {
            nvfp4: Some(uniform.clone()),
            bf16: Some(uniform),
        }
    }

    fn built() -> InklingVllmModel {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let configs = build_configs(&model_cfg(), &parallel(), &demand()).unwrap();
        build("unified".into(), resolve_configs(&configs), &bridge).unwrap()
    }

    #[test]
    fn layer_groups_follow_the_attention_and_expert_schedule() {
        let groups = layer_groups(&model_cfg());
        let summary: Vec<(&str, usize)> = groups
            .iter()
            .map(|g| (g.label.as_str(), g.layers.len()))
            .collect();
        assert_eq!(
            summary,
            [
                ("local_dense", 2),
                ("local_bf16_moe", 1),
                ("local_nvfp4_moe", 32),
                ("global_nvfp4_moe", 7)
            ]
        );
    }

    #[test]
    fn configs_carry_the_capture_shapes() {
        let configs = build_configs(&model_cfg(), &parallel(), &demand()).unwrap();
        assert_eq!(configs.routed_nvfp4.len(), 4);
        assert_eq!(configs.routed_bf16.len(), 4);
        assert_eq!(configs.dense_ffn.intermediate.get(), 4096);
        assert_eq!(configs.sink.intermediate.get(), 1024);
        assert_eq!(configs.lm_head.n.get(), 50256);
        assert!(configs.moe_reduce.has_shared && !configs.reduce.has_shared);
        assert_eq!(configs.attn_local.rel_extent, 512);
        assert_eq!(configs.attn_global.rel_extent, 1024);
    }

    #[test]
    fn tp1_is_refused() {
        let mut layout = parallel();
        layout.tp_size = 1;
        let error = build_configs(&model_cfg(), &layout, &demand()).unwrap_err();
        assert!(error.to_string().contains("tp_size 1"), "{error}");
    }

    #[test]
    fn missing_demand_for_a_present_precision_is_refused() {
        let partial = InklingExpertDemand {
            nvfp4: demand().nvfp4,
            bf16: None,
        };
        let error = build_configs(&model_cfg(), &parallel(), &partial).unwrap_err();
        assert!(error.to_string().contains("Bf16"), "{error}");
    }

    #[test]
    fn manifest_folds_each_group_and_charges_one_lamport_per_sublayer() {
        let model = built();
        let manifest = model.cost_log_manifest();
        let lamports = manifest
            .slots
            .iter()
            .filter(|slot| slot.name.ends_with("rs_sconv_ag_norm"))
            .count();
        // Two per layer group (attention + FFN/MoE), folded by Scale.
        assert_eq!(lamports, 2 * 4);
        assert!(manifest
            .slots
            .iter()
            .any(|slot| slot.name == "unified.global_nvfp4_moe.moe.gate_select"));
    }

    #[test]
    fn a_mixed_step_is_one_attention_batch_of_real_requests() {
        let group = ArchGroupInput {
            batch_tokens: 1992,
            prefill_tokens: 1990,
            decode_tokens: 2,
            prefill_chunk_pairs: vec![(512, 1990)],
            decode_kv_lens: vec![3000, 17],
            ..Default::default()
        };
        let input = UnifiedArchInput {
            groups: vec![group],
            ..Default::default()
        };
        let batch = normalize_input(&input, 8192).unwrap();
        assert_eq!(batch.query_context_pairs, [(1990, 2502), (1, 3000), (1, 17)]);
        assert_eq!((batch.total_tokens, batch.request_count), (1992, 3));
        // Outside attention, the step runs on the next captured graph size.
        assert_eq!(graph_padded_tokens(&parallel().cudagraph_capture_sizes, 25), 32);
        let too_long = UnifiedArchInput {
            groups: vec![ArchGroupInput {
                batch_tokens: 1,
                decode_tokens: 1,
                decode_kv_lens: vec![9000],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(built().check_input(&too_long).is_err());
    }
}
