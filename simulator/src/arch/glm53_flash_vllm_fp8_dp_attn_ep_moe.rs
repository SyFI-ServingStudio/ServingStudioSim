//! GLM-5.3-Flash FP8 block checkpoint on B200 under vLLM data-parallel
//! attention with expert-parallel MoE (`--data-parallel-size N
//! --enable-expert-parallel`, TP 1). Paths below are in the vLLM fork.
//!
//! Every DP rank is its own `EngineCore` with its own scheduler and KV cache
//! (`v1/engine/core.py` `DPEngineCoreProc`), so a request lives on one rank:
//! all 64 KDA and 64 MLA heads, the full dense FFN, shared expert, embedding
//! and lm_head run rank-locally on that rank's tokens, with no collective at
//! TP 1. The routed experts are split over the N ranks (288 / N each).
//!
//! An MoE layer follows vLLM's `allgather_reducescatter` backend with the
//! TRT-LLM FP8 block monolithic experts (`MoEPrepareAndFinalizeNaiveDPEPMonolithic`,
//! `prepare_finalize/naive_dp_ep.py`):
//!
//! ```text
//! per rank   gate x2 (local)   shared expert (local, T_r > 256)   fp8 quant (local)
//! group      all_gatherv [fp8 activations, fp32 router logits, fp32 scales]
//! per rank   trtllm_fp8_block_scale_moe over every gathered token (routing inside)
//! group      reduce_scatterv of the bf16 expert output
//! per rank   shared + routed add
//! ```
//!
//! The collectives carry the exact ragged per-rank token counts: an
//! iteration with a prefill-sized rank runs eager, and eager DP keeps the real
//! counts (`v1/worker/gpu/dp_utils.py`). Only when every rank's batch fits a
//! captured CUDA graph does each rank pad to the common graph size. A rank
//! with nothing scheduled still runs `execute_dummy_batch` (one decode token)
//! so the collectives can complete; it is costed that way.
//!
//! Ranks synchronise at each MoE collective, so each sublayer section between
//! them is a `Max` over ranks. The rank-local sections of the dense layers
//! fold into the same per-sublayer `Max`, a slight over-estimate of the true
//! max-of-sums between collectives when ranks are unevenly loaded.
//!
//! The NVFP4 checkpoint (`glm53_flash_vllm_nvfp4_dp_attn_ep_moe`) takes the
//! modular TRT-LLM NVFP4 experts instead: the monolithic kernel declines a
//! DP + EP config (`experts/trtllm_nvfp4_moe.py:476-481`), so vLLM's grouped
//! top-k selects the experts on each rank first (`moe_runner.py:620-633`),
//! the local tokens are NVFP4-quantized, one all-gatherv carries the packed
//! activations, block scales, top-k weights and IDs
//! (`prepare_finalize/naive_dp_ep.py:112-185`, `moe_ep_quantized_all_gather`),
//! and `trtllm_fp4_block_scale_routed_moe` runs with the routing precomputed.
//! The combine is the same bf16 reduce-scatterv.
//!
//! At or below 256 local tokens vLLM runs the shared expert on an aux stream,
//! launched before the gate and joined after the combine; that copy overlaps
//! the dispatch, routed experts and combine (`Max{overlap}` as in the TP arch).

use anyhow::Result;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::glm53_flash_vllm_fp8_kda_dsa_moe::{
    self as tp_arch, atomic, graph_padded_tokens, push, AttnKind, Boundary, FfnKind,
    Glm53FlashLayerGroup, Glm53FlashModelCfg, Glm53FlashQuant, Glm53FlashVllmConfigs,
    Glm53FlashVllmParallel, ACTIVATION_DTYPE, SHARED_EXPERTS_STREAM_OVERLAP,
    SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD,
};
use crate::op::mhc::{MhcTerminalPostConfig, MhcTerminalPostInput, MhcTerminalPostOp};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    MhcFusedPostPreRmsNormKernel, MhcPreRmsNormKernel, MhcRmsNormKernelInput, MoeEpAllGatherKernel,
    MoeEpAllGatherKernelConfig, MoeEpCollectiveKernelInput, MoeEpQuantizedAllGatherKernel,
    MoeEpQuantizedAllGatherKernelConfig, MoeEpReduceScatterKernel, MoeEpReduceScatterKernelConfig,
    Nvfp4FusedMoeKernel, Nvfp4FusedMoeKernelInput, RmsNormKernel, RmsNormKernelInput,
    SingleGemmKernel, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::{
    Glm53ActivationQuant, Glm53DsaAttnLocalWorklet, Glm53DsaAttnLocalWorkletInput,
    Glm53KdaAttnLocalWorklet, Glm53KdaAttnLocalWorkletInput, Glm53MlpLocalWorklet,
    Glm53MlpLocalWorkletInput, Glm53MoeRouterLocalWorklet, Glm53MoeRouterLocalWorkletInput,
    Glm53RoutedMoeLocalWorklet, Glm53RoutedMoeLocalWorkletConfig,
    Glm53RoutedMoeLocalWorkletResolved,
};

const ARCH_KIND: &str = "glm53_flash_vllm_fp8_dp_attn_ep_moe";
const COLLECTIVE_BACKENDS: &[&str] = &["vllm_pynccl"];
/// The NVFP4 modular experts take the routing chosen before the dispatch.
const NVFP4_ROUTED_MOE_BACKENDS: &[&str] = &["flashinfer_trtllm_routed_sm100"];
const ROUTER_SELECT_BACKENDS: &[&str] = &["triton"];
/// vLLM V1's default `max_num_batched_tokens` for chunked prefill: one rank's
/// scheduler budget, which bounds the collectives' profiled token grid.
const RANK_TOKEN_BUDGET: u32 = 8192;

/// Deployment layout: `dp_size` engine cores, one GPU each, forming one EP group.
#[derive(Clone, Debug)]
pub struct Glm53FlashDpAttnEpParallel {
    pub dp_size: u16,
    pub max_model_len: u32,
    pub gpu_name: String,
    /// vLLM `--cudagraph-capture-sizes`; empty runs eager (no padding).
    pub cudagraph_capture_sizes: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Glm53FlashDpAttnEpConfigs {
    /// Every rank-local sublayer at TP 1 (64 KDA/MLA heads, full dense FFN,
    /// shared expert and vocab). Its `routed` and `all_reduce` are unused.
    pub local: Glm53FlashVllmConfigs,
    pub parallel: Glm53FlashDpAttnEpParallel,
    /// One per EP rank, ranked by routed workload.
    pub routed: Vec<Glm53RoutedMoeLocalWorkletConfig>,
    pub dispatch: Glm53FlashDispatchConfig,
    /// vLLM's grouped top-k ahead of an NVFP4 dispatch: fp32 logits and bias
    /// in, top-k IDs and weights out. FP8 routes inside the fused MoE.
    pub router_select: Option<ElementwiseKernelConfig>,
    pub combine: MoeEpReduceScatterKernelConfig,
}

/// The MoE dispatch all-gather for the checkpoint's routed experts.
#[derive(Clone, Debug)]
pub enum Glm53FlashDispatchConfig {
    /// FP8 activations, fp32 router logits and block scales.
    Fp8(MoeEpAllGatherKernelConfig),
    /// Packed NVFP4 activations, E4M3 group scales, top-k weights and IDs.
    Nvfp4(MoeEpQuantizedAllGatherKernelConfig),
}

pub fn build_configs(
    model: &Glm53FlashModelCfg,
    parallel: &Glm53FlashDpAttnEpParallel,
    demand: &ExpertDemand,
) -> std::result::Result<Glm53FlashDpAttnEpConfigs, BuildError> {
    let dp = parallel.dp_size;
    if dp < 2 || model.n_routed_experts % u32::from(dp) != 0 {
        return Err(fit_failed(format!(
            "DP {dp} must be at least 2 and divide {} routed experts",
            model.n_routed_experts
        )));
    }
    let local = tp_arch::build_configs(
        model,
        &Glm53FlashVllmParallel {
            tp_size: 1,
            // Whole routed intermediate; the EP split over DP ranks follows.
            enable_expert_parallel: true,
            max_model_len: parallel.max_model_len,
            gpu_name: parallel.gpu_name.clone(),
            cudagraph_capture_sizes: parallel.cudagraph_capture_sizes.clone(),
        },
        demand,
    )?;
    let mut template = local.routed[0].clone();
    template.ep_size = dp;
    if model.quant == Glm53FlashQuant::Nvfp4 {
        template.fused_moe_backends = NVFP4_ROUTED_MOE_BACKENDS.to_vec();
    }
    let routed = Glm53RoutedMoeLocalWorkletConfig::split_for_ep(template, demand.clone());
    let gpu = parallel.gpu_name.clone();
    let max_total_tokens = RANK_TOKEN_BUDGET * u32::from(dp);
    let (dispatch, router_select) = match model.quant {
        Glm53FlashQuant::Fp8Block => (
            Glm53FlashDispatchConfig::Fp8(MoeEpAllGatherKernelConfig {
                backends: COLLECTIVE_BACKENDS.to_vec(),
                gpu_name: gpu.clone(),
                num_gpus: dp.into(),
                hidden_size: model.hidden.into(),
                num_experts: model.n_routed_experts.into(),
                hidden_dtype: DType::Fp8E4m3,
                router_dtype: DType::Fp32,
                fabric: "nvlink".into(),
                max_total_tokens,
            }),
            None,
        ),
        Glm53FlashQuant::Nvfp4 => {
            let fp32 = DType::Fp32.size_bytes();
            (
                Glm53FlashDispatchConfig::Nvfp4(MoeEpQuantizedAllGatherKernelConfig {
                    backends: COLLECTIVE_BACKENDS.to_vec(),
                    gpu_name: gpu.clone(),
                    num_gpus: dp.into(),
                    hidden_size: model.hidden.into(),
                    top_k: model.num_experts_per_tok.into(),
                    activation_dtype: DType::Nvfp4E2m1,
                    fabric: "nvlink".into(),
                    max_total_tokens,
                }),
                Some(ElementwiseKernelConfig {
                    backends: ROUTER_SELECT_BACKENDS.to_vec(),
                    gpu_name: gpu.clone(),
                    input_bytes_per_token: (2 * model.n_routed_experts * fp32).into(),
                    output_bytes_per_token: (model.num_experts_per_tok * (fp32 + 4)).into(),
                }),
            )
        }
    };
    Ok(Glm53FlashDpAttnEpConfigs {
        dispatch,
        router_select,
        combine: MoeEpReduceScatterKernelConfig {
            backends: COLLECTIVE_BACKENDS.to_vec(),
            gpu_name: gpu,
            num_gpus: dp.into(),
            hidden_size: model.hidden.into(),
            dtype: ACTIVATION_DTYPE,
            fabric: "nvlink".into(),
            max_total_tokens,
        },
        routed,
        local,
        parallel: parallel.clone(),
    })
}

#[derive(Clone, Debug)]
pub struct Glm53FlashDpAttnEpResolved {
    pub raw_cfg: Glm53FlashDpAttnEpConfigs,
    pub local: tp_arch::Glm53FlashVllmResolved,
    pub routed: Vec<Glm53RoutedMoeLocalWorkletResolved>,
}

pub fn resolve_configs(cfgs: &Glm53FlashDpAttnEpConfigs) -> Glm53FlashDpAttnEpResolved {
    Glm53FlashDpAttnEpResolved {
        local: tp_arch::resolve_configs(&cfgs.local),
        routed: cfgs
            .routed
            .iter()
            .map(Glm53RoutedMoeLocalWorklet::resolve_config)
            .collect(),
        raw_cfg: cfgs.clone(),
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
    shared_expert: Glm53MlpLocalWorklet,
    router_select: Option<Op<ElementwiseKernel>>,
    dispatch_quant: Glm53ActivationQuant,
    dispatch: Dispatch,
    /// One per EP rank, ranked by routed workload.
    fused_moe: Vec<Op<Nvfp4FusedMoeKernel>>,
    combine: Op<MoeEpReduceScatterKernel>,
    combine_glue: Op<ElementwiseKernel>,
}

enum Ffn {
    Dense(Glm53MlpLocalWorklet),
    Moe(MoeBlock),
}

enum Dispatch {
    Fp8(Op<MoeEpAllGatherKernel>),
    Nvfp4(Op<MoeEpQuantizedAllGatherKernel>),
}

impl Dispatch {
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        match self {
            Self::Fp8(op) => op.compile(builder),
            Self::Nvfp4(op) => op.compile(builder),
        }
    }

    fn eval(&self, input: MoeEpCollectiveKernelInput, ev: &mut Evaluator) {
        match self {
            Self::Fp8(op) => push(op, input, ev),
            Self::Nvfp4(op) => push(op, input, ev),
        }
    }
}

struct LayerGroup {
    label: String,
    layers: Vec<u32>,
    attn_boundary: Boundary,
    attn: Attention,
    ffn_boundary: Op<MhcFusedPostPreRmsNormKernel>,
    ffn: Ffn,
}

impl LayerGroup {
    /// Per rank: everything up to the dispatch (the whole layer for a dense
    /// FFN). Then, for an MoE layer, the group-wide expert section and the
    /// per-rank combine add.
    fn compile(&self, dp: usize, builder: &mut CostTreeBuilder) -> CostNode {
        let ranks = (0..dp)
            .map(|_| {
                let mut leaves = vec![
                    self.attn_boundary.compile(builder),
                    match &self.attn {
                        Attention::Kda(worklet) => worklet.compile(builder),
                        Attention::Dsa(worklet) => worklet.compile(builder),
                    },
                    self.ffn_boundary.compile(builder),
                ];
                match &self.ffn {
                    Ffn::Dense(worklet) => leaves.push(worklet.compile(builder)),
                    Ffn::Moe(block) => {
                        leaves.push(block.router.compile(builder));
                        leaves.extend(block.router_select.as_ref().map(|op| op.compile(builder)));
                        leaves.extend([
                            block.input_glue.compile(builder),
                            block.shared_expert.compile(builder),
                            block.dispatch_quant.compile(builder),
                        ]);
                    }
                }
                CostNode::Sum(leaves)
            })
            .collect();
        let mut body = vec![CostNode::Labeled {
            label: format!(
                "{} [Max over DP ranks: rank-local attention and FFN up to the dispatch]",
                self.label
            ),
            child: Box::new(CostNode::Max {
                overlap: 1.0,
                children: ranks,
            }),
        }];
        if let Ffn::Moe(block) = &self.ffn {
            body.extend(block.compile_expert_sections(dp, builder));
        }
        CostNode::Labeled {
            label: format!("{} layers {:?}", self.label, self.layers),
            child: Box::new(CostNode::Scale {
                n: self.layers.len() as u32,
                child: Box::new(CostNode::Sum(body)),
            }),
        }
    }

    fn eval(&self, batch: &NormalizedBatch, ev: &mut Evaluator) {
        for rank in &batch.ranks {
            self.attn_boundary.eval(rank.rows, ev);
            match &self.attn {
                Attention::Kda(worklet) => worklet.eval(&rank.kda, ev),
                Attention::Dsa(worklet) => worklet.eval(&rank.dsa, ev),
            }
            push(
                &self.ffn_boundary,
                MhcRmsNormKernelInput {
                    num_tokens: rank.rows,
                },
                ev,
            );
            match &self.ffn {
                Ffn::Dense(worklet) => worklet.eval(
                    &Glm53MlpLocalWorkletInput {
                        num_tokens: rank.rows,
                    },
                    ev,
                ),
                Ffn::Moe(block) => block.eval_local(rank.rows, ev),
            }
        }
        if let Ffn::Moe(block) = &self.ffn {
            block.eval_expert_sections(batch, ev);
        }
    }
}

impl MoeBlock {
    /// The dispatch / routed experts / combine chain, which the aux-stream
    /// shared expert (T_r <= 256) overlaps, then the per-rank shared + routed add.
    fn compile_expert_sections(&self, dp: usize, builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        let dispatch = self.dispatch.compile(builder);
        let experts = CostNode::Labeled {
            label: format!("{}.ep_ranks (max over EP ranks)", self.name),
            child: Box::new(CostNode::Max {
                overlap: 1.0,
                children: self
                    .fused_moe
                    .iter()
                    .map(|op| op.compile(builder))
                    .collect(),
            }),
        };
        let combine = self.combine.compile(builder);
        let concurrent_shared = CostNode::Max {
            overlap: 1.0,
            children: (0..dp)
                .map(|_| self.shared_expert.compile(builder))
                .collect(),
        };
        let combine_glue = CostNode::Max {
            overlap: 1.0,
            children: (0..dp)
                .map(|_| self.combine_glue.compile(builder))
                .collect(),
        };
        vec![
            CostNode::Labeled {
                label: format!(
                    "{} (MoE) [all_gatherv -> EP{} routed experts -> reduce_scatterv; \
                     aux-stream shared expert at T_r<={} overlaps]",
                    self.name,
                    self.fused_moe.len(),
                    SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
                ),
                child: Box::new(CostNode::Parallel {
                    overlap: SHARED_EXPERTS_STREAM_OVERLAP,
                    children: vec![
                        CostNode::Sum(vec![dispatch, experts, combine]),
                        concurrent_shared,
                    ],
                }),
            },
            CostNode::Labeled {
                label: format!("{}.combine_glue [Max over DP ranks]", self.name),
                child: Box::new(combine_glue),
            },
        ]
    }

    /// Router (twice), routed-input copy, the main-stream shared expert for
    /// T_r > 256, and the fp8 quant of the local tokens ahead of the dispatch.
    fn eval_local(&self, rows: u32, ev: &mut Evaluator) {
        self.router
            .eval(&Glm53MoeRouterLocalWorkletInput { num_tokens: rows }, ev);
        if let Some(select) = &self.router_select {
            push(select, ElementwiseKernelInput { num_tokens: rows }, ev);
        }
        push(
            &self.input_glue,
            ElementwiseKernelInput { num_tokens: rows },
            ev,
        );
        self.shared_expert.eval_or_zero(
            &Glm53MlpLocalWorkletInput { num_tokens: rows },
            rows <= SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD,
            ev,
        );
        self.dispatch_quant.eval_or_zero(rows, false, ev);
    }

    fn eval_expert_sections(&self, batch: &NormalizedBatch, ev: &mut Evaluator) {
        let collective = MoeEpCollectiveKernelInput {
            per_rank_tokens: batch.ranks.iter().map(|rank| rank.rows).collect(),
        };
        self.dispatch.eval(collective.clone(), ev);
        for op in &self.fused_moe {
            push(
                op,
                Nvfp4FusedMoeKernelInput {
                    num_tokens: batch.gathered_tokens,
                },
                ev,
            );
        }
        push(&self.combine, collective, ev);
        for rank in &batch.ranks {
            self.shared_expert.eval_or_zero(
                &Glm53MlpLocalWorkletInput {
                    num_tokens: rank.rows,
                },
                rank.rows > SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD,
                ev,
            );
        }
        for rank in &batch.ranks {
            push(
                &self.combine_glue,
                ElementwiseKernelInput {
                    num_tokens: rank.rows,
                },
                ev,
            );
        }
    }
}

pub struct Glm53FlashDpAttnEpModel {
    pub name: String,
    pub dp_size: u16,
    pub max_model_len: u32,
    cudagraph_capture_sizes: Vec<u32>,
    embedding: Op<ElementwiseKernel>,
    hc_expand: Op<ElementwiseKernel>,
    groups: Vec<LayerGroup>,
    terminal_post: MhcTerminalPostOp,
    hc_contract_mean: Op<ElementwiseKernel>,
    final_norm: Op<RmsNormKernel>,
    lm_head: Op<SingleGemmKernel>,
    total_kv_bytes_per_token: u64,
    recurrent_state_bytes_per_request: u64,
    hybrid_block_size: u32,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: Glm53FlashDpAttnEpResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<Glm53FlashDpAttnEpModel, BuildError> {
    let cfg = &resolved.raw_cfg;
    let local = &cfg.local;
    let n = name.as_str();
    let mut groups = Vec::with_capacity(local.groups.len());
    for group in &local.groups {
        groups.push(build_group(n, group, &resolved, bridge)?);
    }
    let model_cfg = &local.model;
    // One rank holds a request's whole KV: the fp8 MLA latent plus the kpool
    // index cache of each DSA layer. Nothing is replicated across ranks.
    let dsa_bytes_per_token = tp_arch::dsa_layer_bytes_per_token(model_cfg);
    let hybrid_block_size = tp_arch::hybrid_block_tokens(
        tp_arch::kda_layer_state_bytes_per_request(&resolved.local),
        u64::from(model_cfg.kv_lora_rank),
    );
    // vLLM pads each KDA layer's mamba page up to one attention page (block x
    // 512 B; +1.1% at TP1), so that is what a request holds on its rank.
    let recurrent_state_bytes_per_request = model_cfg.kda_layers.len() as u64
        * u64::from(hybrid_block_size)
        * u64::from(model_cfg.kv_lora_rank);
    let mut model = Glm53FlashDpAttnEpModel {
        embedding: atomic(
            n,
            "embedding",
            local.embedding.clone(),
            ElementwiseKernel::build,
            bridge,
        )?,
        hc_expand: atomic(
            n,
            "hc_expand",
            local.hc_expand.clone(),
            ElementwiseKernel::build,
            bridge,
        )?,
        groups,
        terminal_post: MhcTerminalPostOp::build(
            format!("{n}.final_mhc_post"),
            MhcTerminalPostConfig {
                mhc: local.mhc.clone(),
                pre_backends: local.mhc.backends.clone(),
                fused_backends: local.mhc.backends.clone(),
            },
            bridge,
        )?,
        hc_contract_mean: atomic(
            n,
            "hc_contract_mean",
            local.hc_contract_mean.clone(),
            ElementwiseKernel::build,
            bridge,
        )?,
        final_norm: atomic(
            n,
            "final_norm",
            local.final_norm.clone(),
            RmsNormKernel::build,
            bridge,
        )?,
        lm_head: atomic(
            n,
            "lm_head",
            local.lm_head.clone(),
            SingleGemmKernel::build,
            bridge,
        )?,
        dp_size: cfg.parallel.dp_size,
        max_model_len: cfg.parallel.max_model_len,
        cudagraph_capture_sizes: {
            let mut sizes = cfg.parallel.cudagraph_capture_sizes.clone();
            sizes.sort_unstable();
            sizes.dedup();
            sizes
        },
        total_kv_bytes_per_token: u64::from(model_cfg.num_dsa_layers()) * dsa_bytes_per_token,
        recurrent_state_bytes_per_request,
        hybrid_block_size,
        cost_flat: Vec::new(),
        n_slots: 0,
        name,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

fn build_group(
    n: &str,
    group: &Glm53FlashLayerGroup,
    resolved: &Glm53FlashDpAttnEpResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<LayerGroup, BuildError> {
    let cfg = &resolved.raw_cfg;
    let local = &cfg.local;
    let prefix = format!("{n}.{}", group.label);
    let p = prefix.as_str();
    let attn_boundary = if group.opens_stream {
        Boundary::Pre(atomic(
            p,
            "attn_mhc_pre",
            local.mhc.clone(),
            MhcPreRmsNormKernel::build,
            bridge,
        )?)
    } else {
        Boundary::Fused(atomic(
            p,
            "attn_mhc_post_pre",
            local.mhc.clone(),
            MhcFusedPostPreRmsNormKernel::build,
            bridge,
        )?)
    };
    let attn = match group.attn {
        AttnKind::Kda => Attention::Kda(Glm53KdaAttnLocalWorklet::build(
            format!("{p}.kda"),
            resolved.local.kda.clone(),
            bridge,
        )?),
        AttnKind::Dsa => Attention::Dsa(Glm53DsaAttnLocalWorklet::build(
            format!("{p}.dsa"),
            resolved.local.dsa.clone(),
            bridge,
        )?),
    };
    let ffn = match group.ffn {
        FfnKind::Dense => Ffn::Dense(Glm53MlpLocalWorklet::build(
            format!("{p}.dense_ffn"),
            resolved.local.dense_ffn.clone(),
            bridge,
        )?),
        FfnKind::Moe => {
            let moe = format!("{p}.moe");
            let m = moe.as_str();
            Ffn::Moe(MoeBlock {
                router: Glm53MoeRouterLocalWorklet::build(
                    format!("{m}.router"),
                    Glm53MoeRouterLocalWorklet::resolve_config(&local.router),
                    bridge,
                )?,
                input_glue: atomic(
                    m,
                    "input_glue",
                    local.moe_input_glue.clone(),
                    ElementwiseKernel::build,
                    bridge,
                )?,
                shared_expert: Glm53MlpLocalWorklet::build(
                    format!("{m}.shared_expert"),
                    resolved.local.shared_expert.clone(),
                    bridge,
                )?,
                router_select: cfg
                    .router_select
                    .clone()
                    .map(|select| {
                        atomic(m, "router_select", select, ElementwiseKernel::build, bridge)
                    })
                    .transpose()?,
                dispatch_quant: Glm53ActivationQuant::build(
                    m,
                    "dispatch_quant",
                    resolved.routed[0].input_quant.clone(),
                    bridge,
                )?,
                dispatch: match cfg.dispatch.clone() {
                    Glm53FlashDispatchConfig::Fp8(dispatch) => Dispatch::Fp8(atomic(
                        m,
                        "dispatch_all_gather",
                        dispatch,
                        MoeEpAllGatherKernel::build,
                        bridge,
                    )?),
                    Glm53FlashDispatchConfig::Nvfp4(dispatch) => Dispatch::Nvfp4(atomic(
                        m,
                        "dispatch_all_gather",
                        dispatch,
                        MoeEpQuantizedAllGatherKernel::build,
                        bridge,
                    )?),
                },
                fused_moe: resolved
                    .routed
                    .iter()
                    .enumerate()
                    .map(|(rank, routed)| {
                        atomic(
                            m,
                            &format!("routed_rank{rank}.fused_moe"),
                            routed.fused_moe.clone(),
                            Nvfp4FusedMoeKernel::build,
                            bridge,
                        )
                    })
                    .collect::<std::result::Result<_, _>>()?,
                combine: atomic(
                    m,
                    "combine_reduce_scatter",
                    cfg.combine.clone(),
                    MoeEpReduceScatterKernel::build,
                    bridge,
                )?,
                combine_glue: atomic(
                    m,
                    "combine_glue",
                    local.moe_combine_glue.clone(),
                    ElementwiseKernel::build,
                    bridge,
                )?,
                name: moe,
            })
        }
    };
    Ok(LayerGroup {
        label: group.label.clone(),
        layers: group.layers.clone(),
        attn_boundary,
        attn,
        ffn_boundary: atomic(
            p,
            "ffn_mhc_post_pre",
            local.mhc.clone(),
            MhcFusedPostPreRmsNormKernel::build,
            bridge,
        )?,
        ffn,
    })
}

impl Glm53FlashDpAttnEpModel {
    pub fn cost_tree(&self) -> CostTree {
        let dp = usize::from(self.dp_size);
        let mut builder = CostTreeBuilder::new();
        let prologue = (0..dp)
            .map(|_| {
                CostNode::Sum(vec![
                    self.embedding.compile(&mut builder),
                    self.hc_expand.compile(&mut builder),
                ])
            })
            .collect();
        let mut children = vec![CostNode::Labeled {
            label: format!("{}.prologue [Max over DP ranks]", self.name),
            child: Box::new(CostNode::Max {
                overlap: 1.0,
                children: prologue,
            }),
        }];
        for group in &self.groups {
            children.push(group.compile(dp, &mut builder));
        }
        let epilogue = (0..dp)
            .map(|_| {
                CostNode::Sum(vec![
                    self.terminal_post.compile(&mut builder),
                    self.hc_contract_mean.compile(&mut builder),
                    self.final_norm.compile(&mut builder),
                    self.lm_head.compile(&mut builder),
                ])
            })
            .collect();
        children.push(CostNode::Labeled {
            label: format!("{}.epilogue [Max over DP ranks]", self.name),
            child: Box::new(CostNode::Max {
                overlap: 1.0,
                children: epilogue,
            }),
        });
        let root = CostNode::Labeled {
            label: format!(
                "{} (Glm53FlashDpAttnEpModel) [attention DP{}/TP1; MoE EP{} \
                 all_gatherv/reduce_scatterv; timing_context<={}]",
                self.name, self.dp_size, self.dp_size, self.max_model_len
            ),
            child: Box::new(CostNode::Sum(children)),
        };
        builder.finish(root)
    }

    fn eval_into(&self, input: &UnifiedArchInput, ev: &mut Evaluator) {
        let batch = normalize_input(
            input,
            self.dp_size,
            self.max_model_len,
            &self.cudagraph_capture_sizes,
        )
        .unwrap_or_else(|reason| panic!("invalid Glm53FlashDpAttnEpModel input: {reason}"));
        for rank in &batch.ranks {
            let tokens = ElementwiseKernelInput {
                num_tokens: rank.rows,
            };
            push(&self.embedding, tokens.clone(), ev);
            push(&self.hc_expand, tokens, ev);
        }
        for group in &self.groups {
            group.eval(&batch, ev);
        }
        for rank in &batch.ranks {
            self.terminal_post.eval(
                &MhcTerminalPostInput {
                    num_tokens: rank.rows,
                },
                ev,
            );
            push(
                &self.hc_contract_mean,
                ElementwiseKernelInput {
                    num_tokens: rank.rows,
                },
                ev,
            );
            push(&self.final_norm, RmsNormKernelInput { m: rank.rows }, ev);
            let rows = SingleGemmKernelInput {
                m: rank.logits_rows,
            };
            if rank.logits_rows == 0 {
                ev.push(LeafMetrics::ZERO, || rows.into());
            } else {
                push(&self.lm_head, rows, ev);
            }
        }
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

impl IterwiseUnifiedModel for Glm53FlashDpAttnEpModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        normalize_input(
            batch,
            self.dp_size,
            self.max_model_len,
            &self.cudagraph_capture_sizes,
        )
        .map(|_| ())
    }

    /// One DP rank's KV for a token (the rank that owns the request).
    fn total_kv_bytes_per_token(&self) -> u64 {
        self.total_kv_bytes_per_token
    }

    /// One DP rank's KDA state for a request: all 64 heads of 34 layers.
    fn recurrent_state_bytes_per_request(&self) -> u64 {
        self.recurrent_state_bytes_per_request
    }

    fn recurrent_checkpoint_interval_tokens(&self) -> u32 {
        self.hybrid_block_size
    }

    /// The kpool DSA's top-k cap makes necessary work per-request in context.
    fn logs_decode_kv_lens(&self) -> bool {
        true
    }

    fn gpus_per_replica(&self) -> u16 {
        self.dp_size
    }

    fn num_attn_dp_groups(&self) -> u16 {
        self.dp_size
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

/// One DP rank's view of an iteration.
struct RankBatch {
    /// Rows every kernel outside attention runs on: the scheduled tokens, the
    /// one-token dummy batch of an idle rank, or the common graph size.
    rows: u32,
    /// lm_head rows: one per scheduled request (none for a dummy batch).
    logits_rows: u32,
    kda: Glm53KdaAttnLocalWorkletInput,
    dsa: Glm53DsaAttnLocalWorkletInput,
}

struct NormalizedBatch {
    ranks: Vec<RankBatch>,
    /// Rows every EP rank's fused MoE reads after the all-gather.
    gathered_tokens: u32,
}

fn normalize_input(
    input: &UnifiedArchInput,
    dp_size: u16,
    max_model_len: u32,
    sorted_capture_sizes: &[u32],
) -> std::result::Result<NormalizedBatch, String> {
    let dp = usize::from(dp_size);
    if input.groups.len() != dp {
        return Err(format!(
            "expected {dp} attention-DP groups, got {}",
            input.groups.len()
        ));
    }
    if input.tokens_per_source_rank.len() != dp {
        return Err(format!(
            "expected {dp} EP source-rank token counts, got {}",
            input.tokens_per_source_rank.len()
        ));
    }
    let mut ranks = Vec::with_capacity(dp);
    for (rank, group) in input.groups.iter().enumerate() {
        let mut prefill_tokens = 0_u32;
        for (request, &(prefix, append)) in group.prefill_chunk_pairs.iter().enumerate() {
            if append == 0 {
                return Err(format!(
                    "rank {rank} prefill request {request} append must be nonzero"
                ));
            }
            let context = prefix
                .checked_add(append)
                .ok_or_else(|| format!("rank {rank} prefill request {request} overflows u32"))?;
            if context > max_model_len {
                return Err(format!(
                    "rank {rank} prefill request {request} context {context} exceeds \
                     max_model_len {max_model_len}"
                ));
            }
            prefill_tokens = prefill_tokens
                .checked_add(append)
                .ok_or("prefill token sum overflows u32")?;
        }
        if group.prefill_tokens != prefill_tokens {
            return Err(format!(
                "rank {rank} prefill_tokens {} must equal append sum {prefill_tokens}",
                group.prefill_tokens
            ));
        }
        for (request, &context) in group.decode_kv_lens.iter().enumerate() {
            if !(1..=max_model_len).contains(&context) {
                return Err(format!(
                    "rank {rank} decode request {request} context {context} must be in \
                     1..={max_model_len}"
                ));
            }
        }
        let decode_tokens = group.decode_kv_lens.len() as u32;
        if group.decode_tokens != decode_tokens {
            return Err(format!(
                "rank {rank} decode_tokens {} must equal decode_kv_lens length {decode_tokens}",
                group.decode_tokens
            ));
        }
        let total = prefill_tokens + decode_tokens;
        if group.batch_tokens != total || input.tokens_per_source_rank[rank] != total {
            return Err(format!(
                "rank {rank} batch_tokens {} and source-rank tokens {} must equal \
                 prefill+decode {total}",
                group.batch_tokens, input.tokens_per_source_rank[rank]
            ));
        }
        let (kda, dsa) = if total == 0 {
            // `execute_dummy_batch`: one decode token on dummy slots.
            (
                Glm53KdaAttnLocalWorkletInput {
                    prefill_sequence_lengths: Vec::new(),
                    decode_batch_size: 1,
                },
                Glm53DsaAttnLocalWorkletInput {
                    prefill_chunk_pairs: Vec::new(),
                    decode_kv_lens: vec![1],
                },
            )
        } else {
            (
                Glm53KdaAttnLocalWorkletInput {
                    prefill_sequence_lengths: group
                        .prefill_chunk_pairs
                        .iter()
                        .map(|&(_, append)| append)
                        .collect(),
                    decode_batch_size: decode_tokens,
                },
                Glm53DsaAttnLocalWorkletInput {
                    prefill_chunk_pairs: group.prefill_chunk_pairs.clone(),
                    decode_kv_lens: group.decode_kv_lens.clone(),
                },
            )
        };
        ranks.push(RankBatch {
            rows: total.max(1),
            logits_rows: group.request_count(),
            kda,
            dsa,
        });
    }
    if ranks.iter().all(|rank| rank.logits_rows == 0) {
        return Err("an iteration must carry at least one token".into());
    }
    // Every rank picks a graph only when the busiest one fits a captured size;
    // the synced size is then the max, padded to its graph (`dp_utils.py`).
    // Otherwise every rank runs eager on its real rows.
    let busiest = ranks.iter().map(|rank| rank.rows).max().unwrap_or(1);
    if sorted_capture_sizes
        .last()
        .is_some_and(|&largest| busiest <= largest)
    {
        let padded = graph_padded_tokens(sorted_capture_sizes, busiest);
        ranks.iter_mut().for_each(|rank| rank.rows = padded);
    }
    let gathered_tokens = ranks
        .iter()
        .try_fold(0_u32, |sum, rank| sum.checked_add(rank.rows))
        .ok_or("gathered token sum overflows u32")?;
    Ok(NormalizedBatch {
        ranks,
        gathered_tokens,
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
    use std::path::Path;

    use super::*;
    use crate::arch::contract::ArchGroupInput;
    use crate::timing::routing::RoutingDistribution;

    fn model_cfg() -> Glm53FlashModelCfg {
        config("glm53_flash")
    }

    fn config(name: &str) -> Glm53FlashModelCfg {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("model/config/{name}.json"));
        Glm53FlashModelCfg::from_json(&path).unwrap()
    }

    fn parallel(dp_size: u16) -> Glm53FlashDpAttnEpParallel {
        Glm53FlashDpAttnEpParallel {
            dp_size,
            max_model_len: 131072,
            gpu_name: "NVIDIA B200".into(),
            cudagraph_capture_sizes: Vec::new(),
        }
    }

    fn built(dp_size: u16) -> Glm53FlashDpAttnEpModel {
        built_for(&model_cfg(), dp_size)
    }

    fn built_for(model: &Glm53FlashModelCfg, dp_size: u16) -> Glm53FlashDpAttnEpModel {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let demand = ExpertDemand::popularity(&RoutingDistribution::uniform(288), 42);
        let configs = build_configs(model, &parallel(dp_size), &demand).unwrap();
        build("unified".into(), resolve_configs(&configs), &bridge).unwrap()
    }

    fn prefill(prefix: u32, append: u32) -> ArchGroupInput {
        ArchGroupInput {
            batch_tokens: append,
            prefill_tokens: append,
            decode_tokens: 0,
            prefill_chunk_pairs: vec![(prefix, append)],
            decode_kv_lens: Vec::new(),
            total_kv_len: 0,
        }
    }

    fn input(groups: Vec<ArchGroupInput>) -> UnifiedArchInput {
        UnifiedArchInput {
            tokens_per_source_rank: groups.iter().map(|group| group.batch_tokens).collect(),
            groups,
        }
    }

    #[test]
    fn rank_local_sublayers_are_tp1_and_experts_split_over_dp() {
        let demand = ExpertDemand::popularity(&RoutingDistribution::uniform(288), 42);
        let configs = build_configs(&model_cfg(), &parallel(8), &demand).unwrap();
        let resolved = resolve_configs(&configs);
        assert_eq!(configs.local.kda.num_heads.get(), 64);
        assert_eq!(configs.local.dsa.num_heads.get(), 64);
        assert_eq!(configs.local.dense_ffn.intermediate.get(), 12288);
        assert_eq!(configs.local.shared_expert.intermediate.get(), 2048);
        assert_eq!(configs.local.lm_head.n.get(), model_cfg().vocab_size);
        assert_eq!(configs.routed.len(), 8);
        assert_eq!(resolved.routed[0].fused_moe.num_local_experts.get(), 36);
        let Glm53FlashDispatchConfig::Fp8(dispatch) = &configs.dispatch else {
            panic!("the FP8 checkpoint dispatches FP8 activations");
        };
        assert_eq!(dispatch.hidden_dtype, DType::Fp8E4m3);
        assert_eq!((dispatch.num_gpus, dispatch.max_total_tokens), (8, 65536));
        assert!(build_configs(&model_cfg(), &parallel(1), &demand).is_err());
        assert!(build_configs(&model_cfg(), &parallel(5), &demand).is_err());
    }

    #[test]
    fn a_rank_checkpoints_at_the_tp1_hybrid_block() {
        // All 64 KDA heads on one rank: vLLM's TP1 block (the TP arch's
        // `hybrid_block_tokens` test pins TP4 to the logged 2176).
        assert_eq!(built(4).recurrent_checkpoint_interval_tokens(), 8576);
    }

    #[test]
    fn kv_and_state_are_one_ranks_full_heads_not_replicated() {
        for dp in [4, 8] {
            let model = built(dp);
            assert_eq!(model.total_kv_bytes_per_token(), 11 * (512 + 33));
            // The true state is (4 << 20) + 24576 * 3 * 2 bytes per layer;
            // vLLM pads it to one 8576-token attention page.
            assert_eq!(model.recurrent_state_bytes_per_request(), 34 * 8576 * 512);
            assert_eq!(model.gpus_per_replica(), dp);
            assert_eq!(model.num_attn_dp_groups(), dp);
            assert_eq!(model.num_attn_shards(), 1);
        }
    }

    #[test]
    fn compiled_tree_has_a_fixed_slot_count() {
        for dp in [4_usize, 8] {
            let model = built(dp as u16);
            // Per rank: boundaries 2 + attention (KDA 13 / DSA 31) + FFN-local
            // (dense 5; MoE router 2 + glue 1 + serial shared 5 + quant 1).
            // MoE group-wide: dispatch + dp fused + combine + dp x 5 concurrent
            // shared + dp combine glue. Prologue 2/rank, epilogue 4/rank.
            let moe_global = 1 + dp + 1 + 5 * dp + dp;
            let kda_dense = dp * (2 + 13 + 5);
            let dsa_moe = dp * (2 + 31 + 9) + moe_global;
            let kda_moe = dp * (2 + 13 + 9) + moe_global;
            assert_eq!(
                model.n_slots,
                2 * dp + 2 * kda_dense + dsa_moe + kda_moe + 4 * dp
            );
            assert_eq!(model.cost_log_manifest().slots.len(), model.n_slots);
        }
    }

    /// The analyzer picks a location map by arch type and the exact set of
    /// non-communication locations, so EP4 and EP8 each need their own map.
    fn assert_maps_cover(model: &Glm53FlashModelCfg, arch: &str, locations: impl Fn(u16) -> usize) {
        use std::collections::BTreeSet;
        for dp in [4_u16, 8] {
            let model = built_for(model, dp);
            assert!(model.logs_decode_kv_lens());
            let actual: BTreeSet<String> = model
                .cost_log_manifest()
                .slots
                .into_iter()
                .map(|slot| slot.name)
                .collect();
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("model/work/location_maps/{arch}_ep{dp}.json"));
            let map: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            assert_eq!(map["arch_types"], serde_json::json!([arch]));
            let mapped: BTreeSet<String> = map["locations"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row["location"].as_str().unwrap().to_string())
                .collect();
            assert_eq!(actual, mapped, "EP{dp}");
            assert_eq!(mapped.len(), locations(dp));
        }
    }

    #[test]
    fn necessary_work_maps_cover_the_compiled_locations() {
        // The TP map's 128 less 8 per-rank input quants, plus per MoE kind
        // the dispatch quant, gather, scatter and ranks 4..dp.
        assert_maps_cover(&model_cfg(), ARCH_KIND, |dp| {
            128 - 8 + 2 * (3 + usize::from(dp) - 4)
        });
    }

    #[test]
    fn nvfp4_maps_add_the_router_select_and_drop_the_shared_expert_quant() {
        // Per MoE kind: +1 vLLM top-k select, -2 BF16 shared-expert quants.
        assert_maps_cover(
            &config("glm53_flash_nvfp4"),
            "glm53_flash_vllm_nvfp4_dp_attn_ep_moe",
            |dp| 128 - 8 + 2 * (3 + usize::from(dp) - 4) - 2,
        );
    }

    #[test]
    fn nvfp4_dispatches_packed_fp4_after_a_vllm_top_k_select() {
        let demand = ExpertDemand::popularity(&RoutingDistribution::uniform(288), 42);
        let model = config("glm53_flash_nvfp4");
        let configs = build_configs(&model, &parallel(4), &demand).unwrap();
        let Glm53FlashDispatchConfig::Nvfp4(dispatch) = &configs.dispatch else {
            panic!("the NVFP4 checkpoint dispatches packed NVFP4 activations");
        };
        assert_eq!(dispatch.top_k.get(), model.num_experts_per_tok);
        assert!(configs.router_select.is_some());
        let resolved = resolve_configs(&configs);
        assert_eq!(
            resolved.routed[0].fused_moe.backends,
            ["flashinfer_trtllm_routed_sm100"]
        );
        assert_eq!(resolved.routed[0].fused_moe.routing_method, "deepseek_v3");
    }

    fn leaf_order(node: &CostNode, out: &mut Vec<usize>) {
        match node {
            CostNode::Leaf(slot) => out.push(*slot),
            CostNode::Sum(children)
            | CostNode::Max { children, .. }
            | CostNode::Parallel { children, .. } => {
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
        leaf_order(&built(4).cost_tree().root, &mut order);
        assert_eq!(order, (0..order.len()).collect::<Vec<_>>());
    }

    #[test]
    fn eager_batches_gather_ragged_rows_and_idle_ranks_run_one_dummy_token() {
        let batch = normalize_input(
            &input(vec![
                prefill(0, 8192),
                prefill(8192, 384),
                ArchGroupInput::default(),
                prefill(0, 100),
            ]),
            4,
            131072,
            &[],
        )
        .unwrap();
        let rows: Vec<_> = batch.ranks.iter().map(|rank| rank.rows).collect();
        assert_eq!(rows, [8192, 384, 1, 100]);
        assert_eq!(batch.gathered_tokens, 8192 + 384 + 1 + 100);
        assert_eq!(batch.ranks[2].logits_rows, 0);
        assert_eq!(batch.ranks[2].dsa.decode_kv_lens, [1]);
        assert_eq!(batch.ranks[1].dsa.prefill_chunk_pairs, [(8192, 384)]);
    }

    #[test]
    fn graph_sized_batches_pad_every_rank_to_the_busiest_ranks_graph() {
        let sizes = [1, 2, 4, 8, 16, 24, 32];
        let fits = normalize_input(
            &input(vec![
                prefill(0, 3),
                prefill(0, 17),
                ArchGroupInput::default(),
                prefill(0, 1),
            ]),
            4,
            131072,
            &sizes,
        )
        .unwrap();
        assert!(fits.ranks.iter().all(|rank| rank.rows == 24));
        assert_eq!(fits.gathered_tokens, 96);
        // One rank above the largest graph: everyone runs eager, unpadded.
        let eager = normalize_input(
            &input(vec![
                prefill(0, 3),
                prefill(0, 40),
                ArchGroupInput::default(),
                prefill(0, 1),
            ]),
            4,
            131072,
            &sizes,
        )
        .unwrap();
        let rows: Vec<_> = eager.ranks.iter().map(|rank| rank.rows).collect();
        assert_eq!(rows, [3, 40, 1, 1]);
    }

    #[test]
    fn inconsistent_or_empty_inputs_are_rejected() {
        let mut bad = input(vec![
            prefill(0, 8),
            prefill(0, 8),
            prefill(0, 8),
            prefill(0, 8),
        ]);
        bad.tokens_per_source_rank[1] = 7;
        assert!(normalize_input(&bad, 4, 131072, &[]).is_err());
        let empty = input(vec![ArchGroupInput::default(); 4]);
        assert!(normalize_input(&empty, 4, 131072, &[]).is_err());
        let three = input(vec![prefill(0, 8); 3]);
        assert!(normalize_input(&three, 4, 131072, &[]).is_err());
        let long = input(vec![
            prefill(131000, 100),
            prefill(0, 8),
            prefill(0, 8),
            prefill(0, 8),
        ]);
        assert!(normalize_input(&long, 4, 131072, &[]).is_err());
    }

    #[test]
    fn every_dp_size_accepts_mixed_ragged_iterations() {
        for dp in [4_u16, 8] {
            let model = built(dp);
            let mut groups = vec![ArchGroupInput::default(); usize::from(dp)];
            groups[0] = prefill(0, 4096);
            groups[1] = ArchGroupInput {
                batch_tokens: 3,
                prefill_tokens: 0,
                decode_tokens: 3,
                prefill_chunk_pairs: Vec::new(),
                decode_kv_lens: vec![10, 20, 30],
                total_kv_len: 60,
            };
            let batch = input(groups);
            model.check_input(&batch).unwrap();
        }
    }
}
