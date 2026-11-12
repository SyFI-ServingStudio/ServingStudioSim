//! GLM-5.3-Flash (FP8 block or NVFP4, [`Glm53FlashQuant`]) on B200 under vLLM
//! data-parallel attention with expert-parallel MoE (`--data-parallel-size N
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
//!
//! [`Glm53FlashDpAttnEpModel::with_hisparse`] serves the 11 DSA layers' cache
//! through SGLang HiSparse (`arch::hisparse`, a rough analytic model, as on
//! the GLM-5.3 DP graph): their MLA latent moves to host memory, their kpool
//! index cache and a fixed latent buffer per request stay on the GPU, and each
//! step adds the exposed host -> device miss copy as one more `Max` over
//! ranks. The 34 KDA layers' recurrent state and conv window cannot be
//! offloaded and stay on the GPU unchanged.
//!
//! The NVFP4 checkpoint's exchange can instead run as flashinfer's two-sided
//! MNNVL all-to-all (`ep_comm: alltoall`, vLLM `--all2all-backend
//! flashinfer_nvlink_two_sided`), or either exchange can be priced by
//! `arch::ep_comm`'s analytic model (`ep_comm_model: analytic`); the leaves
//! are `arch::ep_exchange`'s, as on the GLM-5.3 DP graph, with rows sized from
//! this model's hidden size (4096), top-k (8) and 288 experts. The profiled
//! EP8 `moe_alltoall` rows (`ep_comm_model: measured`) are keyed by those
//! widths, so they need the Flash shapes in `profile.db`. The FP8 checkpoint
//! keeps the measured all-gather / reduce-scatter. The default, measured
//! all-gather / reduce-scatter, is unchanged.
//!
//! # Attention DP x TP (`attn_tp > 1`)
//!
//! Not a vLLM layout; a design the simulator prices from measured leaves.
//!
//! The `dp_size` GPUs form `dp_size / attn_tp` attention groups of `attn_tp`
//! GPUs. A group's requests share one scheduler and KV partition; its KDA and
//! MLA heads, and their projections, are split over the group's ranks (the TP
//! arch's `tp_size = attn_tp` worklets), so every rank runs attention on all
//! of the group's rows. Everything else is sequence-parallel: each rank owns
//! `ceil(rows / attn_tp)` of the group's rows for the mHC boundaries, the dense
//! FFN, the router, the shared expert and the EP dispatch, so every EP source
//! rank sends its own slice and the expert exchange is the DP one over
//! `dp_size` ranks. Per layer the group moves the attention input from slices
//! to all rows (all-gather) and the attention output back (reduce-scatter);
//! the pair is priced as one TP all-reduce of the group's rows (FlashInfer
//! MNNVL up to its workspace, else the faster of NCCL / NVSHMEM). MLA latent,
//! indexer cache and kpool tail are replicated over the group's ranks; the KDA
//! state is sharded by head.

use anyhow::Result;

use crate::arch::contract::{IterwiseUnifiedModel, KvHostOffload, UnifiedArchInput};
use crate::arch::ep_comm::{EpCommConfig, EpCommKind};
use crate::arch::ep_exchange::{
    AllgatherLeafNames, EpExchange, EpExchangeConfigs, EpExchangeGeometry,
};
use crate::arch::glm53_flash_vllm_fp8_kda_dsa_moe::{
    self as tp_arch, atomic, attention_inputs, graph_padded_tokens, push, AttnKind, Boundary,
    FfnKind, Glm53FlashLayerGroup, Glm53FlashModelCfg, Glm53FlashQuant, Glm53FlashVllmConfigs,
    Glm53FlashVllmParallel, ACTIVATION_DTYPE, SHARED_EXPERTS_STREAM_OVERLAP,
    SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD,
};
use crate::arch::hisparse::{push_rank_step, HisparseConfig, HisparseRankStep};
use crate::arch::kv_layout::{KvCacheLayout, KvCachePart};
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
/// Above FlashInfer's fused workspace the attention-TP all-reduce takes the
/// faster measured large all-reduce (best of N per shape).
const LARGE_ATTN_ALL_REDUCE_BACKENDS: &[&str] = &["nccl", "nvshmem"];

/// Deployment layout: `dp_size` engine cores, one GPU each, forming one EP group.
#[derive(Clone, Debug)]
pub struct Glm53FlashDpAttnEpParallel {
    pub dp_size: u16,
    pub max_model_len: u32,
    pub gpu_name: String,
    /// vLLM `--cudagraph-capture-sizes`; empty runs eager (no padding).
    pub cudagraph_capture_sizes: Vec<u32>,
    pub kernel_path: tp_arch::Glm53FlashKernelPath,
    /// How the expert exchange runs and is priced (`arch::ep_comm`). The
    /// default is the measured all-gather / reduce-scatter; anything else
    /// needs the NVFP4 checkpoint.
    pub ep_comm: EpCommConfig,
    /// What-if: fixed us taken off each fused mHC post/pre boundary (the
    /// leaf keeps only its per-row part and is flagged `extrapolated`).
    pub mhc_fuse_fixed_us: Option<f64>,
    /// Attention tensor-parallel degree inside each attention DP group (module
    /// doc, "Attention DP x TP"): `dp_size / attn_tp` groups; 1 is plain DP
    /// attention.
    pub attn_tp: u16,
    /// Price the attention-TP all-reduce from rows measured at this GPU count
    /// instead of `attn_tp` (a stand-in while the `attn_tp` rows are missing;
    /// the cost-tree label says so). `None` uses `attn_tp`.
    pub attn_all_reduce_row_gpus: Option<u16>,
    /// Which attention layers run at `attn_tp` (module doc).
    pub attn_tp_layers: AttnTpLayers,
    /// How the attention-TP collective is priced (module doc).
    pub attn_tp_collective: AttnTpCollective,
    /// KDA `f_b` / `g_b` as one batched GEMM (worklet
    /// `Glm53KdaAttnLocalWorkletConfig::merge_gate_b`).
    pub kda_gate_b_merged: bool,
}

/// Which attention layers an `attn_tp > 1` group runs tensor-parallel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttnTpLayers {
    /// KDA and DSA (MLA cache replicated over the group).
    #[default]
    All,
    /// KDA only; DSA stays rank-local DP on each request's owner rank.
    KdaOnly,
}

/// Pricing of the attention-TP collective (one all-reduce of the group's rows).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttnTpCollective {
    /// FlashInfer MNNVL up to its workspace, else the faster of NCCL/NVSHMEM.
    #[default]
    Fast,
    /// NCCL only (the pynccl path).
    Nccl,
    /// `fast`, plus FlashInfer's MNNVL two-shot with the workspace sized to
    /// the request above vLLM's budget (not vLLM dispatch); the fastest
    /// measured candidate is billed.
    FastUnbudgeted,
}

/// The unbudgeted MNNVL two-shot candidate (`AttnTpCollective::FastUnbudgeted`).
const UNBUDGETED_ATTN_ALL_REDUCE_BACKENDS: &[&str] = &["flashinfer_mnnvl_unbudgeted_twoshot"];
const UNBUDGETED_MAX_TOKENS_PER_RANK: u32 =
    crate::timing::kernels::all_reduce_fusion::UNBUDGETED_ROWS_PER_RANK[4];

#[derive(Clone, Debug)]
pub struct Glm53FlashDpAttnEpConfigs {
    /// Every rank-local sublayer at TP 1 (64 KDA/MLA heads, full dense FFN,
    /// shared expert and vocab). Its `routed` and `all_reduce` are unused.
    /// Under `attn_tp > 1` its KDA (and, unless KDA-only, DSA) configs and
    /// `all_reduce` are the `attn_tp` ones.
    pub local: Glm53FlashVllmConfigs,
    /// The TP1 KDA config: the RecoverSSM commit and, KDA-only, the KV layout
    /// are priced per owner rank from it.
    pub kda_tp1: crate::worklet::Glm53KdaAttnLocalWorkletConfig,
    pub parallel: Glm53FlashDpAttnEpParallel,
    /// One per EP rank, ranked by routed workload.
    pub routed: Vec<Glm53RoutedMoeLocalWorkletConfig>,
    pub dispatch: Glm53FlashDispatchConfig,
    /// vLLM's grouped top-k ahead of an NVFP4 dispatch: fp32 logits and bias
    /// in, top-k IDs and weights out. FP8 routes inside the fused MoE.
    pub router_select: Option<ElementwiseKernelConfig>,
    pub combine: MoeEpReduceScatterKernelConfig,
    /// The exchange `ep_comm` puts in place of `dispatch` / `combine`; empty
    /// on the measured all-gather / reduce-scatter.
    pub exchange: EpExchangeConfigs,
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
    if model.quant != Glm53FlashQuant::Nvfp4 && !parallel.ep_comm.is_default_measured() {
        return Err(fit_failed(
            "ep_comm other than the measured all-gather / reduce-scatter needs the NVFP4 \
             checkpoint (its NVFP4 activations and block scales are the exchanged rows)",
        ));
    }
    let exchange = EpExchangeConfigs::new(
        &parallel.ep_comm,
        dp,
        &parallel.gpu_name,
        EpExchangeGeometry {
            hidden: model.hidden,
            top_k: model.num_experts_per_tok,
            num_experts: model.n_routed_experts,
        },
        demand,
    )
    .map_err(fit_failed)?;
    let local = tp_arch::build_configs(
        model,
        &Glm53FlashVllmParallel {
            tp_size: 1,
            // Whole routed intermediate; the EP split over DP ranks follows.
            enable_expert_parallel: true,
            max_model_len: parallel.max_model_len,
            gpu_name: parallel.gpu_name.clone(),
            cudagraph_capture_sizes: parallel.cudagraph_capture_sizes.clone(),
            kernel_path: parallel.kernel_path.clone(),
        },
        demand,
    )?;
    let attn_tp = parallel.attn_tp.max(1);
    if dp % attn_tp != 0 {
        return Err(fit_failed(format!(
            "attention TP {attn_tp} must divide the {dp} EP ranks"
        )));
    }
    let mut local = local;
    local.kda.merge_gate_b = parallel.kda_gate_b_merged;
    let kda_tp1 = local.kda.clone();
    if attn_tp > 1 {
        let attn = tp_arch::build_configs(
            model,
            &Glm53FlashVllmParallel {
                tp_size: attn_tp,
                enable_expert_parallel: true,
                max_model_len: parallel.max_model_len,
                gpu_name: parallel.gpu_name.clone(),
                cudagraph_capture_sizes: parallel.cudagraph_capture_sizes.clone(),
                kernel_path: parallel.kernel_path.clone(),
            },
            demand,
        )?;
        local.kda = attn.kda;
        local.kda.merge_gate_b = parallel.kda_gate_b_merged;
        if parallel.attn_tp_layers == AttnTpLayers::All {
            local.dsa = attn.dsa;
        }
        let row_gpus = u32::from(parallel.attn_all_reduce_row_gpus.unwrap_or(attn_tp));
        local.all_reduce = attn.all_reduce;
        local.all_reduce.num_gpus = row_gpus;
        local.large_all_reduce = attn.large_all_reduce;
        local.large_all_reduce.num_gpus = row_gpus;
        local.large_all_reduce.backends = match parallel.attn_tp_collective {
            AttnTpCollective::Fast | AttnTpCollective::FastUnbudgeted => {
                LARGE_ATTN_ALL_REDUCE_BACKENDS.to_vec()
            }
            AttnTpCollective::Nccl => vec!["nccl"],
        };
    }
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
        kda_tp1,
        parallel: parallel.clone(),
        exchange,
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
    exchange: Exchange,
    /// One per EP rank, ranked by routed workload.
    fused_moe: Vec<Op<Nvfp4FusedMoeKernel>>,
    combine_glue: Op<ElementwiseKernel>,
}

/// How the routed tokens reach the experts and come back.
enum Exchange {
    /// The profiled all-gatherv and bf16 reduce-scatterv.
    MeasuredAllgather {
        dispatch: Dispatch,
        combine: Op<MoeEpReduceScatterKernel>,
    },
    /// What `ep_comm` selects instead (`arch::ep_exchange`).
    Configured(EpExchange),
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
    /// The attention-TP all-gather + reduce-scatter pair, priced as one TP
    /// all-reduce of the group's rows; `None` at attention TP 1.
    attn_all_reduce: Option<tp_arch::TpAllReduce>,
    ffn_boundary: Op<MhcFusedPostPreRmsNormKernel>,
    ffn: Ffn,
    mhc_fuse_fixed_us: Option<f64>,
}

/// A fused mHC post/pre boundary, less `fixed_us` when the what-if is on.
fn push_mhc_fused(
    op: &Op<MhcFusedPostPreRmsNormKernel>,
    num_tokens: u32,
    fixed_us: Option<f64>,
    ev: &mut Evaluator,
) {
    let input = MhcRmsNormKernelInput { num_tokens };
    let mut metrics = op.kernel.eval(&input);
    if let Some(fixed) = fixed_us {
        let measured = f64::from(metrics.m.time_ms);
        metrics.m.time_ms = (measured - fixed / 1e3).max(0.0) as f32;
        metrics.coverage = metrics.coverage | crate::timing::CoverageFlags::EXTRAPOLATED;
    }
    ev.push(metrics, || input.clone().into());
}

impl LayerGroup {
    /// Per rank: everything up to the dispatch (the whole layer for a dense
    /// FFN). Then, for an MoE layer, the group-wide expert section and the
    /// per-rank combine add.
    fn compile(
        &self,
        dp: usize,
        kda_groups: usize,
        kda_only: bool,
        builder: &mut CostTreeBuilder,
    ) -> CostNode {
        if kda_only {
            if let Attention::Kda(worklet) = &self.attn {
                return self.compile_kda_tp(worklet, dp, kda_groups, builder);
            }
        }
        let ranks = (0..dp)
            .map(|_| {
                let mut leaves = vec![
                    self.attn_boundary.compile(builder),
                    match &self.attn {
                        Attention::Kda(worklet) => worklet.compile(builder),
                        Attention::Dsa(worklet) => worklet.compile(builder),
                    },
                ];
                if !kda_only {
                    leaves.extend(self.attn_all_reduce.as_ref().map(|op| op.compile(builder)));
                }
                leaves.push(self.ffn_boundary.compile(builder));
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

    /// KDA-only attention TP: every owner rank runs its boundary, the group
    /// all-gathers the KDA input, runs KDA on all its rows with heads split,
    /// reduce-scatters the output back to the owners (the pair priced as one
    /// all-reduce of the group's rows), and every owner rank runs the rest.
    fn compile_kda_tp(
        &self,
        worklet: &Glm53KdaAttnLocalWorklet,
        dp: usize,
        kda_groups: usize,
        builder: &mut CostTreeBuilder,
    ) -> CostNode {
        let boundary = CostNode::Max {
            overlap: 1.0,
            children: (0..dp)
                .map(|_| self.attn_boundary.compile(builder))
                .collect(),
        };
        let kda = CostNode::Max {
            overlap: 1.0,
            children: (0..kda_groups)
                .map(|_| {
                    let mut leaves = vec![worklet.compile(builder)];
                    leaves.extend(self.attn_all_reduce.as_ref().map(|op| op.compile(builder)));
                    CostNode::Sum(leaves)
                })
                .collect(),
        };
        let ranks = (0..dp)
            .map(|_| {
                let mut leaves = vec![self.ffn_boundary.compile(builder)];
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
        let mut body = vec![
            CostNode::Labeled {
                label: format!("{} attention boundary [Max over owner ranks]", self.label),
                child: Box::new(boundary),
            },
            CostNode::Labeled {
                label: format!(
                    "{} KDA tensor-parallel over the group [Max over groups; all-gather + \
                     reduce-scatter priced as one all-reduce of the group's rows]",
                    self.label
                ),
                child: Box::new(kda),
            },
            CostNode::Labeled {
                label: format!("{} [Max over owner ranks: FFN up to the dispatch]", self.label),
                child: Box::new(CostNode::Max {
                    overlap: 1.0,
                    children: ranks,
                }),
            },
        ];
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
        if batch.kda_only {
            if let Attention::Kda(worklet) = &self.attn {
                for rank in &batch.ranks {
                    self.eval_attn_boundary(rank.rows, ev);
                }
                for group in &batch.kda_groups {
                    worklet.eval(&group.kda, ev);
                    if let Some(all_reduce) = &self.attn_all_reduce {
                        all_reduce.eval(group.rows, ev);
                    }
                }
                for rank in &batch.ranks {
                    self.eval_ffn_local(rank, ev);
                }
                if let Ffn::Moe(block) = &self.ffn {
                    block.eval_expert_sections(batch, ev);
                }
                return;
            }
        }
        for rank in &batch.ranks {
            self.eval_attn_boundary(rank.rows, ev);
            match &self.attn {
                Attention::Kda(worklet) => worklet.eval(&rank.kda, ev),
                Attention::Dsa(worklet) => worklet.eval(&rank.dsa, ev),
            }
            // KDA-only: DSA layers are rank-local and reduce nothing.
            if !batch.kda_only {
                if let Some(all_reduce) = &self.attn_all_reduce {
                    all_reduce.eval(rank.attn_rows, ev);
                }
            }
            self.eval_ffn_local(rank, ev);
        }
        if let Ffn::Moe(block) = &self.ffn {
            block.eval_expert_sections(batch, ev);
        }
    }

    fn eval_attn_boundary(&self, rows: u32, ev: &mut Evaluator) {
        match &self.attn_boundary {
            Boundary::Fused(op) => push_mhc_fused(op, rows, self.mhc_fuse_fixed_us, ev),
            boundary => boundary.eval(rows, ev),
        }
    }

    fn eval_ffn_local(&self, rank: &RankBatch, ev: &mut Evaluator) {
        push_mhc_fused(&self.ffn_boundary, rank.rows, self.mhc_fuse_fixed_us, ev);
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
}

impl MoeBlock {
    /// The dispatch / routed experts / combine chain, which the aux-stream
    /// shared expert (T_r <= 256) overlaps, then the per-rank shared + routed add.
    fn compile_expert_sections(&self, dp: usize, builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut chain = match &self.exchange {
            Exchange::MeasuredAllgather { dispatch, .. } => vec![dispatch.compile(builder)],
            Exchange::Configured(exchange) => exchange.compile_dispatch(dp, builder),
        };
        chain.push(CostNode::Labeled {
            label: format!("{}.ep_ranks (max over EP ranks)", self.name),
            child: Box::new(CostNode::Max {
                overlap: 1.0,
                children: self
                    .fused_moe
                    .iter()
                    .map(|op| op.compile(builder))
                    .collect(),
            }),
        });
        let path = match &self.exchange {
            Exchange::MeasuredAllgather { combine, .. } => {
                chain.push(combine.compile(builder));
                format!(
                    "all_gatherv -> EP{} routed experts -> reduce_scatterv",
                    self.fused_moe.len()
                )
            }
            Exchange::Configured(exchange) => {
                chain.extend(exchange.compile_combine(dp, builder));
                format!(
                    "{} around EP{} routed experts",
                    exchange.label(),
                    self.fused_moe.len()
                )
            }
        };
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
                    "{} (MoE) [{path}; aux-stream shared expert at T_r<={} overlaps]",
                    self.name, SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
                ),
                child: Box::new(CostNode::Parallel {
                    overlap: SHARED_EXPERTS_STREAM_OVERLAP,
                    children: vec![CostNode::Sum(chain), concurrent_shared],
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
        // EP source ranks: each rank's own (sequence-parallel) slice.
        let tokens: Vec<u32> = batch.source_rank_tokens.clone();
        let step = match &self.exchange {
            Exchange::MeasuredAllgather { dispatch, .. } => {
                dispatch.eval(
                    MoeEpCollectiveKernelInput {
                        per_rank_tokens: tokens.clone(),
                    },
                    ev,
                );
                None
            }
            Exchange::Configured(exchange) => {
                let step = exchange.step(&tokens);
                exchange.eval_dispatch(&step, &tokens, ev);
                Some(step)
            }
        };
        for op in &self.fused_moe {
            push(
                op,
                Nvfp4FusedMoeKernelInput {
                    num_tokens: batch.gathered_tokens,
                },
                ev,
            );
        }
        match (&self.exchange, step) {
            (Exchange::MeasuredAllgather { combine, .. }, _) => push(
                combine,
                MoeEpCollectiveKernelInput {
                    per_rank_tokens: tokens.clone(),
                },
                ev,
            ),
            (Exchange::Configured(exchange), Some(step)) => {
                exchange.eval_combine(&step, &tokens, ev)
            }
            (Exchange::Configured(_), None) => unreachable!("the dispatch computed the step"),
        }
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
    /// Attention TP inside each attention DP group; 1 is plain DP attention.
    pub attn_tp: u16,
    /// `Some(n)` when the attention all-reduce is priced from `n`-GPU rows.
    attn_all_reduce_row_gpus: Option<u16>,
    attn_tp_layers: AttnTpLayers,
    attn_tp_collective: AttnTpCollective,
    pub max_model_len: u32,
    /// Query rows each decode request presents (see `Glm53FlashVllmModel`).
    pub verify_width: u32,
    cudagraph_capture_sizes: Vec<u32>,
    ep_comm_kind: EpCommKind,
    embedding: Op<ElementwiseKernel>,
    hc_expand: Op<ElementwiseKernel>,
    groups: Vec<LayerGroup>,
    terminal_post: MhcTerminalPostOp,
    hc_contract_mean: Op<ElementwiseKernel>,
    final_norm: Op<RmsNormKernel>,
    lm_head: Op<SingleGemmKernel>,
    kv_cache_layout: KvCacheLayout,
    /// The RecoverSSM commit of a verify graph (see `Glm53FlashVllmModel`).
    pub(crate) kda_commit: Option<tp_arch::KdaCommitPlan>,
    /// The layers with a per-token cache, and one layer's MLA latent of one
    /// token: what HiSparse moves to host memory.
    num_dsa_layers: u32,
    dsa_latent_bytes_per_token: u64,
    /// SGLang HiSparse KV offload (see [`Glm53FlashDpAttnEpModel::with_hisparse`]);
    /// `None` keeps the whole cache on the GPU.
    hisparse: Option<HisparseConfig>,
    host_offload: Option<KvHostOffload>,
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
    // One rank holds a request's whole KV and all 64 KDA heads: nothing is
    // replicated across ranks, and the block is vLLM's TP1 hybrid block.
    let dsa_layers = model_cfg.num_dsa_layers();
    let kda_layers = model_cfg.kda_layers.len() as u32;
    // TP1 resolution: the commit is priced per owner rank from it, and under
    // KDA-only each owner rank holds its own requests' whole KDA state (the
    // group's ranks hold equal head slices of equal request counts).
    let mut tp1_cfg = local.clone();
    tp1_cfg.kda = cfg.kda_tp1.clone();
    tp1_cfg.kda.decode_query_len = local.kda.decode_query_len;
    tp1_cfg.kda.ssm_state_dtype = local.kda.ssm_state_dtype;
    let tp1 = tp_arch::resolve_configs(&tp1_cfg);
    let kda_only = cfg.parallel.attn_tp_layers == AttnTpLayers::KdaOnly;
    let layout_tp = if kda_only { 1 } else { cfg.parallel.attn_tp.max(1) };
    let kda_state_bytes = tp_arch::kda_layer_state_bytes_per_request(if kda_only {
        &tp1
    } else {
        &resolved.local
    });
    let kv_cache_layout = tp_arch::hybrid_kv_cache_layout(
        model_cfg,
        kda_state_bytes,
        dsa_layers,
        kda_layers,
        kda_layers.div_ceil(dsa_layers.max(1)),
        tp_arch::hybrid_block_tokens(kda_state_bytes, u64::from(model_cfg.kv_lora_rank)),
        layout_tp,
    );
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
        ep_comm_kind: cfg.parallel.ep_comm.kind,
        attn_tp: cfg.parallel.attn_tp.max(1),
        attn_all_reduce_row_gpus: cfg.parallel.attn_all_reduce_row_gpus,
        attn_tp_layers: cfg.parallel.attn_tp_layers,
        attn_tp_collective: cfg.parallel.attn_tp_collective,
        max_model_len: cfg.parallel.max_model_len,
        verify_width: tp_arch::verify_width(&cfg.local)?,
        cudagraph_capture_sizes: {
            let mut sizes = cfg.parallel.cudagraph_capture_sizes.clone();
            sizes.sort_unstable();
            sizes.dedup();
            sizes
        },
        kv_cache_layout,
        kda_commit: tp_arch::kda_commit_plan(&tp1),
        num_dsa_layers: dsa_layers,
        dsa_latent_bytes_per_token: tp_arch::dsa_layer_latent_bytes_per_token(model_cfg),
        hisparse: None,
        host_offload: None,
        cost_flat: Vec::new(),
        n_slots: 0,
        name,
    };
    model.recompile();
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
            local.mhc_fused.clone(),
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
                exchange: build_exchange(m, cfg, bridge)?,
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
    let tp_layer = cfg.parallel.attn_tp > 1
        && (cfg.parallel.attn_tp_layers == AttnTpLayers::All || group.attn == AttnKind::Kda);
    let attn_all_reduce = tp_layer
        .then(|| tp_arch::TpAllReduce::build(p, "attn_tp_all_reduce", local, bridge))
        .transpose()?
        .map(|op| -> std::result::Result<_, BuildError> {
            Ok(match cfg.parallel.attn_tp_collective {
                AttnTpCollective::Fast => op,
                AttnTpCollective::Nccl => op.without_fused(),
                AttnTpCollective::FastUnbudgeted => {
                    let mut unbudgeted = local.all_reduce.clone();
                    unbudgeted.backends = UNBUDGETED_ATTN_ALL_REDUCE_BACKENDS.to_vec();
                    unbudgeted.fused_token_limit =
                        Some(UNBUDGETED_MAX_TOKENS_PER_RANK * unbudgeted.num_gpus);
                    op.with_unbudgeted(atomic(
                        p,
                        "attn_tp_all_reduce_unbudgeted",
                        unbudgeted,
                        crate::timing::kernels::AllReduceFusionKernel::build,
                        bridge,
                    )?)
                }
            })
        })
        .transpose()?;
    Ok(LayerGroup {
        label: group.label.clone(),
        layers: group.layers.clone(),
        attn_boundary,
        attn,
        attn_all_reduce,
        ffn_boundary: atomic(
            p,
            "ffn_mhc_post_pre",
            local.mhc_fused.clone(),
            MhcFusedPostPreRmsNormKernel::build,
            bridge,
        )?,
        ffn,
        mhc_fuse_fixed_us: cfg.parallel.mhc_fuse_fixed_us,
    })
}

/// The configured exchange, else the measured all-gatherv / reduce-scatterv.
fn build_exchange(
    m: &str,
    cfg: &Glm53FlashDpAttnEpConfigs,
    bridge: &PerfApiBridge,
) -> std::result::Result<Exchange, BuildError> {
    let configured = EpExchange::build(
        m,
        AllgatherLeafNames {
            dispatch: "dispatch_all_gather",
            combine: "combine_reduce_scatter",
        },
        &cfg.exchange,
        bridge,
    )?;
    if let Some(exchange) = configured {
        return Ok(Exchange::Configured(exchange));
    }
    Ok(Exchange::MeasuredAllgather {
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
        combine: atomic(
            m,
            "combine_reduce_scatter",
            cfg.combine.clone(),
            MoeEpReduceScatterKernel::build,
            bridge,
        )?,
    })
}

impl Glm53FlashDpAttnEpModel {
    fn recompile(&mut self) {
        let tree = self.cost_tree();
        self.cost_flat = tree.flatten();
        self.n_slots = tree.n_slots();
    }

    /// Serve the DSA layers' cache through SGLang HiSparse (`arch::hisparse`),
    /// as [`Glm52VllmNvfp4DpAttnDsaMoeModel::with_hisparse`] does for GLM-5.3:
    /// each rank keeps the 11 DSA layers' kpool index cache per token and a
    /// `buffer_tokens` latent buffer per request on the GPU, their whole
    /// latent in its share of the node's host DRAM (`gpus_per_node` GPUs per
    /// node unless the config says otherwise), and pays the exposed host ->
    /// device miss copy of those 11 layers as one more `Max` over ranks per
    /// step. The KDA state is not offloadable and is unchanged. Kernels are
    /// unchanged.
    ///
    /// [`Glm52VllmNvfp4DpAttnDsaMoeModel::with_hisparse`]:
    /// crate::arch::glm52_vllm_nvfp4_dp_attn_dsa_moe::Glm52VllmNvfp4DpAttnDsaMoeModel::with_hisparse
    pub fn with_hisparse(
        mut self,
        config: HisparseConfig,
        gpus_per_node: u16,
    ) -> std::result::Result<Self, BuildError> {
        config.validate().map_err(fit_failed)?;
        // The latent leaves the GPU's per-token cache; the index cache, the
        // KDA state and the kpool tail stay as vLLM's layout declares them.
        self.kv_cache_layout
            .parts
            .retain(|part| part.name != tp_arch::MLA_LATENT_PART);
        let host_offload = config.kv_host_offload(
            u64::from(self.num_dsa_layers) * self.dsa_latent_bytes_per_token,
            0,
            gpus_per_node,
            self.cache_replicas(),
        );
        // In its place each request holds a fixed latent buffer on the GPU.
        self.kv_cache_layout.parts.push(
            KvCachePart::per_request(
                crate::arch::hisparse::DEVICE_BUFFER_PART,
                host_offload.device_bytes_per_request,
            )
            .not_prefix_reusable()
            // Every rank of an attention TP group attends the whole latent.
            .replicated(u32::from(self.cache_replicas())),
        );
        self.host_offload = Some(host_offload);
        self.hisparse = Some(config);
        self.recompile();
        Ok(self)
    }

    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let root = self.compile_root(&mut builder);
        builder.finish(root)
    }

    /// The whole forward as one node in `builder`, so a speculative model can
    /// compile it beside its draft stage. [`Self::eval_into`] fills it.
    fn kda_only(&self) -> bool {
        self.attn_tp > 1 && self.attn_tp_layers == AttnTpLayers::KdaOnly
    }

    /// Scheduler / KV partitions: one per rank under KDA-only, else one per
    /// attention group.
    fn num_groups(&self) -> u16 {
        if self.kda_only() {
            self.dp_size
        } else {
            self.dp_size / self.attn_tp
        }
    }

    /// GPUs holding a copy of a request's MLA cache.
    fn cache_replicas(&self) -> u16 {
        if self.kda_only() {
            1
        } else {
            self.attn_tp
        }
    }

    /// Ranks that split one partition's drafts and commit (attention TP over
    /// a shared partition); 1 when every rank owns its requests.
    pub(crate) fn partition_ranks(&self) -> u16 {
        self.dp_size / self.num_groups()
    }

    pub(crate) fn compile_root(&self, builder: &mut CostTreeBuilder) -> CostNode {
        // One representative rank per attention DP group: a group's ranks run
        // the same shapes.
        let dp = usize::from(self.num_groups());
        let prologue = (0..dp)
            .map(|_| {
                CostNode::Sum(vec![
                    self.embedding.compile(builder),
                    self.hc_expand.compile(builder),
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
        let kda_groups = usize::from(self.dp_size / self.attn_tp);
        for group in &self.groups {
            children.push(group.compile(dp, kda_groups, self.kda_only(), builder));
        }
        let epilogue = (0..dp)
            .map(|_| {
                CostNode::Sum(vec![
                    self.terminal_post.compile(builder),
                    self.hc_contract_mean.compile(builder),
                    self.final_norm.compile(builder),
                    self.lm_head.compile(builder),
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
        if let Some(hisparse) = &self.hisparse {
            children.push(hisparse.compile(&self.name, self.num_groups(), builder));
        }
        CostNode::Labeled {
            label: format!(
                "{} (Glm53FlashDpAttnEpModel) [attention DP{}/TP{}{}; MoE EP{} {}; \
                 timing_context<={}]",
                self.name,
                self.num_groups(),
                self.attn_tp,
                match self.attn_all_reduce_row_gpus {
                    _ if self.attn_tp == 1 => String::new(),
                    row_gpus => format!(
                        " ({}; sequence-parallel FFN/MoE; collective {:?}{})",
                        if self.kda_only() { "KDA layers only, DSA DP" } else { "KDA and DSA" },
                        self.attn_tp_collective,
                        match row_gpus {
                            Some(gpus) if gpus != self.attn_tp =>
                                format!(", EXTRAPOLATED from {gpus}-GPU rows"),
                            _ => String::new(),
                        }
                    ),
                },
                self.dp_size,
                match self.ep_comm_kind {
                    EpCommKind::AllgatherReducescatter => "all_gatherv/reduce_scatterv",
                    EpCommKind::Alltoall => "MNNVL all-to-all",
                    EpCommKind::FastAllgather => "fast direct-write all-gather + combine",
                    EpCommKind::FusedAllgather => "fused direct-write all-gather + combine",
                    EpCommKind::FastSparse => "fast sparse direct writes",
                },
                self.max_model_len
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub(crate) fn eval_into(&self, input: &UnifiedArchInput, ev: &mut Evaluator) {
        let batch = normalize_input(
            input,
            self.dp_size,
            self.attn_tp,
            self.kda_only(),
            self.max_model_len,
            self.verify_width,
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
        for step in self.hisparse_steps(input) {
            push_rank_step(step, ev);
        }
    }

    /// Each rank's HiSparse miss copy; empty without HiSparse. Only the DSA
    /// layers hold a latent cache, so only they miss.
    fn hisparse_steps(&self, input: &UnifiedArchInput) -> Vec<HisparseRankStep> {
        let Some(hisparse) = &self.hisparse else {
            return Vec::new();
        };
        input
            .groups
            .iter()
            .map(|group| {
                hisparse.rank_step(
                    &group.decode_kv_lens,
                    self.verify_width,
                    self.dsa_latent_bytes_per_token,
                    self.num_dsa_layers,
                )
            })
            .collect()
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
            self.attn_tp,
            self.kda_only(),
            self.max_model_len,
            self.verify_width,
            &self.cudagraph_capture_sizes,
        )
        .map(|_| ())
    }

    /// One DP rank's cache for a request (the rank that owns it): the DSA
    /// layers' KV and all 64 KDA heads of 34 layers. Under HiSparse the DSA
    /// layers' MLA latent is not in it: the GPU keeps only their index cache.
    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout.clone()
    }

    fn kv_host_offload(&self) -> Option<KvHostOffload> {
        self.host_offload
    }

    /// The kpool DSA's top-k cap makes necessary work per-request in context.
    fn logs_decode_kv_lens(&self) -> bool {
        true
    }

    fn gpus_per_replica(&self) -> u16 {
        self.dp_size
    }

    fn num_attn_dp_groups(&self) -> u16 {
        self.num_groups()
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
    /// Rows every kernel outside attention runs on, on one rank: the
    /// scheduled tokens, the one-token dummy batch of an idle rank, or the
    /// common graph size; under attention TP the rank's sequence-parallel
    /// slice of them.
    rows: u32,
    /// Rows the group's attention runs on (each rank, heads split): `rows`
    /// before the sequence-parallel slice.
    attn_rows: u32,
    /// lm_head rows: one per scheduled request (none for a dummy batch).
    logits_rows: u32,
    kda: Glm53KdaAttnLocalWorkletInput,
    dsa: Glm53DsaAttnLocalWorkletInput,
}

/// One KDA tensor-parallel group's view under KDA-only attention TP.
struct KdaGroupBatch {
    /// The group's rows, all-gathered for KDA.
    rows: u32,
    kda: Glm53KdaAttnLocalWorkletInput,
}

struct NormalizedBatch {
    /// One per partition: a representative rank per attention DP group, or
    /// every owner rank under KDA-only.
    ranks: Vec<RankBatch>,
    /// KDA-only: the KDA groups, `attn_tp` ranks each.
    kda_groups: Vec<KdaGroupBatch>,
    kda_only: bool,
    /// Tokens each of the `dp_size` EP source ranks dispatches.
    source_rank_tokens: Vec<u32>,
    /// Rows every EP rank's fused MoE reads after the all-gather.
    gathered_tokens: u32,
}

fn normalize_input(
    input: &UnifiedArchInput,
    dp_size: u16,
    attn_tp: u16,
    kda_only: bool,
    max_model_len: u32,
    verify_width: u32,
    sorted_capture_sizes: &[u32],
) -> std::result::Result<NormalizedBatch, String> {
    let dp = usize::from(if kda_only { dp_size } else { dp_size / attn_tp });
    if input.groups.len() != dp {
        return Err(format!(
            "expected {dp} attention-DP groups, got {}",
            input.groups.len()
        ));
    }
    // A single attention group's worker sends no per-source-rank counts.
    let source_counts = if dp == 1 && input.tokens_per_source_rank.is_empty() {
        vec![input.groups[0].batch_tokens]
    } else {
        input.tokens_per_source_rank.clone()
    };
    if source_counts.len() != dp {
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
            if !(verify_width..=max_model_len).contains(&context) {
                return Err(format!(
                    "rank {rank} decode request {request} context {context} must be in \
                     {verify_width}..={max_model_len}"
                ));
            }
        }
        let decode_tokens = group.decode_kv_lens.len() as u32 * verify_width;
        if group.decode_tokens != decode_tokens {
            return Err(format!(
                "rank {rank} decode_tokens {} must equal decode_kv_lens length x verify width \
                 {verify_width}",
                group.decode_tokens
            ));
        }
        let total = prefill_tokens + decode_tokens;
        if group.batch_tokens != total || source_counts[rank] != total {
            return Err(format!(
                "rank {rank} batch_tokens {} and source-rank tokens {} must equal \
                 prefill+decode {total}",
                group.batch_tokens, source_counts[rank]
            ));
        }
        let (kda, dsa) = if total == 0 {
            // `execute_dummy_batch`: one decode request on dummy slots.
            (
                Glm53KdaAttnLocalWorkletInput {
                    prefill_sequence_lengths: Vec::new(),
                    decode_batch_size: 1,
                },
                Glm53DsaAttnLocalWorkletInput {
                    prefill_chunk_pairs: Vec::new(),
                    decode_kv_lens: vec![verify_width],
                },
            )
        } else {
            attention_inputs(&group.prefill_chunk_pairs, &group.decode_kv_lens)
        };
        ranks.push(RankBatch {
            rows: total.max(verify_width),
            attn_rows: 0,
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
    if kda_only {
        // Every rank owns its rows; each group's KDA runs on its ranks' rows
        // and requests together.
        let tp = usize::from(attn_tp);
        let kda_groups = ranks
            .chunks(tp)
            .map(|members| KdaGroupBatch {
                rows: members.iter().map(|rank| rank.rows).sum(),
                kda: Glm53KdaAttnLocalWorkletInput {
                    prefill_sequence_lengths: members
                        .iter()
                        .flat_map(|rank| rank.kda.prefill_sequence_lengths.iter().copied())
                        .collect(),
                    decode_batch_size: members.iter().map(|rank| rank.kda.decode_batch_size).sum(),
                },
            })
            .collect();
        for rank in &mut ranks {
            rank.attn_rows = rank.rows;
        }
        let source_rank_tokens: Vec<u32> = ranks.iter().map(|rank| rank.rows).collect();
        let gathered_tokens = source_rank_tokens
            .iter()
            .try_fold(0_u32, |sum, &rows| sum.checked_add(rows))
            .ok_or("gathered token sum overflows u32")?;
        return Ok(NormalizedBatch {
            ranks,
            kda_groups,
            kda_only: true,
            source_rank_tokens,
            gathered_tokens,
        });
    }
    // Sequence-parallel slice: each of a group's `attn_tp` ranks owns
    // ceil(rows / attn_tp) rows outside attention (the busiest rank's share)
    // and one lm_head row per owned request.
    let tp = u32::from(attn_tp);
    for rank in &mut ranks {
        rank.attn_rows = rank.rows;
        rank.rows = rank.rows.div_ceil(tp);
        rank.logits_rows = rank.logits_rows.div_ceil(tp);
    }
    let source_rank_tokens: Vec<u32> = ranks
        .iter()
        .flat_map(|rank| std::iter::repeat_n(rank.rows, usize::from(attn_tp)))
        .collect();
    let gathered_tokens = source_rank_tokens
        .iter()
        .try_fold(0_u32, |sum, &rows| sum.checked_add(rows))
        .ok_or("gathered token sum overflows u32")?;
    Ok(NormalizedBatch {
        ranks,
        kda_groups: Vec::new(),
        kda_only: false,
        source_rank_tokens,
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
    use crate::arch::ep_comm::EpCommModel;
    use crate::timing::kernels::MoeAlltoallDirection;
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
            kernel_path: Default::default(),
            ep_comm: EpCommConfig::default(),
            mhc_fuse_fixed_us: None,
            attn_tp: 1,
            attn_all_reduce_row_gpus: None,
            attn_tp_layers: Default::default(),
            attn_tp_collective: Default::default(),
            kda_gate_b_merged: false,
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
        assert_eq!(built(4).kv_cache_layout().engine.block_tokens, Some(8576));
    }

    #[test]
    fn kv_and_state_are_one_ranks_full_heads_not_replicated() {
        for dp in [4, 8] {
            let model = built(dp);
            let layout = model.kv_cache_layout();
            assert_eq!(layout.per_token_bytes(), 11 * (512 + 33));
            // The true state is (4 << 20) + 24576 * 3 * 2 bytes per layer.
            let kda = layout.parts.iter().find(|part| part.name == "kda").unwrap();
            assert_eq!(kda.bytes_per_entry, 34 * ((4 << 20) + 24576 * 3 * 2));
            assert_eq!(model.gpus_per_replica(), dp);
            assert_eq!(model.num_attn_dp_groups(), dp);
            assert_eq!(model.num_attn_shards(), 1);
        }
    }

    #[test]
    fn compiled_tree_has_a_fixed_slot_count() {
        for dp in [4_usize, 8] {
            let model = built(dp as u16);
            // Per rank: boundaries 2 + attention (KDA 12 / DSA 31) + FFN-local
            // (dense 5; MoE router 2 + glue 1 + serial shared 5 + quant 1).
            // MoE group-wide: dispatch + dp fused + combine + dp x 5 concurrent
            // shared + dp combine glue. Prologue 2/rank, epilogue 4/rank.
            let moe_global = 1 + dp + 1 + 5 * dp + dp;
            let kda_dense = dp * (2 + 12 + 5);
            let dsa_moe = dp * (2 + 31 + 9) + moe_global;
            let kda_moe = dp * (2 + 12 + 9) + moe_global;
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
                .filter(|slot| {
                    !matches!(
                        slot.kind.as_str(),
                        "all_gather"
                            | "reduce_scatter"
                            | "moe_ep_all_gather"
                            | "moe_ep_quantized_all_gather"
                            | "moe_ep_reduce_scatter"
                    )
                })
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
        // The TP map's 125 less 8 per-rank input quants, plus per MoE kind
        // the dispatch quant and ranks 4..dp (the gather and scatter are
        // communication, which no map lists).
        assert_maps_cover(&model_cfg(), ARCH_KIND, |dp| {
            125 - 8 + 2 * (1 + usize::from(dp) - 4)
        });
    }

    #[test]
    fn nvfp4_maps_add_the_router_select_and_drop_the_shared_expert_quant() {
        // Per MoE kind: +1 vLLM top-k select, -2 BF16 shared-expert quants.
        assert_maps_cover(
            &config("glm53_flash_nvfp4"),
            "glm53_flash_vllm_nvfp4_dp_attn_ep_moe",
            |dp| 125 - 8 + 2 * (1 + usize::from(dp) - 4) - 2,
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

    fn decode(kv_lens: &[u32]) -> ArchGroupInput {
        ArchGroupInput {
            batch_tokens: kv_lens.len() as u32,
            prefill_tokens: 0,
            decode_tokens: kv_lens.len() as u32,
            prefill_chunk_pairs: Vec::new(),
            decode_kv_lens: kv_lens.to_vec(),
            total_kv_len: kv_lens.iter().sum(),
        }
    }

    fn hisparse(buffer_tokens: u32) -> HisparseConfig {
        HisparseConfig {
            buffer_tokens,
            host_dram_gb_per_node: 1500.0,
            ..HisparseConfig::default()
        }
    }

    /// Catches HiSparse offloading the wrong part of a Flash token: only the
    /// 11 DSA layers' 512-byte latent leaves the GPU, their 33-byte kpool
    /// index stays, and the latent buffer is charged per request.
    #[test]
    fn hisparse_moves_only_the_dsa_latent_to_the_host() {
        let plain = built(8);
        assert!(plain.kv_host_offload().is_none());
        assert_eq!(plain.kv_cache_layout().per_token_bytes(), 11 * (512 + 33));
        let model = built(8).with_hisparse(hisparse(49_152), 8).unwrap();
        assert_eq!(model.kv_cache_layout().per_token_bytes(), 11 * 33);
        assert_eq!(
            model.kv_host_offload(),
            Some(KvHostOffload {
                device_bytes_per_request: 49_152 * 11 * 512,
                host_bytes_per_token: 11 * 512,
                host_bytes_per_partition: 187_500_000_000,
            })
        );
        // GPU + host bytes of one token are the plain per-token bytes.
        assert_eq!(
            model.kv_cache_layout().per_token_bytes() + 11 * 512,
            plain.kv_cache_layout().per_token_bytes()
        );
        // An explicit node size splits the host DRAM over that many GPUs.
        let shared = built(4)
            .with_hisparse(
                HisparseConfig {
                    gpus_per_node: Some(8),
                    ..hisparse(4096)
                },
                4,
            )
            .unwrap();
        assert_eq!(
            shared.kv_host_offload().unwrap().host_bytes_per_partition,
            187_500_000_000
        );
        assert!(built(8).with_hisparse(hisparse(0), 8).is_err());
    }

    /// Catches HiSparse touching the KDA layers: their recurrent state and
    /// conv window are not offloadable and keep their size and checkpoint.
    #[test]
    fn hisparse_leaves_the_kda_state_on_the_gpu_unchanged() {
        let plain = built(8);
        let model = built(8).with_hisparse(hisparse(49_152), 8).unwrap();
        let (layout, plain_layout) = (model.kv_cache_layout(), plain.kv_cache_layout());
        let kda = |layout: &KvCacheLayout| {
            *layout
                .parts
                .iter()
                .find(|part| part.name == "kda")
                .expect("kda part")
        };
        assert_eq!(kda(&layout), kda(&plain_layout));
        assert_eq!(layout.engine, plain_layout.engine);
        assert_eq!(layout.engine.block_tokens, Some(8576));
        // The latent part left the GPU layout for its per-request buffer.
        assert_eq!(layout.parts.len(), plain_layout.parts.len());
        assert!(layout.parts.iter().all(|part| part.name != tp_arch::MLA_LATENT_PART));
        let buffer = layout
            .parts
            .iter()
            .find(|part| part.name == crate::arch::hisparse::DEVICE_BUFFER_PART)
            .expect("device buffer part");
        assert_eq!(buffer.one_copy_bytes_per_entry(), 49_152 * 11 * 512);
        assert!(!buffer.prefix_reusable);
    }

    /// Catches the miss copy billing all 45 layers, or a KDA layer: bytes are
    /// the misses times one 512-byte latent times the 11 DSA layers, and one
    /// copy leaf per rank is appended after the unchanged forward.
    #[test]
    fn hisparse_miss_io_scales_with_the_dsa_layers_only() {
        let plain = built(8);
        let model = built(8).with_hisparse(hisparse(4096), 8).unwrap();
        assert_eq!(model.n_slots, plain.n_slots + 8);
        let copies: Vec<_> = model
            .cost_log_manifest()
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.kind == crate::arch::hisparse::HISPARSE_H2D_IO_KIND)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(copies, (plain.n_slots..model.n_slots).collect::<Vec<_>>());
        let mut groups = vec![ArchGroupInput::default(); 8];
        groups[0] = decode(&[100_000, 2_000]);
        groups[5] = decode(&[200_000]);
        let steps = model.hisparse_steps(&input(groups));
        assert_eq!(steps.len(), 8);
        // r = 4096 / 2048 = 2: 13.4% of the top-k misses per layer.
        let misses = 0.134 * 2048.0;
        assert_eq!(steps[0].offloaded_requests, 1);
        assert!((steps[0].misses_per_layer - misses).abs() < 1e-6);
        assert!((steps[0].io_bytes - misses * 512.0 * 11.0).abs() < 1e-3);
        assert!((steps[5].io_bytes - steps[0].io_bytes).abs() < 1e-6);
        assert_eq!(steps[1], HisparseRankStep::default());
        assert!(plain
            .hisparse_steps(&input(vec![decode(&[100_000]); 8]))
            .is_empty());
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
            1,
            false,
            131072,
            1,
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
            1,
            false,
            131072,
            1,
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
            1,
            false,
            131072,
            1,
            &sizes,
        )
        .unwrap();
        let rows: Vec<_> = eager.ranks.iter().map(|rank| rank.rows).collect();
        assert_eq!(rows, [3, 40, 1, 1]);
    }

    #[test]
    fn attention_tp_all_slices_group_rows_over_its_ranks() {
        // DP2 x TP2 over 4 ranks: one input group per TP group; every rank
        // runs the group's attention rows and a ceil(rows/2) FFN slice.
        let batch =
            normalize_input(&input(vec![prefill(0, 101), prefill(0, 8)]), 4, 2, false, 131072, 1, &[])
                .unwrap();
        assert!(!batch.kda_only);
        let attn: Vec<_> = batch.ranks.iter().map(|rank| rank.attn_rows).collect();
        let rows: Vec<_> = batch.ranks.iter().map(|rank| rank.rows).collect();
        assert_eq!(attn, [101, 8]);
        assert_eq!(rows, [51, 4]);
        assert!(batch.kda_groups.is_empty());
    }

    #[test]
    fn attention_tp_kda_only_keeps_dp_ranks_and_merges_kda_groups() {
        // KDA-only TP2 over 4 ranks: four DP partitions (DSA stays on the owner
        // rank) and two KDA groups of two ranks each.
        let batch = normalize_input(
            &input(vec![prefill(0, 10), prefill(0, 20), prefill(0, 30), prefill(0, 40)]),
            4,
            2,
            true,
            131072,
            1,
            &[],
        )
        .unwrap();
        assert!(batch.kda_only);
        let rows: Vec<_> = batch.ranks.iter().map(|rank| rank.rows).collect();
        assert_eq!(rows, [10, 20, 30, 40]);
        let kda: Vec<_> = batch.kda_groups.iter().map(|group| group.rows).collect();
        assert_eq!(kda, [30, 70]);
        assert_eq!(batch.gathered_tokens, 100);
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
        assert!(normalize_input(&bad, 4, 1, false, 131072, 1, &[]).is_err());
        let empty = input(vec![ArchGroupInput::default(); 4]);
        assert!(normalize_input(&empty, 4, 1, false, 131072, 1, &[]).is_err());
        let three = input(vec![prefill(0, 8); 3]);
        assert!(normalize_input(&three, 4, 1, false, 131072, 1, &[]).is_err());
        let long = input(vec![
            prefill(131000, 100),
            prefill(0, 8),
            prefill(0, 8),
            prefill(0, 8),
        ]);
        assert!(normalize_input(&long, 4, 1, false, 131072, 1, &[]).is_err());
    }

    fn exchange(dp_size: u16, kind: EpCommKind, model: EpCommModel) -> Glm53FlashDpAttnEpParallel {
        Glm53FlashDpAttnEpParallel {
            ep_comm: EpCommConfig {
                kind,
                model,
                ..EpCommConfig::default()
            },
            ..parallel(dp_size)
        }
    }

    fn built_with(
        model: &Glm53FlashModelCfg,
        p: &Glm53FlashDpAttnEpParallel,
    ) -> Glm53FlashDpAttnEpModel {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let demand = ExpertDemand::popularity(&RoutingDistribution::uniform(288), 42);
        let configs = build_configs(model, p, &demand).unwrap();
        build("unified".into(), resolve_configs(&configs), &bridge).unwrap()
    }

    /// `(leaf, kind)` of every slot, the leaf being the name's last segment.
    fn leaves(model: &Glm53FlashDpAttnEpModel) -> Vec<(String, String)> {
        model
            .cost_log_manifest()
            .slots
            .into_iter()
            .map(|slot| (slot.name.rsplit('.').next().unwrap().to_string(), slot.kind))
            .collect()
    }

    fn named<'a>(leaves: &'a [(String, String)], kind: &str) -> Vec<&'a str> {
        leaves
            .iter()
            .filter(|(_, k)| k == kind)
            .map(|(leaf, _)| leaf.as_str())
            .collect()
    }

    /// Catches a default `ep_comm` changing the cost tree: the measured
    /// all-gatherv / reduce-scatterv stay, once per MoE kind, and nothing of
    /// the all-to-all appears.
    #[test]
    fn the_default_exchange_is_the_measured_gather_and_scatter() {
        let model = config("glm53_flash_nvfp4");
        let demand = ExpertDemand::popularity(&RoutingDistribution::uniform(288), 42);
        let configs = build_configs(&model, &parallel(8), &demand).unwrap();
        assert!(configs.exchange.is_measured_allgather());
        let explicit = built_with(
            &model,
            &exchange(8, EpCommKind::AllgatherReducescatter, EpCommModel::Measured),
        );
        let default = built_for(&model, 8);
        assert_eq!(default.n_slots, explicit.n_slots);
        assert_eq!(
            serde_json::to_value(default.cost_log_manifest()).unwrap(),
            serde_json::to_value(explicit.cost_log_manifest()).unwrap()
        );
        let slots = leaves(&default);
        assert_eq!(
            named(&slots, "moe_ep_quantized_all_gather"),
            ["dispatch_all_gather"; 2]
        );
        assert_eq!(
            named(&slots, "moe_ep_reduce_scatter"),
            ["combine_reduce_scatter"; 2]
        );
        assert!(slots.iter().all(|(leaf, kind)| {
            !kind.starts_with("moe_alltoall")
                && kind != crate::arch::ep_comm::EP_COMM_ANALYTIC_KIND
                && !leaf.contains("alltoall")
        }));
    }

    /// Catches the all-to-all sized from GLM-5.3's widths instead of Flash's:
    /// 4096 hidden gives 2,048-byte NVFP4 rows, 256-byte block scales and
    /// 8,192-byte bf16 combine rows; 288 experts are 288 MNNVL slots.
    #[test]
    fn the_alltoall_is_sized_from_the_flash_geometry() {
        let model = config("glm53_flash_nvfp4");
        let demand = ExpertDemand::popularity(&RoutingDistribution::uniform(288), 42);
        let configs = build_configs(
            &model,
            &exchange(8, EpCommKind::Alltoall, EpCommModel::Measured),
            &demand,
        )
        .unwrap();
        let kernels = configs.exchange.measured_alltoall.as_ref().unwrap();
        for (kernel, bytes, direction) in [
            (&kernels.dispatch, 2_048, MoeAlltoallDirection::Dispatch),
            (
                &kernels.dispatch_scales,
                256,
                MoeAlltoallDirection::Dispatch,
            ),
            (&kernels.combine, 8_192, MoeAlltoallDirection::Combine),
        ] {
            assert_eq!(kernel.hidden_bytes.get(), bytes);
            assert_eq!(kernel.direction, direction);
            assert_eq!(
                (kernel.ep_size, kernel.top_k, kernel.slot_count),
                (8, 8, 288)
            );
            assert_eq!(kernel.backends, ["flashinfer_mnnvl"]);
        }
        assert_eq!(
            (
                kernels.prepare.ep_size,
                kernels.prepare.top_k,
                kernels.prepare.slot_count
            ),
            (8, 8, 288)
        );
        let elementwise = configs.exchange.alltoall.as_ref().unwrap();
        for (config, input, output) in [
            (&elementwise.scale_swizzle, 256, 256),
            (&elementwise.combine_fill, 0, 8 * 8_192),
            (&elementwise.combine_reduce, 8 * 8_192, 8_192),
        ] {
            assert_eq!(
                (
                    config.input_bytes_per_token.get(),
                    config.output_bytes_per_token.get()
                ),
                (input, output)
            );
        }
        // The analytic legs price the same widths: one exchange of 16 tokens
        // per rank moves (rows x width) bytes on the busiest rank.
        let comm = configs.exchange.analytic.as_ref().unwrap();
        assert_eq!((comm.hidden, comm.top_k), (4096, 8));
        let step = comm.step(&[16; 8]);
        let rows = step.dispatch.max_send_rows.max(step.dispatch.max_recv_rows);
        assert_eq!(step.dispatch.bytes, rows * 2_048.0);
        assert_eq!(step.dispatch_scales.bytes, rows * 256.0);
        assert_eq!(step.combine.bytes, rows * 8_192.0);
        // 288 experts, 36 per rank: 1 - C(252, 8) / C(288, 8) of tokens reach a rank.
        let mut miss = 1.0;
        for i in 0..8 {
            miss *= f64::from(252 - i) / f64::from(288 - i);
        }
        assert!((comm.uniform_hit - (1.0 - miss)).abs() < 1e-12);
        // The analytic all-gather row: NVFP4 activations, scales, ids, weights.
        assert_eq!(comm.gather_row_bytes(), f64::from(2_048 + 256 + 8 * 8));
    }

    /// Catches the all-to-all leaving the gather / scatter in place or
    /// dropping a leaf: per MoE kind it adds the prepare, a second dispatch
    /// call and three elementwise leaves per rank, under the GLM-5.3 DP
    /// graph's leaf names, measured or analytic. The analytic all-gather keeps
    /// the measured slot names and count.
    #[test]
    fn the_alltoall_replaces_the_gather_and_scatter_leaves() {
        let model = config("glm53_flash_nvfp4");
        let base = built_for(&model, 8).n_slots;
        let measured = built_with(
            &model,
            &exchange(8, EpCommKind::Alltoall, EpCommModel::Measured),
        );
        assert_eq!(measured.n_slots, base + 2 * (2 + 3 * 8));
        let slots = leaves(&measured);
        assert_eq!(
            named(&slots, "moe_alltoall_prepare"),
            ["dispatch_alltoall_prepare"; 2]
        );
        assert_eq!(
            named(&slots, "moe_alltoall"),
            [
                "dispatch_alltoall",
                "dispatch_alltoall_scales",
                "combine_alltoall"
            ]
            .repeat(2)
        );
        assert!(slots.iter().all(|(_, kind)| !kind.starts_with("moe_ep_")));
        for leaf in [
            "dispatch_scale_swizzle",
            "combine_output_fill",
            "combine_reduce",
        ] {
            assert_eq!(
                slots
                    .iter()
                    .filter(|(l, kind)| l == leaf && kind == "elementwise")
                    .count(),
                2 * 8,
                "{leaf}"
            );
        }
        let mut order = Vec::new();
        leaf_order(&measured.cost_tree().root, &mut order);
        assert_eq!(order, (0..order.len()).collect::<Vec<_>>());

        let analytic = built_with(
            &model,
            &exchange(8, EpCommKind::Alltoall, EpCommModel::Analytic),
        );
        assert_eq!(analytic.n_slots, measured.n_slots);
        assert_eq!(
            named(
                &leaves(&analytic),
                crate::arch::ep_comm::EP_COMM_ANALYTIC_KIND
            ),
            [
                "dispatch_alltoall_prepare",
                "dispatch_alltoall",
                "dispatch_alltoall_scales",
                "combine_alltoall"
            ]
            .repeat(2)
        );

        let gather = built_with(
            &model,
            &exchange(8, EpCommKind::AllgatherReducescatter, EpCommModel::Analytic),
        );
        assert_eq!(gather.n_slots, base);
        assert_eq!(
            named(
                &leaves(&gather),
                crate::arch::ep_comm::EP_COMM_ANALYTIC_KIND
            ),
            ["dispatch_all_gather", "combine_reduce_scatter"].repeat(2)
        );
    }

    /// Catches a fast exchange keeping the gather / scatter kernels or the
    /// all-to-all's prepare: one dispatch and one combine leaf per MoE kind,
    /// analytic, or the combine from the measured nvshmem all-reduce.
    #[test]
    fn the_fast_exchanges_replace_the_collectives_with_two_leaves() {
        use crate::arch::ep_comm::{EpFastConfig, FastBackend, FastCombine};
        let model = config("glm53_flash_nvfp4");
        let base = built_for(&model, 8).n_slots;
        let fast_parallel = |kind, comm_model, fast| Glm53FlashDpAttnEpParallel {
            ep_comm: EpCommConfig {
                kind,
                model: comm_model,
                fast,
                ..EpCommConfig::default()
            },
            ..parallel(8)
        };
        for (kind, dispatch, combine) in [
            (
                EpCommKind::FastAllgather,
                "dispatch_fast_all_gather",
                "combine_fast_reduce_scatter",
            ),
            (
                EpCommKind::FusedAllgather,
                "dispatch_fused_all_gather",
                "combine_fused_reduce",
            ),
            (
                EpCommKind::FastSparse,
                "dispatch_sparse_write",
                "combine_sparse_write",
            ),
        ] {
            let built = built_with(
                &model,
                &fast_parallel(kind, EpCommModel::Analytic, EpFastConfig::default()),
            );
            // The reduce-scatter combines add the receiver's sum on each of
            // the 8 ranks, per MoE kind; the fused combine folds it away.
            let sums = if kind == EpCommKind::FusedAllgather { 0 } else { 2 * 8 };
            assert_eq!(built.n_slots, base + sums, "{kind:?}");
            let slots = leaves(&built);
            assert_eq!(
                slots.iter().filter(|(l, k)| l == "combine_sum" && k == "elementwise").count(),
                sums
            );
            assert_eq!(
                named(&slots, crate::arch::ep_comm::EP_COMM_ANALYTIC_KIND),
                [dispatch, combine].repeat(2)
            );
            assert!(slots.iter().all(|(_, k)| !k.starts_with("moe_ep_")
                && !k.starts_with("moe_alltoall")));
        }
        let measured = built_with(
            &model,
            &fast_parallel(
                EpCommKind::FastAllgather,
                EpCommModel::Measured,
                EpFastConfig {
                    backend: FastBackend::Nvshmem,
                    combine: FastCombine::AllReduce,
                    ..EpFastConfig::default()
                },
            ),
        );
        let slots = leaves(&measured);
        assert_eq!(named(&slots, "all_reduce"), ["combine_fast_all_reduce"; 2]);
        assert_eq!(
            named(&slots, crate::arch::ep_comm::EP_COMM_ANALYTIC_KIND),
            ["dispatch_fast_all_gather"; 2]
        );
    }

    /// Catches an exchange the rows cannot price: the measured all-to-all is
    /// EP8 only, and FP8 dispatches FP8 activations no `ep_comm` exchange
    /// models.
    #[test]
    fn the_exchange_rejects_what_it_cannot_price() {
        let demand = ExpertDemand::popularity(&RoutingDistribution::uniform(288), 42);
        let nvfp4 = config("glm53_flash_nvfp4");
        let error = build_configs(
            &nvfp4,
            &exchange(4, EpCommKind::Alltoall, EpCommModel::Measured),
            &demand,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("ep_size 4"), "{error}");
        assert!(build_configs(
            &nvfp4,
            &exchange(4, EpCommKind::Alltoall, EpCommModel::Analytic),
            &demand,
        )
        .is_ok());
        for (kind, comm_model) in [
            (EpCommKind::Alltoall, EpCommModel::Measured),
            (EpCommKind::AllgatherReducescatter, EpCommModel::Analytic),
        ] {
            assert!(build_configs(&model_cfg(), &exchange(8, kind, comm_model), &demand).is_err());
        }
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
