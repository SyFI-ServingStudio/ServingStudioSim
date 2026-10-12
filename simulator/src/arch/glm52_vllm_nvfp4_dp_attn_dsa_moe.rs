//! GLM-5.2/5.3 NVIDIA NVFP4 under data-parallel attention and expert-parallel
//! MoE, aligned to vLLM on B200 (`--data-parallel-size N
//! --enable-expert-parallel`, tensor parallel 1).
//!
//! GLM-5.3 NVFP4 is the same graph, so one arch covers both checkpoints.
//!
//! Every GPU is its own vLLM engine: it schedules its own batch and owns the
//! KV of the requests it admitted, so the iteration has `ep_size` attention
//! groups and a token's cache lives on exactly one GPU. Attention, the dense
//! FFN, the shared expert, the embedding and the full-vocabulary lm_head run at
//! TP1 -- all 64 heads, no all-reduce -- on each GPU's own tokens. Their
//! leaves come from the TP/EP graph's EP1 recipe (`glm52_vllm_nvfp4_dsa_moe.rs`)
//! and keep its names, so the GLM-5.2 kernel labels apply.
//!
//! Only the routed experts are sharded. Under DP plus EP vLLM selects the
//! modular TRTLLM NVFP4 experts (the monolithic kernel declines an all-to-all
//! parallel config) behind the naive DP/EP prepare-finalize, so a sparse layer
//! runs, per GPU:
//!
//! 1. the post-attention residual RMSNorm, the router GEMM and the bias top-k
//!    select on the GPU's own tokens (the router-select leaf exists only here);
//! 2. the shared expert, inline above 256 local tokens and on an auxiliary
//!    stream at or below it, where it overlaps everything up to the combine;
//! 3. NVFP4 quantization of the local tokens with the static scale;
//! 4. one grouped all-gatherv of the packed activations, block scales, top-k
//!    weights and top-k IDs (`moe_ep_quantized_all_gather`);
//! 5. the precomputed-routing fused MoE over every gathered token, on this
//!    GPU's 256 / ep_size experts (`flashinfer_trtllm_routed_sm100`);
//! 6. the bf16 reduce-scatterv back to each GPU's own tokens
//!    (`moe_ep_reduce_scatter`).
//!
//! The ranks synchronize only at the gather and the scatter, so the work
//! between two collectives is billed as the slowest rank's whole segment: a
//! sparse layer's attention, router, serial shared expert and quantization are
//! one `Max` over ranks of their sum. Dense layers, the embedding and the
//! output head are `Max` over ranks per section, which is exact when the
//! ranks carry similar batches and an upper bound otherwise.
//!
//! A rank with nothing scheduled runs vLLM's one-token dummy batch to join the
//! collectives. The ranks then agree on a step size
//! (`vllm/v1/worker/dp_utils.py` `_synchronize_dp_ranks`): when the busiest
//! rank's count fits a captured CUDA graph, every rank, the dummy ones
//! included, pads to that graph's size, so the gather carries `ep_size` times
//! the padded count; when it does not, the step runs eager and the
//! collectives see the real per-rank counts, with one token from each idle
//! rank. Under padding every kernel outside the attention graph break runs on
//! the padded rows; attention and the lm_head (on the sampled rows only) keep
//! the real counts. With no capture sizes every step is eager, and an idle
//! rank's compute is billed as zero (it hides under the busy ranks' `Max`).
//!
//! The same graph is the target of a DFlash2 verify
//! (`glm53_vllm_nvfp4_dp_attn_ep_moe_dflash2`, built through
//! [`build_verify_configs`]): each decode request then presents
//! `draft_tokens + 1` rows at its final KV length, the DSA indexer scores them
//! as one `next_n` group, the lm_head scores every one, and the routed experts
//! see the request's rows as one verify block. Padding and the dummy batch are
//! unchanged: they count tokens, and an idle rank's `execute_dummy_batch`
//! still schedules one.
//!
//! [`Glm52VllmNvfp4DpAttnDsaMoeModel::with_hisparse`] serves the cache through
//! SGLang HiSparse (`arch::hisparse`, a rough analytic model): the per-token
//! GPU cache shrinks to the indexer keys, a fixed latent buffer per request and
//! a host tier join the capacity contract (`kv_host_offload`), and each step
//! adds the exposed host -> device miss copy as one more `Max` over ranks.
//!
//! The exchange can instead run as flashinfer's two-sided MNNVL all-to-all
//! (`ep_comm: alltoall`), which adds its metadata prepare, the receiver's NVFP4
//! scale swizzle, and the combine's zeroed `tokens x top_k` staging buffer and
//! top-k sum (measured elementwise leaves). At EP8 its prepare and transfers
//! can be the profiled B200 `moe_alltoall_prepare` / `moe_alltoall` rows
//! (`ep_comm_model: measured`): the prepare at the busiest rank's tokens, the
//! NVFP4 activations and their block scales as two lookups, the bf16 combine as
//! one, at `arch::ep_comm`'s busiest-sender / busiest-receiver rows.
//! `arch::ep_comm`'s analytic link model (`ep_comm_model: analytic`) prices
//! either exchange at any size and is the only way past the measured 8 GPUs: an
//! EP group of up to the whole NVLink domain (GB200 NVL72), `ep_size` dividing
//! the 256 experts. The default, measured all-gather / reduce-scatter, is
//! unchanged.

use std::sync::Arc;

use crate::arch::contract::{IterwiseUnifiedModel, KvHostOffload, UnifiedArchInput};
use crate::arch::ep_comm::{
    push_leg, AlltoallRows, AnalyticEpComm, EpCommConfig, EpCommKind, EpCommModel,
};
use crate::arch::glm52_model_cfg::{Glm52ModelCfg, Glm52MtpMode};
use crate::arch::glm52_vllm_nvfp4_dsa_moe::{
    self as ep_graph, build_atomic, eval_atomic_or_zero, normalize_group, state_bytes_per_token,
    Glm52VllmNvfp4DsaMoeConfigs, Glm52VllmNvfp4DsaMoeParallel, Glm52VllmNvfp4DsaMoeResolved,
    NormalizedGroup, CHECKPOINT_MAX_CONTEXT, NUM_DENSE_LAYERS, NUM_INITIAL_SHARED_LAYERS,
};
use crate::arch::glm52_vllm_nvfp4_pp_dsa_moe::eval_compiled;
use crate::arch::glm53_flash_vllm_fp8_kda_dsa_moe::graph_padded_tokens;
use crate::arch::hisparse::{push_rank_step, HisparseConfig, HisparseRankStep};
use crate::common::Fabric;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, MoeEpCollectiveKernelInput,
    MoeAlltoallDirection, MoeAlltoallKernel, MoeAlltoallKernelConfig, MoeAlltoallKernelInput,
    MoeAlltoallPrepareKernel, MoeAlltoallPrepareKernelConfig, MoeAlltoallPrepareKernelInput,
    MoeEpQuantizedAllGatherKernel, MoeEpQuantizedAllGatherKernelConfig, MoeEpReduceScatterKernel,
    MoeEpReduceScatterKernelConfig, Nvfp4FusedMoeKernel, Nvfp4FusedMoeKernelInput,
    Nvfp4QuantKernel, Nvfp4QuantKernelInput, ResidualRmsNormKernel, ResidualRmsNormKernelInput,
    SingleGemmKernel, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::{
    Glm52DenseFfnLocalWorklet, Glm52DenseFfnLocalWorkletInput, Glm52MoeRouterLocalWorklet,
    Glm52MoeRouterLocalWorkletInput, Glm52SharedExpertLocalWorklet,
    Glm52SharedExpertLocalWorkletInput, Nvfp4MoeLocalWorklet, Nvfp4MoeLocalWorkletConfig,
    Nvfp4MoeLocalWorkletResolved, VllmGlm52DsaAttnLocalWorklet,
    VllmGlm52DsaAttnLocalWorkletResolved,
};

const ARCH_KIND: &str = "glm52_vllm_nvfp4_dp_attn_dsa_moe";
const NUM_SPARSE_CYCLES: u32 = 18;
const NUM_SHARED_PER_CYCLE: u32 = 3;
/// vLLM's `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`: at or below this many
/// local tokens the shared expert runs on an auxiliary stream.
const SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: u32 = 256;
/// One engine's scheduler budget (vLLM's default `max_num_batched_tokens`).
/// It bounds the collective grids at `ep_size` times this many gathered
/// tokens; a larger budget extrapolates past the grid.
const MAX_RANK_BATCH_TOKENS: u32 = 8_192;
const COLLECTIVE_BACKENDS: &[&str] = &["vllm_pynccl"];
const ALLTOALL_BACKENDS: &[&str] = &["flashinfer_mnnvl"];
/// The only group the `moe_alltoall` / `moe_alltoall_prepare` rows span.
const MEASURED_ALLTOALL_EP_SIZE: u16 = 8;
const ROUTED_MOE_BACKENDS: &[&str] = &["flashinfer_trtllm_routed_sm100"];
const ELEMENTWISE_BACKENDS: &[&str] = &["triton"];

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

#[derive(Clone, Debug)]
pub struct Glm52VllmNvfp4DpAttnParallel {
    /// Data-parallel attention ranks, which are also the expert-parallel group.
    pub ep_size: u16,
    pub nvl_num_gpu: u16,
    pub max_model_len: u32,
    pub gpu_name: String,
    /// vLLM's CUDA-graph capture sizes, ascending and deduplicated. Empty:
    /// every step runs eager.
    pub cudagraph_capture_sizes: Vec<u32>,
    /// How the expert exchange runs and is priced (`arch::ep_comm`). The
    /// default is the measured all-gather / reduce-scatter.
    pub ep_comm: EpCommConfig,
}

/// The EP1 recipe every GPU's non-expert work builds from, plus the expert
/// shards and the two collectives around them.
#[derive(Clone, Debug)]
pub struct Glm52VllmNvfp4DpAttnConfigs {
    pub parallel: Glm52VllmNvfp4DpAttnParallel,
    /// TP1 attention, dense FFN, shared expert, router (with its top-k select)
    /// and full-vocabulary head. Its own `nvfp4_moe` is empty: the routed
    /// experts are [`Self::routed_experts`].
    pub local: Glm52VllmNvfp4DsaMoeConfigs,
    /// One identity-free active-count-ranked workload per EP rank, on the
    /// precomputed-routing backend.
    pub routed_experts: Vec<Nvfp4MoeLocalWorkletConfig>,
    pub dispatch: MoeEpQuantizedAllGatherKernelConfig,
    pub combine: MoeEpReduceScatterKernelConfig,
    /// The analytic exchange, and the all-to-all's row model on the measured
    /// path; `None` on the measured all-gather / reduce-scatter.
    pub analytic: Option<AnalyticEpComm>,
    /// The all-to-all's elementwise leaves; `None` unless `ep_comm` is
    /// `alltoall`.
    pub alltoall: Option<AlltoallElementwise>,
    /// The measured all-to-all's kernels; `None` unless `ep_comm` is
    /// `alltoall` and `ep_comm_model` is `measured`.
    pub measured_alltoall: Option<MeasuredAlltoall>,
}

/// The profiled flashinfer MNNVL all-to-all of vLLM's two-sided path.
#[derive(Clone, Debug)]
pub struct MeasuredAlltoall {
    /// `mnnvl_moe_alltoallv_prepare_without_allgather`.
    pub prepare: MoeAlltoallPrepareKernelConfig,
    /// The packed NVFP4 activations, `hidden / 2` bytes per row.
    pub dispatch: MoeAlltoallKernelConfig,
    /// Their fp8 block scales, `hidden / 16` bytes per row: a second
    /// `moe_comm` call.
    pub dispatch_scales: MoeAlltoallKernelConfig,
    /// The bf16 expert outputs back, `2 hidden` bytes per row.
    pub combine: MoeAlltoallKernelConfig,
}

/// The all-to-all's measured elementwise work around its two transfers.
#[derive(Clone, Debug)]
pub struct AlltoallElementwise {
    /// `nvfp4_block_scale_interleave` on the received block scales, over the
    /// receive allocation (`max tokens per rank x ep_size` rows).
    pub scale_swizzle: ElementwiseKernelConfig,
    /// `torch.zeros(tokens * top_k, hidden)` bf16 staging buffer.
    pub combine_fill: ElementwiseKernelConfig,
    /// `torch.sum` over the top-k axis back to one bf16 row per token.
    pub combine_reduce: ElementwiseKernelConfig,
}

#[derive(Clone, Debug)]
pub struct Glm52VllmNvfp4DpAttnResolved {
    pub parallel: Glm52VllmNvfp4DpAttnParallel,
    pub local: Glm52VllmNvfp4DsaMoeResolved,
    pub routed_experts: Vec<Nvfp4MoeLocalWorkletResolved>,
    pub dispatch: MoeEpQuantizedAllGatherKernelConfig,
    pub combine: MoeEpReduceScatterKernelConfig,
    pub analytic: Option<AnalyticEpComm>,
    pub alltoall: Option<AlltoallElementwise>,
    pub measured_alltoall: Option<MeasuredAlltoall>,
}

/// `body_demand` is the whole body's routed demand, folded over `ep_size`
/// ranks as the TP/EP graph folds it.
pub fn build_configs(
    model: &Glm52ModelCfg,
    parallel: &Glm52VllmNvfp4DpAttnParallel,
    body_demand: &ExpertDemand,
    fp8: bool,
) -> Result<Glm52VllmNvfp4DpAttnConfigs, BuildError> {
    build_configs_for(model, parallel, body_demand, fp8, None)
}

/// The target of a speculative deployment whose proposer is a separate
/// checkpoint (GLM-5.3's DFlash2): every rank verifies `draft_tokens + 1` rows
/// per decode request, through the TP/EP graph's
/// [`build_verify_configs`](ep_graph::build_verify_configs) recipe at TP1.
/// `body_demand` must group the verify width's rows.
pub fn build_verify_configs(
    model: &Glm52ModelCfg,
    parallel: &Glm52VllmNvfp4DpAttnParallel,
    body_demand: &ExpertDemand,
    fp8: bool,
    draft_tokens: u32,
) -> Result<Glm52VllmNvfp4DpAttnConfigs, BuildError> {
    build_configs_for(model, parallel, body_demand, fp8, Some(draft_tokens))
}

fn build_configs_for(
    model: &Glm52ModelCfg,
    parallel: &Glm52VllmNvfp4DpAttnParallel,
    body_demand: &ExpertDemand,
    fp8: bool,
    draft_tokens: Option<u32>,
) -> Result<Glm52VllmNvfp4DpAttnConfigs, BuildError> {
    parallel.ep_comm.validate().map_err(fit_failed)?;
    match parallel.ep_comm.model {
        EpCommModel::Measured if parallel.ep_comm.kind == EpCommKind::Alltoall => {
            // The MNNVL rows are profiled on one 8-GPU group.
            if parallel.ep_size != MEASURED_ALLTOALL_EP_SIZE {
                return Err(fit_failed(format!(
                    "ep_size {} has no measured all-to-all rows: ep_comm alltoall with \
                     ep_comm_model measured needs ep_size {MEASURED_ALLTOALL_EP_SIZE} \
                     (ep_comm_model analytic goes further)",
                    parallel.ep_size
                )));
            }
            if parallel.nvl_num_gpu < parallel.ep_size {
                return Err(fit_failed(format!(
                    "nvl_num_gpu {} must hold the ep_size {} group in one NVLink domain",
                    parallel.nvl_num_gpu, parallel.ep_size
                )));
            }
        }
        EpCommModel::Measured => {
            // The two collective kernels are measured on 2, 4 and 8 GPUs of one node.
            if !matches!(parallel.ep_size, 2 | 4 | 8) {
                return Err(fit_failed(format!(
                    "ep_size {} must be 2, 4 or 8: data-parallel attention needs more than one \
                     rank, and the dispatch and combine collectives span one NVLink node \
                     (ep_comm_model analytic goes further)",
                    parallel.ep_size
                )));
            }
            if parallel.nvl_num_gpu != parallel.ep_size {
                return Err(fit_failed(format!(
                    "nvl_num_gpu {} must equal ep_size {}: the group is one NVLink node",
                    parallel.nvl_num_gpu, parallel.ep_size
                )));
            }
        }
        EpCommModel::Analytic => {
            let experts = model.num_experts.get();
            if parallel.ep_size < 2 || experts % u32::from(parallel.ep_size) != 0 {
                return Err(fit_failed(format!(
                    "ep_size {} must be >= 2 and divide the {experts} routed experts",
                    parallel.ep_size
                )));
            }
            if parallel.nvl_num_gpu < parallel.ep_size {
                return Err(fit_failed(format!(
                    "nvl_num_gpu {} must hold the ep_size {} group in one NVLink domain",
                    parallel.nvl_num_gpu, parallel.ep_size
                )));
            }
        }
    }
    if !(1..=CHECKPOINT_MAX_CONTEXT).contains(&parallel.max_model_len) {
        return Err(fit_failed(format!(
            "max_model_len {} must be in 1..={CHECKPOINT_MAX_CONTEXT}",
            parallel.max_model_len
        )));
    }
    let sizes = &parallel.cudagraph_capture_sizes;
    if sizes.first() == Some(&0) || sizes.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(fit_failed(format!(
            "cudagraph_capture_sizes {sizes:?} must be positive, ascending and distinct"
        )));
    }
    let rank = Glm52VllmNvfp4DsaMoeParallel {
        ep_size: 1,
        nvl_num_gpu: 1,
        max_model_len: parallel.max_model_len,
        gpu_name: parallel.gpu_name.clone(),
    };
    let mut local = match draft_tokens {
        None => ep_graph::build_configs(model, &rank, body_demand, None, fp8, Glm52MtpMode::Off)?,
        Some(draft_tokens) => {
            ep_graph::build_verify_configs(model, &rank, body_demand, fp8, draft_tokens)?
        }
    };
    // The modular experts take routing already chosen, so the router runs
    // vLLM's bias top-k on each GPU's own tokens before dispatch.
    local.sparse_router.include_router_select = true;
    let template = local
        .nvfp4_moe
        .first()
        .cloned()
        .ok_or_else(|| fit_failed("the EP1 recipe has no routed-expert config"))?;
    local.nvfp4_moe = Vec::new();
    let routed_experts = Nvfp4MoeLocalWorkletConfig::split_for_ep(
        Nvfp4MoeLocalWorkletConfig {
            ep_size: parallel.ep_size,
            moe_backends: ROUTED_MOE_BACKENDS.to_vec(),
            ..template
        },
        body_demand.clone(),
    );
    let max_total_tokens = MAX_RANK_BATCH_TOKENS
        .checked_mul(u32::from(parallel.ep_size))
        .ok_or_else(|| fit_failed("gathered token bound overflows u32"))?;
    let dispatch = MoeEpQuantizedAllGatherKernelConfig {
        backends: COLLECTIVE_BACKENDS.to_vec(),
        gpu_name: parallel.gpu_name.clone(),
        num_gpus: u32::from(parallel.ep_size),
        hidden_size: model.hidden_dim.clone(),
        top_k: model.router_top_k.into(),
        activation_dtype: DType::Nvfp4E2m1,
        fabric: "nvlink".to_string(),
        max_total_tokens,
    };
    let combine = MoeEpReduceScatterKernelConfig {
        backends: COLLECTIVE_BACKENDS.to_vec(),
        gpu_name: parallel.gpu_name.clone(),
        num_gpus: u32::from(parallel.ep_size),
        hidden_size: model.hidden_dim.clone(),
        dtype: DType::Bf16,
        fabric: "nvlink".to_string(),
        max_total_tokens,
    };
    let hidden = model.hidden_dim.get();
    let top_k = u32::from(model.router_top_k);
    let alltoall_kind = parallel.ep_comm.kind == EpCommKind::Alltoall;
    // The measured all-to-all still takes its rows from the analytic model.
    let analytic = match (parallel.ep_comm.model, alltoall_kind) {
        (EpCommModel::Measured, false) => None,
        _ => Some(
            AnalyticEpComm::new(
                parallel.ep_comm.clone(),
                parallel.ep_size,
                hidden,
                model.num_experts.get(),
                top_k,
                body_demand,
            )
            .map_err(|error| fit_failed(format!("analytic ep_comm: {error:#}")))?,
        ),
    };
    let elementwise = |input: u32, output: u32| ElementwiseKernelConfig {
        backends: ELEMENTWISE_BACKENDS.to_vec(),
        gpu_name: parallel.gpu_name.clone(),
        input_bytes_per_token: input.into(),
        output_bytes_per_token: output.into(),
    };
    let bf16_row = 2 * hidden;
    let alltoall = alltoall_kind.then(|| AlltoallElementwise {
        scale_swizzle: elementwise(hidden / 16, hidden / 16),
        combine_fill: elementwise(0, top_k * bf16_row),
        combine_reduce: elementwise(top_k * bf16_row, bf16_row),
    });
    let alltoall_kernel = |hidden_bytes: u32, direction| MoeAlltoallKernelConfig {
        backends: ALLTOALL_BACKENDS.to_vec(),
        gpu_name: parallel.gpu_name.clone(),
        ep_size: u32::from(parallel.ep_size),
        top_k,
        // No EPLB redundancy: one slot per expert.
        slot_count: model.num_experts.get(),
        hidden_bytes: hidden_bytes.into(),
        direction,
        fabric: Fabric::Nvlink,
    };
    let measured_alltoall = (alltoall_kind && parallel.ep_comm.model == EpCommModel::Measured)
        .then(|| MeasuredAlltoall {
            prepare: MoeAlltoallPrepareKernelConfig {
                backends: ALLTOALL_BACKENDS.to_vec(),
                gpu_name: parallel.gpu_name.clone(),
                ep_size: u32::from(parallel.ep_size),
                top_k,
                slot_count: model.num_experts.get(),
                fabric: Fabric::Nvlink,
            },
            dispatch: alltoall_kernel(hidden / 2, MoeAlltoallDirection::Dispatch),
            dispatch_scales: alltoall_kernel(hidden / 16, MoeAlltoallDirection::Dispatch),
            combine: alltoall_kernel(bf16_row, MoeAlltoallDirection::Combine),
        });
    Ok(Glm52VllmNvfp4DpAttnConfigs {
        parallel: parallel.clone(),
        local,
        routed_experts,
        dispatch,
        combine,
        analytic,
        alltoall,
        measured_alltoall,
    })
}

pub fn resolve_configs(cfgs: &Glm52VllmNvfp4DpAttnConfigs) -> Glm52VllmNvfp4DpAttnResolved {
    Glm52VllmNvfp4DpAttnResolved {
        parallel: cfgs.parallel.clone(),
        local: ep_graph::resolve_configs(&cfgs.local),
        routed_experts: cfgs
            .routed_experts
            .iter()
            .map(Nvfp4MoeLocalWorklet::resolve_config)
            .collect(),
        dispatch: cfgs.dispatch.clone(),
        combine: cfgs.combine.clone(),
        analytic: cfgs.analytic.clone(),
        alltoall: cfgs.alltoall.clone(),
        measured_alltoall: cfgs.measured_alltoall.clone(),
    }
}

/// One sparse decoder layer under DP attention and EP MoE.
struct DpSparseLayer {
    name: String,
    attention: VllmGlm52DsaAttnLocalWorklet,
    router: Glm52MoeRouterLocalWorklet,
    shared_expert: Glm52SharedExpertLocalWorklet,
    input_quant: Op<Nvfp4QuantKernel>,
    exchange: Exchange,
    /// One per EP rank, ordered by identity-free active-expert workload.
    routed_experts: Vec<Op<Nvfp4FusedMoeKernel>>,
}

/// How the routed tokens reach the experts and come back.
enum Exchange {
    /// The profiled quantized all-gather and bf16 reduce-scatter.
    MeasuredAllgather {
        dispatch: Op<MoeEpQuantizedAllGatherKernel>,
        combine: Op<MoeEpReduceScatterKernel>,
    },
    /// `arch::ep_comm`'s all-gather / reduce-scatter.
    AnalyticAllgather(Arc<AnalyticEpComm>),
    /// flashinfer's two-sided MNNVL all-to-all with its measured elementwise
    /// work. `comm` gives the busiest ranks' rows, and prices the prepare and
    /// transfers unless `measured` does.
    Alltoall {
        comm: Arc<AnalyticEpComm>,
        measured: Option<MeasuredAlltoallOps>,
        scale_swizzle: Op<ElementwiseKernel>,
        combine_fill: Op<ElementwiseKernel>,
        combine_reduce: Op<ElementwiseKernel>,
    },
}

/// The profiled all-to-all prepare and transfers (EP8).
struct MeasuredAlltoallOps {
    prepare: Op<MoeAlltoallPrepareKernel>,
    dispatch: Op<MoeAlltoallKernel>,
    dispatch_scales: Op<MoeAlltoallKernel>,
    combine: Op<MoeAlltoallKernel>,
}

/// The measured all-to-all's lookups at the analytic model's rows: the
/// prepare at the busiest rank's tokens, dispatch at (busiest sender,
/// busiest receiver) and combine at its transpose. Each `moe_comm` call
/// moves whole rows.
struct MeasuredAlltoallInputs {
    prepare: MoeAlltoallPrepareKernelInput,
    dispatch: MoeAlltoallKernelInput,
    combine: MoeAlltoallKernelInput,
}

impl MeasuredAlltoallInputs {
    fn new(rows: &AlltoallRows) -> Self {
        let whole = |rows: f64| {
            if rows > 0.0 {
                (rows.round() as u32).max(1)
            } else {
                0
            }
        };
        let (send, recv) = (whole(rows.send), whole(rows.recv));
        Self {
            prepare: MoeAlltoallPrepareKernelInput {
                tokens_per_rank: rows.busiest_tokens,
            },
            dispatch: MoeAlltoallKernelInput {
                max_send_rows: send,
                max_recv_rows: recv,
            },
            // Each expert rank returns one reduced row per row it got.
            combine: MoeAlltoallKernelInput {
                max_send_rows: recv,
                max_recv_rows: send,
            },
        }
    }
}

impl DpSparseLayer {
    fn build(
        name: String,
        attention: VllmGlm52DsaAttnLocalWorkletResolved,
        resolved: &Glm52VllmNvfp4DpAttnResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        // Every rank's worklet keeps the TP/EP graph's slot names. Each one's
        // quantization leaf is the same kernel; only the first is used, once
        // per rank, on that rank's own tokens.
        let experts = resolved
            .routed_experts
            .iter()
            .map(|rank| {
                Nvfp4MoeLocalWorklet::build(
                    format!("{name}.moe.routed_experts"),
                    rank.clone(),
                    bridge,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input_quant = experts
            .first()
            .map(|expert| Op::new(expert.quant.name.clone(), Arc::clone(&expert.quant.kernel)))
            .ok_or_else(|| fit_failed("no routed-expert rank to quantize for"))?;
        Ok(Self {
            attention: VllmGlm52DsaAttnLocalWorklet::build(
                format!("{name}.attention"),
                attention,
                bridge,
            )?,
            router: Glm52MoeRouterLocalWorklet::build(
                format!("{name}.moe.router"),
                resolved.local.sparse_router.clone(),
                bridge,
            )?,
            shared_expert: Glm52SharedExpertLocalWorklet::build(
                format!("{name}.moe.shared_expert"),
                resolved.local.shared_expert.clone(),
                bridge,
            )?,
            input_quant,
            exchange: Exchange::build(&name, resolved, bridge)?,
            routed_experts: experts.into_iter().map(|expert| expert.fused_moe).collect(),
            name,
        })
    }

    fn ranks(&self) -> usize {
        self.routed_experts.len()
    }

    /// Leaves are numbered in the order they are declared here, and `eval`
    /// pushes them in the same order.
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let ranks = self.ranks();
        let local = labeled_max(
            format!(
                "{} rank-local segment [Max over DP ranks: attention -> router + select -> \
                 serial shared expert -> NVFP4 input quant]",
                self.name
            ),
            (0..ranks)
                .map(|_| {
                    CostNode::Sum(vec![
                        self.attention.compile(builder),
                        self.router.compile(builder),
                        self.shared_expert.compile(builder),
                        self.input_quant.compile(builder),
                    ])
                })
                .collect(),
        );
        let aux_shared = labeled_max(
            format!(
                "{}.moe.shared_expert on the auxiliary stream [Max over DP ranks; <= {} local tokens]",
                self.name, SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            ),
            (0..ranks)
                .map(|_| self.shared_expert.compile(builder))
                .collect(),
        );
        let ranks_max = |builder: &mut CostTreeBuilder, op: &Op<ElementwiseKernel>, what: &str| {
            labeled_max(
                format!("{}.moe.{what} [Max over EP ranks]", self.name),
                (0..ranks).map(|_| op.compile(builder)).collect(),
            )
        };
        let mut chain = Vec::new();
        match &self.exchange {
            Exchange::MeasuredAllgather { dispatch, .. } => chain.push(dispatch.compile(builder)),
            Exchange::AnalyticAllgather(comm) => chain.push(comm.compile_leg(
                format!("{}.moe.dispatch_ep_all_gather", self.name),
                "dispatch",
                builder,
            )),
            Exchange::Alltoall {
                comm,
                measured,
                scale_swizzle,
                ..
            } => {
                match measured {
                    Some(measured) => {
                        chain.push(measured.prepare.compile(builder));
                        chain.push(measured.dispatch.compile(builder));
                        chain.push(measured.dispatch_scales.compile(builder));
                    }
                    None => {
                        chain.push(comm.compile_leg(
                            format!("{}.moe.dispatch_alltoall_prepare", self.name),
                            "prepare",
                            builder,
                        ));
                        chain.push(comm.compile_leg(
                            format!("{}.moe.dispatch_alltoall", self.name),
                            "dispatch",
                            builder,
                        ));
                        chain.push(comm.compile_leg(
                            format!("{}.moe.dispatch_alltoall_scales", self.name),
                            "dispatch_scales",
                            builder,
                        ));
                    }
                }
                chain.push(ranks_max(builder, scale_swizzle, "dispatch_scale_swizzle"));
            }
        }
        let routed = labeled_max(
            format!(
                "{}.moe.routed_experts [Max over EP ranks; every gathered token]",
                self.name
            ),
            self.routed_experts
                .iter()
                .map(|expert| expert.compile(builder))
                .collect(),
        );
        chain.push(routed);
        match &self.exchange {
            Exchange::MeasuredAllgather { combine, .. } => chain.push(combine.compile(builder)),
            Exchange::AnalyticAllgather(comm) => chain.push(comm.compile_leg(
                format!("{}.moe.combine_ep_reduce_scatter", self.name),
                "combine",
                builder,
            )),
            Exchange::Alltoall {
                comm,
                measured,
                combine_fill,
                combine_reduce,
                ..
            } => {
                chain.push(ranks_max(builder, combine_fill, "combine_output_fill"));
                chain.push(match measured {
                    Some(measured) => measured.combine.compile(builder),
                    None => comm.compile_leg(
                        format!("{}.moe.combine_alltoall", self.name),
                        "combine",
                        builder,
                    ),
                });
                chain.push(ranks_max(builder, combine_reduce, "combine_reduce"));
            }
        }
        let path = match &self.exchange {
            Exchange::MeasuredAllgather { .. } => {
                "quantized all-gather -> routed experts -> reduce-scatter".to_string()
            }
            Exchange::Alltoall {
                measured: Some(_), ..
            } => "measured flashinfer MNNVL all-to-all (EP8 rows)".to_string(),
            Exchange::AnalyticAllgather(comm) | Exchange::Alltoall { comm, .. } => comm.label(),
        };
        let exchange = CostNode::Labeled {
            label: format!(
                "{}.moe exchange [{path}, auxiliary-stream shared expert alongside]",
                self.name
            ),
            child: Box::new(CostNode::Parallel {
                overlap: 1.0,
                children: vec![aux_shared, CostNode::Sum(chain)],
            }),
        };
        CostNode::Labeled {
            label: format!(
                "{} [sparse layer; DP attention x{ranks}; EP={ranks} NVFP4 routed experts]",
                self.name
            ),
            child: Box::new(CostNode::Sum(vec![local, exchange])),
        }
    }

    fn eval(&self, batch: &DpBatch, ev: &mut Evaluator) {
        for (group, &rows) in batch.groups.iter().zip(&batch.rows) {
            self.attention.eval(&group.attention_input, ev);
            // No all-reduce owns the post-attention residual RMSNorm at TP1.
            self.router.eval_with_post_attn_norm(
                &Glm52MoeRouterLocalWorkletInput { batch_tokens: rows },
                true,
                ev,
            );
            self.shared_expert.eval_or_zero(
                &Glm52SharedExpertLocalWorkletInput { batch_tokens: rows },
                shares_aux_stream(rows),
                ev,
            );
            eval_atomic_or_zero(
                &self.input_quant,
                Nvfp4QuantKernelInput { num_tokens: rows },
                rows == 0,
                ev,
            );
        }
        for &rows in &batch.rows {
            self.shared_expert.eval_or_zero(
                &Glm52SharedExpertLocalWorkletInput { batch_tokens: rows },
                !shares_aux_stream(rows),
                ev,
            );
        }
        let idle = batch.gathered_tokens == 0;
        let analytic = match &self.exchange {
            Exchange::AnalyticAllgather(comm)
            | Exchange::Alltoall {
                comm,
                measured: None,
                ..
            } => comm.step(&batch.collective_tokens),
            _ => Default::default(),
        };
        let measured_inputs = match &self.exchange {
            Exchange::Alltoall {
                comm,
                measured: Some(_),
                ..
            } => Some(MeasuredAlltoallInputs::new(
                &comm.alltoall_rows(&batch.collective_tokens),
            )),
            _ => None,
        };
        match &self.exchange {
            Exchange::MeasuredAllgather { dispatch, .. } => eval_atomic_or_zero(
                dispatch,
                MoeEpCollectiveKernelInput {
                    per_rank_tokens: batch.collective_tokens.clone(),
                },
                idle,
                ev,
            ),
            Exchange::AnalyticAllgather(_) => push_leg("dispatch", analytic.dispatch, ev),
            Exchange::Alltoall {
                measured,
                scale_swizzle,
                ..
            } => {
                match (measured, &measured_inputs) {
                    (Some(measured), Some(inputs)) => {
                        eval_atomic_or_zero(&measured.prepare, inputs.prepare.clone(), idle, ev);
                        eval_atomic_or_zero(&measured.dispatch, inputs.dispatch.clone(), idle, ev);
                        // The block scales are a second `moe_comm` of the same rows.
                        eval_atomic_or_zero(
                            &measured.dispatch_scales,
                            inputs.dispatch.clone(),
                            idle,
                            ev,
                        );
                    }
                    _ => {
                        push_leg("prepare", analytic.prepare, ev);
                        push_leg("dispatch", analytic.dispatch, ev);
                        push_leg("dispatch_scales", analytic.dispatch_scales, ev);
                    }
                }
                // Every rank swizzles its whole receive allocation.
                let busiest = batch.collective_tokens.iter().copied().max().unwrap_or(0);
                let allocation = busiest.saturating_mul(self.ranks() as u32);
                for _ in 0..self.ranks() {
                    eval_atomic_or_zero(
                        scale_swizzle,
                        ElementwiseKernelInput {
                            num_tokens: allocation,
                        },
                        idle,
                        ev,
                    );
                }
            }
        }
        for expert in &self.routed_experts {
            eval_atomic_or_zero(
                expert,
                Nvfp4FusedMoeKernelInput {
                    num_tokens: batch.gathered_tokens,
                },
                idle,
                ev,
            );
        }
        match &self.exchange {
            Exchange::MeasuredAllgather { combine, .. } => eval_atomic_or_zero(
                combine,
                MoeEpCollectiveKernelInput {
                    per_rank_tokens: batch.collective_tokens.clone(),
                },
                idle,
                ev,
            ),
            Exchange::AnalyticAllgather(_) => push_leg("combine", analytic.combine, ev),
            Exchange::Alltoall {
                measured,
                combine_fill,
                combine_reduce,
                ..
            } => {
                // The staging buffer and the sum run on each rank's own tokens.
                for &tokens in &batch.collective_tokens {
                    eval_atomic_or_zero(
                        combine_fill,
                        ElementwiseKernelInput { num_tokens: tokens },
                        tokens == 0,
                        ev,
                    );
                }
                match (measured, measured_inputs) {
                    (Some(measured), Some(inputs)) => {
                        eval_atomic_or_zero(&measured.combine, inputs.combine, idle, ev);
                    }
                    _ => push_leg("combine", analytic.combine, ev),
                }
                for &tokens in &batch.collective_tokens {
                    eval_atomic_or_zero(
                        combine_reduce,
                        ElementwiseKernelInput { num_tokens: tokens },
                        tokens == 0,
                        ev,
                    );
                }
            }
        }
    }
}

impl Exchange {
    fn build(
        name: &str,
        resolved: &Glm52VllmNvfp4DpAttnResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let Some(comm) = &resolved.analytic else {
            return Ok(Self::MeasuredAllgather {
                dispatch: build_atomic(
                    format!("{name}.moe.dispatch_ep_all_gather"),
                    resolved.dispatch.clone(),
                    MoeEpQuantizedAllGatherKernel::build,
                    bridge,
                )?,
                combine: build_atomic(
                    format!("{name}.moe.combine_ep_reduce_scatter"),
                    resolved.combine.clone(),
                    MoeEpReduceScatterKernel::build,
                    bridge,
                )?,
            });
        };
        let comm = Arc::new(comm.clone());
        let Some(alltoall) = &resolved.alltoall else {
            return Ok(Self::AnalyticAllgather(comm));
        };
        let elementwise = |leaf: &str, config: &ElementwiseKernelConfig| {
            build_atomic(
                format!("{name}.moe.{leaf}"),
                config.clone(),
                ElementwiseKernel::build,
                bridge,
            )
        };
        let measured = resolved
            .measured_alltoall
            .as_ref()
            .map(|kernels| -> Result<_, BuildError> {
                let transfer = |leaf: &str, config: &MoeAlltoallKernelConfig| {
                    build_atomic(
                        format!("{name}.moe.{leaf}"),
                        config.clone(),
                        MoeAlltoallKernel::build,
                        bridge,
                    )
                };
                Ok(MeasuredAlltoallOps {
                    prepare: build_atomic(
                        format!("{name}.moe.dispatch_alltoall_prepare"),
                        kernels.prepare.clone(),
                        MoeAlltoallPrepareKernel::build,
                        bridge,
                    )?,
                    dispatch: transfer("dispatch_alltoall", &kernels.dispatch)?,
                    dispatch_scales: transfer(
                        "dispatch_alltoall_scales",
                        &kernels.dispatch_scales,
                    )?,
                    combine: transfer("combine_alltoall", &kernels.combine)?,
                })
            })
            .transpose()?;
        Ok(Self::Alltoall {
            comm,
            measured,
            scale_swizzle: elementwise("dispatch_scale_swizzle", &alltoall.scale_swizzle)?,
            combine_fill: elementwise("combine_output_fill", &alltoall.combine_fill)?,
            combine_reduce: elementwise("combine_reduce", &alltoall.combine_reduce)?,
        })
    }
}

/// vLLM moves the shared expert to its auxiliary stream only for small batches.
fn shares_aux_stream(local_tokens: u32) -> bool {
    local_tokens > 0 && local_tokens <= SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
}

fn labeled_max(label: String, children: Vec<CostNode>) -> CostNode {
    CostNode::Labeled {
        label,
        child: Box::new(CostNode::Max {
            overlap: 1.0,
            children,
        }),
    }
}

/// One iteration's attention groups, one per DP rank, and what the expert
/// exchange sees of them.
struct DpBatch {
    groups: Vec<NormalizedGroup>,
    /// Rows each rank's kernels outside attention run on: the shared CUDA
    /// graph size when the step replays graphs, else the rank's own batch
    /// (zero on an idle rank, whose dummy compute is not billed).
    rows: Vec<u32>,
    /// Tokens each rank contributes to the gather and receives from the
    /// scatter: the shared graph size when padded, else its batch, or vLLM's
    /// one-token dummy batch when it has none.
    collective_tokens: Vec<u32>,
    /// Every token the routed experts see on each rank; zero only when no
    /// rank has work.
    gathered_tokens: u32,
    /// Each rank's HiSparse miss copy; empty without HiSparse.
    hisparse: Vec<HisparseRankStep>,
}

/// vLLM's DP step agreement (`vllm/v1/worker/dp_utils.py`
/// `_synchronize_dp_ranks`): each rank, an idle one with its one-token dummy
/// batch, dispatches a CUDA graph only if its count fits the largest captured
/// size; the ranks keep the minimum mode, and pad to the largest padded count
/// unless that minimum is eager. Padding is monotone, so the shared size is
/// the busiest rank's graph. `None`: the step runs eager and ragged.
fn dp_graph_rows(sorted_capture_sizes: &[u32], collective_tokens: &[u32]) -> Option<u32> {
    let busiest = collective_tokens.iter().copied().max()?;
    let largest = sorted_capture_sizes.last().copied()?;
    (busiest > 0 && busiest <= largest).then(|| graph_padded_tokens(sorted_capture_sizes, busiest))
}

fn normalize_input(
    input: &UnifiedArchInput,
    ep_size: u16,
    verify_width: u32,
    max_model_len: u32,
    sorted_capture_sizes: &[u32],
) -> Result<DpBatch, String> {
    let ranks = usize::from(ep_size);
    if input.groups.len() != ranks {
        return Err(format!(
            "expected one attention group per data-parallel rank ({ranks}), got {}",
            input.groups.len()
        ));
    }
    let groups = input
        .groups
        .iter()
        .enumerate()
        .map(|(index, group)| normalize_group(index, group, verify_width, max_model_len))
        .collect::<Result<Vec<_>, _>>()?;
    if !input.tokens_per_source_rank.is_empty() {
        let expected: Vec<u32> = groups.iter().map(|group| group.batch_tokens).collect();
        if input.tokens_per_source_rank != expected {
            return Err(format!(
                "tokens_per_source_rank {:?} must equal the groups' batch tokens {expected:?}",
                input.tokens_per_source_rank
            ));
        }
    }
    let any_work = groups.iter().any(|group| group.batch_tokens > 0);
    let mut collective_tokens: Vec<u32> = groups
        .iter()
        .map(|group| match (any_work, group.batch_tokens) {
            (false, _) => 0,
            (true, 0) => 1,
            (true, tokens) => tokens,
        })
        .collect();
    let rows = match dp_graph_rows(sorted_capture_sizes, &collective_tokens) {
        Some(padded) => {
            collective_tokens.fill(padded);
            vec![padded; ranks]
        }
        None => groups.iter().map(|group| group.batch_tokens).collect(),
    };
    let gathered_tokens = collective_tokens
        .iter()
        .try_fold(0_u32, |sum, &tokens| sum.checked_add(tokens))
        .ok_or_else(|| "gathered token sum overflows u32".to_string())?;
    Ok(DpBatch {
        groups,
        rows,
        collective_tokens,
        gathered_tokens,
        hisparse: Vec::new(),
    })
}

/// The whole replica: `ep_size` GPUs, each its own attention DP rank.
pub struct Glm52VllmNvfp4DpAttnDsaMoeModel {
    name: String,
    ep_size: u16,
    max_model_len: u32,
    embedding: Op<ElementwiseKernel>,
    dense_attention: VllmGlm52DsaAttnLocalWorklet,
    dense_ffn: Glm52DenseFfnLocalWorklet,
    initial_shared: DpSparseLayer,
    cycle_full: DpSparseLayer,
    cycle_shared: DpSparseLayer,
    final_norm: Op<ResidualRmsNormKernel>,
    /// Full vocabulary: a TP1 `ParallelLMHead` shards nothing.
    lm_head: Op<SingleGemmKernel>,
    kv_bytes_per_token: u64,
    /// Query rows each decode request presents: 1, or `draft_tokens + 1` for
    /// the target of a DFlash2 verify.
    verify_width: u32,
    cudagraph_capture_sizes: Vec<u32>,
    /// SGLang HiSparse KV offload (see [`Self::with_hisparse`]); `None` keeps
    /// the whole cache on the GPU.
    hisparse: Option<HisparseConfig>,
    host_offload: Option<KvHostOffload>,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: Glm52VllmNvfp4DpAttnResolved,
    bridge: &PerfApiBridge,
) -> Result<Glm52VllmNvfp4DpAttnDsaMoeModel, BuildError> {
    let recipe = &resolved.local.raw_cfg;
    if recipe.parallel.ep_size != 1 || recipe.parallel.nvl_num_gpu != 1 {
        return Err(fit_failed(format!(
            "the per-GPU recipe must be TP1, got ep_size {} and nvl_num_gpu {}",
            recipe.parallel.ep_size, recipe.parallel.nvl_num_gpu
        )));
    }
    // A verify is supported only for a proposer outside this graph: the MTP
    // layer's draft passes are not modeled under data-parallel attention.
    if recipe.mtp_mode != Glm52MtpMode::Off {
        return Err(fit_failed("data-parallel attention does not run the MTP layer"));
    }
    let verify_width = match recipe.speculative_draft_tokens {
        None => 1,
        Some(draft_tokens) => draft_tokens
            .checked_add(1)
            .filter(|&width| width > 1)
            .ok_or_else(|| fit_failed("speculative draft_tokens must be in 1..u32::MAX"))?,
    };
    let ep_size = resolved.parallel.ep_size;
    if resolved.routed_experts.len() != usize::from(ep_size)
        || resolved.dispatch.num_gpus != u32::from(ep_size)
        || resolved.combine.num_gpus != u32::from(ep_size)
        || resolved
            .analytic
            .as_ref()
            .is_some_and(|comm| comm.ep_size != ep_size)
    {
        return Err(fit_failed(format!(
            "the expert shards and collectives must span ep_size {ep_size} ranks"
        )));
    }
    let local = &resolved.local;
    let sparse = |section: &str, attention: &VllmGlm52DsaAttnLocalWorkletResolved| {
        DpSparseLayer::build(
            format!("{name}.body.{section}"),
            attention.clone(),
            &resolved,
            bridge,
        )
    };
    let initial_shared = sparse(
        "sparse_initial_index_share",
        &local.initial_shared_attention,
    )?;
    let cycle_full = sparse("sparse_cycle_full_index", &local.cycle_full_attention)?;
    let cycle_shared = sparse("sparse_cycle_index_share", &local.cycle_shared_attention)?;
    let mut model = Glm52VllmNvfp4DpAttnDsaMoeModel {
        embedding: build_atomic(
            format!("{name}.main.embedding"),
            local.embedding.clone(),
            ElementwiseKernel::build,
            bridge,
        )?,
        dense_attention: VllmGlm52DsaAttnLocalWorklet::build(
            format!("{name}.body.dense_full_index.attention"),
            local.dense_full_index_attention.clone(),
            bridge,
        )?,
        dense_ffn: Glm52DenseFfnLocalWorklet::build(
            format!("{name}.body.dense_full_index.ffn"),
            local.dense_ffn.clone(),
            bridge,
        )?,
        initial_shared,
        cycle_full,
        cycle_shared,
        final_norm: build_atomic(
            format!("{name}.main.final_residual_rms_norm"),
            local.final_norm.clone(),
            ResidualRmsNormKernel::build,
            bridge,
        )?,
        lm_head: build_atomic(
            format!("{name}.main.lm_head"),
            local.lm_head.clone(),
            SingleGemmKernel::build,
            bridge,
        )?,
        // A token's whole cache lives on the one rank that admitted it.
        kv_bytes_per_token: state_bytes_per_token(1, Glm52MtpMode::Off)?,
        name,
        ep_size,
        max_model_len: recipe.parallel.max_model_len,
        verify_width,
        cudagraph_capture_sizes: resolved.parallel.cudagraph_capture_sizes.clone(),
        hisparse: None,
        host_offload: None,
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    model.recompile();
    Ok(model)
}

impl Glm52VllmNvfp4DpAttnDsaMoeModel {
    pub fn ep_size(&self) -> u16 {
        self.ep_size
    }

    pub fn verify_width(&self) -> u32 {
        self.verify_width
    }

    fn recompile(&mut self) {
        let tree = self.cost_tree();
        self.cost_flat = tree.flatten();
        self.n_slots = tree.n_slots();
    }

    /// Serve the cache through SGLang HiSparse (`arch::hisparse`): each rank
    /// keeps every layer's indexer keys and a `buffer_tokens` latent buffer
    /// per request on the GPU, the whole latent cache in its share of the
    /// node's host DRAM (`nvl_num_gpu` GPUs per node unless the config says
    /// otherwise), and pays the exposed host -> device miss copy as one more
    /// `Max` over ranks per step. Kernels are unchanged.
    pub fn with_hisparse(
        mut self,
        config: HisparseConfig,
        nvl_num_gpu: u16,
    ) -> Result<Self, BuildError> {
        config.validate().map_err(fit_failed)?;
        let layers = u64::from(ep_graph::NUM_LAYERS);
        let latent_bytes_per_token = layers * ep_graph::mla_latent_bytes_per_layer_token();
        // A token's indexer keys stay on the GPU: one per full-index layer
        // (IndexShare layers build no indexer and cache no key).
        let index_layers = ep_graph::FULL_INDEX_LAYERS.len() as u64;
        self.kv_bytes_per_token = index_layers * ep_graph::index_key_bytes_per_layer_token();
        self.host_offload = Some(config.kv_host_offload(latent_bytes_per_token, 0, nvl_num_gpu, 1));
        self.hisparse = Some(config);
        self.recompile();
        Ok(self)
    }

    fn per_rank(
        &self,
        builder: &mut CostTreeBuilder,
        leaf: impl Fn(&mut CostTreeBuilder) -> CostNode,
    ) -> Vec<CostNode> {
        (0..self.ep_size).map(|_| leaf(builder)).collect()
    }

    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let root = self.compile_root(&mut builder);
        builder.finish(root)
    }

    /// The whole graph as one node of `builder`, so a speculative model can
    /// compile it beside its proposer. Slots are declared in `eval_into` order.
    pub(crate) fn compile_root(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let embedding = labeled_max(
            format!("{}.main.embedding [Max over DP ranks]", self.name),
            self.per_rank(builder, |b| self.embedding.compile(b)),
        );
        let dense = CostNode::Labeled {
            label: "layers 0..2: dense + full index (3 layers) [Max over DP ranks]".to_string(),
            child: Box::new(CostNode::Scale {
                n: NUM_DENSE_LAYERS,
                child: Box::new(CostNode::Max {
                    overlap: 1.0,
                    children: self.per_rank(builder, |b| {
                        CostNode::Sum(vec![
                            self.dense_attention.compile(b),
                            self.dense_ffn.compile(b),
                        ])
                    }),
                }),
            }),
        };
        let initial_shared = CostNode::Labeled {
            label: "layers 3..5: sparse + IndexShare (3 layers)".to_string(),
            child: Box::new(CostNode::Scale {
                n: NUM_INITIAL_SHARED_LAYERS,
                child: Box::new(self.initial_shared.compile(builder)),
            }),
        };
        let cycle = CostNode::Labeled {
            label: "layers 6..77: 18 cycles of full-index + 3 IndexShare".to_string(),
            child: Box::new(CostNode::Scale {
                n: NUM_SPARSE_CYCLES,
                child: Box::new(CostNode::Sum(vec![
                    self.cycle_full.compile(builder),
                    CostNode::Scale {
                        n: NUM_SHARED_PER_CYCLE,
                        child: Box::new(self.cycle_shared.compile(builder)),
                    },
                ])),
            }),
        };
        let head = labeled_max(
            format!(
                "{}.main output head [Max over DP ranks: final residual RMSNorm -> full-vocab lm_head]",
                self.name
            ),
            self.per_rank(builder, |b| {
                CostNode::Sum(vec![self.final_norm.compile(b), self.lm_head.compile(b)])
            }),
        );
        let mut sections = vec![embedding, dense, initial_shared, cycle, head];
        if let Some(hisparse) = &self.hisparse {
            sections.push(hisparse.compile(&self.name, self.ep_size, builder));
        }
        CostNode::Labeled {
            label: format!(
                "{} (Glm52VllmNvfp4DpAttnDsaMoeModel) [DP attention x{n} at TP1; EP{n} NVFP4 MoE; \
                 {} query row(s) per decode request; timing_context<={}]",
                self.name,
                self.verify_width,
                self.max_model_len,
                n = self.ep_size
            ),
            child: Box::new(CostNode::Sum(sections)),
        }
    }

    fn normalize(&self, input: &UnifiedArchInput) -> DpBatch {
        let mut batch = normalize_input(
            input,
            self.ep_size,
            self.verify_width,
            self.max_model_len,
            &self.cudagraph_capture_sizes,
        )
        .unwrap_or_else(|reason| panic!("invalid Glm52VllmNvfp4DpAttnDsaMoeModel input: {reason}"));
        if let Some(hisparse) = &self.hisparse {
            batch.hisparse = input
                .groups
                .iter()
                .map(|group| {
                    hisparse.rank_step(
                        &group.decode_kv_lens,
                        self.verify_width,
                        ep_graph::mla_latent_bytes_per_layer_token(),
                        ep_graph::NUM_LAYERS,
                    )
                })
                .collect();
        }
        batch
    }

    /// Fill the slots of [`Self::compile_root`] for one iteration.
    pub(crate) fn eval_into(&self, input: &UnifiedArchInput, ev: &mut Evaluator) {
        self.eval_normalized(&self.normalize(input), ev);
    }

    fn eval_normalized(&self, batch: &DpBatch, ev: &mut Evaluator) {
        for &rows in &batch.rows {
            eval_atomic_or_zero(
                &self.embedding,
                ElementwiseKernelInput { num_tokens: rows },
                rows == 0,
                ev,
            );
        }
        for (group, &rows) in batch.groups.iter().zip(&batch.rows) {
            self.dense_attention.eval(&group.attention_input, ev);
            self.dense_ffn.eval_with_post_attn_norm(
                &Glm52DenseFfnLocalWorkletInput { batch_tokens: rows },
                true,
                ev,
            );
        }
        self.initial_shared.eval(batch, ev);
        self.cycle_full.eval(batch, ev);
        self.cycle_shared.eval(batch, ev);
        for (group, &rows) in batch.groups.iter().zip(&batch.rows) {
            eval_atomic_or_zero(
                &self.final_norm,
                ResidualRmsNormKernelInput { m: rows },
                rows == 0,
                ev,
            );
            // vLLM computes logits outside the graph, on the sampled rows.
            eval_atomic_or_zero(
                &self.lm_head,
                SingleGemmKernelInput {
                    m: group.logits_rows,
                },
                group.logits_rows == 0,
                ev,
            );
        }
        for &step in &batch.hisparse {
            push_rank_step(step, ev);
        }
    }
}

impl IterwiseUnifiedModel for Glm52VllmNvfp4DpAttnDsaMoeModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        normalize_input(
            batch,
            self.ep_size,
            self.verify_width,
            self.max_model_len,
            &self.cudagraph_capture_sizes,
        )
        .map(|_| ())
    }

    /// One token's cache, which one rank holds whole: 78 layers of MLA latent
    /// plus index key, with nothing replicated.
    fn total_kv_bytes_per_token(&self) -> u64 {
        self.kv_bytes_per_token
    }

    fn kv_host_offload(&self) -> Option<KvHostOffload> {
        self.host_offload
    }

    fn gpus_per_replica(&self) -> u16 {
        self.ep_size
    }

    /// One KV partition per engine.
    fn num_attn_dp_groups(&self) -> u16 {
        self.ep_size
    }

    /// Each partition is one GPU's cache.
    fn num_attn_shards(&self) -> u16 {
        1
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
        let batch = self.normalize(batch);
        eval_compiled(&self.cost_flat, self.n_slots, slots, scratch, None, |ev| {
            self.eval_normalized(&batch, ev)
        })
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        let batch = self.normalize(batch);
        eval_compiled(
            &self.cost_flat,
            self.n_slots,
            slots,
            scratch,
            Some(inputs),
            |ev| self.eval_normalized(&batch, ev),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::ArchGroupInput;
    use crate::arch::glm52_vllm_nvfp4_dsa_moe::{decoder_layer_state_bytes_per_token, NUM_LAYERS};
    use crate::timing::routing::RoutingDistribution;
    use std::collections::BTreeSet;
    use std::path::Path;

    // Leaves per rank in each section; the TP/EP graph's counts at TP1, plus
    // the router's top-k select that only this graph runs outside the experts.
    const ATTN_FULL_SLOTS: usize = 29;
    const ATTN_SHARED_SLOTS: usize = 14;
    const DENSE_FFN_SLOTS: usize = 4;
    const ROUTER_SLOTS: usize = 3;
    const SHARED_EXPERT_SLOTS: usize = 3;
    const INPUT_QUANT_SLOTS: usize = 1;
    const FUSED_MOE_SLOTS: usize = 1;
    /// The dispatch gather and the combine scatter.
    const COLLECTIVE_SLOTS: usize = 2;

    fn expected_slot_count(ep_size: u16) -> usize {
        let ep = usize::from(ep_size);
        let sparse = |attention: usize| {
            ep * (attention
                + ROUTER_SLOTS
                + 2 * SHARED_EXPERT_SLOTS
                + INPUT_QUANT_SLOTS
                + FUSED_MOE_SLOTS)
                + COLLECTIVE_SLOTS
        };
        let embedding = ep;
        let dense = ep * (ATTN_FULL_SLOTS + DENSE_FFN_SLOTS);
        let head = 2 * ep;
        embedding
            + dense
            + sparse(ATTN_SHARED_SLOTS)
            + sparse(ATTN_FULL_SLOTS)
            + sparse(ATTN_SHARED_SLOTS)
            + head
    }

    fn model_cfg(stem: &str) -> Glm52ModelCfg {
        Glm52ModelCfg::from_json(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("model/config/{stem}.json")),
        )
        .unwrap()
    }

    fn parallel(ep_size: u16) -> Glm52VllmNvfp4DpAttnParallel {
        Glm52VllmNvfp4DpAttnParallel {
            ep_size,
            nvl_num_gpu: ep_size,
            max_model_len: 131_072,
            gpu_name: "NVIDIA B200".to_string(),
            cudagraph_capture_sizes: Vec::new(),
            ep_comm: EpCommConfig::default(),
        }
    }

    /// vLLM's default capture grid up to `max`
    /// (`vllm/config/vllm.py` `_set_cudagraph_sizes`).
    fn vllm_capture_sizes(max: u32) -> Vec<u32> {
        [1, 2, 4]
            .into_iter()
            .chain((8..256.min(max + 1)).step_by(8))
            .chain((256..=max).step_by(16))
            .filter(|&size| size <= max)
            .collect()
    }

    // Structure-only tests: the routing law does not change the graph.
    fn demand() -> ExpertDemand {
        ExpertDemand::popularity(&RoutingDistribution::uniform(256), 1)
    }

    fn configs(stem: &str, ep_size: u16) -> Glm52VllmNvfp4DpAttnConfigs {
        build_configs(&model_cfg(stem), &parallel(ep_size), &demand(), false).unwrap()
    }

    fn built(stem: &str, ep_size: u16) -> Glm52VllmNvfp4DpAttnDsaMoeModel {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        build(
            "unified".to_string(),
            resolve_configs(&configs(stem, ep_size)),
            &bridge,
        )
        .unwrap()
    }

    fn group(prefill: &[(u32, u32)], decode: &[u32]) -> ArchGroupInput {
        let prefill_tokens = prefill.iter().map(|&(_, append)| append).sum();
        ArchGroupInput {
            batch_tokens: prefill_tokens + decode.len() as u32,
            prefill_tokens,
            decode_tokens: decode.len() as u32,
            prefill_chunk_pairs: prefill.to_vec(),
            decode_kv_lens: decode.to_vec(),
            total_kv_len: decode.iter().sum(),
        }
    }

    /// A verify group: each decode request presents `width` rows.
    fn verify_group(decode: &[u32], width: u32) -> ArchGroupInput {
        let decode_tokens = decode.len() as u32 * width;
        ArchGroupInput {
            batch_tokens: decode_tokens,
            prefill_tokens: 0,
            decode_tokens,
            prefill_chunk_pairs: Vec::new(),
            decode_kv_lens: decode.to_vec(),
            total_kv_len: decode.iter().map(|&kv| kv.saturating_sub(width)).sum(),
        }
    }

    fn input(groups: Vec<ArchGroupInput>) -> UnifiedArchInput {
        UnifiedArchInput {
            tokens_per_source_rank: groups.iter().map(|group| group.batch_tokens).collect(),
            groups,
        }
    }

    #[test]
    fn slot_count_follows_the_rank_count() {
        // GLM-5.3 NVFP4 shares this structure (model/catalog.yaml).
        for ep_size in [4_u16, 8] {
            let model = built("glm52_nvfp4", ep_size);
            assert_eq!(model.n_slots, expected_slot_count(ep_size), "EP{ep_size}");
            assert_eq!(model.cost_log_manifest().slots.len(), model.n_slots);
        }
        assert_eq!(expected_slot_count(4), 510);
    }

    #[test]
    fn layers_reduce_nothing_and_exchange_experts_through_two_collectives() {
        for ep_size in [4_u16, 8] {
            let slots = built("glm52_nvfp4", ep_size).cost_log_manifest().slots;
            for slot in &slots {
                assert!(
                    !slot.kind.starts_with("all_reduce") && !slot.name.contains("allreduce"),
                    "{} ({}) is a TP all-reduce",
                    slot.name,
                    slot.kind
                );
            }
            let kinds = |kind: &str| {
                slots
                    .iter()
                    .filter(|slot| slot.kind == kind)
                    .map(|slot| slot.name.as_str())
                    .collect::<BTreeSet<_>>()
            };
            let sections = [
                "sparse_initial_index_share",
                "sparse_cycle_full_index",
                "sparse_cycle_index_share",
            ];
            assert_eq!(
                kinds("moe_ep_quantized_all_gather"),
                sections
                    .iter()
                    .map(|section| format!("unified.body.{section}.moe.dispatch_ep_all_gather"))
                    .collect::<BTreeSet<_>>()
                    .iter()
                    .map(String::as_str)
                    .collect()
            );
            assert_eq!(kinds("moe_ep_reduce_scatter").len(), sections.len());
            // One routed-expert shard per rank in each sparse section.
            let routed = slots
                .iter()
                .filter(|slot| slot.kind == "nvfp4_fused_moe")
                .count();
            assert_eq!(routed, sections.len() * usize::from(ep_size));
        }
    }

    #[test]
    fn every_rank_runs_whole_attention_and_head_and_a_slice_of_experts() {
        for ep_size in [4_u16, 8] {
            let cfgs = configs("glm52_nvfp4", ep_size);
            let resolved = resolve_configs(&cfgs);
            for attention in [
                &resolved.local.dense_full_index_attention,
                &resolved.local.initial_shared_attention,
                &resolved.local.cycle_full_attention,
                &resolved.local.cycle_shared_attention,
            ] {
                assert_eq!(attention.main_rope.num_heads, 64);
            }
            assert_eq!(cfgs.local.dense_ffn.tp_size, 1);
            assert_eq!(cfgs.local.shared_expert.tp_size, 1);
            assert_eq!(cfgs.local.lm_head.n.get(), 154_880);
            assert!(cfgs.local.sparse_router.include_router_select);
            assert!(cfgs.local.nvfp4_moe.is_empty());
            assert_eq!(resolved.routed_experts.len(), usize::from(ep_size));
            for (position, rank) in resolved.routed_experts.iter().enumerate() {
                assert_eq!(rank.experts_per_device.get(), 256 / u32::from(ep_size));
                assert_eq!(rank.intermediate_per_rank.get(), 2_048);
                assert_eq!(rank.raw_cfg.folded_rank_position, position as u32);
                assert_eq!(
                    rank.fused_moe.backends,
                    vec!["flashinfer_trtllm_routed_sm100"]
                );
            }
            assert_eq!(cfgs.dispatch.num_gpus, u32::from(ep_size));
            assert_eq!(cfgs.dispatch.top_k.get(), 8);
            assert_eq!(cfgs.dispatch.activation_dtype, DType::Nvfp4E2m1);
            assert_eq!(cfgs.dispatch.max_total_tokens, 8_192 * u32::from(ep_size));
            assert_eq!(cfgs.combine.num_gpus, u32::from(ep_size));
            assert_eq!(cfgs.combine.dtype, DType::Bf16);
            assert_eq!(cfgs.combine.max_total_tokens, 8_192 * u32::from(ep_size));
        }
    }

    #[test]
    fn each_rank_holds_whole_tokens_of_kv_in_its_own_partition() {
        // A full-index layer caches latent + index key, an IndexShare layer the
        // latent only: 78 * 576 + 21 * 132.
        assert_eq!(decoder_layer_state_bytes_per_token(0), 708);
        assert_eq!(decoder_layer_state_bytes_per_token(3), 576);
        for ep_size in [4_u16, 8] {
            let model = built("glm52_nvfp4", ep_size);
            assert_eq!(
                model.total_kv_bytes_per_token(),
                u64::from(NUM_LAYERS) * 576 + 21 * 132
            );
            assert_eq!(model.total_kv_bytes_per_token(), 47_700);
            assert_eq!(model.gpus_per_replica(), ep_size);
            assert_eq!(model.num_attn_dp_groups(), ep_size);
            assert_eq!(model.num_attn_shards(), 1);
            // The chunked-prefill worker sizes each partition as
            // attn_kv_bytes * num_attn_shards / total_kv_bytes_per_token.
            let attn_kv_bytes: u64 = 45_140_539_392;
            let per_partition = attn_kv_bytes * u64::from(model.num_attn_shards())
                / model.total_kv_bytes_per_token();
            assert_eq!(per_partition, 946_342);
        }
    }

    #[test]
    fn hisparse_keeps_the_indexer_on_the_gpu_and_bills_one_copy_per_rank() {
        use crate::arch::hisparse::{HisparseConfig, HISPARSE_H2D_IO_KIND};
        let ep_size = 8;
        let model = built("glm52_nvfp4", ep_size)
            .with_hisparse(HisparseConfig::default(), ep_size)
            .unwrap();
        // 47,700 = 78 * 576 latent + 21 * 132 indexer key: only the keys stay.
        assert_eq!(model.total_kv_bytes_per_token(), 21 * 132);
        assert_eq!(model.total_kv_bytes_per_token(), 2_772);
        let offload = model.kv_host_offload().unwrap();
        assert_eq!(offload.host_bytes_per_token, 78 * 576);
        assert_eq!(offload.device_bytes_per_request, 4096 * 44_928);
        assert_eq!(offload.host_bytes_per_partition, 250_000_000_000);
        // One analytic leaf per rank, after every kernel slot.
        assert_eq!(
            model.n_slots,
            expected_slot_count(ep_size) + usize::from(ep_size)
        );
        let slots = model.cost_log_manifest().slots;
        let copies: Vec<_> = slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.kind == HISPARSE_H2D_IO_KIND)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(
            copies,
            (expected_slot_count(ep_size)..model.n_slots).collect::<Vec<_>>()
        );
        // Per-rank work from each rank's decode lengths at the verify width.
        let mut groups = vec![group(&[], &[]); usize::from(ep_size)];
        groups[0] = group(&[], &[100_000, 2_000]);
        groups[3] = group(&[], &[120_000]);
        let batch = model.normalize(&input(groups));
        assert_eq!(batch.hisparse.len(), usize::from(ep_size));
        assert_eq!(batch.hisparse[0].offloaded_requests, 1);
        assert_eq!(batch.hisparse[0].context_tokens, 102_000);
        assert!((batch.hisparse[0].misses_per_layer - 0.134 * 2048.0).abs() < 1e-6);
        assert_eq!(batch.hisparse[3].offloaded_requests, 1);
        assert_eq!(batch.hisparse[1].decode_requests, 0);
        // Off: no slot, no work, the GPU-resident accounting.
        let plain = built("glm52_nvfp4", ep_size);
        assert!(plain.kv_host_offload().is_none());
        assert!(plain
            .normalize(&input(vec![group(&[], &[100_000]); 8]))
            .hisparse
            .is_empty());
    }

    #[test]
    fn location_map_matches_every_unique_noncommunication_manifest_location() {
        let map_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("model/work/location_maps/glm52_vllm_nvfp4_dp_attn_dsa_moe_unified.json");
        let map: serde_json::Value =
            serde_json::from_slice(&std::fs::read(map_path).unwrap()).unwrap();
        assert_eq!(map["arch_types"], serde_json::json!([ARCH_KIND]));
        let mapped: BTreeSet<String> = map["locations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["location"].as_str().unwrap().to_string())
            .collect();
        for ep_size in [4_u16, 8] {
            let locations: BTreeSet<String> = built("glm52_nvfp4", ep_size)
                .cost_log_manifest()
                .slots
                .into_iter()
                .filter(|slot| !slot.kind.starts_with("moe_ep_"))
                .map(|slot| slot.name)
                .collect();
            // The TP/EP graph's 114 locations plus the three routers' select.
            assert_eq!(locations.len(), 117);
            assert_eq!(mapped, locations);
        }
    }

    #[test]
    fn idle_ranks_join_the_exchange_with_one_dummy_token() {
        let busy = normalize_input(
            &input(vec![
                group(&[(0, 4_096)], &[]),
                group(&[], &[]),
                group(&[(1_024, 2_048)], &[512, 900]),
                group(&[], &[]),
            ]),
            4,
            1,
            131_072,
            &[],
        )
        .unwrap();
        assert_eq!(busy.collective_tokens, vec![4_096, 1, 2_050, 1]);
        assert_eq!(busy.rows, vec![4_096, 0, 2_050, 0]);
        assert_eq!(busy.gathered_tokens, 6_148);
        assert_eq!(busy.groups[2].batch_tokens, 2_050);

        let idle = normalize_input(&input(vec![group(&[], &[]); 4]), 4, 1, 131_072, &[]).unwrap();
        assert_eq!(idle.collective_tokens, vec![0; 4]);
        assert_eq!(idle.gathered_tokens, 0);
    }

    #[test]
    fn graph_steps_pad_every_rank_to_the_busiest_ranks_graph() {
        let sizes = vllm_capture_sizes(2_048);
        assert_eq!(sizes.len(), 147);
        assert_eq!(sizes[..5], [1, 2, 4, 8, 16]);
        assert_eq!(sizes.last(), Some(&2_048));
        let step = |groups: Vec<ArchGroupInput>| {
            normalize_input(&input(groups), 4, 1, 131_072, &sizes).unwrap()
        };

        // One 1000-token prompt: the busy rank's graph is 1008, and the three
        // idle ranks' dummy batches pad to it too.
        let one = step(vec![
            group(&[(0, 1_000)], &[]),
            group(&[], &[]),
            group(&[], &[]),
            group(&[], &[]),
        ]);
        assert_eq!(one.rows, vec![1_008; 4]);
        assert_eq!(one.collective_tokens, vec![1_008; 4]);
        assert_eq!(one.gathered_tokens, 4_032);
        assert_eq!(one.groups[0].batch_tokens, 1_000);
        assert_eq!(one.groups[1].batch_tokens, 0);

        // Small decode batches pad to the busiest rank's bucket.
        let decode = step(vec![
            group(&[], &[100, 200, 300]),
            group(&[], &[400]),
            group(&[], &[]),
            group(&[], &[50; 9]),
        ]);
        assert_eq!(decode.rows, vec![16; 4]);
        assert_eq!(decode.gathered_tokens, 64);

        // The largest captured size itself still replays a graph.
        let edge = step(vec![
            group(&[(0, 2_048)], &[]),
            group(&[(0, 8)], &[]),
            group(&[], &[]),
            group(&[], &[]),
        ]);
        assert_eq!(edge.collective_tokens, vec![2_048; 4]);

        // Past the largest graph the busy rank runs eager, so every rank does
        // and the counts stay ragged.
        let eager = step(vec![
            group(&[(0, 2_049)], &[]),
            group(&[(0, 8)], &[]),
            group(&[], &[]),
            group(&[], &[]),
        ]);
        assert_eq!(eager.rows, vec![2_049, 8, 0, 0]);
        assert_eq!(eager.collective_tokens, vec![2_049, 8, 1, 1]);
        assert_eq!(eager.gathered_tokens, 2_059);

        let idle = step(vec![group(&[], &[]); 4]);
        assert_eq!(idle.rows, vec![0; 4]);
        assert_eq!(idle.gathered_tokens, 0);
    }

    /// Catches a verify target that bills a request's `k + 1` rows as `k + 1`
    /// requests, scores only one logit per request, or pads the request count
    /// rather than its rows.
    #[test]
    fn a_verify_step_lowers_each_request_once_with_its_rows() {
        let sizes = vllm_capture_sizes(2_048);
        let batch = normalize_input(
            &input(vec![
                verify_group(&[100, 2_000, 300], 4),
                verify_group(&[50], 4),
                group(&[], &[]),
                verify_group(&[], 4),
            ]),
            4,
            4,
            131_072,
            &sizes,
        )
        .unwrap();
        let busy = &batch.groups[0];
        assert_eq!(busy.batch_tokens, 12);
        assert_eq!(busy.request_count, 3);
        assert_eq!(busy.logits_rows, 12);
        assert_eq!(busy.endpoint_context_lens, vec![100, 2_000, 300]);
        let decode = busy.attention_input.decode.as_ref().unwrap();
        assert_eq!(busy.attention_input.num_new_tokens, 12);
        assert_eq!(decode.batch_size, 3);
        assert_eq!(decode.context_len, 2_000);
        // Twelve rows replay the 16-token graph on every rank.
        assert_eq!(batch.rows, vec![16; 4]);
        assert_eq!(batch.collective_tokens, vec![16; 4]);
        assert_eq!(batch.gathered_tokens, 64);

        let ragged = |group: ArchGroupInput| {
            normalize_input(&input(vec![group; 4]), 4, 4, 131_072, &sizes).is_err()
        };
        // A decode request must present exactly the verify width ...
        assert!(ragged(group(&[], &[100, 200])));
        // ... at a context that holds it.
        assert!(ragged(verify_group(&[3], 4)));
    }

    #[test]
    fn a_verify_target_compiles_the_same_leaves_at_its_width() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let cfgs = build_verify_configs(
            &model_cfg("glm52_nvfp4"),
            &parallel(8),
            &ExpertDemand::popularity(&RoutingDistribution::uniform(256), 4),
            false,
            3,
        )
        .unwrap();
        assert_eq!(cfgs.local.speculative_draft_tokens, Some(3));
        for attention in [
            &cfgs.local.dense_full_index_attention,
            &cfgs.local.initial_shared_attention,
            &cfgs.local.cycle_full_attention,
            &cfgs.local.cycle_shared_attention,
        ] {
            assert_eq!(attention.decode_next_n, 4);
        }
        assert!(cfgs.local.mtp_attention.is_none());
        let model = build("unified".to_string(), resolve_configs(&cfgs), &bridge).unwrap();
        assert_eq!(model.verify_width(), 4);
        assert_eq!(model.n_slots, expected_slot_count(8));
        assert_eq!(model.total_kv_bytes_per_token(), 47_700);
        assert_eq!(built("glm52_nvfp4", 8).verify_width(), 1);
    }

    fn analytic(ep_size: u16, nvl_num_gpu: u16, kind: EpCommKind) -> Glm52VllmNvfp4DpAttnParallel {
        Glm52VllmNvfp4DpAttnParallel {
            nvl_num_gpu,
            ep_comm: EpCommConfig {
                kind,
                model: EpCommModel::Analytic,
                ..EpCommConfig::default()
            },
            ..parallel(ep_size)
        }
    }

    fn built_with(p: &Glm52VllmNvfp4DpAttnParallel) -> Glm52VllmNvfp4DpAttnDsaMoeModel {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let cfgs = build_configs(&model_cfg("glm52_nvfp4"), p, &demand(), false).unwrap();
        build("unified".to_string(), resolve_configs(&cfgs), &bridge).unwrap()
    }

    #[test]
    fn the_analytic_exchange_spans_an_nvl72_domain() {
        for ep_size in [8_u16, 16, 64] {
            // All-gather / reduce-scatter: the two collectives become analytic
            // leaves under the same names, nothing else changes.
            let model = built_with(&analytic(ep_size, 72, EpCommKind::AllgatherReducescatter));
            assert_eq!(model.n_slots, expected_slot_count(ep_size), "EP{ep_size}");
            let slots = model.cost_log_manifest().slots;
            let leaves: Vec<_> = slots
                .iter()
                .filter(|slot| slot.kind == crate::arch::ep_comm::EP_COMM_ANALYTIC_KIND)
                .collect();
            assert_eq!(leaves.len(), 6);
            assert!(slots
                .iter()
                .all(|slot| !slot.kind.starts_with("moe_ep_")));
            // All-to-all: a prepare and a second dispatch call (the block
            // scales) per section and three elementwise leaves per rank per
            // section on top, named as the measured EP8 lookups.
            let model = built_with(&analytic(ep_size, 72, EpCommKind::Alltoall));
            assert_eq!(
                model.n_slots,
                expected_slot_count(ep_size) + 3 * (2 + 3 * usize::from(ep_size)),
                "EP{ep_size}"
            );
            let analytic_leaves: Vec<_> = model
                .cost_log_manifest()
                .slots
                .into_iter()
                .filter(|slot| slot.kind == crate::arch::ep_comm::EP_COMM_ANALYTIC_KIND)
                .map(|slot| slot.name.rsplit('.').next().unwrap().to_string())
                .collect();
            assert_eq!(
                analytic_leaves,
                [
                    "dispatch_alltoall_prepare",
                    "dispatch_alltoall",
                    "dispatch_alltoall_scales",
                    "combine_alltoall"
                ]
                .repeat(3)
            );
            assert_eq!(model.gpus_per_replica(), ep_size);
            let resolved = resolve_configs(
                &build_configs(
                    &model_cfg("glm52_nvfp4"),
                    &analytic(ep_size, 72, EpCommKind::Alltoall),
                    &demand(),
                    false,
                )
                .unwrap(),
            );
            for rank in &resolved.routed_experts {
                assert_eq!(rank.experts_per_device.get(), 256 / u32::from(ep_size));
            }
        }
    }

    fn measured_alltoall(ep_size: u16, nvl_num_gpu: u16) -> Glm52VllmNvfp4DpAttnParallel {
        Glm52VllmNvfp4DpAttnParallel {
            nvl_num_gpu,
            ep_comm: EpCommConfig {
                kind: EpCommKind::Alltoall,
                model: EpCommModel::Measured,
                ..EpCommConfig::default()
            },
            ..parallel(ep_size)
        }
    }

    #[test]
    fn the_measured_alltoall_looks_up_the_ep8_mnnvl_rows() {
        let p = measured_alltoall(8, 72);
        let cfgs = build_configs(&model_cfg("glm52_nvfp4"), &p, &demand(), false).unwrap();
        let kernels = cfgs.measured_alltoall.as_ref().unwrap();
        // NVFP4 activations and their block scales are two lookups; the combine
        // moves bf16 rows.
        for (kernel, bytes, direction) in [
            (&kernels.dispatch, 3_072, MoeAlltoallDirection::Dispatch),
            (&kernels.dispatch_scales, 384, MoeAlltoallDirection::Dispatch),
            (&kernels.combine, 12_288, MoeAlltoallDirection::Combine),
        ] {
            assert_eq!(kernel.hidden_bytes.get(), bytes);
            assert_eq!(kernel.direction, direction);
            assert_eq!(kernel.backends, ["flashinfer_mnnvl"]);
            assert_eq!(
                (kernel.ep_size, kernel.top_k, kernel.slot_count),
                (8, 8, 256)
            );
            assert_eq!(kernel.gpu_name, "NVIDIA B200");
        }
        assert_eq!(
            (kernels.prepare.ep_size, kernels.prepare.top_k, kernels.prepare.slot_count),
            (8, 8, 256)
        );

        // Three sections, each with a prepare, two dispatch calls and a combine
        // in place of the all-gather and reduce-scatter, plus the three
        // elementwise leaves per rank.
        let model = built_with(&p);
        assert_eq!(model.n_slots, expected_slot_count(8) + 3 * (2 + 3 * 8));
        let slots = model.cost_log_manifest().slots;
        let named = |kind: &str| -> Vec<String> {
            slots
                .iter()
                .filter(|slot| slot.kind == kind)
                .map(|slot| slot.name.rsplit('.').next().unwrap().to_string())
                .collect()
        };
        assert_eq!(named("moe_alltoall_prepare"), ["dispatch_alltoall_prepare"; 3]);
        assert_eq!(
            named("moe_alltoall"),
            ["dispatch_alltoall", "dispatch_alltoall_scales", "combine_alltoall"].repeat(3)
        );
        assert!(named(crate::arch::ep_comm::EP_COMM_ANALYTIC_KIND).is_empty());
        assert!(slots.iter().all(|slot| !slot.kind.starts_with("moe_ep_")));

        // The lookups sit at the analytic model's busiest-rank rows.
        let comm = cfgs.analytic.as_ref().unwrap();
        let tokens = [40, 64, 8, 1, 1, 30, 64, 12];
        let rows = comm.alltoall_rows(&tokens);
        let step = comm.step(&tokens);
        assert_eq!(rows.send, step.dispatch.max_send_rows);
        assert_eq!(rows.recv, step.dispatch.max_recv_rows);
        let inputs = MeasuredAlltoallInputs::new(&rows);
        assert_eq!(inputs.prepare.tokens_per_rank, 64);
        assert_eq!(inputs.dispatch.max_send_rows, rows.send.round() as u32);
        assert_eq!(inputs.dispatch.max_recv_rows, rows.recv.round() as u32);
        assert_eq!(inputs.combine.max_send_rows, inputs.dispatch.max_recv_rows);
        assert_eq!(inputs.combine.max_recv_rows, inputs.dispatch.max_send_rows);
        // 64 tokens reach ~5.3 of 8 ranks each; the busiest receiver gets at
        // least the mean share of all 220 tokens.
        assert!((330..=345).contains(&inputs.dispatch.max_send_rows));
        assert!(inputs.dispatch.max_recv_rows >= (220.0 * comm.uniform_hit).floor() as u32);
        // A lone token still moves whole rows; nothing moves nothing.
        let one = MeasuredAlltoallInputs::new(&comm.alltoall_rows(&[1, 0, 0, 0, 0, 0, 0, 0]));
        assert!(one.dispatch.max_send_rows >= 1 && one.dispatch.max_recv_rows >= 1);
        let none = MeasuredAlltoallInputs::new(&comm.alltoall_rows(&[0; 8]));
        assert_eq!(
            (
                none.prepare.tokens_per_rank,
                none.dispatch.max_send_rows,
                none.dispatch.max_recv_rows
            ),
            (0, 0, 0)
        );
    }

    #[test]
    fn measured_exchange_stays_inside_the_profiled_groups() {
        let model = model_cfg("glm52_nvfp4");
        // Measured all-to-all: EP8 rows only, inside one NVLink domain.
        for (ep_size, nvl) in [(8_u16, 8_u16), (8, 72)] {
            let p = measured_alltoall(ep_size, nvl);
            assert!(build_configs(&model, &p, &demand(), false).is_ok());
        }
        for (ep_size, nvl) in [(4_u16, 4_u16), (16, 72), (64, 72), (8, 4)] {
            let error = build_configs(&model, &measured_alltoall(ep_size, nvl), &demand(), false)
                .err()
                .unwrap_or_else(|| panic!("EP{ep_size} nvl {nvl} must fail"));
            assert!(error.to_string().contains("ep_size"), "{error}");
        }
        // Analytic: the group must divide the experts and fit the domain.
        for (ep_size, nvl) in [(1_u16, 72_u16), (3, 72), (96, 128), (64, 32)] {
            let p = analytic(ep_size, nvl, EpCommKind::Alltoall);
            assert!(build_configs(&model, &p, &demand(), false).is_err(), "{ep_size}/{nvl}");
        }
    }

    #[test]
    fn input_needs_one_consistent_group_per_rank() {
        let model = built("glm52_nvfp4", 4);
        let four = input(vec![group(&[(0, 8)], &[]); 4]);
        assert!(model.check_input(&four).is_ok());
        assert!(model
            .check_input(&input(vec![group(&[(0, 8)], &[]); 1]))
            .is_err());
        let mut skewed = four.clone();
        skewed.tokens_per_source_rank = vec![8, 8, 8, 9];
        assert!(model.check_input(&skewed).is_err());
        let too_long = input(vec![group(&[(131_000, 4_096)], &[]); 4]);
        assert!(model.check_input(&too_long).is_err());
    }

    #[test]
    fn an_idle_iteration_fills_every_slot_with_zero() {
        let model = built("glm52_nvfp4", 8);
        let mut slots = Vec::new();
        let mut scratch = Vec::new();
        let mut inputs = Vec::new();
        let total = model.eval_iter_with_inputs(
            &input(vec![group(&[], &[]); 8]),
            &mut slots,
            &mut scratch,
            &mut inputs,
        );
        assert_eq!(slots.len(), model.n_slots);
        assert_eq!(inputs.len(), model.n_slots);
        assert_eq!(total.m.time_ms, 0.0);
    }

    #[test]
    fn invalid_groups_fail_closed() {
        let model = model_cfg("glm52_nvfp4");
        for (ep_size, nvl) in [(1_u16, 1_u16), (3, 3), (16, 16), (4, 8), (8, 4)] {
            let mut p = parallel(ep_size);
            p.nvl_num_gpu = nvl;
            assert!(build_configs(&model, &p, &demand(), false).is_err());
        }
        let mut too_long = parallel(4);
        too_long.max_model_len = CHECKPOINT_MAX_CONTEXT + 1;
        assert!(build_configs(&model, &too_long, &demand(), false).is_err());
        assert!(build_configs(&model, &parallel(4), &demand(), true).is_err());
        for sizes in [vec![0, 8], vec![16, 8], vec![8, 8]] {
            let mut p = parallel(4);
            p.cudagraph_capture_sizes = sizes;
            assert!(build_configs(&model, &p, &demand(), false).is_err());
        }
    }
}
