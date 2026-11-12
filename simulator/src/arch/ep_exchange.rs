//! The expert exchange an arch's `ep_comm` config selects in place of the
//! measured all-gather / reduce-scatter, as the cost-tree leaves around the
//! routed experts of one MoE layer.
//!
//! Two exchanges live here, both priced by `arch::ep_comm`:
//!
//! * the **analytic all-gather / reduce-scatter** (`ep_comm_model: analytic`):
//!   one leaf each, under the names the arch gives its measured collectives;
//! * flashinfer's **two-sided MNNVL all-to-all** (`ep_comm: alltoall`,
//!   vLLM `--all2all-backend flashinfer_nvlink_two_sided`): the metadata
//!   prepare, the NVFP4 activations and their block scales as two `moe_comm`
//!   calls, the receiver's scale swizzle, then after the experts the zeroed
//!   `tokens x top_k` bf16 staging fill, the bf16 combine `moe_comm` and the
//!   top-k sum. The prepare and the three transfers are the profiled EP8
//!   `moe_alltoall_prepare` / `moe_alltoall` rows (`ep_comm_model: measured`)
//!   looked up at `arch::ep_comm`'s busiest-sender / busiest-receiver rows, or
//!   that module's analytic legs; the swizzle, fill and sum are measured
//!   elementwise leaves either way.
//!
//! * the **fast exchanges** (`ep_comm: fast_allgather | fused_allgather |
//!   fast_sparse`): one dispatch leg and one combine leg each, no prepare.
//!   With `ep_comm_model: analytic` both legs are `arch::ep_comm`'s fit,
//!   flagged `extrapolated`. With `ep_comm_model: measured` (MNNVL, EP8):
//!   `fast_allgather` + `ep_fast_combine: reduce_scatter` reads the
//!   fixed-pattern exchange kernels' rows (`moe_ep_quantized_all_gather` and
//!   `moe_ep_reduce_scatter`, backend `mnnvl_lamport`; the owner's sum is
//!   inside the combine kernel, so no separate sum leaf); `fused_allgather`
//!   reads `moe_ep_quantize_and_all_gather` (the arch's dispatch quantization
//!   is inside it, see [`EpExchange::folds_dispatch_quant`]) and
//!   `moe_ep_reduce_scatter` `mnnvl_pull` (the owner reads every rank's
//!   partials from fabric memory and sums them; measured as its own launch,
//!   not fused into the next op); `fast_allgather` + `ep_fast_combine:
//!   all_reduce` takes the combine from the backend's profiled all-reduce
//!   (`all_reduce` nvshmem at `T x row` bytes, or `all_reduce_fusion`
//!   flashinfer_mnnvl at `T` tokens, which extrapolates past vLLM's 128-token
//!   TP8 cap) and its dispatch from the fit; `fast_sparse` has no kernel and
//!   stays on the fit.
//!
//! Every row width comes from the model's hidden size, top-k and routed
//! expert count ([`EpExchangeGeometry`]), so one exchange serves every
//! checkpoint. The measured all-gather / reduce-scatter, the default, is not
//! here: each arch keeps its own kernels and leaf names for it, so the default
//! cost tree is the arch's own.

use std::sync::Arc;

use crate::arch::ep_comm::{
    push_extrapolated_leg, push_leg, AlltoallRows, AnalyticEpComm, EpCommConfig, EpCommKind,
    EpCommModel, EpCommStep, FastBackend, FastCombine,
};
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    AllReduceFusionKernel, AllReduceFusionKernelConfig, AllReduceFusionKernelInput,
    AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput,
};
use crate::common::Fabric;
use crate::op::Op;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, MoeAlltoallDirection,
    MoeAlltoallKernel, MoeAlltoallKernelConfig, MoeAlltoallKernelInput, MoeAlltoallPrepareKernel,
    MoeAlltoallPrepareKernelConfig, MoeAlltoallPrepareKernelInput, MoeEpCollectiveKernelInput,
    MoeEpQuantizeAndAllGatherKernel, MoeEpQuantizeAndAllGatherKernelConfig,
    MoeEpQuantizedAllGatherKernel, MoeEpQuantizedAllGatherKernelConfig, MoeEpReduceScatterKernel,
    MoeEpReduceScatterKernelConfig,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Evaluator, LeafMetrics, PerfApiBridge, Probe, SlotInput,
};

/// FlashInfer 0.6's `trtllm_alltoall` (`MnnvlMoe`), the `moe_comm` kernel
/// vLLM's `--all2all-backend flashinfer_nvlink_two_sided` called before
/// FlashInfer 0.7 removed it; its rows cover 384 / 3,072 / 12,288-byte rows at
/// 256 slots.
pub const MNNVL_ALLTOALL_BACKENDS: &[&str] = &["flashinfer_mnnvl"];
/// FlashInfer 0.7's `moe_ep` `NVLinkTwoSidedAlltoAll`, the same `moe_comm`
/// kernel under today's vLLM path; its rows cover 256 / 2,048 / 2,368 /
/// 8,192-byte rows at 288 slots. A backend list names one of the two: the
/// prebuild fills every listed backend's grid, so a second backend without
/// rows at the config would demand a GPU fill.
pub const NVLINK_TWO_SIDED_ALLTOALL_BACKENDS: &[&str] = &["flashinfer_nvlink_two_sided"];
/// The only group the `moe_alltoall` / `moe_alltoall_prepare` rows span.
pub const MEASURED_ALLTOALL_EP_SIZE: u16 = 8;
const ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
/// The fixed-pattern MNNVL exchange kernels
/// (`profiling/runners/comm/csrc/mnnvl_fixed_exchange.cuh`): direct stores
/// into every peer's landing rows with Lamport-style arrival.
pub const FIXED_EXCHANGE_PUSH_BACKENDS: &[&str] = &["mnnvl_lamport"];
/// The same exchange's reader-side combine: the owner reads every rank's
/// partials from fabric memory and sums them.
pub const FIXED_EXCHANGE_PULL_BACKENDS: &[&str] = &["mnnvl_pull"];
/// The only group the fixed-pattern exchange rows span.
pub const MEASURED_FIXED_EXCHANGE_EP_SIZE: u16 = 8;

/// The model dimensions the exchange's rows are sized from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpExchangeGeometry {
    pub hidden: u32,
    pub top_k: u32,
    /// Routed experts; one MNNVL slot each (no EPLB redundancy).
    pub num_experts: u32,
    /// Largest token count summed over the EP group (the arch's per-rank
    /// batch budget times `ep_size`): the measured exchange grids' total axis.
    pub max_total_tokens: u32,
}

/// The fixed-pattern exchange's profiled dispatch.
#[derive(Clone, Debug)]
pub enum FixedDispatchConfig {
    /// `fast_allgather`: the already-quantized rows, gathered.
    Gather(MoeEpQuantizedAllGatherKernelConfig),
    /// `fused_allgather`: bf16 rows quantized inside the gather.
    QuantizeAndGather(MoeEpQuantizeAndAllGatherKernelConfig),
}

/// The fixed-pattern MNNVL exchange's profiled dispatch and combine.
#[derive(Clone, Debug)]
pub struct MeasuredFixedExchange {
    pub dispatch: FixedDispatchConfig,
    /// `mnnvl_lamport` (pushed partials) or `mnnvl_pull` (read by the owner).
    pub combine: MoeEpReduceScatterKernelConfig,
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

/// What an `ep_comm` config adds to an arch's configs. Every field is `None`
/// on the default, measured all-gather / reduce-scatter.
#[derive(Clone, Debug, Default)]
pub struct EpExchangeConfigs {
    /// The analytic exchange, and the all-to-all's row model on the measured
    /// path.
    pub analytic: Option<AnalyticEpComm>,
    /// The all-to-all's elementwise leaves; `None` unless `ep_comm` is
    /// `alltoall`.
    pub alltoall: Option<AlltoallElementwise>,
    /// The measured all-to-all's kernels; `None` unless `ep_comm` is
    /// `alltoall` and `ep_comm_model` is `measured`.
    pub measured_alltoall: Option<MeasuredAlltoall>,
    /// A fast exchange's measured all-reduce combine; `None` unless `ep_comm`
    /// is `fast_allgather`, `ep_fast_combine` `all_reduce` and `ep_comm_model`
    /// `measured`.
    pub measured_fast_combine: Option<MeasuredFastCombine>,
    /// The fixed-pattern exchange kernels; `None` unless `ep_comm_model` is
    /// `measured` and `ep_comm` is `fused_allgather`, or `fast_allgather`
    /// with `ep_fast_combine: reduce_scatter`.
    pub measured_fixed: Option<MeasuredFixedExchange>,
    /// A fast reduce-scatter combine's receiver sum of the `ep_size` partial
    /// rows per token (dense slots, bf16), the all-to-all `combine_reduce`
    /// shape at EP8 top-8; `None` for the all-reduce (the sum is in the
    /// collective) and `fused_allgather` (fused into the next op).
    pub fast_combine_sum: Option<ElementwiseKernelConfig>,
}

/// The profiled all-reduce a fast exchange's combine runs as.
#[derive(Clone, Debug)]
pub enum MeasuredFastCombine {
    /// `all_reduce` nvshmem over `T x row` bytes.
    Nvshmem(AllReduceKernelConfig),
    /// `all_reduce_fusion` flashinfer_mnnvl over `T` bf16 rows.
    Mnnvl(AllReduceFusionKernelConfig),
}

impl EpExchangeConfigs {
    /// `demand` is the routed demand folded over `ep_size` ranks, as the
    /// arch's experts see it. Checks only what every arch shares: the params
    /// and the EP8 extent of the measured all-to-all rows. The arch checks its
    /// own group (domain size, measured all-gather group sizes).
    /// `alltoall_backends` is the profiled `moe_alltoall` backend whose rows
    /// cover the arch's row widths and slot count.
    pub fn new(
        ep_comm: &EpCommConfig,
        ep_size: u16,
        gpu_name: &str,
        geometry: EpExchangeGeometry,
        alltoall_backends: &[&'static str],
        demand: &ExpertDemand,
    ) -> Result<Self, String> {
        ep_comm.validate()?;
        let alltoall_kind = ep_comm.kind == EpCommKind::Alltoall;
        let measured_alltoall_kind = alltoall_kind && ep_comm.model == EpCommModel::Measured;
        if measured_alltoall_kind && ep_size != MEASURED_ALLTOALL_EP_SIZE {
            return Err(format!(
                "ep_size {ep_size} has no measured all-to-all rows: ep_comm alltoall with \
                 ep_comm_model measured needs ep_size {MEASURED_ALLTOALL_EP_SIZE} \
                 (ep_comm_model analytic goes further)"
            ));
        }
        if ep_comm.is_default_measured() {
            return Ok(Self::default());
        }
        let measured_fast = ep_comm.kind == EpCommKind::FastAllgather
            && ep_comm.model == EpCommModel::Measured
            && ep_comm.fast.combine == FastCombine::AllReduce;
        let measured_fast_combine = if measured_fast {
            if ep_size != MEASURED_ALLTOALL_EP_SIZE {
                return Err(format!(
                    "ep_size {ep_size}: the measured fast all-reduce combine has B200 rows at \
                     {MEASURED_ALLTOALL_EP_SIZE} GPUs only (ep_comm_model analytic goes further)"
                ));
            }
            Some(match ep_comm.fast.backend {
                FastBackend::Nvshmem => MeasuredFastCombine::Nvshmem(AllReduceKernelConfig {
                    backends: vec!["nvshmem"],
                    gpu_name: gpu_name.to_string(),
                    num_gpus: u32::from(ep_size),
                    fabric: Fabric::Nvlink,
                }),
                FastBackend::Mnnvl => {
                    if ep_comm
                        .fast
                        .combine_row_bytes
                        .is_some_and(|bytes| bytes != 2 * geometry.hidden)
                    {
                        return Err("the measured MNNVL all-reduce combine is bf16 rows; \
                                    ep_combine_row_bytes needs ep_comm_model analytic \
                                    or the nvshmem backend"
                            .to_string());
                    }
                    MeasuredFastCombine::Mnnvl(AllReduceFusionKernelConfig {
                        backends: vec!["flashinfer_mnnvl"],
                        gpu_name: gpu_name.to_string(),
                        num_gpus: u32::from(ep_size),
                        hidden_dim: geometry.hidden,
                        dtype: DType::Bf16,
                        fabric: Fabric::Nvlink,
                        fused_token_limit: None,
                    })
                }
                FastBackend::Lamport => {
                    return Err("no measured Lamport all-reduce of the bare partial sums \
                                (its rows fuse a short conv and RMSNorm); use \
                                ep_comm_model analytic"
                        .to_string())
                }
            })
        } else {
            None
        };
        let EpExchangeGeometry {
            hidden,
            top_k,
            num_experts,
            max_total_tokens,
        } = geometry;
        let measured_fixed_kind = ep_comm.model == EpCommModel::Measured
            && match ep_comm.kind {
                EpCommKind::FusedAllgather => true,
                EpCommKind::FastAllgather => ep_comm.fast.combine == FastCombine::ReduceScatter,
                _ => false,
            };
        let measured_fixed = if measured_fixed_kind {
            if ep_size != MEASURED_FIXED_EXCHANGE_EP_SIZE {
                return Err(format!(
                    "ep_size {ep_size}: the fixed-pattern exchange has B200 rows at \
                     {MEASURED_FIXED_EXCHANGE_EP_SIZE} GPUs only (ep_comm_model analytic goes \
                     further)"
                ));
            }
            if ep_comm.fast.backend != FastBackend::Mnnvl {
                return Err(format!(
                    "the measured fixed-pattern exchange runs on MNNVL fabric memory: \
                     ep_fast_backend mnnvl, not {:?} (or ep_comm_model analytic)",
                    ep_comm.fast.backend
                ));
            }
            if ep_comm.kind == EpCommKind::FusedAllgather
                && ep_comm.fast.combine != FastCombine::ReduceScatter
            {
                return Err("fused_allgather with ep_comm_model measured reads the owner-side \
                            reduce-scatter rows: ep_fast_combine reduce_scatter"
                    .to_string());
            }
            if ep_comm
                .fast
                .combine_row_bytes
                .is_some_and(|bytes| bytes != 2 * hidden)
            {
                return Err("the measured fixed-pattern combine moves bf16 rows; \
                            ep_combine_row_bytes needs ep_comm_model analytic"
                    .to_string());
            }
            let fused = ep_comm.kind == EpCommKind::FusedAllgather;
            let dispatch = if fused {
                FixedDispatchConfig::QuantizeAndGather(MoeEpQuantizeAndAllGatherKernelConfig {
                    backends: FIXED_EXCHANGE_PUSH_BACKENDS.to_vec(),
                    gpu_name: gpu_name.to_string(),
                    num_gpus: u32::from(ep_size),
                    hidden_size: hidden.into(),
                    top_k: top_k.into(),
                    activation_dtype: DType::Nvfp4E2m1,
                    fabric: "nvlink".into(),
                    max_total_tokens,
                })
            } else {
                FixedDispatchConfig::Gather(MoeEpQuantizedAllGatherKernelConfig {
                    backends: FIXED_EXCHANGE_PUSH_BACKENDS.to_vec(),
                    gpu_name: gpu_name.to_string(),
                    num_gpus: u32::from(ep_size),
                    hidden_size: hidden.into(),
                    top_k: top_k.into(),
                    activation_dtype: DType::Nvfp4E2m1,
                    fabric: "nvlink".into(),
                    max_total_tokens,
                })
            };
            Some(MeasuredFixedExchange {
                dispatch,
                combine: MoeEpReduceScatterKernelConfig {
                    backends: if fused {
                        FIXED_EXCHANGE_PULL_BACKENDS.to_vec()
                    } else {
                        FIXED_EXCHANGE_PUSH_BACKENDS.to_vec()
                    },
                    gpu_name: gpu_name.to_string(),
                    num_gpus: u32::from(ep_size),
                    hidden_size: hidden.into(),
                    dtype: DType::Bf16,
                    fabric: "nvlink".into(),
                    max_total_tokens,
                },
            })
        } else {
            None
        };
        // The measured all-to-all still takes its rows from the analytic model.
        let analytic =
            AnalyticEpComm::new(ep_comm.clone(), ep_size, hidden, num_experts, top_k, demand)
                .map_err(|error| format!("analytic ep_comm: {error:#}"))?;
        let elementwise = |input: u32, output: u32| ElementwiseKernelConfig {
            backends: ELEMENTWISE_BACKENDS.to_vec(),
            gpu_name: gpu_name.to_string(),
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
            backends: alltoall_backends.to_vec(),
            gpu_name: gpu_name.to_string(),
            ep_size: u32::from(ep_size),
            top_k,
            slot_count: num_experts,
            hidden_bytes: hidden_bytes.into(),
            direction,
            fabric: Fabric::Nvlink,
        };
        let measured_alltoall = measured_alltoall_kind.then(|| MeasuredAlltoall {
            prepare: MoeAlltoallPrepareKernelConfig {
                backends: alltoall_backends.to_vec(),
                gpu_name: gpu_name.to_string(),
                ep_size: u32::from(ep_size),
                top_k,
                slot_count: num_experts,
                fabric: Fabric::Nvlink,
            },
            dispatch: alltoall_kernel(hidden / 2, MoeAlltoallDirection::Dispatch),
            dispatch_scales: alltoall_kernel(hidden / 16, MoeAlltoallDirection::Dispatch),
            combine: alltoall_kernel(bf16_row, MoeAlltoallDirection::Combine),
        });
        // The measured fixed-pattern combine sums inside its kernel.
        let fast_combine_sum = (ep_comm.kind.is_fast()
            && measured_fixed.is_none()
            && ep_comm.kind != EpCommKind::FusedAllgather
            && (ep_comm.kind == EpCommKind::FastSparse
                || ep_comm.fast.combine == FastCombine::ReduceScatter))
            .then(|| elementwise(u32::from(ep_size) * bf16_row, bf16_row));
        Ok(Self {
            analytic: Some(analytic),
            alltoall,
            measured_alltoall,
            measured_fast_combine,
            measured_fixed,
            fast_combine_sum,
        })
    }

    /// The default: the arch's own measured all-gather / reduce-scatter.
    pub fn is_measured_allgather(&self) -> bool {
        self.analytic.is_none()
    }
}

/// The leaf names an arch gives its all-gather and reduce-scatter; the
/// analytic all-gather / reduce-scatter keeps them.
#[derive(Clone, Copy, Debug)]
pub struct AllgatherLeafNames<'a> {
    pub dispatch: &'a str,
    pub combine: &'a str,
}

/// A fast exchange's measured all-reduce combine op.
pub enum FastCombineOp {
    Nvshmem(Op<AllReduceKernel>),
    Mnnvl(Op<AllReduceFusionKernel>),
}

/// The fixed-pattern exchange's profiled dispatch op.
pub enum FixedDispatchOp {
    Gather(Op<MoeEpQuantizedAllGatherKernel>),
    QuantizeAndGather(Op<MoeEpQuantizeAndAllGatherKernel>),
}

/// The fixed-pattern exchange's profiled ops.
pub struct MeasuredFixedOps {
    dispatch: FixedDispatchOp,
    combine: Op<MoeEpReduceScatterKernel>,
}

/// One MoE layer's exchange, built from non-default [`EpExchangeConfigs`].
pub enum EpExchange {
    /// `arch::ep_comm`'s fast exchange: one dispatch and one combine leaf.
    Fast {
        comm: Arc<AnalyticEpComm>,
        dispatch: String,
        combine: String,
        /// The measured all-reduce in place of the analytic combine leg.
        measured_combine: Option<FastCombineOp>,
        /// The fixed-pattern exchange kernels in place of both analytic legs.
        measured_fixed: Option<MeasuredFixedOps>,
        /// The receiver's sum of the partial rows (measured elementwise).
        combine_sum: Option<Op<ElementwiseKernel>>,
        prefix: String,
    },
    /// `arch::ep_comm`'s all-gather / reduce-scatter.
    AnalyticAllgather {
        comm: Arc<AnalyticEpComm>,
        dispatch: String,
        combine: String,
    },
    /// flashinfer's two-sided MNNVL all-to-all with its measured elementwise
    /// work. `comm` gives the busiest ranks' rows, and prices the prepare and
    /// transfers unless `measured` does.
    Alltoall {
        /// The MoE layer's name, which every leaf extends.
        prefix: String,
        comm: Arc<AnalyticEpComm>,
        measured: Option<MeasuredAlltoallOps>,
        scale_swizzle: Op<ElementwiseKernel>,
        combine_fill: Op<ElementwiseKernel>,
        combine_reduce: Op<ElementwiseKernel>,
    },
}

/// The profiled all-to-all prepare and transfers (EP8).
pub struct MeasuredAlltoallOps {
    prepare: Op<MoeAlltoallPrepareKernel>,
    dispatch: Op<MoeAlltoallKernel>,
    dispatch_scales: Op<MoeAlltoallKernel>,
    combine: Op<MoeAlltoallKernel>,
}

/// The measured all-to-all's lookups at the analytic model's rows: the
/// prepare at the busiest rank's tokens, dispatch at (busiest sender,
/// busiest receiver) and combine at its transpose. Each `moe_comm` call
/// moves whole rows.
#[derive(Clone)]
pub struct MeasuredAlltoallInputs {
    pub prepare: MoeAlltoallPrepareKernelInput,
    pub dispatch: MoeAlltoallKernelInput,
    pub combine: MoeAlltoallKernelInput,
}

impl MeasuredAlltoallInputs {
    pub fn new(rows: &AlltoallRows) -> Self {
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

/// One iteration's exchange, computed once for its dispatch and combine.
pub struct EpExchangeStep {
    analytic: EpCommStep,
    measured: Option<MeasuredAlltoallInputs>,
}

impl EpExchange {
    /// `prefix` is the MoE layer's name (`<layer>.moe`); every leaf is
    /// `<prefix>.<leaf>`. `None` on the default measured all-gather /
    /// reduce-scatter, which the arch builds itself.
    pub fn build(
        prefix: &str,
        allgather: AllgatherLeafNames<'_>,
        configs: &EpExchangeConfigs,
        bridge: &PerfApiBridge,
    ) -> Result<Option<Self>, BuildError> {
        let Some(comm) = &configs.analytic else {
            return Ok(None);
        };
        let comm = Arc::new(comm.clone());
        let kind = comm.config.kind;
        if kind.is_fast() {
            let (dispatch, combine) = match kind {
                EpCommKind::FastSparse => ("dispatch_sparse_write", "combine_sparse_write"),
                EpCommKind::FusedAllgather => ("dispatch_fused_all_gather", "combine_fused_reduce"),
                _ => (
                    "dispatch_fast_all_gather",
                    match comm.config.fast.combine {
                        FastCombine::ReduceScatter => "combine_fast_reduce_scatter",
                        FastCombine::AllReduce => "combine_fast_all_reduce",
                    },
                ),
            };
            let combine = format!("{prefix}.{combine}");
            let measured_combine = match &configs.measured_fast_combine {
                None => None,
                Some(MeasuredFastCombine::Nvshmem(config)) => Some(FastCombineOp::Nvshmem(
                    build_op(combine.clone(), config.clone(), AllReduceKernel::build, bridge)?,
                )),
                Some(MeasuredFastCombine::Mnnvl(config)) => Some(FastCombineOp::Mnnvl(build_op(
                    combine.clone(),
                    config.clone(),
                    AllReduceFusionKernel::build,
                    bridge,
                )?)),
            };
            let dispatch = format!("{prefix}.{dispatch}");
            let measured_fixed = configs
                .measured_fixed
                .as_ref()
                .map(|kernels| -> Result<_, BuildError> {
                    Ok(MeasuredFixedOps {
                        dispatch: match &kernels.dispatch {
                            FixedDispatchConfig::Gather(config) => {
                                FixedDispatchOp::Gather(build_op(
                                    dispatch.clone(),
                                    config.clone(),
                                    MoeEpQuantizedAllGatherKernel::build,
                                    bridge,
                                )?)
                            }
                            FixedDispatchConfig::QuantizeAndGather(config) => {
                                FixedDispatchOp::QuantizeAndGather(build_op(
                                    dispatch.clone(),
                                    config.clone(),
                                    MoeEpQuantizeAndAllGatherKernel::build,
                                    bridge,
                                )?)
                            }
                        },
                        combine: build_op(
                            combine.clone(),
                            kernels.combine.clone(),
                            MoeEpReduceScatterKernel::build,
                            bridge,
                        )?,
                    })
                })
                .transpose()?;
            let combine_sum = configs
                .fast_combine_sum
                .as_ref()
                .map(|config| {
                    build_op(
                        format!("{prefix}.combine_sum"),
                        config.clone(),
                        ElementwiseKernel::build,
                        bridge,
                    )
                })
                .transpose()?;
            return Ok(Some(Self::Fast {
                comm,
                dispatch,
                combine,
                measured_combine,
                measured_fixed,
                combine_sum,
                prefix: prefix.to_string(),
            }));
        }
        let Some(alltoall) = &configs.alltoall else {
            return Ok(Some(Self::AnalyticAllgather {
                comm,
                dispatch: format!("{prefix}.{}", allgather.dispatch),
                combine: format!("{prefix}.{}", allgather.combine),
            }));
        };
        let elementwise = |leaf: &str, config: &ElementwiseKernelConfig| {
            build_op(
                format!("{prefix}.{leaf}"),
                config.clone(),
                ElementwiseKernel::build,
                bridge,
            )
        };
        let measured = configs
            .measured_alltoall
            .as_ref()
            .map(|kernels| -> Result<_, BuildError> {
                let transfer = |leaf: &str, config: &MoeAlltoallKernelConfig| {
                    build_op(
                        format!("{prefix}.{leaf}"),
                        config.clone(),
                        MoeAlltoallKernel::build,
                        bridge,
                    )
                };
                Ok(MeasuredAlltoallOps {
                    prepare: build_op(
                        format!("{prefix}.dispatch_alltoall_prepare"),
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
        Ok(Some(Self::Alltoall {
            comm,
            measured,
            scale_swizzle: elementwise("dispatch_scale_swizzle", &alltoall.scale_swizzle)?,
            combine_fill: elementwise("combine_output_fill", &alltoall.combine_fill)?,
            combine_reduce: elementwise("combine_reduce", &alltoall.combine_reduce)?,
            prefix: prefix.to_string(),
        }))
    }

    /// The measured fused dispatch quantizes inside the gather: the arch's
    /// own dispatch quantization must not be billed again.
    pub fn folds_dispatch_quant(&self) -> bool {
        matches!(
            self,
            Self::Fast {
                measured_fixed: Some(MeasuredFixedOps {
                    dispatch: FixedDispatchOp::QuantizeAndGather(_),
                    ..
                }),
                ..
            }
        )
    }

    /// The path, for the exchange's cost-tree label.
    pub fn label(&self) -> String {
        match self {
            Self::Fast {
                measured_fixed: Some(ops),
                ..
            } => match ops.dispatch {
                FixedDispatchOp::Gather(_) => "measured fixed-pattern MNNVL all-gather + \
                                               reduce-scatter (mnnvl_lamport EP8 rows)"
                    .to_string(),
                FixedDispatchOp::QuantizeAndGather(_) => "measured fused MNNVL quantize + \
                                                          all-gather (mnnvl_lamport) and owner-read \
                                                          reduce-scatter (mnnvl_pull), EP8 rows"
                    .to_string(),
            },
            Self::Alltoall {
                measured: Some(_), ..
            } => "measured flashinfer MNNVL all-to-all (EP8 rows)".to_string(),
            Self::Fast {
                comm,
                measured_combine: Some(op),
                ..
            } => format!(
                "{}; combine measured {}",
                comm.label(),
                match op {
                    FastCombineOp::Nvshmem(_) => "nvshmem all_reduce",
                    FastCombineOp::Mnnvl(_) => "flashinfer_mnnvl all_reduce_fusion",
                }
            ),
            Self::AnalyticAllgather { comm, .. }
            | Self::Alltoall { comm, .. }
            | Self::Fast { comm, .. } => comm.label(),
        }
    }

    /// Everything between the rank-local work and the routed experts, over
    /// `ranks` EP ranks. Leaves are declared in [`Self::eval_dispatch`] order.
    pub fn compile_dispatch(&self, ranks: usize, builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        match self {
            Self::Fast {
                measured_fixed: Some(ops),
                ..
            } => vec![match &ops.dispatch {
                FixedDispatchOp::Gather(op) => op.compile(builder),
                FixedDispatchOp::QuantizeAndGather(op) => op.compile(builder),
            }],
            Self::AnalyticAllgather { comm, dispatch, .. } | Self::Fast { comm, dispatch, .. } => {
                vec![comm.compile_leg(dispatch.clone(), "dispatch", builder)]
            }
            Self::Alltoall {
                prefix,
                comm,
                measured,
                scale_swizzle,
                ..
            } => {
                let mut chain = match measured {
                    Some(measured) => vec![
                        measured.prepare.compile(builder),
                        measured.dispatch.compile(builder),
                        measured.dispatch_scales.compile(builder),
                    ],
                    None => [
                        ("dispatch_alltoall_prepare", "prepare"),
                        ("dispatch_alltoall", "dispatch"),
                        ("dispatch_alltoall_scales", "dispatch_scales"),
                    ]
                    .into_iter()
                    .map(|(leaf, op)| comm.compile_leg(format!("{prefix}.{leaf}"), op, builder))
                    .collect(),
                };
                chain.push(ranks_max(
                    prefix,
                    "dispatch_scale_swizzle",
                    scale_swizzle,
                    ranks,
                    builder,
                ));
                chain
            }
        }
    }

    /// Everything between the routed experts and the per-rank add, over
    /// `ranks` EP ranks. Leaves are declared in [`Self::eval_combine`] order.
    pub fn compile_combine(&self, ranks: usize, builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        match self {
            Self::AnalyticAllgather { comm, combine, .. } => {
                vec![comm.compile_leg(combine.clone(), "combine", builder)]
            }
            Self::Fast {
                comm,
                combine,
                measured_combine,
                measured_fixed,
                combine_sum,
                prefix,
                ..
            } => {
                let mut chain = vec![match (measured_fixed, measured_combine) {
                    (Some(ops), _) => ops.combine.compile(builder),
                    (None, Some(FastCombineOp::Nvshmem(op))) => op.compile(builder),
                    (None, Some(FastCombineOp::Mnnvl(op))) => op.compile(builder),
                    (None, None) => comm.compile_leg(combine.clone(), "combine", builder),
                }];
                if let Some(sum) = combine_sum {
                    chain.push(ranks_max(prefix, "combine_sum", sum, ranks, builder));
                }
                chain
            }
            Self::Alltoall {
                prefix,
                comm,
                measured,
                combine_fill,
                combine_reduce,
                ..
            } => vec![
                ranks_max(prefix, "combine_output_fill", combine_fill, ranks, builder),
                match measured {
                    Some(measured) => measured.combine.compile(builder),
                    None => {
                        comm.compile_leg(format!("{prefix}.combine_alltoall"), "combine", builder)
                    }
                },
                ranks_max(prefix, "combine_reduce", combine_reduce, ranks, builder),
            ],
        }
    }

    /// `tokens` are each rank's rows in the exchange (padded, with an idle
    /// rank's dummy token; all zero when no rank has work).
    pub fn step(&self, tokens: &[u32]) -> EpExchangeStep {
        match self {
            Self::AnalyticAllgather { comm, .. }
            | Self::Fast { comm, .. }
            | Self::Alltoall {
                comm,
                measured: None,
                ..
            } => EpExchangeStep {
                analytic: comm.step(tokens),
                measured: None,
            },
            Self::Alltoall { comm, .. } => EpExchangeStep {
                analytic: EpCommStep::default(),
                measured: Some(MeasuredAlltoallInputs::new(&comm.alltoall_rows(tokens))),
            },
        }
    }

    pub fn eval_dispatch(&self, step: &EpExchangeStep, tokens: &[u32], ev: &mut Evaluator) {
        match self {
            Self::AnalyticAllgather { .. } => push_leg("dispatch", step.analytic.dispatch, ev),
            Self::Fast {
                measured_fixed: Some(ops),
                ..
            } => {
                let input = MoeEpCollectiveKernelInput {
                    per_rank_tokens: tokens.to_vec(),
                };
                let idle = tokens.iter().all(|&rows| rows == 0);
                match &ops.dispatch {
                    FixedDispatchOp::Gather(op) => push_or_zero(op, input, idle, ev),
                    FixedDispatchOp::QuantizeAndGather(op) => push_or_zero(op, input, idle, ev),
                }
            }
            Self::Fast { .. } => push_extrapolated_leg("dispatch", step.analytic.dispatch, ev),
            Self::Alltoall {
                measured,
                scale_swizzle,
                ..
            } => {
                let idle = tokens.iter().all(|&rows| rows == 0);
                match (measured, &step.measured) {
                    (Some(measured), Some(inputs)) => {
                        push_or_zero(&measured.prepare, inputs.prepare.clone(), idle, ev);
                        push_or_zero(&measured.dispatch, inputs.dispatch.clone(), idle, ev);
                        // The block scales are a second `moe_comm` of the same rows.
                        push_or_zero(&measured.dispatch_scales, inputs.dispatch.clone(), idle, ev);
                    }
                    _ => {
                        push_leg("prepare", step.analytic.prepare, ev);
                        push_leg("dispatch", step.analytic.dispatch, ev);
                        push_leg("dispatch_scales", step.analytic.dispatch_scales, ev);
                    }
                }
                // Every rank swizzles its whole receive allocation.
                let busiest = tokens.iter().copied().max().unwrap_or(0);
                let allocation = busiest.saturating_mul(tokens.len() as u32);
                for _ in tokens {
                    push_or_zero(
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
    }

    pub fn eval_combine(&self, step: &EpExchangeStep, tokens: &[u32], ev: &mut Evaluator) {
        match self {
            Self::AnalyticAllgather { .. } => push_leg("combine", step.analytic.combine, ev),
            Self::Fast {
                comm,
                measured_combine,
                measured_fixed,
                combine_sum,
                ..
            } => {
                let total: u32 = tokens.iter().sum();
                let idle = total == 0;
                if let Some(ops) = measured_fixed {
                    let input = MoeEpCollectiveKernelInput {
                        per_rank_tokens: tokens.to_vec(),
                    };
                    push_or_zero(&ops.combine, input, idle, ev);
                }
                match measured_combine {
                    _ if measured_fixed.is_some() => {}
                    None => push_extrapolated_leg("combine", step.analytic.combine, ev),
                    // Every rank's partial sums of all `T` tokens, all-reduced.
                    Some(FastCombineOp::Nvshmem(op)) => push_or_zero(
                        op,
                        AllReduceKernelInput {
                            message_size_bytes: (f64::from(total) * comm.fast_combine_row_bytes())
                                as u64,
                        },
                        idle,
                        ev,
                    ),
                    Some(FastCombineOp::Mnnvl(op)) => push_or_zero(
                        op,
                        AllReduceFusionKernelInput { num_tokens: total },
                        idle,
                        ev,
                    ),
                }
                if let Some(sum) = combine_sum {
                    for &rows in tokens {
                        push_or_zero(
                            sum,
                            ElementwiseKernelInput { num_tokens: rows },
                            rows == 0,
                            ev,
                        );
                    }
                }
            }
            Self::Alltoall {
                measured,
                combine_fill,
                combine_reduce,
                ..
            } => {
                let idle = tokens.iter().all(|&rows| rows == 0);
                // The staging buffer and the sum run on each rank's own tokens.
                for &rows in tokens {
                    push_or_zero(
                        combine_fill,
                        ElementwiseKernelInput { num_tokens: rows },
                        rows == 0,
                        ev,
                    );
                }
                match (measured, &step.measured) {
                    (Some(measured), Some(inputs)) => {
                        push_or_zero(&measured.combine, inputs.combine.clone(), idle, ev);
                    }
                    _ => push_leg("combine", step.analytic.combine, ev),
                }
                for &rows in tokens {
                    push_or_zero(
                        combine_reduce,
                        ElementwiseKernelInput { num_tokens: rows },
                        rows == 0,
                        ev,
                    );
                }
            }
        }
    }
}

fn ranks_max(
    prefix: &str,
    leaf: &str,
    op: &Op<ElementwiseKernel>,
    ranks: usize,
    builder: &mut CostTreeBuilder,
) -> CostNode {
    CostNode::Labeled {
        label: format!("{prefix}.{leaf} [Max over EP ranks]"),
        child: Box::new(CostNode::Max {
            overlap: 1.0,
            children: (0..ranks).map(|_| op.compile(builder)).collect(),
        }),
    }
}

fn build_op<K, C, F>(
    name: String,
    config: C,
    build_kernel: F,
    bridge: &PerfApiBridge,
) -> Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError>,
{
    Ok(Op::new(
        name.clone(),
        Arc::new(build_kernel(name, config, bridge)?),
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
