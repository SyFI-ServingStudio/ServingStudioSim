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
//! # Options for the DFlash2 ladder ([`Glm53FlashDpAttnEpParallel`])
//!
//! The defaults are vLLM's layout above. The rest are designs the simulator
//! prices from measured leaves:
//!
//! - `expert_exchange: FixedPattern`: the NVFP4 exchange as the fixed-pattern
//!   MNNVL all-gather + reduce-scatter ([`FixedExpertExchange`], EP8 rows).
//! - `kda_tp_gpus = b > 1`: the KDA layers only run tensor-parallel over
//!   groups of `b` ranks. Every rank still owns its requests (scheduler, KV
//!   cache, DSA layers, FFN and MoE stay rank-local); each KDA layer
//!   all-gathers its group's inputs, runs the group's rows with `64 / b` heads
//!   per rank, and reduce-scatters the output back to the owners
//!   ([`GroupGatherScatter`], two launches on the fixed-pattern rows measured
//!   at `b` GPUs). Each owner keeps its requests' whole KDA state.
//! - `mhc_fused_saving_us`, `moe_combine_add_in_mhc_post`, `moe_input_copy`,
//!   `router_gate_once`, `moe_shared_exposed`: see their field docs.
//!
//! A speculative target ([`Glm53FlashVllmConfigs`] KDA `decode_query_len` and
//! DSA `decode_next_n` = `k + 1`) presents `k + 1` query rows per decode
//! request; [`Glm53FlashDpAttnEpModel::compile_root`] and `eval_into` let the
//! DFlash2 verify model compile it beside its draft.

use anyhow::Result;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::glm53_flash_vllm_fp8_kda_dsa_moe::{
    self as tp_arch, atomic, attention_inputs, graph_padded_tokens, push, AttnKind, Boundary,
    FfnKind, Glm53FlashLayerGroup, Glm53FlashModelCfg, Glm53FlashQuant, Glm53FlashVllmConfigs,
    Glm53FlashVllmParallel, ACTIVATION_DTYPE, SHARED_EXPERTS_STREAM_OVERLAP,
    SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD,
};
use crate::arch::kv_layout::KvCacheLayout;
use crate::op::comm::{
    FixedExpertExchange, FixedExpertExchangeConfig, GroupGatherScatter, GroupGatherScatterConfig,
};
use crate::op::mhc::{MhcTerminalPostConfig, MhcTerminalPostInput, MhcTerminalPostOp};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    MhcFusedPostPreRmsNormKernel, MhcFusedSumKernelConfig, MhcFusedSumPostPreRmsNormKernel,
    MhcPreRmsNormKernel, MhcRmsNormKernelInput, MoeEpAllGatherKernel, MoeEpAllGatherKernelConfig,
    MoeEpCollectiveKernelInput, MoeEpQuantizedAllGatherKernel, MoeEpQuantizedAllGatherKernelConfig,
    MoeEpReduceScatterKernel, MoeEpReduceScatterKernelConfig, Nvfp4FusedMoeKernel,
    Nvfp4FusedMoeKernelInput, RmsNormKernel, RmsNormKernelInput, SingleGemmKernel,
    SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, CoverageFlags, Evaluator,
    FlatCostNode, LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::{
    Glm53ActivationQuant, Glm53DsaAttnLocalWorklet, Glm53DsaAttnLocalWorkletInput,
    Glm53KdaAttnLocalWorklet, Glm53KdaAttnLocalWorkletConfig, Glm53KdaAttnLocalWorkletInput,
    Glm53MlpLocalWorklet, Glm53MlpLocalWorkletInput, Glm53MoeRouterLocalWorklet,
    Glm53MoeRouterLocalWorkletInput, Glm53RoutedMoeLocalWorklet, Glm53RoutedMoeLocalWorkletConfig,
    Glm53RoutedMoeLocalWorkletResolved,
};

const ARCH_KIND: &str = "glm53_flash_vllm_fp8_dp_attn_ep_moe";
const COLLECTIVE_BACKENDS: &[&str] = &["vllm_pynccl"];
/// The NVFP4 modular experts take the routing chosen before the dispatch.
const NVFP4_ROUTED_MOE_BACKENDS: &[&str] = &["flashinfer_trtllm_routed_sm100"];
const ROUTER_SELECT_BACKENDS: &[&str] = &["triton"];
/// vLLM's after-MoE boundary with the shared + routed add read by the post
/// kernel (`moe_combine_add_in_mhc_post`).
const MHC_POST_ADDEND_BACKENDS: &[&str] = &["vllm_tilelang_post_addend"];
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
    pub kernel_path: tp_arch::Glm53FlashKernelPath,
    pub expert_exchange: ExpertExchange,
    /// KDA-only attention TP group size (module doc); 1 is plain DP attention.
    pub kda_tp_gpus: u16,
    /// KDA `f_b` / `g_b` as one batched GEMM (worklet
    /// `Glm53KdaAttnLocalWorkletConfig::merge_gate_b`).
    pub kda_gate_b_merged: bool,
    /// What-if: every fused mHC post/pre boundary (all but layer 0's pre and
    /// the terminal post) costs vLLM's row less this many us (the leaf keeps
    /// its per-row part and is flagged extrapolated). `None` is vLLM's rows.
    pub mhc_fused_saving_us: Option<f64>,
    /// Compute the router gate once per MoE layer (the fork since #55736)
    /// instead of twice (the 2026-09 alignment capture's line).
    pub router_gate_once: bool,
    /// Price the routed-input bf16 copy ahead of the dispatch quant
    /// (`moe.input_glue`, from the 2026-09 FP8 TP4/EP4 capture). The fork's
    /// NVFP4 naive DP/EP path has no such copy; `false` drops the leaf.
    pub moe_input_copy: bool,
    /// Fold the MoE block's shared + routed add into the next mHC post
    /// (`MHC_POST_ADDEND_BACKENDS`): the boundary after each MoE layer and the
    /// last layer's post take the sum kind, and `combine_glue` is not billed.
    pub moe_combine_add_in_mhc_post: bool,
    /// `None`: the aux-stream shared expert (local rows <= 256) overlaps the
    /// whole dispatch + routed + combine chain by
    /// `SHARED_EXPERTS_STREAM_OVERLAP`. `Some`: no overlap factor; the
    /// measured time it adds to the MoE block (block with it minus block
    /// without it, by rows per rank) is added to the routed call at the
    /// busiest such rank's rows. When every rank's rows are > 256 the shared
    /// expert runs on the main stream and nothing is added.
    pub moe_shared_exposed: Option<MoeSharedPair>,
}

impl Glm53FlashDpAttnEpParallel {
    /// vLLM's layout (module doc), every option off.
    pub fn new(
        dp_size: u16,
        max_model_len: u32,
        gpu_name: String,
        cudagraph_capture_sizes: Vec<u32>,
        kernel_path: tp_arch::Glm53FlashKernelPath,
    ) -> Self {
        Self {
            dp_size,
            max_model_len,
            gpu_name,
            cudagraph_capture_sizes,
            kernel_path,
            expert_exchange: ExpertExchange::MeasuredAllgather,
            kda_tp_gpus: 1,
            kda_gate_b_merged: false,
            mhc_fused_saving_us: None,
            router_gate_once: false,
            moe_input_copy: true,
            moe_combine_add_in_mhc_post: false,
            moe_shared_exposed: None,
        }
    }
}

/// How the routed tokens reach the experts and come back (`expert_exchange`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertExchange {
    /// vLLM's profiled all-gatherv and bf16 reduce-scatterv.
    #[default]
    MeasuredAllgather,
    /// [`FixedExpertExchange`] (NVFP4, EP 8 only).
    FixedPattern,
}

/// The `expert_exchange` choices the launcher schema advertises.
pub const EXPERT_EXCHANGES: [&str; 2] = ["measured_allgather", "fixed_pattern"];

/// How the aux-stream shared expert (local rows <= 256) overlaps the MoE
/// chain (`moe_shared_overlap`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeSharedOverlap {
    /// The overlap factor over the dispatch + routed + combine chain.
    #[default]
    Chain,
    /// [`MoeSharedPair`]: the measured time it adds to the MoE block.
    Exposed,
}

/// The `moe_shared_overlap` choices the launcher schema advertises.
pub const MOE_SHARED_OVERLAPS: [&str; 2] = ["chain", "exposed"];

/// Measured time the aux-stream shared expert adds to an MoE block, by rows
/// per rank. Linear between points, the nearest point outside them; the
/// routed leaves it touches are flagged extrapolated.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeSharedPair {
    pub points: Vec<MoeSharedPairPoint>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeSharedPairPoint {
    pub rows: u32,
    pub added_us: f64,
}

impl MoeSharedPair {
    /// Reads column `<column>_us` of an exposure table
    /// `{"points": [{"rows", "<column>_us", ...}, ...]}`.
    pub fn load_exposed(path: &str, column: &str) -> Result<Self> {
        let err = |e: String| anyhow::anyhow!("moe_shared_pair_file {path}: {e}");
        let text = std::fs::read_to_string(path).map_err(|e| err(e.to_string()))?;
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Table {
            points: Vec<std::collections::BTreeMap<String, f64>>,
        }
        let table: Table = serde_json::from_str(&text).map_err(|e| err(e.to_string()))?;
        let key = format!("{column}_us");
        let points = table
            .points
            .iter()
            .map(|p| {
                let rows = p
                    .get("rows")
                    .ok_or_else(|| err("a point has no rows".into()))?;
                anyhow::ensure!(
                    rows.fract() == 0.0 && *rows >= 1.0 && *rows <= f64::from(u32::MAX),
                    err(format!("rows {rows} is not a positive integer"))
                );
                let us = p
                    .get(&key)
                    .ok_or_else(|| err(format!("no {key} at {rows} rows")))?;
                anyhow::ensure!(
                    *us >= 0.0,
                    err(format!("{key} must be >= 0 at {rows} rows"))
                );
                Ok(MoeSharedPairPoint {
                    rows: *rows as u32,
                    added_us: *us,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        anyhow::ensure!(!points.is_empty(), err("no points".into()));
        anyhow::ensure!(
            points.windows(2).all(|w| w[0].rows < w[1].rows),
            err("points must be sorted by rows, strictly".into())
        );
        Ok(Self { points })
    }

    /// The added time in ms at `rows`.
    pub fn added_ms(&self, rows: u32) -> f64 {
        let i = self.points.partition_point(|p| p.rows < rows);
        let us = match (i.checked_sub(1).map(|j| self.points[j]), self.points.get(i)) {
            (_, Some(hi)) if hi.rows == rows => hi.added_us,
            (Some(lo), Some(hi)) => {
                let f = f64::from(rows - lo.rows) / f64::from(hi.rows - lo.rows);
                lo.added_us + f * (hi.added_us - lo.added_us)
            }
            (None, Some(hi)) => hi.added_us,
            (Some(lo), None) => lo.added_us,
            (None, None) => 0.0,
        };
        us / 1e3
    }
}

#[derive(Clone, Debug)]
pub struct Glm53FlashDpAttnEpConfigs {
    /// Every rank-local sublayer at TP 1 (64 KDA/MLA heads, full dense FFN,
    /// shared expert and vocab). Its `routed` and `all_reduce` are unused.
    /// Under `kda_tp_gpus > 1` its KDA config is the group's (heads split).
    pub local: Glm53FlashVllmConfigs,
    /// The TP1 KDA config: the RecoverSSM commit and the KV layout are priced
    /// per owner rank from it.
    pub kda_tp1: Glm53KdaAttnLocalWorkletConfig,
    pub parallel: Glm53FlashDpAttnEpParallel,
    /// One per EP rank, ranked by routed workload.
    pub routed: Vec<Glm53RoutedMoeLocalWorkletConfig>,
    pub dispatch: Glm53FlashDispatchConfig,
    /// vLLM's grouped top-k ahead of an NVFP4 dispatch: fp32 logits and bias
    /// in, top-k IDs and weights out. FP8 routes inside the fused MoE.
    pub router_select: Option<ElementwiseKernelConfig>,
    pub combine: MoeEpReduceScatterKernelConfig,
    /// `Some` under `ExpertExchange::FixedPattern`, in place of `dispatch` /
    /// `combine`.
    pub fixed_exchange: Option<FixedExpertExchangeConfig>,
    /// `Some` under `kda_tp_gpus > 1`.
    pub kda_gather_scatter: Option<GroupGatherScatterConfig>,
    /// `Some` under `moe_combine_add_in_mhc_post`: the boundary after an MoE
    /// layer (and the last layer's post) with the shared + routed add in the
    /// post kernel.
    pub mhc_post_addend: Option<MhcFusedSumKernelConfig>,
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
    let kda_tp = parallel.kda_tp_gpus;
    if kda_tp == 0 || dp % kda_tp != 0 {
        return Err(fit_failed(format!(
            "KDA tensor-parallel group {kda_tp} must divide the {dp} DP ranks"
        )));
    }
    let tp_parallel = |tp_size| Glm53FlashVllmParallel {
        tp_size,
        // Whole routed intermediate; the EP split over DP ranks follows.
        enable_expert_parallel: true,
        max_model_len: parallel.max_model_len,
        gpu_name: parallel.gpu_name.clone(),
        cudagraph_capture_sizes: parallel.cudagraph_capture_sizes.clone(),
        kernel_path: parallel.kernel_path.clone(),
    };
    let mut local = tp_arch::build_configs(model, &tp_parallel(1), demand)?;
    local.kda.merge_gate_b = parallel.kda_gate_b_merged;
    let kda_tp1 = local.kda.clone();
    let gpu = parallel.gpu_name.clone();
    let kda_gather_scatter = if kda_tp > 1 {
        let group = tp_arch::build_configs(model, &tp_parallel(kda_tp), demand)?;
        local.kda = group.kda;
        local.kda.merge_gate_b = parallel.kda_gate_b_merged;
        let gpus = u32::from(kda_tp);
        Some(GroupGatherScatterConfig::new(
            &gpu,
            gpus,
            model.hidden,
            RANK_TOKEN_BUDGET * gpus,
        ))
    } else {
        None
    };
    let mhc_post_addend = parallel.moe_combine_add_in_mhc_post.then(|| {
        // The boundary after a dense layer has no MoE output to add.
        local.groups = split_groups_after_dense(model, std::mem::take(&mut local.groups));
        MhcFusedSumKernelConfig {
            backends: MHC_POST_ADDEND_BACKENDS.to_vec(),
            gpu_name: local.mhc_fused.gpu_name.clone(),
            hidden_size: local.mhc_fused.hidden_size.clone(),
            hc_mult: local.mhc_fused.hc_mult,
            hidden_dtype: local.mhc_fused.hidden_dtype,
            num_partials: 1,
            shared_addend: true,
        }
    });
    let max_total_tokens = RANK_TOKEN_BUDGET * u32::from(dp);
    let fixed_exchange = match parallel.expert_exchange {
        ExpertExchange::MeasuredAllgather => None,
        ExpertExchange::FixedPattern if model.quant != Glm53FlashQuant::Nvfp4 => {
            return Err(fit_failed(
                "the fixed-pattern expert exchange moves NVFP4 activations: it needs the \
                 NVFP4 checkpoint",
            ));
        }
        ExpertExchange::FixedPattern => Some(
            FixedExpertExchangeConfig::new(
                &gpu,
                dp.into(),
                model.hidden,
                model.num_experts_per_tok,
                max_total_tokens,
            )
            .map_err(fit_failed)?,
        ),
    };
    let mut template = local.routed[0].clone();
    template.ep_size = dp;
    if model.quant == Glm53FlashQuant::Nvfp4 {
        template.fused_moe_backends = NVFP4_ROUTED_MOE_BACKENDS.to_vec();
    }
    let routed = Glm53RoutedMoeLocalWorkletConfig::split_for_ep(template, demand.clone());
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
        fixed_exchange,
        kda_gather_scatter,
        mhc_post_addend,
    })
}

/// Splits each MoE group's layers that follow a dense layer into their own
/// group (`<label>_after_dense`): their attention boundary has no MoE output
/// to sum.
fn split_groups_after_dense(
    model: &Glm53FlashModelCfg,
    groups: Vec<Glm53FlashLayerGroup>,
) -> Vec<Glm53FlashLayerGroup> {
    let mut out = Vec::with_capacity(groups.len() + 1);
    for group in groups {
        let (after_dense, after_moe): (Vec<u32>, Vec<u32>) = group
            .layers
            .iter()
            .partition(|&&layer| layer == 0 || model.is_dense(layer - 1));
        if group.ffn != FfnKind::Moe || after_dense.is_empty() || after_moe.is_empty() {
            out.push(group);
            continue;
        }
        out.push(Glm53FlashLayerGroup {
            label: format!("{}_after_dense", group.label),
            layers: after_dense,
            ..group.clone()
        });
        out.push(Glm53FlashLayerGroup {
            layers: after_moe,
            ..group
        });
    }
    out
}

/// Whether the layer before every layer of `group` is an MoE layer.
fn follows_moe(model: &Glm53FlashModelCfg, group: &Glm53FlashLayerGroup) -> bool {
    group
        .layers
        .iter()
        .all(|&layer| layer > 0 && !model.is_dense(layer - 1))
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
    /// `None` under `moe_input_copy: false`.
    input_glue: Option<Op<ElementwiseKernel>>,
    shared_expert: Glm53MlpLocalWorklet,
    router_select: Option<Op<ElementwiseKernel>>,
    dispatch_quant: Glm53ActivationQuant,
    exchange: Exchange,
    /// One per EP rank, ranked by routed workload.
    fused_moe: Vec<Op<Nvfp4FusedMoeKernel>>,
    /// `Glm53FlashDpAttnEpParallel::moe_shared_exposed`.
    shared_exposed: Option<MoeSharedPair>,
    /// The shared + routed add; `None` when the next mHC post does it.
    combine_glue: Option<Op<ElementwiseKernel>>,
}

enum Exchange {
    Measured {
        dispatch: Dispatch,
        combine: Op<MoeEpReduceScatterKernel>,
    },
    Fixed(FixedExpertExchange),
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

/// The mHC boundary ahead of a layer's attention.
enum AttnBoundary {
    /// Layer 0's pre, or a fused post/pre of a single block output.
    Mhc(Boundary),
    /// After an MoE layer under `moe_combine_add_in_mhc_post`: the shared +
    /// routed add read by the post, then the fused post/pre.
    SumAfterMoe(Op<MhcFusedSumPostPreRmsNormKernel>),
}

impl AttnBoundary {
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        match self {
            Self::Mhc(boundary) => boundary.compile(builder),
            Self::SumAfterMoe(op) => op.compile(builder),
        }
    }
}

/// A fused mHC post/pre boundary, less `saving_us` when the what-if is on.
fn push_mhc_fused(
    op: &Op<MhcFusedPostPreRmsNormKernel>,
    num_tokens: u32,
    saving_us: Option<f64>,
    ev: &mut Evaluator,
) {
    let input = MhcRmsNormKernelInput { num_tokens };
    let Some(saving) = saving_us else {
        return push(op, input, ev);
    };
    let mut metrics = op.kernel.eval(&input);
    let measured = f64::from(metrics.m.time_ms);
    metrics.m.time_ms = (measured - saving / 1e3).max(0.0) as f32;
    metrics.coverage = metrics.coverage | CoverageFlags::EXTRAPOLATED;
    ev.push(metrics, || input.clone().into());
}

struct LayerGroup {
    label: String,
    layers: Vec<u32>,
    attn_boundary: AttnBoundary,
    attn: Attention,
    /// The KDA group's input all-gather and output reduce-scatter; `Some` on
    /// a KDA layer under `kda_tp_gpus > 1`.
    kda_gather_scatter: Option<GroupGatherScatter>,
    ffn_boundary: Op<MhcFusedPostPreRmsNormKernel>,
    ffn: Ffn,
    mhc_fused_saving_us: Option<f64>,
}

impl LayerGroup {
    /// Per rank: everything up to the dispatch (the whole layer for a dense
    /// FFN). Then, for an MoE layer, the group-wide expert section and the
    /// per-rank combine add.
    fn compile(&self, dp: usize, kda_groups: usize, builder: &mut CostTreeBuilder) -> CostNode {
        if let (Attention::Kda(worklet), Some(pair)) = (&self.attn, &self.kda_gather_scatter) {
            return self.compile_kda_tp(worklet, pair, dp, kda_groups, builder);
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
                leaves.extend(self.compile_ffn_local(builder));
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
        self.scaled(body)
    }

    /// KDA-only attention TP: every owner rank runs its boundary, each KDA
    /// group runs its ranks' rows with heads split between its gather and
    /// scatter, and every owner rank runs the rest.
    fn compile_kda_tp(
        &self,
        worklet: &Glm53KdaAttnLocalWorklet,
        pair: &GroupGatherScatter,
        dp: usize,
        kda_groups: usize,
        builder: &mut CostTreeBuilder,
    ) -> CostNode {
        let boundary = (0..dp)
            .map(|_| self.attn_boundary.compile(builder))
            .collect();
        let kda = (0..kda_groups)
            .map(|_| CostNode::Sum(vec![worklet.compile(builder), pair.compile(builder)]))
            .collect();
        let ranks = (0..dp)
            .map(|_| CostNode::Sum(self.compile_ffn_local(builder)))
            .collect();
        let mut body = vec![
            CostNode::Labeled {
                label: format!("{} attention boundary [Max over owner ranks]", self.label),
                child: Box::new(CostNode::Max {
                    overlap: 1.0,
                    children: boundary,
                }),
            },
            CostNode::Labeled {
                label: format!(
                    "{} KDA tensor-parallel over the group [Max over groups; {}]",
                    self.label,
                    GroupGatherScatter::PRICING
                ),
                child: Box::new(CostNode::Max {
                    overlap: 1.0,
                    children: kda,
                }),
            },
            CostNode::Labeled {
                label: format!(
                    "{} [Max over owner ranks: FFN up to the dispatch]",
                    self.label
                ),
                child: Box::new(CostNode::Max {
                    overlap: 1.0,
                    children: ranks,
                }),
            },
        ];
        if let Ffn::Moe(block) = &self.ffn {
            body.extend(block.compile_expert_sections(dp, builder));
        }
        self.scaled(body)
    }

    fn scaled(&self, body: Vec<CostNode>) -> CostNode {
        CostNode::Labeled {
            label: format!("{} layers {:?}", self.label, self.layers),
            child: Box::new(CostNode::Scale {
                n: self.layers.len() as u32,
                child: Box::new(CostNode::Sum(body)),
            }),
        }
    }

    /// One rank's FFN boundary and FFN up to the dispatch.
    fn compile_ffn_local(&self, builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut leaves = vec![self.ffn_boundary.compile(builder)];
        match &self.ffn {
            Ffn::Dense(worklet) => leaves.push(worklet.compile(builder)),
            Ffn::Moe(block) => {
                leaves.push(block.router.compile(builder));
                leaves.extend(block.router_select.as_ref().map(|op| op.compile(builder)));
                leaves.extend(block.input_glue.as_ref().map(|op| op.compile(builder)));
                leaves.extend([
                    block.shared_expert.compile(builder),
                    block.dispatch_quant.compile(builder),
                ]);
            }
        }
        leaves
    }

    fn eval(&self, batch: &NormalizedBatch, ev: &mut Evaluator) {
        if let (Attention::Kda(worklet), Some(pair)) = (&self.attn, &self.kda_gather_scatter) {
            for rank in &batch.ranks {
                self.eval_attn_boundary(rank.rows, ev);
            }
            for group in &batch.kda_groups {
                worklet.eval(&group.kda, ev);
                pair.eval(&group.member_rows, ev);
            }
            for rank in &batch.ranks {
                self.eval_ffn_local(rank.rows, ev);
            }
        } else {
            for rank in &batch.ranks {
                self.eval_attn_boundary(rank.rows, ev);
                match &self.attn {
                    Attention::Kda(worklet) => worklet.eval(&rank.kda, ev),
                    Attention::Dsa(worklet) => worklet.eval(&rank.dsa, ev),
                }
                self.eval_ffn_local(rank.rows, ev);
            }
        }
        if let Ffn::Moe(block) = &self.ffn {
            block.eval_expert_sections(batch, ev);
        }
    }

    fn eval_attn_boundary(&self, rows: u32, ev: &mut Evaluator) {
        match &self.attn_boundary {
            AttnBoundary::Mhc(Boundary::Fused(op)) => {
                push_mhc_fused(op, rows, self.mhc_fused_saving_us, ev)
            }
            AttnBoundary::Mhc(boundary) => boundary.eval(rows, ev),
            AttnBoundary::SumAfterMoe(op) => {
                push(op, MhcRmsNormKernelInput { num_tokens: rows }, ev)
            }
        }
    }

    fn eval_ffn_local(&self, rows: u32, ev: &mut Evaluator) {
        push_mhc_fused(&self.ffn_boundary, rows, self.mhc_fused_saving_us, ev);
        match &self.ffn {
            Ffn::Dense(worklet) => {
                worklet.eval(&Glm53MlpLocalWorkletInput { num_tokens: rows }, ev)
            }
            Ffn::Moe(block) => block.eval_local(rows, ev),
        }
    }
}

impl MoeBlock {
    /// The dispatch / routed experts / combine chain, which the aux-stream
    /// shared expert (T_r <= 256) runs beside, then the per-rank shared +
    /// routed add.
    fn compile_expert_sections(&self, dp: usize, builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        let dispatch = match &self.exchange {
            Exchange::Measured { dispatch, .. } => dispatch.compile(builder),
            Exchange::Fixed(exchange) => exchange.compile_dispatch(builder),
        };
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
        let (combine, path) = match &self.exchange {
            Exchange::Measured { combine, .. } => (
                combine.compile(builder),
                format!(
                    "all_gatherv -> EP{} routed experts -> reduce_scatterv",
                    self.fused_moe.len()
                ),
            ),
            Exchange::Fixed(exchange) => (
                exchange.compile_combine(builder),
                format!(
                    "{} around EP{} routed experts",
                    FixedExpertExchange::LABEL,
                    self.fused_moe.len()
                ),
            ),
        };
        let concurrent_shared = CostNode::Max {
            overlap: 1.0,
            children: (0..dp)
                .map(|_| self.shared_expert.compile(builder))
                .collect(),
        };
        let exposed = self.shared_exposed.is_some();
        let mut sections = vec![CostNode::Labeled {
            label: format!(
                "{} (MoE) [{path}; aux-stream shared expert at T_r<={} {}]",
                self.name,
                SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD,
                if exposed {
                    "adds its measured exposed time on the routed call"
                } else {
                    "overlaps"
                }
            ),
            child: Box::new(CostNode::Parallel {
                // The measured exposed time already sits on the routed leaves.
                overlap: if exposed {
                    1.0
                } else {
                    SHARED_EXPERTS_STREAM_OVERLAP
                },
                children: vec![
                    CostNode::Sum(vec![dispatch, experts, combine]),
                    concurrent_shared,
                ],
            }),
        }];
        if let Some(glue) = &self.combine_glue {
            sections.push(CostNode::Labeled {
                label: format!("{}.combine_glue [Max over DP ranks]", self.name),
                child: Box::new(CostNode::Max {
                    overlap: 1.0,
                    children: (0..dp).map(|_| glue.compile(builder)).collect(),
                }),
            });
        }
        sections
    }

    /// Router, the routed-input copy, the main-stream shared expert for
    /// T_r > 256, and the quant of the local tokens ahead of the dispatch.
    fn eval_local(&self, rows: u32, ev: &mut Evaluator) {
        self.router
            .eval(&Glm53MoeRouterLocalWorkletInput { num_tokens: rows }, ev);
        if let Some(select) = &self.router_select {
            push(select, ElementwiseKernelInput { num_tokens: rows }, ev);
        }
        if let Some(glue) = &self.input_glue {
            push(glue, ElementwiseKernelInput { num_tokens: rows }, ev);
        }
        self.shared_expert.eval_or_zero(
            &Glm53MlpLocalWorkletInput { num_tokens: rows },
            rows <= SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD,
            ev,
        );
        self.dispatch_quant.eval_or_zero(rows, false, ev);
    }

    fn eval_expert_sections(&self, batch: &NormalizedBatch, ev: &mut Evaluator) {
        let tokens: Vec<u32> = batch.ranks.iter().map(|rank| rank.rows).collect();
        match &self.exchange {
            Exchange::Measured { dispatch, .. } => dispatch.eval(
                MoeEpCollectiveKernelInput {
                    per_rank_tokens: tokens.clone(),
                },
                ev,
            ),
            Exchange::Fixed(exchange) => exchange.eval_dispatch(&tokens, ev),
        }
        // The busiest aux-stream rank's measured exposed time.
        let exposed_ms = self.shared_exposed.as_ref().and_then(|table| {
            tokens
                .iter()
                .copied()
                .filter(|&rows| rows > 0 && rows <= SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD)
                .max()
                .map(|rows| table.added_ms(rows))
        });
        let input = Nvfp4FusedMoeKernelInput {
            num_tokens: batch.gathered_tokens,
        };
        for op in &self.fused_moe {
            match exposed_ms {
                None => push(op, input.clone(), ev),
                Some(added) => {
                    let mut metrics = op.kernel.eval(&input);
                    metrics.m.time_ms = (f64::from(metrics.m.time_ms) + added) as f32;
                    metrics.coverage = metrics.coverage | CoverageFlags::EXTRAPOLATED;
                    ev.push(metrics, || input.clone().into());
                }
            }
        }
        match &self.exchange {
            Exchange::Measured { combine, .. } => push(
                combine,
                MoeEpCollectiveKernelInput {
                    per_rank_tokens: tokens.clone(),
                },
                ev,
            ),
            Exchange::Fixed(exchange) => exchange.eval_combine(&tokens, ev),
        }
        for &rows in &tokens {
            self.shared_expert.eval_or_zero(
                &Glm53MlpLocalWorkletInput { num_tokens: rows },
                rows > SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD,
                ev,
            );
        }
        if let Some(glue) = &self.combine_glue {
            for &rows in &tokens {
                push(glue, ElementwiseKernelInput { num_tokens: rows }, ev);
            }
        }
    }
}

pub struct Glm53FlashDpAttnEpModel {
    pub name: String,
    pub dp_size: u16,
    /// KDA-only attention TP group size; 1 is plain DP attention.
    pub kda_tp_gpus: u16,
    pub max_model_len: u32,
    /// Query rows each decode request presents: 1, or `k + 1` for a
    /// speculative verify.
    pub verify_width: u32,
    cudagraph_capture_sizes: Vec<u32>,
    fixed_exchange: bool,
    embedding: Op<ElementwiseKernel>,
    hc_expand: Op<ElementwiseKernel>,
    groups: Vec<LayerGroup>,
    terminal_post: MhcTerminalPostOp,
    hc_contract_mean: Op<ElementwiseKernel>,
    final_norm: Op<RmsNormKernel>,
    lm_head: Op<SingleGemmKernel>,
    kv_cache_layout: KvCacheLayout,
    /// The RecoverSSM commit a verify graph's KDA layers need, per owner rank;
    /// `None` at width 1.
    pub(crate) kda_commit: Option<tp_arch::KdaCommitPlan>,
    /// A verify graph whose KDA layers store a state per position and so need
    /// no commit.
    pub(crate) kda_commitless_verify: bool,
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
    // Each owner rank holds its requests' whole KV and all 64 KDA heads of
    // their state (a KDA group's ranks hold equal head slices of equal request
    // counts): the cache and the commit are the TP1 ones, at the verify width.
    let mut tp1_cfg = local.clone();
    tp1_cfg.kda = cfg.kda_tp1.clone();
    tp1_cfg.kda.decode_query_len = local.kda.decode_query_len;
    tp1_cfg.kda.ssm_state_dtype = local.kda.ssm_state_dtype;
    tp1_cfg.kda.spec_verify = local.kda.spec_verify;
    let tp1 = tp_arch::resolve_configs(&tp1_cfg);
    let dsa_layers = model_cfg.num_dsa_layers();
    let kda_layers = model_cfg.kda_layers.len() as u32;
    let kda_state_bytes = tp_arch::kda_layer_state_bytes_per_request(&tp1);
    let kv_cache_layout = tp_arch::hybrid_kv_cache_layout(
        model_cfg,
        kda_state_bytes,
        dsa_layers,
        kda_layers,
        kda_layers.div_ceil(dsa_layers.max(1)),
        tp1.kda.state_pages_per_request,
        tp_arch::hybrid_block_tokens(kda_state_bytes, u64::from(model_cfg.kv_lora_rank)),
        1,
    );
    let terminal_post = {
        let post = MhcTerminalPostConfig {
            mhc: local.mhc.clone(),
            pre_backends: local.mhc.backends.clone(),
            fused_backends: local.mhc.backends.clone(),
        };
        let name = format!("{n}.final_mhc_post");
        match &cfg.mhc_post_addend {
            // The last layer is an MoE layer: its add moves into this post.
            Some(sum) if !model_cfg.is_dense(model_cfg.num_layers - 1) => {
                MhcTerminalPostOp::build_after_moe_sum(name, post, sum.clone(), bridge)?
            }
            _ => MhcTerminalPostOp::build(name, post, bridge)?,
        }
    };
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
        terminal_post,
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
        kda_tp_gpus: cfg.parallel.kda_tp_gpus,
        max_model_len: cfg.parallel.max_model_len,
        verify_width: tp_arch::verify_width(local)?,
        cudagraph_capture_sizes: {
            let mut sizes = cfg.parallel.cudagraph_capture_sizes.clone();
            sizes.sort_unstable();
            sizes.dedup();
            sizes
        },
        fixed_exchange: cfg.fixed_exchange.is_some(),
        kv_cache_layout,
        kda_commit: tp_arch::kda_commit_plan(&tp1),
        kda_commitless_verify: tp_arch::commitless_verify(&tp1),
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
    let attn_boundary = match &cfg.mhc_post_addend {
        _ if group.opens_stream => AttnBoundary::Mhc(Boundary::Pre(atomic(
            p,
            "attn_mhc_pre",
            local.mhc.clone(),
            MhcPreRmsNormKernel::build,
            bridge,
        )?)),
        Some(sum) if follows_moe(&local.model, group) => AttnBoundary::SumAfterMoe(atomic(
            p,
            "attn_mhc_sum_post_pre",
            sum.clone(),
            MhcFusedSumPostPreRmsNormKernel::build,
            bridge,
        )?),
        _ => AttnBoundary::Mhc(Boundary::Fused(atomic(
            p,
            "attn_mhc_post_pre",
            local.mhc_fused.clone(),
            MhcFusedPostPreRmsNormKernel::build,
            bridge,
        )?)),
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
    let kda_gather_scatter = match (&cfg.kda_gather_scatter, group.attn) {
        (Some(pair), AttnKind::Kda) => Some(GroupGatherScatter::build(p, pair.clone(), bridge)?),
        _ => None,
    };
    let ffn = match group.ffn {
        FfnKind::Dense => Ffn::Dense(Glm53MlpLocalWorklet::build(
            format!("{p}.dense_ffn"),
            resolved.local.dense_ffn.clone(),
            bridge,
        )?),
        FfnKind::Moe => Ffn::Moe(build_moe(format!("{p}.moe"), resolved, bridge)?),
    };
    Ok(LayerGroup {
        label: group.label.clone(),
        layers: group.layers.clone(),
        attn_boundary,
        attn,
        kda_gather_scatter,
        ffn_boundary: atomic(
            p,
            "ffn_mhc_post_pre",
            local.mhc_fused.clone(),
            MhcFusedPostPreRmsNormKernel::build,
            bridge,
        )?,
        ffn,
        mhc_fused_saving_us: cfg.parallel.mhc_fused_saving_us,
    })
}

fn build_moe(
    moe: String,
    resolved: &Glm53FlashDpAttnEpResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<MoeBlock, BuildError> {
    let cfg = &resolved.raw_cfg;
    let local = &cfg.local;
    let m = moe.as_str();
    // Built in launch order: router, input copy, shared expert, select,
    // dispatch quant, dispatch, experts, combine, combine add.
    let router = Glm53MoeRouterLocalWorklet::build(
        format!("{m}.router"),
        Glm53MoeRouterLocalWorklet::resolve_config(&local.router),
        cfg.parallel.router_gate_once,
        bridge,
    )?;
    let input_glue = cfg
        .parallel
        .moe_input_copy
        .then(|| {
            atomic(
                m,
                "input_glue",
                local.moe_input_glue.clone(),
                ElementwiseKernel::build,
                bridge,
            )
        })
        .transpose()?;
    let shared_expert = Glm53MlpLocalWorklet::build(
        format!("{m}.shared_expert"),
        resolved.local.shared_expert.clone(),
        bridge,
    )?;
    let router_select = cfg
        .router_select
        .clone()
        .map(|select| atomic(m, "router_select", select, ElementwiseKernel::build, bridge))
        .transpose()?;
    let dispatch_quant = Glm53ActivationQuant::build(
        m,
        "dispatch_quant",
        resolved.routed[0].input_quant.clone(),
        bridge,
    )?;
    let fixed = cfg
        .fixed_exchange
        .as_ref()
        .map(|fixed| FixedExpertExchange::build(m, fixed.clone(), bridge))
        .transpose()?;
    let dispatch = match (&fixed, cfg.dispatch.clone()) {
        (Some(_), _) => None,
        (None, Glm53FlashDispatchConfig::Fp8(dispatch)) => Some(Dispatch::Fp8(atomic(
            m,
            "dispatch_all_gather",
            dispatch,
            MoeEpAllGatherKernel::build,
            bridge,
        )?)),
        (None, Glm53FlashDispatchConfig::Nvfp4(dispatch)) => Some(Dispatch::Nvfp4(atomic(
            m,
            "dispatch_all_gather",
            dispatch,
            MoeEpQuantizedAllGatherKernel::build,
            bridge,
        )?)),
    };
    let fused_moe = resolved
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
        .collect::<std::result::Result<_, _>>()?;
    let exchange = match (fixed, dispatch) {
        (Some(fixed), _) => Exchange::Fixed(fixed),
        (None, Some(dispatch)) => Exchange::Measured {
            dispatch,
            combine: atomic(
                m,
                "combine_reduce_scatter",
                cfg.combine.clone(),
                MoeEpReduceScatterKernel::build,
                bridge,
            )?,
        },
        (None, None) => unreachable!("a measured exchange builds its dispatch"),
    };
    let combine_glue = cfg
        .mhc_post_addend
        .is_none()
        .then(|| {
            atomic(
                m,
                "combine_glue",
                local.moe_combine_glue.clone(),
                ElementwiseKernel::build,
                bridge,
            )
        })
        .transpose()?;
    Ok(MoeBlock {
        router,
        input_glue,
        shared_expert,
        router_select,
        dispatch_quant,
        exchange,
        fused_moe,
        shared_exposed: cfg.parallel.moe_shared_exposed.clone(),
        combine_glue,
        name: moe,
    })
}

impl Glm53FlashDpAttnEpModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let root = self.compile_root(&mut builder);
        builder.finish(root)
    }

    /// The whole forward as one node in `builder`, so a speculative model can
    /// compile it beside its draft stage. [`Self::eval_into`] fills it.
    pub(crate) fn compile_root(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let dp = usize::from(self.dp_size);
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
        let kda_groups = usize::from(self.dp_size / self.kda_tp_gpus);
        for group in &self.groups {
            children.push(group.compile(dp, kda_groups, builder));
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
        CostNode::Labeled {
            label: format!(
                "{} (Glm53FlashDpAttnEpModel) [attention DP{}/TP1{}; MoE EP{} {}; \
                 timing_context<={}]",
                self.name,
                self.dp_size,
                if self.kda_tp_gpus > 1 {
                    format!(
                        " (KDA layers TP{} over groups of owner ranks)",
                        self.kda_tp_gpus
                    )
                } else {
                    String::new()
                },
                self.dp_size,
                if self.fixed_exchange {
                    "fixed-pattern all-gather + reduce-scatter"
                } else {
                    "all_gatherv/reduce_scatterv"
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
            self.kda_tp_gpus,
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
            self.kda_tp_gpus,
            self.max_model_len,
            self.verify_width,
            &self.cudagraph_capture_sizes,
        )
        .map(|_| ())
    }

    /// One DP rank's cache for a request (the rank that owns it): the DSA
    /// layers' KV and all 64 KDA heads of 34 layers.
    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout.clone()
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
    /// dummy batch of an idle rank, or the common graph size.
    rows: u32,
    /// lm_head rows: one per prefill request plus every verify row of a
    /// decode request (none for a dummy batch).
    logits_rows: u32,
    kda: Glm53KdaAttnLocalWorkletInput,
    dsa: Glm53DsaAttnLocalWorkletInput,
}

/// One KDA tensor-parallel group under `kda_tp_gpus > 1`.
struct KdaGroupBatch {
    /// Each member rank's own rows.
    member_rows: Vec<u32>,
    /// The members' requests together.
    kda: Glm53KdaAttnLocalWorkletInput,
}

struct NormalizedBatch {
    ranks: Vec<RankBatch>,
    /// Empty unless `kda_tp_gpus > 1`.
    kda_groups: Vec<KdaGroupBatch>,
    /// Rows every EP rank's fused MoE reads after the all-gather.
    gathered_tokens: u32,
}

fn normalize_input(
    input: &UnifiedArchInput,
    dp_size: u16,
    kda_tp_gpus: u16,
    max_model_len: u32,
    verify_width: u32,
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
                "rank {rank} decode_tokens {} must equal decode_kv_lens length x verify \
                 width {verify_width}",
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
            // vLLM's target logits: one row per prefill request plus every
            // verify row of a decode request (`num_reqs + num_draft_tokens`).
            logits_rows: group.prefill_chunk_pairs.len() as u32 + decode_tokens,
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
    let kda_groups = if kda_tp_gpus > 1 {
        ranks
            .chunks(usize::from(kda_tp_gpus))
            .map(|members| KdaGroupBatch {
                member_rows: members.iter().map(|rank| rank.rows).collect(),
                kda: Glm53KdaAttnLocalWorkletInput {
                    prefill_sequence_lengths: members
                        .iter()
                        .flat_map(|rank| rank.kda.prefill_sequence_lengths.iter().copied())
                        .collect(),
                    decode_batch_size: members.iter().map(|rank| rank.kda.decode_batch_size).sum(),
                },
            })
            .collect()
    } else {
        Vec::new()
    };
    let gathered_tokens = ranks
        .iter()
        .try_fold(0_u32, |sum, rank| sum.checked_add(rank.rows))
        .ok_or("gathered token sum overflows u32")?;
    Ok(NormalizedBatch {
        ranks,
        kda_groups,
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
        Glm53FlashDpAttnEpParallel::new(
            dp_size,
            131072,
            "NVIDIA B200".into(),
            Vec::new(),
            Default::default(),
        )
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
            131072,
            1,
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
        assert!(normalize_input(&bad, 4, 1, 131072, 1, &[]).is_err());
        let empty = input(vec![ArchGroupInput::default(); 4]);
        assert!(normalize_input(&empty, 4, 1, 131072, 1, &[]).is_err());
        let three = input(vec![prefill(0, 8); 3]);
        assert!(normalize_input(&three, 4, 1, 131072, 1, &[]).is_err());
        let long = input(vec![
            prefill(131000, 100),
            prefill(0, 8),
            prefill(0, 8),
            prefill(0, 8),
        ]);
        assert!(normalize_input(&long, 4, 1, 131072, 1, &[]).is_err());
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
