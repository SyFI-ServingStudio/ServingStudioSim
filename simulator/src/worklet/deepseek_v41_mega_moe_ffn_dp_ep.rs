//! DeepSeek-V4.1-Flash MoE FFN as DeepInfer runs it under DP attention + EP
//! experts (DeepInfer a679bb8, `model_executor/models/v4_modules/mlp.py`
//! `mega_moe_forward`, 528-559), one layer over every DP rank.
//!
//! ```text
//! per rank   entry mHC (DeepGEMM `mega_mhc`, `sf_layout = mega_moe`: its FP8
//!            output and scales go straight into the DeepGEMM symmetric
//!            buffer) -> DeepGEMM mega gate (router GEMM, scoring, top-k,
//!            EPLB slot map in one launch)
//! group      DeepGEMM fp8_fp4_mega_moe: dispatch, routed experts, the fused
//!            shared expert and the combine in one kernel over the EP group
//! ```
//!
//! The mega MoE kernel returns routed + shared, so there is no separate shared
//! expert, all-gather, reduce-scatter or residual add (DeepInfer capture,
//! `wt-dsv41-deepinfer/logs/20261114_2_deepinfer_prefill/nsys`). Its EP group
//! synchronizes inside the kernel, so the group section is one leaf priced at
//! the busiest rank's token count, with every slot's rows drawn from the
//! routed demand under DeepInfer's EPLB placement.
//!
//! DeepInfer's `hc.py` has no mHC side stream, so there is no overlap
//! boundary: the entry is always the one `mega_mhc` launch.

use std::sync::Arc;

use super::deepseek_v41_common::eval_or_zero;
use super::deepseek_v41_moe_ffn_dp_ep::{
    DeepseekV41MoeFfnDpEpWorkletInput, DeepseekV41MoeFfnDpRankInput,
};
use crate::op::moe::MegaMoeLayersOp;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::expert_placement::SlotPlacement;
use crate::timing::kernels::{
    MegaGateKernel, MegaGateKernelConfig, MegaGateKernelInput, MegaMhcKernel,
    MegaMhcKernelConfig, MegaMhcKernelInput, MegaMoeKernelConfig, MegaMoeKernelInput,
};
use crate::timing::kernels::mega_mhc::SF_LAYOUT_MEGA_MOE;
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct DeepseekV41MegaMoeFfnDpEpWorkletConfig {
    pub ep_size: u32,
    pub hidden_size: Dim,
    pub hc_mult: u32,
    /// Logical routed experts: the gate's output width.
    pub num_logical_experts: Dim,
    /// Physical expert slots over the EP group: logical experts plus the EPLB
    /// redundant copies (DeepInfer `num_extra_experts`).
    pub num_physical_experts: u32,
    pub num_shared_experts: u32,
    pub top_k: u32,
    /// The router's scoring function (the model config's `scoring_func`).
    pub scoring_func: String,
    pub moe_intermediate_size: Dim,
    /// One rank's token budget: the mega MoE symmetric buffer's capacity.
    pub max_rank_tokens: u32,
    pub gpu_name: String,
    /// DeepGEMM `mega_mhc`.
    pub mhc_backends: Vec<&'static str>,
    pub mega_gate_backends: Vec<&'static str>,
    pub mega_moe_backends: Vec<&'static str>,
    /// Each layer this body stands for, its routed demand over the logical
    /// experts; with the placement it sets every rank's rows in that layer.
    pub layer_demands: Vec<ExpertDemand>,
    pub placement: SlotPlacement,
}

#[derive(Clone, Debug)]
pub struct DeepseekV41MegaMoeFfnDpEpWorkletResolved {
    pub raw_cfg: DeepseekV41MegaMoeFfnDpEpWorkletConfig,
    pub entry: MegaMhcKernelConfig,
    pub gate: MegaGateKernelConfig,
    /// One config per layer the body stands for.
    pub mega_moe: Vec<MegaMoeKernelConfig>,
}

pub struct DeepseekV41MegaMoeFfnDpEpWorklet {
    pub name: String,
    pub entry: Op<MegaMhcKernel>,
    pub gate: Op<MegaGateKernel>,
    pub mega_moe: MegaMoeLayersOp,
    resolved: DeepseekV41MegaMoeFfnDpEpWorkletResolved,
}

impl DeepseekV41MegaMoeFfnDpEpWorklet {
    pub fn resolve_config(
        cfg: &DeepseekV41MegaMoeFfnDpEpWorkletConfig,
    ) -> DeepseekV41MegaMoeFfnDpEpWorkletResolved {
        assert!(cfg.ep_size > 0, "ep_size must be positive");
        assert_eq!(
            cfg.num_physical_experts % cfg.ep_size,
            0,
            "physical expert slots must split evenly over the EP ranks"
        );
        assert!(
            cfg.num_physical_experts >= cfg.num_logical_experts.get(),
            "every logical expert needs a physical slot"
        );
        DeepseekV41MegaMoeFfnDpEpWorkletResolved {
            entry: MegaMhcKernelConfig {
                backends: cfg.mhc_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden_size.clone(),
                hc_mult: cfg.hc_mult,
                hidden_dtype: DType::Bf16,
                sf_layout: SF_LAYOUT_MEGA_MOE.to_string(),
            },
            gate: MegaGateKernelConfig {
                backends: cfg.mega_gate_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden_size.clone(),
                num_experts: cfg.num_logical_experts.get(),
                num_physical_experts: cfg.num_physical_experts,
                top_k: cfg.top_k,
                scoring_func: cfg.scoring_func.clone(),
                dtype: DType::Bf16,
                max_tokens_per_rank: cfg.max_rank_tokens,
            },
            mega_moe: cfg
                .layer_demands
                .iter()
                .map(|demand| MegaMoeKernelConfig {
                backends: cfg.mega_moe_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                ep_size: cfg.ep_size,
                hidden_size: cfg.hidden_size.clone(),
                intermediate_size: cfg.moe_intermediate_size.clone(),
                num_experts: cfg.num_physical_experts,
                num_shared_experts: cfg.num_shared_experts,
                top_k: cfg.top_k,
                // DeepInfer's `fp8_mlp` + `mxfp4` checkpoint path (mma fp8xfp4).
                activation_dtype: DType::Mxfp8E4m3,
                weight_dtype: DType::Mxfp4E2m1,
                max_tokens_per_rank: cfg.max_rank_tokens,
                expert_demand: demand.clone(),
                placement: cfg.placement,
            })
                .collect(),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: DeepseekV41MegaMoeFfnDpEpWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        macro_rules! op {
            ($kernel:ty, $cfg:expr, $suffix:literal) => {{
                let op_name = format!("{name}.{}", $suffix);
                Op::new(
                    op_name.clone(),
                    Arc::new(<$kernel>::build(op_name, $cfg, bridge)?),
                )
            }};
        }
        let r = resolved.clone();
        Ok(Self {
            entry: op!(MegaMhcKernel, r.entry, "entry.mega_mhc"),
            gate: op!(MegaGateKernel, r.gate, "routed.mega_gate"),
            mega_moe: MegaMoeLayersOp::build(format!("{name}.routed.mega_moe"), &r.mega_moe, bridge)?,
            name,
            resolved,
        })
    }

    /// One rank's launches before the group kernel.
    pub fn compile_local(&self, b: &mut CostTreeBuilder) -> Vec<CostNode> {
        vec![self.entry.compile(b), self.gate.compile(b)]
    }

    /// The group kernel.
    pub fn compile_exchange(&self, b: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} exchange [DeepGEMM mega MoE over EP{}: dispatch, {} physical expert slots, \
                 fused shared expert, combine; busiest rank]",
                self.name,
                self.resolved.raw_cfg.ep_size,
                self.resolved.raw_cfg.num_physical_experts
            ),
            child: Box::new(self.mega_moe.compile(b)),
        }
    }

    pub fn eval_local(&self, input: &DeepseekV41MoeFfnDpRankInput, ev: &mut Evaluator) {
        let rows = input.num_tokens;
        let zero = rows == 0;
        eval_or_zero(&self.entry, MegaMhcKernelInput { num_tokens: rows }, zero, ev);
        eval_or_zero(
            &self.gate,
            MegaGateKernelInput { num_tokens: rows },
            zero,
            ev,
        );
    }

    pub fn eval_exchange(&self, input: &DeepseekV41MoeFfnDpEpWorkletInput, ev: &mut Evaluator) {
        // Every rank's rows enter the kernel (padding included: DeepInfer
        // routes them like any other); the group waits for the busiest.
        let busiest = input
            .ranks
            .iter()
            .map(|rank| rank.num_tokens)
            .max()
            .unwrap_or(0);
        let group = MegaMoeKernelInput {
            num_tokens: busiest,
        };
        if busiest == 0 {
            self.mega_moe.eval_zero(&group, ev);
        } else {
            self.mega_moe.eval(&group, ev);
        }
    }
}
