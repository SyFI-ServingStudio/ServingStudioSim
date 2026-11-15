//! DeepSeek-V4.1-Flash MoE FFN as DeepInfer runs it under DP attention + EP
//! experts (DeepInfer a679bb8, `model_executor/models/v4_modules/mlp.py`
//! `mega_moe_forward`, 528-559), one layer over every DP rank.
//!
//! ```text
//! per rank   entry mHC (its FP8 output is written straight into the
//!            DeepGEMM symmetric buffer) -> DeepGEMM mega gate (router GEMM,
//!            scoring, top-k, EPLB slot map in one launch)
//! group      DeepGEMM fp8_fp4_mega_moe: dispatch, routed experts, the fused
//!            shared expert and the combine in one kernel over the EP group
//! ```
//!
//! The mega MoE kernel returns routed + shared, so there is no separate shared
//! expert, all-gather, reduce-scatter or residual add (DeepInfer capture,
//! `wt-dsv41-deepinfer/logs/20261114_2_deepinfer_prefill/nsys`). Its EP group
//! synchronizes inside the kernel, so the group section is one leaf priced at
//! the busiest rank's token count.
//!

use std::sync::Arc;

use super::deepseek_v41_common::{
    eval_or_zero, MhcOverlapBoundary, MhcOverlapBoundaryConfig, MhcOverlapBoundaryResolved,
};
use super::deepseek_v41_moe_ffn_dp_ep::{
    DeepseekV41MoeFfnDpEpWorkletInput, DeepseekV41MoeFfnDpRankInput,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    MegaGateKernel, MegaGateKernelConfig, MegaGateKernelInput, MegaMoeKernel, MegaMoeKernelConfig,
    MegaMoeKernelInput, MhcFusedPostPreRmsNormKernel, MhcRmsNormKernelConfig,
    MhcRmsNormKernelInput,
};
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
    pub mhc_backends: Vec<&'static str>,
    pub mhc_tilelang_backends: Vec<&'static str>,
    pub mega_gate_backends: Vec<&'static str>,
    pub mega_moe_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct DeepseekV41MegaMoeFfnDpEpWorkletResolved {
    pub raw_cfg: DeepseekV41MegaMoeFfnDpEpWorkletConfig,
    pub entry: MhcRmsNormKernelConfig,
    pub mhc_overlap: MhcOverlapBoundaryResolved,
    pub gate: MegaGateKernelConfig,
    pub mega_moe: MegaMoeKernelConfig,
}

pub struct DeepseekV41MegaMoeFfnDpEpWorklet {
    pub name: String,
    pub entry: Op<MhcFusedPostPreRmsNormKernel>,
    pub mhc_overlap: MhcOverlapBoundary,
    pub gate: Op<MegaGateKernel>,
    pub mega_moe: Op<MegaMoeKernel>,
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
            entry: MhcRmsNormKernelConfig {
                backends: cfg.mhc_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden_size.clone(),
                hc_mult: cfg.hc_mult,
                hidden_dtype: DType::Bf16,
            },
            mhc_overlap: MhcOverlapBoundary::resolve(&MhcOverlapBoundaryConfig {
                hidden_size: cfg.hidden_size.clone(),
                hc_mult: cfg.hc_mult,
                gpu_name: cfg.gpu_name.clone(),
                mhc_tilelang_backends: cfg.mhc_tilelang_backends.clone(),
                elementwise_backends: cfg.elementwise_backends.clone(),
            }),
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
            mega_moe: MegaMoeKernelConfig {
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
            },
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
            entry: op!(
                MhcFusedPostPreRmsNormKernel,
                r.entry,
                "entry.mega_mhc_post_pre"
            ),
            mhc_overlap: MhcOverlapBoundary::build(
                &format!("{name}.entry"),
                r.mhc_overlap,
                bridge,
            )?,
            gate: op!(MegaGateKernel, r.gate, "routed.mega_gate"),
            mega_moe: op!(MegaMoeKernel, r.mega_moe, "routed.mega_moe"),
            name,
            resolved,
        })
    }

    /// One rank's launches before the group kernel.
    pub fn compile_local(&self, b: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut leaves = vec![self.entry.compile(b)];
        leaves.extend(self.mhc_overlap.compile_entry(b));
        leaves.push(self.gate.compile(b));
        leaves
    }

    /// The group kernel, with every rank's mHC coefficient branch beside it.
    pub fn compile_exchange(&self, ranks: usize, b: &mut CostTreeBuilder) -> CostNode {
        // Slots are assigned in compile order and filled in `eval_exchange`
        // order: the group kernel first.
        let mega_moe = self.mega_moe.compile(b);
        let coefficients = CostNode::Max {
            overlap: 1.0,
            children: (0..ranks)
                .map(|_| self.mhc_overlap.compile_beside(Vec::new(), b))
                .collect(),
        };
        CostNode::Labeled {
            label: format!(
                "{} exchange [DeepGEMM mega MoE over EP{}: dispatch, {} physical expert slots, \
                 fused shared expert, combine; busiest rank]",
                self.name,
                self.resolved.raw_cfg.ep_size,
                self.resolved.raw_cfg.num_physical_experts
            ),
            child: Box::new(CostNode::Parallel {
                overlap: 1.0,
                children: vec![mega_moe, coefficients],
            }),
        }
    }

    pub fn eval_local(&self, input: &DeepseekV41MoeFfnDpRankInput, ev: &mut Evaluator) {
        let rows = input.num_tokens;
        let zero = rows == 0;
        let mhc_overlap = !zero && input.mhc_overlap;
        eval_or_zero(
            &self.entry,
            MhcRmsNormKernelInput { num_tokens: rows },
            zero || mhc_overlap,
            ev,
        );
        self.mhc_overlap.eval_entry(rows, mhc_overlap, ev);
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
        eval_or_zero(
            &self.mega_moe,
            MegaMoeKernelInput {
                num_tokens: busiest,
            },
            busiest == 0,
            ev,
        );
        for rank in &input.ranks {
            let live = rank.num_tokens > 0 && rank.mhc_overlap;
            self.mhc_overlap.eval_side(rank.num_tokens, live, false, ev);
        }
    }
}
