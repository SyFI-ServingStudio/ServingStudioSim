//! GLM-5.3-Flash MoE router and routed experts on one EP rank.
//!
//! The router gate is `GateLinear` (BF16 weight, fp32 logits) and vLLM computes
//! it twice per layer (`glm5next/nvidia/model.py:264` and `moe_runner.py`),
//! so [`Glm53MoeRouterLocalWorklet`] has two identical leaves.
//!
//! [`Glm53RoutedMoeLocalWorklet`] is the routed path through TRT-LLM's
//! DeepSeek-FP8 block-scale fused MoE: a row-major UE8M0 per-token-group quant
//! of the routed input, then `trtllm_fp8_block_scale_moe` (routing, FC1,
//! activation, FC2, finalize as one measured boundary). One config per EP rank
//! comes from [`Glm53RoutedMoeLocalWorkletConfig::split_for_ep`], which ranks
//! the ranks' routed workloads from one global demand source.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{
    Fp8PerTokenGroupQuantKernel, Fp8PerTokenGroupQuantKernelConfig,
    Fp8PerTokenGroupQuantKernelInput, GemmFp32OutputKernel, GemmFp32OutputKernelConfig,
    GemmFp32OutputKernelInput, Nvfp4FusedMoeKernel, Nvfp4FusedMoeKernelConfig,
    Nvfp4FusedMoeKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, push_or_zero};

/// TRT-LLM's FP8 block MoE reads fp32 scales in row-major order.
pub const ROW_MAJOR_SCALE_FORMAT: &str = "ue8m0_row_major";
const QUANT_GROUP_SIZE: u32 = 128;

#[derive(Clone, Debug)]
pub struct Glm53MoeRouterLocalWorkletConfig {
    pub hidden: Dim,
    pub num_experts: Dim,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub gemm_backends: Vec<&'static str>,
}

#[derive(Clone, Debug, Default)]
pub struct Glm53MoeRouterLocalWorkletInput {
    pub num_tokens: u32,
}

pub struct Glm53MoeRouterLocalWorklet {
    pub name: String,
    pub gate: Op<GemmFp32OutputKernel>,
    pub gate_recompute: Op<GemmFp32OutputKernel>,
}

impl Glm53MoeRouterLocalWorklet {
    pub fn resolve_config(cfg: &Glm53MoeRouterLocalWorkletConfig) -> GemmFp32OutputKernelConfig {
        GemmFp32OutputKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: cfg.num_experts.clone(),
            k: cfg.hidden.clone(),
            input_dtype: cfg.activation_dtype,
        }
    }

    pub fn build(
        name: String,
        resolved: GemmFp32OutputKernelConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let n = name.as_str();
        Ok(Self {
            gate: atomic(
                n,
                "gate",
                resolved.clone(),
                GemmFp32OutputKernel::build,
                bridge,
            )?,
            gate_recompute: atomic(
                n,
                "gate_recompute",
                resolved,
                GemmFp32OutputKernel::build,
                bridge,
            )?,
            name,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (Glm53MoeRouterLocalWorklet) [gate computed twice]",
                self.name
            ),
            child: Box::new(CostNode::Sum(vec![
                self.gate.compile(builder),
                self.gate_recompute.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Glm53MoeRouterLocalWorkletInput, ev: &mut Evaluator) {
        let rows = GemmFp32OutputKernelInput {
            m: input.num_tokens,
        };
        let zero = input.num_tokens == 0;
        push_or_zero(&self.gate, rows.clone(), zero, ev);
        push_or_zero(&self.gate_recompute, rows, zero, ev);
    }
}

#[derive(Clone, Debug)]
pub struct Glm53RoutedMoeLocalWorkletConfig {
    pub hidden: Dim,
    pub moe_intermediate: Dim,
    pub num_experts: Dim,
    pub ep_size: u16,
    pub top_k: u32,
    pub n_group: u32,
    pub topk_group: u32,
    pub routed_scaling_numerator: u32,
    pub routed_scaling_denominator: u32,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub quant_backends: Vec<&'static str>,
    pub fused_moe_backends: Vec<&'static str>,
    pub expert_demand: ExpertDemand,
    pub folded_rank_position: u32,
}

impl Glm53RoutedMoeLocalWorkletConfig {
    /// One config per EP rank, ranked by active-expert workload from the same
    /// global routing evidence (as `Nvfp4MoeLocalWorkletConfig::split_for_ep`).
    pub fn split_for_ep(mut template: Self, demand: ExpertDemand) -> Vec<Self> {
        let ep = usize::from(template.ep_size);
        assert!(ep > 0, "ep_size must be non-zero");
        assert_eq!(
            template.num_experts.get() as usize % ep,
            0,
            "experts must evenly partition EP"
        );
        assert_eq!(
            demand.num_experts(),
            template.num_experts.get() as usize,
            "demand source width must match num_experts"
        );
        template.expert_demand = demand;
        (0..ep)
            .map(|position| {
                let mut rank = template.clone();
                rank.folded_rank_position = position as u32;
                rank
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct Glm53RoutedMoeLocalWorkletResolved {
    pub raw_cfg: Glm53RoutedMoeLocalWorkletConfig,
    pub input_quant: Fp8PerTokenGroupQuantKernelConfig,
    pub fused_moe: Nvfp4FusedMoeKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Glm53RoutedMoeLocalWorkletInput {
    pub num_tokens: u32,
}

pub struct Glm53RoutedMoeLocalWorklet {
    pub name: String,
    pub input_quant: Op<Fp8PerTokenGroupQuantKernel>,
    pub fused_moe: Op<Nvfp4FusedMoeKernel>,
    resolved: Glm53RoutedMoeLocalWorkletResolved,
}

impl Glm53RoutedMoeLocalWorklet {
    pub fn resolve_config(
        cfg: &Glm53RoutedMoeLocalWorkletConfig,
    ) -> Glm53RoutedMoeLocalWorkletResolved {
        let local_experts = cfg.num_experts.get() / u32::from(cfg.ep_size);
        Glm53RoutedMoeLocalWorkletResolved {
            input_quant: Fp8PerTokenGroupQuantKernelConfig {
                backends: cfg.quant_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden.clone(),
                group_size: QUANT_GROUP_SIZE,
                input_dtype: cfg.activation_dtype,
                scale_format: ROW_MAJOR_SCALE_FORMAT.to_string(),
            },
            fused_moe: Nvfp4FusedMoeKernelConfig {
                backends: cfg.fused_moe_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden.clone(),
                intermediate_size: cfg.moe_intermediate.clone(),
                num_experts: cfg.num_experts.clone(),
                num_local_experts: local_experts.into(),
                top_k: cfg.top_k,
                input_dtype: cfg.activation_dtype,
                weight_format: "fp8_e4m3_block".to_string(),
                group_size: QUANT_GROUP_SIZE,
                routing_method: "deepseek_v3".to_string(),
                n_group: cfg.n_group,
                topk_group: cfg.topk_group,
                routed_scaling_numerator: cfg.routed_scaling_numerator,
                routed_scaling_denominator: cfg.routed_scaling_denominator,
                expert_demand: cfg.expert_demand.clone(),
                folded_rank_position: cfg.folded_rank_position,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Glm53RoutedMoeLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            input_quant: atomic(
                n,
                "input_quant",
                r.input_quant,
                Fp8PerTokenGroupQuantKernel::build,
                bridge,
            )?,
            fused_moe: atomic(
                n,
                "fused_moe",
                r.fused_moe,
                Nvfp4FusedMoeKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (Glm53RoutedMoeLocalWorklet) [EP{} ranked rank {}; {} local experts, top-{}]",
                self.name,
                cfg.ep_size,
                cfg.folded_rank_position,
                self.resolved.fused_moe.num_local_experts,
                cfg.top_k
            ),
            child: Box::new(CostNode::Sum(vec![
                self.input_quant.compile(builder),
                self.fused_moe.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Glm53RoutedMoeLocalWorkletInput, ev: &mut Evaluator) {
        let zero = input.num_tokens == 0;
        push_or_zero(
            &self.input_quant,
            Fp8PerTokenGroupQuantKernelInput {
                num_tokens: input.num_tokens,
            },
            zero,
            ev,
        );
        push_or_zero(
            &self.fused_moe,
            Nvfp4FusedMoeKernelInput {
                num_tokens: input.num_tokens,
            },
            zero,
            ev,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::routing::RoutingDistribution;

    fn cfg() -> Glm53RoutedMoeLocalWorkletConfig {
        Glm53RoutedMoeLocalWorkletConfig {
            hidden: 4096.into(),
            moe_intermediate: 2048.into(),
            num_experts: 288.into(),
            ep_size: 4,
            top_k: 8,
            n_group: 1,
            topk_group: 1,
            routed_scaling_numerator: 5,
            routed_scaling_denominator: 2,
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".into(),
            quant_backends: vec!["vllm_fork_cuda"],
            fused_moe_backends: vec!["flashinfer_trtllm_fp8_block_sm100"],
            expert_demand: ExpertDemand::popularity(&RoutingDistribution::uniform(288), 1),
            folded_rank_position: 0,
        }
    }

    #[test]
    fn routed_path_is_the_fp8_block_trtllm_moe_with_row_major_input_quant() {
        let r = Glm53RoutedMoeLocalWorklet::resolve_config(&cfg());
        assert_eq!(r.fused_moe.num_local_experts.get(), 72);
        assert_eq!(r.fused_moe.weight_format, "fp8_e4m3_block");
        assert_eq!(r.fused_moe.routing_method, "deepseek_v3");
        assert_eq!(r.input_quant.scale_format, ROW_MAJOR_SCALE_FORMAT);
    }

    #[test]
    fn split_for_ep_assigns_every_ranked_position() {
        let demand = ExpertDemand::popularity(&RoutingDistribution::uniform(288), 42);
        let ranks = Glm53RoutedMoeLocalWorkletConfig::split_for_ep(cfg(), demand);
        let positions: Vec<_> = ranks.iter().map(|rank| rank.folded_rank_position).collect();
        assert_eq!(positions, [0, 1, 2, 3]);
    }
}
