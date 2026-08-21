//! Rank-local NVFP4 routed MoE section observed on B200.
//!
//! vLLM first quantizes the full BF16 hidden-state batch, then invokes one
//! monolithic FlashInfer operation that owns routing, both expert GEMMs, and
//! finalize. Expert ownership is contiguous and derived from `ep_size`.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    Nvfp4MoeKernel, Nvfp4MoeKernelConfig, Nvfp4MoeKernelInput, Nvfp4QuantKernel,
    Nvfp4QuantKernelConfig, Nvfp4QuantKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct VllmNvfp4MoeLocalWorkletConfig {
    pub hidden: Dim,
    pub moe_intermediate: Dim,
    pub num_experts: Dim,
    pub ep_size: u16,
    pub ep_rank: u16,
    pub top_k: u32,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub quant_backends: Vec<&'static str>,
    pub moe_backends: Vec<&'static str>,
    pub weight_format: String,
    pub group_size: u32,
    pub routing_method: String,
    pub n_group: u32,
    pub topk_group: u32,
    pub routed_scaling_numerator: u32,
    pub routed_scaling_denominator: u32,
}

impl VllmNvfp4MoeLocalWorkletConfig {
    pub fn split_for_ep(mut template: Self) -> Vec<Self> {
        assert!(template.ep_size > 0, "ep_size must be non-zero");
        assert_eq!(template.num_experts.get() % u32::from(template.ep_size), 0);
        template.ep_rank = 0;
        (0..template.ep_size)
            .map(|ep_rank| {
                let mut rank = template.clone();
                rank.ep_rank = ep_rank;
                rank
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct VllmNvfp4MoeLocalWorkletResolved {
    pub raw_cfg: VllmNvfp4MoeLocalWorkletConfig,
    pub quant: Nvfp4QuantKernelConfig,
    pub moe: Nvfp4MoeKernelConfig,
    pub experts_per_rank: Dim,
}

#[derive(Clone, Debug, Default)]
pub struct VllmNvfp4MoeLocalWorkletInput {
    pub num_tokens: u32,
}

pub struct VllmNvfp4MoeLocalWorklet {
    pub name: String,
    pub quant: Op<Nvfp4QuantKernel>,
    pub moe: Op<Nvfp4MoeKernel>,
    resolved: VllmNvfp4MoeLocalWorkletResolved,
}

impl VllmNvfp4MoeLocalWorklet {
    pub fn resolve_config(
        cfg: &VllmNvfp4MoeLocalWorkletConfig,
    ) -> VllmNvfp4MoeLocalWorkletResolved {
        assert_eq!(cfg.activation_dtype, DType::Bf16);
        assert!(cfg.ep_size > 0, "ep_size must be non-zero");
        assert!(cfg.ep_rank < cfg.ep_size, "ep_rank must be below ep_size");
        assert_eq!(cfg.num_experts.get() % u32::from(cfg.ep_size), 0);
        let experts_per_rank = cfg.num_experts.clone() / Dim::param("ep", u32::from(cfg.ep_size));
        let local_expert_offset = u32::from(cfg.ep_rank)
            .checked_mul(experts_per_rank.get())
            .expect("local expert offset must fit u32");
        VllmNvfp4MoeLocalWorkletResolved {
            quant: Nvfp4QuantKernelConfig {
                backends: cfg.quant_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden.clone(),
                group_size: cfg.group_size,
                input_dtype: cfg.activation_dtype,
                scale_format: "linear_e4m3".to_string(),
            },
            moe: Nvfp4MoeKernelConfig {
                backends: cfg.moe_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden.clone(),
                intermediate_size: cfg.moe_intermediate.clone(),
                num_experts: cfg.num_experts.clone(),
                num_local_experts: experts_per_rank.clone(),
                local_expert_offset,
                top_k: cfg.top_k,
                input_dtype: cfg.activation_dtype,
                weight_format: cfg.weight_format.clone(),
                group_size: cfg.group_size,
                routing_method: cfg.routing_method.clone(),
                n_group: cfg.n_group,
                topk_group: cfg.topk_group,
                routed_scaling_numerator: cfg.routed_scaling_numerator,
                routed_scaling_denominator: cfg.routed_scaling_denominator,
            },
            experts_per_rank,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: VllmNvfp4MoeLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let quant_name = format!("{name}.input_quant");
        let moe_name = format!("{name}.fused_moe");
        let quant = Op::new(
            quant_name.clone(),
            Arc::new(Nvfp4QuantKernel::build(
                quant_name,
                resolved.quant.clone(),
                bridge,
            )?),
        );
        let moe = Op::new(
            moe_name.clone(),
            Arc::new(Nvfp4MoeKernel::build(
                moe_name,
                resolved.moe.clone(),
                bridge,
            )?),
        );
        Ok(Self {
            name,
            quant,
            moe,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (VllmNvfp4MoeLocalWorklet) [ep_rank={}/{}; experts={:?}]",
                self.name,
                self.resolved.raw_cfg.ep_rank,
                self.resolved.raw_cfg.ep_size,
                self.resolved.experts_per_rank,
            ),
            child: Box::new(CostNode::Sum(vec![
                self.quant.compile(builder),
                self.moe.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &VllmNvfp4MoeLocalWorkletInput, ev: &mut Evaluator) {
        self.quant.eval(
            &Nvfp4QuantKernelInput {
                num_tokens: input.num_tokens,
            },
            ev,
        );
        self.moe.eval(
            &Nvfp4MoeKernelInput {
                num_tokens: input.num_tokens,
            },
            ev,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template(ep_size: u16) -> VllmNvfp4MoeLocalWorkletConfig {
        VllmNvfp4MoeLocalWorkletConfig {
            hidden: 6144.into(),
            moe_intermediate: 2048.into(),
            num_experts: 256.into(),
            ep_size,
            ep_rank: 0,
            top_k: 8,
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".to_string(),
            quant_backends: vec!["vllm_cuda"],
            moe_backends: vec!["flashinfer_trtllm"],
            weight_format: "nvfp4_e2m1".to_string(),
            group_size: 16,
            routing_method: "minimax2".to_string(),
            n_group: 1,
            topk_group: 1,
            routed_scaling_numerator: 5,
            routed_scaling_denominator: 2,
        }
    }

    #[test]
    fn ep4_and_ep8_derive_distinct_contiguous_shards() {
        let ep4 = VllmNvfp4MoeLocalWorkletConfig::split_for_ep(template(4));
        let ep8 = VllmNvfp4MoeLocalWorkletConfig::split_for_ep(template(8));
        assert_eq!(ep4.len(), 4);
        assert_eq!(ep8.len(), 8);
        let ep4_last = VllmNvfp4MoeLocalWorklet::resolve_config(&ep4[3]);
        let ep8_last = VllmNvfp4MoeLocalWorklet::resolve_config(&ep8[7]);
        assert_eq!(ep4_last.experts_per_rank.get(), 64);
        assert_eq!(ep4_last.moe.local_expert_offset, 192);
        assert_eq!(ep8_last.experts_per_rank.get(), 32);
        assert_eq!(ep8_last.moe.local_expert_offset, 224);
    }
}
