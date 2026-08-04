//! vLLM-aligned local MoE expert compute: BF16 activation block quantization +
//! TRT-LLM blockscale grouped GEMM for gate-up and down, with SwiGLU between.

use std::sync::Arc;

use crate::op::moe::{
    GroupedFp8GemmWithQuantConfig, GroupedFp8GemmWithQuantInput, GroupedFp8GemmWithQuantOp,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, Fp8BlockQuantKernelConfig,
    Fp8BlockscaleGroupedGemmKernelConfig,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct VllmFp8MoeExpertComputeLocalWorkletConfig {
    pub hidden: Dim,
    pub moe_intermediate: Dim,
    pub num_experts: Dim,
    pub ep_size: u16,
    pub top_k: u32,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub act_backends: Vec<&'static str>,
    pub fp8_quant_backends: Vec<&'static str>,
    pub fp8_grouped_gemm_backends: Vec<&'static str>,
    pub local_ppm: Vec<u32>,
}

impl VllmFp8MoeExpertComputeLocalWorkletConfig {
    pub fn split_for_ep(mut template: Self, global_ppm: &[u32]) -> Vec<Self> {
        let ep_size = usize::from(template.ep_size);
        assert!(ep_size > 0, "ep_size must be non-zero");
        let num_experts = template.num_experts.get() as usize;
        assert_eq!(global_ppm.len(), num_experts);
        assert_eq!(num_experts % ep_size, 0);
        let experts_per_rank = num_experts / ep_size;
        template.local_ppm.clear();
        (0..ep_size)
            .map(|rank| {
                let mut rank_config = template.clone();
                let start = rank * experts_per_rank;
                rank_config.local_ppm = global_ppm[start..start + experts_per_rank].to_vec();
                rank_config
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct VllmFp8MoeExpertComputeLocalWorkletResolved {
    pub raw_cfg: VllmFp8MoeExpertComputeLocalWorkletConfig,
    pub gate_up: GroupedFp8GemmWithQuantConfig,
    pub act: ElementwiseKernelConfig,
    pub down: GroupedFp8GemmWithQuantConfig,
    pub experts_per_gpu: Dim,
}

#[derive(Clone, Debug, Default)]
pub struct VllmFp8MoeExpertComputeLocalWorkletInput {
    pub global_expert_selections: u32,
}

pub struct VllmFp8MoeExpertComputeLocalWorklet {
    pub name: String,
    pub gate_up: GroupedFp8GemmWithQuantOp,
    pub act: Op<ElementwiseKernel>,
    pub down: GroupedFp8GemmWithQuantOp,
    resolved: VllmFp8MoeExpertComputeLocalWorkletResolved,
}

impl VllmFp8MoeExpertComputeLocalWorklet {
    pub fn resolve_config(
        cfg: &VllmFp8MoeExpertComputeLocalWorkletConfig,
    ) -> VllmFp8MoeExpertComputeLocalWorkletResolved {
        assert_eq!(cfg.activation_dtype, DType::Bf16);
        let ep_size = u32::from(cfg.ep_size);
        assert!(ep_size > 0, "ep_size must be non-zero");
        assert_eq!(cfg.num_experts.get() % ep_size, 0);
        let experts_per_gpu = cfg.num_experts.clone() / Dim::param("ep", ep_size);
        assert_eq!(cfg.local_ppm.len(), experts_per_gpu.get() as usize);
        let quant_config = |hidden_size: Dim| Fp8BlockQuantKernelConfig {
            backends: cfg.fp8_quant_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden_size,
            num_problems: experts_per_gpu.clone(),
            input_dtype: cfg.activation_dtype,
        };
        let gemm_config = |n: Dim, k: Dim| Fp8BlockscaleGroupedGemmKernelConfig {
            backends: cfg.fp8_grouped_gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n,
            k,
            dtype: DType::Fp8E4m3,
            experts_per_token: cfg.top_k,
            local_ppm: cfg.local_ppm.clone(),
        };
        let bytes = Dim::param("activation_bytes", cfg.activation_dtype.size_bytes());
        VllmFp8MoeExpertComputeLocalWorkletResolved {
            gate_up: GroupedFp8GemmWithQuantConfig {
                quant: quant_config(cfg.hidden.clone()),
                gemm: gemm_config(2 * cfg.moe_intermediate.clone(), cfg.hidden.clone()),
            },
            act: ElementwiseKernelConfig {
                backends: cfg.act_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: 2 * cfg.moe_intermediate.clone() * bytes.clone(),
                output_bytes_per_token: cfg.moe_intermediate.clone() * bytes,
            },
            down: GroupedFp8GemmWithQuantConfig {
                quant: quant_config(cfg.moe_intermediate.clone()),
                gemm: gemm_config(cfg.hidden.clone(), cfg.moe_intermediate.clone()),
            },
            experts_per_gpu,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: VllmFp8MoeExpertComputeLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let gate_up = GroupedFp8GemmWithQuantOp::build(
            format!("{name}.gate_up"),
            resolved.gate_up.clone(),
            bridge,
        )?;
        let act_name = format!("{name}.activation");
        let act = Op::new(
            act_name.clone(),
            Arc::new(ElementwiseKernel::build(
                act_name,
                resolved.act.clone(),
                bridge,
            )?),
        );
        let down = GroupedFp8GemmWithQuantOp::build(
            format!("{name}.down"),
            resolved.down.clone(),
            bridge,
        )?;
        Ok(Self {
            name,
            gate_up,
            act,
            down,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (VllmFp8MoeExpertComputeLocalWorklet) [local; experts={:?}]",
                self.name, self.resolved.experts_per_gpu
            ),
            child: Box::new(CostNode::Sum(vec![
                self.gate_up.compile(builder),
                self.act.compile(builder),
                self.down.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &VllmFp8MoeExpertComputeLocalWorkletInput, ev: &mut Evaluator) {
        let global_expert_selections = input.global_expert_selections;
        self.gate_up.eval(
            &GroupedFp8GemmWithQuantInput {
                global_expert_selections,
            },
            ev,
        );
        let local_ppm_sum: u64 = self
            .resolved
            .raw_cfg
            .local_ppm
            .iter()
            .map(|&x| u64::from(x))
            .sum();
        let local_routed_tokens =
            ((u64::from(global_expert_selections) * local_ppm_sum + 999_999) / 1_000_000) as u32;
        self.act.eval(
            &ElementwiseKernelInput {
                num_tokens: local_routed_tokens,
            },
            ev,
        );
        self.down.eval(
            &GroupedFp8GemmWithQuantInput {
                global_expert_selections,
            },
            ev,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_two_quantized_blockscale_gemms() {
        let resolved = VllmFp8MoeExpertComputeLocalWorklet::resolve_config(
            &VllmFp8MoeExpertComputeLocalWorkletConfig {
                hidden: 4096.into(),
                moe_intermediate: 1536.into(),
                num_experts: 8.into(),
                ep_size: 2,
                top_k: 2,
                activation_dtype: DType::Bf16,
                gpu_name: "H100".into(),
                act_backends: vec!["triton"],
                fp8_quant_backends: vec!["flashinfer_trtllm"],
                fp8_grouped_gemm_backends: vec!["flashinfer_trtllm"],
                local_ppm: vec![125_000; 4],
            },
        );
        assert_eq!(resolved.gate_up.gemm.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.down.gemm.dtype, DType::Fp8E4m3);
    }
}
