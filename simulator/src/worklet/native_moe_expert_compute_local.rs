//! Native local `MoE` expert compute: direct grouped gate-up GEMM, activation,
//! then direct grouped down GEMM. BF16 and native FP8 share this exact op graph;
//! dtype/backend remain ordinary L1 config, not L3 graph selection.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, GroupedGemmKernel,
    GroupedGemmKernelConfig, GroupedGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct NativeMoeExpertComputeLocalWorkletConfig {
    pub hidden: Dim,
    pub moe_intermediate: Dim,
    pub num_experts: Dim,
    pub ep_size: u16,
    pub top_k: u32,
    pub dtype: DType,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub act_backends: Vec<&'static str>,
    pub grouped_gemm_backends: Vec<&'static str>,
    pub local_ppm: Vec<u32>,
}

impl NativeMoeExpertComputeLocalWorkletConfig {
    #[must_use]
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
pub struct NativeMoeExpertComputeLocalWorkletResolved {
    pub raw_cfg: NativeMoeExpertComputeLocalWorkletConfig,
    pub gate_up: GroupedGemmKernelConfig,
    pub act: ElementwiseKernelConfig,
    pub down: GroupedGemmKernelConfig,
    pub experts_per_gpu: Dim,
    pub activation_dtype_bytes: u32,
}

#[derive(Clone, Debug, Default)]
pub struct NativeMoeExpertComputeLocalWorkletInput {
    pub global_expert_selections: u32,
}

pub struct NativeMoeExpertComputeLocalWorklet {
    pub name: String,
    pub gate_up: Op<GroupedGemmKernel>,
    pub act: Op<ElementwiseKernel>,
    pub down: Op<GroupedGemmKernel>,
    resolved: NativeMoeExpertComputeLocalWorkletResolved,
}

impl NativeMoeExpertComputeLocalWorklet {
    #[must_use]
    pub fn resolve_config(
        cfg: &NativeMoeExpertComputeLocalWorkletConfig,
    ) -> NativeMoeExpertComputeLocalWorkletResolved {
        let ep_size = u32::from(cfg.ep_size);
        assert!(ep_size > 0, "ep_size must be non-zero");
        assert_eq!(cfg.num_experts.get() % ep_size, 0);
        let experts_per_gpu = cfg.num_experts.clone() / Dim::param("ep", ep_size);
        assert_eq!(cfg.local_ppm.len(), experts_per_gpu.get() as usize);
        let activation_dtype_bytes = cfg.activation_dtype.size_bytes();
        let bytes = Dim::param("activation_bytes", activation_dtype_bytes);
        NativeMoeExpertComputeLocalWorkletResolved {
            gate_up: GroupedGemmKernelConfig {
                backends: cfg.grouped_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: 2 * cfg.moe_intermediate.clone(),
                k: cfg.hidden.clone(),
                dtype: cfg.dtype,
                local_ppm: cfg.local_ppm.clone(),
            },
            act: ElementwiseKernelConfig {
                backends: cfg.act_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: 2 * cfg.moe_intermediate.clone() * bytes.clone(),
                output_bytes_per_token: cfg.moe_intermediate.clone() * bytes,
            },
            down: GroupedGemmKernelConfig {
                backends: cfg.grouped_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden.clone(),
                k: cfg.moe_intermediate.clone(),
                dtype: cfg.dtype,
                local_ppm: cfg.local_ppm.clone(),
            },
            experts_per_gpu,
            activation_dtype_bytes,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: NativeMoeExpertComputeLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let gate_up_name = format!("{name}.gate_up");
        let act_name = format!("{name}.activation");
        let down_name = format!("{name}.down");
        let gate_up = Op::new(
            gate_up_name.clone(),
            Arc::new(GroupedGemmKernel::build(
                gate_up_name,
                resolved.gate_up.clone(),
                bridge,
            )?),
        );
        let act = Op::new(
            act_name.clone(),
            Arc::new(ElementwiseKernel::build(
                act_name,
                resolved.act.clone(),
                bridge,
            )?),
        );
        let down = Op::new(
            down_name.clone(),
            Arc::new(GroupedGemmKernel::build(
                down_name,
                resolved.down.clone(),
                bridge,
            )?),
        );
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
                "{} (NativeMoeExpertComputeLocalWorklet) [local; experts={:?}]",
                self.name, self.resolved.experts_per_gpu
            ),
            child: Box::new(CostNode::Sum(vec![
                self.gate_up.compile(builder),
                self.act.compile(builder),
                self.down.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &NativeMoeExpertComputeLocalWorkletInput, ev: &mut Evaluator) {
        let global_expert_selections = input.global_expert_selections;
        self.gate_up.eval(
            &GroupedGemmKernelInput {
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
            (u64::from(global_expert_selections) * local_ppm_sum).div_ceil(1_000_000) as u32;
        self.act.eval(
            &ElementwiseKernelInput {
                num_tokens: local_routed_tokens,
            },
            ev,
        );
        self.down.eval(
            &GroupedGemmKernelInput {
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
    fn native_fp8_remains_the_same_direct_three_slot_graph() {
        let resolved = NativeMoeExpertComputeLocalWorklet::resolve_config(
            &NativeMoeExpertComputeLocalWorkletConfig {
                hidden: 4096.into(),
                moe_intermediate: 1536.into(),
                num_experts: 8.into(),
                ep_size: 2,
                top_k: 2,
                dtype: DType::Fp8E4m3,
                activation_dtype: DType::Bf16,
                gpu_name: "H100".into(),
                act_backends: vec!["triton"],
                grouped_gemm_backends: vec!["deepgemm"],
                local_ppm: vec![125_000; 4],
            },
        );
        assert_eq!(resolved.gate_up.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.down.dtype, DType::Fp8E4m3);
    }
}
