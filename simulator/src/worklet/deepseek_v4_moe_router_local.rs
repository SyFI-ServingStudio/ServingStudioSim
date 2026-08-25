//! DeepSeek V4's post-attention normalization and local MoE gate projection.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    GemmFp32OutputKernel, GemmFp32OutputKernelConfig, GemmFp32OutputKernelInput,
    MhcFusedPostPreRmsNormKernel, MhcRmsNormKernelConfig, MhcRmsNormKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

#[derive(Clone, Debug)]
pub struct DeepseekV4MoeRouterLocalWorkletConfig {
    pub hidden_size: Dim,
    pub num_experts: Dim,
    pub hidden_dtype: DType,
    pub gpu_name: String,
    pub hc_mult: u32,
    pub mhc_backends: Vec<&'static str>,
    pub gate_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct DeepseekV4MoeRouterLocalWorkletResolved {
    pub ffn_pre: MhcRmsNormKernelConfig,
    pub gate: GemmFp32OutputKernelConfig,
}

#[derive(Clone, Debug)]
pub struct DeepseekV4MoeRouterLocalWorkletInput {
    pub num_local_tokens: u32,
}

pub struct DeepseekV4MoeRouterLocalWorklet {
    pub name: String,
    pub ffn_pre: Op<MhcFusedPostPreRmsNormKernel>,
    pub gate: Op<GemmFp32OutputKernel>,
}

impl DeepseekV4MoeRouterLocalWorklet {
    pub fn resolve_config(
        config: &DeepseekV4MoeRouterLocalWorkletConfig,
    ) -> DeepseekV4MoeRouterLocalWorkletResolved {
        assert_eq!(config.hidden_size.get(), 4096);
        assert_eq!(config.num_experts.get(), 256);
        assert_eq!(config.hidden_dtype, DType::Bf16);
        assert_eq!(config.hc_mult, 4);
        DeepseekV4MoeRouterLocalWorkletResolved {
            ffn_pre: MhcRmsNormKernelConfig {
                backends: config.mhc_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                hidden_size: config.hidden_size.clone(),
                hc_mult: config.hc_mult,
                hidden_dtype: config.hidden_dtype,
            },
            gate: GemmFp32OutputKernelConfig {
                backends: config.gate_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                n: config.num_experts.clone(),
                k: config.hidden_size.clone(),
                input_dtype: config.hidden_dtype,
            },
        }
    }

    pub fn build(
        name: String,
        resolved: DeepseekV4MoeRouterLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let ffn_pre_name = format!("{name}.mhc_fused_post_pre");
        let ffn_pre = Op::new(
            ffn_pre_name.clone(),
            Arc::new(MhcFusedPostPreRmsNormKernel::build(
                ffn_pre_name,
                resolved.ffn_pre,
                bridge,
            )?),
        );
        let op_name = format!("{name}.gate_fp32_output");
        let gate = Op::new(
            op_name.clone(),
            Arc::new(GemmFp32OutputKernel::build(op_name, resolved.gate, bridge)?),
        );
        Ok(Self {
            name,
            ffn_pre,
            gate,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!("{} (DeepseekV4MoeRouterLocalWorklet)", self.name),
            child: Box::new(CostNode::Sum(vec![
                self.ffn_pre.compile(builder),
                self.gate.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &DeepseekV4MoeRouterLocalWorkletInput, evaluator: &mut Evaluator) {
        let zero = input.num_local_tokens == 0;
        eval_or_zero(
            &self.ffn_pre,
            MhcRmsNormKernelInput {
                num_tokens: input.num_local_tokens,
            },
            zero,
            evaluator,
        );
        eval_or_zero(
            &self.gate,
            GemmFp32OutputKernelInput {
                m: input.num_local_tokens,
            },
            zero,
            evaluator,
        );
    }
}

fn eval_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, evaluator: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    if zero {
        evaluator.push(LeafMetrics::ZERO, || input.into());
    } else {
        op.eval(&input, evaluator);
    }
}
