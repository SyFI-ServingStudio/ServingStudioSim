//! DeepSeek V4's DP-local shared-expert MLP at vLLM kernel granularity.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ClampedSwigluKernel, ClampedSwigluKernelConfig, ClampedSwigluKernelInput,
    Fp8PerTokenGroupQuantKernel, Fp8PerTokenGroupQuantKernelConfig,
    Fp8PerTokenGroupQuantKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

#[derive(Clone, Debug)]
pub struct DeepseekV4SharedExpertLocalWorkletConfig {
    pub hidden_size: Dim,
    pub intermediate_size: Dim,
    pub num_shared_experts: u32,
    pub hidden_dtype: DType,
    pub gpu_name: String,
    pub quant_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub activation_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct DeepseekV4SharedExpertLocalWorkletResolved {
    pub gate_up_quant: Fp8PerTokenGroupQuantKernelConfig,
    pub gate_up: SingleGemmKernelConfig,
    pub activation: ClampedSwigluKernelConfig,
    pub down_quant: Fp8PerTokenGroupQuantKernelConfig,
    pub down: SingleGemmKernelConfig,
}

#[derive(Clone, Debug)]
pub struct DeepseekV4SharedExpertLocalWorkletInput {
    pub num_local_tokens: u32,
}

pub struct DeepseekV4SharedExpertLocalWorklet {
    pub name: String,
    pub gate_up_quant: Op<Fp8PerTokenGroupQuantKernel>,
    pub gate_up: Op<SingleGemmKernel>,
    pub activation: Op<ClampedSwigluKernel>,
    pub down_quant: Op<Fp8PerTokenGroupQuantKernel>,
    pub down: Op<SingleGemmKernel>,
}

impl DeepseekV4SharedExpertLocalWorklet {
    pub fn resolve_config(
        config: &DeepseekV4SharedExpertLocalWorkletConfig,
    ) -> DeepseekV4SharedExpertLocalWorkletResolved {
        assert_eq!(config.hidden_size.get(), 4096);
        assert_eq!(config.intermediate_size.get(), 2048);
        assert_eq!(config.num_shared_experts, 1);
        assert_eq!(config.hidden_dtype, DType::Bf16);
        let shared_width = config.intermediate_size.get() * config.num_shared_experts;
        let quant = |hidden_size: u32| Fp8PerTokenGroupQuantKernelConfig {
            backends: config.quant_backends.clone(),
            gpu_name: config.gpu_name.clone(),
            hidden_size: hidden_size.into(),
            group_size: 128,
            input_dtype: config.hidden_dtype,
            scale_format: "ue8m0_column_major".to_string(),
        };
        DeepseekV4SharedExpertLocalWorkletResolved {
            gate_up_quant: quant(config.hidden_size.get()),
            gate_up: SingleGemmKernelConfig {
                backends: config.gemm_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                n: (2 * shared_width).into(),
                k: config.hidden_size.clone(),
                dtype: DType::Fp8E4m3,
            },
            activation: ClampedSwigluKernelConfig {
                backends: config.activation_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                hidden_dim: shared_width.into(),
                dtype: config.hidden_dtype,
            },
            down_quant: quant(shared_width),
            down: SingleGemmKernelConfig {
                backends: config.gemm_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                n: config.hidden_size.clone(),
                k: shared_width.into(),
                dtype: DType::Fp8E4m3,
            },
        }
    }

    pub fn build(
        name: String,
        resolved: DeepseekV4SharedExpertLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        macro_rules! build_op {
            ($field:ident, $kernel:ty, $suffix:literal) => {{
                let op_name = format!("{name}.{}", $suffix);
                Op::new(
                    op_name.clone(),
                    Arc::new(<$kernel>::build(op_name, resolved.$field, bridge)?),
                )
            }};
        }
        Ok(Self {
            name: name.clone(),
            gate_up_quant: build_op!(
                gate_up_quant,
                Fp8PerTokenGroupQuantKernel,
                "gate_up_input_quant"
            ),
            gate_up: build_op!(gate_up, SingleGemmKernel, "gate_up_projection"),
            activation: build_op!(activation, ClampedSwigluKernel, "clamped_swiglu"),
            down_quant: build_op!(down_quant, Fp8PerTokenGroupQuantKernel, "down_input_quant"),
            down: build_op!(down, SingleGemmKernel, "down_projection"),
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!("{} (DeepseekV4SharedExpertLocalWorklet)", self.name),
            child: Box::new(CostNode::Sum(vec![
                self.gate_up_quant.compile(builder),
                self.gate_up.compile(builder),
                self.activation.compile(builder),
                self.down_quant.compile(builder),
                self.down.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &DeepseekV4SharedExpertLocalWorkletInput, evaluator: &mut Evaluator) {
        let tokens = input.num_local_tokens;
        let zero = tokens == 0;
        eval_or_zero(
            &self.gate_up_quant,
            Fp8PerTokenGroupQuantKernelInput { num_tokens: tokens },
            zero,
            evaluator,
        );
        eval_or_zero(
            &self.gate_up,
            SingleGemmKernelInput { m: tokens },
            zero,
            evaluator,
        );
        eval_or_zero(
            &self.activation,
            ClampedSwigluKernelInput { num_rows: tokens },
            zero,
            evaluator,
        );
        eval_or_zero(
            &self.down_quant,
            Fp8PerTokenGroupQuantKernelInput { num_tokens: tokens },
            zero,
            evaluator,
        );
        eval_or_zero(
            &self.down,
            SingleGemmKernelInput { m: tokens },
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
