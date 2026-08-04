//! Dense FP8 GEMM together with vLLM's required per-token-group quantization.

use std::sync::Arc;

use crate::timing::kernels::{
    Fp8PerTokenGroupQuantKernel, Fp8PerTokenGroupQuantKernelConfig,
    Fp8PerTokenGroupQuantKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge, Probe};

#[derive(Clone, Debug)]
pub struct SingleFp8GemmWithQuantConfig {
    pub quant: Fp8PerTokenGroupQuantKernelConfig,
    pub gemm: SingleGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct SingleFp8GemmWithQuantInput {
    pub num_tokens: u32,
}

pub struct SingleFp8GemmWithQuantOp {
    pub name: String,
    quant: Arc<Fp8PerTokenGroupQuantKernel>,
    gemm: Arc<SingleGemmKernel>,
}

impl SingleFp8GemmWithQuantOp {
    pub fn build(
        name: String,
        config: SingleFp8GemmWithQuantConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let quant = Arc::new(Fp8PerTokenGroupQuantKernel::build(
            format!("{name}.input_quant"),
            config.quant,
            bridge,
        )?);
        let gemm = Arc::new(SingleGemmKernel::build(
            format!("{name}.gemm"),
            config.gemm,
            bridge,
        )?);
        Ok(Self { name, quant, gemm })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            builder.leaf(
                format!("{}.input_quant", self.name),
                self.quant.kind(),
                self.quant.describe_config(),
            ),
            builder.leaf(
                format!("{}.gemm", self.name),
                self.gemm.kind(),
                self.gemm.describe_config(),
            ),
        ])
    }

    pub fn eval(&self, input: &SingleFp8GemmWithQuantInput, ev: &mut Evaluator) {
        let quant_input = Fp8PerTokenGroupQuantKernelInput {
            num_tokens: input.num_tokens,
        };
        ev.push(self.quant.eval(&quant_input), || quant_input.into());

        let gemm_input = SingleGemmKernelInput {
            m: input.num_tokens,
        };
        ev.push(self.gemm.eval(&gemm_input), || gemm_input.into());
    }
}
