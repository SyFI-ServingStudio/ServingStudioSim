//! FP8 grouped GEMM together with its required BF16 input quantization.
//!
//! The two launches form one semantic operation: the quantized activation and
//! block scales exist only for the immediately following grouped GEMM. Keeping
//! their row-collapse logic here prevents L3 worklets from seeing sub-kernels.

use std::sync::Arc;

use crate::timing::kernels::{
    Fp8BlockQuantKernel, Fp8BlockQuantKernelConfig, Fp8BlockQuantKernelInput,
    Fp8BlockscaleGroupedGemmKernel, Fp8BlockscaleGroupedGemmKernelConfig,
    Fp8BlockscaleGroupedGemmKernelInput,
};
use crate::timing::routing::RoutingDistribution;
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge, Probe};

#[derive(Clone, Debug)]
pub struct GroupedFp8GemmWithQuantConfig {
    pub quant: Fp8BlockQuantKernelConfig,
    pub gemm: Fp8BlockscaleGroupedGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct GroupedFp8GemmWithQuantInput {
    pub global_expert_selections: u32,
}

pub struct GroupedFp8GemmWithQuantOp {
    pub name: String,
    quant: Arc<Fp8BlockQuantKernel>,
    gemm: Arc<Fp8BlockscaleGroupedGemmKernel>,
    local_ppm: Vec<u32>,
    experts_per_token: u32,
}

impl GroupedFp8GemmWithQuantOp {
    pub fn build(
        name: String,
        config: GroupedFp8GemmWithQuantConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        assert_eq!(
            config.quant.num_problems.get() as usize,
            config.gemm.local_ppm.len(),
            "grouped FP8 quant num_problems must equal local expert count"
        );
        let quant = Arc::new(Fp8BlockQuantKernel::build(
            format!("{name}.input_quant"),
            config.quant,
            bridge,
        )?);
        let local_ppm = config.gemm.local_ppm.clone();
        let experts_per_token = config.gemm.experts_per_token;
        let gemm = Arc::new(Fp8BlockscaleGroupedGemmKernel::build(
            format!("{name}.gemm"),
            config.gemm,
            bridge,
        )?);
        Ok(Self {
            name,
            quant,
            gemm,
            local_ppm,
            experts_per_token,
        })
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

    pub fn eval(&self, input: &GroupedFp8GemmWithQuantInput, ev: &mut Evaluator) {
        let local_rows = local_quant_rows(input.global_expert_selections, &self.local_ppm);
        let quant_input = Fp8BlockQuantKernelInput {
            num_tokens: local_rows,
        };
        ev.push(self.quant.eval(&quant_input), || quant_input.into());

        assert!(
            self.experts_per_token > 0
                && input.global_expert_selections % self.experts_per_token == 0,
            "global expert selections must be divisible by experts_per_token"
        );
        let gemm_input = Fp8BlockscaleGroupedGemmKernelInput {
            num_input_tokens: input.global_expert_selections / self.experts_per_token,
        };
        ev.push(self.gemm.eval(&gemm_input), || gemm_input.into());
    }
}

fn local_quant_rows(global_expert_selections: u32, local_ppm: &[u32]) -> u32 {
    RoutingDistribution::to_per_expert_counts(global_expert_selections, local_ppm)
        .into_iter()
        .sum()
}

#[cfg(test)]
mod tests {
    use super::local_quant_rows;

    #[test]
    fn quant_rows_are_the_grouped_gemm_rank_share() {
        assert_eq!(local_quant_rows(512, &[250_000, 125_000]), 192);
    }
}
