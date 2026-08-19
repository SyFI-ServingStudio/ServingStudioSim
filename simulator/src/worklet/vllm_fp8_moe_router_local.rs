//! vLLM-aligned FP8 router section. The upstream vLLM attention worklet owns
//! the residual/RMSNorm boundary, so this local section is only quant + router.

use crate::op::gemm::{
    SingleFp8GemmWithQuantConfig, SingleFp8GemmWithQuantInput, SingleFp8GemmWithQuantOp,
};
use crate::timing::bridge::DType;
use crate::timing::kernels::{Fp8PerTokenGroupQuantKernelConfig, SingleGemmKernelConfig};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct VllmFp8MoeRouterLocalWorkletConfig {
    pub hidden: Dim,
    pub num_experts: Dim,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub gemm_backends: Vec<&'static str>,
    pub fp8_quant_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct VllmFp8MoeRouterLocalWorkletResolved {
    pub raw_cfg: VllmFp8MoeRouterLocalWorkletConfig,
    pub router: SingleFp8GemmWithQuantConfig,
}

#[derive(Clone, Debug, Default)]
pub struct VllmFp8MoeRouterLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct VllmFp8MoeRouterLocalWorklet {
    pub name: String,
    pub router: SingleFp8GemmWithQuantOp,
    resolved: VllmFp8MoeRouterLocalWorkletResolved,
}

impl VllmFp8MoeRouterLocalWorklet {
    #[must_use]
    pub fn resolve_config(
        cfg: &VllmFp8MoeRouterLocalWorkletConfig,
    ) -> VllmFp8MoeRouterLocalWorkletResolved {
        VllmFp8MoeRouterLocalWorkletResolved {
            router: SingleFp8GemmWithQuantConfig {
                quant: Fp8PerTokenGroupQuantKernelConfig {
                    backends: cfg.fp8_quant_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    hidden_size: cfg.hidden.clone(),
                    group_size: 128,
                    input_dtype: cfg.activation_dtype,
                    scale_format: "ue8m0_column_major".to_string(),
                },
                gemm: SingleGemmKernelConfig {
                    backends: cfg.gemm_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    n: cfg.num_experts.clone(),
                    k: cfg.hidden.clone(),
                    dtype: DType::Fp8E4m3,
                },
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: VllmFp8MoeRouterLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let router = SingleFp8GemmWithQuantOp::build(
            format!("{name}.router_gemm"),
            resolved.router.clone(),
            bridge,
        )?;
        Ok(Self {
            name,
            router,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (VllmFp8MoeRouterLocalWorklet) [local; hidden={:?}]",
                self.name, self.resolved.raw_cfg.hidden
            ),
            child: Box::new(self.router.compile(builder)),
        }
    }

    pub fn eval(&self, input: &VllmFp8MoeRouterLocalWorkletInput, ev: &mut Evaluator) {
        self.router.eval(
            &SingleFp8GemmWithQuantInput {
                num_tokens: input.batch_tokens,
            },
            ev,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_router_without_a_norm_slot() {
        let resolved =
            VllmFp8MoeRouterLocalWorklet::resolve_config(&VllmFp8MoeRouterLocalWorkletConfig {
                hidden: 4096.into(),
                num_experts: 128.into(),
                activation_dtype: DType::Bf16,
                gpu_name: "H100".into(),
                gemm_backends: vec!["deepgemm"],
                fp8_quant_backends: vec!["vllm_cuda"],
            });
        assert_eq!(resolved.router.gemm.dtype, DType::Fp8E4m3);
    }
}
