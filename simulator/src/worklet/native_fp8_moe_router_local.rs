//! Native FP8 MoE router section: post-attention RMSNorm followed by explicit
//! per-token-group quantization and the FP8 router GEMM.

use std::sync::Arc;

use crate::op::gemm::{
    SingleFp8GemmWithQuantConfig, SingleFp8GemmWithQuantInput, SingleFp8GemmWithQuantOp,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    Fp8PerTokenGroupQuantKernelConfig, RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput,
    SingleGemmKernelConfig,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct NativeFp8MoeRouterLocalWorkletConfig {
    pub hidden: Dim,
    pub num_experts: Dim,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub fp8_quant_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct NativeFp8MoeRouterLocalWorkletResolved {
    pub raw_cfg: NativeFp8MoeRouterLocalWorkletConfig,
    pub post_norm: RmsNormKernelConfig,
    pub router: SingleFp8GemmWithQuantConfig,
}

#[derive(Clone, Debug, Default)]
pub struct NativeFp8MoeRouterLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct NativeFp8MoeRouterLocalWorklet {
    pub name: String,
    pub post_norm: Op<RmsNormKernel>,
    pub router: SingleFp8GemmWithQuantOp,
    resolved: NativeFp8MoeRouterLocalWorkletResolved,
}

impl NativeFp8MoeRouterLocalWorklet {
    pub fn resolve_config(
        cfg: &NativeFp8MoeRouterLocalWorkletConfig,
    ) -> NativeFp8MoeRouterLocalWorkletResolved {
        let quant = Fp8PerTokenGroupQuantKernelConfig {
            backends: cfg.fp8_quant_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden_size: cfg.hidden.clone(),
            group_size: 128,
            input_dtype: cfg.activation_dtype,
            scale_format: "ue8m0_column_major".to_string(),
        };
        NativeFp8MoeRouterLocalWorkletResolved {
            post_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.clone(),
                dtype: cfg.activation_dtype,
            },
            router: SingleFp8GemmWithQuantConfig {
                quant,
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
        resolved: NativeFp8MoeRouterLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let norm_name = format!("{name}.post_attn_norm");
        let post_norm = Op::new(
            norm_name.clone(),
            Arc::new(RmsNormKernel::build(
                norm_name,
                resolved.post_norm.clone(),
                bridge,
            )?),
        );
        let router = SingleFp8GemmWithQuantOp::build(
            format!("{name}.router_gemm"),
            resolved.router.clone(),
            bridge,
        )?;
        Ok(Self {
            name,
            post_norm,
            router,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (NativeFp8MoeRouterLocalWorklet) [local; hidden={:?}]",
                self.name, self.resolved.raw_cfg.hidden
            ),
            child: Box::new(CostNode::Sum(vec![
                self.post_norm.compile(builder),
                self.router.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &NativeFp8MoeRouterLocalWorkletInput, ev: &mut Evaluator) {
        let batch_tokens = input.batch_tokens;
        self.post_norm
            .eval(&RmsNormKernelInput { m: batch_tokens }, ev);
        self.router.eval(
            &SingleFp8GemmWithQuantInput {
                num_tokens: batch_tokens,
            },
            ev,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_concrete_norm_then_quantized_router() {
        let resolved =
            NativeFp8MoeRouterLocalWorklet::resolve_config(&NativeFp8MoeRouterLocalWorkletConfig {
                hidden: 4096.into(),
                num_experts: 128.into(),
                activation_dtype: DType::Bf16,
                gpu_name: "H100".into(),
                norm_backends: vec!["flashinfer"],
                gemm_backends: vec!["deepgemm"],
                fp8_quant_backends: vec!["vllm_cuda"],
            });
        assert_eq!(resolved.post_norm.dtype, DType::Bf16);
        assert_eq!(resolved.router.gemm.dtype, DType::Fp8E4m3);
    }
}
