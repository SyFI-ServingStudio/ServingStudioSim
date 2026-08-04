//! Native BF16 MoE router section: post-attention RMSNorm followed by a direct
//! router GEMM. The norm is unconditionally owned here because native attention
//! ends at a pure TP all-reduce and does not fuse the residual/norm boundary.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct NativeMoeRouterLocalWorkletConfig {
    pub hidden: Dim,
    pub num_experts: Dim,
    pub dtype: DType,
    pub gpu_name: String,
    pub norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct NativeMoeRouterLocalWorkletResolved {
    pub raw_cfg: NativeMoeRouterLocalWorkletConfig,
    pub post_norm: RmsNormKernelConfig,
    pub router: SingleGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct NativeMoeRouterLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct NativeMoeRouterLocalWorklet {
    pub name: String,
    pub post_norm: Op<RmsNormKernel>,
    pub router: Op<SingleGemmKernel>,
    resolved: NativeMoeRouterLocalWorkletResolved,
}

impl NativeMoeRouterLocalWorklet {
    pub fn resolve_config(
        cfg: &NativeMoeRouterLocalWorkletConfig,
    ) -> NativeMoeRouterLocalWorkletResolved {
        assert_ne!(
            cfg.dtype,
            DType::Fp8E4m3,
            "native BF16 router requires a non-FP8 dtype"
        );
        NativeMoeRouterLocalWorkletResolved {
            post_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            router: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.num_experts.clone(),
                k: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: NativeMoeRouterLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let norm_name = format!("{name}.post_attn_norm");
        let router_name = format!("{name}.router_gemm");
        let post_norm = Op::new(
            norm_name.clone(),
            Arc::new(RmsNormKernel::build(
                norm_name,
                resolved.post_norm.clone(),
                bridge,
            )?),
        );
        let router = Op::new(
            router_name.clone(),
            Arc::new(SingleGemmKernel::build(
                router_name,
                resolved.router.clone(),
                bridge,
            )?),
        );
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
                "{} (NativeMoeRouterLocalWorklet) [local; hidden={:?}]",
                self.name, self.resolved.raw_cfg.hidden
            ),
            child: Box::new(CostNode::Sum(vec![
                self.post_norm.compile(builder),
                self.router.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &NativeMoeRouterLocalWorkletInput, ev: &mut Evaluator) {
        let batch_tokens = input.batch_tokens;
        self.post_norm
            .eval(&RmsNormKernelInput { m: batch_tokens }, ev);
        self.router
            .eval(&SingleGemmKernelInput { m: batch_tokens }, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_concrete_norm_then_bf16_router() {
        let resolved =
            NativeMoeRouterLocalWorklet::resolve_config(&NativeMoeRouterLocalWorkletConfig {
                hidden: 4096.into(),
                num_experts: 128.into(),
                dtype: DType::Bf16,
                gpu_name: "H100".into(),
                norm_backends: vec!["flashinfer"],
                gemm_backends: vec!["torch"],
            });
        assert_eq!(resolved.post_norm.dtype, DType::Bf16);
        assert_eq!(resolved.router.dtype, DType::Bf16);
    }
}
