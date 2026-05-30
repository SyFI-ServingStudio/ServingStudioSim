//! `MoeRouterLocalWorklet` — the post-attention norm + MoE router GEMM that
//! every EP rank runs on its OWN home tokens *before* the MoE dispatch all-to-
//! all. `Local` group suffix (L3 §1.5): one sync section, single GPU, no
//! collective inside.
//!
//! Both leaves are per-token full-hidden ops:
//!   - `post_attn_norm` reads the [tokens × hidden] residual stream and emits
//!     [tokens × hidden] normalized values;
//!   - `router_gemm` is a tiny dense `[hidden → num_experts]` projection that
//!     produces the per-token routing scores (the L2 input to the
//!     `MoeDispatchOp` that follows in the L4 cost tree).
//!
//! `hidden` is NOT sharded here — every EP rank holds the full hidden vector
//! for its home tokens after the upstream attention `tp_allreduce`. The worklet
//! is the L4 cost-tree boundary between the attention block (which is
//! Max-fanned over `num_dp_groups`) and the MoE dispatch all-to-all (which is
//! the next sync section).

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

/// Raw config. The two leaf shapes are derivable directly (no partition).
#[derive(Clone, Debug)]
pub struct MoeRouterLocalWorkletConfig {
    pub hidden: u32,
    pub num_experts: u32,
    pub dtype: DType,
    pub gpu_name: String,
    pub norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct MoeRouterLocalWorkletResolved {
    pub raw_cfg: MoeRouterLocalWorkletConfig,
    pub post_norm: RmsNormKernelConfig,
    pub router: SingleGemmKernelConfig,
}

/// Per-call shape: this rank's home-token count drives both leaves.
#[derive(Clone, Debug, Default)]
pub struct MoeRouterLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct MoeRouterLocalWorklet {
    pub name: String,
    pub post_norm: Op<RmsNormKernel>,
    pub router: Op<SingleGemmKernel>,
    resolved: MoeRouterLocalWorkletResolved,
}

impl MoeRouterLocalWorklet {
    pub fn resolve_config(cfg: &MoeRouterLocalWorkletConfig) -> MoeRouterLocalWorkletResolved {
        MoeRouterLocalWorkletResolved {
            post_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden,
                dtype: cfg.dtype,
            },
            router: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.num_experts,
                k: cfg.hidden,
                dtype: cfg.dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: MoeRouterLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let pn_name = format!("{name}.post_attn_norm");
        let r_name = format!("{name}.router_gemm");
        let post_norm = Op::new(
            pn_name.clone(),
            Arc::new(RmsNormKernel::build(pn_name, resolved.post_norm.clone(), bridge)?),
        );
        let router = Op::new(
            r_name.clone(),
            Arc::new(SingleGemmKernel::build(r_name, resolved.router.clone(), bridge)?),
        );
        Ok(Self {
            name,
            post_norm,
            router,
            resolved,
        })
    }

    /// CostTree compile: `Sum(post_norm, router_gemm)` under a labeled header.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let r = &self.resolved;
        let label = format!(
            "{} (MoeRouterLocalWorklet) [hidden={}, num_experts={}]",
            self.name, r.raw_cfg.hidden, r.raw_cfg.num_experts,
        );
        CostNode::Labeled {
            label,
            child: Box::new(CostNode::Sum(vec![
                self.post_norm.compile(builder),
                self.router.compile(builder),
            ])),
        }
    }

    /// CostTree eval: fill slots in `compile` child order — norm, router.
    pub fn eval(&self, input: &MoeRouterLocalWorkletInput, ev: &mut Evaluator) {
        let m = input.batch_tokens;
        self.post_norm.eval(&RmsNormKernelInput { m }, ev);
        self.router.eval(&SingleGemmKernelInput { m }, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MoeRouterLocalWorkletConfig {
        MoeRouterLocalWorkletConfig {
            hidden: 4096,
            num_experts: 128,
            dtype: DType::Bf16,
            gpu_name: "H100".to_string(),
            norm_backends: vec!["flashinfer"],
            gemm_backends: vec!["torch"],
        }
    }

    #[test]
    fn resolve_threads_hidden_into_norm_and_gemm_shapes() {
        let r = MoeRouterLocalWorklet::resolve_config(&cfg());
        assert_eq!(r.post_norm.hidden, 4096);
        assert_eq!(r.router.n, 128); // n = num_experts
        assert_eq!(r.router.k, 4096); // k = hidden
        assert_eq!(r.router.dtype, DType::Bf16);
    }
}
