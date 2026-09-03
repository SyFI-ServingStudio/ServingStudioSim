//! SGLang's explicit finalize after a deferred-finalize fused MoE callable.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    MoeFinalizeFuseSharedKernel, MoeFinalizeFuseSharedKernelConfig,
    MoeFinalizeFuseSharedKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge,
};

#[derive(Clone, Debug)]
pub struct SglangMoeFinalizeLocalWorkletConfig {
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub top_k: u32,
    pub hidden_dim: Dim,
    pub dtype: DType,
    pub fuse_shared_output: bool,
}

#[derive(Clone, Debug)]
pub struct SglangMoeFinalizeLocalWorkletResolved {
    pub finalize: MoeFinalizeFuseSharedKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct SglangMoeFinalizeLocalWorkletInput {
    pub num_tokens: u32,
}

pub struct SglangMoeFinalizeLocalWorklet {
    pub name: String,
    pub finalize: Op<MoeFinalizeFuseSharedKernel>,
}

impl SglangMoeFinalizeLocalWorklet {
    pub fn resolve_config(
        cfg: &SglangMoeFinalizeLocalWorkletConfig,
    ) -> SglangMoeFinalizeLocalWorkletResolved {
        assert_eq!(cfg.top_k, 8, "GLM-5.2 router_top_k must be 8");
        assert_eq!(cfg.hidden_dim.get(), 6_144, "GLM-5.2 hidden must be 6144");
        assert_eq!(cfg.dtype, DType::Bf16, "GLM-5.2 output must be BF16");
        assert!(
            cfg.fuse_shared_output,
            "SGLang GLM-5.2 finalizes with the shared output"
        );
        SglangMoeFinalizeLocalWorkletResolved {
            finalize: MoeFinalizeFuseSharedKernelConfig {
                backends: cfg.backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                top_k: cfg.top_k,
                hidden_dim: cfg.hidden_dim.clone(),
                dtype: cfg.dtype,
                fuse_shared_output: cfg.fuse_shared_output,
            },
        }
    }

    pub fn build(
        name: String,
        resolved: SglangMoeFinalizeLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let slot = format!("{name}.finalize");
        Ok(Self {
            name,
            finalize: Op::new(
                slot.clone(),
                Arc::new(MoeFinalizeFuseSharedKernel::build(
                    slot,
                    resolved.finalize,
                    bridge,
                )?),
            ),
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!("{} (SglangMoeFinalizeLocalWorklet)", self.name),
            child: Box::new(self.finalize.compile(builder)),
        }
    }

    pub fn eval(&self, input: &SglangMoeFinalizeLocalWorkletInput, ev: &mut Evaluator) {
        let shape = MoeFinalizeFuseSharedKernelInput {
            num_tokens: input.num_tokens,
        };
        let metrics = if input.num_tokens == 0 {
            LeafMetrics::ZERO
        } else {
            self.finalize.kernel.eval(&shape)
        };
        ev.push(metrics, || shape.clone().into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_keeps_finalize_outside_the_neutral_moe_callable() {
        let r =
            SglangMoeFinalizeLocalWorklet::resolve_config(&SglangMoeFinalizeLocalWorkletConfig {
                backends: vec!["sglang_cuda"],
                gpu_name: "NVIDIA B200".to_string(),
                top_k: 8,
                hidden_dim: 6144.into(),
                dtype: DType::Bf16,
                fuse_shared_output: true,
            });
        assert_eq!(r.finalize.backends, vec!["sglang_cuda"]);
        assert!(r.finalize.fuse_shared_output);
    }
}
