//! The Inkling MoE router on one rank, ahead of the routed and sink experts
//! (`ink/moe.py`): the replicated BF16 router GEMM with fp32 logits over the
//! routed and sink experts (`_linear_with_fp32_out`, padded to 264 columns),
//! the gate select (sigmoid + bias top-k over the routed experts, then the
//! log-sigmoid renorm with the sink logits, `_inkling_gate_select_kernel`),
//! and the two `contiguous()` copies of the routed weight and id columns.
//! Every rank routes every token, so nothing here is sharded.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, GemmFp32OutputKernel,
    GemmFp32OutputKernelConfig, GemmFp32OutputKernelInput, MoeSinkGateSelectKernel,
    MoeSinkGateSelectKernelConfig, MoeSinkGateSelectKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, elementwise, push_or_zero, repeated};

/// `weights[:, :top_k]` and `ids[:, :top_k]`: one launch each.
const TOPK_COPIES: u32 = 2;

/// Whether the top-k `contiguous()` copies launch. A one-row slice is already
/// contiguous (PyTorch ignores the stride of a size-1 dim), so at one token
/// both calls return the view and launch nothing.
pub fn topk_copies_launch(num_tokens: u32) -> bool {
    num_tokens > 1
}

#[derive(Clone, Debug)]
pub struct InklingRouterLocalWorkletConfig {
    pub hidden: Dim,
    pub n_routed_experts: u32,
    pub n_shared_experts: u32,
    pub top_k: u32,
    /// The router GEMM's padded output width, the logits row stride.
    pub logits_width: u32,
    pub has_bias: bool,
    pub has_global_scale: bool,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub router_backends: Vec<&'static str>,
    pub gate_select_backends: Vec<&'static str>,
    pub copy_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct InklingRouterLocalWorkletResolved {
    pub raw_cfg: InklingRouterLocalWorkletConfig,
    pub router: GemmFp32OutputKernelConfig,
    pub gate_select: MoeSinkGateSelectKernelConfig,
    pub topk_copy: ElementwiseKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct InklingRouterLocalWorkletInput {
    pub num_tokens: u32,
}

pub struct InklingRouterLocalWorklet {
    pub name: String,
    pub router: Op<GemmFp32OutputKernel>,
    pub gate_select: Op<MoeSinkGateSelectKernel>,
    pub topk_copy: Op<ElementwiseKernel>,
    resolved: InklingRouterLocalWorkletResolved,
}

impl InklingRouterLocalWorklet {
    pub fn resolve_config(
        cfg: &InklingRouterLocalWorkletConfig,
    ) -> InklingRouterLocalWorkletResolved {
        assert!(
            cfg.logits_width >= cfg.n_routed_experts + cfg.n_shared_experts,
            "the router logits must hold every routed and sink expert"
        );
        // One fp32 weight or int32 id per selected routed expert.
        let copy_bytes = cfg.top_k * 4;
        InklingRouterLocalWorkletResolved {
            router: GemmFp32OutputKernelConfig {
                backends: cfg.router_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.logits_width.into(),
                k: cfg.hidden.clone(),
                input_dtype: cfg.activation_dtype,
            },
            gate_select: MoeSinkGateSelectKernelConfig {
                backends: cfg.gate_select_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_routed_experts: cfg.n_routed_experts,
                num_shared_experts: cfg.n_shared_experts,
                top_k: cfg.top_k,
                logits_row_stride: cfg.logits_width,
                has_bias: cfg.has_bias,
                has_global_scale: cfg.has_global_scale,
                logits_dtype: DType::Fp32,
            },
            topk_copy: elementwise(&cfg.copy_backends, &cfg.gpu_name, copy_bytes, copy_bytes),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: InklingRouterLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            router: atomic(n, "router", r.router, GemmFp32OutputKernel::build, bridge)?,
            gate_select: atomic(
                n,
                "gate_select",
                r.gate_select,
                MoeSinkGateSelectKernel::build,
                bridge,
            )?,
            topk_copy: atomic(n, "topk_copy", r.topk_copy, ElementwiseKernel::build, bridge)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (InklingRouterLocalWorklet) [replicated; top-{} of {} + {} sinks]",
                self.name, cfg.top_k, cfg.n_routed_experts, cfg.n_shared_experts
            ),
            child: Box::new(CostNode::Sum(vec![
                self.router.compile(builder),
                self.gate_select.compile(builder),
                repeated(&self.topk_copy, TOPK_COPIES, builder),
            ])),
        }
    }

    pub fn eval(&self, input: &InklingRouterLocalWorkletInput, ev: &mut Evaluator) {
        let tokens = input.num_tokens;
        push_or_zero(&self.router, GemmFp32OutputKernelInput { m: tokens }, false, ev);
        push_or_zero(
            &self.gate_select,
            MoeSinkGateSelectKernelInput { num_tokens: tokens },
            false,
            ev,
        );
        push_or_zero(
            &self.topk_copy,
            ElementwiseKernelInput { num_tokens: tokens },
            !topk_copies_launch(tokens),
            ev,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_logits_cover_routed_and_sink_experts() {
        let resolved = InklingRouterLocalWorklet::resolve_config(&InklingRouterLocalWorkletConfig {
            hidden: 4096.into(),
            n_routed_experts: 256,
            n_shared_experts: 2,
            top_k: 6,
            logits_width: 264,
            has_bias: true,
            has_global_scale: true,
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".into(),
            router_backends: vec!["vllm_ll_bf16_auto"],
            gate_select_backends: vec!["vllm_triton"],
            copy_backends: vec!["torch"],
        });
        assert_eq!((resolved.router.n.get(), resolved.router.k.get()), (264, 4096));
        assert_eq!(resolved.gate_select.logits_row_stride, 264);
        assert_eq!(resolved.topk_copy.input_bytes_per_token.get(), 24);
    }

    #[test]
    fn a_single_token_launches_no_topk_copy() {
        assert!(!topk_copies_launch(1));
        assert!(topk_copies_launch(2));
        assert!(topk_copies_launch(32));
    }
}
