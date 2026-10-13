//! FP8-block sibling of [`super::vllm_nvfp4_moe_shared_stream_local`]: the same
//! router, aux-stream shared expert and add, around a TRT-LLM FP8 block-scale
//! routed MoE (a checkpoint whose experts are W8A8 FP8 with 128-wide blocks,
//! e.g. an MTP layer next to NVFP4 target layers).
//!
//! The routed callable is [`Glm53RoutedMoeLocalWorklet`] as is: a row-major
//! UE8M0 `per_token_group_quant_8bit` (the capture's template is
//! `<bf16, e4m3, IS_COLUMN_MAJOR=false, SCALE_UE8M0=true>`, which is that L1
//! `scale_format`) and `trtllm_fp8_block_scale_moe`. Before the fused call
//! the DeepSeek-FP8 path transposes the FP32 scales,
//! `a1q_scale.t().contiguous()` (fork
//! `fused_moe/experts/trtllm_fp8_moe.py:469`), which is one strided copy of
//! `hidden / group` FP32 values per row. That copy is not in the reused
//! callable, so it is this worklet's `scale_transpose` leaf, at the end of the
//! routed branch. It launches between the quant and the fused call; the
//! branch is a serial `Sum`, so the manifest position does not change the
//! cost.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::bf16_gated_shared_expert_local::{
    Bf16GatedSharedExpertLocalWorklet, Bf16GatedSharedExpertLocalWorkletConfig,
    Bf16GatedSharedExpertLocalWorkletResolved,
};
use super::deepseek_v41_common::SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD;
use super::glm53_common::{
    atomic, elementwise, push_or_zero, Glm53ActivationQuantConfig, Glm53WeightPrecision,
};
use super::glm53_moe_local::{
    Glm53RoutedMoeLocalWorklet, Glm53RoutedMoeLocalWorkletConfig, Glm53RoutedMoeLocalWorkletInput,
    Glm53RoutedMoeLocalWorkletResolved,
};
use super::vllm_nvfp4_moe_shared_stream_local::{
    add_config, build_shared_pair, compose_block, eval_shared_and_add, resolve_shared, router_gemm,
};

/// Bytes of one FP32 activation scale.
const SCALE_BYTES: u32 = 4;

#[derive(Clone, Debug)]
pub struct VllmFp8MoeSharedStreamLocalWorkletConfig {
    pub hidden: Dim,
    pub num_experts: Dim,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub router_backends: Vec<&'static str>,
    /// The FP32 scale transpose ahead of the fused call.
    pub scale_transpose_backends: Vec<&'static str>,
    pub add_backends: Vec<&'static str>,
    pub shared: Bf16GatedSharedExpertLocalWorkletConfig,
    /// FP8 block routed experts; `expert_demand` as in the NVFP4 block.
    pub routed: Glm53RoutedMoeLocalWorkletConfig,
}

#[derive(Clone, Debug)]
pub struct VllmFp8MoeSharedStreamLocalWorkletResolved {
    pub raw_cfg: VllmFp8MoeSharedStreamLocalWorkletConfig,
    pub router: SingleGemmKernelConfig,
    pub routed: Glm53RoutedMoeLocalWorkletResolved,
    pub scale_transpose: ElementwiseKernelConfig,
    pub shared: Bf16GatedSharedExpertLocalWorkletResolved,
    pub add: ElementwiseKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct VllmFp8MoeSharedStreamLocalWorkletInput {
    /// Rows the block runs at: the padded size under a CUDA graph.
    pub num_tokens: u32,
}

pub struct VllmFp8MoeSharedStreamLocalWorklet {
    pub name: String,
    pub router: Op<SingleGemmKernel>,
    pub routed: Glm53RoutedMoeLocalWorklet,
    pub scale_transpose: Op<ElementwiseKernel>,
    pub shared: Bf16GatedSharedExpertLocalWorklet,
    pub shared_serial: Bf16GatedSharedExpertLocalWorklet,
    pub add: Op<ElementwiseKernel>,
    resolved: VllmFp8MoeSharedStreamLocalWorkletResolved,
}

impl VllmFp8MoeSharedStreamLocalWorklet {
    pub fn resolve_config(
        cfg: &VllmFp8MoeSharedStreamLocalWorkletConfig,
    ) -> VllmFp8MoeSharedStreamLocalWorkletResolved {
        assert_eq!(
            cfg.routed.precision,
            Glm53WeightPrecision::Fp8Block,
            "the FP8 block sibling takes FP8 block routed experts"
        );
        assert_eq!(
            cfg.routed.hidden.get(),
            cfg.hidden.get(),
            "routed hidden must match the block"
        );
        assert_eq!(
            cfg.routed.num_experts.get(),
            cfg.num_experts.get(),
            "routed num_experts must match the router"
        );
        assert_eq!(cfg.routed.ep_size, 1, "a local block holds every expert");
        let routed = Glm53RoutedMoeLocalWorklet::resolve_config(&cfg.routed);
        let Glm53ActivationQuantConfig::Fp8(quant) = &routed.input_quant else {
            panic!("an FP8 block routed callable quantizes its input to FP8");
        };
        let groups = cfg.hidden.get() / quant.group_size;
        let scale_row = groups * SCALE_BYTES;
        VllmFp8MoeSharedStreamLocalWorkletResolved {
            router: router_gemm(
                &cfg.router_backends,
                &cfg.gpu_name,
                cfg.hidden.clone(),
                cfg.num_experts.clone(),
                cfg.activation_dtype,
            ),
            scale_transpose: elementwise(
                &cfg.scale_transpose_backends,
                &cfg.gpu_name,
                scale_row,
                scale_row,
            ),
            routed,
            shared: resolve_shared(&cfg.shared, &cfg.hidden),
            add: add_config(
                &cfg.add_backends,
                &cfg.gpu_name,
                &cfg.hidden,
                cfg.activation_dtype,
            ),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: VllmFp8MoeSharedStreamLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        let (shared, shared_serial) = build_shared_pair(n, &r.shared, bridge)?;
        Ok(Self {
            router: atomic(n, "router", r.router, SingleGemmKernel::build, bridge)?,
            routed: Glm53RoutedMoeLocalWorklet::build(format!("{n}.routed"), r.routed, bridge)?,
            scale_transpose: atomic(
                n,
                "routed_scale_transpose",
                r.scale_transpose,
                ElementwiseKernel::build,
                bridge,
            )?,
            shared,
            shared_serial,
            add: atomic(
                n,
                "shared_routed_add",
                r.add,
                ElementwiseKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let routed_branch = vec![
            self.router.compile(builder),
            self.routed.compile(builder),
            self.scale_transpose.compile(builder),
        ];
        let cfg = &self.resolved.raw_cfg;
        compose_block(
            format!(
                "{} (VllmFp8MoeSharedStreamLocalWorklet) [local (1 GPU); FP8 block routed, {} \
                 experts top-{}; BF16 shared on aux stream at <= {} rows]",
                self.name, cfg.num_experts, cfg.routed.top_k, SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            ),
            routed_branch,
            &self.shared,
            &self.shared_serial,
            &self.add,
            builder,
        )
    }

    pub fn eval(&self, input: &VllmFp8MoeSharedStreamLocalWorkletInput, ev: &mut Evaluator) {
        let rows = input.num_tokens;
        let zero = rows == 0;
        push_or_zero(&self.router, SingleGemmKernelInput { m: rows }, zero, ev);
        self.routed
            .eval(&Glm53RoutedMoeLocalWorkletInput { num_tokens: rows }, ev);
        push_or_zero(
            &self.scale_transpose,
            ElementwiseKernelInput { num_tokens: rows },
            zero,
            ev,
        );
        eval_shared_and_add(rows, &self.shared, &self.shared_serial, &self.add, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::cache::interp::CoverageFlags;
    use crate::timing::expert_demand::ExpertDemand;
    use crate::timing::routing::RoutingDistribution;
    use crate::timing::{LeafMetrics, SlotInput};

    use super::super::bf16_gated_shared_expert_local::tests::cfg as shared_cfg;

    fn cfg() -> VllmFp8MoeSharedStreamLocalWorkletConfig {
        VllmFp8MoeSharedStreamLocalWorkletConfig {
            hidden: 2560.into(),
            num_experts: 512.into(),
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".into(),
            router_backends: vec!["torch_linear_vllm"],
            scale_transpose_backends: vec!["torch_strided_copy"],
            add_backends: vec!["torch"],
            shared: shared_cfg(),
            routed: Glm53RoutedMoeLocalWorkletConfig {
                hidden: 2560.into(),
                moe_intermediate: 640.into(),
                num_experts: 512.into(),
                ep_size: 1,
                top_k: 10,
                n_group: 0,
                topk_group: 0,
                routed_scaling_numerator: 1,
                routed_scaling_denominator: 1,
                precision: Glm53WeightPrecision::Fp8Block,
                routing_method: "renormalize_naive".into(),
                activation_dtype: DType::Bf16,
                gpu_name: "NVIDIA B200".into(),
                quant_backends: vec!["vllm_cuda"],
                fused_moe_backends: vec!["flashinfer_trtllm_fp8_block_sm100"],
                expert_demand: ExpertDemand::popularity(&RoutingDistribution::uniform(512), 1),
                folded_rank_position: 0,
            },
        }
    }

    fn enumerate_worklet() -> VllmFp8MoeSharedStreamLocalWorklet {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        VllmFp8MoeSharedStreamLocalWorklet::build(
            "mtp.mlp".into(),
            VllmFp8MoeSharedStreamLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap()
    }

    #[test]
    fn routed_path_is_row_major_fp8_quant_plus_scale_transpose() {
        let r = VllmFp8MoeSharedStreamLocalWorklet::resolve_config(&cfg());
        let Glm53ActivationQuantConfig::Fp8(quant) = &r.routed.input_quant else {
            panic!("expected an FP8 input quant");
        };
        assert_eq!(quant.scale_format, "ue8m0_row_major");
        assert_eq!(r.routed.fused_moe.weight_format, DType::Fp8E4m3);
        assert_eq!(r.routed.fused_moe.routing_method, "renormalize_naive");
        // 2560 / 128 groups x 4 B.
        assert_eq!(
            (
                r.scale_transpose.input_bytes_per_token.get(),
                r.scale_transpose.output_bytes_per_token.get()
            ),
            (80, 80)
        );
        assert_eq!((r.router.n.get(), r.router.k.get()), (512, 2560));
    }

    #[test]
    fn the_stream_group_holds_router_routed_and_transpose_against_the_shared_expert() {
        let worklet = enumerate_worklet();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        let names: Vec<_> = tree
            .slots
            .iter()
            .map(|slot| slot.name.strip_prefix("mtp.mlp.").unwrap().to_string())
            .collect();
        assert_eq!(
            &names[..4],
            [
                "router",
                "routed.input_quant",
                "routed.fused_moe",
                "routed_scale_transpose"
            ]
        );
        assert_eq!(names.last().unwrap(), "shared_routed_add");
        assert_eq!(tree.n_slots(), 4 + 6 + 6 + 1);
    }

    #[test]
    fn an_empty_call_fills_every_slot_with_zeros() {
        let worklet = enumerate_worklet();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        let mut metrics = vec![LeafMetrics::MISS; tree.n_slots()];
        let mut inputs: Vec<SlotInput> = Vec::new();
        let mut ev = Evaluator::with_inputs(&mut metrics, &mut inputs);
        worklet.eval(
            &VllmFp8MoeSharedStreamLocalWorkletInput { num_tokens: 0 },
            &mut ev,
        );
        assert_eq!(ev.filled(), tree.n_slots());
        assert!(metrics.iter().all(|m| m.coverage == CoverageFlags::EMPTY));
    }
}
