//! Device-local vLLM MoE block: router, NVFP4 routed experts and a gated BF16
//! shared expert, with the shared expert on vLLM's auxiliary stream for small
//! batches.
//!
//! Launch order (fork `model_executor/layers/fused_moe/runner/moe_runner.py`
//! `_forward_impl`, 865-935):
//!
//! 1. `maybe_forward_async` launches the shared expert on the aux stream when
//!    `T <= VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` (256,
//!    `runner/shared_experts.py:104-116`). `T` is the row count the block
//!    runs at, which is the padded size under a CUDA graph.
//! 2. The runner-held router `GateLinear` (BF16 logits, a cuBLAS GEMM) runs
//!    **after** that launch on the main stream. It belongs to the routed branch
//!    and overlaps the shared expert.
//! 3. The routed callable: input quant plus the monolithic fused MoE (routing,
//!    FC1, activation, FC2, finalize).
//! 4. The main stream waits for the shared output, then adds it
//!    (`CUDAFunctor_add`).
//!
//! Above the threshold, `_apply_quant_method` runs the shared expert inline
//! (`NO_OVERLAP`): router, shared, routed, add, all serial. The diverse_100
//! capture shows both shapes: decode and draft steps put the shared chain on its
//! own stream, and the 2048-token mixed steps run it inline between the router
//! and the routed quant.
//!
//! The CostTree is `Sum[Parallel{Sum[router, routed], shared}, shared_serial,
//! add]`. Both shared copies are compiled and exactly one is non-zero per call.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::bf16_gated_shared_expert_local::{
    Bf16GatedSharedExpertLocalWorklet, Bf16GatedSharedExpertLocalWorkletConfig,
    Bf16GatedSharedExpertLocalWorkletInput, Bf16GatedSharedExpertLocalWorkletResolved,
};
use super::deepseek_v41_common::SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD;
use super::glm53_common::{atomic, elementwise, push_or_zero};
use super::nvfp4_moe_local::{
    Nvfp4MoeLocalWorklet, Nvfp4MoeLocalWorkletConfig, Nvfp4MoeLocalWorkletInput,
    Nvfp4MoeLocalWorkletResolved,
};

/// Whether the shared expert runs on the aux stream beside the routed branch
/// for a block call over `num_tokens` rows (padded under a CUDA graph).
pub fn shared_on_aux_stream(num_tokens: u32) -> bool {
    num_tokens > 0 && num_tokens <= SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
}

/// Which shared-expert copies are zeroed this call: `(aux-stream copy, inline
/// copy)`. Exactly one runs for a non-empty call.
pub(super) fn shared_copies_off(num_tokens: u32) -> (bool, bool) {
    let overlapped = shared_on_aux_stream(num_tokens);
    (!overlapped, overlapped || num_tokens == 0)
}

#[derive(Clone, Debug)]
pub struct VllmMoeSharedStreamLocalWorkletConfig {
    pub hidden: Dim,
    pub num_experts: Dim,
    pub activation_dtype: DType,
    pub gpu_name: String,
    /// The router `GateLinear` (BF16 weight, BF16 logits).
    pub router_backends: Vec<&'static str>,
    /// The shared + routed output add (`CUDAFunctor_add`; eager torch
    /// elementwise, fan-in 1 at the output bytes).
    pub add_backends: Vec<&'static str>,
    pub shared: Bf16GatedSharedExpertLocalWorkletConfig,
    /// The routed callable. Its `expert_demand` is the corpus when the
    /// deployment supplies one, otherwise a synthetic (uniform) marginal.
    pub routed: Nvfp4MoeLocalWorkletConfig,
}

#[derive(Clone, Debug)]
pub struct VllmMoeSharedStreamLocalWorkletResolved {
    pub raw_cfg: VllmMoeSharedStreamLocalWorkletConfig,
    pub router: SingleGemmKernelConfig,
    pub shared: Bf16GatedSharedExpertLocalWorkletResolved,
    pub routed: Nvfp4MoeLocalWorkletResolved,
    pub add: ElementwiseKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct VllmMoeSharedStreamLocalWorkletInput {
    /// Rows the block runs at: the padded size under a CUDA graph.
    pub num_tokens: u32,
}

pub struct VllmMoeSharedStreamLocalWorklet {
    pub name: String,
    pub router: Op<SingleGemmKernel>,
    pub routed: Nvfp4MoeLocalWorklet,
    pub shared: Bf16GatedSharedExpertLocalWorklet,
    pub shared_serial: Bf16GatedSharedExpertLocalWorklet,
    pub add: Op<ElementwiseKernel>,
    resolved: VllmMoeSharedStreamLocalWorkletResolved,
}

impl VllmMoeSharedStreamLocalWorklet {
    pub fn resolve_config(
        cfg: &VllmMoeSharedStreamLocalWorkletConfig,
    ) -> VllmMoeSharedStreamLocalWorkletResolved {
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
        assert_eq!(
            (cfg.routed.ep_size, cfg.routed.tp_size),
            (1, 1),
            "a local block holds every expert whole"
        );
        VllmMoeSharedStreamLocalWorkletResolved {
            router: router_config(cfg.hidden.clone(), cfg.num_experts.clone(), cfg),
            shared: resolve_shared(&cfg.shared, &cfg.hidden),
            routed: Nvfp4MoeLocalWorklet::resolve_config(&cfg.routed),
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
        resolved: VllmMoeSharedStreamLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        let (shared, shared_serial) = build_shared_pair(n, &r.shared, bridge)?;
        Ok(Self {
            router: atomic(n, "router", r.router, SingleGemmKernel::build, bridge)?,
            routed: Nvfp4MoeLocalWorklet::build(format!("{n}.routed"), r.routed, bridge)?,
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
        let routed_branch = vec![self.router.compile(builder), self.routed.compile(builder)];
        let cfg = &self.resolved.raw_cfg;
        compose_block(
            format!(
                "{} (VllmMoeSharedStreamLocalWorklet) [local (1 GPU); NVFP4 routed, {} experts \
                 top-{}; BF16 shared on aux stream at <= {} rows]",
                self.name, cfg.num_experts, cfg.routed.top_k, SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            ),
            routed_branch,
            &self.shared,
            &self.shared_serial,
            &self.add,
            builder,
        )
    }

    pub fn eval(&self, input: &VllmMoeSharedStreamLocalWorkletInput, ev: &mut Evaluator) {
        let rows = input.num_tokens;
        let zero = rows == 0;
        push_or_zero(&self.router, SingleGemmKernelInput { m: rows }, zero, ev);
        self.routed
            .eval_or_zero(&Nvfp4MoeLocalWorkletInput { num_tokens: rows }, zero, ev);
        eval_shared_and_add(rows, &self.shared, &self.shared_serial, &self.add, ev);
    }
}

pub(super) fn router_config(
    hidden: Dim,
    num_experts: Dim,
    cfg: &VllmMoeSharedStreamLocalWorkletConfig,
) -> SingleGemmKernelConfig {
    router_gemm(
        &cfg.router_backends,
        &cfg.gpu_name,
        hidden,
        num_experts,
        cfg.activation_dtype,
    )
}

/// The router `GateLinear`: `[T, hidden] x [hidden, num_experts]`, BF16 out.
pub(super) fn router_gemm(
    backends: &[&'static str],
    gpu_name: &str,
    hidden: Dim,
    num_experts: Dim,
    dtype: DType,
) -> SingleGemmKernelConfig {
    assert_eq!(dtype, DType::Bf16, "the router GateLinear runs in BF16");
    SingleGemmKernelConfig {
        backends: backends.to_vec(),
        gpu_name: gpu_name.to_string(),
        n: num_experts,
        k: hidden,
        dtype,
    }
}

pub(super) fn resolve_shared(
    cfg: &Bf16GatedSharedExpertLocalWorkletConfig,
    hidden: &Dim,
) -> Bf16GatedSharedExpertLocalWorkletResolved {
    assert_eq!(
        cfg.hidden.get(),
        hidden.get(),
        "shared-expert hidden must match the block"
    );
    Bf16GatedSharedExpertLocalWorklet::resolve_config(cfg)
}

/// `shared_out + routed_out`, priced as the fan-in-1 byte pass at the add's
/// output bytes (`glue_map.md` D3: 0.95-1.39 against the real add; the
/// two-rows-in pass measured 1.6-9.4x).
pub(super) fn add_config(
    backends: &[&'static str],
    gpu_name: &str,
    hidden: &Dim,
    dtype: DType,
) -> ElementwiseKernelConfig {
    let row = hidden.get() * dtype.size_bytes();
    elementwise(backends, gpu_name, row, row)
}

/// The aux-stream copy (`{name}.shared`) and the inline copy
/// (`{name}.shared_serial`) of the shared expert.
pub(super) fn build_shared_pair(
    name: &str,
    resolved: &Bf16GatedSharedExpertLocalWorkletResolved,
    bridge: &PerfApiBridge,
) -> Result<
    (
        Bf16GatedSharedExpertLocalWorklet,
        Bf16GatedSharedExpertLocalWorklet,
    ),
    BuildError,
> {
    Ok((
        Bf16GatedSharedExpertLocalWorklet::build(
            format!("{name}.shared"),
            resolved.clone(),
            bridge,
        )?,
        Bf16GatedSharedExpertLocalWorklet::build(
            format!("{name}.shared_serial"),
            resolved.clone(),
            bridge,
        )?,
    ))
}

/// `Sum[Parallel{Sum[routed_branch], shared}, shared_serial, add]`.
pub(super) fn compose_block(
    label: String,
    routed_branch: Vec<CostNode>,
    shared: &Bf16GatedSharedExpertLocalWorklet,
    shared_serial: &Bf16GatedSharedExpertLocalWorklet,
    add: &Op<ElementwiseKernel>,
    builder: &mut CostTreeBuilder,
) -> CostNode {
    let routed = CostNode::Sum(routed_branch);
    let concurrent = CostNode::Parallel {
        overlap: 1.0,
        children: vec![routed, shared.compile(builder)],
    };
    CostNode::Labeled {
        label,
        child: Box::new(CostNode::Sum(vec![
            concurrent,
            shared_serial.compile(builder),
            add.compile(builder),
        ])),
    }
}

/// Fill the two shared copies (one non-zero) and the add, after the routed
/// branch's leaves.
pub(super) fn eval_shared_and_add(
    rows: u32,
    shared: &Bf16GatedSharedExpertLocalWorklet,
    shared_serial: &Bf16GatedSharedExpertLocalWorklet,
    add: &Op<ElementwiseKernel>,
    ev: &mut Evaluator,
) {
    let (concurrent_off, serial_off) = shared_copies_off(rows);
    let input = Bf16GatedSharedExpertLocalWorkletInput { batch_tokens: rows };
    shared.eval_or_zero(&input, concurrent_off, ev);
    shared_serial.eval_or_zero(&input, serial_off, ev);
    push_or_zero(
        add,
        ElementwiseKernelInput { num_tokens: rows },
        rows == 0,
        ev,
    );
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::timing::cache::interp::CoverageFlags;
    use crate::timing::expert_demand::ExpertDemand;
    use crate::timing::routing::RoutingDistribution;
    use crate::timing::{LeafMetrics, SlotInput};

    use super::super::bf16_gated_shared_expert_local::tests::cfg as shared_cfg;

    pub(crate) fn routed_cfg() -> Nvfp4MoeLocalWorkletConfig {
        Nvfp4MoeLocalWorkletConfig {
            hidden: 2560.into(),
            moe_intermediate: 640.into(),
            num_experts: 512.into(),
            ep_size: 1,
            tp_size: 1,
            top_k: 10,
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".into(),
            quant_backends: vec!["vllm_cuda"],
            moe_backends: vec!["flashinfer_trtllm_sm100"],
            weight_format: DType::Nvfp4E2m1,
            activation_format: None,
            group_size: 16,
            routing_method: "renormalize_naive".into(),
            n_group: 0,
            topk_group: 0,
            routed_scaling_numerator: 1,
            routed_scaling_denominator: 1,
            expert_demand: ExpertDemand::popularity(&RoutingDistribution::uniform(512), 1),
            folded_rank_position: 0,
        }
    }

    fn cfg() -> VllmMoeSharedStreamLocalWorkletConfig {
        VllmMoeSharedStreamLocalWorkletConfig {
            hidden: 2560.into(),
            num_experts: 512.into(),
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".into(),
            router_backends: vec!["torch_linear_vllm"],
            add_backends: vec!["torch"],
            shared: shared_cfg(),
            routed: routed_cfg(),
        }
    }

    fn enumerate_worklet() -> VllmMoeSharedStreamLocalWorklet {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        VllmMoeSharedStreamLocalWorklet::build(
            "model.mlp".into(),
            VllmMoeSharedStreamLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap()
    }

    #[test]
    fn router_routed_and_add_shapes_derive_from_the_config() {
        let r = VllmMoeSharedStreamLocalWorklet::resolve_config(&cfg());
        assert_eq!((r.router.n.get(), r.router.k.get()), (512, 2560));
        assert_eq!(r.router.dtype, DType::Bf16);
        assert_eq!(r.routed.fused_moe.routing_method, "renormalize_naive");
        assert_eq!(
            (r.routed.fused_moe.n_group, r.routed.fused_moe.topk_group),
            (0, 0)
        );
        assert_eq!(r.routed.fused_moe.num_local_experts.get(), 512);
        assert_eq!(
            (
                r.add.input_bytes_per_token.get(),
                r.add.output_bytes_per_token.get()
            ),
            (5120, 5120)
        );
    }

    #[test]
    fn the_router_overlaps_the_shared_expert_inside_the_parallel_group() {
        let worklet = enumerate_worklet();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        let names: Vec<_> = tree
            .slots
            .iter()
            .map(|slot| slot.name.strip_prefix("model.mlp.").unwrap().to_string())
            .collect();
        assert_eq!(
            names,
            [
                "router",
                "routed.input_quant",
                "routed.fused_moe",
                "shared.gate_up",
                "shared.act_and_mul",
                "shared.down",
                "shared.gate",
                "shared.gate_sigmoid",
                "shared.gate_mul",
                "shared_serial.gate_up",
                "shared_serial.act_and_mul",
                "shared_serial.down",
                "shared_serial.gate",
                "shared_serial.gate_sigmoid",
                "shared_serial.gate_mul",
                "shared_routed_add",
            ]
        );
        let CostNode::Labeled { child, .. } = tree.root else {
            panic!("expected Labeled");
        };
        let CostNode::Sum(children) = *child else {
            panic!("expected Sum");
        };
        assert_eq!(children.len(), 3);
        let CostNode::Parallel {
            children: streams, ..
        } = &children[0]
        else {
            panic!("expected the stream group first");
        };
        assert_eq!(streams.len(), 2);
    }

    #[test]
    fn the_stream_gate_is_256_rows() {
        assert!(!shared_on_aux_stream(0));
        assert!(shared_on_aux_stream(1));
        assert!(shared_on_aux_stream(256));
        assert!(!shared_on_aux_stream(257));
        assert!(!shared_on_aux_stream(2048));
    }

    #[test]
    fn exactly_one_shared_copy_runs_and_it_switches_at_256_rows() {
        assert_eq!(shared_copies_off(1), (false, true));
        assert_eq!(shared_copies_off(256), (false, true));
        assert_eq!(shared_copies_off(257), (true, false));
        assert_eq!(shared_copies_off(8192), (true, false));
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
            &VllmMoeSharedStreamLocalWorkletInput { num_tokens: 0 },
            &mut ev,
        );
        assert_eq!(ev.filled(), tree.n_slots());
        assert!(metrics.iter().all(|m| m.coverage == CoverageFlags::EMPTY));
    }

    #[test]
    #[should_panic(expected = "every expert whole")]
    fn rejects_a_sharded_routed_callable() {
        let mut bad = cfg();
        bad.routed.ep_size = 2;
        let _ = VllmMoeSharedStreamLocalWorklet::resolve_config(&bad);
    }
}
