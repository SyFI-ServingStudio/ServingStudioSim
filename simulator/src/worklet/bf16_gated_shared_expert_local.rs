//! One device-local BF16 shared expert with a scalar sigmoid gate (vLLM).
//!
//! `Qwen3NextMLP(expert_gate=shared_expert_gate)` (fork
//! `model_executor/models/qwen3_next.py:206-216`, `qwen2_moe.py:120`): the
//! BF16 sibling of `qwen36_shared_expert_local` with no FP8 quant leaves.
//! Order: gate/up GEMM, `act_and_mul`, down GEMM, the `[hidden, 1]` gate GEMM,
//! eager `sigmoid`, and the broadcast `sigmoid(gate) * out` multiply.
//!
//! * `act_and_mul` is `SiluAndMul.forward_cuda`, vLLM's `act_and_mul_kernel`
//!   custom op, priced on the `silu_and_mul` kind at the intermediate width.
//! * The gate multiply is a strided broadcast `MulFunctor`; it is priced at its
//!   output bytes on the realization the arch passes (`glue_map.md` D2: the
//!   strided copy matched it to 0.98-1.05, the plain torch pass did not).
//!
//! Every dimension comes from the config; routing, routed experts, the
//! shared + routed add and stream placement belong to the consumer.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, SiluAndMulKernel,
    SiluAndMulKernelConfig, SiluAndMulKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, elementwise, push_or_zero};

#[derive(Clone, Debug)]
pub struct Bf16GatedSharedExpertLocalWorkletConfig {
    pub hidden: Dim,
    /// `shared_expert_intermediate_size`.
    pub intermediate: Dim,
    pub activation_dtype: DType,
    pub gpu_name: String,
    /// BF16 `F.linear` realization for gate/up, down and the scalar gate.
    pub gemm_backends: Vec<&'static str>,
    /// `silu_and_mul` realization for `act_and_mul`.
    pub act_backends: Vec<&'static str>,
    /// Eager `sigmoid` on the `[T, 1]` gate.
    pub sigmoid_backends: Vec<&'static str>,
    /// Broadcast `sigmoid(gate) * out`, priced at output bytes.
    pub gate_mul_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Bf16GatedSharedExpertLocalWorkletResolved {
    pub raw_cfg: Bf16GatedSharedExpertLocalWorkletConfig,
    pub gate_up: SingleGemmKernelConfig,
    pub act_and_mul: SiluAndMulKernelConfig,
    pub down: SingleGemmKernelConfig,
    pub gate: SingleGemmKernelConfig,
    pub gate_sigmoid: ElementwiseKernelConfig,
    pub gate_mul: ElementwiseKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Bf16GatedSharedExpertLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct Bf16GatedSharedExpertLocalWorklet {
    pub name: String,
    pub gate_up: Op<SingleGemmKernel>,
    pub act_and_mul: Op<SiluAndMulKernel>,
    pub down: Op<SingleGemmKernel>,
    pub gate: Op<SingleGemmKernel>,
    pub gate_sigmoid: Op<ElementwiseKernel>,
    pub gate_mul: Op<ElementwiseKernel>,
    resolved: Bf16GatedSharedExpertLocalWorkletResolved,
}

impl Bf16GatedSharedExpertLocalWorklet {
    pub fn resolve_config(
        cfg: &Bf16GatedSharedExpertLocalWorkletConfig,
    ) -> Bf16GatedSharedExpertLocalWorkletResolved {
        assert_eq!(
            cfg.activation_dtype,
            DType::Bf16,
            "the BF16 shared expert runs on BF16 activations"
        );
        let hidden = cfg.hidden.get();
        let width = cfg.intermediate.get();
        assert!(
            hidden > 0 && width > 0,
            "shared-expert dims must be non-zero"
        );
        let bytes = cfg.activation_dtype.size_bytes();
        let gpu = cfg.gpu_name.as_str();
        let gemm = |n: Dim, k: Dim| SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n,
            k,
            dtype: cfg.activation_dtype,
        };
        let row_bytes = hidden * bytes;
        Bf16GatedSharedExpertLocalWorkletResolved {
            gate_up: gemm((2 * width).into(), cfg.hidden.clone()),
            act_and_mul: SiluAndMulKernelConfig {
                backends: cfg.act_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_dim: cfg.intermediate.clone(),
                dtype: cfg.activation_dtype,
            },
            down: gemm(cfg.hidden.clone(), cfg.intermediate.clone()),
            // `shared_expert_gate`: ReplicatedLinear(hidden, 1, quant_config=None).
            gate: gemm(1.into(), cfg.hidden.clone()),
            gate_sigmoid: elementwise(&cfg.sigmoid_backends, gpu, bytes, bytes),
            gate_mul: elementwise(&cfg.gate_mul_backends, gpu, row_bytes, row_bytes),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Bf16GatedSharedExpertLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            gate_up: atomic(n, "gate_up", r.gate_up, SingleGemmKernel::build, bridge)?,
            act_and_mul: atomic(
                n,
                "act_and_mul",
                r.act_and_mul,
                SiluAndMulKernel::build,
                bridge,
            )?,
            down: atomic(n, "down", r.down, SingleGemmKernel::build, bridge)?,
            gate: atomic(n, "gate", r.gate, SingleGemmKernel::build, bridge)?,
            gate_sigmoid: atomic(
                n,
                "gate_sigmoid",
                r.gate_sigmoid,
                ElementwiseKernel::build,
                bridge,
            )?,
            gate_mul: atomic(n, "gate_mul", r.gate_mul, ElementwiseKernel::build, bridge)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (Bf16GatedSharedExpertLocalWorklet) [local (1 GPU); hidden={}; width={}]",
                self.name, cfg.hidden, cfg.intermediate,
            ),
            child: Box::new(CostNode::Sum(vec![
                self.gate_up.compile(builder),
                self.act_and_mul.compile(builder),
                self.down.compile(builder),
                self.gate.compile(builder),
                self.gate_sigmoid.compile(builder),
                self.gate_mul.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Bf16GatedSharedExpertLocalWorkletInput, ev: &mut Evaluator) {
        self.eval_or_zero(input, false, ev);
    }

    /// Push zeros instead when this copy of the call does not run (a consumer
    /// that compiles a concurrent and a serial copy, of which one runs).
    pub fn eval_or_zero(
        &self,
        input: &Bf16GatedSharedExpertLocalWorkletInput,
        zero: bool,
        ev: &mut Evaluator,
    ) {
        let rows = input.batch_tokens;
        let zero = zero || rows == 0;
        let gemm = SingleGemmKernelInput { m: rows };
        let ew = ElementwiseKernelInput { num_tokens: rows };
        push_or_zero(&self.gate_up, gemm.clone(), zero, ev);
        let act = SiluAndMulKernelInput { num_tokens: rows };
        push_or_zero(&self.act_and_mul, act, zero, ev);
        push_or_zero(&self.down, gemm.clone(), zero, ev);
        push_or_zero(&self.gate, gemm, zero, ev);
        push_or_zero(&self.gate_sigmoid, ew.clone(), zero, ev);
        push_or_zero(&self.gate_mul, ew, zero, ev);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn cfg() -> Bf16GatedSharedExpertLocalWorkletConfig {
        Bf16GatedSharedExpertLocalWorkletConfig {
            hidden: 2560.into(),
            intermediate: 640.into(),
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".into(),
            gemm_backends: vec!["torch_linear_vllm"],
            act_backends: vec!["vllm_cuda"],
            sigmoid_backends: vec!["torch"],
            gate_mul_backends: vec!["torch_strided_copy"],
        }
    }

    #[test]
    fn shapes_and_byte_rates_derive_from_the_config() {
        let r = Bf16GatedSharedExpertLocalWorklet::resolve_config(&cfg());
        assert_eq!((r.gate_up.n.get(), r.gate_up.k.get()), (1280, 2560));
        assert_eq!((r.down.n.get(), r.down.k.get()), (2560, 640));
        assert_eq!((r.gate.n.get(), r.gate.k.get()), (1, 2560));
        assert!([&r.gate_up, &r.down, &r.gate]
            .iter()
            .all(|gemm| gemm.dtype == DType::Bf16));
        let rates = |c: &ElementwiseKernelConfig| {
            (
                c.input_bytes_per_token.get(),
                c.output_bytes_per_token.get(),
            )
        };
        assert_eq!(r.act_and_mul.hidden_dim.get(), 640);
        assert_eq!(rates(&r.gate_sigmoid), (2, 2));
        assert_eq!(rates(&r.gate_mul), (5120, 5120));
        assert_eq!(r.gate_mul.backends, ["torch_strided_copy"]);
    }

    #[test]
    #[should_panic(expected = "BF16 activations")]
    fn rejects_non_bf16_activations() {
        let mut bad = cfg();
        bad.activation_dtype = DType::Fp16;
        let _ = Bf16GatedSharedExpertLocalWorklet::resolve_config(&bad);
    }
}
