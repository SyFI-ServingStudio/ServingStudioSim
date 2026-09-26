//! GLM-5.3-Flash FP8 block-scaled SwiGLU MLP on one TP rank.
//!
//! One shape serves both the dense FFN of layers 0-2 and the MoE layers'
//! shared expert (`Glm5NextMLP`, `glm5next/nvidia/model.py`): packed-UE8M0
//! per-token-group FP8 quant, DeepGEMM `fp8_gemm_nt` gate/up, `act_and_mul`,
//! quant, DeepGEMM down. The activation is a byte-sized elementwise
//! placeholder. The TP all-reduce belongs to the arch.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    Fp8PerTokenGroupQuantKernel, Fp8PerTokenGroupQuantKernelConfig,
    Fp8PerTokenGroupQuantKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, elementwise, push_or_zero};

/// DeepGEMM's Blackwell operand layout: four UE8M0 exponents per int32.
pub const PACKED_SCALE_FORMAT: &str = "ue8m0_packed_int32";
const QUANT_GROUP_SIZE: u32 = 128;

#[derive(Clone, Debug)]
pub struct Glm53Fp8MlpLocalWorkletConfig {
    pub hidden: Dim,
    /// Intermediate width on this rank.
    pub intermediate: Dim,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub quant_backends: Vec<&'static str>,
    pub fp8_gemm_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Glm53Fp8MlpLocalWorkletResolved {
    pub raw_cfg: Glm53Fp8MlpLocalWorkletConfig,
    pub gate_up_quant: Fp8PerTokenGroupQuantKernelConfig,
    pub gate_up: SingleGemmKernelConfig,
    pub activation: ElementwiseKernelConfig,
    pub down_quant: Fp8PerTokenGroupQuantKernelConfig,
    pub down: SingleGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Glm53Fp8MlpLocalWorkletInput {
    pub num_tokens: u32,
}

pub struct Glm53Fp8MlpLocalWorklet {
    pub name: String,
    pub gate_up_quant: Op<Fp8PerTokenGroupQuantKernel>,
    pub gate_up: Op<SingleGemmKernel>,
    pub activation: Op<ElementwiseKernel>,
    pub down_quant: Op<Fp8PerTokenGroupQuantKernel>,
    pub down: Op<SingleGemmKernel>,
    resolved: Glm53Fp8MlpLocalWorkletResolved,
}

impl Glm53Fp8MlpLocalWorklet {
    pub fn resolve_config(cfg: &Glm53Fp8MlpLocalWorkletConfig) -> Glm53Fp8MlpLocalWorkletResolved {
        let quant = |hidden_size: Dim| Fp8PerTokenGroupQuantKernelConfig {
            backends: cfg.quant_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden_size,
            group_size: QUANT_GROUP_SIZE,
            input_dtype: cfg.activation_dtype,
            scale_format: PACKED_SCALE_FORMAT.to_string(),
        };
        let gemm = |n: Dim, k: Dim| SingleGemmKernelConfig {
            backends: cfg.fp8_gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n,
            k,
            dtype: DType::Fp8E4m3,
        };
        let act_out = cfg.intermediate.get() * cfg.activation_dtype.size_bytes();
        Glm53Fp8MlpLocalWorkletResolved {
            gate_up_quant: quant(cfg.hidden.clone()),
            gate_up: gemm(cfg.intermediate.clone() * 2, cfg.hidden.clone()),
            // SiluAndMul reads gate and up, writes one intermediate row.
            activation: elementwise(
                &cfg.elementwise_backends,
                &cfg.gpu_name,
                2 * act_out,
                act_out,
            ),
            down_quant: quant(cfg.intermediate.clone()),
            down: gemm(cfg.hidden.clone(), cfg.intermediate.clone()),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Glm53Fp8MlpLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            gate_up_quant: atomic(
                n,
                "gate_up_input_quant",
                r.gate_up_quant,
                Fp8PerTokenGroupQuantKernel::build,
                bridge,
            )?,
            gate_up: atomic(n, "gate_up", r.gate_up, SingleGemmKernel::build, bridge)?,
            activation: atomic(
                n,
                "act_and_mul",
                r.activation,
                ElementwiseKernel::build,
                bridge,
            )?,
            down_quant: atomic(
                n,
                "down_input_quant",
                r.down_quant,
                Fp8PerTokenGroupQuantKernel::build,
                bridge,
            )?,
            down: atomic(n, "down", r.down, SingleGemmKernel::build, bridge)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (Glm53Fp8MlpLocalWorklet) [TP rank; H={}, I={}]",
                self.name, cfg.hidden, cfg.intermediate
            ),
            child: Box::new(CostNode::Sum(vec![
                self.gate_up_quant.compile(builder),
                self.gate_up.compile(builder),
                self.activation.compile(builder),
                self.down_quant.compile(builder),
                self.down.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Glm53Fp8MlpLocalWorkletInput, ev: &mut Evaluator) {
        self.eval_or_zero(input, false, ev);
    }

    /// Push zeros instead when this copy of the MLP does not run (the shared
    /// expert's concurrent and serial copies, of which one runs).
    pub fn eval_or_zero(
        &self,
        input: &Glm53Fp8MlpLocalWorkletInput,
        zero: bool,
        ev: &mut Evaluator,
    ) {
        let tokens = input.num_tokens;
        let zero = zero || tokens == 0;
        let quant = Fp8PerTokenGroupQuantKernelInput { num_tokens: tokens };
        let rows = SingleGemmKernelInput { m: tokens };
        push_or_zero(&self.gate_up_quant, quant.clone(), zero, ev);
        push_or_zero(&self.gate_up, rows.clone(), zero, ev);
        push_or_zero(
            &self.activation,
            ElementwiseKernelInput { num_tokens: tokens },
            zero,
            ev,
        );
        push_or_zero(&self.down_quant, quant, zero, ev);
        push_or_zero(&self.down, rows, zero, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(intermediate: u32) -> Glm53Fp8MlpLocalWorkletConfig {
        Glm53Fp8MlpLocalWorkletConfig {
            hidden: 4096.into(),
            intermediate: intermediate.into(),
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".into(),
            quant_backends: vec!["vllm_fork_cuda"],
            fp8_gemm_backends: vec!["deepgemm_vllm_fork"],
            elementwise_backends: vec!["triton"],
        }
    }

    #[test]
    fn dense_and_shared_expert_shapes_match_the_capture() {
        let dense = Glm53Fp8MlpLocalWorklet::resolve_config(&cfg(3072));
        assert_eq!((dense.gate_up.n.get(), dense.gate_up.k.get()), (6144, 4096));
        assert_eq!((dense.down.n.get(), dense.down.k.get()), (4096, 3072));
        assert_eq!(dense.down_quant.hidden_size.get(), 3072);
        let shared = Glm53Fp8MlpLocalWorklet::resolve_config(&cfg(512));
        assert_eq!(
            (shared.gate_up.n.get(), shared.gate_up.k.get()),
            (1024, 4096)
        );
        assert_eq!((shared.down.n.get(), shared.down.k.get()), (4096, 512));
        assert_eq!(shared.gate_up_quant.scale_format, PACKED_SCALE_FORMAT);
    }
}
