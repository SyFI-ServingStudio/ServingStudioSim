//! GLM-5.3-Flash SwiGLU MLP on one TP rank, at the checkpoint's precision.
//!
//! One shape serves both the dense FFN of layers 0-2 and the MoE layers'
//! shared expert (`Glm5NextMLP`, `glm5next/nvidia/model.py`): gate/up GEMM,
//! `act_and_mul`, down GEMM. A quantized linear adds its activation quant in
//! front of each GEMM:
//!
//! - FP8 block (`zai-org/GLM-5.3-Flash-FP8`): packed-UE8M0 per-token-group FP8
//!   quant, DeepGEMM `fp8_gemm_nt`;
//! - NVFP4 (`nvidia/GLM-5.3-Flash-NVFP4`): `scaled_fp4_quant` with swizzled
//!   E4M3 group scales, then the NVFP4 GEMM;
//! - BF16 (the NVFP4 checkpoint's shared expert): no quant.
//!
//! The activation is a byte-sized elementwise placeholder. The TP all-reduce
//! belongs to the arch.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    Fp8PerTokenGroupQuantKernelConfig, Nvfp4QuantKernelConfig, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::glm53_common::{
    atomic, elementwise, push_or_zero, Glm53ActivationQuant, Glm53ActivationQuantConfig,
    Glm53WeightPrecision,
};

/// DeepGEMM's Blackwell operand layout: four UE8M0 exponents per int32.
pub const PACKED_SCALE_FORMAT: &str = "ue8m0_packed_int32";
/// The 128x4-tile swizzled E4M3 scales an NVFP4 linear GEMM reads.
pub const NVFP4_SWIZZLED_SCALE_FORMAT: &str = "swizzled_e4m3";
const FP8_QUANT_GROUP_SIZE: u32 = 128;
/// ModelOpt NVFP4's 16-element FP4 group.
pub(crate) const NVFP4_GROUP_SIZE: u32 = 16;

#[derive(Clone, Debug)]
pub struct Glm53MlpLocalWorkletConfig {
    pub hidden: Dim,
    /// Intermediate width on this rank.
    pub intermediate: Dim,
    pub precision: Glm53WeightPrecision,
    pub activation_dtype: DType,
    pub gpu_name: String,
    /// Unused at BF16.
    pub quant_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Glm53MlpLocalWorkletResolved {
    pub raw_cfg: Glm53MlpLocalWorkletConfig,
    pub gate_up_quant: Option<Glm53ActivationQuantConfig>,
    pub gate_up: SingleGemmKernelConfig,
    pub activation: ElementwiseKernelConfig,
    pub down_quant: Option<Glm53ActivationQuantConfig>,
    pub down: SingleGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Glm53MlpLocalWorkletInput {
    pub num_tokens: u32,
}

pub struct Glm53MlpLocalWorklet {
    pub name: String,
    pub gate_up_quant: Option<Glm53ActivationQuant>,
    pub gate_up: Op<SingleGemmKernel>,
    pub activation: Op<ElementwiseKernel>,
    pub down_quant: Option<Glm53ActivationQuant>,
    pub down: Op<SingleGemmKernel>,
    resolved: Glm53MlpLocalWorkletResolved,
}

impl Glm53MlpLocalWorklet {
    pub fn resolve_config(cfg: &Glm53MlpLocalWorkletConfig) -> Glm53MlpLocalWorkletResolved {
        let quant = |hidden_size: Dim| match cfg.precision {
            Glm53WeightPrecision::Bf16 => None,
            Glm53WeightPrecision::Fp8Block => Some(Glm53ActivationQuantConfig::Fp8(
                Fp8PerTokenGroupQuantKernelConfig {
                    backends: cfg.quant_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    hidden_size,
                    group_size: FP8_QUANT_GROUP_SIZE,
                    input_dtype: cfg.activation_dtype,
                    scale_format: PACKED_SCALE_FORMAT.to_string(),
                },
            )),
            Glm53WeightPrecision::Nvfp4 => {
                Some(Glm53ActivationQuantConfig::Nvfp4(Nvfp4QuantKernelConfig {
                    backends: cfg.quant_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    hidden_size,
                    group_size: NVFP4_GROUP_SIZE,
                    input_dtype: cfg.activation_dtype,
                    scale_format: NVFP4_SWIZZLED_SCALE_FORMAT.to_string(),
                }))
            }
        };
        let gemm = |n: Dim, k: Dim| SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n,
            k,
            dtype: cfg.precision.gemm_dtype(),
        };
        let act_out = cfg.intermediate.get() * cfg.activation_dtype.size_bytes();
        Glm53MlpLocalWorkletResolved {
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
        resolved: Glm53MlpLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        let quant = |suffix: &str, cfg: Option<Glm53ActivationQuantConfig>| {
            cfg.map(|cfg| Glm53ActivationQuant::build(n, suffix, cfg, bridge))
                .transpose()
        };
        Ok(Self {
            gate_up_quant: quant("gate_up_input_quant", r.gate_up_quant)?,
            gate_up: atomic(n, "gate_up", r.gate_up, SingleGemmKernel::build, bridge)?,
            activation: atomic(
                n,
                "act_and_mul",
                r.activation,
                ElementwiseKernel::build,
                bridge,
            )?,
            down_quant: quant("down_input_quant", r.down_quant)?,
            down: atomic(n, "down", r.down, SingleGemmKernel::build, bridge)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        let mut children = Vec::with_capacity(5);
        children.extend(self.gate_up_quant.as_ref().map(|q| q.compile(builder)));
        children.push(self.gate_up.compile(builder));
        children.push(self.activation.compile(builder));
        children.extend(self.down_quant.as_ref().map(|q| q.compile(builder)));
        children.push(self.down.compile(builder));
        CostNode::Labeled {
            label: format!(
                "{} (Glm53MlpLocalWorklet) [TP rank; {:?}; H={}, I={}]",
                self.name, cfg.precision, cfg.hidden, cfg.intermediate
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &Glm53MlpLocalWorkletInput, ev: &mut Evaluator) {
        self.eval_or_zero(input, false, ev);
    }

    /// Push zeros instead when this copy of the MLP does not run (the shared
    /// expert's concurrent and serial copies, of which one runs).
    pub fn eval_or_zero(&self, input: &Glm53MlpLocalWorkletInput, zero: bool, ev: &mut Evaluator) {
        let tokens = input.num_tokens;
        let zero = zero || tokens == 0;
        let rows = SingleGemmKernelInput { m: tokens };
        if let Some(quant) = &self.gate_up_quant {
            quant.eval_or_zero(tokens, zero, ev);
        }
        push_or_zero(&self.gate_up, rows.clone(), zero, ev);
        push_or_zero(
            &self.activation,
            ElementwiseKernelInput { num_tokens: tokens },
            zero,
            ev,
        );
        if let Some(quant) = &self.down_quant {
            quant.eval_or_zero(tokens, zero, ev);
        }
        push_or_zero(&self.down, rows, zero, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(intermediate: u32, precision: Glm53WeightPrecision) -> Glm53MlpLocalWorkletConfig {
        Glm53MlpLocalWorkletConfig {
            hidden: 4096.into(),
            intermediate: intermediate.into(),
            precision,
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".into(),
            quant_backends: vec!["vllm_cuda"],
            gemm_backends: vec!["deepgemm"],
            elementwise_backends: vec!["triton"],
        }
    }

    #[test]
    fn dense_and_shared_expert_shapes_match_the_capture() {
        let dense =
            Glm53MlpLocalWorklet::resolve_config(&cfg(3072, Glm53WeightPrecision::Fp8Block));
        assert_eq!((dense.gate_up.n.get(), dense.gate_up.k.get()), (6144, 4096));
        assert_eq!((dense.down.n.get(), dense.down.k.get()), (4096, 3072));
        let Some(Glm53ActivationQuantConfig::Fp8(down_quant)) = &dense.down_quant else {
            panic!("FP8 block MLP quantizes the down input to FP8");
        };
        assert_eq!(down_quant.hidden_size.get(), 3072);
        let shared =
            Glm53MlpLocalWorklet::resolve_config(&cfg(512, Glm53WeightPrecision::Fp8Block));
        assert_eq!(
            (shared.gate_up.n.get(), shared.gate_up.k.get()),
            (1024, 4096)
        );
        assert_eq!((shared.down.n.get(), shared.down.k.get()), (4096, 512));
        let Some(Glm53ActivationQuantConfig::Fp8(gate_up_quant)) = &shared.gate_up_quant else {
            panic!("FP8 block MLP quantizes the gate/up input to FP8");
        };
        assert_eq!(gate_up_quant.scale_format, PACKED_SCALE_FORMAT);
    }

    #[test]
    fn nvfp4_quantizes_both_gemm_inputs_and_bf16_none() {
        let fp4 = Glm53MlpLocalWorklet::resolve_config(&cfg(3072, Glm53WeightPrecision::Nvfp4));
        assert_eq!(fp4.gate_up.dtype, DType::Nvfp4E2m1);
        for (quant, width) in [(&fp4.gate_up_quant, 4096), (&fp4.down_quant, 3072)] {
            let Some(Glm53ActivationQuantConfig::Nvfp4(quant)) = quant else {
                panic!("NVFP4 MLP quantizes each GEMM input to NVFP4");
            };
            assert_eq!(quant.hidden_size.get(), width);
            assert_eq!(quant.group_size, 16);
            assert_eq!(quant.scale_format, NVFP4_SWIZZLED_SCALE_FORMAT);
        }
        let bf16 = Glm53MlpLocalWorklet::resolve_config(&cfg(2048, Glm53WeightPrecision::Bf16));
        assert!(bf16.gate_up_quant.is_none() && bf16.down_quant.is_none());
        assert_eq!(bf16.down.dtype, DType::Bf16);
    }
}
