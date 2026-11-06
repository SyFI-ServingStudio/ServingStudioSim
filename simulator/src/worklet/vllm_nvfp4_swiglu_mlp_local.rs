//! A dense SwiGLU MLP with ModelOpt NVFP4 weights on one GPU, as vLLM's
//! compiled graph launches it: NVFP4 quant of the input
//! (`scaled_fp4_quant`, swizzled E4M3 group scales), gate/up GEMM, the
//! Inductor SiLU-and-mul, NVFP4 quant of the intermediate, down GEMM.
//!
//! The sibling of `glm53_mlp_local` for a recipe whose activation is the
//! measured `silu_and_mul` kind instead of a byte placeholder. The norm in
//! front and the residual add behind belong to the arch.
//!
//! `weight_only` is the Hopper path: without FP4 tensor cores vLLM runs NVFP4
//! linears as Marlin W4A16 (`MarlinNvFp4LinearKernel`), so the BF16 activation is
//! never quantized and both quant leaves drop out.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    Nvfp4QuantKernel, Nvfp4QuantKernelConfig, Nvfp4QuantKernelInput, SiluAndMulKernel,
    SiluAndMulKernelConfig, SiluAndMulKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, push_or_zero};
use super::glm53_mlp_local::{NVFP4_GROUP_SIZE, NVFP4_SWIZZLED_SCALE_FORMAT};

#[derive(Clone, Debug)]
pub struct VllmNvfp4SwigluMlpLocalWorkletConfig {
    pub hidden: u32,
    pub intermediate: u32,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub quant_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub activation_backends: Vec<&'static str>,
    /// W4A16 (Marlin): no activation quant before either GEMM.
    pub weight_only: bool,
}

#[derive(Clone, Debug)]
pub struct VllmNvfp4SwigluMlpLocalWorkletResolved {
    pub raw_cfg: VllmNvfp4SwigluMlpLocalWorkletConfig,
    pub gate_up_quant: Option<Nvfp4QuantKernelConfig>,
    pub gate_up: SingleGemmKernelConfig,
    pub silu_and_mul: SiluAndMulKernelConfig,
    pub down_quant: Option<Nvfp4QuantKernelConfig>,
    pub down: SingleGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct VllmNvfp4SwigluMlpLocalWorkletInput {
    pub num_tokens: u32,
}

pub struct VllmNvfp4SwigluMlpLocalWorklet {
    pub name: String,
    gate_up_quant: Option<Op<Nvfp4QuantKernel>>,
    gate_up: Op<SingleGemmKernel>,
    silu_and_mul: Op<SiluAndMulKernel>,
    down_quant: Option<Op<Nvfp4QuantKernel>>,
    down: Op<SingleGemmKernel>,
    resolved: VllmNvfp4SwigluMlpLocalWorkletResolved,
}

impl VllmNvfp4SwigluMlpLocalWorklet {
    pub fn resolve_config(
        cfg: &VllmNvfp4SwigluMlpLocalWorkletConfig,
    ) -> VllmNvfp4SwigluMlpLocalWorkletResolved {
        let quant = |hidden_size: u32| {
            (!cfg.weight_only).then(|| Nvfp4QuantKernelConfig {
                backends: cfg.quant_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: hidden_size.into(),
                group_size: NVFP4_GROUP_SIZE,
                input_dtype: cfg.activation_dtype,
                scale_format: NVFP4_SWIZZLED_SCALE_FORMAT.to_string(),
            })
        };
        let gemm = |n: u32, k: u32| SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: n.into(),
            k: k.into(),
            dtype: DType::Nvfp4E2m1,
        };
        VllmNvfp4SwigluMlpLocalWorkletResolved {
            gate_up_quant: quant(cfg.hidden),
            gate_up: gemm(2 * cfg.intermediate, cfg.hidden),
            silu_and_mul: SiluAndMulKernelConfig {
                backends: cfg.activation_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_dim: cfg.intermediate.into(),
                dtype: cfg.activation_dtype,
            },
            down_quant: quant(cfg.intermediate),
            down: gemm(cfg.hidden, cfg.intermediate),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: VllmNvfp4SwigluMlpLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            gate_up_quant: r
                .gate_up_quant
                .map(|c| atomic(n, "gate_up_input_quant", c, Nvfp4QuantKernel::build, bridge))
                .transpose()?,
            gate_up: atomic(n, "gate_up", r.gate_up, SingleGemmKernel::build, bridge)?,
            silu_and_mul: atomic(
                n,
                "silu_and_mul",
                r.silu_and_mul,
                SiluAndMulKernel::build,
                bridge,
            )?,
            down_quant: r
                .down_quant
                .map(|c| atomic(n, "down_input_quant", c, Nvfp4QuantKernel::build, bridge))
                .transpose()?,
            down: atomic(n, "down", r.down, SingleGemmKernel::build, bridge)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (VllmNvfp4SwigluMlpLocalWorklet) [local (1 GPU); H={}, I={}]",
                self.name, cfg.hidden, cfg.intermediate
            ),
            child: Box::new(CostNode::Sum(
                [
                    self.gate_up_quant.as_ref().map(|q| q.compile(builder)),
                    Some(self.gate_up.compile(builder)),
                    Some(self.silu_and_mul.compile(builder)),
                    self.down_quant.as_ref().map(|q| q.compile(builder)),
                    Some(self.down.compile(builder)),
                ]
                .into_iter()
                .flatten()
                .collect(),
            )),
        }
    }

    pub fn eval(&self, input: &VllmNvfp4SwigluMlpLocalWorkletInput, ev: &mut Evaluator) {
        let num_tokens = input.num_tokens;
        let rows = SingleGemmKernelInput { m: num_tokens };
        let quant = Nvfp4QuantKernelInput { num_tokens };
        if let Some(q) = &self.gate_up_quant {
            push_or_zero(q, quant.clone(), false, ev);
        }
        push_or_zero(&self.gate_up, rows.clone(), false, ev);
        push_or_zero(
            &self.silu_and_mul,
            SiluAndMulKernelInput { num_tokens },
            false,
            ev,
        );
        if let Some(q) = &self.down_quant {
            push_or_zero(q, quant, false, ev);
        }
        push_or_zero(&self.down, rows, false, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_are_the_profiled_qwen38_rows() {
        let r =
            VllmNvfp4SwigluMlpLocalWorklet::resolve_config(&VllmNvfp4SwigluMlpLocalWorkletConfig {
                hidden: 5120,
                intermediate: 17408,
                activation_dtype: DType::Bf16,
                gpu_name: "NVIDIA B200".into(),
                quant_backends: vec!["vllm_cuda"],
                gemm_backends: vec!["flashinfer_cutedsl"],
                activation_backends: vec!["vllm_inductor"],
                weight_only: false,
            });
        assert_eq!((r.gate_up.n.get(), r.gate_up.k.get()), (34816, 5120));
        assert_eq!((r.down.n.get(), r.down.k.get()), (5120, 17408));
        let (gq, dq) = (r.gate_up_quant.unwrap(), r.down_quant.unwrap());
        assert_eq!((gq.hidden_size.get(), dq.hidden_size.get()), (5120, 17408));
        assert_eq!(dq.scale_format, "swizzled_e4m3");
        assert_eq!(r.silu_and_mul.hidden_dim.get(), 17408);
    }

    #[test]
    fn weight_only_drops_both_quants() {
        let r =
            VllmNvfp4SwigluMlpLocalWorklet::resolve_config(&VllmNvfp4SwigluMlpLocalWorkletConfig {
                hidden: 5120,
                intermediate: 17408,
                activation_dtype: DType::Bf16,
                gpu_name: "NVIDIA H200".into(),
                quant_backends: vec![],
                gemm_backends: vec!["vllm_marlin_nvfp4"],
                activation_backends: vec!["vllm_inductor"],
                weight_only: true,
            });
        assert!(r.gate_up_quant.is_none() && r.down_quant.is_none());
        assert_eq!(r.gate_up.backends, vec!["vllm_marlin_nvfp4"]);
    }
}
