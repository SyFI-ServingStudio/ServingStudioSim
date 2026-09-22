//! Kimi-K3 layer-0 dense SwiGLU worklet.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::kimi_k3_common::{build_atomic, eval_atomic_or_zero};

const HIDDEN: u32 = 7_168;
const INTERMEDIATE: u32 = 33_792;

#[cfg(test)]
const SOURCE_ORDER: [&str; 3] = ["gate_up", "situ", "down"];

#[derive(Clone, Debug)]
pub struct KimiK3DenseLocalWorkletConfig {
    pub gpu_name: String,
    pub hidden: Dim,
    pub intermediate: Dim,
    pub dtype: DType,
    pub gemm_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct KimiK3DenseLocalWorkletResolved {
    pub raw_cfg: KimiK3DenseLocalWorkletConfig,
    pub gate_up: SingleGemmKernelConfig,
    pub situ: ElementwiseKernelConfig,
    pub down: SingleGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct KimiK3DenseLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct KimiK3DenseLocalWorklet {
    pub name: String,
    pub gate_up: Op<SingleGemmKernel>,
    pub situ: Op<ElementwiseKernel>,
    pub down: Op<SingleGemmKernel>,
    resolved: KimiK3DenseLocalWorkletResolved,
}

impl KimiK3DenseLocalWorklet {
    pub fn resolve_config(cfg: &KimiK3DenseLocalWorkletConfig) -> KimiK3DenseLocalWorkletResolved {
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid KimiK3DenseLocalWorkletConfig: {reason}"));
        KimiK3DenseLocalWorkletResolved {
            gate_up: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: (2 * cfg.intermediate.get()).into(),
                k: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            situ: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: (2 * cfg.intermediate.get() * 2).into(),
                output_bytes_per_token: (cfg.intermediate.get() * 2).into(),
            },
            down: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden.clone(),
                k: cfg.intermediate.clone(),
                dtype: cfg.dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: KimiK3DenseLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, crate::timing::BuildError> {
        Ok(Self {
            gate_up: build_atomic(
                &name,
                "gate_up",
                resolved.gate_up.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            situ: build_atomic(
                &name,
                "situ",
                resolved.situ.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            down: build_atomic(
                &name,
                "down",
                resolved.down.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (KimiK3DenseLocalWorklet) [hidden={}; intermediate={}]",
                self.name, self.resolved.raw_cfg.hidden, self.resolved.raw_cfg.intermediate
            ),
            child: Box::new(CostNode::Sum(vec![
                self.gate_up.compile(builder),
                self.situ.compile(builder),
                self.down.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &KimiK3DenseLocalWorkletInput, evaluator: &mut Evaluator) {
        let rows = input.batch_tokens;
        eval_atomic_or_zero(
            &self.gate_up,
            SingleGemmKernelInput { m: rows },
            rows == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.situ,
            ElementwiseKernelInput { num_tokens: rows },
            rows == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.down,
            SingleGemmKernelInput { m: rows },
            rows == 0,
            evaluator,
        );
    }
}

fn validate_config(cfg: &KimiK3DenseLocalWorkletConfig) -> Result<(), String> {
    if cfg.hidden.get() != HIDDEN {
        return Err(format!("hidden must be {HIDDEN}, got {}", cfg.hidden));
    }
    if cfg.intermediate.get() != INTERMEDIATE {
        return Err(format!(
            "intermediate must be {INTERMEDIATE}, got {}",
            cfg.intermediate
        ));
    }
    if cfg.dtype != DType::Bf16 {
        return Err("K3 dense activations use bf16".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> KimiK3DenseLocalWorkletConfig {
        KimiK3DenseLocalWorkletConfig {
            gpu_name: "NVIDIA B200".to_string(),
            hidden: HIDDEN.into(),
            intermediate: INTERMEDIATE.into(),
            dtype: DType::Bf16,
            gemm_backends: vec!["sglang_bf16_auto"],
            elementwise_backends: vec!["triton"],
        }
    }

    #[test]
    fn dense_swiglu_shapes_and_byte_rates_are_frozen() {
        assert_eq!(SOURCE_ORDER.len(), 3);
        let resolved = KimiK3DenseLocalWorklet::resolve_config(&config());
        assert_eq!(resolved.gate_up.n, 67_584);
        assert_eq!(resolved.gate_up.k, HIDDEN);
        assert_eq!(resolved.situ.input_bytes_per_token, 135_168);
        assert_eq!(resolved.situ.output_bytes_per_token, 67_584);
        assert_eq!(resolved.down.k, INTERMEDIATE);
        assert_eq!(resolved.down.n, HIDDEN);
    }
}
