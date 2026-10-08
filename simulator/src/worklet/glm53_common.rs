//! Shared construction helpers for the GLM-5.3-Flash worklets.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernelConfig, Fp8PerTokenGroupQuantKernel, Fp8PerTokenGroupQuantKernelConfig,
    Fp8PerTokenGroupQuantKernelInput, Nvfp4QuantKernel, Nvfp4QuantKernelConfig,
    Nvfp4QuantKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Evaluator, LeafMetrics, PerfApiBridge, Probe, SlotInput,
};

/// Build one atomic op named `{prefix}.{suffix}`.
pub(crate) fn atomic<K, C, F>(
    prefix: &str,
    suffix: &str,
    config: C,
    build_kernel: F,
    bridge: &PerfApiBridge,
) -> Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError>,
{
    let name = format!("{prefix}.{suffix}");
    Ok(Op::new(
        name.clone(),
        Arc::new(build_kernel(name, config, bridge)?),
    ))
}

/// Push the op's metrics, or zero when the launch does not happen this
/// iteration. The slot exists either way (INV-1).
pub(crate) fn push_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        op.kernel.eval(&input)
    };
    ev.push(metrics, || input.clone().into());
}

/// `n` identical launches of one leaf: the leaf is costed once and folded.
pub(crate) fn repeated<K: Probe>(op: &Op<K>, n: u32, builder: &mut CostTreeBuilder) -> CostNode {
    CostNode::Scale {
        n,
        child: Box::new(op.compile(builder)),
    }
}

/// A byte-sized elementwise placeholder.
pub(crate) fn elementwise(
    backends: &[&'static str],
    gpu_name: &str,
    input_bytes_per_token: u32,
    output_bytes_per_token: u32,
) -> ElementwiseKernelConfig {
    ElementwiseKernelConfig {
        backends: backends.to_vec(),
        gpu_name: gpu_name.to_string(),
        input_bytes_per_token: input_bytes_per_token.into(),
        output_bytes_per_token: output_bytes_per_token.into(),
    }
}

/// The activation quant ahead of one quantized GEMM or fused MoE.
#[derive(Clone, Debug)]
pub enum Glm53ActivationQuantConfig {
    Fp8(Fp8PerTokenGroupQuantKernelConfig),
    Nvfp4(Nvfp4QuantKernelConfig),
}

pub enum Glm53ActivationQuant {
    Fp8(Op<Fp8PerTokenGroupQuantKernel>),
    Nvfp4(Op<Nvfp4QuantKernel>),
}

impl Glm53ActivationQuant {
    pub(crate) fn build(
        prefix: &str,
        suffix: &str,
        cfg: Glm53ActivationQuantConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        Ok(match cfg {
            Glm53ActivationQuantConfig::Fp8(cfg) => Self::Fp8(atomic(
                prefix,
                suffix,
                cfg,
                Fp8PerTokenGroupQuantKernel::build,
                bridge,
            )?),
            Glm53ActivationQuantConfig::Nvfp4(cfg) => Self::Nvfp4(atomic(
                prefix,
                suffix,
                cfg,
                Nvfp4QuantKernel::build,
                bridge,
            )?),
        })
    }

    pub(crate) fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        match self {
            Self::Fp8(op) => op.compile(builder),
            Self::Nvfp4(op) => op.compile(builder),
        }
    }

    pub(crate) fn eval_or_zero(&self, num_tokens: u32, zero: bool, ev: &mut Evaluator) {
        match self {
            Self::Fp8(op) => push_or_zero(
                op,
                Fp8PerTokenGroupQuantKernelInput { num_tokens },
                zero,
                ev,
            ),
            Self::Nvfp4(op) => push_or_zero(op, Nvfp4QuantKernelInput { num_tokens }, zero, ev),
        }
    }
}

/// How the checkpoint stored one module's weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm53WeightPrecision {
    Bf16,
    Fp8Block,
    Nvfp4,
}

impl Glm53WeightPrecision {
    pub fn gemm_dtype(self) -> DType {
        match self {
            Self::Bf16 => DType::Bf16,
            Self::Fp8Block => DType::Fp8E4m3,
            Self::Nvfp4 => DType::Nvfp4E2m1,
        }
    }
}
