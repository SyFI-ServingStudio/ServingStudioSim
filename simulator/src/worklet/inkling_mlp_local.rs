//! An Inkling BF16 gated MLP on one TP rank: the dense FFN of layers
//! `0..dense_mlp_idx` (`ink/mlp.py`) and the MoE layers' sink experts
//! (`ink/moe.py`, both sinks' weights merged on each rank).
//!
//! Gate/up GEMM, a Triton activation, down GEMM. The dense FFN then scales its
//! output in place by the checkpoint's global scale (`x.mul_(global_scale)`);
//! the sink epilogue instead folds each token's sink gate weight into the
//! activation (`_sink_epilogue`). The output is a partial sum the arch's
//! Lamport call reduces.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, elementwise, push_or_zero};

#[derive(Clone, Debug)]
pub struct InklingMlpLocalWorkletConfig {
    pub hidden: Dim,
    /// Gated intermediate width on this rank (both sinks together for the
    /// sink MLP).
    pub intermediate: Dim,
    /// The dense FFN's trailing in-place `global_scale` multiply.
    pub global_scale_mul: bool,
    pub dtype: DType,
    pub gpu_name: String,
    pub gemm_backends: Vec<&'static str>,
    pub activation_backends: Vec<&'static str>,
    pub scale_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct InklingMlpLocalWorkletResolved {
    pub raw_cfg: InklingMlpLocalWorkletConfig,
    pub gate_up: SingleGemmKernelConfig,
    pub activation: ElementwiseKernelConfig,
    pub down: SingleGemmKernelConfig,
    pub global_scale: Option<ElementwiseKernelConfig>,
}

#[derive(Clone, Debug, Default)]
pub struct InklingMlpLocalWorkletInput {
    pub num_tokens: u32,
}

pub struct InklingMlpLocalWorklet {
    pub name: String,
    pub gate_up: Op<SingleGemmKernel>,
    pub activation: Op<ElementwiseKernel>,
    pub down: Op<SingleGemmKernel>,
    pub global_scale: Option<Op<ElementwiseKernel>>,
    resolved: InklingMlpLocalWorkletResolved,
}

impl InklingMlpLocalWorklet {
    pub fn resolve_config(cfg: &InklingMlpLocalWorkletConfig) -> InklingMlpLocalWorkletResolved {
        let gemm = |n: Dim, k: Dim| SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n,
            k,
            dtype: cfg.dtype,
        };
        let act_out = cfg.intermediate.get() * cfg.dtype.size_bytes();
        let hidden_bytes = cfg.hidden.get() * cfg.dtype.size_bytes();
        InklingMlpLocalWorkletResolved {
            gate_up: gemm(cfg.intermediate.clone() * 2, cfg.hidden.clone()),
            // Reads gate and up, writes one intermediate row.
            activation: elementwise(
                &cfg.activation_backends,
                &cfg.gpu_name,
                2 * act_out,
                act_out,
            ),
            down: gemm(cfg.hidden.clone(), cfg.intermediate.clone()),
            global_scale: cfg.global_scale_mul.then(|| {
                elementwise(&cfg.scale_backends, &cfg.gpu_name, hidden_bytes, hidden_bytes)
            }),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: InklingMlpLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            gate_up: atomic(n, "gate_up", r.gate_up, SingleGemmKernel::build, bridge)?,
            activation: atomic(
                n,
                "activation",
                r.activation,
                ElementwiseKernel::build,
                bridge,
            )?,
            down: atomic(n, "down", r.down, SingleGemmKernel::build, bridge)?,
            global_scale: r
                .global_scale
                .map(|cfg| atomic(n, "global_scale", cfg, ElementwiseKernel::build, bridge))
                .transpose()?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        let mut children = vec![
            self.gate_up.compile(builder),
            self.activation.compile(builder),
            self.down.compile(builder),
        ];
        children.extend(self.global_scale.as_ref().map(|op| op.compile(builder)));
        CostNode::Labeled {
            label: format!(
                "{} (InklingMlpLocalWorklet) [TP rank; BF16; H={}, I={}]",
                self.name, cfg.hidden, cfg.intermediate
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &InklingMlpLocalWorkletInput, ev: &mut Evaluator) {
        self.eval_or_zero(input, false, ev);
    }

    /// Push zeros instead when this copy does not run (the sink experts'
    /// aux-stream and serial copies, of which one runs).
    pub fn eval_or_zero(
        &self,
        input: &InklingMlpLocalWorkletInput,
        zero: bool,
        ev: &mut Evaluator,
    ) {
        let tokens = input.num_tokens;
        let zero = zero || tokens == 0;
        let rows = SingleGemmKernelInput { m: tokens };
        let bytes = ElementwiseKernelInput { num_tokens: tokens };
        push_or_zero(&self.gate_up, rows.clone(), zero, ev);
        push_or_zero(&self.activation, bytes.clone(), zero, ev);
        push_or_zero(&self.down, rows, zero, ev);
        if let Some(op) = &self.global_scale {
            push_or_zero(op, bytes, zero, ev);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(intermediate: u32, global_scale_mul: bool) -> InklingMlpLocalWorkletConfig {
        InklingMlpLocalWorkletConfig {
            hidden: 4096.into(),
            intermediate: intermediate.into(),
            global_scale_mul,
            dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".into(),
            gemm_backends: vec!["torch_linear_vllm"],
            activation_backends: vec!["triton"],
            scale_backends: vec!["torch"],
        }
    }

    #[test]
    fn dense_and_sink_shapes_match_the_capture() {
        // Dense FFN at TP4: 16384 / 4 = 4096 per rank, gate_up N=8192.
        let dense = InklingMlpLocalWorklet::resolve_config(&cfg(4096, true));
        assert_eq!((dense.gate_up.n.get(), dense.gate_up.k.get()), (8192, 4096));
        assert_eq!((dense.down.n.get(), dense.down.k.get()), (4096, 4096));
        let scale = dense.global_scale.expect("dense FFN scales its output");
        assert_eq!(scale.input_bytes_per_token.get(), 8192);
        // Two sinks x 2048 / 4 = 1024 per rank: gate_up N=2048, down K=1024.
        let sink = InklingMlpLocalWorklet::resolve_config(&cfg(1024, false));
        assert_eq!((sink.gate_up.n.get(), sink.down.k.get()), (2048, 1024));
        assert!(sink.global_scale.is_none());
    }
}
