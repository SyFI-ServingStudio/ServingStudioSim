//! A complete compiled decoder invocation on one Trainium2 LNC2 unit.
//! Prefill and decode are mutually exclusive executable slots. Batch validation
//! belongs to L4; this section never sums two phases for the same request.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::kernels::{
    NeuronLlamaDecoderKernel, NeuronLlamaDecoderKernelConfig, NeuronLlamaDecoderKernelInput,
    NeuronLlamaDecoderPhase,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, DType, Dim, Evaluator, LeafMetrics, PerfApiBridge,
};

#[derive(Clone, Debug)]
pub struct NeuronLlamaDecoderLocalWorkletConfig {
    pub hidden: Dim,
    pub intermediate: Dim,
    pub q_heads: Dim,
    pub kv_heads: Dim,
    pub head_dim: Dim,
    pub kv_capacity: Dim,
    pub dtype: DType,
    pub gpu_name: String,
    pub backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct NeuronLlamaDecoderLocalWorkletResolved {
    pub raw_cfg: NeuronLlamaDecoderLocalWorkletConfig,
    pub prefill: NeuronLlamaDecoderKernelConfig,
    pub decode: NeuronLlamaDecoderKernelConfig,
}

pub struct NeuronLlamaDecoderLocalWorkletInput {
    pub phase: NeuronLlamaDecoderPhase,
    pub q_tokens: u32,
}

pub struct NeuronLlamaDecoderLocalWorklet {
    pub name: String,
    pub prefill: Op<NeuronLlamaDecoderKernel>,
    pub decode: Op<NeuronLlamaDecoderKernel>,
    resolved: NeuronLlamaDecoderLocalWorkletResolved,
}

impl NeuronLlamaDecoderLocalWorklet {
    pub fn resolve_config(
        cfg: &NeuronLlamaDecoderLocalWorkletConfig,
    ) -> NeuronLlamaDecoderLocalWorkletResolved {
        let kernel = |phase| NeuronLlamaDecoderKernelConfig {
            backends: cfg.backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            phase,
            batch: Dim::param("batch", 1),
            kv_capacity: cfg.kv_capacity.clone(),
            hidden: cfg.hidden.clone(),
            intermediate: cfg.intermediate.clone(),
            q_heads: cfg.q_heads.clone(),
            kv_heads: cfg.kv_heads.clone(),
            head_dim: cfg.head_dim.clone(),
            dtype: cfg.dtype,
        };
        NeuronLlamaDecoderLocalWorkletResolved {
            raw_cfg: cfg.clone(),
            prefill: kernel(NeuronLlamaDecoderPhase::Prefill),
            decode: kernel(NeuronLlamaDecoderPhase::Decode),
        }
    }

    pub fn build(
        name: String,
        resolved: NeuronLlamaDecoderLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let prefill_name = format!("{name}.prefill");
        let decode_name = format!("{name}.decode");
        Ok(Self {
            name,
            prefill: Op::new(
                prefill_name.clone(),
                Arc::new(NeuronLlamaDecoderKernel::build(
                    prefill_name,
                    resolved.prefill.clone(),
                    bridge,
                )?),
            ),
            decode: Op::new(
                decode_name.clone(),
                Arc::new(NeuronLlamaDecoderKernel::build(
                    decode_name,
                    resolved.decode.clone(),
                    bridge,
                )?),
            ),
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} [local LNC2; batch=1; KV capacity={}]",
                self.name, self.resolved.raw_cfg.kv_capacity
            ),
            child: Box::new(CostNode::Sum(vec![
                self.prefill.compile(builder),
                self.decode.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &NeuronLlamaDecoderLocalWorkletInput, ev: &mut Evaluator) {
        let prefill = input.phase == NeuronLlamaDecoderPhase::Prefill;
        for (op, active) in [(&self.prefill, prefill), (&self.decode, !prefill)] {
            let shape = NeuronLlamaDecoderKernelInput {
                q_tokens: if active { input.q_tokens } else { 0 },
            };
            if active {
                op.eval(&shape, ev);
            } else {
                // Preserve the fixed manifest order; no query or profile for
                // the executable absent from this iteration.
                ev.push(LeafMetrics::ZERO, || shape.into());
            }
        }
    }
}
