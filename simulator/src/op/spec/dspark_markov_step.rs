//! One DSpark Markov step: the sequential half of the DSpark draft head.
//!
//! DSpark's backbone produces one base-logits row per drafted position in a
//! single pass; the Markov head then walks the positions in order, because each
//! position's bias depends on the token sampled at the previous one (fork
//! `v1/worker/gpu/spec_decode/dspark/speculator.py` `_sample_markov_chain`,
//! `model_executor/models/qwen3_dspark.py` `DSparkMarkovHead`). One step, over
//! one row per request:
//!
//! 1. `markov_w1[prev]`: an embedding-row gather of the previous token
//!    (`vectorized_gather_kernel`), `[R] -> [R, markov_rank]` bf16;
//! 2. `markov_w2`: the replicated bf16 projection `[R, markov_rank] -> [R, V]`
//!    (`ParallelLMHead(disable_tp=True)`, one cuBLAS nvjet launch);
//! 3. `base_logits[:, i] + bias`: fp32 base row plus bf16 bias, fp32 out
//!    (one `elementwise_kernel` over `R x V`);
//! 4. greedy argmax (`_argmax_partial` + `_argmax_finish`): the default
//!    `draft_sample_method` is `"greedy"` (`config/speculative.py:699`).
//!
//! The four launches share the step's state (the sampled token feeds the next
//! step's gather), so they are one compound op; the arch folds the
//! `num_speculative_tokens` sequential steps with `Scale`. Only the projection
//! has an L1 kind (`single_gemm`); the gather, the bias add and the argmax are
//! byte placeholders (design D7), sized per request row.
//!
//! Captured TP4 decode iteration 938 (R = 26, device 0, `phase1_split.md`
//! §1.1): per step 8-17 us `markov_w2`, ~11 us bias add, ~5 us argmax pair.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};
use crate::worklet::deepseek_v41_common::{eval_or_zero, placeholder};

/// Token ids are int64 in the speculator's buffers.
const TOKEN_ID_BYTES: u32 = 8;
const BF16: u32 = 2;
const FP32: u32 = 4;

#[derive(Clone, Debug)]
pub struct DsparkMarkovStepConfig {
    pub gpu_name: String,
    /// Draft vocabulary (`draft_vocab_size`, the full vocab for DSpark V4.1).
    pub vocab_size: u32,
    /// `dspark_markov_rank`.
    pub markov_rank: u32,
    pub gemm_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug, Default)]
pub struct DsparkMarkovStepInput {
    /// Rows this step samples: one per request in the draft graph.
    pub num_requests: u32,
}

pub struct DsparkMarkovStepOp {
    pub name: String,
    pub embed_gather: Op<ElementwiseKernel>,
    pub markov_w2: Op<SingleGemmKernel>,
    pub bias_add: Op<ElementwiseKernel>,
    pub argmax: Op<ElementwiseKernel>,
}

/// The four sub-kernel configs of one step, in launch order.
pub(crate) fn sub_configs(
    cfg: &DsparkMarkovStepConfig,
) -> (
    ElementwiseKernelConfig,
    SingleGemmKernelConfig,
    ElementwiseKernelConfig,
    ElementwiseKernelConfig,
) {
    let ew = |input: u32, output: u32| {
        placeholder(&cfg.elementwise_backends, &cfg.gpu_name, input, output)
    };
    let vocab = cfg.vocab_size;
    let rank = cfg.markov_rank;
    (
        ew(TOKEN_ID_BYTES + rank * BF16, rank * BF16),
        SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: vocab.into(),
            k: rank.into(),
            dtype: DType::Bf16,
        },
        ew(vocab * (FP32 + BF16), vocab * FP32),
        ew(vocab * FP32, TOKEN_ID_BYTES),
    )
}

impl DsparkMarkovStepOp {
    pub fn build(
        name: String,
        cfg: DsparkMarkovStepConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let (gather, w2, bias_add, argmax) = sub_configs(&cfg);
        let ew = |slot: &str, config: ElementwiseKernelConfig| -> Result<_, BuildError> {
            let op_name = format!("{name}.{slot}");
            Ok(Op::new(
                op_name.clone(),
                Arc::new(ElementwiseKernel::build(op_name, config, bridge)?),
            ))
        };
        let w2_name = format!("{name}.markov_w2");
        Ok(Self {
            embed_gather: ew("markov_w1_gather", gather)?,
            markov_w2: Op::new(
                w2_name.clone(),
                Arc::new(SingleGemmKernel::build(w2_name, w2, bridge)?),
            ),
            bias_add: ew("bias_add", bias_add)?,
            argmax: ew("greedy_argmax", argmax)?,
            name,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            self.embed_gather.compile(builder),
            self.markov_w2.compile(builder),
            self.bias_add.compile(builder),
            self.argmax.compile(builder),
        ])
    }

    pub fn eval(&self, input: &DsparkMarkovStepInput, ev: &mut Evaluator) {
        let rows = input.num_requests;
        let zero = rows == 0;
        let ew = ElementwiseKernelInput { num_tokens: rows };
        eval_or_zero(&self.embed_gather, ew.clone(), zero, ev);
        eval_or_zero(&self.markov_w2, SingleGemmKernelInput { m: rows }, zero, ev);
        eval_or_zero(&self.bias_add, ew.clone(), zero, ev);
        eval_or_zero(&self.argmax, ew, zero, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markov_projection_is_the_replicated_rank_to_vocab_gemm() {
        let cfg = DsparkMarkovStepConfig {
            gpu_name: "NVIDIA B200".into(),
            vocab_size: 129_280,
            markov_rank: 256,
            gemm_backends: vec!["torch_linear"],
            elementwise_backends: vec!["triton"],
        };
        let (gather, w2, bias_add, argmax) = sub_configs(&cfg);
        assert_eq!(
            (w2.k.get(), w2.n.get(), w2.dtype),
            (256, 129_280, DType::Bf16)
        );
        assert_eq!(gather.output_bytes_per_token.get(), 512);
        // fp32 base row + bf16 bias in, fp32 row out: one V-sized pass.
        assert_eq!(
            bias_add.input_bytes_per_token.get() + bias_add.output_bytes_per_token.get(),
            129_280 * 10
        );
        // A V-wide read reduced to one id keeps its bytes but streams evenly.
        assert_eq!(
            argmax.input_bytes_per_token.get(),
            argmax.output_bytes_per_token.get()
        );
    }
}
