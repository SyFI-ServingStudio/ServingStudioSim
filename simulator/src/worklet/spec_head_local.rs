//! Local vocabulary head of a speculative step: row gather, lm_head GEMM and,
//! for a draft head, the greedy token reduction.
//!
//! One worklet serves both heads of an MTP deployment (vLLM Model Runner V2,
//! fork 892da082):
//!
//! - **Target head** (`model_runner.py:1560-1584`): `hidden[logits_indices]`
//!   gathers the logits rows, then `compute_logits` runs the untied BF16
//!   lm_head. The logits go to the rejection sampler unreduced, so its
//!   `logits_reduce` leaf is always zero (`greedy_argmax = false`; the leaf
//!   stays compiled so every head names the same locations); their reduction
//!   is `spec_sample_local`'s `logits_stats`.
//! - **Draft head** (`spec_decode/speculator.py:400-432`): the request
//!   endpoints are gathered (`index_select`), projected by the shared lm_head
//!   (the target's: the checkpoint gives the drafter no dedicated head), and
//!   reduced by `logits.argmax(dim=-1)` (`greedy_argmax = true`).
//!
//! The gather and the argmax are elementwise byte placeholders: the gather
//! moves one hidden row plus an int64 index per row; the argmax reads one
//! logits row (`logits_dtype`, BF16 in the capture's `ArgMaxOps<BFloat16>`),
//! priced as one same-size streaming pass over the row (in = out = row bytes;
//! see `spec_sample_local` for why a tiny output is not used). The lm_head GEMM
//! is measured. TP1: no collective.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};
use crate::worklet::glm53_common::{atomic, elementwise, push_or_zero};

const INDEX_BYTES: u32 = 8;

#[derive(Clone, Debug)]
pub struct SpecHeadLocalWorkletConfig {
    pub gpu_name: String,
    pub hidden: Dim,
    pub vocab_size: Dim,
    pub activation_dtype: DType,
    pub lm_head_dtype: DType,
    /// Dtype `compute_logits` returns.
    pub logits_dtype: DType,
    /// The head reduces its logits to one greedy token per row (draft head).
    pub greedy_argmax: bool,
    pub elementwise_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct SpecHeadLocalWorkletResolved {
    pub raw_cfg: SpecHeadLocalWorkletConfig,
    pub logits_gather: ElementwiseKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub logits_reduce: ElementwiseKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct SpecHeadLocalWorkletInput {
    /// Rows that get logits: target logits rows, or draft request endpoints.
    pub rows: u32,
}

pub struct SpecHeadLocalWorklet {
    pub name: String,
    pub logits_gather: Op<ElementwiseKernel>,
    pub lm_head: Op<SingleGemmKernel>,
    pub logits_reduce: Op<ElementwiseKernel>,
    resolved: SpecHeadLocalWorkletResolved,
}

impl SpecHeadLocalWorklet {
    pub fn resolve_config(cfg: &SpecHeadLocalWorkletConfig) -> SpecHeadLocalWorkletResolved {
        let hidden_bytes = cfg
            .hidden
            .get()
            .checked_mul(cfg.activation_dtype.size_bytes())
            .expect("hidden row bytes overflow u32");
        let logits_row_bytes = cfg
            .vocab_size
            .get()
            .checked_mul(cfg.logits_dtype.size_bytes())
            .expect("logits row bytes overflow u32");
        assert!(hidden_bytes > 0, "hidden must be positive");
        assert!(logits_row_bytes > 0, "vocab_size must be positive");
        let ew = &cfg.elementwise_backends;
        SpecHeadLocalWorkletResolved {
            logits_gather: elementwise(ew, &cfg.gpu_name, hidden_bytes + INDEX_BYTES, hidden_bytes),
            lm_head: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.vocab_size.clone(),
                k: cfg.hidden.clone(),
                dtype: cfg.lm_head_dtype,
            },
            logits_reduce: elementwise(ew, &cfg.gpu_name, logits_row_bytes, logits_row_bytes),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: SpecHeadLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = &resolved;
        Ok(Self {
            logits_gather: atomic(
                &name,
                "logits_gather",
                r.logits_gather.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            lm_head: atomic(
                &name,
                "lm_head",
                r.lm_head.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            logits_reduce: atomic(
                &name,
                "logits_reduce",
                r.logits_reduce.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (SpecHeadLocalWorklet) [local (1 GPU); hidden={}; vocab={}; gather/argmax=elementwise placeholders]",
                self.name,
                cfg.hidden.get(),
                cfg.vocab_size.get()
            ),
            child: Box::new(CostNode::Sum(vec![
                self.logits_gather.compile(builder),
                self.lm_head.compile(builder),
                self.logits_reduce.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &SpecHeadLocalWorkletInput, ev: &mut Evaluator) {
        let rows = input.rows;
        let none = rows == 0;
        let tokens = ElementwiseKernelInput { num_tokens: rows };
        push_or_zero(&self.logits_gather, tokens.clone(), none, ev);
        push_or_zero(&self.lm_head, SingleGemmKernelInput { m: rows }, none, ev);
        let reduce = self.resolved.raw_cfg.greedy_argmax;
        push_or_zero(&self.logits_reduce, tokens, none || !reduce, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SpecHeadLocalWorkletConfig {
        SpecHeadLocalWorkletConfig {
            gpu_name: "NVIDIA B200".to_string(),
            hidden: Dim::param("hidden", 2560),
            vocab_size: Dim::param("vocab_size", 248_320),
            activation_dtype: DType::Bf16,
            lm_head_dtype: DType::Bf16,
            logits_dtype: DType::Bf16,
            greedy_argmax: true,
            elementwise_backends: vec!["torch"],
            gemm_backends: vec!["torch_linear_vllm"],
        }
    }

    #[test]
    fn resolve_sizes_lm_head_and_logits_row_from_config() {
        let r = SpecHeadLocalWorklet::resolve_config(&cfg());
        assert_eq!((r.lm_head.n.get(), r.lm_head.k.get()), (248_320, 2560));
        assert_eq!(r.logits_reduce.input_bytes_per_token.get(), 248_320 * 2);
        assert_eq!(r.logits_reduce.output_bytes_per_token.get(), 248_320 * 2);
        assert_eq!(r.logits_gather.output_bytes_per_token.get(), 2560 * 2);
    }
}
