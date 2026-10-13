//! Local rejection-sampling placeholders of a speculative verify step.
//!
//! After the target head, Model Runner V2's rejection sampler
//! (fork 892da082, `v1/worker/gpu/spec_decode/rejection_sampler_utils.py`)
//! launches:
//!
//! - `_compute_local_logits_stats_kernel` over `(logits rows, vocab blocks)`
//!   (`:1088`): reads every target logits row once and writes per-block
//!   argmax / max / sum-exp partials (`VOCAB_BLOCK_SIZE` 8192, `:1070`).
//!   Priced as `logits_stats`, one elementwise row per logits row.
//! - `_rejection_kernel` per request (`:1198`), `_resample_kernel` over
//!   `(requests, vocab blocks)` (`:1249`, rereads one logits row per request),
//!   `_insert_resampled_kernel` and the accepted-count bookkeeping launches.
//!   Priced together as `rejection_resample`, one elementwise row per request
//!   reading one logits row.
//!
//! Both leaves are elementwise byte placeholders (decision D8), with byte rates
//! from the vocabulary and the logits dtype only. Each is priced as one
//! same-size streaming pass over the logits row (in = out = row bytes). The
//! elementwise kernel realizes in/out bytes as a fan-in of round(in/out) that
//! it loops over serially inside one output tile, so a vocab-wide reduction
//! priced with its real tiny output becomes a single-CTA latency chain instead
//! of the many-CTA memory-bound read the real kernels do.
//!
//! For scale, in the diverse_100 capture
//! (`wt-qf-capture/logs/20261013_1_qf_diverse100/phase0`) the sample
//! phase was 0.93% of kernel time, of which the target lm_head GEMM was ~0.7%
//! (priced in `spec_head_local`) and these launches ~0.15%: logits stats
//! ~20 us/iter, resample ~25, rejection ~7, the small rest ~15. The capture is
//! documentation, not a calibration input. The post-step bookkeeping launches
//! (`_post_update_kernel`, `postprocess_mamba_fused_kernel`, ~0.07%) are
//! explicitly not modeled.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};
use crate::worklet::glm53_common::{atomic, elementwise, push_or_zero};

#[derive(Clone, Debug)]
pub struct SpecSampleLocalWorkletConfig {
    pub gpu_name: String,
    pub vocab_size: Dim,
    /// Dtype of the target logits the sampler reads.
    pub logits_dtype: DType,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct SpecSampleLocalWorkletResolved {
    pub raw_cfg: SpecSampleLocalWorkletConfig,
    pub logits_stats: ElementwiseKernelConfig,
    pub rejection_resample: ElementwiseKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct SpecSampleLocalWorkletInput {
    /// Target logits rows (prefill requests plus verify rows).
    pub logits_rows: u32,
    /// Requests sampled this step.
    pub requests: u32,
}

pub struct SpecSampleLocalWorklet {
    pub name: String,
    pub logits_stats: Op<ElementwiseKernel>,
    pub rejection_resample: Op<ElementwiseKernel>,
    resolved: SpecSampleLocalWorkletResolved,
}

impl SpecSampleLocalWorklet {
    pub fn resolve_config(cfg: &SpecSampleLocalWorkletConfig) -> SpecSampleLocalWorkletResolved {
        let vocab = cfg.vocab_size.get();
        assert!(vocab > 0, "vocab_size must be positive");
        let row_bytes = vocab
            .checked_mul(cfg.logits_dtype.size_bytes())
            .expect("logits row bytes overflow u32");
        let ew = &cfg.elementwise_backends;
        SpecSampleLocalWorkletResolved {
            logits_stats: elementwise(ew, &cfg.gpu_name, row_bytes, row_bytes),
            rejection_resample: elementwise(ew, &cfg.gpu_name, row_bytes, row_bytes),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: SpecSampleLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        Ok(Self {
            logits_stats: atomic(
                &name,
                "logits_stats",
                resolved.logits_stats.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            rejection_resample: atomic(
                &name,
                "rejection_resample",
                resolved.rejection_resample.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (SpecSampleLocalWorklet) [local (1 GPU); vocab={}; elementwise placeholders]",
                self.name,
                self.resolved.raw_cfg.vocab_size.get()
            ),
            child: Box::new(CostNode::Sum(vec![
                self.logits_stats.compile(builder),
                self.rejection_resample.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &SpecSampleLocalWorkletInput, ev: &mut Evaluator) {
        let (stats, resample) = derive_work(input);
        push_or_zero(&self.logits_stats, stats, input.logits_rows == 0, ev);
        push_or_zero(&self.rejection_resample, resample, input.requests == 0, ev);
    }
}

fn derive_work(
    input: &SpecSampleLocalWorkletInput,
) -> (ElementwiseKernelInput, ElementwiseKernelInput) {
    assert!(
        input.requests <= input.logits_rows,
        "requests {} exceed logits rows {}",
        input.requests,
        input.logits_rows
    );
    (
        ElementwiseKernelInput {
            num_tokens: input.logits_rows,
        },
        ElementwiseKernelInput {
            num_tokens: input.requests,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_and_resample_stream_one_logits_row() {
        let r = SpecSampleLocalWorklet::resolve_config(&SpecSampleLocalWorkletConfig {
            gpu_name: "NVIDIA B200".to_string(),
            vocab_size: Dim::param("vocab_size", 248_320),
            logits_dtype: DType::Bf16,
            elementwise_backends: vec!["triton"],
        });
        for leaf in [&r.logits_stats, &r.rejection_resample] {
            assert_eq!(leaf.input_bytes_per_token.get(), 248_320 * 2);
            assert_eq!(leaf.output_bytes_per_token.get(), 248_320 * 2);
        }
    }

    #[test]
    fn stats_scale_with_rows_and_resample_with_requests() {
        let (stats, resample) = derive_work(&SpecSampleLocalWorkletInput {
            logits_rows: 4 * 8 + 2,
            requests: 10,
        });
        assert_eq!((stats.num_tokens, resample.num_tokens), (34, 10));
    }

    #[test]
    #[should_panic(expected = "exceed logits rows")]
    fn rejects_more_requests_than_rows() {
        derive_work(&SpecSampleLocalWorkletInput {
            logits_rows: 3,
            requests: 4,
        });
    }
}
