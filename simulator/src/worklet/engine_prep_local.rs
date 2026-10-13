//! Local engine-preprocess placeholder group of one serving step.
//!
//! Before the forward pass, Model Runner V2 (fork 892da082) builds the step's
//! inputs on the GPU: about 150 small launches per step in the diverse_100
//! capture (index / copy / fill / arange / scan launches plus
//! `_prepare_pos_seq_lens_kernel`, `_combine_sampled_and_draft_tokens_kernel`,
//! `_gather_block_tables_kernel`, `_compute_slot_mappings_kernel`), the
//! sparse-attention metadata build (`_build_qsa_metadata_kernel`,
//! `models/qwen4_exp/common/qsa_cache.py:431`, gridded over token blocks), and
//! the Mamba align-mode pre-copy (`precopy_mamba_align_fused_kernel`,
//! `v1/worker/mamba_utils.py:1212`, gridded over `(requests, states)`, copying a
//! state only when its request crosses a state block).
//!
//! Decision D8 prices this as one placeholder group of three elementwise
//! leaves: `input_prep` (per request), `qsa_metadata` (per token) and
//! `mamba_align_precopy` (per request). Their byte rates are geometry the arch
//! derives from the config and the KV layout; this worklet only carries them.
//! For scale (documentation only, not a calibration input): the capture's
//! preprocess phase was 0.78% of kernel time, ~230 us/iter of 29.2 ms, of which
//! the pre-copy was ~12 us and the metadata build ~7 us; the rest is the
//! launch-bound input build, which three single launches under-price.

use crate::op::Op;
use crate::timing::kernels::{ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};
use crate::worklet::glm53_common::{atomic, elementwise, push_or_zero};

/// Bytes one unit (a request or a token) reads and writes in a placeholder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitBytes {
    pub input: u32,
    pub output: u32,
}

#[derive(Clone, Debug)]
pub struct EnginePrepLocalWorkletConfig {
    pub gpu_name: String,
    /// Per request: token ids, positions, block-table row and slot mapping.
    pub input_prep_bytes_per_request: UnitBytes,
    /// Per token: the sparse-attention slot mapping and work metadata.
    pub qsa_metadata_bytes_per_token: UnitBytes,
    /// Per request: the align-mode state pre-copy.
    pub mamba_align_precopy_bytes_per_request: UnitBytes,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct EnginePrepLocalWorkletResolved {
    pub raw_cfg: EnginePrepLocalWorkletConfig,
    pub input_prep: ElementwiseKernelConfig,
    pub qsa_metadata: ElementwiseKernelConfig,
    pub mamba_align_precopy: ElementwiseKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct EnginePrepLocalWorkletInput {
    /// Requests scheduled this step.
    pub requests: u32,
    /// Tokens scheduled this step.
    pub tokens: u32,
}

pub struct EnginePrepLocalWorklet {
    pub name: String,
    pub input_prep: Op<ElementwiseKernel>,
    pub qsa_metadata: Op<ElementwiseKernel>,
    pub mamba_align_precopy: Op<ElementwiseKernel>,
    resolved: EnginePrepLocalWorkletResolved,
}

impl EnginePrepLocalWorklet {
    pub fn resolve_config(cfg: &EnginePrepLocalWorkletConfig) -> EnginePrepLocalWorkletResolved {
        let placeholder = |role: &str, bytes: UnitBytes| {
            assert!(
                bytes.input > 0 && bytes.output > 0,
                "{role} byte rates must be positive, got {bytes:?}"
            );
            elementwise(
                &cfg.elementwise_backends,
                &cfg.gpu_name,
                bytes.input,
                bytes.output,
            )
        };
        EnginePrepLocalWorkletResolved {
            input_prep: placeholder("input_prep", cfg.input_prep_bytes_per_request),
            qsa_metadata: placeholder("qsa_metadata", cfg.qsa_metadata_bytes_per_token),
            mamba_align_precopy: placeholder(
                "mamba_align_precopy",
                cfg.mamba_align_precopy_bytes_per_request,
            ),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: EnginePrepLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = &resolved;
        Ok(Self {
            input_prep: atomic(
                &name,
                "input_prep",
                r.input_prep.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            qsa_metadata: atomic(
                &name,
                "qsa_metadata",
                r.qsa_metadata.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            mamba_align_precopy: atomic(
                &name,
                "mamba_align_precopy",
                r.mamba_align_precopy.clone(),
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
                "{} (EnginePrepLocalWorklet) [local (1 GPU); elementwise placeholders; bytes in/out: input_prep {}/{} per request, qsa_metadata {}/{} per token, mamba_align_precopy {}/{} per request]",
                self.name,
                cfg.input_prep_bytes_per_request.input,
                cfg.input_prep_bytes_per_request.output,
                cfg.qsa_metadata_bytes_per_token.input,
                cfg.qsa_metadata_bytes_per_token.output,
                cfg.mamba_align_precopy_bytes_per_request.input,
                cfg.mamba_align_precopy_bytes_per_request.output,
            ),
            child: Box::new(CostNode::Sum(vec![
                self.input_prep.compile(builder),
                self.qsa_metadata.compile(builder),
                self.mamba_align_precopy.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &EnginePrepLocalWorkletInput, ev: &mut Evaluator) {
        let [prep, meta, precopy] = derive_work(input);
        let idle = input.requests == 0;
        push_or_zero(&self.input_prep, prep, idle, ev);
        push_or_zero(&self.qsa_metadata, meta, idle, ev);
        push_or_zero(&self.mamba_align_precopy, precopy, idle, ev);
    }
}

fn derive_work(input: &EnginePrepLocalWorkletInput) -> [ElementwiseKernelInput; 3] {
    assert!(
        input.requests <= input.tokens,
        "requests {} exceed tokens {}",
        input.requests,
        input.tokens
    );
    [
        ElementwiseKernelInput {
            num_tokens: input.requests,
        },
        ElementwiseKernelInput {
            num_tokens: input.tokens,
        },
        ElementwiseKernelInput {
            num_tokens: input.requests,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> EnginePrepLocalWorkletConfig {
        EnginePrepLocalWorkletConfig {
            gpu_name: "NVIDIA B200".to_string(),
            input_prep_bytes_per_request: UnitBytes {
                input: 64,
                output: 32,
            },
            qsa_metadata_bytes_per_token: UnitBytes {
                input: 16,
                output: 16,
            },
            mamba_align_precopy_bytes_per_request: UnitBytes {
                input: 12,
                output: 4,
            },
            elementwise_backends: vec!["triton"],
        }
    }

    #[test]
    fn metadata_scales_with_tokens_and_the_rest_with_requests() {
        let [prep, meta, precopy] = derive_work(&EnginePrepLocalWorkletInput {
            requests: 9,
            tokens: 2048 + 32,
        });
        assert_eq!(
            (prep.num_tokens, meta.num_tokens, precopy.num_tokens),
            (9, 2080, 9)
        );
    }

    #[test]
    fn resolve_carries_the_arch_byte_rates() {
        let r = EnginePrepLocalWorklet::resolve_config(&cfg());
        assert_eq!(r.qsa_metadata.input_bytes_per_token.get(), 16);
        assert_eq!(r.mamba_align_precopy.output_bytes_per_token.get(), 4);
    }

    #[test]
    #[should_panic(expected = "byte rates must be positive")]
    fn rejects_an_empty_placeholder() {
        let mut c = cfg();
        c.input_prep_bytes_per_request.output = 0;
        EnginePrepLocalWorklet::resolve_config(&c);
    }
}
