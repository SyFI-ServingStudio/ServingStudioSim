//! Compound Gated `DeltaNet` prefill operation.
//!
//! One logical prefill owns the causal-convolution fan-in, the post-convolution
//! projection, and the chunked delta rule. Request-local convolution costs
//! aggregate into one fixed leaf; the other two launches consume the aggregate
//! `(T, L)` geometry. This keeps request count out of the `CostTree` shape.
//!
//! The delta rule is ONE leaf, not six. vLLM resolves the GDN prefill backend
//! to `FlashInfer` on any SM90 part, and that realization is a single fused
//! CUTLASS launch; the six-launch FLA Triton decomposition this operation used
//! to model over-predicted the measured operation by 110% on a
//! Qwen3.6-35B-A3B-FP8 H200 capture, because it round-trips h/w/u/A through HBM
//! five extra times. The `gdn_chunk_*` kinds stay registered for a deployment
//! that really does select the Triton path.

use std::sync::Arc;

use crate::timing::kernels::{
    GdnCausalConvPrefillKernel, GdnCausalConvPrefillKernelConfig, GdnCausalConvPrefillKernelInput,
    GdnChunkDeltaRuleKernel, GdnChunkDeltaRuleKernelConfig, GdnChunkDeltaRuleKernelInput,
    GdnPrefillPostConvKernel, GdnPrefillPostConvKernelConfig, GdnPrefillPostConvKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Evaluator, GdnCausalConvPrefillLog, LeafMetrics,
    PerfApiBridge, Probe,
};

/// Already-resolved configs for the three production launches, in execution
/// order. Backend choice remains owned by each kernel config.
#[derive(Clone, Debug)]
pub struct GdnPrefillOpConfig {
    pub causal_conv: GdnCausalConvPrefillKernelConfig,
    pub post_conv: GdnPrefillPostConvKernelConfig,
    pub chunk_delta_rule: GdnChunkDeltaRuleKernelConfig,
}

/// Fresh-prefill sequences participating in this operation.
#[derive(Clone, Debug, Default)]
pub struct GdnPrefillOpInput {
    pub sequence_lengths: Vec<u32>,
}

pub struct GdnPrefillOp {
    pub name: String,
    pub causal_conv: Arc<GdnCausalConvPrefillKernel>,
    pub post_conv: Arc<GdnPrefillPostConvKernel>,
    pub chunk_delta_rule: Arc<GdnChunkDeltaRuleKernel>,
}

impl GdnPrefillOp {
    pub fn build(
        name: String,
        config: GdnPrefillOpConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let causal_conv = Arc::new(GdnCausalConvPrefillKernel::build(
            format!("{name}.causal_conv"),
            config.causal_conv,
            bridge,
        )?);
        let post_conv = Arc::new(GdnPrefillPostConvKernel::build(
            format!("{name}.post_conv"),
            config.post_conv,
            bridge,
        )?);
        let chunk_delta_rule = Arc::new(GdnChunkDeltaRuleKernel::build(
            format!("{name}.chunk_delta_rule"),
            config.chunk_delta_rule,
            bridge,
        )?);
        Ok(Self {
            name,
            causal_conv,
            post_conv,
            chunk_delta_rule,
        })
    }

    /// Three fixed leaves, independent of the number of prefill sequences.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            leaf(builder, &self.name, "causal_conv", &*self.causal_conv),
            leaf(builder, &self.name, "post_conv", &*self.post_conv),
            leaf(
                builder,
                &self.name,
                "chunk_delta_rule",
                &*self.chunk_delta_rule,
            ),
        ])
    }

    /// Fill the three compile slots in identical order. Empty input is valid
    /// zero-work and deliberately bypasses every L1 cache lookup.
    pub fn eval(&self, input: &GdnPrefillOpInput, ev: &mut Evaluator) {
        eval_with(self, input, ev);
    }
}

/// Private evaluation seam: production delegates to the concrete cached
/// kernels, while tests substitute deterministic probes without a bridge or DB.
trait GdnPrefillEval {
    fn causal_conv(&self, input: &GdnCausalConvPrefillKernelInput) -> LeafMetrics;
    fn post_conv(&self, input: &GdnPrefillPostConvKernelInput) -> LeafMetrics;
    fn chunk_delta_rule(&self, input: &GdnChunkDeltaRuleKernelInput) -> LeafMetrics;
}

impl GdnPrefillEval for GdnPrefillOp {
    fn causal_conv(&self, input: &GdnCausalConvPrefillKernelInput) -> LeafMetrics {
        self.causal_conv.eval(input)
    }

    fn post_conv(&self, input: &GdnPrefillPostConvKernelInput) -> LeafMetrics {
        self.post_conv.eval(input)
    }

    fn chunk_delta_rule(&self, input: &GdnChunkDeltaRuleKernelInput) -> LeafMetrics {
        self.chunk_delta_rule.eval(input)
    }
}

fn eval_with(kernels: &impl GdnPrefillEval, input: &GdnPrefillOpInput, ev: &mut Evaluator) {
    let geometry = derive_geometry(&input.sequence_lengths)
        .unwrap_or_else(|reason| panic!("invalid GdnPrefillOpInput: {reason}"));

    let mut causal_conv_metrics = LeafMetrics::ZERO;
    if geometry.is_some() {
        causal_conv_metrics =
            aggregate_causal_conv(&input.sequence_lengths, |shape| kernels.causal_conv(shape));
    }
    ev.push(causal_conv_metrics, || {
        GdnCausalConvPrefillLog {
            sequence_lengths: input.sequence_lengths.clone(),
        }
        .into()
    });

    let work = work_inputs(geometry);
    let zero = geometry.is_none();
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        kernels.post_conv(&work.post_conv)
    };
    ev.push(metrics, || work.post_conv.into());
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        kernels.chunk_delta_rule(&work.chunk_delta_rule)
    };
    ev.push(metrics, || work.chunk_delta_rule.into());
}

fn leaf<K: Probe>(builder: &mut CostTreeBuilder, name: &str, suffix: &str, kernel: &K) -> CostNode {
    builder.leaf(
        format!("{name}.{suffix}"),
        kernel.kind(),
        kernel.describe_config(),
    )
}

/// `max_sequence_length` is the delta rule's critical path: the inter-chunk
/// recurrence is sequential inside a sequence and independent across sequences,
/// so the longest sequence bounds the launch while `num_tokens` sets the
/// aggregate work. Chunk counts are no longer derived here -- `FlashInfer` owns
/// its own chunk width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GdnPrefillGeometry {
    num_tokens: u32,
    max_sequence_length: u32,
}

fn derive_geometry(sequence_lengths: &[u32]) -> Result<Option<GdnPrefillGeometry>, String> {
    if sequence_lengths.is_empty() {
        return Ok(None);
    }

    let mut num_tokens = 0_u32;
    let mut max_sequence_length = 0_u32;
    for (index, &length) in sequence_lengths.iter().enumerate() {
        if length == 0 {
            return Err(format!("sequence_lengths[{index}] must be positive"));
        }
        num_tokens = num_tokens
            .checked_add(length)
            .ok_or_else(|| "total token count overflows u32".to_string())?;
        max_sequence_length = max_sequence_length.max(length);
    }

    Ok(Some(GdnPrefillGeometry {
        num_tokens,
        max_sequence_length,
    }))
}

fn aggregate_causal_conv(
    sequence_lengths: &[u32],
    mut eval: impl FnMut(&GdnCausalConvPrefillKernelInput) -> LeafMetrics,
) -> LeafMetrics {
    let mut metrics = LeafMetrics::ZERO;
    for &sequence_length in sequence_lengths {
        metrics.add_fanin(eval(&GdnCausalConvPrefillKernelInput {
            batch_size: 1,
            sequence_length,
        }));
    }
    metrics
}

struct WorkInputs {
    post_conv: GdnPrefillPostConvKernelInput,
    chunk_delta_rule: GdnChunkDeltaRuleKernelInput,
}

fn work_inputs(geometry: Option<GdnPrefillGeometry>) -> WorkInputs {
    let geometry = geometry.unwrap_or(GdnPrefillGeometry {
        num_tokens: 0,
        max_sequence_length: 0,
    });
    WorkInputs {
        post_conv: GdnPrefillPostConvKernelInput {
            num_tokens: geometry.num_tokens,
        },
        chunk_delta_rule: GdnChunkDeltaRuleKernelInput {
            num_tokens: geometry.num_tokens,
            max_sequence_length: geometry.max_sequence_length,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::bridge::DType;
    use crate::timing::cache::interp::{CoverageFlags, Metrics4};
    use crate::timing::{CostTreeBuilder, PerfApiBridge, SlotInput};
    use std::cell::RefCell;

    const SLOT_SUFFIXES: [&str; 3] = ["causal_conv", "post_conv", "chunk_delta_rule"];

    fn config() -> GdnPrefillOpConfig {
        let gpu_name = "NVIDIA H200".to_string();
        GdnPrefillOpConfig {
            causal_conv: GdnCausalConvPrefillKernelConfig {
                backends: vec!["vllm_triton"],
                gpu_name: gpu_name.clone(),
                channels: 8192.into(),
                kernel_size: 4.into(),
                dtype: DType::Bf16,
                state_dtype: DType::Bf16,
            },
            post_conv: GdnPrefillPostConvKernelConfig {
                backends: vec!["vllm_triton"],
                gpu_name: gpu_name.clone(),
                num_qk_heads: 16.into(),
                num_value_heads: 32.into(),
                key_head_dim: 128.into(),
                value_head_dim: 128.into(),
                dtype: DType::Bf16,
            },
            chunk_delta_rule: GdnChunkDeltaRuleKernelConfig {
                backends: vec!["flashinfer"],
                gpu_name,
                num_key_heads: 16.into(),
                num_heads: 32.into(),
                key_head_dim: 128.into(),
                value_head_dim: 128.into(),
                dtype: DType::Bf16,
            },
        }
    }

    fn enumerate_op() -> GdnPrefillOp {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        GdnPrefillOp::build("model.gdn.prefill".to_string(), config(), &bridge).unwrap()
    }

    #[derive(Default)]
    struct FakeEval {
        calls: RefCell<Vec<&'static str>>,
    }

    impl FakeEval {
        fn metric(&self, name: &'static str, time_ms: f32) -> LeafMetrics {
            self.calls.borrow_mut().push(name);
            LeafMetrics {
                m: Metrics4 {
                    time_ms,
                    flops: 1.0,
                    bytes: 2.0,
                    energy_j: 3.0,
                },
                coverage: CoverageFlags::EMPTY,
                backend_index: 0,
            }
        }
    }

    impl GdnPrefillEval for FakeEval {
        fn causal_conv(&self, input: &GdnCausalConvPrefillKernelInput) -> LeafMetrics {
            #[allow(
                clippy::cast_precision_loss,
                reason = "sequence_length is a test sequence length (hundreds of tokens), far under f32's 24-bit exact integer range"
            )]
            self.metric("causal_conv", input.sequence_length as f32)
        }

        fn post_conv(&self, _input: &GdnPrefillPostConvKernelInput) -> LeafMetrics {
            self.metric("post_conv", 2.0)
        }

        fn chunk_delta_rule(&self, _input: &GdnChunkDeltaRuleKernelInput) -> LeafMetrics {
            self.metric("chunk_delta_rule", 3.0)
        }
    }

    #[test]
    fn geometry_reduces_to_token_total_and_longest_sequence() {
        assert_eq!(derive_geometry(&[]).unwrap(), None);
        assert_eq!(
            derive_geometry(&[128]).unwrap(),
            Some(GdnPrefillGeometry {
                num_tokens: 128,
                max_sequence_length: 128,
            })
        );
        // The longest sequence is the delta rule's critical path, so a ragged
        // batch must not be averaged into an equivalent balanced one.
        assert_eq!(
            derive_geometry(&[3, 65, 2]).unwrap(),
            Some(GdnPrefillGeometry {
                num_tokens: 70,
                max_sequence_length: 65,
            })
        );
        assert_eq!(
            derive_geometry(&[8163, 22]).unwrap(),
            Some(GdnPrefillGeometry {
                num_tokens: 8185,
                max_sequence_length: 8163,
            })
        );
    }

    #[test]
    fn geometry_rejects_zero_lengths_and_checked_overflow() {
        assert!(derive_geometry(&[1, 0, 2])
            .unwrap_err()
            .contains("must be positive"));
        assert!(derive_geometry(&[u32::MAX, 1])
            .unwrap_err()
            .contains("overflows u32"));
    }

    #[test]
    fn compile_has_exact_fixed_slot_order_kinds_and_configs() {
        let op = enumerate_op();
        let mut builder = CostTreeBuilder::new();
        let root = op.compile(&mut builder);
        let tree = builder.finish(root);
        assert_eq!(tree.n_slots(), 3);
        assert_eq!(
            tree.slots
                .iter()
                .map(|slot| slot.name.as_str())
                .collect::<Vec<_>>(),
            SLOT_SUFFIXES
                .iter()
                .map(|suffix| format!("model.gdn.prefill.{suffix}"))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            tree.slots
                .iter()
                .map(|slot| slot.kind.as_str())
                .collect::<Vec<_>>(),
            [
                "gdn_causal_conv_prefill",
                "gdn_prefill_post_conv",
                "gdn_chunk_delta_rule",
            ]
        );
        assert_eq!(
            tree.slots[0].kernel_config,
            op.causal_conv.describe_config()
        );
        assert_eq!(
            tree.slots[2].kernel_config,
            op.chunk_delta_rule.describe_config()
        );
    }

    #[test]
    fn empty_eval_pushes_three_zero_slots_and_exact_inputs() {
        let op = enumerate_op();
        let mut buf = vec![LeafMetrics::MISS; 3];
        let mut inputs = Vec::new();
        let mut evaluator = Evaluator::with_inputs(&mut buf, &mut inputs);
        op.eval(&GdnPrefillOpInput::default(), &mut evaluator);
        assert_eq!(evaluator.filled(), 3);
        assert!(buf.iter().all(|metrics| metrics.m.time_ms == 0.0));
        assert!(buf
            .iter()
            .all(|metrics| metrics.coverage == CoverageFlags::EMPTY));
        assert_eq!(
            serde_json::to_value(&inputs).unwrap(),
            serde_json::json!([
                {"sequence_lengths": []},
                {"num_tokens": 0},
                {"num_tokens": 0, "max_sequence_length": 0},
            ])
        );
    }

    #[test]
    fn nonempty_eval_preserves_fixed_slot_order_for_single_and_ragged() {
        for (lengths, num_tokens, max_sequence_length) in
            [(vec![128], 128_u32, 128_u32), (vec![3, 65, 2], 70, 65)]
        {
            let fake = FakeEval::default();
            let mut buf = vec![LeafMetrics::ZERO; 3];
            let mut inputs = Vec::new();
            let mut evaluator = Evaluator::with_inputs(&mut buf, &mut inputs);
            eval_with(
                &fake,
                &GdnPrefillOpInput {
                    sequence_lengths: lengths.clone(),
                },
                &mut evaluator,
            );

            assert_eq!(evaluator.filled(), 3);
            #[allow(
                clippy::cast_precision_loss,
                reason = "num_tokens is a test token count (at most 128), far under f32's 24-bit exact integer range"
            )]
            let expected_time_ms = num_tokens as f32;
            assert_eq!(buf[0].m.time_ms, expected_time_ms);
            assert_eq!(
                &buf[1..].iter().map(|m| m.m.time_ms).collect::<Vec<_>>(),
                &[2.0, 3.0]
            );
            let mut expected_calls = vec!["causal_conv"; lengths.len()];
            expected_calls.extend_from_slice(&SLOT_SUFFIXES[1..]);
            assert_eq!(*fake.calls.borrow(), expected_calls);

            let json = serde_json::to_value(&inputs).unwrap();
            assert_eq!(json[0], serde_json::json!({"sequence_lengths": lengths}));
            assert_eq!(json[1], serde_json::json!({"num_tokens": num_tokens}));
            assert_eq!(
                json[2],
                serde_json::json!({
                    "num_tokens": num_tokens,
                    "max_sequence_length": max_sequence_length,
                })
            );
        }
    }

    #[test]
    fn causal_conv_fanin_is_b1_per_sequence_and_adopts_backend() {
        let mut seen = Vec::new();
        let metrics = aggregate_causal_conv(&[3, 65, 2], |shape| {
            seen.push((shape.batch_size, shape.sequence_length));
            #[allow(
                clippy::cast_precision_loss,
                reason = "sequence_length is a test sequence length (at most 65), far under f32's 24-bit exact integer range"
            )]
            let time_ms = shape.sequence_length as f32;
            LeafMetrics {
                m: Metrics4 {
                    time_ms,
                    flops: 1.0,
                    bytes: 2.0,
                    energy_j: 3.0,
                },
                coverage: CoverageFlags::EMPTY,
                backend_index: 2,
            }
        });
        assert_eq!(seen, [(1, 3), (1, 65), (1, 2)]);
        assert_eq!(metrics.m.time_ms, 70.0);
        assert_eq!(metrics.m.flops, 3.0);
        assert_eq!(metrics.backend_index, 2);
    }

    #[test]
    fn fanin_log_serializes_the_exact_sequence_vector() {
        let slot: SlotInput = GdnCausalConvPrefillLog {
            sequence_lengths: vec![3, 65, 2],
        }
        .into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"sequence_lengths":[3,65,2]})
        );
    }

    #[test]
    fn request_count_never_changes_compile_leaf_count() {
        let op = enumerate_op();
        for _input in [
            GdnPrefillOpInput::default(),
            GdnPrefillOpInput {
                sequence_lengths: vec![128],
            },
            GdnPrefillOpInput {
                sequence_lengths: vec![3, 65, 2],
            },
        ] {
            let mut builder = CostTreeBuilder::new();
            let root = op.compile(&mut builder);
            assert_eq!(builder.finish(root).n_slots(), 3);
        }
    }

    #[test]
    fn the_delta_rule_leaf_is_the_only_non_triton_backend_role() {
        let cfg = config();
        assert_eq!(cfg.causal_conv.backends, vec!["vllm_triton"]);
        assert_eq!(cfg.post_conv.backends, vec!["vllm_triton"]);
        assert_eq!(cfg.chunk_delta_rule.backends, vec!["flashinfer"]);
    }
}
