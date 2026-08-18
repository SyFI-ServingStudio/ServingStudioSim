//! Compound Gated DeltaNet prefill operation.
//!
//! One logical prefill owns the causal-convolution fan-in and the seven
//! downstream chunk-rule launches. Request-local convolution costs aggregate
//! into one fixed leaf; all other launches consume the exact aggregate
//! `(T, C, N, M)` geometry. This keeps request count out of the CostTree shape.

use std::sync::Arc;

use crate::timing::kernels::{
    GdnCausalConvPrefillKernel, GdnCausalConvPrefillKernelConfig,
    GdnCausalConvPrefillKernelInput, GdnChunkLocalCumsumKernel,
    GdnChunkLocalCumsumKernelConfig, GdnChunkLocalCumsumKernelInput, GdnChunkOutputKernel,
    GdnChunkOutputKernelConfig, GdnChunkOutputKernelInput, GdnChunkRecomputeWUKernel,
    GdnChunkRecomputeWUKernelConfig, GdnChunkRecomputeWUKernelInput,
    GdnChunkScaledDotKktKernel, GdnChunkScaledDotKktKernelConfig,
    GdnChunkScaledDotKktKernelInput, GdnChunkSolveTrilKernel,
    GdnChunkSolveTrilKernelConfig, GdnChunkSolveTrilKernelInput, GdnChunkStateUpdateKernel,
    GdnChunkStateUpdateKernelConfig, GdnChunkStateUpdateKernelInput, GdnPrefillPostConvKernel,
    GdnPrefillPostConvKernelConfig, GdnPrefillPostConvKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Evaluator, GdnCausalConvPrefillLog, LeafMetrics,
    PerfApiBridge, Probe,
};

const CHUNK_SIZE: u64 = 64;

/// Already-resolved configs for the eight production launches, in execution
/// order. Backend choice remains owned by each kernel config.
#[derive(Clone, Debug)]
pub struct GdnPrefillOpConfig {
    pub causal_conv: GdnCausalConvPrefillKernelConfig,
    pub post_conv: GdnPrefillPostConvKernelConfig,
    pub cumsum: GdnChunkLocalCumsumKernelConfig,
    pub kkt: GdnChunkScaledDotKktKernelConfig,
    pub solve: GdnChunkSolveTrilKernelConfig,
    pub recompute_w_u: GdnChunkRecomputeWUKernelConfig,
    pub state_update: GdnChunkStateUpdateKernelConfig,
    pub output: GdnChunkOutputKernelConfig,
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
    pub cumsum: Arc<GdnChunkLocalCumsumKernel>,
    pub kkt: Arc<GdnChunkScaledDotKktKernel>,
    pub solve: Arc<GdnChunkSolveTrilKernel>,
    pub recompute_w_u: Arc<GdnChunkRecomputeWUKernel>,
    pub state_update: Arc<GdnChunkStateUpdateKernel>,
    pub output: Arc<GdnChunkOutputKernel>,
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
        let cumsum = Arc::new(GdnChunkLocalCumsumKernel::build(
            format!("{name}.cumsum"),
            config.cumsum,
            bridge,
        )?);
        let kkt = Arc::new(GdnChunkScaledDotKktKernel::build(
            format!("{name}.kkt"),
            config.kkt,
            bridge,
        )?);
        let solve = Arc::new(GdnChunkSolveTrilKernel::build(
            format!("{name}.solve"),
            config.solve,
            bridge,
        )?);
        let recompute_w_u = Arc::new(GdnChunkRecomputeWUKernel::build(
            format!("{name}.recompute_w_u"),
            config.recompute_w_u,
            bridge,
        )?);
        let state_update = Arc::new(GdnChunkStateUpdateKernel::build(
            format!("{name}.state_update"),
            config.state_update,
            bridge,
        )?);
        let output = Arc::new(GdnChunkOutputKernel::build(
            format!("{name}.output"),
            config.output,
            bridge,
        )?);
        Ok(Self {
            name,
            causal_conv,
            post_conv,
            cumsum,
            kkt,
            solve,
            recompute_w_u,
            state_update,
            output,
        })
    }

    /// Eight fixed leaves, independent of the number of prefill sequences.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            leaf(builder, &self.name, "causal_conv", &*self.causal_conv),
            leaf(builder, &self.name, "post_conv", &*self.post_conv),
            leaf(builder, &self.name, "cumsum", &*self.cumsum),
            leaf(builder, &self.name, "kkt", &*self.kkt),
            leaf(builder, &self.name, "solve", &*self.solve),
            leaf(
                builder,
                &self.name,
                "recompute_w_u",
                &*self.recompute_w_u,
            ),
            leaf(
                builder,
                &self.name,
                "state_update",
                &*self.state_update,
            ),
            leaf(builder, &self.name, "output", &*self.output),
        ])
    }

    /// Fill the eight compile slots in identical order. Empty input is valid
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
    fn cumsum(&self, input: &GdnChunkLocalCumsumKernelInput) -> LeafMetrics;
    fn kkt(&self, input: &GdnChunkScaledDotKktKernelInput) -> LeafMetrics;
    fn solve(&self, input: &GdnChunkSolveTrilKernelInput) -> LeafMetrics;
    fn recompute_w_u(&self, input: &GdnChunkRecomputeWUKernelInput) -> LeafMetrics;
    fn state_update(&self, input: &GdnChunkStateUpdateKernelInput) -> LeafMetrics;
    fn output(&self, input: &GdnChunkOutputKernelInput) -> LeafMetrics;
}

impl GdnPrefillEval for GdnPrefillOp {
    fn causal_conv(&self, input: &GdnCausalConvPrefillKernelInput) -> LeafMetrics {
        self.causal_conv.eval(input)
    }

    fn post_conv(&self, input: &GdnPrefillPostConvKernelInput) -> LeafMetrics {
        self.post_conv.eval(input)
    }

    fn cumsum(&self, input: &GdnChunkLocalCumsumKernelInput) -> LeafMetrics {
        self.cumsum.eval(input)
    }

    fn kkt(&self, input: &GdnChunkScaledDotKktKernelInput) -> LeafMetrics {
        self.kkt.eval(input)
    }

    fn solve(&self, input: &GdnChunkSolveTrilKernelInput) -> LeafMetrics {
        self.solve.eval(input)
    }

    fn recompute_w_u(&self, input: &GdnChunkRecomputeWUKernelInput) -> LeafMetrics {
        self.recompute_w_u.eval(input)
    }

    fn state_update(&self, input: &GdnChunkStateUpdateKernelInput) -> LeafMetrics {
        self.state_update.eval(input)
    }

    fn output(&self, input: &GdnChunkOutputKernelInput) -> LeafMetrics {
        self.output.eval(input)
    }
}

fn eval_with(
    kernels: &impl GdnPrefillEval,
    input: &GdnPrefillOpInput,
    ev: &mut Evaluator,
) {
    let geometry = derive_geometry(&input.sequence_lengths)
        .unwrap_or_else(|reason| panic!("invalid GdnPrefillOpInput: {reason}"));

    let mut causal_conv_metrics = LeafMetrics::ZERO;
    if geometry.is_some() {
        causal_conv_metrics = aggregate_causal_conv(&input.sequence_lengths, |shape| {
            kernels.causal_conv(shape)
        });
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
        kernels.cumsum(&work.cumsum)
    };
    ev.push(metrics, || work.cumsum.into());
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        kernels.kkt(&work.kkt)
    };
    ev.push(metrics, || work.kkt.into());
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        kernels.solve(&work.solve)
    };
    ev.push(metrics, || work.solve.into());
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        kernels.recompute_w_u(&work.recompute_w_u)
    };
    ev.push(metrics, || work.recompute_w_u.into());
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        kernels.state_update(&work.state_update)
    };
    ev.push(metrics, || work.state_update.into());
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        kernels.output(&work.output)
    };
    ev.push(metrics, || work.output.into());
}

fn leaf<K: Probe>(
    builder: &mut CostTreeBuilder,
    name: &str,
    suffix: &str,
    kernel: &K,
) -> CostNode {
    builder.leaf(
        format!("{name}.{suffix}"),
        kernel.kind(),
        kernel.describe_config(),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GdnPrefillGeometry {
    num_tokens: u32,
    num_chunks: u32,
    num_sequences: u32,
    max_chunks_per_sequence: u32,
}

fn derive_geometry(sequence_lengths: &[u32]) -> Result<Option<GdnPrefillGeometry>, String> {
    if sequence_lengths.is_empty() {
        return Ok(None);
    }

    let num_sequences = u32::try_from(sequence_lengths.len())
        .map_err(|_| "sequence count exceeds u32".to_string())?;
    let mut num_tokens = 0_u32;
    let mut num_chunks = 0_u32;
    let mut max_chunks_per_sequence = 0_u32;
    for (index, &length) in sequence_lengths.iter().enumerate() {
        if length == 0 {
            return Err(format!("sequence_lengths[{index}] must be positive"));
        }
        num_tokens = num_tokens
            .checked_add(length)
            .ok_or_else(|| "total token count overflows u32".to_string())?;
        let chunks = u32::try_from((u64::from(length) + CHUNK_SIZE - 1) / CHUNK_SIZE)
            .expect("u32 sequence length chunk count fits u32");
        num_chunks = num_chunks
            .checked_add(chunks)
            .ok_or_else(|| "total chunk count overflows u32".to_string())?;
        max_chunks_per_sequence = max_chunks_per_sequence.max(chunks);
    }

    Ok(Some(GdnPrefillGeometry {
        num_tokens,
        num_chunks,
        num_sequences,
        max_chunks_per_sequence,
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
    cumsum: GdnChunkLocalCumsumKernelInput,
    kkt: GdnChunkScaledDotKktKernelInput,
    solve: GdnChunkSolveTrilKernelInput,
    recompute_w_u: GdnChunkRecomputeWUKernelInput,
    state_update: GdnChunkStateUpdateKernelInput,
    output: GdnChunkOutputKernelInput,
}

fn work_inputs(geometry: Option<GdnPrefillGeometry>) -> WorkInputs {
    let geometry = geometry.unwrap_or(GdnPrefillGeometry {
        num_tokens: 0,
        num_chunks: 0,
        num_sequences: 0,
        max_chunks_per_sequence: 0,
    });
    WorkInputs {
        post_conv: GdnPrefillPostConvKernelInput {
            num_tokens: geometry.num_tokens,
        },
        cumsum: GdnChunkLocalCumsumKernelInput {
            num_tokens: geometry.num_tokens,
            num_chunks: geometry.num_chunks,
        },
        kkt: GdnChunkScaledDotKktKernelInput {
            num_tokens: geometry.num_tokens,
            num_chunks: geometry.num_chunks,
        },
        solve: GdnChunkSolveTrilKernelInput {
            num_tokens: geometry.num_tokens,
            num_chunks: geometry.num_chunks,
        },
        recompute_w_u: GdnChunkRecomputeWUKernelInput {
            num_tokens: geometry.num_tokens,
            num_chunks: geometry.num_chunks,
        },
        state_update: GdnChunkStateUpdateKernelInput {
            num_tokens: geometry.num_tokens,
            num_chunks: geometry.num_chunks,
            num_sequences: geometry.num_sequences,
            max_chunks_per_sequence: geometry.max_chunks_per_sequence,
        },
        output: GdnChunkOutputKernelInput {
            num_tokens: geometry.num_tokens,
            num_chunks: geometry.num_chunks,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::cache::interp::{CoverageFlags, Metrics4};
    use crate::timing::bridge::DType;
    use crate::timing::{CostTreeBuilder, Dim, PerfApiBridge, SlotInput};
    use std::cell::RefCell;

    const SLOT_SUFFIXES: [&str; 8] = [
        "causal_conv",
        "post_conv",
        "cumsum",
        "kkt",
        "solve",
        "recompute_w_u",
        "state_update",
        "output",
    ];

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
            cumsum: GdnChunkLocalCumsumKernelConfig {
                backends: vec!["vllm_triton"],
                gpu_name: gpu_name.clone(),
                num_heads: 32.into(),
                dtype: DType::Fp32,
            },
            kkt: GdnChunkScaledDotKktKernelConfig {
                backends: vec!["vllm_triton"],
                gpu_name: gpu_name.clone(),
                num_key_heads: 16.into(),
                num_heads: 32.into(),
                key_head_dim: 128.into(),
                dtype: DType::Bf16,
            },
            solve: GdnChunkSolveTrilKernelConfig {
                backends: vec!["vllm_triton"],
                gpu_name: gpu_name.clone(),
                max_chunk_tokens: 64.into(),
                num_heads: 32.into(),
                dtype: DType::Bf16,
            },
            recompute_w_u: GdnChunkRecomputeWUKernelConfig {
                backends: vec!["vllm_triton"],
                gpu_name: gpu_name.clone(),
                num_key_heads: 16.into(),
                num_heads: 32.into(),
                key_head_dim: 128.into(),
                value_head_dim: 128.into(),
                dtype: DType::Bf16,
            },
            state_update: GdnChunkStateUpdateKernelConfig {
                backends: vec!["vllm_triton"],
                gpu_name: gpu_name.clone(),
                num_key_heads: 16.into(),
                num_heads: 32.into(),
                key_head_dim: 128.into(),
                value_head_dim: 128.into(),
                dtype: DType::Bf16,
            },
            output: GdnChunkOutputKernelConfig {
                backends: vec!["vllm_triton"],
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
            self.metric("causal_conv", input.sequence_length as f32)
        }

        fn post_conv(&self, _input: &GdnPrefillPostConvKernelInput) -> LeafMetrics {
            self.metric("post_conv", 2.0)
        }

        fn cumsum(&self, _input: &GdnChunkLocalCumsumKernelInput) -> LeafMetrics {
            self.metric("cumsum", 3.0)
        }

        fn kkt(&self, _input: &GdnChunkScaledDotKktKernelInput) -> LeafMetrics {
            self.metric("kkt", 4.0)
        }

        fn solve(&self, _input: &GdnChunkSolveTrilKernelInput) -> LeafMetrics {
            self.metric("solve", 5.0)
        }

        fn recompute_w_u(&self, _input: &GdnChunkRecomputeWUKernelInput) -> LeafMetrics {
            self.metric("recompute_w_u", 6.0)
        }

        fn state_update(&self, _input: &GdnChunkStateUpdateKernelInput) -> LeafMetrics {
            self.metric("state_update", 7.0)
        }

        fn output(&self, _input: &GdnChunkOutputKernelInput) -> LeafMetrics {
            self.metric("output", 8.0)
        }
    }

    #[test]
    fn geometry_handles_empty_single_and_ragged() {
        assert_eq!(derive_geometry(&[]).unwrap(), None);
        assert_eq!(
            derive_geometry(&[128]).unwrap(),
            Some(GdnPrefillGeometry {
                num_tokens: 128,
                num_chunks: 2,
                num_sequences: 1,
                max_chunks_per_sequence: 2,
            })
        );
        assert_eq!(
            derive_geometry(&[3, 65, 2]).unwrap(),
            Some(GdnPrefillGeometry {
                num_tokens: 70,
                num_chunks: 4,
                num_sequences: 3,
                max_chunks_per_sequence: 2,
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
        assert_eq!(tree.n_slots(), 8);
        assert_eq!(
            tree.slots.iter().map(|slot| slot.name.as_str()).collect::<Vec<_>>(),
            SLOT_SUFFIXES
                .iter()
                .map(|suffix| format!("model.gdn.prefill.{suffix}"))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            tree.slots.iter().map(|slot| slot.kind.as_str()).collect::<Vec<_>>(),
            [
                "gdn_causal_conv_prefill",
                "gdn_prefill_post_conv",
                "gdn_chunk_local_cumsum",
                "gdn_chunk_scaled_dot_kkt",
                "gdn_chunk_solve_tril",
                "gdn_chunk_recompute_w_u",
                "gdn_chunk_state_update",
                "gdn_chunk_output",
            ]
        );
        assert_eq!(tree.slots[0].kernel_config, op.causal_conv.describe_config());
        assert_eq!(tree.slots[7].kernel_config, op.output.describe_config());
    }

    #[test]
    fn empty_eval_pushes_eight_zero_slots_and_exact_inputs() {
        let op = enumerate_op();
        let mut buf = vec![LeafMetrics::MISS; 8];
        let mut inputs = Vec::new();
        let mut evaluator = Evaluator::with_inputs(&mut buf, &mut inputs);
        op.eval(&GdnPrefillOpInput::default(), &mut evaluator);
        assert_eq!(evaluator.filled(), 8);
        assert!(buf.iter().all(|metrics| metrics.m.time_ms == 0.0));
        assert!(buf
            .iter()
            .all(|metrics| metrics.coverage == CoverageFlags::EMPTY));
        assert_eq!(inputs.len(), 8);
        assert_eq!(
            serde_json::to_value(&inputs).unwrap(),
            serde_json::json!([
                {"sequence_lengths": []},
                {"num_tokens": 0},
                {"num_tokens": 0, "num_chunks": 0},
                {"num_tokens": 0, "num_chunks": 0},
                {"num_tokens": 0, "num_chunks": 0},
                {"num_tokens": 0, "num_chunks": 0},
                {"num_tokens": 0, "num_chunks": 0, "num_sequences": 0,
                 "max_chunks_per_sequence": 0},
                {"num_tokens": 0, "num_chunks": 0},
            ])
        );
    }

    #[test]
    fn nonempty_eval_preserves_fixed_slot_order_for_single_and_ragged() {
        for (lengths, expected) in [
            (vec![128], (128, 2, 1, 2)),
            (vec![3, 65, 2], (70, 4, 3, 2)),
        ] {
            let fake = FakeEval::default();
            let mut buf = vec![LeafMetrics::ZERO; 8];
            let mut inputs = Vec::new();
            let mut evaluator = Evaluator::with_inputs(&mut buf, &mut inputs);
            eval_with(
                &fake,
                &GdnPrefillOpInput {
                    sequence_lengths: lengths.clone(),
                },
                &mut evaluator,
            );

            assert_eq!(evaluator.filled(), 8);
            assert_eq!(buf[0].m.time_ms, expected.0 as f32);
            assert_eq!(
                &buf[1..].iter().map(|m| m.m.time_ms).collect::<Vec<_>>(),
                &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            );
            let mut expected_calls = vec!["causal_conv"; lengths.len()];
            expected_calls.extend_from_slice(&SLOT_SUFFIXES[1..]);
            assert_eq!(*fake.calls.borrow(), expected_calls);
            assert_eq!(inputs.len(), 8);

            let json = serde_json::to_value(&inputs).unwrap();
            assert_eq!(json[0], serde_json::json!({"sequence_lengths": lengths}));
            assert_eq!(json[1], serde_json::json!({"num_tokens": expected.0}));
            for index in [2, 3, 4, 5, 7] {
                assert_eq!(
                    json[index],
                    serde_json::json!({
                        "num_tokens": expected.0,
                        "num_chunks": expected.1,
                    })
                );
            }
            assert_eq!(
                json[6],
                serde_json::json!({
                    "num_tokens": expected.0,
                    "num_chunks": expected.1,
                    "num_sequences": expected.2,
                    "max_chunks_per_sequence": expected.3,
                })
            );
        }
    }

    #[test]
    fn work_inputs_map_single_and_ragged_geometry_exactly() {
        for (lengths, expected) in [
            (&[128][..], (128, 2, 1, 2)),
            (&[3, 65, 2][..], (70, 4, 3, 2)),
        ] {
            let geometry = derive_geometry(lengths).unwrap().unwrap();
            let work = work_inputs(Some(geometry));
            assert_eq!(work.post_conv.num_tokens, expected.0);
            assert_eq!(
                (work.cumsum.num_tokens, work.cumsum.num_chunks),
                (expected.0, expected.1)
            );
            assert_eq!(
                (work.kkt.num_tokens, work.kkt.num_chunks),
                (expected.0, expected.1)
            );
            assert_eq!(
                (work.solve.num_tokens, work.solve.num_chunks),
                (expected.0, expected.1)
            );
            assert_eq!(
                (
                    work.recompute_w_u.num_tokens,
                    work.recompute_w_u.num_chunks,
                ),
                (expected.0, expected.1)
            );
            assert_eq!(
                (
                    work.state_update.num_tokens,
                    work.state_update.num_chunks,
                    work.state_update.num_sequences,
                    work.state_update.max_chunks_per_sequence,
                ),
                expected
            );
            assert_eq!(
                (work.output.num_tokens, work.output.num_chunks),
                (expected.0, expected.1)
            );
        }
    }

    #[test]
    fn causal_conv_fanin_is_b1_per_sequence_and_adopts_backend() {
        let mut seen = Vec::new();
        let metrics = aggregate_causal_conv(&[3, 65, 2], |shape| {
            seen.push((shape.batch_size, shape.sequence_length));
            LeafMetrics {
                m: Metrics4 {
                    time_ms: shape.sequence_length as f32,
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
            assert_eq!(builder.finish(root).n_slots(), 8);
        }
    }

    #[test]
    fn config_keeps_independent_backend_roles() {
        let cfg = config();
        assert_eq!(cfg.causal_conv.backends, vec!["vllm_triton"]);
        assert_eq!(cfg.post_conv.backends, vec!["vllm_triton"]);
        assert_eq!(cfg.cumsum.backends, vec!["vllm_triton"]);
        assert_eq!(cfg.output.backends, vec!["vllm_triton"]);
        assert_eq!(cfg.solve.max_chunk_tokens, Dim::from(64));
    }
}
