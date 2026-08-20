//! Compound Gated DeltaNet decode operation.
//!
//! Causal convolution updates each request's convolution state before the
//! recurrent DeltaNet launch consumes and updates its recurrent state. The
//! shared request-local state makes this a strict, zero-overlap sum boundary.

use std::sync::Arc;

use crate::timing::kernels::{
    GdnCausalConvDecodeKernel, GdnCausalConvDecodeKernelConfig, GdnCausalConvDecodeKernelInput,
    GdnRecurrentDecodeKernel, GdnRecurrentDecodeKernelConfig, GdnRecurrentDecodeKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Evaluator, LeafMetrics, PerfApiBridge, Probe,
};

/// Already-resolved configs for the two production launches, in execution
/// order. Backend selection remains owned by each kernel config.
#[derive(Clone, Debug)]
pub struct GdnDecodeOpConfig {
    pub causal_conv: GdnCausalConvDecodeKernelConfig,
    pub recurrent: GdnRecurrentDecodeKernelConfig,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GdnDecodeOpInput {
    pub batch_size: u32,
}

pub struct GdnDecodeOp {
    pub name: String,
    pub causal_conv: Arc<GdnCausalConvDecodeKernel>,
    pub recurrent: Arc<GdnRecurrentDecodeKernel>,
}

impl GdnDecodeOp {
    pub fn build(
        name: String,
        config: GdnDecodeOpConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let causal_conv = Arc::new(GdnCausalConvDecodeKernel::build(
            format!("{name}.causal_conv"),
            config.causal_conv,
            bridge,
        )?);
        let recurrent = Arc::new(GdnRecurrentDecodeKernel::build(
            format!("{name}.recurrent"),
            config.recurrent,
            bridge,
        )?);
        Ok(Self {
            name,
            causal_conv,
            recurrent,
        })
    }

    /// Two fixed sequential leaves, independent of decode batch size.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            builder.leaf(
                format!("{}.causal_conv", self.name),
                self.causal_conv.kind(),
                self.causal_conv.describe_config(),
            ),
            builder.leaf(
                format!("{}.recurrent", self.name),
                self.recurrent.kind(),
                self.recurrent.describe_config(),
            ),
        ])
    }

    /// Fill the two compile slots in identical order. A zero batch is valid
    /// zero-work and deliberately bypasses both L1 cache lookups while logging
    /// the faithful typed zero inputs.
    pub fn eval(&self, input: &GdnDecodeOpInput, ev: &mut Evaluator) {
        eval_with(self, input, ev);
    }
}

/// Private test seam: production delegates to cached kernels; focused tests
/// substitute deterministic probes without requiring a bridge or profile DB.
trait GdnDecodeEval {
    fn causal_conv(&self, input: &GdnCausalConvDecodeKernelInput) -> LeafMetrics;
    fn recurrent(&self, input: &GdnRecurrentDecodeKernelInput) -> LeafMetrics;
}

impl GdnDecodeEval for GdnDecodeOp {
    fn causal_conv(&self, input: &GdnCausalConvDecodeKernelInput) -> LeafMetrics {
        self.causal_conv.eval(input)
    }

    fn recurrent(&self, input: &GdnRecurrentDecodeKernelInput) -> LeafMetrics {
        self.recurrent.eval(input)
    }
}

fn eval_with(kernels: &impl GdnDecodeEval, input: &GdnDecodeOpInput, ev: &mut Evaluator) {
    let causal_conv = GdnCausalConvDecodeKernelInput {
        batch_size: input.batch_size,
    };
    let metrics = if input.batch_size == 0 {
        LeafMetrics::ZERO
    } else {
        kernels.causal_conv(&causal_conv)
    };
    ev.push(metrics, || causal_conv.into());

    let recurrent = GdnRecurrentDecodeKernelInput {
        batch_size: input.batch_size,
    };
    let metrics = if input.batch_size == 0 {
        LeafMetrics::ZERO
    } else {
        kernels.recurrent(&recurrent)
    };
    ev.push(metrics, || recurrent.into());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    use crate::timing::bridge::DType;
    use crate::timing::cache::interp::{CoverageFlags, Metrics4};
    use crate::timing::{CostTreeBuilder, PerfApiBridge};

    fn config() -> GdnDecodeOpConfig {
        GdnDecodeOpConfig {
            causal_conv: GdnCausalConvDecodeKernelConfig {
                backends: vec!["torch", "vllm_triton"],
                gpu_name: "NVIDIA H200".to_string(),
                channels: 8192.into(),
                kernel_size: 4.into(),
                dtype: DType::Bf16,
                state_dtype: DType::Bf16,
            },
            recurrent: GdnRecurrentDecodeKernelConfig {
                backends: vec!["vllm_triton"],
                gpu_name: "NVIDIA H200".to_string(),
                num_qk_heads: 16.into(),
                num_value_heads: 32.into(),
                key_head_dim: 128.into(),
                value_head_dim: 128.into(),
                dtype: DType::Bf16,
                state_dtype: DType::Fp32,
            },
        }
    }

    fn enumerate_op() -> GdnDecodeOp {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        GdnDecodeOp::build("model.gdn.decode".to_string(), config(), &bridge).unwrap()
    }

    #[derive(Default)]
    struct FakeEval {
        calls: RefCell<Vec<(&'static str, u32)>>,
    }

    impl FakeEval {
        fn metric(&self, name: &'static str, batch_size: u32, time_ms: f32) -> LeafMetrics {
            self.calls.borrow_mut().push((name, batch_size));
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

    impl GdnDecodeEval for FakeEval {
        fn causal_conv(&self, input: &GdnCausalConvDecodeKernelInput) -> LeafMetrics {
            self.metric("causal_conv", input.batch_size, 1.0)
        }

        fn recurrent(&self, input: &GdnRecurrentDecodeKernelInput) -> LeafMetrics {
            self.metric("recurrent", input.batch_size, 2.0)
        }
    }

    #[test]
    fn config_and_input_fields_are_exact() {
        let cfg = config();
        assert_eq!(cfg.causal_conv.channels, 8192);
        assert_eq!(cfg.causal_conv.kernel_size, 4);
        assert_eq!(cfg.recurrent.num_qk_heads, 16);
        assert_eq!(cfg.recurrent.num_value_heads, 32);
        assert_eq!(cfg.recurrent.key_head_dim, 128);
        assert_eq!(cfg.recurrent.value_head_dim, 128);
        assert_eq!(GdnDecodeOpInput { batch_size: 7 }.batch_size, 7);
    }

    #[test]
    fn compile_has_exact_fixed_slot_order_kinds_and_configs() {
        let op = enumerate_op();
        let mut builder = CostTreeBuilder::new();
        let root = op.compile(&mut builder);
        let tree = builder.finish(root);

        assert_eq!(tree.n_slots(), 2);
        assert_eq!(tree.slots[0].name, "model.gdn.decode.causal_conv");
        assert_eq!(tree.slots[1].name, "model.gdn.decode.recurrent");
        assert_eq!(tree.slots[0].kind, "gdn_causal_conv_decode");
        assert_eq!(tree.slots[1].kind, "gdn_recurrent_decode");
        assert_eq!(
            tree.slots[0].kernel_config,
            op.causal_conv.describe_config()
        );
        assert_eq!(tree.slots[1].kernel_config, op.recurrent.describe_config());
    }

    #[test]
    fn positive_eval_maps_identical_batch_size_in_compile_order() {
        for batch_size in [1, 7] {
            let fake = FakeEval::default();
            let mut metrics = vec![LeafMetrics::MISS; 2];
            let mut inputs = Vec::new();
            let mut evaluator = Evaluator::with_inputs(&mut metrics, &mut inputs);

            eval_with(&fake, &GdnDecodeOpInput { batch_size }, &mut evaluator);

            assert_eq!(evaluator.filled(), 2);
            assert_eq!(metrics[0].m.time_ms, 1.0);
            assert_eq!(metrics[1].m.time_ms, 2.0);
            assert_eq!(
                *fake.calls.borrow(),
                [("causal_conv", batch_size), ("recurrent", batch_size)]
            );
            assert_eq!(
                serde_json::to_value(&inputs).unwrap(),
                serde_json::json!([
                    {"batch_size": batch_size},
                    {"batch_size": batch_size},
                ])
            );
        }
    }

    #[test]
    fn zero_batch_pushes_typed_zero_slots_without_kernel_evaluation() {
        let fake = FakeEval::default();
        let mut metrics = vec![LeafMetrics::MISS; 2];
        let mut inputs = Vec::new();
        let mut evaluator = Evaluator::with_inputs(&mut metrics, &mut inputs);

        eval_with(&fake, &GdnDecodeOpInput { batch_size: 0 }, &mut evaluator);

        assert_eq!(evaluator.filled(), 2);
        assert!(fake.calls.borrow().is_empty());
        assert!(metrics.iter().all(|metric| metric.m.time_ms == 0.0));
        assert!(metrics
            .iter()
            .all(|metric| metric.coverage == CoverageFlags::EMPTY));
        assert_eq!(
            serde_json::to_value(&inputs).unwrap(),
            serde_json::json!([{"batch_size": 0}, {"batch_size": 0}])
        );
    }

    #[test]
    fn compile_leaf_count_never_varies_with_batch_size() {
        let op = enumerate_op();
        for batch_size in [0, 1, 256] {
            let _input = GdnDecodeOpInput { batch_size };
            let mut builder = CostTreeBuilder::new();
            let root = op.compile(&mut builder);
            assert_eq!(builder.finish(root).n_slots(), 2);
        }
    }

    #[test]
    fn configured_backend_selection_is_preserved_per_leaf() {
        let op = enumerate_op();
        assert_eq!(op.causal_conv.config.backends, ["torch", "vllm_triton"]);
        assert_eq!(op.recurrent.config.backends, ["vllm_triton"]);
    }
}
