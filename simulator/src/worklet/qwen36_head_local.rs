//! Qwen3.6 TP1 local final decoder head.
//!
//! The final decoder layer leaves its MLP output and delayed residual separate.
//! This section owns their fused residual add plus final RMSNorm exactly once,
//! then applies the untied vocabulary head only to the rows selected for logits.
//! Intermediate delayed residuals instead belong to the following attention
//! worklet's entry norm. Token embedding remains a future L4 atomic leaf.
//!
//! The pinned Qwen3.6 FP8 checkpoint explicitly excludes `lm_head` from FP8:
//! it is one unquantized BF16 GEMM with no input-quantization child. TP1 needs no
//! vocabulary gather, TP/EP collective, network, or communication child.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ResidualRmsNormKernel, ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput,
    SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

const HIDDEN: u32 = 2048;
const VOCAB_SIZE: u32 = 248_320;
const VOCAB_ALIGNMENT: u32 = 64;

#[derive(Clone, Debug)]
pub struct Qwen36HeadLocalWorkletConfig {
    pub hidden: Dim,
    pub vocab_size: Dim,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub residual_rms_norm_backends: Vec<&'static str>,
    pub bf16_gemm_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Qwen36HeadLocalWorkletResolved {
    pub final_add_rms_norm: ResidualRmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub raw_cfg: Qwen36HeadLocalWorkletConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Qwen36HeadLocalWorkletInput {
    pub final_norm_tokens: u32,
    pub logits_tokens: u32,
}

pub struct Qwen36HeadLocalWorklet {
    pub name: String,
    pub final_add_rms_norm: Op<ResidualRmsNormKernel>,
    pub lm_head: Op<SingleGemmKernel>,
    resolved: Qwen36HeadLocalWorkletResolved,
}

impl Qwen36HeadLocalWorklet {
    pub fn resolve_config(cfg: &Qwen36HeadLocalWorkletConfig) -> Qwen36HeadLocalWorkletResolved {
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid Qwen36HeadLocalWorkletConfig: {reason}"));

        Qwen36HeadLocalWorkletResolved {
            final_add_rms_norm: ResidualRmsNormKernelConfig {
                backends: cfg.residual_rms_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.clone(),
                dtype: cfg.activation_dtype,
            },
            lm_head: SingleGemmKernelConfig {
                backends: cfg.bf16_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.vocab_size.clone(),
                k: cfg.hidden.clone(),
                dtype: cfg.activation_dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Qwen36HeadLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let final_add_rms_norm = build_atomic(
            &name,
            "final_add_rms_norm",
            resolved.final_add_rms_norm.clone(),
            ResidualRmsNormKernel::build,
            bridge,
        )?;
        let lm_head = build_atomic(
            &name,
            "lm_head",
            resolved.lm_head.clone(),
            SingleGemmKernel::build,
            bridge,
        )?;
        Ok(Self {
            name,
            final_add_rms_norm,
            lm_head,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (Qwen36HeadLocalWorklet) [local (1 GPU); hidden={}, vocab={}; BF16 untied head]",
                self.name, self.resolved.raw_cfg.hidden, self.resolved.raw_cfg.vocab_size,
            ),
            child: Box::new(CostNode::Sum(vec![
                self.final_add_rms_norm.compile(builder),
                self.lm_head.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Qwen36HeadLocalWorkletInput, ev: &mut Evaluator) {
        let work = derive_work(input)
            .unwrap_or_else(|reason| panic!("invalid Qwen36HeadLocalWorkletInput: {reason}"));
        eval_atomic_or_zero(
            &self.final_add_rms_norm,
            work.final_add_rms_norm,
            input.final_norm_tokens == 0,
            ev,
        );
        eval_atomic_or_zero(&self.lm_head, work.lm_head, input.logits_tokens == 0, ev);
    }
}

struct WorkInputs {
    final_add_rms_norm: ResidualRmsNormKernelInput,
    lm_head: SingleGemmKernelInput,
}

fn derive_work(input: &Qwen36HeadLocalWorkletInput) -> Result<WorkInputs, String> {
    if input.logits_tokens > input.final_norm_tokens {
        return Err(format!(
            "logits_tokens {} must not exceed final_norm_tokens {}",
            input.logits_tokens, input.final_norm_tokens
        ));
    }
    Ok(WorkInputs {
        final_add_rms_norm: ResidualRmsNormKernelInput {
            m: input.final_norm_tokens,
        },
        lm_head: SingleGemmKernelInput {
            m: input.logits_tokens,
        },
    })
}

fn validate_config(cfg: &Qwen36HeadLocalWorkletConfig) -> Result<(), String> {
    for (name, actual, expected) in [
        ("hidden", cfg.hidden.get(), HIDDEN),
        ("vocab_size", cfg.vocab_size.get(), VOCAB_SIZE),
    ] {
        if actual != expected {
            return Err(format!("{name} must be {expected}, got {actual}"));
        }
    }
    if cfg.activation_dtype != DType::Bf16 {
        return Err("activation_dtype must be BF16".into());
    }
    debug_assert_eq!(VOCAB_SIZE % VOCAB_ALIGNMENT, 0);
    Ok(())
}

fn build_atomic<K, C, F>(
    parent: &str,
    suffix: &str,
    config: C,
    build: F,
    bridge: &PerfApiBridge,
) -> Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError>,
{
    let name = format!("{parent}.{suffix}");
    Ok(Op::new(
        name.clone(),
        Arc::new(build(name, config, bridge)?),
    ))
}

fn eval_atomic_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, ev: &mut Evaluator)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::cache::interp::CoverageFlags;
    use crate::timing::{CostTreeBuilder, KernelConfig, PerfApiBridge};

    fn cfg() -> Qwen36HeadLocalWorkletConfig {
        Qwen36HeadLocalWorkletConfig {
            hidden: HIDDEN.into(),
            vocab_size: VOCAB_SIZE.into(),
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA H200".into(),
            residual_rms_norm_backends: vec!["vllm_cuda"],
            bf16_gemm_backends: vec!["torch_linear"],
        }
    }

    fn enumerate_worklet() -> Qwen36HeadLocalWorklet {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        Qwen36HeadLocalWorklet::build(
            "model.head".into(),
            Qwen36HeadLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap()
    }

    #[test]
    fn resolution_freezes_checkpoint_bf16_head_contract() {
        let resolved = Qwen36HeadLocalWorklet::resolve_config(&cfg());
        assert_eq!(resolved.final_add_rms_norm.hidden.get(), 2048);
        assert_eq!(resolved.final_add_rms_norm.dtype, DType::Bf16);
        assert_eq!(resolved.final_add_rms_norm.backends, ["vllm_cuda"]);
        assert_eq!(resolved.lm_head.k.get(), 2048);
        assert_eq!(resolved.lm_head.n.get(), 248_320);
        assert_eq!(resolved.lm_head.dtype, DType::Bf16);
        assert_eq!(resolved.lm_head.backends, ["torch_linear"]);

        let padded_vocab = VOCAB_SIZE.div_ceil(VOCAB_ALIGNMENT) * VOCAB_ALIGNMENT;
        assert_eq!(padded_vocab, VOCAB_SIZE);
    }

    #[test]
    fn wrong_checkpoint_identity_or_dtype_is_rejected() {
        for mutate in [
            |cfg: &mut Qwen36HeadLocalWorkletConfig| cfg.hidden = 4096.into(),
            |cfg: &mut Qwen36HeadLocalWorkletConfig| cfg.vocab_size = 248_321.into(),
        ] {
            let mut bad = cfg();
            mutate(&mut bad);
            assert!(
                std::panic::catch_unwind(|| { Qwen36HeadLocalWorklet::resolve_config(&bad) })
                    .is_err()
            );
        }
        let mut bad = cfg();
        bad.activation_dtype = DType::Fp16;
        assert!(
            std::panic::catch_unwind(|| { Qwen36HeadLocalWorklet::resolve_config(&bad) }).is_err()
        );
    }

    #[test]
    fn compile_has_exact_two_atomic_bf16_leaves() {
        let worklet = enumerate_worklet();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        assert_eq!(tree.n_slots(), 2);
        assert_eq!(
            tree.slots
                .iter()
                .map(|slot| slot.name.strip_prefix("model.head.").unwrap())
                .collect::<Vec<_>>(),
            ["final_add_rms_norm", "lm_head"]
        );
        assert_eq!(
            tree.slots
                .iter()
                .map(|slot| slot.kind.as_str())
                .collect::<Vec<_>>(),
            ["residual_rms_norm", "single_gemm"]
        );
        let resolved = Qwen36HeadLocalWorklet::resolve_config(&cfg());
        assert_eq!(
            tree.slots[0].kernel_config,
            resolved.final_add_rms_norm.describe_config()
        );
        assert_eq!(
            tree.slots[1].kernel_config,
            resolved.lm_head.describe_config()
        );
        assert!(!tree.slots.iter().any(|slot| {
            slot.kind.contains("fp8")
                || slot.kind.contains("quant")
                || matches!(
                    slot.kind.as_str(),
                    "all_reduce" | "all_to_all" | "send_recv"
                )
        }));
        match tree.root {
            CostNode::Labeled { child, .. } => match *child {
                CostNode::Sum(children) => assert_eq!(children.len(), 2),
                _ => panic!("expected Sum"),
            },
            _ => panic!("expected Labeled"),
        }
    }

    #[test]
    fn ordinary_generation_rows_map_independently() {
        for (name, final_norm_tokens, logits_tokens) in
            [("prefill", 70, 3), ("decode", 4, 4), ("mixed", 74, 7)]
        {
            let work = derive_work(&Qwen36HeadLocalWorkletInput {
                final_norm_tokens,
                logits_tokens,
            })
            .unwrap_or_else(|reason| panic!("{name}: {reason}"));
            assert_eq!(work.final_add_rms_norm.m, final_norm_tokens, "{name}");
            assert_eq!(work.lm_head.m, logits_tokens, "{name}");
        }
        // Ragged/chunked prefill [3, 65, 2] has 70 active rows but only one
        // ordinary-generation logits row for each of its three requests.
        let chunked = derive_work(&Qwen36HeadLocalWorkletInput {
            final_norm_tokens: 70,
            logits_tokens: 3,
        })
        .unwrap();
        assert_eq!((chunked.final_add_rms_norm.m, chunked.lm_head.m), (70, 3));
    }

    #[test]
    fn input_validation_allows_zero_and_norm_only_but_rejects_extra_logits() {
        assert!(derive_work(&Qwen36HeadLocalWorkletInput::default()).is_ok());
        let norm_only = derive_work(&Qwen36HeadLocalWorkletInput {
            final_norm_tokens: 8,
            logits_tokens: 0,
        })
        .unwrap();
        assert_eq!(
            (norm_only.final_add_rms_norm.m, norm_only.lm_head.m),
            (8, 0)
        );
        assert!(derive_work(&Qwen36HeadLocalWorkletInput {
            final_norm_tokens: 0,
            logits_tokens: 1,
        })
        .is_err());
    }

    #[test]
    fn zero_work_keeps_two_faithful_typed_slots_without_cache_evaluation() {
        let worklet = enumerate_worklet();
        let mut metrics = [LeafMetrics::MISS; 2];
        let mut inputs = Vec::new();
        let mut evaluator = Evaluator::with_inputs(&mut metrics, &mut inputs);
        worklet.eval(&Qwen36HeadLocalWorkletInput::default(), &mut evaluator);
        assert_eq!(evaluator.filled(), 2);
        assert!(metrics.iter().all(|metric| {
            metric.m.time_ms == 0.0
                && metric.m.flops == 0.0
                && metric.m.bytes == 0.0
                && metric.m.energy_j == 0.0
                && metric.coverage == CoverageFlags::EMPTY
        }));
        assert_eq!(
            serde_json::to_value(inputs).unwrap(),
            serde_json::json!([{"m": 0}, {"m": 0}])
        );
    }

    struct ReturnNormMetrics;

    impl Probe for ReturnNormMetrics {
        type Input = ResidualRmsNormKernelInput;

        fn eval(&self, input: &Self::Input) -> LeafMetrics {
            let mut metrics = LeafMetrics::ZERO;
            metrics.m.time_ms = input.m as f32;
            metrics
        }

        fn kind(&self) -> &'static str {
            "residual_rms_norm"
        }

        fn describe_config(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    struct MustNotEvalNorm;

    impl Probe for MustNotEvalNorm {
        type Input = ResidualRmsNormKernelInput;

        fn eval(&self, _input: &Self::Input) -> LeafMetrics {
            panic!("zero-row final norm must not evaluate its cache")
        }

        fn kind(&self) -> &'static str {
            "residual_rms_norm"
        }

        fn describe_config(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    struct MustNotEvalHead;

    impl Probe for MustNotEvalHead {
        type Input = SingleGemmKernelInput;

        fn eval(&self, _input: &Self::Input) -> LeafMetrics {
            panic!("zero-row lm_head must not evaluate its cache")
        }

        fn kind(&self) -> &'static str {
            "single_gemm"
        }

        fn describe_config(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    #[test]
    fn norm_only_executes_norm_and_skips_zero_row_head_cache() {
        let norm = Op::new("norm".into(), Arc::new(ReturnNormMetrics));
        let head = Op::new("head".into(), Arc::new(MustNotEvalHead));
        let mut metrics = [LeafMetrics::MISS; 2];
        let mut inputs = Vec::new();
        let mut evaluator = Evaluator::with_inputs(&mut metrics, &mut inputs);
        eval_atomic_or_zero(
            &norm,
            ResidualRmsNormKernelInput { m: 8 },
            false,
            &mut evaluator,
        );
        eval_atomic_or_zero(&head, SingleGemmKernelInput { m: 0 }, true, &mut evaluator);
        assert_eq!(metrics[0].m.time_ms, 8.0);
        assert_eq!(metrics[1].m.time_ms, 0.0);
        assert_eq!(metrics[1].m.flops, 0.0);
        assert_eq!(metrics[1].m.bytes, 0.0);
        assert_eq!(metrics[1].m.energy_j, 0.0);
        assert_eq!(metrics[1].coverage, CoverageFlags::EMPTY);
        assert_eq!(
            serde_json::to_value(inputs).unwrap(),
            serde_json::json!([{"m": 8}, {"m": 0}])
        );
    }

    #[test]
    fn zero_rows_skip_both_atomic_caches_independently() {
        let norm = Op::new("norm".into(), Arc::new(MustNotEvalNorm));
        let head = Op::new("head".into(), Arc::new(MustNotEvalHead));
        let mut metrics = [LeafMetrics::MISS; 2];
        let mut inputs = Vec::new();
        let mut evaluator = Evaluator::with_inputs(&mut metrics, &mut inputs);
        eval_atomic_or_zero(
            &norm,
            ResidualRmsNormKernelInput { m: 0 },
            true,
            &mut evaluator,
        );
        eval_atomic_or_zero(&head, SingleGemmKernelInput { m: 0 }, true, &mut evaluator);
        assert!(metrics.iter().all(|metric| metric.m.time_ms == 0.0));
        assert_eq!(
            serde_json::to_value(inputs).unwrap(),
            serde_json::json!([{"m": 0}, {"m": 0}])
        );
    }

    #[test]
    fn compile_topology_is_fixed_for_all_valid_row_combinations() {
        let worklet = enumerate_worklet();
        for input in [
            Qwen36HeadLocalWorkletInput::default(),
            Qwen36HeadLocalWorkletInput {
                final_norm_tokens: 8,
                logits_tokens: 0,
            },
            Qwen36HeadLocalWorkletInput {
                final_norm_tokens: 70,
                logits_tokens: 3,
            },
        ] {
            assert!(derive_work(&input).is_ok());
            let mut builder = CostTreeBuilder::new();
            let root = worklet.compile(&mut builder);
            assert_eq!(builder.finish(root).n_slots(), 2);
        }
    }
}
