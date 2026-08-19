//! Qwen3.6 TP1/EP1 local MoE router and token-alignment section.
//!
//! The preceding attention worklet owns the post-attention residual-add +
//! RMSNorm boundary, so this section consumes normalized hidden states and has
//! no norm leaf. It ends after local token alignment, before expert compute.
//! There are no dispatch, combine, TP, EP, collective, or network children.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    MoeAlignBlockSizeKernel, MoeAlignBlockSizeKernelConfig, MoeAlignBlockSizeKernelInput,
    MoeFusedTopkKernel, MoeFusedTopkKernelConfig, MoeFusedTopkKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

const HIDDEN: u32 = 2048;
const NUM_EXPERTS: u32 = 256;
const TOP_K: u32 = 8;
const BLOCK_SIZE: u32 = 16;
const ALIGNMENT_MIN_TOKENS: u32 = 9;

#[derive(Clone, Debug)]
pub struct Qwen36MoeRouterLocalWorkletConfig {
    pub hidden: Dim,
    pub num_experts: Dim,
    pub top_k: u32,
    pub block_size: u32,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub bf16_gemm_backends: Vec<&'static str>,
    pub fused_topk_backends: Vec<&'static str>,
    pub align_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Qwen36MoeRouterLocalWorkletResolved {
    pub router: SingleGemmKernelConfig,
    pub topk: MoeFusedTopkKernelConfig,
    pub align: MoeAlignBlockSizeKernelConfig,
    pub raw_cfg: Qwen36MoeRouterLocalWorkletConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Qwen36MoeRouterLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct Qwen36MoeRouterLocalWorklet {
    pub name: String,
    pub router: Op<SingleGemmKernel>,
    pub topk: Op<MoeFusedTopkKernel>,
    pub align: Op<MoeAlignBlockSizeKernel>,
    resolved: Qwen36MoeRouterLocalWorkletResolved,
}

impl Qwen36MoeRouterLocalWorklet {
    pub fn resolve_config(
        cfg: &Qwen36MoeRouterLocalWorkletConfig,
    ) -> Qwen36MoeRouterLocalWorkletResolved {
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid Qwen36MoeRouterLocalWorkletConfig: {reason}"));

        Qwen36MoeRouterLocalWorkletResolved {
            // The checkpoint excludes `mlp.gate` from FP8, and vLLM constructs
            // it with `quant_config=None`: this is one BF16 F.linear GEMM.
            router: SingleGemmKernelConfig {
                backends: cfg.bf16_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.num_experts.clone(),
                k: cfg.hidden.clone(),
                dtype: cfg.activation_dtype,
            },
            topk: MoeFusedTopkKernelConfig {
                backends: cfg.fused_topk_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_experts: cfg.num_experts.clone(),
                top_k: cfg.top_k,
                dtype: cfg.activation_dtype,
            },
            align: MoeAlignBlockSizeKernelConfig {
                backends: cfg.align_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_experts: cfg.num_experts.clone(),
                top_k: cfg.top_k,
                block_size: cfg.block_size,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Qwen36MoeRouterLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let router = build_atomic(
            &name,
            "router.gemm",
            resolved.router.clone(),
            SingleGemmKernel::build,
            bridge,
        )?;
        let topk = build_atomic(
            &name,
            "topk",
            resolved.topk.clone(),
            MoeFusedTopkKernel::build,
            bridge,
        )?;
        let align = build_atomic(
            &name,
            "align",
            resolved.align.clone(),
            MoeAlignBlockSizeKernel::build,
            bridge,
        )?;
        Ok(Self {
            name,
            router,
            topk,
            align,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (Qwen36MoeRouterLocalWorklet) [local (1 GPU); hidden={}, E={}, K={}, B={}]",
                self.name, cfg.hidden, cfg.num_experts, cfg.top_k, cfg.block_size,
            ),
            child: Box::new(CostNode::Sum(vec![
                self.router.compile(builder),
                self.topk.compile(builder),
                self.align.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Qwen36MoeRouterLocalWorkletInput, ev: &mut Evaluator) {
        let work = derive_work(input)
            .unwrap_or_else(|reason| panic!("invalid Qwen36MoeRouterLocalWorkletInput: {reason}"));
        self.router.eval(&work.router, ev);
        self.topk.eval(&work.topk, ev);
        eval_align_or_zero(&self.align, work.align, work.run_alignment, ev);
    }
}

struct WorkInputs {
    router: SingleGemmKernelInput,
    topk: MoeFusedTopkKernelInput,
    align: MoeAlignBlockSizeKernelInput,
    run_alignment: bool,
}

fn derive_work(input: &Qwen36MoeRouterLocalWorkletInput) -> Result<WorkInputs, String> {
    if input.batch_tokens == 0 {
        return Err("batch_tokens must be positive".into());
    }
    Ok(WorkInputs {
        router: SingleGemmKernelInput {
            m: input.batch_tokens,
        },
        topk: MoeFusedTopkKernelInput {
            num_tokens: input.batch_tokens,
        },
        align: MoeAlignBlockSizeKernelInput {
            num_tokens: input.batch_tokens,
        },
        run_alignment: input.batch_tokens >= ALIGNMENT_MIN_TOKENS,
    })
}

fn validate_config(cfg: &Qwen36MoeRouterLocalWorkletConfig) -> Result<(), String> {
    for (name, actual, expected) in [
        ("hidden", cfg.hidden.get(), HIDDEN),
        ("num_experts", cfg.num_experts.get(), NUM_EXPERTS),
        ("top_k", cfg.top_k, TOP_K),
        ("block_size", cfg.block_size, BLOCK_SIZE),
    ] {
        if actual != expected {
            return Err(format!("{name} must be {expected}, got {actual}"));
        }
    }
    if cfg.activation_dtype != DType::Bf16 {
        return Err("activation_dtype must be BF16".into());
    }
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

fn eval_align_or_zero<K>(
    align: &Op<K>,
    input: MoeAlignBlockSizeKernelInput,
    run_alignment: bool,
    ev: &mut Evaluator,
) where
    K: Probe<Input = MoeAlignBlockSizeKernelInput>,
{
    let metrics = if run_alignment {
        align.kernel.eval(&input)
    } else {
        LeafMetrics::ZERO
    };
    ev.push(metrics, || SlotInput::from(input));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::cache::interp::CoverageFlags;
    use crate::timing::{CostTreeBuilder, KernelConfig, PerfApiBridge};

    fn cfg() -> Qwen36MoeRouterLocalWorkletConfig {
        Qwen36MoeRouterLocalWorkletConfig {
            hidden: HIDDEN.into(),
            num_experts: NUM_EXPERTS.into(),
            top_k: TOP_K,
            block_size: BLOCK_SIZE,
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA H200".into(),
            bf16_gemm_backends: vec!["torch_linear"],
            fused_topk_backends: vec!["vllm_cuda"],
            align_backends: vec!["vllm_cuda"],
        }
    }

    fn enumerate_worklet() -> Qwen36MoeRouterLocalWorklet {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        Qwen36MoeRouterLocalWorklet::build(
            "model.moe_router".into(),
            Qwen36MoeRouterLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap()
    }

    #[test]
    fn resolution_freezes_qwen_router_topk_and_alignment_contracts() {
        let resolved = Qwen36MoeRouterLocalWorklet::resolve_config(&cfg());
        assert_eq!(
            (
                resolved.router.k.get(),
                resolved.router.n.get(),
                resolved.router.dtype,
            ),
            (2048, 256, DType::Bf16)
        );
        assert_eq!(resolved.router.gpu_name, "NVIDIA H200");
        assert_eq!(resolved.router.backends, ["torch_linear"]);

        assert_eq!(resolved.topk.num_experts.get(), 256);
        assert_eq!(resolved.topk.top_k, 8);
        assert_eq!(resolved.topk.dtype, DType::Bf16);
        assert_eq!(resolved.topk.backends, ["vllm_cuda"]);
        assert_eq!(resolved.align.num_experts.get(), 256);
        assert_eq!(resolved.align.top_k, 8);
        assert_eq!(resolved.align.block_size, 16);
        assert_eq!(resolved.align.backends, ["vllm_cuda"]);
    }

    #[test]
    fn invalid_qwen_identity_and_dtype_are_rejected() {
        for mutate in [
            |cfg: &mut Qwen36MoeRouterLocalWorkletConfig| cfg.hidden = 0.into(),
            |cfg: &mut Qwen36MoeRouterLocalWorkletConfig| cfg.hidden = 4096.into(),
            |cfg: &mut Qwen36MoeRouterLocalWorkletConfig| cfg.num_experts = 128.into(),
            |cfg: &mut Qwen36MoeRouterLocalWorkletConfig| cfg.top_k = 4,
            |cfg: &mut Qwen36MoeRouterLocalWorkletConfig| cfg.block_size = 8,
        ] {
            let mut bad = cfg();
            mutate(&mut bad);
            assert!(std::panic::catch_unwind(|| {
                Qwen36MoeRouterLocalWorklet::resolve_config(&bad)
            })
            .is_err());
        }
        let mut bad = cfg();
        bad.activation_dtype = DType::Fp16;
        assert!(
            std::panic::catch_unwind(|| { Qwen36MoeRouterLocalWorklet::resolve_config(&bad) })
                .is_err()
        );
    }

    #[test]
    fn compile_has_exact_three_children_and_three_flattened_leaves() {
        let worklet = enumerate_worklet();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        assert_eq!(tree.n_slots(), 3);
        assert_eq!(
            tree.slots
                .iter()
                .map(|slot| slot.name.strip_prefix("model.moe_router.").unwrap())
                .collect::<Vec<_>>(),
            ["router.gemm", "topk", "align"]
        );
        assert_eq!(
            tree.slots
                .iter()
                .map(|slot| slot.kind.as_str())
                .collect::<Vec<_>>(),
            ["single_gemm", "moe_fused_topk", "moe_align_block_size"]
        );
        assert_eq!(tree.slots[0].kernel_config, resolved_router_description());
        assert_eq!(tree.slots[1].kernel_config, resolved_topk_description());
        assert_eq!(tree.slots[2].kernel_config, resolved_align_description());
        assert!(!tree
            .slots
            .iter()
            .any(|slot| slot.name.ends_with("router.input_quant")));
        assert!(!tree.slots.iter().any(|slot| {
            slot.kind.contains("norm")
                || slot.kind.contains("expert")
                || matches!(
                    slot.kind.as_str(),
                    "all_reduce" | "all_to_all" | "send_recv"
                )
        }));
        match tree.root {
            CostNode::Labeled { child, .. } => match *child {
                CostNode::Sum(children) => assert_eq!(children.len(), 3),
                _ => panic!("expected Sum"),
            },
            _ => panic!("expected Labeled"),
        }
    }

    fn resolved_router_description() -> serde_json::Value {
        Qwen36MoeRouterLocalWorklet::resolve_config(&cfg())
            .router
            .describe_config()
    }

    fn resolved_topk_description() -> serde_json::Value {
        Qwen36MoeRouterLocalWorklet::resolve_config(&cfg())
            .topk
            .describe_config()
    }

    fn resolved_align_description() -> serde_json::Value {
        Qwen36MoeRouterLocalWorklet::resolve_config(&cfg())
            .align
            .describe_config()
    }

    struct MustNotEval;

    impl Probe for MustNotEval {
        type Input = MoeAlignBlockSizeKernelInput;

        fn eval(&self, _input: &Self::Input) -> LeafMetrics {
            panic!("T<=8 alignment bypass must not evaluate the cache")
        }

        fn kind(&self) -> &'static str {
            "moe_align_block_size"
        }

        fn describe_config(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    #[test]
    fn t1_and_t8_bypass_alignment_with_faithful_typed_zero_logs() {
        let align = Op::new("align".into(), Arc::new(MustNotEval));
        for tokens in [1, 8] {
            let work = derive_work(&Qwen36MoeRouterLocalWorkletInput {
                batch_tokens: tokens,
            })
            .unwrap();
            assert!(!work.run_alignment);
            let mut metrics = [LeafMetrics::MISS];
            let mut inputs = Vec::new();
            let mut evaluator = Evaluator::with_inputs(&mut metrics, &mut inputs);
            eval_align_or_zero(&align, work.align, work.run_alignment, &mut evaluator);
            assert_eq!(metrics[0].m.time_ms, 0.0);
            assert_eq!(metrics[0].coverage, CoverageFlags::EMPTY);
            assert_eq!(
                serde_json::to_value(inputs).unwrap(),
                serde_json::json!([{"num_tokens": tokens}])
            );
        }
    }

    struct ReturnMetrics;

    impl Probe for ReturnMetrics {
        type Input = MoeAlignBlockSizeKernelInput;

        fn eval(&self, input: &Self::Input) -> LeafMetrics {
            let mut metrics = LeafMetrics::ZERO;
            metrics.m.time_ms = input.num_tokens as f32;
            metrics
        }

        fn kind(&self) -> &'static str {
            "moe_align_block_size"
        }

        fn describe_config(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    #[test]
    fn t9_and_t128_evaluate_alignment_with_the_exact_shared_token_count() {
        let align = Op::new("align".into(), Arc::new(ReturnMetrics));
        for tokens in [9, 128] {
            let work = derive_work(&Qwen36MoeRouterLocalWorkletInput {
                batch_tokens: tokens,
            })
            .unwrap();
            assert_eq!(work.router.m, tokens);
            assert_eq!(work.topk.num_tokens, tokens);
            assert_eq!(work.align.num_tokens, tokens);
            assert!(work.run_alignment);
            let mut metrics = [LeafMetrics::MISS];
            let mut inputs = Vec::new();
            let mut evaluator = Evaluator::with_inputs(&mut metrics, &mut inputs);
            eval_align_or_zero(&align, work.align, work.run_alignment, &mut evaluator);
            assert_eq!(metrics[0].m.time_ms, tokens as f32);
            assert_eq!(
                serde_json::to_value(inputs).unwrap(),
                serde_json::json!([{"num_tokens": tokens}])
            );
        }
    }

    #[test]
    fn input_rejects_zero_and_topology_remains_fixed_for_every_positive_t() {
        assert!(derive_work(&Qwen36MoeRouterLocalWorkletInput { batch_tokens: 0 }).is_err());
        let worklet = enumerate_worklet();
        for tokens in [1, 8, 9, 128] {
            assert!(derive_work(&Qwen36MoeRouterLocalWorkletInput {
                batch_tokens: tokens,
            })
            .is_ok());
            let mut builder = CostTreeBuilder::new();
            let root = worklet.compile(&mut builder);
            assert_eq!(builder.finish(root).n_slots(), 3);
        }
    }
}
