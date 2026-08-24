//! Qwen3.6 TP1/EP1 local shared-expert section.
//!
//! The section consumes already-normalized hidden states and executes one
//! shared expert sequentially. At the higher composition layer it can run in
//! parallel with routed-expert compute, but this worklet itself is strictly
//! gate/up -> activation -> down -> scalar gate -> sigmoid -> gate application.
//! It excludes routing, routed experts, finalize/reduction, routed/shared
//! addition, norms, TP, EP, collectives, and network work.
//!
//! The gate is three launches, not one. `sigmoid(gate(x)) * shared_out` is
//! written in eager `PyTorch`, so the sigmoid is its own `at::native`
//! elementwise kernel rather than an epilogue on the projection. Folding it
//! into the projection leaf under-predicted the measured gate operation by
//! 52.5% against a vLLM Qwen3.6-35B-A3B-FP8 capture (224.5 ms simulated vs
//! 472.5 ms measured over 966 iterations) — at a one-column projection the
//! sigmoid's own launch is a comparable cost, not a rounding error.

use std::sync::Arc;

use crate::op::gemm::{
    SingleFp8GemmWithQuantConfig, SingleFp8GemmWithQuantInput, SingleFp8GemmWithQuantOp,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    Fp8PerTokenGroupQuantKernelConfig, Fp8PerTokenGroupQuantKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

const HIDDEN: u32 = 2048;
const INTERMEDIATE: u32 = 512;
const NUM_SHARED_EXPERTS: u32 = 1;
const FP8_GROUP_SIZE: u32 = 128;
const SCALE_FORMAT: &str = "ue8m0_column_major";

#[derive(Clone, Debug)]
pub struct Qwen36SharedExpertLocalWorkletConfig {
    pub hidden: Dim,
    pub intermediate: Dim,
    pub num_shared_experts: u32,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub fp8_quant_backends: Vec<&'static str>,
    pub fp8_gemm_backends: Vec<&'static str>,
    pub bf16_gemm_backends: Vec<&'static str>,
    /// Realization for `silu_and_mul`, which vLLM computes with its own fused
    /// `act_and_mul_kernel` rather than eager tensor arithmetic.
    pub elementwise_backends: Vec<&'static str>,
    /// Realization for the gate path (`sigmoid` and its application), which the
    /// model source writes as plain `PyTorch` and so runs on `TensorIterator`.
    pub gate_elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Qwen36SharedExpertLocalWorkletResolved {
    pub gate_up: SingleFp8GemmWithQuantConfig,
    pub silu_and_mul: ElementwiseKernelConfig,
    pub down: SingleFp8GemmWithQuantConfig,
    pub shared_gate: SingleGemmKernelConfig,
    pub shared_gate_sigmoid: ElementwiseKernelConfig,
    pub apply_shared_gate: ElementwiseKernelConfig,
    pub raw_cfg: Qwen36SharedExpertLocalWorkletConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Qwen36SharedExpertLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct Qwen36SharedExpertLocalWorklet {
    pub name: String,
    pub gate_up: SingleFp8GemmWithQuantOp,
    pub silu_and_mul: Op<ElementwiseKernel>,
    pub down: SingleFp8GemmWithQuantOp,
    pub shared_gate: Op<SingleGemmKernel>,
    pub shared_gate_sigmoid: Op<ElementwiseKernel>,
    pub apply_shared_gate: Op<ElementwiseKernel>,
    resolved: Qwen36SharedExpertLocalWorkletResolved,
}

impl Qwen36SharedExpertLocalWorklet {
    #[must_use]
    pub fn resolve_config(
        cfg: &Qwen36SharedExpertLocalWorkletConfig,
    ) -> Qwen36SharedExpertLocalWorkletResolved {
        validate_config(cfg).unwrap_or_else(|reason| {
            panic!("invalid Qwen36SharedExpertLocalWorkletConfig: {reason}")
        });

        let shared_width = checked_product(
            "shared expert width",
            &[cfg.num_shared_experts, cfg.intermediate.get()],
        )
        .expect("validated shared-expert width must fit u32");
        let gate_up_width = checked_product("gate/up width", &[2, shared_width])
            .expect("validated gate/up width must fit u32");
        let silu_input_bytes = checked_product(
            "SiLU-and-multiply input bytes",
            &[2, shared_width, cfg.activation_dtype.size_bytes()],
        )
        .expect("validated SiLU input byte rate must fit u32");
        let silu_output_bytes = checked_product(
            "SiLU-and-multiply output bytes",
            &[shared_width, cfg.activation_dtype.size_bytes()],
        )
        .expect("validated SiLU output byte rate must fit u32");
        let apply_gate_input_bytes = checked_product(
            "shared gate application expert bytes",
            &[cfg.hidden.get(), cfg.activation_dtype.size_bytes()],
        )
        .and_then(|expert_bytes| {
            expert_bytes
                .checked_add(cfg.activation_dtype.size_bytes())
                .ok_or_else(|| "shared gate application input bytes overflow".to_string())
        })
        .expect("validated gate-application input byte rate must fit u32");
        let apply_gate_output_bytes = checked_product(
            "shared gate application output bytes",
            &[cfg.hidden.get(), cfg.activation_dtype.size_bytes()],
        )
        .expect("validated gate-application output byte rate must fit u32");
        let gate_scalar_bytes = cfg.activation_dtype.size_bytes();
        let quant = |hidden_size: Dim| Fp8PerTokenGroupQuantKernelConfig {
            backends: cfg.fp8_quant_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden_size,
            group_size: FP8_GROUP_SIZE,
            input_dtype: cfg.activation_dtype,
            scale_format: SCALE_FORMAT.to_string(),
        };

        Qwen36SharedExpertLocalWorkletResolved {
            gate_up: SingleFp8GemmWithQuantConfig {
                quant: quant(cfg.hidden.clone()),
                gemm: SingleGemmKernelConfig {
                    backends: cfg.fp8_gemm_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    n: gate_up_width.into(),
                    k: cfg.hidden.clone(),
                    dtype: DType::Fp8E4m3,
                },
            },
            silu_and_mul: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: silu_input_bytes.into(),
                output_bytes_per_token: silu_output_bytes.into(),
            },
            down: SingleFp8GemmWithQuantConfig {
                quant: quant(shared_width.into()),
                gemm: SingleGemmKernelConfig {
                    backends: cfg.fp8_gemm_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    n: cfg.hidden.clone(),
                    k: shared_width.into(),
                    dtype: DType::Fp8E4m3,
                },
            },
            // Checkpoint `shared_expert_gate` is ReplicatedLinear(...,
            // quant_config=None): it remains an atomic BF16 GEMM and consumes
            // the original normalized hidden input after the down projection.
            shared_gate: SingleGemmKernelConfig {
                backends: cfg.bf16_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: 1.into(),
                k: cfg.hidden.clone(),
                dtype: cfg.activation_dtype,
            },
            // One BF16 scalar in, one out, per token: the projection emits
            // `n=1`, so this leaf is entirely launch-bound and its byte rate
            // exists only to keep the elementwise contract honest.
            shared_gate_sigmoid: ElementwiseKernelConfig {
                backends: cfg.gate_elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: gate_scalar_bytes.into(),
                output_bytes_per_token: gate_scalar_bytes.into(),
            },
            apply_shared_gate: ElementwiseKernelConfig {
                backends: cfg.gate_elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: apply_gate_input_bytes.into(),
                output_bytes_per_token: apply_gate_output_bytes.into(),
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Qwen36SharedExpertLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let gate_up = SingleFp8GemmWithQuantOp::build(
            format!("{name}.gate_up"),
            resolved.gate_up.clone(),
            bridge,
        )?;
        let silu_and_mul = build_atomic(
            &name,
            "silu_and_mul",
            resolved.silu_and_mul.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        let down =
            SingleFp8GemmWithQuantOp::build(format!("{name}.down"), resolved.down.clone(), bridge)?;
        let shared_gate = build_atomic(
            &name,
            "shared_gate",
            resolved.shared_gate.clone(),
            SingleGemmKernel::build,
            bridge,
        )?;
        let shared_gate_sigmoid = build_atomic(
            &name,
            "shared_gate_sigmoid",
            resolved.shared_gate_sigmoid.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        let apply_shared_gate = build_atomic(
            &name,
            "apply_shared_gate",
            resolved.apply_shared_gate.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        Ok(Self {
            name,
            gate_up,
            silu_and_mul,
            down,
            shared_gate,
            shared_gate_sigmoid,
            apply_shared_gate,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (Qwen36SharedExpertLocalWorklet) [local (1 GPU); shared_experts={}; width={}]",
                self.name, cfg.num_shared_experts, cfg.intermediate,
            ),
            child: Box::new(CostNode::Sum(vec![
                self.gate_up.compile(builder),
                self.silu_and_mul.compile(builder),
                self.down.compile(builder),
                self.shared_gate.compile(builder),
                self.shared_gate_sigmoid.compile(builder),
                self.apply_shared_gate.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Qwen36SharedExpertLocalWorkletInput, ev: &mut Evaluator) {
        let work = work_inputs(input.batch_tokens);
        let zero = input.batch_tokens == 0;
        eval_fp8_or_zero(&self.gate_up, work.gate_up, zero, ev);
        eval_atomic_or_zero(&self.silu_and_mul, work.silu_and_mul, zero, ev);
        eval_fp8_or_zero(&self.down, work.down, zero, ev);
        eval_atomic_or_zero(&self.shared_gate, work.shared_gate, zero, ev);
        eval_atomic_or_zero(
            &self.shared_gate_sigmoid,
            work.shared_gate_sigmoid,
            zero,
            ev,
        );
        eval_atomic_or_zero(&self.apply_shared_gate, work.apply_shared_gate, zero, ev);
    }
}

struct WorkInputs {
    gate_up: SingleFp8GemmWithQuantInput,
    silu_and_mul: ElementwiseKernelInput,
    down: SingleFp8GemmWithQuantInput,
    shared_gate: SingleGemmKernelInput,
    shared_gate_sigmoid: ElementwiseKernelInput,
    apply_shared_gate: ElementwiseKernelInput,
}

fn work_inputs(batch_tokens: u32) -> WorkInputs {
    WorkInputs {
        gate_up: SingleFp8GemmWithQuantInput {
            num_tokens: batch_tokens,
        },
        silu_and_mul: ElementwiseKernelInput {
            num_tokens: batch_tokens,
        },
        down: SingleFp8GemmWithQuantInput {
            num_tokens: batch_tokens,
        },
        shared_gate: SingleGemmKernelInput { m: batch_tokens },
        shared_gate_sigmoid: ElementwiseKernelInput {
            num_tokens: batch_tokens,
        },
        apply_shared_gate: ElementwiseKernelInput {
            num_tokens: batch_tokens,
        },
    }
}

fn validate_config(cfg: &Qwen36SharedExpertLocalWorkletConfig) -> Result<(), String> {
    for (name, actual, expected) in [
        ("hidden", cfg.hidden.get(), HIDDEN),
        ("intermediate", cfg.intermediate.get(), INTERMEDIATE),
        (
            "num_shared_experts",
            cfg.num_shared_experts,
            NUM_SHARED_EXPERTS,
        ),
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

fn checked_product(name: &str, factors: &[u32]) -> Result<u32, String> {
    factors.iter().try_fold(1_u32, |value, &factor| {
        value
            .checked_mul(factor)
            .ok_or_else(|| format!("{name} overflows u32"))
    })
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

fn eval_fp8_or_zero(
    op: &SingleFp8GemmWithQuantOp,
    input: SingleFp8GemmWithQuantInput,
    zero: bool,
    ev: &mut Evaluator,
) {
    if zero {
        ev.push(LeafMetrics::ZERO, || {
            SlotInput::from(Fp8PerTokenGroupQuantKernelInput { num_tokens: 0 })
        });
        ev.push(LeafMetrics::ZERO, || {
            SlotInput::from(SingleGemmKernelInput { m: 0 })
        });
    } else {
        op.eval(&input, ev);
    }
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
    use crate::timing::{CostTreeBuilder, PerfApiBridge};

    fn cfg() -> Qwen36SharedExpertLocalWorkletConfig {
        Qwen36SharedExpertLocalWorkletConfig {
            hidden: HIDDEN.into(),
            intermediate: INTERMEDIATE.into(),
            num_shared_experts: NUM_SHARED_EXPERTS,
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA H200".into(),
            fp8_quant_backends: vec!["vllm_cuda"],
            fp8_gemm_backends: vec!["deepgemm"],
            bf16_gemm_backends: vec!["torch_linear"],
            elementwise_backends: vec!["triton"],
            gate_elementwise_backends: vec!["torch"],
        }
    }

    fn enumerate_worklet() -> Qwen36SharedExpertLocalWorklet {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        Qwen36SharedExpertLocalWorklet::build(
            "model.shared_expert".into(),
            Qwen36SharedExpertLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap()
    }

    #[test]
    fn resolution_freezes_fp8_dense_shapes_and_elementwise_byte_rates() {
        let resolved = Qwen36SharedExpertLocalWorklet::resolve_config(&cfg());
        assert_eq!(
            (
                resolved.gate_up.quant.hidden_size.get(),
                resolved.gate_up.gemm.k.get(),
                resolved.gate_up.gemm.n.get(),
            ),
            (2048, 2048, 1024)
        );
        assert_eq!(resolved.gate_up.quant.group_size, 128);
        assert_eq!(resolved.gate_up.quant.input_dtype, DType::Bf16);
        assert_eq!(resolved.gate_up.quant.scale_format, SCALE_FORMAT);
        assert_eq!(resolved.gate_up.gemm.dtype, DType::Fp8E4m3);
        assert_eq!(
            (
                resolved.down.quant.hidden_size.get(),
                resolved.down.gemm.k.get(),
                resolved.down.gemm.n.get(),
            ),
            (512, 512, 2048)
        );
        assert_eq!(resolved.down.gemm.dtype, DType::Fp8E4m3);
        assert_eq!(
            (
                resolved.silu_and_mul.input_bytes_per_token.get(),
                resolved.silu_and_mul.output_bytes_per_token.get(),
            ),
            (2048, 1024)
        );
        assert_eq!(
            (
                resolved.apply_shared_gate.input_bytes_per_token.get(),
                resolved.apply_shared_gate.output_bytes_per_token.get(),
            ),
            (4098, 4096)
        );
    }

    #[test]
    fn shared_gate_is_explicit_unquantized_bf16_k2048_n1_after_down() {
        let resolved = Qwen36SharedExpertLocalWorklet::resolve_config(&cfg());
        assert_eq!(resolved.shared_gate.k.get(), 2048);
        assert_eq!(resolved.shared_gate.n.get(), 1);
        assert_eq!(resolved.shared_gate.dtype, DType::Bf16);
        assert_eq!(resolved.shared_gate.backends, ["torch_linear"]);

        let worklet = enumerate_worklet();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        assert_eq!(tree.slots[5].name, "model.shared_expert.shared_gate");
        assert_eq!(tree.slots[5].kind, "single_gemm");
        // The eager `sigmoid` is its own launch and must stay its own leaf; see
        // the module header for what folding it away cost.
        assert_eq!(
            tree.slots[6].name,
            "model.shared_expert.shared_gate_sigmoid"
        );
        assert_eq!(tree.slots[6].kind, "elementwise");
        assert_eq!(
            (
                resolved.shared_gate_sigmoid.input_bytes_per_token.get(),
                resolved.shared_gate_sigmoid.output_bytes_per_token.get(),
            ),
            (2, 2)
        );
        assert_eq!(tree.slots[7].name, "model.shared_expert.apply_shared_gate");
    }

    #[test]
    fn invalid_qwen_identity_dtype_and_shared_count_are_rejected() {
        for mutate in [
            |cfg: &mut Qwen36SharedExpertLocalWorkletConfig| cfg.hidden = 4096.into(),
            |cfg: &mut Qwen36SharedExpertLocalWorkletConfig| cfg.intermediate = 1024.into(),
            |cfg: &mut Qwen36SharedExpertLocalWorkletConfig| cfg.num_shared_experts = 2,
        ] {
            let mut bad = cfg();
            mutate(&mut bad);
            assert!(std::panic::catch_unwind(|| {
                Qwen36SharedExpertLocalWorklet::resolve_config(&bad)
            })
            .is_err());
        }
        let mut bad = cfg();
        bad.activation_dtype = DType::Fp16;
        assert!(std::panic::catch_unwind(|| {
            Qwen36SharedExpertLocalWorklet::resolve_config(&bad)
        })
        .is_err());
    }

    #[test]
    fn compile_has_exact_six_children_and_eight_flattened_leaves() {
        let worklet = enumerate_worklet();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        assert_eq!(tree.n_slots(), 8);
        assert_eq!(
            tree.slots
                .iter()
                .map(|slot| slot.name.strip_prefix("model.shared_expert.").unwrap())
                .collect::<Vec<_>>(),
            [
                "gate_up.input_quant",
                "gate_up.gemm",
                "silu_and_mul",
                "down.input_quant",
                "down.gemm",
                "shared_gate",
                "shared_gate_sigmoid",
                "apply_shared_gate",
            ]
        );
        assert_eq!(
            tree.slots
                .iter()
                .map(|slot| slot.kind.as_str())
                .collect::<Vec<_>>(),
            [
                "fp8_per_token_group_quant",
                "single_gemm",
                "elementwise",
                "fp8_per_token_group_quant",
                "single_gemm",
                "single_gemm",
                "elementwise",
                "elementwise",
            ]
        );
        assert!(!tree.slots.iter().any(|slot| {
            slot.kind.contains("norm")
                || slot.kind.contains("router")
                || slot.kind.contains("expert_compute")
                || slot.kind.contains("finalize")
                || matches!(
                    slot.kind.as_str(),
                    "all_reduce" | "all_to_all" | "send_recv"
                )
        }));
        match tree.root {
            CostNode::Labeled { child, .. } => match *child {
                CostNode::Sum(children) => assert_eq!(children.len(), 6),
                _ => panic!("expected Sum"),
            },
            _ => panic!("expected Labeled"),
        }
    }

    #[test]
    fn zero_batch_emits_eight_typed_zero_slots_without_cache_evaluation() {
        let worklet = enumerate_worklet();
        let mut metrics = [LeafMetrics::MISS; 8];
        let mut inputs = Vec::new();
        let mut evaluator = Evaluator::with_inputs(&mut metrics, &mut inputs);
        worklet.eval(
            &Qwen36SharedExpertLocalWorkletInput { batch_tokens: 0 },
            &mut evaluator,
        );
        assert_eq!(evaluator.filled(), 8);
        assert!(metrics.iter().all(|metric| {
            metric.m.time_ms == 0.0
                && metric.m.flops == 0.0
                && metric.m.bytes == 0.0
                && metric.m.energy_j == 0.0
                && metric.coverage == CoverageFlags::EMPTY
        }));
        assert_eq!(
            serde_json::to_value(inputs).unwrap(),
            serde_json::json!([
                {"num_tokens": 0},
                {"m": 0},
                {"num_tokens": 0},
                {"num_tokens": 0},
                {"m": 0},
                {"m": 0},
                {"num_tokens": 0},
                {"num_tokens": 0},
            ])
        );
    }

    #[test]
    fn positive_batch_maps_the_same_token_count_to_every_child() {
        let work = work_inputs(128);
        assert_eq!(work.gate_up.num_tokens, 128);
        assert_eq!(work.silu_and_mul.num_tokens, 128);
        assert_eq!(work.down.num_tokens, 128);
        assert_eq!(work.shared_gate.m, 128);
        assert_eq!(work.shared_gate_sigmoid.num_tokens, 128);
        assert_eq!(work.apply_shared_gate.num_tokens, 128);
    }

    #[test]
    fn compiled_configs_preserve_independent_backend_roles() {
        let worklet = enumerate_worklet();
        assert_eq!(worklet.silu_and_mul.kernel.config.backends, ["triton"]);
        assert_eq!(worklet.shared_gate.kernel.config.backends, ["torch_linear"]);
        // The gate path is torch-realized; `silu_and_mul` is not. A single
        // shared backend list here would silently re-merge them.
        assert_eq!(
            worklet.shared_gate_sigmoid.kernel.config.backends,
            ["torch"]
        );
        assert_eq!(worklet.apply_shared_gate.kernel.config.backends, ["torch"]);
        assert_eq!(worklet.shared_gate.kernel.config.dtype, DType::Bf16);
    }
}
