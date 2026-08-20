//! Qwen3.6 TP1/EP1 local MoE finalization section.
//!
//! This section consumes local routed-expert rows and the already-gated local
//! shared-expert output. It first unpermutes, reweights, and reduces routed rows,
//! then adds the routed and shared outputs. Delayed residual handling belongs to
//! the next attention section (or final head), so no norm is present here. The
//! worklet contains no router, alignment, expert GEMM, dispatch/combine, TP, EP,
//! collective, or network child.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, MoeFinalizeRoutingKernel,
    MoeFinalizeRoutingKernelConfig, MoeFinalizeRoutingKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

const HIDDEN: u32 = 2048;
const NUM_EXPERTS: u32 = 256;
const TOP_K: u32 = 8;
const TOTAL_PPM: u64 = 1_000_000;

#[derive(Clone, Debug)]
pub struct Qwen36MoeFinalizeLocalWorkletConfig {
    pub hidden: Dim,
    pub num_experts: Dim,
    pub top_k: u32,
    pub activation_dtype: DType,
    pub local_ppm: Vec<u32>,
    pub gpu_name: String,
    pub finalize_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Qwen36MoeFinalizeLocalWorkletResolved {
    pub finalize: MoeFinalizeRoutingKernelConfig,
    pub shared_routed_add: ElementwiseKernelConfig,
    pub raw_cfg: Qwen36MoeFinalizeLocalWorkletConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Qwen36MoeFinalizeLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct Qwen36MoeFinalizeLocalWorklet {
    pub name: String,
    pub finalize: Op<MoeFinalizeRoutingKernel>,
    pub shared_routed_add: Op<ElementwiseKernel>,
    resolved: Qwen36MoeFinalizeLocalWorkletResolved,
}

impl Qwen36MoeFinalizeLocalWorklet {
    pub fn resolve_config(
        cfg: &Qwen36MoeFinalizeLocalWorkletConfig,
    ) -> Qwen36MoeFinalizeLocalWorkletResolved {
        validate_config(cfg).unwrap_or_else(|reason| {
            panic!("invalid Qwen36MoeFinalizeLocalWorkletConfig: {reason}")
        });

        let output_bytes = cfg
            .hidden
            .get()
            .checked_mul(cfg.activation_dtype.size_bytes())
            .expect("validated hidden BF16 output byte rate must fit u32");
        let input_bytes = output_bytes
            .checked_mul(2)
            .expect("validated shared/routed input byte rate must fit u32");

        Qwen36MoeFinalizeLocalWorkletResolved {
            finalize: MoeFinalizeRoutingKernelConfig {
                backends: cfg.finalize_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden.clone(),
                top_k: cfg.top_k,
                num_experts_per_rank: cfg.num_experts.get(),
                local_ppm: cfg.local_ppm.clone(),
                dtype: cfg.activation_dtype,
            },
            shared_routed_add: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: input_bytes.into(),
                output_bytes_per_token: output_bytes.into(),
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Qwen36MoeFinalizeLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let finalize = build_atomic(
            &name,
            "finalize",
            resolved.finalize.clone(),
            MoeFinalizeRoutingKernel::build,
            bridge,
        )?;
        let shared_routed_add = build_atomic(
            &name,
            "shared_routed_add",
            resolved.shared_routed_add.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        Ok(Self {
            name,
            finalize,
            shared_routed_add,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (Qwen36MoeFinalizeLocalWorklet) [local (1 GPU); E={}, K={}, ppm={}]",
                self.name,
                cfg.num_experts,
                cfg.top_k,
                cfg.local_ppm.iter().map(|&ppm| u64::from(ppm)).sum::<u64>(),
            ),
            child: Box::new(CostNode::Sum(vec![
                self.finalize.compile(builder),
                self.shared_routed_add.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Qwen36MoeFinalizeLocalWorkletInput, ev: &mut Evaluator) {
        let work = work_inputs(input.batch_tokens);
        let zero = input.batch_tokens == 0;
        eval_atomic_or_zero(&self.finalize, work.finalize, zero, ev);
        eval_atomic_or_zero(&self.shared_routed_add, work.shared_routed_add, zero, ev);
    }
}

struct WorkInputs {
    finalize: MoeFinalizeRoutingKernelInput,
    shared_routed_add: ElementwiseKernelInput,
}

fn work_inputs(batch_tokens: u32) -> WorkInputs {
    WorkInputs {
        finalize: MoeFinalizeRoutingKernelInput {
            token_count: batch_tokens,
        },
        shared_routed_add: ElementwiseKernelInput {
            num_tokens: batch_tokens,
        },
    }
}

fn validate_config(cfg: &Qwen36MoeFinalizeLocalWorkletConfig) -> Result<(), String> {
    for (name, actual, expected) in [
        ("hidden", cfg.hidden.get(), HIDDEN),
        ("num_experts", cfg.num_experts.get(), NUM_EXPERTS),
        ("top_k", cfg.top_k, TOP_K),
    ] {
        if actual != expected {
            return Err(format!("{name} must be {expected}, got {actual}"));
        }
    }
    if cfg.activation_dtype != DType::Bf16 {
        return Err("activation_dtype must be BF16".into());
    }
    if cfg.local_ppm.len() != cfg.num_experts.get() as usize {
        return Err(format!(
            "local_ppm length {} must equal local EP1 expert count {}",
            cfg.local_ppm.len(),
            cfg.num_experts.get()
        ));
    }
    let ppm_sum = cfg.local_ppm.iter().try_fold(0_u64, |sum, &ppm| {
        sum.checked_add(u64::from(ppm))
            .ok_or("local_ppm sum overflow")
    })?;
    if ppm_sum != TOTAL_PPM {
        return Err(format!(
            "EP1 local_ppm must sum to {TOTAL_PPM}, got {ppm_sum}"
        ));
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
    use crate::worklet::uniform_local_ppm;

    fn exact_uniform_ep1_ppm() -> Vec<u32> {
        // The shared helper deliberately floors each expert's share. Start
        // from it, then distribute the 64-ppm remainder in expert-ID order so
        // this complete EP1 fixture has the required exact million total.
        let mut ppm = uniform_local_ppm(NUM_EXPERTS, 1);
        let sum: u32 = ppm.iter().sum();
        let remainder = (TOTAL_PPM as u32) - sum;
        for expert_ppm in ppm.iter_mut().take(remainder as usize) {
            *expert_ppm += 1;
        }
        ppm
    }

    fn cfg() -> Qwen36MoeFinalizeLocalWorkletConfig {
        Qwen36MoeFinalizeLocalWorkletConfig {
            hidden: HIDDEN.into(),
            num_experts: NUM_EXPERTS.into(),
            top_k: TOP_K,
            activation_dtype: DType::Bf16,
            local_ppm: exact_uniform_ep1_ppm(),
            gpu_name: "NVIDIA H200".into(),
            finalize_backends: vec!["flashinfer_trtllm"],
            elementwise_backends: vec!["triton"],
        }
    }

    fn enumerate_worklet() -> Qwen36MoeFinalizeLocalWorklet {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        Qwen36MoeFinalizeLocalWorklet::build(
            "model.moe_finalize".into(),
            Qwen36MoeFinalizeLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap()
    }

    #[test]
    fn uniform_ep1_distribution_and_resolved_configs_are_exact() {
        let ppm = exact_uniform_ep1_ppm();
        assert_eq!(ppm.len(), 256);
        assert_eq!(
            ppm.iter().map(|&value| u64::from(value)).sum::<u64>(),
            TOTAL_PPM
        );
        assert_eq!(ppm.iter().filter(|&&value| value == 3907).count(), 64);
        assert_eq!(ppm.iter().filter(|&&value| value == 3906).count(), 192);

        let resolved = Qwen36MoeFinalizeLocalWorklet::resolve_config(&cfg());
        assert_eq!(resolved.finalize.hidden_size.get(), 2048);
        assert_eq!(resolved.finalize.top_k, 8);
        assert_eq!(resolved.finalize.num_experts_per_rank, 256);
        assert_eq!(resolved.finalize.local_ppm, ppm);
        assert_eq!(resolved.finalize.dtype, DType::Bf16);
        assert_eq!(resolved.finalize.backends, ["flashinfer_trtllm"]);
        assert_eq!(
            (
                resolved.shared_routed_add.input_bytes_per_token.get(),
                resolved.shared_routed_add.output_bytes_per_token.get(),
            ),
            (8192, 4096)
        );
        assert_eq!(resolved.shared_routed_add.backends, ["triton"]);
    }

    #[test]
    fn invalid_qwen_identity_dtype_and_ppm_are_rejected() {
        for mutate in [
            |cfg: &mut Qwen36MoeFinalizeLocalWorkletConfig| cfg.hidden = 4096.into(),
            |cfg: &mut Qwen36MoeFinalizeLocalWorkletConfig| cfg.num_experts = 128.into(),
            |cfg: &mut Qwen36MoeFinalizeLocalWorkletConfig| cfg.top_k = 4,
        ] {
            let mut bad = cfg();
            mutate(&mut bad);
            assert!(std::panic::catch_unwind(|| {
                Qwen36MoeFinalizeLocalWorklet::resolve_config(&bad)
            })
            .is_err());
        }
        let mut bad = cfg();
        bad.activation_dtype = DType::Fp16;
        assert!(std::panic::catch_unwind(|| {
            Qwen36MoeFinalizeLocalWorklet::resolve_config(&bad)
        })
        .is_err());
        let mut bad = cfg();
        bad.local_ppm.pop();
        assert!(std::panic::catch_unwind(|| {
            Qwen36MoeFinalizeLocalWorklet::resolve_config(&bad)
        })
        .is_err());
        let mut bad = cfg();
        bad.local_ppm[0] -= 1;
        assert!(std::panic::catch_unwind(|| {
            Qwen36MoeFinalizeLocalWorklet::resolve_config(&bad)
        })
        .is_err());
    }

    #[test]
    fn compile_has_exact_two_children_and_two_flattened_leaves() {
        let worklet = enumerate_worklet();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        assert_eq!(tree.n_slots(), 2);
        assert_eq!(
            tree.slots
                .iter()
                .map(|slot| slot.name.strip_prefix("model.moe_finalize.").unwrap())
                .collect::<Vec<_>>(),
            ["finalize", "shared_routed_add"]
        );
        assert_eq!(
            tree.slots
                .iter()
                .map(|slot| slot.kind.as_str())
                .collect::<Vec<_>>(),
            ["moe_finalize_routing", "elementwise"]
        );
        let resolved = Qwen36MoeFinalizeLocalWorklet::resolve_config(&cfg());
        assert_eq!(
            tree.slots[0].kernel_config,
            resolved.finalize.describe_config()
        );
        assert_eq!(
            tree.slots[1].kernel_config,
            resolved.shared_routed_add.describe_config()
        );
        assert!(!tree.slots.iter().any(|slot| {
            slot.kind.contains("norm")
                || slot.kind.contains("router")
                || slot.kind.contains("align")
                || slot.kind.contains("expert")
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
    fn zero_batch_emits_two_typed_zero_logs_without_cache_evaluation() {
        let worklet = enumerate_worklet();
        let mut metrics = [LeafMetrics::MISS; 2];
        let mut inputs = Vec::new();
        let mut evaluator = Evaluator::with_inputs(&mut metrics, &mut inputs);
        worklet.eval(
            &Qwen36MoeFinalizeLocalWorkletInput { batch_tokens: 0 },
            &mut evaluator,
        );
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
            serde_json::json!([{"token_count": 0}, {"num_tokens": 0}])
        );
    }

    #[test]
    fn positive_batches_map_exact_token_count_to_both_children() {
        for tokens in [1, 128] {
            let work = work_inputs(tokens);
            assert_eq!(work.finalize.token_count, tokens);
            assert_eq!(work.shared_routed_add.num_tokens, tokens);
        }
    }
}
