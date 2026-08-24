//! GLM-5.2 local shared-expert MLP worklet in **vLLM kernel granularity**.
//!
//! Same section as [`super::glm52_shared_expert_local`]: fused gate/up
//! projection -> SiLU-and-multiply -> down projection, excluding normalization,
//! routing, routed-expert work, communication, weighting, and the final
//! routed/shared addition.
//!
//! The one divergence: vLLM launches a BF16->FP8 block quantisation
//! (`fp8_blockscale_gemm::scale_1x128_kernel`) before **each** of the two dense
//! FP8 GEMMs, so this worklet cuts those as explicit leaves. The native worklet
//! has no such leaf because the L1 GEMM runner quantises outside its timed
//! closure -- the cost is absent there, not folded in. Measured: 75 sparse
//! layers x 2 = 150 launches/iteration (see `doc/alignment/glm52_dp8_ep8_report.md`
//! section 4.3, where all 411 unmodelled quant launches are decoded by GEMM shape).
//!
//! The quant leaves are emitted only when `gemm_dtype` is FP8; a BF16 GEMM takes
//! no quantisation on either side.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    Fp8PerTokenGroupQuantKernel, Fp8PerTokenGroupQuantKernelConfig,
    Fp8PerTokenGroupQuantKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

const HIDDEN_DIM: u32 = 6144;
const MOE_INTERMEDIATE_DIM: u32 = 2048;
const N_SHARED_EXPERTS: u32 = 1;

#[cfg(test)]
const SOURCE_ORDER: [&str; 5] = [
    "gate_up_input_quant",
    "gate_up_proj",
    "silu_and_mul",
    "down_input_quant",
    "down_proj",
];

/// Raw GLM-5.2 one-shared-expert identity. There is no parallelism or
/// collective configuration inside this local section.
#[derive(Clone, Debug)]
pub struct VllmGlm52SharedExpertLocalWorkletConfig {
    pub gemm_backends: Vec<&'static str>,
    /// Backends for the pre-GEMM activation quantisation. Unused when
    /// `gemm_dtype` is BF16.
    pub fp8_quant_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    pub moe_intermediate_dim: Dim,
    pub n_shared_experts: u32,
    pub dtype: DType,
    pub gemm_dtype: DType,
}

/// Pure resolved data with the five atomic child configs fully baked. The two
/// quant configs are `Some` exactly when `gemm_dtype` is FP8.
#[derive(Clone, Debug)]
pub struct VllmGlm52SharedExpertLocalWorkletResolved {
    pub raw_cfg: VllmGlm52SharedExpertLocalWorkletConfig,
    pub gate_up_input_quant: Option<Fp8PerTokenGroupQuantKernelConfig>,
    pub gate_up_proj: SingleGemmKernelConfig,
    pub silu_and_mul: ElementwiseKernelConfig,
    pub down_input_quant: Option<Fp8PerTokenGroupQuantKernelConfig>,
    pub down_proj: SingleGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct VllmGlm52SharedExpertLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct VllmGlm52SharedExpertLocalWorklet {
    pub name: String,
    pub gate_up_input_quant: Option<Op<Fp8PerTokenGroupQuantKernel>>,
    pub gate_up_proj: Op<SingleGemmKernel>,
    pub silu_and_mul: Op<ElementwiseKernel>,
    pub down_input_quant: Option<Op<Fp8PerTokenGroupQuantKernel>>,
    pub down_proj: Op<SingleGemmKernel>,
    resolved: VllmGlm52SharedExpertLocalWorkletResolved,
}

impl VllmGlm52SharedExpertLocalWorklet {
    /// Resolve the one supported GLM-5.2 shared-expert identity without
    /// touching a bridge, GPU, cache, or `Arc`.
    #[must_use]
    pub fn resolve_config(
        cfg: &VllmGlm52SharedExpertLocalWorkletConfig,
    ) -> VllmGlm52SharedExpertLocalWorkletResolved {
        validate_config(cfg).unwrap_or_else(|reason| {
            panic!("invalid VllmGlm52SharedExpertLocalWorkletConfig: {reason}")
        });

        let shared_width = checked_product(
            "shared expert width",
            &[cfg.n_shared_experts, cfg.moe_intermediate_dim.get()],
        )
        .expect("validated GLM-5.2 shared-expert width must fit u32");
        let gate_up_n = checked_product("gate_up_proj.n", &[2, shared_width])
            .expect("validated GLM-5.2 shared gate/up dimension must fit u32");
        let silu_input_bytes = checked_product(
            "silu_and_mul.input_bytes_per_token",
            &[2, shared_width, cfg.dtype.size_bytes()],
        )
        .expect("validated GLM-5.2 shared SiLU input byte rate must fit u32");
        let silu_output_bytes = checked_product(
            "silu_and_mul.output_bytes_per_token",
            &[shared_width, cfg.dtype.size_bytes()],
        )
        .expect("validated GLM-5.2 shared SiLU output byte rate must fit u32");

        // vLLM quantises the activation once per dense FP8 GEMM: the hidden
        // row feeding gate_up, then the SiLU output feeding down.
        //
        // The dense path runs vLLM's own `per_token_group_quant_8bit_kernel`,
        // NOT the TensorRT-LLM `scale_1x128_kernel` that the routed grouped
        // GEMM uses -- the nsys trace shows both kernels side by side in the
        // same iteration. Pricing this leaf off the routed curve over-predicted
        // it by ~2.05x at prefill and ~40% at decode.
        let quant_config = |hidden_size: Dim| {
            (cfg.gemm_dtype == DType::Fp8E4m3).then(|| Fp8PerTokenGroupQuantKernelConfig {
                backends: cfg.fp8_quant_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size,
                group_size: 128,
                input_dtype: cfg.dtype,
                scale_format: "ue8m0_column_major".to_string(),
            })
        };

        VllmGlm52SharedExpertLocalWorkletResolved {
            gate_up_input_quant: quant_config(cfg.hidden_dim.clone()),
            gate_up_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: gate_up_n.into(),
                k: cfg.hidden_dim.clone(),
                dtype: cfg.gemm_dtype,
            },
            silu_and_mul: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: silu_input_bytes.into(),
                output_bytes_per_token: silu_output_bytes.into(),
            },
            down_input_quant: quant_config(shared_width.into()),
            down_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden_dim.clone(),
                k: shared_width.into(),
                dtype: cfg.gemm_dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: VllmGlm52SharedExpertLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let gate_up_input_quant = resolved
            .gate_up_input_quant
            .clone()
            .map(|config| {
                build_atomic(
                    &name,
                    "gate_up_input_quant",
                    config,
                    Fp8PerTokenGroupQuantKernel::build,
                    bridge,
                )
            })
            .transpose()?;
        let gate_up_proj = build_atomic(
            &name,
            "gate_up_proj",
            resolved.gate_up_proj.clone(),
            SingleGemmKernel::build,
            bridge,
        )?;
        let silu_and_mul = build_atomic(
            &name,
            "silu_and_mul",
            resolved.silu_and_mul.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        let down_input_quant = resolved
            .down_input_quant
            .clone()
            .map(|config| {
                build_atomic(
                    &name,
                    "down_input_quant",
                    config,
                    Fp8PerTokenGroupQuantKernel::build,
                    bridge,
                )
            })
            .transpose()?;
        let down_proj = build_atomic(
            &name,
            "down_proj",
            resolved.down_proj.clone(),
            SingleGemmKernel::build,
            bridge,
        )?;

        Ok(Self {
            name,
            gate_up_input_quant,
            gate_up_proj,
            silu_and_mul,
            down_input_quant,
            down_proj,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: worklet_label(&self.name, &self.resolved.raw_cfg),
            child: Box::new(CostNode::Sum(
                [
                    self.gate_up_input_quant
                        .as_ref()
                        .map(|quant| quant.compile(builder)),
                    Some(self.gate_up_proj.compile(builder)),
                    Some(self.silu_and_mul.compile(builder)),
                    self.down_input_quant
                        .as_ref()
                        .map(|quant| quant.compile(builder)),
                    Some(self.down_proj.compile(builder)),
                ]
                .into_iter()
                .flatten()
                .collect(),
            )),
        }
    }

    pub fn eval(&self, input: &VllmGlm52SharedExpertLocalWorkletInput, ev: &mut Evaluator) {
        let work = work_inputs(input.batch_tokens);
        let zero = input.batch_tokens == 0;

        if let Some(quant) = &self.gate_up_input_quant {
            eval_atomic_or_zero(quant, work.input_quant.clone(), zero, ev);
        }
        eval_atomic_or_zero(&self.gate_up_proj, work.gate_up_proj, zero, ev);
        eval_atomic_or_zero(&self.silu_and_mul, work.silu_and_mul, zero, ev);
        if let Some(quant) = &self.down_input_quant {
            eval_atomic_or_zero(quant, work.input_quant.clone(), zero, ev);
        }
        eval_atomic_or_zero(&self.down_proj, work.down_proj, zero, ev);
    }
}

struct WorkInputs {
    /// Both quant leaves take the same row count; only the config's
    /// `hidden_size` differs.
    input_quant: Fp8PerTokenGroupQuantKernelInput,
    gate_up_proj: SingleGemmKernelInput,
    silu_and_mul: ElementwiseKernelInput,
    down_proj: SingleGemmKernelInput,
}

fn validate_config(cfg: &VllmGlm52SharedExpertLocalWorkletConfig) -> Result<(), String> {
    for (name, actual, required) in [
        ("hidden_dim", cfg.hidden_dim.get(), HIDDEN_DIM),
        (
            "moe_intermediate_dim",
            cfg.moe_intermediate_dim.get(),
            MOE_INTERMEDIATE_DIM,
        ),
        ("n_shared_experts", cfg.n_shared_experts, N_SHARED_EXPERTS),
    ] {
        if actual != required {
            return Err(format!("{name} must be {required}, got {actual}"));
        }
    }
    if cfg.dtype != DType::Bf16 {
        return Err(format!(
            "dtype must be {}, got {}",
            DType::Bf16.as_str(),
            cfg.dtype.as_str()
        ));
    }
    if !matches!(cfg.gemm_dtype, DType::Bf16 | DType::Fp8E4m3) {
        return Err(format!(
            "gemm_dtype must be {} or {}, got {}",
            DType::Bf16.as_str(),
            DType::Fp8E4m3.as_str(),
            cfg.gemm_dtype.as_str()
        ));
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

fn work_inputs(batch_tokens: u32) -> WorkInputs {
    WorkInputs {
        input_quant: Fp8PerTokenGroupQuantKernelInput {
            num_tokens: batch_tokens,
        },
        gate_up_proj: SingleGemmKernelInput { m: batch_tokens },
        silu_and_mul: ElementwiseKernelInput {
            num_tokens: batch_tokens,
        },
        down_proj: SingleGemmKernelInput { m: batch_tokens },
    }
}

fn worklet_label(name: &str, cfg: &VllmGlm52SharedExpertLocalWorkletConfig) -> String {
    let shared_width = checked_product(
        "shared expert label width",
        &[cfg.n_shared_experts, cfg.moe_intermediate_dim.get()],
    )
    .expect("validated GLM-5.2 shared-expert label width must fit u32");
    format!(
        "{name} (VllmGlm52SharedExpertLocalWorklet) [local (1 GPU); shared_experts={}; shared_width={}]",
        cfg.n_shared_experts, shared_width
    )
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

    fn cfg() -> VllmGlm52SharedExpertLocalWorkletConfig {
        VllmGlm52SharedExpertLocalWorkletConfig {
            fp8_quant_backends: vec!["flashinfer_trtllm"],
            gemm_backends: vec!["torch_linear"],
            elementwise_backends: vec!["triton"],
            gpu_name: "NVIDIA H200".to_string(),
            hidden_dim: Dim::param("hidden_dim", HIDDEN_DIM),
            moe_intermediate_dim: Dim::param("moe_intermediate_dim", MOE_INTERMEDIATE_DIM),
            n_shared_experts: N_SHARED_EXPERTS,
            dtype: DType::Bf16,
            gemm_dtype: DType::Bf16,
        }
    }

    #[test]
    fn source_order_is_exact_local_and_excludes_other_sections() {
        // Each dense FP8 GEMM is preceded by its own activation quantisation.
        assert_eq!(
            SOURCE_ORDER,
            [
                "gate_up_input_quant",
                "gate_up_proj",
                "silu_and_mul",
                "down_input_quant",
                "down_proj",
            ]
        );
        assert_eq!(SOURCE_ORDER.len(), 5);
        for forbidden in [
            "norm",
            "router",
            "routed",
            "dispatch",
            "combine",
            "all_reduce",
            "all_to_all",
            "final_add",
            "attention",
            "mtp",
        ] {
            assert!(!SOURCE_ORDER.iter().any(|child| child.contains(forbidden)));
        }
    }

    #[test]
    fn resolve_bakes_exact_one_shared_expert_shapes_bytes_and_backends() {
        let r = VllmGlm52SharedExpertLocalWorklet::resolve_config(&cfg());

        assert_eq!(r.raw_cfg.hidden_dim, 6144);
        assert_eq!(r.raw_cfg.moe_intermediate_dim, 2048);
        assert_eq!(r.raw_cfg.n_shared_experts, 1);
        assert_eq!(r.raw_cfg.dtype, DType::Bf16);
        assert_eq!(r.gate_up_proj.n, 4096);
        assert_eq!(r.gate_up_proj.k, 6144);
        assert_eq!(r.gate_up_proj.dtype, DType::Bf16);
        assert_eq!(r.gate_up_proj.backends, vec!["torch_linear"]);
        assert_eq!(r.silu_and_mul.input_bytes_per_token, 8192);
        assert_eq!(r.silu_and_mul.output_bytes_per_token, 4096);
        assert_eq!(r.silu_and_mul.backends, vec!["triton"]);
        assert_eq!(r.down_proj.n, 6144);
        assert_eq!(r.down_proj.k, 2048);
        assert_eq!(r.down_proj.dtype, DType::Bf16);
        assert_eq!(r.down_proj.backends, vec!["torch_linear"]);
        assert_eq!(r.raw_cfg.gpu_name, "NVIDIA H200");
    }

    #[test]
    fn input_and_zero_case_feed_all_three_leaves_faithfully() {
        let work = work_inputs(73);
        assert_eq!(work.gate_up_proj.m, 73);
        assert_eq!(work.silu_and_mul.num_tokens, 73);
        assert_eq!(work.down_proj.m, 73);

        let zero = work_inputs(0);
        assert_eq!(zero.gate_up_proj.m, 0);
        assert_eq!(zero.silu_and_mul.num_tokens, 0);
        assert_eq!(zero.down_proj.m, 0);
        assert_eq!(
            VllmGlm52SharedExpertLocalWorkletInput::default().batch_tokens,
            0
        );
    }

    #[test]
    fn unsupported_glm_identity_fails_during_pure_resolution() {
        let mut bad_hidden = cfg();
        bad_hidden.hidden_dim = 4096.into();
        assert!(validate_config(&bad_hidden)
            .unwrap_err()
            .contains("hidden_dim must be 6144"));

        let mut bad_intermediate = cfg();
        bad_intermediate.moe_intermediate_dim = 4096.into();
        assert!(validate_config(&bad_intermediate)
            .unwrap_err()
            .contains("moe_intermediate_dim must be 2048"));

        let mut bad_shared_count = cfg();
        bad_shared_count.n_shared_experts = 2;
        assert!(validate_config(&bad_shared_count)
            .unwrap_err()
            .contains("n_shared_experts must be 1"));

        let mut bad_dtype = cfg();
        bad_dtype.dtype = DType::Fp16;
        assert!(validate_config(&bad_dtype)
            .unwrap_err()
            .contains("dtype must be bf16"));
    }

    #[test]
    fn checked_dimension_and_byte_math_rejects_overflow() {
        assert_eq!(checked_product("shared width", &[1, 2048]).unwrap(), 2048);
        assert_eq!(checked_product("gate n", &[2, 2048]).unwrap(), 4096);
        assert_eq!(checked_product("input", &[2, 2048, 2]).unwrap(), 8192);
        assert_eq!(checked_product("output", &[2048, 2]).unwrap(), 4096);
        assert!(checked_product("overflow", &[u32::MAX, 2])
            .unwrap_err()
            .contains("overflows u32"));
    }

    #[test]
    fn label_documents_local_one_shared_expert_boundary() {
        let label = worklet_label("layer.shared_expert", &cfg());
        assert!(label.contains("VllmGlm52SharedExpertLocalWorklet"));
        assert!(label.contains("local (1 GPU)"));
        assert!(label.contains("shared_experts=1"));
        assert!(label.contains("shared_width=2048"));
        for forbidden in ["tp=", "allreduce", "collective"] {
            assert!(!label.contains(forbidden));
        }
    }
}
