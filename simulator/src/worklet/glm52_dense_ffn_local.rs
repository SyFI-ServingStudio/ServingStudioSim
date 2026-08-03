//! GLM-5.2 local dense-FFN worklet for decoder layers 0–2.
//!
//! This is one self-completing, single-GPU sync section: fused residual-add
//! RMSNorm → fused gate/up projection → SiLU-and-multiply → down projection.
//! It has no TP partition or collective. The residual addition after the down
//! projection is owned by the following layer's fused residual RMSNorm, matching
//! the accepted GLM-5.2 attention-worklet convention.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, ResidualRmsNormKernel,
    ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

const HIDDEN_DIM: u32 = 6144;
const INTERMEDIATE_DIM: u32 = 12288;

#[cfg(test)]
const SOURCE_ORDER: [&str; 4] = [
    "post_attn_add_rms_norm",
    "gate_up_proj",
    "silu_and_mul",
    "down_proj",
];

/// Raw GLM-5.2 dense-FFN identity. This local worklet deliberately carries no
/// parallelism or collective configuration.
#[derive(Clone, Debug)]
pub struct Glm52DenseFfnLocalWorkletConfig {
    pub residual_norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    pub intermediate_dim: Dim,
    /// Base activation/norm dtype. GEMM leaves use `gemm_dtype`.
    pub dtype: DType,
    pub gemm_dtype: DType,
}

/// Pure resolved data with every atomic child config fully baked.
#[derive(Clone, Debug)]
pub struct Glm52DenseFfnLocalWorkletResolved {
    pub raw_cfg: Glm52DenseFfnLocalWorkletConfig,
    pub post_attn_add_rms_norm: ResidualRmsNormKernelConfig,
    pub gate_up_proj: SingleGemmKernelConfig,
    pub silu_and_mul: ElementwiseKernelConfig,
    pub down_proj: SingleGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Glm52DenseFfnLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct Glm52DenseFfnLocalWorklet {
    pub name: String,
    pub post_attn_add_rms_norm: Op<ResidualRmsNormKernel>,
    pub gate_up_proj: Op<SingleGemmKernel>,
    pub silu_and_mul: Op<ElementwiseKernel>,
    pub down_proj: Op<SingleGemmKernel>,
    resolved: Glm52DenseFfnLocalWorkletResolved,
}

impl Glm52DenseFfnLocalWorklet {
    /// Resolve the one supported GLM-5.2 dense-FFN identity without touching a
    /// bridge, GPU, cache, or `Arc`.
    pub fn resolve_config(
        cfg: &Glm52DenseFfnLocalWorkletConfig,
    ) -> Glm52DenseFfnLocalWorkletResolved {
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid Glm52DenseFfnLocalWorkletConfig: {reason}"));

        let dtype_bytes = cfg.dtype.size_bytes();
        let gate_up_n = checked_product("gate_up_proj.n", &[2, cfg.intermediate_dim.get()])
            .expect("validated GLM-5.2 gate/up dimensions must fit u32");
        let silu_input_bytes = checked_product(
            "silu_and_mul.input_bytes_per_token",
            &[2, cfg.intermediate_dim.get(), dtype_bytes],
        )
        .expect("validated GLM-5.2 SiLU input byte rate must fit u32");
        let silu_output_bytes = checked_product(
            "silu_and_mul.output_bytes_per_token",
            &[cfg.intermediate_dim.get(), dtype_bytes],
        )
        .expect("validated GLM-5.2 SiLU output byte rate must fit u32");

        Glm52DenseFfnLocalWorkletResolved {
            post_attn_add_rms_norm: ResidualRmsNormKernelConfig {
                backends: cfg.residual_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden_dim.clone(),
                dtype: cfg.dtype,
            },
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
            down_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden_dim.clone(),
                k: cfg.intermediate_dim.clone(),
                dtype: cfg.gemm_dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Glm52DenseFfnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let post_attn_add_rms_norm = build_atomic(
            &name,
            "post_attn_add_rms_norm",
            resolved.post_attn_add_rms_norm.clone(),
            ResidualRmsNormKernel::build,
            bridge,
        )?;
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
        let down_proj = build_atomic(
            &name,
            "down_proj",
            resolved.down_proj.clone(),
            SingleGemmKernel::build,
            bridge,
        )?;

        Ok(Self {
            name,
            post_attn_add_rms_norm,
            gate_up_proj,
            silu_and_mul,
            down_proj,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: worklet_label(&self.name, &self.resolved.raw_cfg),
            child: Box::new(CostNode::Sum(vec![
                self.post_attn_add_rms_norm.compile(builder),
                self.gate_up_proj.compile(builder),
                self.silu_and_mul.compile(builder),
                self.down_proj.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Glm52DenseFfnLocalWorkletInput, ev: &mut Evaluator) {
        let work = work_inputs(input.batch_tokens);
        let zero = input.batch_tokens == 0;

        eval_atomic_or_zero(
            &self.post_attn_add_rms_norm,
            work.post_attn_add_rms_norm,
            zero,
            ev,
        );
        eval_atomic_or_zero(&self.gate_up_proj, work.gate_up_proj, zero, ev);
        eval_atomic_or_zero(&self.silu_and_mul, work.silu_and_mul, zero, ev);
        eval_atomic_or_zero(&self.down_proj, work.down_proj, zero, ev);
    }
}

struct WorkInputs {
    post_attn_add_rms_norm: ResidualRmsNormKernelInput,
    gate_up_proj: SingleGemmKernelInput,
    silu_and_mul: ElementwiseKernelInput,
    down_proj: SingleGemmKernelInput,
}

fn validate_config(cfg: &Glm52DenseFfnLocalWorkletConfig) -> Result<(), String> {
    for (name, actual, required) in [
        ("hidden_dim", cfg.hidden_dim.get(), HIDDEN_DIM),
        (
            "intermediate_dim",
            cfg.intermediate_dim.get(),
            INTERMEDIATE_DIM,
        ),
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
        post_attn_add_rms_norm: ResidualRmsNormKernelInput { m: batch_tokens },
        gate_up_proj: SingleGemmKernelInput { m: batch_tokens },
        silu_and_mul: ElementwiseKernelInput {
            num_tokens: batch_tokens,
        },
        down_proj: SingleGemmKernelInput { m: batch_tokens },
    }
}

fn worklet_label(name: &str, cfg: &Glm52DenseFfnLocalWorkletConfig) -> String {
    format!(
        "{name} (Glm52DenseFfnLocalWorklet) [local (1 GPU); hidden={}; intermediate={}]",
        cfg.hidden_dim, cfg.intermediate_dim
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

    fn cfg() -> Glm52DenseFfnLocalWorkletConfig {
        Glm52DenseFfnLocalWorkletConfig {
            residual_norm_backends: vec!["vllm_cuda"],
            gemm_backends: vec!["torch"],
            elementwise_backends: vec!["triton"],
            gpu_name: "NVIDIA H200".to_string(),
            hidden_dim: Dim::param("hidden_dim", HIDDEN_DIM),
            intermediate_dim: Dim::param("intermediate_dim", INTERMEDIATE_DIM),
            dtype: DType::Bf16,
            gemm_dtype: DType::Bf16,
        }
    }

    #[test]
    fn source_order_is_exact_local_and_excludes_other_sections() {
        assert_eq!(
            SOURCE_ORDER,
            [
                "post_attn_add_rms_norm",
                "gate_up_proj",
                "silu_and_mul",
                "down_proj",
            ]
        );
        assert_eq!(SOURCE_ORDER.len(), 4);
        for forbidden in [
            "attention",
            "router",
            "moe",
            "all_reduce",
            "communication",
            "mtp",
        ] {
            assert!(!SOURCE_ORDER.iter().any(|child| child.contains(forbidden)));
        }
    }

    #[test]
    fn resolve_bakes_exact_glm_shapes_bytes_and_backend_roles() {
        let r = Glm52DenseFfnLocalWorklet::resolve_config(&cfg());

        assert_eq!(r.raw_cfg.hidden_dim, HIDDEN_DIM);
        assert_eq!(r.raw_cfg.intermediate_dim, INTERMEDIATE_DIM);
        assert_eq!(r.raw_cfg.dtype, DType::Bf16);
        assert_eq!(r.post_attn_add_rms_norm.hidden, HIDDEN_DIM);
        assert_eq!(r.post_attn_add_rms_norm.dtype, DType::Bf16);
        assert_eq!(r.post_attn_add_rms_norm.backends, vec!["vllm_cuda"]);
        assert_eq!(r.gate_up_proj.n, 24576);
        assert_eq!(r.gate_up_proj.k, 6144);
        assert_eq!(r.gate_up_proj.dtype, DType::Bf16);
        assert_eq!(r.gate_up_proj.backends, vec!["torch"]);
        assert_eq!(r.silu_and_mul.input_bytes_per_token, 49152);
        assert_eq!(r.silu_and_mul.output_bytes_per_token, 24576);
        assert_eq!(r.silu_and_mul.backends, vec!["triton"]);
        assert_eq!(r.down_proj.n, 6144);
        assert_eq!(r.down_proj.k, 12288);
        assert_eq!(r.down_proj.dtype, DType::Bf16);
        assert_eq!(r.down_proj.backends, vec!["torch"]);
        assert_eq!(r.raw_cfg.gpu_name, "NVIDIA H200");
    }

    #[test]
    fn input_shape_and_zero_case_feed_all_four_leaves_faithfully() {
        let input = Glm52DenseFfnLocalWorkletInput { batch_tokens: 73 };
        let work = work_inputs(input.batch_tokens);
        assert_eq!(work.post_attn_add_rms_norm.m, 73);
        assert_eq!(work.gate_up_proj.m, 73);
        assert_eq!(work.silu_and_mul.num_tokens, 73);
        assert_eq!(work.down_proj.m, 73);

        let zero = work_inputs(0);
        assert_eq!(zero.post_attn_add_rms_norm.m, 0);
        assert_eq!(zero.gate_up_proj.m, 0);
        assert_eq!(zero.silu_and_mul.num_tokens, 0);
        assert_eq!(zero.down_proj.m, 0);
        assert_eq!(Glm52DenseFfnLocalWorkletInput::default().batch_tokens, 0);
    }

    #[test]
    fn unsupported_glm_identity_fails_during_pure_resolution() {
        let mut bad_hidden = cfg();
        bad_hidden.hidden_dim = 4096.into();
        assert!(
            validate_config(&bad_hidden)
                .unwrap_err()
                .contains("hidden_dim must be 6144")
        );

        let mut bad_intermediate = cfg();
        bad_intermediate.intermediate_dim = 14336.into();
        assert!(
            validate_config(&bad_intermediate)
                .unwrap_err()
                .contains("intermediate_dim must be 12288")
        );

        let mut bad_dtype = cfg();
        bad_dtype.dtype = DType::Fp16;
        assert!(
            validate_config(&bad_dtype)
                .unwrap_err()
                .contains("dtype must be bf16")
        );
    }

    #[test]
    fn checked_dimension_and_byte_math_rejects_overflow() {
        assert_eq!(checked_product("gate", &[2, 12288]).unwrap(), 24576);
        assert_eq!(checked_product("input", &[2, 12288, 2]).unwrap(), 49152);
        assert_eq!(checked_product("output", &[12288, 2]).unwrap(), 24576);
        assert!(
            checked_product("overflow", &[u32::MAX, 2])
                .unwrap_err()
                .contains("overflows u32")
        );
    }

    #[test]
    fn label_documents_self_completing_local_boundary() {
        let label = worklet_label("layer.dense_ffn", &cfg());
        assert!(label.contains("Glm52DenseFfnLocalWorklet"));
        assert!(label.contains("local (1 GPU)"));
        assert!(label.contains("hidden=hidden_dim"));
        assert!(label.contains("intermediate=intermediate_dim"));
        for forbidden in ["tp=", "allreduce", "collective"] {
            assert!(!label.contains(forbidden));
        }
    }
}
