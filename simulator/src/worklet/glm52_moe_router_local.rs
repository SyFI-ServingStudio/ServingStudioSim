//! GLM-5.2 local sparse-MoE router worklet.
//!
//! This single-GPU section preserves the checkpoint's FP32 router semantics:
//! BF16 activations are cast to FP32 before the router GEMM, then sigmoid,
//! correction bias, grouped top-8 selection, normalization, and 5/2 scaling
//! produce routed weights and expert indices. VibeSim has no measured FP32
//! `single_gemm` backend for this shape, so `router_gemm_bf16_proxy` is an
//! explicitly labeled BF16 timing proxy. It is not production-exact.
//!
//! Dispatch, routed/shared expert computation, combine, and communication are
//! subsequent sections and are deliberately absent here.

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
const NUM_EXPERTS: u32 = 256;
const TOP_K: u32 = 8;
const N_GROUP: u32 = 1;
const TOPK_GROUP: u32 = 1;
const ROUTED_SCALING_NUMERATOR: u32 = 5;
const ROUTED_SCALING_DENOMINATOR: u32 = 2;
const INT32_BYTES: u32 = 4;

#[cfg(test)]
const SOURCE_ORDER: [&str; 4] = [
    "post_attn_add_rms_norm",
    "router_fp32_cast",
    "router_gemm_bf16_proxy",
    "router_select",
];

/// Raw GLM-5.2 router identity. Semantic and timing-proxy dtypes remain
/// separate so the proxy cannot be mistaken for the checkpoint contract.
#[derive(Clone, Debug)]
pub struct Glm52MoeRouterLocalWorkletConfig {
    pub residual_norm_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub proxy_gemm_backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    pub num_experts: Dim,
    pub top_k: u32,
    pub n_group: u32,
    pub topk_group: u32,
    pub base_dtype: DType,
    pub router_semantic_dtype: DType,
    pub proxy_gemm_dtype: DType,
    pub index_dtype: String,
    pub scoring_func: String,
    pub topk_method: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_numerator: u32,
    pub routed_scaling_denominator: u32,
}

/// Pure resolved data with all four measured child configs baked.
#[derive(Clone, Debug)]
pub struct Glm52MoeRouterLocalWorkletResolved {
    pub raw_cfg: Glm52MoeRouterLocalWorkletConfig,
    pub post_attn_add_rms_norm: ResidualRmsNormKernelConfig,
    pub router_fp32_cast: ElementwiseKernelConfig,
    pub router_gemm_bf16_proxy: SingleGemmKernelConfig,
    pub router_select: ElementwiseKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Glm52MoeRouterLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct Glm52MoeRouterLocalWorklet {
    pub name: String,
    pub post_attn_add_rms_norm: Op<ResidualRmsNormKernel>,
    pub router_fp32_cast: Op<ElementwiseKernel>,
    pub router_gemm_bf16_proxy: Op<SingleGemmKernel>,
    pub router_select: Op<ElementwiseKernel>,
    resolved: Glm52MoeRouterLocalWorkletResolved,
}

impl Glm52MoeRouterLocalWorklet {
    /// Resolve the one supported GLM-5.2 router identity without touching a
    /// bridge, GPU, cache, or `Arc`.
    pub fn resolve_config(
        cfg: &Glm52MoeRouterLocalWorkletConfig,
    ) -> Glm52MoeRouterLocalWorkletResolved {
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid Glm52MoeRouterLocalWorkletConfig: {reason}"));

        let cast_input_bytes = checked_product(
            "router_fp32_cast.input_bytes_per_token",
            &[cfg.hidden_dim.get(), cfg.base_dtype.size_bytes()],
        )
        .expect("validated GLM-5.2 router cast input byte rate must fit u32");
        let cast_output_bytes = checked_product(
            "router_fp32_cast.output_bytes_per_token",
            &[cfg.hidden_dim.get(), cfg.router_semantic_dtype.size_bytes()],
        )
        .expect("validated GLM-5.2 router cast output byte rate must fit u32");
        let select_input_bytes = checked_product(
            "router_select.input_bytes_per_token",
            &[
                2,
                cfg.num_experts.get(),
                cfg.router_semantic_dtype.size_bytes(),
            ],
        )
        .expect("validated GLM-5.2 router-select input byte rate must fit u32");
        let select_output_width = cfg
            .router_semantic_dtype
            .size_bytes()
            .checked_add(INT32_BYTES)
            .expect("validated GLM-5.2 router-select output width must fit u32");
        let select_output_bytes = checked_product(
            "router_select.output_bytes_per_token",
            &[cfg.top_k, select_output_width],
        )
        .expect("validated GLM-5.2 router-select output byte rate must fit u32");

        Glm52MoeRouterLocalWorkletResolved {
            post_attn_add_rms_norm: ResidualRmsNormKernelConfig {
                backends: cfg.residual_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden_dim.clone(),
                dtype: cfg.base_dtype,
            },
            router_fp32_cast: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: cast_input_bytes.into(),
                output_bytes_per_token: cast_output_bytes.into(),
            },
            router_gemm_bf16_proxy: SingleGemmKernelConfig {
                backends: cfg.proxy_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.num_experts.clone(),
                k: cfg.hidden_dim.clone(),
                dtype: cfg.proxy_gemm_dtype,
            },
            router_select: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: select_input_bytes.into(),
                output_bytes_per_token: select_output_bytes.into(),
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Glm52MoeRouterLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let post_attn_add_rms_norm = build_atomic(
            &name,
            "post_attn_add_rms_norm",
            resolved.post_attn_add_rms_norm.clone(),
            ResidualRmsNormKernel::build,
            bridge,
        )?;
        let router_fp32_cast = build_atomic(
            &name,
            "router_fp32_cast",
            resolved.router_fp32_cast.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        let router_gemm_bf16_proxy = build_atomic(
            &name,
            "router_gemm_bf16_proxy",
            resolved.router_gemm_bf16_proxy.clone(),
            SingleGemmKernel::build,
            bridge,
        )?;
        let router_select = build_atomic(
            &name,
            "router_select",
            resolved.router_select.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;

        Ok(Self {
            name,
            post_attn_add_rms_norm,
            router_fp32_cast,
            router_gemm_bf16_proxy,
            router_select,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: worklet_label(&self.name, &self.resolved.raw_cfg),
            child: Box::new(CostNode::Sum(vec![
                self.post_attn_add_rms_norm.compile(builder),
                self.router_fp32_cast.compile(builder),
                self.router_gemm_bf16_proxy.compile(builder),
                self.router_select.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Glm52MoeRouterLocalWorkletInput, ev: &mut Evaluator) {
        let work = work_inputs(input.batch_tokens);
        let zero = input.batch_tokens == 0;

        eval_atomic_or_zero(
            &self.post_attn_add_rms_norm,
            work.post_attn_add_rms_norm,
            zero,
            ev,
        );
        eval_atomic_or_zero(&self.router_fp32_cast, work.router_fp32_cast, zero, ev);
        eval_atomic_or_zero(
            &self.router_gemm_bf16_proxy,
            work.router_gemm_bf16_proxy,
            zero,
            ev,
        );
        eval_atomic_or_zero(&self.router_select, work.router_select, zero, ev);
    }
}

struct WorkInputs {
    post_attn_add_rms_norm: ResidualRmsNormKernelInput,
    router_fp32_cast: ElementwiseKernelInput,
    router_gemm_bf16_proxy: SingleGemmKernelInput,
    router_select: ElementwiseKernelInput,
}

fn validate_config(cfg: &Glm52MoeRouterLocalWorkletConfig) -> Result<(), String> {
    if cfg.routed_scaling_denominator == 0 {
        return Err("routed_scaling_denominator must be nonzero".to_string());
    }
    for (name, actual, required) in [
        ("hidden_dim", cfg.hidden_dim.get(), HIDDEN_DIM),
        ("num_experts", cfg.num_experts.get(), NUM_EXPERTS),
        ("top_k", cfg.top_k, TOP_K),
        ("n_group", cfg.n_group, N_GROUP),
        ("topk_group", cfg.topk_group, TOPK_GROUP),
        (
            "routed_scaling_numerator",
            cfg.routed_scaling_numerator,
            ROUTED_SCALING_NUMERATOR,
        ),
        (
            "routed_scaling_denominator",
            cfg.routed_scaling_denominator,
            ROUTED_SCALING_DENOMINATOR,
        ),
    ] {
        if actual != required {
            return Err(format!("{name} must be {required}, got {actual}"));
        }
    }
    for (name, actual, required) in [
        ("base_dtype", cfg.base_dtype, DType::Bf16),
        (
            "router_semantic_dtype",
            cfg.router_semantic_dtype,
            DType::Fp32,
        ),
        ("proxy_gemm_dtype", cfg.proxy_gemm_dtype, DType::Bf16),
    ] {
        if actual != required {
            return Err(format!(
                "{name} must be {}, got {}",
                required.as_str(),
                actual.as_str()
            ));
        }
    }
    if cfg.index_dtype != "int32" {
        return Err(format!(
            "index_dtype must be int32, got {:?}",
            cfg.index_dtype
        ));
    }
    if cfg.scoring_func != "sigmoid" {
        return Err(format!(
            "scoring_func must be sigmoid, got {:?}",
            cfg.scoring_func
        ));
    }
    if cfg.topk_method != "noaux_tc" {
        return Err(format!(
            "topk_method must be noaux_tc, got {:?}",
            cfg.topk_method
        ));
    }
    if !cfg.norm_topk_prob {
        return Err("norm_topk_prob must be enabled".to_string());
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
        router_fp32_cast: ElementwiseKernelInput {
            num_tokens: batch_tokens,
        },
        router_gemm_bf16_proxy: SingleGemmKernelInput { m: batch_tokens },
        router_select: ElementwiseKernelInput {
            num_tokens: batch_tokens,
        },
    }
}

fn worklet_label(name: &str, cfg: &Glm52MoeRouterLocalWorkletConfig) -> String {
    format!(
        "{name} (Glm52MoeRouterLocalWorklet) [local (1 GPU); router_semantic_dtype={}; timing_proxy_dtype={}; proxy=not-production-exact]",
        cfg.router_semantic_dtype.as_str(),
        cfg.proxy_gemm_dtype.as_str()
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

    fn cfg() -> Glm52MoeRouterLocalWorkletConfig {
        Glm52MoeRouterLocalWorkletConfig {
            residual_norm_backends: vec!["vllm_cuda"],
            elementwise_backends: vec!["triton"],
            proxy_gemm_backends: vec!["torch_linear"],
            gpu_name: "NVIDIA H200".to_string(),
            hidden_dim: Dim::param("hidden_dim", HIDDEN_DIM),
            num_experts: Dim::param("num_experts", NUM_EXPERTS),
            top_k: TOP_K,
            n_group: N_GROUP,
            topk_group: TOPK_GROUP,
            base_dtype: DType::Bf16,
            router_semantic_dtype: DType::Fp32,
            proxy_gemm_dtype: DType::Bf16,
            index_dtype: "int32".to_string(),
            scoring_func: "sigmoid".to_string(),
            topk_method: "noaux_tc".to_string(),
            norm_topk_prob: true,
            routed_scaling_numerator: ROUTED_SCALING_NUMERATOR,
            routed_scaling_denominator: ROUTED_SCALING_DENOMINATOR,
        }
    }

    #[test]
    fn source_order_is_exact_local_and_excludes_following_sections() {
        assert_eq!(
            SOURCE_ORDER,
            [
                "post_attn_add_rms_norm",
                "router_fp32_cast",
                "router_gemm_bf16_proxy",
                "router_select",
            ]
        );
        assert_eq!(SOURCE_ORDER.len(), 4);
        for forbidden in [
            "dispatch",
            "expert",
            "shared",
            "combine",
            "all_reduce",
            "all_to_all",
            "attention",
            "mtp",
        ] {
            assert!(!SOURCE_ORDER.iter().any(|child| child.contains(forbidden)));
        }
    }

    #[test]
    fn resolve_preserves_fp32_semantics_and_bakes_bf16_proxy_shapes() {
        let r = Glm52MoeRouterLocalWorklet::resolve_config(&cfg());

        assert_eq!(r.raw_cfg.hidden_dim, HIDDEN_DIM);
        assert_eq!(r.raw_cfg.num_experts, NUM_EXPERTS);
        assert_eq!(r.raw_cfg.top_k, TOP_K);
        assert_eq!(r.raw_cfg.n_group, N_GROUP);
        assert_eq!(r.raw_cfg.topk_group, TOPK_GROUP);
        assert_eq!(r.raw_cfg.router_semantic_dtype, DType::Fp32);
        assert_eq!(r.raw_cfg.proxy_gemm_dtype, DType::Bf16);
        assert_ne!(
            r.raw_cfg.router_semantic_dtype,
            r.router_gemm_bf16_proxy.dtype
        );

        assert_eq!(r.post_attn_add_rms_norm.hidden, 6144);
        assert_eq!(r.post_attn_add_rms_norm.dtype, DType::Bf16);
        assert_eq!(r.router_fp32_cast.input_bytes_per_token, 12288);
        assert_eq!(r.router_fp32_cast.output_bytes_per_token, 24576);
        assert_eq!(r.router_gemm_bf16_proxy.n, 256);
        assert_eq!(r.router_gemm_bf16_proxy.k, 6144);
        assert_eq!(r.router_gemm_bf16_proxy.dtype, DType::Bf16);
        assert_eq!(r.router_select.input_bytes_per_token, 2048);
        assert_eq!(r.router_select.output_bytes_per_token, 64);
    }

    #[test]
    fn semantic_modes_scaling_and_backend_roles_remain_explicit() {
        let r = Glm52MoeRouterLocalWorklet::resolve_config(&cfg());
        assert_eq!(r.raw_cfg.index_dtype, "int32");
        assert_eq!(r.raw_cfg.scoring_func, "sigmoid");
        assert_eq!(r.raw_cfg.topk_method, "noaux_tc");
        assert!(r.raw_cfg.norm_topk_prob);
        assert_eq!(r.raw_cfg.routed_scaling_numerator, 5);
        assert_eq!(r.raw_cfg.routed_scaling_denominator, 2);
        assert_eq!(r.post_attn_add_rms_norm.backends, vec!["vllm_cuda"]);
        assert_eq!(r.router_fp32_cast.backends, vec!["triton"]);
        assert_eq!(r.router_select.backends, vec!["triton"]);
        assert_eq!(r.router_gemm_bf16_proxy.backends, vec!["torch_linear"]);
        assert_eq!(r.raw_cfg.gpu_name, "NVIDIA H200");
    }

    #[test]
    fn input_and_zero_case_feed_all_four_leaves_faithfully() {
        let work = work_inputs(73);
        assert_eq!(work.post_attn_add_rms_norm.m, 73);
        assert_eq!(work.router_fp32_cast.num_tokens, 73);
        assert_eq!(work.router_gemm_bf16_proxy.m, 73);
        assert_eq!(work.router_select.num_tokens, 73);

        let zero = work_inputs(0);
        assert_eq!(zero.post_attn_add_rms_norm.m, 0);
        assert_eq!(zero.router_fp32_cast.num_tokens, 0);
        assert_eq!(zero.router_gemm_bf16_proxy.m, 0);
        assert_eq!(zero.router_select.num_tokens, 0);
        assert_eq!(Glm52MoeRouterLocalWorkletInput::default().batch_tokens, 0);
    }

    #[test]
    fn label_discloses_semantic_and_proxy_dtypes_and_approximation() {
        let label = worklet_label("layer.moe_router", &cfg());
        assert!(label.contains("Glm52MoeRouterLocalWorklet"));
        assert!(label.contains("local (1 GPU)"));
        assert!(label.contains("router_semantic_dtype=fp32"));
        assert!(label.contains("timing_proxy_dtype=bf16"));
        assert!(label.contains("proxy=not-production-exact"));
    }

    #[test]
    fn checked_dimension_and_byte_math_rejects_overflow() {
        assert_eq!(checked_product("cast input", &[6144, 2]).unwrap(), 12288);
        assert_eq!(checked_product("cast output", &[6144, 4]).unwrap(), 24576);
        assert_eq!(checked_product("select input", &[2, 256, 4]).unwrap(), 2048);
        assert_eq!(checked_product("select output", &[8, 8]).unwrap(), 64);
        assert!(
            checked_product("overflow", &[u32::MAX, 2])
                .unwrap_err()
                .contains("overflows u32")
        );
    }

    #[test]
    fn every_unsupported_static_identity_fails_clearly() {
        let mut invalid = Vec::new();

        let mut value = cfg();
        value.hidden_dim = 4096.into();
        invalid.push(value);
        let mut value = cfg();
        value.num_experts = 128.into();
        invalid.push(value);
        let mut value = cfg();
        value.top_k = 4;
        invalid.push(value);
        let mut value = cfg();
        value.n_group = 2;
        invalid.push(value);
        let mut value = cfg();
        value.topk_group = 2;
        invalid.push(value);
        let mut value = cfg();
        value.base_dtype = DType::Fp32;
        invalid.push(value);
        let mut value = cfg();
        value.router_semantic_dtype = DType::Bf16;
        invalid.push(value);
        let mut value = cfg();
        value.proxy_gemm_dtype = DType::Fp32;
        invalid.push(value);
        let mut value = cfg();
        value.index_dtype = "int64".to_string();
        invalid.push(value);
        let mut value = cfg();
        value.scoring_func = "softmax".to_string();
        invalid.push(value);
        let mut value = cfg();
        value.topk_method = "greedy".to_string();
        invalid.push(value);
        let mut value = cfg();
        value.norm_topk_prob = false;
        invalid.push(value);
        let mut value = cfg();
        value.routed_scaling_numerator = 4;
        invalid.push(value);
        let mut value = cfg();
        value.routed_scaling_denominator = 3;
        invalid.push(value);
        let mut value = cfg();
        value.routed_scaling_denominator = 0;
        invalid.push(value);

        for value in invalid {
            assert!(validate_config(&value).is_err());
        }
        let mut zero_denominator = cfg();
        zero_denominator.routed_scaling_denominator = 0;
        assert!(
            validate_config(&zero_denominator)
                .unwrap_err()
                .contains("must be nonzero")
        );
    }
}
