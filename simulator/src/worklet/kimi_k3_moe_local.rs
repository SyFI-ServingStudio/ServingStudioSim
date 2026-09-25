//! Kimi-K3 rank-local shared/routed MoE worklet.
//!
//! SGLang merges the shared gate/up, router, and latent-down front weights into
//! one BF16 projection. Routing and both routed expert GEMMs remain inside the
//! registered MXFP4 fused-MoE leaf. There is deliberately no standalone router
//! leaf in this recipe.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, GemmFp32OutputKernel,
    GemmFp32OutputKernelConfig, GemmFp32OutputKernelInput, K3Add3PrefillKernel,
    K3Add3PrefillKernelConfig, K3Add3PrefillKernelInput, K3SituAndMulPrefillKernel,
    K3SituAndMulPrefillKernelConfig, K3SituAndMulPrefillKernelInput, Mxfp4FusedMoeKernel,
    Mxfp4FusedMoeKernelConfig, Mxfp4FusedMoeKernelInput, RmsNormKernel, RmsNormKernelConfig,
    RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::routing::RoutingDistribution;
use crate::timing::{CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::kimi_k3_common::{
    build_atomic, eval_atomic_or_zero, zero_moe_alltoall, ZeroMoeAlltoallProbe,
};

const HIDDEN: u32 = 7_168;
const LATENT_HIDDEN: u32 = 3_584;
const NUM_EXPERTS: u32 = 896;
const MOE_INTERMEDIATE: u32 = 3_072;
const TOP_K: u32 = 16;
const SHARED_INTERMEDIATE: u32 = 6_144;

// The realistic rank-1 payloads expose timing/kernel tables but no expert-count
// histogram. The analytic fallback is a uniform expectation over 112 local
// experts; independent weighted top-2 sampling below produces the expected
// Poisson-like realized load for 128 x 2 = 256 assignments. A future alignment
// payload with measured counts can replace this source without changing the
// production global routing path.

#[cfg(test)]
const SOURCE_ORDER: [&str; 14] = [
    "merged_front",
    "merged_front_prefill",
    "shared_gate_up_activation",
    "shared_gate_up_activation_prefill",
    "shared_down",
    "shared_down_prefill",
    "mxfp4_fused_moe",
    "mxfp4_fused_moe_prefill",
    "routed_norm",
    "latent_up",
    "latent_up_prefill",
    "add3",
    "add3_prefill",
    "ep_alltoall_zero",
];

#[derive(Clone, Debug)]
pub struct KimiK3MoeLocalWorkletConfig {
    pub gpu_name: String,
    pub hidden: Dim,
    pub latent_hidden: Dim,
    pub num_experts: Dim,
    /// Expert width passed to the rank-local FlashInfer call. The production
    /// graph routes over all 896 experts; the rank-1 alignment driver shrinks
    /// that call to its 112 local experts and keeps EP as metadata.
    pub routing_experts: Dim,
    pub local_experts: Dim,
    pub moe_intermediate: Dim,
    pub shared_intermediate: Dim,
    pub top_k: u32,
    /// Measured alignment probabilities/counts, when a payload supplied them.
    /// `None` selects the analytic rank-local fallback.
    pub routing_histogram: Option<Vec<f32>>,
    pub ep_size: u16,
    pub dtype: DType,
    pub gemm_backends: Vec<&'static str>,
    pub prefill_gemm_backends: Vec<&'static str>,
    pub prefill_bf16_gemm_backends: Vec<&'static str>,
    pub prefill_activation_backends: Vec<&'static str>,
    pub prefill_add3_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub rms_norm_backends: Vec<&'static str>,
    pub moe_backends: Vec<&'static str>,
    pub prefill_moe_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct KimiK3MoeLocalWorkletResolved {
    pub raw_cfg: KimiK3MoeLocalWorkletConfig,
    pub merged_front: SingleGemmKernelConfig,
    pub merged_front_prefill: GemmFp32OutputKernelConfig,
    pub shared_gate_up_activation: ElementwiseKernelConfig,
    pub shared_gate_up_activation_prefill: K3SituAndMulPrefillKernelConfig,
    pub shared_down: SingleGemmKernelConfig,
    pub shared_down_prefill: SingleGemmKernelConfig,
    pub mxfp4_fused_moe: Mxfp4FusedMoeKernelConfig,
    pub mxfp4_fused_moe_prefill: Mxfp4FusedMoeKernelConfig,
    pub routed_norm: RmsNormKernelConfig,
    pub latent_up: SingleGemmKernelConfig,
    pub latent_up_prefill: SingleGemmKernelConfig,
    pub add3: ElementwiseKernelConfig,
    pub add3_prefill: K3Add3PrefillKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct KimiK3MoeLocalWorkletInput {
    pub num_tokens: u32,
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
}

pub struct KimiK3MoeLocalWorklet {
    pub name: String,
    pub merged_front: Op<SingleGemmKernel>,
    pub merged_front_prefill: Op<GemmFp32OutputKernel>,
    pub shared_gate_up_activation: Op<ElementwiseKernel>,
    pub shared_gate_up_activation_prefill: Op<K3SituAndMulPrefillKernel>,
    pub shared_down: Op<SingleGemmKernel>,
    pub shared_down_prefill: Op<SingleGemmKernel>,
    pub mxfp4_fused_moe: Op<Mxfp4FusedMoeKernel>,
    pub mxfp4_fused_moe_prefill: Op<Mxfp4FusedMoeKernel>,
    pub routed_norm: Op<RmsNormKernel>,
    pub latent_up: Op<SingleGemmKernel>,
    pub latent_up_prefill: Op<SingleGemmKernel>,
    pub add3: Op<ElementwiseKernel>,
    pub add3_prefill: Op<K3Add3PrefillKernel>,
    pub ep_alltoall_zero: Op<ZeroMoeAlltoallProbe>,
    resolved: KimiK3MoeLocalWorkletResolved,
}

impl KimiK3MoeLocalWorklet {
    pub fn resolve_config(cfg: &KimiK3MoeLocalWorkletConfig) -> KimiK3MoeLocalWorkletResolved {
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid KimiK3MoeLocalWorkletConfig: {reason}"));
        let merged_front_n =
            2 * cfg.shared_intermediate.get() + cfg.num_experts.get() + cfg.latent_hidden.get();
        let merged_front_prefill_n =
            2 * cfg.shared_intermediate.get() + cfg.routing_experts.get() + cfg.latent_hidden.get();
        let measured_routing = cfg.routing_histogram.is_some();
        let analytic_rank_local_routing = !measured_routing
            && cfg.routing_experts.get() == cfg.local_experts.get()
            && cfg.top_k < TOP_K;
        let ppm = cfg
            .routing_histogram
            .as_deref()
            .map(|histogram| {
                assert_eq!(histogram.len(), cfg.routing_experts.get() as usize);
                RoutingDistribution::from_profile(histogram).ppm().to_vec()
            })
            .unwrap_or_else(|| {
                if analytic_rank_local_routing {
                    analytic_rank1_ppm(cfg.routing_experts.get())
                } else {
                    uniform_ppm(cfg.routing_experts.get())
                }
            });
        KimiK3MoeLocalWorkletResolved {
            merged_front: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: merged_front_n.into(),
                k: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            merged_front_prefill: GemmFp32OutputKernelConfig {
                backends: cfg.prefill_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: merged_front_prefill_n.into(),
                k: cfg.hidden.clone(),
                input_dtype: cfg.dtype,
            },
            shared_gate_up_activation: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: (2 * cfg.shared_intermediate.get() * 2).into(),
                output_bytes_per_token: (cfg.shared_intermediate.get() * 2).into(),
            },
            shared_gate_up_activation_prefill: K3SituAndMulPrefillKernelConfig {
                backends: cfg.prefill_activation_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.shared_intermediate.clone(),
                input_dtype: DType::Fp32,
                output_dtype: cfg.dtype,
                beta: 4,
                linear_beta: 25,
            },
            shared_down: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden.clone(),
                k: cfg.shared_intermediate.clone(),
                dtype: cfg.dtype,
            },
            shared_down_prefill: SingleGemmKernelConfig {
                backends: cfg.prefill_bf16_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden.clone(),
                k: cfg.shared_intermediate.clone(),
                dtype: cfg.dtype,
            },
            mxfp4_fused_moe: Mxfp4FusedMoeKernelConfig {
                backends: cfg.moe_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.latent_hidden.clone(),
                intermediate_size: cfg.moe_intermediate.clone(),
                num_experts: cfg.routing_experts.clone(),
                num_local_experts: cfg.local_experts.clone(),
                top_k: cfg.top_k,
                input_dtype: cfg.dtype,
                weight_format: "mxfp4_e2m1_ue8m0".to_string(),
                group_size: 32,
                routing_method: "deepseek_v3_sigmoid".to_string(),
                activation: "situ".to_string(),
                n_group: 1,
                topk_group: 1,
                routed_scaling_numerator: 1,
                routed_scaling_denominator: 1,
                gemm1_alpha: 4,
                gemm1_clamp_limit: 25,
                layerwise_global_ppm: vec![ppm.clone()],
                folded_rank_position: 0,
                stochastic_routing: analytic_rank_local_routing || measured_routing,
            },
            mxfp4_fused_moe_prefill: Mxfp4FusedMoeKernelConfig {
                backends: cfg.prefill_moe_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.latent_hidden.clone(),
                intermediate_size: cfg.moe_intermediate.clone(),
                num_experts: cfg.routing_experts.clone(),
                num_local_experts: cfg.local_experts.clone(),
                top_k: cfg.top_k,
                input_dtype: cfg.dtype,
                weight_format: "mxfp4_e2m1_ue8m0".to_string(),
                group_size: 32,
                routing_method: "deepseek_v3_sigmoid".to_string(),
                activation: "situ".to_string(),
                n_group: 1,
                topk_group: 1,
                routed_scaling_numerator: 1,
                routed_scaling_denominator: 1,
                gemm1_alpha: 4,
                gemm1_clamp_limit: 25,
                layerwise_global_ppm: vec![ppm],
                folded_rank_position: 0,
                stochastic_routing: analytic_rank_local_routing || measured_routing,
            },
            routed_norm: RmsNormKernelConfig {
                backends: cfg.rms_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.latent_hidden.clone(),
                dtype: cfg.dtype,
            },
            latent_up: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden.clone(),
                k: cfg.latent_hidden.clone(),
                dtype: cfg.dtype,
            },
            latent_up_prefill: SingleGemmKernelConfig {
                backends: cfg.prefill_bf16_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden.clone(),
                k: cfg.latent_hidden.clone(),
                dtype: cfg.dtype,
            },
            add3: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: (3 * cfg.hidden.get() * 2).into(),
                output_bytes_per_token: (cfg.hidden.get() * 2).into(),
            },
            add3_prefill: K3Add3PrefillKernelConfig {
                backends: cfg.prefill_add3_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: KimiK3MoeLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, crate::timing::BuildError> {
        let ep_alltoall_zero = zero_moe_alltoall(
            format!("{name}.ep_alltoall_zero"),
            serde_json::json!({
                "gpu_name": resolved.raw_cfg.gpu_name,
                "ep_size": resolved.raw_cfg.ep_size,
                "zero_time": true,
            }),
        );
        Ok(Self {
            merged_front: build_atomic(
                &name,
                "merged_front",
                resolved.merged_front.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            merged_front_prefill: build_atomic(
                &name,
                "merged_front_prefill",
                resolved.merged_front_prefill.clone(),
                GemmFp32OutputKernel::build,
                bridge,
            )?,
            shared_gate_up_activation: build_atomic(
                &name,
                "shared_gate_up_activation",
                resolved.shared_gate_up_activation.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            shared_gate_up_activation_prefill: build_atomic(
                &name,
                "shared_gate_up_activation_prefill",
                resolved.shared_gate_up_activation_prefill.clone(),
                K3SituAndMulPrefillKernel::build,
                bridge,
            )?,
            shared_down: build_atomic(
                &name,
                "shared_down",
                resolved.shared_down.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            shared_down_prefill: build_atomic(
                &name,
                "shared_down_prefill",
                resolved.shared_down_prefill.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            mxfp4_fused_moe: build_atomic(
                &name,
                "mxfp4_fused_moe",
                resolved.mxfp4_fused_moe.clone(),
                Mxfp4FusedMoeKernel::build,
                bridge,
            )?,
            mxfp4_fused_moe_prefill: build_atomic(
                &name,
                "mxfp4_fused_moe_prefill",
                resolved.mxfp4_fused_moe_prefill.clone(),
                Mxfp4FusedMoeKernel::build,
                bridge,
            )?,
            routed_norm: build_atomic(
                &name,
                "routed_norm",
                resolved.routed_norm.clone(),
                RmsNormKernel::build,
                bridge,
            )?,
            latent_up: build_atomic(
                &name,
                "latent_up",
                resolved.latent_up.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            latent_up_prefill: build_atomic(
                &name,
                "latent_up_prefill",
                resolved.latent_up_prefill.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            add3: build_atomic(
                &name,
                "add3",
                resolved.add3.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            add3_prefill: build_atomic(
                &name,
                "add3_prefill",
                resolved.add3_prefill.clone(),
                K3Add3PrefillKernel::build,
                bridge,
            )?,
            ep_alltoall_zero,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (KimiK3MoeLocalWorklet) [EP{}; local_experts={}]",
                self.name, self.resolved.raw_cfg.ep_size, self.resolved.raw_cfg.local_experts
            ),
            child: Box::new(CostNode::Sum(vec![
                self.merged_front.compile(builder),
                self.merged_front_prefill.compile(builder),
                self.shared_gate_up_activation.compile(builder),
                self.shared_gate_up_activation_prefill.compile(builder),
                self.shared_down.compile(builder),
                self.shared_down_prefill.compile(builder),
                self.mxfp4_fused_moe.compile(builder),
                self.mxfp4_fused_moe_prefill.compile(builder),
                self.routed_norm.compile(builder),
                self.latent_up.compile(builder),
                self.latent_up_prefill.compile(builder),
                self.add3.compile(builder),
                self.add3_prefill.compile(builder),
                self.ep_alltoall_zero.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &KimiK3MoeLocalWorkletInput, evaluator: &mut Evaluator) {
        let is_prefill = !input.prefill_chunk_pairs.is_empty();
        let rows = phase_token_count(input.num_tokens, &input.prefill_chunk_pairs);
        eval_atomic_or_zero(
            &self.merged_front,
            SingleGemmKernelInput { m: rows },
            rows == 0 || is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.merged_front_prefill,
            GemmFp32OutputKernelInput { m: rows },
            rows == 0 || !is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.shared_gate_up_activation,
            ElementwiseKernelInput { num_tokens: rows },
            rows == 0 || is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.shared_gate_up_activation_prefill,
            K3SituAndMulPrefillKernelInput { num_tokens: rows },
            rows == 0 || !is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.shared_down,
            SingleGemmKernelInput { m: rows },
            rows == 0 || is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.shared_down_prefill,
            SingleGemmKernelInput { m: rows },
            rows == 0 || !is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.mxfp4_fused_moe,
            Mxfp4FusedMoeKernelInput { num_tokens: rows },
            rows == 0 || is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.mxfp4_fused_moe_prefill,
            Mxfp4FusedMoeKernelInput { num_tokens: rows },
            rows == 0 || !is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.routed_norm,
            RmsNormKernelInput { m: rows },
            rows == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.latent_up,
            SingleGemmKernelInput { m: rows },
            rows == 0 || is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.latent_up_prefill,
            SingleGemmKernelInput { m: rows },
            rows == 0 || !is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.add3,
            ElementwiseKernelInput { num_tokens: rows },
            rows == 0 || is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.add3_prefill,
            K3Add3PrefillKernelInput { num_tokens: rows },
            rows == 0 || !is_prefill,
            evaluator,
        );
        self.ep_alltoall_zero.eval(
            &crate::timing::kernels::MoeAlltoallKernelInput {
                max_send_rows: rows,
                max_recv_rows: rows,
            },
            evaluator,
        );
    }
}

fn phase_token_count(batch_tokens: u32, pairs: &[(u32, u32)]) -> u32 {
    if pairs.is_empty() {
        return batch_tokens;
    }
    pairs
        .iter()
        .try_fold(0_u32, |total, &(_prefix, append)| total.checked_add(append))
        .expect("Kimi-K3 MoE prefill token count must fit u32")
}

fn uniform_ppm(num_experts: u32) -> Vec<u32> {
    let base = 1_000_000 / num_experts;
    let remainder = 1_000_000 % num_experts;
    (0..num_experts)
        .map(|index| base + u32::from(index < remainder))
        .collect()
}

fn analytic_rank1_ppm(num_experts: u32) -> Vec<u32> {
    assert_eq!(num_experts, 112);
    RoutingDistribution::uniform(num_experts).ppm().to_vec()
}

fn validate_config(cfg: &KimiK3MoeLocalWorkletConfig) -> Result<(), String> {
    for (name, actual, required) in [
        ("hidden", cfg.hidden.get(), HIDDEN),
        ("latent_hidden", cfg.latent_hidden.get(), LATENT_HIDDEN),
        ("num_experts", cfg.num_experts.get(), NUM_EXPERTS),
        (
            "moe_intermediate",
            cfg.moe_intermediate.get(),
            MOE_INTERMEDIATE,
        ),
        (
            "shared_intermediate",
            cfg.shared_intermediate.get(),
            SHARED_INTERMEDIATE,
        ),
    ] {
        if actual != required {
            return Err(format!("{name} must be {required}, got {actual}"));
        }
    }
    if cfg.local_experts.get() == 0 || cfg.num_experts.get() % cfg.local_experts.get() != 0 {
        return Err("local_experts must be a positive divisor of num_experts".to_string());
    }
    if cfg.routing_experts != cfg.local_experts && cfg.routing_experts != cfg.num_experts {
        return Err("routing_experts must equal local_experts or num_experts".to_string());
    }
    if cfg.top_k == 0 || cfg.routing_experts.get() < cfg.top_k {
        return Err("routing_experts must be at least top_k".to_string());
    }
    if cfg.top_k > TOP_K {
        return Err(format!("top_k must not exceed {TOP_K}, got {}", cfg.top_k));
    }
    if cfg.routing_experts == cfg.num_experts && cfg.top_k != TOP_K {
        return Err(format!(
            "global routing must use top_k={TOP_K}, got {}",
            cfg.top_k
        ));
    }
    if cfg.dtype != DType::Bf16 {
        return Err("K3 MoE activations use bf16".to_string());
    }
    if cfg.prefill_gemm_backends.is_empty() {
        return Err("K3 MoE prefill requires a FP32-output GEMM backend".to_string());
    }
    if cfg.prefill_moe_backends.is_empty() {
        return Err("K3 MoE prefill requires an MXFP4 backend".to_string());
    }
    if cfg.prefill_activation_backends.is_empty() {
        return Err("K3 MoE prefill requires a SiTU activation backend".to_string());
    }
    if cfg.prefill_add3_backends.is_empty() {
        return Err("K3 MoE prefill requires an add3 backend".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> KimiK3MoeLocalWorkletConfig {
        KimiK3MoeLocalWorkletConfig {
            gpu_name: "NVIDIA B200".to_string(),
            hidden: HIDDEN.into(),
            latent_hidden: LATENT_HIDDEN.into(),
            num_experts: NUM_EXPERTS.into(),
            routing_experts: NUM_EXPERTS.into(),
            local_experts: 112.into(),
            moe_intermediate: MOE_INTERMEDIATE.into(),
            shared_intermediate: SHARED_INTERMEDIATE.into(),
            top_k: TOP_K,
            routing_histogram: None,
            ep_size: 8,
            dtype: DType::Bf16,
            gemm_backends: vec!["sglang_bf16_auto"],
            prefill_gemm_backends: vec!["sglang_k3_fp32_auto"],
            prefill_bf16_gemm_backends: vec!["sglang_k3_raw_bf16"],
            prefill_activation_backends: vec!["sglang_k3"],
            prefill_add3_backends: vec!["sglang_k3"],
            elementwise_backends: vec!["triton"],
            rms_norm_backends: vec!["flashinfer"],
            moe_backends: vec!["sglang_trtllm_mxfp4"],
            prefill_moe_backends: vec!["sglang_trtllm_mxfp4_prefill"],
        }
    }

    #[test]
    fn merged_front_and_expert_shapes_are_frozen() {
        assert_eq!(SOURCE_ORDER.len(), 14);
        let resolved = KimiK3MoeLocalWorklet::resolve_config(&config());
        assert_eq!(resolved.merged_front.n, 16_768);
        assert_eq!(resolved.merged_front.k, HIDDEN);
        assert_eq!(resolved.merged_front_prefill.n, 16_768);
        assert_eq!(resolved.merged_front_prefill.k, HIDDEN);
        assert_eq!(resolved.shared_down.k, 6_144);
        assert_eq!(resolved.shared_down.n, HIDDEN);
        assert_eq!(
            resolved.shared_gate_up_activation_prefill.hidden_size,
            6_144
        );
        assert_eq!(
            resolved.shared_gate_up_activation_prefill.backends,
            vec!["sglang_k3"]
        );
        assert_eq!(resolved.mxfp4_fused_moe.hidden_size, LATENT_HIDDEN);
        assert_eq!(resolved.mxfp4_fused_moe.intermediate_size, MOE_INTERMEDIATE);
        assert_eq!(resolved.mxfp4_fused_moe.num_experts, NUM_EXPERTS);
        assert_eq!(resolved.mxfp4_fused_moe.num_local_experts, 112);
        assert_eq!(resolved.mxfp4_fused_moe.top_k, TOP_K);
        assert!(!resolved.mxfp4_fused_moe.stochastic_routing);
        assert_eq!(
            resolved.mxfp4_fused_moe_prefill.backends,
            vec!["sglang_trtllm_mxfp4_prefill"]
        );
        assert_eq!(resolved.routed_norm.hidden, LATENT_HIDDEN);
        assert_eq!(resolved.latent_up.k, LATENT_HIDDEN);
        assert_eq!(resolved.latent_up.n, HIDDEN);
        assert_eq!(resolved.add3_prefill.hidden_size, HIDDEN);
    }

    #[test]
    fn uniform_routing_table_is_exactly_normalized() {
        let ppm = uniform_ppm(NUM_EXPERTS);
        assert_eq!(ppm.len(), NUM_EXPERTS as usize);
        assert_eq!(
            ppm.iter().map(|&value| u64::from(value)).sum::<u64>(),
            1_000_000
        );
    }

    #[test]
    fn rank1_analytic_routing_profile_is_normalized_and_poisson_like() {
        let ppm = analytic_rank1_ppm(112);
        assert_eq!(ppm.len(), 112);
        assert_eq!(
            ppm.iter().map(|&value| u64::from(value)).sum::<u64>(),
            1_000_000
        );
        assert!(ppm.iter().all(|&value| value > 0));
        let counts =
            crate::timing::routing::sample_random_topk_expert_counts(&ppm, 2, 128, 0xF01D_5EED);
        assert_eq!(counts.iter().sum::<u32>(), 256);
        assert!(counts.iter().min() < counts.iter().max());

        let mut cfg = config();
        cfg.routing_experts = 112.into();
        cfg.ep_size = 1;
        cfg.top_k = 2;
        let resolved = KimiK3MoeLocalWorklet::resolve_config(&cfg);
        assert!(resolved.mxfp4_fused_moe.stochastic_routing);
        assert_eq!(resolved.mxfp4_fused_moe.num_experts, 112);
        assert_eq!(resolved.mxfp4_fused_moe.top_k, 2);
        assert_eq!(resolved.merged_front_prefill.n, 15_984);
    }

    #[test]
    fn zero_tokens_still_have_a_stable_input_shape() {
        assert_eq!(KimiK3MoeLocalWorkletInput::default().num_tokens, 0);
        assert_eq!(
            KimiK3MoeLocalWorkletInput {
                num_tokens: 32,
                prefill_chunk_pairs: Vec::new(),
            }
            .num_tokens,
            32
        );
    }

    #[test]
    fn prefill_front_is_selected_by_chunk_pairs() {
        assert_eq!(phase_token_count(99, &[]), 99);
        assert_eq!(phase_token_count(99, &[(49_152, 16_384)]), 16_384);
    }
}
