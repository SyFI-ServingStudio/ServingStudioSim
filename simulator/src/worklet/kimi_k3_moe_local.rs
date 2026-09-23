//! Kimi-K3 rank-local shared/routed MoE worklet.
//!
//! SGLang merges the shared gate/up, router, and latent-down front weights into
//! one BF16 projection. Routing and both routed expert GEMMs remain inside the
//! registered MXFP4 fused-MoE leaf. There is deliberately no standalone router
//! leaf in this recipe.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, Mxfp4FusedMoeKernel,
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

// The rank-1 alignment driver uses the same seeded random-initialized K3 gate
// as the layer probe. These are its synchronized B=128 top-k hit counts,
// treated as a popularity profile so every simulator batch gets a reproducible
// realization with the same sparse expert load shape. Production EP8 keeps
// the existing 896-expert routing law below.
const K3_RANK1_DRIVER_B128_COUNTS: &[f32] = &[
    0.0, 27.0, 5.0, 0.0, 0.0, 40.0, 1.0, 124.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 102.0, 28.0, 128.0, 92.0, 0.0, 1.0, 15.0, 18.0, 10.0, 10.0, 0.0, 1.0, 111.0, 0.0,
    6.0, 0.0, 0.0, 49.0, 0.0, 123.0, 0.0, 0.0, 0.0, 0.0, 47.0, 0.0, 0.0, 0.0, 0.0, 0.0, 79.0, 1.0,
    1.0, 32.0, 17.0, 0.0, 0.0, 0.0, 126.0, 0.0, 0.0, 7.0, 0.0, 125.0, 6.0, 0.0, 0.0, 119.0, 0.0,
    0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 23.0, 0.0, 5.0, 66.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 14.0,
    0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 74.0, 0.0, 0.0, 128.0, 0.0, 0.0,
    88.0, 0.0, 126.0, 0.0, 0.0, 0.0, 4.0,
];

#[cfg(test)]
const SOURCE_ORDER: [&str; 8] = [
    "merged_front",
    "shared_gate_up_activation",
    "shared_down",
    "mxfp4_fused_moe",
    "routed_norm",
    "latent_up",
    "add3",
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
    pub ep_size: u16,
    pub dtype: DType,
    pub gemm_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub rms_norm_backends: Vec<&'static str>,
    pub moe_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct KimiK3MoeLocalWorkletResolved {
    pub raw_cfg: KimiK3MoeLocalWorkletConfig,
    pub merged_front: SingleGemmKernelConfig,
    pub shared_gate_up_activation: ElementwiseKernelConfig,
    pub shared_down: SingleGemmKernelConfig,
    pub mxfp4_fused_moe: Mxfp4FusedMoeKernelConfig,
    pub routed_norm: RmsNormKernelConfig,
    pub latent_up: SingleGemmKernelConfig,
    pub add3: ElementwiseKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct KimiK3MoeLocalWorkletInput {
    pub num_tokens: u32,
}

pub struct KimiK3MoeLocalWorklet {
    pub name: String,
    pub merged_front: Op<SingleGemmKernel>,
    pub shared_gate_up_activation: Op<ElementwiseKernel>,
    pub shared_down: Op<SingleGemmKernel>,
    pub mxfp4_fused_moe: Op<Mxfp4FusedMoeKernel>,
    pub routed_norm: Op<RmsNormKernel>,
    pub latent_up: Op<SingleGemmKernel>,
    pub add3: Op<ElementwiseKernel>,
    pub ep_alltoall_zero: Op<ZeroMoeAlltoallProbe>,
    resolved: KimiK3MoeLocalWorkletResolved,
}

impl KimiK3MoeLocalWorklet {
    pub fn resolve_config(cfg: &KimiK3MoeLocalWorkletConfig) -> KimiK3MoeLocalWorkletResolved {
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid KimiK3MoeLocalWorkletConfig: {reason}"));
        let merged_front_n =
            2 * cfg.shared_intermediate.get() + cfg.num_experts.get() + cfg.latent_hidden.get();
        let rank1_driver_profile = cfg.routing_experts.get() == cfg.local_experts.get();
        let ppm = if rank1_driver_profile {
            k3_rank1_driver_ppm()
        } else {
            uniform_ppm(cfg.routing_experts.get())
        };
        KimiK3MoeLocalWorkletResolved {
            merged_front: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: merged_front_n.into(),
                k: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            shared_gate_up_activation: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: (2 * cfg.shared_intermediate.get() * 2).into(),
                output_bytes_per_token: (cfg.shared_intermediate.get() * 2).into(),
            },
            shared_down: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
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
                layerwise_global_ppm: vec![ppm],
                folded_rank_position: 0,
                stochastic_routing: rank1_driver_profile,
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
            add3: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: (3 * cfg.hidden.get() * 2).into(),
                output_bytes_per_token: (cfg.hidden.get() * 2).into(),
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
            shared_gate_up_activation: build_atomic(
                &name,
                "shared_gate_up_activation",
                resolved.shared_gate_up_activation.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            shared_down: build_atomic(
                &name,
                "shared_down",
                resolved.shared_down.clone(),
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
            add3: build_atomic(
                &name,
                "add3",
                resolved.add3.clone(),
                ElementwiseKernel::build,
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
                self.shared_gate_up_activation.compile(builder),
                self.shared_down.compile(builder),
                self.mxfp4_fused_moe.compile(builder),
                self.routed_norm.compile(builder),
                self.latent_up.compile(builder),
                self.add3.compile(builder),
                self.ep_alltoall_zero.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &KimiK3MoeLocalWorkletInput, evaluator: &mut Evaluator) {
        let rows = input.num_tokens;
        eval_atomic_or_zero(
            &self.merged_front,
            SingleGemmKernelInput { m: rows },
            rows == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.shared_gate_up_activation,
            ElementwiseKernelInput { num_tokens: rows },
            rows == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.shared_down,
            SingleGemmKernelInput { m: rows },
            rows == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.mxfp4_fused_moe,
            Mxfp4FusedMoeKernelInput { num_tokens: rows },
            rows == 0,
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
            rows == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.add3,
            ElementwiseKernelInput { num_tokens: rows },
            rows == 0,
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

fn uniform_ppm(num_experts: u32) -> Vec<u32> {
    let base = 1_000_000 / num_experts;
    let remainder = 1_000_000 % num_experts;
    (0..num_experts)
        .map(|index| base + u32::from(index < remainder))
        .collect()
}

fn k3_rank1_driver_ppm() -> Vec<u32> {
    assert_eq!(K3_RANK1_DRIVER_B128_COUNTS.len(), 112);
    RoutingDistribution::from_profile(K3_RANK1_DRIVER_B128_COUNTS)
        .ppm()
        .to_vec()
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
    if cfg.routing_experts.get() < cfg.top_k {
        return Err("routing_experts must be at least top_k".to_string());
    }
    if cfg.top_k != TOP_K {
        return Err(format!("top_k must be {TOP_K}, got {}", cfg.top_k));
    }
    if cfg.dtype != DType::Bf16 {
        return Err("K3 MoE activations use bf16".to_string());
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
            ep_size: 8,
            dtype: DType::Bf16,
            gemm_backends: vec!["sglang_bf16_auto"],
            elementwise_backends: vec!["triton"],
            rms_norm_backends: vec!["flashinfer"],
            moe_backends: vec!["sglang_trtllm_mxfp4"],
        }
    }

    #[test]
    fn merged_front_and_expert_shapes_are_frozen() {
        assert_eq!(SOURCE_ORDER.len(), 8);
        let resolved = KimiK3MoeLocalWorklet::resolve_config(&config());
        assert_eq!(resolved.merged_front.n, 16_768);
        assert_eq!(resolved.merged_front.k, HIDDEN);
        assert_eq!(resolved.shared_down.k, 6_144);
        assert_eq!(resolved.shared_down.n, HIDDEN);
        assert_eq!(resolved.mxfp4_fused_moe.hidden_size, LATENT_HIDDEN);
        assert_eq!(resolved.mxfp4_fused_moe.intermediate_size, MOE_INTERMEDIATE);
        assert_eq!(resolved.mxfp4_fused_moe.num_experts, NUM_EXPERTS);
        assert_eq!(resolved.mxfp4_fused_moe.num_local_experts, 112);
        assert_eq!(resolved.mxfp4_fused_moe.top_k, TOP_K);
        assert!(!resolved.mxfp4_fused_moe.stochastic_routing);
        assert_eq!(resolved.routed_norm.hidden, LATENT_HIDDEN);
        assert_eq!(resolved.latent_up.k, LATENT_HIDDEN);
        assert_eq!(resolved.latent_up.n, HIDDEN);
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
    fn rank1_driver_routing_profile_is_normalized_and_topk_compatible() {
        let ppm = k3_rank1_driver_ppm();
        assert_eq!(ppm.len(), 112);
        assert_eq!(
            ppm.iter().map(|&value| u64::from(value)).sum::<u64>(),
            1_000_000
        );
        assert!(ppm.iter().filter(|&&value| value > 0).count() >= TOP_K as usize);

        let mut cfg = config();
        cfg.routing_experts = 112.into();
        cfg.ep_size = 1;
        let resolved = KimiK3MoeLocalWorklet::resolve_config(&cfg);
        assert!(resolved.mxfp4_fused_moe.stochastic_routing);
        assert_eq!(resolved.mxfp4_fused_moe.num_experts, 112);
    }

    #[test]
    fn zero_tokens_still_have_a_stable_input_shape() {
        assert_eq!(KimiK3MoeLocalWorkletInput::default().num_tokens, 0);
        assert_eq!(KimiK3MoeLocalWorkletInput { num_tokens: 32 }.num_tokens, 32);
    }
}
