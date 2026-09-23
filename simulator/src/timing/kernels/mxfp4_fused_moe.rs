//! Kimi-K3 FlashInfer TRT-LLM MXFP4 fused MoE.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::routing::{
    sample_and_fold_layerwise_random_topk_expert_counts,
    sample_and_fold_layerwise_topk_expert_counts,
};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

const FOLD_SEED: u64 = 0xF01D_5EED;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Mxfp4FusedMoeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_size: Dim,
    pub intermediate_size: Dim,
    pub num_experts: Dim,
    pub num_local_experts: Dim,
    pub top_k: u32,
    #[compute_dtype]
    pub input_dtype: DType,
    pub weight_format: String,
    pub group_size: u32,
    pub routing_method: String,
    pub activation: String,
    pub n_group: u32,
    pub topk_group: u32,
    pub routed_scaling_numerator: u32,
    pub routed_scaling_denominator: u32,
    pub gemm1_alpha: u32,
    pub gemm1_clamp_limit: u32,
    pub layerwise_global_ppm: Vec<Vec<u32>>,
    pub folded_rank_position: u32,
    #[serde(default)]
    pub stochastic_routing: bool,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct Mxfp4FusedMoeKernelInput {
    pub num_tokens: u32,
}

pub struct Mxfp4FusedMoeSpec;

impl KernelSpec for Mxfp4FusedMoeSpec {
    type Config = Mxfp4FusedMoeKernelConfig;
    type Input = Mxfp4FusedMoeKernelInput;

    const KIND: KernelKind = "mxfp4_fused_moe";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::chain([
            Axis::values([1, 4, 8, 16, 32, 48]),
            Axis::token_axis(),
        ])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        let num_experts = config.num_experts.get() as usize;
        let local_experts = config.num_local_experts.get() as usize;
        assert!(local_experts > 0);
        assert_eq!(num_experts % local_experts, 0);
        let rank_offset = config.folded_rank_position as usize * local_experts;
        assert!(rank_offset + local_experts <= num_experts);

        grid.expand_1d(|num_tokens| {
            let mut per_expert_batches = if config.stochastic_routing {
                sample_and_fold_layerwise_random_topk_expert_counts(
                    &config.layerwise_global_ppm,
                    config.top_k,
                    num_tokens as u32,
                    local_experts,
                    FOLD_SEED,
                )
            } else {
                sample_and_fold_layerwise_topk_expert_counts(
                    &config.layerwise_global_ppm,
                    config.top_k,
                    num_tokens as u32,
                    local_experts,
                    FOLD_SEED,
                )
            };
            per_expert_batches.rotate_left(rank_offset);
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
                .with("hidden_size", config.hidden_size.get())
                .with("intermediate_size", config.intermediate_size.get())
                .with("num_experts", config.num_experts.get())
                .with("num_local_experts", config.num_local_experts.get())
                .with("top_k", config.top_k)
                .with("input_dtype", config.input_dtype.as_str())
                .with("weight_format", config.weight_format.as_str())
                .with("group_size", config.group_size)
                .with("routing_method", config.routing_method.as_str())
                .with("activation", config.activation.as_str())
                .with("n_group", config.n_group)
                .with("topk_group", config.topk_group)
                .with(
                    "routed_scaling_factor",
                    config.routed_scaling_numerator as f64
                        / config.routed_scaling_denominator as f64,
                )
                .with("gemm1_alpha", config.gemm1_alpha as f64)
                .with("gemm1_clamp_limit", config.gemm1_clamp_limit as f64)
                .with("per_expert_batches", per_expert_batches)
        })
    }
}

register_kernel!(Mxfp4FusedMoeKernel, Mxfp4FusedMoeSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};

    fn config() -> Mxfp4FusedMoeKernelConfig {
        let mut ppm = vec![1116_u32; 896];
        ppm[0] += 64;
        Mxfp4FusedMoeKernelConfig {
            backends: vec!["sglang_trtllm_mxfp4"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden_size: 3584.into(),
            intermediate_size: 3072.into(),
            num_experts: 896.into(),
            num_local_experts: 112.into(),
            top_k: 16,
            input_dtype: DType::Bf16,
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
            stochastic_routing: false,
        }
    }

    #[test]
    fn k3_config_and_grid_are_registered() {
        let cfg = config();
        assert_eq!(Mxfp4FusedMoeSpec::KIND, "mxfp4_fused_moe");
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(Mxfp4FusedMoeSpec::sweep_grid(&cfg).axes()[0][0], 1.0);
        assert_eq!(
            Mxfp4FusedMoeSpec::cache_kind("sglang_trtllm_mxfp4"),
            CacheKind::Cache1DLinear
        );
    }

    #[test]
    fn payload_contains_activation_scalars_and_exact_global_histogram() {
        let payload = Mxfp4FusedMoeSpec::enumerate(
            &config(),
            &SweepGrid::new(vec![Axis::values([16])]),
            "sglang_trtllm_mxfp4",
        )[0]
        .clone();
        let fields = payload.fields();
        assert_eq!(fields.len(), 18);
        assert_eq!(fields["activation"], serde_json::json!("situ"));
        assert_eq!(fields["gemm1_alpha"], serde_json::json!(4.0));
        assert_eq!(fields["gemm1_clamp_limit"], serde_json::json!(25.0));
        assert_eq!(fields["per_expert_batches"].as_array().unwrap().len(), 896);
        assert_eq!(
            fields["per_expert_batches"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .sum::<u64>(),
            16 * 16
        );
    }
}
