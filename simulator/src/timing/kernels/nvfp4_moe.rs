//! Monolithic FlashInfer TRT-LLM NVFP4 MoE timing leaf on Blackwell.
//!
//! The framework call owns routing, both expert GEMMs, and finalize. Activation
//! quantization remains a separate leaf because nsys observes it immediately
//! before this call.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Nvfp4MoeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_size: Dim,
    pub intermediate_size: Dim,
    pub num_experts: Dim,
    pub num_local_experts: Dim,
    pub local_expert_offset: u32,
    pub top_k: u32,
    #[compute_dtype]
    pub input_dtype: DType,
    pub weight_format: String,
    pub group_size: u32,
    pub routing_method: String,
    pub n_group: u32,
    pub topk_group: u32,
    pub routed_scaling_numerator: u32,
    pub routed_scaling_denominator: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct Nvfp4MoeKernelInput {
    pub num_tokens: u32,
}

pub struct Nvfp4MoeSpec;

impl KernelSpec for Nvfp4MoeSpec {
    type Config = Nvfp4MoeKernelConfig;
    type Input = Nvfp4MoeKernelInput;

    const KIND: KernelKind = "nvfp4_moe";

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
        grid.expand_1d(|num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
                .with("hidden_size", config.hidden_size.get())
                .with("intermediate_size", config.intermediate_size.get())
                .with("num_experts", config.num_experts.get())
                .with("num_local_experts", config.num_local_experts.get())
                .with("local_expert_offset", config.local_expert_offset)
                .with("top_k", config.top_k)
                .with("input_dtype", config.input_dtype.as_str())
                .with("weight_format", config.weight_format.as_str())
                .with("group_size", config.group_size)
                .with("routing_method", config.routing_method.as_str())
                .with("n_group", config.n_group)
                .with("topk_group", config.topk_group)
                .with("routed_scaling_numerator", config.routed_scaling_numerator)
                .with(
                    "routed_scaling_denominator",
                    config.routed_scaling_denominator,
                )
        })
    }
}

register_kernel!(Nvfp4MoeKernel, Nvfp4MoeSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn config(local_experts: u32, offset: u32) -> Nvfp4MoeKernelConfig {
        Nvfp4MoeKernelConfig {
            backends: vec!["flashinfer_trtllm"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden_size: 6144.into(),
            intermediate_size: 2048.into(),
            num_experts: 256.into(),
            num_local_experts: local_experts.into(),
            local_expert_offset: offset,
            top_k: 8,
            input_dtype: DType::Bf16,
            weight_format: "nvfp4_e2m1".to_string(),
            group_size: 16,
            routing_method: "minimax2".to_string(),
            n_group: 1,
            topk_group: 1,
            routed_scaling_numerator: 5,
            routed_scaling_denominator: 2,
        }
    }

    #[test]
    fn expert_partition_is_cache_identity() {
        assert_ne!(config(64, 0), config(32, 0));
        assert_ne!(config(64, 0), config(64, 64));
    }

    #[test]
    fn enumerate_matches_python_schema() {
        let config = config(64, 128);
        let grid = Nvfp4MoeSpec::sweep_grid(&config);
        let fields = Nvfp4MoeSpec::enumerate(&config, &grid, "flashinfer_trtllm")[0]
            .fields()
            .clone();
        assert_eq!(fields.len(), 16);
        assert_eq!(fields.get("num_local_experts"), Some(&Value::from(64_u32)));
        assert_eq!(
            fields.get("local_expert_offset"),
            Some(&Value::from(128_u32))
        );
        assert_eq!(
            fields.get("weight_format"),
            Some(&Value::from("nvfp4_e2m1"))
        );
        assert_eq!(
            fields.get("routed_scaling_numerator"),
            Some(&Value::from(5_u32))
        );
    }
}
