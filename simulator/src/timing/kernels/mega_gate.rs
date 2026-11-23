//! DeepGEMM's fused MoE router (`deep_gemm.bf16_mega_gate`).
//!
//! One `sm100_bf16_mega_gate_impl` launch per MoE layer and GPU: router
//! logits (bf16 x bf16, fp32 accumulation), scoring, score-correction bias,
//! top-k and the logical-to-physical (EPLB) slot map. DeepInfer calls it from
//! `v4_modules/mlp.py` (`use_mega_gate`) before the mega MoE.
//!
//! The query is the GPU's token count. `max_tokens_per_rank` only bounds the
//! token axis.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mega_moe::token_points;
use crate::timing::sweep::SweepGrid;
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MegaGateKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_size: Dim,
    /// Logical routed experts the router scores.
    pub num_experts: u32,
    /// Physical slots, redundant (EPLB) slots included, choices map to.
    pub num_physical_experts: u32,
    pub top_k: u32,
    /// The model's router scoring function (`sqrtsoftplus`, `sigmoid`, ...).
    pub scoring_func: String,
    #[compute_dtype]
    pub dtype: DType,
    /// Largest per-GPU token count the deployment runs. Bounds the token axis only.
    pub max_tokens_per_rank: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MegaGateKernelInput {
    pub num_tokens: u32,
}

pub struct MegaGateSpec;

impl KernelSpec for MegaGateSpec {
    type Config = MegaGateKernelConfig;
    type Input = MegaGateKernelInput;

    const KIND: KernelKind = "mega_gate";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![token_points(config.max_tokens_per_rank)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.num_physical_experts >= config.num_experts,
            "num_physical_experts {} < num_experts {}",
            config.num_physical_experts,
            config.num_experts
        );
        anyhow::ensure!(
            config.top_k >= 1 && config.top_k <= config.num_experts,
            "top_k {} must be in 1..=num_experts",
            config.top_k
        );
        anyhow::ensure!(
            config.max_tokens_per_rank >= 1,
            "max_tokens_per_rank must be positive"
        );
        Ok(())
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
                .with("num_experts", config.num_experts)
                .with("num_physical_experts", config.num_physical_experts)
                .with("top_k", config.top_k)
                .with("scoring_func", config.scoring_func.as_str())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(MegaGateKernel, MegaGateSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::sweep::Axis;

    #[test]
    fn enumerate_emits_every_python_arg() {
        let config = MegaGateKernelConfig {
            backends: vec!["deepgemm_sm100"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden_size: 5120.into(),
            num_experts: 384,
            num_physical_experts: 416,
            top_k: 6,
            scoring_func: "sqrtsoftplus".to_string(),
            dtype: DType::Bf16,
            max_tokens_per_rank: 65536,
        };
        let payloads = MegaGateSpec::enumerate(
            &config,
            &SweepGrid::new(vec![Axis::values([140])]),
            "deepgemm_sm100",
        );
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "dtype",
                "hidden_size",
                "num_experts",
                "num_physical_experts",
                "num_tokens",
                "scoring_func",
                "top_k",
            ]
        );
        assert!(MegaGateSpec::validate_config(&config).is_ok());
    }
}
