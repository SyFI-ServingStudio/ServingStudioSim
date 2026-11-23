//! DeepInfer's prefill KV compressor (`compress_and_update_state_prefill`):
//! score-weighted pooling of every `compress_ratio` rows plus the state-ring
//! update, two launches per call. The query is the new row count.

use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mega_moe::token_points;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompressStatePrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub head_dim: u32,
    pub compress_ratio: u32,
    pub state_size: u32,
    /// Largest per-GPU row count the deployment runs. Bounds the axis only.
    pub max_tokens_per_rank: u32,
}

#[derive(Clone, Debug, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct CompressStatePrefillKernelInput {
    pub num_tokens: u32,
}

pub struct CompressStatePrefillSpec;

impl KernelSpec for CompressStatePrefillSpec {
    type Config = CompressStatePrefillKernelConfig;
    type Input = CompressStatePrefillKernelInput;

    const KIND: KernelKind = "compress_state_prefill";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![token_points(config.max_tokens_per_rank)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
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
                .with("head_dim", config.head_dim)
                .with("compress_ratio", config.compress_ratio)
                .with("state_size", config.state_size)
        })
    }
}

register_kernel!(CompressStatePrefillKernel, CompressStatePrefillSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> CompressStatePrefillKernelConfig {
        CompressStatePrefillKernelConfig {
            backends: vec!["tilelang"],
            gpu_name: "NVIDIA B200".to_string(),
            head_dim: 512,
            compress_ratio: 2,
            state_size: 7,
            max_tokens_per_rank: 65536,
        }
    }

    #[test]
    fn enumerate_emits_every_python_arg() {
        let config = config();
        let grid = CompressStatePrefillSpec::sweep_grid(&config);
        let payloads = CompressStatePrefillSpec::enumerate(&config, &grid, "tilelang");
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "compress_ratio",
                "head_dim",
                "num_tokens",
                "state_size"
            ]
        );
        assert!(payloads.len() <= 500);
    }
}
