//! DeepInfer's fused RMSNorm + RoPE + cast + optional paged store
//! (`fused_norm_rope_hadamard_cast_store_kvcache`), one launch per call.
//!
//! DeepSeek-V4.1 prefill: SWA KV (`mxfp8`, `cast_back`), main compressed KV
//! store (`nvfp4`, `cache_and_norm`), index Q (`mxfp4`, `quantized`) and index
//! K store (`mxfp4`, `cache`). The query is the call's row count.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mega_moe::token_points;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FusedNormRopeCastKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: u32,
    pub head_dim: u32,
    pub rope_dim: u32,
    pub norm: bool,
    pub cast_format: String,
    pub output: String,
    pub block_size: u32,
    #[compute_dtype]
    pub dtype: DType,
    /// Largest per-GPU row count the deployment runs. Bounds the axis only.
    pub max_tokens_per_rank: u32,
}

#[derive(Clone, Debug, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct FusedNormRopeCastKernelInput {
    pub num_tokens: u32,
}

pub struct FusedNormRopeCastSpec;

impl KernelSpec for FusedNormRopeCastSpec {
    type Config = FusedNormRopeCastKernelConfig;
    type Input = FusedNormRopeCastKernelInput;

    const KIND: KernelKind = "fused_norm_rope_cast";

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
        anyhow::ensure!(
            ["mxfp8", "nvfp4", "mxfp4"].contains(&config.cast_format.as_str()),
            "cast_format must be one of mxfp8, nvfp4, mxfp4, got {}",
            config.cast_format
        );
        anyhow::ensure!(
            ["cast_back", "quantized", "cache", "cache_and_norm"].contains(&config.output.as_str()),
            "output must be one of cast_back, quantized, cache, cache_and_norm, got {}",
            config.output
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
                .with("num_heads", config.num_heads)
                .with("head_dim", config.head_dim)
                .with("rope_dim", config.rope_dim)
                .with("norm", config.norm)
                .with("cast_format", config.cast_format.as_str())
                .with("output", config.output.as_str())
                .with("block_size", config.block_size)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(FusedNormRopeCastKernel, FusedNormRopeCastSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> FusedNormRopeCastKernelConfig {
        FusedNormRopeCastKernelConfig {
            backends: vec!["tilelang"],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads: 1,
            head_dim: 512,
            rope_dim: 64,
            norm: true,
            cast_format: "mxfp8".to_string(),
            output: "cast_back".to_string(),
            block_size: 0,
            dtype: DType::Bf16,
            max_tokens_per_rank: 65536,
        }
    }

    #[test]
    fn enumerate_emits_every_python_arg() {
        let config = config();
        let grid = FusedNormRopeCastSpec::sweep_grid(&config);
        let payloads = FusedNormRopeCastSpec::enumerate(&config, &grid, "tilelang");
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "block_size",
                "cast_format",
                "dtype",
                "head_dim",
                "norm",
                "num_heads",
                "num_tokens",
                "output",
                "rope_dim"
            ]
        );
        assert!(payloads.len() <= 500);
    }
}
