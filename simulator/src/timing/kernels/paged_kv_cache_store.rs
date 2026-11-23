//! DeepInfer's paged-cache store (`store_kv_cache`), one launch: the prefill
//! SWA store, whose slot map covers every new row and stores only each
//! sequence's last `block_size` rows. The query is the row count.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mega_moe::token_points;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PagedKvCacheStoreKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub block_size: u32,
    pub cache_format: String,
    #[compute_dtype]
    pub dtype: DType,
    /// Largest per-GPU row count the deployment runs. Bounds the axis only.
    pub max_tokens_per_rank: u32,
}

#[derive(Clone, Debug, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct PagedKvCacheStoreKernelInput {
    pub num_tokens: u32,
}

pub struct PagedKvCacheStoreSpec;

impl KernelSpec for PagedKvCacheStoreSpec {
    type Config = PagedKvCacheStoreKernelConfig;
    type Input = PagedKvCacheStoreKernelInput;

    const KIND: KernelKind = "paged_kv_cache_store";

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
            ["mxfp8", "nvfp4", "mxfp4"].contains(&config.cache_format.as_str()),
            "cache_format must be one of mxfp8, nvfp4, mxfp4, got {}",
            config.cache_format
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
                .with("block_size", config.block_size)
                .with("cache_format", config.cache_format.as_str())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(PagedKvCacheStoreKernel, PagedKvCacheStoreSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> PagedKvCacheStoreKernelConfig {
        PagedKvCacheStoreKernelConfig {
            backends: vec!["tilelang"],
            gpu_name: "NVIDIA B200".to_string(),
            block_size: 134,
            cache_format: "mxfp8".to_string(),
            dtype: DType::Bf16,
            max_tokens_per_rank: 65536,
        }
    }

    #[test]
    fn enumerate_emits_every_python_arg() {
        let config = config();
        let grid = PagedKvCacheStoreSpec::sweep_grid(&config);
        let payloads = PagedKvCacheStoreSpec::enumerate(&config, &grid, "tilelang");
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "block_size",
                "cache_format",
                "dtype",
                "num_tokens"
            ]
        );
        assert!(payloads.len() <= 500);
    }
}
