//! DeepInfer's paged-cache load (`load_paged_kv_cache`), one launch: the SWA
//! history, the main compressed KV at a KV source, the index K at an index-K
//! owner. The query is the loaded token count (`total_k`), which follows the
//! context, so the axis runs past the per-rank token budget to `max_tokens`.

use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mega_moe::token_points;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PagedKvCacheLoadKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub block_size: u32,
    pub segment_tokens: u32,
    pub cache_format: String,
    /// Largest loaded token count the deployment reaches. Bounds the axis only.
    pub max_tokens: u32,
}

#[derive(Clone, Debug, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct PagedKvCacheLoadKernelInput {
    pub num_tokens: u32,
}

pub struct PagedKvCacheLoadSpec;

/// The per-rank token axis to 131072, then alternating 1.5x / 4/3x steps
/// (powers of two and their midpoints) up to the first point at or above
/// `max_tokens`: prefill loads cover the whole context.
fn load_points(max_tokens: u32) -> Vec<f64> {
    let mut points = token_points(max_tokens.min(131_072));
    let mut power = 131_072_f64;
    while *points.last().unwrap() < f64::from(max_tokens) {
        let last = *points.last().unwrap();
        if last == power {
            points.push(power * 1.5);
        } else {
            power *= 2.0;
            points.push(power);
        }
    }
    points
}

impl KernelSpec for PagedKvCacheLoadSpec {
    type Config = PagedKvCacheLoadKernelConfig;
    type Input = PagedKvCacheLoadKernelInput;

    const KIND: KernelKind = "paged_kv_cache_load";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![load_points(config.max_tokens)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(config.max_tokens >= 1, "max_tokens must be positive");
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
                .with("segment_tokens", config.segment_tokens)
                .with("cache_format", config.cache_format.as_str())
        })
    }
}

register_kernel!(PagedKvCacheLoadKernel, PagedKvCacheLoadSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> PagedKvCacheLoadKernelConfig {
        PagedKvCacheLoadKernelConfig {
            backends: vec!["tilelang"],
            gpu_name: "NVIDIA B200".to_string(),
            block_size: 134,
            segment_tokens: 128,
            cache_format: "mxfp8".to_string(),
            max_tokens: 4_194_304,
        }
    }

    #[test]
    fn enumerate_emits_every_python_arg() {
        let config = config();
        let grid = PagedKvCacheLoadSpec::sweep_grid(&config);
        let payloads = PagedKvCacheLoadSpec::enumerate(&config, &grid, "tilelang");
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "block_size",
                "cache_format",
                "num_tokens",
                "segment_tokens"
            ]
        );
        assert!(payloads.len() <= 500);
    }

    #[test]
    fn load_axis_reaches_max_tokens() {
        let points = load_points(4_194_304);
        assert_eq!(*points.last().unwrap(), 4_194_304.0);
        assert!(points.windows(2).all(|w| w[0] < w[1]));
        assert!(points.contains(&65536.0) && points.contains(&196_608.0));
    }
}
