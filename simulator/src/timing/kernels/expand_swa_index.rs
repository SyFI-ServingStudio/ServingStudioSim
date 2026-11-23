//! DeepInfer's SWA index expansion (`expand_swa_index`): four pivots per row
//! to the row's window of SWA key indices. The query is the row count.

use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mega_moe::token_points;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExpandSwaIndexKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub window_size: u32,
    /// Largest per-GPU row count the deployment runs. Bounds the axis only.
    pub max_tokens_per_rank: u32,
}

#[derive(Clone, Debug, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct ExpandSwaIndexKernelInput {
    pub num_rows: u32,
}

pub struct ExpandSwaIndexSpec;

impl KernelSpec for ExpandSwaIndexSpec {
    type Config = ExpandSwaIndexKernelConfig;
    type Input = ExpandSwaIndexKernelInput;

    const KIND: KernelKind = "expand_swa_index";

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
        grid.expand_1d(|num_rows| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_rows", num_rows as u32)
                .with("window_size", config.window_size)
        })
    }
}

register_kernel!(ExpandSwaIndexKernel, ExpandSwaIndexSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ExpandSwaIndexKernelConfig {
        ExpandSwaIndexKernelConfig {
            backends: vec!["tilelang"],
            gpu_name: "NVIDIA B200".to_string(),
            window_size: 128,
            max_tokens_per_rank: 65536,
        }
    }

    #[test]
    fn enumerate_emits_every_python_arg() {
        let config = config();
        let grid = ExpandSwaIndexSpec::sweep_grid(&config);
        let payloads = ExpandSwaIndexSpec::enumerate(&config, &grid, "tilelang");
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(names, ["backend", "num_rows", "window_size"]);
        assert!(payloads.len() <= 500);
    }
}
