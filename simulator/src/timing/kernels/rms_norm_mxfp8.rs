//! DeepInfer's RMSNorm with the fused MXFP8 cast (`norm` with
//! `get_fp8_cast_params()`), one `_norm_kernel` launch: `q_a_norm` and the
//! attention layer norm after a shifted mHC pre. The query is the row count.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mega_moe::token_points;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RmsNormMxfp8KernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden: u32,
    pub return_bf16: bool,
    #[compute_dtype]
    pub dtype: DType,
    /// Largest per-GPU row count the deployment runs. Bounds the axis only.
    pub max_tokens_per_rank: u32,
}

#[derive(Clone, Debug, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct RmsNormMxfp8KernelInput {
    pub num_tokens: u32,
}

pub struct RmsNormMxfp8Spec;

impl KernelSpec for RmsNormMxfp8Spec {
    type Config = RmsNormMxfp8KernelConfig;
    type Input = RmsNormMxfp8KernelInput;

    const KIND: KernelKind = "rms_norm_mxfp8";

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
                .with("hidden", config.hidden)
                .with("return_bf16", config.return_bf16)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(RmsNormMxfp8Kernel, RmsNormMxfp8Spec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> RmsNormMxfp8KernelConfig {
        RmsNormMxfp8KernelConfig {
            backends: vec!["tilelang"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden: 1280,
            return_bf16: false,
            dtype: DType::Bf16,
            max_tokens_per_rank: 65536,
        }
    }

    #[test]
    fn enumerate_emits_every_python_arg() {
        let config = config();
        let grid = RmsNormMxfp8Spec::sweep_grid(&config);
        let payloads = RmsNormMxfp8Spec::enumerate(&config, &grid, "tilelang");
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            ["backend", "dtype", "hidden", "num_tokens", "return_bf16"]
        );
        assert!(payloads.len() <= 500);
    }
}
