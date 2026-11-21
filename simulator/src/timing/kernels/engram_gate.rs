//! TileKernels' Engram gate (`tile_kernels.engram.engram_gate_fwd`).
//!
//! One `engram_gate_fwd_kernel_cuda_kernel` launch per Engram layer and GPU
//! gates the projected Engram value into each of the `hc_mult` streams in
//! place (`v4_modules/engram.py` `Engram.forward`, after the
//! `fused_kv_projection` linear).
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
pub struct EngramGateKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hc_mult: u32,
    pub hidden_size: Dim,
    #[compute_dtype]
    pub dtype: DType,
    /// Largest per-GPU token count the deployment runs. Bounds the token axis only.
    pub max_tokens_per_rank: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct EngramGateKernelInput {
    pub num_tokens: u32,
}

pub struct EngramGateSpec;

impl KernelSpec for EngramGateSpec {
    type Config = EngramGateKernelConfig;
    type Input = EngramGateKernelInput;

    const KIND: KernelKind = "engram_gate";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![token_points(config.max_tokens_per_rank)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(config.hc_mult >= 1, "hc_mult must be positive");
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
                .with("hc_mult", config.hc_mult)
                .with("hidden_size", config.hidden_size.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(EngramGateKernel, EngramGateSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::sweep::Axis;

    #[test]
    fn enumerate_emits_every_python_arg() {
        let config = EngramGateKernelConfig {
            backends: vec!["tile_kernels"],
            gpu_name: "NVIDIA B200".to_string(),
            hc_mult: 4,
            hidden_size: 5120.into(),
            dtype: DType::Bf16,
            max_tokens_per_rank: 65536,
        };
        let payloads = EngramGateSpec::enumerate(
            &config,
            &SweepGrid::new(vec![Axis::values([140])]),
            "tile_kernels",
        );
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            ["backend", "dtype", "hc_mult", "hidden_size", "num_tokens"]
        );
    }
}
