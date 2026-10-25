//! Fused PLE project + gate + conv + residual (after the all-reduce).
//!
//! Neuron backends: `neuron_torch`. Contract:
//! `doc/pr/trn2-flashnext/L1_CONTRACT.md`; shapes are per rank.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_grid;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PlePostKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// PLE embedding width (heads x head_dim).
    pub embed_dim: u32,
    /// Model hidden width.
    pub hidden_size: u32,
    /// Hyper-connection residual streams.
    pub hc_mult: u32,
    /// Causal conv taps.
    pub conv_kernel_size: u32,
    /// Spacing between conv taps, in tokens.
    pub dilation: u32,
    /// Activation dtype.
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PlePostKernelInput {
    /// Token rows of the call.
    pub num_tokens: u32,
}

pub struct PlePostSpec;

impl KernelSpec for PlePostSpec {
    type Config = PlePostKernelConfig;
    type Input = PlePostKernelInput;
    const KIND: KernelKind = "ple_post";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        neuron_grid::assert_neuron(Self::KIND, &config.backends);
        neuron_grid::grid_1d(neuron_grid::TOKENS)
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
                .with("num_tokens", num_tokens.round() as u32)
                .with("embed_dim", config.embed_dim)
                .with("hidden_size", config.hidden_size)
                .with("hc_mult", config.hc_mult)
                .with("conv_kernel_size", config.conv_kernel_size)
                .with("dilation", config.dilation)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(PlePostKernel, PlePostSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(backends: Vec<&'static str>) -> PlePostKernelConfig {
        PlePostKernelConfig {
            backends,
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            embed_dim: 2560,
            hidden_size: 2560,
            hc_mult: 4,
            conv_kernel_size: 4,
            dilation: 3,
            dtype: DType::Bf16,
        }
    }

    /// Catches a payload field the Python args schema would reject, and a grid
    /// over the 500 feasible-coordinate ceiling.
    #[test]
    fn payload_matches_python_args() {
        let cfg = config(vec!["neuron_torch"]);
        let grid = PlePostSpec::sweep_grid(&cfg);
        let payloads = PlePostSpec::enumerate(&cfg, &grid, "neuron_torch");
        assert_eq!(payloads.len(), 14);
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "conv_kernel_size",
                "dilation",
                "dtype",
                "embed_dim",
                "hc_mult",
                "hidden_size",
                "num_tokens",
            ]
        );
    }

    #[test]
    #[should_panic(expected = "Neuron kind")]
    fn rejects_a_gpu_backend() {
        PlePostSpec::sweep_grid(&config(vec!["torch"]));
    }
}
