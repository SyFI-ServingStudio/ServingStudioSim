//! Gated DeltaNet q/k/v split and gating over one prefill request chunk.
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
pub struct GdnSplitGatingKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Model hidden width.
    pub hidden_size: u32,
    /// GDN key heads on this rank (16 / tp).
    pub num_key_heads: u32,
    /// GDN value heads on this rank (48 / tp).
    pub num_value_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Causal conv taps.
    pub conv_kernel_size: u32,
    /// Activation dtype.
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnSplitGatingKernelInput {
    /// Token rows of the call.
    pub num_tokens: u32,
}

pub struct GdnSplitGatingSpec;

impl KernelSpec for GdnSplitGatingSpec {
    type Config = GdnSplitGatingKernelConfig;
    type Input = GdnSplitGatingKernelInput;
    const KIND: KernelKind = "gdn_split_gating";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        neuron_grid::assert_neuron(Self::KIND, &config.backends);
        neuron_grid::grid_1d(neuron_grid::CHUNK)
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
                .with("hidden_size", config.hidden_size)
                .with("num_key_heads", config.num_key_heads)
                .with("num_value_heads", config.num_value_heads)
                .with("head_dim", config.head_dim)
                .with("conv_kernel_size", config.conv_kernel_size)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(GdnSplitGatingKernel, GdnSplitGatingSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(backends: Vec<&'static str>) -> GdnSplitGatingKernelConfig {
        GdnSplitGatingKernelConfig {
            backends,
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            hidden_size: 2560,
            num_key_heads: 4,
            num_value_heads: 12,
            head_dim: 128,
            conv_kernel_size: 4,
            dtype: DType::Bf16,
        }
    }

    /// Catches a payload field the Python args schema would reject, and a grid
    /// over the 500 feasible-coordinate ceiling.
    #[test]
    fn payload_matches_python_args() {
        let cfg = config(vec!["neuron_torch"]);
        let grid = GdnSplitGatingSpec::sweep_grid(&cfg);
        let payloads = GdnSplitGatingSpec::enumerate(&cfg, &grid, "neuron_torch");
        assert_eq!(payloads.len(), 8);
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "conv_kernel_size",
                "dtype",
                "head_dim",
                "hidden_size",
                "num_key_heads",
                "num_tokens",
                "num_value_heads",
            ]
        );
    }

    #[test]
    #[should_panic(expected = "Neuron kind")]
    fn rejects_a_gpu_backend() {
        GdnSplitGatingSpec::sweep_grid(&config(vec!["torch"]));
    }
}
