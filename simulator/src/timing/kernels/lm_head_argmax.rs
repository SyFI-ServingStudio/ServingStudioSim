//! LM head GEMM fused with the local argmax.
//!
//! Neuron backends: `neuron_torch`, `neuron_nki`. Contract:
//! `doc/pr/trn2-flashnext/L1_CONTRACT.md`; shapes are per rank.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_grid;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LmHeadArgmaxKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Model hidden width.
    pub hidden_size: u32,
    /// Vocabulary rows on this rank.
    pub vocab_size: u32,
    /// Activation dtype.
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LmHeadArgmaxKernelInput {
    /// Token rows of the call.
    pub num_tokens: u32,
}

pub struct LmHeadArgmaxSpec;

impl KernelSpec for LmHeadArgmaxSpec {
    type Config = LmHeadArgmaxKernelConfig;
    type Input = LmHeadArgmaxKernelInput;
    const KIND: KernelKind = "lm_head_argmax";

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
                .with("hidden_size", config.hidden_size)
                .with("vocab_size", config.vocab_size)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(LmHeadArgmaxKernel, LmHeadArgmaxSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(backends: Vec<&'static str>) -> LmHeadArgmaxKernelConfig {
        LmHeadArgmaxKernelConfig {
            backends,
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            hidden_size: 2560,
            vocab_size: 62080,
            dtype: DType::Bf16,
        }
    }

    /// Catches a payload field the Python args schema would reject, and a grid
    /// over the 500 feasible-coordinate ceiling.
    #[test]
    fn payload_matches_python_args() {
        let cfg = config(vec!["neuron_torch"]);
        let grid = LmHeadArgmaxSpec::sweep_grid(&cfg);
        let payloads = LmHeadArgmaxSpec::enumerate(&cfg, &grid, "neuron_torch");
        assert_eq!(payloads.len(), 14);
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "dtype",
                "hidden_size",
                "num_tokens",
                "vocab_size",
            ]
        );
    }

    #[test]
    #[should_panic(expected = "Neuron kind")]
    fn rejects_a_gpu_backend() {
        LmHeadArgmaxSpec::sweep_grid(&config(vec!["torch"]));
    }
}
