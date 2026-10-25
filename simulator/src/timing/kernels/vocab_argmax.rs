//! Argmax over this rank's local vocabulary rows.
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
pub struct VocabArgmaxKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Vocabulary rows on this rank.
    pub vocab_size: u32,
    /// Activation dtype.
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VocabArgmaxKernelInput {
    /// Token rows of the call.
    pub num_tokens: u32,
}

pub struct VocabArgmaxSpec;

impl KernelSpec for VocabArgmaxSpec {
    type Config = VocabArgmaxKernelConfig;
    type Input = VocabArgmaxKernelInput;
    const KIND: KernelKind = "vocab_argmax";

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
                .with("vocab_size", config.vocab_size)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(VocabArgmaxKernel, VocabArgmaxSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(backends: Vec<&'static str>) -> VocabArgmaxKernelConfig {
        VocabArgmaxKernelConfig {
            backends,
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            vocab_size: 62080,
            dtype: DType::Bf16,
        }
    }

    /// Catches a payload field the Python args schema would reject, and a grid
    /// over the 500 feasible-coordinate ceiling.
    #[test]
    fn payload_matches_python_args() {
        let cfg = config(vec!["neuron_torch"]);
        let grid = VocabArgmaxSpec::sweep_grid(&cfg);
        let payloads = VocabArgmaxSpec::enumerate(&cfg, &grid, "neuron_torch");
        assert_eq!(payloads.len(), 14);
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(keys, ["backend", "dtype", "num_tokens", "vocab_size",]);
    }

    #[test]
    #[should_panic(expected = "Neuron kind")]
    fn rejects_a_gpu_backend() {
        VocabArgmaxSpec::sweep_grid(&config(vec!["torch"]));
    }
}
