//! Dense FFN partial (Neuron): One rank's 1/ep column shard of a dense FFN layer (layers 0-2) over every gathered row.
//!
//! Neuron backends only (`neuron_<impl>`, the op module's `IMPLS` key).
//! Contract: `doc/pr/trn2-glm53-flash/L1_CONTRACT.md`; shapes are per rank. Rows come from the
//! glm5-trn2 bench (bench_moe.py kind dense_ffn_partial) through `tools/neuron-bench-import`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_grid;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DenseFfnPartialKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Model hidden width.
    pub hidden_size: u32,
    /// Intermediate columns on this rank (12288 / ep).
    pub intermediate_size: u32,
    /// Activation dtype.
    #[compute_dtype]
    pub dtype: DType,
    /// Weight storage dtype (FP8, upcast in-graph).
    pub weight_dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DenseFfnPartialKernelInput {
    /// Rows gathered over the world.
    pub num_tokens: u32,
}

pub struct DenseFfnPartialSpec;

impl KernelSpec for DenseFfnPartialSpec {
    type Config = DenseFfnPartialKernelConfig;
    type Input = DenseFfnPartialKernelInput;
    const KIND: KernelKind = "dense_ffn_partial";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        neuron_grid::assert_neuron(Self::KIND, &config.backends);
        neuron_grid::grid_1d(neuron_grid::FFN_ROWS)
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
                .with("intermediate_size", config.intermediate_size)
                .with("dtype", config.dtype.as_str())
                .with("weight_dtype", config.weight_dtype.as_str())
        })
    }
}

register_kernel!(DenseFfnPartialKernel, DenseFfnPartialSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(backends: Vec<&'static str>) -> DenseFfnPartialKernelConfig {
        DenseFfnPartialKernelConfig {
            backends,
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            hidden_size: 8,
            intermediate_size: 8,
            dtype: DType::Bf16,
            weight_dtype: DType::Fp8E4m3,
        }
    }

    /// Catches a payload field the Python args schema would reject.
    #[test]
    fn payload_matches_python_args() {
        let cfg = config(vec!["neuron_torch"]);
        let grid = DenseFfnPartialSpec::sweep_grid(&cfg);
        let payloads = DenseFfnPartialSpec::enumerate(&cfg, &grid, "neuron_torch");
        assert!(!payloads.is_empty());
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "dtype",
                "hidden_size",
                "intermediate_size",
                "num_tokens",
                "weight_dtype"
            ]
        );
    }

    #[test]
    #[should_panic(expected = "Neuron kind")]
    fn rejects_a_gpu_backend() {
        DenseFfnPartialSpec::sweep_grid(&config(vec!["torch"]));
    }
}
