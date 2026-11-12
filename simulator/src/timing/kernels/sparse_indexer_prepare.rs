//! Sparse-indexer preparation for one step: RMS-normalize and rotate the
//! indexer queries, push raw keys into each request's ring, and pool every
//! completed group of `compress_ratio` keys into the compressed cache.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparseIndexerPrepareKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub compress_ratio: u32,
    pub ring_size: u32,
    #[compute_dtype]
    pub input_dtype: DType,
    #[kv_dtype]
    pub cache_dtype: DType,
}

#[derive(Clone, Debug, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct SparseIndexerPrepareKernelInput {
    pub num_tokens: u32,
}

pub struct SparseIndexerPrepareSpec;

impl KernelSpec for SparseIndexerPrepareSpec {
    type Config = SparseIndexerPrepareKernelConfig;
    type Input = SparseIndexerPrepareKernelInput;

    const KIND: KernelKind = "sparse_indexer_prepare";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::chain([Axis::pow2(0, 4), Axis::token_axis()])])
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
                .with("num_tokens", num_tokens as u32)
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("compress_ratio", config.compress_ratio)
                .with("ring_size", config.ring_size)
                .with("input_dtype", config.input_dtype.as_str())
                .with("cache_dtype", config.cache_dtype.as_str())
        })
    }
}

register_kernel!(SparseIndexerPrepareKernel, SparseIndexerPrepareSpec);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_emits_the_python_args() {
        let cfg = SparseIndexerPrepareKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".into(),
            num_heads: Dim::param("num_heads", 4),
            head_dim: Dim::param("head_dim", 128),
            compress_ratio: 4,
            ring_size: 8,
            input_dtype: DType::Bf16,
            cache_dtype: DType::Fp8E4m3,
        };
        let grid = SparseIndexerPrepareSpec::sweep_grid(&cfg);
        let specs = SparseIndexerPrepareSpec::enumerate(&cfg, &grid, "vllm_triton");
        let names: Vec<&str> = specs[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            names,
            [
                "backend",
                "cache_dtype",
                "compress_ratio",
                "head_dim",
                "input_dtype",
                "num_heads",
                "num_tokens",
                "ring_size"
            ]
        );
    }
}
