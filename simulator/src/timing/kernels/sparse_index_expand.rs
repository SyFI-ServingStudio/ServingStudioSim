//! Expand each row's selected compressed groups into the packed token list a
//! sparse attention call reads: every group's `compress_ratio` tokens, the
//! row's open causal tail, and the valid count.

use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparseIndexExpandKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub token_topk: u32,
    pub compress_ratio: u32,
}

#[derive(Clone, Debug, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct SparseIndexExpandKernelInput {
    pub num_rows: u32,
}

pub struct SparseIndexExpandSpec;

impl KernelSpec for SparseIndexExpandSpec {
    type Config = SparseIndexExpandKernelConfig;
    type Input = SparseIndexExpandKernelInput;

    const KIND: KernelKind = "sparse_index_expand";

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
        grid.expand_1d(|num_rows| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_rows", num_rows as u32)
                .with("token_topk", config.token_topk)
                .with("compress_ratio", config.compress_ratio)
        })
    }
}

register_kernel!(SparseIndexExpandKernel, SparseIndexExpandSpec);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_emits_the_python_args() {
        let cfg = SparseIndexExpandKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".into(),
            token_topk: 2048,
            compress_ratio: 4,
        };
        let grid = SparseIndexExpandSpec::sweep_grid(&cfg);
        let specs = SparseIndexExpandSpec::enumerate(&cfg, &grid, "vllm_triton");
        let names: Vec<&str> = specs[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            names,
            ["backend", "compress_ratio", "num_rows", "token_topk"]
        );
    }
}
