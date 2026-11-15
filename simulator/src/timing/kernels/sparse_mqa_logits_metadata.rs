//! DeepGEMM's schedule for the candidate-restricted indexer logits
//! (`deep_gemm.get_sparse_mqa_logits_metadata`, one
//! `sm100_sparse_mqa_logits_metadata` launch).
//!
//! DeepInfer caches it per query chunk across a candidate group's consumer
//! layers (`index_attn.py` `_get_sparse_prefill_metadata`), so it runs once
//! per chunk and group. Coordinates as
//! [`sparse_mqa_logits_prefill`](super::sparse_mqa_logits_prefill).

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mqa_logits_prefill_chunk::{
    context_points, tail_infeasible, CHUNK_QUERY_POINTS,
};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparseMqaLogitsMetadataKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub compress_ratio: u32,
    pub candidate_blocks: u32,
    pub candidate_block_size: u32,
    #[compute_dtype]
    pub dtype: DType,
    /// Longest context the deployment runs. Bounds the context axis only.
    pub max_context_len: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct SparseMqaLogitsMetadataKernelInput {
    pub num_queries: u32,
    pub context_len: u32,
}

pub struct SparseMqaLogitsMetadataSpec;

impl KernelSpec for SparseMqaLogitsMetadataSpec {
    type Config = SparseMqaLogitsMetadataKernelConfig;
    type Input = SparseMqaLogitsMetadataKernelInput;

    const KIND: KernelKind = "sparse_mqa_logits_metadata";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values(CHUNK_QUERY_POINTS),
            context_points(config.max_context_len),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Clamp)
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        tail_infeasible(grid)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|num_queries, context_len| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_queries", num_queries as u32)
                .with("context_len", context_len as u32)
                .with("compress_ratio", config.compress_ratio)
                .with("candidate_blocks", config.candidate_blocks)
                .with("candidate_block_size", config.candidate_block_size)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(SparseMqaLogitsMetadataKernel, SparseMqaLogitsMetadataSpec);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_emits_every_python_arg() {
        let config = SparseMqaLogitsMetadataKernelConfig {
            backends: vec!["deepgemm_sm100"],
            gpu_name: "NVIDIA B200".to_string(),
            compress_ratio: 1,
            candidate_blocks: 2048,
            candidate_block_size: 8,
            dtype: DType::Mxfp4E2m1,
            max_context_len: 131072,
        };
        let grid = SweepGrid::new(vec![Axis::values([140]), Axis::values([131072])]);
        let payloads = SparseMqaLogitsMetadataSpec::enumerate(&config, &grid, "deepgemm_sm100");
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "candidate_block_size",
                "candidate_blocks",
                "compress_ratio",
                "context_len",
                "dtype",
                "num_queries",
            ]
        );
    }
}
