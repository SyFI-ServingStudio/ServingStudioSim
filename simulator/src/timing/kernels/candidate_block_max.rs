//! DeepInfer's candidate block scores (`tilelang_ops/candidate_blocks.py`
//! `candidate_block_max`: a task-list launch and the block-max launch).
//!
//! On the candidate source layer each row's dense bf16 logits are reduced to
//! one score per `block_size` keys before DeepSelect keeps the top blocks.
//! Coordinates as [`mqa_logits_prefill_chunk`](super::mqa_logits_prefill_chunk):
//! the chunk's rows and the context they are a tail of.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mqa_logits_prefill_chunk::{
    context_points, tail_infeasible, CHUNK_QUERY_POINTS,
};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CandidateBlockMaxKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub compress_ratio: u32,
    pub block_size: u32,
    #[compute_dtype]
    pub dtype: DType,
    /// Longest context the deployment runs. Bounds the context axis only.
    pub max_context_len: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct CandidateBlockMaxKernelInput {
    pub num_queries: u32,
    pub context_len: u32,
}

pub struct CandidateBlockMaxSpec;

impl KernelSpec for CandidateBlockMaxSpec {
    type Config = CandidateBlockMaxKernelConfig;
    type Input = CandidateBlockMaxKernelInput;

    const KIND: KernelKind = "candidate_block_max";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values(CHUNK_QUERY_POINTS),
            context_points(config.max_context_len),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // A streaming pass over rows x window.
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        tail_infeasible(grid)
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            matches!(config.block_size, 8 | 16),
            "block_size must be 8 or 16"
        );
        Ok(())
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
                .with("block_size", config.block_size)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(CandidateBlockMaxKernel, CandidateBlockMaxSpec);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_emits_every_python_arg() {
        let config = CandidateBlockMaxKernelConfig {
            backends: vec!["tilelang"],
            gpu_name: "NVIDIA B200".to_string(),
            compress_ratio: 1,
            block_size: 8,
            dtype: DType::Bf16,
            max_context_len: 131072,
        };
        let grid = SweepGrid::new(vec![Axis::values([140]), Axis::values([131072])]);
        let payloads = CandidateBlockMaxSpec::enumerate(&config, &grid, "tilelang");
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "block_size",
                "compress_ratio",
                "context_len",
                "dtype",
                "num_queries"
            ]
        );
    }
}
