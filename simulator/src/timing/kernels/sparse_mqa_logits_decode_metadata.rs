//! DeepGEMM's schedule for the paged candidate-restricted decode indexer
//! logits (`deep_gemm.get_paged_sparse_mqa_logits_metadata`, one
//! `sm100_sparse_mqa_logits_metadata` launch).
//!
//! DeepInfer caches it per candidate group for the step
//! (`index_attn.py` `_get_decode_metadata`), so it runs once per group, not
//! per consumer layer. Coordinates as
//! [`sparse_mqa_logits_decode`](super::sparse_mqa_logits_decode).

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::sparse_mqa_logits_decode::{
    paged_fp4_infeasible, sparse_context_points, validate_candidates, DECODE_BATCH_POINTS,
};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

/// The check launch builds the 128-wide FP4 index-K cache.
const CHECK_HEAD_DIM: u32 = 128;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparseMqaLogitsDecodeMetadataKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub next_n: u32,
    pub block_size: u32,
    pub candidate_blocks: u32,
    pub candidate_block_size: u32,
    #[compute_dtype]
    pub dtype: DType,
    /// Longest index-key context the deployment runs. Bounds the context axis only.
    pub max_context_len: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct SparseMqaLogitsDecodeMetadataKernelInput {
    pub batch_size: u32,
    pub context_len: u32,
}

pub struct SparseMqaLogitsDecodeMetadataSpec;

impl KernelSpec for SparseMqaLogitsDecodeMetadataSpec {
    type Config = SparseMqaLogitsDecodeMetadataKernelConfig;
    type Input = SparseMqaLogitsDecodeMetadataKernelInput;

    const KIND: KernelKind = "sparse_mqa_logits_decode_metadata";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values(DECODE_BATCH_POINTS),
            sparse_context_points(config.max_context_len),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Clamp)
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        paged_fp4_infeasible(grid, CHECK_HEAD_DIM)
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        validate_candidates(
            config.block_size,
            config.candidate_blocks,
            config.candidate_block_size,
        )?;
        anyhow::ensure!(config.next_n >= 1, "next_n must be positive");
        Ok(())
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|batch_size, context_len| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", batch_size as u32)
                .with("context_len", context_len as u32)
                .with("next_n", config.next_n)
                .with("block_size", config.block_size)
                .with("candidate_blocks", config.candidate_blocks)
                .with("candidate_block_size", config.candidate_block_size)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(
    SparseMqaLogitsDecodeMetadataKernel,
    SparseMqaLogitsDecodeMetadataSpec
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_emits_every_python_arg() {
        let config = SparseMqaLogitsDecodeMetadataKernelConfig {
            backends: vec!["deepgemm_sm100"],
            gpu_name: "NVIDIA B200".to_string(),
            next_n: 6,
            block_size: 128,
            candidate_blocks: 2048,
            candidate_block_size: 8,
            dtype: DType::Mxfp4E2m1,
            max_context_len: 131072,
        };
        assert!(SparseMqaLogitsDecodeMetadataSpec::validate_config(&config).is_ok());
        let grid = SweepGrid::new(vec![Axis::values([64]), Axis::values([131072])]);
        let payloads =
            SparseMqaLogitsDecodeMetadataSpec::enumerate(&config, &grid, "deepgemm_sm100");
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "batch_size",
                "block_size",
                "candidate_block_size",
                "candidate_blocks",
                "context_len",
                "dtype",
                "next_n",
            ]
        );
    }
}
