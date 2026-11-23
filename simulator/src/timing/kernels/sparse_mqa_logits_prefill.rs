//! DeepGEMM's FP4 candidate-restricted prefill indexer logits
//! (`deep_gemm.fp8_fp4_sparse_mqa_logits`, one `sm100_sparse_mqa_logits`
//! launch per consumer layer and query chunk).
//!
//! Each row scores the keys of its `candidate_blocks` blocks of
//! `candidate_block_size` keys (`v4_modules/index_attn.py`
//! `_compute_prefill_p_sum`, sparse branch). The coordinates are the chunk's
//! rows and the context they are a tail of, as in
//! [`mqa_logits_prefill_chunk`](super::mqa_logits_prefill_chunk); a packed
//! chunk is queried at its
//! [`equivalent_tail_context`](super::mqa_logits_prefill_chunk::equivalent_tail_context).
//! Once a row has `candidate_blocks` valid blocks its work stops growing with
//! the context, so the cache holds the edge past the grid.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mqa_logits_prefill_chunk::{
    context_points, tail_infeasible, CHUNK_QUERY_POINTS,
};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparseMqaLogitsPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub compress_ratio: u32,
    pub candidate_blocks: u32,
    /// Compressed keys per candidate block at this layer's ratio.
    pub candidate_block_size: u32,
    pub num_heads: Dim,
    pub head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
    /// Longest context the deployment runs. Bounds the context axis only.
    pub max_context_len: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct SparseMqaLogitsPrefillKernelInput {
    pub num_queries: u32,
    pub context_len: u32,
}

pub struct SparseMqaLogitsPrefillSpec;

impl KernelSpec for SparseMqaLogitsPrefillSpec {
    type Config = SparseMqaLogitsPrefillKernelConfig;
    type Input = SparseMqaLogitsPrefillKernelInput;

    const KIND: KernelKind = "sparse_mqa_logits_prefill";

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

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.compress_ratio >= 1,
            "compress_ratio must be positive"
        );
        anyhow::ensure!(
            config.candidate_blocks % 4 == 0 && (4..=4096).contains(&config.candidate_blocks),
            "candidate_blocks must be a multiple of 4 in 4..=4096"
        );
        anyhow::ensure!(
            matches!(config.candidate_block_size, 8 | 16),
            "candidate_block_size must be 8 or 16"
        );
        anyhow::ensure!(
            config.max_context_len >= 1,
            "max_context_len must be positive"
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
                .with("candidate_blocks", config.candidate_blocks)
                .with("candidate_block_size", config.candidate_block_size)
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(SparseMqaLogitsPrefillKernel, SparseMqaLogitsPrefillSpec);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_emits_every_python_arg() {
        let config = SparseMqaLogitsPrefillKernelConfig {
            backends: vec!["deepgemm_sm100"],
            gpu_name: "NVIDIA B200".to_string(),
            compress_ratio: 1,
            candidate_blocks: 2048,
            candidate_block_size: 8,
            num_heads: 32.into(),
            head_dim: 128.into(),
            dtype: DType::Mxfp4E2m1,
            max_context_len: 131072,
        };
        assert!(SparseMqaLogitsPrefillSpec::validate_config(&config).is_ok());
        let grid = SweepGrid::new(vec![Axis::values([140]), Axis::values([131072])]);
        let payloads = SparseMqaLogitsPrefillSpec::enumerate(&config, &grid, "deepgemm_sm100");
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
                "head_dim",
                "num_heads",
                "num_queries",
            ]
        );
    }
}
