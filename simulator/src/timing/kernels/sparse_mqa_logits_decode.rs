//! DeepGEMM's FP4 candidate-restricted decode indexer logits over paged keys
//! (`deep_gemm.fp8_fp4_paged_sparse_mqa_logits`, one
//! `sm100_paged_sparse_mqa_logits` launch per consumer layer).
//!
//! DeepInfer's decode candidate consumers (`v4_modules/index_attn.py`
//! `_compute_decode_p_sum`, sparse branch) score each of `batch_size x next_n`
//! rows against the keys of its `candidate_blocks` blocks, read through the
//! row's block table. Coordinates: the requests and their index-key context.
//! Once a row has `candidate_blocks` valid blocks its work stops growing with
//! the context; past the grid the cache extrapolates on the product.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// Requests per decode step on one rank; 148 is B200's SM count.
pub(crate) const DECODE_BATCH_POINTS: [u32; 13] =
    [1, 2, 4, 8, 16, 32, 64, 128, 148, 192, 256, 384, 512];

/// Index keys per request. The candidate width (2048 x 8 = 16384 keys at the
/// V4.1 geometry) is where the sparse work saturates.
const SPARSE_CONTEXT_POINTS: [u32; 12] = [
    8, 64, 512, 2048, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
];

/// Largest single profiling allocation (the scattered paged cache), as
/// `dsa_paged_mqa_logits_decode`.
const MAX_PROFILE_ALLOCATION_BYTES: f64 = 32.0 * 1024.0 * 1024.0 * 1024.0;

/// The context axis up to the first point at or past `max_context_len`.
pub(crate) fn sparse_context_points(max_context_len: u32) -> Vec<f64> {
    let cut = SPARSE_CONTEXT_POINTS
        .iter()
        .position(|&c| c >= max_context_len)
        .map_or(SPARSE_CONTEXT_POINTS.len(), |i| i + 1);
    Axis::values(SPARSE_CONTEXT_POINTS[..cut].iter().copied())
}

/// Cells whose scattered FP4 cache (two pages per used page, `head_dim / 2 + 4`
/// bytes per key) would not fit one profiling allocation.
pub(crate) fn paged_fp4_infeasible(grid: &SweepGrid, head_dim: u32) -> Vec<bool> {
    let bytes_per_key = 2.0 * f64::from(head_dim / 2 + 4);
    grid.expand_2d(|batch_size, context_len| {
        batch_size * context_len * bytes_per_key > MAX_PROFILE_ALLOCATION_BYTES
    })
}

pub(crate) fn validate_candidates(
    block_size: u32,
    candidate_blocks: u32,
    candidate_block_size: u32,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        candidate_blocks % 4 == 0 && (4..=4096).contains(&candidate_blocks),
        "candidate_blocks must be a multiple of 4 in 4..=4096"
    );
    anyhow::ensure!(
        matches!(candidate_block_size, 8 | 16),
        "candidate_block_size must be 8 or 16"
    );
    anyhow::ensure!(
        block_size > 0 && block_size % candidate_block_size == 0,
        "page size must hold whole candidate blocks"
    );
    Ok(())
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparseMqaLogitsDecodeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Query rows per request (verify width).
    pub next_n: u32,
    /// Index keys per cache page (128 / compress ratio in DeepInfer).
    pub block_size: u32,
    pub candidate_blocks: u32,
    /// Index keys per candidate block at this layer's ratio.
    pub candidate_block_size: u32,
    pub num_heads: Dim,
    pub head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
    /// Longest index-key context the deployment runs. Bounds the context axis only.
    pub max_context_len: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct SparseMqaLogitsDecodeKernelInput {
    pub batch_size: u32,
    /// Index keys per request.
    pub context_len: u32,
}

pub struct SparseMqaLogitsDecodeSpec;

impl KernelSpec for SparseMqaLogitsDecodeSpec {
    type Config = SparseMqaLogitsDecodeKernelConfig;
    type Input = SparseMqaLogitsDecodeKernelInput;

    const KIND: KernelKind = "sparse_mqa_logits_decode";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values(DECODE_BATCH_POINTS),
            sparse_context_points(config.max_context_len),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // Every request's rows score min(context, candidates) keys.
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        paged_fp4_infeasible(grid, config.head_dim.get())
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        validate_candidates(
            config.block_size,
            config.candidate_blocks,
            config.candidate_block_size,
        )?;
        anyhow::ensure!(config.next_n >= 1, "next_n must be positive");
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
        grid.expand_2d(|batch_size, context_len| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", batch_size as u32)
                .with("context_len", context_len as u32)
                .with("next_n", config.next_n)
                .with("block_size", config.block_size)
                .with("candidate_blocks", config.candidate_blocks)
                .with("candidate_block_size", config.candidate_block_size)
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(SparseMqaLogitsDecodeKernel, SparseMqaLogitsDecodeSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SparseMqaLogitsDecodeKernelConfig {
        SparseMqaLogitsDecodeKernelConfig {
            backends: vec!["deepgemm_sm100"],
            gpu_name: "NVIDIA B200".to_string(),
            next_n: 6,
            block_size: 128,
            candidate_blocks: 2048,
            candidate_block_size: 8,
            num_heads: 32.into(),
            head_dim: 128.into(),
            dtype: DType::Mxfp4E2m1,
            max_context_len: 1048576,
        }
    }

    #[test]
    fn enumerate_emits_every_python_arg() {
        let config = config();
        assert!(SparseMqaLogitsDecodeSpec::validate_config(&config).is_ok());
        let grid = SweepGrid::new(vec![Axis::values([64]), Axis::values([131072])]);
        let payloads = SparseMqaLogitsDecodeSpec::enumerate(&config, &grid, "deepgemm_sm100");
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
                "head_dim",
                "next_n",
                "num_heads",
            ]
        );
    }

    /// A cell whose scattered cache cannot be allocated must not reach the
    /// profiler; one that fits must.
    #[test]
    fn mask_drops_only_unallocatable_cells() {
        let config = config();
        let grid = SparseMqaLogitsDecodeSpec::sweep_grid(&config);
        let mask = SparseMqaLogitsDecodeSpec::infeasible_mask(&config, &grid);
        let cells = grid.expand_2d(|b, c| (b as u32, c as u32));
        let masked = |b: u32, c: u32| mask[cells.iter().position(|&x| x == (b, c)).unwrap()];
        assert!(!masked(64, 131072));
        assert!(masked(512, 1048576));
    }

    #[test]
    fn page_must_hold_whole_candidate_blocks() {
        let mut config = config();
        config.block_size = 4;
        assert!(SparseMqaLogitsDecodeSpec::validate_config(&config).is_err());
    }
}
