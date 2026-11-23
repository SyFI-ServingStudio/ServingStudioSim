//! DeepGEMM's FP4 dense prefill indexer logits over one query chunk
//! (`deep_gemm.fp8_fp4_mqa_logits`, one `sm100_mqa_logits` launch).
//!
//! DeepInfer's prefill indexer (`v4_modules/index_attn.py`
//! `_compute_prefill_p_sum`, dense branch) scores its query rows in chunks of
//! at most `compute_chunk_size` rows (8192), batched across requests. One
//! profile row is one request's causal-tail chunk: `num_queries` rows ending a
//! `context_len`-token context, row `i` seeing
//! `(context_len - num_queries + i + 1) / compress_ratio` keys.
//!
//! A chunk that packs several requests is queried at the single request with
//! the same rows and the same mean key window,
//! [`equivalent_tail_context`]; the kernel's work is the summed windows.
//! `max_context_len` only bounds the context axis.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// Query rows of one call. DeepInfer caps a chunk at 8192 rows
/// (`min(8192, DEEPINFER_INDEX_ATTN_MEMORY_LIMIT // psum_row_bytes // 8 * 8)`).
pub(crate) const CHUNK_QUERY_POINTS: [u32; 12] =
    [1, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192];

const CONTEXT_POINTS: [u32; 15] = [
    1024, 2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072, 196608, 262144, 393216, 524288,
    786432, 1048576,
];

/// The context axis up to `max_context_len`, plus one point past it: a packed
/// chunk's [`equivalent_tail_context`] exceeds the real context by up to half
/// a chunk.
pub(crate) fn context_points(max_context_len: u32) -> Vec<f64> {
    let cut = CONTEXT_POINTS
        .iter()
        .position(|&c| c >= max_context_len)
        .map_or(CONTEXT_POINTS.len(), |i| (i + 2).min(CONTEXT_POINTS.len()));
    Axis::values(CONTEXT_POINTS[..cut].iter().copied())
}

/// Context of the single request whose `num_queries`-row causal tail has the
/// same mean key window as a chunk of `num_queries` rows packing requests
/// whose rows are `rows_per_request`-row tails of `context_len`-token
/// contexts: `context_len - (rows_per_request - 1)/2 + (num_queries - 1)/2`.
pub fn equivalent_tail_context(num_queries: u32, rows_per_request: u32, context_len: u32) -> u32 {
    let doubled = 2 * u64::from(context_len) + u64::from(num_queries) - u64::from(rows_per_request);
    u32::try_from(doubled.div_ceil(2)).expect("context fits u32")
}

/// Grid cells whose rows cannot be a tail of the context.
pub(crate) fn tail_infeasible(grid: &SweepGrid) -> Vec<bool> {
    grid.expand_2d(|num_queries, context_len| num_queries > context_len)
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MqaLogitsPrefillChunkKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub compress_ratio: u32,
    pub num_heads: Dim,
    pub head_dim: Dim,
    /// Index Q and K element type (`mxfp4_e2m1`).
    #[compute_dtype]
    pub dtype: DType,
    /// Longest context the deployment runs. Bounds the context axis only.
    pub max_context_len: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MqaLogitsPrefillChunkKernelInput {
    pub num_queries: u32,
    pub context_len: u32,
}

pub struct MqaLogitsPrefillChunkSpec;

impl KernelSpec for MqaLogitsPrefillChunkSpec {
    type Config = MqaLogitsPrefillChunkKernelConfig;
    type Input = MqaLogitsPrefillChunkKernelInput;

    const KIND: KernelKind = "mqa_logits_prefill_chunk";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values(CHUNK_QUERY_POINTS),
            context_points(config.max_context_len),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // Rows times window is the work.
        CacheKind::Cache2DLinear(Extrapolation::Product)
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
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(MqaLogitsPrefillChunkKernel, MqaLogitsPrefillChunkSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MqaLogitsPrefillChunkKernelConfig {
        MqaLogitsPrefillChunkKernelConfig {
            backends: vec!["deepgemm_sm100"],
            gpu_name: "NVIDIA B200".to_string(),
            compress_ratio: 2,
            num_heads: 32.into(),
            head_dim: 128.into(),
            dtype: DType::Mxfp4E2m1,
            max_context_len: 131072,
        }
    }

    #[test]
    fn enumerate_emits_every_python_arg() {
        let grid = SweepGrid::new(vec![Axis::values([4096]), Axis::values([131072])]);
        let payloads = MqaLogitsPrefillChunkSpec::enumerate(&config(), &grid, "deepgemm_sm100");
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "compress_ratio",
                "context_len",
                "dtype",
                "head_dim",
                "num_heads",
                "num_queries"
            ]
        );
    }

    /// A packed chunk past the real context must still land inside the grid.
    #[test]
    fn context_axis_reaches_one_point_past_the_deployment_max() {
        let axis = context_points(131072);
        assert_eq!(axis.last(), Some(&196608.0));
        let yoco = equivalent_tail_context(2240, 140, 131072);
        assert!(f64::from(yoco) < 196608.0);
        assert_eq!(context_points(1048576).last(), Some(&1048576.0));
    }

    /// Two 4096-row tails of 131072 keep the mean window of one 8192-row tail
    /// of 133120: rows 0..8191 of the latter see 124929..133120 tokens, mean
    /// 129024.5, as do the two 4096-row tails (126977..131072, twice).
    #[test]
    fn equivalent_context_keeps_the_mean_window() {
        assert_eq!(equivalent_tail_context(8192, 4096, 131072), 133120);
        assert_eq!(equivalent_tail_context(4096, 4096, 131072), 131072);
        assert_eq!(equivalent_tail_context(2240, 140, 131072), 132122);
    }

    #[test]
    fn rows_longer_than_the_context_are_never_profiled() {
        let grid = SweepGrid::new(vec![Axis::values([1024, 2048]), Axis::values([1024])]);
        assert_eq!(tail_infeasible(&grid), vec![false, true]);
    }
}
