//! Sparse GQA paged attention with a sigmoid output gate: each query row attends
//! the tokens an indexer selected (complete compressed groups plus its own open
//! causal tail) through the block table. One public call: split-K partials and,
//! when it splits, the merge.
//!
//! The physical coordinates are `(num_rows, context_len)`. The context matters
//! past the selection budget too: selected groups scatter over the whole
//! context, which measured +12% from 2048 to 8192 tokens and +3% more to 131072
//! on B200 (fp8 cache, 24/2/256). `prefill_config` is the framework's launch
//! table switch for steps that hold a prefill row; it changes time only above
//! 1024 rows (bp = 2·rows > 2048 for two K/V heads), where it measured 1.65x.
//! The rows axis brackets each split-K table boundary (bp = rows·kv_heads at
//! 24, 32, 64, 128, 256, 512, 2048), and the context axis holds the selection
//! width `token_topk + compress_ratio - 1` where the selection saturates; both
//! come from the config.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparseGqaPagedAttentionKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_q_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub token_topk: u32,
    pub compress_ratio: u32,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
    pub prefill_config: bool,
}

#[derive(Clone, Debug, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct SparseGqaPagedAttentionKernelInput {
    pub num_rows: u32,
    /// Tokens visible to every row, its own token included.
    pub context_len: u32,
}

/// Base rows: powers of two and their 1.5x midpoints up to the edge.
const BASE_ROWS: [u32; 23] = [
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 6144,
    8192, 16384,
];
/// Upper bounds of the default (SM100) split-K table rows, in base programs
/// `num_rows * num_kv_heads` (fork `ops/qsa.py` `_select_config`).
const SPLIT_K_BOUNDARIES: [u32; 7] = [24, 32, 64, 128, 256, 512, 2048];
const BASE_CONTEXT: [u32; 13] = [
    1, 4, 16, 64, 256, 512, 1024, 2048, 4096, 8192, 32768, 131072, 262144,
];
const MAX_ROWS: u32 = 16384;
const MAX_CONTEXT: u32 = 262144;

/// Base rows plus, per split-K boundary, the last and first row count on
/// either side of it for this config's K/V heads.
fn rows_axis(config: &SparseGqaPagedAttentionKernelConfig) -> Vec<u32> {
    let kv_heads = config.num_kv_heads.get().max(1);
    let mut rows: Vec<u32> = BASE_ROWS.to_vec();
    for boundary in SPLIT_K_BOUNDARIES {
        let last_inside = boundary / kv_heads;
        rows.extend(
            [last_inside, last_inside + 1]
                .into_iter()
                .filter(|&r| r >= 1),
        );
    }
    rows.retain(|&r| r <= MAX_ROWS);
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// Base contexts plus the selection width, past which every row selects
/// `token_topk` tokens.
fn context_axis(config: &SparseGqaPagedAttentionKernelConfig) -> Vec<u32> {
    let selection = config.token_topk + config.compress_ratio.saturating_sub(1);
    let mut context: Vec<u32> = BASE_CONTEXT.to_vec();
    context.push(selection.min(MAX_CONTEXT));
    context.sort_unstable();
    context.dedup();
    context
}

pub struct SparseGqaPagedAttentionSpec;

impl KernelSpec for SparseGqaPagedAttentionSpec {
    type Config = SparseGqaPagedAttentionKernelConfig;
    type Input = SparseGqaPagedAttentionKernelInput;

    const KIND: KernelKind = "sparse_gqa_paged_attention";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let (rows, context) = (rows_axis(config), context_axis(config));
        assert!(
            rows.len() * context.len() <= 500,
            "sparse_gqa_paged_attention grid has {} points, over the 500 ceiling",
            rows.len() * context.len()
        );
        SweepGrid::new(vec![Axis::values(rows), Axis::values(context)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // Rows are independent programs, so time scales with rows at fixed context.
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    /// Past the last profiled context the selection is saturated and the scatter
    /// flat (under 3% from 8192 to 262144), so the edge holds; rows past the
    /// edge run as consecutive full-edge pieces.
    fn off_grid(
        _config: &Self::Config,
        input: &Self::Input,
        _backend: &'static str,
    ) -> OffGrid<Self::Input> {
        let (max_rows, max_context) = (MAX_ROWS, MAX_CONTEXT);
        if input.num_rows <= max_rows && input.context_len <= max_context {
            return OffGrid::Cache;
        }
        let context_len = input.context_len.min(max_context);
        let mut pieces = Vec::new();
        let mut rows = input.num_rows;
        while rows > 0 {
            let piece = rows.min(max_rows);
            pieces.push(SparseGqaPagedAttentionKernelInput {
                num_rows: piece,
                context_len,
            });
            rows -= piece;
        }
        OffGrid::Launches(pieces)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|num_rows, context_len| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_rows", num_rows as u32)
                .with("context_len", context_len as u32)
                .with("num_q_heads", config.num_q_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("token_topk", config.token_topk)
                .with("compress_ratio", config.compress_ratio)
                .with("q_dtype", config.q_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("prefill_config", config.prefill_config)
        })
    }
}

register_kernel!(SparseGqaPagedAttentionKernel, SparseGqaPagedAttentionSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::engine::KernelSpec;

    fn config() -> SparseGqaPagedAttentionKernelConfig {
        SparseGqaPagedAttentionKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".into(),
            num_q_heads: Dim::param("num_q_heads", 24),
            num_kv_heads: Dim::param("num_kv_heads", 2),
            head_dim: Dim::param("head_dim", 256),
            token_topk: 2048,
            compress_ratio: 4,
            q_dtype: DType::Bf16,
            kv_dtype: DType::Fp8E4m3,
            prefill_config: false,
        }
    }

    /// The payload must carry exactly the Python args, or every lookup misses.
    #[test]
    fn enumerate_emits_the_python_args() {
        let cfg = config();
        let grid = SparseGqaPagedAttentionSpec::sweep_grid(&cfg);
        let specs = SparseGqaPagedAttentionSpec::enumerate(&cfg, &grid, "vllm_triton");
        assert_eq!(
            specs.len(),
            rows_axis(&cfg).len() * context_axis(&cfg).len()
        );
        assert!(specs.len() <= 500);
        let names: Vec<&str> = specs[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            names,
            [
                "backend",
                "compress_ratio",
                "context_len",
                "head_dim",
                "kv_dtype",
                "num_kv_heads",
                "num_q_heads",
                "num_rows",
                "prefill_config",
                "q_dtype",
                "token_topk"
            ]
        );
    }

    /// The config-derived axes must reproduce the grid the B200 rows were
    /// profiled on (24/2/256, top-2048 over ratio 4), or every lookup misses.
    #[test]
    fn derived_axes_match_the_profiled_grid() {
        let cfg = config();
        assert_eq!(
            rows_axis(&cfg),
            [
                1, 2, 4, 8, 12, 13, 16, 17, 24, 32, 33, 48, 64, 65, 96, 128, 129, 192, 256, 257,
                384, 512, 768, 1024, 1025, 1536, 2048, 4096, 6144, 8192, 16384,
            ]
        );
        assert_eq!(
            context_axis(&cfg),
            [1, 4, 16, 64, 256, 512, 1024, 2048, 2051, 4096, 8192, 32768, 131072, 262144]
        );
    }

    /// Catches boundaries placed for two K/V heads on another head count: each
    /// split-K switch must have a profiled row count on both sides. (Odd head
    /// counts such as 3 and 5 add 14 new points and reach 504 cells, past the
    /// 500 ceiling; such a config needs its own grid decision.)
    #[test]
    fn rows_bracket_every_split_k_switch_for_any_kv_heads() {
        for kv_heads in [1, 4, 6, 8, 16] {
            let cfg = SparseGqaPagedAttentionKernelConfig {
                num_kv_heads: Dim::param("num_kv_heads", kv_heads),
                ..config()
            };
            let rows = rows_axis(&cfg);
            for boundary in SPLIT_K_BOUNDARIES {
                let inside = rows.iter().filter(|&&r| r * kv_heads <= boundary).max();
                let outside = rows.iter().filter(|&&r| r * kv_heads > boundary).min();
                let (Some(&inside), Some(&outside)) = (inside, outside) else {
                    panic!("kv {kv_heads}: boundary {boundary} not bracketed");
                };
                assert_eq!(outside, inside + 1, "kv {kv_heads}, boundary {boundary}");
            }
            SparseGqaPagedAttentionSpec::sweep_grid(&cfg);
        }
    }

    /// Past the grid the context holds and the rows split into edge pieces.
    #[test]
    fn off_grid_clamps_context_and_splits_rows() {
        let input = SparseGqaPagedAttentionKernelInput {
            num_rows: 20000,
            context_len: 1 << 20,
        };
        match SparseGqaPagedAttentionSpec::off_grid(&config(), &input, "vllm_triton") {
            OffGrid::Launches(pieces) => {
                assert_eq!(
                    pieces.iter().map(|p| p.num_rows).collect::<Vec<_>>(),
                    [16384, 3616]
                );
                assert!(pieces.iter().all(|p| p.context_len == 262144));
            }
            _ => panic!("expected launches"),
        }
    }
}
