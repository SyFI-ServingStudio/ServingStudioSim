//! Speculative-tree verify attention: TensorFold's
//! `tensorfold.cuda.kernels.attention.attention` as `multi_tree_forward`
//! (families/qwen3_5/cuda/forward.py) calls it once per softmax-attention
//! layer over every decoding stream's draft window. Each window row attends to
//! its stream's committed keys and to its own root-to-row path. The call is
//! three Triton launches over all streams together (`_shared` committed
//! chunks, `_tail` last chunk plus path, `_merge`); the profiler counts all
//! three. The plan (host work and one `_paths` launch, once a forward) is not
//! part of it.
//!
//! The public input is the exact launch, `(window rows, committed keys)` per
//! stream. The cache projects it to `(streams, mean rows a stream, mean
//! committed keys a stream)` and interpolates over uniform batches of that
//! shape. The projection keeps the launch's two costs:
//!
//! - every stream's programs share one launch, and the committed-key reads
//!   are bandwidth bound: on GB10 ragged batches of 2-16 streams (rows 1-16,
//!   committed 0-131072) all ran at 175-205 GB/s of committed K and V, the
//!   same as uniform batches, so the total (= streams x mean) committed keys
//!   sets the time;
//! - `_tail` and `_merge` run one program per row and KV head, so the total
//!   rows (= streams x mean rows) set their time. Rows past 16 only occur in a
//!   single stream (copy chains up to 128 rows); there `_shared` turns compute
//!   bound and time grows with the rows.
//!
//! The tree's parent structure is not an input: it only masks `_tail` loads,
//! and same-row chains and bushy trees measure within repeat noise.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// Streams verified in one call (`--parallel` up to 16).
const STREAMS_AXIS: [u32; 5] = [1, 2, 4, 8, 16];
/// Window rows a stream. `_tail` / `_merge` launch one program per row and KV
/// head: a lone stream's 4 x rows tail programs fill GB10's 48 SMs at 12 rows
/// and take a second wave from 13 (3000 keys: 0.114 ms at 12 rows, 0.140 at
/// 13, flat to 17), so 12 | 13 straddle the step. `_shared` launches one
/// program per 32 (row, head) pairs; past 16 rows (single-stream copy chains,
/// up to `attention.MAX_NODES` = 128) it is compute bound.
const ROWS_AXIS: [u32; 10] = [1, 2, 4, 8, 12, 13, 16, 32, 64, 128];
/// Rows a stream in a multi-stream window (DFlash2 trees of at most 16 nodes).
const MULTI_STREAM_MAX_ROWS: u32 = 16;
/// Committed keys a stream, linear (bandwidth bound) past a few thousand keys
/// to the 131072-token context. Below that:
///
/// - under 512 keys no stream has a whole chunk and `_shared` does not
///   launch. 384 is the last count where a window of up to 128 rows also
///   keeps its path in one tail chunk, so 0..384 is the no-`_shared`,
///   one-tail-chunk segment (511 measured up to 2.2x the time at 512: two
///   tail chunks a row);
/// - a lone stream's `_shared` programs (whole chunks x 32-pair query tiles x
///   KV heads) fill the SMs in whole waves; 2560 and 3072 mark the first
///   steps past 2048.
///
/// Every knot from 512 up is a whole number of chunks, so it runs one tail
/// chunk a row. A stream runs two when `committed % 512 + rows > 512`, which
/// the cache does not carry: a 512-key sawtooth whose height grows with the
/// rows (96 rows at 3000 keys: 0.73 ms against 0.47-0.48 ms at 2900 and 3072).
const COMMITTED_AXIS: [u32; 13] = [
    0, 384, 512, 1024, 2048, 2560, 3072, 4096, 8192, 16384, 32768, 65536, 131072,
];

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TreeVerifyAttentionKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
    pub o_dtype: DType,
}

/// One call's streams as `(window rows, committed keys)`, the committed keys
/// excluding the window, in stream order.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TreeVerifyAttentionKernelInput {
    pub query_context_pairs: Vec<(u32, u32)>,
}

impl SweepCoords for TreeVerifyAttentionKernelInput {
    fn coords(&self) -> Coords {
        let pairs = &self.query_context_pairs;
        if pairs.is_empty() {
            return Coords::new([f64::NAN; 3]);
        }
        let streams = pairs.len() as f64;
        let rows: f64 = pairs.iter().map(|&(rows, _)| f64::from(rows)).sum();
        let committed: f64 = pairs.iter().map(|&(_, keys)| f64::from(keys)).sum();
        Coords::new([streams, rows / streams, committed / streams])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["streams", "mean_window_rows", "mean_committed_len"]
    }
}

pub struct TreeVerifyAttentionSpec;

impl KernelSpec for TreeVerifyAttentionSpec {
    type Config = TreeVerifyAttentionKernelConfig;
    type Input = TreeVerifyAttentionKernelInput;

    const KIND: KernelKind = "tree_verify_attention";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert!(
            config.num_qo_heads.get() % config.num_kv_heads.get() == 0,
            "num_qo_heads must be a multiple of num_kv_heads"
        );
        SweepGrid::new(vec![
            Axis::values(STREAMS_AXIS),
            Axis::values(ROWS_AXIS),
            Axis::values(COMMITTED_AXIS),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    /// Several streams of more than 16 rows do not occur: only a lone stream
    /// verifies a copy chain past the drafter's 16-node tree.
    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_3d(|streams, rows, _| streams > 1.0 && rows > f64::from(MULTI_STREAM_MAX_ROWS))
    }

    /// More streams than the grid: the launch pools every stream's programs
    /// and its time adds up across streams once the GPU is full (uniform
    /// batches measured 0.96-1.06x the sum of their single streams past 4096
    /// keys), so it runs as grid-sized pieces.
    fn off_grid(
        _config: &Self::Config,
        input: &Self::Input,
        _backend: &'static str,
    ) -> OffGrid<Self::Input> {
        let max_streams = STREAMS_AXIS[STREAMS_AXIS.len() - 1] as usize;
        if input.query_context_pairs.len() <= max_streams {
            return OffGrid::Cache;
        }
        OffGrid::Launches(
            input
                .query_context_pairs
                .chunks(max_streams)
                .map(|pairs| TreeVerifyAttentionKernelInput {
                    query_context_pairs: pairs.to_vec(),
                })
                .collect(),
        )
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_3d(|streams, rows, committed| {
            let pairs = vec![(rows as u32, committed as u32); streams as usize];
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_qo_heads", config.num_qo_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("q_dtype", config.q_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("o_dtype", config.o_dtype.as_str())
                .with("query_context_pairs", serde_json::json!(pairs))
        })
    }
}

register_kernel!(TreeVerifyAttentionKernel, TreeVerifyAttentionSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Qwen3.8-27B's attention layers on one GB10 (24 q / 4 kv heads, 256).
    fn config() -> TreeVerifyAttentionKernelConfig {
        TreeVerifyAttentionKernelConfig {
            backends: vec!["tensorfold_triton"],
            gpu_name: "NVIDIA GB10".into(),
            num_qo_heads: 24.into(),
            num_kv_heads: 4.into(),
            head_dim: 256.into(),
            q_dtype: DType::Bf16,
            kv_dtype: DType::Bf16,
            o_dtype: DType::Bf16,
        }
    }

    #[test]
    fn ragged_batch_projects_to_its_totals() {
        // Defect caught: a projection that drops the batch's total committed
        // keys (e.g. the longest stream instead of the mean), which sets the
        // bandwidth-bound time of a ragged batch.
        let input = TreeVerifyAttentionKernelInput {
            query_context_pairs: vec![(16, 131072), (16, 100), (1, 100), (15, 0)],
        };
        let c = input.coords();
        assert_eq!(c.as_slice(), &[4.0, 12.0, (131072.0 + 200.0) / 4.0]);
        let single = TreeVerifyAttentionKernelInput {
            query_context_pairs: vec![(128, 30000)],
        };
        assert_eq!(single.coords().as_slice(), &[1.0, 128.0, 30000.0]);
    }

    #[test]
    fn grid_keeps_long_windows_to_one_stream() {
        // Defect caught: a grid that profiles 16 streams of 128-row chains
        // (no such launch, and ~200 ms a call) or one that drops the
        // single-stream copy chains.
        let cfg = config();
        let grid = TreeVerifyAttentionSpec::sweep_grid(&cfg);
        let mask = TreeVerifyAttentionSpec::infeasible_mask(&cfg, &grid);
        let feasible = mask.iter().filter(|&&m| !m).count();
        assert_eq!(feasible, 10 * 13 + 4 * 7 * 13);
        assert!(feasible <= 500);
        let specs = TreeVerifyAttentionSpec::enumerate(&cfg, &grid, "tensorfold_triton");
        let kept: Vec<&ArgsPayload> = specs
            .iter()
            .zip(&mask)
            .filter_map(|(spec, &masked)| (!masked).then_some(spec))
            .collect();
        let rows_of = |spec: &ArgsPayload| spec.fields()["query_context_pairs"][0][0].as_u64();
        assert!(kept.iter().any(|spec| rows_of(spec) == Some(128)));
        assert!(kept.iter().all(|spec| {
            let pairs = spec.fields()["query_context_pairs"]
                .as_array()
                .unwrap()
                .len();
            pairs == 1 || rows_of(spec).unwrap() <= 16
        }));
    }

    #[test]
    fn payload_matches_python_schema() {
        // profiling/kernels/tree_verify_attention.py TreeVerifyAttentionArgs + backend.
        let cfg = config();
        let grid = SweepGrid::new(vec![
            Axis::values([3]),
            Axis::values([16]),
            Axis::values([4096]),
        ]);
        let payload = &TreeVerifyAttentionSpec::enumerate(&cfg, &grid, "tensorfold_triton")[0];
        let fields = payload.fields();
        let mut names: Vec<&str> = fields.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "head_dim",
                "kv_dtype",
                "num_kv_heads",
                "num_qo_heads",
                "o_dtype",
                "q_dtype",
                "query_context_pairs"
            ]
        );
        assert_eq!(fields.get("head_dim"), Some(&Value::from(256)));
        assert_eq!(
            fields.get("query_context_pairs"),
            Some(&serde_json::json!([[16, 4096], [16, 4096], [16, 4096]]))
        );
    }

    #[test]
    fn more_streams_than_the_grid_run_as_pieces() {
        // Defect caught: 20 streams extrapolated along the streams axis
        // instead of pooled pieces that the grid measured.
        let input = TreeVerifyAttentionKernelInput {
            query_context_pairs: vec![(16, 5000); 20],
        };
        match TreeVerifyAttentionSpec::off_grid(&config(), &input, "tensorfold_triton") {
            OffGrid::Launches(pieces) => {
                let sizes: Vec<usize> =
                    pieces.iter().map(|p| p.query_context_pairs.len()).collect();
                assert_eq!(sizes, [16, 4]);
            }
            _ => panic!("expected pieces"),
        }
    }
}
