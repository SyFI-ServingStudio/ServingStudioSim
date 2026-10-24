//! Gated DeltaNet state replay: TensorFold's commit of a verify round,
//! `tensorfold.cuda.kernels.gdn.replay` as `_commit`
//! (families/qwen3_5/cuda/forward.py) calls it. One `replay_kernel` launch
//! re-runs the gated delta rule from every stream's committed fp32 state over
//! that stream's accepted window rows, for every GDN layer, and writes the
//! states back. No query, no output.
//!
//! The launch spans the layers, so `num_layers` is Config identity (time is
//! linear in it). The public input is the batch's totals, `(streams, accepted
//! rows)`; the cache interpolates over `(streams, mean rows a stream)`:
//!
//! - every (stream, layer) state is read and written once: a stream of 1-4
//!   rows costs its 48 states' 302 MB at about 225 GB/s on GB10 (1.32 ms), and
//!   the time is linear in streams at a fixed per-stream count;
//! - past 4 rows each block's serial walk adds about 0.035 ms a row a stream;
//! - a ragged batch measured within 0-3% of the even batch with the same
//!   streams and total rows (2-16 streams, rows 1-16 mixed), so the per-stream
//!   lengths are not an input.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// Streams committed in one launch (`--parallel` up to 16). Time is linear in
/// streams, so the knots only bound the interpolation.
const STREAMS_AXIS: [u32; 6] = [1, 2, 4, 8, 12, 16];
/// Accepted rows a stream. 1-4 is the flat, bandwidth-bound floor; past it the
/// serial walk grows linearly (GB10, one stream: 4 1.32 ms, 6 1.40, 8 1.44,
/// 16 1.72, 32 2.24, 64 3.41, 128 5.90). Past 16 rows only a lone stream's
/// copy chain, up to `attention.MAX_NODES` = 128.
const ROWS_AXIS: [u32; 11] = [1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128];
/// Rows a stream in a multi-stream commit (a DFlash2 tree has 16 rows).
const MULTI_STREAM_MAX_ROWS: u32 = 16;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnStateReplayKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// GDN layers the one launch commits (all the GPU holds).
    pub num_layers: Dim,
    pub num_qk_heads: Dim,
    pub num_value_heads: Dim,
    pub key_head_dim: Dim,
    pub value_head_dim: Dim,
    /// Saved keys and values.
    #[compute_dtype]
    pub dtype: DType,
    /// Recurrent state and gates.
    pub state_dtype: DType,
}

/// One commit: the streams and their accepted rows in total (each stream at
/// least one).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnStateReplayKernelInput {
    pub batch_size: u32,
    pub num_tokens: u32,
}

impl SweepCoords for GdnStateReplayKernelInput {
    fn coords(&self) -> Coords {
        if self.batch_size == 0 {
            return Coords::new([f64::NAN; 2]);
        }
        let streams = f64::from(self.batch_size);
        Coords::new([streams, f64::from(self.num_tokens) / streams])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["streams", "mean_accepted_rows"]
    }
}

/// `total` rows over `streams`, the first `total % streams` one longer (the
/// profiler's split).
fn even_split(total: u32, streams: u32) -> impl Iterator<Item = u32> {
    let (base, extra) = (total / streams, total % streams);
    (0..streams).map(move |s| base + u32::from(s < extra))
}

pub struct GdnStateReplaySpec;

impl KernelSpec for GdnStateReplaySpec {
    type Config = GdnStateReplayKernelConfig;
    type Input = GdnStateReplayKernelInput;

    const KIND: KernelKind = "gdn_state_replay";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert!(
            config.num_value_heads.get() % config.num_qk_heads.get() == 0,
            "num_value_heads must be a multiple of num_qk_heads"
        );
        SweepGrid::new(vec![Axis::values(STREAMS_AXIS), Axis::values(ROWS_AXIS)])
    }

    /// Time = streams x (cost of one stream's rows): the bilinear cross term is
    /// the physics. Past the grid the input runs as pieces (`off_grid`).
    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    /// Several streams of more than 16 rows do not occur: only a lone stream
    /// commits a copy chain past the drafter's 16-row tree.
    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_2d(|streams, rows| streams > 1.0 && rows > f64::from(MULTI_STREAM_MAX_ROWS))
    }

    /// Streams are independent blocks of one launch and their time adds
    /// (linear in streams on GB10), so a commit past the grid runs as pieces:
    /// more than 16 streams as 16-stream pieces, and several streams averaging
    /// more than 16 rows (no such commit in TensorFold, only masked cells) as
    /// one piece a stream.
    fn off_grid(
        _config: &Self::Config,
        input: &Self::Input,
        _backend: &'static str,
    ) -> OffGrid<Self::Input> {
        let max_streams = STREAMS_AXIS[STREAMS_AXIS.len() - 1];
        let streams = input.batch_size;
        let long = streams > 1 && input.num_tokens > streams * MULTI_STREAM_MAX_ROWS;
        if streams == 0 || (streams <= max_streams && !long) {
            return OffGrid::Cache;
        }
        let piece = if long { 1 } else { max_streams };
        let rows: Vec<u32> = even_split(input.num_tokens, streams).collect();
        OffGrid::Launches(
            rows.chunks(piece as usize)
                .map(|chunk| GdnStateReplayKernelInput {
                    batch_size: chunk.len() as u32,
                    num_tokens: chunk.iter().sum(),
                })
                .collect(),
        )
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|streams, rows| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_layers", config.num_layers.get())
                .with("batch_size", streams as u32)
                .with("num_qk_heads", config.num_qk_heads.get())
                .with("num_value_heads", config.num_value_heads.get())
                .with("key_head_dim", config.key_head_dim.get())
                .with("value_head_dim", config.value_head_dim.get())
                .with("dtype", config.dtype.as_str())
                .with("state_dtype", config.state_dtype.as_str())
                .with("num_tokens", (streams * rows) as u32)
        })
    }
}

register_kernel!(GdnStateReplayKernel, GdnStateReplaySpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Qwen3.8-27B's GDN layers on one GB10.
    fn config() -> GdnStateReplayKernelConfig {
        GdnStateReplayKernelConfig {
            backends: vec!["tensorfold_cuda"],
            gpu_name: "NVIDIA GB10".into(),
            num_layers: 48.into(),
            num_qk_heads: 16.into(),
            num_value_heads: 48.into(),
            key_head_dim: 128.into(),
            value_head_dim: 128.into(),
            dtype: DType::Bf16,
            state_dtype: DType::Fp32,
        }
    }

    #[test]
    fn input_projects_to_streams_and_mean_rows() {
        // Defect caught: a projection onto total rows (which would price 16
        // streams of one row like one stream of 16) instead of rows a stream.
        let input = GdnStateReplayKernelInput {
            batch_size: 8,
            num_tokens: 68,
        };
        assert_eq!(input.coords().as_slice(), &[8.0, 8.5]);
    }

    #[test]
    fn grid_keeps_long_paths_to_one_stream() {
        // Defect caught: a grid that profiles 16 streams of 128-row chains (no
        // such commit) or one that drops the single-stream copy chains.
        let cfg = config();
        let grid = GdnStateReplaySpec::sweep_grid(&cfg);
        let mask = GdnStateReplaySpec::infeasible_mask(&cfg, &grid);
        let feasible = mask.iter().filter(|&&m| !m).count();
        assert_eq!(feasible, 11 + 5 * 8);
        let specs = GdnStateReplaySpec::enumerate(&cfg, &grid, "tensorfold_cuda");
        let kept: Vec<(u64, u64)> = specs
            .iter()
            .zip(&mask)
            .filter(|(_, &masked)| !masked)
            .map(|(spec, _)| {
                let f = spec.fields();
                (
                    f["batch_size"].as_u64().unwrap(),
                    f["num_tokens"].as_u64().unwrap(),
                )
            })
            .collect();
        assert!(kept.contains(&(1, 128)));
        assert!(kept.contains(&(16, 256)));
        assert!(kept.iter().all(|&(s, t)| s == 1 || t <= 16 * s));
    }

    #[test]
    fn payload_matches_python_schema() {
        // profiling/kernels/gdn_state_replay.py GdnStateReplayArgs + backend.
        let cfg = config();
        let grid = SweepGrid::new(vec![Axis::values([4]), Axis::values([6])]);
        let payload = &GdnStateReplaySpec::enumerate(&cfg, &grid, "tensorfold_cuda")[0];
        let fields = payload.fields();
        let mut names: Vec<&str> = fields.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "batch_size",
                "dtype",
                "key_head_dim",
                "num_layers",
                "num_qk_heads",
                "num_tokens",
                "num_value_heads",
                "state_dtype",
                "value_head_dim"
            ]
        );
        assert_eq!(fields.get("num_layers"), Some(&Value::from(48)));
        assert_eq!(fields.get("num_tokens"), Some(&Value::from(24)));
        assert_eq!(fields.get("state_dtype"), Some(&Value::from("fp32")));
    }

    #[test]
    fn commits_past_the_grid_run_as_pieces() {
        // Defect caught: 20 streams extrapolated along the streams axis, or
        // two long chains read off masked cells, instead of measured pieces
        // whose rows add up to the input's.
        let pieces = |batch_size, num_tokens| {
            let input = GdnStateReplayKernelInput {
                batch_size,
                num_tokens,
            };
            match GdnStateReplaySpec::off_grid(&config(), &input, "tensorfold_cuda") {
                OffGrid::Launches(p) => p
                    .iter()
                    .map(|i| (i.batch_size, i.num_tokens))
                    .collect::<Vec<_>>(),
                OffGrid::Cache => Vec::new(),
                OffGrid::Bandwidth(_) => panic!("no bandwidth policy"),
            }
        };
        assert_eq!(pieces(20, 100), [(16, 80), (4, 20)]);
        assert_eq!(pieces(2, 81), [(1, 41), (1, 40)]);
        assert!(pieces(16, 256).is_empty());
        assert!(pieces(1, 128).is_empty());
    }
}
