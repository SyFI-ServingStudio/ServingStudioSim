//! Sliding block attention: TensorFold's DFlash2 drafter attention,
//! `draft_attention.block_attention` as `_layer_fast` (families/qwen3_5/cuda/
//! dflash2.py) calls it once per drafter layer over every live stream's block.
//! One `_block_attention` launch, grid (streams, kv heads): each block row
//! attends its stream's window-clipped context (a separate contiguous tensor
//! a stream) and its whole block (or the earlier rows when causal).
//!
//! The public input is the exact launch, each stream's context keys. On GB10
//! the time is linear in the context (one stream: 7.5 us at 16 keys, 89 us at
//! 2047) and grows with streams once their 8 programs each pass the 48 SMs
//! (2047 keys: 4 streams 0.15 ms, 8 0.32, 16 0.60). Each program walks its
//! stream's context serially, so a batch costs at least what its longest
//! streams cost alone, and at least what an even batch of the same total keys
//! costs. The cache is over uniform batches `(streams, keys a stream)`, and
//! `cache_coords` projects a ragged batch onto the dominant one of
//!
//! - all `S` streams at the mean context (the total-work bound), and
//! - the `k` longest streams at the `k`-th longest context, for each `k`
//!   (one 2047-key stream beside seven 16-key ones costs 0.097 ms, about a
//!   lone 2047-key stream; the mean alone read 0.052),
//!
//! ranked by `max(k, 2.5) x keys`: time is about 35 us a stream past a few
//! streams and flat at the one-stream latency below ~2.5 streams (1 stream
//! 89 us, 2 streams 102 us at 2047 keys).

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// Streams drafted in one call (`--parallel` up to 16).
const STREAMS_AXIS: [u32; 7] = [1, 2, 4, 6, 8, 12, 16];
/// Context keys a stream (64-key tiles; linear). Knots at or past `window`
/// are dropped and `window` itself is always a knot (the steady state).
const CONTEXT_AXIS: [u32; 6] = [1, 64, 256, 512, 1024, 1536];

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SlidingBlockAttentionKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
    /// Query rows of each stream's block (TensorFold on GB10: 16).
    pub block_rows: Dim,
    /// Context keys a row may see at most (DFlash2: sliding_window - 1).
    pub window: Dim,
    /// Whether a block row sees only the block rows up to itself (the
    /// checkpoint's `is_causal`).
    pub causal: bool,
}

/// One call: each live stream's context keys, in stream order.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SlidingBlockAttentionKernelInput {
    pub context_lens: Vec<u32>,
}

/// Below about this many streams the launch sits at one stream's latency.
const LATENCY_FLOOR_STREAMS: f64 = 2.5;

impl SweepCoords for SlidingBlockAttentionKernelInput {
    /// The dominant uniform batch (module doc), contexts unclamped.
    fn coords(&self) -> Coords {
        let streams = self.context_lens.len();
        if streams == 0 {
            return Coords::new([f64::NAN; 2]);
        }
        let mean = self.context_lens.iter().map(|&c| f64::from(c)).sum::<f64>() / streams as f64;
        let rank = |k: f64, keys: f64| k.max(LATENCY_FLOOR_STREAMS) * keys;
        let mut sorted = self.context_lens.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        // Candidates by growing batch, the even batch last; on a tie (below
        // the floor) the larger batch wins, since 2 streams cost more than 1.
        let candidates = sorted
            .iter()
            .enumerate()
            .map(|(i, &keys)| ((i + 1) as f64, f64::from(keys)))
            .chain(std::iter::once((streams as f64, mean)));
        let mut best = (1.0, f64::from(sorted[0]));
        for candidate in candidates {
            if rank(candidate.0, candidate.1) >= rank(best.0, best.1) {
                best = candidate;
            }
        }
        Coords::new([best.0, best.1])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["streams", "context_len"]
    }
}

fn context_axis(window: u32) -> Vec<f64> {
    let mut knots: Vec<u32> = CONTEXT_AXIS.into_iter().filter(|&k| k < window).collect();
    knots.push(window);
    Axis::values(knots)
}

pub struct SlidingBlockAttentionSpec;

impl KernelSpec for SlidingBlockAttentionSpec {
    type Config = SlidingBlockAttentionKernelConfig;
    type Input = SlidingBlockAttentionKernelInput;

    const KIND: KernelKind = "sliding_block_attention";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert!(
            config.num_qo_heads.get() % config.num_kv_heads.get() == 0,
            "num_qo_heads must be a multiple of num_kv_heads"
        );
        SweepGrid::new(vec![
            Axis::values(STREAMS_AXIS),
            context_axis(config.window.get()),
        ])
    }

    /// A context never exceeds the window: clamp the mean there.
    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        let coords = input.coords();
        let [streams, keys] = [coords.as_slice()[0], coords.as_slice()[1]];
        Coords::new([streams, keys.min(f64::from(config.window.get()))])
    }

    /// Time ~ streams x context keys within the grid; past 16 streams the call
    /// runs as pieces (`off_grid`).
    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    /// Streams are independent programs of one launch; more than 16 run as
    /// 16-stream pieces, in stream order.
    fn off_grid(
        _config: &Self::Config,
        input: &Self::Input,
        _backend: &'static str,
    ) -> OffGrid<Self::Input> {
        let max_streams = STREAMS_AXIS[STREAMS_AXIS.len() - 1] as usize;
        if input.context_lens.len() <= max_streams {
            return OffGrid::Cache;
        }
        OffGrid::Launches(
            input
                .context_lens
                .chunks(max_streams)
                .map(|chunk| SlidingBlockAttentionKernelInput {
                    context_lens: chunk.to_vec(),
                })
                .collect(),
        )
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|streams, keys| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_qo_heads", config.num_qo_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("dtype", config.dtype.as_str())
                .with("block_rows", config.block_rows.get())
                .with("window", config.window.get())
                .with("causal", config.causal)
                .with("batch_size", streams as u32)
                .with("context_len", keys as u32)
        })
    }
}

register_kernel!(SlidingBlockAttentionKernel, SlidingBlockAttentionSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// The DFlash2 drafter (Qwen3.8-27B) on one GB10.
    fn config() -> SlidingBlockAttentionKernelConfig {
        SlidingBlockAttentionKernelConfig {
            backends: vec!["tensorfold_triton"],
            gpu_name: "NVIDIA GB10".into(),
            num_qo_heads: 32.into(),
            num_kv_heads: 8.into(),
            head_dim: 128.into(),
            dtype: DType::Bf16,
            block_rows: 16.into(),
            window: 2047.into(),
            causal: false,
        }
    }

    fn coords(context_lens: Vec<u32>) -> Vec<f64> {
        SlidingBlockAttentionSpec::cache_coords(
            &config(),
            &SlidingBlockAttentionKernelInput { context_lens },
        )
        .as_slice()
        .to_vec()
    }

    #[test]
    fn even_batches_project_to_streams_and_context_clamped_at_the_window() {
        // Defect caught: total keys read as a stream's context (8 streams x
        // 2047 keys priced as one 16k-key stream) or a context past the window.
        assert_eq!(coords(vec![2047; 8]), [8.0, 2047.0]);
        assert_eq!(coords(vec![5000, 5000]), [2.0, 2047.0]);
    }

    #[test]
    fn a_ragged_batch_costs_at_least_its_longest_streams() {
        // Defect caught: one 2047-key stream beside seven 16-key ones priced at
        // the mean (270 keys, half the measured 0.097 ms); a many-stream batch
        // still reads as all its streams at the mean.
        let mut lone = vec![16; 7];
        lone.insert(0, 2047);
        assert_eq!(coords(lone), [1.0, 2047.0]);
        assert_eq!(coords(vec![2047, 256, 2047, 256]), [2.0, 2047.0]);
        assert_eq!(coords(vec![2047, 256].repeat(4)), [8.0, 1151.5]);
    }

    #[test]
    fn grid_ends_at_the_window_and_payload_matches_python_schema() {
        // profiling/kernels/sliding_block_attention.py SlidingBlockAttentionArgs
        // + backend. Defect caught: a context knot past the window (the runner
        // refuses it), an empty context, or a field rename.
        let cfg = config();
        let grid = SlidingBlockAttentionSpec::sweep_grid(&cfg);
        assert_eq!(grid.axes()[1].first(), Some(&1.0));
        assert_eq!(grid.axes()[1].last(), Some(&2047.0));
        let specs = SlidingBlockAttentionSpec::enumerate(&cfg, &grid, "tensorfold_triton");
        assert_eq!(specs.len(), 7 * 7);
        let fields = specs[0].fields();
        let mut names: Vec<&str> = fields.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "batch_size",
                "block_rows",
                "causal",
                "context_len",
                "dtype",
                "head_dim",
                "num_kv_heads",
                "num_qo_heads",
                "window"
            ]
        );
        assert_eq!(fields.get("causal"), Some(&Value::from(false)));
        assert_eq!(fields.get("block_rows"), Some(&Value::from(16)));
    }

    #[test]
    fn more_than_sixteen_streams_run_as_pieces() {
        // Defect caught: 24 streams extrapolated along the streams axis, or
        // pieces that drop or reorder streams.
        let input = SlidingBlockAttentionKernelInput {
            context_lens: (0..24).collect(),
        };
        match SlidingBlockAttentionSpec::off_grid(&config(), &input, "tensorfold_triton") {
            OffGrid::Launches(p) => {
                assert_eq!(p.len(), 2);
                assert_eq!(p[0].context_lens, (0..16).collect::<Vec<u32>>());
                assert_eq!(p[1].context_lens, (16..24).collect::<Vec<u32>>());
            }
            _ => panic!("24 streams must split"),
        }
    }
}
