//! Sliding-window K/V append: TensorFold's DFlash2 context update,
//! `draft_attention.append` as `add_taps_streams` (families/qwen3_5/cuda/
//! dflash2.py) calls it for keys and for values of every drafter layer. One
//! `_append` launch writes, for every stream, a fresh tensor holding the last
//! `keep = min(window, context + new)` rows of [old context | new rows].
//!
//! The old window is copied, not extended, so the cost follows the kept rows,
//! not the new ones: on GB10 8 streams x 2047 kept rows took 0.26 ms for 1, 16,
//! 256 or 2048 new rows alike, with or without an old context. The public
//! input is the call's totals `(streams, context rows before, new rows)`; the
//! cache interpolates over `(streams, mean kept rows a stream)`, the mean rows
//! after the append clamped at the window. A ragged batch
//! copies the same bytes as an even one with its mean, and the launch's extra
//! row blocks for the longest stream are masked no-ops.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// Streams appended in one call (`--parallel` up to 16). Past one stream the
/// copy is bandwidth bound and the time linear in streams.
const STREAMS_AXIS: [u32; 6] = [1, 2, 4, 8, 12, 16];
/// Kept rows a stream: a few rows are launch bound (~2 us); the copy reaches
/// its bandwidth by a few hundred rows. Knots past `window` are dropped and
/// `window` itself is always a knot (the steady state).
const KEPT_AXIS: [u32; 8] = [1, 4, 16, 64, 256, 512, 1024, 1536];
/// New rows a grid point appends to each stream (a DFlash2 tree accepts at
/// most 16); the rest of the kept rows come from the old context.
const GRID_NEW_ROWS: u32 = 16;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SlidingWindowKvAppendKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// KV heads of the one tensor (keys or values) a call appends.
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
    /// Rows a stream's context keeps at most (DFlash2: sliding_window - 1).
    pub window: Dim,
}

/// One call: its streams, their context rows before the append and their new
/// rows, each in total over the streams.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SlidingWindowKvAppendKernelInput {
    pub batch_size: u32,
    pub context_len: u32,
    pub num_tokens: u32,
}

impl SweepCoords for SlidingWindowKvAppendKernelInput {
    fn coords(&self) -> Coords {
        if self.batch_size == 0 {
            return Coords::new([f64::NAN; 2]);
        }
        let streams = f64::from(self.batch_size);
        let rows = f64::from(self.context_len) + f64::from(self.num_tokens);
        Coords::new([streams, rows / streams])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["streams", "mean_kept_rows"]
    }
}

fn kept_axis(window: u32) -> Vec<f64> {
    let mut knots: Vec<u32> = KEPT_AXIS.into_iter().filter(|&k| k < window).collect();
    knots.push(window);
    Axis::values(knots)
}

pub struct SlidingWindowKvAppendSpec;

impl KernelSpec for SlidingWindowKvAppendSpec {
    type Config = SlidingWindowKvAppendKernelConfig;
    type Input = SlidingWindowKvAppendKernelInput;

    const KIND: KernelKind = "sliding_window_kv_append";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let window = config.window.get();
        assert!(window >= 1, "window must be at least one row");
        SweepGrid::new(vec![Axis::values(STREAMS_AXIS), kept_axis(window)])
    }

    /// A stream keeps at most `window` rows: the mean rows after the append
    /// clamp there (a prefill's 2048 new rows keep 2047).
    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        let coords = input.coords();
        let [streams, rows] = [coords.as_slice()[0], coords.as_slice()[1]];
        Coords::new([streams, rows.min(f64::from(config.window.get()))])
    }

    /// Time = streams x (copy of one stream's kept rows).
    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    /// Streams are independent programs of one bandwidth-bound launch, so more
    /// than 16 run as 16-stream pieces with the rows split evenly.
    fn off_grid(
        _config: &Self::Config,
        input: &Self::Input,
        _backend: &'static str,
    ) -> OffGrid<Self::Input> {
        let max_streams = STREAMS_AXIS[STREAMS_AXIS.len() - 1];
        let streams = input.batch_size;
        if streams <= max_streams {
            return OffGrid::Cache;
        }
        let share =
            |total: u32, n: u32| (u64::from(total) * u64::from(n) / u64::from(streams)) as u32;
        let mut pieces = Vec::new();
        let (mut done, mut ctx_done, mut new_done) = (0, 0, 0);
        while done < streams {
            let n = (streams - done).min(max_streams);
            let ctx = share(input.context_len, done + n) - ctx_done;
            let new = share(input.num_tokens, done + n) - new_done;
            pieces.push(SlidingWindowKvAppendKernelInput {
                batch_size: n,
                context_len: ctx,
                num_tokens: new,
            });
            done += n;
            ctx_done += ctx;
            new_done += new;
        }
        OffGrid::Launches(pieces)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|streams, kept| {
            let kept = kept as u32;
            let new = kept.min(GRID_NEW_ROWS);
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("dtype", config.dtype.as_str())
                .with("window", config.window.get())
                .with("batch_size", streams as u32)
                .with("context_len", kept - new)
                .with("num_tokens", new)
        })
    }
}

register_kernel!(SlidingWindowKvAppendKernel, SlidingWindowKvAppendSpec);

#[cfg(test)]
mod tests {
    use super::*;

    /// The DFlash2 drafter's K (or V) tensor on one GB10.
    fn config() -> SlidingWindowKvAppendKernelConfig {
        SlidingWindowKvAppendKernelConfig {
            backends: vec!["tensorfold_triton"],
            gpu_name: "NVIDIA GB10".into(),
            num_kv_heads: 8.into(),
            head_dim: 128.into(),
            dtype: DType::Bf16,
            window: 2047.into(),
        }
    }

    fn input(
        batch_size: u32,
        context_len: u32,
        num_tokens: u32,
    ) -> SlidingWindowKvAppendKernelInput {
        SlidingWindowKvAppendKernelInput {
            batch_size,
            context_len,
            num_tokens,
        }
    }

    #[test]
    fn input_projects_to_streams_and_mean_kept_rows() {
        // Defect caught: pricing the new rows (a 16-row commit into full
        // windows costs a 2047-row copy) or letting a prefill's 2048 new rows
        // read past the window's knot.
        let coords = |i| SlidingWindowKvAppendSpec::cache_coords(&config(), &i);
        assert_eq!(
            coords(input(8, 8 * 2047, 8 * 16)).as_slice(),
            &[8.0, 2047.0]
        );
        assert_eq!(coords(input(1, 0, 2048)).as_slice(), &[1.0, 2047.0]);
        assert_eq!(coords(input(4, 400, 40)).as_slice(), &[4.0, 110.0]);
    }

    #[test]
    fn grid_ends_at_the_window_and_payload_matches_python_schema() {
        // profiling/kernels/sliding_window_kv_append.py SlidingWindowKvAppendArgs
        // + backend. Defect caught: a kept-rows knot past the window (the
        // runner refuses a context longer than it) or a payload field rename.
        let cfg = config();
        let grid = SlidingWindowKvAppendSpec::sweep_grid(&cfg);
        assert_eq!(grid.axes()[1].last(), Some(&2047.0));
        let specs = SlidingWindowKvAppendSpec::enumerate(&cfg, &grid, "tensorfold_triton");
        assert_eq!(specs.len(), 6 * 9);
        let mut names: Vec<&str> = specs[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "batch_size",
                "context_len",
                "dtype",
                "head_dim",
                "num_kv_heads",
                "num_tokens",
                "window"
            ]
        );
        for spec in &specs {
            let f = spec.fields();
            let ctx = f["context_len"].as_u64().unwrap();
            let new = f["num_tokens"].as_u64().unwrap();
            assert!(new >= 1 && ctx + new <= 2047);
        }
    }

    #[test]
    fn more_than_sixteen_streams_run_as_pieces() {
        // Defect caught: 20 streams extrapolated along the streams axis, or
        // pieces whose rows do not add up to the input's.
        let pieces = match SlidingWindowKvAppendSpec::off_grid(
            &config(),
            &input(20, 20 * 2047, 20 * 5),
            "tensorfold_triton",
        ) {
            OffGrid::Launches(p) => p,
            _ => panic!("20 streams must split"),
        };
        let shapes: Vec<(u32, u32, u32)> = pieces
            .iter()
            .map(|p| (p.batch_size, p.context_len, p.num_tokens))
            .collect();
        assert_eq!(shapes, [(16, 16 * 2047, 80), (4, 4 * 2047, 20)]);
        assert!(matches!(
            SlidingWindowKvAppendSpec::off_grid(&config(), &input(16, 0, 16), "tensorfold_triton"),
            OffGrid::Cache
        ));
    }
}
