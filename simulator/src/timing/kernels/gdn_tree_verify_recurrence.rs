//! Gated DeltaNet recurrence over speculative draft trees: TensorFold's
//! `tensorfold.cuda.kernels.gdn.tree` as `multi_tree_forward`
//! (families/qwen3_5/cuda/forward.py) calls it once per GDN layer over every
//! decoding stream's window, one `tree_kernel` launch. Each window row steps the
//! fp32 delta rule along its root-to-row path from its stream's committed state
//! (read only) and writes one read-out.
//!
//! The input is `(streams, total window rows, slot class)`. The slot class is the
//! kernel variant TensorFold's `schedule` picks from the launch's largest
//! shared-memory state count (rule and a reference implementation:
//! `profiling/runners/attention/gdn_tree_verify_recurrence_reference.py`); on
//! GB10 equal rows cost 1.2-1.35x a chain in `slots_1_2` and about 1.8x in
//! `slots_3_4`, so it is a cache axis, not a detail. The cache projects to
//! `(class, streams, mean rows a stream)`: a ragged launch costs what the
//! uniform one with its total rows does (GB10, within ~5%).
//!
//! The class axis holds the classes a window of at most 16 rows a stream can
//! reach (a 16-row tree needs at most 5 states). The input's class always lands
//! on an axis point, so the axis never interpolates. `slots_9_16` and
//! `slots_17_32` need far larger trees and are priced as `slots_5_8`, a lower
//! bound.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// The `tree_kernel` variant of a launch: the class of the largest number of
/// shared-memory states any stream's tree needs (gdn.cu `dispatch_tree`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GdnTreeSlotClass {
    /// Every window is a chain: no state is kept, 8 value rows a warp.
    #[serde(rename = "chain")]
    Chain,
    /// 1-2 states (a star, a spine with leaves): 4 value rows a warp.
    #[serde(rename = "slots_1_2")]
    Slots1To2,
    /// 3-4 states (a 16-row binary heap): 2 value rows a warp.
    #[serde(rename = "slots_3_4")]
    Slots3To4,
    /// 5-8 states: 2 value rows a warp, 8 slots of shared memory.
    #[serde(rename = "slots_5_8")]
    Slots5To8,
    /// 9-16 states: 2 warps a block.
    #[serde(rename = "slots_9_16")]
    Slots9To16,
    /// 17-32 states: 1 warp a block.
    #[serde(rename = "slots_17_32")]
    Slots17To32,
}

impl GdnTreeSlotClass {
    /// The Python `slot_class` value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chain => "chain",
            Self::Slots1To2 => "slots_1_2",
            Self::Slots3To4 => "slots_3_4",
            Self::Slots5To8 => "slots_5_8",
            Self::Slots9To16 => "slots_9_16",
            Self::Slots17To32 => "slots_17_32",
        }
    }

    /// Rows of the runner's smallest tree in this class (its seed); `None` for
    /// the classes the runner does not profile.
    fn min_rows(self) -> Option<u32> {
        match self {
            Self::Chain => Some(1),
            Self::Slots1To2 => Some(3),
            Self::Slots3To4 => Some(8),
            Self::Slots5To8 => Some(16),
            Self::Slots9To16 | Self::Slots17To32 => None,
        }
    }

    /// Position on the class axis; past `slots_5_8` it clamps (see module doc).
    fn axis_index(self) -> f64 {
        match self {
            Self::Chain => 0.0,
            Self::Slots1To2 => 1.0,
            Self::Slots3To4 => 2.0,
            Self::Slots5To8 | Self::Slots9To16 | Self::Slots17To32 => 3.0,
        }
    }
}

/// The profiled classes, in axis order.
const CLASS_AXIS: [GdnTreeSlotClass; 4] = [
    GdnTreeSlotClass::Chain,
    GdnTreeSlotClass::Slots1To2,
    GdnTreeSlotClass::Slots3To4,
    GdnTreeSlotClass::Slots5To8,
];
/// Streams verified in one launch (`--parallel` up to 16). Every stream adds
/// 48 x (128 / rows-a-warp x 4) blocks, so GB10's 48 SMs see whole extra
/// waves between these points and the time stays near-linear in them.
const STREAMS_AXIS: [u32; 8] = [1, 2, 3, 4, 6, 8, 12, 16];
/// Mean window rows a stream. Each block walks its stream's rows serially, so
/// time grows linearly in rows; past 16 only a lone stream occurs (copy chains
/// up to 128 rows, trees of up to 24 at batch 1).
const ROWS_AXIS: [u32; 12] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128];
/// Rows a stream in a multi-stream window (DFlash2 trees of at most 16 nodes).
const MULTI_STREAM_MAX_ROWS: u32 = 16;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnTreeVerifyRecurrenceKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qk_heads: Dim,
    pub num_value_heads: Dim,
    pub key_head_dim: Dim,
    pub value_head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
    pub state_dtype: DType,
}

/// One launch: its streams, their window rows in total, and its slot class.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GdnTreeVerifyRecurrenceKernelInput {
    pub batch_size: u32,
    pub num_tokens: u32,
    pub slot_class: GdnTreeSlotClass,
}

impl SweepCoords for GdnTreeVerifyRecurrenceKernelInput {
    fn coords(&self) -> Coords {
        if self.batch_size == 0 {
            return Coords::new([f64::NAN; 3]);
        }
        let streams = f64::from(self.batch_size);
        Coords::new([
            self.slot_class.axis_index(),
            streams,
            f64::from(self.num_tokens) / streams,
        ])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["slot_class", "batch_size", "mean_window_rows"]
    }
}

/// A grid cell no launch of uniform-or-lopsided rows can reach: the class's
/// seed tree plus one row for every other stream must fit, and several
/// streams never carry more than 16 rows each.
fn infeasible(class: f64, streams: f64, rows: f64) -> bool {
    let need = CLASS_AXIS[class as usize]
        .min_rows()
        .expect("profiled classes have a seed");
    let tokens = streams * rows;
    tokens - f64::from(need) < streams - 1.0
        || (streams > 1.0 && rows > f64::from(MULTI_STREAM_MAX_ROWS))
}

pub struct GdnTreeVerifyRecurrenceSpec;

impl KernelSpec for GdnTreeVerifyRecurrenceSpec {
    type Config = GdnTreeVerifyRecurrenceKernelConfig;
    type Input = GdnTreeVerifyRecurrenceKernelInput;

    const KIND: KernelKind = "gdn_tree_verify_recurrence";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert!(
            config.num_value_heads.get() % config.num_qk_heads.get() == 0,
            "num_value_heads must be a multiple of num_qk_heads"
        );
        SweepGrid::new(vec![
            Axis::arithmetic(0, CLASS_AXIS.len() as u32 - 1, 1),
            Axis::values(STREAMS_AXIS),
            Axis::values(ROWS_AXIS),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_3d(infeasible)
    }

    /// More streams than the grid: the launch's blocks already fill GB10 many
    /// times over at 16 streams and each stream adds its own blocks, so it runs
    /// as grid-sized pieces of the same class, rows split evenly.
    fn off_grid(
        _config: &Self::Config,
        input: &Self::Input,
        _backend: &'static str,
    ) -> OffGrid<Self::Input> {
        let max_streams = STREAMS_AXIS[STREAMS_AXIS.len() - 1];
        if input.batch_size <= max_streams {
            return OffGrid::Cache;
        }
        let (base, extra) = (
            input.num_tokens / input.batch_size,
            input.num_tokens % input.batch_size,
        );
        let mut pieces = Vec::new();
        let mut first = 0;
        while first < input.batch_size {
            let streams = max_streams.min(input.batch_size - first);
            let extra_here = extra.saturating_sub(first).min(streams);
            pieces.push(GdnTreeVerifyRecurrenceKernelInput {
                batch_size: streams,
                num_tokens: base * streams + extra_here,
                slot_class: input.slot_class,
            });
            first += streams;
        }
        OffGrid::Launches(pieces)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_3d(|class, streams, rows| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", streams as u32)
                .with("num_tokens", (streams * rows) as u32)
                .with("num_qk_heads", config.num_qk_heads.get())
                .with("num_value_heads", config.num_value_heads.get())
                .with("key_head_dim", config.key_head_dim.get())
                .with("value_head_dim", config.value_head_dim.get())
                .with("dtype", config.dtype.as_str())
                .with("state_dtype", config.state_dtype.as_str())
                .with("slot_class", CLASS_AXIS[class as usize].as_str())
        })
    }
}

register_kernel!(GdnTreeVerifyRecurrenceKernel, GdnTreeVerifyRecurrenceSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Qwen3.8-27B's GDN layers on one GB10 (16 key, 48 value heads, 128).
    fn config() -> GdnTreeVerifyRecurrenceKernelConfig {
        GdnTreeVerifyRecurrenceKernelConfig {
            backends: vec!["tensorfold_cuda"],
            gpu_name: "NVIDIA GB10".into(),
            num_qk_heads: 16.into(),
            num_value_heads: 48.into(),
            key_head_dim: 128.into(),
            value_head_dim: 128.into(),
            dtype: DType::Bf16,
            state_dtype: DType::Fp32,
        }
    }

    #[test]
    fn input_projects_to_class_streams_and_mean_rows() {
        // Defect caught: a class axis that does not land on the profiled point
        // (interpolating between variants), or classes past slots_5_8 that run
        // off the grid instead of taking its last class.
        let input: GdnTreeVerifyRecurrenceKernelInput = serde_json::from_value(
            serde_json::json!({"batch_size": 8, "num_tokens": 92, "slot_class": "slots_3_4"}),
        )
        .unwrap();
        assert_eq!(input.coords().as_slice(), &[2.0, 8.0, 11.5]);
        let big = GdnTreeVerifyRecurrenceKernelInput {
            batch_size: 1,
            num_tokens: 200,
            slot_class: GdnTreeSlotClass::Slots17To32,
        };
        assert_eq!(big.coords().as_slice()[0], 3.0);
        assert_eq!(
            serde_json::to_value(GdnTreeSlotClass::Slots1To2).unwrap(),
            Value::from("slots_1_2")
        );
    }

    #[test]
    fn grid_keeps_what_a_launch_can_reach() {
        // Defect caught: profiling cells the runner refuses (a class's seed
        // tree that does not fit) or multi-stream windows past 16 rows, or
        // dropping the single-stream copy chains.
        let cfg = config();
        let grid = GdnTreeVerifyRecurrenceSpec::sweep_grid(&cfg);
        let mask = GdnTreeVerifyRecurrenceSpec::infeasible_mask(&cfg, &grid);
        let specs = GdnTreeVerifyRecurrenceSpec::enumerate(&cfg, &grid, "tensorfold_cuda");
        assert_eq!(specs.len(), mask.len());
        let kept: Vec<&ArgsPayload> = specs
            .iter()
            .zip(&mask)
            .filter_map(|(spec, &masked)| (!masked).then_some(spec))
            .collect();
        assert!(kept.len() <= 500, "{} feasible cells", kept.len());
        let field = |spec: &ArgsPayload, name: &str| spec.fields()[name].clone();
        let need = |class: &Value| match class.as_str().unwrap() {
            "chain" => 1,
            "slots_1_2" => 3,
            "slots_3_4" => 8,
            _ => 16,
        };
        for spec in &kept {
            let streams = field(spec, "batch_size").as_u64().unwrap();
            let tokens = field(spec, "num_tokens").as_u64().unwrap();
            assert!(tokens >= need(&field(spec, "slot_class")) + streams - 1);
            assert!(streams == 1 || tokens <= 16 * streams);
        }
        assert!(kept
            .iter()
            .any(|spec| field(spec, "num_tokens") == Value::from(128)
                && field(spec, "batch_size") == Value::from(1)
                && field(spec, "slot_class") == Value::from("chain")));
        assert!(kept
            .iter()
            .any(|spec| field(spec, "batch_size") == Value::from(16)
                && field(spec, "slot_class") == Value::from("slots_5_8")));
    }

    #[test]
    fn payload_matches_python_schema() {
        // profiling/kernels/gdn_tree_verify_recurrence.py GdnTreeVerifyRecurrenceArgs + backend.
        let grid = SweepGrid::new(vec![
            Axis::values([2]),
            Axis::values([4]),
            Axis::values([16]),
        ]);
        let payload =
            &GdnTreeVerifyRecurrenceSpec::enumerate(&config(), &grid, "tensorfold_cuda")[0];
        let mut names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "batch_size",
                "dtype",
                "key_head_dim",
                "num_qk_heads",
                "num_tokens",
                "num_value_heads",
                "slot_class",
                "state_dtype",
                "value_head_dim"
            ]
        );
        assert_eq!(payload.fields()["num_tokens"], Value::from(64));
        assert_eq!(payload.fields()["slot_class"], Value::from("slots_3_4"));
        assert_eq!(payload.fields()["state_dtype"], Value::from("fp32"));
    }

    #[test]
    fn more_streams_than_the_grid_run_as_pieces() {
        // Defect caught: 20 streams extrapolated along the streams axis, or
        // pieces that lose rows or change the class.
        let input = GdnTreeVerifyRecurrenceKernelInput {
            batch_size: 20,
            num_tokens: 230,
            slot_class: GdnTreeSlotClass::Slots1To2,
        };
        match GdnTreeVerifyRecurrenceSpec::off_grid(&config(), &input, "tensorfold_cuda") {
            OffGrid::Launches(pieces) => {
                let shape: Vec<(u32, u32)> = pieces
                    .iter()
                    .map(|p| (p.batch_size, p.num_tokens))
                    .collect();
                assert_eq!(shape, [(16, 16 * 11 + 10), (4, 44)]);
                assert!(pieces
                    .iter()
                    .all(|p| p.slot_class == GdnTreeSlotClass::Slots1To2));
            }
            _ => panic!("expected pieces"),
        }
    }
}
