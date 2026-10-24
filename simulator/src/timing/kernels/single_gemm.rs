//! Single GEMM kernel: one cached perf model per `(n, k, dtype)` config.
//!
//! Everything generic (build / eval / the `Probe` impl /
//! for-backend loops) lives in `engine::Kernel<S>`. This file declares the
//! GEMM-specific Config / Input, the `KIND` wire string, and the `enumerate`
//! body that lifts (config, sweep coord, backend) to the on-wire
//! `ArgsPayload`. The Python `SingleGemmArgs` dataclass owns the schema.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SingleGemmKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub n: Dim,
    pub k: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct SingleGemmKernelInput {
    pub m: u32,
}

pub struct SingleGemmSpec;

/// Largest `m` the hybrid-bucket grid measures; beyond it the cache extrapolates.
const HYBRID_BUCKET_M_MAX: u32 = 8192;

/// `m` axis for a GEMM autotuned per FlashInfer hybrid token bucket
/// (`flashinfer_mxfp8`'s `mm_mxfp8`, `flashinfer_fp8_tensor`'s `bmm_fp8`).
///
/// `get_hybrid_num_tokens_buckets` yields powers of two to 256, step 256 to
/// 2048, step 512 to 4096, then powers of two, and a runtime `m` maps UP to its
/// bucket (`map_to_hybrid_bucket_uncapped`), so the tactic is constant on each
/// `(b_prev, b]` and may switch between `b` and `b + 1`. The axis measures every
/// bucket `b` and `b + 1`, so linear interpolation never spans a tactic switch,
/// plus mid-bucket points where a bucket is wider than 128 rows (tile-count
/// steps inside one tactic).
fn hybrid_bucket_m_axis() -> Vec<f64> {
    let buckets: Vec<u32> = (0..=8)
        .map(|i| 1u32 << i)
        .chain((512..=2048).step_by(256))
        .chain((2560..=4096).step_by(512))
        .chain([HYBRID_BUCKET_M_MAX])
        .collect();
    let mut m: Vec<u32> = buckets
        .iter()
        .flat_map(|&b| [b, b + 1])
        .chain((384..=2048).step_by(256))
        .chain((2304..=4096).step_by(512))
        .chain((4608..=HYBRID_BUCKET_M_MAX).step_by(512))
        .filter(|&v| v <= HYBRID_BUCKET_M_MAX)
        .collect();
    m.sort_unstable();
    m.dedup();
    Axis::values(m)
}

/// `m` axis for `flashinfer_fp8_tensor`: the hybrid-bucket axis plus every
/// multiple of 64 up to its top.
///
/// Its bucket winner is often a cuBLASLt or CUTLASS kernel whose tile count
/// steps with the actual `m`, not the bucket, so time inside one wide bucket is
/// a staircase (steps every 64-256 rows on B200). A step-8 scan of two
/// projection shapes cut the mean linear-interpolation error from 2.7-3.3%
/// (bucket points only) to 2.0%.
fn fp8_tensor_m_axis() -> Vec<f64> {
    let mut m: Vec<u32> = hybrid_bucket_m_axis()
        .into_iter()
        .map(|v| v as u32)
        .chain((64..=HYBRID_BUCKET_M_MAX).step_by(64))
        .collect();
    m.sort_unstable();
    m.dedup();
    Axis::values(m)
}

/// Rows TensorFold's decode/verify lane GEMM serves: 16 streams of 16 verify
/// rows (single-stream copy chains reach 128). Beyond it the cache extrapolates.
const TENSORFOLD_LANE_M_MAX: u32 = 256;
/// TensorFold's prefill chunk ceiling (`families/qwen3_5/cuda/prefill.py` `CHUNK`).
const TENSORFOLD_CK_M_MAX: u32 = 4096;

/// `m` axis for `tensorfold_lane` (`lane_kernel`, `checkpoint.lane_tile`).
///
/// A block covers 16 rows up to m = 16, 32 rows up to 32, then 64-row blocks
/// side by side (128-row blocks only from 96 SMs), and a partly filled last
/// block runs the row-guarded variant. Time is a staircase over row blocks, so
/// the axis measures each block edge `b` and `b + 1` plus the block's middle,
/// and linear interpolation never spans a step.
fn tensorfold_lane_m_axis() -> Vec<f64> {
    let mut m: Vec<u32> = [1, 2, 4, 8, 12, 16, 17, 24, 32, 33, 48]
        .into_iter()
        .chain((64..=TENSORFOLD_LANE_M_MAX).step_by(32))
        .chain((64..TENSORFOLD_LANE_M_MAX).step_by(64).map(|b| b + 1))
        .collect();
    m.sort_unstable();
    m.dedup();
    Axis::values(m)
}

/// Rows TensorFold's affine 4-bit lane matmul serves: the DFlash2 drafter's
/// block rows and decode taps stay within 16 streams x 16 rows, but a prefill
/// feeds the drafter up to its 2047-row sliding window of taps at once.
const TENSORFOLD_QMM_M_MAX: u32 = 2048;

/// `m` axis for `tensorfold_qmm` (`group_kernel`, `qmm_group.cu` `dispatch`).
///
/// On a GB10 the block covers 16 rows up to m = 16, 32 up to 32, then 64-row
/// blocks side by side (tile 4 to m = 64, tile 5 with 128-column blocks
/// past it), so the row-block edges are those of `tensorfold_lane`: its axis
/// to 256, then every 64-row edge `b` and `b + 1` up to the drafter's window.
fn tensorfold_qmm_m_axis() -> Vec<f64> {
    let top = TENSORFOLD_LANE_M_MAX;
    let mut m: Vec<u32> = tensorfold_lane_m_axis()
        .into_iter()
        .map(|v| v as u32)
        .chain((top + 64..=TENSORFOLD_QMM_M_MAX).step_by(64))
        .chain((top..TENSORFOLD_QMM_M_MAX).step_by(64).map(|b| b + 1))
        .collect();
    m.sort_unstable();
    m.dedup();
    Axis::values(m)
}

/// `m` axis for `tensorfold_ck` (`gemm_kernel`, `gemm_ck.cu` `by_tile`).
///
/// The prompt GEMM tiles rows by 128 on a GB10 (`tile = 1`), so time steps at
/// every multiple of 128; the axis measures each edge `b` and `b + 1` and the
/// tile's middle, with powers of two below the first tile.
fn tensorfold_ck_m_axis() -> Vec<f64> {
    let mut m: Vec<u32> = [1, 2, 4, 8, 16, 32]
        .into_iter()
        .chain((64..=TENSORFOLD_CK_M_MAX).step_by(64))
        .chain((128..TENSORFOLD_CK_M_MAX).step_by(128).map(|b| b + 1))
        .collect();
    m.sort_unstable();
    m.dedup();
    Axis::values(m)
}

/// `m` axis for `tensorfold_b16` (`b16_kernel`, `b16.cu` `by_rows`).
///
/// A warp computes one output column for R rows: R = 1 below 64 rows, 4 below
/// 512, 16 from 512, so the block count grows every R rows and time is a
/// staircase of waves. On a GB10 a wave is 24 rows at R = 1 and 96 rows at
/// R = 4 (verify: 12 us to 24 rows, 17 us at 28; 32 us at 96, 43 us at 100).
/// The axis measures every 4 rows to the verify ceiling of 256, the first row
/// past each wave edge, both sides of the R switches at 64 and 512, then
/// coarser points to 512 and one beyond it for extrapolation.
fn tensorfold_b16_m_axis() -> Vec<f64> {
    let top = TENSORFOLD_LANE_M_MAX;
    let mut m: Vec<u32> = [1, 2, 3, 25, 49, 63, 97, 193, 289, 385, 481]
        .into_iter()
        .chain((4..=top).step_by(4))
        .chain((top + 32..512).step_by(32))
        .chain([511, 512, 1024])
        .collect();
    m.sort_unstable();
    m.dedup();
    Axis::values(m)
}

/// `m` axis for `tensorfold_b16_prompt` (`b16_prompt_kernel`, `b16.cu`
/// `prompt_bm`).
///
/// Rows tile by BM = 32 (64 past about 3000 rows of a 96-wide pair on a GB10,
/// 128 past about 6000), each tile running the whole K chain, so time is a
/// staircase over tiles and waves. The axis measures both sides of every
/// 32-row tile edge up to TensorFold's 4096-row prefill chunk, then two
/// points past it: the pair slows sharply once the rows leave L2 (0.27 ms at
/// 4096 rows, 0.78 ms at 8192), which the last tile segment cannot extrapolate.
fn tensorfold_b16_prompt_m_axis() -> Vec<f64> {
    let mut m: Vec<u32> = [1, 2, 4, 8, 16]
        .into_iter()
        .chain((32..=TENSORFOLD_CK_M_MAX).step_by(32))
        .chain((32..TENSORFOLD_CK_M_MAX).step_by(32).map(|b| b + 1))
        .chain([6144, 8192])
        .collect();
    m.sort_unstable();
    m.dedup();
    Axis::values(m)
}

impl KernelSpec for SingleGemmSpec {
    type Config = SingleGemmKernelConfig;
    type Input = SingleGemmKernelInput;

    const KIND: KernelKind = "single_gemm";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        if config.dtype == DType::Mxfp8E4m3 {
            return SweepGrid::new(vec![hybrid_bucket_m_axis()]);
        }
        // A mixed backend list keeps the generic axis below.
        match config.backends.as_slice() {
            ["flashinfer_fp8_tensor"] => return SweepGrid::new(vec![fp8_tensor_m_axis()]),
            ["tensorfold_lane"] => return SweepGrid::new(vec![tensorfold_lane_m_axis()]),
            ["tensorfold_ck"] => return SweepGrid::new(vec![tensorfold_ck_m_axis()]),
            ["tensorfold_qmm"] => return SweepGrid::new(vec![tensorfold_qmm_m_axis()]),
            ["tensorfold_b16"] => return SweepGrid::new(vec![tensorfold_b16_m_axis()]),
            ["tensorfold_b16_prompt"] => {
                return SweepGrid::new(vec![tensorfold_b16_prompt_m_axis()]);
            }
            _ => {}
        }
        // Dense decode routinely evaluates GEMMs below the shared token axis'
        // m=32 floor. Keep those points measured instead of extrapolating the
        // first [32, 64] segment into the scheduler's common 1..16 batches.
        // This extension is GEMM-local so attention/norm/elementwise grids do
        // not inherit forty unrelated Llama-3 profiling rows.
        SweepGrid::new(vec![Axis::chain([Axis::pow2(0, 4), Axis::token_axis()])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|m| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("m", m as u32)
                .with("n", config.n.get())
                .with("k", config.k.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(SingleGemmKernel, SingleGemmSpec);

#[cfg(test)]
mod tests {
    use super::{SingleGemmKernelConfig, SingleGemmKernelInput, SingleGemmSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    #[test]
    fn config_identity_includes_backend_and_shape() {
        let cfg = SingleGemmKernelConfig {
            backends: vec!["torch"],
            gpu_name: "H100".to_string(),
            n: 128.into(),
            k: 256.into(),
            dtype: DType::Fp16,
        };
        assert_eq!(cfg.backends, vec!["torch"]);
        assert_eq!(cfg.n, 128);
        assert_eq!(cfg.k, 256);
        assert_eq!(cfg.dtype, DType::Fp16);
    }

    #[test]
    fn describe_config_renders_tidy_field_list() {
        let cfg = SingleGemmKernelConfig {
            backends: vec!["torch"],
            gpu_name: "H100".to_string(),
            n: 8192.into(),
            k: 8192.into(),
            dtype: DType::Bf16,
        };
        assert_eq!(
            cfg.describe_config(),
            serde_json::json!({
                "backends": ["torch"], "gpu_name": "H100",
                "n": {"value": 8192, "expression": null, "bindings": {}},
                "k": {"value": 8192, "expression": null, "bindings": {}},
                "dtype": "bf16",
            })
        );

        let multi = SingleGemmKernelConfig {
            backends: vec!["torch", "triton"],
            gpu_name: "H100".to_string(),
            n: 8192.into(),
            k: 8192.into(),
            dtype: DType::Bf16,
        };
        assert_eq!(
            multi.describe_config(),
            serde_json::json!({
                "backends": ["torch", "triton"], "gpu_name": "H100",
                "n": {"value": 8192, "expression": null, "bindings": {}},
                "k": {"value": 8192, "expression": null, "bindings": {}},
                "dtype": "bf16",
            })
        );
    }

    #[test]
    fn symbol_bindings_unions_dim_field_provenance() {
        use crate::timing::Dim;
        let hidden = Dim::param("hidden", 4096);
        let qo = Dim::param("num_qo_heads", 64);
        let head = Dim::param("head_dim", 128);
        let tp = Dim::param("attn_tp", 4);
        let cfg = SingleGemmKernelConfig {
            backends: vec!["torch"],
            gpu_name: "H100".to_string(),
            n: qo / tp * head, // (num_qo_heads/attn_tp)*head_dim
            k: hidden,         // hidden
            dtype: DType::Bf16,
        };
        // The derive unions `bindings()` over the Dim-typed fields (n, k); the
        // non-Dim fields (dtype/backends/gpu_name) contribute nothing.
        let b = cfg.symbol_bindings();
        assert_eq!(b.get("num_qo_heads"), Some(&64));
        assert_eq!(b.get("attn_tp"), Some(&4));
        assert_eq!(b.get("head_dim"), Some(&128));
        assert_eq!(b.get("hidden"), Some(&4096));
        assert_eq!(b.len(), 4);
    }

    #[test]
    fn input_sweep_coords_flatten_m_field() {
        let input = SingleGemmKernelInput { m: 1024 };
        assert_eq!(&*input.coords(), &[1024.0]);
    }

    #[test]
    fn enumerate_emits_payload_with_all_wire_fields() {
        let cfg = SingleGemmKernelConfig {
            backends: vec!["torch"],
            gpu_name: "H100".to_string(),
            n: 4096.into(),
            k: 8192.into(),
            dtype: DType::Bf16,
        };
        let grid = SingleGemmSpec::sweep_grid(&cfg);
        let payloads = SingleGemmSpec::enumerate(&cfg, &grid, "torch");

        assert!(
            !payloads.is_empty(),
            "token sweep axis must yield at least one point"
        );
        let first = &payloads[0];
        let fields = first.fields();

        // Wire schema: { backend, m, n, k, dtype } — must stay aligned with
        // Python SingleGemmArgs in profiling/kernels/single_gemm.py.
        assert_eq!(fields.len(), 5);
        assert_eq!(fields.get("backend"), Some(&Value::from("torch")));
        assert_eq!(fields.get("n"), Some(&Value::from(4096_u32)));
        assert_eq!(fields.get("k"), Some(&Value::from(8192_u32)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert!(fields.get("m").and_then(Value::as_u64).is_some());
        assert_eq!(first.backend(), Some("torch"));
    }

    #[test]
    fn sweep_measures_decode_microbatches_before_the_shared_token_axis() {
        let cfg = SingleGemmKernelConfig {
            backends: vec!["torch"],
            gpu_name: "H100".to_string(),
            n: 4096.into(),
            k: 4096.into(),
            dtype: DType::Bf16,
        };

        let grid = SingleGemmSpec::sweep_grid(&cfg);

        assert_eq!(&grid.axes()[0][..6], &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]);
        assert_eq!(grid.axes()[0].len(), 68);
    }

    fn mxfp8_cfg() -> SingleGemmKernelConfig {
        SingleGemmKernelConfig {
            backends: vec!["flashinfer_mxfp8"],
            gpu_name: "NVIDIA B200".to_string(),
            n: 1792.into(),
            k: 5120.into(),
            dtype: DType::Mxfp8E4m3,
        }
    }

    /// FlashInfer rounds `m` up to its hybrid tuning bucket, so a tactic switch
    /// sits between bucket `b` and `b + 1`. Missing either side makes the
    /// linear cache interpolate across two different kernels.
    #[test]
    fn mxfp8_grid_brackets_every_flashinfer_tuning_bucket() {
        let grid = SingleGemmSpec::sweep_grid(&mxfp8_cfg());
        let axis = &grid.axes()[0];
        let has = |v: u32| axis.contains(&(v as f64));
        let buckets = [
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072,
            3584, 4096, 8192,
        ];
        for b in buckets {
            assert!(has(b), "bucket {b} missing");
            if b < 8192 {
                assert!(has(b + 1), "first m past bucket {b} missing");
            }
        }
        assert!(axis.windows(2).all(|w| w[0] < w[1]), "axis must be strictly increasing");
        assert_eq!(axis.first(), Some(&1.0));
        assert_eq!(axis.last(), Some(&8192.0));
        assert_eq!(axis.len(), 58);
    }

    /// Per-tensor FP8 `bmm_fp8(backend="auto")` is tuned per the same hybrid
    /// buckets, so its axis must bracket every bucket; block-scaled DeepGEMM
    /// shares the `fp8_e4m3` dtype but no tuner, and a mixed backend list must
    /// not inherit one backend's axis.
    #[test]
    fn fp8_tensor_axis_brackets_buckets_and_follows_the_backend() {
        let fp8 = |backends: Vec<&'static str>| SingleGemmKernelConfig {
            backends,
            gpu_name: "NVIDIA B200".to_string(),
            n: 16384.into(),
            k: 5120.into(),
            dtype: DType::Fp8E4m3,
        };
        let hybrid = SingleGemmSpec::sweep_grid(&mxfp8_cfg());
        let generic = SingleGemmSpec::sweep_grid(&fp8(vec!["deepgemm"]));
        let tensor = SingleGemmSpec::sweep_grid(&fp8(vec!["flashinfer_fp8_tensor"]));
        let axis = &tensor.axes()[0];
        assert!(hybrid.axes()[0].iter().all(|m| axis.contains(m)));
        assert!(axis.windows(2).all(|w| w[0] < w[1] && w[1] - w[0] <= 64.0));
        assert_eq!(axis.last(), Some(&8192.0));
        assert!(axis.len() <= 500);
        assert_ne!(generic.axes(), tensor.axes());
        assert_eq!(
            SingleGemmSpec::sweep_grid(&fp8(vec!["flashinfer_fp8_tensor", "deepgemm"])).axes(),
            generic.axes()
        );
    }

    #[test]
    fn mxfp8_enumerate_forwards_the_python_wire_schema() {
        let cfg = mxfp8_cfg();
        let grid = SingleGemmSpec::sweep_grid(&cfg);
        let payloads = SingleGemmSpec::enumerate(&cfg, &grid, "flashinfer_mxfp8");
        assert_eq!(payloads.len(), grid.axes()[0].len());
        let fields = payloads[0].fields();
        assert_eq!(fields.len(), 5);
        assert_eq!(fields.get("dtype"), Some(&Value::from("mxfp8_e4m3")));
        assert_eq!(fields.get("backend"), Some(&Value::from("flashinfer_mxfp8")));
    }

    fn tensorfold_cfg(backend: &'static str, dtype: DType) -> SingleGemmKernelConfig {
        SingleGemmKernelConfig {
            backends: vec![backend],
            gpu_name: "NVIDIA GB10".to_string(),
            n: 5120.into(),
            k: 17408.into(),
            dtype,
        }
    }

    /// TensorFold's GEMM times step at each row block (16 / 32 / 64 rows for
    /// `lane_kernel`, 128 for the prompt `gemm_kernel` on a GB10). An axis
    /// missing either side of an edge would interpolate across a step.
    #[test]
    fn tensorfold_axes_bracket_every_row_block_edge() {
        for (backend, edges, top) in [
            ("tensorfold_lane", vec![16, 32, 64, 128, 192], 256.0),
            ("tensorfold_ck", (1..32).map(|j| 128 * j).collect::<Vec<u32>>(), 4096.0),
        ] {
            let grid = SingleGemmSpec::sweep_grid(&tensorfold_cfg(backend, DType::Nvfp4E2m1));
            let axis = &grid.axes()[0];
            for b in edges {
                assert!(axis.contains(&(b as f64)), "{backend}: edge {b} missing");
                assert!(axis.contains(&(b as f64 + 1.0)), "{backend}: {b} + 1 missing");
            }
            assert_eq!(axis.first(), Some(&1.0));
            assert_eq!(axis.last(), Some(&top));
            assert!(axis.windows(2).all(|w| w[0] < w[1]));
            assert!(axis.len() <= 500);
        }
        // The axis follows the backend, not the dtype: fp8 and nvfp4 share it.
        assert_eq!(
            SingleGemmSpec::sweep_grid(&tensorfold_cfg("tensorfold_lane", DType::Fp8E4m3)).axes(),
            SingleGemmSpec::sweep_grid(&tensorfold_cfg("tensorfold_lane", DType::Nvfp4E2m1)).axes(),
        );
    }

    /// The drafter's lane matmul steps at the same row blocks as `lane_kernel`
    /// and must reach the 2047-row tap window a prefill feeds it.
    #[test]
    fn tensorfold_qmm_axis_brackets_row_blocks_to_the_drafter_window() {
        let cfg = SingleGemmKernelConfig {
            backends: vec!["tensorfold_qmm"],
            gpu_name: "NVIDIA GB10".to_string(),
            n: 5120.into(),
            k: 25600.into(),
            dtype: DType::W4a16AffineG64,
        };
        let grid = SingleGemmSpec::sweep_grid(&cfg);
        let axis = &grid.axes()[0];
        let edges = [16, 32].into_iter().chain((1..32).map(|j| 64 * j));
        for b in edges {
            assert!(axis.contains(&(b as f64)), "edge {b} missing");
            assert!(axis.contains(&(b as f64 + 1.0)), "{b} + 1 missing");
        }
        assert_eq!(axis.first(), Some(&1.0));
        assert_eq!(axis.last(), Some(&2048.0));
        assert!(axis.windows(2).all(|w| w[0] < w[1]));
        assert!(axis.len() <= 500);
        let payloads = SingleGemmSpec::enumerate(&cfg, &grid, "tensorfold_qmm");
        assert_eq!(payloads.len(), axis.len());
        let fields = payloads[0].fields();
        assert_eq!(fields.len(), 5);
        assert_eq!(fields.get("dtype"), Some(&Value::from("w4a16_affine_g64")));
        assert_eq!(fields.get("backend"), Some(&Value::from("tensorfold_qmm")));
    }

    #[test]
    fn tensorfold_enumerate_forwards_the_python_wire_schema() {
        let cfg = tensorfold_cfg("tensorfold_ck", DType::Fp8E4m3);
        let grid = SingleGemmSpec::sweep_grid(&cfg);
        let payloads = SingleGemmSpec::enumerate(&cfg, &grid, "tensorfold_ck");
        assert_eq!(payloads.len(), grid.axes()[0].len());
        let fields = payloads[0].fields();
        assert_eq!(fields.len(), 5);
        assert_eq!(fields.get("dtype"), Some(&Value::from("fp8_e4m3")));
        assert_eq!(fields.get("backend"), Some(&Value::from("tensorfold_ck")));
        assert_eq!(fields.get("k"), Some(&Value::from(17408_u32)));
    }

    /// `b16_kernel` adds a row block every 1 / 4 rows (R = 1 below 64, 4 to 512):
    /// the axis must step at most 4 rows through the verify range and keep both
    /// sides of the R switch at 64, or interpolation smooths over a wave step.
    #[test]
    fn tensorfold_b16_axis_steps_every_row_block_through_verify() {
        let cfg = SingleGemmKernelConfig {
            backends: vec!["tensorfold_b16"],
            gpu_name: "NVIDIA GB10".to_string(),
            n: 96.into(),
            k: 5120.into(),
            dtype: DType::Bf16,
        };
        let grid = SingleGemmSpec::sweep_grid(&cfg);
        let axis = &grid.axes()[0];
        let verify: Vec<f64> = axis.iter().copied().filter(|&m| m <= 256.0).collect();
        assert!(verify.windows(2).all(|w| w[1] - w[0] <= 4.0));
        for m in [
            1.0, 25.0, 63.0, 64.0, 97.0, 193.0, 256.0, 511.0, 512.0, 1024.0,
        ] {
            assert!(axis.contains(&m), "m = {m} missing");
        }
        assert!(axis.windows(2).all(|w| w[0] < w[1]));
        assert!(axis.len() <= 500);
        let payloads = SingleGemmSpec::enumerate(&cfg, &grid, "tensorfold_b16");
        let fields = payloads[0].fields();
        assert_eq!(fields.get("n"), Some(&Value::from(96_u32)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("backend"), Some(&Value::from("tensorfold_b16")));
    }

    /// `b16_prompt_kernel` tiles rows by 32 below ~3000 rows: the axis must
    /// bracket every 32-row edge to the 4096-row prefill chunk.
    #[test]
    fn tensorfold_b16_prompt_axis_brackets_every_tile_edge() {
        let cfg = SingleGemmKernelConfig {
            backends: vec!["tensorfold_b16_prompt"],
            gpu_name: "NVIDIA GB10".to_string(),
            n: 96.into(),
            k: 5120.into(),
            dtype: DType::Bf16,
        };
        let grid = SingleGemmSpec::sweep_grid(&cfg);
        let axis = &grid.axes()[0];
        for b in (1..128).map(|j| 32 * j) {
            assert!(axis.contains(&(b as f64)), "edge {b} missing");
            assert!(axis.contains(&(b as f64 + 1.0)), "{b} + 1 missing");
        }
        assert_eq!(axis.first(), Some(&1.0));
        assert!(axis.contains(&4096.0));
        assert_eq!(axis.last(), Some(&8192.0));
        assert!(axis.windows(2).all(|w| w[0] < w[1]));
        assert!(axis.len() <= 500);
    }
}
