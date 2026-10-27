//! Single GEMM kernel: one cached perf model per `(n, k, dtype)` config.
//!
//! Everything generic (build / eval / the `Probe` impl /
//! for-backend loops) lives in `engine::Kernel<S>`. This file declares the
//! GEMM-specific Config / Input, the `KIND` wire string, and the `enumerate`
//! body that lifts (config, sweep coord, backend) to the on-wire
//! `ArgsPayload`. The Python `SingleGemmArgs` dataclass owns the schema.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
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

/// `m` axis for `vllm_cutlass_fp8` (`cutlass_scaled_mm` below SM10x).
///
/// The SM89 dispatch (`scaled_mm_c2x_sm89_fp8_dispatch.cuh`) switches tile
/// config at `next_pow_2(m)` = 16, 32, 64, 128, 256, so a config switch sits
/// between `b` and `b + 1` for each of those. Past 256 the default config tiles
/// M by 128 or 256 rows, and each extra tile row is a step: on an L40S at
/// n=16384, k=5120 time rises 15-22% between `128 j` and `128 j + 1` up to
/// m=1024. The axis is the FP8-tensor one (every power of two to 256 and its
/// `b + 1`, then every multiple of 64) plus every `128 j + 1`.
fn cutlass_fp8_m_axis() -> Vec<f64> {
    let mut m: Vec<u32> = fp8_tensor_m_axis()
        .into_iter()
        .map(|v| v as u32)
        .chain((384..HYBRID_BUCKET_M_MAX).step_by(128).map(|b| b + 1))
        .collect();
    m.sort_unstable();
    m.dedup();
    Axis::values(m)
}

/// Largest `m` the Marlin and CUTLASS FP8 grids measure; past it the call is
/// split (see `off_grid`).
const ADA_M_MAX: u32 = HYBRID_BUCKET_M_MAX;

/// Below this `m` the Marlin axis also brackets the 16-row tail step.
const MARLIN_DENSE_M: u32 = 1024;

/// `m` axis for `vllm_marlin_nvfp4` (vLLM `marlin_gemm`, W4A16).
///
/// `marlin.cu` walks `m` in slices: each launch takes the largest multiple of
/// 64 rows up to `max_par * 64` (1024 when `n > 4096`, 8192 otherwise), and a
/// remainder below 64 rows is one more launch. Every launch streams the whole
/// weight, so time jumps by a weight pass at each `64 j + 1` and falls back at
/// `64 (j + 1)`, where the remainder joins the full slice. The remainder
/// launch itself steps again past 16 rows (one to two 16-row thread blocks):
/// on an L40S at n=34816, k=5120, m=64 takes 0.148 ms, m=65..80 0.224-0.229 ms
/// and m=81 0.297 ms. The axis measures `64 j`, `64 j + 1` and `64 j + 63` to
/// the top, plus `64 j + 16` and `64 j + 17` below `MARLIN_DENSE_M`, where one
/// such step is a large share of the call (under 4% past it). Below 64 rows it
/// brackets the single launch's tile steps at 8, 16, 32 and 48 rows.
fn marlin_m_axis() -> Vec<f64> {
    let mut m: Vec<u32> = [1, 2, 4, 8, 9, 16, 17, 32, 33, 48, 49, 63]
        .into_iter()
        .chain((64..ADA_M_MAX).step_by(64).flat_map(|b| {
            let dense = b < MARLIN_DENSE_M;
            [b, b + 1, b + 63]
                .into_iter()
                .chain(dense.then_some([b + 16, b + 17]).into_iter().flatten())
        }))
        .chain([ADA_M_MAX])
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
            ["vllm_cutlass_fp8"] => return SweepGrid::new(vec![cutlass_fp8_m_axis()]),
            ["vllm_marlin_nvfp4"] => return SweepGrid::new(vec![marlin_m_axis()]),
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

    /// Past the grid, the two Ada backends answer as consecutive calls of at
    /// most `ADA_M_MAX` rows. For Marlin that is the kernel's own structure:
    /// `marlin.cu` already runs `m` as slice launches and `ADA_M_MAX` is a
    /// whole number of slices for any `n`. CUTLASS FP8 is one launch, but at
    /// the top of the grid it is compute bound with flat TFLOPS, so time is
    /// additive in `m`: on an L40S at n=14336, k=5120, m=16384 takes 8.35 ms
    /// against 2 x 4.14 ms at m=8192. The cache's own linear extension follows
    /// the last 64-row segment, whose slope is tile-step noise (it predicted
    /// 0.06 ms there), and holding the edge bandwidth undercounts the weight
    /// term's share (-15%). Every other backend extends its cache.
    fn off_grid(
        _config: &Self::Config,
        input: &Self::Input,
        backend: &'static str,
    ) -> OffGrid<Self::Input> {
        if !matches!(backend, "vllm_marlin_nvfp4" | "vllm_cutlass_fp8") || input.m <= ADA_M_MAX {
            return OffGrid::Cache;
        }
        let mut launches = Vec::new();
        let mut rest = input.m;
        while rest > 0 {
            let piece = rest.min(ADA_M_MAX);
            launches.push(SingleGemmKernelInput { m: piece });
            rest -= piece;
        }
        OffGrid::Launches(launches)
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
        assert!(
            axis.windows(2).all(|w| w[0] < w[1]),
            "axis must be strictly increasing"
        );
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
        assert_eq!(
            fields.get("backend"),
            Some(&Value::from("flashinfer_mxfp8"))
        );
    }

    fn ada_cfg(backend: &'static str, n: u32, k: u32, dtype: DType) -> SingleGemmKernelConfig {
        SingleGemmKernelConfig {
            backends: vec![backend],
            gpu_name: "NVIDIA L40S".to_string(),
            n: n.into(),
            k: k.into(),
            dtype,
        }
    }

    /// The SM89 CUTLASS FP8 dispatch switches tile config at m = 16, 32, 64,
    /// 128, 256, and past 256 every extra 128-row tile is a step; both sides
    /// of every switch and step must be measured.
    #[test]
    fn cutlass_fp8_axis_brackets_sm89_tile_switches() {
        let cfg = ada_cfg("vllm_cutlass_fp8", 16384, 5120, DType::Fp8E4m3);
        let grid = SingleGemmSpec::sweep_grid(&cfg);
        let axis = &grid.axes()[0];
        for b in [16.0, 32.0, 64.0, 128.0, 256.0] {
            assert!(
                axis.contains(&b) && axis.contains(&(b + 1.0)),
                "switch at {b}"
            );
        }
        for j in 3..64 {
            assert!(
                axis.contains(&(128.0 * j as f64 + 1.0)),
                "M tile step {}",
                128 * j
            );
        }
        assert_eq!(axis.first(), Some(&1.0));
        assert_eq!(axis.last(), Some(&8192.0));
        assert!(axis.len() <= 500);
    }

    /// Marlin adds a whole weight pass at 64 j + 1 and drops it at 64 (j + 1):
    /// a missing side makes the linear cache smear one launch across 64 rows.
    #[test]
    fn marlin_axis_brackets_every_remainder_launch() {
        let cfg = ada_cfg("vllm_marlin_nvfp4", 34816, 5120, DType::Nvfp4E2m1);
        let grid = SingleGemmSpec::sweep_grid(&cfg);
        let axis = &grid.axes()[0];
        let has = |v: u32| axis.contains(&(v as f64));
        for j in 1..128 {
            assert!(
                has(64 * j) && has(64 * j + 1) && has(64 * j + 63),
                "edge {}",
                64 * j
            );
            if 64 * j < 1024 {
                assert!(has(64 * j + 16) && has(64 * j + 17), "tail step {}", 64 * j);
            }
        }
        assert!(axis.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(axis.first(), Some(&1.0));
        assert_eq!(axis.last(), Some(&8192.0));
        assert!(axis.len() <= 500, "{} points", axis.len());
    }

    /// Past the grid, both Ada backends split into whole measured calls; other
    /// backends keep the cache's own extrapolation.
    #[test]
    fn ada_off_grid_splits_into_measured_calls() {
        use crate::timing::kernels::engine::OffGrid;
        let input = SingleGemmKernelInput { m: 20000 };
        for (backend, dtype) in [
            ("vllm_marlin_nvfp4", DType::Nvfp4E2m1),
            ("vllm_cutlass_fp8", DType::Fp8E4m3),
        ] {
            let cfg = ada_cfg(backend, 34816, 5120, dtype);
            let OffGrid::Launches(launches) = SingleGemmSpec::off_grid(&cfg, &input, backend)
            else {
                panic!("m past the {backend} grid must split");
            };
            let sizes: Vec<u32> = launches.iter().map(|l| l.m).collect();
            assert_eq!(sizes, [8192, 8192, 3616]);
        }
        let torch = ada_cfg("torch", 16384, 5120, DType::Bf16);
        assert!(matches!(
            SingleGemmSpec::off_grid(&torch, &input, "torch"),
            OffGrid::Cache
        ));
    }

    #[test]
    fn ada_backends_forward_the_python_wire_schema() {
        for (backend, dtype, wire) in [
            ("vllm_cutlass_fp8", DType::Fp8E4m3, "fp8_e4m3"),
            ("vllm_marlin_nvfp4", DType::Nvfp4E2m1, "nvfp4_e2m1"),
        ] {
            let cfg = ada_cfg(backend, 5120, 6144, dtype);
            let grid = SingleGemmSpec::sweep_grid(&cfg);
            let payloads = SingleGemmSpec::enumerate(&cfg, &grid, backend);
            assert_eq!(payloads.len(), grid.axes()[0].len());
            let fields = payloads[0].fields();
            assert_eq!(fields.len(), 5);
            assert_eq!(fields.get("backend"), Some(&Value::from(backend)));
            assert_eq!(fields.get("dtype"), Some(&Value::from(wire)));
            assert_eq!(fields.get("n"), Some(&Value::from(5120_u32)));
            assert_eq!(fields.get("k"), Some(&Value::from(6144_u32)));
        }
    }
}
