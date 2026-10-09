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
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

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

/// Largest `m` the MXFP8 grid measures; beyond it the cache extrapolates.
const MXFP8_M_MAX: u32 = 8192;

/// `m` axis for the block-scaled MXFP8 GEMM (`flashinfer_mxfp8`).
///
/// FlashInfer autotunes `mm_mxfp8` per hybrid token bucket
/// (`get_hybrid_num_tokens_buckets`: powers of two to 256, step 256 to 2048,
/// step 512 to 4096, then powers of two) and maps a runtime `m` UP to its
/// bucket, so the tactic is constant on each `(b_prev, b]` and may switch
/// between `b` and `b + 1`. The axis measures every bucket `b` and `b + 1`, so
/// linear interpolation never spans a tactic switch, plus mid-bucket points
/// where a bucket is wider than 128 rows (tile-count steps inside one tactic).
fn mxfp8_m_axis() -> Vec<f64> {
    let buckets: Vec<u32> = (0..=8)
        .map(|i| 1u32 << i)
        .chain((512..=2048).step_by(256))
        .chain((2560..=4096).step_by(512))
        .chain([MXFP8_M_MAX])
        .collect();
    let mut m: Vec<u32> = buckets
        .iter()
        .flat_map(|&b| [b, b + 1])
        .chain((384..=2048).step_by(256))
        .chain((2304..=4096).step_by(512))
        .chain((4608..=MXFP8_M_MAX).step_by(512))
        .filter(|&v| v <= MXFP8_M_MAX)
        .collect();
    m.sort_unstable();
    m.dedup();
    Axis::values(m)
}

impl KernelSpec for SingleGemmSpec {
    type Config = SingleGemmKernelConfig;
    type Input = SingleGemmKernelInput;

    const KIND: KernelKind = "single_gemm";

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        if neuron_only(config) {
            anyhow::ensure!(
                config.dtype == DType::Bf16,
                "Neuron GEMM supports BF16 only"
            );
            anyhow::ensure!(
                config.n.get() > 0
                    && config.k.get() > 0
                    && config.n.get() % 256 == 0
                    && config.k.get() % 256 == 0,
                "Neuron LNC2 GEMM feature dimensions must be positive multiples of256"
            );
            anyhow::ensure!(
                matches!(
                    config.gpu_name.as_str(),
                    "AWS Trainium2 LNC2" | "Trainium2-LNC2"
                ),
                "Neuron timing requires a Trainium2 LNC2 target"
            );
        }
        Ok(())
    }

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        if neuron_only(config) {
            // The initial batch1 architecture selects one last-token hidden
            // vector for logits. Compiling unused full-vocabulary m>1 shapes
            // costs minutes without covering another reachable LM-head input.
            let axis = if config.n == 128256 && config.k == 4096 {
                Axis::values([1u32])
            } else {
                // The entry switches to CTE above 96. Its default LNC2
                // allocation exceeds SBUF for H4096; only TKG is verified.
                Axis::values([1u32, 2, 4, 8, 16, 32, 64, 96])
            };
            return SweepGrid::new(vec![axis]);
        }
        if config.dtype == DType::Mxfp8E4m3 {
            return SweepGrid::new(vec![mxfp8_m_axis()]);
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

    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        if neuron_only(config) {
            if config.n == 128256 && config.k == 4096 {
                assert_eq!(input.m, 1, "Neuron batch1 LM head requires one token row");
            } else {
                assert!(
                    (1..=96).contains(&input.m),
                    "Neuron GEMM TKG supports 1..96 token rows"
                );
            }
        }
        input.coords()
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

fn neuron_only(config: &SingleGemmKernelConfig) -> bool {
    !config.backends.is_empty()
        && config
            .backends
            .iter()
            .all(|backend| *backend == "neuron_nki_qkv")
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
    fn neuron_grid_profiles_verified_tkg_without_expanding_single_token_head() {
        let mut config = SingleGemmKernelConfig {
            backends: vec!["neuron_nki_qkv"],
            gpu_name: "AWS Trainium2 LNC2".into(),
            n: 4096.into(),
            k: 4096.into(),
            dtype: DType::Bf16,
        };
        SingleGemmSpec::validate_config(&config).unwrap();
        assert_eq!(
            SingleGemmSpec::sweep_grid(&config).axes()[0],
            [1., 2., 4., 8., 16., 32., 64., 96.]
        );
        config.n = 6144.into();
        assert_eq!(
            SingleGemmSpec::sweep_grid(&config).axes()[0],
            [1., 2., 4., 8., 16., 32., 64., 96.]
        );
        config.n = 128256.into();
        assert_eq!(SingleGemmSpec::sweep_grid(&config).axes()[0], [1.]);
        config.dtype = DType::Fp16;
        assert!(SingleGemmSpec::validate_config(&config).is_err());
    }

    #[test]
    fn neuron_gemm_rejects_out_of_domain_queries_while_cuda_still_extrapolates() {
        let mut config = SingleGemmKernelConfig {
            backends: vec!["neuron_nki_qkv"],
            gpu_name: "AWS Trainium2 LNC2".into(),
            n: 6144.into(),
            k: 4096.into(),
            dtype: DType::Bf16,
        };
        assert!(std::panic::catch_unwind(|| SingleGemmSpec::cache_coords(
            &config,
            &SingleGemmKernelInput { m: 97 }
        ))
        .is_err());
        config.n = 128256.into();
        assert!(std::panic::catch_unwind(|| SingleGemmSpec::cache_coords(
            &config,
            &SingleGemmKernelInput { m: 2 }
        ))
        .is_err());
        config.backends = vec!["torch"];
        assert_eq!(
            &*SingleGemmSpec::cache_coords(&config, &SingleGemmKernelInput { m: 256 }),
            &[256.]
        );
    }

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
}
