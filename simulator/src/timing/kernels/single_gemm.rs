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

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Deserialize)]
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

impl KernelSpec for SingleGemmSpec {
    type Config = SingleGemmKernelConfig;
    type Input = SingleGemmKernelInput;

    const KIND: KernelKind = "single_gemm";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
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
        // Every field in declaration order, no struct-name/braces wrapper.
        assert_eq!(
            cfg.describe_config(),
            r#"backends=["torch"] gpu_name="H100" n=8192 k=8192 dtype=Bf16"#
        );

        // A Vec field renders via `{:?}` — standard bracketed, comma-separated.
        let multi = SingleGemmKernelConfig {
            backends: vec!["torch", "triton"],
            gpu_name: "H100".to_string(),
            n: 8192.into(),
            k: 8192.into(),
            dtype: DType::Bf16,
        };
        assert_eq!(
            multi.describe_config(),
            r#"backends=["torch", "triton"] gpu_name="H100" n=8192 k=8192 dtype=Bf16"#
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
}
