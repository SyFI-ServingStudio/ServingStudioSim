//! Single GEMM kernel: one cached perf model per `(n, k, dtype)` config.
//!
//! Everything generic (build / eval / the `Probe` impl /
//! for-backend loops) lives in `engine::Kernel<S>`. This file declares the
//! GEMM-specific Config / Input, the `KIND` wire string, and the `enumerate`
//! body that lifts (config, sweep coord, backend) to the on-wire
//! `ArgsPayload`. The Python `SingleGemmArgs` dataclass owns the schema.

use crate::timing::bridge::{ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{Kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug)]
pub struct SingleGemmKernelConfig {
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub n: u32,
    pub k: u32,
    pub dtype: DType,
}

#[derive(SweepCoords)]
pub struct SingleGemmKernelInput {
    pub m: u32,
}

pub struct SingleGemmSpec;

impl KernelSpec for SingleGemmSpec {
    type Config = SingleGemmKernelConfig;
    type Input = SingleGemmKernelInput;

    const KIND: KernelKind = "single_gemm";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::token_axis()])
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
                .with("n", config.n)
                .with("k", config.k)
                .with("dtype", config.dtype.as_str())
        })
    }
}

pub type SingleGemmKernel = Kernel<SingleGemmSpec>;

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
            n: 128,
            k: 256,
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
            n: 8192,
            k: 8192,
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
            n: 8192,
            k: 8192,
            dtype: DType::Bf16,
        };
        assert_eq!(
            multi.describe_config(),
            r#"backends=["torch", "triton"] gpu_name="H100" n=8192 k=8192 dtype=Bf16"#
        );
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
            n: 4096,
            k: 8192,
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
}
