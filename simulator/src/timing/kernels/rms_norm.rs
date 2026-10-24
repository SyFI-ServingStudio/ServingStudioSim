//! RMSNorm kernel: one cached perf model per `(hidden, dtype)` config.
//!
//! Everything generic (build / eval / the `Probe` impl /
//! for-backend loops) lives in `engine::Kernel<S>`. This file declares the
//! RMSNorm-specific Config / Input, the `KIND` wire string, and the
//! `enumerate` body that lifts (config, sweep coord, backend) to the on-wire
//! `ArgsPayload`. The Python `RmsNormArgs` dataclass owns the schema.
//!
//! Shape split (L1 design §8.1): static config is `(hidden, dtype)`; the token
//! count `m` is the runtime sweep axis (`Cache1DLinear`, single token axis).
//!
//! Backends: `flashinfer`, `vllm_cuda`, `vllm_gemma_native`, and
//! `tensorfold_triton` (TensorFold's Triton `_add_rmsnorm` called without a
//! residual, its first layer's norm; bf16 only), and `tensorfold_torch_eager`
//! (the DFlash2 drafter's eager fp32 `_norm`, whose axis also starts at one row).
//! All share the cache.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RmsNormKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct RmsNormKernelInput {
    pub m: u32,
}

pub struct RmsNormSpec;

/// TensorFold's eager fp32 RMSNorm (DFlash2 `_norm`).
pub const TENSORFOLD_TORCH_EAGER: &str = "tensorfold_torch_eager";

impl KernelSpec for RmsNormSpec {
    type Config = RmsNormKernelConfig;
    type Input = RmsNormKernelInput;

    const KIND: KernelKind = "rms_norm";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        if config.backends.contains(&TENSORFOLD_TORCH_EAGER) {
            // The drafter's tap norm runs on a round's accepted rows (from 1) or
            // a prefill chunk's tap rows (up to 2048 a call).
            return SweepGrid::new(vec![Axis::chain([Axis::pow2(0, 4), Axis::token_axis()])]);
        }
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
                .with("hidden", config.hidden.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(RmsNormKernel, RmsNormSpec);

#[cfg(test)]
mod tests {
    use super::{RmsNormKernelConfig, RmsNormKernelInput, RmsNormSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    #[test]
    fn config_identity_includes_backend_and_shape() {
        let cfg = RmsNormKernelConfig {
            backends: vec!["flashinfer"],
            gpu_name: "H100".to_string(),
            hidden: 4096.into(),
            dtype: DType::Fp16,
        };
        assert_eq!(cfg.backends, vec!["flashinfer"]);
        assert_eq!(cfg.hidden, 4096);
        assert_eq!(cfg.dtype, DType::Fp16);
    }

    #[test]
    fn describe_config_renders_tidy_field_list() {
        let cfg = RmsNormKernelConfig {
            backends: vec!["flashinfer"],
            gpu_name: "H100".to_string(),
            hidden: 8192.into(),
            dtype: DType::Bf16,
        };
        assert_eq!(
            cfg.describe_config(),
            serde_json::json!({
                "backends": ["flashinfer"], "gpu_name": "H100",
                "hidden": {"value": 8192, "expression": null, "bindings": {}},
                "dtype": "bf16",
            })
        );

        let multi = RmsNormKernelConfig {
            backends: vec!["flashinfer", "triton"],
            gpu_name: "H100".to_string(),
            hidden: 8192.into(),
            dtype: DType::Bf16,
        };
        assert_eq!(
            multi.describe_config(),
            serde_json::json!({
                "backends": ["flashinfer", "triton"], "gpu_name": "H100",
                "hidden": {"value": 8192, "expression": null, "bindings": {}},
                "dtype": "bf16",
            })
        );
    }

    #[test]
    fn input_sweep_coords_flatten_m_field() {
        let input = RmsNormKernelInput { m: 1024 };
        assert_eq!(&*input.coords(), &[1024.0]);
    }

    #[test]
    fn enumerate_emits_payload_with_all_wire_fields() {
        let cfg = RmsNormKernelConfig {
            backends: vec!["flashinfer"],
            gpu_name: "H100".to_string(),
            hidden: 4096.into(),
            dtype: DType::Bf16,
        };
        let grid = RmsNormSpec::sweep_grid(&cfg);
        let payloads = RmsNormSpec::enumerate(&cfg, &grid, "flashinfer");

        assert!(
            !payloads.is_empty(),
            "token sweep axis must yield at least one point"
        );
        let first = &payloads[0];
        let fields = first.fields();

        // Wire schema: { backend, m, hidden, dtype } — must stay aligned with
        // Python RmsNormArgs in profiling/kernels/rms_norm.py.
        assert_eq!(fields.len(), 4);
        assert_eq!(fields.get("backend"), Some(&Value::from("flashinfer")));
        assert_eq!(fields.get("hidden"), Some(&Value::from(4096_u32)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert!(fields.get("m").and_then(Value::as_u64).is_some());
        assert_eq!(first.backend(), Some("flashinfer"));
    }

    #[test]
    fn tensorfold_triton_payloads_carry_the_backend_and_python_fields() {
        // Catches the TensorFold first-layer rows being requested under another
        // backend or with fields RmsNormArgs does not have.
        let cfg = RmsNormKernelConfig {
            backends: vec!["tensorfold_triton"],
            gpu_name: "NVIDIA GB10".to_string(),
            hidden: 5120.into(),
            dtype: DType::Bf16,
        };
        let grid = RmsNormSpec::sweep_grid(&cfg);
        for payload in RmsNormSpec::enumerate(&cfg, &grid, "tensorfold_triton") {
            let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
            assert_eq!(names, ["backend", "dtype", "hidden", "m"]);
            assert_eq!(payload.backend(), Some("tensorfold_triton"));
        }
    }
}
