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

impl KernelSpec for RmsNormSpec {
    type Config = RmsNormKernelConfig;
    type Input = RmsNormKernelInput;

    const KIND: KernelKind = "rms_norm";

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        if neuron_only(config) {
            anyhow::ensure!(
                config.dtype == DType::Bf16,
                "Neuron RMSNorm supports BF16 only"
            );
            anyhow::ensure!(
                config.hidden.get() > 0,
                "Neuron RMSNorm hidden width must be positive"
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
            // Small compiled shapes have discontinuous native timings, so
            // measure each integer through 16 instead of smoothing them.
            return SweepGrid::new(vec![Axis::values([
                1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128,
            ])]);
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

fn neuron_only(config: &RmsNormKernelConfig) -> bool {
    !config.backends.is_empty()
        && config
            .backends
            .iter()
            .all(|backend| *backend == "neuron_torch_rms")
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
    fn neuron_norm_profiles_small_queries_without_changing_cuda_grid() {
        let mut config = RmsNormKernelConfig {
            backends: vec!["neuron_torch_rms"],
            gpu_name: "AWS Trainium2 LNC2".into(),
            hidden: 4096.into(),
            dtype: DType::Bf16,
        };
        RmsNormSpec::validate_config(&config).unwrap();
        assert_eq!(
            RmsNormSpec::sweep_grid(&config).axes()[0],
            [
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 32., 64.,
                128.
            ]
        );
        config.backends = vec!["flashinfer"];
        assert_eq!(
            RmsNormSpec::sweep_grid(&config).axes()[0],
            crate::timing::sweep::Axis::token_axis()
        );
    }

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
}
