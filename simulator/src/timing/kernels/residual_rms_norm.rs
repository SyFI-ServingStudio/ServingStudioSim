//! Fused residual-add RMSNorm kernel: one cached perf model per
//! `(hidden, dtype)` config.
//!
//! This kind owns a distinct Python profile table and fused launch boundary;
//! it intentionally does not reuse `rms_norm` through `profile_kind()`.
//! `hidden` and `dtype` are static identity, while token count `m` is the
//! runtime one-dimensional interpolation axis.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResidualRmsNormKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct ResidualRmsNormKernelInput {
    pub m: u32,
}

pub struct ResidualRmsNormSpec;

impl KernelSpec for ResidualRmsNormSpec {
    type Config = ResidualRmsNormKernelConfig;
    type Input = ResidualRmsNormKernelInput;

    const KIND: KernelKind = "residual_rms_norm";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
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
                .with("hidden", config.hidden.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(ResidualRmsNormKernel, ResidualRmsNormSpec);

#[cfg(test)]
mod tests {
    use super::{ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput, ResidualRmsNormSpec};
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> ResidualRmsNormKernelConfig {
        ResidualRmsNormKernelConfig {
            backends: vec!["torch", "vllm_cuda"],
            gpu_name: "NVIDIA H200".to_string(),
            hidden: 6144.into(),
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_matches_the_fused_profile_kind() {
        let cfg = config();

        assert_eq!(ResidualRmsNormSpec::KIND, "residual_rms_norm");
        assert_eq!(ResidualRmsNormSpec::profile_kind(), "residual_rms_norm");
        assert_eq!(cfg.backends(), &["torch", "vllm_cuda"]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.hidden, 6144);
        assert_eq!(cfg.dtype, DType::Bf16);
    }

    #[test]
    fn describe_config_preserves_rich_hidden_dim() {
        assert_eq!(
            config().describe_config(),
            serde_json::json!({
                "backends": ["torch", "vllm_cuda"],
                "gpu_name": "NVIDIA H200",
                "hidden": {"value": 6144, "expression": null, "bindings": {}},
                "dtype": "bf16",
            })
        );
    }

    #[test]
    fn input_coords_are_exactly_m() {
        let input = ResidualRmsNormKernelInput { m: 128 };

        assert_eq!(&*input.coords(), &[128.0]);
        assert_eq!(ResidualRmsNormKernelInput::coord_field_names(), &["m"]);
    }

    #[test]
    fn sweep_has_frozen_decode_and_prefill_coverage() {
        let cfg = config();
        let grid = ResidualRmsNormSpec::sweep_grid(&cfg);

        assert_eq!(grid.axes().len(), 1);
        assert_eq!(&grid.axes()[0][..6], &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]);
        assert_eq!(grid.axes()[0].last(), Some(&65536.0));
        assert_eq!(grid.axes()[0].len(), 68);
        assert!(ResidualRmsNormSpec::infeasible_mask(&cfg, &grid).is_empty());
    }

    #[test]
    fn both_backends_use_linear_1d_cache() {
        assert_eq!(
            ResidualRmsNormSpec::cache_kind("torch"),
            CacheKind::Cache1DLinear
        );
        assert_eq!(
            ResidualRmsNormSpec::cache_kind("vllm_cuda"),
            CacheKind::Cache1DLinear
        );
    }

    #[test]
    fn enumerate_matches_python_args_exactly_for_each_backend() {
        let cfg = config();
        let grid = ResidualRmsNormSpec::sweep_grid(&cfg);

        for backend in ["torch", "vllm_cuda"] {
            let payloads = ResidualRmsNormSpec::enumerate(&cfg, &grid, backend);
            assert_eq!(payloads.len(), 68);
            let first = &payloads[0];
            let fields = first.fields();
            let field_names: Vec<&str> = fields.keys().map(String::as_str).collect();

            assert_eq!(field_names, ["backend", "dtype", "hidden", "m"]);
            assert_eq!(fields.len(), 4);
            assert_eq!(fields.get("backend"), Some(&Value::from(backend)));
            assert_eq!(fields.get("m"), Some(&Value::from(1_u32)));
            assert_eq!(fields.get("hidden"), Some(&Value::from(6144_u32)));
            assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
            assert_eq!(first.backend(), Some(backend));
        }
    }

    #[test]
    fn dtype_tags_expose_compute_only() {
        let cfg = config();

        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
    }
}
