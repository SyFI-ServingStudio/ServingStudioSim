//! Qwen Gated DeltaNet fused gated RMS normalization kernel.
//!
//! Hidden width and compute dtype are static config identity. The flattened row
//! count `m` is the sole physical runtime interpolation axis: Qwen maps it from
//! tokens as `m = num_tokens * 32` for its 32 value heads.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnGatedRmsNormKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct GdnGatedRmsNormKernelInput {
    pub m: u32,
}

pub struct GdnGatedRmsNormSpec;

impl KernelSpec for GdnGatedRmsNormSpec {
    type Config = GdnGatedRmsNormKernelConfig;
    type Input = GdnGatedRmsNormKernelInput;

    const KIND: KernelKind = "gdn_gated_rms_norm";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // One to 8192 power-of-two tokens after flattening 32 value heads.
        SweepGrid::new(vec![Axis::pow2(5, 18)])
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

register_kernel!(GdnGatedRmsNormKernel, GdnGatedRmsNormSpec);

#[cfg(test)]
mod tests {
    use super::{GdnGatedRmsNormKernelConfig, GdnGatedRmsNormKernelInput, GdnGatedRmsNormSpec};
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> GdnGatedRmsNormKernelConfig {
        GdnGatedRmsNormKernelConfig {
            backends: vec!["torch", "vllm_triton"],
            gpu_name: "NVIDIA H200".to_string(),
            hidden: 128.into(),
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_description_and_serde_match_the_python_handoff() {
        let cfg = config();

        assert_eq!(GdnGatedRmsNormSpec::KIND, "gdn_gated_rms_norm");
        assert_eq!(GdnGatedRmsNormSpec::profile_kind(), "gdn_gated_rms_norm");
        assert_eq!(cfg.backends(), &["torch", "vllm_triton"]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.hidden, 128);
        assert_eq!(cfg.dtype, DType::Bf16);
        assert_eq!(
            cfg.describe_config(),
            serde_json::json!({
                "backends": ["torch", "vllm_triton"],
                "gpu_name": "NVIDIA H200",
                "hidden": {"value": 128, "expression": null, "bindings": {}},
                "dtype": "bf16",
            })
        );

        let encoded = serde_json::to_value(&cfg).unwrap();
        let decoded: GdnGatedRmsNormKernelConfig = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
    }

    #[test]
    fn dtype_tag_exposes_the_compute_axis_only() {
        let cfg = config();

        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
    }

    #[test]
    fn input_coords_field_name_serde_and_slot_payload_are_exact() {
        let input = GdnGatedRmsNormKernelInput { m: 32 };

        assert_eq!(&*input.coords(), &[32.0]);
        assert_eq!(GdnGatedRmsNormKernelInput::coord_field_names(), &["m"]);
        let slot: SlotInput = input.clone().into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"m": 32})
        );

        let round_trip: GdnGatedRmsNormKernelInput =
            serde_json::from_value(serde_json::to_value(input).unwrap()).unwrap();
        assert_eq!(&*round_trip.coords(), &[32.0]);
    }

    #[test]
    fn sweep_is_exactly_the_frozen_flattened_row_axis_without_mask() {
        let cfg = config();
        let grid = GdnGatedRmsNormSpec::sweep_grid(&cfg);

        assert_eq!(grid.axes().len(), 1);
        assert_eq!(
            grid.axes()[0],
            [
                32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0,
                65536.0, 131072.0, 262144.0,
            ]
        );
        assert_eq!(grid.axes()[0].len(), 14);
        assert!(GdnGatedRmsNormSpec::infeasible_mask(&cfg, &grid).is_empty());
    }

    #[test]
    fn every_backend_uses_linear_1d_cache() {
        for backend in ["torch", "vllm_triton"] {
            assert_eq!(
                GdnGatedRmsNormSpec::cache_kind(backend),
                CacheKind::Cache1DLinear
            );
        }
    }

    #[test]
    fn enumerate_emits_exact_python_fields_and_production_values() {
        let cfg = config();
        let grid = GdnGatedRmsNormSpec::sweep_grid(&cfg);
        let payloads = GdnGatedRmsNormSpec::enumerate(&cfg, &grid, "vllm_triton");

        assert_eq!(payloads.len(), 14);
        for payload in &payloads {
            let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
            assert_eq!(names, ["backend", "dtype", "hidden", "m"]);
            assert_eq!(payload.fields().len(), 4);
            assert_eq!(payload.backend(), Some("vllm_triton"));
        }

        let first = payloads[0].fields();
        assert_eq!(first.get("backend"), Some(&Value::from("vllm_triton")));
        assert_eq!(first.get("m"), Some(&Value::from(32_u32)));
        assert_eq!(first.get("hidden"), Some(&Value::from(128_u32)));
        assert_eq!(first.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(payloads[13].fields()["m"], Value::from(262144_u32));
    }
}
