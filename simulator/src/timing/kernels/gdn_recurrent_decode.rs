//! Qwen Gated DeltaNet fused recurrent-decode kernel.
//!
//! The model geometry and dtypes are static config identity. Decode batch size
//! is the sole physical runtime interpolation axis. `state_dtype` remains an
//! explicit profiler/cache key, but is not a standard KV capability axis.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnRecurrentDecodeKernelConfig {
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

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct GdnRecurrentDecodeKernelInput {
    pub batch_size: u32,
}

pub struct GdnRecurrentDecodeSpec;

impl KernelSpec for GdnRecurrentDecodeSpec {
    type Config = GdnRecurrentDecodeKernelConfig;
    type Input = GdnRecurrentDecodeKernelInput;

    const KIND: KernelKind = "gdn_recurrent_decode";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::pow2(0, 8)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|batch_size| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", batch_size as u32)
                .with("num_qk_heads", config.num_qk_heads.get())
                .with("num_value_heads", config.num_value_heads.get())
                .with("key_head_dim", config.key_head_dim.get())
                .with("value_head_dim", config.value_head_dim.get())
                .with("dtype", config.dtype.as_str())
                .with("state_dtype", config.state_dtype.as_str())
        })
    }
}

register_kernel!(GdnRecurrentDecodeKernel, GdnRecurrentDecodeSpec);

#[cfg(test)]
mod tests {
    use super::{
        GdnRecurrentDecodeKernelConfig, GdnRecurrentDecodeKernelInput, GdnRecurrentDecodeSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> GdnRecurrentDecodeKernelConfig {
        GdnRecurrentDecodeKernelConfig {
            backends: vec!["torch", "vllm_triton"],
            gpu_name: "NVIDIA H200".to_string(),
            num_qk_heads: 16.into(),
            num_value_heads: 32.into(),
            key_head_dim: 128.into(),
            value_head_dim: 128.into(),
            dtype: DType::Bf16,
            state_dtype: DType::Fp32,
        }
    }

    #[test]
    fn config_identity_and_description_match_the_python_handoff() {
        let cfg = config();

        assert_eq!(GdnRecurrentDecodeSpec::KIND, "gdn_recurrent_decode");
        assert_eq!(
            GdnRecurrentDecodeSpec::profile_kind(),
            "gdn_recurrent_decode"
        );
        assert_eq!(cfg.backends(), &["torch", "vllm_triton"]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.num_qk_heads, 16);
        assert_eq!(cfg.num_value_heads, 32);
        assert_eq!(cfg.key_head_dim, 128);
        assert_eq!(cfg.value_head_dim, 128);
        assert_eq!(cfg.dtype, DType::Bf16);
        assert_eq!(cfg.state_dtype, DType::Fp32);
        assert_eq!(
            cfg.describe_config(),
            serde_json::json!({
                "backends": ["torch", "vllm_triton"],
                "gpu_name": "NVIDIA H200",
                "num_qk_heads": {"value": 16, "expression": null, "bindings": {}},
                "num_value_heads": {"value": 32, "expression": null, "bindings": {}},
                "key_head_dim": {"value": 128, "expression": null, "bindings": {}},
                "value_head_dim": {"value": 128, "expression": null, "bindings": {}},
                "dtype": "bf16",
                "state_dtype": "fp32",
            })
        );
    }

    #[test]
    fn dtype_tag_exposes_compute_but_not_state_as_kv() {
        let cfg = config();

        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
    }

    #[test]
    fn input_coords_field_names_and_slot_payload_are_exact() {
        let input = GdnRecurrentDecodeKernelInput { batch_size: 32 };

        assert_eq!(&*input.coords(), &[32.0]);
        assert_eq!(
            GdnRecurrentDecodeKernelInput::coord_field_names(),
            &["batch_size"]
        );
        let slot: SlotInput = input.clone().into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"batch_size": 32})
        );

        let round_trip: GdnRecurrentDecodeKernelInput =
            serde_json::from_value(serde_json::to_value(input).unwrap()).unwrap();
        assert_eq!(&*round_trip.coords(), &[32.0]);
    }

    #[test]
    fn sweep_is_exactly_the_frozen_batch_axis() {
        let cfg = config();
        let grid = GdnRecurrentDecodeSpec::sweep_grid(&cfg);

        assert_eq!(grid.axes().len(), 1);
        assert_eq!(
            grid.axes()[0],
            [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
        );
        assert!(GdnRecurrentDecodeSpec::infeasible_mask(&cfg, &grid).is_empty());
    }

    #[test]
    fn every_backend_uses_linear_1d_cache() {
        for backend in ["torch", "vllm_triton"] {
            assert_eq!(
                GdnRecurrentDecodeSpec::cache_kind(backend),
                CacheKind::Cache1DLinear
            );
        }
    }

    #[test]
    fn enumerate_emits_exact_python_fields_and_production_values() {
        let cfg = config();
        let grid = GdnRecurrentDecodeSpec::sweep_grid(&cfg);
        let payloads = GdnRecurrentDecodeSpec::enumerate(&cfg, &grid, "vllm_triton");

        assert_eq!(payloads.len(), 9);
        for payload in &payloads {
            let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
            assert_eq!(
                names,
                [
                    "backend",
                    "batch_size",
                    "dtype",
                    "key_head_dim",
                    "num_qk_heads",
                    "num_value_heads",
                    "state_dtype",
                    "value_head_dim",
                ]
            );
            assert_eq!(payload.fields().len(), 8);
            assert_eq!(payload.backend(), Some("vllm_triton"));
        }

        let first = payloads[0].fields();
        assert_eq!(first.get("backend"), Some(&Value::from("vllm_triton")));
        assert_eq!(first.get("batch_size"), Some(&Value::from(1_u32)));
        assert_eq!(first.get("num_qk_heads"), Some(&Value::from(16_u32)));
        assert_eq!(first.get("num_value_heads"), Some(&Value::from(32_u32)));
        assert_eq!(first.get("key_head_dim"), Some(&Value::from(128_u32)));
        assert_eq!(first.get("value_head_dim"), Some(&Value::from(128_u32)));
        assert_eq!(first.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(first.get("state_dtype"), Some(&Value::from("fp32")));
        assert_eq!(payloads[8].fields()["batch_size"], Value::from(256_u32));
    }

    #[test]
    fn config_serde_round_trip_preserves_backends_dimensions_and_dtypes() {
        let encoded = serde_json::to_value(config()).unwrap();
        let decoded: GdnRecurrentDecodeKernelConfig =
            serde_json::from_value(encoded.clone()).unwrap();

        assert_eq!(decoded, config());
        assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
    }
}
