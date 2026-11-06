//! Gated DeltaNet fused prefill post-convolution preparation kernel.
//!
//! Model head geometry and compute dtype are static config identity. The
//! physical token count is the sole runtime interpolation axis. The explicit
//! token-17 sample protects the first measured launch-count cliff after the
//! production kernel's 16-token block; later fidelity work owns any further
//! breakpoint decisions.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnPrefillPostConvKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qk_heads: Dim,
    pub num_value_heads: Dim,
    pub key_head_dim: Dim,
    pub value_head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct GdnPrefillPostConvKernelInput {
    pub num_tokens: u32,
}

pub struct GdnPrefillPostConvSpec;

impl KernelSpec for GdnPrefillPostConvSpec {
    type Config = GdnPrefillPostConvKernelConfig;
    type Input = GdnPrefillPostConvKernelInput;

    const KIND: KernelKind = "gdn_prefill_post_conv";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::chain([
            Axis::pow2(0, 4),
            Axis::values([17]),
            Axis::pow2(5, 18),
        ])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    /// The kernel indexes the mixed QKV rows as `token * stride + column` in
    /// int32 (`fused_gdn_prefill_post_conv.py`), so a launch whose last offset
    /// (`num_tokens * stride - 1`) passes `i32::MAX` reads out of bounds: for
    /// Qwen3.8 (stride 10240) the 262144 knot faults with an illegal address on
    /// H100. Such steps cannot run, so their cells are not profiled.
    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        let stride = 2 * u64::from(config.num_qk_heads.get() * config.key_head_dim.get())
            + u64::from(config.num_value_heads.get() * config.value_head_dim.get());
        grid.expand_1d(|num_tokens| num_tokens as u64 * stride > 1 << 31)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
                .with("num_qk_heads", config.num_qk_heads.get())
                .with("num_value_heads", config.num_value_heads.get())
                .with("key_head_dim", config.key_head_dim.get())
                .with("value_head_dim", config.value_head_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(GdnPrefillPostConvKernel, GdnPrefillPostConvSpec);

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{
        GdnPrefillPostConvKernelConfig, GdnPrefillPostConvKernelInput, GdnPrefillPostConvSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> GdnPrefillPostConvKernelConfig {
        GdnPrefillPostConvKernelConfig {
            backends: vec!["torch", "vllm_triton"],
            gpu_name: "NVIDIA H200".to_string(),
            num_qk_heads: 16.into(),
            num_value_heads: 32.into(),
            key_head_dim: 128.into(),
            value_head_dim: 128.into(),
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_description_serde_and_routing_match_python() {
        let cfg = config();

        assert_eq!(GdnPrefillPostConvSpec::KIND, "gdn_prefill_post_conv");
        assert_eq!(
            GdnPrefillPostConvSpec::profile_kind(),
            "gdn_prefill_post_conv"
        );
        assert_eq!(cfg.backends(), &["torch", "vllm_triton"]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.num_qk_heads, 16);
        assert_eq!(cfg.num_value_heads, 32);
        assert_eq!(cfg.key_head_dim, 128);
        assert_eq!(cfg.value_head_dim, 128);
        assert_eq!(cfg.dtype, DType::Bf16);
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
            })
        );

        let encoded = serde_json::to_value(&cfg).unwrap();
        let decoded: GdnPrefillPostConvKernelConfig =
            serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
    }

    #[test]
    fn dtype_tag_exposes_compute_and_no_kv_axis() {
        let cfg = config();

        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
    }

    #[test]
    fn input_coords_field_name_serde_and_slot_payload_are_physical() {
        let input = GdnPrefillPostConvKernelInput { num_tokens: 128 };

        assert_eq!(&*input.coords(), &[128.0]);
        assert_eq!(
            GdnPrefillPostConvKernelInput::coord_field_names(),
            &["num_tokens"]
        );
        let slot: SlotInput = input.clone().into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_tokens": 128})
        );

        let round_trip: GdnPrefillPostConvKernelInput =
            serde_json::from_value(serde_json::to_value(input).unwrap()).unwrap();
        assert_eq!(&*round_trip.coords(), &[128.0]);
    }

    #[test]
    fn sweep_is_exact_ordered_unique_twenty_point_axis_with_l17_cliff() {
        let cfg = config();
        let grid = GdnPrefillPostConvSpec::sweep_grid(&cfg);
        let expected = [
            1.0, 2.0, 4.0, 8.0, 16.0, 17.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0,
            4096.0, 8192.0, 16384.0, 32768.0, 65536.0, 131072.0, 262144.0,
        ];

        assert_eq!(grid.axes().len(), 1);
        assert_eq!(grid.axes()[0], expected);
        assert_eq!(grid.axes()[0].len(), 20);
        assert_eq!(grid.axes()[0].first(), Some(&1.0));
        assert_eq!(grid.axes()[0].last(), Some(&262144.0));
        assert!(grid.axes()[0].windows(2).all(|pair| pair[0] < pair[1]));
        let unique: HashSet<u64> = grid.axes()[0].iter().map(|value| value.to_bits()).collect();
        assert_eq!(unique.len(), 20);
        assert_eq!(
            grid.axes()[0]
                .iter()
                .filter(|&&value| value == 17.0)
                .count(),
            1
        );
        assert_eq!(grid.axes()[0][4..7], [16.0, 17.0, 32.0]);
        // Stride 8192: 262144 rows end exactly at i32::MAX, so every cell runs.
        assert!(!GdnPrefillPostConvSpec::infeasible_mask(&cfg, &grid).contains(&true));
    }

    /// Qwen3.8's 48 value heads (stride 10240) overflow the kernel's int32
    /// offsets at the 262144 knot only.
    #[test]
    fn cells_past_the_int32_offsets_are_masked() {
        let cfg = GdnPrefillPostConvKernelConfig {
            num_value_heads: 48.into(),
            ..config()
        };
        let grid = GdnPrefillPostConvSpec::sweep_grid(&cfg);
        let mask = GdnPrefillPostConvSpec::infeasible_mask(&cfg, &grid);
        let masked: Vec<f64> = grid.axes()[0]
            .iter()
            .zip(&mask)
            .filter(|(_, &m)| m)
            .map(|(&t, _)| t)
            .collect();
        assert_eq!(masked, [262144.0]);
    }

    #[test]
    fn every_backend_uses_linear_1d_cache() {
        for backend in ["torch", "vllm_triton"] {
            assert_eq!(
                GdnPrefillPostConvSpec::cache_kind(backend),
                CacheKind::Cache1DLinear
            );
        }
    }

    #[test]
    fn enumerate_emits_exact_python_fields_and_key_runtime_values() {
        let cfg = config();
        let grid = GdnPrefillPostConvSpec::sweep_grid(&cfg);
        let payloads = GdnPrefillPostConvSpec::enumerate(&cfg, &grid, "vllm_triton");

        assert_eq!(payloads.len(), 20);
        for payload in &payloads {
            let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
            assert_eq!(
                names,
                [
                    "backend",
                    "dtype",
                    "key_head_dim",
                    "num_qk_heads",
                    "num_tokens",
                    "num_value_heads",
                    "value_head_dim",
                ]
            );
            assert_eq!(payload.fields().len(), 7);
            assert_eq!(payload.backend(), Some("vllm_triton"));
            assert_eq!(payload.fields()["num_qk_heads"], Value::from(16_u32));
            assert_eq!(payload.fields()["num_value_heads"], Value::from(32_u32));
            assert_eq!(payload.fields()["key_head_dim"], Value::from(128_u32));
            assert_eq!(payload.fields()["value_head_dim"], Value::from(128_u32));
            assert_eq!(payload.fields()["dtype"], Value::from("bf16"));
        }

        assert_eq!(payloads[0].fields()["num_tokens"], Value::from(1_u32));
        assert_eq!(payloads[5].fields()["num_tokens"], Value::from(17_u32));
        assert_eq!(payloads[8].fields()["num_tokens"], Value::from(128_u32));
        assert_eq!(payloads[19].fields()["num_tokens"], Value::from(262144_u32));
    }
}
