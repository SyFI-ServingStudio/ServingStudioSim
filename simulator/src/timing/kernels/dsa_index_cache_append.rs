//! GLM-5.2 DSA index-key quantization and page-planar cache append.
//!
//! Static identity captures the index width, page and quantization block sizes,
//! input/cache dtypes, scale encoding, and mixed cache format. Runtime is the
//! number of actual mapped tokens. This kind owns its Python profile table.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaIndexCacheAppendKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub index_dim: Dim,
    pub block_size: u32,
    pub quant_block_size: u32,
    #[compute_dtype]
    pub input_dtype: DType,
    #[kv_dtype]
    pub cache_dtype: DType,
    pub scale_format: String,
    pub cache_format: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DsaIndexCacheAppendKernelInput {
    pub num_tokens: u32,
}

pub struct DsaIndexCacheAppendSpec;

impl KernelSpec for DsaIndexCacheAppendSpec {
    type Config = DsaIndexCacheAppendKernelConfig;
    type Input = DsaIndexCacheAppendKernelInput;

    const KIND: KernelKind = "dsa_index_cache_append";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // Decode needs direct small-token coverage. The adjacent 264/265 points
        // preserve the established H200 paged-cache wave boundary; R.4 validates
        // whether linear interpolation is adequate for this fused kernel.
        SweepGrid::new(vec![Axis::chain([
            Axis::pow2(0, 8),
            Axis::values([264, 265]),
            Axis::token_axis(),
        ])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_tokens| {
            // Sweep axis values come from Axis::pow2/values/token_axis, all
            // non-negative integers well under u32::MAX.
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "sweep axis values are non-negative integers far below u32::MAX"
            )]
            let num_tokens = num_tokens as u32;
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens)
                .with("index_dim", config.index_dim.get())
                .with("block_size", config.block_size)
                .with("quant_block_size", config.quant_block_size)
                .with("input_dtype", config.input_dtype.as_str())
                .with("cache_dtype", config.cache_dtype.as_str())
                .with("scale_format", config.scale_format.clone())
                .with("cache_format", config.cache_format.clone())
        })
    }
}

register_kernel!(DsaIndexCacheAppendKernel, DsaIndexCacheAppendSpec);

#[cfg(test)]
mod tests {
    use super::{
        DsaIndexCacheAppendKernelConfig, DsaIndexCacheAppendKernelInput, DsaIndexCacheAppendSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{Dim, SlotInput, SweepCoords};
    use serde_json::Value;

    const TORCH_BACKEND: &str = "torch";
    const VLLM_BACKEND: &str = "vllm_cuda";

    fn config() -> DsaIndexCacheAppendKernelConfig {
        DsaIndexCacheAppendKernelConfig {
            backends: vec![TORCH_BACKEND, VLLM_BACKEND],
            gpu_name: "NVIDIA H200".to_string(),
            index_dim: Dim::param("index_dim", 128),
            block_size: 64,
            quant_block_size: 128,
            input_dtype: DType::Bf16,
            cache_dtype: DType::Fp8E4m3,
            scale_format: "ue8m0".to_string(),
            cache_format: "page_planar_fp8_fp32_scale".to_string(),
        }
    }

    #[test]
    fn config_identity_matches_the_dsa_index_cache_path() {
        let cfg = config();

        assert_eq!(DsaIndexCacheAppendSpec::KIND, "dsa_index_cache_append");
        assert_eq!(
            DsaIndexCacheAppendSpec::profile_kind(),
            "dsa_index_cache_append"
        );
        assert_eq!(cfg.backends(), &[TORCH_BACKEND, VLLM_BACKEND]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.index_dim, 128);
        assert_eq!(cfg.block_size, 64);
        assert_eq!(cfg.quant_block_size, 128);
        assert_eq!(cfg.input_dtype, DType::Bf16);
        assert_eq!(cfg.cache_dtype, DType::Fp8E4m3);
        assert_eq!(cfg.scale_format, "ue8m0");
        assert_eq!(cfg.cache_format, "page_planar_fp8_fp32_scale");
    }

    #[test]
    fn describe_config_preserves_rich_index_dim() {
        assert_eq!(
            config().describe_config(),
            serde_json::json!({
                "backends": [TORCH_BACKEND, VLLM_BACKEND],
                "gpu_name": "NVIDIA H200",
                "index_dim": {
                    "value": 128,
                    "expression": "index_dim",
                    "bindings": {"index_dim": 128},
                },
                "block_size": 64,
                "quant_block_size": 128,
                "input_dtype": "bf16",
                "cache_dtype": "fp8_e4m3",
                "scale_format": "ue8m0",
                "cache_format": "page_planar_fp8_fp32_scale",
            })
        );
    }

    #[test]
    fn input_coords_and_slot_input_are_exactly_num_tokens() {
        let input = DsaIndexCacheAppendKernelInput { num_tokens: 128 };

        assert_eq!(&*input.coords(), &[128.0]);
        assert_eq!(
            DsaIndexCacheAppendKernelInput::coord_field_names(),
            &["num_tokens"]
        );

        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_tokens": 128})
        );
    }

    #[test]
    fn sweep_has_frozen_decode_wave_and_prefill_coverage() {
        let cfg = config();
        let grid = DsaIndexCacheAppendSpec::sweep_grid(&cfg);
        let axis = &grid.axes()[0];

        assert_eq!(grid.axes().len(), 1);
        assert_eq!(&axis[..6], &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]);
        assert!(axis.windows(2).any(|pair| pair == [264.0, 265.0]));
        assert_eq!(axis.last(), Some(&65536.0));
        assert_eq!(axis.len(), 70);
        assert!(axis.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(DsaIndexCacheAppendSpec::infeasible_mask(&cfg, &grid).is_empty());
    }

    #[test]
    fn both_backends_use_linear_1d_cache() {
        assert_eq!(
            DsaIndexCacheAppendSpec::cache_kind(TORCH_BACKEND),
            CacheKind::Cache1DLinear
        );
        assert_eq!(
            DsaIndexCacheAppendSpec::cache_kind(VLLM_BACKEND),
            CacheKind::Cache1DLinear
        );
    }

    #[test]
    fn enumerate_matches_the_python_wire_schema_for_both_backends() {
        let cfg = config();
        let grid = DsaIndexCacheAppendSpec::sweep_grid(&cfg);

        for backend in [TORCH_BACKEND, VLLM_BACKEND] {
            let payloads = DsaIndexCacheAppendSpec::enumerate(&cfg, &grid, backend);
            assert_eq!(payloads.len(), 70);

            let first = &payloads[0];
            let fields = first.fields();
            let field_names: Vec<&str> = fields.keys().map(String::as_str).collect();

            assert_eq!(
                field_names,
                [
                    "backend",
                    "block_size",
                    "cache_dtype",
                    "cache_format",
                    "index_dim",
                    "input_dtype",
                    "num_tokens",
                    "quant_block_size",
                    "scale_format",
                ]
            );
            assert_eq!(fields.len(), 9);
            assert_eq!(fields.get("backend"), Some(&Value::from(backend)));
            assert_eq!(fields.get("num_tokens"), Some(&Value::from(1_u32)));
            assert_eq!(fields.get("index_dim"), Some(&Value::from(128_u32)));
            assert_eq!(fields.get("block_size"), Some(&Value::from(64_u32)));
            assert_eq!(fields.get("quant_block_size"), Some(&Value::from(128_u32)));
            assert_eq!(fields.get("input_dtype"), Some(&Value::from("bf16")));
            assert_eq!(fields.get("cache_dtype"), Some(&Value::from("fp8_e4m3")));
            assert_eq!(fields.get("scale_format"), Some(&Value::from("ue8m0")));
            assert_eq!(
                fields.get("cache_format"),
                Some(&Value::from("page_planar_fp8_fp32_scale"))
            );
            assert_eq!(first.backend(), Some(backend));
        }
    }

    #[test]
    fn dtype_tags_expose_compute_bf16_and_kv_fp8_e4m3() {
        let cfg = config();

        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), Some(DType::Fp8E4m3));
    }
}
