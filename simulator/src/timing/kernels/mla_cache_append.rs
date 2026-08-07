//! GLM-5.2 plain MLA cache append: concatenate latent and RoPE rows into one
//! paged-cache entry.
//!
//! Static identity captures the latent/RoPE widths, page size, input/cache
//! dtypes, and cache format. Runtime is the number of actual mapped tokens.
//! This kind owns its Python profile table and does not reuse `kv_cache_append`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MlaCacheAppendKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub kv_lora_rank: Dim,
    pub rope_dim: Dim,
    pub block_size: u32,
    #[compute_dtype]
    pub input_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
    pub cache_format: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MlaCacheAppendKernelInput {
    pub num_tokens: u32,
}

pub struct MlaCacheAppendSpec;

impl KernelSpec for MlaCacheAppendSpec {
    type Config = MlaCacheAppendKernelConfig;
    type Input = MlaCacheAppendKernelInput;

    const KIND: KernelKind = "mla_cache_append";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // Decode needs direct small-token coverage. On H200, 264 and 265
        // preserve the adjacent two-block-per-SM wave boundary used by the
        // closest paged cache-append kind.
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
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
                .with("kv_lora_rank", config.kv_lora_rank.get())
                .with("rope_dim", config.rope_dim.get())
                .with("block_size", config.block_size)
                .with("input_dtype", config.input_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("cache_format", config.cache_format.clone())
        })
    }
}

register_kernel!(MlaCacheAppendKernel, MlaCacheAppendSpec);

#[cfg(test)]
mod tests {
    use super::{MlaCacheAppendKernelConfig, MlaCacheAppendKernelInput, MlaCacheAppendSpec};
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{Dim, SlotInput, SweepCoords};
    use serde_json::Value;

    const TORCH_BACKEND: &str = "torch";
    const VLLM_BACKEND: &str = "vllm_cuda";

    fn config() -> MlaCacheAppendKernelConfig {
        MlaCacheAppendKernelConfig {
            backends: vec![TORCH_BACKEND, VLLM_BACKEND],
            gpu_name: "NVIDIA H200".to_string(),
            kv_lora_rank: Dim::param("kv_lora_rank", 512),
            rope_dim: Dim::param("rope_dim", 64),
            block_size: 64,
            input_dtype: DType::Bf16,
            kv_dtype: DType::Bf16,
            cache_format: "plain".to_string(),
        }
    }

    #[test]
    fn config_identity_matches_the_plain_mla_cache_path() {
        let cfg = config();

        assert_eq!(MlaCacheAppendSpec::KIND, "mla_cache_append");
        assert_eq!(MlaCacheAppendSpec::profile_kind(), "mla_cache_append");
        assert_eq!(cfg.backends(), &[TORCH_BACKEND, VLLM_BACKEND]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.kv_lora_rank, 512);
        assert_eq!(cfg.rope_dim, 64);
        assert_eq!(cfg.block_size, 64);
        assert_eq!(cfg.input_dtype, DType::Bf16);
        assert_eq!(cfg.kv_dtype, DType::Bf16);
        assert_eq!(cfg.cache_format, "plain");
    }

    #[test]
    fn describe_config_preserves_rich_width_dims() {
        assert_eq!(
            config().describe_config(),
            serde_json::json!({
                "backends": [TORCH_BACKEND, VLLM_BACKEND],
                "gpu_name": "NVIDIA H200",
                "kv_lora_rank": {
                    "value": 512,
                    "expression": "kv_lora_rank",
                    "bindings": {"kv_lora_rank": 512},
                },
                "rope_dim": {
                    "value": 64,
                    "expression": "rope_dim",
                    "bindings": {"rope_dim": 64},
                },
                "block_size": 64,
                "input_dtype": "bf16",
                "kv_dtype": "bf16",
                "cache_format": "plain",
            })
        );
    }

    #[test]
    fn input_coords_and_slot_input_are_exactly_num_tokens() {
        let input = MlaCacheAppendKernelInput { num_tokens: 128 };

        assert_eq!(&*input.coords(), &[128.0]);
        assert_eq!(
            MlaCacheAppendKernelInput::coord_field_names(),
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
        let grid = MlaCacheAppendSpec::sweep_grid(&cfg);
        let axis = &grid.axes()[0];

        assert_eq!(grid.axes().len(), 1);
        assert_eq!(&axis[..6], &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]);
        assert!(axis.windows(2).any(|pair| pair == [264.0, 265.0]));
        assert_eq!(axis.last(), Some(&65536.0));
        assert_eq!(axis.len(), 70);
        assert!(axis.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(MlaCacheAppendSpec::infeasible_mask(&cfg, &grid).is_empty());
    }

    #[test]
    fn both_backends_use_linear_1d_cache() {
        assert_eq!(
            MlaCacheAppendSpec::cache_kind(TORCH_BACKEND),
            CacheKind::Cache1DLinear
        );
        assert_eq!(
            MlaCacheAppendSpec::cache_kind(VLLM_BACKEND),
            CacheKind::Cache1DLinear
        );
    }

    #[test]
    fn enumerate_matches_the_python_wire_schema_for_both_backends() {
        let cfg = config();
        let grid = MlaCacheAppendSpec::sweep_grid(&cfg);

        for backend in [TORCH_BACKEND, VLLM_BACKEND] {
            let payloads = MlaCacheAppendSpec::enumerate(&cfg, &grid, backend);
            assert_eq!(payloads.len(), 70);

            let first = &payloads[0];
            let fields = first.fields();
            let field_names: Vec<&str> = fields.keys().map(String::as_str).collect();

            assert_eq!(
                field_names,
                [
                    "backend",
                    "block_size",
                    "cache_format",
                    "input_dtype",
                    "kv_dtype",
                    "kv_lora_rank",
                    "num_tokens",
                    "rope_dim",
                ]
            );
            assert_eq!(fields.len(), 8);
            assert_eq!(fields.get("backend"), Some(&Value::from(backend)));
            assert_eq!(fields.get("num_tokens"), Some(&Value::from(1_u32)));
            assert_eq!(fields.get("kv_lora_rank"), Some(&Value::from(512_u32)));
            assert_eq!(fields.get("rope_dim"), Some(&Value::from(64_u32)));
            assert_eq!(fields.get("block_size"), Some(&Value::from(64_u32)));
            assert_eq!(fields.get("input_dtype"), Some(&Value::from("bf16")));
            assert_eq!(fields.get("kv_dtype"), Some(&Value::from("bf16")));
            assert_eq!(fields.get("cache_format"), Some(&Value::from("plain")));
            assert_eq!(first.backend(), Some(backend));
        }
    }

    #[test]
    fn dtype_tags_expose_compute_and_kv_bf16() {
        let cfg = config();

        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), Some(DType::Bf16));
    }
}
