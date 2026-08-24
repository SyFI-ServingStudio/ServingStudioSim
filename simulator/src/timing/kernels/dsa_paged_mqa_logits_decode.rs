//! GLM-5.2 DSA paged decode MQA-logits kernel.
//!
//! The first cache stays on the physical `(batch_size, context_len)` coordinates.
//! Its boundary points bracket the block-64 page seams, the 256-token `DeepGEMM`
//! scheduler segments, and the H200's 132-SM batch wave. Measured R.4 misses add
//! batches 130/185 for interior interpolation, context 5793 for the diagonal
//! interaction, and batches 384/512 to bound the tested extrapolation domain.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaPagedMqaLogitsDecodeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub next_n: u32,
    pub max_model_len: Dim,
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub block_size: u32,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub cache_dtype: DType,
    pub scale_dtype: DType,
    pub weight_dtype: DType,
    pub output_dtype: DType,
    pub context_mode: String,
    pub page_mapping: String,
    pub cache_format: String,
    pub clean_logits: bool,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DsaPagedMqaLogitsDecodeKernelInput {
    pub batch_size: u32,
    pub context_len: u32,
}

/// Largest single profiling allocation this kernel may ask a GPU for, in bytes.
///
/// Raising the DSA context domain to 1,048,576 tokens makes the grid's far
/// corner physically unprofilable — not because the shape is invalid, but
/// because its operand would not fit on any card. Cells this drops are stored
/// non-finite and `Cache2DLinear` renormalizes over the surviving corners, so
/// the grid stays rectangular.
const MAX_PROFILE_ALLOCATION_BYTES: f64 = 32.0 * 1024.0 * 1024.0 * 1024.0;

/// Each cached index token carries one fp32 scale beside its key bytes.
const SCALE_BYTES_PER_TOKEN: u32 = 4;

pub struct DsaPagedMqaLogitsDecodeSpec;

impl KernelSpec for DsaPagedMqaLogitsDecodeSpec {
    type Config = DsaPagedMqaLogitsDecodeKernelConfig;
    type Input = DsaPagedMqaLogitsDecodeKernelInput;

    const KIND: KernelKind = "dsa_paged_mqa_logits_decode";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values([
                1, 2, 4, 8, 16, 32, 64, 127, 128, 129, 130, 131, 132, 133, 185, 256, 384, 512,
            ]),
            Axis::values([
                1, 63, 64, 65, 128, 255, 256, 257, 512, 1024, 2048, 4096, 5793, 8192, 16384, 32768,
                65536, 131072, 262144, 524288, 1048576,
            ]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // Each of the `batch_size` sequences scores its whole `context_len`, so
        // the work is the product of the axes.
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        let max_model_len = f64::from(config.max_model_len.get());
        // The paged index cache is one page set per sequence, so its size grows
        // with batch x context. At a 1M context the far corner would ask for
        // tens of terabytes; bound it rather than hand the profiler a shape no
        // card can hold.
        let cache_bytes_per_token = f64::from(config.head_dim.get() + SCALE_BYTES_PER_TOKEN);
        grid.expand_2d(|batch_size, context_len| {
            context_len > max_model_len
                || batch_size * context_len * cache_bytes_per_token > MAX_PROFILE_ALLOCATION_BYTES
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "batch_size/context_len are non-negative sweep-grid coordinates, far below u32::MAX"
        )]
        grid.expand_2d(|batch_size, context_len| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", batch_size as u32)
                .with("context_len", context_len as u32)
                .with("next_n", config.next_n)
                .with("max_model_len", config.max_model_len.get())
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("block_size", config.block_size)
                .with("q_dtype", config.q_dtype.as_str())
                .with("cache_dtype", config.cache_dtype.as_str())
                .with("scale_dtype", config.scale_dtype.as_str())
                .with("weight_dtype", config.weight_dtype.as_str())
                .with("output_dtype", config.output_dtype.as_str())
                .with("context_mode", config.context_mode.clone())
                .with("page_mapping", config.page_mapping.clone())
                .with("cache_format", config.cache_format.clone())
                .with("clean_logits", config.clean_logits)
        })
    }
}

register_kernel!(DsaPagedMqaLogitsDecodeKernel, DsaPagedMqaLogitsDecodeSpec);

#[cfg(test)]
mod tests {
    use super::{
        DsaPagedMqaLogitsDecodeKernelConfig, DsaPagedMqaLogitsDecodeKernelInput,
        DsaPagedMqaLogitsDecodeSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{Dim, SlotInput, SweepCoords};
    use serde_json::Value;

    const TORCH_BACKEND: &str = "torch";
    const DEEPGEMM_BACKEND: &str = "vllm_deepgemm_fp8";
    const BATCH_AXIS: &[f64] = &[
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0,
        185.0, 256.0, 384.0, 512.0,
    ];
    const CONTEXT_AXIS: &[f64] = &[
        1.0, 63.0, 64.0, 65.0, 128.0, 255.0, 256.0, 257.0, 512.0, 1024.0, 2048.0, 4096.0, 5793.0,
        8192.0, 16384.0, 32768.0, 65536.0, 131072.0, 262144.0, 524288.0, 1048576.0,
    ];

    /// The arch's full DSA timing domain — the context the grid now reaches.
    const FULL_MAX_MODEL_LEN: u32 = 1_048_576;

    fn config(max_model_len: u32) -> DsaPagedMqaLogitsDecodeKernelConfig {
        DsaPagedMqaLogitsDecodeKernelConfig {
            backends: vec![TORCH_BACKEND, DEEPGEMM_BACKEND],
            gpu_name: "NVIDIA H200".to_string(),
            next_n: 1,
            max_model_len: Dim::param("max_model_len", max_model_len),
            num_heads: Dim::param("num_index_heads", 64),
            head_dim: Dim::param("index_head_dim", 128),
            block_size: 64,
            q_dtype: DType::Fp8E4m3,
            cache_dtype: DType::Fp8E4m3,
            scale_dtype: DType::Fp32,
            weight_dtype: DType::Fp32,
            output_dtype: DType::Fp32,
            context_mode: "uniform".to_string(),
            page_mapping: "unique_scattered".to_string(),
            cache_format: "page_planar_fp8_fp32_scale".to_string(),
            clean_logits: false,
        }
    }

    #[test]
    fn config_identity_matches_the_paged_decode_logits_path() {
        let cfg = config(131072);

        assert_eq!(
            DsaPagedMqaLogitsDecodeSpec::KIND,
            "dsa_paged_mqa_logits_decode"
        );
        assert_eq!(
            DsaPagedMqaLogitsDecodeSpec::profile_kind(),
            "dsa_paged_mqa_logits_decode"
        );
        assert_eq!(cfg.backends(), &[TORCH_BACKEND, DEEPGEMM_BACKEND]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.next_n, 1);
        assert_eq!(cfg.max_model_len, 131072);
        assert_eq!(cfg.num_heads, 64);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.block_size, 64);
        assert_eq!(cfg.q_dtype, DType::Fp8E4m3);
        assert_eq!(cfg.cache_dtype, DType::Fp8E4m3);
        assert_eq!(cfg.scale_dtype, DType::Fp32);
        assert_eq!(cfg.weight_dtype, DType::Fp32);
        assert_eq!(cfg.output_dtype, DType::Fp32);
        assert_eq!(cfg.context_mode, "uniform");
        assert_eq!(cfg.page_mapping, "unique_scattered");
        assert_eq!(cfg.cache_format, "page_planar_fp8_fp32_scale");
        assert!(!cfg.clean_logits);
    }

    #[test]
    fn describe_config_preserves_rich_deployment_dimensions() {
        assert_eq!(
            config(131072).describe_config(),
            serde_json::json!({
                "backends": [TORCH_BACKEND, DEEPGEMM_BACKEND],
                "gpu_name": "NVIDIA H200",
                "next_n": 1,
                "max_model_len": {
                    "value": 131072,
                    "expression": "max_model_len",
                    "bindings": {"max_model_len": 131072},
                },
                "num_heads": {
                    "value": 64,
                    "expression": "num_index_heads",
                    "bindings": {"num_index_heads": 64},
                },
                "head_dim": {
                    "value": 128,
                    "expression": "index_head_dim",
                    "bindings": {"index_head_dim": 128},
                },
                "block_size": 64,
                "q_dtype": "fp8_e4m3",
                "cache_dtype": "fp8_e4m3",
                "scale_dtype": "fp32",
                "weight_dtype": "fp32",
                "output_dtype": "fp32",
                "context_mode": "uniform",
                "page_mapping": "unique_scattered",
                "cache_format": "page_planar_fp8_fp32_scale",
                "clean_logits": false,
            })
        );
    }

    #[test]
    fn input_coords_deserialization_and_slot_input_are_physical_batch_context() {
        let input: DsaPagedMqaLogitsDecodeKernelInput =
            serde_json::from_str(r#"{"batch_size":16,"context_len":4096}"#).unwrap();

        assert_eq!(&*input.coords(), &[16.0, 4096.0]);
        assert_eq!(
            DsaPagedMqaLogitsDecodeKernelInput::coord_field_names(),
            &["batch_size", "context_len"]
        );

        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"batch_size": 16, "context_len": 4096})
        );
    }

    #[test]
    fn sweep_grid_has_the_frozen_physical_axes_and_boundaries() {
        let grid = DsaPagedMqaLogitsDecodeSpec::sweep_grid(&config(FULL_MAX_MODEL_LEN));
        let axes = grid.axes();

        assert_eq!(axes.len(), 2);
        assert_eq!(axes[0], BATCH_AXIS);
        assert_eq!(axes[1], CONTEXT_AXIS);
        assert_eq!(axes[0].len(), 18);
        assert_eq!(axes[1].len(), 21);
        assert!(axes[0].windows(2).all(|pair| pair[0] < pair[1]));
        assert!(axes[1].windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(&axes[0][7..10], &[127.0, 128.0, 129.0]);
        assert_eq!(&axes[0][9..12], &[129.0, 130.0, 131.0]);
        assert_eq!(&axes[0][11..14], &[131.0, 132.0, 133.0]);
        assert_eq!(&axes[0][13..16], &[133.0, 185.0, 256.0]);
        assert_eq!(&axes[0][15..18], &[256.0, 384.0, 512.0]);
        assert_eq!(&axes[1][1..4], &[63.0, 64.0, 65.0]);
        assert_eq!(&axes[1][5..8], &[255.0, 256.0, 257.0]);
        assert_eq!(&axes[1][11..14], &[4096.0, 5793.0, 8192.0]);
        assert_eq!(axes[0].first(), Some(&1.0));
        assert_eq!(axes[0].last(), Some(&512.0));
        assert_eq!(axes[1].first(), Some(&1.0));
        assert_eq!(axes[1].last(), Some(&1_048_576.0));
        assert_eq!(axes[0].len() * axes[1].len(), 378);
    }

    #[test]
    fn infeasible_mask_respects_max_model_len_and_its_boundary() {
        let full_cfg = config(FULL_MAX_MODEL_LEN);
        let grid = DsaPagedMqaLogitsDecodeSpec::sweep_grid(&full_cfg);
        let full_mask = DsaPagedMqaLogitsDecodeSpec::infeasible_mask(&full_cfg, &grid);
        // Over the full domain nothing is dropped for exceeding the model
        // length; the only four dropped cells are the batch x context corners
        // whose paged index cache would not fit on a card.
        assert_eq!(full_mask.len(), 378);
        assert_eq!(full_mask.iter().filter(|&&masked| masked).count(), 4);
        assert_eq!(full_mask.iter().filter(|&&masked| !masked).count(), 374);

        let bounded_cfg = config(256);
        let bounded_mask = DsaPagedMqaLogitsDecodeSpec::infeasible_mask(&bounded_cfg, &grid);
        assert_eq!(bounded_mask.len(), 378);
        assert_eq!(bounded_mask.iter().filter(|&&masked| masked).count(), 252);
        assert_eq!(bounded_mask.iter().filter(|&&masked| !masked).count(), 126);

        let context_count = grid.axes()[1].len();
        let cell = |mask: &[bool], batch_size: f64, context_len: f64| {
            let i = grid.axes()[0]
                .iter()
                .position(|&value| value == batch_size)
                .unwrap();
            let j = grid.axes()[1]
                .iter()
                .position(|&value| value == context_len)
                .unwrap();
            mask[i * context_count + j]
        };
        for batch_size in [1.0, 130.0, 132.0, 185.0, 384.0, 512.0] {
            assert!(!cell(&bounded_mask, batch_size, 255.0));
            assert!(!cell(&bounded_mask, batch_size, 256.0));
            assert!(cell(&bounded_mask, batch_size, 257.0));
            assert!(cell(&bounded_mask, batch_size, 5793.0));
        }
        // The allocation staircase over the full domain: a 1M context is
        // profilable at small batch and only drops once the page set does not
        // fit, and a large batch is fine at every shorter context.
        assert!(!cell(&full_mask, 1.0, 1_048_576.0));
        assert!(!cell(&full_mask, 128.0, 1_048_576.0));
        assert!(cell(&full_mask, 256.0, 1_048_576.0));
        assert!(cell(&full_mask, 512.0, 524_288.0));
        assert!(!cell(&full_mask, 512.0, 262_144.0));
    }

    #[test]
    fn both_backends_use_physical_bilinear_cache() {
        assert_eq!(
            DsaPagedMqaLogitsDecodeSpec::cache_kind(TORCH_BACKEND),
            CacheKind::Cache2DLinear(Extrapolation::Product)
        );
        assert_eq!(
            DsaPagedMqaLogitsDecodeSpec::cache_kind(DEEPGEMM_BACKEND),
            CacheKind::Cache2DLinear(Extrapolation::Product)
        );
    }

    #[test]
    fn enumerate_matches_the_full_python_wire_schema_before_masking() {
        let cfg = config(FULL_MAX_MODEL_LEN);
        let grid = DsaPagedMqaLogitsDecodeSpec::sweep_grid(&cfg);
        let payloads = DsaPagedMqaLogitsDecodeSpec::enumerate(&cfg, &grid, DEEPGEMM_BACKEND);

        assert_eq!(payloads.len(), 378);
        let expected_names = [
            "backend",
            "batch_size",
            "block_size",
            "cache_dtype",
            "cache_format",
            "clean_logits",
            "context_len",
            "context_mode",
            "head_dim",
            "max_model_len",
            "next_n",
            "num_heads",
            "output_dtype",
            "page_mapping",
            "q_dtype",
            "scale_dtype",
            "weight_dtype",
        ];
        for payload in &payloads {
            let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
            assert_eq!(names, expected_names);
            assert_eq!(payload.fields().len(), 17);
        }

        assert_payload(&payloads[0], DEEPGEMM_BACKEND, 1, 1);
        assert_payload(&payloads[20], DEEPGEMM_BACKEND, 1, 1_048_576);
        assert_payload(&payloads[21], DEEPGEMM_BACKEND, 2, 1);
        assert_payload(&payloads[218], DEEPGEMM_BACKEND, 130, 512);
        assert_payload(&payloads[306], DEEPGEMM_BACKEND, 185, 5793);
        assert_payload(&payloads[344], DEEPGEMM_BACKEND, 384, 512);
        assert_payload(payloads.last().unwrap(), DEEPGEMM_BACKEND, 512, 1_048_576);
    }

    fn assert_payload(
        payload: &crate::timing::bridge::ArgsPayload,
        backend: &str,
        batch_size: u32,
        context_len: u32,
    ) {
        let fields = payload.fields();
        assert_eq!(fields.get("backend"), Some(&Value::from(backend)));
        assert_eq!(fields.get("batch_size"), Some(&Value::from(batch_size)));
        assert_eq!(fields.get("context_len"), Some(&Value::from(context_len)));
        assert_eq!(fields.get("next_n"), Some(&Value::from(1_u32)));
        assert_eq!(
            fields.get("max_model_len"),
            Some(&Value::from(FULL_MAX_MODEL_LEN))
        );
        assert_eq!(fields.get("num_heads"), Some(&Value::from(64_u32)));
        assert_eq!(fields.get("head_dim"), Some(&Value::from(128_u32)));
        assert_eq!(fields.get("block_size"), Some(&Value::from(64_u32)));
        assert_eq!(fields.get("q_dtype"), Some(&Value::from("fp8_e4m3")));
        assert_eq!(fields.get("cache_dtype"), Some(&Value::from("fp8_e4m3")));
        assert_eq!(fields.get("scale_dtype"), Some(&Value::from("fp32")));
        assert_eq!(fields.get("weight_dtype"), Some(&Value::from("fp32")));
        assert_eq!(fields.get("output_dtype"), Some(&Value::from("fp32")));
        assert_eq!(fields.get("context_mode"), Some(&Value::from("uniform")));
        assert_eq!(
            fields.get("page_mapping"),
            Some(&Value::from("unique_scattered"))
        );
        assert_eq!(
            fields.get("cache_format"),
            Some(&Value::from("page_planar_fp8_fp32_scale"))
        );
        assert_eq!(fields.get("clean_logits"), Some(&Value::from(false)));
        assert_eq!(payload.backend(), Some(backend));
    }

    #[test]
    fn dtype_tags_expose_fp8_compute_and_kv() {
        let cfg = config(131072);

        assert_eq!(cfg.compute_dtype(), Some(DType::Fp8E4m3));
        assert_eq!(cfg.kv_dtype(), Some(DType::Fp8E4m3));
    }
}
