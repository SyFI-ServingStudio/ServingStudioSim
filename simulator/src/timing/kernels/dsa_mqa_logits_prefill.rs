//! GLM-5.2 DSA prefill MQA-logits kernel.
//!
//! The cache stays on the physical `(num_queries, num_keys)` coordinates for
//! the first implementation. DeepGEMM's two-query and 256-key schedule may
//! create interpolation boundaries within that space; the adjacent sweep
//! points below preserve those boundaries for the later fidelity gate.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaMqaLogitsPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_sequences: u32,
    pub num_heads: Dim,
    pub head_dim: Dim,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub k_dtype: DType,
    pub k_scale_dtype: DType,
    pub weight_dtype: DType,
    pub output_dtype: DType,
    pub span_mode: String,
    pub clean_logits: bool,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DsaMqaLogitsPrefillKernelInput {
    pub num_queries: u32,
    pub num_keys: u32,
}

pub struct DsaMqaLogitsPrefillSpec;

impl KernelSpec for DsaMqaLogitsPrefillSpec {
    type Config = DsaMqaLogitsPrefillKernelConfig;
    type Input = DsaMqaLogitsPrefillKernelInput;

    const KIND: KernelKind = "dsa_mqa_logits_prefill";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values([
                1, 2, 3, 4, 8, 16, 32, 64, 127, 128, 129, 255, 256, 257, 512, 1024, 2048, 4096,
            ]),
            Axis::values([
                1, 2, 4, 8, 16, 32, 64, 128, 255, 256, 257, 512, 1024, 2048, 4096, 8192, 16384,
                32768, 65536,
            ]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        // The Python contract requires 0 < num_queries <= num_keys. Preserve
        // the full rectangular cache grid, but never ask the profiler for its
        // physically invalid M>N corner.
        grid.expand_2d(|num_queries, num_keys| num_queries > num_keys)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|num_queries, num_keys| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_queries", num_queries as u32)
                .with("num_keys", num_keys as u32)
                .with("num_sequences", config.num_sequences)
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("q_dtype", config.q_dtype.as_str())
                .with("k_dtype", config.k_dtype.as_str())
                .with("k_scale_dtype", config.k_scale_dtype.as_str())
                .with("weight_dtype", config.weight_dtype.as_str())
                .with("output_dtype", config.output_dtype.as_str())
                .with("span_mode", config.span_mode.clone())
                .with("clean_logits", config.clean_logits)
        })
    }
}

register_kernel!(DsaMqaLogitsPrefillKernel, DsaMqaLogitsPrefillSpec);

#[cfg(test)]
mod tests {
    use super::{
        DsaMqaLogitsPrefillKernelConfig, DsaMqaLogitsPrefillKernelInput, DsaMqaLogitsPrefillSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{Dim, SlotInput, SweepCoords};
    use serde_json::Value;

    const TORCH_BACKEND: &str = "torch";
    const DEEPGEMM_BACKEND: &str = "vllm_deepgemm_fp8";
    const QUERY_AXIS: &[f64] = &[
        1.0, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0, 127.0, 128.0, 129.0, 255.0, 256.0, 257.0, 512.0,
        1024.0, 2048.0, 4096.0,
    ];
    const KEY_AXIS: &[f64] = &[
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 255.0, 256.0, 257.0, 512.0, 1024.0, 2048.0,
        4096.0, 8192.0, 16384.0, 32768.0, 65536.0,
    ];

    fn config() -> DsaMqaLogitsPrefillKernelConfig {
        DsaMqaLogitsPrefillKernelConfig {
            backends: vec![TORCH_BACKEND, DEEPGEMM_BACKEND],
            gpu_name: "NVIDIA H200".to_string(),
            num_sequences: 1,
            num_heads: Dim::param("num_index_heads", 64),
            head_dim: Dim::param("index_head_dim", 128),
            q_dtype: DType::Fp8E4m3,
            k_dtype: DType::Fp8E4m3,
            k_scale_dtype: DType::Fp32,
            weight_dtype: DType::Fp32,
            output_dtype: DType::Fp32,
            span_mode: "single_causal_tail".to_string(),
            clean_logits: false,
        }
    }

    #[test]
    fn config_identity_matches_the_prefill_logits_path() {
        let cfg = config();

        assert_eq!(DsaMqaLogitsPrefillSpec::KIND, "dsa_mqa_logits_prefill");
        assert_eq!(
            DsaMqaLogitsPrefillSpec::profile_kind(),
            "dsa_mqa_logits_prefill"
        );
        assert_eq!(cfg.backends(), &[TORCH_BACKEND, DEEPGEMM_BACKEND]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.num_sequences, 1);
        assert_eq!(cfg.num_heads, 64);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.q_dtype, DType::Fp8E4m3);
        assert_eq!(cfg.k_dtype, DType::Fp8E4m3);
        assert_eq!(cfg.k_scale_dtype, DType::Fp32);
        assert_eq!(cfg.weight_dtype, DType::Fp32);
        assert_eq!(cfg.output_dtype, DType::Fp32);
        assert_eq!(cfg.span_mode, "single_causal_tail");
        assert!(!cfg.clean_logits);
    }

    #[test]
    fn describe_config_preserves_rich_head_dimensions() {
        assert_eq!(
            config().describe_config(),
            serde_json::json!({
                "backends": [TORCH_BACKEND, DEEPGEMM_BACKEND],
                "gpu_name": "NVIDIA H200",
                "num_sequences": 1,
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
                "q_dtype": "fp8_e4m3",
                "k_dtype": "fp8_e4m3",
                "k_scale_dtype": "fp32",
                "weight_dtype": "fp32",
                "output_dtype": "fp32",
                "span_mode": "single_causal_tail",
                "clean_logits": false,
            })
        );
    }

    #[test]
    fn input_coords_deserialization_and_slot_input_are_physical_m_n() {
        let input: DsaMqaLogitsPrefillKernelInput =
            serde_json::from_str(r#"{"num_queries":128,"num_keys":4096}"#).unwrap();

        assert_eq!(&*input.coords(), &[128.0, 4096.0]);
        assert_eq!(
            DsaMqaLogitsPrefillKernelInput::coord_field_names(),
            &["num_queries", "num_keys"]
        );

        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_queries": 128, "num_keys": 4096})
        );
    }

    #[test]
    fn sweep_grid_has_the_frozen_physical_axes() {
        let grid = DsaMqaLogitsPrefillSpec::sweep_grid(&config());
        let axes = grid.axes();

        assert_eq!(axes.len(), 2);
        assert_eq!(axes[0], QUERY_AXIS);
        assert_eq!(axes[1], KEY_AXIS);
        assert_eq!(axes[0].len(), 18);
        assert_eq!(axes[1].len(), 19);
        assert!(axes[0].windows(2).all(|pair| pair[0] < pair[1]));
        assert!(axes[1].windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(&axes[0][..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&axes[0][8..11], &[127.0, 128.0, 129.0]);
        assert_eq!(&axes[0][11..14], &[255.0, 256.0, 257.0]);
        assert_eq!(&axes[1][8..11], &[255.0, 256.0, 257.0]);
        assert_eq!(axes[0].last(), Some(&4096.0));
        assert_eq!(axes[1].last(), Some(&65536.0));
        assert_eq!(axes[0].len() * axes[1].len(), 342);
    }

    #[test]
    fn infeasible_mask_drops_exactly_the_m_greater_than_n_corner() {
        let cfg = config();
        let grid = DsaMqaLogitsPrefillSpec::sweep_grid(&cfg);
        let mask = DsaMqaLogitsPrefillSpec::infeasible_mask(&cfg, &grid);
        let key_count = grid.axes()[1].len();

        assert_eq!(mask.len(), 342);
        assert_eq!(mask.iter().filter(|&&masked| masked).count(), 122);
        assert_eq!(mask.iter().filter(|&&masked| !masked).count(), 220);

        let masked = |m: f64, n: f64| {
            let i = grid.axes()[0].iter().position(|&value| value == m).unwrap();
            let j = grid.axes()[1].iter().position(|&value| value == n).unwrap();
            mask[i * key_count + j]
        };
        assert!(!masked(1.0, 1.0));
        assert!(!masked(128.0, 128.0));
        assert!(!masked(128.0, 4096.0));
        assert!(masked(2.0, 1.0));
        assert!(masked(4096.0, 2048.0));
    }

    #[test]
    fn both_backends_use_physical_bilinear_cache() {
        assert_eq!(
            DsaMqaLogitsPrefillSpec::cache_kind(TORCH_BACKEND),
            CacheKind::Cache2DLinear
        );
        assert_eq!(
            DsaMqaLogitsPrefillSpec::cache_kind(DEEPGEMM_BACKEND),
            CacheKind::Cache2DLinear
        );
    }

    #[test]
    fn enumerate_matches_the_full_python_wire_schema_before_masking() {
        let cfg = config();
        let grid = DsaMqaLogitsPrefillSpec::sweep_grid(&cfg);
        let payloads = DsaMqaLogitsPrefillSpec::enumerate(&cfg, &grid, DEEPGEMM_BACKEND);

        assert_eq!(payloads.len(), 342);
        let expected_names = [
            "backend",
            "clean_logits",
            "head_dim",
            "k_dtype",
            "k_scale_dtype",
            "num_heads",
            "num_keys",
            "num_queries",
            "num_sequences",
            "output_dtype",
            "q_dtype",
            "span_mode",
            "weight_dtype",
        ];
        for payload in &payloads {
            let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
            assert_eq!(names, expected_names);
            assert_eq!(payload.fields().len(), 13);
        }

        assert_payload(&payloads[0], DEEPGEMM_BACKEND, 1, 1);
        assert_payload(payloads.last().unwrap(), DEEPGEMM_BACKEND, 4096, 65536);
    }

    fn assert_payload(
        payload: &crate::timing::bridge::ArgsPayload,
        backend: &str,
        num_queries: u32,
        num_keys: u32,
    ) {
        let fields = payload.fields();
        assert_eq!(fields.get("backend"), Some(&Value::from(backend)));
        assert_eq!(fields.get("num_queries"), Some(&Value::from(num_queries)));
        assert_eq!(fields.get("num_keys"), Some(&Value::from(num_keys)));
        assert_eq!(fields.get("num_sequences"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("num_heads"), Some(&Value::from(64_u32)));
        assert_eq!(fields.get("head_dim"), Some(&Value::from(128_u32)));
        assert_eq!(fields.get("q_dtype"), Some(&Value::from("fp8_e4m3")));
        assert_eq!(fields.get("k_dtype"), Some(&Value::from("fp8_e4m3")));
        assert_eq!(fields.get("k_scale_dtype"), Some(&Value::from("fp32")));
        assert_eq!(fields.get("weight_dtype"), Some(&Value::from("fp32")));
        assert_eq!(fields.get("output_dtype"), Some(&Value::from("fp32")));
        assert_eq!(
            fields.get("span_mode"),
            Some(&Value::from("single_causal_tail"))
        );
        assert_eq!(fields.get("clean_logits"), Some(&Value::from(false)));
        assert_eq!(payload.backend(), Some(backend));
    }

    #[test]
    fn dtype_tags_expose_fp8_compute_and_kv() {
        let cfg = config();

        assert_eq!(cfg.compute_dtype(), Some(DType::Fp8E4m3));
        assert_eq!(cfg.kv_dtype(), Some(DType::Fp8E4m3));
    }
}
