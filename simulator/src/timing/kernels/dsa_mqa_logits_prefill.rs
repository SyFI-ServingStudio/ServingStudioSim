//! GLM-5.2 DSA prefill MQA-logits kernel.
//!
//! The cache stays on the physical `(num_queries, num_keys)` coordinates for
//! the first implementation. The measured R.4 gate missed at `(363, 363)`
//! (cache/truth ratio 0.85575) and `(363, 2896)` (ratio 0.73969), so the shared
//! 363 boundary below targets both the masked diagonal and interior M=363 miss
//! without changing the public query contract or cache algorithm.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
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

/// Largest single profiling allocation this kernel may ask a GPU for, in bytes.
///
/// Raising the DSA context domain to 1,048,576 tokens makes the grid's far
/// corner physically unprofilable — not because the shape is invalid, but
/// because its operand would not fit on any card. The bound is a staircase
/// rather than a per-axis cap: a large query count is fine against a short
/// context and vice versa, which is exactly how the real workload is shaped
/// (a session's big fresh input lands on round 0, when its context is still
/// empty). Cells it drops are stored non-finite and `Cache2DLinear`
/// renormalizes over the surviving corners, so the grid stays rectangular.
const MAX_PROFILE_ALLOCATION_BYTES: f64 = 32.0 * 1024.0 * 1024.0 * 1024.0;

/// One fp32 logit plus one bool validity flag per (query, key) pair.
const DENSE_PAIR_BYTES_PER_ELEMENT: f64 = 5.0;

pub struct DsaMqaLogitsPrefillSpec;

impl KernelSpec for DsaMqaLogitsPrefillSpec {
    type Config = DsaMqaLogitsPrefillKernelConfig;
    type Input = DsaMqaLogitsPrefillKernelInput;

    const KIND: KernelKind = "dsa_mqa_logits_prefill";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values([
                1, 2, 3, 4, 8, 16, 32, 64, 127, 128, 129, 255, 256, 257, 363, 512, 1024, 2048,
                4096, 8192, 16384, 32768, 65536,
            ]),
            Axis::values([
                1, 2, 4, 8, 16, 32, 64, 128, 255, 256, 257, 363, 512, 1024, 2048, 4096, 8192,
                16384, 32768, 65536, 131072, 262144, 524288, 1048576,
            ]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // The logits this kernel produces are the `num_queries x num_keys`
        // matrix itself, so the axes are the two factors of the work.
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        // The Python contract requires 0 < num_queries <= num_keys. Preserve
        // the full rectangular cache grid, but never ask the profiler for its
        // physically invalid M>N corner.
        grid.expand_2d(|num_queries, num_keys| {
            // The runner materializes a dense [queries, keys] validity mask
            // alongside the fp32 logits, so the operand grows with the product.
            let dense_pair_bytes = num_queries * num_keys * DENSE_PAIR_BYTES_PER_ELEMENT;
            num_queries > num_keys || dense_pair_bytes > MAX_PROFILE_ALLOCATION_BYTES
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
            reason = "num_queries/num_keys are non-negative sweep-grid coordinates, far below u32::MAX"
        )]
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
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{Dim, SlotInput, SweepCoords};
    use serde_json::Value;

    const TORCH_BACKEND: &str = "torch";
    const DEEPGEMM_BACKEND: &str = "vllm_deepgemm_fp8";
    const QUERY_AXIS: &[f64] = &[
        1.0, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0, 127.0, 128.0, 129.0, 255.0, 256.0, 257.0, 363.0,
        512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0,
    ];
    const KEY_AXIS: &[f64] = &[
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 255.0, 256.0, 257.0, 363.0, 512.0, 1024.0,
        2048.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0, 131072.0, 262144.0, 524288.0, 1048576.0,
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
        assert_eq!(axes[0].len(), 23);
        assert_eq!(axes[1].len(), 24);
        assert!(axes[0].windows(2).all(|pair| pair[0] < pair[1]));
        assert!(axes[1].windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(&axes[0][..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&axes[0][8..11], &[127.0, 128.0, 129.0]);
        assert_eq!(&axes[0][11..14], &[255.0, 256.0, 257.0]);
        assert_eq!(&axes[1][8..11], &[255.0, 256.0, 257.0]);
        assert_eq!(&axes[0][13..16], &[257.0, 363.0, 512.0]);
        assert_eq!(&axes[1][10..13], &[257.0, 363.0, 512.0]);
        // Both axes reach the arch's 1,048,576-token timing domain on the key
        // side; the query side stops at 65,536, past which a single padded
        // logits block no longer fits on a card.
        assert_eq!(axes[0].last(), Some(&65536.0));
        assert_eq!(axes[1].last(), Some(&1_048_576.0));
        assert_eq!(axes[0].len() * axes[1].len(), 552);
    }

    #[test]
    fn infeasible_mask_drops_exactly_the_m_greater_than_n_corner() {
        let cfg = config();
        let grid = DsaMqaLogitsPrefillSpec::sweep_grid(&cfg);
        let mask = DsaMqaLogitsPrefillSpec::infeasible_mask(&cfg, &grid);
        let key_count = grid.axes()[1].len();

        assert_eq!(mask.len(), 552);
        assert_eq!(mask.iter().filter(|&&masked| masked).count(), 217);
        assert_eq!(mask.iter().filter(|&&masked| !masked).count(), 335);

        let masked = |m: f64, n: f64| {
            let i = grid.axes()[0].iter().position(|&value| value == m).unwrap();
            let j = grid.axes()[1].iter().position(|&value| value == n).unwrap();
            mask[i * key_count + j]
        };
        assert!(!masked(1.0, 1.0));
        assert!(!masked(128.0, 128.0));
        assert!(!masked(128.0, 4096.0));
        assert!(masked(2.0, 1.0));
        assert!(masked(363.0, 257.0));
        assert!(!masked(363.0, 363.0));
        assert!(!masked(363.0, 512.0));
        assert!(masked(512.0, 363.0));
        assert!(masked(4096.0, 2048.0));
        // The allocation staircase: a wide query block is fine against a short
        // context and dropped against the longest one.
        assert!(!masked(4096.0, 65536.0));
        assert!(masked(65536.0, 1_048_576.0));
    }

    #[test]
    fn both_backends_use_physical_bilinear_cache() {
        assert_eq!(
            DsaMqaLogitsPrefillSpec::cache_kind(TORCH_BACKEND),
            CacheKind::Cache2DLinear(Extrapolation::Product)
        );
        assert_eq!(
            DsaMqaLogitsPrefillSpec::cache_kind(DEEPGEMM_BACKEND),
            CacheKind::Cache2DLinear(Extrapolation::Product)
        );
    }

    #[test]
    fn enumerate_matches_the_full_python_wire_schema_before_masking() {
        let cfg = config();
        let grid = DsaMqaLogitsPrefillSpec::sweep_grid(&cfg);
        let payloads = DsaMqaLogitsPrefillSpec::enumerate(&cfg, &grid, DEEPGEMM_BACKEND);

        assert_eq!(payloads.len(), 552);
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
        assert_payload(payloads.last().unwrap(), DEEPGEMM_BACKEND, 65536, 1_048_576);
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
