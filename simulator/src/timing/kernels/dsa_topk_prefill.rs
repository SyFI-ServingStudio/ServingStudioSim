//! GLM-5.2 DSA prefill top-k kernel.
//!
//! The cache stays on physical `(num_queries, num_keys)` coordinates and
//! brackets the observed row-wave, work-tile, and semantic top-k boundaries.
//! Measured R.4 evidence added the shared 1448 diagonal, paired M=2047 with the
//! existing N=2047 point, and extended N to 1048576 to cover the arch's full
//! timing domain. This changes neither the cache algorithm nor the public query
//! contract. Python's `logits_row_stride` remains derived from `num_keys` during
//! enumeration to reproduce the padded DeepGEMM-logits layout.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaTopkPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_sequences: u32,
    pub top_k: u32,
    #[compute_dtype]
    pub logits_dtype: DType,
    pub index_dtype: String,
    pub span_mode: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DsaTopkPrefillKernelInput {
    pub num_queries: u32,
    pub num_keys: u32,
}

pub struct DsaTopkPrefillSpec;

impl KernelSpec for DsaTopkPrefillSpec {
    type Config = DsaTopkPrefillKernelConfig;
    type Input = DsaTopkPrefillKernelInput;

    const KIND: KernelKind = "dsa_topk_prefill";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values([
                1, 2, 4, 8, 16, 32, 64, 127, 128, 129, 131, 132, 133, 255, 256, 257, 512, 1024,
                1448, 2047, 2048, 4096, 8192, 16384, 32768, 65536,
            ]),
            Axis::values([
                1, 2, 4, 8, 16, 32, 64, 128, 255, 256, 257, 512, 1024, 1448, 2047, 2048, 2049,
                4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
            ]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // Top-k runs over the `num_queries x num_keys` logits matrix, so the
        // axes are the two factors of the work.
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        // The production backend requires 0 < num_queries <= num_keys. Keep the
        // rectangular interpolation grid, but never profile its invalid M>N corner.
        grid.expand_2d(|num_queries, num_keys| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "num_keys is a non-negative Axis::values sweep coordinate, capped at 1,048,576, \
                          far below u32::MAX"
            )]
            let padded_logits_bytes = num_queries
                * f64::from(logits_row_stride(num_keys as u32))
                * f64::from(DType::Fp32.size_bytes());
            num_queries > num_keys || padded_logits_bytes > MAX_PROFILE_ALLOCATION_BYTES
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|num_queries, num_keys| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "num_keys is a non-negative Axis::values sweep coordinate, capped at 1,048,576, \
                          far below u32::MAX"
            )]
            let num_keys = num_keys as u32;
            ArgsPayload::new()
                .with("backend", backend)
                .with(
                    "num_queries",
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "num_queries is a non-negative Axis::values sweep coordinate, capped at \
                                  65,536, far below u32::MAX"
                    )]
                    {
                        num_queries as u32
                    },
                )
                .with("num_keys", num_keys)
                .with("num_sequences", config.num_sequences)
                .with("top_k", config.top_k)
                .with("logits_row_stride", logits_row_stride(num_keys))
                .with("logits_dtype", config.logits_dtype.as_str())
                .with("index_dtype", config.index_dtype.clone())
                .with("span_mode", config.span_mode.clone())
        })
    }
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

/// `DeepGEMM` logits pad N to 256 and retain one additional 256-column tile.
fn logits_row_stride(num_keys: u32) -> u32 {
    num_keys.div_ceil(256) * 256 + 256
}

register_kernel!(DsaTopkPrefillKernel, DsaTopkPrefillSpec);

#[cfg(test)]
mod tests {
    use super::{
        logits_row_stride, DsaTopkPrefillKernelConfig, DsaTopkPrefillKernelInput,
        DsaTopkPrefillSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    const TORCH_BACKEND: &str = "torch";
    const VLLM_BACKEND: &str = "vllm_cuda";
    const QUERY_AXIS: &[f64] = &[
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 127.0, 128.0, 129.0, 131.0, 132.0, 133.0, 255.0,
        256.0, 257.0, 512.0, 1024.0, 1448.0, 2047.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0,
        65536.0,
    ];
    const KEY_AXIS: &[f64] = &[
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 255.0, 256.0, 257.0, 512.0, 1024.0, 1448.0,
        2047.0, 2048.0, 2049.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0, 131072.0, 262144.0,
        524288.0, 1048576.0,
    ];

    fn config() -> DsaTopkPrefillKernelConfig {
        DsaTopkPrefillKernelConfig {
            backends: vec![TORCH_BACKEND, VLLM_BACKEND],
            gpu_name: "NVIDIA H200".to_string(),
            num_sequences: 1,
            top_k: 2048,
            logits_dtype: DType::Fp32,
            index_dtype: "int32".to_string(),
            span_mode: "single_causal_tail".to_string(),
        }
    }

    #[test]
    fn config_and_kind_identity_match_the_prefill_topk_path() {
        let cfg = config();

        assert_eq!(DsaTopkPrefillSpec::KIND, "dsa_topk_prefill");
        assert_eq!(DsaTopkPrefillSpec::profile_kind(), "dsa_topk_prefill");
        assert_eq!(cfg.backends(), &[TORCH_BACKEND, VLLM_BACKEND]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.num_sequences, 1);
        assert_eq!(cfg.top_k, 2048);
        assert_eq!(cfg.logits_dtype, DType::Fp32);
        assert_eq!(cfg.index_dtype, "int32");
        assert_eq!(cfg.span_mode, "single_causal_tail");
    }

    #[test]
    fn describe_config_is_the_exact_static_identity() {
        assert_eq!(
            config().describe_config(),
            serde_json::json!({
                "backends": [TORCH_BACKEND, VLLM_BACKEND],
                "gpu_name": "NVIDIA H200",
                "num_sequences": 1,
                "top_k": 2048,
                "logits_dtype": "fp32",
                "index_dtype": "int32",
                "span_mode": "single_causal_tail",
            })
        );
    }

    #[test]
    fn input_coords_deserialization_and_slot_input_are_physical_m_n() {
        let input: DsaTopkPrefillKernelInput =
            serde_json::from_str(r#"{"num_queries":128,"num_keys":8192}"#).unwrap();

        assert_eq!(&*input.coords(), &[128.0, 8192.0]);
        assert_eq!(
            DsaTopkPrefillKernelInput::coord_field_names(),
            &["num_queries", "num_keys"]
        );

        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_queries": 128, "num_keys": 8192})
        );
    }

    #[test]
    fn sweep_grid_has_the_frozen_physical_axes_and_boundaries() {
        let grid = DsaTopkPrefillSpec::sweep_grid(&config());
        let axes = grid.axes();

        assert_eq!(axes.len(), 2);
        assert_eq!(axes[0], QUERY_AXIS);
        assert_eq!(axes[1], KEY_AXIS);
        assert_eq!(axes[0].len(), 26);
        assert_eq!(axes[1].len(), 26);
        assert!(axes[0].windows(2).all(|pair| pair[0] < pair[1]));
        assert!(axes[1].windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(&axes[0][7..10], &[127.0, 128.0, 129.0]);
        assert_eq!(&axes[0][10..13], &[131.0, 132.0, 133.0]);
        assert_eq!(&axes[0][13..16], &[255.0, 256.0, 257.0]);
        assert_eq!(&axes[0][17..21], &[1024.0, 1448.0, 2047.0, 2048.0]);
        assert_eq!(&axes[1][8..11], &[255.0, 256.0, 257.0]);
        assert_eq!(&axes[1][12..17], &[1024.0, 1448.0, 2047.0, 2048.0, 2049.0]);
        assert_eq!(&axes[1][21..23], &[65536.0, 131072.0]);
        assert_eq!(axes[0].first(), Some(&1.0));
        assert_eq!(axes[0].last(), Some(&65536.0));
        assert_eq!(axes[1].first(), Some(&1.0));
        // The key axis reaches the arch's full 1,048,576-token timing domain.
        assert_eq!(axes[1].last(), Some(&1_048_576.0));
        assert_eq!(axes[0].len() * axes[1].len(), 676);
    }

    #[test]
    fn infeasible_mask_drops_exactly_the_m_greater_than_n_corner() {
        let cfg = config();
        let grid = DsaTopkPrefillSpec::sweep_grid(&cfg);
        let mask = DsaTopkPrefillSpec::infeasible_mask(&cfg, &grid);
        let key_count = grid.axes()[1].len();

        assert_eq!(mask.len(), 676);
        assert_eq!(mask.iter().filter(|&&masked| masked).count(), 264);
        assert_eq!(mask.iter().filter(|&&masked| !masked).count(), 412);

        let masked = |m: f64, n: f64| {
            let i = grid.axes()[0].iter().position(|&value| value == m).unwrap();
            let j = grid.axes()[1].iter().position(|&value| value == n).unwrap();
            mask[i * key_count + j]
        };
        assert!(!masked(1.0, 1.0));
        assert!(!masked(128.0, 128.0));
        assert!(!masked(128.0, 2048.0));
        assert!(masked(1448.0, 1024.0));
        assert!(!masked(1448.0, 1448.0));
        assert!(!masked(1448.0, 2047.0));
        assert!(masked(2047.0, 1448.0));
        assert!(!masked(2047.0, 2047.0));
        assert!(!masked(4096.0, 65536.0));
        assert!(!masked(4096.0, 131072.0));
        assert!(masked(2.0, 1.0));
        assert!(masked(129.0, 128.0));
        assert!(masked(4096.0, 2049.0));
        // The padded-logits allocation staircase. This kernel's operand is
        // `num_queries * logits_row_stride`, so the far corner is dropped for
        // its query count, not its context: 4096 rows survive a 1M context.
        assert!(!masked(4096.0, 1_048_576.0));
        assert!(masked(65536.0, 1_048_576.0));
    }

    #[test]
    fn both_backends_use_physical_bilinear_cache() {
        assert_eq!(
            DsaTopkPrefillSpec::cache_kind(TORCH_BACKEND),
            CacheKind::Cache2DLinear(Extrapolation::Product)
        );
        assert_eq!(
            DsaTopkPrefillSpec::cache_kind(VLLM_BACKEND),
            CacheKind::Cache2DLinear(Extrapolation::Product)
        );
    }

    #[test]
    fn row_stride_is_derived_from_each_physical_key_count() {
        assert_eq!(logits_row_stride(1), 512);
        assert_eq!(logits_row_stride(2048), 2304);
        assert_eq!(logits_row_stride(2049), 2560);
        assert_eq!(logits_row_stride(4096), 4352);
        assert_eq!(logits_row_stride(8192), 8448);
        assert_eq!(logits_row_stride(65536), 65792);
        assert_eq!(logits_row_stride(131072), 131328);
        assert_eq!(logits_row_stride(1_048_576), 1_048_832);
    }

    #[test]
    fn enumerate_matches_the_full_python_wire_schema_before_masking() {
        let cfg = config();
        let grid = DsaTopkPrefillSpec::sweep_grid(&cfg);
        let payloads = DsaTopkPrefillSpec::enumerate(&cfg, &grid, VLLM_BACKEND);

        assert_eq!(payloads.len(), 676);
        let expected_names = [
            "backend",
            "index_dtype",
            "logits_dtype",
            "logits_row_stride",
            "num_keys",
            "num_queries",
            "num_sequences",
            "span_mode",
            "top_k",
        ];
        for payload in &payloads {
            let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
            assert_eq!(names, expected_names);
            assert_eq!(payload.fields().len(), 9);
        }

        assert_payload(&payloads[0], VLLM_BACKEND, 1, 1, 512);
        assert_payload(
            payload_for(&payloads, &grid, 128, 2048),
            VLLM_BACKEND,
            128,
            2048,
            2304,
        );
        assert_payload(
            payload_for(&payloads, &grid, 128, 2049),
            VLLM_BACKEND,
            128,
            2049,
            2560,
        );
        assert_payload(
            payload_for(&payloads, &grid, 128, 8192),
            VLLM_BACKEND,
            128,
            8192,
            8448,
        );
        assert_payload(
            payload_for(&payloads, &grid, 1448, 1448),
            VLLM_BACKEND,
            1448,
            1448,
            1792,
        );
        assert_payload(
            payload_for(&payloads, &grid, 2047, 2047),
            VLLM_BACKEND,
            2047,
            2047,
            2304,
        );
        assert_payload(
            payloads.last().unwrap(),
            VLLM_BACKEND,
            65536,
            1_048_576,
            1_048_832,
        );
    }

    fn payload_for<'a>(
        payloads: &'a [crate::timing::bridge::ArgsPayload],
        grid: &crate::timing::sweep::SweepGrid,
        num_queries: u32,
        num_keys: u32,
    ) -> &'a crate::timing::bridge::ArgsPayload {
        let query_index = grid.axes()[0]
            .iter()
            .position(|&value| value == f64::from(num_queries))
            .unwrap();
        let key_index = grid.axes()[1]
            .iter()
            .position(|&value| value == f64::from(num_keys))
            .unwrap();
        &payloads[query_index * grid.axes()[1].len() + key_index]
    }

    fn assert_payload(
        payload: &crate::timing::bridge::ArgsPayload,
        backend: &str,
        num_queries: u32,
        num_keys: u32,
        stride: u32,
    ) {
        let fields = payload.fields();
        assert_eq!(fields.get("backend"), Some(&Value::from(backend)));
        assert_eq!(fields.get("num_queries"), Some(&Value::from(num_queries)));
        assert_eq!(fields.get("num_keys"), Some(&Value::from(num_keys)));
        assert_eq!(fields.get("num_sequences"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("top_k"), Some(&Value::from(2048_u32)));
        assert_eq!(fields.get("logits_row_stride"), Some(&Value::from(stride)));
        assert_eq!(fields.get("logits_dtype"), Some(&Value::from("fp32")));
        assert_eq!(fields.get("index_dtype"), Some(&Value::from("int32")));
        assert_eq!(
            fields.get("span_mode"),
            Some(&Value::from("single_causal_tail"))
        );
        assert_eq!(payload.backend(), Some(backend));
    }

    #[test]
    fn dtype_tags_expose_fp32_compute_and_no_kv_dtype() {
        let cfg = config();

        assert_eq!(cfg.compute_dtype(), Some(DType::Fp32));
        assert_eq!(cfg.kv_dtype(), None);
    }
}
