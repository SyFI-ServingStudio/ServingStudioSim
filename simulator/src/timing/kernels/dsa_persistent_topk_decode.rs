//! GLM-5.2 DSA persistent decode top-k kernel.
//!
//! The cache stays on physical `(batch_size, context_len)` coordinates. Measured
//! `next_n=2` R.4 failures added B23/B24/B39 and C256/C2050/C2897/C5792 to repair
//! speculative interpolation while retaining all prior `next_n=1` and production-
//! boundary samples. This changes neither the cache algorithm nor the public
//! physical query contract.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaPersistentTopkDecodeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub next_n: u32,
    pub max_model_len: Dim,
    pub top_k: u32,
    pub logits_row_stride: Dim,
    #[compute_dtype]
    pub logits_dtype: DType,
    pub index_dtype: String,
    pub context_mode: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DsaPersistentTopkDecodeKernelInput {
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

pub struct DsaPersistentTopkDecodeSpec;

impl KernelSpec for DsaPersistentTopkDecodeSpec {
    type Config = DsaPersistentTopkDecodeKernelConfig;
    type Input = DsaPersistentTopkDecodeKernelInput;

    const KIND: KernelKind = "dsa_persistent_topk_decode";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values([
                1, 2, 4, 8, 12, 15, 16, 17, 23, 24, 31, 32, 33, 39, 46, 48, 64, 65, 66, 67, 92, 96,
                127, 128, 129, 130, 131, 132, 133, 197, 198, 199, 255, 256,
            ]),
            Axis::values([
                0, 1, 2, 128, 256, 512, 1024, 2046, 2047, 2048, 2049, 2050, 2897, 4096, 5792, 8191,
                8192, 8193, 16384, 32767, 32768, 32769, 65536, 131072, 262144, 524288, 1048576,
            ]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // Top-k runs over `batch_size` rows of `context_len` logits each, so the
        // work is the product of the axes.
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        let min_context_len = f64::from(config.next_n.saturating_sub(1));
        let max_model_len = f64::from(config.max_model_len.get());
        // The runner allocates the full padded logits block, whose row width is
        // `logits_row_stride` regardless of the context actually used — so the
        // cost is set by the batch, and raising the stride to 1M raises it
        // everywhere on the grid.
        let padded_row_bytes = f64::from(config.logits_row_stride.get())
            * f64::from(DType::Fp32.size_bytes())
            * f64::from(config.next_n);
        grid.expand_2d(|batch_size, context_len| {
            context_len < min_context_len
                || context_len > max_model_len
                || batch_size * padded_row_bytes > MAX_PROFILE_ALLOCATION_BYTES
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
                .with("top_k", config.top_k)
                .with("logits_row_stride", config.logits_row_stride.get())
                .with("logits_dtype", config.logits_dtype.as_str())
                .with("index_dtype", config.index_dtype.clone())
                .with("context_mode", config.context_mode.clone())
        })
    }
}

register_kernel!(DsaPersistentTopkDecodeKernel, DsaPersistentTopkDecodeSpec);

#[cfg(test)]
mod tests {
    use super::{
        DsaPersistentTopkDecodeKernelConfig, DsaPersistentTopkDecodeKernelInput,
        DsaPersistentTopkDecodeSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{Dim, SlotInput, SweepCoords};
    use serde_json::Value;

    const TORCH_BACKEND: &str = "torch";
    const VLLM_BACKEND: &str = "vllm_cuda";
    const BATCH_AXIS: &[f64] = &[
        1.0, 2.0, 4.0, 8.0, 12.0, 15.0, 16.0, 17.0, 23.0, 24.0, 31.0, 32.0, 33.0, 39.0, 46.0, 48.0,
        64.0, 65.0, 66.0, 67.0, 92.0, 96.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 197.0,
        198.0, 199.0, 255.0, 256.0,
    ];
    const CONTEXT_AXIS: &[f64] = &[
        0.0, 1.0, 2.0, 128.0, 256.0, 512.0, 1024.0, 2046.0, 2047.0, 2048.0, 2049.0, 2050.0, 2897.0,
        4096.0, 5792.0, 8191.0, 8192.0, 8193.0, 16384.0, 32767.0, 32768.0, 32769.0, 65536.0,
        131072.0, 262144.0, 524288.0, 1048576.0,
    ];

    /// The arch's full DSA timing domain, which is also the padded logits row
    /// width the production config carries.
    const FULL_MAX_MODEL_LEN: u32 = 1_048_576;

    fn config(next_n: u32) -> DsaPersistentTopkDecodeKernelConfig {
        DsaPersistentTopkDecodeKernelConfig {
            backends: vec![TORCH_BACKEND, VLLM_BACKEND],
            gpu_name: "NVIDIA H200".to_string(),
            next_n,
            max_model_len: Dim::param("max_model_len", FULL_MAX_MODEL_LEN),
            top_k: 2048,
            logits_row_stride: Dim::param("logits_row_stride", FULL_MAX_MODEL_LEN),
            logits_dtype: DType::Fp32,
            index_dtype: "int32".to_string(),
            context_mode: "uniform".to_string(),
        }
    }

    #[test]
    fn config_and_kind_identity_match_both_production_dispatches() {
        for next_n in [1, 2] {
            let cfg = config(next_n);

            assert_eq!(
                DsaPersistentTopkDecodeSpec::KIND,
                "dsa_persistent_topk_decode"
            );
            assert_eq!(
                DsaPersistentTopkDecodeSpec::profile_kind(),
                "dsa_persistent_topk_decode"
            );
            assert_eq!(cfg.backends(), &[TORCH_BACKEND, VLLM_BACKEND]);
            assert_eq!(cfg.gpu_name(), "NVIDIA H200");
            assert_eq!(cfg.next_n, next_n);
            assert_eq!(cfg.max_model_len, FULL_MAX_MODEL_LEN);
            assert_eq!(cfg.top_k, 2048);
            assert_eq!(cfg.logits_row_stride, FULL_MAX_MODEL_LEN);
            assert_eq!(cfg.logits_dtype, DType::Fp32);
            assert_eq!(cfg.index_dtype, "int32");
            assert_eq!(cfg.context_mode, "uniform");
        }
    }

    #[test]
    fn describe_config_preserves_rich_deployment_dimensions() {
        assert_eq!(
            config(2).describe_config(),
            serde_json::json!({
                "backends": [TORCH_BACKEND, VLLM_BACKEND],
                "gpu_name": "NVIDIA H200",
                "next_n": 2,
                "max_model_len": {
                    "value": 1048576,
                    "expression": "max_model_len",
                    "bindings": {"max_model_len": 1048576},
                },
                "top_k": 2048,
                "logits_row_stride": {
                    "value": 1048576,
                    "expression": "logits_row_stride",
                    "bindings": {"logits_row_stride": 1048576},
                },
                "logits_dtype": "fp32",
                "index_dtype": "int32",
                "context_mode": "uniform",
            })
        );
    }

    #[test]
    fn inputs_are_physical_batch_context_coordinates_and_slot_payloads() {
        for json in [
            r#"{"batch_size":16,"context_len":8192}"#,
            r#"{"batch_size":1,"context_len":0}"#,
        ] {
            let input: DsaPersistentTopkDecodeKernelInput = serde_json::from_str(json).unwrap();
            assert_eq!(
                DsaPersistentTopkDecodeKernelInput::coord_field_names(),
                &["batch_size", "context_len"]
            );

            let expected = serde_json::from_str::<Value>(json).unwrap();
            let expected_coords = [
                expected["batch_size"].as_f64().unwrap(),
                expected["context_len"].as_f64().unwrap(),
            ];
            assert_eq!(&*input.coords(), &expected_coords);

            let slot: SlotInput = input.into();
            assert_eq!(serde_json::to_value(slot).unwrap(), expected);
        }
    }

    #[test]
    fn sweep_grid_has_every_measured_batch_cliff_wave_and_context_boundary() {
        let grid = DsaPersistentTopkDecodeSpec::sweep_grid(&config(1));
        let axes = grid.axes();

        assert_eq!(axes.len(), 2);
        assert_eq!(axes[0], BATCH_AXIS);
        assert_eq!(axes[1], CONTEXT_AXIS);
        assert_eq!(axes[0].len(), 34);
        assert_eq!(axes[1].len(), 27);
        assert!(axes[0].windows(2).all(|pair| pair[0] < pair[1]));
        assert!(axes[1].windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(&axes[0][3..6], &[8.0, 12.0, 15.0]);
        assert_eq!(&axes[0][5..8], &[15.0, 16.0, 17.0]);
        assert_eq!(&axes[0][7..11], &[17.0, 23.0, 24.0, 31.0]);
        assert_eq!(&axes[0][10..13], &[31.0, 32.0, 33.0]);
        assert_eq!(&axes[0][12..15], &[33.0, 39.0, 46.0]);
        assert_eq!(&axes[0][13..17], &[39.0, 46.0, 48.0, 64.0]);
        assert_eq!(&axes[0][17..20], &[65.0, 66.0, 67.0]);
        assert_eq!(&axes[0][19..23], &[67.0, 92.0, 96.0, 127.0]);
        assert_eq!(&axes[0][22..25], &[127.0, 128.0, 129.0]);
        assert_eq!(&axes[0][24..27], &[129.0, 130.0, 131.0]);
        assert_eq!(&axes[0][26..29], &[131.0, 132.0, 133.0]);
        assert_eq!(&axes[0][29..32], &[197.0, 198.0, 199.0]);
        assert_eq!(&axes[1][3..6], &[128.0, 256.0, 512.0]);
        assert_eq!(&axes[1][6..11], &[1024.0, 2046.0, 2047.0, 2048.0, 2049.0]);
        assert_eq!(&axes[1][8..11], &[2047.0, 2048.0, 2049.0]);
        assert_eq!(&axes[1][9..14], &[2048.0, 2049.0, 2050.0, 2897.0, 4096.0]);
        assert_eq!(&axes[1][13..16], &[4096.0, 5792.0, 8191.0]);
        assert_eq!(&axes[1][15..18], &[8191.0, 8192.0, 8193.0]);
        assert_eq!(&axes[1][19..22], &[32767.0, 32768.0, 32769.0]);
        assert_eq!(axes[0].first(), Some(&1.0));
        assert_eq!(axes[0].last(), Some(&256.0));
        assert_eq!(axes[1].first(), Some(&0.0));
        assert_eq!(axes[1].last(), Some(&1_048_576.0));
        assert_eq!(axes[0].len() * axes[1].len(), 918);
    }

    #[test]
    fn infeasible_mask_respects_native_speculation_minimum_and_maximum() {
        let next_one = config(1);
        let grid = DsaPersistentTopkDecodeSpec::sweep_grid(&next_one);
        let next_one_mask = DsaPersistentTopkDecodeSpec::infeasible_mask(&next_one, &grid);
        assert_eq!(next_one_mask.len(), 918);
        assert_eq!(next_one_mask.iter().filter(|&&masked| masked).count(), 0);
        let next_one_feasible = next_one_mask.iter().filter(|&&masked| !masked).count();
        assert_eq!(next_one_feasible, 918);

        let next_two = config(2);
        let next_two_mask = DsaPersistentTopkDecodeSpec::infeasible_mask(&next_two, &grid);
        assert_eq!(next_two_mask.len(), 918);
        assert_eq!(next_two_mask.iter().filter(|&&masked| masked).count(), 34);
        let next_two_feasible = next_two_mask.iter().filter(|&&masked| !masked).count();
        assert_eq!(next_two_feasible, 884);
        assert_eq!(next_one_feasible + next_two_feasible, 1802);

        let context_count = grid.axes()[1].len();
        let masked = |batch_size: f64, context_len: f64| {
            let i = grid.axes()[0]
                .iter()
                .position(|&value| value == batch_size)
                .unwrap();
            let j = grid.axes()[1]
                .iter()
                .position(|&value| value == context_len)
                .unwrap();
            next_two_mask[i * context_count + j]
        };
        for batch_size in BATCH_AXIS {
            assert!(masked(*batch_size, 0.0));
            assert!(!masked(*batch_size, 1.0));
            assert!(!masked(*batch_size, 131072.0));
            // The padded logits block is `batch x logits_row_stride`, so even a
            // 1M stride at the largest batch on this axis stays under the
            // allocation budget: nothing here is dropped for size.
            assert!(!masked(*batch_size, 1_048_576.0));
        }
        for batch_size in [12.0, 23.0, 24.0, 39.0, 46.0, 48.0, 92.0, 96.0, 130.0] {
            assert!(masked(batch_size, 0.0));
            assert!(!masked(batch_size, 1.0));
            assert!(!masked(batch_size, 131072.0));
        }
    }

    #[test]
    fn both_backends_use_physical_bilinear_cache() {
        assert_eq!(
            DsaPersistentTopkDecodeSpec::cache_kind(TORCH_BACKEND),
            CacheKind::Cache2DLinear(Extrapolation::Product)
        );
        assert_eq!(
            DsaPersistentTopkDecodeSpec::cache_kind(VLLM_BACKEND),
            CacheKind::Cache2DLinear(Extrapolation::Product)
        );
    }

    #[test]
    fn enumerate_matches_the_full_python_wire_schema_before_masking() {
        for next_n in [1, 2] {
            let cfg = config(next_n);
            let grid = DsaPersistentTopkDecodeSpec::sweep_grid(&cfg);
            let payloads = DsaPersistentTopkDecodeSpec::enumerate(&cfg, &grid, VLLM_BACKEND);

            assert_eq!(payloads.len(), 918);
            let expected_names = [
                "backend",
                "batch_size",
                "context_len",
                "context_mode",
                "index_dtype",
                "logits_dtype",
                "logits_row_stride",
                "max_model_len",
                "next_n",
                "top_k",
            ];
            for payload in &payloads {
                let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
                assert_eq!(names, expected_names);
                assert_eq!(payload.fields().len(), 10);
            }

            assert_payload(&payloads[0], 1, 0, next_n);
            assert_payload(payload_for(&payloads, &grid, 16, 8192), 16, 8192, next_n);
            let (legacy_lower_batch, legacy_upper_batch) =
                if next_n == 1 { (32, 33) } else { (16, 17) };
            assert_payload(
                payload_for(&payloads, &grid, legacy_lower_batch, 2048),
                legacy_lower_batch,
                2048,
                next_n,
            );
            assert_payload(
                payload_for(&payloads, &grid, legacy_upper_batch, 2049),
                legacy_upper_batch,
                2049,
                next_n,
            );
            for batch_size in [12, 23, 24, 39, 46, 48, 92, 96, 130] {
                assert_payload(
                    payload_for(&payloads, &grid, batch_size, 2046),
                    batch_size,
                    2046,
                    next_n,
                );
            }
            for (batch_size, context_len) in [
                (23, 4096),
                (24, 131072),
                (32, 2897),
                (39, 5792),
                (39, 65536),
                (64, 2050),
                (66, 256),
                (128, 2050),
                (256, 2050),
            ] {
                assert_payload(
                    payload_for(&payloads, &grid, batch_size, context_len),
                    batch_size,
                    context_len,
                    next_n,
                );
            }
            assert_payload(
                payload_for(&payloads, &grid, 132, 32768),
                132,
                32768,
                next_n,
            );
            assert_payload(payload_for(&payloads, &grid, 199, 8192), 199, 8192, next_n);
            assert_payload(payloads.last().unwrap(), 256, 1_048_576, next_n);
        }
    }

    fn payload_for<'a>(
        payloads: &'a [crate::timing::bridge::ArgsPayload],
        grid: &crate::timing::sweep::SweepGrid,
        batch_size: u32,
        context_len: u32,
    ) -> &'a crate::timing::bridge::ArgsPayload {
        let batch_index = grid.axes()[0]
            .iter()
            .position(|&value| value == f64::from(batch_size))
            .unwrap();
        let context_index = grid.axes()[1]
            .iter()
            .position(|&value| value == f64::from(context_len))
            .unwrap();
        &payloads[batch_index * grid.axes()[1].len() + context_index]
    }

    fn assert_payload(
        payload: &crate::timing::bridge::ArgsPayload,
        batch_size: u32,
        context_len: u32,
        next_n: u32,
    ) {
        let fields = payload.fields();
        assert_eq!(fields.get("backend"), Some(&Value::from(VLLM_BACKEND)));
        assert_eq!(fields.get("batch_size"), Some(&Value::from(batch_size)));
        assert_eq!(fields.get("context_len"), Some(&Value::from(context_len)));
        assert_eq!(fields.get("next_n"), Some(&Value::from(next_n)));
        assert_eq!(
            fields.get("max_model_len"),
            Some(&Value::from(FULL_MAX_MODEL_LEN))
        );
        assert_eq!(fields.get("top_k"), Some(&Value::from(2048_u32)));
        assert_eq!(
            fields.get("logits_row_stride"),
            Some(&Value::from(FULL_MAX_MODEL_LEN))
        );
        assert_eq!(fields.get("logits_dtype"), Some(&Value::from("fp32")));
        assert_eq!(fields.get("index_dtype"), Some(&Value::from("int32")));
        assert_eq!(fields.get("context_mode"), Some(&Value::from("uniform")));
        assert_eq!(payload.backend(), Some(VLLM_BACKEND));
    }

    #[test]
    fn dtype_tags_expose_fp32_compute_and_no_kv_dtype() {
        let cfg = config(1);

        assert_eq!(cfg.compute_dtype(), Some(DType::Fp32));
        assert_eq!(cfg.kv_dtype(), None);
    }
}
