//! Qwen Gated `DeltaNet` fused causal-convolution fresh-prefill kernel.
//!
//! The cache uses the physical `(batch_size, sequence_length)` caller shape.
//! Power-of-two sequence lengths retain every previously accepted profile row,
//! while an explicit `L=9` point captures the measured first post-`BLOCK_M=8`
//! launch cliff. The rectangular sweep is capped at the checkpoint's 262,144
//! total tokens through `infeasible_mask`. `state_dtype` remains an explicit
//! profiler/cache key, but is not a standard KV capability axis.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

const MAX_TOTAL_TOKENS: u64 = 262_144;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnCausalConvPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub channels: Dim,
    pub kernel_size: Dim,
    #[compute_dtype]
    pub dtype: DType,
    pub state_dtype: DType,
}

/// Physical caller input and physical `(B, L)` cache coordinates.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct GdnCausalConvPrefillKernelInput {
    pub batch_size: u32,
    pub sequence_length: u32,
}

impl SweepCoords for GdnCausalConvPrefillKernelInput {
    fn coords(&self) -> Coords {
        Coords::new([f64::from(self.batch_size), f64::from(self.sequence_length)])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["batch_size", "sequence_length"]
    }
}

pub struct GdnCausalConvPrefillSpec;

impl KernelSpec for GdnCausalConvPrefillSpec {
    type Config = GdnCausalConvPrefillKernelConfig;
    type Input = GdnCausalConvPrefillKernelInput;

    const KIND: KernelKind = "gdn_causal_conv_prefill";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        let sequence_lengths =
            Axis::chain([Axis::pow2(0, 3), Axis::values([9]), Axis::pow2(4, 18)]);
        SweepGrid::new(vec![Axis::pow2(0, 8), sequence_lengths])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_2d(|batch_size, sequence_length| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "batch_size is a non-negative Axis::pow2 sweep coordinate, always a small power of two"
            )]
            let batch_size = batch_size.round() as u64;
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "sequence_length is a non-negative Axis sweep coordinate bounded by MAX_TOTAL_TOKENS below"
            )]
            let sequence_length = sequence_length.round() as u64;
            batch_size
                .checked_mul(sequence_length)
                .expect("GDN causal-convolution prefill sweep product must fit u64")
                > MAX_TOTAL_TOKENS
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|batch_size, sequence_length| {
            ArgsPayload::new()
                .with("backend", backend)
                .with(
                    "batch_size",
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "batch_size is a non-negative sweep coordinate capped under MAX_TOTAL_TOKENS \
                                  by infeasible_mask's checked_mul, so rounding to u64 is always in range \
                                  ahead of the checked try_from into u32"
                    )]
                    {
                        u32::try_from(batch_size.round() as u64).expect("batch sweep must fit u32")
                    },
                )
                .with(
                    "sequence_length",
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "sequence_length is a non-negative sweep coordinate capped under MAX_TOTAL_TOKENS \
                                  by infeasible_mask's checked_mul, so rounding to u64 is always in range \
                                  ahead of the checked try_from into u32"
                    )]
                    {
                        u32::try_from(sequence_length.round() as u64)
                            .expect("sequence-length sweep must fit u32")
                    },
                )
                .with("channels", config.channels.get())
                .with("kernel_size", config.kernel_size.get())
                .with("dtype", config.dtype.as_str())
                .with("state_dtype", config.state_dtype.as_str())
        })
    }
}

register_kernel!(GdnCausalConvPrefillKernel, GdnCausalConvPrefillSpec);

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{
        GdnCausalConvPrefillKernelConfig, GdnCausalConvPrefillKernelInput,
        GdnCausalConvPrefillSpec, MAX_TOTAL_TOKENS,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> GdnCausalConvPrefillKernelConfig {
        GdnCausalConvPrefillKernelConfig {
            backends: vec!["torch", "vllm_triton"],
            gpu_name: "NVIDIA H200".to_string(),
            channels: 8192.into(),
            kernel_size: 4.into(),
            dtype: DType::Bf16,
            state_dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_description_serde_and_dtype_tags_match_handoff() {
        let cfg = config();

        assert_eq!(GdnCausalConvPrefillSpec::KIND, "gdn_causal_conv_prefill");
        assert_eq!(
            GdnCausalConvPrefillSpec::profile_kind(),
            "gdn_causal_conv_prefill"
        );
        assert_eq!(cfg.backends(), &["torch", "vllm_triton"]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.channels, 8192);
        assert_eq!(cfg.kernel_size, 4);
        assert_eq!(cfg.dtype, DType::Bf16);
        assert_eq!(cfg.state_dtype, DType::Bf16);
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
        assert_eq!(
            cfg.describe_config(),
            serde_json::json!({
                "backends": ["torch", "vllm_triton"],
                "gpu_name": "NVIDIA H200",
                "channels": {"value": 8192, "expression": null, "bindings": {}},
                "kernel_size": {"value": 4, "expression": null, "bindings": {}},
                "dtype": "bf16",
                "state_dtype": "bf16",
            })
        );

        let encoded = serde_json::to_value(&cfg).unwrap();
        let decoded: GdnCausalConvPrefillKernelConfig =
            serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
    }

    #[test]
    fn physical_input_serde_coords_names_and_slot_payload_are_exact() {
        let input = GdnCausalConvPrefillKernelInput {
            batch_size: 3,
            sequence_length: 17,
        };

        assert_eq!(&*input.coords(), &[3.0, 17.0]);
        assert_eq!(
            GdnCausalConvPrefillKernelInput::coord_field_names(),
            &["batch_size", "sequence_length"]
        );
        let slot: SlotInput = input.clone().into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"batch_size": 3, "sequence_length": 17})
        );
        let round_trip: GdnCausalConvPrefillKernelInput =
            serde_json::from_value(serde_json::to_value(input).unwrap()).unwrap();
        assert_eq!(&*round_trip.coords(), &[3.0, 17.0]);
    }

    #[test]
    fn sweep_is_exactly_nine_by_twenty_physical_axes() {
        let grid = GdnCausalConvPrefillSpec::sweep_grid(&config());

        assert_eq!(grid.axes().len(), 2);
        assert_eq!(
            grid.axes()[0],
            [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
        );
        assert_eq!(
            grid.axes()[1],
            [
                1.0, 2.0, 4.0, 8.0, 9.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0,
                4096.0, 8192.0, 16_384.0, 32_768.0, 65_536.0, 131_072.0, 262_144.0,
            ]
        );
        assert_eq!(
            grid.axes()[1]
                .iter()
                .filter(|&&length| length == 9.0)
                .count(),
            1
        );
        assert_eq!(grid.axes()[0].len() * grid.axes()[1].len(), 180);
    }

    #[test]
    fn infeasible_mask_is_exactly_the_checkpoint_total_token_cap() {
        let cfg = config();
        let grid = GdnCausalConvPrefillSpec::sweep_grid(&cfg);
        let mask = GdnCausalConvPrefillSpec::infeasible_mask(&cfg, &grid);
        let sequence_axis = &grid.axes()[1];

        assert_eq!(mask.len(), 180);
        assert_eq!(mask.iter().filter(|&&masked| masked).count(), 36);
        assert_eq!(mask.iter().filter(|&&masked| !masked).count(), 144);
        for (batch_index, &batch) in grid.axes()[0].iter().enumerate() {
            for (length_index, &sequence_length) in sequence_axis.iter().enumerate() {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "batch,sequence_length are non-negative Axis sweep coordinates from \
                              config(), bounded well under u64 range"
                )]
                let expected_masked = (batch as u64) * (sequence_length as u64) > MAX_TOTAL_TOKENS;
                assert_eq!(
                    mask[batch_index * sequence_axis.len() + length_index],
                    expected_masked,
                    "B={batch} L={sequence_length}"
                );
            }
        }
        let length_9_index = sequence_axis
            .iter()
            .position(|&value| value == 9.0)
            .unwrap();
        for batch_index in 0..grid.axes()[0].len() {
            assert!(!mask[batch_index * sequence_axis.len()]);
            assert!(!mask[batch_index * sequence_axis.len() + length_9_index]);
        }
        assert!(!mask[19]); // B=1,L=262144 is exactly the cap.
        assert!(mask[39]); // B=2,L=262144 exceeds the cap.
        assert!(!mask[8 * 20 + 11]); // B=256,L=1024 is exactly the cap.
        assert!(mask[8 * 20 + 12]); // B=256,L=2048 exceeds the cap.
    }

    #[test]
    fn both_backends_use_linear_2d_cache() {
        for backend in ["torch", "vllm_triton"] {
            assert_eq!(
                GdnCausalConvPrefillSpec::cache_kind(backend),
                CacheKind::Cache2DLinear(Extrapolation::Product)
            );
        }
    }

    #[test]
    fn enumerate_emits_exact_python_fields_and_backend_routing() {
        let cfg = config();
        let grid = GdnCausalConvPrefillSpec::sweep_grid(&cfg);
        let payloads = GdnCausalConvPrefillSpec::enumerate(&cfg, &grid, "vllm_triton");

        assert_eq!(payloads.len(), 180);
        let expected_names = BTreeSet::from([
            "backend",
            "batch_size",
            "channels",
            "dtype",
            "kernel_size",
            "sequence_length",
            "state_dtype",
        ]);
        for payload in &payloads {
            let names = payload
                .fields()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            assert_eq!(names, expected_names);
            assert_eq!(payload.fields().len(), 7);
            assert_eq!(payload.backend(), Some("vllm_triton"));
            assert_eq!(payload.fields()["channels"], Value::from(8192_u32));
            assert_eq!(payload.fields()["kernel_size"], Value::from(4_u32));
            assert_eq!(payload.fields()["dtype"], Value::from("bf16"));
            assert_eq!(payload.fields()["state_dtype"], Value::from("bf16"));
        }
    }

    #[test]
    fn enumeration_is_physical_and_preserves_prior_rows_plus_nine_breakpoints() {
        let cfg = config();
        let grid = GdnCausalConvPrefillSpec::sweep_grid(&cfg);
        let mask = GdnCausalConvPrefillSpec::infeasible_mask(&cfg, &grid);
        let payloads = GdnCausalConvPrefillSpec::enumerate(&cfg, &grid, "vllm_triton");
        let cache_cells = grid.expand_2d(|batch, sequence_length| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "batch,sequence_length are non-negative Axis sweep coordinates from config(), \
                          bounded well under u64 range"
            )]
            (batch as u64, sequence_length as u64)
        });
        let mut feasible = BTreeSet::new();

        for ((payload, &masked), &(batch, sequence_length)) in
            payloads.iter().zip(&mask).zip(&cache_cells)
        {
            let payload_batch = payload.fields()["batch_size"].as_u64().unwrap();
            assert_eq!(payload_batch, batch);
            assert_eq!(
                payload.fields()["sequence_length"].as_u64().unwrap(),
                sequence_length
            );
            assert_eq!(
                masked,
                batch * sequence_length > MAX_TOTAL_TOKENS,
                "B={batch} L={sequence_length}"
            );
            if !masked {
                feasible.insert((batch, sequence_length));
            }
        }

        let prior_feasible = (0_u32..=8)
            .flat_map(|batch_log2| {
                (0_u32..=(18 - batch_log2))
                    .map(move |length_log2| (1_u64 << batch_log2, 1_u64 << length_log2))
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(prior_feasible.len(), 135);
        assert!(prior_feasible.is_subset(&feasible));
        assert_eq!(feasible.len(), 144);
        let added = feasible
            .difference(&prior_feasible)
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(
            added,
            [1_u64, 2, 4, 8, 16, 32, 64, 128, 256]
                .into_iter()
                .map(|batch| (batch, 9))
                .collect::<Vec<_>>()
        );

        // Row-major physical coordinates: B=1,L=128 and B=256,L=1024.
        assert_eq!(payloads[8].fields()["batch_size"], Value::from(1_u32));
        assert_eq!(
            payloads[8].fields()["sequence_length"],
            Value::from(128_u32)
        );
        assert_eq!(payloads[171].fields()["batch_size"], Value::from(256_u32));
        assert_eq!(
            payloads[171].fields()["sequence_length"],
            Value::from(1024_u32)
        );
        assert!(!mask[8]);
        assert!(!mask[171]);

        // Masked payloads remain their exact physical grid coordinates; no
        // synthetic guard shape is introduced during enumeration.
        assert_eq!(payloads[179].fields()["batch_size"], Value::from(256_u32));
        assert_eq!(
            payloads[179].fields()["sequence_length"],
            Value::from(262_144_u32)
        );
        assert!(mask[179]);
    }
}
