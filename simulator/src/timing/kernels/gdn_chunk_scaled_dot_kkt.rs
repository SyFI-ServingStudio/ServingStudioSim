//! Qwen Gated `DeltaNet` chunk-local scaled-dot KKT kernel.
//!
//! Public inputs remain the physical `(num_tokens, num_chunks)` caller shape,
//! while the cache projects them to `(C, D=T/C)`: launch chunks and average
//! valid rows per chunk. `C` captures the launch grid and `D` captures the
//! quadratic local-row work. The projection rectangularizes the physical
//! feasible boundaries `C=T` and `C=ceil(T/64)`, avoiding interpolation through
//! a triangular physical-grid mask while retaining the public Python schema.
//! The explicit `C=5` breakpoint captures stable measured launch-count behavior
//! that linear interpolation between `C=4` and `C=8` cannot represent.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

const MAX_TOKENS: u64 = 262_144;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnChunkScaledDotKktKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_key_heads: Dim,
    pub num_heads: Dim,
    pub key_head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct GdnChunkScaledDotKktKernelInput {
    pub num_tokens: u32,
    pub num_chunks: u32,
}

impl SweepCoords for GdnChunkScaledDotKktKernelInput {
    fn coords(&self) -> Coords {
        Coords::new([
            f64::from(self.num_chunks),
            f64::from(self.num_tokens) / f64::from(self.num_chunks),
        ])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["num_tokens", "num_chunks"]
    }
}

pub struct GdnChunkScaledDotKktSpec;

impl KernelSpec for GdnChunkScaledDotKktSpec {
    type Config = GdnChunkScaledDotKktKernelConfig;
    type Input = GdnChunkScaledDotKktKernelInput;

    const KIND: KernelKind = "gdn_chunk_scaled_dot_kkt";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::chain([Axis::pow2(0, 2), Axis::values([5]), Axis::pow2(3, 18)]),
            Axis::pow2(0, 6),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_2d(|num_chunks, tokens_per_chunk| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "num_chunks is a non-negative Axis sweep coordinate, always a small integer (see the chained axis above)"
            )]
            let num_chunks = num_chunks.round() as u64;
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "tokens_per_chunk is a non-negative Axis::pow2 sweep coordinate, always a small power of two"
            )]
            let tokens_per_chunk = tokens_per_chunk.round() as u64;
            num_chunks
                .checked_mul(tokens_per_chunk)
                .expect("GDN scaled-dot KKT re-axis product must fit u64")
                > MAX_TOKENS
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|num_chunks, tokens_per_chunk| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "num_chunks is a non-negative Axis sweep coordinate, always a small integer (see the chained axis above)"
            )]
            let num_chunks = num_chunks.round() as u64;
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "tokens_per_chunk is a non-negative Axis::pow2 sweep coordinate, always a small power of two"
            )]
            let tokens_per_chunk = tokens_per_chunk.round() as u64;
            let num_tokens = num_chunks
                .checked_mul(tokens_per_chunk)
                .expect("GDN scaled-dot KKT re-axis product must fit u64");
            ArgsPayload::new()
                .with("backend", backend)
                .with(
                    "num_tokens",
                    u32::try_from(num_tokens).expect("token sweep must fit u32"),
                )
                .with(
                    "num_chunks",
                    u32::try_from(num_chunks).expect("chunk sweep must fit u32"),
                )
                .with("num_key_heads", config.num_key_heads.get())
                .with("num_heads", config.num_heads.get())
                .with("key_head_dim", config.key_head_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(GdnChunkScaledDotKktKernel, GdnChunkScaledDotKktSpec);

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashSet};

    use super::{
        GdnChunkScaledDotKktKernelConfig, GdnChunkScaledDotKktKernelInput,
        GdnChunkScaledDotKktSpec, MAX_TOKENS,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::sweep::Axis;
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> GdnChunkScaledDotKktKernelConfig {
        GdnChunkScaledDotKktKernelConfig {
            backends: vec!["torch", "vllm_triton"],
            gpu_name: "NVIDIA H200".to_string(),
            num_key_heads: 16.into(),
            num_heads: 32.into(),
            key_head_dim: 128.into(),
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_description_serde_routing_and_dtype_tags_match_handoff() {
        let cfg = config();

        assert_eq!(GdnChunkScaledDotKktSpec::KIND, "gdn_chunk_scaled_dot_kkt");
        assert_eq!(
            GdnChunkScaledDotKktSpec::profile_kind(),
            "gdn_chunk_scaled_dot_kkt"
        );
        assert_eq!(cfg.backends(), &["torch", "vllm_triton"]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.num_key_heads, 16);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.key_head_dim, 128);
        assert_eq!(cfg.dtype, DType::Bf16);
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
        assert_eq!(
            cfg.describe_config(),
            serde_json::json!({
                "backends": ["torch", "vllm_triton"],
                "gpu_name": "NVIDIA H200",
                "num_key_heads": {"value": 16, "expression": null, "bindings": {}},
                "num_heads": {"value": 32, "expression": null, "bindings": {}},
                "key_head_dim": {"value": 128, "expression": null, "bindings": {}},
                "dtype": "bf16",
            })
        );

        let encoded = serde_json::to_value(&cfg).unwrap();
        let decoded: GdnChunkScaledDotKktKernelConfig =
            serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
    }

    #[test]
    fn physical_input_serde_names_and_slot_payload_survive_reaxis() {
        let input = GdnChunkScaledDotKktKernelInput {
            num_tokens: 128,
            num_chunks: 2,
        };

        assert_eq!(&*input.coords(), &[2.0, 64.0]);
        assert_eq!(
            GdnChunkScaledDotKktKernelInput::coord_field_names(),
            &["num_tokens", "num_chunks"]
        );
        let slot: SlotInput = input.clone().into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_tokens": 128, "num_chunks": 2})
        );
        let round_trip: GdnChunkScaledDotKktKernelInput =
            serde_json::from_value(serde_json::to_value(input).unwrap()).unwrap();
        assert_eq!(&*round_trip.coords(), &[2.0, 64.0]);
    }

    #[test]
    fn cache_coords_cover_edges_and_representative_interior_ratios() {
        for (input, expected) in [
            (
                GdnChunkScaledDotKktKernelInput {
                    num_tokens: 128,
                    num_chunks: 2,
                },
                [2.0, 64.0],
            ),
            (
                GdnChunkScaledDotKktKernelInput {
                    num_tokens: 128,
                    num_chunks: 128,
                },
                [128.0, 1.0],
            ),
            (
                GdnChunkScaledDotKktKernelInput {
                    num_tokens: 384,
                    num_chunks: 48,
                },
                [48.0, 8.0],
            ),
            (
                GdnChunkScaledDotKktKernelInput {
                    num_tokens: 130,
                    num_chunks: 3,
                },
                [3.0, 130.0 / 3.0],
            ),
        ] {
            assert_eq!(&*input.coords(), &expected);
        }
    }

    #[test]
    fn sweep_is_exactly_unique_twenty_by_seven_with_c5_breakpoint() {
        let grid = GdnChunkScaledDotKktSpec::sweep_grid(&config());
        let expected_chunks = [
            1.0, 2.0, 4.0, 5.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0,
            8192.0, 16384.0, 32768.0, 65536.0, 131072.0, 262144.0,
        ];
        let expected_density = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];

        assert_eq!(
            grid.axes(),
            &[expected_chunks.to_vec(), expected_density.to_vec()]
        );
        assert_eq!(grid.axes().len(), 2);
        assert_eq!(grid.axes()[0].len() * grid.axes()[1].len(), 140);
        for axis in grid.axes() {
            assert!(axis.windows(2).all(|pair| pair[0] < pair[1]));
            assert_eq!(
                axis.iter()
                    .map(|value| value.to_bits())
                    .collect::<HashSet<_>>()
                    .len(),
                axis.len()
            );
        }
    }

    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "grid axis values are chunk/token-per-chunk counts (bounded by the sweep grid's \
                  own pow2 axes), well within u64's exact range and always non-negative"
    )]
    fn mask_is_exactly_the_checked_checkpoint_token_cap() {
        let grid = GdnChunkScaledDotKktSpec::sweep_grid(&config());
        let mask = GdnChunkScaledDotKktSpec::infeasible_mask(&config(), &grid);
        let density_axis_len = grid.axes()[1].len();

        assert_eq!(mask.len(), 140);
        assert_eq!(mask.iter().filter(|&&masked| masked).count(), 21);
        assert_eq!(mask.iter().filter(|&&masked| !masked).count(), 119);
        for (chunk_index, &num_chunks) in grid.axes()[0].iter().enumerate() {
            for (density_index, &tokens_per_chunk) in grid.axes()[1].iter().enumerate() {
                let product = (num_chunks as u64)
                    .checked_mul(tokens_per_chunk as u64)
                    .unwrap();
                assert_eq!(
                    mask[chunk_index * density_axis_len + density_index],
                    product > MAX_TOKENS,
                    "C={num_chunks} D={tokens_per_chunk}"
                );
            }
        }

        let masked = |chunks: u64, density: u64| chunks.checked_mul(density).unwrap() > MAX_TOKENS;
        assert!(!masked(262_144, 1));
        assert!(!masked(4_096, 64));
        assert!(masked(262_144, 2));
        assert!(masked(8_192, 64));
    }

    #[test]
    fn both_backends_keep_linear_2d_cache_and_profile_identity() {
        for backend in ["torch", "vllm_triton"] {
            assert_eq!(
                GdnChunkScaledDotKktSpec::cache_kind(backend),
                CacheKind::Cache2DLinear(Extrapolation::Product)
            );
        }
        assert_eq!(
            GdnChunkScaledDotKktSpec::profile_kind(),
            GdnChunkScaledDotKktSpec::KIND
        );
    }

    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "chunks/density here come from Axis::pow2(0, 18) and Axis::pow2(0, 6), so the \
                  f64 values are exact powers of two well within u64's exact range and always \
                  non-negative"
    )]
    fn enumerate_maps_cache_cells_to_exact_physical_python_payloads() {
        let cfg = config();
        let grid = GdnChunkScaledDotKktSpec::sweep_grid(&cfg);
        let payloads = GdnChunkScaledDotKktSpec::enumerate(&cfg, &grid, "vllm_triton");
        let mask = GdnChunkScaledDotKktSpec::infeasible_mask(&cfg, &grid);

        assert_eq!(payloads.len(), 140);
        let expected_names = BTreeSet::from([
            "backend",
            "dtype",
            "key_head_dim",
            "num_chunks",
            "num_heads",
            "num_key_heads",
            "num_tokens",
        ]);
        let mut feasible_pairs = BTreeSet::new();
        for (index, payload) in payloads.iter().enumerate() {
            let names = payload
                .fields()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            assert_eq!(names, expected_names);
            assert_eq!(payload.fields().len(), 7);
            assert_eq!(payload.backend(), Some("vllm_triton"));
            assert_eq!(payload.fields()["num_key_heads"], Value::from(16_u32));
            assert_eq!(payload.fields()["num_heads"], Value::from(32_u32));
            assert_eq!(payload.fields()["key_head_dim"], Value::from(128_u32));
            assert_eq!(payload.fields()["dtype"], Value::from("bf16"));

            let tokens = payload.fields()["num_tokens"].as_u64().unwrap();
            let chunks = payload.fields()["num_chunks"].as_u64().unwrap();
            assert_eq!(tokens % chunks, 0);
            assert_eq!(mask[index], tokens > MAX_TOKENS);
            if !mask[index] {
                assert!(tokens.div_ceil(64) <= chunks && chunks <= tokens);
                assert!(feasible_pairs.insert((tokens, chunks)));
            }
        }
        assert_eq!(feasible_pairs.len(), 119);

        let old_feasible_pairs = Axis::pow2(0, 18)
            .into_iter()
            .flat_map(|chunks| {
                Axis::pow2(0, 6).into_iter().filter_map(move |density| {
                    let tokens = chunks as u64 * density as u64;
                    (tokens <= MAX_TOKENS).then_some((tokens, chunks as u64))
                })
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(old_feasible_pairs.len(), 112);
        assert!(old_feasible_pairs.is_subset(&feasible_pairs));
        assert_eq!(
            feasible_pairs
                .difference(&old_feasible_pairs)
                .copied()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                (5, 5),
                (10, 5),
                (20, 5),
                (40, 5),
                (80, 5),
                (160, 5),
                (320, 5),
            ])
        );

        for density_index in 0..7 {
            let payload = &payloads[3 * 7 + density_index];
            assert_eq!(payload.fields().len(), 7);
            assert_eq!(payload.fields()["num_chunks"], Value::from(5_u32));
            assert_eq!(
                payload.fields()["num_tokens"],
                Value::from(5_u32 * (1_u32 << density_index))
            );
        }

        let qwen_index = 7 + 6;
        assert_eq!(
            payloads[qwen_index].fields()["num_tokens"],
            Value::from(128_u32)
        );
        assert_eq!(
            payloads[qwen_index].fields()["num_chunks"],
            Value::from(2_u32)
        );
        let minimum_at_cap_index = 13 * 7 + 6;
        assert_eq!(
            payloads[minimum_at_cap_index].fields()["num_tokens"],
            Value::from(262_144_u32)
        );
        assert_eq!(
            payloads[minimum_at_cap_index].fields()["num_chunks"],
            Value::from(4_096_u32)
        );
        assert_eq!(
            payloads[133].fields()["num_tokens"],
            Value::from(262_144_u32)
        );
        assert_eq!(
            payloads[133].fields()["num_chunks"],
            Value::from(262_144_u32)
        );
        assert_eq!(
            payloads[139].fields()["num_tokens"],
            Value::from(16_777_216_u32)
        );
        assert!(mask[139]);
    }
}
