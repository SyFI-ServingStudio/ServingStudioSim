//! Qwen Gated `DeltaNet` scalar chunk-local cumulative-sum kernel.
//!
//! Public inputs remain the physical `(num_tokens, num_chunks)` caller shape,
//! while the cache projects them to `(C, D=T/C)`: launch chunks and average
//! tokens per chunk. These coordinates follow the kernel's launch/data scaling
//! and turn the physical feasible boundaries `C=T` and `C=ceil(T/64)` into the
//! approximately constant edges `D=1` and `D=64`. This avoids interpolation
//! through the triangular physical-grid mask. The rectangular re-axis is a
//! bijection over the 112 accepted power-of-two profile rows, so it requires no
//! new measurements.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

const MAX_TOKENS: u64 = 262_144;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnChunkLocalCumsumKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct GdnChunkLocalCumsumKernelInput {
    pub num_tokens: u32,
    pub num_chunks: u32,
}

impl SweepCoords for GdnChunkLocalCumsumKernelInput {
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

pub struct GdnChunkLocalCumsumSpec;

impl KernelSpec for GdnChunkLocalCumsumSpec {
    type Config = GdnChunkLocalCumsumKernelConfig;
    type Input = GdnChunkLocalCumsumKernelInput;

    const KIND: KernelKind = "gdn_chunk_local_cumsum";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::pow2(0, 18), Axis::pow2(0, 6)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_2d(|num_chunks, tokens_per_chunk| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "num_chunks is a non-negative Axis::pow2 sweep coordinate, always a small power of two"
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
                .expect("GDN chunk-cumsum re-axis product must fit u64")
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
                reason = "num_chunks is a non-negative Axis::pow2 sweep coordinate, always a small power of two"
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
                .expect("GDN chunk-cumsum re-axis product must fit u64");
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
                .with("num_heads", config.num_heads.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(GdnChunkLocalCumsumKernel, GdnChunkLocalCumsumSpec);

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::{
        GdnChunkLocalCumsumKernelConfig, GdnChunkLocalCumsumKernelInput, GdnChunkLocalCumsumSpec,
        MAX_TOKENS,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> GdnChunkLocalCumsumKernelConfig {
        GdnChunkLocalCumsumKernelConfig {
            backends: vec!["torch", "vllm_triton"],
            gpu_name: "NVIDIA H200".to_string(),
            num_heads: 32.into(),
            dtype: DType::Fp32,
        }
    }

    #[test]
    fn config_identity_description_serde_routing_and_dtype_tags_match_handoff() {
        let cfg = config();

        assert_eq!(GdnChunkLocalCumsumSpec::KIND, "gdn_chunk_local_cumsum");
        assert_eq!(
            GdnChunkLocalCumsumSpec::profile_kind(),
            "gdn_chunk_local_cumsum"
        );
        assert_eq!(cfg.backends(), &["torch", "vllm_triton"]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.dtype, DType::Fp32);
        assert_eq!(cfg.compute_dtype(), Some(DType::Fp32));
        assert_eq!(cfg.kv_dtype(), None);
        assert_eq!(
            cfg.describe_config(),
            serde_json::json!({
                "backends": ["torch", "vllm_triton"],
                "gpu_name": "NVIDIA H200",
                "num_heads": {"value": 32, "expression": null, "bindings": {}},
                "dtype": "fp32",
            })
        );

        let encoded = serde_json::to_value(&cfg).unwrap();
        let decoded: GdnChunkLocalCumsumKernelConfig =
            serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
    }

    #[test]
    fn physical_input_serde_names_and_slot_payload_survive_reaxis() {
        let input = GdnChunkLocalCumsumKernelInput {
            num_tokens: 128,
            num_chunks: 2,
        };

        assert_eq!(&*input.coords(), &[2.0, 64.0]);
        assert_eq!(
            GdnChunkLocalCumsumKernelInput::coord_field_names(),
            &["num_tokens", "num_chunks"]
        );
        let slot: SlotInput = input.clone().into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_tokens": 128, "num_chunks": 2})
        );
        let round_trip: GdnChunkLocalCumsumKernelInput =
            serde_json::from_value(serde_json::to_value(input).unwrap()).unwrap();
        assert_eq!(&*round_trip.coords(), &[2.0, 64.0]);
    }

    #[test]
    fn cache_coords_cover_edges_and_representative_interior_ratios() {
        for (input, expected) in [
            (
                GdnChunkLocalCumsumKernelInput {
                    num_tokens: 128,
                    num_chunks: 2,
                },
                [2.0, 64.0],
            ),
            (
                GdnChunkLocalCumsumKernelInput {
                    num_tokens: 128,
                    num_chunks: 128,
                },
                [128.0, 1.0],
            ),
            (
                GdnChunkLocalCumsumKernelInput {
                    num_tokens: 384,
                    num_chunks: 48,
                },
                [48.0, 8.0],
            ),
            (
                GdnChunkLocalCumsumKernelInput {
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
    fn sweep_is_exactly_unique_nineteen_by_seven_power_of_two_axes() {
        let cfg = config();
        let grid = GdnChunkLocalCumsumSpec::sweep_grid(&cfg);
        let expected_chunks = [
            1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0,
            8192.0, 16384.0, 32768.0, 65536.0, 131072.0, 262144.0,
        ];
        let expected_density = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];

        assert_eq!(grid.axes().len(), 2);
        assert_eq!(grid.axes()[0], expected_chunks);
        assert_eq!(grid.axes()[1], expected_density);
        for (axis, expected_len, expected_last) in [
            (grid.axes()[0].as_slice(), 19, 262144.0),
            (grid.axes()[1].as_slice(), 7, 64.0),
        ] {
            assert_eq!(axis.len(), expected_len);
            assert_eq!(axis.first(), Some(&1.0));
            assert_eq!(axis.last(), Some(&expected_last));
            assert!(axis.windows(2).all(|pair| pair[0] < pair[1]));
            let unique = axis
                .iter()
                .map(|value| value.to_bits())
                .collect::<BTreeSet<_>>();
            assert_eq!(unique.len(), expected_len);
        }
        assert_eq!(grid.axes()[0].len() * grid.axes()[1].len(), 133);
    }

    #[test]
    fn mask_is_exactly_the_checkpoint_token_cap() {
        let cfg = config();
        let grid = GdnChunkLocalCumsumSpec::sweep_grid(&cfg);
        let mask = GdnChunkLocalCumsumSpec::infeasible_mask(&cfg, &grid);
        let density_axis_len = grid.axes()[1].len();

        assert_eq!(mask.len(), 133);
        assert_eq!(mask.iter().filter(|&&masked| masked).count(), 21);
        assert_eq!(mask.iter().filter(|&&masked| !masked).count(), 112);
        for (chunk_index, &num_chunks) in grid.axes()[0].iter().enumerate() {
            for (density_index, &tokens_per_chunk) in grid.axes()[1].iter().enumerate() {
                let num_chunks = num_chunks as u64;
                let tokens_per_chunk = tokens_per_chunk as u64;
                let expected = num_chunks.checked_mul(tokens_per_chunk).unwrap() > MAX_TOKENS;
                assert_eq!(
                    mask[chunk_index * density_axis_len + density_index],
                    expected,
                    "C={num_chunks} D={tokens_per_chunk}"
                );
            }
        }

        let is_masked = |num_chunks: u64, tokens_per_chunk: u64| {
            num_chunks.checked_mul(tokens_per_chunk).unwrap() > MAX_TOKENS
        };
        for (num_chunks, tokens_per_chunk) in [(262_144, 1), (4_096, 64), (1, 64)] {
            assert!(!is_masked(num_chunks, tokens_per_chunk));
        }
        assert!(is_masked(262_144, 2));
        assert!(is_masked(8_192, 64));
    }

    #[test]
    fn both_backends_keep_linear_2d_cache_and_profile_identity() {
        for backend in ["torch", "vllm_triton"] {
            assert_eq!(
                GdnChunkLocalCumsumSpec::cache_kind(backend),
                CacheKind::Cache2DLinear(Extrapolation::Product)
            );
        }
        assert_eq!(
            GdnChunkLocalCumsumSpec::profile_kind(),
            GdnChunkLocalCumsumSpec::KIND
        );
    }

    #[test]
    fn enumerate_maps_cache_cells_to_exact_physical_python_payloads() {
        let cfg = config();
        let grid = GdnChunkLocalCumsumSpec::sweep_grid(&cfg);
        let payloads = GdnChunkLocalCumsumSpec::enumerate(&cfg, &grid, "vllm_triton");
        let mask = GdnChunkLocalCumsumSpec::infeasible_mask(&cfg, &grid);

        assert_eq!(payloads.len(), 133);
        let expected_names =
            BTreeSet::from(["backend", "dtype", "num_chunks", "num_heads", "num_tokens"]);
        for (index, payload) in payloads.iter().enumerate() {
            let names = payload
                .fields()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            assert_eq!(names, expected_names);
            assert_eq!(payload.fields().len(), 5);
            assert_eq!(payload.backend(), Some("vllm_triton"));
            assert_eq!(payload.fields()["num_heads"], Value::from(32_u32));
            assert_eq!(payload.fields()["dtype"], Value::from("fp32"));

            let num_tokens = payload.fields()["num_tokens"].as_u64().unwrap();
            let num_chunks = payload.fields()["num_chunks"].as_u64().unwrap();
            assert_eq!(num_tokens % num_chunks, 0);
            assert_eq!(mask[index], num_tokens > MAX_TOKENS);
        }

        assert_eq!(payloads[0].fields()["num_tokens"], Value::from(1_u32));
        assert_eq!(payloads[0].fields()["num_chunks"], Value::from(1_u32));
        let qwen_index = 1 * 7 + 6;
        assert_eq!(
            payloads[qwen_index].fields()["num_tokens"],
            Value::from(128_u32)
        );
        assert_eq!(
            payloads[qwen_index].fields()["num_chunks"],
            Value::from(2_u32)
        );
        let minimum_at_cap_index = 12 * 7 + 6;
        assert_eq!(
            payloads[minimum_at_cap_index].fields()["num_tokens"],
            Value::from(262_144_u32)
        );
        assert_eq!(
            payloads[minimum_at_cap_index].fields()["num_chunks"],
            Value::from(4_096_u32)
        );
        assert_eq!(
            payloads[126].fields()["num_tokens"],
            Value::from(262_144_u32)
        );
        assert_eq!(
            payloads[126].fields()["num_chunks"],
            Value::from(262_144_u32)
        );
        assert_eq!(
            payloads[132].fields()["num_tokens"],
            Value::from(16_777_216_u32)
        );
        assert!(mask[132]);
    }

    #[test]
    fn feasible_reaxis_cells_are_a_bijection_with_the_old_accepted_rows() {
        let cfg = config();
        let grid = GdnChunkLocalCumsumSpec::sweep_grid(&cfg);
        let payloads = GdnChunkLocalCumsumSpec::enumerate(&cfg, &grid, "vllm_triton");
        let mask = GdnChunkLocalCumsumSpec::infeasible_mask(&cfg, &grid);

        let new_rows = payloads
            .iter()
            .zip(&mask)
            .filter_map(|(payload, &masked)| {
                (!masked).then(|| {
                    (
                        payload.fields()["num_tokens"].as_u64().unwrap(),
                        payload.fields()["num_chunks"].as_u64().unwrap(),
                    )
                })
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(new_rows.len(), 112);
        assert!(new_rows
            .iter()
            .all(|&(tokens, chunks)| { tokens.div_ceil(64) <= chunks && chunks <= tokens }));

        let old_axis = (0..=18).map(|power| 1_u64 << power).collect::<Vec<_>>();
        let old_rows = old_axis
            .iter()
            .flat_map(|&tokens| {
                old_axis.iter().filter_map(move |&chunks| {
                    (tokens.div_ceil(64) <= chunks && chunks <= tokens).then_some((tokens, chunks))
                })
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(old_rows.len(), 112);
        assert_eq!(new_rows, old_rows);

        let reverse = new_rows
            .iter()
            .map(|&(tokens, chunks)| ((chunks, tokens / chunks), (tokens, chunks)))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(reverse.len(), 112);
        for &(tokens, chunks) in &old_rows {
            assert_eq!(reverse[&(chunks, tokens / chunks)], (tokens, chunks));
        }
    }
}
