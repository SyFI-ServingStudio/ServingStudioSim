//! Qwen Gated `DeltaNet` chunk-local WY recomputation kernel.
//!
//! Public inputs remain the physical `(num_tokens, num_chunks)` caller shape,
//! while the cache projects them to `(C, D=T/C)`: launched chunks and average
//! valid rows per chunk. These coordinates capture the `(C,H)` launch grid and
//! per-chunk matrix occupancy while rectangularizing the physical feasibility
//! bounds `C=T` and `C=ceil(T/64)`. The sparse power-of-two landmarks are the
//! initial grid; the cache-fidelity gate decides whether later breakpoints are
//! justified. Sequence distribution and maximum occupancy are audited invariants,
//! not public axes.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

const MAX_TOKENS: u64 = 262_144;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnChunkRecomputeWUKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_key_heads: Dim,
    pub num_heads: Dim,
    pub key_head_dim: Dim,
    pub value_head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct GdnChunkRecomputeWUKernelInput {
    pub num_tokens: u32,
    pub num_chunks: u32,
}

impl SweepCoords for GdnChunkRecomputeWUKernelInput {
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

pub struct GdnChunkRecomputeWUSpec;

impl KernelSpec for GdnChunkRecomputeWUSpec {
    type Config = GdnChunkRecomputeWUKernelConfig;
    type Input = GdnChunkRecomputeWUKernelInput;

    const KIND: KernelKind = "gdn_chunk_recompute_w_u";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::pow2(0, 18), Axis::pow2(0, 6)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_2d(|num_chunks, tokens_per_chunk| {
            let num_chunks = num_chunks.round() as u64;
            let tokens_per_chunk = tokens_per_chunk.round() as u64;
            num_chunks
                .checked_mul(tokens_per_chunk)
                .expect("GDN WY re-axis product must fit u64")
                > MAX_TOKENS
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|num_chunks, tokens_per_chunk| {
            let num_chunks = num_chunks.round() as u64;
            let tokens_per_chunk = tokens_per_chunk.round() as u64;
            let num_tokens = num_chunks
                .checked_mul(tokens_per_chunk)
                .expect("GDN WY re-axis product must fit u64");
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
                .with("value_head_dim", config.value_head_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(GdnChunkRecomputeWUKernel, GdnChunkRecomputeWUSpec);

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashSet};

    use super::{
        GdnChunkRecomputeWUKernelConfig, GdnChunkRecomputeWUKernelInput, GdnChunkRecomputeWUSpec,
        MAX_TOKENS,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> GdnChunkRecomputeWUKernelConfig {
        GdnChunkRecomputeWUKernelConfig {
            backends: vec!["torch", "vllm_triton"],
            gpu_name: "NVIDIA H200".to_string(),
            num_key_heads: 16.into(),
            num_heads: 32.into(),
            key_head_dim: 128.into(),
            value_head_dim: 128.into(),
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_description_serde_routing_and_dtype_tags_match_handoff() {
        let cfg = config();

        assert_eq!(GdnChunkRecomputeWUSpec::KIND, "gdn_chunk_recompute_w_u");
        assert_eq!(
            GdnChunkRecomputeWUSpec::profile_kind(),
            "gdn_chunk_recompute_w_u"
        );
        assert_eq!(cfg.backends(), &["torch", "vllm_triton"]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.num_key_heads, 16);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.key_head_dim, 128);
        assert_eq!(cfg.value_head_dim, 128);
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
                "value_head_dim": {"value": 128, "expression": null, "bindings": {}},
                "dtype": "bf16",
            })
        );

        let encoded = serde_json::to_value(&cfg).unwrap();
        let decoded: GdnChunkRecomputeWUKernelConfig =
            serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
    }

    #[test]
    fn physical_input_serde_names_slot_payload_and_cache_coords_are_exact() {
        let qwen = GdnChunkRecomputeWUKernelInput {
            num_tokens: 128,
            num_chunks: 2,
        };
        assert_eq!(&*qwen.coords(), &[2.0, 64.0]);
        assert_eq!(
            GdnChunkRecomputeWUKernelInput::coord_field_names(),
            &["num_tokens", "num_chunks"]
        );
        let slot: SlotInput = qwen.clone().into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_tokens": 128, "num_chunks": 2})
        );
        let round_trip: GdnChunkRecomputeWUKernelInput =
            serde_json::from_value(serde_json::to_value(qwen).unwrap()).unwrap();
        assert_eq!(&*round_trip.coords(), &[2.0, 64.0]);

        for (input, expected) in [
            (
                GdnChunkRecomputeWUKernelInput {
                    num_tokens: 128,
                    num_chunks: 128,
                },
                [128.0, 1.0],
            ),
            (
                GdnChunkRecomputeWUKernelInput {
                    num_tokens: 384,
                    num_chunks: 48,
                },
                [48.0, 8.0],
            ),
            (
                GdnChunkRecomputeWUKernelInput {
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
    fn sweep_is_exactly_nineteen_by_seven_ordered_power_axes() {
        let grid = GdnChunkRecomputeWUSpec::sweep_grid(&config());
        let expected_chunks = [
            1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0,
            8192.0, 16384.0, 32768.0, 65536.0, 131072.0, 262144.0,
        ];
        let expected_density = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];

        assert_eq!(
            grid.axes(),
            &[expected_chunks.to_vec(), expected_density.to_vec()]
        );
        assert_eq!(grid.axes()[0].len() * grid.axes()[1].len(), 133);
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
    fn checked_mask_has_exact_counts_and_context_boundaries() {
        let grid = GdnChunkRecomputeWUSpec::sweep_grid(&config());
        let mask = GdnChunkRecomputeWUSpec::infeasible_mask(&config(), &grid);
        let density_count = grid.axes()[1].len();

        assert_eq!(mask.len(), 133);
        assert_eq!(mask.iter().filter(|&&masked| masked).count(), 21);
        assert_eq!(mask.iter().filter(|&&masked| !masked).count(), 112);
        for (chunk_index, &chunks) in grid.axes()[0].iter().enumerate() {
            for (density_index, &density) in grid.axes()[1].iter().enumerate() {
                let product = (chunks as u64).checked_mul(density as u64).unwrap();
                assert_eq!(
                    mask[chunk_index * density_count + density_index],
                    product > MAX_TOKENS,
                    "C={chunks} D={density}"
                );
            }
        }
        for (chunks, density) in [(262_144_u64, 1_u64), (4_096, 64), (1, 64)] {
            assert!(chunks.checked_mul(density).unwrap() <= MAX_TOKENS);
        }
        assert!(262_144_u64.checked_mul(2).unwrap() > MAX_TOKENS);
        assert!(8_192_u64.checked_mul(64).unwrap() > MAX_TOKENS);
    }

    #[test]
    fn enumeration_has_exact_schema_static_config_and_qwen_anchor() {
        let cfg = config();
        let grid = GdnChunkRecomputeWUSpec::sweep_grid(&cfg);
        let payloads = GdnChunkRecomputeWUSpec::enumerate(&cfg, &grid, "vllm_triton");
        let mask = GdnChunkRecomputeWUSpec::infeasible_mask(&cfg, &grid);
        let expected_names = BTreeSet::from([
            "backend",
            "dtype",
            "key_head_dim",
            "num_chunks",
            "num_heads",
            "num_key_heads",
            "num_tokens",
            "value_head_dim",
        ]);

        assert_eq!(payloads.len(), 133);
        let mut feasible_rows = BTreeSet::new();
        for (index, payload) in payloads.iter().enumerate() {
            let names = payload
                .fields()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            assert_eq!(names, expected_names);
            assert_eq!(payload.fields().len(), 8);
            assert_eq!(payload.backend(), Some("vllm_triton"));
            assert_eq!(payload.fields()["num_key_heads"], Value::from(16_u32));
            assert_eq!(payload.fields()["num_heads"], Value::from(32_u32));
            assert_eq!(payload.fields()["key_head_dim"], Value::from(128_u32));
            assert_eq!(payload.fields()["value_head_dim"], Value::from(128_u32));
            assert_eq!(payload.fields()["dtype"], Value::from("bf16"));

            let tokens = payload.fields()["num_tokens"].as_u64().unwrap();
            let chunks = payload.fields()["num_chunks"].as_u64().unwrap();
            assert_eq!(tokens % chunks, 0);
            assert_eq!(mask[index], tokens > MAX_TOKENS);
            if !mask[index] {
                assert!(tokens.div_ceil(64) <= chunks && chunks <= tokens);
                assert!(tokens <= MAX_TOKENS);
                assert!(feasible_rows.insert((tokens, chunks)));
            }
        }
        assert_eq!(feasible_rows.len(), 112);

        let qwen = &payloads[1 * 7 + 6];
        assert_eq!(qwen.fields()["num_tokens"], Value::from(128_u32));
        assert_eq!(qwen.fields()["num_chunks"], Value::from(2_u32));
        assert_eq!(qwen.fields()["num_key_heads"], Value::from(16_u32));
        assert_eq!(qwen.fields()["num_heads"], Value::from(32_u32));
        assert_eq!(qwen.fields()["key_head_dim"], Value::from(128_u32));
        assert_eq!(qwen.fields()["value_head_dim"], Value::from(128_u32));
        assert_eq!(qwen.fields()["dtype"], Value::from("bf16"));
    }

    #[test]
    fn both_backends_keep_static_values_linear_2d_and_physical_fields() {
        let cfg = config();
        let grid = GdnChunkRecomputeWUSpec::sweep_grid(&cfg);
        for backend in ["torch", "vllm_triton"] {
            assert_eq!(
                GdnChunkRecomputeWUSpec::cache_kind(backend),
                CacheKind::Cache2DLinear(Extrapolation::Product)
            );
            let first = &GdnChunkRecomputeWUSpec::enumerate(&cfg, &grid, backend)[0];
            assert_eq!(first.backend(), Some(backend));
            assert_eq!(first.fields()["num_key_heads"], Value::from(16_u32));
            assert_eq!(first.fields()["num_heads"], Value::from(32_u32));
            assert_eq!(first.fields()["key_head_dim"], Value::from(128_u32));
            assert_eq!(first.fields()["value_head_dim"], Value::from(128_u32));
            assert_eq!(first.fields()["dtype"], Value::from("bf16"));
        }
        assert_eq!(
            GdnChunkRecomputeWUSpec::profile_kind(),
            GdnChunkRecomputeWUSpec::KIND
        );
        assert_eq!(
            GdnChunkRecomputeWUKernelInput::coord_field_names(),
            &["num_tokens", "num_chunks"]
        );
    }
}
