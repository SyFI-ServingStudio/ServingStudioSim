//! Qwen Gated `DeltaNet` fused chunk-output kernel.
//!
//! The public input retains physical `(T, C)`, while the cache projects to
//! `(C, D=T/C)`. `C` controls the launch count, and `D` captures the measured
//! valid-row occupancy effect. The explicit `C=6` landmark covers a measured
//! launch-count interpolation cliff.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnChunkOutputKernelConfig {
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
pub struct GdnChunkOutputKernelInput {
    pub num_tokens: u32,
    pub num_chunks: u32,
}

impl SweepCoords for GdnChunkOutputKernelInput {
    fn coords(&self) -> Coords {
        Coords::new([f64::from(self.num_tokens), f64::from(self.num_chunks)])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["num_tokens", "num_chunks"]
    }
}

pub struct GdnChunkOutputSpec;

impl KernelSpec for GdnChunkOutputSpec {
    type Config = GdnChunkOutputKernelConfig;
    type Input = GdnChunkOutputKernelInput;

    const KIND: KernelKind = "gdn_chunk_output";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::chain([Axis::pow2(0, 2), Axis::values([6]), Axis::pow2(3, 6)]),
            Axis::values([1, 64]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn cache_coords(_config: &Self::Config, input: &Self::Input) -> Coords {
        Coords::new([
            f64::from(input.num_chunks),
            f64::from(input.num_tokens) / f64::from(input.num_chunks),
        ])
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
                .expect("GDN chunk-output token product must fit u64");
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

register_kernel!(GdnChunkOutputKernel, GdnChunkOutputSpec);

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{GdnChunkOutputKernelConfig, GdnChunkOutputKernelInput, GdnChunkOutputSpec};
    use crate::timing::bridge::DType;
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> GdnChunkOutputKernelConfig {
        GdnChunkOutputKernelConfig {
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
    fn config_identity_serde_description_routing_and_dtype_tag_are_exact() {
        let cfg = config();
        assert_eq!(GdnChunkOutputSpec::KIND, "gdn_chunk_output");
        assert_eq!(GdnChunkOutputSpec::profile_kind(), "gdn_chunk_output");
        assert_eq!(cfg.backends(), &["torch", "vllm_triton"]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
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
        let decoded: GdnChunkOutputKernelConfig = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
    }

    #[test]
    fn physical_input_names_serde_slot_and_two_dimensional_projection_are_exact() {
        let input = GdnChunkOutputKernelInput {
            num_tokens: 97,
            num_chunks: 2,
        };
        assert_eq!(&*input.coords(), &[97.0, 2.0]);
        assert_eq!(
            GdnChunkOutputKernelInput::coord_field_names(),
            &["num_tokens", "num_chunks"]
        );
        assert_eq!(
            &*GdnChunkOutputSpec::cache_coords(&config(), &input),
            &[2.0, 48.5]
        );
        assert_eq!(
            &*GdnChunkOutputSpec::cache_coords(
                &config(),
                &GdnChunkOutputKernelInput {
                    num_tokens: 65,
                    num_chunks: 2,
                },
            ),
            &[2.0, 32.5]
        );
        assert_eq!(
            &*GdnChunkOutputSpec::cache_coords(
                &config(),
                &GdnChunkOutputKernelInput {
                    num_tokens: 128,
                    num_chunks: 2,
                },
            ),
            &[2.0, 64.0]
        );
        let slot: SlotInput = input.clone().into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_tokens": 97, "num_chunks": 2})
        );
        let decoded: GdnChunkOutputKernelInput =
            serde_json::from_value(serde_json::to_value(input).unwrap()).unwrap();
        assert_eq!(&*decoded.coords(), &[97.0, 2.0]);
    }

    #[test]
    fn grid_cache_mask_and_enumeration_are_exact_and_include_qwen() {
        let cfg = config();
        let grid = GdnChunkOutputSpec::sweep_grid(&cfg);
        assert_eq!(
            grid.axes(),
            &[
                vec![1.0, 2.0, 4.0, 6.0, 8.0, 16.0, 32.0, 64.0],
                vec![1.0, 64.0],
            ]
        );
        assert!(GdnChunkOutputSpec::infeasible_mask(&cfg, &grid).is_empty());
        for backend in ["torch", "vllm_triton"] {
            assert_eq!(
                GdnChunkOutputSpec::cache_kind(backend),
                CacheKind::Cache2DLinear(Extrapolation::Product)
            );
            let payloads = GdnChunkOutputSpec::enumerate(&cfg, &grid, backend);
            assert_eq!(payloads.len(), 16);
            let mut rows = BTreeSet::new();
            for payload in &payloads {
                let fields = payload.fields();
                assert_eq!(
                    fields.keys().map(String::as_str).collect::<BTreeSet<_>>(),
                    BTreeSet::from([
                        "backend",
                        "dtype",
                        "key_head_dim",
                        "num_chunks",
                        "num_heads",
                        "num_key_heads",
                        "num_tokens",
                        "value_head_dim",
                    ])
                );
                let tokens = fields["num_tokens"].as_u64().unwrap();
                let chunks = fields["num_chunks"].as_u64().unwrap();
                assert!(matches!(tokens / chunks, 1 | 64));
                assert_eq!(tokens % chunks, 0);
                assert!((tokens + 63) / 64 <= chunks && chunks <= tokens);
                assert_eq!(fields["backend"], Value::from(backend));
                assert_eq!(fields["num_key_heads"], Value::from(16));
                assert_eq!(fields["num_heads"], Value::from(32));
                assert_eq!(fields["key_head_dim"], Value::from(128));
                assert_eq!(fields["value_head_dim"], Value::from(128));
                assert_eq!(fields["dtype"], Value::from("bf16"));
                rows.insert((tokens, chunks));
            }
            assert_eq!(rows.len(), 16);
            assert!(rows.contains(&(128, 2)));
            let qwen = &payloads[3];
            assert_eq!(qwen.fields()["num_tokens"], Value::from(128));
            assert_eq!(qwen.fields()["num_chunks"], Value::from(2));

            let old_rows = BTreeSet::from([
                (64, 1),
                (128, 2),
                (256, 4),
                (512, 8),
                (1024, 16),
                (2048, 32),
                (4096, 64),
            ]);
            assert_eq!(
                rows.intersection(&old_rows)
                    .copied()
                    .collect::<BTreeSet<_>>(),
                old_rows
            );
            assert_eq!(
                rows.difference(&old_rows).copied().collect::<BTreeSet<_>>(),
                BTreeSet::from([
                    (1, 1),
                    (2, 2),
                    (4, 4),
                    (6, 6),
                    (8, 8),
                    (16, 16),
                    (32, 32),
                    (64, 64),
                    (384, 6),
                ])
            );
        }
    }
}
