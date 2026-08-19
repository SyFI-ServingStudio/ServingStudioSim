//! Qwen MoE token-to-expert block-alignment kernel.
//!
//! Routed-token count is the sole runtime interpolation axis. At the fixed
//! E256 specialization the first launch has two blocks of fixed work, while
//! the second launch scales as `ceil(T * 8 / 256) = ceil(T / 32)`. Routing
//! popularity and the padded assignment count are deliberately absent from the
//! cache key because matched-T H200 measurements found them timing-equivalent.

use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeAlignBlockSizeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_experts: Dim,
    pub top_k: u32,
    pub block_size: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MoeAlignBlockSizeKernelInput {
    pub num_tokens: u32,
}

pub struct MoeAlignBlockSizeSpec;

impl KernelSpec for MoeAlignBlockSizeSpec {
    type Config = MoeAlignBlockSizeKernelConfig;
    type Input = MoeAlignBlockSizeKernelInput;

    const KIND: KernelKind = "moe_align_block_size";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // T<9 bypasses alignment in vLLM's real expert-assignment path. T31 and
        // T33 resolve the measured launch-count cliff around the T32 boundary;
        // the remaining landmarks follow powers of two through the context cap.
        SweepGrid::new(vec![Axis::values([
            9, 16, 31, 32, 33, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
            131072, 262144,
        ])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
                .with("num_experts", config.num_experts.get())
                .with("top_k", config.top_k)
                .with("block_size", config.block_size)
        })
    }
}

register_kernel!(MoeAlignBlockSizeKernel, MoeAlignBlockSizeSpec);

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{
        MoeAlignBlockSizeKernelConfig, MoeAlignBlockSizeKernelInput, MoeAlignBlockSizeSpec,
    };
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> MoeAlignBlockSizeKernelConfig {
        MoeAlignBlockSizeKernelConfig {
            backends: vec!["torch", "vllm_cuda"],
            gpu_name: "NVIDIA H200".to_string(),
            num_experts: 256.into(),
            top_k: 8,
            block_size: 16,
        }
    }

    #[test]
    fn config_identity_serde_description_routing_and_no_dtype_tags_are_exact() {
        let cfg = config();
        assert_eq!(MoeAlignBlockSizeSpec::KIND, "moe_align_block_size");
        assert_eq!(
            MoeAlignBlockSizeSpec::profile_kind(),
            "moe_align_block_size"
        );
        assert_eq!(cfg.backends(), &["torch", "vllm_cuda"]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.compute_dtype(), None);
        assert_eq!(cfg.kv_dtype(), None);
        assert_eq!(
            cfg.describe_config(),
            serde_json::json!({
                "backends": ["torch", "vllm_cuda"],
                "gpu_name": "NVIDIA H200",
                "num_experts": {"value": 256, "expression": null, "bindings": {}},
                "top_k": 8,
                "block_size": 16,
            })
        );

        let encoded = serde_json::to_value(&cfg).unwrap();
        let decoded: MoeAlignBlockSizeKernelConfig =
            serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);

        let mut changed = config();
        changed.num_experts = 128.into();
        assert_ne!(changed, cfg);
        let mut changed = config();
        changed.top_k = 4;
        assert_ne!(changed, cfg);
        let mut changed = config();
        changed.block_size = 8;
        assert_ne!(changed, cfg);
    }

    #[test]
    fn input_names_coords_serde_and_slot_payload_are_exact_num_tokens() {
        let input = MoeAlignBlockSizeKernelInput { num_tokens: 128 };
        assert_eq!(&*input.coords(), &[128.0]);
        assert_eq!(
            MoeAlignBlockSizeKernelInput::coord_field_names(),
            &["num_tokens"]
        );
        let encoded = serde_json::to_value(&input).unwrap();
        assert_eq!(encoded, serde_json::json!({"num_tokens": 128}));
        let decoded: MoeAlignBlockSizeKernelInput = serde_json::from_value(encoded).unwrap();
        assert_eq!(&*decoded.coords(), &[128.0]);

        let slot: SlotInput = input.into();
        assert!(matches!(slot, SlotInput::MoeAlignBlockSize(_)));
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_tokens": 128})
        );
    }

    #[test]
    fn grid_is_exactly_eighteen_unmasked_unique_rows_with_only_measured_delta() {
        let cfg = config();
        let grid = MoeAlignBlockSizeSpec::sweep_grid(&cfg);
        let expected = vec![
            9.0, 16.0, 31.0, 32.0, 33.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0,
            16384.0, 32768.0, 65536.0, 131072.0, 262144.0,
        ];
        assert_eq!(grid.axes(), &[expected]);
        assert!(MoeAlignBlockSizeSpec::infeasible_mask(&cfg, &grid).is_empty());

        let payloads = MoeAlignBlockSizeSpec::enumerate(&cfg, &grid, "vllm_cuda");
        let tokens: HashSet<u32> = payloads
            .iter()
            .map(|payload| payload.fields()["num_tokens"].as_u64().unwrap() as u32)
            .collect();
        assert_eq!(payloads.len(), 18);
        assert_eq!(tokens.len(), 18);
        assert!(tokens.iter().all(|&tokens| (9..=262144).contains(&tokens)));

        let old: HashSet<u32> = [
            9, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
            262144,
        ]
        .into_iter()
        .collect();
        assert!(old.is_subset(&tokens));
        assert_eq!(tokens.intersection(&old).count(), 16);
        assert_eq!(
            tokens.difference(&old).copied().collect::<HashSet<_>>(),
            HashSet::from([31, 33])
        );
    }

    #[test]
    fn both_backends_use_linear_1d_cache() {
        for backend in ["torch", "vllm_cuda"] {
            assert_eq!(
                MoeAlignBlockSizeSpec::cache_kind(backend),
                CacheKind::Cache1DLinear
            );
        }
    }

    #[test]
    fn payload_fields_static_identity_domain_and_qwen_anchor_are_exact() {
        let cfg = config();
        let grid = MoeAlignBlockSizeSpec::sweep_grid(&cfg);

        for backend in ["torch", "vllm_cuda"] {
            let payloads = MoeAlignBlockSizeSpec::enumerate(&cfg, &grid, backend);
            for payload in &payloads {
                let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
                assert_eq!(
                    names,
                    [
                        "backend",
                        "block_size",
                        "num_experts",
                        "num_tokens",
                        "top_k"
                    ]
                );
                assert_eq!(payload.fields().len(), 5);
                assert_eq!(payload.backend(), Some(backend));
                assert!(payload.fields()["num_tokens"].as_u64().unwrap() >= 9);
                assert_eq!(payload.fields()["num_experts"], Value::from(256_u32));
                assert_eq!(payload.fields()["top_k"], Value::from(8_u32));
                assert_eq!(payload.fields()["block_size"], Value::from(16_u32));
            }
        }

        let qwen = MoeAlignBlockSizeSpec::enumerate(&cfg, &grid, "vllm_cuda")
            .into_iter()
            .find(|payload| payload.fields()["num_tokens"] == Value::from(128_u32))
            .unwrap();
        assert_eq!(
            qwen.fields(),
            &serde_json::from_value(serde_json::json!({
                "backend": "vllm_cuda",
                "num_tokens": 128,
                "num_experts": 256,
                "top_k": 8,
                "block_size": 16,
            }))
            .unwrap()
        );
        let qwen_input = MoeAlignBlockSizeKernelInput { num_tokens: 128 };
        assert_eq!(&*qwen_input.coords(), &[128.0]);
    }
}
