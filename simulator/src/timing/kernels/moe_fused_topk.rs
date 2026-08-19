//! Qwen fused `MoE` softmax/top-k router-selection kernel.
//!
//! The number of routed tokens is the sole runtime interpolation axis. Expert
//! count, top-k, and activation dtype identify the fixed production CUDA
//! specialization; the initial Qwen contract is E256/K8/BF16 on H200.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeFusedTopkKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_experts: Dim,
    pub top_k: u32,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MoeFusedTopkKernelInput {
    pub num_tokens: u32,
}

pub struct MoeFusedTopkSpec;

impl KernelSpec for MoeFusedTopkSpec {
    type Config = MoeFusedTopkKernelConfig;
    type Input = MoeFusedTopkKernelInput;

    const KIND: KernelKind = "moe_fused_topk";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // Power landmarks through the 262,144-token context cap, plus the two
        // measured large-token interpolation breakpoints required by fidelity.
        SweepGrid::new(vec![Axis::values([
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
            98304, 131072, 196608, 262144,
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
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(MoeFusedTopkKernel, MoeFusedTopkSpec);

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{MoeFusedTopkKernelConfig, MoeFusedTopkKernelInput, MoeFusedTopkSpec};
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> MoeFusedTopkKernelConfig {
        MoeFusedTopkKernelConfig {
            backends: vec!["torch", "vllm_cuda"],
            gpu_name: "NVIDIA H200".to_string(),
            num_experts: 256.into(),
            top_k: 8,
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_description_serde_routing_and_dtype_are_exact() {
        let cfg = config();
        assert_eq!(MoeFusedTopkSpec::KIND, "moe_fused_topk");
        assert_eq!(MoeFusedTopkSpec::profile_kind(), "moe_fused_topk");
        assert_eq!(cfg.backends(), &["torch", "vllm_cuda"]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
        assert_eq!(
            cfg.describe_config(),
            serde_json::json!({
                "backends": ["torch", "vllm_cuda"],
                "gpu_name": "NVIDIA H200",
                "num_experts": {"value": 256, "expression": null, "bindings": {}},
                "top_k": 8,
                "dtype": "bf16",
            })
        );

        let encoded = serde_json::to_value(&cfg).unwrap();
        let decoded: MoeFusedTopkKernelConfig = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);

        let mut changed = config();
        changed.num_experts = 128.into();
        assert_ne!(changed, cfg);
        let mut changed = config();
        changed.top_k = 4;
        assert_ne!(changed, cfg);
        let mut changed = config();
        changed.dtype = DType::Fp16;
        assert_ne!(changed, cfg);
    }

    #[test]
    fn input_field_coords_serde_and_slot_payload_are_physical_num_tokens() {
        let input = MoeFusedTopkKernelInput { num_tokens: 128 };
        assert_eq!(&*input.coords(), &[128.0]);
        assert_eq!(
            MoeFusedTopkKernelInput::coord_field_names(),
            &["num_tokens"]
        );

        let encoded = serde_json::to_value(&input).unwrap();
        assert_eq!(encoded, serde_json::json!({"num_tokens": 128}));
        let decoded: MoeFusedTopkKernelInput = serde_json::from_value(encoded).unwrap();
        assert_eq!(&*decoded.coords(), &[128.0]);

        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_tokens": 128})
        );
    }

    #[test]
    fn grid_is_exactly_twenty_one_unique_rows_with_only_two_new_breakpoints() {
        let cfg = config();
        let grid = MoeFusedTopkSpec::sweep_grid(&cfg);
        let expected = vec![
            1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0,
            8192.0, 16384.0, 32768.0, 65536.0, 98304.0, 131072.0, 196608.0, 262144.0,
        ];
        assert_eq!(grid.axes(), &[expected]);
        assert_eq!(grid.axes()[0].len(), 21);
        assert!(MoeFusedTopkSpec::infeasible_mask(&cfg, &grid).is_empty());

        let payloads = MoeFusedTopkSpec::enumerate(&cfg, &grid, "vllm_cuda");
        let tokens: HashSet<u32> = payloads
            .iter()
            .map(|payload| payload.fields()["num_tokens"].as_u64().unwrap() as u32)
            .collect();
        assert_eq!(payloads.len(), 21);
        assert_eq!(tokens.len(), 21);

        let former: HashSet<u32> = (0..=18).map(|log2| 1_u32 << log2).collect();
        assert!(former.is_subset(&tokens));
        let delta: HashSet<u32> = tokens.difference(&former).copied().collect();
        assert_eq!(delta, HashSet::from([98304, 196608]));
    }

    #[test]
    fn both_backends_use_linear_1d_cache() {
        for backend in ["torch", "vllm_cuda"] {
            assert_eq!(
                MoeFusedTopkSpec::cache_kind(backend),
                CacheKind::Cache1DLinear
            );
        }
    }

    #[test]
    fn payload_fields_static_identity_domain_and_qwen_anchor_are_exact() {
        let cfg = config();
        let grid = MoeFusedTopkSpec::sweep_grid(&cfg);

        for backend in ["torch", "vllm_cuda"] {
            let payloads = MoeFusedTopkSpec::enumerate(&cfg, &grid, backend);
            for payload in &payloads {
                let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
                assert_eq!(
                    names,
                    ["backend", "dtype", "num_experts", "num_tokens", "top_k"]
                );
                assert_eq!(payload.fields().len(), 5);
                assert_eq!(payload.backend(), Some(backend));
                assert!(payload.fields()["num_tokens"].as_u64().unwrap() > 0);
                assert_eq!(payload.fields()["num_experts"], Value::from(256_u32));
                assert_eq!(payload.fields()["top_k"], Value::from(8_u32));
                assert_eq!(payload.fields()["dtype"], Value::from("bf16"));
            }
        }

        let qwen = MoeFusedTopkSpec::enumerate(&cfg, &grid, "vllm_cuda")
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
                "dtype": "bf16",
            }))
            .unwrap()
        );
        let input = MoeFusedTopkKernelInput { num_tokens: 128 };
        assert_eq!(&*input.coords(), &[128.0]);
    }
}
