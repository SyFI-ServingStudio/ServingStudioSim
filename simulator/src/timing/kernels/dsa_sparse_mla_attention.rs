//! GLM-5.2 DSA sparse MLA attention kernel.
//!
//! The cache uses physical `(num_queries, num_cache_tokens)` coordinates. The
//! static `valid_counts_pattern` deterministically derives Python's canonical
//! flattened valid-slot-count RLE during enumeration; RLE text is never a cache
//! coordinate. Query points bracket sparse-attention row waves, selected-K and
//! page boundaries, and the established 131072-token serving domain.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaSparseMlaAttentionKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub num_kv_heads: Dim,
    pub selected_k: u32,
    pub latent_dim: Dim,
    pub rope_dim: Dim,
    pub value_dim: Dim,
    pub softmax_scale_denominator: u32,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub cache_dtype: DType,
    pub index_dtype: String,
    pub output_dtype: DType,
    pub valid_counts_pattern: String,
    pub index_distribution: String,
    pub cache_layout: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DsaSparseMlaAttentionKernelInput {
    pub num_queries: u32,
    pub num_cache_tokens: u32,
}

pub struct DsaSparseMlaAttentionSpec;

impl KernelSpec for DsaSparseMlaAttentionSpec {
    type Config = DsaSparseMlaAttentionKernelConfig;
    type Input = DsaSparseMlaAttentionKernelInput;

    const KIND: KernelKind = "dsa_sparse_mla_attention";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values([
                1, 2, 4, 8, 16, 32, 64, 127, 128, 129, 131, 132, 133, 255, 256, 257, 263, 264, 265,
                512, 1024, 2048, 4096,
            ]),
            Axis::values([
                1, 2, 4, 8, 16, 32, 63, 64, 65, 127, 128, 129, 256, 512, 1024, 2047, 2048, 2049,
                4096, 8192, 16384, 32768, 65536, 131072,
            ]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        match config.valid_counts_pattern.as_str() {
            "uniform_full" => vec![false; grid.axes()[0].len() * grid.axes()[1].len()],
            "causal_tail" => {
                grid.expand_2d(|num_queries, num_cache_tokens| num_queries > num_cache_tokens)
            }
            "speculative_pairs" => grid.expand_2d(|num_queries, _| num_queries as u32 % 2 != 0),
            pattern => panic!("unsupported valid_counts_pattern {pattern:?}"),
        }
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        assert!(
            config.softmax_scale_denominator > 0,
            "softmax_scale_denominator must be positive"
        );
        grid.expand_2d(|num_queries, num_cache_tokens| {
            let num_queries = num_queries as u32;
            let num_cache_tokens = num_cache_tokens as u32;
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_queries", num_queries)
                .with("num_cache_tokens", num_cache_tokens)
                .with("num_heads", config.num_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("selected_k", config.selected_k)
                .with("latent_dim", config.latent_dim.get())
                .with("rope_dim", config.rope_dim.get())
                .with("value_dim", config.value_dim.get())
                .with(
                    "softmax_scale",
                    1.0 / f64::from(config.softmax_scale_denominator),
                )
                .with("q_dtype", config.q_dtype.as_str())
                .with("cache_dtype", config.cache_dtype.as_str())
                .with("index_dtype", config.index_dtype.clone())
                .with("output_dtype", config.output_dtype.as_str())
                .with(
                    "valid_counts",
                    canonical_valid_counts(
                        &config.valid_counts_pattern,
                        num_queries,
                        num_cache_tokens,
                        config.selected_k,
                    ),
                )
                .with("index_distribution", config.index_distribution.clone())
                .with("cache_layout", config.cache_layout.clone())
        })
    }
}

/// Derive the exact canonical Python valid-count encoding from physical axes.
fn canonical_valid_counts(pattern: &str, q: u32, s: u32, k: u32) -> String {
    match pattern {
        "uniform_full" => format!("u:{}x{q}", s.min(k)),
        "causal_tail" => {
            if q > s {
                return masked_placeholder(q);
            }
            let first = s - q + 1;
            let last = s;
            if q == 1 || first >= k {
                format!("u:{}x{q}", first.min(k))
            } else if last <= k {
                format!("r:{first}..{last}")
            } else {
                debug_assert!(first < k && k < last);
                format!("c:{first}..{last}@{k}")
            }
        }
        "speculative_pairs" => {
            if q % 2 != 0 {
                return masked_placeholder(q);
            }
            let first = s.saturating_sub(1).min(k);
            let second = s.min(k);
            if first == second {
                format!("u:{first}x{q}")
            } else if q == 2 {
                debug_assert_eq!(second, first + 1);
                format!("r:{first}..{second}")
            } else {
                format!("g:({first},{second})x{}", q / 2)
            }
        }
        _ => panic!("unsupported valid_counts_pattern {pattern:?}"),
    }
}

fn masked_placeholder(q: u32) -> String {
    format!("u:0x{q}")
}

register_kernel!(DsaSparseMlaAttentionKernel, DsaSparseMlaAttentionSpec);

#[cfg(test)]
mod tests {
    use super::{
        canonical_valid_counts, DsaSparseMlaAttentionKernelConfig,
        DsaSparseMlaAttentionKernelInput, DsaSparseMlaAttentionSpec,
    };
    use crate::timing::bridge::{ArgsPayload, DType};
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{Dim, SlotInput, SweepCoords};
    use serde_json::Value;

    const TORCH_BACKEND: &str = "torch";
    const FLASHMLA_BACKEND: &str = "vllm_flashmla_bf16";
    const QUERY_AXIS: &[f64] = &[
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 127.0, 128.0, 129.0, 131.0, 132.0, 133.0, 255.0,
        256.0, 257.0, 263.0, 264.0, 265.0, 512.0, 1024.0, 2048.0, 4096.0,
    ];
    const CACHE_AXIS: &[f64] = &[
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 63.0, 64.0, 65.0, 127.0, 128.0, 129.0, 256.0, 512.0,
        1024.0, 2047.0, 2048.0, 2049.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0, 131072.0,
    ];

    fn config(pattern: &str) -> DsaSparseMlaAttentionKernelConfig {
        DsaSparseMlaAttentionKernelConfig {
            backends: vec![TORCH_BACKEND, FLASHMLA_BACKEND],
            gpu_name: "NVIDIA H200".to_string(),
            num_heads: Dim::param("num_attention_heads", 64),
            num_kv_heads: Dim::param("num_kv_heads", 1),
            selected_k: 2048,
            latent_dim: Dim::param("kv_lora_rank", 512),
            rope_dim: Dim::param("qk_rope_head_dim", 64),
            value_dim: Dim::param("kv_lora_rank", 512),
            softmax_scale_denominator: 16,
            q_dtype: DType::Bf16,
            cache_dtype: DType::Bf16,
            index_dtype: "int32".to_string(),
            output_dtype: DType::Bf16,
            valid_counts_pattern: pattern.to_string(),
            index_distribution: "recent_contiguous".to_string(),
            cache_layout: "token_major_mqa_bf16_latent_rope".to_string(),
        }
    }

    #[test]
    fn config_kind_and_dtype_identity_match_the_python_handoff() {
        let cfg = config("uniform_full");

        assert_eq!(DsaSparseMlaAttentionSpec::KIND, "dsa_sparse_mla_attention");
        assert_eq!(
            DsaSparseMlaAttentionSpec::profile_kind(),
            "dsa_sparse_mla_attention"
        );
        assert_eq!(cfg.backends(), &[TORCH_BACKEND, FLASHMLA_BACKEND]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.num_heads, 64);
        assert_eq!(cfg.num_kv_heads, 1);
        assert_eq!(cfg.selected_k, 2048);
        assert_eq!(cfg.latent_dim, 512);
        assert_eq!(cfg.rope_dim, 64);
        assert_eq!(cfg.value_dim, 512);
        assert_eq!(cfg.softmax_scale_denominator, 16);
        assert_eq!(cfg.q_dtype, DType::Bf16);
        assert_eq!(cfg.cache_dtype, DType::Bf16);
        assert_eq!(cfg.output_dtype, DType::Bf16);
        assert_eq!(cfg.index_dtype, "int32");
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), Some(DType::Bf16));
    }

    #[test]
    fn describe_config_preserves_rich_dimensions_and_static_modes() {
        assert_eq!(
            config("causal_tail").describe_config(),
            serde_json::json!({
                "backends": [TORCH_BACKEND, FLASHMLA_BACKEND],
                "gpu_name": "NVIDIA H200",
                "num_heads": rich_dim("num_attention_heads", 64),
                "num_kv_heads": rich_dim("num_kv_heads", 1),
                "selected_k": 2048,
                "latent_dim": rich_dim("kv_lora_rank", 512),
                "rope_dim": rich_dim("qk_rope_head_dim", 64),
                "value_dim": rich_dim("kv_lora_rank", 512),
                "softmax_scale_denominator": 16,
                "q_dtype": "bf16",
                "cache_dtype": "bf16",
                "index_dtype": "int32",
                "output_dtype": "bf16",
                "valid_counts_pattern": "causal_tail",
                "index_distribution": "recent_contiguous",
                "cache_layout": "token_major_mqa_bf16_latent_rope",
            })
        );
    }

    #[test]
    fn input_coordinates_deserialization_and_slot_input_are_physical() {
        let input: DsaSparseMlaAttentionKernelInput =
            serde_json::from_str(r#"{"num_queries":128,"num_cache_tokens":2049}"#).unwrap();

        assert_eq!(&*input.coords(), &[128.0, 2049.0]);
        assert_eq!(
            DsaSparseMlaAttentionKernelInput::coord_field_names(),
            &["num_queries", "num_cache_tokens"]
        );
        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_queries":128,"num_cache_tokens":2049})
        );
    }

    #[test]
    fn grid_has_the_exact_frozen_axes_and_boundaries() {
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&config("uniform_full"));
        let axes = grid.axes();

        assert_eq!(axes.len(), 2);
        assert_eq!(axes[0], QUERY_AXIS);
        assert_eq!(axes[1], CACHE_AXIS);
        assert_eq!(axes[0].len(), 23);
        assert_eq!(axes[1].len(), 24);
        assert!(axes[0].windows(2).all(|pair| pair[0] < pair[1]));
        assert!(axes[1].windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(&axes[0][7..10], &[127.0, 128.0, 129.0]);
        assert_eq!(&axes[0][10..13], &[131.0, 132.0, 133.0]);
        assert_eq!(
            &axes[0][13..19],
            &[255.0, 256.0, 257.0, 263.0, 264.0, 265.0]
        );
        assert_eq!(&axes[1][6..9], &[63.0, 64.0, 65.0]);
        assert_eq!(&axes[1][9..12], &[127.0, 128.0, 129.0]);
        assert_eq!(&axes[1][15..18], &[2047.0, 2048.0, 2049.0]);
        assert_eq!(axes[0].first(), Some(&1.0));
        assert_eq!(axes[0].last(), Some(&4096.0));
        assert_eq!(axes[1].first(), Some(&1.0));
        assert_eq!(axes[1].last(), Some(&131072.0));
        assert_eq!(axes[0].len() * axes[1].len(), 552);
    }

    #[test]
    fn masks_match_each_frozen_pattern_domain() {
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&config("uniform_full"));

        let uniform = mask_for("uniform_full", &grid);
        assert_mask_split(&uniform, 552, 0);

        let causal = mask_for("causal_tail", &grid);
        assert_mask_split(&causal, 327, 225);
        assert!(!masked(&causal, &grid, 1, 1));
        assert!(!masked(&causal, &grid, 128, 128));
        assert!(!masked(&causal, &grid, 128, 2049));
        assert!(masked(&causal, &grid, 129, 128));
        assert!(masked(&causal, &grid, 4096, 2049));

        let speculative = mask_for("speculative_pairs", &grid);
        assert_mask_split(&speculative, 336, 216);
        assert!(masked(&speculative, &grid, 1, 1));
        assert!(!masked(&speculative, &grid, 2, 1));
        assert!(!masked(&speculative, &grid, 32, 2048));
        assert!(masked(&speculative, &grid, 127, 2048));
        assert!(!masked(&speculative, &grid, 4096, 131072));
    }

    #[test]
    fn canonical_valid_counts_obeys_python_precedence_for_every_pattern() {
        assert_eq!(canonical_valid_counts("uniform_full", 1, 1, 2048), "u:1x1");
        assert_eq!(
            canonical_valid_counts("uniform_full", 256, 131072, 2048),
            "u:2048x256"
        );

        assert_eq!(canonical_valid_counts("causal_tail", 1, 1, 2048), "u:1x1");
        assert_eq!(
            canonical_valid_counts("causal_tail", 128, 128, 2048),
            "r:1..128"
        );
        assert_eq!(
            canonical_valid_counts("causal_tail", 128, 2048, 2048),
            "r:1921..2048"
        );
        assert_eq!(
            canonical_valid_counts("causal_tail", 128, 2049, 2048),
            "c:1922..2049@2048"
        );
        assert_eq!(
            canonical_valid_counts("causal_tail", 128, 4096, 2048),
            "u:2048x128"
        );
        assert_eq!(
            canonical_valid_counts("causal_tail", 4096, 1, 2048),
            "u:0x4096"
        );

        assert_eq!(
            canonical_valid_counts("speculative_pairs", 2, 1, 2048),
            "r:0..1"
        );
        assert_eq!(
            canonical_valid_counts("speculative_pairs", 32, 2048, 2048),
            "g:(2047,2048)x16"
        );
        assert_eq!(
            canonical_valid_counts("speculative_pairs", 2, 2049, 2048),
            "u:2048x2"
        );
        assert_eq!(
            canonical_valid_counts("speculative_pairs", 127, 2048, 2048),
            "u:0x127"
        );
    }

    #[test]
    #[should_panic(expected = "unsupported valid_counts_pattern")]
    fn unknown_pattern_fails_clearly() {
        let _ = canonical_valid_counts("requests", 1, 1, 2048);
    }

    #[test]
    fn both_backends_use_the_existing_physical_bilinear_cache() {
        assert_eq!(
            DsaSparseMlaAttentionSpec::cache_kind(TORCH_BACKEND),
            CacheKind::Cache2DLinear
        );
        assert_eq!(
            DsaSparseMlaAttentionSpec::cache_kind(FLASHMLA_BACKEND),
            CacheKind::Cache2DLinear
        );
    }

    #[test]
    fn enumeration_emits_backend_plus_the_exact_python_schema() {
        for pattern in ["uniform_full", "causal_tail", "speculative_pairs"] {
            let cfg = config(pattern);
            let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
            let payloads = DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, FLASHMLA_BACKEND);

            assert_eq!(payloads.len(), 552);
            let expected_names = [
                "backend",
                "cache_dtype",
                "cache_layout",
                "index_distribution",
                "index_dtype",
                "latent_dim",
                "num_cache_tokens",
                "num_heads",
                "num_kv_heads",
                "num_queries",
                "output_dtype",
                "q_dtype",
                "rope_dim",
                "selected_k",
                "softmax_scale",
                "valid_counts",
                "value_dim",
            ];
            for payload in &payloads {
                let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
                assert_eq!(names, expected_names);
                assert_eq!(payload.fields().len(), 17);
            }

            assert_payload(
                &payloads[0],
                1,
                1,
                canonical_valid_counts(pattern, 1, 1, 2048),
            );
            assert_payload(
                payload_for(&payloads, &grid, 32, 2048),
                32,
                2048,
                canonical_valid_counts(pattern, 32, 2048, 2048),
            );
            assert_payload(
                payload_for(&payloads, &grid, 128, 2049),
                128,
                2049,
                canonical_valid_counts(pattern, 128, 2049, 2048),
            );
            for q in [131, 132, 133] {
                assert_payload(
                    payload_for(&payloads, &grid, q, 8192),
                    q,
                    8192,
                    canonical_valid_counts(pattern, q, 8192, 2048),
                );
            }
            assert_payload(
                payloads.last().unwrap(),
                4096,
                131072,
                canonical_valid_counts(pattern, 4096, 131072, 2048),
            );
        }
    }

    fn rich_dim(name: &str, value: u32) -> Value {
        serde_json::json!({
            "value": value,
            "expression": name,
            "bindings": {name: value},
        })
    }

    fn mask_for(pattern: &str, grid: &crate::timing::sweep::SweepGrid) -> Vec<bool> {
        DsaSparseMlaAttentionSpec::infeasible_mask(&config(pattern), grid)
    }

    fn assert_mask_split(mask: &[bool], feasible: usize, masked: usize) {
        assert_eq!(mask.len(), 552);
        assert_eq!(mask.iter().filter(|&&value| !value).count(), feasible);
        assert_eq!(mask.iter().filter(|&&value| value).count(), masked);
    }

    fn masked(mask: &[bool], grid: &crate::timing::sweep::SweepGrid, q: u32, s: u32) -> bool {
        let q_index = grid.axes()[0]
            .iter()
            .position(|&value| value == f64::from(q))
            .unwrap();
        let s_index = grid.axes()[1]
            .iter()
            .position(|&value| value == f64::from(s))
            .unwrap();
        mask[q_index * grid.axes()[1].len() + s_index]
    }

    fn payload_for<'a>(
        payloads: &'a [ArgsPayload],
        grid: &crate::timing::sweep::SweepGrid,
        q: u32,
        s: u32,
    ) -> &'a ArgsPayload {
        let q_index = grid.axes()[0]
            .iter()
            .position(|&value| value == f64::from(q))
            .unwrap();
        let s_index = grid.axes()[1]
            .iter()
            .position(|&value| value == f64::from(s))
            .unwrap();
        &payloads[q_index * grid.axes()[1].len() + s_index]
    }

    fn assert_payload(payload: &ArgsPayload, q: u32, s: u32, valid_counts: String) {
        let fields = payload.fields();
        assert_eq!(fields.get("backend"), Some(&Value::from(FLASHMLA_BACKEND)));
        assert_eq!(fields.get("num_queries"), Some(&Value::from(q)));
        assert_eq!(fields.get("num_cache_tokens"), Some(&Value::from(s)));
        assert_eq!(fields.get("num_heads"), Some(&Value::from(64_u32)));
        assert_eq!(fields.get("num_kv_heads"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("selected_k"), Some(&Value::from(2048_u32)));
        assert_eq!(fields.get("latent_dim"), Some(&Value::from(512_u32)));
        assert_eq!(fields.get("rope_dim"), Some(&Value::from(64_u32)));
        assert_eq!(fields.get("value_dim"), Some(&Value::from(512_u32)));
        assert_eq!(fields.get("softmax_scale"), Some(&Value::from(0.0625)));
        assert_eq!(fields.get("q_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("cache_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("index_dtype"), Some(&Value::from("int32")));
        assert_eq!(fields.get("output_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("valid_counts"), Some(&Value::from(valid_counts)));
        assert_eq!(
            fields.get("index_distribution"),
            Some(&Value::from("recent_contiguous"))
        );
        assert_eq!(
            fields.get("cache_layout"),
            Some(&Value::from("token_major_mqa_bf16_latent_rope"))
        );
        assert_eq!(payload.backend(), Some(FLASHMLA_BACKEND));
    }
}
