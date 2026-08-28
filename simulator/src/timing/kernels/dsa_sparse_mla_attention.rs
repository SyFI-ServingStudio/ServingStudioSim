//! GLM-5.2 DSA sparse MLA attention kernel.
//!
//! The cache uses physical `(num_queries, num_cache_tokens)` coordinates. The
//! static `valid_counts_pattern` deterministically derives Python's canonical
//! flattened valid-slot-count RLE during enumeration; RLE text is never a cache
//! coordinate. The first R.4 gate found pattern-specific interpolation defects:
//! uniform adds S6/S11/S91, causal pins its domain boundary and scattered-cache
//! misses, and speculative removes unsupported odd-Q grid holes while adding
//! Q134/Q266 and S3/S23/S45/S182. Cache math and the public query contract are
//! unchanged.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

const UNIFORM_QUERY_AXIS: [u32; 27] = [
    1, 2, 4, 8, 16, 32, 64, 127, 128, 129, 131, 132, 133, 255, 256, 257, 263, 264, 265, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536,
];
const UNIFORM_CACHE_AXIS: [u32; 30] = [
    1, 2, 4, 6, 8, 11, 16, 32, 63, 64, 65, 91, 127, 128, 129, 256, 512, 1024, 2047, 2048, 2049,
    4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
];
const CAUSAL_QUERY_AXIS: [u32; 37] = [
    1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 127, 128, 129, 130, 131, 132, 133, 200, 254, 255,
    256, 257, 258, 263, 264, 265, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];
const CAUSAL_CACHE_AXIS: [u32; 41] = [
    1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 63, 64, 65, 90, 127, 128, 129, 130, 182, 200, 255, 256,
    260, 511, 512, 513, 724, 1024, 2047, 2048, 2049, 4096, 8192, 16384, 32768, 65536, 131072,
    262144, 524288, 1048576,
];
const SPECULATIVE_QUERY_AXIS: [u32; 20] = [
    2, 4, 8, 16, 32, 64, 128, 132, 134, 256, 264, 266, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    65536,
];
const SPECULATIVE_CACHE_AXIS: [u32; 31] = [
    1, 2, 3, 4, 8, 16, 23, 32, 45, 63, 64, 65, 127, 128, 129, 182, 256, 512, 1024, 2047, 2048,
    2049, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
];

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

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let (query_axis, cache_axis): (&[u32], &[u32]) = match config.valid_counts_pattern.as_str()
        {
            "uniform_full" => (&UNIFORM_QUERY_AXIS, &UNIFORM_CACHE_AXIS),
            "causal_tail" => (&CAUSAL_QUERY_AXIS, &CAUSAL_CACHE_AXIS),
            "speculative_pairs" => (&SPECULATIVE_QUERY_AXIS, &SPECULATIVE_CACHE_AXIS),
            pattern => panic!("unsupported valid_counts_pattern {pattern:?}"),
        };
        SweepGrid::new(vec![
            Axis::values(query_axis.iter().copied()),
            Axis::values(cache_axis.iter().copied()),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // Sparse attention reads `selected_k` tokens per query, and `selected_k`
        // is a Config field, not an axis. Work is therefore ~linear in
        // `num_queries`; `num_cache_tokens` sets the gather footprint the
        // indices point into, not how many of them are read.
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        // TRTLLM-gen rejects the 65,536-query launch on B200. Production caps
        // one iteration far below that, so 32,768 remains a generous measured
        // extrapolation guard without requiring an unsupported launch.
        let trtllm_query_too_large = |num_queries: f64| {
            config.backends.contains(&"flashinfer_trtllm_fp8") && num_queries > 32_768.0
        };
        match config.valid_counts_pattern.as_str() {
            "uniform_full" => grid.expand_2d(|num_queries, _| trtllm_query_too_large(num_queries)),
            "causal_tail" => grid.expand_2d(|num_queries, num_cache_tokens| {
                num_queries > num_cache_tokens || trtllm_query_too_large(num_queries)
            }),
            "speculative_pairs" => grid.expand_2d(|num_queries, _| {
                num_queries as u32 % 2 != 0 || trtllm_query_too_large(num_queries)
            }),
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
        DsaSparseMlaAttentionKernelInput, DsaSparseMlaAttentionSpec, CAUSAL_CACHE_AXIS,
        CAUSAL_QUERY_AXIS, SPECULATIVE_CACHE_AXIS, SPECULATIVE_QUERY_AXIS, UNIFORM_CACHE_AXIS,
        UNIFORM_QUERY_AXIS,
    };
    use crate::timing::bridge::{ArgsPayload, DType};
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{Dim, SlotInput, SweepCoords};
    use serde_json::Value;
    use std::collections::BTreeSet;

    const TORCH_BACKEND: &str = "torch";
    const FLASHMLA_BACKEND: &str = "vllm_flashmla_bf16";
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
    fn grids_have_the_exact_pattern_specific_axes_and_boundaries() {
        assert_grid(
            "uniform_full",
            &UNIFORM_QUERY_AXIS,
            &UNIFORM_CACHE_AXIS,
            810,
        );
        assert_grid("causal_tail", &CAUSAL_QUERY_AXIS, &CAUSAL_CACHE_AXIS, 1_517);
        assert_grid(
            "speculative_pairs",
            &SPECULATIVE_QUERY_AXIS,
            &SPECULATIVE_CACHE_AXIS,
            620,
        );

        // Every pattern's cache axis now reaches the arch's full 1,048,576-token
        // timing domain, and every query axis reaches 65,536.
        for pattern in ["uniform_full", "causal_tail", "speculative_pairs"] {
            let grid = DsaSparseMlaAttentionSpec::sweep_grid(&config(pattern));
            assert_eq!(grid.axes()[0].last(), Some(&65536.0));
            assert_eq!(grid.axes()[1].last(), Some(&1_048_576.0));
        }

        let uniform = DsaSparseMlaAttentionSpec::sweep_grid(&config("uniform_full"));
        assert_contiguous(&uniform.axes()[0], &[127, 128, 129]);
        assert_contiguous(&uniform.axes()[0], &[131, 132, 133]);
        assert_contiguous(&uniform.axes()[0], &[255, 256, 257, 263, 264, 265]);
        assert_contiguous(&uniform.axes()[1], &[2, 4, 6, 8, 11, 16]);
        assert_contiguous(&uniform.axes()[1], &[65, 91, 127]);
        assert_contiguous(&uniform.axes()[1], &[2047, 2048, 2049]);

        let causal = DsaSparseMlaAttentionSpec::sweep_grid(&config("causal_tail"));
        assert_contiguous(&causal.axes()[0], &[1, 2, 3, 4, 6, 8, 11]);
        assert_contiguous(&causal.axes()[0], &[127, 128, 129, 130, 131, 132, 133]);
        assert_contiguous(&causal.axes()[0], &[254, 255, 256, 257, 258, 263, 264, 265]);
        assert_contiguous(&causal.axes()[1], &[127, 128, 129, 130, 182, 200]);
        assert_contiguous(&causal.axes()[1], &[255, 256, 260, 511, 512, 513, 724]);
        assert_contiguous(&causal.axes()[1], &[2047, 2048, 2049]);

        let speculative = DsaSparseMlaAttentionSpec::sweep_grid(&config("speculative_pairs"));
        assert!(speculative.axes()[0]
            .iter()
            .all(|query| *query as u32 % 2 == 0));
        assert_contiguous(&speculative.axes()[0], &[128, 132, 134]);
        assert_contiguous(&speculative.axes()[0], &[256, 264, 266]);
        assert_contiguous(&speculative.axes()[1], &[16, 23, 32, 45, 63]);
        assert_contiguous(&speculative.axes()[1], &[129, 182, 256]);
        assert_contiguous(&speculative.axes()[1], &[2047, 2048, 2049]);
    }

    #[test]
    fn masks_match_each_frozen_pattern_domain() {
        let uniform_grid = DsaSparseMlaAttentionSpec::sweep_grid(&config("uniform_full"));
        let uniform = mask_for("uniform_full", &uniform_grid);
        assert_mask_split(&uniform, 810, 0);

        let causal_grid = DsaSparseMlaAttentionSpec::sweep_grid(&config("causal_tail"));
        let causal = mask_for("causal_tail", &causal_grid);
        assert_mask_split(&causal, 858, 659);
        assert!(!masked(&causal, &causal_grid, 3, 3));
        assert!(!masked(&causal, &causal_grid, 3, 4));
        assert!(masked(&causal, &causal_grid, 3, 2));
        assert!(!masked(&causal, &causal_grid, 132, 182));
        assert!(masked(&causal, &causal_grid, 200, 182));
        assert!(!masked(&causal, &causal_grid, 255, 255));
        assert!(masked(&causal, &causal_grid, 256, 255));
        assert!(!masked(&causal, &causal_grid, 4096, 131072));
        assert!(!masked(&causal, &causal_grid, 65536, 1_048_576));
        assert!(masked(&causal, &causal_grid, 65536, 32768));

        let speculative_grid = DsaSparseMlaAttentionSpec::sweep_grid(&config("speculative_pairs"));
        let speculative = mask_for("speculative_pairs", &speculative_grid);
        assert_mask_split(&speculative, 620, 0);
        assert!(!masked(&speculative, &speculative_grid, 2, 1));
        assert!(!masked(&speculative, &speculative_grid, 32, 2048));
        assert!(!masked(&speculative, &speculative_grid, 134, 65536));
        assert!(!masked(&speculative, &speculative_grid, 266, 65536));
        assert!(!masked(&speculative, &speculative_grid, 4096, 131072));
        assert!(!masked(&speculative, &speculative_grid, 65536, 1_048_576));
    }

    #[test]
    fn trtllm_fp8_masks_only_the_rejected_65536_query_row() {
        let mut cfg = config("causal_tail");
        cfg.backends = vec!["flashinfer_trtllm_fp8"];
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        let mask = DsaSparseMlaAttentionSpec::infeasible_mask(&cfg, &grid);

        assert!(!masked(&mask, &grid, 32768, 1_048_576));
        assert!(masked(&mask, &grid, 65536, 1_048_576));
    }

    #[test]
    fn speculative_odd_query_defense_remains_for_synthetic_grids() {
        let grid = crate::timing::sweep::SweepGrid::new(vec![
            crate::timing::sweep::Axis::values([2, 3, 4]),
            crate::timing::sweep::Axis::values([1, 2]),
        ]);
        let mask = mask_for("speculative_pairs", &grid);

        assert_mask_split(&mask, 4, 2);
        assert!(masked(&mask, &grid, 3, 1));
        assert!(masked(&mask, &grid, 3, 2));
        assert_eq!(
            canonical_valid_counts("speculative_pairs", 3, 2, 2048),
            "u:0x3"
        );
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
    fn unknown_pattern_fails_during_valid_count_derivation() {
        let _ = canonical_valid_counts("requests", 1, 1, 2048);
    }

    #[test]
    #[should_panic(expected = "unsupported valid_counts_pattern")]
    fn unknown_pattern_fails_during_grid_construction() {
        let _ = DsaSparseMlaAttentionSpec::sweep_grid(&config("requests"));
    }

    #[test]
    fn both_backends_use_the_existing_physical_bilinear_cache() {
        assert_eq!(
            DsaSparseMlaAttentionSpec::cache_kind(TORCH_BACKEND),
            CacheKind::Cache2DLinear(Extrapolation::Weighted)
        );
        assert_eq!(
            DsaSparseMlaAttentionSpec::cache_kind(FLASHMLA_BACKEND),
            CacheKind::Cache2DLinear(Extrapolation::Weighted)
        );
    }

    #[test]
    fn enumeration_emits_backend_plus_the_exact_python_schema() {
        for (pattern, expected_payloads) in [
            ("uniform_full", 810),
            ("causal_tail", 1_517),
            ("speculative_pairs", 620),
        ] {
            let cfg = config(pattern);
            let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
            let payloads = DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, FLASHMLA_BACKEND);

            assert_eq!(payloads.len(), expected_payloads);
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

            let first_q = grid.axes()[0][0] as u32;
            let first_s = grid.axes()[1][0] as u32;
            assert_payload(
                &payloads[0],
                first_q,
                first_s,
                canonical_valid_counts(pattern, first_q, first_s, 2048),
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
            let last_q = *grid.axes()[0].last().unwrap() as u32;
            let last_s = *grid.axes()[1].last().unwrap() as u32;
            assert_payload(
                payloads.last().unwrap(),
                last_q,
                last_s,
                canonical_valid_counts(pattern, last_q, last_s, 2048),
            );
        }

        assert_critical_payloads(
            "uniform_full",
            &[(132, 6), (132, 11), (132, 91), (131, 8192), (133, 8192)],
        );
        assert_critical_payloads(
            "causal_tail",
            &[
                (3, 3),
                (132, 182),
                (32, 511),
                (32, 513),
                (255, 724),
                (130, 130),
                (254, 255),
                (258, 260),
                (4096, 1),
            ],
        );
        assert_critical_payloads(
            "speculative_pairs",
            &[
                (32, 3),
                (132, 23),
                (132, 45),
                (132, 182),
                (134, 65536),
                (266, 65536),
                (4096, 131072),
            ],
        );
    }

    #[test]
    fn remediated_pattern_union_matches_the_frozen_next_phase_inventory() {
        let distributions = [
            "recent_contiguous",
            "unique_scattered_pages",
            "clustered_pages",
            "uniform_stride",
        ];
        let mut all_distributions = BTreeSet::new();
        let mut memberships = 0;

        for distribution in distributions {
            let uniform = feasible_payload_keys("uniform_full", distribution);
            let causal = feasible_payload_keys("causal_tail", distribution);
            let speculative = feasible_payload_keys("speculative_pairs", distribution);

            assert_eq!(uniform.len(), 810);
            assert_eq!(causal.len(), 858);
            assert_eq!(speculative.len(), 620);
            assert_eq!(uniform.intersection(&causal).count(), 249);
            assert_eq!(uniform.intersection(&speculative).count(), 180);
            assert_eq!(causal.intersection(&speculative).count(), 168);
            assert_eq!(
                uniform
                    .intersection(&causal)
                    .filter(|payload| speculative.contains(*payload))
                    .count(),
                148
            );

            memberships += uniform.len() + causal.len() + speculative.len();
            let per_distribution: BTreeSet<_> = uniform
                .into_iter()
                .chain(causal)
                .chain(speculative)
                .collect();
            assert_eq!(per_distribution.len(), 1_839);
            all_distributions.extend(per_distribution);
        }

        assert_eq!(memberships, 9_152);
        assert_eq!(all_distributions.len(), 7_356);
    }

    fn rich_dim(name: &str, value: u32) -> Value {
        serde_json::json!({
            "value": value,
            "expression": name,
            "bindings": {name: value},
        })
    }

    fn assert_grid(pattern: &str, query_axis: &[u32], cache_axis: &[u32], cells: usize) {
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&config(pattern));
        let axes = grid.axes();

        assert_eq!(axes.len(), 2);
        assert_eq!(axes[0], axis_values(query_axis));
        assert_eq!(axes[1], axis_values(cache_axis));
        assert!(axes[0].windows(2).all(|pair| pair[0] < pair[1]));
        assert!(axes[1].windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(axes[0].first().copied(), Some(f64::from(query_axis[0])));
        assert_eq!(
            axes[0].last().copied(),
            Some(f64::from(*query_axis.last().unwrap()))
        );
        assert_eq!(axes[1].first().copied(), Some(f64::from(cache_axis[0])));
        assert_eq!(
            axes[1].last().copied(),
            Some(f64::from(*cache_axis.last().unwrap()))
        );
        assert_eq!(axes[0].len() * axes[1].len(), cells);
    }

    fn axis_values(axis: &[u32]) -> Vec<f64> {
        axis.iter().map(|&value| f64::from(value)).collect()
    }

    fn assert_contiguous(axis: &[f64], expected: &[u32]) {
        let expected = axis_values(expected);
        assert!(
            axis.windows(expected.len())
                .any(|window| window == expected.as_slice()),
            "axis {axis:?} does not contain contiguous sequence {expected:?}"
        );
    }

    fn assert_critical_payloads(pattern: &str, coordinates: &[(u32, u32)]) {
        let cfg = config(pattern);
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        let payloads = DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, FLASHMLA_BACKEND);

        for &(q, s) in coordinates {
            assert_payload(
                payload_for(&payloads, &grid, q, s),
                q,
                s,
                canonical_valid_counts(pattern, q, s, 2048),
            );
        }
    }

    fn feasible_payload_keys(pattern: &str, distribution: &str) -> BTreeSet<String> {
        let mut cfg = config(pattern);
        cfg.index_distribution = distribution.to_string();
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        let mask = DsaSparseMlaAttentionSpec::infeasible_mask(&cfg, &grid);
        DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, FLASHMLA_BACKEND)
            .into_iter()
            .zip(mask)
            .filter_map(|(payload, masked)| {
                (!masked).then(|| serde_json::to_string(payload.fields()).unwrap())
            })
            .collect()
    }

    fn mask_for(pattern: &str, grid: &crate::timing::sweep::SweepGrid) -> Vec<bool> {
        DsaSparseMlaAttentionSpec::infeasible_mask(&config(pattern), grid)
    }

    fn assert_mask_split(mask: &[bool], feasible: usize, masked: usize) {
        assert_eq!(mask.len(), feasible + masked);
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
