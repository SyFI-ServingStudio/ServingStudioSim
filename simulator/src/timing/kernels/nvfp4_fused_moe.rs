//! Whole FlashInfer SM100 NVFP4 fused-MoE assembly.
//!
//! Routing, gate/up, down, and finalize are one measured L1 boundary because
//! Programmatic Dependent Launch overlaps their physical kernel intervals.
//!
//! Despite the kind name, `weight_format` selects the expert weight recipe and
//! the backend string selects the callable. The NVFP4 backends
//! (`flashinfer_trtllm_sm100*`) take `nvfp4_e2m1` / group 16. The GLM-5.3-Flash
//! backend `flashinfer_trtllm_fp8_block_sm100` (`trtllm_fp8_block_scale_moe`,
//! DeepSeek-FP8) takes `fp8_e4m3_block` / group 128 (128x128 weight blocks,
//! per-token-group-128 activations) with `deepseek_v3` routing,
//! `n_group = topk_group = 1`, and routed scaling 5/2. Its autotuner stops at
//! 8192 tokens, so larger grid points reuse the top tactic bucket.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Nvfp4FusedMoeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_size: Dim,
    pub intermediate_size: Dim,
    pub num_experts: Dim,
    pub num_local_experts: Dim,
    pub top_k: u32,
    #[compute_dtype]
    pub input_dtype: DType,
    pub weight_format: String,
    pub group_size: u32,
    pub routing_method: String,
    pub n_group: u32,
    pub topk_group: u32,
    pub routed_scaling_numerator: u32,
    pub routed_scaling_denominator: u32,
    /// Where the routed demand for a profiled shape comes from.
    pub expert_demand: ExpertDemand,
    /// Position in the active-count-ranked EP workload list. This chooses a
    /// representative local histogram without adding physical rank identity to
    /// the Python profiler key.
    pub folded_rank_position: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct Nvfp4FusedMoeKernelInput {
    pub num_tokens: u32,
}

pub struct Nvfp4FusedMoeSpec;

impl KernelSpec for Nvfp4FusedMoeSpec {
    type Config = Nvfp4FusedMoeKernelConfig;
    type Input = Nvfp4FusedMoeKernelInput;

    const KIND: KernelKind = "nvfp4_fused_moe";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![config.expert_demand.token_axis(Axis::chain([
            Axis::values([1, 4, 8, 16, 32, 48]),
            Axis::token_axis(),
        ]))])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    /// The corpus arm names a payload on disk, and `enumerate` has no way to
    /// report that it moved.
    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        config.expert_demand.prepare().map(|_| ())
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        // Resolved once: a token corpus is hundreds of megabytes on disk and
        // the grid has tens of points. The arch builder has already proven it
        // readable, so a failure here is a corpus that changed underneath a
        // built config.
        let demand = config
            .expert_demand
            .prepare()
            .expect("validate_config proved this source readable");

        grid.expand_1d(|num_tokens| {
            let per_expert_batches = demand.per_expert_batches(
                config.top_k,
                num_tokens as u32,
                config.num_experts.get() as usize,
                config.num_local_experts.get() as usize,
                config.folded_rank_position,
            );

            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
                .with("hidden_size", config.hidden_size.get())
                .with("intermediate_size", config.intermediate_size.get())
                .with("num_experts", config.num_experts.get())
                .with("num_local_experts", config.num_local_experts.get())
                .with("top_k", config.top_k)
                .with("input_dtype", config.input_dtype.as_str())
                .with("weight_format", config.weight_format.as_str())
                .with("group_size", config.group_size)
                .with("routing_method", config.routing_method.as_str())
                .with("n_group", config.n_group)
                .with("topk_group", config.topk_group)
                .with("routed_scaling_numerator", config.routed_scaling_numerator)
                .with(
                    "routed_scaling_denominator",
                    config.routed_scaling_denominator,
                )
                .with("per_expert_batches", per_expert_batches)
        })
    }
}

register_kernel!(Nvfp4FusedMoeKernel, Nvfp4FusedMoeSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn config() -> Nvfp4FusedMoeKernelConfig {
        Nvfp4FusedMoeKernelConfig {
            backends: vec!["flashinfer_trtllm_sm100"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden_size: 6144.into(),
            intermediate_size: 2048.into(),
            num_experts: 8.into(),
            num_local_experts: 4.into(),
            top_k: 2,
            input_dtype: DType::Bf16,
            weight_format: "nvfp4_e2m1".to_string(),
            group_size: 16,
            routing_method: "minimax2".to_string(),
            n_group: 1,
            topk_group: 1,
            routed_scaling_numerator: 5,
            routed_scaling_denominator: 2,
            expert_demand: ExpertDemand::Popularity {
                layerwise_global_ppm: vec![
                    vec![
                        250_000, 200_000, 150_000, 100_000, 100_000, 80_000, 70_000, 50_000,
                    ],
                    vec![
                        220_000, 200_000, 160_000, 120_000, 100_000, 80_000, 70_000, 50_000,
                    ],
                ],
            },
            folded_rank_position: 0,
        }
    }

    #[test]
    fn enumerate_matches_whole_operation_schema_without_rank_identity() {
        let payloads = Nvfp4FusedMoeSpec::enumerate(
            &config(),
            &SweepGrid::new(vec![Axis::values([16])]),
            "flashinfer_trtllm_sm100",
        );
        let fields = payloads[0].fields();
        let batches = fields["per_expert_batches"].as_array().unwrap();

        assert_eq!(fields.len(), 16);
        assert_eq!(fields.get("num_tokens"), Some(&Value::from(16_u32)));
        assert_eq!(batches.len(), 8);
        assert_eq!(
            batches
                .iter()
                .map(|value| value.as_u64().unwrap())
                .sum::<u64>(),
            32
        );
        assert!(!fields.contains_key("folded_rank_position"));
        assert!(!fields.contains_key("ep_rank"));
        assert!(!fields.contains_key("local_expert_offset"));
        assert!(!fields.contains_key("launch_role"));
    }

    /// GLM-5.3-Flash EP4 on the DeepSeek-FP8 block-scale backend.
    fn glm53_flash_ep4_fp8() -> Nvfp4FusedMoeKernelConfig {
        Nvfp4FusedMoeKernelConfig {
            backends: vec!["flashinfer_trtllm_fp8_block_sm100"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden_size: 4096.into(),
            intermediate_size: 2048.into(),
            num_experts: 288.into(),
            num_local_experts: 72.into(),
            top_k: 8,
            input_dtype: DType::Bf16,
            weight_format: "fp8_e4m3_block".to_string(),
            group_size: 128,
            routing_method: "deepseek_v3".to_string(),
            n_group: 1,
            topk_group: 1,
            routed_scaling_numerator: 5,
            routed_scaling_denominator: 2,
            expert_demand: ExpertDemand::popularity(
                &crate::timing::routing::RoutingDistribution::uniform(288),
                1,
            ),
            folded_rank_position: 0,
        }
    }

    /// The FP8 backend reuses the NVFP4 args schema; a dropped or renamed
    /// field would key a different profile row than the Python runner reads.
    #[test]
    fn fp8_block_backend_forwards_exactly_the_python_args() {
        let config = glm53_flash_ep4_fp8();
        let grid = SweepGrid::new(vec![Axis::values([1, 100, 3000])]);
        let payloads =
            Nvfp4FusedMoeSpec::enumerate(&config, &grid, "flashinfer_trtllm_fp8_block_sm100");
        assert_eq!(payloads.len(), 3);
        for (payload, tokens) in payloads.iter().zip([1_u64, 100, 3000]) {
            let fields = payload.fields();
            let mut names: Vec<&str> = fields.keys().map(String::as_str).collect();
            names.sort_unstable();
            assert_eq!(
                names,
                [
                    "backend",
                    "group_size",
                    "hidden_size",
                    "input_dtype",
                    "intermediate_size",
                    "n_group",
                    "num_experts",
                    "num_local_experts",
                    "num_tokens",
                    "per_expert_batches",
                    "routed_scaling_denominator",
                    "routed_scaling_numerator",
                    "routing_method",
                    "top_k",
                    "topk_group",
                    "weight_format",
                ]
            );
            let expect = [
                ("backend", Value::from("flashinfer_trtllm_fp8_block_sm100")),
                ("num_tokens", Value::from(tokens)),
                ("hidden_size", Value::from(4096)),
                ("intermediate_size", Value::from(2048)),
                ("num_experts", Value::from(288)),
                ("num_local_experts", Value::from(72)),
                ("top_k", Value::from(8)),
                ("input_dtype", Value::from("bf16")),
                ("weight_format", Value::from("fp8_e4m3_block")),
                ("group_size", Value::from(128)),
                ("routing_method", Value::from("deepseek_v3")),
                ("n_group", Value::from(1)),
                ("topk_group", Value::from(1)),
                ("routed_scaling_numerator", Value::from(5)),
                ("routed_scaling_denominator", Value::from(2)),
            ];
            for (name, value) in expect {
                assert_eq!(fields[name], value, "{name} at T={tokens}");
            }
            let batches: Vec<u64> = fields["per_expert_batches"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap())
                .collect();
            assert_eq!(batches.len(), 288, "global width at T={tokens}");
            assert_eq!(batches.iter().sum::<u64>(), tokens * 8, "assignments at T={tokens}");
            assert!(batches.iter().all(|&rows| rows <= tokens), "top-k is distinct");
        }
    }

    /// Fidelity helper, not a check: prints the exact payloads `enumerate`
    /// sends the profiler, so ground truth is measured on the histograms the
    /// cache was built from rather than on a Python re-sampling of them.
    ///
    /// `NVFP4_MOE_DUMP_CONFIG` names a `Nvfp4FusedMoeKernelConfig` JSON file,
    /// `NVFP4_MOE_DUMP_TOKENS` a comma list of token counts. Each payload is
    /// printed on one line after the `PAYLOAD ` marker. Used by
    /// `tools/cache-fidelity-analyzer/nvfp4_fused_moe_fidelity.py`.
    #[test]
    #[ignore = "fidelity payload dump; driven by nvfp4_fused_moe_fidelity.py"]
    fn dump_enumerate_payloads() {
        let (Ok(path), Ok(tokens)) = (
            std::env::var("NVFP4_MOE_DUMP_CONFIG"),
            std::env::var("NVFP4_MOE_DUMP_TOKENS"),
        ) else {
            return;
        };
        let config: Nvfp4FusedMoeKernelConfig =
            serde_json::from_slice(&std::fs::read(&path).expect("dump config readable"))
                .expect("dump config parses");
        Nvfp4FusedMoeSpec::validate_config(&config).expect("dump config valid");
        let tokens: Vec<f64> = tokens
            .split(',')
            .map(|t| t.trim().parse().expect("token count"))
            .collect();
        let grid = SweepGrid::new(vec![tokens]);
        for backend in &config.backends {
            for payload in Nvfp4FusedMoeSpec::enumerate(&config, &grid, backend) {
                println!("PAYLOAD {}", serde_json::to_string(payload.fields()).unwrap());
            }
        }
    }

    #[test]
    fn a_corpus_that_moved_is_reported_rather_than_panicking_the_query() {
        // `kernel-query` deserializes a config nothing built, so the file it
        // names may be gone. `enumerate` cannot say so; this is where it is said.
        let mut moved = config();
        moved.expert_demand = crate::timing::expert_demand::ExpertDemand::Corpus(
            crate::timing::token_corpus::TokenCorpusConfig {
                schema_version: 1,
                data_file: "/nonexistent/routes.u16".into(),
                num_tokens: 128,
                num_layers: 4,
                num_experts: 64,
                top_k: 8,
                checksum_fnv1a64: 0,
                group_size: 8,
                layer_start: 0,
                layer_end: 4,
                seed: 0,
                sampling_candidates: 16,
            },
        );

        let error = Nvfp4FusedMoeSpec::validate_config(&moved).expect_err("a moved corpus");
        assert!(format!("{error:#}").contains("token corpus data"));
    }

    #[test]
    fn ranked_positions_select_distinct_local_histograms_without_rank_args() {
        let grid = SweepGrid::new(vec![Axis::values([16])]);
        let first = Nvfp4FusedMoeSpec::enumerate(&config(), &grid, "flashinfer_trtllm_sm100");
        let mut second_config = config();
        second_config.folded_rank_position = 1;
        let second = Nvfp4FusedMoeSpec::enumerate(&second_config, &grid, "flashinfer_trtllm_sm100");

        assert_ne!(
            first[0].fields()["per_expert_batches"],
            second[0].fields()["per_expert_batches"]
        );
    }
}
