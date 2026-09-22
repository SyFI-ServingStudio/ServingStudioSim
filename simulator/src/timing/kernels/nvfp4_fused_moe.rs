//! Whole FlashInfer SM100 NVFP4 fused-MoE assembly.
//!
//! Routing, gate/up, down, and finalize are one measured L1 boundary because
//! Programmatic Dependent Launch overlaps their physical kernel intervals.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::routing::sample_and_fold_layerwise_topk_expert_counts;
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

const FOLD_SEED: u64 = 0xF01D_5EED;

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
    /// Identity-free global popularity for every modeled MoE layer.
    pub layerwise_global_ppm: Vec<Vec<u32>>,
    #[serde(default)]
    pub token_corpus: Option<crate::timing::token_corpus::TokenCorpusConfig>,
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
        let mut values = Axis::chain([Axis::values([1, 4, 8, 16, 32, 48]), Axis::token_axis()]);
        if let Some(corpus) = &config.token_corpus {
            values = Axis::chain([
                values,
                (1..=8)
                    .map(|i| f64::from(corpus.group_size) * f64::from(i))
                    .collect(),
            ]);
        }
        SweepGrid::new(vec![values])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        let num_experts = config.num_experts.get() as usize;
        let num_local_experts = config.num_local_experts.get() as usize;
        assert!(num_local_experts > 0, "num_local_experts must be non-zero");
        assert_eq!(
            num_experts % num_local_experts,
            0,
            "local experts must evenly partition global experts"
        );
        let rank_offset = config.folded_rank_position as usize * num_local_experts;
        assert!(
            rank_offset + num_local_experts <= num_experts,
            "folded_rank_position must select an EP rank"
        );

        let corpus = config.token_corpus.as_ref().map(|c| {
            assert_eq!(c.num_experts, num_experts);
            assert_eq!(c.top_k, config.top_k as usize);
            c.load()
                .expect("validated token corpus must remain readable and unchanged")
        });
        grid.expand_1d(|num_tokens| {
            let mut per_expert_batches = if let Some(corpus) = &corpus {
                corpus.sample_and_fold(num_tokens as u32, num_local_experts)
            } else {
                sample_and_fold_layerwise_topk_expert_counts(
                    &config.layerwise_global_ppm,
                    config.top_k,
                    num_tokens as u32,
                    num_local_experts,
                    FOLD_SEED,
                )
            };
            assert_eq!(
                per_expert_batches.len(),
                num_experts,
                "routing profile width must match num_experts"
            );
            per_expert_batches.rotate_left(rank_offset);

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
            layerwise_global_ppm: vec![
                vec![
                    250_000, 200_000, 150_000, 100_000, 100_000, 80_000, 70_000, 50_000,
                ],
                vec![
                    220_000, 200_000, 160_000, 120_000, 100_000, 80_000, 70_000, 50_000,
                ],
            ],
            token_corpus: None,
            folded_rank_position: 0,
        }
    }

    #[test]
    fn corpus_grid_includes_verify_multiples_and_stays_bounded() {
        let mut config = config();
        config.token_corpus = Some(crate::timing::token_corpus::TokenCorpusConfig {
            schema_version: 1,
            data_file: String::new(),
            num_tokens: 100,
            num_layers: 2,
            num_experts: 8,
            top_k: 2,
            checksum_fnv1a64: 0,
            group_size: 8,
            seed: 42,
            sampling_candidates: 16,
        });
        let grid = Nvfp4FusedMoeSpec::sweep_grid(&config);
        let values = grid.expand_1d(|n| n as u32);
        assert!(values.contains(&24) && values.contains(&40) && values.contains(&56));
        assert!(values.len() <= 500);
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
