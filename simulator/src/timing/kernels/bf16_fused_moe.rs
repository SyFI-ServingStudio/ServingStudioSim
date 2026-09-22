//! Whole FlashInfer SM100 BF16 fused-MoE assembly.
//!
//! Routing, gate/up, down, and finalize are one measured L1 boundary because the
//! production `trtllm_bf16_moe` callable owns the complete launch sequence and
//! never exposes an unfinalized intermediate.
//!
//! This is the unquantized sibling of `nvfp4_fused_moe`: same routing identity,
//! same per-rank expert histogram, same sweep coordinate. It exists because the
//! GLM-5.2 MTP draft layer runs BF16 weights while the target layers run NVFP4,
//! so costing the draft through the NVFP4 table would price the wrong weights.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Bf16FusedMoeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_size: Dim,
    pub intermediate_size: Dim,
    pub num_experts: Dim,
    pub num_local_experts: Dim,
    pub top_k: u32,
    #[compute_dtype]
    pub dtype: DType,
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
pub struct Bf16FusedMoeKernelInput {
    pub num_tokens: u32,
}

pub struct Bf16FusedMoeSpec;

impl KernelSpec for Bf16FusedMoeSpec {
    type Config = Bf16FusedMoeKernelConfig;
    type Input = Bf16FusedMoeKernelInput;

    const KIND: KernelKind = "bf16_fused_moe";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        // Deliberately the same axis as `nvfp4_fused_moe`. A draft step and a
        // target step are measured at the same token counts, so the two surfaces
        // stay directly comparable instead of requiring interpolation on one
        // side of every speculative comparison.
        SweepGrid::new(vec![config.expert_demand.token_axis(Axis::chain([
            Axis::values([1, 4, 8, 16, 32, 48]),
            Axis::token_axis(),
        ]))])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
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
            .expect("a validated expert-demand source must stay readable");

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
                .with("dtype", config.dtype.as_str())
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

register_kernel!(Bf16FusedMoeKernel, Bf16FusedMoeSpec);

#[cfg(test)]
mod tests {
    use super::{Bf16FusedMoeKernelConfig, Bf16FusedMoeKernelInput, Bf16FusedMoeSpec};
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::expert_demand::ExpertDemand;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::kernels::nvfp4_fused_moe::Nvfp4FusedMoeSpec;
    use crate::timing::{Dim, SweepCoords};
    use serde_json::Value;

    const BACKEND: &str = "flashinfer_trtllm_sm100";
    const NUM_EXPERTS: u32 = 8;
    const NUM_LOCAL_EXPERTS: u32 = 4;
    const TOP_K: u32 = 2;

    /// Two modeled layers, deliberately skewed so a rank rotation is observable.
    fn popularity() -> ExpertDemand {
        ExpertDemand::Popularity {
            layerwise_global_ppm: vec![
                vec![
                    400_000, 300_000, 100_000, 100_000, 50_000, 30_000, 15_000, 5_000,
                ],
                vec![
                    300_000, 300_000, 150_000, 100_000, 80_000, 40_000, 20_000, 10_000,
                ],
            ],
        }
    }

    fn config(folded_rank_position: u32) -> Bf16FusedMoeKernelConfig {
        Bf16FusedMoeKernelConfig {
            backends: vec![BACKEND],
            gpu_name: "NVIDIA B200".to_string(),
            hidden_size: Dim::param("hidden_size", 6144),
            intermediate_size: Dim::param("moe_intermediate_size", 1536),
            num_experts: Dim::param("n_routed_experts", NUM_EXPERTS),
            num_local_experts: Dim::param("num_local_experts", NUM_LOCAL_EXPERTS),
            top_k: TOP_K,
            dtype: DType::Bf16,
            routing_method: "minimax2".to_string(),
            n_group: 1,
            topk_group: 1,
            routed_scaling_numerator: 1,
            routed_scaling_denominator: 1,
            expert_demand: popularity(),
            folded_rank_position,
        }
    }

    #[test]
    fn config_kind_and_dtype_identity_match_the_python_handoff() {
        let cfg = config(0);

        assert_eq!(Bf16FusedMoeSpec::KIND, "bf16_fused_moe");
        assert_eq!(Bf16FusedMoeSpec::profile_kind(), "bf16_fused_moe");
        assert_eq!(cfg.backends(), &[BACKEND]);
        assert_eq!(cfg.gpu_name(), "NVIDIA B200");
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
        assert_eq!(
            Bf16FusedMoeSpec::cache_kind(BACKEND),
            CacheKind::Cache1DLinear
        );
    }

    #[test]
    fn input_is_the_single_physical_token_coordinate() {
        let input: Bf16FusedMoeKernelInput = serde_json::from_str(r#"{"num_tokens":48}"#).unwrap();

        assert_eq!(&*input.coords(), &[48.0]);
        assert_eq!(
            Bf16FusedMoeKernelInput::coord_field_names(),
            &["num_tokens"]
        );
    }

    /// The draft and target MoE surfaces are compared step by step, so a token
    /// axis that drifted from the NVFP4 sibling would force one side of every
    /// speculative comparison through interpolation.
    #[test]
    fn token_axis_matches_the_nvfp4_sibling_exactly() {
        let bf16 = Bf16FusedMoeSpec::sweep_grid(&config(0));
        let nvfp4_axis = Nvfp4FusedMoeSpec::sweep_grid(
            &crate::timing::kernels::nvfp4_fused_moe::Nvfp4FusedMoeKernelConfig {
                backends: vec![BACKEND],
                gpu_name: "NVIDIA B200".to_string(),
                hidden_size: Dim::param("hidden_size", 6144),
                intermediate_size: Dim::param("moe_intermediate_size", 1536),
                num_experts: Dim::param("n_routed_experts", NUM_EXPERTS),
                num_local_experts: Dim::param("num_local_experts", NUM_LOCAL_EXPERTS),
                top_k: TOP_K,
                input_dtype: DType::Bf16,
                weight_format: "nvfp4_e2m1".to_string(),
                group_size: 16,
                routing_method: "minimax2".to_string(),
                n_group: 1,
                topk_group: 1,
                routed_scaling_numerator: 1,
                routed_scaling_denominator: 1,
                expert_demand: popularity(),
                folded_rank_position: 0,
            },
        );

        assert_eq!(bf16.axes(), nvfp4_axis.axes());
        assert_eq!(bf16.axes()[0].first(), Some(&1.0));
        assert_eq!(bf16.axes()[0].last(), Some(&65536.0));
        assert!(bf16.axes()[0].windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn enumerate_matches_the_full_python_wire_schema() {
        let cfg = config(0);
        let grid = Bf16FusedMoeSpec::sweep_grid(&cfg);
        let payloads = Bf16FusedMoeSpec::enumerate(&cfg, &grid, BACKEND);

        assert_eq!(payloads.len(), grid.axes()[0].len());
        let expected_names = [
            "backend",
            "dtype",
            "hidden_size",
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
        ];
        for payload in &payloads {
            let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
            assert_eq!(names, expected_names);
        }

        let first = payloads[0].fields();
        assert_eq!(first.get("num_tokens"), Some(&Value::from(1_u32)));
        assert_eq!(first.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(first.get("hidden_size"), Some(&Value::from(6144_u32)));
        assert_eq!(first.get("top_k"), Some(&Value::from(TOP_K)));
    }

    /// The runner realizes these counts as exact per-token top-k degrees, so a
    /// histogram that did not sum to `num_tokens * top_k`, or that was not
    /// rotated onto the profiled rank, would make the runner reject the spec.
    #[test]
    fn per_expert_batches_are_a_rank_local_exact_topk_histogram() {
        for folded_rank_position in [0, 1] {
            let cfg = config(folded_rank_position);
            let grid = Bf16FusedMoeSpec::sweep_grid(&cfg);
            let payloads = Bf16FusedMoeSpec::enumerate(&cfg, &grid, BACKEND);

            for (index, payload) in payloads.iter().enumerate() {
                let num_tokens = grid.axes()[0][index] as u64;
                let batches: Vec<u64> = payload.fields()["per_expert_batches"]
                    .as_array()
                    .expect("per_expert_batches must be an array")
                    .iter()
                    .map(|value| value.as_u64().expect("counts must be unsigned"))
                    .collect();

                assert_eq!(batches.len(), NUM_EXPERTS as usize);
                assert_eq!(batches.iter().sum::<u64>(), num_tokens * u64::from(TOP_K));
                assert!(
                    batches.iter().all(|&batch| batch <= num_tokens),
                    "one expert cannot take more than one row per token"
                );
            }
        }
    }

    /// `folded_rank_position` selects which slice of the global histogram lands
    /// in the leading `num_local_experts` entries the runner bills as local. A
    /// rotation that did not move would price every EP rank as rank 0.
    #[test]
    fn folded_rank_position_rotates_the_local_slice() {
        let grid = Bf16FusedMoeSpec::sweep_grid(&config(0));
        let index = grid.axes()[0]
            .iter()
            .position(|&tokens| tokens == 256.0)
            .expect("256 tokens is on the shared axis");

        let local_of = |folded_rank_position: u32| -> Vec<u64> {
            let cfg = config(folded_rank_position);
            let payloads = Bf16FusedMoeSpec::enumerate(&cfg, &grid, BACKEND);
            payloads[index].fields()["per_expert_batches"]
                .as_array()
                .unwrap()
                .iter()
                .take(NUM_LOCAL_EXPERTS as usize)
                .map(|value| value.as_u64().unwrap())
                .collect()
        };

        assert_ne!(local_of(0), local_of(1));
    }

    #[test]
    #[should_panic(expected = "folded_rank_position must select an EP rank")]
    fn folded_rank_position_past_the_last_rank_is_rejected() {
        let cfg = config(NUM_EXPERTS / NUM_LOCAL_EXPERTS);
        let grid = Bf16FusedMoeSpec::sweep_grid(&cfg);
        let _ = Bf16FusedMoeSpec::enumerate(&cfg, &grid, BACKEND);
    }
}
