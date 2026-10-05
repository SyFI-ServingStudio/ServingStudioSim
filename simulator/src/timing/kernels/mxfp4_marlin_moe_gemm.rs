//! One packed-MXFP4 Marlin expert GEMM using the shared routing projection.

use std::collections::BTreeSet;

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::routing::RoutingDistribution;
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

const BLOCK_SIZES_M: [u32; 5] = [8, 16, 32, 48, 64];

#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mxfp4MarlinMoeFcRole {
    Fc1,
    Fc2,
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Mxfp4MarlinMoeGemmKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub fc_role: Mxfp4MarlinMoeFcRole,
    pub n: Dim,
    pub k: Dim,
    #[compute_dtype]
    pub dtype: DType,
    pub routing_top_k: u32,
    pub local_ppm: Vec<u32>,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct Mxfp4MarlinMoeGemmKernelInput {
    pub num_input_tokens: u32,
}

/// Returns the number of local experts (`E = w1.size(0)` in vLLM's
/// `fused_marlin_moe`), one per `local_ppm` entry.
fn validate_config(config: &Mxfp4MarlinMoeGemmKernelConfig) -> u32 {
    assert_eq!(
        config.dtype,
        DType::Bf16,
        "Marlin activation dtype must be bf16"
    );
    assert!(config.routing_top_k > 0, "routing_top_k must be positive");
    assert!(
        config.n.get() > 0 && config.k.get() > 0,
        "Marlin n and k must be positive"
    );
    assert!(
        !config.local_ppm.is_empty(),
        "local_ppm must hold one entry per local expert"
    );
    let total_ppm: u64 = config.local_ppm.iter().map(|&value| u64::from(value)).sum();
    assert!(
        (1..=1_000_000).contains(&total_ppm),
        "local ppm mass must be valid"
    );
    u32::try_from(config.local_ppm.len()).expect("local expert count must fit u32")
}

pub(crate) fn marlin_block_size_m(
    num_input_tokens: u32,
    routing_top_k: u32,
    num_local_experts: u32,
) -> u32 {
    let routed_rows = u64::from(num_input_tokens) * u64::from(routing_top_k);
    BLOCK_SIZES_M
        .into_iter()
        .find(|&block_size| {
            routed_rows * 10 < 9 * u64::from(num_local_experts) * u64::from(block_size)
        })
        .unwrap_or(64)
}

fn token_axis(routing_top_k: u32, num_local_experts: u32) -> Vec<f64> {
    let mut points: BTreeSet<u32> = [1, 2, 3, 4, 8, 16, 32, 48].into_iter().collect();
    points.extend(Axis::token_axis().into_iter().map(|value| value as u32));
    for block_size in [8, 16, 32, 48] {
        let transition = (9 * num_local_experts * block_size).div_ceil(10 * routing_top_k);
        if transition > 1 {
            points.insert(transition - 1);
            points.insert(transition);
        }
    }
    points.into_iter().map(f64::from).collect()
}

pub struct Mxfp4MarlinMoeGemmSpec;

impl KernelSpec for Mxfp4MarlinMoeGemmSpec {
    type Config = Mxfp4MarlinMoeGemmKernelConfig;
    type Input = Mxfp4MarlinMoeGemmKernelInput;

    const KIND: KernelKind = "mxfp4_marlin_moe_gemm";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let num_local_experts = validate_config(config);
        SweepGrid::new(vec![token_axis(config.routing_top_k, num_local_experts)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        let num_local_experts = validate_config(config);
        grid.expand_1d(|num_input_tokens| {
            let num_input_tokens = num_input_tokens as u32;
            let global_expert_selections = num_input_tokens
                .checked_mul(config.routing_top_k)
                .expect("num_input_tokens * routing_top_k must fit u32");
            let per_group_batches = RoutingDistribution::to_per_expert_counts(
                global_expert_selections,
                &config.local_ppm,
            );
            let (m, input_top_k, mul_topk_weights) = match config.fc_role {
                Mxfp4MarlinMoeFcRole::Fc1 => (num_input_tokens, config.routing_top_k, false),
                Mxfp4MarlinMoeFcRole::Fc2 => (global_expert_selections, 1, true),
            };
            ArgsPayload::new()
                .with("backend", backend)
                .with("m", m)
                .with("n", config.n.get())
                .with("k", config.k.get())
                .with("dtype", config.dtype.as_str())
                .with("input_top_k", input_top_k)
                .with(
                    "block_size_m",
                    marlin_block_size_m(num_input_tokens, config.routing_top_k, num_local_experts),
                )
                .with("mul_topk_weights", mul_topk_weights)
                .with("per_group_batches", per_group_batches)
        })
    }
}

register_kernel!(Mxfp4MarlinMoeGemmKernel, Mxfp4MarlinMoeGemmSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::engine::KernelSpec;

    fn config(fc_role: Mxfp4MarlinMoeFcRole) -> Mxfp4MarlinMoeGemmKernelConfig {
        let (n, k) = match fc_role {
            Mxfp4MarlinMoeFcRole::Fc1 => (4096, 4096),
            Mxfp4MarlinMoeFcRole::Fc2 => (4096, 2048),
        };
        Mxfp4MarlinMoeGemmKernelConfig {
            backends: vec!["vllm_marlin"],
            gpu_name: "NVIDIA H200".to_string(),
            fc_role,
            n: n.into(),
            k: k.into(),
            dtype: DType::Bf16,
            routing_top_k: 6,
            local_ppm: vec![3_906; 64],
        }
    }

    #[test]
    fn block_axis_contains_both_sides_of_every_production_transition() {
        let grid = Mxfp4MarlinMoeGemmSpec::sweep_grid(&config(Mxfp4MarlinMoeFcRole::Fc1));
        for tokens in [76.0, 77.0, 153.0, 154.0, 307.0, 308.0, 460.0, 461.0] {
            assert!(grid.axes()[0].contains(&tokens));
        }
    }

    #[test]
    fn enumeration_uses_shared_active_set_routing_and_exact_python_fields() {
        let config = config(Mxfp4MarlinMoeFcRole::Fc1);
        let grid = SweepGrid::new(vec![vec![128.0]]);
        let payload = &Mxfp4MarlinMoeGemmSpec::enumerate(&config, &grid, "vllm_marlin")[0];
        let expected = RoutingDistribution::to_per_expert_counts(128 * 6, &config.local_ppm);
        assert!(
            expected.contains(&0),
            "active-set routing must leave inactive experts"
        );
        assert_eq!(
            payload.fields()["per_group_batches"],
            serde_json::json!(expected)
        );
        assert_eq!(
            payload
                .fields()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            [
                "backend",
                "block_size_m",
                "dtype",
                "input_top_k",
                "k",
                "m",
                "mul_topk_weights",
                "n",
                "per_group_batches",
            ]
        );
    }

    #[test]
    fn fc2_expands_m_but_keeps_the_same_shared_routing_projection() {
        let config = config(Mxfp4MarlinMoeFcRole::Fc2);
        let grid = SweepGrid::new(vec![vec![128.0]]);
        let payload = &Mxfp4MarlinMoeGemmSpec::enumerate(&config, &grid, "vllm_marlin")[0];
        assert_eq!(payload.fields()["m"], 768);
        assert_eq!(payload.fields()["input_top_k"], 1);
        assert_eq!(payload.fields()["mul_topk_weights"], true);
    }

    #[test]
    fn local_expert_count_follows_the_shard_width() {
        // An EP8 shard of 256 experts holds 32; block size uses E = 32.
        let config = Mxfp4MarlinMoeGemmKernelConfig {
            local_ppm: vec![7_812; 32],
            ..config(Mxfp4MarlinMoeFcRole::Fc1)
        };
        let grid = SweepGrid::new(vec![vec![128.0]]);
        let payload = &Mxfp4MarlinMoeGemmSpec::enumerate(&config, &grid, "vllm_marlin")[0];
        let batches = payload.fields()["per_group_batches"].as_array().unwrap();
        assert_eq!(batches.len(), 32);
        assert_eq!(
            payload.fields()["block_size_m"],
            marlin_block_size_m(128, 6, 32)
        );
        let grid = Mxfp4MarlinMoeGemmSpec::sweep_grid(&config);
        let axis = &grid.axes()[0];
        // 0.9 * 32 * 8 / 6 = 38.4: the 8 -> 16 transition sits at 39 tokens.
        assert!(axis.contains(&38.0) && axis.contains(&39.0));
    }
}
