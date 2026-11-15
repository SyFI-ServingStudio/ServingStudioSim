//! DeepGEMM's fused expert-parallel MoE layer (`deep_gemm.fp8_fp4_mega_moe`).
//!
//! One `sm100_fp8_fp4_mega_moe_impl` launch per GPU runs the whole routed MoE
//! of one EP rank: dispatch over NVLink from every peer's symmetric buffer,
//! gate/up GEMM, clamped SwiGLU, down GEMM, combine back to the source ranks,
//! and the fused shared expert. DeepInfer calls it once per MoE layer
//! (`v4_modules/mlp.py` `mega_moe_forward`).
//!
//! The query is the per-rank token count. The key also carries the rows every
//! physical slot of the group receives (`per_slot_rows`), drawn from the
//! deployment's expert demand and placed by its [`SlotPlacement`]: a rank's
//! work is its slots' M blocks (`ceil(rows / block_m)` per slot, DeepGEMM
//! `scheduler/mega_moe.cuh`) plus the shared expert on its own tokens, and
//! the combine waits for every peer, so the slowest rank sets the layer's
//! time. The tile (`get_block_m_for_mega_moe`) follows from `ep_size`,
//! `num_experts`, `top_k` and the token count only, all in the key.
//!
//! `max_tokens_per_rank` (the symmetric buffer's capacity, a kernel template
//! argument) only bounds the token axis; it is not a profiler key because an
//! A/B at fixed token count showed no timing effect.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::expert_placement::SlotPlacement;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MegaMoeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// GPUs in the expert-parallel group; the kernel synchronizes all of them.
    pub ep_size: u32,
    pub hidden_size: Dim,
    /// One routed expert's intermediate width (the shared expert's is
    /// `num_shared_experts` times it).
    pub intermediate_size: Dim,
    /// Physical routed slots over the whole group, redundant (EPLB) slots
    /// included; each GPU holds `num_experts / ep_size`.
    pub num_experts: u32,
    pub num_shared_experts: u32,
    pub top_k: u32,
    /// MXFP8 activations (UE8M0 scale per 32).
    pub activation_dtype: DType,
    /// MXFP4 routed weights; the tensor-core format the backend gates on.
    #[compute_dtype]
    pub weight_dtype: DType,
    /// Largest per-rank token count the deployment runs (the symmetric
    /// buffer's capacity). Bounds the token axis only.
    pub max_tokens_per_rank: u32,
    /// Where the routed rows come from (over the logical experts).
    pub expert_demand: ExpertDemand,
    /// How the logical experts sit on the `num_experts` physical slots.
    pub placement: SlotPlacement,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MegaMoeKernelInput {
    /// Tokens entering the layer on each rank (the busiest rank when they differ).
    pub num_tokens: u32,
}

pub struct MegaMoeSpec;

/// Small-batch head of the token axis, below `Axis::token_axis`'s 32.
const SMALL_TOKENS: [u32; 5] = [1, 2, 4, 8, 16];
/// Points past `Axis::token_axis`'s 65536, for deployments whose per-rank
/// budget exceeds it.
const LARGE_TOKENS: [u32; 2] = [98304, 131072];

/// The generic per-rank token axis, cut after the first point at or above
/// `max_tokens_per_rank` so the deployment's largest batch interpolates.
pub(crate) fn token_points(max_tokens_per_rank: u32) -> Vec<f64> {
    let full = Axis::chain([
        Axis::values(SMALL_TOKENS),
        Axis::token_axis(),
        Axis::values(LARGE_TOKENS),
    ]);
    let cut = full
        .iter()
        .position(|&tokens| tokens >= f64::from(max_tokens_per_rank))
        .map_or(full.len(), |index| index + 1);
    full[..cut].to_vec()
}

impl KernelSpec for MegaMoeSpec {
    type Config = MegaMoeKernelConfig;
    type Input = MegaMoeKernelInput;

    const KIND: KernelKind = "mega_moe";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![config
            .expert_demand
            .token_axis(token_points(config.max_tokens_per_rank))])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        // ep_size 1: one GPU holds every expert (DeepGEMM's single-process mode).
        anyhow::ensure!(config.ep_size >= 1, "mega_moe needs ep_size >= 1");
        anyhow::ensure!(
            config.num_experts % config.ep_size == 0,
            "num_experts {} must divide by ep_size {}",
            config.num_experts,
            config.ep_size
        );
        anyhow::ensure!(
            config.top_k >= 1 && config.top_k <= config.num_experts,
            "top_k {} must be in 1..=num_experts",
            config.top_k
        );
        anyhow::ensure!(
            config.max_tokens_per_rank >= 1,
            "max_tokens_per_rank must be positive"
        );
        let slots = config
            .placement
            .num_slots(config.expert_demand.num_experts());
        anyhow::ensure!(
            slots == config.num_experts as usize,
            "{:?} over {} experts gives {slots} slots, config has {}",
            config.placement,
            config.expert_demand.num_experts(),
            config.num_experts
        );
        config.expert_demand.prepare().map(|_| ())
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        // The group's histogram at the group's token count; drawn for the
        // whole axis at once (see `ExpertDemand::per_slot_batches`).
        let global: Vec<u32> = grid.expand_1d(|num_tokens| num_tokens as u32 * config.ep_size);
        let mut slots = config
            .expert_demand
            .per_slot_batches(
                config.top_k,
                &global,
                config.placement,
                config.ep_size as usize,
            )
            .expect("validate_config proved this source readable")
            .into_iter();
        grid.expand_1d(|num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("ep_size", config.ep_size)
                .with("num_tokens", num_tokens as u32)
                .with("hidden_size", config.hidden_size.get())
                .with("intermediate_size", config.intermediate_size.get())
                .with("num_experts", config.num_experts)
                .with("num_shared_experts", config.num_shared_experts)
                .with("top_k", config.top_k)
                .with("activation_dtype", config.activation_dtype.as_str())
                .with("weight_dtype", config.weight_dtype.as_str())
                .with(
                    "per_slot_rows",
                    slots.next().expect("one histogram per grid point"),
                )
        })
    }
}

register_kernel!(MegaMoeKernel, MegaMoeSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn dsv41_flash_ep4(max_tokens_per_rank: u32) -> MegaMoeKernelConfig {
        MegaMoeKernelConfig {
            backends: vec!["deepgemm_sm100"],
            gpu_name: "NVIDIA B200".to_string(),
            ep_size: 4,
            hidden_size: 5120.into(),
            intermediate_size: 2304.into(),
            num_experts: 416,
            num_shared_experts: 1,
            top_k: 6,
            activation_dtype: DType::Mxfp8E4m3,
            weight_dtype: DType::Mxfp4E2m1,
            max_tokens_per_rank,
            expert_demand: ExpertDemand::popularity(
                &crate::timing::routing::RoutingDistribution::uniform(384),
                2,
            ),
            placement: SlotPlacement::Eplb { num_redundant: 32 },
        }
    }

    /// The capacity must not leak into the profiler key (the A/B found no
    /// effect), and every Python arg must be emitted under its own name.
    #[test]
    fn enumerate_emits_python_args_without_capacity() {
        let config = dsv41_flash_ep4(5760);
        let payloads = MegaMoeSpec::enumerate(
            &config,
            &SweepGrid::new(vec![Axis::values([140])]),
            "deepgemm_sm100",
        );
        let fields = payloads[0].fields();
        let mut names: Vec<&str> = fields.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "activation_dtype",
                "backend",
                "ep_size",
                "hidden_size",
                "intermediate_size",
                "num_experts",
                "num_shared_experts",
                "num_tokens",
                "per_slot_rows",
                "top_k",
                "weight_dtype",
            ]
        );
        let rows: Vec<u64> = fields["per_slot_rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(rows.len(), 416);
        assert_eq!(rows.iter().sum::<u64>(), 140 * 4 * 6);
        assert_eq!(fields["num_tokens"], Value::from(140_u32));
        assert_eq!(fields["weight_dtype"], Value::from("mxfp4_e2m1"));
    }

    /// A grid that stopped below the deployment's largest batch would
    /// extrapolate its most expensive calls.
    #[test]
    fn token_axis_covers_capacity_with_one_point_beyond() {
        let points = token_points(5760);
        assert_eq!(points.first(), Some(&1.0));
        assert_eq!(points.last(), Some(&5888.0));
        assert!(points.iter().filter(|&&p| p >= 5760.0).count() == 1);
        assert_eq!(token_points(65536).last(), Some(&65536.0));
        assert_eq!(token_points(1_000_000).last(), Some(&131072.0));
        assert!(token_points(1_000_000).len() <= 500);
    }

    #[test]
    fn config_rejects_slots_the_placement_does_not_make() {
        let mut config = dsv41_flash_ep4(4096);
        config.placement = SlotPlacement::Contiguous;
        assert!(MegaMoeSpec::validate_config(&config).is_err());
        config.num_experts = 384;
        assert!(MegaMoeSpec::validate_config(&config).is_ok());
    }

    #[test]
    fn config_rejects_experts_not_divisible_by_group() {
        let mut config = dsv41_flash_ep4(4096);
        config.num_experts = 414;
        assert!(MegaMoeSpec::validate_config(&config).is_err());
    }
}
