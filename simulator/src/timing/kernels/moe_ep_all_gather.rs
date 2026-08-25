//! Ragged hidden-state plus router-logit all-gatherv for naive DP/EP MoE.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct MoeEpCollectiveKernelInput {
    pub per_rank_tokens: Vec<u32>,
}

impl SweepCoords for MoeEpCollectiveKernelInput {
    fn coords(&self) -> Coords {
        let total = self.per_rank_tokens.iter().copied().sum::<u32>();
        let maximum = self.per_rank_tokens.iter().copied().max().unwrap_or(0);
        Coords::new([f64::from(total), f64::from(maximum)])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["total_tokens", "max_rank_tokens"]
    }
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeEpAllGatherKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_gpus: u32,
    pub hidden_size: Dim,
    pub num_experts: Dim,
    #[compute_dtype]
    pub hidden_dtype: DType,
    pub router_dtype: DType,
    pub fabric: String,
}

pub struct MoeEpAllGatherSpec;

impl KernelSpec for MoeEpAllGatherSpec {
    type Config = MoeEpAllGatherKernelConfig;
    type Input = MoeEpCollectiveKernelInput;

    const KIND: KernelKind = "moe_ep_all_gather";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        let axis = Axis::values([1, 4, 16, 64, 256, 1024, 4096, 8192]);
        SweepGrid::new(vec![axis.clone(), axis])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_2d(|total, maximum| {
            canonical_tokens(config.num_gpus, total as u32, maximum as u32).is_none()
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        validate_config(config);
        grid.expand_2d(|total, maximum| {
            let per_rank_tokens = canonical_tokens(config.num_gpus, total as u32, maximum as u32)
                .unwrap_or_else(|| vec![total as u32; config.num_gpus as usize]);
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_gpus", config.num_gpus)
                .with("per_rank_tokens", per_rank_tokens)
                .with("hidden_size", config.hidden_size.get())
                .with("num_experts", config.num_experts.get())
                .with("hidden_dtype", config.hidden_dtype.as_str())
                .with("router_dtype", config.router_dtype.as_str())
                .with("fabric", config.fabric.clone())
        })
    }
}

pub(crate) fn canonical_tokens(
    num_gpus: u32,
    total_tokens: u32,
    max_rank_tokens: u32,
) -> Option<Vec<u32>> {
    if num_gpus < 2
        || total_tokens == 0
        || max_rank_tokens == 0
        || max_rank_tokens > total_tokens
        || total_tokens > max_rank_tokens.checked_mul(num_gpus)?
    {
        return None;
    }
    let mut tokens = vec![0; num_gpus as usize];
    tokens[0] = max_rank_tokens;
    let mut remaining = total_tokens - max_rank_tokens;
    for rank in 1..num_gpus as usize {
        let ranks_left = num_gpus as usize - rank;
        let count = remaining.div_ceil(ranks_left as u32).min(max_rank_tokens);
        tokens[rank] = count;
        remaining -= count;
    }
    (remaining == 0).then_some(tokens)
}

fn validate_config(config: &MoeEpAllGatherKernelConfig) {
    assert!(matches!(config.num_gpus, 2 | 4 | 8));
    assert_eq!(config.hidden_dtype, DType::Bf16);
    assert_eq!(config.router_dtype, DType::Fp32);
    assert_eq!(config.fabric, "nvlink");
}

register_kernel!(MoeEpAllGatherKernel, MoeEpAllGatherSpec);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ragged_topology_is_exact_while_cache_coordinates_capture_load_and_imbalance() {
        let input = MoeEpCollectiveKernelInput {
            per_rank_tokens: vec![128, 96, 0, 32],
        };
        assert_eq!(&*input.coords(), &[256.0, 128.0]);
        assert_eq!(canonical_tokens(4, 256, 128), Some(vec![128, 43, 43, 42]));
        assert_eq!(canonical_tokens(4, 256, 32), None);
    }
}
