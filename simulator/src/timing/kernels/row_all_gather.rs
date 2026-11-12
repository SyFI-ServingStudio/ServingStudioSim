//! Ragged all-gather of bf16 rows: every rank of a group contributes its rows
//! and receives every rank's rows (all-gatherv semantics). Used where a group
//! gathers activations ahead of a sharded op, e.g. an attention-TP group
//! gathering its members' inputs.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::moe_ep_all_gather::{
    canonical_tokens, collective_grid, MoeEpCollectiveKernelInput,
};
use crate::timing::sweep::SweepGrid;
use crate::timing::{Dim, KernelConfig};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RowAllGatherKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_gpus: u32,
    pub hidden_size: Dim,
    #[compute_dtype]
    pub dtype: DType,
    pub fabric: String,
    /// Largest row count summed over the group; bounds the total-row axis.
    pub max_total_tokens: u32,
}

pub struct RowAllGatherSpec;

impl KernelSpec for RowAllGatherSpec {
    type Config = RowAllGatherKernelConfig;
    type Input = MoeEpCollectiveKernelInput;

    const KIND: KernelKind = "row_all_gather";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        collective_grid(config.max_total_tokens)
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_2d(|total, ragged_max| {
            canonical_tokens(config.num_gpus, total as u32, ragged_max as u32).is_none()
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        assert!((2..=8).contains(&config.num_gpus));
        assert_eq!(config.dtype, DType::Bf16);
        assert_eq!(config.fabric, "nvlink");
        grid.expand_2d(|total, ragged_max| {
            let per_rank_tokens =
                canonical_tokens(config.num_gpus, total as u32, ragged_max as u32)
                    .unwrap_or_else(|| vec![total as u32; config.num_gpus as usize]);
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_gpus", config.num_gpus)
                .with("per_rank_tokens", per_rank_tokens)
                .with("hidden_size", config.hidden_size.get())
                .with("dtype", config.dtype.as_str())
                .with("fabric", config.fabric.clone())
        })
    }
}

register_kernel!(RowAllGatherKernel, RowAllGatherSpec);
