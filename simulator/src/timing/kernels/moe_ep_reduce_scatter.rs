//! Ragged reduce-scatterv that combines naive DP/EP MoE outputs.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::moe_ep_all_gather::{canonical_tokens, MoeEpCollectiveKernelInput};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeEpReduceScatterKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_gpus: u32,
    pub hidden_size: Dim,
    #[compute_dtype]
    pub dtype: DType,
    pub fabric: String,
}

pub struct MoeEpReduceScatterSpec;

impl KernelSpec for MoeEpReduceScatterSpec {
    type Config = MoeEpReduceScatterKernelConfig;
    type Input = MoeEpCollectiveKernelInput;

    const KIND: KernelKind = "moe_ep_reduce_scatter";

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
        assert!(matches!(config.num_gpus, 2 | 4 | 8));
        assert_eq!(config.dtype, DType::Bf16);
        assert_eq!(config.fabric, "nvlink");
        grid.expand_2d(|total, maximum| {
            let per_rank_tokens = canonical_tokens(config.num_gpus, total as u32, maximum as u32)
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

register_kernel!(MoeEpReduceScatterKernel, MoeEpReduceScatterSpec);
