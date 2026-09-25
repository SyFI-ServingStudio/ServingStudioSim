//! Kimi-K3 chunked-prefix MLA index creation and latent-KV gather.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MlaPrefixGatherKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub kv_lora_rank: Dim,
    pub rope_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
    #[kv_dtype]
    pub cache_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MlaPrefixGatherKernelInput {
    pub batch_size: u32,
    pub num_tokens: u32,
}

pub struct MlaPrefixGatherSpec;

impl KernelSpec for MlaPrefixGatherSpec {
    type Config = MlaPrefixGatherKernelConfig;
    type Input = MlaPrefixGatherKernelInput;

    const KIND: KernelKind = "mla_prefix_gather";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values([1, 2, 4, 8]),
            Axis::values([128, 1024, 4096, 16_384, 49_152]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(crate::timing::cache::Extrapolation::Product)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|batch_size, num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", batch_size.round() as u32)
                .with("num_tokens", num_tokens.round() as u32)
                .with("kv_lora_rank", config.kv_lora_rank.get())
                .with("rope_dim", config.rope_dim.get())
                .with("dtype", config.dtype.as_str())
                .with("cache_dtype", config.cache_dtype.as_str())
        })
    }
}

register_kernel!(MlaPrefixGatherKernel, MlaPrefixGatherSpec);
