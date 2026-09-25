//! Kimi-K3 fused attention-residual TMA launches during chunked prefill.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct K3AttnResPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_size: Dim,
    pub num_valid_blocks: u32,
    pub num_launches: u32,
    pub write_prefix: bool,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct K3AttnResPrefillKernelInput {
    pub num_tokens: u32,
}

pub struct K3AttnResPrefillSpec;

impl KernelSpec for K3AttnResPrefillSpec {
    type Config = K3AttnResPrefillKernelConfig;
    type Input = K3AttnResPrefillKernelInput;

    const KIND: KernelKind = "k3_attn_res_prefill";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::chain([Axis::pow2(0, 4), Axis::token_axis()])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
                .with("hidden_size", config.hidden_size.get())
                .with("num_valid_blocks", config.num_valid_blocks)
                .with("num_launches", config.num_launches)
                .with("write_prefix", config.write_prefix)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(K3AttnResPrefillKernel, K3AttnResPrefillSpec);
