//! Kimi-K3 chunked-prefix MLA output-state merge.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MlaMergeStateKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub value_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MlaMergeStateKernelInput {
    pub num_tokens: u32,
}

pub struct MlaMergeStateSpec;

impl KernelSpec for MlaMergeStateSpec {
    type Config = MlaMergeStateKernelConfig;
    type Input = MlaMergeStateKernelInput;

    const KIND: KernelKind = "mla_merge_state";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::values([128, 1024, 4096, 16_384])])
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
                .with("num_tokens", num_tokens.round() as u32)
                .with("num_heads", config.num_heads.get())
                .with("value_dim", config.value_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(MlaMergeStateKernel, MlaMergeStateSpec);
