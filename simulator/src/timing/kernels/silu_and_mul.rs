//! SiLU-and-mul activation of a dense gated MLP, as Inductor compiles vLLM's
//! native `SiluAndMul` into one Triton pointwise kernel.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SiluAndMulKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Width of the gate half and of the up half (the output width).
    pub hidden_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct SiluAndMulKernelInput {
    pub num_tokens: u32,
}

pub struct SiluAndMulSpec;

impl KernelSpec for SiluAndMulSpec {
    type Config = SiluAndMulKernelConfig;
    type Input = SiluAndMulKernelInput;

    const KIND: KernelKind = "silu_and_mul";

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
                .with("hidden_dim", config.hidden_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(SiluAndMulKernel, SiluAndMulSpec);
