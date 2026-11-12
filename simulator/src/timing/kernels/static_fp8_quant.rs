//! Static per-tensor FP8 quantization of an activation, as Inductor compiles
//! vLLM's native `QuantFP8`. An attention output gate (`x * sigmoid(g)`) in
//! front of the projection fuses into the same kernel and adds a full read of
//! the gate rows, so it is a config field.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StaticFp8QuantKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    /// Each row is multiplied by the sigmoid of a same-width gate row first.
    pub gated: bool,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct StaticFp8QuantKernelInput {
    pub num_tokens: u32,
}

pub struct StaticFp8QuantSpec;

impl KernelSpec for StaticFp8QuantSpec {
    type Config = StaticFp8QuantKernelConfig;
    type Input = StaticFp8QuantKernelInput;

    const KIND: KernelKind = "static_fp8_quant";

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
                .with("gated", config.gated)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(StaticFp8QuantKernel, StaticFp8QuantSpec);
