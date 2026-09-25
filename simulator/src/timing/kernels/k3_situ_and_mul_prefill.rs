//! Kimi-K3's eager FP32-input SiTU-and-multiply activation.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct K3SituAndMulPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_size: Dim,
    #[compute_dtype]
    pub input_dtype: DType,
    pub output_dtype: DType,
    pub beta: u32,
    pub linear_beta: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct K3SituAndMulPrefillKernelInput {
    pub num_tokens: u32,
}

pub struct K3SituAndMulPrefillSpec;

impl KernelSpec for K3SituAndMulPrefillSpec {
    type Config = K3SituAndMulPrefillKernelConfig;
    type Input = K3SituAndMulPrefillKernelInput;

    const KIND: KernelKind = "k3_situ_and_mul_prefill";

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
                .with("input_dtype", config.input_dtype.as_str())
                .with("output_dtype", config.output_dtype.as_str())
                .with("beta", f64::from(config.beta))
                .with("linear_beta", f64::from(config.linear_beta))
        })
    }
}

register_kernel!(K3SituAndMulPrefillKernel, K3SituAndMulPrefillSpec);
