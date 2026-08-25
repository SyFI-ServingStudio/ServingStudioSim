//! DeepSeek MHC pre block with fused RMSNorm.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MhcRmsNormKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_size: Dim,
    pub hc_mult: u32,
    #[compute_dtype]
    pub hidden_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MhcRmsNormKernelInput {
    pub num_tokens: u32,
}

pub(crate) fn sweep_grid() -> SweepGrid {
    SweepGrid::new(vec![Axis::chain([Axis::pow2(0, 4), Axis::token_axis()])])
}

pub(crate) fn enumerate(
    config: &MhcRmsNormKernelConfig,
    grid: &SweepGrid,
    backend: &'static str,
) -> Vec<ArgsPayload> {
    grid.expand_1d(|num_tokens| {
        ArgsPayload::new()
            .with("backend", backend)
            .with("num_tokens", num_tokens as u32)
            .with("hidden_size", config.hidden_size.get())
            .with("hc_mult", config.hc_mult)
            .with("hidden_dtype", config.hidden_dtype.as_str())
    })
}

pub struct MhcPreRmsNormSpec;

impl KernelSpec for MhcPreRmsNormSpec {
    type Config = MhcRmsNormKernelConfig;
    type Input = MhcRmsNormKernelInput;

    const KIND: KernelKind = "mhc_pre_rms_norm";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        sweep_grid()
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        enumerate(config, grid, backend)
    }
}

register_kernel!(MhcPreRmsNormKernel, MhcPreRmsNormSpec);

#[cfg(test)]
mod tests {
    use super::{MhcPreRmsNormSpec, MhcRmsNormKernelConfig};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn sweep_covers_small_and_large_public_dispatch_paths() {
        let config = MhcRmsNormKernelConfig {
            backends: vec!["vllm_tilelang"],
            gpu_name: "NVIDIA H200".to_string(),
            hidden_size: 7168.into(),
            hc_mult: 4,
            hidden_dtype: DType::Bf16,
        };
        let payloads = MhcPreRmsNormSpec::enumerate(
            &config,
            &MhcPreRmsNormSpec::sweep_grid(&config),
            "vllm_tilelang",
        );
        for num_tokens in [8, 128] {
            assert!(payloads
                .iter()
                .any(|payload| payload.fields()["num_tokens"] == num_tokens));
        }
    }
}
