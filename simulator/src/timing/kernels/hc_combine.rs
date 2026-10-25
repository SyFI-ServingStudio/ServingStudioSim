//! Standalone delayed hyper-connection combine: the vLLM fork's (892da082)
//! `hc_combine` Triton op, one program per (token, 512-feature tile).
//!
//! The schema is the multi-stream residual one the mHC kinds use (`hidden_size`,
//! `hc_mult`, `hidden_dtype`, swept over `num_tokens`), so the Config, Input,
//! grid and payload are shared with them.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mhc_pre_rms_norm::{
    enumerate, sweep_grid, MhcRmsNormKernelConfig, MhcRmsNormKernelInput,
};
use crate::timing::kernels::neuron_grid;
use crate::timing::sweep::SweepGrid;

pub struct HcCombineSpec;

impl KernelSpec for HcCombineSpec {
    type Config = MhcRmsNormKernelConfig;
    type Input = MhcRmsNormKernelInput;

    const KIND: KernelKind = "hc_combine";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        if neuron_grid::is_neuron(&config.backends) {
            return neuron_grid::grid_1d(neuron_grid::TOKENS);
        }
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

register_kernel!(HcCombineKernel, HcCombineSpec);

#[cfg(test)]
mod tests {
    use super::HcCombineSpec;
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;
    use crate::timing::kernels::mhc_pre_rms_norm::{sweep_grid, MhcRmsNormKernelConfig};
    use crate::timing::kernels::neuron_grid;

    /// Catches the Neuron branch leaking into a GPU or mixed config (which would
    /// resample every GPU row) or a Neuron config missing the contract grid.
    #[test]
    fn neuron_backends_take_the_tokens_grid_and_gpu_grid_is_unchanged() {
        let grid = |backends: Vec<&'static str>| {
            HcCombineSpec::sweep_grid(&MhcRmsNormKernelConfig {
                backends,
                gpu_name: "AWS Trainium2 LNC2".to_string(),
                hidden_size: 2560.into(),
                hc_mult: 4,
                hidden_dtype: DType::Bf16,
            })
            .axes()
            .to_vec()
        };
        assert_eq!(
            grid(vec!["neuron_torch"]),
            [neuron_grid::axis(neuron_grid::TOKENS)]
        );
        assert_eq!(grid(vec!["vllm_triton"]), sweep_grid().axes());
        assert_eq!(
            grid(vec!["neuron_torch", "vllm_triton"]),
            sweep_grid().axes()
        );
        let cfg = MhcRmsNormKernelConfig {
            backends: vec!["neuron_torch"],
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            hidden_size: 2560.into(),
            hc_mult: 4,
            hidden_dtype: DType::Bf16,
        };
        let payloads =
            HcCombineSpec::enumerate(&cfg, &HcCombineSpec::sweep_grid(&cfg), "neuron_torch");
        assert_eq!(payloads.len(), 14);
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "hc_mult",
                "hidden_dtype",
                "hidden_size",
                "num_tokens"
            ]
        );
    }
}
