//! DeepSeek V4 terminal MHC-post, MTP-buffer copy, hc-head, and RMSNorm.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mhc_pre_rms_norm::{
    enumerate, sweep_grid, MhcRmsNormKernelConfig, MhcRmsNormKernelInput,
};
use crate::timing::sweep::SweepGrid;

pub struct DeepseekV4TerminalMhcHeadSpec;

impl KernelSpec for DeepseekV4TerminalMhcHeadSpec {
    type Config = MhcRmsNormKernelConfig;
    type Input = MhcRmsNormKernelInput;

    const KIND: KernelKind = "deepseek_v4_terminal_mhc_head";

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

register_kernel!(
    DeepseekV4TerminalMhcHeadKernel,
    DeepseekV4TerminalMhcHeadSpec
);
