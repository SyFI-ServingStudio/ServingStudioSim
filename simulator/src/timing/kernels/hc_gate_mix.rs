//! Hyper-connection block input, the gated mean over the normalized streams: the
//! fork's `hc_gate_mix` Triton op, one program per (token, 512-feature tile).
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
use crate::timing::sweep::SweepGrid;

pub struct HcGateMixSpec;

impl KernelSpec for HcGateMixSpec {
    type Config = MhcRmsNormKernelConfig;
    type Input = MhcRmsNormKernelInput;

    const KIND: KernelKind = "hc_gate_mix";

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

register_kernel!(HcGateMixKernel, HcGateMixSpec);
