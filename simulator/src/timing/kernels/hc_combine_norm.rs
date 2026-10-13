//! Delayed hyper-connection combine fused with the next HC module's per-stream
//! Gemma RMSNorm: the fork's `hc_combine_norm` Triton op, one program per
//! (token, stream).
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

pub struct HcCombineNormSpec;

impl KernelSpec for HcCombineNormSpec {
    type Config = MhcRmsNormKernelConfig;
    type Input = MhcRmsNormKernelInput;

    const KIND: KernelKind = "hc_combine_norm";

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

register_kernel!(HcCombineNormKernel, HcCombineNormSpec);
