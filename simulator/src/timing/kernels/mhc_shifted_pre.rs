//! DeepInfer's unfused shifted mHC pre (`_hc_pre` with `hc_shifted`, before
//! the layer norm): apply-mix, DeepGEMM `tf32_hc_prenorm_gemm` (fixed 16
//! K-splits) + reduce, split-mixes and Sinkhorn, five launches. Layer 0 and the
//! Engram layers' attention entries.
//!
//! Same args and token grid as the other mHC kinds.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mhc_pre_rms_norm::{
    enumerate, sweep_grid, MhcRmsNormKernelConfig, MhcRmsNormKernelInput,
};
use crate::timing::sweep::SweepGrid;

pub struct MhcShiftedPreSpec;

impl KernelSpec for MhcShiftedPreSpec {
    type Config = MhcRmsNormKernelConfig;
    type Input = MhcRmsNormKernelInput;

    const KIND: KernelKind = "mhc_shifted_pre";

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

register_kernel!(MhcShiftedPreKernel, MhcShiftedPreSpec);
