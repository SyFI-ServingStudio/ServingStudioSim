//! Fused mHC post/pre block with fused RMSNorm.
//!
//! Backends share the args and the per-token work (see the Python kind):
//! `vllm_tilelang` is vLLM's TileLang call, and `deepgemm_mega` is the
//! one-launch DeepGEMM `mega_mhc` with the shifted collapse.
//!
//! `deepgemm_mega` picks its K-split count on the host from `num_tokens`. On
//! B200 at hidden 5120 it measured 40 splits up to T=192, 27 up to T=320, 20 up
//! to T=448 and 16 above, and the time steps at each change. The shared token
//! axis has no point between 128 and 256 or between 384 and 512, so linear
//! interpolation would smear each step over a whole cell. Configs that name
//! this backend therefore add the last T before each switch, the first T after
//! it, and one probe step (8) on either side.
//!
//! Past T=448 the 16-split launch also steps by ~10 us each time its work
//! spills into another wave: every 64-token block runs 16 splits on B200's 148
//! SMs, so wave k ends at T = 64 * floor(148 * k / 16) (576, 1152, 1728, ...).
//! Up to T=4096, where one extra wave still adds 10-50%, the grid holds the
//! last T of each wave and T+1. Above that a step is at most ~12% and the
//! 256-token spacing keeps the error within a few percent.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mhc_pre_rms_norm::{
    enumerate, sweep_grid, MhcRmsNormKernelConfig, MhcRmsNormKernelInput,
};
use crate::timing::sweep::{Axis, SweepGrid};

/// Grid for the fused boundary: the shared mHC token grid plus T=17.
///
/// vLLM's `mhc_fused_post_pre_tilelang` switches launch path at
/// `num_tokens <= 16` (2 launches: small-FMA fused + big_fuse) versus
/// T > 16 (3 launches: post + tf32 prenorm GEMM + big_fuse). The switch is a
/// ~+2.2 us step (B200: 10.25 us at T=16, 12.46 us at T=17). Without T=17 the
/// 1D linear cache interpolates 17..31 across the step and under-predicts
/// T=17 by ~17%. T=16 and T=17 bracket the step exactly.
fn tilelang_sweep_grid() -> SweepGrid {
    SweepGrid::new(vec![Axis::chain([
        Axis::pow2(0, 4),
        Axis::values([17]),
        Axis::token_axis(),
    ])])
}

const DEEPGEMM_MEGA: &str = "deepgemm_mega";

/// Last token count of each `mega_mhc` K-split specialization (40/27/20), the
/// first token count of the next one, and one probe step (8) on either side,
/// plus 256 inside the 27-split range. The time jumps between T and T+1.
const DEEPGEMM_MEGA_SPLIT_POINTS: [u32; 13] = [
    184, 192, 193, 200, 256, 312, 320, 321, 328, 440, 448, 449, 456,
];

/// Last token count of each 16-split wave up to T=4096 and the first of the
/// next wave; the time jumps between the two.
const DEEPGEMM_MEGA_WAVE_POINTS: [u32; 14] = [
    576, 577, 1152, 1153, 1728, 1729, 2368, 2369, 2944, 2945, 3520, 3521, 4096, 4097,
];

/// The shared MHC token grid, plus the K-split and wave points when
/// `deepgemm_mega` is a candidate. Only that backend's configs get the extra
/// rows. Other backends use `tilelang_sweep_grid`. The `deepgemm_mega` rows were
/// measured on the shared mHC grid, without the TileLang T=17 point.
fn fused_sweep_grid(config: &MhcRmsNormKernelConfig) -> SweepGrid {
    if !config.backends.contains(&DEEPGEMM_MEGA) {
        return tilelang_sweep_grid();
    }
    let base = sweep_grid();
    let mut axis = Axis::chain([
        base.axes()[0].clone(),
        Axis::values(DEEPGEMM_MEGA_SPLIT_POINTS),
        Axis::values(DEEPGEMM_MEGA_WAVE_POINTS),
    ]);
    axis.sort_by(f64::total_cmp);
    SweepGrid::new(vec![axis])
}

pub struct MhcFusedPostPreRmsNormSpec;

impl KernelSpec for MhcFusedPostPreRmsNormSpec {
    type Config = MhcRmsNormKernelConfig;
    type Input = MhcRmsNormKernelInput;

    const KIND: KernelKind = "mhc_fused_post_pre_rms_norm";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        fused_sweep_grid(config)
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

register_kernel!(MhcFusedPostPreRmsNormKernel, MhcFusedPostPreRmsNormSpec);

#[cfg(test)]
mod tests {
    use super::MhcFusedPostPreRmsNormSpec;
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;
    use crate::timing::kernels::mhc_pre_rms_norm::MhcRmsNormKernelConfig;

    fn config(backend: &'static str, gpu: &str, hidden: u32) -> MhcRmsNormKernelConfig {
        MhcRmsNormKernelConfig {
            backends: vec![backend],
            gpu_name: gpu.to_string(),
            hidden_size: hidden.into(),
            hc_mult: 4,
            hidden_dtype: DType::Bf16,
        }
    }

    /// Catches a grid that interpolates across a `mega_mhc` K-split or wave step
    /// (the cache would smear the 192->193 jump over 128..256, or the 576->577
    /// wave jump over 512..640) or that is unsorted.
    #[test]
    fn deepgemm_mega_grid_brackets_each_k_split_and_wave_step() {
        let cfg = config("deepgemm_mega", "NVIDIA B200", 5120);
        let grid = MhcFusedPostPreRmsNormSpec::sweep_grid(&cfg);
        let axis = &grid.axes()[0];
        assert!(
            axis.windows(2).all(|w| w[0] < w[1]),
            "axis must be strictly increasing"
        );
        for t in [
            184.0, 192.0, 193.0, 200.0, 256.0, 312.0, 320.0, 321.0, 328.0, 440.0, 448.0, 449.0,
            456.0, 576.0, 577.0, 1728.0, 1729.0, 2368.0, 2369.0, 3520.0, 3521.0, 4097.0,
        ] {
            assert!(axis.contains(&t), "missing K-split or wave point {t}");
        }
        assert_eq!(axis.len(), 91);
        let payloads = MhcFusedPostPreRmsNormSpec::enumerate(&cfg, &grid, "deepgemm_mega");
        let p = &payloads[axis.iter().position(|&t| t == 192.0).unwrap()];
        let mut keys: Vec<_> = p.fields().keys().map(String::as_str).collect();
        keys.sort();
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
        assert_eq!(p.fields()["num_tokens"], 192);
        assert_eq!(p.fields()["hidden_size"], 5120);
    }

    /// Catches the K-split points leaking into the TileLang config, whose
    /// profiled rows would then read as missing.
    #[test]
    fn vllm_tilelang_keeps_its_own_grid() {
        let cfg = config("vllm_tilelang", "NVIDIA H200", 4096);
        assert_eq!(
            MhcFusedPostPreRmsNormSpec::sweep_grid(&cfg),
            super::tilelang_sweep_grid()
        );
    }

    /// Catches losing the grid points that bracket the T<=16 / T>16 launch
    /// path switch, which would make the cache interpolate across the step.
    #[test]
    fn sweep_brackets_the_small_fma_path_switch() {
        let config = config("vllm_tilelang", "NVIDIA B200", 4096);
        let payloads = MhcFusedPostPreRmsNormSpec::enumerate(
            &config,
            &MhcFusedPostPreRmsNormSpec::sweep_grid(&config),
            "vllm_tilelang",
        );
        let tokens: Vec<u64> = payloads
            .iter()
            .map(|p| p.fields()["num_tokens"].as_u64().unwrap())
            .collect();
        let i16 = tokens.iter().position(|&t| t == 16).unwrap();
        assert_eq!(tokens[i16 + 1], 17);
        assert_eq!(tokens[i16 + 2], 32);
        assert!(tokens.windows(2).all(|w| w[0] < w[1]));
    }
}
