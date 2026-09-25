//! DeepSeek fused MHC post/pre block with fused RMSNorm.
//!
//! Backends share the args and the per-token work (see the Python kind):
//! `vllm_tilelang` is DeepSeek V4's TileLang call, and `deepgemm_mega` is
//! DeepSeek-V4.1's one-launch DeepGEMM `mega_mhc` with the shifted collapse.
//!
//! `deepgemm_mega` picks its K-split count on the host from `num_tokens`. On
//! B200 at hidden 5120 it measured 40 splits up to T=192, 27 up to T=320, 20 up
//! to T=448 and 16 above, and the time steps at each change. The shared token
//! axis has no point between 128 and 256 or between 384 and 512, so linear
//! interpolation would smear each step over a whole cell. Configs that name
//! this backend therefore add points just below, at and above each switch.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mhc_pre_rms_norm::{
    enumerate, sweep_grid, MhcRmsNormKernelConfig, MhcRmsNormKernelInput,
};
use crate::timing::sweep::{Axis, SweepGrid};

const DEEPGEMM_MEGA: &str = "deepgemm_mega";

/// Last token count of each `mega_mhc` K-split specialization (40/27/20) and
/// one probe step (8) on either side, plus 256 inside the 27-split range.
const DEEPGEMM_MEGA_SPLIT_POINTS: [u32; 10] = [184, 192, 200, 256, 312, 320, 328, 440, 448, 456];

/// The shared MHC token grid, plus the K-split points when `deepgemm_mega` is
/// a candidate. Only that backend's configs get the extra rows; `vllm_tilelang`
/// runs on H200 only and never shares a config with it.
fn fused_sweep_grid(config: &MhcRmsNormKernelConfig) -> SweepGrid {
    if !config.backends.contains(&DEEPGEMM_MEGA) {
        return sweep_grid();
    }
    let base = sweep_grid();
    let mut axis = Axis::chain([
        base.axes()[0].clone(),
        Axis::values(DEEPGEMM_MEGA_SPLIT_POINTS),
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
    use crate::timing::kernels::mhc_pre_rms_norm::{sweep_grid, MhcRmsNormKernelConfig};

    fn config(backend: &'static str, gpu: &str, hidden: u32) -> MhcRmsNormKernelConfig {
        MhcRmsNormKernelConfig {
            backends: vec![backend],
            gpu_name: gpu.to_string(),
            hidden_size: hidden.into(),
            hc_mult: 4,
            hidden_dtype: DType::Bf16,
        }
    }

    /// Catches a grid that interpolates across a `mega_mhc` K-split step (the
    /// cache would smear the 192->193 jump over 128..256) or that is unsorted.
    #[test]
    fn deepgemm_mega_grid_brackets_each_k_split_switch() {
        let cfg = config("deepgemm_mega", "NVIDIA B200", 5120);
        let grid = MhcFusedPostPreRmsNormSpec::sweep_grid(&cfg);
        let axis = &grid.axes()[0];
        assert!(
            axis.windows(2).all(|w| w[0] < w[1]),
            "axis must be strictly increasing"
        );
        for t in [
            184.0, 192.0, 200.0, 256.0, 312.0, 320.0, 328.0, 440.0, 448.0, 456.0,
        ] {
            assert!(axis.contains(&t), "missing K-split point {t}");
        }
        assert_eq!(axis.len(), 77);
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

    /// Catches the K-split points leaking into DeepSeek V4's TileLang config,
    /// whose profiled rows would then read as missing.
    #[test]
    fn vllm_tilelang_keeps_the_shared_grid() {
        let cfg = config("vllm_tilelang", "NVIDIA H200", 4096);
        assert_eq!(MhcFusedPostPreRmsNormSpec::sweep_grid(&cfg), sweep_grid());
    }
}
