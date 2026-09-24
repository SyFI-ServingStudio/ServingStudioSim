//! DeepSeek fused MHC post/pre block with fused RMSNorm.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mhc_pre_rms_norm::{
    enumerate, MhcRmsNormKernelConfig, MhcRmsNormKernelInput,
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
pub(crate) fn fused_sweep_grid() -> SweepGrid {
    SweepGrid::new(vec![Axis::chain([
        Axis::pow2(0, 4),
        Axis::values([17]),
        Axis::token_axis(),
    ])])
}

pub struct MhcFusedPostPreRmsNormSpec;

impl KernelSpec for MhcFusedPostPreRmsNormSpec {
    type Config = MhcRmsNormKernelConfig;
    type Input = MhcRmsNormKernelInput;

    const KIND: KernelKind = "mhc_fused_post_pre_rms_norm";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        fused_sweep_grid()
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

    /// Catches losing the grid points that bracket the T<=16 / T>16 launch
    /// path switch, which would make the cache interpolate across the step.
    #[test]
    fn sweep_brackets_the_small_fma_path_switch() {
        let config = MhcRmsNormKernelConfig {
            backends: vec!["vllm_tilelang"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden_size: 4096.into(),
            hc_mult: 4,
            hidden_dtype: DType::Bf16,
        };
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
