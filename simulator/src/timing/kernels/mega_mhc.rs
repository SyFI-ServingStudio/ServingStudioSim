//! DeepGEMM's fused mHC boundary with the FP8 cast (`deep_gemm.mega_mhc` as
//! DeepInfer's `HCLayerMixin._mega_mhc` calls it).
//!
//! One `sm100_mega_mhc_impl` launch per sub-layer entry and GPU: the previous
//! sub-layer's post, this sub-layer's prenorm GEMM, gates and Sinkhorn, the
//! shifted collapse, the RMSNorm and the FP8 E4M3 cast. `sf_layout` names the
//! FP8 consumer: `gemm` (attention entry) or `mega_moe` (FFN entry, which also
//! writes the shared expert's copy of the scales).
//!
//! The query is the GPU's row count. DeepGEMM picks the K-split count from it
//! with the same host heuristic as the vLLM fork's `mega_mhc` (40 / 27 / 20 /
//! 16 splits at hidden 5120 on 148 SMs), and the 16-split launch steps at each
//! extra wave, so the grid carries `deepgemm_mega`'s split and wave points.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mhc_fused_post_pre_rms_norm::{
    DEEPGEMM_MEGA_SPLIT_POINTS, DEEPGEMM_MEGA_WAVE_POINTS,
};
use crate::timing::kernels::mhc_pre_rms_norm::sweep_grid as mhc_token_grid;
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// FP8 consumer of an attention entry: the MXFP8 input projections.
pub const SF_LAYOUT_GEMM: &str = "gemm";
/// FP8 consumer of an FFN entry: the mega MoE symmetric buffer.
pub const SF_LAYOUT_MEGA_MOE: &str = "mega_moe";

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MegaMhcKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_size: Dim,
    pub hc_mult: u32,
    #[compute_dtype]
    pub hidden_dtype: DType,
    /// [`SF_LAYOUT_GEMM`] or [`SF_LAYOUT_MEGA_MOE`].
    pub sf_layout: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MegaMhcKernelInput {
    pub num_tokens: u32,
}

pub struct MegaMhcSpec;

impl KernelSpec for MegaMhcSpec {
    type Config = MegaMhcKernelConfig;
    type Input = MegaMhcKernelInput;

    const KIND: KernelKind = "mega_mhc";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        let mut axis = Axis::chain([
            mhc_token_grid().axes()[0].clone(),
            Axis::values(DEEPGEMM_MEGA_SPLIT_POINTS),
            Axis::values(DEEPGEMM_MEGA_WAVE_POINTS),
        ]);
        axis.sort_by(f64::total_cmp);
        axis.dedup();
        SweepGrid::new(vec![axis])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.sf_layout == SF_LAYOUT_GEMM || config.sf_layout == SF_LAYOUT_MEGA_MOE,
            "sf_layout must be {SF_LAYOUT_GEMM:?} or {SF_LAYOUT_MEGA_MOE:?}, got {:?}",
            config.sf_layout
        );
        Ok(())
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
                .with("hc_mult", config.hc_mult)
                .with("hidden_dtype", config.hidden_dtype.as_str())
                .with("sf_layout", config.sf_layout.as_str())
        })
    }
}

register_kernel!(MegaMhcKernel, MegaMhcSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(sf_layout: &str) -> MegaMhcKernelConfig {
        MegaMhcKernelConfig {
            backends: vec!["deepgemm_sm100"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden_size: 5120.into(),
            hc_mult: 4,
            hidden_dtype: DType::Bf16,
            sf_layout: sf_layout.to_string(),
        }
    }

    /// Catches a payload that drops `sf_layout` (both entries would read one
    /// table row) or misnames a Python arg.
    #[test]
    fn enumerate_emits_every_python_arg() {
        let cfg = config(SF_LAYOUT_MEGA_MOE);
        let grid = MegaMhcSpec::sweep_grid(&cfg);
        let payloads = MegaMhcSpec::enumerate(&cfg, &grid, "deepgemm_sm100");
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "hc_mult",
                "hidden_dtype",
                "hidden_size",
                "num_tokens",
                "sf_layout"
            ]
        );
        assert_eq!(payloads[0].fields()["sf_layout"], "mega_moe");
        assert!(MegaMhcSpec::validate_config(&cfg).is_ok());
        assert!(MegaMhcSpec::validate_config(&config("bf16")).is_err());
    }

    /// Catches a grid that interpolates across a K-split or wave step (the
    /// 192->193 split jump over 128..256, the 576->577 wave jump over
    /// 512..640) or that is unsorted.
    #[test]
    fn grid_brackets_k_split_and_wave_steps() {
        let axis = MegaMhcSpec::sweep_grid(&config(SF_LAYOUT_GEMM)).axes()[0].clone();
        assert!(axis.windows(2).all(|w| w[0] < w[1]));
        for t in [
            1.0, 192.0, 193.0, 448.0, 449.0, 576.0, 577.0, 4096.0, 4097.0, 65536.0,
        ] {
            assert!(axis.contains(&t), "missing grid point {t}");
        }
    }
}
