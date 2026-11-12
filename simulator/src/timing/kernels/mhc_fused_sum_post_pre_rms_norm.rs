//! Sum of partial block outputs, then the fused mHC post/pre block with
//! RMSNorm (see the Python kind): the boundary after an expert-parallel MoE
//! layer whose combine leaves `num_partials` partial sums in the token owner's
//! memory, with the shared expert's output added when `shared_addend` is set.
//!
//! The query is the token count, as for `mhc_fused_post_pre_rms_norm`; the
//! partial count and the addend are static identity.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mhc_pre_rms_norm::{sweep_grid, MhcRmsNormKernelInput};
use crate::timing::sweep::SweepGrid;
use crate::timing::{Dim, KernelConfig};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MhcFusedSumKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_size: Dim,
    pub hc_mult: u32,
    #[compute_dtype]
    pub hidden_dtype: DType,
    pub num_partials: u32,
    pub shared_addend: bool,
}

pub struct MhcFusedSumPostPreRmsNormSpec;

impl KernelSpec for MhcFusedSumPostPreRmsNormSpec {
    type Config = MhcFusedSumKernelConfig;
    type Input = MhcRmsNormKernelInput;

    const KIND: KernelKind = "mhc_fused_sum_post_pre_rms_norm";

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
        grid.expand_1d(|num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
                .with("hidden_size", config.hidden_size.get())
                .with("hc_mult", config.hc_mult)
                .with("hidden_dtype", config.hidden_dtype.as_str())
                .with("num_partials", config.num_partials)
                .with("shared_addend", config.shared_addend)
        })
    }
}

register_kernel!(MhcFusedSumPostPreRmsNormKernel, MhcFusedSumPostPreRmsNormSpec);

#[cfg(test)]
mod tests {
    use super::{MhcFusedSumKernelConfig, MhcFusedSumPostPreRmsNormSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    /// Catches a payload that drops or renames the partial count or the addend
    /// flag (the Python args would reject it, or read another row).
    #[test]
    fn payload_carries_partials_and_addend() {
        let cfg = MhcFusedSumKernelConfig {
            backends: vec!["cuda_one_launch"],
            gpu_name: "NVIDIA B200".into(),
            hidden_size: 4096.into(),
            hc_mult: 4,
            hidden_dtype: DType::Bf16,
            num_partials: 8,
            shared_addend: true,
        };
        let grid = MhcFusedSumPostPreRmsNormSpec::sweep_grid(&cfg);
        let payloads = MhcFusedSumPostPreRmsNormSpec::enumerate(&cfg, &grid, "cuda_one_launch");
        let p = &payloads[0];
        let mut keys: Vec<_> = p.fields().keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "backend",
                "hc_mult",
                "hidden_dtype",
                "hidden_size",
                "num_partials",
                "num_tokens",
                "shared_addend"
            ]
        );
        assert_eq!(p.fields()["num_partials"], 8);
        assert_eq!(p.fields()["shared_addend"], true);
    }
}
