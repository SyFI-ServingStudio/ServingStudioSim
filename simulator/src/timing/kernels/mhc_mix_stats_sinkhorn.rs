//! mHC mixing statistics with the Sinkhorn-normalized comb (SGLang
//! `hc_mix_stats_sinkhorn`: `_hc_mix_stats_partial` + `_hc_mix_reduce_sinkhorn`).

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mhc_pre_rms_norm::{sweep_grid, MhcRmsNormKernelInput};
use crate::timing::sweep::SweepGrid;
use crate::timing::{Dim, KernelConfig};

/// `mhc_pre_rms_norm`'s identity plus the Sinkhorn round count, which is a
/// model-config constant (`hc_sinkhorn_iters`) and sets the reduce kernel's work.
#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MhcMixStatsSinkhornKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_size: Dim,
    pub hc_mult: u32,
    pub sinkhorn_iters: u32,
    #[compute_dtype]
    pub hidden_dtype: DType,
}

pub struct MhcMixStatsSinkhornSpec;

impl KernelSpec for MhcMixStatsSinkhornSpec {
    type Config = MhcMixStatsSinkhornKernelConfig;
    type Input = MhcRmsNormKernelInput;

    const KIND: KernelKind = "mhc_mix_stats_sinkhorn";

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
                .with("sinkhorn_iters", config.sinkhorn_iters)
                .with("hidden_dtype", config.hidden_dtype.as_str())
        })
    }
}

register_kernel!(MhcMixStatsSinkhornKernel, MhcMixStatsSinkhornSpec);

#[cfg(test)]
mod tests {
    use super::{MhcMixStatsSinkhornKernelConfig, MhcMixStatsSinkhornSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    /// The payload must carry exactly the Python `MhcMixStatsSinkhornArgs`
    /// columns, including the Sinkhorn round count that keys the row.
    #[test]
    fn payload_matches_the_python_args_columns() {
        let config = MhcMixStatsSinkhornKernelConfig {
            backends: vec!["sglang_triton"],
            gpu_name: "NVIDIA RTX PRO 6000 Blackwell Server Edition".to_string(),
            hidden_size: 4096.into(),
            hc_mult: 4,
            sinkhorn_iters: 20,
            hidden_dtype: DType::Bf16,
        };
        let payloads = MhcMixStatsSinkhornSpec::enumerate(
            &config,
            &MhcMixStatsSinkhornSpec::sweep_grid(&config),
            "sglang_triton",
        );
        let fields: Vec<_> = payloads[0].fields().keys().cloned().collect();
        assert_eq!(
            fields,
            [
                "backend",
                "hc_mult",
                "hidden_dtype",
                "hidden_size",
                "num_tokens",
                "sinkhorn_iters"
            ]
        );
        assert!(payloads
            .iter()
            .all(|payload| payload.fields()["sinkhorn_iters"] == 20));
    }
}
