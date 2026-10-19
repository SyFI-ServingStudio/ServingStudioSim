//! Grouped Gemma RMSNorm: each of `num_groups` contiguous `hidden_size`-wide
//! groups of a row gets its own RMS. The vLLM fork's (892da082)
//! `grouped_gemma_rmsnorm` Triton op, one program per (token, group); gated
//! hyper-connections run it on the stream-outer residual when an HC module has
//! no pending combine to fuse.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GroupedRmsNormKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_size: Dim,
    pub num_groups: u32,
    #[compute_dtype]
    pub hidden_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct GroupedRmsNormKernelInput {
    pub num_tokens: u32,
}

pub struct GroupedRmsNormSpec;

impl KernelSpec for GroupedRmsNormSpec {
    type Config = GroupedRmsNormKernelConfig;
    type Input = GroupedRmsNormKernelInput;

    const KIND: KernelKind = "grouped_rms_norm";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::chain([Axis::pow2(0, 4), Axis::token_axis()])])
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
                .with("num_groups", config.num_groups)
                .with("hidden_dtype", config.hidden_dtype.as_str())
        })
    }
}

register_kernel!(GroupedRmsNormKernel, GroupedRmsNormSpec);

#[cfg(test)]
mod tests {
    use super::{GroupedRmsNormKernelConfig, GroupedRmsNormSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    /// The payload must carry exactly the Python `GroupedRmsNormArgs` fields, or
    /// every row lookup misses and the launcher JIT-profiles a wrong spec.
    #[test]
    fn payload_carries_exactly_the_python_args() {
        let config = GroupedRmsNormKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden_size: 2560.into(),
            num_groups: 4,
            hidden_dtype: DType::Bf16,
        };
        let payloads = GroupedRmsNormSpec::enumerate(
            &config,
            &GroupedRmsNormSpec::sweep_grid(&config),
            "vllm_triton",
        );
        let mut fields: Vec<_> = payloads[0].fields().keys().cloned().collect();
        fields.sort();
        assert_eq!(
            fields,
            [
                "backend",
                "hidden_dtype",
                "hidden_size",
                "num_groups",
                "num_tokens"
            ]
        );
        assert_eq!(payloads[0].fields()["num_tokens"], 1);
    }
}
