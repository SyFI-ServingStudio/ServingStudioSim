//! DeepSeek routed-expert clamped SwiGLU activation.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClampedSwigluKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct ClampedSwigluKernelInput {
    pub num_rows: u32,
}

pub struct ClampedSwigluSpec;

impl KernelSpec for ClampedSwigluSpec {
    type Config = ClampedSwigluKernelConfig;
    type Input = ClampedSwigluKernelInput;

    const KIND: KernelKind = "clamped_swiglu";

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
        grid.expand_1d(|num_rows| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_rows", num_rows as u32)
                .with("hidden_dim", config.hidden_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(ClampedSwigluKernel, ClampedSwigluSpec);

#[cfg(test)]
mod tests {
    use super::{ClampedSwigluKernelConfig, ClampedSwigluSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn payload_uses_routed_rows_not_source_tokens() {
        let config = ClampedSwigluKernelConfig {
            backends: vec!["vllm_inductor"],
            gpu_name: "NVIDIA H200".to_string(),
            hidden_dim: 2048.into(),
            dtype: DType::Bf16,
        };
        let payload = ClampedSwigluSpec::enumerate(
            &config,
            &ClampedSwigluSpec::sweep_grid(&config),
            "vllm_inductor",
        )
        .into_iter()
        .find(|payload| payload.fields()["num_rows"] == 128)
        .unwrap();
        assert_eq!(payload.fields()["hidden_dim"], 2048);
        assert_eq!(payload.fields().len(), 4);
    }
}
