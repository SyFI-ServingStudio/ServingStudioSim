//! SiLU of an input divided by a constant: the vLLM fork's (892da082) `hc_silu`
//! Triton op, one program per row. The divisor is a compile-time scalar that does not change
//! the time, so it is not a field.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ScaledSiluKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub width: Dim,
    #[compute_dtype]
    pub hidden_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct ScaledSiluKernelInput {
    pub num_tokens: u32,
}

pub struct ScaledSiluSpec;

impl KernelSpec for ScaledSiluSpec {
    type Config = ScaledSiluKernelConfig;
    type Input = ScaledSiluKernelInput;

    const KIND: KernelKind = "scaled_silu";

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
                .with("width", config.width.get())
                .with("hidden_dtype", config.hidden_dtype.as_str())
        })
    }
}

register_kernel!(ScaledSiluKernel, ScaledSiluSpec);

#[cfg(test)]
mod tests {
    use super::{ScaledSiluKernelConfig, ScaledSiluSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    /// The payload must carry exactly the Python `ScaledSiluArgs` fields: in
    /// particular no divisor, which the schema deliberately leaves out.
    #[test]
    fn payload_carries_exactly_the_python_args() {
        let config = ScaledSiluKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            width: 320.into(),
            hidden_dtype: DType::Bf16,
        };
        let payloads =
            ScaledSiluSpec::enumerate(&config, &ScaledSiluSpec::sweep_grid(&config), "vllm_triton");
        let mut fields: Vec<_> = payloads[0].fields().keys().cloned().collect();
        fields.sort();
        assert_eq!(fields, ["backend", "hidden_dtype", "num_tokens", "width"]);
    }
}
