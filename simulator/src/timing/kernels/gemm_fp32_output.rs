//! GEMM (BF16 or FP32 inputs) whose production callable materializes an FP32 output.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GemmFp32OutputKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub n: Dim,
    pub k: Dim,
    #[compute_dtype]
    pub input_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct GemmFp32OutputKernelInput {
    pub m: u32,
}

pub struct GemmFp32OutputSpec;

impl KernelSpec for GemmFp32OutputSpec {
    type Config = GemmFp32OutputKernelConfig;
    type Input = GemmFp32OutputKernelInput;

    const KIND: KernelKind = "gemm_fp32_output";

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
        grid.expand_1d(|m| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("m", m as u32)
                .with("n", config.n.get())
                .with("k", config.k.get())
                .with("input_dtype", config.input_dtype.as_str())
        })
    }
}

register_kernel!(GemmFp32OutputKernel, GemmFp32OutputSpec);

#[cfg(test)]
mod tests {
    use super::{GemmFp32OutputKernelConfig, GemmFp32OutputSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn output_precision_is_kind_semantics_not_a_payload_knob() {
        let config = GemmFp32OutputKernelConfig {
            backends: vec!["torch_cublas"],
            gpu_name: "NVIDIA H200".to_string(),
            n: 256.into(),
            k: 4096.into(),
            input_dtype: DType::Bf16,
        };
        let payload = GemmFp32OutputSpec::enumerate(
            &config,
            &GemmFp32OutputSpec::sweep_grid(&config),
            "torch_cublas",
        )
        .remove(0);
        assert_eq!(payload.fields()["m"], 1);
        assert_eq!(payload.fields()["n"], 256);
        assert_eq!(payload.fields()["k"], 4096);
        assert!(payload.fields().get("output_dtype").is_none());
    }
}
