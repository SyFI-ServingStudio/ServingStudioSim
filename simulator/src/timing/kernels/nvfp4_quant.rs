//! vLLM BF16-to-NVFP4 activation quantization on Blackwell.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Nvfp4QuantKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_size: Dim,
    pub group_size: u32,
    #[compute_dtype]
    pub input_dtype: DType,
    pub scale_format: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct Nvfp4QuantKernelInput {
    pub num_tokens: u32,
}

pub struct Nvfp4QuantSpec;

impl KernelSpec for Nvfp4QuantSpec {
    type Config = Nvfp4QuantKernelConfig;
    type Input = Nvfp4QuantKernelInput;

    const KIND: KernelKind = "nvfp4_quant";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::chain([
            Axis::values([1, 4, 8, 16, 32, 48]),
            Axis::token_axis(),
        ])])
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
                .with("group_size", config.group_size)
                .with("input_dtype", config.input_dtype.as_str())
                .with("scale_format", config.scale_format.as_str())
        })
    }
}

register_kernel!(Nvfp4QuantKernel, Nvfp4QuantSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn enumerate_matches_python_schema() {
        let config = Nvfp4QuantKernelConfig {
            backends: vec!["vllm_cuda"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden_size: 6144.into(),
            group_size: 16,
            input_dtype: DType::Bf16,
            scale_format: "trtllm_swizzled_e4m3".to_string(),
        };
        let grid = Nvfp4QuantSpec::sweep_grid(&config);
        let fields = Nvfp4QuantSpec::enumerate(&config, &grid, "vllm_cuda")[0]
            .fields()
            .clone();
        assert_eq!(fields.len(), 6);
        assert_eq!(fields.get("num_tokens"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("hidden_size"), Some(&Value::from(6144_u32)));
        assert_eq!(fields.get("group_size"), Some(&Value::from(16_u32)));
        assert_eq!(fields.get("input_dtype"), Some(&Value::from("bf16")));
    }
}
