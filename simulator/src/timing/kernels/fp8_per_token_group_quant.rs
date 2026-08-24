//! vLLM dense BF16-to-FP8 per-token-group input quantization.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Fp8PerTokenGroupQuantKernelConfig {
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
pub struct Fp8PerTokenGroupQuantKernelInput {
    pub num_tokens: u32,
}

pub struct Fp8PerTokenGroupQuantSpec;

impl KernelSpec for Fp8PerTokenGroupQuantSpec {
    type Config = Fp8PerTokenGroupQuantKernelConfig;
    type Input = Fp8PerTokenGroupQuantKernelInput;

    const KIND: KernelKind = "fp8_per_token_group_quant";

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
            // Sweep axis values come from Axis::values/token_axis, all
            // non-negative integers well under u32::MAX.
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "sweep axis values are non-negative integers far below u32::MAX"
            )]
            let num_tokens = num_tokens as u32;
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens)
                .with("hidden_size", config.hidden_size.get())
                .with("group_size", config.group_size)
                .with("input_dtype", config.input_dtype.as_str())
                .with("scale_format", config.scale_format.clone())
        })
    }
}

register_kernel!(Fp8PerTokenGroupQuantKernel, Fp8PerTokenGroupQuantSpec);

#[cfg(test)]
mod tests {
    use super::{Fp8PerTokenGroupQuantKernelConfig, Fp8PerTokenGroupQuantSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;
    use crate::timing::Dim;
    use serde_json::Value;

    fn config() -> Fp8PerTokenGroupQuantKernelConfig {
        Fp8PerTokenGroupQuantKernelConfig {
            backends: vec!["vllm_cuda"],
            gpu_name: "NVIDIA H200".to_string(),
            hidden_size: Dim::from(4096),
            group_size: 128,
            input_dtype: DType::Bf16,
            scale_format: "ue8m0_column_major".to_string(),
        }
    }

    #[test]
    fn enumerate_matches_python_wire_schema() {
        let config = config();
        let grid = Fp8PerTokenGroupQuantSpec::sweep_grid(&config);
        let payload = &Fp8PerTokenGroupQuantSpec::enumerate(&config, &grid, "vllm_cuda")[0];
        let fields = payload.fields();
        assert_eq!(fields.len(), 6);
        assert_eq!(fields.get("num_tokens"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("hidden_size"), Some(&Value::from(4096_u32)));
        assert_eq!(fields.get("group_size"), Some(&Value::from(128_u32)));
        assert_eq!(fields.get("input_dtype"), Some(&Value::from("bf16")));
        assert_eq!(
            fields.get("scale_format"),
            Some(&Value::from("ue8m0_column_major"))
        );
    }
}
