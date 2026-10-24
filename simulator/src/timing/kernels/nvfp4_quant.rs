//! BF16-to-NVFP4 activation quantization: vLLM's and FlashInfer's on
//! Blackwell, and TensorFold's `quant4_kernel` (`tensorfold_cuda`, SM12x,
//! `scale_format = "kslice_e4m3"`: scales `[hidden/64, ceil(m/64)*64, 4]`
//! under the checkpoint's static input scale). Every backend shares the token
//! grid and the linear 1-D cache; `scale_format` is static identity.

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

/// TensorFold's quantizer backend and the most rows one launch can take.
const TENSORFOLD_CUDA: &str = "tensorfold_cuda";
const TENSORFOLD_MAX_ROWS: f64 = 65535.0;

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

    /// TensorFold's quantizer launches one block row per token row
    /// (`gridDim.y = num_tokens`), so it cannot run past CUDA's 65535 limit;
    /// its prefill chunks stay at or under 4096 rows. A config that names
    /// `tensorfold_cuda` drops those grid rows; queries past the last row
    /// extrapolate the linear cache.
    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        if !config.backends.contains(&TENSORFOLD_CUDA) {
            return Vec::new();
        }
        grid.axes()[0]
            .iter()
            .map(|&num_tokens| num_tokens > TENSORFOLD_MAX_ROWS)
            .collect()
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
            scale_format: "linear_e4m3".to_string(),
        };
        let grid = Nvfp4QuantSpec::sweep_grid(&config);
        let payloads = Nvfp4QuantSpec::enumerate(&config, &grid, "vllm_cuda");
        let fields = payloads[0].fields();

        assert_eq!(fields.len(), 6);
        assert_eq!(fields.get("backend"), Some(&Value::from("vllm_cuda")));
        assert_eq!(fields.get("num_tokens"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("hidden_size"), Some(&Value::from(6144_u32)));
        assert_eq!(fields.get("group_size"), Some(&Value::from(16_u32)));
        assert_eq!(fields.get("input_dtype"), Some(&Value::from("bf16")));
        assert_eq!(
            fields.get("scale_format"),
            Some(&Value::from("linear_e4m3"))
        );
    }

    #[test]
    fn tensorfold_cuda_payload_carries_its_kslice_scale_format() {
        // The tensorfold_cuda runner accepts only kslice_e4m3; a payload that
        // dropped or rewrote the config's scale_format would miss its rows.
        let config = Nvfp4QuantKernelConfig {
            backends: vec!["tensorfold_cuda"],
            gpu_name: "NVIDIA GB10".to_string(),
            hidden_size: 17408.into(),
            group_size: 16,
            input_dtype: DType::Bf16,
            scale_format: "kslice_e4m3".to_string(),
        };
        let grid = Nvfp4QuantSpec::sweep_grid(&config);
        let payloads = Nvfp4QuantSpec::enumerate(&config, &grid, "tensorfold_cuda");
        assert_eq!(payloads.len(), grid.axes()[0].len());
        for payload in &payloads {
            let fields = payload.fields();
            assert_eq!(fields.get("backend"), Some(&Value::from("tensorfold_cuda")));
            assert_eq!(
                fields.get("scale_format"),
                Some(&Value::from("kslice_e4m3"))
            );
            assert_eq!(fields.get("hidden_size"), Some(&Value::from(17408_u32)));
        }
        assert_eq!(
            Nvfp4QuantSpec::cache_kind("tensorfold_cuda"),
            CacheKind::Cache1DLinear
        );
        // Only the 65536-row point is past the launch's 65535-row limit; a
        // mask that kept it would fail the build on a row that cannot exist.
        let mask = Nvfp4QuantSpec::infeasible_mask(&config, &grid);
        let dropped: Vec<f64> = grid.axes()[0]
            .iter()
            .zip(&mask)
            .filter_map(|(&m, &drop)| drop.then_some(m))
            .collect();
        assert_eq!(dropped, vec![65536.0]);
        let vllm = Nvfp4QuantKernelConfig {
            backends: vec!["vllm_cuda"],
            ..config
        };
        assert!(Nvfp4QuantSpec::infeasible_mask(&vllm, &grid).is_empty());
    }
}
