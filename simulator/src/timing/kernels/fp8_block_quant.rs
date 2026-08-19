//! BF16-to-FP8 E4M3 dynamic quantization in independent 128-element blocks.
//!
//! L3 resolves routing and EP ownership before this L1 leaf is evaluated, so
//! `num_tokens` is the final local-rank row count. The full per-expert batch
//! vector does not enter the cache identity, but `num_problems` does: the local
//! expert count changes the grouped launch policy and scale-storage padding.
//! The first backend's output dtype and block size are fixed by the
//! FlashInfer/TensorRT-LLM `scale_1x128_kernel`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Fp8BlockQuantKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_size: Dim,
    pub num_problems: Dim,
    #[compute_dtype]
    pub input_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct Fp8BlockQuantKernelInput {
    pub num_tokens: u32,
}

pub struct Fp8BlockQuantSpec;

impl KernelSpec for Fp8BlockQuantSpec {
    type Config = Fp8BlockQuantKernelConfig;
    type Input = Fp8BlockQuantKernelInput;

    const KIND: KernelKind = "fp8_block_quant";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // Routed decode rows can be much smaller than the shared token axis's
        // first point. Keep explicit low-count samples before joining the
        // profiler-wide token curve used by larger routed prefill workloads.
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
                .with("num_problems", config.num_problems.get())
                .with("input_dtype", config.input_dtype.as_str())
        })
    }
}

register_kernel!(Fp8BlockQuantKernel, Fp8BlockQuantSpec);

#[cfg(test)]
mod tests {
    use super::{Fp8BlockQuantKernelConfig, Fp8BlockQuantKernelInput, Fp8BlockQuantSpec};
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> Fp8BlockQuantKernelConfig {
        Fp8BlockQuantKernelConfig {
            backends: vec!["flashinfer_trtllm"],
            gpu_name: "NVIDIA H200".to_string(),
            hidden_size: 4096.into(),
            num_problems: 32.into(),
            input_dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_includes_backend_gpu_shape_and_dtype() {
        let base = config();
        assert_eq!(base, base.clone());

        let mut changed_backend = config();
        changed_backend.backends = vec!["another_backend"];
        assert_ne!(base, changed_backend);

        let mut changed_gpu = config();
        changed_gpu.gpu_name = "NVIDIA H100".to_string();
        assert_ne!(base, changed_gpu);

        let mut changed_shape = config();
        changed_shape.hidden_size = 8192.into();
        assert_ne!(base, changed_shape);

        let mut changed_problem_count = config();
        changed_problem_count.num_problems = 16.into();
        assert_ne!(base, changed_problem_count);

        let mut changed_dtype = config();
        changed_dtype.input_dtype = DType::Fp16;
        assert_ne!(base, changed_dtype);
    }

    #[test]
    fn describe_config_preserves_the_static_identity() {
        assert_eq!(
            config().describe_config(),
            serde_json::json!({
                "backends": ["flashinfer_trtllm"],
                "gpu_name": "NVIDIA H200",
                "hidden_size": {"value": 4096, "expression": null, "bindings": {}},
                "num_problems": {"value": 32, "expression": null, "bindings": {}},
                "input_dtype": "bf16",
            })
        );
    }

    #[test]
    fn input_projects_to_num_tokens_axis() {
        let input = Fp8BlockQuantKernelInput { num_tokens: 48 };
        assert_eq!(&*input.coords(), &[48.0]);
        assert_eq!(
            Fp8BlockQuantKernelInput::coord_field_names(),
            &["num_tokens"]
        );
    }

    #[test]
    fn grid_covers_small_routed_rows_and_shared_token_curve() {
        let grid = Fp8BlockQuantSpec::sweep_grid(&config());
        assert_eq!(grid.axes().len(), 1);
        assert_eq!(
            &grid.axes()[0][..7],
            &[1.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0]
        );
        assert!(grid.axes()[0].contains(&2048.0));
        assert_eq!(grid.axes()[0].last().copied(), Some(65536.0));
        assert_eq!(
            Fp8BlockQuantSpec::cache_kind("flashinfer_trtllm"),
            CacheKind::Cache1DLinear
        );
    }

    #[test]
    fn input_dtype_is_the_compute_dtype_tag() {
        assert_eq!(config().compute_dtype(), Some(DType::Bf16));
    }

    #[test]
    fn enumerate_matches_python_args_exactly() {
        let config = config();
        let grid = Fp8BlockQuantSpec::sweep_grid(&config);
        let payload = &Fp8BlockQuantSpec::enumerate(&config, &grid, "flashinfer_trtllm")[0];
        let fields = payload.fields();

        assert_eq!(fields.len(), 5);
        assert_eq!(
            fields.get("backend"),
            Some(&Value::from("flashinfer_trtllm"))
        );
        assert_eq!(fields.get("num_tokens"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("hidden_size"), Some(&Value::from(4096_u32)));
        assert_eq!(fields.get("num_problems"), Some(&Value::from(32_u32)));
        assert_eq!(fields.get("input_dtype"), Some(&Value::from("bf16")));
        assert_eq!(payload.backend(), Some("flashinfer_trtllm"));
    }
}
