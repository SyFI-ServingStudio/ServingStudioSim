//! Static per-tensor FP8 quantization of an activation, as Inductor compiles
//! vLLM's native `QuantFP8`. An attention output gate (`x * sigmoid(g)`) in
//! front of the projection fuses into the same kernel and adds a full read of
//! the gate rows, so it is a config field.
//!
//! Backends: `vllm_inductor`, and `tensorfold_cuda` (TensorFold's
//! `quant8_kernel`, SM12x, ungated only: TensorFold applies the attention gate
//! in its own launch). Both share the token grid and the linear 1-D cache.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StaticFp8QuantKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    /// Each row is multiplied by the sigmoid of a same-width gate row first.
    pub gated: bool,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct StaticFp8QuantKernelInput {
    pub num_tokens: u32,
}

pub struct StaticFp8QuantSpec;

/// TensorFold's quantizer backend and the most rows one launch can take.
const TENSORFOLD_CUDA: &str = "tensorfold_cuda";
const TENSORFOLD_MAX_ROWS: f64 = 65535.0;

impl KernelSpec for StaticFp8QuantSpec {
    type Config = StaticFp8QuantKernelConfig;
    type Input = StaticFp8QuantKernelInput;

    const KIND: KernelKind = "static_fp8_quant";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::chain([Axis::pow2(0, 4), Axis::token_axis()])])
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
                .with("hidden_dim", config.hidden_dim.get())
                .with("gated", config.gated)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(StaticFp8QuantKernel, StaticFp8QuantSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn tensorfold_cuda_payload_matches_python_schema() {
        // The tensorfold_cuda runner keys rows on exactly these four fields and
        // refuses gated=true; a payload with another field set or a flipped
        // gate would never find its rows.
        let config = StaticFp8QuantKernelConfig {
            backends: vec!["tensorfold_cuda"],
            gpu_name: "NVIDIA GB10".to_string(),
            hidden_dim: 6144.into(),
            gated: false,
            dtype: DType::Bf16,
        };
        let grid = StaticFp8QuantSpec::sweep_grid(&config);
        let payloads = StaticFp8QuantSpec::enumerate(&config, &grid, "tensorfold_cuda");
        assert_eq!(payloads.len(), grid.axes()[0].len());
        let fields = payloads[0].fields();
        assert_eq!(fields.len(), 5);
        assert_eq!(fields.get("backend"), Some(&Value::from("tensorfold_cuda")));
        assert_eq!(fields.get("num_tokens"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("hidden_dim"), Some(&Value::from(6144_u32)));
        assert_eq!(fields.get("gated"), Some(&Value::from(false)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(
            StaticFp8QuantSpec::cache_kind("tensorfold_cuda"),
            CacheKind::Cache1DLinear
        );
        // The launch takes at most 65535 rows; only the 65536 point drops.
        let mask = StaticFp8QuantSpec::infeasible_mask(&config, &grid);
        assert_eq!(mask.iter().filter(|&&drop| drop).count(), 1);
        assert_eq!(mask.last(), Some(&true));
        let inductor = StaticFp8QuantKernelConfig {
            backends: vec!["vllm_inductor"],
            ..config
        };
        assert!(StaticFp8QuantSpec::infeasible_mask(&inductor, &grid).is_empty());
    }
}
