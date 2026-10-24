//! Gated attention's output gate: each attention output head times the sigmoid
//! of its gate head, read in place from the `[q | gate]` projection rows.
//! TensorFold's Triton `_gate_mul`, one program per (token, head); vLLM fuses
//! the same multiply into `static_fp8_quant` (gated) instead.
//!
//! Backend: `tensorfold_triton` (bf16 only). Heads and head width are static
//! identity; the token count is the one runtime axis, cached 1-D linear.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AttnOutputGateKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct AttnOutputGateKernelInput {
    pub num_tokens: u32,
}

pub struct AttnOutputGateSpec;

impl KernelSpec for AttnOutputGateSpec {
    type Config = AttnOutputGateKernelConfig;
    type Input = AttnOutputGateKernelInput;

    const KIND: KernelKind = "attn_output_gate";

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
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(AttnOutputGateKernel, AttnOutputGateSpec);

#[cfg(test)]
mod tests {
    use super::{AttnOutputGateKernelConfig, AttnOutputGateSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;
    use serde_json::Value;

    #[test]
    fn payloads_carry_exactly_the_python_fields() {
        // Catches a field the Python AttnOutputGateArgs does not have (or lacks
        // one it has), which would key rows the profiler never writes.
        let cfg = AttnOutputGateKernelConfig {
            backends: vec!["tensorfold_triton"],
            gpu_name: "NVIDIA GB10".to_string(),
            num_heads: 24.into(),
            head_dim: 256.into(),
            dtype: DType::Bf16,
        };
        let grid = AttnOutputGateSpec::sweep_grid(&cfg);
        let payloads = AttnOutputGateSpec::enumerate(&cfg, &grid, "tensorfold_triton");
        let first = payloads[0].fields();
        let names: Vec<&str> = first.keys().map(String::as_str).collect();
        assert_eq!(
            names,
            ["backend", "dtype", "head_dim", "num_heads", "num_tokens"]
        );
        assert_eq!(first["num_heads"], Value::from(24_u32));
        assert_eq!(payloads[0].backend(), Some("tensorfold_triton"));
    }
}
