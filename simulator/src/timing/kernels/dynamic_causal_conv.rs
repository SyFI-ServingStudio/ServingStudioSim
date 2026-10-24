//! Two-tap grouped dynamic causal convolution of the DFlash2 drafter: each row
//! mixed with the previous row of its block by per-row, per-channel-group taps,
//! with the residual add fused on the output-side calls. TensorFold's Triton
//! `_dconv_kernel` (`dflash2.py` `_dconv`), four launches a drafter layer.
//!
//! Backend: `tensorfold_triton` (bf16). Width, tap group and whether a residual
//! is read are static identity; the row count is the runtime axis, cached 1-D
//! linear.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DynamicCausalConvKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden: Dim,
    pub group_size: Dim,
    pub residual: bool,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DynamicCausalConvKernelInput {
    pub num_tokens: u32,
}

pub struct DynamicCausalConvSpec;

impl KernelSpec for DynamicCausalConvSpec {
    type Config = DynamicCausalConvKernelConfig;
    type Input = DynamicCausalConvKernelInput;

    const KIND: KernelKind = "dynamic_causal_conv";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // From one row (one stream's accepted rows) to past TensorFold's
        // largest call (a 4096-row prefill chunk); a byte-bound launch is
        // linear between the 1.5x steps.
        SweepGrid::new(vec![Axis::values([
            1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536,
            2048, 3072, 4096, 6144, 8192,
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
                .with("hidden", config.hidden.get())
                .with("group_size", config.group_size.get())
                .with("residual", config.residual)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(DynamicCausalConvKernel, DynamicCausalConvSpec);

#[cfg(test)]
mod tests {
    use super::{DynamicCausalConvKernelConfig, DynamicCausalConvSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn payloads_carry_exactly_the_python_fields() {
        // Catches a field the Python args do not have (or lack one they have),
        // which would key rows the profiler never writes.
        let cfg = DynamicCausalConvKernelConfig {
            backends: vec!["tensorfold_triton"],
            gpu_name: "NVIDIA GB10".to_string(),
            hidden: 5120.into(),
            group_size: 16.into(),
            residual: true,
            dtype: DType::Bf16,
        };
        let grid = DynamicCausalConvSpec::sweep_grid(&cfg);
        assert_eq!(grid.axes()[0][0], 1.0);
        let payloads = DynamicCausalConvSpec::enumerate(&cfg, &grid, "tensorfold_triton");
        let names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            names,
            [
                "backend",
                "dtype",
                "group_size",
                "hidden",
                "num_tokens",
                "residual"
            ]
        );
        assert_eq!(payloads[0].backend(), Some("tensorfold_triton"));
    }
}
