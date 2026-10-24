//! Per-head QK RMSNorm and full rotate-half RoPE that also moves q, k and v
//! from the fused projection row into head-major `[heads, rows, head_dim]`
//! tensors. TensorFold's Triton `_prep_kernel` (DFlash2 `_prep`): drafting runs
//! it on the qkv projection, the context taps on the kv projection alone
//! (`num_q_heads` = 0).
//!
//! Backend: `tensorfold_triton` (bf16). Head counts and width are static
//! identity; the row count is the runtime axis, cached 1-D linear.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HeadMajorQkNormRopeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_q_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct HeadMajorQkNormRopeKernelInput {
    pub num_tokens: u32,
}

pub struct HeadMajorQkNormRopeSpec;

impl KernelSpec for HeadMajorQkNormRopeSpec {
    type Config = HeadMajorQkNormRopeKernelConfig;
    type Input = HeadMajorQkNormRopeKernelInput;

    const KIND: KernelKind = "head_major_qk_norm_rope";

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
                .with("num_q_heads", config.num_q_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(HeadMajorQkNormRopeKernel, HeadMajorQkNormRopeSpec);

#[cfg(test)]
mod tests {
    use super::{HeadMajorQkNormRopeKernelConfig, HeadMajorQkNormRopeSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn payloads_carry_exactly_the_python_fields() {
        // Catches a field the Python args do not have (or lack one they have),
        // which would key rows the profiler never writes.
        let cfg = HeadMajorQkNormRopeKernelConfig {
            backends: vec!["tensorfold_triton"],
            gpu_name: "NVIDIA GB10".to_string(),
            num_q_heads: 0.into(),
            num_kv_heads: 8.into(),
            head_dim: 128.into(),
            dtype: DType::Bf16,
        };
        let grid = HeadMajorQkNormRopeSpec::sweep_grid(&cfg);
        assert_eq!(grid.axes()[0][0], 1.0);
        let payloads = HeadMajorQkNormRopeSpec::enumerate(&cfg, &grid, "tensorfold_triton");
        let names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            names,
            [
                "backend",
                "dtype",
                "head_dim",
                "num_kv_heads",
                "num_q_heads",
                "num_tokens"
            ]
        );
        assert_eq!(payloads[0].backend(), Some("tensorfold_triton"));
    }
}
