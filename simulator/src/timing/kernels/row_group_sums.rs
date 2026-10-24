//! Row group sums: the fp32 sum of every `group_size`-element group of each
//! bf16 activation row, which an affine 4-bit GEMM reads for its bias term.
//! TensorFold's Triton `_group_sums` (`cuda/kernels/qmm.py`), launched before
//! every DFlash2 drafter projection whose caller has no sums yet.
//!
//! Backend: `tensorfold_triton` (bf16). Row width and group size are static
//! identity; the row count is the one runtime axis, cached 1-D linear.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RowGroupSumsKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden: Dim,
    pub group_size: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct RowGroupSumsKernelInput {
    pub num_tokens: u32,
}

pub struct RowGroupSumsSpec;

impl KernelSpec for RowGroupSumsSpec {
    type Config = RowGroupSumsKernelConfig;
    type Input = RowGroupSumsKernelInput;

    const KIND: KernelKind = "row_group_sums";

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
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(RowGroupSumsKernel, RowGroupSumsSpec);

#[cfg(test)]
mod tests {
    use super::{RowGroupSumsKernelConfig, RowGroupSumsSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn payloads_carry_exactly_the_python_fields() {
        // Catches a field the Python args do not have (or lack one they have),
        // which would key rows the profiler never writes.
        let cfg = RowGroupSumsKernelConfig {
            backends: vec!["tensorfold_triton"],
            gpu_name: "NVIDIA GB10".to_string(),
            hidden: 5120.into(),
            group_size: 64.into(),
            dtype: DType::Bf16,
        };
        let grid = RowGroupSumsSpec::sweep_grid(&cfg);
        assert_eq!(grid.axes()[0][0], 1.0);
        let payloads = RowGroupSumsSpec::enumerate(&cfg, &grid, "tensorfold_triton");
        let names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            names,
            ["backend", "dtype", "group_size", "hidden", "num_tokens"]
        );
        assert_eq!(payloads[0].backend(), Some("tensorfold_triton"));
    }
}
