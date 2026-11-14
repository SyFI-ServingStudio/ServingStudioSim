//! Grouped block convolution (finish half: the prepared per-group taps applied
//! within each block) fused with the residual add and the RMSNorm that follows
//! it. The Python `GroupedConvResidualRmsNormArgs` dataclass owns the schema.
//!
//! One axis, the row count -> `Cache1DLinear`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::SweepGrid;
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GroupedConvResidualRmsNormKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden: Dim,
    pub group_size: u32,
    pub taps: u32,
    pub block_len: u32,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct GroupedConvResidualRmsNormKernelInput {
    pub num_tokens: u32,
}

pub struct GroupedConvResidualRmsNormSpec;

impl KernelSpec for GroupedConvResidualRmsNormSpec {
    type Config = GroupedConvResidualRmsNormKernelConfig;
    type Input = GroupedConvResidualRmsNormKernelInput;

    const KIND: KernelKind = "grouped_conv_residual_rms_norm";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![super::draft_row_axis()])
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
                .with("group_size", config.group_size)
                .with("taps", config.taps)
                .with("block_len", config.block_len)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(
    GroupedConvResidualRmsNormKernel,
    GroupedConvResidualRmsNormSpec
);

#[cfg(test)]
mod tests {
    use super::{GroupedConvResidualRmsNormKernelConfig, GroupedConvResidualRmsNormSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn enumerate_forwards_every_wire_field() {
        let cfg = GroupedConvResidualRmsNormKernelConfig {
            backends: vec!["triton"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden: 4096.into(),
            group_size: 16,
            taps: 2,
            block_len: 8,
            dtype: DType::Bf16,
        };
        let grid = GroupedConvResidualRmsNormSpec::sweep_grid(&cfg);
        let payloads = GroupedConvResidualRmsNormSpec::enumerate(&cfg, &grid, "triton");
        assert_eq!(payloads.len(), crate::timing::kernels::DRAFT_ROW_AXIS.len());
        // Wire schema must stay aligned with the Python args dataclass.
        assert_eq!(payloads[0].fields().len(), 7);
    }
}
