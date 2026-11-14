//! One launch that K-normalizes, rotates and writes the context K/V rows of
//! every draft layer into that layer's fp8 paged cache, from the stacked K/V
//! projection of all layers. It replaces the per-layer norm + RoPE + cache
//! insert chain; the Python `LayeredKvNormRopeCacheWriteArgs` dataclass owns
//! the schema.
//!
//! One axis, the context row count -> `Cache1DLinear`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::SweepGrid;
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LayeredKvNormRopeCacheWriteKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_layers: u32,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub page_size: u32,
    #[compute_dtype]
    pub input_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct LayeredKvNormRopeCacheWriteKernelInput {
    pub num_tokens: u32,
}

pub struct LayeredKvNormRopeCacheWriteSpec;

impl KernelSpec for LayeredKvNormRopeCacheWriteSpec {
    type Config = LayeredKvNormRopeCacheWriteKernelConfig;
    type Input = LayeredKvNormRopeCacheWriteKernelInput;

    const KIND: KernelKind = "layered_kv_norm_rope_cache_write";

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
                .with("num_layers", config.num_layers)
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("page_size", config.page_size)
                .with("input_dtype", config.input_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
        })
    }
}

register_kernel!(
    LayeredKvNormRopeCacheWriteKernel,
    LayeredKvNormRopeCacheWriteSpec
);

#[cfg(test)]
mod tests {
    use super::{LayeredKvNormRopeCacheWriteKernelConfig, LayeredKvNormRopeCacheWriteSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn enumerate_forwards_every_wire_field() {
        let cfg = LayeredKvNormRopeCacheWriteKernelConfig {
            backends: vec!["triton"],
            gpu_name: "NVIDIA B200".to_string(),
            num_layers: 5,
            num_kv_heads: 8.into(),
            head_dim: 128.into(),
            page_size: 16,
            input_dtype: DType::Bf16,
            kv_dtype: DType::Fp8E4m3,
        };
        let grid = LayeredKvNormRopeCacheWriteSpec::sweep_grid(&cfg);
        let payloads = LayeredKvNormRopeCacheWriteSpec::enumerate(&cfg, &grid, "triton");
        assert_eq!(payloads.len(), crate::timing::kernels::DRAFT_ROW_AXIS.len());
        // Wire schema must stay aligned with the Python args dataclass.
        assert_eq!(payloads[0].fields().len(), 8);
    }
}
