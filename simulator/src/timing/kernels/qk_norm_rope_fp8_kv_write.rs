//! Fused per-head q/k RMSNorm + NeoX RoPE + fp8 query quant + fp8 paged K/V
//! write over a GQA layer's packed `qkv` projection output. One launch replaces
//! q_norm, k_norm, RoPE, the query quant and the cache insert; the Python
//! `QkNormRopeFp8KvWriteArgs` dataclass owns the schema.
//!
//! One axis, the row count -> `Cache1DLinear`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::SweepGrid;
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QkNormRopeFp8KvWriteKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub page_size: u32,
    #[compute_dtype]
    pub input_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct QkNormRopeFp8KvWriteKernelInput {
    pub num_tokens: u32,
}

pub struct QkNormRopeFp8KvWriteSpec;

impl KernelSpec for QkNormRopeFp8KvWriteSpec {
    type Config = QkNormRopeFp8KvWriteKernelConfig;
    type Input = QkNormRopeFp8KvWriteKernelInput;

    const KIND: KernelKind = "qk_norm_rope_fp8_kv_write";

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
                .with("num_qo_heads", config.num_qo_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("page_size", config.page_size)
                .with("input_dtype", config.input_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
        })
    }
}

register_kernel!(QkNormRopeFp8KvWriteKernel, QkNormRopeFp8KvWriteSpec);

#[cfg(test)]
mod tests {
    use super::{QkNormRopeFp8KvWriteKernelConfig, QkNormRopeFp8KvWriteSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn enumerate_forwards_every_wire_field() {
        let cfg = QkNormRopeFp8KvWriteKernelConfig {
            backends: vec!["triton"],
            gpu_name: "NVIDIA B200".to_string(),
            num_qo_heads: 32.into(),
            num_kv_heads: 8.into(),
            head_dim: 128.into(),
            page_size: 16,
            input_dtype: DType::Bf16,
            kv_dtype: DType::Fp8E4m3,
        };
        let grid = QkNormRopeFp8KvWriteSpec::sweep_grid(&cfg);
        let payloads = QkNormRopeFp8KvWriteSpec::enumerate(&cfg, &grid, "triton");
        assert_eq!(payloads.len(), crate::timing::kernels::DRAFT_ROW_AXIS.len());
        // Wire schema must stay aligned with the Python args dataclass.
        assert_eq!(payloads[0].fields().len(), 8);
    }
}
