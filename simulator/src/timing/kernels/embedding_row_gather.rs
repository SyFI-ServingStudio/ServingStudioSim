//! Raw embedding row gather from a large per-head-partitioned table.
//!
//! One call of the fork's per-layer n-gram (PLE) table lookup: it copies
//! `num_tokens * num_heads` rows of `head_dim` table elements, unchanged, into
//! a device staging buffer. With `residency = host_uva` (the production
//! `cpu_offload=True` default) the table is in pinned host memory and the fork's
//! `_lookup_ple_embedding_from_pinned_kernel` reads it over UVA, one program per
//! row; `device` is the HBM table read by `F.embedding` (`cpu_offload=False`).
//! Dequantization is not part of this kind: the per-tensor scale is applied by
//! separate elementwise launches. (`engram_lookup` is the block-scaled,
//! dequantizing variant with a different launch and output.)
//!
//! Overlap is not modeled here. Production issues the host lookup on a side
//! stream before layer 0 and joins it at the consuming layer; the composition
//! above this kernel decides how much of it is hidden.
//!
//! `table_rows` is an exact identity, never interpolated: the uniform random
//! gather's TLB and page footprint grows with the table size.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

const RESIDENCIES: [&str; 2] = ["host_uva", "device"];

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingRowGatherKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Ids per token, one per disjoint table range.
    pub num_heads: u32,
    /// Table elements per row.
    pub head_dim: u32,
    /// Table length (exact identity: sets the gather footprint).
    pub table_rows: u64,
    /// `host_uva` (pinned host over UVA) or `device` (HBM).
    pub residency: String,
    #[compute_dtype]
    pub weight_dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingRowGatherKernelInput {
    /// Rows of the [tokens, heads] id tensor (scheduled tokens).
    pub num_tokens: u32,
}

pub struct EmbeddingRowGatherSpec;

fn validate_config(config: &EmbeddingRowGatherKernelConfig) {
    assert!(config.num_heads >= 1, "num_heads must be positive");
    assert!(config.head_dim >= 1, "head_dim must be positive");
    assert!(
        config.table_rows >= u64::from(config.num_heads),
        "table_rows must be at least num_heads, got {}",
        config.table_rows
    );
    assert!(
        RESIDENCIES.contains(&config.residency.as_str()),
        "residency must be one of {RESIDENCIES:?}, got {}",
        config.residency
    );
}

impl KernelSpec for EmbeddingRowGatherSpec {
    type Config = EmbeddingRowGatherKernelConfig;
    type Input = EmbeddingRowGatherKernelInput;
    const KIND: KernelKind = "embedding_row_gather";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        validate_config(config);
        // 17 brackets PyTorch's small-index path (`F.embedding` on <= 16
        // indices costs ~5x the 17-index launch), which 16 -> 32 would smear.
        SweepGrid::new(vec![Axis::values([
            1, 2, 4, 8, 16, 17, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
            3072, 4096, 6144, 8192, 12288, 16384, 32768,
        ])])
    }

    fn cache_kind(_: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", tokens as u32)
                .with("num_heads", config.num_heads)
                .with("head_dim", config.head_dim)
                .with("table_rows", config.table_rows)
                .with("residency", config.residency.clone())
                .with("weight_dtype", config.weight_dtype.as_str())
        })
    }
}

register_kernel!(EmbeddingRowGatherKernel, EmbeddingRowGatherSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(residency: &str) -> EmbeddingRowGatherKernelConfig {
        EmbeddingRowGatherKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads: 16,
            head_dim: 160,
            table_rows: 320_001_536,
            residency: residency.to_string(),
            weight_dtype: DType::Fp8E4m3,
        }
    }

    /// Catches a payload that rounds the exact table size (it sets the gather
    /// footprint) or emits a field the Python args schema would reject.
    #[test]
    fn payload_forwards_exact_table_rows_and_matches_python_args() {
        let cfg = config("host_uva");
        let grid = EmbeddingRowGatherSpec::sweep_grid(&cfg);
        let payloads = EmbeddingRowGatherSpec::enumerate(&cfg, &grid, "vllm_triton");
        assert_eq!(payloads.len(), 26);
        let fields = payloads[8].fields();
        let keys: Vec<_> = fields.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "head_dim",
                "num_heads",
                "num_tokens",
                "residency",
                "table_rows",
                "weight_dtype"
            ]
        );
        assert_eq!(fields["num_tokens"], 64);
        assert_eq!(fields["table_rows"], 320_001_536u64);
    }

    #[test]
    #[should_panic(expected = "residency")]
    fn rejects_unknown_residency() {
        EmbeddingRowGatherSpec::sweep_grid(&config("managed"));
    }
}
