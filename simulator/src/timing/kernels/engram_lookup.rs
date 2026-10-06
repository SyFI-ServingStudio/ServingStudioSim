//! DeepSeek V4.1 Engram n-gram row lookup (FP8 rows + UE8M0 scales -> BF16).
//!
//! One launch of the fork's `_engram_lookup_kernel`
//! (`ParallelEngramEmbedding.lookup`) for one Engram layer: it gathers
//! `num_tokens * local_heads` rows of `head_dim` FP8 bytes plus
//! `head_dim / quant_block_size` UE8M0 scale bytes, dequantizes, and writes
//! BF16 rows to a device staging buffer. With `residency = host_uva` (the
//! production `cpu_offload=True` default) the tables sit in pinned host memory,
//! are read over UVA, and the launch uses the half-SM background grid;
//! `device` is the HBM table with the full-SM grid.
//!
//! Overlap is not modeled here. Production issues the two Engram layers'
//! lookups right after the n-gram hash kernel at the start of the forward, each
//! on its own per-layer prefetch side stream, so the two run concurrently. They
//! share the host link and TLBs, so a concurrent pair takes about the sum of two
//! isolated lookups (722 us span vs 2 x ~305 us at T=2048). The L3 composition,
//! not this kernel, applies that sum and combines the pair with the main path
//! (forward start to the layer-1 Engram consumer) by max.
//!
//! `table_rows` is an exact identity, never interpolated: the random gather is
//! TLB-bound, so time grows with the table footprint (about 3.6x at T=2048 from
//! a 1 GiB to the 23.6 GiB production table, flattening past 16 GiB). Profile
//! the rank's exact `part_num_embeddings`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

/// The runner's only row geometry (V4.1-Flash): 256 FP8 values, 32 per scale.
const HEAD_DIM: u32 = 256;
const QUANT_BLOCK_SIZE: u32 = 32;
/// `(max_ngram - 1) * n_heads` on V4.1-Flash; a rank never owns more columns.
const MAX_LOCAL_HEADS: u32 = 24;
/// The runner's largest pinned table (49 GiB).
const MAX_TABLE_ROWS: u64 = 200_000_000;
const RESIDENCIES: [&str; 2] = ["host_uva", "device"];

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EngramLookupKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Hash columns this rank owns (`part_n_hash_cols`; 6 at TP4).
    pub local_heads: u32,
    pub head_dim: u32,
    pub quant_block_size: u32,
    /// Rows of this rank's table slice (`part_num_embeddings`; 96,000,564 on
    /// V4.1-Flash TP4 rank 0). Exact identity: sets the gather's TLB footprint.
    pub table_rows: u64,
    /// `host_uva` (pinned host, half-SM grid) or `device` (HBM, full grid).
    pub residency: String,
    #[compute_dtype]
    pub weight_dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EngramLookupKernelInput {
    /// Rows of the hash-id tensor this launch reads (scheduled tokens).
    pub num_tokens: u32,
}

pub struct EngramLookupSpec;

fn validate_config(config: &EngramLookupKernelConfig) {
    assert!(
        (1..=MAX_LOCAL_HEADS).contains(&config.local_heads),
        "local_heads must be in [1, {MAX_LOCAL_HEADS}], got {}",
        config.local_heads
    );
    assert_eq!(config.head_dim, HEAD_DIM, "only head_dim {HEAD_DIM}");
    assert_eq!(
        config.quant_block_size, QUANT_BLOCK_SIZE,
        "only quant_block_size {QUANT_BLOCK_SIZE}"
    );
    assert!(
        (u64::from(config.local_heads)..=MAX_TABLE_ROWS).contains(&config.table_rows),
        "table_rows must be in [local_heads, {MAX_TABLE_ROWS}], got {}",
        config.table_rows
    );
    assert!(
        RESIDENCIES.contains(&config.residency.as_str()),
        "residency must be one of {RESIDENCIES:?}, got {}",
        config.residency
    );
    assert_eq!(config.weight_dtype, DType::Fp8E4m3, "only fp8_e4m3 tables");
}

impl KernelSpec for EngramLookupSpec {
    type Config = EngramLookupKernelConfig;
    type Input = EngramLookupKernelInput;
    const KIND: KernelKind = "engram_lookup";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        validate_config(config);
        SweepGrid::new(vec![Axis::values([
            1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072,
            4096, 6144, 8192, 12288, 16384,
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
                .with("local_heads", config.local_heads)
                .with("head_dim", config.head_dim)
                .with("quant_block_size", config.quant_block_size)
                .with("table_rows", config.table_rows)
                .with("residency", config.residency.clone())
                .with("weight_dtype", config.weight_dtype.as_str())
        })
    }
}

register_kernel!(EngramLookupKernel, EngramLookupSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(residency: &str) -> EngramLookupKernelConfig {
        EngramLookupKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            local_heads: 6,
            head_dim: 256,
            quant_block_size: 32,
            table_rows: 96_000_564,
            residency: residency.to_string(),
            weight_dtype: DType::Fp8E4m3,
        }
    }

    /// Catches a payload that rounds the exact table size (it sets the TLB
    /// footprint) or leaks a field the Python args schema would reject.
    #[test]
    fn payload_forwards_exact_table_rows_and_matches_python_args() {
        let cfg = config("host_uva");
        let grid = EngramLookupSpec::sweep_grid(&cfg);
        let payloads = EngramLookupSpec::enumerate(&cfg, &grid, "vllm_triton");
        assert_eq!(payloads.len(), 24);
        let fields = payloads[6].fields();
        let keys: Vec<_> = fields.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "head_dim",
                "local_heads",
                "num_tokens",
                "quant_block_size",
                "residency",
                "table_rows",
                "weight_dtype"
            ]
        );
        assert_eq!(fields["num_tokens"], 48);
        assert_eq!(fields["table_rows"], 96_000_564u64);
        assert_eq!(fields["residency"], "host_uva");
    }

    /// The runner only places tables in pinned host memory or HBM.
    #[test]
    #[should_panic(expected = "residency")]
    fn rejects_unknown_residency() {
        EngramLookupSpec::sweep_grid(&config("managed"));
    }
}
