//! DeepSeek V4.1 fused Q pad and MXFP8 sliding-window KV insert.
//!
//! One launch of the fork's `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`
//! on the SM100 mega-attention path: the Q half copies the live heads and
//! zero-pads them to `padded_heads` in the chunk-interleaved layout (no norm,
//! no rotation; `padded_heads == 0` is the KV-only launch), and the KV half
//! applies GPT-J RoPE, MXFP8-quantizes, and inserts each row into the 32-token
//! SWA pages. Unlike `deepseek_v4_qnorm_rope_kv_insert`, whose rows mean Q
//! RMSNorm plus Q RoPE, this kind carries no Q transform.
//!
//! The op picks its variant from `num_tokens`: the ReducedGrid kernel at
//! `num_tokens >= 1024` when `padded_heads > 0`, else the warp-per-slot kernel.
//! The grid holds both 1023 and 1024 so linear interpolation never bridges
//! the variant step with a sample from farther away.
//!
//! `num_insert_tokens` (the slot-mapping length) equals `num_tokens` in
//! production: the fork passes one slot per row (padding rows carry slot -1),
//! and the V4 arch sets the two equal too. So it is derived in the payload
//! rather than exposed as a cache axis.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

/// The runner's supported live Q head counts per shard.
const LIVE_HEADS: [u32; 5] = [8, 16, 32, 64, 128];
/// Production SWA page (`DeepseekV4SWACache(block_size=32)`).
const BLOCK_SIZE: u32 = 32;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QPadKvRopeMxfp8InsertKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Live Q heads on this shard.
    pub num_heads: u32,
    /// FlashMLA Q width (64 or 128), or 0 for the KV-only launch.
    pub padded_heads: u32,
    pub block_size: u32,
    #[compute_dtype]
    pub input_dtype: DType,
    pub swa_cache_format: String,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QPadKvRopeMxfp8InsertKernelInput {
    pub num_tokens: u32,
}

pub struct QPadKvRopeMxfp8InsertSpec;

fn validate_config(config: &QPadKvRopeMxfp8InsertKernelConfig) {
    assert!(
        LIVE_HEADS.contains(&config.num_heads),
        "num_heads must be one of {LIVE_HEADS:?}, got {}",
        config.num_heads
    );
    let padded = if config.num_heads <= 64 { 64 } else { 128 };
    assert!(
        config.padded_heads == padded
            || (config.padded_heads == 0 && config.num_heads == padded),
        "padded_heads must be {padded} for {} live heads, or 0 (KV-only) when \
         the shard is already that wide; got {}",
        config.num_heads,
        config.padded_heads
    );
    assert_eq!(config.block_size, BLOCK_SIZE, "only the 32-token SWA page");
    assert_eq!(config.input_dtype, DType::Bf16, "only bf16 input");
    assert_eq!(config.swa_cache_format, "mxfp8", "only the mxfp8 SWA record");
}

impl KernelSpec for QPadKvRopeMxfp8InsertSpec {
    type Config = QPadKvRopeMxfp8InsertKernelConfig;
    type Input = QPadKvRopeMxfp8InsertKernelInput;
    const KIND: KernelKind = "q_pad_kv_rope_mxfp8_insert";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        validate_config(config);
        SweepGrid::new(vec![Axis::values([
            1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1023, 1024, 1536, 2048,
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
            let tokens = tokens as u32;
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", tokens)
                .with("num_insert_tokens", tokens)
                .with("num_heads", config.num_heads)
                .with("padded_heads", config.padded_heads)
                .with("block_size", config.block_size)
                .with("input_dtype", config.input_dtype.as_str())
                .with("swa_cache_format", config.swa_cache_format.clone())
        })
    }
}

register_kernel!(
    QPadKvRopeMxfp8InsertKernel,
    QPadKvRopeMxfp8InsertSpec
);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(num_heads: u32, padded_heads: u32) -> QPadKvRopeMxfp8InsertKernelConfig {
        QPadKvRopeMxfp8InsertKernelConfig {
            backends: vec!["vllm_cuda"],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads,
            padded_heads,
            block_size: 32,
            input_dtype: DType::Bf16,
            swa_cache_format: "mxfp8".to_string(),
        }
    }

    /// The variant step sits between 1023 and 1024; if either left the grid,
    /// interpolation would blend one variant's sample into the other's range.
    #[test]
    fn grid_brackets_the_reduced_grid_cutoff_with_adjacent_points() {
        let grid = QPadKvRopeMxfp8InsertSpec::sweep_grid(&config(16, 64));
        let axis = &grid.axes()[0];
        let i = axis.iter().position(|&t| t == 1023.0).unwrap();
        assert_eq!(axis[i + 1], 1024.0);
        assert!(axis.len() <= 500);
    }

    /// Catches a payload that drops the derived insert count or leaks a field
    /// the Python args schema would reject.
    #[test]
    fn payload_derives_insert_tokens_and_matches_python_args() {
        let cfg = config(16, 64);
        let grid = QPadKvRopeMxfp8InsertSpec::sweep_grid(&cfg);
        let payloads = QPadKvRopeMxfp8InsertSpec::enumerate(&cfg, &grid, "vllm_cuda");
        let fields = payloads[15].fields();
        let keys: Vec<_> = fields.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "block_size",
                "input_dtype",
                "num_heads",
                "num_insert_tokens",
                "num_tokens",
                "padded_heads",
                "swa_cache_format"
            ]
        );
        assert_eq!(fields["num_tokens"], 1023);
        assert_eq!(fields["num_insert_tokens"], 1023);
    }

    /// KV-only launches exist only when the shard is already padded wide.
    #[test]
    #[should_panic(expected = "padded_heads")]
    fn rejects_kv_only_for_a_narrow_shard() {
        QPadKvRopeMxfp8InsertSpec::sweep_grid(&config(16, 0));
    }
}
