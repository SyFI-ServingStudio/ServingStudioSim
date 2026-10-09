//! Fused short-conv QKVR prep: one `fused_qkvr_prep` call per attention layer.
//!
//! From the per-rank `qkvr` GEMM row (q | k | v | r) the call short-convs K
//! and V (taps from the paged conv-state cache for earlier forwards), RMS-norms
//! q and k per head, multiplies q and the rel logits by tau on global layers,
//! inserts K/V into the paged LBNHC attention cache and projects r to
//! `rel_extent` relative-position logits. One row is the whole call priced by
//! wall span (GPU-active union of a CUDA-graph replay), never a kernel sum.
//!
//! The call switches implementation twice along `num_tokens`:
//! - below 128 tokens one `_qkvr_qkv_kernel` launch; from 128 a two-stream
//!   fork (`_kv_kernel` ∥ `_q_kernel` + rel projection);
//! - inside the fork the rel projection takes its throughput variant once
//!   `8 * num_tokens` rows reach 8192 (`rel_extent` 512) or 2048 (otherwise),
//!   i.e. at 1024 or 256 tokens for 8 heads.
//!
//! The grid holds 127/128, 255/256 and 1023/1024 so linear interpolation
//! never bridges a switch with a sample from farther away. Rows are profiled
//! on all-decode batches (every conv tap read from the cache); prefill-heavy
//! batches of the same size run 5-20% faster on B200, a band that did not
//! justify a tap-source axis, so `num_tokens` is the only input.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SconvQkvrPrepKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Query heads on this rank.
    pub num_q_heads: u32,
    /// KV heads on this rank.
    pub num_kv_heads: u32,
    pub head_dim: u32,
    /// Width of one head's relative-position branch.
    pub d_rel: u32,
    /// Relative distances covered: the sliding window on local layers, the
    /// model's `rel_extent` on global layers.
    pub rel_extent: u32,
    /// tau scaling of q and the rel logits (global layers only).
    pub apply_log_scaling: bool,
    pub sconv_kernel_size: u32,
    /// Tokens per conv-state cache block (the conv kernel size in vLLM).
    pub conv_block_size: u32,
    /// Tokens per attention KV cache page.
    pub page_size: u32,
    /// qkvr, weights, outputs and both caches share one dtype.
    #[compute_dtype]
    #[kv_dtype]
    pub dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SconvQkvrPrepKernelInput {
    pub num_tokens: u32,
}

pub struct SconvQkvrPrepSpec;

fn validate_config(config: &SconvQkvrPrepKernelConfig) {
    // fused_qkvr_prep asserts both (qkvr_prep.py:820-821).
    assert_eq!(config.head_dim, 128, "fused_qkvr_prep needs head_dim 128");
    assert_eq!(config.d_rel, 16, "fused_qkvr_prep needs d_rel 16");
    assert!(
        config.num_kv_heads > 0 && config.num_q_heads % config.num_kv_heads == 0,
        "num_q_heads must be a multiple of num_kv_heads, got {}/{}",
        config.num_q_heads,
        config.num_kv_heads
    );
    assert!(config.rel_extent > 0, "rel_extent must be positive");
    assert_eq!(config.dtype, DType::Bf16, "the conv-state cache is bf16 only");
}

impl KernelSpec for SconvQkvrPrepSpec {
    type Config = SconvQkvrPrepKernelConfig;
    type Input = SconvQkvrPrepKernelInput;
    const KIND: KernelKind = "sconv_qkvr_prep";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        validate_config(config);
        SweepGrid::new(vec![Axis::values([
            1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96, 112, 120, 127,
            128, 136, 152, 176, 192, 224, 255, 256, 288, 320, 384, 448, 512, 640, 768, 896, 1023,
            1024, 1088, 1280, 1536, 1792, 2048, 3072, 4096, 6144, 8192,
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
                .with("num_q_heads", config.num_q_heads)
                .with("num_kv_heads", config.num_kv_heads)
                .with("head_dim", config.head_dim)
                .with("d_rel", config.d_rel)
                .with("rel_extent", config.rel_extent)
                .with("apply_log_scaling", config.apply_log_scaling)
                .with("sconv_kernel_size", config.sconv_kernel_size)
                .with("conv_block_size", config.conv_block_size)
                .with("page_size", config.page_size)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(SconvQkvrPrepKernel, SconvQkvrPrepSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(rel_extent: u32, apply_log_scaling: bool) -> SconvQkvrPrepKernelConfig {
        SconvQkvrPrepKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            num_q_heads: 8,
            num_kv_heads: 2,
            head_dim: 128,
            d_rel: 16,
            rel_extent,
            apply_log_scaling,
            sconv_kernel_size: 4,
            conv_block_size: 4,
            page_size: 32,
            dtype: DType::Bf16,
        }
    }

    /// Each implementation switch (fork at 128 tokens, rel-projection variant
    /// at 256 / 1024 tokens for 8 heads) must have its two sides as adjacent
    /// grid points; otherwise interpolation blends one path's sample into the
    /// other's range.
    #[test]
    fn grid_brackets_every_implementation_switch_with_adjacent_points() {
        let grid = SconvQkvrPrepSpec::sweep_grid(&config(512, false));
        let axis = &grid.axes()[0];
        for switch in [128.0, 256.0, 1024.0] {
            let i = axis.iter().position(|&t| t == switch).unwrap();
            assert_eq!(axis[i - 1], switch - 1.0, "no point just below {switch}");
        }
        assert!(axis.len() <= 500);
    }

    /// Catches a payload that drops or renames a field the Python
    /// `SconvQkvrPrepArgs` schema requires, or leaks an extra one.
    #[test]
    fn payload_matches_python_args() {
        let cfg = config(1024, true);
        let grid = SconvQkvrPrepSpec::sweep_grid(&cfg);
        let payloads = SconvQkvrPrepSpec::enumerate(&cfg, &grid, "vllm_triton");
        let fields = payloads[21].fields();
        let keys: Vec<_> = fields.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "apply_log_scaling",
                "backend",
                "conv_block_size",
                "d_rel",
                "dtype",
                "head_dim",
                "num_kv_heads",
                "num_q_heads",
                "num_tokens",
                "page_size",
                "rel_extent",
                "sconv_kernel_size"
            ]
        );
        assert_eq!(fields["num_tokens"], 127);
        assert_eq!(fields["apply_log_scaling"], true);
        assert_eq!(fields["dtype"], "bf16");
    }

    /// The production call asserts head_dim 128; a config that slipped past
    /// would profile a shape vLLM never runs.
    #[test]
    #[should_panic(expected = "head_dim 128")]
    fn rejects_a_head_dim_the_call_asserts_away() {
        let mut cfg = config(512, false);
        cfg.head_dim = 64;
        SconvQkvrPrepSpec::sweep_grid(&cfg);
    }
}
