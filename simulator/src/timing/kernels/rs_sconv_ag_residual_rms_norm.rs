//! Inkling's fused TP collective: reduce-scatter over the hidden dim + causal
//! depthwise short conv (k=4, with skip) on each rank's shard through the
//! paged conv-state cache + all-gather + residual add + RMSNorm.
//!
//! One leaf is one whole `LamportRSConv.rs_sconv_ag_add_norm` call: four
//! PDL-chained Triton launches coupled by cross-GPU spin-waits, which cannot be
//! timed apart. The profile is the balanced (all ranks arrive together) cost;
//! cross-rank skew from upstream work belongs to that work's max over ranks.
//!
//! `hidden_dim`, `num_gpus` and `has_shared` (the MoE sink partial) are static
//! config; `num_tokens` (every rank holds every token) is the sweep axis. The
//! kernel splits its grid 8 ways only at one token, so the grid keeps T=1..8
//! exactly, then follows the CTA-count growth up to the symmetric buffer's
//! 16384-token capacity.

use crate::common::Fabric;
use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

/// `lamport._MAX_TOKENS`: the Lamport symmetric buffers hold at most this many
/// tokens; vLLM uses `min(16384, max_num_batched_tokens)` and falls back to
/// NCCL above it.
pub const RS_SCONV_MAX_TOKENS: u32 = 16384;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RsSconvAgResidualRmsNormKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_gpus: u32,
    pub hidden_dim: u32,
    #[compute_dtype]
    pub dtype: DType,
    pub fabric: Fabric,
    /// Short-conv window; the production kernel requires 4.
    pub conv_kernel_size: u32,
    /// Whether the call also sums the MoE sink partial (MoE MLP sublayers).
    pub has_shared: bool,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct RsSconvAgResidualRmsNormKernelInput {
    pub num_tokens: u32,
}

pub struct RsSconvAgResidualRmsNormSpec;

impl KernelSpec for RsSconvAgResidualRmsNormSpec {
    type Config = RsSconvAgResidualRmsNormKernelConfig;
    type Input = RsSconvAgResidualRmsNormKernelInput;

    const KIND: KernelKind = "rs_sconv_ag_residual_rms_norm";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // T=1 is its own 8-way-split launch shape; 2..8 are latency-bound
        // single-CTA-per-token grids. Above that the cost grows with the CTA
        // count, with warp-count switches at 128, 256 and 2048 tokens.
        SweepGrid::new(vec![Axis::chain([
            Axis::arithmetic(1, 8, 1),
            Axis::arithmetic(12, 32, 4),
            Axis::arithmetic(40, 64, 8),
            Axis::arithmetic(80, 128, 16),
            Axis::arithmetic(160, 256, 32),
            Axis::arithmetic(320, 512, 64),
            Axis::arithmetic(640, 1024, 128),
            Axis::arithmetic(1280, 2048, 256),
            Axis::values([2560, 3072, 4096, 6144, 8192, 12288, RS_SCONV_MAX_TOKENS]),
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
                .with("num_gpus", config.num_gpus)
                .with("num_tokens", num_tokens as u32)
                .with("hidden_dim", config.hidden_dim)
                .with("conv_kernel_size", config.conv_kernel_size)
                .with("has_shared", config.has_shared)
                .with("dtype", config.dtype.as_str())
                .with("fabric", config.fabric.as_str())
        })
    }
}

register_kernel!(RsSconvAgResidualRmsNormKernel, RsSconvAgResidualRmsNormSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn config(has_shared: bool) -> RsSconvAgResidualRmsNormKernelConfig {
        RsSconvAgResidualRmsNormKernelConfig {
            backends: vec!["vllm_lamport_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            num_gpus: 4,
            hidden_dim: 4096,
            dtype: DType::Bf16,
            fabric: Fabric::Nvlink,
            conv_kernel_size: 4,
            has_shared,
        }
    }

    #[test]
    fn grid_keeps_the_split_token_and_ends_at_buffer_capacity() {
        // T=1 is the only 8-way split launch; interpolating it from T=2 would
        // hide that shape. Above the buffer capacity vLLM leaves this path.
        let grid = RsSconvAgResidualRmsNormSpec::sweep_grid(&config(false));
        let tokens = &grid.axes()[0];
        assert_eq!(&tokens[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(tokens.last(), Some(&(RS_SCONV_MAX_TOKENS as f64)));
        for boundary in [128.0, 256.0, 2048.0] {
            assert!(tokens.contains(&boundary), "missing warp switch {boundary}");
        }
        assert!(tokens.len() <= 500);
    }

    #[test]
    fn enumerate_matches_python_args_exactly() {
        // A missing, renamed or extra key misses every profile.db row.
        let config = config(true);
        let grid = RsSconvAgResidualRmsNormSpec::sweep_grid(&config);
        let payloads =
            RsSconvAgResidualRmsNormSpec::enumerate(&config, &grid, "vllm_lamport_triton");
        assert_eq!(payloads.len(), grid.axes()[0].len());
        let fields = payloads[0].fields();
        let mut keys: Vec<&str> = fields.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "backend",
                "conv_kernel_size",
                "dtype",
                "fabric",
                "has_shared",
                "hidden_dim",
                "num_gpus",
                "num_tokens"
            ]
        );
        assert_eq!(fields.get("num_tokens"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("has_shared"), Some(&Value::from(true)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("fabric"), Some(&Value::from("nvlink")));
    }

    #[test]
    fn has_shared_is_config_identity() {
        // MoE and non-MoE calls must resolve to different cache instances.
        assert_ne!(config(true).identity(), config(false).identity());
        assert_eq!(config(true).compute_dtype(), Some(DType::Bf16));
    }
}
