//! Gated DeltaNet verify recurrence with the gated output RMSNorm fused in
//! (vLLM's `fused_gdn_decode_post_conv_mtp`, a pure MTP verify step).
//!
//! Model geometry, dtypes and the step's tokens per request (accepted token plus
//! drafts, at most 8) are static config identity. Verified requests are the sole
//! runtime interpolation axis; the profile arg `num_tokens` is
//! `tokens_per_request · batch_size`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::gdn_recurrent_decode::verify_batch_axis;
use crate::timing::sweep::SweepGrid;
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnRecurrentDecodeGatedNormKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qk_heads: Dim,
    pub num_value_heads: Dim,
    pub key_head_dim: Dim,
    pub value_head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
    pub state_dtype: DType,
    /// The accepted token plus drafts (1..=8).
    pub tokens_per_request: Dim,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct GdnRecurrentDecodeGatedNormKernelInput {
    pub batch_size: u32,
}

/// Most tokens per request the kernel takes (`kMaxMtpTokens`, vLLM
/// `csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu`).
const MAX_MTP_TOKENS: u32 = 8;

pub struct GdnRecurrentDecodeGatedNormSpec;

impl KernelSpec for GdnRecurrentDecodeGatedNormSpec {
    type Config = GdnRecurrentDecodeGatedNormKernelConfig;
    type Input = GdnRecurrentDecodeGatedNormKernelInput;

    const KIND: KernelKind = "gdn_recurrent_decode_gated_norm";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let q = config.tokens_per_request.get();
        assert!(
            (1..=MAX_MTP_TOKENS).contains(&q),
            "tokens_per_request must be in 1..={MAX_MTP_TOKENS}, got {q}"
        );
        // One CTA per (request, value head) at every q, so dense over 1..8 even at q = 1.
        SweepGrid::new(vec![verify_batch_axis()])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        let per_request = config.tokens_per_request.get();
        grid.expand_1d(|batch_size| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", batch_size as u32)
                .with("num_tokens", batch_size as u32 * per_request)
                .with("num_qk_heads", config.num_qk_heads.get())
                .with("num_value_heads", config.num_value_heads.get())
                .with("key_head_dim", config.key_head_dim.get())
                .with("value_head_dim", config.value_head_dim.get())
                .with("dtype", config.dtype.as_str())
                .with("state_dtype", config.state_dtype.as_str())
        })
    }
}

register_kernel!(
    GdnRecurrentDecodeGatedNormKernel,
    GdnRecurrentDecodeGatedNormSpec
);

#[cfg(test)]
mod tests {
    use super::{GdnRecurrentDecodeGatedNormKernelConfig, GdnRecurrentDecodeGatedNormSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;
    use serde_json::Value;

    fn config(tokens_per_request: u32) -> GdnRecurrentDecodeGatedNormKernelConfig {
        GdnRecurrentDecodeGatedNormKernelConfig {
            backends: vec!["vllm_cuda"],
            gpu_name: "NVIDIA B200".to_string(),
            num_qk_heads: 16.into(),
            num_value_heads: 48.into(),
            key_head_dim: 128.into(),
            value_head_dim: 128.into(),
            dtype: DType::Bf16,
            state_dtype: DType::Fp32,
            tokens_per_request: tokens_per_request.into(),
        }
    }

    /// Catches a verify width the kernel rejects (`1 <= S <= 8`) reaching the
    /// profiler as rows that can never be measured.
    #[test]
    #[should_panic(expected = "tokens_per_request")]
    fn rejects_more_tokens_per_request_than_the_kernel_takes() {
        GdnRecurrentDecodeGatedNormSpec::sweep_grid(&config(9));
    }

    #[test]
    #[should_panic(expected = "tokens_per_request")]
    fn rejects_zero_tokens_per_request() {
        GdnRecurrentDecodeGatedNormSpec::sweep_grid(&config(0));
    }

    /// The payload must carry exactly the Python args (num_tokens = q · batch).
    #[test]
    fn enumerate_emits_the_python_args_with_num_tokens_per_request() {
        let cfg = config(4);
        let grid = GdnRecurrentDecodeGatedNormSpec::sweep_grid(&cfg);
        let payloads = GdnRecurrentDecodeGatedNormSpec::enumerate(&cfg, &grid, "vllm_cuda");
        assert_eq!(payloads.len(), 13);
        let last = payloads[12].fields();
        let names: Vec<&str> = last.keys().map(String::as_str).collect();
        assert_eq!(
            names,
            [
                "backend",
                "batch_size",
                "dtype",
                "key_head_dim",
                "num_qk_heads",
                "num_tokens",
                "num_value_heads",
                "state_dtype",
                "value_head_dim",
            ]
        );
        assert_eq!(last["batch_size"], Value::from(256_u32));
        assert_eq!(last["num_tokens"], Value::from(1024_u32));
    }
}
