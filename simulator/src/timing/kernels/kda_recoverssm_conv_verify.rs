//! KDA RecoverSSM speculative verify with the short convolution fused in, one
//! `CuteRecoverSSMVerify` call (vLLM fork RecoverSSM KDA layer on SM10x).
//!
//! One persistent tensor-core launch per layer: the width-4 causal
//! convolution + SiLU over each request's conv history and window (written
//! back as the rolled window), then the RecoverSSM verify of the window from
//! the request's bf16 checkpoint, which is only read, writing the outputs and
//! the correction and key/gate records `kda_recoverssm_commit` folds in. It
//! replaces `gdn_causal_conv_decode` + `kda_recoverssm_verify`, the Triton
//! fallback. The fork selects it where it can run: SM10x, head dim 128, a
//! window of at most 8 tokens, and bf16 activations, conv state and
//! checkpoint, so the pool dtype is not an axis. Head geometry, window and
//! the conv taps' dtype are static identity; the number of verified requests
//! is the only runtime axis.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::kda_recoverssm_verify::{batch_axis, value_blocks};
use crate::timing::sweep::SweepGrid;
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KdaRecoverSSMConvVerifyKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub spec_query_len: Dim,
    #[compute_dtype]
    pub dtype: DType,
    /// Element type of the short convolution's taps.
    pub conv_weight_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct KdaRecoverSSMConvVerifyKernelInput {
    /// Spec-decode requests verified together.
    pub batch_size: u32,
}

pub struct KdaRecoverSSMConvVerifySpec;

impl KernelSpec for KdaRecoverSSMConvVerifySpec {
    type Config = KdaRecoverSSMConvVerifyKernelConfig;
    type Input = KdaRecoverSSMConvVerifyKernelInput;

    const KIND: KernelKind = "kda_recoverssm_conv_verify";

    /// The verify kind's batch axis: every batch while the (request, head,
    /// value block) work is a few waves deep, then powers of two to 512.
    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let programs = value_blocks(config.head_dim.get()) * config.num_heads.get();
        SweepGrid::new(vec![batch_axis(programs)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|batch_size| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", batch_size as u32)
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("spec_query_len", config.spec_query_len.get())
                .with("dtype", config.dtype.as_str())
                .with("conv_weight_dtype", config.conv_weight_dtype.as_str())
        })
    }
}

register_kernel!(KdaRecoverSSMConvVerifyKernel, KdaRecoverSSMConvVerifySpec);

#[cfg(test)]
mod tests {
    use super::{KdaRecoverSSMConvVerifyKernelConfig, KdaRecoverSSMConvVerifySpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use serde_json::Value;

    /// Catches payload keys that would miss the Python KernelArgs DB key, or
    /// conv taps' dtype not forwarded (the fork compiles a variant per dtype),
    /// or a pool dtype tagged as a KV axis the kind does not have.
    #[test]
    fn enumerate_emits_exact_python_fields() {
        let cfg = KdaRecoverSSMConvVerifyKernelConfig {
            backends: vec!["vllm_cute"],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads: 64.into(),
            head_dim: 128.into(),
            spec_query_len: 8.into(),
            dtype: DType::Bf16,
            conv_weight_dtype: DType::Fp32,
        };
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
        let grid = KdaRecoverSSMConvVerifySpec::sweep_grid(&cfg);
        let payloads = KdaRecoverSSMConvVerifySpec::enumerate(&cfg, &grid, "vllm_cute");
        assert_eq!(payloads.len(), 21);
        let names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            names,
            [
                "backend",
                "batch_size",
                "conv_weight_dtype",
                "dtype",
                "head_dim",
                "num_heads",
                "spec_query_len"
            ]
        );
        assert_eq!(payloads[20].fields()["batch_size"], Value::from(512_u32));
        assert_eq!(
            payloads[20].fields()["conv_weight_dtype"],
            Value::from("fp32")
        );
    }
}
