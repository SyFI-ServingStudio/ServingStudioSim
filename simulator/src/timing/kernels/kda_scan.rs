//! KDA (Kimi Delta Attention) chunked linear-attention scan kernel: the fused
//! gated delta-rule state update + intra-chunk attention over per-head `S ∈
//! R^{head_dim × head_dim}` recurrent state (Kimi-K3's linear-attention layers).
//!
//! Static identity captures the physical scan shape (`num_heads`, `head_dim`,
//! the short-conv width feeding the scan, and the compute dtype). Runtime is
//! the total token count across the batch — the chunked scan walks all tokens
//! once, so cost scales with the pooled token total (per-request chunk
//! boundaries are second-order and folded into the 1-D curve), mirroring
//! `kv_cache_append`'s `num_tokens` identity.
//!
//! **Unprofiled for now**: the kernel is registered (Rust `KernelSpec` +
//! `register_kernel!`, enumerable via `emit-backends` / `kernel-query grid`)
//! but has no `profile.db` rows and no Python runner yet — the `triton` backend
//! rows will be measured in a later profiling phase (fla-style chunked kernel).

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Deserialize)]
pub struct KdaScanKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: u32,
    pub head_dim: u32,
    /// Width of the short causal conv feeding q/k/v into the scan (identity
    /// only — the conv itself is billed as a separate elementwise leaf).
    pub short_conv_kernel_size: u32,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct KdaScanKernelInput {
    /// Total tokens scanned this call (prefill chunk tokens + one per decode).
    pub total_tokens: u32,
}

pub struct KdaScanSpec;

impl KernelSpec for KdaScanSpec {
    type Config = KdaScanKernelConfig;
    type Input = KdaScanKernelInput;

    const KIND: KernelKind = "kda_scan";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // Decode batches run the scan on a handful of tokens (one per request),
        // so cover 1..32 below the shared token axis' floor — same shape as
        // `kv_cache_append`.
        SweepGrid::new(vec![Axis::chain([Axis::pow2(0, 4), Axis::token_axis()])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|total_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_heads", config.num_heads)
                .with("head_dim", config.head_dim)
                .with("short_conv_kernel_size", config.short_conv_kernel_size)
                .with("dtype", config.dtype.as_str())
                .with("total_tokens", total_tokens as u32)
        })
    }
}

register_kernel!(KdaScanKernel, KdaScanSpec);

#[cfg(test)]
mod tests {
    use super::{KdaScanKernelConfig, KdaScanKernelInput, KdaScanSpec};
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> KdaScanKernelConfig {
        KdaScanKernelConfig {
            backends: vec!["triton"],
            gpu_name: "NVIDIA H200".to_string(),
            num_heads: 96,
            head_dim: 128,
            short_conv_kernel_size: 4,
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_describes_the_scan_shape() {
        assert_eq!(
            config().describe_config(),
            r#"backends=["triton"] gpu_name="NVIDIA H200" num_heads=96 head_dim=128 short_conv_kernel_size=4 dtype=Bf16"#
        );
        assert_eq!(config().compute_dtype(), Some(DType::Bf16));
    }

    #[test]
    fn input_is_the_pooled_token_total() {
        let input = KdaScanKernelInput { total_tokens: 512 };
        assert_eq!(&*input.coords(), &[512.0]);
        assert_eq!(KdaScanKernelInput::coord_field_names(), &["total_tokens"]);
    }

    #[test]
    fn grid_covers_decode_and_prefill_sizes() {
        let grid = KdaScanSpec::sweep_grid(&config());
        assert_eq!(grid.axes().len(), 1);
        assert_eq!(&grid.axes()[0][..5], &[1.0, 2.0, 4.0, 8.0, 16.0]);
        assert!(grid.axes()[0].contains(&2048.0));
        assert_eq!(KdaScanSpec::cache_kind("triton"), CacheKind::Cache1DLinear);
    }

    #[test]
    fn enumerate_emits_backend_plus_schema_fields() {
        let cfg = config();
        let grid = KdaScanSpec::sweep_grid(&cfg);
        let payload = &KdaScanSpec::enumerate(&cfg, &grid, "triton")[0];
        let fields = payload.fields();
        assert_eq!(fields.len(), 6);
        assert_eq!(fields.get("backend"), Some(&Value::from("triton")));
        assert_eq!(fields.get("num_heads"), Some(&Value::from(96_u32)));
        assert_eq!(fields.get("head_dim"), Some(&Value::from(128_u32)));
        assert_eq!(
            fields.get("short_conv_kernel_size"),
            Some(&Value::from(4_u32))
        );
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("total_tokens"), Some(&Value::from(1_u32)));
    }
}
