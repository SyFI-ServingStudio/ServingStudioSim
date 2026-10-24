//! Gated DeltaNet fused causal convolution, q/k normalization and gates.
//!
//! One launch takes a GDN layer's packed q|k|v projection rows to the
//! recurrence's inputs: each token's depthwise causal convolution over its own
//! window (state rows of its stream, then earlier rows of the launch), SiLU,
//! L2-normalized q and k, v, the decay and beta. The head geometry, filter
//! width and dtype are static config identity; the launch's row count
//! `num_tokens` is the only runtime axis. Streams do not enter: the window
//! table and per-stream state offsets measured within noise of one stream.
//!
//! Backend: `tensorfold_triton` (TensorFold's Triton `_gdn_pre`, bf16 only).

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnConvNormGatesKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qk_heads: Dim,
    pub num_value_heads: Dim,
    pub key_head_dim: Dim,
    pub value_head_dim: Dim,
    pub kernel_size: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct GdnConvNormGatesKernelInput {
    pub num_tokens: u32,
}

pub struct GdnConvNormGatesSpec;

impl KernelSpec for GdnConvNormGatesSpec {
    type Config = GdnConvNormGatesKernelConfig;
    type Input = GdnConvNormGatesKernelInput;

    const KIND: KernelKind = "gdn_conv_norm_gates";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // One small program per (token, head): time is close to linear in
        // rows past a few waves. Dense through the verify range (16 streams x
        // 16 rows), then the prefill range to a 4096-row chunk, and one point
        // past it for extrapolation.
        let rows: Vec<u32> = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256]
            .into_iter()
            .chain([320, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192])
            .collect();
        SweepGrid::new(vec![Axis::values(rows)])
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
                .with("num_qk_heads", config.num_qk_heads.get())
                .with("num_value_heads", config.num_value_heads.get())
                .with("key_head_dim", config.key_head_dim.get())
                .with("value_head_dim", config.value_head_dim.get())
                .with("kernel_size", config.kernel_size.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(GdnConvNormGatesKernel, GdnConvNormGatesSpec);

#[cfg(test)]
mod tests {
    use super::{GdnConvNormGatesKernelConfig, GdnConvNormGatesKernelInput, GdnConvNormGatesSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> GdnConvNormGatesKernelConfig {
        GdnConvNormGatesKernelConfig {
            backends: vec!["tensorfold_triton"],
            gpu_name: "NVIDIA GB10".to_string(),
            num_qk_heads: 16.into(),
            num_value_heads: 48.into(),
            key_head_dim: 128.into(),
            value_head_dim: 128.into(),
            kernel_size: 4.into(),
            dtype: DType::Bf16,
        }
    }

    /// The payload must carry exactly the Python `GdnConvNormGatesArgs` fields
    /// plus `backend`; a missing or renamed field misses every DB row.
    #[test]
    fn enumerate_forwards_the_python_args_exactly() {
        let cfg = config();
        let grid = GdnConvNormGatesSpec::sweep_grid(&cfg);
        let payloads = GdnConvNormGatesSpec::enumerate(&cfg, &grid, "tensorfold_triton");
        assert_eq!(payloads.len(), grid.axes()[0].len());
        let fields = payloads[0].fields();
        let mut names: Vec<&str> = fields.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "dtype",
                "kernel_size",
                "key_head_dim",
                "num_qk_heads",
                "num_tokens",
                "num_value_heads",
                "value_head_dim"
            ]
        );
        assert_eq!(fields.get("num_tokens"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("num_value_heads"), Some(&Value::from(48_u32)));
        assert_eq!(fields.get("kernel_size"), Some(&Value::from(4_u32)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
    }

    /// Verify rounds reach 16 streams x 16 rows and prefill chunks 4096 rows;
    /// the axis must cover both without a gap the linear cache would smooth.
    #[test]
    fn sweep_spans_verify_and_prefill_rows() {
        let grid = GdnConvNormGatesSpec::sweep_grid(&config());
        let axis = &grid.axes()[0];
        assert_eq!(axis.first(), Some(&1.0));
        assert!(axis.contains(&256.0) && axis.contains(&4096.0));
        assert_eq!(axis.last(), Some(&8192.0));
        assert!(axis.windows(2).all(|w| w[0] < w[1]));
        assert!(GdnConvNormGatesSpec::infeasible_mask(&config(), &grid).is_empty());
    }

    #[test]
    fn input_is_the_row_count_and_logs_as_such() {
        let input = GdnConvNormGatesKernelInput { num_tokens: 114 };
        assert_eq!(&*input.coords(), &[114.0]);
        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_tokens": 114})
        );
    }
}
