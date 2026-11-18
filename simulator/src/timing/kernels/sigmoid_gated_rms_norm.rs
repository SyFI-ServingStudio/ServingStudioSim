//! Sigmoid-gated RMSNorm: `y = rmsnorm(x) * weight * sigmoid(g)` per row.
//!
//! Hidden width and compute dtype are static config identity. The flattened row
//! count `m` is the sole runtime interpolation axis; a per-head output norm maps
//! it from tokens as `m = num_tokens * rank_local_heads`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SigmoidGatedRmsNormKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct SigmoidGatedRmsNormKernelInput {
    pub m: u32,
}

pub struct SigmoidGatedRmsNormSpec;

impl KernelSpec for SigmoidGatedRmsNormSpec {
    type Config = SigmoidGatedRmsNormKernelConfig;
    type Input = SigmoidGatedRmsNormKernelInput;

    const KIND: KernelKind = "sigmoid_gated_rms_norm";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![m_axis()])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|m| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("m", m as u32)
                .with("hidden", config.hidden.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

/// Flattened rows (tokens x rank-local heads): power-of-two up to 4096, then
/// every 2048 rows (32 tokens at 64 heads) to 65536, every 8192 to 131072, and
/// 2^18, 2^19 for an 8192-token prefill step at up to 64 heads.
fn m_axis() -> Vec<f64> {
    Axis::chain([
        Axis::pow2(3, 12),
        Axis::arithmetic(4096, 65536, 2048),
        Axis::arithmetic(65536, 131072, 8192),
        Axis::pow2(18, 19),
    ])
}

register_kernel!(SigmoidGatedRmsNormKernel, SigmoidGatedRmsNormSpec);

#[cfg(test)]
mod tests {
    use super::{SigmoidGatedRmsNormKernelConfig, SigmoidGatedRmsNormSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;
    use serde_json::Value;

    /// Catches a grid that misses the decode rows: every 32 tokens at 64 heads
    /// (and so every 64 at 32 heads) up to 1024 tokens is a measured point.
    #[test]
    fn grid_covers_decode_rows_densely_and_enumerates_the_python_fields() {
        let cfg = SigmoidGatedRmsNormKernelConfig {
            backends: vec!["vllm_fla_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden: 128.into(),
            dtype: DType::Bf16,
        };
        let grid = SigmoidGatedRmsNormSpec::sweep_grid(&cfg);
        let m = &grid.axes()[0];
        assert!(m.windows(2).all(|w| w[0] < w[1]));
        for tokens in (64..=1024).step_by(32) {
            assert!(
                m.contains(&f64::from(tokens * 64)),
                "{tokens} tokens at 64 heads"
            );
        }
        assert_eq!((m[0], *m.last().unwrap()), (8.0, 524288.0));
        let payloads = SigmoidGatedRmsNormSpec::enumerate(&cfg, &grid, "vllm_fla_triton");
        let first = payloads[0].fields();
        assert_eq!(first.get("m"), Some(&Value::from(8_u32)));
        assert_eq!(first.get("hidden"), Some(&Value::from(128_u32)));
        assert_eq!(first.get("dtype"), Some(&Value::from("bf16")));
    }
}
