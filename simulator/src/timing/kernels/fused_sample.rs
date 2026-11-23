//! DeepInfer's fused sampler (`tilelang_ops/fused_sample.py` `fused_sample`):
//! a per-row log-sum-exp launch, then one CTA per row that applies the
//! temperature, finds the top-p / top-k thresholds by radix selection, writes
//! the filtered log-softmax and draws a seeded sample, with rejection sampling
//! of the row's draft token on verify calls.
//!
//! Row width, rejection sampling and which filters are active pick the code
//! path and live in the Config; the measured filters change the time by up to
//! 1.8x at one shape, the temperature value does not. Only the row count
//! moves with the batch: one interpolated curve over `num_rows`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

const ROW_POINTS: [u32; 20] = [
    1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
];

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FusedSampleKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Logits in one row: the force_topk width, or the vocabulary.
    pub num_columns: Dim,
    /// Verify call: rejection-samples each row's draft token.
    pub reject_sampling: bool,
    /// The rows' top-p filter is active (top_p < 1).
    pub top_p_filter: bool,
    /// The rows' top-k filter is active (top_k > 0).
    pub top_k_filter: bool,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct FusedSampleKernelInput {
    /// Logit rows sampled, one per sampled position.
    pub num_rows: u32,
}

pub struct FusedSampleSpec;

impl KernelSpec for FusedSampleSpec {
    type Config = FusedSampleKernelConfig;
    type Input = FusedSampleKernelInput;

    const KIND: KernelKind = "fused_sample";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::values(ROW_POINTS)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.num_columns.get() >= 32,
            "fused_sample needs num_columns >= 32"
        );
        Ok(())
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_rows| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_rows", num_rows as u32)
                .with("num_columns", config.num_columns.get())
                .with("reject_sampling", config.reject_sampling)
                .with("top_p_filter", config.top_p_filter)
                .with("top_k_filter", config.top_k_filter)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(FusedSampleKernel, FusedSampleSpec);

#[cfg(test)]
mod tests {
    use super::{FusedSampleKernelConfig, FusedSampleKernelInput, FusedSampleSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;

    fn config() -> FusedSampleKernelConfig {
        FusedSampleKernelConfig {
            backends: vec!["tilelang"],
            gpu_name: "NVIDIA B200".to_string(),
            num_columns: 1024.into(),
            reject_sampling: true,
            top_p_filter: true,
            top_k_filter: false,
            dtype: DType::Fp32,
        }
    }

    /// Verify and draft calls, and filtered and unfiltered rows, take different
    /// code paths; they must never share a curve.
    #[test]
    fn code_path_flags_are_part_of_the_cache_identity() {
        let cfg = config();
        assert_eq!(cfg.compute_dtype(), Some(DType::Fp32));
        for change in [
            |c: &mut FusedSampleKernelConfig| c.reject_sampling = false,
            |c: &mut FusedSampleKernelConfig| c.top_p_filter = false,
            |c: &mut FusedSampleKernelConfig| c.top_k_filter = true,
            |c: &mut FusedSampleKernelConfig| c.num_columns = 129280.into(),
        ] {
            let mut other = config();
            change(&mut other);
            assert_ne!(other, cfg);
        }
    }

    /// The payload must carry exactly the Python `FusedSampleArgs` fields.
    #[test]
    fn payload_carries_exactly_the_python_args() {
        let cfg = config();
        let grid = FusedSampleSpec::sweep_grid(&cfg);
        let payloads = FusedSampleSpec::enumerate(&cfg, &grid, "tilelang");
        assert_eq!(payloads.len(), 20);
        for p in &payloads {
            let names: Vec<&str> = p.fields().keys().map(String::as_str).collect();
            assert_eq!(
                names,
                [
                    "backend",
                    "dtype",
                    "num_columns",
                    "num_rows",
                    "reject_sampling",
                    "top_k_filter",
                    "top_p_filter"
                ]
            );
        }
        let input = FusedSampleKernelInput { num_rows: 64 };
        assert_eq!(&*input.coords(), &[64.0]);
    }
}
