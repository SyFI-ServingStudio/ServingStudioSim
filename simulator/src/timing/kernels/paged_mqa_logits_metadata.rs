//! DeepGEMM's schedule for the dense paged decode indexer logits on SM100
//! (`deep_gemm.get_paged_mqa_logits_metadata(..., indices=...)`, one
//! single-CTA `sm100_paged_mqa_logits_metadata` launch).
//!
//! DeepInfer caches it per compress ratio for the step (`index_attn.py`
//! `_get_decode_metadata`), so it runs once per ratio, not per index layer.
//! The coordinate is the request count; the CTA's work is per row.

use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::sparse_mqa_logits_decode::DECODE_BATCH_POINTS;
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PagedMqaLogitsMetadataKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Query rows per request (verify width).
    pub next_n: u32,
    /// Index keys per cache page.
    pub block_size: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct PagedMqaLogitsMetadataKernelInput {
    pub batch_size: u32,
}

pub struct PagedMqaLogitsMetadataSpec;

impl KernelSpec for PagedMqaLogitsMetadataSpec {
    type Config = PagedMqaLogitsMetadataKernelConfig;
    type Input = PagedMqaLogitsMetadataKernelInput;

    const KIND: KernelKind = "paged_mqa_logits_metadata";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::values(DECODE_BATCH_POINTS)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            matches!(config.block_size, 32 | 64 | 128),
            "SM100 paged MQA logits pages hold 32, 64 or 128 keys"
        );
        anyhow::ensure!(config.next_n >= 1, "next_n must be positive");
        Ok(())
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
                .with("next_n", config.next_n)
                .with("block_size", config.block_size)
        })
    }
}

register_kernel!(PagedMqaLogitsMetadataKernel, PagedMqaLogitsMetadataSpec);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_emits_every_python_arg() {
        let config = PagedMqaLogitsMetadataKernelConfig {
            backends: vec!["deepgemm_sm100"],
            gpu_name: "NVIDIA B200".to_string(),
            next_n: 6,
            block_size: 64,
        };
        assert!(PagedMqaLogitsMetadataSpec::validate_config(&config).is_ok());
        let grid = SweepGrid::new(vec![Axis::values([64])]);
        let payloads = PagedMqaLogitsMetadataSpec::enumerate(&config, &grid, "deepgemm_sm100");
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(names, ["backend", "batch_size", "block_size", "next_n"]);
    }
}
