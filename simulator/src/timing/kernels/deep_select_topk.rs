//! DeepSelect's bf16 row top-k (`deep_select.interface.topk`, one
//! `topk_select_bf16_normal::topk_kernel` launch, one CTA per row).
//!
//! DeepInfer's prefill indexer calls it on every logits chunk (top 512) and on
//! the candidate source layer's block scores (top 2048). The coordinates are
//! the row count and the valid scores per row; a call whose rows end at
//! different points (causal windows) is queried at their mean length.
//!
//! The row axis brackets DeepSelect's dispatch: a single wave of CTAs (rows
//! <= SM count, 148 on SM100) runs the 512-thread kernel, more rows the
//! 256-thread one, and at most six rows of at least 512K columns the cluster
//! kernel (`csrc/api.cpp`).

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

const ROW_POINTS: [u32; 16] = [
    1, 6, 7, 16, 32, 64, 128, 148, 149, 192, 256, 512, 1024, 2048, 4096, 8192,
];

const ROW_LEN_POINTS: [u32; 14] = [
    512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072, 196608, 262144, 524288,
    1048576,
];

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeepSelectTopkKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub top_k: u32,
    #[compute_dtype]
    pub dtype: DType,
    /// Longest row the deployment selects from. Bounds the row-length axis only.
    pub max_row_len: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DeepSelectTopkKernelInput {
    pub num_rows: u32,
    pub row_len: u32,
}

pub struct DeepSelectTopkSpec;

impl KernelSpec for DeepSelectTopkSpec {
    type Config = DeepSelectTopkKernelConfig;
    type Input = DeepSelectTopkKernelInput;

    const KIND: KernelKind = "deep_select_topk";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let cut = ROW_LEN_POINTS
            .iter()
            .position(|&n| n >= config.max_row_len)
            .map_or(ROW_LEN_POINTS.len(), |i| i + 1);
        SweepGrid::new(vec![
            Axis::values(ROW_POINTS),
            Axis::values(ROW_LEN_POINTS[..cut].iter().copied()),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // One CTA streams each row: rows x length is the work.
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            (1..=4096).contains(&config.top_k),
            "DeepSelect bf16 top-k takes top_k in 1..=4096"
        );
        anyhow::ensure!(config.max_row_len >= 1, "max_row_len must be positive");
        Ok(())
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|num_rows, row_len| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_rows", num_rows as u32)
                .with("row_len", row_len as u32)
                .with("top_k", config.top_k)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(DeepSelectTopkKernel, DeepSelectTopkSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(max_row_len: u32) -> DeepSelectTopkKernelConfig {
        DeepSelectTopkKernelConfig {
            backends: vec!["deep_select"],
            gpu_name: "NVIDIA B200".to_string(),
            top_k: 512,
            dtype: DType::Bf16,
            max_row_len,
        }
    }

    #[test]
    fn enumerate_emits_every_python_arg() {
        let grid = SweepGrid::new(vec![Axis::values([4096]), Axis::values([65536])]);
        let payloads = DeepSelectTopkSpec::enumerate(&config(131072), &grid, "deep_select");
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(names, ["backend", "dtype", "num_rows", "row_len", "top_k"]);
    }

    /// The single-wave / multi-wave dispatch edge must be a pair of grid
    /// points, or interpolation smears the switch across 128..192 rows.
    #[test]
    fn row_axis_brackets_the_sm100_wave_edge() {
        let grid = DeepSelectTopkSpec::sweep_grid(&config(131072));
        let rows = &grid.axes()[0];
        assert!(rows.contains(&148.0) && rows.contains(&149.0));
        assert_eq!(grid.axes()[1].last(), Some(&131072.0));
    }
}
