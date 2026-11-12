//! Sparse-indexer decode scoring: every query row of a decode or verify step
//! scores its request's compressed keys (ReLU-weighted MQA over the indexer
//! heads) into a padded logits row. The top-k that follows is its own kind.
//!
//! `logits_width` is the padded row the framework allocates, about
//! `max_model_len / compress_ratio` columns. It is a swept `Input` axis (and so
//! a DB key column) rather than `Config` identity: the kernel fills the
//! padding too, which measured 1.7x at 256 requests from 4800 to 32800 columns
//! on B200, and the width is a deployment value, so the grid sweeps it and any
//! deployment interpolates.
//!
//! Time is a sawtooth in the batch: the launch packs 1, 2, 4 or 8 column tiles
//! per program once `batch · ⌈width/64⌉` programs reach 16384, 32768 or 131072
//! (fork `ops/qsa_indexer.py:279-288`), so time rises with the batch and drops
//! at each switch (B200, 700 keys, width 32768: 9.0 µs at 28 requests, 5.9 µs
//! at 32). At power-of-two widths the switches fall on powers of two, so the
//! batch axis holds each power of two and the batch just below it. Time is flat
//! in the keys up to about 1024 and linear past 2048.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparseIndexerLogitsDecodeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub next_n: u32,
    pub num_heads: Dim,
    pub head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, Debug, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct SparseIndexerLogitsDecodeKernelInput {
    pub batch_size: u32,
    /// Compressed keys visible to every request.
    pub num_keys: u32,
    /// Padded logits row width in columns.
    pub logits_width: u32,
}

const BATCH_AXIS: [u32; 16] = [
    1, 2, 4, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512,
];
const KEYS_AXIS: [u32; 8] = [1, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
const WIDTH_AXIS: [u32; 5] = [1024, 4096, 16384, 32768, 65536];

pub struct SparseIndexerLogitsDecodeSpec;

impl KernelSpec for SparseIndexerLogitsDecodeSpec {
    type Config = SparseIndexerLogitsDecodeKernelConfig;
    type Input = SparseIndexerLogitsDecodeKernelInput;

    const KIND: KernelKind = "sparse_indexer_logits_decode";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values(BATCH_AXIS),
            Axis::values(KEYS_AXIS),
            Axis::values(WIDTH_AXIS),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_3d(|_batch, keys, width| keys > width)
    }

    /// Requests are independent programs: a batch past the grid runs as
    /// full-edge pieces.
    fn off_grid(
        _config: &Self::Config,
        input: &Self::Input,
        _backend: &'static str,
    ) -> OffGrid<Self::Input> {
        let max_batch = BATCH_AXIS[BATCH_AXIS.len() - 1];
        if input.batch_size <= max_batch {
            return OffGrid::Cache;
        }
        let mut pieces = Vec::new();
        let mut batch = input.batch_size;
        while batch > 0 {
            let piece = batch.min(max_batch);
            pieces.push(SparseIndexerLogitsDecodeKernelInput {
                batch_size: piece,
                ..input.clone()
            });
            batch -= piece;
        }
        OffGrid::Launches(pieces)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_3d(|batch_size, num_keys, logits_width| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", batch_size as u32)
                .with("num_keys", num_keys as u32)
                .with("logits_width", logits_width as u32)
                .with("next_n", config.next_n)
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(
    SparseIndexerLogitsDecodeKernel,
    SparseIndexerLogitsDecodeSpec
);

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SparseIndexerLogitsDecodeKernelConfig {
        SparseIndexerLogitsDecodeKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".into(),
            next_n: 4,
            num_heads: Dim::param("num_heads", 4),
            head_dim: Dim::param("head_dim", 128),
            dtype: DType::Fp8E4m3,
        }
    }

    #[test]
    fn grid_masks_keys_past_the_row_and_emits_python_args() {
        let cfg = config();
        let grid = SparseIndexerLogitsDecodeSpec::sweep_grid(&cfg);
        let mask = SparseIndexerLogitsDecodeSpec::infeasible_mask(&cfg, &grid);
        let feasible = mask.iter().filter(|&&m| !m).count();
        assert_eq!(feasible, 432);
        let specs = SparseIndexerLogitsDecodeSpec::enumerate(&cfg, &grid, "vllm_triton");
        let names: Vec<&str> = specs[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            names,
            [
                "backend",
                "batch_size",
                "dtype",
                "head_dim",
                "logits_width",
                "next_n",
                "num_heads",
                "num_keys"
            ]
        );
    }
}
