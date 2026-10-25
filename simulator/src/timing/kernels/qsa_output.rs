//! QSA output projection.
//!
//! Neuron backends: `neuron_torch`. Contract:
//! `doc/pr/trn2-flashnext/L1_CONTRACT.md`; shapes are per rank.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_grid;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QsaOutputKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// `prefill` (one request chunk) or `decode` (one row per request).
    pub mode: String,
    /// Model hidden width.
    pub hidden_size: u32,
    /// QSA query heads on this rank (24 / tp).
    pub num_q_heads: u32,
    /// QSA KV heads on this rank (2 at tp 1, else 1).
    pub num_kv_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Indexer heads.
    pub index_heads: u32,
    /// Indexer head dimension.
    pub index_head_dim: u32,
    /// Selection budget, in tokens.
    pub index_topk_tokens: u32,
    /// Indexer key compression ratio.
    pub compress_ratio: u32,
    /// Activation dtype.
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QsaOutputKernelInput {
    /// Prefill: tokens of the request chunk; decode: requests (one row each).
    pub num_rows: u32,
}

pub struct QsaOutputSpec;

impl KernelSpec for QsaOutputSpec {
    type Config = QsaOutputKernelConfig;
    type Input = QsaOutputKernelInput;
    const KIND: KernelKind = "qsa_output";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        neuron_grid::assert_neuron(Self::KIND, &config.backends);
        neuron_grid::qsa_rows_grid(Self::KIND, &config.mode)
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_rows| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_rows", num_rows.round() as u32)
                .with("mode", config.mode.clone())
                .with("hidden_size", config.hidden_size)
                .with("num_q_heads", config.num_q_heads)
                .with("num_kv_heads", config.num_kv_heads)
                .with("head_dim", config.head_dim)
                .with("index_heads", config.index_heads)
                .with("index_head_dim", config.index_head_dim)
                .with("index_topk_tokens", config.index_topk_tokens)
                .with("compress_ratio", config.compress_ratio)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(QsaOutputKernel, QsaOutputSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(mode: &str, backends: Vec<&'static str>) -> QsaOutputKernelConfig {
        QsaOutputKernelConfig {
            backends,
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            mode: mode.to_string(),
            hidden_size: 2560,
            num_q_heads: 6,
            num_kv_heads: 1,
            head_dim: 128,
            index_heads: 4,
            index_head_dim: 128,
            index_topk_tokens: 2048,
            compress_ratio: 4,
            dtype: DType::Bf16,
        }
    }

    /// Catches a payload field the Python args schema would reject, and a grid
    /// over the 500 feasible-coordinate ceiling.
    #[test]
    fn payload_matches_python_args_in_both_modes() {
        let cfg = config("prefill", vec!["neuron_torch"]);
        let grid = QsaOutputSpec::sweep_grid(&cfg);
        let payloads = QsaOutputSpec::enumerate(&cfg, &grid, "neuron_torch");
        assert_eq!(payloads.len(), 4);
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "compress_ratio",
                "dtype",
                "head_dim",
                "hidden_size",
                "index_head_dim",
                "index_heads",
                "index_topk_tokens",
                "mode",
                "num_kv_heads",
                "num_q_heads",
                "num_rows",
            ]
        );
        let cfg = config("decode", vec!["neuron_torch"]);
        let grid = QsaOutputSpec::sweep_grid(&cfg);
        assert_eq!(grid.axes()[0], [1.0, 8.0, 64.0, 256.0]);
        assert_eq!(cfg.mode, "decode");
    }

    #[test]
    #[should_panic(expected = "mode")]
    fn rejects_unknown_mode() {
        QsaOutputSpec::sweep_grid(&config("mixed", vec!["neuron_torch"]));
    }
}
