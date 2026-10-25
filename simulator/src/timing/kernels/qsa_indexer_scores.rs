//! QSA indexer scores (bench `indexer_scores`).
//!
//! Neuron backends: `neuron_torch`, `neuron_torch_bf16`. Contract:
//! `doc/pr/trn2-flashnext/L1_CONTRACT.md`; shapes are per rank.
//!
//! Cache axes depend on `mode`: prefill `(num_rows, prior = context_len -
//! num_rows)` over `QSA_PREFILL_ROWS x QSA_PRIOR`, decode `(num_rows,
//! context_len)` over `QSA_DECODE_ROWS x QSA_CONTEXT` (`cache_coords`).
//! `cache_kind` cannot see `mode`, so both use `Extrapolation::Weighted`; on
//! the two-point prior axis its slope fit has too few top-band cells and it
//! holds the boundary, which is the contract's prefill `Clamp`
//! (`neuron_grid::qsa_cache_tests` pins that).

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_grid;
use crate::timing::sweep::SweepGrid;
use crate::timing::{Coords, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QsaIndexerScoresKernelConfig {
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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QsaIndexerScoresKernelInput {
    /// Prefill: tokens of the request chunk; decode: requests (one row each).
    pub num_rows: u32,
    /// Prefill: prior + num_rows (the chunk's last row's visible context); decode: KV length including the new token.
    pub context_len: u32,
}

/// Mode-free projection `(num_rows, context_len)`, the decode cache axes.
/// The prefill re-axis to the prior needs `mode` and lives in
/// `KernelSpec::cache_coords`.
impl SweepCoords for QsaIndexerScoresKernelInput {
    fn coords(&self) -> Coords {
        Coords::new([f64::from(self.num_rows), f64::from(self.context_len)])
    }
    fn coord_field_names() -> &'static [&'static str] {
        &["num_rows", "prior_or_context_len"]
    }
}

pub struct QsaIndexerScoresSpec;

impl KernelSpec for QsaIndexerScoresSpec {
    type Config = QsaIndexerScoresKernelConfig;
    type Input = QsaIndexerScoresKernelInput;
    const KIND: KernelKind = "qsa_indexer_scores";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        neuron_grid::assert_neuron(Self::KIND, &config.backends);
        neuron_grid::qsa_context_grid(Self::KIND, &config.mode)
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        let prefill = neuron_grid::qsa_is_prefill(Self::KIND, &config.mode);
        Coords::new([
            f64::from(input.num_rows),
            neuron_grid::qsa_cache_axis1(prefill, input.num_rows, input.context_len),
        ])
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        let prefill = neuron_grid::qsa_is_prefill(Self::KIND, &config.mode);
        grid.expand_2d(|rows, axis1| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_rows", rows.round() as u32)
                .with(
                    "context_len",
                    neuron_grid::qsa_payload_context_len(prefill, rows, axis1),
                )
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

register_kernel!(QsaIndexerScoresKernel, QsaIndexerScoresSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(mode: &str, backends: Vec<&'static str>) -> QsaIndexerScoresKernelConfig {
        QsaIndexerScoresKernelConfig {
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
        let grid = QsaIndexerScoresSpec::sweep_grid(&cfg);
        let payloads = QsaIndexerScoresSpec::enumerate(&cfg, &grid, "neuron_torch");
        assert_eq!(payloads.len(), 8);
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "compress_ratio",
                "context_len",
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
        // Prefill cells carry context = prior + rows; decode cells the KV length.
        let last = payloads.last().unwrap().fields();
        assert_eq!(last["num_rows"], 8192);
        assert_eq!(last["context_len"], 8192 + 32768);
        let cfg = config("decode", vec!["neuron_torch"]);
        let grid = QsaIndexerScoresSpec::sweep_grid(&cfg);
        let payloads = QsaIndexerScoresSpec::enumerate(&cfg, &grid, "neuron_torch");
        assert_eq!(payloads.len(), 12);
        assert_eq!(payloads[1].fields()["context_len"], 32768);
    }

    /// Catches a prefill input whose cache coordinate is not its prior.
    #[test]
    fn prefill_caches_on_prior_and_decode_on_context() {
        let input = QsaIndexerScoresKernelInput {
            num_rows: 512,
            context_len: 4608,
        };
        let prefill =
            QsaIndexerScoresSpec::cache_coords(&config("prefill", vec!["neuron_torch"]), &input);
        assert_eq!(&*prefill, &[512.0, 4096.0]);
        let decode =
            QsaIndexerScoresSpec::cache_coords(&config("decode", vec!["neuron_torch"]), &input);
        assert_eq!(&*decode, &[512.0, 4608.0]);
    }

    #[test]
    #[should_panic(expected = "mode")]
    fn rejects_unknown_mode() {
        QsaIndexerScoresSpec::sweep_grid(&config("mixed", vec!["neuron_torch"]));
    }
}
