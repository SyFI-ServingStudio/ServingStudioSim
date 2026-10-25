//! MoE routed experts on the critical-path EP rank, with real layer-0 routing
//! (bench `routing == "real"`).
//!
//! Neuron backends: `neuron_auto`, `neuron_nki`, `neuron_nki_tkg`, `neuron_nki_tkg_all`, `neuron_nki_cte`, `neuron_nki_cte_cf2`, `neuron_torch_gather_w`, `neuron_torch_dense`, `neuron_torch_gather`, `neuron_torch_onehot`. Contract:
//! `doc/pr/trn2-flashnext/L1_CONTRACT.md`; shapes are per rank.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_grid;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

/// Backends that cannot run part of `TOKENS`: `(backend, max_tokens, min_tokens)`.
const TOKEN_LIMITS: [(&str, u32, u32); 7] = [
    ("neuron_nki_tkg", 128, 1),
    ("neuron_torch_gather_w", 16, 1),
    ("neuron_torch_dense", 256, 1),
    ("neuron_nki_tkg_all", 512, 1),
    ("neuron_nki_cte", u32::MAX, 128),
    ("neuron_nki_cte_cf2", u32::MAX, 128),
    ("neuron_torch_onehot", u32::MAX, 128),
];

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeRoutedExpertsKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Model hidden width.
    pub hidden_size: u32,
    /// Routed experts in the model.
    pub num_experts: u32,
    /// Experts per token.
    pub top_k: u32,
    /// Routed experts on this EP rank (num_experts / ep).
    pub local_experts: u32,
    /// Expert intermediate width on this rank.
    pub intermediate_size: u32,
    /// Activation dtype.
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeRoutedExpertsKernelInput {
    /// Token rows of the call.
    pub num_tokens: u32,
}

pub struct MoeRoutedExpertsSpec;

impl KernelSpec for MoeRoutedExpertsSpec {
    type Config = MoeRoutedExpertsKernelConfig;
    type Input = MoeRoutedExpertsKernelInput;
    const KIND: KernelKind = "moe_routed_experts";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        neuron_grid::assert_neuron(Self::KIND, &config.backends);
        neuron_grid::grid_1d(neuron_grid::TOKENS)
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    /// With exactly one backend, the `TOKENS` cells that backend's kernel cannot
    /// run. A multi-backend config profiles the whole grid for every backend.
    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        let [backend] = config.backends.as_slice() else {
            return Vec::new();
        };
        let Some(&(_, max_tokens, min_tokens)) =
            TOKEN_LIMITS.iter().find(|(name, _, _)| name == backend)
        else {
            return Vec::new();
        };
        grid.expand_1d(|tokens| {
            let tokens = tokens.round() as u32;
            tokens > max_tokens || tokens < min_tokens
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens.round() as u32)
                .with("hidden_size", config.hidden_size)
                .with("num_experts", config.num_experts)
                .with("top_k", config.top_k)
                .with("local_experts", config.local_experts)
                .with("intermediate_size", config.intermediate_size)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(MoeRoutedExpertsKernel, MoeRoutedExpertsSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(backends: Vec<&'static str>) -> MoeRoutedExpertsKernelConfig {
        MoeRoutedExpertsKernelConfig {
            backends,
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            hidden_size: 2560,
            num_experts: 512,
            top_k: 10,
            local_experts: 128,
            intermediate_size: 160,
            dtype: DType::Bf16,
        }
    }

    /// Catches a payload field the Python args schema would reject, and a grid
    /// over the 500 feasible-coordinate ceiling.
    #[test]
    fn payload_matches_python_args() {
        let cfg = config(vec!["neuron_auto"]);
        let grid = MoeRoutedExpertsSpec::sweep_grid(&cfg);
        let payloads = MoeRoutedExpertsSpec::enumerate(&cfg, &grid, "neuron_auto");
        assert_eq!(payloads.len(), 14);
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "dtype",
                "hidden_size",
                "intermediate_size",
                "local_experts",
                "num_experts",
                "num_tokens",
                "top_k",
            ]
        );
    }

    /// Catches a single-backend config sending the profiler token counts its
    /// kernel cannot run, and a multi-backend config dropping cells.
    #[test]
    fn single_backend_masks_unrunnable_token_counts() {
        let feasible = |backends: Vec<&'static str>| -> Vec<u32> {
            let cfg = config(backends);
            let grid = MoeRoutedExpertsSpec::sweep_grid(&cfg);
            let mask = MoeRoutedExpertsSpec::infeasible_mask(&cfg, &grid);
            let tokens = grid.expand_1d(|t| t as u32);
            if mask.is_empty() {
                return tokens;
            }
            tokens
                .into_iter()
                .zip(mask)
                .filter_map(|(t, drop)| (!drop).then_some(t))
                .collect()
        };
        assert_eq!(feasible(vec!["neuron_nki_tkg"]).last(), Some(&128));
        assert_eq!(feasible(vec!["neuron_torch_gather_w"]).last(), Some(&16));
        assert_eq!(feasible(vec!["neuron_torch_dense"]).last(), Some(&256));
        assert_eq!(feasible(vec!["neuron_nki_tkg_all"]).last(), Some(&512));
        for backend in [
            "neuron_nki_cte",
            "neuron_nki_cte_cf2",
            "neuron_torch_onehot",
        ] {
            let tokens = feasible(vec![backend]);
            assert_eq!(tokens.first(), Some(&128));
            assert_eq!(tokens.len(), 7);
        }
        assert_eq!(feasible(vec!["neuron_auto"]).len(), 14);
        assert_eq!(feasible(vec!["neuron_nki_tkg", "neuron_nki_cte"]).len(), 14);
    }
}
