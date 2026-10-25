//! MoE router: gate GEMM, softmax and top-k.
//!
//! Neuron backends: `neuron_torch`, `neuron_torch_topk_first`, `neuron_torch_bf16`, `neuron_nki_topk`. Contract:
//! `doc/pr/trn2-flashnext/L1_CONTRACT.md`; shapes are per rank.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_grid;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeRouterKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Model hidden width.
    pub hidden_size: u32,
    /// Routed experts in the model.
    pub num_experts: u32,
    /// Experts per token.
    pub top_k: u32,
    /// Activation dtype.
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeRouterKernelInput {
    /// Token rows of the call.
    pub num_tokens: u32,
}

pub struct MoeRouterSpec;

impl KernelSpec for MoeRouterSpec {
    type Config = MoeRouterKernelConfig;
    type Input = MoeRouterKernelInput;
    const KIND: KernelKind = "moe_router";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        neuron_grid::assert_neuron(Self::KIND, &config.backends);
        neuron_grid::grid_1d(neuron_grid::TOKENS)
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
                .with("num_tokens", num_tokens.round() as u32)
                .with("hidden_size", config.hidden_size)
                .with("num_experts", config.num_experts)
                .with("top_k", config.top_k)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(MoeRouterKernel, MoeRouterSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(backends: Vec<&'static str>) -> MoeRouterKernelConfig {
        MoeRouterKernelConfig {
            backends,
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            hidden_size: 2560,
            num_experts: 512,
            top_k: 10,
            dtype: DType::Bf16,
        }
    }

    /// Catches a payload field the Python args schema would reject, and a grid
    /// over the 500 feasible-coordinate ceiling.
    #[test]
    fn payload_matches_python_args() {
        let cfg = config(vec!["neuron_torch"]);
        let grid = MoeRouterSpec::sweep_grid(&cfg);
        let payloads = MoeRouterSpec::enumerate(&cfg, &grid, "neuron_torch");
        assert_eq!(payloads.len(), 14);
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "dtype",
                "hidden_size",
                "num_experts",
                "num_tokens",
                "top_k",
            ]
        );
    }

    #[test]
    #[should_panic(expected = "Neuron kind")]
    fn rejects_a_gpu_backend() {
        MoeRouterSpec::sweep_grid(&config(vec!["torch"]));
    }
}
