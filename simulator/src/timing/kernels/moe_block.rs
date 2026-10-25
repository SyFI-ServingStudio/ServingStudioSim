//! Whole MoE block (router torch_topk_first, routed `<impl>`, shared torch),
//! real routing.
//!
//! Neuron backends: `neuron_auto`, `neuron_nki`. Contract:
//! `doc/pr/trn2-flashnext/L1_CONTRACT.md`; shapes are per rank.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_grid;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeBlockKernelConfig {
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
    /// Shared-expert intermediate width on this rank.
    pub shared_intermediate_size: u32,
    /// Activation dtype.
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeBlockKernelInput {
    /// Token rows of the call.
    pub num_tokens: u32,
}

pub struct MoeBlockSpec;

impl KernelSpec for MoeBlockSpec {
    type Config = MoeBlockKernelConfig;
    type Input = MoeBlockKernelInput;
    const KIND: KernelKind = "moe_block";

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
                .with("local_experts", config.local_experts)
                .with("intermediate_size", config.intermediate_size)
                .with("shared_intermediate_size", config.shared_intermediate_size)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(MoeBlockKernel, MoeBlockSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(backends: Vec<&'static str>) -> MoeBlockKernelConfig {
        MoeBlockKernelConfig {
            backends,
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            hidden_size: 2560,
            num_experts: 512,
            top_k: 10,
            local_experts: 128,
            intermediate_size: 160,
            shared_intermediate_size: 160,
            dtype: DType::Bf16,
        }
    }

    /// Catches a payload field the Python args schema would reject, and a grid
    /// over the 500 feasible-coordinate ceiling.
    #[test]
    fn payload_matches_python_args() {
        let cfg = config(vec!["neuron_auto"]);
        let grid = MoeBlockSpec::sweep_grid(&cfg);
        let payloads = MoeBlockSpec::enumerate(&cfg, &grid, "neuron_auto");
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
                "shared_intermediate_size",
                "top_k",
            ]
        );
    }

    #[test]
    #[should_panic(expected = "Neuron kind")]
    fn rejects_a_gpu_backend() {
        MoeBlockSpec::sweep_grid(&config(vec!["torch"]));
    }
}
