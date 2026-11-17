//! In-graph reduce-scatter; the message is the per-rank full input.
//!
//! Neuron backends: `neuron_cc`. Contract:
//! `doc/pr/trn2-flashnext/L1_CONTRACT.md`; shapes are per rank.

use crate::common::Fabric;
use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_grid;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

/// Neuron collectives are size-keyed like `all_reduce`: profiled once at bf16.
const PROFILE_DTYPE: DType = DType::Bf16;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReduceScatterKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Ranks in the group.
    pub num_gpus: u32,
    pub fabric: Fabric,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReduceScatterKernelInput {
    /// Per-rank message, in bytes.
    pub message_size_bytes: u64,
}

pub struct ReduceScatterSpec;

impl KernelSpec for ReduceScatterSpec {
    type Config = ReduceScatterKernelConfig;
    type Input = ReduceScatterKernelInput;
    const KIND: KernelKind = "reduce_scatter";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        neuron_grid::assert_neuron(Self::KIND, &config.backends);
        neuron_grid::comm_bytes_grid()
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|message_size| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_gpus", config.num_gpus)
                .with("message_size_bytes", message_size.round() as u64)
                .with("dtype", PROFILE_DTYPE.as_str())
                .with("fabric", config.fabric.as_str())
        })
    }
}

register_kernel!(ReduceScatterKernel, ReduceScatterSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(backends: Vec<&'static str>) -> ReduceScatterKernelConfig {
        ReduceScatterKernelConfig {
            backends,
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            num_gpus: 8,
            fabric: Fabric::NeuronC2,
        }
    }

    /// Catches a payload field the Python args schema would reject, and a grid
    /// over the 500 feasible-coordinate ceiling.
    #[test]
    fn payload_matches_python_args() {
        let cfg = config(vec!["neuron_cc"]);
        let grid = ReduceScatterSpec::sweep_grid(&cfg);
        let payloads = ReduceScatterSpec::enumerate(&cfg, &grid, "neuron_cc");
        assert_eq!(payloads.len(), 5);
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "dtype",
                "fabric",
                "message_size_bytes",
                "num_gpus",
            ]
        );
    }

    #[test]
    #[should_panic(expected = "Neuron kind")]
    fn rejects_a_gpu_backend() {
        ReduceScatterSpec::sweep_grid(&config(vec!["torch"]));
    }
}
