//! Host launch overhead: pipelined wall per step of `num_regions` back-to-back
//! trivial compiled regions (host launch + inter-region gap). Linear
//! extrapolation past 64 regions is intended.
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
pub struct RegionLaunchOverheadKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Activation dtype.
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegionLaunchOverheadKernelInput {
    /// Back-to-back compiled regions.
    pub num_regions: u32,
}

pub struct RegionLaunchOverheadSpec;

impl KernelSpec for RegionLaunchOverheadSpec {
    type Config = RegionLaunchOverheadKernelConfig;
    type Input = RegionLaunchOverheadKernelInput;
    const KIND: KernelKind = "region_launch_overhead";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        neuron_grid::assert_neuron(Self::KIND, &config.backends);
        neuron_grid::grid_1d(neuron_grid::REGIONS)
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_regions| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_regions", num_regions.round() as u32)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(RegionLaunchOverheadKernel, RegionLaunchOverheadSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(backends: Vec<&'static str>) -> RegionLaunchOverheadKernelConfig {
        RegionLaunchOverheadKernelConfig {
            backends,
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            dtype: DType::Bf16,
        }
    }

    /// Catches a payload field the Python args schema would reject, and a grid
    /// over the 500 feasible-coordinate ceiling.
    #[test]
    fn payload_matches_python_args() {
        let cfg = config(vec!["neuron_torch"]);
        let grid = RegionLaunchOverheadSpec::sweep_grid(&cfg);
        let payloads = RegionLaunchOverheadSpec::enumerate(&cfg, &grid, "neuron_torch");
        assert_eq!(payloads.len(), 7);
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(keys, ["backend", "dtype", "num_regions",]);
    }

    #[test]
    #[should_panic(expected = "Neuron kind")]
    fn rejects_a_gpu_backend() {
        RegionLaunchOverheadSpec::sweep_grid(&config(vec!["torch"]));
    }
}
