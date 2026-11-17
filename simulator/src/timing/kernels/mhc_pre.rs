//! mHC pre (Neuron): Hyper-connection pre: stream mixing weights (Sinkhorn), the collapse to one row and the module's RMSNorm, on the rank's own rows.
//!
//! Neuron backends only (`neuron_<impl>`, the op module's `IMPLS` key).
//! Contract: `doc/pr/trn2-glm53-flash/L1_CONTRACT.md`; shapes are per rank. Rows come from the
//! glm5-trn2 bench (bench_mhc_head.py kind mhc_pre) through `tools/neuron-bench-import`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_grid;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MhcPreKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Model hidden width.
    pub hidden_size: u32,
    /// Residual streams.
    pub hc_mult: u32,
    /// Sinkhorn iterations.
    pub sinkhorn_iters: u32,
    /// Activation dtype.
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MhcPreKernelInput {
    /// Rows on this rank (its sequence-parallel slice).
    pub num_tokens: u32,
}

pub struct MhcPreSpec;

impl KernelSpec for MhcPreSpec {
    type Config = MhcPreKernelConfig;
    type Input = MhcPreKernelInput;
    const KIND: KernelKind = "mhc_pre";

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
                .with("hc_mult", config.hc_mult)
                .with("sinkhorn_iters", config.sinkhorn_iters)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(MhcPreKernel, MhcPreSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(backends: Vec<&'static str>) -> MhcPreKernelConfig {
        MhcPreKernelConfig {
            backends,
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            hidden_size: 8,
            hc_mult: 8,
            sinkhorn_iters: 8,
            dtype: DType::Bf16,
        }
    }

    /// Catches a payload field the Python args schema would reject.
    #[test]
    fn payload_matches_python_args() {
        let cfg = config(vec!["neuron_torch"]);
        let grid = MhcPreSpec::sweep_grid(&cfg);
        let payloads = MhcPreSpec::enumerate(&cfg, &grid, "neuron_torch");
        assert!(!payloads.is_empty());
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "dtype",
                "hc_mult",
                "hidden_size",
                "num_tokens",
                "sinkhorn_iters"
            ]
        );
    }

    #[test]
    #[should_panic(expected = "Neuron kind")]
    fn rejects_a_gpu_backend() {
        MhcPreSpec::sweep_grid(&config(vec!["torch"]));
    }
}
