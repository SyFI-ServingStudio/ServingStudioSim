//! PLE similarity gate: one fused launch of the fork's `_ple_gate_kernel`
//! (grid `(tokens, hc_mult)`). Per token and residual stream: RMS-normalize
//! the key slice and the hidden stream, turn their scaled dot product into a
//! signed-square-root sigmoid gate, gate the shared value row and
//! RMS-normalize it for the short convolution. Memory bound in `num_tokens`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

/// Decode/verify rows through the largest scheduled step.
const TOKEN_AXIS: [u32; 25] = [
    1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
    6144, 8192, 12288, 16384, 32768,
];

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PleGateKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Elements per residual stream (the norm group).
    pub hidden_size: u32,
    /// Residual streams sharing one value row.
    pub hc_mult: u32,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PleGateKernelInput {
    pub num_tokens: u32,
}

pub struct PleGateSpec;

impl KernelSpec for PleGateSpec {
    type Config = PleGateKernelConfig;
    type Input = PleGateKernelInput;
    const KIND: KernelKind = "ple_gate";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert!(
            config.hidden_size > 0 && config.hc_mult > 0,
            "ple_gate needs a positive shape"
        );
        SweepGrid::new(vec![Axis::values(TOKEN_AXIS)])
    }

    fn cache_kind(_: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", tokens as u32)
                .with("hidden_size", config.hidden_size)
                .with("hc_mult", config.hc_mult)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(PleGateKernel, PleGateSpec);

#[cfg(test)]
mod tests {
    use super::*;

    /// Catches a payload field the Python args schema would reject.
    #[test]
    fn payload_matches_python_args() {
        let cfg = PleGateKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden_size: 2560,
            hc_mult: 4,
            dtype: DType::Bf16,
        };
        let grid = PleGateSpec::sweep_grid(&cfg);
        let payloads = PleGateSpec::enumerate(&cfg, &grid, "vllm_triton");
        assert_eq!(payloads.len(), TOKEN_AXIS.len());
        let fields = payloads[3].fields();
        let keys: Vec<_> = fields.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            ["backend", "dtype", "hc_mult", "hidden_size", "num_tokens"]
        );
        assert_eq!(fields["num_tokens"], 8);
    }
}
