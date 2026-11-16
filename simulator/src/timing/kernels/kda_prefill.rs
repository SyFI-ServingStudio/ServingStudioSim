//! KDA prefill (Neuron): The whole KDA block of one rank over one request chunk (projections, short conv, chunked delta rule, gated norm, o_proj partial).
//!
//! Neuron backends only (`neuron_<impl>`, the op module's `IMPLS` key).
//! Contract: `doc/pr/trn2-glm53-flash/L1_CONTRACT.md`; shapes are per rank. Rows come from the
//! glm5-trn2 bench (bench_kda.py kind kda_prefill) through `tools/neuron-bench-import`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_grid;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KdaPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Model hidden width.
    pub hidden_size: u32,
    /// KDA heads on this rank (64 / attn_tp).
    pub num_heads: u32,
    /// KDA head dimension.
    pub head_dim: u32,
    /// Short-conv kernel width.
    pub conv_kernel_size: u32,
    /// Activation dtype.
    #[compute_dtype]
    pub dtype: DType,
    /// Weight storage dtype (attention stays BF16 in the FP8 checkpoint).
    pub weight_dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KdaPrefillKernelInput {
    /// Tokens of the one request chunk (bucket-padded).
    pub num_tokens: u32,
}

pub struct KdaPrefillSpec;

impl KernelSpec for KdaPrefillSpec {
    type Config = KdaPrefillKernelConfig;
    type Input = KdaPrefillKernelInput;
    const KIND: KernelKind = "kda_prefill";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        neuron_grid::assert_neuron(Self::KIND, &config.backends);
        neuron_grid::grid_1d(neuron_grid::CHUNK)
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
                .with("num_heads", config.num_heads)
                .with("head_dim", config.head_dim)
                .with("conv_kernel_size", config.conv_kernel_size)
                .with("dtype", config.dtype.as_str())
                .with("weight_dtype", config.weight_dtype.as_str())
        })
    }
}

register_kernel!(KdaPrefillKernel, KdaPrefillSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(backends: Vec<&'static str>) -> KdaPrefillKernelConfig {
        KdaPrefillKernelConfig {
            backends,
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            hidden_size: 8,
            num_heads: 8,
            head_dim: 8,
            conv_kernel_size: 8,
            dtype: DType::Bf16,
            weight_dtype: DType::Fp8E4m3,
        }
    }

    /// Catches a payload field the Python args schema would reject.
    #[test]
    fn payload_matches_python_args() {
        let cfg = config(vec!["neuron_torch"]);
        let grid = KdaPrefillSpec::sweep_grid(&cfg);
        let payloads = KdaPrefillSpec::enumerate(&cfg, &grid, "neuron_torch");
        assert!(!payloads.is_empty());
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(keys, ["backend", "conv_kernel_size", "dtype", "head_dim", "hidden_size", "num_heads", "num_tokens", "weight_dtype"]);
    }

    #[test]
    #[should_panic(expected = "Neuron kind")]
    fn rejects_a_gpu_backend() {
        KdaPrefillSpec::sweep_grid(&config(vec!["torch"]));
    }
}
