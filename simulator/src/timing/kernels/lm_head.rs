//! LM head (Neuron): Final stream mean + norm, the vocab-parallel lm_head GEMM over the rank's vocab shard and the local greedy argmax.
//!
//! Neuron backends only (`neuron_<impl>`, the op module's `IMPLS` key).
//! Contract: `doc/pr/trn2-glm53-flash/L1_CONTRACT.md`; shapes are per rank. Rows come from the
//! glm5-trn2 bench (bench_mhc_head.py kind lm_head) through `tools/neuron-bench-import`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_grid;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LmHeadKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Model hidden width.
    pub hidden_size: u32,
    /// Vocabulary rows on this rank (vocab / attn_tp).
    pub vocab_size: u32,
    /// Residual streams.
    pub hc_mult: u32,
    /// Activation dtype.
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LmHeadKernelInput {
    /// Rows sampled (decode rows, or one per prefill chunk).
    pub num_tokens: u32,
}

pub struct LmHeadSpec;

impl KernelSpec for LmHeadSpec {
    type Config = LmHeadKernelConfig;
    type Input = LmHeadKernelInput;
    const KIND: KernelKind = "lm_head";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        neuron_grid::assert_neuron(Self::KIND, &config.backends);
        neuron_grid::grid_1d(neuron_grid::HEAD_ROWS)
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
                .with("vocab_size", config.vocab_size)
                .with("hc_mult", config.hc_mult)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(LmHeadKernel, LmHeadSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(backends: Vec<&'static str>) -> LmHeadKernelConfig {
        LmHeadKernelConfig {
            backends,
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            hidden_size: 8,
            vocab_size: 8,
            hc_mult: 8,
            dtype: DType::Bf16,
        }
    }

    /// Catches a payload field the Python args schema would reject.
    #[test]
    fn payload_matches_python_args() {
        let cfg = config(vec!["neuron_torch"]);
        let grid = LmHeadSpec::sweep_grid(&cfg);
        let payloads = LmHeadSpec::enumerate(&cfg, &grid, "neuron_torch");
        assert!(!payloads.is_empty());
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(keys, ["backend", "dtype", "hc_mult", "hidden_size", "num_tokens", "vocab_size"]);
    }

    #[test]
    #[should_panic(expected = "Neuron kind")]
    fn rejects_a_gpu_backend() {
        LmHeadSpec::sweep_grid(&config(vec!["torch"]));
    }
}
