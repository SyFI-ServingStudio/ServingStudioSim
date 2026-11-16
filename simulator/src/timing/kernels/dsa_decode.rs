//! DSA decode (Neuron): The whole DSA block of one rank over a decode batch at one context bucket.
//!
//! Neuron backends only (`neuron_<impl>`, the op module's `IMPLS` key).
//! Contract: `doc/pr/trn2-glm53-flash/L1_CONTRACT.md`; shapes are per rank. Rows come from the
//! glm5-trn2 bench (bench_dsa.py kind dsa_decode) through `tools/neuron-bench-import`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_grid;
use crate::timing::sweep::SweepGrid;
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaDecodeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Model hidden width.
    pub hidden_size: u32,
    /// MLA heads on this rank (64 / attn_tp).
    pub num_heads: u32,
    /// Latent KV width.
    pub kv_lora_rank: u32,
    /// MLA query/key (NoPE) and value head dimension.
    pub head_dim: u32,
    /// Indexer heads (replicated).
    pub index_heads: u32,
    /// Indexer head dimension.
    pub index_head_dim: u32,
    /// Selected tokens per query row.
    pub index_topk: u32,
    /// Activation dtype.
    #[compute_dtype]
    pub dtype: DType,
    /// Weight storage dtype.
    pub weight_dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaDecodeKernelInput {
    /// Decode rows (requests, bucket-padded).
    pub batch_size: u32,
    /// Context bucket the region reads ([:C]).
    pub context_len: u32,
}

pub struct DsaDecodeSpec;

impl KernelSpec for DsaDecodeSpec {
    type Config = DsaDecodeKernelConfig;
    type Input = DsaDecodeKernelInput;
    const KIND: KernelKind = "dsa_decode";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        neuron_grid::assert_neuron(Self::KIND, &config.backends);
        neuron_grid::dsa_grid(&neuron_grid::DECODE)
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|batch_size, context_len| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", batch_size.round() as u32)
                .with("context_len", context_len.round() as u32)
                .with("hidden_size", config.hidden_size)
                .with("num_heads", config.num_heads)
                .with("kv_lora_rank", config.kv_lora_rank)
                .with("head_dim", config.head_dim)
                .with("index_heads", config.index_heads)
                .with("index_head_dim", config.index_head_dim)
                .with("index_topk", config.index_topk)
                .with("dtype", config.dtype.as_str())
                .with("weight_dtype", config.weight_dtype.as_str())
        })
    }
}

register_kernel!(DsaDecodeKernel, DsaDecodeSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(backends: Vec<&'static str>) -> DsaDecodeKernelConfig {
        DsaDecodeKernelConfig {
            backends,
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            hidden_size: 8,
            num_heads: 8,
            kv_lora_rank: 8,
            head_dim: 8,
            index_heads: 8,
            index_head_dim: 8,
            index_topk: 8,
            dtype: DType::Bf16,
            weight_dtype: DType::Fp8E4m3,
        }
    }

    /// Catches a payload field the Python args schema would reject.
    #[test]
    fn payload_matches_python_args() {
        let cfg = config(vec!["neuron_torch"]);
        let grid = DsaDecodeSpec::sweep_grid(&cfg);
        let payloads = DsaDecodeSpec::enumerate(&cfg, &grid, "neuron_torch");
        assert!(!payloads.is_empty());
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "batch_size",
                "context_len",
                "dtype",
                "head_dim",
                "hidden_size",
                "index_head_dim",
                "index_heads",
                "index_topk",
                "kv_lora_rank",
                "num_heads",
                "weight_dtype"
            ]
        );
    }

    #[test]
    #[should_panic(expected = "Neuron kind")]
    fn rejects_a_gpu_backend() {
        DsaDecodeSpec::sweep_grid(&config(vec!["torch"]));
    }
}
