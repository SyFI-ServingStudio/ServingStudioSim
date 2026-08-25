//! DeepSeek V4 C4 indexer-prefill top-k over one complete request batch.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::deepseek_v4_indexer_mqa_logits_prefill::{
    canonical_pairs, infeasible_mask, sweep_grid, DeepseekV4IndexerPrefillKernelInput,
};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::SweepGrid;
use crate::timing::KernelConfig;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeepseekV4IndexerTopkPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub max_model_len: u32,
    pub max_num_batched_tokens: u32,
    pub max_logits_bytes: u64,
    pub compress_ratio: u32,
    pub top_k: u32,
    #[compute_dtype]
    pub logits_dtype: DType,
    pub index_dtype: String,
}

pub struct DeepseekV4IndexerTopkPrefillSpec;

impl KernelSpec for DeepseekV4IndexerTopkPrefillSpec {
    type Config = DeepseekV4IndexerTopkPrefillKernelConfig;
    type Input = DeepseekV4IndexerPrefillKernelInput;

    const KIND: KernelKind = "deepseek_v4_indexer_topk_prefill";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert_eq!(config.max_num_batched_tokens, 8192);
        assert_eq!(config.max_logits_bytes, 512 * 1024 * 1024);
        assert_eq!(config.compress_ratio, 4);
        sweep_grid(config.max_model_len)
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        infeasible_mask(grid)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand(|coordinates| {
            let pairs = canonical_pairs(
                coordinates[0] as u32,
                coordinates[1] as u32,
                coordinates[2] as u32,
            )
            .unwrap_or_else(|| vec![(1, 4)]);
            ArgsPayload::new()
                .with("backend", backend)
                .with("query_context_pairs", serde_json::json!(pairs))
                .with("max_model_len", config.max_model_len)
                .with("max_num_batched_tokens", config.max_num_batched_tokens)
                .with("max_logits_bytes", config.max_logits_bytes)
                .with("compress_ratio", config.compress_ratio)
                .with("top_k", config.top_k)
                .with("logits_dtype", config.logits_dtype.as_str())
                .with("index_dtype", config.index_dtype.clone())
        })
    }
}

register_kernel!(
    DeepseekV4IndexerTopkPrefillKernel,
    DeepseekV4IndexerTopkPrefillSpec
);
