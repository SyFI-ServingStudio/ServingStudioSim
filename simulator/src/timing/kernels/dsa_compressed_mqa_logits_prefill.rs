//! DSA indexer-prefill MQA logits over compressed keys, for one complete request batch.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, RateWork};
use crate::timing::kernels::causal_rows;
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaCompressedPrefillKernelInput {
    pub query_context_pairs: Vec<(u32, u32)>,
}

impl DsaCompressedPrefillKernelInput {
    fn work(&self) -> (u32, f64, u32) {
        assert!(!self.query_context_pairs.is_empty());
        let mut num_queries = 0_u32;
        let mut total_context = 0_u64;
        for &(queries, context) in &self.query_context_pairs {
            assert!(queries > 0 && queries <= context);
            num_queries = num_queries
                .checked_add(queries)
                .expect("query total must fit u32");
            total_context += u64::from(context);
        }
        (
            num_queries,
            total_context as f64 / self.query_context_pairs.len() as f64,
            self.query_context_pairs.len() as u32,
        )
    }
}

impl SweepCoords for DsaCompressedPrefillKernelInput {
    fn coords(&self) -> Coords {
        let (num_queries, mean_context, num_requests) = self.work();
        Coords::new([
            f64::from(num_queries),
            mean_context,
            f64::from(num_requests),
        ])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["num_queries", "mean_context", "num_requests"]
    }
}

pub(crate) fn sweep_grid(max_model_len: u32) -> SweepGrid {
    SweepGrid::new(vec![
        Axis::values([1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]),
        Axis::values([4, 32, 128, 512, 2048, 8192, 32768, 65536, 262144, 1048576])
            .into_iter()
            .filter(|&context| context <= f64::from(max_model_len))
            .collect(),
        Axis::values([1, 4, 16, 64]),
    ])
}

pub(crate) fn canonical_pairs(
    num_queries: u32,
    context: u32,
    num_requests: u32,
) -> Option<Vec<(u32, u32)>> {
    if num_requests == 0 || num_requests > num_queries {
        return None;
    }
    let base = num_queries / num_requests;
    let remainder = num_queries % num_requests;
    let pairs = (0..num_requests)
        .map(|request| {
            let queries = base + u32::from(request < remainder);
            (queries, context)
        })
        .collect::<Vec<_>>();
    pairs
        .iter()
        .all(|&(queries, _)| queries <= context)
        .then_some(pairs)
}

pub(crate) fn infeasible_mask(grid: &SweepGrid) -> Vec<bool> {
    grid.expand(|coordinates| {
        canonical_pairs(
            coordinates[0] as u32,
            coordinates[1] as u32,
            coordinates[2] as u32,
        )
        .is_none()
    })
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaCompressedMqaLogitsPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub max_model_len: u32,
    pub max_num_batched_tokens: u32,
    pub max_logits_bytes: u64,
    pub compress_ratio: u32,
    pub num_heads: Dim,
    pub head_dim: Dim,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub k_dtype: DType,
    pub k_scale_dtype: DType,
    pub weight_dtype: DType,
    pub output_dtype: DType,
    pub clean_logits: bool,
}

/// The profiler's logical bytes: q, each (query, compressed key) pair's key,
/// key scale and logit, and each query's head weights and row bounds.
fn logical_bytes(
    config: &DsaCompressedMqaLogitsPrefillKernelConfig,
    input: &DsaCompressedPrefillKernelInput,
) -> f64 {
    let heads = f64::from(config.num_heads.get());
    let head_dim = f64::from(config.head_dim.get());
    let (mut queries, mut pairs) = (0.0, 0.0);
    for &(request_queries, context) in &input.query_context_pairs {
        queries += f64::from(request_queries);
        pairs += causal_rows::compressed(request_queries, context, config.compress_ratio, None);
    }
    let size = |dtype: DType| f64::from(dtype.size_bytes());
    queries * (size(config.q_dtype) * heads * head_dim + size(config.weight_dtype) * heads + 8.0)
        + pairs
            * (size(config.k_dtype) * head_dim
                + size(config.k_scale_dtype)
                + size(config.output_dtype))
}

pub struct DsaCompressedMqaLogitsPrefillSpec;

impl KernelSpec for DsaCompressedMqaLogitsPrefillSpec {
    type Config = DsaCompressedMqaLogitsPrefillKernelConfig;
    type Input = DsaCompressedPrefillKernelInput;

    const KIND: KernelKind = "dsa_compressed_mqa_logits_prefill";

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

    /// Past the grid the launches hold their bandwidth. On H200 the largest
    /// measured points sit closer to their neighbor's bandwidth than to its
    /// TFLOPS, and the 64-request edge already runs in the chunked regime
    /// that more requests stay in.
    fn off_grid(
        config: &Self::Config,
        input: &Self::Input,
        _backend: &'static str,
    ) -> OffGrid<Self::Input> {
        OffGrid::Rate(RateWork::Bytes(logical_bytes(config, input)))
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
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("q_dtype", config.q_dtype.as_str())
                .with("k_dtype", config.k_dtype.as_str())
                .with("k_scale_dtype", config.k_scale_dtype.as_str())
                .with("weight_dtype", config.weight_dtype.as_str())
                .with("output_dtype", config.output_dtype.as_str())
                .with("clean_logits", config.clean_logits)
        })
    }
}

register_kernel!(
    DsaCompressedMqaLogitsPrefillKernel,
    DsaCompressedMqaLogitsPrefillSpec
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::SweepCoords;

    #[test]
    fn one_large_request_stays_one_semantic_input() {
        let input = DsaCompressedPrefillKernelInput {
            query_context_pairs: vec![(8192, 1_048_576)],
        };
        assert_eq!(&*input.coords(), &[8192.0, 1_048_576.0, 1.0]);
    }

    #[test]
    fn logical_bytes_match_a_measured_row() {
        // A profiled H200 row's logged bandwidth x time.
        let config: DsaCompressedMqaLogitsPrefillKernelConfig =
            serde_json::from_value(serde_json::json!({"gpu_name": "NVIDIA H200", "max_model_len": 1048576, "max_num_batched_tokens": 8192, "max_logits_bytes": 536870912, "compress_ratio": 4, "num_heads": 64, "head_dim": 128, "q_dtype": "fp8_e4m3", "k_dtype": "fp8_e4m3", "k_scale_dtype": "fp32", "weight_dtype": "fp32", "output_dtype": "fp32", "clean_logits": false, "backends": ["vllm_deepgemm_fp8"]}))
            .unwrap();
        let input = DsaCompressedPrefillKernelInput {
            query_context_pairs: vec![(4, 512); 4],
        };
        assert_eq!(logical_bytes(&config, &input), 412_192.0);
    }
}
