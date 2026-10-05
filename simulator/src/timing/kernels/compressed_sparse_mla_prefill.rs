//! Compressed sparse MLA prefill over one complete request batch.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::causal_rows;
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompressedSparseMlaPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub max_model_len: u32,
    pub max_num_batched_tokens: u32,
    pub prefill_chunk_size: u32,
    pub compress_ratio: u32,
    pub window_size: u32,
    pub selected_k: u32,
    pub selected_index_pattern: String,
    pub num_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub value_dim: Dim,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub cache_dtype: DType,
    pub index_dtype: String,
    pub output_dtype: DType,
    pub cache_layout: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompressedSparseMlaPrefillKernelInput {
    pub query_context_pairs: Vec<(u32, u32)>,
}

impl CompressedSparseMlaPrefillKernelInput {
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
        let mean_context = total_context as f64 / self.query_context_pairs.len() as f64;
        let physical_launches = self.query_context_pairs.len().div_ceil(4) as u32;
        (num_queries, mean_context, physical_launches)
    }
}

impl SweepCoords for CompressedSparseMlaPrefillKernelInput {
    fn coords(&self) -> Coords {
        let (num_queries, mean_context, physical_launches) = self.work();
        Coords::new([
            f64::from(num_queries),
            mean_context,
            f64::from(physical_launches),
        ])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["num_queries", "mean_context", "physical_launches"]
    }
}

fn request_count(num_queries: u32, physical_launches: u32) -> Option<u32> {
    let minimum = 4 * (physical_launches - 1) + 1;
    (num_queries >= minimum).then(|| num_queries.min(4 * physical_launches))
}

fn canonical_pairs(
    num_queries: u32,
    context: u32,
    physical_launches: u32,
) -> Option<Vec<(u32, u32)>> {
    let num_requests = request_count(num_queries, physical_launches)?;
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

/// The most requests a profiled batch holds: 16 launches of four.
const PROFILED_REQUESTS: usize = 64;

/// The profiler's logical bytes: q, the padded top-k plus window indices and
/// their lengths, the selected cache rows (compressed top-k and window), the
/// output, and the max/lse rows.
fn logical_bytes(
    config: &CompressedSparseMlaPrefillKernelConfig,
    input: &CompressedSparseMlaPrefillKernelInput,
) -> f64 {
    let heads = f64::from(config.num_heads.get());
    let head_dim = f64::from(config.head_dim.get());
    let (mut queries, mut valid) = (0.0, 0.0);
    for &(request_queries, context) in &input.query_context_pairs {
        queries += f64::from(request_queries);
        valid += causal_rows::capped(request_queries, context, config.window_size);
        if config.compress_ratio > 1 {
            valid += causal_rows::compressed(
                request_queries,
                context,
                config.compress_ratio,
                Some(config.selected_k),
            );
        }
    }
    let per_query = f64::from(config.q_dtype.size_bytes()) * heads * head_dim
        + 4.0 * f64::from(config.window_size + config.selected_k)
        + 4.0
        + f64::from(config.output_dtype.size_bytes()) * heads * f64::from(config.value_dim.get())
        + 8.0 * heads;
    queries * per_query + f64::from(config.cache_dtype.size_bytes()) * valid * head_dim
}

pub struct CompressedSparseMlaPrefillSpec;

impl KernelSpec for CompressedSparseMlaPrefillSpec {
    type Config = CompressedSparseMlaPrefillKernelConfig;
    type Input = CompressedSparseMlaPrefillKernelInput;

    const KIND: KernelKind = "compressed_sparse_mla_prefill";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert_eq!(config.prefill_chunk_size, 4);
        assert_eq!(
            config.selected_k,
            if config.compress_ratio == 1 { 0 } else { 512 }
        );
        SweepGrid::new(vec![
            causal_rows::query_axis(&causal_rows::PREFILL_QUERIES, config.max_num_batched_tokens),
            Axis::values([1, 32, 128, 512, 2048, 8192, 32768, 65536, 262144, 1048576])
                .into_iter()
                .filter(|&context| context <= f64::from(config.max_model_len))
                .collect(),
            Axis::values([1, 2, 4, 8, 16]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand(|coordinates| {
            canonical_pairs(
                coordinates[0] as u32,
                coordinates[1] as u32,
                coordinates[2] as u32,
            )
            .is_none()
        })
    }

    /// Requests run four to a launch, one launch after another, so a batch
    /// past the profiled 16 launches is the sum of 64-request batches. Past
    /// the query and context axes the launch holds its bandwidth: on H200 the
    /// largest measured points sit within 9% of their neighbor's bandwidth and
    /// up to 39% off its TFLOPS.
    fn off_grid(
        config: &Self::Config,
        input: &Self::Input,
        _backend: &'static str,
    ) -> OffGrid<Self::Input> {
        if input.query_context_pairs.len() > PROFILED_REQUESTS {
            return OffGrid::Launches(
                input
                    .query_context_pairs
                    .chunks(PROFILED_REQUESTS)
                    .map(|pairs| CompressedSparseMlaPrefillKernelInput {
                        query_context_pairs: pairs.to_vec(),
                    })
                    .collect(),
            );
        }
        OffGrid::Bandwidth(logical_bytes(config, input))
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
            .unwrap_or_else(|| vec![(1, 1)]);
            ArgsPayload::new()
                .with("backend", backend)
                .with("query_context_pairs", serde_json::json!(pairs))
                .with("max_model_len", config.max_model_len)
                .with("max_num_batched_tokens", config.max_num_batched_tokens)
                .with("prefill_chunk_size", config.prefill_chunk_size)
                .with("compress_ratio", config.compress_ratio)
                .with("window_size", config.window_size)
                .with("selected_k", config.selected_k)
                .with(
                    "selected_index_pattern",
                    config.selected_index_pattern.clone(),
                )
                .with("num_heads", config.num_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("value_dim", config.value_dim.get())
                .with(
                    "softmax_scale",
                    1.0 / f64::from(config.head_dim.get()).sqrt(),
                )
                .with("q_dtype", config.q_dtype.as_str())
                .with("cache_dtype", config.cache_dtype.as_str())
                .with("index_dtype", config.index_dtype.clone())
                .with("output_dtype", config.output_dtype.as_str())
                .with("cache_layout", config.cache_layout.clone())
        })
    }
}

register_kernel!(
    CompressedSparseMlaPrefillKernel,
    CompressedSparseMlaPrefillSpec
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::SweepCoords;

    #[test]
    fn five_requests_remain_one_input_with_two_physical_launches() {
        let input = CompressedSparseMlaPrefillKernelInput {
            query_context_pairs: vec![(4, 256), (3, 129), (2, 64), (1, 32), (5, 257)],
        };
        assert_eq!(input.coords()[0], 15.0);
        assert_eq!(input.coords()[2], 2.0);
    }

    #[test]
    fn logical_bytes_match_a_measured_row() {
        // A profiled H200 row's logged bandwidth x time.
        let config: CompressedSparseMlaPrefillKernelConfig =
            serde_json::from_value(serde_json::json!({"gpu_name": "NVIDIA H200", "max_model_len": 1048576, "max_num_batched_tokens": 8192, "prefill_chunk_size": 4, "compress_ratio": 4, "window_size": 128, "selected_k": 512, "selected_index_pattern": "request_local_topk_plus_swa", "num_heads": 64, "num_kv_heads": 1, "head_dim": 512, "value_dim": 512, "q_dtype": "bf16", "cache_dtype": "bf16", "index_dtype": "int32", "output_dtype": "bf16", "cache_layout": "request_slot_major_flat_mqa_bf16_d512", "backends": ["vllm_flashmla_bf16"]}))
            .unwrap();
        let input = CompressedSparseMlaPrefillKernelInput {
            query_context_pairs: vec![(1, 8192); 4],
        };
        assert_eq!(logical_bytes(&config, &input), 3_158_032.0);
    }

    #[test]
    fn more_than_sixteen_launches_split_into_profiled_batches() {
        let config: CompressedSparseMlaPrefillKernelConfig =
            serde_json::from_value(serde_json::json!({"gpu_name": "NVIDIA H200", "max_model_len": 1048576, "max_num_batched_tokens": 8192, "prefill_chunk_size": 4, "compress_ratio": 4, "window_size": 128, "selected_k": 512, "selected_index_pattern": "request_local_topk_plus_swa", "num_heads": 64, "num_kv_heads": 1, "head_dim": 512, "value_dim": 512, "q_dtype": "bf16", "cache_dtype": "bf16", "index_dtype": "int32", "output_dtype": "bf16", "cache_layout": "request_slot_major_flat_mqa_bf16_d512", "backends": ["vllm_flashmla_bf16"]}))
            .unwrap();
        let input = CompressedSparseMlaPrefillKernelInput {
            query_context_pairs: vec![(16, 4096); 130],
        };
        let OffGrid::Launches(launches) =
            CompressedSparseMlaPrefillSpec::off_grid(&config, &input, "vllm_flashmla_bf16")
        else {
            panic!("130 requests must split into launches");
        };
        let sizes: Vec<_> = launches
            .iter()
            .map(|l| l.query_context_pairs.len())
            .collect();
        assert_eq!(sizes, [64, 64, 2]);
    }
}
