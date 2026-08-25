//! DeepSeek V4 sparse MLA prefill over one complete request batch.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeepseekV4SparseMlaPrefillKernelConfig {
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
pub struct DeepseekV4SparseMlaPrefillKernelInput {
    pub query_context_pairs: Vec<(u32, u32)>,
}

impl DeepseekV4SparseMlaPrefillKernelInput {
    fn work(&self) -> (u32, f64, u32) {
        assert!(!self.query_context_pairs.is_empty());
        assert!(self.query_context_pairs.len() <= 64);
        let mut num_queries = 0_u32;
        let mut total_context = 0_u64;
        for &(queries, context) in &self.query_context_pairs {
            assert!(queries > 0 && queries <= context);
            num_queries = num_queries
                .checked_add(queries)
                .expect("query total must fit u32");
            total_context += u64::from(context);
        }
        assert!(num_queries <= 8192);
        let mean_context = total_context as f64 / self.query_context_pairs.len() as f64;
        let physical_launches = self.query_context_pairs.len().div_ceil(4) as u32;
        (num_queries, mean_context, physical_launches)
    }
}

impl SweepCoords for DeepseekV4SparseMlaPrefillKernelInput {
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

pub struct DeepseekV4SparseMlaPrefillSpec;

impl KernelSpec for DeepseekV4SparseMlaPrefillSpec {
    type Config = DeepseekV4SparseMlaPrefillKernelConfig;
    type Input = DeepseekV4SparseMlaPrefillKernelInput;

    const KIND: KernelKind = "deepseek_v4_sparse_mla_prefill";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert_eq!(config.prefill_chunk_size, 4);
        assert_eq!(config.max_num_batched_tokens, 8192);
        assert_eq!(
            config.selected_k,
            if config.compress_ratio == 1 { 0 } else { 512 }
        );
        SweepGrid::new(vec![
            Axis::values([1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]),
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
    DeepseekV4SparseMlaPrefillKernel,
    DeepseekV4SparseMlaPrefillSpec
);

#[cfg(test)]
mod tests {
    use super::DeepseekV4SparseMlaPrefillKernelInput;
    use crate::timing::SweepCoords;

    #[test]
    fn five_requests_remain_one_input_with_two_physical_launches() {
        let input = DeepseekV4SparseMlaPrefillKernelInput {
            query_context_pairs: vec![(4, 256), (3, 129), (2, 64), (1, 32), (5, 257)],
        };
        assert_eq!(input.coords()[0], 15.0);
        assert_eq!(input.coords()[2], 2.0);
    }
}
