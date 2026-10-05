//! Varlen sparse-MLA prefill over one complete request batch.
//!
//! Production issues one FlashInfer decode launch for the whole ragged batch.
//! Keep every request's `(query_rows, context_len)` pair in `Input` and in the
//! profiler payload: merging requests changes the per-row causal valid counts.
//! The cache projects that topology to total query rows, request count, and the
//! endpoint context of an equal-chunk canonical batch with the same mean causal
//! context per query row. This preserves every canonical and one-request row,
//! while accounting for the fact that a request's rows span
//! `context - queries + 1 ..= context` rather than all seeing the endpoint.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::causal_rows;
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// The most requests a profiled canonical batch splits into. The production
/// launch has no request dimension (one row per query token), so this bounds
/// only the grid; past it the launch holds the edge's bandwidth (`off_grid`).
const MAX_REQUESTS: usize = 64;

/// The most query rows one launch takes: the kernel puts one CTA row per query
/// row on the grid's z dimension, which CUDA caps at 65535. More rows fail the
/// launch.
pub const MAX_LAUNCH_QUERY_ROWS: u32 = 65_535;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaSparseMlaPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub max_model_len: u32,
    pub num_heads: Dim,
    pub num_kv_heads: Dim,
    pub selected_k: u32,
    pub latent_dim: Dim,
    pub rope_dim: Dim,
    pub value_dim: Dim,
    pub softmax_scale_denominator: u32,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub cache_dtype: DType,
    pub index_dtype: String,
    pub output_dtype: DType,
    pub index_distribution: String,
    pub cache_layout: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaSparseMlaPrefillKernelInput {
    pub query_context_pairs: Vec<(u32, u32)>,
}

impl DsaSparseMlaPrefillKernelInput {
    fn work(&self) -> (u32, u32, f64) {
        assert!(!self.query_context_pairs.is_empty());

        let mut num_queries = 0_u32;
        let mut causal_context_sum = 0_f64;
        for &(queries, context) in &self.query_context_pairs {
            assert!(queries > 0 && queries <= context);
            num_queries = num_queries
                .checked_add(queries)
                .expect("total query rows must fit u32");
            let mean_causal_context = f64::from(context) - f64::from(queries - 1) / 2.0;
            causal_context_sum += f64::from(queries) * mean_causal_context;
        }

        let num_requests = self.query_context_pairs.len() as u32;
        let mean_causal_context = causal_context_sum / f64::from(num_queries);
        let canonical_queries_per_request = f64::from(num_queries) / f64::from(num_requests);
        let canonical_endpoint = mean_causal_context + (canonical_queries_per_request - 1.0) / 2.0;
        (num_queries, num_requests, canonical_endpoint)
    }
}

impl SweepCoords for DsaSparseMlaPrefillKernelInput {
    fn coords(&self) -> Coords {
        let (num_queries, num_requests, upper_context) = self.work();
        Coords::new([
            f64::from(num_queries),
            f64::from(num_requests),
            upper_context,
        ])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["num_queries", "num_requests", "causal_equivalent_context"]
    }
}

fn canonical_pairs(num_queries: u32, num_requests: u32, context: u32) -> Option<Vec<(u32, u32)>> {
    if num_requests == 0 || num_requests > num_queries || num_requests as usize > MAX_REQUESTS {
        return None;
    }
    let base = num_queries / num_requests;
    let remainder = num_queries % num_requests;
    let pairs = (0..num_requests)
        .map(|request| (base + u32::from(request < remainder), context))
        .collect::<Vec<_>>();
    pairs
        .iter()
        .all(|&(queries, _)| queries <= context)
        .then_some(pairs)
}

/// The profiler's logical bytes for this launch: q, the per-row top-k
/// indices, the cache rows each query row reads, the output, and the max/lse
/// rows. Every term but the cache read scales with query rows.
fn logical_bytes(
    config: &DsaSparseMlaPrefillKernelConfig,
    input: &DsaSparseMlaPrefillKernelInput,
) -> f64 {
    let heads = f64::from(config.num_heads.get());
    let score_dim = f64::from(config.latent_dim.get() + config.rope_dim.get());
    let value_dim = f64::from(config.value_dim.get());
    let (mut queries, mut cache_rows) = (0.0, 0.0);
    for &(request_queries, context) in &input.query_context_pairs {
        queries += f64::from(request_queries);
        cache_rows += causal_rows::capped(request_queries, context, config.selected_k);
    }
    let per_query = f64::from(config.q_dtype.size_bytes()) * heads * score_dim
        + 4.0 * f64::from(config.selected_k)
        + f64::from(config.output_dtype.size_bytes()) * heads * value_dim
        + 8.0 * heads;
    queries * per_query + f64::from(config.cache_dtype.size_bytes()) * cache_rows * score_dim
}

pub struct DsaSparseMlaPrefillSpec;

impl KernelSpec for DsaSparseMlaPrefillSpec {
    type Config = DsaSparseMlaPrefillKernelConfig;
    type Input = DsaSparseMlaPrefillKernelInput;

    const KIND: KernelKind = "dsa_sparse_mla_prefill";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert!(config.max_model_len > 0);
        assert!(config.softmax_scale_denominator > 0);
        SweepGrid::new(vec![
            Axis::values([1, 8, 24, 64, 128, 256, 512, 1024, 1536, 2048, 8192]),
            Axis::values([1, 2, 4, 8, 16, 64]),
            Axis::values([
                1, 32, 128, 192, 256, 512, 1024, 2048, 4096, 8192, 65536, 1048576,
            ])
            .into_iter()
            .filter(|&context| context <= f64::from(config.max_model_len))
            .collect(),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_3d(|num_queries, num_requests, context| {
            canonical_pairs(num_queries as u32, num_requests as u32, context as u32).is_none()
        })
    }

    /// Past the grid the launch holds its bandwidth: one row per query token
    /// with no request dimension, it streams the cache rows each row selects,
    /// and B200 measurements up to 65535 rows and 512 requests stay within 7%
    /// of the edge's logical bandwidth while its TFLOPS moves by 30%.
    fn off_grid(
        config: &Self::Config,
        input: &Self::Input,
        _backend: &'static str,
    ) -> OffGrid<Self::Input> {
        OffGrid::Bandwidth(logical_bytes(config, input))
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_3d(|num_queries, num_requests, context| {
            let pairs = canonical_pairs(num_queries as u32, num_requests as u32, context as u32)
                .unwrap_or_else(|| vec![(1, 1)]);
            ArgsPayload::new()
                .with("backend", backend)
                .with("query_context_pairs", serde_json::json!(pairs))
                .with("num_heads", config.num_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("selected_k", config.selected_k)
                .with("latent_dim", config.latent_dim.get())
                .with("rope_dim", config.rope_dim.get())
                .with("value_dim", config.value_dim.get())
                .with(
                    "softmax_scale",
                    1.0 / f64::from(config.softmax_scale_denominator),
                )
                .with("q_dtype", config.q_dtype.as_str())
                .with("cache_dtype", config.cache_dtype.as_str())
                .with("index_dtype", config.index_dtype.clone())
                .with("output_dtype", config.output_dtype.as_str())
                .with("index_distribution", config.index_distribution.clone())
                .with("cache_layout", config.cache_layout.clone())
        })
    }
}

register_kernel!(DsaSparseMlaPrefillKernel, DsaSparseMlaPrefillSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn config() -> DsaSparseMlaPrefillKernelConfig {
        DsaSparseMlaPrefillKernelConfig {
            backends: vec!["flashinfer_trtllm_fp8"],
            gpu_name: "NVIDIA B200".to_string(),
            max_model_len: 1_048_576,
            num_heads: 16.into(),
            num_kv_heads: 1.into(),
            selected_k: 2048,
            latent_dim: 512.into(),
            rope_dim: 64.into(),
            value_dim: 512.into(),
            softmax_scale_denominator: 16,
            q_dtype: DType::Fp8E4m3,
            cache_dtype: DType::Fp8E4m3,
            index_dtype: "int32".to_string(),
            output_dtype: DType::Bf16,
            index_distribution: "recent_contiguous".to_string(),
            cache_layout: "hnd_paged_mqa_fp8_latent_rope".to_string(),
        }
    }

    #[test]
    fn ragged_batch_uses_causal_equivalent_context() {
        let input = DsaSparseMlaPrefillKernelInput {
            query_context_pairs: vec![(8, 128), (16, 256)],
        };
        assert_eq!(input.coords()[0], 24.0);
        assert_eq!(input.coords()[1], 2.0);
        assert!((input.coords()[2] - 212.666_666_666_666_66).abs() < 1e-12);
    }

    #[test]
    fn one_request_keeps_the_canonical_context_coordinate() {
        let input = DsaSparseMlaPrefillKernelInput {
            query_context_pairs: vec![(2048, 8192)],
        };
        assert_eq!(&*input.coords(), &[2048.0, 1.0, 8192.0]);
    }

    #[test]
    fn rounded_spec5_prefill_keeps_every_query_row() {
        let input = DsaSparseMlaPrefillKernelInput {
            query_context_pairs: vec![(8196, 16384)],
        };
        assert_eq!(&*input.coords(), &[8196.0, 1.0, 16384.0]);
    }

    #[test]
    fn canonical_payload_matches_python_schema() {
        let config = config();
        let grid = SweepGrid::new(vec![
            Axis::values([24]),
            Axis::values([2]),
            Axis::values([192]),
        ]);
        let payload =
            &DsaSparseMlaPrefillSpec::enumerate(&config, &grid, "flashinfer_trtllm_fp8")[0];
        let fields = payload.fields();

        assert_eq!(fields.len(), 15);
        assert_eq!(
            fields.get("query_context_pairs"),
            Some(&serde_json::json!([[12, 192], [12, 192]]))
        );
        assert_eq!(fields.get("softmax_scale"), Some(&Value::from(0.0625)));
        assert_eq!(fields.get("q_dtype"), Some(&Value::from("fp8_e4m3")));
        assert_eq!(fields.get("output_dtype"), Some(&Value::from("bf16")));
    }

    #[test]
    fn sweep_keeps_the_measured_b200_ramp_anchors() {
        let grid = DsaSparseMlaPrefillSpec::sweep_grid(&config());
        assert_eq!(
            grid.axes()[0],
            [1.0, 8.0, 24.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 1536.0, 2048.0, 8192.0]
        );
        assert_eq!(grid.axes()[1], [1.0, 2.0, 4.0, 8.0, 16.0, 64.0]);
        assert_eq!(
            grid.axes()[2],
            [
                1.0, 32.0, 128.0, 192.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 65536.0,
                1048576.0
            ]
        );
    }

    #[test]
    fn logical_bytes_match_the_profiler() {
        // profiling/runners/attention/dsa_sparse_mla_attention.py `_logical_bytes`
        // with q_bytes=1, cache_bytes=1, 16 heads, one (2, 3) request:
        // 1*2*16*576 + 4*2*2048 + (2+3)*576 + 2*2*16*512 + 8*2*16.
        let input = DsaSparseMlaPrefillKernelInput {
            query_context_pairs: vec![(2, 3)],
        };
        assert_eq!(
            logical_bytes(&config(), &input),
            18432.0 + 16384.0 + 2880.0 + 32768.0 + 256.0
        );
    }

    #[test]
    fn impossible_request_topology_is_masked() {
        let config = config();
        let grid = SweepGrid::new(vec![
            Axis::values([1]),
            Axis::values([2]),
            Axis::values([1]),
        ]);
        assert_eq!(
            DsaSparseMlaPrefillSpec::infeasible_mask(&config, &grid),
            vec![true]
        );
    }
}
