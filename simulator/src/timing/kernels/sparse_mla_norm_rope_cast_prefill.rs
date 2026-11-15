//! FlashMLA's fused prefill core attention as DeepInfer calls it
//! (`fused_norm_rope_attn_rope_cast.prefill`, one
//! `sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::fwd_kernel`
//! launch per layer and GPU).
//!
//! Per query row: Q RMSNorm, RoPE, sparse attention with one latent KV head and
//! a sink over the `window_size (+ index_topk)` rows its index row names in a
//! gathered bf16 KV buffer, inverse RoPE and the FP8 / UE8M0 output cast. The
//! gathers that fill the buffer and the index concatenation are separate
//! launches and other slots.
//!
//! The launch runs one 512-thread CTA per query row (223 KB shared memory, so
//! one CTA per SM). Its time follows the index width (Config) and the rows,
//! not the count of valid index entries (B200: 4 x (140, 131072) at ratio 1
//! took 62.97 us with YoCo's clipped window and 63.04 us without; 16384 rows at
//! ratio 2 took 1503 us with 512 valid compressed columns and 1601 us with
//! none). The compressed rows a request's top-k spreads over do matter: at
//! 16384 rows and ratio 2, contexts 16384 / 131072 / 524288 took 1419 / 1503 /
//! 1799 us, while the same rows in 1, 4 or 16 requests at context 131072 took
//! 1488 / 1503 / 1543 us. The cache therefore runs on two coordinates:
//!
//! - `num_query_tokens`: the rows, summed over requests;
//! - `context_len`: the query-weighted mean context.
//!
//! A grid point's batch puts its rows in as few requests of that context as
//! hold them (`ceil(rows / context)`, rows split evenly). Known residual: many
//! short requests at a long context run over that batch, since each request's
//! rows reuse only their own top-k rows. On B200 at context 131072, 4 x 140
//! rows (ratio 1) took 63.0 us against the cache's 56.6, and 16 x 140 took
//! 249.4 against 225.0; 16 x 4096 rows (ratio 2) took 5880 against 5708.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// Query rows per rank. Steps of 148 rows (one CTA wave on B200) up to 2368,
/// then powers of two.
const TOKENS: [u32; 17] = [
    1, 32, 74, 148, 296, 444, 592, 888, 1184, 1776, 2368, 4096, 8192, 16384, 32768, 65536, 131072,
];
/// Context tokens per request. 1048576 is DeepSeek-V4.1's
/// `max_position_embeddings`, the runner's ceiling.
const CONTEXTS: [u32; 8] = [1024, 4096, 16384, 65536, 131072, 262144, 524288, 1_048_576];

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparseMlaNormRopeCastPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Sliding-window columns of each index row.
    pub window_size: u32,
    /// 0 for a window-only layer; otherwise the layer's compression ratio.
    pub compress_ratio: u32,
    /// Indexer top-k columns after the window; 0 iff `compress_ratio == 0`.
    pub index_topk: u32,
    /// Query heads (64 or 128): with attention data parallel, the model's heads.
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub rope_dim: Dim,
    /// `num_heads / 8`: the output projection's groups.
    pub n_wv_group: u32,
    pub enable_q_norm: bool,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
    pub output_dtype: DType,
    /// Largest per-rank query rows the deployment runs. Bounds the row axis only.
    pub max_tokens_per_rank: u32,
    /// Longest request context the deployment admits (`max_model_len`).
    /// Bounds the context axis only.
    pub max_context: u32,
}

/// The step's prefill requests as `(query rows, context)`, context including
/// the query rows. A YoCo consumer layer passes its kept rows (e.g. 140) at the
/// request's full context.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparseMlaNormRopeCastPrefillKernelInput {
    pub query_context_pairs: Vec<(u32, u32)>,
}

impl SweepCoords for SparseMlaNormRopeCastPrefillKernelInput {
    fn coords(&self) -> Coords {
        Coords::new(project(&self.query_context_pairs))
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["num_query_tokens", "context_len"]
    }
}

/// `[rows, query-weighted mean context]`.
fn project(pairs: &[(u32, u32)]) -> [f64; 2] {
    assert!(!pairs.is_empty(), "query_context_pairs must be non-empty");
    let mut rows = 0_u64;
    let mut weighted = 0_u64;
    for &(query, context) in pairs {
        assert!(
            query > 0 && query <= context,
            "each pair needs 0 < query <= context"
        );
        rows += u64::from(query);
        weighted += u64::from(query) * u64::from(context);
    }
    [rows as f64, weighted as f64 / rows as f64]
}

/// The axis values up to and including the first at or above `max`.
fn cut(values: &[u32], max: u32) -> Vec<f64> {
    let end = values
        .iter()
        .position(|&value| value >= max)
        .map_or(values.len(), |index| index + 1);
    Axis::values(values[..end].iter().copied())
}

/// A grid point's batch: `ceil(rows / context)` requests of `context`, the
/// rows split as evenly as possible.
pub fn canonical_pairs(rows: u32, context: u32) -> Vec<(u32, u32)> {
    let requests = rows.div_ceil(context);
    let (base, extra) = (rows / requests, rows % requests);
    (0..requests)
        .map(|index| (base + u32::from(index < extra), context))
        .collect()
}

pub struct SparseMlaNormRopeCastPrefillSpec;

impl KernelSpec for SparseMlaNormRopeCastPrefillSpec {
    type Config = SparseMlaNormRopeCastPrefillKernelConfig;
    type Input = SparseMlaNormRopeCastPrefillKernelInput;

    const KIND: KernelKind = "sparse_mla_norm_rope_cast_prefill";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            cut(&TOKENS, config.max_tokens_per_rank),
            cut(&CONTEXTS, config.max_context),
        ])
    }

    /// Rows times per-row cost, the per-row cost moving with the context:
    /// the time is the product of the two axes' effects.
    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            (config.compress_ratio == 0) == (config.index_topk == 0),
            "compress_ratio and index_topk are both 0 (window only) or both positive"
        );
        anyhow::ensure!(
            matches!(config.num_heads.get(), 64 | 128),
            "num_heads must be 64 or 128"
        );
        anyhow::ensure!(
            config.n_wv_group * 8 == config.num_heads.get(),
            "n_wv_group must be num_heads / 8"
        );
        anyhow::ensure!(
            config.max_tokens_per_rank >= 1 && config.max_context >= 1,
            "max_tokens_per_rank and max_context must be positive"
        );
        Ok(())
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|rows, context| {
            let pairs = canonical_pairs(rows as u32, context as u32);
            ArgsPayload::new()
                .with("backend", backend)
                .with("query_context_pairs", serde_json::json!(pairs))
                .with("window_size", config.window_size)
                .with("compress_ratio", config.compress_ratio)
                .with("index_topk", config.index_topk)
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("rope_dim", config.rope_dim.get())
                .with("n_wv_group", config.n_wv_group)
                .with("enable_q_norm", config.enable_q_norm)
                .with("q_dtype", config.q_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("output_dtype", config.output_dtype.as_str())
        })
    }
}

register_kernel!(
    SparseMlaNormRopeCastPrefillKernel,
    SparseMlaNormRopeCastPrefillSpec
);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(ratio: u32, topk: u32) -> SparseMlaNormRopeCastPrefillKernelConfig {
        serde_json::from_value(serde_json::json!({
            "backends": ["flashmla_deepinfer"],
            "gpu_name": "NVIDIA B200",
            "window_size": 128,
            "compress_ratio": ratio,
            "index_topk": topk,
            "num_heads": 64,
            "head_dim": 512,
            "rope_dim": 64,
            "n_wv_group": 8,
            "enable_q_norm": true,
            "q_dtype": "bf16",
            "kv_dtype": "bf16",
            "output_dtype": "fp8_e4m3",
            "max_tokens_per_rank": 65536,
            "max_context": 131072,
        }))
        .unwrap()
    }

    /// Catches payload fields drifting from the Python KernelArgs.
    #[test]
    fn enumerate_emits_every_python_arg() {
        let config = config(2, 512);
        let payloads = SparseMlaNormRopeCastPrefillSpec::enumerate(
            &config,
            &SweepGrid::new(vec![Axis::values([560]), Axis::values([131072])]),
            "flashmla_deepinfer",
        );
        let mut names: Vec<&str> = payloads[0].fields().keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "backend",
                "compress_ratio",
                "enable_q_norm",
                "head_dim",
                "index_topk",
                "kv_dtype",
                "n_wv_group",
                "num_heads",
                "output_dtype",
                "q_dtype",
                "query_context_pairs",
                "rope_dim",
                "window_size",
            ]
        );
        assert!(SparseMlaNormRopeCastPrefillSpec::validate_config(&config).is_ok());
        assert!(SparseMlaNormRopeCastPrefillSpec::validate_config(&self::config(0, 512)).is_err());
    }

    /// Catches a grid point whose batch does not hold its own coordinates.
    #[test]
    fn canonical_pairs_project_back_to_the_grid_point() {
        for (rows, context) in [(1, 1024), (560, 131072), (65536, 16384), (2368, 1024)] {
            let pairs = canonical_pairs(rows, context);
            assert!(pairs.iter().all(|&(q, c)| q <= c));
            assert_eq!(project(&pairs), [f64::from(rows), f64::from(context)]);
        }
        assert_eq!(
            canonical_pairs(2368, 1024),
            vec![(790, 1024), (789, 1024), (789, 1024)]
        );
    }

    /// The sweep shapes: 16 x (4096, 131072) and YoCo's 16 x (140, 131072).
    #[test]
    fn project_weights_context_by_rows() {
        assert_eq!(project(&[(4096, 131072); 16]), [65536.0, 131072.0]);
        assert_eq!(project(&[(140, 131072); 4]), [560.0, 131072.0]);
        assert_eq!(project(&[(100, 1000), (300, 3000)]), [400.0, 2500.0]);
    }

    /// The grid stays within the 500-point ceiling at the widest config.
    #[test]
    fn grid_is_bounded() {
        let mut config = config(1, 512);
        config.max_tokens_per_rank = 131072;
        config.max_context = 1_048_576;
        let grid = SparseMlaNormRopeCastPrefillSpec::sweep_grid(&config);
        let count = grid.expand_2d(|_, _| ()).len();
        assert_eq!(count, 17 * 8);
        let grid = SparseMlaNormRopeCastPrefillSpec::sweep_grid(&self::config(2, 512));
        assert_eq!(grid.expand_2d(|_, _| ()).len(), 16 * 5);
    }
}
