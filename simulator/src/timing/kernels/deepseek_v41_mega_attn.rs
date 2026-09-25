//! DeepSeek V4.1 FlashMLA mega attention: one layer's decode or prefill segment.
//!
//! One Python kind covers both segment methods of the fork's
//! `DeepseekV4MegaAttnAttention`; `mode` picks one and is static Config
//! identity, like the compress ratio (0 = SWA only, 1 or 2 = SWA plus an
//! NVFP4/MXFP8 compressed cache). Decode is one fused launch over the paged
//! caches. Prefill is a chunk loop over groups of up to four requests (a
//! compressed gather when the ratio is nonzero, the SWA gather, the index
//! combine and one fused attention launch per chunk); the whole loop is one
//! slot.
//!
//! The runtime input is the exact `query_context_pairs` batch. It projects to
//! three work coordinates, each computed exactly from the pairs:
//!
//! - `num_query_tokens`: flattened query rows;
//! - `mean_keys_per_token`: the mean number of keys each query token attends
//!   to, `min(pos + 1, window) + min((pos + 1) / ratio, index_topk)`. It
//!   saturates at 128 (ratio 0) or 640 once a token's position passes
//!   `ratio * index_topk`;
//! - `compressed_rows`: prefill only, `sum(context / ratio)`, the rows the
//!   compressed gather dequantizes. Decode reads only selected rows, so its
//!   axis is the single value 0.
//!
//! Measured B200 sensitivity decided the axes. At 2048 prefill tokens (ratio 2),
//! moving the mean keys from 240 to 640 raises time from 116 to 183 us. The
//! gather adds about 0.67 us per 1000 compressed rows: a 64-token chunk costs
//! 30 us at context 8192 and 117 us at context 131072 (ratio 1). Each extra
//! four-request chunk launch adds only about 1 us, so the launch count is
//! folded. Canonical grid batches with few mean keys use many small requests,
//! as real batches must.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

const WINDOW: u32 = 128;
const INDEX_TOPK: u32 = 512;
/// The profiler's request ceiling per call.
const MAX_REQUESTS: u32 = 256;
/// Decode rows the profiler accepts.
const MAX_DECODE_ROWS: u32 = 2048;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeepseekV41MegaAttnKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// `decode` or `prefill`: which segment method this slot times.
    pub mode: String,
    /// 0 (SWA only), 1 or 2.
    pub compress_ratio: u32,
    pub window_size: u32,
    pub index_topk: u32,
    /// The kernel's padded Q head count (64 or 128), not the live TP heads.
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub rope_dim: Dim,
    pub max_model_len: u32,
    pub max_num_batched_tokens: u32,
    pub prefill_chunk_size: u32,
    #[compute_dtype]
    pub q_dtype: DType,
    pub swa_cache_format: String,
    /// `none` iff `compress_ratio == 0`; otherwise `nvfp4` or `mxfp8`.
    pub compressed_cache_format: String,
    pub output_dtype: DType,
}

/// One segment's requests as `(query_len, context_len)`, context including the
/// query. Decode flattens every query token into its own row.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeepseekV41MegaAttnKernelInput {
    pub query_context_pairs: Vec<(u32, u32)>,
}

impl DeepseekV41MegaAttnKernelInput {
    /// `(query tokens, mean keys per token, compressed rows)` for `ratio`.
    fn work(&self, ratio: u32) -> (u32, f64, u64) {
        assert!(!self.query_context_pairs.is_empty());
        let (queries, keys, rows) = pair_work(&self.query_context_pairs, ratio);
        (queries, keys as f64 / f64::from(queries), rows)
    }
}

impl SweepCoords for DeepseekV41MegaAttnKernelInput {
    fn coords(&self) -> Coords {
        // The key and gather coordinates depend on the Config's ratio and mode;
        // `cache_coords` owns the real projection.
        let queries = self
            .query_context_pairs
            .iter()
            .map(|&(q, _)| q)
            .sum::<u32>();
        Coords::new([f64::from(queries), 0.0, 0.0])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["num_query_tokens", "mean_keys_per_token", "compressed_rows"]
    }
}

/// `sum_{x=1..n} min(x, cap)`.
fn sum_min(n: u64, cap: u64) -> u64 {
    let m = n.min(cap);
    m * (m + 1) / 2 + (n - m) * cap
}

/// `sum_{x=1..n} min(x / ratio, cap)` with floor division.
fn sum_min_div(n: u64, ratio: u64, cap: u64) -> u64 {
    let limit = ratio * (cap + 1) - 1;
    let m = n.min(limit);
    let (k, rem) = (m / ratio, m % ratio);
    let uncapped = if k == 0 {
        0
    } else {
        ratio * k * (k - 1) / 2 + k * (rem + 1)
    };
    uncapped + (n - m) * cap
}

/// Total keys of the tokens at positions `0..n`.
fn keys_prefix(n: u32, ratio: u32) -> u64 {
    let swa = sum_min(u64::from(n), u64::from(WINDOW));
    let extra = if ratio == 0 {
        0
    } else {
        sum_min_div(u64::from(n), u64::from(ratio), u64::from(INDEX_TOPK))
    };
    swa + extra
}

/// `(query tokens, total keys, compressed rows)` of a batch.
fn pair_work(pairs: &[(u32, u32)], ratio: u32) -> (u32, u64, u64) {
    let mut queries = 0_u32;
    let mut keys = 0_u64;
    let mut rows = 0_u64;
    for &(query, context) in pairs {
        assert!(
            query > 0 && query <= context,
            "each pair needs 0 < query <= context"
        );
        queries = queries
            .checked_add(query)
            .expect("query total must fit u32");
        keys += keys_prefix(context, ratio) - keys_prefix(context - query, ratio);
        if ratio > 0 {
            rows += u64::from(context / ratio);
        }
    }
    (queries, keys, rows)
}

fn validate_config(config: &DeepseekV41MegaAttnKernelConfig) {
    assert!(
        matches!(config.mode.as_str(), "decode" | "prefill"),
        "mode must be decode or prefill, got {:?}",
        config.mode
    );
    assert!(
        config.compress_ratio <= 2,
        "compress_ratio must be 0, 1 or 2"
    );
    assert_eq!(
        (
            config.window_size,
            config.index_topk,
            config.head_dim.get(),
            config.rope_dim.get(),
            config.prefill_chunk_size,
        ),
        (WINDOW, INDEX_TOPK, 512, 64, 4),
        "unsupported (window, index_topk, head_dim, rope_dim, prefill_chunk_size)"
    );
    assert!(matches!(config.num_heads.get(), 64 | 128));
    let expected_none = config.compress_ratio == 0;
    assert_eq!(
        config.compressed_cache_format == "none",
        expected_none,
        "compressed_cache_format is 'none' iff compress_ratio == 0"
    );
    assert!(config.max_num_batched_tokens >= 1 && config.max_model_len >= 1);
}

fn is_decode(config: &DeepseekV41MegaAttnKernelConfig) -> bool {
    config.mode == "decode"
}

/// `count` requests sharing `tokens` queries as evenly as possible, all at
/// context `context`.
fn uniform_pairs(tokens: u32, count: u32, context: u32) -> Vec<(u32, u32)> {
    let (base, remainder) = (tokens / count, tokens % count);
    (0..count)
        .map(|request| (base + u32::from(request < remainder), context))
        .collect()
}

/// The smallest shared context whose uniform batch comes closest to
/// `target_keys` mean keys per token, or `None` if even the longest context
/// falls short.
fn uniform_for_keys(
    tokens: u32,
    count: u32,
    target_keys: f64,
    ratio: u32,
    max_context: u32,
) -> Option<Vec<(u32, u32)>> {
    let lowest = tokens.div_ceil(count);
    if lowest > max_context {
        return None;
    }
    let mean = |context| {
        let (queries, keys, _) = pair_work(&uniform_pairs(tokens, count, context), ratio);
        keys as f64 / f64::from(queries)
    };
    let tolerance = key_tolerance(target_keys);
    if mean(max_context) < target_keys - tolerance || mean(lowest) > target_keys + tolerance {
        return None;
    }
    let (mut lo, mut hi) = (lowest, max_context);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if mean(mid) >= target_keys {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    let context = if lo > lowest && target_keys - mean(lo - 1) < mean(lo) - target_keys {
        lo - 1
    } else {
        lo
    };
    Some(uniform_pairs(tokens, count, context))
}

fn key_tolerance(keys: f64) -> f64 {
    (0.02 * keys).max(1.0)
}

fn row_tolerance(rows: f64) -> f64 {
    (0.02 * rows).max(1.0)
}

/// Whether `pairs` project to the grid point within tolerance.
fn matches_point(
    pairs: &[(u32, u32)],
    tokens: u32,
    keys: f64,
    rows: Option<f64>,
    ratio: u32,
) -> bool {
    let (queries, total_keys, total_rows) = pair_work(pairs, ratio);
    let mean_keys = total_keys as f64 / f64::from(queries);
    queries == tokens
        && (mean_keys - keys).abs() <= key_tolerance(keys)
        && rows.is_none_or(|rows| (total_rows as f64 - rows).abs() <= row_tolerance(rows))
}

/// A decode batch of `rows` flattened rows over at most 256 requests.
fn canonical_decode(
    rows: u32,
    keys: f64,
    config: &DeepseekV41MegaAttnKernelConfig,
) -> Option<Vec<(u32, u32)>> {
    if rows > MAX_DECODE_ROWS {
        return None;
    }
    let ratio = config.compress_ratio;
    let pairs = uniform_for_keys(
        rows,
        rows.min(MAX_REQUESTS),
        keys,
        ratio,
        config.max_model_len,
    )?;
    matches_point(&pairs, rows, keys, None, ratio).then_some(pairs)
}

/// A prefill batch with the fewest requests reaching `keys`: uniform requests
/// sized for the mean keys, then one trailing single-token request carrying
/// any missing compressed rows (its token counts toward the keys too).
fn canonical_prefill(
    tokens: u32,
    keys: f64,
    rows: f64,
    config: &DeepseekV41MegaAttnKernelConfig,
) -> Option<Vec<(u32, u32)>> {
    let ratio = config.compress_ratio;
    let max_context = config.max_model_len;
    let target_rows = (ratio > 0).then_some(rows);
    for count in 1..=tokens.min(MAX_REQUESTS - 1) {
        let Some(base) = uniform_for_keys(tokens, count, keys, ratio, max_context) else {
            continue;
        };
        if ratio == 0 {
            return matches_point(&base, tokens, keys, None, ratio).then_some(base);
        }
        let (_, _, base_rows) = pair_work(&base, ratio);
        if base_rows as f64 >= rows - row_tolerance(rows) {
            // More requests only add rows; the first reachable count decides.
            return matches_point(&base, tokens, keys, target_rows, ratio).then_some(base);
        }
        return with_filler(tokens, count, keys, rows, config);
    }
    None
}

fn with_filler(
    tokens: u32,
    count: u32,
    keys: f64,
    rows: f64,
    config: &DeepseekV41MegaAttnKernelConfig,
) -> Option<Vec<(u32, u32)>> {
    if tokens < 2 {
        return None;
    }
    let base_tokens = tokens - 1;
    (count.min(base_tokens)..=base_tokens.min(MAX_REQUESTS - 1))
        .find_map(|count| filler_batch(tokens, count, keys, rows, config))
}

fn filler_batch(
    tokens: u32,
    count: u32,
    keys: f64,
    rows: f64,
    config: &DeepseekV41MegaAttnKernelConfig,
) -> Option<Vec<(u32, u32)>> {
    let ratio = config.compress_ratio;
    let base_tokens = tokens - 1;
    let mut filler_context = 0_u32;
    let mut pairs = None;
    for _ in 0..4 {
        let filler_keys = if filler_context == 0 {
            0.0
        } else {
            (keys_prefix(filler_context, ratio) - keys_prefix(filler_context - 1, ratio)) as f64
        };
        let base_keys = (f64::from(tokens) * keys - filler_keys) / f64::from(base_tokens);
        let base = uniform_for_keys(base_tokens, count, base_keys, ratio, config.max_model_len)?;
        let (_, _, base_rows) = pair_work(&base, ratio);
        let missing = rows - base_rows as f64;
        if missing < 1.0 {
            return None;
        }
        let context = (missing.round() * f64::from(ratio)).min(f64::from(u32::MAX));
        if context > f64::from(config.max_model_len) {
            return None;
        }
        filler_context = context as u32;
        let mut candidate = base;
        candidate.push((1, filler_context));
        pairs = Some(candidate);
    }
    let pairs = pairs?;
    matches_point(&pairs, tokens, keys, Some(rows), ratio).then_some(pairs)
}

fn token_axis(config: &DeepseekV41MegaAttnKernelConfig) -> Vec<f64> {
    if is_decode(config) {
        return Axis::values([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]);
    }
    Axis::values([1, 4, 16, 64, 256, 512, 1024, 2048, 4096, 8192])
        .into_iter()
        .filter(|&tokens| tokens <= f64::from(config.max_num_batched_tokens))
        .collect()
}

fn key_axis(config: &DeepseekV41MegaAttnKernelConfig) -> Vec<f64> {
    match (is_decode(config), config.compress_ratio) {
        (_, 0) => Axis::values([1, 16, 32, 64, 96, 128]),
        (true, 1) => Axis::values([2, 64, 128, 256, 384, 512, 640]),
        (true, _) => Axis::values([1, 48, 96, 192, 320, 480, 640]),
        (false, 1) => Axis::values([2, 32, 96, 192, 320, 480, 640]),
        (false, _) => Axis::values([1, 32, 96, 192, 320, 480, 640]),
    }
}

fn row_axis(config: &DeepseekV41MegaAttnKernelConfig) -> Vec<f64> {
    if is_decode(config) || config.compress_ratio == 0 {
        return vec![0.0];
    }
    // Ratio 1 always gathers at least one row per query token.
    let lowest = if config.compress_ratio == 1 { 1 } else { 0 };
    let longest = config.max_model_len / config.compress_ratio;
    let mut axis = Axis::values([lowest, 256, 1024, 4096, 16384, 65536])
        .into_iter()
        .filter(|&rows| rows <= f64::from(longest))
        .collect::<Vec<_>>();
    if f64::from(longest) > *axis.last().expect("axis is non-empty") {
        axis.push(f64::from(longest));
    }
    axis
}

fn canonical_pairs(
    config: &DeepseekV41MegaAttnKernelConfig,
    coordinates: &[f64],
) -> Option<Vec<(u32, u32)>> {
    let (tokens, keys, rows) = (coordinates[0] as u32, coordinates[1], coordinates[2]);
    if is_decode(config) {
        canonical_decode(tokens, keys, config)
    } else {
        canonical_prefill(tokens, keys, rows, config)
    }
}

pub struct DeepseekV41MegaAttnSpec;

impl KernelSpec for DeepseekV41MegaAttnSpec {
    type Config = DeepseekV41MegaAttnKernelConfig;
    type Input = DeepseekV41MegaAttnKernelInput;

    const KIND: KernelKind = "deepseek_v41_mega_attn";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        validate_config(config);
        SweepGrid::new(vec![token_axis(config), key_axis(config), row_axis(config)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        let (queries, mean_keys, rows) = input.work(config.compress_ratio);
        let rows = if is_decode(config) { 0.0 } else { rows as f64 };
        Coords::new([f64::from(queries), mean_keys, rows])
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand(|coordinates| canonical_pairs(config, coordinates).is_none())
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand(|coordinates| {
            // Infeasible cells get a placeholder the engine drops before profiling.
            let pairs = canonical_pairs(config, coordinates).unwrap_or_else(|| vec![(1, 1)]);
            ArgsPayload::new()
                .with("backend", backend)
                .with("mode", config.mode.clone())
                .with("query_context_pairs", serde_json::json!(pairs))
                .with("compress_ratio", config.compress_ratio)
                .with("window_size", config.window_size)
                .with("index_topk", config.index_topk)
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("rope_dim", config.rope_dim.get())
                .with("max_model_len", config.max_model_len)
                .with("max_num_batched_tokens", config.max_num_batched_tokens)
                .with("prefill_chunk_size", config.prefill_chunk_size)
                .with("q_dtype", config.q_dtype.as_str())
                .with("swa_cache_format", config.swa_cache_format.clone())
                .with(
                    "compressed_cache_format",
                    config.compressed_cache_format.clone(),
                )
                .with("output_dtype", config.output_dtype.as_str())
        })
    }
}

register_kernel!(DeepseekV41MegaAttnKernel, DeepseekV41MegaAttnSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(mode: &str, ratio: u32) -> DeepseekV41MegaAttnKernelConfig {
        serde_json::from_value(serde_json::json!({
            "backends": ["flashmla_mega"],
            "gpu_name": "NVIDIA B200",
            "mode": mode,
            "compress_ratio": ratio,
            "window_size": 128,
            "index_topk": 512,
            "num_heads": 64,
            "head_dim": 512,
            "rope_dim": 64,
            "max_model_len": 131072,
            "max_num_batched_tokens": 2048,
            "prefill_chunk_size": 4,
            "q_dtype": "bf16",
            "swa_cache_format": "mxfp8",
            "compressed_cache_format": if ratio == 0 { "none" } else { "nvfp4" },
            "output_dtype": "fp8_e4m3",
        }))
        .unwrap()
    }

    fn brute_keys(pairs: &[(u32, u32)], ratio: u32) -> u64 {
        pairs
            .iter()
            .flat_map(|&(q, c)| (c - q)..c)
            .map(|pos| {
                let x = u64::from(pos) + 1;
                x.min(128)
                    + if ratio == 0 {
                        0
                    } else {
                        (x / u64::from(ratio)).min(512)
                    }
            })
            .sum()
    }

    /// Catches a closed-form key count that drifts from the runner's per-token
    /// `min(pos + 1, 128) + min((pos + 1) // ratio, 512)` lengths.
    #[test]
    fn closed_form_keys_match_per_token_lengths() {
        for ratio in 0..=2 {
            for n in [0, 1, 2, 3, 127, 128, 129, 511, 512, 1023, 1024, 1025, 5000] {
                assert_eq!(
                    keys_prefix(n, ratio),
                    brute_keys(&[(n.max(1), n.max(1))], ratio) * u64::from(n > 0)
                );
            }
        }
    }

    /// Catches a projection that averages contexts instead of per-token keys,
    /// or drops the gather rows of the capture's mixed prefill (iteration 310).
    #[test]
    fn mixed_prefill_projects_exact_keys_and_gather_rows() {
        let pairs = vec![(1912, 1912), (91, 4096)];
        let input = DeepseekV41MegaAttnKernelInput {
            query_context_pairs: pairs.clone(),
        };
        let coords = DeepseekV41MegaAttnSpec::cache_coords(&config("prefill", 2), &input);
        assert_eq!(coords[0], 2003.0);
        assert_eq!(coords[1], brute_keys(&pairs, 2) as f64 / 2003.0);
        assert_eq!(coords[2], f64::from(1912 / 2 + 4096 / 2));
        let decode = DeepseekV41MegaAttnSpec::cache_coords(&config("decode", 2), &input);
        assert_eq!(decode[2], 0.0);
    }

    /// Catches canonical profiling batches that do not sit on the grid point
    /// they fill, and a grid past the 500 feasible-coordinate ceiling.
    #[test]
    fn canonical_batches_project_onto_their_grid_points() {
        for mode in ["decode", "prefill"] {
            for ratio in 0..=2 {
                let config = config(mode, ratio);
                let grid = DeepseekV41MegaAttnSpec::sweep_grid(&config);
                let mask = DeepseekV41MegaAttnSpec::infeasible_mask(&config, &grid);
                let feasible = mask.iter().filter(|&&drop| !drop).count();
                assert!(
                    feasible > 0 && feasible <= 500,
                    "{mode} r{ratio}: {feasible}"
                );
                grid.expand(|point| {
                    let Some(pairs) = canonical_pairs(&config, point) else {
                        return;
                    };
                    assert!(pairs.len() <= MAX_REQUESTS as usize);
                    assert!(pairs.iter().all(|&(_, c)| c <= config.max_model_len));
                    let input = DeepseekV41MegaAttnKernelInput {
                        query_context_pairs: pairs,
                    };
                    let coords = DeepseekV41MegaAttnSpec::cache_coords(&config, &input);
                    assert_eq!(coords[0], point[0]);
                    assert!(
                        (coords[1] - point[1]).abs() <= key_tolerance(point[1]),
                        "{point:?} {coords:?}"
                    );
                    assert!(
                        (coords[2] - point[2]).abs() <= row_tolerance(point[2]),
                        "{point:?} {coords:?}"
                    );
                });
                eprintln!("{mode} r{ratio}: {} cells, {feasible} feasible", mask.len());
            }
        }
    }

    /// Catches a ratio-0 layer configured with a compressed cache.
    #[test]
    #[should_panic(expected = "compressed_cache_format")]
    fn swa_only_layer_rejects_compressed_cache() {
        let mut config = config("decode", 0);
        config.compressed_cache_format = "nvfp4".into();
        DeepseekV41MegaAttnSpec::sweep_grid(&config);
    }
}
