//! DeepSeek V4.1 FlashMLA mega attention: one layer's decode or prefill segment.
//!
//! One Python kind covers both segment methods of the fork's
//! `DeepseekV4MegaAttnAttention`. `mode` picks one and is static Config
//! identity, like the compress ratio (0 = SWA only; 1 or 2 = SWA plus a
//! compressed cache). Decode is one fused launch over the paged caches. Prefill
//! is a loop over the chunk plan (a compressed gather when the ratio is
//! nonzero, the SWA gather, the index combine and one fused attention launch
//! per chunk). One prefill input is one chunk of that loop: the plan is the
//! only place the deployment's `max_model_len`, `max_num_batched_tokens` and
//! `PREFILL_CHUNK_SIZE` reach the segment, and a chunk's launches depend only
//! on its own requests, so the rows are keyed by geometry alone. The caller
//! runs [`prefill_chunk_plan`] with the deployment's values and sums the
//! chunks into its one slot (`op/attention/deepseek_v41_mega_attn.rs`).
//!
//! The runtime input is the exact `query_context_pairs` batch. Each token
//! attends to `min(pos + 1, window) + min((pos + 1) / ratio, index_topk)` keys,
//! which saturates at 128 (ratio 0) or 640. The batch projects to three cache
//! coordinates:
//!
//! - `num_query_tokens`: flattened query rows.
//! - `keys_per_token`. In decode, this is the mean over CTA waves of each
//!   wave's largest key count (rows sorted longest first). The launch runs one
//!   CTA per row and 64 heads, so a wave lasts as long as its longest row. A
//!   single-wave batch with a few long rows costs the same as an all-long batch.
//!   In prefill, this is the mean key count per token; the fused attention scales
//!   with it.
//! - `excess_gather_rows`: prefill with ratio > 0 only. The compressed gather
//!   splits a fixed pool of 2048 workers across the requests of the chunk, so
//!   the cost follows the padded area `chunk requests * max rows / ratio`, not
//!   the row total: two requests with 25000 rows cost the same as one with
//!   50000. The coordinate is that area minus `tokens / ratio + index_topk`.
//!   That is the most area any (tokens, keys) point needs without extra
//!   context, so the lowest grid plane is reachable across the whole
//!   (tokens, keys) plane.
//!
//! Decode at 16 or 32 heads is vLLM's swap-AB kernel (backend
//! `swapab_cutedsl`, fork `nvidia/ops/swapab_decode_attn.py`), which
//! `_forward_decode_mega` takes instead of the mega kernel for decode segments
//! of at most `_SWAPAB_MAX_TOKENS` rows on compressed layers with 16 or 32
//! live heads (`flash_mla_mega_attn.py` `_swapab_decode_supported`). It runs
//! only the live heads, so `num_heads` is the live count there. Each row is one
//! thread-block cluster of [`swapab_tiles`] CTAs (64-slot SWA tiles, 128-slot
//! compressed tiles; tiles past a row's length exit early), so at <= 16 rows
//! the launch is one wave and its time follows the longest row.
//!
//! Measured B200 behaviour behind the axes: at 2048 prefill tokens (ratio 2),
//! mean keys 240 -> 640 moves time 116 -> 183 us. The gather alone costs about
//! 80 us for 100000 rows (ratio 1). Decode steps at every 148 rows (148 SMs).
//! Known residual: the gather's per-chunk cost varies up to about 2x with the
//! chunk size (power-of-two worker counts are slower); the area coordinate
//! scales it linearly.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

const WINDOW: u32 = 128;
const INDEX_TOPK: u32 = 512;
/// One `mxfp8` cache record: 512 FP8 E4M3 values, then 16 UE8M0 scales (one
/// per 32 dims). The sliding-window cache always uses it.
pub const MXFP8_RECORD_BYTES: u32 = 528;
/// One `nvfp4` compressed-cache record: 512 E2M1 values packed two per byte,
/// then 32 scale bytes.
pub const NVFP4_RECORD_BYTES: u32 = 288;
/// The profiler's request ceiling per call.
const MAX_REQUESTS: u32 = 256;
/// The longest context the profiler builds: DeepSeek-V4.1's
/// `max_position_embeddings`, which bounds every deployment's `max_model_len`.
const MAX_CONTEXT: u32 = 1_048_576;
/// Decode rows the profiler accepts.
const MAX_DECODE_ROWS: u32 = 2048;
/// SMs on the only supported GPU (NVIDIA B200); the decode wave width at 64 heads.
const B200_SMS: u32 = 148;
/// `_SWAPAB_MAX_TOKENS` (fork `flash_mla_mega_attn.py`): the most decode rows
/// the swap-AB kernel takes.
pub const SWAPAB_MAX_TOKENS: u32 = 16;
/// KV slots per swap-AB CTA: SWA tiles (`_SWA_ROWS`) and compressed tiles
/// (`_TILE`), and the portable cluster bound on a row's tiles (`_MAX_CLUSTER`).
const SWAPAB_SWA_ROWS: u32 = 64;
const SWAPAB_EXTRA_ROWS: u32 = 128;
const SWAPAB_MAX_CLUSTER: u32 = 8;

/// CTAs in one swap-AB row's cluster (`swapab_decode_supported`).
pub fn swapab_tiles(window_size: u32, index_topk: u32) -> u32 {
    window_size.div_ceil(SWAPAB_SWA_ROWS) + index_topk.div_ceil(SWAPAB_EXTRA_ROWS)
}

/// Whether vLLM gives a layer's small decode steps to the swap-AB kernel:
/// a compressed layer with 16 or 32 live heads whose tiles fit one cluster
/// (`_swapab_decode_supported` + `swapab_decode_supported`).
pub fn swapab_layer(
    compress_ratio: u32,
    live_heads: u32,
    window_size: u32,
    index_topk: u32,
) -> bool {
    let tiles = swapab_tiles(window_size, index_topk);
    compress_ratio > 0 && matches!(live_heads, 16 | 32) && (1..=SWAPAB_MAX_CLUSTER).contains(&tiles)
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompressedSparseMlaRopeCastKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// `decode` or `prefill`: which segment method this slot times.
    pub mode: String,
    /// 0 (SWA only), 1 or 2.
    pub compress_ratio: u32,
    pub window_size: u32,
    pub index_topk: u32,
    /// Q heads the kernel runs: the mega kernel's padded count (64 or 128),
    /// or the swap-AB kernel's live count (16 or 32, decode on compressed
    /// layers only).
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub rope_dim: Dim,
    #[compute_dtype]
    pub q_dtype: DType,
    pub swa_cache_format: String,
    /// `none` iff `compress_ratio == 0`; otherwise `nvfp4` or `mxfp8`.
    pub compressed_cache_format: String,
    pub output_dtype: DType,
}

/// One segment's requests as `(query_len, context_len)`, context including the
/// query. Decode flattens every query token into its own row; prefill holds the
/// requests of one planner chunk ([`prefill_chunk_plan`]).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompressedSparseMlaRopeCastKernelInput {
    pub query_context_pairs: Vec<(u32, u32)>,
}

impl SweepCoords for CompressedSparseMlaRopeCastKernelInput {
    fn coords(&self) -> Coords {
        // The key and gather coordinates depend on the Config's mode and ratio;
        // `cache_coords` owns the real projection.
        let queries = self
            .query_context_pairs
            .iter()
            .map(|&(q, _)| q)
            .sum::<u32>();
        Coords::new([f64::from(queries), 0.0, 0.0])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["num_query_tokens", "keys_per_token", "excess_gather_rows"]
    }
}

/// Keys of the token at position `x - 1`.
fn token_keys(x: u32, ratio: u32) -> u32 {
    x.min(WINDOW)
        + if ratio == 0 {
            0
        } else {
            (x / ratio).min(INDEX_TOPK)
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

/// `(query tokens, mean keys per token)` of a batch.
fn mean_keys(pairs: &[(u32, u32)], ratio: u32) -> (u32, f64) {
    assert!(!pairs.is_empty(), "query_context_pairs must be non-empty");
    let mut queries = 0_u32;
    let mut keys = 0_u64;
    for &(query, context) in pairs {
        assert!(
            query > 0 && query <= context,
            "each pair needs 0 < query <= context"
        );
        queries = queries
            .checked_add(query)
            .expect("query total must fit u32");
        keys += keys_prefix(context, ratio) - keys_prefix(context - query, ratio);
    }
    (queries, keys as f64 / f64::from(queries))
}

/// `(rows, mean over waves of the wave's largest row keys)`, rows sorted
/// longest first and `per_wave` rows to a wave.
fn wave_keys(pairs: &[(u32, u32)], ratio: u32, per_wave: usize) -> (u32, f64) {
    let (rows, _) = mean_keys(pairs, ratio);
    let mut keys = pairs
        .iter()
        .flat_map(|&(query, context)| {
            (context - query + 1..=context).map(move |x| token_keys(x, ratio))
        })
        .collect::<Vec<_>>();
    keys.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
    let maxima = keys.iter().step_by(per_wave).copied().collect::<Vec<_>>();
    let total = maxima.iter().map(|&keys| f64::from(keys)).sum::<f64>();
    (rows, total / maxima.len() as f64)
}

fn validate_config(config: &CompressedSparseMlaRopeCastKernelConfig) {
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
        ),
        (WINDOW, INDEX_TOPK, 512, 64),
        "unsupported (window, index_topk, head_dim, rope_dim)"
    );
    match config.num_heads.get() {
        64 | 128 => {}
        16 | 32 => assert!(
            is_decode(config) && config.compress_ratio > 0,
            "16 or 32 heads (swap-AB) is decode on a compressed layer only"
        ),
        other => panic!("num_heads must be 64 or 128 (padded) or 16 or 32 (live), got {other}"),
    }
    assert_eq!(
        config.compressed_cache_format == "none",
        config.compress_ratio == 0,
        "compressed_cache_format is 'none' iff compress_ratio == 0"
    );
}

fn is_decode(config: &CompressedSparseMlaRopeCastKernelConfig) -> bool {
    config.mode == "decode"
}

/// Whether the config is the swap-AB kernel's: it alone runs live heads.
fn is_swapab(config: &CompressedSparseMlaRopeCastKernelConfig) -> bool {
    config.num_heads.get() < 64
}

/// Decode rows per CTA wave: one CTA per row and 64 heads for the mega
/// kernel, one cluster of [`swapab_tiles`] CTAs per row for swap-AB.
fn rows_per_wave(config: &CompressedSparseMlaRopeCastKernelConfig) -> u32 {
    if is_swapab(config) {
        return B200_SMS / swapab_tiles(config.window_size, config.index_topk);
    }
    B200_SMS * 64 / config.num_heads.get()
}

/// The fork's prefill chunk plan (`DeepseekSparseSWAMetadata.
/// get_prefill_chunk_plan`, `v1/attention/backends/mla/sparse_swa.py`): the
/// requests, in order, grouped greedily into chunks whose padded area
/// `requests * (max compressed rows + max gathered SWA rows)` stays within
/// `prefill_chunk_size * (ceil(max_model_len / ratio) + window +
/// max_num_batched_tokens)`. Each range is one chunk; a chunk always takes its
/// first request, and can hold more than `prefill_chunk_size` short ones.
pub fn prefill_chunk_plan(
    pairs: &[(u32, u32)],
    compress_ratio: u32,
    window_size: u32,
    max_model_len: u32,
    max_num_batched_tokens: u32,
    prefill_chunk_size: u32,
) -> Vec<std::ops::Range<usize>> {
    let ratio = compress_ratio;
    let budget = u64::from(prefill_chunk_size)
        * (if ratio == 0 {
            0
        } else {
            u64::from(max_model_len.div_ceil(ratio))
        } + u64::from(window_size)
            + u64::from(max_num_batched_tokens));
    let compressed = |(_, context): (u32, u32)| {
        if ratio == 0 {
            0
        } else {
            u64::from(context / ratio)
        }
    };
    let gathered =
        |(query, context): (u32, u32)| u64::from(query + (context - query).min(window_size - 1));
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < pairs.len() {
        let mut max_compressed = compressed(pairs[start]);
        let mut max_gathered = gathered(pairs[start]);
        let mut end = start + 1;
        while end < pairs.len() {
            let next_compressed = max_compressed.max(compressed(pairs[end]));
            let next_gathered = max_gathered.max(gathered(pairs[end]));
            if (end - start + 1) as u64 * (next_compressed + next_gathered) > budget {
                break;
            }
            (max_compressed, max_gathered) = (next_compressed, next_gathered);
            end += 1;
        }
        chunks.push(start..end);
        start = end;
    }
    chunks
}

/// The padded compressed-gather area of one chunk: `requests * max
/// compressed rows`.
fn gather_area(pairs: &[(u32, u32)], ratio: u32) -> u64 {
    if ratio == 0 {
        return 0;
    }
    let longest = pairs.iter().map(|&(_, c)| c / ratio).max().unwrap_or(0);
    pairs.len() as u64 * u64::from(longest)
}

/// Bytes of one cache record: the SWA cache is always MXFP8; the compressed
/// cache is NVFP4 or MXFP8.
fn record_bytes(format: &str) -> f64 {
    match format {
        "mxfp8" => f64::from(MXFP8_RECORD_BYTES),
        "nvfp4" => f64::from(NVFP4_RECORD_BYTES),
        other => panic!("unknown cache format {other:?}"),
    }
}

/// The batch's logical bytes as the profiler counts them: Q in BF16 and the
/// FP8 output with one scale byte per 32, plus, in decode, the index words of
/// every row's window and top-k slots and the selected records; in prefill,
/// each gathered record read and written back as BF16, then one BF16 row and
/// one index word per selected key.
fn logical_bytes(
    config: &CompressedSparseMlaRopeCastKernelConfig,
    input: &CompressedSparseMlaRopeCastKernelInput,
) -> f64 {
    let ratio = config.compress_ratio;
    let heads = f64::from(config.num_heads.get());
    let head_dim = f64::from(config.head_dim.get());
    let swa_record = record_bytes(&config.swa_cache_format);
    let extra_record = if ratio == 0 {
        0.0
    } else {
        record_bytes(&config.compressed_cache_format)
    };
    let (mut rows, mut swa_keys, mut extra_keys) = (0.0, 0_u64, 0_u64);
    let (mut gathered_swa, mut gathered_extra) = (0_u64, 0_u64);
    for &(query, context) in &input.query_context_pairs {
        let (n, before) = (u64::from(context), u64::from(context - query));
        rows += f64::from(query);
        swa_keys += sum_min(n, u64::from(WINDOW)) - sum_min(before, u64::from(WINDOW));
        gathered_swa += u64::from(query + (context - query).min(WINDOW - 1));
        if ratio > 0 {
            let (ratio, topk) = (u64::from(ratio), u64::from(INDEX_TOPK));
            extra_keys += sum_min_div(n, ratio, topk) - sum_min_div(before, ratio, topk);
            gathered_extra += n / ratio;
        }
    }
    let (swa_keys, extra_keys) = (swa_keys as f64, extra_keys as f64);
    let q_and_out = rows * heads * (2.0 * head_dim + head_dim + head_dim / 32.0);
    if is_decode(config) {
        let slots = f64::from(WINDOW + if ratio == 0 { 0 } else { INDEX_TOPK });
        return q_and_out + rows * 4.0 * slots + swa_keys * swa_record + extra_keys * extra_record;
    }
    let bf16_row = 2.0 * head_dim;
    q_and_out
        + gathered_swa as f64 * (swa_record + bf16_row)
        + gathered_extra as f64 * (extra_record + bf16_row)
        + (swa_keys + extra_keys) * (bf16_row + 4.0)
}

/// The gather area every (tokens, keys) point reaches without extra context.
fn area_floor(tokens: u32, ratio: u32) -> f64 {
    f64::from(tokens.div_ceil(ratio) + INDEX_TOPK)
}

/// `(tokens, keys, excess gather rows)` under this config.
fn project(pairs: &[(u32, u32)], config: &CompressedSparseMlaRopeCastKernelConfig) -> [f64; 3] {
    let ratio = config.compress_ratio;
    if is_decode(config) {
        let (rows, keys) = wave_keys(pairs, ratio, rows_per_wave(config) as usize);
        return [f64::from(rows), keys, 0.0];
    }
    let (tokens, keys) = mean_keys(pairs, ratio);
    let excess = if ratio == 0 {
        0.0
    } else {
        (gather_area(pairs, ratio) as f64 - area_floor(tokens, ratio)).max(0.0)
    };
    [f64::from(tokens), keys, excess]
}

/// `count` requests sharing `tokens` queries as evenly as possible, all at
/// context `context`.
fn uniform_pairs(tokens: u32, count: u32, context: u32) -> Vec<(u32, u32)> {
    let (base, remainder) = uniform_split(tokens, count);
    (0..count)
        .map(|request| (base + u32::from(request < remainder), context))
        .collect()
}

/// How [`uniform_pairs`] shares `tokens` queries over `count` requests:
/// `(base, remainder)`, the first `remainder` requests one query longer. The
/// closed forms below read the batch through this, so they cannot drift from
/// the batch they stand in for.
fn uniform_split(tokens: u32, count: u32) -> (u32, u32) {
    (tokens / count, tokens % count)
}

/// [`uniform_split`] plus the longest query, checked as [`mean_keys`] checks
/// each pair of the built batch.
fn checked_uniform_split(tokens: u32, count: u32, context: u32) -> (u32, u32, u32) {
    let (base, remainder) = uniform_split(tokens, count);
    let longest = base + u32::from(remainder > 0);
    assert!(
        base > 0 && longest <= context,
        "each pair needs 0 < query <= context"
    );
    (base, remainder, longest)
}

/// [`mean_keys`] of `uniform_pairs(tokens, count, context)`, without building
/// it. The canonical searches evaluate this at every probe of a binary search
/// over the context, for every request count, so the per-request sum dominated
/// a DeepSeek V4 build. The batch has two query lengths at one context, so its
/// key total is two prefix differences: the same integer total, hence the same
/// mean to the bit.
fn uniform_mean_keys(tokens: u32, count: u32, context: u32, ratio: u32) -> f64 {
    let (base, remainder, longest) = checked_uniform_split(tokens, count, context);
    let full = keys_prefix(context, ratio);
    let keys = u64::from(remainder) * (full - keys_prefix(context - longest, ratio))
        + u64::from(count - remainder) * (full - keys_prefix(context - base, ratio));
    keys as f64 / f64::from(tokens)
}

/// [`wave_keys`]' mean of `uniform_pairs(tokens, count, context)`, without
/// building or sorting the rows. Every request ends at `context` and a row's
/// keys never fall as its position rises, so longest first is position
/// `context` `count` times, then `context - 1` and so on, the extra rows of
/// the `remainder` longer requests last. Wave maxima are read off that order
/// and summed in it, so the mean matches to the bit.
fn uniform_wave_keys(tokens: u32, count: u32, context: u32, ratio: u32, per_wave: usize) -> f64 {
    let (base, _, _) = checked_uniform_split(tokens, count, context);
    let shared_rows = (base * count) as usize;
    let maxima = (0..tokens as usize).step_by(per_wave).map(|row| {
        let back = if row < shared_rows {
            row as u32 / count
        } else {
            base
        };
        f64::from(token_keys(context - back, ratio))
    });
    let waves = (tokens as usize).div_ceil(per_wave);
    maxima.sum::<f64>() / waves as f64
}

fn key_tolerance(keys: f64) -> f64 {
    (0.02 * keys).max(1.0)
}

fn area_tolerance(area: f64) -> f64 {
    (0.02 * area).max(1.0)
}

/// Smallest `x` in `lo..=hi` with `f(x) >= target`, assuming `f` rises with `x`.
fn lower_bound(lo: u32, hi: u32, target: f64, f: impl Fn(u32) -> f64) -> u32 {
    let (mut lo, mut hi) = (lo, hi);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if f(mid) >= target {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// The smallest shared context whose uniform batch comes closest to
/// `target` keys, or `None` if no context reaches it. `uniform_keys(tokens,
/// count, context)` is the keys of `uniform_pairs(tokens, count, context)`.
fn uniform_for_keys(
    tokens: u32,
    count: u32,
    target: f64,
    max_context: u32,
    uniform_keys: &impl Fn(u32, u32, u32) -> f64,
) -> Option<Vec<(u32, u32)>> {
    let lowest = tokens.div_ceil(count);
    if lowest > max_context {
        return None;
    }
    let keys = |context| uniform_keys(tokens, count, context);
    let tolerance = key_tolerance(target);
    if keys(max_context) < target - tolerance || keys(lowest) > target + tolerance {
        return None;
    }
    let found = lower_bound(lowest, max_context, target, keys);
    let context = if found > lowest && target - keys(found - 1) < keys(found) - target {
        found - 1
    } else {
        found
    };
    Some(uniform_pairs(tokens, count, context))
}

/// Whether `pairs` project onto `point` within tolerance.
fn on_point(
    pairs: &[(u32, u32)],
    point: &[f64],
    config: &CompressedSparseMlaRopeCastKernelConfig,
) -> bool {
    let projected = project(pairs, config);
    let floor = if is_decode(config) || config.compress_ratio == 0 {
        0.0
    } else {
        area_floor(point[0] as u32, config.compress_ratio)
    };
    pairs.len() <= MAX_REQUESTS as usize
        && pairs.iter().all(|&(_, c)| c <= MAX_CONTEXT)
        && projected[0] == point[0]
        && (projected[1] - point[1]).abs() <= key_tolerance(point[1])
        && (projected[2] - point[2]).abs() <= area_tolerance(floor + point[2])
}

/// A decode batch of `rows` flattened rows over at most 256 requests.
fn canonical_decode(
    point: &[f64],
    config: &CompressedSparseMlaRopeCastKernelConfig,
) -> Option<Vec<(u32, u32)>> {
    let rows = point[0] as u32;
    if rows > MAX_DECODE_ROWS || (is_swapab(config) && rows > SWAPAB_MAX_TOKENS) {
        return None;
    }
    let per_wave = rows_per_wave(config) as usize;
    let uniform_keys = |tokens, count, context| {
        uniform_wave_keys(tokens, count, context, config.compress_ratio, per_wave)
    };
    let pairs = uniform_for_keys(
        rows,
        rows.min(MAX_REQUESTS),
        point[1],
        MAX_CONTEXT,
        &uniform_keys,
    )?;
    on_point(&pairs, point, config).then_some(pairs)
}

/// A prefill batch with the fewest requests reaching the keys. It adds gather
/// area by lengthening every context while the keys stay put (saturated
/// tokens); otherwise it appends one single-token request with a long context.
fn canonical_prefill(
    point: &[f64],
    config: &CompressedSparseMlaRopeCastKernelConfig,
) -> Option<Vec<(u32, u32)>> {
    let (tokens, keys) = (point[0] as u32, point[1]);
    let ratio = config.compress_ratio;
    let uniform_keys = |tokens, count, context| uniform_mean_keys(tokens, count, context, ratio);
    for count in 1..=tokens.min(MAX_REQUESTS - 1) {
        let Some(base) = uniform_for_keys(tokens, count, keys, MAX_CONTEXT, &uniform_keys) else {
            continue;
        };
        if ratio == 0 {
            return on_point(&base, point, config).then_some(base);
        }
        let target = area_floor(tokens, ratio) + point[2];
        let area = gather_area(&base, ratio) as f64;
        if area > target + area_tolerance(target) {
            return None;
        }
        if on_point(&base, point, config) {
            return Some(base);
        }
        let stretched = lower_bound(base[0].1, MAX_CONTEXT, target, |context| {
            gather_area(&uniform_pairs(tokens, count, context), ratio) as f64
        });
        let stretched = uniform_pairs(tokens, count, stretched);
        if on_point(&stretched, point, config) {
            return Some(stretched);
        }
        return (count..tokens.min(MAX_REQUESTS - 1))
            .find_map(|count| with_filler(point, count, target, config));
    }
    None
}

/// `count` uniform requests over `tokens - 1` queries plus a trailing
/// `(1, context)` request sized to reach `target` gather area.
fn with_filler(
    point: &[f64],
    count: u32,
    target: f64,
    config: &CompressedSparseMlaRopeCastKernelConfig,
) -> Option<Vec<(u32, u32)>> {
    let (tokens, keys) = (point[0] as u32, point[1]);
    if tokens < 2 || count > tokens - 1 {
        return None;
    }
    let ratio = config.compress_ratio;
    let uniform_keys = |tokens, count, context| uniform_mean_keys(tokens, count, context, ratio);
    let mut filler = 1_u32;
    let mut pairs = Vec::new();
    for _ in 0..4 {
        let filler_keys = f64::from(token_keys(filler, ratio));
        let base_keys = (f64::from(tokens) * keys - filler_keys) / f64::from(tokens - 1);
        let base = uniform_for_keys(tokens - 1, count, base_keys, MAX_CONTEXT, &uniform_keys)?;
        let with = |context| {
            let mut candidate = base.clone();
            candidate.push((1, context));
            candidate
        };
        filler = lower_bound(1, MAX_CONTEXT, target, |context| {
            gather_area(&with(context), ratio) as f64
        });
        pairs = with(filler);
    }
    on_point(&pairs, point, config).then_some(pairs)
}

fn token_axis(config: &CompressedSparseMlaRopeCastKernelConfig) -> Vec<f64> {
    if is_swapab(config) {
        // Every step vLLM sends here: 1..=16 rows, one wave.
        return Axis::values([1, 2, 4, 6, 8, 10, 12, 14, 16]);
    }
    if is_decode(config) {
        // The decode time steps one wave at a time: bracket every wave edge.
        let wave = rows_per_wave(config);
        let edges = (1..)
            .map(|waves| waves * wave)
            .take_while(|&rows| rows < MAX_DECODE_ROWS)
            .flat_map(|rows| [rows, rows + 1]);
        // End on the profiler's row limit, not on a wave edge, so any
        // extrapolation keeps the average per-wave slope.
        let mut axis = [1, 2, 4, 8, 16, 32, 64, 128]
            .into_iter()
            .filter(|&rows| rows < wave)
            .chain(edges)
            .chain([MAX_DECODE_ROWS])
            .collect::<Vec<_>>();
        axis.sort_unstable();
        return Axis::values(axis);
    }
    // One chunk holds at most one step's tokens; 16384 is the largest step
    // budget the V4.1 deployments run. Larger chunks hold the edge bandwidth.
    Axis::values(PREFILL_TOKENS)
}

/// The prefill query-token axis.
const PREFILL_TOKENS: [u32; 11] = [1, 4, 16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384];

fn key_axis(config: &CompressedSparseMlaRopeCastKernelConfig) -> Vec<f64> {
    match (is_decode(config), config.compress_ratio) {
        (_, 0) => Axis::values([1, 16, 32, 64, 96, 128]),
        (true, 1) => Axis::values([2, 64, 128, 256, 384, 512, 640]),
        (true, _) => Axis::values([1, 48, 96, 192, 320, 480, 640]),
        (false, 1) => Axis::values([2, 32, 96, 192, 320, 480, 640]),
        (false, _) => Axis::values([1, 32, 96, 192, 320, 480, 640]),
    }
}

fn excess_axis(config: &CompressedSparseMlaRopeCastKernelConfig) -> Vec<f64> {
    if is_decode(config) || config.compress_ratio == 0 {
        return vec![0.0];
    }
    // The planes step by 4x up to the 256K-row middle, then one and two
    // longest-context requests at the largest token count; the second covers
    // chunks that pad to two long requests. Larger areas hold the edge
    // bandwidth (the gather is a streaming copy).
    let ratio = config.compress_ratio;
    let longest = f64::from(MAX_CONTEXT / ratio);
    let floor = area_floor(*PREFILL_TOKENS.last().unwrap(), ratio);
    let (one, two) = (longest - floor, 2.0 * longest - floor);
    let mut axis = Axis::values([0, 1024, 4096, 16384, 65536, 262144])
        .into_iter()
        .filter(|&rows| rows < one)
        .collect::<Vec<_>>();
    axis.extend([one, two]);
    axis
}

fn canonical_pairs(
    config: &CompressedSparseMlaRopeCastKernelConfig,
    point: &[f64],
) -> Option<Vec<(u32, u32)>> {
    if is_decode(config) {
        canonical_decode(point, config)
    } else {
        canonical_prefill(point, config)
    }
}

pub struct CompressedSparseMlaRopeCastSpec;

impl KernelSpec for CompressedSparseMlaRopeCastSpec {
    type Config = CompressedSparseMlaRopeCastKernelConfig;
    type Input = CompressedSparseMlaRopeCastKernelInput;

    const KIND: KernelKind = "compressed_sparse_mla_rope_cast";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        validate_config(config);
        SweepGrid::new(vec![
            token_axis(config),
            key_axis(config),
            excess_axis(config),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        Coords::new(project(&input.query_context_pairs, config))
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand(|point| canonical_pairs(config, point).is_none())
    }

    /// Past the grid (more query rows than the step budget, or more gather
    /// area than two longest requests) the launch holds its bandwidth, as
    /// compressed_sparse_mla_decode and _prefill do: on B200 the last two
    /// measured points along the query axis sit within 2% of each other's
    /// bandwidth in decode and within 13% in prefill.
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
        grid.expand(|point| {
            // Infeasible cells get a placeholder the engine drops before profiling.
            let pairs = canonical_pairs(config, point).unwrap_or_else(|| vec![(1, 1)]);
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

register_kernel!(
    CompressedSparseMlaRopeCastKernel,
    CompressedSparseMlaRopeCastSpec
);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(mode: &str, ratio: u32) -> CompressedSparseMlaRopeCastKernelConfig {
        config_with_heads(mode, ratio, 64)
    }

    fn config_with_heads(
        mode: &str,
        ratio: u32,
        heads: u32,
    ) -> CompressedSparseMlaRopeCastKernelConfig {
        serde_json::from_value(serde_json::json!({
            "backends": ["flashmla_mega"],
            "gpu_name": "NVIDIA B200",
            "mode": mode,
            "compress_ratio": ratio,
            "window_size": 128,
            "index_topk": 512,
            "num_heads": heads,
            "head_dim": 512,
            "rope_dim": 64,
            "q_dtype": "bf16",
            "swa_cache_format": "mxfp8",
            "compressed_cache_format": if ratio == 0 { "none" } else { "nvfp4" },
            "output_dtype": "fp8_e4m3",
        }))
        .unwrap()
    }

    #[test]
    fn logical_bytes_match_measured_rows() {
        // Profiled B200 rows' logged bandwidth x time.
        let bytes = |mode: &str, ratio: u32, pairs: Vec<(u32, u32)>| {
            logical_bytes(
                &config(mode, ratio),
                &CompressedSparseMlaRopeCastKernelInput {
                    query_context_pairs: pairs,
                },
            )
        };
        assert_eq!(bytes("decode", 0, vec![(1, 128)]), 167_424.0);
        assert_eq!(bytes("decode", 1, vec![(1, 32)]), 128_000.0);
        assert_eq!(bytes("decode", 2, vec![(1, 704), (1, 704)]), 541_696.0);
        assert_eq!(bytes("prefill", 0, vec![(64, 64)]), 8_594_560.0);
        assert_eq!(bytes("prefill", 1, vec![(64, 66112)]), 135_499_248.0);
        assert_eq!(
            bytes("prefill", 2, vec![(1023, 1237), (1, 5120)]),
            612_957_244.0
        );
    }

    fn coords(mode: &str, ratio: u32, pairs: &[(u32, u32)]) -> Coords {
        let input = CompressedSparseMlaRopeCastKernelInput {
            query_context_pairs: pairs.to_vec(),
        };
        CompressedSparseMlaRopeCastSpec::cache_coords(&config(mode, ratio), &input)
    }

    /// Catches a closed-form key count that drifts from the runner's per-token
    /// `min(pos + 1, 128) + min((pos + 1) // ratio, 512)` lengths.
    #[test]
    fn closed_form_keys_match_per_token_lengths() {
        for ratio in 0..=2 {
            for n in [1, 2, 3, 127, 128, 129, 511, 512, 1023, 1024, 1025, 5000] {
                let brute = (1..=n)
                    .map(|x| u64::from(token_keys(x, ratio)))
                    .sum::<u64>();
                assert_eq!(keys_prefix(n, ratio), brute);
            }
        }
    }

    #[test]
    fn uniform_keys_match_the_built_batch() {
        for ratio in 0..=2 {
            for (tokens, count) in [(1_u32, 1_u32), (7, 3), (256, 255), (4096, 17), (8192, 255)] {
                let lowest = tokens.div_ceil(count);
                for context in [lowest, lowest + 1, 129, 1025, 70_000, 1_048_576] {
                    if context < lowest {
                        continue;
                    }
                    let pairs = uniform_pairs(tokens, count, context);
                    let closed = uniform_mean_keys(tokens, count, context, ratio);
                    assert_eq!(closed.to_bits(), mean_keys(&pairs, ratio).1.to_bits());
                    for per_wave in [1, 74, 148, 1000] {
                        let closed = uniform_wave_keys(tokens, count, context, ratio, per_wave);
                        let built = wave_keys(&pairs, ratio, per_wave).1;
                        assert_eq!(closed.to_bits(), built.to_bits());
                    }
                }
            }
        }
    }

    /// Catches a decode projection that averages keys: one long row in a
    /// single wave costs as much as an all-long wave.
    #[test]
    fn decode_keys_follow_the_longest_row_of_each_wave() {
        let mixed = [(1, 100), (1, 5000), (1, 100)];
        assert_eq!(coords("decode", 1, &mixed)[1], 640.0);
        let two_waves = [vec![(1, 5000)], vec![(1, 64); 148]].concat();
        assert_eq!(coords("decode", 1, &two_waves)[1], (640.0 + 128.0) / 2.0);
        assert_eq!(coords("decode", 1, &two_waves)[2], 0.0);
    }

    /// Catches a planner that drifts from the fork's `get_prefill_chunk_plan`:
    /// the area budget `chunk_size * (ceil(max_model_len / ratio) + window +
    /// max_num_batched_tokens)`, greedy in request order, more than
    /// `prefill_chunk_size` short requests per chunk, and a budget that moves
    /// with `max_num_batched_tokens` and `max_model_len`.
    #[test]
    fn prefill_chunk_plan_replays_the_fork_budget() {
        let plan = |pairs: &[(u32, u32)], ratio, max_model_len, mbt| {
            prefill_chunk_plan(pairs, ratio, WINDOW, max_model_len, mbt, 4)
        };
        // Budget 4 * (131072 + 128 + 2048) = 532992 at ratio 1; the long
        // request pads every request of its chunk to 100000 + 128.
        let five = [vec![(16, 16); 4], vec![(1, 100000)]].concat();
        assert_eq!(plan(&five, 1, 131072, 2048), vec![0..5]);
        let six = [vec![(16, 16); 5], vec![(1, 100000)]].concat();
        assert_eq!(plan(&six, 1, 131072, 2048), vec![0..5, 5..6]);
        // A larger step budget, or a longer max_model_len, fits it.
        assert_eq!(plan(&six, 1, 131072, 32768), vec![0..6]);
        assert_eq!(plan(&six, 1, 1_048_576, 2048), vec![0..6]);
        // Ratio 0 pads only the gathered SWA width, 2048 + 127 per request:
        // four fit 4 * (128 + 2048), thirty fit 4 * (128 + 16384).
        let short = vec![(2048, 4096); 40];
        let fours = (0..10).map(|c| 4 * c..4 * c + 4).collect::<Vec<_>>();
        assert_eq!(plan(&short, 0, 131072, 2048), fours);
        assert_eq!(plan(&short, 0, 131072, 16384), vec![0..30, 30..40]);
        assert!(plan(&[], 2, 131072, 2048).is_empty());
    }

    /// Catches a gather coordinate that sums rows instead of padding the chunk
    /// to its longest request.
    #[test]
    fn prefill_gather_area_pads_the_chunk() {
        assert_eq!(gather_area(&[(1, 50000)], 1), 50000);
        assert_eq!(gather_area(&[(1, 25000), (1, 25000)], 1), 50000);
        assert_eq!(gather_area(&[(16, 16), (1, 100000)], 1), 200000);
        let pairs = [(1912, 1912), (91, 4096)];
        let projected = coords("prefill", 2, &pairs);
        let keys = (keys_prefix(1912, 2) + keys_prefix(4096, 2) - keys_prefix(4005, 2)) as f64;
        assert_eq!(projected[1], keys / 2003.0);
        assert_eq!(projected[2], 2.0 * 2048.0 - (1002.0 + 512.0));
    }

    /// Catches canonical profiling batches that miss the grid point they fill,
    /// and a grid past the 500 feasible-coordinate ceiling.
    #[test]
    fn canonical_batches_project_onto_their_grid_points() {
        let shapes = [("decode", 64), ("decode", 128), ("prefill", 64)];
        let cases = shapes
            .into_iter()
            .flat_map(|shape| (0..=2).map(move |ratio| (shape, ratio)))
            .collect::<Vec<_>>();
        // Each grid's canonical batches are independent; search them in parallel.
        std::thread::scope(|scope| {
            for &((mode, heads), ratio) in &cases {
                scope.spawn(move || {
                    let config = config_with_heads(mode, ratio, heads);
                    let grid = CompressedSparseMlaRopeCastSpec::sweep_grid(&config);
                    let mask = CompressedSparseMlaRopeCastSpec::infeasible_mask(&config, &grid);
                    let feasible = mask.iter().filter(|&&drop| !drop).count();
                    assert!(
                        feasible > 0 && feasible <= 500,
                        "{mode} r{ratio}: {feasible}"
                    );
                    eprintln!(
                        "{mode} h{heads} r{ratio}: {} cells, {feasible} feasible",
                        mask.len()
                    );
                });
            }
        });
    }

    /// The gather-area planes: a 4x ladder to 256K rows, then one and two
    /// longest-context requests, the same for every deployment.
    #[test]
    fn excess_planes_ladder_up_to_the_longest_requests() {
        let planes = |ratio| excess_axis(&config("prefill", ratio));
        assert_eq!(
            planes(1),
            [0.0, 1024.0, 4096.0, 16384.0, 65536.0, 262144.0, 1031680.0, 2080256.0]
        );
        assert_eq!(
            planes(2),
            [0.0, 1024.0, 4096.0, 16384.0, 65536.0, 262144.0, 515584.0, 1039872.0]
        );
        assert_eq!(excess_axis(&config("decode", 2)), [0.0]);
    }

    /// Catches a swap-AB grid that leaves the steps vLLM routes there (more
    /// than 16 rows) or whose canonical batches miss their grid points.
    #[test]
    fn swapab_grid_spans_one_to_sixteen_rows_in_one_wave() {
        assert_eq!(swapab_tiles(128, 512), 6);
        assert!(swapab_layer(1, 16, 128, 512) && swapab_layer(2, 32, 128, 512));
        assert!(!swapab_layer(0, 16, 128, 512) && !swapab_layer(2, 64, 128, 512));
        assert!(
            !swapab_layer(2, 16, 128, 1024),
            "10 tiles exceed one cluster"
        );
        for ratio in 1..=2 {
            for heads in [16, 32] {
                let config = config_with_heads("decode", ratio, heads);
                assert_eq!(rows_per_wave(&config), 24);
                let grid = CompressedSparseMlaRopeCastSpec::sweep_grid(&config);
                assert_eq!(grid.axes()[0].last(), Some(&16.0));
                let mask = CompressedSparseMlaRopeCastSpec::infeasible_mask(&config, &grid);
                let feasible = mask.iter().filter(|&&drop| !drop).count();
                assert!(
                    feasible > 40 && feasible <= 500,
                    "r{ratio} h{heads}: {feasible}"
                );
            }
        }
        // One wave: a single long row sets the keys of the whole step.
        let config = config_with_heads("decode", 1, 16);
        let input = CompressedSparseMlaRopeCastKernelInput {
            query_context_pairs: [vec![(1, 5000)], vec![(1, 64); 15]].concat(),
        };
        assert_eq!(
            CompressedSparseMlaRopeCastSpec::cache_coords(&config, &input)[1],
            640.0
        );
    }

    /// Catches swap-AB heads configured for a step the mega kernel serves.
    #[test]
    #[should_panic(expected = "swap-AB")]
    fn live_heads_are_decode_on_compressed_layers_only() {
        CompressedSparseMlaRopeCastSpec::sweep_grid(&config_with_heads("prefill", 2, 16));
    }

    /// Catches a ratio-0 layer configured with a compressed cache.
    #[test]
    #[should_panic(expected = "compressed_cache_format")]
    fn swa_only_layer_rejects_compressed_cache() {
        let mut config = config("decode", 0);
        config.compressed_cache_format = "nvfp4".into();
        CompressedSparseMlaRopeCastSpec::sweep_grid(&config);
    }
}
