//! Relative-position attention: the fork's tml FA4 call over one
//! varlen batch (`InklingFA4RelAttentionKernel.kernel`).
//!
//! Per layer the model makes ONE call for the whole scheduler step, decode and
//! prefill requests together. The call issues ShearingBias, FlashAttention
//! forward with bias tiles, and, depending on the batch, the cu-blocks metadata
//! pair (packed rows, bucketed max query x q heads per KV head, > 256) and the
//! split scheduler + combine (global layers, `inkling_fa4_num_splits` > 1). The
//! profiler counts every launch. `is_local` selects the sliding window of
//! `rel_extent` tokens; `max_model_len` drives the global split count, so both
//! are Config identity.
//!
//! The runtime input is the exact `query_context_pairs` batch. A mixed batch is
//! one lookup, never a decode lookup plus a prefill lookup: the launches are
//! shared and the decode CTAs run on the prefill's tile shape. The batch
//! projects to three cache coordinates:
//!
//! - `num_decode_requests`: requests with one query token. In decode-only
//!   steps this is the batch size and sets the global split count.
//! - `prefill_tokens`: query tokens of the other requests. It sets the bucketed
//!   tile shape, the M-tile count and the ShearingBias rows. Up to 128 tokens
//!   it is rounded up to its power-of-two bucket (at least 32): there the
//!   bucket, not the count, decides the q_stage, the cu-blocks launches, the
//!   split rule and the M-tile count (one tile per `128 * q_stage` packed
//!   rows), so a 40-token chunk runs like a 64-token one, not like the mean of
//!   32 and 64.
//! - `equivalent_context`: the context `C` at which the canonical batch with
//!   the same two counts has the same modelled FA4 makespan as the real batch.
//!   The canonical batch is `num_decode_requests` decodes at context `C` and,
//!   when `prefill_tokens > 0`, one chunk of `prefill_tokens` with prefix
//!   `min(C - 1, max_model_len - prefill_tokens)`.
//!
//! The makespan model counts KV tiles (128 keys) per CTA: one CTA per
//! (request, KV head, M tile, split), the M tile holding `128 * q_stage` packed
//! query rows, a local tile reading only its window. A CTA costs its tiles plus
//! a fixed overhead, and the launch lasts `max(longest CTA, total / SMs)` plus
//! a quarter of `total / SMs` for the bandwidth the CTAs share. It
//! is monotone in `C`, so the projection is exact on every canonical point and
//! a heterogeneous batch maps to the uniform context that keeps the same
//! critical path: a single-wave batch follows its longest request, a
//! many-wave batch its total work.
//!
//! Between two grid decode counts the decode coordinate is placed by the same
//! CTA model counted in whole waves of 148 CTAs (`decode_coordinate`), so a
//! wave or split-count step lands where it happens instead of being averaged
//! across the cell.
//!
//! A FULL CUDA-graph decode step launches over the captured batch size, not
//! the real one: the model runner pads `seq_lens` with zero-length requests
//! up to the capture size (`gpu_model_runner._build_attention_metadata`) and
//! `InklingAttention._attention` passes `seq_lens.shape[0]` to
//! `inkling_fa4_num_splits`. `padded_requests` carries that size. The split
//! count follows it (9 real decodes padded to 16 run 4 splits, not 7), and the
//! padding slots count as decodes in the cache coordinate, so the step reads
//! the canonical rows of the padded size, which have the same split count.
//! The zero-length slots add no CTA work. Local layers never split, so they
//! keep the real batch. In the standard diverse_100 capture the combine grid
//! steps at 1, 2, 4, 8, 16 requests and nowhere between.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// SMs on the only supported GPU (NVIDIA B200).
const B200_SMS: f64 = 148.0;
/// FA4 SM100 tile: 128 packed query rows per stage, 128 keys per KV tile.
const TILE_M: u32 = 128;
const TILE_N: f64 = 128.0;
/// Per-CTA prologue/epilogue cost in KV-tile units.
const CTA_OVERHEAD_TILES: f64 = 1.0;
/// Weight of the per-SM share of all work on top of the critical path: CTAs
/// share HBM and L2, so work off the critical path still slows it. It also
/// keeps the model strictly increasing where the critical path is flat (a
/// local prefill chunk beside short decodes).
const SHARED_WORK_WEIGHT: f64 = 0.25;
/// Decode-request counts on the grid, placed for GQA 4 with 2 KV heads. A
/// local decode runs one CTA per (request, KV head), so the first wave fills
/// 148 SMs at `148 / num_kv_heads` requests; 80 sits just past it at 2 KV heads
/// (74 requests; B200 local decode goes 20 -> 31 us from 64 to 128 requests).
/// 2 and 4 are left out: 1 -> 8 interpolates them within 7% (B200 global
/// decode at max_model_len 131072, 131072 keys: 2 requests 78 us measured,
/// 83 us interpolated).
const DECODE_AXIS: [u32; 9] = [0, 1, 8, 16, 32, 64, 80, 128, 256];
/// Prefill-token counts, placed for GQA 4. q_stage goes to 2 when the packed
/// rows (bucketed tokens x q heads per KV head) exceed `TILE_M`, and the
/// cu-blocks pair launches when they exceed 256; at GQA 4 those switches fall
/// between 32 and 64 and between 64 and 128 tokens. 2048 is the step budget.
/// No point below 32: at GQA 4 up to 32 tokens (128 packed rows) a chunk is
/// one M tile at q_stage 1 whatever its size (B200 global, 4175 keys: 28.6 us
/// at 16 tokens, 28.2 us at 32).
const PREFILL_AXIS: [u32; 8] = [0, 32, 64, 128, 256, 512, 1024, 2048];
/// Contexts sit half a KV tile past a tile boundary (`128 k + 64`): a real
/// context is rarely tile-aligned, and an aligned one reads one KV tile fewer
/// (B200 local decode: 18.7 us at 1024 keys, 20.8 us at 6000).
const CONTEXT_TILE_OFFSET: u32 = 64;
/// Global contexts double from 512 keys up to this many, then grow by 4x.
/// Up to here the split scheduler still raises the per-request split count
/// with the context (one KV tile per split while a request has fewer tiles
/// than splits); past it every CTA reads a fixed share of its request, so
/// time is linear in the context (B200 at max_model_len 131072, 8256 to
/// 131072 keys: 4x steps interpolate 1 to 256 decodes and 2048-token chunks
/// within 10%).
const CONTEXT_DOUBLING_END: u32 = 2048;
const BISECTION_STEPS: usize = 48;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RelBiasAttentionKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    /// Sliding-window layer: the window is `rel_extent` tokens including self.
    pub is_local: bool,
    pub rel_extent: u32,
    pub page_size: u32,
    /// `lbnhc`: K and V of one (token, head) side by side in a page.
    pub kv_layout: String,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
    pub max_model_len: u32,
}

/// One scheduler step's requests as `(query_len, context_len)`, context
/// including the query, in the order the batch holds them.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RelBiasAttentionKernelInput {
    pub query_context_pairs: Vec<(u32, u32)>,
    /// Requests the call is launched over when a FULL CUDA graph pads the
    /// batch with zero-length requests (the capture size); 0, or anything up
    /// to the real count, means unpadded.
    #[serde(default)]
    pub padded_requests: u32,
}

impl SweepCoords for RelBiasAttentionKernelInput {
    fn coords(&self) -> Coords {
        // The equivalent context needs the Config (window, heads, splits);
        // `cache_coords` owns the real projection. This config-free view
        // reports the longest context.
        let (decodes, prefill) = counts(&self.query_context_pairs);
        let longest = self
            .query_context_pairs
            .iter()
            .map(|&(_, context)| context)
            .max()
            .unwrap_or(0);
        Coords::new([f64::from(decodes), f64::from(prefill), f64::from(longest)])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &[
            "num_decode_requests",
            "prefill_tokens",
            "equivalent_context",
        ]
    }
}

/// `(decode requests, prefill query tokens)`.
fn counts(pairs: &[(u32, u32)]) -> (u32, u32) {
    let decodes = pairs.iter().filter(|&&(query, _)| query == 1).count() as u32;
    let prefill = pairs
        .iter()
        .filter(|&&(query, _)| query > 1)
        .map(|&(query, _)| query)
        .sum();
    (decodes, prefill)
}

/// `bucket_max_seqlen_q`: the max query length rounded up to a power of two.
fn bucket(max_query: u32) -> u32 {
    max_query.max(1).next_power_of_two()
}

fn q_heads_per_kv_head(config: &RelBiasAttentionKernelConfig) -> u32 {
    config.num_qo_heads.get() / config.num_kv_heads.get()
}

/// `inkling_fa4_num_splits` (fa4_rel_attention.py) on SM10x: the static
/// split cap, set by the batch size, the bucketed query and `max_model_len`.
/// On device `FlashPrepareScheduler` lowers each request's count to
/// `ceil(tiles / tiles_per_sm)`, `tiles_per_sm` being the batch's KV tiles
/// times 1.157 x KV heads over 148 SMs, so a request never gets more splits
/// than KV tiles and a lone decode stays near 64. The CTA model keeps the cap:
/// the canonical rows carry the real count, and the cap only steers the
/// context projection, where a lower count on short requests moves little.
fn num_splits(config: &RelBiasAttentionKernelConfig, batch: u32, bucket: u32) -> u32 {
    if config.is_local {
        return 1;
    }
    let kv_heads = config.num_kv_heads.get();
    let max_kv_len = config.max_model_len;
    let q_tiles = (bucket * q_heads_per_kv_head(config)).div_ceil(256);
    let base_ctas = batch * kv_heads * q_tiles;
    let single = q_tiles == 1 && batch == 1;
    let target = if single {
        256
    } else if q_tiles == 1 {
        128
    } else {
        64
    };
    let max_splits = if single {
        if kv_heads == 8 {
            16
        } else if kv_heads == 4 || max_kv_len <= 8192 {
            32
        } else if max_kv_len <= 65536 {
            64
        } else {
            128
        }
    } else {
        128
    };
    (target / base_ctas)
        .min(max_splits)
        .min(max_kv_len.div_ceil(128))
        .max(1)
}

/// Every CTA's modelled work in KV-tile units (see the module doc).
/// `launched` is the batch the split rule sees: `pairs.len()`, or the
/// graph-padded size (the padding slots add no CTA work).
fn cta_works(
    config: &RelBiasAttentionKernelConfig,
    pairs: &[(u32, u32)],
    launched: u32,
) -> Vec<f64> {
    assert!(!pairs.is_empty(), "query_context_pairs must be non-empty");
    let max_query = pairs.iter().map(|&(query, _)| query).max().unwrap();
    let bucket = bucket(max_query);
    let g = q_heads_per_kv_head(config);
    let q_stage = if bucket * g > TILE_M { 2 } else { 1 };
    let tile_tokens = (TILE_M * q_stage / g).max(1);
    let splits = num_splits(config, launched.max(pairs.len() as u32), bucket);
    let copies = (splits * config.num_kv_heads.get()) as usize;
    let window = f64::from(config.rel_extent);

    let mut works = Vec::new();
    for &(query, context) in pairs {
        assert!(
            query > 0 && query <= context,
            "each pair needs 0 < query <= context"
        );
        let prefix = f64::from(context - query);
        for tile in 0..query.div_ceil(tile_tokens) {
            let first = prefix + f64::from(tile * tile_tokens);
            let end = prefix + f64::from(((tile + 1) * tile_tokens).min(query));
            let start = if config.is_local {
                (first + 1.0 - window).max(0.0)
            } else {
                0.0
            };
            let cta = (end - start) / TILE_N / f64::from(splits) + CTA_OVERHEAD_TILES;
            works.extend(std::iter::repeat_n(cta, copies));
        }
    }
    works
}

/// Modelled FA4 makespan: `max(longest CTA, total / SMs)` plus the shared
/// share. Continuous and monotone in every context; the context projection
/// inverts it.
fn makespan(config: &RelBiasAttentionKernelConfig, pairs: &[(u32, u32)], launched: u32) -> f64 {
    let works = cta_works(config, pairs, launched);
    let longest = works.iter().copied().fold(0.0, f64::max);
    let per_sm = works.iter().sum::<f64>() / B200_SMS;
    longest.max(per_sm) + SHARED_WORK_WEIGHT * per_sm
}

/// Modelled FA4 makespan with whole CTA waves: CTAs longest first, each wave of
/// 148 lasting as long as its first. It carries the steps a batch-size change
/// makes (another wave, a different split count), which the decode-count
/// coordinate follows between grid points.
fn wave_makespan(config: &RelBiasAttentionKernelConfig, pairs: &[(u32, u32)]) -> f64 {
    let mut works = cta_works(config, pairs, pairs.len() as u32);
    works.sort_unstable_by(|lhs, rhs| rhs.total_cmp(lhs));
    let waves = works.iter().step_by(B200_SMS as usize).sum::<f64>();
    waves + SHARED_WORK_WEIGHT * works.iter().sum::<f64>() / B200_SMS
}

/// The decode-count coordinate. On the grid it is the count; between two grid
/// counts it is placed where the wave model of the canonical batch puts it, so
/// a step (a third CTA wave at 150 requests, the global split count dropping
/// from 3 to 1 between 4 and 8 requests beside a 128-token chunk) lands where
/// it happens rather than being spread over the cell.
fn decode_coordinate(
    config: &RelBiasAttentionKernelConfig,
    decodes: u32,
    prefill: u32,
    context: f64,
) -> f64 {
    let Some(hi_index) = DECODE_AXIS.iter().position(|&grid| grid >= decodes) else {
        return f64::from(decodes);
    };
    let hi = DECODE_AXIS[hi_index];
    if hi == decodes || hi_index == 0 {
        return f64::from(decodes);
    }
    let lo = DECODE_AXIS[hi_index - 1];
    let context = context.round() as u32;
    let model = |count: u32| {
        let pairs = canonical_pairs(config, count, prefill, context)
            .expect("a grid neighbour of a decode count >= 1 is non-empty");
        wave_makespan(config, &pairs)
    };
    let (m_lo, m, m_hi) = (model(lo), model(decodes), model(hi));
    let linear = f64::from(decodes - lo) / f64::from(hi - lo);
    let t = if m_hi - m_lo > 1e-9 {
        ((m - m_lo) / (m_hi - m_lo)).clamp(0.0, 1.0)
    } else {
        linear
    };
    // Where the model is linear across the cell, keep the count itself exact.
    if (t - linear).abs() < 1e-6 {
        return f64::from(decodes);
    }
    f64::from(lo) + t * f64::from(hi - lo)
}

/// The canonical batch at a grid point; `None` for the empty batch. Its
/// prefill tokens are one chunk, however many requests they came from.
fn canonical_pairs(
    config: &RelBiasAttentionKernelConfig,
    decodes: u32,
    prefill: u32,
    context: u32,
) -> Option<Vec<(u32, u32)>> {
    if decodes == 0 && prefill == 0 {
        return None;
    }
    let context = context.clamp(1, config.max_model_len);
    let mut pairs = vec![(1, context); decodes as usize];
    if prefill > 0 {
        let prefix = (context - 1).min(config.max_model_len.saturating_sub(prefill));
        pairs.push((prefill, prefix + prefill));
    }
    Some(pairs)
}

/// The cache coordinates of a real batch launched over `padded_requests`
/// slots (see the module doc; 0 = unpadded).
fn project(
    config: &RelBiasAttentionKernelConfig,
    pairs: &[(u32, u32)],
    padded_requests: u32,
) -> [f64; 3] {
    // Known limit: several prefill requests in one step are projected as one
    // chunk of their summed query tokens. When the real batch has less work
    // than that chunk at C = 1 (eight fresh 256-token prompts against one
    // 2048-token chunk), the context clamps to C = 1 and the step is
    // overpriced.
    let (decodes, prefill) = counts(pairs);
    let prefill = prefill_coordinate(prefill);
    // Local layers never split, so padding leaves their launch unchanged.
    let launched = if config.is_local {
        pairs.len() as u32
    } else {
        padded_requests.max(pairs.len() as u32)
    };
    // Padding slots are decode-shaped: the canonical batch of `decodes +
    // padding` decodes is launched over the same `launched` requests.
    let decodes = decodes + (launched - pairs.len() as u32);
    let context = equivalent_context(config, pairs, launched, decodes, prefill);
    [
        decode_coordinate(config, decodes, prefill, context),
        f64::from(prefill),
        context,
    ]
}

/// The context at which the canonical batch has the real batch's makespan.
fn equivalent_context(
    config: &RelBiasAttentionKernelConfig,
    pairs: &[(u32, u32)],
    launched: u32,
    decodes: u32,
    prefill: u32,
) -> f64 {
    let target = makespan(config, pairs, launched);
    let canonical = |context: f64| {
        let pairs = canonical_pairs(config, decodes, prefill, context.round() as u32)
            .expect("a non-empty batch has a canonical batch");
        makespan(config, &pairs, pairs.len() as u32)
    };
    // Fractional contexts: the model is linear in the context between integer
    // points, so interpolate there to keep the projection continuous.
    let model = |context: f64| {
        let lo = context.floor();
        let t = context - lo;
        canonical(lo) * (1.0 - t) + canonical(lo + 1.0) * t
    };
    let (mut lo, mut hi) = (1.0_f64, f64::from(context_top(config)));
    if target <= model(lo) {
        return lo;
    }
    if target >= model(hi) {
        return hi;
    }
    for _ in 0..BISECTION_STEPS {
        let mid = 0.5 * (lo + hi);
        if model(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The top of the context axis: one window past the window, a local layer's
/// work is flat, so that is where a longer local batch lands.
fn context_top(config: &RelBiasAttentionKernelConfig) -> u32 {
    if config.is_local {
        (2 * config.rel_extent + CONTEXT_TILE_OFFSET).min(config.max_model_len)
    } else {
        config.max_model_len
    }
}

fn context_axis(config: &RelBiasAttentionKernelConfig) -> Vec<f64> {
    let mut values: Vec<u32> = if config.is_local {
        let window = config.rel_extent;
        vec![1, window * 5 / 8, window + CONTEXT_TILE_OFFSET]
    } else {
        // 1, then 512, 1024, 2048, 8192, 32768, ... keys, each half a tile
        // past the boundary.
        let boundaries = std::iter::successors(Some(512_u32), |&keys| {
            let step = if keys < CONTEXT_DOUBLING_END { 2 } else { 4 };
            keys.checked_mul(step)
        });
        std::iter::once(1)
            .chain(boundaries.map(|keys| keys + CONTEXT_TILE_OFFSET))
            .take_while(|&context| context < config.max_model_len)
            .collect()
    };
    values.retain(|&context| context < context_top(config));
    values.push(context_top(config));
    Axis::values(values)
}

/// The cache's prefill coordinate: the bucket up to 128 tokens (see the
/// module doc; at least the 32-token grid floor), the token count past it.
fn prefill_coordinate(prefill: u32) -> u32 {
    match prefill {
        0 => 0,
        1..=128 => bucket(prefill).max(PREFILL_AXIS[1]),
        _ => prefill,
    }
}

pub struct RelBiasAttentionSpec;

impl KernelSpec for RelBiasAttentionSpec {
    type Config = RelBiasAttentionKernelConfig;
    type Input = RelBiasAttentionKernelInput;

    const KIND: KernelKind = "rel_bias_attention";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert!(config.max_model_len > 0 && config.rel_extent > 0);
        assert!(
            config.num_qo_heads.get() % config.num_kv_heads.get() == 0,
            "num_qo_heads must be a multiple of num_kv_heads"
        );
        SweepGrid::new(vec![
            Axis::values(DECODE_AXIS),
            Axis::values(PREFILL_AXIS),
            context_axis(config),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        Coords::new(project(
            config,
            &input.query_context_pairs,
            input.padded_requests,
        ))
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_3d(|decodes, prefill, context| {
            canonical_pairs(config, decodes as u32, prefill as u32, context as u32).is_none()
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_3d(|decodes, prefill, context| {
            // The empty-batch cell is masked; it gets a placeholder the engine drops.
            let pairs = canonical_pairs(config, decodes as u32, prefill as u32, context as u32)
                .unwrap_or_else(|| vec![(1, 1)]);
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_qo_heads", config.num_qo_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("is_local", config.is_local)
                .with("rel_extent", config.rel_extent)
                .with("page_size", config.page_size)
                .with("kv_layout", config.kv_layout.clone())
                .with("q_dtype", config.q_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("max_model_len", config.max_model_len)
                .with("query_context_pairs", serde_json::json!(pairs))
        })
    }
}

register_kernel!(RelBiasAttentionKernel, RelBiasAttentionSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn config(is_local: bool) -> RelBiasAttentionKernelConfig {
        RelBiasAttentionKernelConfig {
            backends: vec!["tml_fa4"],
            gpu_name: "NVIDIA B200".to_string(),
            num_qo_heads: 8.into(),
            num_kv_heads: 2.into(),
            head_dim: 128.into(),
            is_local,
            rel_extent: if is_local { 512 } else { 1024 },
            page_size: 32,
            kv_layout: "lbnhc".to_string(),
            q_dtype: DType::Bf16,
            kv_dtype: DType::Bf16,
            max_model_len: 8192,
        }
    }

    #[test]
    fn split_rule_matches_the_fork() {
        // Defect caught: a global split count that drifts from
        // inkling_fa4_num_splits, which picks the prepare/combine launches.
        let global = config(false);
        // Decode (bucket 1): B = 1 gets 256 // 2 capped at 32; B >= 2 gets
        // 128 // (2 B).
        assert_eq!(num_splits(&global, 1, 1), 32);
        assert_eq!(num_splits(&global, 2, 1), 32);
        assert_eq!(num_splits(&global, 7, 1), 9);
        assert_eq!(num_splits(&global, 32, 1), 2);
        assert_eq!(num_splits(&global, 64, 1), 1);
        // A lone prefill chunk: 128 -> 16 splits, 1024 -> 2, 2048 -> 1.
        assert_eq!(num_splits(&global, 1, 128), 16);
        assert_eq!(num_splits(&global, 1, 1024), 2);
        assert_eq!(num_splits(&global, 1, 2048), 1);
        // The captured ~2018-token mixed step with 30 decodes does not split.
        assert_eq!(num_splits(&global, 31, 2048), 1);
        assert_eq!(num_splits(&config(true), 1, 1), 1);
        // At max_model_len 131072 a lone decode's cap rises to 128; two
        // requests keep 128 // (2 B) = 32.
        let long = RelBiasAttentionKernelConfig {
            max_model_len: 131072,
            ..config(false)
        };
        assert_eq!(num_splits(&long, 1, 1), 128);
        assert_eq!(num_splits(&long, 2, 1), 32);
        assert_eq!(num_splits(&long, 33, 2048), 1);
    }

    #[test]
    fn projection_is_exact_on_canonical_points() {
        // Defect caught: a projection that moves grid points, so the cache
        // never returns its own measurements.
        for is_local in [false, true] {
            let cfg = config(is_local);
            for (decodes, prefill, context) in [
                (32, 0, 2112),
                (1, 0, 4160),
                (30, 2048, 1088),
                (0, 512, 576),
                (8, 64, 1),
                (80, 32, 320),
            ] {
                if is_local && context > context_top(&cfg) {
                    continue;
                }
                let pairs = canonical_pairs(&cfg, decodes, prefill, context).unwrap();
                let [d, a, c] = project(&cfg, &pairs, 0);
                assert_eq!((d, a), (f64::from(decodes), f64::from(prefill)));
                let tolerance = if is_local && context >= cfg.rel_extent {
                    // Past the window the local model is flat in the context.
                    f64::from(cfg.rel_extent)
                } else {
                    1.0
                };
                assert!(
                    (c - f64::from(context)).abs() <= tolerance,
                    "local={is_local} ({decodes}, {prefill}, {context}) -> {c}"
                );
            }
        }
    }

    #[test]
    fn mixed_step_with_a_long_decode_projects_past_the_mean_context() {
        // Defect caught: averaging contexts in a single-wave mixed step, where
        // the longest request sets the launch time (captured: a 256-token
        // prefill beside decodes up to 4100 keys costs ~80 us of FA4, as much
        // as a 2048-token chunk).
        let cfg = config(false);
        let mut pairs = vec![(256, 256)];
        pairs.extend((0..30).map(|i| (1, 1000 + 100 * i)));
        let [d, a, c] = project(&cfg, &pairs, 0);
        assert_eq!((d, a), (30.0, 256.0));
        let mean = pairs.iter().map(|&(_, kv)| f64::from(kv)).sum::<f64>() / 31.0;
        assert!(c > mean + 500.0, "equivalent context {c} vs mean {mean}");
    }

    #[test]
    fn several_prefills_project_as_one_summed_chunk() {
        // Known limit, not a target: eight fresh 256-token prompts project as
        // one 2048-token chunk. The real batch has less work than that chunk
        // at C = 1, so the context clamps to 1 and the step is overpriced.
        for is_local in [false, true] {
            let cfg = config(is_local);
            let pairs = vec![(256, 256); 8];
            assert_eq!(project(&cfg, &pairs, 0), [0.0, 2048.0, 1.0]);
            let chunk = canonical_pairs(&cfg, 0, 2048, 1).unwrap();
            assert_eq!(chunk, [(2048, 2048)]);
            assert!(makespan(&cfg, &pairs, 8) < makespan(&cfg, &chunk, 1));
        }
    }

    #[test]
    fn decode_count_between_grid_points_follows_the_wave_step() {
        // Defect caught: 150 local decodes (300 CTAs, a third wave on 148 SMs)
        // read as 128 + 22/128 of the way to 256 (B200: 46.9 us measured,
        // 37.4 us from linear interpolation of the 128/256 rows).
        let cfg = config(true);
        let d = decode_coordinate(&cfg, 150, 0, 1088.0);
        assert!((180.0..=200.0).contains(&d), "{d}");
        // Inside one wave the placement stays linear.
        let d = decode_coordinate(&cfg, 100, 0, 1088.0);
        assert!((d - 100.0).abs() < 3.0, "{d}");
        assert_eq!(decode_coordinate(&cfg, 64, 0, 1088.0), 64.0);
        assert_eq!(decode_coordinate(&cfg, 300, 0, 1088.0), 300.0);
    }

    #[test]
    fn small_prefill_rounds_up_to_its_bucket() {
        // Defect caught: interpolating a 40-token chunk between the 32- and
        // 64-token grid points, though it runs the 64-token tile shape (B200
        // local, 20 decodes: measured 23.6 us, 32/64 mean 19.4 us).
        let cfg = config(true);
        let mut pairs = vec![(1, 2000); 20];
        pairs.push((40, 2040));
        assert_eq!(project(&cfg, &pairs, 0)[1], 64.0);
        assert_eq!(prefill_coordinate(3), 32);
        assert_eq!(prefill_coordinate(100), 128);
        assert_eq!(prefill_coordinate(300), 300);
    }

    #[test]
    fn canonical_chunk_stays_within_max_model_len() {
        // Defect caught: a grid point asking the profiler for a context past
        // max_model_len (rejected by the runner and by vLLM).
        let cfg = config(false);
        let pairs = canonical_pairs(&cfg, 4, 2048, 8192).unwrap();
        assert_eq!(pairs.last(), Some(&(2048, 8192)));
        assert!(pairs.iter().all(|&(_, kv)| kv <= 8192));
        assert_eq!(canonical_pairs(&cfg, 0, 0, 512), None);
    }

    #[test]
    fn grid_stays_under_the_coordinate_ceiling() {
        for max_model_len in [8192, 131072] {
            for is_local in [false, true] {
                let cfg = RelBiasAttentionKernelConfig {
                    max_model_len,
                    ..config(is_local)
                };
                let grid = RelBiasAttentionSpec::sweep_grid(&cfg);
                let feasible = RelBiasAttentionSpec::infeasible_mask(&cfg, &grid)
                    .iter()
                    .filter(|&&masked| !masked)
                    .count();
                assert!(
                    feasible <= 500,
                    "local={is_local} max_model_len={max_model_len}: {feasible} coordinates"
                );
            }
        }
    }

    #[test]
    fn context_axis_reaches_max_model_len() {
        // Defect caught: a global axis that stops short of max_model_len
        // (long requests extrapolate) or doubles all the way up (past the
        // coordinate ceiling); and a local axis that grows with max_model_len
        // though a local layer reads only its window.
        let axis = |is_local: bool, max_model_len: u32| {
            let cfg = RelBiasAttentionKernelConfig {
                max_model_len,
                ..config(is_local)
            };
            RelBiasAttentionSpec::sweep_grid(&cfg).axes()[2].clone()
        };
        assert_eq!(axis(true, 8192), [1.0, 320.0, 576.0, 1088.0]);
        assert_eq!(axis(true, 131072), [1.0, 320.0, 576.0, 1088.0]);
        assert_eq!(axis(false, 8192), [1.0, 576.0, 1088.0, 2112.0, 8192.0]);
        assert_eq!(
            axis(false, 131072),
            [1.0, 576.0, 1088.0, 2112.0, 8256.0, 32832.0, 131072.0]
        );
        // A context past the window and the longest global request both
        // project inside the axis.
        let global = RelBiasAttentionKernelConfig {
            max_model_len: 131072,
            ..config(false)
        };
        let [_, _, c] = project(&global, &[(1, 131072)], 0);
        assert!((c - 131072.0).abs() <= 1.0, "{c}");
        let [_, _, c] = project(&global, &[(1, 45000), (2048, 130048)], 0);
        assert!(c > 125_000.0 && c <= 131_072.0, "{c}");
    }

    #[test]
    fn graph_padded_decode_reads_the_padded_rows() {
        // Defect caught: a FULL-graph decode step priced at the real batch's
        // split count. The fork's num_splits sees the capture size: 9 real
        // decodes padded to 16 run 4 splits (combine grid steps 8 -> 16 in
        // the diverse_100 capture), not 7, so their CTAs read ~1.75x longer
        // slices than the 8-request rows the unpadded projection leans on.
        let cfg = RelBiasAttentionKernelConfig {
            max_model_len: 131072,
            ..config(false)
        };
        assert_eq!(num_splits(&cfg, 9, 1), 7);
        assert_eq!(num_splits(&cfg, 16, 1), 4);
        // Captured iteration 4843 (9 decodes).
        let contexts = [654, 934, 1048, 2294, 17473, 25436, 34561, 64887, 84167];
        let pairs: Vec<(u32, u32)> = contexts.iter().map(|&c| (1, c)).collect();
        let [d_real, _, _] = project(&cfg, &pairs, 0);
        assert!(d_real > 8.0 && d_real < 16.0, "{d_real}");
        let [d, a, c] = project(&cfg, &pairs, 16);
        assert_eq!((d, a), (16.0, 0.0));
        // Single wave (9 x 2 x 4 CTAs): the longest request sets the context,
        // less the shared-work share the 16 canonical requests carry.
        assert!(c > 60_000.0 && c < 84_167.0, "{c}");
        // Padding is exact on a canonical padded batch, a no-op at or below
        // the real count, and ignored on local layers, which never split.
        let full = vec![(1, 32832); 16];
        let [d, _, c] = project(&cfg, &full, 16);
        assert_eq!(d, 16.0);
        assert!((c - 32832.0).abs() <= 1.0, "{c}");
        assert_eq!(project(&cfg, &pairs, 9), project(&cfg, &pairs, 0));
        assert_eq!(project(&cfg, &pairs, 4), project(&cfg, &pairs, 0));
        let local = config(true);
        assert_eq!(
            project(&local, &pairs[..3], 4),
            project(&local, &pairs[..3], 0)
        );
    }

    #[test]
    fn padded_requests_defaults_to_unpadded() {
        // Defect caught: an older input payload without the field failing to
        // deserialize.
        let input: RelBiasAttentionKernelInput =
            serde_json::from_value(serde_json::json!({"query_context_pairs": [[1, 10]]})).unwrap();
        assert_eq!(input.padded_requests, 0);
    }

    #[test]
    fn payload_matches_python_schema() {
        let cfg = config(true);
        let grid = SweepGrid::new(vec![
            Axis::values([2]),
            Axis::values([64]),
            Axis::values([512]),
        ]);
        let payload = &RelBiasAttentionSpec::enumerate(&cfg, &grid, "tml_fa4")[0];
        let fields = payload.fields();
        // profiling/kernels/rel_bias_attention.py RelBiasAttentionArgs + backend.
        assert_eq!(fields.len(), 12);
        assert_eq!(fields.get("is_local"), Some(&Value::from(true)));
        assert_eq!(fields.get("kv_layout"), Some(&Value::from("lbnhc")));
        assert_eq!(fields.get("kv_dtype"), Some(&Value::from("bf16")));
        assert_eq!(
            fields.get("query_context_pairs"),
            Some(&serde_json::json!([[1, 512], [1, 512], [64, 575]]))
        );
    }
}
