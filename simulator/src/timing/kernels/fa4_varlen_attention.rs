//! FlashAttention-4 varlen attention over one paged batch: vLLM's
//! `flash_attn_varlen_func(fa_version=4)` as `FlashAttentionDiffKVImpl.forward`
//! issues it (vllm/v1/attention/backends/flash_attn_diffkv.py), one call per
//! layer for the whole scheduler step, decode and prefill requests together.
//!
//! The call issues the FA4 SM100 forward and, when the KV range is split, the
//! `FlashPrepareScheduler` launch before it and `_flash_attn_fwd_combine`
//! after it; a non-split batch over 256 requests also runs the tile-cumsum
//! helpers. The profiler counts every launch. The KV insert is a separate op.
//!
//! The batch projects to `(num_decode_requests, prefill_tokens,
//! equivalent_context)` by the shared FA4 varlen projection
//! (`fa4_varlen_projection`); this file holds the FA4 rules it is
//! parameterized by, read from the FA4 cute interface vLLM pins (flash-attention
//! 9cd61de, `vllm/vllm_flash_attn/cute/interface.py`):
//!
//! - vLLM passes `num_splits = 0` on FA4: the builder sets its split cap only
//!   for the ahead-of-time scheduler, which is FA3's (flash_attn.py
//!   `aot_schedule = fa_version == 3`). FA4 then picks the split count itself
//!   (`_get_fwd_config` + `num_splits_heuristic`, [`static_splits`]) and, when
//!   it splits a varlen batch, lowers it per request on device
//!   (`FlashPrepareScheduler`, [`request_splits`]).
//! - q_stage is 2 when the packed query rows (`max_seqlen_q` x q heads per KV
//!   head, no bucketing) exceed one 128-row tile; an M tile then holds
//!   `128 * q_stage` packed rows.
//! - The split decision reads `max_seqlen_k` whole, also on a sliding-window
//!   layer: `_get_fwd_config`'s local bound `(window_size_right or
//!   max_seqlen_k)` takes `max_seqlen_k` for the causal right window of 0, so
//!   a window layer splits its window range like a global layer splits its
//!   context.
//!
//! A decode-only step of at most `full_cudagraph_max_size` requests replays a
//! FULL CUDA graph (FULL_AND_PIECEWISE): the host-side split choice was made at
//! capture, over the capture size (`padded_requests`, zero-length padding
//! requests) and `max_seqlen_k = max_model_len`
//! (`gpu_model_runner._build_attention_metadata`, `for_cudagraph_capture`).
//! The device-side per-request split still reads the real context lengths.
//! Every other step runs eagerly over its real requests and longest context.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::fa4_varlen_projection::{
    canonical_pairs, counts, global_context_axis, project, Fa4VarlenCall, VarlenTiling, B200_SMS,
    CONTEXT_TILE_OFFSET,
};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// FA4 SM100 tile: 128 packed query rows per stage, 128 keys per KV tile.
const TILE_M: u32 = 128;
const TILE_N: u32 = 128;
/// `num_splits_heuristic`'s split cap.
const MAX_SPLITS: u32 = 128;
/// With different QK and V head dims a split needs at least this many 128-key
/// tiles (the float32 partial output then fits in shared memory only at a
/// 64-key tile); below it FA4 does not split. So an eager step splits only
/// past `SPLIT_MIN_TILES * TILE_N` = 8192 keys (exactly: past 8064).
const SPLIT_MIN_TILES: u32 = 64;
/// `FlashPrepareScheduler` handles 31 requests per warp and up to 32 warps
/// in one CTA; past that its grid has more than one CTA and every request
/// keeps one split.
const PREPARE_MAX_REQUESTS: u32 = 31 * 32;
/// Prefill-token counts on the grid past the first M tile, besides the
/// per-request split cliffs (`split_cliff_prefills`): a 4x step from 64
/// (a fresh chunk's causal work grows with the square of its length), up to
/// the 2048-token step budget.
const PREFILL_KNOTS: [u32; 3] = [64, 256, 2048];
/// The grid ceiling for this kind, above the usual 500: a long-context chunk's
/// time halves where `FlashPrepareScheduler` drops it from two KV splits to
/// one (and falls by a third from three to two), so the prefill axis carries
/// a knot on each side of the one-split cliff and at the two-split edge.
/// That is 8 prefill knots against the 7 a doubling axis needs: GA 568 and
/// SWA 497 feasible coordinates at 16 query heads.
const GRID_CEILING: usize = 600;
/// Decode-request counts on the grid besides the split edge
/// (`first_unsplit_decodes`). A graph decode splits `148 / (B x KV heads)`
/// ways, so time grows about linearly in B between powers of two and the
/// wave-placed decode coordinate puts the steps between them; 256 is the usual
/// `max_num_seqs`.
const DECODE_KNOTS: [u32; 8] = [0, 1, 8, 16, 32, 64, 128, 256];

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Fa4VarlenAttentionKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim_qk: Dim,
    pub head_dim_v: Dim,
    /// Keys a query sees on a sliding-window layer, itself included (vLLM
    /// `window_size = [window - 1, 0]`); 0 = full causal.
    pub window: u32,
    /// Per-head learnable sink logits (`s_aux`) in the softmax denominator.
    pub sinks: bool,
    pub page_size: u32,
    /// `lbnhc`: K and V of one (token, head) side by side in a page.
    pub kv_layout: String,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
    /// The split count's `max_seqlen_k` in a FULL CUDA graph, and the context
    /// axis top.
    pub max_model_len: u32,
    /// Largest decode-only step that replays a FULL CUDA graph (vLLM
    /// `max_cudagraph_capture_size` under a mode with full decode graphs); 0
    /// when no step does.
    pub full_cudagraph_max_size: u32,
}

/// One scheduler step's requests as `(query_len, context_len)`, context
/// including the query, in the order the batch holds them.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Fa4VarlenAttentionKernelInput {
    pub query_context_pairs: Vec<(u32, u32)>,
    /// The capture size a FULL CUDA graph decode step is launched over (the
    /// real requests plus zero-length padding); 0, or anything up to the real
    /// count, means no padding. Ignored on a step that runs eagerly.
    #[serde(default)]
    pub padded_requests: u32,
}

impl SweepCoords for Fa4VarlenAttentionKernelInput {
    fn coords(&self) -> Coords {
        // The equivalent context needs the Config; `cache_coords` owns the
        // real projection. This config-free view reports the longest context.
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

impl Fa4VarlenAttentionKernelConfig {
    fn q_heads_per_kv_head(&self) -> u32 {
        self.num_qo_heads.get() / self.num_kv_heads.get()
    }

    /// Whether `pairs` replays a FULL CUDA graph: decode-only and within the
    /// largest capture size.
    fn replays_graph(&self, pairs: &[(u32, u32)]) -> bool {
        self.full_cudagraph_max_size > 0
            && pairs.len() as u32 <= self.full_cudagraph_max_size
            && pairs.iter().all(|&(query, _)| query == 1)
    }

    /// A chunk of at most `2 * TILE_M / g` tokens fits one M tile (q_stage 1
    /// up to `TILE_M / g` tokens, q_stage 2 past it), so it runs one tile
    /// whatever its length.
    fn first_tile_tokens(&self) -> u32 {
        (2 * TILE_M / self.q_heads_per_kv_head()).max(1)
    }

    /// The first decode count a graph decode no longer splits: from here
    /// `148 / (B x KV heads)` is 1 (75 requests at 1 KV head, 38 at 2), and
    /// the longest CTA reads a whole request instead of half of it.
    fn first_unsplit_decodes(&self) -> u32 {
        B200_SMS as u32 / 2 / self.num_kv_heads.get() + 1
    }
}

/// `num_splits_heuristic`: no split for a short KV range, else as many splits
/// as fill the SMs once, capped. 0 (more CTAs than SMs) runs unsplit.
fn split_heuristic(total_m_blocks: u32, kv_blocks: u32) -> u32 {
    if kv_blocks <= 4 || total_m_blocks == 0 {
        return 1;
    }
    (B200_SMS as u32 / total_m_blocks)
        .min(MAX_SPLITS)
        .min(kv_blocks)
}

/// `_get_fwd_config` on SM100 with `num_splits = 0`: the static split count
/// and the KV tile it runs, from the launch's CTA count (requests x KV heads x
/// M blocks) and `max_seqlen_k`.
fn static_splits(
    config: &Fa4VarlenAttentionKernelConfig,
    total_m_blocks: u32,
    max_kv: u32,
) -> (u32, u32) {
    let (qk, v) = (config.head_dim_qk.get(), config.head_dim_v.get());
    if qk == 256 && v == 256 {
        // The dedicated hd256 kernel has no split-KV variant.
        return (1, TILE_N);
    }
    let kv_blocks = max_kv.div_ceil(TILE_N);
    let splits = split_heuristic(total_m_blocks, kv_blocks);
    if qk != v && splits > 1 {
        if kv_blocks >= SPLIT_MIN_TILES && v != 512 {
            let tile_n = TILE_N / 2;
            let splits = split_heuristic(total_m_blocks, max_kv.div_ceil(tile_n));
            return (splits.max(1), tile_n);
        }
        return (1, TILE_N);
    }
    (splits.max(1), TILE_N)
}

/// `FlashPrepareScheduler` (prepare_scheduler.py): each request's split count,
/// its KV blocks over the blocks every SM gets (the batch's blocks x 1.157 x
/// KV heads over 148 SMs, in float32 as the device computes it), capped by the
/// static count. Padding requests have no KV blocks.
fn request_splits(
    config: &Fa4VarlenAttentionKernelConfig,
    pairs: &[(u32, u32)],
    launched: u32,
    static_count: u32,
    tile_n: u32,
) -> Vec<u32> {
    if static_count <= 1 || launched > PREPARE_MAX_REQUESTS {
        return vec![1; pairs.len()];
    }
    let g = config.q_heads_per_kv_head();
    let blocks =
        |&(query, context): &(u32, u32)| (context.div_ceil(tile_n), (query * g).div_ceil(TILE_M));
    let total: u32 = pairs
        .iter()
        .map(|pair| {
            let (kv_blocks, m_blocks) = blocks(pair);
            kv_blocks * m_blocks
        })
        .sum();
    let sms = B200_SMS as f32;
    let per_sm =
        ((total as f32 * prepare_margin() * config.num_kv_heads.get() as f32 / sms) as u32).max(1);
    pairs
        .iter()
        .map(|pair| blocks(pair).0.div_ceil(per_sm).min(static_count).max(1))
        .collect()
}

/// `FlashPrepareScheduler`'s `sm_margin` (prepare_scheduler.py).
fn prepare_margin() -> f32 {
    (B200_SMS as f32 / 128.0 + 0.001).max(1.1)
}

/// The prefill lengths where a lone chunk over a long context changes its
/// per-request split count: `FlashPrepareScheduler` gives a request
/// `ceil(kv_blocks / per_sm)` splits with `per_sm` about `kv_blocks x m_blocks
/// x margin x KV heads / 148`, so the chunk keeps at most `k` splits once its
/// M blocks (`ceil(P x G / 128)`) reach `148 / (k x margin x KV heads)`,
/// whatever the context. Returns the first two-split length, the last
/// two-split length and the first one-split length.
fn split_cliff_prefills(config: &Fa4VarlenAttentionKernelConfig) -> [u32; 3] {
    let per_block = prepare_margin() * config.num_kv_heads.get() as f32;
    let first_with_at_most = |splits: u32| {
        let m_blocks = (1..)
            .find(|&m| m as f32 * per_block * splits as f32 >= B200_SMS as f32)
            .expect("an M-block count reaches the SMs");
        (m_blocks - 1) * TILE_M / config.q_heads_per_kv_head() + 1
    };
    let one = first_with_at_most(1);
    [first_with_at_most(2), one - 1, one]
}

impl Fa4VarlenCall for Fa4VarlenAttentionKernelConfig {
    fn num_kv_heads(&self) -> u32 {
        self.num_kv_heads.get()
    }

    fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    /// The keys charged to a windowed query: FA4 reads whole 128-key tiles
    /// (block_info.py `get_n_block_min_max` rounds the window start down),
    /// and a window that starts off a tile boundary (almost every one) spans
    /// on average `window + TILE_N - 1` keys of tiles; a 128-key window reads
    /// two tiles, not one.
    fn window(&self) -> Option<u32> {
        (self.window > 0).then_some(self.window + TILE_N - 1)
    }

    /// A step of more CTAs than SMs runs in waves: the SM100 forward fills
    /// shared memory to 224 KiB (flash_fwd_sm100.py `kv_stage`), so one CTA
    /// is resident per SM; a non-split decode-only launch walks its tiles
    /// with the static persistent scheduler (interface.py
    /// `is_static_persistent`, grid `min(SMs, tiles)`), any other varlen
    /// launch is one CTA per tile (`SingleTileVarlenScheduler`). A mixed
    /// step's chunk CTAs alone outnumber the SMs at 2048 tokens.
    fn whole_waves(&self) -> bool {
        true
    }

    /// A window layer splits each request by its full context
    /// (`FlashPrepareScheduler`, prepare_scheduler.py: `ceil(kv_blocks /
    /// per_sm)` over `seqused_k`) while each split reads only the window
    /// (block_info.py `get_n_block_min_max`). Beside one long request the
    /// short ones keep one split and read the whole window in one CTA: a
    /// ragged step's critical path, which no uniform batch has. Its short
    /// requests sit one window of tiles plus half a tile past the start, where
    /// every query reads a whole window.
    fn ragged_short_context(&self) -> Option<u32> {
        (self.window > 0).then_some(self.window + TILE_N + CONTEXT_TILE_OFFSET)
    }

    fn decode_axis(&self) -> Vec<u32> {
        let mut axis = DECODE_KNOTS.to_vec();
        axis.push(self.first_unsplit_decodes());
        axis.sort_unstable();
        axis.dedup();
        axis
    }

    /// A full-causal layer's work grows to `max_model_len`. A window layer's
    /// CTAs read only the window, but its split count keeps changing with the
    /// context: an eager step starts splitting past 8192 keys, and the
    /// per-request count settles once the batch's KV blocks give every SM more
    /// than one; 4x past the split edge it is flat.
    fn context_top(&self) -> u32 {
        if self.window > 0 {
            (4 * SPLIT_MIN_TILES * TILE_N + CONTEXT_TILE_OFFSET).min(self.max_model_len)
        } else {
            self.max_model_len
        }
    }

    /// A chunk within the first q_stage-2 M tile runs that one tile whatever
    /// its length; past it the count is the coordinate (no bucketing: vLLM
    /// passes the exact `max_query_len`).
    fn prefill_coordinate(&self, prefill: u32) -> u32 {
        match prefill {
            0 => 0,
            prefill => prefill.max(self.first_tile_tokens()),
        }
    }

    /// An eager canonical batch starts splitting where its longest context
    /// reaches `SPLIT_MIN_TILES` KV tiles (the makespan drops there): at `C`
    /// for its decodes, at `C - 1 + prefill` for its chunk. A graph decode's
    /// split count does not move with the context.
    fn context_breaks(&self, decodes: u32, prefill: u32) -> Vec<u32> {
        let edge = (SPLIT_MIN_TILES - 1) * TILE_N + 1;
        let replays = self.full_cudagraph_max_size > 0
            && prefill == 0
            && decodes <= self.full_cudagraph_max_size;
        let mut breaks = Vec::new();
        if replays {
            return breaks;
        }
        if decodes > 0 {
            breaks.push(edge - 1);
        }
        if prefill > 0 && prefill < edge {
            breaks.push(edge - prefill);
        }
        breaks
    }

    fn launched_requests(&self, pairs: &[(u32, u32)], padded_requests: u32) -> u32 {
        if self.replays_graph(pairs) {
            padded_requests.max(pairs.len() as u32)
        } else {
            pairs.len() as u32
        }
    }

    fn tiling(&self, pairs: &[(u32, u32)], launched: u32) -> VarlenTiling {
        let g = self.q_heads_per_kv_head();
        // A replayed graph keeps the capture's host-side choice: one query
        // token per request and `max_seqlen_k = max_model_len`.
        let (max_query, max_kv) = if self.replays_graph(pairs) {
            (1, self.max_model_len)
        } else {
            (
                pairs.iter().map(|&(query, _)| query).max().unwrap(),
                pairs.iter().map(|&(_, context)| context).max().unwrap(),
            )
        };
        let q_stage = if max_query * g > TILE_M { 2 } else { 1 };
        let m_blocks = (max_query * g).div_ceil(TILE_M * q_stage);
        let total_m_blocks = launched * self.num_kv_heads.get() * m_blocks;
        let (static_count, tile_n) = static_splits(self, total_m_blocks, max_kv);
        VarlenTiling {
            tile_tokens: (TILE_M * q_stage / g).max(1),
            splits: request_splits(self, pairs, launched, static_count, tile_n),
        }
    }
}

fn context_axis(config: &Fa4VarlenAttentionKernelConfig) -> Vec<f64> {
    let top = config.context_top();
    let split_edge = SPLIT_MIN_TILES * TILE_N;
    let values = if config.window == 0 {
        // 1, then 512 .. 8192 keys doubling and 4x past the split edge, each
        // half a tile past the boundary. Below the edge the per-request split
        // count still rises with the context; past it every CTA reads a fixed
        // share of its request and time is linear in the context.
        global_context_axis(config.max_model_len, split_edge, top)
    } else {
        // Below the window a query sees its whole prefix (5/8 of it; one KV
        // tile past it the work is flat); past it only the split count moves:
        // with the context while a graph decode has fewer KV blocks than
        // splits, and on at the eager split edge.
        let window = config.window;
        let mut values = vec![
            1,
            window * 5 / 8,
            window + CONTEXT_TILE_OFFSET,
            1024 + CONTEXT_TILE_OFFSET,
            4096 + CONTEXT_TILE_OFFSET,
            split_edge + CONTEXT_TILE_OFFSET,
        ];
        values.retain(|&context| context > 0 && context < top);
        values.sort_unstable();
        values.dedup();
        values.push(top);
        // The ragged segment's end: one decode at `max_model_len`, the rest
        // short (`ragged_short_context`, `canonical_pairs`).
        if top < config.max_model_len {
            values.push(config.max_model_len);
        }
        values
    };
    Axis::values(values)
}

fn prefill_axis(config: &Fa4VarlenAttentionKernelConfig) -> Vec<f64> {
    let first = config.first_tile_tokens();
    let mut values: Vec<u32> = PREFILL_KNOTS
        .into_iter()
        .chain(split_cliff_prefills(config))
        .filter(|&knot| knot > first)
        .collect();
    values.extend([0, first]);
    values.sort_unstable();
    values.dedup();
    Axis::values(values)
}

pub struct Fa4VarlenAttentionSpec;

impl KernelSpec for Fa4VarlenAttentionSpec {
    type Config = Fa4VarlenAttentionKernelConfig;
    type Input = Fa4VarlenAttentionKernelInput;

    const KIND: KernelKind = "fa4_varlen_attention";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert!(config.max_model_len > 0, "max_model_len must be positive");
        assert!(
            config.num_qo_heads.get() % config.num_kv_heads.get() == 0,
            "num_qo_heads must be a multiple of num_kv_heads"
        );
        SweepGrid::new(vec![
            Axis::values(config.decode_axis()),
            prefill_axis(config),
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
                .with("head_dim_qk", config.head_dim_qk.get())
                .with("head_dim_v", config.head_dim_v.get())
                .with("window", config.window)
                .with("sinks", config.sinks)
                .with("page_size", config.page_size)
                .with("kv_layout", config.kv_layout.clone())
                .with("q_dtype", config.q_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("max_model_len", config.max_model_len)
                .with("full_cudagraph_max_size", config.full_cudagraph_max_size)
                .with("query_context_pairs", serde_json::json!(pairs))
                .with("padded_requests", 0)
        })
    }
}

register_kernel!(Fa4VarlenAttentionKernel, Fa4VarlenAttentionSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::fa4_varlen_projection::makespan;
    use serde_json::Value;

    /// The two MiMo-V2.6-Flash TP4 attention layers the kind was built for:
    /// full causal (16 Q / 1 KV head, page 32) and the 128-token window with
    /// sinks (16 Q / 2 KV heads, page 16).
    fn config(window: bool) -> Fa4VarlenAttentionKernelConfig {
        Fa4VarlenAttentionKernelConfig {
            backends: vec!["vllm_fa4"],
            gpu_name: "NVIDIA B200".to_string(),
            num_qo_heads: 16.into(),
            num_kv_heads: (if window { 2 } else { 1 }).into(),
            head_dim_qk: 192.into(),
            head_dim_v: 128.into(),
            window: if window { 128 } else { 0 },
            sinks: window,
            page_size: if window { 16 } else { 32 },
            kv_layout: "lbnhc".to_string(),
            q_dtype: DType::Bf16,
            kv_dtype: DType::Bf16,
            max_model_len: 131072,
            full_cudagraph_max_size: 200,
        }
    }

    #[test]
    fn static_split_matches_fa4_get_fwd_config() {
        // Defect caught: a split count, q_stage or KV tile that drifts from
        // FA4's host-side choice. Rows are `_get_fwd_config(arch=100,
        // head_dim=192, head_dim_v=128, num_splits=0)` run on the pinned cute
        // interface with FLASH_ATTENTION_NUM_SMS=148: (KV heads, batch,
        // max_seqlen_q, max_seqlen_k, q_stage * 1000 + tile_n, num_splits).
        let rows: [(u32, u32, u32, u32, u32, u32); 42] = [
            (1, 1, 1, 8064, 1128, 1),
            (1, 1, 1, 8065, 1064, 127),
            (1, 1, 1, 131072, 1064, 128),
            (1, 1, 8, 513, 1128, 1),
            (1, 1, 17, 8192, 2064, 74),
            (1, 1, 2048, 131072, 2128, 1),
            (1, 2, 64, 30000, 2064, 18),
            (1, 3, 200, 131072, 2064, 3),
            (1, 9, 9, 9000, 2064, 16),
            (1, 16, 1, 512, 1128, 1),
            (1, 37, 1, 131072, 1064, 4),
            (1, 38, 1, 131072, 1064, 3),
            (1, 74, 1, 131072, 1064, 2),
            (1, 75, 1, 131072, 1128, 1),
            (2, 1, 1, 8064, 1128, 1),
            (2, 1, 1, 8065, 1064, 74),
            (2, 1, 1, 131072, 1064, 74),
            (2, 1, 8, 513, 1128, 1),
            (2, 1, 17, 8192, 2064, 74),
            (2, 1, 2048, 131072, 2128, 1),
            (2, 2, 64, 30000, 2064, 18),
            (2, 3, 200, 131072, 2064, 3),
            (2, 9, 9, 9000, 1064, 8),
            (2, 16, 1, 512, 1128, 1),
            (2, 37, 1, 131072, 1064, 2),
            (2, 38, 1, 131072, 1128, 1),
            (2, 74, 1, 131072, 1128, 1),
            (2, 75, 1, 131072, 1128, 0),
            (1, 100, 8, 8064, 1128, 1),
            (2, 200, 200, 131072, 2128, 0),
            (1, 17, 64, 8064, 2128, 1),
            (2, 9, 2048, 8065, 2128, 0),
            (2, 74, 9, 512, 1128, 1),
            (1, 8, 9, 30000, 2064, 18),
            (1, 1, 64, 4000, 2128, 1),
            (2, 74, 1, 512, 1128, 1),
            (1, 200, 9, 513, 2128, 0),
            (1, 75, 2048, 131072, 2128, 0),
            (1, 50, 200, 8065, 2128, 0),
            (2, 74, 1, 8192, 1128, 1),
            (2, 200, 64, 8065, 2128, 0),
            (2, 74, 16, 512, 1128, 1),
        ];
        for (kv_heads, batch, max_q, max_kv, stage_tile, splits) in rows {
            let cfg = config(kv_heads == 2);
            let g = cfg.q_heads_per_kv_head();
            let q_stage = if max_q * g > TILE_M { 2 } else { 1 };
            let m_blocks = (max_q * g).div_ceil(TILE_M * q_stage);
            let (got, tile_n) = static_splits(&cfg, batch * kv_heads * m_blocks, max_kv);
            let case = (kv_heads, batch, max_q, max_kv);
            assert_eq!(q_stage * 1000 + tile_n, stage_tile, "{case:?}");
            // FA4 reports 0 when the CTAs already exceed the SMs: unsplit.
            assert_eq!(got, splits.max(1), "{case:?}");
        }
    }

    #[test]
    fn graph_decode_splits_at_capture_while_eager_does_not() {
        // Defect caught: a FULL-graph decode priced with the eager split rule.
        // The graph was captured at max_seqlen_k = max_model_len, so 9 decodes
        // at 1000 keys padded to 16 split 148 / 16 = 9 ways (64-key tiles),
        // where the same batch run eagerly (graphs off) reads 8 tiles and
        // never splits.
        let cfg = config(false);
        let pairs = vec![(1, 1000); 9];
        assert_eq!(cfg.launched_requests(&pairs, 16), 16);
        let graph = cfg.tiling(&pairs, 16);
        assert_eq!(graph.splits, vec![9; 9]);
        let eager = Fa4VarlenAttentionKernelConfig {
            full_cudagraph_max_size: 0,
            ..config(false)
        };
        assert_eq!(eager.launched_requests(&pairs, 16), 9);
        assert_eq!(eager.tiling(&pairs, 9).splits, vec![1; 9]);
        // A mixed step never replays a graph: padding is ignored.
        let mut mixed = pairs.clone();
        mixed.push((64, 5000));
        assert_eq!(cfg.launched_requests(&mixed, 16), 10);
    }

    #[test]
    fn device_split_follows_each_requests_share() {
        // Defect caught: one uniform split count for a ragged split batch.
        // FlashPrepareScheduler gives a short request fewer splits than the
        // static cap: one 131072-key decode beside 39 one-token ones (eager,
        // 40 CTAs -> 3 static splits) keeps 3, the short ones 1.
        let cfg = Fa4VarlenAttentionKernelConfig {
            full_cudagraph_max_size: 0,
            ..config(false)
        };
        let mut pairs = vec![(1, 131072)];
        pairs.extend(std::iter::repeat_n((1, 1), 39));
        let tiling = cfg.tiling(&pairs, 40);
        assert_eq!(tiling.splits[0], 3);
        assert!(tiling.splits[1..].iter().all(|&s| s == 1));
    }

    #[test]
    fn window_layer_splits_its_window_like_a_global_layer() {
        // Defect caught: a window layer assumed never to split. FA4 reads
        // max_seqlen_k whole for the split choice, so a graph decode of one
        // request over the 128-token window splits (the capture's combine and
        // scheduler run on every SWA decode step).
        let cfg = config(true);
        let splits = cfg.tiling(&[(1, 5000)], 1).splits;
        assert!(splits[0] > 1, "{splits:?}");
    }

    #[test]
    fn projection_is_exact_on_canonical_points() {
        // Defect caught: a projection that moves grid points, so the cache
        // never returns its own measurements.
        for window in [false, true] {
            let cfg = config(window);
            for (decodes, prefill, context) in [
                (16, 0, 2112),
                (1, 0, 4160),
                (8, 2048, 1088),
                (0, 512, 576),
                (8, 64, 1),
                (cfg.first_unsplit_decodes(), cfg.first_tile_tokens(), 192),
                (128, 0, 32832),
                (16, 0, 131072),
                (8, 256, 131072),
            ] {
                let pairs = canonical_pairs(&cfg, decodes, prefill, context).unwrap();
                let [d, a, c] = project(&cfg, &pairs, 0);
                assert_eq!(
                    (d, a),
                    (
                        f64::from(decodes),
                        f64::from(cfg.prefill_coordinate(prefill))
                    )
                );
                // The model is flat in the context where nothing moves (past a
                // window layer's settled split count, or a graph decode past
                // its split saturation); there any context with the same
                // makespan is exact.
                let same = makespan(&cfg, &pairs, pairs.len() as u32);
                let back = canonical_pairs(&cfg, decodes, prefill, c.round() as u32).unwrap();
                let rel = (makespan(&cfg, &back, back.len() as u32) - same).abs() / same;
                assert!(
                    (c - f64::from(context)).abs() <= 1.0 || rel < 1e-3,
                    "window={window} ({decodes}, {prefill}, {context}) -> {c}"
                );
            }
        }
    }

    #[test]
    fn ragged_window_decode_reaches_past_the_uniform_top() {
        // Defect caught: a window graph decode with long and short requests
        // clamped to the uniform top. The short ones keep one split (FA4
        // splits by the full context) and read their whole window in one CTA,
        // longer than any uniform batch's CTA (capture 6264, iteration 4094:
        // nine decodes padded to 16, measured 13% over the uniform top).
        let cfg = config(true);
        let real = [
            (1, 85403),
            (1, 26672),
            (1, 2879),
            (1, 2830),
            (1, 66789),
            (1, 19428),
            (1, 4550),
            (1, 124417),
            (1, 2112),
        ];
        assert_eq!(cfg.tiling(&real, 16).splits, [4, 4, 1, 1, 4, 4, 1, 4, 1]);
        let [d, a, c] = project(&cfg, &real, 16);
        assert_eq!((d, a), (16.0, 0.0));
        assert!(c > 32832.0 && c < 131072.0, "{c}");
        // The segment's end holds that critical path: the anchor splits, the
        // short decodes do not.
        let anchor = canonical_pairs(&cfg, 16, 0, 131072).unwrap();
        let splits = cfg.tiling(&anchor, 16).splits;
        assert_eq!((splits[0], splits[1]), (4, 1));
        // An all-long batch stays on the uniform axis.
        let long = [(1, 90000), (1, 120000), (1, 100000)];
        assert!(project(&cfg, &long, 16)[2] <= 32832.0);
        // A full-causal layer has no segment.
        assert_eq!(config(false).ragged_short_context(), None);
    }

    #[test]
    fn small_chunk_reads_the_one_tile_row() {
        // Defect caught: a 3-token chunk interpolated toward the empty-prefill
        // plane though it runs a whole M tile.
        let cfg = config(false);
        assert_eq!(cfg.prefill_coordinate(3), 16);
        assert_eq!(cfg.prefill_coordinate(40), 40);
        assert_eq!(config(true).prefill_coordinate(20), 32);
    }

    #[test]
    fn grid_axes_follow_the_config_and_stay_under_the_ceiling() {
        // Defect caught: knots that ignore the KV-head count (the split edge
        // moves with it), a prefill axis that misses the split cliffs, and
        // grids past the kind's ceiling.
        let axes = |window| {
            Fa4VarlenAttentionSpec::sweep_grid(&config(window))
                .axes()
                .to_vec()
        };
        let global = axes(false);
        assert_eq!(
            global[0],
            [0.0, 1.0, 8.0, 16.0, 32.0, 64.0, 75.0, 128.0, 256.0]
        );
        assert_eq!(
            global[1],
            [0.0, 16.0, 64.0, 256.0, 505.0, 1016.0, 1017.0, 2048.0]
        );
        assert_eq!(
            global[2],
            [1.0, 576.0, 1088.0, 2112.0, 4160.0, 8256.0, 32832.0, 131072.0]
        );
        let local = axes(true);
        assert!(local[0].contains(&38.0));
        assert_eq!(
            local[1],
            [0.0, 32.0, 64.0, 256.0, 497.0, 1008.0, 1009.0, 2048.0]
        );
        assert_eq!(
            local[2],
            [1.0, 80.0, 192.0, 1088.0, 4160.0, 8256.0, 32832.0, 131072.0]
        );
        for window in [false, true] {
            let cfg = config(window);
            let grid = Fa4VarlenAttentionSpec::sweep_grid(&cfg);
            let feasible = Fa4VarlenAttentionSpec::infeasible_mask(&cfg, &grid)
                .iter()
                .filter(|&&masked| !masked)
                .count();
            assert!(feasible <= GRID_CEILING, "window={window}: {feasible}");
        }
    }

    #[test]
    fn prefill_knots_sit_on_the_lone_chunk_split_cliffs() {
        // Defect caught: a cliff knot off by one M block, so a cell
        // interpolates across a split change (a 1000-token chunk at 121000
        // keys measured 1.26 ms against 2.38 ms interpolated toward 1024).
        for window in [false, true] {
            let cfg = config(window);
            let [two, last_two, one] = split_cliff_prefills(&cfg);
            let split = |prefill: u32| {
                let pairs = [(prefill, cfg.max_model_len)];
                cfg.tiling(&pairs, 1).splits[0]
            };
            assert_eq!(split(two - 1), 3, "window={window}");
            assert_eq!(split(two), 2, "window={window}");
            assert_eq!(split(last_two), 2, "window={window}");
            assert_eq!(split(one), 1, "window={window}");
        }
    }

    #[test]
    fn padded_requests_defaults_to_unpadded() {
        // Defect caught: an input payload without the field failing to
        // deserialize.
        let input: Fa4VarlenAttentionKernelInput =
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
        let payload = &Fa4VarlenAttentionSpec::enumerate(&cfg, &grid, "vllm_fa4")[0];
        let fields = payload.fields();
        // profiling/kernels/fa4_varlen_attention.py Fa4VarlenAttentionArgs + backend.
        assert_eq!(fields.len(), 15);
        assert_eq!(fields.get("window"), Some(&Value::from(128)));
        assert_eq!(fields.get("sinks"), Some(&Value::from(true)));
        assert_eq!(
            fields.get("full_cudagraph_max_size"),
            Some(&Value::from(200))
        );
        assert_eq!(fields.get("padded_requests"), Some(&Value::from(0)));
        assert_eq!(
            fields.get("query_context_pairs"),
            Some(&serde_json::json!([[1, 512], [1, 512], [64, 575]]))
        );
    }
}
