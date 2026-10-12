//! The cache projection shared by kinds that cost ONE FlashAttention-4 SM100
//! varlen forward over a whole scheduler step (decode and prefill requests in
//! one call).
//!
//! The runtime input is the exact `query_context_pairs` batch (context
//! including the query) plus the request count a FULL CUDA graph launches over.
//! It projects to three cache coordinates:
//!
//! - `num_decode_requests`: requests with one query token, plus graph padding
//!   slots (they are decode-shaped and change the launch the same way);
//! - `prefill_tokens`: query tokens of the other requests, through the kind's
//!   `prefill_coordinate` rule;
//! - `equivalent_context`: the context `C` at which the canonical batch with the
//!   same two counts has the same modelled FA4 makespan as the real batch. The
//!   canonical batch is `num_decode_requests` decodes at context `C` and, when
//!   `prefill_tokens > 0`, one chunk of `prefill_tokens` with prefix
//!   `min(C - 1, max_model_len - prefill_tokens)`.
//!
//! The makespan model counts KV tiles (128 keys) per CTA: one CTA per (request,
//! KV head, M tile, split), the M tile holding the kind's `tile_tokens` query
//! tokens, a sliding-window tile reading only its window. A CTA costs its tiles
//! over its split count plus a fixed overhead, and the launch lasts
//! `max(longest CTA, total / SMs)` plus a quarter of `total / SMs` for the
//! bandwidth the CTAs share. Where it is monotone in `C` the projection is exact
//! on every canonical point, and a heterogeneous batch maps to the uniform
//! context that keeps the same critical path: a single-wave batch follows its
//! longest request, a many-wave batch its total work.
//!
//! Between two grid decode counts the decode coordinate is placed by the same
//! CTA model counted in whole waves of 148 CTAs (`decode_coordinate`), so a
//! wave or split-count step lands where it happens instead of being averaged
//! across the cell.
//!
//! What differs between FA4 call sites (the split rule, the query tiling, the
//! window, which steps a graph pads, the grid axes) is the kind's
//! [`Fa4VarlenCall`] implementation.

/// SMs on the only supported GPU (NVIDIA B200).
pub(crate) const B200_SMS: f64 = 148.0;
/// The work unit: one KV tile of 128 keys.
const TILE_N: f64 = 128.0;
/// Per-CTA prologue/epilogue cost in KV-tile units.
const CTA_OVERHEAD_TILES: f64 = 1.0;
/// Weight of the per-SM share of all work on top of the critical path: CTAs
/// share HBM and L2, so work off the critical path still slows it. It also
/// keeps the model strictly increasing where the critical path is flat (a
/// sliding-window prefill chunk beside short decodes).
const SHARED_WORK_WEIGHT: f64 = 0.25;
/// Contexts on the grid sit half a KV tile past a tile boundary (`128 k + 64`):
/// a real context is rarely tile-aligned, and an aligned one reads one KV tile
/// fewer (B200 local decode: 18.7 us at 1024 keys, 20.8 us at 6000).
pub(crate) const CONTEXT_TILE_OFFSET: u32 = 64;
const BISECTION_STEPS: usize = 48;

/// How one call site launches FA4 over a batch.
pub(crate) struct VarlenTiling {
    /// Query tokens per M tile (packed rows per tile over q heads per KV head).
    pub tile_tokens: u32,
    /// KV splits of each request, in batch order.
    pub splits: Vec<u32>,
}

/// The call-specific rules of one FA4 varlen call site.
pub(crate) trait Fa4VarlenCall {
    fn num_kv_heads(&self) -> u32;
    fn max_model_len(&self) -> u32;
    /// Keys a query sees, itself included, on a sliding-window layer; `None`
    /// for full causal attention.
    fn window(&self) -> Option<u32>;
    /// Decode-request counts on the grid.
    fn decode_axis(&self) -> Vec<u32>;
    /// The top of the context axis, where the longest batch lands.
    fn context_top(&self) -> u32;
    /// The cache coordinate of `prefill` query tokens.
    fn prefill_coordinate(&self, prefill: u32) -> u32;
    /// Requests the call is launched over: `pairs.len()`, or the graph-padded
    /// size when this step runs from a FULL CUDA graph (padding slots are
    /// zero-length requests that add no CTA work).
    fn launched_requests(&self, pairs: &[(u32, u32)], padded_requests: u32) -> u32;
    /// Contexts after which the canonical batch's modelled makespan drops (a
    /// split count switching on with the context): the inversion takes the
    /// first segment between them that reaches the target. Empty where the
    /// model is monotone in the context.
    fn context_breaks(&self, _decodes: u32, _prefill: u32) -> Vec<u32> {
        Vec::new()
    }

    /// The tile shape and per-request split counts of a launch over `pairs`
    /// padded to `launched` requests.
    fn tiling(&self, pairs: &[(u32, u32)], launched: u32) -> VarlenTiling;
}

/// `(decode requests, prefill query tokens)`.
pub(crate) fn counts(pairs: &[(u32, u32)]) -> (u32, u32) {
    let decodes = pairs.iter().filter(|&&(query, _)| query == 1).count() as u32;
    let prefill = pairs
        .iter()
        .filter(|&&(query, _)| query > 1)
        .map(|&(query, _)| query)
        .sum();
    (decodes, prefill)
}

/// Every CTA's modelled work in KV-tile units (see the module doc).
/// `launched` is the batch the split rule sees: `pairs.len()`, or the
/// graph-padded size (the padding slots add no CTA work).
fn cta_works<C: Fa4VarlenCall>(call: &C, pairs: &[(u32, u32)], launched: u32) -> Vec<f64> {
    assert!(!pairs.is_empty(), "query_context_pairs must be non-empty");
    let VarlenTiling {
        tile_tokens,
        splits,
    } = call.tiling(pairs, launched);
    debug_assert_eq!(splits.len(), pairs.len());
    let kv_heads = call.num_kv_heads();
    let window = call.window().map(f64::from);

    let mut works = Vec::new();
    for (&(query, context), &splits) in pairs.iter().zip(&splits) {
        assert!(
            query > 0 && query <= context,
            "each pair needs 0 < query <= context"
        );
        let copies = (splits * kv_heads) as usize;
        let prefix = f64::from(context - query);
        for tile in 0..query.div_ceil(tile_tokens) {
            let first = prefix + f64::from(tile * tile_tokens);
            let end = prefix + f64::from(((tile + 1) * tile_tokens).min(query));
            let start = match window {
                Some(window) => (first + 1.0 - window).max(0.0),
                None => 0.0,
            };
            let cta = (end - start) / TILE_N / f64::from(splits) + CTA_OVERHEAD_TILES;
            works.extend(std::iter::repeat_n(cta, copies));
        }
    }
    works
}

/// Modelled FA4 makespan: `max(longest CTA, total / SMs)` plus the shared
/// share. Continuous in every context; the context projection inverts it.
pub(crate) fn makespan<C: Fa4VarlenCall>(call: &C, pairs: &[(u32, u32)], launched: u32) -> f64 {
    let works = cta_works(call, pairs, launched);
    let longest = works.iter().copied().fold(0.0, f64::max);
    let per_sm = works.iter().sum::<f64>() / B200_SMS;
    longest.max(per_sm) + SHARED_WORK_WEIGHT * per_sm
}

/// Modelled FA4 makespan with whole CTA waves: CTAs longest first, each wave of
/// 148 lasting as long as its first. It carries the steps a batch-size change
/// makes (another wave, a different split count), which the decode-count
/// coordinate follows between grid points.
fn wave_makespan<C: Fa4VarlenCall>(call: &C, pairs: &[(u32, u32)]) -> f64 {
    let mut works = cta_works(call, pairs, pairs.len() as u32);
    works.sort_unstable_by(|lhs, rhs| rhs.total_cmp(lhs));
    let waves = works.iter().step_by(B200_SMS as usize).sum::<f64>();
    waves + SHARED_WORK_WEIGHT * works.iter().sum::<f64>() / B200_SMS
}

/// The decode-count coordinate. On the grid it is the count; between two grid
/// counts it is placed where the wave model of the canonical batch puts it, so
/// a step (another CTA wave, a split count changing with the batch) lands where
/// it happens rather than being spread over the cell.
pub(crate) fn decode_coordinate<C: Fa4VarlenCall>(
    call: &C,
    decodes: u32,
    prefill: u32,
    context: f64,
) -> f64 {
    let axis = call.decode_axis();
    let Some(hi_index) = axis.iter().position(|&grid| grid >= decodes) else {
        return f64::from(decodes);
    };
    let hi = axis[hi_index];
    if hi == decodes || hi_index == 0 {
        return f64::from(decodes);
    }
    let lo = axis[hi_index - 1];
    let context = context.round() as u32;
    let model = |count: u32| {
        let pairs = canonical_pairs(call, count, prefill, context)
            .expect("a grid neighbour of a decode count >= 1 is non-empty");
        wave_makespan(call, &pairs)
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
pub(crate) fn canonical_pairs<C: Fa4VarlenCall>(
    call: &C,
    decodes: u32,
    prefill: u32,
    context: u32,
) -> Option<Vec<(u32, u32)>> {
    if decodes == 0 && prefill == 0 {
        return None;
    }
    let max_model_len = call.max_model_len();
    let context = context.clamp(1, max_model_len);
    let mut pairs = vec![(1, context); decodes as usize];
    if prefill > 0 {
        let prefix = (context - 1).min(max_model_len.saturating_sub(prefill));
        pairs.push((prefill, prefix + prefill));
    }
    Some(pairs)
}

/// The cache coordinates of a real batch launched over `padded_requests`
/// slots (see the module doc; 0 = unpadded).
pub(crate) fn project<C: Fa4VarlenCall>(
    call: &C,
    pairs: &[(u32, u32)],
    padded_requests: u32,
) -> [f64; 3] {
    // Known limit: several prefill requests in one step are projected as one
    // chunk of their summed query tokens. When the real batch has less work
    // than that chunk at C = 1 (eight fresh 256-token prompts against one
    // 2048-token chunk), the context clamps to C = 1 and the step is
    // overpriced.
    let (decodes, prefill) = counts(pairs);
    let prefill = call.prefill_coordinate(prefill);
    let launched = call.launched_requests(pairs, padded_requests);
    // Padding slots are decode-shaped: the canonical batch of `decodes +
    // padding` decodes is launched over the same `launched` requests.
    let decodes = decodes + (launched - pairs.len() as u32);
    let context = equivalent_context(call, pairs, launched, decodes, prefill);
    [
        decode_coordinate(call, decodes, prefill, context),
        f64::from(prefill),
        context,
    ]
}

/// The context at which the canonical batch has the real batch's makespan.
fn equivalent_context<C: Fa4VarlenCall>(
    call: &C,
    pairs: &[(u32, u32)],
    launched: u32,
    decodes: u32,
    prefill: u32,
) -> f64 {
    let target = makespan(call, pairs, launched);
    let canonical = |context: f64| {
        let pairs = canonical_pairs(call, decodes, prefill, context.round() as u32)
            .expect("a non-empty batch has a canonical batch");
        makespan(call, &pairs, pairs.len() as u32)
    };
    // Fractional contexts: the model is linear in the context between integer
    // points, so interpolate there to keep the projection continuous.
    let model = |context: f64| {
        let lo = context.floor();
        let t = context - lo;
        canonical(lo) * (1.0 - t) + canonical(lo + 1.0) * t
    };
    let mut lo = 1.0_f64;
    if target <= model(lo) {
        return lo;
    }
    // The first segment between breaks (or the whole axis) whose end reaches
    // the target; the model is monotone inside each segment. A segment that
    // ends flat at the target is searched for where it reaches it; at the
    // axis top a batch as long as that, or longer, takes the top.
    let top = call.context_top();
    let mut ends: Vec<u32> = call
        .context_breaks(decodes, prefill)
        .into_iter()
        .filter(|&end| end > 1 && end < top)
        .collect();
    ends.sort_unstable();
    ends.push(top);
    let mut reached = None;
    for end in ends {
        let reaches = if end == top {
            model(f64::from(end)) > target
        } else {
            model(f64::from(end)) >= target
        };
        let end = f64::from(end);
        if reaches {
            reached = Some(end);
            break;
        }
        lo = end;
    }
    let Some(mut hi) = reached else {
        return f64::from(top);
    };
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

/// The full-causal context axis: 1, then contexts doubling from 512 keys up to
/// `doubling_end` and growing 4x past it, each half a tile past the boundary,
/// capped by `top`.
pub(crate) fn global_context_axis(max_model_len: u32, doubling_end: u32, top: u32) -> Vec<u32> {
    let boundaries = std::iter::successors(Some(512_u32), |&keys| {
        let step = if keys < doubling_end { 2 } else { 4 };
        keys.checked_mul(step)
    });
    let mut values: Vec<u32> = std::iter::once(1)
        .chain(boundaries.map(|keys| keys + CONTEXT_TILE_OFFSET))
        .take_while(|&context| context < max_model_len)
        .collect();
    values.retain(|&context| context < top);
    values.push(top);
    values
}
