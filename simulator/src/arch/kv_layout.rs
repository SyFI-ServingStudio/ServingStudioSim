//! What one request keeps in the KV cache, described part by part.
//!
//! A model's persistent attention state is not one per-token number. Global
//! attention grows with every token, a sliding-window layer keeps only its last
//! `W` tokens, a compressed cache (a DSA indexer, DeepSeek-V4's C4/C128) keeps
//! one entry per few tokens, and a recurrent layer (GDN, KDA) keeps one state per
//! request. An arch lists each kind as a [`KvCachePart`]; L5 turns the list into
//! capacity, admission and prefix-cache charges (`worker::kv::KvAccounting`).
//!
//! Bytes are summed over the part's layers **and over every rank of one KV
//! partition** (one attention TP group). Ranks share a request's cache in one of
//! two ways:
//!
//! - **split by KV head** (GQA): each rank stores its share of the heads, and a
//!   rank whose TP exceeds the head count still stores one whole head;
//!   [`gqa_kv_bytes_per_token`] counts those replicated heads.
//! - **one copy on each rank** (an MLA latent, a DSA index key, a KDA state, a
//!   DeepSeek-V4 compressed cache): the arch describes one rank's layout and
//!   [`KvCacheLayout::held_on_each_rank`] scales every byte count by the ranks.
//!
//! Each part carries two byte counts. `bytes_per_entry` is what the model must
//! store. [`PartOverhead`] and [`EngineOverhead`] add what the serving engine
//! allocates on top (padding, extra state copies, large blocks, fixed buffers);
//! the worker's `kv_overhead` option decides whether they are charged.

use serde::Serialize;

/// One request's persistent attention state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct KvCacheLayout {
    pub parts: Vec<KvCachePart>,
    pub engine: EngineOverhead,
}

/// One kind of cache in the layout, such as global attention, sliding-window
/// attention, a short-conv window, a DSA indexer or a GDN state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct KvCachePart {
    pub name: &'static str,
    pub growth: KvGrowth,
    /// Bytes of one entry: one token's KV for `PerToken { tokens_per_entry: 1 }`
    /// and `SlidingWindow`, one compressed row for a larger `tokens_per_entry`,
    /// the whole state for `PerRequest`.
    pub bytes_per_entry: u64,
    /// Whether a later request can resume from this part's retained state. A
    /// sliding window the engine always replays (DeepSeek-V4.1's bounded-replay
    /// SWA) is not reusable: a prefix hit stands, and that part recomputes its
    /// window.
    pub prefix_reusable: bool,
    #[serde(flatten)]
    pub overhead: PartOverhead,
}

/// How a part grows with the request's context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KvGrowth {
    /// One entry per `tokens_per_entry` context tokens, kept for the request's
    /// whole life. `1` for ordinary attention KV.
    PerToken { tokens_per_entry: u32 },
    /// One entry per token, only for the last `window_tokens` tokens.
    SlidingWindow { window_tokens: u32 },
    /// One entry per request, whatever its length.
    PerRequest,
}

/// What the serving engine adds to one part.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PartOverhead {
    /// Bytes the engine allocates per entry when larger than the model's own,
    /// such as a head size padded to a power of two or a mamba page padded to
    /// the attention page.
    pub padded_bytes_per_entry: Option<u64>,
    /// State copies the engine keeps per request whatever the cache mode: one
    /// plus vLLM's speculative blocks. `None` means one. L5 adds align mode's
    /// previous-step copy itself when prefix caching is on.
    pub copies: Option<u32>,
}

/// What the serving engine adds to the whole pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct EngineOverhead {
    /// Block size, where it is large enough to matter: vLLM's hybrid allocator
    /// scales the attention block to fit a mamba page (1056 to 2304 tokens).
    /// Per-token parts are allocated in whole blocks, prefix hits and recurrent
    /// checkpoints land on block boundaries, and prefill chunks align to them.
    /// `None` leaves block rounding out, as it is for ordinary 16-token blocks.
    pub block_tokens: Option<u32>,
    /// Bytes the engine takes out of the KV budget before sizing the pool
    /// (null block, persistent device buffers).
    pub fixed_device_bytes: u64,
}

impl KvCacheLayout {
    /// A layout with no parts yet and no engine overhead.
    pub fn new() -> Self {
        Self {
            parts: Vec::new(),
            engine: EngineOverhead::default(),
        }
    }

    pub fn with_part(mut self, part: KvCachePart) -> Self {
        self.parts.push(part);
        self
    }

    pub fn with_engine(mut self, engine: EngineOverhead) -> Self {
        self.engine = engine;
        self
    }

    /// The partition's layout when each of `ranks` ranks holds this one-rank
    /// layout whole: every byte count, padding and fixed buffer included, scales
    /// by `ranks`; block tokens and copies do not.
    pub fn held_on_each_rank(mut self, ranks: u64) -> Self {
        for part in &mut self.parts {
            part.bytes_per_entry *= ranks;
            if let Some(padded) = &mut part.overhead.padded_bytes_per_entry {
                *padded *= ranks;
            }
        }
        self.engine.fixed_device_bytes *= ranks;
        self
    }

    /// Bytes one more token of context adds once every sliding window is full:
    /// the per-token parts only. Used to size transfers and logs that speak in
    /// tokens.
    pub fn per_token_bytes(&self) -> u64 {
        self.parts
            .iter()
            .filter_map(|part| match part.growth {
                KvGrowth::PerToken { tokens_per_entry } => {
                    Some(part.bytes_per_entry / u64::from(tokens_per_entry.max(1)))
                }
                _ => None,
            })
            .sum()
    }

    /// Reject layouts L5 cannot account: no parts, a zero-byte part, a zero
    /// window or entry span, or an engine block of zero tokens.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.parts.is_empty(), "KV cache layout has no parts");
        for part in &self.parts {
            anyhow::ensure!(
                part.bytes_per_entry > 0,
                "KV cache part {} has zero bytes per entry",
                part.name
            );
            if let Some(padded) = part.overhead.padded_bytes_per_entry {
                anyhow::ensure!(
                    padded >= part.bytes_per_entry,
                    "KV cache part {} pads {} bytes down to {padded}",
                    part.name,
                    part.bytes_per_entry
                );
            }
            anyhow::ensure!(
                part.overhead.copies != Some(0),
                "KV cache part {} keeps zero copies",
                part.name
            );
            match part.growth {
                KvGrowth::PerToken { tokens_per_entry } => anyhow::ensure!(
                    tokens_per_entry > 0,
                    "KV cache part {} has zero tokens per entry",
                    part.name
                ),
                KvGrowth::SlidingWindow { window_tokens } => anyhow::ensure!(
                    window_tokens > 0,
                    "KV cache part {} has a zero-token window",
                    part.name
                ),
                KvGrowth::PerRequest => {}
            }
        }
        anyhow::ensure!(
            self.engine.block_tokens != Some(0),
            "KV cache layout has a zero-token engine block"
        );
        Ok(())
    }
}

impl Default for KvCacheLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl KvCachePart {
    /// Ordinary attention KV: one entry per token, kept for the whole request.
    pub fn per_token(name: &'static str, bytes_per_token: u64) -> Self {
        Self::compressed(name, 1, bytes_per_token)
    }

    /// One entry per `tokens_per_entry` tokens (a DSA kpool indexer, a
    /// DeepSeek-V4 compressed cache).
    pub fn compressed(name: &'static str, tokens_per_entry: u32, bytes_per_entry: u64) -> Self {
        Self {
            name,
            growth: KvGrowth::PerToken { tokens_per_entry },
            bytes_per_entry,
            prefix_reusable: true,
            overhead: PartOverhead::default(),
        }
    }

    /// The last `window_tokens` tokens only (sliding-window attention, a
    /// short-conv window).
    pub fn sliding_window(name: &'static str, window_tokens: u32, bytes_per_token: u64) -> Self {
        Self {
            name,
            growth: KvGrowth::SlidingWindow { window_tokens },
            bytes_per_entry: bytes_per_token,
            prefix_reusable: true,
            overhead: PartOverhead::default(),
        }
    }

    /// One state per request (GDN / KDA conv + SSM state, a circular buffer).
    pub fn per_request(name: &'static str, bytes_per_request: u64) -> Self {
        Self {
            name,
            growth: KvGrowth::PerRequest,
            bytes_per_entry: bytes_per_request,
            prefix_reusable: true,
            overhead: PartOverhead::default(),
        }
    }

    pub fn padded_to(mut self, padded_bytes_per_entry: u64) -> Self {
        self.overhead.padded_bytes_per_entry = Some(padded_bytes_per_entry);
        self
    }

    pub fn with_copies(mut self, copies: u32) -> Self {
        self.overhead.copies = Some(copies);
        self
    }

    pub fn not_prefix_reusable(mut self) -> Self {
        self.prefix_reusable = false;
        self
    }
}

/// Per-token bytes of `layers` GQA layers over one attention TP group of
/// `attn_tp` ranks. When `attn_tp` exceeds `num_kv_heads`, each rank still
/// stores one whole KV head, so the group stores `attn_tp` heads, not
/// `num_kv_heads`.
pub fn gqa_kv_bytes_per_token(
    layers: u64,
    num_kv_heads: u64,
    k_head_dim: u64,
    v_head_dim: u64,
    dtype_bytes: u64,
    attn_tp: u64,
) -> u64 {
    layers * num_kv_heads.max(attn_tp.max(1)) * (k_head_dim + v_head_dim) * dtype_bytes
}

/// vLLM's hybrid attention block (`_align_hybrid_block_size`): the smallest
/// multiple of the attention kernel's `kernel_block_tokens` whose per-layer
/// attention page (`attn_page_bytes_per_token` per token) holds one layer's
/// recurrent-state page. The state page is then padded up to that attention
/// page. Both operands are per layer and per rank.
pub fn hybrid_attention_block_tokens(
    kernel_block_tokens: u64,
    state_page_bytes: u64,
    attn_page_bytes_per_token: u64,
) -> u32 {
    let aligned_page = kernel_block_tokens * attn_page_bytes_per_token.max(1);
    let blocks = state_page_bytes.div_ceil(aligned_page).max(1);
    u32::try_from(kernel_block_tokens * blocks).expect("hybrid block size exceeds u32")
}

/// One request's cache in vLLM's hybrid allocator for a model of
/// `attn_layers` per-token attention layers and `state_layers` recurrent
/// layers (GDN): the attention KV per token, and one state page per
/// recurrent layer padded to the attention page of one hybrid block
/// ([`hybrid_attention_block_tokens`]). One null block of every attention
/// layer's page leaves the pool. With no recurrent layer it is plain
/// per-token attention.
pub fn hybrid_attention_state_layout(
    kernel_block_tokens: u64,
    attn_page_bytes_per_token: u64,
    attn_layers: u64,
    state_page_bytes: u64,
    state_layers: u64,
) -> KvCacheLayout {
    let kv_bytes_per_token = attn_page_bytes_per_token * attn_layers;
    let mut layout = KvCacheLayout::new();
    if kv_bytes_per_token > 0 {
        layout = layout.with_part(KvCachePart::per_token("gqa", kv_bytes_per_token));
    }
    if state_layers == 0 {
        return layout;
    }
    let block_tokens = hybrid_attention_block_tokens(
        kernel_block_tokens,
        state_page_bytes,
        attn_page_bytes_per_token,
    );
    let padded_state_page = u64::from(block_tokens) * attn_page_bytes_per_token;
    layout
        .with_part(
            KvCachePart::per_request("gdn", state_page_bytes * state_layers)
                .padded_to(padded_state_page * state_layers),
        )
        .with_engine(EngineOverhead {
            block_tokens: Some(block_tokens),
            fixed_device_bytes: u64::from(block_tokens) * kv_bytes_per_token,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gqa_bytes_count_replicated_heads_once_tp_exceeds_kv_heads() {
        // Llama-3-8B: 32 layers, 8 KV heads of 128, bf16.
        let tp8 = gqa_kv_bytes_per_token(32, 8, 128, 128, 2, 8);
        assert_eq!(tp8, 131_072);
        assert_eq!(gqa_kv_bytes_per_token(32, 8, 128, 128, 2, 1), tp8);
        // TP16 stores 16 heads: every rank holds one.
        assert_eq!(gqa_kv_bytes_per_token(32, 8, 128, 128, 2, 16), 2 * tp8);
    }

    #[test]
    fn a_layout_held_on_each_rank_scales_bytes_but_not_blocks_or_copies() {
        let one_rank = KvCacheLayout::new()
            .with_part(KvCachePart::per_token("mla", 576).padded_to(640))
            .with_part(KvCachePart::per_request("state", 1000).with_copies(2))
            .with_engine(EngineOverhead {
                block_tokens: Some(64),
                fixed_device_bytes: 4096,
            });
        let tp4 = one_rank.clone().held_on_each_rank(4);
        assert_eq!(tp4.parts[0].bytes_per_entry, 4 * 576);
        assert_eq!(tp4.parts[0].overhead.padded_bytes_per_entry, Some(4 * 640));
        assert_eq!(tp4.parts[1].bytes_per_entry, 4 * 1000);
        assert_eq!(tp4.parts[1].overhead.copies, Some(2));
        assert_eq!(tp4.engine.block_tokens, Some(64));
        assert_eq!(tp4.engine.fixed_device_bytes, 4 * 4096);
        assert_eq!(one_rank.clone().held_on_each_rank(1), one_rank);
    }

    #[test]
    fn per_token_bytes_sum_only_the_parts_that_grow() {
        let layout = KvCacheLayout::new()
            .with_part(KvCachePart::per_token("global", 1000))
            .with_part(KvCachePart::compressed("indexer", 4, 400))
            .with_part(KvCachePart::sliding_window("swa", 128, 7))
            .with_part(KvCachePart::per_request("state", 1 << 20));
        assert_eq!(layout.per_token_bytes(), 1100);
    }

    #[test]
    fn validate_rejects_parts_l5_cannot_account() {
        assert!(KvCacheLayout::new().validate().is_err());
        let zero_window = KvCacheLayout::new().with_part(KvCachePart::sliding_window("swa", 0, 8));
        assert!(zero_window.validate().is_err());
        let shrinking =
            KvCacheLayout::new().with_part(KvCachePart::per_token("kv", 8).padded_to(4));
        assert!(shrinking.validate().is_err());
        let ok = KvCacheLayout::new().with_part(KvCachePart::per_token("kv", 8));
        assert!(ok.validate().is_ok());
    }
}
