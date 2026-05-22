//! L4 ↔ L5 data contract: the per-iteration `ArchInput` a worker hands to a
//! model_arch, plus the iter-wise query trait. See L4 design.md §3.1 / §4.1.
//!
//! v1 scope is the **unified** worker (co-located attn + ffn, runs the whole
//! model per iteration). The AFD attn/ffn-split `AttnArchInput` / `FfnArchInput`
//! and the layer-wise traits are deferred.

use crate::timing::LookupResult;

/// One HP group's batch state. Under a `Local`/unified single-GPU deployment
/// there is exactly one group; the field set follows L4 §3.1 so the AFD/HP
/// generalization is additive later.
#[derive(Clone, Debug, Default)]
pub struct ArchGroupInput {
    /// Tokens this group processes this forward pass.
    pub batch_tokens: u32,
    /// Of those, the prefill portion (token count).
    pub prefill_tokens: u32,
    /// Of those, the decode portion (= `decode_kv_lens.len()`).
    pub decode_tokens: u32,
    /// Per prefill/chunked request: `(prefix_len, append_len)`.
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
    /// KV length per decode request (one `q = 1` token each).
    pub decode_kv_lens: Vec<u32>,
    /// Cumulative KV occupancy for this group (KV-cache pressure).
    pub total_kv_len: u32,
}

/// Unified worker's per-iteration input (attn view `groups` + ffn routing view
/// `tokens_per_source_rank`). Dense local has a single group and no routing.
#[derive(Clone, Debug, Default)]
pub struct UnifiedArchInput {
    pub groups: Vec<ArchGroupInput>,
    pub tokens_per_source_rank: Vec<u32>,
}

/// Iter-wise query face for a unified/TBO worker: one call costs the whole
/// iteration (embedding → layers → lm_head). `&UnifiedArchInput` is concrete on
/// the signature (no `dyn`); L5 binds via `<M: IterwiseUnifiedModel>` generic.
pub trait IterwiseUnifiedModel: Send + Sync + 'static {
    fn cost_whole_iter(&self, batch: &UnifiedArchInput) -> LookupResult;
}
