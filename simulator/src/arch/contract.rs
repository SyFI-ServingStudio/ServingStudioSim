//! L4 ↔ L5 data contract: the per-iteration `ArchInput` a worker hands to a
//! model_arch, plus the iter-wise query trait. See L4 design.md §3.1 / §4.1.
//!
//! v1 scope is the **unified** worker (co-located attn + ffn, runs the whole
//! model per iteration). The AFD attn/ffn-split `AttnArchInput` / `FfnArchInput`
//! and the layer-wise traits are deferred.

use crate::timing::{CostManifest, LeafMetrics};

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
    /// Per-iter cost of the whole iteration (embedding → layers → lm_head) via the
    /// compiled CostTree path: stream each leaf's [`LeafMetrics`] into `slots`
    /// (the caller's reused buffer — cleared + refilled to the manifest length),
    /// then aggregate the cached structure (the homogeneous-layer fold supplies
    /// `×num_layers`). Returns the aggregate: `.m.time_ms` is the per-iter sim
    /// clock, the rest is rolled-up flops/bytes/energy + coverage. `slots` is left
    /// holding the per-leaf breakdown so a `cost_log` row can carry it — callers
    /// that only want the clock just ignore the buffer (filling it is free: the
    /// eval pass materializes it either way). Models with no compiled CostTree
    /// clear `slots` and return their fixed aggregate. O(slots) flat writes, no
    /// per-tick allocation when the caller reuses the buffer.
    fn eval_iter(&self, batch: &UnifiedArchInput, slots: &mut Vec<LeafMetrics>) -> LeafMetrics;

    /// The `cost_log` manifest: the ordered slots plus the flattened aggregation
    /// nodes, so a consumer can reproduce `total_time_ms` from a row's per-slot
    /// breakdown (the `Scale{n}` fold / `Sum` / `Max` operators, not just names).
    /// Empty (default) means the model has no compiled CostTree; models with one
    /// override it.
    fn cost_log_manifest(&self) -> CostManifest {
        CostManifest {
            slots: Vec::new(),
            nodes: Vec::new(),
            node_labels: Vec::new(),
        }
    }

    /// KV-cache bytes one token occupies across the whole model. The worker
    /// divides its memory allowance by this to size its `KvPool` (L5 owns the
    /// division; arch owns this footprint).
    fn kv_bytes_per_token(&self) -> u64;
}
