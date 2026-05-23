//! L4 ↔ L5 data contract: the per-iteration `ArchInput` a worker hands to a
//! model_arch, plus the iter-wise query trait. See L4 design.md §3.1 / §4.1.
//!
//! v1 scope is the **unified** worker (co-located attn + ffn, runs the whole
//! model per iteration). The AFD attn/ffn-split `AttnArchInput` / `FfnArchInput`
//! and the layer-wise traits are deferred.

use crate::common::Time;
use crate::timing::{CoverageFlags, LeafMetrics, LookupResult, Metrics4};

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

    /// Wallclock-only cost of the whole iteration, for the per-iter sim clock.
    /// The default builds the full `LookupResult` and discards everything but
    /// `time`; models with a homogeneous layer stack override this to skip the
    /// per-iter tree/`Vec`/`Arc` allocation (the dominant tick-loop cost).
    fn cost_whole_iter_time(&self, batch: &UnifiedArchInput) -> Time {
        self.cost_whole_iter(batch).time
    }

    /// Per-iter cost via the compiled CostTree path: stream each leaf's
    /// [`LeafMetrics`] into a flat buffer, then aggregate the cached structure
    /// (the homogeneous-layer fold supplies `×num_layers`). Full metrics +
    /// coverage like `cost_whole_iter`, but with O(slots) flat writes instead of
    /// an O(nodes) `LookupResult` tree of `Arc<str>` names + `Vec` breakdowns.
    /// The default narrows `cost_whole_iter`; models with a compiled CostTree
    /// override it.
    fn cost_whole_iter_metrics(&self, batch: &UnifiedArchInput) -> LeafMetrics {
        let r = self.cost_whole_iter(batch);
        let mut coverage = CoverageFlags::EMPTY;
        for w in &r.warnings {
            coverage |= CoverageFlags::from_kind(w.kind);
        }
        LeafMetrics {
            m: Metrics4 {
                time_ms: r.time.as_ms() as f32,
                flops: r.flops as f32,
                bytes: r.bytes as f32,
                energy_j: r.energy_j as f32,
            },
            coverage,
        }
    }

    /// CostTree leaf-slot names, in slot order — the `cost_log` manifest. Empty
    /// (default) means the model has no compiled CostTree, so per-slot cost
    /// logging is unavailable; models with one override it.
    fn cost_log_manifest(&self) -> Vec<String> {
        Vec::new()
    }

    /// Like [`Self::cost_whole_iter_metrics`], but also writes the per-slot
    /// [`LeafMetrics`] into `slots` (cleared + refilled to the manifest length),
    /// so a `cost_log` row can carry the breakdown. The default leaves `slots`
    /// empty and just returns the aggregate; models with a compiled CostTree
    /// override it to fill the per-leaf buffer in one eval pass.
    fn cost_whole_iter_with_slots(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        slots.clear();
        self.cost_whole_iter_metrics(batch)
    }

    /// KV-cache bytes one token occupies across the whole model. The worker
    /// divides its memory allowance by this to size its `KvPool` (L5 owns the
    /// division; arch owns this footprint).
    fn kv_bytes_per_token(&self) -> u64;
}
