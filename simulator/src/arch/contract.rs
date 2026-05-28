//! L4 ↔ L5 data contract: the per-iteration `ArchInput` a worker hands to a
//! model_arch, plus the iter-wise query trait. See L4 design.md §3.1 / §4.1.
//!
//! v1 scope is the **unified** worker (co-located attn + ffn, runs the whole
//! model per iteration). The AFD attn/ffn-split `AttnArchInput` / `FfnArchInput`
//! and the layer-wise traits are deferred.

use crate::timing::{CostManifest, LeafMetrics, SlotInput};

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

impl ArchGroupInput {
    /// Reset to an empty group, retaining `Vec` capacity. Lets a worker refill a
    /// held `UnifiedArchInput` in place each iteration instead of allocating a
    /// fresh group + growing `decode_kv_lens` from zero every forward pass.
    pub fn clear(&mut self) {
        self.batch_tokens = 0;
        self.prefill_tokens = 0;
        self.decode_tokens = 0;
        self.total_kv_len = 0;
        self.prefill_chunk_pairs.clear();
        self.decode_kv_lens.clear();
    }
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
    ///
    /// `scratch` is a second caller-owned buffer the CostTree aggregation reuses
    /// for its per-node rollup (one `LeafMetrics` per flat node); threading it in
    /// keeps the per-iter aggregate walk allocation-free. Callers reuse the same
    /// `scratch` across iterations; its contents are not meaningful on return.
    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics;

    /// Like [`Self::eval_iter`], but also captures each leaf's typed kernel input
    /// into `inputs` (slot-aligned, in visit order) for the `cost_log`
    /// `slot_input` column. The default clears `inputs` and falls back to the
    /// non-capturing path — models with a compiled CostTree override it to record.
    /// The clone per leaf is paid only on this path; `eval_iter` stays untouched.
    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        inputs.clear();
        self.eval_iter(batch, slots, scratch)
    }

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

    /// GPUs one replica of this model spans. The model_arch is the source of truth
    /// for this: it resolved the parallel layout, so it knows the real extent — the
    /// EP span with TP/HP groups nested inside it, *not* a `tp×ep×hp` product. L5/L6
    /// only read it (to size the run's GPU inventory); they never derive it. A dense
    /// local arch returns 1.
    fn gpus_per_replica(&self) -> u16;

    /// Number of independent attention DP shards (HP groups) the worker must
    /// maintain — one `Batch` per shard, each seeing a different slice of the
    /// batch (L4 §3.3 fan-out). Iter-wise archs with a single attention TP group
    /// return 1 (the default); a DP-attention arch returns `ffn_tp / attn_tp`.
    fn num_attn_dp_groups(&self) -> u16 {
        1
    }
}
