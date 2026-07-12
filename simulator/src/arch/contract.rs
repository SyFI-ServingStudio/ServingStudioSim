//! L4 ↔ L5 data contract: the per-iteration `ArchInput` a worker hands to a
//! model_arch, plus the iter-wise query trait. See L4 design.md §3.1 / §4.1.
//!
//! The iter-wise contract ([`IterwiseUnifiedModel`]) is used by co-located
//! attn+ffn workers: barebone/unified, HP unified, and the PD prefill/decode
//! worker pair. The layer-wise AFD contract ([`AttnLayerwiseModel`] /
//! [`FfnLayerwiseModel`] with [`AttnArchInput`] / [`FfnArchInput`]) splits a model
//! at the per-layer attn/ffn boundary so the two disaggregated worker pools can
//! interleave attn-of-layer-N with ffn-of-layer-(N-1).

use crate::timing::{CostManifest, CostManifestDoc, LeafMetrics, SlotInput};

/// One attention/FFN group view for an iter-wise batch. Local/unified and PD
/// prefill/decode workers usually pass one group; HP/DP-attn variants may pass
/// multiple groups.
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
    /// Requests represented by this group's current non-speculative iteration.
    ///
    /// Each listed prefill chunk and decode KV length owns one request and one
    /// lm-head row. Input modes with different logits semantics must use a
    /// different ArchInput type instead of adding optional axes here.
    pub fn request_count(&self) -> u32 {
        u32::try_from(self.prefill_chunk_pairs.len() + self.decode_kv_lens.len())
            .expect("ArchGroupInput request count must fit u32")
    }

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

#[cfg(test)]
mod arch_group_input_tests {
    use super::ArchGroupInput;

    #[test]
    fn request_count_counts_prefill_and_decode_requests_not_tokens() {
        let group = ArchGroupInput {
            batch_tokens: 2048,
            prefill_tokens: 2047,
            decode_tokens: 1,
            prefill_chunk_pairs: vec![(0, 512), (0, 512), (0, 512), (0, 511)],
            decode_kv_lens: vec![512],
            total_kv_len: 2560,
        };

        assert_eq!(group.request_count(), 5);
    }
}

/// Unified worker's per-iteration input (attn view `groups` + ffn routing view
/// `tokens_per_source_rank`). Dense local has a single group and no routing.
#[derive(Clone, Debug, Default)]
pub struct UnifiedArchInput {
    pub groups: Vec<ArchGroupInput>,
    pub tokens_per_source_rank: Vec<u32>,
}

/// Iter-wise query face for co-located workers: one call costs the whole
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

    /// **Total** KV-cache bytes one token occupies — summed over **all** layers,
    /// **all** KV heads, **all** attention ranks. This is the wire size of a
    /// token's KV (what a PD handoff transfers across the comm group); it is
    /// *not* per-GPU.
    ///
    /// To size a worker's `KvPool` for one attn shard, the worker reads
    /// `attn_kv_bytes` (per-GPU budget), multiplies by `num_attn_shards()` (the
    /// GPUs in one attn shard set), and divides by this total to get capacity
    /// in tokens. To compute a PD transfer size, the prefill worker multiplies
    /// tokens by this to get total wire bytes; the cluster then divides by
    /// link count to recover per-rank bytes.
    fn total_kv_bytes_per_token(&self) -> u64;

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

    /// GPUs one attention shard (HP group) of this model spans — its attn-TP /
    /// head-parallel degree. This is the count of physical send/recv links a KV
    /// transfer rides on at this side of a PD handoff. Derived (`gpus_per_replica /
    /// num_attn_dp_groups`), so the arch owns it and L5/L6 only read — neither the
    /// prefill worker nor the PD flow re-derives the formula.
    fn num_attn_shards(&self) -> u16 {
        (self.gpus_per_replica() / self.num_attn_dp_groups().max(1)).max(1)
    }
}

// ── layer-wise contract (AFD) ────────────────────────────────────────────────
//
// AFD (attention-FFN disaggregation) splits a model at the per-layer attn/ffn
// boundary into two independent worker pools: the attn pool computes only
// attention; the ffn pool computes qkv / o_proj / router / MoE. Each side is a
// separate model_arch implementing one of the two traits below. Unlike the
// iter-wise contract (one `eval_iter` over the whole iteration), these expose
// *per-layer* cost so the orchestrator can interleave attn-of-layer-N with
// ffn-of-layer-(N-1). See L4 design.md §4.1 (AFD-style trait formalization) + §7.
//
// Both traits use the same `(slots, scratch) -> LeafMetrics` compiled-CostTree
// protocol as `IterwiseUnifiedModel::eval_iter`: each cost method compiles its own
// small CostTree at build, then per call fills `slots` (cleared + resized to that
// group's slot count) and aggregates. `cost_log_manifest` exposes those per-section
// trees as a [`CostManifestDoc`] (one named section per cost group), so the
// section-aware `cost_log` writer can name each layer-wise row's slots. The
// `slot_input` capture (`*_with_inputs`) mirrors the iter-wise
// `eval_iter_with_inputs` shape: each section method has a sibling that also
// records per-leaf inputs (default clears + falls back to the non-capturing path).

/// Attn-side per-iteration input: one [`ArchGroupInput`] per attention DP shard.
/// The attn worker fans its attention out over these groups (`max` over shards).
/// This is the `groups` half of [`UnifiedArchInput`], with no ffn routing view.
#[derive(Clone, Debug, Default)]
pub struct AttnArchInput {
    pub groups: Vec<ArchGroupInput>,
}

/// Ffn-side per-iteration input. The ffn cost depends ONLY on per-DP-shard token
/// counts — qkv / o_proj fan out per shard (`Max`-ed across shards), and their
/// pooled total drives router / MoE / lm_head. So this carries just the counts,
/// NOT the attention-shaped [`ArchGroupInput`] (prefill chunks / decode KV lengths)
/// the attn side needs — the ffn never reads those. Routing distribution is also
/// not here: it is a build-time `RoutingDistribution` baked into the model at
/// `build_configs` (see `qwen3_moe_dp_attn_ep_ffn`).
///
/// `Deserialize` so this type doubles as the offline `timing-predict` ffn case:
/// the case→input lowering is the identity (the raw token counts ARE the input),
/// with no shorthand to expand. That is deliberate — the ffn case must NOT be
/// forced through the attention-shaped predict case the iter/attn sides share;
/// each arch owns the shape its cost actually reads. `deny_unknown_fields` keeps a
/// predicted ffn case from carrying stray attention vocabulary it would silently
/// drop.
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FfnArchInput {
    /// Tokens processed by each attention DP shard this forward pass (length =
    /// `num_dp_groups`). `Max`-fanned across shards for qkv / o_proj / home-reduce;
    /// summed for the router / MoE / lm_head total.
    pub tokens_per_group: Vec<u32>,
}

/// Attn-side layer-wise query face (AFD attn worker). Attention is the only
/// per-layer group on this side, so a single cost method. `&AttnArchInput` is
/// concrete (no `dyn`); L5 binds via `<M: AttnLayerwiseModel>`.
pub trait AttnLayerwiseModel: Send + Sync + 'static {
    fn num_layers(&self) -> u32;

    /// Independent attention DP shards (one `Batch` per shard). A single attn-TP
    /// group returns 1 (default); a DP-attention arch returns `ep_size / attn_tp`.
    fn num_attn_dp_groups(&self) -> u16 {
        1
    }

    /// GPUs one replica of the attn side spans (`attn_tp_size × num_attn_dp_groups`).
    fn gpus_per_replica(&self) -> u16;

    /// GPUs one attention shard spans (its attn-TP degree). Derived; L5/L6 only read.
    fn num_attn_shards(&self) -> u16 {
        (self.gpus_per_replica() / self.num_attn_dp_groups().max(1)).max(1)
    }

    /// **Total** KV-cache bytes one token occupies (all layers / KV heads / attn
    /// ranks). Sizes the attn worker's `KvPool` (same definition as the iter-wise
    /// `total_kv_bytes_per_token`).
    fn total_kv_bytes_per_token(&self) -> u64;

    /// Bytes the attn side emits per token to the ffn side after a layer's
    /// attention (the attention output, `q_dim · bpe`). The attn worker attaches
    /// `this × tokens` to its handoff; the ffn receiver reads it off the message —
    /// it never recomputes the size. Only the *outgoing* direction lives here.
    fn attn_to_ffn_bytes_per_token(&self) -> u64;

    /// Per-layer attention cost over the compiled CostTree: stream each leaf's
    /// [`LeafMetrics`] into `slots` (cleared + resized to this group's slot count),
    /// then aggregate. `.m.time_ms` is the attention compute time for `layer_idx`.
    /// Same buffer protocol as [`IterwiseUnifiedModel::eval_iter`].
    fn attn_cost(
        &self,
        layer_idx: usize,
        batch: &AttnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics;

    /// Like [`Self::attn_cost`], but also captures each leaf's typed kernel input
    /// into `inputs` (slot-aligned, in visit order) for the `cost_log` `slot_input`
    /// column. The default clears `inputs` and falls back to the non-capturing path;
    /// the concrete model overrides it. Mirrors [`IterwiseUnifiedModel::eval_iter_with_inputs`].
    fn attn_cost_with_inputs(
        &self,
        layer_idx: usize,
        batch: &AttnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        inputs.clear();
        self.attn_cost(layer_idx, batch, slots, scratch)
    }

    /// The `cost_log` manifest: the attn side has one cost group, so a single
    /// `attn` section naming this shard's attention CostTree slots. Empty (default)
    /// means no compiled CostTree; the concrete model overrides it.
    fn cost_log_manifest(&self) -> CostManifestDoc {
        CostManifestDoc::empty()
    }
}

/// Ffn-side layer-wise query face (AFD ffn worker). Per layer there are two
/// groups — pre-attn (`qkv`) and post-attn (`o_proj` + router + MoE) — plus an
/// iteration prologue (embedding) and epilogue (final_norm + lm_head).
///
/// The Bridge / Bootstrap / Terminal fused-kernel split (L4 §4.1): the ffn side's
/// physical mid-layer kernel fuses post-attn-of-L with pre-attn-of-(L+1). The
/// split convention impls follow:
///   - `pre_attn_cost(0)`       = real qkv cost (layer-0 Bootstrap);
///   - `pre_attn_cost(L > 0)`   = [`LeafMetrics::ZERO`] (the fused Bridge bills
///                                pre(L+1) inside `post_attn_cost(L)`);
///   - `post_attn_cost(L < last)` bills post(L) **plus** the fused pre(L+1);
///   - `post_attn_cost(last)`     is post-only (Terminal).
///
/// So `Σ_L pre_attn + Σ_L post_attn` totals exactly one qkv + one o_proj + one MoE
/// per layer — the L4 §4.1 M_form_consistency the cost-consistency test guards.
pub trait FfnLayerwiseModel: Send + Sync + 'static {
    fn num_layers(&self) -> u32;

    /// GPUs one replica of the ffn side spans (`ep_size`).
    fn gpus_per_replica(&self) -> u16;

    /// Number of attention DP shards the ffn side pools. NOT a configurable knob:
    /// only `attn_tp_size` + `ep_size` are configured, and `build_configs` derives
    /// this as `ep_size / attn_tp_size` (with the `ep_size % attn_tp_size == 0`
    /// assert) and caches it — this accessor just exposes the resolved value. The
    /// ffn worker partitions its workload across this many groups before costing
    /// (post_norm + router + MoE home-reduce run replicated per shard on its own
    /// token slice — the `Max` fan-out that models DP load imbalance). L5 (the
    /// worker) owns the partition; L6 only hands over the total workload.
    fn num_dp_groups(&self) -> u16;

    /// Bytes the ffn side emits per token to the attn side (the QKV projection
    /// output, `(q_dim + 2·kv_dim) · bpe`). The ffn worker attaches `this × tokens`
    /// to its handoff; the attn receiver reads it off the message, never recomputes.
    /// Only the *outgoing* direction lives here.
    fn ffn_to_attn_bytes_per_token(&self) -> u64;

    /// Pre-attention (qkv) cost for `layer_idx`. Per the split convention, layer 0
    /// returns the real qkv cost and layers > 0 return [`LeafMetrics::ZERO`] (the
    /// fused pre(L+1) is billed inside `post_attn_cost(L)`).
    fn pre_attn_cost(
        &self,
        layer_idx: usize,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics;

    /// Post-attention cost for `layer_idx` (o_proj + post_norm + router + MoE);
    /// mid-layers additionally bill the fused pre-attn of `layer_idx + 1`, the last
    /// layer is post-only.
    fn post_attn_cost(
        &self,
        layer_idx: usize,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics;

    /// Once-per-iteration prologue before the layer loop (embedding).
    fn prologue_cost(
        &self,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics;

    /// Once-per-iteration epilogue after the layer loop (final_norm + lm_head).
    fn epilogue_cost(
        &self,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics;

    // ── `*_with_inputs` siblings ──────────────────────────────────────────────
    // Each mirrors the matching cost method but also records per-leaf inputs into
    // `inputs` (slot-aligned) for the `cost_log` `slot_input` column. The defaults
    // clear `inputs` and fall back to the non-capturing path; the concrete model
    // overrides them. Same shape as [`IterwiseUnifiedModel::eval_iter_with_inputs`].

    fn pre_attn_cost_with_inputs(
        &self,
        layer_idx: usize,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        inputs.clear();
        self.pre_attn_cost(layer_idx, batch, slots, scratch)
    }

    fn post_attn_cost_with_inputs(
        &self,
        layer_idx: usize,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        inputs.clear();
        self.post_attn_cost(layer_idx, batch, slots, scratch)
    }

    fn prologue_cost_with_inputs(
        &self,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        inputs.clear();
        self.prologue_cost(batch, slots, scratch)
    }

    fn epilogue_cost_with_inputs(
        &self,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        inputs.clear();
        self.epilogue_cost(batch, slots, scratch)
    }

    /// The `cost_log` manifest: the ffn side has several distinct cost groups
    /// (different CostTrees / slot sets), so one section each — `prologue`,
    /// `pre_attn`, `post_attn`, `epilogue`. A `cost_log` row's `section` field
    /// selects which section names its slots. Empty (default) means no compiled
    /// CostTree; the concrete model overrides it.
    fn cost_log_manifest(&self) -> CostManifestDoc {
        CostManifestDoc::empty()
    }
}
