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
    /// To size one worker KV partition for an attention shard, the worker reads
    /// `attn_kv_bytes` (per-GPU budget), multiplies by `num_attn_shards()` (the
    /// GPUs in one attn shard set), and divides by this total to get capacity
    /// in tokens. To compute a PD transfer size, the prefill worker multiplies
    /// tokens by this to get total wire bytes; the cluster then divides by
    /// link count to recover per-rank bytes.
    fn total_kv_bytes_per_token(&self) -> u64;

    /// **Total** recurrent-state bytes one *request* occupies — summed over
    /// **all** recurrent (SSM / linear-attention) layers, **all** heads, **all**
    /// attention ranks, and including both the SSM state and the causal-conv
    /// window (neither alone can resume a sequence). Same "total, not per-GPU"
    /// convention as [`Self::total_kv_bytes_per_token`], so a worker converts it
    /// to that function's token unit by dividing.
    ///
    /// Unlike KV, this is **fixed per request**: it does not grow as the request
    /// decodes. A pure full-attention model has no such state and returns 0
    /// (the default), which is what keeps every dense arch untouched.
    fn recurrent_state_bytes_per_request(&self) -> u64 {
        0
    }

    /// Token interval at which a checkpoint of [`Self::recurrent_state_bytes_per_request`]
    /// can be taken and later resumed from — vLLM's hybrid `block_size`.
    ///
    /// A recurrent state is a single rolling snapshot, not per-token entries, so
    /// it is only reusable at positions where a snapshot was actually written.
    /// vLLM writes them at multiples of this interval (`mamba_cache_mode`), which
    /// quantizes every prefix-cache hit to a multiple of it. `0` means the model
    /// has no recurrent state; `1` would mean "resumable anywhere", which no real
    /// recurrent layer is.
    ///
    /// The arch owns the derivation because only it knows the state and KV page
    /// shapes; L5 only reads the value (and may override it from a preset).
    fn recurrent_checkpoint_interval_tokens(&self) -> u32 {
        0
    }

    /// GPUs one replica of this model spans. The model_arch is the source of truth
    /// for this: it resolved the parallel layout, so it knows the real extent — the
    /// EP span with TP/HP groups nested inside it, *not* a `tp×ep×hp` product. L5/L6
    /// only read it (to size the run's GPU inventory); they never derive it. A dense
    /// local arch returns 1.
    fn gpus_per_replica(&self) -> u16;

    /// Number of independent attention DP shards (HP groups) the worker must
    /// maintain — one KV partition state per shard, each seeing a different slice
    /// of the iteration batch (L4 §3.3 fan-out). Iter-wise archs with a single
    /// attention TP group return 1 (the default); a DP-attention arch returns
    /// `ffn_tp / attn_tp`.
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

// ── speculative iter-wise contract ───────────────────────────────────────────
//
// A speculative iteration is not an ordinary iteration with more tokens. One
// decode request presents `k + 1` query rows to the target verify pass and one
// row to each of the `k` draft passes, so "decode rows" and "decode requests"
// stop being the same number and the lm-head row count stops following either.
//
// `ArchGroupInput::request_count` documents the rule this obeys: an input mode
// with different logits semantics gets its own ArchInput type rather than an
// optional axis on the ordinary one. The same rule is applied one level up —
// the capability is a separate trait, so an ordinary recipe fails its type
// bound instead of selecting logits semantics through a runtime role branch,
// and a speculative model compiles its own CostTree instead of reusing the
// ordinary one with dead slots.

/// One decode request in a speculative verify batch.
///
/// Unlike [`ArchGroupInput::decode_kv_lens`], `query_len` is explicit because a
/// verify step may present more than one query row for one request. Keeping the
/// pair together prevents callers from recovering request count from the
/// query-row count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpeculativeDecodeInput {
    /// Timing context visible to the final query row of this request's verify
    /// group. A full-width verify at the model-length boundary clamps this to
    /// the model limit; durable KV advancement still follows accepted tokens.
    pub kv_len: u32,
    /// Query rows emitted for this request in the current verify group.
    pub query_len: u32,
}

/// One attention/FFN group view for a speculative iter-wise batch.
#[derive(Clone, Debug, Default)]
pub struct SpeculativeArchGroupInput {
    /// Tokens this group processes this forward pass.
    pub batch_tokens: u32,
    /// Of those, the prefill portion (token count).
    pub prefill_tokens: u32,
    /// Of those, speculative decode query rows — **not** decode request count.
    pub decode_tokens: u32,
    /// Per prefill/chunked request: `(prefix_len, append_len)`.
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
    /// Per speculative decode request: final KV length plus verify query rows.
    pub decode_requests: Vec<SpeculativeDecodeInput>,
    /// Cumulative resident KV before this verify pass (KV-cache pressure).
    /// `decode_requests[*].kv_len` is the final verify-row length instead.
    pub total_kv_len: u32,
}

impl SpeculativeArchGroupInput {
    /// Requests represented by this group, counting each decode request once
    /// however many verify rows it presents.
    pub fn request_count(&self) -> u32 {
        u32::try_from(self.prefill_chunk_pairs.len() + self.decode_requests.len())
            .expect("SpeculativeArchGroupInput request count must fit u32")
    }

    /// Reset to an empty group, retaining `Vec` capacity, so a worker can refill
    /// a held input in place each iteration.
    pub fn clear(&mut self) {
        self.batch_tokens = 0;
        self.prefill_tokens = 0;
        self.decode_tokens = 0;
        self.total_kv_len = 0;
        self.prefill_chunk_pairs.clear();
        self.decode_requests.clear();
    }
}

/// Speculative worker's per-iteration input. Deliberately separate from
/// [`UnifiedArchInput`]: decode query rows and decode request cardinality are
/// different quantities under draft/verify execution.
#[derive(Clone, Debug, Default)]
pub struct SpeculativeArchInput {
    /// Candidate positions drafted per request this iteration.
    ///
    /// This is **not** the number of draft forward passes, and the pass count is
    /// not derivable from it: a recurrent drafter (MTP, EAGLE) runs one pass per
    /// candidate, while a block-parallel drafter produces a whole block in one
    /// pass. The pass count belongs to whatever the model composes as its draft
    /// stage; this is only the candidate budget the verify pass must cover.
    ///
    /// It is also not the verify width. For a chain the width is `k + 1`, but a
    /// tree submits one row per tree node while advancing at most its depth, so
    /// per-request rows live in [`SpeculativeDecodeInput::query_len`] instead.
    /// Models validate this against the width they were built for; it selects a
    /// profiled kernel shape and therefore cannot vary per iteration.
    pub draft_tokens: u32,
    pub groups: Vec<SpeculativeArchGroupInput>,
    pub tokens_per_source_rank: Vec<u32>,
}

/// Iter-wise query face for a speculative co-located worker.
///
/// Mirrors [`IterwiseUnifiedModel`]'s reusable-buffer protocol but accepts only
/// [`SpeculativeArchInput`]. A model implements this **instead of**, not in
/// addition to, the ordinary trait: the two walk different compiled trees, so a
/// type that offered both would be two models wearing one name.
pub trait SpeculativeUnifiedModel: Send + Sync + 'static {
    /// Per-iter cost of one whole speculative iteration: the target verify pass
    /// over `k + 1` rows per decode request, then the `k` draft passes. Same
    /// `(slots, scratch) -> LeafMetrics` protocol as
    /// [`IterwiseUnifiedModel::eval_iter`].
    fn eval_speculative_iter(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics;

    /// Like [`Self::eval_speculative_iter`], but also captures each leaf's typed
    /// kernel input for the `cost_log` `slot_input` column.
    fn eval_speculative_iter_with_inputs(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        inputs.clear();
        self.eval_speculative_iter(batch, slots, scratch)
    }

    /// The `cost_log` manifest for this model's own compiled tree. A speculative
    /// model's manifest is not the ordinary model's: the draft passes are real
    /// slots, so the two are structurally distinguishable in a cost-log row.
    fn cost_log_manifest(&self) -> CostManifest {
        CostManifest {
            slots: Vec::new(),
            nodes: Vec::new(),
            node_labels: Vec::new(),
        }
    }

    /// See [`IterwiseUnifiedModel::total_kv_bytes_per_token`].
    fn total_kv_bytes_per_token(&self) -> u64;

    /// Maximum context accepted by the model's timing kernels. A verify group
    /// reaching this bound clamps its final row rather than widening the shape.
    fn max_model_len(&self) -> u32;

    /// See [`IterwiseUnifiedModel::gpus_per_replica`].
    fn gpus_per_replica(&self) -> u16;

    /// See [`IterwiseUnifiedModel::num_attn_dp_groups`].
    fn num_attn_dp_groups(&self) -> u16 {
        1
    }

    /// See [`IterwiseUnifiedModel::num_attn_shards`].
    fn num_attn_shards(&self) -> u16 {
        (self.gpus_per_replica() / self.num_attn_dp_groups().max(1)).max(1)
    }
}

#[cfg(test)]
mod speculative_arch_input_tests {
    use super::{SpeculativeArchGroupInput, SpeculativeDecodeInput};

    #[test]
    fn request_count_counts_requests_not_verify_rows() {
        // The defect this catches: recovering request count from `decode_tokens`
        // or from a flat KV-length vector. Two decode requests present twelve
        // query rows at k=5, and the group holds three requests, not thirteen.
        let mut group = SpeculativeArchGroupInput {
            batch_tokens: 15,
            prefill_tokens: 3,
            decode_tokens: 12,
            prefill_chunk_pairs: vec![(0, 3)],
            decode_requests: vec![
                SpeculativeDecodeInput {
                    kv_len: 64,
                    query_len: 6,
                },
                SpeculativeDecodeInput {
                    kv_len: 91,
                    query_len: 6,
                },
            ],
            total_kv_len: 155,
        };

        assert_eq!(group.request_count(), 3);

        group.clear();
        assert_eq!(group.request_count(), 0);
        assert_eq!(group.batch_tokens, 0);
        assert_eq!(group.decode_tokens, 0);
        assert!(group.decode_requests.capacity() >= 2);
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

    /// Independent attention DP shards (one KV partition state per shard). A
    /// single attn-TP group returns 1 (default); a DP-attention arch returns
    /// `ep_size / attn_tp`.
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
    /// ranks). Sizes the attention worker's KV partition capacity (same
    /// definition as the iter-wise `total_kv_bytes_per_token`).
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
