//! IterModelExecution axis — execution contract + ArchInput builder (interfaces doc §4).
//!
//! Owns the model-specific `Input`, cost evaluation, and KV layout knowledge.
//! `build_iteration_input` is the ArchInput builder (doc §8 resolution: builder lives on the
//! model side; the shell only supplies the partition/grouping and never reads
//! `Input` fields).
//!
//! `IterModelExecution<K>` takes the KV type as a trait PARAMETER (not a method-generic
//! `build_iteration_input<K>`), symmetric with `IterAdmission<K>`. This lets an execution impl escalate
//! the KV view it reads to a capability sub-trait in its `impl` block — e.g. a
//! tier-aware execution
//! `impl<M, K: IterWorkerKv + TieredKvView> IterModelExecution<K> for TierAwareIterExecution<M>`
//! reads `resident_by_tier` to charge offload, and a multimodal execution could bound
//! `K: HybridKvView` to read per-modality state. A method-generic
//! `build_iteration_input<K: IterWorkerKv>` could NOT add a stronger bound than the
//! trait declares, so KV capabilities reached `IterAdmission` but not the cost model —
//! this closes that asymmetry (matrix §15 Root ②).
//! The common execs (`UnifiedIterExecution`/`MultiModelIterExecution`) still impl for ALL
//! `K: IterWorkerKv`,
//! so they compose with every KV; only capability-hungry execs restrict K.

use crate::common::{SharedRequests, Time};
use crate::worker::types::FfnTaskKind;

use super::kv::{IterWorkerKv, SlotPipelineKv};
use super::shared::advance_scope::AdvanceScope;

mod attention_layer;
mod draft_verify;
mod ffn_task;
mod multi_model_iter;
mod tier_aware_iter;
mod unified_iter;
pub use attention_layer::AttentionLayerExecutionAdapter;
#[allow(unused_imports)] // Public input type required by external DraftVerifyModel impls.
pub use draft_verify::{
    AcceptanceOracle, DraftVerifyExecution, DraftVerifyInput, DraftVerifyModel,
    DraftVerifyModelExecution, DraftVerifyResult, SpeculativeProposal,
};
pub use ffn_task::FfnSectionExecutionAdapter;
pub use multi_model_iter::MultiModelIterExecution;
pub use tier_aware_iter::TierAwareIterExecution;
pub use unified_iter::UnifiedIterExecution;

/// Per-model KV sizing facts the shell feeds to KV construction.
pub struct ModelKvLayout {
    pub total_kv_bytes_per_token: u64,
    pub num_attn_shards: u16,
}

pub trait IterModelExecution<K: IterWorkerKv> {
    type Input: Default;

    fn model_kv_layout(&self) -> ModelKvLayout;

    /// Build the WHOLE iteration's input — one arch group per KV partition (DP
    /// shard). Loops `0..kv_store.num_partitions()`, so barebone (1 partition → 1 group)
    /// and HP/DP (N partitions → N groups) use the SAME builder; the arch's `Max`
    /// fan-out over the groups supplies the DP wallclock. No partition arg: the iter
    /// input is the whole batch, unlike AFD's per-slot
    /// `AttentionLayerExecution::build_slot_input`. `K` is a trait param, so an impl
    /// may require a capability view (e.g. `TieredKvView`).
    fn build_iteration_input(&self, kv_store: &K, requests: &SharedRequests, out: &mut Self::Input);

    fn evaluate_iteration(&mut self, input: &Self::Input, iter: u64, now: Time) -> Time;
}

/// AFD-attn family execution (interfaces doc §4, per-family surface). Layer-wise
/// (one `evaluate_attention_layer` per attention layer) rather than whole-iteration,
/// and its `build_slot_input` takes a request `AdvanceScope` (the slot's members) +
/// `SlotPipelineKv` instead of a partition + `IterWorkerKv`: the SHELL owns which requests are in a
/// slot, KV only answers per-request `current_kv`. The attn→ffn handoff byte size
/// is a model fact the shell attaches to its `AttnLayerOutputsReady` event.
pub trait AttentionLayerExecution {
    type Input: Default;

    fn num_layers(&self) -> u16;
    fn model_kv_layout(&self) -> ModelKvLayout;
    fn attn_to_ffn_bytes_per_token(&self) -> u64;

    /// Build a slot's attention input from its request `AdvanceScope` + KV facts;
    /// returns the batch's query-token count (the shell needs it for the attn→ffn
    /// handoff byte size, and `Input` is opaque to the shell).
    fn build_slot_input<K: SlotPipelineKv>(
        &self,
        grouping: AdvanceScope<'_>,
        kv_store: &K,
        requests: &SharedRequests,
        out: &mut Self::Input,
    ) -> u64;

    /// Evaluate ONE attention layer for a slot. `(iter, slot)` is the cost cache key.
    /// Returns the wall-time DELTA (like `IterModelExecution::evaluate_iteration`);
    /// the shell arms `now + Δ`.
    fn evaluate_attention_layer(
        &mut self,
        layer: u16,
        slot: u8,
        input: &Self::Input,
        iter: u64,
        now: Time,
    ) -> Time;
}

/// AFD-ffn family execution (interfaces doc §4, per-family surface). The purest
/// execution: NO KV and NO store — it operates purely on token counts (the ffn side has
/// no attention, so a token's originating request is irrelevant; only per-shard
/// counts drive qkv/o_proj/router/MoE). The SHELL derives the token total (store
/// scan for Bootstrap, threaded from the attn side otherwise) and owns which
/// section a task computes; the execution only partitions tokens → arch input and runs
/// the section(s). `build_task_input` is the DP-shard split; `evaluate_ffn_task` dispatches the
/// Bootstrap/Bridge/Terminal cost sections. This is why an AFD-ffn worker composes
/// as `<E>`-only (no `<K, A, E>` triple): two of the three axes degenerate away.
pub trait FfnTaskExecution {
    type Input: Default;

    fn num_dp_groups(&self) -> u16;
    fn ffn_to_attn_bytes_per_token(&self) -> u64;

    /// Partition a token total evenly across DP shards into the arch input.
    fn build_task_input(&self, tokens: u64, out: &mut Self::Input);

    /// Run the section(s) this task kind covers (Bootstrap = prologue + pre_attn(0);
    /// Bridge = post_attn(upstream); Terminal = post_attn(last) + epilogue), threading
    /// the per-section trace cursor from `start`. Returns the summed wall-time delta.
    fn evaluate_ffn_task(
        &mut self,
        kind: FfnTaskKind,
        slot: u8,
        iter_id: u64,
        input: &Self::Input,
        start: Time,
    ) -> Time;
}
