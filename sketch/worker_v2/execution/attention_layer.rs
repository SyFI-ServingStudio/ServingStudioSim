//! `AttentionLayerExecutionAdapter<M>` — the AFD-attn (layer-wise) execution adapter.
//!
//! Holds `Arc<M: AttnLayerwiseModel>` + `CostBuffers`; `Input = AttnArchInput`.
//! `build_slot_input` renders ONE slot's `ArchGroupInput` from its request grouping:
//! prefill members contribute `(prefix_kv, active_chunk_len)` chunk pairs, decode
//! members contribute `current_kv` (read from `SlotPipelineKv`).
//! `evaluate_attention_layer` calls the
//! real `CostBuffers::run_section("attn", layer, iter, slot, &groups, Some(key), ..)`
//! with the model's `attn_cost[_with_inputs]` closure — the same protocol the real
//! `DisaggAttnWorker::run_attn_layer` uses (disagg_attn.rs:928).

use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, AttnArchInput, AttnLayerwiseModel};
use crate::common::{RequestId, SharedRequests, Time};
use crate::worker::cost_buffers::CostBuffers;

use super::super::kv::SlotPipelineKv;
use super::super::shared::advance_scope::AdvanceScope;
use super::{AttentionLayerExecution, ModelKvLayout};

pub struct AttentionLayerExecutionAdapter<M: AttnLayerwiseModel> {
    model: Arc<M>,
    cost: CostBuffers,
}

impl<M: AttnLayerwiseModel> AttentionLayerExecutionAdapter<M> {
    pub fn new(model: Arc<M>, cost: CostBuffers) -> Self {
        Self { model, cost }
    }
}

impl<M: AttnLayerwiseModel> AttentionLayerExecution for AttentionLayerExecutionAdapter<M> {
    type Input = AttnArchInput;

    #[inline]
    fn num_layers(&self) -> u16 {
        self.model.num_layers() as u16
    }

    fn model_kv_layout(&self) -> ModelKvLayout {
        ModelKvLayout {
            total_kv_bytes_per_token: self.model.total_kv_bytes_per_token(),
            num_attn_shards: self.model.num_attn_shards(),
        }
    }

    #[inline]
    fn attn_to_ffn_bytes_per_token(&self) -> u64 {
        self.model.attn_to_ffn_bytes_per_token()
    }

    fn build_slot_input<K: SlotPipelineKv>(
        &self,
        grouping: AdvanceScope<'_>,
        kv_store: &K,
        requests: &SharedRequests,
        out: &mut AttnArchInput,
    ) -> u64 {
        let request_ids: &[RequestId] = match grouping {
            AdvanceScope::RequestSubset { request_ids, .. } => request_ids,
            // A slot never renders a whole partition; be defensive, not panicky.
            AdvanceScope::WholePartition(_) => &[],
        };

        let store = requests.borrow();
        let mut group = ArchGroupInput::default();
        for &rid in request_ids {
            let record = &store[rid];
            if record.is_prefill() {
                group
                    .prefill_chunk_pairs
                    .push((record.prefix_kv, record.active_chunk_len));
                group.prefill_tokens += record.active_chunk_len;
            } else {
                let current_kv = kv_store.current_kv(0, rid).unwrap_or(0) as u32;
                group.decode_kv_lens.push(current_kv);
                group.total_kv_len += current_kv;
            }
        }
        group.decode_tokens = group.decode_kv_lens.len() as u32;
        group.batch_tokens = group.prefill_tokens + group.decode_tokens;
        let tokens = group.batch_tokens as u64;

        out.groups.clear();
        out.groups.push(group);
        tokens
    }

    fn evaluate_attention_layer(
        &mut self,
        layer: u16,
        slot: u8,
        input: &AttnArchInput,
        iter: u64,
        now: Time,
    ) -> Time {
        // Clone the Arc so the closure borrows `model` disjointly from `&mut self.cost`.
        let model = Arc::clone(&self.model);
        // attn's cheap identity key: input never repeats across iters, so (iter, slot).
        let cache_key = [iter as u32, slot as u32];
        self.cost.run_section(
            "attn",
            layer as i16,
            iter,
            slot as u64,
            &input.groups,
            Some(&cache_key),
            now,
            |slots, scratch, inputs| match inputs {
                Some(captured) => {
                    model.attn_cost_with_inputs(layer as usize, input, slots, scratch, captured)
                }
                None => model.attn_cost(layer as usize, input, slots, scratch),
            },
        )
    }
}
