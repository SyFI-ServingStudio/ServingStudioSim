//! Layer-wise attention input construction and cost evaluation for AFD workers.

use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, AttnArchInput, AttnLayerwiseModel};
use crate::common::{RequestId, SharedRequests, Time};
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::execution::{AttentionLayerExecution, ModelKvLayout};
use crate::worker::kv::{PrefixKv, SlotPipelineKv};
use crate::worker::shared::advance_scope::AdvanceScope;

pub struct AttentionLayerExecutionAdapter<M: AttnLayerwiseModel> {
    model: Arc<M>,
    cost: CostBuffers,
}

impl<M: AttnLayerwiseModel> AttentionLayerExecutionAdapter<M> {
    pub(crate) fn new(model: Arc<M>, cost: CostBuffers) -> Self {
        Self { model, cost }
    }
}

impl<M: AttnLayerwiseModel> AttentionLayerExecution for AttentionLayerExecutionAdapter<M> {
    type Input = AttnArchInput;

    fn num_layers(&self) -> u16 {
        self.model.num_layers() as u16
    }

    fn model_kv_layout(&self) -> ModelKvLayout {
        ModelKvLayout {
            total_kv_bytes_per_token: self.model.total_kv_bytes_per_token(),
            num_attn_shards: self.model.num_attn_shards(),
        }
    }

    fn attn_to_ffn_bytes_per_token(&self) -> u64 {
        self.model.attn_to_ffn_bytes_per_token()
    }

    fn build_slot_input<K: SlotPipelineKv + PrefixKv>(
        &self,
        scope: AdvanceScope<'_>,
        kv_store: &K,
        _requests: &SharedRequests,
        output: &mut Self::Input,
    ) -> u64 {
        let request_ids: &[RequestId] = match scope {
            AdvanceScope::RequestSubset { request_ids, .. } => request_ids,
            AdvanceScope::WholePartition(_) => {
                debug_assert!(false, "attention slot input requires a request subset");
                &[]
            }
        };

        if output.groups.is_empty() {
            output.groups.push(ArchGroupInput::default());
        }
        output.groups.truncate(1);
        let group = &mut output.groups[0];
        group.clear();

        for &request in request_ids {
            if kv_store.has_reservation(request) {
                let (resident_prefix_tokens, prefill_compute_tokens) =
                    kv_store.prefill_pair(request);
                group
                    .prefill_chunk_pairs
                    .push((resident_prefix_tokens, prefill_compute_tokens));
                group.prefill_tokens += prefill_compute_tokens;
                group.batch_tokens += prefill_compute_tokens;
            } else {
                let current_kv = kv_store.current_kv(0, request).unwrap_or(0) as u32;
                group.decode_kv_lens.push(current_kv);
                group.decode_tokens += 1;
                group.batch_tokens += 1;
                group.total_kv_len += current_kv;
            }
        }
        group.batch_tokens as u64
    }

    fn evaluate_attention_layer(
        &mut self,
        layer: u16,
        slot: u8,
        input: &Self::Input,
        iteration: u64,
        now: Time,
    ) -> Time {
        let model = Arc::clone(&self.model);
        let cache_key = [iteration as u32, (iteration >> 32) as u32, slot as u32];
        self.cost.run_section(
            "attn",
            layer as i16,
            iteration,
            slot as u64,
            &input.groups,
            Some(&cache_key),
            now,
            |slots, scratch, captured_inputs| match captured_inputs {
                Some(captured_inputs) => model.attn_cost_with_inputs(
                    layer as usize,
                    input,
                    slots,
                    scratch,
                    captured_inputs,
                ),
                None => model.attn_cost(layer as usize, input, slots, scratch),
            },
        )
    }
}
