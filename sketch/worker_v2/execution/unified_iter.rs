//! `UnifiedIterExecution<M>` — the family-A (whole-iteration) execution adapter.
//!
//! Holds `Arc<M>` + `CostBuffers`; `Input = UnifiedArchInput`. `build_iteration_input`
//! reproduces the real `BareboneWorker::build_arch_input` (unified.rs:383): read
//! `prefix_kv`/`active_chunk_len` per prefill admit, `current_kv` per decode.
//! `evaluate_iteration` calls the real `CostBuffers::run_iter(model, input, iter, now)`.

use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{SharedRequests, Time};
use crate::worker::cost_buffers::CostBuffers;

use super::super::kv::IterWorkerKv;
use super::{IterModelExecution, ModelKvLayout};

pub struct UnifiedIterExecution<M: IterwiseUnifiedModel> {
    model: Arc<M>,
    cost: CostBuffers,
}

impl<M: IterwiseUnifiedModel> UnifiedIterExecution<M> {
    pub fn new(model: Arc<M>, cost: CostBuffers) -> Self {
        Self { model, cost }
    }
}

impl<M: IterwiseUnifiedModel, K: IterWorkerKv> IterModelExecution<K> for UnifiedIterExecution<M> {
    type Input = UnifiedArchInput;

    fn model_kv_layout(&self) -> ModelKvLayout {
        ModelKvLayout {
            total_kv_bytes_per_token: self.model.total_kv_bytes_per_token(),
            num_attn_shards: self.model.num_attn_shards(),
        }
    }

    fn build_iteration_input(
        &self,
        kv_store: &K,
        requests: &SharedRequests,
        out: &mut UnifiedArchInput,
    ) {
        let store = requests.borrow();
        out.groups.clear();
        // One ArchGroupInput per KV partition (DP shard): N=1 barebone → 1 group,
        // N>1 HP/DP → N groups. Same code path for both.
        for partition in 0..kv_store.num_partitions() as u16 {
            let mut group = ArchGroupInput::default();
            for rid in kv_store.prefill_admits(partition) {
                let record = &store[rid];
                group
                    .prefill_chunk_pairs
                    .push((record.prefix_kv, record.active_chunk_len));
                group.prefill_tokens += record.active_chunk_len;
            }
            for (_rid, current_kv) in kv_store.decode_members(partition) {
                group.decode_kv_lens.push(current_kv as u32);
                group.total_kv_len += current_kv as u32;
            }
            group.decode_tokens = group.decode_kv_lens.len() as u32;
            group.batch_tokens = group.prefill_tokens + group.decode_tokens;
            out.groups.push(group);
        }
        out.tokens_per_source_rank.clear();
    }

    fn evaluate_iteration(&mut self, input: &UnifiedArchInput, iter: u64, now: Time) -> Time {
        self.cost.run_iter(self.model.as_ref(), input, iter, now)
    }
}
