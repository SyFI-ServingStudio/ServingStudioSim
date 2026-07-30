//! `MultiModelIterExecution<M>` — execution for N co-resident models (interfaces doc §4,
//! multi-arch). Where `UnifiedIterExecution` runs ONE model over all DP-shard groups (the arch
//! `Max`es them for the DP wallclock), this runs a DIFFERENT model per partition —
//! partition ≡ `ModelId`, matching `ModelPartitionedKv`'s pool-per-model layout. Each model
//! keeps its own `CostBuffers` (separate cost caches), and the per-model segment times
//! combine with `Max` (the models are co-resident and run concurrently on the shard).
//!
//! For the sketch the N models share one type `M` (e.g. sizes/fine-tunes selected by
//! config); genuinely heterogeneous archs would need a boxed-model vector, an additive
//! change. `build_iteration_input` is byte-identical to `UnifiedIterExecution`'s
//! (one group per partition), so the KV/store reading seam is unchanged — only
//! `evaluate_iteration` fans out per model.

use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{SharedRequests, Time};
use crate::worker::cost_buffers::CostBuffers;

use super::super::kv::IterWorkerKv;
use super::{IterModelExecution, ModelKvLayout};

pub struct MultiModelIterExecution<M: IterwiseUnifiedModel> {
    models: Vec<Arc<M>>,
    costs: Vec<CostBuffers>,
    /// Reused single-group input handed to one model's `run_iter` (avoids reallocating).
    scratch: UnifiedArchInput,
}

impl<M: IterwiseUnifiedModel> MultiModelIterExecution<M> {
    pub fn new(models: Vec<Arc<M>>, costs: Vec<CostBuffers>) -> Self {
        Self {
            models,
            costs,
            scratch: UnifiedArchInput::default(),
        }
    }
}

impl<M: IterwiseUnifiedModel, K: IterWorkerKv> IterModelExecution<K>
    for MultiModelIterExecution<M>
{
    type Input = UnifiedArchInput;

    fn model_kv_layout(&self) -> ModelKvLayout {
        // Representative layout (model 0). Heterogeneous per-model KV sizing would key
        // the pool capacities per model — deferred with the shared-budget refinement.
        ModelKvLayout {
            total_kv_bytes_per_token: self.models[0].total_kv_bytes_per_token(),
            num_attn_shards: self.models[0].num_attn_shards(),
        }
    }

    /// Same builder as `UnifiedIterExecution`: one `ArchGroupInput` per KV partition (here, per
    /// co-resident model). The shell/KV never learn which model a partition is.
    fn build_iteration_input(
        &self,
        kv_store: &K,
        requests: &SharedRequests,
        out: &mut UnifiedArchInput,
    ) {
        let store = requests.borrow();
        out.groups.clear();
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

    /// Run each non-empty partition through ITS model's cost; the co-resident models run
    /// concurrently, so the iteration wallclock is the `Max` of their segment deltas.
    fn evaluate_iteration(&mut self, input: &UnifiedArchInput, iter: u64, now: Time) -> Time {
        let mut max_delta = Time::ZERO;
        for (partition, group) in input.groups.iter().enumerate() {
            if partition >= self.models.len() || group.batch_tokens == 0 {
                continue;
            }
            self.scratch.groups.clear();
            self.scratch.groups.push(group.clone());
            self.scratch.tokens_per_source_rank.clear();
            let delta = self.costs[partition].run_iter(
                self.models[partition].as_ref(),
                &self.scratch,
                iter,
                now,
            );
            max_delta = max_delta.max(delta);
        }
        max_delta
    }
}
