//! Whole-iteration input construction and model-cost evaluation.
//!
//! `UnifiedIterExecution` owns the model and `CostBuffers`. It renders directly
//! from borrowed KV membership views, preserving the former worker's allocation
//! shape instead of materializing intermediate request snapshots.

use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{SharedRequests, Time};
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::execution::{IterModelExecution, ModelKvLayout};
use crate::worker::kv::{IterWorkerKv, PrefixKv};
use crate::worker::types::IterBatchPlan;

pub struct UnifiedIterExecution<M: IterwiseUnifiedModel> {
    model: Arc<M>,
    cost: CostBuffers,
}

impl<M: IterwiseUnifiedModel> UnifiedIterExecution<M> {
    pub(crate) fn new(model: Arc<M>, cost: CostBuffers) -> Self {
        Self { model, cost }
    }

    fn build_input<K: IterWorkerKv + PrefixKv>(
        &self,
        kv_store: &K,
        _requests: &SharedRequests,
        batch_plan: &IterBatchPlan,
        out: &mut UnifiedArchInput,
    ) {
        let num_partitions = kv_store.num_partitions();
        out.groups
            .resize_with(num_partitions, ArchGroupInput::default);
        out.groups.truncate(num_partitions);

        for partition in 0..num_partitions as u16 {
            let group = &mut out.groups[partition as usize];
            group.clear();
            kv_store.visit_prefill_admits(partition, |request| {
                let resolved_prefill = kv_store.resolved_prefill_context(request);
                let (prefix_tokens, chunk_tokens) = resolved_prefill.active_chunk();
                group
                    .prefill_chunk_pairs
                    .push((prefix_tokens, chunk_tokens));
                group.prefill_tokens += chunk_tokens;
            });
            if batch_plan.partition_runs_decode(partition) {
                kv_store.visit_decode_members(partition, |_, current_kv| {
                    group.decode_kv_lens.push(current_kv as u32);
                    group.total_kv_len += current_kv as u32;
                });
            }
            group.decode_tokens = group.decode_kv_lens.len() as u32;
            group.batch_tokens = group.prefill_tokens + group.decode_tokens;
        }
        if num_partitions > 1 {
            out.tokens_per_source_rank.clear();
            out.tokens_per_source_rank
                .extend(out.groups.iter().map(|group| group.batch_tokens));
        } else {
            out.tokens_per_source_rank.clear();
        }
    }

    pub(crate) fn evaluate_iteration(
        &mut self,
        input: &UnifiedArchInput,
        iteration: u64,
        now: Time,
    ) -> Time {
        self.cost
            .run_iter(self.model.as_ref(), input, iteration, now)
    }
}

impl<M, K> IterModelExecution<K> for UnifiedIterExecution<M>
where
    M: IterwiseUnifiedModel,
    K: IterWorkerKv + PrefixKv,
{
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
        batch_plan: &IterBatchPlan,
        out: &mut Self::Input,
    ) {
        self.build_input(kv_store, requests, batch_plan, out);
    }

    fn evaluate_iteration(&mut self, input: &Self::Input, iteration: u64, now: Time) -> Time {
        UnifiedIterExecution::evaluate_iteration(self, input, iteration, now)
    }
}
