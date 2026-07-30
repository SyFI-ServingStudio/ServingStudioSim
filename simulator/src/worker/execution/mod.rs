//! Model input construction and cost-evaluation axis.

use crate::common::{SharedRequests, Time};
use crate::worker::kv::{IterWorkerKv, SlotPipelineKv};
use crate::worker::shared::advance_scope::AdvanceScope;

mod attention_layer_execution;
mod ffn_task_execution;
mod unified_iter_execution;

pub use attention_layer_execution::AttentionLayerExecutionAdapter;
pub use ffn_task_execution::FfnSectionExecutionAdapter;
pub use unified_iter_execution::UnifiedIterExecution;

/// Model-owned facts used to size a worker's KV store and transfer payloads.
#[derive(Clone, Copy)]
pub struct ModelKvLayout {
    pub total_kv_bytes_per_token: u64,
    pub num_attn_shards: u16,
}

pub trait IterModelExecution<K: IterWorkerKv> {
    type Input: Default;

    fn model_kv_layout(&self) -> ModelKvLayout;
    fn build_iteration_input(&self, kv_store: &K, requests: &SharedRequests, out: &mut Self::Input);
    fn evaluate_iteration(&mut self, input: &Self::Input, iteration: u64, now: Time) -> Time;
}

pub trait AttentionLayerExecution {
    type Input: Default;

    fn num_layers(&self) -> u16;
    fn model_kv_layout(&self) -> ModelKvLayout;
    fn attn_to_ffn_bytes_per_token(&self) -> u64;
    fn build_slot_input<K: SlotPipelineKv>(
        &self,
        scope: AdvanceScope<'_>,
        kv_store: &K,
        requests: &SharedRequests,
        output: &mut Self::Input,
    ) -> u64;
    fn evaluate_attention_layer(
        &mut self,
        layer: u16,
        slot: u8,
        input: &Self::Input,
        iteration: u64,
        now: Time,
    ) -> Time;
}

pub trait FfnTaskExecution {
    type Input: Default;

    fn ffn_to_attn_bytes_per_token(&self) -> u64;
    fn build_task_input(&self, tokens: u64, output: &mut Self::Input);
    fn evaluate_ffn_task(
        &mut self,
        kind: crate::worker::types::FfnTaskKind,
        slot: u8,
        iteration: u64,
        input: &Self::Input,
        start: Time,
    ) -> Time;
}
