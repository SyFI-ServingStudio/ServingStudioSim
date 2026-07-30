//! `FfnSectionExecutionAdapter<M>` — the AFD-ffn (layer-wise, section-dispatched) execution
//! adapter. Holds `Arc<M: FfnLayerwiseModel>` + `CostBuffers`; `Input = FfnArchInput`.
//!
//! Store-free and KV-free: `build_task_input` splits a token total evenly across the
//! model's DP shards; `evaluate_ffn_task` dispatches the Bridge/Bootstrap/Terminal cost
//! sections through the real `CostBuffers::run_section`, threading the trace cursor
//! from `start` (mirrors `DisaggFfnWorker::{build_arch_input, compute_time}`,
//! disagg_ffn.rs:259/296). The section→layer split follows the L4 §4.1 convention:
//! Bootstrap = prologue + pre_attn(0); Bridge{u} = post_attn(u); Terminal =
//! post_attn(last) + epilogue.

use std::sync::Arc;

use crate::arch::contract::{FfnArchInput, FfnLayerwiseModel};
use crate::common::Time;
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::types::FfnTaskKind;

use super::FfnTaskExecution;

pub struct FfnSectionExecutionAdapter<M: FfnLayerwiseModel> {
    model: Arc<M>,
    cost: CostBuffers,
}

impl<M: FfnLayerwiseModel> FfnSectionExecutionAdapter<M> {
    pub fn new(model: Arc<M>, cost: CostBuffers) -> Self {
        Self { model, cost }
    }
}

impl<M: FfnLayerwiseModel> FfnTaskExecution for FfnSectionExecutionAdapter<M> {
    type Input = FfnArchInput;

    #[inline]
    fn num_dp_groups(&self) -> u16 {
        self.model.num_dp_groups()
    }

    #[inline]
    fn ffn_to_attn_bytes_per_token(&self) -> u64 {
        self.model.ffn_to_attn_bytes_per_token()
    }

    fn build_task_input(&self, tokens: u64, output: &mut FfnArchInput) {
        // Even split, remainder to the first groups (matches build_arch_input). The
        // ffn arch reads only per-shard counts, so no attention-shaped vocabulary.
        let num_groups = self.model.num_dp_groups().max(1) as u64;
        let tokens_per_group = (tokens / num_groups) as u32;
        let remainder = tokens % num_groups;
        output.tokens_per_group.clear();
        output.tokens_per_group.extend(
            (0..num_groups)
                .map(|group_index| tokens_per_group + u32::from(group_index < remainder)),
        );
    }

    fn evaluate_ffn_task(
        &mut self,
        kind: FfnTaskKind,
        slot: u8,
        iteration_id: u64,
        input: &FfnArchInput,
        start_time: Time,
    ) -> Time {
        // Clone the Arc so the model borrow does not alias the `&mut self.cost` borrow.
        let model = Arc::clone(&self.model);
        let last_layer = model.num_layers().saturating_sub(1) as usize;
        let batch_id = slot as u64;
        let groups = &input.tokens_per_group;
        let key = input.tokens_per_group.as_slice();

        // Sections lay out back-to-back from `start`; `cursor` is each section's
        // trace start (post-scale wall), `total` the summed wall-time delta.
        let mut total_time = Time::ZERO;
        let mut section_start = start_time;
        match kind {
            FfnTaskKind::Bootstrap => {
                let section_time = self.cost.run_section(
                    "prologue",
                    -1,
                    iteration_id,
                    batch_id,
                    groups,
                    Some(key),
                    section_start,
                    |compute_metrics, communication_metrics, slot_inputs| match slot_inputs {
                        Some(slot_inputs) => model.prologue_cost_with_inputs(
                            input,
                            compute_metrics,
                            communication_metrics,
                            slot_inputs,
                        ),
                        None => model.prologue_cost(input, compute_metrics, communication_metrics),
                    },
                );
                total_time += section_time;
                section_start += section_time;
                let section_time = self.cost.run_section(
                    "pre_attn",
                    0,
                    iteration_id,
                    batch_id,
                    groups,
                    Some(key),
                    section_start,
                    |compute_metrics, communication_metrics, slot_inputs| match slot_inputs {
                        Some(slot_inputs) => model.pre_attn_cost_with_inputs(
                            0,
                            input,
                            compute_metrics,
                            communication_metrics,
                            slot_inputs,
                        ),
                        None => {
                            model.pre_attn_cost(0, input, compute_metrics, communication_metrics)
                        }
                    },
                );
                total_time += section_time;
            }
            FfnTaskKind::Bridge { upstream } => {
                let section_time = self.cost.run_section(
                    "post_attn",
                    upstream as i16,
                    iteration_id,
                    batch_id,
                    groups,
                    Some(key),
                    section_start,
                    |compute_metrics, communication_metrics, slot_inputs| match slot_inputs {
                        Some(slot_inputs) => model.post_attn_cost_with_inputs(
                            upstream as usize,
                            input,
                            compute_metrics,
                            communication_metrics,
                            slot_inputs,
                        ),
                        None => model.post_attn_cost(
                            upstream as usize,
                            input,
                            compute_metrics,
                            communication_metrics,
                        ),
                    },
                );
                total_time += section_time;
            }
            FfnTaskKind::Terminal => {
                let section_time = self.cost.run_section(
                    "post_attn_last",
                    last_layer as i16,
                    iteration_id,
                    batch_id,
                    groups,
                    Some(key),
                    section_start,
                    |compute_metrics, communication_metrics, slot_inputs| match slot_inputs {
                        Some(slot_inputs) => model.post_attn_cost_with_inputs(
                            last_layer,
                            input,
                            compute_metrics,
                            communication_metrics,
                            slot_inputs,
                        ),
                        None => model.post_attn_cost(
                            last_layer,
                            input,
                            compute_metrics,
                            communication_metrics,
                        ),
                    },
                );
                total_time += section_time;
                section_start += section_time;
                let section_time = self.cost.run_section(
                    "epilogue",
                    -1,
                    iteration_id,
                    batch_id,
                    groups,
                    Some(key),
                    section_start,
                    |compute_metrics, communication_metrics, slot_inputs| match slot_inputs {
                        Some(slot_inputs) => model.epilogue_cost_with_inputs(
                            input,
                            compute_metrics,
                            communication_metrics,
                            slot_inputs,
                        ),
                        None => model.epilogue_cost(input, compute_metrics, communication_metrics),
                    },
                );
                total_time += section_time;
            }
        }
        total_time
    }
}
