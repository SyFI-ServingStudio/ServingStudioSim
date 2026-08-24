//! Store-free FFN task input construction and section cost dispatch.

use std::sync::Arc;

use crate::arch::contract::{FfnArchInput, FfnLayerwiseModel};
use crate::common::Time;
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::execution::FfnTaskExecution;
use crate::worker::types::FfnTaskKind;

pub struct FfnSectionExecutionAdapter<M: FfnLayerwiseModel> {
    model: Arc<M>,
    cost: CostBuffers,
}

impl<M: FfnLayerwiseModel> FfnSectionExecutionAdapter<M> {
    pub(crate) fn new(model: Arc<M>, cost: CostBuffers) -> Self {
        Self { model, cost }
    }
}

impl<M: FfnLayerwiseModel> FfnTaskExecution for FfnSectionExecutionAdapter<M> {
    type Input = FfnArchInput;

    fn ffn_to_attn_bytes_per_token(&self) -> u64 {
        self.model.ffn_to_attn_bytes_per_token()
    }

    fn build_task_input(&self, tokens: u64, output: &mut Self::Input) {
        let num_groups = u64::from(self.model.num_dp_groups().max(1));
        #[allow(
            clippy::cast_possible_truncation,
            reason = "per-group token counts are bounded by the batch size, far below u32::MAX"
        )]
        let base = (tokens / num_groups) as u32;
        let remainder = tokens % num_groups;
        output.tokens_per_group.clear();
        output
            .tokens_per_group
            .extend((0..num_groups).map(|group| base + u32::from(group < remainder)));
    }

    fn evaluate_ffn_task(
        &mut self,
        kind: FfnTaskKind,
        slot: u8,
        iteration: u64,
        input: &Self::Input,
        start: Time,
    ) -> Time {
        let model = Arc::clone(&self.model);
        let last_layer = model.num_layers().saturating_sub(1) as usize;
        let batch_id = u64::from(slot);
        let groups = &input.tokens_per_group;
        let cache_key = input.tokens_per_group.as_slice();
        let mut total = Time::ZERO;
        let mut cursor = start;

        match kind {
            FfnTaskKind::Bootstrap => {
                let duration = self.cost.run_section(
                    "prologue",
                    -1,
                    iteration,
                    batch_id,
                    groups,
                    Some(cache_key),
                    cursor,
                    |slots, scratch, captured_inputs| match captured_inputs {
                        Some(captured_inputs) => {
                            model.prologue_cost_with_inputs(input, slots, scratch, captured_inputs)
                        }
                        None => model.prologue_cost(input, slots, scratch),
                    },
                );
                total += duration;
                cursor += duration;
                let duration = self.cost.run_section(
                    "pre_attn",
                    0,
                    iteration,
                    batch_id,
                    groups,
                    Some(cache_key),
                    cursor,
                    |slots, scratch, captured_inputs| match captured_inputs {
                        Some(captured_inputs) => model.pre_attn_cost_with_inputs(
                            0,
                            input,
                            slots,
                            scratch,
                            captured_inputs,
                        ),
                        None => model.pre_attn_cost(0, input, slots, scratch),
                    },
                );
                total += duration;
            }
            FfnTaskKind::Bridge { upstream } => {
                #[allow(
                    clippy::cast_possible_wrap,
                    reason = "layer indices are a handful to low hundreds, far below i16::MAX"
                )]
                let upstream_key = upstream as i16;
                let duration = self.cost.run_section(
                    "post_attn",
                    upstream_key,
                    iteration,
                    batch_id,
                    groups,
                    Some(cache_key),
                    cursor,
                    |slots, scratch, captured_inputs| match captured_inputs {
                        Some(captured_inputs) => model.post_attn_cost_with_inputs(
                            upstream as usize,
                            input,
                            slots,
                            scratch,
                            captured_inputs,
                        ),
                        None => model.post_attn_cost(upstream as usize, input, slots, scratch),
                    },
                );
                total += duration;
            }
            FfnTaskKind::Terminal => {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_possible_wrap,
                    reason = "last_layer is the model's final layer index, a handful to low hundreds, far below i16::MAX"
                )]
                let last_layer_key = last_layer as i16;
                let duration = self.cost.run_section(
                    "post_attn_last",
                    last_layer_key,
                    iteration,
                    batch_id,
                    groups,
                    Some(cache_key),
                    cursor,
                    |slots, scratch, captured_inputs| match captured_inputs {
                        Some(captured_inputs) => model.post_attn_cost_with_inputs(
                            last_layer,
                            input,
                            slots,
                            scratch,
                            captured_inputs,
                        ),
                        None => model.post_attn_cost(last_layer, input, slots, scratch),
                    },
                );
                total += duration;
                cursor += duration;
                let duration = self.cost.run_section(
                    "epilogue",
                    -1,
                    iteration,
                    batch_id,
                    groups,
                    Some(cache_key),
                    cursor,
                    |slots, scratch, captured_inputs| match captured_inputs {
                        Some(captured_inputs) => {
                            model.epilogue_cost_with_inputs(input, slots, scratch, captured_inputs)
                        }
                        None => model.epilogue_cost(input, slots, scratch),
                    },
                );
                total += duration;
            }
        }
        total
    }
}
