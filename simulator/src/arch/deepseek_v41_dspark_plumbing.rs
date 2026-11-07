//! The step's framework plumbing around the DSpark verify forward: attention
//! metadata, input preparation, sampler bookkeeping and draft input prep.
//!
//! Each launch is a measured kernel in the capture (`logs/20261106_0_dsv41_dspark_capture`,
//! nsys of the TP4/DP4 runs) that has no L1 kind of its own; it is priced as an
//! elementwise placeholder whose bytes come from the checkpoint, the engine
//! config and the step's batch, and whose launch count comes from the call
//! site (vLLM fork, MRV2). Most are launch-bound, so the placeholder's
//! measured small-size floor is what carries them.
//!
//! Notation: `T_p` the rows the verify forward runs (padded to its FULL graph),
//! `R` the scheduled requests, `N` `--max-num-batched-tokens`, `M`
//! `max_num_seqs`, `R_d` the draft graph's (padded) requests.
//!
//! | leaf | launches | call site |
//! |---|---|---|
//! | `swa_indices` | SWA KV groups | `mla/sparse_swa.py:668` (one `DeepseekSparseSWAMetadataBuilder.build` per group) |
//! | `swa_valid_mask` | SWA KV groups | `sparse_swa.py:625` `torch.ge(slot_mapping, 0)` |
//! | `swa_lens_tail_fill` | SWA KV groups | `sparse_swa.py:636` `decode_swa_lens[T:] = 0` |
//! | `indexer_schedule_metadata` | indexer specs | `mla/indexer.py:1837` `get_paged_mqa_logits_metadata` |
//! | `indexer_decode_metadata` | indexer specs | `mla/indexer.py:1756` (copies each row's block table) |
//! | `compressed_slot_fill` / `compressed_slot_mapping` | compressed builders | `mla/compressor_utils.py:208/218` |
//! | `gather_block_tables` | 1 | `block_table.py:197` |
//! | `combine_tokens` | 1 | `input_batch.py:495` |
//! | `slot_mappings` | 1 | `block_table.py:242` (pads every group's row to `N`) |
//! | `engram_lookback` | 1 | `nvidia/model_state.py:313` (grid `M`) |
//! | `token_request_mapping` | 1 | `ops/metadata.py:43` |
//! | `indexer_topk_fill` | index-source layers | `sparse_attn_indexer.py:721` / `sparse_mqa_indexer.py:163` `topk_indices_buffer[:T] = -1` |
//! | `logits_gather_copy` | 1 (TP > 1) | `cuda_communicator.py:518-524` movedim copy after the logits all-gather |
//! | `logits_hidden_gather` | 1 | `model_runner.py:1776` `hidden_states[logits_indices]` |
//! | `gather_draft_sampled` | 1 | `rejection_sampler.py:132` |
//! | `cu_num_logits_offset` | 1 | `rejection_sampler.py:354` |
//! | `num_sampled_and_rejected` | 1 | `input_batch.py:554` |
//! | `draft.prepare_inputs` | draft KV groups | `dflash/speculator.py:517-548` -> `_prepare_dflash_inputs_kernel` |
//! | `draft.swa_indices` / `draft.swa_valid_mask` / `draft.token_request_mapping` | draft KV groups | `sparse_swa.py:857-871`, in the draft graph |
//! | `draft.logits_gather_copy` | 1 (TP > 1) | the draft head's logits all-gather copy |
//! | `draft.hidden_gather` | 1 | `dspark/speculator.py:261` `head_hidden[sample_indices]` |
//! | `draft.token_column_write` | draft tokens | `dspark/speculator.py:312` `out_tokens[:, i] = ...` |
//! | `draft.anchor_gather` | 1 | `dspark/speculator.py:272` |
//! | `draft.draft_tokens_scatter` | 1 | `model_runner.py:2510` `draft_tokens[idx_mapping] = ...` |
//! | `post_update` | 1 | `input_batch.py:668` |
//!
//! Not modeled (source not pinned down in the capture): the forward's
//! `bfloat16_copy` launches, a second draft `<128,4>` copy, extra draft
//! gathers, mixed-step int ops; `_apply_write_kernel` and `_zero_kv_blocks`
//! (they follow block allocation, which the step input does not carry); the
//! bounded-replay batch's second metadata build in mixed steps.

use std::sync::Arc;

use crate::arch::deepseek_v41_vllm::{
    kv_groups, DeepseekV41ModelCfg, ELEMENTWISE_BACKENDS, KV_BLOCK_SIZE, MAX_NUM_SEQS,
    SWA_BLOCK_SIZE,
};
use crate::op::Op;
use crate::timing::kernels::{ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};
use crate::worklet::deepseek_v41_common::{eval_or_zero, placeholder};

const I32: u32 = 4;
const I64: u32 = 8;
const BF16: u32 = 2;

/// What a leaf's row count follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rows {
    /// `T_p`: the verify forward's rows.
    PaddedTokens,
    /// `N - T_p`: the tail a builder clears.
    PaddedTail,
    /// `N`.
    MaxBatched,
    /// `G x N`: every KV group's slot-mapping row.
    GroupsTimesMaxBatched,
    /// `M`.
    MaxSeqs,
    /// `R`.
    Requests,
    /// `R + 1`.
    RequestsPlusOne,
    /// Block-table entries the step's requests hold over every group.
    BlockEntries,
    /// Rows the sampler sees.
    LogitsRows,
    /// `R_d`.
    DraftRequests,
    /// `draft_tokens x R_d`.
    DraftRows,
}

/// One step's sizes the plumbing reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PlumbingShape {
    pub padded_tokens: u32,
    pub requests: u32,
    /// Context length of each scheduled request (its KV after this step).
    pub contexts: Vec<u32>,
    pub logits_rows: u32,
    pub draft_requests: u32,
    pub draft_rows: u32,
}

#[derive(Clone, Debug)]
pub struct StepPlumbingConfig {
    gpu_name: String,
    tp_size: u32,
    max_num_batched_tokens: u32,
    vocab_size: u32,
    hidden_size: u32,
    sliding_window: u32,
    index_topk: u32,
    draft_tokens: u32,
    max_model_len: u32,
    swa_groups: u32,
    full_groups: u32,
    ring_groups: u32,
    draft_kv_groups: u32,
    index_specs: u32,
    compressed_builders: u32,
    index_source_layers: u32,
    lookback_depth: u32,
}

impl StepPlumbingConfig {
    pub(crate) fn new(
        model: &DeepseekV41ModelCfg,
        num_draft_layers: u32,
        draft_tokens: u32,
        tp_size: u32,
        gpu_name: &str,
        max_model_len: u32,
        max_num_batched_tokens: u32,
    ) -> Self {
        let groups = kv_groups(model, num_draft_layers);
        let distinct = |layers: &[u32], compressed_only: bool| {
            let mut ratios: Vec<u32> = layers
                .iter()
                .map(|&l| model.compress_ratios[l as usize])
                .filter(|&r| !compressed_only || r > 1)
                .collect();
            ratios.sort_unstable();
            ratios.dedup();
            ratios.len() as u32
        };
        Self {
            gpu_name: gpu_name.to_string(),
            tp_size,
            max_num_batched_tokens,
            vocab_size: model.vocab_size,
            hidden_size: model.hidden_size,
            sliding_window: model.sliding_window,
            index_topk: model.index_topk,
            draft_tokens,
            max_model_len,
            swa_groups: groups.swa,
            full_groups: groups.full,
            ring_groups: groups.ring,
            // `_prepare_dflash_inputs` loops over the groups holding a draft
            // layer's SWA cache (`draft_kv_cache_group_ids`).
            draft_kv_groups: num_draft_layers.min(groups.swa),
            // One attention group per KV spec: each compress ratio among the
            // index sources is its own indexer builder.
            index_specs: distinct(&model.index_source_layer_ids, false),
            // `get_compressed_slot_mapping` runs in every ratio>1 MLA and
            // indexer builder.
            compressed_builders: distinct(&model.kv_source_layer_ids, true)
                + distinct(&model.index_source_layer_ids, true),
            index_source_layers: model.index_source_layer_ids.len() as u32,
            lookback_depth: model.engram_lookback_depth,
        }
    }

    fn kv_groups_total(&self) -> u32 {
        self.swa_groups + self.full_groups + self.ring_groups
    }
}

struct Leaf {
    op: Op<ElementwiseKernel>,
    launches: u32,
    rows: Rows,
}

/// One phase's leaves under a labeled node.
struct Phase {
    label: String,
    leaves: Vec<Leaf>,
}

pub(crate) struct StepPlumbing {
    phases: Vec<Phase>,
    cfg: StepPlumbingConfig,
}

impl StepPlumbing {
    pub(crate) fn build(
        name: &str,
        cfg: &StepPlumbingConfig,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        let c = cfg;
        let gather_copy = c.tp_size > 1;
        let w = c.sliding_window;
        let index_table_width = c.max_model_len.div_ceil(KV_BLOCK_SIZE);
        let verify = c.draft_tokens + 1;
        // (leaf, launches, rows, input bytes/row, output bytes/row)
        let preprocess = vec![
            // Reads slot, position, seq_len, query start and the window's
            // block-table entries; writes the window's indices and its length.
            (
                "swa_indices",
                c.swa_groups,
                Rows::PaddedTokens,
                I64 + 3 * I32 + I32 * w.div_ceil(SWA_BLOCK_SIZE),
                I32 * w + I32,
            ),
            ("swa_valid_mask", c.swa_groups, Rows::PaddedTokens, I64, 1),
            ("swa_lens_tail_fill", c.swa_groups, Rows::PaddedTail, 0, I32),
            // Reads context lengths and indices; the per-SM schedule it
            // writes is fixed-size (folded as one byte per row).
            (
                "indexer_schedule_metadata",
                c.index_specs,
                Rows::PaddedTokens,
                2 * I32,
                1,
            ),
            (
                "indexer_decode_metadata",
                c.index_specs,
                Rows::PaddedTokens,
                I32 * index_table_width,
                I32 * index_table_width + 3 * I32,
            ),
            ("compressed_slot_fill", c.compressed_builders, Rows::MaxBatched, 0, I64),
            (
                "compressed_slot_mapping",
                c.compressed_builders,
                Rows::PaddedTokens,
                I64 + I32 / 2,
                I64,
            ),
            ("gather_block_tables", 1, Rows::BlockEntries, I32, I32),
            // Reads the drafts and last sampled token; writes the verify
            // rows' input ids (i32) and logits indices (i64).
            (
                "combine_tokens",
                1,
                Rows::Requests,
                c.draft_tokens * I64 + 2 * I64,
                verify * (I32 + I64),
            ),
            ("slot_mappings", 1, Rows::GroupsTimesMaxBatched, 0, I64),
            ("engram_lookback", 1, Rows::MaxSeqs, 0, I32 * c.lookback_depth),
            ("token_request_mapping", 1, Rows::PaddedTokens, 0, I32),
        ];
        let forward = vec![(
            "indexer_topk_fill",
            c.index_source_layers,
            Rows::PaddedTokens,
            0,
            I32 * c.index_topk,
        )];
        let mut sample = Vec::new();
        if gather_copy {
            sample.push((
                "logits_gather_copy",
                1,
                Rows::LogitsRows,
                BF16 * c.vocab_size,
                BF16 * c.vocab_size,
            ));
        }
        sample.extend([
            (
                "logits_hidden_gather",
                1,
                Rows::LogitsRows,
                BF16 * c.hidden_size + I64,
                BF16 * c.hidden_size,
            ),
            ("gather_draft_sampled", 1, Rows::LogitsRows, 4 * I64 + I32, I64),
            ("cu_num_logits_offset", 1, Rows::RequestsPlusOne, I32, I32),
            ("num_sampled_and_rejected", 1, Rows::Requests, 2 * I64 + I32, 2 * I32),
        ]);
        let mut draft = vec![
            // Per request: the draft query rows' ids, positions, padding flag,
            // slot and sample bookkeeping, the verify rows' context positions
            // and slots, and per-request lengths/sampling params.
            (
                "draft.prepare_inputs",
                c.draft_kv_groups,
                Rows::Requests,
                verify * I64 + 5 * I64,
                c.draft_tokens * (I32 + 4 * I64 + 1) + verify * 2 * I64 + 3 * I32 + I64,
            ),
            (
                "draft.swa_indices",
                c.draft_kv_groups,
                Rows::DraftRows,
                I64 + 3 * I32 + I32 * w.div_ceil(SWA_BLOCK_SIZE),
                I32 * (w + c.draft_tokens) + I32,
            ),
            ("draft.swa_valid_mask", c.draft_kv_groups, Rows::DraftRows, I64, 1),
            ("draft.token_request_mapping", c.draft_kv_groups, Rows::DraftRows, 0, I32),
        ];
        if gather_copy {
            draft.push((
                "draft.logits_gather_copy",
                1,
                Rows::DraftRows,
                BF16 * c.vocab_size,
                BF16 * c.vocab_size,
            ));
        }
        draft.extend([
            (
                "draft.hidden_gather",
                1,
                Rows::DraftRows,
                BF16 * c.hidden_size + I64,
                BF16 * c.hidden_size,
            ),
            ("draft.token_column_write", c.draft_tokens, Rows::DraftRequests, I64, I64),
            ("draft.anchor_gather", 1, Rows::DraftRequests, I64 + I32, I64),
            (
                "draft.draft_tokens_scatter",
                1,
                Rows::Requests,
                c.draft_tokens * I64 + I64,
                c.draft_tokens * I64,
            ),
        ]);
        let bookkeep = vec![(
            "post_update",
            1,
            Rows::Requests,
            verify * I64 + I64,
            verify * I32 + 2 * I64,
        )];

        let gpu = &c.gpu_name;
        let mut phases = Vec::new();
        for (phase, spec) in [
            ("preprocess", preprocess),
            ("forward", forward),
            ("sample", sample),
            ("draft", draft),
            ("bookkeep", bookkeep),
        ] {
            let mut leaves = Vec::with_capacity(spec.len());
            for (leaf, launches, rows, input, output) in spec {
                if launches == 0 {
                    continue;
                }
                let op_name = format!("{name}.{phase}.{leaf}");
                let config: ElementwiseKernelConfig =
                    placeholder(ELEMENTWISE_BACKENDS, gpu, input, output);
                leaves.push(Leaf {
                    op: Op::new(
                        op_name.clone(),
                        Arc::new(ElementwiseKernel::build(op_name, config, bridge)?),
                    ),
                    launches,
                    rows,
                });
            }
            phases.push(Phase {
                label: format!("{name}.{phase} [framework plumbing, elementwise placeholders]"),
                leaves,
            });
        }
        Ok(Self {
            phases,
            cfg: cfg.clone(),
        })
    }

    pub(crate) fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(
            self.phases
                .iter()
                .map(|phase| CostNode::Labeled {
                    label: phase.label.clone(),
                    child: Box::new(CostNode::Sum(
                        phase
                            .leaves
                            .iter()
                            .map(|leaf| {
                                let node = leaf.op.compile(b);
                                if leaf.launches == 1 {
                                    node
                                } else {
                                    CostNode::Scale {
                                        n: leaf.launches,
                                        child: Box::new(node),
                                    }
                                }
                            })
                            .collect(),
                    )),
                })
                .collect(),
        )
    }

    fn rows(&self, rows: Rows, shape: &PlumbingShape) -> u32 {
        let c = &self.cfg;
        let n = c.max_num_batched_tokens;
        match rows {
            Rows::PaddedTokens => shape.padded_tokens,
            Rows::PaddedTail => n.saturating_sub(shape.padded_tokens),
            Rows::MaxBatched => n,
            Rows::GroupsTimesMaxBatched => c.kv_groups_total() * n,
            Rows::MaxSeqs => MAX_NUM_SEQS,
            Rows::Requests => shape.requests,
            Rows::RequestsPlusOne => {
                if shape.requests == 0 {
                    0
                } else {
                    shape.requests + 1
                }
            }
            Rows::BlockEntries => shape
                .contexts
                .iter()
                .map(|&kv| {
                    c.swa_groups * kv.div_ceil(SWA_BLOCK_SIZE)
                        + c.full_groups * kv.div_ceil(KV_BLOCK_SIZE)
                        // A compressor ring is one block per request.
                        + c.ring_groups
                })
                .sum(),
            Rows::LogitsRows => shape.logits_rows,
            Rows::DraftRequests => shape.draft_requests,
            Rows::DraftRows => shape.draft_rows,
        }
    }

    pub(crate) fn eval(&self, shape: &PlumbingShape, ev: &mut Evaluator) {
        for phase in &self.phases {
            for leaf in &phase.leaves {
                let rows = self.rows(leaf.rows, shape);
                eval_or_zero(
                    &leaf.op,
                    ElementwiseKernelInput { num_tokens: rows },
                    rows == 0,
                    ev,
                );
            }
        }
    }
}
