//! GLM-5.3-Flash NVFP4 with a DFlash2 block-parallel proposer.
//!
//! A composition of two models that already exist, with no new L1 kernel:
//!
//! - the **target** is the ordinary GLM-5.3-Flash graph, either the TP/EP one
//!   ([`Glm53FlashVllmModel`]) or the DP-attention/EP one
//!   ([`Glm53FlashDpAttnEpModel`]), compiled unchanged;
//! - the **proposer** is the GLM-5.3 [`Dflash2DraftStage`], one per attention
//!   DP rank.
//!
//! # The verify pass is billed as decode rows
//!
//! A verify step scores `draft_tokens + 1` query rows per resident request.
//! The Flash graph has no multi-query decode path: its DSA indexer and sparse
//! MLA are built at `next_n = 1` and the KDA recurrent kernel steps one token.
//! So each verify row is lowered to its own decode row at its own context
//! length, and every kernel the target bills is a profiled decode kernel.
//!
//! That is an upper bound on the verify cost, not a measurement: the rows of
//! one request re-read that request's KV, indexer keys and KDA state once per
//! row, where a multi-query kernel would read them once. Nothing here was
//! aligned against a speculative Flash capture.
//!
//! # The proposer is the GLM-5.3 one
//!
//! No DFlash2 checkpoint for Flash is modeled; the draft keeps the GLM-5.3
//! proposer's dimensions (hidden 6144, six GQA layers) so its profiled rows are
//! reused. Under attention DP each rank drafts its own requests at TP1, and the
//! ranks draft concurrently, so the draft costs the slowest rank's call.

use crate::arch::contract::{
    ArchGroupInput, IterwiseUnifiedModel, SpeculativeArchInput, SpeculativeUnifiedModel,
    UnifiedArchInput,
};
use crate::arch::glm53_flash_vllm_fp8_dp_attn_ep_moe::Glm53FlashDpAttnEpModel;
use crate::arch::glm53_flash_vllm_fp8_kda_dsa_moe::Glm53FlashVllmModel;
use crate::arch::glm53_vllm_nvfp4_dsa_moe_dflash2::{
    draft_kv_bytes_per_token, Dflash2DraftResolved, Dflash2DraftStage,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};

const ARCH_KIND: &str = "glm53_flash_vllm_nvfp4_dflash2";

/// A Flash graph a DFlash2 verify pass can run on: compiled as one node beside
/// the draft, and filled from an ordinary unified batch.
pub trait Glm53FlashVerifyTarget: IterwiseUnifiedModel {
    fn compile_target(&self, builder: &mut CostTreeBuilder) -> CostNode;
    fn eval_target(&self, input: &UnifiedArchInput, ev: &mut Evaluator);
}

impl Glm53FlashVerifyTarget for Glm53FlashVllmModel {
    fn compile_target(&self, builder: &mut CostTreeBuilder) -> CostNode {
        self.compile_root(builder)
    }

    fn eval_target(&self, input: &UnifiedArchInput, ev: &mut Evaluator) {
        self.eval_into(input, ev);
    }
}

impl Glm53FlashVerifyTarget for Glm53FlashDpAttnEpModel {
    fn compile_target(&self, builder: &mut CostTreeBuilder) -> CostNode {
        self.compile_root(builder)
    }

    fn eval_target(&self, input: &UnifiedArchInput, ev: &mut Evaluator) {
        self.eval_into(input, ev);
    }
}

/// GLM-5.3-Flash NVFP4 deployed with a DFlash2 proposer in front of it.
pub struct Glm53FlashVllmNvfp4Dflash2Model<T> {
    target: T,
    /// One proposer per attention DP group, in group order.
    drafts: Vec<Dflash2DraftStage>,
    /// Candidate positions drafted per request per iteration. The compiled tree
    /// assumes this depth.
    draft_tokens: u32,
    max_model_len: u32,
    /// The draft's per-token KV, summed over one group's attention shards, as
    /// the worker's capacity contract counts the target's.
    draft_kv_bytes_per_token: u64,
    /// The draft's KV held per request instead, when its window caps it.
    draft_kv_bytes_per_request: u64,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

impl<T: Glm53FlashVerifyTarget> Glm53FlashVllmNvfp4Dflash2Model<T> {
    fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let target = self.target.compile_target(&mut builder);
        let mut drafts: Vec<CostNode> = self
            .drafts
            .iter()
            .map(|draft| draft.compile(&mut builder))
            .collect();
        let draft = if drafts.len() == 1 {
            drafts.pop().expect("one draft stage")
        } else {
            CostNode::Max {
                overlap: 1.0,
                children: drafts,
            }
        };
        let width = self.draft_tokens + 1;
        builder.finish(CostNode::Labeled {
            label: format!(
                "GLM-5.3-Flash NVFP4 + DFlash2 [draft_tokens={}; verify width={width} as decode rows; \
                 {} draft rank(s)]",
                self.draft_tokens,
                self.drafts.len()
            ),
            child: Box::new(CostNode::Sum(vec![
                CostNode::Labeled {
                    label: format!("target verify [{width} decode rows per resident request]"),
                    child: Box::new(target),
                },
                CostNode::Labeled {
                    label: "dflash2 proposer [Max over attention DP ranks]".to_string(),
                    child: Box::new(draft),
                },
            ])),
        })
    }

    fn eval_into(&self, input: &SpeculativeArchInput, ev: &mut Evaluator) {
        self.check_input(input).unwrap_or_else(|reason| {
            panic!("invalid Glm53FlashVllmNvfp4Dflash2Model input: {reason}")
        });
        self.target.eval_target(&verify_as_decode_rows(input), ev);
        for (draft, group) in self.drafts.iter().zip(&input.groups) {
            let endpoints: Vec<u32> = group
                .prefill_chunk_pairs
                .iter()
                .map(|&(prefix, append)| prefix + append)
                .chain(group.decode_requests.iter().map(|decode| decode.kv_len))
                .collect();
            draft.eval_group(
                group.request_count(),
                group.batch_tokens,
                &endpoints,
                self.draft_tokens,
                ev,
            );
        }
    }

    fn eval_checked(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: Option<&mut Vec<SlotInput>>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut evaluator = match inputs {
            Some(inputs) => Evaluator::with_inputs(slots, inputs),
            None => Evaluator::new(slots),
        };
        self.eval_into(batch, &mut evaluator);
        assert_eq!(
            evaluator.filled(),
            self.n_slots,
            "eval must fill every compiled slot"
        );
        drop(evaluator);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }
}

/// The verify batch as the target sees it: prefill chunks unchanged, and each
/// decode request's `query_len` rows as that many decode rows ending at its
/// final KV length.
fn verify_as_decode_rows(input: &SpeculativeArchInput) -> UnifiedArchInput {
    UnifiedArchInput {
        groups: input
            .groups
            .iter()
            .map(|group| {
                let decode_kv_lens: Vec<u32> = group
                    .decode_requests
                    .iter()
                    .flat_map(|decode| {
                        let first = decode.kv_len - decode.query_len + 1;
                        first..=decode.kv_len
                    })
                    .collect();
                let decode_tokens = decode_kv_lens.len() as u32;
                ArchGroupInput {
                    batch_tokens: group.prefill_tokens + decode_tokens,
                    prefill_tokens: group.prefill_tokens,
                    decode_tokens,
                    prefill_chunk_pairs: group.prefill_chunk_pairs.clone(),
                    decode_kv_lens,
                    total_kv_len: group.total_kv_len,
                }
            })
            .collect(),
        tokens_per_source_rank: input.tokens_per_source_rank.clone(),
    }
}

impl<T: Glm53FlashVerifyTarget> SpeculativeUnifiedModel for Glm53FlashVllmNvfp4Dflash2Model<T> {
    fn check_input(&self, batch: &SpeculativeArchInput) -> Result<(), String> {
        if batch.draft_tokens != self.draft_tokens {
            return Err(format!(
                "draft_tokens {} must equal the compiled {}",
                batch.draft_tokens, self.draft_tokens
            ));
        }
        if batch.groups.len() != self.drafts.len() {
            return Err(format!(
                "{} groups for {} attention DP groups",
                batch.groups.len(),
                self.drafts.len()
            ));
        }
        let width = self.draft_tokens + 1;
        for (index, group) in batch.groups.iter().enumerate() {
            for decode in &group.decode_requests {
                if decode.query_len != width || decode.kv_len < width {
                    return Err(format!(
                        "group {index} decode request (kv_len {}, query_len {}) needs query_len {width} \
                         and kv_len >= it",
                        decode.kv_len, decode.query_len
                    ));
                }
            }
            let decode_tokens = group.decode_requests.len() as u32 * width;
            if group.decode_tokens != decode_tokens
                || group.batch_tokens != group.prefill_tokens + decode_tokens
            {
                return Err(format!("group {index} token counts are inconsistent"));
            }
        }
        self.target.check_input(&verify_as_decode_rows(batch))
    }

    fn total_kv_bytes_per_token(&self) -> u64 {
        self.target.total_kv_bytes_per_token() + self.draft_kv_bytes_per_token
    }

    fn recurrent_state_bytes_per_request(&self) -> u64 {
        self.target.recurrent_state_bytes_per_request() + self.draft_kv_bytes_per_request
    }

    fn recurrent_checkpoint_interval_tokens(&self) -> u32 {
        self.target.recurrent_checkpoint_interval_tokens()
    }

    fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    fn gpus_per_replica(&self) -> u16 {
        self.target.gpus_per_replica()
    }

    fn num_attn_dp_groups(&self) -> u16 {
        self.target.num_attn_dp_groups()
    }

    fn num_attn_shards(&self) -> u16 {
        self.target.num_attn_shards()
    }

    /// DFlash drafts `draft_tokens` mask queries per request beside its own
    /// rows; vLLM reserves them in the scheduling budget.
    fn drafting_slots_per_request(&self) -> u32 {
        self.draft_tokens
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    fn eval_speculative_iter(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        self.eval_checked(batch, slots, scratch, None)
    }

    fn eval_speculative_iter_with_inputs(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        let total = self.eval_checked(batch, slots, scratch, Some(inputs));
        assert_eq!(
            inputs.len(),
            self.n_slots,
            "slot inputs must align with compiled slots"
        );
        total
    }
}

/// Wrap a built Flash `target` with one DFlash2 proposer per attention DP group.
///
/// `draft_kv_full_attention` true bills the draft's KV per token, uncapped by
/// its window, as vLLM allocated it for GLM-5.3 (`KV cache page sizes cannot be
/// unified`); false holds `sliding_window` tokens of it per request instead.
pub fn build<T: Glm53FlashVerifyTarget>(
    name: &str,
    target: T,
    max_model_len: u32,
    draft_tokens: u32,
    draft: &Dflash2DraftResolved,
    draft_kv_full_attention: bool,
    bridge: &PerfApiBridge,
) -> Result<Glm53FlashVllmNvfp4Dflash2Model<T>, BuildError> {
    if draft_tokens == 0 {
        return Err(BuildError::FitFailed {
            kind: ARCH_KIND,
            reason: "DFlash2 draft_tokens must be positive".to_string(),
        });
    }
    let groups = usize::from(target.num_attn_dp_groups().max(1));
    let drafts = (0..groups)
        .map(|_| Dflash2DraftStage::build(format!("{name}.dflash2"), draft, bridge))
        .collect::<Result<Vec<_>, _>>()?;
    let draft_bytes = draft_kv_bytes_per_token(draft)?;
    let (draft_kv_bytes_per_token, draft_kv_bytes_per_request) = if draft_kv_full_attention {
        (draft_bytes, 0)
    } else {
        (0, draft_bytes * u64::from(draft.sliding_window))
    };
    let mut model = Glm53FlashVllmNvfp4Dflash2Model {
        target,
        drafts,
        draft_tokens,
        max_model_len,
        draft_kv_bytes_per_token,
        draft_kv_bytes_per_request,
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::{SpeculativeArchGroupInput, SpeculativeDecodeInput};

    #[test]
    fn each_verify_row_becomes_a_decode_row_at_its_own_context() {
        // Two decodes at width 4 beside one prefill chunk: the target must see
        // eight decode rows ending at each request's final KV length, and the
        // prefill chunk unchanged.
        let input = SpeculativeArchInput {
            draft_tokens: 3,
            groups: vec![SpeculativeArchGroupInput {
                batch_tokens: 16 + 8,
                prefill_tokens: 16,
                decode_tokens: 8,
                prefill_chunk_pairs: vec![(100, 16)],
                decode_requests: vec![
                    SpeculativeDecodeInput {
                        kv_len: 10,
                        query_len: 4,
                    },
                    SpeculativeDecodeInput {
                        kv_len: 50,
                        query_len: 4,
                    },
                ],
                total_kv_len: 6 + 46,
            }],
            tokens_per_source_rank: vec![24],
        };
        let rows = verify_as_decode_rows(&input);
        let group = &rows.groups[0];
        assert_eq!(group.decode_kv_lens, vec![7, 8, 9, 10, 47, 48, 49, 50]);
        assert_eq!(group.decode_tokens, 8);
        assert_eq!(group.prefill_tokens, 16);
        assert_eq!(group.batch_tokens, 24);
        assert_eq!(group.prefill_chunk_pairs, vec![(100, 16)]);
        assert_eq!(group.total_kv_len, 52);
        assert_eq!(rows.tokens_per_source_rank, vec![24]);
    }
}
