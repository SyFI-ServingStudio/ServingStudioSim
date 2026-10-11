//! GLM-5.3-Flash NVFP4 with a DFlash2 block-parallel proposer.
//!
//! A composition of two models:
//!
//! - the **target** is the GLM-5.3-Flash graph, either the TP/EP one
//!   ([`Glm53FlashVllmModel`]) or the DP-attention/EP one
//!   ([`Glm53FlashDpAttnEpModel`]), compiled at verify width
//!   `draft_tokens + 1`;
//! - the **proposer** is the `incoai/GLM-5.3-Flash-DFlash2` checkpoint
//!   (five GQA layers, hidden 4096, block 8) priced through the shared
//!   [`Dflash2DraftStage`], one per attention DP rank.
//!
//! # The verify pass
//!
//! A verify step scores `draft_tokens + 1` query rows per resident request,
//! and the target takes each request once with that many rows. Each kernel
//! bills the work it does on them:
//!
//! - the DSA indexer scores a request's rows in one DeepGEMM paged MQA-logits
//!   group (`next_n` = width), reading the request's index keys once; top-k,
//!   sparse MLA and every projection run per row;
//! - KDA verifies with RecoverSSM: the verify reads the request's checkpoint
//!   state once and caches a correction and a key/gate vector per row, and a
//!   commit folds the accepted rows into the state with one read and one write.
//!   vLLM commits once per KDA KV-cache group (9, 9, 8 and 8 layers), and the
//!   commit of a rank's requests costs their mean committed rows (the target
//!   token plus the accepted drafts). The store holds one state plus those
//!   per-row vectors, not a state per row;
//! - MoE routing groups the request's consecutive rows (`verify_width` of the
//!   expert demand).
//!
//! A request whose first step after a P/D transfer has no draft yet is billed
//! as a full-width request. Nothing here was aligned against a speculative
//! Flash capture.
//!
//! # The proposer
//!
//! Under attention DP each rank drafts its own requests at TP1, and the ranks
//! draft concurrently, so the draft costs the slowest rank's call. Every draft
//! layer attends a 2048-token sliding window, and the draft's KV is held as
//! that window per request: the store charges it beside the KDA state, not per
//! context token. (vLLM's GLM-5.3 build allocated the window layers as full
//! attention because their pages could not be unified with the MLA latent; that
//! fallback needs every layer to be attention and is unavailable beside KDA, so
//! a Flash engine pages the draft as its own group.)

use crate::arch::contract::{
    ArchGroupInput, IterwiseUnifiedModel, SpeculativeArchInput, SpeculativeUnifiedModel,
    UnifiedArchInput,
};
use crate::arch::glm53_flash_vllm_fp8_dp_attn_ep_moe::Glm53FlashDpAttnEpModel;
use crate::arch::glm53_flash_vllm_fp8_kda_dsa_moe::{
    atomic, push, Glm53FlashVllmModel, KdaCommitPlan,
};
use crate::arch::glm53_vllm_nvfp4_dsa_moe_dflash2::{
    draft_kv_bytes_per_token, Dflash2DraftResolved, Dflash2DraftStage,
};
use crate::op::Op;
use crate::timing::kernels::{
    KdaRecoverSSMCommitKernel, KdaRecoverSSMCommitKernelConfig, KdaRecoverSSMCommitKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};

const ARCH_KIND: &str = "glm53_flash_vllm_nvfp4_dflash2";

/// A Flash graph a DFlash2 verify pass can run on: compiled as one node beside
/// the draft, and filled from an ordinary unified batch.
pub trait Glm53FlashVerifyTarget: IterwiseUnifiedModel {
    /// Query rows per decode request the graph was compiled for.
    fn verify_width(&self) -> u32;
    /// The RecoverSSM commit its KDA layers need after a verify.
    fn kda_commit(&self) -> Option<&KdaCommitPlan>;
    fn compile_target(&self, builder: &mut CostTreeBuilder) -> CostNode;
    fn eval_target(&self, input: &UnifiedArchInput, ev: &mut Evaluator);
}

impl Glm53FlashVerifyTarget for Glm53FlashVllmModel {
    fn verify_width(&self) -> u32 {
        self.verify_width
    }

    fn kda_commit(&self) -> Option<&KdaCommitPlan> {
        self.kda_commit.as_ref()
    }

    fn compile_target(&self, builder: &mut CostTreeBuilder) -> CostNode {
        self.compile_root(builder)
    }

    fn eval_target(&self, input: &UnifiedArchInput, ev: &mut Evaluator) {
        self.eval_into(input, ev);
    }
}

impl Glm53FlashVerifyTarget for Glm53FlashDpAttnEpModel {
    fn verify_width(&self) -> u32 {
        self.verify_width
    }

    fn kda_commit(&self) -> Option<&KdaCommitPlan> {
        self.kda_commit.as_ref()
    }

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
    /// One RecoverSSM commit call per KDA KV-cache group, run on every
    /// attention DP rank.
    commit: Vec<Op<KdaRecoverSSMCommitKernel>>,
    /// One proposer per attention DP group, in group order.
    drafts: Vec<Dflash2DraftStage>,
    /// Candidate positions drafted per request per iteration. The compiled tree
    /// assumes this depth.
    draft_tokens: u32,
    max_model_len: u32,
    /// The draft's sliding-window KV held per request, summed over one group's
    /// attention shards as the worker's capacity contract counts the target's.
    draft_kv_bytes_per_request: u64,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

impl<T: Glm53FlashVerifyTarget> Glm53FlashVllmNvfp4Dflash2Model<T> {
    fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let target = self.target.compile_target(&mut builder);
        // Slots are filled in compile order: target, commit, then the drafts,
        // as `eval_into` pushes them.
        let commit_ranks: Vec<CostNode> = self
            .drafts
            .iter()
            .map(|_| {
                CostNode::Sum(
                    self.commit
                        .iter()
                        .map(|op| op.compile(&mut builder))
                        .collect(),
                )
            })
            .collect();
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
                "GLM-5.3-Flash NVFP4 + DFlash2 [draft_tokens={}; verify width={width}; \
                 {} draft rank(s)]",
                self.draft_tokens,
                self.drafts.len()
            ),
            child: Box::new(CostNode::Sum(vec![
                CostNode::Labeled {
                    label: format!("target verify [{width} query rows per resident request]"),
                    child: Box::new(target),
                },
                CostNode::Labeled {
                    label: format!(
                        "kda recoverssm commit [{} KDA cache group call(s); Max over attention DP \
                         ranks]",
                        self.commit.len()
                    ),
                    child: Box::new(CostNode::Max {
                        overlap: 1.0,
                        children: commit_ranks,
                    }),
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
        self.target.eval_target(&verify_batch(input), ev);
        for group in &input.groups {
            let requests = group.decode_requests.len() as u32;
            let committed: u32 = group.decode_requests.iter().map(|d| d.committed_rows).sum();
            let commit = KdaRecoverSSMCommitKernelInput {
                batch_size: requests,
                mean_commit_len: if requests == 0 {
                    1.0
                } else {
                    f64::from(committed) / f64::from(requests)
                },
            };
            for op in &self.commit {
                if requests == 0 {
                    ev.push(LeafMetrics::ZERO, || commit.clone().into());
                } else {
                    push(op, commit.clone(), ev);
                }
            }
        }
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
/// decode request as one entry at its final KV length that presents the
/// target's verify width of query rows.
fn verify_batch(input: &SpeculativeArchInput) -> UnifiedArchInput {
    UnifiedArchInput {
        groups: input
            .groups
            .iter()
            .map(|group| ArchGroupInput {
                batch_tokens: group.batch_tokens,
                prefill_tokens: group.prefill_tokens,
                decode_tokens: group.decode_tokens,
                prefill_chunk_pairs: group.prefill_chunk_pairs.clone(),
                decode_kv_lens: group.decode_requests.iter().map(|d| d.kv_len).collect(),
                total_kv_len: group.total_kv_len,
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
                if decode.query_len != width
                    || decode.kv_len < width
                    || !(1..=width).contains(&decode.committed_rows)
                {
                    return Err(format!(
                        "group {index} decode request (kv_len {}, query_len {}, committed_rows {}) \
                         needs query_len {width}, kv_len >= it and 1..={width} committed rows",
                        decode.kv_len, decode.query_len, decode.committed_rows
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
        self.target.check_input(&verify_batch(batch))
    }

    fn total_kv_bytes_per_token(&self) -> u64 {
        self.target.total_kv_bytes_per_token()
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

/// Wrap a built Flash `target` with one DFlash2 proposer per attention DP group,
/// holding `draft.sliding_window` tokens of the draft's KV per request.
pub fn build<T: Glm53FlashVerifyTarget>(
    name: &str,
    target: T,
    max_model_len: u32,
    draft_tokens: u32,
    draft: &Dflash2DraftResolved,
    bridge: &PerfApiBridge,
) -> Result<Glm53FlashVllmNvfp4Dflash2Model<T>, BuildError> {
    if draft_tokens == 0 || target.verify_width() != draft_tokens + 1 {
        return Err(BuildError::FitFailed {
            kind: ARCH_KIND,
            reason: format!(
                "DFlash2 draft_tokens {draft_tokens} needs a target compiled at verify width \
                 draft_tokens + 1, got {}",
                target.verify_width()
            ),
        });
    }
    let plan = target.kda_commit().ok_or_else(|| BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: "the verify target compiled no RecoverSSM commit plan".to_string(),
    })?;
    let commit = plan
        .cache_group_layers
        .iter()
        .enumerate()
        .map(|(index, &layers)| {
            let verify = &plan.verify;
            atomic(
                name,
                &format!("kda_recoverssm_commit_group{index}"),
                KdaRecoverSSMCommitKernelConfig {
                    backends: verify.backends.clone(),
                    gpu_name: verify.gpu_name.clone(),
                    num_heads: verify.num_heads.clone(),
                    head_dim: verify.head_dim.clone(),
                    spec_query_len: verify.spec_query_len.clone(),
                    num_layers: layers.into(),
                    dtype: verify.dtype,
                },
                KdaRecoverSSMCommitKernel::build,
                bridge,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let groups = usize::from(target.num_attn_dp_groups().max(1));
    let drafts = (0..groups)
        .map(|_| Dflash2DraftStage::build(format!("{name}.dflash2"), draft, bridge))
        .collect::<Result<Vec<_>, _>>()?;
    let draft_kv_bytes_per_request = draft_kv_bytes_per_token(draft)?
        .checked_mul(u64::from(draft.sliding_window))
        .ok_or_else(|| BuildError::FitFailed {
            kind: ARCH_KIND,
            reason: "DFlash2 draft window KV overflows u64".to_string(),
        })?;
    let mut model = Glm53FlashVllmNvfp4Dflash2Model {
        target,
        commit,
        drafts,
        draft_tokens,
        max_model_len,
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
    fn each_verify_request_reaches_the_target_once_at_its_final_context() {
        // Two decodes at width 4 beside one prefill chunk: the target sees two
        // decode requests at their final KV lengths, and eight decode tokens.
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
                        committed_rows: 2,
                    },
                    SpeculativeDecodeInput {
                        kv_len: 50,
                        query_len: 4,
                        committed_rows: 4,
                    },
                ],
                total_kv_len: 6 + 46,
            }],
            tokens_per_source_rank: vec![24],
        };
        let target = verify_batch(&input);
        let group = &target.groups[0];
        assert_eq!(group.decode_kv_lens, vec![10, 50]);
        assert_eq!(group.decode_tokens, 8);
        assert_eq!(group.prefill_tokens, 16);
        assert_eq!(group.batch_tokens, 24);
        assert_eq!(group.prefill_chunk_pairs, vec![(100, 16)]);
        assert_eq!(group.total_kv_len, 52);
        assert_eq!(target.tokens_per_source_rank, vec![24]);
    }
}
