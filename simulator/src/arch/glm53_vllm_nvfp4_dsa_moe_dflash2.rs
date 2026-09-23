//! GLM-5.3 NVFP4 with a DFlash2 block-parallel proposer.
//!
//! The target is the GLM-5.2 graph. That is a measured claim, not a
//! convenience: `tests/test_model_work.py` pins that the GLM-5.3 and GLM-5.2
//! NVFP4 configs agree on every field but `transformers_version` and produce
//! identical parameter counts, and the phase-0 DFlash2 capture (B200, tp=4)
//! shows a `forward` phase whose kernels are the GLM-5.2 ones -- NVFP4 MoE `bmm_E2m1_*`, `rmsNormLamport` +
//! `twoshotAllreduce`, MLA sparse `fmhaSm100f*`. So this model composes
//! [`Glm52TargetForward`] unchanged and owns only the proposer.
//!
//! # How DFlash2 differs from the MTP proposer
//!
//! [`Glm52MtpDraftStage`](super::glm52_vllm_nvfp4_dsa_moe) runs one whole-batch
//! pass and folds `draft_tokens - 1` recurrent passes after it, because MTP
//! proposes one token at a time and feeds each back. DFlash2 drafts a whole
//! block in **one** forward pass, so there is no recurrence, no fold over draft
//! positions, and none of the `mean_advance` reasoning that a recurrence forces
//! (there is no "step `i` runs after `i` positions were appended" here).
//!
//! What replaces it is an asymmetry of *shape*. One draft call runs two pieces
//! of work whose row counts differ by more than an order of magnitude:
//!
//! - the **context-KV precomputation** is prefill-shaped, over the target rows
//!   this step scheduled, and is nearly independent of the draft layer count
//!   because it hand-fuses across layers;
//! - the **draft forward** is decode-shaped, over `requests * (1 + draft_tokens)`
//!   query rows, and is an ordinary six-layer stack.
//!
//! Billing them at one row count would be wrong in both directions, so they are
//! separate stages.
//!
//! # Measured share
//!
//! From the same capture (B200, tp=4, 296 decode iterations): the whole draft
//! is 1.61 ms of a 16.90 ms decode iteration — 9.5% — against the target
//! forward's 15.02 ms. No single new kernel exceeds 2.1% of an iteration, which
//! is why this arch composes existing L1 leaves rather than introducing new
//! ones.

use std::sync::Arc;

use crate::arch::contract::{SpeculativeArchInput, SpeculativeUnifiedModel};
use crate::arch::glm52_model_cfg::Glm52MtpMode;
use crate::arch::glm52_vllm_nvfp4_dsa_moe::{
    fit_failed, normalize_speculative_input, state_bytes_per_token, Glm52TargetForward,
    Glm52VllmNvfp4DsaMoeResolved, NormalizedBatch,
};
use crate::op::Op;
use crate::timing::kernels::{AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, Probe, SlotInput,
};
use crate::worklet::{
    Dflash2ContextKvLocalWorklet, Dflash2ContextKvLocalWorkletInput,
    Dflash2ContextKvLocalWorkletResolved, Dflash2DraftAttnLocalWorklet,
    Dflash2DraftAttnLocalWorkletInput, Dflash2DraftAttnLocalWorkletResolved,
    Dflash2DraftFfnLocalWorklet, Dflash2DraftFfnLocalWorkletInput,
    Dflash2DraftFfnLocalWorkletResolved, Dflash2SelectorLocalWorklet,
    Dflash2SelectorLocalWorkletInput, Dflash2SelectorLocalWorkletResolved,
};

/// Everything the proposer needs that the target config does not carry. The
/// draft is a separate checkpoint, so none of it can be derived from the GLM-5.2
/// expansion.
#[derive(Clone, Debug)]
pub struct Dflash2DraftResolved {
    /// `sliding_window` from the draft checkpoint. The draft's attention never
    /// sees more context than this, however long the request is.
    pub sliding_window: u32,
    /// Its `num_draft_layers` is the layer count of the whole stack: the
    /// context-KV projection is hand-fused across exactly those layers.
    pub context_kv: Dflash2ContextKvLocalWorkletResolved,
    pub draft_attn: Dflash2DraftAttnLocalWorkletResolved,
    pub draft_ffn: Dflash2DraftFfnLocalWorkletResolved,
    pub selector: Dflash2SelectorLocalWorkletResolved,
    pub tp_allreduce: AllReduceKernelConfig,
}

/// The proposer: one context-KV precomputation, `num_draft_layers` identical
/// decoder layers, and the selector head.
struct Dflash2DraftStage {
    name: String,
    num_draft_layers: u32,
    sliding_window: u32,
    context_kv: Dflash2ContextKvLocalWorklet,
    draft_attn: Dflash2DraftAttnLocalWorklet,
    draft_ffn: Dflash2DraftFfnLocalWorklet,
    selector: Dflash2SelectorLocalWorklet,
    attn_allreduce: Op<AllReduceKernel>,
    ffn_allreduce: Op<AllReduceKernel>,
    /// Each collective reduces one hidden vector per query row.
    allreduce_bytes_per_token: u32,
}

impl Dflash2DraftStage {
    fn build(
        name: String,
        resolved: &Dflash2DraftResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let context_kv = Dflash2ContextKvLocalWorklet::build(
            format!("{name}.context_kv"),
            resolved.context_kv.clone(),
            bridge,
        )?;
        let draft_attn = Dflash2DraftAttnLocalWorklet::build(
            format!("{name}.layer.attn"),
            resolved.draft_attn.clone(),
            bridge,
        )?;
        let draft_ffn = Dflash2DraftFfnLocalWorklet::build(
            format!("{name}.layer.ffn"),
            resolved.draft_ffn.clone(),
            bridge,
        )?;
        let selector = Dflash2SelectorLocalWorklet::build(
            format!("{name}.selector"),
            resolved.selector.clone(),
            bridge,
        )?;
        let attn_allreduce = build_atomic(
            format!("{name}.layer.attn_allreduce"),
            resolved.tp_allreduce.clone(),
            AllReduceKernel::build,
            bridge,
        )?;
        let ffn_allreduce = build_atomic(
            format!("{name}.layer.ffn_allreduce"),
            resolved.tp_allreduce.clone(),
            AllReduceKernel::build,
            bridge,
        )?;
        Ok(Self {
            name,
            num_draft_layers: resolved.context_kv.raw_cfg.num_draft_layers,
            sliding_window: resolved.sliding_window,
            context_kv,
            draft_attn,
            draft_ffn,
            selector,
            attn_allreduce,
            ffn_allreduce,
            allreduce_bytes_per_token: resolved
                .draft_attn
                .input_add_rms_norm
                .hidden
                .get()
                .checked_mul(u32::from(
                    resolved.draft_attn.input_add_rms_norm.dtype.size_bytes(),
                ))
                .ok_or_else(|| fit_failed("DFlash2 all-reduce byte rate overflows u32"))?,
        })
    }

    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let layers = self.num_draft_layers;
        CostNode::Sum(vec![
            CostNode::Labeled {
                label: format!(
                    "{}.context_kv [prefill-shaped; target rows this step]",
                    self.name
                ),
                child: Box::new(self.context_kv.compile(builder)),
            },
            CostNode::Labeled {
                label: format!(
                    "{}.layers [{layers} identical layers; block-parallel, one pass]",
                    self.name
                ),
                // Every draft layer sees the same query rows and -- because the
                // window caps context identically for all of them -- the same
                // `(q_len, kv_len)` rectangle. The fold is exact, not a mean
                // over advancing contexts the way a recurrent proposer needs.
                child: Box::new(CostNode::Scale {
                    n: layers,
                    child: Box::new(CostNode::Sum(vec![
                        self.draft_attn.compile(builder),
                        self.attn_allreduce.compile(builder),
                        self.draft_ffn.compile(builder),
                        self.ffn_allreduce.compile(builder),
                    ])),
                }),
            },
            CostNode::Labeled {
                label: format!("{}.selector [scored edge lattice]", self.name),
                child: Box::new(self.selector.compile(builder)),
            },
        ])
    }

    fn eval(&self, batch: &NormalizedBatch, draft_tokens: u32, ev: &mut Evaluator) {
        let group = &batch.groups[0];
        let query_width = draft_tokens.saturating_add(1);
        let query_tokens = group.request_count.saturating_mul(query_width);

        // The proposer consumes the target's hidden states for every row this
        // step scheduled -- prefilling and verifying alike.
        self.context_kv.eval(
            &Dflash2ContextKvLocalWorkletInput {
                context_tokens: group.batch_tokens,
            },
            ev,
        );

        // One rectangle per request: the whole query block against the context
        // the sliding window admits.
        let rectangles = group
            .endpoint_context_lens
            .iter()
            .map(|&context| (query_width, context.min(self.sliding_window)))
            .collect::<Vec<_>>();
        self.draft_attn.eval(
            &Dflash2DraftAttnLocalWorkletInput {
                query_tokens,
                rectangles,
            },
            ev,
        );
        eval_atomic_or_zero(
            &self.attn_allreduce,
            self.allreduce_input(query_tokens),
            query_tokens == 0,
            ev,
        );
        self.draft_ffn
            .eval(&Dflash2DraftFfnLocalWorkletInput { query_tokens }, ev);
        eval_atomic_or_zero(
            &self.ffn_allreduce,
            self.allreduce_input(query_tokens),
            query_tokens == 0,
            ev,
        );

        // Only drafted positions reach the selector; the bonus token is the
        // target's, not a proposal.
        self.selector.eval(
            &Dflash2SelectorLocalWorkletInput {
                scored_rows: group.request_count.saturating_mul(draft_tokens),
            },
            ev,
        );
    }

    fn allreduce_input(&self, query_tokens: u32) -> AllReduceKernelInput {
        AllReduceKernelInput {
            message_size_bytes: u64::from(query_tokens)
                .saturating_mul(u64::from(self.allreduce_bytes_per_token)),
        }
    }
}

/// GLM-5.3 NVFP4 deployed with a DFlash2 proposer in front of it.
///
/// A sibling of the GLM-5.2 speculative model, not a wrapper: the two compile
/// separate trees, so a DFlash2 iteration's cost log carries its own draft
/// stages as real slots and cannot be mistaken for an MTP one.
pub struct Glm53VllmNvfp4DsaMoeDflash2Model {
    target: Glm52TargetForward,
    draft: Dflash2DraftStage,
    /// Candidate positions drafted per request per iteration. Not an `Option`:
    /// this model always speculates. The compiled tree assumes this depth.
    draft_tokens: u32,
    max_model_len: u32,
    num_attn_dp_groups: u16,
    num_attn_shards: u16,
    total_state_bytes_per_token: u64,
    /// Per-token bytes the draft's own KV adds to the target's, summed over the
    /// attention ranks. See [`Self::draft_kv_bytes_per_token`].
    draft_kv_bytes_per_token: u64,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

impl Glm53VllmNvfp4DsaMoeDflash2Model {
    /// The draft's KV is billed **per token**, uncapped by its window.
    ///
    /// The window bounds what the draft *attends to*, so the natural guess is
    /// that it also bounds what the draft *stores* -- a flat per-request reserve
    /// of `sliding_window` tokens instead of a per-token addend. The capture
    /// says otherwise, in one line at startup:
    ///
    /// ```text
    /// kv_cache_utils.py:1868] KV cache page sizes cannot be unified; treating
    /// sliding-window layers as full attention for cache allocation.
    /// Sliding-window attention compute is unchanged.
    /// ```
    ///
    /// The draft's six layers cannot share a page size with the target's MLA
    /// latent, so vLLM gives up on unifying them and allocates them as full
    /// attention out of the one pool it reports as `GPU KV cache size`. The
    /// window survives only on the compute side, which is exactly where this
    /// model keeps it: the attention rectangles cap `kv_len` at
    /// `sliding_window` while this addend does not.
    ///
    /// Folding it into the per-token scalar is therefore both the faithful
    /// model and the one the worker contract already carries -- no per-request
    /// reserve has to be subtracted in `build_speculative_worker`.
    pub fn draft_kv_bytes_per_token(&self) -> u64 {
        self.draft_kv_bytes_per_token
    }

    fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let mut children = self.target.compile_children(&mut builder);
        children.push(CostNode::Labeled {
            label: format!(
                "{} [DFlash2 proposer; draft_tokens={}; query width={}; one pass]",
                self.draft.name,
                self.draft_tokens,
                self.draft_tokens + 1
            ),
            child: Box::new(self.draft.compile(&mut builder)),
        });
        builder.finish(CostNode::Labeled {
            label: format!(
                "GLM-5.3 NVFP4 + DFlash2 [ep={}; draft_tokens={}; verify_width={}; max_model_len={}]",
                self.num_attn_shards,
                self.draft_tokens,
                self.draft_tokens + 1,
                self.max_model_len
            ),
            child: Box::new(CostNode::Sum(children)),
        })
    }

    fn eval_into(&self, input: &SpeculativeArchInput, ev: &mut Evaluator) {
        let batch = normalize_speculative_input(input, self.draft_tokens, self.max_model_len)
            .unwrap_or_else(|reason| {
                panic!("invalid Glm53VllmNvfp4DsaMoeDflash2Model input: {reason}")
            });
        self.target.eval(&batch, ev);
        self.draft.eval(&batch, self.draft_tokens, ev);
    }
}

impl SpeculativeUnifiedModel for Glm53VllmNvfp4DsaMoeDflash2Model {
    fn total_kv_bytes_per_token(&self) -> u64 {
        self.total_state_bytes_per_token
    }

    fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    fn gpus_per_replica(&self) -> u16 {
        self.target.ep_size
    }

    fn num_attn_dp_groups(&self) -> u16 {
        self.num_attn_dp_groups
    }

    fn num_attn_shards(&self) -> u16 {
        self.num_attn_shards
    }

    /// DFlash drafts one bonus query plus `draft_tokens` mask queries per
    /// request; vLLM reserves the mask queries in the scheduling budget.
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
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut evaluator = Evaluator::new(slots);
        self.eval_into(batch, &mut evaluator);
        assert_eq!(
            evaluator.filled(),
            self.n_slots,
            "eval must fill every compiled slot"
        );
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }

    fn eval_speculative_iter_with_inputs(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut evaluator = Evaluator::with_inputs(slots, inputs);
        self.eval_into(batch, &mut evaluator);
        assert_eq!(
            evaluator.filled(),
            self.n_slots,
            "eval must fill every compiled slot"
        );
        let total = CostTree::aggregate(&self.cost_flat, slots, scratch);
        assert_eq!(
            inputs.len(),
            self.n_slots,
            "slot inputs must align with compiled slots"
        );
        total
    }
}

pub fn build_dflash2(
    name: String,
    resolved: Glm52VllmNvfp4DsaMoeResolved,
    draft: Dflash2DraftResolved,
    bridge: &PerfApiBridge,
) -> Result<Glm53VllmNvfp4DsaMoeDflash2Model, BuildError> {
    let draft_tokens = resolved
        .raw_cfg
        .speculative_draft_tokens
        .ok_or_else(|| fit_failed("build_dflash2 requires a speculative recipe"))?;
    if draft_tokens == 0 {
        return Err(fit_failed("DFlash2 draft_tokens must be positive"));
    }
    let ep_size = resolved.raw_cfg.parallel.ep_size;
    let max_model_len = resolved.raw_cfg.parallel.max_model_len;
    let draft_kv_bytes_per_token = draft_kv_bytes_per_token(&draft)?;
    let target = Glm52TargetForward::build(name.clone(), &resolved, bridge)?;
    let stage = Dflash2DraftStage::build(format!("{name}.dflash2"), &draft, bridge)?;

    let mut model = Glm53VllmNvfp4DsaMoeDflash2Model {
        target,
        draft: stage,
        draft_tokens,
        max_model_len,
        num_attn_dp_groups: 1,
        num_attn_shards: ep_size,
        // `Glm52MtpMode::Off` is right and not an oversight: the proposer is a
        // separate checkpoint, so it adds none of MTP's *target-side* state.
        // What it does add is its own six GQA layers, which vLLM allocates out
        // of the same pool -- see `draft_kv_bytes_per_token`.
        total_state_bytes_per_token: state_bytes_per_token(ep_size, Glm52MtpMode::Off)?
            .checked_add(draft_kv_bytes_per_token)
            .ok_or_else(|| fit_failed("total state bytes per token overflow u64"))?,
        draft_kv_bytes_per_token,
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

/// `layers * kv_heads * head_dim * 2 (K and V) * kv_dtype`, summed over the
/// attention ranks so it matches the worker's physical-total KV contract.
///
/// `sliding_window` is deliberately absent -- see
/// [`Glm53VllmNvfp4DsaMoeDflash2Model::draft_kv_bytes_per_token`].
pub(crate) fn draft_kv_bytes_per_token(draft: &Dflash2DraftResolved) -> Result<u64, BuildError> {
    let append = &draft.context_kv.context_kv_append;
    let per_rank_per_token = u64::from(draft.context_kv.raw_cfg.num_draft_layers)
        .checked_mul(u64::from(append.num_kv_heads.get()))
        .and_then(|value| value.checked_mul(u64::from(append.head_dim.get())))
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(u64::from(append.kv_dtype.size_bytes())))
        .ok_or_else(|| fit_failed("DFlash2 draft KV bytes per token overflow u64"))?;
    let ranks = u64::from(draft.context_kv.raw_cfg.tp_size);
    per_rank_per_token
        .checked_mul(ranks)
        .ok_or_else(|| fit_failed("DFlash2 draft KV bytes per token overflows u64"))
}

fn build_atomic<K, C, F>(
    name: String,
    config: C,
    build_kernel: F,
    bridge: &PerfApiBridge,
) -> Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> std::result::Result<K, BuildError>,
{
    Ok(Op::new(
        name.clone(),
        Arc::new(build_kernel(name, config, bridge)?),
    ))
}

fn eval_atomic_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        op.kernel.eval(&input)
    };
    ev.push(metrics, || input.clone().into());
}
