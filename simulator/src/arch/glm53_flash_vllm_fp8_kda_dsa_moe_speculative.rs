//! GLM-5.3-Flash FP8 on B200 (TP4 / EP4, the vLLM fork) deployed with a
//! speculative drafter: the checkpoint's own MTP layer, or the separate
//! DFlash2 block-parallel proposer (`incoai/GLM-5.3-Flash-DFlash2`).
//!
//! The target is [`Glm53FlashTargetForward`], built for a verify width
//! `q = draft_tokens + 1`:
//!
//! - every decode request submits `q` query rows, so every non-attention
//!   kernel runs `B * q` rows (graph padded to vLLM's spec-adjusted capture
//!   sizes, each rounded up to a multiple of `q`:
//!   `CompilationConfig.adjust_cudagraph_sizes_for_spec_decode`);
//! - KDA takes the spec-verify path (`causal_conv1d_update` and
//!   `fused_recurrent_kda` over `q` tokens per request with `q` state slots,
//!   `glm5next/common/kda.py`), see `Glm53KdaAttnLocalWorklet`;
//! - DSA runs its decode kernels on the `[B, q]` grid
//!   (`Glm53DsaSpecMode::verify`);
//! - the routed experts see `q` consecutive rows per request, so their demand
//!   is folded at group size `q`;
//! - the head scores every verify row, and the rejection sampler reads them.
//!
//! MTP (`glm5next/common/mtp.py`): one DSA + MoE layer without mHC. Step 0
//! forwards every target row (embedding, `fused_eh_norm`, the BF16 `eh_proj`,
//! `input_layernorm`, DSA at width `q`, `post_attention_layernorm`, MoE,
//! `shared_head.norm`) and samples one greedy token per request; steps
//! `1..draft_tokens` forward one row per request and reuse step 0's top-k
//! (`index_share_for_mtp_iteration`), folded as one pass at the mean context
//! advance. `compact_topk_indices` runs once between step 0 and step 1.
//!
//! DFlash2: the target additionally produces one aux hidden state per
//! `target_layer_ids` entry (`Glm5NextModel._aux_hidden_state`, an mHC post
//! over every target row), then the proposer of
//! [`glm53_vllm_nvfp4_dsa_moe_dflash2`](super::glm53_vllm_nvfp4_dsa_moe_dflash2)
//! runs at the Flash drafter's dimensions.
//!
//! KV, as vLLM main / 0.32 allocate it ([`FlashKvAllocator::MAIN`]): the KDA
//! state is held once per verify row plus a prefill checkpoint page
//! (`MambaSpec` with `num_speculative_blocks`, `num_prefill_checkpoint_blocks`),
//! the kpool tail ring widens with the drafts, the MTP layer adds one DSA layer
//! to the MLA group, and the DFlash2 draft layers add a sliding-window group
//! of whole pool blocks ([`flash_kv_cache_layout_for`]).

use crate::arch::contract::{SpeculativeArchInput, SpeculativeUnifiedModel};
use crate::arch::glm53_flash_vllm_fp8_kda_dsa_moe::{
    atomic, build_moe_block, flash_kv_cache_layout_for, graph_padded_tokens, push,
    sorted_capture_sizes, DraftWindowKv, Glm53FlashTargetForward,
    Glm53FlashVllmResolved, MoeBlock,
    NormalizedBatch, TpAllReduce, ACTIVATION_DTYPE, MHC_BACKENDS,
};
use crate::arch::glm53_vllm_nvfp4_dsa_moe_dflash2::{
    draft_kv_bytes_per_token, Dflash2DraftResolved, Dflash2DraftStage,
};
use crate::arch::kv_layout::KvCacheLayout;
use crate::op::mhc::{MhcTerminalPostConfig, MhcTerminalPostInput, MhcTerminalPostOp};
use crate::op::Op;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, ResidualRmsNormKernel,
    ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput, RmsNormKernel, RmsNormKernelConfig,
    RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::glm53_common::elementwise;
use crate::worklet::{
    Glm53DsaAttnLocalWorklet, Glm53DsaAttnLocalWorkletInput, Glm53DsaAttnLocalWorkletResolved,
    Glm53KdaAttnLocalWorkletInput, Glm53RoutedMoeLocalWorkletResolved, SpecHeadLocalWorklet,
    SpecHeadLocalWorkletInput, SpecHeadLocalWorkletResolved, SpecSampleLocalWorklet,
    SpecSampleLocalWorkletInput, SpecSampleLocalWorkletResolved,
};

const ARCH_KIND: &str = "glm53_flash_vllm_fp8_kda_dsa_moe_speculative";
const RMS_NORM_BACKENDS: &[&str] = &["vllm_cuda"];
const BF16_GEMM_BACKENDS: &[&str] = &["torch_linear_vllm"];
const ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
/// `topk_indices_buffer` rows are int32.
const TOPK_INDEX_BYTES: u32 = 4;

/// Everything the MTP drafter needs beyond the target's resolved configs.
#[derive(Clone, Debug)]
pub struct Glm53FlashMtpResolved {
    /// Step 0: the target's verify width, full indexer.
    pub dsa_step_0: Glm53DsaAttnLocalWorkletResolved,
    /// Steps 1+: one row per request, step 0's indices reused.
    pub dsa_recurrent: Glm53DsaAttnLocalWorkletResolved,
    /// The MTP layer's routed slices at group size `q` (step 0) and 1 (steps 1+).
    pub routed_step_0: Vec<Glm53RoutedMoeLocalWorkletResolved>,
    pub routed_recurrent: Vec<Glm53RoutedMoeLocalWorkletResolved>,
    /// `fused_eh_norm`: reads the embedding and the previous hidden row (and a
    /// position), writes the concatenated `[2H]` row.
    pub eh_norm: ElementwiseKernelConfig,
    /// `eh_proj`: a replicated BF16 `nn.Linear(2H, H)` (not tensor-parallel).
    pub eh_proj: SingleGemmKernelConfig,
    pub input_norm: RmsNormKernelConfig,
    pub post_attention_norm: ResidualRmsNormKernelConfig,
    pub shared_head_norm: ResidualRmsNormKernelConfig,
    /// The target's lm_head shard with a vocab-parallel greedy argmax.
    pub head: SpecHeadLocalWorkletResolved,
    /// `compact_topk_indices`: one `selected_k` int32 row per request gathered.
    pub compact_topk: ElementwiseKernelConfig,
}

#[derive(Clone, Debug)]
pub struct Glm53FlashDflash2Resolved {
    pub draft: Dflash2DraftResolved,
    /// `target_layer_ids`: one aux mHC post each, over every target row.
    pub num_aux_layers: u32,
}

#[derive(Clone, Debug)]
pub enum Glm53FlashDrafterResolved {
    Mtp(Glm53FlashMtpResolved),
    Dflash2(Glm53FlashDflash2Resolved),
}

#[derive(Clone, Debug)]
pub struct Glm53FlashSpecResolved {
    /// Built at the verify width: KDA and DSA spec modes, routed demand folded
    /// at group size `q`.
    pub target: Glm53FlashVllmResolved,
    pub draft_tokens: u32,
    pub head: SpecHeadLocalWorkletResolved,
    pub sample: SpecSampleLocalWorkletResolved,
    pub drafter: Glm53FlashDrafterResolved,
}

pub(crate) fn norm_configs(
    resolved: &Glm53FlashVllmResolved,
) -> (RmsNormKernelConfig, ResidualRmsNormKernelConfig) {
    let model = &resolved.raw_cfg.model;
    let gpu = resolved.raw_cfg.parallel.gpu_name.clone();
    (
        RmsNormKernelConfig {
            backends: RMS_NORM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden: model.hidden.into(),
            dtype: ACTIVATION_DTYPE,
        },
        ResidualRmsNormKernelConfig {
            backends: RMS_NORM_BACKENDS.to_vec(),
            gpu_name: gpu,
            hidden: model.hidden.into(),
            dtype: ACTIVATION_DTYPE,
        },
    )
}

pub(crate) fn mtp_glue_configs(
    resolved: &Glm53FlashVllmResolved,
) -> (ElementwiseKernelConfig, SingleGemmKernelConfig, ElementwiseKernelConfig) {
    let cfg = &resolved.raw_cfg;
    let hidden = cfg.model.hidden;
    let row = hidden * ACTIVATION_DTYPE.size_bytes();
    let gpu = cfg.parallel.gpu_name.as_str();
    (
        elementwise(ELEMENTWISE_BACKENDS, gpu, 2 * row + 8, 2 * row),
        SingleGemmKernelConfig {
            backends: BF16_GEMM_BACKENDS.to_vec(),
            gpu_name: gpu.to_string(),
            n: hidden.into(),
            k: (2 * hidden).into(),
            dtype: ACTIVATION_DTYPE,
        },
        elementwise(
            ELEMENTWISE_BACKENDS,
            gpu,
            cfg.dsa.selected_k * TOPK_INDEX_BYTES,
            cfg.dsa.selected_k * TOPK_INDEX_BYTES,
        ),
    )
}

/// One forward pass of the MTP layer.
struct MtpPass {
    embedding: Op<ElementwiseKernel>,
    embedding_all_reduce: TpAllReduce,
    eh_norm: Op<ElementwiseKernel>,
    eh_proj: Op<SingleGemmKernel>,
    input_norm: Op<RmsNormKernel>,
    dsa: Glm53DsaAttnLocalWorklet,
    attn_all_reduce: TpAllReduce,
    post_attention_norm: Op<ResidualRmsNormKernel>,
    moe: MoeBlock,
    ffn_all_reduce: TpAllReduce,
    shared_head_norm: Op<ResidualRmsNormKernel>,
    head: SpecHeadLocalWorklet,
}

impl MtpPass {
    fn build(
        p: &str,
        target: &Glm53FlashVllmResolved,
        mtp: &Glm53FlashMtpResolved,
        dsa: &Glm53DsaAttnLocalWorkletResolved,
        routed: &[Glm53RoutedMoeLocalWorkletResolved],
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let cfg = &target.raw_cfg;
        Ok(Self {
            embedding: atomic(
                p,
                "embedding",
                cfg.embedding.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            embedding_all_reduce: TpAllReduce::build(p, "embedding_all_reduce", cfg, bridge)?,
            eh_norm: atomic(
                p,
                "fused_eh_norm",
                mtp.eh_norm.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            eh_proj: atomic(
                p,
                "eh_proj",
                mtp.eh_proj.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            input_norm: atomic(
                p,
                "input_layernorm",
                mtp.input_norm.clone(),
                RmsNormKernel::build,
                bridge,
            )?,
            dsa: Glm53DsaAttnLocalWorklet::build(format!("{p}.dsa"), dsa.clone(), bridge)?,
            attn_all_reduce: TpAllReduce::build(p, "attn_all_reduce", cfg, bridge)?,
            post_attention_norm: atomic(
                p,
                "post_attention_layernorm",
                mtp.post_attention_norm.clone(),
                ResidualRmsNormKernel::build,
                bridge,
            )?,
            moe: build_moe_block(&format!("{p}.moe"), target, routed, bridge)?,
            ffn_all_reduce: TpAllReduce::build(p, "ffn_all_reduce", cfg, bridge)?,
            shared_head_norm: atomic(
                p,
                "shared_head_norm",
                mtp.shared_head_norm.clone(),
                ResidualRmsNormKernel::build,
                bridge,
            )?,
            head: SpecHeadLocalWorklet::build(format!("{p}.head"), mtp.head.clone(), bridge)?,
        })
    }

    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            self.embedding.compile(builder),
            self.embedding_all_reduce.compile(builder),
            self.eh_norm.compile(builder),
            self.eh_proj.compile(builder),
            self.input_norm.compile(builder),
            self.dsa.compile(builder),
            self.attn_all_reduce.compile(builder),
            self.post_attention_norm.compile(builder),
            self.moe.compile(builder),
            self.ffn_all_reduce.compile(builder),
            self.shared_head_norm.compile(builder),
            self.head.compile(builder),
        ])
    }

    /// `rows` is the pass's graph-padded row count, `dsa` its attention input,
    /// `endpoints` the requests that sample a draft token.
    fn eval(
        &self,
        rows: u32,
        dsa: &Glm53DsaAttnLocalWorkletInput,
        endpoints: u32,
        ev: &mut Evaluator,
    ) {
        let tokens = ElementwiseKernelInput { num_tokens: rows };
        push(&self.embedding, tokens.clone(), ev);
        self.embedding_all_reduce.eval(rows, ev);
        push(&self.eh_norm, tokens, ev);
        push(&self.eh_proj, SingleGemmKernelInput { m: rows }, ev);
        push(&self.input_norm, RmsNormKernelInput { m: rows }, ev);
        self.dsa.eval(dsa, ev);
        self.attn_all_reduce.eval(rows, ev);
        push(
            &self.post_attention_norm,
            ResidualRmsNormKernelInput { m: rows },
            ev,
        );
        self.moe.eval(rows, ev);
        self.ffn_all_reduce.eval(rows, ev);
        push(
            &self.shared_head_norm,
            ResidualRmsNormKernelInput { m: rows },
            ev,
        );
        self.head
            .eval(&SpecHeadLocalWorkletInput { rows: endpoints }, ev);
    }
}

struct MtpDrafter {
    step_0: MtpPass,
    compact_topk: Op<ElementwiseKernel>,
    /// Steps `1..draft_tokens`, compiled once and folded.
    recurrent: Option<MtpPass>,
    recurrent_count: u32,
}

impl MtpDrafter {
    fn compile(&self, builder: &mut CostTreeBuilder, name: &str) -> CostNode {
        let mut passes = vec![
            CostNode::Labeled {
                label: format!("{name}.mtp.step_0 [every target row; computes the top-k]"),
                child: Box::new(self.step_0.compile(builder)),
            },
            self.compact_topk.compile(builder),
        ];
        if let Some(recurrent) = &self.recurrent {
            passes.push(CostNode::Labeled {
                label: format!(
                    "{name}.mtp.recurrent [request-endpoint pass, shared top-k; {} steps]",
                    self.recurrent_count
                ),
                child: Box::new(CostNode::Scale {
                    n: self.recurrent_count,
                    child: Box::new(recurrent.compile(builder)),
                }),
            });
        }
        CostNode::Sum(passes)
    }

    fn eval(&self, step: &SpecStep, ev: &mut Evaluator) {
        self.step_0
            .eval(step.padded_tokens, &step.dsa, step.request_count, ev);
        push(
            &self.compact_topk,
            ElementwiseKernelInput {
                num_tokens: step.request_count,
            },
            ev,
        );
        if let Some(recurrent) = &self.recurrent {
            recurrent.eval(
                step.recurrent_padded,
                &step.recurrent_dsa,
                step.request_count,
                ev,
            );
        }
    }
}

struct Dflash2Drafter {
    /// One per `target_layer_ids` entry, compiled once and folded.
    aux_post: MhcTerminalPostOp,
    num_aux_layers: u32,
    stage: Dflash2DraftStage,
}

impl Dflash2Drafter {
    fn compile(&self, builder: &mut CostTreeBuilder, draft_tokens: u32) -> CostNode {
        CostNode::Sum(vec![
            CostNode::Labeled {
                label: format!(
                    "{}.aux_hidden_states [{} target layers; mHC post over every row]",
                    self.stage.name, self.num_aux_layers
                ),
                child: Box::new(CostNode::Scale {
                    n: self.num_aux_layers,
                    child: Box::new(self.aux_post.compile(builder)),
                }),
            },
            CostNode::Labeled {
                label: format!(
                    "{} [DFlash2 proposer; draft_tokens={draft_tokens}; one pass]",
                    self.stage.name
                ),
                child: Box::new(self.stage.compile(builder)),
            },
        ])
    }

    fn eval(&self, step: &SpecStep, draft_tokens: u32, ev: &mut Evaluator) {
        self.aux_post.eval(
            &MhcTerminalPostInput {
                num_tokens: step.padded_tokens,
            },
            ev,
        );
        self.stage.eval_parts(
            step.request_count,
            step.total_tokens,
            &step.endpoint_context_lens,
            draft_tokens,
            ev,
        );
    }
}

enum Drafter {
    Mtp(MtpDrafter),
    Dflash2(Dflash2Drafter),
}

/// GLM-5.3-Flash with an MTP or DFlash2 drafter.
pub struct Glm53FlashVllmSpeculativeModel {
    pub name: String,
    tp_size: u16,
    target: Glm53FlashTargetForward,
    head: SpecHeadLocalWorklet,
    sample: SpecSampleLocalWorklet,
    drafter: Drafter,
    draft_tokens: u32,
    max_model_len: u32,
    /// Capture sizes rounded up to multiples of the verify width, sorted.
    cudagraph_capture_sizes: Vec<u32>,
    kv_cache_layout: KvCacheLayout,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

/// vLLM's spec-decode capture sizes: each size rounded up to a multiple of the
/// verify width, kept while it stays within the largest configured size
/// (`adjust_cudagraph_sizes_for_spec_decode`; sequence parallelism off).
pub(crate) fn spec_capture_sizes(sizes: &[u32], verify_width: u32) -> Vec<u32> {
    let sorted = sorted_capture_sizes(sizes);
    let Some(&largest) = sorted.last() else {
        return sorted;
    };
    if verify_width <= 1 {
        return sorted;
    }
    let mut rounded: Vec<u32> = sorted
        .iter()
        .map(|&size| size.div_ceil(verify_width) * verify_width)
        .filter(|&size| size <= largest)
        .collect();
    rounded.dedup();
    if rounded.is_empty() && verify_width <= largest {
        rounded.push(verify_width);
    }
    rounded
}

pub fn build(
    name: String,
    resolved: Glm53FlashSpecResolved,
    bridge: &PerfApiBridge,
) -> Result<Glm53FlashVllmSpeculativeModel, BuildError> {
    let target = &resolved.target;
    let cfg = &target.raw_cfg;
    let n = name.as_str();
    let draft_tokens = resolved.draft_tokens;
    if draft_tokens == 0 {
        return Err(fit_failed("draft_tokens must be positive"));
    }
    let verify_width = draft_tokens + 1;
    if target.kda.verify_width != verify_width || target.dsa.spec.verify_width != verify_width {
        return Err(fit_failed(format!(
            "target built for KDA width {} / DSA width {}, expected {verify_width}",
            target.kda.verify_width, target.dsa.spec.verify_width
        )));
    }
    let (drafter, kv_cache_layout) = match &resolved.drafter {
        Glm53FlashDrafterResolved::Mtp(mtp) => {
            let recurrent_count = draft_tokens - 1;
            (
                Drafter::Mtp(MtpDrafter {
                    step_0: MtpPass::build(
                        &format!("{n}.mtp.step_0"),
                        target,
                        mtp,
                        &mtp.dsa_step_0,
                        &mtp.routed_step_0,
                        bridge,
                    )?,
                    compact_topk: atomic(
                        n,
                        "mtp.compact_topk_indices",
                        mtp.compact_topk.clone(),
                        ElementwiseKernel::build,
                        bridge,
                    )?,
                    recurrent: (recurrent_count > 0)
                        .then(|| {
                            MtpPass::build(
                                &format!("{n}.mtp.recurrent"),
                                target,
                                mtp,
                                &mtp.dsa_recurrent,
                                &mtp.routed_recurrent,
                                bridge,
                            )
                        })
                        .transpose()?,
                    recurrent_count,
                }),
                flash_kv_cache_layout_for(target, 1, None, cfg.parallel.kernel_path.kv_allocator),
            )
        }
        Glm53FlashDrafterResolved::Dflash2(dflash) => (
            Drafter::Dflash2(Dflash2Drafter {
                aux_post: MhcTerminalPostOp::build(
                    format!("{n}.dflash2.aux_mhc_post"),
                    MhcTerminalPostConfig {
                        mhc: cfg.mhc.clone(),
                        pre_backends: MHC_BACKENDS.to_vec(),
                        fused_backends: MHC_BACKENDS.to_vec(),
                    },
                    bridge,
                )?,
                num_aux_layers: dflash.num_aux_layers,
                stage: Dflash2DraftStage::build(format!("{n}.dflash2"), &dflash.draft, bridge)?,
            }),
            flash_kv_cache_layout_for(
                target,
                0,
                Some(dflash2_draft_window(&dflash.draft, cfg.parallel.tp_size)?),
                cfg.parallel.kernel_path.kv_allocator,
            ),
        ),
    };
    let mut model = Glm53FlashVllmSpeculativeModel {
        tp_size: cfg.parallel.tp_size,
        target: Glm53FlashTargetForward::build(n, target, bridge)?,
        head: SpecHeadLocalWorklet::build(format!("{n}.head"), resolved.head.clone(), bridge)?,
        sample: SpecSampleLocalWorklet::build(
            format!("{n}.sample"),
            resolved.sample.clone(),
            bridge,
        )?,
        drafter,
        draft_tokens,
        max_model_len: cfg.parallel.max_model_len,
        cudagraph_capture_sizes: spec_capture_sizes(
            &cfg.parallel.cudagraph_capture_sizes,
            verify_width,
        ),
        kv_cache_layout,
        cost_flat: Vec::new(),
        n_slots: 0,
        name,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

impl Glm53FlashVllmSpeculativeModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        // The target's own lm_head is billed at zero rows: the verify head
        // below scores the rows, so one GEMM is not charged twice.
        let mut children = self.target.compile_children(&mut builder);
        children.push(self.head.compile(&mut builder));
        children.push(self.sample.compile(&mut builder));
        let drafter = match &self.drafter {
            Drafter::Mtp(mtp) => CostNode::Labeled {
                label: format!("MTP layer [{}-deep draft]", self.draft_tokens),
                child: Box::new(mtp.compile(&mut builder, &self.name)),
            },
            Drafter::Dflash2(dflash) => dflash.compile(&mut builder, self.draft_tokens),
        };
        children.push(drafter);
        builder.finish(CostNode::Labeled {
            label: format!(
                "{} (Glm53FlashVllmSpeculativeModel) [TP=EP{}; draft_tokens={}; verify width={}; \
                 timing_context<={}]",
                self.name,
                self.tp_size,
                self.draft_tokens,
                self.draft_tokens + 1,
                self.max_model_len
            ),
            child: Box::new(CostNode::Sum(children)),
        })
    }

    fn normalize(&self, input: &SpeculativeArchInput) -> Result<SpecStep, String> {
        normalize_speculative_input(
            input,
            self.draft_tokens,
            self.max_model_len,
            &self.cudagraph_capture_sizes,
        )
    }

    fn eval_checked(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: Option<&mut Vec<SlotInput>>,
    ) -> LeafMetrics {
        let step = self.normalize(batch).unwrap_or_else(|reason| {
            panic!("invalid Glm53FlashVllmSpeculativeModel input: {reason}")
        });
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut ev = match inputs {
            Some(inputs) => Evaluator::with_inputs(slots, inputs),
            None => Evaluator::new(slots),
        };
        self.target
            .eval(&step.target, step.padded_tokens, 0, &mut ev);
        self.head.eval(
            &SpecHeadLocalWorkletInput {
                rows: step.logits_rows,
            },
            &mut ev,
        );
        self.sample.eval(
            &SpecSampleLocalWorkletInput {
                logits_rows: step.logits_rows,
                requests: step.request_count,
            },
            &mut ev,
        );
        match &self.drafter {
            Drafter::Mtp(mtp) => mtp.eval(&step, &mut ev),
            Drafter::Dflash2(dflash) => dflash.eval(&step, self.draft_tokens, &mut ev),
        }
        assert_eq!(ev.filled(), self.n_slots, "eval must fill every compiled slot");
        drop(ev);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }
}

impl SpeculativeUnifiedModel for Glm53FlashVllmSpeculativeModel {
    fn check_input(&self, batch: &SpeculativeArchInput) -> Result<(), String> {
        self.normalize(batch).map(|_| ())
    }

    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout.clone()
    }

    fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    fn gpus_per_replica(&self) -> u16 {
        self.tp_size
    }

    fn num_attn_shards(&self) -> u16 {
        self.tp_size
    }

    /// DFlash drafts one bonus query plus `draft_tokens` mask queries per
    /// request, and vLLM reserves the mask queries in the scheduling budget;
    /// MTP reuses the verify rows.
    fn drafting_slots_per_request(&self) -> u32 {
        match self.drafter {
            Drafter::Mtp(_) => 0,
            Drafter::Dflash2(_) => self.draft_tokens,
        }
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

/// One speculative step lowered to what each pass reads.
struct SpecStep {
    /// The target verify pass (and MTP step 0, which forwards the same rows).
    target: NormalizedBatch,
    dsa: Glm53DsaAttnLocalWorkletInput,
    total_tokens: u32,
    padded_tokens: u32,
    request_count: u32,
    /// Prefill requests plus every verify row.
    logits_rows: u32,
    endpoint_context_lens: Vec<u32>,
    /// MTP steps 1+: one row per request at the folded context.
    recurrent_dsa: Glm53DsaAttnLocalWorkletInput,
    recurrent_padded: u32,
}

fn normalize_speculative_input(
    input: &SpeculativeArchInput,
    draft_tokens: u32,
    max_model_len: u32,
    capture_sizes: &[u32],
) -> Result<SpecStep, String> {
    if input.draft_tokens != draft_tokens {
        return Err(format!(
            "input draft_tokens {} must match model draft_tokens {draft_tokens}",
            input.draft_tokens
        ));
    }
    let [group] = input.groups.as_slice() else {
        return Err(format!(
            "expected exactly one tensor-parallel attention group, got {}",
            input.groups.len()
        ));
    };
    let mut prefill_tokens = 0_u32;
    let mut dsa_prefills = Vec::with_capacity(group.prefill_chunk_pairs.len());
    let mut kda_prefills = Vec::with_capacity(group.prefill_chunk_pairs.len());
    let mut endpoints =
        Vec::with_capacity(group.prefill_chunk_pairs.len() + group.decode_requests.len());
    for (request, &(prefix, append)) in group.prefill_chunk_pairs.iter().enumerate() {
        if append == 0 {
            return Err(format!("prefill request {request} append must be nonzero"));
        }
        let context = prefix
            .checked_add(append)
            .ok_or_else(|| format!("prefill request {request} context overflows u32"))?;
        if context > max_model_len {
            return Err(format!(
                "prefill request {request} context {context} exceeds max_model_len {max_model_len}"
            ));
        }
        prefill_tokens = prefill_tokens
            .checked_add(append)
            .ok_or("prefill token sum overflows u32")?;
        dsa_prefills.push((prefix, append));
        kda_prefills.push(append);
        endpoints.push(context);
    }
    if group.prefill_tokens != prefill_tokens {
        return Err(format!(
            "prefill_tokens {} must equal append sum {prefill_tokens}",
            group.prefill_tokens
        ));
    }
    let mut decode_tokens = 0_u32;
    let mut verify_kv_lens = Vec::with_capacity(group.decode_requests.len());
    for (request, decode) in group.decode_requests.iter().enumerate() {
        // A full window runs the uniform `[B, q]` kernels; acceptance truncates
        // output, not compute. A verify the scheduler clipped at
        // max_model_len leaves the uniform batch: DSA prices it on its
        // prefill path and KDA as a varlen sequence of its rows (rare: the
        // last verifies of a request ending within `draft_tokens` of the cap).
        let clipped = decode
            .check_verify_row(draft_tokens, max_model_len)
            .map_err(|reason| format!("decode request {request}: {reason}"))?;
        decode_tokens = decode_tokens
            .checked_add(decode.query_len)
            .ok_or("decode token sum overflows u32")?;
        endpoints.push(decode.kv_len);
        if clipped {
            dsa_prefills.push((decode.kv_len - decode.query_len, decode.query_len));
            kda_prefills.push(decode.query_len);
        } else {
            verify_kv_lens.push(decode.kv_len);
        }
    }
    if group.decode_tokens != decode_tokens {
        return Err(format!(
            "decode_tokens {} must equal query_len sum {decode_tokens}",
            group.decode_tokens
        ));
    }
    let total_tokens = prefill_tokens
        .checked_add(decode_tokens)
        .ok_or("batch token sum overflows u32")?;
    if group.batch_tokens != total_tokens {
        return Err(format!(
            "batch_tokens {} must equal prefill+decode {total_tokens}",
            group.batch_tokens
        ));
    }
    if total_tokens == 0 {
        return Err("an iteration must carry at least one token".into());
    }
    let request_count = group.request_count();
    let dsa = Glm53DsaAttnLocalWorkletInput {
        prefill_chunk_pairs: dsa_prefills,
        decode_kv_lens: verify_kv_lens.clone(),
    };
    let kda = Glm53KdaAttnLocalWorkletInput {
        prefill_sequence_lengths: kda_prefills,
        decode_batch_size: verify_kv_lens.len() as u32,
    };
    // Steps 1..draft_tokens run after 1..draft_tokens-1 appended positions;
    // the fold bills one evaluation at their mean advance, rounded up.
    let mean_advance = draft_tokens.div_ceil(2);
    let recurrent_dsa = Glm53DsaAttnLocalWorkletInput {
        prefill_chunk_pairs: Vec::new(),
        decode_kv_lens: endpoints
            .iter()
            .map(|&context| context.saturating_add(mean_advance).min(max_model_len))
            .collect(),
    };
    Ok(SpecStep {
        target: NormalizedBatch::from_parts(total_tokens, request_count, kda, dsa.clone()),
        dsa,
        total_tokens,
        padded_tokens: graph_padded_tokens(capture_sizes, total_tokens),
        request_count,
        logits_rows: group.prefill_chunk_pairs.len() as u32 + decode_tokens,
        endpoint_context_lens: endpoints,
        recurrent_dsa,
        // The drafter's recurrent passes replay its own graphs, captured at
        // the configured sizes (one row per request).
        recurrent_padded: request_count,
    })
}

/// The DFlash2 draft's sliding-window cache on one rank.
fn dflash2_draft_window(
    draft: &Dflash2DraftResolved,
    tp_size: u16,
) -> Result<DraftWindowKv, BuildError> {
    Ok(DraftWindowKv {
        bytes_per_token: draft_kv_bytes_per_token(draft)? / u64::from(tp_size.max(1)),
        window_tokens: draft.sliding_window,
    })
}

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::glm53_flash_vllm_fp8_kda_dsa_moe::FlashKvAllocator;
    use crate::arch::build::{dflash2_draft_resolved_for, GLM53_FLASH_DFLASH2_DIMS};
    use crate::arch::glm53_flash_vllm_fp8_kda_dsa_moe::{
        build_configs, resolve_configs, Glm53FlashKernelPath, Glm53FlashModelCfg,
        Glm53FlashVllmParallel,
    };
    use crate::timing::expert_demand::ExpertDemand;
    use crate::timing::routing::RoutingDistribution;
    use crate::worker::kv::{KvAccounting, KvAccountingOptions};
    use crate::worklet::glm53_dsa_attn_local::Glm53DsaSpecMode;
    use crate::worklet::{Glm53DsaAttnLocalWorklet, Glm53KdaAttnLocalWorklet};
    use std::path::Path;

    fn verify_target(verify_width: u32) -> Glm53FlashVllmResolved {
        let model = Glm53FlashModelCfg::from_json(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/glm53_flash.json"),
        )
        .unwrap();
        let parallel = Glm53FlashVllmParallel {
            tp_size: 4,
            enable_expert_parallel: true,
            max_model_len: 131_072,
            gpu_name: "NVIDIA B200".into(),
            cudagraph_capture_sizes: Vec::new(),
            kernel_path: Glm53FlashKernelPath::default(),
        };
        let demand = ExpertDemand::popularity(&RoutingDistribution::uniform(288), 42);
        let cfgs = build_configs(&model, &parallel, &demand).unwrap();
        let mut resolved = resolve_configs(&cfgs);
        resolved.kda =
            Glm53KdaAttnLocalWorklet::resolve_config_with_verify_width(&cfgs.kda, verify_width);
        resolved.dsa = Glm53DsaAttnLocalWorklet::resolve_config_for(
            &cfgs.dsa,
            Glm53DsaSpecMode::verify(verify_width),
        );
        resolved
    }

    fn concurrency(layout: &KvCacheLayout, max_batch_tokens: u32) -> f64 {
        let options = KvAccountingOptions {
            max_in_flight_tokens: Some(2 * max_batch_tokens),
            prefix_caching: true,
            ..Default::default()
        };
        let accounting = KvAccounting::resolve(layout, options, 60_129_542_144 * 4).unwrap();
        accounting.concurrency_at(131_072)
    }

    fn batch_tokens(k: u32) -> u32 {
        if 2048 % (k + 1) == 0 {
            2048
        } else {
            (2048 / (k + 1) + 1) * (k + 1)
        }
    }

    /// The measured Flash servers' "Maximum concurrency for 131,072 tokens per
    /// request" (KV pinned at 60129542144 bytes per GPU, prefix caching on so
    /// the KDA state runs in align mode, V2 runner with async scheduling: two
    /// batches in flight). No-spec and DFlash2 ran vLLM main 187a0eb98, MTP
    /// 0.32.0rc0. Hand-built specs fed through each vLLM's own
    /// `get_kv_cache_configs` on CPU reproduce every line exactly; the residual
    /// here (< 0.02%) is the null block the sim takes out of the pool.
    #[test]
    fn spec_kv_layouts_reproduce_logged_concurrency() {
        let check = |label: &str, layout: &KvCacheLayout, k: u32, logged: f64| {
            let concurrency = concurrency(layout, batch_tokens(k));
            assert!(
                (concurrency / logged - 1.0).abs() < 0.005,
                "{label}: {concurrency:.3} vs logged {logged}"
            );
        };
        let none = flash_kv_cache_layout_for(&verify_target(1), 0, None, FlashKvAllocator::MAIN);
        check("none", &none, 0, 60.11);
        for (k, logged) in [(1, 55.65), (3, 52.25), (5, 48.43), (7, 45.12)] {
            let layout =
                flash_kv_cache_layout_for(&verify_target(k + 1), 1, None, FlashKvAllocator::MAIN);
            check(&format!("mtp k{k}"), &layout, k, logged);
        }
        for (k, logged) in [(3, 49.73), (5, 45.50), (7, 41.93)] {
            let draft =
                dflash2_draft_resolved_for(&GLM53_FLASH_DFLASH2_DIMS, 4, "NVIDIA B200", k, 2048)
                    .unwrap();
            let layout = flash_kv_cache_layout_for(
                &verify_target(k + 1),
                0,
                Some(dflash2_draft_window(&draft, 4).unwrap()),
                FlashKvAllocator::MAIN,
            );
            check(&format!("dflash k{k}"), &layout, k, logged);
        }
    }

    /// The fork's MTP K=3 server (job 9122: same pinned KV, max_model_len
    /// 16384, `kv cache group sizes [2304, 4, 2304, 2304, 2304]`): 166.25x, a
    /// KV pool of 2,723,840 tokens. 3990 unrounded 15,068,160-byte block ids
    /// over 8 latent + 1 tail + 3 KDA groups x (2 + 3) pages.
    #[test]
    fn fork_mtp_layout_reproduces_its_logged_concurrency() {
        let layout =
            flash_kv_cache_layout_for(&verify_target(4), 1, None, FlashKvAllocator::FORK);
        assert_eq!(layout.engine.fixed_device_bytes, 4 * 15_068_160);
        let options = KvAccountingOptions {
            max_in_flight_tokens: Some(2 * 2048),
            prefix_caching: true,
            ..Default::default()
        };
        let accounting = KvAccounting::resolve(&layout, options, 60_129_542_144 * 4).unwrap();
        let concurrency = accounting.concurrency_at(16_384);
        assert!((concurrency / 166.25 - 1.0).abs() < 0.005, "{concurrency}");
        // The fork's ring stays one 4-token pool.
        let tail = layout.parts.iter().find(|part| part.name == "kpool_tail").unwrap();
        assert_eq!(tail.bytes_per_entry, 4 * 12 * 4 * 512);
    }

    /// The pieces of vLLM's layout `vllm_kv_capacity.py` prints, per GPU: block
    /// ids of `round_up(n_dsa * (B * 512 + B / 4 * 132), 540_672)` bytes, the
    /// KDA groups at `2 + K + 1` pages each, the kpool ring and the draft
    /// window's whole blocks.
    #[test]
    fn spec_kv_layouts_follow_vllms_block_ids() {
        // Bytes of one block id per GPU, and the block ids vLLM's estimate
        // charges one 131,072-token request.
        let block_ids = |layout: &KvCacheLayout, k: u32| {
            let options = KvAccountingOptions {
                max_in_flight_tokens: Some(2 * batch_tokens(k)),
                prefix_caching: true,
                ..Default::default()
            };
            let accounting = KvAccounting::resolve(layout, options, 60_129_542_144 * 4).unwrap();
            let block_units = layout.engine.fixed_device_bytes / accounting.bytes_per_unit();
            let request_units =
                accounting.capacity_units() as f64 / accounting.concurrency_at(131_072);
            (
                layout.engine.fixed_device_bytes / 4,
                request_units / block_units as f64,
            )
        };
        // (layout, K, bytes per block id, block tokens, block ids per request)
        let none = flash_kv_cache_layout_for(&verify_target(1), 0, None, FlashKvAllocator::MAIN);
        let mtp7 = flash_kv_cache_layout_for(&verify_target(8), 1, None, FlashKvAllocator::MAIN);
        let draft = dflash2_draft_resolved_for(&GLM53_FLASH_DFLASH2_DIMS, 4, "NVIDIA B200", 3, 2048)
            .unwrap();
        let dflash3 = flash_kv_cache_layout_for(
            &verify_target(4),
            0,
            Some(dflash2_draft_window(&draft, 4).unwrap()),
            FlashKvAllocator::MAIN,
        );
        for (layout, k, block_bytes, block_tokens, blocks_per_request) in [
            // 4 KDA groups x 3 pages + 61 latent blocks + 1 tail.
            (&none, 0, 13_516_800, 2_176, 74),
            // 3 x 10 + 57 + 1.
            (&mtp7, 7, 15_138_816, 2_304, 88),
            // 4 x 6 + 57 + 1 + cdiv(2047 + 2 * 2048, 2304) + 1 window blocks.
            (&dflash3, 3, 14_057_472, 2_304, 86),
        ] {
            let (block, request_blocks) = block_ids(layout, k);
            assert_eq!(block, block_bytes);
            assert_eq!(layout.engine.block_tokens, Some(block_tokens));
            assert!(
                (request_blocks - f64::from(blocks_per_request)).abs() < 1e-6,
                "{request_blocks} vs {blocks_per_request}"
            );
        }
        // The tail ring widens with the drafts: 4, 8, 16 tokens of [K, gate].
        let tail = |layout: &KvCacheLayout| {
            layout.parts.iter().find(|part| part.name == "kpool_tail").unwrap().bytes_per_entry / 4
        };
        assert_eq!(tail(&none), 11 * 4 * 512);
        assert_eq!(tail(&dflash3), 11 * 8 * 512);
        assert_eq!(tail(&mtp7), 12 * 16 * 512);
    }
}
