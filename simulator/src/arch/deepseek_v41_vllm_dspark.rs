//! DeepSeek-V4.1-Flash with its DSpark proposer, vLLM on B200 (TP4 attention,
//! EP4 experts): `--speculative-config {"method":"dspark","num_speculative_tokens":5}`.
//!
//! A speculative sibling of [`DeepseekV41VllmModel`]: it composes that target
//! forward unchanged and owns the sample phase and the proposer (pattern:
//! `glm53_vllm_nvfp4_dsa_moe_dflash2`). One engine step is
//! `forward -> sample -> draft` (capture `logs/20261106_0_dsv41_dspark_capture`,
//! `phase1_split.md` §1):
//!
//! ```text
//! target forward over 6R + P rows (verify width 6 per decode request)
//!   ... its head runs lm_head + logits AllGather over every verified row
//! sample: logits fp32 cast, rejection statistics, per-request resample
//! draft (FULL graph keyed by request count):
//!   context KV over all target rows (main_proj over layers 37/38/39)
//!   embedding (+ TP all-reduce), 3 reduced target layers on 5R rows
//!   head: mhc_post -> hc collapse -> norm -> lm_head (+ AllGather) -> cast
//!   5 sequential Markov steps on R rows
//! ```
//!
//! # The draft layers are target layers
//!
//! DSpark's draft layers are `DeepseekV4DecoderLayer`s (fork
//! `models/deepseek_v41/nvidia/dspark.py:128-138`) with `compress_ratio = 0`
//! (the last three `compress_ratios` entries), so each is a target SWA-only
//! layer and is built from the target's own attention and MoE worklets
//! through the target's `layer_config`. Two things differ:
//!
//! - the MoE geometry: `dspark_n_routed_experts = 128`, top
//!   `dspark_num_experts_per_tok = 3` (32 local experts under EP4). Draft
//!   routes are not captured; the demand is uniform (design D1);
//! - the mHC entry never takes the small-batch overlap path: the draft layers
//!   are constructed without an `mhc_stream` (`dspark.py:128-138`), so
//!   `model.py:524-531` always runs the fused `mega_mhc`, which the capture
//!   shows at 5R = 130 rows. Draft layer 0 has the layer-0 entry (hc expand,
//!   prenorm GEMM, pre_big_fuse).
//!
//! Draft attention: each of the 5R query rows attends its request's 128-token
//! sliding window plus the 5 block tokens (non-causal block through the top-k
//! index list, `dspark.py:9-10`), priced as decode rows on the ratio-0
//! FlashMLA mega kernel.
//!
//! # What is not modeled
//!
//! The confidence head (off: `enable_adaptive_verification`), the fused
//! Markov sampler (off), and the per-layer draft metadata
//! (`_prepare_dflash_inputs`, `_token_request_mapping`, `CombineTopkSwaIndices`:
//! launch-bound plumbing, like the target's attention metadata).

use std::path::Path;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

use crate::arch::contract::{SpeculativeArchInput, SpeculativeUnifiedModel};
use crate::arch::deepseek_v41_vllm::{
    self, kv_cache_layout, layer_config, normalize_step, prefill_pairs, BoundedReplay,
    DeepseekV41CudaGraphs, DeepseekV41LayerConfig, DeepseekV41ModelCfg, DeepseekV41ReplayConfig,
    DeepseekV41Step, DeepseekV41VllmConfigs, DeepseekV41VllmModel, DeepseekV41VllmParallel,
    DeepseekV41VllmResolved, LayerKind, NormalizedInput, ELEMENTWISE_BACKENDS, KV_INSERT_BACKENDS,
    LM_HEAD_BACKENDS, MAX_NUM_SEQS, MXFP8_GEMM_BACKENDS,
};
use crate::arch::kv_layout::KvCacheLayout;
use crate::op::attention::DeepseekV41CandidateRole;
use crate::op::spec::{DsparkMarkovStepConfig, DsparkMarkovStepInput, DsparkMarkovStepOp};
use crate::op::Op;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{
    AllReduceFusionKernel, AllReduceFusionKernelConfig, AllReduceFusionKernelInput,
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::deepseek_v41_common::{eval_or_zero, placeholder};
use crate::worklet::{
    DeepseekV41AttentionEntry, DeepseekV41AttentionLayer, DeepseekV41AttentionTpWorklet,
    DeepseekV41AttentionTpWorkletConfig, DeepseekV41AttentionTpWorkletInput,
    DeepseekV41AttentionTpWorkletResolved, DeepseekV41DsparkContextKvLocalWorklet,
    DeepseekV41DsparkContextKvLocalWorkletConfig, DeepseekV41DsparkContextKvLocalWorkletInput,
    DeepseekV41DsparkContextKvLocalWorkletResolved, DeepseekV41HeadTpWorklet,
    DeepseekV41HeadTpWorkletConfig, DeepseekV41HeadTpWorkletInput,
    DeepseekV41HeadTpWorkletResolved, DeepseekV41IndexRole, DeepseekV41MoeFfnEpWorklet,
    DeepseekV41MoeFfnEpWorkletInput, DeepseekV41MoeFfnEpWorkletResolved,
    DeepseekV41PrologueTpWorklet,
};

/// The draft layers' compress ratio: SWA only.
const DRAFT_COMPRESS_RATIO: u32 = 0;

const ARCH_KIND: &str = "deepseek_v41_vllm_dspark";

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

/// The DSpark fields of the checkpoint's `text_config`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DsparkCfg {
    /// `num_nextn_predict_layers`: draft decoder layers (3).
    pub num_draft_layers: u32,
    /// `dspark_block_size`: positions one proposal drafts (5).
    pub block_size: u32,
    /// `dspark_target_layer_ids`: target layers feeding `main_proj` ([37, 38, 39]).
    pub target_layer_ids: Vec<u32>,
    /// `dspark_markov_rank` (256).
    pub markov_rank: u32,
    /// `dspark_n_routed_experts` (128).
    pub num_experts: u32,
    /// `dspark_num_experts_per_tok` (3).
    pub top_k: u32,
}

#[derive(Deserialize)]
struct JsonDsparkConfig {
    num_nextn_predict_layers: u32,
    num_hidden_layers: u32,
    compress_ratios: Vec<u32>,
    dspark_block_size: u32,
    dspark_target_layer_ids: Vec<u32>,
    dspark_markov_rank: u32,
    dspark_n_routed_experts: u32,
    dspark_num_experts_per_tok: u32,
}

impl DsparkCfg {
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading DeepSeek V4.1 config {}", path.display()))?;
        let raw: JsonDsparkConfig =
            serde_json::from_str(&text).context("parsing the DSpark fields")?;
        let body = raw.num_hidden_layers as usize;
        ensure!(
            raw.compress_ratios.len() == body + raw.num_nextn_predict_layers as usize
                && raw.compress_ratios[body..]
                    .iter()
                    .all(|&ratio| ratio == DRAFT_COMPRESS_RATIO),
            "DSpark draft layers must be SWA-only (compress ratio 0)"
        );
        ensure!(
            !raw.dspark_target_layer_ids.is_empty()
                && raw
                    .dspark_target_layer_ids
                    .iter()
                    .all(|&layer| layer < raw.num_hidden_layers),
            "dspark_target_layer_ids must name body layers, got {:?}",
            raw.dspark_target_layer_ids
        );
        ensure!(
            raw.num_nextn_predict_layers > 0
                && raw.dspark_block_size > 0
                && raw.dspark_markov_rank > 0
                && raw.dspark_num_experts_per_tok > 0
                && raw.dspark_num_experts_per_tok <= raw.dspark_n_routed_experts,
            "DSpark config fields must be positive with top-k <= experts"
        );
        Ok(Self {
            num_draft_layers: raw.num_nextn_predict_layers,
            block_size: raw.dspark_block_size,
            target_layer_ids: raw.dspark_target_layer_ids,
            markov_rank: raw.dspark_markov_rank,
            num_experts: raw.dspark_n_routed_experts,
            top_k: raw.dspark_num_experts_per_tok,
        })
    }
}

/// Every proposer config, beside the target's.
pub struct DeepseekV41DsparkConfigs {
    pub target: DeepseekV41VllmConfigs,
    pub draft_tokens: u32,
    pub num_draft_layers: u32,
    pub context_kv: DeepseekV41DsparkContextKvLocalWorkletConfig,
    /// Draft layer 0 (layer-0 mHC entry) and the shared body of the rest.
    pub draft_first: DeepseekV41LayerConfig,
    pub draft_rest: DeepseekV41LayerConfig,
    pub head: DeepseekV41HeadTpWorkletConfig,
    pub markov: DsparkMarkovStepConfig,
    pub vocab_size: u32,
    pub kv_cache_layout: KvCacheLayout,
    /// `--cudagraph-capture-sizes`: the verify (and draft) FULL graphs'
    /// request counts derive from it at the verify width.
    pub cudagraph_capture_sizes: Vec<u32>,
}

pub struct DeepseekV41DsparkResolved {
    target: DeepseekV41VllmResolved,
    draft_tokens: u32,
    num_draft_layers: u32,
    context_kv: DeepseekV41DsparkContextKvLocalWorkletResolved,
    embedding: ElementwiseKernelConfig,
    embedding_all_reduce: Option<AllReduceFusionKernelConfig>,
    first_attention: DeepseekV41AttentionTpWorkletResolved,
    first_ffn: DeepseekV41MoeFfnEpWorkletResolved,
    rest_attention: DeepseekV41AttentionTpWorkletResolved,
    rest_ffn: DeepseekV41MoeFfnEpWorkletResolved,
    head: DeepseekV41HeadTpWorkletResolved,
    markov: DsparkMarkovStepConfig,
    logits: LogitsPlaceholders,
    kv_cache_layout: KvCacheLayout,
    cudagraph_capture_sizes: Vec<u32>,
}

/// The draft MoE geometry over the target's checkpoint facts.
pub(crate) fn draft_model(model: &DeepseekV41ModelCfg, dspark: &DsparkCfg) -> DeepseekV41ModelCfg {
    DeepseekV41ModelCfg {
        num_experts: dspark.num_experts,
        top_k: dspark.top_k,
        ..model.clone()
    }
}

pub(crate) fn draft_kind(entry: DeepseekV41AttentionEntry) -> LayerKind {
    LayerKind {
        attention: DeepseekV41AttentionLayer {
            entry,
            compress_ratio: DRAFT_COMPRESS_RATIO,
            kv_source: false,
            index: DeepseekV41IndexRole::None,
            candidate: DeepseekV41CandidateRole::None,
        },
        engram: false,
    }
}

/// Checks shared by the TP and DP proposers.
pub(crate) fn check_draft(
    dspark: &DsparkCfg,
    draft_demand: &ExpertDemand,
    draft_tokens: u32,
) -> std::result::Result<(), BuildError> {
    if draft_tokens != dspark.block_size {
        return Err(fit_failed(format!(
            "DSpark drafts dspark_block_size = {} positions per proposal, got draft_tokens {draft_tokens}",
            dspark.block_size
        )));
    }
    if draft_demand.num_experts() != dspark.num_experts as usize {
        return Err(fit_failed(format!(
            "draft expert demand covers {} experts, expected {}",
            draft_demand.num_experts(),
            dspark.num_experts
        )));
    }
    Ok(())
}

/// The `dspark_target_layer_ids` hidden states are produced inside the
/// replay batch (all past the cut) and scattered back to batch rows with the
/// hc streams: one bf16 hidden row per captured layer.
pub(crate) fn widen_replay_scatter(
    replay: Option<&mut DeepseekV41ReplayConfig>,
    model: &DeepseekV41ModelCfg,
    dspark: &DsparkCfg,
) {
    if let Some(replay) = replay {
        let aux = dspark.target_layer_ids.len() as u32 * model.hidden_size * 2;
        replay.scatter.input_bytes_per_token =
            (replay.scatter.input_bytes_per_token.get() + aux).into();
        replay.scatter.output_bytes_per_token =
            (replay.scatter.output_bytes_per_token.get() + aux).into();
    }
}

/// Draft layer 0 (layer-0 mHC entry) and the shared body of the rest, at
/// `parallel`'s rank shape.
pub(crate) fn draft_layer_configs(
    model: &DeepseekV41ModelCfg,
    dspark: &DsparkCfg,
    parallel: &DeepseekV41VllmParallel,
    draft_demand: &ExpertDemand,
) -> (DeepseekV41LayerConfig, DeepseekV41LayerConfig) {
    let draft = draft_model(model, dspark);
    let first_layer = model.num_layers;
    let first = layer_config(
        &draft,
        parallel,
        draft_demand,
        draft_kind(DeepseekV41AttentionEntry::Layer0Pre),
        vec![first_layer],
    );
    let rest = layer_config(
        &draft,
        parallel,
        draft_demand,
        draft_kind(DeepseekV41AttentionEntry::FusedPostPre),
        (first_layer + 1..first_layer + dspark.num_draft_layers).collect(),
    );
    (first, rest)
}

/// The context-KV section; `attention` is a draft layer's attention config
/// (its padded Q width and SWA page size).
pub(crate) fn context_kv_config(
    model: &DeepseekV41ModelCfg,
    dspark: &DsparkCfg,
    attention: &DeepseekV41AttentionTpWorkletConfig,
) -> DeepseekV41DsparkContextKvLocalWorkletConfig {
    DeepseekV41DsparkContextKvLocalWorkletConfig {
        hidden_size: model.hidden_size.into(),
        hc_mult: model.hc_mult,
        num_target_layers: dspark.target_layer_ids.len() as u32,
        num_draft_layers: dspark.num_draft_layers,
        head_dim: model.head_dim.into(),
        // The KV-only insert launches at the Q width it would pad to.
        insert_num_heads: attention.padded_heads.get(),
        swa_block_size: attention.swa_block_size,
        gpu_name: attention.gpu_name.clone(),
        gemm_backends: MXFP8_GEMM_BACKENDS.to_vec(),
        kv_insert_backends: KV_INSERT_BACKENDS.to_vec(),
        elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
    }
}

pub(crate) fn markov_config(
    model: &DeepseekV41ModelCfg,
    dspark: &DsparkCfg,
    gpu: &str,
) -> DsparkMarkovStepConfig {
    DsparkMarkovStepConfig {
        gpu_name: gpu.to_string(),
        vocab_size: model.vocab_size,
        markov_rank: dspark.markov_rank,
        gemm_backends: LM_HEAD_BACKENDS.to_vec(),
        elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
    }
}

/// The placeholder launches around the logits: the draft head's fp32 cast
/// and the sampler's cast, rejection statistics and resample.
#[derive(Clone)]
pub(crate) struct LogitsPlaceholders {
    pub draft_logits_cast: ElementwiseKernelConfig,
    pub sample_logits_cast: ElementwiseKernelConfig,
    pub sample_logits_stats: ElementwiseKernelConfig,
    pub sample_resample: ElementwiseKernelConfig,
}

pub(crate) fn logits_placeholders(vocab: u32, gpu: &str) -> LogitsPlaceholders {
    let ew = |input: u32, output: u32| placeholder(ELEMENTWISE_BACKENDS, gpu, input, output);
    LogitsPlaceholders {
        // `LogitsProcessor` upcasts the bf16 logits to fp32.
        draft_logits_cast: ew(vocab * 2, vocab * 4),
        sample_logits_cast: ew(vocab * 2, vocab * 4),
        // Rejection sampler (fork `rejection_sampler_utils.py`), no L1 kind
        // (design D7): `_compute_local_logits_stats` and `_rejection_kernel`
        // read each verified row's fp32 target and draft logits ...
        sample_logits_stats: ew(2 * vocab * 4, 8),
        // ... and `_resample_kernel` re-reads one row per request.
        sample_resample: ew(vocab * 4, 4),
    }
}

/// `target` is the target's expansion (lane A's `build_configs`, priced at
/// verify width 6); `draft_demand` covers the draft MoE's `dspark_n_routed_experts`.
pub fn build_configs(
    model: &DeepseekV41ModelCfg,
    dspark: &DsparkCfg,
    parallel: &DeepseekV41VllmParallel,
    mut target: DeepseekV41VllmConfigs,
    draft_demand: &ExpertDemand,
    draft_tokens: u32,
) -> std::result::Result<DeepseekV41DsparkConfigs, BuildError> {
    check_draft(dspark, draft_demand, draft_tokens)?;
    widen_replay_scatter(target.replay.as_mut(), model, dspark);
    let (draft_first, draft_rest) = draft_layer_configs(model, dspark, parallel, draft_demand);
    Ok(DeepseekV41DsparkConfigs {
        draft_tokens,
        num_draft_layers: dspark.num_draft_layers,
        context_kv: context_kv_config(model, dspark, &draft_first.attention),
        draft_first,
        draft_rest,
        // The draft head is the target's: `ParallelLMHead` aliased from the
        // target (`has_own_lm_head = False`), vocab-sharded under TP.
        head: target.head.clone(),
        markov: markov_config(model, dspark, &parallel.gpu_name),
        vocab_size: model.vocab_size,
        kv_cache_layout: kv_cache_layout(model, parallel.tp_size, dspark.num_draft_layers),
        cudagraph_capture_sizes: parallel.cudagraph_capture_sizes.clone(),
        target,
    })
}

pub fn resolve_configs(configs: DeepseekV41DsparkConfigs) -> DeepseekV41DsparkResolved {
    let prologue = DeepseekV41PrologueTpWorklet::resolve_config(&configs.target.prologue);
    let logits = logits_placeholders(configs.vocab_size, &configs.markov.gpu_name);
    DeepseekV41DsparkResolved {
        draft_tokens: configs.draft_tokens,
        num_draft_layers: configs.num_draft_layers,
        context_kv: DeepseekV41DsparkContextKvLocalWorklet::resolve_config(&configs.context_kv),
        // The draft embeds its block with the target's vocab-parallel table
        // (`has_own_embed_tokens = False`) and all-reduces it under TP.
        embedding: prologue.embedding,
        embedding_all_reduce: prologue.embedding_all_reduce,
        first_attention: DeepseekV41AttentionTpWorklet::resolve_config(
            &configs.draft_first.attention,
        ),
        first_ffn: DeepseekV41MoeFfnEpWorklet::resolve_config(&configs.draft_first.ffn),
        rest_attention: DeepseekV41AttentionTpWorklet::resolve_config(
            &configs.draft_rest.attention,
        ),
        rest_ffn: DeepseekV41MoeFfnEpWorklet::resolve_config(&configs.draft_rest.ffn),
        head: DeepseekV41HeadTpWorklet::resolve_config(&configs.head),
        markov: configs.markov,
        logits,
        kv_cache_layout: configs.kv_cache_layout,
        cudagraph_capture_sizes: configs.cudagraph_capture_sizes,
        target: deepseek_v41_vllm::resolve_configs(&configs.target),
    }
}

/// One draft decoder layer: the target's attention and MoE worklets.
struct DraftLayer {
    attention: DeepseekV41AttentionTpWorklet,
    ffn: DeepseekV41MoeFfnEpWorklet,
}

impl DraftLayer {
    fn build(
        name: &str,
        attention: DeepseekV41AttentionTpWorkletResolved,
        ffn: DeepseekV41MoeFfnEpWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        Ok(Self {
            attention: DeepseekV41AttentionTpWorklet::build(
                format!("{name}.attn"),
                attention,
                bridge,
            )?,
            ffn: DeepseekV41MoeFfnEpWorklet::build(format!("{name}.ffn"), ffn, bridge)?,
        })
    }

    fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![self.attention.compile(b), self.ffn.compile(b)])
    }

    fn eval(&self, input: &DraftStep, ev: &mut Evaluator) {
        self.attention.eval(&input.attention, ev);
        self.ffn.eval(
            &DeepseekV41MoeFfnEpWorkletInput {
                num_tokens: input.rows,
                mhc_overlap: false,
            },
            ev,
        );
    }
}

/// The draft call's shapes.
struct DraftStep {
    /// Target rows scheduled this step: the context-KV rows.
    context_tokens: u32,
    /// Requests the draft graph runs (padded to its FULL graph size).
    requests: u32,
    /// `draft_tokens x requests`: the backbone's rows.
    rows: u32,
    attention: DeepseekV41AttentionTpWorkletInput,
}

/// The proposer: context KV, embedding, draft layers, head, Markov steps.
struct DsparkDraftStage {
    name: String,
    num_draft_layers: u32,
    context_kv: DeepseekV41DsparkContextKvLocalWorklet,
    embedding: Op<ElementwiseKernel>,
    embedding_all_reduce: Option<Op<AllReduceFusionKernel>>,
    first: DraftLayer,
    rest: DraftLayer,
    head: DeepseekV41HeadTpWorklet,
    logits_cast: Op<ElementwiseKernel>,
    markov: DsparkMarkovStepOp,
}

pub(crate) fn atomic_elementwise(
    name: String,
    config: ElementwiseKernelConfig,
    bridge: &PerfApiBridge,
) -> std::result::Result<Op<ElementwiseKernel>, BuildError> {
    Ok(Op::new(
        name.clone(),
        Arc::new(ElementwiseKernel::build(name, config, bridge)?),
    ))
}

impl DsparkDraftStage {
    fn build(
        name: String,
        r: &DeepseekV41DsparkResolved,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        let embedding_all_reduce = match &r.embedding_all_reduce {
            Some(config) => {
                let op_name = format!("{name}.embedding_all_reduce");
                Some(Op::new(
                    op_name.clone(),
                    Arc::new(AllReduceFusionKernel::build(
                        op_name,
                        config.clone(),
                        bridge,
                    )?),
                ))
            }
            None => None,
        };
        Ok(Self {
            num_draft_layers: r.num_draft_layers,
            context_kv: DeepseekV41DsparkContextKvLocalWorklet::build(
                format!("{name}.context_kv"),
                r.context_kv.clone(),
                bridge,
            )?,
            embedding: atomic_elementwise(
                format!("{name}.embedding"),
                r.embedding.clone(),
                bridge,
            )?,
            embedding_all_reduce,
            first: DraftLayer::build(
                &format!("{name}.layer0"),
                r.first_attention.clone(),
                r.first_ffn.clone(),
                bridge,
            )?,
            rest: DraftLayer::build(
                &format!("{name}.layer"),
                r.rest_attention.clone(),
                r.rest_ffn.clone(),
                bridge,
            )?,
            head: DeepseekV41HeadTpWorklet::build(format!("{name}.head"), r.head.clone(), bridge)?,
            logits_cast: atomic_elementwise(
                format!("{name}.head.logits_fp32_cast"),
                r.logits.draft_logits_cast.clone(),
                bridge,
            )?,
            markov: DsparkMarkovStepOp::build(format!("{name}.markov"), r.markov.clone(), bridge)?,
            name,
        })
    }

    fn compile(&self, b: &mut CostTreeBuilder, draft_tokens: u32) -> CostNode {
        let mut embedding = vec![self.embedding.compile(b)];
        if let Some(all_reduce) = &self.embedding_all_reduce {
            embedding.push(all_reduce.compile(b));
        }
        let rest = self.num_draft_layers - 1;
        CostNode::Sum(vec![
            CostNode::Labeled {
                label: format!("{}.context_kv [every target row of the step]", self.name),
                child: Box::new(self.context_kv.compile(b)),
            },
            CostNode::Labeled {
                label: format!("{}.embedding [draft rows]", self.name),
                child: Box::new(CostNode::Sum(embedding)),
            },
            CostNode::Labeled {
                label: format!("{}.layer0 [layer-0 mHC entry]", self.name),
                child: Box::new(self.first.compile(b)),
            },
            CostNode::Labeled {
                label: format!("{}.layers [{rest} identical SWA-only layers]", self.name),
                child: Box::new(CostNode::Scale {
                    n: rest,
                    child: Box::new(self.rest.compile(b)),
                }),
            },
            CostNode::Labeled {
                label: format!("{}.head [base logits of every drafted position]", self.name),
                child: Box::new(CostNode::Sum(vec![
                    self.head.compile(b),
                    self.logits_cast.compile(b),
                ])),
            },
            CostNode::Labeled {
                label: format!("{}.markov [{draft_tokens} sequential steps]", self.name),
                child: Box::new(CostNode::Scale {
                    n: draft_tokens,
                    child: Box::new(self.markov.compile(b)),
                }),
            },
        ])
    }

    fn eval(&self, step: &DraftStep, ev: &mut Evaluator) {
        self.context_kv.eval(
            &DeepseekV41DsparkContextKvLocalWorkletInput {
                context_tokens: step.context_tokens,
            },
            ev,
        );
        let rows = step.rows;
        let zero = rows == 0;
        eval_or_zero(
            &self.embedding,
            ElementwiseKernelInput { num_tokens: rows },
            zero,
            ev,
        );
        if let Some(all_reduce) = &self.embedding_all_reduce {
            eval_or_zero(
                all_reduce,
                AllReduceFusionKernelInput { num_tokens: rows },
                zero,
                ev,
            );
        }
        self.first.eval(step, ev);
        self.rest.eval(step, ev);
        self.head.eval(
            &DeepseekV41HeadTpWorkletInput {
                num_tokens: rows,
                mhc_post_rows: rows,
                logits_rows: rows,
            },
            ev,
        );
        eval_or_zero(
            &self.logits_cast,
            ElementwiseKernelInput { num_tokens: rows },
            zero,
            ev,
        );
        self.markov.eval(
            &DsparkMarkovStepInput {
                num_requests: step.requests,
            },
            ev,
        );
    }
}

/// The rejection sampler between the verify forward and the draft.
pub(crate) struct SampleStage {
    logits_cast: Op<ElementwiseKernel>,
    logits_stats: Op<ElementwiseKernel>,
    resample: Op<ElementwiseKernel>,
}

impl SampleStage {
    pub(crate) fn build(
        name: &str,
        logits: &LogitsPlaceholders,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        Ok(Self {
            logits_cast: atomic_elementwise(
                format!("{name}.logits_fp32_cast"),
                logits.sample_logits_cast.clone(),
                bridge,
            )?,
            logits_stats: atomic_elementwise(
                format!("{name}.rejection_logits_stats"),
                logits.sample_logits_stats.clone(),
                bridge,
            )?,
            resample: atomic_elementwise(
                format!("{name}.resample"),
                logits.sample_resample.clone(),
                bridge,
            )?,
        })
    }

    pub(crate) fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            self.logits_cast.compile(b),
            self.logits_stats.compile(b),
            self.resample.compile(b),
        ])
    }

    pub(crate) fn eval(&self, logits_rows: u32, decode_requests: u32, ev: &mut Evaluator) {
        let rows = ElementwiseKernelInput {
            num_tokens: logits_rows,
        };
        eval_or_zero(&self.logits_cast, rows.clone(), logits_rows == 0, ev);
        eval_or_zero(&self.logits_stats, rows, decode_requests == 0, ev);
        eval_or_zero(
            &self.resample,
            ElementwiseKernelInput {
                num_tokens: decode_requests,
            },
            decode_requests == 0,
            ev,
        );
    }
}

/// DeepSeek-V4.1-Flash + DSpark, TP4 attention + EP4 experts.
pub struct DeepseekV41VllmDsparkModel {
    name: String,
    target: DeepseekV41VllmModel,
    sample: SampleStage,
    draft: DsparkDraftStage,
    /// Positions drafted per request (5); the verify width is one more.
    draft_tokens: u32,
    max_model_len: u32,
    /// Graphs at the verify width (decode query length `draft_tokens + 1`).
    graphs: DeepseekV41CudaGraphs,
    tp_size: u16,
    kv_cache_layout: KvCacheLayout,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: DeepseekV41DsparkResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<DeepseekV41VllmDsparkModel, BuildError> {
    let max_model_len = resolved.first_attention.raw_cfg.max_model_len;
    let tp_size = resolved.first_attention.raw_cfg.tp_size as u16;
    let draft = DsparkDraftStage::build(format!("{name}.dspark"), &resolved, bridge)?;
    let sample = SampleStage::build(&format!("{name}.sample"), &resolved.logits, bridge)?;
    let draft_tokens = resolved.draft_tokens;
    let kv_cache_layout = resolved.kv_cache_layout.clone();
    let target = deepseek_v41_vllm::build(name.clone(), resolved.target, bridge)?;
    let mut model = DeepseekV41VllmDsparkModel {
        name,
        target,
        sample,
        draft,
        draft_tokens,
        max_model_len,
        graphs: DeepseekV41CudaGraphs::new(
            &resolved.cudagraph_capture_sizes,
            MAX_NUM_SEQS,
            &[draft_tokens + 1],
        ),
        tp_size,
        kv_cache_layout,
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

impl DeepseekV41VllmDsparkModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let mut sections = self.target.compile_sections(&mut b);
        sections.push(CostNode::Labeled {
            label: format!(
                "{}.sample [rejection sampler over the verified rows]",
                self.name
            ),
            child: Box::new(self.sample.compile(&mut b)),
        });
        sections.push(CostNode::Labeled {
            label: format!(
                "{} [DSpark proposer; draft_tokens={}; {} draft layers]",
                self.draft.name, self.draft_tokens, self.draft.num_draft_layers
            ),
            child: Box::new(self.draft.compile(&mut b, self.draft_tokens)),
        });
        b.finish(CostNode::Labeled {
            label: format!(
                "{} (DeepseekV41VllmDsparkModel) [TP{} attention, EP{} experts; verify_width={}; \
                 max_model_len={}]",
                self.name,
                self.tp_size,
                self.tp_size,
                self.draft_tokens + 1,
                self.max_model_len
            ),
            child: Box::new(CostNode::Sum(sections)),
        })
    }

    fn normalize(
        &self,
        input: &SpeculativeArchInput,
    ) -> std::result::Result<(NormalizedInput, u32, DraftStep), String> {
        normalize_dspark_input(
            &self.graphs,
            input,
            self.draft_tokens,
            self.max_model_len,
            self.target.replay_rule(),
        )
    }

    fn eval_into(&self, input: &SpeculativeArchInput, ev: &mut Evaluator) {
        let (target, decode_requests, draft) = self
            .normalize(input)
            .unwrap_or_else(|reason| panic!("invalid DeepseekV41VllmDsparkModel input: {reason}"));
        self.target.eval_sections(&target, ev);
        self.sample.eval(target.logits_rows, decode_requests, ev);
        self.draft.eval(&draft, ev);
    }

    fn eval_checked(&self, input: &SpeculativeArchInput, ev: &mut Evaluator) {
        self.eval_into(input, ev);
        assert_eq!(
            ev.filled(),
            self.n_slots,
            "eval must fill every compiled slot"
        );
    }
}

/// Split one speculative step into the target verify input, the decode
/// request count the sampler resamples, and the draft call's shapes.
fn normalize_dspark_input(
    graphs: &DeepseekV41CudaGraphs,
    input: &SpeculativeArchInput,
    draft_tokens: u32,
    max_model_len: u32,
    replay: Option<&BoundedReplay>,
) -> std::result::Result<(NormalizedInput, u32, DraftStep), String> {
    if input.draft_tokens != draft_tokens {
        return Err(format!(
            "input draft_tokens {} must match model draft_tokens {draft_tokens}",
            input.draft_tokens
        ));
    }
    if input.groups.len() != 1 {
        return Err(format!(
            "TP attention takes exactly one group, got {}",
            input.groups.len()
        ));
    }
    let group = &input.groups[0];
    let prefill = prefill_pairs(&group.prefill_chunk_pairs, max_model_len)?;
    let mut decode = Vec::with_capacity(group.decode_requests.len());
    for (index, request) in group.decode_requests.iter().enumerate() {
        request
            .check_verify_row(draft_tokens, max_model_len)
            .map_err(|reason| format!("decode request {index}: {reason}"))?;
        decode.push((request.query_len, request.kv_len));
    }
    let prefill_tokens: u32 = prefill.iter().map(|&(q, _)| q).sum();
    let decode_tokens: u32 = decode.iter().map(|&(q, _)| q).sum();
    if group.prefill_tokens != prefill_tokens
        || group.decode_tokens != decode_tokens
        || group.batch_tokens != prefill_tokens + decode_tokens
    {
        return Err("group token accounting is inconsistent".into());
    }
    // Every prefill chunk yields one logits row; every verify row one more.
    let logits_rows = prefill.len() as u32 + decode_tokens;
    let step = DeepseekV41Step { prefill, decode };
    let target = normalize_step(graphs, &step, draft_tokens + 1, logits_rows, replay);

    // Every scheduled request drafts a block. The draft forward is one FULL
    // graph keyed by the request count (the verify width's capture sizes);
    // past the largest it runs eager at the live count.
    let live = step.request_count();
    let graph = graphs.full_decode_requests(live, draft_tokens + 1);
    let requests = if live == 0 { 0 } else { graph.unwrap_or(live) };
    let rows = requests * draft_tokens;
    // Each draft row attends its request's window plus the block before it.
    let decode_kv_lens = step
        .prefill
        .iter()
        .map(|&(_, context)| context)
        .chain(step.decode.iter().map(|&(_, kv)| kv))
        .flat_map(|context| {
            (0..draft_tokens).map(move |offset| {
                context
                    .saturating_add(offset)
                    .min(max_model_len.saturating_sub(1))
            })
        })
        .collect();
    let draft = DraftStep {
        context_tokens: group.batch_tokens,
        requests,
        rows,
        attention: DeepseekV41AttentionTpWorkletInput {
            num_tokens: rows,
            prefill_query_context_pairs: Vec::new(),
            decode_kv_lens,
            aux_stream_live: graph.is_some(),
            full_graph: graph.is_some(),
            // Draft layers have no `mhc_stream` (`dspark.py:128-138`).
            mhc_overlap: false,
            write_kv_only: false,
        },
    };
    Ok((target, step.decode.len() as u32, draft))
}

impl SpeculativeUnifiedModel for DeepseekV41VllmDsparkModel {
    fn check_input(&self, batch: &SpeculativeArchInput) -> Result<(), String> {
        self.normalize(batch).map(|_| ())
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
        self.eval_checked(batch, &mut evaluator);
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
        self.eval_checked(batch, &mut evaluator);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    /// The target's packed groups plus one 128-token SWA cache per draft
    /// layer (`get_draft_kv_cache_layer_names`, `dspark.py:352-355`).
    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout.clone()
    }

    fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    fn gpus_per_replica(&self) -> u16 {
        self.tp_size
    }

    fn num_attn_dp_groups(&self) -> u16 {
        1
    }

    fn num_attn_shards(&self) -> u16 {
        self.tp_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::config::ModelSpec;
    use crate::arch::contract::{SpeculativeArchGroupInput, SpeculativeDecodeInput};

    const CONFIG: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/model/config/deepseek_v41_flash.json"
    );

    fn model_cfg() -> DeepseekV41ModelCfg {
        let spec = ModelSpec {
            model_config: CONFIG.into(),
            num_layers: None,
            sim_num_layers: None,
            fp8: true,
        };
        DeepseekV41ModelCfg::from_json(Path::new(CONFIG), &spec).unwrap()
    }

    fn dspark() -> DsparkCfg {
        DsparkCfg::from_json(Path::new(CONFIG)).unwrap()
    }

    fn parallel() -> DeepseekV41VllmParallel {
        DeepseekV41VllmParallel {
            tp_size: 4,
            ep_size: 4,
            gpu_name: "NVIDIA B200".into(),
            serialize_streams: false,
            decoder_swa_bounded_replay: true,
            max_model_len: 131_072,
            max_num_batched_tokens: 2048,
            cudagraph_capture_sizes: deepseek_v41_vllm::vllm_default_cudagraph_capture_sizes(2048),
        }
    }

    fn uniform(experts: usize, layers: usize) -> ExpertDemand {
        ExpertDemand::Popularity {
            layerwise_global_ppm: vec![vec![1_000_000 / experts as u32; experts]; layers],
        }
    }

    fn configs() -> DeepseekV41DsparkConfigs {
        let model = model_cfg();
        let target =
            deepseek_v41_vllm::build_configs(&model, &parallel(), &uniform(384, 40)).unwrap();
        build_configs(&model, &dspark(), &parallel(), target, &uniform(128, 1), 5).unwrap()
    }

    #[test]
    fn checkpoint_dspark_fields_load() {
        assert_eq!(
            dspark(),
            DsparkCfg {
                num_draft_layers: 3,
                block_size: 5,
                target_layer_ids: vec![37, 38, 39],
                markov_rank: 256,
                num_experts: 128,
                top_k: 3,
            }
        );
    }

    #[test]
    fn draft_layers_are_swa_only_target_layers_with_the_draft_moe() {
        let configs = configs();
        assert_eq!(configs.draft_first.layers, vec![40]);
        assert_eq!(configs.draft_rest.layers, vec![41, 42]);
        assert_eq!(
            configs.draft_first.attention.layer.entry,
            DeepseekV41AttentionEntry::Layer0Pre
        );
        assert_eq!(
            configs.draft_rest.attention.layer.entry,
            DeepseekV41AttentionEntry::FusedPostPre
        );
        for layer in [&configs.draft_first, &configs.draft_rest] {
            assert_eq!(layer.attention.layer.compress_ratio, 0);
            assert!(layer.engram.is_none());
            assert_eq!((layer.ffn.num_experts.get(), layer.ffn.top_k), (128, 3));
        }
        let resolved = resolve_configs(configs);
        assert_eq!(resolved.rest_ffn.fused_moe.num_local_experts.get(), 32);
        assert!(resolved.rest_attention.indexer.is_none());
        assert!(resolved.rest_attention.compressor_proj.is_none());
        // Ratio-0 draft layers never take the swap-AB decode kernel
        // (`_swapab_decode_supported` requires a compressed layer).
        for attention in [&resolved.first_attention, &resolved.rest_attention] {
            assert!(crate::op::attention::deepseek_v41_mega_attn::swapab_config(
                &attention.mega_attn
            )
            .is_none());
        }
        // Draft lm_head is the target's vocab-sharded head.
        assert_eq!(resolved.head.lm_head.n.get(), 129_280 / 4);
    }

    #[test]
    fn draft_tokens_must_be_the_dspark_block() {
        let model = model_cfg();
        let target =
            deepseek_v41_vllm::build_configs(&model, &parallel(), &uniform(384, 40)).unwrap();
        assert!(
            build_configs(&model, &dspark(), &parallel(), target, &uniform(128, 1), 4).is_err()
        );
    }

    /// The TP4 DSpark capture's KV pool: 566,723 blocks per rank of
    /// 152,064 B logged "GPU KV cache size: 40,746,855 tokens, Maximum
    /// concurrency for 131,072 tokens per request: 310.87x". The draft adds three SWA layers
    /// to the target's 40 (six 32-token SWA groups instead of five).
    #[test]
    fn kv_cache_layout_reproduces_the_dspark_capture_pool() {
        use crate::worker::kv::{KvAccounting, KvAccountingOptions};
        let layout = configs().kv_cache_layout;
        let block = 152_064_u64;
        assert_eq!(layout.engine.fixed_device_bytes, 4 * block);
        let options = KvAccountingOptions {
            max_in_flight_tokens: Some(2 * 2048),
            ..Default::default()
        };
        let budget = 4 * 566_723 * block;
        let accounting = KvAccounting::resolve(&layout, options, budget).unwrap();
        let concurrency = accounting.concurrency_at(131_072);
        let logged = 310.87;
        assert!(
            (concurrency / logged - 1.0).abs() < 0.005,
            "{concurrency} vs {logged}"
        );
    }

    fn step(decode: &[(u32, u32)], prefill: &[(u32, u32)]) -> SpeculativeArchInput {
        let decode_requests: Vec<_> = decode
            .iter()
            .map(|&(kv_len, query_len)| SpeculativeDecodeInput { kv_len, query_len })
            .collect();
        let prefill_tokens = prefill.iter().map(|&(_, a)| a).sum();
        let decode_tokens = decode.iter().map(|&(_, q)| q).sum();
        SpeculativeArchInput {
            draft_tokens: 5,
            groups: vec![SpeculativeArchGroupInput {
                batch_tokens: prefill_tokens + decode_tokens,
                prefill_tokens,
                decode_tokens,
                prefill_chunk_pairs: prefill.to_vec(),
                decode_requests,
                total_kv_len: 0,
            }],
            tokens_per_source_rank: Vec::new(),
        }
    }

    fn graphs() -> DeepseekV41CudaGraphs {
        DeepseekV41CudaGraphs::new(&parallel().cudagraph_capture_sizes, MAX_NUM_SEQS, &[6])
    }

    #[test]
    fn draft_runs_five_rows_per_padded_request_and_context_kv_every_target_row() {
        let (target, decode, draft) = normalize_dspark_input(
            &graphs(),
            &step(&[(1000, 6); 3], &[(0, 200)]),
            5,
            131_072,
            None,
        )
        .unwrap();
        assert_eq!(decode, 3);
        assert_eq!(target.logits_rows, 1 + 18);
        assert_eq!(draft.context_tokens, 218);
        // Four requests run the four-request draft graph.
        assert_eq!((draft.requests, draft.rows), (4, 20));
        assert_eq!(draft.attention.decode_kv_lens.len(), 20);
        // Five requests pad to the six-request graph (no 5 at q = 6).
        let (target, _, draft) =
            normalize_dspark_input(&graphs(), &step(&[(1000, 6); 5], &[]), 5, 131_072, None)
                .unwrap();
        assert_eq!(target.full.rows, 36);
        assert_eq!((draft.requests, draft.rows), (6, 30));
        assert!(
            normalize_dspark_input(&graphs(), &step(&[(1000, 6)], &[]), 4, 131_072, None).is_err()
        );
    }
}
