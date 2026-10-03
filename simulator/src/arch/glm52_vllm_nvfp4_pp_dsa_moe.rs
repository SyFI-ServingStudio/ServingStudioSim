//! GLM-5.2 NVIDIA NVFP4 under pure pipeline parallelism, aligned to vLLM on B200.
//!
//! GLM-5.3 NVFP4 is the same graph (`glm53_vllm_nvfp4_dsa_moe_dflash2.rs`), so
//! one arch covers both checkpoints.
//!
//! Each pipeline stage is one GPU running a contiguous range of decoder layers
//! at EP1: all 64 attention heads and all 256 routed experts are local, so a
//! stage reduces nothing and builds no collective. The leaves are exactly the
//! vLLM NVFP4 EP graph's (`glm52_vllm_nvfp4_dsa_moe.rs`), built from its EP1
//! recipe and named as that graph names them, so the GLM-5.2 semantic location
//! map and kernel labels apply unchanged. A stage's identity lives only in the
//! `Labeled` node that wraps it.
//!
//! Without a fused all-reduce at the attention boundary, the post-attention
//! residual RMSNorm always runs as the FFN's own standalone leaf.
//!
//! The layer split follows vLLM's `get_pp_indices` ([`pp_indices`]). Stage 0
//! also runs the token embedding; the last stage runs the final residual
//! RMSNorm and the full-vocabulary lm_head. Between stages vLLM sends the
//! `hidden_states` and `residual` intermediate tensors
//! ([`Glm52VllmNvfp4PpDsaMoeModel::activation_bytes_per_token`]).
//!
//! Two models live here. [`Glm52VllmNvfp4PpStageModel`] is one stage, the unit a
//! pipeline worker runs. [`Glm52VllmNvfp4PpDsaMoeModel`] is the whole pipeline
//! seen as one iteration -- every stage in order, no transfer and no overlap --
//! which is what offline prediction and supported-build listing cost.

use std::sync::Arc;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::glm52_model_cfg::{Glm52ModelCfg, Glm52MtpMode};
use crate::arch::glm52_vllm_nvfp4_dsa_moe::{
    self as ep_graph, build_atomic, decoder_layer_state_bytes_per_token, eval_atomic_or_zero,
    normalize_input, Glm52SparseBody, Glm52VllmNvfp4DsaMoeConfigs, Glm52VllmNvfp4DsaMoeParallel,
    Glm52VllmNvfp4DsaMoeResolved, NormalizedBatch, CHECKPOINT_MAX_CONTEXT, FULL_INDEX_LAYERS,
    HIDDEN_DIM, NUM_DENSE_LAYERS, NUM_INITIAL_SHARED_LAYERS,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelInput, ResidualRmsNormKernel, ResidualRmsNormKernelInput,
    SingleGemmKernel, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::{
    Glm52DenseFfnLocalWorklet, Glm52DenseFfnLocalWorkletInput, VllmGlm52DsaAttnLocalWorklet,
};

const ARCH_KIND: &str = "glm52_vllm_nvfp4_pp_dsa_moe";

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

/// vLLM's `get_pp_indices` without `VLLM_PP_LAYER_PARTITION`: the
/// `[start, end)` layer range of every stage, in stage order.
///
/// Each stage gets `num_layers / pp_size` layers. The remainder goes one layer
/// each to the stages before the last, starting from the second-to-last and
/// moving toward stage 0: the last stage carries the final norm and lm_head,
/// and while the remainder is at most `pp_size - 2` stage 0 (the embedding) is
/// spared too.
///
/// Panics unless `1 <= pp_size <= num_layers`; callers validate first.
pub fn pp_indices(num_layers: u32, pp_size: u16) -> Vec<(u32, u32)> {
    let stages = u32::from(pp_size);
    assert!(
        (1..=num_layers).contains(&stages),
        "pp_size {pp_size} must be in 1..={num_layers}"
    );
    let mut partitions = vec![num_layers / stages; pp_size as usize];
    let remaining = num_layers % stages;
    for i in 2..remaining + 2 {
        partitions[(stages - i) as usize] += 1;
    }
    let mut start = 0;
    partitions
        .into_iter()
        .map(|layers| {
            let range = (start, start + layers);
            start += layers;
            range
        })
        .collect()
}

/// How many layers of each GLM-5.2 decoder-layer type a layer range holds.
///
/// The four types are the EP graph's four built sections: dense FFN with a
/// full indexer (layers 0..=2), sparse IndexShare before the first cycle
/// (3..=5), and the 18 cycles of one full-index plus three IndexShare sparse
/// layers (6..=77). Within a type every layer is the same launch graph, so a
/// stage folds each type with `Scale` and the order inside a stage changes
/// nothing but labels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageLayerCounts {
    pub dense: u32,
    pub initial_shared: u32,
    pub cycle_full: u32,
    pub cycle_shared: u32,
}

impl StageLayerCounts {
    pub fn of(layers: (u32, u32)) -> Self {
        let mut counts = Self::default();
        for layer in layers.0..layers.1 {
            if layer < NUM_DENSE_LAYERS {
                counts.dense += 1;
            } else if FULL_INDEX_LAYERS.contains(&layer) {
                counts.cycle_full += 1;
            } else if layer < NUM_DENSE_LAYERS + NUM_INITIAL_SHARED_LAYERS {
                counts.initial_shared += 1;
            } else {
                counts.cycle_shared += 1;
            }
        }
        counts
    }

    pub fn total(&self) -> u32 {
        self.dense + self.initial_shared + self.cycle_full + self.cycle_shared
    }
}

#[derive(Clone, Debug)]
pub struct Glm52VllmNvfp4PpParallel {
    /// Pipeline stages, one GPU each.
    pub pp_size: u16,
    pub max_model_len: u32,
    pub gpu_name: String,
}

/// The pipeline's layout plus the EP1 recipe every stage builds from.
#[derive(Clone, Debug)]
pub struct Glm52VllmNvfp4PpConfigs {
    pub parallel: Glm52VllmNvfp4PpParallel,
    pub stage: Glm52VllmNvfp4DsaMoeConfigs,
}

#[derive(Clone, Debug)]
pub struct Glm52VllmNvfp4PpResolved {
    pub parallel: Glm52VllmNvfp4PpParallel,
    pub stage: Glm52VllmNvfp4DsaMoeResolved,
}

/// The EP graph's recipe at one GPU: `ep_size = nvl_num_gpu = 1`, no MTP layer
/// and no speculation. `body_demand` is the whole body's routed demand; every
/// stage prices its experts against it, as the EP graph does for every layer.
pub fn build_configs(
    model: &Glm52ModelCfg,
    parallel: &Glm52VllmNvfp4PpParallel,
    body_demand: &ExpertDemand,
    fp8: bool,
) -> Result<Glm52VllmNvfp4PpConfigs, BuildError> {
    validate_pp_size(parallel.pp_size, model.num_layers)?;
    if !(1..=CHECKPOINT_MAX_CONTEXT).contains(&parallel.max_model_len) {
        return Err(fit_failed(format!(
            "max_model_len {} must be in 1..={CHECKPOINT_MAX_CONTEXT}",
            parallel.max_model_len
        )));
    }
    let stage = ep_graph::build_configs(
        model,
        &Glm52VllmNvfp4DsaMoeParallel {
            ep_size: 1,
            nvl_num_gpu: 1,
            max_model_len: parallel.max_model_len,
            gpu_name: parallel.gpu_name.clone(),
        },
        body_demand,
        None,
        fp8,
        Glm52MtpMode::Off,
    )?;
    Ok(Glm52VllmNvfp4PpConfigs {
        parallel: parallel.clone(),
        stage,
    })
}

pub fn resolve_configs(cfgs: &Glm52VllmNvfp4PpConfigs) -> Glm52VllmNvfp4PpResolved {
    Glm52VllmNvfp4PpResolved {
        parallel: cfgs.parallel.clone(),
        stage: ep_graph::resolve_configs(&cfgs.stage),
    }
}

fn validate_pp_size(pp_size: u16, num_layers: u32) -> Result<(), BuildError> {
    if pp_size < 2 || u32::from(pp_size) > num_layers {
        return Err(fit_failed(format!(
            "pp_size {pp_size} must be in 2..={num_layers}; one stage is not a pipeline"
        )));
    }
    Ok(())
}

/// Build every stage, then the whole-pipeline view over them.
pub fn build(
    name: String,
    resolved: Glm52VllmNvfp4PpResolved,
    bridge: &PerfApiBridge,
) -> Result<Glm52VllmNvfp4PpDsaMoeModel, BuildError> {
    let recipe = &resolved.stage.raw_cfg;
    validate_pp_size(resolved.parallel.pp_size, recipe.model.num_layers)?;
    if recipe.parallel.ep_size != 1 || recipe.parallel.nvl_num_gpu != 1 {
        return Err(fit_failed(format!(
            "every pipeline stage runs at EP1 on one GPU, got ep_size {} and nvl_num_gpu {}",
            recipe.parallel.ep_size, recipe.parallel.nvl_num_gpu
        )));
    }
    if recipe.mtp_mode != Glm52MtpMode::Off || recipe.speculative_draft_tokens.is_some() {
        return Err(fit_failed(
            "a pipeline stage runs neither the MTP layer nor a speculative verify",
        ));
    }
    let pp_size = resolved.parallel.pp_size;
    let ranges = pp_indices(recipe.model.num_layers, pp_size);
    let sections = StageSections::build(&name, &resolved.stage, bridge)?;
    let stages = ranges
        .into_iter()
        .enumerate()
        .map(|(index, layers)| {
            Glm52VllmNvfp4PpStageModel::build(
                &name,
                &resolved.stage,
                &sections,
                index as u16,
                pp_size,
                layers,
            )
            .map(Arc::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut model = Glm52VllmNvfp4PpDsaMoeModel {
        name,
        stages,
        max_model_len: recipe.parallel.max_model_len,
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

/// A dense decoder layer: full-index attention and the dense FFN.
struct DenseLayer {
    attention: VllmGlm52DsaAttnLocalWorklet,
    ffn: Glm52DenseFfnLocalWorklet,
}

/// The last stage's tail: the final residual RMSNorm and the lm_head.
struct OutputHead {
    final_norm: Op<ResidualRmsNormKernel>,
    /// Full vocabulary: a TP1 `ParallelLMHead` shards nothing.
    lm_head: Op<SingleGemmKernel>,
}

/// Every section of the graph, built once and shared by the stages that run
/// it. A section's kernels do not depend on which stage runs it, so building
/// them per stage only repeated the same profile lookups `pp_size` times.
struct StageSections {
    embedding: Arc<Op<ElementwiseKernel>>,
    dense: Arc<DenseLayer>,
    initial_shared: Arc<Glm52SparseBody>,
    cycle_full: Arc<Glm52SparseBody>,
    cycle_shared: Arc<Glm52SparseBody>,
    head: Arc<OutputHead>,
}

impl StageSections {
    fn build(
        name: &str,
        resolved: &Glm52VllmNvfp4DsaMoeResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let sparse = |section: &str, attention: &_| {
            Glm52SparseBody::build_rank_local(
                format!("{name}.body.{section}"),
                Clone::clone(attention),
                resolved,
                bridge,
            )
            .map(Arc::new)
        };
        Ok(Self {
            embedding: Arc::new(build_atomic(
                format!("{name}.main.embedding"),
                resolved.embedding.clone(),
                ElementwiseKernel::build,
                bridge,
            )?),
            dense: Arc::new(DenseLayer {
                attention: VllmGlm52DsaAttnLocalWorklet::build(
                    format!("{name}.body.dense_full_index.attention"),
                    resolved.dense_full_index_attention.clone(),
                    bridge,
                )?,
                ffn: Glm52DenseFfnLocalWorklet::build(
                    format!("{name}.body.dense_full_index.ffn"),
                    resolved.dense_ffn.clone(),
                    bridge,
                )?,
            }),
            initial_shared: sparse(
                "sparse_initial_index_share",
                &resolved.initial_shared_attention,
            )?,
            cycle_full: sparse("sparse_cycle_full_index", &resolved.cycle_full_attention)?,
            cycle_shared: sparse("sparse_cycle_index_share", &resolved.cycle_shared_attention)?,
            head: Arc::new(OutputHead {
                final_norm: build_atomic(
                    format!("{name}.main.final_residual_rms_norm"),
                    resolved.final_norm.clone(),
                    ResidualRmsNormKernel::build,
                    bridge,
                )?,
                lm_head: build_atomic(
                    format!("{name}.main.lm_head"),
                    resolved.lm_head.clone(),
                    SingleGemmKernel::build,
                    bridge,
                )?,
            }),
        })
    }
}

/// One pipeline stage: one GPU and its contiguous layer range.
///
/// Each section exists only when the stage runs it, so a stage mints no slot
/// for work it never does.
pub struct Glm52VllmNvfp4PpStageModel {
    name: String,
    stage_index: u16,
    num_stages: u16,
    layer_range: (u32, u32),
    counts: StageLayerCounts,
    embedding: Option<Arc<Op<ElementwiseKernel>>>,
    dense: Option<Arc<DenseLayer>>,
    initial_shared: Option<Arc<Glm52SparseBody>>,
    cycle_full: Option<Arc<Glm52SparseBody>>,
    cycle_shared: Option<Arc<Glm52SparseBody>>,
    head: Option<Arc<OutputHead>>,
    max_model_len: u32,
    kv_bytes_per_token: u64,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

impl Glm52VllmNvfp4PpStageModel {
    fn build(
        name: &str,
        resolved: &Glm52VllmNvfp4DsaMoeResolved,
        sections: &StageSections,
        stage_index: u16,
        num_stages: u16,
        layer_range: (u32, u32),
    ) -> Result<Self, BuildError> {
        let counts = StageLayerCounts::of(layer_range);
        let pick =
            |count: u32, section: &Arc<Glm52SparseBody>| (count > 0).then(|| Arc::clone(section));
        let embedding = (stage_index == 0).then(|| Arc::clone(&sections.embedding));
        let dense = (counts.dense > 0).then(|| Arc::clone(&sections.dense));
        let initial_shared = pick(counts.initial_shared, &sections.initial_shared);
        let cycle_full = pick(counts.cycle_full, &sections.cycle_full);
        let cycle_shared = pick(counts.cycle_shared, &sections.cycle_shared);
        let head = (stage_index + 1 == num_stages).then(|| Arc::clone(&sections.head));
        // vLLM allocates cache only for the layers a stage owns.
        let kv_bytes_per_token = u64::from(counts.total())
            .checked_mul(decoder_layer_state_bytes_per_token())
            .ok_or_else(|| fit_failed("stage state bytes overflow u64"))?;
        let mut stage = Self {
            name: name.to_string(),
            stage_index,
            num_stages,
            layer_range,
            counts,
            embedding,
            dense,
            initial_shared,
            cycle_full,
            cycle_shared,
            head,
            max_model_len: resolved.raw_cfg.parallel.max_model_len,
            kv_bytes_per_token,
            cost_flat: Vec::new(),
            n_slots: 0,
        };
        let tree = stage.cost_tree();
        stage.cost_flat = tree.flatten();
        stage.n_slots = tree.n_slots();
        Ok(stage)
    }

    /// This stage's position in the pipeline, from 0.
    pub fn stage_index(&self) -> u16 {
        self.stage_index
    }

    pub fn num_stages(&self) -> u16 {
        self.num_stages
    }

    /// The `[start, end)` decoder layers this stage runs.
    pub fn layer_range(&self) -> (u32, u32) {
        self.layer_range
    }

    pub fn layer_counts(&self) -> StageLayerCounts {
        self.counts
    }

    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let root = self.compile(&mut builder);
        builder.finish(root)
    }

    /// The stage's subtree, minted into a caller's builder so the
    /// whole-pipeline view can lay the stages end to end.
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let mut children = Vec::new();
        if let Some(embedding) = &self.embedding {
            children.push(embedding.compile(builder));
        }
        if let Some(dense) = &self.dense {
            children.push(CostNode::Labeled {
                label: format!(
                    "{} dense + full index layers (Scale {})",
                    self.counts.dense, self.counts.dense
                ),
                child: Box::new(CostNode::Scale {
                    n: self.counts.dense,
                    child: Box::new(CostNode::Sum(vec![
                        dense.attention.compile(builder),
                        dense.ffn.compile(builder),
                    ])),
                }),
            });
        }
        for (body, count, kind) in [
            (
                &self.initial_shared,
                self.counts.initial_shared,
                "sparse + IndexShare (before the first cycle)",
            ),
            (
                &self.cycle_full,
                self.counts.cycle_full,
                "sparse + full index (cycle)",
            ),
            (
                &self.cycle_shared,
                self.counts.cycle_shared,
                "sparse + IndexShare (cycle)",
            ),
        ] {
            if let Some(body) = body {
                children.push(CostNode::Labeled {
                    label: format!("{count} {kind} layers (Scale {count})"),
                    child: Box::new(CostNode::Scale {
                        n: count,
                        child: Box::new(body.compile(builder)),
                    }),
                });
            }
        }
        if let Some(head) = &self.head {
            children.push(CostNode::Labeled {
                label: format!(
                    "{}.main output head [final residual RMSNorm -> full-vocab lm_head]",
                    self.name
                ),
                child: Box::new(CostNode::Sum(vec![
                    head.final_norm.compile(builder),
                    head.lm_head.compile(builder),
                ])),
            });
        }
        CostNode::Labeled {
            label: format!(
                "{} pipeline stage {} of {} (Glm52VllmNvfp4PpStageModel) [layers {}..{}; \
                 EP1 on one GPU, no collectives; timing_context<={}]",
                self.name,
                self.stage_index,
                self.num_stages,
                self.layer_range.0,
                self.layer_range.1,
                self.max_model_len
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    /// Streams this stage's leaves in `compile` order.
    fn eval_normalized(&self, batch: &NormalizedBatch, ev: &mut Evaluator) {
        let group = &batch.groups[0];
        if let Some(embedding) = &self.embedding {
            eval_atomic_or_zero(
                embedding,
                ElementwiseKernelInput {
                    num_tokens: group.batch_tokens,
                },
                group.batch_tokens == 0,
                ev,
            );
        }
        if let Some(dense) = &self.dense {
            dense.attention.eval(&group.attention_input, ev);
            // No all-reduce owns the post-attention residual RMSNorm here.
            dense.ffn.eval_with_post_attn_norm(
                &Glm52DenseFfnLocalWorkletInput {
                    batch_tokens: group.batch_tokens,
                },
                true,
                ev,
            );
        }
        for body in [&self.initial_shared, &self.cycle_full, &self.cycle_shared]
            .into_iter()
            .flatten()
        {
            body.eval(batch, ev);
        }
        if let Some(head) = &self.head {
            eval_atomic_or_zero(
                &head.final_norm,
                ResidualRmsNormKernelInput {
                    m: group.batch_tokens,
                },
                group.batch_tokens == 0,
                ev,
            );
            eval_atomic_or_zero(
                &head.lm_head,
                SingleGemmKernelInput {
                    m: group.logits_rows,
                },
                group.logits_rows == 0,
                ev,
            );
        }
    }

    fn normalize(&self, input: &UnifiedArchInput) -> NormalizedBatch {
        normalize_input(input, 1, self.max_model_len)
            .unwrap_or_else(|reason| panic!("invalid Glm52VllmNvfp4PpStageModel input: {reason}"))
    }
}

impl IterwiseUnifiedModel for Glm52VllmNvfp4PpStageModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        normalize_input(batch, 1, self.max_model_len).map(|_| ())
    }

    /// This stage's layers only: vLLM allocates cache per stage.
    fn total_kv_bytes_per_token(&self) -> u64 {
        self.kv_bytes_per_token
    }

    fn gpus_per_replica(&self) -> u16 {
        1
    }

    fn num_attn_dp_groups(&self) -> u16 {
        1
    }

    fn num_attn_shards(&self) -> u16 {
        1
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        let batch = self.normalize(batch);
        eval_compiled(&self.cost_flat, self.n_slots, slots, scratch, None, |ev| {
            self.eval_normalized(&batch, ev)
        })
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        let batch = self.normalize(batch);
        eval_compiled(
            &self.cost_flat,
            self.n_slots,
            slots,
            scratch,
            Some(inputs),
            |ev| self.eval_normalized(&batch, ev),
        )
    }
}

/// The whole pipeline as one iteration: every stage in order, summed.
///
/// This is one microbatch's compute latency through the pipeline, without the
/// stage-to-stage transfers and without the overlap of microbatches in
/// flight -- those belong to the pipeline workers that run the stages. The
/// stages are shared by `Arc`, so a deployment hands the same built stage to
/// its worker that this view costs.
pub struct Glm52VllmNvfp4PpDsaMoeModel {
    name: String,
    stages: Vec<Arc<Glm52VllmNvfp4PpStageModel>>,
    max_model_len: u32,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

impl Glm52VllmNvfp4PpDsaMoeModel {
    pub fn stages(&self) -> &[Arc<Glm52VllmNvfp4PpStageModel>] {
        &self.stages
    }

    pub fn pp_size(&self) -> u16 {
        self.stages.len() as u16
    }

    /// Cache bytes per token of the most constrained stage. vLLM sizes one
    /// shared block table by the stage with the fewest blocks, so a token's
    /// capacity on every GPU is set by the stage with the most layers.
    pub fn pipeline_kv_bytes_per_token(&self) -> u64 {
        self.stages
            .iter()
            .map(|stage| stage.kv_bytes_per_token)
            .max()
            .unwrap_or(0)
    }

    /// Bytes one token sends from one stage to the next: vLLM's PP
    /// `IntermediateTensors` carry `hidden_states` and `residual`, both BF16.
    pub fn activation_bytes_per_token(&self) -> u64 {
        2 * u64::from(HIDDEN_DIM) * u64::from(DType::Bf16.size_bytes())
    }

    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let stages = self
            .stages
            .iter()
            .map(|stage| stage.compile(&mut builder))
            .collect();
        let root = CostNode::Labeled {
            label: format!(
                "{} (Glm52VllmNvfp4PpDsaMoeModel) [PP{}; one EP1 GPU per stage; stages in \
                 sequence, no transfer; timing_context<={}]",
                self.name,
                self.stages.len(),
                self.max_model_len
            ),
            child: Box::new(CostNode::Sum(stages)),
        };
        builder.finish(root)
    }

    fn normalize(&self, input: &UnifiedArchInput) -> NormalizedBatch {
        normalize_input(input, 1, self.max_model_len)
            .unwrap_or_else(|reason| panic!("invalid Glm52VllmNvfp4PpDsaMoeModel input: {reason}"))
    }

    fn eval_normalized(&self, batch: &NormalizedBatch, ev: &mut Evaluator) {
        for stage in &self.stages {
            stage.eval_normalized(batch, ev);
        }
    }
}

impl IterwiseUnifiedModel for Glm52VllmNvfp4PpDsaMoeModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        normalize_input(batch, 1, self.max_model_len).map(|_| ())
    }

    /// The physical total over all stages: each stage caches only its own
    /// layers, so the stages' bytes add up to one token's whole state.
    fn total_kv_bytes_per_token(&self) -> u64 {
        self.stages
            .iter()
            .map(|stage| stage.kv_bytes_per_token)
            .sum()
    }

    fn gpus_per_replica(&self) -> u16 {
        self.pp_size()
    }

    fn num_attn_dp_groups(&self) -> u16 {
        1
    }

    /// Every stage's GPU holds a slice of each token's cache.
    fn num_attn_shards(&self) -> u16 {
        self.pp_size()
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        let batch = self.normalize(batch);
        eval_compiled(&self.cost_flat, self.n_slots, slots, scratch, None, |ev| {
            self.eval_normalized(&batch, ev)
        })
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        let batch = self.normalize(batch);
        eval_compiled(
            &self.cost_flat,
            self.n_slots,
            slots,
            scratch,
            Some(inputs),
            |ev| self.eval_normalized(&batch, ev),
        )
    }
}

/// Fill `slots` through `fill` and aggregate the compiled tree, capturing slot
/// inputs when `inputs` is given.
fn eval_compiled(
    cost_flat: &[FlatCostNode],
    n_slots: usize,
    slots: &mut Vec<LeafMetrics>,
    scratch: &mut Vec<LeafMetrics>,
    inputs: Option<&mut Vec<SlotInput>>,
    fill: impl FnOnce(&mut Evaluator),
) -> LeafMetrics {
    slots.clear();
    slots.resize(n_slots, LeafMetrics::ZERO);
    match inputs {
        Some(inputs) => {
            let mut evaluator = Evaluator::with_inputs(slots, &mut *inputs);
            fill(&mut evaluator);
            assert_eq!(
                evaluator.filled(),
                n_slots,
                "eval must fill every compiled slot"
            );
            assert_eq!(
                inputs.len(),
                n_slots,
                "slot inputs must align with compiled slots"
            );
        }
        None => {
            let mut evaluator = Evaluator::new(slots);
            fill(&mut evaluator);
            assert_eq!(
                evaluator.filled(),
                n_slots,
                "eval must fill every compiled slot"
            );
        }
    }
    CostTree::aggregate(cost_flat, slots, scratch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::ArchGroupInput;
    use crate::arch::glm52_vllm_nvfp4_dsa_moe::{state_bytes_per_token, NUM_LAYERS};
    use crate::test_helpers::lm;
    use crate::timing::routing::RoutingDistribution;
    use std::collections::BTreeSet;
    use std::path::Path;

    fn model_cfg(stem: &str) -> Glm52ModelCfg {
        Glm52ModelCfg::from_json(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("model/config/{stem}.json")),
        )
        .unwrap()
    }

    fn parallel(pp_size: u16) -> Glm52VllmNvfp4PpParallel {
        Glm52VllmNvfp4PpParallel {
            pp_size,
            max_model_len: 131_072,
            gpu_name: "NVIDIA B200".to_string(),
        }
    }

    fn uniform() -> ExpertDemand {
        ExpertDemand::popularity(&RoutingDistribution::uniform(256), 1)
    }

    fn built(stem: &str, pp_size: u16) -> Glm52VllmNvfp4PpDsaMoeModel {
        let cfgs = build_configs(&model_cfg(stem), &parallel(pp_size), &uniform(), false).unwrap();
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        build("unified".to_string(), resolve_configs(&cfgs), &bridge).unwrap()
    }

    #[test]
    fn pp_indices_follow_vllm_get_pp_indices() {
        // The remainder goes to the stages before the last, nearest it first.
        assert_eq!(
            pp_indices(78, 4),
            vec![(0, 19), (19, 39), (39, 59), (59, 78)]
        );
        let sizes = |n, p| {
            pp_indices(n, p)
                .into_iter()
                .map(|(start, end)| end - start)
                .collect::<Vec<_>>()
        };
        assert_eq!(sizes(78, 4), vec![19, 20, 20, 19]);
        assert_eq!(sizes(78, 8), vec![9, 10, 10, 10, 10, 10, 10, 9]);
        assert_eq!(sizes(78, 2), vec![39, 39]);
        // Remainder pp_size - 1 reaches stage 0 too, as vLLM's loop does.
        assert_eq!(sizes(11, 4), vec![3, 3, 3, 2]);
        assert_eq!(sizes(13, 4), vec![3, 3, 4, 3]);
        for pp_size in [2, 3, 4, 5, 8, 13, 78] {
            let ranges = pp_indices(78, pp_size);
            assert_eq!(ranges.len(), usize::from(pp_size));
            assert_eq!(ranges[0].0, 0);
            assert_eq!(ranges.last().unwrap().1, 78);
            assert!(ranges.windows(2).all(|pair| pair[0].1 == pair[1].0));
        }
    }

    #[test]
    fn layer_types_cover_the_glm52_schedule() {
        assert_eq!(
            StageLayerCounts::of((0, NUM_LAYERS)),
            StageLayerCounts {
                dense: 3,
                initial_shared: 3,
                cycle_full: 18,
                cycle_shared: 54,
            }
        );
        // PP8 stage 0 runs layers 0..9: the dense prefix, the initial
        // IndexShare layers, then the first cycle's full-index layer 6.
        assert_eq!(
            StageLayerCounts::of((0, 9)),
            StageLayerCounts {
                dense: 3,
                initial_shared: 3,
                cycle_full: 1,
                cycle_shared: 2,
            }
        );
        // PP4 stage 1 runs layers 19..39; full-index layers 22, 26, 30, 34, 38.
        assert_eq!(
            StageLayerCounts::of((19, 39)),
            StageLayerCounts {
                dense: 0,
                initial_shared: 0,
                cycle_full: 5,
                cycle_shared: 15,
            }
        );
    }

    #[test]
    fn stages_split_the_whole_view_without_collectives() {
        for stem in ["glm52_nvfp4", "glm53_nvfp4"] {
            for pp_size in [4_u16, 8] {
                let model = built(stem, pp_size);
                let stages = model.stages();
                assert_eq!(model.pp_size(), pp_size);
                assert_eq!(model.gpus_per_replica(), pp_size);
                assert_eq!(stages.len(), usize::from(pp_size));

                // The whole view is the stages laid end to end.
                let whole = model.cost_log_manifest();
                let mut concatenated = Vec::new();
                for (index, stage) in stages.iter().enumerate() {
                    assert_eq!(usize::from(stage.stage_index()), index);
                    assert_eq!(stage.num_stages(), pp_size);
                    assert_eq!(stage.gpus_per_replica(), 1);
                    assert_eq!(stage.num_attn_shards(), 1);
                    assert_eq!(stage.layer_range(), pp_indices(78, pp_size)[index]);
                    let manifest = stage.cost_log_manifest();
                    assert_eq!(manifest.slots.len(), stage.n_slots);
                    concatenated.extend(manifest.slots);
                }
                assert_eq!(model.n_slots, concatenated.len());
                assert_eq!(whole.slots, concatenated);

                // A stage reduces nothing.
                for slot in &whole.slots {
                    assert!(
                        !slot.kind.starts_with("all_reduce") && !slot.name.contains("allreduce"),
                        "{} ({}) is a collective",
                        slot.name,
                        slot.kind
                    );
                }

                // Embedding on the first stage, output head on the last.
                for (index, stage) in stages.iter().enumerate() {
                    let names: BTreeSet<String> = stage
                        .cost_log_manifest()
                        .slots
                        .into_iter()
                        .map(|slot| slot.name)
                        .collect();
                    let first = index == 0;
                    let last = index + 1 == stages.len();
                    assert_eq!(names.contains("unified.main.embedding"), first);
                    assert_eq!(names.contains("unified.main.lm_head"), last);
                    assert_eq!(names.contains("unified.main.final_residual_rms_norm"), last);
                }
            }
        }
    }

    #[test]
    fn the_last_stage_bills_the_whole_vocabulary() {
        let cfgs =
            build_configs(&model_cfg("glm52_nvfp4"), &parallel(4), &uniform(), false).unwrap();
        assert_eq!(cfgs.stage.lm_head.n.get(), 154_880);
        assert_eq!(cfgs.stage.parallel.ep_size, 1);
        assert_eq!(cfgs.stage.nvfp4_moe.len(), 1);
        let resolved = resolve_configs(&cfgs);
        assert_eq!(resolved.stage.nvfp4_moe[0].experts_per_device, 256);
        // All 64 heads are local.
        assert_eq!(
            resolved
                .stage
                .dense_full_index_attention
                .main_rope
                .num_heads,
            64
        );
    }

    #[test]
    fn whole_view_cost_is_the_sum_of_its_stages() {
        for pp_size in [4_u16, 8] {
            let model = built("glm52_nvfp4", pp_size);
            // Exact small integers, so f32 sums are order independent.
            let slots: Vec<LeafMetrics> = (0..model.n_slots)
                .map(|slot| lm(((slot * 7) % 5 + 1) as f64))
                .collect();
            let mut scratch = Vec::new();
            let whole = CostTree::aggregate(&model.cost_flat, &slots, &mut scratch);
            let mut offset = 0;
            let mut summed = 0.0_f32;
            for stage in model.stages() {
                let part = &slots[offset..offset + stage.n_slots];
                summed += CostTree::aggregate(&stage.cost_flat, part, &mut scratch)
                    .m
                    .time_ms;
                offset += stage.n_slots;
            }
            assert_eq!(offset, model.n_slots);
            assert!(whole.m.time_ms > 0.0);
            assert_eq!(whole.m.time_ms, summed);
        }
    }

    #[test]
    fn cache_and_activation_bytes_follow_the_stage_layers() {
        // 576 B FP8 MLA latent + rope, 128 B FP8 index key + 4 B FP32 scale.
        assert_eq!(decoder_layer_state_bytes_per_token(), 708);
        for pp_size in [4_u16, 8] {
            let model = built("glm52_nvfp4", pp_size);
            let mut max = 0;
            for stage in model.stages() {
                let (start, end) = stage.layer_range();
                let bytes = u64::from(end - start) * decoder_layer_state_bytes_per_token();
                assert_eq!(stage.total_kv_bytes_per_token(), bytes);
                max = max.max(bytes);
            }
            // The physical total is the EP1 graph's whole-model state.
            assert_eq!(
                model.total_kv_bytes_per_token(),
                state_bytes_per_token(1, Glm52MtpMode::Off).unwrap()
            );
            assert_eq!(model.pipeline_kv_bytes_per_token(), max);
            assert_eq!(model.activation_bytes_per_token(), 2 * 6_144 * 2);
        }
        let pp4 = built("glm52_nvfp4", 4);
        assert_eq!(pp4.pipeline_kv_bytes_per_token(), 20 * 708);
    }

    #[test]
    fn invalid_pipelines_fail_closed() {
        let model = model_cfg("glm52_nvfp4");
        for pp_size in [0_u16, 1, 79] {
            let error = build_configs(&model, &parallel(pp_size), &uniform(), false).unwrap_err();
            assert!(error.to_string().contains("pp_size"), "{error}");
        }
        let mut too_long = parallel(4);
        too_long.max_model_len = CHECKPOINT_MAX_CONTEXT + 1;
        assert!(build_configs(&model, &too_long, &uniform(), false).is_err());
        let mut empty = parallel(4);
        empty.max_model_len = 0;
        assert!(build_configs(&model, &empty, &uniform(), false).is_err());

        // A recipe sharded over a rank group cannot build a stage.
        let ep4 = ep_graph::build_configs(
            &model,
            &Glm52VllmNvfp4DsaMoeParallel {
                ep_size: 4,
                nvl_num_gpu: 4,
                max_model_len: 131_072,
                gpu_name: "NVIDIA B200".to_string(),
            },
            &uniform(),
            None,
            false,
            Glm52MtpMode::Off,
        )
        .unwrap();
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let error = build(
            "unified".to_string(),
            Glm52VllmNvfp4PpResolved {
                parallel: parallel(4),
                stage: ep_graph::resolve_configs(&ep4),
            },
            &bridge,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("EP1"), "{error}");
    }

    #[test]
    fn input_validation_matches_the_ep_graph() {
        let model = built("glm52_nvfp4", 4);
        let group = |pairs: Vec<(u32, u32)>| ArchGroupInput {
            batch_tokens: pairs.iter().map(|&(_, append)| append).sum(),
            prefill_tokens: pairs.iter().map(|&(_, append)| append).sum(),
            decode_tokens: 0,
            prefill_chunk_pairs: pairs,
            decode_kv_lens: Vec::new(),
            total_kv_len: 0,
        };
        let one = UnifiedArchInput {
            groups: vec![group(vec![(0, 4_096)])],
            tokens_per_source_rank: Vec::new(),
        };
        assert!(model.check_input(&one).is_ok());
        assert!(model.stages()[2].check_input(&one).is_ok());
        let two = UnifiedArchInput {
            groups: vec![group(vec![(0, 8)]), group(vec![(0, 8)])],
            tokens_per_source_rank: Vec::new(),
        };
        assert!(model.check_input(&two).is_err());
        let too_long = UnifiedArchInput {
            groups: vec![group(vec![(131_000, 4_096)])],
            tokens_per_source_rank: Vec::new(),
        };
        assert!(model.check_input(&too_long).is_err());
    }

    #[test]
    fn location_map_matches_every_unique_noncommunication_manifest_location() {
        let map_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("model/work/location_maps/glm52_vllm_nvfp4_dsa_moe_unified.json");
        let map: serde_json::Value =
            serde_json::from_slice(&std::fs::read(map_path).unwrap()).unwrap();
        assert!(map["arch_types"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tag| tag == ARCH_KIND));
        let mapped_locations: BTreeSet<String> = map["locations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["location"].as_str().unwrap().to_string())
            .collect();
        for pp_size in [4_u16, 8] {
            let manifest_locations: BTreeSet<String> = built("glm52_nvfp4", pp_size)
                .cost_log_manifest()
                .slots
                .into_iter()
                .filter(|slot| {
                    !matches!(
                        slot.kind.as_str(),
                        "all_reduce" | "all_reduce_fusion" | "all_reduce_residual_rms_norm"
                    )
                })
                .map(|slot| slot.name)
                .collect();
            assert_eq!(manifest_locations.len(), 114);
            assert_eq!(mapped_locations, manifest_locations);
        }
    }
}
