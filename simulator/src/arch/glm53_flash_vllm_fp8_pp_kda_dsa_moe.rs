//! GLM-5.3-Flash (FP8 block or NVFP4, [`Glm53FlashQuant`]) under pure pipeline
//! parallelism, aligned to vLLM on B200.
//!
//! [`Glm53FlashQuant`]: crate::arch::glm53_flash_vllm_fp8_kda_dsa_moe::Glm53FlashQuant
//!
//! Each pipeline stage is one GPU running a contiguous range of the 45 decoder
//! layers at TP1 / EP1: all 64 KDA heads, all 64 MLA heads, and all 288 routed
//! experts are local, so a stage reduces nothing and builds no collective. The
//! leaves are the TP = EP graph's (`glm53_flash_vllm_fp8_kda_dsa_moe.rs`), built
//! from its recipe at `tp_size = 1` without the sublayer all-reduces and named
//! as that graph names them, so its semantic location map applies unchanged. A
//! stage's identity lives only in the `Labeled` node that wraps it.
//!
//! The layer split follows vLLM's `get_pp_indices`
//! ([`crate::arch::glm52_vllm_nvfp4_pp_dsa_moe::pp_indices`]) unless
//! `layer_partition` gives the per-stage layer counts, as vLLM's
//! `VLLM_PP_LAYER_PARTITION` does. The layer types
//! differ per stage: which of the 34 KDA and 11 DSA layers, and whether the
//! three dense-FFN layers, fall in a stage's range set its cost and its cache.
//! Stage 0 also runs the token embedding and the mHC expand; the last stage
//! runs the terminal mHC post, the `hc_contract` mean, the final RMSNorm, and
//! the full-vocabulary lm_head.
//!
//! vLLM fork status: at the pinned commit `Glm5NextForConditionalGeneration`
//! declares `SupportsPP` but never defines `make_empty_intermediate_tensors`, so
//! every rank after the first fails at startup, and its `IntermediateTensors`
//! drop the pending mHC `post`/`comb`. This arch models PP as it runs once that
//! gap is closed the DeepSeek-V4 way: each stage finishes its last `hc_post` and
//! sends the `hc_mult`-wide residual stream. A standalone post plus the next
//! stage's standalone pre cost what the one fused post+pre they replace costs
//! (the terminal post is itself priced as fused minus pre), so a stage boundary
//! adds no leaf.
//!
//! Cache follows vLLM's hybrid manager under PP (`kv_cache_utils.py`
//! `_get_kv_cache_groups_glm5_next`): one block pool, block size
//! [`hybrid_block_tokens`] at TP1 (8576), one attention group for the 11 DSA
//! layers, one kpool-tail group, and `G` KDA groups, `G` the largest
//! `ceil(kda / dsa)` over the stages and the whole model. KDA pages live inside
//! the DSA tensors, so a stage's bytes per block are its DSA layers' alone, and
//! a stage with KDA but no DSA layer is rejected, as vLLM rejects it.
//!
//! Two models live here. [`Glm53FlashVllmFp8PpStageModel`] is one stage, the
//! unit a pipeline worker runs. [`Glm53FlashVllmFp8PpModel`] is the whole
//! pipeline seen as one iteration -- every stage in order, no transfer and no
//! overlap -- which is what offline prediction and supported-build listing cost.

use std::sync::Arc;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::glm52_vllm_nvfp4_pp_dsa_moe::{eval_compiled, pp_indices};
use crate::arch::glm53_flash_vllm_fp8_kda_dsa_moe::{
    self as flash, atomic, build_layer_group, dsa_layer_bytes_per_token, graph_padded_tokens,
    hybrid_block_tokens, kda_layer_state_bytes_per_request, normalize_input, push,
    Glm53FlashModelCfg, Glm53FlashVllmConfigs, Glm53FlashVllmParallel, Glm53FlashVllmResolved,
    LayerGroup, NormalizedBatch, ACTIVATION_DTYPE, MHC_BACKENDS,
};
use crate::op::mhc::{MhcTerminalPostConfig, MhcTerminalPostInput, MhcTerminalPostOp};
use crate::op::Op;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelInput, RmsNormKernel, RmsNormKernelInput, SingleGemmKernel,
    SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};

const ARCH_KIND: &str = "glm53_flash_vllm_fp8_pp_kda_dsa_moe";

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

#[derive(Clone, Debug)]
pub struct Glm53FlashVllmFp8PpParallel {
    /// Pipeline stages, one GPU each.
    pub pp_size: u16,
    pub max_model_len: u32,
    pub gpu_name: String,
    /// vLLM `--cudagraph-capture-sizes`; empty runs eager (no padding).
    pub cudagraph_capture_sizes: Vec<u32>,
    /// vLLM `VLLM_PP_LAYER_PARTITION`: layers per stage, in stage order.
    /// Empty: vLLM's default `get_pp_indices` split.
    pub layer_partition: Vec<u32>,
}

/// Each stage's `[start, end)` layer range: `layer_partition` when given,
/// else vLLM's `get_pp_indices`.
pub fn stage_ranges(
    num_layers: u32,
    parallel: &Glm53FlashVllmFp8PpParallel,
) -> Result<Vec<(u32, u32)>, BuildError> {
    validate_pp_size(parallel.pp_size, num_layers)?;
    let partition = &parallel.layer_partition;
    if partition.is_empty() {
        return Ok(pp_indices(num_layers, parallel.pp_size));
    }
    if partition.len() != usize::from(parallel.pp_size)
        || partition.contains(&0)
        || partition.iter().sum::<u32>() != num_layers
    {
        return Err(fit_failed(format!(
            "layer_partition {partition:?} must give {} positive layer counts summing to {num_layers}",
            parallel.pp_size
        )));
    }
    let mut start = 0;
    Ok(partition
        .iter()
        .map(|&layers| {
            let range = (start, start + layers);
            start += layers;
            range
        })
        .collect())
}

/// The pipeline's layout plus the TP1 recipe every stage builds from.
#[derive(Clone, Debug)]
pub struct Glm53FlashVllmFp8PpConfigs {
    pub parallel: Glm53FlashVllmFp8PpParallel,
    pub stage: Glm53FlashVllmConfigs,
}

#[derive(Clone, Debug)]
pub struct Glm53FlashVllmFp8PpResolved {
    pub parallel: Glm53FlashVllmFp8PpParallel,
    pub stage: Glm53FlashVllmResolved,
}

/// The TP = EP graph's recipe on one GPU. `demand` is the 42 routed layers'
/// demand at EP1; every stage prices its experts against it, as the TP graph
/// does for every layer.
pub fn build_configs(
    model: &Glm53FlashModelCfg,
    parallel: &Glm53FlashVllmFp8PpParallel,
    demand: &ExpertDemand,
) -> Result<Glm53FlashVllmFp8PpConfigs, BuildError> {
    kda_group_count(model, &stage_ranges(model.num_layers, parallel)?)?;
    if parallel.max_model_len == 0 {
        return Err(fit_failed("max_model_len must be positive"));
    }
    let stage = flash::build_configs(
        model,
        &Glm53FlashVllmParallel {
            tp_size: 1,
            // One GPU owns all 288 experts either way.
            enable_expert_parallel: true,
            max_model_len: parallel.max_model_len,
            gpu_name: parallel.gpu_name.clone(),
            cudagraph_capture_sizes: parallel.cudagraph_capture_sizes.clone(),
        },
        demand,
    )?;
    Ok(Glm53FlashVllmFp8PpConfigs {
        parallel: parallel.clone(),
        stage,
    })
}

pub fn resolve_configs(cfgs: &Glm53FlashVllmFp8PpConfigs) -> Glm53FlashVllmFp8PpResolved {
    Glm53FlashVllmFp8PpResolved {
        parallel: cfgs.parallel.clone(),
        stage: flash::resolve_configs(&cfgs.stage),
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

/// vLLM's `_pp_balanced_mamba_group_count`: the KDA cache-group count `G`. Each
/// KDA group stores one layer per stage inside one of that stage's DSA tensors,
/// so every stage needs `G >= ceil(kda / dsa)`, and a stage with KDA layers but
/// no DSA layer has nowhere to put them.
pub fn kda_group_count(
    model: &Glm53FlashModelCfg,
    stage_ranges: &[(u32, u32)],
) -> Result<u32, BuildError> {
    let pp_size = stage_ranges.len();
    let whole = StageLayerCounts::of(model, (0, model.num_layers));
    let mut groups = whole.kda.div_ceil(whole.dsa.max(1));
    for (stage, &range) in stage_ranges.iter().enumerate() {
        let counts = StageLayerCounts::of(model, range);
        if counts.kda == 0 {
            continue;
        }
        if counts.dsa == 0 {
            return Err(fit_failed(format!(
                "PP{pp_size} stage {stage} (layers {}..{}) has KDA layers but no DSA layer; \
                 vLLM's hybrid cache grouping cannot place its state",
                range.0, range.1
            )));
        }
        groups = groups.max(counts.kda.div_ceil(counts.dsa));
    }
    Ok(groups)
}

/// Build every stage, then the whole-pipeline view over them.
pub fn build(
    name: String,
    resolved: Glm53FlashVllmFp8PpResolved,
    bridge: &PerfApiBridge,
) -> Result<Glm53FlashVllmFp8PpModel, BuildError> {
    let recipe = &resolved.stage.raw_cfg;
    let ranges = stage_ranges(recipe.model.num_layers, &resolved.parallel)?;
    if recipe.parallel.tp_size != 1 || recipe.routed.len() != 1 {
        return Err(fit_failed(format!(
            "every pipeline stage runs at TP1 / EP1 on one GPU, got tp_size {} and {} routed ranks",
            recipe.parallel.tp_size,
            recipe.routed.len()
        )));
    }
    let pp_size = resolved.parallel.pp_size;
    let kda_groups = kda_group_count(&recipe.model, &ranges)?;
    let block_tokens = hybrid_block_tokens(
        kda_layer_state_bytes_per_request(&resolved.stage),
        u64::from(recipe.model.kv_lora_rank),
    );
    let sections = StageSections::build(&name, &resolved.stage, bridge)?;
    let stages = ranges
        .into_iter()
        .enumerate()
        .map(|(index, layers)| {
            Glm53FlashVllmFp8PpStageModel::build(
                &name,
                &resolved.stage,
                &sections,
                index as u16,
                pp_size,
                layers,
                block_tokens,
            )
            .map(Arc::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut model = Glm53FlashVllmFp8PpModel {
        name,
        stages,
        max_model_len: recipe.parallel.max_model_len,
        hidden: recipe.model.hidden,
        hc_mult: recipe.model.hc_mult,
        block_tokens,
        kda_groups,
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

/// Stage 0's head: token embedding, then the mHC expand to `hc_mult` streams.
struct InputHead {
    embedding: Op<ElementwiseKernel>,
    hc_expand: Op<ElementwiseKernel>,
}

/// The last stage's tail.
struct OutputHead {
    terminal_post: MhcTerminalPostOp,
    hc_contract_mean: Op<ElementwiseKernel>,
    final_norm: Op<RmsNormKernel>,
    /// Full vocabulary: a TP1 `ParallelLMHead` shards nothing.
    lm_head: Op<SingleGemmKernel>,
}

/// Every section of the graph, built once and shared by the stages that run
/// it. A section's kernels do not depend on which stage runs it.
struct StageSections {
    input: Arc<InputHead>,
    groups: Vec<Arc<LayerGroup>>,
    output: Arc<OutputHead>,
}

impl StageSections {
    fn build(
        name: &str,
        resolved: &Glm53FlashVllmResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let cfg = &resolved.raw_cfg;
        let groups = cfg
            .groups
            .iter()
            .map(|group| build_layer_group(name, group, resolved, false, bridge).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            input: Arc::new(InputHead {
                embedding: atomic(
                    name,
                    "embedding",
                    cfg.embedding.clone(),
                    ElementwiseKernel::build,
                    bridge,
                )?,
                hc_expand: atomic(
                    name,
                    "hc_expand",
                    cfg.hc_expand.clone(),
                    ElementwiseKernel::build,
                    bridge,
                )?,
            }),
            groups,
            output: Arc::new(OutputHead {
                terminal_post: MhcTerminalPostOp::build(
                    format!("{name}.final_mhc_post"),
                    MhcTerminalPostConfig {
                        mhc: cfg.mhc.clone(),
                        pre_backends: MHC_BACKENDS.to_vec(),
                        fused_backends: MHC_BACKENDS.to_vec(),
                    },
                    bridge,
                )?,
                hc_contract_mean: atomic(
                    name,
                    "hc_contract_mean",
                    cfg.hc_contract_mean.clone(),
                    ElementwiseKernel::build,
                    bridge,
                )?,
                final_norm: atomic(
                    name,
                    "final_norm",
                    cfg.final_norm.clone(),
                    RmsNormKernel::build,
                    bridge,
                )?,
                lm_head: atomic(
                    name,
                    "lm_head",
                    cfg.lm_head.clone(),
                    SingleGemmKernel::build,
                    bridge,
                )?,
            }),
        })
    }
}

/// How many layers of each kind one stage runs, and so what it caches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageLayerCounts {
    /// Kimi Delta Attention layers: per-request recurrent state.
    pub kda: u32,
    /// DeepSeek sparse-attention layers: per-token MLA latent + kpool index.
    pub dsa: u32,
    /// Dense-FFN layers (the first three).
    pub dense_ffn: u32,
}

impl StageLayerCounts {
    pub fn of(model: &Glm53FlashModelCfg, layers: (u32, u32)) -> Self {
        let mut counts = Self::default();
        for layer in layers.0..layers.1 {
            if model.kda_layers.contains(&layer) {
                counts.kda += 1;
            } else {
                counts.dsa += 1;
            }
            if layer < model.first_k_dense_replace {
                counts.dense_ffn += 1;
            }
        }
        counts
    }

    pub fn total(&self) -> u32 {
        self.kda + self.dsa
    }
}

/// One pipeline stage: one GPU and its contiguous layer range.
///
/// Each section exists only when the stage runs it, so a stage mints no slot
/// for work it never does.
pub struct Glm53FlashVllmFp8PpStageModel {
    name: String,
    stage_index: u16,
    num_stages: u16,
    layer_range: (u32, u32),
    counts: StageLayerCounts,
    input: Option<Arc<InputHead>>,
    /// Each layer group this stage runs, with how many of its layers.
    groups: Vec<(Arc<LayerGroup>, u32)>,
    output: Option<Arc<OutputHead>>,
    max_model_len: u32,
    cudagraph_capture_sizes: Vec<u32>,
    kv_bytes_per_token: u64,
    recurrent_state_bytes_per_request: u64,
    block_tokens: u32,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

impl Glm53FlashVllmFp8PpStageModel {
    fn build(
        name: &str,
        resolved: &Glm53FlashVllmResolved,
        sections: &StageSections,
        stage_index: u16,
        num_stages: u16,
        layer_range: (u32, u32),
        block_tokens: u32,
    ) -> Result<Self, BuildError> {
        let cfg = &resolved.raw_cfg;
        let counts = StageLayerCounts::of(&cfg.model, layer_range);
        let groups = sections
            .groups
            .iter()
            .filter_map(|group| {
                let count = group
                    .layers()
                    .iter()
                    .filter(|&&layer| (layer_range.0..layer_range.1).contains(&layer))
                    .count() as u32;
                (count > 0).then(|| (Arc::clone(group), count))
            })
            .collect::<Vec<_>>();
        debug_assert_eq!(
            groups.iter().map(|(_, count)| count).sum::<u32>(),
            counts.total()
        );
        // vLLM allocates cache only for the layers a stage owns.
        let kv_bytes_per_token = u64::from(counts.dsa) * dsa_layer_bytes_per_token(&cfg.model);
        let recurrent_state_bytes_per_request =
            u64::from(counts.kda) * kda_layer_state_bytes_per_request(resolved);
        let mut sizes = cfg.parallel.cudagraph_capture_sizes.clone();
        sizes.sort_unstable();
        sizes.dedup();
        let mut stage = Self {
            name: name.to_string(),
            stage_index,
            num_stages,
            layer_range,
            counts,
            input: (stage_index == 0).then(|| Arc::clone(&sections.input)),
            groups,
            output: (stage_index + 1 == num_stages).then(|| Arc::clone(&sections.output)),
            max_model_len: cfg.parallel.max_model_len,
            cudagraph_capture_sizes: sizes,
            kv_bytes_per_token,
            recurrent_state_bytes_per_request,
            block_tokens,
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
        if let Some(input) = &self.input {
            children.push(input.embedding.compile(builder));
            children.push(input.hc_expand.compile(builder));
        }
        let (start, end) = self.layer_range;
        for (group, count) in &self.groups {
            // The stage's own layers of this group, so a manifest says which
            // layer each section folds (tools/pp-layer-balance reads them).
            let layers: Vec<u32> = group
                .layers()
                .iter()
                .copied()
                .filter(|layer| (start..end).contains(layer))
                .collect();
            children.push(CostNode::Labeled {
                label: format!(
                    "{} x{count} (Scale {count}) layers {layers:?}",
                    group.label()
                ),
                child: Box::new(CostNode::Scale {
                    n: *count,
                    child: Box::new(group.compile_layer(builder)),
                }),
            });
        }
        if let Some(output) = &self.output {
            children.push(CostNode::Labeled {
                label: format!(
                    "{}.main output head [terminal mHC post -> hc_contract mean -> final \
                     RMSNorm -> full-vocab lm_head]",
                    self.name
                ),
                child: Box::new(CostNode::Sum(vec![
                    output.terminal_post.compile(builder),
                    output.hc_contract_mean.compile(builder),
                    output.final_norm.compile(builder),
                    output.lm_head.compile(builder),
                ])),
            });
        }
        CostNode::Labeled {
            label: format!(
                "{} pipeline stage {} of {} (Glm53FlashVllmFp8PpStageModel) [layers {}..{}: \
                 {} KDA + {} DSA; TP1/EP1 on one GPU, no collectives; timing_context<={}]",
                self.name,
                self.stage_index,
                self.num_stages,
                self.layer_range.0,
                self.layer_range.1,
                self.counts.kda,
                self.counts.dsa,
                self.max_model_len
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    /// Streams this stage's leaves in `compile` order.
    fn eval_normalized(&self, batch: &NormalizedBatch, ev: &mut Evaluator) {
        let tokens = graph_padded_tokens(&self.cudagraph_capture_sizes, batch.total_tokens);
        if let Some(input) = &self.input {
            push(
                &input.embedding,
                ElementwiseKernelInput { num_tokens: tokens },
                ev,
            );
            push(
                &input.hc_expand,
                ElementwiseKernelInput { num_tokens: tokens },
                ev,
            );
        }
        for (group, _) in &self.groups {
            group.eval(batch, tokens, ev);
        }
        if let Some(output) = &self.output {
            output
                .terminal_post
                .eval(&MhcTerminalPostInput { num_tokens: tokens }, ev);
            push(
                &output.hc_contract_mean,
                ElementwiseKernelInput { num_tokens: tokens },
                ev,
            );
            push(&output.final_norm, RmsNormKernelInput { m: tokens }, ev);
            push(
                &output.lm_head,
                SingleGemmKernelInput {
                    m: batch.request_count,
                },
                ev,
            );
        }
    }

    fn normalize(&self, input: &UnifiedArchInput) -> NormalizedBatch {
        normalize_input(input, self.max_model_len).unwrap_or_else(|reason| {
            panic!("invalid Glm53FlashVllmFp8PpStageModel input: {reason}")
        })
    }
}

impl IterwiseUnifiedModel for Glm53FlashVllmFp8PpStageModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        normalize_input(batch, self.max_model_len).map(|_| ())
    }

    /// This stage's DSA layers only: vLLM allocates cache per stage.
    fn total_kv_bytes_per_token(&self) -> u64 {
        self.kv_bytes_per_token
    }

    /// This stage's KDA layers only.
    fn recurrent_state_bytes_per_request(&self) -> u64 {
        self.recurrent_state_bytes_per_request
    }

    fn recurrent_checkpoint_interval_tokens(&self) -> u32 {
        self.block_tokens
    }

    fn logs_decode_kv_lens(&self) -> bool {
        true
    }

    fn gpus_per_replica(&self) -> u16 {
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
/// flight -- those belong to the pipeline workers that run the stages.
pub struct Glm53FlashVllmFp8PpModel {
    name: String,
    stages: Vec<Arc<Glm53FlashVllmFp8PpStageModel>>,
    max_model_len: u32,
    hidden: u32,
    hc_mult: u32,
    block_tokens: u32,
    kda_groups: u32,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

impl Glm53FlashVllmFp8PpModel {
    pub fn stages(&self) -> &[Arc<Glm53FlashVllmFp8PpStageModel>] {
        &self.stages
    }

    pub fn pp_size(&self) -> u16 {
        self.stages.len() as u16
    }

    /// Cache bytes per token on the stage with the most DSA layers. vLLM sizes
    /// one shared block pool by the stage with the fewest blocks.
    pub fn pipeline_kv_bytes_per_token(&self) -> u64 {
        self.stages
            .iter()
            .map(|stage| stage.kv_bytes_per_token)
            .max()
            .unwrap_or(0)
    }

    /// Tokens per cache block, in every group.
    pub fn block_tokens(&self) -> u32 {
        self.block_tokens
    }

    /// vLLM's KDA cache-group count `G`; see [`kda_group_count`].
    pub fn kda_groups(&self) -> u32 {
        self.kda_groups
    }

    /// Blocks a request holds whatever its length: one per KDA group (its live
    /// state) plus one kpool-tail scratch block.
    pub fn state_blocks_per_request(&self) -> u32 {
        self.kda_groups + 1
    }

    /// Bytes one token sends from one stage to the next: vLLM's PP
    /// `IntermediateTensors` carry the `hc_mult`-wide mHC residual stream.
    pub fn activation_bytes_per_token(&self) -> u64 {
        u64::from(self.hc_mult) * u64::from(self.hidden) * u64::from(ACTIVATION_DTYPE.size_bytes())
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
                "{} (Glm53FlashVllmFp8PpModel) [PP{}; one TP1/EP1 GPU per stage; stages in \
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
        normalize_input(input, self.max_model_len)
            .unwrap_or_else(|reason| panic!("invalid Glm53FlashVllmFp8PpModel input: {reason}"))
    }

    fn eval_normalized(&self, batch: &NormalizedBatch, ev: &mut Evaluator) {
        for stage in &self.stages {
            stage.eval_normalized(batch, ev);
        }
    }
}

impl IterwiseUnifiedModel for Glm53FlashVllmFp8PpModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        normalize_input(batch, self.max_model_len).map(|_| ())
    }

    /// The physical total over all stages: each stage caches only its own
    /// layers, so the stages' bytes add up to one token's whole state.
    fn total_kv_bytes_per_token(&self) -> u64 {
        self.stages
            .iter()
            .map(|stage| stage.kv_bytes_per_token)
            .sum()
    }

    fn recurrent_state_bytes_per_request(&self) -> u64 {
        self.stages
            .iter()
            .map(|stage| stage.recurrent_state_bytes_per_request)
            .sum()
    }

    fn recurrent_checkpoint_interval_tokens(&self) -> u32 {
        self.block_tokens
    }

    fn logs_decode_kv_lens(&self) -> bool {
        true
    }

    fn gpus_per_replica(&self) -> u16 {
        self.pp_size()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::ArchGroupInput;
    use crate::test_helpers::lm;
    use crate::timing::routing::RoutingDistribution;
    use std::collections::BTreeSet;
    use std::path::Path;

    fn model_cfg() -> Glm53FlashModelCfg {
        config("glm53_flash")
    }

    fn config(name: &str) -> Glm53FlashModelCfg {
        Glm53FlashModelCfg::from_json(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("model/config/{name}.json")),
        )
        .unwrap()
    }

    fn parallel(pp_size: u16) -> Glm53FlashVllmFp8PpParallel {
        Glm53FlashVllmFp8PpParallel {
            pp_size,
            max_model_len: 131_072,
            gpu_name: "NVIDIA B200".to_string(),
            cudagraph_capture_sizes: Vec::new(),
            layer_partition: Vec::new(),
        }
    }

    fn uniform() -> ExpertDemand {
        ExpertDemand::popularity(&RoutingDistribution::uniform(288), 1)
    }

    fn built_named(name: &str, pp_size: u16) -> Glm53FlashVllmFp8PpModel {
        built_for(&model_cfg(), name, pp_size)
    }

    fn built_for(model: &Glm53FlashModelCfg, name: &str, pp_size: u16) -> Glm53FlashVllmFp8PpModel {
        let cfgs = build_configs(model, &parallel(pp_size), &uniform()).unwrap();
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        build(name.to_string(), resolve_configs(&cfgs), &bridge).unwrap()
    }

    fn built(pp_size: u16) -> Glm53FlashVllmFp8PpModel {
        built_named("unified", pp_size)
    }

    fn sizes(pp_size: u16) -> Vec<u32> {
        pp_indices(45, pp_size)
            .into_iter()
            .map(|(start, end)| end - start)
            .collect()
    }

    #[test]
    fn stages_follow_vllm_get_pp_indices_over_45_layers() {
        assert_eq!(sizes(4), [11, 11, 12, 11]);
        assert_eq!(sizes(8), [5, 5, 6, 6, 6, 6, 6, 5]);
        assert_eq!(sizes(9), [5; 9]);
        assert_eq!(sizes(11), [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4]);
        assert_eq!(sizes(15), [3; 15]);
    }

    #[test]
    fn stage_layer_kinds_follow_the_hybrid_schedule() {
        let model = model_cfg();
        let counts = |range| StageLayerCounts::of(&model, range);
        let kinds = |c: StageLayerCounts| (c.kda, c.dsa, c.dense_ffn);
        // PP4: DSA layers 3, 7 | 11, 15, 19 | 23, 27, 31 | 35, 39, 43.
        let pp4: Vec<_> = pp_indices(45, 4)
            .into_iter()
            .map(|range| kinds(counts(range)))
            .collect();
        assert_eq!(pp4, [(9, 2, 3), (8, 3, 0), (9, 3, 0), (8, 3, 0)]);
        assert_eq!(kinds(counts((0, 45))), (34, 11, 3));
        // One layer per stage: a stage is either a KDA or a DSA layer.
        assert_eq!(kinds(counts((3, 4))), (0, 1, 0));
        assert_eq!(kinds(counts((2, 3))), (1, 0, 1));
    }

    #[test]
    fn stages_split_the_whole_view_without_collectives() {
        for pp_size in [4_u16, 8, 11] {
            let model = built(pp_size);
            let stages = model.stages();
            assert_eq!(model.gpus_per_replica(), pp_size);
            assert_eq!(stages.len(), usize::from(pp_size));
            let whole = model.cost_log_manifest();
            let mut concatenated = Vec::new();
            for (index, stage) in stages.iter().enumerate() {
                assert_eq!(usize::from(stage.stage_index()), index);
                assert_eq!(stage.num_stages(), pp_size);
                assert_eq!(stage.gpus_per_replica(), 1);
                assert_eq!(stage.layer_range(), pp_indices(45, pp_size)[index]);
                let manifest = stage.cost_log_manifest();
                assert_eq!(manifest.slots.len(), stage.n_slots);
                let names: BTreeSet<String> = manifest
                    .slots
                    .iter()
                    .map(|slot| slot.name.clone())
                    .collect();
                let first = index == 0;
                let last = index + 1 == stages.len();
                assert_eq!(names.contains("unified.embedding"), first);
                assert_eq!(names.contains("unified.hc_expand"), first);
                assert_eq!(names.contains("unified.lm_head"), last);
                assert_eq!(names.contains("unified.hc_contract_mean"), last);
                concatenated.extend(manifest.slots);
            }
            assert_eq!(whole.slots, concatenated);
            for slot in &whole.slots {
                assert!(
                    !slot.kind.starts_with("all_reduce") && !slot.name.contains("all_reduce"),
                    "{} ({}) is a collective",
                    slot.name,
                    slot.kind
                );
                assert!(!slot.name.contains("routed_rank1"), "{}", slot.name);
            }
        }
    }

    /// tools/pp-layer-balance maps layers to kinds from these labels.
    #[test]
    fn stage_group_labels_name_their_layers() {
        let model = built(4);
        let mut seen = Vec::new();
        for stage in model.stages() {
            let (start, end) = stage.layer_range();
            let mut stage_layers = Vec::new();
            for label in stage.cost_log_manifest().node_labels.iter().flatten() {
                let Some((head, list)) = label.split_once(" layers [") else {
                    continue;
                };
                if !head.contains("(Scale ") {
                    continue;
                }
                let (_, scale) = head.rsplit_once(" (Scale ").unwrap();
                let count: usize = scale.trim_end_matches(')').parse().unwrap();
                let layers: Vec<u32> = list
                    .trim_end_matches(']')
                    .split(", ")
                    .map(|layer| layer.parse().unwrap())
                    .collect();
                assert_eq!(layers.len(), count, "{label}");
                stage_layers.extend(layers);
            }
            stage_layers.sort_unstable();
            assert_eq!(stage_layers, (start..end).collect::<Vec<_>>());
            seen.extend(stage_layers);
        }
        assert_eq!(seen, (0..45).collect::<Vec<_>>());
    }

    #[test]
    fn one_gpu_holds_every_head_expert_and_the_whole_vocabulary() {
        let cfgs = build_configs(&model_cfg(), &parallel(4), &uniform()).unwrap();
        assert_eq!(cfgs.stage.routed.len(), 1);
        assert_eq!(cfgs.stage.lm_head.n.get(), 154_880);
        let resolved = resolve_configs(&cfgs);
        assert_eq!(resolved.stage.raw_cfg.kda.num_heads.get(), 64);
        assert_eq!(resolved.stage.raw_cfg.dsa.num_heads.get(), 64);
        assert_eq!(resolved.stage.raw_cfg.dense_ffn.intermediate.get(), 12_288);
    }

    #[test]
    fn whole_view_cost_is_the_sum_of_its_stages() {
        for pp_size in [4_u16, 8] {
            let model = built(pp_size);
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
    fn each_stage_caches_only_its_own_layers() {
        let model = built(4);
        let kda_state = 64 * 128 * 128 * 4 + 3 * 8192 * 3 * 2;
        for stage in model.stages() {
            let counts = stage.layer_counts();
            assert_eq!(
                stage.total_kv_bytes_per_token(),
                u64::from(counts.dsa) * (512 + 33)
            );
            assert_eq!(
                stage.recurrent_state_bytes_per_request(),
                u64::from(counts.kda) * kda_state
            );
        }
        // The stages add up to the whole model at TP1.
        assert_eq!(model.total_kv_bytes_per_token(), 11 * 545);
        assert_eq!(model.recurrent_state_bytes_per_request(), 34 * kda_state);
        assert_eq!(model.activation_bytes_per_token(), 4 * 4096 * 2);
    }

    #[test]
    fn kda_groups_and_block_size_follow_vllms_hybrid_manager() {
        let model = model_cfg();
        let groups: Vec<_> = [2_u16, 3, 4, 5, 8, 9, 11]
            .into_iter()
            .map(|pp_size| kda_group_count(&model, &pp_indices(45, pp_size)).unwrap())
            .collect();
        assert_eq!(groups, [4, 4, 5, 4, 5, 4, 4]);
        let pp4 = built(4);
        assert_eq!(pp4.block_tokens(), 8_576);
        assert_eq!(pp4.kda_groups(), 5);
        assert_eq!(pp4.state_blocks_per_request(), 6);
        assert_eq!(pp4.pipeline_kv_bytes_per_token(), 3 * 545);
        for stage in pp4.stages() {
            assert_eq!(stage.recurrent_checkpoint_interval_tokens(), 8_576);
        }
        assert_eq!(built(8).pipeline_kv_bytes_per_token(), 2 * 545);
    }

    #[test]
    fn layer_partition_overrides_the_default_split() {
        let mut balanced = parallel(4);
        balanced.layer_partition = vec![12, 11, 11, 11];
        assert_eq!(
            stage_ranges(45, &balanced).unwrap(),
            [(0, 12), (12, 23), (23, 34), (34, 45)]
        );
        assert_eq!(stage_ranges(45, &parallel(4)).unwrap(), pp_indices(45, 4));
        for bad in [vec![12, 11, 11], vec![12, 11, 11, 12], vec![0, 15, 15, 15]] {
            let mut wrong = parallel(4);
            wrong.layer_partition = bad;
            let error = stage_ranges(45, &wrong).unwrap_err();
            assert!(error.to_string().contains("layer_partition"), "{error}");
        }
        // A stage of only dense KDA layers has no DSA layer to hold their state.
        let mut no_dsa = parallel(4);
        no_dsa.layer_partition = vec![3, 14, 14, 14];
        let error = build_configs(&model_cfg(), &no_dsa, &uniform()).unwrap_err();
        assert!(error.to_string().contains("no DSA layer"), "{error}");
    }

    #[test]
    fn invalid_pipelines_fail_closed() {
        let model = model_cfg();
        for pp_size in [0_u16, 1, 46] {
            let error = build_configs(&model, &parallel(pp_size), &uniform()).unwrap_err();
            assert!(error.to_string().contains("pp_size"), "{error}");
        }
        // From PP12 stage 0 (layers 0-2 or fewer) has KDA layers and no DSA.
        for pp_size in [12_u16, 15, 45] {
            let error = build_configs(&model, &parallel(pp_size), &uniform()).unwrap_err();
            assert!(error.to_string().contains("no DSA layer"), "{error}");
        }
    }

    #[test]
    fn input_validation_matches_the_tp_graph() {
        let model = built(4);
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
        let too_long = UnifiedArchInput {
            groups: vec![group(vec![(131_000, 4_096)])],
            tokens_per_source_rank: Vec::new(),
        };
        assert!(model.check_input(&too_long).is_err());
    }

    fn assert_map_covers(model: &Glm53FlashModelCfg, arch: &str, map: &str, locations: usize) {
        let map: serde_json::Value = serde_json::from_str(map).unwrap();
        assert_eq!(map["arch_types"], serde_json::json!([arch]));
        let mapped: BTreeSet<String> = map["locations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["location"].as_str().unwrap().to_string())
            .collect();
        for pp_size in [4_u16, 8, 11] {
            let actual: BTreeSet<String> = built_for(model, "pp", pp_size)
                .cost_log_manifest()
                .slots
                .into_iter()
                .map(|slot| slot.name)
                .collect();
            assert_eq!(actual.len(), locations);
            assert_eq!(mapped, actual);
        }
    }

    #[test]
    fn pp_location_map_matches_every_noncommunication_manifest_location() {
        assert_map_covers(
            &model_cfg(),
            ARCH_KIND,
            include_str!(
                "../../../model/work/location_maps/glm53_flash_vllm_fp8_pp_kda_dsa_moe_pp.json"
            ),
            116,
        );
    }

    #[test]
    fn nvfp4_pp_map_drops_the_bf16_shared_expert_quant() {
        assert_map_covers(
            &config("glm53_flash_nvfp4"),
            "glm53_flash_vllm_nvfp4_pp_kda_dsa_moe",
            include_str!(
                "../../../model/work/location_maps/glm53_flash_vllm_nvfp4_pp_kda_dsa_moe_pp.json"
            ),
            112,
        );
    }
}
