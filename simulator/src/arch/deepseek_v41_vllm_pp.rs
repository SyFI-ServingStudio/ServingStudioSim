//! DeepSeek-V4.1-Flash under plain pipeline parallelism on B200: each stage is
//! one GPU running a contiguous range of the 40 decoder layers at TP1 / EP1
//! (all 64 attention heads and all 384 routed experts local, no collective).
//!
//! The leaves are the TP arch's ([`deepseek_v41_vllm`]) built at TP1 / EP1,
//! as the DP arch builds its rank-local attention: the same worklets, the
//! same layer fold within a stage, the same step normalization (graph mode,
//! padding, bounded replay).
//!
//! Cuts. Any contiguous partition is modeled. The pinned fork accepts only
//! cuts between KV-sharing groups (`attention.py`: a compressed layer reads
//! the compressed KV and index K cache of the latest KV source at or below
//! it, and a rank without that source raises "PP splits inside a v4.1
//! kv-sharing group are not supported"; with KV sources 2/8/14/20 a stage may
//! start only at 0/1/2/8/14/20), and replays only on a stage holding layer 20
//! and every later layer (`nvidia/model.py` `_decoder_replay_supported`).
//! This arch models the other cuts by their physical cost instead of refusing
//! them ([`fork_accepts`] says which ones the fork runs):
//! - a stage whose layers read a KV source another stage holds keeps its own
//!   copy of that source's compressed cache (and index K cache when one of
//!   its layers is an index non-owner of that source); the copy is a part of
//!   the stage's KV layout ([`stage_kv_cache_layout`]);
//! - every boundary carries, beside the hidden state, the new compressed
//!   entries (and index K rows) of every upstream source read downstream, the
//!   step's top-k indices when the next stage's first layer reuses an upstream
//!   indexer's, and layer 20's candidate blocks when a later candidate
//!   consumer sits downstream ([`DeepseekV41VllmPpStageModel::send_bytes`]);
//! - bounded replay spans stages: the stage holding layer 20 runs its
//!   `write_kv` over every row and the gather; every later layer, on any
//!   stage, runs on the replay rows; a boundary past layer 20 carries only
//!   the replay rows; the last stage scatters back before the head.
//!
//! Per stage, as the fork runs it (`nvidia/model.py`):
//! - the first stage runs the token embedding; the hc expand is layer 0's;
//! - every stage holding an Engram layer runs the n-gram hash and one host-UVA
//!   lookup per local Engram layer (`engram_config.cpu_offload` is on by
//!   default, so the tables sit in pinned host memory, not HBM);
//! - the last stage runs the head (hc collapse, final norm, full-vocabulary
//!   lm_head; no logits gather at TP1).
//!
//! Stage boundary: the sender finishes its last layer with a standalone hc
//! post (`_collapse`) and the receiver's first layer runs a standalone hc pre
//! (`residual is None`), where an unsplit model runs one fused post+pre. As
//! in the GLM-5.3-Flash PP arch the pair is priced as the fused boundary it
//! replaces, so a cut adds no leaf; the receiver keeps its layer's entry.
//!
//! Cache: each stage allocates only its own layers
//! ([`stage_kv_cache_layout`]); the pipeline head sizes token capacity by the
//! stage with the most bytes per token ([`DeepseekV41VllmPpModel::pipeline_kv_cache_layout`]).
//!
//! Two models live here, as in the GLM PP archs: [`DeepseekV41VllmPpStageModel`]
//! is one stage, the unit a pipeline worker runs; [`DeepseekV41VllmPpModel`] is
//! the whole pipeline seen as one iteration (stages in order, no transfer, no
//! overlap), which offline prediction costs.

use std::sync::Arc;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::deepseek_v41_vllm::{
    self as tp_arch, fold_layers, index_plan, layer_config, normalize_step, resolve_layer,
    stage_kv_cache_layout, BoundedReplay, DeepseekV41CudaGraphs, DeepseekV41LayerBody,
    DeepseekV41LayerConfig, DeepseekV41LayerResolved, DeepseekV41ModelCfg, DeepseekV41ReplayConfig,
    DeepseekV41Step, DeepseekV41VllmParallel, LayerKind, NormalizedInput, PlanNode,
};
use crate::arch::deepseek_v41_vllm_dp_attn_ep_moe::ENGRAM_TABLE_ROWS_TP1;
use crate::arch::glm52_vllm_nvfp4_pp_dsa_moe::{eval_compiled, pp_indices};
use crate::arch::kv_layout::KvCacheLayout;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::compressed_sparse_mla_rope_cast::NVFP4_RECORD_BYTES;
use crate::timing::kernels::kv_compress_store::FP8_INDEXER_ROW_BYTES;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, EngramLookupKernel,
    EngramLookupKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::deepseek_v41_common::eval_or_zero;
use crate::worklet::{
    DeepseekV41AttentionEntry, DeepseekV41AttentionTpWorklet, DeepseekV41AttentionTpWorkletInput,
    DeepseekV41AttentionTpWorkletResolved, DeepseekV41EngramPrefetchLocalWorklet,
    DeepseekV41EngramPrefetchLocalWorkletConfig, DeepseekV41EngramPrefetchLocalWorkletResolved,
    DeepseekV41HeadTpWorklet, DeepseekV41HeadTpWorkletConfig, DeepseekV41HeadTpWorkletInput,
    DeepseekV41HeadTpWorkletResolved, DeepseekV41PrologueTpWorklet,
    DeepseekV41PrologueTpWorkletConfig,
};

const ARCH_KIND: &str = "deepseek_v41_vllm_pp";

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

#[derive(Clone, Debug)]
pub struct DeepseekV41VllmPpParallel {
    /// Pipeline stages, one GPU each.
    pub pp_size: u16,
    /// vLLM `VLLM_PP_LAYER_PARTITION`: layers per stage, in stage order.
    /// Empty: vLLM's default `get_pp_indices` split.
    pub layer_partition: Vec<u32>,
    pub gpu_name: String,
    /// Resolved `--max-model-len` ([`tp_arch::resolve_max_model_len`]).
    pub max_model_len: u32,
    /// vLLM's decoder SWA bounded replay ([`BoundedReplay`], on by default).
    pub decoder_swa_bounded_replay: bool,
    /// `--max-num-batched-tokens` (the chunk size).
    pub max_num_batched_tokens: u32,
    /// `--cudagraph-capture-sizes` (PIECEWISE token sizes; empty = eager).
    pub cudagraph_capture_sizes: Vec<u32>,
}

/// The KV source whose compressed cache `layer` reads: the latest source at
/// or below it, for a compressed layer (fork `attention.py`
/// `kv_source_layer_id`).
fn kv_source_of(model: &DeepseekV41ModelCfg, layer: u32) -> Option<u32> {
    if model.compress_ratios[layer as usize] == 0 {
        return None;
    }
    model
        .kv_source_layer_ids
        .iter()
        .copied()
        .filter(|&source| source <= layer)
        .max()
}

/// An index source that scores against another layer's index K cache.
fn is_index_non_owner(model: &DeepseekV41ModelCfg, layer: u32) -> bool {
    model.index_source_layer_ids.contains(&layer) && !model.kv_source_layer_ids.contains(&layer)
}

/// Each stage's `[start, end)` layer range: `layer_partition` when given,
/// else vLLM's `get_pp_indices`.
pub fn stage_ranges(
    num_layers: u32,
    parallel: &DeepseekV41VllmPpParallel,
) -> Result<Vec<(u32, u32)>, BuildError> {
    if parallel.pp_size < 2 || u32::from(parallel.pp_size) > num_layers {
        return Err(fit_failed(format!(
            "pp_size {} must be in 2..={num_layers}; one stage is not a pipeline",
            parallel.pp_size
        )));
    }
    let partition = &parallel.layer_partition;
    if partition.is_empty() {
        return Ok(pp_indices(num_layers, parallel.pp_size));
    }
    if partition.len() != usize::from(parallel.pp_size)
        || partition.contains(&0)
        || partition.iter().sum::<u32>() != num_layers
    {
        return Err(fit_failed(format!(
            "layer_partition {partition:?} must give {} positive layer counts summing to \
             {num_layers}",
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

/// Whether the pinned vLLM fork runs this split: no stage reads a KV source
/// another stage holds, and (with bounded replay) the stage holding the cut
/// layer holds every later layer.
pub fn fork_accepts(model: &DeepseekV41ModelCfg, ranges: &[(u32, u32)], replay: bool) -> bool {
    let groups_intact = ranges.iter().all(|&(start, end)| {
        (start..end).all(|layer| kv_source_of(model, layer).map_or(true, |s| s >= start))
    });
    let replay_ok = !replay
        || BoundedReplay::from_model(model).map_or(true, |rule| {
            ranges
                .iter()
                .find(|&&(start, end)| (start..end).contains(&rule.cut_layer))
                .is_some_and(|&(_, end)| end == model.num_layers)
        });
    groups_intact && replay_ok
}

/// The upstream KV sources stage `[start, end)` reads, with whether it reads
/// each one's index K cache: the copies it keeps.
fn copied_sources(model: &DeepseekV41ModelCfg, (start, end): (u32, u32)) -> Vec<(u32, bool)> {
    let mut copies: Vec<(u32, bool)> = Vec::new();
    for layer in start..end {
        let Some(source) = kv_source_of(model, layer).filter(|&s| s < start) else {
            continue;
        };
        let indexed = is_index_non_owner(model, layer);
        match copies.iter_mut().find(|(s, _)| *s == source) {
            Some(copy) => copy.1 |= indexed,
            None => copies.push((source, indexed)),
        }
    }
    copies
}

/// What crosses the boundary after a stage, beside the hidden state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BoundaryPayload {
    /// The boundary is past the replay cut: on a replay step only the replay
    /// rows cross.
    pub after_cut: bool,
    /// `(ratio, bytes per compressed entry)` per upstream KV source a later
    /// stage reads: its compressed record, plus its index K row when a later
    /// index non-owner scores against it.
    pub kv_entries: Vec<(u32, u64)>,
    /// Per row: the step's top-k indices, when the next stage's first layer
    /// reuses an upstream indexer's (`topk_indices_buffer`, int32).
    pub topk_row_bytes: u64,
    /// Per row: the candidate source's candidate blocks, when a later
    /// candidate consumer sits downstream (`candidate_block_buffer`, int32).
    pub candidate_row_bytes: u64,
}

impl BoundaryPayload {
    fn at(model: &DeepseekV41ModelCfg, boundary: u32, replay: Option<&BoundedReplay>) -> Self {
        let downstream = boundary..model.num_layers;
        let kv_entries = model
            .kv_source_layer_ids
            .iter()
            .copied()
            .filter(|&source| source < boundary)
            .filter_map(|source| {
                let readers: Vec<u32> = downstream
                    .clone()
                    .filter(|&layer| kv_source_of(model, layer) == Some(source))
                    .collect();
                if readers.is_empty() {
                    return None;
                }
                let indexed = readers.iter().any(|&l| is_index_non_owner(model, l));
                let bytes = u64::from(NVFP4_RECORD_BYTES)
                    + if indexed {
                        u64::from(FP8_INDEXER_ROW_BYTES)
                    } else {
                        0
                    };
                Some((model.compress_ratios[source as usize], bytes))
            })
            .collect();
        let first_reuses_topk = boundary < model.num_layers
            && model.compress_ratios[boundary as usize] > 0
            && !model.index_source_layer_ids.contains(&boundary);
        let candidate_downstream = model.candidate_source_layer_id < boundary
            && downstream.clone().any(|layer| {
                is_index_non_owner(model, layer) && layer > model.candidate_source_layer_id
            });
        Self {
            after_cut: replay.is_some_and(|rule| boundary > rule.cut_layer),
            kv_entries,
            topk_row_bytes: if first_reuses_topk {
                u64::from(model.index_topk) * 4
            } else {
                0
            },
            candidate_row_bytes: if candidate_downstream {
                u64::from(model.candidate_topk_blocks) * 4
            } else {
                0
            },
        }
    }
}

/// One stage's recipe.
pub struct DeepseekV41PpStageConfigs {
    pub layers: (u32, u32),
    /// Layer 0's body, on the first stage.
    pub layer0: Option<DeepseekV41LayerConfig>,
    /// Bodies of the stage's other layers, in first-visit order of `plan`.
    pub bodies: Vec<DeepseekV41LayerConfig>,
    plan: Vec<PlanNode>,
    /// The bounded-replay rule, on every stage with a layer at or past the
    /// cut (they normalize the replay batch).
    pub replay_rule: Option<BoundedReplay>,
    /// The cut layer's `write_kv` and gather, on the stage holding it.
    pub cut: Option<DeepseekV41ReplayConfig>,
    /// The scatter back to batch rows, on the last stage.
    pub scatter: Option<ElementwiseKernelConfig>,
    pub late_bodies: Vec<bool>,
    /// What this stage sends to the next one; `None` on the last stage.
    pub send: Option<BoundaryPayload>,
    pub kv_cache_layout: KvCacheLayout,
    /// Upstream KV sources whose caches this stage keeps a copy of.
    pub copied_sources: Vec<(u32, bool)>,
}

pub struct DeepseekV41VllmPpConfigs {
    pub parallel: DeepseekV41VllmPpParallel,
    pub engram_layer_ids: Vec<u32>,
    /// Embedding and Engram hash, at TP1 (on the stages that run them).
    pub prologue: DeepseekV41PrologueTpWorkletConfig,
    /// One host-UVA lookup per Engram layer, at TP1.
    pub engram_prefetch: DeepseekV41EngramPrefetchLocalWorkletConfig,
    pub head: DeepseekV41HeadTpWorkletConfig,
    pub stages: Vec<DeepseekV41PpStageConfigs>,
    /// Whether the pinned vLLM fork runs this split ([`fork_accepts`]).
    pub fork_accepts: bool,
    /// The whole model's cache, one copy over the stages (the stage copies
    /// of upstream KV sources are in the stage layouts only).
    pub kv_cache_layout: KvCacheLayout,
    pub graphs: DeepseekV41CudaGraphs,
}

pub fn build_configs(
    model: &DeepseekV41ModelCfg,
    parallel: &DeepseekV41VllmPpParallel,
    demand: &ExpertDemand,
) -> Result<DeepseekV41VllmPpConfigs, BuildError> {
    if !(1..=model.max_position_embeddings).contains(&parallel.max_model_len) {
        return Err(fit_failed(format!(
            "max_model_len must be 1..={} (max_position_embeddings), got {}",
            model.max_position_embeddings, parallel.max_model_len
        )));
    }
    if demand.num_experts() != model.num_experts as usize {
        return Err(fit_failed(format!(
            "expert demand covers {} experts, expected {}",
            demand.num_experts(),
            model.num_experts
        )));
    }
    let layer0_kind = model.layer_kind(0);
    if layer0_kind.engram || layer0_kind.attention.entry != DeepseekV41AttentionEntry::Layer0Pre {
        return Err(fit_failed("layer 0 must be the plain layer-0 entry"));
    }
    let ranges = stage_ranges(model.num_layers, parallel)?;
    // One GPU per stage: TP1 attention, all experts local.
    let local = DeepseekV41VllmParallel {
        tp_size: 1,
        ep_size: 1,
        gpu_name: parallel.gpu_name.clone(),
        serialize_streams: false,
        decoder_swa_bounded_replay: false,
        max_model_len: parallel.max_model_len,
        max_num_batched_tokens: parallel.max_num_batched_tokens,
        cudagraph_capture_sizes: parallel.cudagraph_capture_sizes.clone(),
    };
    let rule = if parallel.decoder_swa_bounded_replay {
        Some(BoundedReplay::from_model(model)?)
    } else {
        None
    };
    let gpu = parallel.gpu_name.clone();
    let last = ranges.len() - 1;
    let mut stages = Vec::with_capacity(ranges.len());
    for (index, &(start, end)) in ranges.iter().enumerate() {
        let layer0 =
            (start == 0).then(|| layer_config(model, &local, demand, layer0_kind, vec![0]));
        let tail: Vec<(u32, LayerKind)> = (start.max(1)..end)
            .map(|layer| (layer, model.layer_kind(layer)))
            .collect();
        let mut bodies = Vec::new();
        let plan = index_plan(&fold_layers(&tail), &mut |kind, layers| {
            bodies.push(layer_config(model, &local, demand, kind, layers.to_vec()));
            bodies.len() - 1
        });
        let replay_rule = rule.filter(|rule| end > rule.cut_layer);
        let late_bodies = match replay_rule {
            Some(rule) => bodies
                .iter()
                .map(|body: &DeepseekV41LayerConfig| rule.is_late(&body.layers))
                .collect::<Result<Vec<_>, _>>()?,
            None => vec![false; bodies.len()],
        };
        let cut = match replay_rule.filter(|rule| start <= rule.cut_layer) {
            Some(rule) => {
                let cut_body = bodies
                    .iter()
                    .position(|body| body.layers == [rule.cut_layer])
                    .ok_or_else(|| {
                        fit_failed(format!(
                            "decoder SWA bounded replay: cut layer {} shares its body",
                            rule.cut_layer
                        ))
                    })?;
                let write_kv = bodies[cut_body].attention.clone();
                Some(DeepseekV41ReplayConfig::new(
                    rule, cut_body, write_kv, model, &gpu,
                ))
            }
            None => None,
        };
        // The scatter's config is the replay boundary's, built from any cut.
        let scatter = match (index == last, rule) {
            (true, Some(rule)) => {
                let probe = DeepseekV41ReplayConfig::new(
                    rule,
                    0,
                    layer_config(
                        model,
                        &local,
                        demand,
                        model.layer_kind(rule.cut_layer),
                        vec![rule.cut_layer],
                    )
                    .attention,
                    model,
                    &gpu,
                );
                Some(probe.scatter)
            }
            _ => None,
        };
        let copies = copied_sources(model, (start, end));
        stages.push(DeepseekV41PpStageConfigs {
            layers: (start, end),
            layer0,
            bodies,
            plan,
            replay_rule,
            cut,
            scatter,
            late_bodies,
            send: (index != last).then(|| BoundaryPayload::at(model, end, rule.as_ref())),
            kv_cache_layout: stage_kv_cache_layout(model, (start, end), &copies),
            copied_sources: copies,
        });
    }
    let kv_cache_layout =
        tp_arch::kv_cache_layout(model, 1, 0).sharded(u32::from(parallel.pp_size));
    Ok(DeepseekV41VllmPpConfigs {
        parallel: parallel.clone(),
        engram_layer_ids: model.engram_layer_ids.clone(),
        prologue: DeepseekV41PrologueTpWorkletConfig {
            tp_size: 1,
            hidden_size: model.hidden_size.into(),
            engram_num_heads: model.engram_hash_heads,
            metadata_bytes_per_token: None,
            gpu_name: gpu.clone(),
            all_reduce_backends: tp_arch::ALL_REDUCE_BACKENDS.to_vec(),
            elementwise_backends: tp_arch::ELEMENTWISE_BACKENDS.to_vec(),
        },
        engram_prefetch: DeepseekV41EngramPrefetchLocalWorkletConfig {
            tp_size: 1,
            engram_num_heads: model.engram_hash_heads,
            engram_head_dim: model.engram_head_dim,
            quant_block_size: 32,
            table_rows: ENGRAM_TABLE_ROWS_TP1,
            residency: "host_uva".into(),
            weight_dtype: DType::Fp8E4m3,
            gpu_name: gpu.clone(),
            backends: tp_arch::ENGRAM_LOOKUP_BACKENDS.to_vec(),
        },
        head: DeepseekV41HeadTpWorkletConfig {
            tp_size: 1,
            hidden_size: model.hidden_size.into(),
            hc_mult: model.hc_mult,
            vocab_size: model.vocab_size,
            gpu_name: gpu,
            lm_head_backends: tp_arch::LM_HEAD_BACKENDS.to_vec(),
            all_gather_backends: tp_arch::ALL_GATHER_PROXY_BACKENDS.to_vec(),
            elementwise_backends: tp_arch::ELEMENTWISE_BACKENDS.to_vec(),
        },
        fork_accepts: fork_accepts(model, &ranges, rule.is_some()),
        stages,
        kv_cache_layout,
        graphs: DeepseekV41CudaGraphs::new(
            &parallel.cudagraph_capture_sizes,
            tp_arch::MAX_NUM_SEQS,
            &[1],
        ),
    })
}

pub struct DeepseekV41PpStageResolved {
    layers: (u32, u32),
    layer0: Option<DeepseekV41LayerResolved>,
    bodies: Vec<DeepseekV41LayerResolved>,
    plan: Vec<PlanNode>,
    replay_rule: Option<BoundedReplay>,
    cut: Option<CutResolved>,
    scatter: Option<ElementwiseKernelConfig>,
    late_bodies: Vec<bool>,
    send: Option<BoundaryPayload>,
    kv_cache_layout: KvCacheLayout,
}

struct CutResolved {
    cut_layer: u32,
    cut_body: usize,
    write_kv: DeepseekV41AttentionTpWorkletResolved,
    gather: ElementwiseKernelConfig,
}

pub struct DeepseekV41VllmPpResolved {
    parallel: DeepseekV41VllmPpParallel,
    engram_layer_ids: Vec<u32>,
    embedding: ElementwiseKernelConfig,
    engram_hash: ElementwiseKernelConfig,
    engram_prefetch: DeepseekV41EngramPrefetchLocalWorkletResolved,
    head: DeepseekV41HeadTpWorkletResolved,
    stages: Vec<DeepseekV41PpStageResolved>,
    fork_accepts: bool,
    kv_cache_layout: KvCacheLayout,
    graphs: DeepseekV41CudaGraphs,
    hc_mult: u32,
    hidden: u32,
}

pub fn resolve_configs(configs: &DeepseekV41VllmPpConfigs) -> DeepseekV41VllmPpResolved {
    let prologue = DeepseekV41PrologueTpWorklet::resolve_config(&configs.prologue);
    DeepseekV41VllmPpResolved {
        parallel: configs.parallel.clone(),
        engram_layer_ids: configs.engram_layer_ids.clone(),
        embedding: prologue.embedding,
        engram_hash: prologue.engram_hash,
        engram_prefetch: DeepseekV41EngramPrefetchLocalWorklet::resolve_config(
            &configs.engram_prefetch,
        ),
        head: DeepseekV41HeadTpWorklet::resolve_config(&configs.head),
        stages: configs
            .stages
            .iter()
            .map(|stage| DeepseekV41PpStageResolved {
                layers: stage.layers,
                layer0: stage.layer0.as_ref().map(resolve_layer),
                bodies: stage.bodies.iter().map(resolve_layer).collect(),
                plan: stage.plan.clone(),
                replay_rule: stage.replay_rule,
                cut: stage.cut.as_ref().map(|cut| CutResolved {
                    cut_layer: cut.rule.cut_layer,
                    cut_body: cut.cut_body,
                    write_kv: DeepseekV41AttentionTpWorklet::resolve_config(&cut.write_kv),
                    gather: cut.gather.clone(),
                }),
                scatter: stage.scatter.clone(),
                late_bodies: stage.late_bodies.clone(),
                send: stage.send.clone(),
                kv_cache_layout: stage.kv_cache_layout.clone(),
            })
            .collect(),
        fork_accepts: configs.fork_accepts,
        kv_cache_layout: configs.kv_cache_layout.clone(),
        graphs: configs.graphs.clone(),
        hc_mult: configs.head.hc_mult,
        hidden: configs.head.hidden_size.get(),
    }
}

/// The built cut-layer boundary of the stage holding the replay cut.
struct StageCut {
    cut_body: usize,
    write_kv: DeepseekV41AttentionTpWorklet,
    gather: Op<ElementwiseKernel>,
}

fn elementwise(
    leaf: String,
    config: ElementwiseKernelConfig,
    bridge: &PerfApiBridge,
) -> Result<Op<ElementwiseKernel>, BuildError> {
    Ok(Op::new(
        leaf.clone(),
        Arc::new(ElementwiseKernel::build(leaf, config, bridge)?),
    ))
}

/// Shared facts every stage builds from.
struct StageContext<'a> {
    name: &'a str,
    num_stages: u16,
    engram_layer_ids: &'a [u32],
    embedding: &'a ElementwiseKernelConfig,
    engram_hash: &'a ElementwiseKernelConfig,
    engram_prefetch: &'a DeepseekV41EngramPrefetchLocalWorkletResolved,
    head: &'a DeepseekV41HeadTpWorkletResolved,
    graphs: &'a DeepseekV41CudaGraphs,
    max_model_len: u32,
    hidden_row_bytes: u64,
}

/// One pipeline stage: one GPU and its contiguous layer range.
pub struct DeepseekV41VllmPpStageModel {
    name: String,
    stage_index: u16,
    num_stages: u16,
    layer_range: (u32, u32),
    embedding: Option<Op<ElementwiseKernel>>,
    engram_hash: Option<Op<ElementwiseKernel>>,
    /// One lookup per Engram layer this stage holds.
    engram_lookups: Vec<Op<EngramLookupKernel>>,
    layer0: Option<DeepseekV41LayerBody>,
    bodies: Vec<DeepseekV41LayerBody>,
    plan: Vec<PlanNode>,
    replay_rule: Option<BoundedReplay>,
    cut: Option<StageCut>,
    scatter: Option<Op<ElementwiseKernel>>,
    late_bodies: Vec<bool>,
    head: Option<DeepseekV41HeadTpWorklet>,
    send: Option<BoundaryPayload>,
    hidden_row_bytes: u64,
    graphs: DeepseekV41CudaGraphs,
    max_model_len: u32,
    kv_cache_layout: KvCacheLayout,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

impl DeepseekV41VllmPpStageModel {
    fn build(
        ctx: &StageContext,
        stage: DeepseekV41PpStageResolved,
        stage_index: u16,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let name = ctx.name;
        let (start, end) = stage.layers;
        let local_engram: Vec<u32> = ctx
            .engram_layer_ids
            .iter()
            .copied()
            .filter(|layer| (start..end).contains(layer))
            .collect();
        let engram_lookups = local_engram
            .iter()
            .map(|layer| {
                let leaf = format!("{name}.engram_prefetch.layer{layer}.engram_lookup");
                Ok(Op::new(
                    leaf.clone(),
                    Arc::new(EngramLookupKernel::build(
                        leaf,
                        ctx.engram_prefetch.lookup.clone(),
                        bridge,
                    )?),
                ))
            })
            .collect::<Result<Vec<_>, BuildError>>()?;
        let cut = match stage.cut {
            Some(cut) => Some(StageCut {
                cut_body: cut.cut_body,
                write_kv: DeepseekV41AttentionTpWorklet::build(
                    format!("{name}.layer{}.write_kv", cut.cut_layer),
                    cut.write_kv,
                    bridge,
                )?,
                gather: elementwise(format!("{name}.replay.gather"), cut.gather, bridge)?,
            }),
            None => None,
        };
        let is_last = stage_index + 1 == ctx.num_stages;
        let mut model = Self {
            name: name.to_string(),
            stage_index,
            num_stages: ctx.num_stages,
            layer_range: stage.layers,
            embedding: match stage_index {
                0 => Some(elementwise(
                    format!("{name}.prologue.embedding"),
                    ctx.embedding.clone(),
                    bridge,
                )?),
                _ => None,
            },
            engram_hash: match local_engram.is_empty() {
                false => Some(elementwise(
                    format!("{name}.prologue.engram_hash"),
                    ctx.engram_hash.clone(),
                    bridge,
                )?),
                true => None,
            },
            engram_lookups,
            layer0: match stage.layer0 {
                Some(layer0) => Some(DeepseekV41LayerBody::build(name, layer0, bridge)?),
                None => None,
            },
            bodies: stage
                .bodies
                .into_iter()
                .map(|body| DeepseekV41LayerBody::build(name, body, bridge))
                .collect::<Result<Vec<_>, _>>()?,
            plan: stage.plan,
            replay_rule: stage.replay_rule,
            cut,
            scatter: match stage.scatter {
                Some(config) => Some(elementwise(
                    format!("{name}.replay.scatter"),
                    config,
                    bridge,
                )?),
                None => None,
            },
            late_bodies: stage.late_bodies,
            head: match is_last {
                true => Some(DeepseekV41HeadTpWorklet::build(
                    format!("{name}.head"),
                    ctx.head.clone(),
                    bridge,
                )?),
                false => None,
            },
            send: stage.send,
            hidden_row_bytes: ctx.hidden_row_bytes,
            graphs: ctx.graphs.clone(),
            max_model_len: ctx.max_model_len,
            kv_cache_layout: stage.kv_cache_layout,
            cost_flat: Vec::new(),
            n_slots: 0,
        };
        let tree = model.cost_tree();
        model.cost_flat = tree.flatten();
        model.n_slots = tree.n_slots();
        Ok(model)
    }

    pub fn stage_index(&self) -> u16 {
        self.stage_index
    }

    /// The `[start, end)` decoder layers this stage runs.
    pub fn layer_range(&self) -> (u32, u32) {
        self.layer_range
    }

    /// What this stage sends beside the hidden state; `None` on the last stage.
    pub fn send_payload(&self) -> Option<&BoundaryPayload> {
        self.send.as_ref()
    }

    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let root = self.compile(&mut builder);
        builder.finish(root)
    }

    /// The stage's subtree, in evaluation order: embedding, Engram hash,
    /// layer 0, Engram lookups, layers (the cut layer's `write_kv` and the
    /// gather before its body), the scatter, the head.
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let mut children = Vec::new();
        if let Some(embedding) = &self.embedding {
            children.push(embedding.compile(builder));
        }
        if let Some(hash) = &self.engram_hash {
            children.push(hash.compile(builder));
        }
        if let Some(layer0) = &self.layer0 {
            children.push(layer0.compile(builder));
        }
        if !self.engram_lookups.is_empty() {
            // Host-UVA row gathers contending with the main path (the TP
            // arch's `compile_joined`): serial.
            children.push(CostNode::Labeled {
                label: format!("{}.engram_prefetch [host_uva lookups]", self.name),
                child: Box::new(CostNode::Sum(
                    self.engram_lookups
                        .iter()
                        .map(|op| op.compile(builder))
                        .collect(),
                )),
            });
        }
        children.extend(self.compile_plan(&self.plan, builder));
        if let Some(scatter) = &self.scatter {
            children.push(scatter.compile(builder));
        }
        if let Some(head) = &self.head {
            children.push(head.compile(builder));
        }
        CostNode::Labeled {
            label: format!(
                "{} pipeline stage {} of {} (DeepseekV41VllmPpStageModel) [layers {}..{}; TP1/EP1 \
                 on one GPU, no collectives; replay_cut_here={}; sends={:?}]",
                self.name,
                self.stage_index,
                self.num_stages,
                self.layer_range.0,
                self.layer_range.1,
                self.cut.is_some(),
                self.send,
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    fn compile_plan(&self, plan: &[PlanNode], builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut out = Vec::with_capacity(plan.len());
        for node in plan {
            match node {
                PlanNode::Body(index) => {
                    if let Some(cut) = self.cut.as_ref().filter(|c| c.cut_body == *index) {
                        out.push(cut.write_kv.compile(builder));
                        out.push(cut.gather.compile(builder));
                    }
                    out.push(self.bodies[*index].compile(builder));
                }
                PlanNode::Repeat { n, body } => {
                    let children = self.compile_plan(body, builder);
                    out.push(CostNode::Scale {
                        n: *n,
                        child: Box::new(if children.len() == 1 {
                            children.into_iter().next().unwrap()
                        } else {
                            CostNode::Sum(children)
                        }),
                    });
                }
            }
        }
        out
    }

    fn normalize(
        &self,
        batch: &UnifiedArchInput,
    ) -> Result<(DeepseekV41Step, NormalizedInput), String> {
        let step = DeepseekV41Step::from_unified(batch, self.max_model_len)?;
        let logits_rows = step.request_count();
        let input = normalize_step(
            &self.graphs,
            &step,
            1,
            logits_rows,
            self.replay_rule.as_ref(),
        );
        Ok((step, input))
    }

    fn eval_normalized(&self, input: &NormalizedInput, ev: &mut Evaluator) {
        let rows = input.full.rows;
        let zero = rows == 0;
        if let Some(embedding) = &self.embedding {
            eval_or_zero(
                embedding,
                ElementwiseKernelInput { num_tokens: rows },
                zero,
                ev,
            );
        }
        if let Some(hash) = &self.engram_hash {
            eval_or_zero(hash, ElementwiseKernelInput { num_tokens: rows }, zero, ev);
        }
        if let Some(layer0) = &self.layer0 {
            layer0.eval(&input.full, ev);
        }
        for lookup in &self.engram_lookups {
            eval_or_zero(
                lookup,
                EngramLookupKernelInput { num_tokens: rows },
                zero,
                ev,
            );
        }
        self.eval_plan(&self.plan, input, ev);
        if let Some(scatter) = &self.scatter {
            let scattered = if input.replay.is_some() { rows } else { 0 };
            eval_or_zero(
                scatter,
                ElementwiseKernelInput {
                    num_tokens: scattered,
                },
                scattered == 0,
                ev,
            );
        }
        if let Some(head) = &self.head {
            head.eval(
                &DeepseekV41HeadTpWorkletInput {
                    num_tokens: rows,
                    mhc_post_rows: input.replay.as_ref().map_or(rows, |late| late.rows),
                    logits_rows: input.logits_rows,
                },
                ev,
            );
        }
    }

    fn eval_plan(&self, plan: &[PlanNode], input: &NormalizedInput, ev: &mut Evaluator) {
        for node in plan {
            match node {
                PlanNode::Body(index) => {
                    if let Some(cut) = self.cut.as_ref().filter(|c| c.cut_body == *index) {
                        // `write_kv` and the gather run only on replay steps.
                        let write_kv = if input.replay.is_some() {
                            DeepseekV41AttentionTpWorkletInput {
                                write_kv_only: true,
                                ..input.full.attention.clone()
                            }
                        } else {
                            DeepseekV41AttentionTpWorkletInput {
                                num_tokens: 0,
                                prefill_query_context_pairs: Vec::new(),
                                decode_kv_lens: Vec::new(),
                                aux_stream_live: false,
                                full_graph: false,
                                mhc_overlap: false,
                                write_kv_only: true,
                            }
                        };
                        cut.write_kv.eval(&write_kv, ev);
                        let gathered = input.replay.as_ref().map_or(0, |late| late.rows);
                        eval_or_zero(
                            &cut.gather,
                            ElementwiseKernelInput {
                                num_tokens: gathered,
                            },
                            gathered == 0,
                            ev,
                        );
                    }
                    let body_input = match &input.replay {
                        Some(late) if self.late_bodies[*index] => late,
                        _ => &input.full,
                    };
                    self.bodies[*index].eval(body_input, ev)
                }
                PlanNode::Repeat { body, .. } => self.eval_plan(body, input, ev),
            }
        }
    }

    fn eval_batch(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        let (_, input) = self
            .normalize(batch)
            .unwrap_or_else(|reason| panic!("invalid DeepSeek-V4.1 PP stage input: {reason}"));
        self.eval_normalized(&input, ev);
    }

    /// Bytes this stage sends to the next for `batch`: the hidden state of
    /// every row the next stage runs (the replay rows past the cut on a
    /// replay step; graph padding included, as `IntermediateTensors` are
    /// sized to the padded batch), the top-k and candidate rows it reuses,
    /// and every forwarded KV source's new compressed entries. `None` on the
    /// last stage.
    pub fn send_bytes(&self, batch: &UnifiedArchInput) -> Option<u64> {
        let payload = self.send.as_ref()?;
        let (step, input) = self
            .normalize(batch)
            .unwrap_or_else(|reason| panic!("invalid DeepSeek-V4.1 PP stage input: {reason}"));
        let rows = match &input.replay {
            Some(late) if payload.after_cut => late.rows,
            _ => input.full.rows,
        };
        let row_bytes =
            self.hidden_row_bytes + payload.topk_row_bytes + payload.candidate_row_bytes;
        // Each request's new positions `[end - q, end)`.
        let spans: Vec<(u32, u32)> = step
            .prefill
            .iter()
            .chain(&step.decode)
            .map(|&(q, end)| (end - q, end))
            .collect();
        let kv: u64 = payload
            .kv_entries
            .iter()
            .map(|&(ratio, bytes)| {
                let entries: u64 = spans
                    .iter()
                    .map(|&(from, to)| u64::from(to / ratio - from / ratio))
                    .sum();
                entries * bytes
            })
            .sum();
        Some(u64::from(rows) * row_bytes + kv)
    }
}

impl IterwiseUnifiedModel for DeepseekV41VllmPpStageModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        self.normalize(batch).map(|_| ())
    }

    /// This stage's layers (and its copies of upstream KV sources).
    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout.clone()
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

    fn pipeline_send_bytes(&self, batch: &UnifiedArchInput) -> Option<u64> {
        self.send_bytes(batch)
    }

    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        eval_compiled(&self.cost_flat, self.n_slots, slots, scratch, None, |ev| {
            self.eval_batch(batch, ev)
        })
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        eval_compiled(
            &self.cost_flat,
            self.n_slots,
            slots,
            scratch,
            Some(inputs),
            |ev| self.eval_batch(batch, ev),
        )
    }
}

/// Build every stage, then the whole-pipeline view over them.
pub fn build(
    name: String,
    resolved: DeepseekV41VllmPpResolved,
    bridge: &PerfApiBridge,
) -> Result<DeepseekV41VllmPpModel, BuildError> {
    let DeepseekV41VllmPpResolved {
        parallel,
        engram_layer_ids,
        embedding,
        engram_hash,
        engram_prefetch,
        head,
        stages,
        fork_accepts,
        kv_cache_layout,
        graphs,
        hc_mult,
        hidden,
    } = resolved;
    let hc = u64::from(hc_mult);
    let ctx = StageContext {
        name: &name,
        num_stages: stages.len() as u16,
        engram_layer_ids: &engram_layer_ids,
        embedding: &embedding,
        engram_hash: &engram_hash,
        engram_prefetch: &engram_prefetch,
        head: &head,
        graphs: &graphs,
        max_model_len: parallel.max_model_len,
        // The fork's PP `IntermediateTensors`
        // (`make_empty_intermediate_tensors`): the `(hc_mult, hidden)` bf16
        // stream and the `(hc_mult,)` fp32 pre-mix.
        hidden_row_bytes: hc * u64::from(hidden) * 2 + hc * 4,
    };
    let stages = stages
        .into_iter()
        .enumerate()
        .map(|(index, stage)| {
            DeepseekV41VllmPpStageModel::build(&ctx, stage, index as u16, bridge).map(Arc::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let hidden_row_bytes = ctx.hidden_row_bytes;
    let mut model = DeepseekV41VllmPpModel {
        name,
        stages,
        max_model_len: parallel.max_model_len,
        hidden_row_bytes,
        fork_accepts,
        kv_cache_layout,
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

/// The whole pipeline as one iteration: every stage in order, summed. One
/// microbatch's compute latency, without stage-to-stage transfers and without
/// the overlap of microbatches in flight (those belong to the pipeline
/// workers).
pub struct DeepseekV41VllmPpModel {
    name: String,
    stages: Vec<Arc<DeepseekV41VllmPpStageModel>>,
    max_model_len: u32,
    hidden_row_bytes: u64,
    fork_accepts: bool,
    kv_cache_layout: KvCacheLayout,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

impl DeepseekV41VllmPpModel {
    pub fn stages(&self) -> &[Arc<DeepseekV41VllmPpStageModel>] {
        &self.stages
    }

    pub fn pp_size(&self) -> u16 {
        self.stages.len() as u16
    }

    /// Whether the pinned vLLM fork runs this split ([`fork_accepts`]).
    pub fn fork_accepts(&self) -> bool {
        self.fork_accepts
    }

    /// The cache of the stage with the most bytes per token: the pipeline's
    /// one block pool holds the same tokens on every stage.
    pub fn pipeline_kv_cache_layout(&self) -> KvCacheLayout {
        self.stages
            .iter()
            .map(|stage| &stage.kv_cache_layout)
            .max_by_key(|layout| layout.per_token_bytes())
            .expect("a pipeline has stages")
            .clone()
    }

    /// Hidden-state bytes per row at a boundary; the stages' `send_bytes`
    /// add what else crosses. The head uses it only for a stage that sends
    /// no explicit size.
    pub fn activation_bytes_per_token(&self) -> u64 {
        self.hidden_row_bytes
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
                "{} (DeepseekV41VllmPpModel) [PP{}; one TP1/EP1 GPU per stage; stages in \
                 sequence, no transfer; fork_accepts={}; max_model_len={}]",
                self.name,
                self.stages.len(),
                self.fork_accepts,
                self.max_model_len
            ),
            child: Box::new(CostNode::Sum(stages)),
        };
        builder.finish(root)
    }

    fn eval_batch(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        for stage in &self.stages {
            stage.eval_batch(batch, ev);
        }
    }
}

impl IterwiseUnifiedModel for DeepseekV41VllmPpModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        self.stages[0].check_input(batch)
    }

    /// Every stage's cache together, one copy over the stages.
    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout.clone()
    }

    fn gpus_per_replica(&self) -> u16 {
        self.pp_size()
    }

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
        eval_compiled(&self.cost_flat, self.n_slots, slots, scratch, None, |ev| {
            self.eval_batch(batch, ev)
        })
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        eval_compiled(
            &self.cost_flat,
            self.n_slots,
            slots,
            scratch,
            Some(inputs),
            |ev| self.eval_batch(batch, ev),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::config::ModelSpec;
    use crate::arch::contract::ArchGroupInput;
    use crate::timing::routing::RoutingDistribution;
    use std::path::Path;

    const CONFIG: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/model/config/deepseek_v41_flash.json"
    );

    fn model_cfg() -> DeepseekV41ModelCfg {
        let spec: ModelSpec =
            serde_json::from_str(&format!(r#"{{"model_config":"{CONFIG}","fp8":true}}"#)).unwrap();
        DeepseekV41ModelCfg::from_json(Path::new(CONFIG), &spec).unwrap()
    }

    fn parallel(partition: &[u32]) -> DeepseekV41VllmPpParallel {
        DeepseekV41VllmPpParallel {
            pp_size: partition.len() as u16,
            layer_partition: partition.to_vec(),
            gpu_name: "NVIDIA B200".into(),
            max_model_len: 1_048_576,
            decoder_swa_bounded_replay: true,
            max_num_batched_tokens: 8192,
            cudagraph_capture_sizes: Vec::new(),
        }
    }

    fn uniform() -> ExpertDemand {
        ExpertDemand::popularity(&RoutingDistribution::uniform(384), 1)
    }

    fn configs(partition: &[u32]) -> DeepseekV41VllmPpConfigs {
        build_configs(&model_cfg(), &parallel(partition), &uniform()).unwrap()
    }

    fn built(partition: &[u32]) -> DeepseekV41VllmPpModel {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        build("pp".into(), resolve_configs(&configs(partition)), &bridge).unwrap()
    }

    fn prefill(append: u32, context: u32) -> UnifiedArchInput {
        let mut group = ArchGroupInput::default();
        group.prefill_chunk_pairs.push((context - append, append));
        group.prefill_tokens = append;
        group.batch_tokens = append;
        UnifiedArchInput {
            groups: vec![group],
            tokens_per_source_rank: Vec::new(),
        }
    }

    #[test]
    fn fork_accepts_only_cuts_between_kv_sharing_groups() {
        let model = model_cfg();
        let ranges = |p: &[u32]| stage_ranges(40, &parallel(p)).unwrap();
        assert!(fork_accepts(&model, &ranges(&[8, 6, 6, 20]), true));
        assert!(fork_accepts(&model, &ranges(&[1, 1, 6, 6, 6, 20]), true));
        assert!(fork_accepts(&model, &ranges(&[8, 12, 20]), true));
        // Starts at 5, 12, 18: inside [2, 7], [8, 13] and [14, 19].
        assert!(!fork_accepts(&model, &ranges(&[5, 7, 6, 22]), true));
        // Layer 24 reads layer 20's cache from another stage.
        assert!(!fork_accepts(&model, &ranges(&[8, 6, 6, 4, 16]), true));
        // Any partition with positive counts builds.
        assert!(build_configs(&model, &parallel(&[5, 7, 6, 22]), &uniform()).is_ok());
        assert!(build_configs(&model, &parallel(&[1, 2, 4, 3, 4, 3, 7, 16]), &uniform()).is_ok());
        assert!(stage_ranges(40, &parallel(&[0, 40])).is_err());
    }

    #[test]
    fn a_stage_reading_an_upstream_source_keeps_a_copy() {
        let cfgs = configs(&[5, 7, 6, 22]);
        let copies: Vec<&[(u32, bool)]> = cfgs
            .stages
            .iter()
            .map(|s| s.copied_sources.as_slice())
            .collect();
        // 5..12 reads 2 (5-7); 12..18 reads 8 (12-13); 18..40 reads 14 (18-19).
        assert_eq!(
            copies,
            [&[][..], &[(2, false)], &[(8, false)], &[(14, false)]]
        );
        let cfgs = configs(&[8, 6, 6, 4, 16]);
        // 24..40 reads layer 20's compressed KV and, for the index
        // non-owners 24/28/32/36, its index K.
        assert_eq!(cfgs.stages[4].copied_sources, [(20, true)]);
        let own = stage_kv_cache_layout(&model_cfg(), (24, 40), &[]);
        assert!(cfgs.stages[4].kv_cache_layout.per_token_bytes() > own.per_token_bytes());
    }

    #[test]
    fn boundaries_carry_what_the_next_stage_reads() {
        let model = model_cfg();
        let rule = BoundedReplay::from_model(&model).unwrap();
        // Between groups: hidden only.
        assert_eq!(
            BoundaryPayload::at(&model, 8, Some(&rule)),
            BoundaryPayload::default()
        );
        // Inside [2, 7]: layer 2's compressed record and its top-k.
        let p = BoundaryPayload::at(&model, 5, Some(&rule));
        assert_eq!(p.kv_entries, [(2, u64::from(NVFP4_RECORD_BYTES))]);
        assert_eq!(p.topk_row_bytes, 512 * 4);
        assert!(!p.after_cut && p.candidate_row_bytes == 0);
        // At 24: past the cut, layer 20's record + index K, its candidates;
        // 24 is an index source, so no top-k.
        let p = BoundaryPayload::at(&model, 24, Some(&rule));
        assert!(p.after_cut);
        assert_eq!(
            p.kv_entries,
            [(1, u64::from(NVFP4_RECORD_BYTES + FP8_INDEXER_ROW_BYTES))]
        );
        assert_eq!((p.topk_row_bytes, p.candidate_row_bytes), (0, 2048 * 4));
    }

    #[test]
    fn replay_spans_the_stages_after_the_cut() {
        let model = built(&[1, 2, 4, 3, 4, 3, 7, 16]);
        let stages = model.stages();
        // 17..24 holds the cut; 24..40 runs only late layers.
        assert!(stages[6].cut.is_some() && stages[6].replay_rule.is_some());
        assert!(stages[7].cut.is_none() && stages[7].late_bodies.iter().all(|&l| l));
        assert!(stages[7].scatter.is_some());
        assert!(stages[..6].iter().all(|s| s.replay_rule.is_none()));
        // A 4096-token prefill chunk: the 24 boundary carries 128 replay rows,
        // the 17 boundary all 4096 with their top-k (17 reuses layer 14's
        // indexer) and layer 14's 2048 new ratio-2 records.
        let batch = prefill(4096, 4096);
        let at24 = stages[6].send_bytes(&batch).unwrap();
        let at17 = stages[5].send_bytes(&batch).unwrap();
        let hidden = model.activation_bytes_per_token();
        assert_eq!(
            at17,
            4096 * (hidden + 512 * 4) + 2048 * u64::from(NVFP4_RECORD_BYTES)
        );
        assert_eq!(
            at24,
            128 * (hidden + 2048 * 4)
                + 4096 * u64::from(NVFP4_RECORD_BYTES + FP8_INDEXER_ROW_BYTES)
        );
        assert!(stages[7].send_bytes(&batch).is_none());
    }

    #[test]
    fn sections_live_on_the_stages_that_run_them() {
        let model = built(&[1, 1, 6, 6, 6, 20]);
        let stages = model.stages();
        assert!(model.fork_accepts());
        assert!(stages[0].embedding.is_some() && stages[0].layer0.is_some());
        assert!(stages[1..]
            .iter()
            .all(|s| s.embedding.is_none() && s.layer0.is_none()));
        let lookups: Vec<usize> = stages.iter().map(|s| s.engram_lookups.len()).collect();
        assert_eq!(lookups, [0, 1, 0, 0, 1, 0]);
        let heads: Vec<bool> = stages.iter().map(|s| s.head.is_some()).collect();
        assert_eq!(heads, [false, false, false, false, false, true]);
        assert_eq!(model.activation_bytes_per_token(), 4 * 5120 * 2 + 4 * 4);
        // Fork-legal: replay only on the last stage.
        assert!(stages[5].cut.is_some() && stages[..5].iter().all(|s| s.replay_rule.is_none()));
    }

    #[test]
    fn a_batch_evaluates_through_every_stage() {
        let model = built(&[5, 7, 6, 22]);
        let batch = prefill(4096, 10_000);
        assert!(model.check_input(&batch).is_ok());
        for stage in model.stages() {
            assert!(stage.check_input(&batch).is_ok());
        }
    }
    /// Per-stage cache and boundary payload of the splits in `DSV41_PP_SPLITS`
    /// (`;`-separated partitions such as `8,6,6,20;5,7,6,22`), one JSON line per
    /// stage. Feeds `logs/*_dsv41_pp_*/stage_memory.py`.
    #[test]
    #[ignore = "report: cargo test stage_layout_report -- --ignored --nocapture"]
    fn stage_layout_report() {
        let model = model_cfg();
        let splits = std::env::var("DSV41_PP_SPLITS").unwrap_or_else(|_| "8,6,6,20".into());
        for split in splits.split(';') {
            let partition: Vec<u32> = split
                .split(',')
                .map(|v| v.trim().parse().unwrap())
                .collect();
            let cfgs = configs(&partition);
            for (index, stage) in cfgs.stages.iter().enumerate() {
                let own = crate::arch::deepseek_v41_vllm::stage_kv_cache_layout(
                    &model,
                    stage.layers,
                    &[],
                );
                let send = stage.send.as_ref().map(|send| {
                    serde_json::json!({
                        "after_cut": send.after_cut,
                        "kv_entries": send.kv_entries,
                        "topk_row_bytes": send.topk_row_bytes,
                        "candidate_row_bytes": send.candidate_row_bytes,
                    })
                });
                let line = serde_json::json!({
                    "split": split,
                    "stage": index,
                    "layers": [stage.layers.0, stage.layers.1],
                    "copied_sources": stage.copied_sources,
                    "per_token_bytes": stage.kv_cache_layout.per_token_bytes(),
                    "own_per_token_bytes": own.per_token_bytes(),
                    "fixed_device_bytes": stage.kv_cache_layout.engine.fixed_device_bytes,
                    "layout": stage.kv_cache_layout,
                    "send": send,
                    "fork_accepts": fork_accepts(
                        &model,
                        &stage_ranges(40, &parallel(&partition)).unwrap(),
                        true,
                    ),
                });
                println!("STAGE_LAYOUT {line}");
            }
        }
    }
}
