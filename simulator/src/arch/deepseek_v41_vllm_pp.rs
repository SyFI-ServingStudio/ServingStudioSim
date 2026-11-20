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
use crate::arch::deepseek_v41_engram_fetch::{self, EngramHostGather};
use crate::arch::deepseek_v41_vllm::{
    self as tp_arch, fold_layers, index_plan, layer_config, normalize_step, resolve_layer,
    stage_kv_cache_layout, BoundedReplay, DeepseekV41CudaGraphs, DeepseekV41LayerBody,
    DeepseekV41LayerConfig, DeepseekV41LayerResolved, DeepseekV41ModelCfg, DeepseekV41ReplayConfig,
    DeepseekV41Step, DeepseekV41VllmParallel, LayerKind, NormalizedInput, PlanNode,
};
use crate::arch::deepseek_v41_vllm_dp_attn_ep_moe::{
    self as dp_arch, DeepInferEngine, ENGRAM_TABLE_ROWS_TP1,
};
use crate::arch::glm52_vllm_nvfp4_pp_dsa_moe::{eval_compiled, pp_indices};
use crate::arch::kv_layout::KvCacheLayout;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::compressed_sparse_mla_rope_cast::{
    MXFP8_RECORD_BYTES, NVFP4_RECORD_BYTES,
};
use crate::timing::kernels::kv_compress_store::FP8_INDEXER_ROW_BYTES;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, EngramLookupKernel,
    EngramLookupKernelInput,
};
use crate::timing::Metrics4;
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

use crate::op::attention::DeepseekV41CandidateRole;

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
    /// Per stage, the decoder layer ranges it holds, overriding
    /// `layer_partition`. A stage may hold several ranges (a folded split:
    /// GPU0 `[0,10) + [20,30)`, GPU1 `[10,20) + [30,40)`); the layers before
    /// the replay cut and those after it must each run in stage order. A split
    /// whose back layers sit on a stage before the cut's front reaches it runs
    /// only with a deferred back pass ([`UnifiedArchInput::back_pass`]).
    pub stage_layers: Vec<Vec<(u32, u32)>>,
    /// How the cut layer's input reaches it ([`CutHandoff`]).
    pub cut_handoff: CutHandoff,
    /// `Some`: the DeepInfer engine's kernels on every stage (DeepGEMM mega
    /// MoE over the GPU's own 384 experts, DeepGEMM MXFP8 linears, YoCo keep
    /// rows), as for the DP arch ([`DeepInferEngine`]).
    pub deepinfer: Option<DeepInferEngine>,
}

/// Where the replay cut layer's `write_kv` runs and what carries the rows
/// that continue past the cut to the stage holding it.
/// - `V1`: every row's hidden state (4 hc streams + pre-mix) crosses;
///   `write_kv` runs on the cut layer's stage.
/// - `V2`: the continuing rows' full state crosses, every other row only its
///   collapsed hidden (bf16); `write_kv` on the cut layer's stage.
/// - `V3`: only the continuing rows' state crosses; `write_kv` runs on the
///   stage holding the layer before the cut and the cut source's new KV
///   entries (and the cut layer's SWA records the continuing rows read)
///   cross instead.
///
/// On a folded split `V1`/`V2` would run `write_kv` on the next microbatch
/// through the cut stage; it is costed on the same microbatch's front rows
/// (same work, shifted by one microbatch).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CutHandoff {
    #[default]
    V1,
    V2,
    V3,
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

/// `stage_layers` as a preset spells it: `start-end` ranges (end exclusive)
/// joined by `,`, stages joined by `;`. Empty: none.
pub fn parse_stage_layers(text: &str) -> anyhow::Result<Vec<Vec<(u32, u32)>>> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    text.split(';')
        .map(|stage| {
            stage
                .split(',')
                .map(|range| {
                    let (start, end) = range.trim().split_once('-').ok_or_else(|| {
                        anyhow::anyhow!("stage_layers range {range:?} is not start-end")
                    })?;
                    Ok((start.trim().parse()?, end.trim().parse()?))
                })
                .collect()
        })
        .collect()
}

/// Each stage's layer ranges: `stage_layers` when given (validated), else
/// one contiguous range from [`stage_ranges`].
pub fn stage_layer_lists(
    num_layers: u32,
    parallel: &DeepseekV41VllmPpParallel,
) -> Result<Vec<Vec<(u32, u32)>>, BuildError> {
    if parallel.stage_layers.is_empty() {
        return Ok(stage_ranges(num_layers, parallel)?
            .into_iter()
            .map(|range| vec![range])
            .collect());
    }
    let lists = &parallel.stage_layers;
    if lists.len() != usize::from(parallel.pp_size) || parallel.pp_size < 2 {
        return Err(fit_failed(format!(
            "stage_layers {lists:?} must list pp_size {} >= 2 stages",
            parallel.pp_size
        )));
    }
    let mut seen = vec![false; num_layers as usize];
    for list in lists {
        if list.is_empty() || list.windows(2).any(|w| w[0].1 > w[1].0) {
            return Err(fit_failed(format!(
                "stage_layers {list:?}: a stage needs ascending, disjoint ranges"
            )));
        }
        for &(start, end) in list {
            if start >= end || end > num_layers {
                return Err(fit_failed(format!("stage_layers range {start}..{end}")));
            }
            for layer in start..end {
                if std::mem::replace(&mut seen[layer as usize], true) {
                    return Err(fit_failed(format!("layer {layer} is on two stages")));
                }
            }
        }
    }
    if seen.iter().any(|&held| !held) {
        return Err(fit_failed(format!(
            "stage_layers {lists:?} must cover layers 0..{num_layers}"
        )));
    }
    Ok(lists.clone())
}

/// The stage of every layer.
fn layer_places(num_layers: u32, lists: &[Vec<(u32, u32)>]) -> Vec<u16> {
    let mut place = vec![0_u16; num_layers as usize];
    for (stage, list) in lists.iter().enumerate() {
        for &(start, end) in list {
            for layer in start..end {
                place[layer as usize] = stage as u16;
            }
        }
    }
    place
}

/// A stage's ranges split at `cut` (front first, then late, each ascending).
fn stage_segments(list: &[(u32, u32)], cut: Option<u32>) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for &(start, end) in list {
        match cut {
            Some(cut) if start < cut && cut < end => {
                out.push((start, cut));
                out.push((cut, end));
            }
            _ => out.push((start, end)),
        }
    }
    out
}

/// Whether the pinned vLLM fork runs this split: no stage reads a KV source
/// another stage holds, and (with bounded replay) the stage holding the cut
/// layer holds every later layer.
pub fn fork_accepts(model: &DeepseekV41ModelCfg, ranges: &[(u32, u32)], replay: bool) -> bool {
    let lists: Vec<Vec<(u32, u32)>> = ranges.iter().map(|&range| vec![range]).collect();
    fork_accepts_lists(model, &lists, replay)
}

/// [`fork_accepts`] over per-stage layer ranges.
pub fn fork_accepts_lists(
    model: &DeepseekV41ModelCfg,
    lists: &[Vec<(u32, u32)>],
    replay: bool,
) -> bool {
    let place = layer_places(model.num_layers, lists);
    let groups_intact = (0..model.num_layers).all(|layer| {
        kv_source_of(model, layer).map_or(true, |s| place[s as usize] == place[layer as usize])
    });
    let replay_ok = !replay
        || BoundedReplay::from_model(model).map_or(true, |rule| {
            (rule.cut_layer..model.num_layers)
                .all(|layer| place[layer as usize] == place[rule.cut_layer as usize])
        });
    groups_intact && replay_ok
}

/// The KV sources a stage holding `layers` reads but does not hold, with
/// whether it reads each one's index K cache: the copies it keeps.
fn copied_sources(model: &DeepseekV41ModelCfg, layers: &[u32]) -> Vec<(u32, bool)> {
    let mut copies: Vec<(u32, bool)> = Vec::new();
    for &layer in layers {
        let Some(source) = kv_source_of(model, layer).filter(|s| !layers.contains(s)) else {
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

/// What one hop between stages carries, beside each row's hidden state. A
/// microbatch makes the forward hops stage by stage; on a folded split the
/// last stage also sends a return hop to the first, which the next
/// microbatch's back pass reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HopPayload {
    /// The front batch (layers before the cut) crosses: per row, the hidden
    /// state plus these bytes (the step's top-k indices when the next front
    /// layer reuses an upstream indexer's).
    pub front_extra_row_bytes: Option<u64>,
    /// The late batch (layers from the cut on) crosses: per row, the hidden
    /// state plus these bytes (top-k as above, and the candidate source's
    /// candidate blocks when a later candidate consumer is past the hop).
    pub late_extra_row_bytes: Option<u64>,
    /// The cut layer's input crosses, as this variant carries it.
    pub handoff: Option<CutHandoff>,
    /// `(ratio, bytes per compressed entry)` per KV source written before the
    /// hop and read past it, over the front batch's new tokens: its
    /// compressed record, plus its index K row when an index non-owner past
    /// the hop scores against it.
    pub kv_entries: Vec<(u32, u64)>,
}

/// The facts a hop is derived from.
struct HopGeometry<'a> {
    model: &'a DeepseekV41ModelCfg,
    place: Vec<u16>,
    cut: Option<u32>,
    handoff: CutHandoff,
}

impl HopGeometry<'_> {
    fn is_front(&self, layer: u32) -> bool {
        self.cut.map_or(true, |cut| layer < cut)
    }

    /// The stage that writes a KV source's entries.
    fn writer(&self, source: u32) -> u16 {
        match self.cut {
            Some(cut) if source == cut && self.handoff == CutHandoff::V3 => {
                self.place[cut as usize - 1]
            }
            _ => self.place[source as usize],
        }
    }

    fn extra_row_bytes(&self, first_down: u32, up: &dyn Fn(u32) -> bool, down: &[u32]) -> u64 {
        let model = self.model;
        let topk = if model.compress_ratios[first_down as usize] > 0
            && !model.index_source_layer_ids.contains(&first_down)
        {
            u64::from(model.index_topk) * 4
        } else {
            0
        };
        let cand = model.candidate_source_layer_id;
        let candidate = if up(cand)
            && down
                .iter()
                .any(|&l| is_index_non_owner(model, l) && l > cand)
        {
            u64::from(model.candidate_topk_blocks) * 4
        } else {
            0
        };
        topk + candidate
    }

    fn entries(&self, readers_of: &dyn Fn(u32, u16) -> Vec<u32>) -> Vec<(u32, u64)> {
        let model = self.model;
        model
            .kv_source_layer_ids
            .iter()
            .copied()
            .filter_map(|source| {
                let readers = readers_of(source, self.writer(source));
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
            .collect()
    }

    fn readers(&self, source: u32) -> impl Iterator<Item = u32> + '_ {
        (0..self.model.num_layers).filter(move |&l| kv_source_of(self.model, l) == Some(source))
    }

    /// The hop from stage `stage` to `stage + 1`.
    fn forward(&self, stage: u16) -> HopPayload {
        let n = self.model.num_layers;
        let up = |l: u32| self.place[l as usize] <= stage;
        let down: Vec<u32> = (0..n).filter(|&l| !up(l)).collect();
        let crosses = |front: bool| -> Option<u64> {
            let has_up = (0..n).any(|l| self.is_front(l) == front && up(l));
            let downs: Vec<u32> = down
                .iter()
                .copied()
                .filter(|&l| self.is_front(l) == front)
                .collect();
            (has_up && !downs.is_empty()).then(|| self.extra_row_bytes(downs[0], &up, &downs))
        };
        let handoff = self.cut.and_then(|cut| {
            let (from, to) = (self.place[cut as usize - 1], self.place[cut as usize]);
            (from <= stage && stage < to).then_some(self.handoff)
        });
        HopPayload {
            front_extra_row_bytes: crosses(true),
            late_extra_row_bytes: crosses(false),
            handoff,
            kv_entries: self.entries(&|source, writer| {
                if writer > stage {
                    return Vec::new();
                }
                self.readers(source)
                    .filter(|&l| self.place[l as usize] > stage)
                    .collect()
            }),
        }
    }

    /// The return hop from the last stage to the first (folded splits).
    fn back(&self) -> Result<Option<HopPayload>, BuildError> {
        let handoff = match self.cut {
            Some(cut) => {
                let (from, to) = (self.place[cut as usize - 1], self.place[cut as usize]);
                match to {
                    _ if to >= from => None,
                    0 => Some(self.handoff),
                    _ => {
                        return Err(fit_failed(format!(
                            "the cut layer {cut} sits on stage {to}, before layer {} on stage \
                             {from}: only a return to the first stage is modeled",
                            cut - 1
                        )))
                    }
                }
            }
            None => None,
        };
        let misplaced = std::cell::Cell::new(None);
        let kv_entries = self.entries(&|source, writer| {
            let readers: Vec<u32> = self
                .readers(source)
                .filter(|&l| self.place[l as usize] < writer)
                .collect();
            if readers.iter().any(|&l| self.place[l as usize] != 0) {
                misplaced.set(Some(source));
            }
            readers
        });
        if let Some(source) = misplaced.get() {
            return Err(fit_failed(format!(
                "KV source {source} is read on a stage after the first but before its writer"
            )));
        }
        Ok(
            (handoff.is_some() || !kv_entries.is_empty()).then_some(HopPayload {
                handoff,
                kv_entries,
                ..HopPayload::default()
            }),
        )
    }
}

/// One stage's recipe.
pub struct DeepseekV41PpStageConfigs {
    /// The stage's layer ranges, split at the replay cut.
    pub layers: Vec<(u32, u32)>,
    /// Layer 0's body, on the first stage.
    pub layer0: Option<DeepseekV41LayerConfig>,
    /// Bodies of the stage's other layers, in first-visit order of `plan`.
    pub bodies: Vec<DeepseekV41LayerConfig>,
    plan: Vec<PlanNode>,
    /// The bounded-replay rule, on every stage with a layer at or past the
    /// cut (they normalize the replay batch).
    pub replay_rule: Option<BoundedReplay>,
    /// The cut layer's `write_kv` and gather, on the stage [`CutHandoff`]
    /// puts them, run before top-level plan node `cut_at`.
    pub cut: Option<DeepseekV41ReplayConfig>,
    pub cut_at: usize,
    /// The scatter back to batch rows, on the last stage.
    pub scatter: Option<ElementwiseKernelConfig>,
    pub late_bodies: Vec<bool>,
    /// The cut layer's own body, on the stage holding it: on a replay step
    /// the cut's `write_kv` already stored its KV.
    pub cut_body: Option<usize>,
    /// What this stage sends to the next one; `None` on the last stage.
    pub send: Option<HopPayload>,
    /// What the last stage sends back to the first (a folded split).
    pub return_hop: Option<HopPayload>,
    /// The split's back layers run before its front reaches the cut: the
    /// stage costs only a deferred back pass.
    pub deferred_only: bool,
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
    /// `None`: the Engram fetch priced out.
    pub engram_prefetch: Option<DeepseekV41EngramPrefetchLocalWorkletConfig>,
    pub head: DeepseekV41HeadTpWorkletConfig,
    pub stages: Vec<DeepseekV41PpStageConfigs>,
    /// Whether the pinned vLLM fork runs this split ([`fork_accepts`]).
    pub fork_accepts: bool,
    /// The whole model's cache, one copy over the stages (the stage copies
    /// of upstream KV sources are in the stage layouts only).
    pub kv_cache_layout: KvCacheLayout,
    pub graphs: DeepseekV41CudaGraphs,
    /// The sliding window (rows a request keeps past the cut).
    pub window: u32,
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
    let lists = stage_layer_lists(model.num_layers, parallel)?;
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
    // DeepInfer's YoCo truncation is the bounded-replay cut keeping
    // `yoco_keep_tokens` rows per request (as in the DP arch).
    let replay_on = match &parallel.deepinfer {
        Some(engine) => engine.yoco_keep_tokens.is_some(),
        None => parallel.decoder_swa_bounded_replay,
    };
    let rule = if replay_on {
        let mut rule = BoundedReplay::from_model(model)?;
        if let Some(keep) = parallel
            .deepinfer
            .and_then(|engine| engine.yoco_keep_tokens)
        {
            if keep < rule.window {
                return Err(fit_failed(format!(
                    "yoco_keep_tokens {keep} must cover the sliding window {}",
                    rule.window
                )));
            }
            rule.window = keep;
        }
        Some(rule)
    } else {
        None
    };
    if parallel
        .deepinfer
        .is_some_and(|engine| engine.num_redundant_experts > 0)
    {
        return Err(fit_failed(
            "a PP stage holds every expert (EP1): no redundant expert slots",
        ));
    }
    if parallel.deepinfer.is_some() {
        // The mega MoE prices each layer from its own routes.
        for layer in 0..model.num_layers {
            demand
                .layer(layer as usize)
                .map_err(|err| fit_failed(format!("mega MoE demand: {err:#}")))?;
        }
    }
    let graphs = DeepseekV41CudaGraphs::new(
        &parallel.cudagraph_capture_sizes,
        tp_arch::MAX_NUM_SEQS,
        &[1],
    );
    let max_gpu_tokens = parallel.max_num_batched_tokens.max(graphs.max_tokens());
    // A layer's configs on one GPU: the TP arch's at TP1/EP1, with the
    // engine's kernels.
    let stage_layer = |kind: LayerKind, layers: Vec<u32>| {
        let mut config = layer_config(model, &local, demand, kind, layers);
        if let Some(engine) = &parallel.deepinfer {
            dp_arch::deepinfer_layer(&mut config, model, &kind, engine, max_gpu_tokens);
        }
        config
    };
    let multi_range = lists.iter().any(|list| list.len() > 1);
    if multi_range && rule.is_none() {
        return Err(fit_failed(
            "a stage with several layer ranges needs bounded replay (the back pass)",
        ));
    }
    let geometry = HopGeometry {
        model,
        place: layer_places(model.num_layers, &lists),
        cut: rule.map(|rule| rule.cut_layer),
        handoff: parallel.cut_handoff,
    };
    let place = &geometry.place;
    // Front layers, and late layers, each run in stage order.
    for (name, layers) in [
        (
            "front",
            (0..geometry.cut.unwrap_or(model.num_layers)).collect::<Vec<_>>(),
        ),
        (
            "late",
            (geometry.cut.unwrap_or(model.num_layers)..model.num_layers).collect(),
        ),
    ] {
        if layers
            .windows(2)
            .any(|w| place[w[0] as usize] > place[w[1] as usize])
        {
            return Err(fit_failed(format!(
                "the {name} layers must run in stage order: {lists:?}"
            )));
        }
    }
    let return_hop = geometry.back()?;
    let deferred_only = return_hop.as_ref().is_some_and(|hop| hop.handoff.is_some());
    // The stage that runs the cut's `write_kv`.
    let cut_stage = rule.map(|rule| geometry.writer(rule.cut_layer));
    let gpu = parallel.gpu_name.clone();
    let last = lists.len() - 1;
    let mut stages = Vec::with_capacity(lists.len());
    for (index, list) in lists.iter().enumerate() {
        let segments = stage_segments(list, geometry.cut);
        let layer0 = (segments[0].0 == 0).then(|| stage_layer(layer0_kind, vec![0]));
        let mut bodies = Vec::new();
        let mut plan = Vec::new();
        let mut front_nodes = 0;
        for &(start, end) in &segments {
            let mut tail: Vec<(u32, LayerKind)> = (start.max(1)..end)
                .map(|layer| (layer, model.layer_kind(layer)))
                .collect();
            if parallel.deepinfer.is_some() {
                // As in the DP arch: a forward's first candidate consumer
                // builds the sparse schedule the later ones reuse. Each
                // segment is its own forward on this GPU.
                if let Some((_, kind)) = tail.iter_mut().find(|(_, kind)| {
                    kind.attention.candidate == DeepseekV41CandidateRole::Consumer
                }) {
                    kind.candidate_metadata = true;
                }
            }
            let nodes = index_plan(&fold_layers(&tail), &mut |kind, layers| {
                bodies.push(stage_layer(kind, layers.to_vec()));
                bodies.len() - 1
            });
            if geometry.is_front(start) {
                front_nodes += nodes.len();
            }
            plan.extend(nodes);
        }
        let holds_late = geometry
            .cut
            .is_some_and(|cut| segments.iter().any(|&(_, end)| end > cut));
        let replay_rule = rule.filter(|_| holds_late || cut_stage == Some(index as u16));
        let late_bodies = match replay_rule {
            Some(rule) => bodies
                .iter()
                .map(|body: &DeepseekV41LayerConfig| rule.is_late(&body.layers))
                .collect::<Result<Vec<_>, _>>()?,
            None => vec![false; bodies.len()],
        };
        let cut_body =
            match rule.filter(|rule| geometry.place[rule.cut_layer as usize] == index as u16) {
                Some(rule) => {
                    let body = bodies
                        .iter()
                        .position(|body| body.layers == [rule.cut_layer])
                        .ok_or_else(|| {
                            fit_failed(format!(
                                "decoder SWA bounded replay: cut layer {} shares its body",
                                rule.cut_layer
                            ))
                        })?;
                    // As in the DP arch: DeepInfer rebuilds `kv_swa_full_index`
                    // for the compact rows at the cut.
                    if let Some(di) = bodies[body].attention.deepinfer.as_mut() {
                        di.builds_swa_index = true;
                    }
                    Some(body)
                }
                None => None,
            };
        let cut = match rule.filter(|_| cut_stage == Some(index as u16)) {
            Some(rule) => {
                let write_kv =
                    stage_layer(model.layer_kind(rule.cut_layer), vec![rule.cut_layer]).attention;
                Some(DeepseekV41ReplayConfig::new(rule, 0, write_kv, model, &gpu))
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
        let held: Vec<u32> = list.iter().flat_map(|&(start, end)| start..end).collect();
        let copies = copied_sources(model, &held);
        stages.push(DeepseekV41PpStageConfigs {
            layers: segments,
            layer0,
            bodies,
            plan,
            replay_rule,
            cut,
            cut_at: front_nodes,
            scatter,
            late_bodies,
            cut_body,
            send: (index != last).then(|| geometry.forward(index as u16)),
            return_hop: (index == last).then(|| return_hop.clone()).flatten(),
            deferred_only,
            kv_cache_layout: match &parallel.deepinfer {
                Some(engine) => {
                    let own: Vec<u32> = model
                        .kv_source_layer_ids
                        .iter()
                        .copied()
                        .filter(|source| held.contains(source))
                        .collect();
                    let copied: Vec<u32> = copies.iter().map(|&(source, _)| source).collect();
                    // DeepInfer's MTP layers keep their sliding windows on
                    // the stage holding the last body layer.
                    let mtp = match index == last {
                        true => u64::from(tp_arch::MTP_LAYERS),
                        false => 0,
                    };
                    dp_arch::deepinfer_kv_cache_layout(
                        model,
                        &own,
                        &copied,
                        held.len() as u64 + mtp,
                        engine,
                    )
                }
                None => stage_kv_cache_layout(model, &held, &copies),
            },
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
        engram_prefetch: Some(DeepseekV41EngramPrefetchLocalWorkletConfig {
            tp_size: 1,
            engram_num_heads: model.engram_hash_heads,
            engram_head_dim: model.engram_head_dim,
            quant_block_size: 32,
            table_rows: ENGRAM_TABLE_ROWS_TP1,
            residency: "host_uva".into(),
            weight_dtype: DType::Fp8E4m3,
            gpu_name: gpu.clone(),
            backends: tp_arch::ENGRAM_LOOKUP_BACKENDS.to_vec(),
            token_scale: 1,
        }),
        head: DeepseekV41HeadTpWorkletConfig {
            tp_size: 1,
            hidden_size: model.hidden_size.into(),
            hc_mult: model.hc_mult,
            vocab_size: model.vocab_size,
            gpu_name: gpu,
            lm_head_backends: tp_arch::LM_HEAD_BACKENDS.to_vec(),
            all_gather_backends: tp_arch::ALL_GATHER_PROXY_BACKENDS.to_vec(),
            elementwise_backends: tp_arch::ELEMENTWISE_BACKENDS.to_vec(),
            mhc_post_backends: parallel
                .deepinfer
                .map(|_| dp_arch::DEEPINFER_MHC_UNFUSED_BACKENDS.to_vec()),
        },
        fork_accepts: fork_accepts_lists(model, &lists, rule.is_some()),
        stages,
        kv_cache_layout,
        window: rule.map_or(model.sliding_window, |rule| rule.window),
        graphs,
    })
}

pub struct DeepseekV41PpStageResolved {
    layers: Vec<(u32, u32)>,
    layer0: Option<DeepseekV41LayerResolved>,
    bodies: Vec<DeepseekV41LayerResolved>,
    plan: Vec<PlanNode>,
    replay_rule: Option<BoundedReplay>,
    cut: Option<CutResolved>,
    scatter: Option<ElementwiseKernelConfig>,
    late_bodies: Vec<bool>,
    cut_body: Option<usize>,
    send: Option<HopPayload>,
    return_hop: Option<HopPayload>,
    deferred_only: bool,
    kv_cache_layout: KvCacheLayout,
}

struct CutResolved {
    cut_layer: u32,
    cut_at: usize,
    write_kv: DeepseekV41AttentionTpWorkletResolved,
    gather: ElementwiseKernelConfig,
}

pub struct DeepseekV41VllmPpResolved {
    parallel: DeepseekV41VllmPpParallel,
    engram_layer_ids: Vec<u32>,
    embedding: ElementwiseKernelConfig,
    engram_hash: ElementwiseKernelConfig,
    engram_prefetch: Option<DeepseekV41EngramPrefetchLocalWorkletResolved>,
    head: DeepseekV41HeadTpWorkletResolved,
    stages: Vec<DeepseekV41PpStageResolved>,
    fork_accepts: bool,
    kv_cache_layout: KvCacheLayout,
    graphs: DeepseekV41CudaGraphs,
    hc_mult: u32,
    hidden: u32,
    window: u32,
}

pub fn resolve_configs(configs: &DeepseekV41VllmPpConfigs) -> DeepseekV41VllmPpResolved {
    let prologue = DeepseekV41PrologueTpWorklet::resolve_config(&configs.prologue);
    DeepseekV41VllmPpResolved {
        parallel: configs.parallel.clone(),
        engram_layer_ids: configs.engram_layer_ids.clone(),
        embedding: prologue.embedding,
        engram_hash: prologue.engram_hash,
        engram_prefetch: configs
            .engram_prefetch
            .as_ref()
            .map(DeepseekV41EngramPrefetchLocalWorklet::resolve_config),
        head: DeepseekV41HeadTpWorklet::resolve_config(&configs.head),
        stages: configs
            .stages
            .iter()
            .map(|stage| DeepseekV41PpStageResolved {
                layers: stage.layers.clone(),
                layer0: stage.layer0.as_ref().map(resolve_layer),
                bodies: stage.bodies.iter().map(resolve_layer).collect(),
                plan: stage.plan.clone(),
                replay_rule: stage.replay_rule,
                cut: stage.cut.as_ref().map(|cut| CutResolved {
                    cut_layer: cut.rule.cut_layer,
                    cut_at: stage.cut_at,
                    write_kv: DeepseekV41AttentionTpWorklet::resolve_config(&cut.write_kv),
                    gather: cut.gather.clone(),
                }),
                scatter: stage.scatter.clone(),
                late_bodies: stage.late_bodies.clone(),
                cut_body: stage.cut_body,
                send: stage.send.clone(),
                return_hop: stage.return_hop.clone(),
                deferred_only: stage.deferred_only,
                kv_cache_layout: stage.kv_cache_layout.clone(),
            })
            .collect(),
        fork_accepts: configs.fork_accepts,
        kv_cache_layout: configs.kv_cache_layout.clone(),
        graphs: configs.graphs.clone(),
        hc_mult: configs.head.hc_mult,
        hidden: configs.head.hidden_size.get(),
        window: configs.window,
    }
}

/// The built cut-layer boundary of the stage holding the replay cut.
struct StageCut {
    /// Top-level plan node the boundary runs before.
    at: usize,
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
    engram_prefetch: Option<&'a DeepseekV41EngramPrefetchLocalWorkletResolved>,
    head: &'a DeepseekV41HeadTpWorkletResolved,
    graphs: &'a DeepseekV41CudaGraphs,
    max_model_len: u32,
    hidden_row_bytes: u64,
    /// One row's collapsed hidden state (bf16), what `V2` sends for a row
    /// that stops at the cut.
    collapsed_row_bytes: u64,
    window: u32,
    /// The `cpu_gather` Engram fetch; `None` is UVA.
    host_gather: Option<EngramHostGather>,
    /// DeepInfer's host time per stage step (ms; 0 = not modeled).
    host_step_ms: f32,
    /// DeepInfer's GPU idle inside a stage step (ms; 0 = not modeled).
    step_gap_ms: f32,
    /// Both fixed per-step times on stage 0 alone (one per pipeline round).
    first_stage_only: bool,
}

/// One pipeline stage: one GPU and its layer ranges.
pub struct DeepseekV41VllmPpStageModel {
    name: String,
    stage_index: u16,
    num_stages: u16,
    layer_ranges: Vec<(u32, u32)>,
    embedding: Option<Op<ElementwiseKernel>>,
    engram_hash: Option<Op<ElementwiseKernel>>,
    /// One lookup per Engram layer this stage holds.
    engram_lookups: Vec<Op<EngramLookupKernel>>,
    /// Lookup tokens per scheduled token (the prefetch config's `token_scale`).
    engram_lookup_scale: u32,
    layer0: Option<DeepseekV41LayerBody>,
    bodies: Vec<DeepseekV41LayerBody>,
    plan: Vec<PlanNode>,
    replay_rule: Option<BoundedReplay>,
    cut: Option<StageCut>,
    scatter: Option<Op<ElementwiseKernel>>,
    late_bodies: Vec<bool>,
    /// The cut layer's body: on a replay step it skips the KV stores the
    /// cut's `write_kv` ran.
    cut_body: Option<usize>,
    head: Option<DeepseekV41HeadTpWorklet>,
    send: Option<HopPayload>,
    return_hop: Option<HopPayload>,
    deferred_only: bool,
    hidden_row_bytes: u64,
    collapsed_row_bytes: u64,
    window: u32,
    graphs: DeepseekV41CudaGraphs,
    max_model_len: u32,
    kv_cache_layout: KvCacheLayout,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
    /// `Some` under the `cpu_gather` Engram fetch on a stage holding an
    /// Engram layer: the last slot is the GPU time a microbatch waits for its
    /// host-gathered rows.
    host_gather: Option<EngramHostGather>,
    /// DeepInfer's host time per step with this GPU idle (ms), one
    /// `host_step` leaf per stage step; 0 = no leaf.
    host_step_ms: f32,
    /// DeepInfer's GPU idle inside a stage step (ms), one `step_gaps` leaf
    /// per stage step; 0 = no leaf.
    step_gap_ms: f32,
    /// Per local Engram layer, the stage's layers that run before it.
    engram_layers_ahead: Vec<u32>,
    /// Decoder layers of each root section of the stage's cost tree, in
    /// order (see [`deepseek_v41_engram_fetch::time_ahead_ms`]).
    section_layers: Vec<u32>,
}

impl DeepseekV41VllmPpStageModel {
    fn build(
        ctx: &StageContext,
        stage: DeepseekV41PpStageResolved,
        stage_index: u16,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let name = ctx.name;
        let local_engram: Vec<u32> = ctx
            .engram_layer_ids
            .iter()
            .copied()
            .filter(|layer| stage.layers.iter().any(|(s, e)| (*s..*e).contains(layer)))
            .collect();
        let engram_lookups = local_engram
            .iter()
            .filter_map(|layer| ctx.engram_prefetch.map(|prefetch| (layer, prefetch)))
            .map(|(layer, prefetch)| {
                let leaf = format!("{name}.engram_prefetch.layer{layer}.engram_lookup");
                Ok(Op::new(
                    leaf.clone(),
                    Arc::new(EngramLookupKernel::build(
                        leaf,
                        prefetch.lookup.clone(),
                        bridge,
                    )?),
                ))
            })
            .collect::<Result<Vec<_>, BuildError>>()?;
        let cut = match stage.cut {
            Some(cut) => Some(StageCut {
                at: cut.cut_at,
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
            layer_ranges: stage.layers,
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
            engram_lookup_scale: ctx.engram_prefetch.map_or(1, |p| p.raw_cfg.token_scale),
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
            cut_body: stage.cut_body,
            head: match is_last {
                true => Some(DeepseekV41HeadTpWorklet::build(
                    format!("{name}.head"),
                    ctx.head.clone(),
                    bridge,
                )?),
                false => None,
            },
            send: stage.send,
            return_hop: stage.return_hop,
            deferred_only: stage.deferred_only,
            hidden_row_bytes: ctx.hidden_row_bytes,
            collapsed_row_bytes: ctx.collapsed_row_bytes,
            window: ctx.window,
            graphs: ctx.graphs.clone(),
            max_model_len: ctx.max_model_len,
            kv_cache_layout: stage.kv_cache_layout,
            cost_flat: Vec::new(),
            n_slots: 0,
            host_gather: ctx.host_gather.filter(|_| !local_engram.is_empty()),
            host_step_ms: match ctx.first_stage_only && stage_index > 0 {
                true => 0.0,
                false => ctx.host_step_ms,
            },
            step_gap_ms: match ctx.first_stage_only && stage_index > 0 {
                true => 0.0,
                false => ctx.step_gap_ms,
            },
            engram_layers_ahead: Vec::new(),
            section_layers: Vec::new(),
        };
        let order: Vec<u32> = model
            .layer_ranges
            .iter()
            .flat_map(|&(start, end)| start..end)
            .collect();
        model.engram_layers_ahead = local_engram
            .iter()
            .map(|layer| order.iter().position(|l| l == layer).unwrap() as u32)
            .collect();
        let tree = model.cost_tree();
        model.cost_flat = tree.flatten();
        model.n_slots = tree.n_slots();
        model.section_layers = model.section_layers();
        Ok(model)
    }

    /// Decoder layers of each root section of [`Self::compile`], in its order.
    fn section_layers(&self) -> Vec<u32> {
        let mut out = Vec::new();
        out.extend(self.embedding.iter().map(|_| 0));
        out.extend(self.engram_hash.iter().map(|_| 0));
        out.extend(self.layer0.iter().map(|_| 1));
        if !self.engram_lookups.is_empty() {
            out.push(0);
        }
        let cut_at = self.cut.as_ref().map(|c| c.at);
        for (position, node) in self.plan.iter().enumerate() {
            if cut_at == Some(position) {
                out.extend([0, 0]);
            }
            out.push(deepseek_v41_engram_fetch::plan_layers(
                std::slice::from_ref(node),
            ));
        }
        if cut_at == Some(self.plan.len()) {
            out.extend([0, 0]);
        }
        out.extend(self.scatter.iter().map(|_| 0));
        out.extend(self.head.iter().map(|_| 0));
        out.extend(
            [
                self.host_step_ms > 0.0,
                self.step_gap_ms > 0.0,
                self.host_gather.is_some(),
            ]
            .into_iter()
            .filter(|&on| on)
            .map(|_| 0),
        );
        let layers: u32 = self.layer_ranges.iter().map(|&(s, e)| e - s).sum();
        assert_eq!(
            out.iter().sum::<u32>(),
            layers,
            "stage sections cover its layers"
        );
        out
    }

    /// Fill the `cpu_gather` wait slot (the last) of a stage evaluation and
    /// add it to `total`: GPU time ahead of a local Engram layer is the
    /// priced sections of this stage before it (`scratch`: the aggregate's
    /// node buffer).
    fn add_host_gather_wait(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut [LeafMetrics],
        scratch: &[LeafMetrics],
        mut total: LeafMetrics,
    ) -> LeafMetrics {
        let Some(gather) = &self.host_gather else {
            return total;
        };
        let (_, input) = self
            .normalize(batch)
            .unwrap_or_else(|reason| panic!("invalid DeepSeek-V4.1 PP stage input: {reason}"));
        // GPU time only: the host step runs before the forward.
        let gpu_ms = f64::from(total.m.time_ms - self.host_step_ms);
        let available = deepseek_v41_engram_fetch::time_ahead_ms(
            &self.cost_flat,
            scratch,
            &self.section_layers,
            &self.engram_layers_ahead,
        );
        let wait = gather.exposed_ms(input.full.rows, &available, gpu_ms) as f32;
        slots[self.n_slots - 1].m.time_ms = wait;
        total.m.time_ms += wait;
        total
    }

    pub fn stage_index(&self) -> u16 {
        self.stage_index
    }

    /// The `[start, end)` decoder layer ranges this stage runs (split at the
    /// replay cut).
    pub fn layer_ranges(&self) -> &[(u32, u32)] {
        &self.layer_ranges
    }

    /// What this stage sends beside the hidden state; `None` on the last stage.
    pub fn send_payload(&self) -> Option<&HopPayload> {
        self.send.as_ref()
    }

    /// What the last stage of a folded split sends back to the first.
    pub fn return_payload(&self) -> Option<&HopPayload> {
        self.return_hop.as_ref()
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
        for (position, node) in self.plan.iter().enumerate() {
            if let Some(cut) = self.cut.as_ref().filter(|c| c.at == position) {
                children.push(cut.write_kv.compile(builder));
                children.push(cut.gather.compile(builder));
            }
            children.extend(self.compile_plan(std::slice::from_ref(node), builder));
        }
        if let Some(cut) = self.cut.as_ref().filter(|c| c.at == self.plan.len()) {
            children.push(cut.write_kv.compile(builder));
            children.push(cut.gather.compile(builder));
        }
        if let Some(scatter) = &self.scatter {
            children.push(scatter.compile(builder));
        }
        if let Some(head) = &self.head {
            children.push(head.compile(builder));
        }
        if self.host_step_ms > 0.0 {
            children.push(CostNode::Labeled {
                label: format!("{}.stage{}.host_step [engine host time per step, GPU idle: schedule, input prep, post-process]", self.name, self.stage_index),
                child: Box::new(builder.leaf(
                    format!("{}.stage{}.host_step", self.name, self.stage_index),
                    "host_step",
                    serde_json::json!({}),
                )),
            });
        }
        if self.step_gap_ms > 0.0 {
            children.push(CostNode::Labeled {
                label: format!(
                    "{}.stage{}.step_gaps [GPU idle between the kernels of one eager forward]",
                    self.name, self.stage_index
                ),
                child: Box::new(builder.leaf(
                    format!("{}.stage{}.step_gaps", self.name, self.stage_index),
                    "step_gaps",
                    serde_json::json!({}),
                )),
            });
        }
        if self.host_gather.is_some() {
            children.push(CostNode::Labeled {
                label: format!(
                    "{}.stage{}.engram_host_gather [GPU wait for CPU gather + H2D]",
                    self.name, self.stage_index
                ),
                child: Box::new(builder.leaf(
                    format!("{}.stage{}.engram_host_gather", self.name, self.stage_index),
                    "engram_host_gather_wait",
                    serde_json::json!({}),
                )),
            });
        }
        CostNode::Labeled {
            label: format!(
                "{} pipeline stage {} of {} (DeepseekV41VllmPpStageModel) [layers {:?}; TP1/EP1 \
                 on one GPU, no collectives; replay_cut_here={}; sends={:?}; returns={:?}]",
                self.name,
                self.stage_index,
                self.num_stages,
                self.layer_ranges,
                self.cut.is_some(),
                self.send,
                self.return_hop,
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    fn compile_plan(&self, plan: &[PlanNode], builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut out = Vec::with_capacity(plan.len());
        for node in plan {
            match node {
                PlanNode::Body(index) => out.push(self.bodies[*index].compile(builder)),
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
        // A stage wholly before the cut (no replay rule) runs every row of
        // its layers whether or not the back pass is deferred; the back pass
        // is not its work.
        let back = match self.replay_rule {
            Some(_) => tp_arch::group_back_pairs(batch, self.max_model_len)?,
            None => None,
        };
        if self.deferred_only && back.is_none() && !step.prefill.is_empty() {
            return Err("a folded split runs prefill only with a deferred back pass".into());
        }
        let input = normalize_step(
            &self.graphs,
            &step,
            1,
            logits_rows,
            self.replay_rule.as_ref(),
            back.as_deref(),
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
                EngramLookupKernelInput {
                    num_tokens: rows * self.engram_lookup_scale,
                },
                zero,
                ev,
            );
        }
        for (position, node) in self.plan.iter().enumerate() {
            if let Some(cut) = self.cut.as_ref().filter(|c| c.at == position) {
                self.eval_cut(cut, input, ev);
            }
            self.eval_plan(std::slice::from_ref(node), input, ev);
        }
        if let Some(cut) = self.cut.as_ref().filter(|c| c.at == self.plan.len()) {
            self.eval_cut(cut, input, ev);
        }
        if let Some(scatter) = &self.scatter {
            let scattered = input.scatter_rows();
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
                    num_tokens: input.head_rows(),
                    mhc_post_rows: input.replay.as_ref().map_or(rows, |late| late.rows),
                    logits_rows: input.logits_rows,
                },
                ev,
            );
        }
        if self.host_step_ms > 0.0 {
            let host = LeafMetrics {
                m: Metrics4 {
                    time_ms: self.host_step_ms,
                    ..Metrics4::ZERO
                },
                ..LeafMetrics::ZERO
            };
            ev.push(host, || {
                SlotInput::from(ElementwiseKernelInput { num_tokens: rows })
            });
        }
        if self.step_gap_ms > 0.0 {
            let gaps = LeafMetrics {
                m: Metrics4 {
                    time_ms: self.step_gap_ms,
                    ..Metrics4::ZERO
                },
                ..LeafMetrics::ZERO
            };
            ev.push(gaps, || {
                SlotInput::from(ElementwiseKernelInput { num_tokens: rows })
            });
        }
        if self.host_gather.is_some() {
            // Filled after the stage's total is known (`add_host_gather_wait`);
            // the whole-pipeline view keeps it at zero.
            ev.push(LeafMetrics::ZERO, || {
                SlotInput::from(ElementwiseKernelInput { num_tokens: rows })
            });
        }
    }

    fn eval_plan(&self, plan: &[PlanNode], input: &NormalizedInput, ev: &mut Evaluator) {
        for node in plan {
            match node {
                PlanNode::Body(index) => {
                    let body_input = match &input.replay {
                        Some(late) if self.cut_body == Some(*index) => {
                            // The cut's `write_kv` stored this layer's main and index KV.
                            let mut late = late.clone();
                            late.attention.kv_written = true;
                            self.bodies[*index].eval(&late, ev);
                            continue;
                        }
                        Some(late) if self.late_bodies[*index] => late,
                        _ => &input.full,
                    };
                    self.bodies[*index].eval(body_input, ev)
                }
                PlanNode::Repeat { body, .. } => self.eval_plan(body, input, ev),
            }
        }
    }

    /// The cut's `write_kv` over every front row and the gather of the late
    /// rows; both zero on a step that does not replay.
    fn eval_cut(&self, cut: &StageCut, input: &NormalizedInput, ev: &mut Evaluator) {
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
                kv_written: false,
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
        Some(self.hop_bytes(payload, batch))
    }

    /// Bytes the last stage of a folded split sends back to the first for
    /// `batch` (the continuing rows and the cut source's entries the first
    /// stage's back layers read). `None` without a return hop.
    pub fn return_bytes(&self, batch: &UnifiedArchInput) -> Option<u64> {
        let payload = self.return_hop.as_ref()?;
        Some(self.hop_bytes(payload, batch))
    }

    fn hop_bytes(&self, payload: &HopPayload, batch: &UnifiedArchInput) -> u64 {
        let (step, input) = self
            .normalize(batch)
            .unwrap_or_else(|reason| panic!("invalid DeepSeek-V4.1 PP stage input: {reason}"));
        let front = u64::from(input.full.rows);
        let late = u64::from(
            input
                .replay
                .as_ref()
                .map_or(input.full.rows, |late| late.rows),
        );
        let hidden = self.hidden_row_bytes;
        let mut bytes = 0;
        if let Some(extra) = payload.front_extra_row_bytes {
            bytes += front * (hidden + extra);
        }
        if let Some(extra) = payload.late_extra_row_bytes {
            bytes += late * (hidden + extra);
        }
        if let Some(handoff) = payload.handoff {
            // The rows that continue past the cut: this step's own replay
            // rows, or (deferred) every chunk's last `window` rows, an upper
            // bound of a prompt-tail rule.
            let continuing = if input.deferred {
                step.prefill
                    .iter()
                    .chain(&step.decode)
                    .map(|&(q, _)| q.min(self.window))
                    .sum::<u32>()
            } else {
                input
                    .replay
                    .as_ref()
                    .map_or(input.full.rows, |late| late.rows)
            };
            let continuing = u64::from(continuing).min(front);
            bytes += match handoff {
                CutHandoff::V1 => front * hidden,
                CutHandoff::V2 => {
                    continuing * hidden + (front - continuing) * self.collapsed_row_bytes
                }
                CutHandoff::V3 => {
                    // The cut layer's SWA records below the continuing rows.
                    let swa: u64 = step
                        .prefill
                        .iter()
                        .map(|&(q, _)| {
                            let kept = q.min(self.window);
                            u64::from((q - kept).min(self.window))
                        })
                        .sum();
                    continuing * hidden + swa * u64::from(MXFP8_RECORD_BYTES)
                }
            };
        }
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
            .map(|&(ratio, entry_bytes)| {
                let entries: u64 = spans
                    .iter()
                    .map(|&(from, to)| u64::from(to / ratio - from / ratio))
                    .sum();
                entries * entry_bytes
            })
            .sum();
        bytes + kv
    }
}

impl IterwiseUnifiedModel for DeepseekV41VllmPpStageModel {
    fn back_pass_window(&self) -> Option<u32> {
        Some(self.window)
    }

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
        let total = eval_compiled(&self.cost_flat, self.n_slots, slots, scratch, None, |ev| {
            self.eval_batch(batch, ev)
        });
        self.add_host_gather_wait(batch, slots, scratch, total)
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        let total = eval_compiled(
            &self.cost_flat,
            self.n_slots,
            slots,
            scratch,
            Some(inputs),
            |ev| self.eval_batch(batch, ev),
        );
        self.add_host_gather_wait(batch, slots, scratch, total)
    }
}

/// Build every stage, then the whole-pipeline view over them.
pub fn build(
    name: String,
    resolved: DeepseekV41VllmPpResolved,
    bridge: &PerfApiBridge,
) -> Result<DeepseekV41VllmPpModel, BuildError> {
    build_with_engram_fetch(name, resolved, None, bridge)
}

/// [`build`] with the Engram fetch: `Some` is `cpu_gather` (the configs carry
/// the unpack's prefetch config, [`EngramHostGather::from_prefetch`]).
pub fn build_with_engram_fetch(
    name: String,
    resolved: DeepseekV41VllmPpResolved,
    host_gather: Option<EngramHostGather>,
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
        window,
    } = resolved;
    let hc = u64::from(hc_mult);
    let ctx = StageContext {
        host_gather,
        host_step_ms: parallel
            .deepinfer
            .map_or(0.0, |engine| engine.host_step_us as f32 / 1000.0),
        step_gap_ms: parallel
            .deepinfer
            .map_or(0.0, |engine| engine.step_gap_us as f32 / 1000.0),
        first_stage_only: parallel
            .deepinfer
            .is_some_and(|engine| engine.host_first_stage_only),
        name: &name,
        num_stages: stages.len() as u16,
        engram_layer_ids: &engram_layer_ids,
        embedding: &embedding,
        engram_hash: &engram_hash,
        engram_prefetch: engram_prefetch.as_ref(),
        head: &head,
        graphs: &graphs,
        max_model_len: parallel.max_model_len,
        // The fork's PP `IntermediateTensors`
        // (`make_empty_intermediate_tensors`): the `(hc_mult, hidden)` bf16
        // stream and the `(hc_mult,)` fp32 pre-mix.
        hidden_row_bytes: hc * u64::from(hidden) * 2 + hc * 4,
        collapsed_row_bytes: u64::from(hidden) * 2,
        window,
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
    fn back_pass_window(&self) -> Option<u32> {
        self.stages[0].back_pass_window()
    }

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
            stage_layers: Vec::new(),
            cut_handoff: CutHandoff::V1,
            deepinfer: None,
        }
    }

    /// GPU0 `[0, 10) + [20, back)`, GPU1 `[10, 20) + [back, 40)`.
    fn fold_parallel(back: u32, handoff: CutHandoff) -> DeepseekV41VllmPpParallel {
        DeepseekV41VllmPpParallel {
            stage_layers: vec![vec![(0, 10), (20, back)], vec![(10, 20), (back, 40)]],
            cut_handoff: handoff,
            ..parallel(&[20, 20])
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
            back_pass: None,
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
        let own = stage_kv_cache_layout(&model_cfg(), &(24..40).collect::<Vec<_>>(), &[]);
        assert!(cfgs.stages[4].kv_cache_layout.per_token_bytes() > own.per_token_bytes());
    }

    #[test]
    fn boundaries_carry_what_the_next_stage_reads() {
        let model = model_cfg();
        let hop = |partition: &[u32], stage: usize| {
            configs(partition).stages[stage].send.clone().unwrap()
        };
        // Between groups: hidden only.
        let p = hop(&[8, 32], 0);
        assert_eq!(p.front_extra_row_bytes, Some(0));
        assert!(p.kv_entries.is_empty() && p.late_extra_row_bytes.is_none() && p.handoff.is_none());
        // Inside [2, 7]: layer 2's compressed record and its top-k.
        let p = hop(&[5, 35], 0);
        assert_eq!(p.kv_entries, [(2, u64::from(NVFP4_RECORD_BYTES))]);
        assert_eq!(p.front_extra_row_bytes, Some(512 * 4));
        // At 24: past the cut, layer 20's record + index K, its candidates;
        // 24 is an index source, so no top-k.
        let p = hop(&[24, 16], 0);
        assert!(p.front_extra_row_bytes.is_none());
        assert_eq!(p.late_extra_row_bytes, Some(2048 * 4));
        assert_eq!(
            p.kv_entries,
            [(1, u64::from(NVFP4_RECORD_BYTES + FP8_INDEXER_ROW_BYTES))]
        );
        let _ = model;
    }

    #[test]
    fn a_fold_places_ranges_copies_and_the_v3_cut() {
        let cfgs =
            build_configs(&model_cfg(), &fold_parallel(30, CutHandoff::V3), &uniform()).unwrap();
        let (gpu0, gpu1) = (&cfgs.stages[0], &cfgs.stages[1]);
        assert_eq!(gpu0.layers, [(0, 10), (20, 30)]);
        assert_eq!(gpu1.layers, [(10, 20), (30, 40)]);
        assert!(!cfgs.fork_accepts);
        // V3: write_kv on GPU1 right after its front layers 10-19.
        assert!(gpu0.cut.is_none() && gpu1.cut.is_some());
        let front_nodes = {
            let tail: Vec<(u32, LayerKind)> =
                (10..20).map(|l| (l, model_cfg().layer_kind(l))).collect();
            fold_layers(&tail).len()
        };
        assert_eq!(gpu1.cut_at, front_nodes);
        // KV: GPU0 owns 2, 8, 20 (840 B/token unpadded); GPU1 owns 14 and
        // copies 8's record and 20's record + index K (774).
        assert!(gpu0.copied_sources.is_empty());
        assert_eq!(gpu1.copied_sources, [(8, false), (20, true)]);
        // Forward hop: the front (top-k for 10) and the late batch (top-k
        // for 30, candidates for 32/36), layer 8's records for 10-13.
        let fwd = gpu0.send.clone().unwrap();
        assert_eq!(fwd.front_extra_row_bytes, Some(512 * 4));
        assert_eq!(fwd.late_extra_row_bytes, Some(512 * 4 + 2048 * 4));
        assert_eq!(fwd.kv_entries, [(2, u64::from(NVFP4_RECORD_BYTES))]);
        // Return hop: the tails' state and layer 20's entries for GPU0.
        let back = gpu1.return_hop.clone().unwrap();
        assert_eq!(back.handoff, Some(CutHandoff::V3));
        assert_eq!(
            back.kv_entries,
            [(1, u64::from(NVFP4_RECORD_BYTES + FP8_INDEXER_ROW_BYTES))]
        );
        assert!(gpu0.deferred_only && gpu1.deferred_only);
        // The back cut is a parameter: 20-28 | 29-39.
        let cfgs =
            build_configs(&model_cfg(), &fold_parallel(29, CutHandoff::V3), &uniform()).unwrap();
        assert_eq!(cfgs.stages[1].layers, [(10, 20), (29, 40)]);
        // Front layers out of stage order are refused.
        let bad = DeepseekV41VllmPpParallel {
            stage_layers: vec![vec![(10, 20), (20, 30)], vec![(0, 10), (30, 40)]],
            ..fold_parallel(30, CutHandoff::V3)
        };
        assert!(build_configs(&model_cfg(), &bad, &uniform()).is_err());
    }

    #[test]
    fn a_fold_costs_front_and_back_on_each_gpu() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let model = build(
            "pp".into(),
            resolve_configs(
                &build_configs(&model_cfg(), &fold_parallel(30, CutHandoff::V3), &uniform())
                    .unwrap(),
            ),
            &bridge,
        )
        .unwrap();
        let mut batch = prefill(4096, 4096);
        // Without a back pass a fold refuses prefill.
        assert!(model.stages()[0].check_input(&batch).is_err());
        batch.back_pass = Some(vec![vec![(10_000, 128), (20_000, 100)]]);
        for stage in model.stages() {
            assert!(stage.check_input(&batch).is_ok());
        }
        let hidden = model.activation_bytes_per_token();
        // Return hop: 128 tail rows of the chunk + 128 SWA records below them
        // + layer 20's entries for all 4096 tokens.
        assert_eq!(
            model.stages()[1].return_bytes(&batch).unwrap(),
            128 * hidden
                + 128 * u64::from(MXFP8_RECORD_BYTES)
                + 4096 * u64::from(NVFP4_RECORD_BYTES + FP8_INDEXER_ROW_BYTES)
        );
        // An empty back-only step is accepted too (the stash probe's base).
        let mut empty = UnifiedArchInput {
            groups: vec![ArchGroupInput::default()],
            tokens_per_source_rank: Vec::new(),
            back_pass: Some(vec![vec![(1000, 128)]]),
        };
        assert!(model.check_input(&empty).is_ok());
        empty.back_pass = Some(vec![vec![]]);
        let _ = model.check_input(&empty);
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
    fn deepinfer_stages_run_an_ep1_mega_moe_and_keep_yoco_rows() {
        let engine = DeepInferEngine {
            num_redundant_experts: 0,
            yoco_keep_tokens: Some(140),
            engram_fetch: false,
            force_load_balance: false,
            prefill_chunk_tokens: None,
            host_step_us: 0,
            step_gap_us: 0,
            host_first_stage_only: false,
            max_num_seqs: 256,
        };
        let fold = DeepseekV41VllmPpParallel {
            deepinfer: Some(engine),
            ..fold_parallel(30, CutHandoff::V3)
        };
        let cfgs = build_configs(&model_cfg(), &fold, &uniform()).unwrap();
        assert_eq!(cfgs.window, 140);
        for stage in &cfgs.stages {
            for body in stage.bodies.iter().chain(&stage.layer0) {
                let mega = body
                    .mega
                    .as_ref()
                    .expect("every DeepInfer body runs the mega MoE");
                assert_eq!((mega.ep_size, mega.num_physical_experts), (1, 384));
                assert_eq!(body.attention.gemm_backends, vec!["deepgemm_mxfp8"]);
                assert!(
                    body.attention.deepinfer.is_some(),
                    "DeepInfer attention kinds"
                );
            }
        }
        assert!(cfgs.head.mhc_post_backends.is_some());
        // Each stage's back segment is its own forward: its first candidate
        // consumer builds the sparse schedule (24 on GPU0, 32 on GPU1).
        let metadata: Vec<Vec<u32>> = cfgs
            .stages
            .iter()
            .flat_map(|stage| &stage.bodies)
            .filter(|body| {
                body.attention
                    .deepinfer
                    .as_ref()
                    .unwrap()
                    .candidate_metadata
            })
            .map(|body| body.layers.clone())
            .collect();
        assert_eq!(metadata, [vec![24], vec![32]]);
        // `kv_swa_full_index` is built by layer 0 and again at the YOCO cut.
        let swa_index: Vec<Vec<u32>> = cfgs
            .stages
            .iter()
            .flat_map(|stage| stage.layer0.iter().chain(&stage.bodies))
            .filter(|body| body.attention.deepinfer.as_ref().unwrap().builds_swa_index)
            .map(|body| body.layers.clone())
            .collect();
        assert_eq!(swa_index, [vec![0], vec![20]]);
        assert_eq!(
            cfgs.stages[0]
                .cut_body
                .map(|b| cfgs.stages[0].bodies[b].layers.clone()),
            Some(vec![20])
        );
        assert_eq!(cfgs.stages[1].cut_body, None);
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let model = build("pp".into(), resolve_configs(&cfgs), &bridge).unwrap();
        assert_eq!(model.back_pass_window(), Some(140));
        let manifest = format!("{:?}", model.stages()[0].cost_log_manifest());
        assert!(manifest.contains("mega_moe"), "no mega_moe leaf on stage 0");
        // DeepInfer's host time per step: one leaf on every stage.
        let host = DeepseekV41VllmPpParallel {
            deepinfer: Some(DeepInferEngine {
                host_step_us: 6490,
                step_gap_us: 0,
                host_first_stage_only: false,
                max_num_seqs: 256,
                ..engine
            }),
            ..fold_parallel(30, CutHandoff::V3)
        };
        let cfgs = build_configs(&model_cfg(), &host, &uniform()).unwrap();
        let hosted = build("pp".into(), resolve_configs(&cfgs), &bridge).unwrap();
        for stage in hosted.stages() {
            assert_eq!(stage.host_step_ms, 6.49);
            let manifest = format!("{:?}", stage.cost_log_manifest());
            assert_eq!(
                manifest.matches(".host_step\"").count(),
                1,
                "one host_step leaf per stage"
            );
        }
        // The lower bound: host step and in-step gaps on stage 0 alone.
        let first = DeepseekV41VllmPpParallel {
            deepinfer: Some(DeepInferEngine {
                host_step_us: 6490,
                step_gap_us: 1570,
                host_first_stage_only: true,
                ..engine
            }),
            ..fold_parallel(30, CutHandoff::V3)
        };
        let cfgs = build_configs(&model_cfg(), &first, &uniform()).unwrap();
        let first = build("pp".into(), resolve_configs(&cfgs), &bridge).unwrap();
        let fixed: Vec<(f32, f32)> = first
            .stages()
            .iter()
            .map(|stage| (stage.host_step_ms, stage.step_gap_ms))
            .collect();
        assert_eq!(fixed, [(6.49, 1.57), (0.0, 0.0)]);
        let manifest = format!("{:?}", first.stages()[0].cost_log_manifest());
        assert_eq!(manifest.matches(".step_gaps\"").count(), 1);
        // DeepInfer's cache per stage: the KV sources a stage's layers read
        // (2, 8, 20 on GPU0; 14 and copies of 8 and 20 on GPU1): two ratio-2
        // and one ratio-1 source, 178 + 178 + 356 B per token each; GPU1 also
        // holds the MTP layers' windows.
        let stage_bytes: Vec<u64> = cfgs
            .stages
            .iter()
            .map(|stage| stage.kv_cache_layout.per_token_bytes())
            .collect();
        assert_eq!(stage_bytes, [712, 712]);
        assert!(
            cfgs.stages[1].kv_cache_layout.engine.fixed_device_bytes
                > cfgs.stages[0].kv_cache_layout.engine.fixed_device_bytes
        );
        // Redundant slots have no meaning when one GPU holds every expert.
        let redundant = DeepseekV41VllmPpParallel {
            deepinfer: Some(DeepInferEngine {
                num_redundant_experts: 32,
                ..engine
            }),
            ..fold_parallel(30, CutHandoff::V3)
        };
        assert!(build_configs(&model_cfg(), &redundant, &uniform()).is_err());
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
        // `DSV41_PP_SPLITS` entries are partitions (`8,6,6,20`) or folds by
        // their back cut (`fold:30` = GPU0 0-9 + 20-29, GPU1 10-19 + 30-39, V3).
        let splits = std::env::var("DSV41_PP_SPLITS").unwrap_or_else(|_| "8,6,6,20".into());
        for split in splits.split(';') {
            let (cfgs, _partition) = match split.strip_prefix("fold:") {
                Some(back) => (
                    build_configs(
                        &model,
                        &fold_parallel(back.parse().unwrap(), CutHandoff::V3),
                        &uniform(),
                    )
                    .unwrap(),
                    vec![20, 20],
                ),
                None => {
                    let partition: Vec<u32> = split
                        .split(',')
                        .map(|v| v.trim().parse().unwrap())
                        .collect();
                    (configs(&partition), partition)
                }
            };
            for (index, stage) in cfgs.stages.iter().enumerate() {
                let held: Vec<u32> = stage.layers.iter().flat_map(|&(s, e)| s..e).collect();
                let own = crate::arch::deepseek_v41_vllm::stage_kv_cache_layout(&model, &held, &[]);
                let send = stage.send.as_ref().map(|send| {
                    serde_json::json!({
                        "front_extra_row_bytes": send.front_extra_row_bytes,
                        "late_extra_row_bytes": send.late_extra_row_bytes,
                        "kv_entries": send.kv_entries,
                    })
                });
                let line = serde_json::json!({
                    "split": split,
                    "stage": index,
                    "layers": stage.layers,
                    "copied_sources": stage.copied_sources,
                    "per_token_bytes": stage.kv_cache_layout.per_token_bytes(),
                    "own_per_token_bytes": own.per_token_bytes(),
                    "fixed_device_bytes": stage.kv_cache_layout.engine.fixed_device_bytes,
                    "layout": stage.kv_cache_layout,
                    "send": send,
                    "fork_accepts": cfgs.fork_accepts,
                });
                println!("STAGE_LAYOUT {line}");
            }
        }
    }
}
