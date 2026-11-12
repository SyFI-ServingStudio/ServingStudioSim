//! DeepSeek-V4.1-Flash under vLLM data-parallel attention with expert-parallel
//! MoE on B200 (`--data-parallel-size 4 --enable-expert-parallel`, TP1;
//! DP4/EP4 capture `logs/20261106_0_dsv41_dspark_capture/profile_dp4ep4_*`,
//! `phase1_split.md` §1.3, §3, §6.2).
//!
//! Every DP rank is its own engine core with its own scheduler and KV cache,
//! so a request lives on one rank. Each rank runs the prologue (embedding,
//! Engram hash), all 64 attention heads, the Engram blocks and lookups, the
//! whole shared expert and the full-vocabulary lm_head on its own tokens at
//! TP1 — the TP arch's leaves ([`deepseek_v41_vllm`]) at TP1/EP1 shapes, with
//! no TP all-reduce and no logits gather. Only the routed experts are split,
//! 384 / 4 per rank, behind an MXFP8 all-gatherv and a bf16 reduce-scatterv
//! ([`DeepseekV41MoeFfnDpEpWorklet`]).
//!
//! Ranks synchronise only at those two collectives, so each segment between
//! them is a `Max` over ranks of that rank's whole segment, each rank costed
//! with its own requests and KV:
//!
//! ```text
//! Max_r [prologue, layer-0 attention, layer-0 FFN up to the dispatch, Engram lookups]
//! layer-0 exchange (all_gatherv -> experts -> reduce_scatterv)
//! for layer in 1..40:
//!     Max_r [previous add, Engram block?, attention, FFN up to the dispatch]
//!     exchange
//! Max_r [last add, head]
//! ```
//!
//! (The previous layer's shared + routed add is its own `Max_r` right after the
//! exchange, a slight over-estimate of the max-of-sums when ranks differ.)
//!
//! Step shape ([`normalize_dp_steps`], `v1/worker/gpu/dp_utils.py`
//! `sync_cudagraph_and_dp_padding`): each rank's batch is a TP-arch
//! [`DeepseekV41Step`]; a rank with nothing scheduled runs vLLM's one-token
//! dummy batch so the collectives complete. If any rank must run eager, every
//! rank runs eager on its own rows and the collectives are ragged; a FULL
//! decode graph only when every rank is a uniform decode, padded to the
//! busiest rank's FULL request count; otherwise every rank pads to the
//! busiest rank's PIECEWISE size, so the collectives move `4 x padded` rows;
//! the routed experts see only the scheduled ones (padding rows route to id -1,
//! an idle rank's dummy batch is all padding). FULL
//! graphs keep the attention aux stream and, at <= 16 rows, the mHC overlap
//! path, evaluated per rank on that rank's rows. Bounded replay
//! ([`BoundedReplay`]) runs on every rank when it runs on any, each rank's
//! replay batch padded to the busiest one's replay graph.
//!
//! Layer folding is the TP arch's ([`fold_layers`]): equal layer kinds share
//! one body under `Scale{n}`; each body is one whole layer's DP segments.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;

use crate::arch::config::ModelSpec;
use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::deepseek_v41_vllm::{
    self as tp_arch, fold_layers, index_plan, layer_config, layer_input, prefill_pairs,
    replay_rows, BoundedReplay, DeepseekV41CudaGraphs, DeepseekV41LayerConfig, DeepseekV41ModelCfg,
    DeepseekV41ReplayConfig, DeepseekV41Step, DeepseekV41VllmParallel, GraphMode, LayerInput,
    LayerKind, PlanNode,
};
use crate::arch::kv_layout::KvCacheLayout;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::deepseek_v41_common::eval_or_zero;
use crate::worklet::{
    DeepseekV41AttentionEntry, DeepseekV41AttentionTpWorklet, DeepseekV41AttentionTpWorkletConfig,
    DeepseekV41AttentionTpWorkletInput, DeepseekV41AttentionTpWorkletResolved,
    DeepseekV41EngramPrefetchLocalWorklet, DeepseekV41EngramPrefetchLocalWorkletConfig,
    DeepseekV41EngramPrefetchLocalWorkletInput, DeepseekV41EngramPrefetchLocalWorkletResolved,
    DeepseekV41EngramTpWorklet, DeepseekV41EngramTpWorkletConfig, DeepseekV41EngramTpWorkletInput,
    DeepseekV41EngramTpWorkletResolved, DeepseekV41HeadTpWorklet, DeepseekV41HeadTpWorkletConfig,
    DeepseekV41HeadTpWorkletInput, DeepseekV41HeadTpWorkletResolved, DeepseekV41MoeFfnDpEpWorklet,
    DeepseekV41MoeFfnDpEpWorkletConfig, DeepseekV41MoeFfnDpEpWorkletInput,
    DeepseekV41MoeFfnDpEpWorkletResolved, DeepseekV41MoeFfnDpRankInput,
    DeepseekV41PrologueTpWorklet, DeepseekV41PrologueTpWorkletConfig,
    DeepseekV41PrologueTpWorkletInput, DeepseekV41PrologueTpWorkletResolved,
};

const ARCH_KIND: &str = "deepseek_v41_vllm_dp_attn_ep_moe";
/// DP ranks, which are also the EP group (the captured deployment; the
/// selector takes no size, as the TP arch takes none).
pub(crate) const DP_SIZE: u32 = 4;
const COLLECTIVE_BACKENDS: &[&str] = &["vllm_pynccl"];
/// Rows of the whole Engram table at TP1 (DP4 server log `engram.py:1134`,
/// layer 1: "384006168 rows x 256"): the lookup kind's table identity.
const ENGRAM_TABLE_ROWS_TP1: u64 = 384_006_168;

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

#[derive(Clone, Debug)]
pub struct DeepseekV41VllmDpParallel {
    pub gpu_name: String,
    /// Resolved `--max-model-len` ([`tp_arch::resolve_max_model_len`]).
    pub max_model_len: u32,
    /// vLLM's decoder SWA bounded replay ([`BoundedReplay`], on by default).
    pub decoder_swa_bounded_replay: bool,
    /// `--max-num-batched-tokens` per rank (the chunk size).
    pub max_num_batched_tokens: u32,
    /// `--cudagraph-capture-sizes` per rank (PIECEWISE token sizes; empty =
    /// eager), as for the TP arch ([`DeepseekV41CudaGraphs`]).
    pub cudagraph_capture_sizes: Vec<u32>,
}

/// One body layer's configs: the TP arch's at TP1, with the DP FFN.
#[derive(Clone)]
pub struct DeepseekV41DpLayerConfig {
    pub layers: Vec<u32>,
    pub engram: Option<DeepseekV41EngramTpWorkletConfig>,
    pub attention: DeepseekV41AttentionTpWorkletConfig,
    pub ffn: DeepseekV41MoeFfnDpEpWorkletConfig,
}

pub struct DeepseekV41VllmDpConfigs {
    pub prologue: DeepseekV41PrologueTpWorkletConfig,
    pub engram_prefetch: DeepseekV41EngramPrefetchLocalWorkletConfig,
    pub layer0: DeepseekV41DpLayerConfig,
    /// Bodies of layers 1..40, in first-visit order of `plan`.
    pub bodies: Vec<DeepseekV41DpLayerConfig>,
    plan: Vec<PlanNode>,
    pub head: DeepseekV41HeadTpWorkletConfig,
    /// One DP rank's cache: the TP arch's layout at TP1.
    pub kv_cache_layout: KvCacheLayout,
    /// `Some` with `decoder_swa_bounded_replay` on: the TP arch's boundary,
    /// run by every rank.
    pub replay: Option<DeepseekV41ReplayConfig>,
    /// Per body: evaluated on the replay batch.
    pub late_bodies: Vec<bool>,
    /// Each rank's captured graphs, decode query length 1 (no speculation).
    pub graphs: DeepseekV41CudaGraphs,
}

pub fn build_configs(
    model: &DeepseekV41ModelCfg,
    parallel: &DeepseekV41VllmDpParallel,
    demand: &ExpertDemand,
) -> std::result::Result<DeepseekV41VllmDpConfigs, BuildError> {
    if !(1..=model.max_position_embeddings).contains(&parallel.max_model_len) {
        return Err(fit_failed(format!(
            "max_model_len must be 1..={} (max_position_embeddings), got {}",
            model.max_position_embeddings, parallel.max_model_len
        )));
    }
    if model.num_experts % DP_SIZE != 0 {
        return Err(fit_failed(format!(
            "{} routed experts must split over {DP_SIZE} EP ranks",
            model.num_experts
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
    // The rank-local recipe: the TP arch's per-layer configs at TP1, with the
    // routed experts split over the DP group.
    let local = DeepseekV41VllmParallel {
        tp_size: 1,
        ep_size: DP_SIZE,
        gpu_name: parallel.gpu_name.clone(),
        serialize_streams: false,
        decoder_swa_bounded_replay: false,
        max_model_len: parallel.max_model_len,
        max_num_batched_tokens: parallel.max_num_batched_tokens,
        cudagraph_capture_sizes: parallel.cudagraph_capture_sizes.clone(),
    };
    let graphs = DeepseekV41CudaGraphs::new(
        &parallel.cudagraph_capture_sizes,
        tp_arch::MAX_NUM_SEQS,
        &[1],
    );
    // A rank's rows: its scheduled tokens, or the graph it pads to.
    let max_rank_tokens = parallel.max_num_batched_tokens.max(graphs.max_tokens());
    let dp_layer = |kind: LayerKind, layers: Vec<u32>| {
        dp_layer_config(
            layer_config(model, &local, demand, kind, layers),
            max_rank_tokens,
        )
    };
    let tail: Vec<(u32, LayerKind)> = (1..model.num_layers)
        .map(|layer| (layer, model.layer_kind(layer)))
        .collect();
    let mut bodies = Vec::new();
    let plan = index_plan(&fold_layers(&tail), &mut |kind, layers| {
        bodies.push(dp_layer(kind, layers.to_vec()));
        bodies.len() - 1
    });
    let gpu = parallel.gpu_name.clone();
    let (replay, late_bodies) = if parallel.decoder_swa_bounded_replay {
        let rule = BoundedReplay::from_model(model)?;
        let late_bodies = bodies
            .iter()
            .map(|body: &DeepseekV41DpLayerConfig| rule.is_late(&body.layers))
            .collect::<std::result::Result<Vec<_>, _>>()?;
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
        (
            Some(DeepseekV41ReplayConfig::new(
                rule, cut_body, write_kv, model, &gpu,
            )),
            late_bodies,
        )
    } else {
        (None, vec![false; bodies.len()])
    };
    Ok(DeepseekV41VllmDpConfigs {
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
        layer0: dp_layer(layer0_kind, vec![0]),
        bodies,
        plan,
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
        kv_cache_layout: tp_arch::kv_cache_layout(model, 1, 0),
        replay,
        late_bodies,
        graphs,
    })
}

/// `max_rank_tokens`: the most rows one rank sends to the all-gatherv (its
/// token budget, or its largest graph when padded past it).
pub(crate) fn dp_layer_config(
    tp1: DeepseekV41LayerConfig,
    max_rank_tokens: u32,
) -> DeepseekV41DpLayerConfig {
    DeepseekV41DpLayerConfig {
        layers: tp1.layers,
        engram: tp1.engram,
        attention: tp1.attention,
        ffn: DeepseekV41MoeFfnDpEpWorkletConfig {
            ffn: tp1.ffn,
            collective_backends: COLLECTIVE_BACKENDS.to_vec(),
            max_rank_tokens,
        },
    }
}

pub struct DeepseekV41DpLayerResolved {
    pub(crate) layers: Vec<u32>,
    pub(crate) engram: Option<DeepseekV41EngramTpWorkletResolved>,
    pub(crate) attention: DeepseekV41AttentionTpWorkletResolved,
    pub(crate) ffn: DeepseekV41MoeFfnDpEpWorkletResolved,
}

pub struct DeepseekV41VllmDpResolved {
    prologue: DeepseekV41PrologueTpWorkletResolved,
    engram_prefetch: DeepseekV41EngramPrefetchLocalWorkletResolved,
    layer0: DeepseekV41DpLayerResolved,
    bodies: Vec<DeepseekV41DpLayerResolved>,
    plan: Vec<PlanNode>,
    head: DeepseekV41HeadTpWorkletResolved,
    kv_cache_layout: KvCacheLayout,
    max_model_len: u32,
    replay: Option<DpReplayResolved>,
    late_bodies: Vec<bool>,
    graphs: DeepseekV41CudaGraphs,
}

pub struct DpReplayResolved {
    rule: BoundedReplay,
    cut_body: usize,
    write_kv: DeepseekV41AttentionTpWorkletResolved,
    gather: ElementwiseKernelConfig,
    scatter: ElementwiseKernelConfig,
}

pub(crate) fn resolve_layer(config: &DeepseekV41DpLayerConfig) -> DeepseekV41DpLayerResolved {
    DeepseekV41DpLayerResolved {
        layers: config.layers.clone(),
        engram: config
            .engram
            .as_ref()
            .map(DeepseekV41EngramTpWorklet::resolve_config),
        attention: DeepseekV41AttentionTpWorklet::resolve_config(&config.attention),
        ffn: DeepseekV41MoeFfnDpEpWorklet::resolve_config(&config.ffn),
    }
}

pub fn resolve_configs(configs: &DeepseekV41VllmDpConfigs) -> DeepseekV41VllmDpResolved {
    DeepseekV41VllmDpResolved {
        prologue: DeepseekV41PrologueTpWorklet::resolve_config(&configs.prologue),
        engram_prefetch: DeepseekV41EngramPrefetchLocalWorklet::resolve_config(
            &configs.engram_prefetch,
        ),
        layer0: resolve_layer(&configs.layer0),
        bodies: configs.bodies.iter().map(resolve_layer).collect(),
        plan: configs.plan.clone(),
        head: DeepseekV41HeadTpWorklet::resolve_config(&configs.head),
        kv_cache_layout: configs.kv_cache_layout.clone(),
        max_model_len: configs.layer0.attention.max_model_len,
        replay: configs.replay.as_ref().map(|replay| DpReplayResolved {
            rule: replay.rule,
            cut_body: replay.cut_body,
            write_kv: DeepseekV41AttentionTpWorklet::resolve_config(&replay.write_kv),
            gather: replay.gather.clone(),
            scatter: replay.scatter.clone(),
        }),
        late_bodies: configs.late_bodies.clone(),
        graphs: configs.graphs.clone(),
    }
}

/// One built body layer: optional Engram block, TP1 attention, DP MoE FFN.
pub(crate) struct DpLayerBody {
    name: String,
    layers: Vec<u32>,
    engram: Option<DeepseekV41EngramTpWorklet>,
    attention: DeepseekV41AttentionTpWorklet,
    ffn: DeepseekV41MoeFfnDpEpWorklet,
}

impl DpLayerBody {
    pub(crate) fn build(
        model_name: &str,
        resolved: DeepseekV41DpLayerResolved,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        let name = format!("{model_name}.layer{}", resolved.layers[0]);
        Ok(Self {
            engram: match resolved.engram {
                Some(engram) => Some(DeepseekV41EngramTpWorklet::build(
                    format!("{name}.engram"),
                    engram,
                    bridge,
                )?),
                None => None,
            },
            attention: DeepseekV41AttentionTpWorklet::build(
                format!("{name}.attn"),
                resolved.attention,
                bridge,
            )?,
            ffn: DeepseekV41MoeFfnDpEpWorklet::build(format!("{name}.ffn"), resolved.ffn, bridge)?,
            layers: resolved.layers,
            name,
        })
    }

    /// One rank's segment before the dispatch.
    fn compile_rank(&self, b: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut leaves = Vec::new();
        if let Some(engram) = &self.engram {
            leaves.push(engram.compile(b));
        }
        leaves.push(self.attention.compile(b));
        leaves.extend(self.ffn.compile_local(b));
        leaves
    }

    /// The exchange and every rank's post-combine add.
    fn compile_tail(&self, b: &mut CostTreeBuilder) -> Vec<CostNode> {
        let ranks = DP_SIZE as usize;
        vec![
            self.ffn.compile_exchange(ranks, b),
            labeled_max(
                format!("{}.ffn.residual_add [Max over DP ranks]", self.name),
                (0..ranks).map(|_| self.ffn.compile_post(b)).collect(),
            ),
        ]
    }

    /// The whole layer; at the replay cut each rank's segment opens with the
    /// boundary's `write_kv` and gather.
    pub(crate) fn compile(
        &self,
        boundary: Option<&DpReplayBoundary>,
        b: &mut CostTreeBuilder,
    ) -> CostNode {
        let ranks = (0..DP_SIZE)
            .map(|_| {
                let mut leaves = Vec::new();
                if let Some(boundary) = boundary {
                    leaves.push(boundary.write_kv.compile(b));
                    leaves.push(boundary.gather.compile(b));
                }
                leaves.extend(self.compile_rank(b));
                CostNode::Sum(leaves)
            })
            .collect();
        let mut children = vec![labeled_max(
            format!(
                "{} [Max over DP ranks: {}Engram? -> attention -> FFN up to the dispatch]",
                self.name,
                if boundary.is_some() {
                    "replay write_kv -> gather -> "
                } else {
                    ""
                },
            ),
            ranks,
        )];
        children.extend(self.compile_tail(b));
        CostNode::Labeled {
            label: format!("{} [layers {:?}]", self.name, self.layers),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    fn eval_rank(&self, rank: &LayerInput, ev: &mut Evaluator) {
        if let Some(engram) = &self.engram {
            engram.eval(
                &DeepseekV41EngramTpWorkletInput {
                    num_tokens: rank.rows,
                },
                ev,
            );
        }
        self.attention.eval(&rank.attention, ev);
        self.ffn.eval_local(&ffn_rank_input(rank), ev);
    }

    fn eval_tail(&self, input: &DpLayerInput, ev: &mut Evaluator) {
        self.ffn.eval_exchange(&input.exchange, ev);
        for rank in &input.exchange.ranks {
            self.ffn.eval_post(rank, ev);
        }
    }

    /// `input` is the batch this body runs (the replay batch for a late
    /// body); `boundary` is set on the cut body.
    pub(crate) fn eval(
        &self,
        input: &DpLayerInput,
        boundary: Option<(&DpReplayBoundary, &DpStep)>,
        ev: &mut Evaluator,
    ) {
        for (rank, layer) in input.ranks.iter().enumerate() {
            if let Some((boundary, step)) = boundary {
                boundary.eval_rank(step, rank, ev);
            }
            self.eval_rank(layer, ev);
        }
        self.eval_tail(input, ev);
    }
}

fn ffn_rank_input(rank: &LayerInput) -> DeepseekV41MoeFfnDpRankInput {
    DeepseekV41MoeFfnDpRankInput {
        num_tokens: rank.rows,
        scheduled_tokens: rank.scheduled_rows,
        mhc_overlap: rank.attention.mhc_overlap,
    }
}

pub(crate) fn labeled_max(label: String, children: Vec<CostNode>) -> CostNode {
    CostNode::Labeled {
        label,
        child: Box::new(CostNode::Max {
            overlap: 1.0,
            children,
        }),
    }
}

/// The built bounded-replay boundary, run by every rank on its own rows.
pub(crate) struct DpReplayBoundary {
    cut_body: usize,
    write_kv: DeepseekV41AttentionTpWorklet,
    gather: Op<ElementwiseKernel>,
    scatter: Op<ElementwiseKernel>,
}

impl DpReplayBoundary {
    /// The cut layer's `write_kv` over the rank's whole batch and the gather
    /// of its replay rows; both zero on a step that does not replay.
    fn eval_rank(&self, step: &DpStep, rank: usize, ev: &mut Evaluator) {
        let write_kv = match &step.replay {
            Some(_) => DeepseekV41AttentionTpWorkletInput {
                write_kv_only: true,
                ..step.full.ranks[rank].attention.clone()
            },
            None => DeepseekV41AttentionTpWorkletInput {
                num_tokens: 0,
                prefill_query_context_pairs: Vec::new(),
                decode_kv_lens: Vec::new(),
                aux_stream_live: false,
                full_graph: false,
                mhc_overlap: false,
                write_kv_only: true,
            },
        };
        self.write_kv.eval(&write_kv, ev);
        let gathered = step.replay.as_ref().map_or(0, |late| late.ranks[rank].rows);
        eval_or_zero(
            &self.gather,
            ElementwiseKernelInput {
                num_tokens: gathered,
            },
            gathered == 0,
            ev,
        );
    }
}

pub struct DeepseekV41VllmDpModel {
    name: String,
    prologue: DeepseekV41PrologueTpWorklet,
    engram_prefetch: DeepseekV41EngramPrefetchLocalWorklet,
    layer0: DpLayerBody,
    bodies: Vec<DpLayerBody>,
    plan: Vec<PlanNode>,
    head: DeepseekV41HeadTpWorklet,
    max_model_len: u32,
    kv_cache_layout: KvCacheLayout,
    replay_rule: Option<BoundedReplay>,
    replay: Option<DpReplayBoundary>,
    late_bodies: Vec<bool>,
    graphs: DeepseekV41CudaGraphs,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: DeepseekV41VllmDpResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<DeepseekV41VllmDpModel, BuildError> {
    let bodies = resolved
        .bodies
        .into_iter()
        .map(|body| DpLayerBody::build(&name, body, bridge))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let replay_rule = resolved.replay.as_ref().map(|replay| replay.rule);
    let replay = match resolved.replay {
        Some(replay) => {
            let cut = format!("{name}.layer{}", replay.rule.cut_layer);
            let elementwise = |leaf: String, config| -> std::result::Result<_, BuildError> {
                Ok(Op::new(
                    leaf.clone(),
                    Arc::new(ElementwiseKernel::build(leaf, config, bridge)?),
                ))
            };
            Some(DpReplayBoundary {
                cut_body: replay.cut_body,
                write_kv: DeepseekV41AttentionTpWorklet::build(
                    format!("{cut}.write_kv"),
                    replay.write_kv,
                    bridge,
                )?,
                gather: elementwise(format!("{name}.replay.gather"), replay.gather)?,
                scatter: elementwise(format!("{name}.replay.scatter"), replay.scatter)?,
            })
        }
        None => None,
    };
    let mut model = DeepseekV41VllmDpModel {
        prologue: DeepseekV41PrologueTpWorklet::build(
            format!("{name}.prologue"),
            resolved.prologue,
            bridge,
        )?,
        engram_prefetch: DeepseekV41EngramPrefetchLocalWorklet::build(
            format!("{name}.engram_prefetch"),
            resolved.engram_prefetch,
            bridge,
        )?,
        layer0: DpLayerBody::build(&name, resolved.layer0, bridge)?,
        bodies,
        plan: resolved.plan,
        head: DeepseekV41HeadTpWorklet::build(format!("{name}.head"), resolved.head, bridge)?,
        max_model_len: resolved.max_model_len,
        kv_cache_layout: resolved.kv_cache_layout,
        replay_rule,
        replay,
        late_bodies: resolved.late_bodies,
        graphs: resolved.graphs,
        cost_flat: Vec::new(),
        n_slots: 0,
        name,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

impl DeepseekV41VllmDpModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let sections = self.compile_sections(&mut b);
        let root = CostNode::Labeled {
            label: format!(
                "{} (DeepseekV41VllmDpModel) [DP{DP_SIZE} attention at TP1, EP{DP_SIZE} experts \
                 behind MXFP8 all_gatherv / bf16 reduce_scatterv; {} unique bodies over 40 \
                 layers; bounded_replay={}]",
                self.name,
                self.bodies.len() + 1,
                self.replay.is_some(),
            ),
            child: Box::new(CostNode::Sum(sections)),
        };
        b.finish(root)
    }

    /// The target forward, prologue to head, in evaluation order
    /// ([`Self::eval_step`]); a wrapper (the DSpark drafter) sums these with
    /// its own sections.
    pub(crate) fn compile_sections(&self, b: &mut CostTreeBuilder) -> Vec<CostNode> {
        // Layer 0's rank segment opens with the prologue and closes with the
        // two Engram lookups, which contend with it on the same device
        // (`compile_joined`) and finish before the first dispatch.
        let layer0_ranks = (0..DP_SIZE)
            .map(|_| {
                let mut main = vec![self.prologue.compile(b)];
                main.extend(self.layer0.compile_rank(b));
                self.engram_prefetch.compile_joined(b, CostNode::Sum(main))
            })
            .collect();
        let mut layer0 = vec![labeled_max(
            format!(
                "{} [Max over DP ranks: prologue -> layer-0 attention -> FFN up to the \
                 dispatch, Engram lookups]",
                self.layer0.name
            ),
            layer0_ranks,
        )];
        layer0.extend(self.layer0.compile_tail(b));
        let mut sections = vec![CostNode::Labeled {
            label: format!("{} [layers [0]]", self.layer0.name),
            child: Box::new(CostNode::Sum(layer0)),
        }];
        sections.extend(self.compile_plan(&self.plan, b));
        let heads = (0..DP_SIZE)
            .map(|_| {
                let mut leaves = Vec::new();
                if let Some(replay) = &self.replay {
                    leaves.push(replay.scatter.compile(b));
                }
                leaves.push(self.head.compile(b));
                CostNode::Sum(leaves)
            })
            .collect();
        sections.push(labeled_max(
            format!(
                "{}.head [Max over DP ranks: {}head, full vocabulary]",
                self.name,
                if self.replay.is_some() {
                    "replay scatter -> "
                } else {
                    ""
                }
            ),
            heads,
        ));
        sections
    }

    pub fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    pub fn replay_rule(&self) -> Option<&BoundedReplay> {
        self.replay_rule.as_ref()
    }

    /// Each rank's captured graphs (decode query length 1); a speculative
    /// wrapper builds its own for its verify width.
    pub fn graphs(&self) -> &DeepseekV41CudaGraphs {
        &self.graphs
    }

    fn boundary_at(&self, index: usize) -> Option<&DpReplayBoundary> {
        self.replay.as_ref().filter(|r| r.cut_body == index)
    }

    fn compile_plan(&self, plan: &[PlanNode], b: &mut CostTreeBuilder) -> Vec<CostNode> {
        plan.iter()
            .map(|node| match node {
                PlanNode::Body(index) => self.bodies[*index].compile(self.boundary_at(*index), b),
                PlanNode::Repeat { n, body } => {
                    let children = self.compile_plan(body, b);
                    CostNode::Scale {
                        n: *n,
                        child: Box::new(if children.len() == 1 {
                            children.into_iter().next().unwrap()
                        } else {
                            CostNode::Sum(children)
                        }),
                    }
                }
            })
            .collect()
    }

    fn eval_plan(&self, plan: &[PlanNode], step: &DpStep, ev: &mut Evaluator) {
        for node in plan {
            match node {
                PlanNode::Body(index) => {
                    let input = match &step.replay {
                        Some(late) if self.late_bodies[*index] => late,
                        _ => &step.full,
                    };
                    let boundary = self.boundary_at(*index).map(|boundary| (boundary, step));
                    self.bodies[*index].eval(input, boundary, ev)
                }
                PlanNode::Repeat { body, .. } => self.eval_plan(body, step, ev),
            }
        }
    }

    pub(crate) fn eval_step(&self, step: &DpStep, ev: &mut Evaluator) {
        for rank in &step.full.ranks {
            let tokens = rank.rows;
            self.prologue.eval(
                &DeepseekV41PrologueTpWorkletInput { num_tokens: tokens },
                ev,
            );
            self.layer0.eval_rank(rank, ev);
            self.engram_prefetch.eval(
                &DeepseekV41EngramPrefetchLocalWorkletInput { num_tokens: tokens },
                ev,
            );
        }
        self.layer0.eval_tail(&step.full, ev);
        self.eval_plan(&self.plan, step, ev);
        for (index, rank) in step.full.ranks.iter().enumerate() {
            let late_rows = step.replay.as_ref().map(|late| late.ranks[index].rows);
            if let Some(replay) = &self.replay {
                let scattered = if late_rows.is_some() { rank.rows } else { 0 };
                eval_or_zero(
                    &replay.scatter,
                    ElementwiseKernelInput {
                        num_tokens: scattered,
                    },
                    scattered == 0,
                    ev,
                );
            }
            self.head.eval(
                &DeepseekV41HeadTpWorkletInput {
                    num_tokens: rank.rows,
                    mhc_post_rows: late_rows.unwrap_or(rank.rows),
                    logits_rows: step.logits_rows[index],
                },
                ev,
            );
        }
    }

    fn normalize(&self, batch: &UnifiedArchInput) -> std::result::Result<DpStep, String> {
        normalize_input(
            &self.graphs,
            batch,
            self.max_model_len,
            self.replay_rule.as_ref(),
        )
    }

    fn eval_checked(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: Option<&mut Vec<SlotInput>>,
    ) -> LeafMetrics {
        let step = self
            .normalize(batch)
            .unwrap_or_else(|reason| panic!("invalid DeepSeek-V4.1 DP4 input: {reason}"));
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut evaluator = match inputs {
            Some(inputs) => Evaluator::with_inputs(slots, inputs),
            None => Evaluator::new(slots),
        };
        self.eval_step(&step, &mut evaluator);
        assert_eq!(
            evaluator.filled(),
            self.n_slots,
            "eval must fill every compiled slot"
        );
        drop(evaluator);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }
}

impl IterwiseUnifiedModel for DeepseekV41VllmDpModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        self.normalize(batch).map(|_| ())
    }

    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        self.eval_checked(batch, slots, scratch, None)
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        self.eval_checked(batch, slots, scratch, Some(inputs))
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    /// One DP rank's cache for the requests it owns: the TP arch's packed
    /// groups at TP1 (nothing replicated).
    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout.clone()
    }

    fn gpus_per_replica(&self) -> u16 {
        DP_SIZE as u16
    }

    /// One KV partition and one scheduler per engine core.
    fn num_attn_dp_groups(&self) -> u16 {
        DP_SIZE as u16
    }

    /// Each partition is one GPU's cache.
    fn num_attn_shards(&self) -> u16 {
        1
    }
}

/// One batch (the whole step, or the replay batch) on every rank, and the
/// collectives' per-rank inputs.
#[derive(Clone, Debug)]
pub(crate) struct DpLayerInput {
    pub ranks: Vec<LayerInput>,
    pub exchange: DeepseekV41MoeFfnDpEpWorkletInput,
}

impl DpLayerInput {
    pub(crate) fn new(ranks: Vec<LayerInput>) -> Self {
        let ffn: Vec<_> = ranks.iter().map(ffn_rank_input).collect();
        Self {
            exchange: DeepseekV41MoeFfnDpEpWorkletInput {
                collective_tokens: ffn.iter().map(|rank| rank.num_tokens).collect(),
                ranks: ffn,
            },
            ranks,
        }
    }
}

/// A DP step, normalized: the graph mode every rank agreed on, the batch
/// before the replay cut (the whole forward when nothing replays), the replay
/// batch, and each rank's logits rows.
#[derive(Clone, Debug)]
pub(crate) struct DpStep {
    /// For a wrapper (the DSpark drafter) that follows the target's graph.
    #[allow(dead_code)]
    pub mode: GraphMode,
    pub full: DpLayerInput,
    pub replay: Option<DpLayerInput>,
    pub logits_rows: Vec<u32>,
}

/// One rank's no-spec step: each decode request is one row after its
/// resident KV.
fn rank_step(
    rank: usize,
    group: &ArchGroupInput,
    max_model_len: u32,
) -> std::result::Result<DeepseekV41Step, String> {
    let prefill = prefill_pairs(&group.prefill_chunk_pairs, max_model_len)
        .map_err(|reason| format!("rank {rank}: {reason}"))?;
    let prefill_tokens: u32 = prefill.iter().map(|&(q, _)| q).sum();
    let decode = group
        .decode_kv_lens
        .iter()
        .enumerate()
        .map(|(request, &kv)| {
            if (1..max_model_len).contains(&kv) {
                Ok((1, kv + 1))
            } else {
                Err(format!(
                    "rank {rank} decode request {request} resident KV {kv} must be in \
                     1..{max_model_len}"
                ))
            }
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let decode_tokens = decode.len() as u32;
    if group.prefill_tokens != prefill_tokens
        || group.decode_tokens != decode_tokens
        || group.batch_tokens != prefill_tokens + decode_tokens
    {
        return Err(format!("rank {rank} token accounting is inconsistent"));
    }
    Ok(DeepseekV41Step { prefill, decode })
}

fn query_lens(step: &DeepseekV41Step) -> Vec<u32> {
    step.prefill
        .iter()
        .chain(&step.decode)
        .map(|&(q, _)| q)
        .collect()
}

/// vLLM's DP step agreement over per-rank steps (`dp_utils.py`
/// `sync_cudagraph_and_dp_padding`): an idle rank joins with a dummy uniform
/// decode of one request at `uniform_query_len` rows (`execute_dummy_batch`,
/// no sampled row); if any rank must run eager
/// every rank runs eager on its own rows (ragged collectives); a FULL decode
/// graph only when every rank's step is a uniform decode, all padded to the
/// busiest rank's FULL request count; otherwise every rank pads to the
/// busiest rank's PIECEWISE token size. `uniform_query_len` is 1 without
/// speculation and the verify width with it; `logits_rows` gives each rank's
/// sampled rows.
///
/// Bounded replay: the ranks replay together (the replay batch is one more
/// graph launch around the collectives), when any rank's step replays by the
/// TP rule ([`replay_rows`]); every rank then keeps its requests' last
/// `window` rows (a dummy keeps its `uniform_query_len` rows) and pads to
/// the busiest rank's replay graph ([`DeepseekV41CudaGraphs::replay_rows`]).
pub(crate) fn normalize_dp_steps(
    graphs: &DeepseekV41CudaGraphs,
    steps: &[DeepseekV41Step],
    uniform_query_len: u32,
    logits_rows: Vec<u32>,
    replay: Option<&BoundedReplay>,
) -> DpStep {
    // `execute_dummy_batch` -> `_dummy_run(uniform_decode_query_len,
    // uniform_decode=True)`: one request of `uniform_query_len` rows, a
    // uniform decode, so it never knocks the step out of a FULL graph. Its
    // rows attend after one resident token.
    let dummy = DeepseekV41Step {
        prefill: Vec::new(),
        decode: vec![(uniform_query_len, uniform_query_len + 1)],
    };
    let steps: Vec<&DeepseekV41Step> = steps
        .iter()
        .map(|step| {
            if step.request_count() == 0 {
                &dummy
            } else {
                step
            }
        })
        .collect();
    let lens: Vec<Vec<u32>> = steps.iter().map(|step| query_lens(step)).collect();
    let modes: Vec<GraphMode> = lens
        .iter()
        .map(|lens| graphs.step_graph(lens, uniform_query_len).0)
        .collect();
    let tokens: Vec<u32> = lens.iter().map(|lens| lens.iter().sum()).collect();
    let mode = if modes.contains(&GraphMode::Eager) {
        GraphMode::Eager
    } else if modes.iter().all(|&mode| mode == GraphMode::Full) {
        GraphMode::Full
    } else {
        GraphMode::Piecewise
    };
    let busiest_tokens = tokens.iter().copied().max().unwrap_or(0);
    let rows: Vec<u32> = match mode {
        GraphMode::Eager => tokens,
        GraphMode::Full => {
            let requests = steps.iter().map(|s| s.request_count()).max().unwrap_or(1);
            let padded = graphs
                .full_decode_requests(requests, uniform_query_len)
                .expect("every rank holds a FULL graph, so the busiest one does");
            vec![padded * uniform_query_len; steps.len()]
        }
        GraphMode::Piecewise => {
            let padded = graphs
                .piecewise_rows(busiest_tokens)
                .expect("no rank runs eager, so the busiest one fits a graph");
            vec![padded; steps.len()]
        }
    };
    // MRV2 builds an idle rank's dummy batch with every row marked padding
    // (`InputBatch.make_dummy(..., is_padding=not is_profile)`): nothing it
    // runs is scheduled.
    let idle: Vec<bool> = steps.iter().map(|&step| std::ptr::eq(step, &dummy)).collect();
    let unscheduled = |mut layer: LayerInput, idle: bool| {
        if idle {
            layer.scheduled_rows = 0;
        }
        layer
    };
    let full = DpLayerInput::new(
        steps
            .iter()
            .zip(&rows)
            .zip(&idle)
            .map(|((step, &rows), &idle)| {
                unscheduled(layer_input(mode, rows, step.prefill.clone(), &step.decode), idle)
            })
            .collect(),
    );
    let replay = replay
        .filter(|rule| {
            steps
                .iter()
                .zip(&rows)
                .any(|(step, &rows)| replay_rows(graphs, rule, mode, rows, step).is_some())
        })
        .map(|rule| {
            let trimmed: Vec<(Vec<(u32, u32)>, u32)> = steps
                .iter()
                .map(|step| {
                    let prefill: Vec<(u32, u32)> = step
                        .prefill
                        .iter()
                        .map(|&(q, context)| (q.min(rule.window), context))
                        .collect();
                    let kept = prefill.iter().map(|&(q, _)| q).sum::<u32>()
                        + step
                            .decode
                            .iter()
                            .map(|&(q, _)| q.min(rule.window))
                            .sum::<u32>();
                    (prefill, kept)
                })
                .collect();
            let busiest = trimmed.iter().map(|&(_, kept)| kept).max().unwrap_or(0);
            let late_rows = graphs.replay_rows(busiest, rule.window);
            DpLayerInput::new(
                steps
                    .iter()
                    .zip(trimmed)
                    .zip(&idle)
                    .map(|((step, (prefill, _)), &idle)| {
                        unscheduled(
                            layer_input(GraphMode::Piecewise, late_rows, prefill, &step.decode),
                            idle,
                        )
                    })
                    .collect(),
            )
        });
    DpStep {
        mode,
        full,
        replay,
        logits_rows,
    }
}

/// A no-spec DP batch: one group per rank.
fn normalize_input(
    graphs: &DeepseekV41CudaGraphs,
    input: &UnifiedArchInput,
    max_model_len: u32,
    replay: Option<&BoundedReplay>,
) -> std::result::Result<DpStep, String> {
    let dp = DP_SIZE as usize;
    if input.groups.len() != dp {
        return Err(format!(
            "expected one attention group per DP rank ({dp}), got {}",
            input.groups.len()
        ));
    }
    let steps = input
        .groups
        .iter()
        .enumerate()
        .map(|(rank, group)| rank_step(rank, group, max_model_len))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if !input.tokens_per_source_rank.is_empty() {
        let expected: Vec<u32> = input.groups.iter().map(|g| g.batch_tokens).collect();
        if input.tokens_per_source_rank != expected {
            return Err(format!(
                "tokens_per_source_rank {:?} must equal the groups' batch tokens {expected:?}",
                input.tokens_per_source_rank
            ));
        }
    }
    if steps.iter().all(|step| step.request_count() == 0) {
        return Err("an iteration must carry at least one token".into());
    }
    let logits_rows = steps.iter().map(DeepseekV41Step::request_count).collect();
    Ok(normalize_dp_steps(graphs, &steps, 1, logits_rows, replay))
}

/// Build from the checkpoint config (used by `arch::build`).
pub fn model_cfg(spec: &ModelSpec) -> Result<DeepseekV41ModelCfg> {
    DeepseekV41ModelCfg::from_json(Path::new(&spec.model_config), spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GPU_NAME: &str = "NVIDIA B200";
    const CONFIG: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/model/config/deepseek_v41_flash.json"
    );

    fn cfg() -> DeepseekV41ModelCfg {
        model_cfg(&ModelSpec {
            model_config: CONFIG.into(),
            num_layers: None,
            sim_num_layers: None,
            fp8: true,
        })
        .unwrap()
    }

    fn parallel() -> DeepseekV41VllmDpParallel {
        DeepseekV41VllmDpParallel {
            gpu_name: GPU_NAME.into(),
            max_model_len: 131_072,
            decoder_swa_bounded_replay: true,
            max_num_batched_tokens: 2048,
            cudagraph_capture_sizes: tp_arch::vllm_default_cudagraph_capture_sizes(2048),
        }
    }

    fn graphs() -> DeepseekV41CudaGraphs {
        let p = parallel();
        DeepseekV41CudaGraphs::new(&p.cudagraph_capture_sizes, tp_arch::MAX_NUM_SEQS, &[1])
    }

    fn uniform_demand() -> ExpertDemand {
        ExpertDemand::Popularity {
            layerwise_global_ppm: vec![vec![1_000_000 / 384; 384]; 40],
        }
    }

    fn built() -> DeepseekV41VllmDpModel {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let configs = build_configs(&cfg(), &parallel(), &uniform_demand()).unwrap();
        build("dp".into(), resolve_configs(&configs), &bridge).unwrap()
    }

    fn group(prefill: &[(u32, u32)], decode: &[u32]) -> ArchGroupInput {
        let prefill_tokens: u32 = prefill.iter().map(|&(_, append)| append).sum();
        ArchGroupInput {
            prefill_chunk_pairs: prefill.to_vec(),
            decode_kv_lens: decode.to_vec(),
            prefill_tokens,
            decode_tokens: decode.len() as u32,
            batch_tokens: prefill_tokens + decode.len() as u32,
            ..Default::default()
        }
    }

    fn input(groups: Vec<ArchGroupInput>) -> UnifiedArchInput {
        UnifiedArchInput {
            groups,
            ..Default::default()
        }
    }

    #[test]
    fn configs_run_every_dense_part_at_tp1_and_split_only_the_experts() {
        let configs = build_configs(&cfg(), &parallel(), &uniform_demand()).unwrap();
        assert_eq!(configs.bodies.len(), 9);
        let resolved = resolve_configs(&configs);
        for body in resolved.bodies.iter().chain([&resolved.layer0]) {
            assert_eq!(body.attention.raw_cfg.tp_size, 1);
            assert_eq!(body.ffn.ffn.fused_moe.num_local_experts.get(), 96);
            assert!(body.ffn.ffn.all_reduce.is_none());
            assert_eq!(body.ffn.dispatch.num_gpus, 4);
        }
        assert_eq!(resolved.head.lm_head.n.get(), 129_280);
        assert!(resolved.head.logits_all_gather.is_none());
        assert!(resolved.prologue.embedding_all_reduce.is_none());
        assert_eq!(resolved.engram_prefetch.lookup.local_heads, 24);
        let engram = resolved.bodies[0].engram.as_ref().unwrap();
        assert!(engram.all_gather.is_none());
    }

    #[test]
    fn kv_layout_is_one_rank_of_the_tp_layout() {
        use crate::worker::kv::{KvAccounting, KvAccountingOptions};
        let configs = build_configs(&cfg(), &parallel(), &uniform_demand()).unwrap();
        let stride = 152_064;
        // Nothing replicated: one rank's packed block and null block.
        assert_eq!(configs.kv_cache_layout.engine.fixed_device_bytes, stride);
        // DP4/EP4 + DSpark capture (fork 5629b26146, 1017 checkpoint): rank 0
        // fit 493,892 blocks and logged "GPU KV cache size: 35,510,374 tokens,
        // Maximum concurrency for 131,072 tokens per request: 270.92x".
        let dspark = tp_arch::kv_cache_layout(&cfg(), 1, 3);
        let options = KvAccountingOptions {
            max_in_flight_tokens: Some(2 * 2048),
            ..Default::default()
        };
        let accounting = KvAccounting::resolve(&dspark, options, 493_892 * stride).unwrap();
        let concurrency = accounting.concurrency_at(131_072);
        assert!(
            (concurrency / 270.92 - 1.0).abs() < 0.005,
            "{concurrency} vs 270.92"
        );
    }

    #[test]
    fn idle_ranks_join_with_a_dummy_token_and_graph_steps_pad_to_the_busiest() {
        // Decode-only: ranks of 12, 5, 0 and 1 requests pad to graph 16.
        let step = normalize_input(
            &graphs(),
            &input(vec![
                group(&[], &[100; 12]),
                group(&[], &[200; 5]),
                group(&[], &[]),
                group(&[], &[300]),
            ]),
            131_072,
            None,
        )
        .unwrap();
        assert_eq!(step.mode, GraphMode::Full);
        assert!(step.full.ranks.iter().all(|rank| rank.rows == 16));
        assert_eq!(step.full.exchange.collective_tokens, vec![16; 4]);
        assert_eq!(step.logits_rows, vec![12, 5, 0, 1]);
        assert_eq!(step.full.ranks[2].attention.decode_kv_lens, vec![1]);
        // FULL graph at 16 rows: aux stream and mHC overlap on every rank.
        assert!(step.full.ranks.iter().all(|r| r.attention.aux_stream_live));
        assert!(step.full.exchange.ranks.iter().all(|r| r.mhc_overlap));
        // 17 requests on one rank: FULL graph 24 (no overlap above 16 rows).
        let wide = normalize_input(
            &graphs(),
            &input(vec![
                group(&[], &[100; 17]),
                group(&[], &[200; 5]),
                group(&[], &[]),
                group(&[], &[300]),
            ]),
            131_072,
            None,
        )
        .unwrap();
        assert_eq!(wide.mode, GraphMode::Full);
        assert!(wide.full.ranks.iter().all(|rank| rank.rows == 24));
        assert!(wide.full.exchange.ranks.iter().all(|r| !r.mhc_overlap));

        // One prefill rank makes the step PIECEWISE: no aux stream, no overlap.
        let mixed = normalize_input(
            &graphs(),
            &input(vec![
                group(&[(0, 1000)], &[100; 3]),
                group(&[], &[200; 5]),
                group(&[], &[]),
                group(&[], &[300]),
            ]),
            131_072,
            None,
        )
        .unwrap();
        assert_eq!(mixed.mode, GraphMode::Piecewise);
        assert!(mixed.full.ranks.iter().all(|r| r.rows == 1008));
        assert!(mixed
            .full
            .ranks
            .iter()
            .all(|r| !r.attention.aux_stream_live));
        assert!(mixed.full.exchange.ranks.iter().all(|r| !r.mhc_overlap));
        assert!(mixed.replay.is_none());
    }

    #[test]
    fn ranks_replay_together_padded_to_the_busiest_replay_graph() {
        let rule = BoundedReplay::from_model(&cfg()).unwrap();
        // One 1000-row prefill pads every rank to PIECEWISE 1008 >= 768, so
        // the step replays: rank 0 keeps 128 + 3 rows, the others their decode
        // rows (the dummy its one row at q = 1); all pad to the 256-row replay graph.
        let step = normalize_input(
            &graphs(),
            &input(vec![
                group(&[(0, 1000)], &[100; 3]),
                group(&[], &[200; 5]),
                group(&[], &[]),
                group(&[], &[300]),
            ]),
            131_072,
            Some(&rule),
        )
        .unwrap();
        let late = step.replay.as_ref().unwrap();
        assert!(late.ranks.iter().all(|rank| rank.rows == 256));
        assert_eq!(late.exchange.collective_tokens, vec![256; 4]);
        assert_eq!(
            late.ranks[0].attention.prefill_query_context_pairs,
            vec![(128, 1000)]
        );
        assert_eq!(late.ranks[2].attention.decode_kv_lens, vec![1]);
        // A small PIECEWISE step does not replay on any rank.
        let small = normalize_input(
            &graphs(),
            &input(vec![
                group(&[(0, 500)], &[100; 3]),
                group(&[], &[200; 5]),
                group(&[], &[]),
                group(&[], &[300]),
            ]),
            131_072,
            Some(&rule),
        )
        .unwrap();
        assert!(small.replay.is_none());
        // Eager: only rank 0 has a > 128-row prefill, yet every rank replays.
        let eager = normalize_input(
            &graphs(),
            &input(vec![
                group(&[(0, 2100)], &[]),
                group(&[], &[200; 5]),
                group(&[], &[]),
                group(&[], &[300]),
            ]),
            131_072,
            Some(&rule),
        )
        .unwrap();
        assert_eq!(eager.mode, GraphMode::Eager);
        assert_eq!(eager.full.exchange.collective_tokens, vec![2100, 5, 1, 1]);
        assert_eq!(
            eager.replay.unwrap().exchange.collective_tokens,
            vec![128; 4]
        );
    }

    #[test]
    fn no_captured_graphs_run_every_rank_eager() {
        let eager = DeepseekV41CudaGraphs::new(&[], tp_arch::MAX_NUM_SEQS, &[1]);
        let step = normalize_input(
            &eager,
            &input(vec![
                group(&[], &[100; 12]),
                group(&[], &[200; 5]),
                group(&[], &[]),
                group(&[], &[300]),
            ]),
            131_072,
            None,
        )
        .unwrap();
        assert_eq!(step.mode, GraphMode::Eager);
        assert_eq!(step.full.exchange.collective_tokens, vec![12, 5, 1, 1]);
        let mut parallel = parallel();
        parallel.cudagraph_capture_sizes = Vec::new();
        let configs = build_configs(&cfg(), &parallel, &uniform_demand()).unwrap();
        assert_eq!(configs.graphs.max_tokens(), 0);
    }

    #[test]
    fn a_spec_dummy_is_one_uniform_request_and_keeps_the_full_graph() {
        let spec = DeepseekV41CudaGraphs::new(
            &tp_arch::vllm_default_cudagraph_capture_sizes(2048),
            tp_arch::MAX_NUM_SEQS,
            &[6],
        );
        let busy = DeepseekV41Step {
            prefill: Vec::new(),
            decode: vec![(6, 1000); 5],
        };
        let idle = DeepseekV41Step::default();
        let step = normalize_dp_steps(
            &spec,
            &[busy.clone(), busy.clone(), idle, busy],
            6,
            vec![30, 30, 0, 30],
            None,
        );
        assert_eq!(step.mode, GraphMode::Full);
        let padded = spec.full_decode_requests(5, 6).unwrap() * 6;
        assert!(step.full.ranks.iter().all(|rank| rank.rows == padded));
        assert_eq!(
            step.full.ranks[2].attention.decode_kv_lens,
            vec![1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn invalid_inputs_are_refused() {
        assert!(normalize_input(&graphs(), &input(vec![group(&[], &[1])]), 131_072, None).is_err());
        let idle = vec![group(&[], &[]); 4];
        assert!(normalize_input(&graphs(), &input(idle), 131_072, None).is_err());
        let too_long = vec![
            group(&[(131_000, 1000)], &[]),
            group(&[], &[1]),
            group(&[], &[1]),
            group(&[], &[1]),
        ];
        assert!(normalize_input(&graphs(), &input(too_long), 131_072, None).is_err());
    }

    #[test]
    fn segments_are_max_over_ranks_between_the_collectives() {
        let model = built();
        let tree = model.cost_tree();
        let manifest = tree.manifest();
        let names: Vec<&str> = manifest.slots.iter().map(|s| s.name.as_str()).collect();
        let count = |suffix: &str| names.iter().filter(|n| n.ends_with(suffix)).count();
        // One dispatch, experts and combine per unique layer body (10).
        assert_eq!(count(".routed.dispatch_all_gather"), 10);
        assert_eq!(count(".routed.combine_reduce_scatter"), 10);
        assert_eq!(count(".routed.fused_moe"), 10);
        // Every rank-local leaf appears once per rank.
        assert_eq!(count(".routed.router_gate"), 40);
        assert_eq!(count(".lm_head"), 4);
        assert_eq!(count(".engram_hash"), 4);
        // The replay boundary runs on every rank.
        assert_eq!(count(".layer20.write_kv.kv_insert"), 0);
        assert_eq!(count(".replay.gather"), 4);
        assert_eq!(count(".replay.scatter"), 4);
        // No TP all-reduce or logits gather anywhere.
        assert!(!names
            .iter()
            .any(|n| n.contains("all_reduce") || n.contains("logits_all_gather")));
    }

    #[test]
    fn mixed_and_decode_steps_pass_the_input_check() {
        // Kernel evaluation needs profile.db rows (timing-predict covers it);
        // here only the step shape is checked.
        let model = built();
        for batch in [
            input(vec![
                group(&[(0, 2000)], &[100; 40]),
                group(&[], &[200; 5]),
                group(&[], &[]),
                group(&[(4096, 512)], &[]),
            ]),
            input(vec![
                group(&[], &[100; 12]),
                group(&[], &[200; 11]),
                group(&[], &[50_000; 13]),
                group(&[], &[300; 12]),
            ]),
        ] {
            model.check_input(&batch).unwrap();
        }
        // An over-budget rank runs eager: every rank keeps its own rows.
        let eager = normalize_input(
            &graphs(),
            &input(vec![
                group(&[(0, 2040)], &[100; 40]),
                group(&[], &[200; 5]),
                group(&[], &[]),
                group(&[], &[1]),
            ]),
            131_072,
            None,
        )
        .unwrap();
        assert_eq!(eager.mode, GraphMode::Eager);
        assert_eq!(eager.full.exchange.collective_tokens, vec![2080, 5, 1, 1]);
    }
}
