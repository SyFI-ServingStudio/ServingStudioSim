//! DeepSeek-V4.1-Flash as SGLang v0.5.23 serves it on SM12x (4x RTX PRO 6000,
//! PCIe): TP4 attention and EP4 experts over the same four ranks, chunked
//! prefill 2048, FP8 KV, Engram in host memory.
//!
//! The checkpoint facts, the per-layer type and the layer fold are the vLLM
//! graph's ([`DeepseekV41ModelCfg`], [`fold_layers`]); the launches are
//! SGLang's (`worklet/sglang_dsv41_tp.rs`, `doc/pr/dsv41-sglang-sm120/`).
//!
//! ```text
//! prologue (embedding + all-reduce, Engram hash, hc broadcast)
//! for layer in 0..40: [Engram block at 1, 14] -> attention -> MoE FFN
//! head (hc collapse, final norm, lm_head, logits all-gather)
//! ```
//!
//! This arch prices the target model's one-token decode. DSpark speculative
//! decoding is `deepseek_v41_sglang_dspark`, which reuses this target graph at
//! the verify width ([`DeepseekV41SglangSettings::verify_width`]).

use anyhow::{ensure, Result};

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::deepseek_v41_vllm::{fold_layers, DeepseekV41ModelCfg, FoldPlan};
use crate::arch::kv_layout::{EngineOverhead, KvCacheLayout, KvCachePart};
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::deepseek_v41_attention_tp::DeepseekV41IndexRole;
use crate::worklet::sglang_dsv41_tp::{
    SglangDsv41HeadWorklet, SglangDsv41KernelPath, SglangDsv41LayerConfig, SglangDsv41LayerInput,
    SglangDsv41LayerType, SglangDsv41LayerWorklet, SglangDsv41PrologueWorklet, PAGE_SIZE,
};

pub const ARCH_KIND: &str = "deepseek_v41_sglang";
pub const TP_SIZE: u32 = 4;
pub const EP_SIZE: u32 = 4;

/// SGLang decode CUDA-graph batch sizes for `--max-running-requests 16`
/// (server log `Capture ... CUDA graph end. bs=[1..8, 10, 12, 14, 16]`).
const DECODE_GRAPH_BATCHES: [u32; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16];
/// `sglang::... DSV4 memory calculation: bytes_per_full_token=1673.31` on one
/// rank: every per-token cache (compressed KV, index K, states) amortized per
/// full-context token, as SGLang sizes its `full` pool.
const FULL_TOKEN_BYTES_PER_RANK: u64 = 1673;
/// One V4-layout SWA row (`select_dsv4_kv_layout` off SM100/103), per layer.
const SWA_ROW_BYTES: u64 = 584;
/// SGLang's SWA pool at `--mem-fraction-static 0.86` (`DSV4 pool sizes:
/// swa=49664`, cap mode): a fixed allocation beside the `full` pool.
const SWA_POOL_TOKENS: u64 = 49_664;
/// Engram host-table rows per layer (SGLang shared host table; runner report).
const ENGRAM_TABLE_ROWS: [(u32, u64); 2] = [(1, 384_006_168), (14, 384_016_682)];
/// Engram local heads, head dim and the `wkv` output width (runner report).
const ENGRAM_LOCAL_HEADS: u32 = 24;
const ENGRAM_WKV_OUT: u32 = 25_600;
pub(crate) const HC_SINKHORN_ITERS: u32 = 20;

#[derive(Clone, Debug)]
pub struct DeepseekV41SglangSettings {
    pub gpu_name: String,
    /// From the GPU ([`SglangDsv41KernelPath::of`]).
    pub path: SglangDsv41KernelPath,
    pub max_model_len: u32,
    /// `engram_lookup` residency (`host_shared_huge` with transparent huge
    /// pages on, `host_shared_4k` without).
    pub engram_residency: String,
    /// Query rows per decoding request the target runs: 1 here, the DSpark
    /// verify width (`draft_tokens + 1`) under speculative decoding.
    pub verify_width: u32,
    /// SGLang `--enable-decoder-swa-bounded-replay` ([`BoundedReplay`]).
    pub decoder_swa_bounded_replay: bool,
}

/// SGLang's decoder SWA bounded replay (`models/deepseek_v4.py`,
/// `late_layer_start` and the `LateLayerTail` slicing in
/// `DeepseekV4Model.forward`; on by default in v0.5.23). With
/// `tail_writes_source_kv`, the tail starts AT the last KV source layer (20
/// on V4.1-Flash): `write_tail_source_kv` first writes that layer's
/// compressed KV for every extend row, then layers 20..40 run each extend
/// request's last `SWA_WINDOW` (128) rows only. Nothing later reads the
/// skipped rows' late-layer state except through the 128-token window, and
/// the DSpark draft reads tail rows only, so its KV inject also covers just
/// the tail. Only extend forwards without speculation take it: decode and
/// DSpark verify run every row.
///
/// Measured in the B2 nsys capture (`07_balanced_anchor_c16`): in every
/// 2048-token chunk layers 0-19 run at full M and layers 20-39 at tail M
/// (all-reduce 1.7-1.9 ms vs 0.13-0.27 ms; wq_b 0.44 vs 0.03 ms).
///
/// Approximation: the tail-start layer's own full-row `mhc.combine` before
/// the source-KV write is not priced (one small launch per prefill chunk).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundedReplay {
    pub late_layer_start: u32,
    /// `SWA_WINDOW`: replayed extend rows per request.
    pub window: u32,
}

impl BoundedReplay {
    /// The tail start and SGLang's preconditions (`DeepseekV4Model.__init__`):
    /// the last KV source carries no Engram and is an index source; later
    /// layers compress at ratio 0 or 1 and own no KV or Engram.
    pub fn from_model(model: &DeepseekV41ModelCfg) -> std::result::Result<Self, String> {
        let start = *model
            .kv_source_layer_ids
            .iter()
            .max()
            .ok_or("decoder SWA bounded replay needs kv_source_layer_ids")?;
        if model.engram_layer_ids.contains(&start) || !model.index_source_layer_ids.contains(&start)
        {
            return Err(format!(
                "bounded replay: the last KV source layer {start} must be an index source without Engram"
            ));
        }
        for layer in start + 1..model.num_layers {
            if model.compress_ratios[layer as usize] > 1
                || model.kv_source_layer_ids.contains(&layer)
                || model.engram_layer_ids.contains(&layer)
            {
                return Err(format!(
                    "bounded replay: late layer {layer} must compress at ratio 0/1 and own no KV or Engram"
                ));
            }
        }
        Ok(Self {
            late_layer_start: start,
            window: model.sliding_window,
        })
    }

    fn is_late(&self, layers: &[u32]) -> std::result::Result<bool, String> {
        let late = layers
            .iter()
            .filter(|&&l| l >= self.late_layer_start)
            .count();
        match late {
            0 => Ok(false),
            n if n == layers.len() => Ok(true),
            _ => Err(format!(
                "body for layers {layers:?} straddles late_layer_start {}",
                self.late_layer_start
            )),
        }
    }

    /// The late layers' input: each prefill request keeps its last `window`
    /// rows (same context); decode rows are unchanged; the source-KV writes
    /// keep every row. `None` when the forward is not a prefill (extend).
    pub fn tail_input(&self, full: &SglangDsv41LayerInput) -> Option<SglangDsv41LayerInput> {
        if full.prefill.is_empty() {
            return None;
        }
        let appended: u32 = full.prefill.iter().map(|&(a, _)| a).sum();
        let prefill: Vec<(u32, u32)> = full
            .prefill
            .iter()
            .map(|&(a, c)| (a.min(self.window), c))
            .collect();
        let kept: u32 = prefill.iter().map(|&(a, _)| a).sum();
        Some(SglangDsv41LayerInput {
            rows: full.rows - appended + kept,
            decode_only: false,
            prefill,
            decode_kv_lens: full.decode_kv_lens.clone(),
            decode_rows_per_request: full.decode_rows_per_request,
            source_rows: Some(full.rows),
        })
    }
}

enum PlanNode {
    Body(usize),
    Repeat { n: u32, body: Vec<PlanNode> },
}

pub struct DeepseekV41SglangModel {
    name: String,
    tp_size: u32,
    prologue: SglangDsv41PrologueWorklet,
    bodies: Vec<SglangDsv41LayerWorklet>,
    plan: Vec<PlanNode>,
    /// Per body: all its layers are in the bounded-replay tail.
    late_bodies: Vec<bool>,
    bounded_replay: Option<BoundedReplay>,
    head: SglangDsv41HeadWorklet,
    max_model_len: u32,
    kv_cache_layout: KvCacheLayout,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

fn layer_type(model: &DeepseekV41ModelCfg, layer: u32) -> SglangDsv41LayerType {
    let kind = model.layer_kind(layer);
    SglangDsv41LayerType {
        compress_ratio: kind.attention.compress_ratio,
        kv_source: kind.attention.kv_source,
        index_source: kind.attention.index != DeepseekV41IndexRole::None,
        engram: kind.engram,
    }
}

pub fn build(
    name: String,
    model: &DeepseekV41ModelCfg,
    settings: &DeepseekV41SglangSettings,
    demand: &ExpertDemand,
    bridge: &PerfApiBridge,
) -> std::result::Result<DeepseekV41SglangModel, BuildError> {
    let path = settings.path;
    let layers: Vec<_> = (0..model.num_layers)
        .map(|l| (l, model.layer_kind(l)))
        .collect();
    let folded = fold_layers(&layers);
    let bounded_replay = if settings.decoder_swa_bounded_replay {
        Some(
            BoundedReplay::from_model(model).map_err(|reason| BuildError::FitFailed {
                kind: ARCH_KIND,
                reason,
            })?,
        )
    } else {
        None
    };
    let mut bodies = Vec::new();
    let mut body_layers: Vec<Vec<u32>> = Vec::new();
    fn plan_of(
        folded: &[FoldPlan],
        model: &DeepseekV41ModelCfg,
        settings: &DeepseekV41SglangSettings,
        path: SglangDsv41KernelPath,
        demand: &ExpertDemand,
        name: &str,
        bodies: &mut Vec<SglangDsv41LayerWorklet>,
        body_layers: &mut Vec<Vec<u32>>,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Vec<PlanNode>, BuildError> {
        let mut out = Vec::new();
        for node in folded {
            match node {
                FoldPlan::Body { layers, .. } => {
                    let first = layers[0];
                    let ty = layer_type(model, first);
                    let table_rows = ENGRAM_TABLE_ROWS
                        .iter()
                        .find(|(l, _)| *l == first)
                        .map(|&(_, rows)| rows)
                        .unwrap_or(ENGRAM_TABLE_ROWS[0].1);
                    let cfg = layer_config(model, settings, ty, demand, table_rows);
                    let label = if layers.len() == 1 {
                        format!("{name}.layer{first}")
                    } else {
                        format!("{name}.layers{}", layers_label(layers))
                    };
                    bodies.push(SglangDsv41LayerWorklet::build(label, cfg, bridge)?);
                    body_layers.push(layers.clone());
                    out.push(PlanNode::Body(bodies.len() - 1));
                }
                FoldPlan::Repeat { n, body } => out.push(PlanNode::Repeat {
                    n: *n,
                    body: plan_of(
                        body,
                        model,
                        settings,
                        path,
                        demand,
                        name,
                        bodies,
                        body_layers,
                        bridge,
                    )?,
                }),
            }
        }
        Ok(out)
    }
    let plan = plan_of(
        &folded,
        model,
        settings,
        path,
        demand,
        &name,
        &mut bodies,
        &mut body_layers,
        bridge,
    )?;
    let late_bodies = match &bounded_replay {
        Some(replay) => body_layers
            .iter()
            .map(|layers| replay.is_late(layers))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|reason| BuildError::FitFailed {
                kind: ARCH_KIND,
                reason,
            })?,
        None => vec![false; bodies.len()],
    };
    let prologue = SglangDsv41PrologueWorklet::build(
        format!("{name}.prologue"),
        path,
        &settings.gpu_name,
        model.hidden_size,
        model.hc_mult,
        TP_SIZE,
        bridge,
    )?;
    let head = SglangDsv41HeadWorklet::build(
        format!("{name}.head"),
        path,
        &settings.gpu_name,
        model.hidden_size,
        model.hc_mult,
        model.vocab_size,
        TP_SIZE,
        bridge,
    )?;
    let mut out = DeepseekV41SglangModel {
        name,
        tp_size: TP_SIZE,
        prologue,
        bodies,
        plan,
        late_bodies,
        bounded_replay,
        head,
        max_model_len: settings.max_model_len,
        kv_cache_layout: kv_cache_layout(model),
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    let tree = out.cost_tree();
    out.cost_flat = tree.flatten();
    out.n_slots = tree.n_slots();
    Ok(out)
}

/// One body layer's config: the checkpoint geometry at TP4/EP4 with the
/// deployment's kernel path. The DSpark draft stages reuse it with their own
/// layer type and expert geometry.
pub(crate) fn layer_config(
    model: &DeepseekV41ModelCfg,
    settings: &DeepseekV41SglangSettings,
    ty: SglangDsv41LayerType,
    demand: &ExpertDemand,
    table_rows: u64,
) -> SglangDsv41LayerConfig {
    SglangDsv41LayerConfig {
        path: settings.path,
        gpu_name: settings.gpu_name.clone(),
        layer: ty,
        tp_size: TP_SIZE,
        ep_size: EP_SIZE,
        hidden_size: model.hidden_size,
        hc_mult: model.hc_mult,
        hc_sinkhorn_iters: HC_SINKHORN_ITERS,
        num_attention_heads: model.num_attention_heads,
        head_dim: model.head_dim,
        q_lora_rank: model.q_lora_rank,
        o_lora_rank: model.o_lora_rank,
        o_groups: model.o_groups,
        sliding_window: model.sliding_window,
        index_num_heads: model.index_num_heads,
        index_head_dim: model.index_head_dim,
        index_topk: model.index_topk,
        num_experts: model.num_experts,
        top_k: model.top_k,
        moe_intermediate_size: model.moe_intermediate_size,
        shared_intermediate_size: model.moe_intermediate_size * model.num_shared_experts,
        expert_demand: demand.clone(),
        engram_local_heads: ENGRAM_LOCAL_HEADS,
        engram_head_dim: model.engram_head_dim,
        engram_table_rows: table_rows,
        engram_residency: settings.engram_residency.clone(),
        engram_wkv_n: ENGRAM_WKV_OUT,
        verify_width: settings.verify_width.max(1),
        max_model_len: settings.max_model_len,
    }
}

fn layers_label(layers: &[u32]) -> String {
    let parts: Vec<String> = layers.iter().map(u32::to_string).collect();
    format!("[{}]", parts.join(","))
}

/// SGLang's DSV4 pools on SM12x, one rank, held on each of the TP ranks:
/// the `full` pool at `bytes_per_full_token` per context token, and the SWA
/// pool (`cap` mode) as a fixed allocation, 584-byte rows on every layer.
pub(crate) fn kv_cache_layout(model: &DeepseekV41ModelCfg) -> KvCacheLayout {
    kv_cache_layout_with_draft(model, 0)
}

/// [`kv_cache_layout`] plus `draft_swa_layers` window-only draft layers: the
/// DSpark draft's own SWA pool, sized like the target's (`DSV4 SWA storage:
/// worker=draft`).
pub(crate) fn kv_cache_layout_with_draft(
    model: &DeepseekV41ModelCfg,
    draft_swa_layers: u64,
) -> KvCacheLayout {
    let swa_layers = model.compress_ratios.len() as u64 + draft_swa_layers;
    // Every TP rank holds the whole MQA cache (one latent KV per token), so each
    // part is replicated over the TP ranks, and so is the fixed SWA pool.
    KvCacheLayout::new()
        .with_part(
            KvCachePart::per_token("sglang_dsv4_full", FULL_TOKEN_BYTES_PER_RANK)
                .replicated(TP_SIZE),
        )
        .with_engine(EngineOverhead {
            block_tokens: Some(PAGE_SIZE),
            fixed_device_bytes: SWA_POOL_TOKENS * SWA_ROW_BYTES * swa_layers * u64::from(TP_SIZE),
        })
}

/// The decode batch SGLang's CUDA graph runs `requests` at; larger batches run
/// eager at their own size.
pub fn decode_graph_batch(requests: u32) -> u32 {
    DECODE_GRAPH_BATCHES
        .iter()
        .copied()
        .find(|&b| b >= requests)
        .unwrap_or(requests)
}

pub(crate) fn normalize_input(
    input: &UnifiedArchInput,
    max_model_len: u32,
) -> std::result::Result<(SglangDsv41LayerInput, u32), String> {
    if input.groups.len() != 1 {
        return Err(format!(
            "TP4 attention takes exactly one group, got {}",
            input.groups.len()
        ));
    }
    let group = &input.groups[0];
    let mut prefill = Vec::with_capacity(group.prefill_chunk_pairs.len());
    let mut prefill_tokens = 0_u32;
    for (i, &(prefix, append)) in group.prefill_chunk_pairs.iter().enumerate() {
        let context = prefix
            .checked_add(append)
            .ok_or("prefill context overflows u32")?;
        if append == 0 || context > max_model_len {
            return Err(format!(
                "prefill request {i} ({prefix}, {append}) must append 1..=max_model_len {max_model_len}"
            ));
        }
        prefill_tokens += append;
        prefill.push((append, context));
    }
    let decodes =
        u32::try_from(group.decode_kv_lens.len()).map_err(|_| "decode count exceeds u32")?;
    if group.prefill_tokens != prefill_tokens
        || group.decode_tokens != decodes
        || group.batch_tokens != prefill_tokens + decodes
    {
        return Err("group token accounting is inconsistent".into());
    }
    let decode_only = prefill_tokens == 0 && decodes > 0;
    let rows = if decode_only {
        decode_graph_batch(decodes)
    } else {
        group.batch_tokens
    };
    Ok((
        SglangDsv41LayerInput {
            rows,
            decode_only,
            prefill,
            decode_kv_lens: group.decode_kv_lens.clone(),
            decode_rows_per_request: 1,
            source_rows: None,
        },
        group.request_count(),
    ))
}

impl DeepseekV41SglangModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let sections = self.compile_target(&mut b);
        let root = CostNode::Labeled {
            label: format!(
                "{} (DeepseekV41SglangModel) [SGLang SM12x; TP4 attention, EP4 experts; {} unique bodies over 40 layers]",
                self.name,
                self.bodies.len()
            ),
            child: Box::new(CostNode::Sum(sections)),
        };
        b.finish(root)
    }

    fn compile_plan(&self, plan: &[PlanNode], b: &mut CostTreeBuilder) -> Vec<CostNode> {
        plan.iter()
            .map(|node| match node {
                PlanNode::Body(i) => self.bodies[*i].compile(b),
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

    fn eval_plan(
        &self,
        plan: &[PlanNode],
        input: &SglangDsv41LayerInput,
        tail: Option<&SglangDsv41LayerInput>,
        ev: &mut Evaluator,
    ) {
        for node in plan {
            match node {
                PlanNode::Body(i) => {
                    let body_input = match tail {
                        Some(tail) if self.late_bodies[*i] => tail,
                        _ => input,
                    };
                    self.bodies[*i].eval(body_input, ev)
                }
                PlanNode::Repeat { body, .. } => self.eval_plan(body, input, tail, ev),
            }
        }
    }

    fn eval_into(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        let (input, logits_rows) = normalize_input(batch, self.max_model_len)
            .unwrap_or_else(|reason| panic!("invalid {ARCH_KIND} input: {reason}"));
        self.eval_target(&input, logits_rows, ev);
    }

    /// The target forward's sections, in mint order (prologue, layers, head).
    pub(crate) fn compile_target(&self, b: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut sections = vec![self.prologue.compile(b)];
        sections.extend(self.compile_plan(&self.plan, b));
        sections.push(self.head.compile(b));
        sections
    }

    /// Evaluate [`Self::compile_target`]'s slots for one forward. Returns the
    /// rows the last layers ran (the bounded-replay tail of a prefill, else
    /// every row): the hidden states the DSpark KV inject reads.
    pub(crate) fn eval_target(
        &self,
        input: &SglangDsv41LayerInput,
        logits_rows: u32,
        ev: &mut Evaluator,
    ) -> u32 {
        let tail = self
            .bounded_replay
            .and_then(|replay| replay.tail_input(input));
        self.prologue.eval(input.rows, ev);
        self.eval_plan(&self.plan, input, tail.as_ref(), ev);
        self.head.eval(logits_rows, self.tp_size, ev);
        tail.map_or(input.rows, |t| t.rows)
    }

    pub(crate) fn unique_bodies(&self) -> usize {
        self.bodies.len()
    }
}

impl IterwiseUnifiedModel for DeepseekV41SglangModel {
    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut ev = Evaluator::new(slots);
        self.eval_into(batch, &mut ev);
        debug_assert_eq!(ev.filled(), self.n_slots);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut ev = Evaluator::with_inputs(slots, inputs);
        self.eval_into(batch, &mut ev);
        debug_assert_eq!(ev.filled(), self.n_slots);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout.clone()
    }

    fn gpus_per_replica(&self) -> u16 {
        TP_SIZE as u16
    }

    fn num_attn_dp_groups(&self) -> u16 {
        1
    }

    fn num_attn_shards(&self) -> u16 {
        TP_SIZE as u16
    }
}

/// `max_model_len` defaults to the checkpoint's `max_position_embeddings`
/// (SGLang `--context-length`), and nothing longer is accepted.
pub fn resolve_max_model_len(requested: Option<u32>, max_position_embeddings: u32) -> Result<u32> {
    let len = requested.unwrap_or(max_position_embeddings);
    ensure!(
        (1..=max_position_embeddings).contains(&len),
        "{ARCH_KIND}: max_model_len must be 1..={max_position_embeddings}, got {len}"
    );
    Ok(len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_replay_keeps_each_prefill_tail_and_every_source_row() {
        let replay = BoundedReplay {
            late_layer_start: 20,
            window: 128,
        };
        let full = SglangDsv41LayerInput {
            rows: 2048 + 100 + 2,
            decode_only: false,
            prefill: vec![(2048, 50_000), (100, 100)],
            decode_kv_lens: vec![1000, 3000],
            decode_rows_per_request: 1,
            source_rows: None,
        };
        let tail = replay.tail_input(&full).expect("a prefill forward");
        assert_eq!(tail.prefill, vec![(128, 50_000), (100, 100)]);
        assert_eq!(tail.rows, 128 + 100 + 2);
        assert_eq!(tail.source_rows, Some(full.rows));
        assert_eq!(tail.decode_kv_lens, full.decode_kv_lens);
        let decode = SglangDsv41LayerInput {
            rows: 16,
            decode_only: true,
            decode_kv_lens: vec![1000; 16],
            decode_rows_per_request: 1,
            ..Default::default()
        };
        assert!(replay.tail_input(&decode).is_none());
        assert_eq!(replay.is_late(&[20]), Ok(true));
        assert_eq!(replay.is_late(&[2, 8]), Ok(false));
        assert!(replay.is_late(&[19, 20]).is_err());
    }

    #[test]
    fn decode_batches_pad_to_sglang_graph_sizes() {
        assert_eq!(decode_graph_batch(1), 1);
        assert_eq!(decode_graph_batch(9), 10);
        assert_eq!(decode_graph_batch(15), 16);
        assert_eq!(decode_graph_batch(17), 17);
    }
}
