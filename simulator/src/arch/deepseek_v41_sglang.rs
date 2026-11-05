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
//! Speculative decoding (DSpark) is not modeled yet: this arch prices the
//! target model's one-token decode, and the speculative selector refuses it.

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
const HC_SINKHORN_ITERS: u32 = 20;

#[derive(Clone, Debug)]
pub struct DeepseekV41SglangSettings {
    pub gpu_name: String,
    /// From the GPU ([`SglangDsv41KernelPath::of`]).
    pub path: SglangDsv41KernelPath,
    pub max_model_len: u32,
    /// `engram_lookup` residency (`host_shared_huge` with transparent huge
    /// pages on, `host_shared_4k` without).
    pub engram_residency: String,
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
    let mut bodies = Vec::new();
    fn plan_of(
        folded: &[FoldPlan],
        model: &DeepseekV41ModelCfg,
        settings: &DeepseekV41SglangSettings,
        path: SglangDsv41KernelPath,
        demand: &ExpertDemand,
        name: &str,
        bodies: &mut Vec<SglangDsv41LayerWorklet>,
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
                    let cfg = SglangDsv41LayerConfig {
                        path,
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
                        shared_intermediate_size: model.moe_intermediate_size
                            * model.num_shared_experts,
                        expert_demand: demand.clone(),
                        engram_local_heads: ENGRAM_LOCAL_HEADS,
                        engram_head_dim: model.engram_head_dim,
                        engram_table_rows: table_rows,
                        engram_residency: settings.engram_residency.clone(),
                        engram_wkv_n: ENGRAM_WKV_OUT,
                    };
                    let label = if layers.len() == 1 {
                        format!("{name}.layer{first}")
                    } else {
                        format!("{name}.layers{}", layers_label(layers))
                    };
                    bodies.push(SglangDsv41LayerWorklet::build(label, cfg, bridge)?);
                    out.push(PlanNode::Body(bodies.len() - 1));
                }
                FoldPlan::Repeat { n, body } => out.push(PlanNode::Repeat {
                    n: *n,
                    body: plan_of(body, model, settings, path, demand, name, bodies, bridge)?,
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
        bridge,
    )?;
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

fn layers_label(layers: &[u32]) -> String {
    let parts: Vec<String> = layers.iter().map(u32::to_string).collect();
    format!("[{}]", parts.join(","))
}

/// SGLang's DSV4 pools on SM12x, one rank, held on each of the TP ranks:
/// the `full` pool at `bytes_per_full_token` per context token, and the SWA
/// pool (`cap` mode) as a fixed allocation, 584-byte rows on every layer.
fn kv_cache_layout(model: &DeepseekV41ModelCfg) -> KvCacheLayout {
    let swa_layers = model.compress_ratios.len() as u64;
    KvCacheLayout::new()
        .with_part(KvCachePart::per_token(
            "sglang_dsv4_full",
            FULL_TOKEN_BYTES_PER_RANK,
        ))
        .with_engine(EngineOverhead {
            block_tokens: Some(PAGE_SIZE),
            fixed_device_bytes: SWA_POOL_TOKENS * SWA_ROW_BYTES * swa_layers,
        })
        .held_on_each_rank(u64::from(TP_SIZE))
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

fn normalize_input(
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
        },
        group.request_count(),
    ))
}

impl DeepseekV41SglangModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let mut sections = vec![self.prologue.compile(&mut b)];
        sections.extend(self.compile_plan(&self.plan, &mut b));
        sections.push(self.head.compile(&mut b));
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

    fn eval_plan(&self, plan: &[PlanNode], input: &SglangDsv41LayerInput, ev: &mut Evaluator) {
        for node in plan {
            match node {
                PlanNode::Body(i) => self.bodies[*i].eval(input, ev),
                PlanNode::Repeat { body, .. } => self.eval_plan(body, input, ev),
            }
        }
    }

    fn eval_into(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        let (input, logits_rows) = normalize_input(batch, self.max_model_len)
            .unwrap_or_else(|reason| panic!("invalid {ARCH_KIND} input: {reason}"));
        self.prologue.eval(input.rows, ev);
        self.eval_plan(&self.plan, &input, ev);
        self.head.eval(logits_rows, self.tp_size, ev);
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
    fn decode_batches_pad_to_sglang_graph_sizes() {
        assert_eq!(decode_graph_batch(1), 1);
        assert_eq!(decode_graph_batch(9), 10);
        assert_eq!(decode_graph_batch(15), 16);
        assert_eq!(decode_graph_batch(17), 17);
    }
}
