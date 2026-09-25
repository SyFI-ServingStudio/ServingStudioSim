//! DeepSeek-V4.1-Flash in vLLM's measured kernel granularity: TP4 attention,
//! EP4 routed experts, B200 (the fork `servingstudio-alignment-v41` @ 892da0822f,
//! capture 2 = Slurm job 1185).
//!
//! One iteration is one sync spine on the critical TP rank (every rank holds
//! every token after the attention all-reduce):
//!
//! ```text
//! prologue (embedding, its all-reduce, Engram hash)
//! Sum[layer-0 attention, layer-0 FFN, Engram lookups for layers 1 and 14]
//! for layer in 1..40: [Engram block at 1, 14] -> attention(layer) -> MoE FFN
//! head (hc post/collapse, final norm, lm_head, logits AllGather)
//! ```
//!
//! The lookups run on side streams but contend with the layer-0 main path on
//! the same device, so they are charged serially after it; they are compiled
//! and evaluated after that path
//! (`DeepseekV41EngramPrefetchLocalWorklet::compile_joined`).
//!
//! Layer folding: each body layer's type is its attention type (entry, compress
//! ratio, KV source, index role, candidate role) plus whether an Engram block
//! precedes it. Consecutive equal types fold into `Scale{n}`, and a repeated
//! sequence of such runs folds into an outer `Scale{n}` (layers 2-13 are
//! `2 x [KV source, 5 x ratio-2]`, layers 21-36 are
//! `4 x [3 x ratio-1, candidate consumer]`). Two layers only share a body when
//! every field of their type matches, so no fold crosses a ratio, KV/index
//! source, candidate, or Engram boundary.
//!
//! Routed demand: every MoE layer shares one `ExpertDemand` over all 40 routed
//! layers (the capture's token corpus, `routing = corpus`), priced on the busiest
//! EP rank (`folded_rank_position = 0`).
//!
//! Model config: `model/config/deepseek_v41_flash.json` is the HF checkpoint's
//! `text_config` (snapshot dba1be0a40aa45a94ad051997016db3960a90277) flattened to
//! the top level, with the top-level `architectures`, `dtype` (as `torch_dtype`)
//! and `quantization_config` kept and the vision tower dropped.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

use crate::arch::config::ModelSpec;
use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::op::attention::DeepseekV41CandidateRole;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::deepseek_v41_common::attention_aux_stream_live;
use crate::worklet::{
    DeepseekV41AttentionEntry, DeepseekV41AttentionLayer, DeepseekV41AttentionTpWorklet,
    DeepseekV41AttentionTpWorkletConfig, DeepseekV41AttentionTpWorkletInput,
    DeepseekV41AttentionTpWorkletResolved, DeepseekV41EngramPrefetchLocalWorklet,
    DeepseekV41EngramPrefetchLocalWorkletConfig, DeepseekV41EngramPrefetchLocalWorkletInput,
    DeepseekV41EngramPrefetchLocalWorkletResolved, DeepseekV41EngramTpWorklet,
    DeepseekV41EngramTpWorkletConfig, DeepseekV41EngramTpWorkletInput,
    DeepseekV41EngramTpWorkletResolved, DeepseekV41HeadTpWorklet, DeepseekV41HeadTpWorkletConfig,
    DeepseekV41HeadTpWorkletInput, DeepseekV41HeadTpWorkletResolved, DeepseekV41IndexRole,
    DeepseekV41MoeFfnEpWorklet, DeepseekV41MoeFfnEpWorkletConfig, DeepseekV41MoeFfnEpWorkletInput,
    DeepseekV41MoeFfnEpWorkletResolved, DeepseekV41PrologueTpWorklet,
    DeepseekV41PrologueTpWorkletConfig, DeepseekV41PrologueTpWorkletInput,
    DeepseekV41PrologueTpWorkletResolved,
};

const ARCH_KIND: &str = "deepseek_v41_vllm";
const GPU_NAME: &str = "NVIDIA B200";
const TP_SIZE: u32 = 4;
const EP_SIZE: u32 = 4;

// Checkpoint identity (HF text_config); `from_json` rejects anything else.
const BODY_LAYERS: u32 = 40;
const MTP_LAYERS: u32 = 3;
const HIDDEN: u32 = 5120;
const NUM_EXPERTS: u32 = 384;
const TOP_K: u32 = 6;
const EXPERT_WIDTH: u32 = 2304;
const VOCAB_SIZE: u32 = 129_280;
const KV_SOURCE_LAYERS: [u32; 4] = [2, 8, 14, 20];
const INDEX_SOURCE_LAYERS: [u32; 8] = [2, 8, 14, 20, 24, 28, 32, 36];
const CANDIDATE_SOURCE_LAYER: u32 = 20;
const ENGRAM_LAYERS: [u32; 2] = [1, 14];

// Deployment identity (capture 2 `profile.yaml` / server log).
/// `--max-model-len 131072`.
const MAX_MODEL_LEN: u32 = 131_072;
/// `max_num_batched_tokens` (= `chunk_size` 2048).
const MAX_BATCHED_TOKENS: u32 = 2048;
/// `--block-size 128`: one KV page; an index page holds `128 / ratio` keys.
const KV_BLOCK_SIZE: u32 = 128;
/// Index-cache page used to price ratio-1 index layers. Production pages hold
/// `128 / ratio` keys (128 at ratio 1), but the `dsa_paged_mqa_logits_decode`
/// DeepGEMM runner only builds 64-key pages (`_BLOCK_SIZE = 64`), so ratio-1
/// scoring is priced on 64-key pages: the same keys in twice the pages.
const RATIO1_INDEX_PAGE_FALLBACK: u32 = 64;
/// SWA page of the fused q-norm/rope/KV insert (the profiled kernel identity).
const SWA_BLOCK_SIZE: u32 = 32;
/// FlashMLA's padded per-rank Q width.
const PADDED_HEADS: u32 = 64;
/// Mega-attention prefill chunking (the profiled op identity).
const PREFILL_CHUNK_SIZE: u32 = 4;
/// FULL CUDA graphs cover decode batches up to `max_num_seqs` 64 (11 sizes);
/// only they keep the attention aux stream (`attention_aux_stream_live`).
const MAX_FULL_GRAPH_TOKENS: u32 = 64;
/// `max_cudagraph_capture_size 2048`: the piecewise graph sizes run by every
/// iteration up to 2048 tokens pad the dense spine to the next capture size.
const MAX_CUDAGRAPH_CAPTURE_SIZE: u32 = 2048;
/// Rows of TP rank 0's Engram table slice (`part_num_embeddings`, server log
/// `engram.py:269`, layer 1): the lookup kind's exact table identity.
const ENGRAM_TABLE_ROWS_RANK0: u64 = 96_000_564;
/// `wkv` output width of the Engram block (captured MXFP8 GEMM 6144 -> 25600).
const ENGRAM_WKV_OUT: u32 = 25_600;

const MHC_BACKENDS: &[&str] = &["deepgemm_mega"];
const MXFP8_GEMM_BACKENDS: &[&str] = &["flashinfer_mxfp8"];
const FP32_GEMM_BACKENDS: &[&str] = &["torch_cublas_vllm_fork"];
const KV_INSERT_BACKENDS: &[&str] = &["vllm_cuda"];
const MEGA_ATTN_BACKENDS: &[&str] = &["flashmla_mega"];
const WO_A_BACKENDS: &[&str] = &["deepgemm_mxfp8_einsum_dsv41_wo_a"];
const ALL_REDUCE_BACKENDS: &[&str] = &["flashinfer_mnnvl"];
const ALL_GATHER_PROXY_BACKENDS: &[&str] = &["nccl"];
const INDEX_LOGITS_BACKENDS: &[&str] = &["deepgemm_fp8"];
const ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
const MOE_BACKENDS: &[&str] = &["flashinfer_trtllm_sm100_mxfp4"];
const ENGRAM_LOOKUP_BACKENDS: &[&str] = &["vllm_triton"];
const LM_HEAD_BACKENDS: &[&str] = &["torch_linear"];

/// The checkpoint facts this arch reads, validated against the exact V4.1-Flash
/// identity.
#[derive(Clone, Debug)]
pub struct DeepseekV41ModelCfg {
    pub num_layers: u32,
    pub hidden_size: u32,
    pub hc_mult: u32,
    pub num_attention_heads: u32,
    pub head_dim: u32,
    pub rope_dim: u32,
    pub q_lora_rank: u32,
    pub o_lora_rank: u32,
    pub o_groups: u32,
    pub sliding_window: u32,
    pub index_num_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub moe_intermediate_size: u32,
    pub num_shared_experts: u32,
    pub vocab_size: u32,
    /// Body layers only (the three MTP entries are dropped); 0, 1 or 2.
    pub compress_ratios: Vec<u32>,
    pub kv_source_layer_ids: Vec<u32>,
    pub index_source_layer_ids: Vec<u32>,
    pub candidate_source_layer_id: u32,
    pub engram_layer_ids: Vec<u32>,
    /// Hash columns = `(engram_max_ngram_size - 1) * engram_n_heads` (24).
    pub engram_hash_heads: u32,
    pub engram_head_dim: u32,
}

#[derive(Deserialize)]
struct JsonQuantization {
    quant_method: String,
    expert_dtype: String,
}

#[derive(Deserialize)]
struct JsonDeepseekV41Config {
    architectures: Vec<String>,
    model_type: String,
    torch_dtype: String,
    quantization_config: JsonQuantization,
    hidden_size: u32,
    hc_mult: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    qk_rope_head_dim: u32,
    q_lora_rank: u32,
    o_lora_rank: u32,
    o_groups: u32,
    sliding_window: u32,
    index_n_heads: u32,
    index_head_dim: u32,
    index_topk: u32,
    n_routed_experts: u32,
    num_experts_per_tok: u32,
    moe_intermediate_size: u32,
    n_shared_experts: u32,
    vocab_size: u32,
    max_position_embeddings: u32,
    /// Absent on V4.1 (V4 routed its first three layers by token hash).
    #[serde(default)]
    num_hash_layers: u32,
    num_nextn_predict_layers: u32,
    scoring_func: String,
    topk_method: String,
    norm_topk_prob: bool,
    compress_ratios: Vec<u32>,
    kv_source_layer_ids: Vec<u32>,
    index_source_layer_ids: Vec<u32>,
    candidate_source_layer_id: u32,
    candidate_topk_blocks: u32,
    candidate_block_size: u32,
    engram_layer_ids: Vec<u32>,
    engram_max_ngram_size: u32,
    engram_n_heads: u32,
    engram_head_dim: u32,
}

impl DeepseekV41ModelCfg {
    pub fn from_json(path: &Path, spec: &ModelSpec) -> Result<Self> {
        ensure!(
            spec.fp8,
            "{ARCH_KIND} requires fp8=true (MXFP8 dense GEMMs, FP8 KV)"
        );
        ensure!(
            spec.num_layers.is_none() && spec.sim_num_layers.is_none(),
            "{ARCH_KIND} requires the exact heterogeneous 40-layer schedule"
        );
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading DeepSeek V4.1 config {}", path.display()))?;
        let raw: JsonDeepseekV41Config = serde_json::from_str(&text).context("parsing JSON")?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: JsonDeepseekV41Config) -> Result<Self> {
        ensure!(raw.architectures == ["DeepseekV41ForCausalLM"]);
        ensure!(raw.model_type == "deepseek_v41_text");
        ensure!(raw.torch_dtype == "bfloat16");
        ensure!(raw.quantization_config.quant_method == "fp8");
        ensure!(raw.quantization_config.expert_dtype == "fp4");
        ensure!(raw.scoring_func == "sqrtsoftplus");
        ensure!(raw.topk_method == "noaux_tc" && raw.norm_topk_prob);
        for (name, actual, expected) in [
            ("hidden_size", raw.hidden_size, HIDDEN),
            ("hc_mult", raw.hc_mult, 4),
            ("num_hidden_layers", raw.num_hidden_layers, BODY_LAYERS),
            ("num_attention_heads", raw.num_attention_heads, 64),
            ("num_key_value_heads", raw.num_key_value_heads, 1),
            ("head_dim", raw.head_dim, 512),
            ("qk_rope_head_dim", raw.qk_rope_head_dim, 64),
            ("q_lora_rank", raw.q_lora_rank, 1280),
            ("o_lora_rank", raw.o_lora_rank, 1024),
            ("o_groups", raw.o_groups, 8),
            ("sliding_window", raw.sliding_window, 128),
            ("index_n_heads", raw.index_n_heads, 32),
            ("index_head_dim", raw.index_head_dim, 128),
            ("index_topk", raw.index_topk, 512),
            ("n_routed_experts", raw.n_routed_experts, NUM_EXPERTS),
            ("num_experts_per_tok", raw.num_experts_per_tok, TOP_K),
            (
                "moe_intermediate_size",
                raw.moe_intermediate_size,
                EXPERT_WIDTH,
            ),
            ("n_shared_experts", raw.n_shared_experts, 1),
            ("vocab_size", raw.vocab_size, VOCAB_SIZE),
            (
                "max_position_embeddings",
                raw.max_position_embeddings,
                1_048_576,
            ),
            ("num_hash_layers", raw.num_hash_layers, 0),
            (
                "num_nextn_predict_layers",
                raw.num_nextn_predict_layers,
                MTP_LAYERS,
            ),
            (
                "candidate_source_layer_id",
                raw.candidate_source_layer_id,
                CANDIDATE_SOURCE_LAYER,
            ),
            ("candidate_topk_blocks", raw.candidate_topk_blocks, 2048),
            ("candidate_block_size", raw.candidate_block_size, 8),
            ("engram_max_ngram_size", raw.engram_max_ngram_size, 4),
            ("engram_n_heads", raw.engram_n_heads, 8),
            ("engram_head_dim", raw.engram_head_dim, 256),
        ] {
            ensure!(
                actual == expected,
                "{name} must be {expected}, got {actual}"
            );
        }
        ensure!(
            raw.kv_source_layer_ids == KV_SOURCE_LAYERS,
            "kv_source_layer_ids must be {KV_SOURCE_LAYERS:?}, got {:?}",
            raw.kv_source_layer_ids
        );
        ensure!(
            raw.index_source_layer_ids == INDEX_SOURCE_LAYERS,
            "index_source_layer_ids must be {INDEX_SOURCE_LAYERS:?}, got {:?}",
            raw.index_source_layer_ids
        );
        ensure!(
            raw.engram_layer_ids == ENGRAM_LAYERS,
            "engram_layer_ids must be {ENGRAM_LAYERS:?}, got {:?}",
            raw.engram_layer_ids
        );
        let body = BODY_LAYERS as usize;
        ensure!(
            raw.compress_ratios.len() == body + MTP_LAYERS as usize
                && raw.compress_ratios[body..].iter().all(|&r| r == 0),
            "compress_ratios must list 40 body layers then 3 zero MTP entries"
        );
        let ratios = raw.compress_ratios[..body].to_vec();
        ensure!(
            ratios.iter().enumerate().all(|(layer, &ratio)| ratio
                == match layer {
                    0..=1 => 0,
                    2..=19 => 2,
                    _ => 1,
                }),
            "compress_ratios must be 0 x2, 2 x18, 1 x20, got {ratios:?}"
        );
        Ok(Self {
            num_layers: raw.num_hidden_layers,
            hidden_size: raw.hidden_size,
            hc_mult: raw.hc_mult,
            num_attention_heads: raw.num_attention_heads,
            head_dim: raw.head_dim,
            rope_dim: raw.qk_rope_head_dim,
            q_lora_rank: raw.q_lora_rank,
            o_lora_rank: raw.o_lora_rank,
            o_groups: raw.o_groups,
            sliding_window: raw.sliding_window,
            index_num_heads: raw.index_n_heads,
            index_head_dim: raw.index_head_dim,
            index_topk: raw.index_topk,
            num_experts: raw.n_routed_experts,
            top_k: raw.num_experts_per_tok,
            moe_intermediate_size: raw.moe_intermediate_size,
            num_shared_experts: raw.n_shared_experts,
            vocab_size: raw.vocab_size,
            compress_ratios: ratios,
            kv_source_layer_ids: raw.kv_source_layer_ids,
            index_source_layer_ids: raw.index_source_layer_ids,
            candidate_source_layer_id: raw.candidate_source_layer_id,
            engram_layer_ids: raw.engram_layer_ids,
            engram_hash_heads: (raw.engram_max_ngram_size - 1) * raw.engram_n_heads,
            engram_head_dim: raw.engram_head_dim,
        })
    }

    /// One body layer's type, read off the config's layer lists.
    pub fn layer_kind(&self, layer: u32) -> LayerKind {
        let kv_source = self.kv_source_layer_ids.contains(&layer);
        let index_source = self.index_source_layer_ids.contains(&layer);
        let engram = self.engram_layer_ids.contains(&layer);
        LayerKind {
            attention: DeepseekV41AttentionLayer {
                entry: if layer == 0 {
                    DeepseekV41AttentionEntry::Layer0Pre
                } else if engram {
                    DeepseekV41AttentionEntry::AfterEngram
                } else {
                    DeepseekV41AttentionEntry::FusedPostPre
                },
                compress_ratio: self.compress_ratios[layer as usize],
                kv_source,
                index: match (index_source, kv_source) {
                    (false, _) => DeepseekV41IndexRole::None,
                    (true, true) => DeepseekV41IndexRole::Owner,
                    (true, false) => DeepseekV41IndexRole::NonOwner,
                },
                candidate: if layer == self.candidate_source_layer_id {
                    DeepseekV41CandidateRole::Writer
                } else if index_source && !kv_source {
                    DeepseekV41CandidateRole::Consumer
                } else {
                    DeepseekV41CandidateRole::None
                },
            },
            engram,
        }
    }
}

/// Everything that decides a body layer's launches. Equal kinds cost the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerKind {
    pub attention: DeepseekV41AttentionLayer,
    /// An Engram block runs before this layer's attention.
    pub engram: bool,
}

/// The layer fold: `Body` is one built body layer standing for `layers`;
/// `Repeat` is `Scale{n}` over a sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FoldPlan {
    Body { kind: LayerKind, layers: Vec<u32> },
    Repeat { n: u32, body: Vec<FoldPlan> },
}

impl FoldPlan {
    fn kind_sequence(&self) -> Vec<LayerKind> {
        match self {
            Self::Body { kind, .. } => vec![*kind],
            Self::Repeat { n, body } => {
                let once: Vec<_> = body.iter().flat_map(FoldPlan::kind_sequence).collect();
                (0..*n).flat_map(|_| once.clone()).collect()
            }
        }
    }

    fn same_shape(&self, other: &Self) -> bool {
        self.kind_sequence() == other.kind_sequence()
    }

    /// Merge `other` (same shape) into `self`: bodies absorb its layers.
    fn absorb(&mut self, other: &Self) {
        match (self, other) {
            (Self::Body { layers, .. }, Self::Body { layers: more, .. }) => {
                layers.extend_from_slice(more)
            }
            (Self::Repeat { body, .. }, Self::Repeat { body: more, .. }) => {
                for (mine, theirs) in body.iter_mut().zip(more) {
                    mine.absorb(theirs);
                }
            }
            _ => unreachable!("absorb requires the same shape"),
        }
    }
}

/// Longest repeated sequence of run-length items tried by [`fold_layers`].
const MAX_FOLD_PERIOD: usize = 4;

/// Fold `(layer, kind)` in order: run-length encode equal kinds into
/// `Scale{n}`, then fold consecutive repeats of 2..=4 such items into an outer
/// `Scale{n}`. Expanding the result reproduces the input kind sequence.
pub fn fold_layers(layers: &[(u32, LayerKind)]) -> Vec<FoldPlan> {
    let mut items: Vec<FoldPlan> = Vec::new();
    let mut index = 0;
    while index < layers.len() {
        let (first, kind) = layers[index];
        let mut run = vec![first];
        while index + run.len() < layers.len() && layers[index + run.len()].1 == kind {
            run.push(layers[index + run.len()].0);
        }
        index += run.len();
        let n = run.len() as u32;
        let body = FoldPlan::Body { kind, layers: run };
        items.push(if n == 1 {
            body
        } else {
            FoldPlan::Repeat {
                n,
                body: vec![body],
            }
        });
    }

    let mut folded = Vec::new();
    let mut index = 0;
    while index < items.len() {
        let mut best = (1, 1);
        for period in 2..=MAX_FOLD_PERIOD {
            let mut reps = 1;
            while index + (reps + 1) * period <= items.len()
                && (0..period).all(|offset| {
                    items[index + offset].same_shape(&items[index + reps * period + offset])
                })
            {
                reps += 1;
            }
            if reps >= 2 && reps * period > best.0 * best.1 {
                best = (period, reps);
            }
        }
        let (period, reps) = best;
        if reps == 1 {
            folded.push(items[index].clone());
            index += 1;
            continue;
        }
        let mut body: Vec<FoldPlan> = items[index..index + period].to_vec();
        for rep in 1..reps {
            for offset in 0..period {
                body[offset].absorb(&items[index + rep * period + offset]);
            }
        }
        folded.push(FoldPlan::Repeat {
            n: reps as u32,
            body,
        });
        index += period * reps;
    }
    folded
}

#[derive(Clone, Debug)]
pub struct DeepseekV41VllmParallel {
    pub tp_size: u32,
    pub ep_size: u32,
    pub gpu_name: String,
    /// The serial-stream counterfactual: every gated side branch runs serial.
    pub serialize_streams: bool,
}

/// One body layer's worklet configs.
#[derive(Clone)]
pub struct DeepseekV41LayerConfig {
    pub layers: Vec<u32>,
    pub engram: Option<DeepseekV41EngramTpWorkletConfig>,
    pub attention: DeepseekV41AttentionTpWorkletConfig,
    pub ffn: DeepseekV41MoeFfnEpWorkletConfig,
}

/// Fold plan with each body replaced by its index into the body list.
#[derive(Clone, Debug, PartialEq, Eq)]
enum PlanNode {
    Body(usize),
    Repeat { n: u32, body: Vec<PlanNode> },
}

pub struct DeepseekV41VllmConfigs {
    pub prologue: DeepseekV41PrologueTpWorkletConfig,
    pub engram_prefetch: DeepseekV41EngramPrefetchLocalWorkletConfig,
    pub layer0: DeepseekV41LayerConfig,
    /// Bodies of layers 1..40, in first-visit order of `plan`.
    pub bodies: Vec<DeepseekV41LayerConfig>,
    plan: Vec<PlanNode>,
    pub head: DeepseekV41HeadTpWorkletConfig,
    pub total_kv_bytes_per_token: u64,
}

pub struct DeepseekV41LayerResolved {
    layers: Vec<u32>,
    engram: Option<DeepseekV41EngramTpWorkletResolved>,
    attention: DeepseekV41AttentionTpWorkletResolved,
    ffn: DeepseekV41MoeFfnEpWorkletResolved,
}

pub struct DeepseekV41VllmResolved {
    prologue: DeepseekV41PrologueTpWorkletResolved,
    engram_prefetch: DeepseekV41EngramPrefetchLocalWorkletResolved,
    layer0: DeepseekV41LayerResolved,
    bodies: Vec<DeepseekV41LayerResolved>,
    plan: Vec<PlanNode>,
    head: DeepseekV41HeadTpWorkletResolved,
    total_kv_bytes_per_token: u64,
}

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

pub fn build_configs(
    model: &DeepseekV41ModelCfg,
    parallel: &DeepseekV41VllmParallel,
    demand: &ExpertDemand,
) -> std::result::Result<DeepseekV41VllmConfigs, BuildError> {
    if parallel.tp_size != TP_SIZE || parallel.ep_size != EP_SIZE || parallel.gpu_name != GPU_NAME {
        return Err(fit_failed(
            "DeepSeek-V4.1-Flash is profiled only as B200 TP4 attention + EP4 experts",
        ));
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
    let tail: Vec<(u32, LayerKind)> = (1..model.num_layers)
        .map(|layer| (layer, model.layer_kind(layer)))
        .collect();
    let fold = fold_layers(&tail);
    let mut bodies = Vec::new();
    let plan = index_plan(&fold, &mut |kind, layers| {
        bodies.push(layer_config(model, parallel, demand, kind, layers.to_vec()));
        bodies.len() - 1
    });
    let gpu = parallel.gpu_name.clone();
    Ok(DeepseekV41VllmConfigs {
        prologue: DeepseekV41PrologueTpWorkletConfig {
            tp_size: parallel.tp_size,
            hidden_size: model.hidden_size.into(),
            engram_num_heads: model.engram_hash_heads,
            // Launch-bound metadata: left to the framework-overhead term.
            metadata_bytes_per_token: None,
            gpu_name: gpu.clone(),
            all_reduce_backends: ALL_REDUCE_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        },
        engram_prefetch: DeepseekV41EngramPrefetchLocalWorkletConfig {
            tp_size: parallel.tp_size,
            engram_num_heads: model.engram_hash_heads,
            engram_head_dim: model.engram_head_dim,
            quant_block_size: 32,
            table_rows: ENGRAM_TABLE_ROWS_RANK0,
            residency: "host_uva".into(),
            weight_dtype: DType::Fp8E4m3,
            gpu_name: gpu.clone(),
            backends: ENGRAM_LOOKUP_BACKENDS.to_vec(),
        },
        layer0: layer_config(model, parallel, demand, layer0_kind, vec![0]),
        bodies,
        plan,
        head: DeepseekV41HeadTpWorkletConfig {
            tp_size: parallel.tp_size,
            hidden_size: model.hidden_size.into(),
            hc_mult: model.hc_mult,
            vocab_size: model.vocab_size,
            gpu_name: gpu,
            lm_head_backends: LM_HEAD_BACKENDS.to_vec(),
            all_gather_backends: ALL_GATHER_PROXY_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        },
        total_kv_bytes_per_token: total_kv_bytes_per_token(model, parallel.tp_size),
    })
}

fn index_plan(
    fold: &[FoldPlan],
    push: &mut impl FnMut(LayerKind, &[u32]) -> usize,
) -> Vec<PlanNode> {
    fold.iter()
        .map(|node| match node {
            FoldPlan::Body { kind, layers } => PlanNode::Body(push(*kind, layers)),
            FoldPlan::Repeat { n, body } => PlanNode::Repeat {
                n: *n,
                body: index_plan(body, push),
            },
        })
        .collect()
}

/// All-rank bytes of per-token KV state. Each TP rank holds the one KV head
/// (replicated): the four KV-source layers' NVFP4 compressed rows (288 B per
/// `ratio` tokens) and the index owners' FP8 keys (132 B per `ratio` tokens).
/// The ratio-0 sliding window (128 tokens per layer) is a per-request constant,
/// not per-token state, so it is left out.
fn total_kv_bytes_per_token(model: &DeepseekV41ModelCfg, tp_size: u32) -> u64 {
    let per_rank: u64 = model
        .kv_source_layer_ids
        .iter()
        .map(|&layer| {
            let ratio = u64::from(model.compress_ratios[layer as usize]);
            (288 + 132) / ratio
        })
        .sum();
    per_rank * u64::from(tp_size)
}

fn layer_config(
    model: &DeepseekV41ModelCfg,
    parallel: &DeepseekV41VllmParallel,
    demand: &ExpertDemand,
    kind: LayerKind,
    layers: Vec<u32>,
) -> DeepseekV41LayerConfig {
    let gpu = parallel.gpu_name.clone();
    DeepseekV41LayerConfig {
        layers,
        engram: kind.engram.then(|| DeepseekV41EngramTpWorkletConfig {
            tp_size: parallel.tp_size,
            hidden_size: model.hidden_size.into(),
            hc_mult: model.hc_mult,
            engram_num_heads: model.engram_hash_heads,
            engram_head_dim: model.engram_head_dim,
            wkv_out_dim: ENGRAM_WKV_OUT.into(),
            gpu_name: gpu.clone(),
            gemm_backends: MXFP8_GEMM_BACKENDS.to_vec(),
            all_gather_backends: ALL_GATHER_PROXY_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        }),
        attention: DeepseekV41AttentionTpWorkletConfig {
            layer: kind.attention,
            tp_size: parallel.tp_size,
            serialize_streams: parallel.serialize_streams,
            hidden_size: model.hidden_size.into(),
            hc_mult: model.hc_mult,
            num_attention_heads: model.num_attention_heads.into(),
            padded_heads: PADDED_HEADS.into(),
            head_dim: model.head_dim.into(),
            rope_dim: model.rope_dim.into(),
            q_lora_rank: model.q_lora_rank.into(),
            o_lora_rank: model.o_lora_rank.into(),
            o_groups: model.o_groups.into(),
            index_num_heads: model.index_num_heads.into(),
            index_head_dim: model.index_head_dim.into(),
            index_topk: model.index_topk,
            window_size: model.sliding_window,
            // `kv_block_size` only sizes the index page (`kv_block_size / ratio`).
            kv_block_size: if kind.attention.compress_ratio == 1 {
                RATIO1_INDEX_PAGE_FALLBACK
            } else {
                KV_BLOCK_SIZE
            },
            swa_block_size: SWA_BLOCK_SIZE,
            max_model_len: MAX_MODEL_LEN,
            max_num_batched_tokens: MAX_BATCHED_TOKENS,
            prefill_chunk_size: PREFILL_CHUNK_SIZE,
            gpu_name: gpu.clone(),
            mhc_backends: MHC_BACKENDS.to_vec(),
            gemm_backends: MXFP8_GEMM_BACKENDS.to_vec(),
            fp32_gemm_backends: FP32_GEMM_BACKENDS.to_vec(),
            kv_insert_backends: KV_INSERT_BACKENDS.to_vec(),
            mega_attn_backends: MEGA_ATTN_BACKENDS.to_vec(),
            wo_a_backends: WO_A_BACKENDS.to_vec(),
            all_reduce_backends: ALL_REDUCE_BACKENDS.to_vec(),
            index_logits_prefill_backends: INDEX_LOGITS_BACKENDS.to_vec(),
            index_logits_decode_backends: INDEX_LOGITS_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        },
        ffn: DeepseekV41MoeFfnEpWorkletConfig {
            tp_size: parallel.tp_size,
            ep_size: parallel.ep_size,
            serialize_streams: parallel.serialize_streams,
            hidden_size: model.hidden_size.into(),
            hc_mult: model.hc_mult,
            num_experts: model.num_experts.into(),
            top_k: model.top_k,
            moe_intermediate_size: model.moe_intermediate_size.into(),
            shared_intermediate_size: (model.num_shared_experts * model.moe_intermediate_size)
                .into(),
            gpu_name: gpu,
            mhc_backends: MHC_BACKENDS.to_vec(),
            router_backends: FP32_GEMM_BACKENDS.to_vec(),
            gemm_backends: MXFP8_GEMM_BACKENDS.to_vec(),
            moe_backends: MOE_BACKENDS.to_vec(),
            all_reduce_backends: ALL_REDUCE_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
            moe_input_dtype: DType::Bf16,
            weight_format: "mxfp4_ue8m0".into(),
            group_size: 32,
            // Finished top-k ids come in; the router already applied the
            // routed scaling, so the kernel sees 1/1 and no grouping.
            routing_method: "precomputed_dsv4".into(),
            n_group: 1,
            topk_group: 1,
            routed_scaling_numerator: 1,
            routed_scaling_denominator: 1,
            expert_demand: demand.clone(),
            folded_rank_position: 0,
        },
    }
}

fn resolve_layer(config: &DeepseekV41LayerConfig) -> DeepseekV41LayerResolved {
    DeepseekV41LayerResolved {
        layers: config.layers.clone(),
        engram: config
            .engram
            .as_ref()
            .map(DeepseekV41EngramTpWorklet::resolve_config),
        attention: DeepseekV41AttentionTpWorklet::resolve_config(&config.attention),
        ffn: DeepseekV41MoeFfnEpWorklet::resolve_config(&config.ffn),
    }
}

pub fn resolve_configs(configs: &DeepseekV41VllmConfigs) -> DeepseekV41VllmResolved {
    DeepseekV41VllmResolved {
        prologue: DeepseekV41PrologueTpWorklet::resolve_config(&configs.prologue),
        engram_prefetch: DeepseekV41EngramPrefetchLocalWorklet::resolve_config(
            &configs.engram_prefetch,
        ),
        layer0: resolve_layer(&configs.layer0),
        bodies: configs.bodies.iter().map(resolve_layer).collect(),
        plan: configs.plan.clone(),
        head: DeepseekV41HeadTpWorklet::resolve_config(&configs.head),
        total_kv_bytes_per_token: configs.total_kv_bytes_per_token,
    }
}

/// One built body layer: optional Engram block, attention, MoE FFN.
struct DeepseekV41LayerBody {
    name: String,
    layers: Vec<u32>,
    engram: Option<DeepseekV41EngramTpWorklet>,
    attention: DeepseekV41AttentionTpWorklet,
    ffn: DeepseekV41MoeFfnEpWorklet,
}

impl DeepseekV41LayerBody {
    fn build(
        model_name: &str,
        resolved: DeepseekV41LayerResolved,
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
            ffn: DeepseekV41MoeFfnEpWorklet::build(format!("{name}.ffn"), resolved.ffn, bridge)?,
            layers: resolved.layers,
            name,
        })
    }

    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let mut children = Vec::new();
        if let Some(engram) = &self.engram {
            children.push(engram.compile(builder));
        }
        children.push(self.attention.compile(builder));
        children.push(self.ffn.compile(builder));
        CostNode::Labeled {
            label: format!("{} [layers {:?}]", self.name, self.layers),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    fn eval(&self, input: &NormalizedInput, evaluator: &mut Evaluator) {
        if let Some(engram) = &self.engram {
            engram.eval(
                &DeepseekV41EngramTpWorkletInput {
                    num_tokens: input.rows,
                },
                evaluator,
            );
        }
        self.attention.eval(&input.attention, evaluator);
        self.ffn.eval(
            &DeepseekV41MoeFfnEpWorkletInput {
                num_tokens: input.rows,
            },
            evaluator,
        );
    }
}

pub struct DeepseekV41VllmModel {
    name: String,
    prologue: DeepseekV41PrologueTpWorklet,
    engram_prefetch: DeepseekV41EngramPrefetchLocalWorklet,
    layer0: DeepseekV41LayerBody,
    bodies: Vec<DeepseekV41LayerBody>,
    plan: Vec<PlanNode>,
    head: DeepseekV41HeadTpWorklet,
    serialize_streams: bool,
    total_kv_bytes_per_token: u64,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: DeepseekV41VllmResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<DeepseekV41VllmModel, BuildError> {
    let serialize_streams = resolved.layer0.attention.raw_cfg.serialize_streams;
    let bodies = resolved
        .bodies
        .into_iter()
        .map(|body| DeepseekV41LayerBody::build(&name, body, bridge))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut model = DeepseekV41VllmModel {
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
        layer0: DeepseekV41LayerBody::build(&name, resolved.layer0, bridge)?,
        bodies,
        plan: resolved.plan,
        head: DeepseekV41HeadTpWorklet::build(format!("{name}.head"), resolved.head, bridge)?,
        serialize_streams,
        total_kv_bytes_per_token: resolved.total_kv_bytes_per_token,
        cost_flat: Vec::new(),
        n_slots: 0,
        name,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

impl DeepseekV41VllmModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let mut sections = vec![self.prologue.compile(&mut builder)];
        // Hash -> layer-1 consumer: layer 0 plus the two contending Engram
        // lookups, which are minted after it.
        let main_path = self.layer0.compile(&mut builder);
        sections.push(self.engram_prefetch.compile_joined(&mut builder, main_path));
        sections.extend(self.compile_plan(&self.plan, &mut builder));
        sections.push(self.head.compile(&mut builder));
        let root = CostNode::Labeled {
            label: format!(
                "{} (DeepseekV41VllmModel) [TP4 attention, EP4 experts; {} unique bodies over 40 layers; serialize_streams={}]",
                self.name,
                self.bodies.len() + 1,
                self.serialize_streams,
            ),
            child: Box::new(CostNode::Sum(sections)),
        };
        builder.finish(root)
    }

    fn compile_plan(&self, plan: &[PlanNode], builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        plan.iter()
            .map(|node| match node {
                PlanNode::Body(index) => self.bodies[*index].compile(builder),
                PlanNode::Repeat { n, body } => {
                    let children = self.compile_plan(body, builder);
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

    fn eval_plan(&self, plan: &[PlanNode], input: &NormalizedInput, evaluator: &mut Evaluator) {
        for node in plan {
            match node {
                PlanNode::Body(index) => self.bodies[*index].eval(input, evaluator),
                PlanNode::Repeat { body, .. } => self.eval_plan(body, input, evaluator),
            }
        }
    }

    fn eval_into(&self, batch: &UnifiedArchInput, evaluator: &mut Evaluator) {
        let input = normalize_input(batch)
            .unwrap_or_else(|reason| panic!("invalid DeepSeek-V4.1 input: {reason}"));
        let tokens = DeepseekV41PrologueTpWorkletInput {
            num_tokens: input.rows,
        };
        self.prologue.eval(&tokens, evaluator);
        self.layer0.eval(&input, evaluator);
        self.engram_prefetch.eval(
            &DeepseekV41EngramPrefetchLocalWorkletInput {
                num_tokens: input.rows,
            },
            evaluator,
        );
        self.eval_plan(&self.plan, &input, evaluator);
        self.head.eval(
            &DeepseekV41HeadTpWorkletInput {
                num_tokens: input.rows,
                logits_rows: input.logits_rows,
            },
            evaluator,
        );
    }
}

impl IterwiseUnifiedModel for DeepseekV41VllmModel {
    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut evaluator = Evaluator::new(slots);
        self.eval_into(batch, &mut evaluator);
        debug_assert_eq!(evaluator.filled(), self.n_slots);
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
        let mut evaluator = Evaluator::with_inputs(slots, inputs);
        self.eval_into(batch, &mut evaluator);
        debug_assert_eq!(evaluator.filled(), self.n_slots);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    fn total_kv_bytes_per_token(&self) -> u64 {
        self.total_kv_bytes_per_token
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

struct NormalizedInput {
    /// Rows every dense launch runs: scheduled tokens padded to the CUDA-graph
    /// capture size.
    rows: u32,
    logits_rows: u32,
    attention: DeepseekV41AttentionTpWorkletInput,
}

/// The CUDA-graph size an iteration of `tokens` runs at (capture 2 server log
/// `cudagraph_capture_sizes`: 1, 2, 4, 8, then every 8 to 256, every 16 to
/// 2048). Larger iterations run eager at their own size.
pub fn cudagraph_rows(tokens: u32) -> u32 {
    match tokens {
        0 => 0,
        1..=2 => tokens,
        3..=4 => 4,
        5..=8 => 8,
        9..=256 => tokens.div_ceil(8) * 8,
        257..=MAX_CUDAGRAPH_CAPTURE_SIZE => tokens.div_ceil(16) * 16,
        _ => tokens,
    }
}

fn normalize_input(input: &UnifiedArchInput) -> std::result::Result<NormalizedInput, String> {
    if input.groups.len() != 1 {
        return Err(format!(
            "TP4 attention takes exactly one group, got {}",
            input.groups.len()
        ));
    }
    let group = &input.groups[0];
    let mut prefill_tokens = 0_u32;
    let mut prefill_query_context_pairs = Vec::with_capacity(group.prefill_chunk_pairs.len());
    for (request, &(prefix, append)) in group.prefill_chunk_pairs.iter().enumerate() {
        let context = prefix
            .checked_add(append)
            .ok_or_else(|| format!("prefill request {request} context overflows u32"))?;
        if append == 0 || context > MAX_MODEL_LEN {
            return Err(format!(
                "prefill request {request} ({prefix}, {append}) must append 1..=max_model_len"
            ));
        }
        prefill_tokens = prefill_tokens
            .checked_add(append)
            .ok_or("prefill token sum overflows u32")?;
        prefill_query_context_pairs.push((append, context));
    }
    let decode_tokens =
        u32::try_from(group.decode_kv_lens.len()).map_err(|_| "decode count exceeds u32")?;
    if group.prefill_tokens != prefill_tokens
        || group.decode_tokens != decode_tokens
        || group.batch_tokens != prefill_tokens + decode_tokens
    {
        return Err("group token accounting is inconsistent".into());
    }
    let rows = cudagraph_rows(group.batch_tokens);
    let decode_only = prefill_tokens == 0 && decode_tokens > 0;
    Ok(NormalizedInput {
        rows,
        logits_rows: group.request_count(),
        attention: DeepseekV41AttentionTpWorkletInput {
            num_tokens: rows,
            prefill_query_context_pairs,
            decode_kv_lens: group.decode_kv_lens.clone(),
            aux_stream_live: attention_aux_stream_live(decode_only, rows, MAX_FULL_GRAPH_TOKENS),
        },
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::arch::contract::ArchGroupInput;
    use crate::timing::LeafDesc;
    use crate::worklet::deepseek_v41_attention_tp::production_layer;
    use crate::worklet::deepseek_v41_common::SERIAL_COPY_SUFFIX;

    const CONFIG: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/model/config/deepseek_v41_flash.json"
    );

    fn spec() -> ModelSpec {
        ModelSpec {
            model_config: CONFIG.into(),
            num_layers: None,
            sim_num_layers: None,
            fp8: true,
        }
    }

    fn model_cfg() -> DeepseekV41ModelCfg {
        DeepseekV41ModelCfg::from_json(Path::new(CONFIG), &spec()).unwrap()
    }

    fn raw_json() -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(CONFIG).unwrap()).unwrap()
    }

    fn parallel() -> DeepseekV41VllmParallel {
        DeepseekV41VllmParallel {
            tp_size: 4,
            ep_size: 4,
            gpu_name: GPU_NAME.into(),
            serialize_streams: false,
        }
    }

    fn uniform_demand() -> ExpertDemand {
        ExpertDemand::Popularity {
            layerwise_global_ppm: vec![vec![1_000_000 / 384; 384]; 40],
        }
    }

    #[test]
    fn checkpoint_config_loads_with_the_exact_v41_identity() {
        let cfg = model_cfg();
        assert_eq!(cfg.num_layers, 40);
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(
            (cfg.num_experts, cfg.top_k, cfg.moe_intermediate_size),
            (384, 6, 2304)
        );
        assert_eq!(
            (cfg.q_lora_rank, cfg.index_num_heads, cfg.index_topk),
            (1280, 32, 512)
        );
        assert_eq!(cfg.compress_ratios.len(), 40);
        assert_eq!(cfg.engram_hash_heads, 24);
        assert_eq!(cfg.engram_layer_ids, [1, 14]);
    }

    #[test]
    fn config_drift_is_rejected() {
        for (field, value) in [
            ("hidden_size", serde_json::json!(4096)),
            ("num_hidden_layers", serde_json::json!(43)),
            ("n_routed_experts", serde_json::json!(256)),
            ("index_n_heads", serde_json::json!(64)),
            ("num_hash_layers", serde_json::json!(3)),
            ("kv_source_layer_ids", serde_json::json!([2, 8, 14])),
            ("index_source_layer_ids", serde_json::json!([2, 8, 14, 20])),
            ("candidate_source_layer_id", serde_json::json!(24)),
            ("engram_layer_ids", serde_json::json!([1])),
            ("engram_n_heads", serde_json::json!(4)),
        ] {
            let mut json = raw_json();
            json[field] = value;
            let raw: JsonDeepseekV41Config = serde_json::from_value(json).unwrap();
            assert!(
                DeepseekV41ModelCfg::from_raw(raw).is_err(),
                "{field} drift must be rejected"
            );
        }
        let mut json = raw_json();
        json["compress_ratios"][5] = serde_json::json!(1);
        let raw: JsonDeepseekV41Config = serde_json::from_value(json).unwrap();
        assert!(DeepseekV41ModelCfg::from_raw(raw).is_err());
        let mut truncated = spec();
        truncated.num_layers = Some(4);
        assert!(DeepseekV41ModelCfg::from_json(Path::new(CONFIG), &truncated).is_err());
    }

    #[test]
    fn config_layer_kinds_match_the_worklet_production_schedule() {
        let cfg = model_cfg();
        for layer in 0..40 {
            let kind = cfg.layer_kind(layer);
            assert_eq!(kind.attention, production_layer(layer), "layer {layer}");
            assert_eq!(kind.engram, [1, 14].contains(&layer), "layer {layer}");
        }
    }

    fn body_layers(plan: &[FoldPlan]) -> Vec<Vec<u32>> {
        plan.iter()
            .flat_map(|node| match node {
                FoldPlan::Body { layers, .. } => vec![layers.clone()],
                FoldPlan::Repeat { body, .. } => body_layers(body),
            })
            .collect()
    }

    #[test]
    fn layer_fold_keeps_order_and_never_merges_distinct_types() {
        let cfg = model_cfg();
        let tail: Vec<_> = (1..40).map(|l| (l, cfg.layer_kind(l))).collect();
        let fold = fold_layers(&tail);
        // Expansion reproduces the schedule exactly.
        let expanded: Vec<LayerKind> = fold.iter().flat_map(FoldPlan::kind_sequence).collect();
        let expected: Vec<LayerKind> = tail.iter().map(|&(_, kind)| kind).collect();
        assert_eq!(expanded, expected);
        // Every body stands only for layers of its own type.
        for node in &fold {
            fn check(node: &FoldPlan, cfg: &DeepseekV41ModelCfg) {
                match node {
                    FoldPlan::Body { kind, layers } => {
                        for &layer in layers {
                            assert_eq!(cfg.layer_kind(layer), *kind, "layer {layer}");
                        }
                    }
                    FoldPlan::Repeat { body, .. } => body.iter().for_each(|n| check(n, cfg)),
                }
            }
            check(node, &cfg);
        }
        // The expected shape: L1 | 2x[KV 2/8, 5x r2] | L14 | 5x r2 | L20 |
        // 4x[3x r1, consumer] | 3x r1.
        assert_eq!(
            body_layers(&fold),
            vec![
                vec![1],
                vec![2, 8],
                vec![3, 4, 5, 6, 7, 9, 10, 11, 12, 13],
                vec![14],
                vec![15, 16, 17, 18, 19],
                vec![20],
                vec![21, 22, 23, 25, 26, 27, 29, 30, 31, 33, 34, 35],
                vec![24, 28, 32, 36],
                vec![37, 38, 39],
            ]
        );
        assert!(matches!(&fold[1], FoldPlan::Repeat { n: 2, body } if body.len() == 2));
        assert!(matches!(&fold[5], FoldPlan::Repeat { n: 4, body } if body.len() == 2));
        assert!(matches!(&fold[6], FoldPlan::Repeat { n: 3, body } if body.len() == 1));
    }

    #[test]
    fn build_configs_threads_the_deployment_and_rejects_others() {
        let cfg = model_cfg();
        let configs = build_configs(&cfg, &parallel(), &uniform_demand()).unwrap();
        assert_eq!(configs.bodies.len(), 9);
        assert_eq!(
            configs.layer0.attention.layer.entry,
            DeepseekV41AttentionEntry::Layer0Pre
        );
        assert!(configs.layer0.engram.is_none());
        assert_eq!(
            configs.bodies.iter().filter(|b| b.engram.is_some()).count(),
            2
        );
        for body in configs.bodies.iter().chain([&configs.layer0]) {
            assert_eq!(body.attention.gpu_name, GPU_NAME);
            assert_eq!((body.attention.tp_size, body.ffn.ep_size), (4, 4));
            assert_eq!(body.ffn.folded_rank_position, 0);
            assert_eq!(body.ffn.expert_demand, uniform_demand());
        }
        assert_eq!(configs.head.vocab_size, 129_280);
        assert_eq!(configs.engram_prefetch.table_rows, ENGRAM_TABLE_ROWS_RANK0);
        // (288 + 132) / 2 x 3 + (288 + 132) x 1, replicated on 4 ranks.
        assert_eq!(configs.total_kv_bytes_per_token, (210 * 3 + 420) * 4);

        let resolved = resolve_configs(&configs);
        assert_eq!(resolved.head.lm_head.n.get(), 32_320);
        // Index pages: 64 keys at ratio 2 (production) and ratio 1 (fallback).
        for body in &resolved.bodies {
            if let Some(indexer) = &body.attention.indexer {
                assert_eq!(indexer.page_block_size, 64, "layers {:?}", body.layers);
            }
        }
        assert_eq!(
            resolved.bodies[0].engram.as_ref().unwrap().wkv.k.get(),
            6144
        );

        let mut h200 = parallel();
        h200.gpu_name = "NVIDIA H200".into();
        assert!(build_configs(&cfg, &h200, &uniform_demand()).is_err());
        let mut ep8 = parallel();
        ep8.ep_size = 8;
        assert!(build_configs(&cfg, &ep8, &uniform_demand()).is_err());
        let narrow = ExpertDemand::Popularity {
            layerwise_global_ppm: vec![vec![1_000_000 / 256; 256]],
        };
        assert!(build_configs(&cfg, &parallel(), &narrow).is_err());
    }

    fn built_tree() -> CostTree {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let configs = build_configs(&model_cfg(), &parallel(), &uniform_demand()).unwrap();
        build("unified".into(), resolve_configs(&configs), &bridge)
            .unwrap()
            .cost_tree()
    }

    /// How many times each slot runs per iteration (the product of enclosing
    /// `Scale` factors).
    fn slot_multiplicity(tree: &CostTree) -> Vec<u32> {
        fn walk(node: &CostNode, factor: u32, out: &mut Vec<u32>) {
            match node {
                CostNode::Leaf(slot) => out[*slot] = factor,
                CostNode::Sum(children) | CostNode::Max { children, .. } => {
                    children.iter().for_each(|c| walk(c, factor, out))
                }
                CostNode::Scale { n, child } => walk(child, factor * n, out),
                CostNode::Labeled { child, .. } => walk(child, factor, out),
            }
        }
        let mut out = vec![0; tree.slots.len()];
        walk(&tree.root, 1, &mut out);
        out
    }

    /// The folded tree launches exactly what the unfolded L3 iteration test
    /// (`worklet/deepseek_v41_iteration_tests.rs`) counts.
    #[test]
    fn folded_iteration_has_the_captured_launch_counts() {
        let tree = built_tree();
        let times = slot_multiplicity(&tree);
        assert!(times.iter().all(|&t| t > 0), "every slot is reachable");
        let serial = format!(".{SERIAL_COPY_SUFFIX}");
        let launches: Vec<_> = tree
            .slots
            .iter()
            .zip(&times)
            .filter(|(s, _)| !s.name.ends_with(&serial))
            .collect();
        let count = |kind: &str| -> u32 {
            launches
                .iter()
                .filter(|(s, _)| s.kind == kind)
                .map(|(_, &t)| t)
                .sum()
        };
        let named = |suffix: &str| -> u32 {
            launches
                .iter()
                .filter(|(s, _)| s.name.ends_with(suffix))
                .map(|(_, &t)| t)
                .sum()
        };
        assert_eq!(count("mhc_fused_post_pre_rms_norm"), 77);
        assert_eq!(count("all_reduce_fusion"), 81);
        assert_eq!(count("dsa_paged_mqa_logits_decode"), 8);
        assert_eq!(count("dsa_mqa_logits_prefill"), 8);
        assert_eq!(named(".decode_topk"), 8);
        assert_eq!(named(".prefill_topk"), 8);
        assert_eq!(named("compressor.kv_score_proj"), 4);
        assert_eq!(named("compressor.save_compress_norm"), 4);
        assert_eq!(named("compressor.nvfp4_insert"), 4);
        assert_eq!(count("engram_lookup"), 2);
        assert_eq!(count("deepseek_v41_qnorm_rope_kv_insert"), 40);
        assert_eq!(named("mega_attn.decode"), 40);
        assert_eq!(named("mega_attn.prefill"), 40);
        assert_eq!(count("nvfp4_fused_moe"), 40);
        assert_eq!(count("batched_gemm"), 40);
        assert_eq!(count("all_reduce"), 3);

        let n = |s: &LeafDesc| s.kernel_config["n"]["value"].as_u64().unwrap();
        let mut fp32 = BTreeMap::<u64, u32>::new();
        for (slot, &t) in launches
            .iter()
            .filter(|(s, _)| s.kind == "gemm_fp32_output")
        {
            *fp32.entry(n(slot)).or_default() += t;
        }
        assert_eq!(fp32, BTreeMap::from([(384, 40), (512, 1), (1024, 3)]));

        let mut mxfp8 = BTreeMap::<(u64, u64), u32>::new();
        for (slot, &t) in launches
            .iter()
            .filter(|(s, _)| s.kind == "single_gemm" && s.kernel_config["dtype"] == "mxfp8_e4m3")
        {
            let k = slot.kernel_config["k"]["value"].as_u64().unwrap();
            *mxfp8.entry((k, n(slot))).or_default() += t;
        }
        assert_eq!(
            mxfp8,
            BTreeMap::from([
                ((5120, 1792), 40),
                ((1280, 8192), 40),
                ((2048, 5120), 40),
                ((576, 5120), 40),
                ((5120, 1152), 40),
                ((1280, 4096), 8),
                ((6144, 25600), 2),
            ])
        );
        let lm_head: Vec<_> = launches
            .iter()
            .filter(|(s, _)| s.kind == "single_gemm" && s.kernel_config["dtype"] == "bf16")
            .collect();
        assert_eq!(lm_head.len(), 1);
        assert_eq!(*lm_head[0].1, 1);
        assert_eq!(n(lm_head[0].0), 32_320);
    }

    #[test]
    fn engram_lookups_are_minted_after_the_layer0_main_path() {
        let tree = built_tree();
        let position = |needle: &str| {
            tree.slots
                .iter()
                .position(|s| s.name.contains(needle))
                .unwrap_or_else(|| panic!("no slot named {needle}"))
        };
        let last_layer0 = tree
            .slots
            .iter()
            .rposition(|s| s.name.starts_with("unified.layer0."))
            .unwrap();
        assert!(position("engram_prefetch.layer1.engram_lookup") > last_layer0);
        assert!(
            position("engram_prefetch.layer14.engram_lookup") < position("unified.layer1.engram")
        );
        assert!(position("unified.prologue.engram_hash") < position("unified.layer0.attn"));
    }

    fn group(prefill: Vec<(u32, u32)>, decode: Vec<u32>) -> ArchGroupInput {
        let prefill_tokens = prefill.iter().map(|&(_, a)| a).sum::<u32>();
        let decode_tokens = decode.len() as u32;
        ArchGroupInput {
            batch_tokens: prefill_tokens + decode_tokens,
            prefill_tokens,
            decode_tokens,
            prefill_chunk_pairs: prefill,
            total_kv_len: decode.iter().sum(),
            decode_kv_lens: decode,
        }
    }

    #[test]
    fn input_pads_to_the_graph_size_and_gates_the_aux_stream() {
        let decode = normalize_input(&UnifiedArchInput {
            groups: vec![group(vec![], vec![1000; 48])],
            tokens_per_source_rank: vec![],
        })
        .unwrap();
        assert_eq!((decode.rows, decode.logits_rows), (48, 48));
        assert!(decode.attention.aux_stream_live);

        let mixed = normalize_input(&UnifiedArchInput {
            groups: vec![group(vec![(512, 126)], vec![900; 48])],
            tokens_per_source_rank: vec![],
        })
        .unwrap();
        assert_eq!((mixed.rows, mixed.logits_rows), (176, 49));
        assert!(!mixed.attention.aux_stream_live);
        assert_eq!(mixed.attention.prefill_query_context_pairs, [(126, 638)]);

        let big_decode = normalize_input(&UnifiedArchInput {
            groups: vec![group(vec![], vec![10; 65])],
            tokens_per_source_rank: vec![],
        })
        .unwrap();
        assert_eq!(big_decode.rows, 72);
        assert!(!big_decode.attention.aux_stream_live);

        assert!(normalize_input(&UnifiedArchInput {
            groups: vec![ArchGroupInput::default(), ArchGroupInput::default()],
            tokens_per_source_rank: vec![],
        })
        .is_err());
    }

    #[test]
    fn cudagraph_rows_follow_the_capture_sizes() {
        let sizes = [
            (1, 1),
            (3, 4),
            (5, 8),
            (48, 48),
            (49, 56),
            (174, 176),
            (257, 272),
            (2048, 2048),
            (2049, 2049),
        ];
        for (tokens, rows) in sizes {
            assert_eq!(cudagraph_rows(tokens), rows, "{tokens}");
        }
    }
}
