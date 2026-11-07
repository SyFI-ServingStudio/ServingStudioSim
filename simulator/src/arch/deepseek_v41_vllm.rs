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
//! Step shape (vLLM MRV2, [`normalize_step`]): a uniform decode batch replays
//! the FULL graph of its padded request count ([`full_decode_graph_requests`]),
//! any other batch the PIECEWISE graph of its padded token count, and batches
//! past the capture limit run eager. FULL graphs keep the attention aux stream
//! and, at most 16 rows, take the mHC overlap path at every fused mHC boundary.
//!
//! Decoder SWA bounded replay ([`BoundedReplay`], on by default): eager steps
//! with a long prefill chunk and PIECEWISE steps of 768+ rows run layer 20's KV
//! write over every row, then layers 20-39 on each request's last 128 rows
//! (gathered, then scattered back after the last hc post).
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
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

use crate::arch::config::ModelSpec;
use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::kv_layout::{EngineOverhead, KvCacheLayout, KvCachePart};
use crate::op::attention::DeepseekV41CandidateRole;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::all_reduce_fusion::fused_all_reduce_refusal;
use crate::timing::kernels::compressed_sparse_mla_rope_cast::{
    MXFP8_RECORD_BYTES, NVFP4_RECORD_BYTES,
};
use crate::timing::kernels::kv_compress_store::FP8_INDEXER_ROW_BYTES;
use crate::timing::kernels::{ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::deepseek_v41_common::{eval_or_zero, mhc_overlap_live, placeholder};
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
/// The one attention-TP and expert-parallel width the graph models, over the
/// same four ranks; the selectors take neither, and the builder reads both here.
pub(crate) const TP_SIZE: u32 = 4;
pub(crate) const EP_SIZE: u32 = 4;

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

// Deployment identity (capture 2 `profile.yaml` / server log). Its
// `--max-model-len 131072` is the selector's `max_model_len`, which the
// capture presets pin; see [`resolve_max_model_len`].
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
/// `--max-num-seqs 64`: the most requests a FULL decode graph holds.
pub(crate) const MAX_NUM_SEQS: u32 = 64;
/// `max_cudagraph_capture_size 2048`: the piecewise graph sizes run by every
/// iteration up to 2048 tokens pad the dense spine to the next capture size.
const MAX_CUDAGRAPH_CAPTURE_SIZE: u32 = 2048;
/// Rows of TP rank 0's Engram table slice (`part_num_embeddings`, server log
/// `engram.py:269`, layer 1): the lookup kind's exact table identity.
const ENGRAM_TABLE_ROWS_RANK0: u64 = 96_000_564;
/// `wkv` output width of the Engram block (captured MXFP8 GEMM 6144 -> 25600).
const ENGRAM_WKV_OUT: u32 = 25_600;

pub(crate) const MHC_BACKENDS: &[&str] = &["deepgemm_mega"];
pub(crate) const MHC_TILELANG_BACKENDS: &[&str] = &["vllm_tilelang"];
pub(crate) const MXFP8_GEMM_BACKENDS: &[&str] = &["flashinfer_mxfp8"];
pub(crate) const FP32_GEMM_BACKENDS: &[&str] = &["torch_cublas"];
pub(crate) const KV_INSERT_BACKENDS: &[&str] = &["vllm_cuda"];
pub(crate) const MEGA_ATTN_BACKENDS: &[&str] = &["flashmla_mega"];
pub(crate) const WO_A_BACKENDS: &[&str] = &["deepgemm_mxfp8_einsum_grouped_o_proj"];
pub(crate) const ALL_REDUCE_BACKENDS: &[&str] = &["flashinfer_mnnvl"];
pub(crate) const ALL_GATHER_PROXY_BACKENDS: &[&str] = &["nccl"];
pub(crate) const INDEX_LOGITS_BACKENDS: &[&str] = &["deepgemm_fp8"];
pub(crate) const ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
pub(crate) const MOE_BACKENDS: &[&str] = &["flashinfer_trtllm_sm100_mxfp4"];
pub(crate) const ENGRAM_LOOKUP_BACKENDS: &[&str] = &["vllm_triton"];
pub(crate) const LM_HEAD_BACKENDS: &[&str] = &["torch_linear"];

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
    /// The longest context the checkpoint is trained for, and vLLM's default
    /// `--max-model-len`.
    pub max_position_embeddings: u32,
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
            max_position_embeddings: raw.max_position_embeddings,
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
    /// Decoder SWA bounded replay (vLLM default on; see [`BoundedReplay`]).
    pub decoder_swa_bounded_replay: bool,
    /// Resolved `--max-model-len` (see [`resolve_max_model_len`]).
    pub max_model_len: u32,
}

/// The selector's `max_model_len`, defaulted to the checkpoint's
/// `max_position_embeddings` as vLLM does. vLLM refuses a larger value without
/// `VLLM_ALLOW_LONG_MAX_MODEL_LEN`, and nothing past the trained positions is
/// modeled, so neither is it accepted here.
///
/// The value is a timing input, not only an admission bound: the FlashMLA
/// prefill planner sizes its compressed-KV workspace for `max_model_len /
/// ratio` keys and the indexer's decode logits rows are padded to it, so the
/// capture presets pin the captured 131072 to reproduce its rows.
pub(crate) fn resolve_max_model_len(
    requested: Option<u32>,
    max_position_embeddings: u32,
) -> Result<u32> {
    let max_model_len = requested.unwrap_or(max_position_embeddings);
    ensure!(
        (1..=max_position_embeddings).contains(&max_model_len),
        "{ARCH_KIND}: max_model_len must be 1..={max_position_embeddings} \
         (the checkpoint's max_position_embeddings), got {max_model_len}"
    );
    Ok(max_model_len)
}

/// vLLM's decoder-side SWA bounded replay (fork
/// `models/deepseek_v41/decoder_replay_layers.py`, `nvidia/model.py`
/// `decoder_replay_start`, `nvidia/model_state.py` `_prepare_replay_batch`),
/// on by default (`cache_config.swa_bounded_replay`).
///
/// The cut is the last KV-source layer (`max(kv_source_layer_ids)`, layer 20
/// on V4.1-Flash). Layers past it own only sliding-window KV, so on a replay
/// step the cut layer first writes its KV for every row (`write_kv`: entry
/// mHC, input projections, norms and the KV/compressor/index-K stores, no
/// attention), then the cut layer and every later layer run on each
/// request's last `min(query_len, window)` rows only, gathered into a replay
/// batch and scattered back to batch rows after the last layer's hc post.
///
/// Which steps replay ([`replay_rows`]): FULL decode graphs and PIECEWISE
/// graphs under `trim_threshold` padded tokens run the whole batch; PIECEWISE
/// graphs at or above it always break out to the replay batch; eager steps
/// replay when some prefilling request has more than `window` query rows.
/// The replay batch runs in a PIECEWISE replay graph sized to a multiple of
/// `window` up to the capture limit, eager above it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundedReplay {
    /// First layer of the replay batch; it also runs `write_kv` on every row.
    pub cut_layer: u32,
    /// Rows kept per request (`sliding_window`, 128).
    pub window: u32,
    /// `decoder_replay_trim_threshold`: PIECEWISE graphs of this many padded
    /// tokens or more break out to the replay batch.
    pub trim_threshold: u32,
}

/// `decoder_replay_trim_threshold` of the fork's MRV2 model state.
pub const DECODER_REPLAY_TRIM_THRESHOLD: u32 = 768;

impl BoundedReplay {
    /// The cut and the fork's preconditions (`_decoder_replay_supported`, the
    /// `cut < end_layer - 1` and `compress_ratio <= 1` asserts): a layer past
    /// the cut, no Engram layer at or after it, and a cut whose compressed KV
    /// (if any) is written row by row.
    pub fn from_model(model: &DeepseekV41ModelCfg) -> std::result::Result<Self, BuildError> {
        let cut_layer =
            *model.kv_source_layer_ids.iter().max().ok_or_else(|| {
                fit_failed("decoder SWA bounded replay needs kv_source_layer_ids")
            })?;
        if cut_layer + 1 >= model.num_layers {
            return Err(fit_failed(format!(
                "decoder SWA bounded replay: cut layer {cut_layer} leaves no layer after it"
            )));
        }
        if model.compress_ratios[cut_layer as usize] > 1 {
            return Err(fit_failed(format!(
                "decoder SWA bounded replay: cut layer {cut_layer} compresses at ratio {}",
                model.compress_ratios[cut_layer as usize]
            )));
        }
        if let Some(layer) = model.engram_layer_ids.iter().find(|&&l| l >= cut_layer) {
            return Err(fit_failed(format!(
                "decoder SWA bounded replay: Engram layer {layer} sits at or after the cut \
                 layer {cut_layer}"
            )));
        }
        Ok(Self {
            cut_layer,
            window: model.sliding_window,
            trim_threshold: DECODER_REPLAY_TRIM_THRESHOLD,
        })
    }

    /// Whether a body runs on the replay batch. A body may not straddle the cut.
    fn is_late(&self, layers: &[u32]) -> std::result::Result<bool, BuildError> {
        let late = layers.iter().filter(|&&l| l >= self.cut_layer).count();
        match late {
            0 => Ok(false),
            n if n == layers.len() => Ok(true),
            _ => Err(fit_failed(format!(
                "body for layers {layers:?} straddles the replay cut layer {}",
                self.cut_layer
            ))),
        }
    }
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
    pub kv_cache_layout: KvCacheLayout,
    /// `Some` with `decoder_swa_bounded_replay` on.
    pub replay: Option<DeepseekV41ReplayConfig>,
    /// Per body: evaluated on the replay batch (all layers at or past the cut).
    pub late_bodies: Vec<bool>,
}

/// The bounded-replay boundary: the cut layer's `write_kv` over every row,
/// the gather into the replay batch, and the scatter of its outputs back.
#[derive(Clone)]
pub struct DeepseekV41ReplayConfig {
    pub rule: BoundedReplay,
    /// Index into `bodies` of the cut layer's body (exactly the cut layer).
    pub cut_body: usize,
    /// The cut layer's attention config, run with `write_kv_only`.
    pub write_kv: DeepseekV41AttentionTpWorkletConfig,
    /// `index_select` of the layer inputs (hidden, residual streams, mixes,
    /// positions, ids) by the replay rows, per replay row.
    pub gather: ElementwiseKernelConfig,
    /// `new_zeros(..).index_copy_` of the outputs (hc streams, pre-mix, any
    /// aux hidden states) back to batch rows, per batch row. A drafter that
    /// captures late aux hidden states widens it.
    pub scatter: ElementwiseKernelConfig,
}

impl DeepseekV41ReplayConfig {
    fn new(
        rule: BoundedReplay,
        cut_body: usize,
        write_kv: DeepseekV41AttentionTpWorkletConfig,
        model: &DeepseekV41ModelCfg,
        gpu: &str,
    ) -> Self {
        let (hidden, hc) = (model.hidden_size, model.hc_mult);
        // hidden (bf16) + residual hc streams (bf16) + pre/post (hc) and res
        // (hc x hc) fp32 mixes + int64 position + int32 id.
        let gathered = hidden * 2 + hc * hidden * 2 + (2 * hc + hc * hc) * 4 + 8 + 4;
        // hc streams (bf16) + the last FFN's pre-mix (hc fp32).
        let scattered = hc * hidden * 2 + hc * 4;
        Self {
            rule,
            cut_body,
            write_kv,
            gather: placeholder(ELEMENTWISE_BACKENDS, gpu, gathered, gathered),
            scatter: placeholder(ELEMENTWISE_BACKENDS, gpu, scattered, scattered),
        }
    }
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
    kv_cache_layout: KvCacheLayout,
    replay: Option<DeepseekV41ReplayResolved>,
    late_bodies: Vec<bool>,
}

pub struct DeepseekV41ReplayResolved {
    rule: BoundedReplay,
    cut_body: usize,
    write_kv: DeepseekV41AttentionTpWorkletResolved,
    gather: ElementwiseKernelConfig,
    scatter: ElementwiseKernelConfig,
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
    // The graph's per-rank shapes (padded heads, Engram table slices, local
    // experts) are written for TP4 attention + EP4 experts on the same ranks.
    if parallel.tp_size != TP_SIZE || parallel.ep_size != EP_SIZE {
        return Err(fit_failed(format!(
            "the DeepSeek-V4.1-Flash graph is TP{TP_SIZE} attention + EP{EP_SIZE} experts, \
             got TP{} + EP{}",
            parallel.tp_size, parallel.ep_size
        )));
    }
    // The GPU is not pinned: a GPU without rows is a profile.db gap the
    // cache reports. Only the fused all-reduce needs vLLM's FlashInfer budget.
    if let Some(reason) = fused_all_reduce_refusal(ALL_REDUCE_BACKENDS, &parallel.gpu_name, TP_SIZE)
    {
        return Err(fit_failed(reason));
    }
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
    let (replay, late_bodies) = if parallel.decoder_swa_bounded_replay {
        let rule = BoundedReplay::from_model(model)?;
        let late_bodies = bodies
            .iter()
            .map(|body: &DeepseekV41LayerConfig| rule.is_late(&body.layers))
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
        kv_cache_layout: kv_cache_layout(model, parallel.tp_size, 0),
        replay,
        late_bodies,
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

/// One request's cache over the TP group, in vLLM's packed groups (fork
/// `v1/kv_cache_interface.py`, BLHNC layout). Each rank holds the one KV head,
/// so every part is one rank's bytes replicated on each of the `tp_size` ranks,
/// and so is the null block.
///
/// Per 128-token block, the full-attention group packs every KV-source layer's
/// NVFP4 compressed page (288 B per `ratio` tokens) and each owned indexer FP8
/// key page (132 B per `ratio` tokens), each page rounded up to the 512 B TMA
/// stride: with sources 2/8/14 at ratio 2 and 20 at ratio 1 that is 135,168 B
/// per rank (the ratio-2 index pages round 8448 B up to 8704 B). This
/// unaligned block sets how many layers' pages share one block
/// (`states_per_block`). With an MXFP8 KV cache, FlashMLA reads the
/// interleaved SWA (528 B) and compressed (288 B) records as rows of the whole
/// cache, so the block stride is a multiple of `lcm(528, 288)` = 3168 B and of
/// the 512 B page (`attention.py` `packed_block_stride_alignment`): 50,688 B,
/// which rounds the pool's block to 152,064 B. Every block of every group pays
/// that stride; the full-attention parts carry the rounding per token.
///
/// Every layer keeps a 128-token sliding window of 528 B MXFP8 rows in 32-token
/// blocks, eight layers to a group; `extra_swa_layers` adds a drafter's
/// sliding-window-only layers (DSpark: 3) to the same groups. vLLM's SWA
/// bounded replay (on by default) recomputes the window after a prefix hit
/// instead of reusing it. Each ratio-2 source's compressor keeps a ring of
/// eight fp32 [kv, score] rows, one block per request, also never reused.
/// vLLM reserves one null block.
pub(crate) fn kv_cache_layout(
    model: &DeepseekV41ModelCfg,
    tp_size: u32,
    extra_swa_layers: u32,
) -> KvCacheLayout {
    const BLOCK_TOKENS: u64 = KV_BLOCK_SIZE as u64;
    const PAGE_ALIGN: u64 = 512;
    const RING_ROWS: u64 = 8;
    const FP32: u64 = 4;
    let page = |entries: u64, bytes: u64| (entries * bytes).div_ceil(PAGE_ALIGN) * PAGE_ALIGN;
    let (record, index_row, swa_row) = (
        u64::from(NVFP4_RECORD_BYTES),
        u64::from(FP8_INDEXER_ROW_BYTES),
        u64::from(MXFP8_RECORD_BYTES),
    );
    let sources_at = |ratio: u32, indexed: bool| {
        model
            .kv_source_layer_ids
            .iter()
            .filter(|&&layer| model.compress_ratios[layer as usize] == ratio)
            .filter(|layer| !indexed || model.index_source_layer_ids.contains(layer))
            .count() as u64
    };

    // The unaligned full-attention block (the packing anchor) and its stride.
    let mut block = 0;
    let mut full_parts = Vec::new();
    for (ratio, kv_name, index_name) in [(2, "c2_kv", "c2_indexer"), (1, "c1_kv", "c1_indexer")] {
        let entries = BLOCK_TOKENS / u64::from(ratio);
        let (kv_layers, index_layers) = (sources_at(ratio, false), sources_at(ratio, true));
        let (kv_page, index_page) = (page(entries, record), page(entries, index_row));
        block += kv_layers * kv_page + index_layers * index_page;
        full_parts.push((
            kv_name,
            ratio,
            kv_layers * record,
            kv_layers * kv_page / entries,
        ));
        full_parts.push((
            index_name,
            ratio,
            index_layers * index_row,
            index_layers * index_page / entries,
        ));
    }
    let stride_align = lcm(PAGE_ALIGN, lcm(swa_row, record));
    let stride = block.div_ceil(stride_align) * stride_align;
    // The stride rounding, per token, carried by the last ratio-1 part.
    let rounding = stride - block;
    assert_eq!(
        rounding % BLOCK_TOKENS,
        0,
        "block stride rounding must split per token"
    );
    if let Some(last) = full_parts.last_mut() {
        last.3 += rounding / BLOCK_TOKENS;
    }

    let tp = tp_size;
    let mut layout = KvCacheLayout::new();
    for (name, ratio, bytes, padded) in full_parts {
        layout = layout.with_part(
            KvCachePart::compressed(name, ratio, bytes)
                .padded_to(padded)
                .replicated(tp),
        );
    }
    // A kind whose layers' pages fill whole groups pays a stride per group.
    let groups = |pages: u64| pages.div_ceil(block);

    let swa_layers = model.compress_ratios.len() as u64 + u64::from(extra_swa_layers);
    let swa_block = u64::from(SWA_BLOCK_SIZE);
    let swa_groups = groups(swa_layers * page(swa_block, swa_row));
    let ring_layers = model
        .kv_source_layer_ids
        .iter()
        .filter(|&&layer| model.compress_ratios[layer as usize] > 1)
        .count() as u64;
    let ring_layer_bytes = RING_ROWS * 2 * u64::from(model.head_dim) * FP32;
    layout
        .with_part(
            KvCachePart::sliding_window("swa", model.sliding_window, swa_layers * swa_row)
                .padded_to(swa_groups * stride / swa_block)
                .not_prefix_reusable()
                .replicated(tp),
        )
        .with_part(
            KvCachePart::per_request("compressor_ring", ring_layers * ring_layer_bytes)
                .padded_to(groups(ring_layers * page(1, ring_layer_bytes)) * stride)
                .not_prefix_reusable()
                .replicated(tp),
        )
        .with_engine(EngineOverhead {
            block_tokens: Some(KV_BLOCK_SIZE),
            fixed_device_bytes: stride * u64::from(tp),
        })
}

fn lcm(a: u64, b: u64) -> u64 {
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    a / gcd(a, b) * b
}

pub(crate) fn layer_config(
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
            max_model_len: parallel.max_model_len,
            max_num_batched_tokens: MAX_BATCHED_TOKENS,
            prefill_chunk_size: PREFILL_CHUNK_SIZE,
            gpu_name: gpu.clone(),
            mhc_backends: MHC_BACKENDS.to_vec(),
            mhc_tilelang_backends: MHC_TILELANG_BACKENDS.to_vec(),
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
            mhc_tilelang_backends: MHC_TILELANG_BACKENDS.to_vec(),
            router_backends: FP32_GEMM_BACKENDS.to_vec(),
            gemm_backends: MXFP8_GEMM_BACKENDS.to_vec(),
            moe_backends: MOE_BACKENDS.to_vec(),
            all_reduce_backends: ALL_REDUCE_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
            moe_input_dtype: DType::Bf16,
            weight_format: DType::Mxfp4E2m1,
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
        kv_cache_layout: configs.kv_cache_layout.clone(),
        replay: configs
            .replay
            .as_ref()
            .map(|replay| DeepseekV41ReplayResolved {
                rule: replay.rule,
                cut_body: replay.cut_body,
                write_kv: DeepseekV41AttentionTpWorklet::resolve_config(&replay.write_kv),
                gather: replay.gather.clone(),
                scatter: replay.scatter.clone(),
            }),
        late_bodies: configs.late_bodies.clone(),
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

    fn eval(&self, input: &LayerInput, evaluator: &mut Evaluator) {
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
                mhc_overlap: input.attention.mhc_overlap,
            },
            evaluator,
        );
    }
}

/// The built bounded-replay boundary ([`DeepseekV41ReplayConfig`]).
struct DeepseekV41ReplayBoundary {
    cut_body: usize,
    write_kv: DeepseekV41AttentionTpWorklet,
    gather: Op<ElementwiseKernel>,
    scatter: Op<ElementwiseKernel>,
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
    /// Longest request context; longer inputs are refused.
    max_model_len: u32,
    kv_cache_layout: KvCacheLayout,
    replay_rule: Option<BoundedReplay>,
    replay: Option<DeepseekV41ReplayBoundary>,
    late_bodies: Vec<bool>,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: DeepseekV41VllmResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<DeepseekV41VllmModel, BuildError> {
    let serialize_streams = resolved.layer0.attention.raw_cfg.serialize_streams;
    let max_model_len = resolved.layer0.attention.raw_cfg.max_model_len;
    let bodies = resolved
        .bodies
        .into_iter()
        .map(|body| DeepseekV41LayerBody::build(&name, body, bridge))
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
            Some(DeepseekV41ReplayBoundary {
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
        max_model_len,
        kv_cache_layout: resolved.kv_cache_layout,
        replay_rule,
        replay,
        late_bodies: resolved.late_bodies,
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
        let sections = self.compile_sections(&mut builder);
        let root = CostNode::Labeled {
            label: format!(
                "{} (DeepseekV41VllmModel) [TP4 attention, EP4 experts; {} unique bodies over 40 layers; bounded_replay={}; serialize_streams={}]",
                self.name,
                self.bodies.len() + 1,
                self.replay.is_some(),
                self.serialize_streams,
            ),
            child: Box::new(CostNode::Sum(sections)),
        };
        builder.finish(root)
    }

    /// The target forward, prologue to head, in evaluation order; a wrapper
    /// (the DSpark drafter) sums these with its own sections.
    pub(crate) fn compile_sections(&self, builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut sections = vec![self.prologue.compile(builder)];
        // Hash -> layer-1 consumer: layer 0 plus the two contending Engram
        // lookups, which are minted after it.
        let main_path = self.layer0.compile(builder);
        sections.push(self.engram_prefetch.compile_joined(builder, main_path));
        sections.extend(self.compile_plan(&self.plan, builder));
        if let Some(replay) = &self.replay {
            sections.push(replay.scatter.compile(builder));
        }
        sections.push(self.head.compile(builder));
        sections
    }

    fn compile_plan(&self, plan: &[PlanNode], builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut out = Vec::with_capacity(plan.len());
        for node in plan {
            match node {
                PlanNode::Body(index) => {
                    if let Some(replay) = self.replay.as_ref().filter(|r| r.cut_body == *index) {
                        out.push(replay.write_kv.compile(builder));
                        out.push(replay.gather.compile(builder));
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

    fn eval_plan(&self, plan: &[PlanNode], input: &NormalizedInput, evaluator: &mut Evaluator) {
        for node in plan {
            match node {
                PlanNode::Body(index) => {
                    if let Some(replay) = self.replay.as_ref().filter(|r| r.cut_body == *index) {
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
                                mhc_overlap: false,
                                write_kv_only: true,
                            }
                        };
                        replay.write_kv.eval(&write_kv, evaluator);
                        let gathered = input.replay.as_ref().map_or(0, |late| late.rows);
                        eval_or_zero(
                            &replay.gather,
                            ElementwiseKernelInput {
                                num_tokens: gathered,
                            },
                            gathered == 0,
                            evaluator,
                        );
                    }
                    let body_input = match &input.replay {
                        Some(late) if self.late_bodies[*index] => late,
                        _ => &input.full,
                    };
                    self.bodies[*index].eval(body_input, evaluator)
                }
                PlanNode::Repeat { body, .. } => self.eval_plan(body, input, evaluator),
            }
        }
    }

    /// Evaluate the target forward's sections (see [`Self::compile_sections`]).
    pub(crate) fn eval_sections(&self, input: &NormalizedInput, evaluator: &mut Evaluator) {
        let rows = input.full.rows;
        self.prologue.eval(
            &DeepseekV41PrologueTpWorkletInput { num_tokens: rows },
            evaluator,
        );
        self.layer0.eval(&input.full, evaluator);
        self.engram_prefetch.eval(
            &DeepseekV41EngramPrefetchLocalWorkletInput { num_tokens: rows },
            evaluator,
        );
        self.eval_plan(&self.plan, input, evaluator);
        if let Some(replay) = &self.replay {
            let scattered = if input.replay.is_some() { rows } else { 0 };
            eval_or_zero(
                &replay.scatter,
                ElementwiseKernelInput {
                    num_tokens: scattered,
                },
                scattered == 0,
                evaluator,
            );
        }
        self.head.eval(
            &DeepseekV41HeadTpWorkletInput {
                num_tokens: rows,
                mhc_post_rows: input.replay.as_ref().map_or(rows, |late| late.rows),
                logits_rows: input.logits_rows,
            },
            evaluator,
        );
    }

    /// The step a no-spec batch runs: every decode request is one query row.
    pub(crate) fn normalize(&self, batch: &UnifiedArchInput) -> Result<NormalizedInput, String> {
        let step = DeepseekV41Step::from_unified(batch, self.max_model_len)?;
        let logits_rows = step.request_count();
        Ok(normalize_step(
            &step,
            1,
            logits_rows,
            self.replay_rule.as_ref(),
        ))
    }

    pub fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    pub fn replay_rule(&self) -> Option<&BoundedReplay> {
        self.replay_rule.as_ref()
    }

    fn eval_into(&self, batch: &UnifiedArchInput, evaluator: &mut Evaluator) {
        let input = self
            .normalize(batch)
            .unwrap_or_else(|reason| panic!("invalid DeepSeek-V4.1 input: {reason}"));
        self.eval_sections(&input, evaluator);
    }
}

impl IterwiseUnifiedModel for DeepseekV41VllmModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> std::result::Result<(), String> {
        self.normalize(batch).map(|_| ())
    }

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

/// How vLLM's MRV2 runner launches a step (`v1/worker/gpu/cudagraph_utils.py`
/// `CudaGraphManager.dispatch`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphMode {
    /// A uniform decode batch replays the FULL graph of its padded request count.
    Full,
    /// Any other batch up to the capture limit replays the PIECEWISE graph of
    /// its padded token count.
    Piecewise,
    /// Larger batches run eager at their own size.
    Eager,
}

/// The token-keyed (PIECEWISE) graph sizes: `prefill_cudagraph_capture_token_sizes`
/// in the server log, 1, 2, 4, then every 8 to 256 and every 16 to 2048.
fn piecewise_graph_sizes() -> impl Iterator<Item = u32> {
    [1, 2, 4]
        .into_iter()
        .chain((8..=256).step_by(8))
        .chain((272..=MAX_CUDAGRAPH_CAPTURE_SIZE).step_by(16))
}

/// The padded request count of the FULL decode graph a uniform batch of
/// `requests` requests with `query_len` rows each replays, or `None` when no
/// FULL graph holds it.
///
/// vLLM captures one FULL decode graph per request count in
/// `decode_cudagraph_capture_request_sizes` = `{min(cdiv(s, q), max_num_seqs)}`
/// over the shared token sizes `s` with `cdiv(s, q) * q` within the capture
/// limit (`config/vllm.py`), and a batch takes the smallest one that holds it.
/// For `q = 1` these are the PIECEWISE sizes up to 64 (11 graphs); for the
/// DSpark verify width `q = 6`, 41 graphs (1, 2, 3, 4, 6, 7, 8, ..., 59, 62, 64).
pub(crate) fn full_decode_graph_requests(requests: u32, query_len: u32) -> Option<u32> {
    if requests == 0 || query_len == 0 || requests > MAX_NUM_SEQS {
        return None;
    }
    piecewise_graph_sizes()
        .filter(|&s| s.div_ceil(query_len) * query_len <= MAX_CUDAGRAPH_CAPTURE_SIZE)
        .map(|s| s.div_ceil(query_len).min(MAX_NUM_SEQS))
        .find(|&n| n >= requests)
}

/// The graph a step of `query_lens` (one entry per request) launches, and the
/// rows its dense spine runs. A step is a uniform decode when every request
/// has exactly `uniform_query_len` rows (1, or the verify width).
pub(crate) fn step_graph(query_lens: &[u32], uniform_query_len: u32) -> (GraphMode, u32) {
    let tokens: u32 = query_lens.iter().sum();
    let requests = query_lens.len() as u32;
    if query_lens.iter().all(|&q| q == uniform_query_len) {
        if let Some(padded) = full_decode_graph_requests(requests, uniform_query_len) {
            return (GraphMode::Full, padded * uniform_query_len);
        }
    }
    if tokens <= MAX_CUDAGRAPH_CAPTURE_SIZE {
        (GraphMode::Piecewise, cudagraph_rows(tokens))
    } else {
        (GraphMode::Eager, tokens)
    }
}

/// The rows a replay batch of `kept` tokens runs at: the smallest replay graph
/// (`decoder_replay_cudagraph.py`: multiples of `window` up to
/// `min(max_cudagraph_capture_size, max_num_seqs * window)`), eager above.
pub(crate) fn replay_graph_rows(kept: u32, window: u32) -> u32 {
    let bound = MAX_CUDAGRAPH_CAPTURE_SIZE.min(MAX_NUM_SEQS * window);
    if kept <= bound {
        kept.div_ceil(window) * window
    } else {
        kept
    }
}

/// One step's requests as the target sees them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeepseekV41Step {
    /// `(query_len, context_len)` per prefilling request, context including
    /// the query.
    pub prefill: Vec<(u32, u32)>,
    /// `(query_len, kv_len)` per decode request: 1 row without speculation,
    /// the verify width with it; `kv_len` includes every query row.
    pub decode: Vec<(u32, u32)>,
}

impl DeepseekV41Step {
    pub fn request_count(&self) -> u32 {
        (self.prefill.len() + self.decode.len()) as u32
    }

    /// A no-spec batch: each decode request is one row after its resident KV.
    pub fn from_unified(input: &UnifiedArchInput, max_model_len: u32) -> Result<Self, String> {
        if input.groups.len() != 1 {
            return Err(format!(
                "TP4 attention takes exactly one group, got {}",
                input.groups.len()
            ));
        }
        let group = &input.groups[0];
        let prefill = prefill_pairs(&group.prefill_chunk_pairs, max_model_len)?;
        let prefill_tokens: u32 = prefill.iter().map(|&(q, _)| q).sum();
        let decode_tokens =
            u32::try_from(group.decode_kv_lens.len()).map_err(|_| "decode count exceeds u32")?;
        if group.prefill_tokens != prefill_tokens
            || group.decode_tokens != decode_tokens
            || group.batch_tokens != prefill_tokens + decode_tokens
        {
            return Err("group token accounting is inconsistent".into());
        }
        let decode = group
            .decode_kv_lens
            .iter()
            .map(|&kv| {
                kv.checked_add(1)
                    .map(|kv| (1, kv))
                    .ok_or_else(|| "decode kv length overflows u32".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { prefill, decode })
    }
}

/// `(prefix, append)` chunks as `(query, context)` pairs, bounded by
/// `max_model_len`.
pub(crate) fn prefill_pairs(
    chunks: &[(u32, u32)],
    max_model_len: u32,
) -> Result<Vec<(u32, u32)>, String> {
    let mut tokens = 0_u32;
    chunks
        .iter()
        .enumerate()
        .map(|(request, &(prefix, append))| {
            let context = prefix
                .checked_add(append)
                .ok_or_else(|| format!("prefill request {request} context overflows u32"))?;
            if append == 0 || context > max_model_len {
                return Err(format!(
                    "prefill request {request} ({prefix}, {append}) must append \
                     1..=max_model_len {max_model_len}"
                ));
            }
            tokens = tokens
                .checked_add(append)
                .ok_or("prefill token sum overflows u32")?;
            Ok((append, context))
        })
        .collect()
}

/// One layer group's input: the rows its dense launches run and its attention
/// batch (whose `mhc_overlap` also gates the FFN's boundary).
#[derive(Clone, Debug)]
pub struct LayerInput {
    pub rows: u32,
    pub attention: DeepseekV41AttentionTpWorkletInput,
}

/// A step, normalized: the batch the layers before the replay cut (and the
/// whole forward when nothing replays) run, the replay batch, and the logits
/// rows.
#[derive(Clone, Debug)]
pub struct NormalizedInput {
    pub mode: GraphMode,
    pub full: LayerInput,
    /// `Some` when the step runs the layers from the cut on the replay batch.
    pub replay: Option<LayerInput>,
    pub logits_rows: u32,
}

/// The CUDA-graph size an iteration of `tokens` runs at (server log
/// `prefill_cudagraph_capture_token_sizes`: 1, 2, 4, 8, then every 8 to 256,
/// every 16 to 2048). Larger iterations run eager at their own size.
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

/// Decode requests as one-token attention rows: a request of `q` rows ending
/// at `kv` attends from resident lengths `kv - q .. kv - 1` (the decode
/// segment flattens each verify row into its own request).
fn decode_rows(decode: &[(u32, u32)]) -> Vec<u32> {
    decode
        .iter()
        .flat_map(|&(q, kv)| (0..q).map(move |i| kv - q + i))
        .collect()
}

fn layer_input(
    mode: GraphMode,
    rows: u32,
    prefill: Vec<(u32, u32)>,
    decode: &[(u32, u32)],
) -> LayerInput {
    let full_graph = mode == GraphMode::Full;
    LayerInput {
        rows,
        attention: DeepseekV41AttentionTpWorkletInput {
            num_tokens: rows,
            prefill_query_context_pairs: prefill,
            decode_kv_lens: decode_rows(decode),
            // The attention aux stream exists only outside breakable capture
            // (`utils/multi_stream_utils.py`), i.e. in FULL graphs.
            aux_stream_live: full_graph && rows > 0,
            mhc_overlap: mhc_overlap_live(full_graph, rows),
            write_kv_only: false,
        },
    }
}

/// Normalize a step: graph mode and padded rows, the replay batch
/// ([`BoundedReplay`]), and the attention batches. `uniform_query_len` is 1
/// without speculation and the verify width with it.
pub(crate) fn normalize_step(
    step: &DeepseekV41Step,
    uniform_query_len: u32,
    logits_rows: u32,
    replay: Option<&BoundedReplay>,
) -> NormalizedInput {
    let query_lens: Vec<u32> = step
        .prefill
        .iter()
        .chain(&step.decode)
        .map(|&(q, _)| q)
        .collect();
    let (mode, rows) = step_graph(&query_lens, uniform_query_len);
    let full = layer_input(mode, rows, step.prefill.clone(), &step.decode);
    let replay = replay
        .and_then(|rule| replay_rows(rule, mode, rows, step))
        .map(|(late_rows, prefill)| {
            layer_input(GraphMode::Piecewise, late_rows, prefill, &step.decode)
        });
    NormalizedInput {
        mode,
        full,
        replay,
        logits_rows,
    }
}

/// Whether a step replays (`_prepare_replay_batch`), and the replay batch's
/// rows and trimmed prefill pairs. Decode requests keep all their rows
/// (`min(q, window)` with `q <= window`).
pub(crate) fn replay_rows(
    rule: &BoundedReplay,
    mode: GraphMode,
    rows: u32,
    step: &DeepseekV41Step,
) -> Option<(u32, Vec<(u32, u32)>)> {
    let replays = match mode {
        GraphMode::Full => false,
        GraphMode::Piecewise => rows >= rule.trim_threshold,
        GraphMode::Eager => step.prefill.iter().any(|&(q, _)| q > rule.window),
    };
    if !replays {
        return None;
    }
    let prefill: Vec<(u32, u32)> = step
        .prefill
        .iter()
        .map(|&(q, context)| (q.min(rule.window), context))
        .collect();
    let kept: u32 = prefill.iter().map(|&(q, _)| q).sum::<u32>()
        + step
            .decode
            .iter()
            .map(|&(q, _)| q.min(rule.window))
            .sum::<u32>();
    Some((replay_graph_rows(kept, rule.window), prefill))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::arch::contract::ArchGroupInput;
    use crate::timing::LeafDesc;

    use crate::worklet::deepseek_v41_attention_tp::production_layer;
    use crate::worklet::deepseek_v41_common::SERIAL_COPY_SUFFIX;

    /// The GPU of the capture the profile rows were measured for.
    const GPU_NAME: &str = "NVIDIA B200";
    /// The capture's `--max-model-len`, which its profile rows are keyed on.
    const CAPTURED_MAX_MODEL_LEN: u32 = 131_072;
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
            decoder_swa_bounded_replay: true,
            max_model_len: CAPTURED_MAX_MODEL_LEN,
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
    fn kv_cache_layout_reproduces_vllms_packed_groups() {
        use crate::worker::kv::{KvAccounting, KvAccountingOptions};
        let layout = kv_cache_layout(&model_cfg(), 4, 0);
        let part = |name: &str| *layout.parts.iter().find(|part| part.name == name).unwrap();
        let padded = |name: &str| part(name).overhead.padded_bytes_per_entry.unwrap();
        // vLLM block of 128 tokens: 3 x (18432 + 8704) + (36864 + 16896)
        // = 135168 B per rank packs the groups; the MXFP8 stride alignment
        // lcm(528, 288, 512) = 50688 B rounds the block to 152064 B
        // (1188 B/token), replicated on 4 ranks.
        let (anchor, stride) = (135_168, 152_064);
        assert_eq!(layout.engine.fixed_device_bytes, 4 * stride);
        assert_eq!(
            padded("c2_kv") / 2 + padded("c2_indexer") / 2 + padded("c1_kv") + padded("c1_indexer"),
            4 * stride / 128
        );
        // Five SWA groups of eight layers (by the unaligned anchor), each
        // paying the aligned stride per 32-token block.
        assert_eq!(anchor / (32 * 528), 8);
        assert_eq!(padded("swa"), 4 * 5 * stride / 32);
        assert_eq!(part("swa").bytes_per_entry, 4 * 40 * 528);
        // Three compressor rings of 32 KiB share one block per request.
        assert_eq!(
            (
                part("compressor_ring").bytes_per_entry,
                padded("compressor_ring")
            ),
            (4 * 3 * 32_768, 4 * stride)
        );
        assert!(!part("swa").prefix_reusable && !part("compressor_ring").prefix_reusable);

        // The DSpark drafter's three SWA layers make 43: six SWA groups.
        let dspark = kv_cache_layout(&model_cfg(), 4, 3);
        let dspark_swa = *dspark.parts.iter().find(|part| part.name == "swa").unwrap();
        assert_eq!(dspark_swa.bytes_per_entry, 4 * 43 * 528);
        assert_eq!(
            dspark_swa.overhead.padded_bytes_per_entry.unwrap(),
            4 * 6 * stride / 32
        );
        // TP4 + DSpark capture (fork 5629b26146, 1017 checkpoint): 566,723
        // blocks (80.26 GiB) per rank logged "GPU KV cache size: 40,746,855
        // tokens, Maximum concurrency for 131,072 tokens per request: 310.87x".
        let options = KvAccountingOptions {
            max_in_flight_tokens: Some(2 * 2048),
            ..Default::default()
        };
        let accounting = KvAccounting::resolve(&dspark, options, 4 * 566_723 * stride).unwrap();
        let concurrency = accounting.concurrency_at(131_072);
        assert!(
            (concurrency / 310.87 - 1.0).abs() < 0.005,
            "{concurrency} vs 310.87"
        );
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
        // Three ratio-2 sources with their indexers and one ratio-1 source,
        // replicated on 4 ranks.
        assert_eq!(
            configs.kv_cache_layout.per_token_bytes(),
            (3 * (288 + 132) / 2 + 288 + 132) * 4
        );
        // Bounded replay is on by default, cut at the last KV source.
        let replay = configs.replay.as_ref().unwrap();
        assert_eq!(replay.rule.cut_layer, 20);
        assert_eq!(configs.bodies[replay.cut_body].layers, [20]);

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

        // No GPU pin; only a GPU vLLM's FlashInfer all-reduce has no budget for.
        let mut b300 = parallel();
        b300.gpu_name = "NVIDIA B300".into();
        assert!(build_configs(&cfg, &b300, &uniform_demand()).is_ok());
        let mut a100 = parallel();
        a100.gpu_name = "NVIDIA A100".into();
        assert!(build_configs(&cfg, &a100, &uniform_demand()).is_err());
        let mut ep8 = parallel();
        ep8.ep_size = 8;
        assert!(build_configs(&cfg, &ep8, &uniform_demand()).is_err());
        let narrow = ExpertDemand::Popularity {
            layerwise_global_ppm: vec![vec![1_000_000 / 256; 256]],
        };
        assert!(build_configs(&cfg, &parallel(), &narrow).is_err());
    }

    /// The forward without the bounded-replay boundary (its `write_kv`
    /// launches are extra; `replay_boundary_runs_only_on_replay_steps`).
    fn built_tree() -> CostTree {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let configs =
            build_configs(&model_cfg(), &no_replay_parallel(), &uniform_demand()).unwrap();
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
                CostNode::Sum(children)
                | CostNode::Max { children, .. }
                | CostNode::Parallel { children, .. } => {
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
            .filter(|(s, _)| !s.name.ends_with(&serial) && !s.name.contains(".mhc_overlap."))
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
        assert_eq!(count("q_pad_kv_rope_mxfp8_insert"), 40);
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
        let normalize = |prefill, decode| normalize_input(&one_group(prefill, decode), None);
        let decode = normalize(vec![], vec![1000; 48]).unwrap();
        assert_eq!(decode.mode, GraphMode::Full);
        assert_eq!((decode.full.rows, decode.logits_rows), (48, 48));
        assert!(decode.full.attention.aux_stream_live);
        assert!(!decode.full.attention.mhc_overlap);
        // The decode segment sees each row's resident KV.
        assert_eq!(decode.full.attention.decode_kv_lens, vec![1000; 48]);

        let mixed = normalize(vec![(512, 126)], vec![900; 48]).unwrap();
        assert_eq!(mixed.mode, GraphMode::Piecewise);
        assert_eq!((mixed.full.rows, mixed.logits_rows), (176, 49));
        assert!(!mixed.full.attention.aux_stream_live);
        assert_eq!(
            mixed.full.attention.prefill_query_context_pairs,
            [(126, 638)]
        );

        // 65 decodes exceed max_num_seqs: no FULL graph holds them.
        let big_decode = normalize(vec![], vec![10; 65]).unwrap();
        assert_eq!(big_decode.mode, GraphMode::Piecewise);
        assert_eq!(big_decode.full.rows, 72);
        assert!(!big_decode.full.attention.aux_stream_live);

        let eager = normalize(vec![(0, 2049)], vec![]).unwrap();
        assert_eq!((eager.mode, eager.full.rows), (GraphMode::Eager, 2049));

        assert!(normalize_input(
            &UnifiedArchInput {
                groups: vec![ArchGroupInput::default(), ArchGroupInput::default()],
                tokens_per_source_rank: vec![],
            },
            None,
        )
        .is_err());
    }

    #[test]
    fn full_decode_graphs_are_keyed_by_request_count() {
        let sizes = |q: u32| -> Vec<u32> {
            let mut out: Vec<u32> = (1..=MAX_NUM_SEQS)
                .filter_map(|r| full_decode_graph_requests(r, q))
                .collect();
            out.dedup();
            out
        };
        // Server log `decode_cudagraph_capture_request_sizes` (TP4 + DSpark).
        assert_eq!(
            sizes(6),
            [
                1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 26, 27, 28,
                30, 31, 32, 34, 35, 36, 38, 39, 40, 42, 43, 46, 48, 51, 54, 56, 59, 62, 64
            ]
        );
        assert_eq!(sizes(1), [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]);
        assert_eq!(full_decode_graph_requests(65, 1), None);
        assert_eq!(full_decode_graph_requests(0, 6), None);

        // A verify batch of 5 requests x 6 rows replays the 6-request graph.
        assert_eq!(step_graph(&[6; 5], 6), (GraphMode::Full, 36));
        // A clipped request breaks uniformity: PIECEWISE by tokens.
        assert_eq!(step_graph(&[6, 6, 4], 6), (GraphMode::Piecewise, 16));
        assert_eq!(step_graph(&[1; 3], 1), (GraphMode::Full, 4));
    }

    #[test]
    fn verify_rows_flatten_and_small_full_graphs_take_the_mhc_overlap() {
        let step = DeepseekV41Step {
            prefill: vec![],
            decode: vec![(6, 106), (6, 10)],
        };
        let input = normalize_step(&step, 6, 12, None);
        assert_eq!((input.mode, input.full.rows), (GraphMode::Full, 12));
        assert_eq!(
            input.full.attention.decode_kv_lens,
            [100, 101, 102, 103, 104, 105, 4, 5, 6, 7, 8, 9]
        );
        assert!(input.full.attention.mhc_overlap && input.full.attention.aux_stream_live);
        // 3 requests pad to the 3-request graph, 18 rows: past the overlap.
        let three = DeepseekV41Step {
            prefill: vec![],
            decode: vec![(6, 50); 3],
        };
        let input = normalize_step(&three, 6, 18, None);
        assert_eq!(input.full.rows, 18);
        assert!(!input.full.attention.mhc_overlap);
        // PIECEWISE steps never take it.
        let mixed = normalize_input(&one_group(vec![(0, 4)], vec![9; 4]), None).unwrap();
        assert_eq!((mixed.mode, mixed.full.rows), (GraphMode::Piecewise, 8));
        assert!(!mixed.full.attention.mhc_overlap);
        let small = normalize_input(&one_group(vec![], vec![9; 16]), None).unwrap();
        assert!(small.full.attention.mhc_overlap);
    }

    /// `max_model_len` defaults to the checkpoint's 1M positions, refuses more,
    /// and bounds each prefill request's context.
    #[test]
    fn max_model_len_defaults_to_the_positions_and_bounds_requests() {
        let cfg = model_cfg();
        assert_eq!(cfg.max_position_embeddings, 1_048_576);
        let default = resolve_max_model_len(None, cfg.max_position_embeddings).unwrap();
        assert_eq!(default, 1_048_576);
        assert_eq!(
            resolve_max_model_len(Some(131_072), cfg.max_position_embeddings).unwrap(),
            131_072
        );
        for bad in [0, 1_048_577] {
            assert!(resolve_max_model_len(Some(bad), cfg.max_position_embeddings).is_err());
            let over = DeepseekV41VllmParallel {
                max_model_len: bad,
                ..parallel()
            };
            assert!(build_configs(&cfg, &over, &uniform_demand()).is_err());
        }
        // The resolved value reaches every attention config.
        let full = DeepseekV41VllmParallel {
            max_model_len: default,
            ..parallel()
        };
        let configs = build_configs(&cfg, &full, &uniform_demand()).unwrap();
        assert!(std::iter::once(&configs.layer0)
            .chain(&configs.bodies)
            .all(|body| body.attention.max_model_len == default));

        // A request whose context is exactly max_model_len is served; one
        // token more is refused.
        let last_chunk = one_group(vec![(default - 2048, 2048)], vec![]);
        assert!(DeepseekV41Step::from_unified(&last_chunk, default).is_ok());
        let past = one_group(vec![(default - 2047, 2048)], vec![]);
        let error = DeepseekV41Step::from_unified(&past, default).err().unwrap();
        assert!(error.contains("max_model_len 1048576"), "{error}");
        // The capture's pin refuses what the default serves.
        assert!(DeepseekV41Step::from_unified(&last_chunk, CAPTURED_MAX_MODEL_LEN).is_err());
    }

    fn no_replay_parallel() -> DeepseekV41VllmParallel {
        DeepseekV41VllmParallel {
            decoder_swa_bounded_replay: false,
            ..parallel()
        }
    }

    fn one_group(prefill: Vec<(u32, u32)>, decode: Vec<u32>) -> UnifiedArchInput {
        UnifiedArchInput {
            groups: vec![group(prefill, decode)],
            tokens_per_source_rank: vec![],
        }
    }

    fn normalize_input(
        batch: &UnifiedArchInput,
        replay: Option<&BoundedReplay>,
    ) -> std::result::Result<NormalizedInput, String> {
        let step = DeepseekV41Step::from_unified(batch, CAPTURED_MAX_MODEL_LEN)?;
        Ok(normalize_step(&step, 1, step.request_count(), replay))
    }

    #[test]
    fn bounded_replay_cuts_at_the_last_kv_source_and_splits_no_body() {
        let cfg = model_cfg();
        let off = build_configs(&cfg, &no_replay_parallel(), &uniform_demand()).unwrap();
        assert!(off.replay.is_none());
        assert!(off.late_bodies.iter().all(|&late| !late));

        let on = build_configs(&cfg, &parallel(), &uniform_demand()).unwrap();
        assert_eq!(
            on.replay.as_ref().unwrap().rule,
            BoundedReplay {
                cut_layer: 20,
                window: 128,
                trim_threshold: 768,
            }
        );
        // Same bodies and plan: the flag only adds the boundary and picks inputs.
        assert_eq!(on.plan, off.plan);
        let late: Vec<_> = on
            .bodies
            .iter()
            .zip(&on.late_bodies)
            .filter(|(_, &late)| late)
            .map(|(body, _)| body.layers.clone())
            .collect();
        assert_eq!(
            late,
            vec![
                vec![20],
                vec![21, 22, 23, 25, 26, 27, 29, 30, 31, 33, 34, 35],
                vec![24, 28, 32, 36],
                vec![37, 38, 39],
            ]
        );
    }

    #[test]
    fn bounded_replay_follows_the_fork_preconditions() {
        let replay = BoundedReplay::from_model(&model_cfg()).unwrap();
        assert!(replay.is_late(&[19, 20]).is_err());
        assert_eq!(replay.is_late(&[20, 39]).ok(), Some(true));
        assert_eq!(replay.is_late(&[1, 19]).ok(), Some(false));

        let mut engram = model_cfg();
        engram.engram_layer_ids.push(30);
        assert!(BoundedReplay::from_model(&engram).is_err());
        let mut no_sources = model_cfg();
        no_sources.kv_source_layer_ids.clear();
        assert!(BoundedReplay::from_model(&no_sources).is_err());
        let mut compressing_cut = model_cfg();
        compressing_cut.compress_ratios[20] = 2;
        assert!(BoundedReplay::from_model(&compressing_cut).is_err());
        let mut last = model_cfg();
        last.kv_source_layer_ids.push(39);
        assert!(BoundedReplay::from_model(&last).is_err());
        // A later last source moves the cut with it.
        let mut later = model_cfg();
        later.kv_source_layer_ids.push(24);
        assert_eq!(BoundedReplay::from_model(&later).unwrap().cut_layer, 24);
    }

    #[test]
    fn replay_trims_eager_and_large_piecewise_steps_only() {
        let rule = BoundedReplay::from_model(&model_cfg()).unwrap();
        let reduce = |prefill, decode| normalize_input(&one_group(prefill, decode), Some(&rule));

        // A full 2048-token chunk: PIECEWISE 2048 >= 768 -> 128 replay rows.
        let cold = reduce(vec![(0, 2048)], vec![]).unwrap();
        assert_eq!((cold.mode, cold.full.rows), (GraphMode::Piecewise, 2048));
        let late = cold.replay.as_ref().unwrap();
        assert_eq!(late.rows, 128);
        assert_eq!(late.attention.prefill_query_context_pairs, [(128, 2048)]);
        assert!(!late.attention.aux_stream_live && !late.attention.mhc_overlap);

        // Mixed at 2048 rows: 45 decodes + 128 + 91 = 264 kept -> 384 rows.
        let mixed = reduce(vec![(0, 1912), (4005, 91)], vec![300; 45]).unwrap();
        let late = mixed.replay.as_ref().unwrap();
        assert_eq!((mixed.full.rows, late.rows), (2048, 384));
        assert_eq!(
            late.attention.prefill_query_context_pairs,
            [(128, 1912), (91, 4096)]
        );
        assert_eq!(late.attention.decode_kv_lens, vec![300; 45]);

        // Forced at the threshold even with nothing to trim: 768 rows of
        // decodes and short chunks keep every row (768 -> 768).
        let forced = reduce(vec![(0, 100); 7], vec![50; 68]).unwrap();
        assert_eq!(forced.full.rows, 768);
        assert_eq!(forced.replay.as_ref().unwrap().rows, 768);

        // Under the threshold the whole batch runs.
        assert!(reduce(vec![(0, 700)], vec![]).unwrap().replay.is_none());
        // FULL decode graphs never replay.
        assert!(reduce(vec![], vec![1000; 48]).unwrap().replay.is_none());

        // Eager steps replay only when a prefilling request exceeds the window.
        let eager = reduce(vec![(0, 4096)], vec![]).unwrap();
        assert_eq!(eager.mode, GraphMode::Eager);
        assert_eq!(eager.replay.as_ref().unwrap().rows, 128);
        let short = reduce(vec![(0, 100); 21], vec![]).unwrap();
        assert_eq!(short.mode, GraphMode::Eager);
        assert!(short.replay.is_none());
        // A replay batch past the replay graphs runs eager at its size.
        let many = reduce(vec![(0, 200); 17], vec![]).unwrap();
        assert_eq!(many.replay.as_ref().unwrap().rows, 17 * 128);
    }

    #[test]
    fn replay_boundary_runs_only_on_replay_steps() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let build_model = |parallel: &DeepseekV41VllmParallel| {
            let configs = build_configs(&model_cfg(), parallel, &uniform_demand()).unwrap();
            build("unified".into(), resolve_configs(&configs), &bridge).unwrap()
        };
        let (off, on) = (build_model(&no_replay_parallel()), build_model(&parallel()));
        // write_kv attention + gather + scatter.
        let names: Vec<String> = on
            .cost_tree()
            .slots
            .iter()
            .map(|slot| slot.name.clone())
            .collect();
        assert!(names.iter().any(|n| n == "unified.replay.gather"));
        assert!(names.iter().any(|n| n == "unified.replay.scatter"));
        assert!(names
            .iter()
            .any(|n| n.starts_with("unified.layer20.write_kv.")));
        assert!(on.n_slots > off.n_slots);
        assert!(off
            .cost_tree()
            .slots
            .iter()
            .all(|slot| !slot.name.contains("replay") && !slot.name.contains("write_kv")));
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
