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
use crate::timing::kernels::all_reduce_fusion::fused_all_reduce_refusal;
use crate::timing::kernels::compressed_sparse_mla_rope_cast::NVFP4_RECORD_BYTES;
use crate::timing::kernels::kv_compress_store::FP8_INDEXER_ROW_BYTES;
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
const FP32_GEMM_BACKENDS: &[&str] = &["torch_cublas"];
const KV_INSERT_BACKENDS: &[&str] = &["vllm_cuda"];
const MEGA_ATTN_BACKENDS: &[&str] = &["flashmla_mega"];
const WO_A_BACKENDS: &[&str] = &["deepgemm_mxfp8_einsum_grouped_o_proj"];
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
    /// Decoder SWA bounded replay (what-if; see [`BoundedReplay`]).
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

/// Decoder SWA bounded replay: a counterfactual with no framework capture
/// behind it. vLLM PR #58132 and SGLang `--enable-decoder-swa-bounded-replay`
/// (`python/sglang/srt/models/deepseek_v4.py`, `late_layer_start` and the
/// `late_layer_tail` slicing in `forward`) run the layers past the last KV
/// source over each prefill request's last `sliding_window` extend tokens only:
/// those layers own no compressed KV (ratio 0/1, not KV sources), so the only
/// state they leave for later decode steps is the sliding window, and only the
/// last token's hidden state reaches the head.
///
/// Modeled as a reduced input for every body whose layers are all
/// `>= late_layer_start` (= `max(kv_source_layer_ids) + 1`, layer 21 on
/// V4.1-Flash): each prefill chunk `(append, context)` becomes
/// `(min(append, window), context)`, decode rows are unchanged, and the dense
/// rows re-pad to the CUDA-graph size of the reduced token count (which also
/// re-derives the aux-stream gate). Layer 0, the prologue, the Engram prefetch,
/// layers `1..late_layer_start` and the head keep the full input. The cost tree
/// is the same with the flag on or off; only slot inputs change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundedReplay {
    pub late_layer_start: u32,
    /// Replayed extend tokens per prefill chunk (`sliding_window`, 128).
    pub window: u32,
}

impl BoundedReplay {
    /// Derive the tail start from the config and check the SGLang
    /// preconditions: every late layer compresses at ratio 0 or 1, is no KV
    /// source, and has no Engram block.
    pub fn from_model(model: &DeepseekV41ModelCfg) -> std::result::Result<Self, BuildError> {
        let late_layer_start = model
            .kv_source_layer_ids
            .iter()
            .max()
            .map(|&last| last + 1)
            .ok_or_else(|| fit_failed("decoder SWA bounded replay needs kv_source_layer_ids"))?;
        if late_layer_start >= model.num_layers {
            return Err(fit_failed(format!(
                "decoder SWA bounded replay: late_layer_start {late_layer_start} leaves no tail"
            )));
        }
        for layer in late_layer_start..model.num_layers {
            let ratio = model.compress_ratios[layer as usize];
            if ratio > 1
                || model.kv_source_layer_ids.contains(&layer)
                || model.engram_layer_ids.contains(&layer)
            {
                return Err(fit_failed(format!(
                    "decoder SWA bounded replay: late layer {layer} must be a non-KV-source, \
                     non-Engram layer with compress ratio 0 or 1 (got ratio {ratio})"
                )));
            }
        }
        Ok(Self {
            late_layer_start,
            window: model.sliding_window,
        })
    }

    fn is_late(&self, layers: &[u32]) -> std::result::Result<bool, BuildError> {
        let late = layers
            .iter()
            .filter(|&&l| l >= self.late_layer_start)
            .count();
        match late {
            0 => Ok(false),
            n if n == layers.len() => Ok(true),
            _ => Err(fit_failed(format!(
                "body for layers {layers:?} straddles late_layer_start {}",
                self.late_layer_start
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
    pub total_kv_bytes_per_token: u64,
    /// `Some` with `decoder_swa_bounded_replay` on.
    pub bounded_replay: Option<BoundedReplay>,
    /// Per body: evaluated on the bounded-replay input (all layers late).
    pub late_bodies: Vec<bool>,
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
    bounded_replay: Option<BoundedReplay>,
    late_bodies: Vec<bool>,
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
    let bounded_replay = parallel
        .decoder_swa_bounded_replay
        .then(|| BoundedReplay::from_model(model))
        .transpose()?;
    let late_bodies = match &bounded_replay {
        Some(replay) => bodies
            .iter()
            .map(|body: &DeepseekV41LayerConfig| replay.is_late(&body.layers))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        None => vec![false; bodies.len()],
    };
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
        bounded_replay,
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

/// All-rank bytes of per-token KV state, in vLLM's page layout. Each TP rank
/// holds the one KV head (replicated). vLLM packs, per 128-token block, every
/// KV-source layer's NVFP4 compressed page (288 B per `ratio` tokens) and each
/// owned indexer FP8 key page (132 B per `ratio` tokens), each page rounded up
/// to the 512 B TMA stride. With sources 2/8/14 at ratio 2 and 20 at ratio 1
/// that is 135168 B per block, 1056 B per token per rank (the ratio-2 index
/// pages round 8448 B up to 8704 B).
///
/// The sliding-window cache (40 layers x 528 B, 128-token window) and the
/// ratio-2 compressor rings are per-request constants vLLM frees outside the
/// window, not per-token state, so they are left out (a few MB per request).
fn total_kv_bytes_per_token(model: &DeepseekV41ModelCfg, tp_size: u32) -> u64 {
    const BLOCK_TOKENS: u64 = 128;
    const PAGE_ALIGN: u64 = 512;
    let page = |states: u64, bytes: u64| (states * bytes).div_ceil(PAGE_ALIGN) * PAGE_ALIGN;
    let per_rank_block: u64 = model
        .kv_source_layer_ids
        .iter()
        .map(|&layer| {
            let states = BLOCK_TOKENS / u64::from(model.compress_ratios[layer as usize]);
            // Index K caches are owned by KV sources that are also index sources.
            let index = if model.index_source_layer_ids.contains(&layer) {
                page(states, u64::from(FP8_INDEXER_ROW_BYTES))
            } else {
                0
            };
            page(states, u64::from(NVFP4_RECORD_BYTES)) + index
        })
        .sum();
    per_rank_block / BLOCK_TOKENS * u64::from(tp_size)
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
            max_model_len: parallel.max_model_len,
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
        total_kv_bytes_per_token: configs.total_kv_bytes_per_token,
        bounded_replay: configs.bounded_replay,
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
    /// Longest request context; longer inputs are refused.
    max_model_len: u32,
    total_kv_bytes_per_token: u64,
    bounded_replay: Option<BoundedReplay>,
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
        total_kv_bytes_per_token: resolved.total_kv_bytes_per_token,
        bounded_replay: resolved.bounded_replay,
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

    /// `late` is the bounded-replay input, used for bodies in `late_bodies`.
    fn eval_plan(
        &self,
        plan: &[PlanNode],
        input: &NormalizedInput,
        late: &NormalizedInput,
        evaluator: &mut Evaluator,
    ) {
        for node in plan {
            match node {
                PlanNode::Body(index) => {
                    let body_input = if self.late_bodies[*index] {
                        late
                    } else {
                        input
                    };
                    self.bodies[*index].eval(body_input, evaluator)
                }
                PlanNode::Repeat { body, .. } => self.eval_plan(body, input, late, evaluator),
            }
        }
    }

    fn eval_into(&self, batch: &UnifiedArchInput, evaluator: &mut Evaluator) {
        let input = normalize_input(batch, self.max_model_len)
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
        let late = self
            .bounded_replay
            .map(|replay| bounded_replay_input(&input, replay.window));
        self.eval_plan(
            &self.plan,
            &input,
            late.as_ref().unwrap_or(&input),
            evaluator,
        );
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

fn normalize_input(
    input: &UnifiedArchInput,
    max_model_len: u32,
) -> std::result::Result<NormalizedInput, String> {
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
        if append == 0 || context > max_model_len {
            return Err(format!(
                "prefill request {request} ({prefix}, {append}) must append 1..=max_model_len \
                 {max_model_len}"
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

/// The input of a bounded-replay late layer ([`BoundedReplay`]): each prefill
/// chunk keeps only its last `window` extend tokens; decode rows are unchanged.
fn bounded_replay_input(full: &NormalizedInput, window: u32) -> NormalizedInput {
    let prefill_query_context_pairs: Vec<(u32, u32)> = full
        .attention
        .prefill_query_context_pairs
        .iter()
        .map(|&(append, context)| (append.min(window), context))
        .collect();
    let decode_tokens = full.attention.decode_kv_lens.len() as u32;
    let tokens = decode_tokens
        + prefill_query_context_pairs
            .iter()
            .map(|&(append, _)| append)
            .sum::<u32>();
    let rows = cudagraph_rows(tokens);
    let decode_only = prefill_query_context_pairs.is_empty() && decode_tokens > 0;
    NormalizedInput {
        rows,
        logits_rows: full.logits_rows,
        attention: DeepseekV41AttentionTpWorkletInput {
            num_tokens: rows,
            prefill_query_context_pairs,
            decode_kv_lens: full.attention.decode_kv_lens.clone(),
            aux_stream_live: attention_aux_stream_live(decode_only, rows, MAX_FULL_GRAPH_TOKENS),
        },
    }
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
            decoder_swa_bounded_replay: false,
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
        // vLLM block of 128 tokens: 3 x (18432 + 8704) + (36864 + 16896)
        // = 135168 B per rank -> 1056 B/token, replicated on 4 ranks.
        assert_eq!(configs.total_kv_bytes_per_token, 1056 * 4);

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
        let decode = normalize_input(
            &UnifiedArchInput {
                groups: vec![group(vec![], vec![1000; 48])],
                tokens_per_source_rank: vec![],
            },
            CAPTURED_MAX_MODEL_LEN,
        )
        .unwrap();
        assert_eq!((decode.rows, decode.logits_rows), (48, 48));
        assert!(decode.attention.aux_stream_live);

        let mixed = normalize_input(
            &UnifiedArchInput {
                groups: vec![group(vec![(512, 126)], vec![900; 48])],
                tokens_per_source_rank: vec![],
            },
            CAPTURED_MAX_MODEL_LEN,
        )
        .unwrap();
        assert_eq!((mixed.rows, mixed.logits_rows), (176, 49));
        assert!(!mixed.attention.aux_stream_live);
        assert_eq!(mixed.attention.prefill_query_context_pairs, [(126, 638)]);

        let big_decode = normalize_input(
            &UnifiedArchInput {
                groups: vec![group(vec![], vec![10; 65])],
                tokens_per_source_rank: vec![],
            },
            CAPTURED_MAX_MODEL_LEN,
        )
        .unwrap();
        assert_eq!(big_decode.rows, 72);
        assert!(!big_decode.attention.aux_stream_live);

        assert!(normalize_input(
            &UnifiedArchInput {
                groups: vec![ArchGroupInput::default(), ArchGroupInput::default()],
                tokens_per_source_rank: vec![],
            },
            CAPTURED_MAX_MODEL_LEN,
        )
        .is_err());
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
        assert!(normalize_input(&last_chunk, default).is_ok());
        let past = one_group(vec![(default - 2047, 2048)], vec![]);
        let error = normalize_input(&past, default).err().unwrap();
        assert!(error.contains("max_model_len 1048576"), "{error}");
        // The capture's pin refuses what the default serves.
        assert!(normalize_input(&last_chunk, CAPTURED_MAX_MODEL_LEN).is_err());
    }

    fn replay_parallel() -> DeepseekV41VllmParallel {
        DeepseekV41VllmParallel {
            decoder_swa_bounded_replay: true,
            ..parallel()
        }
    }

    fn one_group(prefill: Vec<(u32, u32)>, decode: Vec<u32>) -> UnifiedArchInput {
        UnifiedArchInput {
            groups: vec![group(prefill, decode)],
            tokens_per_source_rank: vec![],
        }
    }

    #[test]
    fn bounded_replay_starts_after_the_last_kv_source_and_splits_no_body() {
        let cfg = model_cfg();
        let off = build_configs(&cfg, &parallel(), &uniform_demand()).unwrap();
        assert_eq!(off.bounded_replay, None);
        assert!(off.late_bodies.iter().all(|&late| !late));

        let on = build_configs(&cfg, &replay_parallel(), &uniform_demand()).unwrap();
        assert_eq!(
            on.bounded_replay,
            Some(BoundedReplay {
                late_layer_start: 21,
                window: 128
            })
        );
        // Same bodies and plan: the flag only picks inputs.
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
                vec![21, 22, 23, 25, 26, 27, 29, 30, 31, 33, 34, 35],
                vec![24, 28, 32, 36],
                vec![37, 38, 39],
            ]
        );
    }

    #[test]
    fn bounded_replay_rejects_a_tail_sglang_would_reject() {
        let replay = BoundedReplay::from_model(&model_cfg()).unwrap();
        assert!(replay.is_late(&[20, 21]).is_err());
        assert_eq!(replay.is_late(&[21, 39]).ok(), Some(true));
        assert_eq!(replay.is_late(&[1, 20]).ok(), Some(false));

        let mut compressing = model_cfg();
        compressing.compress_ratios[30] = 2;
        assert!(BoundedReplay::from_model(&compressing).is_err());
        let mut engram = model_cfg();
        engram.engram_layer_ids.push(30);
        assert!(BoundedReplay::from_model(&engram).is_err());
        let mut no_sources = model_cfg();
        no_sources.kv_source_layer_ids.clear();
        assert!(BoundedReplay::from_model(&no_sources).is_err());
        // A later last source moves the tail start with it.
        let mut later = model_cfg();
        later.kv_source_layer_ids.push(24);
        assert_eq!(
            BoundedReplay::from_model(&later).unwrap().late_layer_start,
            25
        );
    }

    #[test]
    fn bounded_replay_input_keeps_the_last_window_of_each_chunk() {
        let reduce = |prefill, decode| {
            let full =
                normalize_input(&one_group(prefill, decode), CAPTURED_MAX_MODEL_LEN).unwrap();
            let late = bounded_replay_input(&full, 128);
            (full, late)
        };
        // Cold 2048-token chunk: 2048 rows -> 128 rows at the late layers.
        let (full, late) = reduce(vec![(0, 2048)], vec![]);
        assert_eq!((full.rows, late.rows), (2048, 128));
        assert_eq!(late.attention.num_tokens, 128);
        assert_eq!(late.attention.prefill_query_context_pairs, [(128, 2048)]);
        assert_eq!(late.logits_rows, 1);
        assert!(!late.attention.aux_stream_live);

        // A chunk no longer than the window is untouched.
        let (full, late) = reduce(vec![(300, 100)], vec![]);
        assert_eq!((full.rows, late.rows), (104, 104));
        assert_eq!(late.attention.prefill_query_context_pairs, [(100, 400)]);

        // Mixed: 45 decodes + chunks 1912 and 91 -> 45 + 128 + 91 = 264 tokens
        // pad to 272 (full batch 2048).
        let (full, late) = reduce(vec![(0, 1912), (4005, 91)], vec![300; 45]);
        assert_eq!((full.rows, late.rows), (2048, 272));
        assert_eq!(
            late.attention.prefill_query_context_pairs,
            [(128, 1912), (91, 4096)]
        );
        assert_eq!(late.attention.decode_kv_lens, vec![300; 45]);
        assert!(!late.attention.aux_stream_live);

        // Decode-only batches are unchanged, including the aux-stream gate.
        let (full, late) = reduce(vec![], vec![1000; 48]);
        assert_eq!((full.rows, late.rows), (48, 48));
        assert_eq!(full.attention.decode_kv_lens, late.attention.decode_kv_lens);
        assert!(late.attention.aux_stream_live && full.attention.aux_stream_live);

        // Two decodes + a 200-token chunk -> 2 + 128 = 130 tokens -> 136 rows;
        // a prefill is present, so the aux stream stays off.
        let (_, late) = reduce(vec![(0, 200)], vec![10; 2]);
        assert_eq!(late.rows, 136);
        assert!(!late.attention.aux_stream_live);
    }

    #[test]
    fn bounded_replay_keeps_the_cost_tree_shape() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let tree = |parallel: &DeepseekV41VllmParallel| {
            let configs = build_configs(&model_cfg(), parallel, &uniform_demand()).unwrap();
            build("unified".into(), resolve_configs(&configs), &bridge).unwrap()
        };
        let (off, on) = (tree(&parallel()), tree(&replay_parallel()));
        assert_eq!(off.n_slots, on.n_slots);
        assert_eq!(off.cost_log_manifest(), on.cost_log_manifest());
        assert_eq!(off.cost_flat.len(), on.cost_flat.len());
    }

    /// Real-cost checks: needs the warm `profiling/profile.db` rows and the
    /// capture-2 token corpus. Run from the repo with
    /// `cargo test -p simulator --lib bounded_replay_costs -- --ignored`.
    #[test]
    #[ignore = "needs warm profile.db rows and the capture-2 token corpus"]
    fn bounded_replay_costs_against_profile_db() {
        use crate::arch::config::IterArchSel;

        let root = env!("CARGO_MANIFEST_DIR");
        let corpus = format!(
            "{root}/logs/20260924_0_dsv41_flash_capture/profile_corpus/token_corpus/manifest.json"
        );
        let selector = |extra: serde_json::Value| -> IterArchSel {
            let mut json = serde_json::json!({
                "type": "deepseek_v41_vllm",
                "model_config": CONFIG,
                "fp8": true,
                "routing": "corpus",
                "token_corpus_file": corpus,
            });
            json.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            serde_json::from_value(json).unwrap()
        };
        let bridge = PerfApiBridge::new().expect("perf_api bridge");
        let build_model = |sel: &IterArchSel| {
            crate::arch::build::build_iter_model(sel, GPU_NAME, "unified", &bridge).unwrap()
        };
        let default = build_model(&selector(serde_json::json!({})));
        let off = build_model(&selector(
            serde_json::json!({"decoder_swa_bounded_replay": false}),
        ));
        let on = build_model(&selector(
            serde_json::json!({"decoder_swa_bounded_replay": true}),
        ));
        let names: Vec<String> = default
            .cost_log_manifest()
            .slots
            .iter()
            .map(|s| s.name.clone())
            .collect();
        let run = |model: &dyn IterwiseUnifiedModel, batch: &UnifiedArchInput| {
            let (mut slots, mut scratch, mut inputs) = (Vec::new(), Vec::new(), Vec::new());
            let total = model.eval_iter_with_inputs(batch, &mut slots, &mut scratch, &mut inputs);
            let inputs: Vec<serde_json::Value> = inputs
                .iter()
                .map(|i| serde_json::to_value(i).unwrap())
                .collect();
            (total, format!("{slots:?}"), inputs)
        };
        let batches = [
            one_group(vec![], vec![1000; 48]),
            one_group(vec![(0, 2048)], vec![]),
            one_group(vec![(300, 100)], vec![]),
            one_group(vec![(0, 1912), (4005, 91)], vec![300; 45]),
        ];
        for batch in &batches {
            // Flag off is bit-identical to the default (field absent) selector.
            let (t_default, s_default, _) = run(default.as_ref(), batch);
            let (t_off, s_off, _) = run(off.as_ref(), batch);
            assert_eq!(format!("{t_default:?}"), format!("{t_off:?}"));
            assert_eq!(s_default, s_off);
        }
        // Decode-only and a chunk within the window: identical on and off.
        for batch in [&batches[0], &batches[2]] {
            let (t_off, s_off, _) = run(off.as_ref(), batch);
            let (t_on, s_on, _) = run(on.as_ref(), batch);
            assert_eq!(format!("{t_off:?}"), format!("{t_on:?}"));
            assert_eq!(s_off, s_on);
        }
        // Cold 2048 chunk: cheaper, late layers at 128 rows, earlier at 2048.
        let (t_off, _, in_off) = run(off.as_ref(), &batches[1]);
        let (t_on, _, in_on) = run(on.as_ref(), &batches[1]);
        assert!(
            t_on.m.time_ms < t_off.m.time_ms,
            "on {} >= off {}",
            t_on.m.time_ms,
            t_off.m.time_ms
        );
        // Every dense row leaf of a late body runs 128 rows (2048 off). The
        // indexer's key gathers count context keys, which stay at 2048.
        for prefix in ["unified.layer21.", "unified.layer24.", "unified.layer37."] {
            let row_leaves: Vec<usize> = (0..names.len())
                .filter(|&slot| {
                    names[slot].starts_with(prefix)
                        && !names[slot].contains(".indexer.")
                        && in_on[slot].get("num_tokens").is_some()
                })
                .collect();
            assert!(row_leaves.len() >= 4, "{prefix}");
            for slot in row_leaves {
                assert_eq!(in_on[slot]["num_tokens"], 128, "{}", names[slot]);
                assert_eq!(in_off[slot]["num_tokens"], 2048, "{}", names[slot]);
            }
        }
        // The routed-expert and shared-expert GEMMs see the reduced rows too.
        let ffn_m = |inputs: &[serde_json::Value], prefix: &str| -> Vec<serde_json::Value> {
            (0..names.len())
                .filter(|&slot| names[slot].starts_with(prefix) && names[slot].contains(".ffn."))
                .filter_map(|slot| inputs[slot].get("m").cloned())
                .collect()
        };
        let late_m = ffn_m(&in_on, "unified.layer21.");
        assert!(
            !late_m.is_empty() && late_m.iter().all(|m| m == 128),
            "{late_m:?}"
        );
        assert!(ffn_m(&in_off, "unified.layer21.").iter().all(|m| m == 2048));
        for (slot, name) in names.iter().enumerate() {
            let early = [
                "unified.prologue",
                "unified.layer0.",
                "unified.layer20.",
                "unified.head",
                "unified.engram_prefetch",
            ];
            if early.iter().any(|p| name.starts_with(p)) {
                assert_eq!(in_on[slot], in_off[slot], "{name}");
            }
        }
    }

    /// A 500K-token request at the default (1M) `max_model_len`: its last
    /// prefill chunk and its decode cost from profile.db rows. Needs the warm
    /// rows and the capture-2 token corpus; run from the repo with
    /// `PYTHONPATH=$PWD uv run cargo test -p simulator --lib long_context -- --ignored`.
    #[test]
    #[ignore = "needs warm profile.db rows and the capture-2 token corpus"]
    fn long_context_request_costs_against_profile_db() {
        use crate::arch::config::IterArchSel;
        use crate::timing::cache::interp::CoverageFlags;

        let root = env!("CARGO_MANIFEST_DIR");
        let selector: IterArchSel = serde_json::from_value(serde_json::json!({
            "type": "deepseek_v41_vllm",
            "model_config": CONFIG,
            "fp8": true,
            "routing": "corpus",
            "token_corpus_file": format!(
                "{root}/logs/20260924_0_dsv41_flash_capture/profile_corpus/token_corpus/manifest.json"
            ),
        }))
        .unwrap();
        assert_eq!(selector.max_model_len().unwrap(), 1_048_576);
        let bridge = PerfApiBridge::new().expect("perf_api bridge");
        let model =
            crate::arch::build::build_iter_model(&selector, GPU_NAME, "unified", &bridge).unwrap();
        let names: Vec<String> = model
            .cost_log_manifest()
            .slots
            .iter()
            .map(|s| s.name.clone())
            .collect();
        let (mut slots, mut scratch, mut inputs) = (Vec::new(), Vec::new(), Vec::new());
        let mut eval = |batch: &UnifiedArchInput| {
            let total = model.eval_iter_with_inputs(batch, &mut slots, &mut scratch, &mut inputs);
            assert!(total.m.time_ms.is_finite() && total.m.time_ms > 0.0);
            assert!(!total.coverage.contains(CoverageFlags::NO_COVERAGE));
            let flagged: Vec<(String, serde_json::Value)> = (0..names.len())
                .filter(|&slot| !slots[slot].coverage.is_empty())
                .map(|slot| {
                    (
                        names[slot].clone(),
                        serde_json::to_value(&inputs[slot]).unwrap(),
                    )
                })
                .collect();
            // FlashMLA and the indexer logits are on grid at any context.
            for (slot, name) in names.iter().enumerate() {
                if name.contains(".mega_attn.") || name.contains("_logits") {
                    assert!(slots[slot].coverage.is_empty(), "{name}");
                }
            }
            (total.m.time_ms, flagged)
        };
        // Past the elementwise grid only through the context-sized indexer
        // placeholders, which hold the edge bandwidth and say so.
        let held = [".prefill_topk", ".prefill_k_gather", ".candidates"];
        let (prefill_ms, flagged) = eval(&one_group(vec![(497_952, 2048)], vec![]));
        assert!(!flagged.is_empty());
        for (name, _) in &flagged {
            assert!(
                name.contains(".attn.indexer.score.") && held.iter().any(|h| name.ends_with(h)),
                "{name}"
            );
        }
        for leaf in held {
            assert!(
                flagged.iter().any(|(name, _)| name.ends_with(leaf)),
                "{leaf}"
            );
        }
        let (mixed_ms, mixed_flagged) = eval(&one_group(vec![(497_952, 2048)], vec![500_000; 7]));
        assert!(mixed_ms > prefill_ms);
        for (name, _) in &mixed_flagged {
            assert!(held.iter().any(|h| name.ends_with(h)), "{name}");
        }
        // Decode: context changes no flag; only the row-sized leaves below the
        // 32-token grid start of a one-row batch are flagged.
        let (decode_ms, flagged) = eval(&one_group(vec![], vec![500_000]));
        assert!(decode_ms < prefill_ms);
        for (name, input) in &flagged {
            assert!(
                input["num_tokens"].as_u64().is_some_and(|n| n < 32),
                "{name} {input}"
            );
        }
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
