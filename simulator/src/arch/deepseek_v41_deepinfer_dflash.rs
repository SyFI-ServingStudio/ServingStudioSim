//! DeepInfer's DFlash speculative step around the DP target (DeepInfer
//! a679bb8 + CPU-Engram patches, `model_executor/layers/sampler.py`,
//! `model_executor/models/mtp/dflash.py`, `models/deepseek.py`,
//! `worker/*cuda_graph_runner.py`): every launch of one rank's step that is
//! not a target layer, a draft layer or a head.
//!
//! Capture: Slurm job 24554 (`logs/20261120_0_deepinfer_decode_nsys`), DP4/EP4,
//! 64 decode requests per rank at ~131k context, verify width 6, CUDA graphs.
//! Per rank and step the launches fall in five phases:
//!
//! | phase | where | launches (us/step in the capture) |
//! |---|---|---|
//! | prep | before the main graph | block tables, decode metadata, ZOS input gathers, Engram hash (~31) |
//! | main cache | inside the main graph | `_dflash_input_store` in the three target layers (11), then `update_main_cache`: `projection` GEMM, `projection_norm`, `main_kv_proj` GEMM, one SWA-cache store per draft layer (~60) |
//! | verify | sampler, eager | DeepSelect top-k over the fp32 logits (135), fused sample (29), verification bookkeeping (~45); the lm_head is the target head's |
//! | draft prep | DFlash graph | input copy, fused input index, draft-token embedding gather (~10) |
//! | Markov | DFlash graph, per drafted position | `markov_w1` gather, `markov_w2` accumulated into the fp32 logits (`out_acc`), confidence head, top-k, fused sample, scatter (~80 per position); `store_mtp_info` once |
//!
//! The draft layers are DP layer bodies and the draft head is the target's
//! head worklet (hc post, norm, fp32 lm_head), both priced by the arch.
//! Launches with an L1 kind take it; the rest are elementwise placeholders
//! sized from the checkpoint and the step (launch-bound, so the placeholder's
//! small-size floor carries them).

use std::sync::Arc;

use crate::arch::deepseek_v41_vllm::{DeepseekV41ModelCfg, ELEMENTWISE_BACKENDS};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    DeepSelectTopkKernel, DeepSelectTopkKernelConfig, DeepSelectTopkKernelInput,
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, FusedNormRopeCastKernel,
    FusedNormRopeCastKernelConfig, FusedNormRopeCastKernelInput, GemmFp32OutputKernel,
    GemmFp32OutputKernelConfig, GemmFp32OutputKernelInput, RmsNormMxfp8Kernel,
    RmsNormMxfp8KernelConfig, RmsNormMxfp8KernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};
use crate::worklet::deepseek_v41_common::{eval_or_zero, placeholder};

const I32: u32 = 4;
const I64: u32 = 8;
const BF16: u32 = 2;
const FP32: u32 = 4;

/// DeepInfer's graph sizes (`cuda_graph_runner.py` `build_graph_batch_sizes`):
/// powers of two below `interval`, then multiples of `interval`, then `max`.
pub(crate) fn graph_sizes(max_size: u32, interval: u32) -> Vec<u32> {
    assert!(max_size > 0 && interval > 0, "graph sizes need a positive max and interval");
    let interval = interval.min(max_size);
    let mut sizes = Vec::new();
    let mut size = 1;
    while size < interval {
        sizes.push(size);
        size *= 2;
    }
    sizes.extend((interval..max_size).step_by(interval as usize));
    sizes.push(max_size);
    sizes
}

/// The captured size a replay of `n` uses (`select`: the first size >= n);
/// `None` past the largest graph.
pub(crate) fn graph_for(sizes: &[u32], n: u32) -> Option<u32> {
    sizes.iter().copied().find(|&size| size >= n)
}

#[derive(Clone, Debug)]
pub struct DeepInferDflashConfig {
    pub gpu_name: String,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub head_dim: u32,
    pub rope_dim: u32,
    pub markov_rank: u32,
    pub num_draft_layers: u32,
    pub draft_tokens: u32,
    /// `DEEPINFER_SAMPLER_FORCE_TOPK`: the sampler keeps this many logits.
    pub force_topk: u32,
    /// Engram n-gram hashing in the step prep (`engram_hash_kernel`).
    pub engram_heads: u32,
    pub engram_max_ngram: u32,
    pub sliding_window: u32,
    pub max_model_len: u32,
    /// DeepInfer's KV page (block-table width = `max_model_len / block`).
    pub block_size: u32,
    /// One request's SWA ring (`window + backtracking`): a draft layer's
    /// SWA cache page.
    pub swa_ring_tokens: u32,
    /// Most rows one rank runs (the tilelang kinds' row axis).
    pub max_tokens_per_rank: u32,
    pub gemm_backends: Vec<&'static str>,
    pub fp32_gemm_backends: Vec<&'static str>,
    pub topk_backends: Vec<&'static str>,
    pub kv_backends: Vec<&'static str>,
}

/// What a placeholder's row count follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rows {
    /// The main graph's rows (padded).
    TargetRows,
    /// Scheduled requests.
    Requests,
    /// Rows the sampler verifies.
    LogitsRows,
    /// The DFlash graph's (padded) requests.
    DraftRequests,
    /// `draft_tokens x` draft requests.
    DraftRows,
}

/// One rank's step shapes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DflashShape {
    pub target_rows: u32,
    pub requests: u32,
    pub logits_rows: u32,
    pub draft_requests: u32,
    pub draft_rows: u32,
}

struct Leaf {
    op: Op<ElementwiseKernel>,
    launches: u32,
    rows: Rows,
}

struct Phase {
    label: String,
    leaves: Vec<Leaf>,
}

/// One rank's DFlash edges.
pub(crate) struct DeepInferDflash {
    name: String,
    draft_tokens: u32,
    num_draft_layers: u32,
    vocab_size: u32,
    verify_topk: Op<DeepSelectTopkKernel>,
    projection: Op<SingleGemmKernel>,
    projection_norm: Op<RmsNormMxfp8Kernel>,
    main_kv_proj: Op<SingleGemmKernel>,
    main_kv_store: Op<FusedNormRopeCastKernel>,
    markov_w2: Op<GemmFp32OutputKernel>,
    markov_topk: Op<DeepSelectTopkKernel>,
    phases: Vec<Phase>,
}

/// The verify phase runs DeepSelect twice over the verify rows (capture:
/// two `topk_kernel` launches between the target lm_head and the fused sample).
const VERIFY_TOPK_LAUNCHES: u32 = 2;

impl DeepInferDflashConfig {
    pub(crate) fn new(
        model: &DeepseekV41ModelCfg,
        num_draft_layers: u32,
        markov_rank: u32,
        draft_tokens: u32,
        gpu_name: &str,
        max_model_len: u32,
        max_tokens_per_rank: u32,
        backends: DeepInferDflashBackends,
    ) -> Self {
        Self {
            gpu_name: gpu_name.to_string(),
            hidden_size: model.hidden_size,
            vocab_size: model.vocab_size,
            head_dim: model.head_dim,
            rope_dim: model.rope_dim,
            markov_rank,
            num_draft_layers,
            draft_tokens,
            force_topk: DEEPINFER_SAMPLER_FORCE_TOPK,
            engram_heads: model.engram_hash_heads,
            // The hash reads each row's n-gram: the token and its lookback.
            engram_max_ngram: model.engram_lookback_depth + 1,
            sliding_window: model.sliding_window,
            max_model_len,
            block_size: backends.block_size,
            swa_ring_tokens: backends.swa_ring_tokens,
            max_tokens_per_rank,
            gemm_backends: backends.gemm.to_vec(),
            fp32_gemm_backends: backends.fp32_gemm.to_vec(),
            topk_backends: backends.topk.to_vec(),
            kv_backends: backends.kv.to_vec(),
        }
    }
}

/// `DEEPINFER_SAMPLER_FORCE_TOPK` as DeepInfer's benchmark sets it
/// (`sampler_force_topk 1024` in the capture's results.jsonl).
pub(crate) const DEEPINFER_SAMPLER_FORCE_TOPK: u32 = 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct DeepInferDflashBackends {
    pub gemm: &'static [&'static str],
    pub fp32_gemm: &'static [&'static str],
    pub topk: &'static [&'static str],
    pub kv: &'static [&'static str],
    pub block_size: u32,
    pub swa_ring_tokens: u32,
}

impl DeepInferDflash {
    pub(crate) fn build(
        name: &str,
        c: &DeepInferDflashConfig,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        macro_rules! op {
            ($kernel:ty, $cfg:expr, $suffix:expr) => {{
                let op_name = format!("{name}.{}", $suffix);
                Op::new(
                    op_name.clone(),
                    Arc::new(<$kernel>::build(op_name, $cfg, bridge)?),
                )
            }};
        }
        let gpu = c.gpu_name.clone();
        let hidden = c.hidden_size;
        let topk = DeepSelectTopkKernelConfig {
            backends: c.topk_backends.clone(),
            gpu_name: gpu.clone(),
            top_k: c.force_topk,
            dtype: DType::Fp32,
            max_row_len: c.vocab_size,
        };
        let mxfp8 = |n: u32, k: u32| SingleGemmKernelConfig {
            backends: c.gemm_backends.clone(),
            gpu_name: gpu.clone(),
            n: Dim::from(n),
            k: Dim::from(k),
            dtype: DType::Mxfp8E4m3,
        };

        let verify_rows = c.draft_tokens + 1;
        let table_width = c.max_model_len.div_ceil(c.block_size);
        let ngram_ids = c.engram_heads * c.engram_max_ngram;
        // (leaf, launches, rows, input bytes/row, output bytes/row)
        let prep = vec![
            // `_compute_seqlens_k_zos`: last step's accepted counts -> lengths.
            ("compute_seqlens_k", 1, Rows::Requests, 3 * I32, 2 * I32),
            // `_prepare_v4_decode_metadata`: positions and slots per row.
            ("prepare_decode_metadata", 1, Rows::TargetRows, 2 * I32, 2 * I32 + I64),
            ("copy_block_tables", 1, Rows::Requests, I32 * table_width, I32 * table_width),
            ("gather_input_ids", 1, Rows::TargetRows, I64 + I32, I32),
            ("gather_engram_tokens", 1, Rows::TargetRows, c.engram_max_ngram * I32, c.engram_max_ngram * I32),
            ("gather", 1, Rows::TargetRows, I32, I32),
            ("engram_hash", 1, Rows::TargetRows, c.engram_max_ngram * I32, ngram_ids * I64),
            // `_load_kv_index_kernel`: each row's SWA window indices.
            ("load_kv_index", 1, Rows::TargetRows, I32, c.sliding_window * I32),
        ];
        let main_cache = vec![
            // `_dflash_input_store` in each target feature layer.
            ("dflash_input_store", c.num_draft_layers, Rows::TargetRows, hidden * BF16, hidden * BF16),
        ];
        // Each verified row: its kept logits and indices in, its token and
        // log-probabilities out (`_fused_sample_kernel_1`, then the request
        // reduce `_fused_sample_kernel`).
        let sample_in = c.force_topk * (FP32 + I32);
        let sample_out = c.force_topk * FP32 + I64;
        let verify = vec![
            ("expand_sampling_params", 1, Rows::LogitsRows, 2 * FP32, 2 * FP32),
            ("prepare_verification_metadata", 1, Rows::Requests, verify_rows * I32, verify_rows * I32),
            ("fused_sample", 1, Rows::LogitsRows, sample_in, sample_out),
            ("fused_sample_reduce", 1, Rows::LogitsRows, sample_out, I64),
            ("map_lookup", 2, Rows::LogitsRows, c.force_topk * I32 + I64, I64),
            ("remap_draft_topk", 1, Rows::LogitsRows, c.force_topk * I32, c.force_topk * I32),
            ("scatter_gather", 1, Rows::LogitsRows, c.force_topk * FP32, c.force_topk * FP32),
            ("get_accept_result", 1, Rows::Requests, verify_rows * 2 * I64, 2 * I32),
            ("finalize_verification", 2, Rows::Requests, verify_rows * I64, verify_rows * I64),
        ];
        let draft_prep = vec![
            ("copy_decode_input", 1, Rows::DraftRequests, 2 * I32, 2 * I32),
            ("dflash_get_input", 1, Rows::DraftRows, 2 * I32, 3 * I32),
            // `_embed_draft_init_tokens`: the block's token embeddings.
            ("embed_tokens", 1, Rows::DraftRows, I32, hidden * BF16),
        ];
        let rank = c.markov_rank;
        let markov = vec![
            ("markov_w1_gather", c.draft_tokens, Rows::DraftRequests, I64, rank * BF16),
            // `confidence_head_fused`: the position's hc hidden and Markov
            // embedding in, one probability out.
            ("confidence_head", c.draft_tokens, Rows::DraftRequests, hidden * BF16 + rank * BF16, FP32),
            ("fused_sample", c.draft_tokens, Rows::DraftRequests, sample_in, sample_out),
            ("fused_sample_reduce", c.draft_tokens, Rows::DraftRequests, sample_out, I64),
            ("token_scatter", c.draft_tokens, Rows::DraftRequests, c.force_topk * I32 + I64, I64),
            ("store_mtp_info", 1, Rows::DraftRequests, c.draft_tokens * (I64 + FP32), c.draft_tokens * (I64 + FP32)),
        ];

        let mut phases = Vec::new();
        for (phase, spec) in [
            ("prep", prep),
            ("main_cache", main_cache),
            ("verify", verify),
            ("draft_prep", draft_prep),
            ("markov", markov),
        ] {
            let mut leaves = Vec::with_capacity(spec.len());
            for (leaf, launches, rows, input, output) in spec {
                if launches == 0 {
                    continue;
                }
                let config: ElementwiseKernelConfig =
                    placeholder(ELEMENTWISE_BACKENDS, &gpu, input, output);
                leaves.push(Leaf {
                    op: op!(ElementwiseKernel, config, format!("{phase}.{leaf}")),
                    launches,
                    rows,
                });
            }
            phases.push(Phase {
                label: format!("{name}.{phase} [DeepInfer step launches, elementwise placeholders]"),
                leaves,
            });
        }

        Ok(Self {
            name: name.to_string(),
            draft_tokens: c.draft_tokens,
            num_draft_layers: c.num_draft_layers,
            vocab_size: c.vocab_size,
            verify_topk: op!(DeepSelectTopkKernel, topk.clone(), "verify.topk"),
            // dflash.py `update_main_cache`: the three feature layers'
            // hiddens projected to one hidden, normed to MXFP8, then every
            // draft layer's main-token key.
            projection: op!(
                SingleGemmKernel,
                mxfp8(hidden, c.num_draft_layers * hidden),
                "main_cache.projection"
            ),
            projection_norm: op!(
                RmsNormMxfp8Kernel,
                RmsNormMxfp8KernelConfig {
                    backends: c.kv_backends.clone(),
                    gpu_name: gpu.clone(),
                    hidden,
                    return_bf16: false,
                    dtype: DType::Bf16,
                    max_tokens_per_rank: c.max_tokens_per_rank,
                },
                "main_cache.projection_norm"
            ),
            main_kv_proj: op!(
                SingleGemmKernel,
                mxfp8(c.num_draft_layers * c.head_dim, hidden),
                "main_cache.main_kv_proj"
            ),
            // attn.py `DFlashAttention.update_main_cache`: norm, RoPE and
            // MXFP8 cast stored into the draft layer's SWA cache.
            main_kv_store: op!(
                FusedNormRopeCastKernel,
                FusedNormRopeCastKernelConfig {
                    backends: c.kv_backends.clone(),
                    gpu_name: gpu.clone(),
                    num_heads: 1,
                    head_dim: c.head_dim,
                    rope_dim: c.rope_dim,
                    norm: true,
                    cast_format: "mxfp8".to_string(),
                    output: "cache".to_string(),
                    block_size: c.swa_ring_tokens,
                    dtype: DType::Bf16,
                    max_tokens_per_rank: c.max_tokens_per_rank,
                },
                "main_cache.kv_store"
            ),
            // `markov_w2(w1, out_acc=logits)`: rank -> vocab, accumulated
            // into the position's fp32 logits.
            markov_w2: op!(
                GemmFp32OutputKernel,
                GemmFp32OutputKernelConfig {
                    backends: c.fp32_gemm_backends.clone(),
                    gpu_name: gpu.clone(),
                    n: Dim::from(c.vocab_size),
                    k: Dim::from(rank),
                    input_dtype: DType::Bf16,
                },
                "markov.markov_w2"
            ),
            markov_topk: op!(DeepSelectTopkKernel, topk, "markov.topk"),
            phases,
        })
    }

    pub(crate) fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        let scale = |n: u32, node: CostNode| {
            if n == 1 {
                node
            } else {
                CostNode::Scale {
                    n,
                    child: Box::new(node),
                }
            }
        };
        let mut children = vec![
            scale(VERIFY_TOPK_LAUNCHES, self.verify_topk.compile(b)),
            self.projection.compile(b),
            self.projection_norm.compile(b),
            self.main_kv_proj.compile(b),
            scale(self.num_draft_layers, self.main_kv_store.compile(b)),
            scale(self.draft_tokens, self.markov_w2.compile(b)),
            scale(self.draft_tokens, self.markov_topk.compile(b)),
        ];
        for phase in &self.phases {
            children.push(CostNode::Labeled {
                label: phase.label.clone(),
                child: Box::new(CostNode::Sum(
                    phase
                        .leaves
                        .iter()
                        .map(|leaf| scale(leaf.launches, leaf.op.compile(b)))
                        .collect(),
                )),
            });
        }
        CostNode::Labeled {
            label: format!(
                "{} [DeepInfer DFlash step: verify top-k, main cache, Markov GEMM/top-k, \
                 step launches]",
                self.name
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub(crate) fn eval(&self, shape: &DflashShape, ev: &mut Evaluator) {
        let topk = |num_rows| DeepSelectTopkKernelInput {
            num_rows,
            row_len: self.vocab_size,
        };
        eval_or_zero(&self.verify_topk, topk(shape.logits_rows), shape.logits_rows == 0, ev);
        let rows = shape.target_rows;
        let idle = rows == 0;
        eval_or_zero(&self.projection, SingleGemmKernelInput { m: rows }, idle, ev);
        eval_or_zero(
            &self.projection_norm,
            RmsNormMxfp8KernelInput { num_tokens: rows },
            idle,
            ev,
        );
        eval_or_zero(&self.main_kv_proj, SingleGemmKernelInput { m: rows }, idle, ev);
        eval_or_zero(
            &self.main_kv_store,
            FusedNormRopeCastKernelInput { num_tokens: rows },
            idle,
            ev,
        );
        let requests = shape.draft_requests;
        eval_or_zero(
            &self.markov_w2,
            GemmFp32OutputKernelInput { m: requests },
            requests == 0,
            ev,
        );
        eval_or_zero(&self.markov_topk, topk(requests), requests == 0, ev);
        for phase in &self.phases {
            for leaf in &phase.leaves {
                let n = match leaf.rows {
                    Rows::TargetRows => shape.target_rows,
                    Rows::Requests => shape.requests,
                    Rows::LogitsRows => shape.logits_rows,
                    Rows::DraftRequests => shape.draft_requests,
                    Rows::DraftRows => shape.draft_rows,
                };
                eval_or_zero(&leaf.op, ElementwiseKernelInput { num_tokens: n }, n == 0, ev);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_sizes_follow_deepinfer() {
        // build_graph_batch_sizes(384, 32) and (64, 8).
        let target = graph_sizes(384, 32);
        assert_eq!(&target[..6], &[1, 2, 4, 8, 16, 32]);
        assert_eq!(target.last(), Some(&384));
        assert_eq!(graph_for(&target, 380), Some(384));
        assert_eq!(graph_for(&target, 33), Some(64));
        assert_eq!(graph_for(&target, 385), None);
        let draft = graph_sizes(64, 8);
        assert_eq!(draft, vec![1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]);
        // interval past max collapses to the max.
        assert_eq!(graph_sizes(6, 8), vec![1, 2, 4, 6]);
    }
}
