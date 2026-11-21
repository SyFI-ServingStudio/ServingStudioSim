//! DeepInfer's own launches around the DeepSeek-V4.1 attention of one layer
//! (DeepInfer a679bb8, `v4_modules/attn.py`, `compress.py`, `index_attn.py`).
//! Part of [`super::deepseek_v41_attention_tp::DeepseekV41AttentionTpWorklet`]
//! under `engine: deepinfer`. A DeepInfer step is either a prompt step or a
//! decode step (`input_metadata.is_prompt`); the table below is the prompt
//! step, priced over the prefill rows.
//!
//! | launch (nsys name) | call | rows / keys | leaf |
//! |---|---|---|---|
//! | `_norm_kernel` | `q_a_norm` (RMSNorm + MXFP8) | prefill rows | `q_a_norm` |
//! | `_fused_norm_rope_hadamard_cast_store_kvcache_kernel` | SWA KV norm, RoPE, MXFP8 round trip | prefill rows | `kv_swa` |
//! | DtoD memcpy | `kv_swa_cur.copy_(kv_swa)` into the prefill KV buffer | prefill rows | `kv_swa_copy` |
//! | `_load_paged_kv_cache_kernel` | SWA history | `window` per request | `swa_load` |
//! | `_store_kv_cache_kernel` | SWA store (after the load) | prefill rows | `swa_store` |
//! | `_expand_swa_index_kernel` | `kv_swa_full_index` (first layer, and the YOCO cut on the compact rows) | prefill rows | `swa_index` |
//! | `compress_and_update_state_prefill_kernel` (+ `_1`) | ratio-2 KV source compressor | prefill rows | `compress` |
//! | fused transform | main KV NVFP4 store + post-norm rows | `q / ratio` per request | `main_store` |
//! | `_load_paged_kv_cache_kernel` | main compressed KV (KV sources) | `ctx / ratio` per request | `main_load` |
//! | fused transform | index Q RoPE + MXFP4 | prefill rows | `index_q` |
//! | fused transform | index K store (index owners) | `q / ratio` per request | `index_k_store` |
//! | `_load_paged_kv_cache_kernel` | index K (index owners) | `ctx / ratio` per request | `index_k_load` |
//! | `_get_candidate_compact_lens_kernel` | the step's candidate schedule (first consumer) | prefill rows | `compact_lens` |
//! | `_restore_candidate_indices_inplace_kernel` | top-k back to key indices (consumers), once per indexer chunk | prefill rows | `restore` |
//! | `_kv_index_cat_kernel` | SWA + top-k index rows (compressed layers) | prefill rows | `kv_index_cat` |
//!
//! `write_kv_only` (the YOCO cut's `_store_yoco_full_kv` over every row) runs
//! only the compressor and the main / index K stores; `kv_written` (the cut
//! layer's compact body after that write) skips them.
//!
//! A decode step (capture: Slurm job 24554, `logs/20261120_0_deepinfer_decode_nsys`,
//! 384 verify rows per rank) launches, per layer, over the decode rows:
//!
//! | launch (nsys name) | call | rows | leaf |
//! |---|---|---|---|
//! | `_fused_norm_rope_hadamard_cast_store_kvcache_kernel` | `_prepare_swa_kv` (decode): norm, RoPE, MXFP8 stored straight into the SWA ring | decode rows | `kv_swa_store` |
//! | `_norm_kernel` | `q_a_norm` | decode rows | `q_a_norm` |
//! | `_compress_and_update_state_decode_kernel` (+ `_1`) | ratio-2 KV source compressor | decode rows | `compress_decode` |
//! | fused transform | main KV store, index Q, index K store (as the prompt step) | rows completing a compressed entry / decode rows | `main_store`, `index_q`, `index_k_store` |
//! | `_get_candidate_compact_lens_kernel` | first consumer | decode rows | `compact_lens` |
//! | `_restore_candidate_indices_inplace_kernel` | consumers, one launch | decode rows | `restore` |
//! | `_convert_kv_index_kernel` | index sources: top-k slots to cache slots (`convert_kv_index`) | decode rows | `convert_kv_index` |
//!
//! No paged loads, SWA copy or index assembly run in decode: the decode
//! attention reads the caches through the converted indices.

use std::sync::Arc;

use super::deepseek_v41_common::{eval_or_zero, placeholder};
use crate::op::attention::deepseek_v41_deepinfer_indexer::chunk_rows;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    CandidateCompactLensKernel, CandidateCompactLensKernelConfig, CandidateCompactLensKernelInput,
    CandidateIndexRestoreKernel, CandidateIndexRestoreKernelConfig,
    CandidateIndexRestoreKernelInput, CompressStatePrefillKernel, CompressStatePrefillKernelConfig,
    CompressStatePrefillKernelInput, ElementwiseKernel, ElementwiseKernelConfig,
    ElementwiseKernelInput, ExpandSwaIndexKernel, ExpandSwaIndexKernelConfig,
    ExpandSwaIndexKernelInput, FusedNormRopeCastKernel, FusedNormRopeCastKernelConfig,
    FusedNormRopeCastKernelInput, KvIndexCatKernel, KvIndexCatKernelConfig, KvIndexCatKernelInput,
    PagedKvCacheLoadKernel, PagedKvCacheLoadKernelConfig, PagedKvCacheLoadKernelInput,
    PagedKvCacheStoreKernel, PagedKvCacheStoreKernelConfig, PagedKvCacheStoreKernelInput,
    RmsNormMxfp8Kernel, RmsNormMxfp8KernelConfig, RmsNormMxfp8KernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, LeafMetrics, PerfApiBridge};

/// One layer's geometry and roles, as the attention worklet resolves them.
#[derive(Clone, Debug)]
pub struct DeepseekV41DeepInferKvConfig {
    pub gpu_name: String,
    pub kv_backends: Vec<&'static str>,
    pub memcpy_backends: Vec<&'static str>,
    /// Placeholder backends of the launches with no kind.
    pub elementwise_backends: Vec<&'static str>,
    pub q_lora_rank: u32,
    pub head_dim: u32,
    pub rope_dim: u32,
    pub index_num_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub window_size: u32,
    /// 0 (SWA only), 1 or 2.
    pub compress_ratio: u32,
    pub kv_source: bool,
    pub index_source: bool,
    pub index_owner: bool,
    pub candidate_consumer: bool,
    pub candidate_metadata: bool,
    pub candidate_topk_blocks: u32,
    /// Key tokens per candidate block before compression.
    pub candidate_block_size: u32,
    /// Tokens per KV page; a compressed page holds `block_size / ratio`.
    pub block_size: u32,
    /// Tokens of one request's SWA ring (`window + backtracking`).
    pub swa_ring_tokens: u32,
    pub builds_swa_index: bool,
    pub max_tokens_per_rank: u32,
    pub max_model_len: u32,
}

#[derive(Clone, Debug)]
pub struct DeepseekV41DeepInferKvResolved {
    pub q_a_norm: RmsNormMxfp8KernelConfig,
    pub kv_swa: FusedNormRopeCastKernelConfig,
    pub kv_swa_copy: ElementwiseKernelConfig,
    pub swa_load: PagedKvCacheLoadKernelConfig,
    pub swa_store: PagedKvCacheStoreKernelConfig,
    pub swa_index: Option<ExpandSwaIndexKernelConfig>,
    pub compress: Option<CompressStatePrefillKernelConfig>,
    pub main_store: Option<FusedNormRopeCastKernelConfig>,
    pub main_load: Option<PagedKvCacheLoadKernelConfig>,
    pub index_q: Option<FusedNormRopeCastKernelConfig>,
    pub index_k_store: Option<FusedNormRopeCastKernelConfig>,
    pub index_k_load: Option<PagedKvCacheLoadKernelConfig>,
    pub compact_lens: Option<CandidateCompactLensKernelConfig>,
    pub restore: Option<CandidateIndexRestoreKernelConfig>,
    /// `index_attn.py` `compute_chunk_size`: `_topk_prefill` restores one
    /// chunk of rows per launch.
    pub restore_chunk_rows: u32,
    pub kv_index_cat: Option<KvIndexCatKernelConfig>,
    /// Decode: the SWA row stored straight into its ring.
    pub kv_swa_store: FusedNormRopeCastKernelConfig,
    /// Decode: `compress_and_update_state_decode` (two launches).
    pub compress_decode: Option<ElementwiseKernelConfig>,
    /// Decode: `convert_kv_index` (index sources).
    pub convert_kv_index: Option<ElementwiseKernelConfig>,
    pub compress_ratio: u32,
}

#[derive(Clone, Debug, Default)]
pub struct DeepseekV41DeepInferKvInput {
    /// `(query_len, context_len)` per prefill request.
    pub prefill_query_context_pairs: Vec<(u32, u32)>,
    pub write_kv_only: bool,
    /// The YOCO cut's `write_kv` already stored this layer's main / index KV.
    pub kv_written: bool,
    /// Resident KV after each decode row (a decode step's rows).
    pub decode_kv_lens: Vec<u32>,
    /// Rows the decode launches run (the graph's padded rows).
    pub decode_rows: u32,
}

pub struct DeepseekV41DeepInferKv {
    pub q_a_norm: Op<RmsNormMxfp8Kernel>,
    pub kv_swa: Op<FusedNormRopeCastKernel>,
    pub kv_swa_copy: Op<ElementwiseKernel>,
    pub swa_load: Op<PagedKvCacheLoadKernel>,
    pub swa_store: Op<PagedKvCacheStoreKernel>,
    pub swa_index: Option<Op<ExpandSwaIndexKernel>>,
    pub compress: Option<Op<CompressStatePrefillKernel>>,
    pub main_store: Option<Op<FusedNormRopeCastKernel>>,
    pub main_load: Option<Op<PagedKvCacheLoadKernel>>,
    pub index_q: Option<Op<FusedNormRopeCastKernel>>,
    pub index_k_store: Option<Op<FusedNormRopeCastKernel>>,
    pub index_k_load: Option<Op<PagedKvCacheLoadKernel>>,
    pub compact_lens: Option<Op<CandidateCompactLensKernel>>,
    pub restore: Option<Op<CandidateIndexRestoreKernel>>,
    pub kv_index_cat: Option<Op<KvIndexCatKernel>>,
    pub kv_swa_store: Op<FusedNormRopeCastKernel>,
    pub compress_decode: Option<Op<ElementwiseKernel>>,
    pub convert_kv_index: Option<Op<ElementwiseKernel>>,
    resolved: DeepseekV41DeepInferKvResolved,
}

impl DeepseekV41DeepInferKv {
    pub fn resolve(cfg: &DeepseekV41DeepInferKvConfig) -> DeepseekV41DeepInferKvResolved {
        let ratio = cfg.compress_ratio;
        assert!(ratio <= 2, "compress ratio must be 0, 1 or 2");
        assert!(
            ratio > 0 || !(cfg.kv_source || cfg.index_source),
            "ratio-0 layers have no compressed KV"
        );
        let gpu = || cfg.gpu_name.clone();
        let backends = || cfg.kv_backends.clone();
        let rows = cfg.max_tokens_per_rank;
        let page = cfg.block_size / ratio.max(1);
        let fused = |heads: u32, dim: u32, norm: bool, cast: &str, output: &str, block: u32| {
            FusedNormRopeCastKernelConfig {
                backends: backends(),
                gpu_name: gpu(),
                num_heads: heads,
                head_dim: dim,
                rope_dim: cfg.rope_dim,
                norm,
                cast_format: cast.to_string(),
                output: output.to_string(),
                block_size: block,
                dtype: DType::Bf16,
                max_tokens_per_rank: rows,
            }
        };
        // A prefill load reads the whole context: its key count runs to
        // `max_model_len` per request, past it by the cache's extrapolation.
        let load = |block: u32, segment: u32, format: &str| PagedKvCacheLoadKernelConfig {
            backends: backends(),
            gpu_name: gpu(),
            block_size: block,
            segment_tokens: segment,
            cache_format: format.to_string(),
            max_tokens: cfg.max_model_len,
        };
        let candidate_block = cfg.candidate_block_size / ratio.max(1);
        // The consumer's logits rows are its candidate tokens (the indexer
        // op's `chunk_rows`).
        let restore_chunk_rows = if cfg.candidate_consumer {
            chunk_rows(cfg.candidate_topk_blocks * candidate_block)
        } else {
            0
        };
        DeepseekV41DeepInferKvResolved {
            // attn.py `q_a_norm` with fp8_attn: RMSNorm + MXFP8 of q_a.
            q_a_norm: RmsNormMxfp8KernelConfig {
                backends: backends(),
                gpu_name: gpu(),
                hidden: cfg.q_lora_rank,
                return_bf16: false,
                dtype: DType::Bf16,
                max_tokens_per_rank: rows,
            },
            // attn.py `_prepare_swa_kv` (prompt): MXFP8 cast back to bf16.
            kv_swa: fused(1, cfg.head_dim, true, "mxfp8", "cast_back", 0),
            kv_swa_copy: ElementwiseKernelConfig {
                backends: cfg.memcpy_backends.clone(),
                gpu_name: gpu(),
                input_bytes_per_token: (cfg.head_dim * 2).into(),
                output_bytes_per_token: (cfg.head_dim * 2).into(),
            },
            // One `window` run from each request's SWA ring.
            swa_load: load(cfg.swa_ring_tokens, cfg.window_size, "mxfp8"),
            swa_store: PagedKvCacheStoreKernelConfig {
                backends: backends(),
                gpu_name: gpu(),
                block_size: cfg.swa_ring_tokens,
                cache_format: "mxfp8".to_string(),
                dtype: DType::Bf16,
                max_tokens_per_rank: rows,
            },
            swa_index: cfg.builds_swa_index.then(|| ExpandSwaIndexKernelConfig {
                backends: backends(),
                gpu_name: gpu(),
                window_size: cfg.window_size,
                max_tokens_per_rank: rows,
            }),
            // compress.py `forward_pre_norm`: ratio 1 passes the projection through.
            compress: (cfg.kv_source && ratio > 1).then(|| CompressStatePrefillKernelConfig {
                backends: backends(),
                gpu_name: gpu(),
                head_dim: cfg.head_dim,
                compress_ratio: ratio,
                // cache_engine.py `get_inference_compress_block_size`.
                state_size: ratio + cfg.swa_ring_tokens - cfg.window_size - 1,
                max_tokens_per_rank: rows,
            }),
            // compress.py `forward_store`: the KProj indexer reads the post-norm rows.
            main_store: cfg.kv_source.then(|| {
                let output = if cfg.index_source {
                    "cache_and_norm"
                } else {
                    "cache"
                };
                fused(1, cfg.head_dim, true, "nvfp4", output, page)
            }),
            main_load: cfg.kv_source.then(|| load(page, page, "nvfp4")),
            // index_attn.py `_quantize_q`.
            index_q: cfg.index_source.then(|| {
                fused(
                    cfg.index_num_heads,
                    cfg.index_head_dim,
                    false,
                    "mxfp4",
                    "quantized",
                    0,
                )
            }),
            // index_attn.py `store_index_kv`.
            index_k_store: cfg
                .index_owner
                .then(|| fused(1, cfg.index_head_dim, true, "mxfp4", "cache", page)),
            index_k_load: cfg.index_owner.then(|| load(page, page, "mxfp4")),
            compact_lens: cfg
                .candidate_metadata
                .then(|| CandidateCompactLensKernelConfig {
                    backends: backends(),
                    gpu_name: gpu(),
                    topk_blocks: cfg.candidate_topk_blocks,
                    block_size: candidate_block,
                    max_tokens_per_rank: rows,
                }),
            restore: cfg
                .candidate_consumer
                .then(|| CandidateIndexRestoreKernelConfig {
                    backends: backends(),
                    gpu_name: gpu(),
                    topk_tokens: cfg.index_topk,
                    topk_blocks: cfg.candidate_topk_blocks,
                    block_size: candidate_block,
                    max_tokens_per_rank: rows.min(restore_chunk_rows),
                }),
            restore_chunk_rows,
            kv_index_cat: (ratio > 0).then(|| KvIndexCatKernelConfig {
                backends: backends(),
                gpu_name: gpu(),
                window_size: cfg.window_size,
                topk: cfg.index_topk,
                max_tokens_per_rank: rows,
            }),
            // attn.py `_prepare_swa_kv` with `KVCacheOutArgs(swa_cache, ...)`:
            // one MXFP8 row per decode row into the request's SWA ring.
            kv_swa_store: fused(1, cfg.head_dim, true, "mxfp8", "cache", cfg.swa_ring_tokens),
            // compress.py `forward_pre_norm` (decode): each row's fp32
            // `[kv, score]` pair in, the request's compressor state updated
            // and a completed entry out (two launches, no kind).
            compress_decode: (cfg.kv_source && ratio > 1).then(|| {
                placeholder(&cfg.elementwise_backends, &cfg.gpu_name, 2 * cfg.head_dim * 4, 2 * cfg.head_dim * 4)
            }),
            // index_attn.py `convert_kv_index`: each row's top-k logical
            // indices and their block-table entries in, cache slots out.
            convert_kv_index: cfg.index_source.then(|| {
                placeholder(&cfg.elementwise_backends, &cfg.gpu_name, 2 * 4 * cfg.index_topk, 4 * cfg.index_topk)
            }),
            compress_ratio: ratio,
        }
    }

    pub fn build(
        name: &str,
        resolved: DeepseekV41DeepInferKvResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        macro_rules! op {
            ($kernel:ty, $cfg:expr, $suffix:literal) => {{
                let op_name = format!("{name}.{}", $suffix);
                Op::new(
                    op_name.clone(),
                    Arc::new(<$kernel>::build(op_name, $cfg, bridge)?),
                )
            }};
        }
        macro_rules! maybe {
            ($kernel:ty, $cfg:expr, $suffix:literal) => {
                match $cfg {
                    Some(config) => Some(op!($kernel, config, $suffix)),
                    None => None,
                }
            };
        }
        let r = resolved.clone();
        Ok(Self {
            q_a_norm: op!(RmsNormMxfp8Kernel, r.q_a_norm, "main.q_a_norm"),
            kv_swa: op!(
                FusedNormRopeCastKernel,
                r.kv_swa,
                "main.kv_swa_norm_rope_cast"
            ),
            kv_swa_copy: op!(ElementwiseKernel, r.kv_swa_copy, "attention.kv_swa_copy"),
            swa_load: op!(PagedKvCacheLoadKernel, r.swa_load, "attention.swa_load"),
            swa_store: op!(PagedKvCacheStoreKernel, r.swa_store, "attention.swa_store"),
            swa_index: maybe!(ExpandSwaIndexKernel, r.swa_index, "attention.swa_index"),
            compress: maybe!(
                CompressStatePrefillKernel,
                r.compress,
                "compressor.compress_state"
            ),
            main_store: maybe!(FusedNormRopeCastKernel, r.main_store, "compressor.kv_store"),
            main_load: maybe!(PagedKvCacheLoadKernel, r.main_load, "attention.main_load"),
            index_q: maybe!(FusedNormRopeCastKernel, r.index_q, "indexer.q_rope_cast"),
            index_k_store: maybe!(
                FusedNormRopeCastKernel,
                r.index_k_store,
                "indexer.k_norm_rope_cast_store"
            ),
            index_k_load: maybe!(PagedKvCacheLoadKernel, r.index_k_load, "indexer.k_load"),
            compact_lens: maybe!(
                CandidateCompactLensKernel,
                r.compact_lens,
                "indexer.candidate_compact_lens"
            ),
            restore: maybe!(
                CandidateIndexRestoreKernel,
                r.restore,
                "indexer.candidate_restore"
            ),
            kv_index_cat: maybe!(KvIndexCatKernel, r.kv_index_cat, "attention.kv_index_cat"),
            kv_swa_store: op!(
                FusedNormRopeCastKernel,
                r.kv_swa_store,
                "main.kv_swa_store_decode"
            ),
            compress_decode: maybe!(
                ElementwiseKernel,
                r.compress_decode,
                "compressor.compress_state_decode"
            ),
            convert_kv_index: maybe!(
                ElementwiseKernel,
                r.convert_kv_index,
                "indexer.convert_kv_index"
            ),
            resolved,
        })
    }

    /// The launches before the indexer scoring (projections' consumers).
    pub fn compile_front(&self, b: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut children = vec![
            self.q_a_norm.compile(b),
            self.kv_swa.compile(b),
            self.kv_swa_store.compile(b),
        ];
        for op in [&self.compress].into_iter().flatten() {
            children.push(op.compile(b));
        }
        if let Some(op) = &self.compress_decode {
            children.push(CostNode::Scale {
                n: 2,
                child: Box::new(op.compile(b)),
            });
        }
        for op in [&self.main_store, &self.index_q, &self.index_k_store]
            .into_iter()
            .flatten()
        {
            children.push(op.compile(b));
        }
        for op in [&self.main_load, &self.index_k_load].into_iter().flatten() {
            children.push(op.compile(b));
        }
        children
    }

    /// The launches between the scoring and the prefill attention.
    pub fn compile_back(&self, b: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut children = Vec::new();
        if let Some(op) = &self.compact_lens {
            children.push(op.compile(b));
        }
        if let Some(op) = &self.restore {
            children.push(op.compile(b));
        }
        children.push(self.kv_swa_copy.compile(b));
        children.push(self.swa_load.compile(b));
        children.push(self.swa_store.compile(b));
        if let Some(op) = &self.swa_index {
            children.push(op.compile(b));
        }
        if let Some(op) = &self.kv_index_cat {
            children.push(op.compile(b));
        }
        if let Some(op) = &self.convert_kv_index {
            children.push(op.compile(b));
        }
        children
    }

    pub fn eval_front(&self, input: &DeepseekV41DeepInferKvInput, ev: &mut Evaluator) {
        let pairs = &input.prefill_query_context_pairs;
        let prefill_rows: u32 = pairs.iter().map(|&(q, _)| q).sum();
        let decode_rows = if input.decode_kv_lens.is_empty() {
            0
        } else {
            input.decode_rows
        };
        let rows = prefill_rows + decode_rows;
        let zero = rows == 0;
        // `_store_yoco_full_kv` stores; the full attention path does not run.
        let attend = zero || input.write_kv_only;
        let store = zero || input.kv_written;
        let ratio = self.resolved.compress_ratio.max(1);
        // prepare_tensor_input_v4.cpp: each request stores the compressed rows
        // its new tokens complete (at least one entry), and loads its whole
        // compressed context. A decode row completes an entry when its
        // position closes a `ratio` group.
        let stored: u32 = pairs
            .iter()
            .map(|&(q, ctx)| (ctx / ratio - (ctx - q) / ratio).max(1))
            .sum::<u32>()
            + input
                .decode_kv_lens
                .iter()
                .filter(|&&kv| kv % ratio == 0)
                .count() as u32;
        let loaded: u32 = pairs.iter().map(|&(_, ctx)| (ctx / ratio).max(1)).sum();
        let tokens = |num_tokens| FusedNormRopeCastKernelInput { num_tokens };
        eval_or_zero(
            &self.q_a_norm,
            RmsNormMxfp8KernelInput { num_tokens: rows },
            attend,
            ev,
        );
        eval_or_zero(&self.kv_swa, tokens(prefill_rows), attend || prefill_rows == 0, ev);
        eval_or_zero(
            &self.kv_swa_store,
            tokens(decode_rows),
            input.write_kv_only || decode_rows == 0,
            ev,
        );
        if let Some(op) = &self.compress {
            eval_or_zero(
                op,
                CompressStatePrefillKernelInput { num_tokens: prefill_rows },
                store || prefill_rows == 0,
                ev,
            );
        }
        if let Some(op) = &self.compress_decode {
            eval_or_zero(
                op,
                ElementwiseKernelInput { num_tokens: decode_rows },
                decode_rows == 0,
                ev,
            );
        }
        if let Some(op) = &self.main_store {
            eval_or_zero(op, tokens(stored), store || stored == 0, ev);
        }
        if let Some(op) = &self.index_q {
            eval_or_zero(op, tokens(rows), attend, ev);
        }
        if let Some(op) = &self.index_k_store {
            eval_or_zero(op, tokens(stored), store || stored == 0, ev);
        }
        for op in [&self.main_load, &self.index_k_load].into_iter().flatten() {
            eval_or_zero(
                op,
                PagedKvCacheLoadKernelInput { num_tokens: loaded },
                attend || prefill_rows == 0,
                ev,
            );
        }
    }

    pub fn eval_back(&self, input: &DeepseekV41DeepInferKvInput, ev: &mut Evaluator) {
        let pairs = &input.prefill_query_context_pairs;
        let rows: u32 = pairs.iter().map(|&(q, _)| q).sum();
        let skip = rows == 0 || input.write_kv_only;
        let decode_rows = if input.decode_kv_lens.is_empty() || input.write_kv_only {
            0
        } else {
            input.decode_rows
        };
        // The candidate launches run once over a decode step's rows.
        let scored = if skip { decode_rows } else { rows + decode_rows };
        if let Some(op) = &self.compact_lens {
            eval_or_zero(
                op,
                CandidateCompactLensKernelInput { num_rows: scored },
                scored == 0,
                ev,
            );
        }
        if let Some(op) = &self.restore {
            let chunk = self.resolved.restore_chunk_rows;
            let restore = |num_rows| CandidateIndexRestoreKernelInput { num_rows };
            let mut metrics = LeafMetrics::ZERO;
            if !skip {
                for start in (0..rows).step_by(chunk as usize) {
                    metrics.add_fanin(op.kernel.eval(&restore(chunk.min(rows - start))));
                }
            }
            // `_topk_decode`: one restore over the decode rows.
            if decode_rows > 0 {
                metrics.add_fanin(op.kernel.eval(&restore(decode_rows)));
            }
            ev.push(metrics, || restore(scored.min(chunk)).into());
        }
        eval_or_zero(
            &self.kv_swa_copy,
            ElementwiseKernelInput { num_tokens: rows },
            skip,
            ev,
        );
        // attn.py: `window` SWA keys per request (`total_k_swa`).
        let swa_keys = pairs.len() as u32 * self.resolved.swa_load.segment_tokens;
        eval_or_zero(
            &self.swa_load,
            PagedKvCacheLoadKernelInput {
                num_tokens: swa_keys,
            },
            skip,
            ev,
        );
        eval_or_zero(
            &self.swa_store,
            PagedKvCacheStoreKernelInput { num_tokens: rows },
            skip,
            ev,
        );
        if let Some(op) = &self.swa_index {
            eval_or_zero(op, ExpandSwaIndexKernelInput { num_rows: rows }, skip, ev);
        }
        if let Some(op) = &self.kv_index_cat {
            eval_or_zero(op, KvIndexCatKernelInput { num_rows: rows }, skip, ev);
        }
        if let Some(op) = &self.convert_kv_index {
            eval_or_zero(
                op,
                ElementwiseKernelInput {
                    num_tokens: decode_rows,
                },
                decode_rows == 0,
                ev,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn config(layer: u32) -> DeepseekV41DeepInferKvConfig {
        let kv_source = [2, 8, 14, 20].contains(&layer);
        let index_source = [2, 8, 14, 20, 24, 28, 32, 36].contains(&layer);
        DeepseekV41DeepInferKvConfig {
            gpu_name: "NVIDIA B200".into(),
            kv_backends: vec!["tilelang"],
            memcpy_backends: vec!["torch_memcpy_deepinfer"],
            elementwise_backends: vec!["triton"],
            q_lora_rank: 1280,
            head_dim: 512,
            rope_dim: 64,
            index_num_heads: 32,
            index_head_dim: 128,
            index_topk: 512,
            window_size: 128,
            compress_ratio: match layer {
                0..=1 => 0,
                2..=19 => 2,
                _ => 1,
            },
            kv_source,
            index_source,
            index_owner: kv_source,
            candidate_consumer: index_source && !kv_source,
            candidate_metadata: layer == 24,
            candidate_topk_blocks: 2048,
            candidate_block_size: 8,
            block_size: 128,
            swa_ring_tokens: 134,
            builds_swa_index: layer == 0,
            max_tokens_per_rank: 65536,
            max_model_len: 131072,
        }
    }

    #[test]
    fn kv_sources_own_the_store_and_loads() {
        let r = DeepseekV41DeepInferKv::resolve(&config(2));
        assert_eq!(r.compress.as_ref().unwrap().state_size, 7);
        let main = r.main_store.as_ref().unwrap();
        assert_eq!(
            (main.block_size, main.output.as_str()),
            (64, "cache_and_norm")
        );
        assert_eq!(r.main_load.as_ref().unwrap().block_size, 64);
        assert!(r.restore.is_none() && r.swa_index.is_none());

        let r = DeepseekV41DeepInferKv::resolve(&config(20));
        assert!(r.compress.is_none(), "ratio 1 has no compressor launch");
        assert_eq!(r.index_k_store.as_ref().unwrap().block_size, 128);

        let r = DeepseekV41DeepInferKv::resolve(&config(24));
        assert!(r.main_store.is_none() && r.index_k_load.is_none());
        assert_eq!(r.restore.as_ref().unwrap().block_size, 8);
        assert!(r.compact_lens.is_some() && r.index_q.is_some());

        let r = DeepseekV41DeepInferKv::resolve(&config(0));
        assert!(r.kv_index_cat.is_none() && r.swa_index.is_some());
        assert_eq!(r.swa_load.block_size, 134);
    }
}
