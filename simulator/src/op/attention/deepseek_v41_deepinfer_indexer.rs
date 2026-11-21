//! DeepSeek-V4.1 sparse-indexer prefill scoring as DeepInfer runs it
//! (deepseek-ai/DeepInfer a679bb8, `model_executor/models/v4_modules/
//! index_attn.py` `_topk_prefill`, `_compute_prefill_p_sum`,
//! `_topk_candidate_block`).
//!
//! The step's prefill rows (every request's rows, concatenated) are scored in
//! chunks of `compute_chunk_size` rows (see [`chunk_rows`]); each chunk runs,
//! in order:
//!
//! - a dense index source (layers 2/8/14, and the candidate writer 20):
//!   DeepGEMM `fp8_fp4_mqa_logits` ([`MqaLogitsPrefillChunkKernel`]);
//! - a candidate consumer (24/28/32/36): the sparse schedule
//!   (`get_sparse_mqa_logits_metadata`, [`SparseMqaLogitsMetadataKernel`];
//!   built once per chunk and step by the first consumer and cached for the
//!   rest, so only a layer with `candidate_metadata` launches it), then
//!   `fp8_fp4_sparse_mqa_logits` ([`SparseMqaLogitsPrefillKernel`]);
//! - the candidate writer only: `candidate_block_max`
//!   ([`CandidateBlockMaxKernel`]) and the DeepSelect top-k over the block
//!   scores, `candidate_topk_blocks` wide;
//! - every layer: the DeepSelect top-k of the row's `index_topk` keys
//!   ([`DeepSelectTopkKernel`]).
//!
//! Each slot is one leaf; a step's chunks are summed into it (the chunk loop
//! is the callable's runtime loop, not a slot).
//!
//! Keys (the lane's measured proxies, `logs/20261115_0_lane_reports/
//! indexer.txt`): a row with context `x` (its own token included) scores
//! `floor(x / ratio)` compressed keys (`_write_ks_ke`). A chunk of `q` rows is
//! priced as one request of `q` rows whose mean key window equals the chunk's
//! ([`equivalent_context`]), and the top-k as uniform rows of the chunk's mean
//! valid length ([`ChunkWork::row_len`]).
//!
//! Decode rows (`_topk_decode`, `_compute_decode_p_sum`; capture
//! `logs/20261120_0_deepinfer_decode_nsys`, per-call list in the lane's
//! `idx_calls_graph.txt`) run once per step over all `batch x next_n` rows:
//!
//! - a dense index source and the candidate writer: the FP4 paged logits
//!   ([`DsaPagedMqaLogitsDecodeKernel`], backend `deepgemm_fp4`), after the
//!   paged schedule ([`PagedMqaLogitsMetadataKernel`]) that the first such
//!   layer of each compress ratio builds and the rest reuse
//!   (`_get_decode_metadata`, `paged_metadata`);
//! - a candidate consumer: the paged sparse schedule
//!   ([`SparseMqaLogitsDecodeMetadataKernel`], first consumer only) and the
//!   candidate-restricted logits ([`SparseMqaLogitsDecodeKernel`]);
//! - the candidate writer: `candidate_block_max` and the block top-k;
//! - every layer: the row top-k.
//!
//! A decode request's keys are `floor(context / ratio)` of its rows' mean
//! context; the kernels take the request count and that key window.
//!
//! Not priced here (the KV worklet's, [`crate::worklet::deepseek_v41_deepinfer_kv`]):
//! `_get_candidate_compact_lens_kernel` (once per step, first consumer),
//! `restore_candidate_indices_inplace` (after each consumer top-k) and
//! `convert_kv_index`.

use std::sync::Arc;

use crate::op::attention::DeepseekV41CandidateRole;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    CandidateBlockMaxKernel, CandidateBlockMaxKernelConfig, CandidateBlockMaxKernelInput,
    DeepSelectTopkKernel, DeepSelectTopkKernelConfig, DeepSelectTopkKernelInput,
    DsaPagedMqaLogitsDecodeKernel, DsaPagedMqaLogitsDecodeKernelConfig,
    DsaPagedMqaLogitsDecodeKernelInput, MqaLogitsPrefillChunkKernel,
    MqaLogitsPrefillChunkKernelConfig, MqaLogitsPrefillChunkKernelInput,
    PagedMqaLogitsMetadataKernel, PagedMqaLogitsMetadataKernelConfig,
    PagedMqaLogitsMetadataKernelInput, SparseMqaLogitsDecodeKernel,
    SparseMqaLogitsDecodeKernelConfig, SparseMqaLogitsDecodeKernelInput,
    SparseMqaLogitsDecodeMetadataKernel, SparseMqaLogitsDecodeMetadataKernelConfig,
    SparseMqaLogitsDecodeMetadataKernelInput, SparseMqaLogitsMetadataKernel,
    SparseMqaLogitsMetadataKernelConfig, SparseMqaLogitsMetadataKernelInput,
    SparseMqaLogitsPrefillKernel, SparseMqaLogitsPrefillKernelConfig,
    SparseMqaLogitsPrefillKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

/// `compute_chunk_size`'s ceiling (`index_attn.py`: "min'ed with 8192 because
/// mqa downclocks when the chunk size is large").
const MAX_CHUNK_ROWS: u64 = 8192;
/// `DEEPINFER_INDEX_ATTN_MEMORY_LIMIT` default (`environ.py`): the bf16
/// logits one chunk may hold.
const LOGITS_MEMORY_LIMIT: u64 = 10 << 30;
/// The logits (`psum_dtype = torch.bfloat16`) element size.
const LOGIT_BYTES: u64 = 2;

#[derive(Clone, Debug)]
pub struct DeepseekV41DeepInferIndexerOpConfig {
    pub gpu_name: String,
    /// The index source's compress ratio (1 or 2).
    pub compress_ratio: u32,
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub index_topk: u32,
    pub candidate: DeepseekV41CandidateRole,
    /// The candidate consumer that builds the step's sparse schedule.
    pub candidate_metadata: bool,
    /// Candidate blocks per row and tokens per block before compression.
    pub candidate_topk_blocks: u32,
    pub candidate_block_size: u32,
    /// `runtime_max_model_len`: the logits row width is `max_model_len / ratio`.
    pub max_model_len: u32,
    pub logits_backends: Vec<&'static str>,
    pub candidate_block_max_backends: Vec<&'static str>,
    pub topk_backends: Vec<&'static str>,
    /// The FP4 dense decode logits (`dsa_paged_mqa_logits_decode`).
    pub decode_logits_backends: Vec<&'static str>,
    /// This layer builds its ratio's dense decode schedule.
    pub paged_metadata: bool,
    /// Query rows per decode request: 1, or the verify width.
    pub decode_next_n: u32,
    /// DeepInfer's KV page in tokens; the index page holds `block_size / ratio` keys.
    pub block_size: u32,
}

#[derive(Clone, Debug, Default)]
pub struct DeepseekV41DeepInferIndexerOpInput {
    /// `(query_len, context_len)` per prefill request, context including the
    /// query.
    pub prefill_query_context_pairs: Vec<(u32, u32)>,
    /// Resident KV before each live decode row.
    pub decode_kv_lens: Vec<u32>,
    /// Decode rows the launches cover: the graph's padded rows inside a FULL
    /// graph, else the live rows.
    pub decode_rows: u32,
}

#[derive(Clone, Debug)]
pub struct DeepseekV41DeepInferIndexerOpResolved {
    pub compress_ratio: u32,
    /// Compressed tokens per candidate block (0 off the candidate scheme).
    pub candidate_block: u32,
    /// `candidate_topk_blocks * candidate_block` on a consumer, else 0.
    pub candidate_tokens: u32,
    pub chunk_rows: u32,
    pub dense_logits: Option<MqaLogitsPrefillChunkKernelConfig>,
    pub sparse_metadata: Option<SparseMqaLogitsMetadataKernelConfig>,
    pub sparse_logits: Option<SparseMqaLogitsPrefillKernelConfig>,
    pub candidate_block_max: Option<CandidateBlockMaxKernelConfig>,
    pub candidate_topk: Option<DeepSelectTopkKernelConfig>,
    pub topk: DeepSelectTopkKernelConfig,
    pub decode_next_n: u32,
    pub decode_logits: Option<DsaPagedMqaLogitsDecodeKernelConfig>,
    pub decode_metadata: Option<PagedMqaLogitsMetadataKernelConfig>,
    pub decode_sparse_metadata: Option<SparseMqaLogitsDecodeMetadataKernelConfig>,
    pub decode_sparse_logits: Option<SparseMqaLogitsDecodeKernelConfig>,
}

pub struct DeepseekV41DeepInferIndexerOp {
    pub name: String,
    pub dense_logits: Option<Arc<MqaLogitsPrefillChunkKernel>>,
    pub sparse_metadata: Option<Arc<SparseMqaLogitsMetadataKernel>>,
    pub sparse_logits: Option<Arc<SparseMqaLogitsPrefillKernel>>,
    pub candidate_block_max: Option<Arc<CandidateBlockMaxKernel>>,
    pub candidate_topk: Option<Arc<DeepSelectTopkKernel>>,
    pub topk: Arc<DeepSelectTopkKernel>,
    pub decode_logits: Option<Arc<DsaPagedMqaLogitsDecodeKernel>>,
    pub decode_metadata: Option<Arc<PagedMqaLogitsMetadataKernel>>,
    pub decode_sparse_metadata: Option<Arc<SparseMqaLogitsDecodeMetadataKernel>>,
    pub decode_sparse_logits: Option<Arc<SparseMqaLogitsDecodeKernel>>,
    pub decode_candidate_block_max: Option<Arc<CandidateBlockMaxKernel>>,
    pub decode_candidate_topk: Option<Arc<DeepSelectTopkKernel>>,
    pub decode_topk: Arc<DeepSelectTopkKernel>,
    resolved: DeepseekV41DeepInferIndexerOpResolved,
}

/// `compute_chunk_size = min(8192, LIMIT // psum_row_bytes // 8 * 8)`.
pub fn chunk_rows(row_width: u32) -> u32 {
    let row_bytes = u64::from(row_width) * LOGIT_BYTES;
    let rows = (LOGITS_MEMORY_LIMIT / row_bytes / 8 * 8).min(MAX_CHUNK_ROWS);
    assert!(
        rows > 0,
        "one logits row of {row_bytes} B exceeds the chunk budget"
    );
    rows as u32
}

impl DeepseekV41DeepInferIndexerOp {
    pub fn resolve(
        cfg: &DeepseekV41DeepInferIndexerOpConfig,
    ) -> DeepseekV41DeepInferIndexerOpResolved {
        let ratio = cfg.compress_ratio;
        assert!(
            matches!(ratio, 1 | 2),
            "V4.1 index sources are ratio 1 or 2, got {ratio}"
        );
        let consumer = cfg.candidate == DeepseekV41CandidateRole::Consumer;
        let writer = cfg.candidate == DeepseekV41CandidateRole::Writer;
        assert!(
            !cfg.candidate_metadata || consumer,
            "only a candidate consumer builds the sparse schedule"
        );
        let candidate_block = if consumer || writer {
            assert_eq!(
                cfg.candidate_block_size % ratio,
                0,
                "candidate blocks must hold whole compressed tokens"
            );
            cfg.candidate_block_size / ratio
        } else {
            0
        };
        let candidate_tokens = if consumer {
            cfg.candidate_topk_blocks * candidate_block
        } else {
            0
        };
        let max_keys = cfg.max_model_len / ratio;
        assert!(cfg.decode_next_n > 0, "decode rows per request must be positive");
        assert_eq!(cfg.block_size % ratio, 0, "an index page holds whole keys");
        let page = cfg.block_size / ratio;
        let gpu = cfg.gpu_name.clone();
        let logits_dtype = DType::Mxfp4E2m1;
        let topk = |top_k: u32, max_row_len: u32| DeepSelectTopkKernelConfig {
            backends: cfg.topk_backends.clone(),
            gpu_name: gpu.clone(),
            top_k,
            dtype: DType::Bf16,
            max_row_len,
        };
        DeepseekV41DeepInferIndexerOpResolved {
            compress_ratio: ratio,
            candidate_block,
            candidate_tokens,
            chunk_rows: chunk_rows(if consumer { candidate_tokens } else { max_keys }),
            dense_logits: (!consumer).then(|| MqaLogitsPrefillChunkKernelConfig {
                backends: cfg.logits_backends.clone(),
                gpu_name: gpu.clone(),
                compress_ratio: ratio,
                num_heads: cfg.num_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                dtype: logits_dtype,
                max_context_len: cfg.max_model_len,
            }),
            sparse_metadata: cfg
                .candidate_metadata
                .then(|| SparseMqaLogitsMetadataKernelConfig {
                    backends: cfg.logits_backends.clone(),
                    gpu_name: gpu.clone(),
                    compress_ratio: ratio,
                    candidate_blocks: cfg.candidate_topk_blocks,
                    candidate_block_size: candidate_block,
                    dtype: logits_dtype,
                    max_context_len: cfg.max_model_len,
                }),
            sparse_logits: consumer.then(|| SparseMqaLogitsPrefillKernelConfig {
                backends: cfg.logits_backends.clone(),
                gpu_name: gpu.clone(),
                compress_ratio: ratio,
                candidate_blocks: cfg.candidate_topk_blocks,
                candidate_block_size: candidate_block,
                num_heads: cfg.num_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                dtype: logits_dtype,
                max_context_len: cfg.max_model_len,
            }),
            candidate_block_max: writer.then(|| CandidateBlockMaxKernelConfig {
                backends: cfg.candidate_block_max_backends.clone(),
                gpu_name: gpu.clone(),
                compress_ratio: ratio,
                block_size: candidate_block,
                dtype: DType::Bf16,
                max_context_len: cfg.max_model_len,
            }),
            candidate_topk: writer.then(|| {
                topk(
                    cfg.candidate_topk_blocks,
                    max_keys.div_ceil(candidate_block.max(1)),
                )
            }),
            topk: topk(
                cfg.index_topk,
                if consumer { candidate_tokens } else { max_keys },
            ),
            decode_next_n: cfg.decode_next_n,
            // `deep_gemm.fp8_fp4_paged_mqa_logits` over the page-planar FP4
            // index cache (packed UE8M0 scales), bf16 weights and logits.
            decode_logits: (!consumer).then(|| DsaPagedMqaLogitsDecodeKernelConfig {
                backends: cfg.decode_logits_backends.clone(),
                gpu_name: gpu.clone(),
                next_n: cfg.decode_next_n,
                max_model_len: max_keys.into(),
                num_heads: cfg.num_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                block_size: page,
                q_dtype: logits_dtype,
                cache_dtype: logits_dtype,
                scale_dtype: logits_dtype,
                weight_dtype: DType::Bf16,
                output_dtype: DType::Bf16,
                context_mode: "uniform".to_string(),
                page_mapping: "unique_scattered".to_string(),
                cache_format: "page_planar_fp4_ue8m0_scale".to_string(),
                clean_logits: false,
            }),
            decode_metadata: (cfg.paged_metadata && !consumer).then(|| {
                PagedMqaLogitsMetadataKernelConfig {
                    backends: cfg.logits_backends.clone(),
                    gpu_name: gpu.clone(),
                    next_n: cfg.decode_next_n,
                    block_size: page,
                }
            }),
            decode_sparse_metadata: cfg.candidate_metadata.then(|| {
                SparseMqaLogitsDecodeMetadataKernelConfig {
                    backends: cfg.logits_backends.clone(),
                    gpu_name: gpu.clone(),
                    next_n: cfg.decode_next_n,
                    block_size: page,
                    candidate_blocks: cfg.candidate_topk_blocks,
                    candidate_block_size: candidate_block,
                    dtype: logits_dtype,
                    max_context_len: max_keys,
                }
            }),
            decode_sparse_logits: consumer.then(|| SparseMqaLogitsDecodeKernelConfig {
                backends: cfg.logits_backends.clone(),
                gpu_name: gpu.clone(),
                next_n: cfg.decode_next_n,
                block_size: page,
                candidate_blocks: cfg.candidate_topk_blocks,
                candidate_block_size: candidate_block,
                num_heads: cfg.num_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                dtype: logits_dtype,
                max_context_len: max_keys,
            }),
        }
    }

    pub fn build(
        name: String,
        resolved: DeepseekV41DeepInferIndexerOpResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let slot = |suffix: &str| format!("{name}.{suffix}");
        macro_rules! maybe {
            ($kernel:ty, $cfg:expr, $suffix:literal) => {
                $cfg.clone()
                    .map(|config| <$kernel>::build(slot($suffix), config, bridge))
                    .transpose()?
                    .map(Arc::new)
            };
        }
        Ok(Self {
            dense_logits: maybe!(MqaLogitsPrefillChunkKernel, resolved.dense_logits, "logits"),
            sparse_metadata: maybe!(
                SparseMqaLogitsMetadataKernel,
                resolved.sparse_metadata,
                "sparse_metadata"
            ),
            sparse_logits: maybe!(
                SparseMqaLogitsPrefillKernel,
                resolved.sparse_logits,
                "sparse_logits"
            ),
            candidate_block_max: maybe!(
                CandidateBlockMaxKernel,
                resolved.candidate_block_max,
                "candidate_block_max"
            ),
            candidate_topk: maybe!(
                DeepSelectTopkKernel,
                resolved.candidate_topk,
                "candidate_topk"
            ),
            topk: Arc::new(DeepSelectTopkKernel::build(
                slot("topk"),
                resolved.topk.clone(),
                bridge,
            )?),
            decode_logits: maybe!(
                DsaPagedMqaLogitsDecodeKernel,
                resolved.decode_logits,
                "decode_logits"
            ),
            decode_metadata: maybe!(
                PagedMqaLogitsMetadataKernel,
                resolved.decode_metadata,
                "decode_metadata"
            ),
            decode_sparse_metadata: maybe!(
                SparseMqaLogitsDecodeMetadataKernel,
                resolved.decode_sparse_metadata,
                "decode_sparse_metadata"
            ),
            decode_sparse_logits: maybe!(
                SparseMqaLogitsDecodeKernel,
                resolved.decode_sparse_logits,
                "decode_sparse_logits"
            ),
            decode_candidate_block_max: maybe!(
                CandidateBlockMaxKernel,
                resolved.candidate_block_max,
                "decode_candidate_block_max"
            ),
            decode_candidate_topk: maybe!(
                DeepSelectTopkKernel,
                resolved.candidate_topk,
                "decode_candidate_topk"
            ),
            decode_topk: Arc::new(DeepSelectTopkKernel::build(
                slot("decode_topk"),
                resolved.topk.clone(),
                bridge,
            )?),
            resolved,
            name,
        })
    }

    pub fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        let mut leaves = Vec::new();
        let name = &self.name;
        if let Some(k) = &self.dense_logits {
            leaves.push(leaf(b, name, "logits", &**k));
        }
        if let Some(k) = &self.sparse_metadata {
            leaves.push(leaf(b, name, "sparse_metadata", &**k));
        }
        if let Some(k) = &self.sparse_logits {
            leaves.push(leaf(b, name, "sparse_logits", &**k));
        }
        if let Some(k) = &self.candidate_block_max {
            leaves.push(leaf(b, name, "candidate_block_max", &**k));
        }
        if let Some(k) = &self.candidate_topk {
            leaves.push(leaf(b, name, "candidate_topk", &**k));
        }
        leaves.push(leaf(b, name, "topk", &*self.topk));
        if let Some(k) = &self.decode_metadata {
            leaves.push(leaf(b, name, "decode_metadata", &**k));
        }
        if let Some(k) = &self.decode_logits {
            leaves.push(leaf(b, name, "decode_logits", &**k));
        }
        if let Some(k) = &self.decode_sparse_metadata {
            leaves.push(leaf(b, name, "decode_sparse_metadata", &**k));
        }
        if let Some(k) = &self.decode_sparse_logits {
            leaves.push(leaf(b, name, "decode_sparse_logits", &**k));
        }
        if let Some(k) = &self.decode_candidate_block_max {
            leaves.push(leaf(b, name, "decode_candidate_block_max", &**k));
        }
        if let Some(k) = &self.decode_candidate_topk {
            leaves.push(leaf(b, name, "decode_candidate_topk", &**k));
        }
        leaves.push(leaf(b, name, "decode_topk", &*self.decode_topk));
        CostNode::Sum(leaves)
    }

    pub fn eval(&self, input: &DeepseekV41DeepInferIndexerOpInput, ev: &mut Evaluator) {
        let r = &self.resolved;
        let chunks: Vec<ChunkWork> = chunk_plan(&input.prefill_query_context_pairs, r.chunk_rows)
            .iter()
            .map(|pieces| ChunkWork::new(pieces, r.compress_ratio, r.candidate_tokens))
            .collect();
        let logits = |c: &ChunkWork| (c.rows, c.context);
        if let Some(k) = &self.dense_logits {
            sum_push(
                ev,
                &chunks,
                |c| {
                    let (num_queries, context_len) = logits(c);
                    k.eval(&MqaLogitsPrefillChunkKernelInput {
                        num_queries,
                        context_len,
                    })
                },
                |c| {
                    let (num_queries, context_len) = logits(c);
                    MqaLogitsPrefillChunkKernelInput {
                        num_queries,
                        context_len,
                    }
                    .into()
                },
            );
        }
        if let Some(k) = &self.sparse_metadata {
            sum_push(
                ev,
                &chunks,
                |c| {
                    let (num_queries, context_len) = logits(c);
                    k.eval(&SparseMqaLogitsMetadataKernelInput {
                        num_queries,
                        context_len,
                    })
                },
                |c| {
                    let (num_queries, context_len) = logits(c);
                    SparseMqaLogitsMetadataKernelInput {
                        num_queries,
                        context_len,
                    }
                    .into()
                },
            );
        }
        if let Some(k) = &self.sparse_logits {
            sum_push(
                ev,
                &chunks,
                |c| {
                    let (num_queries, context_len) = logits(c);
                    k.eval(&SparseMqaLogitsPrefillKernelInput {
                        num_queries,
                        context_len,
                    })
                },
                |c| {
                    let (num_queries, context_len) = logits(c);
                    SparseMqaLogitsPrefillKernelInput {
                        num_queries,
                        context_len,
                    }
                    .into()
                },
            );
        }
        if let Some(k) = &self.candidate_block_max {
            sum_push(
                ev,
                &chunks,
                |c| {
                    let (num_queries, context_len) = logits(c);
                    k.eval(&CandidateBlockMaxKernelInput {
                        num_queries,
                        context_len,
                    })
                },
                |c| {
                    let (num_queries, context_len) = logits(c);
                    CandidateBlockMaxKernelInput {
                        num_queries,
                        context_len,
                    }
                    .into()
                },
            );
        }
        let block = r.candidate_block.max(1);
        let block_topk = |c: &ChunkWork| DeepSelectTopkKernelInput {
            num_rows: c.rows,
            // `candidate_block_max`'s valid blocks: the row's keys in blocks.
            row_len: c.row_len.div_ceil(block),
        };
        if let Some(k) = &self.candidate_topk {
            sum_push(
                ev,
                &chunks,
                |c| k.eval(&block_topk(c)),
                |c| block_topk(c).into(),
            );
        }
        let topk = |c: &ChunkWork| DeepSelectTopkKernelInput {
            num_rows: c.rows,
            row_len: c.row_len,
        };
        sum_push(
            ev,
            &chunks,
            |c| self.topk.eval(&topk(c)),
            |c| topk(c).into(),
        );
        self.eval_decode(input, ev);
    }

    /// The decode leaves, in [`Self::compile`] order; zero without decode rows.
    fn eval_decode(&self, input: &DeepseekV41DeepInferIndexerOpInput, ev: &mut Evaluator) {
        let r = &self.resolved;
        let work = decode_work(input, r);
        macro_rules! push {
            ($kernel:expr, $input:expr) => {
                if let Some(k) = $kernel {
                    match &work {
                        Some(w) => {
                            let i = $input(w);
                            ev.push(k.eval(&i), || i.clone().into());
                        }
                        None => ev.push(LeafMetrics::ZERO, || $input(&DecodeWork::EMPTY).into()),
                    }
                }
            };
        }
        push!(&self.decode_metadata, |w: &DecodeWork| {
            PagedMqaLogitsMetadataKernelInput {
                batch_size: w.requests,
            }
        });
        push!(&self.decode_logits, |w: &DecodeWork| {
            DsaPagedMqaLogitsDecodeKernelInput {
                batch_size: w.requests,
                context_len: w.keys,
            }
        });
        push!(&self.decode_sparse_metadata, |w: &DecodeWork| {
            SparseMqaLogitsDecodeMetadataKernelInput {
                batch_size: w.requests,
                context_len: w.keys,
            }
        });
        push!(&self.decode_sparse_logits, |w: &DecodeWork| {
            SparseMqaLogitsDecodeKernelInput {
                batch_size: w.requests,
                context_len: w.keys,
            }
        });
        push!(&self.decode_candidate_block_max, |w: &DecodeWork| {
            CandidateBlockMaxKernelInput {
                num_queries: w.rows,
                context_len: w.context,
            }
        });
        let block = r.candidate_block.max(1);
        push!(&self.decode_candidate_topk, |w: &DecodeWork| {
            DeepSelectTopkKernelInput {
                num_rows: w.rows,
                row_len: w.keys.div_ceil(block),
            }
        });
        push!(Some(&self.decode_topk), |w: &DecodeWork| {
            DeepSelectTopkKernelInput {
                num_rows: w.rows,
                row_len: w.row_len,
            }
        });
    }
}

/// One step's decode coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodeWork {
    /// Rows the launches cover.
    pub rows: u32,
    /// `ceil(rows / next_n)`: the requests the logits kernels take.
    pub requests: u32,
    /// Mean context (tokens, own row included) over the live rows.
    pub context: u32,
    /// Index keys of that context: `max(1, floor(context / ratio))`.
    pub keys: u32,
    /// The top-k's row length: the keys, capped at a consumer's candidates.
    pub row_len: u32,
}

impl DecodeWork {
    const EMPTY: Self = Self {
        rows: 0,
        requests: 0,
        context: 0,
        keys: 0,
        row_len: 0,
    };
}

pub(crate) fn decode_work(
    input: &DeepseekV41DeepInferIndexerOpInput,
    r: &DeepseekV41DeepInferIndexerOpResolved,
) -> Option<DecodeWork> {
    let live = input.decode_kv_lens.len() as u64;
    if live == 0 {
        return None;
    }
    let rows = input.decode_rows.max(live as u32);
    let context = input
        .decode_kv_lens
        .iter()
        .map(|&kv| u64::from(kv) + 1)
        .sum::<u64>()
        .div_ceil(live);
    let context = u32::try_from(context).expect("mean decode context fits u32");
    let keys = (context / r.compress_ratio).max(1);
    let row_len = if r.candidate_tokens == 0 {
        keys
    } else {
        keys.min(r.candidate_tokens)
    };
    Some(DecodeWork {
        rows,
        requests: rows.div_ceil(r.decode_next_n),
        context,
        keys,
        row_len,
    })
}

fn leaf<K: Probe>(b: &mut CostTreeBuilder, name: &str, suffix: &str, kernel: &K) -> CostNode {
    b.leaf(
        format!("{name}.{suffix}"),
        kernel.kind(),
        kernel.describe_config(),
    )
}

/// One leaf: every chunk's call summed, the first chunk's shape logged.
fn sum_push(
    ev: &mut Evaluator,
    chunks: &[ChunkWork],
    eval: impl Fn(&ChunkWork) -> LeafMetrics,
    log: impl Fn(&ChunkWork) -> SlotInput,
) {
    let mut metrics = LeafMetrics::ZERO;
    for chunk in chunks {
        metrics.add_fanin(eval(chunk));
    }
    match chunks.first() {
        Some(first) => ev.push(metrics, || log(first)),
        None => ev.push(metrics, || {
            log(&ChunkWork {
                rows: 0,
                context: 0,
                row_len: 0,
            })
        }),
    }
}

/// Consecutive rows of one request: `(rows, context of the last row)`; row
/// `j` of `n` has context `last - n + 1 + j`.
type Piece = (u32, u32);

/// The step's prefill rows, request after request, cut into chunks of
/// `chunk_rows` rows (`_topk_prefill`'s `range(0, total_q, chunk_size)`); a
/// request crossing a cut splits into two pieces.
pub(crate) fn chunk_plan(pairs: &[(u32, u32)], chunk_rows: u32) -> Vec<Vec<Piece>> {
    let mut chunks = Vec::new();
    let mut current: Vec<Piece> = Vec::new();
    let mut room = chunk_rows;
    for &(queries, context) in pairs {
        assert!(
            queries > 0 && queries <= context,
            "invalid prefill pair ({queries}, {context})"
        );
        let mut done = 0;
        while done < queries {
            let take = (queries - done).min(room);
            done += take;
            current.push((take, context - queries + done));
            room -= take;
            if room == 0 {
                chunks.push(std::mem::take(&mut current));
                room = chunk_rows;
            }
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// One chunk's kernel coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChunkWork {
    pub rows: u32,
    /// The single-request context with the chunk's mean key window.
    pub context: u32,
    /// Mean valid logits per row: the top-k's row length.
    pub row_len: u32,
}

impl ChunkWork {
    /// `cap`: a consumer's candidate tokens (its logits row width); 0 = none.
    pub(crate) fn new(pieces: &[Piece], ratio: u32, cap: u32) -> Self {
        let rows: u64 = pieces.iter().map(|&(n, _)| u64::from(n)).sum();
        let cap = if cap == 0 { u64::MAX } else { u64::from(cap) };
        let keys: u64 = pieces
            .iter()
            .map(|&(n, last)| {
                let first = u64::from(last) - u64::from(n) + 1;
                capped_floor_sum(u64::from(last), ratio.into(), cap)
                    - capped_floor_sum(first - 1, ratio.into(), cap)
            })
            .sum();
        Self {
            rows: rows as u32,
            context: equivalent_context(pieces),
            row_len: u32::try_from(keys / rows).expect("row length fits u32"),
        }
    }
}

/// The context of one request of `q = sum n` rows with the same mean context
/// per row as the pieces: `C - (q - 1) / 2 = mean(last - (n - 1) / 2)`, rounded
/// up. For `R`-row requests of context `C` this is `ceil((2C + q - R) / 2)`,
/// the lane's `equivalent_tail_context`. Never below `q` (one causal request).
pub(crate) fn equivalent_context(pieces: &[Piece]) -> u32 {
    let q: u64 = pieces.iter().map(|&(n, _)| u64::from(n)).sum();
    let weighted: u64 = pieces
        .iter()
        .map(|&(n, last)| u64::from(n) * (2 * u64::from(last) + 1 - u64::from(n)))
        .sum();
    let context = (q * (q - 1) + weighted).div_ceil(2 * q);
    u32::try_from(context.max(q)).expect("context fits u32")
}

/// `sum_{x=0}^{n} min(floor(x / r), cap)`.
fn capped_floor_sum(n: u64, r: u64, cap: u64) -> u64 {
    let floor_sum = |n: u64| {
        let m = n / r;
        r * m * m.saturating_sub(1) / 2 + m * (n - m * r + 1)
    };
    match cap.checked_mul(r) {
        Some(edge) if n >= edge => floor_sum(edge - 1) + cap * (n - edge + 1),
        _ => floor_sum(n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Catches a key count or window rounding that drifts from the lane's
    /// per-layer call list (`indexer.txt`, 4096-token requests at 131072).
    #[test]
    fn chunks_reproduce_the_lane_call_list() {
        let dense = |pairs: &[(u32, u32)], ratio, cap| {
            chunk_plan(pairs, chunk_rows(131_072 / ratio))
                .iter()
                .map(|p| ChunkWork::new(p, ratio, cap))
                .collect::<Vec<_>>()
        };
        // Layers 2/8/14 at 16384 tokens per rank: two 8192-row chunks.
        let request = (4096, 131_072);
        let r2 = dense(&[request; 4], 2, 0);
        assert_eq!(
            r2,
            [ChunkWork {
                rows: 8192,
                context: 133_120,
                row_len: 64_512
            }; 2]
        );
        // Layer 20 after YoCo at 65536 tokens: 16 x 140 rows, one chunk.
        let r1 = dense(&[(140, 131_072); 16], 1, 0);
        assert_eq!(
            r1,
            [ChunkWork {
                rows: 2240,
                context: 132_122,
                row_len: 131_002
            }]
        );
        assert_eq!(r1[0].row_len.div_ceil(8), 16_376);
        // Consumers: row length capped at the 2048 x 8 candidate tokens.
        let consumer = dense(&[(140, 131_072); 16], 1, 16_384);
        assert_eq!(consumer[0].row_len, 16_384);
    }

    #[test]
    fn chunk_rows_follow_the_logits_budget() {
        assert_eq!(chunk_rows(65_536), 8192);
        assert_eq!(chunk_rows(131_072), 8192);
        // 1M context at ratio 1: 10 GiB / 2 MiB rows.
        assert_eq!(chunk_rows(1_048_576), 5120);
    }

    /// Catches a request split at a chunk cut losing or double-counting rows,
    /// or giving its first piece the request's full context.
    #[test]
    fn a_request_crossing_a_cut_splits_by_rows() {
        let chunks = chunk_plan(&[(3000, 10_000), (6000, 6000)], 8192);
        assert_eq!(
            chunks,
            vec![vec![(3000, 10_000), (5192, 5192)], vec![(808, 6000)]]
        );
        let brute = |pieces: &[Piece], ratio: u32, cap: u64| -> u64 {
            pieces
                .iter()
                .flat_map(|&(n, last)| (last - n + 1)..=last)
                .map(|x| u64::from(x / ratio).min(cap))
                .sum()
        };
        for ratio in [1, 2] {
            for cap in [0_u32, 1000] {
                let work = ChunkWork::new(&chunks[0], ratio, cap);
                let limit = if cap == 0 { u64::MAX } else { u64::from(cap) };
                assert_eq!(
                    u64::from(work.row_len),
                    brute(&chunks[0], ratio, limit) / 8192
                );
            }
        }
    }
}
