//! GLM-5.3-Flash DeepSeek-sparse-attention (DSA) sublayer on one TP rank.
//!
//! Starts after the mHC pre boundary and ends before the TP all-reduce, which
//! the arch owns. Launch order follows the vLLM fork
//! (`glm5next/nvidia/attention.py`, `sparse_attn_indexer_kpool.py`,
//! `mla_attention.py`):
//!
//! * BF16 projections (`quant_config=None`): `fused_qkv_a`, `q_b`, the
//!   indexer's `wq_b`, `wk_weights`, `gate_score`, and `o_proj`; the indexer
//!   head weights are recomputed by an fp32 `torch.mm`.
//! * The pooled (kpool) indexer: FWHT-128 + FP8 quant of q, the kpool cache
//!   update, paged MQA logits and persistent top-k over `ceil(ctx / 4)` pools
//!   (decode rows), the prefill MQA logits and per-row top-k over each row's
//!   `(pos + 1) / 4` pools, and the pool-to-token expansion. The indexer
//!   replicates its 32 heads on every rank. vLLM runs the prefill logits and
//!   top-k once per chunk of its metadata builder's plan: requests packed while
//!   `rows x pools` fp32 logits fit `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`, and a
//!   longer request split along its queries (31 calls for 16,384 rows over
//!   245,760 pools at the 512 MiB default).
//! * MLA: `W_UK` absorb, the FP8 query quant, the kpool sparse-MLA compound op
//!   (cache append, remap, token-sparse attention for prefill and decode rows),
//!   and `W_UV`.
//!
//! Byte-sized elementwise placeholders stand in for the small indexer kernels
//! (b2-dsa section 7 byte formulas) and the framework glue (index plumbing, the query concat, the output copy). The
//! indexer k LayerNorm and the weights x q-scale multiply are Inductor-fused
//! elementwise kernels with their own leaves. Copies that run far below
//! streaming bandwidth (the transposed query concat, the output masked fill,
//! the Hadamard quant) carry an effective-bandwidth byte factor anchored to
//! the measured T = 2048 kernels, and a prefill-bearing iteration adds its
//! `[rows, selected_k]` index plumbing (`prefill_glue`).
//!
//! A decode batch's attention shapes collapse to the batch's MAX context (the
//! measured decode grids are uniform in context); the indexer's MQA logits
//! price at the batch's mean context and its top-k list-schedules the rows
//! over the SMs.

use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::common::gpu::{compute_capability, sm_count};

use crate::op::attention::{
    Glm53KpoolSparseMlaConfig, Glm53KpoolSparseMlaInput, Glm53KpoolSparseMlaOp,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::dsa_topk_prefill::takes_row_starts;
use crate::timing::kernels::sparse_indexer_topk::COOPERATIVE_MAX_ROWS;
use crate::timing::kernels::{
    BatchedGemmKernel, BatchedGemmKernelConfig, BatchedGemmKernelInput,
    BatchedGemmStaticFp8QuantKernel, BatchedGemmStaticFp8QuantKernelConfig,
    BatchedGemmStaticFp8QuantKernelInput, DsaMqaLogitsPrefillKernel,
    DsaMqaLogitsPrefillKernelConfig, DsaMqaLogitsPrefillKernelInput, DsaPagedMqaLogitsDecodeKernel,
    DsaPagedMqaLogitsDecodeKernelConfig, DsaPagedMqaLogitsDecodeKernelInput,
    DsaFp32SplitStateUpdateKernel, DsaFp32SplitStateUpdateKernelConfig,
    DsaFp32SplitStateUpdateKernelInput, DsaPooledIndexCacheUpdateKernel,
    DsaPooledIndexCacheUpdateKernelConfig, DsaPooledIndexCacheUpdateKernelInput,
    DsaPersistentTopkDecodeKernel, DsaPersistentTopkDecodeKernelConfig,
    DsaPersistentTopkDecodeKernelInput, DsaPooledLogitsPageTableKernel,
    DsaPooledLogitsPageTableKernelConfig, DsaPooledLogitsPageTableKernelInput,
    DsaPooledTopkPageTableKernel, DsaPooledTopkPageTableKernelConfig,
    DsaPooledTopkPageTableKernelInput, DsaTopkPrefillKernel, DsaTopkPrefillKernelConfig,
    DsaTopkPrefillKernelInput, ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    Fp32RowSplitLayerNormKernel, Fp32RowSplitLayerNormKernelConfig,
    Fp32RowSplitLayerNormKernelInput, GemmFp32OutputKernel, GemmFp32OutputKernelConfig,
    GemmFp32OutputKernelInput,
    HadamardFp8QuantKernel, HadamardFp8QuantKernelConfig, HadamardFp8QuantKernelInput,
    HadamardFp8QuantWeightScaleKernel, HadamardFp8QuantWeightScaleKernelConfig,
    HadamardFp8QuantWeightScaleKernelInput, QKvRmsNormKernel,
    QKvRmsNormKernelConfig, QKvRmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::slot_input::DsaIndexerPrefillLog;
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, SlotInput,
};

use super::glm53_common::{atomic, elementwise, push_or_zero, repeated};

/// Small launches of index/mask plumbing present in every iteration.
const GLUE_LAUNCHES: u32 = 17;
const GLUE_BYTES_PER_TOKEN: u32 = 64;
/// Extra index/mask plumbing of a prefill-bearing iteration: copies, compares
/// and masked fills over `[rows, selected_k]` int32 index rows (vLLM's prefill
/// topk/remap path). Capture 20260924_0 (T = 2048) has ~25 glue launches per
/// DSA layer, 73.7 us in total, against ~12 launches and 20 us in decode.
const PREFILL_GLUE_LAUNCHES: u32 = 8;
/// Effective-bandwidth factors for copies that run far below streaming
/// bandwidth. The placeholder still reads and writes the real tensor; the
/// factor scales those bytes to the measured time at T = 2048
/// (capture 20260924_0, iteration 886), because the elementwise curve is a
/// streaming kernel.
///
/// * `q_concat`: `torch.cat((ql_nope.transpose(0, 1), q_pe))` with an empty
///   `q_pe` (rope 0) is a `CatArrayBatchedCopy` of a transposed `[H, T, L]`
///   view: 64 MiB in 98.4 us, ~1/10 of streaming bandwidth.
const Q_CONCAT_EFFECTIVE_FACTOR: u32 = 10;
/// * `output_copy`: the non-vectorized `masked_fill_` over the `[T, H, L]`
///   attention output: 64 MiB in 32.2 us, ~1/3.5 of streaming bandwidth
///   (applied as 7/2).
const OUTPUT_COPY_EFFECTIVE_NUM: u32 = 7;
const OUTPUT_COPY_EFFECTIVE_DEN: u32 = 2;
/// `k[idx]` and `gate_score[idx]` gathers ahead of the prefill kpool write.
const PREFILL_GATHER_LAUNCHES: u32 = 2;

/// How one group of DSA-front projections that read the same operand
/// launches. Not the fork's path except `Separate`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DsaFrontGemm {
    /// The fork: one launch per projection.
    #[default]
    Separate,
    /// The group's bf16 projections as one GEMM over concatenated weights:
    /// the same callable and backends as the separate ones, a wider N. The
    /// surrounding launches stay.
    Concat,
    /// The whole cut as one kernel with its epilogues, priced only by rows of
    /// this backend: a missing row fails the build, and an override to any
    /// other backend is rejected.
    Fused(&'static str),
    /// Cut A only: the whole cut as an fp32-output GEMM over every column
    /// (`gemm_fp32_output`, the separate `head_weights` callable at the full
    /// width) plus one row-split epilogue kernel on this backend that rounds
    /// the bf16 column groups and LayerNorms the indexer k
    /// (`fp32_row_split_layer_norm`). Absorbs what `Fused` absorbs.
    Fp32Split(&'static str),
    /// Cut A only: `Fp32Split`'s GEMM, whose decode rows' epilogue is one
    /// state-update kernel on this backend (`dsa_fp32_split_state_update`):
    /// besides the row split it RMS-norms q and kv, appends the kv rows to the
    /// FP8 MLA cache, casts the kpool positions and runs the kpool decode
    /// update ([`SPLIT_STATE_WRITER_ABSORBS`]). Prefill rows keep `Fp32Split`'s
    /// row split (also on this backend) and the fork's norm and cache append.
    /// Exclusive with cut D.
    Fp32SplitStateWriter(&'static str),
    /// Cut B only: `Concat`'s GEMM, plus one epilogue kernel on this backend
    /// that reads the indexer query heads in place from the concatenated
    /// output, Hadamard-rotates and FP8-quantizes them and scales the head
    /// weights (`hadamard_fp8_quant_weight_scale`). Absorbs the indexer's
    /// `q_fwht_quant` and `weight_scale`; the q/kv RMS norm stays.
    ConcatHadamardEpilogue(&'static str),
}

/// Which launch list prices the per-iteration elementwise glue.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DsaGlue {
    /// `GLUE_LAUNCHES` launches of `GLUE_BYTES_PER_TOKEN` per token (kept so
    /// earlier runs reproduce).
    #[default]
    Placeholder,
    /// The fork's decode launch list at its real bytes
    /// (logs/20261114_0_mainline_v3/dsa_audit.md, glue G1-G7): the hidden
    /// state's fp32 cast for `head_weights`, the top-k buffer fill, two
    /// position casts, the pool-id cast to int64, the `+ 1` on the top-k
    /// sequence lengths, and the copy of the expanded indices into the top-k
    /// buffer.
    ForkLaunchList,
}

/// Fusions of the DSA front (the launches before sparse attention), by input
/// operand (`from_moe16.md` cuts A-D). The default is the fork's launches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Glm53DsaFront {
    /// Cut A, the GEMMs reading the hidden state. `Concat`: `fused_qkv_a`,
    /// `wk_weights` without its head-weight columns, and `kpool_gate_score`
    /// as one bf16 GEMM; `head_weights` and its cast stay. `Fused`: also
    /// the fp32 `head_weights` (its 32 columns once, with an fp32 epilogue)
    /// with its cast and `k_norm` in the epilogue.
    pub hidden_gemm: DsaFrontGemm,
    /// Cut B, the GEMMs reading the normed q latent. `Concat`: `q_b` and the
    /// indexer's `wq_b` as one bf16 GEMM. `Fused`: also the q half of the
    /// q/kv RMS norm, the indexer's FWHT + quant and its weight scale.
    /// `ConcatHadamardEpilogue`: `Concat` plus one epilogue launch for the
    /// FWHT + quant and weight scale (the norm stays).
    pub q_lora_gemm: DsaFrontGemm,
    /// Cut C: `q_absorb` with the fp8 quant in its epilogue, on this backend
    /// only (`batched_gemm_static_fp8_quant`).
    pub q_absorb_fp8: Option<&'static str>,
    /// Cut D: one state writer (kv RMS norm + MLA cache append + kpool decode
    /// update), on this backend only.
    pub state_writer: Option<&'static str>,
    /// Cut E (indexer back, post-top-k): the decode rows' selected pools to
    /// the final physical page table and valid counts in one kernel
    /// (`dsa_pooled_topk_page_table`), on this backend only. It absorbs the
    /// decode rows' pool expand and index remap and the fork glue's top-k
    /// buffer fill, pool-id cast, seq-len `+ 1` and buffer copy. Prefill rows
    /// keep the fork's expand and remap.
    pub topk_page_table: Option<&'static str>,
    /// Cut E with the top-k (indexer back from the logits): the decode rows'
    /// top-k and the cut E page table in one kernel
    /// (`dsa_pooled_logits_page_table`), on this backend only. It absorbs the
    /// decode top-k (persistent and cooperative) and everything cut E does.
    /// Without [`Self::topk_to_page_table_max_pools`] the two cuts are
    /// exclusive.
    pub topk_to_page_table: Option<&'static str>,
    /// Dispatch by context: cut E + top-k runs only for decode launches whose
    /// longest row has at most this many pools; longer launches run the
    /// decode top-k and cut E (both backends set). Its threshold comes from a
    /// 2048-value sample whose target rank shrinks with the row length, so on
    /// long rows a row overshoots and takes a one-CTA exact fallback over the
    /// whole row, and the launch waits for it.
    pub topk_to_page_table_max_pools: Option<u32>,
    pub glue: DsaGlue,
}

/// The decode-row launches cut E replaces (worklet slot suffixes): the pool
/// expand and index remap of the decode rows, and the fork glue's top-k
/// buffer fill, pool-id cast, seq-len `+ 1` and buffer copy.
pub const PAGE_TABLE_ABSORBS: [&str; 6] = [
    "indexer.expand_pools",
    "sparse_mla.index_remap",
    "glue.topk_buffer_fill",
    "glue.pool_ids_to_int64",
    "glue.seq_lens_plus_one",
    "glue.topk_buffer_copy",
];
/// What cut E + top-k replaces: cut E's set and the decode top-k slot (the
/// persistent kernel, DeepSelect, or the cooperative kernel at <= 64 rows).
pub const TOPK_TO_PAGE_TABLE_ABSORBS: [&str; 7] = [
    "indexer.topk_decode",
    "indexer.expand_pools",
    "sparse_mla.index_remap",
    "glue.topk_buffer_fill",
    "glue.pool_ids_to_int64",
    "glue.seq_lens_plus_one",
    "glue.topk_buffer_copy",
];

/// The decode-row launches `Fp32SplitStateWriter` replaces beyond
/// `Fp32Split` (worklet slot suffixes): both halves of the q/kv RMS norm, the
/// MLA cache append, one of the fork glue's two position casts (G3, the kpool
/// `dec_pos`; G5, the top-k seq lens, stays) and the kpool decode update.
pub const SPLIT_STATE_WRITER_ABSORBS: [&str; 4] = [
    "q_kv_a_norm",
    "sparse_mla.mla_cache_append",
    "glue.positions_to_int32",
    "indexer.kpool_decode_update",
];

/// Which page-table cut a decode launch runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodePageTableCut {
    /// The fork's top-k and post-top-k launches.
    None,
    /// Cut E after the decode top-k.
    PageTable,
    /// Cut E + top-k.
    TopkToPageTable,
}

impl Glm53DsaFront {
    /// Cut A writes the decode rows' state in its epilogue.
    fn split_state_writer(self) -> bool {
        matches!(self.hidden_gemm, DsaFrontGemm::Fp32SplitStateWriter(_))
    }

    /// The cut a decode launch whose longest row has `longest_pools` pools
    /// runs.
    pub fn decode_page_table_cut(self, longest_pools: u32) -> DecodePageTableCut {
        let fused_fits = self
            .topk_to_page_table_max_pools
            .is_none_or(|max_pools| longest_pools <= max_pools);
        match (self.topk_page_table, self.topk_to_page_table) {
            (_, Some(_)) if fused_fits => DecodePageTableCut::TopkToPageTable,
            (Some(_), _) => DecodePageTableCut::PageTable,
            _ => DecodePageTableCut::None,
        }
    }

    /// The decode-row slots a launch of `longest_pools` pools bills zero (the
    /// cut's own slot prices them instead).
    pub fn decode_slots_absorbed(self, longest_pools: u32) -> &'static [&'static str] {
        match self.decode_page_table_cut(longest_pools) {
            DecodePageTableCut::TopkToPageTable => &TOPK_TO_PAGE_TABLE_ABSORBS,
            DecodePageTableCut::PageTable => &PAGE_TABLE_ABSORBS,
            DecodePageTableCut::None => &[],
        }
    }
}

impl DsaFrontGemm {
    fn merged(self) -> bool {
        self != Self::Separate
    }

    fn fused(self) -> bool {
        matches!(
            self,
            Self::Fused(_) | Self::Fp32Split(_) | Self::Fp32SplitStateWriter(_)
        )
    }

    /// The indexer query's FWHT quant and weight scale run inside the cut.
    fn absorbs_index_q_quant(self) -> bool {
        matches!(self, Self::Fused(_) | Self::ConcatHadamardEpilogue(_))
    }
}

#[derive(Clone, Debug)]
pub struct Glm53DsaAttnLocalWorkletConfig {
    pub hidden: Dim,
    /// MLA query heads on this rank.
    pub num_heads: Dim,
    pub q_lora_rank: Dim,
    pub kv_lora_rank: Dim,
    pub qk_nope_head_dim: Dim,
    pub v_head_dim: Dim,
    pub index_num_heads: Dim,
    pub index_head_dim: Dim,
    pub index_topk: u32,
    pub index_kpool: u32,
    /// Sparse page-table width, `round_up(index_topk + index_kpool - 1, 128)`.
    pub selected_k: u32,
    pub max_model_len: u32,
    pub cache_block_size: u32,
    pub rms_eps: f64,
    pub gpu_name: String,
    pub bf16_gemm_backends: Vec<&'static str>,
    pub fp32_gemm_backends: Vec<&'static str>,
    pub qkv_norm_backends: Vec<&'static str>,
    pub mla_bmm_q_absorb_backends: Vec<&'static str>,
    pub mla_bmm_v_up_backends: Vec<&'static str>,
    pub mqa_logits_backends: Vec<&'static str>,
    pub topk_backends: Vec<&'static str>,
    /// vLLM's `cooperative_topk`, which the fork's "auto" decode top-k runs
    /// for at most 64 rows (`model_executor/layers/indexer_topk.py:215-229`).
    /// Empty where the platform has no cooperative top-k.
    pub cooperative_topk_backends: Vec<&'static str>,
    pub mqa_logits_prefill_backends: Vec<&'static str>,
    pub topk_prefill_backends: Vec<&'static str>,
    pub sparse_attention_backends: Vec<&'static str>,
    pub mla_cache_append_backends: Vec<&'static str>,
    pub index_remap_backends: Vec<&'static str>,
    /// The indexer query's Hadamard-128 + per-row UE8M0 FP8 quant.
    pub hadamard_quant_backends: Vec<&'static str>,
    /// The pooled indexer key cache's decode/verify update.
    pub pooled_index_update_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    /// The fork's head-major MLA layout copies (`q_concat`, `output_copy`).
    /// Current vLLM writes `ql_nope` token-major through `bmm(out=)` and
    /// skips the empty-query `masked_fill_` unless DCP or HiSparse is on, so
    /// `false` zeroes both.
    pub mla_layout_copies: bool,
    /// vLLM's prefill-indexer logits budget (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`
    /// in bytes): the fp32 `[rows, pools]` matrix of one MQA-logits call stays
    /// within it, and each call is followed by its own top-k.
    pub indexer_max_logits_bytes: u64,
    /// Query rows per decode request: 1 for plain decode, `k + 1` for a
    /// speculative verify, whose rows sit at the request's last `k + 1`
    /// positions. The decode indexer scores a request's rows in one group
    /// (DeepGEMM `fp8_paged_mqa_logits` at `next_n`, one read of the request's
    /// index keys) and its top-k selects per row; every other leaf, sparse MLA
    /// included, bills the flattened rows.
    pub decode_next_n: u32,
    /// Not the fork's path unless default: fused DSA-front launches.
    pub front: Glm53DsaFront,
}

/// The fork's decode glue launches ([`DsaGlue::ForkLaunchList`]).
#[derive(Clone, Debug)]
pub struct DsaForkGlueConfigs {
    /// G1, `hidden_states.float()` ahead of `head_weights`, every row.
    pub hidden_fp32_cast: ElementwiseKernelConfig,
    /// G2, `topk_indices_buffer[:rows] = -1`.
    pub topk_buffer_fill: ElementwiseKernelConfig,
    /// G3 + G5, `positions[:n].to(int32)` (kpool `dec_pos`, top-k seq lens).
    pub positions_to_int32: ElementwiseKernelConfig,
    /// G4, `pool_topk.to(int64)`.
    pub pool_ids_to_int64: ElementwiseKernelConfig,
    /// G6, the top-k seq lens `+ 1`.
    pub seq_lens_plus_one: ElementwiseKernelConfig,
    /// G7, `topk_indices_buffer[:rows, :window] = out`.
    pub topk_buffer_copy: ElementwiseKernelConfig,
}

#[derive(Clone, Debug)]
pub struct Glm53DsaAttnLocalWorkletResolved {
    pub raw_cfg: Glm53DsaAttnLocalWorkletConfig,
    pub fused_qkv_a: SingleGemmKernelConfig,
    pub q_kv_norm: QKvRmsNormKernelConfig,
    pub q_b: SingleGemmKernelConfig,
    pub index_wq_b: SingleGemmKernelConfig,
    pub index_wk_weights: SingleGemmKernelConfig,
    pub index_head_weights: GemmFp32OutputKernelConfig,
    pub index_k_norm: ElementwiseKernelConfig,
    pub index_q_fwht_quant: HadamardFp8QuantKernelConfig,
    pub index_weight_scale: ElementwiseKernelConfig,
    pub kpool_gate_score: SingleGemmKernelConfig,
    pub prefill_gather: ElementwiseKernelConfig,
    pub kpool_decode_update: DsaPooledIndexCacheUpdateKernelConfig,
    pub kpool_prefill_write: ElementwiseKernelConfig,
    pub kpool_tail_seed: ElementwiseKernelConfig,
    pub mqa_logits_decode: DsaPagedMqaLogitsDecodeKernelConfig,
    pub topk_decode: DsaPersistentTopkDecodeKernelConfig,
    /// `topk_decode` on the cooperative backends (None when there are none).
    pub topk_decode_cooperative: Option<DsaPersistentTopkDecodeKernelConfig>,
    pub mqa_logits_prefill: DsaMqaLogitsPrefillKernelConfig,
    pub topk_prefill: DsaTopkPrefillKernelConfig,
    pub expand_pools: ElementwiseKernelConfig,
    pub q_absorb: BatchedGemmKernelConfig,
    pub q_concat: ElementwiseKernelConfig,
    pub q_fp8_quant: ElementwiseKernelConfig,
    pub sparse_mla: Glm53KpoolSparseMlaConfig,
    pub output_copy: ElementwiseKernelConfig,
    pub v_up: BatchedGemmKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
    pub glue: ElementwiseKernelConfig,
    pub prefill_glue: ElementwiseKernelConfig,
    /// Cut A as one GEMM (`front.hidden_gemm` not `Separate`).
    pub hidden_gemm: Option<SingleGemmKernelConfig>,
    /// Cut A as `Fp32Split`: the fp32-output GEMM and its row-split epilogue.
    pub hidden_gemm_fp32: Option<GemmFp32OutputKernelConfig>,
    pub hidden_row_split: Option<Fp32RowSplitLayerNormKernelConfig>,
    /// Cut A as `Fp32SplitStateWriter`: the decode rows' state-update epilogue
    /// (`hidden_row_split` then prices the prefill rows).
    pub hidden_state_writer: Option<DsaFp32SplitStateUpdateKernelConfig>,
    /// Cut B as one GEMM (`front.q_lora_gemm` not `Separate`).
    pub q_lora_gemm: Option<SingleGemmKernelConfig>,
    /// Cut B as `ConcatHadamardEpilogue`: the indexer query epilogue.
    pub q_lora_epilogue: Option<HadamardFp8QuantWeightScaleKernelConfig>,
    /// Cut C (`front.q_absorb_fp8`).
    pub q_absorb_fp8: Option<BatchedGemmStaticFp8QuantKernelConfig>,
    /// Cut D (`front.state_writer`).
    pub state_writer: Option<ElementwiseKernelConfig>,
    /// `front.glue == ForkLaunchList`.
    pub fork_glue: Option<DsaForkGlueConfigs>,
    /// Cut E (`front.topk_page_table`).
    pub topk_page_table: Option<DsaPooledTopkPageTableKernelConfig>,
    /// Cut E with the top-k (`front.topk_to_page_table`).
    pub topk_to_page_table: Option<DsaPooledLogitsPageTableKernelConfig>,
}

/// One iteration's DSA work on this rank.
#[derive(Clone, Debug, Default)]
pub struct Glm53DsaAttnLocalWorkletInput {
    /// `(prefix, append)` per prefill request.
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
    /// KV length per decode request, including every query row it presents
    /// (`decode_next_n` of them, ending at this length).
    pub decode_kv_lens: Vec<u32>,
}

pub struct Glm53DsaAttnLocalWorklet {
    pub name: String,
    pub fused_qkv_a: Op<SingleGemmKernel>,
    pub q_kv_norm: Op<QKvRmsNormKernel>,
    pub q_b: Op<SingleGemmKernel>,
    pub index_wq_b: Op<SingleGemmKernel>,
    pub index_wk_weights: Op<SingleGemmKernel>,
    pub index_head_weights: Op<GemmFp32OutputKernel>,
    pub index_k_norm: Op<ElementwiseKernel>,
    pub index_q_fwht_quant: Op<HadamardFp8QuantKernel>,
    pub index_weight_scale: Op<ElementwiseKernel>,
    pub kpool_gate_score: Op<SingleGemmKernel>,
    pub prefill_gather: Op<ElementwiseKernel>,
    pub kpool_decode_update: Op<DsaPooledIndexCacheUpdateKernel>,
    pub kpool_prefill_write: Op<ElementwiseKernel>,
    pub kpool_tail_seed: Op<ElementwiseKernel>,
    pub mqa_logits_decode: Op<DsaPagedMqaLogitsDecodeKernel>,
    pub topk_decode: Op<DsaPersistentTopkDecodeKernel>,
    pub mqa_logits_prefill: Op<DsaMqaLogitsPrefillKernel>,
    pub topk_prefill: Op<DsaTopkPrefillKernel>,
    /// Top-k for launches that pack several requests: `topk_prefill`'s
    /// backends that take row starts (`dsa_topk_prefill::takes_row_starts`),
    /// when some of them do not. Its time lands in the `topk_prefill` slot.
    topk_prefill_packed: Option<Arc<DsaTopkPrefillKernel>>,
    /// `cooperative_topk`, which the fork's "auto" dispatch runs for at most
    /// [`COOPERATIVE_MAX_ROWS`] rows; its time lands in the `topk_decode`
    /// slot. None without cooperative backends, on a GPU the fork's dispatch
    /// skips (below SM90, or the SM12x family), or when `topk_decode` is
    /// overridden to a backend outside [`LONG_ROW_BACKENDS`].
    topk_decode_cooperative: Option<Arc<DsaPersistentTopkDecodeKernel>>,
    /// Rows `topk_decode` runs at once (the GPU's SM count, one CTA per row),
    /// or `None` when the GPU catalog does not record it.
    topk_decode_slots: Option<u32>,
    pub expand_pools: Op<ElementwiseKernel>,
    pub q_absorb: Op<BatchedGemmKernel>,
    pub q_concat: Op<ElementwiseKernel>,
    pub q_fp8_quant: Op<ElementwiseKernel>,
    pub sparse_mla: Glm53KpoolSparseMlaOp,
    pub output_copy: Op<ElementwiseKernel>,
    pub v_up: Op<BatchedGemmKernel>,
    pub o_proj: Op<SingleGemmKernel>,
    pub glue: Op<ElementwiseKernel>,
    pub prefill_glue: Op<ElementwiseKernel>,
    pub hidden_gemm: Option<Op<SingleGemmKernel>>,
    pub hidden_gemm_fp32: Option<Op<GemmFp32OutputKernel>>,
    pub hidden_row_split: Option<Op<Fp32RowSplitLayerNormKernel>>,
    pub hidden_state_writer: Option<Op<DsaFp32SplitStateUpdateKernel>>,
    pub q_lora_gemm: Option<Op<SingleGemmKernel>>,
    pub q_lora_epilogue: Option<Op<HadamardFp8QuantWeightScaleKernel>>,
    pub q_absorb_fp8: Option<Op<BatchedGemmStaticFp8QuantKernel>>,
    pub state_writer: Option<Op<ElementwiseKernel>>,
    pub fork_glue: Option<DsaForkGlue>,
    pub topk_page_table: Option<Op<DsaPooledTopkPageTableKernel>>,
    pub topk_to_page_table: Option<Op<DsaPooledLogitsPageTableKernel>>,
    resolved: Glm53DsaAttnLocalWorkletResolved,
}

/// The ops of [`DsaForkGlueConfigs`].
pub struct DsaForkGlue {
    pub hidden_fp32_cast: Op<ElementwiseKernel>,
    pub topk_buffer_fill: Op<ElementwiseKernel>,
    pub positions_to_int32: Op<ElementwiseKernel>,
    pub pool_ids_to_int64: Op<ElementwiseKernel>,
    pub seq_lens_plus_one: Op<ElementwiseKernel>,
    pub topk_buffer_copy: Op<ElementwiseKernel>,
}

/// G3 and G5 are the same int64 -> int32 position cast.
const POSITION_CASTS: u32 = 2;

impl Glm53DsaAttnLocalWorklet {
    pub fn resolve_config(
        cfg: &Glm53DsaAttnLocalWorkletConfig,
    ) -> Glm53DsaAttnLocalWorkletResolved {
        let bf16 = DType::Bf16;
        let heads = cfg.num_heads.get();
        let latent = cfg.kv_lora_rank.get();
        let index_heads = cfg.index_num_heads.get();
        let index_dim = cfg.index_head_dim.get();
        let kpool = cfg.index_kpool;
        let gemm = |n: u32, k: Dim| SingleGemmKernelConfig {
            backends: cfg.bf16_gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: n.into(),
            k,
            dtype: bf16,
        };
        let ew = |input: u32, output: u32| {
            elementwise(&cfg.elementwise_backends, &cfg.gpu_name, input, output)
        };
        let bmm = |backends: &[&'static str], n: Dim, k: Dim| BatchedGemmKernelConfig {
            backends: backends.to_vec(),
            gpu_name: cfg.gpu_name.clone(),
            num_batches: cfg.num_heads.clone(),
            n,
            k,
            dtype: bf16,
        };
        let max_model_len = Dim::param("max_model_len", cfg.max_model_len);
        // The per-head latent query, bf16, and its FP8 copy with one fp32
        // scale per head.
        let latent_q_bytes = heads * latent * bf16.size_bytes();
        let window = cfg.index_topk + kpool - 1;
        let mut resolved = Glm53DsaAttnLocalWorkletResolved {
            fused_qkv_a: gemm(cfg.q_lora_rank.get() + latent, cfg.hidden.clone()),
            q_kv_norm: QKvRmsNormKernelConfig {
                backends: cfg.qkv_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                q_dim: cfg.q_lora_rank.clone(),
                kv_dim: cfg.kv_lora_rank.clone(),
                rms_eps_bits: cfg.rms_eps.to_bits(),
                dtype: bf16,
            },
            q_b: gemm(heads * cfg.qk_nope_head_dim.get(), cfg.q_lora_rank.clone()),
            index_wq_b: gemm(index_heads * index_dim, cfg.q_lora_rank.clone()),
            // Indexer k plus one weight per indexer head.
            index_wk_weights: gemm(index_dim + index_heads, cfg.hidden.clone()),
            index_head_weights: GemmFp32OutputKernelConfig {
                backends: cfg.fp32_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.index_num_heads.clone(),
                k: cfg.hidden.clone(),
                input_dtype: DType::Fp32,
            },
            // Inductor-fused fp32 LayerNorm of the indexer k: reads and writes
            // one bf16 D row per token.
            index_k_norm: ew(2 * index_dim, 2 * index_dim),
            index_q_fwht_quant: HadamardFp8QuantKernelConfig {
                backends: cfg.hadamard_quant_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                head_dim: index_dim,
                input_dtype: bf16,
            },
            // `weights * q_scale * scale`: two fp32 Hi rows in, one out.
            index_weight_scale: ew(2 * 4 * index_heads, 4 * index_heads),
            kpool_gate_score: gemm(index_dim, cfg.hidden.clone()),
            // 2*P*2D bytes gathered per prefill token, split over two launches.
            prefill_gather: ew(kpool * 2 * index_dim, kpool * 2 * index_dim),
            // One program per request walks its next_n rows in order; the
            // tail ring holds one pool (`index_kpool` slots), as the fork's
            // KpoolTailSpec(block_size = index_kpool).
            kpool_decode_update: DsaPooledIndexCacheUpdateKernelConfig {
                backends: cfg.pooled_index_update_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                next_n: cfg.decode_next_n,
                pool_size: kpool,
                tail_ring: kpool,
                index_dim: cfg.index_head_dim.clone(),
                block_size: cfg.cache_block_size,
                input_dtype: bf16,
                cache_dtype: DType::Fp8E4m3,
            },
            // Per prefill token: 9 + P*2*2D/P read, (D+4)/P written.
            kpool_prefill_write: ew(9 + 4 * index_dim, (index_dim + 4) / kpool),
            // Reads each prefill token's position; the < P tail rows are noise.
            kpool_tail_seed: ew(8, 4),
            mqa_logits_decode: DsaPagedMqaLogitsDecodeKernelConfig {
                backends: cfg.mqa_logits_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                next_n: cfg.decode_next_n,
                max_model_len: max_model_len.clone(),
                num_heads: cfg.index_num_heads.clone(),
                head_dim: cfg.index_head_dim.clone(),
                block_size: cfg.cache_block_size,
                q_dtype: DType::Fp8E4m3,
                cache_dtype: DType::Fp8E4m3,
                scale_dtype: DType::Fp32,
                weight_dtype: DType::Fp32,
                output_dtype: DType::Fp32,
                context_mode: "uniform".into(),
                page_mapping: "unique_scattered".into(),
                cache_format: "page_planar_fp8_fp32_scale".into(),
                clean_logits: false,
            },
            topk_decode: DsaPersistentTopkDecodeKernelConfig {
                backends: cfg.topk_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                next_n: cfg.decode_next_n,
                max_model_len: max_model_len.clone(),
                top_k: cfg.index_topk / kpool,
                logits_row_stride: max_model_len.clone(),
                logits_dtype: DType::Fp32,
                index_dtype: "int32".into(),
                // Scores drawn from the measured logit distribution: real
                // GLM-5.3-Flash rows replay within ~5% of it, against 1.1-1.2x
                // for uniform random and 1.5x for linspace scores.
                context_mode: "uniform_measured".into(),
            },
            // One launch per iteration over every prefill row; each row spans
            // its own pool-granular context (`compress_ratio == index_kpool`).
            mqa_logits_prefill: DsaMqaLogitsPrefillKernelConfig {
                backends: cfg.mqa_logits_prefill_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_sequences: 1,
                num_heads: cfg.index_num_heads.clone(),
                head_dim: cfg.index_head_dim.clone(),
                q_dtype: DType::Fp8E4m3,
                k_dtype: DType::Fp8E4m3,
                k_scale_dtype: DType::Fp32,
                weight_dtype: DType::Fp32,
                output_dtype: DType::Fp32,
                span_mode: "single_causal_tail".into(),
                clean_logits: false,
            },
            topk_prefill: DsaTopkPrefillKernelConfig {
                backends: cfg.topk_prefill_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_sequences: 1,
                top_k: cfg.index_topk / kpool,
                logits_dtype: DType::Fp32,
                index_dtype: "int32".into(),
                span_mode: "single_causal_tail".into(),
            },
            // Per row: int64 pool ids in, `index_topk + P - 1` int32 slots out.
            expand_pools: ew(8 * cfg.index_topk / kpool + 4, 4 * window),
            q_absorb: bmm(
                &cfg.mla_bmm_q_absorb_backends,
                cfg.kv_lora_rank.clone(),
                cfg.qk_nope_head_dim.clone(),
            ),
            q_concat: ew(
                Q_CONCAT_EFFECTIVE_FACTOR * latent_q_bytes,
                Q_CONCAT_EFFECTIVE_FACTOR * latent_q_bytes,
            ),
            q_fp8_quant: ew(latent_q_bytes, heads * (latent + 4)),
            sparse_mla: Glm53KpoolSparseMlaConfig {
                sparse_attention_backends: cfg.sparse_attention_backends.clone(),
                mla_cache_append_backends: cfg.mla_cache_append_backends.clone(),
                index_remap_backends: cfg.index_remap_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.num_heads.clone(),
                latent_dim: cfg.kv_lora_rank.clone(),
                rope_dim: Dim::param("qk_rope_head_dim", 0),
                selected_k: cfg.selected_k,
                index_topk: cfg.index_topk,
                index_kpool: kpool,
                softmax_scale_denominator: integer_sqrt(cfg.qk_nope_head_dim.get()),
                activation_dtype: bf16,
                q_dtype: DType::Fp8E4m3,
                cache_dtype: DType::Fp8E4m3,
                output_dtype: bf16,
                index_dtype: "int32".into(),
                index_distribution: "unique_scattered_pages".into(),
                cache_layout: "hnd_paged_mqa_fp8_latent".into(),
                mla_cache_block_size: cfg.cache_block_size,
                mla_cache_format: "plain".into(),
                page_table_mapping: "interleaved_requests".into(),
                max_model_len: cfg.max_model_len,
            },
            output_copy: ew(
                latent_q_bytes * OUTPUT_COPY_EFFECTIVE_NUM / OUTPUT_COPY_EFFECTIVE_DEN,
                latent_q_bytes * OUTPUT_COPY_EFFECTIVE_NUM / OUTPUT_COPY_EFFECTIVE_DEN,
            ),
            v_up: bmm(
                &cfg.mla_bmm_v_up_backends,
                cfg.v_head_dim.clone(),
                cfg.kv_lora_rank.clone(),
            ),
            o_proj: gemm(cfg.hidden.get(), (heads * cfg.v_head_dim.get()).into()),
            glue: ew(GLUE_BYTES_PER_TOKEN, GLUE_BYTES_PER_TOKEN),
            // One int32 index row of `selected_k` lanes in and out per row.
            prefill_glue: ew(4 * cfg.selected_k, 4 * cfg.selected_k),
            topk_decode_cooperative: None,
            hidden_gemm: None,
            hidden_gemm_fp32: None,
            hidden_row_split: None,
            hidden_state_writer: None,
            q_lora_gemm: None,
            q_lora_epilogue: None,
            q_absorb_fp8: None,
            state_writer: None,
            fork_glue: None,
            topk_page_table: None,
            topk_to_page_table: None,
            raw_cfg: cfg.clone(),
        };
        let front = cfg.front;
        let front_gemm = |mode: DsaFrontGemm, concat_n: u32, fused_n: u32, k: Dim| match mode {
            DsaFrontGemm::Separate
            | DsaFrontGemm::Fp32Split(_)
            | DsaFrontGemm::Fp32SplitStateWriter(_) => None,
            DsaFrontGemm::Concat | DsaFrontGemm::ConcatHadamardEpilogue(_) => {
                Some(gemm(concat_n, k))
            }
            DsaFrontGemm::Fused(backend) => Some(SingleGemmKernelConfig {
                backends: vec![backend],
                ..gemm(fused_n, k)
            }),
        };
        // Cut A. Concat is [qkv_a | k | gate score] in bf16 (2,304 columns for
        // GLM-5.3-Flash): the 32 bf16 head-weight columns of `wk_weights` are
        // dropped and `head_weights` stays its own fp32-output GEMM, because a
        // bf16 output for them changes the selected pools in 62% of rows
        // (from_moe16.md Addendum 2). The fused cut computes those columns
        // once with an fp32 epilogue (2,336 columns).
        let hidden_concat_n = cfg.q_lora_rank.get() + latent + index_dim + index_dim;
        resolved.hidden_gemm = front_gemm(
            front.hidden_gemm,
            hidden_concat_n,
            hidden_concat_n + index_heads,
            cfg.hidden.clone(),
        );
        if let DsaFrontGemm::Fp32Split(backend) | DsaFrontGemm::Fp32SplitStateWriter(backend) =
            front.hidden_gemm
        {
            // [q_a | kv_a | k | gate score | head weights], 2,336 columns.
            resolved.hidden_gemm_fp32 = Some(GemmFp32OutputKernelConfig {
                backends: cfg.fp32_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: (hidden_concat_n + index_heads).into(),
                k: cfg.hidden.clone(),
                input_dtype: bf16,
            });
            resolved.hidden_row_split = Some(Fp32RowSplitLayerNormKernelConfig {
                backends: vec![backend],
                gpu_name: cfg.gpu_name.clone(),
                bf16_lead_cols: (cfg.q_lora_rank.get() + latent).into(),
                norm_cols: index_dim.into(),
                bf16_trail_cols: index_dim.into(),
                fp32_tail_cols: index_heads.into(),
            });
        }
        if let DsaFrontGemm::Fp32SplitStateWriter(backend) = front.hidden_gemm {
            // The same row as `hidden_row_split`; the pooled index update's
            // geometry is `kpool_decode_update`'s.
            let kpool_update = &resolved.kpool_decode_update;
            resolved.hidden_state_writer = Some(DsaFp32SplitStateUpdateKernelConfig {
                backends: vec![backend],
                gpu_name: cfg.gpu_name.clone(),
                next_n: cfg.decode_next_n,
                q_lora_cols: cfg.q_lora_rank.clone(),
                kv_lora_cols: cfg.kv_lora_rank.clone(),
                index_dim: cfg.index_head_dim.clone(),
                fp32_tail_cols: cfg.index_num_heads.clone(),
                pool_size: kpool_update.pool_size,
                tail_ring: kpool_update.tail_ring,
                block_size: kpool_update.block_size,
            });
        }
        // Cut B: [q_b | wq_b] over the normed q latent.
        let q_lora_n = heads * cfg.qk_nope_head_dim.get() + index_heads * index_dim;
        resolved.q_lora_gemm = front_gemm(
            front.q_lora_gemm,
            q_lora_n,
            q_lora_n,
            cfg.q_lora_rank.clone(),
        );
        if let DsaFrontGemm::ConcatHadamardEpilogue(backend) = front.q_lora_gemm {
            resolved.q_lora_epilogue = Some(HadamardFp8QuantWeightScaleKernelConfig {
                backends: vec![backend],
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.index_num_heads.clone(),
                head_dim: cfg.index_head_dim.clone(),
                // The indexer heads follow `q_b`'s columns in the concat row.
                lead_cols: (heads * cfg.qk_nope_head_dim.get()).into(),
                input_dtype: bf16,
            });
        }
        // Cut C: `q_absorb`'s GEMM, its output quantized in the epilogue.
        resolved.q_absorb_fp8 = front.q_absorb_fp8.map(|backend| {
            let absorb = &resolved.q_absorb;
            BatchedGemmStaticFp8QuantKernelConfig {
                backends: vec![backend],
                gpu_name: absorb.gpu_name.clone(),
                num_batches: absorb.num_batches.clone(),
                n: absorb.n.clone(),
                k: absorb.k.clone(),
                dtype: absorb.dtype,
            }
        });
        // Cut D moves what its three launches must: the kv latent in (bf16)
        // and its fp8 cache row out (the normed bf16 copy stays on chip),
        // plus `kpool_decode_update`'s bytes.
        resolved.state_writer = front.state_writer.map(|backend| ElementwiseKernelConfig {
            backends: vec![backend],
            ..ew(
                2 * latent + 524 + 1536 / kpool,
                latent + 512 + (index_dim + 4) / kpool,
            )
        });
        resolved.fork_glue = (front.glue == DsaGlue::ForkLaunchList).then(|| DsaForkGlueConfigs {
            hidden_fp32_cast: ew(2 * cfg.hidden.get(), 4 * cfg.hidden.get()),
            topk_buffer_fill: ew(0, 4 * cfg.selected_k),
            positions_to_int32: ew(8, 4),
            pool_ids_to_int64: ew(4 * cfg.index_topk / kpool, 8 * cfg.index_topk / kpool),
            seq_lens_plus_one: ew(4, 4),
            topk_buffer_copy: ew(4 * window, 4 * window),
        });
        // Cuts E and E + top-k. The page-table kernels are compiled for
        // `index_topk / kpool` pools, `selected_k` slots and the MLA cache
        // block; the block table spans `max_model_len` tokens.
        resolved.topk_page_table =
            front
                .topk_page_table
                .map(|backend| DsaPooledTopkPageTableKernelConfig {
                    backends: vec![backend],
                    gpu_name: cfg.gpu_name.clone(),
                    next_n: cfg.decode_next_n,
                    max_model_len: max_model_len.clone(),
                    top_k: cfg.index_topk / kpool,
                    kpool,
                    page_table_width: cfg.selected_k,
                    block_size: cfg.cache_block_size,
                    index_dtype: "int32".into(),
                });
        resolved.topk_to_page_table =
            front
                .topk_to_page_table
                .map(|backend| DsaPooledLogitsPageTableKernelConfig {
                    backends: vec![backend],
                    gpu_name: cfg.gpu_name.clone(),
                    next_n: cfg.decode_next_n,
                    max_model_len: max_model_len.clone(),
                    // The fork's decode logits rows are `max_model_len / kpool` wide.
                    logits_row_stride: Dim::param("logits_row_stride", cfg.max_model_len / kpool),
                    top_k: cfg.index_topk / kpool,
                    kpool,
                    page_table_width: cfg.selected_k,
                    block_size: cfg.cache_block_size,
                    logits_dtype: DType::Fp32,
                    index_dtype: "int32".into(),
                    context_mode: "uniform_measured".into(),
                });
        resolved.topk_decode_cooperative = (!cfg.cooperative_topk_backends.is_empty()).then(|| {
            DsaPersistentTopkDecodeKernelConfig {
                backends: cfg.cooperative_topk_backends.clone(),
                ..resolved.topk_decode.clone()
            }
        });
        resolved
    }

    pub fn build(
        name: String,
        resolved: Glm53DsaAttnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let topk_decode_slots = sm_count(&r.topk_decode.gpu_name);
        let n = name.as_str();
        let ew = |suffix: &str, config| atomic(n, suffix, config, ElementwiseKernel::build, bridge);
        let gemm =
            |suffix: &str, config| atomic(n, suffix, config, SingleGemmKernel::build, bridge);
        let mut worklet = Self {
            fused_qkv_a: gemm("fused_qkv_a", r.fused_qkv_a)?,
            q_kv_norm: atomic(
                n,
                "q_kv_a_norm",
                r.q_kv_norm,
                QKvRmsNormKernel::build,
                bridge,
            )?,
            q_b: gemm("q_b_proj", r.q_b)?,
            index_wq_b: gemm("indexer.wq_b", r.index_wq_b)?,
            index_wk_weights: gemm("indexer.wk_weights", r.index_wk_weights)?,
            index_head_weights: atomic(
                n,
                "indexer.head_weights",
                r.index_head_weights,
                GemmFp32OutputKernel::build,
                bridge,
            )?,
            index_k_norm: ew("indexer.k_norm", r.index_k_norm)?,
            index_q_fwht_quant: atomic(
                n,
                "indexer.q_fwht_quant",
                r.index_q_fwht_quant,
                HadamardFp8QuantKernel::build,
                bridge,
            )?,
            index_weight_scale: ew("indexer.weight_scale", r.index_weight_scale)?,
            kpool_gate_score: gemm("indexer.kpool_gate_score", r.kpool_gate_score)?,
            prefill_gather: ew("indexer.prefill_gather", r.prefill_gather)?,
            kpool_decode_update: atomic(
                n,
                "indexer.kpool_decode_update",
                r.kpool_decode_update,
                DsaPooledIndexCacheUpdateKernel::build,
                bridge,
            )?,
            kpool_prefill_write: ew("indexer.kpool_prefill_write", r.kpool_prefill_write)?,
            kpool_tail_seed: ew("indexer.kpool_tail_seed", r.kpool_tail_seed)?,
            mqa_logits_decode: atomic(
                n,
                "indexer.mqa_logits_decode",
                r.mqa_logits_decode,
                DsaPagedMqaLogitsDecodeKernel::build,
                bridge,
            )?,
            topk_decode: atomic(
                n,
                "indexer.topk_decode",
                r.topk_decode,
                DsaPersistentTopkDecodeKernel::build,
                bridge,
            )?,
            mqa_logits_prefill: atomic(
                n,
                "indexer.mqa_logits_prefill",
                r.mqa_logits_prefill,
                DsaMqaLogitsPrefillKernel::build,
                bridge,
            )?,
            topk_prefill_packed: None,
            topk_decode_cooperative: None,
            topk_decode_slots,
            topk_prefill: atomic(
                n,
                "indexer.topk_prefill",
                r.topk_prefill,
                DsaTopkPrefillKernel::build,
                bridge,
            )?,
            expand_pools: ew("indexer.expand_pools", r.expand_pools)?,
            q_absorb: atomic(n, "q_absorb", r.q_absorb, BatchedGemmKernel::build, bridge)?,
            q_concat: ew("q_concat", r.q_concat)?,
            q_fp8_quant: ew("q_fp8_quant", r.q_fp8_quant)?,
            sparse_mla: Glm53KpoolSparseMlaOp::build(
                format!("{n}.sparse_mla"),
                r.sparse_mla,
                bridge,
            )?,
            output_copy: ew("output_copy", r.output_copy)?,
            v_up: atomic(n, "v_up", r.v_up, BatchedGemmKernel::build, bridge)?,
            o_proj: gemm("o_proj", r.o_proj)?,
            glue: ew("glue", r.glue)?,
            prefill_glue: ew("prefill_glue", r.prefill_glue)?,
            hidden_gemm: r.hidden_gemm.map(|c| gemm("hidden_gemm", c)).transpose()?,
            hidden_gemm_fp32: r
                .hidden_gemm_fp32
                .map(|c| atomic(n, "hidden_gemm", c, GemmFp32OutputKernel::build, bridge))
                .transpose()?,
            hidden_row_split: r
                .hidden_row_split
                .map(|c| {
                    atomic(
                        n,
                        "hidden_row_split",
                        c,
                        Fp32RowSplitLayerNormKernel::build,
                        bridge,
                    )
                })
                .transpose()?,
            hidden_state_writer: r
                .hidden_state_writer
                .map(|c| {
                    atomic(
                        n,
                        "hidden_state_writer",
                        c,
                        DsaFp32SplitStateUpdateKernel::build,
                        bridge,
                    )
                })
                .transpose()?,
            q_lora_gemm: r.q_lora_gemm.map(|c| gemm("q_lora_gemm", c)).transpose()?,
            q_lora_epilogue: r
                .q_lora_epilogue
                .map(|c| {
                    atomic(
                        n,
                        "q_lora_epilogue",
                        c,
                        HadamardFp8QuantWeightScaleKernel::build,
                        bridge,
                    )
                })
                .transpose()?,
            q_absorb_fp8: r
                .q_absorb_fp8
                .map(|c| {
                    atomic(
                        n,
                        "q_absorb_fp8",
                        c,
                        BatchedGemmStaticFp8QuantKernel::build,
                        bridge,
                    )
                })
                .transpose()?,
            state_writer: r.state_writer.map(|c| ew("state_writer", c)).transpose()?,
            fork_glue: r
                .fork_glue
                .map(|g| -> Result<DsaForkGlue, BuildError> {
                    Ok(DsaForkGlue {
                        hidden_fp32_cast: ew("glue.hidden_fp32_cast", g.hidden_fp32_cast)?,
                        topk_buffer_fill: ew("glue.topk_buffer_fill", g.topk_buffer_fill)?,
                        positions_to_int32: ew("glue.positions_to_int32", g.positions_to_int32)?,
                        pool_ids_to_int64: ew("glue.pool_ids_to_int64", g.pool_ids_to_int64)?,
                        seq_lens_plus_one: ew("glue.seq_lens_plus_one", g.seq_lens_plus_one)?,
                        topk_buffer_copy: ew("glue.topk_buffer_copy", g.topk_buffer_copy)?,
                    })
                })
                .transpose()?,
            topk_page_table: r
                .topk_page_table
                .map(|c| {
                    atomic(
                        n,
                        "indexer.topk_page_table",
                        c,
                        DsaPooledTopkPageTableKernel::build,
                        bridge,
                    )
                })
                .transpose()?,
            topk_to_page_table: r
                .topk_to_page_table
                .map(|c| {
                    atomic(
                        n,
                        "indexer.topk_to_page_table",
                        c,
                        DsaPooledLogitsPageTableKernel::build,
                        bridge,
                    )
                })
                .transpose()?,
            name,
            resolved,
        };
        // A fused cut is priced by its own backend's rows or not at all: a
        // run-config override to another backend would price a different
        // kernel under its name.
        let front = worklet.resolved.raw_cfg.front;
        let fused = [
            (
                front.hidden_gemm,
                worklet
                    .hidden_gemm
                    .as_ref()
                    .map(|op| (&op.name, &op.kernel.config.backends, "single_gemm")),
            ),
            (
                front.q_lora_gemm,
                worklet
                    .q_lora_gemm
                    .as_ref()
                    .map(|op| (&op.name, &op.kernel.config.backends, "single_gemm")),
            ),
        ]
        .into_iter()
        .filter_map(|(mode, op)| match mode {
            DsaFrontGemm::Fused(backend) => {
                op.map(|(role, built, kind)| (backend, role, built, kind))
            }
            _ => None,
        })
        .chain(match front.hidden_gemm {
            DsaFrontGemm::Fp32Split(backend) | DsaFrontGemm::Fp32SplitStateWriter(backend) => {
                worklet.hidden_row_split.as_ref().map(|op| {
                    (
                        backend,
                        &op.name,
                        &op.kernel.config.backends,
                        "fp32_row_split_layer_norm",
                    )
                })
            }
            _ => None,
        })
        .chain(match front.hidden_gemm {
            DsaFrontGemm::Fp32SplitStateWriter(backend) => {
                worklet.hidden_state_writer.as_ref().map(|op| {
                    (
                        backend,
                        &op.name,
                        &op.kernel.config.backends,
                        "dsa_fp32_split_state_update",
                    )
                })
            }
            _ => None,
        })
        .chain(match front.q_lora_gemm {
            DsaFrontGemm::ConcatHadamardEpilogue(backend) => {
                worklet.q_lora_epilogue.as_ref().map(|op| {
                    (
                        backend,
                        &op.name,
                        &op.kernel.config.backends,
                        "hadamard_fp8_quant_weight_scale",
                    )
                })
            }
            _ => None,
        })
        .chain(
            front
                .q_absorb_fp8
                .zip(worklet.q_absorb_fp8.as_ref())
                .map(|(backend, op)| {
                    (
                        backend,
                        &op.name,
                        &op.kernel.config.backends,
                        "batched_gemm_static_fp8_quant",
                    )
                }),
        )
        .chain(
            front
                .state_writer
                .zip(worklet.state_writer.as_ref())
                .map(|(backend, op)| {
                    (backend, &op.name, &op.kernel.config.backends, "elementwise")
                }),
        )
        .chain(
            front
                .topk_page_table
                .zip(worklet.topk_page_table.as_ref())
                .map(|(backend, op)| {
                    (
                        backend,
                        &op.name,
                        &op.kernel.config.backends,
                        "dsa_pooled_topk_page_table",
                    )
                }),
        )
        .chain(
            front
                .topk_to_page_table
                .zip(worklet.topk_to_page_table.as_ref())
                .map(|(backend, op)| {
                    (
                        backend,
                        &op.name,
                        &op.kernel.config.backends,
                        "dsa_pooled_logits_page_table",
                    )
                }),
        );
        for (backend, role, built, kind) in fused {
            if built.as_slice() != [backend] {
                return Err(BuildError::NoBackendForLaunch {
                    kind,
                    role: role.clone(),
                    backends: built.clone(),
                    launch: "a fused DSA-front cut, which only its configured backend prices",
                });
            }
        }
        if front.split_state_writer() && front.state_writer.is_some() {
            return Err(BuildError::FitFailed {
                kind: "dsa_fp32_split_state_update",
                reason: "cut A's split state writer and cut D (state_writer) write the same kv \
                         norm, MLA cache append and kpool update; set one"
                    .into(),
            });
        }
        let both = front.topk_page_table.is_some() && front.topk_to_page_table.is_some();
        match front.topk_to_page_table_max_pools {
            None if both => {
                return Err(BuildError::FitFailed {
                    kind: "dsa_pooled_logits_page_table",
                    reason: "cut E (topk_page_table) and cut E + top-k (topk_to_page_table) build \
                             the same page table; set one, or both with \
                             topk_to_page_table_max_pools to dispatch by context"
                        .into(),
                });
            }
            Some(max_pools) if !both || max_pools == 0 => {
                return Err(BuildError::FitFailed {
                    kind: "dsa_pooled_logits_page_table",
                    reason: "topk_to_page_table_max_pools dispatches between cut E + top-k and \
                             cut E: it needs both backends and a positive pool count"
                        .into(),
                });
            }
            _ => {}
        }
        // The backends as built, after any run-config override.
        let topk = &worklet.topk_prefill;
        let backends = topk.kernel.config.backends.clone();
        let packed: Vec<&'static str> = backends
            .iter()
            .copied()
            .filter(|&backend| takes_row_starts(backend))
            .collect();
        if packed.is_empty() {
            return Err(BuildError::NoBackendForLaunch {
                kind: "dsa_topk_prefill",
                role: topk.name.clone(),
                backends,
                launch: "a launch packing several requests (row starts)",
            });
        }
        if packed.len() < backends.len() {
            worklet.topk_prefill_packed = Some(Arc::new(DsaTopkPrefillKernel::build(
                format!("{}.packed", topk.name),
                DsaTopkPrefillKernelConfig {
                    backends: packed,
                    ..topk.kernel.config.clone()
                },
                bridge,
            )?));
        }
        // The fork's "auto" chain runs `cooperative_topk` up to 64 rows and the
        // persistent kernel past it; checked on the backends as built, after
        // any run-config override. DeepSelect past 64 rows is the same chain
        // with its long-row kernel swapped.
        let cooperative_capable = compute_capability(&worklet.topk_decode.kernel.config.gpu_name)
            .is_some_and(|(major, _)| major >= 9 && major != 12);
        let auto_chain = worklet
            .topk_decode
            .kernel
            .config
            .backends
            .iter()
            .all(|backend| LONG_ROW_BACKENDS.contains(backend));
        if let (Some(config), true) = (
            &r.topk_decode_cooperative,
            cooperative_capable && auto_chain,
        ) {
            worklet.topk_decode_cooperative = Some(Arc::new(DsaPersistentTopkDecodeKernel::build(
                format!("{}.indexer.topk_decode.cooperative", worklet.name),
                config.clone(),
                bridge,
            )?));
        }
        Ok(worklet)
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (Glm53DsaAttnLocalWorklet) [TP rank; H={}, index heads={} replicated, \
                 kpool top-{} x {}]",
                self.name, cfg.num_heads, cfg.index_num_heads, cfg.index_topk, cfg.index_kpool
            ),
            child: Box::new(CostNode::Sum(self.compile_slots(builder))),
        }
    }

    /// The fork's launches, then each enabled DSA-front fusion (the launches
    /// it replaces bill zero in `eval`).
    fn compile_slots(&self, builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut slots = vec![
            self.fused_qkv_a.compile(builder),
            self.q_kv_norm.compile(builder),
            self.q_b.compile(builder),
            self.index_wq_b.compile(builder),
            self.index_wk_weights.compile(builder),
            self.index_head_weights.compile(builder),
            self.index_k_norm.compile(builder),
            self.index_q_fwht_quant.compile(builder),
            self.index_weight_scale.compile(builder),
            self.kpool_gate_score.compile(builder),
            repeated(&self.prefill_gather, PREFILL_GATHER_LAUNCHES, builder),
            self.kpool_decode_update.compile(builder),
            self.kpool_prefill_write.compile(builder),
            self.kpool_tail_seed.compile(builder),
            self.mqa_logits_decode.compile(builder),
            self.topk_decode.compile(builder),
            self.mqa_logits_prefill.compile(builder),
            self.topk_prefill.compile(builder),
            self.expand_pools.compile(builder),
            self.q_absorb.compile(builder),
            self.q_concat.compile(builder),
            self.q_fp8_quant.compile(builder),
            self.sparse_mla.compile(builder),
            self.output_copy.compile(builder),
            self.v_up.compile(builder),
            self.o_proj.compile(builder),
            repeated(&self.glue, GLUE_LAUNCHES, builder),
            repeated(&self.prefill_glue, PREFILL_GLUE_LAUNCHES, builder),
        ];
        slots.extend(self.hidden_gemm.as_ref().map(|op| op.compile(builder)));
        slots.extend(self.hidden_gemm_fp32.as_ref().map(|op| op.compile(builder)));
        slots.extend(self.hidden_row_split.as_ref().map(|op| op.compile(builder)));
        slots.extend(self.hidden_state_writer.as_ref().map(|op| op.compile(builder)));
        slots.extend(self.q_lora_gemm.as_ref().map(|op| op.compile(builder)));
        slots.extend(self.q_lora_epilogue.as_ref().map(|op| op.compile(builder)));
        slots.extend(self.q_absorb_fp8.as_ref().map(|op| op.compile(builder)));
        slots.extend(self.state_writer.as_ref().map(|op| op.compile(builder)));
        slots.extend(self.topk_page_table.as_ref().map(|op| op.compile(builder)));
        slots.extend(self.topk_to_page_table.as_ref().map(|op| op.compile(builder)));
        if let Some(g) = &self.fork_glue {
            slots.extend([
                g.hidden_fp32_cast.compile(builder),
                g.topk_buffer_fill.compile(builder),
                repeated(&g.positions_to_int32, self.position_casts(), builder),
                g.pool_ids_to_int64.compile(builder),
                g.seq_lens_plus_one.compile(builder),
                g.topk_buffer_copy.compile(builder),
            ]);
        }
        slots
    }

    pub fn eval(&self, input: &Glm53DsaAttnLocalWorkletInput, ev: &mut Evaluator) {
        let cfg = &self.resolved.raw_cfg;
        let w = derive_work(input, &self.indexer_budget())
            .unwrap_or_else(|reason| panic!("invalid Glm53DsaAttnLocalWorkletInput: {reason}"));
        let rows = SingleGemmKernelInput { m: w.total_tokens };
        let tokens = |num_tokens| ElementwiseKernelInput { num_tokens };
        let all = tokens(w.total_tokens);
        let no_prefill = w.prefill_tokens == 0;
        let no_decode = w.decode_rows == 0;
        // What the DSA-front fusions take over (each bills in its own slot).
        let front = cfg.front;
        let hidden_merged = front.hidden_gemm.merged();
        let hidden_fused = front.hidden_gemm.fused();
        let q_lora_merged = front.q_lora_gemm.merged();
        let q_lora_fused = front.q_lora_gemm.fused();
        let index_q_absorbed = front.q_lora_gemm.absorbs_index_q_quant();
        let absorb_fused = front.q_absorb_fp8.is_some();
        let state_writer = front.state_writer.is_some();
        // Cut A's split state writer takes the decode rows' q/kv norm, cache
        // append, kpool position cast and kpool update; prefill rows keep them.
        let split_writer = front.split_state_writer();
        let fork_glue = front.glue == DsaGlue::ForkLaunchList;
        // Cut E or E + top-k builds the decode rows' page table in one launch;
        // E + top-k also selects their pools. With a dispatch threshold the
        // launch's longest row picks the cut.
        let cut = front.decode_page_table_cut(w.decode_pools);
        let absorbed = |slot: &str| front.decode_slots_absorbed(w.decode_pools).contains(&slot);
        push_or_zero(&self.fused_qkv_a, rows.clone(), hidden_merged, ev);
        // The q half of the norm folds into cut B's prologue and the kv half
        // into cut D; with only one of them the other half still launches,
        // priced as the whole (latency-bound) norm.
        let norm_tokens = if split_writer {
            w.prefill_tokens
        } else {
            w.total_tokens
        };
        push_or_zero(
            &self.q_kv_norm,
            QKvRmsNormKernelInput {
                num_tokens: norm_tokens,
            },
            (q_lora_fused && state_writer) || norm_tokens == 0,
            ev,
        );
        push_or_zero(&self.q_b, rows.clone(), q_lora_merged, ev);
        push_or_zero(&self.index_wq_b, rows.clone(), q_lora_merged, ev);
        push_or_zero(&self.index_wk_weights, rows.clone(), hidden_merged, ev);
        push_or_zero(
            &self.index_head_weights,
            GemmFp32OutputKernelInput { m: w.total_tokens },
            hidden_fused,
            ev,
        );
        push_or_zero(&self.index_k_norm, all.clone(), hidden_fused, ev);
        push_or_zero(
            &self.index_q_fwht_quant,
            HadamardFp8QuantKernelInput {
                num_rows: w.total_tokens * cfg.index_num_heads.get(),
            },
            index_q_absorbed,
            ev,
        );
        push_or_zero(&self.index_weight_scale, all.clone(), index_q_absorbed, ev);
        push_or_zero(&self.kpool_gate_score, rows.clone(), hidden_merged, ev);
        push_or_zero(
            &self.prefill_gather,
            tokens(w.prefill_tokens),
            no_prefill,
            ev,
        );
        push_or_zero(
            &self.kpool_decode_update,
            DsaPooledIndexCacheUpdateKernelInput {
                // Logged as the requests this slot bills.
                num_requests: if split_writer { 0 } else { w.decode_requests },
            },
            no_decode || state_writer || split_writer,
            ev,
        );
        push_or_zero(
            &self.kpool_prefill_write,
            tokens(w.prefill_tokens),
            no_prefill,
            ev,
        );
        push_or_zero(
            &self.kpool_tail_seed,
            tokens(w.prefill_tokens),
            no_prefill,
            ev,
        );
        push_or_zero(
            &self.mqa_logits_decode,
            DsaPagedMqaLogitsDecodeKernelInput {
                batch_size: w.decode_requests,
                context_len: w.decode_mean_pools,
            },
            no_decode,
            ev,
        );
        if absorbed("indexer.topk_decode") {
            push_or_zero(
                &self.topk_decode,
                DsaPersistentTopkDecodeKernelInput {
                    batch_size: 0,
                    context_len: 0,
                },
                true,
                ev,
            );
        } else {
            self.eval_decode_topk(&input.decode_kv_lens, &w, ev);
        }
        self.eval_prefill_indexer(&w.prefill_indexer_chunks, ev);
        // Cuts E and E + top-k expand the decode rows' pools themselves.
        let expand_rows = w.prefill_indexer_rows
            + if absorbed("indexer.expand_pools") {
                0
            } else {
                w.decode_rows
            };
        push_or_zero(
            &self.expand_pools,
            tokens(expand_rows),
            expand_rows == 0,
            ev,
        );
        push_or_zero(
            &self.q_absorb,
            BatchedGemmKernelInput { m: w.total_tokens },
            absorb_fused,
            ev,
        );
        push_or_zero(&self.q_concat, all.clone(), !cfg.mla_layout_copies, ev);
        push_or_zero(&self.q_fp8_quant, all.clone(), absorb_fused, ev);
        self.sparse_mla.eval(
            &Glm53KpoolSparseMlaInput {
                prefill_query_cache_pairs: input
                    .prefill_chunk_pairs
                    .iter()
                    .map(|&(prefix, append)| (append, prefix + append))
                    .collect(),
                decode_context_lens: decode_row_context_lens(
                    &input.decode_kv_lens,
                    cfg.decode_next_n,
                ),
                cache_append_elsewhere: state_writer,
                decode_cache_append_elsewhere: split_writer,
                decode_remap_elsewhere: absorbed("sparse_mla.index_remap"),
            },
            ev,
        );
        push_or_zero(&self.output_copy, all.clone(), !cfg.mla_layout_copies, ev);
        push_or_zero(
            &self.v_up,
            BatchedGemmKernelInput { m: w.total_tokens },
            false,
            ev,
        );
        push_or_zero(&self.o_proj, rows.clone(), false, ev);
        push_or_zero(&self.glue, all.clone(), fork_glue, ev);
        push_or_zero(&self.prefill_glue, all.clone(), no_prefill, ev);
        if let Some(op) = &self.hidden_gemm {
            push_or_zero(op, rows.clone(), false, ev);
        }
        if let Some(op) = &self.hidden_gemm_fp32 {
            push_or_zero(op, GemmFp32OutputKernelInput { m: w.total_tokens }, false, ev);
        }
        if let Some(op) = &self.hidden_row_split {
            let num_tokens = if split_writer {
                w.prefill_tokens
            } else {
                w.total_tokens
            };
            let input = Fp32RowSplitLayerNormKernelInput { num_tokens };
            push_or_zero(op, input, num_tokens == 0, ev);
        }
        if let Some(op) = &self.hidden_state_writer {
            let input = DsaFp32SplitStateUpdateKernelInput {
                num_requests: w.decode_requests,
            };
            push_or_zero(op, input, no_decode, ev);
        }
        if let Some(op) = &self.q_lora_gemm {
            push_or_zero(op, rows.clone(), false, ev);
        }
        if let Some(op) = &self.q_lora_epilogue {
            let input = HadamardFp8QuantWeightScaleKernelInput {
                num_tokens: w.total_tokens,
            };
            push_or_zero(op, input, false, ev);
        }
        if let Some(op) = &self.q_absorb_fp8 {
            let input = BatchedGemmStaticFp8QuantKernelInput { m: w.total_tokens };
            push_or_zero(op, input, false, ev);
        }
        if let Some(op) = &self.state_writer {
            push_or_zero(op, all.clone(), false, ev);
        }
        if let Some(op) = &self.topk_page_table {
            let input = DsaPooledTopkPageTableKernelInput {
                batch_size: w.decode_requests,
                context_len: w.decode_pools,
            };
            push_or_zero(op, input, no_decode || cut != DecodePageTableCut::PageTable, ev);
        }
        if let Some(op) = &self.topk_to_page_table {
            let input = DsaPooledLogitsPageTableKernelInput {
                batch_size: w.decode_requests,
                context_len: self.fused_topk_context(&w),
            };
            let idle = no_decode || cut != DecodePageTableCut::TopkToPageTable;
            push_or_zero(op, input, idle, ev);
        }
        if let Some(g) = &self.fork_glue {
            let decode = tokens(w.decode_rows);
            push_or_zero(&g.hidden_fp32_cast, all, hidden_fused, ev);
            let gone = |slot: &str| no_decode || absorbed(slot);
            push_or_zero(&g.topk_buffer_fill, decode.clone(), gone("glue.topk_buffer_fill"), ev);
            push_or_zero(&g.positions_to_int32, decode.clone(), no_decode, ev);
            push_or_zero(&g.pool_ids_to_int64, decode.clone(), gone("glue.pool_ids_to_int64"), ev);
            push_or_zero(&g.seq_lens_plus_one, decode.clone(), gone("glue.seq_lens_plus_one"), ev);
            push_or_zero(&g.topk_buffer_copy, decode, gone("glue.topk_buffer_copy"), ev);
        }
    }
}

impl Glm53DsaAttnLocalWorklet {
    /// Fork-glue position casts per decode step: G3 (kpool `dec_pos`) and G5
    /// (top-k seq lens); cut A's split state writer reads int64 positions, so
    /// G3 is gone.
    fn position_casts(&self) -> u32 {
        if self.resolved.raw_cfg.front.split_state_writer() {
            POSITION_CASTS - 1
        } else {
            POSITION_CASTS
        }
    }

    /// The context `topk_to_page_table` prices at. One CTA per row: a batch
    /// within one wave of [`FUSED_TOPK_CTAS_PER_SM`] CTAs per SM takes as long
    /// as its longest row; past a wave the rows queue on the slots and the time
    /// follows the batch's total pools, priced at the mean context.
    fn fused_topk_context(&self, w: &Work) -> u32 {
        let slots = self
            .topk_decode_slots
            .map(|sms| sms * FUSED_TOPK_CTAS_PER_SM);
        match slots {
            Some(slots) if w.decode_rows > slots => w.decode_mean_pools,
            _ => w.decode_pools,
        }
    }

    fn indexer_budget(&self) -> IndexerBudget {
        let cfg = &self.resolved.raw_cfg;
        IndexerBudget {
            decode_next_n: cfg.decode_next_n,
            index_topk: cfg.index_topk,
            index_kpool: cfg.index_kpool,
            max_logits_elems: cfg.indexer_max_logits_bytes / u64::from(DType::Fp32.size_bytes()),
            max_keys: PREFILL_KEY_WORKSPACE_PER_MODEL_LEN * u64::from(cfg.max_model_len),
        }
    }

    /// The decode top-k runs one CTA per row. A batch that fits one wave of
    /// `topk_decode_slots` rows takes as long as its longest row: the uniform
    /// grid at the batch's max context. A larger batch is list-scheduled: rows
    /// go in order to the earliest-free slot, each costing its own context's
    /// per-row share of a one-wave launch, plus that launch's fixed part once.
    /// Against real B200 batches (logs/20261112_7_topk_hybrid, 120-644 rows)
    /// this lands within +1..+6% median and +9% p90, where the max context
    /// overprices by up to 2.4x and the mean context underprices by 10-43%.
    fn eval_decode_topk(&self, kv_lens: &[u32], w: &Work, ev: &mut Evaluator) {
        let input = |batch_size, context_len| DsaPersistentTopkDecodeKernelInput {
            batch_size,
            context_len,
        };
        let longest = input(w.decode_requests, w.decode_pools);
        if let Some(cooperative) = &self.topk_decode_cooperative {
            if w.decode_rows > 0 && w.decode_rows <= COOPERATIVE_MAX_ROWS {
                // One wave of CTA clusters over every row: the longest row
                // sets the time (as the QSA indexer's cooperative leaf).
                let mut metrics = cooperative.eval(&longest);
                metrics.backend_index = LeafMetrics::NO_BACKEND;
                return ev.push(metrics, || longest.into());
            }
        }
        let kernel = &self.topk_decode.kernel;
        let slots = match self.topk_decode_slots {
            Some(slots) if w.decode_rows > slots => slots,
            _ => return push_or_zero(&self.topk_decode, longest, w.decode_rows == 0, ev),
        };
        let cfg = &self.resolved.raw_cfg;
        let next_n = cfg.decode_next_n;
        // The one-wave launch the per-row costs come from: as many requests as
        // fill `TOPK_WAVE_ROWS` rows.
        let wave = (TOPK_WAVE_ROWS / next_n).max(1);
        let fixed = kernel.eval(&input(wave, cfg.index_topk / cfg.index_kpool));
        let fixed_ms = f64::from(fixed.m.time_ms);
        let per_row_ms = |pools| {
            let one_wave = kernel.eval(&input(wave, pools));
            (f64::from(one_wave.m.time_ms) - fixed_ms).max(0.0)
        };
        let time_ms = fixed_ms
            + list_schedule_ms(
                kv_lens.iter().map(|&kv| kv.div_ceil(cfg.index_kpool)),
                next_n,
                slots,
                per_row_ms,
            );
        // Bytes and FLOPs follow the batch's total pools.
        let mut metrics = kernel.eval(&input(w.decode_requests, w.decode_mean_pools));
        let scale = time_ms as f32 / metrics.m.time_ms;
        metrics.m.time_ms = time_ms as f32;
        if scale.is_finite() {
            metrics.m.energy_j *= scale;
        }
        ev.push(metrics, || longest.into());
    }

    /// One MQA-logits and one top-k launch per indexer chunk, each slot summing
    /// its launches. A chunk packing several requests runs its top-k on
    /// `topk_prefill_packed` when the configured backends cannot take it.
    fn eval_prefill_indexer(&self, chunks: &[IndexerChunk], ev: &mut Evaluator) {
        let mut logits = LeafMetrics::ZERO;
        let mut topk = LeafMetrics::ZERO;
        for chunk in chunks {
            logits.add_fanin(self.mqa_logits_prefill.kernel.eval(
                &DsaMqaLogitsPrefillKernelInput {
                    num_queries: chunk.rows,
                    num_keys: chunk.keys,
                },
            ));
            let input = DsaTopkPrefillKernelInput {
                num_queries: chunk.rows,
                num_keys: chunk.keys,
            };
            topk.add_fanin(match (&self.topk_prefill_packed, chunk.single_request) {
                (Some(packed), false) => {
                    let mut metrics = packed.eval(&input);
                    // Report the pick against the slot's own backend list.
                    let picked = packed.config.backends[usize::from(metrics.backend_index)];
                    metrics.backend_index = self
                        .topk_prefill
                        .kernel
                        .config
                        .backends
                        .iter()
                        .position(|&backend| backend == picked)
                        .map_or(LeafMetrics::NO_BACKEND, |index| index as u8);
                    metrics
                }
                _ => self.topk_prefill.kernel.eval(&input),
            });
        }
        let shapes = || {
            SlotInput::from(DsaIndexerPrefillLog {
                prefill_query_key_pairs: chunks.iter().map(|c| (c.rows, c.keys)).collect(),
            })
        };
        ev.push(logits, shapes);
        ev.push(topk, shapes);
    }
}

struct Work {
    total_tokens: u32,
    prefill_tokens: u32,
    decode_rows: u32,
    /// Decode requests: `decode_rows / decode_next_n`.
    decode_requests: u32,
    /// `ceil(max decode context / index_kpool)`.
    decode_pools: u32,
    /// `ceil(Σ ceil(decode context / index_kpool) / decode_requests)`: the
    /// uniform context with the batch's total pools. DeepGEMM's paged MQA
    /// logits balances SMs by each request's own context, so its time follows
    /// the total pools, not `decode_requests * decode_pools`; ragged B200
    /// measurements (logs/20261112_0_indexer_ragged) match a uniform batch at
    /// this mean context within a few percent.
    decode_mean_pools: u32,
    /// Every prefill row once any prefill context exceeds `index_topk`, else
    /// 0: vLLM skips the prefill indexer only when every context fits, and
    /// otherwise scores and selects for all prefill rows.
    prefill_indexer_rows: u32,
    /// The logits + top-k launches over those rows (empty when skipped).
    prefill_indexer_chunks: Vec<IndexerChunk>,
}

/// vLLM `get_max_prefill_buffer_size`: the gathered index-K workspace holds
/// `40 * max_model_len` keys, and a request pack never exceeds it.
const PREFILL_KEY_WORKSPACE_PER_MODEL_LEN: u64 = 40;

/// The `topk_decode` backends that run past [`COOPERATIVE_MAX_ROWS`] behind
/// the cooperative kernel. `vllm_cuda` (persistent_topk) is the fork's "auto"
/// chain, the default. `deep_select` there is not a fork setting: the fork's
/// explicit `deep_select` backend runs every batch on DeepSelect, while this
/// keeps cooperative_topk up to 64 rows (a one-line change to `_resolve_auto`).
/// Any other override (torch) runs every batch on itself.
const LONG_ROW_BACKENDS: [&str; 2] = ["vllm_cuda", "deep_select"];

/// CTAs per SM of the fused top-k + page-table kernel
/// (`__launch_bounds__(512, 2)` with ~70 KB of shared memory per CTA).
const FUSED_TOPK_CTAS_PER_SM: u32 = 2;

/// Rows of the one-wave uniform launch `eval_decode_topk` prices each row
/// from: below the SM count, so the measured B axis does not straddle a wave
/// edge (B200: next_n 4 at B32, next_n 8 at B16, both grid knots).
const TOPK_WAVE_ROWS: u32 = 128;

/// Greedy in-order dispatch of each request's `next_n` rows onto `slots`
/// parallel slots, a row costing `row_ms(pools)`; the makespan.
fn list_schedule_ms(
    pools: impl Iterator<Item = u32>,
    next_n: u32,
    slots: u32,
    mut row_ms: impl FnMut(u32) -> f64,
) -> f64 {
    /// A slot's free time, ordered so the max-heap yields the earliest.
    struct Free(f64);
    impl Ord for Free {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other.0.total_cmp(&self.0)
        }
    }
    impl PartialOrd for Free {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl PartialEq for Free {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other).is_eq()
        }
    }
    impl Eq for Free {}
    let mut heap: BinaryHeap<Free> = (0..slots).map(|_| Free(0.0)).collect();
    for pools in pools {
        let cost = row_ms(pools);
        for _ in 0..next_n {
            let mut slot = heap.peek_mut().expect("at least one slot");
            slot.0 += cost;
        }
    }
    heap.into_iter().map(|free| free.0).fold(0.0, f64::max)
}

/// What bounds the prefill indexer's launches.
struct IndexerBudget {
    decode_next_n: u32,
    index_topk: u32,
    index_kpool: u32,
    /// fp32 logits elements one call may hold.
    max_logits_elems: u64,
    /// Pools one request pack may gather.
    max_keys: u64,
}

/// One MQA-logits call and the top-k over its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IndexerChunk {
    rows: u32,
    /// The single-causal-tail key count whose `rows x keys` triangle holds as
    /// many (row, pool) pairs as the chunk, where the row at position `pos`
    /// spans `(pos + 1) / index_kpool` completed pools. Never below `rows`
    /// (the kernel contract `rows <= keys`).
    keys: u32,
    /// Every row starts at key 0: one request, or a query slice of one. A
    /// pack of several requests offsets each later request's rows.
    single_request: bool,
}

/// vLLM's `_split_indexer_prefill_chunks` (`v1/attention/backends/mla/
/// indexer.py`, which the fork's kpool indexer loops over): pack requests in
/// order while the pack's `rows x pools` stays within the logits budget and
/// its pools within the workspace; a request that alone exceeds the budget is
/// split along its queries into `max_logits_elems / pools`-row slices. Pools
/// are `context / index_kpool` per request; a pack with none is skipped.
fn plan_indexer_chunks(
    pairs: &[(u32, u32)],
    budget: &IndexerBudget,
) -> Result<Vec<IndexerChunk>, String> {
    let kpool = budget.index_kpool;
    let pools = |&(prefix, append): &(u32, u32)| u64::from((prefix + append) / kpool);
    let mut chunks = Vec::new();
    let mut end = 0;
    while end < pairs.len() {
        let start = end;
        let (mut rows, mut keys) = (0_u64, 0_u64);
        while end < pairs.len() {
            let (new_rows, new_keys) = (rows + u64::from(pairs[end].1), keys + pools(&pairs[end]));
            if new_keys > budget.max_keys || new_rows * new_keys > budget.max_logits_elems {
                break;
            }
            (rows, keys) = (new_rows, new_keys);
            end += 1;
        }
        if end == start {
            (rows, keys) = (u64::from(pairs[end].1), pools(&pairs[end]));
            end += 1;
        }
        if keys == 0 {
            continue;
        }
        let pack = &pairs[start..end];
        let slice_rows = (budget.max_logits_elems / keys).max(1);
        let mut offset = 0;
        while offset < rows {
            let slice = slice_rows.min(rows - offset);
            chunks.push(indexer_chunk(pack, offset, slice, kpool)?);
            offset += slice;
        }
    }
    Ok(chunks)
}

/// The chunk over rows `[offset, offset + slice)` of a pack's concatenated
/// queries.
fn indexer_chunk(
    pack: &[(u32, u32)],
    offset: u64,
    slice: u64,
    kpool: u32,
) -> Result<IndexerChunk, String> {
    let mut pooled_pairs = 0_u64;
    let mut first_row = 0_u64;
    for &(prefix, append) in pack {
        let lo = offset.max(first_row);
        let hi = (offset + slice).min(first_row + u64::from(append));
        if lo < hi {
            // Rows of this request at positions prefix + (lo..hi) - first_row.
            let from = u64::from(prefix) + lo - first_row;
            let to = u64::from(prefix) + hi - first_row;
            pooled_pairs += pool_prefix_sum(to as u32, kpool) - pool_prefix_sum(from as u32, kpool);
        }
        first_row += u64::from(append);
    }
    // rows * keys - rows * (rows - 1) / 2 == pooled_pairs, solved for keys.
    let keys = (2 * pooled_pairs + slice * (slice - 1)).div_ceil(2 * slice);
    Ok(IndexerChunk {
        rows: u32::try_from(slice).map_err(|_| "prefill indexer rows overflow u32")?,
        keys: u32::try_from(keys.max(slice)).map_err(|_| "prefill indexer keys overflow u32")?,
        single_request: pack.len() == 1,
    })
}

fn derive_work(
    input: &Glm53DsaAttnLocalWorkletInput,
    budget: &IndexerBudget,
) -> Result<Work, String> {
    let (index_topk, index_kpool) = (budget.index_topk, budget.index_kpool);
    let mut prefill_tokens = 0_u32;
    let mut needs_prefill_indexer = false;
    for (index, &(prefix, append)) in input.prefill_chunk_pairs.iter().enumerate() {
        if append == 0 {
            return Err(format!("prefill request {index} append must be positive"));
        }
        let context = prefix
            .checked_add(append)
            .ok_or("prefill context overflows u32")?;
        prefill_tokens = prefill_tokens
            .checked_add(append)
            .ok_or("prefill token sum overflows u32")?;
        needs_prefill_indexer |= context > index_topk;
    }
    let (prefill_indexer_rows, prefill_indexer_chunks) = if needs_prefill_indexer {
        (
            prefill_tokens,
            plan_indexer_chunks(&input.prefill_chunk_pairs, budget)?,
        )
    } else {
        (0, Vec::new())
    };
    let decode_requests = input.decode_kv_lens.len() as u32;
    let next_n = budget.decode_next_n;
    if input.decode_kv_lens.iter().any(|&context| context < next_n) {
        return Err(format!(
            "decode KV lengths must cover their decode_next_n {next_n} query rows"
        ));
    }
    let decode_rows = decode_requests
        .checked_mul(next_n)
        .ok_or("decode row count overflows u32")?;
    let max_decode = input.decode_kv_lens.iter().copied().max().unwrap_or(0);
    let total_decode_pools: u64 = input
        .decode_kv_lens
        .iter()
        .map(|&context| u64::from(context.div_ceil(index_kpool)))
        .sum();
    let decode_mean_pools = if decode_requests == 0 {
        0
    } else {
        u32::try_from(total_decode_pools.div_ceil(u64::from(decode_requests)))
            .map_err(|_| "decode mean pools overflow u32")?
    };
    let total_tokens = prefill_tokens
        .checked_add(decode_rows)
        .ok_or("token sum overflows u32")?;
    if total_tokens == 0 {
        return Err("an iteration must carry at least one token".into());
    }
    Ok(Work {
        total_tokens,
        prefill_tokens,
        decode_rows,
        decode_requests,
        decode_pools: max_decode.div_ceil(index_kpool),
        decode_mean_pools,
        prefill_indexer_rows,
        prefill_indexer_chunks,
    })
}

/// Each request's query rows, at its last `next_n` positions.
fn decode_row_context_lens(kv_lens: &[u32], next_n: u32) -> Vec<u32> {
    kv_lens
        .iter()
        .flat_map(|&kv_len| kv_len + 1 - next_n..=kv_len)
        .collect()
}

/// `sum over x in 1..=n of x / kpool`: the pools spanned by the rows at
/// positions `0..n`.
fn pool_prefix_sum(n: u32, kpool: u32) -> u64 {
    let (n, kpool) = (u64::from(n), u64::from(kpool));
    let (q, r) = (n / kpool, n % kpool);
    kpool * q * q.saturating_sub(1) / 2 + q * (r + 1)
}

fn integer_sqrt(value: u32) -> u32 {
    let root = f64::from(value).sqrt().round() as u32;
    assert_eq!(root * root, value, "softmax scale needs a square head dim");
    root
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    pub(crate) fn cfg() -> Glm53DsaAttnLocalWorkletConfig {
        Glm53DsaAttnLocalWorkletConfig {
            hidden: 4096.into(),
            num_heads: 16.into(),
            q_lora_rank: 1536.into(),
            kv_lora_rank: 512.into(),
            qk_nope_head_dim: 256.into(),
            v_head_dim: 256.into(),
            index_num_heads: 32.into(),
            index_head_dim: 128.into(),
            index_topk: 2048,
            index_kpool: 4,
            selected_k: 2176,
            max_model_len: 8192,
            cache_block_size: 64,
            rms_eps: 1e-5,
            gpu_name: "NVIDIA B200".into(),
            bf16_gemm_backends: vec!["torch_linear_vllm"],
            fp32_gemm_backends: vec!["torch_cublas"],
            qkv_norm_backends: vec!["vllm_triton"],
            mla_bmm_q_absorb_backends: vec!["torch_mla_q_absorb_no_rope"],
            mla_bmm_v_up_backends: vec!["torch_mla_v_up_unpadded"],
            mqa_logits_backends: vec!["deepgemm_fp8"],
            topk_backends: vec!["vllm_cuda"],
            cooperative_topk_backends: vec!["vllm_cuda_cooperative"],
            mqa_logits_prefill_backends: vec!["deepgemm_fp8"],
            topk_prefill_backends: vec!["vllm_cuda"],
            sparse_attention_backends: vec!["flashinfer_trtllm_fp8"],
            mla_cache_append_backends: vec!["vllm_cuda"],
            hadamard_quant_backends: vec!["vllm_triton"],
            pooled_index_update_backends: vec!["vllm_fork_triton"],
            index_remap_backends: vec!["vllm_triton"],
            elementwise_backends: vec!["triton"],
            mla_layout_copies: true,
            indexer_max_logits_bytes: 512 << 20,
            decode_next_n: 1,
            front: Default::default(),
        }
    }

    /// vLLM's defaults: 512 MiB of fp32 logits, `40 * 1M` workspace pools.
    fn budget() -> IndexerBudget {
        IndexerBudget {
            decode_next_n: 1,
            index_topk: 2048,
            index_kpool: 4,
            max_logits_elems: (512 << 20) / 4,
            max_keys: 40 << 20,
        }
    }

    fn work(prefill_chunk_pairs: Vec<(u32, u32)>, decode_kv_lens: Vec<u32>) -> Work {
        derive_work(
            &Glm53DsaAttnLocalWorkletInput {
                prefill_chunk_pairs,
                decode_kv_lens,
            },
            &budget(),
        )
        .unwrap()
    }

    #[test]
    fn projections_and_indexer_shapes_match_the_tp4_checkpoint() {
        let r = Glm53DsaAttnLocalWorklet::resolve_config(&cfg());
        let nk = |g: &SingleGemmKernelConfig| (g.n.get(), g.k.get());
        assert_eq!(nk(&r.fused_qkv_a), (2048, 4096));
        assert_eq!(nk(&r.q_b), (4096, 1536));
        assert_eq!(nk(&r.index_wq_b), (4096, 1536));
        assert_eq!(nk(&r.index_wk_weights), (160, 4096));
        assert_eq!(nk(&r.kpool_gate_score), (128, 4096));
        assert_eq!(nk(&r.o_proj), (4096, 4096));
        assert_eq!(
            (
                r.index_head_weights.n.get(),
                r.index_head_weights.input_dtype
            ),
            (32, DType::Fp32)
        );
        assert_eq!(r.topk_decode.top_k, 512);
        assert_eq!((r.q_absorb.n.get(), r.q_absorb.k.get()), (512, 256));
        assert_eq!((r.v_up.n.get(), r.v_up.k.get()), (256, 512));
        assert_eq!(r.sparse_mla.softmax_scale_denominator, 16);
        assert_eq!(r.expand_pools.output_bytes_per_token.get(), 4 * 2051);
    }

    /// Catches a verify graph whose indexer bills each query row as its own
    /// request (re-reading the index keys `next_n` times) or that loses rows.
    #[test]
    fn a_verify_request_scores_its_rows_in_one_indexer_group() {
        let budget = IndexerBudget {
            decode_next_n: 8,
            ..budget()
        };
        let input = Glm53DsaAttnLocalWorkletInput {
            prefill_chunk_pairs: Vec::new(),
            decode_kv_lens: vec![1000, 4001],
        };
        let w = derive_work(&input, &budget).unwrap();
        assert_eq!(
            (w.decode_requests, w.decode_rows, w.total_tokens),
            (2, 16, 16)
        );
        assert_eq!(w.decode_pools, 1001);
        // 250 + 1001 pools over two requests.
        assert_eq!(w.decode_mean_pools, 626);
        assert_eq!(
            decode_row_context_lens(&[10, 50], 4),
            [7, 8, 9, 10, 47, 48, 49, 50]
        );
        let short = Glm53DsaAttnLocalWorkletInput {
            prefill_chunk_pairs: Vec::new(),
            decode_kv_lens: vec![7],
        };
        assert!(derive_work(&short, &budget).is_err());
    }

    /// Catches a decode top-k that prices a batch past one wave at its longest
    /// row (each slot then holds one row) or at the total work over the slots
    /// (ignoring that one long row bounds the makespan).
    #[test]
    fn decode_topk_list_schedules_rows_onto_slots() {
        let row_ms = |pools: u32| f64::from(pools);
        // Two slots, two rows per request: each slot takes 10, 1, 1.
        assert_eq!(list_schedule_ms([10, 1, 1].into_iter(), 2, 2, row_ms), 12.0);
        // One long request on its own slot bounds the makespan.
        assert_eq!(
            list_schedule_ms([1, 1, 1, 20].into_iter(), 1, 2, row_ms),
            21.0
        );
        assert_eq!(list_schedule_ms(std::iter::empty(), 4, 148, row_ms), 0.0);
    }

    #[test]
    fn decode_indexer_counts_pools_and_short_prefill_skips_its_indexer() {
        let w = work(vec![(0, 2019)], vec![1000, 4001]);
        assert_eq!(
            (w.total_tokens, w.prefill_tokens, w.decode_rows),
            (2021, 2019, 2)
        );
        assert_eq!(w.decode_pools, 1001);
        assert_eq!(w.prefill_indexer_rows, 0);
        assert!(w.prefill_indexer_chunks.is_empty());
        let long = work(vec![(2000, 100)], Vec::new());
        assert_eq!(long.prefill_indexer_rows, 100);
        // Rows at positions 2000..2099 span 500..=525 pools (51225 pairs); a
        // 100-row causal tail over 562 keys holds the same count.
        assert_eq!(
            long.prefill_indexer_chunks,
            vec![IndexerChunk {
                rows: 100,
                keys: 562,
                single_request: true
            }]
        );
    }

    #[test]
    fn one_long_context_sends_every_prefill_row_through_the_indexer() {
        // 2048 rows x (250 + 65542) pools is just over 2^27 logits, so the two
        // requests are separate launches.
        let w = work(vec![(0, 1000), (261_120, 1048)], vec![500]);
        assert_eq!(w.prefill_indexer_rows, 2048);
        let [short, long] = w.prefill_indexer_chunks[..] else {
            panic!("expected two launches, got {:?}", w.prefill_indexer_chunks);
        };
        // The fresh request spans fewer pools than rows: the contract
        // `rows <= keys` lifts it to a 1000 x 1000 tail.
        assert_eq!(
            (short.rows, short.keys, short.single_request),
            (1000, 1000, true)
        );
        let pairs = (pool_prefix_sum(262_168, 4) - pool_prefix_sum(261_120, 4)) as f64;
        let (rows, keys) = (f64::from(long.rows), f64::from(long.keys));
        assert_eq!(long.rows, 1048);
        assert!((rows * keys - rows * (rows - 1.0) / 2.0 - pairs).abs() < rows);
        // A short context alone keeps the contract `rows <= keys`.
        let short = work(vec![(2048, 2048)], Vec::new());
        assert_eq!(
            short.prefill_indexer_chunks,
            vec![IndexerChunk {
                rows: 2048,
                keys: 2048,
                single_request: true
            }]
        );
    }

    #[test]
    fn a_long_request_splits_into_vllm_query_slices() {
        // vLLM at its 512 MiB default: 245,760 pools leave 2^27 / 245,760 =
        // 546 rows per call, so a 16,384-row request is 30 slices of 546 and
        // one of 4 (the 31 logits + 31 top-k launches measured on B200).
        let prefix = 245_760 * 4 - 16_384;
        let w = work(vec![(prefix, 16_384)], Vec::new());
        let chunks = &w.prefill_indexer_chunks;
        assert_eq!(chunks.len(), 31);
        assert!(chunks[..30].iter().all(|c| c.rows == 546));
        assert_eq!(chunks[30].rows, 16_384 - 30 * 546);
        assert!(chunks.iter().all(|c| c.single_request));
        // A slice's equivalent tail is its rows' mean pool span plus half its
        // rows, rounded up: 242,005 keys for the first, 245,761 for the last.
        assert_eq!(chunks[0].keys, 242_005);
        assert_eq!(chunks[30].keys, 245_761);
        // Every slice's logits fit the budget.
        assert!(chunks
            .iter()
            .all(|c| u64::from(c.rows) * 245_760 <= (512 << 20) / 4));
    }

    #[test]
    fn short_requests_pack_into_one_launch_with_row_starts() {
        // Three 1k-row requests at 8k context: 3000 rows x 6000 pools fit
        // 2^27 elements, so vLLM scores them in one call whose second and
        // third requests start past key 0 (no DeepSelect).
        let w = work(vec![(7000, 1000); 3], Vec::new());
        assert_eq!(w.prefill_indexer_chunks.len(), 1);
        let chunk = w.prefill_indexer_chunks[0];
        assert_eq!(chunk.rows, 3000);
        assert!(!chunk.single_request);
        // Per request: rows at 7000..7999 span 1750..=2000 pools.
        let pairs = 3 * (pool_prefix_sum(8000, 4) - pool_prefix_sum(7000, 4));
        let (r, k) = (3000_u64, u64::from(chunk.keys));
        assert!((r * k - r * (r - 1) / 2).abs_diff(pairs) < r);
    }

    #[test]
    fn a_pack_closes_before_the_request_that_would_overflow_it() {
        // 2000 rows over 30k pools fit 2^27 elements; two such requests do
        // not, so the first closes alone. A 3-token request (no completed
        // pool) still joins the next pack and gives it nonzero row starts.
        let w = work(vec![(118_000, 2000), (118_000, 2000), (0, 3)], Vec::new());
        let shape: Vec<(u32, bool)> = w
            .prefill_indexer_chunks
            .iter()
            .map(|c| (c.rows, c.single_request))
            .collect();
        assert_eq!(shape, vec![(2000, true), (2003, false)]);
        // Alone, a request with no completed pool is no launch at all.
        let w = work(vec![(983_616, 16_384), (0, 3)], Vec::new());
        assert_eq!(w.prefill_indexer_rows, 16_387);
        assert!(w.prefill_indexer_chunks.iter().all(|c| c.single_request));
        let rows: u32 = w.prefill_indexer_chunks.iter().map(|c| c.rows).sum();
        assert_eq!(rows, 16_384);
    }

    #[test]
    fn pool_prefix_sum_counts_completed_pools() {
        let brute = |n: u32| (1..=n).map(|x| u64::from(x / 4)).sum::<u64>();
        for n in [0, 1, 3, 4, 5, 7, 8, 9, 2047, 2048, 2049] {
            assert_eq!(pool_prefix_sum(n, 4), brute(n), "n = {n}");
        }
    }

    #[test]
    fn compile_has_fixed_slots() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let worklet = Glm53DsaAttnLocalWorklet::build(
            "m.dsa".into(),
            Glm53DsaAttnLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        assert_eq!(tree.slots.len(), 24 + 4 + 3);
        assert!(tree
            .slots
            .iter()
            .any(|slot| slot.name == "m.dsa.prefill_glue"));
        assert!(tree
            .slots
            .iter()
            .any(|slot| slot.name == "m.dsa.sparse_mla.decode"));
    }

    fn build_front(
        front: Glm53DsaFront,
        overrides: &[(&str, &str)],
    ) -> Result<Glm53DsaAttnLocalWorklet, BuildError> {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let map: HashMap<String, Vec<String>> = overrides
            .iter()
            .map(|(role, backend)| (role.to_string(), vec![backend.to_string()]))
            .collect();
        let _scope = bridge.with_backend_overrides("main", Some(&map));
        let mut config = cfg();
        config.front = front;
        Glm53DsaAttnLocalWorklet::build(
            "m.dsa".into(),
            Glm53DsaAttnLocalWorklet::resolve_config(&config),
            &bridge,
        )
    }

    fn slot_names(worklet: &Glm53DsaAttnLocalWorklet) -> Vec<String> {
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        builder
            .finish(root)
            .slots
            .iter()
            .map(|s| s.name.clone())
            .collect()
    }

    /// Catches a stage-1 merge that drops or double-counts a projection: the
    /// concatenated GEMM is the replaced projections' callable at their
    /// summed width, and the default front adds no slot.
    #[test]
    fn concatenated_front_gemms_span_the_projections_they_replace() {
        let base = build_front(Glm53DsaFront::default(), &[]).unwrap();
        assert_eq!(slot_names(&base).len(), 24 + 4 + 3);
        let front = Glm53DsaFront {
            hidden_gemm: DsaFrontGemm::Concat,
            q_lora_gemm: DsaFrontGemm::Concat,
            ..Default::default()
        };
        let w = build_front(front, &[]).unwrap();
        let r = &w.resolved;
        let hidden = r.hidden_gemm.as_ref().unwrap();
        let q_lora = r.q_lora_gemm.as_ref().unwrap();
        let n = |g: &SingleGemmKernelConfig| g.n.get();
        // The head-weight columns stay with the fp32 `head_weights` GEMM.
        assert_eq!(
            n(hidden),
            n(&r.fused_qkv_a) + n(&r.index_wk_weights) - r.index_head_weights.n.get()
                + n(&r.kpool_gate_score)
        );
        assert_eq!(hidden.k, r.fused_qkv_a.k);
        assert_eq!(n(q_lora), n(&r.q_b) + n(&r.index_wq_b));
        assert_eq!(q_lora.k, r.q_b.k);
        assert_eq!(hidden.backends, r.fused_qkv_a.backends);
        let names = slot_names(&w);
        assert_eq!(names.len(), 24 + 4 + 3 + 2);
        assert!(names.contains(&"m.dsa.hidden_gemm".to_string()));
        assert!(names.contains(&"m.dsa.q_lora_gemm".to_string()));
    }

    /// Catches a fused cut that silently prices another kernel: it builds
    /// only on its named backend, and a run-config override is rejected.
    #[test]
    fn fused_front_cuts_price_only_their_named_backend() {
        let front = Glm53DsaFront {
            hidden_gemm: DsaFrontGemm::Fused("cut_a_kernel"),
            q_lora_gemm: DsaFrontGemm::Fused("cut_b_kernel"),
            q_absorb_fp8: Some("cut_c_kernel"),
            state_writer: Some("cut_d_kernel"),
            topk_page_table: Some("cut_e_kernel"),
            ..Default::default()
        };
        let w = build_front(front, &[]).unwrap();
        // Cut E prices its own slot; the remap keeps the fork's backend for prefill rows.
        assert_eq!(w.sparse_mla.index_remap.config.backends, ["vllm_triton"]);
        assert_eq!(
            w.topk_page_table.as_ref().unwrap().kernel.config.backends,
            ["cut_e_kernel"]
        );
        assert_eq!(
            w.hidden_gemm.as_ref().unwrap().kernel.config.backends,
            ["cut_a_kernel"]
        );
        assert_eq!(
            w.q_lora_gemm.as_ref().unwrap().kernel.config.backends,
            ["cut_b_kernel"]
        );
        assert_eq!(
            w.q_absorb_fp8.as_ref().unwrap().kernel.config.backends,
            ["cut_c_kernel"]
        );
        assert_eq!(
            w.state_writer.as_ref().unwrap().kernel.config.backends,
            ["cut_d_kernel"]
        );
        // Cut A's fused width: the fp32 weight columns once, not twice.
        assert_eq!(
            w.resolved.hidden_gemm.as_ref().unwrap().n.get(),
            2048 + 128 + 32 + 128
        );
        for role in [
            "m.dsa.hidden_gemm",
            "m.dsa.q_lora_gemm",
            "m.dsa.q_absorb_fp8",
            "m.dsa.state_writer",
            "m.dsa.indexer.topk_page_table",
        ] {
            assert!(
                matches!(
                    build_front(front, &[(role, "torch_linear_vllm")]),
                    Err(BuildError::NoBackendForLaunch { .. })
                ),
                "{role}"
            );
        }
    }

    /// Catches a cut B epilogue that prices the wrong span or kind, keeps the
    /// absorbed FWHT quant and weight scale, or drifts to another backend; the
    /// existing `concat` value must keep its meaning.
    #[test]
    fn concat_hadamard_epilogue_cut_b_is_the_concat_gemm_and_one_epilogue() {
        let front = Glm53DsaFront {
            q_lora_gemm: DsaFrontGemm::ConcatHadamardEpilogue("post_kernel"),
            ..Default::default()
        };
        let w = build_front(front, &[]).unwrap();
        let r = &w.resolved;
        // The GEMM is `concat`'s: the same callable and width.
        let q_lora = r.q_lora_gemm.as_ref().unwrap();
        assert_eq!(q_lora.n.get(), r.q_b.n.get() + r.index_wq_b.n.get());
        assert_eq!(q_lora.backends, r.q_b.backends);
        let epi = r.q_lora_epilogue.as_ref().unwrap();
        assert_eq!(
            (epi.num_heads.get(), epi.head_dim.get(), epi.lead_cols.get()),
            (32, 128, r.q_b.n.get())
        );
        assert_eq!(epi.input_dtype, DType::Bf16);
        let names = slot_names(&w);
        assert_eq!(names.len(), 24 + 4 + 3 + 2);
        assert!(names.contains(&"m.dsa.q_lora_epilogue".to_string()));
        // Absorbed: the FWHT quant and the weight scale; not the q/kv norm.
        assert!(front.q_lora_gemm.absorbs_index_q_quant());
        assert!(!front.q_lora_gemm.fused());
        assert!(!DsaFrontGemm::Concat.absorbs_index_q_quant());
        let concat = build_front(
            Glm53DsaFront {
                q_lora_gemm: DsaFrontGemm::Concat,
                ..Default::default()
            },
            &[],
        )
        .unwrap();
        assert!(concat.q_lora_epilogue.is_none());
        assert!(matches!(
            build_front(front, &[("m.dsa.q_lora_epilogue", "vllm_triton")]),
            Err(BuildError::NoBackendForLaunch { .. })
        ));
    }

    /// Catches cut C priced as the bf16 `batched_gemm` (which has no FP8
    /// output) or at another width than `q_absorb`.
    #[test]
    fn q_absorb_fp8_cut_c_is_the_static_fp8_quant_kind_at_q_absorb_width() {
        let front = Glm53DsaFront {
            q_absorb_fp8: Some("tc_kernel"),
            ..Default::default()
        };
        let w = build_front(front, &[]).unwrap();
        let c = w.resolved.q_absorb_fp8.as_ref().unwrap();
        let a = &w.resolved.q_absorb;
        assert_eq!(
            (c.num_batches.get(), c.n.get(), c.k.get(), c.dtype),
            (a.num_batches.get(), a.n.get(), a.k.get(), a.dtype)
        );
        assert_eq!((c.n.get(), c.k.get()), (512, 256));
        let op = w.q_absorb_fp8.as_ref().unwrap();
        assert_eq!(
            <crate::timing::kernels::BatchedGemmStaticFp8QuantSpec as crate::timing::kernels::KernelSpec>::KIND,
            "batched_gemm_static_fp8_quant"
        );
        assert_eq!(op.kernel.config.backends, ["tc_kernel"]);
    }

    /// Catches a page-table cut that double-bills or misses a launch: cut E
    /// zeroes exactly the decode rows' post-top-k launches, cut E + top-k also
    /// the decode top-k, every absorbed slot exists in the compiled tree, and
    /// each cut adds exactly its own slot on its named backend.
    #[test]
    fn page_table_cuts_absorb_exactly_the_launches_they_replace() {
        assert!(Glm53DsaFront::default().decode_slots_absorbed(0).is_empty());
        let cut_e = Glm53DsaFront {
            topk_page_table: Some("e2a_kernel"),
            glue: DsaGlue::ForkLaunchList,
            ..Default::default()
        };
        let cut_e_topk = Glm53DsaFront {
            topk_to_page_table: Some("t6_kernel"),
            glue: DsaGlue::ForkLaunchList,
            ..Default::default()
        };
        assert_eq!(cut_e.decode_slots_absorbed(1 << 20), PAGE_TABLE_ABSORBS);
        assert_eq!(
            cut_e_topk.decode_slots_absorbed(1 << 20),
            TOPK_TO_PAGE_TABLE_ABSORBS
        );
        let mut fused_set = PAGE_TABLE_ABSORBS.to_vec();
        fused_set.insert(0, "indexer.topk_decode");
        assert_eq!(cut_e_topk.decode_slots_absorbed(1 << 20), fused_set);
        // Not absorbed: the position cast feeds the seq lens either way.
        assert!(!TOPK_TO_PAGE_TABLE_ABSORBS.contains(&"glue.positions_to_int32"));

        let base = slot_names(&build_front(Glm53DsaFront { glue: DsaGlue::ForkLaunchList, ..Default::default() }, &[]).unwrap());
        for (front, slot, backend) in [
            (cut_e, "m.dsa.indexer.topk_page_table", "e2a_kernel"),
            (cut_e_topk, "m.dsa.indexer.topk_to_page_table", "t6_kernel"),
        ] {
            let w = build_front(front, &[]).unwrap();
            let names = slot_names(&w);
            for absorbed in front.decode_slots_absorbed(1) {
                let name = format!("m.dsa.{absorbed}");
                assert!(names.contains(&name), "{name} missing from the tree");
            }
            let added: Vec<_> = names.iter().filter(|n| !base.contains(n)).collect();
            assert_eq!(added, [slot]);
            let built = match slot {
                "m.dsa.indexer.topk_page_table" => &w.topk_page_table.as_ref().unwrap().kernel.config.backends,
                _ => &w.topk_to_page_table.as_ref().unwrap().kernel.config.backends,
            };
            assert_eq!(built, &[backend]);
            assert!(matches!(
                build_front(front, &[(slot, "torch")]),
                Err(BuildError::NoBackendForLaunch { .. })
            ));
        }
        let e2a = build_front(cut_e, &[]).unwrap();
        let cfg = e2a.resolved.topk_page_table.as_ref().unwrap();
        assert_eq!(
            (cfg.top_k, cfg.kpool, cfg.page_table_width, cfg.block_size, cfg.next_n),
            (512, 4, 2176, 64, e2a.resolved.raw_cfg.decode_next_n)
        );
        let t6 = build_front(cut_e_topk, &[]).unwrap();
        let cfg = t6.resolved.topk_to_page_table.as_ref().unwrap();
        assert_eq!(cfg.logits_row_stride.get() * 4, cfg.max_model_len.get());
        assert_eq!(cfg.context_mode, "uniform_measured");
    }

    /// Both page-table cuts at once would bill the decode page table twice,
    /// unless a pool threshold dispatches each launch to one of them; a
    /// threshold needs both cuts.
    #[test]
    fn page_table_cuts_are_exclusive() {
        let both = Glm53DsaFront {
            topk_page_table: Some("e2a_kernel"),
            topk_to_page_table: Some("t6_kernel"),
            ..Default::default()
        };
        assert!(matches!(
            build_front(both, &[]),
            Err(BuildError::FitFailed { .. })
        ));
        for front in [
            Glm53DsaFront { topk_to_page_table_max_pools: Some(0), ..both },
            Glm53DsaFront { topk_page_table: None, topk_to_page_table_max_pools: Some(65_536), ..both },
            Glm53DsaFront { topk_to_page_table: None, topk_to_page_table_max_pools: Some(65_536), ..both },
        ] {
            assert!(matches!(build_front(front, &[]), Err(BuildError::FitFailed { .. })), "{front:?}");
        }
    }

    /// Catches a context dispatch that sends a launch to both cuts, to
    /// neither, or by the wrong row: the fused top-k takes launches whose
    /// longest row is within the threshold (inclusive); a longer launch runs
    /// the decode top-k and cut E, which then bills its own slot and zeroes
    /// only the post-top-k launches.
    #[test]
    fn page_table_dispatch_picks_one_cut_by_the_longest_row() {
        let dispatch = Glm53DsaFront {
            topk_page_table: Some("e2a_kernel"),
            topk_to_page_table: Some("t6_kernel"),
            topk_to_page_table_max_pools: Some(65_536),
            glue: DsaGlue::ForkLaunchList,
            ..Default::default()
        };
        use DecodePageTableCut::{PageTable, TopkToPageTable};
        for (pools, cut) in [
            (1, TopkToPageTable),
            (59_000, TopkToPageTable),
            (65_536, TopkToPageTable),
            (65_537, PageTable),
            (262_144, PageTable),
        ] {
            assert_eq!(dispatch.decode_page_table_cut(pools), cut, "{pools} pools");
        }
        assert_eq!(dispatch.decode_slots_absorbed(65_536), TOPK_TO_PAGE_TABLE_ABSORBS);
        assert_eq!(dispatch.decode_slots_absorbed(65_537), PAGE_TABLE_ABSORBS);
        assert!(!dispatch.decode_slots_absorbed(65_537).contains(&"indexer.topk_decode"));
        // Without a threshold a single cut takes every launch.
        let t6_only = Glm53DsaFront { topk_page_table: None, topk_to_page_table_max_pools: None, ..dispatch };
        assert_eq!(t6_only.decode_page_table_cut(1 << 20), TopkToPageTable);
        // Both cuts are built, each on its own backend, and nothing else is added.
        let w = build_front(dispatch, &[]).unwrap();
        let base = slot_names(&build_front(Glm53DsaFront { glue: DsaGlue::ForkLaunchList, ..Default::default() }, &[]).unwrap());
        let added: Vec<_> = slot_names(&w).into_iter().filter(|n| !base.contains(n)).collect();
        assert_eq!(added, ["m.dsa.indexer.topk_page_table", "m.dsa.indexer.topk_to_page_table"]);
        assert_eq!(w.topk_page_table.as_ref().unwrap().kernel.config.backends, ["e2a_kernel"]);
        assert_eq!(w.topk_to_page_table.as_ref().unwrap().kernel.config.backends, ["t6_kernel"]);
    }

    /// Catches an fp32-split cut A that prices the wrong width or split, or
    /// lets its epilogue drift to another backend.
    #[test]
    fn fp32_split_cut_a_is_one_fp32_gemm_and_one_epilogue() {
        let front = Glm53DsaFront {
            hidden_gemm: DsaFrontGemm::Fp32Split("split_kernel"),
            ..Default::default()
        };
        let w = build_front(front, &[]).unwrap();
        assert!(w.hidden_gemm.is_none());
        let gemm = w.resolved.hidden_gemm_fp32.as_ref().unwrap();
        assert_eq!((gemm.n.get(), gemm.input_dtype), (2048 + 128 + 128 + 32, DType::Bf16));
        let split = w.resolved.hidden_row_split.as_ref().unwrap();
        assert_eq!(
            (
                split.bf16_lead_cols.get(),
                split.norm_cols.get(),
                split.bf16_trail_cols.get(),
                split.fp32_tail_cols.get()
            ),
            (2048, 128, 128, 32)
        );
        let names = slot_names(&w);
        assert!(names.contains(&"m.dsa.hidden_gemm".to_string()));
        assert!(names.contains(&"m.dsa.hidden_row_split".to_string()));
        assert!(matches!(
            build_front(front, &[("m.dsa.hidden_row_split", "torch_vllm")]),
            Err(BuildError::NoBackendForLaunch { .. })
        ));
    }

    /// Each slot's logged input after one eval (structure-only kernels
    /// answer zero time, so the logged sizes are what a test can read).
    fn eval_inputs(
        w: &Glm53DsaAttnLocalWorklet,
        input: &Glm53DsaAttnLocalWorkletInput,
    ) -> HashMap<String, serde_json::Value> {
        let names = slot_names(w);
        let mut metrics = vec![LeafMetrics::MISS; names.len()];
        let mut inputs = Vec::new();
        let mut ev = Evaluator::with_inputs(&mut metrics, &mut inputs);
        w.eval(input, &mut ev);
        assert_eq!(ev.filled(), names.len());
        names
            .into_iter()
            .zip(inputs)
            .map(|(name, input)| (name, serde_json::to_value(input).unwrap()))
            .collect()
    }

    /// Catches the split state writer double-billing or dropping a launch:
    /// on decode rows it alone prices the q/kv norm, cache append, kpool
    /// position cast and kpool update (and the row split); prefill rows keep
    /// the incumbent slots; the slot runs on its own backend only.
    #[test]
    fn split_state_writer_absorbs_exactly_the_decode_state_launches() {
        let split = Glm53DsaFront {
            hidden_gemm: DsaFrontGemm::Fp32Split("triton"),
            glue: DsaGlue::ForkLaunchList,
            ..Default::default()
        };
        let writer = Glm53DsaFront {
            hidden_gemm: DsaFrontGemm::Fp32SplitStateWriter("triton"),
            ..split
        };
        let base = build_front(split, &[]).unwrap();
        let w = build_front(writer, &[]).unwrap();
        let names = slot_names(&w);
        for absorbed in SPLIT_STATE_WRITER_ABSORBS {
            assert!(names.contains(&format!("m.dsa.{absorbed}")), "{absorbed} left the tree");
        }
        let base_names = slot_names(&base);
        let added: Vec<_> = names.iter().filter(|n| !base_names.contains(n)).collect();
        assert_eq!(added, ["m.dsa.hidden_state_writer"]);
        let cfg = w.resolved.hidden_state_writer.as_ref().unwrap();
        let kpool = &w.resolved.kpool_decode_update;
        assert_eq!(
            (cfg.q_lora_cols.get(), cfg.kv_lora_cols.get(), cfg.index_dim.get(), cfg.fp32_tail_cols.get()),
            (1536, 512, 128, 32)
        );
        assert_eq!(
            (cfg.next_n, cfg.pool_size, cfg.tail_ring, cfg.block_size),
            (kpool.next_n, kpool.pool_size, kpool.tail_ring, kpool.block_size)
        );
        // One position cast (G5) left; G3 belongs to the state writer.
        assert_eq!((base.position_casts(), w.position_casts()), (2, 1));

        let next_n = u64::from(w.resolved.raw_cfg.decode_next_n);
        let decode = Glm53DsaAttnLocalWorkletInput {
            prefill_chunk_pairs: Vec::new(),
            decode_kv_lens: vec![4000, 5000],
        };
        let mixed = Glm53DsaAttnLocalWorkletInput {
            prefill_chunk_pairs: vec![(0, 5)],
            ..decode.clone()
        };
        let size = |m: &HashMap<String, serde_json::Value>, s: &str| {
            let v = &m[&format!("m.dsa.{s}")];
            ["num_tokens", "num_requests", "m"]
                .iter()
                .find_map(|k| v.get(*k).and_then(|x| x.as_u64()))
                .unwrap_or_else(|| panic!("{s}: {v}"))
        };
        let (d, b) = (eval_inputs(&w, &decode), eval_inputs(&base, &decode));
        let (m, bm) = (eval_inputs(&w, &mixed), eval_inputs(&base, &mixed));
        // (slot, billed with the writer on decode / mixed, without it on decode / mixed)
        for (s, writer_decode, writer_mixed, base_decode, base_mixed) in [
            ("q_kv_a_norm", 0, 5, 2 * next_n, 5 + 2 * next_n),
            ("sparse_mla.mla_cache_append", 0, 5, 2 * next_n, 5 + 2 * next_n),
            ("indexer.kpool_decode_update", 0, 0, 2, 2),
            ("hidden_row_split", 0, 5, 2 * next_n, 5 + 2 * next_n),
            // Decode rows either way; the scale (casts per step) drops to 1.
            ("glue.positions_to_int32", 2 * next_n, 2 * next_n, 2 * next_n, 2 * next_n),
            ("hidden_gemm", 2 * next_n, 5 + 2 * next_n, 2 * next_n, 5 + 2 * next_n),
            ("hidden_state_writer", 2, 2, 2, 2),
        ] {
            assert_eq!((size(&d, s), size(&m, s)), (writer_decode, writer_mixed), "{s} with the writer");
            if s != "hidden_state_writer" {
                assert_eq!((size(&b, s), size(&bm, s)), (base_decode, base_mixed), "{s} without it");
            }
        }
        assert!(matches!(
            build_front(writer, &[("m.dsa.hidden_state_writer", "torch")]),
            Err(BuildError::NoBackendForLaunch { .. })
        ));
    }

    /// Cut D and the split state writer write the same state.
    #[test]
    fn split_state_writer_and_cut_d_are_exclusive() {
        let both = Glm53DsaFront {
            hidden_gemm: DsaFrontGemm::Fp32SplitStateWriter("triton"),
            state_writer: Some("cut_d_kernel"),
            ..Default::default()
        };
        assert!(matches!(build_front(both, &[]), Err(BuildError::FitFailed { .. })));
    }

    /// Catches a launch-list fix that loses a launch or keeps the old
    /// placeholder: seven fork launches (the two position casts folded).
    #[test]
    fn fork_glue_lists_the_seven_decode_launches() {
        let front = Glm53DsaFront {
            glue: DsaGlue::ForkLaunchList,
            ..Default::default()
        };
        let w = build_front(front, &[]).unwrap();
        let mut builder = CostTreeBuilder::new();
        let root = w.compile(&mut builder);
        let tree = builder.finish(root);
        let glue: Vec<_> = tree
            .slots
            .iter()
            .filter(|s| s.name.starts_with("m.dsa.glue."))
            .collect();
        assert_eq!(glue.len(), 6);
        let g = w.resolved.fork_glue.as_ref().unwrap();
        assert_eq!(g.topk_buffer_fill.input_bytes_per_token, 0);
        assert_eq!(g.topk_buffer_copy.output_bytes_per_token, 4 * 2051);
    }

    #[test]
    fn decode_topk_takes_the_cooperative_kernel_only_ahead_of_a_long_row_kernel() {
        // The fork's "auto" decode top-k runs cooperative_topk up to 64 rows,
        // also with DeepSelect for the longer batches; torch or a pre-SM90 GPU
        // never does.
        let build = |topk: Option<Vec<&str>>, gpu: &str| {
            let bridge = PerfApiBridge::new_uninit_for_test();
            bridge.enable_enumerate();
            let map: HashMap<String, Vec<String>> = topk
                .into_iter()
                .map(|backends| {
                    let backends = backends.iter().map(|b| b.to_string()).collect();
                    ("m.dsa.indexer.topk_decode".to_string(), backends)
                })
                .collect();
            let _scope = bridge.with_backend_overrides("main", Some(&map));
            let mut config = cfg();
            config.gpu_name = gpu.into();
            Glm53DsaAttnLocalWorklet::build(
                "m.dsa".into(),
                Glm53DsaAttnLocalWorklet::resolve_config(&config),
                &bridge,
            )
            .unwrap()
        };
        let worklet = build(None, "NVIDIA B200");
        let cooperative = worklet.topk_decode_cooperative.as_ref().unwrap();
        assert_eq!(cooperative.config.backends, vec!["vllm_cuda_cooperative"]);
        assert_eq!(cooperative.config.context_mode, "uniform_measured");
        assert!(build(Some(vec!["deep_select"]), "NVIDIA B200")
            .topk_decode_cooperative
            .is_some());
        assert!(build(Some(vec!["torch"]), "NVIDIA B200")
            .topk_decode_cooperative
            .is_none());
        assert!(build(None, "A100-SXM4-80GB")
            .topk_decode_cooperative
            .is_none());
        // Its time lands in the topk_decode slot: no extra leaf.
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        assert_eq!(builder.finish(root).slots.len(), 24 + 4 + 3);
    }

    #[test]
    fn packed_launches_run_on_the_backends_that_take_row_starts() {
        // DeepSelect takes no row starts, so a pack of several requests prices
        // its top-k on the other candidates, inside the same slot. The
        // candidates are the backends as built, after any override.
        let build = |overrides: Option<Vec<&str>>| {
            let bridge = PerfApiBridge::new_uninit_for_test();
            bridge.enable_enumerate();
            let map: HashMap<String, Vec<String>> = overrides
                .into_iter()
                .map(|backends| {
                    let backends = backends.iter().map(|b| b.to_string()).collect();
                    ("m.dsa.indexer.topk_prefill".to_string(), backends)
                })
                .collect();
            let _scope = bridge.with_backend_overrides("main", Some(&map));
            Glm53DsaAttnLocalWorklet::build(
                "m.dsa".into(),
                Glm53DsaAttnLocalWorklet::resolve_config(&cfg()),
                &bridge,
            )
        };
        assert!(build(None).unwrap().topk_prefill_packed.is_none());
        let worklet = build(Some(vec!["deep_select", "vllm_cuda"])).unwrap();
        let packed = worklet.topk_prefill_packed.as_ref().unwrap();
        assert_eq!(packed.config.backends, vec!["vllm_cuda"]);
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        assert_eq!(builder.finish(root).slots.len(), 24 + 4 + 3);
        assert!(matches!(
            build(Some(vec!["deep_select"])),
            Err(BuildError::NoBackendForLaunch { .. })
        ));
    }
}
