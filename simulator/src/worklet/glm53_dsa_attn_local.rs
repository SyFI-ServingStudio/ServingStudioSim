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
//!   (decode rows), and the pool-to-token expansion. The indexer replicates its
//!   32 heads on every rank.
//! * MLA: `W_UK` absorb, the FP8 query quant, the kpool sparse-MLA compound op
//!   (cache append, remap, token-sparse attention for prefill and decode rows),
//!   and `W_UV`.
//!
//! Byte-sized elementwise placeholders stand in for the small indexer kernels
//! (b2-dsa section 7 byte formulas), the prefill MQA logits / top-k, which vLLM
//! skips while every prefill context fits `index_topk` tokens, and the
//! framework glue (index plumbing, the query concat, the output copy). The
//! indexer k LayerNorm and the weights x q-scale multiply are Inductor-fused
//! elementwise kernels with their own leaves. Copies that run far below
//! streaming bandwidth (the transposed query concat, the output masked fill,
//! the Hadamard quant) carry an effective-bandwidth byte factor anchored to
//! the measured T = 2048 kernels, and a prefill-bearing iteration adds its
//! `[rows, selected_k]` index plumbing (`prefill_glue`).
//!
//! A decode batch's indexer and attention shapes collapse to the batch's MAX
//! context (the measured decode grids are uniform in context).

use crate::op::attention::{
    Glm53KpoolSparseMlaConfig, Glm53KpoolSparseMlaInput, Glm53KpoolSparseMlaOp,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    BatchedGemmKernel, BatchedGemmKernelConfig, BatchedGemmKernelInput,
    DeepseekV4FusedQKvRmsnormKernel, DeepseekV4FusedQKvRmsnormKernelConfig,
    DeepseekV4FusedQKvRmsnormKernelInput, DsaPagedMqaLogitsDecodeKernel,
    DsaPagedMqaLogitsDecodeKernelConfig, DsaPagedMqaLogitsDecodeKernelInput,
    DsaPersistentTopkDecodeKernel, DsaPersistentTopkDecodeKernelConfig,
    DsaPersistentTopkDecodeKernelInput, ElementwiseKernel, ElementwiseKernelConfig,
    ElementwiseKernelInput, GemmFp32OutputKernel, GemmFp32OutputKernelConfig,
    GemmFp32OutputKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

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
/// * `q_fwht_quant`: the Hadamard-128 + ue8m0 quant is not a streaming kernel;
///   16.8 us at T = 2048 for 25 MiB of real traffic. The factor matches the
///   prefill-bearing time; decode (5.5 us at T = 32) stays latency-bound above
///   any elementwise row and needs a dedicated L1 kind to match.
const FWHT_EFFECTIVE_FACTOR: u32 = 5;
/// `k[idx]` and `gate_score[idx]` gathers ahead of the prefill kpool write.
const PREFILL_GATHER_LAUNCHES: u32 = 2;

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
    pub sparse_attention_backends: Vec<&'static str>,
    pub mla_cache_append_backends: Vec<&'static str>,
    pub index_remap_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Glm53DsaAttnLocalWorkletResolved {
    pub raw_cfg: Glm53DsaAttnLocalWorkletConfig,
    pub fused_qkv_a: SingleGemmKernelConfig,
    pub q_kv_norm: DeepseekV4FusedQKvRmsnormKernelConfig,
    pub q_b: SingleGemmKernelConfig,
    pub index_wq_b: SingleGemmKernelConfig,
    pub index_wk_weights: SingleGemmKernelConfig,
    pub index_head_weights: GemmFp32OutputKernelConfig,
    pub index_k_norm: ElementwiseKernelConfig,
    pub index_q_fwht_quant: ElementwiseKernelConfig,
    pub index_weight_scale: ElementwiseKernelConfig,
    pub kpool_gate_score: SingleGemmKernelConfig,
    pub prefill_gather: ElementwiseKernelConfig,
    pub kpool_decode_update: ElementwiseKernelConfig,
    pub kpool_prefill_write: ElementwiseKernelConfig,
    pub kpool_tail_seed: ElementwiseKernelConfig,
    pub mqa_logits_decode: DsaPagedMqaLogitsDecodeKernelConfig,
    pub topk_decode: DsaPersistentTopkDecodeKernelConfig,
    pub mqa_logits_prefill: ElementwiseKernelConfig,
    pub topk_prefill: ElementwiseKernelConfig,
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
}

/// One iteration's DSA work on this rank.
#[derive(Clone, Debug, Default)]
pub struct Glm53DsaAttnLocalWorkletInput {
    /// `(prefix, append)` per prefill request.
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
    /// KV length per decode request, including the token being decoded.
    pub decode_kv_lens: Vec<u32>,
}

pub struct Glm53DsaAttnLocalWorklet {
    pub name: String,
    pub fused_qkv_a: Op<SingleGemmKernel>,
    pub q_kv_norm: Op<DeepseekV4FusedQKvRmsnormKernel>,
    pub q_b: Op<SingleGemmKernel>,
    pub index_wq_b: Op<SingleGemmKernel>,
    pub index_wk_weights: Op<SingleGemmKernel>,
    pub index_head_weights: Op<GemmFp32OutputKernel>,
    pub index_k_norm: Op<ElementwiseKernel>,
    pub index_q_fwht_quant: Op<ElementwiseKernel>,
    pub index_weight_scale: Op<ElementwiseKernel>,
    pub kpool_gate_score: Op<SingleGemmKernel>,
    pub prefill_gather: Op<ElementwiseKernel>,
    pub kpool_decode_update: Op<ElementwiseKernel>,
    pub kpool_prefill_write: Op<ElementwiseKernel>,
    pub kpool_tail_seed: Op<ElementwiseKernel>,
    pub mqa_logits_decode: Op<DsaPagedMqaLogitsDecodeKernel>,
    pub topk_decode: Op<DsaPersistentTopkDecodeKernel>,
    pub mqa_logits_prefill: Op<ElementwiseKernel>,
    pub topk_prefill: Op<ElementwiseKernel>,
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
    resolved: Glm53DsaAttnLocalWorkletResolved,
}

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
        let pooled_row_bytes = cfg.max_model_len * DType::Fp32.size_bytes();
        let window = cfg.index_topk + kpool - 1;
        Glm53DsaAttnLocalWorkletResolved {
            fused_qkv_a: gemm(cfg.q_lora_rank.get() + latent, cfg.hidden.clone()),
            q_kv_norm: DeepseekV4FusedQKvRmsnormKernelConfig {
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
            // b2-dsa 7: reads 2*D*Hi, writes (D+4)*Hi per token.
            index_q_fwht_quant: ew(
                FWHT_EFFECTIVE_FACTOR * 2 * index_dim * index_heads,
                FWHT_EFFECTIVE_FACTOR * (index_dim + 4) * index_heads,
            ),
            // `weights * q_scale * scale`: two fp32 Hi rows in, one out.
            index_weight_scale: ew(2 * 4 * index_heads, 4 * index_heads),
            kpool_gate_score: gemm(index_dim, cfg.hidden.clone()),
            // 2*P*2D bytes gathered per prefill token, split over two launches.
            prefill_gather: ew(kpool * 2 * index_dim, kpool * 2 * index_dim),
            // Per decode row: 524 read + 1536 per completed pool (1 in P rows),
            // 512 stashed + (D+4) per completed pool.
            kpool_decode_update: ew(524 + 1536 / kpool, 512 + (index_dim + 4) / kpool),
            // Per prefill token: 9 + P*2*2D/P read, (D+4)/P written.
            kpool_prefill_write: ew(9 + 4 * index_dim, (index_dim + 4) / kpool),
            // Reads each prefill token's position; the < P tail rows are noise.
            kpool_tail_seed: ew(8, 4),
            mqa_logits_decode: DsaPagedMqaLogitsDecodeKernelConfig {
                backends: cfg.mqa_logits_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                next_n: 1,
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
                next_n: 1,
                max_model_len: max_model_len.clone(),
                top_k: cfg.index_topk / kpool,
                logits_row_stride: max_model_len,
                logits_dtype: DType::Fp32,
                index_dtype: "int32".into(),
                context_mode: "uniform".into(),
            },
            // FP8 q (+ scale) in, one token-wide fp32 logits row out.
            mqa_logits_prefill: ew(index_heads * (index_dim + 4), pooled_row_bytes),
            topk_prefill: ew(pooled_row_bytes, cfg.index_topk / kpool * 4),
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
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Glm53DsaAttnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        let ew = |suffix: &str, config| atomic(n, suffix, config, ElementwiseKernel::build, bridge);
        let gemm =
            |suffix: &str, config| atomic(n, suffix, config, SingleGemmKernel::build, bridge);
        Ok(Self {
            fused_qkv_a: gemm("fused_qkv_a", r.fused_qkv_a)?,
            q_kv_norm: atomic(
                n,
                "q_kv_a_norm",
                r.q_kv_norm,
                DeepseekV4FusedQKvRmsnormKernel::build,
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
            index_q_fwht_quant: ew("indexer.q_fwht_quant", r.index_q_fwht_quant)?,
            index_weight_scale: ew("indexer.weight_scale", r.index_weight_scale)?,
            kpool_gate_score: gemm("indexer.kpool_gate_score", r.kpool_gate_score)?,
            prefill_gather: ew("indexer.prefill_gather", r.prefill_gather)?,
            kpool_decode_update: ew("indexer.kpool_decode_update", r.kpool_decode_update)?,
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
            mqa_logits_prefill: ew("indexer.mqa_logits_prefill", r.mqa_logits_prefill)?,
            topk_prefill: ew("indexer.topk_prefill", r.topk_prefill)?,
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
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (Glm53DsaAttnLocalWorklet) [TP rank; H={}, index heads={} replicated, \
                 kpool top-{} x {}]",
                self.name, cfg.num_heads, cfg.index_num_heads, cfg.index_topk, cfg.index_kpool
            ),
            child: Box::new(CostNode::Sum(vec![
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
            ])),
        }
    }

    pub fn eval(&self, input: &Glm53DsaAttnLocalWorkletInput, ev: &mut Evaluator) {
        let cfg = &self.resolved.raw_cfg;
        let w = derive_work(input, cfg.index_topk, cfg.index_kpool)
            .unwrap_or_else(|reason| panic!("invalid Glm53DsaAttnLocalWorkletInput: {reason}"));
        let rows = SingleGemmKernelInput { m: w.total_tokens };
        let tokens = |num_tokens| ElementwiseKernelInput { num_tokens };
        let all = tokens(w.total_tokens);
        let no_prefill = w.prefill_tokens == 0;
        let no_decode = w.decode_rows == 0;
        push_or_zero(&self.fused_qkv_a, rows.clone(), false, ev);
        push_or_zero(
            &self.q_kv_norm,
            DeepseekV4FusedQKvRmsnormKernelInput {
                num_tokens: w.total_tokens,
            },
            false,
            ev,
        );
        push_or_zero(&self.q_b, rows.clone(), false, ev);
        push_or_zero(&self.index_wq_b, rows.clone(), false, ev);
        push_or_zero(&self.index_wk_weights, rows.clone(), false, ev);
        push_or_zero(
            &self.index_head_weights,
            GemmFp32OutputKernelInput { m: w.total_tokens },
            false,
            ev,
        );
        push_or_zero(&self.index_k_norm, all.clone(), false, ev);
        push_or_zero(&self.index_q_fwht_quant, all.clone(), false, ev);
        push_or_zero(&self.index_weight_scale, all.clone(), false, ev);
        push_or_zero(&self.kpool_gate_score, rows.clone(), false, ev);
        push_or_zero(
            &self.prefill_gather,
            tokens(w.prefill_tokens),
            no_prefill,
            ev,
        );
        push_or_zero(
            &self.kpool_decode_update,
            tokens(w.decode_rows),
            no_decode,
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
                batch_size: w.decode_rows,
                context_len: w.decode_pools,
            },
            no_decode,
            ev,
        );
        push_or_zero(
            &self.topk_decode,
            DsaPersistentTopkDecodeKernelInput {
                batch_size: w.decode_rows,
                context_len: w.decode_pools,
            },
            no_decode,
            ev,
        );
        let skip_prefill_indexer = w.prefill_indexer_rows == 0;
        push_or_zero(
            &self.mqa_logits_prefill,
            tokens(w.prefill_indexer_rows),
            skip_prefill_indexer,
            ev,
        );
        push_or_zero(
            &self.topk_prefill,
            tokens(w.prefill_indexer_rows),
            skip_prefill_indexer,
            ev,
        );
        let expand_rows = w.decode_rows + w.prefill_indexer_rows;
        push_or_zero(
            &self.expand_pools,
            tokens(expand_rows),
            expand_rows == 0,
            ev,
        );
        push_or_zero(
            &self.q_absorb,
            BatchedGemmKernelInput { m: w.total_tokens },
            false,
            ev,
        );
        push_or_zero(&self.q_concat, all.clone(), false, ev);
        push_or_zero(&self.q_fp8_quant, all.clone(), false, ev);
        self.sparse_mla.eval(
            &Glm53KpoolSparseMlaInput {
                prefill_query_cache_pairs: input
                    .prefill_chunk_pairs
                    .iter()
                    .map(|&(prefix, append)| (append, prefix + append))
                    .collect(),
                decode_context_lens: input.decode_kv_lens.clone(),
            },
            ev,
        );
        push_or_zero(&self.output_copy, all.clone(), false, ev);
        push_or_zero(
            &self.v_up,
            BatchedGemmKernelInput { m: w.total_tokens },
            false,
            ev,
        );
        push_or_zero(&self.o_proj, rows, false, ev);
        push_or_zero(&self.glue, all.clone(), false, ev);
        push_or_zero(&self.prefill_glue, all, no_prefill, ev);
    }
}

struct Work {
    total_tokens: u32,
    prefill_tokens: u32,
    decode_rows: u32,
    /// `ceil(max decode context / index_kpool)`.
    decode_pools: u32,
    /// Prefill rows of requests whose context exceeds `index_topk`; vLLM skips
    /// the prefill indexer when every prefill context fits.
    prefill_indexer_rows: u32,
}

fn derive_work(
    input: &Glm53DsaAttnLocalWorkletInput,
    index_topk: u32,
    index_kpool: u32,
) -> Result<Work, String> {
    let mut prefill_tokens = 0_u32;
    let mut prefill_indexer_rows = 0_u32;
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
        if context > index_topk {
            prefill_indexer_rows += append;
        }
    }
    let decode_rows = input.decode_kv_lens.len() as u32;
    if input.decode_kv_lens.contains(&0) {
        return Err("decode KV lengths must be positive".into());
    }
    let max_decode = input.decode_kv_lens.iter().copied().max().unwrap_or(0);
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
        decode_pools: max_decode.div_ceil(index_kpool),
        prefill_indexer_rows,
    })
}

fn integer_sqrt(value: u32) -> u32 {
    let root = f64::from(value).sqrt().round() as u32;
    assert_eq!(root * root, value, "softmax scale needs a square head dim");
    root
}

#[cfg(test)]
mod tests {
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
            fp32_gemm_backends: vec!["torch_cublas_vllm_fork"],
            qkv_norm_backends: vec!["vllm_fork_triton"],
            mla_bmm_q_absorb_backends: vec!["torch_mla_q_absorb_glm53"],
            mla_bmm_v_up_backends: vec!["torch_mla_v_up_glm53"],
            mqa_logits_backends: vec!["deepgemm_fp8_vllm_fork"],
            topk_backends: vec!["vllm_fork_cuda"],
            sparse_attention_backends: vec!["flashinfer_trtllm_fp8_vllm_fork"],
            mla_cache_append_backends: vec!["vllm_cuda"],
            index_remap_backends: vec!["vllm_fork_triton"],
            elementwise_backends: vec!["triton"],
        }
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

    #[test]
    fn decode_indexer_counts_pools_and_short_prefill_skips_its_indexer() {
        let w = derive_work(
            &Glm53DsaAttnLocalWorkletInput {
                prefill_chunk_pairs: vec![(0, 2019)],
                decode_kv_lens: vec![1000, 4001],
            },
            2048,
            4,
        )
        .unwrap();
        assert_eq!(
            (w.total_tokens, w.prefill_tokens, w.decode_rows),
            (2021, 2019, 2)
        );
        assert_eq!(w.decode_pools, 1001);
        assert_eq!(w.prefill_indexer_rows, 0);
        let long = derive_work(
            &Glm53DsaAttnLocalWorkletInput {
                prefill_chunk_pairs: vec![(2000, 100)],
                decode_kv_lens: Vec::new(),
            },
            2048,
            4,
        )
        .unwrap();
        assert_eq!(long.prefill_indexer_rows, 100);
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
        assert!(tree.slots.iter().any(|slot| slot.name == "m.dsa.prefill_glue"));
        assert!(tree
            .slots
            .iter()
            .any(|slot| slot.name == "m.dsa.sparse_mla.decode"));
    }
}
