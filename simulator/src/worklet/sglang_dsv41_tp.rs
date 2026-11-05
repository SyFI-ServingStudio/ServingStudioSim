//! DeepSeek-V4.1-Flash body layers as SGLang v0.5.23 runs them under TP4
//! attention + EP4 experts on SM12x (RTX PRO 6000, PCIe, `--attention-backend
//! dsv4`, `--moe-runner-backend flashinfer_mxfp4`).
//!
//! Op order is the torch-profiler trace of the 2026-11-05 run (decode batch 1,
//! the kernels between two all-reduces; `doc/pr/dsv41-sglang-sm120/DESIGN.md`).
//! Per layer:
//!
//! - **Engram block** (layers 1 and 14): host-table gather, `wkv` FP8 GEMM,
//!   fused gate.
//! - **Attention sublayer**: mHC mix-stats + sinkhorn (a side stream during
//!   decode, `deepseek_v4_mhc.py` `use_stats_stream`), mHC combine + RMSNorm,
//!   fused q-a/kv-a FP8 GEMM, norms and RoPE, q-b FP8 GEMM, compressor (KV
//!   sources), indexer (index sources: projections, ~20 torch bookkeeping
//!   launches, FP4 logits, ~30 torch top-k launches), paged sparse MLA,
//!   inverse RoPE, `wo_a` (grouped), `wo_b`, NCCL all-reduce (unfused on
//!   SM12x), mHC post.
//! - **FFN sublayer**: mHC mix-stats + combine, router, TP-sharded shared
//!   expert (FP8), routed MXFP4 experts (FlashInfer CUTLASS, MXFP8 act quant
//!   inside the row), NCCL all-reduce (shared partial joins it,
//!   `reduce_results=False`), mHC post.
//!
//! Attention, indexer scoring and top-k run once per forward over every
//! request, so a call is priced as one uniform-geometry call per phase:
//! decode rows at the batch's mean context, prefill chunks at their mean
//! length and mean prefix. SGLang's separate-prefill-priority scheduler keeps
//! the two phases in different iterations, so a mixed call is rare.

use std::sync::Arc;

use super::deepseek_v41_common::{eval_or_zero, placeholder};
use crate::common::Fabric;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::engine::{Kernel, KernelSpec};
use crate::timing::kernels::{
    AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput, ElementwiseKernel,
    ElementwiseKernelInput, EngramGateKernel, EngramLookupKernel, EngramLookupKernelConfig,
    EngramLookupKernelInput, GemmFp32OutputKernel, GemmFp32OutputKernelConfig,
    GemmFp32OutputKernelInput, MhcCombineRmsNormKernel, MhcMixStatsSinkhornKernel,
    MhcMixStatsSinkhornKernelConfig, MhcPostKernel, MhcRmsNormKernelConfig, MhcRmsNormKernelInput,
    Nvfp4FusedMoeKernel, Nvfp4FusedMoeKernelConfig, Nvfp4FusedMoeKernelInput, PagedSparseMlaKernel,
    PagedSparseMlaKernelConfig, PagedSparseMlaKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput, SparseIndexerLogitsDecodeKernel,
    SparseIndexerLogitsDecodeKernelConfig, SparseIndexerLogitsDecodeKernelInput,
    SparseIndexerLogitsPrefillKernel, SparseIndexerLogitsPrefillKernelConfig,
    SparseIndexerLogitsPrefillKernelInput,
};
use crate::timing::kernels::{
    ElementwiseSpec, EngramGateSpec, EngramLookupSpec, GemmFp32OutputSpec, SingleGemmSpec,
    SparseIndexerLogitsDecodeSpec, SparseIndexerLogitsPrefillSpec,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

/// Torch launches the dense-blocks indexer issues per index layer before
/// scoring (position/length bookkeeping, gathers, casts): 20 in the decode
/// trace. Priced as this many byte-floor elementwise launches.
pub const INDEXER_BOOKKEEPING_LAUNCHES: u32 = 20;
/// Torch top-k launches per index layer (`mbtopk` passes, scans, radix sort,
/// gather): about 30 in the decode trace.
pub const INDEXER_TOPK_LAUNCHES: u32 = 30;
/// SGLang KV page size (`server_info.page_size`, V4 layout on SM12x).
pub const PAGE_SIZE: u32 = 256;

/// The kernel family a deployment's GPU takes. Only SM12x is captured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SglangDsv41KernelPath {
    /// RTX PRO 6000 / RTX 50 class: Triton block-32 FP8 linears, FlashInfer
    /// `sparse_mla_sm120`, CUTLASS MXFP8 x MXFP4 experts, NCCL over PCIe.
    Sm120,
}

impl SglangDsv41KernelPath {
    pub fn of(gpu_name: &str) -> anyhow::Result<Self> {
        match crate::common::gpu::compute_capability(gpu_name) {
            Some((12, _)) => Ok(Self::Sm120),
            other => anyhow::bail!(
                "deepseek_v41_sglang is captured on SM12x only; {gpu_name} has compute capability \
                 {other:?}"
            ),
        }
    }

    pub(crate) fn fp8_gemm(self) -> &'static str {
        "sglang_triton_block32_fp8"
    }
    pub(crate) fn bf16_gemm(self) -> &'static str {
        "torch_linear"
    }
    pub(crate) fn fp32_out_gemm(self) -> &'static str {
        "torch_cublas"
    }
    pub(crate) fn mhc(self) -> &'static str {
        "sglang_triton"
    }
    pub(crate) fn mhc_post(self) -> &'static str {
        "sglang_auto"
    }
    pub(crate) fn attention(self) -> &'static str {
        "sglang_flashinfer_sm120"
    }
    pub(crate) fn index_decode(self) -> &'static str {
        "sglang_triton_fp4"
    }
    pub(crate) fn index_prefill(self) -> &'static str {
        "sglang_torch_fp4"
    }
    pub(crate) fn moe(self) -> &'static str {
        "flashinfer_cutlass_sm120_mxfp4"
    }
    pub(crate) fn engram(self) -> &'static str {
        "sglang_triton"
    }
    pub(crate) fn comm(self) -> &'static str {
        "nccl"
    }
    pub(crate) fn elementwise(self) -> &'static str {
        "triton"
    }
}

/// One layer's attention type, as SGLang's launches depend on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SglangDsv41LayerType {
    /// 0 (window only), 1 or 2.
    pub compress_ratio: u32,
    /// Owns a compressor (and the compressed + index K cache).
    pub kv_source: bool,
    /// Scores the indexer.
    pub index_source: bool,
    /// An Engram block runs before this layer.
    pub engram: bool,
}

#[derive(Clone, Debug)]
pub struct SglangDsv41LayerConfig {
    pub path: SglangDsv41KernelPath,
    pub gpu_name: String,
    pub layer: SglangDsv41LayerType,
    pub tp_size: u32,
    pub ep_size: u32,
    pub hidden_size: u32,
    pub hc_mult: u32,
    pub hc_sinkhorn_iters: u32,
    pub num_attention_heads: u32,
    pub head_dim: u32,
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
    pub shared_intermediate_size: u32,
    pub expert_demand: ExpertDemand,
    pub engram_local_heads: u32,
    pub engram_head_dim: u32,
    pub engram_table_rows: u64,
    pub engram_residency: String,
    pub engram_wkv_n: u32,
    /// Query rows per decoding request: 1, or the speculative verify width
    /// (`speculative_num_draft_tokens`, 7 for DSpark block 6). The decode
    /// indexer is captured per width (`next_n`).
    pub verify_width: u32,
}

/// One forward's batch as every layer sees it.
#[derive(Clone, Debug, Default)]
pub struct SglangDsv41LayerInput {
    /// Rows every dense launch runs (decode rows padded to the CUDA-graph size).
    pub rows: u32,
    /// Decode/verify only: the mHC mix-stats side stream is live.
    pub decode_only: bool,
    /// `(append, context_after)` per prefill request.
    pub prefill: Vec<(u32, u32)>,
    /// Resident KV length before each decode row.
    pub decode_kv_lens: Vec<u32>,
    /// Query rows per decode request (1, or the verify width).
    pub decode_rows_per_request: u32,
}

impl SglangDsv41LayerInput {
    fn decode_call(&self) -> Option<PagedSparseMlaKernelInput> {
        let n = self.decode_kv_lens.len() as u64;
        (n > 0).then(|| PagedSparseMlaKernelInput {
            num_requests: n as u32,
            rows_per_request: self.decode_rows_per_request.max(1),
            context_before: (self
                .decode_kv_lens
                .iter()
                .map(|&c| u64::from(c))
                .sum::<u64>()
                / n) as u32,
        })
    }

    fn prefill_call(&self) -> Option<PagedSparseMlaKernelInput> {
        let n = self.prefill.len() as u64;
        (n > 0).then(|| {
            let rows: u64 = self.prefill.iter().map(|&(a, _)| u64::from(a)).sum();
            let prefix: u64 = self.prefill.iter().map(|&(a, c)| u64::from(c - a)).sum();
            PagedSparseMlaKernelInput {
                num_requests: n as u32,
                rows_per_request: rows.div_ceil(n) as u32,
                context_before: (prefix / n) as u32,
            }
        })
    }
}

/// Every op of one body layer.
pub struct SglangDsv41LayerWorklet {
    pub name: String,
    cfg: SglangDsv41LayerConfig,
    // Engram block.
    engram_lookup: Option<Op<EngramLookupKernel>>,
    engram_wkv: Option<Op<SingleGemmKernel>>,
    engram_gate: Option<Op<EngramGateKernel>>,
    // Attention sublayer.
    attn_mix: Op<MhcMixStatsSinkhornKernel>,
    attn_combine: Op<MhcCombineRmsNormKernel>,
    fused_wqa_wkv: Op<SingleGemmKernel>,
    qkv_norm: Op<ElementwiseKernel>,
    wq_b: Op<SingleGemmKernel>,
    rope_kv_store: Op<ElementwiseKernel>,
    compressor_proj: Option<Op<GemmFp32OutputKernel>>,
    compressor_store: Option<Op<ElementwiseKernel>>,
    indexer_weights: Option<Op<ElementwiseKernel>>,
    indexer_wk: Option<Op<ElementwiseKernel>>,
    indexer_wq_b: Option<Op<SingleGemmKernel>>,
    indexer_q_quant: Option<Op<ElementwiseKernel>>,
    indexer_bookkeeping: Option<Op<ElementwiseKernel>>,
    indexer_decode: Option<Op<SparseIndexerLogitsDecodeKernel>>,
    indexer_prefill: Option<Op<SparseIndexerLogitsPrefillKernel>>,
    indexer_topk: Option<Op<ElementwiseKernel>>,
    page_metadata: Op<ElementwiseKernel>,
    attn_decode: Op<PagedSparseMlaKernel>,
    attn_prefill: Op<PagedSparseMlaKernel>,
    inv_rope: Op<ElementwiseKernel>,
    wo_a: Op<SingleGemmKernel>,
    wo_b: Op<SingleGemmKernel>,
    attn_all_reduce: Op<AllReduceKernel>,
    attn_post: Op<MhcPostKernel>,
    // FFN sublayer.
    ffn_mix: Op<MhcMixStatsSinkhornKernel>,
    ffn_combine: Op<MhcCombineRmsNormKernel>,
    router: Op<ElementwiseKernel>,
    shared_gate_up: Op<SingleGemmKernel>,
    shared_act: Op<ElementwiseKernel>,
    shared_down: Op<SingleGemmKernel>,
    routed: Op<Nvfp4FusedMoeKernel>,
    ffn_all_reduce: Op<AllReduceKernel>,
    ffn_post: Op<MhcPostKernel>,
}

pub(super) fn op<S: KernelSpec>(
    name: &str,
    suffix: &str,
    cfg: S::Config,
    bridge: &PerfApiBridge,
) -> Result<Op<Kernel<S>>, BuildError>
where
    Kernel<S>: crate::timing::Probe,
{
    let op_name = format!("{name}.{suffix}");
    Ok(Op::new(
        op_name.clone(),
        Arc::new(Kernel::<S>::build(op_name, cfg, bridge)?),
    ))
}

impl SglangDsv41LayerWorklet {
    pub fn build(
        name: String,
        cfg: SglangDsv41LayerConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let c = &cfg;
        let p = c.path;
        let gpu = c.gpu_name.as_str();
        let l = c.layer;
        let tp = c.tp_size;
        assert!(tp > 0 && c.num_attention_heads % tp == 0 && c.o_groups % tp == 0);
        assert!(c.num_experts % c.ep_size == 0 && c.shared_intermediate_size % tp == 0);
        assert!(l.compress_ratio <= 2);
        assert!(l.compress_ratio > 0 || (!l.kv_source && !l.index_source));
        let ew = |i: u32, o: u32| placeholder(&[p.elementwise()], gpu, i, o);
        let fp8 = |n: u32, k: u32| SingleGemmKernelConfig {
            backends: vec![p.fp8_gemm()],
            gpu_name: c.gpu_name.clone(),
            n: n.into(),
            k: k.into(),
            dtype: DType::Fp8E4m3,
        };
        let bf16 = |n: u32, k: u32| SingleGemmKernelConfig {
            backends: vec![p.bf16_gemm()],
            gpu_name: c.gpu_name.clone(),
            n: n.into(),
            k: k.into(),
            dtype: DType::Bf16,
        };
        let mhc = |backend: &'static str| MhcRmsNormKernelConfig {
            backends: vec![backend],
            gpu_name: c.gpu_name.clone(),
            hidden_size: c.hidden_size.into(),
            hc_mult: c.hc_mult,
            hidden_dtype: DType::Bf16,
        };
        let mix = || MhcMixStatsSinkhornKernelConfig {
            backends: vec![p.mhc()],
            gpu_name: c.gpu_name.clone(),
            hidden_size: c.hidden_size.into(),
            hc_mult: c.hc_mult,
            sinkhorn_iters: c.hc_sinkhorn_iters,
            hidden_dtype: DType::Bf16,
        };
        let all_reduce = || AllReduceKernelConfig {
            backends: vec![p.comm()],
            gpu_name: c.gpu_name.clone(),
            num_gpus: tp,
            fabric: Fabric::Pcie,
        };
        let hidden = c.hidden_size;
        let head_dim = c.head_dim;
        let heads = c.num_attention_heads / tp;
        let groups = c.o_groups / tp;
        let ratio = l.compress_ratio;
        let index_width = c.index_num_heads * c.index_head_dim;
        let shared_local = c.shared_intermediate_size / tp;
        let attn_cfg = PagedSparseMlaKernelConfig {
            backends: vec![p.attention()],
            gpu_name: c.gpu_name.clone(),
            num_heads: heads.into(),
            head_dim: head_dim.into(),
            value_dim: head_dim.into(),
            swa_window: c.sliding_window,
            swa_index_capacity: c.sliding_window.div_ceil(64) * 64,
            compress_ratio: ratio,
            extra_index_capacity: if ratio == 0 { 0 } else { c.index_topk },
            page_size: PAGE_SIZE,
            kv_layout: "v4".into(),
            q_dtype: DType::Bf16,
            cache_dtype: DType::Fp8E4m3,
            output_dtype: DType::Bf16,
        };
        let n = &name;
        Ok(Self {
            engram_lookup: l
                .engram
                .then(|| {
                    op::<EngramLookupSpec>(
                        n,
                        "engram.lookup",
                        EngramLookupKernelConfig {
                            backends: vec![p.engram()],
                            gpu_name: c.gpu_name.clone(),
                            local_heads: c.engram_local_heads,
                            head_dim: c.engram_head_dim,
                            quant_block_size: 32,
                            table_rows: c.engram_table_rows,
                            residency: c.engram_residency.clone(),
                            weight_dtype: DType::Fp8E4m3,
                        },
                        bridge,
                    )
                })
                .transpose()?,
            engram_wkv: l
                .engram
                .then(|| {
                    op::<SingleGemmSpec>(
                        n,
                        "engram.wkv",
                        fp8(c.engram_wkv_n, c.engram_local_heads * c.engram_head_dim),
                        bridge,
                    )
                })
                .transpose()?,
            engram_gate: l
                .engram
                .then(|| op::<EngramGateSpec>(n, "engram.gate", mhc(p.mhc()), bridge))
                .transpose()?,
            attn_mix: op(n, "attn.mhc_mix_stats", mix(), bridge)?,
            attn_combine: op(n, "attn.mhc_combine_norm", mhc(p.mhc()), bridge)?,
            fused_wqa_wkv: op(
                n,
                "attn.fused_wqa_wkv",
                fp8(c.q_lora_rank + head_dim, hidden),
                bridge,
            )?,
            // FlashInfer RMSNorm over q-a and kv: bf16 in, bf16 out.
            qkv_norm: op(
                n,
                "attn.qkv_rmsnorm",
                ew(
                    (c.q_lora_rank + head_dim) * 2,
                    (c.q_lora_rank + head_dim) * 2,
                ),
                bridge,
            )?,
            wq_b: op(n, "attn.wq_b", fp8(heads * head_dim, c.q_lora_rank), bridge)?,
            // `deepseek_rope_kernel` + `fused_k_norm_rope_flashmla` + the SWA
            // store: q and kv in, one 584-byte V4 row per token out.
            rope_kv_store: op(
                n,
                "attn.rope_kv_store",
                ew(
                    heads * head_dim * 2 + head_dim * 2,
                    heads * head_dim * 2 + 584,
                ),
                bridge,
            )?,
            // Compressor kv-score GEMM (cuBLAS, fp32 out) and the C2 store
            // (`flash_c2_decode_kernel` / `c2_prefill_store_kernel`).
            compressor_proj: l
                .kv_source
                .then(|| {
                    op::<GemmFp32OutputSpec>(
                        n,
                        "attn.compressor_kv_score",
                        GemmFp32OutputKernelConfig {
                            backends: vec![p.fp32_out_gemm()],
                            gpu_name: c.gpu_name.clone(),
                            n: (head_dim * ratio).into(),
                            k: hidden.into(),
                            input_dtype: DType::Bf16,
                        },
                        bridge,
                    )
                })
                .transpose()?,
            compressor_store: l
                .kv_source
                .then(|| {
                    op::<ElementwiseSpec>(
                        n,
                        "attn.compressor_store",
                        ew(head_dim * ratio * 4, 584 / ratio.max(1)),
                        bridge,
                    )
                })
                .transpose()?,
            // `sglang::n32k5120_kernel`: head weights (bf16 GEMV).
            indexer_weights: l
                .index_source
                .then(|| {
                    op::<ElementwiseSpec>(
                        n,
                        "indexer.weights_proj",
                        ew(hidden * 2, c.index_num_heads * 2),
                        bridge,
                    )
                })
                .transpose()?,
            // `sglang::n128k512_kernel` + `index_k_kernel`: the index key.
            indexer_wk: l
                .kv_source
                .then(|| {
                    op::<ElementwiseSpec>(
                        n,
                        "indexer.wk_store",
                        ew(head_dim * 2, (c.index_head_dim / 2 + 4) / ratio.max(1)),
                        bridge,
                    )
                })
                .transpose()?,
            indexer_wq_b: l
                .index_source
                .then(|| {
                    op::<SingleGemmSpec>(n, "indexer.wq_b", fp8(index_width, c.q_lora_rank), bridge)
                })
                .transpose()?,
            // `_rope_tail_fake_quant_fp4_kernel`.
            indexer_q_quant: l
                .index_source
                .then(|| {
                    op::<ElementwiseSpec>(
                        n,
                        "indexer.q_rope_quant",
                        ew(index_width * 2, index_width / 2),
                        bridge,
                    )
                })
                .transpose()?,
            indexer_bookkeeping: l
                .index_source
                .then(|| op::<ElementwiseSpec>(n, "indexer.torch_bookkeeping", ew(64, 64), bridge))
                .transpose()?,
            indexer_decode: l
                .index_source
                .then(|| {
                    op::<SparseIndexerLogitsDecodeSpec>(
                        n,
                        "indexer.logits_decode",
                        SparseIndexerLogitsDecodeKernelConfig {
                            backends: vec![p.index_decode()],
                            gpu_name: c.gpu_name.clone(),
                            next_n: c.verify_width.max(1),
                            num_heads: c.index_num_heads.into(),
                            head_dim: c.index_head_dim.into(),
                            dtype: DType::Mxfp4E2m1,
                        },
                        bridge,
                    )
                })
                .transpose()?,
            indexer_prefill: l
                .index_source
                .then(|| {
                    op::<SparseIndexerLogitsPrefillSpec>(
                        n,
                        "indexer.logits_prefill",
                        SparseIndexerLogitsPrefillKernelConfig {
                            backends: vec![p.index_prefill()],
                            gpu_name: c.gpu_name.clone(),
                            num_heads: c.index_num_heads.into(),
                            head_dim: c.index_head_dim.into(),
                            compress_ratio: ratio,
                            dtype: DType::Mxfp4E2m1,
                        },
                        bridge,
                    )
                })
                .transpose()?,
            indexer_topk: l
                .index_source
                .then(|| op::<ElementwiseSpec>(n, "indexer.torch_topk", ew(64, 64), bridge))
                .transpose()?,
            // `_page_mark_kernel` + `_page_split_kernel`.
            page_metadata: op(n, "attn.page_metadata", ew(16, 16), bridge)?,
            attn_decode: op(n, "attn.sparse_mla_decode", attn_cfg.clone(), bridge)?,
            attn_prefill: op(n, "attn.sparse_mla_prefill", attn_cfg, bridge)?,
            inv_rope: op(
                n,
                "attn.inv_rope",
                ew(heads * head_dim * 2, heads * head_dim * 2),
                bridge,
            )?,
            // `_wo_a_partial` + `_wo_a_reduce`: the grouped o-a projection,
            // priced as one BF16 GEMM of the same work (groups x o_lora wide).
            wo_a: op(
                n,
                "attn.wo_a",
                bf16(
                    groups * c.o_lora_rank,
                    c.num_attention_heads * head_dim / c.o_groups,
                ),
                bridge,
            )?,
            wo_b: op(n, "attn.wo_b", fp8(hidden, c.o_lora_rank * groups), bridge)?,
            attn_all_reduce: op(n, "attn.tp_all_reduce", all_reduce(), bridge)?,
            attn_post: op(n, "attn.mhc_post", mhc(p.mhc_post()), bridge)?,
            ffn_mix: op(n, "ffn.mhc_mix_stats", mix(), bridge)?,
            ffn_combine: op(n, "ffn.mhc_combine_norm", mhc(p.mhc()), bridge)?,
            // `tiny_n_gemm_kernel` (bf16 x 384 experts) + `_router_triton_kernel`.
            router: op(
                n,
                "ffn.router",
                ew(hidden * 2, c.num_experts * 4 + c.top_k * 8),
                bridge,
            )?,
            shared_gate_up: op(
                n,
                "ffn.shared_gate_up",
                fp8(2 * shared_local, hidden),
                bridge,
            )?,
            // `silu_mul_clamp_kernel`.
            shared_act: op(
                n,
                "ffn.shared_act",
                ew(2 * shared_local * 2, shared_local * 2),
                bridge,
            )?,
            shared_down: op(n, "ffn.shared_down", fp8(hidden, shared_local), bridge)?,
            routed: op(
                n,
                "ffn.routed_moe",
                Nvfp4FusedMoeKernelConfig {
                    backends: vec![p.moe()],
                    gpu_name: c.gpu_name.clone(),
                    hidden_size: hidden.into(),
                    intermediate_size: c.moe_intermediate_size.into(),
                    num_experts: c.num_experts.into(),
                    num_local_experts: (c.num_experts / c.ep_size).into(),
                    top_k: c.top_k,
                    input_dtype: DType::Bf16,
                    weight_format: DType::Mxfp4E2m1,
                    activation_format: None,
                    group_size: 32,
                    routing_method: "precomputed_dsv4".into(),
                    n_group: 1,
                    topk_group: 1,
                    routed_scaling_numerator: 1,
                    routed_scaling_denominator: 1,
                    expert_demand: c.expert_demand.clone(),
                    folded_rank_position: 0,
                },
                bridge,
            )?,
            ffn_all_reduce: op(n, "ffn.tp_all_reduce", all_reduce(), bridge)?,
            ffn_post: op(n, "ffn.mhc_post", mhc(p.mhc_post()), bridge)?,
            cfg,
            name,
        })
    }

    /// Mint order here is the eval order in [`Self::eval`].
    pub fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        let mut engram = Vec::new();
        if let (Some(a), Some(w), Some(g)) =
            (&self.engram_lookup, &self.engram_wkv, &self.engram_gate)
        {
            engram = vec![a.compile(b), w.compile(b), g.compile(b)];
        }
        // Attention: combine + first GEMM beside mix-stats (side stream, decode),
        // then mix-stats again as the serial copy (prefill).
        let attn_main = CostNode::Sum(vec![
            self.attn_combine.compile(b),
            self.fused_wqa_wkv.compile(b),
        ]);
        let attn_side = self.attn_mix.compile(b);
        let attn_serial = super::deepseek_v41_common::compile_serial_copy(&self.attn_mix, b);
        let mut attn = vec![
            CostNode::Parallel {
                overlap: 1.0,
                children: vec![attn_main, attn_side],
            },
            attn_serial,
            self.qkv_norm.compile(b),
            self.wq_b.compile(b),
            self.rope_kv_store.compile(b),
        ];
        for op in [&self.compressor_proj].into_iter().flatten() {
            attn.push(op.compile(b));
        }
        for op in [
            &self.compressor_store,
            &self.indexer_weights,
            &self.indexer_wk,
        ]
        .into_iter()
        .flatten()
        {
            attn.push(op.compile(b));
        }
        if let Some(op) = &self.indexer_wq_b {
            attn.push(op.compile(b));
        }
        if let Some(op) = &self.indexer_q_quant {
            attn.push(op.compile(b));
        }
        if let Some(op) = &self.indexer_bookkeeping {
            attn.push(CostNode::Scale {
                n: INDEXER_BOOKKEEPING_LAUNCHES,
                child: Box::new(op.compile(b)),
            });
        }
        if let Some(op) = &self.indexer_decode {
            attn.push(op.compile(b));
        }
        if let Some(op) = &self.indexer_prefill {
            attn.push(op.compile(b));
        }
        if let Some(op) = &self.indexer_topk {
            attn.push(CostNode::Scale {
                n: INDEXER_TOPK_LAUNCHES,
                child: Box::new(op.compile(b)),
            });
        }
        attn.extend([
            self.page_metadata.compile(b),
            self.attn_decode.compile(b),
            self.attn_prefill.compile(b),
            self.inv_rope.compile(b),
            self.wo_a.compile(b),
            self.wo_b.compile(b),
            self.attn_all_reduce.compile(b),
            self.attn_post.compile(b),
        ]);
        let ffn_main = CostNode::Sum(vec![self.ffn_combine.compile(b), self.router.compile(b)]);
        let ffn_side = self.ffn_mix.compile(b);
        let ffn_serial = super::deepseek_v41_common::compile_serial_copy(&self.ffn_mix, b);
        let ffn = vec![
            CostNode::Parallel {
                overlap: 1.0,
                children: vec![ffn_main, ffn_side],
            },
            ffn_serial,
            self.shared_gate_up.compile(b),
            self.shared_act.compile(b),
            self.shared_down.compile(b),
            self.routed.compile(b),
            self.ffn_all_reduce.compile(b),
            self.ffn_post.compile(b),
        ];
        let l = self.cfg.layer;
        let mut children = engram;
        children.push(CostNode::Sum(attn));
        children.push(CostNode::Sum(ffn));
        CostNode::Labeled {
            label: format!(
                "{} (SglangDsv41LayerWorklet) [ratio={}; kv_source={}; index={}; engram={}; tp={}; ep={}]",
                self.name, l.compress_ratio, l.kv_source, l.index_source, l.engram, self.cfg.tp_size, self.cfg.ep_size
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &SglangDsv41LayerInput, ev: &mut Evaluator) {
        let rows = input.rows;
        let zero = rows == 0;
        let ew = ElementwiseKernelInput { num_tokens: rows };
        let gemm = SingleGemmKernelInput { m: rows };
        let hc = MhcRmsNormKernelInput { num_tokens: rows };
        let side = input.decode_only;
        let reduce_bytes = u64::from(rows) * u64::from(self.cfg.hidden_size) * 2;
        let all_reduce = AllReduceKernelInput {
            message_size_bytes: reduce_bytes,
        };

        if let (Some(a), Some(w), Some(g)) =
            (&self.engram_lookup, &self.engram_wkv, &self.engram_gate)
        {
            eval_or_zero(a, EngramLookupKernelInput { num_tokens: rows }, zero, ev);
            eval_or_zero(w, gemm.clone(), zero, ev);
            eval_or_zero(g, hc.clone(), zero, ev);
        }
        eval_or_zero(&self.attn_combine, hc.clone(), zero, ev);
        eval_or_zero(&self.fused_wqa_wkv, gemm.clone(), zero, ev);
        eval_or_zero(&self.attn_mix, hc.clone(), zero || !side, ev);
        eval_or_zero(&self.attn_mix, hc.clone(), zero || side, ev);
        eval_or_zero(&self.qkv_norm, ew.clone(), zero, ev);
        eval_or_zero(&self.wq_b, gemm.clone(), zero, ev);
        eval_or_zero(&self.rope_kv_store, ew.clone(), zero, ev);
        if let Some(op) = &self.compressor_proj {
            eval_or_zero(op, GemmFp32OutputKernelInput { m: rows }, zero, ev);
        }
        for op in [
            &self.compressor_store,
            &self.indexer_weights,
            &self.indexer_wk,
        ]
        .into_iter()
        .flatten()
        {
            eval_or_zero(op, ew.clone(), zero, ev);
        }
        if let Some(op) = &self.indexer_wq_b {
            eval_or_zero(op, gemm.clone(), zero, ev);
        }
        if let Some(op) = &self.indexer_q_quant {
            eval_or_zero(op, ew.clone(), zero, ev);
        }
        if let Some(op) = &self.indexer_bookkeeping {
            eval_or_zero(
                op,
                ElementwiseKernelInput {
                    num_tokens: rows.max(1),
                },
                zero,
                ev,
            );
        }
        let ratio = self.cfg.layer.compress_ratio.max(1);
        let decode = input.decode_call();
        let prefill = input.prefill_call();
        if let Some(op) = &self.indexer_decode {
            match &decode {
                Some(call) => {
                    let max_ctx = input.decode_kv_lens.iter().copied().max().unwrap_or(0);
                    let keys = (call.context_before / ratio).max(1);
                    let width = (max_ctx / ratio).max(1).div_ceil(64) * 64;
                    // `next_n` rows per request are one indexer row each at
                    // width 1; at the verify width the kind counts requests.
                    let batch_size = if self.cfg.verify_width > 1 {
                        call.num_requests
                    } else {
                        call.num_requests * call.rows_per_request
                    };
                    op.eval(
                        &SparseIndexerLogitsDecodeKernelInput {
                            batch_size,
                            num_keys: keys,
                            logits_width: width,
                        },
                        ev,
                    );
                }
                None => eval_or_zero(
                    op,
                    SparseIndexerLogitsDecodeKernelInput {
                        batch_size: 1,
                        num_keys: 1,
                        logits_width: 64,
                    },
                    true,
                    ev,
                ),
            }
        }
        if let Some(op) = &self.indexer_prefill {
            match &prefill {
                Some(call) => op.eval(
                    &SparseIndexerLogitsPrefillKernelInput {
                        num_queries: call.num_requests * call.rows_per_request,
                        prefix_len: call.context_before,
                    },
                    ev,
                ),
                None => eval_or_zero(
                    op,
                    SparseIndexerLogitsPrefillKernelInput {
                        num_queries: 1,
                        prefix_len: 0,
                    },
                    true,
                    ev,
                ),
            }
        }
        if let Some(op) = &self.indexer_topk {
            eval_or_zero(
                op,
                ElementwiseKernelInput {
                    num_tokens: rows.max(1),
                },
                zero,
                ev,
            );
        }
        eval_or_zero(&self.page_metadata, ew.clone(), zero, ev);
        let empty = PagedSparseMlaKernelInput {
            num_requests: 1,
            rows_per_request: 1,
            context_before: 0,
        };
        match decode {
            Some(call) => self.attn_decode.eval(&call, ev),
            None => eval_or_zero(&self.attn_decode, empty.clone(), true, ev),
        }
        match prefill {
            Some(call) => self.attn_prefill.eval(&call, ev),
            None => eval_or_zero(&self.attn_prefill, empty, true, ev),
        }
        eval_or_zero(&self.inv_rope, ew.clone(), zero, ev);
        eval_or_zero(&self.wo_a, gemm.clone(), zero, ev);
        eval_or_zero(&self.wo_b, gemm.clone(), zero, ev);
        eval_or_zero(&self.attn_all_reduce, all_reduce.clone(), zero, ev);
        eval_or_zero(&self.attn_post, hc.clone(), zero, ev);

        eval_or_zero(&self.ffn_combine, hc.clone(), zero, ev);
        eval_or_zero(&self.router, ew.clone(), zero, ev);
        eval_or_zero(&self.ffn_mix, hc.clone(), zero || !side, ev);
        eval_or_zero(&self.ffn_mix, hc.clone(), zero || side, ev);
        eval_or_zero(&self.shared_gate_up, gemm.clone(), zero, ev);
        eval_or_zero(&self.shared_act, ew, zero, ev);
        eval_or_zero(&self.shared_down, gemm, zero, ev);
        eval_or_zero(
            &self.routed,
            Nvfp4FusedMoeKernelInput { num_tokens: rows },
            zero,
            ev,
        );
        eval_or_zero(&self.ffn_all_reduce, all_reduce, zero, ev);
        eval_or_zero(&self.ffn_post, hc, zero, ev);
    }
}

/// Embedding (+ all-reduce), the Engram hash and the hc broadcast, once per forward.
pub struct SglangDsv41PrologueWorklet {
    embedding: Op<ElementwiseKernel>,
    all_reduce: Op<AllReduceKernel>,
    engram_hash: Op<ElementwiseKernel>,
    hc_broadcast: Op<ElementwiseKernel>,
    hidden_size: u32,
}

impl SglangDsv41PrologueWorklet {
    pub fn build(
        name: String,
        path: SglangDsv41KernelPath,
        gpu_name: &str,
        hidden_size: u32,
        hc_mult: u32,
        tp_size: u32,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let ew = |i: u32, o: u32| placeholder(&[path.elementwise()], gpu_name, i, o);
        Ok(Self {
            // `_vocab_parallel_embedding_kernel`: one bf16 row per token.
            embedding: op(&name, "embedding", ew(8, hidden_size * 2), bridge)?,
            all_reduce: op(
                &name,
                "embedding_all_reduce",
                AllReduceKernelConfig {
                    backends: vec![path.comm()],
                    gpu_name: gpu_name.to_string(),
                    num_gpus: tp_size,
                    fabric: Fabric::Pcie,
                },
                bridge,
            )?,
            // `_engram_hash_kernel`: token ids in, 24 hash ids out.
            engram_hash: op(&name, "engram_hash", ew(32, 24 * 8), bridge)?,
            // `_hc_broadcast_kernel`: one row to hc_mult copies.
            hc_broadcast: op(
                &name,
                "hc_broadcast",
                ew(hidden_size * 2, hidden_size * 2 * hc_mult),
                bridge,
            )?,
            hidden_size,
        })
    }

    pub fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            self.embedding.compile(b),
            self.all_reduce.compile(b),
            self.engram_hash.compile(b),
            self.hc_broadcast.compile(b),
        ])
    }

    pub fn eval(&self, rows: u32, ev: &mut Evaluator) {
        let zero = rows == 0;
        let ew = ElementwiseKernelInput { num_tokens: rows };
        eval_or_zero(&self.embedding, ew.clone(), zero, ev);
        eval_or_zero(
            &self.all_reduce,
            AllReduceKernelInput {
                message_size_bytes: u64::from(rows) * u64::from(self.hidden_size) * 2,
            },
            zero,
            ev,
        );
        eval_or_zero(&self.engram_hash, ew.clone(), zero, ev);
        eval_or_zero(&self.hc_broadcast, ew, zero, ev);
    }
}

/// hc collapse, final norm, lm_head and the logits all-gather.
pub struct SglangDsv41HeadWorklet {
    collapse: Op<ElementwiseKernel>,
    norm: Op<ElementwiseKernel>,
    lm_head: Op<SingleGemmKernel>,
    logits_all_gather: Op<AllReduceKernel>,
    vocab_per_rank: u32,
}

impl SglangDsv41HeadWorklet {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        name: String,
        path: SglangDsv41KernelPath,
        gpu_name: &str,
        hidden_size: u32,
        hc_mult: u32,
        vocab_size: u32,
        tp_size: u32,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let ew = |i: u32, o: u32| placeholder(&[path.elementwise()], gpu_name, i, o);
        let vocab_per_rank = vocab_size.div_ceil(tp_size);
        Ok(Self {
            // `_mhc_combine_kernel`: hc streams in, one row out.
            collapse: op(
                &name,
                "hc_collapse",
                ew(hidden_size * 2 * hc_mult, hidden_size * 2),
                bridge,
            )?,
            norm: op(
                &name,
                "final_rmsnorm",
                ew(hidden_size * 2, hidden_size * 2),
                bridge,
            )?,
            lm_head: op(
                &name,
                "lm_head",
                SingleGemmKernelConfig {
                    backends: vec![path.bf16_gemm()],
                    gpu_name: gpu_name.to_string(),
                    n: vocab_per_rank.into(),
                    k: hidden_size.into(),
                    dtype: DType::Bf16,
                },
                bridge,
            )?,
            // A ring all-gather moves the same bytes as an all-reduce of half
            // the gathered buffer (see `all_gather_as_all_reduce_bytes`).
            logits_all_gather: op(
                &name,
                "logits_all_gather",
                AllReduceKernelConfig {
                    backends: vec![path.comm()],
                    gpu_name: gpu_name.to_string(),
                    num_gpus: tp_size,
                    fabric: Fabric::Pcie,
                },
                bridge,
            )?,
            vocab_per_rank,
        })
    }

    pub fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            self.collapse.compile(b),
            self.norm.compile(b),
            self.lm_head.compile(b),
            self.logits_all_gather.compile(b),
        ])
    }

    pub fn eval(&self, logits_rows: u32, tp_size: u32, ev: &mut Evaluator) {
        let zero = logits_rows == 0;
        let ew = ElementwiseKernelInput {
            num_tokens: logits_rows,
        };
        eval_or_zero(&self.collapse, ew.clone(), zero, ev);
        eval_or_zero(&self.norm, ew, zero, ev);
        eval_or_zero(
            &self.lm_head,
            SingleGemmKernelInput { m: logits_rows },
            zero,
            ev,
        );
        let gathered =
            u64::from(logits_rows) * u64::from(self.vocab_per_rank) * u64::from(tp_size) * 4;
        eval_or_zero(
            &self.logits_all_gather,
            AllReduceKernelInput {
                message_size_bytes: super::deepseek_v41_common::all_gather_as_all_reduce_bytes(
                    gathered,
                ),
            },
            zero,
            ev,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batch_prices_as_one_uniform_call_per_phase() {
        let input = SglangDsv41LayerInput {
            rows: 4,
            decode_only: false,
            prefill: vec![(2048, 50_000), (1000, 1000)],
            decode_kv_lens: vec![1000, 3000],
            decode_rows_per_request: 1,
        };
        let decode = input.decode_call().expect("two decodes");
        assert_eq!(
            (
                decode.num_requests,
                decode.rows_per_request,
                decode.context_before
            ),
            (2, 1, 2000)
        );
        let prefill = input.prefill_call().expect("two prefills");
        // 3048 rows over 2 requests; prefixes 47952 and 0.
        assert_eq!(
            (
                prefill.num_requests,
                prefill.rows_per_request,
                prefill.context_before
            ),
            (2, 1524, 23_976)
        );
        let empty = SglangDsv41LayerInput::default();
        assert!(empty.decode_call().is_none() && empty.prefill_call().is_none());
    }

    #[test]
    fn only_sm12x_takes_this_graph() {
        assert_eq!(
            SglangDsv41KernelPath::of("NVIDIA RTX PRO 6000 Blackwell Server Edition").unwrap(),
            SglangDsv41KernelPath::Sm120
        );
        assert!(SglangDsv41KernelPath::of("NVIDIA B200").is_err());
    }
}
