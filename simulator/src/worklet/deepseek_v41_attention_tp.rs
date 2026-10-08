//! DeepSeek-V4.1-Flash attention block for one layer on one TP rank (vLLM).
//!
//! One sync section, entry mHC through the TP all-reduce of `wo_b`, at the
//! fork's launch granularity (`models/deepseek_v41/attention.py`,
//! `nvidia/flash_mla_mega_attn.py`). The layer type picks which launches
//! exist: compress ratio 0/1/2, KV source (owns a compressor), index role
//! (owner / non-owner of an index K cache), and the layer-20 candidate scheme.
//! [`production_layer`] is the checkpoint's schedule.
//!
//! Launch order and streams (capture 2 = Slurm job 1185, device 0; decode
//! iteration 1337 = 48 decode rows, mixed iteration 310 = 2048 tokens, mixed
//! iteration 288 = 174 tokens; timestamps in us relative to each iteration's
//! first kernel):
//!
//! 1. entry: `mega_mhc` (previous FFN post + this pre), or at layer 0 the
//!    hc-copy expand plus the non-fused pre (hc_prenorm GEMM, pre_big_fuse).
//!    Engram layers 1 and 14 get their non-fused pre from the Engram worklet.
//! 2. stage A, `_run_parallel_input_projections`: `fused_wqa_wkv` on the main
//!    stream; the compressor `kv_score` fp32 GEMM (KV sources) and the indexer
//!    `weights_proj` (index sources) on aux streams, only when
//!    `T <= VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` (1024). Decode 1337 layer 2:
//!    compressor GEMM s19 835.49-841.18 us, weights_proj s47915 835.87-839.68,
//!    fused_wqa_wkv s47916 836.10-849.28, joined by the QK RMSNorm at 848.58
//!    (real overlap -> `Parallel`). Mixed 288 (T=174) layer 2: s19 / s45358 /
//!    s45359 all start within 4093.69-4094.33 (overlap). Mixed 310 (T=2048)
//!    layer 2: all on s19 back to back, 3721.92-3764.51 (serial -> `Sum`).
//! 3. QK RMSNorm (placeholder).
//! 4. stage B1, `maybe_execute_in_parallel`: `wq_b` + KV insert on main, the
//!    compressor state save/norm on aux (KV sources).
//! 5. stage B2: indexer K (`wk` + K store, owners) and Q (`wq_b` + fused
//!    q-rope-quant, all index sources) on main, the compressor NVFP4 insert on
//!    aux (KV sources). B1/B2 overlap only when the aux stream is live, i.e.
//!    FULL-graph decode (see `attention_aux_stream_live`). Decode 1337 layer 2:
//!    compress_norm s47917 851.17-855.23 under wq_b s19 850.05-857.82; NVFP4
//!    insert s47917 862.11-864.83 under wk s19 860.26-863.74 (overlap). Mixed
//!    288 and 310: both on s19 after the KV insert (4128.57 / 3854.69; serial).
//! 6. indexer scoring (index sources), then mega attention (prefill segment,
//!    decode segment), the `wo_a` MXFP8 einsum, `wo_b`, and the all-reduce.
//!    All serial on the main stream in every iteration inspected.
//!
//! Folded, no leaf: the position cast before attention, the global top-k remap
//! (into the mega-attention op) and the per-layer attention metadata.

use std::sync::Arc;

use super::deepseek_v41_common::{
    compile_serial_copy, eval_or_zero, gated_fanout, placeholder,
    MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
};
use crate::common::Fabric;
use crate::op::attention::{
    DeepseekV41CandidateRole, DeepseekV41IndexerOp, DeepseekV41IndexerOpConfig,
    DeepseekV41IndexerOpInput, DeepseekV41MegaAttnOp, DeepseekV41MegaAttnOpConfig,
    DeepseekV41MegaAttnOpInput,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    AllReduceFusionKernel, AllReduceFusionKernelConfig, AllReduceFusionKernelInput,
    BatchedGemmKernel, BatchedGemmKernelConfig, BatchedGemmKernelInput,
    QPadKvRopeMxfp8InsertKernel, QPadKvRopeMxfp8InsertKernelConfig,
    QPadKvRopeMxfp8InsertKernelInput, ElementwiseKernel, ElementwiseKernelConfig,
    ElementwiseKernelInput, GemmFp32OutputKernel, GemmFp32OutputKernelConfig,
    GemmFp32OutputKernelInput, MhcFusedPostPreRmsNormKernel, MhcRmsNormKernelConfig,
    MhcRmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

/// Layers whose attention entry is the Engram worklet's non-fused pre.
pub const ENGRAM_LAYERS: [u32; 2] = [1, 14];
/// Layers that own a compressor and the compressed KV cache.
pub const KV_SOURCE_LAYERS: [u32; 4] = [2, 8, 14, 20];
/// Layers that score the indexer. The first four own their index K cache;
/// 24/28/32/36 share layer 20's.
pub const INDEX_SOURCE_LAYERS: [u32; 8] = [2, 8, 14, 20, 24, 28, 32, 36];
pub const CANDIDATE_WRITER_LAYER: u32 = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeepseekV41AttentionEntry {
    /// Layer 0: hc-copy expand (`nvidia/model.py:434`) and the non-fused pre
    /// (`mhc_pre_delayed_tilelang`, `:435`).
    Layer0Pre,
    /// Engram layers: the Engram worklet already ran the non-fused pre
    /// (`nvidia/model.py:480`).
    AfterEngram,
    /// One DeepGEMM `mega_mhc` (`nvidia/model.py:497`).
    FusedPostPre,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeepseekV41IndexRole {
    None,
    /// Owns its index K cache: `wk` + K store on top of the Q path.
    Owner,
    /// Scores against layer 20's K cache: Q path only.
    NonOwner,
}

/// One layer's attention type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeepseekV41AttentionLayer {
    pub entry: DeepseekV41AttentionEntry,
    /// 0 (SWA only), 1 or 2.
    pub compress_ratio: u32,
    pub kv_source: bool,
    pub index: DeepseekV41IndexRole,
    pub candidate: DeepseekV41CandidateRole,
}

/// The DeepSeek-V4.1-Flash checkpoint's 40-layer schedule: layers 0-1 ratio 0,
/// 2-19 ratio 2, 20-39 ratio 1.
pub fn production_layer(layer: u32) -> DeepseekV41AttentionLayer {
    assert!(layer < 40, "DeepSeek-V4.1-Flash has 40 body layers, got {layer}");
    let kv_source = KV_SOURCE_LAYERS.contains(&layer);
    let index_source = INDEX_SOURCE_LAYERS.contains(&layer);
    DeepseekV41AttentionLayer {
        entry: if layer == 0 {
            DeepseekV41AttentionEntry::Layer0Pre
        } else if ENGRAM_LAYERS.contains(&layer) {
            DeepseekV41AttentionEntry::AfterEngram
        } else {
            DeepseekV41AttentionEntry::FusedPostPre
        },
        compress_ratio: match layer {
            0..=1 => 0,
            2..=19 => 2,
            _ => 1,
        },
        kv_source,
        index: match (index_source, kv_source) {
            (false, _) => DeepseekV41IndexRole::None,
            (true, true) => DeepseekV41IndexRole::Owner,
            (true, false) => DeepseekV41IndexRole::NonOwner,
        },
        candidate: if layer == CANDIDATE_WRITER_LAYER {
            DeepseekV41CandidateRole::Writer
        } else if index_source && !kv_source {
            DeepseekV41CandidateRole::Consumer
        } else {
            DeepseekV41CandidateRole::None
        },
    }
}

#[derive(Clone, Debug)]
pub struct DeepseekV41AttentionTpWorkletConfig {
    pub layer: DeepseekV41AttentionLayer,
    pub tp_size: u32,
    /// Forces every gated side branch serial (the serial-stream counterfactual).
    pub serialize_streams: bool,
    pub hidden_size: Dim,
    pub hc_mult: u32,
    /// Global attention heads (sharded by TP).
    pub num_attention_heads: Dim,
    /// FlashMLA's padded Q width per rank (64).
    pub padded_heads: Dim,
    pub head_dim: Dim,
    pub rope_dim: Dim,
    pub q_lora_rank: Dim,
    pub o_lora_rank: Dim,
    /// Global `wo_a` groups (sharded by TP).
    pub o_groups: Dim,
    /// Index heads; the indexer is replicated, not sharded.
    pub index_num_heads: Dim,
    pub index_head_dim: Dim,
    pub index_topk: u32,
    pub window_size: u32,
    /// Tokens per KV page; an index page holds `kv_block_size / ratio` keys.
    pub kv_block_size: u32,
    /// Tokens per SWA page of the KV insert.
    pub swa_block_size: u32,
    pub max_model_len: u32,
    pub max_num_batched_tokens: u32,
    pub prefill_chunk_size: u32,
    pub gpu_name: String,
    pub mhc_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub fp32_gemm_backends: Vec<&'static str>,
    pub kv_insert_backends: Vec<&'static str>,
    pub mega_attn_backends: Vec<&'static str>,
    pub wo_a_backends: Vec<&'static str>,
    pub all_reduce_backends: Vec<&'static str>,
    pub index_logits_prefill_backends: Vec<&'static str>,
    pub index_logits_decode_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct DeepseekV41AttentionTpWorkletResolved {
    pub raw_cfg: DeepseekV41AttentionTpWorkletConfig,
    pub heads_per_rank: u32,
    pub o_groups_per_rank: u32,
    pub entry_expand: Option<ElementwiseKernelConfig>,
    pub entry_hc_prenorm: Option<ElementwiseKernelConfig>,
    pub entry_pre_fuse: Option<ElementwiseKernelConfig>,
    pub entry_fused: Option<MhcRmsNormKernelConfig>,
    pub fused_wqa_wkv: SingleGemmKernelConfig,
    pub compressor_proj: Option<GemmFp32OutputKernelConfig>,
    pub indexer_weights_proj: Option<ElementwiseKernelConfig>,
    pub qk_rmsnorm: ElementwiseKernelConfig,
    pub wq_b: SingleGemmKernelConfig,
    pub kv_insert: QPadKvRopeMxfp8InsertKernelConfig,
    pub compressor_norm: Option<ElementwiseKernelConfig>,
    pub indexer_wk: Option<ElementwiseKernelConfig>,
    pub indexer_k_store: Option<ElementwiseKernelConfig>,
    pub indexer_wq_b: Option<SingleGemmKernelConfig>,
    pub indexer_q_rope: Option<ElementwiseKernelConfig>,
    pub compressor_insert: Option<ElementwiseKernelConfig>,
    pub indexer: Option<DeepseekV41IndexerOpConfig>,
    pub mega_attn: DeepseekV41MegaAttnOpConfig,
    pub wo_a: BatchedGemmKernelConfig,
    pub wo_b: SingleGemmKernelConfig,
    pub all_reduce: Option<AllReduceFusionKernelConfig>,
}

#[derive(Clone, Debug, Default)]
pub struct DeepseekV41AttentionTpWorkletInput {
    /// Rows the dense spine runs (scheduled tokens, CUDA-graph padding included).
    pub num_tokens: u32,
    /// `(query_len, context_len)` per prefill request.
    pub prefill_query_context_pairs: Vec<(u32, u32)>,
    /// Resident KV length before each one-token decode row.
    pub decode_kv_lens: Vec<u32>,
    /// `attention_aux_stream_live(..)`: the stage-B aux stream exists.
    pub aux_stream_live: bool,
}

pub struct DeepseekV41AttentionTpWorklet {
    pub name: String,
    pub entry_expand: Option<Op<ElementwiseKernel>>,
    pub entry_hc_prenorm: Option<Op<ElementwiseKernel>>,
    pub entry_pre_fuse: Option<Op<ElementwiseKernel>>,
    pub entry_fused: Option<Op<MhcFusedPostPreRmsNormKernel>>,
    pub fused_wqa_wkv: Op<SingleGemmKernel>,
    pub compressor_proj: Option<Op<GemmFp32OutputKernel>>,
    pub indexer_weights_proj: Option<Op<ElementwiseKernel>>,
    pub qk_rmsnorm: Op<ElementwiseKernel>,
    pub wq_b: Op<SingleGemmKernel>,
    pub kv_insert: Op<QPadKvRopeMxfp8InsertKernel>,
    pub compressor_norm: Option<Op<ElementwiseKernel>>,
    pub indexer_wk: Option<Op<ElementwiseKernel>>,
    pub indexer_k_store: Option<Op<ElementwiseKernel>>,
    pub indexer_wq_b: Option<Op<SingleGemmKernel>>,
    pub indexer_q_rope: Option<Op<ElementwiseKernel>>,
    pub compressor_insert: Option<Op<ElementwiseKernel>>,
    pub indexer: Option<DeepseekV41IndexerOp>,
    pub mega_attn: DeepseekV41MegaAttnOp,
    pub wo_a: Op<BatchedGemmKernel>,
    pub wo_b: Op<SingleGemmKernel>,
    pub all_reduce: Option<Op<AllReduceFusionKernel>>,
    resolved: DeepseekV41AttentionTpWorkletResolved,
}

impl DeepseekV41AttentionTpWorklet {
    pub fn resolve_config(
        cfg: &DeepseekV41AttentionTpWorkletConfig,
    ) -> DeepseekV41AttentionTpWorkletResolved {
        let layer = cfg.layer;
        let tp = cfg.tp_size;
        assert!(tp > 0, "tp_size must be positive");
        assert_eq!(
            cfg.num_attention_heads.get() % tp,
            0,
            "attention heads must divide tp_size"
        );
        assert_eq!(cfg.o_groups.get() % tp, 0, "o_groups must divide tp_size");
        assert!(
            layer.compress_ratio <= 2,
            "compress ratio must be 0, 1 or 2"
        );
        assert!(
            layer.compress_ratio > 0 || (!layer.kv_source && layer.index == DeepseekV41IndexRole::None),
            "ratio-0 layers have no compressor or indexer"
        );
        assert_eq!(
            layer.kv_source,
            layer.index == DeepseekV41IndexRole::Owner,
            "V4.1 index owners are exactly the KV sources"
        );
        assert!(
            layer.candidate == DeepseekV41CandidateRole::None
                || layer.index != DeepseekV41IndexRole::None,
            "candidate roles belong to index sources"
        );
        let heads_per_rank = cfg.num_attention_heads.get() / tp;
        let o_groups_per_rank = cfg.o_groups.get() / tp;
        assert!(
            cfg.padded_heads.get() >= heads_per_rank,
            "padded heads must cover the live heads"
        );

        let gpu = cfg.gpu_name.as_str();
        let ew = |input: u32, output: u32| placeholder(&cfg.elementwise_backends, gpu, input, output);
        let mxfp8 = |n: Dim, k: Dim| SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n,
            k,
            dtype: DType::Mxfp8E4m3,
        };
        let hidden = cfg.hidden_size.get();
        let hc_hidden_bytes = cfg.hc_mult * hidden * 2;
        let mix_bytes = (2 + cfg.hc_mult) * cfg.hc_mult * 4;
        let head_dim = cfg.head_dim.get();
        let index_width = cfg.index_num_heads.get() * cfg.index_head_dim.get();
        let ratio = layer.compress_ratio;
        let compressor_width = head_dim * ratio;
        let layer0 = layer.entry == DeepseekV41AttentionEntry::Layer0Pre;
        let index_source = layer.index != DeepseekV41IndexRole::None;
        let owner = layer.index == DeepseekV41IndexRole::Owner;

        DeepseekV41AttentionTpWorkletResolved {
            heads_per_rank,
            o_groups_per_rank,
            // nvidia/model.py:434: `x.unsqueeze(1).expand(-1, hc, -1).contiguous()`.
            entry_expand: layer0.then(|| ew(hidden * 2, hc_hidden_bytes)),
            // nvidia/model.py:435 -> deep_gemm `tf32_hc_prenorm_gemm`: the hc
            // streams in, the fp32 mix logits out.
            entry_hc_prenorm: layer0.then(|| ew(hc_hidden_bytes, mix_bytes)),
            // nvidia/model.py:435 -> `mhc_pre_big_fuse_with_norm`: streams and
            // mixes in, the normalized attention input out.
            entry_pre_fuse: layer0.then(|| ew(hc_hidden_bytes + mix_bytes, hidden * 2)),
            entry_fused: (layer.entry == DeepseekV41AttentionEntry::FusedPostPre).then(|| {
                MhcRmsNormKernelConfig {
                    backends: cfg.mhc_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    hidden_size: cfg.hidden_size.clone(),
                    hc_mult: cfg.hc_mult,
                    hidden_dtype: DType::Bf16,
                }
            }),
            fused_wqa_wkv: mxfp8(
                cfg.q_lora_rank.clone() + cfg.head_dim.clone(),
                cfg.hidden_size.clone(),
            ),
            // attention.py `compressor_kv_score`: torch.mm(out_dtype=fp32).
            compressor_proj: layer.kv_source.then(|| GemmFp32OutputKernelConfig {
                backends: cfg.fp32_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.head_dim.clone() * ratio,
                k: cfg.hidden_size.clone(),
                input_dtype: DType::Bf16,
            }),
            // attention.py:929-931 `indexer.weights_proj` (bf16 ReplicatedLinear,
            // cuBLAS nvjet): no B200 bf16 GEMM rows at n=32, k=5120.
            indexer_weights_proj: index_source
                .then(|| ew(hidden * 2, cfg.index_num_heads.get() * 2)),
            // attention.py:758 `fused_q_kv_rmsnorm` over the q_a and kv halves.
            qk_rmsnorm: ew(
                (cfg.q_lora_rank.get() + head_dim) * 2,
                (cfg.q_lora_rank.get() + head_dim) * 2,
            ),
            wq_b: mxfp8(
                cfg.head_dim.clone() * heads_per_rank,
                cfg.q_lora_rank.clone(),
            ),
            kv_insert: QPadKvRopeMxfp8InsertKernelConfig {
                backends: cfg.kv_insert_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: heads_per_rank,
                padded_heads: cfg.padded_heads.get(),
                block_size: cfg.swa_block_size,
                input_dtype: DType::Bf16,
                swa_cache_format: "mxfp8".to_string(),
            },
            // compressor.py:279 -> common/ops/fused_compress_quant_cache.py:94
            // `_fused_save_compress_norm`: fp32 kv_score in, fp32 state out.
            compressor_norm: layer
                .kv_source
                .then(|| ew(compressor_width * 4, compressor_width * 4)),
            // attention.py:1371 `wk(latent)` (bf16 ReplicatedLinear 512->128):
            // no B200 bf16 GEMM rows at this shape.
            indexer_wk: owner.then(|| ew(head_dim * 2, cfg.index_head_dim.get() * 2)),
            // attention.py:1372 -> common/ops/indexer_k_store.py:95: one
            // 132-byte key per `ratio` tokens.
            indexer_k_store: owner.then(|| {
                ew(
                    cfg.index_head_dim.get() * 2,
                    (cfg.index_head_dim.get() + 4) / ratio.max(1),
                )
            }),
            indexer_wq_b: index_source.then(|| {
                mxfp8(
                    cfg.index_num_heads.clone() * cfg.index_head_dim.clone(),
                    cfg.q_lora_rank.clone(),
                )
            }),
            // attention.py:1427 -> sparse_attn_indexer.py:235
            // `fused_indexer_q_rope_quant`: bf16 q + weights in, FP8 q + fp32
            // weights out.
            indexer_q_rope: index_source.then(|| {
                ew(
                    index_width * 2 + cfg.index_num_heads.get() * 2,
                    index_width + cfg.index_num_heads.get() * 4,
                )
            }),
            // compressor.py:315 -> fused_compress_quant_cache.py:260-285
            // `_rope_quant_insert_nvfp4`: one 288-byte NVFP4 row per `ratio`
            // tokens.
            compressor_insert: layer.kv_source.then(|| ew(head_dim * 2, 288 / ratio.max(1))),
            indexer: index_source.then(|| DeepseekV41IndexerOpConfig {
                gpu_name: cfg.gpu_name.clone(),
                compress_ratio: ratio,
                num_heads: cfg.index_num_heads.clone(),
                head_dim: cfg.index_head_dim.clone(),
                index_topk: cfg.index_topk,
                page_block_size: cfg.kv_block_size / ratio,
                max_model_len: cfg.max_model_len,
                candidate: layer.candidate,
                logits_prefill_backends: cfg.index_logits_prefill_backends.clone(),
                logits_decode_backends: cfg.index_logits_decode_backends.clone(),
                elementwise_backends: cfg.elementwise_backends.clone(),
            }),
            mega_attn: DeepseekV41MegaAttnOpConfig {
                backends: cfg.mega_attn_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                compress_ratio: ratio,
                window_size: cfg.window_size,
                index_topk: cfg.index_topk,
                padded_heads: cfg.padded_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                rope_dim: cfg.rope_dim.clone(),
                max_model_len: cfg.max_model_len,
                max_num_batched_tokens: cfg.max_num_batched_tokens,
                prefill_chunk_size: cfg.prefill_chunk_size,
                q_dtype: DType::Bf16,
                swa_cache_format: "mxfp8".to_string(),
                compressed_cache_format: "nvfp4".to_string(),
                output_dtype: DType::Fp8E4m3,
            },
            wo_a: BatchedGemmKernelConfig {
                backends: cfg.wo_a_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_batches: (cfg.o_groups.clone() / tp),
                n: cfg.o_lora_rank.clone(),
                k: cfg.num_attention_heads.clone() * cfg.head_dim.clone() / cfg.o_groups.clone(),
                dtype: DType::Mxfp8E4m3,
            },
            wo_b: mxfp8(
                cfg.hidden_size.clone(),
                cfg.o_lora_rank.clone() * o_groups_per_rank,
            ),
            all_reduce: (tp > 1).then(|| AllReduceFusionKernelConfig {
                backends: cfg.all_reduce_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_gpus: tp,
                hidden_dim: hidden,
                dtype: DType::Bf16,
                fabric: Fabric::Nvlink,
                fused_token_limit: None,
            }),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: DeepseekV41AttentionTpWorkletResolved,
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
            entry_expand: maybe!(ElementwiseKernel, r.entry_expand, "entry.hc_expand"),
            entry_hc_prenorm: maybe!(ElementwiseKernel, r.entry_hc_prenorm, "entry.mhc_hc_prenorm"),
            entry_pre_fuse: maybe!(ElementwiseKernel, r.entry_pre_fuse, "entry.mhc_pre_big_fuse"),
            entry_fused: maybe!(
                MhcFusedPostPreRmsNormKernel,
                r.entry_fused,
                "entry.mega_mhc_post_pre"
            ),
            fused_wqa_wkv: op!(SingleGemmKernel, r.fused_wqa_wkv, "input.fused_wqa_wkv"),
            compressor_proj: maybe!(
                GemmFp32OutputKernel,
                r.compressor_proj,
                "compressor.kv_score_proj"
            ),
            indexer_weights_proj: maybe!(
                ElementwiseKernel,
                r.indexer_weights_proj,
                "indexer.weights_proj"
            ),
            qk_rmsnorm: op!(ElementwiseKernel, r.qk_rmsnorm, "main.qk_rmsnorm"),
            wq_b: op!(SingleGemmKernel, r.wq_b, "main.wq_b"),
            kv_insert: op!(QPadKvRopeMxfp8InsertKernel, r.kv_insert, "main.kv_insert"),
            compressor_norm: maybe!(
                ElementwiseKernel,
                r.compressor_norm,
                "compressor.save_compress_norm"
            ),
            indexer_wk: maybe!(ElementwiseKernel, r.indexer_wk, "indexer.wk"),
            indexer_k_store: maybe!(ElementwiseKernel, r.indexer_k_store, "indexer.k_store"),
            indexer_wq_b: maybe!(SingleGemmKernel, r.indexer_wq_b, "indexer.wq_b"),
            indexer_q_rope: maybe!(ElementwiseKernel, r.indexer_q_rope, "indexer.q_rope_quant"),
            compressor_insert: maybe!(
                ElementwiseKernel,
                r.compressor_insert,
                "compressor.nvfp4_insert"
            ),
            indexer: match r.indexer {
                Some(config) => Some(DeepseekV41IndexerOp::build(
                    format!("{name}.indexer.score"),
                    DeepseekV41IndexerOp::resolve(&config),
                    bridge,
                )?),
                None => None,
            },
            mega_attn: DeepseekV41MegaAttnOp::build(
                format!("{name}.attention.mega_attn"),
                r.mega_attn,
                bridge,
            )?,
            wo_a: op!(BatchedGemmKernel, r.wo_a, "output.wo_a_einsum"),
            wo_b: op!(SingleGemmKernel, r.wo_b, "output.wo_b"),
            all_reduce: maybe!(AllReduceFusionKernel, r.all_reduce, "tp_all_reduce"),
            name,
            resolved,
        })
    }

    pub fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        let mut children = Vec::new();
        for entry in [&self.entry_expand, &self.entry_hc_prenorm, &self.entry_pre_fuse]
            .into_iter()
            .flatten()
        {
            children.push(entry.compile(b));
        }
        if let Some(entry) = &self.entry_fused {
            children.push(entry.compile(b));
        }

        // Stage A: fused_wqa_wkv | compressor kv_score | indexer weights_proj.
        let main = vec![self.fused_wqa_wkv.compile(b)];
        let mut concurrent = Vec::new();
        if let Some(op) = &self.compressor_proj {
            concurrent.push(op.compile(b));
        }
        if let Some(op) = &self.indexer_weights_proj {
            concurrent.push(op.compile(b));
        }
        let mut serial = Vec::new();
        if let Some(op) = &self.compressor_proj {
            serial.push(compile_serial_copy(op, b));
        }
        if let Some(op) = &self.indexer_weights_proj {
            serial.push(compile_serial_copy(op, b));
        }
        children.push(gated_fanout(main, concurrent, serial));

        children.push(self.qk_rmsnorm.compile(b));

        // Stage B1: wq_b + KV insert | compressor state save.
        let main = vec![self.wq_b.compile(b), self.kv_insert.compile(b)];
        let (concurrent, serial) = match &self.compressor_norm {
            Some(op) => (vec![op.compile(b)], vec![compile_serial_copy(op, b)]),
            None => (Vec::new(), Vec::new()),
        };
        children.push(gated_fanout(main, concurrent, serial));

        // Stage B2: indexer K and Q preparation | compressor NVFP4 insert.
        let mut main = Vec::new();
        for op in [&self.indexer_wk, &self.indexer_k_store].into_iter().flatten() {
            main.push(op.compile(b));
        }
        if let Some(op) = &self.indexer_wq_b {
            main.push(op.compile(b));
        }
        if let Some(op) = &self.indexer_q_rope {
            main.push(op.compile(b));
        }
        let (concurrent, serial) = match &self.compressor_insert {
            Some(op) => (vec![op.compile(b)], vec![compile_serial_copy(op, b)]),
            None => (Vec::new(), Vec::new()),
        };
        if !main.is_empty() || !concurrent.is_empty() {
            children.push(gated_fanout(main, concurrent, serial));
        }

        if let Some(indexer) = &self.indexer {
            children.push(indexer.compile(b));
        }
        children.push(self.mega_attn.compile(b));
        children.push(self.wo_a.compile(b));
        children.push(self.wo_b.compile(b));
        if let Some(all_reduce) = &self.all_reduce {
            children.push(all_reduce.compile(b));
        }

        let layer = self.resolved.raw_cfg.layer;
        CostNode::Labeled {
            label: format!(
                "{} (DeepseekV41AttentionTpWorklet) [tp={}; heads/rank={}; ratio={}; \
                 kv_source={}; index={:?}; candidate={:?}; entry={:?}]",
                self.name,
                self.resolved.raw_cfg.tp_size,
                self.resolved.heads_per_rank,
                layer.compress_ratio,
                layer.kv_source,
                layer.index,
                layer.candidate,
                layer.entry,
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &DeepseekV41AttentionTpWorkletInput, ev: &mut Evaluator) {
        validate_input(input, &self.resolved.raw_cfg)
            .unwrap_or_else(|reason| panic!("invalid DeepseekV41AttentionTpWorkletInput: {reason}"));
        let rows = input.num_tokens;
        let zero = rows == 0;
        let gates = stream_gates(input, &self.resolved.raw_cfg);
        let ew = ElementwiseKernelInput { num_tokens: rows };
        let gemm = SingleGemmKernelInput { m: rows };

        for entry in [&self.entry_expand, &self.entry_hc_prenorm, &self.entry_pre_fuse]
            .into_iter()
            .flatten()
        {
            eval_or_zero(entry, ew.clone(), zero, ev);
        }
        if let Some(entry) = &self.entry_fused {
            eval_or_zero(entry, MhcRmsNormKernelInput { num_tokens: rows }, zero, ev);
        }

        eval_or_zero(&self.fused_wqa_wkv, gemm.clone(), zero, ev);
        for copy_overlapped in [true, false] {
            let off = zero || gates.input_projections != copy_overlapped;
            if let Some(op) = &self.compressor_proj {
                eval_or_zero(op, GemmFp32OutputKernelInput { m: rows }, off, ev);
            }
            if let Some(op) = &self.indexer_weights_proj {
                eval_or_zero(op, ew.clone(), off, ev);
            }
        }

        eval_or_zero(&self.qk_rmsnorm, ew.clone(), zero, ev);

        eval_or_zero(&self.wq_b, gemm.clone(), zero, ev);
        eval_or_zero(
            &self.kv_insert,
            QPadKvRopeMxfp8InsertKernelInput { num_tokens: rows },
            zero,
            ev,
        );
        if let Some(op) = &self.compressor_norm {
            eval_or_zero(op, ew.clone(), zero || !gates.aux_stream, ev);
            eval_or_zero(op, ew.clone(), zero || gates.aux_stream, ev);
        }

        for op in [&self.indexer_wk, &self.indexer_k_store].into_iter().flatten() {
            eval_or_zero(op, ew.clone(), zero, ev);
        }
        if let Some(op) = &self.indexer_wq_b {
            eval_or_zero(op, gemm.clone(), zero, ev);
        }
        if let Some(op) = &self.indexer_q_rope {
            eval_or_zero(op, ew.clone(), zero, ev);
        }
        if let Some(op) = &self.compressor_insert {
            eval_or_zero(op, ew.clone(), zero || !gates.aux_stream, ev);
            eval_or_zero(op, ew.clone(), zero || gates.aux_stream, ev);
        }

        if let Some(indexer) = &self.indexer {
            indexer.eval(
                &DeepseekV41IndexerOpInput {
                    prefill_query_context_pairs: input.prefill_query_context_pairs.clone(),
                    decode_kv_lens: input.decode_kv_lens.clone(),
                },
                ev,
            );
        }
        self.mega_attn.eval(
            &DeepseekV41MegaAttnOpInput {
                prefill_query_context_pairs: input.prefill_query_context_pairs.clone(),
                decode_kv_lens: input.decode_kv_lens.clone(),
            },
            ev,
        );
        eval_or_zero(&self.wo_a, BatchedGemmKernelInput { m: rows }, zero, ev);
        eval_or_zero(&self.wo_b, gemm, zero, ev);
        if let Some(all_reduce) = &self.all_reduce {
            eval_or_zero(
                all_reduce,
                AllReduceFusionKernelInput { num_tokens: rows },
                zero,
                ev,
            );
        }
    }
}

/// Which gated side branches overlap on this call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StreamGates {
    /// Stage A: compressor / weights_proj beside `fused_wqa_wkv`.
    pub input_projections: bool,
    /// Stages B1/B2: compressor save and insert beside the main path.
    pub aux_stream: bool,
}

pub(crate) fn stream_gates(
    input: &DeepseekV41AttentionTpWorkletInput,
    cfg: &DeepseekV41AttentionTpWorkletConfig,
) -> StreamGates {
    let streams = !cfg.serialize_streams && input.num_tokens > 0;
    StreamGates {
        input_projections: streams && input.num_tokens <= MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
        aux_stream: streams && input.aux_stream_live,
    }
}

fn validate_input(
    input: &DeepseekV41AttentionTpWorkletInput,
    cfg: &DeepseekV41AttentionTpWorkletConfig,
) -> Result<(), String> {
    let mut active = u32::try_from(input.decode_kv_lens.len())
        .map_err(|_| "decode row count exceeds u32".to_string())?;
    for &kv in &input.decode_kv_lens {
        if kv >= cfg.max_model_len {
            return Err(format!("decode context {} exceeds max_model_len", kv + 1));
        }
    }
    for &(queries, context) in &input.prefill_query_context_pairs {
        if queries == 0 || queries > context || context > cfg.max_model_len {
            return Err(format!("invalid prefill pair ({queries}, {context})"));
        }
        active = active
            .checked_add(queries)
            .ok_or_else(|| "active rows overflow u32".to_string())?;
    }
    if active > input.num_tokens {
        return Err(format!(
            "{active} active rows exceed num_tokens {}",
            input.num_tokens
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn config(layer: u32) -> DeepseekV41AttentionTpWorkletConfig {
        DeepseekV41AttentionTpWorkletConfig {
            layer: production_layer(layer),
            tp_size: 4,
            serialize_streams: false,
            hidden_size: 5120.into(),
            hc_mult: 4,
            num_attention_heads: 64.into(),
            padded_heads: 64.into(),
            head_dim: 512.into(),
            rope_dim: 64.into(),
            q_lora_rank: 1280.into(),
            o_lora_rank: 1024.into(),
            o_groups: 8.into(),
            index_num_heads: 32.into(),
            index_head_dim: 128.into(),
            index_topk: 512,
            window_size: 128,
            kv_block_size: 128,
            swa_block_size: 32,
            max_model_len: 131_072,
            max_num_batched_tokens: 2048,
            prefill_chunk_size: 4,
            gpu_name: "NVIDIA B200".into(),
            mhc_backends: vec!["deepgemm_mega"],
            gemm_backends: vec!["flashinfer_mxfp8"],
            fp32_gemm_backends: vec!["torch_cublas"],
            kv_insert_backends: vec!["vllm_cuda"],
            mega_attn_backends: vec!["flashmla_mega"],
            wo_a_backends: vec!["deepgemm_mxfp8_einsum_grouped_o_proj"],
            all_reduce_backends: vec!["flashinfer_mnnvl"],
            index_logits_prefill_backends: vec!["deepgemm_fp8"],
            index_logits_decode_backends: vec!["deepgemm_fp8"],
            elementwise_backends: vec!["triton"],
        }
    }

    #[test]
    fn production_schedule_matches_the_checkpoint_layer_types() {
        let ratio: Vec<u32> = (0..40).map(|l| production_layer(l).compress_ratio).collect();
        assert_eq!(ratio.iter().filter(|&&r| r == 0).count(), 2);
        assert_eq!(ratio.iter().filter(|&&r| r == 2).count(), 18);
        assert_eq!(ratio.iter().filter(|&&r| r == 1).count(), 20);
        assert_eq!(production_layer(0).entry, DeepseekV41AttentionEntry::Layer0Pre);
        assert_eq!(production_layer(14).entry, DeepseekV41AttentionEntry::AfterEngram);
        assert_eq!(production_layer(14).index, DeepseekV41IndexRole::Owner);
        assert_eq!(production_layer(20).candidate, DeepseekV41CandidateRole::Writer);
        assert_eq!(production_layer(28).index, DeepseekV41IndexRole::NonOwner);
        assert_eq!(production_layer(28).candidate, DeepseekV41CandidateRole::Consumer);
        assert_eq!(production_layer(21).index, DeepseekV41IndexRole::None);
    }

    #[test]
    fn tp4_partition_bakes_the_captured_gemm_shapes() {
        let r = DeepseekV41AttentionTpWorklet::resolve_config(&config(2));
        let nk = |c: &SingleGemmKernelConfig| (c.k.get(), c.n.get());
        assert_eq!(r.heads_per_rank, 16);
        assert_eq!(nk(&r.fused_wqa_wkv), (5120, 1792));
        assert_eq!(nk(&r.wq_b), (1280, 8192));
        assert_eq!(nk(r.indexer_wq_b.as_ref().unwrap()), (1280, 4096));
        assert_eq!(nk(&r.wo_b), (2048, 5120));
        assert_eq!(
            (r.wo_a.num_batches.get(), r.wo_a.k.get(), r.wo_a.n.get()),
            (2, 4096, 1024)
        );
        assert_eq!(r.compressor_proj.as_ref().unwrap().n.get(), 1024);
        assert_eq!(
            DeepseekV41AttentionTpWorklet::resolve_config(&config(20))
                .compressor_proj
                .unwrap()
                .n
                .get(),
            512
        );
        assert_eq!((r.kv_insert.num_heads, r.kv_insert.padded_heads), (16, 64));
        assert_eq!(r.indexer.as_ref().unwrap().page_block_size, 64);
    }

    #[test]
    #[should_panic(expected = "attention heads must divide tp_size")]
    fn indivisible_tp_is_rejected() {
        let mut cfg = config(3);
        cfg.tp_size = 3;
        DeepseekV41AttentionTpWorklet::resolve_config(&cfg);
    }

    #[test]
    fn stream_gates_follow_the_production_thresholds() {
        let cfg = config(2);
        let input = |num_tokens, aux_stream_live| DeepseekV41AttentionTpWorkletInput {
            num_tokens,
            aux_stream_live,
            ..Default::default()
        };
        assert_eq!(
            stream_gates(&input(48, true), &cfg),
            StreamGates {
                input_projections: true,
                aux_stream: true
            }
        );
        // Mixed 288 (T=174): stage A overlapped, stage B serial.
        assert_eq!(
            stream_gates(&input(174, false), &cfg),
            StreamGates {
                input_projections: true,
                aux_stream: false
            }
        );
        // Mixed 310 (T=2048): everything serial.
        assert_eq!(
            stream_gates(&input(2048, false), &cfg),
            StreamGates {
                input_projections: false,
                aux_stream: false
            }
        );
        let mut serial = cfg.clone();
        serial.serialize_streams = true;
        assert_eq!(
            stream_gates(&input(48, true), &serial),
            StreamGates {
                input_projections: false,
                aux_stream: false
            }
        );
    }

    #[test]
    fn padding_may_exceed_active_rows_but_not_the_reverse() {
        let cfg = config(2);
        let ok = DeepseekV41AttentionTpWorkletInput {
            num_tokens: 56,
            prefill_query_context_pairs: vec![(4, 16)],
            decode_kv_lens: vec![7; 48],
            aux_stream_live: false,
        };
        assert!(validate_input(&ok, &cfg).is_ok());
        let bad = DeepseekV41AttentionTpWorkletInput {
            num_tokens: 50,
            ..ok
        };
        assert!(validate_input(&bad, &cfg).is_err());
    }
}
