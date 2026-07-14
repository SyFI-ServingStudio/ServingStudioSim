//! `AttnBlockTpWorklet` — tensor-parallel attention block of a dense decoder
//! layer: input RMSNorm → column-parallel fused QKV → attention over per-rank
//! heads → row-parallel o_proj → optional `tp_allreduce`. `TP` group suffix
//! (L3 §1.5): the block is one sync section whose boundary is the all-reduce
//! that sums the row-parallel o_proj output across the `tp_size` ranks.
//!
//! Megatron TP distributes the attention heads: QKV is column-parallel (each
//! rank owns `num_qo_heads / tp` query heads and `num_kv_heads / tp` KV heads),
//! attention runs on those local heads, and the row-parallel o_proj's full
//! `[tokens × hidden]` partial-sum is re-synced with the all-reduce. `hidden` is
//! NOT sharded (qkv input k=hidden, o_proj output n=hidden).
//!
//! `tp_size == 1` degenerates to the single-GPU case: per-rank == full, and the
//! `tp_ar` slot is `None` (no collective), so the cost matches the `Local` path.

use std::sync::Arc;

use crate::common::Fabric;
use crate::op::attention::{
    FlashInferAttentionConfig, FlashInferAttentionInput, FlashInferAttentionOp,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput, RmsNormKernel,
    RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

/// Raw global config + TP degree + the collective fabric/backends. Partition is
/// derived in `resolve_config`.
#[derive(Clone, Debug)]
pub struct AttnBlockTpWorkletConfig {
    pub hidden: Dim,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    /// Base (16-bit) dtype — RMSNorm + attention output/decode-query keep it.
    pub dtype: DType,
    /// FP8 run: qkv / o_proj GEMMs + tp_allreduce move to fp8 (via `gemm_backends`
    /// = deepgemm); the attention op derives its prefill/decode fp8 presets.
    pub fp8: bool,
    pub tp_size: u16,
    /// Symbol name for `tp_size` in the derivation formula (`attn_tp`/`tp`) — the
    /// arch owns which sharding degree this worklet's `tp` is.
    pub tp_name: &'static str,
    pub allreduce_fabric: Fabric,
    pub gpu_name: String,
    pub norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub attn_backends: Vec<&'static str>,
    pub kv_cache_append_backends: Vec<&'static str>,
    pub kv_cache_block_size: u32,
    pub kv_cache_layout: String,
    pub kv_scale_granularity: String,
    pub allreduce_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct AttnBlockTpWorkletResolved {
    pub raw_cfg: AttnBlockTpWorkletConfig,
    pub input_norm: RmsNormKernelConfig,
    pub qkv: SingleGemmKernelConfig,
    pub attn: FlashInferAttentionConfig,
    pub o_proj: SingleGemmKernelConfig,
    pub tp_ar: Option<AllReduceKernelConfig>, // tp_size == 1 → None
    pub num_qo_heads_per_rank: Dim,
    pub num_kv_heads_per_rank: Dim,
    pub dtype_bytes: u32,
}

/// Per-call shape: `batch_tokens` drives the GEMMs/norm/allreduce; the prefill
/// `(prefix_len, append_len)` pairs and decode KV lengths drive attention.
#[derive(Clone, Debug, Default)]
pub struct AttnBlockTpWorkletInput {
    pub batch_tokens: u32,
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
    pub decode_kv_lens: Vec<u32>,
}

pub struct AttnBlockTpWorklet {
    pub name: String,
    pub input_norm: Op<RmsNormKernel>,
    pub qkv: Op<SingleGemmKernel>,
    pub attn: FlashInferAttentionOp,
    pub o_proj: Op<SingleGemmKernel>,
    pub tp_ar: Option<Op<AllReduceKernel>>,
    resolved: AttnBlockTpWorkletResolved,
}

impl AttnBlockTpWorkletConfig {
    /// KV cache dtype: fp8 in an fp8 run (both prefill and decode read fp8 KV),
    /// else the base dtype. Used by the arch's KV-byte accounting (`raw_cfg`).
    pub fn kv_dtype(&self) -> DType {
        if self.fp8 {
            DType::Fp8E4m3
        } else {
            self.dtype
        }
    }
}

impl AttnBlockTpWorklet {
    pub fn resolve_config(cfg: &AttnBlockTpWorkletConfig) -> AttnBlockTpWorkletResolved {
        let tp = cfg.tp_size as u32;
        // GQA dual-divisibility: both head counts split across the TP ranks.
        // tp <= num_kv_heads (no KV-head replication in v1).
        assert!(
            cfg.num_qo_heads.get() % tp == 0,
            "num_qo_heads {} not divisible by tp_size {}",
            cfg.num_qo_heads,
            tp
        );
        assert!(
            cfg.num_kv_heads.get() % tp == 0,
            "num_kv_heads {} not divisible by tp_size {}",
            cfg.num_kv_heads,
            tp
        );
        assert!(
            tp <= cfg.num_kv_heads.get(),
            "tp_size {} exceeds num_kv_heads {} (KV-head replication unsupported)",
            tp,
            cfg.num_kv_heads
        );
        let tp_dim = Dim::param(cfg.tp_name, tp);
        let qo_pr = cfg.num_qo_heads.clone() / tp_dim.clone();
        let kv_pr = cfg.num_kv_heads.clone() / tp_dim;
        // Compute dtype: FP8 for the GEMMs + the tp_allreduce message (halved
        // transfer); RMSNorm and the attention output keep the base `dtype`.
        let compute = if cfg.fp8 { DType::Fp8E4m3 } else { cfg.dtype };
        AttnBlockTpWorkletResolved {
            input_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            qkv: SingleGemmKernelConfig {
                // column-parallel: per-rank fused QKV output = (qo + 2·kv)/tp heads.
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: (qo_pr.clone() + 2 * kv_pr.clone()) * cfg.head_dim.clone(),
                k: cfg.hidden.clone(),
                dtype: compute,
            },
            attn: FlashInferAttentionConfig {
                backends: cfg.attn_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_qo_heads: qo_pr.clone(),
                num_kv_heads: kv_pr.clone(),
                head_dim: cfg.head_dim.clone(),
                dtype: cfg.dtype,
                fp8: cfg.fp8,
                kv_cache_append_backends: cfg.kv_cache_append_backends.clone(),
                kv_cache_block_size: cfg.kv_cache_block_size,
                kv_cache_layout: cfg.kv_cache_layout.clone(),
                kv_scale_granularity: cfg.kv_scale_granularity.clone(),
            },
            o_proj: SingleGemmKernelConfig {
                // row-parallel: input k = per-rank Q heads × head_dim; output n = hidden.
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden.clone(),
                k: qo_pr.clone() * cfg.head_dim.clone(),
                dtype: compute,
            },
            tp_ar: (cfg.tp_size > 1).then(|| AllReduceKernelConfig {
                // Comm is size-keyed: the all-reduce reads the fp8-width message
                // bytes (`dtype_bytes` below) off a dtype-agnostic curve.
                backends: cfg.allreduce_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_gpus: cfg.tp_size as u32,
                fabric: cfg.allreduce_fabric,
            }),
            num_qo_heads_per_rank: qo_pr,
            num_kv_heads_per_rank: kv_pr,
            dtype_bytes: compute.size_bytes(),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: AttnBlockTpWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let norm_name = format!("{name}.input_norm");
        let qkv_name = format!("{name}.qkv_proj");
        let o_name = format!("{name}.o_proj");
        let input_norm = Op::new(
            norm_name.clone(),
            Arc::new(RmsNormKernel::build(
                norm_name,
                resolved.input_norm.clone(),
                bridge,
            )?),
        );
        let qkv = Op::new(
            qkv_name.clone(),
            Arc::new(SingleGemmKernel::build(
                qkv_name,
                resolved.qkv.clone(),
                bridge,
            )?),
        );
        let attn =
            FlashInferAttentionOp::build(format!("{name}.attn"), resolved.attn.clone(), bridge)?;
        let o_proj = Op::new(
            o_name.clone(),
            Arc::new(SingleGemmKernel::build(
                o_name,
                resolved.o_proj.clone(),
                bridge,
            )?),
        );
        let tp_ar = match &resolved.tp_ar {
            Some(ar_cfg) => {
                let ar_name = format!("{name}.tp_allreduce");
                Some(Op::new(
                    ar_name.clone(),
                    Arc::new(AllReduceKernel::build(ar_name, ar_cfg.clone(), bridge)?),
                ))
            }
            None => None,
        };
        Ok(Self {
            name,
            input_norm,
            qkv,
            attn,
            o_proj,
            tp_ar,
            resolved,
        })
    }

    /// CostTree compile: sum input_norm + qkv + attn (append/prefill/decode leaves)
    /// + o_proj + optional tp_allreduce, wrapped in a `Labeled` partition header.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let r = &self.resolved;
        let label = format!(
            "{} (AttnBlockTpWorklet) [tp={}; qo {:?}, kv {:?}, {:?}]",
            self.name,
            r.raw_cfg.tp_size,
            r.num_qo_heads_per_rank,
            r.num_kv_heads_per_rank,
            r.raw_cfg.head_dim,
        );
        let mut parts = vec![
            self.input_norm.compile(builder),
            self.qkv.compile(builder),
            self.attn.compile(builder),
            self.o_proj.compile(builder),
        ];
        if let Some(tp_ar) = &self.tp_ar {
            parts.push(tp_ar.compile(builder));
        }
        CostNode::Labeled {
            label,
            child: Box::new(CostNode::Sum(parts)),
        }
    }

    /// CostTree eval: fill slots in the exact `compile` child order so the
    /// evaluator cursor stays aligned with the minted slot indices.
    pub fn eval(&self, input: &AttnBlockTpWorkletInput, ev: &mut Evaluator) {
        let m = input.batch_tokens;
        self.input_norm.eval(&RmsNormKernelInput { m }, ev);
        self.qkv.eval(&SingleGemmKernelInput { m }, ev);
        self.attn.eval(
            &FlashInferAttentionInput {
                prefill_chunk_pairs: input.prefill_chunk_pairs.clone(),
                decode_kv_lens: input.decode_kv_lens.clone(),
            },
            ev,
        );
        self.o_proj.eval(&SingleGemmKernelInput { m }, ev);
        if let Some(tp_ar) = &self.tp_ar {
            // All-reduce the FULL [tokens × hidden] o_proj partial-sum (see
            // AllReduceKernelInput: message is the complete output, not hidden/tp).
            let message_size_bytes = (m as u64)
                * (self.resolved.raw_cfg.hidden.get() as u64)
                * (self.resolved.dtype_bytes as u64);
            tp_ar.eval(&AllReduceKernelInput { message_size_bytes }, ev);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(tp_size: u16) -> AttnBlockTpWorkletConfig {
        AttnBlockTpWorkletConfig {
            hidden: 4096.into(),
            num_qo_heads: 32.into(),
            num_kv_heads: 8.into(),
            head_dim: 128.into(),
            dtype: DType::Bf16,
            fp8: false,
            tp_size,
            tp_name: "tp",
            allreduce_fabric: Fabric::Nvlink,
            gpu_name: "H100".to_string(),
            norm_backends: vec!["flashinfer"],
            gemm_backends: vec!["torch"],
            attn_backends: vec!["fa2", "fa3"],
            kv_cache_append_backends: vec!["vllm_cuda"],
            kv_cache_block_size: 16,
            kv_cache_layout: "NHD".to_string(),
            kv_scale_granularity: "tensor".to_string(),
            allreduce_backends: vec!["nccl"],
        }
    }

    #[test]
    fn tp1_is_degenerate_full_shapes_no_collective() {
        let r = AttnBlockTpWorklet::resolve_config(&cfg(1));
        // (32 + 2·8)·128 = 6144 fused QKV; o_proj k = 32·128 = 4096.
        assert_eq!(r.qkv.n, 6144);
        assert_eq!(r.qkv.k, 4096);
        assert_eq!(r.o_proj.n, 4096);
        assert_eq!(r.o_proj.k, 4096);
        assert_eq!(r.attn.num_qo_heads, 32);
        assert_eq!(r.attn.num_kv_heads, 8);
        assert!(r.tp_ar.is_none(), "tp=1 must have no collective");
    }

    #[test]
    fn tp4_partitions_heads_and_adds_allreduce() {
        let r = AttnBlockTpWorklet::resolve_config(&cfg(4));
        // per-rank: qo 32/4=8, kv 8/4=2 → fused (8 + 2·2)·128 = 1536.
        assert_eq!(r.num_qo_heads_per_rank, 8);
        assert_eq!(r.num_kv_heads_per_rank, 2);
        assert_eq!(r.qkv.n, 1536);
        assert_eq!(r.qkv.k, 4096); // hidden NOT sharded
        assert_eq!(r.o_proj.n, 4096); // hidden NOT sharded
        assert_eq!(r.o_proj.k, 8 * 128); // per-rank Q heads
        assert_eq!(r.attn.num_qo_heads, 8);
        assert_eq!(r.attn.num_kv_heads, 2);
        let ar = r.tp_ar.expect("tp>1 must add allreduce");
        assert_eq!(ar.num_gpus, 4);
        assert_eq!(ar.fabric, Fabric::Nvlink);
    }

    #[test]
    fn fp8_moves_gemms_to_compute_dtype_but_norm_stays_base() {
        let mut c = cfg(4);
        c.fp8 = true;
        c.gemm_backends = vec!["deepgemm"];
        let r = AttnBlockTpWorklet::resolve_config(&c);
        // GEMMs + allreduce byte width go fp8; RMSNorm keeps the base bf16.
        assert_eq!(r.qkv.dtype, DType::Fp8E4m3);
        assert_eq!(r.o_proj.dtype, DType::Fp8E4m3);
        assert_eq!(r.input_norm.dtype, DType::Bf16);
        assert_eq!(r.dtype_bytes, DType::Fp8E4m3.size_bytes());
        // The attention op carries the base dtype + fp8 flag (it owns the preset).
        assert_eq!(r.attn.dtype, DType::Bf16);
        assert!(r.attn.fp8);
    }

    #[test]
    #[should_panic(expected = "num_kv_heads")]
    fn tp_indivisible_kv_heads_panics() {
        // 8 kv heads, tp=16 → 8 % 16 != 0 (and tp > kv).
        let _ = AttnBlockTpWorklet::resolve_config(&cfg(16));
    }
}
