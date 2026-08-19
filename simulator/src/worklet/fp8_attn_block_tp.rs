//! `Fp8AttnBlockTpWorklet` — tensor-parallel attention block of a dense decoder
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
use crate::op::gemm::{
    SingleFp8GemmWithQuantConfig, SingleFp8GemmWithQuantInput, SingleFp8GemmWithQuantOp,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput,
    Fp8PerTokenGroupQuantKernelConfig, RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput,
    SingleGemmKernelConfig,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

/// Raw global config + TP degree + the collective fabric/backends. Partition is
/// derived in `resolve_config`.
#[derive(Clone, Debug)]
pub struct Fp8AttnBlockTpWorkletConfig {
    pub hidden: Dim,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    /// Base (16-bit) dtype — RMSNorm + attention output/decode-query keep it.
    pub dtype: DType,
    pub tp_size: u16,
    /// Symbol name for `tp_size` in the derivation formula (`attn_tp`/`tp`) — the
    /// arch owns which sharding degree this worklet's `tp` is.
    pub tp_name: &'static str,
    pub allreduce_fabric: Fabric,
    /// Wire dtype of the row-parallel o_proj partial sum. This is deliberately
    /// independent from the GEMM compute dtype: FP8 GEMMs commonly accumulate
    /// and communicate BF16 outputs.
    pub allreduce_dtype: DType,
    pub gpu_name: String,
    pub norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub fp8_quant_backends: Vec<&'static str>,
    pub attn_backends: Vec<&'static str>,
    pub kv_cache_append_backends: Vec<&'static str>,
    pub kv_cache_block_size: u32,
    pub kv_cache_layout: String,
    pub kv_scale_granularity: String,
    pub allreduce_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Fp8AttnBlockTpWorkletResolved {
    pub raw_cfg: Fp8AttnBlockTpWorkletConfig,
    pub input_norm: RmsNormKernelConfig,
    pub qkv: SingleFp8GemmWithQuantConfig,
    pub attn: FlashInferAttentionConfig,
    pub o_proj: SingleFp8GemmWithQuantConfig,
    pub tp_ar: Option<AllReduceKernelConfig>, // tp_size == 1 → None
    pub num_qo_heads_per_rank: Dim,
    pub num_kv_heads_per_rank: Dim,
    pub allreduce_dtype_bytes: u32,
}

/// Per-call shape: `batch_tokens` drives the GEMMs/norm/allreduce; the prefill
/// `(prefix_len, append_len)` pairs and decode KV lengths drive attention.
#[derive(Clone, Debug, Default)]
pub struct Fp8AttnBlockTpWorkletInput {
    pub batch_tokens: u32,
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
    pub decode_kv_lens: Vec<u32>,
}

pub struct Fp8AttnBlockTpWorklet {
    pub name: String,
    pub input_norm: Op<RmsNormKernel>,
    pub qkv: SingleFp8GemmWithQuantOp,
    pub attn: FlashInferAttentionOp,
    pub o_proj: SingleFp8GemmWithQuantOp,
    pub tp_ar: Option<Op<AllReduceKernel>>,
    resolved: Fp8AttnBlockTpWorkletResolved,
}

impl Fp8AttnBlockTpWorkletConfig {
    /// KV cache dtype: fp8 in an fp8 run (both prefill and decode read fp8 KV),
    /// else the base dtype. Used by the arch's KV-byte accounting (`raw_cfg`).
    pub fn kv_dtype(&self) -> DType {
        DType::Fp8E4m3
    }
}

impl Fp8AttnBlockTpWorklet {
    pub fn resolve_config(cfg: &Fp8AttnBlockTpWorkletConfig) -> Fp8AttnBlockTpWorkletResolved {
        let tp = cfg.tp_size as u32;
        // GQA dual-divisibility: both head counts split across the TP ranks.
        // tp <= num_kv_heads (no KV-head replication in v1).
        assert!(
            cfg.num_qo_heads.get().is_multiple_of(tp),
            "num_qo_heads {} not divisible by tp_size {}",
            cfg.num_qo_heads,
            tp
        );
        assert!(
            cfg.num_kv_heads.get().is_multiple_of(tp),
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
        let compute = DType::Fp8E4m3;
        let qkv_gemm = SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: (qo_pr.clone() + 2 * kv_pr.clone()) * cfg.head_dim.clone(),
            k: cfg.hidden.clone(),
            dtype: compute,
        };
        let o_proj_gemm = SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: cfg.hidden.clone(),
            k: qo_pr.clone() * cfg.head_dim.clone(),
            dtype: compute,
        };
        let quant_config = |hidden_size: Dim| Fp8PerTokenGroupQuantKernelConfig {
            backends: cfg.fp8_quant_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden_size,
            group_size: 128,
            input_dtype: cfg.dtype,
            scale_format: "ue8m0_column_major".to_string(),
        };
        Fp8AttnBlockTpWorkletResolved {
            input_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            qkv: SingleFp8GemmWithQuantConfig {
                quant: quant_config(cfg.hidden.clone()),
                gemm: qkv_gemm,
            },
            attn: FlashInferAttentionConfig {
                backends: cfg.attn_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_qo_heads: qo_pr.clone(),
                num_kv_heads: kv_pr.clone(),
                head_dim: cfg.head_dim.clone(),
                dtype: cfg.dtype,
                fp8: true,
                kv_cache_append_backends: cfg.kv_cache_append_backends.clone(),
                kv_cache_block_size: cfg.kv_cache_block_size,
                kv_cache_layout: cfg.kv_cache_layout.clone(),
                kv_scale_granularity: cfg.kv_scale_granularity.clone(),
            },
            o_proj: SingleFp8GemmWithQuantConfig {
                quant: quant_config(qo_pr.clone() * cfg.head_dim.clone()),
                gemm: o_proj_gemm,
            },
            tp_ar: (cfg.tp_size > 1).then(|| AllReduceKernelConfig {
                // Comm is size-keyed: the all-reduce reads the explicitly wired
                // message width off a dtype-agnostic curve.
                backends: cfg.allreduce_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_gpus: cfg.tp_size as u32,
                fabric: cfg.allreduce_fabric,
            }),
            num_qo_heads_per_rank: qo_pr,
            num_kv_heads_per_rank: kv_pr,
            allreduce_dtype_bytes: cfg.allreduce_dtype.size_bytes(),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Fp8AttnBlockTpWorkletResolved,
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
        let qkv = SingleFp8GemmWithQuantOp::build(qkv_name, resolved.qkv.clone(), bridge)?;
        let attn =
            FlashInferAttentionOp::build(format!("{name}.attn"), resolved.attn.clone(), bridge)?;
        let o_proj = SingleFp8GemmWithQuantOp::build(o_name, resolved.o_proj.clone(), bridge)?;
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
            "{} (Fp8AttnBlockTpWorklet) [tp={}; qo {:?}, kv {:?}, {:?}]",
            self.name,
            r.raw_cfg.tp_size,
            r.num_qo_heads_per_rank,
            r.num_kv_heads_per_rank,
            r.raw_cfg.head_dim,
        );
        let mut parts = vec![self.input_norm.compile(builder), self.qkv.compile(builder)];
        parts.push(self.attn.compile(builder));
        parts.push(self.o_proj.compile(builder));
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
    pub fn eval(&self, input: &Fp8AttnBlockTpWorkletInput, ev: &mut Evaluator) {
        let m = input.batch_tokens;
        self.input_norm.eval(&RmsNormKernelInput { m }, ev);
        self.qkv
            .eval(&SingleFp8GemmWithQuantInput { num_tokens: m }, ev);
        self.attn.eval(
            &FlashInferAttentionInput {
                prefill_chunk_pairs: input.prefill_chunk_pairs.clone(),
                decode_kv_lens: input.decode_kv_lens.clone(),
            },
            ev,
        );
        self.o_proj
            .eval(&SingleFp8GemmWithQuantInput { num_tokens: m }, ev);
        let message_size_bytes = (m as u64)
            * (self.resolved.raw_cfg.hidden.get() as u64)
            * (self.resolved.allreduce_dtype_bytes as u64);
        if let Some(tp_ar) = &self.tp_ar {
            let ar_input = AllReduceKernelInput { message_size_bytes };
            tp_ar.eval(&ar_input, ev);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_only_fp8_projection_ops_with_native_boundary() {
        let resolved = Fp8AttnBlockTpWorklet::resolve_config(&Fp8AttnBlockTpWorkletConfig {
            hidden: 4096.into(),
            num_qo_heads: 32.into(),
            num_kv_heads: 8.into(),
            head_dim: 128.into(),
            dtype: DType::Bf16,
            tp_size: 4,
            tp_name: "tp",
            allreduce_fabric: Fabric::Nvlink,
            allreduce_dtype: DType::Bf16,
            gpu_name: "NVIDIA H200".into(),
            norm_backends: vec!["flashinfer"],
            gemm_backends: vec!["deepgemm"],
            fp8_quant_backends: vec!["vllm_cuda"],
            attn_backends: vec!["fa2", "fa3"],
            kv_cache_append_backends: vec!["vllm_cuda"],
            kv_cache_block_size: 16,
            kv_cache_layout: "NHD".into(),
            kv_scale_granularity: "tensor".into(),
            allreduce_backends: vec!["nvshmem"],
        });
        assert_eq!(resolved.qkv.gemm.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.o_proj.gemm.dtype, DType::Fp8E4m3);
        assert!(resolved.attn.fp8);
        assert!(resolved.tp_ar.is_some());
    }
}
