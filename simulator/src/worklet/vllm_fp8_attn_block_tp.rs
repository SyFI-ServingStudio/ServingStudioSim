//! `VllmFp8AttnBlockTpWorklet` — tensor-parallel attention block of a dense decoder
//! layer: input `RMSNorm` → column-parallel fused QKV → attention over per-rank
//! heads → row-parallel `o_proj` → optional `tp_allreduce`. `TP` group suffix
//! (L3 §1.5): the block is one sync section whose boundary is the all-reduce
//! that sums the row-parallel `o_proj` output across the `tp_size` ranks.
//!
//! Megatron TP distributes the attention heads: QKV is column-parallel (each
//! rank owns `num_qo_heads / tp` query heads and `num_kv_heads / tp` KV heads),
//! attention runs on those local heads, and the row-parallel `o_proj`'s full
//! `[tokens × hidden]` partial-sum is re-synced with the all-reduce. `hidden` is
//! NOT sharded (qkv input k=hidden, `o_proj` output n=hidden).
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
    AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput, AllReduceResidualRmsNormKernel,
    AllReduceResidualRmsNormKernelConfig, AllReduceResidualRmsNormKernelInput,
    AllReduceResidualRmsNormSpec, Fp8PerTokenGroupQuantKernelConfig, RmsNormKernel,
    RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernelConfig,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge,
};

/// Raw global config + TP degree + the collective fabric/backends. Partition is
/// derived in `resolve_config`.
#[derive(Clone, Debug)]
pub struct VllmFp8AttnBlockTpWorkletConfig {
    pub hidden: Dim,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    /// Base (16-bit) dtype — `RMSNorm` + attention output/decode-query keep it.
    pub dtype: DType,
    pub tp_size: u16,
    /// Symbol name for `tp_size` in the derivation formula (`attn_tp`/`tp`) — the
    /// arch owns which sharding degree this worklet's `tp` is.
    pub tp_name: &'static str,
    pub allreduce_fabric: Fabric,
    /// Wire dtype of the row-parallel `o_proj` partial sum. This is deliberately
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
    /// Optional vLLM production backend for the small-shape fused
    /// all-reduce + residual + `RMSNorm` boundary. An empty list keeps the
    /// traditional pure-all-reduce path for architectures that do not fuse.
    pub fused_allreduce_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct VllmFp8AttnBlockTpWorkletResolved {
    pub raw_cfg: VllmFp8AttnBlockTpWorkletConfig,
    pub input_norm: RmsNormKernelConfig,
    pub qkv: SingleFp8GemmWithQuantConfig,
    pub attn: FlashInferAttentionConfig,
    pub o_proj: SingleFp8GemmWithQuantConfig,
    pub tp_ar: Option<AllReduceKernelConfig>, // tp_size == 1 → None
    pub tp_ar_fused: Option<AllReduceResidualRmsNormKernelConfig>,
    pub tp_ar_fallback_norm: Option<RmsNormKernelConfig>,
    pub max_fused_tokens: Option<u32>,
    pub num_qo_heads_per_rank: Dim,
    pub num_kv_heads_per_rank: Dim,
    pub allreduce_dtype_bytes: u32,
}

/// Per-call shape: `batch_tokens` drives the GEMMs/norm/allreduce; the prefill
/// `(prefix_len, append_len)` pairs and decode KV lengths drive attention.
#[derive(Clone, Debug, Default)]
pub struct VllmFp8AttnBlockTpWorkletInput {
    pub batch_tokens: u32,
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
    pub decode_kv_lens: Vec<u32>,
}

pub struct VllmFp8AttnBlockTpWorklet {
    pub name: String,
    pub input_norm: Op<RmsNormKernel>,
    pub qkv: SingleFp8GemmWithQuantOp,
    pub attn: FlashInferAttentionOp,
    pub o_proj: SingleFp8GemmWithQuantOp,
    pub tp_ar: Option<Op<AllReduceKernel>>,
    pub tp_ar_fused: Option<Op<AllReduceResidualRmsNormKernel>>,
    pub tp_ar_fallback_norm: Option<Op<RmsNormKernel>>,
    resolved: VllmFp8AttnBlockTpWorkletResolved,
}

impl VllmFp8AttnBlockTpWorkletConfig {
    /// KV cache dtype: fp8 in an fp8 run (both prefill and decode read fp8 KV),
    /// else the base dtype. Used by the arch's KV-byte accounting (`raw_cfg`).
    #[must_use]
    pub fn kv_dtype(&self) -> DType {
        DType::Fp8E4m3
    }
}

impl VllmFp8AttnBlockTpWorklet {
    pub fn resolve_config(
        cfg: &VllmFp8AttnBlockTpWorkletConfig,
    ) -> VllmFp8AttnBlockTpWorkletResolved {
        let tp = u32::from(cfg.tp_size);
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
        let tp_ar_fused =
            (cfg.tp_size > 1 && !cfg.fused_allreduce_backends.is_empty()).then(|| {
                AllReduceResidualRmsNormKernelConfig {
                    backends: cfg.fused_allreduce_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    num_gpus: u32::from(cfg.tp_size),
                    hidden_dim: cfg.hidden.get(),
                    dtype: cfg.allreduce_dtype,
                    fabric: cfg.allreduce_fabric,
                    strategy: "auto".to_string(),
                    launch_with_pdl: true,
                    fp32_acc: true,
                }
            });
        let max_fused_tokens = tp_ar_fused
            .as_ref()
            .map(AllReduceResidualRmsNormSpec::max_fused_tokens);
        VllmFp8AttnBlockTpWorkletResolved {
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
                num_gpus: u32::from(cfg.tp_size),
                fabric: cfg.allreduce_fabric,
            }),
            // This worklet owns the post-attention norm boundary for every
            // topology. TP1 has no collective and therefore always evaluates
            // this standalone norm; TP>1 uses it only above the fused limit.
            tp_ar_fallback_norm: (cfg.tp_size == 1 || tp_ar_fused.is_some()).then(|| {
                RmsNormKernelConfig {
                    backends: cfg.norm_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    hidden: cfg.hidden.clone(),
                    dtype: cfg.dtype,
                }
            }),
            tp_ar_fused,
            max_fused_tokens,
            num_qo_heads_per_rank: qo_pr,
            num_kv_heads_per_rank: kv_pr,
            allreduce_dtype_bytes: cfg.allreduce_dtype.size_bytes(),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: VllmFp8AttnBlockTpWorkletResolved,
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
        let tp_ar_fused = resolved
            .tp_ar_fused
            .as_ref()
            .map(|config| {
                let fused_name = format!("{name}.tp_allreduce_residual_norm");
                AllReduceResidualRmsNormKernel::build(fused_name.clone(), config.clone(), bridge)
                    .map(|kernel| Op::new(fused_name, Arc::new(kernel)))
            })
            .transpose()?;
        let tp_ar_fallback_norm = resolved
            .tp_ar_fallback_norm
            .as_ref()
            .map(|config| {
                let norm_name = format!("{name}.tp_allreduce_fallback_norm");
                RmsNormKernel::build(norm_name.clone(), config.clone(), bridge)
                    .map(|kernel| Op::new(norm_name, Arc::new(kernel)))
            })
            .transpose()?;
        Ok(Self {
            name,
            input_norm,
            qkv,
            attn,
            o_proj,
            tp_ar,
            tp_ar_fused,
            tp_ar_fallback_norm,
            resolved,
        })
    }

    /// `CostTree` compile: sum `input_norm` + qkv + attn (append/prefill/decode leaves)
    /// + `o_proj` + optional `tp_allreduce`, wrapped in a `Labeled` partition header.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let r = &self.resolved;
        let label = format!(
            "{} (VllmFp8AttnBlockTpWorklet) [tp={}; qo {:?}, kv {:?}, {:?}]",
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
        if let Some(tp_ar_fused) = &self.tp_ar_fused {
            parts.push(tp_ar_fused.compile(builder));
        }
        if let Some(tp_ar_fallback_norm) = &self.tp_ar_fallback_norm {
            parts.push(tp_ar_fallback_norm.compile(builder));
        }
        CostNode::Labeled {
            label,
            child: Box::new(CostNode::Sum(parts)),
        }
    }

    /// `CostTree` eval: fill slots in the exact `compile` child order so the
    /// evaluator cursor stays aligned with the minted slot indices.
    pub fn eval(&self, input: &VllmFp8AttnBlockTpWorkletInput, ev: &mut Evaluator) {
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
        let use_fused = use_fused_allreduce(m, self.resolved.max_fused_tokens);
        let message_size_bytes = u64::from(m)
            * u64::from(self.resolved.raw_cfg.hidden.get())
            * u64::from(self.resolved.allreduce_dtype_bytes);
        if let Some(tp_ar) = &self.tp_ar {
            let ar_input = AllReduceKernelInput { message_size_bytes };
            if use_fused || m == 0 {
                ev.push(LeafMetrics::ZERO, || ar_input.into());
            } else {
                tp_ar.eval(&ar_input, ev);
            }
        }
        if let Some(tp_ar_fused) = &self.tp_ar_fused {
            let fused_input = AllReduceResidualRmsNormKernelInput { num_tokens: m };
            if use_fused {
                tp_ar_fused.eval(&fused_input, ev);
            } else {
                ev.push(LeafMetrics::ZERO, || fused_input.into());
            }
        }
        if let Some(tp_ar_fallback_norm) = &self.tp_ar_fallback_norm {
            let norm_input = RmsNormKernelInput { m };
            if use_fused || m == 0 {
                ev.push(LeafMetrics::ZERO, || norm_input.into());
            } else {
                tp_ar_fallback_norm.eval(&norm_input, ev);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_fp8_projections_and_vllm_fused_boundary() {
        let resolved =
            VllmFp8AttnBlockTpWorklet::resolve_config(&VllmFp8AttnBlockTpWorkletConfig {
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
                fused_allreduce_backends: vec!["flashinfer_trtllm"],
            });
        assert_eq!(resolved.qkv.gemm.dtype, DType::Fp8E4m3);
        assert!(resolved.tp_ar_fused.is_some());
        assert!(resolved.tp_ar_fallback_norm.is_some());
        assert_eq!(resolved.max_fused_tokens, Some(256));
    }
}

fn use_fused_allreduce(num_tokens: u32, max_fused_tokens: Option<u32>) -> bool {
    max_fused_tokens.is_some_and(|maximum| num_tokens > 0 && num_tokens <= maximum)
}
