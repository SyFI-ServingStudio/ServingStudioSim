//! `PreAttnProjTpWorklet` — the pre-attention dense projections of a TP decoder
//! layer: input RMSNorm → column-parallel fused QKV. `TP` group suffix (L3
//! §1.5): one sync section, no collective inside (the all-reduce that closes the
//! attention block lives downstream, in the post-attention section).
//!
//! This is the head of the AFD ffn side's per-layer body: the ffn pool runs the
//! input norm + QKV projection on its own DP shard, then ships the projected Q/KV
//! to the attn pool (which owns the attention kernel + KV cache). It is the same
//! Megatron column-parallel partition as the front of [`AttnBlockTpWorklet`]
//! (each rank owns `num_qo_heads / tp` query heads and `num_kv_heads / tp` KV
//! heads; `hidden` is NOT sharded), minus the attention / o_proj / all-reduce.
//!
//! `tp_size == 1` degenerates to the single-GPU case (per-rank == full shapes).
//!
//! [`AttnBlockTpWorklet`]: crate::worklet::AttnBlockTpWorklet

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

/// Raw global config + TP degree. Partition (per-rank head split) is derived in
/// `resolve_config`.
#[derive(Clone, Debug)]
pub struct PreAttnProjTpWorkletConfig {
    pub hidden: Dim,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    /// Base (16-bit) dtype — the input RMSNorm keeps it.
    pub dtype: DType,
    pub tp_size: u16,
    /// Symbol name for `tp_size` in the derivation formula (`attn_tp`/`ffn_tp`/
    /// `tp`) — the arch owns which sharding degree this worklet's `tp` is.
    pub tp_name: &'static str,
    pub gpu_name: String,
    pub norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct PreAttnProjTpWorkletResolved {
    pub raw_cfg: PreAttnProjTpWorkletConfig,
    pub input_norm: RmsNormKernelConfig,
    pub qkv: SingleGemmKernelConfig,
    pub num_qo_heads_per_rank: Dim,
    pub num_kv_heads_per_rank: Dim,
}

/// Per-call shape: `batch_tokens` (this DP shard's token count) drives the norm
/// and the QKV gemm.
#[derive(Clone, Debug, Default)]
pub struct PreAttnProjTpWorkletInput {
    pub batch_tokens: u32,
}

pub struct PreAttnProjTpWorklet {
    pub name: String,
    pub input_norm: Op<RmsNormKernel>,
    pub qkv: Op<SingleGemmKernel>,
    resolved: PreAttnProjTpWorkletResolved,
}

impl PreAttnProjTpWorklet {
    pub fn resolve_config(cfg: &PreAttnProjTpWorkletConfig) -> PreAttnProjTpWorkletResolved {
        let tp = cfg.tp_size as u32;
        // GQA dual-divisibility, identical to the attention TP front: both head
        // counts split across the ranks, tp <= num_kv_heads (no KV replication).
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
        let qkv_gemm = SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: (qo_pr.clone() + 2 * kv_pr.clone()) * cfg.head_dim.clone(),
            k: cfg.hidden.clone(),
            dtype: cfg.dtype,
        };
        PreAttnProjTpWorkletResolved {
            input_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            qkv: qkv_gemm,
            num_qo_heads_per_rank: qo_pr,
            num_kv_heads_per_rank: kv_pr,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: PreAttnProjTpWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let norm_name = format!("{name}.input_norm");
        let qkv_name = format!("{name}.qkv_proj");
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
        Ok(Self {
            name,
            input_norm,
            qkv,
            resolved,
        })
    }

    /// CostTree compile: `Sum(input_norm, qkv)` under a `Labeled` partition header.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let r = &self.resolved;
        let label = format!(
            "{} (PreAttnProjTpWorklet) [tp={}; qo {:?}, kv {:?}, {:?}]",
            self.name,
            r.raw_cfg.tp_size,
            r.num_qo_heads_per_rank,
            r.num_kv_heads_per_rank,
            r.raw_cfg.head_dim,
        );
        CostNode::Labeled {
            label,
            child: Box::new(CostNode::Sum(vec![
                self.input_norm.compile(builder),
                self.qkv.compile(builder),
            ])),
        }
    }

    /// CostTree eval: fill slots in the exact `compile` child order — norm, qkv.
    pub fn eval(&self, input: &PreAttnProjTpWorkletInput, ev: &mut Evaluator) {
        let m = input.batch_tokens;
        self.input_norm.eval(&RmsNormKernelInput { m }, ev);
        self.qkv.eval(&SingleGemmKernelInput { m }, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(tp_size: u16) -> PreAttnProjTpWorkletConfig {
        PreAttnProjTpWorkletConfig {
            hidden: 4096.into(),
            num_qo_heads: 32.into(),
            num_kv_heads: 8.into(),
            head_dim: 128.into(),
            dtype: DType::Bf16,
            tp_size,
            tp_name: "tp",
            gpu_name: "H100".to_string(),
            norm_backends: vec!["flashinfer"],
            gemm_backends: vec!["torch"],
        }
    }

    #[test]
    fn tp1_is_degenerate_full_shapes() {
        let r = PreAttnProjTpWorklet::resolve_config(&cfg(1));
        // (32 + 2·8)·128 = 6144 fused QKV; k = hidden.
        assert_eq!(r.qkv.n, 6144);
        assert_eq!(r.qkv.k, 4096);
        assert_eq!(r.input_norm.hidden, 4096);
    }

    #[test]
    fn tp4_partitions_heads() {
        let r = PreAttnProjTpWorklet::resolve_config(&cfg(4));
        // per-rank: qo 32/4=8, kv 8/4=2 → fused (8 + 2·2)·128 = 1536.
        assert_eq!(r.num_qo_heads_per_rank, 8);
        assert_eq!(r.num_kv_heads_per_rank, 2);
        assert_eq!(r.qkv.n, 1536);
        assert_eq!(r.qkv.k, 4096); // hidden NOT sharded
    }

    #[test]
    #[should_panic(expected = "num_kv_heads")]
    fn tp_indivisible_kv_heads_panics() {
        let _ = PreAttnProjTpWorklet::resolve_config(&cfg(16));
    }
}
