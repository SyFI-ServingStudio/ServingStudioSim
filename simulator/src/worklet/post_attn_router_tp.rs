//! `PostAttnRouterTpWorklet` — the post-attention dense tail + MoE gate of a
//! TP/DP-attention decoder layer: row-parallel o_proj → optional `tp_allreduce`
//! → post-attention RMSNorm → MoE router GEMM. `TP` group suffix (L3 §1.5): the
//! o_proj all-reduce sums the row-parallel partials across the `tp_size` ranks;
//! everything after it (post_norm + router) runs *replicated* on every rank of
//! the shard (the full hidden is replicated post-all-reduce), so it bills the
//! shard's own token slice — no further sharding, no pooled average.
//!
//! This is the tail of the AFD ffn side's per-layer body: the attn pool returns
//! the attention output, the ffn pool projects it back (o_proj) + re-syncs (all-
//! reduce) + normalizes + routes, producing the per-token expert scores that feed
//! the MoE dispatch all-to-all (the next sync section, owned by the arch). The
//! post_norm + router are delegated to the concrete BF16/native
//! [`NativeMoeRouterLocalWorklet`] so the AFD and unified native archs share one
//! routing cost definition without a runtime precision/backend branch.
//!
//! o_proj is the row-parallel partition of [`AttnBlockTpWorklet`]'s tail (input
//! `k = num_qo_heads/tp · head_dim`, output `n = hidden`); `tp_size == 1`
//! degenerates (full shapes, no collective).
//!
//! [`AttnBlockTpWorklet`]: crate::worklet::AttnBlockTpWorklet
//! [`NativeMoeRouterLocalWorklet`]: crate::worklet::NativeMoeRouterLocalWorklet

use std::sync::Arc;

use crate::common::Fabric;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::native_moe_router_local::{
    NativeMoeRouterLocalWorklet, NativeMoeRouterLocalWorkletConfig,
    NativeMoeRouterLocalWorkletInput, NativeMoeRouterLocalWorkletResolved,
};

/// Raw global config + TP degree + the collective fabric/backends. The o_proj
/// partition and the composed router config are derived in `resolve_config`.
#[derive(Clone, Debug)]
pub struct PostAttnRouterTpWorkletConfig {
    pub hidden: Dim,
    pub num_qo_heads: Dim,
    pub head_dim: Dim,
    pub num_experts: Dim,
    /// Base (16-bit) dtype — the post-attention RMSNorm keeps it.
    pub dtype: DType,
    pub tp_size: u16,
    /// Symbol name for `tp_size` in the derivation formula (`attn_tp`/`tp`) — the
    /// arch owns which sharding degree this worklet's `tp` is.
    pub tp_name: &'static str,
    pub allreduce_fabric: Fabric,
    pub gpu_name: String,
    /// post-attention RMSNorm backends (forwarded to the composed router worklet).
    pub norm_backends: Vec<&'static str>,
    /// o_proj AND router GEMM backends.
    pub gemm_backends: Vec<&'static str>,
    pub allreduce_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct PostAttnRouterTpWorkletResolved {
    pub raw_cfg: PostAttnRouterTpWorkletConfig,
    pub o_proj: SingleGemmKernelConfig,
    pub tp_ar: Option<AllReduceKernelConfig>, // tp_size == 1 → None
    pub router: NativeMoeRouterLocalWorkletResolved,
    pub num_qo_heads_per_rank: Dim,
    pub dtype_bytes: u32,
}

/// Per-call shape: `batch_tokens` (this DP shard's token count) drives the
/// o_proj, the all-reduce message, and the composed router.
#[derive(Clone, Debug, Default)]
pub struct PostAttnRouterTpWorkletInput {
    pub batch_tokens: u32,
}

pub struct PostAttnRouterTpWorklet {
    pub name: String,
    pub o_proj: Op<SingleGemmKernel>,
    pub tp_ar: Option<Op<AllReduceKernel>>,
    pub router: NativeMoeRouterLocalWorklet,
    resolved: PostAttnRouterTpWorkletResolved,
}

impl PostAttnRouterTpWorklet {
    pub fn resolve_config(cfg: &PostAttnRouterTpWorkletConfig) -> PostAttnRouterTpWorkletResolved {
        let tp = cfg.tp_size as u32;
        // o_proj only depends on the query-head split (its input is the attention
        // output, `num_qo_heads/tp · head_dim` per rank).
        assert!(
            cfg.num_qo_heads.get().is_multiple_of(tp),
            "num_qo_heads {} not divisible by tp_size {}",
            cfg.num_qo_heads,
            tp
        );
        let qo_pr = cfg.num_qo_heads.clone() / Dim::param(cfg.tp_name, tp);
        let o_proj_gemm = SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: cfg.hidden.clone(),
            k: qo_pr.clone() * cfg.head_dim.clone(),
            dtype: cfg.dtype,
        };
        PostAttnRouterTpWorkletResolved {
            // row-parallel o_proj: input k = per-rank Q heads × head_dim; output n = hidden.
            o_proj: o_proj_gemm,
            tp_ar: (cfg.tp_size > 1).then(|| AllReduceKernelConfig {
                // Comm is size-keyed: the all-reduce reads the fp8-width message
                // bytes (`dtype_bytes` below) off a dtype-agnostic curve.
                backends: cfg.allreduce_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_gpus: cfg.tp_size as u32,
                fabric: cfg.allreduce_fabric,
            }),
            router: NativeMoeRouterLocalWorklet::resolve_config(
                &NativeMoeRouterLocalWorkletConfig {
                    hidden: cfg.hidden.clone(),
                    num_experts: cfg.num_experts.clone(),
                    dtype: cfg.dtype,
                    gpu_name: cfg.gpu_name.clone(),
                    norm_backends: cfg.norm_backends.clone(),
                    gemm_backends: cfg.gemm_backends.clone(),
                },
            ),
            num_qo_heads_per_rank: qo_pr,
            dtype_bytes: cfg.dtype.size_bytes(),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: PostAttnRouterTpWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let o_name = format!("{name}.o_proj");
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
        let router = NativeMoeRouterLocalWorklet::build(
            format!("{name}.moe_router"),
            resolved.router.clone(),
            bridge,
        )?;
        Ok(Self {
            name,
            o_proj,
            tp_ar,
            router,
            resolved,
        })
    }

    /// CostTree compile: `Sum(o_proj, [tp_allreduce], router)` under a `Labeled`
    /// header. `router` expands to its own `Sum(post_norm, router_gemm)`.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let r = &self.resolved;
        let label = format!(
            "{} (PostAttnRouterTpWorklet) [tp={}; o_proj k={}·{}/tp, n={:?}, {:?}]",
            self.name,
            r.raw_cfg.tp_size,
            r.raw_cfg.num_qo_heads,
            r.raw_cfg.head_dim,
            r.raw_cfg.hidden,
            r.raw_cfg.num_experts,
        );
        let mut parts = vec![self.o_proj.compile(builder)];
        if let Some(tp_ar) = &self.tp_ar {
            parts.push(tp_ar.compile(builder));
        }
        parts.push(self.router.compile(builder));
        CostNode::Labeled {
            label,
            child: Box::new(CostNode::Sum(parts)),
        }
    }

    /// CostTree eval: fill slots in the exact `compile` child order — o_proj,
    /// optional tp_allreduce, then the composed router (post_norm, router_gemm).
    pub fn eval(&self, input: &PostAttnRouterTpWorkletInput, ev: &mut Evaluator) {
        let m = input.batch_tokens;
        self.o_proj.eval(&SingleGemmKernelInput { m }, ev);
        if let Some(tp_ar) = &self.tp_ar {
            // All-reduce the FULL [tokens × hidden] o_proj partial-sum.
            let message_size_bytes = (m as u64)
                * (self.resolved.raw_cfg.hidden.get() as u64)
                * (self.resolved.dtype_bytes as u64);
            tp_ar.eval(&AllReduceKernelInput { message_size_bytes }, ev);
        }
        self.router
            .eval(&NativeMoeRouterLocalWorkletInput { batch_tokens: m }, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(tp_size: u16) -> PostAttnRouterTpWorkletConfig {
        PostAttnRouterTpWorkletConfig {
            hidden: 4096.into(),
            num_qo_heads: 32.into(),
            head_dim: 128.into(),
            num_experts: 128.into(),
            dtype: DType::Bf16,
            tp_size,
            tp_name: "tp",
            allreduce_fabric: Fabric::Nvlink,
            gpu_name: "H100".to_string(),
            norm_backends: vec!["flashinfer"],
            gemm_backends: vec!["torch"],
            allreduce_backends: vec!["nccl"],
        }
    }

    #[test]
    fn tp1_is_degenerate_full_shapes_no_collective() {
        let r = PostAttnRouterTpWorklet::resolve_config(&cfg(1));
        // o_proj k = 32·128 = 4096, n = hidden.
        assert_eq!(r.o_proj.k, 4096);
        assert_eq!(r.o_proj.n, 4096);
        assert!(r.tp_ar.is_none(), "tp=1 must have no collective");
        // composed router: n=num_experts, k=hidden.
        assert_eq!(r.router.router.n, 128);
        assert_eq!(r.router.router.k, 4096);
        assert_eq!(r.router.post_norm.hidden, 4096);
    }

    #[test]
    fn tp4_partitions_o_proj_and_adds_allreduce() {
        let r = PostAttnRouterTpWorklet::resolve_config(&cfg(4));
        // per-rank Q heads 32/4=8 → o_proj k = 8·128 = 1024.
        assert_eq!(r.num_qo_heads_per_rank, 8);
        assert_eq!(r.o_proj.k, 8 * 128);
        assert_eq!(r.o_proj.n, 4096); // hidden NOT sharded
        let ar = r.tp_ar.expect("tp>1 must add allreduce");
        assert_eq!(ar.num_gpus, 4);
        assert_eq!(ar.fabric, Fabric::Nvlink);
        // router is replicated (un-sharded) regardless of tp.
        assert_eq!(r.router.router.n, 128);
        assert_eq!(r.router.router.k, 4096);
    }

    #[test]
    #[should_panic(expected = "num_qo_heads")]
    fn tp_indivisible_qo_heads_panics() {
        // 32 qo heads, tp=5 → 32 % 5 != 0.
        let _ = PostAttnRouterTpWorklet::resolve_config(&cfg(5));
    }
}
