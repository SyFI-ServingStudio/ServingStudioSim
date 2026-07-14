//! `MlpBlockTpWorklet` — tensor-parallel MLP block of a dense decoder layer:
//! post-attn RMSNorm → column-parallel fused up_gate → SwiGLU activation →
//! row-parallel down → optional `tp_allreduce`. `TP` group suffix (L3 §1.5): the
//! block is one sync section whose boundary is the all-reduce that sums the
//! row-parallel down output across the `tp_size` ranks.
//!
//! Megatron TP shards the FFN intermediate: up_gate is column-parallel (each
//! rank owns `intermediate / tp` of the gate‖up concat), the SwiGLU activation
//! runs on those local `intermediate / tp` elements, and the row-parallel down's
//! full `[tokens × hidden]` partial-sum is re-synced with the all-reduce.
//! `hidden` is NOT sharded (up_gate input k=hidden, down output n=hidden).
//!
//! `tp_size == 1` degenerates to the single-GPU case: per-rank == full, and the
//! `tp_ar` slot is `None` (no collective), so the cost matches the `Local` path.

use std::sync::Arc;

use crate::common::Fabric;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput, ElementwiseKernel,
    ElementwiseKernelConfig, ElementwiseKernelInput, RmsNormKernel, RmsNormKernelConfig,
    RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

/// Raw global config + TP degree + the collective fabric/backends. Partition is
/// derived in `resolve_config`.
#[derive(Clone, Debug)]
pub struct MlpBlockTpWorkletConfig {
    pub hidden: Dim,
    pub intermediate: Dim,
    pub dtype: DType,
    pub tp_size: u16,
    /// Symbol name for `tp_size` in the derivation formula (`ffn_tp`/`tp`) — the
    /// arch owns which sharding degree this worklet's `tp` is.
    pub tp_name: &'static str,
    pub allreduce_fabric: Fabric,
    pub gpu_name: String,
    pub norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub act_backends: Vec<&'static str>,
    pub allreduce_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct MlpBlockTpWorkletResolved {
    pub raw_cfg: MlpBlockTpWorkletConfig,
    pub post_norm: RmsNormKernelConfig,
    pub up_gate: SingleGemmKernelConfig,
    pub act: ElementwiseKernelConfig,
    pub down: SingleGemmKernelConfig,
    pub tp_ar: Option<AllReduceKernelConfig>, // tp_size == 1 → None
    pub intermediate_per_rank: Dim,
    pub dtype_bytes: u32,
}

/// Per-call shape: `batch_tokens` drives every leaf (the GEMMs, the norm, the
/// activation, and the all-reduce message).
#[derive(Clone, Debug, Default)]
pub struct MlpBlockTpWorkletInput {
    pub batch_tokens: u32,
}

pub struct MlpBlockTpWorklet {
    pub name: String,
    pub post_norm: Op<RmsNormKernel>,
    pub up_gate: Op<SingleGemmKernel>,
    pub act: Op<ElementwiseKernel>,
    pub down: Op<SingleGemmKernel>,
    pub tp_ar: Option<Op<AllReduceKernel>>,
    resolved: MlpBlockTpWorkletResolved,
}

impl MlpBlockTpWorklet {
    pub fn resolve_config(cfg: &MlpBlockTpWorkletConfig) -> MlpBlockTpWorkletResolved {
        let tp = cfg.tp_size as u32;
        assert!(
            cfg.intermediate.get() % tp == 0,
            "intermediate {} not divisible by tp_size {}",
            cfg.intermediate,
            tp
        );
        let inter_pr = cfg.intermediate.clone() / Dim::param(cfg.tp_name, tp);
        let dtype_bytes = cfg.dtype.size_bytes();
        let bytes = Dim::param("bytes", dtype_bytes);
        MlpBlockTpWorkletResolved {
            post_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.clone(), // full hidden, replicated
                dtype: cfg.dtype,
            },
            up_gate: SingleGemmKernelConfig {
                // column-parallel: per-rank gate‖up concat = 2·(intermediate/tp).
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: 2 * inter_pr.clone(),
                k: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            act: ElementwiseKernelConfig {
                // SwiGLU on per-rank intermediate: reads 2·(intermediate/tp),
                // writes (intermediate/tp) elements/token.
                backends: cfg.act_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: 2 * inter_pr.clone() * bytes.clone(),
                output_bytes_per_token: inter_pr.clone() * bytes.clone(),
            },
            down: SingleGemmKernelConfig {
                // row-parallel: input k = per-rank intermediate; output n = hidden.
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden.clone(),
                k: inter_pr.clone(),
                dtype: cfg.dtype,
            },
            tp_ar: (cfg.tp_size > 1).then(|| AllReduceKernelConfig {
                // Comm is size-keyed: the all-reduce reads the message bytes
                // (`dtype_bytes` below) off a dtype-agnostic curve.
                backends: cfg.allreduce_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_gpus: cfg.tp_size as u32,
                fabric: cfg.allreduce_fabric,
            }),
            intermediate_per_rank: inter_pr,
            dtype_bytes,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: MlpBlockTpWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let pn_name = format!("{name}.post_norm");
        let ug_name = format!("{name}.up_gate_proj");
        let act_name = format!("{name}.activation");
        let down_name = format!("{name}.down_proj");

        let post_norm = Op::new(
            pn_name.clone(),
            Arc::new(RmsNormKernel::build(pn_name, resolved.post_norm.clone(), bridge)?),
        );
        let up_gate = Op::new(
            ug_name.clone(),
            Arc::new(SingleGemmKernel::build(ug_name, resolved.up_gate.clone(), bridge)?),
        );
        let act = Op::new(
            act_name.clone(),
            Arc::new(ElementwiseKernel::build(act_name, resolved.act.clone(), bridge)?),
        );
        let down = Op::new(
            down_name.clone(),
            Arc::new(SingleGemmKernel::build(down_name, resolved.down.clone(), bridge)?),
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
            post_norm,
            up_gate,
            act,
            down,
            tp_ar,
            resolved,
        })
    }

    /// CostTree compile: sum post_norm + up_gate + act + down + optional
    /// tp_allreduce, wrapped in a `Labeled` partition header.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let r = &self.resolved;
        let label = format!(
            "{} (MlpBlockTpWorklet) [tp={}; {:?} (replicated), {:?}]",
            self.name,
            r.raw_cfg.tp_size,
            r.raw_cfg.hidden,
            r.intermediate_per_rank,
        );
        let mut parts = vec![
            self.post_norm.compile(builder),
            self.up_gate.compile(builder),
            self.act.compile(builder),
            self.down.compile(builder),
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
    pub fn eval(&self, input: &MlpBlockTpWorkletInput, ev: &mut Evaluator) {
        let m = input.batch_tokens;
        let gemm_in = SingleGemmKernelInput { m };
        self.post_norm.eval(&RmsNormKernelInput { m }, ev);
        self.up_gate.eval(&gemm_in, ev);
        self.act.eval(&ElementwiseKernelInput { num_tokens: m }, ev);
        self.down.eval(&gemm_in, ev);
        if let Some(tp_ar) = &self.tp_ar {
            // All-reduce the FULL [tokens × hidden] down partial-sum (see
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

    fn cfg(tp_size: u16) -> MlpBlockTpWorkletConfig {
        MlpBlockTpWorkletConfig {
            hidden: 4096.into(),
            intermediate: 14336.into(),
            dtype: DType::Bf16,
            tp_size,
            tp_name: "tp",
            allreduce_fabric: Fabric::Nvlink,
            gpu_name: "H100".to_string(),
            norm_backends: vec!["flashinfer"],
            gemm_backends: vec!["torch"],
            act_backends: vec!["triton"],
            allreduce_backends: vec!["nccl"],
        }
    }

    #[test]
    fn tp1_is_degenerate_full_shapes_no_collective() {
        let r = MlpBlockTpWorklet::resolve_config(&cfg(1));
        assert_eq!(r.intermediate_per_rank, 14336);
        assert_eq!(r.up_gate.n, 2 * 14336);
        assert_eq!(r.up_gate.k, 4096);
        assert_eq!(r.down.n, 4096);
        assert_eq!(r.down.k, 14336);
        // bf16 = 2 bytes/elem.
        assert_eq!(r.act.input_bytes_per_token, 2 * 14336 * 2);
        assert_eq!(r.act.output_bytes_per_token, 14336 * 2);
        assert!(r.tp_ar.is_none(), "tp=1 must have no collective");
    }

    #[test]
    fn tp4_shards_intermediate_and_adds_allreduce() {
        let r = MlpBlockTpWorklet::resolve_config(&cfg(4));
        // per-rank intermediate = 14336/4 = 3584.
        assert_eq!(r.intermediate_per_rank, 3584);
        assert_eq!(r.up_gate.n, 2 * 3584);
        assert_eq!(r.up_gate.k, 4096); // hidden NOT sharded
        assert_eq!(r.down.n, 4096); // hidden NOT sharded
        assert_eq!(r.down.k, 3584); // per-rank intermediate
        assert_eq!(r.act.input_bytes_per_token, 2 * 3584 * 2);
        assert_eq!(r.act.output_bytes_per_token, 3584 * 2);
        let ar = r.tp_ar.expect("tp>1 must add allreduce");
        assert_eq!(ar.num_gpus, 4);
        assert_eq!(ar.fabric, Fabric::Nvlink);
    }

    #[test]
    #[should_panic(expected = "intermediate")]
    fn tp_indivisible_intermediate_panics() {
        // 14336 % 3 != 0.
        let _ = MlpBlockTpWorklet::resolve_config(&cfg(3));
    }
}
