//! `MlpBlockNeuronTpWorklet` — the gated MLP block of one Llama 3.1 8B layer as
//! stock vLLM Neuron (0.29) launches it on the four LNC2 ranks of one Trainium2
//! chip: the `nkilib` `mlp` kernel (`neuron_nki_mlp`, gate/up/down with the
//! SiLU gate fused, per rank) between the block's collectives. `TP` group
//! suffix: the collective that closes the row-parallel down projection is the
//! sync boundary.
//!
//! Prefill runs sequence-parallel: `all_gather` of `[T/tp, H]` to `[T, H]`, the
//! kernel at T rows, `reduce_scatter` back. Decode runs the kernel at B rows
//! and closes with `all_reduce`. The kernel selects its CTE or TKG path from
//! the row count, so one leaf covers both phases; the worklet charges zero for
//! the collectives the iteration's phase does not launch.

use std::sync::Arc;

use crate::common::Fabric;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    NeuronCollectiveKernel, NeuronCollectiveKernelConfig, NeuronCollectiveKernelInput,
    NeuronCollectiveOp, NeuronNkiKernelConfig, NeuronNkiKernelInput, NeuronNkiMlpKernel,
    NeuronPhase,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};
use crate::worklet::attn_block_neuron_tp::slot;

/// Raw global dims, the compiled inventory and the leaf backends. The per-rank
/// intermediate is derived in `resolve_config`.
#[derive(Clone, Debug)]
pub struct MlpBlockNeuronTpWorkletConfig {
    pub hidden: Dim,
    pub intermediate: Dim,
    pub dtype: DType,
    pub tp_size: u16,
    pub max_model_len: u32,
    pub prefill_buckets: Vec<u32>,
    pub decode_buckets: Vec<u32>,
    pub gpu_name: String,
    pub nki_backends: Vec<&'static str>,
    pub collective_backends: Vec<&'static str>,
    pub collective_fabric: Fabric,
    /// Whether the block's decode compute (the MLP) is a slot of this
    /// worklet. False under `decode_granularity: layer`: the `mlp` leaf then
    /// carries the prefill inventory only, and `eval` charges the decode
    /// all-reduce zero, because the layer leaf's TP4 forward holds it. The
    /// all-reduce slot stays in the tree.
    pub decode_ops_in_block: bool,
}

/// The leaf configs of the block: one `mlp` leaf with both inventories (the
/// prefill one alone when the decode layer leaf owns the MLP) and the three
/// collectives. The per-rank intermediate appears in the cost-tree label only.
#[derive(Clone, Debug)]
pub struct MlpBlockNeuronTpWorkletResolved {
    pub raw_cfg: MlpBlockNeuronTpWorkletConfig,
    pub mlp: NeuronNkiKernelConfig,
    pub all_gather: NeuronCollectiveKernelConfig,
    pub reduce_scatter: NeuronCollectiveKernelConfig,
    pub all_reduce: NeuronCollectiveKernelConfig,
    pub intermediate_per_rank: Dim,
    pub dtype_bytes: u32,
}

/// Per-call shape: the phase of the compiled graph and the bucket the engine
/// padded the batch to.
#[derive(Clone, Debug)]
pub struct MlpBlockNeuronTpWorkletInput {
    pub phase: NeuronPhase,
    pub token_bucket: u32,
}

/// The built block: four leaf slots, in `compile` order.
pub struct MlpBlockNeuronTpWorklet {
    pub name: String,
    pub all_gather: Op<NeuronCollectiveKernel>,
    pub mlp: Op<NeuronNkiMlpKernel>,
    pub reduce_scatter: Op<NeuronCollectiveKernel>,
    pub all_reduce: Op<NeuronCollectiveKernel>,
    resolved: MlpBlockNeuronTpWorkletResolved,
}

impl MlpBlockNeuronTpWorklet {
    /// Derive the per-rank intermediate and bake the leaf configs.
    pub fn resolve_config(cfg: &MlpBlockNeuronTpWorkletConfig) -> MlpBlockNeuronTpWorkletResolved {
        let tp = u32::from(cfg.tp_size);
        assert!(
            cfg.intermediate.get() % tp == 0,
            "intermediate {} not divisible by tp_size {}",
            cfg.intermediate,
            tp
        );
        let collective = |op| {
            NeuronCollectiveKernelConfig::tp_group(
                &cfg.gpu_name,
                &cfg.collective_backends,
                cfg.collective_fabric,
                cfg.tp_size,
                op,
            )
        };
        let decode_buckets: &[u32] = if cfg.decode_ops_in_block {
            &cfg.decode_buckets
        } else {
            &[]
        };
        MlpBlockNeuronTpWorkletResolved {
            mlp: NeuronNkiKernelConfig::stock(
                &cfg.gpu_name,
                &cfg.nki_backends,
                cfg.dtype,
                cfg.tp_size,
                cfg.max_model_len,
                &cfg.prefill_buckets,
                decode_buckets,
            ),
            all_gather: collective(NeuronCollectiveOp::AllGather),
            reduce_scatter: collective(NeuronCollectiveOp::ReduceScatter),
            all_reduce: collective(NeuronCollectiveOp::AllReduce),
            intermediate_per_rank: cfg.intermediate.clone() / Dim::param("tp", tp),
            dtype_bytes: cfg.dtype.size_bytes(),
            raw_cfg: cfg.clone(),
        }
    }

    /// Build the four leaf slots, named `<name>.<slot>`.
    pub fn build(
        name: String,
        resolved: MlpBlockNeuronTpWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        Ok(Self {
            all_gather: slot!(
                NeuronCollectiveKernel,
                name,
                resolved.all_gather,
                "all_gather",
                bridge
            ),
            mlp: slot!(NeuronNkiMlpKernel, name, resolved.mlp, "mlp", bridge),
            reduce_scatter: slot!(
                NeuronCollectiveKernel,
                name,
                resolved.reduce_scatter,
                "reduce_scatter",
                bridge
            ),
            all_reduce: slot!(
                NeuronCollectiveKernel,
                name,
                resolved.all_reduce,
                "all_reduce",
                bridge
            ),
            name,
            resolved,
        })
    }

    /// `Sum(all_gather, mlp, reduce_scatter, all_reduce)`: the prefill
    /// collectives, the shared kernel, the decode collective.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let r = &self.resolved;
        let label = format!(
            "{} (MlpBlockNeuronTpWorklet) [tp={}; {:?} (replicated), {:?} per rank; \
             prefill buckets {:?}, decode buckets {:?}]",
            self.name,
            r.raw_cfg.tp_size,
            r.raw_cfg.hidden,
            r.intermediate_per_rank,
            r.raw_cfg.prefill_buckets,
            r.raw_cfg.decode_buckets,
        );
        CostNode::Labeled {
            label,
            child: Box::new(CostNode::Sum(vec![
                self.all_gather.compile(builder),
                self.mlp.compile(builder),
                self.reduce_scatter.compile(builder),
                self.all_reduce.compile(builder),
            ])),
        }
    }

    /// Fill the slots in `compile` order; the other phase's collectives are
    /// zero, and so is the `mlp` leaf in decode when the layer leaf owns it.
    pub fn eval(&self, input: &MlpBlockNeuronTpWorkletInput, ev: &mut Evaluator) {
        let prefill = input.phase == NeuronPhase::Prefill;
        let message = NeuronCollectiveKernelInput {
            message_size_bytes: u64::from(input.token_bucket)
                * u64::from(self.resolved.raw_cfg.hidden.get())
                * u64::from(self.resolved.dtype_bytes),
        };
        self.all_gather.eval_if(&message, prefill, ev);
        self.mlp.eval_if(
            &NeuronNkiKernelInput {
                phase: input.phase,
                token_bucket: input.token_bucket,
            },
            prefill || self.resolved.raw_cfg.decode_ops_in_block,
            ev,
        );
        self.reduce_scatter.eval_if(&message, prefill, ev);
        // Under the decode layer leaf the all-reduce runs inside the measured
        // TP4 forward, so the block charges it only with its own decode slots.
        let decode_here = self.resolved.raw_cfg.decode_ops_in_block;
        self.all_reduce
            .eval_if(&message, !prefill && decode_here, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::{NEURON_COLLECTIVE_BACKEND, NKI_BACKEND};

    fn cfg(tp_size: u16) -> MlpBlockNeuronTpWorkletConfig {
        MlpBlockNeuronTpWorkletConfig {
            hidden: 4096.into(),
            intermediate: 14336.into(),
            dtype: DType::Bf16,
            tp_size,
            max_model_len: 512,
            prefill_buckets: vec![512],
            decode_buckets: vec![1, 16],
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            nki_backends: vec![NKI_BACKEND],
            collective_backends: vec![NEURON_COLLECTIVE_BACKEND],
            collective_fabric: Fabric::NeuronLink,
            decode_ops_in_block: true,
        }
    }

    #[test]
    fn one_leaf_carries_both_phases() {
        let r = MlpBlockNeuronTpWorklet::resolve_config(&cfg(4));
        assert_eq!(r.intermediate_per_rank, 3584);
        assert_eq!(r.mlp.prefill_buckets, vec![512]);
        assert_eq!(r.mlp.decode_buckets, vec![1, 16]);
        let mut layer = cfg(4);
        layer.decode_ops_in_block = false;
        let r = MlpBlockNeuronTpWorklet::resolve_config(&layer);
        assert_eq!(r.mlp.prefill_buckets, vec![512]);
        assert!(
            r.mlp.decode_buckets.is_empty(),
            "the layer leaf owns the decode MLP"
        );
    }

    #[test]
    #[should_panic(expected = "intermediate")]
    fn tp_indivisible_intermediate_panics() {
        let _ = MlpBlockNeuronTpWorklet::resolve_config(&cfg(3));
    }
}
