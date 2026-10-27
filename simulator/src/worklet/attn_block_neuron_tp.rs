//! `AttnBlockNeuronTpWorklet` — the attention block of one Llama 3.1 8B layer
//! as stock vLLM Neuron (0.29) launches it on the four LNC2 ranks of one
//! Trainium2 chip. It composes the standalone launch-unit leaves of
//! `doc/pr/llama31-8b-trn2/README.md` ("Kernels") and the NeuronLink
//! collective leaf. `TP` group suffix: the block is one sync section, and the
//! collective that closes the row-parallel output projection is its boundary.
//!
//! The engine runs the two phases as different compiled graphs with different
//! launch units. The worklet mints both branches once and charges zero for the
//! branch an iteration does not run (`Op::eval_if`):
//!
//! - prefill (sequence-parallel): `all_gather` of the `[T/tp, H]` activation to
//!   `[T, H]`, the `qkv` projection with RoPE (`neuron_nki_qkv`), causal flash
//!   attention over the per-rank heads (`neuron_nki_attention_cte`), the output
//!   projection (`neuron_nki_output_projection`), then `reduce_scatter` back to
//!   `[T/tp, H]`;
//! - decode: the fused attention block over the paged KV cache
//!   (`neuron_attention_decode`: projection, RoPE, the K/V block gather,
//!   attention, output projection and the cache write in one call), then
//!   `all_reduce` of `[B, H]`.
//!
//! The leaves are compiled variants. `token_bucket` is the compiled prefill
//! token bucket T or decode batch bucket B the engine padded the batch to, and
//! the collectives move the padded `[bucket x H]` activation in BF16. The
//! norms, residual adds and the prefill KV write of the layer are the separate
//! `neuron_glue` leaf, which the arch charges once per layer.

use std::sync::Arc;

use crate::common::Fabric;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    NeuronAttentionDecodeKernel, NeuronCollectiveKernel, NeuronCollectiveKernelConfig,
    NeuronCollectiveKernelInput, NeuronCollectiveOp, NeuronNkiAttentionCteKernel,
    NeuronNkiKernelConfig, NeuronNkiKernelInput, NeuronNkiOutputProjectionKernel,
    NeuronNkiQkvKernel, NeuronPhase,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

/// Raw global dims, the compiled inventory and the leaf backends. The per-rank
/// head counts are derived in `resolve_config`.
#[derive(Clone, Debug)]
pub struct AttnBlockNeuronTpWorkletConfig {
    pub hidden: Dim,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub dtype: DType,
    pub tp_size: u16,
    /// The compiled context: the span of every decode request's block table.
    pub max_model_len: u32,
    /// Compiled prefill token buckets (`num_batched_tokens_buckets`), ascending.
    pub prefill_buckets: Vec<u32>,
    /// Compiled decode batch buckets (`num_seqs_buckets`), ascending.
    pub decode_buckets: Vec<u32>,
    pub gpu_name: String,
    pub nki_backends: Vec<&'static str>,
    /// The decode attention leaf's path: `NKI_BACKEND` (the per-head block
    /// table, the NKI kernel) or `NEURON_TORCH_BACKEND` (the stock call, the
    /// torch fallback).
    pub attention_decode_backends: Vec<&'static str>,
    pub collective_backends: Vec<&'static str>,
    pub collective_fabric: Fabric,
    /// Whether the block's decode compute (the attention block) is a slot of
    /// this worklet. False under `decode_granularity: layer`: the arch charges
    /// the whole decode layer as one leaf, measured in a TP4 forward that
    /// holds both block all-reduces. `eval` then charges this block's decode
    /// all-reduce zero. The all-reduce slot stays in the tree.
    pub decode_ops_in_block: bool,
}

/// The leaf configs of the block: the three prefill-only kinds carry the
/// prefill inventory, the decode kind (when the block owns it) the decode
/// inventory. The leaves are measured at the fixed per-rank shapes of TP4; the
/// per-rank head counts appear in the cost-tree label only.
#[derive(Clone, Debug)]
pub struct AttnBlockNeuronTpWorkletResolved {
    pub raw_cfg: AttnBlockNeuronTpWorkletConfig,
    pub qkv: NeuronNkiKernelConfig,
    pub attention_cte: NeuronNkiKernelConfig,
    pub output_projection: NeuronNkiKernelConfig,
    pub attention_decode: Option<NeuronNkiKernelConfig>,
    pub all_gather: NeuronCollectiveKernelConfig,
    pub reduce_scatter: NeuronCollectiveKernelConfig,
    pub all_reduce: NeuronCollectiveKernelConfig,
    pub num_qo_heads_per_rank: Dim,
    pub num_kv_heads_per_rank: Dim,
    pub dtype_bytes: u32,
}

/// Per-call shape: the phase of the compiled graph the iteration runs and the
/// bucket the engine padded the batch to. The arch lowers the logical batch
/// to this variant.
#[derive(Clone, Debug)]
pub struct AttnBlockNeuronTpWorkletInput {
    pub phase: NeuronPhase,
    pub token_bucket: u32,
}

/// The built block: seven leaf slots (six when the decode layer leaf owns the
/// attention block), in `compile` order.
pub struct AttnBlockNeuronTpWorklet {
    pub name: String,
    pub all_gather: Op<NeuronCollectiveKernel>,
    pub qkv: Op<NeuronNkiQkvKernel>,
    pub attention_cte: Op<NeuronNkiAttentionCteKernel>,
    pub output_projection: Op<NeuronNkiOutputProjectionKernel>,
    pub reduce_scatter: Op<NeuronCollectiveKernel>,
    pub attention_decode: Option<Op<NeuronAttentionDecodeKernel>>,
    pub all_reduce: Op<NeuronCollectiveKernel>,
    resolved: AttnBlockNeuronTpWorkletResolved,
}

/// One leaf slot named `<name>.<suffix>`. Shared with the MLP block so both
/// blocks name their slots the same way.
macro_rules! slot {
    ($kernel:ident, $name:expr, $cfg:expr, $suffix:literal, $bridge:expr) => {{
        let slot = format!("{}.{}", $name, $suffix);
        Op::new(
            slot.clone(),
            Arc::new($kernel::build(slot, $cfg.clone(), $bridge)?),
        )
    }};
}
pub(crate) use slot;

impl AttnBlockNeuronTpWorklet {
    /// Derive the per-rank head counts and bake one leaf config per kind. The
    /// inventories' order and bounds are the arch's rule (`validate_parallel`);
    /// the leaf identity (TP4, BF16, the stock pool) is checked at build.
    pub fn resolve_config(
        cfg: &AttnBlockNeuronTpWorkletConfig,
    ) -> AttnBlockNeuronTpWorkletResolved {
        let tp = u32::from(cfg.tp_size);
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
        let tp_dim = Dim::param("tp", tp);
        let leaf_on = |backends: &[&'static str], prefill: &[u32], decode: &[u32]| {
            NeuronNkiKernelConfig::stock(
                &cfg.gpu_name,
                backends,
                cfg.dtype,
                cfg.tp_size,
                cfg.max_model_len,
                prefill,
                decode,
            )
        };
        let leaf = |prefill: &[u32], decode: &[u32]| leaf_on(&cfg.nki_backends, prefill, decode);
        let collective = |op| {
            NeuronCollectiveKernelConfig::tp_group(
                &cfg.gpu_name,
                &cfg.collective_backends,
                cfg.collective_fabric,
                cfg.tp_size,
                op,
            )
        };
        AttnBlockNeuronTpWorkletResolved {
            // The three prefill-only kinds have no decode rows: in decode, the
            // fused attention kernel does their projections.
            qkv: leaf(&cfg.prefill_buckets, &[]),
            attention_cte: leaf(&cfg.prefill_buckets, &[]),
            output_projection: leaf(&cfg.prefill_buckets, &[]),
            attention_decode: cfg
                .decode_ops_in_block
                .then(|| leaf_on(&cfg.attention_decode_backends, &[], &cfg.decode_buckets)),
            all_gather: collective(NeuronCollectiveOp::AllGather),
            reduce_scatter: collective(NeuronCollectiveOp::ReduceScatter),
            all_reduce: collective(NeuronCollectiveOp::AllReduce),
            num_qo_heads_per_rank: cfg.num_qo_heads.clone() / tp_dim.clone(),
            num_kv_heads_per_rank: cfg.num_kv_heads.clone() / tp_dim,
            dtype_bytes: cfg.dtype.size_bytes(),
            raw_cfg: cfg.clone(),
        }
    }

    /// Build the leaf slots (seven, six under the layer leaf), named
    /// `<name>.<slot>`.
    pub fn build(
        name: String,
        resolved: AttnBlockNeuronTpWorkletResolved,
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
            qkv: slot!(NeuronNkiQkvKernel, name, resolved.qkv, "qkv_proj", bridge),
            attention_cte: slot!(
                NeuronNkiAttentionCteKernel,
                name,
                resolved.attention_cte,
                "attention_cte",
                bridge
            ),
            output_projection: slot!(
                NeuronNkiOutputProjectionKernel,
                name,
                resolved.output_projection,
                "o_proj",
                bridge
            ),
            reduce_scatter: slot!(
                NeuronCollectiveKernel,
                name,
                resolved.reduce_scatter,
                "reduce_scatter",
                bridge
            ),
            attention_decode: match &resolved.attention_decode {
                Some(cfg) => Some(slot!(
                    NeuronAttentionDecodeKernel,
                    name,
                    cfg,
                    "attention_decode",
                    bridge
                )),
                None => None,
            },
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

    /// Both phase branches in one `Sum`: prefill (all_gather, qkv, attention,
    /// o_proj, reduce_scatter) then decode (attention_decode when the block
    /// owns it, all_reduce).
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let r = &self.resolved;
        let label = format!(
            "{} (AttnBlockNeuronTpWorklet) [tp={}; {:?} q heads, {:?} kv heads per rank; \
             prefill buckets {:?}, decode buckets {:?}]",
            self.name,
            r.raw_cfg.tp_size,
            r.num_qo_heads_per_rank,
            r.num_kv_heads_per_rank,
            r.raw_cfg.prefill_buckets,
            r.raw_cfg.decode_buckets,
        );
        let mut slots = vec![
            self.all_gather.compile(builder),
            self.qkv.compile(builder),
            self.attention_cte.compile(builder),
            self.output_projection.compile(builder),
            self.reduce_scatter.compile(builder),
        ];
        if let Some(attention_decode) = &self.attention_decode {
            slots.push(attention_decode.compile(builder));
        }
        slots.push(self.all_reduce.compile(builder));
        CostNode::Labeled {
            label,
            child: Box::new(CostNode::Sum(slots)),
        }
    }

    /// Fill the slots in `compile` order; the other phase's slots are zero.
    pub fn eval(&self, input: &AttnBlockNeuronTpWorkletInput, ev: &mut Evaluator) {
        let prefill = input.phase == NeuronPhase::Prefill;
        let leaf = NeuronNkiKernelInput {
            phase: input.phase,
            token_bucket: input.token_bucket,
        };
        // The padded [bucket x hidden] BF16 activation: the full buffer of every
        // collective (see neuron_collective.rs).
        let message = NeuronCollectiveKernelInput {
            message_size_bytes: u64::from(input.token_bucket)
                * u64::from(self.resolved.raw_cfg.hidden.get())
                * u64::from(self.resolved.dtype_bytes),
        };
        self.all_gather.eval_if(&message, prefill, ev);
        self.qkv.eval_if(&leaf, prefill, ev);
        self.attention_cte.eval_if(&leaf, prefill, ev);
        self.output_projection.eval_if(&leaf, prefill, ev);
        self.reduce_scatter.eval_if(&message, prefill, ev);
        if let Some(attention_decode) = &self.attention_decode {
            attention_decode.eval_if(&leaf, !prefill, ev);
        }
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

    fn cfg(tp_size: u16) -> AttnBlockNeuronTpWorkletConfig {
        AttnBlockNeuronTpWorkletConfig {
            hidden: 4096.into(),
            num_qo_heads: 32.into(),
            num_kv_heads: 8.into(),
            dtype: DType::Bf16,
            tp_size,
            max_model_len: 512,
            prefill_buckets: vec![512],
            decode_buckets: vec![1, 16],
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            nki_backends: vec![NKI_BACKEND],
            attention_decode_backends: vec![NKI_BACKEND],
            collective_backends: vec![NEURON_COLLECTIVE_BACKEND],
            collective_fabric: Fabric::NeuronLink,
            decode_ops_in_block: true,
        }
    }

    #[test]
    fn phases_split_into_their_kinds_inventories() {
        let r = AttnBlockNeuronTpWorklet::resolve_config(&cfg(4));
        assert_eq!(r.num_qo_heads_per_rank, 8);
        assert_eq!(r.num_kv_heads_per_rank, 2);
        assert!(r.qkv.decode_buckets.is_empty());
        let attention_decode = r.attention_decode.as_ref().expect("the block owns decode");
        assert!(attention_decode.prefill_buckets.is_empty());
        assert_eq!(attention_decode.decode_buckets, vec![1, 16]);
    }

    #[test]
    fn the_layer_leaf_removes_the_attention_slot_and_keeps_the_all_reduce_slot() {
        let mut layer = cfg(4);
        layer.decode_ops_in_block = false;
        let r = AttnBlockNeuronTpWorklet::resolve_config(&layer);
        assert!(r.attention_decode.is_none());
        assert_eq!(r.qkv.prefill_buckets, vec![512]);
    }

    #[test]
    #[should_panic(expected = "num_kv_heads")]
    fn tp_above_kv_heads_panics() {
        let _ = AttnBlockNeuronTpWorklet::resolve_config(&cfg(16));
    }
}
