//! DeepSeek-V4.1-Flash MoE FFN under DP attention + EP experts (vLLM
//! `--data-parallel-size 4 --enable-expert-parallel`, TP1), one layer over
//! every DP rank.
//!
//! Each rank holds only its own tokens, so the routed experts are reached
//! through vLLM's `allgather_reducescatter` manager (`AgRsAll2AllManager`,
//! fork `distributed/device_communicators/all2all.py:118-162`, prepare and
//! finalize in `fused_moe/prepare_finalize/naive_dp_ep.py`). Per MoE layer
//! (DP4 nsys capture, iteration 345, `phase1_split.md` §1.3):
//!
//! ```text
//! per rank   entry mHC -> router gate -> _dsv4_topk -> MXFP8 quantize (local rows)
//!            shared expert inline above 256 local rows
//! group      all_gatherv [MXFP8 hidden, topk weights, topk ids, ue8m0 scales]
//! group      trtllm MXFP4 MoE over every gathered row (routing precomputed)
//! group      reduce_scatterv of the bf16 expert output
//! per rank   shared + routed add
//! ```
//!
//! At or below 256 local rows the shared expert runs on vLLM's aux stream
//! beside the dispatch, experts and combine (capture: `act_and_mul` and the
//! shared GEMMs on their own stream while the AllGather/bmm run). On the
//! small-batch FULL-graph path the mHC coefficient side branch
//! ([`MhcOverlapBoundary`]) joins after the FFN, so it is charged beside the
//! exchange as well.
//!
//! The kernels are the TP/EP worklet's ([`DeepseekV41MoeFfnEpWorklet`]) at
//! TP1: same leaves and names, composed into rank-local, group and per-rank
//! post sections that the L4 arch places between its `Max` over ranks. The
//! fused MoE prices the busiest EP rank (`folded_rank_position = 0`), which is
//! the `Max` over EP ranks of one demand fold.

use std::sync::Arc;

use super::deepseek_v41_common::{compile_serial_copy, eval_or_zero};
use super::deepseek_v41_moe_ffn_ep::{
    shared_on_aux_stream, DeepseekV41MoeFfnEpWorklet, DeepseekV41MoeFfnEpWorkletConfig,
    DeepseekV41MoeFfnEpWorkletResolved,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernelInput, GemmFp32OutputKernelInput, MhcRmsNormKernelInput,
    MoeEpCollectiveKernelInput, MoeEpQuantizedAllGatherKernel,
    MoeEpQuantizedAllGatherKernelConfig, MoeEpReduceScatterKernel,
    MoeEpReduceScatterKernelConfig, Nvfp4FusedMoeKernelInput, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct DeepseekV41MoeFfnDpEpWorkletConfig {
    /// The rank-local FFN at `tp_size = 1`, `ep_size` = the DP group.
    pub ffn: DeepseekV41MoeFfnEpWorkletConfig,
    pub collective_backends: Vec<&'static str>,
    /// One rank's scheduler budget (`max_num_batched_tokens`); the
    /// collectives' total-token axis spans `ep_size` of them.
    pub max_rank_tokens: u32,
}

#[derive(Clone, Debug)]
pub struct DeepseekV41MoeFfnDpEpWorkletResolved {
    pub raw_cfg: DeepseekV41MoeFfnDpEpWorkletConfig,
    pub ffn: DeepseekV41MoeFfnEpWorkletResolved,
    pub dispatch: MoeEpQuantizedAllGatherKernelConfig,
    pub combine: MoeEpReduceScatterKernelConfig,
}

/// One rank's view of the layer.
#[derive(Clone, Debug, Default)]
pub struct DeepseekV41MoeFfnDpRankInput {
    /// Rows the rank's dense kernels run (graph padding included).
    pub num_tokens: u32,
    /// `mhc_overlap_live(..)` on this rank's rows.
    pub mhc_overlap: bool,
}

/// The group-wide exchange: every rank's rows, and what the collectives move.
#[derive(Clone, Debug, Default)]
pub struct DeepseekV41MoeFfnDpEpWorkletInput {
    pub ranks: Vec<DeepseekV41MoeFfnDpRankInput>,
    /// Rows each rank contributes to the all-gather and receives back.
    pub collective_tokens: Vec<u32>,
}

pub struct DeepseekV41MoeFfnDpEpWorklet {
    pub name: String,
    /// The TP1 FFN's ops; its `all_reduce` is absent at TP1.
    pub ffn: DeepseekV41MoeFfnEpWorklet,
    pub dispatch: Op<MoeEpQuantizedAllGatherKernel>,
    pub combine: Op<MoeEpReduceScatterKernel>,
    resolved: DeepseekV41MoeFfnDpEpWorkletResolved,
}

impl DeepseekV41MoeFfnDpEpWorklet {
    pub fn resolve_config(
        cfg: &DeepseekV41MoeFfnDpEpWorkletConfig,
    ) -> DeepseekV41MoeFfnDpEpWorkletResolved {
        let ffn = &cfg.ffn;
        assert_eq!(ffn.tp_size, 1, "DP attention runs the FFN at TP1");
        assert!(
            matches!(ffn.ep_size, 2 | 4 | 8),
            "the dispatch/combine collectives span 2, 4 or 8 GPUs of one node"
        );
        let max_total_tokens = cfg
            .max_rank_tokens
            .checked_mul(ffn.ep_size)
            .expect("gathered token bound overflows u32");
        DeepseekV41MoeFfnDpEpWorkletResolved {
            ffn: DeepseekV41MoeFfnEpWorklet::resolve_config(ffn),
            // `MXFP8QuantizeLinear` output (e4m3 + ue8m0 per 32) plus the
            // router's fp32 top-k weights and int32 ids.
            dispatch: MoeEpQuantizedAllGatherKernelConfig {
                backends: cfg.collective_backends.clone(),
                gpu_name: ffn.gpu_name.clone(),
                num_gpus: ffn.ep_size,
                hidden_size: ffn.hidden_size.clone(),
                top_k: ffn.top_k.into(),
                activation_dtype: DType::Mxfp8E4m3,
                fabric: "nvlink".into(),
                max_total_tokens,
            },
            // bf16 `finalize` output back to each rank's own rows.
            combine: MoeEpReduceScatterKernelConfig {
                backends: cfg.collective_backends.clone(),
                gpu_name: ffn.gpu_name.clone(),
                num_gpus: ffn.ep_size,
                hidden_size: ffn.hidden_size.clone(),
                dtype: DType::Bf16,
                fabric: "nvlink".into(),
                max_total_tokens,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: DeepseekV41MoeFfnDpEpWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let dispatch = format!("{name}.routed.dispatch_all_gather");
        let combine = format!("{name}.routed.combine_reduce_scatter");
        Ok(Self {
            ffn: DeepseekV41MoeFfnEpWorklet::build(name.clone(), resolved.ffn.clone(), bridge)?,
            dispatch: Op::new(
                dispatch.clone(),
                Arc::new(MoeEpQuantizedAllGatherKernel::build(
                    dispatch,
                    resolved.dispatch.clone(),
                    bridge,
                )?),
            ),
            combine: Op::new(
                combine.clone(),
                Arc::new(MoeEpReduceScatterKernel::build(
                    combine,
                    resolved.combine.clone(),
                    bridge,
                )?),
            ),
            name,
            resolved,
        })
    }

    /// One rank's launches up to the dispatch: the entry mHC (fused, or the
    /// overlap path's caller-stream half), router, top-k, MXFP8 quantize, and
    /// the shared expert when it runs inline (its `.serial` copies).
    pub fn compile_local(&self, b: &mut CostTreeBuilder) -> Vec<CostNode> {
        let f = &self.ffn;
        let mut leaves = vec![f.entry.compile(b)];
        leaves.extend(f.mhc_overlap.compile_entry(b));
        leaves.extend([
            f.router_gate.compile(b),
            f.topk.compile(b),
            f.moe_input_quant.compile(b),
            compile_serial_copy(&f.shared_gate_up, b),
            compile_serial_copy(&f.shared_act, b),
            compile_serial_copy(&f.shared_down, b),
        ]);
        leaves
    }

    /// The group section: all-gather -> busiest rank's experts ->
    /// reduce-scatter, with every rank's aux-stream shared expert and mHC
    /// coefficient branch beside it.
    pub fn compile_exchange(&self, ranks: usize, b: &mut CostTreeBuilder) -> CostNode {
        let f = &self.ffn;
        let chain = CostNode::Sum(vec![
            self.dispatch.compile(b),
            f.fused_moe.compile(b),
            self.combine.compile(b),
        ]);
        let shared = max_over_ranks(
            (0..ranks)
                .map(|_| {
                    CostNode::Sum(vec![
                        f.shared_gate_up.compile(b),
                        f.shared_act.compile(b),
                        f.shared_down.compile(b),
                    ])
                })
                .collect(),
        );
        let coefficients = max_over_ranks(
            (0..ranks)
                .map(|_| f.mhc_overlap.compile_beside(Vec::new(), b))
                .collect(),
        );
        CostNode::Labeled {
            label: format!(
                "{} exchange [MXFP8 all_gatherv -> EP{} routed experts (busiest rank) -> \
                 bf16 reduce_scatterv; aux-stream shared expert (<=256 local rows) and mHC \
                 coefficients beside, Max over DP ranks]",
                self.name, self.resolved.raw_cfg.ffn.ep_size
            ),
            child: Box::new(CostNode::Parallel {
                overlap: 1.0,
                children: vec![chain, shared, coefficients],
            }),
        }
    }

    /// One rank's shared + routed add after the combine.
    pub fn compile_post(&self, b: &mut CostTreeBuilder) -> CostNode {
        self.ffn.residual_add.compile(b)
    }

    pub fn eval_local(&self, input: &DeepseekV41MoeFfnDpRankInput, ev: &mut Evaluator) {
        let f = &self.ffn;
        let rows = input.num_tokens;
        let zero = rows == 0;
        let mhc_overlap = !zero && input.mhc_overlap;
        // The inline copies run only above the aux-stream threshold.
        let inline_off = zero || shared_on_aux_stream(rows, false);
        let ew = ElementwiseKernelInput { num_tokens: rows };
        let gemm = SingleGemmKernelInput { m: rows };
        eval_or_zero(
            &f.entry,
            MhcRmsNormKernelInput { num_tokens: rows },
            zero || mhc_overlap,
            ev,
        );
        f.mhc_overlap.eval_entry(rows, mhc_overlap, ev);
        eval_or_zero(&f.router_gate, GemmFp32OutputKernelInput { m: rows }, zero, ev);
        eval_or_zero(&f.topk, ew.clone(), zero, ev);
        eval_or_zero(&f.moe_input_quant, ew.clone(), zero, ev);
        eval_or_zero(&f.shared_gate_up, gemm.clone(), inline_off, ev);
        eval_or_zero(&f.shared_act, ew, inline_off, ev);
        eval_or_zero(&f.shared_down, gemm, inline_off, ev);
    }

    pub fn eval_exchange(&self, input: &DeepseekV41MoeFfnDpEpWorkletInput, ev: &mut Evaluator) {
        assert_eq!(
            input.ranks.len(),
            input.collective_tokens.len(),
            "one collective count per rank"
        );
        let f = &self.ffn;
        let gathered: u32 = input.collective_tokens.iter().sum();
        let idle = gathered == 0;
        let collective = MoeEpCollectiveKernelInput {
            per_rank_tokens: input.collective_tokens.clone(),
        };
        eval_or_zero(&self.dispatch, collective.clone(), idle, ev);
        eval_or_zero(
            &f.fused_moe,
            Nvfp4FusedMoeKernelInput {
                num_tokens: gathered,
            },
            idle,
            ev,
        );
        eval_or_zero(&self.combine, collective, idle, ev);
        for rank in &input.ranks {
            let rows = rank.num_tokens;
            let off = !shared_on_aux_stream(rows, false);
            eval_or_zero(&f.shared_gate_up, SingleGemmKernelInput { m: rows }, off, ev);
            eval_or_zero(&f.shared_act, ElementwiseKernelInput { num_tokens: rows }, off, ev);
            eval_or_zero(&f.shared_down, SingleGemmKernelInput { m: rows }, off, ev);
        }
        for rank in &input.ranks {
            let live = rank.num_tokens > 0 && rank.mhc_overlap;
            f.mhc_overlap.eval_side(rank.num_tokens, live, false, ev);
        }
    }

    pub fn eval_post(&self, input: &DeepseekV41MoeFfnDpRankInput, ev: &mut Evaluator) {
        let rows = input.num_tokens;
        eval_or_zero(
            &self.ffn.residual_add,
            ElementwiseKernelInput { num_tokens: rows },
            rows == 0,
            ev,
        );
    }
}

fn max_over_ranks(children: Vec<CostNode>) -> CostNode {
    CostNode::Max {
        overlap: 1.0,
        children,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worklet::deepseek_v41_moe_ffn_ep::tests::config as ep_config;

    fn config() -> DeepseekV41MoeFfnDpEpWorkletConfig {
        let mut ffn = ep_config();
        ffn.tp_size = 1;
        DeepseekV41MoeFfnDpEpWorkletConfig {
            ffn,
            collective_backends: vec!["vllm_pynccl"],
            max_rank_tokens: 2048,
        }
    }

    #[test]
    fn resolves_tp1_shapes_and_mxfp8_dispatch() {
        let r = DeepseekV41MoeFfnDpEpWorklet::resolve_config(&config());
        assert_eq!(r.ffn.fused_moe.num_local_experts.get(), 96);
        // Whole shared expert per rank: 2 x 2304 gate|up, 2304 down.
        assert_eq!(r.ffn.shared_gate_up.n.get(), 4608);
        assert_eq!(r.ffn.shared_down.k.get(), 2304);
        assert!(r.ffn.all_reduce.is_none());
        assert_eq!(r.dispatch.activation_dtype, DType::Mxfp8E4m3);
        assert_eq!((r.dispatch.num_gpus, r.dispatch.top_k.get()), (4, 6));
        assert_eq!(r.combine.max_total_tokens, 8192);
    }

    #[test]
    #[should_panic(expected = "TP1")]
    fn rejects_tensor_parallel_ffn() {
        let mut cfg = config();
        cfg.ffn.tp_size = 4;
        DeepseekV41MoeFfnDpEpWorklet::resolve_config(&cfg);
    }
}
