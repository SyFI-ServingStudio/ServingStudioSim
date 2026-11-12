//! One TP sublayer boundary under vLLM's `allreduce_rms` fusion pass: the
//! all-reduce of a row-parallel output, the residual add and the next RMSNorm.
//!
//! The pass rewrites the boundary into one FlashInfer call only for compile
//! ranges up to its token cap (`AllReduceFusionPass.is_applicable_for_range`,
//! `allreduce_rms_fusion.py:1128-1132`: the workspace budget of
//! `FI_ALLREDUCE_FUSION_MAX_SIZE_MB`, `:104-121`, over the hidden row) and only
//! at a TP FlashInfer fuses for (`flashinfer_max_size`, `config/compilation.py:
//! 194-207`; the pass is off at TP 1, `:988-991`). Every other step keeps the
//! unfused graph: `CudaCommunicator.all_reduce` (`cuda_communicator.py:335-400`)
//! and then the residual add + RMSNorm, which Inductor emits as one Triton
//! kernel.
//!
//! Above the fused cap the standalone all-reduce's own budgets are spent too:
//! FlashInfer's standalone workspace defaults to the same table
//! (`flashinfer_all_reduce.py:378-397`; SM100 has no tuned MNNVL entry,
//! `all_reduce_utils.py:87-92`), custom AR stops at 2 MiB and torch symmetric
//! memory at 32 MiB at TP 4 (`all_reduce_utils.py:38-43, 65-70`), so the call
//! is pynccl. At TP 8 symmetric memory reaches 128 MiB; the measured NCCL
//! curve stands in for it, as in the GLM-5.3-Flash arch.
//!
//! All leaves are compiled whichever path a step takes; the inactive ones are
//! pushed as zero, so the slot layout does not depend on the batch (INV-1).

use crate::op::Op;
use crate::timing::kernels::{
    AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput, AllReduceResidualRmsNormKernel,
    AllReduceResidualRmsNormKernelConfig, AllReduceResidualRmsNormKernelInput,
    AllReduceResidualRmsNormSpec, ElementwiseKernel, ElementwiseKernelConfig,
    ElementwiseKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Evaluator, LeafMetrics, PerfApiBridge, Probe, SlotInput,
};

#[derive(Clone, Debug)]
pub struct TpAllReduceRmsNormConfig {
    /// The fused FlashInfer boundary; `None` where vLLM does not fuse (one
    /// GPU, or a TP or GPU without a FlashInfer budget).
    pub fused: Option<AllReduceResidualRmsNormKernelConfig>,
    /// The standalone all-reduce of the unfused path; `None` on one GPU.
    pub all_reduce: Option<AllReduceKernelConfig>,
    /// The unfused residual add + RMSNorm.
    pub norm: ElementwiseKernelConfig,
    /// All-reduced bytes per token (the hidden row).
    pub all_reduce_bytes_per_token: u64,
}

pub struct TpAllReduceRmsNormOp {
    fused: Option<Op<AllReduceResidualRmsNormKernel>>,
    all_reduce: Option<Op<AllReduceKernel>>,
    norm: Op<ElementwiseKernel>,
    /// Largest step the fused call takes; 0 without one.
    max_fused_tokens: u32,
    all_reduce_bytes_per_token: u64,
}

impl TpAllReduceRmsNormOp {
    /// `name` is the fused leaf's path; the unfused leaves are its
    /// `_all_reduce` and `_residual_norm` siblings.
    pub fn build(
        name: &str,
        config: TpAllReduceRmsNormConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let max_fused_tokens = config
            .fused
            .as_ref()
            .map_or(0, AllReduceResidualRmsNormSpec::max_fused_tokens);
        let fused = config
            .fused
            .map(|fused| {
                atomic(
                    name.to_string(),
                    fused,
                    AllReduceResidualRmsNormKernel::build,
                    bridge,
                )
            })
            .transpose()?;
        let all_reduce = config
            .all_reduce
            .map(|ar| {
                atomic(
                    format!("{name}_all_reduce"),
                    ar,
                    AllReduceKernel::build,
                    bridge,
                )
            })
            .transpose()?;
        let norm = atomic(
            format!("{name}_residual_norm"),
            config.norm,
            ElementwiseKernel::build,
            bridge,
        )?;
        Ok(Self {
            fused,
            all_reduce,
            norm,
            max_fused_tokens,
            all_reduce_bytes_per_token: config.all_reduce_bytes_per_token,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let mut leaves = Vec::with_capacity(3);
        if let Some(fused) = &self.fused {
            leaves.push(fused.compile(builder));
        }
        if let Some(all_reduce) = &self.all_reduce {
            leaves.push(all_reduce.compile(builder));
        }
        leaves.push(self.norm.compile(builder));
        CostNode::Sum(leaves)
    }

    /// Whether a step of `num_tokens` takes the fused call: vLLM's compiled
    /// graph is chosen by the launched (graph-padded) token count.
    pub fn takes_fused(&self, num_tokens: u32) -> bool {
        self.fused.is_some() && num_tokens <= self.max_fused_tokens
    }

    pub fn eval(&self, num_tokens: u32, ev: &mut Evaluator) {
        let fused_step = self.takes_fused(num_tokens);
        if let Some(fused) = &self.fused {
            let input = AllReduceResidualRmsNormKernelInput { num_tokens };
            push(fused, input, !fused_step, ev);
        }
        let unfused = !fused_step && num_tokens > 0;
        if let Some(all_reduce) = &self.all_reduce {
            let input = AllReduceKernelInput {
                message_size_bytes: u64::from(num_tokens) * self.all_reduce_bytes_per_token,
            };
            push(all_reduce, input, !unfused, ev);
        }
        push(
            &self.norm,
            ElementwiseKernelInput { num_tokens },
            !unfused,
            ev,
        );
    }
}

fn atomic<K, C, F>(
    name: String,
    config: C,
    build_kernel: F,
    bridge: &PerfApiBridge,
) -> Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError>,
{
    let kernel = build_kernel(name.clone(), config, bridge)?;
    Ok(Op::new(name, std::sync::Arc::new(kernel)))
}

/// Push the op's metrics, or zero when the launch does not happen this step.
fn push<K>(op: &Op<K>, input: K::Input, zero: bool, ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        op.kernel.eval(&input)
    };
    ev.push(metrics, || input.into());
}
