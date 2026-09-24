//! Shared construction helpers for the GLM-5.3-Flash worklets.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::kernels::ElementwiseKernelConfig;
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Evaluator, LeafMetrics, PerfApiBridge, Probe, SlotInput,
};

/// Build one atomic op named `{prefix}.{suffix}`.
pub(crate) fn atomic<K, C, F>(
    prefix: &str,
    suffix: &str,
    config: C,
    build_kernel: F,
    bridge: &PerfApiBridge,
) -> Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError>,
{
    let name = format!("{prefix}.{suffix}");
    Ok(Op::new(
        name.clone(),
        Arc::new(build_kernel(name, config, bridge)?),
    ))
}

/// Push the op's metrics, or zero when the launch does not happen this
/// iteration. The slot exists either way (INV-1).
pub(crate) fn push_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        op.kernel.eval(&input)
    };
    ev.push(metrics, || input.clone().into());
}

/// `n` identical launches of one leaf: the leaf is costed once and folded.
pub(crate) fn repeated<K: Probe>(op: &Op<K>, n: u32, builder: &mut CostTreeBuilder) -> CostNode {
    CostNode::Scale {
        n,
        child: Box::new(op.compile(builder)),
    }
}

/// A byte-sized elementwise placeholder.
pub(crate) fn elementwise(
    backends: &[&'static str],
    gpu_name: &str,
    input_bytes_per_token: u32,
    output_bytes_per_token: u32,
) -> ElementwiseKernelConfig {
    ElementwiseKernelConfig {
        backends: backends.to_vec(),
        gpu_name: gpu_name.to_string(),
        input_bytes_per_token: input_bytes_per_token.into(),
        output_bytes_per_token: output_bytes_per_token.into(),
    }
}
