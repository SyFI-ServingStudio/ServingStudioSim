//! Shared pieces of the DeepSeek-V4.1-Flash vLLM worklets.
//!
//! - The production stream-overlap gates. vLLM decides per forward whether a
//!   side stream runs, so each gated side branch is minted twice (concurrent
//!   under a `Parallel`, and serial after it) and `eval` fills the copy the
//!   batch selects, as `glm52_vllm_nvfp4_dsa_moe` does for its shared expert.
//!   The tree stays fixed (INV-1) and `Parallel` still sums the hidden copy's
//!   work.
//! - Elementwise byte placeholders for launches with no L1 kind. Each
//!   placeholder is named after the launch it stands for and sized by the
//!   bytes it reads and writes per token; the source anchor is on the config
//!   field that builds it.
//! - The NCCL AllGather stand-in (see [`all_gather_as_all_reduce_bytes`]).

use crate::common::Fabric;
use crate::op::attention::byte_rate_placeholder_shape;
use crate::op::Op;
use crate::timing::kernels::{AllReduceKernelConfig, ElementwiseKernelConfig};
use crate::timing::{CostNode, CostTreeBuilder, Evaluator, LeafMetrics, Probe, SlotInput};

/// `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` (fork `envs.py:299`, read in
/// `model_executor/layers/fused_moe/runner/shared_experts.py:111-116`): the
/// shared expert runs on the auxiliary stream only up to this many tokens.
/// 256 is a CUDA-graph capture size, so padding cannot cross it.
pub const SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: u32 = 256;

/// `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` (fork `envs.py:300`, the
/// `enable=` of `execute_in_parallel` in `models/deepseek_v41/attention.py`
/// `_run_parallel_input_projections`): the compressor and indexer-weight input
/// GEMMs fan out beside `fused_wqa_wkv` only up to this many tokens.
pub const MULTI_STREAM_GEMM_TOKEN_THRESHOLD: u32 = 1024;

/// Suffix of the serial copy of a gated side-branch leaf.
pub const SERIAL_COPY_SUFFIX: &str = "serial";

/// Whether the attention's `maybe_execute_in_parallel` aux stream is live.
///
/// `utils/multi_stream_utils.py:46-50` drops the aux stream while a breakable
/// CUDA-graph capture is active, which is how the piecewise graphs used by
/// mixed batches are captured; the FULL graphs replayed for uniform decode keep
/// it. Capture 2 (job 1185, device 0): all 2113 decode iterations ran the
/// compressor state/insert kernels on a side stream (4 per iteration), all 123
/// mixed/prefill iterations ran them on the main stream. The FULL graphs cover
/// the capture's decode sizes (11 sizes, up to `max_full_graph_tokens`).
pub fn attention_aux_stream_live(
    decode_only: bool,
    num_tokens: u32,
    max_full_graph_tokens: u32,
) -> bool {
    decode_only && num_tokens > 0 && num_tokens <= max_full_graph_tokens
}

/// Message size that makes the `all_reduce` NCCL curve stand in for a ring
/// AllGather producing `gathered_bytes` per rank.
///
/// There is no all-gather L1 kind. A ring AllGather moves `(N-1)/N` of its
/// output per rank and a ring AllReduce moves `2(N-1)/N` of its buffer, so an
/// all-reduce of half the gathered buffer moves the same bytes. At small sizes
/// the curve's two-phase latency floor (~27 us on B200 TP4) overstates the
/// RING_LL AllGather (15 us for the 48-token Engram gather in capture 2); at
/// the 2048-token mixed size it reads ~57 us against a measured ~70 us.
pub fn all_gather_as_all_reduce_bytes(gathered_bytes: u64) -> u64 {
    gathered_bytes.div_ceil(2)
}

pub(super) fn nccl_all_gather_proxy(
    backends: &[&'static str],
    gpu_name: &str,
    num_gpus: u32,
) -> AllReduceKernelConfig {
    AllReduceKernelConfig {
        backends: backends.to_vec(),
        gpu_name: gpu_name.to_string(),
        num_gpus,
        fabric: Fabric::Nvlink,
    }
}

/// An elementwise byte placeholder, shaped by
/// [`byte_rate_placeholder_shape`] (high fan-in reduces stream their bytes
/// evenly so the `triton` runner measures bandwidth, not a serial loop).
pub(super) fn placeholder(
    backends: &[&'static str],
    gpu_name: &str,
    input_bytes_per_token: u32,
    output_bytes_per_token: u32,
) -> ElementwiseKernelConfig {
    let (input_bytes_per_token, output_bytes_per_token) =
        byte_rate_placeholder_shape(input_bytes_per_token, output_bytes_per_token);
    ElementwiseKernelConfig {
        backends: backends.to_vec(),
        gpu_name: gpu_name.to_string(),
        input_bytes_per_token: input_bytes_per_token.into(),
        output_bytes_per_token: output_bytes_per_token.into(),
    }
}

pub(super) fn eval_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    if zero {
        ev.push(LeafMetrics::ZERO, || input.into());
    } else {
        op.eval(&input, ev);
    }
}

/// The serial copy of a gated side-branch leaf: same kernel, suffixed name.
pub(super) fn compile_serial_copy<K: Probe>(op: &Op<K>, builder: &mut CostTreeBuilder) -> CostNode {
    builder.leaf(
        format!("{}.{SERIAL_COPY_SUFFIX}", op.name),
        op.kernel.kind(),
        op.kernel.describe_config(),
    )
}

/// `main` with its gated side branches: `Sum[Parallel{main, side...}, serial...]`.
///
/// Each element of `concurrent` is one side stream's branch; `serial` holds
/// the same branches' serial copies. With no side branch this is just `main`.
pub(super) fn gated_fanout(
    main: Vec<CostNode>,
    concurrent: Vec<CostNode>,
    serial: Vec<CostNode>,
) -> CostNode {
    if concurrent.is_empty() {
        return CostNode::Sum(main);
    }
    let mut children = vec![CostNode::Sum(main)];
    children.extend(concurrent);
    CostNode::Sum(vec![
        CostNode::Parallel {
            overlap: 1.0,
            children,
        },
        CostNode::Sum(serial),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aux_stream_is_live_only_for_full_graph_decode() {
        assert!(attention_aux_stream_live(true, 48, 64));
        assert!(!attention_aux_stream_live(true, 65, 64));
        assert!(!attention_aux_stream_live(false, 48, 64));
        assert!(!attention_aux_stream_live(true, 0, 64));
    }

    #[test]
    fn all_gather_proxy_matches_ring_bytes() {
        // 48 tokens x 24 heads x 256 x bf16 gathered -> half as an all-reduce.
        assert_eq!(all_gather_as_all_reduce_bytes(48 * 12_288), 294_912);
    }

    #[test]
    fn fanout_without_side_branches_is_the_main_path() {
        let node = gated_fanout(vec![CostNode::Leaf(0)], Vec::new(), Vec::new());
        assert!(matches!(node, CostNode::Sum(ref children) if children.len() == 1));
        let node = gated_fanout(
            vec![CostNode::Leaf(0)],
            vec![CostNode::Leaf(1)],
            vec![CostNode::Leaf(2)],
        );
        let CostNode::Sum(children) = node else {
            panic!("gated fanout is a Sum")
        };
        assert!(
            matches!(children[0], CostNode::Parallel { ref children, .. } if children.len() == 2)
        );
        assert!(matches!(children[1], CostNode::Sum(ref serial) if serial.len() == 1));
    }
}
