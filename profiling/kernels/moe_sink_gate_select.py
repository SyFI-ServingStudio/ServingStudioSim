"""MoE gate select: sigmoid + bias top-k, then a joint renorm with the sink logits.

Source: alignment fork, ``vllm/models/inkling/nvidia/moe.py``
(``inkling_gate_select`` launching ``_inkling_gate_select_kernel``), called once
per MoE layer by ``InklingGate.select_experts`` on the router GEMM's fp32
logits. Per token of the ``[T, logits_row_stride]`` logits (``R`` routed
columns, then ``S`` sink columns, then GEMM padding):

- selection score ``sigmoid(l_e) + bias_e`` over the routed experts only;
- top ``K`` by that score, ties to the lowest expert id;
- the ``K`` selected *raw* logits plus the ``S`` sink logits are renormalized
  jointly, ``w = sigmoid(l) / sum(sigmoid(l))`` (a softmax of log-sigmoids),
  times ``route_scale`` and the checkpoint's global scale;
- outputs ``ids`` int32 and ``weights`` fp32, both ``[T, K + S]``; the last
  ``S`` columns are the sink experts (ids ``R .. R + S - 1``) and their gammas.

One Triton program per token, one launch per call. Not ``moe_fused_topk``
(softmax renorm, no bias, no sinks) nor ``moe_topk_softplus_sqrt``
(sqrt-softplus scores, fixed 1.5 scale): the score function, the sink renorm
and the ``[T, K + S]`` layout differ. The elementwise byte placeholder
predicts about 1 us at 2048 tokens against 11 us measured: the cost is the
``K`` serial block-wide argmax reductions per program, not the bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND = "moe_sink_gate_select"


@dataclass(frozen=True)
class MoeSinkGateSelectArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows of router logits in this call.")
    num_routed_experts: int = arg(unit="experts", doc="Routed experts top-k selects from.")
    num_shared_experts: int = arg(
        unit="experts", doc="Sink (shared) experts every token activates; their logits follow."
    )
    top_k: int = arg(unit="experts", doc="Routed experts selected per token.")
    logits_row_stride: int = arg(
        unit="elements",
        doc="Row stride of the logits: routed + sink columns plus the router GEMM's padding.",
    )
    has_bias: bool = arg(doc="Whether a per-expert selection bias is added to the scores.")
    has_global_scale: bool = arg(doc="Whether the weights are also multiplied by a global scale.")
    logits_dtype: DType = arg(doc="Element type of the router logits.")


DOC = KernelDoc(
    title="Sink-renormalized MoE gate select",
    summary=(
        "Pick each token's top-k routed experts by sigmoid score plus bias, then "
        "renormalize their logits together with the sink experts' logits."
    ),
    description=(
        "Per token, score the routed experts as sigmoid(logit) "
        "plus a selection bias and keep the top_k (ties to the lowest id). The "
        "selected raw logits and the sink experts' logits are renormalized "
        "jointly as sigmoid(l) / Σ sigmoid(l) and scaled by route_scale and the "
        "global scale. The output holds top_k + num_shared_experts ids and "
        "weights per token, the sink entries last. One Triton program per token."
    ),
    category="MoE",
    subcategory="Routing and combine",
    formula=(
        "score_e = sigmoid(l_e) + bias_e,  e < num_routed_experts",
        "A = top_k(score) ∪ sinks,  w_a = route_scale · gs · sigmoid(l_a) / Σ_A sigmoid(l)",
        "GB/s = num_tokens · 4 · (num_routed_experts + num_shared_experts "
        "+ 2 · (top_k + num_shared_experts)) / time",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} One inkling_gate_select call is one "
        "_inkling_gate_select_kernel launch; only that kernel is counted. Every "
        "spec checks the outputs' structure (sink ids, distinct in-range routed "
        "ids, weights summing to route_scale · gs) before timing. The comparison "
        "with the independent PyTorch reference runs only on fixed anchor shapes, "
        "through the runner's manual --oracle command, not during profiling."
    ),
    caveats=(
        "Logits are random fp32 values and the bias is synthetic; the kernel's "
        "work does not depend on them (K full-width argmax passes per token).",
        "route_scale and the global scale do not change the work; the runner "
        "uses route_scale 8 and global scale 1.25.",
        "top_k + num_shared_experts must be a power of two (a Triton arange "
        "spans the selected set).",
        "TFLOPS is not computed. GB/s counts the routed and sink logits read "
        "and the ids and weights written, not the row padding.",
    ),
    reference="profiling.runners.moe.moe_sink_gate_select_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.moe_sink_gate_select_vllm_triton",
            function_name="profile_moe_sink_gate_select_vllm_triton",
        ),
        table_name=KIND,
        args_schema=MoeSinkGateSelectArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # Portable Triton: elementwise math and block reductions, no tl.dot or
        # architecture-specific instructions, so no capability rule.
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
        ),
        # inkling_gate_select exists only in the upstream-rebased alignment fork.
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "The alignment fork's inkling_gate_select: one _inkling_gate_select_kernel "
                "Triton launch, one program per token."
            ),
            url=None,
        ),
    )
)

__all__ = ["KIND", "MoeSinkGateSelectArgs"]
