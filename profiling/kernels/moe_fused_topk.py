"""Fused MoE softmax/top-k router-selection kernel kind.

The Torch backend is a multi-launch semantic baseline. Production simulation
uses vLLM's fused_topk CUDA backend.
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

KIND: str = "moe_fused_topk"


@dataclass(frozen=True)
class MoeFusedTopkArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens whose router logits are selected.")
    num_experts: int = arg(unit="experts", doc="Experts scored for each token.")
    top_k: int = arg(unit="experts", doc="Experts selected per token.")
    dtype: DType = arg(doc="Element type of the router logits.")


DOC = KernelDoc(
    title="Fused MoE top-k routing",
    summary="Select top-k experts from softmax router scores and renormalize their weights.",
    description=(
        "The step after the MoE router projection: from each token's "
        "num_experts logits, take the softmax, keep the top_k experts, and "
        "renormalize their probabilities. It returns the weights, the expert "
        "IDs and the source index of each routed row. The logits are fixed, "
        "tie-free bf16 values rather than model outputs."
    ),
    category="MoE",
    subcategory="Routing and combine",
    formula=(
        "p = softmax(logits); weights = top_k(p) / sum(top_k(p))",
        "source_index(token, slot) = slot · num_tokens + token",
        "TFLOPS = num_tokens · (5·num_experts − 2 + top_k·(num_experts − 1) "
        "− top_k·(top_k − 1)/2 + 2·top_k − 1) / time",
        "GB/s = (2·num_tokens·num_experts + 12·num_tokens·top_k) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "vllm_cuda counts every kernel of one fused_topk call: a topkGating "
        "specialization where vLLM has one for the expert count, otherwise the "
        "moeSoftmax and moeTopK pair. torch counts every launch of its "
        "sequence. Outputs are checked against the reference first."
    ),
    caveats=(
        "TFLOPS counts comparisons and exponentials as nominal operations.",
        "GB/s counts one logits read and the three outputs; the torch backend's"
        " workspaces are not counted.",
    ),
    reference="profiling.runners.moe.moe_fused_topk_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.moe_fused_topk_torch",
            function_name="profile_moe_fused_topk",
        ),
        table_name=KIND,
        args_schema=MoeFusedTopkArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="default_env",
        doc=BackendDoc(summary="PyTorch softmax and iterative top-k with preallocated workspaces."),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        # No capability rule: a vLLM _moe_C CUDA op built for every arch the wheel targets.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.moe_fused_topk_vllm_cuda",
            function_name="profile_moe_fused_topk_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=MoeFusedTopkArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="vLLM fused_topk: one topkGating launch for a specialized expert count.",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py",
        ),
    )
)
