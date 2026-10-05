"""Clamped SwiGLU for routed and shared experts."""

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

KIND = "clamped_swiglu"


@dataclass(frozen=True)
class ClampedSwigluArgs(KernelArgs):
    num_rows: int = arg(
        unit="rows",
        doc=(
            "Rows entering the activation: token-expert pairs for routed experts, "
            "tokens for the shared expert."
        ),
    )
    hidden_dim: int = arg(
        unit="elements", doc="Width of the gate half and of the up half of each row."
    )
    dtype: DType = arg(doc="Element type of the input and output.")


DOC = KernelDoc(
    title="Clamped SwiGLU",
    summary=(
        "Clamp the gate and up projections, then multiply SiLU of the gate by the up projection."
    ),
    description=(
        "Routed and shared experts apply this activation to the output of "
        "their gate-up projection. Each input row holds a gate half and "
        "an up half, each hidden_dim wide; the gate is capped at 10 and the up "
        "half clamped to [−10, 10] before the SiLU product. The measurement uses "
        "seeded random bf16 rows."
    ),
    category="MoE",
    subcategory="Expert compute",
    formula=(
        "y = SiLU(min(gate, 10)) · clamp(up, −10, 10)",
        "GB/s = 6 · num_rows · hidden_dim / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Three untimed calls and a comparison with the PyTorch "
        "reference precede capture. Every launch of vLLM's callable is counted."
    ),
    caveats=(
        "GB/s counts the two bf16 input halves and one bf16 output, without "
        "counting any temporary traffic.",
    ),
    reference="profiling.runners.moe.clamped_swiglu_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_inductor",
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.clamped_swiglu_vllm_inductor",
            function_name="profile_clamped_swiglu_vllm_inductor",
        ),
        table_name=KIND,
        args_schema=ClampedSwigluArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's swiglu_limit_func without topk_ids, which dispatches to its "
                "CUDA silu_and_mul_with_clamp op."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/utils.py",
        ),
    )
)

__all__ = ["KIND"]
