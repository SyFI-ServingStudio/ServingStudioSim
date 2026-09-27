"""Routed-expert BF16 top-k reduction."""

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

KIND = "moe_sum"


@dataclass(frozen=True)
class MoeSumArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens whose expert outputs are reduced.")
    top_k: int = arg(unit="experts", doc="Expert outputs summed for each token.")
    hidden_dim: int = arg(unit="elements", doc="Features in each expert output.")
    dtype: DType = arg(doc="Element type of the expert outputs.")


DOC = KernelDoc(
    title="MoE expert sum",
    summary="Sum the routed expert output rows for each token.",
    description=(
        "Routed experts leave top_k output rows per token; this "
        "step sums them into one hidden_dim-wide row. vLLM's moe_sum has "
        "dedicated kernels only for small top_k in the measured build, so top_k"
        " = 6 falls back to torch's sum over the expert dimension. The expert "
        "rows are random bf16 values."
    ),
    category="MoE",
    subcategory="Routing and combine",
    formula=(
        "output[token, h] = Σₖ expert_output[token, k, h]",
        "TFLOPS = num_tokens · hidden_dim · (top_k − 1) / time",
        "GB/s = 2·num_tokens·hidden_dim·(top_k + 1) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Five warm-up calls run first. Only PyTorch reduce_kernel launches, the"
        " fallback's kernel, are counted."
    ),
    caveats=(
        "Only top_k = 6 and hidden_dim = 4096 in bf16 on H200 are measured.",
        "Newer vLLM has a dedicated vectorized kernel for top_k = 6, so these "
        "rows describe the fallback.",
    ),
    reference="profiling.runners.moe.moe_sum_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.moe_sum_vllm_cuda",
            function_name="profile_moe_sum_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=MoeSumArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="vLLM's _custom_ops.moe_sum, which falls back to torch sum at top_k = 6.",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py",
        ),
    )
)

__all__ = ["KIND"]
