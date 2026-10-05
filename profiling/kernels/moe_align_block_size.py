"""vLLM MoE token-to-expert block alignment."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND = "moe_align_block_size"


@dataclass(frozen=True)
class MoeAlignBlockSizeArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens whose expert assignments are aligned.")
    num_experts: int = arg(unit="experts", doc="Experts receiving routed tokens.")
    top_k: int = arg(unit="experts", doc="Expert assignments per token.")
    block_size: int = arg(unit="tokens", doc="Routed-token positions in each expert block.")


DOC = KernelDoc(
    title="MoE block alignment",
    summary="Group routed token IDs by expert and pad each group to a block boundary.",
    description=(
        "Before the grouped expert GEMM, vLLM sorts the routed token IDs by "
        "expert and pads each expert's group to a multiple of block_size, so "
        "every GEMM block belongs to one expert. It returns the sorted IDs, the"
        " expert of each block and the padded count. The measurement spreads "
        "the num_tokens · top_k routes evenly over the experts."
    ),
    category="MoE",
    subcategory="Routing and combine",
    formula=(
        "routes = num_tokens · top_k; padded_routes = Σₑ block_size · ⌈routesₑ / block_size⌉",
        "GB/s = 4·(routes + padded_routes + padded_routes/block_size + 1) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Five warm-up calls run first, and every launch of the call is counted."
        " Outputs are checked against the reference before timing."
    ),
    caveats=(
        "Routes are spread evenly; skewed expert popularity is not measured.",
        "GB/s counts the visible int32 inputs and outputs, not the kernel's private workspace.",
    ),
    reference="profiling.runners.moe.moe_align_block_size_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.moe_align_block_size_vllm_cuda",
            function_name="profile_moe_align_block_size_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=MoeAlignBlockSizeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # No capability rule: a vLLM _C CUDA op built for every arch the wheel targets.
        supports=BackendSupport(compute=None),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="vLLM moe_align_block_size CUDA operator with preallocated alignment outputs.",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/_custom_ops.py",
        ),
    )
)

__all__ = ["KIND"]
