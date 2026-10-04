"""One production packed-MXFP4 Marlin MoE GEMM launch."""

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

KIND = "mxfp4_marlin_moe_gemm"


@dataclass(frozen=True)
class Mxfp4MarlinMoeGemmArgs(KernelArgs):
    m: int = arg(unit="tokens", doc="Activation rows entering this expert projection.")
    n: int = arg(unit="elements", doc="Output features of this expert projection.")
    k: int = arg(unit="elements", doc="Input features of this expert projection.")
    dtype: DType = arg(doc="Element type of the activation and output.")
    input_top_k: int = arg(
        unit="experts", doc="Expert selections per activation row in this launch."
    )
    block_size_m: int = arg(unit="tokens", doc="Routed token rows in each aligned expert block.")
    mul_topk_weights: bool = arg(
        doc="Whether the projection output is multiplied by router weights."
    )
    per_group_batches: tuple[int, ...] = arg(
        unit="tokens", doc="Selected token count for each local expert, in expert order."
    )


DOC = KernelDoc(
    title="MXFP4 Marlin MoE GEMM",
    summary="Multiply routed BF16 activations by packed MXFP4 expert weights in one Marlin launch.",
    description=(
        "Routed experts run both expert projections as this "
        "Marlin GEMM on MXFP4 weights. The first projection (FC1) reads each "
        "token for its six expert selections and does not apply router weights;"
        " the second (FC2) reads one row per selection and multiplies by the "
        "router weight. The local rows are aligned into block_size_m-row expert"
        " blocks before timing."
    ),
    category="MoE",
    subcategory="Expert compute",
    formula=(
        "local_rows = sum(per_group_batches)",
        "active_experts = count(per_group_batches > 0)",
        "TFLOPS = 2·local_rows·n·k / time",
        "logical_bytes = 2·local_rows·k + active_experts·n·k/2 "
        "+ active_experts·n·⌊k/32⌋ + 2·local_rows·n "
        "+ (4·local_rows if mul_topk_weights else 0)",
        "GB/s = logical_bytes / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "Five warm-up calls run first; route alignment, weight packing and an "
        "output check are not timed."
    ),
    caveats=(
        "TFLOPS counts only routed rows, not work on padded expert blocks. GB/s "
        "counts logical BF16 activations and outputs, packed weights and scales, "
        "and router weights only when applied.",
        "Marlin's thread tiles need n % 64 = 0 and k % 128 = 0, or n % 128 = 0 "
        "and k % 64 = 0. Measured rows use 64 local experts on H200: FC1 "
        "with n = k = 4096 and input_top_k = 6, and FC2 with "
        "n = 4096, k = 2048 and input_top_k = 1.",
    ),
    # The measured vLLM operation has no separate PyTorch reference module.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_marlin",
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.mxfp4_marlin_moe_gemm_vllm_marlin",
            function_name="profile_mxfp4_marlin_moe_gemm_vllm_marlin",
        ),
        table_name=KIND,
        args_schema=Mxfp4MarlinMoeGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "vLLM moe_wna16_marlin_gemm with packed MXFP4 weights and aligned expert routes."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py",
        ),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        subprocess_env="vllm_env",
    )
)

__all__ = ["KIND"]
