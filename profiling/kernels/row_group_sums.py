"""Row group sums: the fp32 sum of every fixed-size group of each activation row."""

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

KIND = "row_group_sums"


@dataclass(frozen=True)
class RowGroupSumsArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Activation rows summed.")
    hidden: int = arg(unit="elements", doc="Elements in each row (the GEMM's K).")
    group_size: int = arg(unit="elements", doc="Elements summed into each output value.")
    dtype: DType = arg(doc="Element type of the rows; the sums are fp32.")


DOC = KernelDoc(
    title="Row group sums",
    summary="Sum each row's consecutive group_size-element groups in fp32.",
    description=(
        "An affine 4-bit weight w = q · s + b dequantizes per group of K, so "
        "x · w = Σ_g s_g (x · q)_g + b_g Σ_g x: the GEMM needs every row's group sums "
        "Σ_g x. TensorFold's W4A16 drafter GEMM (qmm) computes them in a separate small "
        "Triton launch before the matmul whenever the caller has not already got them "
        "from a fused norm, so it is its own kernel: it reads the bf16 rows once and "
        "writes hidden / group_size fp32 values a row."
    ),
    category="Other",
    formula=(
        "out[m, g] = Σ_{j<group_size} x[m, g · group_size + j]  (fp32)",
        "GB/s = num_tokens · (2 · hidden + 4 · hidden / group_size) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} A guard call is compared with fp32 Torch sums before capture; "
        "only the _group_sums launch is counted."
    ),
    caveats=("Bandwidth bound; the fp32 sum order differs from Torch's by rounding only.",),
    reference="profiling.runners.gemm.row_group_sums_reference",
)


# TensorFold perf-study 6469730a, tensorfold/cuda/kernels/qmm.py `group_sums(x, gs)`:
# the Triton `_group_sums` on grid (rows, cdiv(K / gs, 16)), two warps, that every
# tiled 4-bit `matmul` without precomputed sums launches. bf16 rows (tl.float32
# accumulate of bf16 loads); plain Triton, sm_80 for bf16.
TENSORFOLD_TRITON = "tensorfold_triton"

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend=TENSORFOLD_TRITON,
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.row_group_sums_tensorfold_triton",
            function_name="profile_row_group_sums_tensorfold_triton",
        ),
        table_name=KIND,
        args_schema=RowGroupSumsArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's Triton _group_sums (cuda/kernels/qmm.py group_sums): the "
                "activation sums its W4A16 drafter GEMM reads."
            ),
        ),
    )
)

__all__ = ["KIND", "RowGroupSumsArgs"]
