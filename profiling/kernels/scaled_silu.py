"""SiLU of an input divided by a constant, as one Triton launch."""

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

KIND = "scaled_silu"


@dataclass(frozen=True)
class ScaledSiluArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows.")
    width: int = arg(unit="elements", doc="Features per row.")
    hidden_dtype: DType = arg(doc="Element type of the input and output.")


DOC = KernelDoc(
    title="Scaled SiLU",
    summary="Apply SiLU to every element after dividing it by a constant.",
    description=(
        "Gated hyper-connections apply it to the low-rank down projection of the "
        "normalized streams, dividing by the stream count before SiLU. One Triton "
        "program per row covers the row in one power-of-two block. Unlike "
        "silu_and_mul there is no up half: input and output have the same width. "
        "Inputs are seeded random bf16 rows."
    ),
    category="Other",
    formula=(
        "y = SiLU(x / c), computed in fp32",
        "GB/s = 4 · num_tokens · width / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Three warm-up calls and a comparison with an independent "
        "PyTorch reference precede capture; every launch of the call is counted."
    ),
    caveats=(
        "The divisor is a compile-time scalar and does not change the time; rows use c = 4.",
        "The elementwise byte pass does not price it: on B200 it matches below about "
        "1k rows of 320 but is 2.3x to 4x faster from 4k to 16k rows.",
    ),
    reference="profiling.runners.mhc.hc_reference",
)


# A portable Triton kernel; Triton lowers bf16 only on SM80+. Programmatic
# dependent launch is enabled on SM90+ by the op itself and is part of the launch.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.mhc.hc_vllm_triton",
            function_name="profile_scaled_silu_vllm_triton",
        ),
        table_name=KIND,
        args_schema=ScaledSiluArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="The fork's hc_silu Triton op.",
            url="https://github.com/serendipity-zk/vllm/blob/892da0822f7e866cec8e519727f014c45a5ad792/vllm/models/qwen4_exp/nvidia/ops/hc.py",
        ),
    )
)

__all__ = ["KIND", "ScaledSiluArgs"]
