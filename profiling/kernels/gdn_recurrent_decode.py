"""Qwen Gated DeltaNet recurrent-decode kernel kind.

The initial ``torch`` backend measures the complete multi-launch semantic
reference. It is a correctness/performance baseline, not the production fused
decode launch, and must not be selected for production simulation after the
vLLM backend is registered.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "gdn_recurrent_decode"


@dataclass(frozen=True)
class GdnRecurrentDecodeArgs(KernelArgs):
    batch_size: int
    num_qk_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    dtype: DType
    state_dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_recurrent_decode_torch"),
            function_name="profile_gdn_recurrent_decode",
        ),
        table_name=KIND,
        args_schema=GdnRecurrentDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_recurrent_decode_vllm_triton"),
            function_name="profile_gdn_recurrent_decode_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnRecurrentDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
