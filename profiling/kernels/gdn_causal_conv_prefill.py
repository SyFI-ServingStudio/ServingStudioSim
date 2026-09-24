"""Qwen GDN causal-convolution prefill kernel kind.

The initial ``torch`` backend measures the complete multi-launch semantic
reference. It is a correctness/performance baseline, not the production fused
prefill launch, and must not be selected for production simulation after the
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

KIND: str = "gdn_causal_conv_prefill"


@dataclass(frozen=True)
class GdnCausalConvPrefillArgs(KernelArgs):
    batch_size: int
    sequence_length: int
    channels: int
    kernel_size: int
    dtype: DType
    state_dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_causal_conv_prefill_torch"),
            function_name="profile_gdn_causal_conv_prefill",
        ),
        table_name=KIND,
        args_schema=GdnCausalConvPrefillArgs,
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
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_causal_conv_prefill_vllm_triton"),
            function_name="profile_gdn_causal_conv_prefill_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnCausalConvPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
