"""Kimi-K3 SGLang causal-convolution prefill kernel."""

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

KIND = "causal_conv1d_prefill"


@dataclass(frozen=True)
class CausalConv1dPrefillArgs(KernelArgs):
    num_tokens: int
    max_sequence_length: int
    num_sequences: int
    prefix_len: int
    channels: int
    kernel_size: int
    dtype: DType
    state_dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kimi_k3_prefill",
            function_name="profile_causal_conv1d_prefill_sglang_triton",
        ),
        table_name=KIND,
        args_schema=CausalConv1dPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_k3_env",
    )
)
