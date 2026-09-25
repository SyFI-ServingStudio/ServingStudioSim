"""Kimi-K3 fused attention-residual TMA launch used by chunked prefill."""

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

KIND = "k3_attn_res_prefill"


@dataclass(frozen=True)
class K3AttnResPrefillArgs(KernelArgs):
    num_tokens: int
    hidden_size: int
    num_valid_blocks: int
    num_launches: int
    write_prefix: bool
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_k3",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kimi_k3_prefill",
            function_name="profile_k3_attn_res_prefill_sglang_k3",
        ),
        table_name=KIND,
        args_schema=K3AttnResPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_k3_env",
    )
)
