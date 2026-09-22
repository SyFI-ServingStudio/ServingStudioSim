"""Kimi-K3 absorbed MLA decode attention profiling contract."""

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

KIND = "mla_decode_attention"
_RUNNER = "profiling.runners.attention.mla_decode_attention"
_SUPPORT = BackendSupport(
    compute=frozenset({DType.BF16, DType.FP8_E4M3}),
    kv=frozenset({DType.BF16, DType.FP8_E4M3}),
    gpus=frozenset({"NVIDIA B200"}),
)


@dataclass(frozen=True)
class MlaDecodeAttentionArgs(KernelArgs):
    num_heads: int
    kv_lora_rank: int
    rope_dim: int
    q_dtype: DType
    kv_dtype: DType
    page_size: int
    batch_size: int
    kv_len: int


for _backend, _function in (
    ("sglang_cutedsl_mla", "profile_mla_decode_attention_sglang_cutedsl"),
    ("sglang_trtllm_mla", "profile_mla_decode_attention_sglang_trtllm"),
    ("sglang_triton", "profile_mla_decode_attention_sglang_triton"),
):
    register(
        KernelProfilerSpec(
            kernel_kind=KIND,
            backend=_backend,
            supports=_SUPPORT,
            runner_ref=RunnerRef(module_name=_RUNNER, function_name=_function),
            table_name=KIND,
            args_schema=MlaDecodeAttentionArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            subprocess_env="sglang_k3_env",
        )
    )
