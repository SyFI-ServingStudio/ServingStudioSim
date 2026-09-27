"""Kimi-K3 SGLang TRT-LLM ragged MLA prefill attention."""

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

KIND = "mla_prefill_attention"


@dataclass(frozen=True)
class MlaPrefillAttentionArgs(KernelArgs):
    num_heads: int
    qk_head_dim: int
    v_head_dim: int
    q_dtype: DType
    kv_dtype: DType
    o_dtype: DType
    causal: bool
    batch_size: int
    q_len: int
    kv_len: int
    prefix_len: int
    num_prefix_chunks: int


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_trtllm_mla",
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kimi_k3_prefill",
            function_name="profile_mla_prefill_attention_sglang_trtllm",
        ),
        table_name=KIND,
        args_schema=MlaPrefillAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_k3_env",
    )
)
