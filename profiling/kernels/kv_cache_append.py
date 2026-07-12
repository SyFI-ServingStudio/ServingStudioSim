"""Paged KV-cache append kernel kind.

``kv_cache_append`` writes the newly produced K/V rows for ``num_tokens`` into
the paged cache slots selected by ``slot_mapping``.  The semantic operation is
framework-independent; the ``vllm_cuda`` backend wraps vLLM's
``reshape_and_cache_flash_kernel`` while ``torch`` is the small executable
reference used to validate the write contract.

The runtime sweep axis is ``num_tokens``.  KV-head count, head size, page size,
input/cache dtypes, physical cache layout, and scale granularity are static
kernel identity because each can select a different implementation branch.
"""

from __future__ import annotations

from profiling.db.args import DType, KvCacheAppendArgs
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "kv_cache_append"
_RUNNER_MODULE = "profiling.runners.attention.kv_cache_append"


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        # First-pass registry support is the BF16 alignment path. Keeping this
        # narrow avoids claiming unsupported cross-products on BackendSupport's
        # independent compute/KV axes (e.g. bf16 input + fp16 cache).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name=_RUNNER_MODULE,
            function_name="profile_kv_cache_append_torch",
        ),
        table_name=KIND,
        args_schema=KvCacheAppendArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16, DType.FP8_E4M3}),
        ),
        runner_ref=RunnerRef(
            module_name=_RUNNER_MODULE,
            function_name="profile_kv_cache_append_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=KvCacheAppendArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
