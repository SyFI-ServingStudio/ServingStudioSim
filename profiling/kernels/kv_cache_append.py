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
from profiling.db.doc import BackendDoc, KernelDoc
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

DOC = KernelDoc(
    title="Paged KV cache append",
    summary=("Write the keys and values of new tokens into their slots in a paged KV cache."),
    description=(
        "After the K and V projections, each new token's key and value are "
        "stored in the paged cache for later attention. A slot mapping names "
        "one cache slot per token, and block_size is the tokens per page. The "
        "measurement writes num_tokens tokens into distinct random slots of a "
        "cache with at least 256 pages."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "key_cache[slot[t]] ← key[t], value_cache[slot[t]] ← value[t], for each new token t",
        "GB/s = [2 · num_tokens · num_kv_heads · head_dim · (bytes(input_dtype)"
        " + bytes(kv_dtype)) + 8 · num_tokens] / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        "vllm_cuda: GPU kernel time from CUPTI, averaged over repeated launches"
        " of reshape_and_cache_flash_kernel after five warm-ups, with the L2 "
        "cache flushed before each launch. torch: CUDA-event time around a loop"
        " of calls after five warm-ups, the median of three loops; the cache is"
        " not flushed, and the time includes the gap between its two launches."
    ),
    caveats=(
        "The two backends are timed differently, cold kernel time against a "
        "warm loop, so their numbers are not directly comparable.",
        "The torch backend only covers matching input and cache dtypes with one scale per tensor.",
        "vllm_cuda uses K and V scales of 1.",
        "GB/s counts the K and V reads and writes and the int64 slot map, not the scales.",
    ),
    # The torch backend is the PyTorch reference; it lives in the runner module.
    reference=None,
)


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
        doc=BackendDoc(summary="PyTorch indexed assignment writes K and V in two launches."),
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
        doc=BackendDoc(
            summary=(
                "vLLM's reshape_and_cache_flash CUDA kernel writes both K and V to mapped slots."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/cache_kernels.cu",
        ),
    )
)
