"""MI300X rope-free BF16 sparse-MLA prefill via the vLLM Triton ragged kernel.

The ROCm/MI300X backend of the ``dsa_sparse_mla_prefill`` kind -- a ragged batch
of GLM-5.3-Flash prefill requests, each query attending to up to ``selected_k``
causal cache positions. Same callable and dtype as the decode backend
(``rocm_sparse_attn_prefill`` at ``head_dim=512, nope=512, rope=0``, BF16); the
only difference is the query batch shape: the prefill batch has many query rows
per request with ramped causal valid counts, while a decode batch has one query
row per request. Both are AMD analogs of the B200 ``flashinfer_trtllm_fp8`` row,
but the Triton ragged kernel ``_sparse_attn_prefill_ragged_kernel`` on a BF16
latent, not FlashInfer fp8.

The per-query valid counts are derived exactly as the B200 prefill runner derives
them -- ``min(position + 1, selected_k)`` over each request's last
``query`` positions -- so the MI300X and B200 rows describe the same ragged
workload and differ only by ``(gpu_name, backend, rope_dim, dtype)``. Timing,
correctness, and the Triton-path assertion reuse the shared sparse-MLA machinery.
"""

from __future__ import annotations

from typing import Any

import profiling.runners.attention._rocm_triton_mla_sparse_common as common
from profiling.db.args import DType
from profiling.runners.attention.dsa_sparse_mla_attention import _INDEX_DISTRIBUTIONS
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_KIND = "dsa_sparse_mla_prefill"
_BACKEND = "rocm_triton_mla_sparse"
_FULL = f"{_KIND}:{_BACKEND}"
_SUPPORTED_NUM_HEADS = frozenset({8, 16})
# The ragged attention kernel ``_sparse_attn_prefill_ragged_kernel`` plus the one
# ``__amd_rocclr_copyBuffer`` the path issues per launch. Pinned from the first
# real MI300X rocpd capture (lease 448076, gfx942); same callable as the decode
# backend, so the same steady-state count.
_DISPATCHES_PER_LAUNCH: int | None = 2


def _normalize_pairs(query_context_pairs: Any) -> tuple[tuple[int, int], ...]:
    # JSON round-trips tuples as lists; accept either and canonicalize.
    if not isinstance(query_context_pairs, (tuple, list)) or not query_context_pairs:
        raise ValueError("query_context_pairs must be a nonempty sequence")
    pairs: list[tuple[int, int]] = []
    for pair in query_context_pairs:
        pair = tuple(pair)
        if len(pair) != 2 or any(type(v) is not int for v in pair):
            raise TypeError("query_context_pairs must contain integer (query, context) pairs")
        num_queries, context_len = pair
        if num_queries <= 0 or context_len < num_queries:
            raise ValueError("each pair must satisfy 0 < query <= context")
        pairs.append((num_queries, context_len))
    return tuple(pairs)


def _valid_counts(pairs: tuple[tuple[int, int], ...], *, selected_k: int) -> tuple[int, ...]:
    counts: list[int] = []
    for num_queries, context_len in pairs:
        first = context_len - num_queries
        counts.extend(min(position + 1, selected_k) for position in range(first, context_len))
    return tuple(counts)


def _validate(
    *,
    query_context_pairs: Any,
    num_heads: int,
    num_kv_heads: int,
    selected_k: int,
    latent_dim: int,
    rope_dim: int,
    value_dim: int,
    softmax_scale: float,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    index_dtype: str,
    output_dtype: DType | str,
    index_distribution: str,
    cache_layout: str,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    pairs = _normalize_pairs(query_context_pairs)
    # selected_k is checked against the allowed page-table widths separately: the
    # arch composes the 2176 kpool page-table width while 2048 is the raw
    # index_topk, so both are valid coordinates (see common.ALLOWED_SELECTED_K).
    identity = (num_heads, num_kv_heads, latent_dim, rope_dim, value_dim)
    expected = (
        num_heads,
        common.NUM_KV_HEADS,
        common.NOPE_HEAD_DIM,
        common.ROPE_HEAD_DIM,
        common.VALUE_DIM,
    )
    if (
        num_heads not in _SUPPORTED_NUM_HEADS
        or selected_k not in common.ALLOWED_SELECTED_K
        or identity[1:] != expected[1:]
    ):
        raise ProfilerNotImplemented(
            f"{_FULL} requires model identity num_heads in "
            f"{sorted(_SUPPORTED_NUM_HEADS)}, selected_k in "
            f"{sorted(common.ALLOWED_SELECTED_K)}, and "
            f"(num_kv_heads, latent_dim, rope_dim, value_dim)={expected[1:]}, "
            f"got num_heads={num_heads}, selected_k={selected_k}, rest={identity[1:]}"
        )
    if type(softmax_scale) not in {int, float} or isinstance(softmax_scale, bool):
        raise TypeError("softmax_scale must be a real number")
    if float(softmax_scale) != common.SOFTMAX_SCALE:
        raise ProfilerNotImplemented(
            f"{_FULL} requires softmax_scale={common.SOFTMAX_SCALE}, got {softmax_scale}"
        )
    storage = (
        DType.from_value(q_dtype),
        DType.from_value(cache_dtype),
        index_dtype,
        DType.from_value(output_dtype),
        cache_layout,
    )
    expected_storage = (DType.BF16, DType.BF16, "int32", DType.BF16, common.CACHE_LAYOUT)
    if storage != expected_storage:
        raise ProfilerNotImplemented(
            f"{_FULL} requires storage identity {expected_storage}, got {storage}"
        )
    if index_distribution not in _INDEX_DISTRIBUTIONS:
        raise ProfilerNotImplemented(
            f"index_distribution must be one of: {', '.join(sorted(_INDEX_DISTRIBUTIONS))}"
        )
    return pairs, _valid_counts(pairs, selected_k=selected_k)


def _build_batch(torch: Any, valid_counts: tuple[int, ...], *, selected_k: int) -> common.RaggedBatch:
    num_queries = len(valid_counts)
    # Causal prefix: query i selects cache rows [0, count); the pool is as wide as
    # the largest causal window. Which rows are chosen does not change the timing;
    # it keeps every index valid and the reference exact.
    num_cache_tokens = max(selected_k, max(valid_counts) if valid_counts else 1)
    dense = torch.full((num_queries, selected_k), -1, dtype=torch.int32)
    lengths = torch.zeros(num_queries, dtype=torch.int32)
    for row, count in enumerate(valid_counts):
        if count:
            dense[row, :count].copy_(torch.arange(count, dtype=torch.int32))
        lengths[row] = count
    return common.RaggedBatch(
        dense_indices=dense,
        lengths=lengths,
        num_cache_tokens=num_cache_tokens,
        num_queries=num_queries,
        valid_counts=tuple(valid_counts),
    )


def build_dsa_sparse_mla_prefill_rocm_triton_kernel(
    *,
    query_context_pairs: Any,
    num_heads: int,
    num_kv_heads: int,
    selected_k: int,
    latent_dim: int,
    rope_dim: int,
    value_dim: int,
    softmax_scale: float,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    index_dtype: str,
    output_dtype: DType | str,
    index_distribution: str,
    cache_layout: str,
) -> dict[str, Any]:
    """Build the timed call + fixed operands for one prefill sparse-MLA batch."""
    _pairs, valid_counts = _validate(
        query_context_pairs=query_context_pairs,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        selected_k=selected_k,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        value_dim=value_dim,
        softmax_scale=softmax_scale,
        q_dtype=q_dtype,
        cache_dtype=cache_dtype,
        index_dtype=index_dtype,
        output_dtype=output_dtype,
        index_distribution=index_distribution,
        cache_layout=cache_layout,
    )
    import torch

    common.load_callables(torch)
    device = torch.device("cuda", torch.cuda.current_device())
    batch = _build_batch(torch, valid_counts, selected_k=selected_k)
    operands = common.build_operands(torch, batch, num_heads=num_heads, device=device)
    kernel = common.make_kernel(torch, operands)
    return {
        "torch": torch,
        "kernel": kernel,
        "operands": operands,
        "batch": batch,
        "num_heads": num_heads,
        "warmup": common.WARMUP,
        "rep": common.REP,
    }


def _spec(kwargs: dict[str, Any]) -> dict[str, Any]:
    spec = dict(kwargs)
    spec["query_context_pairs"] = [list(p) for p in _normalize_pairs(kwargs["query_context_pairs"])]
    spec["q_dtype"] = DType.from_value(kwargs["q_dtype"]).value
    spec["cache_dtype"] = DType.from_value(kwargs["cache_dtype"]).value
    spec["output_dtype"] = DType.from_value(kwargs["output_dtype"]).value
    return spec


def profile_dsa_sparse_mla_prefill_rocm_triton(
    *,
    query_context_pairs: Any,
    num_heads: int,
    num_kv_heads: int,
    selected_k: int,
    latent_dim: int,
    rope_dim: int,
    value_dim: int,
    softmax_scale: float,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    index_dtype: str,
    output_dtype: DType | str,
    index_distribution: str,
    cache_layout: str,
) -> ComputeMetrics:
    """Profile one GLM-5.3-Flash sparse-MLA prefill batch on an MI300X."""
    kwargs = dict(
        query_context_pairs=query_context_pairs,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        selected_k=selected_k,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        value_dim=value_dim,
        softmax_scale=softmax_scale,
        q_dtype=q_dtype,
        cache_dtype=cache_dtype,
        index_dtype=index_dtype,
        output_dtype=output_dtype,
        index_distribution=index_distribution,
        cache_layout=cache_layout,
    )
    _pairs, valid_counts = _validate(**kwargs)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_FULL}") from exc
    try:
        common.require_mi300(torch, backend=_FULL)
        from profiling.profilers.rocprof_kernel_profiler import measure_registered_via_rocprofv3

        built = build_dsa_sparse_mla_prefill_rocm_triton_kernel(**kwargs)
        common.check_correctness(
            torch, built["operands"], built["batch"], num_heads=num_heads, backend=_FULL
        )

        spec = _spec(kwargs)
        common.assert_triton_path(kind=_KIND, backend=_BACKEND, spec=spec)

        if _DISPATCHES_PER_LAUNCH is None:
            raise ProfilerNotImplemented(
                f"{_FULL}: per-launch dispatch count D not yet pinned; "
                "read it off the first MI300X rocpd capture before timing"
            )
        time_ms = measure_registered_via_rocprofv3(
            kind=_KIND,
            backend=_BACKEND,
            spec=spec,
            kernel_name_contains=None,
            warmup=common.WARMUP,
            rep=common.REP,
            dispatches_per_launch=_DISPATCHES_PER_LAUNCH,
        )
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{_FULL} ran out of device memory") from exc
        raise KernelLaunchFailed(f"{_FULL} failed: {exc}") from exc

    num_queries = len(valid_counts)
    elapsed = time_ms / 1000.0
    flops = common.logical_flops(num_queries=num_queries, num_heads=num_heads, selected_k=selected_k)
    byts = common.logical_bytes(
        num_queries=num_queries,
        num_heads=num_heads,
        selected_k=selected_k,
        valid_counts=valid_counts,
    )
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / elapsed / 1e12 if elapsed > 0 else 0.0,
        memory_bandwidth_gbps=byts / elapsed / 1e9 if elapsed > 0 else 0.0,
        energy_j=0.0,
    )


__all__ = [
    "build_dsa_sparse_mla_prefill_rocm_triton_kernel",
    "profile_dsa_sparse_mla_prefill_rocm_triton",
]
