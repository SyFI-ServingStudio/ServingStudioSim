"""Profile the alignment vLLM fork's sparse-indexer scoring kernels.

The public selection calls (``qsa_select_paged_decode`` / ``_prefill`` in
``models/qwen4_exp/nvidia/ops/qsa_indexer.py``) score and then run a top-k.
Only the scoring launches are counted here, by kernel name; the top-k is the
``dsa_persistent_topk_decode`` kind. Compressed keys live in a
``[pages, page_size, 1, head_dim]`` cache whose pages are scattered at random.
"""

from __future__ import annotations

import math
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

# Page size stays out of the key (it only picks which table entry a column's
# address comes from); 512 divides every profiled width.
_PAGE_SIZE = 512
_POOL_BYTES = 1024**3
# The fork's top-k (k = token_topk / ratio) is a fixed part of the call; 2048 / 4.
_TOKEN_TOPK = 2048
_DECODE_KERNEL = "_qsa_mqa_paged_uniform_kernel"
_PREFILL_KERNEL = "_qsa_mqa_paged_prefill_kernel"


def _torch_dtype(torch: Any, dtype: Any, label: str) -> Any:
    dtype = DType.from_value(dtype)
    if dtype is DType.FP8_E4M3:
        return torch.float8_e4m3fn
    if dtype is DType.BF16:
        return torch.bfloat16
    raise ProfilerNotImplemented(f"{label} supports bf16 and fp8_e4m3, got {dtype}")


def _positive(**values: int) -> None:
    for name, value in values.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")


def _imports(label: str) -> tuple[Any, Any]:
    try:
        import torch
        import vllm._custom_ops  # noqa: F401  # registers torch.ops._C top-k
        from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{label} requires the alignment vLLM fork") from exc
    return torch, qsa_indexer


def _keys_cache(torch: Any, dt: Any, head_dim: int, min_pages: int, device: Any, gen: Any) -> Any:
    page_bytes = _PAGE_SIZE * head_dim * (1 if dt == torch.float8_e4m3fn else 2)
    pool = max(_POOL_BYTES // page_bytes, min_pages)
    values = torch.randn((pool, _PAGE_SIZE, 1, head_dim), device=device, generator=gen) / 4
    return values.to(dt)


def _check(torch: Any, actual: Any, expected: Any) -> None:
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=5e-2)


def profile_sparse_indexer_logits_decode_vllm_triton(
    batch_size: int,
    num_keys: int,
    logits_width: int,
    next_n: int,
    num_heads: int,
    head_dim: int,
    dtype: Any,
) -> ComputeMetrics:
    label = "sparse_indexer_logits_decode:vllm_triton"
    _positive(
        batch_size=batch_size,
        num_keys=num_keys,
        logits_width=logits_width,
        next_n=next_n,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    if num_keys > logits_width:
        raise ValueError("num_keys must be <= logits_width")
    torch, ops = _imports(label)
    from profiling.runners.attention.sparse_indexer_logits_reference import (
        sparse_indexer_logits_reference,
    )

    dt = _torch_dtype(torch, dtype, label)
    compress_ratio = 4
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        gen = torch.Generator(device=device).manual_seed(0)
        table_width = math.ceil(logits_width / _PAGE_SIZE)
        k_cache = _keys_cache(torch, dt, head_dim, table_width, device, gen)
        rows = batch_size * next_n
        q = (torch.randn((rows, num_heads, head_dim), device=device, generator=gen) / 4).to(dt)
        page_table = torch.randint(
            0,
            k_cache.shape[0],
            (batch_size, table_width),
            dtype=torch.int32,
            device=device,
            generator=gen,
        )
        visible = torch.full((rows,), num_keys, dtype=torch.int32, device=device)
        blocks = torch.empty(
            (rows, _TOKEN_TOPK // compress_ratio), dtype=torch.int32, device=device
        )

        def launch() -> None:
            ops.qsa_select_paged_decode(
                q, k_cache, page_table, visible, _TOKEN_TOPK, compress_ratio, next_n, blocks
            )

        launch()
        torch.cuda.synchronize()
        # Recompute the last request's scores and check its top-k picks are its best.
        expected = sparse_indexer_logits_reference(
            torch, q[-next_n:], k_cache, page_table[-1], num_keys
        )
        picked = blocks[-next_n:].long()
        k = min(num_keys, blocks.shape[1])
        best = expected.topk(k, dim=1).values.sort(dim=1).values
        got = expected.gather(1, picked[:, :k].clamp_min(0)).sort(dim=1).values
        _check(torch, got, best)

        time_ms = Timer.cupti(launch, warmup=3, kernel_name=_DECODE_KERNEL)
        energy_j = Energy.perf(launch, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{label} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{label} failed: {exc}") from exc

    seconds = time_ms / 1000.0
    elem = 1 if dt == torch.float8_e4m3fn else 2
    flops = 2 * rows * num_heads * head_dim * num_keys
    traffic = batch_size * num_keys * head_dim * elem + 4 * rows * num_keys
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / seconds / 1e12),
        memory_bandwidth_gbps=float(traffic / seconds / 1e9),
        energy_j=float(energy_j),
    )


def profile_sparse_indexer_logits_prefill_vllm_triton(
    num_queries: int,
    prefix_len: int,
    num_heads: int,
    head_dim: int,
    compress_ratio: int,
    dtype: Any,
) -> ComputeMetrics:
    label = "sparse_indexer_logits_prefill:vllm_triton"
    _positive(
        num_queries=num_queries,
        num_heads=num_heads,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
    )
    if type(prefix_len) is not int or prefix_len < 0:
        raise ValueError(f"prefix_len must be a nonnegative int, got {prefix_len!r}")
    if _TOKEN_TOPK % compress_ratio:
        raise ProfilerNotImplemented(f"{label} needs compress_ratio dividing {_TOKEN_TOPK}")
    torch, ops = _imports(label)
    from profiling.runners.attention.sparse_indexer_logits_reference import (
        sparse_indexer_logits_reference,
    )

    dt = _torch_dtype(torch, dtype, label)
    seq_len = prefix_len + num_queries
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        gen = torch.Generator(device=device).manual_seed(0)
        logits_width = math.ceil(math.ceil(seq_len / compress_ratio) / 64) * 64
        table_width = math.ceil(max(logits_width, 64) / _PAGE_SIZE)
        k_cache = _keys_cache(torch, dt, head_dim, table_width, device, gen)
        q = (torch.randn((num_queries, num_heads, head_dim), device=device, generator=gen) / 4).to(
            dt
        )
        page_table = torch.randint(
            0, k_cache.shape[0], (1, table_width), dtype=torch.int32, device=device, generator=gen
        )
        query_start_loc = torch.tensor([0, num_queries], dtype=torch.int32, device=device)
        visible = (torch.arange(prefix_len + 1, seq_len + 1, device=device) // compress_ratio).to(
            torch.int32
        )
        blocks = torch.empty(
            (num_queries, _TOKEN_TOPK // compress_ratio), dtype=torch.int32, device=device
        )

        def launch() -> None:
            ops.qsa_select_paged_prefill(
                q,
                k_cache,
                page_table,
                query_start_loc,
                visible,
                _TOKEN_TOPK,
                compress_ratio,
                num_queries,
                blocks,
                seq_len,
            )

        launch()
        torch.cuda.synchronize()
        last_keys = int(visible[-1])
        if last_keys:
            expected = sparse_indexer_logits_reference(
                torch, q[-1:], k_cache, page_table[0], last_keys
            )
            k = min(last_keys, blocks.shape[1])
            best = expected.topk(k, dim=1).values.sort(dim=1).values
            got = expected.gather(1, blocks[-1:, :k].long().clamp_min(0)).sort(dim=1).values
            _check(torch, got, best)

        time_ms = Timer.cupti(launch, warmup=3, kernel_name=_PREFILL_KERNEL)
        energy_j = Energy.perf(launch, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{label} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{label} failed: {exc}") from exc

    scored = sum((prefix_len + i + 1) // compress_ratio for i in range(0, num_queries))
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(2 * num_heads * head_dim * scored / seconds / 1e12),
        memory_bandwidth_gbps=float(4 * scored / seconds / 1e9),
        energy_j=float(energy_j),
    )


__all__ = [
    "profile_sparse_indexer_logits_decode_vllm_triton",
    "profile_sparse_indexer_logits_prefill_vllm_triton",
]
