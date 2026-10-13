"""Profile the alignment vLLM fork's split-K sparse GQA paged attention.

Calls ``qsa_sparse_paged_attention`` (``models/qwen4_exp/nvidia/ops/qsa.py``)
the way ``Qwen4ExpQSAFlashAttentionImpl.forward_qsa`` does: K and V are the two
halves of one ``[pages, kv_heads, page, 2·dim]`` cache viewed as
``[pages, page, kv_heads, dim]``, FP8 caches are uint8 storage viewed as e4m3
with host scales, and the packed indices come from the fork's own expansion
kernel.
"""

from __future__ import annotations

import math
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.sparse_selection import (
    random_group_selection,
    selected_tokens,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "sparse_gqa_paged_attention:vllm_triton"
# Page size is not part of the key: it only changes which page-table entry a
# token's address comes from. Pages are drawn at random from a pool far larger
# than L2 so every request reads cold, scattered pages.
#
# Request structure follows the launch table. A uniform-table call is a decode
# or verify step: each row is its own request with its own scattered pages. A
# prefill-table call carries a prefill chunk, whose rows all read one request's
# KV through one block table of consecutive pages, so rows reuse each other's
# pages in L2. Probe logs/20261013_8_qf_probe (B200, fp8, 24/2/256): rows as
# separate requests ran 7-9% slower than the same rows sharing one request, and
# the shared form matched the captured mixed steps to 0-5%.
_PAGE_SIZE = 1024
_POOL_BYTES = 4 * 1024**3
_CHECK_ROWS = 3


def _validate(
    num_rows: int,
    context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    token_topk: int,
    compress_ratio: int,
    q_dtype: Any,
    kv_dtype: Any,
) -> tuple[DType, DType]:
    for name, value in (
        ("num_rows", num_rows),
        ("context_len", context_len),
        ("num_q_heads", num_q_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("token_topk", token_topk),
        ("compress_ratio", compress_ratio),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be a multiple of num_kv_heads")
    if token_topk % compress_ratio:
        raise ValueError("token_topk must be a multiple of compress_ratio")
    if head_dim < 16 or head_dim & (head_dim - 1):
        raise ProfilerNotImplemented(f"{_BACKEND} needs a power-of-two head_dim >= 16")
    q_dtype, kv_dtype = DType.from_value(q_dtype), DType.from_value(kv_dtype)
    if q_dtype is not DType.BF16 or kv_dtype not in (DType.BF16, DType.FP8_E4M3):
        raise ProfilerNotImplemented(f"{_BACKEND} needs bf16 queries and a bf16 or fp8 cache")
    return q_dtype, kv_dtype


def profile_sparse_gqa_paged_attention_vllm_triton(
    num_rows: int,
    context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    token_topk: int,
    compress_ratio: int,
    q_dtype: Any,
    kv_dtype: Any,
    prefill_config: bool,
) -> ComputeMetrics:
    _q_dtype, kv_dtype = _validate(
        num_rows,
        context_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        token_topk,
        compress_ratio,
        q_dtype,
        kv_dtype,
    )
    try:
        import torch
        from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
        from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import expand_qsa_block_indices
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the alignment vLLM fork") from exc

    from profiling.runners.attention.sparse_gqa_paged_attention_reference import (
        sparse_gqa_paged_attention_reference,
    )

    is_fp8 = kv_dtype is DType.FP8_E4M3
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(0)
        kv_bytes = 1 if is_fp8 else 2
        page_bytes = num_kv_heads * _PAGE_SIZE * 2 * head_dim * kv_bytes
        pages_per_row = math.ceil(context_len / _PAGE_SIZE)
        pool = max(_POOL_BYTES // page_bytes, pages_per_row)
        if is_fp8:
            # e4m3 bytes with magnitude below 2 and a random sign.
            shape = (pool, num_kv_heads, _PAGE_SIZE, 2 * head_dim)
            cache = torch.randint(
                0, 0x40, shape, dtype=torch.uint8, device=device, generator=generator
            )
            cache |= (
                torch.randint(0, 2, shape, dtype=torch.uint8, device=device, generator=generator)
                << 7
            )
        else:
            cache = torch.randn(
                (pool, num_kv_heads, _PAGE_SIZE, 2 * head_dim),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
        key_cache, value_cache = cache.transpose(1, 2).split(head_dim, dim=-1)
        if is_fp8:
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
            scales = {"k_scale": 1.0, "v_scale": 1.0}
        else:
            scales = {}
        if prefill_config:
            # One request: consecutive pages from a random base, every row on it.
            base = int(
                torch.randint(0, pool - pages_per_row + 1, (1,), device=device, generator=generator)
            )
            block_table = torch.arange(
                base, base + pages_per_row, dtype=torch.int32, device=device
            ).unsqueeze(0)
            token_to_req = torch.zeros(num_rows, dtype=torch.int32, device=device)
        else:
            # Each row is its own request with its own scattered pages.
            block_table = torch.randint(
                0,
                pool,
                (num_rows, pages_per_row),
                dtype=torch.int32,
                device=device,
                generator=generator,
            )
            token_to_req = torch.arange(num_rows, dtype=torch.int32, device=device)
        blocks, positions, visible = random_group_selection(
            torch,
            num_rows=num_rows,
            context_len=context_len,
            token_topk=token_topk,
            compress_ratio=compress_ratio,
            device=device,
        )
        width = token_topk + compress_ratio - 1
        indices = torch.empty((num_rows, width + 1), dtype=torch.int32, device=device)
        expand_qsa_block_indices(blocks, positions, visible, compress_ratio, token_topk, indices)
        del blocks
        q = torch.randn(
            (num_rows, num_q_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        gate = torch.randn(
            (num_rows, num_q_heads * head_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        out = torch.empty_like(q)

        def launch() -> Any:
            return qsa_sparse_paged_attention(
                q,
                key_cache,
                value_cache,
                indices,
                block_table,
                token_to_req,
                bool(prefill_config),
                out,
                output_gate=gate,
                **scales,
            )

        launch()
        torch.cuda.synchronize()
        check_rows = sorted({0, num_rows // 2, num_rows - 1})[:_CHECK_ROWS]
        expected = sparse_gqa_paged_attention_reference(
            torch, q, key_cache, value_cache, indices, block_table, token_to_req, gate, check_rows
        )
        torch.testing.assert_close(out[check_rows], expected, rtol=3e-2, atol=3e-2)

        time_ms = Timer.cupti(launch, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    attended = selected_tokens(context_len, token_topk, compress_ratio)
    seconds = time_ms / 1000.0
    flops = 4 * num_rows * num_q_heads * head_dim * attended
    kv_traffic = num_rows * attended * num_kv_heads * 2 * head_dim * (1 if is_fp8 else 2)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / seconds / 1e12),
        memory_bandwidth_gbps=float(kv_traffic / seconds / 1e9),
        energy_j=float(energy_j),
    )


__all__ = ["profile_sparse_gqa_paged_attention_vllm_triton"]
