"""Profile the alignment vLLM fork's fused sparse-indexer prepare kernel.

Calls ``qsa_pre_indexer`` (``models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py``)
as ``QSAIndexer.forward`` does for text positions: Q and raw K are disjoint
column views of one projection output, the cos/sin table is the attention's
(``head_dim / 2`` wide), and the K work list is the one the metadata builder
writes (one item per completed group, at least one per request, ``-1`` padded
to ``(tokens + (ratio - 1)·requests) // ratio``).
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "sparse_indexer_prepare:vllm_triton"
_PAGE_SIZE = 512
_EPS = 1e-6


def _reference_q(torch: Any, q: Any, weight: Any, cos_sin: Any, positions: Any) -> Any:
    """Gemma RMSNorm then NeoX RoPE on the first half of every head."""
    x = q.float()
    dim = x.shape[-1]
    half, quarter = dim // 2, dim // 4
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + _EPS) * (weight.float() + 1.0)
    y = y.to(cos_sin.dtype).float()
    cos = cos_sin[positions, :quarter].float()[:, None, :]
    sin = cos_sin[positions, quarter:half].float()[:, None, :]
    r0, r1 = y[..., :quarter], y[..., quarter:half]
    return torch.cat((r0 * cos - r1 * sin, r1 * cos + r0 * sin, y[..., half:]), dim=-1)


def profile_sparse_indexer_prepare_vllm_triton(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    compress_ratio: int,
    ring_size: int,
    input_dtype: Any,
    cache_dtype: Any,
) -> ComputeMetrics:
    for name, value in (
        ("num_tokens", num_tokens),
        ("num_heads", num_heads),
        ("head_dim", head_dim),
        ("compress_ratio", compress_ratio),
        ("ring_size", ring_size),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    if ring_size < compress_ratio:
        raise ValueError("ring_size must hold at least one group")
    if head_dim % 4 or head_dim & (head_dim - 1):
        raise ProfilerNotImplemented(f"{_BACKEND} needs a power-of-two head_dim")
    input_dtype, cache_dtype = DType.from_value(input_dtype), DType.from_value(cache_dtype)
    if input_dtype is not DType.BF16 or cache_dtype not in (DType.BF16, DType.FP8_E4M3):
        raise ProfilerNotImplemented(f"{_BACKEND} needs bf16 inputs and a bf16 or fp8 cache")
    try:
        import torch
        from vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer import qsa_pre_indexer
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the alignment vLLM fork") from exc

    try:
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(0)
        cache_torch_dtype = torch.float8_e4m3fn if cache_dtype is DType.FP8_E4M3 else torch.bfloat16

        def randn(*shape: int) -> Any:
            return torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)

        projected = randn(num_tokens, (num_heads + 1) * head_dim)
        q_in, k_in = projected.split((num_heads * head_dim, head_dim), dim=-1)
        positions = torch.arange(num_tokens, device=device)
        cos_sin = randn(num_tokens + 1, head_dim // 2)
        q_weight, k_weight = 0.1 * randn(head_dim), 0.1 * randn(head_dim)
        q_out = torch.empty(
            (num_tokens, num_heads, head_dim), dtype=cache_torch_dtype, device=device
        )
        # One sequence starting at position 0: one ring block, compressed pages
        # scattered at random.
        state = torch.zeros((2, ring_size, 1, head_dim), dtype=torch.bfloat16, device=device)
        state_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
        state_slots = positions % ring_size
        query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
        groups = num_tokens // compress_ratio
        num_pages = max(1, -(-groups // _PAGE_SIZE))
        pool = num_pages + 64
        compressed = torch.zeros(
            (pool, _PAGE_SIZE, 1, head_dim), dtype=cache_torch_dtype, device=device
        )
        pages = torch.randperm(pool, device=device, generator=generator)[:num_pages]
        group_of = positions // compress_ratio
        slots = pages[group_of // _PAGE_SIZE] * _PAGE_SIZE + group_of % _PAGE_SIZE
        compressed_slots = torch.where((positions + 1) % compress_ratio == 0, slots, -1)
        max_work = (num_tokens + compress_ratio - 1) // compress_ratio
        work = torch.full((max_work, 2), -1, dtype=torch.int32, device=device)
        active = max(groups, 1)
        work[:active, 0] = 0
        work[:active, 1] = torch.arange(active, dtype=torch.int32, device=device)

        def launch() -> None:
            qsa_pre_indexer(
                q_in,
                k_in,
                positions,
                cos_sin,
                q_weight,
                k_weight,
                _EPS,
                q_out,
                state,
                state_slots,
                state_table,
                query_start_loc,
                positions,
                compressed,
                compressed_slots,
                work,
                compress_ratio=compress_ratio,
                mrope_section=None,
                rope_pos_offset=None,
            )

        launch()
        torch.cuda.synchronize()
        rows = sorted({0, num_tokens // 2, num_tokens - 1})
        expected = _reference_q(
            torch,
            q_in[rows].view(len(rows), num_heads, head_dim),
            q_weight,
            cos_sin,
            positions[rows],
        )
        torch.testing.assert_close(q_out[rows].float(), expected, rtol=8e-2, atol=3e-2)
        if groups:
            first = k_in[:compress_ratio].float().mean(0).to(torch.bfloat16)
            want = _reference_q(torch, first.view(1, 1, head_dim), k_weight, cos_sin, positions[:1])
            got = compressed[pages[0], 0, 0].float()
            torch.testing.assert_close(got, want.view(-1), rtol=8e-2, atol=3e-2)

        time_ms = Timer.cupti(launch, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    out_bytes = 1 if cache_torch_dtype == torch.float8_e4m3fn else 2
    traffic = num_tokens * (num_heads * head_dim * (2 + out_bytes) + head_dim * 2)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(traffic / (time_ms / 1000.0) / 1e9),
        energy_j=float(energy_j),
    )


__all__ = ["profile_sparse_indexer_prepare_vllm_triton"]
