"""Profile the vLLM fork's pooled index-key decode/verify cache update.

The production callable is
``vllm.models.glm5next.nvidia.ops.kpool_compress.kpool_decode_update_and_maybe_write_cache_batched``:
one Triton program per request walks that request's ``next_n`` tokens, stashes
each in the request's tail ring and compresses every pool a token completes
into the page-planar FP8 index cache. Operands follow the fork's decode
metadata: ``slot_mapping`` holds the pooled-cache slot only on a pool's last
token (else -1), ``tail_slot_mapping`` is ``tail_block * ring + pos % ring``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "dsa_pooled_index_cache_update:vllm_fork_triton"
# The kernel's Hadamard rotation is hard-wired to 128 elements (_hadamard128).
_INDEX_DIM = 128
# First position of every request before its pool phase; any value works, it
# only has to leave room for the pool phases.
_BASE_POSITION = 1 << 14
# Extra tail-ring blocks and cache pages no request maps, so a stray write lands
# in memory the check can see.
_SPARE_BLOCKS = 4


@dataclass(frozen=True)
class _Shape:
    num_requests: int
    next_n: int
    pool_size: int
    tail_ring: int
    index_dim: int
    block_size: int


def validate_args(
    num_requests: int,
    next_n: int,
    pool_size: int,
    tail_ring: int,
    index_dim: int,
    block_size: int,
    input_dtype: DType | str,
    cache_dtype: DType | str,
) -> _Shape:
    shape = _Shape(
        int(num_requests),
        int(next_n),
        int(pool_size),
        int(tail_ring),
        int(index_dim),
        int(block_size),
    )
    if shape.num_requests < 1:
        raise ValueError(f"num_requests must be >= 1, got {shape.num_requests}")
    if shape.next_n < 1:
        raise ValueError(f"next_n must be >= 1, got {shape.next_n}")
    if shape.pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {shape.pool_size}")
    if shape.tail_ring < shape.pool_size or shape.tail_ring % shape.pool_size:
        raise ValueError(
            "tail_ring must be a multiple of pool_size, "
            f"got tail_ring={shape.tail_ring}, pool_size={shape.pool_size}"
        )
    if shape.block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {shape.block_size}")
    if shape.index_dim != _INDEX_DIM:
        raise ProfilerNotImplemented(
            f"{_BACKEND} rotates exactly {_INDEX_DIM} elements, got index_dim={shape.index_dim}"
        )
    if DType.from_value(input_dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires input_dtype=bf16, got {input_dtype}")
    if DType.from_value(cache_dtype) is not DType.FP8_E4M3:
        raise ProfilerNotImplemented(f"{_BACKEND} requires cache_dtype=fp8_e4m3, got {cache_dtype}")
    return shape


def request_positions(shape: _Shape) -> list[list[int]]:
    """Positions of each request's tokens: request r starts at phase P - 1 - (r mod P)."""
    p = shape.pool_size
    return [
        [_BASE_POSITION + (p - 1 - r % p) + t for t in range(shape.next_n)]
        for r in range(shape.num_requests)
    ]


def completed_pools(shape: _Shape) -> int:
    p = shape.pool_size
    return sum(pos % p == p - 1 for row in request_positions(shape) for pos in row)


def logical_bytes(shape: _Shape) -> int:
    rows = shape.num_requests * shape.next_n
    pools = completed_pools(shape)
    d = shape.index_dim
    return (
        rows * 4 * d  # key + score in (bf16)
        + rows * 4 * d  # key + score into the tail ring (bf16)
        + pools * (shape.pool_size - 1) * 4 * d  # ring reads of the pool's earlier tokens
        + pools * (d + 4)  # fp8 pooled key + scale
        + rows * 12  # three int32 index tensors
    )


def logical_flops(shape: _Shape) -> int:
    pools = completed_pools(shape)
    d = shape.index_dim
    # softmax (exp, add, max, div) and weighted sum per slot, then a log2(d)-stage
    # butterfly and the quantization.
    return pools * (shape.pool_size * d * 6 + d * int(math.log2(d)) + d * 4)


@dataclass(frozen=True)
class _Operands:
    kv_cache: Any
    tail: Any
    tail_initial: Any
    tail_slot_mapping: Any
    key: Any
    slot_score: Any
    ape: Any
    slot_mapping: Any
    positions: Any


def build_operands(torch: Any, shape: _Shape, device: Any) -> _Operands:
    g = torch.Generator(device=device).manual_seed(0)
    r_count, n, p, ring, d, block = (
        shape.num_requests,
        shape.next_n,
        shape.pool_size,
        shape.tail_ring,
        shape.index_dim,
        shape.block_size,
    )
    positions = torch.tensor(request_positions(shape), dtype=torch.int32, device=device)
    # Each request owns a scattered run of cache pages and starts at a seeded
    # in-page offset; a request's pools are consecutive pool slots.
    pools_per_request = n // p + 2
    pages_per_request = (pools_per_request + block - 1) // block + 1
    num_pages = r_count * pages_per_request + _SPARE_BLOCKS
    page_ids = torch.randperm(num_pages, generator=g, device=device)[: r_count * pages_per_request]
    page_ids = page_ids.view(r_count, pages_per_request)
    start = torch.randint(0, block, (r_count, 1), generator=g, device=device)
    local = start + (positions.long() // p - positions[:, :1].long() // p)
    pool_slot = page_ids.gather(1, local // block) * block + local % block
    is_last = positions % p == p - 1
    slot_mapping = torch.where(is_last, pool_slot, torch.full_like(pool_slot, -1)).to(torch.int32)

    tail_blocks = torch.randperm(r_count + _SPARE_BLOCKS, generator=g, device=device)[:r_count]
    tail_slot_mapping = (tail_blocks[:, None] * ring + positions.long() % ring).to(torch.int32)

    def rn(*size: int, scale: float, dtype: Any) -> Any:
        return (torch.randn(*size, generator=g, device=device) * scale).to(dtype)

    tail = rn(r_count + _SPARE_BLOCKS, 2, ring, d, scale=0.5, dtype=torch.bfloat16)
    return _Operands(
        kv_cache=torch.zeros(num_pages, block, d + 4, dtype=torch.uint8, device=device),
        tail=tail,
        tail_initial=tail.clone(),
        tail_slot_mapping=tail_slot_mapping.contiguous(),
        key=rn(r_count, n, d, scale=0.5, dtype=torch.bfloat16),
        slot_score=rn(r_count, n, d, scale=1.0, dtype=torch.bfloat16),
        ape=rn(p, d, scale=0.5, dtype=torch.float32),
        slot_mapping=slot_mapping.contiguous(),
        positions=positions.contiguous(),
    )


def _launch(op: Any, o: _Operands, shape: _Shape) -> None:
    op(
        o.kv_cache,
        o.tail,
        o.tail_slot_mapping,
        o.key,
        o.slot_score,
        o.ape,
        o.slot_mapping,
        o.positions,
        shape.pool_size,
        shape.index_dim,
        round_scale=True,
    )


def _decode_cache(torch: Any, cache: Any, shape: _Shape, mx_scale: bool):
    """Split page-planar pages into fp32 values [pages, block, dim] and scales [pages, block]."""
    pages = cache.reshape(cache.shape[0], -1)
    plane = shape.block_size * shape.index_dim
    key_bytes = pages[:, :plane].reshape(-1, shape.block_size, shape.index_dim)
    scale_bytes = pages[:, plane:].reshape(-1, shape.block_size, 4)
    values = key_bytes.contiguous().view(torch.float8_e4m3fn).float()
    if mx_scale:
        exponents = scale_bytes.to(torch.int32)
        scales = torch.exp2(exponents[..., 0].float() - 127.0)
        equal = (exponents == exponents[..., :1]).all(dim=-1)
    else:
        scales = scale_bytes.contiguous().view(torch.float32)[..., 0]
        equal = torch.ones_like(scales, dtype=torch.bool)
    written = (key_bytes != 0).any(dim=-1) | (scale_bytes != 0).any(dim=-1)
    return values, scales, equal, written


def check_output(torch: Any, op: Any, o: _Operands, shape: _Shape, mx_scale: bool) -> None:
    from profiling.runners.attention.dsa_pooled_index_cache_update_reference import (
        dsa_pooled_index_cache_update_reference,
    )

    _launch(op, o, shape)
    torch.cuda.synchronize(o.kv_cache.device)
    ref_tail = o.tail_initial.float().cpu()
    slots, ref_values, ref_scales = dsa_pooled_index_cache_update_reference(
        ref_tail,
        o.tail_slot_mapping.cpu(),
        o.key.float().cpu(),
        o.slot_score.float().cpu(),
        o.ape.cpu(),
        o.slot_mapping.cpu(),
        o.positions.cpu(),
        shape.pool_size,
    )
    if not torch.equal(o.tail.float().cpu(), ref_tail):
        raise KernelLaunchFailed(f"{_BACKEND} tail ring differs from the reference")
    if slots.numel() != completed_pools(shape) or slots.unique().numel() != slots.numel():
        raise KernelLaunchFailed(f"{_BACKEND} operand pool count mismatch")
    values, scales, equal, written = _decode_cache(torch, o.kv_cache.cpu(), shape, mx_scale)
    page, offset = slots // shape.block_size, slots % shape.block_size
    if not bool(equal[page, offset].all()):
        raise KernelLaunchFailed(f"{_BACKEND} wrote unequal UE8M0 scale bytes")
    actual = values[page, offset] * scales[page, offset, None]
    reference = ref_values * ref_scales[:, None]
    # FP8 E4M3 keeps 3 mantissa bits; summation order may move a value one
    # step, and a scale tie one power of two.
    error = (actual - reference).abs().amax(dim=-1)
    if bool((error > 0.1 * reference.abs().amax(dim=-1)).any()):
        raise KernelLaunchFailed(f"{_BACKEND} pooled keys differ from the reference")
    cosine = torch.nn.functional.cosine_similarity(actual, reference, dim=-1)
    if float(cosine.min()) < 0.995:
        raise KernelLaunchFailed(f"{_BACKEND} pooled keys: min cosine {float(cosine.min())}")
    mapped = torch.zeros_like(written)
    mapped[page, offset] = True
    if bool((written & ~mapped).any()):
        raise KernelLaunchFailed(f"{_BACKEND} wrote cache rows no pool maps to")


def profile_dsa_pooled_index_cache_update_vllm_fork_triton(
    num_requests: int,
    next_n: int,
    pool_size: int,
    tail_ring: int,
    index_dim: int,
    block_size: int,
    input_dtype: DType | str,
    cache_dtype: DType | str,
) -> ComputeMetrics:
    shape = validate_args(
        num_requests,
        next_n,
        pool_size,
        tail_ring,
        index_dim,
        block_size,
        input_dtype,
        cache_dtype,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires PyTorch") from exc
    try:
        from vllm.models.glm5next.nvidia.ops.kpool_compress import (
            kpool_decode_update_and_maybe_write_cache_batched as op,
        )
        from vllm.utils.deep_gemm import mqa_logits_use_mx_scales
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the vLLM fork's kpool_compress") from exc

    try:
        device = torch.device("cuda", torch.cuda.current_device())
        operands = build_operands(torch, shape, device)
        check_output(torch, op, operands, shape, bool(mqa_logits_use_mx_scales()))

        def run() -> None:
            _launch(op, operands, shape)

        time_ms = Timer.cupti(run, kernel_name=None)
        energy_j = Energy.perf(run, per_iter_time_ms=time_ms)
        elapsed_s = time_ms / 1000.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(logical_flops(shape) / elapsed_s / 1e12),
            memory_bandwidth_gbps=float(logical_bytes(shape) / elapsed_s / 1e9),
            energy_j=float(energy_j),
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, TypeError, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc


__all__ = ["profile_dsa_pooled_index_cache_update_vllm_fork_triton"]
