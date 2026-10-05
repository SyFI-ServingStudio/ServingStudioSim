"""Profile the fused Q pad and MXFP8 sliding-window KV insert.

``vllm_cuda`` times one call of
``torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`` with the
flags the SM100 mega-attention layer passes (``attention.py:1035``): no Q norm,
no Q RoPE, chunk-interleaved Q, MXFP8 KV record. The op launches exactly one
kernel: ``...InsertKernelReducedGrid`` when ``num_tokens >= 1024`` and
``padded_heads > 0``, otherwise ``...InsertKernel``.

Inputs follow the producer contract: Q is the ``wq_b`` output in the
chunk-interleaved layout, KV rows are random BF16, positions are random
(decode tokens belong to different requests), and the insert slots are
distinct random slots of a poisoned 32-token-page cache. Outside the timed
region the result is checked against Torch: the padded Q must equal the
zero-padded input bit for bit, and each written record must decode to the
roped KV row within the FP8 tolerance, with untouched slots still poisoned.
"""

from __future__ import annotations

from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

KIND = "q_pad_kv_rope_mxfp8_insert"
_BACKEND = f"{KIND}:vllm_cuda"
_HEAD_DIM = 512
_ROPE_DIM = 64
_BLOCK_SIZE = 32
_RECORD_BYTES = 528
_DATA_BYTES = 512
_SCALE_BYTES = 16
_MAX_TOKENS = 65_536
_MAX_POSITION = 8192
_LIVE_HEADS = (8, 16, 32, 64, 128)
_POISON = 0xA5
# Randomized rows whose bf16 rounding may differ between the kernel's fused
# multiply-adds and Torch's RoPE can move one FP8 code or one scale step.
_MAX_MISMATCH_FRACTION = 1e-3
# MXFP8 (3-bit mantissa) round trip of a Gaussian row gives ~0.02 relative L2.
_REL_TOL = 0.05


def expected_padded_heads(num_heads: int) -> int:
    """The mega-attention layer's padded Q head count (FlashMLA: 64 or 128)."""
    return 64 if num_heads <= 64 else 128


def _validate_args(
    num_tokens: int,
    num_insert_tokens: int,
    num_heads: int,
    padded_heads: int,
    block_size: int,
    input_dtype: object,
    swa_cache_format: str,
) -> None:
    if type(num_tokens) is not int or not 1 <= num_tokens <= _MAX_TOKENS:
        raise ValueError(f"num_tokens must be an integer in [1, {_MAX_TOKENS}]")
    if type(num_insert_tokens) is not int or not 0 <= num_insert_tokens <= num_tokens:
        raise ValueError("num_insert_tokens must be an integer in [0, num_tokens]")
    if num_heads not in _LIVE_HEADS:
        raise ProfilerNotImplemented(f"{_BACKEND} supports live heads {_LIVE_HEADS}")
    padded = expected_padded_heads(num_heads)
    if padded_heads not in (padded, 0) or (padded_heads == 0 and num_heads != padded):
        raise ValueError(
            f"padded_heads must be {padded} for {num_heads} live heads, or 0 (KV-only) "
            "when the shard is already that wide"
        )
    if block_size != _BLOCK_SIZE:
        raise ProfilerNotImplemented(f"{_BACKEND} supports the {_BLOCK_SIZE}-token SWA page")
    if str(input_dtype) != "bf16":
        raise ProfilerNotImplemented(f"{_BACKEND} supports bf16 input")
    if swa_cache_format != "mxfp8":
        raise ProfilerNotImplemented(f"{_BACKEND} supports the mxfp8 (528 B) SWA record")


def reference_q(torch: Any, q_std: Any, padded_heads: int) -> Any:
    """Expected Q output: live heads unchanged, zero pad, chunk-interleaved."""
    from profiling.runners.attention.compressed_sparse_mla_rope_cast_reference import (
        q_to_fused_layout,
    )

    tokens, heads, dim = q_std.shape
    padded = torch.zeros((tokens, padded_heads, dim), dtype=q_std.dtype, device=q_std.device)
    padded[:, :heads] = q_std
    return q_to_fused_layout(padded)


def reference_kv_rows(torch: Any, kv: Any, positions: Any, cos_sin: Any) -> Any:
    """KV rows after GPT-J RoPE on the last 64 dims, rounded to bf16 as the kernel does."""
    rows = kv.float()
    table = cos_sin.float()[positions.long()]
    cos, sin = table[:, : _ROPE_DIM // 2], table[:, _ROPE_DIM // 2 :]
    even, odd = rows[:, -_ROPE_DIM::2], rows[:, -_ROPE_DIM + 1 :: 2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    rows = torch.cat((rows[:, :-_ROPE_DIM], rotated.flatten(-2)), dim=-1)
    return rows.to(torch.bfloat16)


def check_outputs(
    torch: Any,
    q_std: Any,
    q_out: Any,
    padded_heads: int,
    kv: Any,
    positions: Any,
    cos_sin: Any,
    cache: Any,
    slots: Any,
) -> dict:
    """Compare the op's Q output and written cache records against Torch."""
    from profiling.runners.attention.compressed_sparse_mla_rope_cast_reference import (
        decode_records,
        encode_records,
    )

    report: dict = {}
    if padded_heads:
        expected_q = reference_q(torch, q_std, padded_heads)
        report["q_exact"] = bool(torch.equal(q_out.view(torch.int16), expected_q.view(torch.int16)))
        if not report["q_exact"]:
            raise KernelLaunchFailed(f"{_BACKEND} padded Q differs from the zero-padded input")
    elif q_out.numel() != 0:
        raise KernelLaunchFailed(f"{_BACKEND} KV-only launch returned a non-empty Q")

    insert = slots.numel()
    if insert:
        rows = reference_kv_rows(torch, kv[:insert], positions[:insert], cos_sin)
        ref_data, ref_scale = encode_records(rows, "mxfp8")
        blocks = cache.shape[0]
        flat = cache.view(blocks, _BLOCK_SIZE * _RECORD_BYTES)
        data = flat[:, : _BLOCK_SIZE * _DATA_BYTES].reshape(-1, _DATA_BYTES)[slots]
        scale = flat[:, _BLOCK_SIZE * _DATA_BYTES :].reshape(-1, _SCALE_BYTES)[slots]
        scale_gap = int((scale.int() - ref_scale.int()).abs().max().item())
        mismatch = float((data != ref_data).float().mean().item())
        got = decode_records(cache.view(blocks, _BLOCK_SIZE, _RECORD_BYTES), slots, "mxfp8")
        rel = (got - rows.float()).norm(dim=-1) / rows.float().norm(dim=-1).clamp(min=1e-6)
        report.update(
            max_scale_gap=scale_gap,
            data_mismatch_fraction=mismatch,
            max_rel_l2=float(rel.max().item()),
        )
        if scale_gap > 1 or mismatch > _MAX_MISMATCH_FRACTION or report["max_rel_l2"] > _REL_TOL:
            raise KernelLaunchFailed(f"{_BACKEND} cache records mismatch Torch: {report}")

    written = torch.zeros(cache.shape[0] * _BLOCK_SIZE, dtype=torch.bool, device=cache.device)
    written[slots] = True
    idle_data = cache.view(cache.shape[0], -1)[:, : _BLOCK_SIZE * _DATA_BYTES]
    idle = idle_data.reshape(-1, _DATA_BYTES)[~written]
    report["untouched_slots_poisoned"] = bool((idle == _POISON).all().item())
    if not report["untouched_slots_poisoned"]:
        raise KernelLaunchFailed(f"{_BACKEND} wrote outside slot_mapping")
    return report


def build_inputs(torch: Any, device: Any, num_tokens: int, num_insert_tokens: int, heads: int):
    from profiling.runners.attention.compressed_sparse_mla_rope_cast import _cos_sin
    from profiling.runners.attention.compressed_sparse_mla_rope_cast_reference import (
        q_to_fused_layout,
    )

    generator = torch.Generator().manual_seed(0x41C5)
    q_std = torch.randn((num_tokens, heads, _HEAD_DIM), generator=generator).to(torch.bfloat16)
    kv = torch.randn((num_tokens, _HEAD_DIM), generator=generator).to(torch.bfloat16)
    positions = torch.randint(0, _MAX_POSITION, (num_tokens,), generator=generator)
    blocks = 1 + 2 * -(-max(1, num_insert_tokens) // _BLOCK_SIZE)
    slots = torch.randperm(blocks * _BLOCK_SIZE, generator=generator)[:num_insert_tokens]
    cache = torch.full((blocks, _BLOCK_SIZE, _RECORD_BYTES), _POISON, dtype=torch.uint8)
    return (
        q_std.to(device),
        q_to_fused_layout(q_std).contiguous().to(device),
        kv.to(device),
        positions.to(device),
        _cos_sin(torch, _MAX_POSITION, device),
        cache.to(device),
        slots.to(device),
    )


def profile_q_pad_kv_rope_mxfp8_insert_vllm_cuda(
    num_tokens: int,
    num_insert_tokens: int,
    num_heads: int,
    padded_heads: int,
    block_size: int,
    input_dtype: object,
    swa_cache_format: str,
) -> ComputeMetrics:
    """Time the fork's fused Q pad + MXFP8 SWA KV insert."""
    _validate_args(
        num_tokens,
        num_insert_tokens,
        num_heads,
        padded_heads,
        block_size,
        input_dtype,
        swa_cache_format,
    )
    try:
        import torch
        import vllm._C_stable_libtorch  # noqa: F401
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires the upstream-rebased vLLM fork (vllm_upstream_fork_env)"
        ) from exc
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        q_std, q_in, kv, positions, cos_sin, cache, slots = build_inputs(
            torch, device, num_tokens, num_insert_tokens, num_heads
        )
        cache_2d = cache.view(cache.shape[0], -1)

        def launch() -> Any:
            return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q_in,
                kv,
                cache_2d,
                slots,
                positions,
                cos_sin,
                padded_heads,
                1.0e-6,  # eps: read only by the Q norm, which is off
                block_size,
                False,  # apply_q_norm
                True,  # kv_mxfp8
                False,  # apply_q_rope
                True,  # is_q_interleaved
            )

        q_out = launch()
        torch.cuda.synchronize(device)
        check_outputs(torch, q_std, q_out, padded_heads, kv, positions, cos_sin, cache, slots)
        time_ms = Timer.cupti(launch, kernel_name=None)
        energy_j = Energy.perf(launch, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(
            logical_bytes(num_tokens, num_insert_tokens, num_heads, padded_heads)
            / (time_ms / 1000.0)
            / 1e9
        ),
        energy_j=float(energy_j),
    )


def logical_bytes(num_tokens: int, num_insert: int, num_heads: int, padded_heads: int) -> int:
    """Q read + padded Q write (when padding), KV read, positions, cos/sin, records."""
    q_bytes = 2 * _HEAD_DIM * num_tokens * (num_heads + padded_heads) if padded_heads else 0
    kv_bytes = num_insert * (2 * _HEAD_DIM + 8 + 8 + 4 * _ROPE_DIM + _RECORD_BYTES)
    return q_bytes + kv_bytes


__all__ = [
    "check_outputs",
    "profile_q_pad_kv_rope_mxfp8_insert_vllm_cuda",
    "reference_kv_rows",
    "reference_q",
]
