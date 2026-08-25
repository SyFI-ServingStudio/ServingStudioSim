"""Profile the production DeepSeek V4 fused Q/KV cache-insert kernel."""

import math
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "deepseek_v4_qnorm_rope_kv_insert:vllm_cuda"
_IDENTITY = (
    64,
    64,
    512,
    64,
    256,
    1.0e-6,
    "bf16",
    "fp8_ds_mla",
    "block_segregated_data_then_scales",
    "ue8m0",
)


def _validate_args(num_tokens: int, num_insert_tokens: int, identity: tuple[object, ...]) -> None:
    if type(num_tokens) is not int or not 1 <= num_tokens <= 65_536:
        raise ValueError("num_tokens must be an integer in [1, 65536]")
    if type(num_insert_tokens) is not int or not 0 <= num_insert_tokens <= num_tokens:
        raise ValueError("num_insert_tokens must be in [0, num_tokens]")
    if identity != _IDENTITY:
        raise ProfilerNotImplemented(f"{_BACKEND} supports {_IDENTITY}, got {identity}")


def _rope(torch: Any, values: Any, positions: Any, cache: Any) -> Any:
    output = values.float().clone()
    rope = output[..., -64:].reshape(*output.shape[:-1], 32, 2)
    cos_sin = cache[positions]
    cos, sin = cos_sin[..., :32], cos_sin[..., 32:]
    while cos.ndim < rope.ndim - 1:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    even, odd = rope[..., 0], rope[..., 1]
    output[..., -64:] = torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).reshape(
        output.shape[:-1] + (64,)
    )
    return output


def _check(
    torch: Any,
    q: Any,
    kv: Any,
    positions: Any,
    cache: Any,
    output: Any,
    packed: Any,
    insert: int,
    eps: float,
    block_size: int,
) -> None:
    q_f32 = q.float()
    q_ref = q_f32 * torch.rsqrt(q_f32.square().mean(-1, keepdim=True) + eps)
    q_ref = _rope(torch, q_ref, positions, cache).to(torch.bfloat16)
    rows = torch.tensor(sorted({0, q.shape[0] // 2, q.shape[0] - 1}), device=q.device)
    torch.testing.assert_close(output[rows], q_ref[rows], rtol=1e-2, atol=1e-2)
    kv_ref = _rope(torch, kv, positions, cache).to(torch.bfloat16)
    for token in sorted({0, insert // 2, insert - 1} if insert else set()):
        block, offset = divmod(token, block_size)
        raw = packed[block]
        data = raw[offset * 576 : (offset + 1) * 576]
        scales = raw[block_size * 576 + offset * 8 : block_size * 576 + (offset + 1) * 8]
        blocks = kv_ref[token, :448].float().reshape(7, 64)
        exponent = torch.ceil(torch.log2(torch.clamp(blocks.abs().amax(-1), min=1e-4) / 448.0))
        expected_scales = torch.cat(
            (
                (exponent + 127).clamp(0, 255).to(torch.uint8),
                torch.zeros(1, dtype=torch.uint8, device=q.device),
            )
        )
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
        expected_fp8 = (
            (blocks / torch.exp2(exponent)[:, None])
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .reshape(-1)
        )
        assert torch.equal(data[:448].view(torch.float8_e4m3fn).float(), expected_fp8.float())
        assert torch.equal(data[448:].view(torch.bfloat16), kv_ref[token, 448:])


def profile_deepseek_v4_qnorm_rope_kv_insert_vllm_cuda(
    num_tokens: int,
    num_insert_tokens: int,
    num_heads: int,
    padded_heads: int,
    head_dim: int,
    rope_dim: int,
    block_size: int,
    rms_eps: float,
    input_dtype: object,
    cache_dtype: str,
    cache_layout: str,
    scale_format: str,
) -> ComputeMetrics:
    identity = (
        num_heads,
        padded_heads,
        head_dim,
        rope_dim,
        block_size,
        rms_eps,
        str(input_dtype),
        cache_dtype,
        cache_layout,
        scale_format,
    )
    _validate_args(num_tokens, num_insert_tokens, identity)
    try:
        import torch
        import vllm._C_stable_libtorch  # noqa: F401
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires pinned vLLM") from exc
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_name() != "NVIDIA H200":
            raise ProfilerNotImplemented(f"{_BACKEND} requires NVIDIA H200")
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(47)
        q = torch.randn(
            (num_tokens, num_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        kv = torch.randn(
            (num_tokens, head_dim), dtype=torch.bfloat16, device=device, generator=generator
        )
        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
        angles = (
            positions.float()[:, None]
            * (torch.arange(32, device=device).float()[None, :] + 1)
            / 65_536
        )
        cos_sin_cache = torch.cat((angles.cos(), angles.sin()), 1)
        slots = torch.arange(num_insert_tokens, dtype=torch.int64, device=device)
        blocks = max(1, math.ceil(max(1, num_insert_tokens) / block_size))
        packed = torch.full((blocks, block_size * 584), 0xA5, dtype=torch.uint8, device=device)

        def launch():
            return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q, kv, packed, slots, positions, cos_sin_cache, padded_heads, rms_eps, block_size
            )

        output = launch()
        torch.cuda.synchronize()
        _check(
            torch,
            q,
            kv,
            positions,
            cos_sin_cache,
            output,
            packed,
            num_insert_tokens,
            rms_eps,
            block_size,
        )
        time_ms = Timer.cupti(launch, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc
    logical_bytes = (
        2 * num_tokens * (num_heads * head_dim + head_dim + padded_heads * head_dim)
        + 8 * (num_tokens + num_insert_tokens)
        + 4 * num_tokens * rope_dim
        + 584 * num_insert_tokens
    )
    seconds = time_ms / 1000.0
    return ComputeMetrics(float(time_ms), 0.0, logical_bytes / seconds / 1e9, float(energy_j))


__all__ = ["profile_deepseek_v4_qnorm_rope_kv_insert_vllm_cuda"]
