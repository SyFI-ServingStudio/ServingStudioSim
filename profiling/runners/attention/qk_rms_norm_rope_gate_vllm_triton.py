"""Profile vLLM's fused QK RMSNorm + partial RoPE + gate split Triton kernel.

The call is the one ``Qwen3NextAttention._project_qkv_gate`` makes: the
query-and-gate and key operands are column slices of one QKV projection
output, so their row stride is the whole projection width.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.qk_rms_norm_rope_gate_reference import (
    qk_rms_norm_rope_gate_reference,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "qk_rms_norm_rope_gate:vllm_triton"
# Runtime constants that never change the launch: eps is a constexpr scalar, and
# neither the cos/sin table length nor the position values moved the time.
_EPS = 1e-6
_COS_SIN_ROWS = 4096


def _validate_args(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    dtype: DType | str,
) -> None:
    for name, value in (
        ("num_tokens", num_tokens),
        ("num_q_heads", num_q_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("rotary_dim", rotary_dim),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even and <= head_dim, got {rotary_dim}")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires dtype=bf16, got {dtype}")


def _mrope_section(rotary_dim: int) -> tuple[int, int, int]:
    # Any split summing to rotary_dim / 2 launches the same kernel; it only picks
    # which position row each rotated pair loads.
    half = rotary_dim // 2
    return half - 2 * (half // 3), half // 3, half // 3


def profile_qk_rms_norm_rope_gate_vllm_triton(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope: bool,
    dtype: DType | str,
) -> ComputeMetrics:
    _validate_args(num_tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim, dtype)
    try:
        import torch
        from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM environment") from exc

    try:
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(0)

        def randn(*shape: int) -> Any:
            return torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)

        q_size, kv_size = num_q_heads * head_dim, num_kv_heads * head_dim
        qkv = randn(num_tokens, 2 * q_size + 2 * kv_size)
        q_gate, k, _ = qkv.split([2 * q_size, kv_size, kv_size], dim=-1)
        q_weight = 1.0 + 0.1 * randn(head_dim)
        k_weight = 1.0 + 0.1 * randn(head_dim)
        cos_sin_cache = randn(_COS_SIN_ROWS, rotary_dim)
        token_positions = torch.arange(num_tokens, device=device) % _COS_SIN_ROWS
        if mrope:
            # Text tokens carry the same position in all three rows; vLLM slices
            # them from a persistent (3, max_tokens + 1) buffer.
            buffer = torch.zeros((3, num_tokens + 1), dtype=torch.int64, device=device)
            buffer[:, :num_tokens] = token_positions
            positions, section = buffer[:, :num_tokens], _mrope_section(rotary_dim)
        else:
            positions, section = token_positions, None

        def launch() -> Any:
            return fused_qk_rmsnorm_rope_gate(
                q_gate,
                k,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                _EPS,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rotary_dim,
                mrope_section=section,
            )

        actual = launch()
        torch.cuda.synchronize()
        expected = qk_rms_norm_rope_gate_reference(
            torch,
            q_gate,
            k,
            q_weight,
            k_weight,
            cos_sin_cache,
            token_positions,
            _EPS,
            head_dim,
            rotary_dim,
        )
        for actual_plane, expected_plane in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_plane, expected_plane, rtol=2e-2, atol=2e-2)

        time_ms = Timer.cupti(launch, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    logical_bytes = 2 * num_tokens * head_dim * (4 * num_q_heads + 2 * num_kv_heads)
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(logical_bytes / seconds / 1e9),
        energy_j=float(energy_j),
    )


__all__ = ["profile_qk_rms_norm_rope_gate_vllm_triton"]
