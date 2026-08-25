"""Profile the production DeepSeek V4 fused QR/KV RMSNorm call."""

from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "deepseek_v4_fused_q_kv_rmsnorm:vllm_triton"
_IDENTITY = (1536, 512, 1.0e-6, "bf16")


def _validate_args(num_tokens: int, q_dim: int, kv_dim: int, rms_eps: float, dtype: object) -> None:
    if type(num_tokens) is not int or not 1 <= num_tokens <= 65_536:
        raise ValueError("num_tokens must be an integer in [1, 65536]")
    identity = (q_dim, kv_dim, rms_eps, str(dtype))
    if identity != _IDENTITY:
        raise ProfilerNotImplemented(f"{_BACKEND} supports {_IDENTITY}, got {identity}")


def _reference(torch: Any, values: Any, weight: Any, eps: float) -> Any:
    values_f32 = values.float()
    return (
        values_f32 * torch.rsqrt(values_f32.square().mean(-1, keepdim=True) + eps) * weight.float()
    ).to(values.dtype)


def profile_deepseek_v4_fused_q_kv_rmsnorm_vllm_triton(
    num_tokens: int, q_dim: int, kv_dim: int, rms_eps: float, dtype: object
) -> ComputeMetrics:
    _validate_args(num_tokens, q_dim, kv_dim, rms_eps, dtype)
    try:
        import torch
        from vllm.models.deepseek_v4.common.ops import fused_q_kv_rmsnorm
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires pinned vLLM") from exc
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_name() != "NVIDIA H200":
            raise ProfilerNotImplemented(f"{_BACKEND} requires NVIDIA H200")
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(43)
        qr = torch.randn(
            (num_tokens, q_dim), dtype=torch.bfloat16, device=device, generator=generator
        )
        kv = torch.randn(
            (num_tokens, kv_dim), dtype=torch.bfloat16, device=device, generator=generator
        )
        q_weight = torch.randn(q_dim, dtype=torch.bfloat16, device=device, generator=generator)
        kv_weight = torch.randn(kv_dim, dtype=torch.bfloat16, device=device, generator=generator)

        def launch():
            return fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, rms_eps)

        actual_qr, actual_kv = launch()
        torch.cuda.synchronize()
        rows = torch.tensor(sorted({0, num_tokens // 2, num_tokens - 1}), device=device)
        torch.testing.assert_close(
            actual_qr[rows], _reference(torch, qr[rows], q_weight, rms_eps), rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            actual_kv[rows], _reference(torch, kv[rows], kv_weight, rms_eps), rtol=1e-2, atol=1e-2
        )
        time_ms = Timer.cupti(launch, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc
    logical_bytes = 2 * (2 * num_tokens * (q_dim + kv_dim) + q_dim + kv_dim)
    seconds = time_ms / 1000.0
    return ComputeMetrics(float(time_ms), 0.0, logical_bytes / seconds / 1e9, float(energy_j))


__all__ = ["profile_deepseek_v4_fused_q_kv_rmsnorm_vllm_triton"]
