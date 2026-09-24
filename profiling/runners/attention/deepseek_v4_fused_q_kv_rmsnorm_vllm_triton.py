"""Profile the production fused QR/KV RMSNorm call.

``vllm_triton`` times DeepSeek V4's call in the pinned vLLM image.
``vllm_fork_triton`` times the alignment fork's shared
``vllm.models.common.ops.fused_qk_rmsnorm.fused_q_kv_rmsnorm``, which GLM-5.3's
MLA front end calls on the two split views of the ``fused_qkv_a`` output. The
fork's launch differs (PDL, ``num_warps=8`` at a 2048 block), so it is a
separate backend identity.
"""

from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "deepseek_v4_fused_q_kv_rmsnorm:vllm_triton"
_IDENTITY = (1536, 512, 1.0e-6, "bf16")
_FORK_BACKEND = "deepseek_v4_fused_q_kv_rmsnorm:vllm_fork_triton"
# GLM-5.3 uses rms_norm_eps=1e-5; the scalar never changes the launch.
_FORK_IDENTITIES = frozenset({(1536, 512, 1.0e-6, "bf16"), (1536, 512, 1.0e-5, "bf16")})


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


def profile_deepseek_v4_fused_q_kv_rmsnorm_vllm_fork_triton(
    num_tokens: int, q_dim: int, kv_dim: int, rms_eps: float, dtype: object
) -> ComputeMetrics:
    if type(num_tokens) is not int or not 1 <= num_tokens <= 65_536:
        raise ValueError("num_tokens must be an integer in [1, 65536]")
    identity = (q_dim, kv_dim, rms_eps, str(dtype))
    if identity not in _FORK_IDENTITIES:
        raise ProfilerNotImplemented(
            f"{_FORK_BACKEND} supports {sorted(_FORK_IDENTITIES)}, got {identity}"
        )
    try:
        import torch
        from vllm.models.common.ops.fused_qk_rmsnorm import fused_q_kv_rmsnorm
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_FORK_BACKEND} requires the vLLM fork") from exc
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_name() != "NVIDIA B200":
            raise ProfilerNotImplemented(f"{_FORK_BACKEND} requires NVIDIA B200")
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(43)
        # vllm/model_executor/layers/mla.py: q_c, kv_c are split views of the
        # fused_qkv_a output (row stride q_dim + kv_dim), not contiguous tensors.
        fused = torch.randn(
            (num_tokens, q_dim + kv_dim), dtype=torch.bfloat16, device=device, generator=generator
        )
        qr, kv = fused.split([q_dim, kv_dim], dim=-1)
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
        raise OOMError(f"{_FORK_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_FORK_BACKEND} failed") from exc
    logical_bytes = 2 * (2 * num_tokens * (q_dim + kv_dim) + q_dim + kv_dim)
    seconds = time_ms / 1000.0
    return ComputeMetrics(float(time_ms), 0.0, logical_bytes / seconds / 1e9, float(energy_j))
