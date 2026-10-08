"""Torch-on-ROCm runner for the fused query/KV RMSNorm on MI300X.

The timed callable is ``vllm.models.common.ops.fused_q_kv_rmsnorm`` -- the same
fused query-side + KV-side RMSNorm the NVIDIA ``vllm_triton`` backend times (the
DeepSeek-V4 / GLM-5.3 MLA input path). It is plain vLLM common code with no
``is_rocm()`` / aiter branch (verified on the pinned MI300X image: the module
does not reach aiter for this norm), so the ROCm path is the same callable built
for CDNA3. This is the ROCm counterpart of ``q_kv_rms_norm_vllm_triton``.

Timing is kernel-only via rocprofv3 (``measure_registered_via_rocprofv3``), the
ROCm counterpart of the CUPTI path. The NVIDIA backend sums every dispatch of the
call (``Timer.cupti(..., kernel_name=None)``). On the MI300X image the call fuses
to exactly ONE kernel dispatch (measured: constant one dispatch per call), so the
per-call time is that single dispatch summed over the trailing ``rep`` launches
(``dispatches_per_launch=1``), which is robust to the device-init prefix. Operands
are built on the host and moved to the device with ``.to()`` so tensor setup emits
no kernel dispatches into the whole-process trace.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.runners.attention.kda_recurrent_decode_torch_rocm import (
    _device_arch,
    _device_name,
    _is_mi300_series,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "q_kv_rms_norm:torch_rocm"
_KIND = "q_kv_rms_norm"
_MODULE = "vllm.models.common.ops"
_CALLABLE = "fused_q_kv_rmsnorm"
_GPU_SKU_TOKEN = "MI300X"
_WARMUP = 5
_REP = 20
# fused_q_kv_rmsnorm launches one fused kernel per call on the MI300X image
# (measured). The trace's kernel display name is generic ("kernel"), so the sum
# is over every dispatch (kernel_name=None) with the per-call count pinned here.
_DISPATCHES_PER_LAUNCH = 1
# Only this (q_dim, kv_dim) is a measured production shape (DeepSeek-V4 / GLM-5.3).
_Q_DIM = 1536
_KV_DIM = 512


def _validate_args(num_tokens: int, q_dim: int, kv_dim: int, rms_eps: float, dtype: Any):
    if type(num_tokens) is not int or not 1 <= num_tokens <= 65_536:
        raise ValueError("num_tokens must be an integer in [1, 65536]")
    dt = DType.from_value(dtype)
    if (q_dim, kv_dim) != (_Q_DIM, _KV_DIM):
        raise ValueError(f"{_BACKEND} requires q_dim={_Q_DIM}, kv_dim={_KV_DIM}")
    if dt is not DType.BF16:
        raise ValueError(f"{_BACKEND} requires dtype=bf16, got {dt.value}")
    return num_tokens, q_dim, kv_dim, float(rms_eps), dt


def _reference(torch: Any, values: Any, weight: Any, eps: float) -> Any:
    values_f32 = values.float()
    return (
        values_f32 * torch.rsqrt(values_f32.square().mean(-1, keepdim=True) + eps) * weight.float()
    ).to(values.dtype)


def build_q_kv_rms_norm_kernel(
    num_tokens: int, q_dim: int, kv_dim: int, rms_eps: float, dtype: DType | str
) -> dict[str, Any]:
    """Build the timed callable + operands, shared by the runner and the driver.

    Operands are built on the host and moved to the device, so setup launches no
    GPU kernels and the only dispatches the trace sees are the call's own. qr/kv
    are split views of one fused projection, as the production caller passes them.
    """
    num_tokens, q_dim, kv_dim, rms_eps, dt = _validate_args(
        num_tokens, q_dim, kv_dim, rms_eps, dtype
    )
    import importlib

    import torch

    if getattr(torch.version, "hip", None) is None:
        raise ProfilerNotImplemented(f"{_BACKEND} requires a ROCm (HIP) torch build")
    try:
        module = importlib.import_module(_MODULE)
    except Exception as exc:  # pragma: no cover - requires the vLLM-ROCm image
        raise ProfilerNotImplemented(f"{_BACKEND} requires the vllm_rocm_env vLLM ({_MODULE})") from exc
    callable_ = getattr(module, _CALLABLE, None)
    if not callable(callable_):
        raise ProfilerNotImplemented(f"{_BACKEND} requires {_MODULE}.{_CALLABLE}")

    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device="cpu").manual_seed(43)
    fused = torch.randn(num_tokens, q_dim + kv_dim, dtype=torch.float32, generator=gen).to(
        torch.bfloat16
    ).to(device)
    qr, kv = fused.split([q_dim, kv_dim], dim=-1)
    q_weight = torch.randn(q_dim, dtype=torch.float32, generator=gen).to(torch.bfloat16).to(device)
    kv_weight = torch.randn(kv_dim, dtype=torch.float32, generator=gen).to(torch.bfloat16).to(device)

    def kernel() -> Any:
        return callable_(qr, kv, q_weight, kv_weight, rms_eps)

    return {
        "torch": torch,
        "kernel": kernel,
        "qr": qr,
        "kv": kv,
        "q_weight": q_weight,
        "kv_weight": kv_weight,
        "rms_eps": rms_eps,
        "num_tokens": num_tokens,
        "q_dim": q_dim,
        "kv_dim": kv_dim,
        "warmup": _WARMUP,
        "rep": _REP,
    }


def profile_q_kv_rms_norm_torch_rocm(
    num_tokens: int, q_dim: int, kv_dim: int, rms_eps: float, dtype: DType | str
) -> ComputeMetrics:
    """Profile vLLM's fused query/KV RMSNorm on an MI300X."""
    num_tokens, q_dim, kv_dim, rms_eps, dt = _validate_args(
        num_tokens, q_dim, kv_dim, rms_eps, dtype
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc
    try:
        if not torch.cuda.is_available():
            raise ProfilerNotImplemented(f"a ROCm device is required for {_BACKEND}")
        if getattr(torch.version, "hip", None) is None:
            raise ProfilerNotImplemented(f"{_BACKEND} requires a ROCm (HIP) torch build")
        if not _is_mi300_series(torch):
            raise ProfilerNotImplemented(
                f"{_BACKEND} is verified only on {_GPU_SKU_TOKEN} (CDNA3 gfx942), got "
                f"name={_device_name(torch)!r} arch={_device_arch(torch)!r}"
            )
        from profiling.profilers.rocprof_kernel_profiler import (
            measure_registered_via_rocprofv3,
        )

        built = build_q_kv_rms_norm_kernel(num_tokens, q_dim, kv_dim, rms_eps, dtype)
        # Correctness vs the fp32 reference, on sampled rows; also proves the
        # AMD/ROCm call runs correctly.
        aqr, akv = built["kernel"]()
        torch.cuda.synchronize()
        rows = torch.tensor(sorted({0, num_tokens // 2, num_tokens - 1}), device=aqr.device)
        torch.testing.assert_close(
            aqr[rows], _reference(torch, built["qr"][rows], built["q_weight"], rms_eps),
            rtol=1e-2, atol=1e-2,
        )
        torch.testing.assert_close(
            akv[rows], _reference(torch, built["kv"][rows], built["kv_weight"], rms_eps),
            rtol=1e-2, atol=1e-2,
        )

        time_ms = measure_registered_via_rocprofv3(
            kind=_KIND,
            backend="torch_rocm",
            spec={
                "num_tokens": num_tokens,
                "q_dim": q_dim,
                "kv_dim": kv_dim,
                "rms_eps": rms_eps,
                "dtype": dt.value,
            },
            kernel_name_contains=None,
            warmup=_WARMUP,
            rep=_REP,
            dispatches_per_launch=_DISPATCHES_PER_LAUNCH,
        )
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        raise OOMError(f"{_BACKEND} ran out of device memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    logical_bytes = 2 * (2 * num_tokens * (q_dim + kv_dim) + q_dim + kv_dim)
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=logical_bytes / seconds / 1e9 if seconds > 0 else 0.0,
        energy_j=0.0,
    )


__all__ = ["build_q_kv_rms_norm_kernel", "profile_q_kv_rms_norm_torch_rocm"]
