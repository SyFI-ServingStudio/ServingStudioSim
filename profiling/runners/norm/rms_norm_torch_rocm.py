"""Torch-on-ROCm runner for plain (non-residual) RMSNorm on MI300X.

The timed callable is ``torch.nn.functional.rms_norm(input, (hidden,), weight,
eps)`` -- PyTorch's own fused RMSNorm public op -- run on a ROCm/HIP device.
This is the first-step ROCm reference backend for the ``rms_norm`` kind: a real
production public callable (not a hand-rolled elementwise rewrite), portable to
CDNA3, used to prove the ROCm profiling path end to end before an aiter/Triton
fused norm is wired. Timing is kernel-only via ``Timer.rocprof`` (rocprofv3
kernel-dispatch durations), the ROCm counterpart of the CUPTI path the CUDA
backends use.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_EPS = 1e-6
# Count every dispatch of the one logical rms_norm call rather than filtering a
# device kernel name: the fused ROCm kernel name is release-dependent, and the
# call issues a single logical launch, so summing its dispatches is exact.
_KERNEL_NAME: str | None = None
_SUPPORTED_COMPUTE = frozenset({DType.BF16, DType.FP16})


def _validate_args(m: int, hidden: int, dtype: DType | str) -> tuple[int, int, DType]:
    m = int(m)
    hidden = int(hidden)
    dtype = DType.from_value(dtype)
    if m <= 0 or hidden <= 0:
        raise ValueError(f"m and hidden must be > 0, got m={m}, hidden={hidden}")
    if dtype not in _SUPPORTED_COMPUTE:
        raise ValueError(
            f"torch_rocm rms_norm supports {sorted(d.value for d in _SUPPORTED_COMPUTE)}, "
            f"got {dtype.value}"
        )
    return m, hidden, dtype


def _validate_rocm_device(torch: Any) -> None:
    # torch's HIP build exposes the ROCm device through the torch.cuda shim;
    # torch.version.hip distinguishes it from a real CUDA build.
    if getattr(torch.version, "hip", None) is None:
        raise ProfilerNotImplemented(
            "the torch_rocm rms_norm backend requires a ROCm (HIP) torch build"
        )
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("a ROCm device is required for the torch_rocm backend")


def profile_rms_norm_torch_rocm(m: int, hidden: int, dtype: DType | str) -> ComputeMetrics:
    """Profile PyTorch's fused RMSNorm on a ROCm device."""
    m, hidden, dtype = _validate_args(m, hidden, dtype)
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the rms_norm torch_rocm backend"
        ) from exc

    _validate_rocm_device(torch)

    try:
        torch_dtype = dtype.torch()
        generator = torch.Generator(device="cuda").manual_seed(17)
        input_tensor = torch.randn(
            (m, hidden), dtype=torch_dtype, device="cuda", generator=generator
        )
        weight = torch.randn((hidden,), dtype=torch_dtype, device="cuda", generator=generator)

        def kernel() -> Any:
            return F.rms_norm(input_tensor, (hidden,), weight, _EPS)

        output = kernel()
        torch.cuda.synchronize()
        values = input_tensor.float()
        expected = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + _EPS)
        torch.testing.assert_close(
            output.float(),
            (expected * weight.float()).to(torch_dtype).float(),
            rtol=2e-2,
            atol=2e-2,
        )

        time_ms = Timer.rocprof(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)

        # Logical traffic: read input and weight, write output.
        bytes_accessed = int((2 * m * hidden + hidden) * dtype.size_bytes())
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0.0 else 0.0
        flops = 4 * m * hidden + 2 * m
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0.0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
