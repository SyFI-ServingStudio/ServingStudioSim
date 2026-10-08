"""Torch TensorIterator element-wise runner on ROCm/MI300X.

Same byte contract as ``profiling.runners.elementwise.torch`` / ``.triton``
(``input_size_bytes`` -> ``output_size_bytes``, dtype-agnostic ``uint8``), a
ROCm realization timed with rocprofv3 instead of CUPTI. This is the measured
MI300X anchor for the ``elementwise`` byte-placeholder floor: the glue slots the
GLM-5.3-Flash arch already prices as ``elementwise`` (embedding gather, mHC
stream expand/contract, the MoE input/combine copies) need a backend with
MI300X profile.db rows, and this is it. It mirrors ``rms_norm``'s ``torch_rocm``
first-step ROCm reference: a real public PyTorch op (eager TensorIterator), run
on a ROCm/HIP device, timed kernel-only via ``Timer.rocprof`` (rocprofv3
kernel-dispatch durations), and gated to CDNA3 (gfx942) so it never competes
with the NVIDIA ``triton`` / ``torch`` rows.

The device kernel is one of (matching the CUDA torch runner's realizations):

* ``at::native::...fill...`` for zero-fill (fan-in 0),
* a vectorized elementwise map for fan-in 1 (``bitwise_not``),
* a reduce for fan-in >= 2 (``amax`` along the fan axis).

``torch.empty`` allocates the buffers (no ``randn`` setup kernels), so the
rocprofv3 capture needs no kernel-name filter to isolate the measured launch
from input setup; ``kernel_name_contains=None`` sums every dispatch in the
timed window exactly as the CUDA torch runner's ``Timer.cupti(kernel_name=None)``
does.
"""

from __future__ import annotations

from typing import Any

from profiling.profilers.energy import Energy
from profiling.runners.attention.kda_recurrent_decode_torch_rocm import (
    _device_arch,
    _device_name,
    _is_mi300_series,
)
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "torch_rocm"
_KIND = "elementwise"
_GPU_SKU_TOKEN = "MI300X"
_WARMUP = 5
_REP = 20


def _rounded_fan_in(input_size_bytes: int, output_size_bytes: int) -> int:
    """Fan-in the other elementwise runners use, so all backends key the same shape."""
    return max(1, int(input_size_bytes / output_size_bytes + 0.5))


def _validate_args(input_size_bytes: int, output_size_bytes: int) -> tuple[int, int]:
    input_size_bytes = int(input_size_bytes)
    output_size_bytes = int(output_size_bytes)
    if input_size_bytes < 0:
        raise ValueError("input_size_bytes must be >= 0")
    if output_size_bytes <= 0:
        raise ValueError("output_size_bytes must be > 0")
    return input_size_bytes, output_size_bytes


def build_elementwise_kernel(input_size_bytes: int, output_size_bytes: int) -> dict[str, Any]:
    """Construct the timed callable + fixed inputs, shared by the runner and the
    rocprofv3 launch driver (``profiling.profilers.rocprof_run``).

    Returning the pieces (not just the callable) lets the under-rocprofv3 driver
    replay the identical kernel from the same spec, so the captured dispatches
    are the exact launches the runner would have timed.
    """
    input_size_bytes, output_size_bytes = _validate_args(input_size_bytes, output_size_bytes)
    import torch

    if getattr(torch.version, "hip", None) is None:
        raise ProfilerNotImplemented(f"{_BACKEND} requires a ROCm (HIP) torch build")
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"a ROCm device is required for {_BACKEND}")

    output_tensor = torch.empty(output_size_bytes, dtype=torch.uint8, device="cuda")
    fan_in = 0 if input_size_bytes == 0 else _rounded_fan_in(input_size_bytes, output_size_bytes)
    input_tensor = torch.empty(max(fan_in, 1) * output_size_bytes, dtype=torch.uint8, device="cuda")

    if fan_in == 0:
        # Zero-fill: one write pass.
        def kernel() -> Any:
            output_tensor.zero_()
            return output_tensor

    elif fan_in == 1:
        # Unary map that stays in uint8 (no dtype promotion), so measured bytes
        # match the requested contract exactly.
        def kernel() -> Any:
            return torch.bitwise_not(input_tensor, out=output_tensor)

    else:
        # Fan-in reduce. `amax` keeps the uint8 output dtype (`sum` would promote
        # and change the write width).
        fan_view = input_tensor.view(fan_in, output_size_bytes)

        def kernel() -> Any:
            return torch.amax(fan_view, dim=0, out=output_tensor)

    return {
        "torch": torch,
        "kernel": kernel,
        "input_tensor": input_tensor,
        "output_tensor": output_tensor,
        "warmup": _WARMUP,
        "rep": _REP,
        # torch.empty setup launches no device kernel, so the timed window holds
        # only the measured op; no name filter needed (see module docstring).
        "kernel_name": None,
    }


def profile_elementwise_torch_rocm(
    input_size_bytes: int, output_size_bytes: int
) -> ComputeMetrics:
    """Profile the byte-mover elementwise op on an MI300X via rocprofv3."""
    input_size_bytes, output_size_bytes = _validate_args(input_size_bytes, output_size_bytes)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {_BACKEND}") from exc

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
        from profiling.profilers.rocprof_kernel_profiler import measure_registered_via_rocprofv3

        built = build_elementwise_kernel(input_size_bytes, output_size_bytes)
        kernel = built["kernel"]
        kernel()
        torch.cuda.synchronize()

        # Whole-process rocprofv3 capture of the identical kernel, replayed from
        # the same spec under the tracer; kernel-only time from rocpd dispatch
        # durations with warmup launches dropped.
        time_ms = measure_registered_via_rocprofv3(
            kind=_KIND,
            backend=_BACKEND,
            spec={
                "input_size_bytes": input_size_bytes,
                "output_size_bytes": output_size_bytes,
            },
            kernel_name_contains=built["kernel_name"],
            warmup=_WARMUP,
            rep=_REP,
        )
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)

        # Bandwidth-bound byte mover (read input + write output), same accounting
        # as the triton / torch runners so the curves stay comparable. The
        # reference reports no FLOPs for elementwise, so tflops stays 0.
        bytes_accessed = input_size_bytes + output_size_bytes
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
