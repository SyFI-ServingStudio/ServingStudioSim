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
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_EPS = 1e-6
# torch.nn.functional.rms_norm dispatches to the vectorized layer-norm kernel on
# ROCm (verified on MI210 / torch 2.12+rocm: display_name
# "vectorized_layer_norm_kernel<c10::BFloat16, float, true>"). rocprofv3 traces
# the whole process, so the measured launches are isolated from the input-setup
# kernels (randn) by this substring; it is the stable device-kernel family name,
# not release-pinned template args.
_KERNEL_NAME = "vectorized_layer_norm_kernel"
_WARMUP = 5
_REP = 20
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


def build_rms_norm_kernel(m: int, hidden: int, dtype: DType | str) -> dict[str, Any]:
    """Construct the timed callable + fixed inputs, shared by the runner and the
    rocprofv3 launch driver (``profiling.profilers.rocprof_run``).

    Returning the pieces (not just the callable) lets the runner run its
    correctness check and lets the under-rocprofv3 driver replay the identical
    kernel from the same spec with the same fixed seed, so the captured
    dispatches are the exact launches the runner would have timed.
    """
    m, hidden, dtype = _validate_args(m, hidden, dtype)
    import torch
    import torch.nn.functional as F

    _validate_rocm_device(torch)
    torch_dtype = dtype.torch()
    generator = torch.Generator(device="cuda").manual_seed(17)
    input_tensor = torch.randn((m, hidden), dtype=torch_dtype, device="cuda", generator=generator)
    weight = torch.randn((hidden,), dtype=torch_dtype, device="cuda", generator=generator)

    def kernel() -> Any:
        return F.rms_norm(input_tensor, (hidden,), weight, _EPS)

    return {
        "torch": torch,
        "kernel": kernel,
        "input_tensor": input_tensor,
        "weight": weight,
        "torch_dtype": torch_dtype,
        "warmup": _WARMUP,
        "rep": _REP,
        "kernel_name": _KERNEL_NAME,
    }


def profile_rms_norm_torch_rocm(m: int, hidden: int, dtype: DType | str) -> ComputeMetrics:
    """Profile PyTorch's fused RMSNorm on a ROCm device via rocprofv3."""
    m, hidden, dtype = _validate_args(m, hidden, dtype)
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the rms_norm torch_rocm backend"
        ) from exc

    try:
        from profiling.profilers.rocprof_kernel_profiler import measure_registered_via_rocprofv3

        built = build_rms_norm_kernel(m, hidden, dtype)
        torch = built["torch"]
        kernel = built["kernel"]
        output = kernel()
        torch.cuda.synchronize()
        values = built["input_tensor"].float()
        expected = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + _EPS)
        torch.testing.assert_close(
            output.float(),
            (expected * built["weight"].float()).to(built["torch_dtype"]).float(),
            rtol=2e-2,
            atol=2e-2,
        )

        # Whole-process rocprofv3 capture of the identical kernel, replayed from
        # the same spec under the tracer; kernel-only time from rocpd dispatch
        # durations with warmup launches dropped.
        time_ms = measure_registered_via_rocprofv3(
            kind="rms_norm",
            backend="torch_rocm",
            spec={"m": m, "hidden": hidden, "dtype": dtype.value},
            kernel_name_contains=_KERNEL_NAME,
            warmup=_WARMUP,
            rep=_REP,
        )
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
