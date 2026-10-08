"""Torch-on-ROCm BF16 GEMM runners for MI300X.

Two backends share this module, both ``torch_rocm``:

* ``single_gemm`` — ``torch.nn.functional.linear(x, W)`` with vLLM/Transformers'
  ``(n, k)`` weight layout. On GLM-5.3-Flash the bf16 ``single_gemm`` kind carries
  the KDA (in_proj / f_b / g_b / o) and DSA (fused_qkv_a / q_b / o) attention
  projections and the LM head — the 16-bit dense GEMMs the NVIDIA build routes
  through ``torch_linear_vllm``, which is B200-gated.
* ``batched_gemm`` — ``torch.bmm`` for MLA's per-head query absorption and value
  expansion, the AMD counterpart of the NVIDIA ``torch_mla_*`` bmm backends.

Both are real production public callables (``F.linear`` / ``torch.bmm``), the
first-step ROCm reference GEMMs on CDNA3 — portable, proven end to end before an
aiter / hipBLASLt-tuned kernel is wired. Timing is kernel-only via rocprofv3
dispatch durations (the ROCm counterpart of the CUPTI path). Each call is one
library GEMM dispatch; the measure uses the trailing-fold variant
(``dispatches_per_launch=1``) so any device-init / autotune prefix is dropped.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_WARMUP = 5
_REP = 20
_DISPATCHES_PER_LAUNCH = 1
_SUPPORTED_COMPUTE = frozenset({DType.BF16, DType.FP16})


def _check_dtype(dtype: DType) -> None:
    if dtype not in _SUPPORTED_COMPUTE:
        raise ValueError(
            f"torch_rocm GEMM supports {sorted(d.value for d in _SUPPORTED_COMPUTE)}, "
            f"got {dtype.value}. Use rocm_scaled_mm for fp8 dense GEMM."
        )


def _validate_single(m: int, n: int, k: int, dtype: DType | str) -> tuple[int, int, int, DType]:
    m, n, k = int(m), int(n), int(k)
    dtype = DType.from_value(dtype)
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"m, n, k must be > 0, got m={m}, n={n}, k={k}")
    _check_dtype(dtype)
    return m, n, k, dtype


def _validate_batched(
    num_batches: int, m: int, n: int, k: int, dtype: DType | str
) -> tuple[int, int, int, int, DType]:
    num_batches, m, n, k = int(num_batches), int(m), int(n), int(k)
    dtype = DType.from_value(dtype)
    if num_batches <= 0 or m <= 0 or n <= 0 or k <= 0:
        raise ValueError(
            f"num_batches, m, n, k must be > 0, got num_batches={num_batches}, m={m}, n={n}, k={k}"
        )
    _check_dtype(dtype)
    return num_batches, m, n, k, dtype


def _validate_rocm_device(torch: Any, who: str) -> None:
    if getattr(torch.version, "hip", None) is None:
        raise ProfilerNotImplemented(f"the {who} backend requires a ROCm (HIP) torch build")
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"a ROCm device is required for the {who} backend")


# --------------------------------------------------------------------------- #
# single_gemm (F.linear)
# --------------------------------------------------------------------------- #
def build_single_gemm_torch_rocm_kernel(
    m: int, n: int, k: int, dtype: DType | str
) -> dict[str, Any]:
    m, n, k, dtype = _validate_single(m, n, k, dtype)
    import torch
    import torch.nn.functional as F

    _validate_rocm_device(torch, "single_gemm torch_rocm")
    td = dtype.torch()
    gen = torch.Generator(device="cuda").manual_seed(17)
    x = torch.randn((m, k), dtype=td, device="cuda", generator=gen)
    weight = torch.randn((n, k), dtype=td, device="cuda", generator=gen)

    def kernel() -> Any:
        return F.linear(x, weight)

    return {"torch": torch, "kernel": kernel, "x": x, "weight": weight, "warmup": _WARMUP, "rep": _REP}


def profile_single_gemm_torch_rocm(
    m: int, n: int, k: int, dtype: DType | str
) -> ComputeMetrics:
    m, n, k, dtype = _validate_single(m, n, k, dtype)
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the single_gemm torch_rocm backend"
        ) from exc

    try:
        from profiling.profilers.rocprof_kernel_profiler import measure_registered_via_rocprofv3

        built = build_single_gemm_torch_rocm_kernel(m, n, k, dtype)
        torch = built["torch"]
        kernel = built["kernel"]

        output = kernel()
        torch.cuda.synchronize()
        ref = built["x"].float() @ built["weight"].float().t()
        torch.testing.assert_close(output.float(), ref, rtol=2e-2, atol=2e-2)

        time_ms = measure_registered_via_rocprofv3(
            kind="single_gemm",
            backend="torch_rocm",
            spec={"m": m, "n": n, "k": k, "dtype": dtype.value},
            kernel_name_contains=None,
            warmup=_WARMUP,
            rep=_REP,
            dispatches_per_launch=_DISPATCHES_PER_LAUNCH,
        )
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)

        flops = 2 * m * n * k
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0 else 0.0
        bytes_accessed = (m * k + n * k + m * n) * dtype.size_bytes()
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


# --------------------------------------------------------------------------- #
# batched_gemm (torch.bmm)
# --------------------------------------------------------------------------- #
def build_batched_gemm_torch_rocm_kernel(
    num_batches: int, m: int, n: int, k: int, dtype: DType | str
) -> dict[str, Any]:
    num_batches, m, n, k, dtype = _validate_batched(num_batches, m, n, k, dtype)
    import torch

    _validate_rocm_device(torch, "batched_gemm torch_rocm")
    td = dtype.torch()
    gen = torch.Generator(device="cuda").manual_seed(17)
    a = torch.randn((num_batches, m, k), dtype=td, device="cuda", generator=gen)
    b = torch.randn((num_batches, k, n), dtype=td, device="cuda", generator=gen)

    def kernel() -> Any:
        return torch.bmm(a, b)

    return {"torch": torch, "kernel": kernel, "a": a, "b": b, "warmup": _WARMUP, "rep": _REP}


def profile_batched_gemm_torch_rocm(
    num_batches: int, m: int, n: int, k: int, dtype: DType | str
) -> ComputeMetrics:
    num_batches, m, n, k, dtype = _validate_batched(num_batches, m, n, k, dtype)
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the batched_gemm torch_rocm backend"
        ) from exc

    try:
        from profiling.profilers.rocprof_kernel_profiler import measure_registered_via_rocprofv3

        built = build_batched_gemm_torch_rocm_kernel(num_batches, m, n, k, dtype)
        torch = built["torch"]
        kernel = built["kernel"]

        output = kernel()
        torch.cuda.synchronize()
        ref = torch.bmm(built["a"].float(), built["b"].float())
        torch.testing.assert_close(output.float(), ref, rtol=2e-2, atol=2e-2)

        time_ms = measure_registered_via_rocprofv3(
            kind="batched_gemm",
            backend="torch_rocm",
            spec={"num_batches": num_batches, "m": m, "n": n, "k": k, "dtype": dtype.value},
            kernel_name_contains=None,
            warmup=_WARMUP,
            rep=_REP,
            dispatches_per_launch=_DISPATCHES_PER_LAUNCH,
        )
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)

        flops = 2 * num_batches * m * n * k
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0 else 0.0
        bytes_accessed = num_batches * (m * k + k * n + m * n) * dtype.size_bytes()
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
