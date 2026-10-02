"""ROCm FP8 dense-GEMM runner for MI300X — the ``rocm_scaled_mm`` backend of the
``single_gemm`` kind.

This is the MI300X replacement for the NVIDIA ``deepgemm`` FP8 dense GEMM. On
GLM-5.3-Flash the FP8 ``single_gemm`` kind carries the dense FFN and shared-expert
gate_up/down projections (the compute-bound bulk of the model's dense GEMM
FLOPs); DeepGEMM's ``fp8_gemm_nt`` is a Blackwell SM100 kernel with no CDNA3
path, so AMD runs the same multiply through PyTorch's ``torch._scaled_mm``, which
dispatches to hipBLASLt's scaled FP8 GEMM.

Dtype: the kind's schema dtype stays ``fp8_e4m3`` (the wire/table value), but the
device tensors are ``torch.float8_e4m3fnuz`` — the CDNA3 (gfx942) FP8 E4M3
variant. The OCP ``float8_e4m3fn`` NVIDIA uses is rejected by the gfx942 scaled
GEMM path, exactly as the MoE ROCm backend documents. Inputs are per-tensor
scaled (scale = 1.0), fp8 in / bf16 out, matching the deepgemm runner's
fp8-in/bf16-out contract and the single_gemm formula.

Timing is kernel-only via rocprofv3 dispatch durations
(``measure_registered_via_rocprofv3``), the ROCm counterpart of the CUPTI path
the NVIDIA GEMM backends use. hipBLASLt may autotune its heuristic on the first
call, so the measure uses the trailing-fold variant
(``dispatches_per_launch=1``): the one scaled-GEMM dispatch per call, averaged
over the trailing steady-state launches past any warm-up / autotune prefix.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_WARMUP = 5
_REP = 20
# One hipBLASLt scaled-GEMM dispatch per torch._scaled_mm call. The trailing fold
# isolates the steady-state launches past the device-init / autotune prefix.
_DISPATCHES_PER_LAUNCH = 1
_SUPPORTED_COMPUTE = frozenset({DType.FP8_E4M3})


def _validate_args(m: int, n: int, k: int, dtype: DType | str) -> tuple[int, int, int, DType]:
    m, n, k = int(m), int(n), int(k)
    dtype = DType.from_value(dtype)
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"m, n, k must be > 0, got m={m}, n={n}, k={k}")
    if dtype not in _SUPPORTED_COMPUTE:
        raise ValueError(
            "rocm_scaled_mm single-GEMM is FP8-only (CDNA3 float8_e4m3fnuz); "
            f"got dtype={dtype.value}. Use the torch_rocm backend for bf16/fp16 dense GEMM."
        )
    return m, n, k, dtype


def _fp8_dtype(torch: Any) -> Any:
    """The CDNA3 FP8 E4M3 variant (``float8_e4m3fnuz``).

    gfx942's scaled GEMM / quant path uses the ``fnuz`` variant; the OCP
    ``float8_e4m3fn`` NVIDIA uses is rejected there. ``float8_e4m3fnuz`` is always
    present in a ROCm torch build; fall back to the OCP dtype only so the
    import-light / non-ROCm path does not crash before the device check.
    """
    return getattr(torch, "float8_e4m3fnuz", None) or torch.float8_e4m3fn


def _validate_rocm_device(torch: Any) -> None:
    if getattr(torch.version, "hip", None) is None:
        raise ProfilerNotImplemented(
            "the rocm_scaled_mm single-GEMM backend requires a ROCm (HIP) torch build"
        )
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("a ROCm device is required for the rocm_scaled_mm backend")


def build_single_gemm_scaled_mm_kernel(
    m: int, n: int, k: int, dtype: DType | str
) -> dict[str, Any]:
    """Build the timed ``torch._scaled_mm`` callable + fixed FP8 operands.

    Shared by the runner and the rocprofv3 launch driver
    (``profiling.profilers.rocprof_run``) so the captured dispatches are the exact
    scaled-GEMM launches the runner would time. Operands are cast to FP8 once here
    (outside ``kernel()``); every timed call issues only the scaled GEMM.
    """
    m, n, k, dtype = _validate_args(m, n, k, dtype)
    import torch

    _validate_rocm_device(torch)
    fp8 = _fp8_dtype(torch)
    gen = torch.Generator(device="cuda").manual_seed(17)
    # NT layout, as vLLM passes it: A (m, k) row-major, B stored (n, k) row-major
    # so B.t() is the (k, n) column-major right-hand side scaled_mm requires.
    a_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device="cuda", generator=gen)
    b_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device="cuda", generator=gen)
    a_fp8 = a_bf16.to(fp8)
    b_fp8 = b_bf16.to(fp8)
    b_t = b_fp8.t()  # (k, n) column-major
    scale_a = torch.ones((), dtype=torch.float32, device="cuda")
    scale_b = torch.ones((), dtype=torch.float32, device="cuda")

    def kernel() -> Any:
        return torch._scaled_mm(
            a_fp8, b_t, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16
        )

    return {
        "torch": torch,
        "kernel": kernel,
        "a_fp8": a_fp8,
        "b_fp8": b_fp8,
        "warmup": _WARMUP,
        "rep": _REP,
    }


def profile_single_gemm_scaled_mm(
    m: int, n: int, k: int, dtype: DType | str
) -> ComputeMetrics:
    """Profile the FP8 dense GEMM on a ROCm/MI300X device via rocprofv3."""
    m, n, k, dtype = _validate_args(m, n, k, dtype)
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the single_gemm rocm_scaled_mm backend"
        ) from exc

    try:
        from profiling.profilers.rocprof_kernel_profiler import measure_registered_via_rocprofv3

        built = build_single_gemm_scaled_mm_kernel(m, n, k, dtype)
        torch = built["torch"]
        kernel = built["kernel"]

        # Assert the FP8 GEMM actually ran: compare its bf16 output against the
        # dequantized operands' product (scale = 1.0). A silent fallback or a
        # wrong layout would not match. Tolerances are loose because fp8 rounding
        # is lossy by design.
        output = kernel()
        torch.cuda.synchronize()
        ref = built["a_fp8"].to(torch.float32) @ built["b_fp8"].to(torch.float32).t()
        torch.testing.assert_close(output.float(), ref, rtol=5e-2, atol=5e-2)

        time_ms = measure_registered_via_rocprofv3(
            kind="single_gemm",
            backend="rocm_scaled_mm",
            spec={"m": m, "n": n, "k": k, "dtype": dtype.value},
            kernel_name_contains=None,
            warmup=_WARMUP,
            rep=_REP,
            dispatches_per_launch=_DISPATCHES_PER_LAUNCH,
        )
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)

        # fp8 in / bf16 out, mirroring the deepgemm single-GEMM accounting:
        # A fp8 (1B), B fp8 (1B), out bf16 (2B).
        flops = 2 * m * n * k
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0 else 0.0
        bytes_accessed = m * k + n * k + m * n * 2
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
