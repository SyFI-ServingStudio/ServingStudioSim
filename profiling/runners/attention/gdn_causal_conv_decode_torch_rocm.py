"""Torch-on-ROCm runner for the GDN/KDA causal-convolution decode on MI300X.

The short depthwise causal convolution that runs before the KDA recurrent decode
(``Glm5NextLinearAttention._forward`` on ROCm) is vLLM's generic Triton
``causal_conv1d_update`` -- the glm5next plugin imports it straight from
``vllm.model_executor.layers.mamba.ops.causal_conv1d`` with NO ``is_rocm()`` /
aiter branch (common/kda.py:23-25, call site :554), so the ROCm path is the SAME
portable Triton kernel the NVIDIA ``vllm_triton`` backend times, only built for
CDNA3. This runner therefore reuses the NVIDIA runner's operand build, invocation
and correctness check verbatim and only swaps the GPU gate (MI300X) and the timer
(rocprofv3 instead of CUPTI).

Timing matches the NVIDIA backend's ``Timer.cupti(kernel_name=_KERNEL_NAME)``: the
whole-process rocprofv3 trace is filtered to the one fused kernel
``_causal_conv1d_update_kernel``. Because that kernel ``@autotune``s on its first
launch -- a benchmark burst of extra dispatches that share its name -- the mean
is taken over the TRAILING ``rep`` matching dispatches (``dispatches_per_launch=1``),
which skips the burst. One fused decode launches exactly one such kernel, so D=1.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.runners.attention.gdn_causal_conv_decode_torch import (
    _logical_bytes,
    _semantic_flops,
)
from profiling.runners.attention.gdn_causal_conv_decode_vllm_triton import (
    _KERNEL_NAME,
    _build_operands,
    _check_correctness,
    _invoke_fused,
    _load_fused_callable,
    _validate_args,
)
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

_BACKEND = "gdn_causal_conv_decode:torch_rocm"
_KIND = "gdn_causal_conv_decode"
_GPU_SKU_TOKEN = "MI300X"
_WARMUP = 5
_REP = 20


def build_gdn_causal_conv_decode_kernel(
    batch_size: int,
    channels: int,
    kernel_size: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> dict[str, Any]:
    """Build the timed callable + operands for the AMD causal-conv decode call.

    Shared by the runner's correctness check and the rocprofv3 launch driver, so
    both replay the identical fused call from the same fixed seed. The trace is
    name-filtered to ``_causal_conv1d_update_kernel``, so the operand build's own
    setup launches are excluded and need no host->device trick.
    """
    args = _validate_args(batch_size, channels, kernel_size, dtype, state_dtype)
    import torch

    if getattr(torch.version, "hip", None) is None:
        raise ProfilerNotImplemented(f"{_BACKEND} requires a ROCm (HIP) torch build")
    fused_callable = _load_fused_callable()
    device = torch.device("cuda", torch.cuda.current_device())
    operands = _build_operands(torch, args, device=device)

    def kernel() -> Any:
        return _invoke_fused(fused_callable, operands)

    return {
        "torch": torch,
        "kernel": kernel,
        "callable": fused_callable,
        "operands": operands,
        "args": args,
        "warmup": _WARMUP,
        "rep": _REP,
    }


def profile_gdn_causal_conv_decode_torch_rocm(
    batch_size: int,
    channels: int,
    kernel_size: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> ComputeMetrics:
    """Profile the GDN/KDA causal-convolution decode on an MI300X."""
    args = _validate_args(batch_size, channels, kernel_size, dtype, state_dtype)
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

        built = build_gdn_causal_conv_decode_kernel(
            batch_size, channels, kernel_size, dtype, state_dtype
        )
        # Aliasing + semantics vs the naive reference, restoring the operands;
        # this is also the proof the AMD/ROCm Triton kernel runs correctly.
        _check_correctness(torch, built["callable"], built["operands"], args)

        time_ms = measure_registered_via_rocprofv3(
            kind=_KIND,
            backend="torch_rocm",
            spec={
                "batch_size": args.batch_size,
                "channels": args.channels,
                "kernel_size": args.kernel_size,
                "dtype": args.dtype.value,
                "state_dtype": args.state_dtype.value,
            },
            kernel_name_contains=_KERNEL_NAME,
            warmup=_WARMUP,
            rep=_REP,
            dispatches_per_launch=1,
        )
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        raise OOMError(f"{_BACKEND} ran out of device memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    flops = _semantic_flops(
        batch_size=args.batch_size, channels=args.channels, kernel_size=args.kernel_size
    )
    logical_bytes = _logical_bytes(
        batch_size=args.batch_size,
        channels=args.channels,
        kernel_size=args.kernel_size,
        dtype=args.dtype,
        state_dtype=args.state_dtype,
    )
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=0.0,
        tflops=flops / elapsed_s / 1e12 if time_ms > 0.0 else 0.0,
        memory_bandwidth_gbps=logical_bytes / elapsed_s / 1e9 if time_ms > 0.0 else 0.0,
    )


__all__ = [
    "build_gdn_causal_conv_decode_kernel",
    "profile_gdn_causal_conv_decode_torch_rocm",
]
