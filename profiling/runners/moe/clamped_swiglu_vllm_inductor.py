"""Profile vLLM Marlin's production clamped-SwiGLU callable."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.moe.clamped_swiglu_reference import (
    CLAMP_LIMIT,
    clamped_swiglu_reference,
)

_BACKEND = "clamped_swiglu:vllm_inductor"
_GPU_NAME = "NVIDIA H200"
_HIDDEN_DIM = 2048


@dataclass(frozen=True)
class _Launch:
    callable: Callable[..., None]
    output_tensor: Any
    input_tensor: Any

    def run(self) -> None:
        self.callable(self.output_tensor, self.input_tensor, CLAMP_LIMIT)


def _validate_args(
    num_rows: int, hidden_dim: int, dtype: DType | str
) -> tuple[int, DType]:
    if type(num_rows) is not int or num_rows <= 0:
        raise ValueError("num_rows must be a positive integer")
    if hidden_dim != _HIDDEN_DIM:
        raise ProfilerNotImplemented(f"{_BACKEND} requires hidden_dim={_HIDDEN_DIM}")
    resolved_dtype = DType.from_value(dtype)
    if resolved_dtype is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires dtype=bf16")
    return num_rows, resolved_dtype


def _require_h200(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _GPU_NAME:
        raise ProfilerNotImplemented(f"{_BACKEND} is verified only on {_GPU_NAME}, got {gpu_name}")


def _prepare(torch: Any, callable_: Callable[..., None], num_rows: int) -> _Launch:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(0)
    input_tensor = torch.randn(
        (num_rows, 2 * _HIDDEN_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    output_tensor = torch.empty(
        (num_rows, _HIDDEN_DIM), dtype=torch.bfloat16, device=device
    )
    return _Launch(callable_, output_tensor, input_tensor)


def _compile_and_check(torch: Any, launch: _Launch) -> None:
    for _ in range(3):
        launch.run()
    torch.cuda.synchronize()
    expected = clamped_swiglu_reference(torch, launch.input_tensor)
    torch.testing.assert_close(launch.output_tensor, expected, atol=0.02, rtol=0.02)


def profile_clamped_swiglu_vllm_inductor(
    num_rows: int,
    hidden_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    num_rows, _ = _validate_args(num_rows, hidden_dim, dtype)
    try:
        import torch
        from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM environment") from exc

    try:
        _require_h200(torch)
        launch = _prepare(torch, swiglu_limit_func, num_rows)
        _compile_and_check(torch, launch)
        time_ms = Timer.cupti(launch.run, kernel_name=None)
        energy_j = Energy.perf(launch.run, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    elapsed_seconds = time_ms / 1000.0
    logical_bytes = 6 * num_rows * _HIDDEN_DIM
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(
            logical_bytes / elapsed_seconds / 1e9 if elapsed_seconds else 0.0
        ),
        energy_j=float(energy_j),
    )


__all__ = ["profile_clamped_swiglu_vllm_inductor"]
