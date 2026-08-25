"""Profile the exact FP32-output ``torch.mm`` used by DeepSeek V4."""

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gemm_fp32_output:torch_cublas"
_GPU_NAME = "NVIDIA H200"
_K = 4096
_SUPPORTED_N = frozenset({256, 512, 1024, 2048})


@dataclass(frozen=True)
class _Shape:
    m: int
    n: int


@dataclass(frozen=True)
class _Launch:
    torch: Any
    input_tensor: Any
    weight_transposed: Any

    def run(self):
        # Keep the allocation and argument form identical to the production
        # closures in vllm.models.deepseek_v4.attention.
        return self.torch.mm(
            self.input_tensor,
            self.weight_transposed,
            out_dtype=self.torch.float32,
        )


def _validate_args(
    m: int,
    n: int,
    k: int,
    input_dtype: DType | str,
) -> _Shape:
    if type(m) is not int or m <= 0:
        raise ValueError("m must be a positive integer")
    if n not in _SUPPORTED_N or k != _K:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires n in {sorted(_SUPPORTED_N)} and k={_K}"
        )
    if DType.from_value(input_dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires input_dtype=bf16")
    return _Shape(m, n)


def _require_h200(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _GPU_NAME:
        raise ProfilerNotImplemented(f"{_BACKEND} is verified only on {_GPU_NAME}, got {gpu_name}")


def _prepare(torch: Any, shape: _Shape) -> _Launch:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(31)
    input_tensor = torch.randn(
        (shape.m, _K), dtype=torch.bfloat16, device=device, generator=generator
    )
    weight = torch.randn(
        (shape.n, _K), dtype=torch.bfloat16, device=device, generator=generator
    )
    return _Launch(torch, input_tensor, weight.T)


def _check_output(torch: Any, launch: _Launch) -> None:
    expected = launch.input_tensor.cpu().float() @ launch.weight_transposed.cpu().float()
    actual = launch.run()
    torch.cuda.synchronize()
    if actual.dtype is not torch.float32:
        raise AssertionError(f"expected FP32 output, got {actual.dtype}")
    torch.testing.assert_close(actual.cpu(), expected, atol=0.002, rtol=0.01)


def _logical_bytes(shape: _Shape) -> int:
    return 2 * shape.m * _K + 2 * shape.n * _K + 4 * shape.m * shape.n


def profile_gemm_fp32_output_torch_cublas(
    m: int,
    n: int,
    k: int,
    input_dtype: DType | str,
) -> ComputeMetrics:
    shape = _validate_args(m, n, k, input_dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires Torch") from exc

    try:
        _require_h200(torch)
        launch = _prepare(torch, shape)
        _check_output(torch, launch)
        # A logical call may be one NvJet kernel or an ordered split-K pair.
        time_ms = Timer.cupti(launch.run, warmup=5, kernel_name=None)
        energy_j = Energy.perf(launch.run, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    elapsed_seconds = time_ms / 1000.0
    flops = 2 * shape.m * shape.n * _K
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / elapsed_seconds / 1e12 if elapsed_seconds else 0.0),
        memory_bandwidth_gbps=float(
            _logical_bytes(shape) / elapsed_seconds / 1e9 if elapsed_seconds else 0.0
        ),
        energy_j=float(energy_j),
    )


__all__ = ["profile_gemm_fp32_output_torch_cublas"]
