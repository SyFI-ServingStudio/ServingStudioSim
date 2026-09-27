"""Profile the exact FP32-output ``torch.mm`` calls of DeepSeek V4 and GLM-5.3.

``bf16`` is ``torch.mm(x, weight.T, out_dtype=torch.float32)`` (DeepSeek V4
attention closures, GLM-5.3 router n=288). ``fp32`` is GLM-5.3's DSA indexer
head weights, ``torch.mm(x.float(), w)`` over a cached contiguous ``[k, n]``
FP32 weight; the ``x.float()`` cast is a separate launch outside this boundary.
"""

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gemm_fp32_output:torch_cublas"
_GPU_NAMES = frozenset({"NVIDIA H200", "NVIDIA B200"})
_K = 4096
_SUPPORTED_N = {
    DType.BF16: frozenset({256, 288, 512, 1024, 2048}),
    DType.FP32: frozenset({32}),
}


@dataclass(frozen=True)
class _Shape:
    m: int
    n: int
    input_dtype: DType


@dataclass(frozen=True)
class _Launch:
    torch: Any
    input_tensor: Any
    weight_transposed: Any
    fp32_input: bool = False

    def run(self):
        if self.fp32_input:
            return self.torch.mm(self.input_tensor, self.weight_transposed)
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
    dtype = DType.from_value(input_dtype)
    supported_n = sorted(_SUPPORTED_N.get(dtype, ()))
    if n not in supported_n or k != _K:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires k={_K} and n in {supported_n} for {dtype.value}"
        )
    return _Shape(m, n, dtype)


def _require_gpu(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name not in _GPU_NAMES:
        raise ProfilerNotImplemented(f"{_BACKEND} is verified only on {_GPU_NAMES}, got {gpu_name}")


def _prepare(torch: Any, shape: _Shape) -> _Launch:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(31)
    fp32 = shape.input_dtype is DType.FP32
    if fp32 and torch.get_float32_matmul_precision() != "highest":
        raise ProfilerNotImplemented(f"{_BACKEND} fp32 requires matmul precision 'highest'")
    dtype = torch.float32 if fp32 else torch.bfloat16
    input_tensor = torch.randn((shape.m, _K), dtype=dtype, device=device, generator=generator)
    weight = torch.randn((shape.n, _K), dtype=dtype, device=device, generator=generator)
    return _Launch(torch, input_tensor, weight.T.contiguous() if fp32 else weight.T, fp32)


def _check_output(torch: Any, launch: _Launch) -> None:
    expected = launch.input_tensor.cpu().float() @ launch.weight_transposed.cpu().float()
    actual = launch.run()
    torch.cuda.synchronize()
    if actual.dtype is not torch.float32:
        raise AssertionError(f"expected FP32 output, got {actual.dtype}")
    torch.testing.assert_close(actual.cpu(), expected, atol=0.002, rtol=0.01)


def _logical_bytes(shape: _Shape) -> int:
    width = int(shape.input_dtype.size_bytes())
    return width * shape.m * _K + width * shape.n * _K + 4 * shape.m * shape.n


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
        _require_gpu(torch)
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
