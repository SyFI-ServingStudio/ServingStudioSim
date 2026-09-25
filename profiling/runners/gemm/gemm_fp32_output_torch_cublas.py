"""Profile the exact FP32-output ``torch.mm`` calls of DeepSeek V4 and GLM-5.3.

Two production forms share this runner, selected by ``input_dtype``:

- ``bf16``: ``torch.mm(x, weight.T, out_dtype=torch.float32)`` over a
  ``[n, k]`` BF16 weight. DeepSeek V4 attention closures and GLM-5.3 MoE
  ``GateLinear`` Tier 4 (router, n=288).
- ``fp32``: ``torch.mm(x.float(), w)`` over a cached contiguous ``[k, n]`` FP32
  weight (GLM-5.3 DSA indexer head weights, ``_wp_fp32``, n=32). The
  ``x.float()`` cast is a separate elementwise launch, outside this boundary.
  vLLM keeps ``VLLM_FLOAT32_MATMUL_PRECISION=highest``, so this is true FP32.

``torch_cublas`` runs in the profiler container and covers ``bf16`` only. Its
bundled cuBLAS picks a non-split-K SIMT SGEMM for the skinny FP32 form on B200
(about 4x slower than production), so ``fp32`` is profiled only through
``torch_cublas_vllm_fork``. That backend runs on the GLM-5.3 serving stack,
whose cuBLAS emits the ``Kernel2`` SGEMM plus ``splitKreduce`` pair seen in the
capture.
"""

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gemm_fp32_output:torch_cublas"
_SUPPORTED_GPUS = frozenset({"NVIDIA H200", "NVIDIA B200"})
_SUPPORTED_DTYPES = frozenset({DType.BF16})
_FORK_BACKEND = "gemm_fp32_output:torch_cublas_vllm_fork"
_FORK_SUPPORTED_GPUS = frozenset({"NVIDIA B200"})
_FORK_SUPPORTED_DTYPES = frozenset({DType.BF16, DType.FP32})
_K = 4096
# Verified production widths per input dtype.
_SUPPORTED_N = {
    DType.BF16: frozenset({256, 288, 512, 1024, 2048}),
    DType.FP32: frozenset({32}),
}


@dataclass(frozen=True)
class _Shape:
    m: int
    n: int
    input_dtype: DType = DType.BF16


@dataclass(frozen=True)
class _Launch:
    torch: Any
    input_tensor: Any
    weight_transposed: Any
    fp32_input: bool = False

    def run(self):
        if self.fp32_input:
            # GLM-5.3 indexer: torch.mm(hidden_states.float(), self._wp_fp32).
            return self.torch.mm(self.input_tensor, self.weight_transposed)
        # Keep the allocation and argument form identical to the production
        # closures in vllm.models.deepseek_v4.attention and GateLinear Tier 4.
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
    *,
    backend: str = _BACKEND,
    supported_dtypes: frozenset[DType] = _SUPPORTED_DTYPES,
) -> _Shape:
    if type(m) is not int or m <= 0:
        raise ValueError("m must be a positive integer")
    dtype = DType.from_value(input_dtype)
    if dtype not in supported_dtypes:
        names = ", ".join(sorted(d.value for d in supported_dtypes))
        raise ProfilerNotImplemented(f"{backend} requires input_dtype in {{{names}}}")
    if n not in _SUPPORTED_N[dtype] or k != _K:
        raise ProfilerNotImplemented(
            f"{backend} requires n in {sorted(_SUPPORTED_N[dtype])} for "
            f"input_dtype={dtype.value} and k={_K}"
        )
    return _Shape(m, n, dtype)


def _require_supported_gpu(
    torch: Any,
    *,
    backend: str = _BACKEND,
    supported_gpus: frozenset[str] = _SUPPORTED_GPUS,
) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{backend} requires CUDA")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name not in supported_gpus:
        raise ProfilerNotImplemented(
            f"{backend} is verified only on {sorted(supported_gpus)}, got {gpu_name}"
        )


def _prepare(torch: Any, shape: _Shape) -> _Launch:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(31)
    if shape.input_dtype is DType.FP32:
        if torch.get_float32_matmul_precision() != "highest":
            raise ProfilerNotImplemented(f"{_FORK_BACKEND} fp32 requires matmul precision 'highest'")
        input_tensor = torch.randn(
            (shape.m, _K), dtype=torch.float32, device=device, generator=generator
        )
        weight = torch.randn(
            (shape.n, _K), dtype=torch.float32, device=device, generator=generator
        )
        # Production caches weight[head_dim:, :].t().contiguous().float().
        return _Launch(torch, input_tensor, weight.t().contiguous(), fp32_input=True)
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
    width = shape.input_dtype.size_bytes()
    return width * shape.m * _K + width * shape.n * _K + 4 * shape.m * shape.n


def profile_gemm_fp32_output_torch_cublas(
    m: int,
    n: int,
    k: int,
    input_dtype: DType | str,
) -> ComputeMetrics:
    shape = _validate_args(m, n, k, input_dtype)
    return _profile(shape, backend=_BACKEND, supported_gpus=_SUPPORTED_GPUS)


def profile_gemm_fp32_output_torch_cublas_vllm_fork(
    m: int,
    n: int,
    k: int,
    input_dtype: DType | str,
) -> ComputeMetrics:
    shape = _validate_args(
        m,
        n,
        k,
        input_dtype,
        backend=_FORK_BACKEND,
        supported_dtypes=_FORK_SUPPORTED_DTYPES,
    )
    return _profile(shape, backend=_FORK_BACKEND, supported_gpus=_FORK_SUPPORTED_GPUS)


def _profile(shape: _Shape, *, backend: str, supported_gpus: frozenset[str]) -> ComputeMetrics:
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{backend} requires Torch") from exc

    try:
        _require_supported_gpu(torch, backend=backend, supported_gpus=supported_gpus)
        launch = _prepare(torch, shape)
        _check_output(torch, launch)
        # A logical call may be one NvJet kernel or an ordered split-K pair.
        time_ms = Timer.cupti(launch.run, warmup=5, kernel_name=None)
        energy_j = Energy.perf(launch.run, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{backend} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{backend} failed") from exc

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


__all__ = [
    "profile_gemm_fp32_output_torch_cublas",
    "profile_gemm_fp32_output_torch_cublas_vllm_fork",
]
