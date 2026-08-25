"""Profile the production vLLM routed-expert sum callable."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.moe.moe_sum_reference import moe_sum_reference

_BACKEND = "moe_sum:vllm_cuda"
_GPU_NAME = "NVIDIA H200"
_TOP_K = 6
_HIDDEN_DIM = 4096


@dataclass(frozen=True)
class _Launch:
    callable: Callable[[Any, Any], None]
    input_tensor: Any
    output_tensor: Any

    def run(self) -> None:
        self.callable(self.input_tensor, self.output_tensor)


def _validate_args(
    num_tokens: int,
    top_k: int,
    hidden_dim: int,
    dtype: DType | str,
) -> tuple[int, DType]:
    if type(num_tokens) is not int or num_tokens <= 0:
        raise ValueError("num_tokens must be a positive integer")
    if top_k != _TOP_K or hidden_dim != _HIDDEN_DIM:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires top_k={_TOP_K} and hidden_dim={_HIDDEN_DIM}"
        )
    resolved_dtype = DType.from_value(dtype)
    if resolved_dtype is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires dtype=bf16")
    return num_tokens, resolved_dtype


def _require_h200(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _GPU_NAME:
        raise ProfilerNotImplemented(f"{_BACKEND} is verified only on {_GPU_NAME}, got {gpu_name}")


def _prepare(torch: Any, callable_: Callable[[Any, Any], None], num_tokens: int) -> _Launch:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(0)
    input_tensor = torch.randn(
        (num_tokens, _TOP_K, _HIDDEN_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    output_tensor = torch.empty(
        (num_tokens, _HIDDEN_DIM), dtype=torch.bfloat16, device=device
    )
    return _Launch(callable_, input_tensor, output_tensor)


def _check_output(torch: Any, launch: _Launch) -> None:
    expected = moe_sum_reference(torch, launch.input_tensor)
    launch.run()
    torch.cuda.synchronize()
    torch.testing.assert_close(launch.output_tensor, expected, atol=0.02, rtol=0.02)


def profile_moe_sum_vllm_cuda(
    num_tokens: int,
    top_k: int,
    hidden_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    num_tokens, _ = _validate_args(num_tokens, top_k, hidden_dim, dtype)
    try:
        import torch
        from vllm import _custom_ops
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM environment") from exc

    try:
        _require_h200(torch)
        launch = _prepare(torch, _custom_ops.moe_sum, num_tokens)
        _check_output(torch, launch)
        # DeepSeek top-k 6 takes vLLM's at::sum_out fallback, not moe_sum_kernel.
        time_ms = Timer.cupti(launch.run, warmup=5, kernel_name="reduce_kernel")
        energy_j = Energy.perf(launch.run, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    elapsed_seconds = time_ms / 1000.0
    flops = num_tokens * _HIDDEN_DIM * (_TOP_K - 1)
    logical_bytes = 2 * num_tokens * _HIDDEN_DIM * (_TOP_K + 1)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / elapsed_seconds / 1e12 if elapsed_seconds else 0.0),
        memory_bandwidth_gbps=float(
            logical_bytes / elapsed_seconds / 1e9 if elapsed_seconds else 0.0
        ),
        energy_j=float(energy_j),
    )


__all__ = ["profile_moe_sum_vllm_cuda"]
