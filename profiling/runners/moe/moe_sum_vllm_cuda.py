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
) -> tuple[int, int, int]:
    # vLLM's moe_sum takes any top_k and hidden_dim: templated kernels for
    # small top_k, a dynamic-top_k kernel otherwise, and a scalar kernel when
    # hidden_dim is not a whole number of vectors.
    for name, value in (("num_tokens", num_tokens), ("top_k", top_k), ("hidden_dim", hidden_dim)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires dtype=bf16")
    return num_tokens, top_k, hidden_dim


def _require_cuda(torch: Any) -> None:
    # A vLLM _C CUDA op built for every CUDA arch the wheel targets; no GPU
    # model or capability floor beyond CUDA itself.
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")


def _prepare(
    torch: Any,
    callable_: Callable[[Any, Any], None],
    num_tokens: int,
    top_k: int,
    hidden_dim: int,
) -> _Launch:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(0)
    input_tensor = torch.randn(
        (num_tokens, top_k, hidden_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    output_tensor = torch.empty((num_tokens, hidden_dim), dtype=torch.bfloat16, device=device)
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
    num_tokens, top_k, hidden_dim = _validate_args(num_tokens, top_k, hidden_dim, dtype)
    try:
        import torch
        from vllm import _custom_ops
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM environment") from exc

    try:
        _require_cuda(torch)
        launch = _prepare(torch, _custom_ops.moe_sum, num_tokens, top_k, hidden_dim)
        _check_output(torch, launch)
        # The kernel moe_sum launches depends on top_k and the vLLM build (older
        # builds fell back to torch's reduce_kernel at top_k 6); each path is
        # one launch, so count every launch of the call.
        time_ms = Timer.cupti(launch.run, warmup=5, kernel_name=None)
        energy_j = Energy.perf(launch.run, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    elapsed_seconds = time_ms / 1000.0
    flops = num_tokens * hidden_dim * (top_k - 1)
    logical_bytes = 2 * num_tokens * hidden_dim * (top_k + 1)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / elapsed_seconds / 1e12 if elapsed_seconds else 0.0),
        memory_bandwidth_gbps=float(
            logical_bytes / elapsed_seconds / 1e9 if elapsed_seconds else 0.0
        ),
        energy_j=float(energy_j),
    )


__all__ = ["profile_moe_sum_vllm_cuda"]
