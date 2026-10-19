"""Profile vLLM's native SiLU-and-mul as Inductor compiles it into the served graph."""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.elementwise.silu_and_mul_reference import silu_and_mul_reference
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.vllm_inductor import compile_like_vllm

_BACKEND = "silu_and_mul:vllm_inductor"


def profile_silu_and_mul_vllm_inductor(
    num_tokens: int, hidden_dim: int, dtype: DType | str
) -> ComputeMetrics:
    for name, value in (("num_tokens", num_tokens), ("hidden_dim", hidden_dim)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires dtype=bf16, got {dtype}")
    try:
        import torch
        from vllm.model_executor.layers.activation import SiluAndMul
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM environment") from exc

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(0)

    def make_inputs(tokens: int) -> list[Any]:
        return [
            torch.randn(
                (tokens, 2 * hidden_dim), dtype=torch.bfloat16, device=device, generator=generator
            )
        ]

    try:
        compiled = compile_like_vllm(
            torch, SiluAndMul.forward_native, make_inputs, {0: 0}, num_tokens
        )
        inputs = make_inputs(num_tokens)
        actual = compiled(*inputs)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            actual, silu_and_mul_reference(torch, inputs[0]), rtol=2e-2, atol=2e-2
        )

        def launch() -> None:
            compiled(*inputs)

        time_ms = Timer.cupti(launch, kernel_name=None)
        energy_j = Energy.perf(launch, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(6 * num_tokens * hidden_dim / seconds / 1e9),
        energy_j=float(energy_j),
    )


__all__ = ["profile_silu_and_mul_vllm_inductor"]
