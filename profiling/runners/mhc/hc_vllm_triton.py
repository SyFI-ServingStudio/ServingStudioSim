"""Profile the gated hyper-connection (HC) Triton custom ops of the pinned vLLM fork.

Each entry times one public op from ``vllm/models/qwen4_exp/nvidia/ops/hc.py``
(fork 892da082) through its ``torch.ops.vllm`` registration, as
``hyperconnection.py`` calls it. Every op is one Triton launch; CUPTI counts every
launch of the call. Inputs are seeded bf16 tensors in the production layout: the
residual is ``[tokens, hc_mult * hidden_size]`` stream-outer, the block output is
``[tokens, hidden_size]``, the norm weight is per stream (``[hc_mult * hidden_size]``,
the checkpoint's ``hc_per_branch_norm`` layout) and the injection logits are
``[tokens, hc_mult]``. Each call is checked against ``hc_reference`` before timing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.mhc import hc_reference as ref

# eps is a constexpr scalar of the launch; its value does not change the time.
_EPS = 1e-6
_BF16 = 2
_TOL = {"rtol": 2e-2, "atol": 2e-2}


def _validate(backend: str, hidden_dtype: DType | str, **dims: int) -> None:
    for name, value in dims.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if DType.from_value(hidden_dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{backend} requires hidden_dtype=bf16, got {hidden_dtype}")


def _import(backend: str) -> tuple[Any, Any]:
    try:
        import torch
        from vllm.models.qwen4_exp.nvidia.ops import hc as hc_ops
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{backend} requires the pinned vLLM fork environment"
        ) from exc
    return torch, hc_ops


def _measure(
    backend: str,
    build: Callable[[Any, Any, Callable[..., Any]], Callable[[], Any]],
    logical_bytes: int,
) -> ComputeMetrics:
    torch, hc_ops = _import(backend)
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(0)

        def randn(*shape: int, scale: float = 1.0) -> Any:
            values = torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
            return (scale * values).to(torch.bfloat16)

        launch = build(torch, hc_ops, randn)
        time_ms = Timer.cupti(launch, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{backend} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, AssertionError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{backend} failed: {exc}") from exc
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(logical_bytes / seconds / 1e9) if seconds > 0 else 0.0,
        energy_j=float(energy_j),
    )


def profile_grouped_rms_norm_vllm_triton(
    num_tokens: int, hidden_size: int, num_groups: int, hidden_dtype: DType | str
) -> ComputeMetrics:
    """``grouped_gemma_rmsnorm``: the HC mix whose residual has no pending combine."""
    backend = "grouped_rms_norm:vllm_triton"
    _validate(
        backend, hidden_dtype, num_tokens=num_tokens, hidden_size=hidden_size, num_groups=num_groups
    )
    width = num_groups * hidden_size

    def build(torch: Any, hc_ops: Any, randn: Callable[..., Any]) -> Callable[[], Any]:
        x, weight = randn(num_tokens, width), randn(width, scale=0.1)
        torch.testing.assert_close(
            hc_ops.grouped_gemma_rmsnorm(x, weight, _EPS, num_groups),
            ref.grouped_rms_norm_reference(torch, x, weight, _EPS, num_groups),
            **_TOL,
        )
        return lambda: hc_ops.grouped_gemma_rmsnorm(x, weight, _EPS, num_groups)

    return _measure(backend, build, _BF16 * (2 * num_tokens * width + width))


def profile_hc_combine_vllm_triton(
    num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: DType | str
) -> ComputeMetrics:
    """``hc_combine`` with learned injection logits."""
    backend = "hc_combine:vllm_triton"
    _validate(
        backend, hidden_dtype, num_tokens=num_tokens, hidden_size=hidden_size, hc_mult=hc_mult
    )
    width = hc_mult * hidden_size

    def build(torch: Any, hc_ops: Any, randn: Callable[..., Any]) -> Callable[[], Any]:
        residual, block = randn(num_tokens, width), randn(num_tokens, hidden_size)
        injection = randn(num_tokens, hc_mult)
        torch.testing.assert_close(
            hc_ops.hc_combine(residual, block, injection, hc_mult),
            ref.hc_combine_reference(torch, residual, block, injection, hc_mult),
            **_TOL,
        )
        return lambda: hc_ops.hc_combine(residual, block, injection, hc_mult)

    return _measure(backend, build, _BF16 * num_tokens * (2 * width + hidden_size + hc_mult))


def profile_hc_combine_norm_vllm_triton(
    num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: DType | str
) -> ComputeMetrics:
    """``hc_combine_norm`` with learned injection logits."""
    backend = "hc_combine_norm:vllm_triton"
    _validate(
        backend, hidden_dtype, num_tokens=num_tokens, hidden_size=hidden_size, hc_mult=hc_mult
    )
    width = hc_mult * hidden_size

    def build(torch: Any, hc_ops: Any, randn: Callable[..., Any]) -> Callable[[], Any]:
        residual, block = randn(num_tokens, width), randn(num_tokens, hidden_size)
        injection, weight = randn(num_tokens, hc_mult), randn(width, scale=0.1)
        actual = hc_ops.hc_combine_norm(residual, block, injection, weight, _EPS, hc_mult)
        expected = ref.hc_combine_norm_reference(
            torch, residual, block, injection, weight, _EPS, hc_mult
        )
        for actual_plane, expected_plane in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_plane, expected_plane, **_TOL)
        return lambda: hc_ops.hc_combine_norm(residual, block, injection, weight, _EPS, hc_mult)

    # Read residual, block, logits and weight; write the combined and normed streams.
    logical_bytes = _BF16 * (num_tokens * (3 * width + hidden_size + hc_mult) + width)
    return _measure(backend, build, logical_bytes)


def profile_hc_gate_mix_vllm_triton(
    num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: DType | str
) -> ComputeMetrics:
    """``hc_gate_mix``: the block input from the normed streams and the up-projected gate."""
    backend = "hc_gate_mix:vllm_triton"
    _validate(
        backend, hidden_dtype, num_tokens=num_tokens, hidden_size=hidden_size, hc_mult=hc_mult
    )
    width = hc_mult * hidden_size

    def build(torch: Any, hc_ops: Any, randn: Callable[..., Any]) -> Callable[[], Any]:
        x, gate = randn(num_tokens, width), randn(num_tokens, width)
        torch.testing.assert_close(
            hc_ops.hc_gate_mix(x, gate, hc_mult),
            ref.hc_gate_mix_reference(torch, x, gate, hc_mult),
            **_TOL,
        )
        return lambda: hc_ops.hc_gate_mix(x, gate, hc_mult)

    return _measure(backend, build, _BF16 * num_tokens * (2 * width + hidden_size))


def profile_scaled_silu_vllm_triton(
    num_tokens: int, width: int, hidden_dtype: DType | str
) -> ComputeMetrics:
    """``hc_silu``: SiLU of the HC low-rank projection divided by the stream count.

    The divisor is a constexpr scalar of the launch; 4 is the deployed stream count.
    """
    backend = "scaled_silu:vllm_triton"
    _validate(backend, hidden_dtype, num_tokens=num_tokens, width=width)
    divisor = 4

    def build(torch: Any, hc_ops: Any, randn: Callable[..., Any]) -> Callable[[], Any]:
        x = randn(num_tokens, width)
        torch.testing.assert_close(
            hc_ops.hc_silu(x, divisor), ref.scaled_silu_reference(torch, x, divisor), **_TOL
        )
        return lambda: hc_ops.hc_silu(x, divisor)

    return _measure(backend, build, _BF16 * 2 * num_tokens * width)


__all__ = [
    "profile_grouped_rms_norm_vllm_triton",
    "profile_hc_combine_norm_vllm_triton",
    "profile_hc_combine_vllm_triton",
    "profile_hc_gate_mix_vllm_triton",
    "profile_scaled_silu_vllm_triton",
]
