"""Profile vLLM's CUDA static per-tensor FP8 quant (``_C.static_scaled_fp8_quant``).

This is ``QuantFP8.forward_cuda`` with ``static=True`` and a per-tensor group
shape: one launch of ``vllm._custom_ops.scaled_fp8_quant(x, scale)`` over a
contiguous ``(num_tokens, hidden_dim)`` bf16 activation. The gated variant is
an Inductor fusion and has no CUDA counterpart, so this backend refuses it.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.elementwise.static_fp8_quant_reference import static_fp8_quant_reference
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "static_fp8_quant:vllm_cuda"
# A runtime scalar; its value never changes the kernel.
_FP8_SCALE = 1.0 / 16.0


def profile_static_fp8_quant_vllm_cuda(
    num_tokens: int, hidden_dim: int, gated: bool, dtype: DType | str
) -> ComputeMetrics:
    for name, value in (("num_tokens", num_tokens), ("hidden_dim", hidden_dim)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if gated:
        raise ProfilerNotImplemented(f"{_BACKEND} has no gated variant")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires dtype=bf16, got {dtype}")
    try:
        import torch
        from vllm import _custom_ops as vops
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM environment") from exc

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(0)
    scale = torch.tensor([_FP8_SCALE], dtype=torch.float32, device=device)
    try:
        values = torch.randn(
            (num_tokens, hidden_dim), dtype=torch.bfloat16, device=device, generator=generator
        )
        actual, _ = vops.scaled_fp8_quant(values, scale)
        torch.cuda.synchronize()
        expected = static_fp8_quant_reference(torch, values, None, scale)
        # One E4M3 step is 2^-3 relative: rounding order may move a value across it.
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0.13, atol=2e-2)

        def launch() -> None:
            vops.scaled_fp8_quant(values, scale)

        time_ms = Timer.cupti(launch, kernel_name=None)
        energy_j = Energy.perf(launch, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    logical_bytes = 3 * num_tokens * hidden_dim
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(logical_bytes / seconds / 1e9),
        energy_j=float(energy_j),
    )


__all__ = ["profile_static_fp8_quant_vllm_cuda"]
