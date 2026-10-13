"""Profile the PLE similarity gate (``ple_gate``).

``vllm_triton`` times the fork's ``ple_gate`` custom op
(``models/qwen4_exp/nvidia/ops/ple.py``) with key and value as the strided
halves of one ``[tokens, (hc_mult + 1) * H]`` projection output, exactly as
``Qwen4ExpPLELayer.forward`` splits ``kv_proj``'s result.

``reference_gate`` is the eager PyTorch math (``Qwen4ExpPLEGroupedNorm`` and
the gate formula), rounding to the activation dtype at the same tensor
boundaries the kernel mirrors. Reductions run in a different order, so the
comparison uses a tolerance.
"""

from __future__ import annotations

from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

KIND = "ple_gate"
_KERNEL_NAME = "_ple_gate_kernel"
_EPS = 1e-6


def validate_args(num_tokens: int, hidden_size: int, hc_mult: int, dtype: object) -> None:
    # No upper bounds: the grid is (tokens, hc_mult) and one program covers a
    # whole hidden_size stream, so the kernel itself sets no size limit.
    for name, value in (
        ("num_tokens", num_tokens),
        ("hidden_size", hidden_size),
        ("hc_mult", hc_mult),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if str(dtype) != "bf16":
        raise ProfilerNotImplemented(f"{KIND} is profiled for bf16 activations")


def logical_bytes(num_tokens: int, hidden_size: int, hc_mult: int, elem: int = 2) -> int:
    """key + hidden + value read, gated + normed written, three norm weights."""
    width = hc_mult * hidden_size
    return (num_tokens * (5 * width + hidden_size) + 3 * width) * elem


def _grouped_norm(torch: Any, x: Any, weight: Any, group: int) -> Any:
    xf = x.float().unflatten(-1, (-1, group))
    normed = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + _EPS)).flatten(-2)
    return (normed * (1.0 + weight.float())).to(x.dtype)


def reference_gate(torch: Any, key, value, hidden, nk, nq, ncw, hidden_size: int):
    """Eager PLE gate: returns (gated, normed), both [tokens, hc_mult * H]."""
    dtype = hidden.dtype
    k_n = _grouped_norm(torch, key, nk, hidden_size).unflatten(-1, (-1, hidden_size))
    q_n = _grouped_norm(torch, hidden, nq, hidden_size).unflatten(-1, (-1, hidden_size))
    dot = (k_n * q_n).float().sum(-1).to(dtype).float()
    d = (dot / hidden_size**0.5).to(dtype).float()
    gate = torch.sigmoid(torch.sign(d) * d.abs().clamp_min(1e-6).sqrt().to(dtype).float())
    gate = gate.to(dtype)
    gated = (gate[..., None].float() * value[:, None, :].float()).to(dtype).flatten(-2)
    return gated, _grouped_norm(torch, gated, ncw, hidden_size)


def _inputs(torch: Any, device: Any, num_tokens: int, hidden_size: int, hc_mult: int):
    gen = torch.Generator(device=device).manual_seed(0x61E)
    width = hc_mult * hidden_size
    dtype = torch.bfloat16
    kv = torch.randn(num_tokens, width + hidden_size, device=device, generator=gen).to(dtype)
    key, value = kv.split([width, hidden_size], dim=-1)
    hidden = torch.randn(num_tokens, width, device=device, generator=gen).to(dtype)
    weights = [
        (0.1 * torch.randn(width, device=device, generator=gen)).to(dtype) for _ in range(3)
    ]
    return key, value, hidden, weights


def check_gate(torch: Any, got: tuple, ref: tuple) -> None:
    for name, g, r in zip(("gated", "normed"), got, ref):
        bad = ~torch.isclose(g.float(), r.float(), rtol=5e-2, atol=5e-2)
        frac = float(bad.float().mean())
        if frac > 1e-3:
            raise KernelLaunchFailed(f"{KIND} {name} differs from the reference on {frac:.2%}")


def profile_ple_gate_vllm_triton(
    num_tokens: int, hidden_size: int, hc_mult: int, dtype: object
) -> ComputeMetrics:
    """Time one fused PLE gate launch."""
    validate_args(num_tokens, hidden_size, hc_mult, dtype)
    try:
        import torch
        from vllm.models.qwen4_exp.nvidia.ops.ple import ple_gate
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{KIND}:vllm_triton requires the vLLM fork") from exc
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        key, value, hidden, (nk, nq, ncw) = _inputs(torch, device, num_tokens, hidden_size, hc_mult)

        def launch() -> Any:
            return ple_gate(key, value, hidden, nk, nq, ncw, _EPS)

        got = launch()
        torch.cuda.synchronize()
        check_gate(torch, got, reference_gate(torch, key, value, hidden, nk, nq, ncw, hidden_size))
        time_ms = Timer.cupti(launch, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(launch, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{KIND}:vllm_triton ran out of memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{KIND}:vllm_triton failed: {exc}") from exc
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(
            logical_bytes(num_tokens, hidden_size, hc_mult) / (time_ms / 1000.0) / 1e9
        ),
        energy_j=float(energy_j),
    )


__all__ = ["logical_bytes", "profile_ple_gate_vllm_triton", "reference_gate", "validate_args"]
