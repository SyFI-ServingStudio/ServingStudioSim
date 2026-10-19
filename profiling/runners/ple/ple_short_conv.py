"""Profile the PLE dilated short convolution (``ple_short_conv``).

``vllm_triton`` times one call of the fork's ``ple_conv``
(``models/qwen4_exp/nvidia/ops/ple.py``) in the requested mode, with every
launch it issues counted. The state pool uses production's default SD layout
(``[slots, window, channels]``) viewed as ``[slot, channel, window]``, as
``Qwen4ExpPLELayer._short_conv`` canonicalizes it. Its window is the
``(K - 1) * D + spec`` positions the call needs (``spec = query_len - 1`` in
``spec`` mode, else 0). Production sizes every mode's window with the
deployment's ``num_spec_tokens``, so an MTP deployment's decode and prefill
calls see a wider window; that only spreads the slots further apart (the
kernel reads and writes the same positions), so the runner does not model it.
Slot 0 is the null block; request ``r`` owns slot ``r + 1``.

``reference_conv`` and ``reference_state`` recompute the output rows and the
updated state windows in PyTorch from the state and inputs before the call. The
in-place residual accumulates across timed calls, which does not change the
work.
"""

from __future__ import annotations

from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

KIND = "ple_short_conv"
_MODES = ("decode", "spec", "prefill")


def validate_args(
    batch_size: int,
    query_len: int,
    channels: int,
    kernel_size: int,
    dilation: int,
    mode: str,
    dtype: object,
) -> None:
    for name, value in (
        ("batch_size", batch_size),
        ("query_len", query_len),
        ("channels", channels),
        ("kernel_size", kernel_size),
        ("dilation", dilation),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}")
    if mode == "decode" and query_len != 1:
        raise ValueError("decode mode has query_len 1")
    if kernel_size < 2:
        raise ProfilerNotImplemented(f"{KIND} needs a state (kernel_size >= 2)")
    if str(dtype) != "bf16":
        raise ProfilerNotImplemented(f"{KIND} is profiled for bf16")


def logical_bytes(batch_size: int, query_len: int, channels: int, kernel_size: int,
                  dilation: int, elem: int = 2) -> int:
    """x + residual + outer read, residual written, state read and rewritten."""
    tokens = batch_size * query_len
    state = batch_size * channels * (kernel_size - 1) * dilation
    return (4 * tokens * channels + 2 * state) * elem


def reference_conv(torch: Any, x, residual, outer, state, weight, *, query_len: int,
                   dilation: int, mode: str, num_accepted=None) -> Any:
    """New residual rows. ``state`` is [batch, channels, window] for requests in order."""
    batch = state.shape[0]
    channels, taps = weight.shape
    state_len = (taps - 1) * dilation
    xs = x.view(batch, query_len, channels).float()
    if mode == "spec":
        off = (num_accepted.long() - 1).clamp(0, query_len - 1)
        rows = off[:, None] + torch.arange(state_len, device=x.device)[None, :]
        hist = state.float().gather(2, rows[:, None, :].expand(-1, channels, -1))
    else:
        hist = state.float()[..., :state_len]
    full = torch.cat([hist.transpose(1, 2), xs], dim=1)  # [batch, state_len + q, C]
    acc = torch.zeros(batch, query_len, channels, device=x.device)
    for k in range(taps):
        acc += weight[:, k].float() * full[:, dilation * k : dilation * k + query_len]
    conv = acc.to(x.dtype).float()
    y = (conv * torch.sigmoid(conv)).to(x.dtype)
    new = (residual.view(batch, query_len, channels) + y).to(x.dtype)
    return (outer.view(batch, query_len, channels).float() + new.float()).to(x.dtype).view_as(x)


def reference_state(torch: Any, x, state, *, query_len: int, state_len: int, mode: str,
                    num_accepted=None) -> Any:
    """Updated state windows, [batch, channels, window], for requests in order.

    Decode and prefill keep the last ``state_len`` inputs of the history plus
    this call's tokens. Spec rolls back to the last accepted token and keeps
    ``state_len + query_len - 1`` positions, one past that token's history.
    """
    batch, channels = state.shape[:2]
    xs = x.view(batch, query_len, channels).transpose(1, 2)
    if mode == "spec":
        off = (num_accepted.long() - 1).clamp(0, query_len - 1)
        rows = off[:, None] + torch.arange(state_len, device=x.device)[None, :]
        hist = state.gather(2, rows[:, None, :].expand(-1, channels, -1))
        return torch.cat([hist, xs], dim=2)[..., 1:]
    return torch.cat([state[..., :state_len], xs], dim=2)[..., -state_len:]


def profile_ple_short_conv_vllm_triton(
    batch_size: int,
    query_len: int,
    channels: int,
    kernel_size: int,
    dilation: int,
    mode: str,
    dtype: object,
) -> ComputeMetrics:
    """Time one ``ple_conv`` call (all of its launches)."""
    validate_args(batch_size, query_len, channels, kernel_size, dilation, mode, dtype)
    try:
        import torch
        from vllm.models.qwen4_exp.nvidia.ops.ple import ple_conv
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{KIND}:vllm_triton requires the vLLM fork") from exc
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        gen = torch.Generator(device=device).manual_seed(0xC0)
        bf16 = torch.bfloat16
        tokens = batch_size * query_len
        spec = query_len - 1 if mode == "spec" else 0
        state_len = (kernel_size - 1) * dilation
        capacity = state_len + spec
        raw = torch.randn(batch_size + 1, capacity, channels, device=device, generator=gen).to(bf16)
        conv_state = raw.transpose(-1, -2)[..., -capacity:]
        weight = (0.5 * torch.randn(channels, kernel_size, device=device, generator=gen)).to(bf16)
        x = torch.randn(tokens, channels, device=device, generator=gen).to(bf16)
        residual = torch.randn(tokens, channels, device=device, generator=gen).to(bf16)
        outer = torch.randn(tokens, channels, device=device, generator=gen).to(bf16)
        idx32 = dict(dtype=torch.int32, device=device)
        state_indices = torch.arange(1, batch_size + 1, **idx32)
        qsl = torch.arange(0, tokens + 1, query_len, **idx32)
        has_init = torch.ones(batch_size, dtype=torch.bool, device=device)
        num_accepted = (
            torch.randint(1, query_len + 1, (batch_size,), device=device, generator=gen).to(
                torch.int32
            )
            if mode == "spec"
            else None
        )
        kwargs: dict[str, Any] = dict(mode=mode, dilation=dilation)
        if mode == "decode":
            kwargs.update(has_initial_states=has_init)
        elif mode == "spec":
            kwargs.update(
                query_start_loc=qsl, num_accepted_tokens=num_accepted, spec_query_len=query_len
            )
        else:
            kwargs.update(query_start_loc=qsl, has_initial_states=has_init)

        def launch() -> None:
            ple_conv(x, residual, conv_state, weight, state_indices, outer, **kwargs)

        state_before = conv_state[1:].clone()
        ref = reference_conv(
            torch, x, residual, outer, state_before, weight,
            query_len=query_len, dilation=dilation, mode=mode, num_accepted=num_accepted,
        )
        ref_state = reference_state(
            torch, x, state_before, query_len=query_len, state_len=state_len, mode=mode,
            num_accepted=num_accepted,
        )
        launch()
        torch.cuda.synchronize()
        bad = ~torch.isclose(residual.float(), ref.float(), rtol=2e-2, atol=2e-2)
        if float(bad.float().mean()) > 1e-3:
            raise KernelLaunchFailed(
                f"{KIND} output differs from the reference on {float(bad.float().mean()):.2%}"
            )
        # The state update is a pure copy, so it must match exactly.
        if not torch.equal(conv_state[1:, :, : ref_state.shape[-1]], ref_state):
            raise KernelLaunchFailed(f"{KIND} conv_state differs from the reference update")
        time_ms = Timer.cupti(launch, kernel_name=None)
        energy_j = Energy.perf(launch, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{KIND}:vllm_triton ran out of memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{KIND}:vllm_triton failed: {exc}") from exc
    flops = tokens * channels * (2 * kernel_size + 6)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / (time_ms / 1000.0) / 1e12),
        memory_bandwidth_gbps=float(
            logical_bytes(batch_size, query_len, channels, kernel_size, dilation)
            / (time_ms / 1000.0)
            / 1e9
        ),
        energy_j=float(energy_j),
    )


__all__ = [
    "logical_bytes",
    "profile_ple_short_conv_vllm_triton",
    "reference_conv",
    "reference_state",
    "validate_args",
]
