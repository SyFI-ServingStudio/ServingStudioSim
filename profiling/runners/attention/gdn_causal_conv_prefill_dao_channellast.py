"""Dao-AILab causal-conv1d channel-last runner for fresh causal-convolution prefill.

The timed callable is ``causal_conv1d.causal_conv1d_fn`` issued once per
sequence: its channel-last kernel takes no ``cache_indices`` and its varlen
``seq_idx`` path refuses ``return_final_states`` (and mishandles initial
states), so a varlen batch has to run one launch per sequence. Each launch
reads its sequence as a channel-last view of the packed ``(tokens, channels)``
activations and writes its final state straight into the request's slot of the
vLLM-style ``(slots, kernel_size - 1, channels)`` state store through
``final_states_out``. Operand construction and the correctness guards run
before CUPTI and energy measurement.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass, replace
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.gdn_causal_conv_prefill_torch import (
    _logical_bytes,
    _semantic_flops,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gdn_causal_conv_prefill:dao_channellast"
_MODULE = "causal_conv1d"
_DISTRIBUTION = "causal-conv1d"
_KERNEL_NAME = "causal_conv1d_channellast_fwd_kernel"
# causal_conv1d_fwd dispatches widths 2, 3 and 4 only (csrc/causal_conv1d.cpp).
_SUPPORTED_KERNEL_SIZES = frozenset(range(2, 5))
# The channel-last kernel loads 8 channels per vector (kNElts = 8 for 16-bit
# types) and requires the token stride and channel count to be multiples of 8.
_CHANNEL_MULTIPLE = 8
_MAX_GUARD_ACTIVATION_ELEMENTS = 1_048_576
_OUTPUT_ATOL = 1e-2
_OUTPUT_RTOL = 1e-2


@dataclass(frozen=True)
class _ValidatedArgs:
    batch_size: int
    sequence_length: int
    channels: int
    kernel_size: int
    dtype: DType
    state_dtype: DType


@dataclass(frozen=True)
class _Operands:
    semantic_x: Any
    packed_x: Any
    weight: Any
    state_store: Any
    slot_indices: tuple[int, ...]

    @property
    def state(self) -> Any:
        """Dim-first ``(slots, channels, kernel_size - 1)`` view, channels innermost."""
        return self.state_store.transpose(1, 2)


def _validate_args(
    batch_size: int,
    sequence_length: int,
    channels: int,
    kernel_size: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> _ValidatedArgs:
    validated = _ValidatedArgs(
        batch_size=int(batch_size),
        sequence_length=int(sequence_length),
        channels=int(channels),
        kernel_size=int(kernel_size),
        dtype=DType.from_value(dtype),
        state_dtype=DType.from_value(state_dtype),
    )
    if validated.batch_size <= 0 or validated.sequence_length <= 0 or validated.channels <= 0:
        raise ValueError(
            "batch_size, sequence_length, and channels must be > 0, got "
            f"batch_size={validated.batch_size}, "
            f"sequence_length={validated.sequence_length}, "
            f"channels={validated.channels}"
        )
    if validated.kernel_size not in _SUPPORTED_KERNEL_SIZES:
        raise ValueError(
            "kernel_size must be one causal_conv1d_fwd dispatches "
            f"({min(_SUPPORTED_KERNEL_SIZES)}..{max(_SUPPORTED_KERNEL_SIZES)}), "
            f"got {validated.kernel_size}"
        )
    if validated.channels % _CHANNEL_MULTIPLE:
        raise ValueError(
            f"the channel-last kernel needs channels divisible by {_CHANNEL_MULTIPLE}, "
            f"got {validated.channels}"
        )
    if validated.dtype is not DType.BF16 or validated.state_dtype is not DType.BF16:
        raise ValueError(
            "dao_channellast gdn_causal_conv_prefill requires dtype=bf16 and "
            "state_dtype=bf16, got "
            f"{validated.dtype.value} and {validated.state_dtype.value}"
        )
    return validated


def _guard_args(args: _ValidatedArgs) -> _ValidatedArgs:
    """Keep B/L/W exact and cap guard channels near one million activations,
    rounded down to the kernel's channel multiple."""
    budget = _MAX_GUARD_ACTIVATION_ELEMENTS // (args.batch_size * args.sequence_length)
    max_channels = max(_CHANNEL_MULTIPLE, budget - budget % _CHANNEL_MULTIPLE)
    return replace(args, channels=min(args.channels, max_channels))


def _load_callable() -> Any:
    try:
        module = importlib.import_module(_MODULE)
    except Exception as exc:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires the {_DISTRIBUTION!r} package (run under causal_conv1d_env)"
        ) from exc
    fn = getattr(module, "causal_conv1d_fn", None)
    if not callable(fn):
        raise ProfilerNotImplemented(f"{_BACKEND} requires {_MODULE}.causal_conv1d_fn")
    return fn


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    dtype = args.dtype.torch()
    semantic_x = torch.empty(
        (args.batch_size, args.sequence_length, args.channels), dtype=dtype, device=device
    ).uniform_(-0.25, 0.25, generator=generator)
    # Packed tokens as vLLM hands them to the conv: (channels, tokens), channels
    # innermost in memory.
    packed_x = semantic_x.reshape(args.batch_size * args.sequence_length, args.channels).t()
    weight = torch.empty((args.channels, args.kernel_size), dtype=dtype, device=device).uniform_(
        -0.125, 0.125, generator=generator
    )
    # Slot zero is reserved; sequence b owns slot b + 1.
    state_store = torch.empty(
        (args.batch_size + 1, args.kernel_size - 1, args.channels),
        dtype=args.state_dtype.torch(),
        device=device,
    ).uniform_(-0.25, 0.25, generator=generator)
    return _Operands(
        semantic_x=semantic_x,
        packed_x=packed_x,
        weight=weight.contiguous(),
        state_store=state_store,
        slot_indices=tuple(range(1, args.batch_size + 1)),
    )


def _sequence_views(operands: _Operands, args: _ValidatedArgs) -> list[tuple[Any, Any]]:
    """Per-sequence ``(x, final_state)`` views: x is ``(1, C, L)`` channel-last and
    the final state is that request's ``(1, C, W - 1)`` slot of the state store."""
    state = operands.state
    length = args.sequence_length
    return [
        (
            operands.packed_x[:, index * length : (index + 1) * length].unsqueeze(0),
            state[slot : slot + 1],
        )
        for index, slot in enumerate(operands.slot_indices)
    ]


def _invoke(fn: Any, weight: Any, views: list[tuple[Any, Any]]) -> list[Any]:
    outputs = []
    for x, final_state in views:
        output, _ = fn(
            x,
            weight,
            None,
            initial_states=None,
            return_final_states=True,
            final_states_out=final_state,
            activation="silu",
        )
        outputs.append(output)
    return outputs


def _semantic_output(torch: Any, outputs: list[Any]) -> Any:
    """Stack per-sequence ``(1, C, L)`` outputs into the reference's ``[B, L, C]``."""
    return torch.cat([output.transpose(1, 2) for output in outputs], dim=0)


def _check_fresh(torch: Any, fn: Any, operands: _Operands, args: _ValidatedArgs) -> None:
    """Fresh prefill matches the kind reference, writes state in place, and
    ignores prior state; restores every mutable operand afterwards."""
    from profiling.runners.attention.gdn_causal_conv_prefill_reference import (
        gdn_causal_conv_prefill_reference,
    )

    x_snapshot = operands.semantic_x.clone()
    store_snapshot = operands.state_store.clone()
    indices = torch.tensor(operands.slot_indices, dtype=torch.int32, device=x_snapshot.device)
    selected = indices.to(torch.int64)
    expected_output, expected_state = gdn_causal_conv_prefill_reference(
        x_snapshot.clone(), operands.weight, store_snapshot.transpose(1, 2).clone(), indices
    )
    views = _sequence_views(operands, args)
    try:
        outputs = _invoke(fn, operands.weight, views)
        torch.cuda.synchronize()
        for output, (x, _) in zip(outputs, views, strict=True):
            if tuple(output.shape) != tuple(x.shape) or output.dtype is not torch.bfloat16:
                raise AssertionError(
                    f"unexpected output {tuple(output.shape)} {output.dtype} for {tuple(x.shape)}"
                )
            if output.untyped_storage().data_ptr() == x.untyped_storage().data_ptr():
                raise AssertionError("output must not alias the input")
        output = _semantic_output(torch, outputs)
        torch.testing.assert_close(
            output.float(), expected_output.float(), atol=_OUTPUT_ATOL, rtol=_OUTPUT_RTOL
        )
        state = operands.state
        if not torch.equal(
            state.index_select(0, selected), expected_state.index_select(0, selected)
        ):
            raise AssertionError("final state must exactly match the reference")
        if not torch.equal(operands.state_store[0], store_snapshot[0]):
            raise AssertionError("reserved state slot zero was mutated")
        first_output = output.clone()
        first_state = state.index_select(0, selected).clone()

        # A different prior state must not change a fresh prefill.
        operands.state_store.copy_(store_snapshot)
        operands.state_store.index_fill_(0, selected, 0.75)
        alternate = _semantic_output(torch, _invoke(fn, operands.weight, views))
        torch.cuda.synchronize()
        if not torch.equal(alternate, first_output):
            raise AssertionError("fresh prefill output depends on prior state")
        if not torch.equal(state.index_select(0, selected), first_state):
            raise AssertionError("fresh prefill final state depends on prior state")
        if not torch.equal(operands.semantic_x, x_snapshot):
            raise AssertionError("the call mutated its input")
    finally:
        operands.state_store.copy_(store_snapshot)


def _check_initial_state(torch: Any, fn: Any, operands: _Operands, args: _ValidatedArgs) -> None:
    """The continuation contract a serving integration would use: with a prior
    state passed as ``initial_states``, output and final state equal the kind
    reference run over ``history ++ x`` (its first ``W - 1`` outputs dropped).

    Final states go to a separate buffer, as they must when the history is read
    from the same slot. Outside timing; the timed fresh call passes no history.
    """
    from profiling.runners.attention.gdn_causal_conv_prefill_reference import (
        gdn_causal_conv_prefill_reference,
    )

    state_length = args.kernel_size - 1
    history = operands.state_store[1:]  # (B, W - 1, C), one random history per request
    extended = torch.cat([history, operands.semantic_x], dim=1).contiguous()
    indices = torch.arange(1, args.batch_size + 1, dtype=torch.int32, device=extended.device)
    expected_output, expected_state = gdn_causal_conv_prefill_reference(
        extended,
        operands.weight,
        torch.zeros(
            (args.batch_size + 1, args.channels, state_length),
            dtype=operands.state_store.dtype,
            device=extended.device,
        ),
        indices,
    )
    for index, (x, _) in enumerate(_sequence_views(operands, args)):
        final = torch.empty(
            (1, state_length, args.channels),
            dtype=operands.state_store.dtype,
            device=extended.device,
        ).transpose(1, 2)
        output, _ = fn(
            x,
            operands.weight,
            None,
            initial_states=operands.state[index + 1 : index + 2],
            return_final_states=True,
            final_states_out=final,
            activation="silu",
        )
        torch.testing.assert_close(
            output.transpose(1, 2)[0].float(),
            expected_output[index, state_length:].float(),
            atol=_OUTPUT_ATOL,
            rtol=_OUTPUT_RTOL,
        )
        if not torch.equal(final[0], expected_state[index + 1]):
            raise AssertionError("final state after a prior state must exactly match the reference")


def profile_gdn_causal_conv_prefill_dao_channellast(
    batch_size: int,
    sequence_length: int,
    channels: int,
    kernel_size: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> ComputeMetrics:
    """Profile one channel-last causal-conv1d launch per fresh sequence."""
    args = _validate_args(batch_size, sequence_length, channels, kernel_size, dtype, state_dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc

    try:
        fn = _load_callable()
        device = torch.device("cuda", torch.cuda.current_device())
        guard_args = _guard_args(args)
        guard = _build_operands(torch, guard_args, device=device)
        _check_fresh(torch, fn, guard, guard_args)
        _check_initial_state(torch, fn, guard, guard_args)
        del guard
        operands = _build_operands(torch, args, device=device)
        views = _sequence_views(operands, args)
        weight = operands.weight

        def kernel() -> Any:
            return _invoke(fn, weight, views)

        # Fresh prefill ignores prior state and rewrites the same slots each
        # call, so no reset sits inside the measured region.
        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    flops = _semantic_flops(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        channels=args.channels,
        kernel_size=args.kernel_size,
    )
    logical_bytes = _logical_bytes(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        channels=args.channels,
        kernel_size=args.kernel_size,
        dtype=args.dtype,
        state_dtype=args.state_dtype,
    )
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=flops / elapsed_s / 1e12 if time_ms > 0.0 else 0.0,
        memory_bandwidth_gbps=logical_bytes / elapsed_s / 1e9 if time_ms > 0.0 else 0.0,
    )


def row_provenance(**_kwargs: Any) -> str | None:
    """The causal-conv1d build the row was measured with."""
    try:
        return f"{_DISTRIBUTION} {importlib.metadata.version(_DISTRIBUTION)}"
    except importlib.metadata.PackageNotFoundError:
        return None
