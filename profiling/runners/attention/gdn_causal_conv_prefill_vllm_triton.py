"""One-launch vLLM Triton runner for fresh Qwen GDN convolution prefill.

The timed callable is exactly vLLM's fused causal-convolution prefill wrapper.
Packing, metadata preparation, the semantic correctness guard, and restoration
are setup work outside CUPTI and energy measurement. Reported FLOPs and bytes
are semantic logical counts, not physical Triton instructions or device traffic.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, replace
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import require_supported_gpu
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

_BACKEND = "gdn_causal_conv_prefill:vllm_triton"
_CALLABLE_MODULE = "vllm.model_executor.layers.mamba.ops.causal_conv1d"
_CALLABLE_NAME = "causal_conv1d_fn"
_METADATA_MODULE = "vllm.v1.attention.backends.utils"
_METADATA_NAME = "compute_causal_conv1d_metadata"
_KERNEL_NAME = "_causal_conv1d_fwd_kernel"
_SUPPORTED_GPUS = frozenset({"NVIDIA H200", "NVIDIA B200"})
_SUPPORTED_KERNEL_SIZES = frozenset(range(2, 5))
_BLOCK_M = 8
_BLOCK_N = 256
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
class _OperandShapes:
    semantic_x: tuple[int, int, int]
    packed_x: tuple[int, int]
    weight: tuple[int, int]
    state_storage: tuple[int, int, int]
    slot_indices: tuple[int]
    query_start_loc: tuple[int]
    has_initial_state: tuple[int]
    metadata_buffers: tuple[int]
    packed_output: tuple[int, int]
    semantic_output: tuple[int, int, int]
    program_count: int


@dataclass(frozen=True)
class _ConvMetadata:
    nums_dict: dict
    batch_ptr: Any
    token_chunk_offset_ptr: Any


@dataclass(frozen=True)
class _Operands:
    semantic_x: Any
    packed_x: Any
    weight: Any
    state: Any
    slot_indices: Any
    query_start_loc_cpu: Any
    query_start_loc_gpu: Any
    has_initial_state: Any
    metadata: _ConvMetadata


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
            "kernel_size must be supported by the production prefill Triton kernel "
            f"({min(_SUPPORTED_KERNEL_SIZES)}..{max(_SUPPORTED_KERNEL_SIZES)}), "
            f"got {validated.kernel_size}; upstream silently computes W5 incorrectly"
        )
    if validated.dtype is not DType.BF16 or validated.state_dtype is not DType.BF16:
        raise ValueError(
            "vllm_triton gdn_causal_conv_prefill requires dtype=bf16 and "
            "state_dtype=bf16, got "
            f"{validated.dtype.value} and {validated.state_dtype.value}"
        )
    return validated


def _guard_args(args: _ValidatedArgs) -> _ValidatedArgs:
    """Keep B/L/W exact and cap guard channels to roughly one million values.

    When ``B*L`` itself exceeds the bound, one channel is the smallest possible
    representative that preserves the physical batch and sequence axes.
    """
    max_channels = max(
        1,
        _MAX_GUARD_ACTIVATION_ELEMENTS // (args.batch_size * args.sequence_length),
    )
    return replace(args, channels=min(args.channels, max_channels))


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    program_count = args.batch_size * math.ceil(args.sequence_length / _BLOCK_M)
    metadata_capacity = max(1024, program_count) * 2
    return _OperandShapes(
        semantic_x=(args.batch_size, args.sequence_length, args.channels),
        packed_x=(args.channels, args.batch_size * args.sequence_length),
        weight=(args.channels, args.kernel_size),
        state_storage=(
            args.batch_size + 1,
            args.channels,
            args.kernel_size - 1,
        ),
        slot_indices=(args.batch_size,),
        query_start_loc=(args.batch_size + 1,),
        has_initial_state=(args.batch_size,),
        metadata_buffers=(metadata_capacity,),
        packed_output=(args.channels, args.batch_size * args.sequence_length),
        semantic_output=(args.batch_size, args.sequence_length, args.channels),
        program_count=program_count,
    )


def _valid_slot_indices(batch_size: int) -> tuple[int, ...]:
    return tuple(range(1, batch_size + 1))


def _expected_chunk_mapping(
    batch_size: int, sequence_length: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    chunks_per_sequence = math.ceil(sequence_length / _BLOCK_M)
    batch_ids = tuple(batch for batch in range(batch_size) for _chunk in range(chunks_per_sequence))
    offsets = tuple(chunk for _batch in range(batch_size) for chunk in range(chunks_per_sequence))
    return batch_ids, offsets


def _require_supported_gpu(torch: Any) -> None:
    require_supported_gpu(torch, backend=_BACKEND, supported_gpus=_SUPPORTED_GPUS)


def _load_vllm_components() -> tuple[Any, Any]:
    try:
        callable_module = importlib.import_module(_CALLABLE_MODULE)
        metadata_module = importlib.import_module(_METADATA_MODULE)
    except Exception as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the repository vllm_env") from exc
    fused_callable = getattr(callable_module, _CALLABLE_NAME, None)
    metadata_helper = getattr(metadata_module, _METADATA_NAME, None)
    if not callable(fused_callable) or not callable(metadata_helper):
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires {_CALLABLE_MODULE}.{_CALLABLE_NAME} and "
            f"{_METADATA_MODULE}.{_METADATA_NAME}"
        )
    return fused_callable, metadata_helper


def _build_operands(
    torch: Any,
    args: _ValidatedArgs,
    metadata_helper: Any,
    *,
    device: Any,
) -> _Operands:
    shapes = _operand_shapes(args)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    semantic_x = torch.empty(
        shapes.semantic_x,
        dtype=args.dtype.torch(),
        device=device,
    ).uniform_(-0.25, 0.25, generator=generator)
    semantic_x = semantic_x.contiguous()
    packed_x = semantic_x.reshape(args.batch_size * args.sequence_length, args.channels).transpose(
        0, 1
    )
    weight = torch.empty(
        shapes.weight,
        dtype=args.dtype.torch(),
        device=device,
    ).uniform_(-0.125, 0.125, generator=generator)
    state = torch.empty(
        shapes.state_storage,
        dtype=args.state_dtype.torch(),
        device=device,
    ).uniform_(-0.25, 0.25, generator=generator)
    slot_indices = torch.tensor(
        _valid_slot_indices(args.batch_size), dtype=torch.int32, device=device
    )
    query_start_loc_cpu = torch.arange(
        0,
        (args.batch_size + 1) * args.sequence_length,
        args.sequence_length,
        dtype=torch.int32,
        device="cpu",
    )
    query_start_loc_gpu = query_start_loc_cpu.to(device=device)
    has_initial_state = torch.zeros(shapes.has_initial_state, dtype=torch.bool, device=device)
    nums_dict, batch_ptr, token_chunk_offset_ptr = metadata_helper(
        query_start_loc_cpu,
        device=device,
    )
    metadata = _ConvMetadata(
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )
    return _Operands(
        semantic_x=semantic_x,
        packed_x=packed_x,
        weight=weight.contiguous(),
        state=state.contiguous(),
        slot_indices=slot_indices,
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc_gpu=query_start_loc_gpu,
        has_initial_state=has_initial_state,
        metadata=metadata,
    )


def _invoke_fused(fused_callable: Any, operands: _Operands) -> Any:
    return fused_callable(
        operands.packed_x,
        operands.weight,
        None,
        operands.state,
        operands.query_start_loc_gpu,
        cache_indices=operands.slot_indices,
        has_initial_state=operands.has_initial_state,
        activation="silu",
        metadata=operands.metadata,
        validate_data=True,
    )


def _semantic_output(packed_output: Any, args: _ValidatedArgs) -> Any:
    return packed_output.transpose(0, 1).reshape(
        args.batch_size,
        args.sequence_length,
        args.channels,
    )


def _shares_storage(left: Any, right: Any) -> bool:
    return (
        left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.device == right.device
    )


def _metadata_snapshot(metadata: _ConvMetadata) -> dict[str, Any]:
    snapshot = {
        "batch_ptr": metadata.batch_ptr.clone(),
        "token_chunk_offset_ptr": metadata.token_chunk_offset_ptr.clone(),
    }
    for key, value in metadata.nums_dict[_BLOCK_M].items():
        if hasattr(value, "clone"):
            snapshot[f"nums_dict.{key}"] = value.clone()
    return snapshot


def _metadata_tensors(metadata: _ConvMetadata) -> dict[str, Any]:
    tensors = {
        "batch_ptr": metadata.batch_ptr,
        "token_chunk_offset_ptr": metadata.token_chunk_offset_ptr,
    }
    for key, value in metadata.nums_dict[_BLOCK_M].items():
        if hasattr(value, "clone"):
            tensors[f"nums_dict.{key}"] = value
    return tensors


def _metadata_unchanged(
    torch: Any,
    metadata: _ConvMetadata,
    snapshot: dict[str, Any],
) -> bool:
    current = _metadata_tensors(metadata)
    return current.keys() == snapshot.keys() and all(
        torch.equal(current[key], expected) for key, expected in snapshot.items()
    )


def _restore_metadata(metadata: _ConvMetadata, snapshot: dict[str, Any]) -> None:
    for key, tensor in _metadata_tensors(metadata).items():
        tensor.copy_(snapshot[key])


def _check_correctness(
    torch: Any,
    fused_callable: Any,
    operands: _Operands,
    args: _ValidatedArgs,
    *,
    synchronize: Any | None = None,
) -> None:
    """Check fresh-prefill semantics twice, then restore every mutable operand."""
    from profiling.runners.attention.gdn_causal_conv_prefill_reference import (
        gdn_causal_conv_prefill_reference,
    )

    synchronize = torch.cuda.synchronize if synchronize is None else synchronize
    x_snapshot = operands.semantic_x.clone()
    weight_snapshot = operands.weight.clone()
    state_snapshot = operands.state.clone()
    indices_snapshot = operands.slot_indices.clone()
    query_cpu_snapshot = operands.query_start_loc_cpu.clone()
    query_gpu_snapshot = operands.query_start_loc_gpu.clone()
    initial_snapshot = operands.has_initial_state.clone()
    metadata_snapshot = _metadata_snapshot(operands.metadata)
    state_storage_ptr = operands.state.untyped_storage().data_ptr()
    reference_state = state_snapshot.clone()
    expected_output, expected_state = gdn_causal_conv_prefill_reference(
        x_snapshot.clone(),
        weight_snapshot,
        reference_state,
        indices_snapshot,
    )
    selected = indices_snapshot.to(torch.int64)
    try:
        packed_output = _invoke_fused(fused_callable, operands)
        synchronize()
        output = _semantic_output(packed_output, args)
        if tuple(packed_output.shape) != _operand_shapes(args).packed_output:
            raise AssertionError(f"unexpected fused output shape {tuple(packed_output.shape)}")
        if tuple(output.shape) != _operand_shapes(args).semantic_output:
            raise AssertionError(f"unexpected semantic output shape {tuple(output.shape)}")
        if packed_output.dtype is not torch.bfloat16:
            raise AssertionError(f"unexpected fused output dtype {packed_output.dtype}")
        if _shares_storage(packed_output, operands.packed_x):
            raise AssertionError("fused output must not alias input storage")
        if _shares_storage(packed_output, operands.weight):
            raise AssertionError("fused output must not alias weight storage")
        if _shares_storage(packed_output, operands.state):
            raise AssertionError("fused output must not alias state storage")
        torch.testing.assert_close(
            output.float(),
            expected_output.float(),
            atol=_OUTPUT_ATOL,
            rtol=_OUTPUT_RTOL,
        )
        if not torch.equal(
            operands.state.index_select(0, selected),
            expected_state.index_select(0, selected),
        ):
            raise AssertionError("fused selected state must exactly match the reference")
        first_output = packed_output.clone()
        first_selected_state = operands.state.index_select(0, selected).clone()

        # A distinct prior selected state must produce the same fresh-prefill
        # output and final state. This second guard launch remains outside timing.
        operands.state.copy_(state_snapshot)
        operands.state.index_fill_(0, selected, 0.75)
        alternate_output = _invoke_fused(fused_callable, operands)
        synchronize()
        if not torch.equal(alternate_output, first_output):
            raise AssertionError("fresh prefill output unexpectedly depends on prior state")
        if not torch.equal(operands.state.index_select(0, selected), first_selected_state):
            raise AssertionError("fresh prefill final state unexpectedly depends on prior state")

        if operands.state.untyped_storage().data_ptr() != state_storage_ptr:
            raise AssertionError("fused call replaced state storage instead of mutating in place")
        if torch.equal(first_selected_state, state_snapshot.index_select(0, selected)):
            raise AssertionError("fused call did not mutate selected state slots")
        if not torch.equal(operands.state[0], state_snapshot[0]):
            raise AssertionError("reserved state slot zero was mutated")
        if not torch.equal(operands.semantic_x, x_snapshot):
            raise AssertionError("fused call mutated input")
        if not torch.equal(operands.weight, weight_snapshot):
            raise AssertionError("fused call mutated weight")
        if not torch.equal(operands.slot_indices, indices_snapshot):
            raise AssertionError("fused call mutated slot indices")
        if not torch.equal(operands.query_start_loc_cpu, query_cpu_snapshot):
            raise AssertionError("fused call mutated CPU query starts")
        if not torch.equal(operands.query_start_loc_gpu, query_gpu_snapshot):
            raise AssertionError("fused call mutated GPU query starts")
        if not torch.equal(operands.has_initial_state, initial_snapshot):
            raise AssertionError("fused call mutated has_initial_state")
        if not _metadata_unchanged(torch, operands.metadata, metadata_snapshot):
            raise AssertionError("fused call mutated precomputed metadata")
        if not torch.isfinite(alternate_output).all() or not torch.isfinite(operands.state).all():
            raise AssertionError("fused output and updated state must remain finite")
    finally:
        operands.semantic_x.copy_(x_snapshot)
        operands.weight.copy_(weight_snapshot)
        operands.state.copy_(state_snapshot)
        operands.slot_indices.copy_(indices_snapshot)
        operands.query_start_loc_cpu.copy_(query_cpu_snapshot)
        operands.query_start_loc_gpu.copy_(query_gpu_snapshot)
        operands.has_initial_state.copy_(initial_snapshot)
        _restore_metadata(operands.metadata, metadata_snapshot)


def profile_gdn_causal_conv_prefill_vllm_triton(
    batch_size: int,
    sequence_length: int,
    channels: int,
    kernel_size: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> ComputeMetrics:
    """Profile vLLM's fused fresh-prefill convolution on NVIDIA H200 or B200."""
    args = _validate_args(
        batch_size,
        sequence_length,
        channels,
        kernel_size,
        dtype,
        state_dtype,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc

    try:
        _require_supported_gpu(torch)
        fused_callable, metadata_helper = _load_vllm_components()
        device = torch.device("cuda", torch.cuda.current_device())
        guard_args = _guard_args(args)
        guard_operands = _build_operands(torch, guard_args, metadata_helper, device=device)
        _check_correctness(torch, fused_callable, guard_operands, guard_args)
        # Build fresh requested-geometry operands after the guard. Packing and
        # metadata setup are complete before either measurement region starts.
        operands = _build_operands(torch, args, metadata_helper, device=device)

        def kernel() -> Any:
            return _invoke_fused(fused_callable, operands)

        # No reset is included in either measurement region. Fresh prefill
        # ignores prior state and deterministically overwrites selected slots.
        # Matching BF16 x/state makes the wrapper cast a no-op; its internal
        # output allocation adds no neighboring GPU compute launch.
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
        memory_bandwidth_gbps=(logical_bytes / elapsed_s / 1e9 if time_ms > 0.0 else 0.0),
    )
