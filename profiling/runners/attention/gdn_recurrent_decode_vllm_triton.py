"""One-launch vLLM Triton runner for Qwen GDN recurrent decode.

The timed callable is exactly vLLM's fused packed recurrent-decode operation.
Packing, allocation, the semantic correctness guard, and its state restoration
are setup work and remain outside CUPTI and energy measurement. Reported FLOPs
and bytes are semantic logical counts, not physical Triton instructions or
device traffic.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import load_required_callable, require_exact_gpu
from profiling.runners.attention.gdn_recurrent_decode_torch import (
    _logical_bytes,
    _semantic_flops,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gdn_recurrent_decode:vllm_triton"
_CALLABLE_MODULE = "vllm.model_executor.layers.fla.ops.fused_recurrent"
_CALLABLE_NAME = "fused_recurrent_gated_delta_rule_packed_decode"
_KERNEL_NAME = "fused_recurrent_gated_delta_rule_packed_decode_kernel"
_REQUIRED_GPU = "NVIDIA H200"
_OUTPUT_ATOL = 1e-2
_OUTPUT_RTOL = 1e-2
_STATE_ATOL = 2e-5
_STATE_RTOL = 2e-4


@dataclass(frozen=True)
class _ValidatedArgs:
    batch_size: int
    num_qk_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    dtype: DType
    state_dtype: DType


@dataclass(frozen=True)
class _OperandShapes:
    query: tuple[int, int, int]
    value: tuple[int, int, int]
    gates: tuple[int, int]
    parameters: tuple[int]
    mixed_qkv: tuple[int, int]
    state_storage: tuple[int, int, int, int]
    output: tuple[int, int, int, int]


@dataclass(frozen=True)
class _Operands:
    query: Any
    key: Any
    value: Any
    mixed_qkv: Any
    a: Any
    b: Any
    A_log: Any
    dt_bias: Any
    initial_state: Any
    out: Any
    ssm_state_indices: Any


def _validate_args(
    batch_size: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> _ValidatedArgs:
    validated = _ValidatedArgs(
        batch_size=int(batch_size),
        num_qk_heads=int(num_qk_heads),
        num_value_heads=int(num_value_heads),
        key_head_dim=int(key_head_dim),
        value_head_dim=int(value_head_dim),
        dtype=DType.from_value(dtype),
        state_dtype=DType.from_value(state_dtype),
    )
    dimensions = (
        validated.batch_size,
        validated.num_qk_heads,
        validated.num_value_heads,
        validated.key_head_dim,
        validated.value_head_dim,
    )
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(
            "batch_size, num_qk_heads, num_value_heads, key_head_dim, and "
            f"value_head_dim must be > 0, got {dimensions}"
        )
    if validated.num_value_heads % validated.num_qk_heads != 0:
        raise ValueError(
            "num_value_heads must be divisible by num_qk_heads, "
            f"got {validated.num_value_heads} and {validated.num_qk_heads}"
        )
    if validated.dtype is not DType.BF16 or validated.state_dtype is not DType.FP32:
        raise ValueError(
            "vllm_triton gdn_recurrent_decode requires dtype=bf16 and "
            "state_dtype=fp32, got "
            f"{validated.dtype.value} and {validated.state_dtype.value}"
        )
    return validated


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    packed_width = (
        2 * args.num_qk_heads * args.key_head_dim + args.num_value_heads * args.value_head_dim
    )
    return _OperandShapes(
        query=(args.batch_size, args.num_qk_heads, args.key_head_dim),
        value=(args.batch_size, args.num_value_heads, args.value_head_dim),
        gates=(args.batch_size, args.num_value_heads),
        parameters=(args.num_value_heads,),
        mixed_qkv=(args.batch_size, packed_width),
        state_storage=(
            args.batch_size + 1,
            args.num_value_heads,
            args.value_head_dim,
            args.key_head_dim,
        ),
        output=(args.batch_size, 1, args.num_value_heads, args.value_head_dim),
    )


def _valid_slot_indices(batch_size: int) -> tuple[int, ...]:
    return tuple(range(1, batch_size + 1))


def _require_h200(torch: Any) -> None:
    require_exact_gpu(torch, backend=_BACKEND, required_gpu=_REQUIRED_GPU)


def _load_fused_callable() -> Any:
    return load_required_callable(
        importlib.import_module,
        backend=_BACKEND,
        module_name=_CALLABLE_MODULE,
        callable_name=_CALLABLE_NAME,
    )


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    shapes = _operand_shapes(args)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    activation_dtype = args.dtype.torch()
    state_dtype = args.state_dtype.torch()
    activation_kwargs = {
        "dtype": activation_dtype,
        "device": device,
        "generator": generator,
    }
    query = (0.25 * torch.randn(shapes.query, **activation_kwargs)).contiguous()
    key = (0.25 * torch.randn(shapes.query, **activation_kwargs)).contiguous()
    value = (0.25 * torch.randn(shapes.value, **activation_kwargs)).contiguous()
    a = torch.empty(shapes.gates, dtype=activation_dtype, device=device).uniform_(
        -0.25, 0.25, generator=generator
    )
    b = torch.empty(shapes.gates, dtype=activation_dtype, device=device).uniform_(
        -0.25, 0.25, generator=generator
    )
    A_log = torch.zeros(shapes.parameters, dtype=state_dtype, device=device)
    dt_bias = torch.zeros(shapes.parameters, dtype=state_dtype, device=device)
    semantic_state = torch.empty(
        (
            args.batch_size,
            args.num_value_heads,
            args.key_head_dim,
            args.value_head_dim,
        ),
        dtype=state_dtype,
        device=device,
    ).normal_(mean=0.0, std=0.01, generator=generator)
    initial_state = torch.zeros(shapes.state_storage, dtype=state_dtype, device=device)
    initial_state[1:].copy_(semantic_state.transpose(-1, -2))
    mixed_qkv = torch.cat((query.flatten(1), key.flatten(1), value.flatten(1)), dim=-1).contiguous()
    out = torch.empty(shapes.output, dtype=activation_dtype, device=device)
    ssm_state_indices = torch.tensor(
        _valid_slot_indices(args.batch_size), dtype=torch.int32, device=device
    )
    return _Operands(
        query=query,
        key=key,
        value=value,
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        out=out,
        ssm_state_indices=ssm_state_indices,
    )


def _invoke_fused(fused_callable: Any, operands: _Operands, args: _ValidatedArgs) -> Any:
    return fused_callable(
        mixed_qkv=operands.mixed_qkv,
        a=operands.a,
        b=operands.b,
        A_log=operands.A_log,
        dt_bias=operands.dt_bias,
        scale=args.key_head_dim**-0.5,
        initial_state=operands.initial_state,
        out=operands.out,
        ssm_state_indices=operands.ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )


def _check_correctness(
    torch: Any,
    fused_callable: Any,
    operands: _Operands,
    args: _ValidatedArgs,
    *,
    synchronize: Any | None = None,
) -> None:
    """Compare fused output and every valid state slot, then restore state."""
    from profiling.runners.attention.gdn_recurrent_decode_reference import (
        gdn_recurrent_decode_reference,
    )

    synchronize = torch.cuda.synchronize if synchronize is None else synchronize
    state_snapshot = operands.initial_state.clone()
    immutable_snapshots = {
        "mixed_qkv": operands.mixed_qkv.clone(),
        "a": operands.a.clone(),
        "b": operands.b.clone(),
        "A_log": operands.A_log.clone(),
        "dt_bias": operands.dt_bias.clone(),
        "ssm_state_indices": operands.ssm_state_indices.clone(),
    }
    semantic_state = state_snapshot[1:].transpose(-1, -2).contiguous()
    expected_output, expected_state = gdn_recurrent_decode_reference(
        operands.query,
        operands.key,
        operands.value,
        operands.a,
        operands.b,
        operands.A_log,
        operands.dt_bias,
        semantic_state,
    )
    try:
        returned = _invoke_fused(fused_callable, operands, args)
        synchronize()
        if not isinstance(returned, (tuple, list)) or len(returned) != 2:
            raise AssertionError("fused packed decode must return output and state")
        if returned[0] is not operands.out or returned[1] is not operands.initial_state:
            raise AssertionError(
                "fused packed decode must return its output/state buffers by identity"
            )
        if tuple(operands.out.shape) != _operand_shapes(args).output:
            raise AssertionError(f"unexpected fused output shape {tuple(operands.out.shape)}")
        if operands.out.dtype is not torch.bfloat16:
            raise AssertionError(f"unexpected fused output dtype {operands.out.dtype}")
        if not torch.equal(operands.initial_state[0], state_snapshot[0]):
            raise AssertionError("reserved state slot zero was mutated")
        actual_state = operands.initial_state[1:].transpose(-1, -2)
        torch.testing.assert_close(
            operands.out[:, 0].float(),
            expected_output.float(),
            atol=_OUTPUT_ATOL,
            rtol=_OUTPUT_RTOL,
        )
        torch.testing.assert_close(
            actual_state,
            expected_state,
            atol=_STATE_ATOL,
            rtol=_STATE_RTOL,
        )
        if not torch.isfinite(operands.out).all() or not torch.isfinite(actual_state).all():
            raise AssertionError("fused output and updated state must remain finite")
        for name, snapshot in immutable_snapshots.items():
            if not torch.equal(getattr(operands, name), snapshot):
                raise AssertionError(f"fused packed decode mutated {name}")
    finally:
        operands.initial_state.copy_(state_snapshot)
        operands.out.zero_()


def profile_gdn_recurrent_decode_vllm_triton(
    batch_size: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> ComputeMetrics:
    """Profile vLLM's fused packed recurrent decode on an NVIDIA H200."""
    args = _validate_args(
        batch_size,
        num_qk_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        dtype,
        state_dtype,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc

    try:
        _require_h200(torch)
        fused_callable = _load_fused_callable()
        device = torch.device("cuda", torch.cuda.current_device())
        operands = _build_operands(torch, args, device=device)
        _check_correctness(torch, fused_callable, operands, args)

        def kernel() -> Any:
            return _invoke_fused(fused_callable, operands, args)

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
        num_qk_heads=args.num_qk_heads,
        num_value_heads=args.num_value_heads,
        key_head_dim=args.key_head_dim,
        value_head_dim=args.value_head_dim,
    )
    logical_bytes = _logical_bytes(
        batch_size=args.batch_size,
        num_qk_heads=args.num_qk_heads,
        num_value_heads=args.num_value_heads,
        key_head_dim=args.key_head_dim,
        value_head_dim=args.value_head_dim,
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
