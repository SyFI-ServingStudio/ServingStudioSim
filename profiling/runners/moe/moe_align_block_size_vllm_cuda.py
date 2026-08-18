"""Two-launch vLLM CUDA backend for MoE block alignment.

The timed callable is one ``moe_align_block_size`` wrapper invocation. CUPTI
sums its align and count/sort launches. The reported operation and byte rates
remain semantic/logical; energy includes three output allocations and the
internal int32 cumsum allocation.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.moe.moe_align_block_size_torch import (
    _logical_bytes,
    _operand_shapes,
    _semantic_ops,
    _ValidatedArgs,
)
from profiling.runners.moe.moe_align_block_size_torch import (
    _validate_args as _validate_semantic_args,
)

_BACKEND = "moe_align_block_size:vllm_cuda"
_MODULE = "vllm.model_executor.layers.fused_moe.moe_align_block_size"
_CALLABLE = "moe_align_block_size"
_GPU = "NVIDIA H200"
_NUM_EXPERTS = 256
_TOP_K = 8
_BLOCK_SIZE = 16
_MIN_TOKENS = 9
_QWEN_TOKENS = 128
# The wrapper has exactly these two launches for E256. Passing no CUPTI name
# filter sums the complete callable while the one-call closure excludes all
# neighboring router, expert, quantization, and finalize work.
_KERNEL_NAME: None = None
_EXPECTED_KERNEL_SUBSTRINGS = (
    "moe_align_block_size_kernel",
    "count_and_sort_expert_tokens_kernel",
)


@dataclass(frozen=True)
class _Operands:
    topk_ids: Any


def _validate_args(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    block_size: int,
) -> _ValidatedArgs:
    args = _validate_semantic_args(num_tokens, num_experts, top_k, block_size)
    if args.num_tokens < _MIN_TOKENS:
        raise ValueError(f"{_BACKEND} requires num_tokens>=9; smaller calls bypass alignment")
    if args.num_experts != _NUM_EXPERTS:
        raise ValueError(f"{_BACKEND} requires num_experts=256")
    if args.top_k != _TOP_K:
        raise ValueError(f"{_BACKEND} requires top_k=8")
    if args.block_size != _BLOCK_SIZE:
        raise ValueError(f"{_BACKEND} requires block_size=16")
    return args


def _guard_args(args: _ValidatedArgs) -> _ValidatedArgs:
    """Guard exact Qwen directly; otherwise use the bounded T9 witness."""
    if args.num_tokens == _QWEN_TOKENS:
        return args
    return _validate_args(_MIN_TOKENS, _NUM_EXPERTS, _TOP_K, _BLOCK_SIZE)


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    # Balanced cyclic IDs are packed, in range, and row-unique because K <= E.
    topk_ids = (
        torch.arange(args.assignments, dtype=torch.int64, device=device)
        .remainder(args.num_experts)
        .reshape(args.num_tokens, args.top_k)
        .to(torch.int32)
        .contiguous()
    )
    return _Operands(topk_ids=topk_ids)


def _validate_operands(
    torch: Any,
    operands: _Operands,
    args: _ValidatedArgs,
    *,
    require_cuda: bool = True,
) -> None:
    ids = operands.topk_ids
    if tuple(ids.shape) != (args.num_tokens, args.top_k) or ids.dtype is not torch.int32:
        raise ValueError("topk_ids must have exact [T,8] shape and int32 dtype")
    if (
        (require_cuda and not ids.is_cuda)
        or not ids.is_contiguous()
        or ids.stride()
        != (
            args.top_k,
            1,
        )
    ):
        raise ValueError("topk_ids must use packed contiguous CUDA int32 storage")
    if not torch.isfinite(ids).all():
        raise ValueError("topk_ids must be finite")
    if not ((ids >= 0) & (ids < args.num_experts)).all():
        raise ValueError("topk_ids must lie in [0, num_experts)")
    sorted_rows = ids.sort(dim=1).values
    if torch.any(sorted_rows[:, 1:] == sorted_rows[:, :-1]).item():
        raise ValueError("topk_ids must be unique within each token row")


def _require_h200(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"CUDA is required for {_BACKEND}")
    name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if name != _GPU:
        raise ProfilerNotImplemented(f"{_BACKEND} is verified only on {_GPU}, got {name}")


def _load_callable() -> Any:
    try:
        module = importlib.import_module(_MODULE)
        callable_ = getattr(module, _CALLABLE)
    except Exception as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires repository vllm_env") from exc
    if not callable(callable_):
        raise ProfilerNotImplemented(f"missing {_MODULE}.{_CALLABLE}")
    return callable_


def _invoke(callable_: Any, operands: _Operands) -> tuple[Any, Any, Any]:
    return callable_(
        operands.topk_ids,
        _BLOCK_SIZE,
        _NUM_EXPERTS,
        None,
        pad_sorted_ids=False,
        ignore_invalid_experts=False,
    )


def _shares(left: Any, right: Any) -> bool:
    return (
        left.device == right.device
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _validate_outputs(
    torch: Any,
    outputs: tuple[Any, Any, Any],
    operands: _Operands,
    args: _ValidatedArgs,
) -> None:
    if not isinstance(outputs, tuple) or len(outputs) != 3:
        raise AssertionError("moe_align_block_size must return exactly three tensors")
    shapes = _operand_shapes(args)
    expected = (
        ("sorted_token_ids", shapes["sorted_token_ids"]),
        ("expert_ids", shapes["expert_ids"]),
        ("num_tokens_post_pad", shapes["num_tokens_post_pad"]),
    )
    for output, (name, shape) in zip(outputs, expected, strict=True):
        if tuple(output.shape) != shape or output.dtype is not torch.int32:
            raise AssertionError(f"invalid {name} shape/dtype")
        if not output.is_contiguous() or not torch.isfinite(output).all():
            raise AssertionError(f"{name} must be finite contiguous storage")
        if output.device != operands.topk_ids.device:
            raise AssertionError(f"{name} must remain on the input device")
    tensors = (*outputs, operands.topk_ids)
    for index, left in enumerate(tensors):
        if any(_shares(left, right) for right in tensors[index + 1 :]):
            raise AssertionError("input and outputs must use disjoint storage")


def _semantic_signature(
    torch: Any,
    outputs: tuple[Any, Any, Any],
    ids_cpu: Any,
    args: _ValidatedArgs,
) -> tuple[int, tuple[int, ...], tuple[tuple[int, ...], ...]]:
    sorted_ids, expert_ids, post_pad = (tensor.cpu() for tensor in outputs)
    post = int(post_pad.item())
    if not 0 <= post <= args.capacity or post % args.block_size:
        raise AssertionError("invalid padded-token count")
    sentinel = args.assignments
    valid_blocks = post // args.block_size
    if not torch.all(sorted_ids[post:] == sentinel):
        raise AssertionError("inactive sorted-token capacity must contain the sentinel")
    if not torch.all(expert_ids[valid_blocks:] == -1):
        raise AssertionError("inactive blocks must contain expert -1")

    flat = ids_cpu.reshape(-1)
    owners = tuple(int(value) for value in expert_ids[:valid_blocks].tolist())
    assignments_by_expert: list[tuple[int, ...]] = []
    for expert in range(args.num_experts):
        observed = sorted(
            value
            for block, owner in enumerate(owners)
            if owner == expert
            for value in sorted_ids[
                block * args.block_size : (block + 1) * args.block_size
            ].tolist()
            if value != sentinel
        )
        wanted = torch.nonzero(flat == expert, as_tuple=False).flatten().tolist()
        if observed != wanted:
            raise AssertionError(f"expert {expert} assignment multiset is incorrect")
        assignments_by_expert.append(tuple(observed))
    return post, owners, tuple(assignments_by_expert)


def _check_correctness(
    torch: Any,
    callable_: Any,
    operands: _Operands,
    args: _ValidatedArgs,
    *,
    synchronize: Any | None = None,
) -> None:
    from profiling.runners.moe.moe_align_block_size_reference import (
        moe_align_block_size_reference,
    )

    _validate_operands(torch, operands, args, require_cuda=operands.topk_ids.is_cuda)
    ids_before = operands.topk_ids.clone()
    ids_cpu = ids_before.cpu()
    expected = moe_align_block_size_reference(ids_cpu, args.num_experts, args.block_size)
    expected_signature = _semantic_signature(torch, expected, ids_cpu, args)

    first = _invoke(callable_, operands)
    (synchronize or (torch.cuda.synchronize if operands.topk_ids.is_cuda else lambda: None))()
    _validate_outputs(torch, first, operands, args)
    first_signature = _semantic_signature(torch, first, ids_cpu, args)
    if first_signature != expected_signature:
        raise AssertionError("vLLM alignment disagrees with the semantic reference")

    second = _invoke(callable_, operands)
    (synchronize or (torch.cuda.synchronize if operands.topk_ids.is_cuda else lambda: None))()
    _validate_outputs(torch, second, operands, args)
    if _semantic_signature(torch, second, ids_cpu, args) != first_signature:
        raise AssertionError("repeated vLLM alignment is not semantically equivalent")
    if any(_shares(left, right) for left in first for right in second):
        raise AssertionError("each wrapper invocation must allocate fresh outputs")
    if not torch.equal(operands.topk_ids, ids_before):
        raise AssertionError("vLLM alignment mutated topk_ids")


def profile_moe_align_block_size_vllm_cuda(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    block_size: int,
) -> ComputeMetrics:
    args = _validate_args(num_tokens, num_experts, top_k, block_size)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc
    try:
        _require_h200(torch)
        callable_ = _load_callable()
        device = torch.device("cuda", torch.cuda.current_device())
        operands = _build_operands(torch, args, device=device)
        _validate_operands(torch, operands, args)
        guard_args = _guard_args(args)
        guard = (
            operands if guard_args == args else _build_operands(torch, guard_args, device=device)
        )
        _check_correctness(torch, callable_, guard, guard_args)

        def kernel():
            return _invoke(callable_, operands)

        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        # Energy covers both kernels plus wrapper allocations; inputs are immutable.
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    operations = _semantic_ops(
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        top_k=args.top_k,
        block_size=args.block_size,
    )
    logical_bytes = _logical_bytes(
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        top_k=args.top_k,
        block_size=args.block_size,
    )
    elapsed = time_ms / 1000
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=float(operations / elapsed / 1e12 if elapsed else 0),
        memory_bandwidth_gbps=float(logical_bytes / elapsed / 1e9 if elapsed else 0),
    )
