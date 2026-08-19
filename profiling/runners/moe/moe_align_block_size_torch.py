"""Multi-launch Torch semantic baseline for MoE block alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class _ValidatedArgs:
    num_tokens: int
    num_experts: int
    top_k: int
    block_size: int
    assignments: int
    capacity: int
    num_blocks: int


@dataclass(frozen=True)
class _Workspaces:
    flat_experts: Any
    counts: Any
    padded_counts: Any
    raw_offsets: Any
    padded_offsets: Any
    ones: Any
    assignment_ids: Any
    assignment_ranks: Any
    keys: Any
    sorted_keys: Any
    permutation: Any
    sorted_experts: Any
    raw_bases: Any
    padded_bases: Any
    positions: Any
    sorted_assignment_ids: Any
    block_starts: Any
    block_owners: Any
    inactive_blocks: Any


@dataclass(frozen=True)
class _Operands:
    topk_ids: Any
    sorted_token_ids: Any
    expert_ids: Any
    num_tokens_post_pad: Any
    workspaces: _Workspaces


def _exact_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an exact integer, got {value!r}")
    return value


def _checked_mul(left: int, right: int, label: str) -> int:
    value = left * right
    if value > _INT32_MAX:
        raise ValueError(f"{label} exceeds the int32 range")
    return value


def _checked_add(left: int, right: int, label: str) -> int:
    value = left + right
    if value > _INT32_MAX:
        raise ValueError(f"{label} exceeds the int32 range")
    return value


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _validate_args(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    block_size: int,
) -> _ValidatedArgs:
    num_tokens = _exact_int("num_tokens", num_tokens)
    num_experts = _exact_int("num_experts", num_experts)
    top_k = _exact_int("top_k", top_k)
    block_size = _exact_int("block_size", block_size)
    if min(num_tokens, num_experts, top_k, block_size) <= 0:
        raise ValueError("num_tokens, num_experts, top_k, and block_size must be positive")
    if top_k > num_experts:
        raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")
    assignments = _checked_mul(num_tokens, top_k, "assignment count")
    padding = _checked_mul(num_experts, block_size - 1, "padding capacity")
    capacity = _checked_add(assignments, padding, "sorted-token capacity")
    if assignments < num_experts:
        capacity = min(
            _checked_mul(assignments, block_size, "small-assignment capacity"),
            capacity,
        )
    num_blocks = _ceil_div(capacity, block_size)
    if num_blocks > _INT32_MAX:
        raise ValueError("expert block count exceeds the int32 range")
    return _ValidatedArgs(
        num_tokens,
        num_experts,
        top_k,
        block_size,
        assignments,
        capacity,
        num_blocks,
    )


def _operand_shapes(args: _ValidatedArgs) -> dict[str, tuple[int, ...]]:
    return {
        "topk_ids": (args.num_tokens, args.top_k),
        "sorted_token_ids": (args.capacity,),
        "expert_ids": (args.num_blocks,),
        "num_tokens_post_pad": (1,),
    }


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    shapes = _operand_shapes(args)
    # Consecutive cyclic experts are balanced and row-unique because K <= E.
    topk_ids = (
        torch.arange(args.assignments, dtype=torch.int64, device=device)
        .remainder(args.num_experts)
        .reshape(shapes["topk_ids"])
        .to(torch.int32)
        .contiguous()
    )
    assignment_ranks = torch.arange(args.assignments, dtype=torch.int64, device=device)
    block_starts = torch.arange(args.num_blocks, dtype=torch.int64, device=device) * args.block_size
    workspaces = _Workspaces(
        flat_experts=torch.empty(args.assignments, dtype=torch.int64, device=device),
        counts=torch.empty(args.num_experts, dtype=torch.int64, device=device),
        padded_counts=torch.empty(args.num_experts, dtype=torch.int64, device=device),
        raw_offsets=torch.empty(args.num_experts + 1, dtype=torch.int64, device=device),
        padded_offsets=torch.empty(args.num_experts + 1, dtype=torch.int64, device=device),
        ones=torch.ones(args.assignments, dtype=torch.int64, device=device),
        assignment_ids=torch.arange(args.assignments, dtype=torch.int32, device=device),
        assignment_ranks=assignment_ranks,
        keys=torch.empty(args.assignments, dtype=torch.int64, device=device),
        sorted_keys=torch.empty(args.assignments, dtype=torch.int64, device=device),
        permutation=torch.empty(args.assignments, dtype=torch.int64, device=device),
        sorted_experts=torch.empty(args.assignments, dtype=torch.int64, device=device),
        raw_bases=torch.empty(args.assignments, dtype=torch.int64, device=device),
        padded_bases=torch.empty(args.assignments, dtype=torch.int64, device=device),
        positions=torch.empty(args.assignments, dtype=torch.int64, device=device),
        sorted_assignment_ids=torch.empty(args.assignments, dtype=torch.int32, device=device),
        block_starts=block_starts,
        block_owners=torch.empty(args.num_blocks, dtype=torch.int64, device=device),
        inactive_blocks=torch.empty(args.num_blocks, dtype=torch.bool, device=device),
    )
    return _Operands(
        topk_ids=topk_ids,
        sorted_token_ids=torch.empty(shapes["sorted_token_ids"], dtype=torch.int32, device=device),
        expert_ids=torch.empty(shapes["expert_ids"], dtype=torch.int32, device=device),
        num_tokens_post_pad=torch.empty(
            shapes["num_tokens_post_pad"], dtype=torch.int32, device=device
        ),
        workspaces=workspaces,
    )


def _align_into(
    torch: Any,
    topk_ids: Any,
    sorted_token_ids: Any,
    expert_ids: Any,
    num_tokens_post_pad: Any,
    workspaces: _Workspaces,
    *,
    num_experts: int,
    block_size: int,
) -> tuple[Any, Any, Any]:
    """Perform count, scan, deterministic grouping, and padding in-place."""
    assignments = topk_ids.numel()
    sorted_token_ids.fill_(assignments)
    expert_ids.fill_(-1)

    workspaces.flat_experts.copy_(topk_ids.reshape(-1))
    workspaces.counts.zero_()
    workspaces.counts.scatter_add_(0, workspaces.flat_experts, workspaces.ones)
    workspaces.raw_offsets[0] = 0
    torch.cumsum(workspaces.counts, 0, out=workspaces.raw_offsets[1:])

    workspaces.padded_counts.copy_(workspaces.counts).add_(block_size - 1)
    torch.div(
        workspaces.padded_counts,
        block_size,
        rounding_mode="floor",
        out=workspaces.padded_counts,
    )
    workspaces.padded_counts.mul_(block_size)
    workspaces.padded_offsets[0] = 0
    torch.cumsum(workspaces.padded_counts, 0, out=workspaces.padded_offsets[1:])
    num_tokens_post_pad.copy_(workspaces.padded_offsets[-1:])

    workspaces.keys.copy_(workspaces.flat_experts).mul_(assignments + 1)
    workspaces.keys.add_(workspaces.assignment_ranks)
    torch.sort(
        workspaces.keys,
        out=(workspaces.sorted_keys, workspaces.permutation),
    )
    torch.index_select(
        workspaces.flat_experts,
        0,
        workspaces.permutation,
        out=workspaces.sorted_experts,
    )
    torch.index_select(
        workspaces.raw_offsets,
        0,
        workspaces.sorted_experts,
        out=workspaces.raw_bases,
    )
    torch.index_select(
        workspaces.padded_offsets,
        0,
        workspaces.sorted_experts,
        out=workspaces.padded_bases,
    )
    workspaces.positions.copy_(workspaces.assignment_ranks)
    workspaces.positions.sub_(workspaces.raw_bases).add_(workspaces.padded_bases)
    torch.index_select(
        workspaces.assignment_ids,
        0,
        workspaces.permutation,
        out=workspaces.sorted_assignment_ids,
    )
    sorted_token_ids.scatter_(0, workspaces.positions, workspaces.sorted_assignment_ids)

    torch.searchsorted(
        workspaces.padded_offsets[1:],
        workspaces.block_starts,
        right=True,
        out=workspaces.block_owners,
    )
    expert_ids.copy_(workspaces.block_owners)
    torch.ge(
        workspaces.block_starts,
        workspaces.padded_offsets[-1],
        out=workspaces.inactive_blocks,
    )
    expert_ids.masked_fill_(workspaces.inactive_blocks, -1)
    return sorted_token_ids, expert_ids, num_tokens_post_pad


def _canonical_post_pad(args: _ValidatedArgs) -> int:
    quotient, remainder = divmod(args.assignments, args.num_experts)
    high = _ceil_div(quotient + 1, args.block_size) * args.block_size
    low = _ceil_div(quotient, args.block_size) * args.block_size if quotient else 0
    return remainder * high + (args.num_experts - remainder) * low


def _semantic_ops(*, num_tokens: int, num_experts: int, top_k: int, block_size: int) -> int:
    """Nominal count/scan/group/pad operations for canonical routing.

    The estimate counts histogram increments, integer padding work, two prefix
    scans, comparison-sort work, assignment/padding writes, and block-owner
    searches. It intentionally does not model PyTorch's physical launches.
    """
    args = _validate_args(num_tokens, num_experts, top_k, block_size)
    sort_levels = max(1, (args.assignments - 1).bit_length())
    search_levels = max(1, args.num_experts.bit_length())
    post_pad = _canonical_post_pad(args)
    return (
        args.assignments
        + 3 * args.num_experts
        + 2 * max(args.num_experts - 1, 0)
        + args.assignments * sort_levels
        + args.assignments
        + post_pad
        + args.num_blocks * search_levels
    )


def _logical_bytes(*, num_tokens: int, num_experts: int, top_k: int, block_size: int) -> int:
    """Read int32 IDs and write all three production-capacity outputs once."""
    args = _validate_args(num_tokens, num_experts, top_k, block_size)
    return 4 * (args.assignments + args.capacity + args.num_blocks + 1)


def _check_correctness(torch: Any, operands: _Operands, args: _ValidatedArgs) -> None:
    from profiling.runners.moe.moe_align_block_size_reference import (
        moe_align_block_size_reference,
    )

    topk_before = operands.topk_ids.clone()
    expected = moe_align_block_size_reference(
        operands.topk_ids.detach().cpu(), args.num_experts, args.block_size
    )
    actual = _align_into(
        torch,
        operands.topk_ids,
        operands.sorted_token_ids,
        operands.expert_ids,
        operands.num_tokens_post_pad,
        operands.workspaces,
        num_experts=args.num_experts,
        block_size=args.block_size,
    )
    if operands.topk_ids.device.type == "cuda":
        torch.cuda.synchronize()
    for observed, wanted in zip(actual, expected):
        if not torch.equal(observed.cpu(), wanted):
            raise RuntimeError("Torch block alignment disagrees with the semantic reference")
    if not torch.equal(operands.topk_ids, topk_before):
        raise RuntimeError("Torch semantic helper mutated routing IDs")
    pointers = {tensor.data_ptr() for tensor in actual}
    if len(pointers) != 3 or operands.topk_ids.data_ptr() in pointers:
        raise RuntimeError("Torch alignment outputs must use fresh disjoint storage")


def profile_moe_align_block_size(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    block_size: int,
) -> ComputeMetrics:
    args = _validate_args(num_tokens, num_experts, top_k, block_size)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for moe_align_block_size") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for torch moe_align_block_size")
    try:
        operands = _build_operands(torch, args, device=torch.device("cuda"))
        _check_correctness(torch, operands, args)

        def kernel():
            return _align_into(
                torch,
                operands.topk_ids,
                operands.sorted_token_ids,
                operands.expert_ids,
                operands.num_tokens_post_pad,
                operands.workspaces,
                num_experts=args.num_experts,
                block_size=args.block_size,
            )

        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
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
            tflops=float(operations / elapsed / 1e12 if elapsed else 0),
            memory_bandwidth_gbps=float(logical_bytes / elapsed / 1e9 if elapsed else 0),
            energy_j=float(energy_j),
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError("torch moe_align_block_size ran out of CUDA memory") from exc
    except (RuntimeError, AssertionError) as exc:
        raise KernelLaunchFailed(str(exc)) from exc
