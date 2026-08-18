"""Multi-launch Torch semantic baseline for fused MoE top-k selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
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
    dtype: DType


@dataclass(frozen=True)
class _Workspaces:
    probabilities: Any
    row_reduction: Any
    selected_long: Any
    token_indices: Any
    slot_offsets: Any


@dataclass(frozen=True)
class _Operands:
    logits: Any
    weights: Any
    expert_ids: Any
    source_indices: Any
    workspaces: _Workspaces


def _exact_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an exact integer, got {value!r}")
    return value


def _validate_args(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    dtype: DType | str,
) -> _ValidatedArgs:
    num_tokens = _exact_int("num_tokens", num_tokens)
    num_experts = _exact_int("num_experts", num_experts)
    top_k = _exact_int("top_k", top_k)
    if num_tokens <= 0 or num_experts <= 0:
        raise ValueError("num_tokens and num_experts must be positive")
    if not 1 <= top_k <= num_experts:
        raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")
    if num_tokens * top_k - 1 > _INT32_MAX:
        raise ValueError("source indices exceed the int32 range")
    dtype = DType.from_value(dtype)
    if dtype is not DType.BF16:
        raise ValueError(f"torch moe_fused_topk requires dtype=bf16, got {dtype.value}")
    return _ValidatedArgs(num_tokens, num_experts, top_k, dtype)


def _operand_shapes(args: _ValidatedArgs) -> dict[str, tuple[int, ...]]:
    output = (args.num_tokens, args.top_k)
    return {
        "logits": (args.num_tokens, args.num_experts),
        "weights": output,
        "expert_ids": output,
        "source_indices": output,
        "probabilities": (args.num_tokens, args.num_experts),
        "row_reduction": (args.num_tokens, 1),
        "selected_long": output,
        "token_indices": (args.num_tokens, 1),
        "slot_offsets": (1, args.top_k),
    }


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    shapes = _operand_shapes(args)
    # Integer-valued BF16 logits are unique for Qwen's E=256 and avoid tie
    # behavior in the measured baseline. Tests exercise ties independently.
    expert_axis = torch.arange(args.num_experts, dtype=torch.float32, device=device)
    token_axis = torch.arange(args.num_tokens, dtype=torch.float32, device=device).unsqueeze(1)
    logits = (expert_axis.unsqueeze(0) - token_axis / 512).to(torch.bfloat16).contiguous()
    weights = torch.empty(shapes["weights"], dtype=torch.float32, device=device)
    expert_ids = torch.empty(shapes["expert_ids"], dtype=torch.int32, device=device)
    source_indices = torch.empty(shapes["source_indices"], dtype=torch.int32, device=device)
    workspaces = _Workspaces(
        probabilities=torch.empty(shapes["probabilities"], dtype=torch.float32, device=device),
        row_reduction=torch.empty(shapes["row_reduction"], dtype=torch.float32, device=device),
        selected_long=torch.empty(shapes["selected_long"], dtype=torch.int64, device=device),
        token_indices=torch.arange(args.num_tokens, dtype=torch.int32, device=device).unsqueeze(1),
        slot_offsets=(
            torch.arange(args.top_k, dtype=torch.int32, device=device).unsqueeze(0)
            * args.num_tokens
        ),
    )
    return _Operands(logits, weights, expert_ids, source_indices, workspaces)


def _fused_topk_into(
    torch: Any,
    logits: Any,
    weights: Any,
    expert_ids: Any,
    source_indices: Any,
    workspaces: _Workspaces,
) -> tuple[Any, Any, Any]:
    """Overwrite preallocated outputs with the production semantic ordering."""
    probabilities = workspaces.probabilities
    probabilities.copy_(logits)
    torch.amax(probabilities, dim=1, keepdim=True, out=workspaces.row_reduction)
    probabilities.sub_(workspaces.row_reduction).exp_()
    torch.sum(probabilities, dim=1, keepdim=True, out=workspaces.row_reduction)
    probabilities.div_(workspaces.row_reduction)

    for slot in range(weights.shape[1]):
        torch.max(
            probabilities,
            dim=1,
            out=(weights[:, slot], workspaces.selected_long[:, slot]),
        )
        expert_ids[:, slot].copy_(workspaces.selected_long[:, slot])
        probabilities.scatter_(
            1,
            workspaces.selected_long[:, slot : slot + 1],
            -float("inf"),
        )

    torch.sum(weights, dim=1, keepdim=True, out=workspaces.row_reduction)
    weights.div_(workspaces.row_reduction)
    torch.add(workspaces.token_indices, workspaces.slot_offsets, out=source_indices)
    return weights, expert_ids, source_indices


def _semantic_flops(*, num_tokens: int, num_experts: int, top_k: int) -> int:
    """Nominal semantic ops, including comparisons and one op per exp.

    Per row: stable softmax costs ``5E-2`` (max, subtract, exp, sum,
    divide); iterative top-k costs ``K(E-1)-K(K-1)/2`` comparisons; and
    selected renormalization costs ``2K-1``. This excludes PyTorch's physical
    multi-launch/workspace traffic.
    """
    topk_comparisons = top_k * (num_experts - 1) - top_k * (top_k - 1) // 2
    return num_tokens * (5 * num_experts - 2 + topk_comparisons + 2 * top_k - 1)


def _logical_bytes(*, num_tokens: int, num_experts: int, top_k: int) -> int:
    """Read BF16 logits and write FP32 weights plus two int32 outputs once."""
    return 2 * num_tokens * num_experts + 12 * num_tokens * top_k


def _check_correctness(torch: Any, operands: _Operands) -> None:
    from profiling.runners.moe.moe_fused_topk_reference import moe_fused_topk_reference

    logits_before = operands.logits.clone()
    expected = moe_fused_topk_reference(operands.logits.detach().cpu(), operands.weights.shape[1])
    actual = _fused_topk_into(
        torch,
        operands.logits,
        operands.weights,
        operands.expert_ids,
        operands.source_indices,
        operands.workspaces,
    )
    if operands.logits.device.type == "cuda":
        torch.cuda.synchronize()
    torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=1e-5, atol=1e-6)
    if not torch.equal(actual[1].cpu(), expected[1]):
        raise RuntimeError("Torch expert IDs disagree with the semantic reference")
    if not torch.equal(actual[2].cpu(), expected[2]):
        raise RuntimeError("Torch source indices disagree with the semantic reference")
    if not torch.equal(operands.logits, logits_before):
        raise RuntimeError("Torch semantic helper mutated logits")


def profile_moe_fused_topk(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    args = _validate_args(num_tokens, num_experts, top_k, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for moe_fused_topk") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for torch moe_fused_topk")
    try:
        operands = _build_operands(torch, args, device=torch.device("cuda"))
        _check_correctness(torch, operands)

        def kernel():
            return _fused_topk_into(
                torch,
                operands.logits,
                operands.weights,
                operands.expert_ids,
                operands.source_indices,
                operands.workspaces,
            )

        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        flops = _semantic_flops(
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            top_k=args.top_k,
        )
        logical_bytes = _logical_bytes(
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            top_k=args.top_k,
        )
        elapsed = time_ms / 1000
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(flops / elapsed / 1e12 if elapsed else 0),
            memory_bandwidth_gbps=float(logical_bytes / elapsed / 1e9 if elapsed else 0),
            energy_j=float(energy_j),
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError("torch moe_fused_topk ran out of CUDA memory") from exc
    except (RuntimeError, AssertionError) as exc:
        raise KernelLaunchFailed(str(exc)) from exc
