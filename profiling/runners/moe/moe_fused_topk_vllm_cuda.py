"""One-launch vLLM CUDA runner for Qwen MoE fused top-k selection.

The timed callable is exactly one ``fused_topk`` wrapper invocation. CUPTI
selects the ordinary-softmax BF16 E256/K8 ``topkGating`` specialization; router
GEMM, token alignment, expert computation, and final weighting are excluded.
The reported FLOPs and bytes are semantic/logical and are not physical CUDA
work or traffic. Energy includes the wrapper's three fresh output allocations.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.moe.moe_fused_topk_torch import (
    _logical_bytes,
    _semantic_flops,
    _ValidatedArgs,
)
from profiling.runners.moe.moe_fused_topk_torch import (
    _validate_args as _validate_semantic_args,
)

_BACKEND = "moe_fused_topk:vllm_cuda"
_MODULE = "vllm.model_executor.layers.fused_moe.router.fused_topk_router"
_CALLABLE = "fused_topk"
_GPU = "NVIDIA H200"
_NUM_EXPERTS = 256
_TOP_K = 8
_QWEN_TOKENS = 128
_GUARD_TOKENS = 8
_ATOL = _RTOL = 1e-5
# Mangled prefix observed for the selected E256/K8 BF16/int32 ordinary-softmax
# specialization. ``ScoringFuncE0`` is SCORING_SOFTMAX. This excludes the
# softplus/sqrt and generic moeSoftmax/moeTopK implementations.
_KERNEL_NAME = (
    "_ZN4vllm3moe10topkGatingILi8ELi256ELi4ELi16ELi32Ei13__nv_bfloat16LNS0_11ScoringFuncE0EEEv"
)


@dataclass(frozen=True)
class _Operands:
    logits: Any
    hidden_states: Any


def _validate_args(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    dtype: DType | str,
) -> _ValidatedArgs:
    args = _validate_semantic_args(num_tokens, num_experts, top_k, dtype)
    if args.num_experts != _NUM_EXPERTS:
        raise ValueError(f"{_BACKEND} requires num_experts=256")
    if args.top_k != _TOP_K:
        raise ValueError(f"{_BACKEND} requires top_k=8")
    return args


def _guard_args(args: _ValidatedArgs) -> _ValidatedArgs:
    """Preserve exact Qwen; otherwise use a bounded witness with the same E/K."""
    if args.num_tokens == _QWEN_TOKENS:
        return args
    return replace(args, num_tokens=min(args.num_tokens, _GUARD_TOKENS))


def _operand_shapes(args: _ValidatedArgs) -> dict[str, tuple[int, ...]]:
    return {
        "logits": (args.num_tokens, args.num_experts),
        "hidden_states": (args.num_tokens, 1),
        "weights": (args.num_tokens, args.top_k),
        "expert_ids": (args.num_tokens, args.top_k),
        "source_indices": (args.num_tokens, args.top_k),
    }


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    # Bounded, non-tied BF16 logits make the correctness check deterministic.
    expert = torch.arange(args.num_experts, dtype=torch.float32, device=device)
    token = torch.arange(args.num_tokens, dtype=torch.float32, device=device).unsqueeze(1)
    logits = (expert.unsqueeze(0) - 127.0) / 16.0 - token / 512.0
    logits = logits.to(torch.bfloat16).contiguous()
    hidden_states = torch.zeros(
        _operand_shapes(args)["hidden_states"], dtype=torch.bfloat16, device=device
    ).contiguous()
    return _Operands(logits=logits, hidden_states=hidden_states)


def _validate_operands(
    torch: Any,
    operands: _Operands,
    args: _ValidatedArgs,
    *,
    require_cuda: bool = True,
) -> None:
    shapes = _operand_shapes(args)
    devices = set()
    for name in ("logits", "hidden_states"):
        tensor = getattr(operands, name)
        if tuple(tensor.shape) != shapes[name] or tensor.dtype is not torch.bfloat16:
            raise ValueError(f"invalid {name} shape/dtype")
        if (require_cuda and not tensor.is_cuda) or not tensor.is_contiguous():
            raise ValueError(f"{name} must be packed contiguous CUDA BF16 storage")
        if tensor.stride(-1) != 1 or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be packed and finite")
        devices.add(tensor.device)
    if len(devices) != 1:
        raise ValueError("logits and hidden_states must share one device")


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


def _invoke(torch: Any, callable_: Any, operands: _Operands) -> tuple[Any, Any, Any]:
    return callable_(
        hidden_states=operands.hidden_states,
        gating_output=operands.logits,
        topk=_TOP_K,
        renormalize=True,
        indices_type=torch.int32,
        scoring_func="softmax",
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
        raise AssertionError("fused_topk must return weights, expert IDs, and source indices")
    shapes = _operand_shapes(args)
    expected = (
        ("weights", torch.float32),
        ("expert_ids", torch.int32),
        ("source_indices", torch.int32),
    )
    for output, (name, dtype) in zip(outputs, expected, strict=True):
        if tuple(output.shape) != shapes[name] or output.dtype is not dtype:
            raise AssertionError(f"invalid {name} shape/dtype")
        if not output.is_contiguous() or not torch.isfinite(output).all():
            raise AssertionError(f"{name} must be finite contiguous storage")
        if output.device != operands.logits.device:
            raise AssertionError(f"{name} must remain on the input device")
    tensors = (*outputs, operands.logits, operands.hidden_states)
    for index, left in enumerate(tensors):
        if any(_shares(left, right) for right in tensors[index + 1 :]):
            raise AssertionError("inputs and outputs must use disjoint storage")
    ones = torch.ones(args.num_tokens, device=outputs[0].device)
    if not torch.allclose(outputs[0].sum(dim=1), ones, rtol=_RTOL, atol=_ATOL):
        raise AssertionError("selected weights must sum to one")
    if not ((outputs[1] >= 0) & (outputs[1] < args.num_experts)).all():
        raise AssertionError("expert IDs are out of range")
    token = torch.arange(args.num_tokens, dtype=torch.int32, device=outputs[2].device)
    slot = torch.arange(args.top_k, dtype=torch.int32, device=outputs[2].device)
    if not torch.equal(outputs[2], token[:, None] + slot[None, :] * args.num_tokens):
        raise AssertionError("source indices do not use slot-major token order")


def _check_correctness(
    torch: Any,
    callable_: Any,
    operands: _Operands,
    args: _ValidatedArgs,
    *,
    synchronize: Any | None = None,
) -> None:
    from profiling.runners.moe.moe_fused_topk_reference import moe_fused_topk_reference

    _validate_operands(torch, operands, args, require_cuda=operands.logits.is_cuda)
    logits_before = operands.logits.clone()
    hidden_before = operands.hidden_states.clone()
    expected = moe_fused_topk_reference(logits_before.cpu(), args.top_k)
    first = _invoke(torch, callable_, operands)
    (synchronize or (torch.cuda.synchronize if operands.logits.is_cuda else lambda: None))()
    _validate_outputs(torch, first, operands, args)
    torch.testing.assert_close(first[0].cpu(), expected[0], rtol=_RTOL, atol=_ATOL)
    if not torch.equal(first[1].cpu(), expected[1]):
        raise AssertionError("vLLM expert IDs disagree with the semantic reference")
    if not torch.equal(first[2].cpu(), expected[2]):
        raise AssertionError("vLLM source indices disagree with the semantic reference")

    second = _invoke(torch, callable_, operands)
    (synchronize or (torch.cuda.synchronize if operands.logits.is_cuda else lambda: None))()
    _validate_outputs(torch, second, operands, args)
    torch.testing.assert_close(second[0], first[0], rtol=0, atol=0)
    if not torch.equal(second[1], first[1]) or not torch.equal(second[2], first[2]):
        raise AssertionError("vLLM fused_topk outputs must be deterministic")
    if any(_shares(left, right) for left in first for right in second):
        raise AssertionError("each wrapper invocation must allocate fresh outputs")
    if not torch.equal(operands.logits, logits_before) or not torch.equal(
        operands.hidden_states, hidden_before
    ):
        raise AssertionError("vLLM fused_topk mutated an input")


def profile_moe_fused_topk_vllm_cuda(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    args = _validate_args(num_tokens, num_experts, top_k, dtype)
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
            return _invoke(torch, callable_, operands)

        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        # Energy covers the wrapper call, including its three fresh allocations.
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

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
        energy_j=float(energy_j),
        tflops=float(flops / elapsed / 1e12 if elapsed else 0),
        memory_bandwidth_gbps=float(logical_bytes / elapsed / 1e9 if elapsed else 0),
    )
