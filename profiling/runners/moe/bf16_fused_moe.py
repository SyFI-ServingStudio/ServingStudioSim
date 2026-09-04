"""Whole production FlashInfer TRT-LLM BF16 fused-MoE profiler for SM100."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
from profiling.runners.moe.exact_topk import exact_topk_ids

_BACKEND = "flashinfer_trtllm_sm100"
_GPU = "NVIDIA B200"
_ROUTING_METHODS = {"minimax2": 7}
_BLOCK = 128
# TRT-LLM's MiniMax2 routing kernel caps. Exceeding either is a launch
# failure inside FlashInfer, so reject it here where the shape is named.
_MAX_EXPERTS = 1024
_MAX_TOP_K = 32
_CORRECTNESS_ATOL = 0.025
_CORRECTNESS_RTOL = 0.05
_PREPARED_WEIGHT_CACHE: dict[tuple[int, int, int, int], tuple[Any, Any]] = {}
_CORRECTNESS_VERIFIED_DEVICES: set[int] = set()


@dataclass(frozen=True)
class _ValidatedArgs:
    num_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    num_local_experts: int
    top_k: int
    routing_method: str
    n_group: int
    topk_group: int
    routed_scaling_numerator: int
    routed_scaling_denominator: int
    per_expert_batches: tuple[int, ...]

    @property
    def routed_scaling_factor(self) -> float:
        return self.routed_scaling_numerator / self.routed_scaling_denominator


@dataclass(frozen=True)
class _Operands:
    routing_logits: Any
    routing_bias: Any
    hidden_states: Any
    gemm1_weights: Any
    gemm2_weights: Any


@dataclass(frozen=True)
class _Launch:
    callable_: Any
    operands: _Operands
    args: _ValidatedArgs

    def run_once(self) -> Any:
        return _invoke(self.callable_, self.operands, self.args)


def _validate_args(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_local_experts: int,
    top_k: int,
    dtype: DType | str,
    routing_method: str,
    n_group: int,
    topk_group: int,
    routed_scaling_numerator: int,
    routed_scaling_denominator: int,
    per_expert_batches: tuple[int, ...],
) -> _ValidatedArgs:
    integer_values = {
        "num_tokens": num_tokens,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_experts": num_experts,
        "num_local_experts": num_local_experts,
        "top_k": top_k,
        "n_group": n_group,
        "topk_group": topk_group,
        "routed_scaling_numerator": routed_scaling_numerator,
        "routed_scaling_denominator": routed_scaling_denominator,
    }
    normalized = {name: int(value) for name, value in integer_values.items()}
    if any(value <= 0 for value in normalized.values()):
        bad = next(name for name, value in normalized.items() if value <= 0)
        raise ValueError(f"{bad} must be positive")
    if DType.from_value(dtype) is not DType.BF16:
        raise ValueError("BF16 fused MoE requires dtype=bf16")
    if routing_method not in _ROUTING_METHODS:
        raise ValueError(f"unsupported routing method: {routing_method}")
    if normalized["n_group"] != 1 or normalized["topk_group"] != 1:
        raise ValueError("minimax2 profiling supports n_group=topk_group=1")
    if normalized["hidden_size"] % _BLOCK or normalized["intermediate_size"] % _BLOCK:
        raise ValueError("TRT-LLM BF16 dimensions must be divisible by 128")
    if normalized["num_experts"] > _MAX_EXPERTS:
        raise ValueError(f"TRT-LLM MiniMax2 supports at most {_MAX_EXPERTS} experts")
    if normalized["top_k"] > min(_MAX_TOP_K, normalized["num_experts"]):
        raise ValueError(f"TRT-LLM MiniMax2 supports top_k <= {_MAX_TOP_K}")
    if normalized["num_experts"] % normalized["num_local_experts"]:
        raise ValueError("num_local_experts must divide num_experts")
    batches = tuple(int(value) for value in per_expert_batches)
    if len(batches) != normalized["num_experts"]:
        raise ValueError("per_expert_batches must contain one count per global expert")
    exact_topk_ids(
        num_tokens=normalized["num_tokens"],
        top_k=normalized["top_k"],
        per_expert_batches=batches,
    )
    return _ValidatedArgs(
        **normalized,
        routing_method=routing_method,
        per_expert_batches=batches,
    )


def _logical_bytes(args: _ValidatedArgs) -> int:
    """Return useful algorithmic traffic for this rank's fused MoE call.

    Routed BF16 activations are counted once per local expert assignment, while
    W13/W2 bytes are counted only for experts that receive a local row.  This is
    the same logical-work convention used by the grouped-MoE profilers, not an
    estimate of physical HBM transactions or cache reuse.
    """

    local_batches = args.per_expert_batches[: args.num_local_experts]
    local_rows = sum(local_batches)
    active_experts = sum(batch > 0 for batch in local_batches)

    routing = 2 * args.num_tokens * args.num_experts + 2 * args.num_experts
    activations = 2 * local_rows * args.hidden_size
    # BF16 W13 is [2I, H] and W2 is [H, I].
    weights = active_experts * (
        4 * args.intermediate_size * args.hidden_size
        + 2 * args.hidden_size * args.intermediate_size
    )
    output = 2 * args.num_tokens * args.hidden_size
    return routing + activations + weights + output


def _require_b200(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("BF16 fused MoE profiling requires CUDA")
    device = torch.cuda.current_device()
    name = str(torch.cuda.get_device_name(device))
    capability = tuple(torch.cuda.get_device_capability(device))
    if name != _GPU or capability != (10, 0):
        raise ProfilerNotImplemented(
            f"{_BACKEND} is verified only on {_GPU}/SM100, got {name}/{capability}"
        )


def _load_runtime() -> tuple[Any, Any, Callable[[Any, Any], tuple[Any, Any]]]:
    try:
        import flashinfer
        import torch
        import vllm.model_executor.layers.fused_moe  # noqa: F401
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
            convert_moe_weights_to_flashinfer_trtllm_block_layout,
            swap_w13_to_w31,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "FlashInfer 0.6.12 and the pinned repository vLLM are required"
        ) from exc

    callable_ = flashinfer.fused_moe.trtllm_bf16_moe

    def prepare_weights(w13: Any, w2: Any) -> tuple[Any, Any]:
        # This is the same [w1;w3] -> [w3;w1] swap and BlockMajorK conversion
        # performed by UnquantizedFusedMoEMethod before serving.
        return convert_moe_weights_to_flashinfer_trtllm_block_layout({}, swap_w13_to_w31(w13), w2)

    return torch, callable_, prepare_weights


def _make_raw_weights(
    torch: Any,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    seed: int,
) -> tuple[Any, Any]:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(seed)
    w13 = torch.randn(
        (num_local_experts, 2 * intermediate_size, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.02)
    w2 = torch.randn(
        (num_local_experts, hidden_size, intermediate_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.02)
    return w13, w2


def _prepared_weights(
    torch: Any,
    prepare_weights: Callable[[Any, Any], tuple[Any, Any]],
    args: _ValidatedArgs,
) -> tuple[Any, Any]:
    key = (
        torch.cuda.current_device(),
        args.num_local_experts,
        args.hidden_size,
        args.intermediate_size,
    )
    cached = _PREPARED_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    raw = _make_raw_weights(
        torch,
        num_local_experts=args.num_local_experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        seed=20260831,
    )
    cached = prepare_weights(*raw)
    _PREPARED_WEIGHT_CACHE[key] = cached
    return cached


def _routing_inputs(torch: Any, args: _ValidatedArgs) -> tuple[Any, Any]:
    ids = exact_topk_ids(
        num_tokens=args.num_tokens,
        top_k=args.top_k,
        per_expert_batches=args.per_expert_batches,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    routing_logits = torch.full(
        (args.num_tokens, args.num_experts),
        -12.0,
        dtype=torch.bfloat16,
        device=device,
    )
    ids_tensor = torch.tensor(ids, dtype=torch.int64, device=device)
    priorities = torch.arange(args.top_k, 0, -1, dtype=torch.bfloat16, device=device).unsqueeze(0)
    routing_logits.scatter_(1, ids_tensor, priorities.expand(args.num_tokens, -1))
    routing_bias = torch.zeros(args.num_experts, dtype=torch.bfloat16, device=device)
    return routing_logits, routing_bias


def _prepare_launch(
    torch: Any,
    callable_: Any,
    prepare_weights: Callable[[Any, Any], tuple[Any, Any]],
    args: _ValidatedArgs,
) -> _Launch:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(20260830)
    hidden_states = torch.randn(
        (args.num_tokens, args.hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.1)
    routing_logits, routing_bias = _routing_inputs(torch, args)
    gemm1_weights, gemm2_weights = _prepared_weights(torch, prepare_weights, args)
    return _Launch(
        callable_=callable_,
        operands=_Operands(
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            hidden_states=hidden_states,
            gemm1_weights=gemm1_weights,
            gemm2_weights=gemm2_weights,
        ),
        args=args,
    )


def _invoke(callable_: Any, operands: _Operands, args: _ValidatedArgs) -> Any:
    # Keep this keyword set identical to pinned vLLM TrtLlmBf16Experts.apply.
    return callable_(
        routing_logits=operands.routing_logits,
        routing_bias=operands.routing_bias,
        hidden_states=operands.hidden_states,
        gemm1_weights=operands.gemm1_weights,
        gemm2_weights=operands.gemm2_weights,
        num_experts=args.num_experts,
        top_k=args.top_k,
        n_group=args.n_group,
        topk_group=args.topk_group,
        intermediate_size=args.intermediate_size,
        local_expert_offset=0,
        local_num_experts=args.num_local_experts,
        routed_scaling_factor=args.routed_scaling_factor,
        routing_method_type=_ROUTING_METHODS[args.routing_method],
    )


def _validate_output(torch: Any, output: Any, args: _ValidatedArgs) -> None:
    if tuple(output.shape) != (args.num_tokens, args.hidden_size):
        raise AssertionError("trtllm_bf16_moe returned an invalid output shape")
    if output.dtype is not torch.bfloat16 or not output.is_cuda:
        raise AssertionError("trtllm_bf16_moe output must be CUDA BF16")
    if not torch.isfinite(output).all():
        raise AssertionError("trtllm_bf16_moe output must be finite")


def _torch_reference(
    torch: Any,
    operands: _Operands,
    raw_w13: Any,
    raw_w2: Any,
    args: _ValidatedArgs,
) -> Any:
    import torch.nn.functional as functional

    logits = operands.routing_logits.float()
    unbiased = torch.sigmoid(logits)
    selected = torch.topk(unbiased + operands.routing_bias.float(), args.top_k, dim=-1).indices
    selected_weights = torch.gather(unbiased, 1, selected)
    selected_weights = selected_weights / (selected_weights.sum(dim=-1, keepdim=True) + 1e-20)
    result = torch.zeros(
        (args.num_tokens, args.hidden_size), dtype=torch.float32, device=logits.device
    )
    hidden = operands.hidden_states.float()
    for token in range(args.num_tokens):
        for slot in range(args.top_k):
            expert = int(selected[token, slot])
            gate_up = functional.linear(hidden[token], raw_w13[expert].float())
            gate, up = gate_up.chunk(2, dim=-1)
            activated = functional.silu(gate) * up
            expert_output = functional.linear(activated, raw_w2[expert].float())
            result[token].add_(expert_output, alpha=float(selected_weights[token, slot]))
    return result.to(torch.bfloat16)


def _check_small_correctness(
    torch: Any,
    callable_: Any,
    prepare_weights: Callable[[Any, Any], tuple[Any, Any]],
) -> None:
    device_index = torch.cuda.current_device()
    if device_index in _CORRECTNESS_VERIFIED_DEVICES:
        return
    args = _validate_args(
        num_tokens=4,
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        num_local_experts=8,
        top_k=2,
        dtype=DType.BF16,
        routing_method="minimax2",
        n_group=1,
        topk_group=1,
        routed_scaling_numerator=1,
        routed_scaling_denominator=1,
        per_expert_batches=(1,) * 8,
    )
    raw_w13, raw_w2 = _make_raw_weights(
        torch,
        num_local_experts=args.num_local_experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        seed=20260829,
    )
    gemm1_weights, gemm2_weights = prepare_weights(raw_w13, raw_w2)
    routing_logits, routing_bias = _routing_inputs(torch, args)
    generator = torch.Generator(device=routing_logits.device).manual_seed(20260828)
    hidden_states = torch.randn(
        (args.num_tokens, args.hidden_size),
        dtype=torch.bfloat16,
        device=routing_logits.device,
        generator=generator,
    ).mul_(0.1)
    operands = _Operands(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
    )
    expected = _torch_reference(torch, operands, raw_w13, raw_w2, args)
    observed = _invoke(callable_, operands, args)
    torch.cuda.synchronize()
    _validate_output(torch, observed, args)
    torch.testing.assert_close(
        observed,
        expected,
        atol=_CORRECTNESS_ATOL,
        rtol=_CORRECTNESS_RTOL,
    )
    _CORRECTNESS_VERIFIED_DEVICES.add(device_index)


def profile_bf16_fused_moe_sm100(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_local_experts: int,
    top_k: int,
    dtype: DType | str,
    routing_method: str,
    n_group: int,
    topk_group: int,
    routed_scaling_numerator: int,
    routed_scaling_denominator: int,
    per_expert_batches: tuple[int, ...],
) -> ComputeMetrics:
    args = _validate_args(**locals())
    try:
        torch, callable_, prepare_weights = _load_runtime()
        _require_b200(torch)
        _check_small_correctness(torch, callable_, prepare_weights)
        launch = _prepare_launch(torch, callable_, prepare_weights, args)

        # Materialize FlashInfer tactics and validate the production output before
        # entering either the CUPTI or energy timing windows.
        output = launch.run_once()
        torch.cuda.synchronize()
        _validate_output(torch, output, args)
        time_ms = Timer.cupti(
            launch.run_once,
            kernel_name=None,
            interval_union=True,
        )
        energy_j = Energy.perf(launch.run_once, per_iter_time_ms=time_ms)
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError("SM100 BF16 fused MoE ran out of memory") from exc
        raise KernelLaunchFailed(f"SM100 BF16 fused MoE failed: {exc}") from exc

    local_rows = sum(args.per_expert_batches[: args.num_local_experts])
    flops = (
        2
        * local_rows
        * (
            args.hidden_size * 2 * args.intermediate_size
            + args.intermediate_size * args.hidden_size
        )
    )
    elapsed_s = time_ms / 1000.0
    logical_bytes = _logical_bytes(args)
    return ComputeMetrics(
        time_ms=time_ms,
        tflops=flops / elapsed_s / 1e12,
        memory_bandwidth_gbps=logical_bytes / elapsed_s / 1e9,
        energy_j=energy_j,
    )


__all__ = ["profile_bf16_fused_moe_sm100"]
