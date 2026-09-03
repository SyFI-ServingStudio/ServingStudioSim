"""Whole production FlashInfer NVFP4 fused-MoE profiler for SM100."""

from __future__ import annotations

import heapq
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

WEIGHT_FORMAT = "nvfp4_e2m1"
GROUP_SIZE = 16
ROUTING_METHODS = {"minimax2": 7}
_PREPARED_WEIGHT_CACHE: dict[tuple[str, int, int, int, int], tuple[Any, Any, Any, Any]] = {}
SGLANG_PDL_MAX_TOKENS = 8192


def _require_b200(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("NVFP4 MoE profiling requires CUDA")
    if tuple(torch.cuda.get_device_capability()) != (10, 0):
        raise ProfilerNotImplemented("NVFP4 MoE profiling requires SM100")


def _load_vllm_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        import vllm.model_executor.layers.fused_moe  # noqa: F401
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
            prepare_static_weights_for_trtllm_fp4_moe,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented("FlashInfer and instrumented vLLM are required") from exc

    def quantize(source: Any, input_scale: Any) -> tuple[Any, Any, None]:
        packed, scales = ops.scaled_fp4_quant(source, input_scale, is_sf_swizzled_layout=False)
        return packed, scales.view(torch.float8_e4m3fn).reshape(*packed.shape[:-1], -1), None

    def prepare(w1: Any, w2: Any, s1: Any, s2: Any, **dims: Any) -> tuple[Any, ...]:
        return prepare_static_weights_for_trtllm_fp4_moe(
            w1, w2, s1, s2, **dims, is_gated_activation=True
        )

    return torch, quantize, prepare


def _load_sglang_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        from flashinfer import fp4_quantize
        from sglang.srt.layers.quantization.utils import (
            prepare_static_weights_for_trtllm_fp4_moe,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented("the SGLang environment is required") from exc

    def quantize(source: Any, input_scale: Any) -> tuple[Any, Any, Any]:
        return _quantize_sglang_hidden(torch, fp4_quantize, source, input_scale)

    def prepare(w1: Any, w2: Any, s1: Any, s2: Any, **dims: Any) -> tuple[Any, ...]:
        return prepare_static_weights_for_trtllm_fp4_moe(w1, w2, s1, s2, **dims, is_gated=True)

    return torch, quantize, prepare


def _quantize_sglang_hidden(
    torch: Any,
    fp4_quantize: Any,
    source: Any,
    input_scale: Any,
) -> tuple[Any, Any, Any]:
    # The compressed-tensors W4A4 scheme used by this checkpoint sets
    # `use_per_token_activation=False`. Match SGLang's
    # `quantize_hidden_states_fp4` branch rather than its optional per-token
    # activation path.
    packed, scales = fp4_quantize(source, input_scale, GROUP_SIZE, False, False)
    tokens, hidden = source.shape
    return (
        packed.reshape(tokens, hidden // 2),
        scales.view(torch.float8_e4m3fn).reshape(tokens, hidden // GROUP_SIZE),
        None,
    )


def _load_runtime(stack: str) -> tuple[Any, Any, Any]:
    loaders = {"vllm": _load_vllm_runtime, "sglang": _load_sglang_runtime}
    if stack not in loaders:
        raise ValueError(f"unknown serving stack: {stack}")
    torch, quantize, prepare = loaders[stack]()
    _require_b200(torch)
    return torch, quantize, prepare


def _packed_weight(torch: Any, experts: int, n: int, k: int) -> Any:
    return torch.empty((experts, n, k // 2), dtype=torch.uint8, device="cuda")


def _prepared_weights(
    torch: Any,
    prepare: Any,
    *,
    stack: str,
    experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[Any, Any, Any, Any]:
    key = (stack, torch.cuda.current_device(), experts, hidden_size, intermediate_size)
    cached = _PREPARED_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    w1 = _packed_weight(torch, experts, 2 * intermediate_size, hidden_size)
    w2 = _packed_weight(torch, experts, hidden_size, intermediate_size)
    s1 = torch.ones(
        (experts, 2 * intermediate_size, hidden_size // GROUP_SIZE),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    s2 = torch.ones(
        (experts, hidden_size, intermediate_size // GROUP_SIZE),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    cached = prepare(
        w1,
        w2,
        s1,
        s2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
    )
    _PREPARED_WEIGHT_CACHE[key] = cached
    return cached


def _quantized_hidden(
    torch: Any,
    quantize: Any,
    num_tokens: int,
    hidden_size: int,
) -> tuple[Any, Any, Any]:
    source = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    input_scale = torch.ones((), dtype=torch.float32, device="cuda")
    return quantize(source, input_scale)


def _call_trtllm_fp4_moe(
    fn: Any,
    *,
    stack: str,
    per_token_scale: Any,
    kwargs: dict[str, Any],
) -> None:
    call_kwargs = dict(kwargs)
    if stack == "sglang":
        call_kwargs["per_token_scale"] = per_token_scale
    fn(**call_kwargs)


def _exact_topk_ids(
    *, num_tokens: int, top_k: int, per_expert_batches: tuple[int, ...]
) -> list[list[int]]:
    """Realize expert degrees as distinct top-k ids for every token."""

    if len(per_expert_batches) < top_k:
        raise ValueError("num_experts must be at least top_k")
    if any(batch < 0 for batch in per_expert_batches):
        raise ValueError("per_expert_batches cannot contain negative counts")
    if any(batch > num_tokens for batch in per_expert_batches):
        raise ValueError("one expert cannot receive more than one row per token")
    expected = num_tokens * top_k
    if sum(per_expert_batches) != expected:
        raise ValueError(f"per_expert_batches must sum to num_tokens*top_k ({expected})")

    heap = [(-batch, expert) for expert, batch in enumerate(per_expert_batches) if batch]
    heapq.heapify(heap)
    rows: list[list[int]] = []
    for _ in range(num_tokens):
        if len(heap) < top_k:
            raise ValueError("expert counts cannot form distinct top-k rows")
        selected = [heapq.heappop(heap) for _ in range(top_k)]
        rows.append([expert for _negative_count, expert in selected])
        for negative_count, expert in selected:
            if negative_count + 1 < 0:
                heapq.heappush(heap, (negative_count + 1, expert))
    if heap:
        raise ValueError("expert counts were not exhausted by top-k construction")
    return rows


def _validate_args(**kwargs: Any) -> dict[str, Any]:
    args = dict(kwargs)
    for name in (
        "num_tokens",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "num_local_experts",
        "top_k",
        "group_size",
        "n_group",
        "topk_group",
        "routed_scaling_numerator",
        "routed_scaling_denominator",
    ):
        args[name] = int(args[name])
        if args[name] <= 0:
            raise ValueError(f"{name} must be positive")
    args["per_expert_batches"] = tuple(int(value) for value in args["per_expert_batches"])
    if DType.from_value(args["input_dtype"]) is not DType.BF16:
        raise ValueError("NVFP4 fused MoE requires BF16 router/output precision")
    if args["weight_format"] != WEIGHT_FORMAT or args["group_size"] != GROUP_SIZE:
        raise ValueError("NVFP4 fused MoE requires nvfp4_e2m1 weights with group_size=16")
    if args["routing_method"] not in ROUTING_METHODS:
        raise ValueError(f"unsupported routing method: {args['routing_method']}")
    if args["hidden_size"] % 256 or args["intermediate_size"] % 256:
        raise ValueError("TRT-LLM NVFP4 dimensions must be divisible by 256")
    if len(args["per_expert_batches"]) != args["num_experts"]:
        raise ValueError("per_expert_batches must contain one count per global expert")
    if args["num_experts"] % args["num_local_experts"]:
        raise ValueError("num_local_experts must divide num_experts")
    _exact_topk_ids(
        num_tokens=args["num_tokens"],
        top_k=args["top_k"],
        per_expert_batches=args["per_expert_batches"],
    )
    return args


def _profile_nvfp4_fused_moe_sm100(
    *,
    stack: str,
    do_finalize: bool,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_local_experts: int,
    top_k: int,
    input_dtype: DType | str,
    weight_format: str,
    group_size: int,
    routing_method: str,
    n_group: int,
    topk_group: int,
    routed_scaling_numerator: int,
    routed_scaling_denominator: int,
    per_expert_batches: tuple[int, ...],
) -> ComputeMetrics:
    spec = dict(locals())
    spec.pop("stack")
    spec.pop("do_finalize")
    args = _validate_args(**spec)
    try:
        import flashinfer
        from flashinfer.autotuner import autotune

        torch, quantize, prepare = _load_runtime(stack)
        ids = _exact_topk_ids(
            num_tokens=args["num_tokens"],
            top_k=args["top_k"],
            per_expert_batches=args["per_expert_batches"],
        )
        routing_logits = torch.full(
            (args["num_tokens"], args["num_experts"]),
            -16.0,
            dtype=torch.bfloat16,
            device="cuda",
        )
        selected = torch.tensor(ids, dtype=torch.int64, device="cuda")
        priorities = torch.arange(args["top_k"], dtype=torch.bfloat16, device="cuda")
        routing_logits.scatter_(1, selected, (16.0 - priorities).expand_as(selected).contiguous())
        routing_bias = torch.zeros(args["num_experts"], dtype=torch.bfloat16, device="cuda")
        hidden, hidden_scale, per_token_scale = _quantized_hidden(
            torch, quantize, args["num_tokens"], args["hidden_size"]
        )
        w1, s1, w2, s2 = _prepared_weights(
            torch,
            prepare,
            stack=stack,
            experts=args["num_local_experts"],
            hidden_size=args["hidden_size"],
            intermediate_size=args["intermediate_size"],
        )
        expert_scale = torch.ones(args["num_local_experts"], dtype=torch.float32, device="cuda")
        output = (
            torch.empty(
                (args["num_tokens"], args["hidden_size"]),
                dtype=torch.bfloat16,
                device="cuda",
            )
            if do_finalize
            else None
        )
        routed_scale = args["routed_scaling_numerator"] / args["routed_scaling_denominator"]
        stack_kwargs: dict[str, Any] = (
            {
                "tune_max_num_tokens": 1 << (args["num_tokens"] - 1).bit_length(),
                "enable_pdl": args["num_tokens"] <= SGLANG_PDL_MAX_TOKENS,
            }
            if stack == "sglang"
            else {"enable_pdl": True}
        )

        def run_once() -> None:
            _call_trtllm_fp4_moe(
                flashinfer.fused_moe.trtllm_fp4_block_scale_moe,
                stack=stack,
                per_token_scale=per_token_scale,
                kwargs={
                    "routing_logits": routing_logits,
                    "routing_bias": routing_bias,
                    "hidden_states": hidden,
                    "hidden_states_scale": hidden_scale,
                    "gemm1_weights": w1,
                    "gemm1_weights_scale": s1,
                    "gemm1_bias": None,
                    "gemm1_alpha": None,
                    "gemm1_beta": None,
                    "gemm1_clamp_limit": None,
                    "gemm2_weights": w2,
                    "gemm2_weights_scale": s2,
                    "gemm2_bias": None,
                    "output1_scale_scalar": expert_scale,
                    "output1_scale_gate_scalar": expert_scale,
                    "output2_scale_scalar": expert_scale,
                    "num_experts": args["num_experts"],
                    "top_k": args["top_k"],
                    "n_group": args["n_group"],
                    "topk_group": args["topk_group"],
                    "intermediate_size": args["intermediate_size"],
                    "local_expert_offset": 0,
                    "local_num_experts": args["num_local_experts"],
                    "routed_scaling_factor": routed_scale,
                    "routing_method_type": ROUTING_METHODS[args["routing_method"]],
                    "do_finalize": do_finalize,
                    "activation_type": 3,
                    "output": output,
                    **stack_kwargs,
                },
            )

        # vLLM tunes this exact call. SGLang tunes a dummy precomputed-top-k
        # signature, so timing an extra autotune here would select a tactic its
        # production invocation does not use.
        if stack != "sglang":
            with autotune():
                run_once()
        else:
            run_once()
        torch.cuda.synchronize()
        time_ms = Timer.cupti(run_once, warmup=3, interval_union=True)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError("SM100 NVFP4 fused MoE ran out of memory") from exc
        raise KernelLaunchFailed(f"SM100 NVFP4 fused MoE failed: {exc}") from exc

    local_rows = sum(args["per_expert_batches"][: args["num_local_experts"]])
    h, i = args["hidden_size"], args["intermediate_size"]
    flops = 2 * local_rows * (h * 2 * i + i * h)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=time_ms,
        tflops=flops / elapsed_s / 1e12,
        memory_bandwidth_gbps=0.0,
        energy_j=energy_j,
    )


def profile_nvfp4_fused_moe_sm100(**kwargs: Any) -> ComputeMetrics:
    return _profile_nvfp4_fused_moe_sm100(stack="vllm", do_finalize=True, **kwargs)


def profile_nvfp4_fused_moe_deferred_finalize_sm100(**kwargs: Any) -> ComputeMetrics:
    return _profile_nvfp4_fused_moe_sm100(stack="sglang", do_finalize=False, **kwargs)


__all__ = [
    "profile_nvfp4_fused_moe_deferred_finalize_sm100",
    "profile_nvfp4_fused_moe_sm100",
]
