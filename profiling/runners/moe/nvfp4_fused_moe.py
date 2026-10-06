"""Whole production FlashInfer NVFP4 fused-MoE profiler for SM100."""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.autotune_cache import autotune_cached
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.moe.exact_topk import exact_topk_ids
from profiling.runners.moe.fp8_block_fused_moe import forced_routing_logits

WEIGHT_FORMAT = DType.NVFP4_E2M1
GROUP_SIZE = 16
# flashinfer.fused_moe.RoutingMethodType. Both are sigmoid + correction-bias
# top-k with renormalized weights; DeepSeekV3 also applies the routed scale.
# Ungrouped DeepSeekV3 runs the same routingCustom kernel as MiniMax2
# (trtllm_fused_moe_runner.cu, `DeepSeekV3 && nGroup <= 1`).
ROUTING_METHODS = {"minimax2": 7, "deepseek_v3": 2}
# Bytes per router logit and per correction-bias element that vLLM passes.
# MiniMax2 rows keep the BF16 router of their model. DeepSeekV3 rows are for a
# model with `moe_router_dtype: float32`: GateLinear emits FP32 logits
# (router_logits_dtype = gate.out_dtype) and e_score_correction_bias is an FP32
# parameter, both passed to the kernel unchanged.
ROUTING_ELEMENT_BYTES = {"minimax2": 2, "deepseek_v3": 4}
# The SGLang deferred-finalize backend is only measured for MiniMax2.
SGLANG_ROUTING_METHODS = frozenset({"minimax2"})
_PREPARED_WEIGHT_CACHE: dict[tuple[str, int, int, int, int], tuple[Any, Any, Any, Any]] = {}
SGLANG_PDL_MAX_TOKENS = 8192


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
    return loaders[stack]()


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


def vllm_tune_max_num_tokens(num_tokens: int) -> int:
    """FlashInfer's tuning bound for both of vLLM's TRT-LLM NVFP4 MoE calls.

    vLLM passes ``fi_moe_largest_bucket``, max(max_num_batched_tokens * dp_size,
    8192), to the monolithic call, and that capped at the chunk size to the
    precomputed-routing call (``trtllm_nvfp4_moe.py``). Neither the batch budget
    nor the DP size is a kernel argument, so use the smallest power of two that
    covers this call and is at least 8192; the bucket chosen for num_tokens is
    the same for any bound at or above it.
    """
    return max(8192, 1 << (num_tokens - 1).bit_length())


def _precomputed_routing_kwargs(torch: Any, ids: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Routing arguments of vLLM's TrtLlmNvFp4ExpertsModular call: fp32 weights
    and int32 IDs selected before dispatch, with the kernel's routing stage idle
    (trtllm_nvfp4_moe.py `_invoke_kernel`)."""
    topk_ids = torch.tensor(ids, dtype=torch.int32, device="cuda")
    topk_weights = torch.full(
        topk_ids.shape, 1.0 / args["top_k"], dtype=torch.float32, device="cuda"
    )
    return {
        "topk_ids": (topk_ids, topk_weights),
        "routing_bias": None,
        "n_group": 0,
        "topk_group": 0,
        "routed_scaling_factor": None,
        "routing_method_type": 1,
    }


def tuning_label(args: dict[str, Any], stack: str, precomputed_routing: bool) -> str:
    """Name of the persisted tactic file for this call's runner configuration.

    FlashInfer's file key for ``trtllm_fp4_block_scale_moe`` holds only the
    custom-op name, the runner class and the bucketed input shapes (output,
    logits, top-k ids/weights, hidden states and scales); ``MoERunner`` adds no
    ``get_cache_key_extras``. The local expert count, the intermediate size and,
    for the routed call, the global expert count shape the GEMMs but are absent
    from those shapes, so EP4 and TP8 rows of one model would share a key and
    the second would replay the first one's tactic. vLLM never sees that: it
    tunes in-process, where the key also hashes the runner. One file per
    configuration restores that separation. The routing method selects a
    different routing kernel in the monolithic call, so it names the file too.
    """

    label = f"nvfp4_fused_moe.{stack}" + (
        ".routed" if precomputed_routing else f".{args['routing_method']}"
    )
    return (
        f"{label}.e{args['num_experts']}.l{args['num_local_experts']}"
        f".h{args['hidden_size']}.i{args['intermediate_size']}"
    )


def _forced_routing(
    torch: Any, ids: list[list[int]], args: dict[str, Any], device: Any = "cuda"
) -> tuple[Any, Any]:
    """Router logits and zero correction bias whose top-k is exactly ``ids``.

    Both are in the dtype vLLM passes for the routing method (see
    ``ROUTING_ELEMENT_BYTES``). With a zero bias, sigmoid + top-k picks the
    distinct high logits regardless of the routed scale.
    """

    if args["routing_method"] == "deepseek_v3":
        logits = forced_routing_logits(torch, ids, args["num_experts"], device)
        return logits, torch.zeros(args["num_experts"], dtype=torch.float32, device=device)
    logits = torch.full(
        (args["num_tokens"], args["num_experts"]),
        -16.0,
        dtype=torch.bfloat16,
        device=device,
    )
    selected = torch.tensor(ids, dtype=torch.int64, device=device)
    priorities = torch.arange(args["top_k"], dtype=torch.bfloat16, device=device)
    logits.scatter_(1, selected, (16.0 - priorities).expand_as(selected).contiguous())
    return logits, torch.zeros(args["num_experts"], dtype=torch.bfloat16, device=device)


def _validate_args(*, stack: str = "vllm", **kwargs: Any) -> dict[str, Any]:
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
    if stack == "sglang" and args["routing_method"] not in SGLANG_ROUTING_METHODS:
        raise ValueError(f"unsupported routing method for SGLang: {args['routing_method']}")
    # Grouped DeepSeekV3 routing only admits experts from the best topk_group
    # groups, so forced logits could not realize an arbitrary histogram.
    if args["routing_method"] == "deepseek_v3" and (
        args["n_group"] != 1 or args["topk_group"] != 1
    ):
        raise ValueError("only ungrouped deepseek_v3 routing (n_group=topk_group=1) is supported")
    if args["hidden_size"] % 256 or args["intermediate_size"] % 256:
        raise ValueError("TRT-LLM NVFP4 dimensions must be divisible by 256")
    if len(args["per_expert_batches"]) != args["num_experts"]:
        raise ValueError("per_expert_batches must contain one count per global expert")
    if args["num_experts"] % args["num_local_experts"]:
        raise ValueError("num_local_experts must divide num_experts")
    exact_topk_ids(
        num_tokens=args["num_tokens"],
        top_k=args["top_k"],
        per_expert_batches=args["per_expert_batches"],
    )
    return args


def _logical_bytes(
    args: dict[str, Any], *, do_finalize: bool, precomputed_routing: bool = False
) -> int:
    """Return useful algorithmic traffic for this rank's fused MoE call.

    Routed activations are counted once per local expert assignment and expert
    weights/scales only for experts with at least one local row. This matches the
    logical-work convention used by the other grouped-MoE profilers; it is
    deliberately not a claim about physical HBM transactions or cache reuse.

    The output term follows the finalize mode, because the two production
    dispatches do not produce the same thing. A finalized call writes one row per
    input token; a deferred call writes one unfinalized row per local expert
    assignment and leaves the combine to `moe_finalize_fuse_shared`. Its small
    `expert_weights` / `expanded_idx_to_permuted_idx` side outputs are excluded:
    FlashInfer does not document their layout, and at production shapes they are
    three orders of magnitude below the activation rows.
    """

    local_batches = args["per_expert_batches"][: args["num_local_experts"]]
    local_rows = sum(local_batches)
    active_experts = sum(batch > 0 for batch in local_batches)
    tokens = args["num_tokens"]
    hidden = args["hidden_size"]
    intermediate = args["intermediate_size"]
    experts = args["num_experts"]

    # Router logits + correction bias at the router's element size, or fp32
    # weights + int32 IDs already selected.
    routing_element = ROUTING_ELEMENT_BYTES[args["routing_method"]]
    routing = (
        8 * tokens * args["top_k"]
        if precomputed_routing
        else routing_element * (tokens * experts + experts)
    )
    # Packed FP4 routed activations + one FP8 scale per group of GROUP_SIZE.
    activations = local_rows * (hidden // 2 + hidden // GROUP_SIZE)
    # W13 and W2 packed FP4 weights and their FP8 group scales. W13 is the fused
    # gate/up weight, so its packed size is 2 * intermediate * hidden / 2.
    weights = active_experts * (
        intermediate * hidden
        + intermediate * hidden // 8
        + hidden * intermediate // 2
        + hidden * intermediate // GROUP_SIZE
    )
    # Three FP32 per-expert scaling vectors consumed by the production call.
    expert_scales = 3 * active_experts * 4
    output = 2 * hidden * (tokens if do_finalize else local_rows)
    return routing + activations + weights + expert_scales + output


def _profile_nvfp4_fused_moe_sm100(
    *,
    stack: str,
    do_finalize: bool,
    precomputed_routing: bool = False,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_local_experts: int,
    top_k: int,
    input_dtype: DType | str,
    weight_format: DType | str,
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
    spec.pop("precomputed_routing")
    args = _validate_args(stack=stack, **spec)
    try:
        import flashinfer
        from flashinfer.autotuner import autotune

        torch, quantize, prepare = _load_runtime(stack)
        ids = exact_topk_ids(
            num_tokens=args["num_tokens"],
            top_k=args["top_k"],
            per_expert_batches=args["per_expert_batches"],
        )
        routing_logits, routing_bias = _forced_routing(torch, ids, args)
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
            else {
                "enable_pdl": True,
                "tune_max_num_tokens": vllm_tune_max_num_tokens(args["num_tokens"]),
            }
        )
        if precomputed_routing:
            routing_kwargs = _precomputed_routing_kwargs(torch, ids, args)
            # vLLM's modular call leaves enable_pdl at FlashInfer's default.
            stack_kwargs = {"tune_max_num_tokens": vllm_tune_max_num_tokens(args["num_tokens"])}
            fused_moe_fn = flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe
        else:
            routing_kwargs = {
                "routing_logits": routing_logits,
                "routing_bias": routing_bias,
                "n_group": args["n_group"],
                "topk_group": args["topk_group"],
                "routed_scaling_factor": routed_scale,
                "routing_method_type": ROUTING_METHODS[args["routing_method"]],
            }
            fused_moe_fn = flashinfer.fused_moe.trtllm_fp4_block_scale_moe

        def run_once() -> None:
            _call_trtllm_fp4_moe(
                fused_moe_fn,
                stack=stack,
                per_token_scale=per_token_scale,
                kwargs={
                    **routing_kwargs,
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
                    "intermediate_size": args["intermediate_size"],
                    "local_expert_offset": 0,
                    "local_num_experts": args["num_local_experts"],
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
            with autotune_cached(autotune, tuning_label(args, stack, precomputed_routing)):
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
        memory_bandwidth_gbps=_logical_bytes(
            args, do_finalize=do_finalize, precomputed_routing=precomputed_routing
        )
        / elapsed_s
        / 1e9,
        energy_j=energy_j,
    )


def profile_nvfp4_fused_moe_sm100(**kwargs: Any) -> ComputeMetrics:
    return _profile_nvfp4_fused_moe_sm100(stack="vllm", do_finalize=True, **kwargs)


def profile_nvfp4_fused_moe_routed_sm100(**kwargs: Any) -> ComputeMetrics:
    return _profile_nvfp4_fused_moe_sm100(
        stack="vllm", do_finalize=True, precomputed_routing=True, **kwargs
    )


def profile_nvfp4_fused_moe_deferred_finalize_sm100(**kwargs: Any) -> ComputeMetrics:
    return _profile_nvfp4_fused_moe_sm100(stack="sglang", do_finalize=False, **kwargs)


__all__ = [
    "profile_nvfp4_fused_moe_deferred_finalize_sm100",
    "profile_nvfp4_fused_moe_routed_sm100",
    "profile_nvfp4_fused_moe_sm100",
]
