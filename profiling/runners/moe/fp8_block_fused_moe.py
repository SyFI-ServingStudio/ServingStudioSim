"""Whole FlashInfer TRT-LLM DeepSeek-FP8 block-scale fused MoE on SM100.

The timed callable is ``flashinfer.fused_moe.trtllm_fp8_block_scale_moe``
invoked exactly as vLLM's monolithic ``TrtLlmFp8Experts._apply_block_scale``
does for GLM-5.3-Flash (``vllm/model_executor/layers/fused_moe/experts/
trtllm_fp8_moe.py``): FP8 E4M3 activations with per-token-group-128 FP32 scales
(``per_token_group_quant_fp8``, transposed to ``[H/128, T]``), W31 BlockMajorK
shuffled FP8 weights with 128x128 FP32 block scales (``prepare_fp8_moe_layer_for_fi``),
FP32 router logits and correction bias, DeepSeekV3 routing with the routed
scale applied in-kernel, the SwiGLU clamp passed as ``gemm1_clamp_limit``, and
FlashInfer's defaults for PDL and activation type.

One call launches routing (``routingCustom`` scores + cluster/coop), FC1
(``bmm_E4m3_E4m3E4m3``), ``activationDeepSeekKernel``, FC2
(``bmm_Bfloat16_E4m3E4m3``) and ``finalizeKernel``. They are timed together
with CUPTI over every launch, with PDL overlap counted once, as the NVFP4
backend of this kind is.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.autotune_cache import autotune_cached
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.moe.exact_topk import exact_topk_ids

WEIGHT_FORMAT = "fp8_e4m3_block"
GROUP_SIZE = 128
# flashinfer.fused_moe.RoutingMethodType; vLLM selects DeepSeekV3 for
# sigmoid scoring + correction bias + renormalize + num_expert_group > 0
# (fused_moe/config.py:get_routing_method_type).
ROUTING_METHODS = {"deepseek_v3": 2}
# GLM-5.3-Flash `swiglu_limit`. vLLM forwards it as a per-local-expert
# `gemm1_clamp_limit` tensor, and FlashInfer applies it for DeepSeekFp8 SwiGLU.
SWIGLU_LIMIT = 10.0
# vLLM's `fi_moe_largest_bucket`: max(max_num_tokens * dp, 8192). The measured
# GLM-5.3 deployment (chunk 2048, DP1) resolves to the 8192 floor.
TUNE_MAX_NUM_TOKENS = 8192
# Forced router logits: selected experts get distinct high scores, the rest a
# low score, so sigmoid + top-k picks exactly the requested ids.
_SELECTED_LOGIT = 8.0
_SELECTED_STEP = 0.5
_UNSELECTED_LOGIT = -8.0
_W13_SCALE = (0.05, 0.15)
_W2_SCALE = (0.005, 0.015)
_WEIGHT_CHUNK = 8
_PREPARED_WEIGHT_CACHE: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import vllm.model_executor.layers.fused_moe  # noqa: F401  (breaks an import cycle)
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
            _shuffle_deepseek_fp8_moe_weights,
            swap_w13_to_w31,
        )
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented("FlashInfer and the vLLM fork are required") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("FP8 block-scale MoE profiling requires CUDA")
    if tuple(torch.cuda.get_device_capability()) != (10, 0):
        raise ProfilerNotImplemented("FP8 block-scale MoE profiling requires SM100")
    return torch, per_token_group_quant_fp8, swap_w13_to_w31, _shuffle_deepseek_fp8_moe_weights


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
        raise ValueError("FP8 block-scale fused MoE requires BF16 activation/output precision")
    if args["weight_format"] != WEIGHT_FORMAT or args["group_size"] != GROUP_SIZE:
        raise ValueError(
            f"FP8 block-scale fused MoE requires {WEIGHT_FORMAT} weights with "
            f"group_size={GROUP_SIZE}"
        )
    if args["routing_method"] not in ROUTING_METHODS:
        raise ValueError(f"unsupported routing method: {args['routing_method']}")
    # Grouped routing only admits experts from the best `topk_group` groups, so
    # forced logits could not realize an arbitrary per-expert histogram.
    if args["n_group"] != 1 or args["topk_group"] != 1:
        raise ValueError("only ungrouped DeepSeekV3 routing (n_group=topk_group=1) is supported")
    if args["hidden_size"] % GROUP_SIZE or args["intermediate_size"] % GROUP_SIZE:
        raise ValueError("hidden_size and intermediate_size must be multiples of 128")
    # vLLM's asserts before this call: the routing kernel holds one expert per
    # thread (<= 512), wants a multiple of 4 experts, and top-k <= 32.
    if args["num_experts"] % 4 or args["num_experts"] > 512:
        raise ValueError("num_experts must be a multiple of 4 and at most 512")
    if args["top_k"] > 32:
        raise ValueError("top_k must be at most 32")
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


def forced_routing_logits(torch: Any, ids: list[list[int]], num_experts: int, device: Any) -> Any:
    """FP32 router logits whose DeepSeekV3 top-k (zero bias) is exactly ``ids``."""

    selected = torch.tensor(ids, dtype=torch.int64, device=device)
    logits = torch.full(
        (selected.shape[0], num_experts), _UNSELECTED_LOGIT, dtype=torch.float32, device=device
    )
    ranks = torch.arange(selected.shape[1], dtype=torch.float32, device=device)
    values = (_SELECTED_LOGIT - _SELECTED_STEP * ranks).expand_as(selected).contiguous()
    logits.scatter_(1, selected, values)
    return logits


def _random_fp8(torch: Any, shape: tuple[int, ...], generator: Any) -> Any:
    values = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    return values.to(torch.float8_e4m3fn)


def _random_scales(
    torch: Any, shape: tuple[int, ...], bounds: tuple[float, float], generator: Any
) -> Any:
    low, high = bounds
    values = torch.rand(shape, generator=generator, device="cuda", dtype=torch.float32)
    return low + (high - low) * values


def _prepared_weights(
    torch: Any,
    swap_w13_to_w31: Any,
    shuffle: Any,
    *,
    experts: int,
    hidden_size: int,
    intermediate_size: int,
    seed: int,
    keep_checkpoint: bool,
) -> dict[str, Any]:
    """Build checkpoint-order FP8 weights and convert them as vLLM does."""

    key = (torch.cuda.current_device(), experts, hidden_size, intermediate_size, seed)
    cached = _PREPARED_WEIGHT_CACHE.get(key)
    if cached is not None and (not keep_checkpoint or "w13" in cached):
        return cached

    generator = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.empty(
        (experts, 2 * intermediate_size, hidden_size), dtype=torch.float8_e4m3fn, device="cuda"
    )
    w2 = torch.empty(
        (experts, hidden_size, intermediate_size), dtype=torch.float8_e4m3fn, device="cuda"
    )
    for start in range(0, experts, _WEIGHT_CHUNK):
        stop = min(start + _WEIGHT_CHUNK, experts)
        w13[start:stop] = _random_fp8(torch, tuple(w13[start:stop].shape), generator)
        w2[start:stop] = _random_fp8(torch, tuple(w2[start:stop].shape), generator)
    w13_scale = _random_scales(
        torch,
        (experts, 2 * intermediate_size // GROUP_SIZE, hidden_size // GROUP_SIZE),
        _W13_SCALE,
        generator,
    )
    w2_scale = _random_scales(
        torch,
        (experts, hidden_size // GROUP_SIZE, intermediate_size // GROUP_SIZE),
        _W2_SCALE,
        generator,
    )

    # prepare_fp8_moe_layer_for_fi, DeepSeekFp8 + TRT-LLM branch.
    w31, w2_blocked = shuffle(swap_w13_to_w31(w13), w2)
    prepared = {
        "gemm1_weights": w31,
        "gemm1_weights_scale": swap_w13_to_w31(w13_scale).clamp(min=1e-10).contiguous(),
        "gemm2_weights": w2_blocked,
        "gemm2_weights_scale": w2_scale.clamp(min=1e-10),
    }
    if keep_checkpoint:
        prepared.update(w13=w13, w13_scale=w13_scale, w2=w2, w2_scale=w2_scale)
    else:
        del w13, w2
    _PREPARED_WEIGHT_CACHE.clear()
    _PREPARED_WEIGHT_CACHE[key] = prepared
    return prepared


def _build_case(
    args: dict[str, Any], *, seed: int = 0, keep_checkpoint: bool = False
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Return ``(torch, call_kwargs, weights)`` for one validated spec."""

    torch, quantize, swap_w13_to_w31, shuffle = _load_runtime()
    weights = _prepared_weights(
        torch,
        swap_w13_to_w31,
        shuffle,
        experts=args["num_local_experts"],
        hidden_size=args["hidden_size"],
        intermediate_size=args["intermediate_size"],
        seed=seed,
        keep_checkpoint=keep_checkpoint,
    )
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    source = torch.randn(
        (args["num_tokens"], args["hidden_size"]),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    hidden, hidden_scale = quantize(source, GROUP_SIZE)
    ids = exact_topk_ids(
        num_tokens=args["num_tokens"],
        top_k=args["top_k"],
        per_expert_batches=args["per_expert_batches"],
    )
    clamp = torch.full(
        (args["num_local_experts"],), SWIGLU_LIMIT, dtype=torch.float32, device="cuda"
    )
    call_kwargs = {
        "routing_logits": forced_routing_logits(torch, ids, args["num_experts"], "cuda"),
        "routing_bias": torch.zeros(args["num_experts"], dtype=torch.float32, device="cuda"),
        "hidden_states": hidden,
        "hidden_states_scale": hidden_scale.t().contiguous(),
        "gemm1_weights": weights["gemm1_weights"],
        "gemm1_weights_scale": weights["gemm1_weights_scale"],
        "gemm1_alpha": None,
        "gemm1_beta": None,
        "gemm1_clamp_limit": clamp,
        "gemm2_weights": weights["gemm2_weights"],
        "gemm2_weights_scale": weights["gemm2_weights_scale"],
        "num_experts": args["num_experts"],
        "top_k": args["top_k"],
        "n_group": args["n_group"],
        "topk_group": args["topk_group"],
        "intermediate_size": args["intermediate_size"],
        # Rust rotates the rank's experts to the front of per_expert_batches.
        "local_expert_offset": 0,
        "local_num_experts": args["num_local_experts"],
        "routed_scaling_factor": (
            args["routed_scaling_numerator"] / args["routed_scaling_denominator"]
        ),
        "routing_method_type": ROUTING_METHODS[args["routing_method"]],
        "use_shuffled_weight": True,
        "weight_layout": 2,  # WeightLayout.BlockMajorK
        "fp8_quantization_type": 1,  # Fp8QuantizationType.DeepSeekFp8
        "routing_replay_out": None,
        "tune_max_num_tokens": TUNE_MAX_NUM_TOKENS,
    }
    return torch, call_kwargs, weights


def _call(call_kwargs: dict[str, Any]) -> Any:
    from flashinfer.fused_moe import Fp8QuantizationType, trtllm_fp8_block_scale_moe

    kwargs = dict(call_kwargs)
    kwargs["fp8_quantization_type"] = Fp8QuantizationType(kwargs["fp8_quantization_type"])
    return trtllm_fp8_block_scale_moe(**kwargs)


def _logical_bytes(args: dict[str, Any]) -> int:
    """Useful algorithmic traffic of this rank's call (see the NVFP4 runner).

    Routed FP8 activations are counted once per local assignment, weights and
    their FP32 128x128 block scales only for local experts with a row. The
    output is one BF16 row per input token (the call finalizes).
    """

    local_batches = args["per_expert_batches"][: args["num_local_experts"]]
    local_rows = sum(local_batches)
    active_experts = sum(batch > 0 for batch in local_batches)
    tokens = args["num_tokens"]
    hidden = args["hidden_size"]
    intermediate = args["intermediate_size"]
    experts = args["num_experts"]
    blocks = GROUP_SIZE * GROUP_SIZE

    routing = 4 * tokens * experts + 4 * experts
    activations = local_rows * (hidden + 4 * hidden // GROUP_SIZE)
    weights = active_experts * (3 * intermediate * hidden + 4 * 3 * intermediate * hidden // blocks)
    output = 2 * hidden * tokens
    return routing + activations + weights + output


def profile_fp8_block_fused_moe_sm100(
    *,
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
    args = _validate_args(**dict(locals()))
    try:
        from flashinfer.autotuner import autotune

        torch, call_kwargs, _weights = _build_case(args)

        def run_once() -> None:
            _call(call_kwargs)

        with autotune_cached(autotune, "nvfp4_fused_moe.fp8_block_vllm"):
            run_once()
        torch.cuda.synchronize()
        time_ms = Timer.cupti(run_once, warmup=3, interval_union=True)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError("SM100 FP8 block-scale fused MoE ran out of memory") from exc
        raise KernelLaunchFailed(f"SM100 FP8 block-scale fused MoE failed: {exc}") from exc

    local_rows = sum(args["per_expert_batches"][: args["num_local_experts"]])
    h, i = args["hidden_size"], args["intermediate_size"]
    flops = 2 * local_rows * (h * 2 * i + i * h)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=time_ms,
        tflops=flops / elapsed_s / 1e12,
        memory_bandwidth_gbps=_logical_bytes(args) / elapsed_s / 1e9,
        energy_j=energy_j,
    )


def compare_with_reference(
    *, seed: int = 0, random_routing: bool = False, **spec: Any
) -> dict[str, float]:
    """Run the production call once and compare it with the Torch reference.

    With ``random_routing`` the router logits and correction bias are random
    instead of forced, so the kernel's own DeepSeekV3 selection, bias handling,
    renormalization and routed scale are checked too. Diagnostic only: this is
    never on the timed path.
    """

    from profiling.runners.moe.fp8_block_fused_moe_reference import (
        dequantize_blocks,
        dequantize_token_groups,
        fp8_block_fused_moe_reference,
    )

    args = _validate_args(**spec)
    torch, call_kwargs, weights = _build_case(args, seed=seed, keep_checkpoint=True)
    if random_routing:
        generator = torch.Generator(device="cuda").manual_seed(seed + 2)
        call_kwargs["routing_logits"] = torch.randn(
            (args["num_tokens"], args["num_experts"]), generator=generator, device="cuda"
        )
        call_kwargs["routing_bias"] = 0.5 * torch.randn(
            (args["num_experts"],), generator=generator, device="cuda"
        )
    actual = _call(call_kwargs).float()
    torch.cuda.synchronize()

    reference_kwargs = {
        "hidden": call_kwargs["hidden_states"],
        "hidden_scale": call_kwargs["hidden_states_scale"].t(),
        "w13": weights["w13"],
        "w13_scale": weights["w13_scale"],
        "w2": weights["w2"],
        "w2_scale": weights["w2_scale"],
        "routing_logits": call_kwargs["routing_logits"],
        "routing_bias": call_kwargs["routing_bias"],
        "top_k": args["top_k"],
        "routed_scaling_factor": call_kwargs["routed_scaling_factor"],
        "local_offset": 0,
        "num_local": args["num_local_experts"],
    }
    expected = fp8_block_fused_moe_reference(torch, clamp_limit=SWIGLU_LIMIT, **reference_kwargs)
    unclamped = fp8_block_fused_moe_reference(torch, clamp_limit=None, **reference_kwargs)

    def rel(a: Any, b: Any) -> float:
        return float((a - b).norm() / b.norm().clamp(min=1e-30))

    x = dequantize_token_groups(torch, reference_kwargs["hidden"], reference_kwargs["hidden_scale"])
    probe = x @ dequantize_blocks(torch, weights["w13"][0], weights["w13_scale"][0]).t()
    return {
        "rel_l2": rel(actual, expected),
        "max_abs": float((actual - expected).abs().max()),
        "ref_abs_max": float(expected.abs().max()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        ),
        "rel_l2_vs_unclamped_reference": rel(actual, unclamped),
        "clamp_active_fraction_expert0": float((probe.abs() > SWIGLU_LIMIT).float().mean()),
        "finite": bool(torch.isfinite(actual).all()),
    }


__all__ = ["compare_with_reference", "forced_routing_logits", "profile_fp8_block_fused_moe_sm100"]
