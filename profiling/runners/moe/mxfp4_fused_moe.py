"""Whole FlashInfer TRT-LLM MXFP4 x MXFP8 routed MoE on SM100.

The timed callable is ``flashinfer.trtllm_fp4_block_scale_routed_moe`` invoked
exactly as vLLM's modular ``TrtLlmMxfp4ExpertsModular._invoke_kernel`` does for
DeepSeek-V4.1-Flash (``vllm/model_executor/layers/fused_moe/experts/
trtllm_mxfp4_moe.py:361``, reached from ``models/deepseek_v4/nvidia/model.py``
through ``Mxfp4MoEMethod.apply`` with the ``FLASHINFER_TRTLLM_MXFP4_MXFP8``
backend):

- activations: MXFP8 E4M3 with linear (non-swizzled) UE8M0 per-32 scales, from
  vLLM's ``mxfp8_e4m3_quantize(x, False, 256)`` (the ``mx_alignment=256`` quant
  config). That quantization is the modular kernel's *prepare* step, a separate
  launch before this call, so it stays outside the timed slot like every
  backend of this kind (the kind starts from quantized activations);
- weights: MXFP4 E2M1 + UE8M0 group-32 scales, converted by vLLM's own
  ``convert_weight_to_mxfp4_moe_kernel_format`` (W3/W1 row interleave, epilogue
  tile 128 shuffle, scale interleave);
- routing: precomputed ``(topk_ids int32, topk_weights fp32)`` from the DSv4
  top-k router; the call only permutes (``routingIndicesCluster``), with
  ``routing_method_type=Renormalize`` and no routed scale (the router already
  applied it);
- activation: SwiGLU with the ``swiglu_limit`` clamp as ``gemm1_clamp_limit``;
  ``enable_pdl=True``, ``do_finalize=True``, ``tune_max_num_tokens`` from
  ``fi_moe_largest_bucket`` (8192 for the measured DP1 deployment).

One call launches ``routingIndicesCluster`` (or the coop/histogram variants at
large T), FC1 ``bmm_MxE4m3_MxE2m1MxE4m3``, FC2 ``bmm_Bfloat16_MxE2m1MxE4m3``
and ``finalizeKernel``. They are timed together with CUPTI over every launch,
PDL overlap counted once, as the other backends of this kind are.
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

WEIGHT_FORMAT = DType.MXFP4_E2M1
GROUP_SIZE = 32
ROUTING_METHODS = ("precomputed_dsv4",)
# flashinfer RoutingMethodType.Renormalize: what vLLM passes for pre-routed ids.
RENORMALIZE = 1
# flashinfer ActivationType.Swiglu (vLLM MoEActivation.SILU).
SWIGLU = 3
# DeepSeek-V4 `swiglu_limit`, forwarded per local expert as gemm1_clamp_limit.
SWIGLU_LIMIT = 10.0
# vLLM's `fi_moe_largest_bucket`: max(max_num_tokens * dp, 8192); the measured
# V4.1 deployment (2048-token chunks, DP1) resolves to the 8192 floor.
TUNE_MAX_NUM_TOKENS = 8192
# vLLM's mxfp4_mxfp8 quant config for this backend (`mx_alignment=256`).
MXFP8_ALIGNMENT = 256
# TRT-LLM round-up in `mxfp4_round_up_hidden_size_and_intermediate_size`.
HIDDEN_ALIGNMENT = 256
INTERMEDIATE_ALIGNMENT = 128
# TrtLlmMxfp4ExpertsModular._max_supported_tokens: past this vLLM chunks the
# call, so one row would no longer be one launch sequence.
_MAX_GRID_Y = 65535
_MIN_TILE_TOKENS = 8
# UE8M0 exponent bytes (127 = 2^0) for the synthetic weight scales; they keep
# FC1 outputs around the clamp limit so the clamp path is exercised.
_W13_SCALE_BYTES = (121, 122)
_W2_SCALE_BYTES = (117, 118)
_PREPARED_WEIGHT_CACHE: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import vllm.model_executor.layers.fused_moe  # noqa: F401  (breaks an import cycle)
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
            Mxfp4MoeBackend,
            convert_weight_to_mxfp4_moe_kernel_format,
        )
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented("FlashInfer and the vLLM fork are required") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("MXFP4 fused MoE profiling requires CUDA")
    if tuple(torch.cuda.get_device_capability()) != (10, 0):
        raise ProfilerNotImplemented("MXFP4 fused MoE profiling requires SM100")
    return (
        torch,
        mxfp8_e4m3_quantize,
        convert_weight_to_mxfp4_moe_kernel_format,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
    )


def max_tokens_per_call(top_k: int, num_experts: int) -> int:
    return max(1, min(300000, (_MAX_GRID_Y - num_experts) * _MIN_TILE_TOKENS // top_k))


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
        raise ValueError("MXFP4 fused MoE requires BF16 activation/output precision")
    if args["weight_format"] != WEIGHT_FORMAT or args["group_size"] != GROUP_SIZE:
        raise ValueError(
            f"MXFP4 fused MoE requires {WEIGHT_FORMAT} weights with group_size={GROUP_SIZE}"
        )
    if args["routing_method"] not in ROUTING_METHODS:
        raise ValueError(f"unsupported routing method: {args['routing_method']}")
    # The call receives finished top-k ids and weights: no grouping and no
    # routed scale happen inside it, so any other value would key a row by a
    # behavior the measured call does not have.
    if args["n_group"] != 1 or args["topk_group"] != 1:
        raise ValueError("precomputed routing takes n_group=topk_group=1")
    if args["routed_scaling_numerator"] != args["routed_scaling_denominator"]:
        raise ValueError("precomputed routing applies no routed scale (use 1/1)")
    if args["hidden_size"] % HIDDEN_ALIGNMENT:
        raise ValueError(f"hidden_size must be a multiple of {HIDDEN_ALIGNMENT}")
    if args["intermediate_size"] % INTERMEDIATE_ALIGNMENT:
        raise ValueError(f"intermediate_size must be a multiple of {INTERMEDIATE_ALIGNMENT}")
    if len(args["per_expert_batches"]) != args["num_experts"]:
        raise ValueError("per_expert_batches must contain one count per global expert")
    if args["num_experts"] % args["num_local_experts"]:
        raise ValueError("num_local_experts must divide num_experts")
    if args["num_tokens"] > max_tokens_per_call(args["top_k"], args["num_experts"]):
        raise ValueError("num_tokens exceeds one vLLM kernel call (it would be chunked)")
    exact_topk_ids(
        num_tokens=args["num_tokens"],
        top_k=args["top_k"],
        per_expert_batches=args["per_expert_batches"],
    )
    return args


def _random_scale_bytes(torch: Any, shape: tuple[int, ...], bounds: tuple[int, int], g: Any) -> Any:
    low, high = bounds
    return torch.randint(low, high + 1, shape, generator=g, device="cuda", dtype=torch.uint8)


def _prepared_weights(
    torch: Any,
    convert: Any,
    backend: Any,
    *,
    experts: int,
    hidden_size: int,
    intermediate_size: int,
    seed: int,
    keep_checkpoint: bool,
) -> dict[str, Any]:
    """Build checkpoint-order MXFP4 weights and convert them as vLLM does."""

    key = (torch.cuda.current_device(), experts, hidden_size, intermediate_size, seed)
    cached = _PREPARED_WEIGHT_CACHE.get(key)
    if cached is not None and (not keep_checkpoint or "w13" in cached):
        return cached
    _PREPARED_WEIGHT_CACHE.clear()

    g = torch.Generator(device="cuda").manual_seed(seed)
    # Every byte is a valid pair of E2M1 codes (E2M1 has no NaN/Inf).
    w13 = torch.randint(
        0,
        256,
        (experts, 2 * intermediate_size, hidden_size // 2),
        generator=g,
        device="cuda",
        dtype=torch.uint8,
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        generator=g,
        device="cuda",
        dtype=torch.uint8,
    )
    w13_scale = _random_scale_bytes(
        torch, (experts, 2 * intermediate_size, hidden_size // GROUP_SIZE), _W13_SCALE_BYTES, g
    )
    w2_scale = _random_scale_bytes(
        torch, (experts, hidden_size, intermediate_size // GROUP_SIZE), _W2_SCALE_BYTES, g
    )
    gemm1, gemm2, gemm1_scale, gemm2_scale, _b1, _b2 = convert(
        mxfp4_backend=backend,
        layer=None,
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale=w13_scale,
        w2_weight_scale=w2_scale,
        w13_bias=None,
        w2_bias=None,
        _cache_permute_indices={},
    )
    prepared = {
        "gemm1_weights": gemm1,
        "gemm1_weights_scale": gemm1_scale,
        "gemm2_weights": gemm2,
        "gemm2_weights_scale": gemm2_scale,
    }
    if keep_checkpoint:
        prepared.update(w13=w13, w13_scale=w13_scale, w2=w2, w2_scale=w2_scale)
    else:
        del w13, w2, w13_scale, w2_scale
    _PREPARED_WEIGHT_CACHE[key] = prepared
    return prepared


def routing_tensors(torch: Any, ids: list[list[int]], *, seed: int, device: Any) -> tuple[Any, Any]:
    """DSv4-router-shaped ``(int32 ids, fp32 weights)``; weights sum to 1 per row."""

    topk_ids = torch.tensor(ids, dtype=torch.int32, device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    raw = 0.5 + torch.rand(topk_ids.shape, generator=g, device=device, dtype=torch.float32)
    return topk_ids, raw / raw.sum(dim=-1, keepdim=True)


def _build_case(
    args: dict[str, Any], *, seed: int = 0, keep_checkpoint: bool = False
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Return ``(torch, call_kwargs, weights)`` for one validated spec."""

    torch, quantize, convert, backend = _load_runtime()
    weights = _prepared_weights(
        torch,
        convert,
        backend,
        experts=args["num_local_experts"],
        hidden_size=args["hidden_size"],
        intermediate_size=args["intermediate_size"],
        seed=seed,
        keep_checkpoint=keep_checkpoint,
    )
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    source = torch.randn(
        (args["num_tokens"], args["hidden_size"]),
        generator=g,
        device="cuda",
        dtype=torch.bfloat16,
    )
    hidden, hidden_scale = quantize(source, False, MXFP8_ALIGNMENT)
    ids = exact_topk_ids(
        num_tokens=args["num_tokens"],
        top_k=args["top_k"],
        per_expert_batches=args["per_expert_batches"],
    )
    topk_ids, topk_weights = routing_tensors(torch, ids, seed=seed + 2, device="cuda")
    clamp = torch.full(
        (args["num_local_experts"],), SWIGLU_LIMIT, dtype=torch.float32, device="cuda"
    )
    call_kwargs = {
        "topk_ids": (topk_ids, topk_weights),
        "routing_bias": None,
        "hidden_states": hidden,
        "hidden_states_scale": hidden_scale.view(torch.float8_e4m3fn),
        "gemm1_weights": weights["gemm1_weights"],
        "gemm1_weights_scale": weights["gemm1_weights_scale"],
        "gemm1_bias": None,
        "gemm1_alpha": None,
        "gemm1_beta": None,
        "gemm1_clamp_limit": clamp,
        "gemm2_weights": weights["gemm2_weights"],
        "gemm2_weights_scale": weights["gemm2_weights_scale"],
        "gemm2_bias": None,
        "output1_scale_scalar": None,
        "output1_scale_gate_scalar": None,
        "output2_scale_scalar": None,
        "num_experts": args["num_experts"],
        "top_k": args["top_k"],
        "n_group": None,
        "topk_group": None,
        "intermediate_size": args["intermediate_size"],
        # Rust rotates the rank's experts to the front of per_expert_batches.
        "local_expert_offset": 0,
        "local_num_experts": args["num_local_experts"],
        "routed_scaling_factor": None,
        "routing_method_type": RENORMALIZE,
        "do_finalize": True,
        "enable_pdl": True,
        "activation_type": SWIGLU,
        "output": torch.empty(
            (args["num_tokens"], args["hidden_size"]), dtype=torch.bfloat16, device="cuda"
        ),
        "tune_max_num_tokens": TUNE_MAX_NUM_TOKENS,
    }
    return torch, call_kwargs, weights


def _call(call_kwargs: dict[str, Any]) -> Any:
    from flashinfer import trtllm_fp4_block_scale_routed_moe

    trtllm_fp4_block_scale_routed_moe(**call_kwargs)
    return call_kwargs["output"]


def _logical_bytes(args: dict[str, Any]) -> int:
    """Useful algorithmic traffic of this rank's call (see the NVFP4 runner).

    Routed MXFP8 activations (1 B values + 1 B scale per 32) are counted once
    per local assignment; packed E2M1 weights and their UE8M0 group-32 scales
    only for local experts with a row. Routing input is the int32 ids + fp32
    weights; the output is one BF16 row per token (the call finalizes).
    """

    local_batches = args["per_expert_batches"][: args["num_local_experts"]]
    local_rows = sum(local_batches)
    active_experts = sum(batch > 0 for batch in local_batches)
    tokens = args["num_tokens"]
    hidden = args["hidden_size"]
    intermediate = args["intermediate_size"]

    routing = 8 * tokens * args["top_k"]
    activations = local_rows * (hidden + hidden // GROUP_SIZE)
    weight_elements = 3 * intermediate * hidden
    weights = active_experts * (weight_elements // 2 + weight_elements // GROUP_SIZE)
    clamp = 4 * args["num_local_experts"]
    output = 2 * hidden * tokens
    return routing + activations + weights + clamp + output


def profile_mxfp4_fused_moe_sm100(
    *,
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
    args = _validate_args(**dict(locals()))
    try:
        from flashinfer.autotuner import autotune

        torch, call_kwargs, _weights = _build_case(args)

        def run_once() -> None:
            _call(call_kwargs)

        with autotune_cached(autotune, "nvfp4_fused_moe.mxfp4_vllm"):
            run_once()
        torch.cuda.synchronize()
        time_ms = Timer.cupti(run_once, warmup=3, interval_union=True)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError("SM100 MXFP4 fused MoE ran out of memory") from exc
        raise KernelLaunchFailed(f"SM100 MXFP4 fused MoE failed: {exc}") from exc

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


def compare_with_reference(*, seed: int = 0, **spec: Any) -> dict[str, float]:
    """Run the production call once and compare it with the Torch reference.

    Diagnostic only; never on the timed path. Reports the error against the
    reference with and without an MXFP8 round-trip of the FC1 output (the FC1
    kernel writes MXFP8 for FC2), plus the error against an unclamped reference
    to show the clamp is honored.
    """

    from profiling.runners.moe.mxfp4_fused_moe_reference import (
        dequantize_mxfp4,
        dequantize_mxfp8,
        mxfp4_fused_moe_reference,
    )

    args = _validate_args(**spec)
    torch, call_kwargs, weights = _build_case(args, seed=seed, keep_checkpoint=True)
    actual = _call(call_kwargs).float()
    torch.cuda.synchronize()

    topk_ids, topk_weights = call_kwargs["topk_ids"]
    hidden_scale = call_kwargs["hidden_states_scale"].reshape(args["num_tokens"], -1)
    reference_kwargs = {
        "hidden": call_kwargs["hidden_states"],
        "hidden_scale": hidden_scale,
        "w13": weights["w13"],
        "w13_scale": weights["w13_scale"],
        "w2": weights["w2"],
        "w2_scale": weights["w2_scale"],
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "local_offset": 0,
        "num_local": args["num_local_experts"],
    }
    requant = mxfp4_fused_moe_reference(
        torch, clamp_limit=SWIGLU_LIMIT, requantize_intermediate=True, **reference_kwargs
    )
    plain = mxfp4_fused_moe_reference(
        torch, clamp_limit=SWIGLU_LIMIT, requantize_intermediate=False, **reference_kwargs
    )
    unclamped = mxfp4_fused_moe_reference(
        torch, clamp_limit=None, requantize_intermediate=True, **reference_kwargs
    )

    def rel(a: Any, b: Any) -> float:
        return float((a - b).norm() / b.norm().clamp(min=1e-30))

    x = dequantize_mxfp8(torch, reference_kwargs["hidden"], hidden_scale)
    probe = x @ dequantize_mxfp4(torch, weights["w13"][0], weights["w13_scale"][0]).t()
    ref_max = float(requant.abs().max())
    max_abs = float((actual - requant).abs().max())
    return {
        "rel_l2": rel(actual, requant),
        "max_abs": max_abs,
        "ref_abs_max": ref_max,
        "max_abs_over_ref_max": max_abs / max(ref_max, 1e-30),
        "cosine": float(
            torch.nn.functional.cosine_similarity(actual.flatten(), requant.flatten(), dim=0)
        ),
        "rel_l2_vs_no_requant_reference": rel(actual, plain),
        "rel_l2_vs_unclamped_reference": rel(actual, unclamped),
        "clamp_active_fraction_expert0": float((probe.abs() > SWIGLU_LIMIT).float().mean()),
        "finite": bool(torch.isfinite(actual).all()),
    }


__all__ = ["compare_with_reference", "profile_mxfp4_fused_moe_sm100", "routing_tensors"]
