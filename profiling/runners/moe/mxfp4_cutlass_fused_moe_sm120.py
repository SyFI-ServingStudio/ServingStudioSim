"""Whole FlashInfer CUTLASS MXFP4 x MXFP8 routed MoE on SM12x, as SGLang runs it.

The ``flashinfer_cutlass_sm120_mxfp4`` backend of ``nvfp4_fused_moe``. With
``--moe-runner-backend flashinfer_mxfp4`` on SM12x (RTX PRO 6000), SGLang v0.5.23
serves DeepSeek-V4.x's MXFP4 routed experts through
``Mxfp4FlashinferCutlassMoEMethod``
(``srt/layers/quantization/fp8.py:459-466`` picks it for SM90/SM120;
``srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py``), whose ``apply``
reaches the fused func ``_fused_experts_flashinfer_mxfp4_cutlass``
(``srt/layers/moe/moe_runner/flashinfer_cutlass.py:482-619``). Per forward that
func runs, and this row times, exactly:

1. ``flashinfer.mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)``:
   the BF16 hidden states to MXFP8 E4M3 with swizzled UE8M0 per-32 scales
   (``flashinfer_cutlass.py:546-551``; the trace's ``quantize_with_block_size``);
2. ``flashinfer.fused_moe.cutlass_fused_moe`` (``flashinfer_cutlass.py:585-617``)
   with ``use_mxfp8_act_scaling=True``: expert-map build / prefix sums,
   ``expandInputRows``, the FC1 CUTLASS grouped GEMM (MXFP8 x MXFP4), SwiGLU
   with the ``swiglu_limit`` clamp (``doActivation``), the FC2 grouped GEMM and
   ``finalizeMoeRouting`` (the call combines; ``use_fused_finalize`` follows
   ``SGLANG_FLASHINFER_MOE_FUSED_FINALIZE``, default off).

Unlike the other backends of this kind, the activation quantization is inside
the row: SGLang runs it inside the same fused func, with nothing between it and
the call, so the arch charges no separate quant op for this backend.

Weights are built in the layout SGLang's load path leaves:
``Fp8MoEMethod.create_weights`` int8 ``[E, 2I, H/2]`` / ``[E, H, I/2]`` packed
E2M1 with ``float8_e8m0fnu`` ``[E, 2I, H/32]`` / ``[E, H, I/32]`` scales;
``load_up_proj_weight_first`` puts W13 rows in ``[up; gate]`` order; and
``process_weights_after_loading`` (``mxfp4_flashinfer_cutlass_moe.py:183-193``)
rewrites both scale tensors in place with ``block_scale_interleave``. The call
gets the weights viewed as int64, the scales as int32, a neutral per-expert
global scale of 1.0, and DeepSeek-V4's ``swiglu_limit`` per local expert.

Expert parallelism: ``flashinfer_mxfp4`` keeps global expert ids
(``token_dispatcher/standard.py`` ``_keeps_global_expert_ids``), and the call
takes ``ep_size = num_experts / num_local_experts``. Rust rotates this rank's
experts to the front of ``per_expert_batches``, so the row runs as EP rank 0,
whose experts are ids ``[0, num_local_experts)``; the kernel drops the rest.
``tp_size = 1`` (EP, no expert TP).

Tactics: SGLang autotunes this call at startup (``flashinfer_autotune.py``:
``flashinfer_mxfp4`` is in ``moe_needs_autotune``) with
``tune_max_num_tokens = min(max_prefill_buffer_tokens * attn_dp, 8192)``
(``_cutlass_moe_tune_max_num_tokens``); the measured deployment
(``--chunked-prefill-size 2048``, DP1) gives 2048. The row autotunes the same
way before timing. Runs in ``sglang_env``.
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
# Mxfp4FlashinferCutlassMoEMethod.create_weights requires 128-aligned dims.
ALIGNMENT = 128
# flashinfer.mxfp8_quantize alignment SGLang passes on SM120.
MXFP8_ALIGNMENT = 32
# DeepSeek-V4 `swiglu_limit` (moe_runner_config.swiglu_limit), per local expert.
SWIGLU_LIMIT = 10.0
# _cutlass_moe_tune_max_num_tokens for the measured deployment (2048-token
# chunks, attention DP1); FlashInfer's own default is 8192.
TUNE_MAX_NUM_TOKENS = 2048
# FP8 activations (twice: the input and the FC1 output FC2 reads) against an
# FP32 reference on the dequantized weights. E4M3 keeps 3 mantissa bits, a
# relative rounding error of at most 2^-4 per element and ~1.8% RMS, so two
# quantizations plus the BF16 output stay well under 8% in relative L2.
REL_L2_TOL = 8e-2
_CHECK_TOKENS = 64
# UE8M0 exponent bytes (127 = 2^0), as the SM100 MXFP4 runner uses: they keep
# FC1 outputs around the clamp limit so the clamp path is exercised.
_W13_SCALE_BYTES = (121, 122)
_W2_SCALE_BYTES = (117, 118)
_WEIGHT_CACHE: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}


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
        raise ValueError("SM120 MXFP4 fused MoE requires BF16 activation/output precision")
    if (
        DType.from_value(args["weight_format"]) is not WEIGHT_FORMAT
        or args["group_size"] != GROUP_SIZE
    ):
        raise ValueError(
            f"SM120 MXFP4 fused MoE requires {WEIGHT_FORMAT} weights with group_size={GROUP_SIZE}"
        )
    if args["routing_method"] not in ROUTING_METHODS:
        raise ValueError(f"unsupported routing method: {args['routing_method']}")
    # The call takes finished top-k ids and weights: no grouping and no routed
    # scale inside it (Mxfp4FlashinferCutlassMoEMethod fuses the routed scale
    # into the top-k weights upstream).
    if args["n_group"] != 1 or args["topk_group"] != 1:
        raise ValueError("precomputed routing takes n_group=topk_group=1")
    if args["routed_scaling_numerator"] != args["routed_scaling_denominator"]:
        raise ValueError("precomputed routing applies no routed scale (use 1/1)")
    if args["hidden_size"] % ALIGNMENT or args["intermediate_size"] % ALIGNMENT:
        raise ValueError(f"hidden_size and intermediate_size must be multiples of {ALIGNMENT}")
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


def _load_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from flashinfer import block_scale_interleave, mxfp8_quantize
        from flashinfer.autotuner import autotune
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "flashinfer_cutlass_sm120_mxfp4 requires FlashInfer in the SGLang environment "
            "(sglang_env)"
        ) from exc
    return (
        torch,
        block_scale_interleave,
        mxfp8_quantize,
        autotune,
        cutlass_fused_moe,
        ActivationType,
    )


def _scale_bytes(torch: Any, shape: tuple[int, ...], bounds: tuple[int, int], g: Any) -> Any:
    low, high = bounds
    return torch.randint(low, high + 1, shape, generator=g, device="cuda", dtype=torch.uint8)


def _weights(
    torch: Any,
    block_scale_interleave: Any,
    *,
    experts: int,
    hidden: int,
    intermediate: int,
    seed: int,
) -> dict[str, Any]:
    """Checkpoint-order (``[gate; up]``) tensors for the reference, and the
    SGLang-loaded SM120 layout (``[up; gate]``, interleaved E8M0 scales)."""

    key = (torch.cuda.current_device(), experts, hidden, intermediate, seed)
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    _WEIGHT_CACHE.clear()
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Every byte is a valid pair of E2M1 codes (E2M1 has no NaN/Inf).
    w13 = torch.randint(
        0,
        256,
        (experts, 2 * intermediate, hidden // 2),
        generator=g,
        device="cuda",
        dtype=torch.uint8,
    )
    w2 = torch.randint(
        0, 256, (experts, hidden, intermediate // 2), generator=g, device="cuda", dtype=torch.uint8
    )
    w13_scale = _scale_bytes(
        torch, (experts, 2 * intermediate, hidden // GROUP_SIZE), _W13_SCALE_BYTES, g
    )
    w2_scale = _scale_bytes(
        torch, (experts, hidden, intermediate // GROUP_SIZE), _W2_SCALE_BYTES, g
    )

    def up_first(t: Any) -> Any:
        return torch.cat((t[:, intermediate:], t[:, :intermediate]), dim=1).contiguous()

    # The loaded layer: int8 packed weights, [up; gate] rows, E8M0 scales
    # interleaved in place exactly as process_weights_after_loading does.
    layer_w13 = up_first(w13).view(torch.int8)
    layer_w2 = w2.clone().view(torch.int8)
    layer_s13 = up_first(w13_scale)
    layer_s2 = w2_scale.clone()
    for scale_u8 in (layer_s13, layer_s2):
        scale_u8.copy_(block_scale_interleave(scale_u8).reshape_as(scale_u8))
    prepared = {
        "w13": w13,
        "w2": w2,
        "w13_scale": w13_scale,
        "w2_scale": w2_scale,
        "layer_w13": layer_w13,
        "layer_w2": layer_w2,
        "layer_s13": layer_s13,
        "layer_s2": layer_s2,
    }
    _WEIGHT_CACHE[key] = prepared
    return prepared


def _routing(torch: Any, ids: list[list[int]], *, seed: int) -> tuple[Any, Any]:
    """Top-k-router-shaped ``(int32 ids, fp32 weights)``; weights sum to 1 per row."""

    topk_ids = torch.tensor(ids, dtype=torch.int32, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(seed)
    raw = 0.5 + torch.rand(topk_ids.shape, generator=g, device="cuda", dtype=torch.float32)
    return topk_ids, raw / raw.sum(dim=-1, keepdim=True)


def _check(
    torch: Any,
    args: dict[str, Any],
    x: Any,
    out: Any,
    w: dict[str, Any],
    topk_ids: Any,
    topk_weights: Any,
) -> None:
    """Relative L2 of sampled output rows against the FP32 reference."""

    from profiling.runners.moe.mxfp4_fused_moe_reference import (
        mxfp4_routed_experts,
        quantize_dequantize_mxfp8,
    )

    t = args["num_tokens"]
    rows = torch.linspace(0, t - 1, min(t, _CHECK_TOKENS), device="cuda").round().long().unique()
    x_q = quantize_dequantize_mxfp8(torch, x[rows].float())

    def reference(w13: Any, w13_scale: Any) -> Any:
        return mxfp4_routed_experts(
            torch,
            x=x_q,
            w13=w13,
            w13_scale=w13_scale,
            w2=w["w2"],
            w2_scale=w["w2_scale"],
            topk_ids=topk_ids[rows],
            topk_weights=topk_weights[rows],
            local_offset=0,
            num_local=args["num_local_experts"],
            clamp_limit=SWIGLU_LIMIT,
            round_intermediate=lambda act: quantize_dequantize_mxfp8(torch, act),
        )

    def rel(a: Any, b: Any) -> float:
        return float((a - b).norm() / b.norm().clamp(min=1e-30))

    actual = out[rows].float()
    ref = reference(w["w13"], w["w13_scale"])
    err = rel(actual, ref)
    if not (torch.isfinite(actual).all() and err <= REL_L2_TOL):
        # Diagnose a W13 half-order mismatch: the reference with [up; gate] read as [gate; up].
        i = args["intermediate_size"]
        swap = lambda t_: torch.cat((t_[:, i:], t_[:, :i]), dim=1)  # noqa: E731
        swapped = rel(actual, reference(swap(w["w13"]), swap(w["w13_scale"])))
        raise KernelLaunchFailed(
            f"flashinfer_cutlass_sm120_mxfp4 output mismatch: relative L2 {err:.3e} > "
            f"{REL_L2_TOL:.0e} "
            f"(with W13 halves swapped: {swapped:.3e})"
        )


def _logical_bytes(args: dict[str, Any]) -> int:
    """Useful traffic of this rank's call, as the SM100 MXFP4 runner counts it,
    plus the BF16 input the in-row MXFP8 quantization reads and writes."""

    local = args["per_expert_batches"][: args["num_local_experts"]]
    local_rows = sum(local)
    active = sum(batch > 0 for batch in local)
    tokens, hidden, inter = args["num_tokens"], args["hidden_size"], args["intermediate_size"]
    quant = tokens * (2 * hidden + hidden + hidden // GROUP_SIZE)
    routing = 8 * tokens * args["top_k"]
    activations = local_rows * (hidden + hidden // GROUP_SIZE)
    elements = 3 * inter * hidden
    weights = active * (elements // 2 + elements // GROUP_SIZE)
    side = 8 * args["num_local_experts"]  # global scales + clamp limits
    output = 2 * hidden * tokens
    return quant + routing + activations + weights + side + output


def profile_mxfp4_cutlass_fused_moe_sm120(
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
    torch, interleave, mxfp8_quantize, autotune, cutlass_fused_moe, activation_type = (
        _load_runtime()
    )
    try:
        e_local = args["num_local_experts"]
        w = _weights(
            torch,
            interleave,
            experts=e_local,
            hidden=args["hidden_size"],
            intermediate=args["intermediate_size"],
            seed=0,
        )
        g = torch.Generator(device="cuda").manual_seed(1)
        x = torch.randn(
            (args["num_tokens"], args["hidden_size"]),
            generator=g,
            device="cuda",
            dtype=torch.bfloat16,
        )
        ids = exact_topk_ids(
            num_tokens=args["num_tokens"],
            top_k=args["top_k"],
            per_expert_batches=args["per_expert_batches"],
        )
        topk_ids, topk_weights = _routing(torch, ids, seed=2)
        global_scale = torch.ones(e_local, dtype=torch.float32, device="cuda")
        clamp = torch.full((e_local,), SWIGLU_LIMIT, dtype=torch.float32, device="cuda")
        quant_scales = [
            w["layer_s13"].view(torch.int32),
            global_scale,
            w["layer_s2"].view(torch.int32),
            global_scale,
        ]
        fc1 = w["layer_w13"].view(torch.int64)
        fc2 = w["layer_w2"].view(torch.int64)
        out = torch.empty(
            (args["num_tokens"], args["hidden_size"]), dtype=torch.bfloat16, device="cuda"
        )
        ep_size = args["num_experts"] // e_local

        def run_once() -> Any:
            xq, x_sf = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=MXFP8_ALIGNMENT)
            cutlass_fused_moe(
                input=xq,
                token_selected_experts=topk_ids,
                token_final_scales=topk_weights,
                fc1_expert_weights=fc1,
                fc2_expert_weights=fc2,
                output_dtype=torch.bfloat16,
                quant_scales=quant_scales,
                input_sf=x_sf,
                fc1_expert_biases=None,
                fc2_expert_biases=None,
                swiglu_alpha=None,
                swiglu_beta=None,
                swiglu_limit=clamp,
                tp_size=1,
                tp_rank=0,
                ep_size=ep_size,
                ep_rank=0,
                use_w4_group_scaling=False,
                use_mxfp8_act_scaling=True,
                activation_type=activation_type.Swiglu,
                tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
                output=out,
                use_fused_finalize=False,
            )
            return out

        with autotune_cached(autotune, "nvfp4_fused_moe.mxfp4_cutlass_sm120_sglang"):
            run_once()
        torch.cuda.synchronize()
        _check(torch, args, x, run_once(), w, topk_ids, topk_weights)
        time_ms = Timer.cupti(run_once, warmup=3, interval_union=True)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError("SM120 MXFP4 fused MoE ran out of memory") from exc
        raise KernelLaunchFailed(f"SM120 MXFP4 fused MoE failed: {exc}") from exc

    local_rows = sum(args["per_expert_batches"][: args["num_local_experts"]])
    h, i = args["hidden_size"], args["intermediate_size"]
    flops = 2 * local_rows * (h * 2 * i + i * h)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / elapsed_s / 1e12,
        memory_bandwidth_gbps=_logical_bytes(args) / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["profile_mxfp4_cutlass_fused_moe_sm120"]
