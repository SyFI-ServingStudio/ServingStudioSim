"""Whole FlashInfer TRT-LLM MXFP4-weight / BF16-activation MoE on SM100.

The timed callable is ``flashinfer.trtllm_fp4_block_scale_moe`` invoked as
vLLM's ``TrtLlmMxfp4ExpertsMonolithic.apply`` does
(``vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py``) under
the ``FLASHINFER_TRTLLM_MXFP4_BF16`` backend, which vLLM ranks first for MXFP4
MoE on SM100 (``oracle/mxfp4.py``, ``_get_priority_backends_for_gpt_oss``):

- activations: the BF16 hidden states go in unquantized
  (``hidden_states_scale=None``); the GEMMs cast the MXFP4 weights and run on
  BF16 tensor cores (``bmm_Bfloat16_MxE2m1Bfloat16`` cubins), so no
  quantization launch precedes the call;
- weights: MXFP4 E2M1 + UE8M0 group-32 scales, converted by vLLM's
  ``convert_weight_to_mxfp4_moe_kernel_format`` for this backend;
- routing: inside the call, from the router logits and the correction bias
  (``routing_bias=e_score_correction_bias``). ``deepseek_v3`` with
  ``n_group = topk_group = 1`` runs FlashInfer's ``routingCustom`` kernel with
  the SigmoidBias preprocess; its logits and bias are FP32, as the
  ``nvfp4_fused_moe`` NVFP4 runner documents for that routing method. Forced
  logits realize the requested per-expert histogram exactly;
- activation: SwiGLU with no bias, alpha, beta or clamp (vLLM forwards the
  quant config's, which are ``None`` for a plain-SiLU model);
- ``do_finalize=True`` with a BF16 output, ``enable_pdl`` left at FlashInfer's
  default (vLLM does not pass it), ``tune_max_num_tokens`` from
  ``fi_moe_largest_bucket``: the 8192 floor, so larger shapes reuse the
  largest tuned bucket, as the FP8 block-scale and MXFP4 x MXFP8 backends.

One call launches routing, FC1, FC2 and finalize; CUPTI times every launch,
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
from profiling.runners.moe.mxfp4_fused_moe import (
    GROUP_SIZE,
    HIDDEN_ALIGNMENT,
    INTERMEDIATE_ALIGNMENT,
    SWIGLU,
    TUNE_MAX_NUM_TOKENS,
    WEIGHT_FORMAT,
    load_weight_converter,
    prepared_weights,
)
from profiling.runners.moe.nvfp4_fused_moe import (
    ROUTING_ELEMENT_BYTES,
    ROUTING_METHODS,
    forced_routing,
)

# vLLM's Mxfp4MoeBackend member whose weight layout this call reads.
MXFP4_BACKEND = "FLASHINFER_TRTLLM_MXFP4_BF16"
# In-kernel routing methods this call is measured with. Ungrouped DeepSeekV3
# is the only one whose forced logits are proven to realize any histogram
# here (see nvfp4_fused_moe's validation for the grouped case).
BF16_ROUTING_METHODS = ("deepseek_v3",)


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
        raise ValueError("MXFP4/BF16 fused MoE requires BF16 activation/output precision")
    if args["weight_format"] != WEIGHT_FORMAT or args["group_size"] != GROUP_SIZE:
        raise ValueError(
            f"MXFP4/BF16 fused MoE requires {WEIGHT_FORMAT} weights with group_size={GROUP_SIZE}"
        )
    if args["routing_method"] not in BF16_ROUTING_METHODS:
        raise ValueError(f"unsupported routing method: {args['routing_method']}")
    if args["n_group"] != 1 or args["topk_group"] != 1:
        raise ValueError("only ungrouped deepseek_v3 routing (n_group=topk_group=1) is supported")
    if args["hidden_size"] % HIDDEN_ALIGNMENT:
        raise ValueError(f"hidden_size must be a multiple of {HIDDEN_ALIGNMENT}")
    if args["intermediate_size"] % INTERMEDIATE_ALIGNMENT:
        raise ValueError(f"intermediate_size must be a multiple of {INTERMEDIATE_ALIGNMENT}")
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


def tuning_label(args: dict[str, Any]) -> str:
    """Persisted tactic file for this configuration (see nvfp4_fused_moe's
    ``tuning_label``: FlashInfer's key omits the expert counts and sizes)."""

    return (
        f"nvfp4_fused_moe.mxfp4_bf16_vllm.{args['routing_method']}"
        f".e{args['num_experts']}.l{args['num_local_experts']}"
        f".h{args['hidden_size']}.i{args['intermediate_size']}"
    )


def _build_case(
    args: dict[str, Any], *, seed: int = 0, keep_checkpoint: bool = False
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Return ``(torch, call_kwargs, weights)`` for one validated spec."""

    torch, convert, backend = load_weight_converter(MXFP4_BACKEND)
    weights = prepared_weights(
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
    hidden = torch.randn(
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
    logits, bias = forced_routing(torch, ids, args)
    call_kwargs = {
        "routing_logits": logits,
        "routing_bias": bias,
        "hidden_states": hidden,
        "hidden_states_scale": None,
        "gemm1_weights": weights["gemm1_weights"],
        "gemm1_weights_scale": weights["gemm1_weights_scale"],
        "gemm1_bias": None,
        "gemm1_alpha": None,
        "gemm1_beta": None,
        "gemm1_clamp_limit": None,
        "gemm2_weights": weights["gemm2_weights"],
        "gemm2_weights_scale": weights["gemm2_weights_scale"],
        "gemm2_bias": None,
        "output1_scale_scalar": None,
        "output1_scale_gate_scalar": None,
        "output2_scale_scalar": None,
        "num_experts": args["num_experts"],
        "top_k": args["top_k"],
        "n_group": args["n_group"],
        "topk_group": args["topk_group"],
        "intermediate_size": args["intermediate_size"],
        # Rust rotates the timed rank's experts to the front of per_expert_batches.
        "local_expert_offset": 0,
        "local_num_experts": args["num_local_experts"],
        "routed_scaling_factor": args["routed_scaling_numerator"]
        / args["routed_scaling_denominator"],
        "routing_method_type": ROUTING_METHODS[args["routing_method"]],
        "do_finalize": True,
        "activation_type": SWIGLU,
        "output": torch.empty(
            (args["num_tokens"], args["hidden_size"]), dtype=torch.bfloat16, device="cuda"
        ),
        "tune_max_num_tokens": TUNE_MAX_NUM_TOKENS,
    }
    return torch, call_kwargs, weights


def _call(call_kwargs: dict[str, Any]) -> Any:
    from flashinfer import trtllm_fp4_block_scale_moe

    trtllm_fp4_block_scale_moe(**call_kwargs)
    return call_kwargs["output"]


def _logical_bytes(args: dict[str, Any]) -> int:
    """Useful algorithmic traffic of this rank's call (see the NVFP4 runner).

    Routed BF16 activations (2 B per element, no scale) are counted once per
    local assignment; packed E2M1 weights and their UE8M0 group-32 scales only
    for local experts with a row. Routing input is the router logits plus the
    correction bias at the routing method's element size; the output is one
    BF16 row per token (the call finalizes).
    """

    local_batches = args["per_expert_batches"][: args["num_local_experts"]]
    local_rows = sum(local_batches)
    active_experts = sum(batch > 0 for batch in local_batches)
    tokens = args["num_tokens"]
    hidden = args["hidden_size"]
    intermediate = args["intermediate_size"]
    experts = args["num_experts"]

    routing = ROUTING_ELEMENT_BYTES[args["routing_method"]] * (tokens * experts + experts)
    activations = 2 * local_rows * hidden
    weight_elements = 3 * intermediate * hidden
    weights = active_experts * (weight_elements // 2 + weight_elements // GROUP_SIZE)
    output = 2 * hidden * tokens
    return routing + activations + weights + output


def profile_mxfp4_bf16_fused_moe_sm100(
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

        with autotune_cached(autotune, tuning_label(args)):
            run_once()
        torch.cuda.synchronize()
        time_ms = Timer.cupti(run_once, warmup=3, interval_union=True)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError("SM100 MXFP4/BF16 fused MoE ran out of memory") from exc
        raise KernelLaunchFailed(f"SM100 MXFP4/BF16 fused MoE failed: {exc}") from exc

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
    reference with BF16 storage boundaries (route weights and the SwiGLU
    output, as the kernel stores them) and against the all-FP32 reference.
    """

    from profiling.runners.moe.mxfp4_bf16_fused_moe_reference import (
        mxfp4_bf16_fused_moe_reference,
    )

    args = _validate_args(**spec)
    torch, call_kwargs, weights = _build_case(args, seed=seed, keep_checkpoint=True)
    actual = _call(call_kwargs).float()
    torch.cuda.synchronize()

    reference_kwargs = {
        "hidden": call_kwargs["hidden_states"],
        "logits": call_kwargs["routing_logits"],
        "bias": call_kwargs["routing_bias"],
        "w13": weights["w13"],
        "w13_scale": weights["w13_scale"],
        "w2": weights["w2"],
        "w2_scale": weights["w2_scale"],
        "top_k": args["top_k"],
        "routed_scale": call_kwargs["routed_scaling_factor"],
        "local_offset": 0,
        "num_local": args["num_local_experts"],
    }
    bf16 = mxfp4_bf16_fused_moe_reference(torch, bf16_boundaries=True, **reference_kwargs)
    fp32 = mxfp4_bf16_fused_moe_reference(torch, bf16_boundaries=False, **reference_kwargs)

    def rel(a: Any, b: Any) -> float:
        return float((a - b).norm() / b.norm().clamp(min=1e-30))

    ref_max = float(bf16.abs().max())
    max_abs = float((actual - bf16).abs().max())
    return {
        "rel_l2": rel(actual, bf16),
        "max_abs": max_abs,
        "ref_abs_max": ref_max,
        "max_abs_over_ref_max": max_abs / max(ref_max, 1e-30),
        "cosine": float(
            torch.nn.functional.cosine_similarity(actual.flatten(), bf16.flatten(), dim=0)
        ),
        "rel_l2_vs_fp32_reference": rel(actual, fp32),
        "finite": bool(torch.isfinite(actual).all()),
    }


__all__ = ["compare_with_reference", "profile_mxfp4_bf16_fused_moe_sm100"]
