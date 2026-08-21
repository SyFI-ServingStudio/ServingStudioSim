"""Runner for vLLM's monolithic FlashInfer TRT-LLM NVFP4 MoE call."""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

WEIGHT_FORMAT = "nvfp4_e2m1"
GROUP_SIZE = 16
ROUTING_METHODS = {"minimax2": 7}
_PREPARED_WEIGHT_CACHE: dict[tuple[int, int, int, int], tuple[Any, Any, Any, Any]] = {}


def _require_b200(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("NVFP4 MoE profiling requires CUDA")
    if tuple(torch.cuda.get_device_capability()) != (10, 0):
        raise ProfilerNotImplemented("NVFP4 MoE profiling requires SM100")


def _validate_args(**kwargs: Any) -> dict[str, Any]:
    args = dict(kwargs)
    positive = (
        "num_tokens", "hidden_size", "intermediate_size", "num_experts",
        "num_local_experts", "top_k", "group_size",
        "routed_scaling_numerator", "routed_scaling_denominator",
    )
    for name in positive:
        args[name] = int(args[name])
        if args[name] <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("local_expert_offset", "n_group", "topk_group"):
        args[name] = int(args[name])
    if DType.from_value(args["input_dtype"]) is not DType.BF16:
        raise ValueError("NVFP4 MoE requires BF16 router/output precision")
    if args["weight_format"] != WEIGHT_FORMAT or args["group_size"] != GROUP_SIZE:
        raise ValueError("NVFP4 MoE requires nvfp4_e2m1 weights with group_size=16")
    if args["routing_method"] not in ROUTING_METHODS:
        raise ValueError(f"unsupported routing method: {args['routing_method']}")
    if args["hidden_size"] % 256 or args["intermediate_size"] % 256:
        raise ValueError("TRT-LLM NVFP4 dimensions must be divisible by 256")
    if args["num_experts"] % args["num_local_experts"]:
        raise ValueError("num_local_experts must divide num_experts")
    end = args["local_expert_offset"] + args["num_local_experts"]
    if args["local_expert_offset"] < 0 or end > args["num_experts"]:
        raise ValueError("local expert shard is outside the global expert range")
    return args


def _packed_weight(torch: Any, experts: int, n: int, k: int) -> Any:
    # Weight values do not affect the selected GEMM recipe. Avoid spending
    # seconds generating random bytes for the multi-gigabyte EP shard.
    return torch.empty((experts, n, k // 2), dtype=torch.uint8, device="cuda")


def _prepared_weights(
    torch: Any,
    prepare: Any,
    *,
    experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[Any, Any, Any, Any]:
    device_index = torch.cuda.current_device()
    key = (device_index, experts, hidden_size, intermediate_size)
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
        is_gated_activation=True,
    )
    _PREPARED_WEIGHT_CACHE[key] = cached
    return cached


def profile_nvfp4_moe_flashinfer_trtllm(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_local_experts: int,
    local_expert_offset: int,
    top_k: int,
    input_dtype: DType | str,
    weight_format: str,
    group_size: int,
    routing_method: str,
    n_group: int,
    topk_group: int,
    routed_scaling_numerator: int,
    routed_scaling_denominator: int,
) -> ComputeMetrics:
    args = _validate_args(**locals())
    try:
        import flashinfer
        import torch

        # Initialize fused_moe first; importing the quantization helper as the
        # package entry point otherwise enters vLLM's flashinfer_utils cycle.
        import vllm.model_executor.layers.fused_moe  # noqa: F401
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
            prepare_static_weights_for_trtllm_fp4_moe,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented("FlashInfer and instrumented vLLM are required") from exc
    _require_b200(torch)
    try:
        source = torch.randn(
            (args["num_tokens"], args["hidden_size"]),
            dtype=torch.bfloat16,
            device="cuda",
        )
        input_scale = torch.ones((), dtype=torch.float32, device="cuda")
        hidden, hidden_scale = ops.scaled_fp4_quant(
            source, input_scale, is_sf_swizzled_layout=False
        )
        router_logits = torch.randn(
            (args["num_tokens"], args["num_experts"]),
            dtype=torch.bfloat16,
            device="cuda",
        )
        router_bias = torch.zeros(args["num_experts"], dtype=torch.float32, device="cuda")
        e, h, i = args["num_local_experts"], args["hidden_size"], args["intermediate_size"]
        w1, s1, w2, s2 = _prepared_weights(
            torch,
            prepare_static_weights_for_trtllm_fp4_moe,
            experts=e,
            hidden_size=h,
            intermediate_size=i,
        )
        expert_scale = torch.ones(e, dtype=torch.float32, device="cuda")
        routed_scale = args["routed_scaling_numerator"] / args["routed_scaling_denominator"]

        def kernel() -> None:
            flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
                routing_logits=router_logits, routing_bias=router_bias,
                hidden_states=hidden,
                hidden_states_scale=hidden_scale.view(torch.float8_e4m3fn).reshape(
                    *hidden.shape[:-1], -1
                ),
                gemm1_weights=w1, gemm1_weights_scale=s1, gemm1_bias=None,
                gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
                gemm2_weights=w2, gemm2_weights_scale=s2, gemm2_bias=None,
                output1_scale_scalar=expert_scale,
                output1_scale_gate_scalar=expert_scale,
                output2_scale_scalar=expert_scale,
                num_experts=args["num_experts"], top_k=args["top_k"],
                n_group=args["n_group"], topk_group=args["topk_group"],
                intermediate_size=i, local_expert_offset=args["local_expert_offset"],
                local_num_experts=e, routed_scaling_factor=routed_scale,
                routing_method_type=ROUTING_METHODS[args["routing_method"]],
                do_finalize=True, activation_type=3,
            )

        kernel()
        torch.cuda.synchronize()
        time_ms = Timer.cuda_event(kernel, warmup=3)
        energy_j = Energy.perf(kernel, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError("flashinfer_trtllm NVFP4 MoE ran out of memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed("flashinfer_trtllm NVFP4 MoE failed") from exc

    rows = args["num_tokens"] * args["top_k"]
    flops = 2 * rows * (h * 2 * i + i * h)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=time_ms,
        tflops=flops / elapsed_s / 1e12,
        memory_bandwidth_gbps=0.0,
        energy_j=energy_j,
    )
