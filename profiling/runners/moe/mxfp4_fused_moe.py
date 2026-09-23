"""Kimi-K3 FlashInfer TRT-LLM MXFP4 fused-MoE runner."""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.moe.exact_topk import exact_topk_ids

_WEIGHT_FORMAT = "mxfp4_e2m1_ue8m0"
_GROUP_SIZE = 32
_ROUTING_METHOD = "deepseek_v3_sigmoid"
_ACTIVATION = "situ"
_HIDDEN_SIZE = 3584
_INTERMEDIATE_SIZE = 3072
_GLOBAL_NUM_EXPERTS = 896
_NUM_LOCAL_EXPERTS = 112
_SUPPORTED_NUM_EXPERTS = frozenset({_NUM_LOCAL_EXPERTS, _GLOBAL_NUM_EXPERTS})
_TOP_K = 16
_EPILOGUE_TILE_M = 128


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
    ):
        args[name] = int(args[name])
        if args[name] <= 0:
            raise ValueError(f"{name} must be positive")
    args["routed_scaling_factor"] = float(args["routed_scaling_factor"])
    args["gemm1_alpha"] = float(args["gemm1_alpha"])
    args["gemm1_clamp_limit"] = float(args["gemm1_clamp_limit"])
    args["per_expert_batches"] = tuple(int(x) for x in args["per_expert_batches"])
    if DType.from_value(args["input_dtype"]) is not DType.BF16:
        raise ValueError("K3 MXFP4 fused MoE requires input_dtype=bf16")
    expected = {
        "hidden_size": _HIDDEN_SIZE,
        "intermediate_size": _INTERMEDIATE_SIZE,
        "num_local_experts": _NUM_LOCAL_EXPERTS,
        "top_k": _TOP_K,
        "group_size": _GROUP_SIZE,
        "n_group": 1,
        "topk_group": 1,
    }
    for name, value in expected.items():
        if args[name] != value:
            raise ValueError(f"K3 MXFP4 requires {name}={value}, got {args[name]}")
    if args["num_experts"] not in _SUPPORTED_NUM_EXPERTS:
        supported = ", ".join(str(value) for value in sorted(_SUPPORTED_NUM_EXPERTS))
        raise ValueError(
            f"K3 MXFP4 requires num_experts in {{{supported}}}, "
            f"got {args['num_experts']}"
        )
    if args["weight_format"] != _WEIGHT_FORMAT:
        raise ValueError(f"unsupported MXFP4 weight format: {args['weight_format']}")
    if args["routing_method"] != _ROUTING_METHOD or args["activation"] != _ACTIVATION:
        raise ValueError("K3 MXFP4 requires DeepSeek-V3 sigmoid routing and SiTU")
    if args["routed_scaling_factor"] != 1.0:
        raise ValueError("K3 MXFP4 requires routed_scaling_factor=1.0")
    if args["gemm1_alpha"] != 4.0 or args["gemm1_clamp_limit"] != 25.0:
        raise ValueError("K3 MXFP4 requires gemm1_alpha=4.0 and gemm1_clamp_limit=25.0")
    if len(args["per_expert_batches"]) != args["num_experts"]:
        raise ValueError(
            "per_expert_batches must contain one count per routed expert "
            f"({args['num_experts']})"
        )
    exact_topk_ids(
        num_tokens=args["num_tokens"],
        top_k=args["top_k"],
        per_expert_batches=args["per_expert_batches"],
    )
    return args


def _require_b200(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("K3 MXFP4 MoE profiling requires CUDA")
    if tuple(torch.cuda.get_device_capability()) != (10, 0):
        raise ProfilerNotImplemented("K3 MXFP4 MoE profiling requires NVIDIA B200/SM100")


def _prepare_weights(torch: Any, args: dict[str, Any]) -> tuple[Any, ...]:
    """Build the exact post-load layout used by Mxfp4MoEMethod.

    The raw tensors use the checkpoint's uint8 E2M1/UE8M0 layout. The pair
    reorder, device permutations, and FlashInfer scale interleave mirror
    ``sglang.srt.layers.quantization.mxfp4.process_weights_after_loading``.
    """
    from flashinfer import nvfp4_block_scale_interleave
    from sglang.srt.layers.quantization.mxfp4 import (
        _get_flashinfer_mxfp4_device_permute_indices,
    )

    device = torch.device("cuda")
    experts = args["num_local_experts"]
    hidden = args["hidden_size"]
    intermediate = args["intermediate_size"]
    generator = torch.Generator(device=device)
    generator.manual_seed(17)
    w13 = torch.randint(
        0,
        256,
        (experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden, intermediate // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    s13 = torch.full(
        (experts, 2 * intermediate, hidden // _GROUP_SIZE),
        127,
        dtype=torch.uint8,
        device=device,
    )
    s2 = torch.full(
        (experts, hidden, intermediate // _GROUP_SIZE),
        127,
        dtype=torch.uint8,
        device=device,
    )
    b13 = torch.zeros((experts, 2 * intermediate), dtype=torch.float32, device=device)
    b2 = torch.zeros((experts, hidden), dtype=torch.float32, device=device)

    half = intermediate
    pair_idx = torch.empty(2 * half, dtype=torch.long, device=device)
    pair_idx[0::2] = torch.arange(half, device=device) + half
    pair_idx[1::2] = torch.arange(half, device=device)
    w13 = w13[:, pair_idx, :].contiguous()
    s13 = s13[:, pair_idx, :].contiguous()
    b13 = b13[:, pair_idx].contiguous()

    w13_indices = _get_flashinfer_mxfp4_device_permute_indices(
        w13[0].view(torch.uint8), _EPILOGUE_TILE_M
    )
    s13_indices = _get_flashinfer_mxfp4_device_permute_indices(
        s13[0].view(torch.uint8), _EPILOGUE_TILE_M, num_elts_per_sf=16
    )
    b13_indices = _get_flashinfer_mxfp4_device_permute_indices(
        b13[0].reshape(-1, 1), _EPILOGUE_TILE_M
    )
    w2_indices = _get_flashinfer_mxfp4_device_permute_indices(
        w2[0].view(torch.uint8), _EPILOGUE_TILE_M
    )
    s2_indices = _get_flashinfer_mxfp4_device_permute_indices(
        s2[0].view(torch.uint8), _EPILOGUE_TILE_M, num_elts_per_sf=16
    )
    b2_indices = _get_flashinfer_mxfp4_device_permute_indices(
        b2[0].reshape(-1, 1), _EPILOGUE_TILE_M
    )

    w13_out = []
    s13_out = []
    b13_out = []
    w2_out = []
    s2_out = []
    b2_out = []
    for expert in range(experts):
        w13_out.append(w13[expert].view(torch.uint8)[w13_indices].contiguous())
        s13_out.append(
            nvfp4_block_scale_interleave(s13[expert].view(torch.uint8)[s13_indices].contiguous())
        )
        b13_out.append(b13[expert].reshape(-1, 1)[b13_indices].contiguous())
        w2_out.append(w2[expert].view(torch.uint8)[w2_indices].contiguous())
        s2_out.append(
            nvfp4_block_scale_interleave(s2[expert].view(torch.uint8)[s2_indices].contiguous())
        )
        b2_out.append(b2[expert].reshape(-1, 1)[b2_indices].contiguous())
    return (
        torch.stack(w13_out),
        torch.stack(s13_out)
        .reshape(experts, 2 * intermediate, hidden // _GROUP_SIZE)
        .view(torch.float8_e4m3fn),
        torch.stack(b13_out).reshape(experts, -1),
        torch.stack(w2_out),
        torch.stack(s2_out)
        .reshape(experts, hidden, intermediate // _GROUP_SIZE)
        .view(torch.float8_e4m3fn),
        torch.stack(b2_out).reshape(experts, -1),
    )


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def profile_mxfp4_fused_moe(
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
    activation: str,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    gemm1_alpha: float,
    gemm1_clamp_limit: float,
    per_expert_batches: tuple[int, ...],
) -> ComputeMetrics:
    args = _validate_args(**locals())
    try:
        import torch
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        from flashinfer.tllm_enums import ActivationType, RoutingMethodType
        from sglang.kernels.ops.quantization.per_token_group_quant import (
            per_token_group_quant,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented("SGLang MXFP4 and FlashInfer are required") from exc
    _require_b200(torch)
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        ids = exact_topk_ids(
            num_tokens=args["num_tokens"],
            top_k=args["top_k"],
            per_expert_batches=args["per_expert_batches"],
        )
        selected = torch.tensor(ids, dtype=torch.int32, device=device)
        topk_weights = torch.full(
            (args["num_tokens"], args["top_k"]),
            1.0 / args["top_k"],
            dtype=torch.float32,
            device=device,
        )
        hidden_states = torch.randn(
            (args["num_tokens"], args["hidden_size"]),
            dtype=torch.bfloat16,
            device=device,
        )
        hidden_states, hidden_states_scale = per_token_group_quant(
            hidden_states, group_size=_GROUP_SIZE, scale_ue8m0=True
        )
        hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn)
        w13, s13, _, w2, s2, _ = _prepare_weights(torch, args)
        alpha = torch.full(
            (args["num_local_experts"],), args["gemm1_alpha"], dtype=torch.float32, device=device
        )
        clamp = torch.full(
            (args["num_local_experts"],),
            args["gemm1_clamp_limit"],
            dtype=torch.float32,
            device=device,
        )
        output = torch.empty(
            (args["num_tokens"], args["hidden_size"]), dtype=torch.bfloat16, device=device
        )

        def run_once() -> Any:
            return trtllm_fp4_block_scale_routed_moe(
                topk_ids=(selected, topk_weights),
                routing_bias=None,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=w13,
                gemm1_weights_scale=s13,
                gemm1_bias=None,
                gemm1_alpha=alpha,
                gemm1_beta=clamp,
                gemm1_clamp_limit=None,
                gemm2_weights=w2,
                gemm2_weights_scale=s2,
                gemm2_bias=None,
                output1_scale_scalar=None,
                output1_scale_gate_scalar=None,
                output2_scale_scalar=None,
                num_experts=args["num_experts"],
                top_k=args["top_k"],
                n_group=None,
                topk_group=None,
                intermediate_size=args["intermediate_size"],
                local_expert_offset=0,
                local_num_experts=args["num_local_experts"],
                routed_scaling_factor=None,
                routing_method_type=RoutingMethodType.TopK.value,
                do_finalize=True,
                enable_pdl=True,
                activation_type=ActivationType.Situ.value,
                output=output,
                tune_max_num_tokens=_next_power_of_two(args["num_tokens"]),
            )

        run_once()
        torch.cuda.synchronize(device)
        time_ms = Timer.cupti(run_once, warmup=3, interval_union=True)
        energy_j = Energy.perf(run_once, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError("K3 MXFP4 fused MoE ran out of memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"K3 MXFP4 fused MoE failed: {exc}") from exc

    local_rows = sum(args["per_expert_batches"][: args["num_local_experts"]])
    flops = (
        2
        * local_rows
        * (
            args["hidden_size"] * 2 * args["intermediate_size"]
            + args["intermediate_size"] * args["hidden_size"]
        )
    )
    seconds = time_ms / 1000.0
    bytes_accessed = (
        args["num_tokens"] * args["top_k"] * (4 + 4)
        + local_rows * args["hidden_size"] * 2
        + args["num_tokens"] * args["hidden_size"] * 2
    )
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / seconds / 1e12 if seconds else 0.0,
        memory_bandwidth_gbps=bytes_accessed / seconds / 1e9 if seconds else 0.0,
        energy_j=float(energy_j),
    )
