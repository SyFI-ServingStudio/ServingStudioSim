"""Profile one call to vLLM's public packed-MXFP4 Marlin MoE GEMM."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "mxfp4_marlin_moe_gemm:vllm_marlin"
_GPU_NAME = "NVIDIA H200"
_LOCAL_EXPERTS = 64


@dataclass(frozen=True)
class _Shape:
    m: int
    n: int
    k: int
    input_top_k: int
    block_size_m: int
    mul_topk_weights: bool
    per_group_batches: tuple[int, ...]

    @property
    def capacity(self) -> int:
        return self.m * self.input_top_k


@dataclass(frozen=True)
class _Launch:
    callable: Callable[..., Any]
    keyword_args: dict[str, Any]

    def run(self) -> Any:
        return self.callable(**self.keyword_args)


def _validate_args(
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
    input_top_k: int,
    block_size_m: int,
    mul_topk_weights: bool,
    per_group_batches: tuple[int, ...] | list[int],
) -> _Shape:
    for name, value in (("m", m), ("n", n), ("k", k), ("input_top_k", input_top_k)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires dtype=bf16")
    if type(mul_topk_weights) is not bool:
        raise TypeError("mul_topk_weights must be a bool")
    supported_block = type(block_size_m) is int and (
        block_size_m == 8 or block_size_m in (16, 32, 48, 64)
    )
    if not supported_block:
        raise ProfilerNotImplemented(f"{_BACKEND} requires block_size_m=8 or 16/32/48/64")

    is_fc1 = (n, k, input_top_k, mul_topk_weights) == (4096, 4096, 6, False)
    is_fc2 = (n, k, input_top_k, mul_topk_weights) == (4096, 2048, 1, True)
    if not (is_fc1 or is_fc2):
        raise ProfilerNotImplemented(f"{_BACKEND} supports only DeepSeek V4 FC1/FC2 shapes")

    batches = tuple(per_group_batches)
    if len(batches) != _LOCAL_EXPERTS:
        raise ProfilerNotImplemented(f"{_BACKEND} requires {_LOCAL_EXPERTS} local experts")
    if any(type(count) is not int or count < 0 for count in batches):
        raise ValueError("per_group_batches must contain non-negative integers")
    if not 0 < sum(batches) <= m * input_top_k:
        raise ValueError("per_group_batches must describe 1..m*input_top_k local rows")
    return _Shape(m, n, k, input_top_k, block_size_m, mul_topk_weights, batches)


def _build_routes(torch: Any, shape: _Shape, align_routes: Callable[..., Any]) -> tuple[Any, ...]:
    local_ids = [
        expert for expert, count in enumerate(shape.per_group_batches) for _ in range(count)
    ]
    remote_rows = shape.capacity - len(local_ids)
    num_global_experts = _LOCAL_EXPERTS + int(remote_rows > 0)
    route_ids_cpu = torch.tensor(
        local_ids + [_LOCAL_EXPERTS] * remote_rows,
        dtype=torch.int32,
    ).reshape(shape.m, shape.input_top_k)
    route_ids = route_ids_cpu.cuda()
    router_weights = (
        torch.linspace(0.25, 1.0, shape.capacity).reshape(shape.m, shape.input_top_k).cuda()
    )
    expert_map = torch.arange(num_global_experts, dtype=torch.int32)
    if remote_rows:
        expert_map[-1] = -1
    sorted_token_ids, expert_ids, post_padded = align_routes(
        route_ids,
        shape.block_size_m,
        num_global_experts,
        expert_map.cuda(),
        ignore_invalid_experts=True,
    )
    return route_ids_cpu, router_weights, sorted_token_ids, expert_ids, post_padded


def _make_weights(torch: Any, shape: _Shape, dependencies: dict[str, Any]) -> tuple[Any, ...]:
    generator = torch.Generator().manual_seed(17)
    scale_bits = torch.randint(
        110, 120, (shape.n, shape.k // 32), dtype=torch.uint8, generator=generator
    )
    raw_scales = scale_bits.view(torch.float8_e8m0fnu)
    raw_fp4 = torch.randint(0, 256, (shape.n, shape.k // 2), dtype=torch.uint8, generator=generator)
    qweight = raw_fp4.view(torch.int32).T.contiguous().cuda()
    packed = dependencies["repack"](
        qweight, torch.empty(0, dtype=torch.int32, device="cuda"), shape.k, shape.n, 4
    )
    packed_weights = packed.cpu().unsqueeze(0).repeat(_LOCAL_EXPERTS, 1, 1).cuda()
    permuted_scales = dependencies["permute_scales"](
        raw_scales.T.to(torch.bfloat16), shape.k, shape.n, 32, False
    )
    processed_scales = dependencies["process_scales"](permuted_scales, input_dtype=torch.bfloat16)
    weight_scales = processed_scales.unsqueeze(0).repeat(_LOCAL_EXPERTS, 1, 1).cuda()

    high = ((raw_fp4 & 0x80) | ((raw_fp4 & 0x70) >> 2)).view(torch.float8_e4m3fn)
    low_bits = raw_fp4 << 4
    low = ((low_bits & 0x80) | ((low_bits & 0x70) >> 2)).view(torch.float8_e4m3fn)
    logical_weight = torch.stack((low, high), dim=2).reshape(shape.n, shape.k)
    logical_weight = logical_weight.to(torch.bfloat16) * 64
    logical_weight *= raw_scales.repeat_interleave(32, dim=1).to(torch.bfloat16)
    return packed_weights, weight_scales, logical_weight.T.contiguous()


def _prepare(torch: Any, shape: _Shape, dependencies: dict[str, Any]) -> tuple[_Launch, Any, Any]:
    device = torch.device("cuda", torch.cuda.current_device())
    packed_weights, weight_scales, logical_weight = _make_weights(torch, shape, dependencies)
    route_ids_cpu, router_weights, sorted_ids, expert_ids, post_padded = _build_routes(
        torch, shape, dependencies["align_routes"]
    )
    generator = torch.Generator().manual_seed(23)
    activation = (
        torch.randn((shape.m, shape.k), dtype=torch.bfloat16, generator=generator) / 16
    ).cuda()
    output = torch.zeros((shape.capacity, shape.n), dtype=torch.bfloat16).cuda()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.zeros(sms * 4, dtype=torch.int32).cuda()
    keyword_args = {
        "input": activation,
        "output": output,
        "b_qweight": packed_weights,
        "b_bias": None,
        "b_scales": weight_scales,
        "a_scales": None,
        "global_scale": None,
        "b_qzeros": None,
        "g_idx": None,
        "perm": None,
        "workspace": workspace,
        "sorted_token_ids": sorted_ids,
        "expert_ids": expert_ids,
        "num_tokens_past_padded": post_padded,
        "topk_weights": router_weights,
        "moe_block_size": shape.block_size_m,
        "top_k": shape.input_top_k,
        "mul_topk_weights": shape.mul_topk_weights,
        "b_q_type": dependencies["quant_type"],
        "size_m": shape.m,
        "size_n": shape.n,
        "size_k": shape.k,
        "is_k_full": True,
        "use_atomic_add": False,
        "use_fp32_reduce": True,
        "is_zp_float": False,
    }
    return _Launch(dependencies["callable"], keyword_args), logical_weight, route_ids_cpu


def _check_output(torch: Any, launch: _Launch, logical_weight: Any, route_ids: Any) -> None:
    launch.run()
    torch.cuda.synchronize()
    sample_flat_row = int(torch.nonzero(route_ids.reshape(-1) < _LOCAL_EXPERTS).flatten()[0])
    activation = launch.keyword_args["input"]
    source_row = sample_flat_row // launch.keyword_args["top_k"]
    expected = activation[source_row].cpu().float() @ logical_weight.float()
    if launch.keyword_args["mul_topk_weights"]:
        weight = launch.keyword_args["topk_weights"].reshape(-1)[sample_flat_row].cpu()
        expected *= weight
    observed = launch.keyword_args["output"][sample_flat_row].cpu().float()
    torch.testing.assert_close(observed, expected, atol=0.5, rtol=0.02)


def _logical_bytes(shape: _Shape) -> int:
    local_rows = sum(shape.per_group_batches)
    activation = local_rows * shape.k * 2
    active_experts = sum(count > 0 for count in shape.per_group_batches)
    weights = active_experts * shape.n * shape.k // 2
    scales = active_experts * shape.n * (shape.k // 32)
    output = local_rows * shape.n * 2
    router = local_rows * 4 if shape.mul_topk_weights else 0
    return activation + weights + scales + output + router


def profile_mxfp4_marlin_moe_gemm_vllm_marlin(
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
    input_top_k: int,
    block_size_m: int,
    mul_topk_weights: bool,
    per_group_batches: tuple[int, ...] | list[int],
) -> ComputeMetrics:
    shape = _validate_args(
        m, n, k, dtype, input_top_k, block_size_m, mul_topk_weights, per_group_batches
    )
    try:
        import torch
        from vllm import _custom_ops
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_permute_scales,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            mxfp4_marlin_process_scales,
        )
        from vllm.scalar_type import scalar_types
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM environment") from exc

    try:
        if not torch.cuda.is_available():
            raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
        gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
        if gpu_name != _GPU_NAME:
            raise ProfilerNotImplemented(
                f"{_BACKEND} is verified only on {_GPU_NAME}, got {gpu_name}"
            )
        launch, logical_weight, route_ids = _prepare(
            torch,
            shape,
            {
                "callable": _custom_ops.moe_wna16_marlin_gemm,
                "align_routes": moe_align_block_size,
                "repack": _custom_ops.gptq_marlin_repack,
                "permute_scales": marlin_permute_scales,
                "process_scales": mxfp4_marlin_process_scales,
                "quant_type": scalar_types.float4_e2m1f,
            },
        )
        _check_output(torch, launch, logical_weight, route_ids)
        time_ms = Timer.cupti(launch.run, warmup=5, kernel_name=None)
        energy_j = Energy.perf(launch.run, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    elapsed_seconds = time_ms / 1000.0
    flops = 2 * sum(shape.per_group_batches) * shape.n * shape.k
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / elapsed_seconds / 1e12 if elapsed_seconds else 0.0),
        memory_bandwidth_gbps=float(
            _logical_bytes(shape) / elapsed_seconds / 1e9 if elapsed_seconds else 0.0
        ),
        energy_j=float(energy_j),
    )


__all__ = ["profile_mxfp4_marlin_moe_gemm_vllm_marlin"]
