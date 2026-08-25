"""Profile vLLM's production MoE alignment CUDA callable."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.moe.moe_align_block_size_reference import (
    AlignmentShape,
    build_topk_ids,
    ceil_div,
    expected_block_owners,
    logical_bytes,
    validate_shape,
)

_BACKEND = "moe_align_block_size:vllm_cuda"
_GPU_NAME = "NVIDIA H200"


@dataclass
class _Launch:
    callable: Callable[..., None]
    topk_ids: Any
    num_experts: int
    block_size: int
    sorted_token_ids: Any
    expert_ids: Any
    num_tokens_post_pad: Any

    def run(self) -> None:
        self.callable(
            self.topk_ids,
            self.num_experts,
            self.block_size,
            self.sorted_token_ids,
            self.expert_ids,
            self.num_tokens_post_pad,
            None,
        )


def _prepare(torch: Any, callable_: Callable[..., None], shape: AlignmentShape) -> _Launch:
    device = torch.device("cuda", torch.cuda.current_device())
    topk_ids = torch.tensor(build_topk_ids(shape), dtype=torch.int32, device=device)
    capacity = shape.num_routes + shape.num_experts * (shape.block_size - 1)
    if shape.num_routes < shape.num_experts:
        capacity = min(shape.num_routes * shape.block_size, capacity)
    sorted_token_ids = torch.full((capacity,), shape.num_routes, dtype=torch.int32, device=device)
    expert_ids = torch.full(
        (ceil_div(capacity, shape.block_size),), -1, dtype=torch.int32, device=device
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    return _Launch(
        callable=callable_,
        topk_ids=topk_ids,
        num_experts=shape.num_experts,
        block_size=shape.block_size,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_pad=num_tokens_post_pad,
    )


def _check_outputs(torch: Any, launch: _Launch, shape: AlignmentShape) -> None:
    launch.run()
    torch.cuda.synchronize()
    padded_routes = int(launch.num_tokens_post_pad.item())
    if padded_routes != shape.padded_routes:
        raise AssertionError(
            f"padded routes differ: expected {shape.padded_routes}, got {padded_routes}"
        )

    owners = tuple(int(value) for value in launch.expert_ids[: padded_routes // shape.block_size])
    expected_owners = expected_block_owners(shape.expert_counts, shape.block_size)
    if owners != expected_owners:
        raise AssertionError(f"block owners differ: expected {expected_owners}, got {owners}")

    flat_routes = launch.topk_ids.cpu().reshape(-1)
    sorted_routes = launch.sorted_token_ids.cpu()
    sentinel = shape.num_routes
    for expert in range(shape.num_experts):
        observed = sorted(
            route_id
            for block, owner in enumerate(owners)
            if owner == expert
            for route_id in sorted_routes[
                block * shape.block_size : (block + 1) * shape.block_size
            ].tolist()
            if route_id != sentinel
        )
        expected = torch.nonzero(flat_routes == expert, as_tuple=False).flatten().tolist()
        if observed != expected:
            raise AssertionError(f"routes for local expert {expert} differ")


def _require_h200(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _GPU_NAME:
        raise ProfilerNotImplemented(f"{_BACKEND} is verified only on {_GPU_NAME}, got {gpu_name}")


def profile_moe_align_block_size_vllm_cuda(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    block_size: int,
) -> ComputeMetrics:
    shape = validate_shape(num_tokens, num_experts, top_k, block_size)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires PyTorch") from exc
    try:
        from vllm import _custom_ops
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM checkout") from exc

    try:
        _require_h200(torch)
        launch = _prepare(torch, _custom_ops.moe_align_block_size, shape)
        _check_outputs(torch, launch, shape)
        time_ms = Timer.cupti(launch.run, warmup=5, kernel_name=None)
        energy_j = Energy.perf(launch.run, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    elapsed_seconds = time_ms / 1000.0
    bandwidth = logical_bytes(shape) / elapsed_seconds / 1e9 if elapsed_seconds else 0.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(bandwidth),
        energy_j=float(energy_j),
    )
