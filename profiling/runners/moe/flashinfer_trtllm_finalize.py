"""Direct FlashInfer/TensorRT-LLM BF16 MoE finalize-routing runner."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_CUDA_MINIMUM = (12, 8)
_HOPPER_COMPUTE_CAPABILITY = (9, 0)
_KERNEL_NAME = "finalizeMoeRoutingKernel"
_JIT_MODULE_NAME = "vibesim_moe_finalize_routing_bf16_sm90"


@dataclass(frozen=True)
class MoeFinalizeRoutingLaunch:
    run_once: Callable[[], None]
    expanded_permuted_rows: Any
    reduced_unpermuted_output: Any
    final_scales: Any
    unpermuted_row_to_permuted_row: Any
    token_selected_experts: Any
    token_count: int
    hidden_size: int
    top_k: int
    num_experts_per_rank: int
    local_routed_token_count: int


def _validate_args(
    token_count: int,
    hidden_size: int,
    top_k: int,
    num_experts_per_rank: int,
    local_routed_token_count: int,
    dtype: DType | str,
) -> tuple[int, int, int, int, int, DType]:
    token_count = int(token_count)
    hidden_size = int(hidden_size)
    top_k = int(top_k)
    num_experts_per_rank = int(num_experts_per_rank)
    local_routed_token_count = int(local_routed_token_count)
    dtype = DType.from_value(dtype)

    if token_count <= 0:
        raise ValueError(f"token_count must be > 0, got {token_count}")
    if hidden_size <= 0 or hidden_size % 8 != 0:
        raise ValueError(f"hidden_size must be > 0 and divisible by 8, got {hidden_size}")
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if num_experts_per_rank <= 0:
        raise ValueError(f"num_experts_per_rank must be > 0, got {num_experts_per_rank}")
    routed_capacity = token_count * top_k
    if local_routed_token_count < 0 or local_routed_token_count > routed_capacity:
        raise ValueError(
            "local_routed_token_count must be in "
            f"[0, token_count*top_k={routed_capacity}], got {local_routed_token_count}"
        )
    if dtype is not DType.BF16:
        raise ValueError(
            f"flashinfer_trtllm moe_finalize_routing requires dtype=bf16, got {dtype.value}"
        )
    return (
        token_count,
        hidden_size,
        top_k,
        num_experts_per_rank,
        local_routed_token_count,
        dtype,
    )


def _balanced_local_slots(routed_capacity: int, local_count: int) -> tuple[bool, ...]:
    """Spread exactly ``local_count`` assignments across the routed slots."""

    if routed_capacity <= 0:
        raise ValueError("routed_capacity must be > 0")
    if local_count < 0 or local_count > routed_capacity:
        raise ValueError("local_count must be within routed_capacity")
    return tuple(
        ((slot_index + 1) * local_count) // routed_capacity
        > (slot_index * local_count) // routed_capacity
        for slot_index in range(routed_capacity)
    )


def _routing_metadata(
    token_count: int,
    top_k: int,
    num_experts_per_rank: int,
    local_routed_token_count: int,
) -> tuple[list[int], list[int], list[float]]:
    routed_capacity = token_count * top_k
    local_slots = _balanced_local_slots(routed_capacity, local_routed_token_count)
    selected_experts: list[int] = []
    local_assignment_index = 0
    for is_local in local_slots:
        if is_local:
            selected_experts.append(local_assignment_index % num_experts_per_rank)
            local_assignment_index += 1
        else:
            # Rank zero owns [0, E); E is the first expert on the synthetic
            # remote rank and is therefore skipped by the production kernel.
            selected_experts.append(num_experts_per_rank)

    # The upstream permutation groups valid rows contiguously by local expert.
    # Preserve that physical locality rather than merely inventing an arbitrary
    # bijection: the finalize kernel follows this map for every local load.
    local_expert_counts = [0] * num_experts_per_rank
    for expert_id in selected_experts:
        if expert_id < num_experts_per_rank:
            local_expert_counts[expert_id] += 1
    local_expert_offsets: list[int] = []
    running_offset = 0
    for expert_count in local_expert_counts:
        local_expert_offsets.append(running_offset)
        running_offset += expert_count
    next_local_row = local_expert_offsets.copy()
    next_remote_row = local_routed_token_count

    unpermute_map = [0] * routed_capacity
    for token_index in range(token_count):
        for top_k_index in range(top_k):
            column_major_slot = token_index + top_k_index * token_count
            row_major_slot = token_index * top_k + top_k_index
            expert_id = selected_experts[row_major_slot]
            if expert_id < num_experts_per_rank:
                unpermute_map[column_major_slot] = next_local_row[expert_id]
                next_local_row[expert_id] += 1
            else:
                # Remote mappings are not dereferenced by this rank, but a full
                # permutation keeps the synthetic metadata structurally legal.
                unpermute_map[column_major_slot] = next_remote_row
                next_remote_row += 1

    # Non-uniform positive scales make the correctness check exercise DEFAULT
    # scaling rather than accidentally accepting the NO_SCALE specialization.
    final_scales = [0.25 + 0.5 * ((slot_index % 7) / 6.0) for slot_index in range(routed_capacity)]
    return selected_experts, unpermute_map, final_scales


def _parse_cuda_version(cuda_version: object) -> tuple[int, int] | None:
    if cuda_version is None:
        return None
    version_parts = str(cuda_version).split(".")
    if len(version_parts) < 2:
        return None
    try:
        return int(version_parts[0]), int(version_parts[1])
    except ValueError:
        return None


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the flashinfer_trtllm moe_finalize_routing backend"
        )
    cuda_version = _parse_cuda_version(getattr(torch.version, "cuda", None))
    if cuda_version is None or cuda_version < _CUDA_MINIMUM:
        rendered_version = getattr(torch.version, "cuda", None)
        raise ProfilerNotImplemented(
            f"flashinfer_trtllm moe_finalize_routing requires CUDA >= 12.8, got {rendered_version}"
        )
    device = torch.cuda.current_device()
    compute_capability = tuple(torch.cuda.get_device_capability(device))
    if compute_capability != _HOPPER_COMPUTE_CAPABILITY:
        gpu_name = str(torch.cuda.get_device_name(device))
        raise ProfilerNotImplemented(
            "flashinfer_trtllm moe_finalize_routing requires SM90/SM90a Hopper, "
            f"got {gpu_name} with SM{compute_capability[0]}{compute_capability[1]}"
        )


@functools.cache
def _load_finalize_module():
    """Build the thin launcher against the selected FlashInfer wheel."""

    try:
        from flashinfer.jit import env as jit_env
        from flashinfer.jit.core import gen_jit_spec, sm90a_nvcc_flags
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "FlashInfer JIT support is required for moe_finalize_routing"
        ) from exc

    binding_source = Path(__file__).resolve().parent / "csrc" / "moe_finalize_routing.cu"
    if not binding_source.is_file():
        raise ProfilerNotImplemented(f"MoE finalize JIT binding is missing: {binding_source}")

    source_root = jit_env.FLASHINFER_CSRC_DIR
    common_source_root = source_root / "nv_internal" / "cpp" / "common"
    include_paths = [
        source_root,
        source_root / "nv_internal",
        source_root / "nv_internal" / "include",
        source_root / "nv_internal" / "tensorrt_llm" / "cutlass_extensions" / "include",
        source_root / "nv_internal" / "tensorrt_llm" / "kernels" / "cutlass_kernels" / "include",
        source_root / "nv_internal" / "tensorrt_llm" / "kernels" / "cutlass_kernels",
    ]
    nvcc_flags = sm90a_nvcc_flags + [
        "-DCOMPILE_HOPPER_TMA_GEMMS",
        "-DCOMPILE_HOPPER_TMA_GROUPED_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP8_BLOCK_SCALE",
        "-DUSING_OSS_CUTLASS_MOE_GEMM",
        "-DCUTLASS_ENABLE_GDC_FOR_SM90=1",
    ]
    return gen_jit_spec(
        _JIT_MODULE_NAME,
        [
            binding_source,
            common_source_root / "envUtils.cpp",
            common_source_root / "logger.cpp",
            common_source_root / "stringUtils.cpp",
            common_source_root / "tllmException.cpp",
        ],
        extra_cuda_cflags=nvcc_flags,
        extra_include_paths=include_paths,
    ).build_and_load()


def prepare_moe_finalize_routing_launch(
    torch: Any,
    *,
    token_count: int,
    hidden_size: int,
    top_k: int,
    num_experts_per_rank: int,
    local_routed_token_count: int,
) -> MoeFinalizeRoutingLaunch:
    selected_experts, unpermute_map, scale_values = _routing_metadata(
        token_count,
        top_k,
        num_experts_per_rank,
        local_routed_token_count,
    )
    routed_capacity = token_count * top_k
    expanded_permuted_rows = torch.randn(
        (routed_capacity, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    reduced_unpermuted_output = torch.empty(
        (token_count, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    final_scales = torch.tensor(scale_values, dtype=torch.float32, device="cuda")
    unpermuted_row_to_permuted_row = torch.tensor(unpermute_map, dtype=torch.int32, device="cuda")
    token_selected_experts = torch.tensor(selected_experts, dtype=torch.int32, device="cuda")

    def run_once() -> None:
        module = _load_finalize_module()
        module.run_moe_finalize_routing(
            expanded_permuted_rows,
            reduced_unpermuted_output,
            final_scales,
            unpermuted_row_to_permuted_row,
            token_selected_experts,
            token_count,
            hidden_size,
            top_k,
            num_experts_per_rank,
        )

    return MoeFinalizeRoutingLaunch(
        run_once=run_once,
        expanded_permuted_rows=expanded_permuted_rows,
        reduced_unpermuted_output=reduced_unpermuted_output,
        final_scales=final_scales,
        unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
        token_selected_experts=token_selected_experts,
        token_count=token_count,
        hidden_size=hidden_size,
        top_k=top_k,
        num_experts_per_rank=num_experts_per_rank,
        local_routed_token_count=local_routed_token_count,
    )


def torch_reference(launch: MoeFinalizeRoutingLaunch, torch: Any) -> Any:
    """Small-shape semantic reference for the direct launcher."""

    output = torch.zeros(
        (launch.token_count, launch.hidden_size),
        dtype=torch.float32,
        device=launch.expanded_permuted_rows.device,
    )
    expanded = launch.expanded_permuted_rows.float()
    for token_index in range(launch.token_count):
        for top_k_index in range(launch.top_k):
            routed_slot = token_index * launch.top_k + top_k_index
            expert_id = int(launch.token_selected_experts[routed_slot].item())
            if expert_id >= launch.num_experts_per_rank:
                continue
            mapping_slot = token_index + top_k_index * launch.token_count
            permuted_row = int(launch.unpermuted_row_to_permuted_row[mapping_slot].item())
            scale = launch.final_scales[routed_slot]
            output[token_index] += expanded[permuted_row] * scale
    return output.to(torch.bfloat16)


def _logical_bytes(
    token_count: int,
    hidden_size: int,
    top_k: int,
    local_routed_token_count: int,
) -> int:
    return (
        local_routed_token_count * hidden_size * 2
        + token_count * hidden_size * 2
        + local_routed_token_count * 4
        + local_routed_token_count * 4
        + token_count * top_k * 4
    )


def profile_moe_finalize_routing_flashinfer_trtllm(
    token_count: int,
    hidden_size: int,
    top_k: int,
    num_experts_per_rank: int,
    local_routed_token_count: int,
    dtype: DType | str,
) -> ComputeMetrics:
    (
        token_count,
        hidden_size,
        top_k,
        num_experts_per_rank,
        local_routed_token_count,
        _,
    ) = _validate_args(
        token_count,
        hidden_size,
        top_k,
        num_experts_per_rank,
        local_routed_token_count,
        dtype,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for moe_finalize_routing") from exc

    _validate_cuda_device(torch)
    try:
        launch = prepare_moe_finalize_routing_launch(
            torch,
            token_count=token_count,
            hidden_size=hidden_size,
            top_k=top_k,
            num_experts_per_rank=num_experts_per_rank,
            local_routed_token_count=local_routed_token_count,
        )
        launch.run_once()
        torch.cuda.synchronize()
        time_ms = Timer.cupti(launch.run_once, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(launch.run_once, per_iter_time_ms=time_ms)

        flops = 2 * local_routed_token_count * hidden_size
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0 else 0.0
        bytes_accessed = _logical_bytes(token_count, hidden_size, top_k, local_routed_token_count)
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except (RuntimeError, ValueError) as exc:
        raise KernelLaunchFailed(str(exc)) from exc


__all__ = [
    "MoeFinalizeRoutingLaunch",
    "prepare_moe_finalize_routing_launch",
    "profile_moe_finalize_routing_flashinfer_trtllm",
    "torch_reference",
]
