"""Direct FlashInfer/TensorRT-LLM FP8 block-scale grouped GEMM runner.

This module profiles the production ``GroupedWithOffset`` GEMM launch without
constructing token-to-expert routing.  Routing and EP ownership are already
resolved by the caller as ``per_group_batches``.  The thin TVM-FFI binding calls
FlashInfer's vendored ``grouped_gemm_dispatch`` directly; the DeepGEMM kernel,
recipe heuristic, and runtime JIT source stay vendored with FlashInfer.

Physical ABI at the binding boundary (all tensors contiguous):

* activation: ``[align4(num_input_tokens * experts_per_token), K]`` FP8 E4M3;
* activation scales: ``[ceil(K/128), padded_M]`` FP32;
* weight: ``[E, N, K]`` FP8 E4M3.  This row-major storage is the col-major
  ``[K, N]`` matrix consumed by DeepGEMM;
* weight scales: ``[E, ceil(N/128), ceil(K/128)]`` FP32;
* output: ``[align4(num_input_tokens * experts_per_token), N]`` BF16.

Only rows below ``problem_m_offsets[-1]`` are real local work.  The larger
capacity is nevertheless part of the production recipe: TensorRT-LLM sets
``expected_m=num_input_tokens`` and allocates for
``num_input_tokens * experts_per_token`` before EP-local offsets are known.
"""

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

_BLOCK_SIZE = 128
_CUDA_MINIMUM = (12, 8)
_HOPPER_COMPUTE_CAPABILITY = (9, 0)
_JIT_MODULE_NAME = "vibesim_flashinfer_trtllm_fp8_blockscale_grouped_gemm_sm90"


@dataclass(frozen=True)
class DirectGroupedGemmLaunch:
    """Prepared direct launch plus the tensors that define its physical ABI."""

    run_once: Callable[[], None]
    activation: Any
    activation_scales: Any
    weight: Any
    weight_scales: Any
    output: Any
    problem_m_offsets: Any
    expected_m: int
    max_shape_m: int
    max_shape_m_padded: int
    total_local_rows: int
    n: int
    k: int
    num_local_experts: int


def _align_up(value: int, alignment: int) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _compute_grouped_padded_offset(offset: int, problem_index: int) -> int:
    """Mirror ``deep_gemm::compute_padded_offset`` for buffer sizing."""

    alignment = 32
    return (int(offset) + int(problem_index) * (alignment - 1)) // alignment * alignment


def _validate_args(
    n: int,
    k: int,
    dtype: DType | str,
    num_local_experts: int,
    per_group_batches: tuple[int, ...] | list[int],
    num_input_tokens: int,
    experts_per_token: int,
) -> tuple[int, int, DType, int, tuple[int, ...], int, int]:
    n = int(n)
    k = int(k)
    dtype = DType.from_value(dtype)
    num_local_experts = int(num_local_experts)
    batches = tuple(int(batch_size) for batch_size in per_group_batches)
    num_input_tokens = int(num_input_tokens)
    experts_per_token = int(experts_per_token)

    if n <= 0 or n % _BLOCK_SIZE != 0:
        raise ValueError(f"n must be > 0 and divisible by {_BLOCK_SIZE}, got {n}")
    if k <= 0 or k % _BLOCK_SIZE != 0:
        raise ValueError(f"k must be > 0 and divisible by {_BLOCK_SIZE}, got {k}")
    if dtype is not DType.FP8_E4M3:
        raise ValueError(
            "flashinfer_trtllm FP8 block-scale grouped GEMM requires "
            f"dtype=fp8_e4m3, got {dtype.value}"
        )
    if num_local_experts <= 0:
        raise ValueError(f"num_local_experts must be > 0, got {num_local_experts}")
    if len(batches) != num_local_experts:
        raise ValueError(
            f"per_group_batches has {len(batches)} entries, expected "
            f"num_local_experts={num_local_experts}"
        )
    if any(batch_size < 0 for batch_size in batches):
        raise ValueError("per_group_batches must contain only non-negative counts")
    if not any(batches):
        raise ValueError("per_group_batches sums to 0; no local rows to profile")
    if num_input_tokens <= 0:
        raise ValueError(f"num_input_tokens must be > 0, got {num_input_tokens}")
    if experts_per_token <= 0:
        raise ValueError(f"experts_per_token must be > 0, got {experts_per_token}")

    routed_capacity = num_input_tokens * experts_per_token
    total_local_rows = sum(batches)
    if total_local_rows > routed_capacity:
        raise ValueError(
            f"local routed rows {total_local_rows} exceed global routed capacity "
            f"num_input_tokens*experts_per_token={routed_capacity}"
        )
    return (
        n,
        k,
        dtype,
        num_local_experts,
        batches,
        num_input_tokens,
        experts_per_token,
    )


def _problem_m_boundaries(batches: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    boundaries = [0]
    current_offset = 0
    for batch_size in batches:
        current_offset += int(batch_size)
        boundaries.append(current_offset)
    return tuple(boundaries)


def _validate_problem_boundaries(
    boundaries: tuple[int, ...],
    *,
    num_local_experts: int,
    total_local_rows: int,
    max_shape_m: int,
) -> None:
    if len(boundaries) != num_local_experts + 1:
        raise ValueError("problem_m_offsets must contain num_local_experts + 1 boundaries")
    if not boundaries or boundaries[0] != 0:
        raise ValueError("problem_m_offsets must start at 0")
    if any(right < left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("problem_m_offsets must be monotonic")
    if boundaries[-1] != total_local_rows:
        raise ValueError("problem_m_offsets final boundary must equal sum(per_group_batches)")
    if boundaries[-1] > max_shape_m:
        raise ValueError("problem_m_offsets final boundary exceeds max_shape_m")


def _production_capacity(
    num_input_tokens: int,
    experts_per_token: int,
    num_local_experts: int,
) -> tuple[int, int]:
    max_shape_m = _align_up(num_input_tokens * experts_per_token, 4)
    max_shape_m_padded = _compute_grouped_padded_offset(max_shape_m, num_local_experts)
    return max_shape_m, max_shape_m_padded


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


def _validate_cuda_device(torch: Any) -> int:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the FlashInfer/TensorRT-LLM FP8 block-scale grouped GEMM"
        )
    cuda_version = _parse_cuda_version(getattr(torch.version, "cuda", None))
    if cuda_version is None or cuda_version < _CUDA_MINIMUM:
        rendered_version = getattr(torch.version, "cuda", None)
        raise ProfilerNotImplemented(
            "FlashInfer/TensorRT-LLM FP8 block-scale grouped GEMM requires "
            f"CUDA >= 12.8, got {rendered_version}"
        )
    device = torch.cuda.current_device()
    compute_capability = tuple(torch.cuda.get_device_capability(device))
    if compute_capability != _HOPPER_COMPUTE_CAPABILITY:
        gpu_name = str(torch.cuda.get_device_name(device))
        raise ProfilerNotImplemented(
            "FlashInfer/TensorRT-LLM FP8 block-scale grouped GEMM requires SM90/SM90a, "
            f"got {gpu_name} with SM{compute_capability[0]}{compute_capability[1]}"
        )
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


@functools.cache
def _load_direct_grouped_gemm_module():
    """Build the thin binding against the selected FlashInfer wheel."""

    try:
        from flashinfer.jit import env as jit_env
        from flashinfer.jit.core import gen_jit_spec, sm90a_nvcc_flags
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "FlashInfer JIT support is required for the direct FP8 block-scale grouped GEMM"
        ) from exc

    binding_source = (
        Path(__file__).resolve().parent / "csrc" / "flashinfer_trtllm_blockscale_grouped_gemm.cu"
    )
    if not binding_source.is_file():
        raise ProfilerNotImplemented(
            f"direct grouped GEMM JIT binding is missing: {binding_source}"
        )

    source_root = jit_env.FLASHINFER_CSRC_DIR
    common_source_root = source_root / "nv_internal" / "cpp" / "common"
    nvcc_flags = sm90a_nvcc_flags + [
        "-DCOMPILE_HOPPER_TMA_GEMMS",
        "-DCOMPILE_HOPPER_TMA_GROUPED_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP8_BLOCK_SCALE",
        "-DUSING_OSS_CUTLASS_MOE_GEMM",
        "-DCUTLASS_ENABLE_GDC_FOR_SM90=1",
    ]
    include_paths = [
        source_root,
        source_root / "nv_internal",
        source_root / "nv_internal" / "include",
        source_root / "nv_internal" / "tensorrt_llm" / "cutlass_extensions" / "include",
        source_root / "nv_internal" / "tensorrt_llm" / "kernels" / "cutlass_kernels" / "include",
        source_root / "nv_internal" / "tensorrt_llm" / "kernels" / "cutlass_kernels",
    ]
    module = gen_jit_spec(
        _JIT_MODULE_NAME,
        [
            binding_source,
            source_root / "fused_moe" / "cutlass_backend" / "deepgemm_jit_setup.cu",
            common_source_root / "envUtils.cpp",
            common_source_root / "logger.cpp",
            common_source_root / "stringUtils.cpp",
            common_source_root / "tllmException.cpp",
        ],
        extra_cuda_cflags=nvcc_flags,
        extra_ldflags=["-lnvrtc"],
        extra_include_paths=include_paths,
    ).build_and_load()
    deepgemm_include_root = source_root / "nv_internal" / "tensorrt_llm"
    module.set_deepgemm_jit_include_dirs([str(deepgemm_include_root)])
    return module


def run_fp8_blockscale_grouped_gemm_direct(
    activation: Any,
    activation_scales: Any,
    weight: Any,
    weight_scales: Any,
    output: Any,
    problem_m_offsets: Any,
    *,
    expected_m: int,
    max_shape_m: int,
    max_shape_m_padded: int,
    n: int,
    k: int,
    num_local_experts: int,
) -> None:
    """Invoke the direct production-physical ABI; C++ validates tensor shapes."""

    module = _load_direct_grouped_gemm_module()
    module.run_fp8_blockscale_grouped_gemm(
        activation,
        activation_scales,
        weight,
        weight_scales,
        output,
        problem_m_offsets,
        int(expected_m),
        int(max_shape_m),
        int(max_shape_m_padded),
        int(n),
        int(k),
        int(num_local_experts),
    )


def prepare_fp8_blockscale_grouped_gemm_launch(
    torch: Any,
    *,
    n: int,
    k: int,
    dtype: DType | str,
    num_local_experts: int,
    per_group_batches: tuple[int, ...] | list[int],
    num_input_tokens: int,
    experts_per_token: int,
) -> DirectGroupedGemmLaunch:
    """Allocate the exact production capacity and physical tensor layouts."""

    (
        n,
        k,
        _,
        num_local_experts,
        batches,
        num_input_tokens,
        experts_per_token,
    ) = _validate_args(
        n,
        k,
        dtype,
        num_local_experts,
        per_group_batches,
        num_input_tokens,
        experts_per_token,
    )
    max_shape_m, max_shape_m_padded = _production_capacity(
        num_input_tokens,
        experts_per_token,
        num_local_experts,
    )
    boundaries = _problem_m_boundaries(batches)
    _validate_problem_boundaries(
        boundaries,
        num_local_experts=num_local_experts,
        total_local_rows=sum(batches),
        max_shape_m=max_shape_m,
    )

    activation = torch.empty((max_shape_m, k), dtype=torch.float8_e4m3fn, device="cuda")
    activation_scales = torch.ones(
        (k // _BLOCK_SIZE, max_shape_m_padded), dtype=torch.float32, device="cuda"
    )
    # This physical [E, N, K] row-major storage is the col-major [K, N]
    # matrix consumed by DeepGEMM's TMA descriptor.
    weight = torch.empty((num_local_experts, n, k), dtype=torch.float8_e4m3fn, device="cuda")
    weight_scales = torch.ones(
        (num_local_experts, n // _BLOCK_SIZE, k // _BLOCK_SIZE),
        dtype=torch.float32,
        device="cuda",
    )
    output = torch.empty((max_shape_m, n), dtype=torch.bfloat16, device="cuda")
    problem_m_offsets = torch.tensor(boundaries, dtype=torch.int64, device="cuda")

    def run_once() -> None:
        run_fp8_blockscale_grouped_gemm_direct(
            activation,
            activation_scales,
            weight,
            weight_scales,
            output,
            problem_m_offsets,
            expected_m=num_input_tokens,
            max_shape_m=max_shape_m,
            max_shape_m_padded=max_shape_m_padded,
            n=n,
            k=k,
            num_local_experts=num_local_experts,
        )

    return DirectGroupedGemmLaunch(
        run_once=run_once,
        activation=activation,
        activation_scales=activation_scales,
        weight=weight,
        weight_scales=weight_scales,
        output=output,
        problem_m_offsets=problem_m_offsets,
        expected_m=num_input_tokens,
        max_shape_m=max_shape_m,
        max_shape_m_padded=max_shape_m_padded,
        total_local_rows=sum(batches),
        n=n,
        k=k,
        num_local_experts=num_local_experts,
    )


def _uses_swapped_ab(expected_m: int, num_device_sms: int) -> bool:
    threshold = 64 if num_device_sms == 78 else 32
    return expected_m < threshold


def _kernel_name_filter(n: int, k: int, *, swapped_ab: bool) -> str:
    """Return the shape-exact Itanium symbol prefix emitted by CUPTI.

    CUPTI activity records expose the raw symbol on this stack.  Keeping N and
    K in the prefix avoids accidentally folding another grouped GEMM launched
    by a surrounding MoE path into this measurement.
    """

    kernel_family = "fp8_gemm_kernel_swapAB" if swapped_ab else "fp8_gemm_kernel"
    encoded_family_length = len(kernel_family)
    return f"_ZN9deep_gemm{encoded_family_length}{kernel_family}ILj{n}ELj{k}E"


def _demangled_kernel_name_prefix(n: int, k: int, *, swapped_ab: bool) -> str:
    kernel_family = "fp8_gemm_kernel_swapAB" if swapped_ab else "fp8_gemm_kernel"
    return f"deep_gemm::{kernel_family}<(unsigned int){n}, (unsigned int){k}"


def _validate_recipe_names(
    kernel_names: list[str],
    *,
    n: int,
    k: int,
    swapped_ab: bool,
) -> str:
    mangled_prefix = _kernel_name_filter(n, k, swapped_ab=swapped_ab)
    demangled_prefix = _demangled_kernel_name_prefix(n, k, swapped_ab=swapped_ab)
    unique_names = sorted(set(kernel_names))
    if len(unique_names) != 1:
        raise RuntimeError(
            "direct grouped GEMM must resolve to exactly one recipe, got "
            f"{len(unique_names)}: {unique_names}"
        )
    recipe_name = unique_names[0]
    expected_prefix = mangled_prefix if recipe_name.startswith("_Z") else demangled_prefix
    if expected_prefix not in recipe_name:
        raise RuntimeError(f"captured recipe does not match N={n}, K={k}: {recipe_name}")
    if "GroupedWithOffsetScheduler" not in recipe_name:
        raise RuntimeError(f"captured recipe is not GroupedWithOffset: {recipe_name}")
    return recipe_name


def capture_fp8_blockscale_grouped_gemm_recipe(
    launch: DirectGroupedGemmLaunch,
    *,
    num_device_sms: int,
    num_warmup: int = 1,
    num_iter: int = 1,
    clear_l2: bool = False,
):
    """CUPTI-capture and fail-closed validate the full DeepGEMM recipe name."""

    from profiling.profilers.cupti_kernel_profiler import profile_kernel

    swapped_ab = _uses_swapped_ab(launch.expected_m, num_device_sms)
    kernel_filter = _kernel_name_filter(launch.n, launch.k, swapped_ab=swapped_ab)
    summary = profile_kernel(
        launch.run_once,
        num_warmup=num_warmup,
        num_iter=num_iter,
        clear_l2_before_run=clear_l2,
        clear_l2_between_launches=clear_l2,
        kernel_name_contains=kernel_filter,
    )
    recipe_name = _validate_recipe_names(
        summary.matched_kernel_names,
        n=launch.n,
        k=launch.k,
        swapped_ab=swapped_ab,
    )
    return summary, recipe_name


def _logical_bytes(
    n: int,
    k: int,
    batches: tuple[int, ...],
) -> int:
    total_local_rows = sum(batches)
    active_experts = sum(batch_size > 0 for batch_size in batches)
    activation_bytes = total_local_rows * k
    activation_scale_bytes = total_local_rows * (k // _BLOCK_SIZE) * 4
    weight_bytes = active_experts * n * k
    weight_scale_bytes = active_experts * (n // _BLOCK_SIZE) * (k // _BLOCK_SIZE) * 4
    output_bytes = total_local_rows * n * 2
    return (
        activation_bytes + activation_scale_bytes + weight_bytes + weight_scale_bytes + output_bytes
    )


def profile_fp8_blockscale_grouped_gemm_flashinfer_trtllm(
    n: int,
    k: int,
    dtype: DType | str,
    num_local_experts: int,
    per_group_batches: tuple[int, ...] | list[int],
    num_input_tokens: int,
    experts_per_token: int,
) -> ComputeMetrics:
    """Profile only the direct gate+up GroupedWithOffset GEMM launch."""

    (
        n,
        k,
        dtype,
        num_local_experts,
        batches,
        num_input_tokens,
        experts_per_token,
    ) = _validate_args(
        n,
        k,
        dtype,
        num_local_experts,
        per_group_batches,
        num_input_tokens,
        experts_per_token,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the direct FP8 block-scale grouped GEMM"
        ) from exc

    num_device_sms = _validate_cuda_device(torch)
    try:
        launch = prepare_fp8_blockscale_grouped_gemm_launch(
            torch,
            n=n,
            k=k,
            dtype=dtype,
            num_local_experts=num_local_experts,
            per_group_batches=batches,
            num_input_tokens=num_input_tokens,
            experts_per_token=experts_per_token,
        )
        # Runtime JIT and one full recipe-name capture stay outside the formal
        # Timer window.  The validation fails closed if a FlashInfer upgrade
        # changes the family, shape order, or scheduler.
        launch.run_once()
        torch.cuda.synchronize()
        capture_fp8_blockscale_grouped_gemm_recipe(
            launch,
            num_device_sms=num_device_sms,
            num_warmup=0,
            num_iter=1,
            clear_l2=False,
        )

        swapped_ab = _uses_swapped_ab(num_input_tokens, num_device_sms)
        kernel_filter = _kernel_name_filter(n, k, swapped_ab=swapped_ab)
        time_ms = Timer.cupti(launch.run_once, kernel_name=kernel_filter)
        # The direct callable launches only the target GEMM, so unlike the full
        # fused-MoE wrapper this NVML interval does not include routing/quant/GEMM2.
        energy_j = Energy.perf(launch.run_once, per_iter_time_ms=time_ms)

        total_local_rows = sum(batches)
        flops = 2 * total_local_rows * n * k
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0 else 0.0
        bytes_accessed = _logical_bytes(n, k, batches)
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
    "DirectGroupedGemmLaunch",
    "capture_fp8_blockscale_grouped_gemm_recipe",
    "prepare_fp8_blockscale_grouped_gemm_launch",
    "profile_fp8_blockscale_grouped_gemm_flashinfer_trtllm",
    "run_fp8_blockscale_grouped_gemm_direct",
]
