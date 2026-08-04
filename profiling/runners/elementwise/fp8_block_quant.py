"""FlashInfer/TensorRT-LLM BF16-to-FP8 1x128 quantization runner.

The production MoE path uses the grouped overload of ``scale_1x128_kernel``.
FlashInfer does not expose that launch independently from its grouped GEMM, so
this runner JIT-builds a repo-owned TVM-FFI binding that includes the vendored
TensorRT-LLM header and launches only the exact grouped kernel template.  The
kernel implementation itself is not copied into this repository.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BLOCK_SIZE = 128
_INPUT_DTYPE = DType.BF16
_CUDA_MINIMUM = (12, 8)
_HOPPER_COMPUTE_CAPABILITY = (9, 0)
_KERNEL_NAME = "scale_1x128_kernel"
_JIT_MODULE_NAME = "vibesim_fp8_block_quant_grouped_sm90"


def _validate_args(
    num_tokens: int,
    hidden_size: int,
    num_problems: int,
    input_dtype: DType | str,
) -> tuple[int, int, int, DType]:
    num_tokens = int(num_tokens)
    hidden_size = int(hidden_size)
    num_problems = int(num_problems)
    input_dtype = DType.from_value(input_dtype)

    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be > 0, got {num_tokens}")
    if hidden_size <= 0 or hidden_size % _BLOCK_SIZE != 0:
        raise ValueError(
            f"hidden_size must be > 0 and divisible by {_BLOCK_SIZE}, got {hidden_size}"
        )
    if num_problems <= 0:
        raise ValueError(f"num_problems must be > 0, got {num_problems}")
    if input_dtype is not _INPUT_DTYPE:
        raise ValueError(
            f"flashinfer_trtllm fp8_block_quant requires input_dtype=bf16, got {input_dtype.value}"
        )
    return num_tokens, hidden_size, num_problems, input_dtype


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
            "CUDA is required for the flashinfer_trtllm fp8_block_quant backend"
        )

    cuda_version = _parse_cuda_version(getattr(torch.version, "cuda", None))
    if cuda_version is None or cuda_version < _CUDA_MINIMUM:
        rendered_version = getattr(torch.version, "cuda", None)
        raise ProfilerNotImplemented(
            f"flashinfer_trtllm fp8_block_quant requires CUDA >= 12.8, got {rendered_version}"
        )

    device = torch.cuda.current_device()
    compute_capability = tuple(torch.cuda.get_device_capability(device))
    if compute_capability != _HOPPER_COMPUTE_CAPABILITY:
        gpu_name = str(torch.cuda.get_device_name(device))
        raise ProfilerNotImplemented(
            "flashinfer_trtllm fp8_block_quant requires SM90/SM90a Hopper, "
            f"got {gpu_name} with SM{compute_capability[0]}{compute_capability[1]}"
        )


def _compute_grouped_padded_offset(offset: int, problem_index: int) -> int:
    """Mirror ``deep_gemm::compute_padded_offset`` for buffer sizing."""
    alignment = 32
    return (int(offset) + int(problem_index) * (alignment - 1)) // alignment * alignment


def _grouped_scale_shape(
    num_tokens: int,
    hidden_size: int,
    num_problems: int,
) -> tuple[int, int]:
    if num_problems <= 0:
        raise ValueError(f"num_problems must be > 0, got {num_problems}")
    padded_tokens = _compute_grouped_padded_offset(num_tokens, num_problems)
    return hidden_size // _BLOCK_SIZE, padded_tokens


def _uniform_problem_boundaries(num_tokens: int, num_problems: int) -> list[int]:
    """Split final routed rows into contiguous synthetic expert segments.

    Routing and EP ownership are already resolved above L1.  These boundaries
    exist only because the exact production kernel traverses expert segments;
    their final boundary is always the public ``num_tokens`` shape.
    """
    if num_problems <= 0:
        raise ValueError(f"num_problems must be > 0, got {num_problems}")
    tokens_per_problem, problems_with_extra_token = divmod(num_tokens, num_problems)
    boundaries = [0]
    running_tokens = 0
    for problem_index in range(num_problems):
        running_tokens += tokens_per_problem + (
            1 if problem_index < problems_with_extra_token else 0
        )
        boundaries.append(running_tokens)
    return boundaries


def _grouped_launch_policy(
    num_tokens: int,
    hidden_size: int,
    num_problems: int,
    num_device_sms: int,
) -> tuple[int, bool]:
    """Mirror TensorRT-LLM's grid size and grouped-boundary search selector."""
    num_threads = 256
    scales_dim_x = hidden_size // _BLOCK_SIZE
    scales_per_block = num_threads // 32
    num_blocks = min(
        num_device_sms,
        (num_tokens * scales_dim_x + scales_per_block - 1) // scales_per_block,
    )
    if num_problems == 1:
        return num_blocks, True
    work_per_warp_round = (num_tokens * scales_dim_x) / (num_threads * num_blocks / 32)
    binary_search_threshold = num_problems / math.log2(num_problems)
    return num_blocks, work_per_warp_round <= binary_search_threshold


def _logical_bytes(num_tokens: int, hidden_size: int, input_dtype: DType) -> float:
    input_bytes = num_tokens * hidden_size * input_dtype.size_bytes()
    output_bytes = num_tokens * hidden_size * DType.FP8_E4M3.size_bytes()
    scale_bytes = num_tokens * (hidden_size // _BLOCK_SIZE) * DType.FP32.size_bytes()
    return input_bytes + output_bytes + scale_bytes


@functools.cache
def _load_flashinfer_grouped_quantizer():
    """Build and load the exact grouped launch against this FlashInfer wheel."""
    try:
        from flashinfer.jit import env as jit_env
        from flashinfer.jit.core import gen_jit_spec, sm90a_nvcc_flags
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "FlashInfer JIT support is required for the grouped "
            "flashinfer_trtllm fp8_block_quant backend"
        ) from exc

    binding_source = Path(__file__).resolve().parent / "csrc" / "fp8_block_quant_grouped.cu"
    if not binding_source.is_file():
        raise ProfilerNotImplemented(
            f"grouped FP8 quantization JIT binding is missing: {binding_source}"
        )

    nvcc_flags = sm90a_nvcc_flags + [
        "-DCOMPILE_HOPPER_TMA_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP8_BLOCK_SCALE",
        "-DCUTLASS_ENABLE_GDC_FOR_SM90=1",
    ]
    include_paths = [
        jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
        jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include",
        jit_env.FLASHINFER_CSRC_DIR
        / "nv_internal"
        / "tensorrt_llm"
        / "cutlass_extensions"
        / "include",
        jit_env.FLASHINFER_CSRC_DIR
        / "nv_internal"
        / "tensorrt_llm"
        / "kernels"
        / "cutlass_kernels"
        / "include",
        jit_env.FLASHINFER_CSRC_DIR
        / "nv_internal"
        / "tensorrt_llm"
        / "kernels"
        / "cutlass_kernels",
    ]
    common_source_root = jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "cpp" / "common"
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


def _prepare_grouped_quant_launch(
    torch: Any,
    num_tokens: int,
    hidden_size: int,
    num_problems: int,
) -> tuple[Any, Any, Any, Any, Any]:
    """Allocate one grouped-quant launch and return its callable plus tensors."""
    input_tensor = torch.randn(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    quantized_tensor = torch.empty(
        (num_tokens, hidden_size),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    scale_tensor = torch.empty(
        _grouped_scale_shape(num_tokens, hidden_size, num_problems),
        dtype=torch.float32,
        device="cuda",
    )
    problem_m_offsets = torch.tensor(
        _uniform_problem_boundaries(num_tokens, num_problems),
        dtype=torch.int64,
        device="cuda",
    )
    quantizer = _load_flashinfer_grouped_quantizer()

    def run_once() -> None:
        quantizer.run_grouped_fp8_block_quant(
            input_tensor,
            quantized_tensor,
            scale_tensor,
            problem_m_offsets,
            num_problems,
        )

    return run_once, input_tensor, quantized_tensor, scale_tensor, problem_m_offsets


def profile_fp8_block_quant_flashinfer_trtllm(
    num_tokens: int,
    hidden_size: int,
    num_problems: int,
    input_dtype: DType | str,
) -> ComputeMetrics:
    num_tokens, hidden_size, num_problems, input_dtype = _validate_args(
        num_tokens,
        hidden_size,
        num_problems,
        input_dtype,
    )

    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the flashinfer_trtllm fp8_block_quant backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        run_once, _, _, _, _ = _prepare_grouped_quant_launch(
            torch,
            num_tokens,
            hidden_size,
            num_problems,
        )

        time_ms = Timer.cupti(run_once, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
        bytes_accessed = _logical_bytes(num_tokens, hidden_size, input_dtype)
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except (RuntimeError, ValueError) as exc:
        raise KernelLaunchFailed(str(exc)) from exc
