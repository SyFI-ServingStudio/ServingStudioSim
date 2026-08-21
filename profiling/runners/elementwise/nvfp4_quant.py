"""Runner for vLLM's SM100 NVFP4 activation quantization kernel."""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

GROUP_SIZE = 16
SCALE_FORMAT = "linear_e4m3"
# The demangled SM100 symbol is a templated
# `tensorrt_llm::kernels::quantize_with_block_size<...>`. Nsight may render the
# same operation with its shorter implementation label, so match the stable
# symbol stem emitted by CUPTI rather than the UI alias.
KERNEL_NAME = "cvt_fp16_to_fp4_sf_major"


def _validate_args(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    input_dtype: DType | str,
    scale_format: str,
) -> tuple[int, int]:
    dtype = DType.from_value(input_dtype)
    if num_tokens <= 0 or hidden_size <= 0:
        raise ValueError("num_tokens and hidden_size must be positive")
    if hidden_size % GROUP_SIZE != 0 or group_size != GROUP_SIZE:
        raise ValueError("NVFP4 requires hidden_size divisible by group_size=16")
    if dtype is not DType.BF16:
        raise ValueError(f"NVFP4 quant requires bf16 input, got {dtype.value}")
    if scale_format != SCALE_FORMAT:
        raise ValueError(f"unsupported NVFP4 scale format: {scale_format}")
    return int(num_tokens), int(hidden_size)


def _require_b200(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("NVFP4 profiling requires CUDA")
    if tuple(torch.cuda.get_device_capability()) != (10, 0):
        raise ProfilerNotImplemented("NVFP4 profiling requires SM100")


def profile_nvfp4_quant_vllm_cuda(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    input_dtype: DType | str,
    scale_format: str,
) -> ComputeMetrics:
    num_tokens, hidden_size = _validate_args(
        num_tokens, hidden_size, group_size, input_dtype, scale_format
    )
    try:
        import torch
        from vllm import _custom_ops as ops
    except ImportError as exc:
        raise ProfilerNotImplemented("the instrumented vLLM environment is required") from exc
    _require_b200(torch)
    source = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones((), dtype=torch.float32, device="cuda")

    def kernel() -> None:
        try:
            ops.scaled_fp4_quant(
                source, global_scale, is_sf_swizzled_layout=False
            )
        except RuntimeError as exc:
            raise KernelLaunchFailed(f"NVFP4 activation quantization failed: {exc}") from exc

    time_ms = Timer.cupti(kernel, kernel_name=KERNEL_NAME)
    energy_j = Energy.perf(kernel, per_iter_time_ms=time_ms)
    logical_bytes = num_tokens * hidden_size * 2
    logical_bytes += num_tokens * hidden_size // 2
    logical_bytes += num_tokens * hidden_size // GROUP_SIZE
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=time_ms,
        tflops=0.0,
        memory_bandwidth_gbps=logical_bytes / elapsed_s / 1e9,
        energy_j=energy_j,
    )
