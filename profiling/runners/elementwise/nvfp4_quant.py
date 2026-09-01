"""Production SM100 NVFP4 activation-quantization runners."""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_GROUP_SIZE = 16
_SCALE_FORMAT = "linear_e4m3"
# CUPTI reports this stable implementation stem for vLLM's unswizzled
# ``scaled_fp4_quant`` path on SM100.
_KERNEL_NAME = "cvt_fp16_to_fp4_sf_major"
_FLASHINFER_KERNEL_NAME = "nvfp4_quantize"
_E4M3_MAX = 448.0


def _validate_args(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    input_dtype: DType | str,
    scale_format: str,
) -> tuple[int, int]:
    num_tokens = int(num_tokens)
    hidden_size = int(hidden_size)
    input_dtype = DType.from_value(input_dtype)

    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be > 0, got {num_tokens}")
    if hidden_size <= 0 or hidden_size % _GROUP_SIZE != 0:
        raise ValueError(
            f"hidden_size must be > 0 and divisible by {_GROUP_SIZE}, got {hidden_size}"
        )
    if group_size != _GROUP_SIZE:
        raise ValueError(f"NVFP4 requires group_size={_GROUP_SIZE}, got {group_size}")
    if input_dtype is not DType.BF16:
        raise ValueError(f"NVFP4 quant requires input_dtype=bf16, got {input_dtype.value}")
    if scale_format != _SCALE_FORMAT:
        raise ValueError(f"unsupported NVFP4 scale format: {scale_format}")
    return num_tokens, hidden_size


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("NVFP4 profiling requires CUDA")
    device = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability != (10, 0):
        gpu_name = str(torch.cuda.get_device_name(device))
        raise ProfilerNotImplemented(
            f"NVFP4 profiling requires SM100, got {gpu_name} with SM{capability[0]}{capability[1]}"
        )


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

    _validate_cuda_device(torch)
    source = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones((), dtype=torch.float32, device="cuda")

    def run_once() -> None:
        try:
            # Match the modular TRTLLM MoE stage: it consumes row-major E4M3
            # scales, unlike the swizzled scale layout used by NVFP4 linears.
            ops.scaled_fp4_quant(source, global_scale, is_sf_swizzled_layout=False)
        except RuntimeError as exc:
            raise KernelLaunchFailed(f"NVFP4 activation quantization failed: {exc}") from exc

    return _measure(run_once, num_tokens, hidden_size, _KERNEL_NAME)


def profile_nvfp4_quant_flashinfer_cutedsl(
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
        from flashinfer import SfLayout, nvfp4_quantize
    except ImportError as exc:
        raise ProfilerNotImplemented("the SGLang environment is required") from exc

    _validate_cuda_device(torch)
    source = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    global_scale = torch.full((1,), 1.0 / (_E4M3_MAX * 6.0), dtype=torch.float32, device="cuda")

    def run_once() -> None:
        try:
            nvfp4_quantize(
                source,
                global_scale,
                sfLayout=SfLayout.layout_linear,
                per_token_activation=True,
                backend="cute-dsl",
            )
        except RuntimeError as exc:
            raise KernelLaunchFailed(f"NVFP4 activation quantization failed: {exc}") from exc

    # FlashInfer lazily builds this CuTe-DSL kernel on its first invocation.
    run_once()
    torch.cuda.synchronize()
    return _measure(run_once, num_tokens, hidden_size, _FLASHINFER_KERNEL_NAME)


def _measure(
    run_once: Any,
    num_tokens: int,
    hidden_size: int,
    kernel_name: str,
) -> ComputeMetrics:
    time_ms = Timer.cupti(run_once, kernel_name=kernel_name)
    energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    logical_bytes = num_tokens * hidden_size * 2
    logical_bytes += num_tokens * hidden_size // 2
    logical_bytes += num_tokens * hidden_size // _GROUP_SIZE
    return ComputeMetrics(
        time_ms=time_ms,
        tflops=0.0,
        memory_bandwidth_gbps=logical_bytes / (time_ms / 1000.0) / 1e9,
        energy_j=energy_j,
    )
