"""Exact vLLM dense BF16-to-FP8 per-token-group quantization runner."""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_GROUP_SIZE = 128
_INPUT_DTYPE = DType.BF16
_SCALE_FORMAT = "ue8m0_column_major"
_SUPPORTED_GPUS = frozenset({"NVIDIA H100", "NVIDIA H200"})
_HOPPER_COMPUTE_CAPABILITY = (9, 0)
_KERNEL_NAME = "per_token_group_quant_8bit_kernel"
_FP8_E4M3_MIN = -448.0
_FP8_E4M3_MAX = 448.0
_EPSILON = 1e-10


def _validate_args(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    input_dtype: DType | str,
    scale_format: str,
) -> tuple[int, int, int, DType, str]:
    num_tokens = int(num_tokens)
    hidden_size = int(hidden_size)
    group_size = int(group_size)
    input_dtype = DType.from_value(input_dtype)
    scale_format = str(scale_format)

    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be > 0, got {num_tokens}")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
    if group_size <= 0 or hidden_size % group_size != 0:
        raise ValueError(
            "group_size must be > 0 and divide hidden_size, got "
            f"hidden_size={hidden_size}, group_size={group_size}"
        )
    if group_size != _GROUP_SIZE:
        raise ValueError(
            f"vllm_cuda fp8_per_token_group_quant requires group_size={_GROUP_SIZE}, "
            f"got {group_size}"
        )
    if input_dtype is not _INPUT_DTYPE:
        raise ValueError(
            "vllm_cuda fp8_per_token_group_quant requires input_dtype=bf16, "
            f"got {input_dtype.value}"
        )
    if scale_format != _SCALE_FORMAT:
        raise ValueError(
            "vllm_cuda fp8_per_token_group_quant requires "
            f"scale_format={_SCALE_FORMAT!r}, got {scale_format!r}"
        )
    return num_tokens, hidden_size, group_size, input_dtype, scale_format


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for fp8_per_token_group_quant vllm_cuda"
        )
    device = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability != _HOPPER_COMPUTE_CAPABILITY:
        raise ProfilerNotImplemented(
            "fp8_per_token_group_quant vllm_cuda requires Hopper compute "
            f"capability {_HOPPER_COMPUTE_CAPABILITY}, got {capability}"
        )
    gpu_name = str(torch.cuda.get_device_name(device))
    if gpu_name not in _SUPPORTED_GPUS:
        raise ProfilerNotImplemented(
            "fp8_per_token_group_quant vllm_cuda is verified only on "
            f"{sorted(_SUPPORTED_GPUS)}, got {gpu_name}"
        )


def _load_vllm_quant_op(torch: Any) -> Any:
    """Load vLLM's public stable-ABI op without importing model runtime state."""
    try:
        # The extension owns the torch.library registration.  Plain ``import
        # vllm`` does not guarantee this op is registered when CUDA platform
        # auto-detection is unavailable during module import.
        import vllm._C_stable_libtorch  # noqa: F401
    except (ImportError, OSError) as exc:
        raise ProfilerNotImplemented(
            "vLLM stable CUDA extension is required for "
            "fp8_per_token_group_quant vllm_cuda"
        ) from exc

    if not hasattr(torch.ops._C, "per_token_group_fp8_quant"):
        raise ProfilerNotImplemented(
            "vLLM stable CUDA extension does not expose "
            "torch.ops._C.per_token_group_fp8_quant"
        )
    return torch.ops._C.per_token_group_fp8_quant


def _allocate_operands(
    torch: Any,
    *,
    num_tokens: int,
    hidden_size: int,
    group_size: int,
) -> tuple[Any, Any, Any]:
    input_tensor = torch.randn(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    output_quantized = torch.empty(
        (num_tokens, hidden_size),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    # Match vLLM fp8_utils.per_token_group_quant_fp8 exactly: allocate the
    # transposed physical tensor, then expose [M, K/group] with stride (1, M).
    output_scales = torch.empty(
        (hidden_size // group_size, num_tokens),
        dtype=torch.float32,
        device="cuda",
    ).permute(-1, -2)
    return input_tensor, output_quantized, output_scales


def _launch_quant(
    quant_op: Any,
    input_tensor: Any,
    output_quantized: Any,
    output_scales: Any,
    group_size: int,
) -> None:
    quant_op(
        input_tensor,
        output_quantized,
        output_scales,
        group_size,
        _EPSILON,
        _FP8_E4M3_MIN,
        _FP8_E4M3_MAX,
        True,  # SCALE_UE8M0
        True,  # column-major scale layout
        False,  # not the separate TMA-packed scale layout
    )


def _logical_bytes(
    *,
    num_tokens: int,
    hidden_size: int,
    group_size: int,
) -> int:
    input_bytes = num_tokens * hidden_size * _INPUT_DTYPE.size_bytes()
    output_bytes = num_tokens * hidden_size  # FP8 E4M3
    scale_bytes = num_tokens * (hidden_size // group_size) * 4
    return input_bytes + output_bytes + scale_bytes


def profile_fp8_per_token_group_quant_vllm_cuda(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    input_dtype: DType | str,
    scale_format: str,
) -> ComputeMetrics:
    """Profile the exact vLLM QKV/O-projection input quantization kernel."""
    (
        num_tokens,
        hidden_size,
        group_size,
        _input_dtype,
        _scale_format,
    ) = _validate_args(
        num_tokens,
        hidden_size,
        group_size,
        input_dtype,
        scale_format,
    )

    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("PyTorch is required for vllm_cuda") from exc

    _validate_cuda_device(torch)
    quant_op = _load_vllm_quant_op(torch)
    input_tensor, output_quantized, output_scales = _allocate_operands(
        torch,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        group_size=group_size,
    )

    def kernel() -> None:
        try:
            _launch_quant(
                quant_op,
                input_tensor,
                output_quantized,
                output_scales,
                group_size,
            )
        except RuntimeError as exc:
            raise KernelLaunchFailed(
                f"vLLM per-token-group FP8 quantization failed: {exc}"
            ) from exc

    time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
    energy_j = Energy.perf(kernel, warmup=10, per_iter_time_ms=time_ms)
    logical_bytes = _logical_bytes(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        group_size=group_size,
    )
    memory_bandwidth_gbps = (logical_bytes / (time_ms / 1000.0)) / 1e9
    return ComputeMetrics(
        time_ms=time_ms,
        tflops=0.0,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        energy_j=energy_j,
    )
