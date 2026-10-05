"""Exact vLLM dense BF16-to-FP8 per-token-group quantization runner.

``scale_format`` selects the production scale writer:

- ``ue8m0_column_major``: Hopper DeepGEMM dense inputs; FP32 scales, column-major.
- ``ue8m0_row_major``: Blackwell TRT-LLM FP8 block-scale MoE input; FP32 scales,
  row-major (``per_token_group_quant_fp8`` default layout).
- ``ue8m0_packed_int32``: Blackwell DeepGEMM dense inputs
  (``per_token_group_quant_fp8_packed_for_deepgemm``); four UE8M0 bytes per
  int32, MN-major with a TMA-aligned stride.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

# The packed register kernel is compiled for one group width only
# (per_token_group_quant.cu: `STD_TORCH_CHECK(group_size == 128, ...)` and
# `static_assert(GROUP_SIZE == 128)`); the generic kernel takes any group_size.
_PACKED_GROUP_SIZE = 128
_INPUT_DTYPE = DType.BF16
_COLUMN_MAJOR = "ue8m0_column_major"
_ROW_MAJOR = "ue8m0_row_major"
_PACKED_INT32 = "ue8m0_packed_int32"
# scale_format -> CUPTI kernel name. Both kernels are generic CUDA built for
# every arch vLLM ships (their sm90+ code is only optional PDL), so the GPU a
# format is deployed on in production is not a launch constraint.
_SCALE_FORMATS = {
    _COLUMN_MAJOR: "per_token_group_quant_8bit_kernel",
    _ROW_MAJOR: "per_token_group_quant_8bit_kernel",
    _PACKED_INT32: "per_token_group_quant_8bit_packed_register_kernel",
}
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
    if input_dtype is not _INPUT_DTYPE:
        raise ValueError(
            "vllm_cuda fp8_per_token_group_quant requires input_dtype=bf16, "
            f"got {input_dtype.value}"
        )
    if scale_format not in _SCALE_FORMATS:
        raise ValueError(
            "vllm_cuda fp8_per_token_group_quant requires scale_format in "
            f"{sorted(_SCALE_FORMATS)}, got {scale_format!r}"
        )
    if scale_format == _PACKED_INT32 and group_size != _PACKED_GROUP_SIZE:
        raise ValueError(
            f"vllm_cuda fp8_per_token_group_quant {_PACKED_INT32} requires "
            f"group_size={_PACKED_GROUP_SIZE}, got {group_size}"
        )
    return num_tokens, hidden_size, group_size, input_dtype, scale_format


def _load_vllm_quant_op(torch: Any, scale_format: str = _COLUMN_MAJOR) -> Any:
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

    op_name = "per_token_group_fp8_quant"
    if scale_format == _PACKED_INT32:
        op_name += "_packed"
    if not hasattr(torch.ops._C, op_name):
        raise ProfilerNotImplemented(
            f"vLLM stable CUDA extension does not expose torch.ops._C.{op_name}"
        )
    return getattr(torch.ops._C, op_name)


def _allocate_operands(
    torch: Any,
    *,
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    scale_format: str = _COLUMN_MAJOR,
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
    num_groups = hidden_size // group_size
    if scale_format == _ROW_MAJOR:
        output_scales = torch.empty(
            (num_tokens, num_groups), dtype=torch.float32, device="cuda"
        )
    elif scale_format == _PACKED_INT32:
        # [M, ceil(G/4)] int32 with stride (1, round_up(M, 4)), as DeepGEMM expects.
        output_scales = torch.empty_strided(
            (num_tokens, (num_groups + 3) // 4),
            (1, (num_tokens + 3) // 4 * 4),
            dtype=torch.int32,
            device="cuda",
        )
    else:
        # Match vLLM fp8_utils.per_token_group_quant_fp8 exactly: allocate the
        # transposed physical tensor, then expose [M, K/group] with stride (1, M).
        output_scales = torch.empty(
            (num_groups, num_tokens),
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
    scale_format: str = _COLUMN_MAJOR,
) -> None:
    arguments = (
        input_tensor,
        output_quantized,
        output_scales,
        group_size,
        _EPSILON,
        _FP8_E4M3_MIN,
        _FP8_E4M3_MAX,
    )
    if scale_format == _PACKED_INT32:
        quant_op(*arguments)  # the packed writer takes no layout flags
        return
    quant_op(
        *arguments,
        True,  # SCALE_UE8M0
        scale_format == _COLUMN_MAJOR,  # column-major scale layout
        False,  # not the separate TMA-packed scale layout
    )


def _logical_bytes(
    *,
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    scale_format: str = _COLUMN_MAJOR,
) -> int:
    input_bytes = num_tokens * hidden_size * _INPUT_DTYPE.size_bytes()
    output_bytes = num_tokens * hidden_size  # FP8 E4M3
    num_groups = hidden_size // group_size
    if scale_format == _PACKED_INT32:
        num_groups = (num_groups + 3) // 4  # four one-byte scales per int32
    scale_bytes = num_tokens * num_groups * 4
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
        scale_format,
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

    quant_op = _load_vllm_quant_op(torch, scale_format)
    input_tensor, output_quantized, output_scales = _allocate_operands(
        torch,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        group_size=group_size,
        scale_format=scale_format,
    )

    def kernel() -> None:
        try:
            _launch_quant(
                quant_op,
                input_tensor,
                output_quantized,
                output_scales,
                group_size,
                scale_format,
            )
        except RuntimeError as exc:
            raise KernelLaunchFailed(
                f"vLLM per-token-group FP8 quantization failed: {exc}"
            ) from exc

    time_ms = Timer.cupti(kernel, kernel_name=_SCALE_FORMATS[scale_format])
    energy_j = Energy.perf(kernel, warmup=10, per_iter_time_ms=time_ms)
    logical_bytes = _logical_bytes(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        group_size=group_size,
        scale_format=scale_format,
    )
    memory_bandwidth_gbps = (logical_bytes / (time_ms / 1000.0)) / 1e9
    return ComputeMetrics(
        time_ms=time_ms,
        tflops=0.0,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        energy_j=energy_j,
    )
