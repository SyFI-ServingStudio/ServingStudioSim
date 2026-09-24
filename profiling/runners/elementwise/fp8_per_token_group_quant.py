"""Exact vLLM dense BF16-to-FP8 per-token-group quantization runner.

Three scale formats select distinct production writers of one vLLM op family:

- ``ue8m0_column_major``: Hopper DeepGEMM dense inputs. FP32 UE8M0-rounded
  scales in column-major layout (``per_token_group_fp8_quant``).
- ``ue8m0_row_major``: Blackwell TRT-LLM FP8 block-scale MoE input
  (``fused_moe/utils.py::_fp8_quantize`` -> ``per_token_group_quant_fp8``
  with default layout). FP32 UE8M0-rounded scales, row-major.
- ``ue8m0_packed_int32``: Blackwell DeepGEMM dense inputs
  (``fp8_utils.py::per_token_group_quant_fp8_packed_for_deepgemm``). Four UE8M0
  exponent bytes packed per int32, MN-major with a TMA-aligned stride.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_GROUP_SIZE = 128
_INPUT_DTYPE = DType.BF16
_COLUMN_MAJOR = "ue8m0_column_major"
_ROW_MAJOR = "ue8m0_row_major"
_PACKED_INT32 = "ue8m0_packed_int32"
_HOPPER_GPUS = frozenset({"NVIDIA H100", "NVIDIA H200"})
_BLACKWELL_GPUS = frozenset({"NVIDIA B200"})
# scale_format -> (compute capability, verified GPUs, CUPTI kernel name).
_SCALE_FORMATS = {
    _COLUMN_MAJOR: ((9, 0), _HOPPER_GPUS, "per_token_group_quant_8bit_kernel"),
    _ROW_MAJOR: ((10, 0), _BLACKWELL_GPUS, "per_token_group_quant_8bit_kernel"),
    _PACKED_INT32: (
        (10, 0),
        _BLACKWELL_GPUS,
        "per_token_group_quant_8bit_packed_register_kernel",
    ),
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
    if scale_format not in _SCALE_FORMATS:
        raise ValueError(
            "vllm_cuda fp8_per_token_group_quant requires scale_format in "
            f"{sorted(_SCALE_FORMATS)}, got {scale_format!r}"
        )
    return num_tokens, hidden_size, group_size, input_dtype, scale_format


def _validate_cuda_device(torch: Any, scale_format: str = _COLUMN_MAJOR) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for fp8_per_token_group_quant vllm_cuda"
        )
    required_capability, supported_gpus, _kernel_name = _SCALE_FORMATS[scale_format]
    device = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability != required_capability:
        raise ProfilerNotImplemented(
            f"fp8_per_token_group_quant vllm_cuda {scale_format} requires compute "
            f"capability {required_capability}, got {capability}"
        )
    gpu_name = str(torch.cuda.get_device_name(device))
    if gpu_name not in supported_gpus:
        raise ProfilerNotImplemented(
            f"fp8_per_token_group_quant vllm_cuda {scale_format} is verified only on "
            f"{sorted(supported_gpus)}, got {gpu_name}"
        )


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

    op_name = (
        "per_token_group_fp8_quant_packed"
        if scale_format == _PACKED_INT32
        else "per_token_group_fp8_quant"
    )
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
        # per_token_group_quant_fp8 default layout: contiguous [M, K/group].
        output_scales = torch.empty(
            (num_tokens, num_groups),
            dtype=torch.float32,
            device="cuda",
        )
    elif scale_format == _PACKED_INT32:
        # per_token_group_quant_fp8_packed_for_deepgemm: [M, ceil(G/4)] int32
        # with stride (1, tma_aligned_M), tma_aligned_M = round_up(M, 4).
        tma_aligned_tokens = ((num_tokens + 3) // 4) * 4
        output_scales = torch.empty_strided(
            (num_tokens, (num_groups + 3) // 4),
            (1, tma_aligned_tokens),
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
    if scale_format == _PACKED_INT32:
        quant_op(
            input_tensor,
            output_quantized,
            output_scales,
            group_size,
            _EPSILON,
            _FP8_E4M3_MIN,
            _FP8_E4M3_MAX,
        )
        return
    quant_op(
        input_tensor,
        output_quantized,
        output_scales,
        group_size,
        _EPSILON,
        _FP8_E4M3_MIN,
        _FP8_E4M3_MAX,
        True,  # SCALE_UE8M0
        scale_format == _COLUMN_MAJOR,  # column-major scale layout
        False,  # not the separate TMA-packed scale layout
    )


def _check_against_reference(
    torch: Any,
    input_tensor: Any,
    output_quantized: Any,
    output_scales: Any,
    group_size: int,
    scale_format: str,
) -> None:
    """Exact UE8M0 reference for the scales and the FP8 payload (untimed)."""
    num_tokens, hidden_size = input_tensor.shape
    num_groups = hidden_size // group_size
    grouped = input_tensor.float().view(num_tokens, num_groups, group_size)
    absmax = grouped.abs().amax(dim=-1).clamp_min(_EPSILON)
    reference_scales = torch.exp2(torch.ceil(torch.log2(absmax / _FP8_E4M3_MAX)))
    reference_quantized = (
        (grouped / reference_scales.unsqueeze(-1))
        .clamp(_FP8_E4M3_MIN, _FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .view(num_tokens, hidden_size)
    )
    if scale_format == _PACKED_INT32:
        # Byte j of packed word w holds the biased exponent of group 4w + j.
        # Copy into a dense row-major buffer first: a single packed word per row
        # is "contiguous" to torch yet keeps the TMA stride, which view() rejects.
        dense = torch.empty(output_scales.shape, dtype=torch.int32, device=output_scales.device)
        dense.copy_(output_scales)
        exponent_bytes = dense.view(torch.uint8).view(num_tokens, -1)[:, :num_groups]
        actual_scales = torch.exp2(exponent_bytes.float() - 127.0)
    else:
        actual_scales = output_scales
    torch.testing.assert_close(actual_scales, reference_scales, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        output_quantized.float(), reference_quantized.float(), rtol=0.0, atol=0.0
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
        scale_bytes = num_tokens * ((num_groups + 3) // 4) * 4
    else:
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

    _validate_cuda_device(torch, scale_format)
    quant_op = _load_vllm_quant_op(torch, scale_format)
    input_tensor, output_quantized, output_scales = _allocate_operands(
        torch,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        group_size=group_size,
        scale_format=scale_format,
    )
    if scale_format != _COLUMN_MAJOR:
        _launch_quant(
            quant_op, input_tensor, output_quantized, output_scales, group_size, scale_format
        )
        torch.cuda.synchronize()
        _check_against_reference(
            torch, input_tensor, output_quantized, output_scales, group_size, scale_format
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

    time_ms = Timer.cupti(kernel, kernel_name=_SCALE_FORMATS[scale_format][2])
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


_FORK_SCALE_FORMATS = frozenset({_ROW_MAJOR, _PACKED_INT32})


def profile_fp8_per_token_group_quant_vllm_fork_cuda(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    input_dtype: DType | str,
    scale_format: str,
) -> ComputeMetrics:
    """Profile the same ``_C`` quant ops as built into the GLM-5.3 serving stack.

    The fork's ``per_token_group_quant.cu`` changes the launch grid and adds
    PDL, which moves these kernels by about 10% on B200, so the serving stack's
    binary is profiled directly. Only the Blackwell layouts GLM-5.3 uses are
    accepted.
    """
    if scale_format not in _FORK_SCALE_FORMATS:
        raise ProfilerNotImplemented(
            f"vllm_fork_cuda requires scale_format in {sorted(_FORK_SCALE_FORMATS)}"
        )
    return profile_fp8_per_token_group_quant_vllm_cuda(
        num_tokens, hidden_size, group_size, input_dtype, scale_format
    )
