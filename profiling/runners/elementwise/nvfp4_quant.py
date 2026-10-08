"""Production SM100 NVFP4 activation-quantization runners."""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_GROUP_SIZE = 16
# Row-major E4M3 group scales, read by the TensorRT-LLM NVFP4 MoE GEMM.
_LINEAR_SCALE_FORMAT = "linear_e4m3"
# E4M3 group scales in the 128x4 tcgen05 tile layout, read by the dense NVFP4
# linear GEMM.
_SWIZZLED_SCALE_FORMAT = "swizzled_e4m3"
_VLLM_SCALE_FORMATS = (_LINEAR_SCALE_FORMAT, _SWIZZLED_SCALE_FORMAT)
# CUPTI reports mangled names for vLLM's ``scaled_fp4_quant`` kernels on SM100
# (nvfp4_quant_kernels.cu), e.g.
# _ZN4vllm15cvt_fp16_to_fp4I13__nv_bfloat16Lb0ELb0EEEviiiiPKT_PKfPjS7_. The
# swizzled kernel's bare stem prefixes the linear one, so it is matched with
# its length prefix and template marker.
_KERNEL_NAMES = {
    _LINEAR_SCALE_FORMAT: "cvt_fp16_to_fp4_sf_major",
    _SWIZZLED_SCALE_FORMAT: "15cvt_fp16_to_fp4I",
}
_FLASHINFER_KERNEL_NAME = "nvfp4_quantize"
_E4M3_MAX = 448.0
_FP4_MAX = 6.0
# The swizzled scale tensor is padded to whole 128-row x 4-group tiles
# (vllm._custom_ops.create_fp4_scale_tensor).
_SWIZZLE_ROW_TILE = 128
_SWIZZLE_GROUP_TILE = 4


def _validate_args(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    input_dtype: DType | str,
    scale_format: str,
    supported_scale_formats: tuple[str, ...] = (_LINEAR_SCALE_FORMAT,),
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
    if scale_format not in supported_scale_formats:
        raise ValueError(
            f"unsupported NVFP4 scale format: {scale_format!r}; "
            f"expected one of {supported_scale_formats}"
        )
    return num_tokens, hidden_size


def profile_nvfp4_quant_vllm_cuda(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    input_dtype: DType | str,
    scale_format: str,
) -> ComputeMetrics:
    num_tokens, hidden_size = _validate_args(
        num_tokens, hidden_size, group_size, input_dtype, scale_format, _VLLM_SCALE_FORMATS
    )
    try:
        import torch
        from vllm import _custom_ops as ops
    except ImportError as exc:
        raise ProfilerNotImplemented("the instrumented vLLM environment is required") from exc

    source = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")

    if scale_format == _LINEAR_SCALE_FORMAT:
        global_scale = torch.ones((), dtype=torch.float32, device="cuda")

        def run_once() -> None:
            try:
                # Match the modular TRTLLM MoE stage: it consumes row-major
                # E4M3 scales.
                ops.scaled_fp4_quant(source, global_scale, is_sf_swizzled_layout=False)
            except RuntimeError as exc:
                raise KernelLaunchFailed(f"NVFP4 activation quantization failed: {exc}") from exc

    else:
        global_scale = _calibrated_global_scale(source)

        def run_once() -> None:
            try:
                # The exact call of the B200 default NVFP4 linear kernel. Its
                # backend string has no "trtllm", so every m keeps the 128x4
                # scale layout rather than the small-batch 8x4 one.
                ops.scaled_fp4_quant(
                    source,
                    global_scale,
                    is_sf_swizzled_layout=True,
                    backend="flashinfer-cutedsl",
                )
            except RuntimeError as exc:
                raise KernelLaunchFailed(f"NVFP4 activation quantization failed: {exc}") from exc

    return _measure(
        run_once,
        _KERNEL_NAMES[scale_format],
        _logical_bytes(num_tokens, hidden_size, scale_format),
    )


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
    return _measure(
        run_once,
        _FLASHINFER_KERNEL_NAME,
        _logical_bytes(num_tokens, hidden_size, scale_format),
    )


def _calibrated_global_scale(source: Any) -> Any:
    """The scalar FlashInferCuteDslNvFp4LinearKernel passes as input_global_scale_inv.

    vLLM sets it to 1 / input_scale, where the checkpoint calibrates
    input_scale = amax / (448 * 6); derive amax from this input instead.
    """

    return (_E4M3_MAX * _FP4_MAX / source.abs().amax().float()).reshape(())


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _logical_bytes(num_tokens: int, hidden_size: int, scale_format: str) -> int:
    """BF16 read, packed FP4 written, and the E4M3 scale tensor written.

    The swizzled kernel walks round_up(num_tokens, 128) rows and zero-fills the
    padded scale tiles, so the scale term counts the whole padded tensor; the
    FP4 output itself is never padded.
    """

    groups = hidden_size // _GROUP_SIZE
    if scale_format == _SWIZZLED_SCALE_FORMAT:
        scale_bytes = _round_up(num_tokens, _SWIZZLE_ROW_TILE) * _round_up(
            groups, _SWIZZLE_GROUP_TILE
        )
    else:
        scale_bytes = num_tokens * groups
    return num_tokens * hidden_size * 2 + num_tokens * hidden_size // 2 + scale_bytes


def _measure(run_once: Any, kernel_name: str, logical_bytes: int) -> ComputeMetrics:
    time_ms = Timer.cupti(run_once, kernel_name=kernel_name)
    energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    return ComputeMetrics(
        time_ms=time_ms,
        tflops=0.0,
        memory_bandwidth_gbps=logical_bytes / (time_ms / 1000.0) / 1e9,
        energy_j=energy_j,
    )
