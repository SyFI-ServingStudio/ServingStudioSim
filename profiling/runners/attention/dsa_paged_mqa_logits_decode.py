"""Profilers for GLM-5.2's paged-decode DSA MQA logits.

The Torch backend times the complete semantic composite. The production-aligned
backend times only DeepGEMM's fused main kernel; metadata construction and JIT
setup stay outside timing. Accounting follows the production schedule's
block-rounded context and describes logical traffic, not physical TMA/cache
transactions or Torch intermediates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_NEXT_N = 1
_NUM_HEADS = 64
_HEAD_DIM = 128
_BLOCK_SIZE = 64
_Q_DTYPE = DType.FP8_E4M3
_CACHE_DTYPE = DType.FP8_E4M3
_SCALE_DTYPE = DType.FP32
_WEIGHT_DTYPE = DType.FP32
_OUTPUT_DTYPE = DType.FP32
_CONTEXT_MODE = "uniform"
_PAGE_MAPPING = "unique_scattered"
_CACHE_FORMAT = "page_planar_fp8_fp32_scale"
_REQUIRED_GPU = "NVIDIA H200"
_REQUIRED_H200_SMS = 132
_DEEPGEMM_KERNEL_NAME = "sm90_fp8_paged_mqa_logits"


@dataclass(frozen=True)
class _DsaPagedMqaLogitsDecodeOperands:
    q: Any
    cache: Any
    weights: Any
    context_lens: Any
    block_table: Any
    key_view: Any
    scale_view: Any
    logical_pages: int
    padded_context: int


def _validate_args(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    scale_dtype: DType | str,
    weight_dtype: DType | str,
    output_dtype: DType | str,
    context_mode: str,
    page_mapping: str,
    cache_format: str,
    clean_logits: bool,
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    DType,
    DType,
    DType,
    DType,
    DType,
    str,
    str,
    str,
    bool,
]:
    batch_size = int(batch_size)
    context_len = int(context_len)
    next_n = int(next_n)
    max_model_len = int(max_model_len)
    num_heads = int(num_heads)
    head_dim = int(head_dim)
    block_size = int(block_size)
    q_dtype = DType.from_value(q_dtype)
    cache_dtype = DType.from_value(cache_dtype)
    scale_dtype = DType.from_value(scale_dtype)
    weight_dtype = DType.from_value(weight_dtype)
    output_dtype = DType.from_value(output_dtype)
    context_mode = str(context_mode)
    page_mapping = str(page_mapping)
    cache_format = str(cache_format)

    if batch_size <= 0 or context_len <= 0 or max_model_len <= 0:
        raise ValueError(
            "batch_size, context_len, and max_model_len must be > 0, got "
            f"{batch_size}, {context_len}, and {max_model_len}"
        )
    if context_len > max_model_len:
        raise ValueError(
            f"context_len must be <= max_model_len, got {context_len} and {max_model_len}"
        )
    if next_n != _NEXT_N:
        raise ValueError(f"dsa_paged_mqa_logits_decode requires next_n=1, got {next_n}")
    if (num_heads, head_dim, block_size) != (_NUM_HEADS, _HEAD_DIM, _BLOCK_SIZE):
        raise ValueError(
            "dsa_paged_mqa_logits_decode requires "
            "(num_heads, head_dim, block_size) == (64, 128, 64), "
            f"got ({num_heads}, {head_dim}, {block_size})"
        )
    if q_dtype is not _Q_DTYPE or cache_dtype is not _CACHE_DTYPE:
        raise ValueError(
            "dsa_paged_mqa_logits_decode requires "
            "q_dtype=cache_dtype=fp8_e4m3, "
            f"got {q_dtype.value} and {cache_dtype.value}"
        )
    if (
        scale_dtype is not _SCALE_DTYPE
        or weight_dtype is not _WEIGHT_DTYPE
        or output_dtype is not _OUTPUT_DTYPE
    ):
        raise ValueError(
            "dsa_paged_mqa_logits_decode requires "
            "scale_dtype=weight_dtype=output_dtype=fp32, got "
            f"{scale_dtype.value}, {weight_dtype.value}, and {output_dtype.value}"
        )
    if context_mode != _CONTEXT_MODE:
        raise ValueError(
            f"dsa_paged_mqa_logits_decode requires context_mode='{_CONTEXT_MODE}', "
            f"got {context_mode!r}"
        )
    if page_mapping != _PAGE_MAPPING:
        raise ValueError(
            f"dsa_paged_mqa_logits_decode requires page_mapping='{_PAGE_MAPPING}', "
            f"got {page_mapping!r}"
        )
    if cache_format != _CACHE_FORMAT:
        raise ValueError(
            f"dsa_paged_mqa_logits_decode requires cache_format='{_CACHE_FORMAT}', "
            f"got {cache_format!r}"
        )
    if not isinstance(clean_logits, bool):
        raise TypeError("clean_logits must be a bool")
    if clean_logits:
        raise ValueError("dsa_paged_mqa_logits_decode requires clean_logits=false")

    return (
        batch_size,
        context_len,
        next_n,
        max_model_len,
        num_heads,
        head_dim,
        block_size,
        q_dtype,
        cache_dtype,
        scale_dtype,
        weight_dtype,
        output_dtype,
        context_mode,
        page_mapping,
        cache_format,
        clean_logits,
    )


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the torch dsa_paged_mqa_logits_decode backend"
        )
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _REQUIRED_GPU:
        raise ProfilerNotImplemented(
            f"torch dsa_paged_mqa_logits_decode is verified only on {_REQUIRED_GPU}, got {gpu_name}"
        )


def _validate_deepgemm_cuda_device(torch: Any) -> int:
    """Validate the exact H200 deployment identity and return its SM count."""
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the dsa_paged_mqa_logits_decode vllm_deepgemm_fp8 backend"
        )
    device = torch.cuda.current_device()
    gpu_name = str(torch.cuda.get_device_name(device))
    if gpu_name != _REQUIRED_GPU:
        raise ProfilerNotImplemented(
            "dsa_paged_mqa_logits_decode vllm_deepgemm_fp8 is verified only on "
            f"{_REQUIRED_GPU}, got {gpu_name}"
        )
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    if num_sms != _REQUIRED_H200_SMS:
        raise ProfilerNotImplemented(
            "dsa_paged_mqa_logits_decode vllm_deepgemm_fp8 requires the verified "
            f"{_REQUIRED_H200_SMS}-SM H200 schedule, got {num_sms} SMs"
        )
    return num_sms


def _load_deepgemm_backend() -> tuple[Any, Any]:
    """Load vLLM's serving wrapper only inside the selected worker process."""
    try:
        import torch
        from vllm.utils import deep_gemm
    except (ImportError, OSError) as exc:
        raise ProfilerNotImplemented(
            "the instrumented vLLM/DeepGEMM environment is required for "
            "dsa_paged_mqa_logits_decode:vllm_deepgemm_fp8"
        ) from exc

    support_api = getattr(deep_gemm, "is_deep_gemm_supported", None)
    if not callable(support_api):
        raise ProfilerNotImplemented("vllm.utils.deep_gemm.is_deep_gemm_supported is unavailable")
    try:
        supported = support_api()
    except (RuntimeError, OSError) as exc:
        raise ProfilerNotImplemented(
            "DeepGEMM support could not be initialized for "
            "dsa_paged_mqa_logits_decode:vllm_deepgemm_fp8"
        ) from exc
    if not supported:
        raise ProfilerNotImplemented(
            "DeepGEMM is unavailable or unsupported for "
            "dsa_paged_mqa_logits_decode:vllm_deepgemm_fp8"
        )
    if not callable(getattr(deep_gemm, "get_paged_mqa_logits_metadata", None)):
        raise ProfilerNotImplemented(
            "vllm.utils.deep_gemm.get_paged_mqa_logits_metadata is unavailable"
        )
    if _paged_mqa_logits_entry_point(deep_gemm) is None:
        raise ProfilerNotImplemented(
            "vllm.utils.deep_gemm exposes neither fp8_paged_mqa_logits nor "
            "fp8_fp4_paged_mqa_logits"
        )
    return torch, deep_gemm


def _paged_mqa_logits_entry_point(deep_gemm: Any) -> Any:
    """The fork renamed `fp8_paged_mqa_logits` to `fp8_fp4_paged_mqa_logits`
    when it unified the FP8 and MXFP4 dispatch behind a tuple-typed `q`. Both
    names reach the same paged DeepGEMM kernel on the FP8 path."""
    for name in ("fp8_paged_mqa_logits", "fp8_fp4_paged_mqa_logits"):
        entry_point = getattr(deep_gemm, name, None)
        if callable(entry_point):
            return entry_point
    return None


def _stable_values(torch: Any, shape: tuple[int, ...], *, phase: int, device: str) -> Any:
    total = 1
    for dimension in shape:
        total *= dimension
    values = torch.arange(total, dtype=torch.int64, device=device)
    values = ((values + phase) % 17 - 8).to(torch.float32) / 16.0
    return values.reshape(shape)


def _page_views(
    torch: Any,
    cache: Any,
    *,
    block_size: int,
    head_dim: int,
) -> tuple[Any, Any]:
    """Expose page-planar E4M3 keys and raw-FP32 scales as cache aliases."""
    num_pages = cache.shape[0]
    page_key_bytes = block_size * head_dim
    pages = cache.reshape(num_pages, block_size * (head_dim + 4))
    key_view = (
        pages[:, :page_key_bytes].reshape(num_pages, block_size, head_dim).view(torch.float8_e4m3fn)
    )
    scale_bytes = pages[:, page_key_bytes:].reshape(num_pages, block_size, 4)
    scale_view = scale_bytes.view(torch.float32).squeeze(-1)
    return key_view, scale_view


def _cache_page_templates(
    torch: Any,
    *,
    block_size: int,
    head_dim: int,
    device: str,
) -> tuple[Any, Any]:
    """Build deterministic page contents with storage independent of page count."""
    key_template = _stable_values(
        torch,
        (block_size, head_dim),
        phase=5,
        device=device,
    ).to(torch.float8_e4m3fn)
    scale_template = (
        0.75
        + (
            _stable_values(
                torch,
                (block_size,),
                phase=13,
                device=device,
            )
            + 0.5
        )
        / 4.0
    )
    return key_template, scale_template


def _build_operands(
    torch: Any,
    *,
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    device: str,
) -> _DsaPagedMqaLogitsDecodeOperands:
    """Construct exact GLM operands and deterministic unique scattered pages."""
    del max_model_len
    logical_pages = (context_len + block_size - 1) // block_size
    padded_context = logical_pages * block_size
    referenced_page_count = batch_size * logical_pages
    num_pages = max(256, 2 * referenced_page_count + 1)

    q = _stable_values(
        torch,
        (batch_size, next_n, num_heads, head_dim),
        phase=1,
        device=device,
    ).to(torch.float8_e4m3fn)
    weights = (
        _stable_values(
            torch,
            (batch_size * next_n, num_heads),
            phase=9,
            device=device,
        )
        / 4.0
    )
    context_lens = torch.full(
        (batch_size, next_n),
        context_len,
        dtype=torch.int32,
        device=device,
    )
    physical_pages = (
        2
        * torch.arange(
            referenced_page_count,
            dtype=torch.int32,
            device=device,
        )
        + 1
    )
    block_table = physical_pages.reshape(batch_size, logical_pages).contiguous()

    cache = torch.empty(
        (num_pages, block_size, 1, head_dim + 4),
        dtype=torch.uint8,
        device=device,
    )
    key_view, scale_view = _page_views(
        torch,
        cache,
        block_size=block_size,
        head_dim=head_dim,
    )
    key_template, scale_template = _cache_page_templates(
        torch,
        block_size=block_size,
        head_dim=head_dim,
        device=device,
    )
    # Page counts reach O(10^6) at the largest grid points. Never construct
    # page-sized int64/FP32 initialization ramps; copy_ broadcasts these small
    # templates directly into the exact production cache allocation.
    key_view.copy_(key_template)
    scale_view.copy_(scale_template)

    return _DsaPagedMqaLogitsDecodeOperands(
        q=q,
        cache=cache,
        weights=weights,
        context_lens=context_lens,
        block_table=block_table,
        key_view=key_view,
        scale_view=scale_view,
        logical_pages=logical_pages,
        padded_context=padded_context,
    )


def _torch_composite(
    operands: _DsaPagedMqaLogitsDecodeOperands,
    *,
    context_len: int,
    max_model_len: int,
) -> Any:
    """Compute the complete vectorized paged-decode semantic composite."""
    batch_size, next_n, num_heads, head_dim = operands.q.shape
    physical_pages = operands.block_table[:, : operands.logical_pages].long()
    keys = operands.key_view[physical_pages].reshape(batch_size, operands.padded_context, head_dim)
    scales = operands.scale_view[physical_pages].reshape(batch_size, operands.padded_context)
    row_keys = (
        keys.unsqueeze(1)
        .expand(batch_size, next_n, operands.padded_context, head_dim)
        .reshape(batch_size * next_n, operands.padded_context, head_dim)
    )
    row_scales = (
        scales.unsqueeze(1)
        .expand(batch_size, next_n, operands.padded_context)
        .reshape(batch_size * next_n, operands.padded_context)
    )
    queries = operands.q.reshape(batch_size * next_n, num_heads, head_dim)
    dots = torch_einsum(queries.float(), row_keys.float())
    reduced = (dots.relu() * operands.weights.unsqueeze(-1)).sum(dim=1) * row_scales
    output = reduced.new_full((batch_size * next_n, max_model_len), float("nan"))
    output[:, :context_len] = reduced[:, :context_len]
    return output


def torch_einsum(queries: Any, keys: Any) -> Any:
    """Keep the contraction isolated for tests without importing Torch globally."""
    return queries @ keys.transpose(-1, -2)


def _scheduled_work(
    *,
    batch_size: int,
    context_len: int,
    next_n: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
) -> tuple[int, int, int]:
    """Return ``(logical_pages, C, nominal_flops)`` for the padded schedule."""
    if min(batch_size, context_len, next_n, num_heads, head_dim, block_size) <= 0:
        raise ValueError("scheduled-work dimensions must be > 0")
    logical_pages = (context_len + block_size - 1) // block_size
    scheduled_cells = batch_size * next_n * logical_pages * block_size
    nominal_flops = 2 * scheduled_cells * num_heads * head_dim
    return logical_pages, scheduled_cells, nominal_flops


def _logical_scheduled_bytes(
    *,
    batch_size: int,
    context_len: int,
    next_n: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
) -> int:
    """Return scheduled logical bytes, excluding Torch/TMA intermediates.

    The count covers Q, weights, block-rounded cache keys/scales, block/context
    metadata, and ``4 * C`` output bytes. ``max_model_len`` deliberately is not
    an argument: it controls the returned allocation shape, not scheduled cache
    work in the production kernel model.
    """
    logical_pages, scheduled_cells, _nominal_flops = _scheduled_work(
        batch_size=batch_size,
        context_len=context_len,
        next_n=next_n,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
    )
    return (
        batch_size * next_n * num_heads * head_dim
        + 4 * batch_size * next_n * num_heads
        + (head_dim + 4) * scheduled_cells
        + 4 * batch_size * logical_pages
        + 4 * batch_size * next_n
        + 4 * scheduled_cells
    )


def _prepare_deepgemm_call(
    deep_gemm: Any,
    operands: _DsaPagedMqaLogitsDecodeOperands,
    *,
    block_size: int,
    num_sms: int,
    max_model_len: int,
) -> tuple[Any, Any, Any]:
    """Adapt pinned contexts and build metadata before returning the timed call."""
    entry_point = _paged_mqa_logits_entry_point(deep_gemm)
    # The unified entry point takes `q = (values, scales_or_None)`; the FP8 path
    # passes None because the per-token scale is folded into `weights`.
    unified = getattr(entry_point, "__name__", "") == "fp8_fp4_paged_mqa_logits"
    query = (operands.q, None) if unified else operands.q
    # The unified DeepGEMM API asserts `context_lens.dim() == 2`
    # (csrc/apis/attention.hpp), and vLLM's indexer unsqueezes to (B, 1) before
    # calling both this and the metadata builder. The older entry point took the
    # flat (B,) view. Feed each the layout it was built for.
    runnable_context_lens = (
        operands.context_lens.contiguous()
        if unified
        else operands.context_lens[:, 0].contiguous()
    )
    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        runnable_context_lens,
        block_size=block_size,
        num_sms=num_sms,
    )

    def kernel() -> Any:
        return entry_point(
            query,
            operands.cache,
            operands.weights,
            runnable_context_lens,
            operands.block_table,
            schedule_metadata,
            max_model_len,
            clean_logits=False,
        )

    return runnable_context_lens, schedule_metadata, kernel


def profile_dsa_paged_mqa_logits_decode_torch(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    scale_dtype: DType | str,
    weight_dtype: DType | str,
    output_dtype: DType | str,
    context_mode: str,
    page_mapping: str,
    cache_format: str,
    clean_logits: bool,
) -> ComputeMetrics:
    """Profile the complete Torch paged-decode DSA-logits composite."""
    (
        batch_size,
        context_len,
        next_n,
        max_model_len,
        num_heads,
        head_dim,
        block_size,
        _q_dtype,
        _cache_dtype,
        _scale_dtype,
        _weight_dtype,
        _output_dtype,
        _context_mode,
        _page_mapping,
        _cache_format,
        _clean_logits,
    ) = _validate_args(
        batch_size,
        context_len,
        next_n,
        max_model_len,
        num_heads,
        head_dim,
        block_size,
        q_dtype,
        cache_dtype,
        scale_dtype,
        weight_dtype,
        output_dtype,
        context_mode,
        page_mapping,
        cache_format,
        clean_logits,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the torch dsa_paged_mqa_logits_decode backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_operands(
            torch,
            batch_size=batch_size,
            context_len=context_len,
            next_n=next_n,
            max_model_len=max_model_len,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            device="cuda",
        )

        def kernel() -> Any:
            return _torch_composite(
                operands,
                context_len=context_len,
                max_model_len=max_model_len,
            )

        time_ms = Timer.cuda_event(kernel, warmup=5)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )
        _logical_pages, _scheduled_cells, nominal_flops = _scheduled_work(
            batch_size=batch_size,
            context_len=context_len,
            next_n=next_n,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
        )
        logical_bytes = _logical_scheduled_bytes(
            batch_size=batch_size,
            context_len=context_len,
            next_n=next_n,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
        )
        elapsed_seconds = time_ms / 1000.0
        tflops = nominal_flops / elapsed_seconds / 1e12 if elapsed_seconds > 0 else 0.0
        bandwidth_gbps = logical_bytes / elapsed_seconds / 1e9 if elapsed_seconds > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def profile_dsa_paged_mqa_logits_decode_vllm_deepgemm_fp8(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    scale_dtype: DType | str,
    weight_dtype: DType | str,
    output_dtype: DType | str,
    context_mode: str,
    page_mapping: str,
    cache_format: str,
    clean_logits: bool,
) -> ComputeMetrics:
    """Profile vLLM's production-aligned fused DeepGEMM decode kernel."""
    (
        batch_size,
        context_len,
        next_n,
        max_model_len,
        num_heads,
        head_dim,
        block_size,
        _q_dtype,
        _cache_dtype,
        _scale_dtype,
        _weight_dtype,
        _output_dtype,
        _context_mode,
        _page_mapping,
        _cache_format,
        _clean_logits,
    ) = _validate_args(
        batch_size,
        context_len,
        next_n,
        max_model_len,
        num_heads,
        head_dim,
        block_size,
        q_dtype,
        cache_dtype,
        scale_dtype,
        weight_dtype,
        output_dtype,
        context_mode,
        page_mapping,
        cache_format,
        clean_logits,
    )
    torch, deep_gemm = _load_deepgemm_backend()
    num_sms = _validate_deepgemm_cuda_device(torch)

    try:
        operands = _build_operands(
            torch,
            batch_size=batch_size,
            context_len=context_len,
            next_n=next_n,
            max_model_len=max_model_len,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            device="cuda",
        )
        _runnable_context_lens, _schedule_metadata, kernel = _prepare_deepgemm_call(
            deep_gemm,
            operands,
            block_size=block_size,
            num_sms=num_sms,
            max_model_len=max_model_len,
        )

        # Compile and initialize the exact shape before formal CUPTI timing.
        kernel()
        torch.cuda.synchronize()
        time_ms = Timer.cupti(
            kernel,
            kernel_name=_DEEPGEMM_KERNEL_NAME,
        )
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )
        _logical_pages, _scheduled_cells, nominal_flops = _scheduled_work(
            batch_size=batch_size,
            context_len=context_len,
            next_n=next_n,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
        )
        # This models the block-rounded DeepGEMM schedule. Metadata construction,
        # JIT work, physical TMA/cache transactions, and intermediates are excluded.
        logical_bytes = _logical_scheduled_bytes(
            batch_size=batch_size,
            context_len=context_len,
            next_n=next_n,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
        )
        elapsed_seconds = time_ms / 1000.0
        tflops = nominal_flops / elapsed_seconds / 1e12 if elapsed_seconds > 0 else 0.0
        bandwidth_gbps = logical_bytes / elapsed_seconds / 1e9 if elapsed_seconds > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except (RuntimeError, OSError) as exc:
        raise KernelLaunchFailed(str(exc)) from exc
