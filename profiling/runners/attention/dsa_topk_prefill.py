"""Torch profiling runner for GLM-5.2 prefill DSA top-k selection.

The timed callable is the complete vectorized semantic composite. It models
the per-row writes performed by production ``top_k_per_row_prefill``, including
``-1`` slots for short spans. The indexer's separate outer
``topk_indices_buffer[...] = -1`` launch initializes graph-padding rows and is
not part of this kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_NUM_SEQUENCES = 1
_TOP_K = 2048
_LOGITS_DTYPE = DType.FP32
_INDEX_DTYPE = "int32"
_SPAN_MODE = "single_causal_tail"
_REQUIRED_GPU = "NVIDIA H200"
_VLLM_SUPPORTED_GPUS = ("NVIDIA H200", "NVIDIA B200")
_VLLM_KERNEL_NAME = "topKPerRowPrefill"
_SGLANG_SUPPORTED_GPUS = ("NVIDIA B200",)
_SGLANG_KERNEL_NAME = "topk_transform_prefill_kernel"


@dataclass(frozen=True)
class _DsaTopkPrefillOperands:
    logits_backing: Any
    logits: Any
    row_starts: Any
    row_ends: Any
    out: Any
    valid_mask: Any
    natural_output: Any
    long_row_indices: Any


def _validate_args(
    num_queries: int,
    num_keys: int,
    num_sequences: int,
    top_k: int,
    logits_row_stride: int,
    logits_dtype: DType | str,
    index_dtype: str,
    span_mode: str,
) -> tuple[int, int, int, int, int, DType, str, str]:
    num_queries = int(num_queries)
    num_keys = int(num_keys)
    num_sequences = int(num_sequences)
    top_k = int(top_k)
    logits_row_stride = int(logits_row_stride)
    logits_dtype = DType.from_value(logits_dtype)
    index_dtype = str(index_dtype)
    span_mode = str(span_mode)

    if num_queries <= 0 or num_keys <= 0:
        raise ValueError(f"num_queries and num_keys must be > 0, got {num_queries} and {num_keys}")
    if num_queries > num_keys:
        raise ValueError(f"num_queries must be <= num_keys, got {num_queries} and {num_keys}")
    if num_sequences != _NUM_SEQUENCES:
        raise ValueError(f"dsa_topk_prefill requires num_sequences=1, got {num_sequences}")
    if top_k != _TOP_K:
        raise ValueError(f"dsa_topk_prefill requires top_k=2048, got {top_k}")
    if logits_row_stride <= 0 or logits_row_stride < num_keys:
        raise ValueError(
            "logits_row_stride must be positive and >= num_keys, "
            f"got {logits_row_stride} for num_keys={num_keys}"
        )
    if logits_dtype is not _LOGITS_DTYPE:
        raise ValueError(f"dsa_topk_prefill requires logits_dtype=fp32, got {logits_dtype.value}")
    if index_dtype != _INDEX_DTYPE:
        raise ValueError(f"dsa_topk_prefill requires index_dtype='int32', got {index_dtype!r}")
    if span_mode != _SPAN_MODE:
        raise ValueError(f"dsa_topk_prefill requires span_mode='{_SPAN_MODE}', got {span_mode!r}")
    return (
        num_queries,
        num_keys,
        num_sequences,
        top_k,
        logits_row_stride,
        logits_dtype,
        index_dtype,
        span_mode,
    )


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the torch dsa_topk_prefill backend")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _REQUIRED_GPU:
        raise ProfilerNotImplemented(
            f"torch dsa_topk_prefill is verified only on {_REQUIRED_GPU}, got {gpu_name}"
        )


def _validate_vllm_args(
    num_queries: int,
    num_keys: int,
    num_sequences: int,
    top_k: int,
    logits_row_stride: int,
    logits_dtype: DType | str,
    index_dtype: str,
    span_mode: str,
) -> tuple[int, int, int, int, int, DType, str, str]:
    """Validate the production backend's argument identity.

    There is deliberately no `num_queries` ceiling: `top_k_per_row_prefill`
    launches one block per row and falls back to a second radix pass beyond
    12,288 rows, so any row count is a shape the production kernel handles. What
    a caller must not do is ask for an operand that will not fit — that bound
    belongs to the sweep grid's `infeasible_mask`, which sizes the padded logits
    block, not to per-spec validation here.
    """
    validated = _validate_args(
        num_queries,
        num_keys,
        num_sequences,
        top_k,
        logits_row_stride,
        logits_dtype,
        index_dtype,
        span_mode,
    )
    return validated


def _validate_vllm_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the dsa_topk_prefill vllm_cuda backend")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name not in _VLLM_SUPPORTED_GPUS:
        raise ProfilerNotImplemented(
            "dsa_topk_prefill vllm_cuda is verified only on "
            f"{' or '.join(_VLLM_SUPPORTED_GPUS)}, got {gpu_name}"
        )


def _resolve_vllm_prefill_op(torch: Any) -> Any:
    namespace = getattr(torch.ops, "_C", None)
    if namespace is None:
        raise ProfilerNotImplemented("torch.ops._C is unavailable in the vLLM environment")
    try:
        op = namespace.top_k_per_row_prefill
    except AttributeError as exc:
        raise ProfilerNotImplemented("torch.ops._C.top_k_per_row_prefill is unavailable") from exc
    if not callable(op):
        raise ProfilerNotImplemented("torch.ops._C.top_k_per_row_prefill is not callable")
    return op


def _load_vllm_cuda_backend() -> tuple[Any, Any]:
    """Load Torch and vLLM's extension only inside the selected worker."""
    try:
        import torch
        from vllm import _custom_ops as custom_ops
    except (ImportError, OSError, RuntimeError) as exc:
        raise ProfilerNotImplemented(
            "the instrumented vLLM CUDA environment is required for dsa_topk_prefill:vllm_cuda"
        ) from exc

    del custom_ops  # Importing the extension registers the torch.ops._C schema.
    return torch, _resolve_vllm_prefill_op(torch)


def _build_operands(
    torch: Any,
    *,
    num_queries: int,
    num_keys: int,
    top_k: int,
    logits_row_stride: int,
    device: str,
) -> _DsaTopkPrefillOperands:
    """Build padded row-major logits and fixed causal-tail metadata."""
    logits_backing = torch.empty(
        (num_queries, logits_row_stride),
        dtype=torch.float32,
        device=device,
    )
    # A one-row template keeps construction deterministic and avoids a second
    # full padded-logits allocation. Values are finite, signed, and non-tied.
    column_template = torch.linspace(
        -1.0,
        1.0,
        logits_row_stride,
        dtype=torch.float32,
        device=device,
    )
    logits_backing.copy_(column_template)
    logits = logits_backing[:, :num_keys]

    row_starts = torch.zeros(num_queries, dtype=torch.int32, device=device)
    row_ends = torch.arange(
        num_keys - num_queries + 1,
        num_keys + 1,
        dtype=torch.int32,
        device=device,
    )
    out = torch.empty((num_queries, top_k), dtype=torch.int32, device=device)

    key_positions = torch.arange(num_keys, dtype=torch.int32, device=device)
    valid_mask = (key_positions.unsqueeze(0) >= row_starts.unsqueeze(1)) & (
        key_positions.unsqueeze(0) < row_ends.unsqueeze(1)
    )
    span_lengths = row_ends - row_starts
    natural_indices = torch.arange(top_k, dtype=torch.int32, device=device)
    natural_output = torch.where(
        natural_indices.unsqueeze(0) < span_lengths.unsqueeze(1),
        natural_indices.unsqueeze(0),
        torch.full((), -1, dtype=torch.int32, device=device),
    )
    long_row_indices = torch.nonzero(span_lengths > top_k, as_tuple=False).flatten()
    return _DsaTopkPrefillOperands(
        logits_backing=logits_backing,
        logits=logits,
        row_starts=row_starts,
        row_ends=row_ends,
        out=out,
        valid_mask=valid_mask,
        natural_output=natural_output,
        long_row_indices=long_row_indices,
    )


def _torch_composite(operands: _DsaTopkPrefillOperands) -> Any:
    """Write all output slots using a vectorized, loop-free Torch composite."""
    operands.out.copy_(operands.natural_output)
    if operands.long_row_indices.numel() == 0:
        return operands.out

    long_logits = operands.logits.index_select(0, operands.long_row_indices)
    long_mask = operands.valid_mask.index_select(0, operands.long_row_indices)
    scores = long_logits.masked_fill(~long_mask, float("-inf"))
    selected = _topk_indices(scores, operands.out.shape[1], operands.out.dtype)
    operands.out.index_copy_(0, operands.long_row_indices, selected)
    return operands.out


def _topk_indices(scores: Any, top_k: int, index_dtype: Any) -> Any:
    """Return selected indices through the tensor's Torch namespace."""
    # Keeping the operation on the tensor permits CPU helper tests without a
    # module-level Torch import, preserving registry/RunnerRef laziness.
    return scores.topk(top_k, dim=1, largest=True, sorted=True).indices.to(dtype=index_dtype)


def _load_sglang_cuda_backend() -> tuple[Any, Any]:
    try:
        import torch
        from sgl_kernel import fast_topk_transform_fused
    except (ImportError, OSError, RuntimeError) as exc:
        raise ProfilerNotImplemented(
            "dsa_topk_prefill:sglang_cuda requires the SGLang environment"
        ) from exc
    if not callable(fast_topk_transform_fused):
        raise ProfilerNotImplemented("sgl_kernel.fast_topk_transform_fused is not callable")
    return torch, fast_topk_transform_fused


def _validate_sglang_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for dsa_topk_prefill:sglang_cuda")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name not in _SGLANG_SUPPORTED_GPUS:
        raise ProfilerNotImplemented(
            "dsa_topk_prefill:sglang_cuda is verified only on "
            f"{' or '.join(_SGLANG_SUPPORTED_GPUS)}, got {gpu_name}"
        )


@dataclass(frozen=True)
class _SglangPagedExtras:
    lengths: Any
    src_page_table: Any
    cu_seqlens_q: Any


def _build_sglang_paged_extras(
    torch: Any,
    operands: _DsaTopkPrefillOperands,
    *,
    num_queries: int,
    num_keys: int,
    device: str,
) -> _SglangPagedExtras:
    lengths = operands.row_ends - operands.row_starts
    src_page_table = torch.arange(num_keys, dtype=torch.int32, device=device).unsqueeze(0)
    cu_seqlens_q = torch.tensor([0, num_queries], dtype=torch.int32, device=device)
    return _SglangPagedExtras(
        lengths=lengths.contiguous(),
        src_page_table=src_page_table,
        cu_seqlens_q=cu_seqlens_q,
    )


def _launch_sglang_topk(
    callable_: Any,
    operands: _DsaTopkPrefillOperands,
    extras: _SglangPagedExtras,
    *,
    top_k: int,
) -> Any:
    """Forward the exact public SGLang wrapper arguments used by serving."""
    return callable_(
        score=operands.logits,
        lengths=extras.lengths,
        page_table_size_1=extras.src_page_table,
        cu_seqlens_q=extras.cu_seqlens_q,
        topk=top_k,
        row_starts=operands.row_starts,
    )


def _logical_bytes(
    *,
    num_queries: int,
    num_keys: int,
    top_k: int,
) -> int:
    """Return semantic logical traffic, excluding padding and intermediates."""
    if min(num_queries, num_keys, top_k) <= 0:
        raise ValueError("logical-byte dimensions must be > 0")
    if num_queries > num_keys:
        raise ValueError("num_queries must be <= num_keys")
    first_span = num_keys - num_queries + 1
    sum_valid_span_lengths = num_queries * (first_span + num_keys) // 2
    return 4 * sum_valid_span_lengths + 8 * num_queries + 4 * num_queries * top_k


def profile_dsa_topk_prefill_torch(
    num_queries: int,
    num_keys: int,
    num_sequences: int,
    top_k: int,
    logits_row_stride: int,
    logits_dtype: DType | str,
    index_dtype: str,
    span_mode: str,
) -> ComputeMetrics:
    """Profile the complete Torch prefill DSA top-k semantic composite."""
    (
        num_queries,
        num_keys,
        _num_sequences,
        top_k,
        logits_row_stride,
        _logits_dtype,
        _index_dtype,
        _span_mode,
    ) = _validate_args(
        num_queries,
        num_keys,
        num_sequences,
        top_k,
        logits_row_stride,
        logits_dtype,
        index_dtype,
        span_mode,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the torch dsa_topk_prefill backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_operands(
            torch,
            num_queries=num_queries,
            num_keys=num_keys,
            top_k=top_k,
            logits_row_stride=logits_row_stride,
            device="cuda",
        )

        def kernel() -> Any:
            return _torch_composite(operands)

        time_ms = Timer.cuda_event(kernel, warmup=5)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        # Logical traffic counts valid FP32 logits, two int32 span values per
        # row, and all int32 output writes. It excludes padded columns, Torch
        # intermediates, and physical memory transactions.
        logical_bytes = _logical_bytes(
            num_queries=num_queries,
            num_keys=num_keys,
            top_k=top_k,
        )
        bandwidth_gbps = logical_bytes / (time_ms / 1000.0) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def profile_dsa_topk_prefill_vllm_cuda(
    num_queries: int,
    num_keys: int,
    num_sequences: int,
    top_k: int,
    logits_row_stride: int,
    logits_dtype: DType | str,
    index_dtype: str,
    span_mode: str,
) -> ComputeMetrics:
    """Profile vLLM's one-launch production prefill DSA top-k kernel."""
    (
        num_queries,
        num_keys,
        _num_sequences,
        top_k,
        logits_row_stride,
        _logits_dtype,
        _index_dtype,
        _span_mode,
    ) = _validate_vllm_args(
        num_queries,
        num_keys,
        num_sequences,
        top_k,
        logits_row_stride,
        logits_dtype,
        index_dtype,
        span_mode,
    )
    torch, top_k_per_row_prefill = _load_vllm_cuda_backend()
    _validate_vllm_cuda_device(torch)

    try:
        operands = _build_operands(
            torch,
            num_queries=num_queries,
            num_keys=num_keys,
            top_k=top_k,
            logits_row_stride=logits_row_stride,
            device="cuda",
        )

        def kernel() -> None:
            top_k_per_row_prefill(
                operands.logits,
                operands.row_starts,
                operands.row_ends,
                operands.out,
                num_queries,
                operands.logits.stride(0),
                operands.logits.stride(1),
                top_k,
            )

        # Compile/initialize the exact shape before the formal kernel-only
        # capture. The callable itself contains only the custom-op invocation.
        kernel()
        torch.cuda.synchronize()
        time_ms = Timer.cupti(kernel, kernel_name=_VLLM_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)

        # Same semantic accounting as the Torch backend: valid FP32 logits,
        # two int32 span values per row, and all int32 output slots. Padded
        # columns, the excluded outer fill, and physical transactions are not
        # logical traffic for this launch.
        logical_bytes = _logical_bytes(
            num_queries=num_queries,
            num_keys=num_keys,
            top_k=top_k,
        )
        bandwidth_gbps = logical_bytes / (time_ms / 1000.0) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except (RuntimeError, OSError) as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def profile_dsa_topk_prefill_sglang_cuda(
    num_queries: int,
    num_keys: int,
    num_sequences: int,
    top_k: int,
    logits_row_stride: int,
    logits_dtype: DType | str,
    index_dtype: str,
    span_mode: str,
) -> ComputeMetrics:
    """Profile SGLang's fused prefill top-k and page-table transform."""
    (
        num_queries,
        num_keys,
        _num_sequences,
        top_k,
        logits_row_stride,
        _logits_dtype,
        _index_dtype,
        _span_mode,
    ) = _validate_args(
        num_queries,
        num_keys,
        num_sequences,
        top_k,
        logits_row_stride,
        logits_dtype,
        index_dtype,
        span_mode,
    )
    torch, fast_topk_transform_fused = _load_sglang_cuda_backend()
    _validate_sglang_cuda_device(torch)
    try:
        operands = _build_operands(
            torch,
            num_queries=num_queries,
            num_keys=num_keys,
            top_k=top_k,
            logits_row_stride=logits_row_stride,
            device="cuda",
        )
        extras = _build_sglang_paged_extras(
            torch,
            operands,
            num_queries=num_queries,
            num_keys=num_keys,
            device="cuda",
        )

        def kernel() -> Any:
            return _launch_sglang_topk(
                fast_topk_transform_fused,
                operands,
                extras,
                top_k=top_k,
            )

        kernel()
        torch.cuda.synchronize()
        time_ms = Timer.cupti(kernel, kernel_name=_SGLANG_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError("dsa_topk_prefill:sglang_cuda ran out of CUDA memory") from exc
    except (RuntimeError, OSError) as exc:
        raise KernelLaunchFailed(str(exc)) from exc

    logical_bytes = (
        _logical_bytes(
            num_queries=num_queries,
            num_keys=num_keys,
            top_k=top_k,
        )
        + 4 * num_queries * top_k
    )
    bandwidth_gbps = logical_bytes / (time_ms / 1000.0) / 1e9 if time_ms > 0 else 0.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(bandwidth_gbps),
        energy_j=float(energy_j),
    )
