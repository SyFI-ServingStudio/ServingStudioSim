"""Profilers for GLM-5.2 persistent decode DSA top-k.

The Torch backend times the complete vectorized semantic composite. The
production backend packages a corrected vLLM v0.23-derived
``persistent_topk``: lengths through ``top_k`` retain the pinned natural-index
writer, while every longer row uses the bundled cooperative radix path. Its
workspace memset and one persistent kernel write all ``top_k`` slots. The
indexer's separate outer global index-buffer fill remains excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_MIN_BATCH_SIZE = 1
_MAX_BATCH_SIZE = 256
_TOP_K = 2048
_LOGITS_DTYPE = DType.FP32
_INDEX_DTYPE = "int32"
_CONTEXT_MODE = "uniform"
_REQUIRED_GPU = "NVIDIA H200"
_VLLM_SUPPORTED_GPUS = ("NVIDIA H200", "NVIDIA B200")
_WORKSPACE_BYTES = 1024 * 1024
# The vLLM fork (GLM-5.3-Flash production) ships a newer persistent_topk: rows
# are clamped to min(stride, max_seq_len), and the >32-row FilteredTopK path
# gained a <=32K histogram-4096 short path. Its kpool indexer selects
# index_topk / index_kpool = 512 pools, so this backend also accepts the
# callable's other K instantiations.
_FORK_BACKEND = "vllm_fork_cuda"
_FORK_SUPPORTED_GPUS = ("NVIDIA B200",)
_FORK_TOP_K = frozenset({512, 1024, 2048})
# Known fork bug (capture 20260925_0): the medium path
# (histogram_256_topk, 8K < row <= 32K on the <=32-row kernel) buffers at most
# MAX_BUFFERED_ITEMS threshold-bin candidates and silently drops the rest, so
# its top-k is wrong when that bin overflows. The timing is still the
# production kernel's, so those rows are recorded; the output is not trusted.
_FORK_MEDIUM_MIN_EXCLUSIVE = 8192
_FORK_MEDIUM_MAX = 32768
_FORK_MEDIUM_MAX_BUFFERED_ITEMS = 4096
# The same overflow in the >32-row FilteredTopK long path (row > 32K): it
# buffers at most FILTERED_TOPK_SMEM_INPUT_SIZE threshold-bin candidates
# (20260925_4 cases 11/12, stride 524288: the linspace template puts a whole
# 32K-131K row into one coarse bin). Same handling as the medium path.
_FORK_FILTERED_ROWS_MIN_EXCLUSIVE = 32
_FORK_FILTERED_LONG_MIN_EXCLUSIVE = 32768
_FORK_FILTERED_MAX_BUFFERED_ITEMS = 16384
# The two remaining buffered paths drop overflow the same way (cases 11/12/15,
# strides 262144/524288). The <=32-row decode path (row <= HIST2048_THRESHOLD)
# bins by the top 11 fp16-key bits and double-buffers DBUF threshold-bin items;
# the >32-row short path (row <= 32K) bins by the top 12 bits and keeps at most
# top_k ties. A stride-wide ramp is not the cause to engineer away: ramping over
# the live context instead reads 0.48-1.6x these rows' time (median 0.83), while
# the measured topk_decode is already 9-19% slower than the stride-ramp rows.
_FORK_DECODE_MAX = 8192
_FORK_DECODE_MAX_BUFFERED_ITEMS = 3708
_FORK_DECODE_BIN_SHIFT = 5
_FORK_SHORT_BIN_SHIFT = 4
_FORK_BYTE_BIN_SHIFT = 8


@dataclass(frozen=True)
class _DsaPersistentTopkDecodeOperands:
    logits_backing: Any
    logits: Any
    lengths: Any
    flat_lengths: Any
    out: Any
    valid_mask: Any
    natural_output: Any
    long_row_indices: Any


@dataclass(frozen=True)
class _DsaPersistentTopkDecodeNativeOperands:
    logits_backing: Any
    logits: Any
    lengths: Any
    flat_lengths: Any
    out: Any
    workspace: Any


def _validate_args(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    logits_row_stride: int,
    logits_dtype: DType | str,
    index_dtype: str,
    context_mode: str,
    allowed_top_k: frozenset[int] = frozenset({_TOP_K}),
) -> tuple[int, int, int, int, int, int, DType, str, str]:
    batch_size = int(batch_size)
    context_len = int(context_len)
    next_n = int(next_n)
    max_model_len = int(max_model_len)
    top_k = int(top_k)
    logits_row_stride = int(logits_row_stride)
    logits_dtype = DType.from_value(logits_dtype)
    index_dtype = str(index_dtype)
    context_mode = str(context_mode)

    if not _MIN_BATCH_SIZE <= batch_size <= _MAX_BATCH_SIZE:
        raise ValueError(
            f"dsa_persistent_topk_decode requires 1 <= batch_size <= 256, got {batch_size}"
        )
    if next_n <= 0:
        raise ValueError(f"dsa_persistent_topk_decode requires next_n > 0, got {next_n}")
    if context_len < 0:
        raise ValueError(f"context_len must be >= 0, got {context_len}")
    if context_len < next_n - 1:
        raise ValueError(
            "context_len must be >= next_n - 1 so every speculative row length is "
            f"nonnegative, got context_len={context_len}, next_n={next_n}"
        )
    # `max_model_len` and `logits_row_stride` are KernelConfig fields, i.e. part
    # of the profile.db cache key, so a different value is a different row family
    # rather than an invalid request. They used to be pinned to one measured
    # value; that recorded coverage, not a kernel law, and every context-domain
    # change had to chase the literal. The real laws are kept below.
    if max_model_len <= 0:
        raise ValueError(f"max_model_len must be > 0, got {max_model_len}")
    if context_len > max_model_len:
        raise ValueError(
            f"context_len must be <= max_model_len, got {context_len} and {max_model_len}"
        )
    if top_k not in allowed_top_k:
        required = " or ".join(f"top_k={value}" for value in sorted(allowed_top_k))
        raise ValueError(f"dsa_persistent_topk_decode requires {required}, got {top_k}")
    if logits_row_stride <= 0 or logits_row_stride < max_model_len:
        raise ValueError(
            "logits_row_stride must be positive and >= max_model_len, "
            f"got {logits_row_stride} for max_model_len={max_model_len}"
        )
    if logits_dtype is not _LOGITS_DTYPE:
        raise ValueError(
            f"dsa_persistent_topk_decode requires logits_dtype=fp32, got {logits_dtype.value}"
        )
    if index_dtype != _INDEX_DTYPE:
        raise ValueError(
            f"dsa_persistent_topk_decode requires index_dtype='int32', got {index_dtype!r}"
        )
    if context_mode != _CONTEXT_MODE:
        raise ValueError(
            f"dsa_persistent_topk_decode requires context_mode='uniform', got {context_mode!r}"
        )
    return (
        batch_size,
        context_len,
        next_n,
        max_model_len,
        top_k,
        logits_row_stride,
        logits_dtype,
        index_dtype,
        context_mode,
    )


def _validate_cuda_device(torch: Any, *, backend: str = "torch") -> str:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            f"CUDA is required for the {backend} dsa_persistent_topk_decode backend"
        )
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    supported_gpus = {
        "vllm_cuda": _VLLM_SUPPORTED_GPUS,
        _FORK_BACKEND: _FORK_SUPPORTED_GPUS,
    }.get(backend, (_REQUIRED_GPU,))
    if gpu_name not in supported_gpus:
        raise ProfilerNotImplemented(
            f"{backend} dsa_persistent_topk_decode is verified only on "
            f"{' or '.join(supported_gpus)}, got {gpu_name}"
        )
    return gpu_name


def _build_common_operands(
    torch: Any,
    *,
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    logits_row_stride: int,
    device: str,
) -> tuple[Any, Any, Any, Any, Any]:
    """Build exact logits/length/output storage without row-scaled sources."""
    num_rows = batch_size * next_n
    logits_backing = torch.empty(
        (num_rows, logits_row_stride),
        dtype=torch.float32,
        device=device,
    )
    # One row-sized template broadcasts into the exact padded production
    # allocation. Never materialize a second [B*next_n, row_stride] source.
    column_template = torch.linspace(
        -1.0,
        1.0,
        logits_row_stride,
        dtype=torch.float32,
        device=device,
    )
    logits_backing.copy_(column_template)
    logits = logits_backing[:, :max_model_len]

    per_request_lengths = torch.arange(
        context_len - next_n + 1,
        context_len + 1,
        dtype=torch.int32,
        device=device,
    )
    lengths = per_request_lengths.unsqueeze(0).expand(batch_size, next_n).contiguous()
    flat_lengths = lengths.view(-1)
    out = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)
    return logits_backing, logits, lengths, flat_lengths, out


def _build_operands(
    torch: Any,
    *,
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    logits_row_stride: int,
    device: str,
) -> _DsaPersistentTopkDecodeOperands:
    """Build exact padded logits and pinned row-major decode lengths."""
    logits_backing, logits, lengths, flat_lengths, out = _build_common_operands(
        torch,
        batch_size=batch_size,
        context_len=context_len,
        next_n=next_n,
        max_model_len=max_model_len,
        top_k=top_k,
        logits_row_stride=logits_row_stride,
        device=device,
    )

    key_positions = torch.arange(max_model_len, dtype=torch.int32, device=device)
    valid_mask = key_positions.unsqueeze(0) < flat_lengths.unsqueeze(1)
    output_positions = torch.arange(top_k, dtype=torch.int32, device=device)
    natural_output = torch.where(
        output_positions.unsqueeze(0) < flat_lengths.unsqueeze(1),
        output_positions.unsqueeze(0),
        torch.full((), -1, dtype=torch.int32, device=device),
    )
    long_row_indices = torch.nonzero(flat_lengths > top_k, as_tuple=False).flatten()
    return _DsaPersistentTopkDecodeOperands(
        logits_backing=logits_backing,
        logits=logits,
        lengths=lengths,
        flat_lengths=flat_lengths,
        out=out,
        valid_mask=valid_mask,
        natural_output=natural_output,
        long_row_indices=long_row_indices,
    )


def _build_native_operands(
    torch: Any,
    *,
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    logits_row_stride: int,
    device: str,
) -> _DsaPersistentTopkDecodeNativeOperands:
    """Build production operands without Torch-composite mask/top-k tensors."""
    logits_backing, logits, lengths, flat_lengths, out = _build_common_operands(
        torch,
        batch_size=batch_size,
        context_len=context_len,
        next_n=next_n,
        max_model_len=max_model_len,
        top_k=top_k,
        logits_row_stride=logits_row_stride,
        device=device,
    )
    workspace = torch.empty((_WORKSPACE_BYTES,), dtype=torch.uint8, device=device)
    return _DsaPersistentTopkDecodeNativeOperands(
        logits_backing=logits_backing,
        logits=logits,
        lengths=lengths,
        flat_lengths=flat_lengths,
        out=out,
        workspace=workspace,
    )


def _torch_composite(operands: _DsaPersistentTopkDecodeOperands) -> Any:
    """Write all K slots through a vectorized, loop-free Torch composite."""
    operands.out.copy_(operands.natural_output)
    if operands.long_row_indices.numel() == 0:
        return operands.out

    long_logits = operands.logits.index_select(0, operands.long_row_indices)
    long_mask = operands.valid_mask.index_select(0, operands.long_row_indices)
    scores = long_logits.masked_fill(~long_mask, float("-inf"))
    selected = scores.topk(
        operands.out.shape[1],
        dim=1,
        largest=True,
        sorted=True,
    ).indices.to(dtype=operands.out.dtype)
    operands.out.index_copy_(0, operands.long_row_indices, selected)
    return operands.out


def _validate_semantics(
    torch: Any,
    operands: _DsaPersistentTopkDecodeOperands,
    *,
    top_k: int,
    max_seq_len: int,
) -> None:
    """Compare the constructed CUDA operands with the committed reference."""
    from profiling.runners.attention.dsa_persistent_topk_decode_reference import (
        dsa_persistent_topk_decode_reference,
    )

    logits_before = operands.logits_backing.clone()
    lengths_before = operands.lengths.clone()
    expected = torch.empty_like(operands.out)
    out_storage = operands.out.untyped_storage().data_ptr()
    actual = _torch_composite(operands)
    dsa_persistent_topk_decode_reference(
        operands.logits,
        operands.lengths,
        expected,
        top_k=top_k,
        max_seq_len=max_seq_len,
    )

    short_rows = torch.nonzero(
        operands.flat_lengths <= top_k,
        as_tuple=False,
    ).flatten()
    if short_rows.numel() and not torch.equal(
        actual.index_select(0, short_rows),
        expected.index_select(0, short_rows),
    ):
        raise RuntimeError("Torch composite disagrees with natural/-1 reference rows")

    if operands.long_row_indices.numel():
        actual_long = actual.index_select(0, operands.long_row_indices).to(torch.int64)
        expected_long = expected.index_select(0, operands.long_row_indices).to(torch.int64)
        actual_sets = actual_long.sort(dim=1).values
        expected_sets = expected_long.sort(dim=1).values
        if not torch.equal(actual_sets, expected_sets):
            raise RuntimeError("Torch composite disagrees with long-row reference selections")
        actual_values = operands.logits.index_select(0, operands.long_row_indices).gather(
            1, actual_long
        )
        expected_values = operands.logits.index_select(0, operands.long_row_indices).gather(
            1, expected_long
        )
        if not torch.equal(
            actual_values.sort(dim=1).values,
            expected_values.sort(dim=1).values,
        ):
            raise RuntimeError("Torch composite disagrees with long-row reference values")

    if actual is not operands.out or operands.out.untyped_storage().data_ptr() != out_storage:
        raise RuntimeError("Torch composite did not preserve caller-owned output storage")
    if not torch.equal(operands.logits_backing, logits_before):
        raise RuntimeError("Torch composite mutated logits")
    if not torch.equal(operands.lengths, lengths_before):
        raise RuntimeError("Torch composite mutated lengths")
    for tensor in (operands.logits, operands.lengths):
        if torch._C._overlaps(operands.out, tensor):
            raise RuntimeError("Torch composite output aliases an input")


def _native_call(
    op: Any,
    operands: _DsaPersistentTopkDecodeNativeOperands,
    *,
    top_k: int,
    max_seq_len: int,
) -> None:
    """Invoke only the private corrected v0.23-derived callable."""
    return op(
        operands.logits,
        operands.lengths,
        operands.out,
        operands.workspace,
        top_k,
        max_seq_len,
    )


def _fp16_key_bin(torch: Any, values: Any, shift: int) -> Any:
    """Mirror the fork's fp16 sign-magnitude key, keeping its top ``16 - shift`` bits.

    ``shift`` 8 is ``convert_to_uint8``, 5 ``decode_bin`` and 4 the short path's
    ``extract_coarse_bin_N<12>``.
    """
    bits = values.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    key = torch.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits | 0x8000)
    return key >> shift


def _overflow_buffer(num_rows: int, length: int, top_k: int) -> tuple[int, int] | None:
    """``(bin shift, buffered items)`` of the fork path this row takes, if it buffers."""
    if num_rows > _FORK_FILTERED_ROWS_MIN_EXCLUSIVE:
        if length > _FORK_FILTERED_LONG_MIN_EXCLUSIVE:
            return _FORK_BYTE_BIN_SHIFT, _FORK_FILTERED_MAX_BUFFERED_ITEMS
        return _FORK_SHORT_BIN_SHIFT, top_k
    if length <= _FORK_DECODE_MAX:
        return _FORK_DECODE_BIN_SHIFT, _FORK_DECODE_MAX_BUFFERED_ITEMS
    if length <= _FORK_MEDIUM_MAX:
        return _FORK_BYTE_BIN_SHIFT, _FORK_MEDIUM_MAX_BUFFERED_ITEMS
    return None


def _medium_overflow_explains(
    torch: Any,
    row_logits: Any,
    actual_indices: Any,
    expected_values: Any,
    num_rows: int = 1,
) -> bool:
    """Return whether a threshold-bin buffer overflow accounts for one bad row.

    The row must take a buffering path (every fork path but the <=32-row
    cooperative radix one), its threshold bin (the bin of the reference k-th
    value) must hold more candidates than that path buffers, and the kernel must
    still have selected every element strictly above that bin and nothing below
    it -- only the threshold-bin tie-break may differ.
    """
    length = int(row_logits.numel())
    path = _overflow_buffer(num_rows, length, int(actual_indices.numel()))
    if path is None:
        return False
    shift, buffered_items = path
    bins = _fp16_key_bin(torch, row_logits, shift)
    threshold = int(_fp16_key_bin(torch, expected_values.min().reshape(1), shift)[0])
    if int((bins == threshold).sum()) <= buffered_items:
        return False
    selected_bins = bins.index_select(0, actual_indices)
    if bool((selected_bins < threshold).any()):
        return False
    return int((selected_bins > threshold).sum()) == int((bins > threshold).sum())


def _validate_native_semantics(
    torch: Any,
    op: Any,
    operands: _DsaPersistentTopkDecodeNativeOperands,
    *,
    top_k: int,
    max_seq_len: int,
    strict_reference: bool = True,
    medium_overflow_allowed: bool = False,
) -> None:
    """Compare corrected native output with the committed semantic reference.

    ``medium_overflow_allowed`` accepts a mismatching row only when the fork's
    medium-path buffer overflow fully explains it (see ``_FORK_MEDIUM_*``).
    """
    from profiling.runners.attention.dsa_persistent_topk_decode_reference import (
        dsa_persistent_topk_decode_reference,
    )

    logits_before = operands.logits_backing.clone()
    lengths_before = operands.lengths.clone()
    expected = torch.empty_like(operands.out)
    out_storage = operands.out.untyped_storage().data_ptr()
    workspace_storage = operands.workspace.untyped_storage().data_ptr()
    returned = _native_call(
        op,
        operands,
        top_k=top_k,
        max_seq_len=max_seq_len,
    )
    dsa_persistent_topk_decode_reference(
        operands.logits,
        operands.lengths,
        expected,
        top_k=top_k,
        max_seq_len=max_seq_len,
    )

    short_rows = torch.nonzero(
        operands.flat_lengths <= top_k,
        as_tuple=False,
    ).flatten()
    if short_rows.numel() and not torch.equal(
        operands.out.index_select(0, short_rows),
        expected.index_select(0, short_rows),
    ):
        raise RuntimeError("pinned CUDA disagrees with natural/-1 reference rows")

    long_rows = torch.nonzero(operands.flat_lengths > top_k, as_tuple=False).flatten()
    if long_rows.numel():
        actual_long = operands.out.index_select(0, long_rows).to(torch.int64)
        expected_long = expected.index_select(0, long_rows).to(torch.int64)
        if bool((actual_long < 0).any()) or bool(
            (actual_long >= operands.flat_lengths.index_select(0, long_rows).unsqueeze(1)).any()
        ):
            raise RuntimeError("persistent CUDA returned an out-of-range long-row index")
        if bool((actual_long.sort(dim=1).values.diff(dim=1) == 0).any()):
            raise RuntimeError("persistent CUDA returned duplicate long-row indices")
        long_logits = operands.logits.index_select(0, long_rows)
        actual_values = long_logits.gather(1, actual_long)
        expected_values = long_logits.gather(1, expected_long)
        if strict_reference:
            selection_rows = (
                actual_long.sort(dim=1).values != expected_long.sort(dim=1).values
            ).any(dim=1)
            value_rows = (
                actual_values.sort(dim=1).values != expected_values.sort(dim=1).values
            ).any(dim=1)
            for position in torch.nonzero(selection_rows | value_rows).flatten().tolist():
                length = int(operands.flat_lengths[long_rows[position]])
                if not (
                    medium_overflow_allowed
                    and _medium_overflow_explains(
                        torch,
                        long_logits[position, :length],
                        actual_long[position],
                        expected_values[position],
                        num_rows=int(operands.flat_lengths.numel()),
                    )
                ):
                    kind = "values" if bool(value_rows[position]) else "selections"
                    raise RuntimeError(f"pinned CUDA disagrees with long-row reference {kind}")

    if returned is not None:
        raise RuntimeError("pinned persistent_topk must return None")
    if operands.out.untyped_storage().data_ptr() != out_storage:
        raise RuntimeError("pinned CUDA did not preserve caller-owned output storage")
    if operands.workspace.untyped_storage().data_ptr() != workspace_storage:
        raise RuntimeError("pinned CUDA did not preserve caller-owned workspace storage")
    if not torch.equal(operands.logits_backing, logits_before):
        raise RuntimeError("pinned CUDA mutated logits")
    if not torch.equal(operands.lengths, lengths_before):
        raise RuntimeError("pinned CUDA mutated lengths")
    for tensor in (operands.logits, operands.lengths, operands.workspace):
        if torch._C._overlaps(operands.out, tensor):
            raise RuntimeError("pinned CUDA output aliases another operand")


def _logical_bytes(
    *,
    batch_size: int,
    context_len: int,
    next_n: int,
    top_k: int,
) -> int:
    """Return valid-logit, length-metadata, and output logical traffic."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if next_n <= 0:
        raise ValueError("next_n must be > 0")
    if context_len < next_n - 1:
        raise ValueError("context_len must be >= next_n - 1")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    num_rows = batch_size * next_n
    per_request_sum = next_n * (2 * context_len - next_n + 1) // 2
    return 4 * batch_size * per_request_sum + 4 * num_rows + 4 * num_rows * top_k


def profile_dsa_persistent_topk_decode_torch(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    logits_row_stride: int,
    logits_dtype: DType | str,
    index_dtype: str,
    context_mode: str,
) -> ComputeMetrics:
    """Profile the complete Torch persistent decode top-k semantic composite."""
    (
        batch_size,
        context_len,
        next_n,
        max_model_len,
        top_k,
        logits_row_stride,
        _logits_dtype,
        _index_dtype,
        _context_mode,
    ) = _validate_args(
        batch_size,
        context_len,
        next_n,
        max_model_len,
        top_k,
        logits_row_stride,
        logits_dtype,
        index_dtype,
        context_mode,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the torch dsa_persistent_topk_decode backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_operands(
            torch,
            batch_size=batch_size,
            context_len=context_len,
            next_n=next_n,
            max_model_len=max_model_len,
            top_k=top_k,
            logits_row_stride=logits_row_stride,
            device="cuda",
        )
        _validate_semantics(
            torch,
            operands,
            top_k=top_k,
            max_seq_len=context_len,
        )

        def kernel() -> Any:
            return _torch_composite(operands)

        time_ms = Timer.cuda_event(kernel, warmup=5)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        # Logical traffic counts valid FP32 logits, one int32 length per row,
        # and all int32 output slots. It excludes padded columns, persistent
        # scratch workspace, Torch intermediates, and physical transactions.
        logical_bytes = _logical_bytes(
            batch_size=batch_size,
            context_len=context_len,
            next_n=next_n,
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


def _load_native_op(torch: Any) -> Any:
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name == "NVIDIA B200":
        try:
            from vllm import _custom_ops  # noqa: F401
        except ImportError as exc:
            raise ProfilerNotImplemented(
                "the instrumented vLLM extension is required for B200 persistent top-k"
            ) from exc
        return torch.ops._C.persistent_topk

    from profiling.runners.attention.dsa_persistent_topk_native import (
        NativeExtensionBuildError,
        NativeExtensionLoadError,
        NativeExtensionUnsupported,
        load_persistent_topk_op,
    )

    try:
        return load_persistent_topk_op(torch)
    except NativeExtensionUnsupported as exc:
        raise ProfilerNotImplemented(str(exc)) from exc
    except (NativeExtensionBuildError, NativeExtensionLoadError) as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def _profile_native_persistent_topk(
    backend: str,
    allowed_top_k: frozenset[int],
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    logits_row_stride: int,
    logits_dtype: DType | str,
    index_dtype: str,
    context_mode: str,
) -> ComputeMetrics:
    """Profile one packaged persistent_topk callable through its full contract."""
    (
        batch_size,
        context_len,
        next_n,
        max_model_len,
        top_k,
        logits_row_stride,
        _logits_dtype,
        _index_dtype,
        _context_mode,
    ) = _validate_args(
        batch_size,
        context_len,
        next_n,
        max_model_len,
        top_k,
        logits_row_stride,
        logits_dtype,
        index_dtype,
        context_mode,
        allowed_top_k=allowed_top_k,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"torch is required for the {backend} dsa_persistent_topk_decode backend"
        ) from exc

    gpu_name = _validate_cuda_device(torch, backend=backend)
    op = _load_native_op(torch)

    try:
        operands = _build_native_operands(
            torch,
            batch_size=batch_size,
            context_len=context_len,
            next_n=next_n,
            max_model_len=max_model_len,
            top_k=top_k,
            logits_row_stride=logits_row_stride,
            device="cuda",
        )
        _validate_native_semantics(
            torch,
            op,
            operands,
            top_k=top_k,
            max_seq_len=context_len,
            strict_reference=backend == _FORK_BACKEND or gpu_name != "NVIDIA B200",
            medium_overflow_allowed=backend == _FORK_BACKEND,
        )

        def kernel() -> None:
            return _native_call(
                op,
                operands,
                top_k=top_k,
                max_seq_len=context_len,
            )

        # Correctness above is the exact-shape first launch. Synchronize before
        # formal timing so source verification, build/load, and setup are absent.
        torch.cuda.synchronize()
        # CUPTI, not CUDA events. This was the only production backend in the
        # table still on `Timer.cuda_event`; the other 97 runner call sites all
        # price kernel-only time, and summing a launch-inclusive row into a cost
        # tree with kernel-only rows charges this kernel's dispatch twice.
        #
        # Measured over 80 stride-sampled specs (job 613): the event timer reads
        # 1.03x to 3.80x the CUPTI time, median 1.24x, and never less. These
        # kernels are 2-22 us, so a ~5 us launch gap is most of the difference,
        # and the ratio is largest on the smallest shapes. Taking the
        # measurement also cost 1.5 s/spec under events (a flat
        # 3 x DEFAULT_MIN_DURATION_MS, no drift probe) against 0.120 s/spec
        # under CUPTI's adaptive budget -- 670 s of a full fill's 3151 GPU-s.
        time_ms = Timer.cupti(kernel, warmup=5)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        # Logical traffic counts valid FP32 logits, one int32 length per row,
        # and all int32 output slots. It excludes the 1 MiB scratch workspace,
        # padded logits, intermediates, and physical memory transactions.
        logical_bytes = _logical_bytes(
            batch_size=batch_size,
            context_len=context_len,
            next_n=next_n,
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


def profile_dsa_persistent_topk_decode_vllm_cuda(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    logits_row_stride: int,
    logits_dtype: DType | str,
    index_dtype: str,
    context_mode: str,
) -> ComputeMetrics:
    """Profile the complete corrected v0.23-derived persistent callable."""
    return _profile_native_persistent_topk(
        "vllm_cuda",
        frozenset({_TOP_K}),
        batch_size,
        context_len,
        next_n,
        max_model_len,
        top_k,
        logits_row_stride,
        logits_dtype,
        index_dtype,
        context_mode,
    )


def profile_dsa_persistent_topk_decode_vllm_fork_cuda(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    logits_row_stride: int,
    logits_dtype: DType | str,
    index_dtype: str,
    context_mode: str,
) -> ComputeMetrics:
    """Profile the vLLM fork's persistent_topk (GLM-5.3-Flash kpool K=512).

    For kpool, ``context_len`` is the row's pool count (floor(tokens / 4)) and
    ``max_model_len`` the logits width, which vLLM keeps token-granular. The
    call passes ``max_seq_len = context_len``; production passes the batch's
    token max_seq_len instead. The scalar only gates the cooperative radix
    setup (``> 32768``) on the <=32-row path, so the two agree whenever the
    token context is <= 32768 or the batch has more than 32 rows.
    """
    return _profile_native_persistent_topk(
        _FORK_BACKEND,
        _FORK_TOP_K,
        batch_size,
        context_len,
        next_n,
        max_model_len,
        top_k,
        logits_row_stride,
        logits_dtype,
        index_dtype,
        context_mode,
    )
