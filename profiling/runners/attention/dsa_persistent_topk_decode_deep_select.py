"""Profile DeepSelect's exact FP32 top-k on the decode indexer's operands.

The alignment fork offers it as an explicit decode top-k backend
(``kernel_config.sparse_indexer_topk_backend = "deep_select"``,
``model_executor/layers/indexer_topk.py:290-292``): ``torch.ops.deep_select.topk``
over each row's ``[0, end)`` prefix, SM100a/SM103a only. It reuses the
persistent runner's operands; each row's end is its valid length. DeepSelect
writes no defined order, so a row checks as a set: a short row (no longer than
``top_k``) holds exactly its own positions then ``-1``, a long row the value
multiset of the reference selection (ties have no defined winner).
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.dsa_persistent_topk_decode import (
    build_native_operands,
    logical_bytes,
    validate_args,
)
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "dsa_persistent_topk_decode:deep_select"
# vLLM indexer_topk.py IDX_OOB_FILL_VALUE: the fill past a short row's end.
_OOB_FILL = -1


def _call(torch: Any, operands: Any, top_k: int) -> None:
    torch.ops.deep_select.topk(
        operands.logits,
        top_k,
        None,  # begin: not supported
        operands.flat_lengths,
        False,  # sorted_value
        False,  # sorted_index
        None,  # output_value
        operands.out,
        None,  # output_idx_offset
        _OOB_FILL,
        float("-inf"),
        False,  # return_value
        False,  # abort_when_nan_found
    )


def _validate(torch: Any, operands: Any, top_k: int) -> None:
    from profiling.runners.attention.dsa_persistent_topk_decode_reference import (
        dsa_persistent_topk_decode_reference,
    )

    logits_before = operands.logits_backing.clone()
    _call(torch, operands, top_k)
    expected = torch.empty_like(operands.out)
    dsa_persistent_topk_decode_reference(
        operands.logits,
        operands.lengths,
        expected,
        top_k=top_k,
        max_seq_len=int(operands.flat_lengths.max()),
    )
    actual = operands.out.to(torch.int64)
    lengths = operands.flat_lengths.to(torch.int64)
    short = lengths <= top_k
    if bool(short.any()):
        positions = torch.arange(top_k, device=actual.device)
        want = torch.where(positions < lengths[short].unsqueeze(1), positions, _OOB_FILL)
        if not torch.equal(actual[short].sort(1).values, want.sort(1).values):
            raise RuntimeError("deep_select disagrees with natural/-1 short rows")
    long = ~short
    if bool(long.any()):
        got = actual[long]
        if bool((got < 0).any()) or bool((got >= lengths[long].unsqueeze(1)).any()):
            raise RuntimeError("deep_select returned an out-of-range long-row index")
        if bool((got.sort(1).values.diff(dim=1) == 0).any()):
            raise RuntimeError("deep_select returned duplicate long-row indices")
        rows = operands.logits[long]
        values = rows.gather(1, got).sort(1).values
        reference = rows.gather(1, expected[long].to(torch.int64)).sort(1).values
        if not torch.equal(values, reference):
            raise RuntimeError("deep_select disagrees with long-row reference values")
    if not torch.equal(operands.logits_backing, logits_before):
        raise RuntimeError("deep_select mutated logits")


def profile_dsa_persistent_topk_decode_deep_select(
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
    (
        batch_size,
        context_len,
        next_n,
        max_model_len,
        top_k,
        logits_row_stride,
        _logits_dtype,
        _index_dtype,
        context_mode,
    ) = validate_args(
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
        import vllm._deepselect_C  # noqa: F401  registers torch.ops.deep_select
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires vLLM's DeepSelect extension") from exc
    if top_k > 4096 or max_model_len >= 2**23:
        raise ProfilerNotImplemented(f"{_BACKEND} takes top_k <= 4096 and rows below 2^23")
    in_align, _out_align = torch.ops.deep_select.get_alignment_requirement()
    if logits_row_stride * 4 % in_align:
        raise ProfilerNotImplemented(f"{_BACKEND} needs a {in_align} B-aligned logits row stride")
    try:
        operands = build_native_operands(
            torch,
            batch_size=batch_size,
            context_len=context_len,
            next_n=next_n,
            max_model_len=max_model_len,
            top_k=top_k,
            logits_row_stride=logits_row_stride,
            device="cuda",
            context_mode=context_mode,
        )
        _validate(torch, operands, top_k)

        def kernel() -> None:
            _call(torch, operands, top_k)

        torch.cuda.synchronize()
        time_ms = Timer.cupti(kernel, warmup=5)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        num_bytes = logical_bytes(
            batch_size=batch_size, context_len=context_len, next_n=next_n, top_k=top_k
        )
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(num_bytes / (time_ms / 1000.0) / 1e9),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
