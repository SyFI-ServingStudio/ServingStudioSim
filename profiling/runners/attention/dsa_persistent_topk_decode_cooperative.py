"""Profile vLLM's ``cooperative_topk``, the small-batch decode top-k path.

Indexers that dispatch like the DeepSeek indexer (vLLM
``model_executor/layers/indexer_topk.py`` ``_resolve_auto``; the Qwen4Exp QSA
indexer ``models/qwen4_exp/nvidia/ops/qsa_indexer.py:471-499``) call
``torch.ops._C.cooperative_topk`` for at most 64 rows whose logits row stride is
a multiple of four, on SM90+ outside the SM12x family, and ``persistent_topk``
otherwise. Both take the same operands, so this backend reuses the persistent
runner's operands and its reference check; only the op differs.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.dsa_persistent_topk_decode import (
    _build_native_operands,
    _logical_bytes,
    _native_call,
    _validate_args,
    _validate_native_semantics,
)
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "dsa_persistent_topk_decode:vllm_cuda_cooperative"
# csrc/libtorch_stable/cooperative_topk.cu:63 rejects more rows.
MAX_ROWS = 64


def profile_dsa_persistent_topk_decode_vllm_cuda_cooperative(
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
    if batch_size * next_n > MAX_ROWS:
        raise ProfilerNotImplemented(
            f"{_BACKEND} runs at most {MAX_ROWS} rows, got {batch_size * next_n}"
        )
    if logits_row_stride % 4:
        raise ProfilerNotImplemented(f"{_BACKEND} needs a logits row stride divisible by 4")
    try:
        import torch
        from vllm import _custom_ops  # noqa: F401  # registers torch.ops._C
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM build") from exc

    op = getattr(torch.ops._C, "cooperative_topk", None)
    if op is None:
        raise ProfilerNotImplemented(f"{_BACKEND}: torch.ops._C.cooperative_topk is unavailable")
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
            context_mode=context_mode,
        )
        _validate_native_semantics(torch, op, operands, top_k=top_k, max_seq_len=context_len)

        def kernel() -> None:
            return _native_call(op, operands, top_k=top_k, max_seq_len=context_len)

        torch.cuda.synchronize()
        time_ms = Timer.cupti(kernel, warmup=5)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        logical_bytes = _logical_bytes(
            batch_size=batch_size, context_len=context_len, next_n=next_n, top_k=top_k
        )
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(logical_bytes / (time_ms / 1000.0) / 1e9),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


__all__ = ["profile_dsa_persistent_topk_decode_vllm_cuda_cooperative"]
