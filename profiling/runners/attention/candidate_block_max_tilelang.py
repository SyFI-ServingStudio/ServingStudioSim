"""DeepInfer's candidate block scores -- ``tilelang`` backend of ``candidate_block_max``.

The timed call is the one ``MQA._topk_candidate_block`` issues on the
candidate source layer (``v4_modules/index_attn.py``)::

    candidate_block_max(p_sum, k_len, block_size)

from ``deepinfer/model_executor/layers/tilelang_ops/candidate_blocks.py``:
two launches (``_candidate_block_max_kernel`` builds the task list,
``_candidate_block_max_kernel_1`` reduces each row's bf16 logits to one score
per ``block_size`` keys, the last valid block forced to +inf). ``p_sum`` is
the source layer's dense logits chunk: one request's tail rows, bf16, rows
padded to 1024 bytes as DeepGEMM allocates them; ``k_len`` is each row's
window. The top-k over the block scores is the separate ``deep_select_topk``
kind.

Check: the scores and valid-block counts against DeepInfer's own
``candidate_block_max_ref`` on four rows.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention import deepinfer_indexer_common as common
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "tilelang"


def validate(
    num_queries: int, context_len: int, compress_ratio: int, block_size: int, dtype: DType | str
) -> tuple[int, int, int, int]:
    if DType.from_value(dtype) is not DType.BF16:
        raise ValueError(f"{BACKEND} candidate_block_max takes dtype=bf16, got {dtype!r}")
    values = int(num_queries), int(context_len), int(compress_ratio), int(block_size)
    if min(values) < 1 or values[0] > values[1]:
        raise ValueError(f"need positive sizes and num_queries <= context_len, got {values}")
    if values[3] not in (8, 16):
        raise ValueError(f"block_size must be 8 or 16, got {values[3]}")
    return values


def profile_candidate_block_max_tilelang(
    num_queries: int, context_len: int, compress_ratio: int, block_size: int, dtype: DType | str
) -> ComputeMetrics:
    num_queries, context_len, ratio, block_size = validate(
        num_queries, context_len, compress_ratio, block_size, dtype
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    try:
        from deepinfer.model_executor.layers.tilelang_ops.candidate_blocks import (
            candidate_block_max,
            candidate_block_max_ref,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{BACKEND} needs DeepInfer's TileLang ops (deepinfer_env)") from exc
    try:
        torch.manual_seed(0)
        num_keys = common.num_request_keys(context_len, ratio)
        stride = -(-num_keys // 512) * 512  # DeepGEMM pads logits rows to 1024 bytes.
        logits = torch.randn(num_queries, stride, device="cuda", dtype=torch.bfloat16)[:, :num_keys]
        _, k_len = common.build_tail_spans(torch, num_queries, context_len, ratio)

        def run_once() -> Any:
            return candidate_block_max(logits, k_len, block_size)

        scores, valid_blocks = run_once()  # TileLang JIT, outside timing.
        torch.cuda.synchronize()
        rows = sorted({0, num_queries // 2, num_queries - 1})
        idx = torch.tensor(rows, device="cuda")
        want_scores, want_blocks = candidate_block_max_ref(logits[idx], k_len[idx], block_size)
        if not torch.equal(valid_blocks[idx], want_blocks):
            raise KernelLaunchFailed(f"{BACKEND} candidate_block_max valid-block counts differ")
        cols = torch.arange(want_scores.shape[1], device="cuda")
        mask = cols.unsqueeze(0) < want_blocks.unsqueeze(1)
        got = scores[idx, : want_scores.shape[1]][mask]
        if not torch.equal(got.view(torch.int16), want_scores[mask].view(torch.int16)):
            raise KernelLaunchFailed(f"{BACKEND} candidate_block_max scores differ from the reference")

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} candidate_block_max ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} candidate_block_max failed: {exc}") from exc

    keys = sum(common.tail_row_keys(num_queries, context_len, ratio))
    bytes_accessed = 2 * keys + 2 * keys // block_size + 8 * num_queries
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["profile_candidate_block_max_tilelang", "validate"]
