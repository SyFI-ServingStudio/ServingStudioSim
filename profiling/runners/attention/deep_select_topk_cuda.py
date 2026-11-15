"""DeepInfer's indexer top-k -- ``deep_select`` backend of ``deep_select_topk``.

The timed call is DeepSelect's ``deep_select.interface.topk`` as DeepInfer's
prefill indexer issues it (``v4_modules/index_attn.py``)::

    topk(scores, top_k, sorted=False, end=row_ends, indices_type=torch.int32,
         sorted_index=True, output_idx=out, output_idx_offset=offsets,
         idx_oob_fill_value=-1, return_value=False)

over bf16 rows: one ``topk_select_bf16_normal::topk_kernel`` launch, one CTA
per row; DeepSelect picks the CTA shape from the row count against the SM
count (one wave or more) and ``top_k``, and its cluster kernel for at most six
rows of at least 512K columns (``csrc/api.cpp``). Three call sites share it:
the dense logits rows (``_topk_prefill``, top 512, with the row offsets), the
candidate source layer's block scores (``_topk_candidate_block``, top 2048)
and the candidate-restricted logits (top 512). A profiled call has
``num_rows`` rows that each hold ``row_len`` valid scores, in rows padded to
1024 bytes (DeepGEMM's logits stride); scores are seeded standard-normal bf16.

Check: on four rows the selected scores equal ``torch.topk``'s (ties make the
indices themselves ambiguous) and the indices come back ascending.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "deep_select"
ROW_ALIGN_ELEMS = 512  # 1024-byte bf16 rows


def validate(num_rows: int, row_len: int, top_k: int, dtype: DType | str) -> tuple[int, int, int]:
    if DType.from_value(dtype) is not DType.BF16:
        raise ValueError(f"{BACKEND} deep_select_topk takes dtype=bf16, got {dtype!r}")
    values = int(num_rows), int(row_len), int(top_k)
    if min(values) < 1:
        raise ValueError(f"deep_select_topk needs positive sizes, got {values}")
    if values[2] > 4096:
        raise ValueError(f"DeepSelect's bf16 top-k takes top_k <= 4096, got {values[2]}")
    if values[1] >= 1 << 23:
        raise ValueError(f"DeepSelect's bf16 top-k takes rows shorter than 2^23, got {values[1]}")
    return values


def profile_deep_select_topk_cuda(num_rows: int, row_len: int, top_k: int, dtype: DType | str) -> ComputeMetrics:
    num_rows, row_len, top_k = validate(num_rows, row_len, top_k, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    try:
        from deep_select.interface import topk
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{BACKEND} needs DeepInfer's DeepSelect (deepinfer_env)") from exc
    try:
        torch.manual_seed(0)
        stride = -(-row_len // ROW_ALIGN_ELEMS) * ROW_ALIGN_ELEMS
        scores = torch.randn(num_rows, stride, device="cuda", dtype=torch.bfloat16)[:, :row_len]
        ends = torch.full((num_rows,), row_len, device="cuda", dtype=torch.int32)
        offsets = torch.zeros(num_rows, device="cuda", dtype=torch.int32)
        out = torch.empty(num_rows, top_k, device="cuda", dtype=torch.int32)

        def run_once() -> Any:
            return topk(
                scores,
                top_k,
                sorted=False,
                end=ends,
                indices_type=torch.int32,
                sorted_index=True,
                output_idx=out,
                output_idx_offset=offsets,
                idx_oob_fill_value=-1,
                return_value=False,
            )

        run_once()  # module load, outside timing.
        torch.cuda.synchronize()
        take = min(top_k, row_len)
        for row in sorted({0, num_rows // 3, 2 * num_rows // 3, num_rows - 1}):
            idx = out[row, :take].long()
            if take > 1 and not bool((idx[1:] > idx[:-1]).all()):
                raise KernelLaunchFailed(f"{BACKEND} row {row}: indices not ascending")
            got = scores[row][idx].float().sort(descending=True).values
            want = scores[row].float().topk(take).values
            if not torch.equal(got, want):
                raise KernelLaunchFailed(f"{BACKEND} row {row}: selection differs from torch.topk")
            if take < top_k and not bool((out[row, take:] == -1).all()):
                raise KernelLaunchFailed(f"{BACKEND} row {row}: short row not padded with -1")

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} deep_select_topk ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} deep_select_topk failed: {exc}") from exc

    bytes_accessed = num_rows * (2 * row_len + 4 * top_k + 8)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["profile_deep_select_topk_cuda", "validate"]
