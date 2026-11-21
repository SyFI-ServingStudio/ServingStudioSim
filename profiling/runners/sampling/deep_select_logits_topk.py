"""DeepInfer's sampler top-k -- ``deep_select`` backend of ``deep_select_logits_topk``.

The timed call is ``deep_select.interface.topk(logits, top_k, sorted,
indices_type=torch.int32)`` with values returned, as
``sampler_utils.process_sampling`` (force_topk, sorted=False) and
``Sampler._prepare_top_info`` (top logprobs, sorted=True) issue it: one
``topk_select_fp32::topk_kernel`` launch over contiguous fp32 rows, seeded
standard-normal. DeepSelect needs 1024-byte row strides; rows are
padded to that.

Check: on four rows the selected values equal ``torch.topk``'s (as a set when
unsorted, in order when sorted) and the indices point at them.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "deep_select"
ROW_ALIGN_ELEMS = 256  # DeepSelect needs 1024-byte input row strides


def validate(
    num_rows: int, num_columns: int, top_k: int, sorted: bool, dtype: DType | str
) -> tuple[int, int, int, bool]:
    if DType.from_value(dtype) is not DType.FP32:
        raise ValueError(f"{BACKEND} deep_select_logits_topk takes dtype=fp32, got {dtype!r}")
    values = int(num_rows), int(num_columns), int(top_k)
    if min(values) < 1:
        raise ValueError(f"deep_select_logits_topk needs positive sizes, got {values}")
    if values[2] > values[1]:
        raise ValueError(f"top_k {values[2]} exceeds the row width {values[1]}")
    return (*values, bool(sorted))


def profile_deep_select_logits_topk(
    num_rows: int, num_columns: int, top_k: int, sorted: bool, dtype: DType | str
) -> ComputeMetrics:
    num_rows, num_columns, top_k, is_sorted = validate(num_rows, num_columns, top_k, sorted, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    try:
        from deep_select.interface import topk
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{BACKEND} needs DeepInfer's DeepSelect (deepinfer_env)"
        ) from exc
    try:
        g = torch.Generator(device="cuda").manual_seed(0)
        stride = -(-num_columns // ROW_ALIGN_ELEMS) * ROW_ALIGN_ELEMS
        logits = torch.randn(num_rows, stride, device="cuda", dtype=torch.float32, generator=g)[
            :, :num_columns
        ]

        def run_once() -> Any:
            return topk(logits, top_k, is_sorted, indices_type=torch.int32)

        values, indices = run_once()  # module load, outside timing.
        torch.cuda.synchronize()
        for row in sorted_rows(num_rows):
            want = logits[row].topk(top_k).values
            got = values[row]
            if not torch.equal(logits[row][indices[row].long()], got):
                raise KernelLaunchFailed(f"{BACKEND} row {row}: indices do not point at the values")
            if not is_sorted:
                got = got.sort(descending=True).values
            if not torch.equal(got, want):
                raise KernelLaunchFailed(f"{BACKEND} row {row}: selection differs from torch.topk")

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} deep_select_logits_topk ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} deep_select_logits_topk failed: {exc}") from exc

    bytes_accessed = num_rows * (4 * num_columns + 8 * top_k)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


def sorted_rows(num_rows: int) -> list[int]:
    return sorted({0, num_rows // 3, 2 * num_rows // 3, num_rows - 1})


__all__ = ["profile_deep_select_logits_topk", "validate"]
