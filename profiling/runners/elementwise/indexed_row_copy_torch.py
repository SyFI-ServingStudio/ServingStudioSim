"""Eager PyTorch runner for ``indexed_row_copy``.

The timed statements are the ones vLLM writes around a recurrent-state prefill
(``qwen_gdn_linear_attn.py``): ``pool[indices]``, ``rows[~mask, ...] = 0`` and
``pool[indices] = rows``, plus the token-row split and merge of a step that mixes
speculative and plain rows: ``x.index_select(0, indices)`` and
``out.index_copy_(0, indices, rows)``. Torch is the production callable here, so
the runner times it directly and checks the result against plain per-row copies
outside the capture. Every launch of a statement is counted.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "indexed_row_copy:torch"
_OPERATIONS = ("gather", "scatter", "masked_zero", "index_select", "index_copy")


def _validate(num_rows: int, row_elements: int, dtype: DType | str, operation: str):
    num_rows = int(num_rows)
    row_elements = int(row_elements)
    if num_rows <= 0 or row_elements <= 0:
        raise ValueError(f"num_rows and row_elements must be > 0, got {num_rows}, {row_elements}")
    if operation not in _OPERATIONS:
        raise ValueError(f"operation must be one of {_OPERATIONS}, got {operation!r}")
    return num_rows, row_elements, DType.from_value(dtype), operation


def slot_indices(num_rows: int) -> list[int]:
    """Every other slot after a reserved slot 0, so rows are not adjacent in the pool."""
    return list(range(2, 2 * num_rows + 2, 2))


def token_indices(num_rows: int) -> list[int]:
    """Contiguous token rows after one leading row, standing in for the sorted
    int64 token indices a mixed step selects (``argsort`` of the spec-token
    mask), which need not be contiguous."""
    return list(range(1, num_rows + 1))


def logical_bytes(num_rows: int, row_elements: int, dtype: DType, operation: str) -> int:
    row_bytes = int(num_rows * row_elements * dtype.size_bytes())
    return row_bytes if operation == "masked_zero" else 2 * row_bytes


def _build(
    torch: Any, num_rows: int, row_elements: int, dtype: DType, operation: str, *, device: Any
):
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    torch_dtype = dtype.torch()
    if operation in ("index_select", "index_copy"):
        return _build_token_rows(
            torch, num_rows, row_elements, torch_dtype, operation, generator, device
        )
    indices = torch.tensor(slot_indices(num_rows), dtype=torch.int32, device=device)
    if operation == "masked_zero":
        rows = torch.empty(num_rows, row_elements, dtype=torch_dtype, device=device)
        rows.normal_(generator=generator)
        has_prior_state = torch.zeros(num_rows, dtype=torch.bool, device=device)

        def run() -> None:
            rows[~has_prior_state, ...] = 0

        def check() -> None:
            if bool(rows.any()):
                raise AssertionError("masked_zero left a nonzero element")

        return run, check

    pool = torch.empty(2 * num_rows + 2, row_elements, dtype=torch_dtype, device=device)
    pool.normal_(generator=generator)
    if operation == "gather":
        out: list[Any] = []

        def run() -> None:
            out[:] = [pool[indices]]

        def check() -> None:
            expected = torch.stack([pool[i] for i in slot_indices(num_rows)])
            if not torch.equal(out[0], expected):
                raise AssertionError("gather returned the wrong rows")

        return run, check

    rows = torch.empty(num_rows, row_elements, dtype=torch_dtype, device=device)
    rows.normal_(generator=generator)

    def run() -> None:
        pool[indices] = rows

    def check() -> None:
        for row, slot in enumerate(slot_indices(num_rows)):
            if not torch.equal(pool[slot], rows[row]):
                raise AssertionError("scatter wrote the wrong slot")

    return run, check


def _build_token_rows(
    torch: Any,
    num_rows: int,
    row_elements: int,
    torch_dtype: Any,
    operation: str,
    generator: Any,
    device: Any,
):
    tokens = torch.tensor(token_indices(num_rows), dtype=torch.int64, device=device)
    if operation == "index_select":
        source = torch.empty(num_rows + 1, row_elements, dtype=torch_dtype, device=device)
        source.normal_(generator=generator)
        out: list[Any] = []

        def run() -> None:
            out[:] = [source.index_select(0, tokens)]

        def check() -> None:
            if not torch.equal(out[0], source[1:]):
                raise AssertionError("index_select returned the wrong rows")

        return run, check

    destination = torch.zeros(num_rows + 1, row_elements, dtype=torch_dtype, device=device)
    rows = torch.empty(num_rows, row_elements, dtype=torch_dtype, device=device)
    rows.normal_(generator=generator)

    def run() -> None:
        destination.index_copy_(0, tokens, rows)

    def check() -> None:
        if not torch.equal(destination[1:], rows):
            raise AssertionError("index_copy_ wrote the wrong rows")

    return run, check


def profile_indexed_row_copy(
    num_rows: int, row_elements: int, dtype: DType | str, operation: str
) -> ComputeMetrics:
    num_rows, row_elements, dtype, operation = _validate(num_rows, row_elements, dtype, operation)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc

    try:
        device = torch.device("cuda", torch.cuda.current_device())
        run, check = _build(torch, num_rows, row_elements, dtype, operation, device=device)
        run()
        torch.cuda.synchronize()
        check()
        time_ms = Timer.cupti(run, kernel_name=None)
        energy_j = Energy.perf(run, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    elapsed_s = time_ms / 1000.0
    nbytes = logical_bytes(num_rows, row_elements, dtype, operation)
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=0.0,
        memory_bandwidth_gbps=nbytes / elapsed_s / 1e9 if time_ms > 0.0 else 0.0,
    )
