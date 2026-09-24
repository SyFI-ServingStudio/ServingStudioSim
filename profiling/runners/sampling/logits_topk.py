"""Row-wise top-k over a dense score matrix: flashinfer and torch backends.

Both entry points time exactly what vLLM's `logits_processor._topk` runs —
`flashinfer.top_k(scores, k, sorted=True, deterministic=True)` when flashinfer
imports, `torch.topk(scores, k, dim=-1)` otherwise. Neither is a rewrite: the
runner only allocates the operand and calls the public function.

`flashinfer.top_k` is a fixed multi-launch sequence, not a single kernel: the
selection pass (`RadixTopKKernel_Unified` for a wide matrix,
`FilteredTopKUnifiedKernel` + `FinalizeTopKIndicesKernel` for a narrow one)
plus the on-device `StableSortTopKByValueKernel` that `sorted=True` asks for.
CUPTI therefore counts every launch (`kernel_name=None`); filtering to the
selection kernel would drop about a quarter of the call's real GPU time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_SEED = 0


@dataclass(frozen=True)
class _ValidatedArgs:
    num_rows: int
    num_columns: int
    top_k: int
    dtype: DType


def _exact_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an exact integer, got {value!r}")
    return value


def _validate_args(
    num_rows: int,
    num_columns: int,
    top_k: int,
    dtype: DType | str,
) -> _ValidatedArgs:
    num_rows = _exact_int("num_rows", num_rows)
    num_columns = _exact_int("num_columns", num_columns)
    top_k = _exact_int("top_k", top_k)
    if num_rows <= 0 or num_columns <= 0:
        raise ValueError("num_rows and num_columns must be positive")
    if not 1 <= top_k <= num_columns:
        raise ValueError("top_k must satisfy 1 <= top_k <= num_columns")
    dtype = DType.from_value(dtype)
    if dtype not in (DType.BF16, DType.FP16, DType.FP32):
        raise ValueError(f"logits_topk requires a float score dtype, got {dtype.value}")
    return _ValidatedArgs(num_rows, num_columns, top_k, dtype)


def _build_scores(torch: Any, args: _ValidatedArgs) -> Any:
    """Distinct scores per row, in the shape a real logits row has.

    Random normal values are what an LM head emits and what flashinfer's own
    correctness tests use. Distinctness matters: a radix select branches on the
    bucket population at the k-th boundary, so an operand built from a few
    repeated values would exercise a tie-break path production almost never
    reaches. The seed is fixed so a row's population is the same on every host.
    """
    generator = torch.Generator(device="cuda").manual_seed(_SEED)
    scores = torch.randn(
        args.num_rows,
        args.num_columns,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    return scores.to(args.dtype.torch()).contiguous()


def _check_against_torch(torch: Any, scores: Any, top_k: int, values: Any, indices: Any) -> None:
    """flashinfer's selected set must be torch.topk's selected set.

    Compares values, not indices: equal scores may be selected through either
    index, and the DFlash2 selector consumes the ids only after the values have
    ranked them. A genuine selection bug moves a value, so the value comparison
    still catches it.
    """
    expected_values, _ = torch.topk(scores.float(), top_k, dim=-1, sorted=True)
    actual_values = values.float()
    if actual_values.shape != expected_values.shape:
        raise RuntimeError(
            f"flashinfer top_k returned {tuple(actual_values.shape)}, "
            f"expected {tuple(expected_values.shape)}"
        )
    if indices.shape != expected_values.shape:
        raise RuntimeError(
            f"flashinfer top_k returned indices {tuple(indices.shape)}, "
            f"expected {tuple(expected_values.shape)}"
        )
    if not torch.equal(actual_values, expected_values):
        raise RuntimeError("flashinfer top_k selected a different set than torch.topk")


def _semantic_flops(args: _ValidatedArgs) -> int:
    """One comparison per element per row, the lower bound any selection pays.

    A radix select sweeps the row a small constant number of times rather than
    once, so this is a nominal semantic count, not the physical work.
    """
    return args.num_rows * args.num_columns


def _logical_bytes(args: _ValidatedArgs) -> int:
    """Read the score matrix once, write k values plus k int32 indices."""
    element = args.dtype.size_bytes()
    read = args.num_rows * args.num_columns * element
    written = args.num_rows * args.top_k * (element + 4)
    return int(read + written)


def _metrics(args: _ValidatedArgs, time_ms: float, energy_j: float) -> ComputeMetrics:
    elapsed = time_ms / 1000
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(_semantic_flops(args) / elapsed / 1e12 if elapsed else 0),
        memory_bandwidth_gbps=float(_logical_bytes(args) / elapsed / 1e9 if elapsed else 0),
        energy_j=float(energy_j),
    )


def profile_logits_topk_torch(
    num_rows: int,
    num_columns: int,
    top_k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    args = _validate_args(num_rows, num_columns, top_k, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for logits_topk") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for logits_topk")

    try:
        scores = _build_scores(torch, args)

        def kernel():
            return torch.topk(scores, args.top_k, dim=-1)

        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        return _metrics(args, time_ms, energy_j)
    except torch.OutOfMemoryError as exc:
        raise OOMError("torch logits_topk ran out of CUDA memory") from exc
    except (RuntimeError, AssertionError) as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def profile_logits_topk_flashinfer(
    num_rows: int,
    num_columns: int,
    top_k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    args = _validate_args(num_rows, num_columns, top_k, dtype)
    try:
        import torch
        from flashinfer import top_k as flashinfer_top_k
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch + flashinfer are required for the flashinfer logits_topk runner"
        ) from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for logits_topk")

    try:
        scores = _build_scores(torch, args)

        def kernel():
            # The exact call vLLM's `logits_processor._topk` makes.
            return flashinfer_top_k(scores, args.top_k, sorted=True, deterministic=True)

        values, indices = kernel()
        torch.cuda.synchronize()
        _check_against_torch(torch, scores, args.top_k, values, indices)

        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        return _metrics(args, time_ms, energy_j)
    except torch.OutOfMemoryError as exc:
        raise OOMError("flashinfer logits_topk ran out of CUDA memory") from exc
    except (RuntimeError, AssertionError) as exc:
        raise KernelLaunchFailed(str(exc)) from exc
