"""Row-wise top-k over a dense score matrix: flashinfer and torch backends.

Both entry points time exactly what vLLM's `logits_processor._topk` runs —
`flashinfer.top_k(scores, k, sorted=True, deterministic=True)` when flashinfer
imports, `torch.topk(scores, k, dim=-1)` otherwise. Neither is a rewrite: the
runner only allocates the operand and calls the public function.

The `tensorfold_torch` entry times TensorFold's call (perf-study 6469730a),
`torch.topk(scores, k, dim=-1, sorted=False)` on fp32 scores, in TensorFold's
own Torch (2.14.1+cu130). Two call sites make it: the DFlash2 drafter head
(`families/qwen3_5/cuda/dflash2.py` `launch_blocks`, k = 16 over the
98,592-column vocabulary subset, one row per drafted position) and target
sampling (`cuda/sampling.py` `sample_streams`, k = top_k + MARGIN over the full
vocabulary, one row per verify-window row). Both pass `logits.float()`: the
bf16-to-fp32 cast (and `sample_streams`' row `cat`) are separate launches
before the call and are not part of this kind. Torch's multi-block radix select
(`computeBlockDigitCounts` per radix digit, `gatherTopK`, the CUB scans) is a
fixed launch sequence, so CUPTI counts every launch (`kernel_name=None`).

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
# Rows of the tensorfold_torch selection checked against a full sort.
_CHECK_ROWS = 16


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


def _check_unsorted_selection(torch: Any, scores: Any, top_k: int, values: Any, indices: Any) -> None:
    """``sorted=False`` returns the top-k set in an unspecified order: compare it as a set.

    The reference is an independent full sort of each checked row, so a
    selection bug in the radix path (a wrong k-th boundary) moves a value and
    fails here. Rows are selected independently, so the first
    ``_CHECK_ROWS`` rows stand for all of them without sorting a 4 GB matrix.
    """
    scores, values, indices = scores[:_CHECK_ROWS], values[:_CHECK_ROWS], indices[:_CHECK_ROWS]
    expected = torch.sort(scores.float(), dim=-1, descending=True).values[:, :top_k]
    got = torch.sort(values.float(), dim=-1, descending=True).values
    if got.shape != expected.shape or indices.shape != expected.shape:
        raise RuntimeError(
            f"torch.topk(sorted=False) returned {tuple(got.shape)} / {tuple(indices.shape)}, "
            f"expected {tuple(expected.shape)}"
        )
    if not torch.equal(got, expected):
        raise RuntimeError("torch.topk(sorted=False) selected a different set than a full sort")
    if not torch.equal(scores.gather(1, indices.long()), values):
        raise RuntimeError("torch.topk(sorted=False) indices do not point at its values")


def profile_logits_topk_tensorfold_torch(
    num_rows: int,
    num_columns: int,
    top_k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    args = _validate_args(num_rows, num_columns, top_k, dtype)
    if args.dtype is not DType.FP32:
        # TensorFold always selects over `logits.float()`; a bf16 select is a
        # call it never makes.
        raise ProfilerNotImplemented(
            f"logits_topk:tensorfold_torch selects over fp32 scores only, got {args.dtype.value}"
        )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for logits_topk") from exc

    try:
        scores = _build_scores(torch, args)

        def kernel():
            # The exact call dflash2.launch_blocks and sampling.sample_streams make.
            return torch.topk(scores, args.top_k, dim=-1, sorted=False)

        values, indices = kernel()
        torch.cuda.synchronize()
        _check_unsorted_selection(torch, scores, args.top_k, values, indices)

        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        return _metrics(args, time_ms, energy_j)
    except torch.OutOfMemoryError as exc:
        raise OOMError("tensorfold_torch logits_topk ran out of CUDA memory") from exc
    except (RuntimeError, AssertionError) as exc:
        raise KernelLaunchFailed(str(exc)) from exc
