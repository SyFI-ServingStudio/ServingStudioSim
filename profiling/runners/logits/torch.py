"""Profile the public Torch operations used by vLLM logits sampling."""

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError
from profiling.runners.logits._common import logits_dtype, make_logits, require_b200, validate_shape
from profiling.runners.logits.reference import logits_argmax_reference, logits_copy_reference
from profiling.runners.metrics import ComputeMetrics


def copy_logits(torch: Any, logits: Any, output_dtype: Any) -> Any:
    if logits.dtype == output_dtype:
        return logits.clone(memory_format=torch.contiguous_format)
    return logits.to(dtype=output_dtype, memory_format=torch.contiguous_format)


def argmax_logits(logits: Any) -> Any:
    return logits.argmax(dim=-1)


def _profile(torch: Any, launch: Any, expected: Any, logical_bytes: int) -> ComputeMetrics:
    actual = launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if not actual.is_contiguous():
        raise KernelLaunchFailed("logits output must be contiguous")
    # Count every launch, including any multi-stage reduction selected by Torch.
    time_ms = Timer.cupti(launch, warmup=5, kernel_name=None)
    energy_j = Energy.perf(launch, warmup=5, per_iter_time_ms=time_ms)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=logical_bytes / (time_ms * 1e6) if time_ms else 0.0,
        energy_j=float(energy_j),
    )


def profile_logits_copy(
    num_rows: int,
    vocab_size: int,
    row_stride: int,
    input_dtype: DType | str,
    output_dtype: DType | str,
) -> ComputeMetrics:
    validate_shape(num_rows, vocab_size, row_stride)
    input_dtype, output_dtype = logits_dtype(input_dtype), logits_dtype(output_dtype)
    import torch

    require_b200(torch)
    try:
        target_dtype = output_dtype.torch()
        logits = make_logits(
            torch, num_rows, vocab_size, row_stride, input_dtype.torch(), device="cuda"
        )
        expected = logits_copy_reference(torch, logits, target_dtype)
        return _profile(
            torch,
            lambda: copy_logits(torch, logits, target_dtype),
            expected,
            int(num_rows * vocab_size * (input_dtype.size_bytes() + output_dtype.size_bytes())),
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError("logits_copy ran out of CUDA memory") from exc


def profile_logits_argmax(
    num_rows: int,
    vocab_size: int,
    row_stride: int,
    dtype: DType | str,
) -> ComputeMetrics:
    validate_shape(num_rows, vocab_size, row_stride)
    dtype = logits_dtype(dtype)
    import torch

    require_b200(torch)
    try:
        logits = make_logits(torch, num_rows, vocab_size, row_stride, dtype.torch(), device="cuda")
        expected = logits_argmax_reference(torch, logits)
        return _profile(
            torch,
            lambda: argmax_logits(logits),
            expected,
            int(num_rows * (vocab_size * dtype.size_bytes() + 8)),
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError("logits_argmax ran out of CUDA memory") from exc
