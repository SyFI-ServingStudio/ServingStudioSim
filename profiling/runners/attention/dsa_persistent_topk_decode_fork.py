"""Profile the pinned vLLM fork's persistent decode top-k callable on max-ragged rows."""

from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.dsa_persistent_topk_decode import VLLM_TOP_K, WORKSPACE_BYTES
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "dsa_persistent_topk_decode:vllm_fork_cuda"


@dataclass(frozen=True)
class _Operands:
    logits: Any
    lengths: Any
    indices: Any
    workspace: Any
    top_k: int


def _max_ragged_lengths(batch_size: int, context_len: int, next_n: int = 1) -> tuple[int, ...]:
    """Row lengths, request-major; a request's ``next_n`` rows end at its context."""
    return tuple(
        max(0, max(0, context_len - request) - next_n + 1 + token)
        for request in range(batch_size)
        for token in range(next_n)
    )


def _validate_args(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    logits_row_stride: int,
    logits_dtype: object,
    index_dtype: str,
    context_mode: str,
) -> tuple[int, ...]:
    # No row cap: topk.cu dispatches any row count and sizes its grid from the
    # device SM count.
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive int")
    if type(context_len) is not int or context_len < 0:
        raise ValueError("context_len must be a nonnegative int")
    if type(next_n) is not int or next_n < 1:
        raise ValueError("next_n must be a positive int")
    if top_k not in VLLM_TOP_K:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires top_k in {sorted(VLLM_TOP_K)}, got {top_k}"
        )
    if (
        type(max_model_len) is not int
        or max_model_len < 1
        or context_len > max_model_len
        or logits_row_stride < max_model_len
    ):
        raise ValueError(
            f"{_BACKEND} requires context_len <= max_model_len, max_model_len >= 1, "
            "and logits_row_stride >= max_model_len"
        )
    if (str(logits_dtype), index_dtype, context_mode) != (
        "fp32",
        "int32",
        "max_ragged",
    ):
        raise ProfilerNotImplemented(f"{_BACKEND} requires fp32/int32 max-ragged storage identity")
    return _max_ragged_lengths(batch_size, context_len, next_n)


def _build_operands(
    torch: Any,
    *,
    lengths: tuple[int, ...],
    logits_row_stride: int,
    device: Any,
    top_k: int,
) -> _Operands:
    num_rows = len(lengths)
    logits = torch.empty((num_rows, logits_row_stride), dtype=torch.float32, device=device)
    # Use one deterministic production-like row template. A million-point
    # linspace has ~2e-6 spacing and is a pathological radix-selection input;
    # value-set comparison below already handles the rare random tie.
    generator = torch.Generator(device=device).manual_seed(0)
    logits.copy_(
        torch.randn(
            logits_row_stride,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
    )
    return _Operands(
        logits=logits,
        lengths=torch.tensor(lengths, dtype=torch.int32, device=device),
        indices=torch.empty((num_rows, top_k), dtype=torch.int32, device=device),
        workspace=torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device),
        top_k=top_k,
    )


def _launch(public_op: Any, operands: _Operands, max_seq_len: int) -> None:
    public_op(
        operands.logits,
        operands.lengths,
        operands.indices,
        operands.workspace,
        operands.top_k,
        max_seq_len,
    )


def _check_output(torch: Any, operands: _Operands) -> None:
    sampled_rows = sorted({0, len(operands.lengths) // 2, len(operands.lengths) - 1})
    for row in sampled_rows:
        length = int(operands.lengths[row].item())
        selected_count = min(length, operands.top_k)
        if selected_count == 0:
            continue
        actual = operands.indices[row, :selected_count].long()
        if actual.min() < 0 or actual.max() >= length or actual.unique().numel() != selected_count:
            raise KernelLaunchFailed(f"{_BACKEND} returned invalid or duplicate indices")
        expected = torch.topk(operands.logits[row, :length], selected_count).indices
        torch.testing.assert_close(
            operands.logits[row].index_select(0, actual).sort().values,
            operands.logits[row].index_select(0, expected).sort().values,
            atol=0,
            rtol=0,
        )


def profile_dsa_persistent_topk_decode_vllm_fork_cuda(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    logits_row_stride: int,
    logits_dtype: object,
    index_dtype: str,
    context_mode: str,
) -> ComputeMetrics:
    lengths = _validate_args(
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
        import vllm._C_stable_libtorch  # noqa: F401  # Register torch.ops._C.
    except (ImportError, OSError) as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires pinned vLLM") from exc

    try:
        if not torch.cuda.is_available():
            raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
        device = torch.device("cuda", torch.cuda.current_device())
        public_op = torch.ops._C.persistent_topk
        operands = _build_operands(
            torch,
            lengths=lengths,
            logits_row_stride=logits_row_stride,
            device=device,
            top_k=top_k,
        )
        max_seq_len = max(lengths)

        def launch() -> None:
            _launch(public_op, operands, max_seq_len)

        launch()
        torch.cuda.synchronize(device)
        _check_output(torch, operands)
        time_ms = Timer.cupti(launch, kernel_name=None)
        energy_j = Energy.perf(launch, per_iter_time_ms=time_ms)
        num_rows = len(lengths)
        logical_bytes = 4 * sum(lengths) + 4 * num_rows + 4 * num_rows * top_k
        elapsed_seconds = time_ms / 1000.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(logical_bytes / elapsed_seconds / 1e9),
            energy_j=float(energy_j),
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, TypeError, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc


__all__ = ["profile_dsa_persistent_topk_decode_vllm_fork_cuda"]
