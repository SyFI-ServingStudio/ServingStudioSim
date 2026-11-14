"""Profile the vendored Triton fp32 row-split LayerNorm epilogue (backend ``triton``).

The kernel (``row_split_vendor/triton_post_a.py``, one program per row) reads
one fp32 GEMM output row, writes its leading span and a trailing span rounded
to bf16 into two buffers, and LayerNorms the span between them (fp32 math on
the bf16-rounded values, fp32 weight and bias) into a third bf16 buffer. The
fp32 tail past the trailing span is not touched: callers read it in place.

The row geometry is compiled into the vendored kernel, so only that geometry
runs; any other is refused rather than timed on the wrong kernel.
"""

from __future__ import annotations

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "fp32_row_split_layer_norm:triton"
# (bf16_lead_cols, norm_cols, bf16_trail_cols, fp32_tail_cols) built into _post_a.
_COMPILED_GEOMETRY = (2048, 128, 128, 32)
# The kernel's eps is a runtime scalar; it never changes the launch.
_EPS = 1e-6


def _validate_args(
    num_tokens: int,
    bf16_lead_cols: int,
    norm_cols: int,
    bf16_trail_cols: int,
    fp32_tail_cols: int,
) -> None:
    for name, value in (
        ("num_tokens", num_tokens),
        ("bf16_lead_cols", bf16_lead_cols),
        ("norm_cols", norm_cols),
        ("bf16_trail_cols", bf16_trail_cols),
        ("fp32_tail_cols", fp32_tail_cols),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    geometry = (bf16_lead_cols, norm_cols, bf16_trail_cols, fp32_tail_cols)
    if geometry != _COMPILED_GEOMETRY:
        raise ProfilerNotImplemented(
            f"{_BACKEND} has the row geometry {_COMPILED_GEOMETRY} compiled in, got {geometry}"
        )


def logical_bytes(
    num_tokens: int, bf16_lead_cols: int, norm_cols: int, bf16_trail_cols: int
) -> int:
    """fp32 read of the three written spans, their bf16 writes, and the LayerNorm parameters."""
    cols = bf16_lead_cols + norm_cols + bf16_trail_cols
    return num_tokens * cols * (4 + 2) + 2 * 4 * norm_cols


def profile_fp32_row_split_layer_norm_triton(
    num_tokens: int,
    bf16_lead_cols: int,
    norm_cols: int,
    bf16_trail_cols: int,
    fp32_tail_cols: int,
) -> ComputeMetrics:
    _validate_args(num_tokens, bf16_lead_cols, norm_cols, bf16_trail_cols, fp32_tail_cols)
    try:
        import torch

        from profiling.runners.norm.row_split_vendor.triton_post_a import post_a
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires Torch and Triton") from exc
    from profiling.runners.norm.fp32_row_split_layer_norm_reference import (
        check_row_split,
        fp32_row_split_layer_norm_reference,
    )

    width = bf16_lead_cols + norm_cols + bf16_trail_cols + fp32_tail_cols
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(47)
        y = torch.randn(
            (num_tokens, width), dtype=torch.float32, device=device, generator=generator
        )
        weight = 1.0 + 0.1 * torch.randn(norm_cols, device=device, generator=generator)
        bias = 0.1 * torch.randn(norm_cols, device=device, generator=generator)
        lead = torch.empty((num_tokens, bf16_lead_cols), dtype=torch.bfloat16, device=device)
        normed = torch.empty((num_tokens, norm_cols), dtype=torch.bfloat16, device=device)
        trail = torch.empty((num_tokens, bf16_trail_cols), dtype=torch.bfloat16, device=device)

        def launch() -> None:
            post_a(y, lead, normed, trail, weight, bias, _EPS)

        y_before = y.clone()
        for out in (lead, normed, trail):
            out.fill_(float("nan"))
        launch()
        torch.cuda.synchronize()
        check_row_split(
            torch,
            y_before,
            y,
            (lead, normed, trail),
            fp32_row_split_layer_norm_reference(
                torch, y_before, weight, bias, _EPS, bf16_lead_cols, norm_cols, bf16_trail_cols
            ),
        )
        del y_before
        # One launch per call.
        time_ms = Timer.cupti(launch, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc
    seconds = time_ms / 1000.0
    moved = logical_bytes(num_tokens, bf16_lead_cols, norm_cols, bf16_trail_cols)
    return ComputeMetrics(float(time_ms), 0.0, moved / seconds / 1e9, float(energy_j))


__all__ = ["logical_bytes", "profile_fp32_row_split_layer_norm_triton"]
