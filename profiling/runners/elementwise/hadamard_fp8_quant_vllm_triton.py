"""Profile the vLLM fork's fused Hadamard-128 rotation and per-row FP8 quantization.

The timed call is ``fwht128_quant_fp8`` from
``vllm.models.glm5next.nvidia.ops.kpool_compress``: a contiguous
``[num_rows, 128]`` BF16 tensor in, ``[num_rows, 128]`` FP8 E4M3 rows and
``[num_rows, 1]`` FP32 power-of-two scales out, one Triton launch with fixed
``BLOCK_R=32`` / ``num_warps=2`` (no autotune). The indexer hands it the
``wq_b`` projection output viewed as ``[tokens * heads, 128]``.

The oracle is a dense Sylvester Hadamard matmul in float64, independent of the
butterfly order the kernel uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "hadamard_fp8_quant:vllm_triton"
_MODULE = "vllm.models.glm5next.nvidia.ops.kpool_compress"
_HEAD_DIM = 128
_FP8_MAX = 448.0
_SCALE_FLOOR = 1e-4
# Rows checked against the float64 oracle; the kernel's programs are independent
# 32-row blocks, so a bounded prefix plus the ragged tail covers every code path.
_CHECK_ROWS = 4096


@dataclass(frozen=True)
class _Launch:
    callable_: Any
    q: Any

    def run(self) -> Any:
        return self.callable_(self.q)


def _validate_args(num_rows: int, head_dim: int, input_dtype: DType | str) -> None:
    if type(num_rows) is not int or num_rows <= 0:
        raise ValueError(f"num_rows must be a positive integer, got {num_rows!r}")
    if type(head_dim) is not int or head_dim <= 0:
        raise ValueError(f"head_dim must be a positive integer, got {head_dim!r}")
    if head_dim != _HEAD_DIM:
        raise ProfilerNotImplemented(
            f"{_BACKEND} is built for head_dim={_HEAD_DIM}, got {head_dim}"
        )
    if DType.from_value(input_dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires input_dtype=bf16")


def _hadamard(torch: Any, n: int, device: Any) -> Any:
    h = torch.ones((1, 1), dtype=torch.float64, device=device)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / n**0.5


def reference(torch: Any, q: Any) -> tuple[Any, Any]:
    """Float64 rotation, BF16 materialization, then per-row UE8M0 FP8 quant."""
    y = (q.double() @ _hadamard(torch, q.shape[1], q.device)).to(torch.bfloat16).double()
    absmax = y.abs().amax(dim=1, keepdim=True).clamp_min(_SCALE_FLOOR)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / _FP8_MAX)))
    return (y / scale).clamp(-_FP8_MAX, _FP8_MAX), scale


def _checked_rows(torch: Any, rows: int, device: Any) -> Any:
    """Leading rows up to _CHECK_ROWS plus the last 64, each row once."""
    head = min(rows, _CHECK_ROWS)
    return torch.cat(
        (
            torch.arange(head, device=device),
            torch.arange(max(rows - 64, head), rows, device=device),
        )
    )


def _check_output(torch: Any, launch: _Launch) -> None:
    rows = launch.q.shape[0]
    idx = _checked_rows(torch, rows, launch.q.device)
    q_fp8, scale = launch.run()
    torch.cuda.synchronize()
    if q_fp8.shape != (rows, _HEAD_DIM) or q_fp8.dtype != torch.float8_e4m3fn:
        raise AssertionError(f"unexpected FP8 output {tuple(q_fp8.shape)} {q_fp8.dtype}")
    if scale.shape != (rows, 1) or scale.dtype != torch.float32:
        raise AssertionError(f"unexpected scale output {tuple(scale.shape)} {scale.dtype}")
    ref_q, ref_scale = reference(torch, launch.q[idx])
    got_scale = scale[idx].double()
    # The butterfly and the dense matmul round differently in fp32, so the BF16
    # rotated value can move by one ulp; that may flip a power-of-two scale only
    # when a row's absmax sits on a power-of-two boundary.
    scale_mismatch = (got_scale != ref_scale).sum().item()
    if scale_mismatch > max(1, idx.numel() // 1000):
        raise AssertionError(f"{scale_mismatch} of {idx.numel()} row scales differ from oracle")
    got = q_fp8[idx].double() * got_scale
    want = ref_q * ref_scale
    # E4M3 keeps 3 mantissa bits: half-ulp relative error 2^-4, plus subnormal step.
    tol = want.abs() * 2.0**-3 + got_scale * 2.0**-8
    bad = (got - want).abs() > tol
    if bad.any():
        raise AssertionError(f"{int(bad.sum())} FP8 elements differ from the Hadamard oracle")


def profile_hadamard_fp8_quant_vllm_triton(
    num_rows: int,
    head_dim: int,
    input_dtype: DType | str,
) -> ComputeMetrics:
    """Time one fwht128_quant_fp8 launch on random BF16 rows."""
    _validate_args(num_rows, head_dim, input_dtype)
    try:
        import importlib

        import torch

        callable_ = importlib.import_module(_MODULE).fwht128_quant_fp8
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the vLLM fork ({_MODULE})") from exc

    try:
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(0)
        q = torch.randn(
            (num_rows, head_dim), dtype=torch.bfloat16, device=device, generator=generator
        )
        launch = _Launch(callable_, q)
        _check_output(torch, launch)
        time_ms = Timer.cupti(launch.run, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch.run, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {type(exc).__name__}: {exc}") from exc

    elapsed_seconds = time_ms / 1000.0
    logical_bytes = num_rows * (head_dim * (2 + 1) + 4)
    flops = num_rows * head_dim * 7  # log2(128) butterfly add/sub per element
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / elapsed_seconds / 1e12 if elapsed_seconds else 0.0),
        memory_bandwidth_gbps=float(
            logical_bytes / elapsed_seconds / 1e9 if elapsed_seconds else 0.0
        ),
        energy_j=float(energy_j),
    )


__all__ = ["profile_hadamard_fp8_quant_vllm_triton", "reference"]
