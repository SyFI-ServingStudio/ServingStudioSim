"""Profile SGLang's fused DSA indexer-query RoPE and FP8 quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "dsa_indexer_q_rope_quant:sglang_cuda"
_SUPPORTED_GPUS = ("NVIDIA B200",)
_HEAD_DIM = 128
_ROPE_DIM = 64
_ROPE_LAYOUT = "rope_first"
_WEIGHT_SCALE = 0.088_388_347_648_318_45
_ROPE_TABLE_ROWS = 131_072


@dataclass(frozen=True)
class _Launch:
    callable_: Any
    q_input: Any
    weight: Any
    cos_sin_cache: Any
    positions: Any

    def run(self) -> None:
        self.callable_(
            self.q_input,
            self.weight,
            _WEIGHT_SCALE,
            self.cos_sin_cache,
            self.positions,
        )


def _validate_args(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    rope_layout: str,
    hadamard: bool,
    input_dtype: DType | str,
    q_output_dtype: DType | str,
    weight_output_dtype: DType | str,
) -> None:
    for name, value in (
        ("num_tokens", num_tokens),
        ("num_heads", num_heads),
        ("head_dim", head_dim),
        ("rope_dim", rope_dim),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(hadamard) is not bool:
        raise ValueError("hadamard must be a bool")
    if (head_dim, rope_dim) != (_HEAD_DIM, _ROPE_DIM):
        raise ProfilerNotImplemented(
            f"{_BACKEND} is instantiated at (head_dim, rope_dim) == "
            f"({_HEAD_DIM}, {_ROPE_DIM}), got ({head_dim}, {rope_dim})"
        )
    if rope_layout != _ROPE_LAYOUT or hadamard:
        raise ProfilerNotImplemented(
            f"{_BACKEND} measures rope_first without Hadamard, got "
            f"rope_layout={rope_layout!r} hadamard={hadamard}"
        )
    if DType.from_value(input_dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires input_dtype=bf16")
    if DType.from_value(q_output_dtype) is not DType.FP8_E4M3:
        raise ProfilerNotImplemented(f"{_BACKEND} requires q_output_dtype=fp8_e4m3")
    if DType.from_value(weight_output_dtype) is not DType.FP32:
        raise ProfilerNotImplemented(f"{_BACKEND} requires weight_output_dtype=fp32")


def _validate_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name not in _SUPPORTED_GPUS:
        raise ProfilerNotImplemented(
            f"{_BACKEND} is verified only on {' or '.join(_SUPPORTED_GPUS)}, got {gpu_name}"
        )


def _prepare(torch: Any, callable_: Any, *, num_tokens: int, num_heads: int) -> _Launch:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(0)
    q_input = torch.randn(
        (num_tokens, num_heads, _HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    # The gate weight is a trailing slice of wk_weights_proj, so preserve its
    # production row stride rather than allocating a contiguous stand-in.
    key_and_weight = torch.randn(
        (num_tokens, _HEAD_DIM + num_heads),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    cos_sin_cache = torch.randn(
        (_ROPE_TABLE_ROWS, _ROPE_DIM),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    positions = torch.randint(
        0,
        _ROPE_TABLE_ROWS,
        (num_tokens,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    return _Launch(
        callable_=callable_,
        q_input=q_input,
        weight=key_and_weight[:, _HEAD_DIM:],
        cos_sin_cache=cos_sin_cache,
        positions=positions,
    )


def profile_dsa_indexer_q_rope_quant_sglang_cuda(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    rope_layout: str,
    hadamard: bool,
    input_dtype: DType | str,
    q_output_dtype: DType | str,
    weight_output_dtype: DType | str,
) -> ComputeMetrics:
    """Time the public one-launch callable used by GLM-5.2's SGLang indexer."""
    _validate_args(
        num_tokens,
        num_heads,
        head_dim,
        rope_dim,
        rope_layout,
        hadamard,
        input_dtype,
        q_output_dtype,
        weight_output_dtype,
    )
    try:
        import torch
        from sglang.kernels.ops.attention.dsv4 import fused_q_indexer_rope_first_quant
    except (ImportError, OSError) as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the SGLang environment") from exc

    try:
        _validate_device(torch)
        launch = _prepare(
            torch,
            fused_q_indexer_rope_first_quant,
            num_tokens=num_tokens,
            num_heads=num_heads,
        )
        launch.run()
        torch.cuda.synchronize()
        time_ms = Timer.cupti(launch.run, warmup=5)
        energy_j = Energy.perf(launch.run, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    elapsed_seconds = time_ms / 1000.0
    rows = num_tokens * num_heads
    logical_bytes = rows * (head_dim * 3 + 2 + 4) + num_tokens * (rope_dim * 4 + 8)
    flops = rows * (rope_dim // 2 * 6 + head_dim)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / elapsed_seconds / 1e12 if elapsed_seconds else 0.0),
        memory_bandwidth_gbps=float(
            logical_bytes / elapsed_seconds / 1e9 if elapsed_seconds else 0.0
        ),
        energy_j=float(energy_j),
    )


__all__ = ["profile_dsa_indexer_q_rope_quant_sglang_cuda"]
