"""Profile FlashInfer's fused MLA RoPE, FP8 quantization, and query concat."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "mla_rope_quantize_fp8:flashinfer"
_SUPPORTED_GPUS = ("NVIDIA B200",)


@dataclass(frozen=True)
class _Launch:
    callable_: Any
    fp8_dtype: Any
    q_nope: Any
    q_rope: Any
    k_nope: Any
    k_rope: Any
    cos_sin_cache: Any
    pos_ids: Any
    q_nope_out: Any
    q_rope_out: Any
    k_nope_out: Any
    k_rope_out: Any
    is_neox: bool

    def run(self) -> None:
        self.callable_(
            q_rope=self.q_rope,
            k_rope=self.k_rope,
            q_nope=self.q_nope,
            k_nope=self.k_nope,
            cos_sin_cache=self.cos_sin_cache,
            pos_ids=self.pos_ids,
            is_neox=self.is_neox,
            quantize_dtype=self.fp8_dtype,
            q_rope_out=self.q_rope_out,
            k_rope_out=self.k_rope_out,
            q_nope_out=self.q_nope_out,
            k_nope_out=self.k_nope_out,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            enable_pdl=True,
        )


def _validate_args(
    num_tokens: int,
    num_heads: int,
    kv_lora_rank: int,
    rope_dim: int,
    max_position: int,
    is_neox_style: bool,
    input_dtype: DType | str,
    quant_dtype: DType | str,
) -> tuple[DType, DType]:
    for name, value in (
        ("num_tokens", num_tokens),
        ("num_heads", num_heads),
        ("kv_lora_rank", kv_lora_rank),
        ("rope_dim", rope_dim),
        ("max_position", max_position),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(is_neox_style) is not bool:
        raise ValueError("is_neox_style must be a bool")
    if rope_dim % 2 != 0:
        raise ValueError("rope_dim must be even")
    resolved_input = DType.from_value(input_dtype)
    resolved_quant = DType.from_value(quant_dtype)
    if resolved_input is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires input_dtype=bf16")
    if resolved_quant is not DType.FP8_E4M3:
        raise ProfilerNotImplemented(f"{_BACKEND} requires quant_dtype=fp8_e4m3")
    return resolved_input, resolved_quant


def _validate_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name not in _SUPPORTED_GPUS:
        raise ProfilerNotImplemented(
            f"{_BACKEND} is verified only on {' or '.join(_SUPPORTED_GPUS)}, got {gpu_name}"
        )


def _prepare(
    torch: Any,
    callable_: Any,
    *,
    num_tokens: int,
    num_heads: int,
    kv_lora_rank: int,
    rope_dim: int,
    max_position: int,
    is_neox_style: bool,
) -> _Launch:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(0)

    def randn(*shape: int) -> Any:
        return torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)

    q_nope = randn(num_tokens, num_heads, kv_lora_rank)
    q_rope = randn(num_tokens, num_heads, rope_dim)
    k_nope = randn(num_tokens, kv_lora_rank)
    k_rope = randn(num_tokens, rope_dim)
    cos_sin_cache = torch.randn(
        (max_position, rope_dim),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    pos_ids = torch.randint(
        0,
        max_position,
        (num_tokens,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    # Preserve production's strided writes into one [nope | rope] output.
    q_out = q_rope.new_empty(
        num_tokens,
        num_heads,
        kv_lora_rank + rope_dim,
        dtype=torch.float8_e4m3fn,
    )
    return _Launch(
        callable_=callable_,
        fp8_dtype=torch.float8_e4m3fn,
        q_nope=q_nope,
        q_rope=q_rope,
        k_nope=k_nope,
        k_rope=k_rope,
        cos_sin_cache=cos_sin_cache,
        pos_ids=pos_ids,
        q_nope_out=q_out[..., :kv_lora_rank],
        q_rope_out=q_out[..., kv_lora_rank:],
        k_nope_out=k_nope.new_empty(k_nope.shape, dtype=torch.float8_e4m3fn),
        k_rope_out=k_rope.new_empty(k_rope.shape, dtype=torch.float8_e4m3fn),
        is_neox=is_neox_style,
    )


def profile_mla_rope_quantize_fp8_flashinfer(
    num_tokens: int,
    num_heads: int,
    kv_lora_rank: int,
    rope_dim: int,
    max_position: int,
    is_neox_style: bool,
    input_dtype: DType | str,
    quant_dtype: DType | str,
) -> ComputeMetrics:
    resolved_input, resolved_quant = _validate_args(
        num_tokens,
        num_heads,
        kv_lora_rank,
        rope_dim,
        max_position,
        is_neox_style,
        input_dtype,
        quant_dtype,
    )
    try:
        import torch
        from flashinfer.rope import mla_rope_quantize_fp8
    except (ImportError, OSError) as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the SGLang environment") from exc

    try:
        _validate_device(torch)
        launch = _prepare(
            torch,
            mla_rope_quantize_fp8,
            num_tokens=num_tokens,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            rope_dim=rope_dim,
            max_position=max_position,
            is_neox_style=is_neox_style,
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
    row_width = kv_lora_rank + rope_dim
    logical_bytes = num_tokens * (
        (num_heads + 1) * row_width * (resolved_input.size_bytes() + resolved_quant.size_bytes())
        + rope_dim * 4
        + 8
    )
    flops = num_tokens * (num_heads + 1) * rope_dim * 3
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / elapsed_seconds / 1e12 if elapsed_seconds else 0.0),
        memory_bandwidth_gbps=float(
            logical_bytes / elapsed_seconds / 1e9 if elapsed_seconds else 0.0
        ),
        energy_j=float(energy_j),
    )


__all__ = ["profile_mla_rope_quantize_fp8_flashinfer"]
