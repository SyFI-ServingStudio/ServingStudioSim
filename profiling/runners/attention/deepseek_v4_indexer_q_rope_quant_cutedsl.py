"""Profile the public DeepSeek V4 indexer-Q CuteDSL operation."""

from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "deepseek_v4_indexer_q_rope_quant:vllm_cutedsl_fp8"
_GPU_NAME = "NVIDIA H200"
_MODEL_IDENTITY = (64, 128, 64, 128**-0.5, 64**-0.5, 448.0, 1.0e-4)
_STORAGE_IDENTITY = (
    "int64",
    "bf16",
    "fp32",
    "bf16",
    "fp8_e4m3",
    "fp32",
    "gptj_interleaved_trailing",
    "per_token_head_fp8_pow2_ceil_folded_weight",
)


@dataclass(frozen=True)
class _Shape:
    num_tokens: int
    max_model_len: int

    @property
    def coarsen(self) -> int:
        return 1 if self.num_tokens < 512 else 4


@dataclass(frozen=True)
class _Operands:
    positions: Any
    q: Any
    cos_sin_cache: Any
    weights: Any
    q_fp8: Any
    weights_out: Any


def _validate_args(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    fp8_max: float,
    scale_epsilon: float,
    positions_dtype: str,
    q_dtype: object,
    rope_dtype: object,
    weight_dtype: object,
    q_output_dtype: object,
    weight_output_dtype: object,
    rope_style: str,
    quant_mode: str,
) -> _Shape:
    if type(num_tokens) is not int or num_tokens <= 0:
        raise ValueError("num_tokens must be a positive integer")
    if type(max_num_batched_tokens) is not int or not num_tokens <= max_num_batched_tokens <= 32768:
        raise ValueError("max_num_batched_tokens must cover num_tokens and be <=32768")
    if type(max_model_len) is not int or not 1 <= max_model_len <= 1_048_576:
        raise ProfilerNotImplemented(f"{_BACKEND} supports max_model_len <=1048576")
    model_identity = (
        num_heads,
        head_dim,
        rope_dim,
        index_weights_softmax_scale,
        index_weights_head_scale,
        fp8_max,
        scale_epsilon,
    )
    if model_identity != _MODEL_IDENTITY:
        raise ProfilerNotImplemented(
            f"{_BACKEND} supports model identity {_MODEL_IDENTITY}, got {model_identity}"
        )
    storage_identity = (
        positions_dtype,
        str(q_dtype),
        str(rope_dtype),
        str(weight_dtype),
        str(q_output_dtype),
        str(weight_output_dtype),
        rope_style,
        quant_mode,
    )
    if storage_identity != _STORAGE_IDENTITY:
        raise ProfilerNotImplemented(
            f"{_BACKEND} supports storage identity {_STORAGE_IDENTITY}, got {storage_identity}"
        )
    return _Shape(num_tokens, max_model_len)


def _build_operands(torch: Any, shape: _Shape, device: Any) -> _Operands:
    if shape.num_tokens == 1:
        positions = torch.zeros(1, dtype=torch.int64, device=device)
    else:
        positions = torch.div(
            torch.arange(shape.num_tokens, dtype=torch.int64, device=device)
            * (shape.max_model_len - 1),
            shape.num_tokens - 1,
            rounding_mode="floor",
        )
    feature = torch.linspace(-0.75, 0.75, 128, dtype=torch.float32, device=device)
    tokens = torch.arange(shape.num_tokens, dtype=torch.float32, device=device)
    heads = torch.arange(64, dtype=torch.float32, device=device)
    q = (
        feature[None, None, :] + tokens[:, None, None] / 4096.0 + heads[None, :, None] / 1024.0
    ).to(torch.bfloat16)
    weights = (0.25 + tokens[:, None] / 8192.0 + heads[None] / 2048.0).to(torch.bfloat16)
    frequencies = torch.pow(
        10_000.0,
        -torch.arange(32, dtype=torch.float32, device=device) / 32.0,
    )
    angles = positions.float()[:, None] * frequencies[None]
    cos_sin_cache = torch.empty(
        (int(positions[-1].item()) + 1, 64), dtype=torch.float32, device=device
    )
    cos_sin_cache[positions] = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
    return _Operands(
        positions,
        q,
        cos_sin_cache,
        weights,
        torch.empty_like(q, dtype=torch.float8_e4m3fn),
        torch.empty_like(weights, dtype=torch.float32),
    )


def _launch(public_op: Any, operands: _Operands) -> None:
    public_op(
        operands.positions,
        operands.q,
        operands.cos_sin_cache,
        operands.weights,
        128**-0.5,
        64**-0.5,
        operands.q_fp8,
        operands.weights_out,
    )


def _reference(torch: Any, operands: _Operands) -> tuple[Any, Any]:
    selected = operands.cos_sin_cache.index_select(0, operands.positions)
    cosines, sines = selected.chunk(2, dim=1)
    rotated = operands.q.clone()
    rope = operands.q[..., 64:].float()
    even, odd = rope[..., 0::2], rope[..., 1::2]
    rotated[..., 64::2] = (even * cosines[:, None] - odd * sines[:, None]).to(torch.bfloat16)
    rotated[..., 65::2] = (odd * cosines[:, None] + even * sines[:, None]).to(torch.bfloat16)
    amax = rotated.float().abs().amax(dim=-1)
    scale = torch.exp2(torch.ceil(torch.log2(torch.clamp(amax, min=1.0e-4) / 448.0)))
    q_fp8 = (rotated.float() / scale[..., None]).to(torch.float8_e4m3fn)
    weights = operands.weights.float() * scale * float(128**-0.5 * 64**-0.5)
    return q_fp8, weights


def _check_output(torch: Any, public_op: Any, operands: _Operands) -> None:
    expected_q, expected_weights = _reference(torch, operands)
    _launch(public_op, operands)
    torch.cuda.synchronize(operands.q.device)
    torch.testing.assert_close(operands.weights_out, expected_weights, atol=2e-7, rtol=2e-6)
    actual_bits = operands.q_fp8.view(torch.uint8)
    expected_bits = expected_q.view(torch.uint8)
    actual_magnitude = (actual_bits & 0x7F).to(torch.int16)
    expected_magnitude = (expected_bits & 0x7F).to(torch.int16)
    same_sign = (actual_bits & 0x80) == (expected_bits & 0x80)
    accepted = (actual_bits == expected_bits) | (
        same_sign & ((actual_magnitude - expected_magnitude).abs() <= 1)
    )
    if not bool(accepted.all().item()):
        raise KernelLaunchFailed(f"{_BACKEND} differs from Torch by more than one FP8 bin")


def _logical_bytes(num_tokens: int) -> int:
    return (
        8 * num_tokens
        + 2 * num_tokens * 64 * 128
        + 4 * num_tokens * 64
        + 2 * num_tokens * 64
        + num_tokens * 64 * 128
        + 4 * num_tokens * 64
    )


def profile_deepseek_v4_indexer_q_rope_quant_cutedsl(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    fp8_max: float,
    scale_epsilon: float,
    positions_dtype: str,
    q_dtype: object,
    rope_dtype: object,
    weight_dtype: object,
    q_output_dtype: object,
    weight_output_dtype: object,
    rope_style: str,
    quant_mode: str,
) -> ComputeMetrics:
    shape = _validate_args(
        num_tokens,
        num_heads,
        head_dim,
        rope_dim,
        max_model_len,
        max_num_batched_tokens,
        index_weights_softmax_scale,
        index_weights_head_scale,
        fp8_max,
        scale_epsilon,
        positions_dtype,
        q_dtype,
        rope_dtype,
        weight_dtype,
        q_output_dtype,
        weight_output_dtype,
        rope_style,
        quant_mode,
    )
    try:
        import torch
        from vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl import (
            fused_indexer_q_rope_quant_fp8_cutedsl,
        )
        from vllm.utils.import_utils import has_cutedsl
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires pinned vLLM CuteDSL") from exc
    try:
        if not torch.cuda.is_available():
            raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
        device = torch.device("cuda", torch.cuda.current_device())
        if (
            str(torch.cuda.get_device_name(device)) != _GPU_NAME
            or tuple(torch.cuda.get_device_capability(device)) != (9, 0)
            or not has_cutedsl()
        ):
            raise ProfilerNotImplemented(f"{_BACKEND} requires {_GPU_NAME} SM90 CuteDSL")
        operands = _build_operands(torch, shape, device)
        _check_output(torch, fused_indexer_q_rope_quant_fp8_cutedsl, operands)

        def run() -> None:
            _launch(fused_indexer_q_rope_quant_fp8_cutedsl, operands)

        time_ms = Timer.cupti(run, kernel_name=None)
        energy_j = Energy.perf(run, per_iter_time_ms=time_ms)
        elapsed_s = time_ms / 1000.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(_logical_bytes(shape.num_tokens) / elapsed_s / 1e9),
            energy_j=float(energy_j),
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, TypeError, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc


__all__ = ["profile_deepseek_v4_indexer_q_rope_quant_cutedsl"]
