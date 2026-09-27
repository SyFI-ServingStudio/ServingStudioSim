"""Shared production identity and correctness helpers for DeepSeek V4 MHC."""

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.runners.exceptions import ProfilerNotImplemented

BACKEND_GPU = "NVIDIA H200"
HIDDEN_SIZE = 4096
HC_MULT = 4
RMS_EPS = 1e-6
HC_EPS = 1e-6
POST_MULTIPLIER = 2.0
SINKHORN_ITERATIONS = 20
MIX_WIDTH = HC_MULT * (HC_MULT + 2)

_BF16_BYTES = 2
_FP32_BYTES = 4


@dataclass(frozen=True)
class Shape:
    num_tokens: int


@dataclass(frozen=True)
class CommonInputs:
    residual: Any
    fn: Any
    hc_scale: Any
    hc_base: Any
    norm_weight: Any


def validate_args(
    kind: str,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    hidden_dtype: DType | str,
) -> Shape:
    if type(num_tokens) is not int or num_tokens <= 0:
        raise ValueError("num_tokens must be a positive integer")
    if hidden_size != HIDDEN_SIZE or hc_mult != HC_MULT:
        raise ProfilerNotImplemented(
            f"{kind} requires hidden_size={HIDDEN_SIZE} and hc_mult={HC_MULT}"
        )
    if DType.from_value(hidden_dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{kind} requires hidden_dtype=bf16")
    return Shape(num_tokens)


def require_h200(torch: Any, kind: str) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{kind} requires CUDA")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != BACKEND_GPU:
        raise ProfilerNotImplemented(f"{kind} is verified only on {BACKEND_GPU}, got {gpu_name}")


def prepare_common(torch: Any, shape: Shape) -> CommonInputs:
    generator = torch.Generator().manual_seed(41)
    residual = torch.randn(
        (shape.num_tokens, HC_MULT, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        generator=generator,
    ).cuda()
    fn = (torch.randn((MIX_WIDTH, HC_MULT * HIDDEN_SIZE), generator=generator) * 0.01).cuda()
    hc_scale = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).cuda()
    hc_base = torch.zeros(MIX_WIDTH, dtype=torch.float32).cuda()
    norm_weight = torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16).cuda()
    return CommonInputs(residual, fn, hc_scale, hc_base, norm_weight)


def reference_pre(torch: Any, inputs: CommonInputs) -> tuple[Any, Any, Any]:
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch

    post_mix, comb_mix, layer_input = mhc_pre_torch(
        inputs.residual,
        inputs.fn,
        inputs.hc_scale,
        inputs.hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        POST_MULTIPLIER,
        SINKHORN_ITERATIONS,
    )
    normalized = torch.nn.functional.rms_norm(
        layer_input,
        (HIDDEN_SIZE,),
        inputs.norm_weight,
        RMS_EPS,
    )
    return post_mix, comb_mix, normalized


def residual_bytes(num_tokens: int) -> int:
    """The bf16 residual streams, (num_tokens, HC_MULT, HIDDEN_SIZE)."""
    return _BF16_BYTES * num_tokens * HC_MULT * HIDDEN_SIZE


def hidden_bytes(num_tokens: int) -> int:
    """One bf16 hidden state per token, (num_tokens, HIDDEN_SIZE)."""
    return _BF16_BYTES * num_tokens * HIDDEN_SIZE


def mix_bytes(num_tokens: int) -> int:
    """The fp32 post-mix (num_tokens, HC_MULT, 1) and comb-mix (num_tokens, HC_MULT, HC_MULT)."""
    return _FP32_BYTES * num_tokens * (HC_MULT + HC_MULT * HC_MULT)


def pre_weight_bytes() -> int:
    """fn, hc_scale, hc_base and the RMSNorm weight, read once per call."""
    fn = MIX_WIDTH * HC_MULT * HIDDEN_SIZE
    return _FP32_BYTES * (fn + 3 + MIX_WIDTH) + _BF16_BYTES * HIDDEN_SIZE


def head_weight_bytes() -> int:
    """The terminal head's fn rows, scale and base, and the final RMSNorm weight."""
    head_fn = HC_MULT * HC_MULT * HIDDEN_SIZE
    return _FP32_BYTES * (head_fn + 1 + HC_MULT) + _BF16_BYTES * HIDDEN_SIZE


def bandwidth_gbps(logical_bytes: int, time_ms: float) -> float:
    return logical_bytes / (time_ms / 1000.0) / 1e9


def assert_outputs_close(torch: Any, actual: tuple[Any, ...], expected: tuple[Any, ...]) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"expected {len(expected)} MHC outputs, got {len(actual)}")
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0.05, rtol=0.02)


__all__ = [
    "CommonInputs",
    "HC_EPS",
    "HC_MULT",
    "HIDDEN_SIZE",
    "MIX_WIDTH",
    "POST_MULTIPLIER",
    "RMS_EPS",
    "SINKHORN_ITERATIONS",
    "Shape",
    "assert_outputs_close",
    "bandwidth_gbps",
    "head_weight_bytes",
    "hidden_bytes",
    "mix_bytes",
    "pre_weight_bytes",
    "prepare_common",
    "reference_pre",
    "require_h200",
    "residual_bytes",
    "validate_args",
]
