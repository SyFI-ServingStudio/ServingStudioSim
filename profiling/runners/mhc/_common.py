"""Shared production identity and correctness helpers for DeepSeek V4 MHC."""

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.runners.exceptions import ProfilerNotImplemented

# The post/pre boundary callables are shared by DeepSeek V4 and GLM-5.3-Flash.
# GLM-5.3-Flash runs them with rms_norm_eps=1e-5 instead of RMS_EPS below; the
# eps is a scalar and does not change the launch sequence or the timing.
# The TileLang kernels are JIT-compiled for the current GPU and take hidden_size
# and hc_mult from the tensor shapes, so neither the GPU nor the stream geometry
# is a launch constraint; every runner checks its outputs against vLLM's Torch
# implementation before timing, which rejects a geometry a kernel mishandles.
# DeepSeek V4 production geometry, kept as the documented default.
HIDDEN_SIZE = 4096
HC_MULT = 4
RMS_EPS = 1e-6
HC_EPS = 1e-6
POST_MULTIPLIER = 2.0
SINKHORN_ITERATIONS = 20


def mix_width(hc_mult: int) -> int:
    """Rows of fn: hc_mult pre + hc_mult post + hc_mult² comb mixes."""
    return hc_mult * (hc_mult + 2)


MIX_WIDTH = mix_width(HC_MULT)

_BF16_BYTES = 2
_FP32_BYTES = 4


@dataclass(frozen=True)
class Shape:
    num_tokens: int
    hidden_size: int = HIDDEN_SIZE
    hc_mult: int = HC_MULT


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
    for name, value in (
        ("num_tokens", num_tokens),
        ("hidden_size", hidden_size),
        ("hc_mult", hc_mult),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    # The vLLM wrappers assert bf16 residual streams (mhc/tilelang.py).
    if DType.from_value(hidden_dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{kind} requires hidden_dtype=bf16")
    return Shape(num_tokens, hidden_size, hc_mult)


def require_cuda(torch: Any, kind: str) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{kind} requires CUDA")


def prepare_common(torch: Any, shape: Shape) -> CommonInputs:
    generator = torch.Generator().manual_seed(41)
    hidden, streams = shape.hidden_size, shape.hc_mult
    residual = torch.randn(
        (shape.num_tokens, streams, hidden),
        dtype=torch.bfloat16,
        generator=generator,
    ).cuda()
    width = mix_width(streams)
    fn = (torch.randn((width, streams * hidden), generator=generator) * 0.01).cuda()
    hc_scale = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).cuda()
    hc_base = torch.zeros(width, dtype=torch.float32).cuda()
    norm_weight = torch.ones(hidden, dtype=torch.bfloat16).cuda()
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
        (inputs.residual.shape[-1],),
        inputs.norm_weight,
        RMS_EPS,
    )
    return post_mix, comb_mix, normalized


def residual_bytes(shape: Shape) -> int:
    """The bf16 residual streams, (num_tokens, hc_mult, hidden_size)."""
    return _BF16_BYTES * shape.num_tokens * shape.hc_mult * shape.hidden_size


def hidden_bytes(shape: Shape) -> int:
    """One bf16 hidden state per token, (num_tokens, hidden_size)."""
    return _BF16_BYTES * shape.num_tokens * shape.hidden_size


def mix_bytes(shape: Shape) -> int:
    """The fp32 post-mix (num_tokens, hc_mult, 1) and comb-mix (num_tokens, hc_mult, hc_mult)."""
    return _FP32_BYTES * shape.num_tokens * (shape.hc_mult + shape.hc_mult * shape.hc_mult)


def pre_weight_bytes(shape: Shape) -> int:
    """fn, hc_scale, hc_base and the RMSNorm weight, read once per call."""
    width = mix_width(shape.hc_mult)
    fn = width * shape.hc_mult * shape.hidden_size
    return _FP32_BYTES * (fn + 3 + width) + _BF16_BYTES * shape.hidden_size


def head_weight_bytes(shape: Shape) -> int:
    """The terminal head's fn rows, scale and base, and the final RMSNorm weight."""
    head_fn = shape.hc_mult * shape.hc_mult * shape.hidden_size
    return _FP32_BYTES * (head_fn + 1 + shape.hc_mult) + _BF16_BYTES * shape.hidden_size


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
    "mix_width",
    "reference_pre",
    "require_cuda",
    "residual_bytes",
    "validate_args",
]
