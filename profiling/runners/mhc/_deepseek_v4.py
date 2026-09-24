"""Shared production identity and correctness helpers for DeepSeek V4 MHC."""

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.runners.exceptions import ProfilerNotImplemented

TERMINAL_HEAD_GPUS = ("NVIDIA H200",)
# The post/pre boundary callables are shared by DeepSeek V4 (H200) and
# GLM-5.3-Flash (B200); both GPUs were checked against the Torch reference.
# GLM-5.3-Flash runs them with rms_norm_eps=1e-5 instead of RMS_EPS below; the
# eps is a scalar and does not change the launch sequence or the timing. The
# terminal head stays H200-only: GLM-5.3-Flash has no hc_head and ends with
# mhc_post -> mean over hc -> RMSNorm instead.
BOUNDARY_GPUS = ("NVIDIA H200", "NVIDIA B200")
HIDDEN_SIZE = 4096
HC_MULT = 4
RMS_EPS = 1e-6
HC_EPS = 1e-6
POST_MULTIPLIER = 2.0
SINKHORN_ITERATIONS = 20


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


def require_gpu(torch: Any, kind: str, gpus: tuple[str, ...]) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{kind} requires CUDA")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name not in gpus:
        raise ProfilerNotImplemented(f"{kind} is verified only on {gpus}, got {gpu_name}")


def prepare_common(torch: Any, shape: Shape) -> CommonInputs:
    generator = torch.Generator().manual_seed(41)
    mix_width = HC_MULT * (HC_MULT + 2)
    residual = torch.randn(
        (shape.num_tokens, HC_MULT, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        generator=generator,
    ).cuda()
    fn = (torch.randn((mix_width, HC_MULT * HIDDEN_SIZE), generator=generator) * 0.01).cuda()
    hc_scale = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).cuda()
    hc_base = torch.zeros(mix_width, dtype=torch.float32).cuda()
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


def assert_outputs_close(torch: Any, actual: tuple[Any, ...], expected: tuple[Any, ...]) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"expected {len(expected)} MHC outputs, got {len(actual)}")
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0.05, rtol=0.02)


__all__ = [
    "BOUNDARY_GPUS",
    "CommonInputs",
    "HC_EPS",
    "HC_MULT",
    "HIDDEN_SIZE",
    "POST_MULTIPLIER",
    "RMS_EPS",
    "SINKHORN_ITERATIONS",
    "Shape",
    "TERMINAL_HEAD_GPUS",
    "assert_outputs_close",
    "prepare_common",
    "reference_pre",
    "require_gpu",
    "validate_args",
]
