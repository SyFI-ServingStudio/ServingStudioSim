"""Profile the one-launch mHC boundary kernel (csrc/mhc_post_pre_one_launch.cu).

One cooperative launch does the mHC post of the finished block, the next
block's pre (the 24-wide mix GEMM on BF16 tensor cores with fn split into bf16
hi + lo, the gates and the Sinkhorn comb) and its RMSNorm: the math of vLLM's
``mhc_fused_post_pre_tilelang`` with ``norm_weight``. Phase A splits the
GEMM's K over hidden / 32 CTAs and stores each CTA's partial mixes; one grid
barrier; phase B reduces them per token, runs the gates and collapses the
updated streams with this call's own pre mix (the non-shifted order).

Two entries share it:

* ``mhc_fused_post_pre_rms_norm`` backend ``cuda_one_launch``: the block
  output is one bf16 tensor, the existing kind's contract;
* ``mhc_fused_sum_post_pre_rms_norm`` backend ``cuda_one_launch``: the block
  output is the sum of ``num_partials`` bf16 partial outputs (one bf16
  rounding, the MoE combine's reduce), plus, with ``shared_addend``, one more
  bf16 tensor added after that rounding (the shared expert).

The source is built on first use with FlashInfer's JIT (its TVM-FFI binding),
as the other in-repo CUDA runners are. CUPTI time is the sum of every kernel in
the call (``kernel_name=None``); the call is one kernel.
"""

import functools
from pathlib import Path
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.mhc._common import (
    HC_EPS,
    POST_MULTIPLIER,
    RMS_EPS,
    SINKHORN_ITERATIONS,
    CommonInputs,
    Shape,
    assert_outputs_close,
    bandwidth_gbps,
    hidden_bytes,
    mix_width,
    post_pre_one_pass_bytes,
    prepare_common,
    reference_pre,
    residual_bytes,
)
from profiling.runners.mhc._common import validate_args as validate_common_args

_SOURCE = Path(__file__).resolve().parent / "csrc" / "mhc_post_pre_one_launch.cu"
_JIT_MODULE_NAME = "mhc_post_pre_one_launch"
# Phase A gives each CTA 32 hidden columns of all four streams; the grid is
# hidden / 32 CTAs and must be co-resident (cooperative launch).
_SLICE = 32
HC_MULT = 4
MAX_PARTIALS = 8


def validate_args(
    kind: str,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    hidden_dtype: DType | str,
    num_partials: int = 1,
) -> Shape:
    shape = validate_common_args(kind, num_tokens, hidden_size, hc_mult, hidden_dtype)
    if hc_mult != HC_MULT or hidden_size % _SLICE != 0:
        raise ProfilerNotImplemented(
            f"{kind} requires hc_mult == {HC_MULT} and hidden_size % {_SLICE} == 0"
        )
    if not 1 <= num_partials <= MAX_PARTIALS:
        raise ProfilerNotImplemented(f"{kind} sums 1..{MAX_PARTIALS} partial outputs")
    return shape


def call_bytes(shape: Shape, num_partials: int = 1, shared_addend: bool = False) -> int:
    """HBM traffic of one call: one pass over the tensors, the block output read
    once per partial (and the addend), plus phase B's re-read of the updated
    streams. The partial-mix workspace (128 B per token per CTA, L2-resident)
    is left out."""
    extra_inputs = (num_partials - 1) + int(shared_addend)
    return (
        post_pre_one_pass_bytes(shape)
        + extra_inputs * hidden_bytes(shape)
        + residual_bytes(shape)
    )


@functools.cache
def _load_module():
    try:
        from flashinfer.jit.core import gen_jit_spec
    except ImportError as exc:
        raise ProfilerNotImplemented("FlashInfer JIT is required to build the one-launch kernel") from exc
    return gen_jit_spec(
        _JIT_MODULE_NAME,
        [_SOURCE],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"],
    ).build_and_load()


class OneLaunch:
    """Inputs, outputs and workspace of one call, allocated once."""

    def __init__(self, torch: Any, inputs: CommonInputs, parts: Any, shared: Any, post_mix: Any, comb_mix: Any):
        self.module = _load_module()
        num_tokens, hidden = parts.shape[1], parts.shape[2]
        self.args_in = (
            parts,
            shared if shared is not None else torch.empty(0, dtype=torch.bfloat16, device="cuda"),
            inputs.residual,
            post_mix,
            comb_mix,
            inputs.fn,
            inputs.hc_scale,
            inputs.hc_base,
            inputs.norm_weight,
        )
        self.new_residual = torch.empty_like(inputs.residual)
        self.post = torch.empty((num_tokens, HC_MULT, 1), dtype=torch.float32, device="cuda")
        self.comb = torch.empty((num_tokens, HC_MULT, HC_MULT), dtype=torch.float32, device="cuda")
        self.y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        floats = int(self.module.mhc_post_pre_one_launch_workspace_floats(num_tokens, hidden))
        self.workspace = torch.empty(floats, dtype=torch.float32, device="cuda")

    def run(self):
        self.module.mhc_post_pre_one_launch(
            *self.args_in,
            self.new_residual,
            self.post,
            self.comb,
            self.y,
            self.workspace,
            RMS_EPS,
            HC_EPS,
            HC_EPS,
            POST_MULTIPLIER,
            RMS_EPS,
            SINKHORN_ITERATIONS,
            5,
        )
        return self.new_residual, self.post, self.comb, self.y


def _profile(kind: str, shape: Shape, num_partials: int, shared_addend: bool) -> ComputeMetrics:
    try:
        import torch
        from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{kind} requires vLLM's Torch mHC reference (vllm_env)") from exc
    try:
        common = prepare_common(torch, shape)
        generator = torch.Generator().manual_seed(43)
        hc_base = (torch.randn(mix_width(shape.hc_mult), generator=generator) * 0.1).cuda()
        norm_weight = (
            (1.0 + 0.1 * torch.randn(shape.hidden_size, generator=generator)).to(torch.bfloat16).cuda()
        )
        scale = num_partials**-0.5
        parts = (
            torch.randn((num_partials, shape.num_tokens, shape.hidden_size), generator=generator) * scale
        ).to(torch.bfloat16).cuda()
        shared = (
            (torch.randn((shape.num_tokens, shape.hidden_size), generator=generator) * 0.5)
            .to(torch.bfloat16)
            .cuda()
            if shared_addend
            else None
        )
        inputs = CommonInputs(common.residual, common.fn, common.hc_scale, hc_base, norm_weight)
        post_mix, comb_mix, _ = reference_pre(torch, inputs)
        post_mix, comb_mix = post_mix.contiguous(), comb_mix.contiguous()
        launch = OneLaunch(torch, inputs, parts, shared, post_mix, comb_mix)

        # Unfused reference: the combine sum (fp32 accumulate, one bf16
        # rounding), the shared add in bf16, then vLLM's Torch post and pre
        # and the RMSNorm.
        x = parts.float().sum(0).to(torch.bfloat16)
        if shared is not None:
            x = (x.float() + shared.float()).to(torch.bfloat16)
        expected_residual = mhc_post_torch(x, inputs.residual, post_mix, comb_mix)
        expected = (
            expected_residual,
            *reference_pre(
                torch,
                CommonInputs(expected_residual, inputs.fn, inputs.hc_scale, hc_base, norm_weight),
            ),
        )
        actual = launch.run()
        torch.cuda.synchronize()
        assert_outputs_close(torch, actual, expected)
        del actual, expected, expected_residual, x
        time_ms = Timer.cupti(launch.run, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch.run, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{kind} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{kind} failed: {exc}") from exc
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=bandwidth_gbps(call_bytes(shape, num_partials, shared_addend), time_ms),
        energy_j=float(energy_j),
    )


def profile_mhc_fused_post_pre_rms_norm_cuda_one_launch(
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    hidden_dtype: DType | str,
) -> ComputeMetrics:
    kind = "mhc_fused_post_pre_rms_norm:cuda_one_launch"
    shape = validate_args(kind, num_tokens, hidden_size, hc_mult, hidden_dtype)
    return _profile(kind, shape, 1, False)


def profile_mhc_fused_sum_post_pre_rms_norm_cuda_one_launch(
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    hidden_dtype: DType | str,
    num_partials: int,
    shared_addend: bool,
) -> ComputeMetrics:
    kind = "mhc_fused_sum_post_pre_rms_norm:cuda_one_launch"
    shape = validate_args(kind, num_tokens, hidden_size, hc_mult, hidden_dtype, num_partials)
    return _profile(kind, shape, num_partials, bool(shared_addend))


__all__ = [
    "OneLaunch",
    "call_bytes",
    "profile_mhc_fused_post_pre_rms_norm_cuda_one_launch",
    "profile_mhc_fused_sum_post_pre_rms_norm_cuda_one_launch",
    "validate_args",
]
