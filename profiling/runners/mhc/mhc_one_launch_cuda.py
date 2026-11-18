"""Profile the one-launch mHC boundary kernel (csrc/mhc_one_launch_{small,large}.cu).

One launch of 16-CTA clusters; CTA r of a cluster owns hidden columns
[256 r, 256 r + 256) of all four streams. Per tile of TM tokens: the post mix
and the 24 pre mixes on BF16 tensor cores (fn split into bf16 hi + lo), a
cluster barrier, the owner warp gathers the cluster's partial mixes over DSMEM
and runs the gates, a second cluster barrier, then the collapse with this
call's own pre mix and the RMSNorm (the non-shifted order, vLLM's TileLang
math). The last warps of each CTA run the Sinkhorn while the other warps
already collapse the streams.

Two builds of the same algorithm: ``small`` (TM <= 16) for
num_tokens <= SMALL_MAX_TOKENS, ``large`` (TM <= 32) above it. TM is
ceil(num_tokens / clusters), capped, with clusters the co-resident maximum.

Two entries share it:

* ``mhc_fused_post_pre_rms_norm`` backend ``cuda_one_launch``: the block
  output is one bf16 tensor, the existing kind's contract;
* ``mhc_fused_sum_post_pre_rms_norm`` backend ``cuda_one_launch``: the block
  output is the sum of ``num_partials`` bf16 partial outputs (one bf16
  rounding, the MoE combine's reduce), plus, with ``shared_addend``, one more
  bf16 tensor added after that rounding (the shared expert).

The sources are built on first use with FlashInfer's JIT (its TVM-FFI
binding), as the other in-repo CUDA runners are. CUPTI time is the sum of
every kernel in the call (``kernel_name=None``); the call is one kernel,
launched without programmatic dependent launch.
"""

import functools
import math
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
)
from profiling.runners.mhc._common import validate_args as validate_common_args

_CSRC = Path(__file__).resolve().parent / "csrc"
# The kernel is compiled for hidden 4096: 16 CTAs of 256 columns.
HIDDEN = 4096
HC_MULT = 4
CLUSTER = 16
MAX_PARTIALS = 8
SMALL_MAX_TOKENS = 96
# (source, TM cap) per build.
_BUILDS = {"small": ("mhc_one_launch_small.cu", 16), "large": ("mhc_one_launch_large.cu", 32)}
# Fast fp32 Sinkhorn (bit 2), Sinkhorn overlapped with the collapse (bit 4).
_LATE_SINKHORN = 6
_FULL_STAGE = 5


def validate_args(
    kind: str,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    hidden_dtype: DType | str,
    num_partials: int = 1,
) -> Shape:
    shape = validate_common_args(kind, num_tokens, hidden_size, hc_mult, hidden_dtype)
    if hc_mult != HC_MULT or hidden_size != HIDDEN:
        raise ProfilerNotImplemented(
            f"{kind} is built for hc_mult == {HC_MULT} and hidden_size == {HIDDEN}"
        )
    if not 1 <= num_partials <= MAX_PARTIALS:
        raise ProfilerNotImplemented(f"{kind} sums 1..{MAX_PARTIALS} partial outputs")
    return shape


def call_bytes(shape: Shape, num_partials: int = 1, shared_addend: bool = False) -> int:
    """HBM traffic of one call: one pass over the tensors, the block output read
    once per partial (and the addend). The updated streams stay in shared
    memory for the collapse, and the partial mixes move over DSMEM."""
    extra_inputs = (num_partials - 1) + int(shared_addend)
    return post_pre_one_pass_bytes(shape) + extra_inputs * hidden_bytes(shape)


@functools.cache
def _load_module(build: str):
    try:
        from flashinfer.jit.core import gen_jit_spec
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "FlashInfer JIT is required to build the one-launch kernel"
        ) from exc
    source, _ = _BUILDS[build]
    return gen_jit_spec(
        f"mhc_one_launch_{build}",
        [_CSRC / source],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"],
    ).build_and_load()


def build_for(num_tokens: int) -> str:
    return "small" if num_tokens <= SMALL_MAX_TOKENS else "large"


class OneLaunch:
    """Inputs, outputs and launch geometry of one call, allocated once."""

    def __init__(
        self,
        torch: Any,
        inputs: CommonInputs,
        parts: Any,
        shared: Any,
        post_mix: Any,
        comb_mix: Any,
    ):
        num_tokens, hidden = parts.shape[1], parts.shape[2]
        build = build_for(num_tokens)
        self.module = _load_module(build)
        self.clusters = int(self.module.m_max_clusters(CLUSTER))
        self.tile = max(1, min(_BUILDS[build][1], math.ceil(num_tokens / self.clusters)))
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
        # Written only by the ablation switches (skip != 0), which this runner leaves off.
        self.sink = torch.zeros(4, dtype=torch.int32, device="cuda")

    def run(self):
        self.module.m_fused(
            *self.args_in,
            self.new_residual,
            self.post,
            self.comb,
            self.y,
            self.sink,
            self.tile,
            CLUSTER,
            self.clusters,
            _FULL_STAGE,
            0,
            SINKHORN_ITERATIONS,
            _LATE_SINKHORN,
            0,
            RMS_EPS,
            HC_EPS,
            HC_EPS,
            POST_MULTIPLIER,
            RMS_EPS,
        )
        return self.new_residual, self.post, self.comb, self.y


def _profile(kind: str, shape: Shape, num_partials: int, shared_addend: bool) -> ComputeMetrics:
    try:
        import torch
        from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{kind} requires vLLM's Torch mHC reference (vllm_env)"
        ) from exc
    try:
        common = prepare_common(torch, shape)
        generator = torch.Generator().manual_seed(43)
        hc_base = (torch.randn(mix_width(shape.hc_mult), generator=generator) * 0.1).cuda()
        norm_weight = (
            (1.0 + 0.1 * torch.randn(shape.hidden_size, generator=generator))
            .to(torch.bfloat16)
            .cuda()
        )
        scale = num_partials**-0.5
        parts = (
            (
                torch.randn(
                    (num_partials, shape.num_tokens, shape.hidden_size), generator=generator
                )
                * scale
            )
            .to(torch.bfloat16)
            .cuda()
        )
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
        memory_bandwidth_gbps=bandwidth_gbps(
            call_bytes(shape, num_partials, shared_addend), time_ms
        ),
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
