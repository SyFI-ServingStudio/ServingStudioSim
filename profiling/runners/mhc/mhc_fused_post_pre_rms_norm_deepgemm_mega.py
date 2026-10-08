"""Profile DeepGEMM's Mega mHC shifted post/pre/RMSNorm call.

Source: alignment fork ``servingstudio-alignment-v41``. Every mHC boundary
between two sublayers (except layer 0's attention pre and the engram layers'
pre) goes through ``models/deepseek_v41/nvidia/ops/mhc.py:mhc_shifted_post_pre``.
Outside the T<=16 FULL-cudagraph overlap path it calls
``ops/mega_mhc.py:mhc_shifted_post_pre_deep_gemm``, which issues one
``deep_gemm.mega_mhc`` launch (``sm100_mega_mhc_impl<hidden, splits, ...,
kIsShifted=true, kStoreBF16=true, ...>``). That persistent kernel fuses:

1. post of the previous sublayer:
   ``new_residual = post_mix * x + comb_res_mixᵀ @ residual``;
2. the next pre's TF32 GEMM ``flatten(new_residual) @ fnᵀ`` (24 mixes over
   ``hc_mult * hidden``) with its RMS statistic, split over K;
3. the mix epilogue: sigmoid pre/post gates and the 20-step Sinkhorn comb;
4. the shifted collapse ``sum_h shifted_prev_mix[h] * new_residual[h]`` and the
   BF16 RMSNorm of it.

"Shifted" means the layer input is collapsed with the pre-mix carried from the
previous sublayer, and this call's pre gate is returned for the next one. The
work per token is the same as the TileLang fused post/pre; only which pre-mix
feeds the collapse differs, and the carried mix adds 16 bytes per token.

The split count (the second template argument) is picked by DeepGEMM's host
heuristic from ``num_tokens``, so one args schema covers every specialization.
Measured on B200: 40 splits up to T=192 (the decode ``<5120, 40>``), then 27,
20, and 16 from T~449 on (the large-T ``<5120, 16>``); the time steps at each
change.

CUPTI time is the sum of every kernel in the call (``kernel_name=None``).
"""

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_KIND = "mhc_fused_post_pre_rms_norm:deepgemm_mega"

# DeepSeek-V4.1-Flash text_config (hidden_size, hc_mult, hc_sinkhorn_iters,
# hc_eps, rms_norm_eps); the fork fixes hc_post_alpha at 2.0. The eps values are
# kernel scalars and do not change the launch.
HIDDEN_SIZE = 5120
HC_MULT = 4
RMS_EPS = 1e-20
HC_EPS = 1e-6
POST_MULTIPLIER = 2.0
SINKHORN_ITERATIONS = 20
# ``can_use_mega_mhc`` routes larger batches to the TileLang fallback.
MAX_TOKENS = 1 << 20
# Per-element tolerance of the existing kind's Torch check. The kernel uses a
# TF32 GEMM for the mixes and rounds BF16 outputs.
ATOL = 0.05
RTOL = 0.02


@dataclass(frozen=True)
class _Inputs:
    x: Any
    residual: Any
    shifted_prev_mix: Any
    post_mix: Any
    comb_res_mix: Any
    fn: Any
    hc_scale: Any
    hc_base: Any
    norm_weight: Any


@dataclass(frozen=True)
class _Launch:
    callable: Any
    inputs: _Inputs

    def run(self):
        i = self.inputs
        return self.callable(
            i.x,
            i.residual,
            i.shifted_prev_mix,
            i.post_mix,
            i.comb_res_mix,
            i.fn,
            i.hc_scale,
            i.hc_base,
            RMS_EPS,
            HC_EPS,
            POST_MULTIPLIER,
            HC_EPS,
            SINKHORN_ITERATIONS,
            i.norm_weight,
            RMS_EPS,
        )


def validate_args(
    num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: DType | str
) -> int:
    if type(num_tokens) is not int or num_tokens <= 0:
        raise ValueError("num_tokens must be a positive integer")
    if num_tokens > MAX_TOKENS:
        raise ProfilerNotImplemented(f"{_KIND} is dispatched only for num_tokens <= {MAX_TOKENS}")
    if hidden_size != HIDDEN_SIZE or hc_mult != HC_MULT:
        raise ProfilerNotImplemented(
            f"{_KIND} requires hidden_size={HIDDEN_SIZE} and hc_mult={HC_MULT}"
        )
    if DType.from_value(hidden_dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_KIND} requires hidden_dtype=bf16")
    return num_tokens


def _prepare(torch: Any, num_tokens: int, reference_pre: Any) -> _Inputs:
    """Build a residual state and the coefficients the previous pre produced.

    The carried mixes come from the fork's Torch pre on an earlier residual, so
    the post and comb coefficients are the sigmoid gates and the doubly
    stochastic Sinkhorn matrix the kernel sees in production.
    """
    generator = torch.Generator().manual_seed(41)
    mix_width = HC_MULT * (HC_MULT + 2)
    shape = (num_tokens, HC_MULT, HIDDEN_SIZE)
    previous = torch.randn(shape, dtype=torch.bfloat16, generator=generator).cuda()
    residual = torch.randn(shape, dtype=torch.bfloat16, generator=generator).cuda()
    x = torch.randn((num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, generator=generator).cuda()
    fn = (torch.randn((mix_width, HC_MULT * HIDDEN_SIZE), generator=generator) * 0.01).cuda()
    hc_scale = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).cuda()
    hc_base = (torch.randn(mix_width, generator=generator) * 0.1).cuda()
    norm_weight = (
        (1.0 + 0.1 * torch.randn(HIDDEN_SIZE, generator=generator)).to(torch.bfloat16).cuda()
    )
    post_mix, comb_res_mix, _, pre_mix = reference_pre(
        previous,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        POST_MULTIPLIER,
        SINKHORN_ITERATIONS,
    )
    return _Inputs(
        x=x,
        residual=residual,
        shifted_prev_mix=pre_mix.contiguous(),
        post_mix=post_mix.contiguous(),
        comb_res_mix=comb_res_mix.contiguous(),
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        norm_weight=norm_weight,
    )


def _reference(torch: Any, inputs: _Inputs, reference_post: Any, reference_pre: Any):
    new_residual = reference_post(inputs.x, inputs.residual, inputs.post_mix, inputs.comb_res_mix)
    post_mix, comb_res_mix, layer_input, pre_mix = reference_pre(
        new_residual,
        inputs.fn,
        inputs.hc_scale,
        inputs.hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        POST_MULTIPLIER,
        SINKHORN_ITERATIONS,
        pre_mix=inputs.shifted_prev_mix,
    )
    normalized = torch.nn.functional.rms_norm(
        layer_input.float(), (HIDDEN_SIZE,), inputs.norm_weight.float(), RMS_EPS
    ).to(torch.bfloat16)
    return new_residual, post_mix, comb_res_mix, normalized, pre_mix


def assert_outputs_close(torch: Any, actual: tuple[Any, ...], expected: tuple[Any, ...]) -> None:
    names = ("new_residual", "post_mix", "comb_res_mix", "layer_input", "next_pre_mix")
    if len(actual) != len(expected):
        raise AssertionError(f"expected {len(expected)} mHC outputs, got {len(actual)}")
    for name, actual_tensor, expected_tensor in zip(names, actual, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor.float(),
            expected_tensor.float().view_as(actual_tensor),
            atol=ATOL,
            rtol=RTOL,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def profile_mhc_fused_post_pre_rms_norm_deepgemm_mega(
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    hidden_dtype: DType | str,
) -> ComputeMetrics:
    num_tokens = validate_args(num_tokens, hidden_size, hc_mult, hidden_dtype)
    try:
        import torch
        from vllm.model_executor.kernels.mhc.torch import (
            mhc_post_torch,
            mhc_pre_delayed_torch,
        )
        from vllm.models.deepseek_v41.nvidia.ops.mega_mhc import (
            is_mega_mhc_supported,
            mhc_shifted_post_pre_deep_gemm,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{_KIND} requires the upstream-rebased vLLM fork (vllm_upstream_fork_env)"
        ) from exc

    try:
        if not is_mega_mhc_supported(HIDDEN_SIZE, HC_MULT):
            raise ProfilerNotImplemented(f"{_KIND}: the fork's DeepGEMM has no usable mega_mhc")
        inputs = _prepare(torch, num_tokens, mhc_pre_delayed_torch)
        launch = _Launch(mhc_shifted_post_pre_deep_gemm, inputs)
        expected = _reference(torch, inputs, mhc_post_torch, mhc_pre_delayed_torch)
        actual = launch.run()
        torch.cuda.synchronize()
        assert_outputs_close(torch, actual, expected)
        del actual, expected
        time_ms = Timer.cupti(launch.run, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch.run, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_KIND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_KIND} failed: {exc}") from exc

    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=0.0,
        energy_j=float(energy_j),
    )


__all__ = ["profile_mhc_fused_post_pre_rms_norm_deepgemm_mega", "validate_args"]
