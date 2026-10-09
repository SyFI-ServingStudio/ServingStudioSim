"""Profile DeepGEMM's Mega mHC post/pre/RMSNorm call without the shifted collapse.

``deep_gemm.mega_mhc`` called with ``shifted_prev_mix=None`` and
``new_prev_mix=None`` launches ``sm100_mega_mhc_impl<hidden, splits, num_sms,
kIsShifted=false, kStoreBF16=true, ...>``: one persistent kernel that does the
same math as vLLM's ``mhc_fused_post_pre_tilelang``:

1. post of the finished block:
   ``new_residual = post_mix * x + comb_res_mixᵀ @ residual``;
2. the next pre's TF32 GEMM ``flatten(new_residual) @ fnᵀ`` (``hc_mult * (hc_mult
   + 2)`` mixes over ``hc_mult * hidden``) with its RMS statistic, split over K;
3. the mix epilogue: sigmoid pre/post gates and the Sinkhorn comb;
4. the collapse ``sum_h pre_mix[h] * new_residual[h]`` with this call's own pre
   mix, and the BF16 RMSNorm of it.

Step 4 waits for the mixes of step 3, so its "Normal" workers read the updated
streams back from global memory (``ld_evict_first`` in
``epilogue/sm100_mega_mhc.cuh:run_normal_worker``); the shifted variant
(``deepgemm_mega``) collapses with a mix known up front and skips that read.

The call is DeepGEMM's public ``mega_mhc``, as vLLM re-exports it from
``vllm.utils.deep_gemm``. No vLLM model dispatches the non-shifted form yet; the
alignment fork (rebased on upstream 04730e8) vendors the DeepGEMM build that
has it, hence ``vllm_upstream_fork_env``.

DeepGEMM picks the split count (the second template argument) on the host from
``num_tokens``, so one args schema covers every specialization; the time steps
where it changes and at each extra wave of a split count.

CUPTI time is the sum of every kernel in the call (``kernel_name=None``).
"""

from dataclasses import dataclass
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
    mix_width,
    post_pre_one_pass_bytes,
    prepare_common,
    reference_pre,
    residual_bytes,
)
from profiling.runners.mhc._common import (
    validate_args as validate_common_args,
)

_KIND = "mhc_fused_post_pre_rms_norm:deepgemm_mega_nonshifted"

# DeepGEMM's mega_mhc host checks, as vLLM's is_mega_mhc_supported() states
# them: hidden_size a multiple of 1024 and four streams. Batches above 2^20
# tokens go to the TileLang fallback (can_use_mega_mhc).
HIDDEN_SIZE_MULTIPLE = 1024
HC_MULT = 4
MAX_TOKENS = 1 << 20


def validate_args(
    num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: DType | str
) -> Shape:
    shape = validate_common_args(_KIND, num_tokens, hidden_size, hc_mult, hidden_dtype)
    if num_tokens > MAX_TOKENS:
        raise ProfilerNotImplemented(f"{_KIND} is dispatched only for num_tokens <= {MAX_TOKENS}")
    if hidden_size % HIDDEN_SIZE_MULTIPLE != 0 or hc_mult != HC_MULT:
        raise ProfilerNotImplemented(
            f"{_KIND} requires hidden_size % {HIDDEN_SIZE_MULTIPLE} == 0 and hc_mult == {HC_MULT}"
        )
    return shape


def call_bytes(shape: Shape) -> int:
    """HBM traffic of the one launch: one pass plus the Normal workers' re-read
    of the updated streams. The K-split partials (about 100 B per token per
    split) are left out."""
    return post_pre_one_pass_bytes(shape) + residual_bytes(shape)


@dataclass(frozen=True)
class _Launch:
    mega_mhc: Any
    torch: Any
    x: Any
    post_mix: Any
    comb_mix: Any
    inputs: CommonInputs

    def run(self):
        i = self.inputs
        new_residual = self.torch.empty_like(i.residual)
        new_post_mix = self.torch.empty_like(self.post_mix)
        new_comb_mix = self.torch.empty_like(self.comb_mix)
        y = self.x.new_empty(self.x.shape, dtype=self.torch.bfloat16)
        self.mega_mhc(
            x=self.x,
            residual=i.residual,
            shifted_prev_mix=None,
            post_mix=self.post_mix,
            comb_res_mix=self.comb_mix,
            fn=i.fn,
            mix_scales=i.hc_scale,
            mix_bases=i.hc_base,
            hc_mult=i.residual.shape[1],
            hc_norm_eps=RMS_EPS,
            hc_pre_eps=HC_EPS,
            hc_post_scale=POST_MULTIPLIER,
            sinkhorn_eps=HC_EPS,
            num_sinkhorn_iters=SINKHORN_ITERATIONS,
            rmsnorm_weight=i.norm_weight,
            rmsnorm_eps=RMS_EPS,
            rmsnorm_scale=1.0,
            new_residual=new_residual,
            new_prev_mix=None,
            new_post_mix=new_post_mix,
            new_comb_res_mix=new_comb_mix,
            y_bf16=y,
        )
        return new_residual, new_post_mix, new_comb_mix, y


def _prepare(torch: Any, shape: Shape) -> tuple[CommonInputs, Any]:
    """The shared mHC inputs with a non-zero mix bias and a non-unit norm
    weight, so the check also covers the kernel applying both."""
    common = prepare_common(torch, shape)
    generator = torch.Generator().manual_seed(43)
    hc_base = (torch.randn(mix_width(shape.hc_mult), generator=generator) * 0.1).cuda()
    norm_weight = (
        (1.0 + 0.1 * torch.randn(shape.hidden_size, generator=generator)).to(torch.bfloat16).cuda()
    )
    x = torch.randn(
        (shape.num_tokens, shape.hidden_size), dtype=torch.bfloat16, generator=generator
    ).cuda()
    inputs = CommonInputs(common.residual, common.fn, common.hc_scale, hc_base, norm_weight)
    return inputs, x


def profile_mhc_fused_post_pre_rms_norm_deepgemm_mega_nonshifted(
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    hidden_dtype: DType | str,
) -> ComputeMetrics:
    shape = validate_args(num_tokens, hidden_size, hc_mult, hidden_dtype)
    try:
        import torch
        from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
        from vllm.utils.deep_gemm import _import_deep_gemm, is_deep_gemm_supported, mega_mhc
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{_KIND} requires the upstream-rebased vLLM fork (vllm_upstream_fork_env)"
        ) from exc

    try:
        deep_gemm = _import_deep_gemm()
        if not (
            is_deep_gemm_supported()
            and deep_gemm is not None
            and callable(getattr(deep_gemm, "mega_mhc", None))
        ):
            raise ProfilerNotImplemented(f"{_KIND}: the fork's DeepGEMM has no usable mega_mhc")
        inputs, x = _prepare(torch, shape)
        post_mix, comb_mix, _ = reference_pre(torch, inputs)
        launch = _Launch(mega_mhc, torch, x, post_mix.contiguous(), comb_mix.contiguous(), inputs)
        expected_residual = mhc_post_torch(x, inputs.residual, post_mix, comb_mix)
        expected_inputs = CommonInputs(
            expected_residual,
            inputs.fn,
            inputs.hc_scale,
            inputs.hc_base,
            inputs.norm_weight,
        )
        expected = (expected_residual, *reference_pre(torch, expected_inputs))
        actual = launch.run()
        torch.cuda.synchronize()
        assert_outputs_close(torch, actual, expected)
        del actual, expected, expected_inputs, expected_residual
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
        memory_bandwidth_gbps=bandwidth_gbps(call_bytes(shape), time_ms),
        energy_j=float(energy_j),
    )


__all__ = [
    "call_bytes",
    "profile_mhc_fused_post_pre_rms_norm_deepgemm_mega_nonshifted",
    "validate_args",
]
