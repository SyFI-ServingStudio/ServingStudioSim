"""Profile the after-MoE mHC boundary with the shared-expert add read by the post kernel.

vLLM's path (`mhc_fused_sum_post_pre_rms_norm:vllm_tilelang`) adds the shared expert's
output to the routed output (one bf16 add) and then runs `mhc_fused_post_pre_tilelang`.
Above the fork's fused-kernel limit (32 tokens) that call is `mhc_post_tilelang` + the
DeepGEMM prenorm GEMM + `mhc_pre_big_fuse_with_norm`; this backend replaces the add and
`mhc_post_tilelang` with `mhc_post2_tilelang` (vendored, reads both block outputs), the
other two launches unchanged. At 32 tokens or fewer it runs vLLM's path unchanged: a
small-batch kernel reading both outputs measured slower. Outputs are checked bit-for-bit
against vLLM's path before timing."""

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
    bandwidth_gbps,
    prepare_common,
    reference_pre,
    validate_args,
)
from profiling.runners.mhc.mhc_fused_post_pre_rms_norm_vllm_tilelang import call_bytes

_KIND = "mhc_fused_sum_post_pre_rms_norm:vllm_tilelang_post_addend"


def _post_addend_post_pre(routed, shared, inputs: CommonInputs, post_mix, comb_mix):
    """vLLM 5629b2614 `_mhc_fused_post_pre_tilelang_impl` with the shared add read by the
    post kernel."""
    import torch
    from vllm.model_executor.kernels.mhc.tilelang import (
        _hc_prenorm_gemm_outputs,
        mhc_fused_post_pre_tilelang,
    )
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _MHC_PRE_BIG_FUSE_TILELANG_KERNEL,
        mhc_fused_post_pre_split_config,
    )

    from profiling.runners.mhc.post_addend_vendor import mhc_post2_tilelang

    residual = inputs.residual
    hc_mult, hidden = residual.shape[-2], residual.shape[-1]
    num_tokens = residual.view(-1, hc_mult, hidden).shape[0]
    common = (
        inputs.fn,
        inputs.hc_scale,
        inputs.hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        POST_MULTIPLIER,
        SINKHORN_ITERATIONS,
    )
    if mhc_fused_post_pre_split_config(num_tokens, hidden, hc_mult) is not None:
        return mhc_fused_post_pre_tilelang(
            shared + routed,
            residual,
            post_mix,
            comb_mix,
            *common,
            n_splits=1,
            tile_n=1,
            norm_weight=inputs.norm_weight,
            norm_eps=RMS_EPS,
        )
    residual_flat = residual.view(num_tokens, hc_mult, hidden)
    residual_cur = torch.empty_like(residual_flat)
    mhc_post2_tilelang(
        comb_mix.view(num_tokens, hc_mult, hc_mult),
        residual_flat,
        post_mix.view(num_tokens, hc_mult),
        shared.view(num_tokens, hidden),
        routed.view(num_tokens, hidden),
        residual_cur,
        hc_mult,
        hidden,
    )
    gemm_out_mul, gemm_out_sqrsum = _hc_prenorm_gemm_outputs(
        residual_cur.view(num_tokens, hc_mult * hidden),
        inputs.fn,
        hidden_size=hidden,
        hc_mult=hc_mult,
    )
    post_mix_cur = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=residual.device)
    comb_mix_cur = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input_cur = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=residual.device)
    norm_weight = inputs.norm_weight
    if norm_weight is not None and norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)
    _MHC_PRE_BIG_FUSE_TILELANG_KERNEL(
        gemm_out_mul,
        gemm_out_sqrsum,
        inputs.hc_scale,
        inputs.hc_base,
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        POST_MULTIPLIER,
        SINKHORN_ITERATIONS,
        norm_weight=norm_weight,
        norm_eps=RMS_EPS,
    )
    outer = residual.shape[:-2]
    return (
        residual_cur.view(*outer, hc_mult, hidden),
        post_mix_cur.view(*outer, hc_mult, 1),
        comb_mix_cur.view(*outer, hc_mult, hc_mult),
        layer_input_cur.view(*outer, hidden),
    )


def _incumbent(routed, shared, inputs: CommonInputs, post_mix, comb_mix):
    """vLLM's path: `shared_output + fused_output` (moe_runner.py), then the fused call."""
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang

    return mhc_fused_post_pre_tilelang(
        shared + routed,
        inputs.residual,
        post_mix,
        comb_mix,
        inputs.fn,
        inputs.hc_scale,
        inputs.hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        POST_MULTIPLIER,
        SINKHORN_ITERATIONS,
        n_splits=1,
        tile_n=1,
        norm_weight=inputs.norm_weight,
        norm_eps=RMS_EPS,
    )


@dataclass(frozen=True)
class _Launch:
    callable: Any
    routed: Any
    shared: Any
    inputs: CommonInputs
    post_mix: Any
    comb_mix: Any

    def run(self):
        return self.callable(self.routed, self.shared, self.inputs, self.post_mix, self.comb_mix)


def _bits_equal(torch, a, b) -> bool:
    if a.dtype == torch.bfloat16:
        return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def profile_mhc_fused_sum_post_pre_rms_norm_vllm_tilelang_post_addend(
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    hidden_dtype: DType | str,
    num_partials: int,
    shared_addend: bool,
) -> ComputeMetrics:
    """vLLM's after-MoE boundary with the shared add read by the post kernel
    (``num_partials`` 1 with ``shared_addend``)."""
    shape = validate_args(_KIND, num_tokens, hidden_size, hc_mult, hidden_dtype)
    if num_partials != 1 or not shared_addend:
        raise ProfilerNotImplemented(
            f"{_KIND}: one reduced routed output plus the shared expert's output only"
        )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_KIND} requires pinned vLLM and TileLang") from exc
    try:
        inputs = prepare_common(torch, shape)
        routed = (
            (torch.randn((shape.num_tokens, shape.hidden_size)) * 2.0).to(torch.bfloat16).cuda()
        )
        shared = (
            (torch.randn((shape.num_tokens, shape.hidden_size)) * 0.7).to(torch.bfloat16).cuda()
        )
        post_mix, comb_mix, _ = reference_pre(torch, inputs)
        launch = _Launch(_post_addend_post_pre, routed, shared, inputs, post_mix, comb_mix)
        actual = launch.run()
        expected = _incumbent(routed, shared, inputs, post_mix, comb_mix)
        again = launch.run()
        torch.cuda.synchronize()
        for name, a, e, r in zip(
            ("residual", "post_mix", "comb_mix", "layer_input"), actual, expected, again
        ):
            if not _bits_equal(torch, a, e):
                raise AssertionError(
                    f"{_KIND}: {name} differs from vLLM's add + fused call at {num_tokens} tokens"
                )
            if not _bits_equal(torch, a, r):
                raise AssertionError(f"{_KIND}: {name} not deterministic at {num_tokens} tokens")
        time_ms = Timer.cupti(launch.run, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch.run, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_KIND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_KIND} failed") from exc
    # The post kernel reads the second block output (M x hidden bf16) instead of the
    # add's 2 reads + 1 write.
    extra = shape.num_tokens * shape.hidden_size * 2
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=bandwidth_gbps(call_bytes(shape) + extra, time_ms),
        energy_j=float(energy_j),
    )


__all__ = ["profile_mhc_fused_sum_post_pre_rms_norm_vllm_tilelang_post_addend"]
