"""Profile vLLM's complete fused MHC-post/pre production call."""

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.mhc._deepseek_v4 import (
    BOUNDARY_GPUS,
    HC_EPS,
    POST_MULTIPLIER,
    RMS_EPS,
    SINKHORN_ITERATIONS,
    CommonInputs,
    assert_outputs_close,
    prepare_common,
    reference_pre,
    require_gpu,
    validate_args,
)

_KIND = "mhc_fused_post_pre_rms_norm:vllm_tilelang"


@dataclass(frozen=True)
class _Launch:
    callable: Any
    x: Any
    post_mix: Any
    comb_mix: Any
    inputs: CommonInputs

    def run(self):
        return self.callable(
            self.x,
            self.inputs.residual,
            self.post_mix,
            self.comb_mix,
            self.inputs.fn,
            self.inputs.hc_scale,
            self.inputs.hc_base,
            RMS_EPS,
            HC_EPS,
            HC_EPS,
            POST_MULTIPLIER,
            SINKHORN_ITERATIONS,
            n_splits=1,
            tile_n=1,
            norm_weight=self.inputs.norm_weight,
            norm_eps=RMS_EPS,
        )


def _validate_args(
    num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: DType | str
):
    return validate_args(_KIND, num_tokens, hidden_size, hc_mult, hidden_dtype)


def profile_mhc_fused_post_pre_rms_norm_vllm_tilelang(
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    hidden_dtype: DType | str,
) -> ComputeMetrics:
    shape = _validate_args(num_tokens, hidden_size, hc_mult, hidden_dtype)
    try:
        import torch
        from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang
        from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_KIND} requires pinned vLLM and TileLang") from exc

    try:
        require_gpu(torch, _KIND, BOUNDARY_GPUS)
        inputs = prepare_common(torch, shape)
        x = torch.randn((shape.num_tokens, 4096), dtype=torch.bfloat16).cuda()
        post_mix, comb_mix, _ = reference_pre(torch, inputs)
        launch = _Launch(mhc_fused_post_pre_tilelang, x, post_mix, comb_mix, inputs)
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
        time_ms = Timer.cupti(launch.run, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch.run, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_KIND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_KIND} failed") from exc

    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=0.0,
        energy_j=float(energy_j),
    )


__all__ = ["profile_mhc_fused_post_pre_rms_norm_vllm_tilelang"]
