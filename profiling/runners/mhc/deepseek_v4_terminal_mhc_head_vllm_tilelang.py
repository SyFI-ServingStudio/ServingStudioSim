"""Profile the production DeepSeek V4 terminal hidden-state transform."""

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.mhc._deepseek_v4 import (
    HC_EPS,
    RMS_EPS,
    CommonInputs,
    prepare_common,
    reference_pre,
    require_h200,
    validate_args,
)

_KIND = "deepseek_v4_terminal_mhc_head:vllm_tilelang"


@dataclass(frozen=True)
class _Launch:
    mhc_post: Any
    hc_head: Any
    norm: Any
    x: Any
    post_mix: Any
    comb_mix: Any
    inputs: CommonInputs
    head_fn: Any
    head_scale: Any
    head_base: Any
    mtp_buffer: Any

    def run(self):
        hidden = self.mhc_post(
            self.x, self.inputs.residual, self.post_mix, self.comb_mix
        )
        self.mtp_buffer.copy_(hidden.flatten(1))
        hidden = self.hc_head(
            hidden,
            self.head_fn,
            self.head_scale,
            self.head_base,
            RMS_EPS,
            HC_EPS,
        )
        return self.norm(hidden)


def profile_deepseek_v4_terminal_mhc_head_vllm_tilelang(
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    hidden_dtype: DType | str,
) -> ComputeMetrics:
    shape = validate_args(_KIND, num_tokens, hidden_size, hc_mult, hidden_dtype)
    try:
        import torch
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.kernels.mhc.tilelang import (
            hc_head_fused_kernel_tilelang,
            mhc_post_tilelang,
        )
        from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
        from vllm.model_executor.layers.layernorm import RMSNorm
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_KIND} requires pinned vLLM and TileLang") from exc

    try:
        require_h200(torch, _KIND)
        inputs = prepare_common(torch, shape)
        x = torch.randn(
            (shape.num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        post_mix, comb_mix, _ = reference_pre(torch, inputs)
        # CustomOp dispatch is fixed when the module is constructed. A full
        # engine owns this context; the standalone production runner must
        # supply the same default configuration outside the timed closure.
        with set_current_vllm_config(VllmConfig()):
            norm = RMSNorm(hidden_size, eps=RMS_EPS).cuda()
        norm.weight.data.fill_(1.0)
        head_fn = inputs.fn[:4].contiguous()
        head_scale = inputs.hc_scale[:1].contiguous()
        head_base = inputs.hc_base[:4].contiguous()
        mtp_buffer = torch.empty_like(inputs.residual).flatten(1)
        launch = _Launch(
            mhc_post_tilelang,
            hc_head_fused_kernel_tilelang,
            norm,
            x,
            post_mix,
            comb_mix,
            inputs,
            head_fn,
            head_scale,
            head_base,
            mtp_buffer,
        )
        post = mhc_post_torch(x, inputs.residual, post_mix, comb_mix)
        post_float = post.float()
        flattened = post_float.flatten(1)
        inverse_rms = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + RMS_EPS)
        logits = flattened @ head_fn.t()
        weights = torch.sigmoid(logits * inverse_rms * head_scale + head_base) + HC_EPS
        expected = torch.sum(post_float * weights.unsqueeze(-1), dim=1).to(torch.bfloat16)
        expected = torch.nn.functional.rms_norm(
            expected, (hidden_size,), norm.weight, RMS_EPS
        )
        actual = launch.run()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.02)
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


__all__ = ["profile_deepseek_v4_terminal_mhc_head_vllm_tilelang"]
