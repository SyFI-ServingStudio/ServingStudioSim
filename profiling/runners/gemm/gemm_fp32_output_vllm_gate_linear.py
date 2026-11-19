"""Profile vLLM's ``GateLinear`` router GEMM for a bf16 weight and fp32 logits.

``GateLinear.forward`` (``vllm/model_executor/layers/fused_moe/router/
gate_linear.py``), the MoE gate of GLM-5.3-Flash (``glm5next``), picks by token
count. With a bf16 weight, fp32 ``out_dtype``, no bias and SM90+:

- tier 1, ``m <= GateLinear.LL_BF16_MAX_TOKENS`` (16) and cute-DSL importable:
  ``ll_bf16.ll_bf16_gemm(x, weight)``;
- tiers 2 and 3 need an fp32 weight and do not apply;
- tier 4, otherwise: ``torch.mm(x, weight.T, out_dtype=torch.float32)``.

The runner restates that choice with the class's own threshold rather than
constructing the layer (which needs a vLLM config context). A tier-1 shape on a
host without cute-DSL raises ``ProfilerNotImplemented`` instead of timing the
cuBLAS fallback under this label. CUPTI counts every launch of one call,
overlapping launches once.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.gemm.gemm_fp32_output_vllm_ll_bf16 import _check_output, _validate_args
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gemm_fp32_output:vllm_gate_linear"


def profile_gemm_fp32_output_vllm_gate_linear(
    m: int,
    n: int,
    k: int,
    input_dtype: DType | str,
) -> ComputeMetrics:
    m, n, k = _validate_args(m, n, k, input_dtype)
    try:
        import torch
        from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16
        from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
        from vllm.platforms import current_platform
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires vLLM's GateLinear (vllm_env)") from exc
    if not current_platform.has_device_capability(90):
        raise ProfilerNotImplemented(f"{_BACKEND} models the SM90+ tiers only")
    tier1 = m <= GateLinear.LL_BF16_MAX_TOKENS
    if tier1 and not ll_bf16.is_available():
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires cute-DSL for m <= {GateLinear.LL_BF16_MAX_TOKENS}"
        )
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(31)
        hidden = torch.randn((m, k), dtype=torch.bfloat16, device=device, generator=generator)
        weight = torch.randn((n, k), dtype=torch.bfloat16, device=device, generator=generator)

        if tier1:

            def launch():
                return ll_bf16.ll_bf16_gemm(hidden, weight)

        else:

            def launch():
                return torch.mm(hidden, weight.T, out_dtype=torch.float32)

        # First call compiles the cute-DSL kernel for this shape.
        first = launch()
        torch.cuda.synchronize()
        _check_output(torch, hidden, weight, first)
        time_ms = Timer.cupti(launch, warmup=5, interval_union=True)
        energy_j = Energy.perf(launch, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    elapsed_s = time_ms / 1000.0
    flops = 2 * m * n * k
    logical_bytes = 2 * m * k + 2 * n * k + 4 * m * n
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / elapsed_s / 1e12 if elapsed_s else 0.0),
        memory_bandwidth_gbps=float(logical_bytes / elapsed_s / 1e9 if elapsed_s else 0.0),
        energy_j=float(energy_j),
    )


__all__ = ["profile_gemm_fp32_output_vllm_gate_linear"]
