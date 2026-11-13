"""BF16 dense linear through FlashInfer's tuned ``mm_bf16``: the
``flashinfer_cutedsl_bf16`` backend of ``single_gemm``.

vLLM serves an unquantized BF16 linear with ``dispatch_unquantized_gemm``
(``model_executor/layers/utils.py``). Under the default ``--linear-backend=auto``
that is ``F.linear`` (the ``torch_linear_vllm`` row). With
``--linear-backend=flashinfer_cutedsl`` it is ``cuda_flashinfer_bf16_gemm``:
when ``_can_use_flashinfer_cutedsl_bf16`` holds (SM10x, bf16, contiguous, k a
multiple of 128, 32-byte aligned) it calls
``vllm.utils.flashinfer.flashinfer_bf16_mm(x, weight.t(), bias, pdl, "cute-dsl")``,
i.e. ``flashinfer.mm_bf16(..., backend="cute-dsl")``. FlashInfer's AutoTuner
picks the runner and tactic per hybrid token bucket (on B200 at decode m it
lands on a tuned cuBLASLt algorithm), and vLLM tunes those buckets once at
kernel warmup and saves them in its FlashInfer autotune cache file.

The runner reproduces that: the hybrid buckets up to ``m``'s own bucket are
tuned once and persisted through ``autotune_cached`` (one tactic file per
device and FlashInfer version, named by ``tactic_labels``); every later process
loads the file, so the tactic a row is timed with is fixed by the file rather
than re-picked per process. The call is then timed outside the tuning context,
where the lookup maps ``m`` to its bucket as serving does.

Runs in ``vllm_env``: the pinned vLLM image owns FlashInfer.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners import tactic_labels
from profiling.runners.autotune_cache import autotune_cached
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

# _can_use_flashinfer_cutedsl_bf16: k % 128 == 0; anything else falls back to F.linear.
K_ALIGNMENT = 128
# bf16 output rounding plus accumulation-order differences between tactics.
REL_FROBENIUS_TOL = 1e-2


def validate(m: int, n: int, k: int, dtype: DType | str) -> tuple[int, int, int]:
    if DType.from_value(dtype) is not DType.BF16:
        raise ValueError(f"flashinfer_cutedsl_bf16 requires dtype=bf16, got {dtype!r}")
    m, n, k = int(m), int(n), int(k)
    if m < 1 or n < 1:
        raise ValueError(f"flashinfer_cutedsl_bf16 requires m, n >= 1, got m={m}, n={n}")
    if k < 1 or k % K_ALIGNMENT:
        # vLLM would run F.linear for this shape: that is the torch_linear_vllm row.
        raise ValueError(
            f"flashinfer_cutedsl_bf16 requires k to be a positive multiple of {K_ALIGNMENT}, got k={k}"
        )
    return m, n, k


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from flashinfer.autotuner import autotune
        from flashinfer.fused_moe.utils import (
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
        )
        from vllm.utils.flashinfer import flashinfer_bf16_mm
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "flashinfer_cutedsl_bf16 requires the vLLM environment (vllm_env)"
        ) from exc

    def serving_buckets(m: int) -> tuple[int, ...]:
        """The warmup's hybrid buckets, up to the one a runtime ``m`` maps to."""
        return tuple(get_hybrid_num_tokens_buckets(map_to_hybrid_bucket_uncapped(m)))

    return torch, autotune, serving_buckets, flashinfer_bf16_mm


def profile_single_gemm_flashinfer_cutedsl_bf16(
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    m, n, k = validate(m, n, k, dtype)
    torch, autotune, serving_buckets, bf16_mm = _load_runtime()
    try:
        generator = torch.Generator(device="cuda").manual_seed(0)
        x = (torch.randn(m, k, device="cuda", generator=generator) * 0.5).to(torch.bfloat16)
        # The [n, k] row-major weight a linear layer stores; vLLM passes weight.t().
        weight = (torch.randn(n, k, device="cuda", generator=generator) * 0.02).to(torch.bfloat16)
        weight_t = weight.t()
        # vLLM passes pdl=current_platform.is_arch_support_pdl(): True on SM10x.
        pdl = torch.cuda.get_device_capability()[0] >= 9

        def run_once() -> Any:
            return bf16_mm(x, weight_t, None, pdl, "cute-dsl")

        with (
            autotune_cached(autotune, tactic_labels.flashinfer_cutedsl_bf16()),
            autotune(tuning_buckets=serving_buckets(m)),
        ):
            run_once()
        out = run_once()
        torch.cuda.synchronize()
        ref = torch.nn.functional.linear(x.float(), weight.float())
        err = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()
        if not err <= REL_FROBENIUS_TOL:
            raise KernelLaunchFailed(
                f"flashinfer_cutedsl_bf16 output mismatch: relative Frobenius error {err:.3e}"
            )
        del out, ref

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError("flashinfer_cutedsl_bf16 ran out of memory") from exc
        raise KernelLaunchFailed(f"flashinfer_cutedsl_bf16 failed: {exc}") from exc

    bytes_accessed = 2 * (m * k + n * k + m * n)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * m * n * k / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["profile_single_gemm_flashinfer_cutedsl_bf16", "validate"]
