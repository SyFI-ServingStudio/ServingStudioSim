"""DeepInfer's fused sampler -- ``tilelang`` backend of ``fused_sample``.

The timed call is ``tilelang_ops.fused_sample.fused_sample(logits,
SamplingKernelInputs(LogitsTransformInputs(temperatures, top_ps, top_ks),
seq_seeds, sample_positions), tag=...)`` as ``sampler_utils.process_sampling``
issues it: on the Markov steps without a draft distribution (tag "mtp"), on
the verify rows with one (``is_verify``, the draft logprobs, the draft token ids
and ``mtp_info_read_rows``). Logits are contiguous fp32 rows, seeded
standard-normal; temperature 1; top_p 0.95 when the top-p filter is active,
else 1; top_k 50 when the top-k filter is active, else -1. Verify rows mark
every row as a verify row against a seeded draft log-softmax.

Check: DeepInfer's own ``_check_top_p_top_k`` on the returned logprobs (the
surviving set is the minimal top-p/top-k prefix and its probabilities are the
renormalized softmax).
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "tilelang"
TOP_P, TOP_K = 0.95, 50


def validate(
    num_rows: int,
    num_columns: int,
    reject_sampling: bool,
    top_p_filter: bool,
    top_k_filter: bool,
    dtype: DType | str,
) -> tuple[int, int, bool, bool, bool]:
    if DType.from_value(dtype) is not DType.FP32:
        raise ValueError(f"{BACKEND} fused_sample takes dtype=fp32, got {dtype!r}")
    rows, cols = int(num_rows), int(num_columns)
    if rows < 1 or cols < 32:
        raise ValueError(
            f"fused_sample needs num_rows >= 1 and num_columns >= 32, got {(rows, cols)}"
        )
    return rows, cols, bool(reject_sampling), bool(top_p_filter), bool(top_k_filter)


def profile_fused_sample_tilelang(
    num_rows: int,
    num_columns: int,
    reject_sampling: bool,
    top_p_filter: bool,
    top_k_filter: bool,
    dtype: DType | str,
) -> ComputeMetrics:
    rows, cols, reject, top_p_on, top_k_on = validate(
        num_rows, num_columns, reject_sampling, top_p_filter, top_k_filter, dtype
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    try:
        from deepinfer.model_executor.layers.tilelang_ops.fused_sample import (
            _check_top_p_top_k,
            fused_sample,
        )
        from deepinfer.model_executor.sampling_inputs import (
            LogitsTransformInputs,
            SamplingKernelInputs,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{BACKEND} needs DeepInfer's TileLang ops (deepinfer_env)"
        ) from exc
    try:
        dev = "cuda"
        g = torch.Generator(device=dev).manual_seed(0)
        logits = torch.randn(rows, cols, device=dev, dtype=torch.float32, generator=g)
        inputs = SamplingKernelInputs(
            LogitsTransformInputs(
                torch.ones(rows, device=dev),
                torch.full((rows,), TOP_P if top_p_on else 1.0, device=dev),
                torch.full((rows,), TOP_K if top_k_on else -1, device=dev, dtype=torch.int32),
            ),
            torch.arange(rows, device=dev, dtype=torch.int64).to(torch.uint64),
            torch.arange(rows, device=dev, dtype=torch.int32),
        )
        kwargs: dict[str, Any] = {"tag": "mtp"}
        if reject:
            kwargs = {
                "is_verify": torch.ones(rows, device=dev, dtype=torch.bool),
                "last_nextn_token_logprobs": torch.log_softmax(
                    torch.randn(rows, cols, device=dev, generator=g), dim=-1
                ),
                "last_next_n_token_ids": torch.randint(
                    0, cols, (rows,), device=dev, dtype=torch.int32, generator=g
                ),
                "mtp_info_read_rows": torch.arange(rows, device=dev, dtype=torch.int32),
            }

        def run_once() -> Any:
            return fused_sample(logits, inputs, **kwargs)

        got = run_once()[0]  # TileLang JIT, outside timing.
        torch.cuda.synchronize()
        t = inputs.logits
        try:
            n = min(rows, 64)  # a row sample bounds the reference's fp64 softmax
            _check_top_p_top_k(
                got[:n], logits[:n], t.temperatures[:n], t.top_ps[:n], t.top_ks[:n], label=BACKEND
            )
        except AssertionError as exc:
            raise KernelLaunchFailed(f"{BACKEND} fused_sample filter check failed: {exc}") from exc

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} fused_sample ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} fused_sample failed: {exc}") from exc

    bytes_accessed = rows * cols * (8 + 4 * int(reject))
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["profile_fused_sample_tilelang", "validate"]
