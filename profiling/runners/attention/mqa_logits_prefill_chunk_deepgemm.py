"""DeepInfer's dense prefill indexer logits -- ``deepgemm_sm100`` backend of ``mqa_logits_prefill_chunk``.

The timed call is the dense branch of ``MQA._compute_prefill_p_sum``
(``deepinfer/model_executor/models/v4_modules/index_attn.py``)::

    deep_gemm.fp8_fp4_mqa_logits((q_fp4, q_sf), (k_fp4, k_sf), weights, ks, ke,
                                 max_seqlen_k=max_kv_len)

with FP4 E2M1 + packed UE8M0 Q and K, bf16 weights and bf16 logits: one
``sm100_mqa_logits<..., cutlass::float_e2m1_t, ...>`` launch. The call scores
one query chunk of one request whose rows are the tail of its context
(:mod:`deepinfer_indexer_common`). ``max_seqlen_k`` is the request's key count;
DeepInfer passes ``runtime_max_model_len // ratio``, which only sizes the
logits buffer the kernel leaves uninitialized past each row's window.

Check: four rows (first and last, plus two inner) against an fp32
transcription on the dequantized operands.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention import deepinfer_indexer_common as common
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "deepgemm_sm100"
# Relative difference on the dequantized operands; bf16 logit rounding is ~1e-5.
TOL = 1e-3


def validate(
    num_queries: int,
    context_len: int,
    compress_ratio: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> tuple[int, int, int, int, int]:
    if DType.from_value(dtype) is not DType.MXFP4_E2M1:
        raise ValueError(f"{BACKEND} mqa_logits_prefill_chunk takes dtype=mxfp4_e2m1, got {dtype!r}")
    values = int(num_queries), int(context_len), int(compress_ratio), int(num_heads), int(head_dim)
    if min(values) < 1:
        raise ValueError(f"mqa_logits_prefill_chunk needs positive sizes, got {values}")
    if values[0] > values[1]:
        raise ValueError(f"num_queries {values[0]} > context_len {values[1]}")
    if values[4] != 128 or values[3] % 4:
        raise ValueError(f"FP4 MQA logits needs head_dim 128 and heads % 4 == 0, got {values[3:]}")
    return values


def valid_pairs(num_queries: int, context_len: int, compress_ratio: int) -> int:
    return sum(common.tail_row_keys(num_queries, context_len, compress_ratio))


def profile_mqa_logits_prefill_chunk_deepgemm(
    num_queries: int,
    context_len: int,
    compress_ratio: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    num_queries, context_len, ratio, num_heads, head_dim = validate(
        num_queries, context_len, compress_ratio, num_heads, head_dim, dtype
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    try:
        import deep_gemm
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{BACKEND} needs DeepInfer's DeepGEMM (deepinfer_env)") from exc
    try:
        torch.manual_seed(0)
        num_keys = common.num_request_keys(context_len, ratio)
        q_fp, q_sf, q_ref = common.fp4_queries(torch, num_queries, num_heads, head_dim)
        k_fp, k_sf = common.fp4_keys(torch, num_keys, head_dim)
        weights = common.bf16_weights(torch, num_queries, num_heads)
        ks, ke = common.build_tail_spans(torch, num_queries, context_len, ratio)

        def run_once() -> Any:
            return deep_gemm.fp8_fp4_mqa_logits(
                (q_fp, q_sf), (k_fp, k_sf), weights, ks, ke, max_seqlen_k=num_keys
            )

        logits = run_once()  # JIT compile, outside timing.
        torch.cuda.synchronize()
        k_ref = common.fp4_dequant(torch, k_fp, k_sf)
        ends = ke.tolist()
        for row in sorted({0, num_queries // 3, 2 * num_queries // 3, num_queries - 1}):
            want = common.reference_logits_row(torch, q_ref[row], k_ref[: ends[row]], weights[row])
            got = logits[row, : ends[row]].float()
            diff = common.calc_diff(got, want)
            if not diff <= TOL:
                raise KernelLaunchFailed(f"{BACKEND} logits row {row}: diff {diff:.3e} > {TOL:.0e}")
        del logits, k_ref

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} mqa_logits_prefill_chunk ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} mqa_logits_prefill_chunk failed: {exc}") from exc

    pairs = valid_pairs(num_queries, context_len, ratio)
    flops = 2 * pairs * num_heads * head_dim
    # FP4 Q (+ scales) and K (+ scales) once, bf16 weights, bf16 valid logits.
    bytes_accessed = (
        num_queries * num_heads * (head_dim // 2 + 4)
        + num_keys * (head_dim // 2 + 4)
        + 2 * num_queries * num_heads
        + 2 * pairs
    )
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["profile_mqa_logits_prefill_chunk_deepgemm", "valid_pairs", "validate"]
