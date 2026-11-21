"""DeepInfer's dense decode indexer logits on SM100.

Two backends live here, one per DeepGEMM callable of the dense branch of
``MQA._topk_decode`` (``deepinfer/model_executor/models/v4_modules/index_attn.py``):

* ``dsa_paged_mqa_logits_decode`` / ``deepgemm_fp4`` -- the scoring,
  ``deep_gemm.fp8_fp4_paged_mqa_logits((q_fp4, q_sf), kv_cache, weights,
  klen.unsqueeze(1), block_tables, schedule, max_kv_len, indices=query_indices)``:
  one ``sm100_paged_mqa_logits`` launch per index layer.
* ``paged_mqa_logits_metadata`` / ``deepgemm_sm100`` -- the schedule,
  ``deep_gemm.get_paged_mqa_logits_metadata(klen.unsqueeze(1), block_size,
  num_sms, indices=query_indices)``: one single-CTA
  ``sm100_paged_mqa_logits_metadata`` launch. ``_get_decode_metadata`` caches
  it per compress ratio for the step, so it runs once per ratio, not per layer.

DeepInfer's DeepGEMM build runs SM100 paged logits only in its varlen form:
one token per row (``next_n == 1``) with ``indices`` naming each row's request
(``csrc/apis/attention.hpp``). A request's ``next_n`` verify rows are
consecutive, and the scheduler groups them back into one request and loads
its keys once for all of its Q blocks (``scheduler/sm100_paged_mqa_logits.cuh``
"reusing each chunk's KV across its Q blocks"). The kind's ``batch_size`` x
``next_n`` rows are launched that way (:mod:`deepinfer_paged_indexer_common`).

Check (scoring): request 0's first and last rows and the last request's last
row against fp32 math on the dequantized operands.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention import deepinfer_indexer_common as common
from profiling.runners.attention import deepinfer_paged_indexer_common as paged
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "deepgemm_fp4"
TOL = 1e-3
# The FP4 storage identity this backend measures (kind fields after the shape).
CACHE_FORMAT = "page_planar_fp4_ue8m0_scale"
_IDENTITY = {
    "q_dtype": DType.MXFP4_E2M1,
    "cache_dtype": DType.MXFP4_E2M1,
    # MX formats carry their UE8M0 block scales; the scale is the format's own.
    "scale_dtype": DType.MXFP4_E2M1,
    "weight_dtype": DType.BF16,
    "output_dtype": DType.BF16,
}
# A metadata call needs some context; its single CTA's work is per row, not per key.
METADATA_CONTEXT_LEN = 4096


def validate(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    scale_dtype: DType | str,
    weight_dtype: DType | str,
    output_dtype: DType | str,
    context_mode: str,
    page_mapping: str,
    cache_format: str,
    clean_logits: bool,
) -> tuple[int, int, int, int, int, int, int]:
    given = {
        "q_dtype": q_dtype,
        "cache_dtype": cache_dtype,
        "scale_dtype": scale_dtype,
        "weight_dtype": weight_dtype,
        "output_dtype": output_dtype,
    }
    for name, want in _IDENTITY.items():
        if DType.from_value(given[name]) is not want:
            raise ValueError(f"{BACKEND} takes {name}={want.value}, got {given[name]!r}")
    if (context_mode, page_mapping, cache_format) != ("uniform", "unique_scattered", CACHE_FORMAT):
        raise ValueError(
            f"{BACKEND} takes context_mode=uniform, page_mapping=unique_scattered, "
            f"cache_format={CACHE_FORMAT}, got {(context_mode, page_mapping, cache_format)}"
        )
    if clean_logits:
        raise ValueError(f"{BACKEND} takes clean_logits=false (DeepInfer never cleans)")
    values = tuple(
        int(v)
        for v in (batch_size, context_len, next_n, max_model_len, num_heads, head_dim, block_size)
    )
    if min(values) < 1 or values[1] > values[3]:
        raise ValueError(f"need positive sizes and context_len <= max_model_len, got {values}")
    if values[5] != 128 or values[4] % 4:
        raise ValueError(
            f"FP4 paged MQA logits needs head_dim 128 and heads % 4 == 0, got {values[4:6]}"
        )
    if values[6] not in paged.DENSE_PAGE_SIZES:
        raise ValueError(f"SM100 paged MQA logits pages hold 32, 64 or 128 keys, got {values[6]}")
    return values  # type: ignore[return-value]


def _imports() -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    try:
        import deep_gemm
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{BACKEND} needs DeepInfer's DeepGEMM (deepinfer_env)"
        ) from exc
    return torch, deep_gemm


def _failed(exc: Exception, what: str) -> Exception:
    if exc.__class__.__name__ == "OutOfMemoryError":
        return OOMError(f"{BACKEND} {what} ran out of memory")
    return KernelLaunchFailed(f"{BACKEND} {what} failed: {exc}")


def _metadata(deep_gemm: Any, ops: paged.PagedFp4Operands) -> Any:
    return deep_gemm.get_paged_mqa_logits_metadata(
        ops.context_lens, ops.block_size, deep_gemm.get_num_sms(), indices=ops.indices
    )


def _check(torch: Any, ops: paged.PagedFp4Operands, logits: Any) -> None:
    last = ops.batch_size - 1
    for request, rows in ((0, {0, ops.next_n - 1}), (last, {ops.next_n - 1})):
        keys = ops.request_keys(torch, request)
        for local in sorted(rows):
            row = request * ops.next_n + local
            want = common.reference_logits_row(torch, ops.q_ref[row], keys, ops.weights[row])
            got = logits[row, : ops.context_len].float()
            diff = common.calc_diff(got, want)
            if not diff <= TOL:
                raise KernelLaunchFailed(f"{BACKEND} logits row {row}: diff {diff:.3e} > {TOL:.0e}")


def scheduled_keys(batch_size: int, context_len: int, block_size: int) -> int:
    """Keys the kernel reads: each request's whole pages, once for all its rows."""
    return batch_size * paged.pages_per_request(context_len, block_size) * block_size


def profile_dsa_paged_mqa_logits_decode_deepgemm_fp4(**kwargs: Any) -> ComputeMetrics:
    batch_size, context_len, next_n, max_model_len, num_heads, head_dim, block_size = validate(
        **kwargs
    )
    torch, deep_gemm = _imports()
    try:
        ops = paged.build_paged_fp4(
            torch,
            batch_size=batch_size,
            next_n=next_n,
            context_len=context_len,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
        )
        schedule = _metadata(deep_gemm, ops)

        def run_once() -> Any:
            return deep_gemm.fp8_fp4_paged_mqa_logits(
                (ops.q_fp, ops.q_sf),
                ops.cache,
                ops.weights,
                ops.context_lens,
                ops.block_table,
                schedule,
                max_model_len,
                indices=ops.indices,
            )

        logits = run_once()  # JIT compile, outside timing.
        torch.cuda.synchronize()
        _check(torch, ops, logits)
        del logits

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        raise _failed(exc, "dsa_paged_mqa_logits_decode") from exc

    rows = batch_size * next_n
    keys = scheduled_keys(batch_size, context_len, block_size)
    cells = next_n * keys
    flops = 2 * cells * num_heads * head_dim
    pages = paged.pages_per_request(context_len, block_size)
    bytes_accessed = (
        rows * num_heads * (head_dim // 2 + 4)  # FP4 Q + packed scales
        + 2 * rows * num_heads  # bf16 weights
        + keys * (head_dim // 2 + 4)  # FP4 keys + scales, once per request
        + 4 * rows * pages  # per-row block table
        + 8 * rows  # context lens + indices
        + 2 * cells  # bf16 logits
    )
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


def validate_metadata(batch_size: int, next_n: int, block_size: int) -> tuple[int, int, int]:
    values = int(batch_size), int(next_n), int(block_size)
    if min(values) < 1:
        raise ValueError(f"paged_mqa_logits_metadata needs positive sizes, got {values}")
    if values[2] not in paged.DENSE_PAGE_SIZES:
        raise ValueError(f"SM100 paged MQA logits pages hold 32, 64 or 128 keys, got {values[2]}")
    return values


def profile_paged_mqa_logits_metadata_deepgemm(
    batch_size: int, next_n: int, block_size: int
) -> ComputeMetrics:
    batch_size, next_n, block_size = validate_metadata(batch_size, next_n, block_size)
    torch, deep_gemm = _imports()
    try:
        ops = paged.build_paged_fp4(
            torch,
            batch_size=batch_size,
            next_n=next_n,
            context_len=METADATA_CONTEXT_LEN,
            num_heads=32,
            head_dim=128,
            block_size=block_size,
        )

        def run_once() -> Any:
            return _metadata(deep_gemm, ops)

        schedule = run_once()  # JIT compile, outside timing.
        # The schedule is only meaningful through the kernel that reads it.
        logits = deep_gemm.fp8_fp4_paged_mqa_logits(
            (ops.q_fp, ops.q_sf),
            ops.cache,
            ops.weights,
            ops.context_lens,
            ops.block_table,
            schedule,
            METADATA_CONTEXT_LEN,
            indices=ops.indices,
        )
        torch.cuda.synchronize()
        _check(torch, ops, logits)
        del logits

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        raise _failed(exc, "paged_mqa_logits_metadata") from exc

    rows = batch_size * next_n
    # Reads every row's context length and request index; writes (num_sms + 1) x 2 words.
    bytes_accessed = 8 * rows + 8 * (deep_gemm.get_num_sms() + 1)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = [
    "CACHE_FORMAT",
    "profile_dsa_paged_mqa_logits_decode_deepgemm_fp4",
    "profile_paged_mqa_logits_metadata_deepgemm",
    "scheduled_keys",
    "validate",
    "validate_metadata",
]
