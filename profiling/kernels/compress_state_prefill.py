"""DeepInfer's prefill KV compressor (``tilelang_ops/compress.py``
``compress_and_update_state_prefill``).

Wire string: ``"compress_state_prefill"``.

One call, two launches (``compress_and_update_state_prefill_kernel`` and its
``_1`` per-sequence state kernel): softmax-weighted pooling of every
``compress_ratio`` new rows' fp32 latent by their score, plus the update of
the sequence's fp32 compression state ring (``compress_ratio +
DEEPINFER_MAX_BACKTRACKING_TOKENS - 1`` tokens, ``cache_engine.py``
``get_inference_compress_block_size``). DeepSeek-V4.1 runs it on the ratio-2
KV source layers (``compress.py`` ``forward_pre_norm``). The runner prices one
fresh request of ``num_tokens`` rows.

Why a new kind: no existing kind prices this pooling plus state update.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND = "compress_state_prefill"


@dataclass(frozen=True)
class CompressStatePrefillArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="New rows of the request.")
    head_dim: int = arg(unit="elements", doc="Compressed KV width.")
    compress_ratio: int = arg(unit="tokens", doc="Rows pooled into one compressed token (1 or 2).")
    state_size: int = arg(unit="tokens", doc="Rows kept in the compression state ring.")


DOC = KernelDoc(
    title="Prefill KV compression",
    summary="Pools score-weighted fp32 latents into compressed KV rows and updates the state ring.",
    description=(
        "For each compressed token, softmax over the ratio rows' scores weights "
        "their latents into one bf16 row; the request's last state_size rows of "
        "score and latent are written to the fp32 state cache."
    ),
    category="Attention",
    subcategory="Compressed sparse MLA",
    formula=("bytes = 8·T·D read + 2·(T/ratio)·D + 16·min(T, state)·D written",),
    default_metric="time_ms",
    method=CUPTI_METHOD,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tilelang",
        supports=BackendSupport(compute=None, min_compute_capability=(9, 0)),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deepinfer_kv_tilelang",
            function_name="profile_compress_state_prefill_tilelang",
        ),
        table_name=KIND,
        args_schema=CompressStatePrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "DeepInfer's TileLang compress_and_update_state_prefill (both "
                "launches), checked by DeepInfer's _test_compress_state_prefill at "
                "the same dim, ratio and state size."
            ),
        ),
    )
)
