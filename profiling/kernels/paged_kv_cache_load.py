"""DeepInfer's paged-cache load (``tilelang_ops/load_paged_kv_cache.py``).

Wire string: ``"paged_kv_cache_load"``.

One ``_load_paged_kv_cache_kernel`` launch reads compact page ranges
(``load_segments`` rows ``[block, begin, end, dst]``) into a dense tensor,
dequantizing the main/SWA formats to bf16 and copying the index format's
values and scales as they are. DeepSeek-V4.1 prefill loads (``attn.py``
``forward``; ``index_attn.py`` ``_load_prefill_kv``):

- the SWA history, ``window`` tokens of each sequence's 134-token ring
  (MXFP8, ``segment_tokens=128, block_size=134``);
- the main compressed KV at a KV source layer, whole pages of
  ``128 / ratio`` tokens (NVFP4);
- the index K at an index-K owner, whole pages (MXFP4 index format).

Why a new kind: ``packed_kv_cache_gather`` is vLLM's block-table gather with
another layout and key.
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

KIND = "paged_kv_cache_load"


@dataclass(frozen=True)
class PagedKvCacheLoadArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens loaded (total_k).")
    block_size: int = arg(unit="tokens", doc="Tokens per cache page.")
    segment_tokens: int = arg(unit="tokens", doc="Tokens per contiguous range read from one page.")
    cache_format: str = arg(
        doc="Cache encoding: mxfp8 (V41 main/SWA), nvfp4 (V41 FP4 main) or mxfp4 (FP4 index)."
    )


DOC = KernelDoc(
    title="Paged KV cache load",
    summary="Reads page ranges of a DeepInfer paged KV cache into a dense tensor.",
    description=(
        "Each load segment copies one run of tokens from a page to a dense "
        "output offset. The main and SWA encodings are dequantized to bf16 on "
        "the way; the index encoding keeps its FP4 values and UE8M0 scales."
    ),
    category="Attention",
    subcategory="Compressed sparse MLA",
    formula=(
        "bytes = T·bytes_per_token read + T·(2·512 for main/SWA, bytes_per_token "
        "for index) written",
    ),
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
            function_name="profile_paged_kv_cache_load_tilelang",
        ),
        table_name=KIND,
        args_schema=PagedKvCacheLoadArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "DeepInfer's TileLang load_paged_kv_cache, checked bit-exact "
                "against load_paged_kv_cache_ref."
            ),
        ),
    )
)
