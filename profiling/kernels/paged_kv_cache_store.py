"""DeepInfer's paged-cache store (``tilelang_ops/store_kv_cache.py`` ``store_kv_cache``).

Wire string: ``"paged_kv_cache_store"``.

One ``_store_kv_cache_kernel`` launch encodes bf16 rows into the cache through
a slot map, skipping ``-1`` slots. DeepSeek-V4.1 prefill stores the SWA rows
after the SWA load (``attn.py``): the slot map covers every new row but only
each sequence's last ``swa_block_size`` rows have a slot
(``prepare_tensor_input_v4.cpp`` ``slot_mapping_store_swa``). The runner uses
one sequence of ``num_tokens`` rows.

Why a new kind: no existing kind stores into DeepInfer's cache encodings.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND = "paged_kv_cache_store"


@dataclass(frozen=True)
class PagedKvCacheStoreArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows in the slot map.")
    block_size: int = arg(
        unit="tokens",
        doc="Tokens per cache page (the SWA ring); the last block_size rows are stored.",
    )
    cache_format: str = arg(doc="Cache encoding: mxfp8, nvfp4 or mxfp4.")
    dtype: DType = arg(doc="Row element type (bf16).")


DOC = KernelDoc(
    title="Paged KV cache store",
    summary="Encodes bf16 rows into a DeepInfer paged KV cache through a slot map.",
    description=(
        "Every row reads its slot; rows with a slot are quantized to the cache "
        "encoding and written to that page position. In the prefill SWA store "
        "only the last ring's worth of rows of a sequence has a slot."
    ),
    category="Attention",
    subcategory="Compressed sparse MLA",
    formula=("bytes = 4·T slots + stored·(2·512 + bytes_per_token)",),
    default_metric="time_ms",
    method=CUPTI_METHOD,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tilelang",
        supports=BackendSupport(compute=frozenset({DType.BF16}), min_compute_capability=(9, 0)),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deepinfer_kv_tilelang",
            function_name="profile_paged_kv_cache_store_tilelang",
        ),
        table_name=KIND,
        args_schema=PagedKvCacheStoreArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "DeepInfer's TileLang store_kv_cache, checked bit-exact against store_kv_cache_ref."
            ),
        ),
    )
)
