"""DeepInfer's prefill index concatenation (``tilelang_ops/kv_index_cat.py``).

Wire string: ``"kv_index_cat"``.

One ``_kv_index_cat_kernel`` launch per sparse prefill attention layer
(``attn.py`` ``_prefill_attention``): each row's SWA indices [window] followed
by its compressed-KV top-k indices [topk] shifted past the SWA keys (``-1``
kept).

Why a new kind: no existing kind builds this index row.
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

KIND = "kv_index_cat"


@dataclass(frozen=True)
class KvIndexCatArgs(KernelArgs):
    num_rows: int = arg(unit="tokens", doc="Query rows.")
    window_size: int = arg(unit="tokens", doc="SWA indices per row.")
    topk: int = arg(unit="tokens", doc="Compressed-KV indices per row.")


DOC = KernelDoc(
    title="KV index concatenation",
    summary="Concatenates each row's SWA and shifted top-k KV indices.",
    description=(
        "Writes [num_rows, window_size + topk] int32: the SWA indices, then the "
        "top-k indices plus the SWA key count, keeping -1."
    ),
    category="Attention",
    subcategory="Compressed sparse MLA",
    formula=("bytes = 8·num_rows·(window_size + topk)",),
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
            function_name="profile_kv_index_cat_tilelang",
        ),
        table_name=KIND,
        args_schema=KvIndexCatArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=("DeepInfer's TileLang kv_index_cat, checked equal to kv_index_cat_ref."),
        ),
    )
)
