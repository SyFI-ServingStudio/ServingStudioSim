"""DeepInfer's SWA index expansion (``tilelang_ops/swa_index.py`` ``expand_swa_index``).

Wire string: ``"expand_swa_index"``.

One ``_expand_swa_index_kernel`` launch turns each row's four pivots (cached
start, cached count, new start, new count) into its [window] SWA key indices.
DeepSeek-V4.1 prefill builds ``kv_swa_full_index`` once per row set: at the
first layer and again at the YOCO cut on the compact rows.

Why a new kind: no existing kind expands these pivots.
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

KIND = "expand_swa_index"


@dataclass(frozen=True)
class ExpandSwaIndexArgs(KernelArgs):
    num_rows: int = arg(unit="tokens", doc="Query rows.")
    window_size: int = arg(unit="tokens", doc="SWA indices per row.")


DOC = KernelDoc(
    title="SWA index expansion",
    summary="Expands per-row SWA pivots into the row's window of key indices.",
    description=(
        "Reads [num_rows, 4] int32 pivots and writes [num_rows, window_size] "
        "int32: the cached run, then the new run, -1 elsewhere."
    ),
    category="Attention",
    subcategory="Compressed sparse MLA",
    formula=("bytes = 16·num_rows + 4·num_rows·window_size",),
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
            function_name="profile_expand_swa_index_tilelang",
        ),
        table_name=KIND,
        args_schema=ExpandSwaIndexArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "DeepInfer's TileLang expand_swa_index, checked equal to expand_swa_index_ref."
            ),
        ),
    )
)
