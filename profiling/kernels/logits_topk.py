"""Row-wise top-k selection over a dense score matrix.

One call of vLLM's `logits_processor._topk(scores, k)`: the largest `top_k`
entries of every row of a `[num_rows, num_columns]` score matrix, returned
sorted by value together with their column indices.

The production backend is `flashinfer.top_k(scores, k, sorted=True,
deterministic=True)`, which is what `_topk` calls whenever flashinfer is
importable. `sorted` and `deterministic` are not args: vLLM passes both as
`True` at every call site, and flashinfer branches on them (deterministic
rules out the cluster path and selects the `DET_FLAG` kernel specialization,
sorted adds the on-device stable value sort), so profiling any other
combination would measure a code path production never runs.

The `torch` backend is the documented fallback `torch.topk(scores, k, dim=-1)`
that `_topk` takes when flashinfer is missing; vLLM's own log line puts it at
"roughly half the speed". It is also the numerical oracle the flashinfer runner
checks against.

`num_columns` is a config axis, not a sweep axis: a call site's matrix width is
fixed by the model (a vocabulary shard, or a gathered candidate block), while
the row count moves with the batch every iteration.
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

KIND: str = "logits_topk"


@dataclass(frozen=True)
class LogitsTopkArgs(KernelArgs):
    num_rows: int = arg(unit="rows", doc="Score rows, one per scored position.")
    num_columns: int = arg(unit="elements", doc="Scores in each row.")
    top_k: int = arg(unit="elements", doc="Scores kept from each row.")
    dtype: DType = arg(doc="Element type of the input scores and output values.")


DOC = KernelDoc(
    title="Logits top-k",
    summary="Select the largest top_k scores and their column indices from each score row.",
    description=(
        "vLLM's vocab-parallel top-k takes the top_k logits from each GPU's "
        "vocabulary shard, all-gathers the candidates, and takes top_k again from "
        "the gathered block, without gathering the full logits. Both steps are "
        "this kernel at different widths: num_columns is the shard's vocabulary "
        "in the first and top_k · tp_size in the second. Scores are seeded "
        "normal values."
    ),
    category="Other",
    formula=(
        "TFLOPS = num_rows · num_columns / time",
        "GB/s = num_rows · [num_columns · bytes per value + top_k · (bytes per value + 4)] / time",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} Every launch of the call is counted: selection, the "
        "value sort and the index conversion. FlashInfer's selected values are "
        "checked against torch.topk before timing."
    ),
    caveats=(
        "TFLOPS treats one score comparison per input element as nominal work; "
        "the selection algorithms can inspect elements more than once.",
        "GB/s counts one read of the scores and writes of top_k values and "
        "4-byte indices per row. FlashInfer converts its int32 indices to int64, "
        "so this is not total device traffic.",
    ),
    # Both measured calls are library implementations; no separate PyTorch reference exists.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16, DType.FP16, DType.FP32})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.sampling.logits_topk",
            function_name="profile_logits_topk_torch",
        ),
        table_name=KIND,
        args_schema=LogitsTopkArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(summary="torch.topk, vLLM's fallback when FlashInfer is unavailable."),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer",
        # flashinfer's radix top-k templates the value type; `top_k` accepts
        # fp32/fp16/bf16 and nothing else.
        supports=BackendSupport(compute=frozenset({DType.BF16, DType.FP16, DType.FP32})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.sampling.logits_topk",
            function_name="profile_logits_topk_flashinfer",
        ),
        table_name=KIND,
        args_schema=LogitsTopkArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "FlashInfer top_k with sorted=True and deterministic=True, the "
                "call vLLM's logits processor makes."
            ),
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/topk.py",
        ),
    )
)

__all__ = ["KIND", "LogitsTopkArgs"]
