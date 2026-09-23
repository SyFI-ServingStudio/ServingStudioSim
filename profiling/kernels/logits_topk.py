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
    num_rows: int
    num_columns: int
    top_k: int
    dtype: DType


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
    )
)

__all__ = ["KIND", "LogitsTopkArgs"]
