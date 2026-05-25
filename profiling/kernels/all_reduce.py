"""All-reduce collective kernel kind.

All Python-side per-kernel knowledge for ``all_reduce`` lives here: the wire
string ``KIND``, the ``AllReduceArgs`` schema, and the ``register(...)`` calls
that wire the ``nccl`` and ``nvshmem`` backends into ``profiling.db.registry``.

Wire string: ``"all_reduce"`` — matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/all_reduce.rs`` and the Python facade stem used
by ``profiling.facade`` to generate ``get_all_reduce_times`` /
``count_missing_all_reduce``.

Comm-specific vs the compute kernels: ``metric_family=COMM`` (the table carries
algbw/busbw/message_size, not tflops), and ``gpu_count_fn`` tells L1b each spec
needs ``num_gpus`` real GPUs reserved for the multi-rank launcher. The runner
modules are referenced lazily via ``RunnerRef`` so the main process never
eager-imports torch / a multi-process launcher.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "all_reduce"


@dataclass(frozen=True)
class AllReduceArgs(KernelArgs):
    num_gpus: int
    message_size_bytes: int
    dtype: DType
    # Network fabric token (matches Rust `Fabric` serde wire form, e.g.
    # "nvlink"). A row/cache key only — the runner body does not use it.
    fabric: str


def _spec(backend: str, module_name: str) -> KernelProfilerSpec:
    return KernelProfilerSpec(
        kernel_kind=KIND,
        backend=backend,
        runner_ref=RunnerRef(module_name=module_name, function_name="profile_all_reduce"),
        table_name=KIND,
        args_schema=AllReduceArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
    )


register(_spec("nccl", "profiling.runners.comm.nccl"))
register(_spec("nvshmem", "profiling.runners.comm.nvshmem"))
