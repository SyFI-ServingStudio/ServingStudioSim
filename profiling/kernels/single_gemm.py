"""Single-GEMM kernel kind.

All Python-side per-kernel knowledge for ``single_gemm`` lives here: the wire
string ``KIND``, the ``SingleGemmArgs`` schema, and the ``register(...)`` call
that wires this kernel into ``profiling.db.registry``.

Wire string: ``"single_gemm"`` — matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/single_gemm.rs`` and the Python facade stem used
by ``profiling.facade`` to generate ``get_single_gemm_times`` /
``count_missing_single_gemm``.

Importing this module has a side effect: it appends a ``KernelProfilerSpec``
row to the registry. The runner module ``profiling.runners.gemm.torch`` is
referenced lazily via ``RunnerRef`` so the main process never eager-imports
torch/cuda.
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

KIND: str = "single_gemm"


@dataclass(frozen=True)
class SingleGemmArgs(KernelArgs):
    m: int
    n: int
    k: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.torch",
            function_name="profile_single_gemm",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)
