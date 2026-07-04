"""Single-GEMM kernel kind.

All Python-side per-kernel knowledge for ``single_gemm`` lives here: the wire
string ``KIND``, the ``SingleGemmArgs`` schema, and the ``register(...)`` call
that wires this kernel into ``profiling.db.registry``.

Wire string: ``"single_gemm"`` — matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/single_gemm.rs`` and the Python facade stem used
by ``profiling.facade`` to generate ``get_single_gemm_times`` /
``count_missing_single_gemm``.

Two backends share this kind/table/schema: ``torch`` (bf16/fp16 dense GEMM) and
``deepgemm`` (FP8 dense GEMM, ``dtype = fp8_e4m3``). The simulator selects one by
dtype (deepgemm iff fp8), so a bf16 config never resolves to the FP8-only kernel.

Importing this module has a side effect: it appends ``KernelProfilerSpec`` rows
to the registry. The runner modules ``profiling.runners.gemm.{torch,deepgemm}``
are referenced lazily via ``RunnerRef`` so the main process never eager-imports
torch/cuda.
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
        supports=BackendSupport(compute=frozenset({DType.BF16, DType.FP16})),
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

# DeepGEMM FP8 dense kernel — same wire schema / table, FP8 compute
# (dtype = fp8_e4m3, fp8 in / bf16 out). subprocess_env=None (default env;
# deep_gemm is a pinned project dep, see CLAUDE.md / `just sync`).
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm",
        supports=BackendSupport(compute=frozenset({DType.FP8_E4M3})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.deepgemm",
            function_name="profile_single_gemm",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)
