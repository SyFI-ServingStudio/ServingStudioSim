"""Grouped-GEMM kernel kind (distribution-sensitive op, L1 design §2.8).

All Python-side per-kernel knowledge for ``grouped_gemm`` lives here: the wire
string ``KIND``, the ``GroupedGemmArgs`` schema, and the ``register(...)`` call
that wires this kernel into ``profiling.db.registry``.

Wire string: ``"grouped_gemm"`` — matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/grouped_gemm.rs`` and the Python facade stem used
by ``profiling.facade`` to generate ``get_grouped_gemm_times`` /
``count_missing_grouped_gemm``.

Distribution-sensitive twist (§2.8): the runner measures one real timing point
for a concrete token-to-expert distribution, so the DB row carries the full
``per_group_batches`` vector (one count per local expert) — NOT the scalar
``global_expert_selections`` the simulator sweeps. Rust holds the per-GPU ppm
shard in ``GroupedGemmKernelConfig`` identity and turns each swept
``global_expert_selections`` into ``per_group_batches`` via
``RoutingDistribution::to_per_expert_counts`` before querying this table.

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
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "grouped_gemm"


@dataclass(frozen=True)
class GroupedGemmArgs(KernelArgs):
    n: int
    k: int
    dtype: DType
    num_local_experts: int
    # Actual batch per local expert; len == num_local_experts. A tuple so the
    # frozen args stay hashable; stored as a JSON-text DB column (§2.1 default).
    per_group_batches: tuple[int, ...]


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16, DType.FP16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.torch",
            function_name="profile_grouped_gemm",
        ),
        table_name=KIND,
        args_schema=GroupedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)

# DeepGEMM FP8 m-grouped contiguous kernel — same wire schema / table, FP8
# compute (dtype = fp8_e4m3). Per L1 design §8.1, subprocess_env=None (default
# env; deep_gemm is a pinned project dep, see CLAUDE.md / `just sync`).
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm",
        supports=BackendSupport(compute=frozenset({DType.FP8_E4M3})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.deepgemm",
            function_name="profile_grouped_gemm",
        ),
        table_name=KIND,
        args_schema=GroupedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)
