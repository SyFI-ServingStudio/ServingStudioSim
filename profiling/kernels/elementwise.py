"""Element-wise kernel kind.

All Python-side per-kernel knowledge for ``elementwise`` lives here: the wire
string ``KIND``, the ``ElementwiseArgs`` schema, and the ``register(...)`` call
that wires this kernel into ``profiling.db.registry``.

Wire string: ``"elementwise"`` — matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/elementwise.rs`` and the Python facade stem used
by ``profiling.facade`` to generate ``get_elementwise_times`` /
``count_missing_elementwise``.

Model (mirrors ``ref/profile/elementwise/elementwise_triton.py``): a generic
byte-level fan-in elementwise/reduce keyed by *total* ``input_size_bytes`` ->
``output_size_bytes`` (dtype-agnostic, ``uint8``). It covers MoE activation
(2N->N), local MoE reduce (xN->N), copy (N->N), and zero-fill (0->N). The
profiler keys directly by total bytes because the simulator already resolves
token batch shapes.

Shape split: the args here are the TOTAL byte sizes (the DB key and runner
kwargs). On the Rust side those totals are produced by folding a per-token byte
rate (static config) with ``num_tokens`` (the runtime sweep axis,
``Cache1DLinear``) — see the Rust ``enumerate``. So ``num_tokens`` and the
per-token rates never appear in this wire schema.

Importing this module has a side effect: it appends a ``KernelProfilerSpec``
row to the registry. The runner module
``profiling.runners.elementwise.triton`` is referenced lazily via ``RunnerRef``
so the main process never eager-imports torch/triton/cuda.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import KernelArgs
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "elementwise"


@dataclass(frozen=True)
class ElementwiseArgs(KernelArgs):
    input_size_bytes: int
    output_size_bytes: int


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="triton",
        # Byte-keyed / uint8 — dtype-agnostic.
        supports=BackendSupport(compute=None),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.triton",
            function_name="profile_elementwise",
        ),
        table_name=KIND,
        args_schema=ElementwiseArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)
