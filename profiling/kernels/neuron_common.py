"""Shared registration for Neuron backends that time the Flash-Next engine's op modules.

Not a kernel kind: a helper the kind modules call. Every such backend:

- is named ``neuron_<impl>``, with ``<impl>`` the op module's ``IMPLS`` key;
- runs on a Trainium2 logical core (``device_family="neuron"``) in the Neuron
  SDK 2.33 image (``neuron_sdk233_env``);
- reaches the ops through ``profiling.runners.neuron.flashnext_ops``, which
  needs ``SERVINGSTUDIO_NEURON_EXTRA_SOURCE_DIRS`` to name the flashnext-neuron
  project;
- records NRT system-trace device time (the median over 30 calls), not wall
  time. Host launch overhead is a separate kind (``neuron_region_launch``).
"""

from __future__ import annotations

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import BackendDoc
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
)

NEURON_ENV = "neuron_sdk233_env"
NEURON_RUNNER_MODULE = "profiling.runners.neuron.flashnext_ops"
NEURON_PREFIX = "neuron_"
# The flashnext-neuron project has no public URL yet.
OPS_URL = None


def neuron_backend(impl: str) -> str:
    return NEURON_PREFIX + impl


def neuron_supports(compute: frozenset[DType] | None = frozenset({DType.BF16})) -> BackendSupport:
    return BackendSupport(
        compute=compute,
        device_family="neuron",
        architectures=frozenset({"Trainium2"}),
    )


def neuron_spec(
    kind: str,
    impl: str,
    args_schema: type[KernelArgs],
    summary: str,
    *,
    compute: frozenset[DType] | None = frozenset({DType.BF16}),
    metric_family: MetricFamily = MetricFamily.COMPUTE,
    module: str = NEURON_RUNNER_MODULE,
    function: str | None = None,
    list_native: bool = False,
    logical_cores: int = 1,
) -> KernelProfilerSpec:
    """One ``(kind, neuron_<impl>)`` registry row timing a flashnext op region."""
    from profiling.runners.neuron.entry import entry_name  # torch-free

    return KernelProfilerSpec(
        kernel_kind=kind,
        backend=neuron_backend(impl),
        supports=neuron_supports(compute),
        runner_ref=RunnerRef(module_name=module, function_name=function or entry_name(kind, impl)),
        table_name=kind,
        args_schema=args_schema,
        metric_family=metric_family,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env=NEURON_ENV,
        list_native=list_native,
        neuron_logical_cores=logical_cores,
        doc=BackendDoc(summary=summary, url=OPS_URL),
    )
