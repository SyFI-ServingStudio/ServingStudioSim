"""Neuron runners that time the Flash-Next engine's own op modules.

The ops live out of tree, in the flashnext-neuron project
(``flashnext/neuron/ops/<family>.py``); ``SERVINGSTUDIO_NEURON_EXTRA_SOURCE_DIRS``
mounts that project and puts it on the worker's ``PYTHONPATH``. Its adapter
package ``flashnext.neuron.sim_l1`` builds one compiled region per spec from the
same op functions the engine runs, with inputs built as the project's bench
scripts build them, and times it with the NRT system trace
(``runtime.device_bench``): the row's ``time_ms`` is device time, the median
over 30 calls, summed over the NEFFs one call launches.

Every registered Neuron backend is named ``neuron_<impl>``, where ``<impl>`` is
the op module's ``IMPLS`` key. The worker strips ``backend`` before it calls a
runner, so each (kind, impl) pair gets its own entry point, generated below as
``profile_<kind>__<impl>``.
"""

from __future__ import annotations

from collections.abc import Callable

from profiling.runners.metrics import ComputeMetrics
from profiling.runners.neuron.entry import entry_name

# kind -> impls with an entry point. A registry row's RunnerRef names
# `profile_<kind>__<impl>`; `entry_name` builds that name.
ENTRY_POINTS: dict[str, tuple[str, ...]] = {
    "grouped_rms_norm": ("torch",),
}


def _run(kind: str, impl: str, kwargs: dict) -> ComputeMetrics:
    from flashnext.neuron.sim_l1 import measure  # out-of-tree, worker-only import

    result = measure(kind, impl, **kwargs)
    time_ms = float(result["device_ms"])
    seconds = time_ms / 1e3
    return ComputeMetrics(
        time_ms=time_ms,
        tflops=float(result.get("flops") or 0.0) / seconds / 1e12,
        memory_bandwidth_gbps=float(result.get("bytes") or 0.0) / seconds / 1e9,
    )


def _make(kind: str, impl: str) -> Callable[..., ComputeMetrics]:
    def run(**kwargs) -> ComputeMetrics:
        return _run(kind, impl, kwargs)

    run.__name__ = entry_name(kind, impl)
    run.__qualname__ = run.__name__
    return run


for _kind, _impls in ENTRY_POINTS.items():
    for _impl in _impls:
        globals()[entry_name(_kind, _impl)] = _make(_kind, _impl)
