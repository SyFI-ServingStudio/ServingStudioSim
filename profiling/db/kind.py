"""``KernelKind`` — the per-kind wire string used by DB rows, PyO3 marshal,
and the Python facade stem (``get_{kind}_times`` / ``count_missing_{kind}``).

Per L1 design.md §2.1, every runner is registered under a
``(KernelKind, backend)`` pair. Concrete values are declared in
``profiling/kernels/<kind>.py`` as ``KIND`` constants and fed to
``register(KernelProfilerSpec(kernel_kind=KIND, ...))``. There is no central
enum to extend — enumerate registered kinds via
``profiling.db.registry.iter_kernel_profiler_specs()``.

``KernelKind`` is a type alias for ``str`` rather than a ``StrEnum`` so that
adding a kernel does not require editing this file (symmetric with the Rust
``pub type KernelKind = &'static str;`` in ``simulator/src/timing/bridge``).
"""

from __future__ import annotations

KernelKind = str
