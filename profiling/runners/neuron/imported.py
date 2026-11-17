"""Runner of the imported-only Neuron backends (GLM-5.3-Flash on Trainium2).

Their rows are measured by the glm5-trn2 project's own benchmarks (one compiled region per call, NRT
device time) and imported into profile.db by ``tools/neuron-bench-import``; this checkout has no
live Neuron execution path for them (the ``trn2-flashnext`` branch's ``profiling/exec/neuron.py`` is
not ported). A cache miss therefore fails with the command that fills it instead of profiling.
"""

from __future__ import annotations

from typing import Any

FILL_HINT = (
    "uv run python tools/neuron-bench-import/import_bench.py "
    "--bench-dir <glm5-trn2>/logs/bench --fill-grid"
)


def profile(**kwargs: Any) -> Any:
    raise RuntimeError(
        f"Neuron row {kwargs!r} is not in profile.db and cannot be profiled live in this "
        f"checkout; import it: {FILL_HINT}"
    )
