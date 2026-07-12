"""Process-wide context that turns ``Timer.cupti`` into a trend+telemetry probe.

The ``measure`` verb (``python -m profiling measure``) needs the *callable* every
CUPTI runner builds, without editing a single runner. The seam is
``Timer.cupti(fn)``: when a ``MeasureContext`` is active, that call runs the
sustained trend capture (per-launch CUPTI durations + NVML telemetry + plots)
instead of the ordinary mean-only measurement, then returns a representative
time so the runner still completes normally.

The context is set only inside the local worker subprocess for one ``measure``
invocation. Normal ``run`` / ``query`` / simulator calls never set it, so
``Timer.cupti`` behaves exactly as before. ``consumed`` guards against a runner
that calls ``Timer.cupti`` more than once: only the first call under a context
produces an artifact set, and a context that is never consumed tells the driver
this kernel is not CUPTI-timed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MeasureContext:
    """One ``measure`` invocation's capture configuration and result slots."""

    output_dir: Path
    label: str
    shape: dict[str, object]
    duration_s: float = 10.0
    telemetry_hz: float = 20.0
    clear_l2: bool = True
    telemetry: bool = True
    # Filled by the Timer.cupti hook after the capture runs.
    consumed: bool = False
    time_ms: float | None = None
    artifacts: list[str] = field(default_factory=list)


_ACTIVE: MeasureContext | None = None


def set_measure_context(context: MeasureContext) -> None:
    global _ACTIVE
    _ACTIVE = context


def get_measure_context() -> MeasureContext | None:
    return _ACTIVE


def clear_measure_context() -> None:
    global _ACTIVE
    _ACTIVE = None
