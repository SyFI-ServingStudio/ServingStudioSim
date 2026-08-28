"""The `.complete` resume marker (INV-5), shared by the sweep and the campaign.

The marker is written into a run's output directory **only** after a zero-exit
run, through an atomic replace, so a crash mid-write leaves no marker rather than
a truncated one. On the default resume path a directory carrying it is skipped;
`--refresh` ignores it and re-runs.

The marker is a claim about the *exit status*, not about the artifacts. Both
callers therefore pair it with their own validator — `launcher.sweep` with
`validate_simulation_artifacts`, `launcher.alignment_campaign` with the artifact
set each alignment phase is supposed to produce. Presence alone never means
"complete"; that is what keeps a marker left behind by an interrupted phase from
suppressing the re-run.
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

COMPLETE_MARKER = ".complete"


def marker_path(log_dir: Path) -> Path:
    return Path(log_dir) / COMPLETE_MARKER


def has_marker(log_dir: Path) -> bool:
    return marker_path(log_dir).is_file()


def mark_complete(log_dir: Path) -> None:
    """Write the marker atomically. The body is the completion timestamp, which
    is the cheapest way to tell a resumed run's age from the filesystem."""
    log_dir = Path(log_dir)
    marker = marker_path(log_dir)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{COMPLETE_MARKER}.", suffix=".tmp", dir=log_dir
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(datetime.now(UTC).isoformat() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, marker)
    finally:
        temporary_path.unlink(missing_ok=True)


def clear_marker(log_dir: Path) -> None:
    marker_path(log_dir).unlink(missing_ok=True)
