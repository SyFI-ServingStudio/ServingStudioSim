"""Per-GPU golden store — recorded numbers a later run is compared against.

A golden file is `tests/golden/<metric>/<gpu_name>.json`: a flat `{key: float}`
map, one file per (metric, device). Hardware-dependent numbers live per GPU
because the same code produces different values on different silicon, so a
single shared baseline would be meaningless.

This lives in `launcher/` rather than `tests/` because both sides need it:
`pytest --update-golden` records through the `golden` fixture, and
`alignment-campaign compare --record` records accepted alignment metrics from a
production command. One storage implementation, two callers — a second store
would drift.

The store is deliberately metric-agnostic: it reads, holds, and writes numbers.
Whether a missing key skips or fails, what tolerance applies, and when recording
is allowed are the caller's policy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

#: Repository-root-relative home of every golden file. Kept under `tests/` (not
#: `launcher/`) because the pytest tier is the primary reader and the files are
#: reviewed as test data even when a launcher command writes them.
GOLDEN_ROOT = Path(__file__).resolve().parent.parent / "tests" / "golden"


def sanitize_device(gpu_name: str) -> str:
    """Device name → filename stem (`"NVIDIA B200"` → `"NVIDIA_B200"`).

    The unsanitized name stays the in-file identity; only the path is escaped.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", gpu_name)


class Golden:
    """Per-GPU golden file (`tests/golden/<metric>/<gpu_name>.json`). Reader by
    default; when `update` is set it records and persists. The caller owns the
    skip/assert/record policy so the store stays metric-agnostic."""

    def __init__(
        self, metric: str, gpu_name: str, update: bool, *, root: Path | None = None
    ) -> None:
        base = GOLDEN_ROOT if root is None else root
        self.path = base / metric / f"{sanitize_device(gpu_name)}.json"
        self.gpu_name = gpu_name
        self.update_enabled = update
        self.data: dict[str, float] = (
            json.loads(self.path.read_text()) if self.path.is_file() else {}
        )

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str) -> float | None:
        return self.data.get(key)

    def record(self, key: str, value: float) -> None:
        self.data[key] = value
        self.flush()

    def flush(self) -> None:
        """Persist the current map. Sorted keys + trailing newline keep the file
        diff-stable, so a re-record shows only the numbers that actually moved."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
