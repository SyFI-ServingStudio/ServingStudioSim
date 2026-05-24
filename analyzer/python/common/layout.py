"""Run-dir artifact layout — mirror of the Rust `io.rs` convention so the two
sides agree on where payloads live and where PNGs go.

`raw/` (sim parquet) · `reports/` (numbers JSON) · `payloads/` (plot JSON) ·
`plots/` (PNG, written here).
"""

from __future__ import annotations

import json
from pathlib import Path

RAW_DIR = "raw"
REPORTS_DIR = "reports"
PAYLOADS_DIR = "payloads"
PLOTS_DIR = "plots"


def resolve_artifact(log_dir: Path, name: str) -> Path:
    """Find an artifact in the run root or any known subdir (the Rust analyzer
    writes payloads into `payloads/`)."""
    for sub in ("", PAYLOADS_DIR, REPORTS_DIR, RAW_DIR, PLOTS_DIR):
        path = log_dir / name if not sub else log_dir / sub / name
        if path.exists():
            return path
    return log_dir / PAYLOADS_DIR / name


def load_payload(log_dir: Path, name: str) -> dict:
    return json.loads(resolve_artifact(log_dir, name).read_text())


def plot_output_path(log_dir: Path, name: str) -> Path:
    path = log_dir / PLOTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
