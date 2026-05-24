"""Render the SLO CDFs from the Rust `slo_cdf.json` payload.

One PNG per metric series: ttft / tpot / itl / e2e / session_e2e. Reads only the
payload — no parquet. `render` returns one *job* per figure (a callable that draws
it and returns its path); `__main__` runs them in parallel.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

from common.cdf_plot import render_cdf
from common.layout import load_payload, plot_output_path


def render(log_dir: Path) -> list[Callable[[], Path]]:
    payload = load_payload(log_dir, "slo_cdf.json")
    if not payload.get("series"):
        reason = payload.get("meta", {}).get("reason", "no series in slo_cdf.json")
        print(f"[slo_plot] nothing to render: {reason}")
        return []
    run_label = Path(payload.get("meta", {}).get("log_dir", str(log_dir))).name
    return [
        partial(
            render_cdf,
            series,
            plot_output_path(log_dir, f"{series['key']}_cdf.png"),
            run_label=run_label,
        )
        for series in payload["series"]
    ]
