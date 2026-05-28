"""Render the SLO CDFs from a Rust SLO payload.

Shared by both SLO subjects (`__main__` binds the payload file per subject):
`slo_general_cdf.json` (ttft / tpot / e2e) and `slo_detailed_cdf.json`
(itl). One PNG per metric series. Reads only the payload — no parquet. `render`
returns one *job* per figure (a callable that draws it and returns its path);
`__main__` runs them in parallel.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

from common.cdf_plot import render_cdf
from common.layout import load_payload, plot_output_path


def render(log_dir: Path, payload_name: str) -> list[Callable[[], Path]]:
    payload = load_payload(log_dir, payload_name)
    if not payload.get("series"):
        reason = payload.get("meta", {}).get("reason", f"no series in {payload_name}")
        print(f"[slo_plot] nothing to render ({payload_name}): {reason}")
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
