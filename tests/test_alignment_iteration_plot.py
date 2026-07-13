from __future__ import annotations

import os
import sys
from pathlib import Path

ANALYZER_PYTHON = Path(__file__).resolve().parents[1] / "analyzer" / "python"
sys.path.insert(0, str(ANALYZER_PYTHON))

from alignment_iteration.series_plot import (  # noqa: E402
    _evenly_sample_breakdowns,
    _output_is_current,
    _remove_stale_breakdown_outputs,
)


def test_evenly_sample_breakdowns_caps_and_keeps_raw_endpoints() -> None:
    rows = [{"iteration_id": iteration_id} for iteration_id in range(6, 2471)]

    sampled = _evenly_sample_breakdowns(rows)

    sampled_ids = [row["iteration_id"] for row in sampled]
    assert len(sampled_ids) == 128
    assert sampled_ids[0] == 6
    assert sampled_ids[-1] == 2470
    assert sampled_ids == sorted(set(sampled_ids))
    gaps = [right - left for left, right in zip(sampled_ids, sampled_ids[1:])]
    assert max(gaps) - min(gaps) <= 1


def test_evenly_sample_breakdowns_sorts_and_keeps_small_inputs() -> None:
    rows = [{"iteration_id": 9}, {"iteration_id": 3}, {"iteration_id": 7}]

    assert [row["iteration_id"] for row in _evenly_sample_breakdowns(rows)] == [3, 7, 9]


def test_output_is_current_tracks_every_render_input(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    renderer = tmp_path / "renderer.py"
    output = tmp_path / "plot.png"
    payload.write_text("payload")
    renderer.write_text("renderer")

    assert not _output_is_current(output, payload, renderer)

    output.write_bytes(b"png")
    os.utime(payload, ns=(100, 100))
    os.utime(renderer, ns=(200, 200))
    os.utime(output, ns=(300, 300))
    assert _output_is_current(output, payload, renderer)

    os.utime(payload, ns=(400, 400))
    assert not _output_is_current(output, payload, renderer)


def test_remove_stale_breakdowns_keeps_selected_jpg_fallback(tmp_path: Path) -> None:
    plots = tmp_path / "plots"
    selected_dir = plots / "iter_1_to_32"
    stale_dir = plots / "iter_33_to_64"
    selected_dir.mkdir(parents=True)
    stale_dir.mkdir()
    selected_png = selected_dir / "iter_1_breakdown.png"
    selected_jpg = selected_png.with_suffix(".jpg")
    stale_jpg = stale_dir / "iter_33_breakdown.jpg"
    selected_jpg.write_bytes(b"fallback")
    stale_jpg.write_bytes(b"stale")

    _remove_stale_breakdown_outputs(tmp_path, {selected_png})

    assert selected_jpg.is_file()
    assert not stale_jpg.exists()
    assert not stale_dir.exists()
