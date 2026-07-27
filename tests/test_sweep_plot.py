from __future__ import annotations

import json
import sys
from pathlib import Path

ANALYZER_PYTHON = Path(__file__).resolve().parents[1] / "analyzer" / "python"
sys.path.insert(0, str(ANALYZER_PYTHON))

from sweep.grid_plot import _metric_matrix, render  # noqa: E402


def _payload(axes: list[str], domains: dict, runs: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "meta": {"experiment_dir": "logs/test_sweep"},
        "axes": axes,
        "domains": domains,
        "metrics": [
            {
                "key": "total_tps",
                "label": "Total throughput",
                "unit": "tok/s",
                "group": "throughput",
            }
        ],
        "runs": runs,
    }


def test_metric_matrix_keeps_missing_cells_and_selects_facet() -> None:
    payload = _payload(
        ["tp", "rate", "gpu"],
        {"tp": [2, 4], "rate": [40, 60], "gpu": ["H100", "H200"]},
        [
            {
                "coordinates": {"tp": 2, "rate": 40, "gpu": "H200"},
                "metrics": {"total_tps": 100},
            },
            {
                "coordinates": {"tp": 4, "rate": 60, "gpu": "H200"},
                "metrics": {"total_tps": 200},
            },
            {
                "coordinates": {"tp": 2, "rate": 40, "gpu": "H100"},
                "metrics": {"total_tps": 50},
            },
        ],
    )

    matrix = _metric_matrix(
        payload,
        payload["metrics"][0],
        {"gpu": "H200"},
    )

    assert matrix[0][0] == 100
    assert matrix[1][1] == 200
    assert matrix[0][1] != matrix[0][1]  # NaN: missing cell remains blank.


def test_render_emits_line_plot_for_one_axis(tmp_path: Path) -> None:
    payload = _payload(
        ["rate"],
        {"rate": [40, 60]},
        [
            {
                "coordinates": {"rate": 40},
                "labels": {},
                "metrics": {"total_tps": 100},
            },
            {
                "coordinates": {"rate": 60},
                "labels": {},
                "metrics": {"total_tps": 150},
            },
        ],
    )
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    (payload_dir / "sweep_metrics_grid.json").write_text(json.dumps(payload))

    jobs = render(tmp_path)
    paths = [job() for job in jobs]

    assert paths == [tmp_path / "plots/sweep_throughput.png"]
    assert paths[0].is_file()


def test_render_emits_faceted_heatmap_for_three_axes(tmp_path: Path) -> None:
    payload = _payload(
        ["tp", "rate", "gpu"],
        {"tp": [2, 4], "rate": [40, 60], "gpu": ["H100", "H200"]},
        [
            {
                "coordinates": {"tp": tp, "rate": rate, "gpu": gpu},
                "labels": {},
                "metrics": {"total_tps": tp * rate},
            }
            for gpu in ["H100", "H200"]
            for rate in [40, 60]
            for tp in [2, 4]
        ],
    )
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    (payload_dir / "sweep_metrics_grid.json").write_text(json.dumps(payload))

    jobs = render(tmp_path)
    paths = [job() for job in jobs]

    assert paths == [tmp_path / "plots/sweep_total_tps.png"]
    assert paths[0].is_file()
