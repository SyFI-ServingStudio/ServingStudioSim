from __future__ import annotations

import json
from pathlib import Path

from launcher import alignment as alignment_launcher


def _aggregate(signed_pct: float, absolute_pct: float, absolute_ms: float) -> dict:
    return {
        "n": 2,
        "measured_ms": 100.0,
        "simulated_ms": 100.0 + signed_pct,
        "signed_error_ms": signed_pct,
        "signed_error_pct": signed_pct,
        "absolute_error_ms": absolute_ms,
        "absolute_error_pct": absolute_pct,
    }


def _write_report(path: Path, *, all_row: dict, stages: dict, operations: dict) -> None:
    report = path / "reports" / "alignment_iteration_report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "available": True,
                "comparison": {"all": all_row, "by_stage": stages},
                "operations": [
                    {
                        "operation": operation,
                        "comparison": comparison,
                        "missing_measured": 0,
                        "missing_simulated": 0,
                    }
                    for operation, comparison in operations.items()
                ],
            }
        )
    )


def test_compare_uses_report_aggregates_and_sorts_operation_changes(tmp_path, capsys):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_report(
        baseline,
        all_row=_aggregate(4.0, 6.0, 6.0),
        stages={"decode": _aggregate(2.0, 3.0, 3.0)},
        operations={
            "attention": _aggregate(1.0, 2.0, 2.0),
            "moe": _aggregate(3.0, 5.0, 5.0),
        },
    )
    _write_report(
        candidate,
        all_row=_aggregate(1.0, 4.0, 4.0),
        stages={"decode": _aggregate(0.5, 2.0, 2.0)},
        operations={
            "attention": _aggregate(1.0, 7.0, 7.0),
            "moe": _aggregate(0.0, 4.0, 4.0),
        },
    )

    assert alignment_launcher.main(["compare", str(baseline), str(candidate), "--json"]) == 0
    comparison = json.loads(capsys.readouterr().out)

    assert comparison["scopes"][0]["scope"] == "all"
    assert comparison["scopes"][0]["change_absolute_error_pct"] == -2.0
    assert [row["operation"] for row in comparison["operations"]] == ["attention", "moe"]
    assert comparison["operations"][0]["change_absolute_error_ms"] == 5.0


def test_compare_rejects_reports_without_current_aggregates(tmp_path, capsys):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for path in (baseline, candidate):
        report = path / "reports" / "alignment_iteration_report.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({"available": True}))

    assert alignment_launcher.main(["compare", str(baseline), str(candidate)]) == 2
    assert "rerun alignment analyze" in capsys.readouterr().err


def test_compare_preserves_an_operation_without_paired_duration(tmp_path, capsys):
    empty = {
        "n": 0,
        "measured_ms": 0.0,
        "simulated_ms": 0.0,
        "signed_error_ms": 0.0,
        "signed_error_pct": None,
        "absolute_error_ms": 0.0,
        "absolute_error_pct": None,
    }
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for path in (baseline, candidate):
        _write_report(
            path,
            all_row=_aggregate(0.0, 0.0, 0.0),
            stages={},
            operations={"new": empty},
        )

    assert alignment_launcher.main(["compare", str(baseline), str(candidate), "--json"]) == 0
    operation = json.loads(capsys.readouterr().out)["operations"][0]
    assert operation["baseline_absolute_error_pct"] is None
    assert operation["change_absolute_error_pct"] is None
