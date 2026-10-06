import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trace"))

from session_to_prefill_only import write_prefill_only_trace  # noqa: E402

SESSION_HEADER = (
    "request_id,session_id,round_idx,arrival_time_ms,prefix_len,input_len,output_len,"
    "tool_wait_after_ms\n"
)


def _session_trace(path: Path, rounds: list[tuple[int, int]]) -> Path:
    lines = [SESSION_HEADER]
    lines.extend(
        f"session_0_round_{index:06d},0,{index},0.000000,{prefix},{fresh},50,10.000000\n"
        for index, (prefix, fresh) in enumerate(rounds)
    )
    path.write_text("".join(lines))
    return path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_rounds_become_prefill_only_requests_with_pinned_prefix(tmp_path):
    source = _session_trace(tmp_path / "s.csv", [(0, 100), (150, 20), (180, 30)])
    output = tmp_path / "p.csv"

    manifest = write_prefill_only_trace(source, output, requests=None, seed=0, max_context=1024)

    rows = _rows(output)
    assert list(rows[0]) == ["id", "input_len", "output_len", "arrival_time", "prefix_len"]
    by_id = {row["id"]: (int(row["prefix_len"]), int(row["input_len"])) for row in rows}
    assert by_id == {
        "session_0_round_000000": (0, 100),
        "session_0_round_000001": (150, 20),
        "session_0_round_000002": (180, 30),
    }
    assert {row["output_len"] for row in rows} == {"1"}
    arrivals = [float(row["arrival_time"]) for row in rows]
    assert arrivals[0] == 0.0 and arrivals == sorted(arrivals)
    assert manifest["total_fresh_tokens"] == 150
    assert manifest["total_prefix_tokens"] == 330
    assert json.loads(output.with_suffix(".manifest.json").read_text()) == manifest


def test_same_seed_samples_the_same_rounds_and_arrivals(tmp_path):
    rounds = [(index * 10, index + 1) for index in range(50)]
    first = _session_trace(tmp_path / "a.csv", rounds)
    # Another policy splits the same context differently.
    second = _session_trace(tmp_path / "b.csv", [(p + f - 1, 1) for p, f in rounds])

    write_prefill_only_trace(first, tmp_path / "a_out.csv", requests=7, seed=3, max_context=4096)
    write_prefill_only_trace(second, tmp_path / "b_out.csv", requests=7, seed=3, max_context=4096)

    a_rows, b_rows = _rows(tmp_path / "a_out.csv"), _rows(tmp_path / "b_out.csv")
    assert len(a_rows) == 7
    assert [(r["id"], r["arrival_time"]) for r in a_rows] == [
        (r["id"], r["arrival_time"]) for r in b_rows
    ]


def test_rejects_a_context_past_max_context(tmp_path):
    source = _session_trace(tmp_path / "s.csv", [(1000, 24)])
    with pytest.raises(ValueError, match="exceed max_context 1024"):
        write_prefill_only_trace(
            source, tmp_path / "p.csv", requests=None, seed=0, max_context=1024
        )


def test_a_full_prefix_round_recomputes_its_last_token(tmp_path):
    source = _session_trace(tmp_path / "s.csv", [(500, 0), (0, 40)])
    output = tmp_path / "p.csv"

    manifest = write_prefill_only_trace(source, output, requests=None, seed=0, max_context=1024)

    by_id = {row["id"]: (int(row["prefix_len"]), int(row["input_len"])) for row in _rows(output)}
    assert by_id["session_0_round_000000"] == (499, 1)
    assert by_id["session_0_round_000001"] == (0, 40)
    assert manifest["source_full_prefix_rounds_given_one_fresh_token"] == 1
