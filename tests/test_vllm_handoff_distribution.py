"""Extraction of the stock vLLM Neuron hand-off/planning distributions.

Synthetic capture directories exercise the extraction; the checked-in data file
is checked for the shape the simulator embeds.
"""

import json
from pathlib import Path

from alignment.neuron.vllm_handoff_distribution import build_distribution

DATA = Path(__file__).resolve().parents[1] / "alignment/neuron/data/vllm_neuron_handoff.json"
MS = 1_000_000


def write_capture(
    root: Path, name: str, arrivals_ms, receives_ms, plans_ms, ends_ms, phases, sends_ms=None
):
    capture = root / name
    capture.mkdir()
    sends_ms = sends_ms or [None] * len(arrivals_ms)
    with (capture / "request_timings.jsonl").open("w") as handle:
        for arrival, receive, send in zip(arrivals_ms, receives_ms, sends_ms):
            row = {
                "frontend_arrival_monotonic_ns": int(arrival * MS),
                "engine_receive_monotonic_ns": int(receive * MS),
            }
            if send != "absent":  # arrival sub-version 1 rows lack the key
                row["api_send_monotonic_ns"] = None if send is None else int(send * MS)
            handle.write(json.dumps(row) + "\n")
    with (capture / "metrics.jsonl").open("w") as handle:
        for plan, end, phase in zip(plans_ms, ends_ms, phases):
            handle.write(
                json.dumps(
                    {
                        "observed_start_monotonic_ns": int(plan * MS),
                        "observed_end_monotonic_ns": int(end * MS),
                        "phase": phase,
                    }
                )
                + "\n"
            )


def test_extraction_reads_ingest_gaps_plan_offsets_and_outcome(tmp_path):
    # Arrival order is not file order; gaps come from sorted arrivals.
    write_capture(
        tmp_path,
        "c512-1",
        arrivals_ms=[1.5, 0.0, 0.5],
        receives_ms=[60.0, 20.0, 60.0],
        plans_ms=[20.0, 21.0, 60.2, 100.1],
        ends_ms=[60.0, 100.0, 140.0, 150.0],
        phases=["prefill", "decode", "prefill", "prefill"],
    )
    write_capture(
        tmp_path,
        "c8192-1",
        arrivals_ms=[0.0, 2.0],
        receives_ms=[30.0, 30.5],
        plans_ms=[30.0, 30.6, 80.1],
        ends_ms=[80.0, 130.0, 180.0],
        phases=["prefill", "prefill", "decode"],
    )
    (tmp_path / "c512-2").mkdir()  # failed capture without records

    data = build_distribution(tmp_path)

    assert data["provenance"]["captures"] == ["c512-1", "c8192-1"]
    assert data["provenance"]["handoff_gap_source"] == {
        "c512-1": "frontend_arrival",
        "c8192-1": "frontend_arrival",
    }
    assert "c512-2" in data["provenance"]["skipped"]
    assert data["ingest_delay_ms"] == [20.0, 30.0]
    assert data["handoff_gap_ms"] == [0.5, 1.0, 2.0]
    assert data["first_plan_offset_ms"] == [1.0, 0.6]
    assert data["outcome"]["extra_solo_decode"] == 1
    assert data["outcome"]["steps"] == {"c512-1": 4, "c8192-1": 3}
    # Step 3 of c512-1 is planned 0.1 ms after step 1 finishes.
    assert abs(data["steady_plan_offset_median_ms"] - 0.1) < 1e-9


def test_checked_in_distribution_matches_the_race_campaign():
    data = json.loads(DATA.read_text())
    captures = data["provenance"]["captures"]
    assert len(captures) == data["outcome"]["captures"] == 13
    assert len(data["ingest_delay_ms"]) == 13
    assert len(data["handoff_gap_ms"]) == 13 * 15
    assert data["handoff_gap_ms"] == sorted(data["handoff_gap_ms"])
    assert data["outcome"]["extra_solo_decode"] == 10
    assert min(data["ingest_delay_ms"]) > 0


def test_extraction_prefers_measured_api_send_gaps(tmp_path):
    common = dict(
        plans_ms=[30.0, 31.0, 80.1, 90.1],
        ends_ms=[80.0, 90.0, 100.0, 110.0],
        phases=["prefill", "prefill", "decode", "decode"],
    )
    # Stamped capture: send gaps (0.25, 0.5 ms) replace the arrival gaps (0, 0 ms).
    write_capture(
        tmp_path,
        "c512-1",
        [0.0, 0.0, 0.0],
        [30.0, 31.0, 32.0],
        sends_ms=[1.0, 1.75, 1.25],
        **common,
    )
    # One request without a stamp: the capture keeps its frontend-gap fallback.
    write_capture(tmp_path, "c512-2", [0.0, 2.0], [30.0, 31.0], sends_ms=[1.0, None], **common)
    # Arrival sub-version 1 rows have no send field at all.
    write_capture(
        tmp_path, "c512-3", [0.0, 3.0], [30.0, 31.0], sends_ms=["absent", "absent"], **common
    )

    data = build_distribution(tmp_path)

    assert data["provenance"]["handoff_gap_source"] == {
        "c512-1": "api_send",
        "c512-2": "frontend_arrival",
        "c512-3": "frontend_arrival",
    }
    assert data["handoff_gap_ms"] == [0.25, 0.5, 2.0, 3.0]
    # The ingest delay still runs from the first frontend arrival.
    assert data["ingest_delay_ms"] == [30.0, 30.0, 30.0]
