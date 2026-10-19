"""Extraction of the stock vLLM Neuron hand-off/planning distributions.

Synthetic capture directories exercise the extraction; the checked-in data file
is checked for the shape the simulator embeds.
"""

import json
from pathlib import Path

from alignment.neuron.vllm_handoff_distribution import build_distribution

DATA = Path(__file__).resolve().parents[1] / "alignment/neuron/data/vllm_neuron_handoff.json"
MS = 1_000_000


def write_capture(root: Path, name: str, arrivals_ms, sends_ms, receives_ms, plans_ms, phases):
    capture = root / name
    capture.mkdir(parents=True)
    with (capture / "request_timings.jsonl").open("w") as handle:
        for arrival, send, receive in zip(arrivals_ms, sends_ms, receives_ms):
            row = {
                "frontend_arrival_monotonic_ns": int(arrival * MS),
                "engine_receive_monotonic_ns": int(receive * MS),
            }
            if send != "absent":  # arrival sub-version 1 rows lack the key
                row["api_send_monotonic_ns"] = None if send is None else int(send * MS)
            handle.write(json.dumps(row) + "\n")
    with (capture / "metrics.jsonl").open("w") as handle:
        for index, (plan, phase) in enumerate(zip(plans_ms, phases)):
            step = {
                "observed_start_monotonic_ns": int(plan * MS),
                "observed_end_monotonic_ns": int((plan + 40.0) * MS),
                "phase": phase,
                "iteration_index": index,
            }
            handle.write(json.dumps(step) + "\n")


def stamped(root, name, first_gap, plan_offset, ingest=20.0, transport=0.5, solo=True):
    """Four simultaneous requests: send gaps first_gap, 0.25, 0.5 (file order shuffled)."""
    sends = [ingest, ingest + first_gap, ingest + first_gap + 0.25, ingest + first_gap + 0.75]
    receive0 = ingest + transport
    plan1 = receive0 + plan_offset
    # Request 1 is drained before step 1 when it beats the plan, else after step 0.
    late = receive0 + 40.0
    receives = [receive0, late if solo else plan1 - 0.02, late, late]
    order = [2, 0, 3, 1]
    write_capture(
        root,
        name,
        arrivals_ms=[0.0] * 4,
        sends_ms=[sends[i] for i in order],
        receives_ms=[receives[i] for i in order],
        plans_ms=[receive0 + 0.06, plan1, late + 0.1],
        phases=["prefill", "decode" if solo else "prefill", "prefill"],
    )


def test_extraction_composes_position_aware_inputs_per_context(tmp_path):
    stamped_root, old_root = tmp_path / "race2", tmp_path / "race"
    # Contexts 512 and 8192 differ clearly in first gap and plan offset.
    stamped(stamped_root, "c512-1", first_gap=1.1, plan_offset=0.7, transport=0.5)
    stamped(stamped_root, "c512-2", first_gap=1.2, plan_offset=0.8, transport=0.9)
    stamped(stamped_root, "c8192-1", first_gap=0.5, plan_offset=1.2, transport=0.7, solo=False)
    stamped(stamped_root, "c8192-2", first_gap=0.4, plan_offset=1.5, transport=0.6, solo=False)
    # Unstamped: outcome only.
    write_capture(
        old_root,
        "c1024-1",
        arrivals_ms=[0.0, 1.0],
        sends_ms=["absent", "absent"],
        receives_ms=[20.0, 60.0],
        plans_ms=[20.1, 21.0, 60.1],
        phases=["prefill", "decode", "prefill"],
    )
    (old_root / "c512-9").mkdir()  # failed capture without records

    data = build_distribution([stamped_root, old_root])

    assert data["schema_version"] == 2
    assert data["provenance"]["model_captures"] == [
        "race2/c512-1",
        "race2/c512-2",
        "race2/c8192-1",
        "race2/c8192-2",
    ]
    assert data["provenance"]["outcome_only_captures"] == ["race/c1024-1"]
    assert "race/c512-9" in data["provenance"]["skipped"]
    keys = {quantity: entry["key"] for quantity, entry in data["keying"].items()}
    assert keys == {
        "ingest_ms": "pooled",
        "first_gap_ms": "context",
        "send_to_visible_ms": "pooled",
        "first_plan_offset_ms": "context",
        "later_gap_ms": "pooled",
    }
    c512, c8192 = data["contexts"]["512"], data["contexts"]["8192"]
    assert c512["first_gap_ms"] == [1.1, 1.2]
    assert c8192["first_gap_ms"] == [0.5, 0.4]
    assert c512["first_plan_offset_ms"] == [0.7, 0.8]
    assert c8192["first_plan_offset_ms"] == [1.2, 1.5]
    # Pooled quantities repeat every stamped capture in every context.
    assert c512["send_to_visible_ms"] == c8192["send_to_visible_ms"] == [0.5, 0.9, 0.7, 0.6]
    assert c512["ingest_ms"] == [20.0] * 4
    assert c512["later_gap_ms"] == [0.25] * 4 + [0.5] * 4
    # A later receive at the drain before step 1 bounds its transport from above.
    assert c512["later_send_to_receive_upper_bound_ms"] == []
    assert [round(v, 6) for v in c8192["later_send_to_receive_upper_bound_ms"]] == [1.38, 1.68]
    assert c512["outcome"] == {"captures": 2, "extra_solo_decode": 2}
    assert c8192["outcome"] == {"captures": 2, "extra_solo_decode": 0}
    assert data["outcome"]["by_context"]["1024"] == {
        "stamped": {},
        "unstamped": {"race/c1024-1": 3},
    }


def test_checked_in_distribution_matches_the_stamped_campaign():
    data = json.loads(DATA.read_text())
    assert data["schema_version"] == 2
    assert len(data["provenance"]["model_captures"]) == 11
    assert all(name.startswith("race2/") for name in data["provenance"]["model_captures"])
    assert sorted(data["contexts"], key=int) == ["512", "2048", "8192"]
    for entry in data["contexts"].values():
        captures = len(entry["captures"])
        assert len(entry["ingest_ms"]) == len(entry["first_gap_ms"]) == captures
        assert len(entry["first_plan_offset_ms"]) == captures
        assert len(entry["send_to_visible_ms"]) == 11
        assert len(entry["later_gap_ms"]) == 11 * 14
        assert entry["later_gap_ms"] == sorted(entry["later_gap_ms"])
    keys = {quantity: entry["key"] for quantity, entry in data["keying"].items()}
    assert keys["first_gap_ms"] == keys["first_plan_offset_ms"] == "context"
    assert keys["send_to_visible_ms"] == "pooled"
    extra = sum(entry["outcome"]["extra_solo_decode"] for entry in data["contexts"].values())
    assert extra == 7
