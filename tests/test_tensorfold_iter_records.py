"""TensorFold iter-record → speculative timing-predict case conversion (CPU only)."""

from __future__ import annotations

import json

import pytest

from alignment.timing_predict_input.tensorfold_iter_records import (
    ITERATION_RANGE,
    PREFILL_CHUNK_RANGE,
    Phase,
    build_cases,
    decode_outcome,
    main,
    read_phases,
    read_records,
)

CHAIN4 = [-1, 0, 1, 2]
# Two branches off the root: the root's state must stay live while one is walked.
FORK = [-1, 0, 0, 1, 2]


def _stream(sid, pos, parents, mode, accepted):
    return {
        "sid": sid,
        "pos": pos,
        "rows": len(parents),
        "parents": parents,
        "mode": mode,
        "accepted": accepted,
    }


def _multi(start, end, prefill=None, streams=None):
    return {
        "kind": "multi_round",
        "start_ns": start,
        "end_ns": end,
        "prefill": prefill,
        "streams": streams,
    }


def test_decode_outcome_wire_format():
    # Lane S (tf-predsched) owns the final contract; this pins what we emit today.
    assert decode_outcome(3, 1, "tree") == [3, 1, "tree"]


def test_multi_round_lowers_prefill_streams_and_outcomes():
    record = _multi(
        100,
        250,
        prefill=[[7, 4096, 5120], [8, 0, 300]],
        streams=[
            _stream(0, 1000, FORK, 1, 2),
            _stream(1, 50, CHAIN4, 0, 3),
            _stream(2, 9, [-1], 2, 0),
        ],
    )
    cases, sidecar, excluded = build_cases([record])
    assert excluded == []
    assert cases == [
        {
            "groups": [
                {
                    "prefill_chunk_pairs": [[4096, 1024], [0, 300]],
                    "decode_requests": [[1005, 5], [54, 4], [10, 1]],
                    "decode_outcomes": [[2, 1, "tree"], [3, 0, "copy"], [0, 0, "one"]],
                }
            ]
        }
    ]
    entry = sidecar[0]
    assert entry["wall_ns"] == 150
    assert entry["nvtx_range"] == ITERATION_RANGE
    assert (entry["streams"], entry["tree_streams"], entry["verify_rows"]) == (3, 1, 10)
    assert (entry["prefill_pieces"], entry["prefill_rows"]) == (2, 1324)
    assert entry["level"] is None and entry["round_index"] == 0 and entry["record_line"] == 1


def test_prefill_only_and_decode_only_rounds():
    cases, _, _ = build_cases(
        [
            _multi(0, 10, prefill=[[0, 0, 64]]),
            _multi(10, 20, streams=[_stream(0, 64, CHAIN4, 1, 0)]),
        ]
    )
    assert cases[0]["groups"][0]["decode_requests"] == []
    assert cases[0]["groups"][0]["decode_outcomes"] == []
    assert cases[1]["groups"][0]["prefill_chunk_pairs"] == []


def test_phases_filter_levels_and_edges():
    phases = [Phase(1, 100, 200), Phase(2, 300, 400)]
    stream = [_stream(0, 10, CHAIN4, 1, 1)]
    records = [
        _multi(50, 90, streams=stream),  # before every window: never captured
        _multi(110, 150, streams=stream),
        _multi(150, 190, streams=stream),
        _multi(190, 210, streams=stream),  # straddles level 1's end
        _multi(310, 320, streams=stream),
        {
            "kind": "prefill_chunk",
            "start_ns": 320,
            "end_ns": 330,
            "pos": 0,
            "rows": 5,
            "last": True,
        },
        _multi(500, 600, streams=stream),  # after every window
    ]
    cases, sidecar, excluded = build_cases(records, phases)
    assert len(cases) == 3
    assert [(e["level"], e["round_index"], e["record_line"]) for e in sidecar] == [
        (1, 0, 2),
        (1, 1, 3),
        (2, 0, 5),
    ]
    assert [(e["record_line"], e["reason"]) for e in excluded] == [
        (4, "straddles a capture window edge"),
        (6, "sub-step of a multi_round prefill"),
    ]


def test_bad_records_are_excluded_with_reasons():
    records = [
        _multi(0, 10),
        _multi(0, 10, streams=[_stream(0, 5, CHAIN4, 1, 4)]),  # accepted == rows
        _multi(0, 10, streams=[_stream(0, 5, CHAIN4, 9, 0)]),
        _multi(0, 10, streams=[_stream(0, 5, [0, 0], 1, 0)]),  # no root
        _multi(0, 10, prefill=[[0, 5, 5]]),
        _multi(10, 0, streams=[_stream(0, 5, CHAIN4, 1, 0)]),
        _multi(0, 10, streams=[_stream(0, 5, FORK, 0, 0)]),  # a copy window that branches
        _multi(0, 10, streams=[_stream(0, 5, CHAIN4, 2, 0)]),  # a ONE window of 4 rows
        {"kind": "mystery", "start_ns": 0, "end_ns": 1},
    ]
    cases, _, excluded = build_cases(records)
    assert cases == []
    reasons = [e["reason"] for e in excluded]
    assert reasons[0] == "round has neither prefill nor streams"
    assert "accepted" in reasons[1]
    assert "unknown stream mode" in reasons[2]
    assert "root" in reasons[3]
    assert "invalid prefill piece" in reasons[4]
    assert reasons[5] == "invalid start/end_ns"
    assert reasons[6] == "a copy window needs 1 tree slots"
    assert reasons[7] == "a one window has 4 rows"
    assert reasons[8] == "unknown record kind"


def test_single_stream_round_and_prefill_chunk():
    records = [
        {"kind": "prefill_chunk", "start_ns": 0, "end_ns": 5, "pos": 0, "rows": 177, "last": True},
        {
            "kind": "round",
            "start_ns": 10,
            "end_ns": 30,
            "pos": 177,
            "rows": 5,
            "parents": FORK,
            "source": "tree",
            "accepted": 2,
        },
        {
            "kind": "round",
            "start_ns": 30,
            "end_ns": 40,
            "pos": 180,
            "rows": 1,
            "parents": [-1],
            "source": "serial",
            "accepted": 0,
        },
    ]
    cases, sidecar, excluded = build_cases(records)
    assert excluded == []
    assert cases[0]["groups"][0] == {
        "prefill_chunk_pairs": [[0, 177]],
        "decode_requests": [],
        "decode_outcomes": [],
    }
    assert cases[1]["groups"][0]["decode_requests"] == [[182, 5]]
    assert cases[1]["groups"][0]["decode_outcomes"] == [[2, 1, "tree"]]
    assert cases[2]["groups"][0]["decode_outcomes"] == [[0, 0, "one"]]
    assert [e["nvtx_range"] for e in sidecar] == [
        PREFILL_CHUNK_RANGE,
        ITERATION_RANGE,
        ITERATION_RANGE,
    ]


def test_mixed_engines_are_rejected():
    single = {"kind": "round", "start_ns": 0, "end_ns": 1, "pos": 0, "rows": 1, "parents": [-1],
              "source": "serial", "accepted": 0}  # fmt: skip
    with pytest.raises(ValueError, match="mix"):
        build_cases([_multi(0, 1, prefill=[[0, 0, 1]]), single])


def test_readers_and_cli(tmp_path):
    records = tmp_path / "iter_records.jsonl"
    records.write_text(
        json.dumps(_multi(110, 150, streams=[_stream(0, 10, CHAIN4, 1, 1)]))
        + "\n"
        + '{"kind": "multi_ro'  # a killed server's torn last line
    )
    phases = tmp_path / "phases.tsv"
    phases.write_text("1\t100\t200\n")
    assert len(read_records(records)) == 1
    assert read_phases(phases) == [Phase(1, 100, 200)]

    out = tmp_path / "out"
    assert main([str(records), "--phases", str(phases), "--output-dir", str(out)]) == 0
    cases = json.loads((out / "timing_predict_cases.json").read_text())
    sidecar = json.loads((out / "tensorfold_round_sidecar.json").read_text())
    assert len(cases) == len(sidecar["cases"]) == 1
    assert sidecar["cases"][0]["level"] == 1


def test_torn_line_in_the_middle_fails(tmp_path):
    records = tmp_path / "iter_records.jsonl"
    records.write_text('{"kind": "multi_ro\n' + json.dumps(_multi(0, 1, prefill=[[0, 0, 1]])))
    with pytest.raises(ValueError, match="not a JSON record"):
        read_records(records)


def test_outcomes_follow_the_lowering_rules():
    """Every emitted entry passes ``SpeculativePredictGroup``'s outcome checks."""
    streams = [
        _stream(0, 10, FORK, 1, 4),
        _stream(1, 10, CHAIN4, 1, 0),  # a tree window that drafted a chain: 0 slots
        _stream(2, 10, CHAIN4, 0, 3),
        _stream(3, 10, [-1], 2, 0),
    ]
    cases, _, excluded = build_cases([_multi(0, 1, prefill=[[9, 0, 1024]], streams=streams)])
    assert excluded == []
    group = cases[0]["groups"][0]
    assert len(group["decode_outcomes"]) == len(group["decode_requests"])
    for (kv_len, query_len), outcome in zip(
        group["decode_requests"], group["decode_outcomes"], strict=True
    ):
        accepted, slots, mode = outcome
        assert len(outcome) == 3 and mode in {"copy", "tree", "one"}
        assert 0 <= accepted < query_len <= kv_len
        assert slots == 0 or mode == "tree"
        assert mode != "one" or query_len == 1
