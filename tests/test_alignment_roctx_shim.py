"""CPU test for the ROCm roctx iteration shim (``alignment/profiler/roctx_shim``).

The shim is the AMD analog of the vLLM NVTX fork: it emits
``vllm_iteration(N): forward`` roctx ranges and the ``VibeSimAlignmentIteration``
boundary marker. This test proves two things without a GPU or a real roctx stack:

1. the labels it emits match the parser's ``ITER_RE`` (contract parity with the
   nsys/rocpd side) and the counter/record behaviour is correct;
2. the regions it would record, written into a synthetic rocpd and run through
   the *real* ``alignment/rocpd`` ownership join, are recognized as iteration
   markers and own the dispatches launched inside them — i.e. a capture driven by
   this shim feeds the existing offline producer unchanged.

Only the roctx backend is replaced (by the in-module recording backend); the join
and parse code are the real thing.
"""

from __future__ import annotations

import json
import sqlite3

from alignment.nsys.parse import ITER_RE
from alignment.profiler import roctx_shim
from alignment.profiler.roctx_shim import (
    ITERATION_RECORD_TAG,
    IterationAnnotator,
    RecordingBackend,
    iteration_label,
    iteration_record_marker,
    roctx_scopes_enabled,
    select_backend,
)
from alignment.rocpd.evidence import build_ranges_from_rocpd

_GUID = "0000b1c7_c35b_735b_96a7_f0a02ff013cc"
_PID = 4242
_TID = 55
_AGENT = 1


def test_emitted_label_matches_parser_regex():
    label = iteration_label(3, "forward")
    assert label == "vllm_iteration(3): forward"
    match = ITER_RE.match(label)
    assert match is not None
    assert (int(match.group(1)), match.group(2)) == (3, "forward")
    # Every contract phase produces an ITER_RE-matching label.
    for phase in roctx_shim.PHASES:
        assert ITER_RE.match(iteration_label(0, phase)) is not None


def test_sglang_prefix_also_matches():
    assert ITER_RE.match(iteration_label(7, "forward", prefix="sglang")) is not None


def test_iteration_record_marker_is_tagged_json():
    marker = iteration_record_marker({"total_num_scheduled_tokens": 12}, iteration=5)
    tag, _, payload = marker.partition(" ")
    assert tag == ITERATION_RECORD_TAG
    body = json.loads(payload)
    assert body["iteration"] == 5
    assert body["total_num_scheduled_tokens"] == 12
    assert body["schema_version"] == roctx_shim.ITERATION_RECORD_SCHEMA_VERSION


def test_annotator_counts_and_emits_record_before_range():
    backend = RecordingBackend()
    annotator = IterationAnnotator(backend, enabled=True)
    with annotator.forward(record={"total_num_scheduled_tokens": 4}) as n0:
        pass
    with annotator.forward(record={"total_num_scheduled_tokens": 1}) as n1:
        pass
    assert (n0, n1) == (0, 1)
    ops = [(e.op, e.message) for e in backend.events]
    # record mark precedes the forward push, pop closes it (NVIDIA ordering).
    assert ops[0][0] == "mark" and ops[0][1].startswith(ITERATION_RECORD_TAG)
    assert ops[1] == ("push", "vllm_iteration(0): forward")
    assert ops[2][0] == "pop"
    assert ops[4] == ("push", "vllm_iteration(1): forward")


def test_disabled_annotator_is_a_noop():
    backend = RecordingBackend()
    annotator = IterationAnnotator(backend, enabled=False)
    with annotator.forward(record={"x": 1}):
        pass
    annotator.emit_iteration_record({"x": 1}, iteration=0)
    assert backend.events == []


def test_env_gate_and_select_backend_record():
    assert roctx_scopes_enabled({}) is False
    assert roctx_scopes_enabled({roctx_shim.ROCTX_SCOPES_ENV: "1"}) is True
    name, backend = select_backend("record")
    assert name == "record"
    assert isinstance(backend, RecordingBackend)


def _write_rocpd_from_regions(path, regions, dispatches):
    """Write a synthetic rocpd db from ``(label, start, end)`` regions + dispatches.

    ``dispatches`` is ``(start, end, name)``. Mirrors the fixture schema used by
    ``tests/test_alignment_rocpd_parse.py`` so the real readers resolve it.
    """
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE rocpd_string_{_GUID} (id INTEGER PRIMARY KEY, string TEXT)")
    conn.execute(
        f"CREATE TABLE rocpd_info_kernel_symbol_{_GUID} "
        "(id INTEGER PRIMARY KEY, kernel_name TEXT, display_name TEXT)"
    )
    conn.execute(
        f"CREATE TABLE rocpd_kernel_dispatch_{_GUID} "
        "(id INTEGER PRIMARY KEY, pid INTEGER, tid INTEGER, agent_id INTEGER, "
        "kernel_id INTEGER, dispatch_id INTEGER, queue_id INTEGER, stream_id INTEGER, "
        "start BIGINT, end BIGINT, region_name_id INTEGER)"
    )
    conn.execute(
        f"CREATE TABLE rocpd_region_{_GUID} "
        "(id INTEGER PRIMARY KEY, pid INTEGER, tid INTEGER, start BIGINT, end BIGINT, name_id INTEGER)"
    )
    strings: dict[str, int] = {}

    def intern(text):
        if text not in strings:
            strings[text] = len(strings) + 1
            conn.execute(
                f"INSERT INTO rocpd_string_{_GUID} (id, string) VALUES (?, ?)",
                (strings[text], text),
            )
        return strings[text]

    symbols: dict[str, int] = {}
    for _s, _e, name in dispatches:
        if name not in symbols:
            symbols[name] = len(symbols) + 1
            conn.execute(
                f"INSERT INTO rocpd_info_kernel_symbol_{_GUID} (id, kernel_name, display_name) "
                "VALUES (?, ?, ?)",
                (symbols[name], f"_mangled_{name}", name),
            )
    for i, (start, end, name) in enumerate(dispatches, start=1):
        conn.execute(
            f"INSERT INTO rocpd_kernel_dispatch_{_GUID} "
            "(id, pid, tid, agent_id, kernel_id, dispatch_id, queue_id, stream_id, "
            "start, end, region_name_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (i, _PID, _TID, _AGENT, symbols[name], i, 0, 0, start, end, 0),
        )
    for i, (start, end, text) in enumerate(regions, start=1):
        conn.execute(
            f"INSERT INTO rocpd_region_{_GUID} (id, pid, tid, start, end, name_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (i, _PID, _TID, start, end, intern(text)),
        )
    conn.commit()
    conn.close()


def test_shim_regions_feed_the_real_rocpd_join(tmp_path):
    """End-to-end: shim-emitted ranges own dispatches via the real C1 join.

    The annotator records two forward ranges. We map its recorded call ordinals
    to a nanosecond timeline, drop one dispatch inside each range (plus one before
    the first, which must be dropped as warm-up), and run the actual
    ``build_ranges_from_rocpd`` ownership join — the same code the offline producer
    uses. The shim's labels must be recognized and the containment attribution must
    hold.
    """
    backend = RecordingBackend()
    annotator = IterationAnnotator(backend, enabled=True)
    for _ in range(2):
        with annotator.forward(record={"total_num_scheduled_tokens": 2}):
            pass

    # Space recorded ranges onto a timeline with a gap between iterations.
    regions = []
    for ordinal, (label, start_order, end_order) in enumerate(backend.regions()):
        base = 1_000 + ordinal * 10_000
        regions.append((base, base + 5_000, label))

    db = tmp_path / "shim.db"
    _write_rocpd_from_regions(
        db,
        regions=regions,
        dispatches=[
            (500, 900, "warmup_fill_kernel"),  # before iter 0 -> dropped
            (1_500, 2_000, "hipblaslt_gemm_f16"),  # inside iter 0
            (11_500, 12_000, "flash_fwd_attn_kernel"),  # inside iter 1
        ],
    )

    ranges = build_ranges_from_rocpd(str(db))
    by_iter = {item.iteration: item for item in ranges}
    assert set(by_iter) == {0, 1}
    assert by_iter[0].phase == "forward" and by_iter[1].phase == "forward"
    assert by_iter[0].kernel_count == 1
    assert by_iter[1].kernel_count == 1
    assert {e.name for item in ranges for e in item.kernel_events} == {
        "hipblaslt_gemm_f16",
        "flash_fwd_attn_kernel",
    }
